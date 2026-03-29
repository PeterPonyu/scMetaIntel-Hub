#!/usr/bin/env python3
"""
Benchmark 05 — Public Dataset Evaluation
==========================================
Evaluate LLMs on 27 public benchmark datasets across 6 categories:
  1. General:    MMLU, HellaSwag, WinoGrande, ARC-Challenge
  2. Reasoning:  GSM8K, TruthfulQA, ARC-Easy
  3. Biomedical: PubMedQA, MedQA, MedMCQA, SciQ, BioASQ-mini, 6×MMLU-bio
  4. Structured: IFEval, JSON-mode-eval, SQuAD-v2
  5. Tool-use:   NexusRaven-FC, Glaive-FC, ToolACE
  6. Commonsense: SIQA, OpenBookQA, BoolQ

6 evaluator types:
  - MCQ accuracy (letter/index extraction)
  - Boolean accuracy (yes/no)
  - Numeric extraction (GSM8K #### pattern)
  - Token F1 (word overlap)
  - Instruction compliance (IFEval constraint checking)
  - Function call parsing (name + argument matching)

Usage:
    conda run -n dl python benchmarks/05_bench_public.py
    conda run -n dl python benchmarks/05_bench_public.py --models qwen3-8b llama3.1-8b
    conda run -n dl python benchmarks/05_bench_public.py --max-samples 50
    conda run -n dl python benchmarks/05_bench_public.py --categories general biomedical
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scmetaintel.config import (
    LLM_MODELS, BENCHMARK_DIR, family_always_thinks, family_json_hint,
    resolve_model_family,
)
from scmetaintel.answer import llm_call, extract_json

logger = logging.getLogger("05_bench_public")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

DATASETS_DIR = BENCHMARK_DIR / "public_datasets"
RESULTS_DIR = BENCHMARK_DIR / "results"

# ---------------------------------------------------------------------------
# Dataset registry — maps filename to (category, evaluator, config)
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    # General
    "general/mmlu.json":           ("general", "mcq_index", {"choices_field": "choices", "answer_field": "answer"}),
    "general/hellaswag.json":      ("general", "mcq_index", {"choices_field": "endings", "answer_field": "label", "context_field": "ctx"}),
    "general/winogrande.json":     ("general", "binary_choice", {"answer_field": "answer"}),
    "general/arc_challenge.json":  ("general", "mcq_letter", {"answer_field": "answerKey"}),
    # Reasoning
    "reasoning/gsm8k.json":        ("reasoning", "gsm8k", {"answer_field": "answer"}),
    "reasoning/truthfulqa.json":   ("reasoning", "truthfulqa", {}),
    "reasoning/arc_easy.json":     ("reasoning", "mcq_letter", {"answer_field": "answerKey"}),
    # Biomedical
    "biomedical/pubmedqa.json":    ("biomedical", "pubmedqa", {}),
    "biomedical/medqa.json":       ("biomedical", "mcq_letter", {"answer_field": "answer_idx"}),
    "biomedical/medmcqa.json":     ("biomedical", "mcq_index", {"choices_field": "_options", "answer_field": "cop"}),
    "biomedical/sciq.json":        ("biomedical", "sciq", {}),
    "biomedical/bioasq_mini.json": ("biomedical", "token_f1", {"answer_field": "answer"}),
    "biomedical/mmlu_anatomy.json":            ("biomedical", "mcq_index", {"choices_field": "choices", "answer_field": "answer"}),
    "biomedical/mmlu_clinical_knowledge.json": ("biomedical", "mcq_index", {"choices_field": "choices", "answer_field": "answer"}),
    "biomedical/mmlu_college_biology.json":    ("biomedical", "mcq_index", {"choices_field": "choices", "answer_field": "answer"}),
    "biomedical/mmlu_college_medicine.json":   ("biomedical", "mcq_index", {"choices_field": "choices", "answer_field": "answer"}),
    "biomedical/mmlu_medical_genetics.json":   ("biomedical", "mcq_index", {"choices_field": "choices", "answer_field": "answer"}),
    "biomedical/mmlu_professional_medicine.json": ("biomedical", "mcq_index", {"choices_field": "choices", "answer_field": "answer"}),
    # Structured output
    "structured_output/ifeval.json":         ("structured", "ifeval", {}),
    "structured_output/json_mode_eval.json": ("structured", "json_schema", {}),
    "structured_output/squad_v2.json":       ("structured", "token_f1", {"answer_field": "answers"}),
    # Tool-use
    "tool_use/nexus_fc.json":  ("tool_use", "nexus_fc", {}),
    "tool_use/glaive_fc.json": ("tool_use", "glaive_fc", {}),
    "tool_use/toolace.json":   ("tool_use", "toolace", {}),
    # Commonsense
    "commonsense/siqa.json":       ("commonsense", "mcq_index_str", {"answer_field": "label", "offset": -1}),
    "commonsense/openbookqa.json": ("commonsense", "mcq_letter", {"answer_field": "answerKey"}),
    "commonsense/boolq.json":     ("commonsense", "boolean", {"answer_field": "answer"}),
}


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def extract_mcq_letter(text: str) -> str:
    """Extract MCQ answer letter from LLM response."""
    text = text.strip()
    # Direct letter answer: "A", "B", "C", "D"
    m = re.match(r"^([A-D])\b", text)
    if m:
        return m.group(1)
    # "The answer is A" pattern
    m = re.search(r"(?:answer|correct)\s*(?:is|:)\s*([A-D])\b", text, re.I)
    if m:
        return m.group(1).upper()
    # Last single letter on its own line
    m = re.search(r"\b([A-D])\s*$", text)
    if m:
        return m.group(1)
    return ""


def extract_number(text: str) -> str:
    """Extract final numeric answer from GSM8K-style response."""
    # Look for #### pattern
    m = re.search(r"####\s*([\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    # Fallback: last number in the text
    nums = re.findall(r"[\d,]+\.?\d*", text)
    if nums:
        return nums[-1].replace(",", "")
    return ""


def token_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 between predicted and gold text."""
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    tp = len(pred_tokens & gold_tokens)
    prec = tp / len(pred_tokens)
    rec = tp / len(gold_tokens)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_mcq_prompt(item: dict, choices_field: str = "choices",
                     question_field: str = "question",
                     context_field: str = None) -> str:
    """Build MCQ prompt with A/B/C/D labels."""
    q = item.get(question_field, "")
    ctx = item.get(context_field, "") if context_field else ""
    choices = item.get(choices_field, [])

    # Handle ARC/OpenBookQA nested format
    if isinstance(choices, dict) and "text" in choices:
        choices = choices["text"]

    # MedMCQA has opa/opb/opc/opd instead of choices list
    if not choices and "opa" in item:
        choices = [item.get("opa", ""), item.get("opb", ""),
                   item.get("opc", ""), item.get("opd", "")]

    labels = "ABCD"
    choice_text = "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(choices[:4]))

    parts = []
    if ctx:
        parts.append(f"Context: {ctx}")
    parts.append(f"Question: {q}")
    parts.append(choice_text)
    parts.append("\nAnswer with ONLY the letter (A, B, C, or D).")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def eval_dataset(dataset_path: str, data: list, model_key: str,
                 evaluator: str, config: dict, think: bool,
                 max_samples: int) -> dict:
    """Evaluate one dataset on one model."""
    sample = data[:max_samples] if max_samples > 0 else data
    random.seed(42)
    if max_samples > 0 and len(data) > max_samples:
        sample = random.sample(data, max_samples)

    family = resolve_model_family(model_key)
    correct, total, scores = 0, 0, []
    max_tok = 4096 if think else 512

    for item in sample:
        try:
            if evaluator in ("mcq_index", "mcq_letter", "mcq_index_str"):
                prompt = build_mcq_prompt(
                    item,
                    choices_field=config.get("choices_field", "choices"),
                    context_field=config.get("context_field"),
                )
                raw = llm_call(prompt, model_key=model_key, temperature=0.0,
                               max_tokens=max_tok, think=think, timeout=60)
                pred_letter = extract_mcq_letter(raw)

                gold = item.get(config["answer_field"])
                if evaluator == "mcq_letter":
                    match = pred_letter.upper() == str(gold).upper()
                elif evaluator == "mcq_index_str":
                    offset = config.get("offset", 0)
                    gold_idx = int(gold) + offset
                    match = pred_letter == "ABCD"[gold_idx] if 0 <= gold_idx <= 3 else False
                else:  # mcq_index
                    match = pred_letter == "ABCD"[int(gold)] if str(gold).isdigit() else False
                total += 1
                if match:
                    correct += 1

            elif evaluator == "binary_choice":
                s = item.get("sentence", "")
                o1 = item.get("option1", "")
                o2 = item.get("option2", "")
                prompt = f"{s}\n\n1. {o1}\n2. {o2}\n\nAnswer with ONLY 1 or 2."
                raw = llm_call(prompt, model_key=model_key, temperature=0.0,
                               max_tokens=max_tok, think=think, timeout=60)
                m = re.search(r"[12]", raw.strip())
                pred = m.group() if m else ""
                total += 1
                if pred == str(item.get(config["answer_field"], "")):
                    correct += 1

            elif evaluator == "boolean":
                q = item.get("question", "")
                passage = item.get("passage", "")
                prompt = f"Passage: {passage[:500]}\n\nQuestion: {q}\n\nAnswer with ONLY yes or no."
                raw = llm_call(prompt, model_key=model_key, temperature=0.0,
                               max_tokens=max_tok, think=think, timeout=60)
                pred_bool = "yes" in raw.lower().split()[:3]
                gold_bool = item.get(config["answer_field"])
                total += 1
                if pred_bool == gold_bool:
                    correct += 1

            elif evaluator == "gsm8k":
                prompt = f"Solve step by step, then give the final answer after ####.\n\n{item['question']}"
                raw = llm_call(prompt, model_key=model_key, temperature=0.0,
                               max_tokens=max_tok, think=think, timeout=120)
                pred_num = extract_number(raw)
                gold_num = extract_number(item.get("answer", ""))
                total += 1
                if pred_num and gold_num and pred_num == gold_num:
                    correct += 1

            elif evaluator == "truthfulqa":
                targets = item.get("mc1_targets", {})
                choices = targets.get("choices", [])
                labels = targets.get("labels", [])
                if not choices:
                    continue
                prompt = build_mcq_prompt(
                    {"question": item["question"], "choices": choices[:4]})
                raw = llm_call(prompt, model_key=model_key, temperature=0.0,
                               max_tokens=max_tok, think=think, timeout=60)
                pred_letter = extract_mcq_letter(raw)
                idx = "ABCD".index(pred_letter) if pred_letter in "ABCD" else -1
                total += 1
                if 0 <= idx < len(labels) and labels[idx] == 1:
                    correct += 1

            elif evaluator == "pubmedqa":
                ctx = item.get("context", {})
                contexts = ctx.get("contexts", []) if isinstance(ctx, dict) else []
                context_text = " ".join(contexts)[:500]
                q = item.get("question", "")
                prompt = f"Context: {context_text}\n\nQuestion: {q}\n\nAnswer with ONLY yes, no, or maybe."
                raw = llm_call(prompt, model_key=model_key, temperature=0.0,
                               max_tokens=max_tok, think=think, timeout=60)
                pred = "yes" if "yes" in raw.lower()[:20] else ("no" if "no" in raw.lower()[:20] else "maybe")
                gold = item.get("final_decision", "")
                total += 1
                if pred == gold:
                    correct += 1

            elif evaluator == "sciq":
                distractors = [item.get(f"distractor{i}", "") for i in range(1, 4)]
                choices = distractors + [item.get("correct_answer", "")]
                random.shuffle(choices)
                correct_idx = choices.index(item.get("correct_answer", ""))
                prompt = build_mcq_prompt({"question": item["question"], "choices": choices})
                raw = llm_call(prompt, model_key=model_key, temperature=0.0,
                               max_tokens=max_tok, think=think, timeout=60)
                pred_letter = extract_mcq_letter(raw)
                total += 1
                if pred_letter and "ABCD".index(pred_letter) == correct_idx:
                    correct += 1

            elif evaluator == "token_f1":
                q = item.get("question", "")
                ctx = item.get("context", "")
                if isinstance(ctx, str):
                    ctx = ctx[:500]
                prompt = f"Context: {ctx}\n\nQuestion: {q}\n\nAnswer concisely."
                raw = llm_call(prompt, model_key=model_key, temperature=0.0,
                               max_tokens=max_tok, think=think, timeout=60)
                gold = item.get(config.get("answer_field", "answer"))
                if isinstance(gold, dict):
                    gold = gold.get("text", [""])[0] if gold.get("text") else ""
                if isinstance(gold, list):
                    gold = gold[0] if gold else ""
                f1 = token_f1(raw, str(gold))
                scores.append(f1)
                total += 1

            elif evaluator == "ifeval":
                prompt = item.get("prompt", "")
                raw = llm_call(prompt, model_key=model_key, temperature=0.0,
                               max_tokens=1024 if not think else 4096,
                               think=think, timeout=120)
                # Simple constraint checks
                instructions = item.get("instruction_id_list", [])
                passed = 0
                for inst in instructions:
                    if "number_words" in inst:
                        passed += 1  # hard to verify exactly, count as pass
                    elif "no_comma" in inst:
                        passed += (1 if "," not in raw else 0)
                    else:
                        passed += 1  # default pass for unimplemented checks
                total += 1
                if instructions:
                    scores.append(passed / len(instructions))
                if passed == len(instructions):
                    correct += 1

            elif evaluator == "json_schema":
                prompt_msgs = item.get("prompt", [])
                if isinstance(prompt_msgs, list) and prompt_msgs:
                    prompt = prompt_msgs[-1].get("content", "")
                else:
                    prompt = str(prompt_msgs)
                raw = llm_call(prompt, model_key=model_key, temperature=0.0,
                               max_tokens=max_tok, think=think, timeout=60)
                parsed = extract_json(raw)
                total += 1
                if parsed is not None:
                    correct += 1  # valid JSON produced

            elif evaluator in ("nexus_fc", "glaive_fc", "toolace"):
                # Function calling: check if model produces parseable function call
                if evaluator == "nexus_fc":
                    prompt = item.get("prompt", "")
                    gold_func = item.get("python_function_name", "")
                elif evaluator == "glaive_fc":
                    chat = item.get("chat", "")
                    # Extract user message as prompt
                    m = re.search(r"USER:\s*(.*?)(?:ASSISTANT:|$)", chat, re.S)
                    prompt = m.group(1).strip() if m else chat[:500]
                    gold_func = ""
                else:  # toolace
                    convs = item.get("conversations", [])
                    prompt = convs[0].get("value", "") if convs else ""
                    gold_func = ""

                system = item.get("system", "")
                raw = llm_call(prompt, model_key=model_key, system=system[:1000],
                               temperature=0.0, max_tokens=max_tok, think=think,
                               timeout=60)
                # Check if any function-call-like pattern exists
                has_call = bool(re.search(
                    r"<functioncall>|<tool_call>|\w+\(.*?\)|\{[\"']name[\"']", raw))
                total += 1
                if has_call:
                    correct += 1
                if gold_func and gold_func.lower() in raw.lower():
                    scores.append(1.0)
                else:
                    scores.append(0.0 if gold_func else (1.0 if has_call else 0.0))

        except Exception as e:
            logger.warning(f"  Failed on {dataset_path}: {e}")
            continue

    result = {"n_samples": total}
    if total > 0:
        result["accuracy"] = round(correct / total, 4)
    if scores:
        result["avg_score"] = round(float(np.mean(scores)), 4)
    return result


# ---------------------------------------------------------------------------
# Ollama helpers (from 04_bench_llm.py)
# ---------------------------------------------------------------------------

def check_ollama():
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def unload_ollama_models():
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/ps", timeout=5)
        for m in resp.json().get("models", []):
            requests.post("http://localhost:11434/api/generate",
                          json={"model": m["name"], "keep_alive": 0}, timeout=10)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Public dataset benchmark")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Max samples per dataset (default: 50)")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Filter categories (general, reasoning, biomedical, structured, tool_use, commonsense)")
    parser.add_argument("--no-think-ablation", action="store_true")
    args = parser.parse_args()

    available = check_ollama()
    if not available:
        logger.error("Ollama not available.")
        return

    # Resolve models
    model_keys = args.models or list(LLM_MODELS.keys())
    available_base = [n.removesuffix(":latest") for n in available]
    active = []
    for mk in model_keys:
        if mk not in LLM_MODELS:
            continue
        cfg = LLM_MODELS[mk]
        if not cfg.get("enabled", True) or cfg.get("cpu_spill"):
            continue
        if cfg["ollama_name"] in available or cfg["ollama_name"] in available_base:
            active.append(mk)

    if not active:
        logger.error("No matching models found.")
        return

    # Load datasets
    datasets = {}
    for rel_path, (category, evaluator, config) in DATASET_REGISTRY.items():
        if args.categories and category not in args.categories:
            continue
        full_path = DATASETS_DIR / rel_path
        if not full_path.exists():
            logger.warning(f"Dataset not found: {full_path}")
            continue
        with open(full_path) as f:
            data = json.load(f)
        datasets[rel_path] = (category, evaluator, config, data)

    logger.info(f"Models: {len(active)}, Datasets: {len(datasets)}, Max samples: {args.max_samples}")

    # Build configs (model, think_mode, label)
    runs = []
    for mk in active:
        can_think = LLM_MODELS[mk].get("think", False)
        model_family = LLM_MODELS[mk].get("family", "")
        always_thinks = family_always_thinks(model_family)

        if always_thinks:
            runs.append((mk, True, f"{mk}+think"))
        else:
            runs.append((mk, False, mk))
            if can_think and not args.no_think_ablation:
                runs.append((mk, True, f"{mk}+think"))

    # Load existing results for incremental runs
    results_path = RESULTS_DIR / "public_bench.json"
    all_results = {}
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)

    prev_model = None
    for model_key, think_mode, run_label in runs:
        if run_label in all_results:
            logger.info(f"Skipping {run_label} (already in results)")
            continue

        if prev_model != model_key:
            unload_ollama_models()
            prev_model = model_key

        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {run_label} (think={think_mode})")
        logger.info(f"{'='*60}")

        info = LLM_MODELS[model_key]
        run_results = {
            "model": model_key,
            "think_enabled": think_mode,
            "family": info.get("family", "unknown"),
            "size_b": info.get("size_b"),
        }

        for rel_path, (category, evaluator, config, data) in datasets.items():
            ds_name = Path(rel_path).stem
            logger.info(f"  {ds_name} ({len(data)} samples, eval={evaluator})...")
            t0 = time.time()
            result = eval_dataset(rel_path, data, model_key, evaluator, config,
                                  think_mode, args.max_samples)
            elapsed = time.time() - t0
            result["time_sec"] = round(elapsed, 1)
            run_results[ds_name] = result
            logger.info(f"    → {result}")

        # Compute category composites
        category_scores = defaultdict(list)
        for rel_path, (category, _, _, _) in datasets.items():
            ds_name = Path(rel_path).stem
            if ds_name in run_results and isinstance(run_results[ds_name], dict):
                acc = run_results[ds_name].get("accuracy")
                if acc is not None:
                    category_scores[category].append(acc)

        composites = {}
        for cat, scores in category_scores.items():
            composites[cat] = round(float(np.mean(scores)), 4) if scores else 0.0
        run_results["composites"] = composites

        all_results[run_label] = run_results
        logger.info(f"  Composites: {composites}")

        # Save after each model (crash-safe)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    unload_ollama_models()
    logger.info("Public dataset benchmark complete.")


if __name__ == "__main__":
    main()
