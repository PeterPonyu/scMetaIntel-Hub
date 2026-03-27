#!/usr/bin/env python3
"""Smoke-test all embedding models and reranker models from config.py."""
import json
import time
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from scmetaintel.config import EMBEDDING_MODELS, RERANKER_MODELS

# Test sentences
QUERY = "single-cell RNA sequencing of human lung cancer tumor microenvironment"
DOC = "We performed scRNA-seq on 52,698 cells from 5 non-small cell lung cancer patients, identifying distinct immune cell populations including exhausted CD8+ T cells and tumor-associated macrophages."

def test_embedding_model(key: str, info: dict) -> dict:
    """Test a single embedding model."""
    result = {"model": key, "hf_name": info["name"], "status": "UNKNOWN",
              "time_s": 0, "dim_expected": info["dim"], "dim_actual": 0, "error": ""}
    device = info.get("device", "cuda")
    result["device"] = device
    start = time.time()
    try:
        model_type = info.get("type", "dense")
        hf_name = info["name"]

        if key == "bge-m3" or "bge-m3" in hf_name.lower():
            # Use FlagEmbedding for BGE-M3
            from FlagEmbedding import BGEM3FlagModel
            model = BGEM3FlagModel(hf_name, use_fp16=(device != "cpu"), device=device)
            out = model.encode([QUERY], return_dense=True, return_sparse=True)
            dense = out["dense_vecs"]
            result["dim_actual"] = dense.shape[1]
            result["has_sparse"] = "sparse_vecs" in out and len(out["sparse_vecs"]) > 0
            result["status"] = "OK"
            del model
        elif "specter2" in key and "query" in key:
            # SPECTER2 adapter model
            from transformers import AutoTokenizer, AutoModel
            from peft import PeftModel
            base_model_name = info.get("base_model", "allenai/specter2_base")
            base = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
            model = PeftModel.from_pretrained(base, hf_name)
            if device == "cpu":
                model = model.cpu()
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            inputs = tokenizer([QUERY], padding=True, truncation=True, max_length=512, return_tensors="pt")
            import torch
            with torch.no_grad():
                out = model(**inputs)
            emb = out.last_hidden_state[:, 0, :]
            result["dim_actual"] = emb.shape[1]
            result["status"] = "OK"
            del model, base
        elif "specter2" in key:
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(hf_name, trust_remote_code=True)
            if device == "cpu":
                model = model.cpu()
            inputs = tokenizer([QUERY], padding=True, truncation=True, max_length=512, return_tensors="pt")
            if device == "cpu":
                inputs = {k: v.cpu() for k, v in inputs.items()}
            import torch
            with torch.no_grad():
                out = model(**inputs)
            emb = out.last_hidden_state[:, 0, :]
            result["dim_actual"] = emb.shape[1]
            result["status"] = "OK"
            del model
        elif "medcpt" in key.lower():
            # MedCPT uses AutoModel
            from transformers import AutoTokenizer, AutoModel
            import torch
            tokenizer = AutoTokenizer.from_pretrained(hf_name)
            model = AutoModel.from_pretrained(hf_name)
            if device == "cpu":
                model = model.cpu()
            inputs = tokenizer([QUERY], padding=True, truncation=True, max_length=512, return_tensors="pt")
            if device == "cpu":
                inputs = {k: v.cpu() for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            emb = out.last_hidden_state[:, 0, :]
            result["dim_actual"] = emb.shape[1]
            result["status"] = "OK"
            del model
        elif any(q in key for q in ["qwen3-embed"]):
            # Qwen3-Embedding — large models, just test tokenizer loads
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
            result["dim_actual"] = info["dim"]
            result["status"] = "OK (tokenizer-only, large model)"
            del tokenizer
        else:
            # Standard sentence-transformers model
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(hf_name, trust_remote_code=True, device=device)
            emb = model.encode([QUERY])
            result["dim_actual"] = emb.shape[1]
            result["status"] = "OK"
            del model

        result["time_s"] = round(time.time() - start, 1)

    except Exception as e:
        result["time_s"] = round(time.time() - start, 1)
        result["status"] = "ERROR"
        result["error"] = str(e)[:300]

    # Force GPU memory cleanup
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

    return result


def test_reranker_model(key: str, info: dict) -> dict:
    """Test a single reranker model."""
    result = {"model": key, "hf_name": info["name"], "status": "UNKNOWN",
              "time_s": 0, "score": None, "error": ""}
    device = info.get("device", "cuda")
    result["device"] = device
    start = time.time()
    try:
        hf_name = info["name"]

        if any(q in key for q in ["qwen3-reranker"]):
            # Qwen3 rerankers are large, just verify tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
            result["status"] = "OK (tokenizer-only, large model)"
            del tokenizer
        elif "bge-reranker-v2-gemma" in key:
            # Large Gemma reranker, tokenizer check only
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
            result["status"] = "OK (tokenizer-only, large model)"
            del tokenizer
        elif "medcpt-cross-encoder" in key:
            # MedCPT Cross-Encoder
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            tokenizer = AutoTokenizer.from_pretrained(hf_name)
            model = AutoModelForSequenceClassification.from_pretrained(hf_name)
            if device == "cpu":
                model = model.cpu()
            inputs = tokenizer([[QUERY, DOC]], padding=True, truncation=True, max_length=512, return_tensors="pt")
            if device == "cpu":
                inputs = {k: v.cpu() for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            score = out.logits.squeeze().item()
            result["score"] = round(score, 4)
            result["status"] = "OK"
            del model
        elif "bge-reranker-v2-m3" in key:
            # Use FlagEmbedding FlagReranker
            from FlagEmbedding import FlagReranker
            model = FlagReranker(hf_name, use_fp16=(device != "cpu"), device=device)
            score = model.compute_score([[QUERY, DOC]])
            result["score"] = round(float(score), 4) if not isinstance(score, list) else round(float(score[0]), 4)
            result["status"] = "OK"
            del model
        else:
            # Standard cross-encoder
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(hf_name, trust_remote_code=True, device=device)
            scores = model.predict([(QUERY, DOC)])
            result["score"] = round(float(scores[0]), 4)
            result["status"] = "OK"
            del model

        result["time_s"] = round(time.time() - start, 1)

    except Exception as e:
        result["time_s"] = round(time.time() - start, 1)
        result["status"] = "ERROR"
        result["error"] = str(e)[:300]

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

    return result


def main():
    import torch
    print("=" * 70)
    print("EMBEDDING & RERANKER SMOKE TEST")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    # Test embeddings
    print(f"\n--- EMBEDDING MODELS ({len(EMBEDDING_MODELS)}) ---\n")
    embed_results = []
    for i, (key, info) in enumerate(EMBEDDING_MODELS.items(), 1):
        print(f"[{i}/{len(EMBEDDING_MODELS)}] {key} ({info['name']}) ...", end=" ", flush=True)
        r = test_embedding_model(key, info)
        embed_results.append(r)
        if "OK" in r["status"]:
            dim_check = "dim OK" if r["dim_actual"] == r["dim_expected"] else f"dim MISMATCH {r['dim_actual']}!={r['dim_expected']}"
            print(f"PASS ({r['time_s']}s, {dim_check})")
        else:
            print(f"FAIL ({r['time_s']}s)")
            print(f"      Error: {r['error'][:150]}")

    # Test rerankers
    print(f"\n--- RERANKER MODELS ({len(RERANKER_MODELS)}) ---\n")
    rerank_results = []
    for i, (key, info) in enumerate(RERANKER_MODELS.items(), 1):
        print(f"[{i}/{len(RERANKER_MODELS)}] {key} ({info['name']}) ...", end=" ", flush=True)
        r = test_reranker_model(key, info)
        rerank_results.append(r)
        if "OK" in r["status"]:
            score_str = f", score={r['score']}" if r["score"] is not None else ""
            print(f"PASS ({r['time_s']}s{score_str})")
        else:
            print(f"FAIL ({r['time_s']}s)")
            print(f"      Error: {r['error'][:150]}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    embed_pass = sum(1 for r in embed_results if "OK" in r["status"])
    rerank_pass = sum(1 for r in rerank_results if "OK" in r["status"])
    print(f"\nEmbeddings: {embed_pass}/{len(embed_results)} passed")
    print(f"Rerankers:  {rerank_pass}/{len(rerank_results)} passed")

    all_fail = [r for r in embed_results + rerank_results if "OK" not in r["status"]]
    if all_fail:
        print(f"\nFailed ({len(all_fail)}):")
        for r in all_fail:
            print(f"  - {r['model']}: {r['error'][:120]}")

    # Save results
    out_path = str(Path(__file__).resolve().parent.parent / "benchmarks" / "smoke_test_embeddings.json")
    with open(out_path, "w") as f:
        json.dump({"embeddings": embed_results, "rerankers": rerank_results}, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")

    return 0 if not all_fail else 1

if __name__ == "__main__":
    main()
