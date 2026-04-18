#!/usr/bin/env python3
"""
Model download planner for scMetaIntel-Hub LLM benchmarking.

Reads the full LLM_MODELS registry from config.py, checks which models
are already pulled in Ollama, and generates a prioritised download plan
grouped by wave and family.

Usage:
    # Dry-run (default): show plan, no downloads
    python scripts/model_download_plan.py

    # Show only missing models
    python scripts/model_download_plan.py --missing-only

    # Execute downloads (wave by wave, family by family)
    python scripts/model_download_plan.py --execute

    # Execute only a specific wave
    python scripts/model_download_plan.py --execute --wave 1

    # Execute only a specific family
    python scripts/model_download_plan.py --execute --family deepseek
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scmetaintel.config import LLM_MODELS, MODEL_FAMILY_CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_pulled_models() -> set[str]:
    """Return set of Ollama model names currently pulled (strip :latest)."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            print("WARNING: 'ollama list' failed. Is Ollama running?")
            return set()
    except FileNotFoundError:
        print("WARNING: 'ollama' command not found.")
        return set()

    pulled = set()
    for line in result.stdout.strip().split("\n")[1:]:  # skip header
        if not line.strip():
            continue
        name = line.split()[0]
        pulled.add(name)
        # Also add without :latest suffix for flexible matching
        if name.endswith(":latest"):
            pulled.add(name.removesuffix(":latest"))
    return pulled


def estimate_download_gb(info: dict) -> float:
    """Estimate download size from VRAM estimate (download ≈ VRAM at 0 ctx)."""
    return info.get("vram_gb", 5.0)


def format_size(gb: float) -> str:
    if gb < 1:
        return f"{gb * 1024:.0f} MB"
    return f"{gb:.1f} GB"


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------

def build_plan(pulled: set[str]) -> list[dict]:
    """Build ordered download plan from LLM_MODELS registry."""
    plan = []
    for key, info in LLM_MODELS.items():
        ollama_name = info["ollama_name"]
        is_pulled = ollama_name in pulled
        is_enabled = info.get("enabled", True)
        is_cpu_spill = info.get("cpu_spill", False)

        plan.append({
            "key": key,
            "ollama_name": ollama_name,
            "family": info.get("family", "unknown"),
            "size_b": info.get("size_b", 0),
            "quant": info.get("quant", "?"),
            "vram_gb": info.get("vram_gb", 0),
            "ctx": info.get("ctx", 0),
            "wave": info.get("wave", 0),  # 0 = existing/already pulled
            "note": info.get("note", ""),
            "pulled": is_pulled,
            "enabled": is_enabled,
            "cpu_spill": is_cpu_spill,
            "download_gb": estimate_download_gb(info),
        })

    # Sort: wave first, then family, then size descending (pull large ones first)
    plan.sort(key=lambda x: (x["wave"], x["family"], -x["size_b"]))
    return plan


def print_plan(plan: list[dict], missing_only: bool = False,
               wave_filter: int | None = None,
               family_filter: str | None = None):
    """Pretty-print the download plan."""
    filtered = plan
    if missing_only:
        filtered = [p for p in filtered if not p["pulled"]]
    if wave_filter is not None:
        filtered = [p for p in filtered if p["wave"] == wave_filter]
    if family_filter:
        filtered = [p for p in filtered if p["family"] == family_filter]

    # Group by wave
    by_wave: dict[int, list[dict]] = defaultdict(list)
    for entry in filtered:
        by_wave[entry["wave"]].append(entry)

    wave_names = {
        0: "EXISTING (already in registry, most already pulled)",
        1: "WAVE 1 — Fill scaling gaps in existing families",
        2: "WAVE 2 — Add size variants for Phi, Mistral, Gemma2",
        4: "WAVE 4 — IBM Granite (structured output specialists)",
        5: "WAVE 5 — Cross-architecture diversity (Falcon, Aya, GLM)",
    }

    total_models = len(filtered)
    total_pulled = sum(1 for p in filtered if p["pulled"])
    total_missing = total_models - total_pulled
    total_download = sum(p["download_gb"] for p in filtered if not p["pulled"])

    total_configs = sum(1 for p in filtered if p["enabled"] and not p["cpu_spill"])

    print("=" * 78)
    print("  scMetaIntel-Hub — LLM Model Download Plan")
    print("  Hardware: RTX 5090 Laptop (24.5 GB VRAM)")
    print("  Constraint: ALL models must fit 100% in GPU at num_ctx=4096")
    print("=" * 78)
    print()
    print(f"  Total models in plan:    {total_models}")
    print(f"  Already pulled:          {total_pulled}")
    print(f"  Need to download:        {total_missing}")
    print(f"  Estimated download:      {format_size(total_download)}")
    print(f"  Benchmark configs:       ~{total_configs} (single public config per model)")
    print()

    # Family summary
    families = sorted(set(p["family"] for p in filtered))
    print("  Families: " + ", ".join(families))
    print()

    for wave_id in sorted(by_wave.keys()):
        entries = by_wave[wave_id]
        wave_label = wave_names.get(wave_id, f"WAVE {wave_id}")
        wave_download = sum(p["download_gb"] for p in entries if not p["pulled"])

        print(f"{'─' * 78}")
        print(f"  {wave_label}")
        if wave_download > 0:
            print(f"  Download: ~{format_size(wave_download)}")
        print(f"{'─' * 78}")

        # Group by family within wave
        by_family: dict[str, list[dict]] = defaultdict(list)
        for e in entries:
            by_family[e["family"]].append(e)

        for family in sorted(by_family.keys()):
            fam_entries = by_family[family]
            fam_config = MODEL_FAMILY_CONFIG.get(family, {})
            multimodal = fam_config.get("multimodal", False)

            fam_tags = []
            if multimodal:
                overhead = fam_config.get("multimodal_overhead_pct", 0)
                fam_tags.append(f"multimodal(+{overhead}% VRAM overhead)")
            fam_tag_str = f"  [{', '.join(fam_tags)}]" if fam_tags else ""

            print(f"\n  [{family.upper()}]{fam_tag_str}")
            if fam_config.get("note"):
                print(f"  {fam_config['note']}")
            print()

            for e in sorted(fam_entries, key=lambda x: -x["size_b"]):
                status = "✓ PULLED" if e["pulled"] else "○ NEEDED"
                if not e["enabled"]:
                    status = "✗ DISABLED"
                elif e["cpu_spill"]:
                    status = "✗ CPU_SPILL"

                configs = "1 config" if e["enabled"] else "0 config"

                print(f"    {status:14s} {e['key']:25s} "
                      f"{e['size_b']:5.1f}B  {e['quant']:8s}  "
                      f"~{format_size(e['vram_gb']):>7s}  "
                      f"{'':6s}  ({configs})")
                print(f"{'':19s} ollama pull {e['ollama_name']}")
                print()

    # Print download commands summary
    missing = [p for p in filtered
               if not p["pulled"] and p["enabled"] and not p["cpu_spill"]]
    if missing:
        print(f"\n{'=' * 78}")
        print("  DOWNLOAD COMMANDS (copy-paste ready)")
        print(f"{'=' * 78}")
        for wave_id in sorted(set(p["wave"] for p in missing)):
            wave_entries = [p for p in missing if p["wave"] == wave_id]
            wave_dl = sum(p["download_gb"] for p in wave_entries)
            print(f"\n  # --- {wave_names.get(wave_id, f'WAVE {wave_id}')} "
                  f"(~{format_size(wave_dl)}) ---")
            for e in wave_entries:
                print(f"  ollama pull {e['ollama_name']}")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_downloads(plan: list[dict],
                      wave_filter: int | None = None,
                      family_filter: str | None = None):
    """Actually download models via ollama pull."""
    targets = [
        p for p in plan
        if not p["pulled"] and p["enabled"] and not p["cpu_spill"]
    ]
    if wave_filter is not None:
        targets = [p for p in targets if p["wave"] == wave_filter]
    if family_filter:
        targets = [p for p in targets if p["family"] == family_filter]

    if not targets:
        print("Nothing to download — all matching models already pulled.")
        return

    total = len(targets)
    total_gb = sum(p["download_gb"] for p in targets)
    print(f"\nWill download {total} models (~{format_size(total_gb)} total)")
    print("Press Ctrl+C to abort at any time.\n")

    for i, entry in enumerate(targets, 1):
        print(f"[{i}/{total}] Pulling {entry['key']} "
              f"({entry['ollama_name']}, ~{format_size(entry['download_gb'])})")
        try:
            result = subprocess.run(
                ["ollama", "pull", entry["ollama_name"]],
                timeout=3600,  # 1 hour per model max
            )
            if result.returncode != 0:
                print(f"  ERROR: Failed to pull {entry['ollama_name']}")
            else:
                print(f"  OK: {entry['key']} pulled successfully")
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: {entry['ollama_name']} took >1h, skipping")
        except KeyboardInterrupt:
            print("\n\nAborted by user.")
            sys.exit(1)
        print()

    print(f"\nDone. Downloaded {total} models.")


# ---------------------------------------------------------------------------
# VRAM analysis
# ---------------------------------------------------------------------------

def print_vram_analysis(plan: list[dict]):
    """Print VRAM fit analysis for all models at num_ctx=4096."""
    VRAM_BUDGET = 24.5
    print(f"\n{'=' * 78}")
    print(f"  VRAM FIT ANALYSIS (budget: {VRAM_BUDGET} GB, num_ctx=4096)")
    print(f"{'=' * 78}\n")

    enabled = [p for p in plan if p["enabled"] and not p["cpu_spill"]]
    enabled.sort(key=lambda x: x["vram_gb"], reverse=True)

    for e in enabled:
        bar_len = int(e["vram_gb"] / VRAM_BUDGET * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        pct = e["vram_gb"] / VRAM_BUDGET * 100
        fit = "OK" if pct <= 100 else "OVER!"
        print(f"  {e['key']:25s} {e['vram_gb']:5.1f}GB [{bar}] {pct:5.1f}% {fit}")

    # Multimodal overhead warning
    mm_models = [e for e in enabled if
                 MODEL_FAMILY_CONFIG.get(e["family"], {}).get("multimodal")]
    if mm_models:
        print(f"\n  ⚠ MULTIMODAL OVERHEAD: {len(mm_models)} model(s) include "
              "unused vision projector:")
        for e in mm_models:
            overhead = MODEL_FAMILY_CONFIG.get(e["family"], {}).get(
                "multimodal_overhead_pct", 0)
            print(f"    {e['key']:25s} — ~{overhead}% extra VRAM for vision "
                  "(text-only tasks don't use it)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plan and execute LLM model downloads for benchmarking")
    parser.add_argument("--execute", action="store_true",
                        help="Actually download models (default: dry-run)")
    parser.add_argument("--missing-only", action="store_true",
                        help="Show only models not yet pulled")
    parser.add_argument("--wave", type=int, default=None,
                        help="Filter to specific wave (0-5)")
    parser.add_argument("--family", type=str, default=None,
                        help="Filter to specific family (e.g. deepseek, granite)")
    parser.add_argument("--vram", action="store_true",
                        help="Show VRAM fit analysis")
    parser.add_argument("--json", action="store_true",
                        help="Output plan as JSON")
    args = parser.parse_args()

    pulled = get_pulled_models()
    plan = build_plan(pulled)

    if args.json:
        print(json.dumps(plan, indent=2))
        return

    print_plan(plan, missing_only=args.missing_only,
               wave_filter=args.wave, family_filter=args.family)

    if args.vram:
        print_vram_analysis(plan)

    if not args.execute and not args.vram:
        # Always show these in default dry-run
        print_vram_analysis(plan)

    if args.execute:
        execute_downloads(plan, wave_filter=args.wave,
                          family_filter=args.family)


if __name__ == "__main__":
    main()
