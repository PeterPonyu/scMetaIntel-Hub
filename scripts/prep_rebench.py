#!/usr/bin/env python3
"""
Prepare llm_bench.json for a full-scale re-bench.

This script:
1. Archives the current results
2. Clears broken think-mode entries (Gemma3/Granite scoring 0)
3. Clears ALL entries that used small sample sizes
   (so the re-bench runs them at full scale)
4. Optionally keeps results that are already at full scale

Usage:
    python scripts/prep_rebench.py                    # Clear all small-sample results
    python scripts/prep_rebench.py --clear-all        # Clear everything for fresh start
    python scripts/prep_rebench.py --clear-think-only # Only clear broken think entries
    python scripts/prep_rebench.py --dry-run          # Show what would be cleared
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "results"
BENCH_FILE = RESULTS_DIR / "llm_bench.json"

# Full-scale sample sizes (what we want for the re-bench)
FULL_SCALE = {
    "task_a_parsing": ("n_queries", 100),    # at least 100 queries
    "task_b_extraction": ("n_docs", 100),     # at least 100 docs
    "task_c_ontology": ("n_docs", 50),        # at least 50 docs
    "task_d_answer": ("n_queries", 50),       # at least 50 queries
    "task_f_relevance": ("n_pairs", 100),     # at least 100 pairs
    "task_g_domain": ("n_docs", 100),         # at least 100 docs
    "task_h_org_modality": ("n_docs", 100),   # at least 100 docs
}


def is_broken_think(entry: dict) -> bool:
    """Check if this is a broken think-mode entry (all zeros)."""
    if not entry.get("think_enabled"):
        return False
    # Check for error fields or all-zero scores
    for task_key in ["task_b_extraction", "task_c_ontology", "task_d_answer"]:
        task = entry.get(task_key, {})
        if "error" in task:
            return True
    # Check if task_a exact_match is 0 AND task_f f1 is 0 (strong signal of total failure)
    ta = entry.get("task_a_parsing", {})
    tf = entry.get("task_f_relevance", {})
    if ta.get("exact_match", -1) == 0 and tf.get("f1", -1) == 0:
        return True
    return False


def is_small_sample(entry: dict) -> bool:
    """Check if any task used a small sample size."""
    for task_key, (n_field, min_n) in FULL_SCALE.items():
        task = entry.get(task_key, {})
        if "error" in task:
            continue
        n = task.get(n_field, 0)
        if 0 < n < min_n:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-all", action="store_true",
                        help="Clear ALL results for a completely fresh re-bench")
    parser.add_argument("--clear-think-only", action="store_true",
                        help="Only clear broken think-mode entries")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be cleared without modifying files")
    args = parser.parse_args()

    if not BENCH_FILE.exists():
        print(f"No results file found at {BENCH_FILE}")
        return

    with open(BENCH_FILE) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} entries from {BENCH_FILE}")

    # Archive current results
    if not args.dry_run:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = RESULTS_DIR / f"archive_{ts}"
        archive_dir.mkdir(exist_ok=True)
        shutil.copy2(BENCH_FILE, archive_dir / "llm_bench.json")
        print(f"Archived to {archive_dir}/llm_bench.json")

    if args.clear_all:
        to_clear = list(results.keys())
        reason = "clear-all"
    elif args.clear_think_only:
        to_clear = [k for k, v in results.items() if is_broken_think(v)]
        reason = "broken-think"
    else:
        # Default: clear broken think + small sample entries
        to_clear = []
        for k, v in results.items():
            if is_broken_think(v):
                to_clear.append(k)
            elif is_small_sample(v):
                to_clear.append(k)
        reason = "broken-think + small-sample"

    print(f"\n--- Entries to clear ({reason}) ---")
    for k in sorted(to_clear):
        entry = results[k]
        flags = []
        if is_broken_think(entry):
            flags.append("BROKEN_THINK")
        if is_small_sample(entry):
            # Show actual sample sizes
            sizes = []
            for task_key, (n_field, _) in FULL_SCALE.items():
                task = entry.get(task_key, {})
                n = task.get(n_field, 0)
                if n > 0:
                    sizes.append(f"{task_key}:n={n}")
            flags.append(f"SMALL({', '.join(sizes[:3])})")
        print(f"  {k}: {' | '.join(flags)}")

    print(f"\nTotal: {len(to_clear)} / {len(results)} entries to clear")

    if not args.dry_run and to_clear:
        for k in to_clear:
            del results[k]
        with open(BENCH_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nCleared {len(to_clear)} entries. {len(results)} remaining.")
        print(f"Re-bench will re-run cleared entries at full scale.")
    elif args.dry_run:
        print("\n(dry-run — no files modified)")


if __name__ == "__main__":
    main()
