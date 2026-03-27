#!/usr/bin/env python3
"""Smoke-test all installed Ollama LLMs with a minimal biomedical prompt."""
import json
import time
import sys
from pathlib import Path
import requests

OLLAMA_BASE = "http://localhost:11434"
PROMPT = "What is single-cell RNA sequencing? Answer in one sentence."

def get_installed_models():
    """Get list of installed models from Ollama."""
    resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]

def test_model(model_name: str, timeout: int = 120) -> dict:
    """Test a single model with a simple generation."""
    result = {"model": model_name, "status": "UNKNOWN", "time_s": 0, "response": "", "error": ""}
    start = time.time()
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": model_name,
                "prompt": PROMPT,
                "stream": False,
                "think": False,
                "options": {"num_predict": 80, "temperature": 0.1},
            },
            timeout=timeout,
        )
        elapsed = time.time() - start
        result["time_s"] = round(elapsed, 1)

        if resp.status_code == 200:
            data = resp.json()
            text = data.get("response", "").strip()
            thinking = data.get("thinking", "").strip()
            # Some models (Qwen3/3.5) may put content in thinking field
            if not text and thinking:
                text = thinking
                result["used_thinking"] = True
            result["response"] = text[:200]
            if len(text) > 5:
                result["status"] = "OK"
            else:
                result["status"] = "EMPTY"
                result["error"] = "Response too short"
        else:
            result["status"] = "HTTP_ERROR"
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:100]}"
    except requests.exceptions.Timeout:
        result["time_s"] = round(time.time() - start, 1)
        result["status"] = "TIMEOUT"
        result["error"] = f"Timed out after {timeout}s"
    except Exception as e:
        result["time_s"] = round(time.time() - start, 1)
        result["status"] = "ERROR"
        result["error"] = str(e)[:200]
    return result

def main():
    print("=" * 70)
    print("LLM SMOKE TEST — All Installed Ollama Models")
    print("=" * 70)

    models = get_installed_models()
    print(f"\nFound {len(models)} installed models\n")

    results = []
    for i, model in enumerate(models, 1):
        print(f"[{i}/{len(models)}] Testing {model} ...", end=" ", flush=True)
        r = test_model(model)
        results.append(r)
        status_icon = "PASS" if r["status"] == "OK" else "FAIL"
        print(f"{status_icon} ({r['time_s']}s)")
        if r["status"] != "OK":
            print(f"         Error: {r['error']}")
        else:
            print(f"         Response: {r['response'][:100]}...")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in results if r["status"] == "OK")
    failed = [r for r in results if r["status"] != "OK"]
    print(f"\nPassed: {passed}/{len(results)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for r in failed:
            print(f"  - {r['model']}: {r['status']} — {r['error']}")

    # Save results
    out_path = str(Path(__file__).resolve().parent.parent / "benchmarks" / "smoke_test_llms.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")

    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())
