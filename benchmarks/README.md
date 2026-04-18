# Public Benchmarks

This directory keeps the public evaluation surface that remains safe to publish with the software repository.

## Included

- `05_bench_public.py`: runs local models against public datasets
- `public_datasets/`: dataset adapters and packaged public samples used by the harness
- `bench_utils.py`: shared runtime helpers for benchmark execution

## Usage

Run the public benchmark harness:

```bash
python benchmarks/05_bench_public.py --max-samples 50
```

Filter to specific model keys:

```bash
python benchmarks/05_bench_public.py --models qwen3-8b llama3.1-8b
```

Filter to categories:

```bash
python benchmarks/05_bench_public.py --categories general biomedical
```

## Categories

- `general`
- `reasoning`
- `biomedical`
- `structured`
- `tool_use`
- `commonsense`

## Notes

- Results are written to `benchmarks/results/`, which is ignored by git.
- This directory documents only the public benchmark harness that ships with the repository.
