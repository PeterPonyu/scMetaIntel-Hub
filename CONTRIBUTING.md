# Contributing to scMetaIntel-Hub

Thanks for contributing to `scMetaIntel-Hub`.

## Before you open a pull request

1. Create a feature branch from `main`.
2. Keep changes focused and document user-visible behavior in `README.md` when relevant.
3. Preserve generated-data boundaries: source code and docs belong in git; large runtime artifacts stay out.
4. Keep placeholder directories intact so local pipelines have predictable paths.

## Local validation

Run the lightweight repository checks before opening a PR:

```bash
python -m compileall scmetaintel geodh benchmarks scripts tests
python -m unittest discover -s tests -p 'test_repository_health.py' -v
```

If your change touches runtime behavior, also run the smallest relevant smoke test for that area and summarize the result in the PR description.

## Commit and PR guidance

- Use clear, imperative commit messages.
- Explain **what changed**, **why**, and **how you validated it**.
- Link related issues with `Fixes #<number>` when applicable.
- Include screenshots or logs for UI/CLI changes when they help reviewers.

## Issues

Use the repository issue forms for bugs and feature requests so reports include enough detail to reproduce and triage.
