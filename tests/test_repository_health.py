from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parent.parent


class RepositoryHealthTests(unittest.TestCase):
    def test_governance_files_exist(self):
        required_files = [
            ROOT / "LICENSE",
            ROOT / "CONTRIBUTING.md",
            ROOT / "pyproject.toml",
            ROOT / ".gitattributes",
            ROOT / ".editorconfig",
            ROOT / ".env.example",
            ROOT / ".github" / "CODEOWNERS",
            ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md",
            ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml",
            ROOT / ".github" / "ISSUE_TEMPLATE" / "feature_request.yml",
            ROOT / ".github" / "ISSUE_TEMPLATE" / "config.yml",
            ROOT / ".github" / "workflows" / "repo-health.yml",
        ]
        for path in required_files:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), f"Missing required repository file: {path}")

    def test_essential_directories_exist(self):
        """Verify essential project directories tracked in git exist."""
        essential_dirs = [
            ROOT / "benchmarks" / "public_datasets",
            ROOT / "benchmarks",
            ROOT / "scmetaintel",
            ROOT / "scripts",
            ROOT / "tests",
        ]
        for path in essential_dirs:
            with self.subTest(path=path):
                self.assertTrue(path.exists(), f"Missing essential directory: {path}")

    def test_gitignore_covers_artifacts(self):
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        expected_rules = [
            "benchmarks/ground_truth/*.json",
            "benchmarks/results/*.json",
            "benchmarks/results/**",
            "article/",
            "article_figures/",
            "qdrant_data/",
            "!.env.example",
        ]
        for rule in expected_rules:
            with self.subTest(rule=rule):
                self.assertIn(rule, gitignore)

    def test_private_paths_are_not_tracked(self):
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        tracked = set(result.stdout.splitlines())
        forbidden = {
            "article/main.tex",
            "article_figures/article_summary.json",
            "docs/ARTICLE_PLAN.tex",
            "docs/EXPERIMENT_PLAN.md",
            "docs/STATUS_REPORT.md",
            "benchmarks/eval_queries.json",
            "benchmarks/BENCHMARK_DESIGN.md",
            "benchmarks/01_build_ground_truth.py",
            "benchmarks/04_bench_llm.py",
            "benchmarks/08_bench_ablation.py",
            "scripts/prep_rebench.py",
            "scripts/generate_article_figures.py",
            "scmetaintel/eval.py",
            "scmetaintel/evaluate.py",
            "tests/test_all_models.py",
        }
        leaked = sorted(forbidden & tracked)
        self.assertEqual(leaked, [], f"Private paths must not be tracked: {leaked}")

    def test_pyproject_exposes_console_scripts(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('scmetaintel = "scmetaintel.__main__:main"', pyproject)
        self.assertIn('geodh = "geodh.__main__:main"', pyproject)
        self.assertIn('dependencies = { file = ["requirements.txt"] }', pyproject)

    def test_readme_documents_repo_health(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Contributing and local validation", readme)
        self.assertIn(".github/workflows/repo-health.yml", readme)
        self.assertIn("python -m unittest discover -s tests -p 'test_repository_health.py' -v", readme)


if __name__ == "__main__":
    unittest.main()
