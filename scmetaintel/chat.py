"""
Terminal REPL chat interface for scMetaIntel-Hub.
"""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .answer import AnswerGenerator
from .config import (
    TRUNCATE_ABSTRACT,
    TRUNCATE_DETAIL_DESIGN,
    TRUNCATE_DETAIL_SUMMARY,
    truncate_text,
)
from .retrieve import HybridRetriever

logger = logging.getLogger(__name__)
console = Console()

WELCOME = """
[bold cyan]scMetaIntel-Hub[/] — Unified GEO acquisition + metadata intelligence platform

Ask natural-language questions about single-cell datasets:

  [dim]> Find human lung scRNA-seq datasets with fibrosis
  > Which datasets contain CD8 T cells in melanoma?
  > Compare breast cancer scRNA-seq studies
  > Show studies with PBMC CITE-seq[/]

Commands:
  [bold]/help[/]            — Show this help
  [bold]/stats[/]           — Show index statistics
  [bold]/detail GSExxxxx[/] — Show full details for a study
  [bold]/quit[/]            — Exit
"""


class ChatSession:
    def __init__(self, stream: bool = True):
        self.retriever = HybridRetriever()
        self.answerer = AnswerGenerator()
        self.stream = stream

    def _show_welcome(self):
        console.print(Panel(WELCOME, title="Welcome", border_style="cyan"))

    def _show_stats(self):
        try:
            from .embed import StudyEmbedder
            info = StudyEmbedder().get_collection_info()
            table = Table(title="Index Statistics")
            table.add_column("Metric", style="bold")
            table.add_column("Value")
            for k, v in info.items():
                table.add_row(k, str(v))
            console.print(table)
        except Exception as e:
            console.print(f"[red]Error getting stats: {e}[/]")

    def _show_detail(self, gse_id: str):
        study = self.retriever.load_full_study(gse_id)
        if study is None:
            console.print(f"[yellow]Study {gse_id} not found.[/]")
            return

        console.print(f"\n[bold cyan]{study.gse_id}[/]: {study.title}\n")
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        table.add_row("Organism", study.organism)
        table.add_row("Domain", study.domain)
        table.add_row("Modalities", ", ".join(study.modalities) if study.modalities else "N/A")
        table.add_row("Samples", str(study.n_samples))
        table.add_row("Submitted", study.submission_date)
        table.add_row("Platform", study.platform)
        table.add_row("Series Type", study.series_type)

        cs = study.characteristics_summary
        table.add_row("Tissues", ", ".join(cs.tissues) if cs.tissues else "N/A")
        table.add_row("Diseases", ", ".join(cs.diseases) if cs.diseases else "N/A")
        table.add_row("Cell Types", ", ".join(cs.cell_types) if cs.cell_types else "N/A")
        table.add_row("Treatments", ", ".join(cs.treatments) if cs.treatments else "N/A")
        table.add_row("Donors", str(cs.donor_count))
        table.add_row("Sex", ", ".join(cs.sex) if cs.sex else "N/A")
        table.add_row("Age Range", cs.age_range or "N/A")
        console.print(table)

        if study.summary:
            console.print(
                f"\n[bold]Summary:[/]\n{truncate_text(study.summary, TRUNCATE_DETAIL_SUMMARY)}\n"
            )
        if study.overall_design:
            console.print(
                f"[bold]Design:[/]\n{truncate_text(study.overall_design, TRUNCATE_DETAIL_DESIGN)}\n"
            )
        if study.pubmed:
            console.print(f"[bold]PubMed:[/] {study.pubmed.pmid}")
            if study.pubmed.abstract:
                console.print(f"\n{truncate_text(study.pubmed.abstract, TRUNCATE_ABSTRACT)}\n")

    def _handle_query(self, query: str):
        with console.status("[bold cyan]Searching...[/]"):
            parsed, results = self.retriever.search(query)
        if not results:
            console.print("[yellow]No matching datasets found.[/]")
            return

        console.print(f"\n[bold]Found {len(results)} datasets:[/]\n")
        table = Table()
        table.add_column("#", style="dim")
        table.add_column("GSE", style="cyan bold")
        table.add_column("Title", max_width=50)
        table.add_column("Organism")
        table.add_column("Tissues")
        table.add_column("Score", justify="right")
        for i, r in enumerate(results, 1):
            s = r.study
            tissues = ", ".join(s.characteristics_summary.tissues[:3]) or "-"
            score = f"{r.rerank_score:.2f}" if r.rerank_score else f"{r.score:.2f}"
            table.add_row(str(i), s.gse_id, s.title[:50], s.organism, tissues, score)
        console.print(table)

        console.print()
        if self.stream:
            console.print("[bold]Analysis:[/]\n")
            for chunk in self.answerer.generate_streaming(query, parsed, results):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            sys.stdout.write("\n\n")
        else:
            with console.status("[bold cyan]Generating answer...[/]"):
                response = self.answerer.generate(query, parsed, results)
            console.print(Markdown(response.answer_text))
            console.print()

    def run(self):
        self._show_welcome()
        while True:
            try:
                query = console.input("\n[bold green]>[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/]")
                break
            if not query:
                continue
            if query.lower() in ("/quit", "/exit", "/q"):
                console.print("[dim]Goodbye![/]")
                break
            if query.lower() == "/help":
                self._show_welcome()
                continue
            if query.lower() == "/stats":
                self._show_stats()
                continue
            if query.lower().startswith("/detail "):
                self._show_detail(query.split()[1].strip().upper())
                continue
            try:
                self._handle_query(query)
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")
                logger.exception("Query failed")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="scMetaIntel-Hub chat interface")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    ChatSession(stream=not args.no_stream).run()


if __name__ == "__main__":
    main()
