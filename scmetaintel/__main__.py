"""
Unified CLI entry point for scMetaIntel-Hub.

Usage:
    python -m scmetaintel <command> [options]

Core intelligence commands:
    enrich      Extract sample metadata from GEO + PubMed
    ontology    Normalize metadata to ontologies
    embed       Generate embeddings and build vector index
    retrieve    Search datasets
    answer      Generate grounded answer from retrieved studies
    chat        Interactive chat interface

Acquisition bridge commands:
    geo         Delegate to the integrated GEO-DataHub bridge CLI
"""

from __future__ import annotations

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]

    if command == "enrich":
        from .enrich import main as cmd
    elif command == "ontology":
        from .ontology import main as cmd
    elif command == "embed":
        from .embed import main as cmd
    elif command == "retrieve":
        from .retrieve import main as cmd
    elif command == "answer":
        from .answer import main as cmd
    elif command == "chat":
        from .chat import main as cmd
    elif command == "geo":
        from geodh.cli import main as cmd
    else:
        print(f"Unknown command: {command}\n")
        print(__doc__)
        sys.exit(1)

    cmd()


if __name__ == "__main__":
    main()
