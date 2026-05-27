"""
Unified CLI entry point for scMetaIntel-Hub.
"""

from __future__ import annotations
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        prog="scmetaintel",
        description="scMetaIntel-Hub: Intelligent metadata enrichment for GEO datasets",
        epilog="Use '<command> -h' for command-specific help."
    )
    
    subparsers = parser.add_subparsers(dest="command", title="Commands", metavar="COMMAND")
    
    # Core intelligence commands
    subparsers.add_parser("enrich", help="Extract sample metadata from GEO + PubMed")
    subparsers.add_parser("ontology", help="Normalize metadata to ontologies")
    subparsers.add_parser("embed", help="Generate embeddings and build vector index")
    subparsers.add_parser("retrieve", help="Search datasets")
    subparsers.add_parser("answer", help="Generate grounded answer from retrieved studies")
    subparsers.add_parser("chat", help="Interactive chat interface")
    
    # Acquisition bridge command
    subparsers.add_parser("geo", help="Delegate to the integrated GEO-DataHub bridge CLI")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Route to the appropriate sub-command
    sys.argv = [sys.argv[0], *sys.argv[2:]]  # keep only args after command
    
    if args.command == "enrich":
        from .enrich import main as cmd
    elif args.command == "ontology":
        from .ontology import main as cmd
    elif args.command == "embed":
        from .embed import main as cmd
    elif args.command == "retrieve":
        from .retrieve import main as cmd
    elif args.command == "answer":
        from .answer import main as cmd
    elif args.command == "chat":
        from .chat import main as cmd
    elif args.command == "geo":
        from geodh.cli import main as cmd
    else:
        # Should not happen due to argparse, but keep safe
        print(f"Unknown command: {args.command}")
        sys.exit(1)
    
    cmd()

if __name__ == "__main__":
    main()