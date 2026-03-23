"""
scMetaIntel — Single-Cell Metadata Intelligence Platform
=========================================================
LLM-powered retrieval system for single-cell dataset discovery,
metadata harmonization, and hypothesis support.

Pipeline: Query → Parse → Embed → Retrieve → Rerank → Answer

Built on top of the GEO-DataHub data acquisition backbone.
"""

from .config import Config, get_config

__version__ = "0.2.0"
__all__ = ["Config", "get_config"]
