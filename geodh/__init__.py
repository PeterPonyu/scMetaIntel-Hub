"""Vendored GEO acquisition commands for scMetaIntel-Hub.

This package contains the full GEO-DataHub data acquisition pipeline,
vendored from the standalone GEO-DataHub project.
"""

from .geodh import main as geodh_main
from .geo_search import search_geo, GEOSeriesInfo
