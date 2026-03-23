"""
GEO-DataHub: Link Verification Module
=======================================
Verify that download links are accessible without full download.
Uses HTTP HEAD requests to check file availability and sizes.

Author: GEO-DataHub Pipeline
"""

import logging
import time
from typing import List, Optional, Tuple
from dataclasses import dataclass

import requests

logger = logging.getLogger("geo_verify")


@dataclass
class LinkVerification:
    """Result of verifying a single download link."""
    url: str
    filename: str
    status_code: int = 0
    accessible: bool = False
    file_size: Optional[int] = None       # bytes
    file_size_str: str = ""
    content_type: str = ""
    error: str = ""


def verify_link(url: str, timeout: int = 15) -> LinkVerification:
    """
    Verify a single download link using HTTP HEAD request.
    
    Does NOT download the file — only checks accessibility and metadata.
    """
    filename = url.rstrip("/").split("/")[-1]
    result = LinkVerification(url=url, filename=filename)
    
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        result.status_code = resp.status_code
        result.accessible = (resp.status_code == 200)
        
        if result.accessible:
            # Parse content length
            cl = resp.headers.get("Content-Length")
            if cl:
                result.file_size = int(cl)
                result.file_size_str = _format_size(result.file_size)
            
            result.content_type = resp.headers.get("Content-Type", "")
    
    except requests.exceptions.Timeout:
        result.error = "timeout"
    except requests.exceptions.ConnectionError:
        result.error = "connection_error"
    except Exception as e:
        result.error = str(e)
    
    return result


def verify_links(urls: List[str], timeout: int = 15, delay: float = 0.2) -> List[LinkVerification]:
    """
    Verify multiple download links.
    
    Args:
        urls: List of URLs to verify
        timeout: Timeout per request in seconds
        delay: Delay between requests to be polite
    
    Returns:
        List of LinkVerification results
    """
    results = []
    for url in urls:
        result = verify_link(url, timeout=timeout)
        results.append(result)
        time.sleep(delay)
    return results


def print_verification_report(results: List[LinkVerification], gse_id: str = ""):
    """Print a formatted verification report."""
    accessible = sum(1 for r in results if r.accessible)
    total_size = sum(r.file_size or 0 for r in results if r.accessible)
    
    header = f"  Link Verification: {gse_id}" if gse_id else "  Link Verification"
    print(f"\n{header}")
    print(f"  {'─'*60}")
    
    for r in results:
        if r.accessible:
            size_str = r.file_size_str or "size unknown"
            print(f"    ✓ {r.filename:50s} [{size_str}]")
        else:
            err = r.error or f"HTTP {r.status_code}"
            print(f"    ✗ {r.filename:50s} [{err}]")
    
    print(f"  {'─'*60}")
    print(f"  {accessible}/{len(results)} accessible | Total: {_format_size(total_size)}")


def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024 and i < len(units) - 1:
        size /= 1024
        i += 1
    return f"{size:.1f} {units[i]}"
