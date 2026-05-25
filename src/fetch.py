"""Fetch a URL and extract clean, readable article text.

`fetch_url` is also exposed to the research agent as its `fetch_url` client-side tool
(see agent.py) so Claude can pull the full text of a page it decides is worth reading.
"""
from __future__ import annotations

import requests
import trafilatura

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
_TIMEOUT = 20


def fetch_url(url: str) -> str:
    """Return the main readable text of `url`. Raises on download/extraction failure."""
    html = _download(url)
    text = trafilatura.extract(
        html, include_comments=False, include_tables=True, url=url
    )
    if not text:
        text = _bs4_text(html)
    text = (text or "").strip()
    if not text:
        raise ValueError(f"Could not extract readable text from {url}")
    return text


def _download(url: str) -> str:
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def _bs4_text(html: str) -> str:
    """Fallback extraction: strip chrome tags and take the visible text."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)
