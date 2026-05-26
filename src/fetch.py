"""Fetch a URL and extract clean, readable article text.

`fetch_url` is also exposed to the research agent as its `fetch_url` client-side tool.
"""
from __future__ import annotations

import logging
import re
import requests
import trafilatura

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
_TIMEOUT = 20
_THIN = 500       # chars; below this we assume an SPA shell and try the next tier
_JINA_BASE = "https://r.jina.ai/"

log = logging.getLogger("podify.fetch")


def fetch_url(url: str) -> str:
    """Return the main readable text of `url`. Raises on download/extraction failure."""
    html = _download(url)

    text = trafilatura.extract(html, include_comments=False, include_tables=True, url=url)
    if not text:
        text = _bs4_text(html)

    if len((text or "").strip()) < _THIN:
        spa = _spa_preload_text(html)
        if spa:
            log.info("fetch: used preload JSON fallback")
            text = spa

    if len((text or "").strip()) < _THIN:
        log.info("fetch: falling back to Jina Reader")
        text = _jina_fetch(url)

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


def _spa_preload_text(html: str) -> str:
    """Recover article text from common SPA JSON-preload patterns.

    Detects the pattern from the HTML before parsing — only one branch runs.
    """
    if re.search(r'window\.\w+\s*=\s*JSON\.parse\("', html):
        return _window_json_parse(html)
    from bs4 import BeautifulSoup
    if BeautifulSoup(html, "html.parser").find("script", {"type": "application/json"}):
        return _inline_script_json(html)
    return ""


def _window_json_parse(html: str) -> str:
    """window.X = JSON.parse("...escaped string...") — used by Substack and similar."""
    import json
    m = re.search(r'window\.\w+\s*=\s*JSON\.parse\("((?:[^"\\]|\\.)*)"\)', html)
    if not m:
        return ""
    try:
        inner = json.loads('"' + m.group(1) + '"')  # un-escape the JSON string value
        data = json.loads(inner)
        body_html = _find_key(data, "body_html")
        return _bs4_text(body_html) if body_html else ""
    except Exception:
        return ""


def _inline_script_json(html: str) -> str:
    """<script type="application/json"> — used by Next.js (__NEXT_DATA__) and similar."""
    import json
    from bs4 import BeautifulSoup
    try:
        tag = BeautifulSoup(html, "html.parser").find("script", {"type": "application/json"})
        if not tag or not tag.string:
            return ""
        data = json.loads(tag.string)
        body_html = _find_key(data, "body_html")
        return _bs4_text(body_html) if body_html else ""
    except Exception:
        return ""


def _jina_fetch(url: str) -> str:
    """Last-resort: fetch via Jina Reader, which handles JS-rendered pages."""
    try:
        resp = requests.get(_JINA_BASE + url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        text = resp.text
        marker = "Markdown Content:\n"
        idx = text.find(marker)
        return text[idx + len(marker):].strip() if idx != -1 else text.strip()
    except Exception as e:
        log.warning("Jina Reader failed for %s: %s", url, e)
        return ""


def _find_key(obj, key: str):
    """DFS search for a key anywhere in a nested dict/list."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = _find_key(item, key)
            if r:
                return r
    return None
