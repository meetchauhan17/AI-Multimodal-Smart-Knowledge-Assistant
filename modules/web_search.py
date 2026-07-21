"""
modules/web_search.py — Live web search utility for real-time news and current events.

Uses DuckDuckGo HTML endpoint (free, no API key required) to fetch web snippets.
"""

from __future__ import annotations

import urllib.parse
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup

from core.logger import logger


def search_web(query: str, max_results: int = 4) -> List[Dict[str, str]]:
    """Search the web using DuckDuckGo and return a list of result dicts with 'title' and 'snippet'."""
    try:
        logger.info(f"Performing live web search for: '{query}'")
        encoded = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []

        # Parse DuckDuckGo HTML results
        for res in soup.find_all("div", class_=lambda c: c and "result" in c):
            snippet_elem = res.find("a", class_="result__snippet")
            title_elem = res.find("a", class_="result__a")
            
            if snippet_elem and title_elem:
                title = title_elem.get_text(strip=True)
                snippet = snippet_elem.get_text(strip=True)
                results.append({"title": title, "snippet": snippet})
                if len(results) >= max_results:
                    break

        logger.info(f"Web search returned {len(results)} snippets.")
        return results
    except Exception as exc:
        logger.warning(f"Web search failed: {exc}")
        return []
