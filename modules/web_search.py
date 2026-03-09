"""
Web Search Fallback — queries the arXiv API for live research papers.
"""

import requests
import xmltodict
from typing import Optional


ARXIV_API_URL = "http://export.arxiv.org/api/query"


def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """
    Search arXiv for papers matching the query.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: title, abstract, authors, url, published
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        response = requests.get(ARXIV_API_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[WebSearch] arXiv API error: {e}")
        return []

    try:
        data = xmltodict.parse(response.text)
        feed = data.get("feed", {})
        entries = feed.get("entry", [])

        # If single result, xmltodict returns a dict instead of list
        if isinstance(entries, dict):
            entries = [entries]

        results = []
        for entry in entries:
            # Handle authors (can be list or single dict)
            authors_raw = entry.get("author", [])
            if isinstance(authors_raw, dict):
                authors_raw = [authors_raw]
            authors = [a.get("name", "") for a in authors_raw]

            results.append({
                "title": entry.get("title", "").strip().replace("\n", " "),
                "abstract": entry.get("summary", "").strip().replace("\n", " "),
                "authors": authors,
                "url": entry.get("id", ""),
                "published": entry.get("published", ""),
                "source": "arxiv_web",
            })

        return results

    except Exception as e:
        print(f"[WebSearch] Parse error: {e}")
        return []


def format_for_context(papers: list[dict]) -> list[dict]:
    """
    Format arXiv results into context chunks suitable for the aggregator.

    Returns list of {text, metadata} dicts.
    """
    chunks = []
    for paper in papers:
        text = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
        metadata = {
            "source": "arxiv_web",
            "title": paper["title"],
            "authors": ", ".join(paper["authors"][:3]),
            "url": paper["url"],
            "published": paper["published"],
        }
        chunks.append({"text": text, "metadata": metadata})
    return chunks
