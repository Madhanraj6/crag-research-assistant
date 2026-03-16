"""
Data Ingestion Script — fetches AI paper abstracts from arXiv and loads them into FAISS.
"""

import requests
import xmltodict
from modules import vector_store

ARXIV_API_URL = "http://export.arxiv.org/api/query"

# Categories to fetch
CATEGORIES = [
    ("cs.AI", "Artificial Intelligence"),
    ("cs.LG", "Machine Learning"),
    ("cs.CL", "Computational Linguistics / NLP"),
]


def fetch_arxiv_papers(category: str, max_results: int = 100) -> list[dict]:
    """Fetch papers from a specific arXiv category."""
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    try:
        response = requests.get(ARXIV_API_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  ✗ Error fetching {category}: {e}")
        return []

    try:
        data = xmltodict.parse(response.text)
        entries = data.get("feed", {}).get("entry", [])

        if isinstance(entries, dict):
            entries = [entries]

        papers = []
        for entry in entries:
            title = entry.get("title", "").strip().replace("\n", " ")
            abstract = entry.get("summary", "").strip().replace("\n", " ")
            url = entry.get("id", "")
            published = entry.get("published", "")

            authors_raw = entry.get("author", [])
            if isinstance(authors_raw, dict):
                authors_raw = [authors_raw]
            authors = ", ".join(a.get("name", "") for a in authors_raw[:5])

            if title and abstract:
                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "url": url,
                    "published": published,
                    "category": category,
                })

        return papers

    except Exception as e:
        print(f"  ✗ Parse error for {category}: {e}")
        return []


def ingest(papers_per_category: int = 100) -> dict:
    """
    Fetch papers from arXiv and ingest into FAISS vector store.

    Returns summary stats.
    """
    print("=" * 60)
    print("  Corrective Agentic RAG - Data Ingestion (FAISS)")
    print("=" * 60)

    all_papers = []

    for cat_code, cat_name in CATEGORIES:
        print(f"\n[Ingest] Fetching {papers_per_category} papers from {cat_name} ({cat_code})...")
        papers = fetch_arxiv_papers(cat_code, max_results=papers_per_category)
        print(f"  - Fetched {len(papers)} papers")
        all_papers.extend(papers)

    if not all_papers:
        return {"status": "error", "message": "No papers fetched", "count": 0}

    print(f"\n[Ingest] Total papers fetched: {len(all_papers)}")

    # Prepare texts and metadata
    texts = [
        f"Title: {p['title']}\nAbstract: {p['abstract']}"
        for p in all_papers
    ]
    metadatas = [
        {
            "title": p["title"],
            "authors": p["authors"],
            "url": p["url"],
            "published": p["published"],
            "category": p["category"],
            "source": "arxiv_ingested",
        }
        for p in all_papers
    ]

    # Add to FAISS (embeddings generated inside add_documents)
    print("\n[Ingest] Generating embeddings & upserting into FAISS...")
    count = vector_store.add_documents(texts, metadatas)
    print(f"  - Upserted {count} documents")

    total = vector_store.get_collection_count()
    print(f"\n[Ingest] Index now has {total} documents total")
    print("=" * 60)

    return {
        "status": "success",
        "fetched": len(all_papers),
        "upserted": count,
        "total_in_collection": total,
        "categories": [c[0] for c in CATEGORIES],
    }


if __name__ == "__main__":
    import sys
    count = 100
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            pass
    ingest(papers_per_category=count)
