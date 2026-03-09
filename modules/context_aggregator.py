"""
Context Aggregator — merges vector retrieval and web search results into
a normalized prompt context for the LLM.

Key design: source labels are derived from stable document identifiers
(arXiv IDs, paper titles) instead of sequential numbers, so references
like [KB: Attention Is All You Need] never break across conversation turns.
"""


def _stable_label(meta: dict, prefix: str, fallback_index: int) -> str:
    """
    Build a human-readable, stable source label from document metadata.

    Priority:
      1. arXiv URL  →  arXiv:2304.12345
      2. Paper title (first 40 chars)
      3. Fallback sequential ID
    """
    url = meta.get("url", "")
    # arXiv URL pattern: https://arxiv.org/abs/2304.12345
    if "arxiv.org/abs/" in url:
        arxiv_id = url.split("arxiv.org/abs/")[-1].split("v")[0].strip()
        return f"{prefix}: arXiv:{arxiv_id}"

    title = meta.get("title", "").strip()
    if title:
        short_title = title[:45].strip()
        if len(title) > 45:
            short_title += "…"
        return f"{prefix}: {short_title}"

    return f"{prefix}-{fallback_index}"


def aggregate(
    vector_results: list[dict],
    web_results: list[dict] | None = None,
) -> dict:
    """
    Combine vector DB results and web search results into a structured context.

    Args:
        vector_results: Results from vector store search [{text, score, metadata}]
        web_results: Results from web search [{text, metadata}] (optional)

    Returns:
        {
            "context_text": str,       # Formatted context for LLM prompt
            "sources": list[dict],     # Source references for citation
            "source_map": dict,        # label → title (stable, for conversation memory)
            "vector_count": int,
            "web_count": int,
        }
    """
    sources = []
    context_parts = []
    source_map = {}  # Maps the label used in the prompt → human-readable title

    # --- Vector DB Results ---
    if vector_results:
        context_parts.append("=== RETRIEVED FROM KNOWLEDGE BASE ===\n")
        for i, result in enumerate(vector_results, 1):
            meta = result.get("metadata", {})
            label = _stable_label(meta, "KB", i)
            source_map[label] = meta.get("title", f"Document {i}")

            context_parts.append(f"[{label}]\n{result['text']}\n")
            sources.append({
                "id": label,
                "type": "knowledge_base",
                "score": result.get("score", 0),
                "title": meta.get("title", f"Document {i}"),
                "authors": meta.get("authors", "Unknown Authors"),
                "url": meta.get("url", ""),
                "category": meta.get("category", "Local KB"),
                "text": result["text"],
                "preview": result["text"][:150] + "..." if len(result["text"]) > 150 else result["text"],
            })

    # --- Web Search Results ---
    web_count = 0
    if web_results:
        context_parts.append("\n=== RETRIEVED FROM WEB (arXiv) ===\n")
        for i, result in enumerate(web_results, 1):
            meta = result.get("metadata", {})
            label = _stable_label(meta, "WEB", i)
            source_map[label] = meta.get("title", f"Web Result {i}")

            context_parts.append(f"[{label}]\n{result['text']}\n")
            sources.append({
                "id": label,
                "type": "arxiv_web",
                "title": meta.get("title", f"Web Result {i}"),
                "url": meta.get("url", ""),
                "authors": meta.get("authors", "Unknown Authors"),
                "category": meta.get("category", "Web Fallback"),
                "text": result["text"],
                "preview": result["text"][:150] + "..." if len(result["text"]) > 150 else result["text"],
            })
            web_count += 1

    context_text = "\n".join(context_parts)

    return {
        "context_text": context_text,
        "sources": sources,
        "source_map": source_map,
        "vector_count": len(vector_results) if vector_results else 0,
        "web_count": web_count,
    }
