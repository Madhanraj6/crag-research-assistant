"""
Relevance Evaluator — determines whether retrieved documents are sufficient
to answer the user's specific question.

Two-layer evaluation (no extra LLM call needed):
  1. Embedding similarity score gate  (fast — scores are pre-computed)
  2. Keyword overlap check            (fast — no API call)
"""

import re
from config import settings


# ---------------------------------------------------------------------------
# Keyword overlap helpers
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "on",
    "at", "to", "for", "with", "by", "from", "as", "into", "through",
    "and", "or", "but", "not", "what", "how", "why", "when", "where",
    "which", "who", "that", "this", "these", "those", "it", "its",
    "about", "explain", "describe", "tell", "me", "i", "you", "we",
}


def _keywords(text: str) -> set[str]:
    """Extract meaningful words from text, lowercased and de-stopworded."""
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _keyword_overlap(query: str, doc_text: str) -> float:
    """
    Jaccard-like overlap between query keywords and document keywords.
    Returns 0.0–1.0 (1.0 = all query keywords found in the doc).
    """
    q_kw = _keywords(query)
    if not q_kw:
        return 0.5  # neutral if query has no meaningful keywords
    d_kw = _keywords(doc_text)
    matched = q_kw & d_kw
    return len(matched) / len(q_kw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(search_results: list[dict], query: str = "") -> dict:
    """
    Evaluate the relevance of search results.

    Args:
        search_results: List of {text, score, metadata} dicts from FAISS
        query: The original user question (used for keyword overlap check)

    Returns:
        {
            "relevant": bool,
            "avg_score": float,
            "reason": str,
            "scores": list[float]
        }
    """
    if not search_results:
        return {
            "relevant": False,
            "avg_score": 0.0,
            "reason": "No documents retrieved",
            "scores": [],
        }

    scores = [r["score"] for r in search_results]
    avg_score = sum(scores) / len(scores)
    above_threshold = sum(1 for s in scores if s >= settings.RELEVANCE_THRESHOLD)
    ratio = above_threshold / len(scores)

    # ---- Layer 1: score gate ----
    score_ok = avg_score >= settings.RELEVANCE_THRESHOLD and ratio >= 0.5

    # ---- Layer 2: keyword overlap (only if query is provided) ----
    if query:
        overlaps = [_keyword_overlap(query, r["text"]) for r in search_results]
        avg_overlap = sum(overlaps) / len(overlaps)
        # At least one doc should have ≥30% keyword overlap
        keyword_ok = avg_overlap >= 0.20 or max(overlaps) >= 0.30
    else:
        keyword_ok = True  # skip check if no query passed
        avg_overlap = 0.0

    relevant = score_ok and keyword_ok

    if relevant:
        reason = (
            f"Context relevant — {above_threshold}/{len(scores)} docs above score threshold "
            f"(avg={avg_score:.3f})"
            + (f", keyword_overlap={avg_overlap:.2f}" if query else "")
        )
    elif not score_ok:
        reason = (
            f"Low similarity — avg={avg_score:.3f}, "
            f"only {above_threshold}/{len(scores)} docs above threshold"
        )
    else:
        reason = (
            f"Off-topic docs — score ok (avg={avg_score:.3f}) but "
            f"keyword_overlap={avg_overlap:.2f} too low for this query"
        )

    return {
        "relevant": relevant,
        "avg_score": round(avg_score, 4),
        "reason": reason,
        "scores": scores,
    }