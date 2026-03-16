import re
from config import settings


_STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","shall",
    "should","may","might","must","can","could","of","in","on",
    "at","to","for","with","by","from","as","into","through",
    "and","or","but","not","what","how","why","when","where",
    "which","who","that","this","these","those","it","its",
    "about","explain","describe","tell","me","i","you","we"
}


def _normalize(word: str) -> str:
    for suffix in ("ing", "tion", "ers", "ed", "er", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 3:
            return word[:-len(suffix)]
    return word


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {_normalize(w) for w in words if w not in _STOPWORDS}


def _keyword_overlap(query: str, doc_text: str) -> float:
    q_kw = _keywords(query)
    d_kw = _keywords(doc_text)

    if not q_kw or not d_kw:
        return 0.0

    intersection = q_kw & d_kw
    union = q_kw | d_kw

    return len(intersection) / len(union)


def evaluate(search_results: list[dict], query: str = "") -> dict:

    if not search_results:
        return {
            "relevant": False,
            "avg_score": 0.0,
            "reason": "No documents retrieved",
            "scores": [],
        }

    scores = [r["score"] for r in search_results]

    avg_score = sum(scores) / len(scores)
    max_score = max(scores)

    score_ok = (
        avg_score >= settings.RELEVANCE_THRESHOLD
        and max_score >= settings.RELEVANCE_THRESHOLD
    )

    if query:
        overlaps = [_keyword_overlap(query, r["text"]) for r in search_results]
        avg_overlap = sum(overlaps) / len(overlaps)

        keyword_ok = avg_overlap >= 0.25 or max(overlaps) >= 0.40
    else:
        keyword_ok = True
        avg_overlap = 0.0

    relevant = score_ok and keyword_ok

    return {
        "relevant": relevant,
        "avg_score": round(avg_score, 4),
        "max_score": round(max_score, 4),
        "keyword_overlap": round(avg_overlap, 4),
        "scores": scores,
    }