
from sentence_transformers import CrossEncoder

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def rerank(query: str, docs: list[dict], top_n: int = 3) -> list[dict]:
    reranker = get_reranker()
    pairs = [(query, d["text"]) for d in docs]
    scores = reranker.predict(pairs)
    for doc, score in zip(docs, scores):
        doc["rerank_score"] = float(score)
    return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:top_n]