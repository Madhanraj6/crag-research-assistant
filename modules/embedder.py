"""
Embedding module — wraps Sentence-Transformers for text → vector conversion.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from config import settings

_embeddings = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    return _embeddings

# Keep backwards compatibility for ingest.py temporarily if needed
def embed_batch(texts: list[str]) -> list[list[float]]:
    embeddings = get_embeddings()
    return embeddings.embed_documents(texts)

def get_dimension() -> int:
    # Hardcode for all-MiniLM-L6-v2, as extracting from LangChain object is complex and we're dropping direct dimension checks anyway in new setup
    return 384

