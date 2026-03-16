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
    # BGE-small-en-v1.5 and MiniLM-L6-v2 both output 384-dimensional vectors.
    # If you switch to a model with a different dimension, update this value accordingly.
    return 384

