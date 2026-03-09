"""
Vector Store module — FAISS-based local vector storage for document retrieval.
No server required. Index is persisted to the faiss_index/ directory.
"""

import os
from langchain_community.vectorstores import FAISS
from modules.embedder import get_embeddings
from config import settings

_vector_store: FAISS | None = None

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _index_path() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, settings.FAISS_INDEX_PATH)


def _store_exists() -> bool:
    idx = _index_path()
    return os.path.exists(os.path.join(idx, "index.faiss"))


# -------------------------------------------------------------------
# Public API  (same surface as the old Qdrant module)
# -------------------------------------------------------------------

def get_vector_store() -> FAISS:
    """Return the singleton FAISS vector store (loads from disk on first call)."""
    global _vector_store
    if _vector_store is None:
        embeddings = get_embeddings()
        if _store_exists():
            _vector_store = FAISS.load_local(
                _index_path(),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            # Bootstrap with a dummy doc so FAISS has a valid index to work with.
            # It gets replaced the first time Ingest Papers is run.
            _vector_store = FAISS.from_texts(
                ["placeholder — please ingest papers to populate the knowledge base"],
                embeddings,
            )
    return _vector_store


def save_vector_store() -> None:
    """Persist the in-memory FAISS index to disk."""
    global _vector_store
    if _vector_store is not None:
        os.makedirs(_index_path(), exist_ok=True)
        _vector_store.save_local(_index_path())


def add_documents(texts: list[str], metadatas: list[dict]) -> int:
    """Add new documents to the store and persist. Returns count added."""
    global _vector_store
    embeddings = get_embeddings()
    if _vector_store is None or not _store_exists():
        _vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    else:
        _vector_store.add_texts(texts, metadatas=metadatas)
    save_vector_store()
    return len(texts)


def get_collection_count() -> int:
    """Return approximate number of documents in the index."""
    try:
        store = get_vector_store()
        return store.index.ntotal
    except Exception:
        return 0
