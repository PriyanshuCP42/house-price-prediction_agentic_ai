"""
Embedding model setup for RAG — uses sentence-transformers (runs locally, no API).
"""

from config.settings import EMBEDDING_MODEL

_embedder = None


def get_embedding_function():
    """Returns a HuggingFace embedding function for ChromaDB."""
    global _embedder
    if _embedder is None:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        _embedder = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return _embedder
