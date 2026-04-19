"""
Lightweight RAG Retriever Tool.

Streamlit Cloud can be slow or memory-constrained when installing ChromaDB,
Torch, and sentence-transformers. This retriever keeps deployment reliable by
using lexical search over the checked-in markdown knowledge sources. The
public function signatures match the original Chroma-backed retriever.
"""

import os
import re
from functools import lru_cache
from langchain_core.tools import tool
from config.settings import KNOWLEDGE_SOURCES_PATH, RAG_TOP_K, CHUNK_SIZE, CHUNK_OVERLAP


STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "into", "about",
    "property", "real", "estate", "house", "home", "market", "county",
    "king", "price", "prices", "zipcode",
}


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())
        if token not in STOPWORDS
    }


def _chunk_text(text: str, source_name: str) -> list[dict]:
    sections = text.split("\n## ")
    chunks = []
    for index, section in enumerate(sections):
        if index > 0:
            section = "## " + section
        header = section.split("\n", 1)[0].strip("# ").strip() or source_name

        if len(section) <= CHUNK_SIZE:
            chunks.append({
                "text": section.strip(),
                "metadata": {"source": source_name, "section": header},
                "tokens": _tokenize(section),
            })
            continue

        start = 0
        while start < len(section):
            end = min(len(section), start + CHUNK_SIZE)
            chunk = section[start:end].strip()
            if chunk:
                chunks.append({
                    "text": chunk,
                    "metadata": {"source": source_name, "section": header},
                    "tokens": _tokenize(chunk),
                })
            if end == len(section):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


@lru_cache(maxsize=1)
def _load_knowledge_chunks() -> tuple[dict, ...]:
    chunks: list[dict] = []
    if not os.path.isdir(KNOWLEDGE_SOURCES_PATH):
        return tuple()

    for filename in sorted(os.listdir(KNOWLEDGE_SOURCES_PATH)):
        if not filename.endswith(".md"):
            continue
        path = os.path.join(KNOWLEDGE_SOURCES_PATH, filename)
        with open(path, "r", encoding="utf-8") as file:
            text = file.read()
        source_name = filename.replace(".md", "").replace("_", " ").title()
        chunks.extend(_chunk_text(text, source_name))
    return tuple(chunks)


def _score_chunk(query_tokens: set[str], chunk: dict) -> float:
    if not query_tokens:
        return 0.0
    chunk_tokens = chunk["tokens"]
    overlap = query_tokens & chunk_tokens
    if not overlap:
        return 0.0

    source = f"{chunk['metadata'].get('source', '')} {chunk['metadata'].get('section', '')}".lower()
    title_boost = sum(1 for token in query_tokens if token in source) * 0.35
    coverage = len(overlap) / max(len(query_tokens), 1)
    density = len(overlap) / max(len(chunk_tokens), 1)
    return coverage * 0.75 + density * 0.25 + title_boost


def _get_collection():
    """Compatibility helper for older callers that checked KB availability."""
    return _load_knowledge_chunks()


@tool
def rag_search(query: str, n_results: int = 5) -> str:
    """Search the real estate knowledge base for relevant market insights,
    zipcode profiles, investment guidelines, or legal disclaimers.
    Returns the most relevant text passages with source attribution."""

    results = rag_search_raw(query, n_results=n_results)
    if not results["documents"]:
        return "No relevant information found in the knowledge base."

    passages = []
    sources = set()
    for doc, meta in zip(results["documents"], results["metadatas"]):
        source = meta.get("source", "Unknown")
        section = meta.get("section", "")
        sources.add(source)
        passages.append(f"[Source: {source} - {section}]\n{doc}")

    return "\n\n---\n\n".join(passages) + "\n\n---\nSources referenced: " + ", ".join(sorted(sources))


def rag_search_raw(query: str, n_results: int = 5, max_distance: float = 1.2) -> dict:
    """Raw RAG search with Chroma-compatible return shape.

    Distance is represented as 1 - normalized lexical relevance, so lower is
    better and callers can keep their existing max_distance checks.
    """

    query_tokens = _tokenize(query)
    chunks = _load_knowledge_chunks()
    scored = []

    for chunk in chunks:
        score = _score_chunk(query_tokens, chunk)
        if score <= 0:
            continue
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[:min(max(1, n_results), RAG_TOP_K)]
    if not selected:
        return {"documents": [], "metadatas": [], "distances": []}

    top_score = selected[0][0] or 1.0
    documents, metadatas, distances = [], [], []
    for score, chunk in selected:
        distance = max(0.0, 1.0 - (score / top_score))
        if distance <= max_distance:
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])
            distances.append(distance)

    return {
        "documents": documents,
        "metadatas": metadatas,
        "distances": distances,
    }
