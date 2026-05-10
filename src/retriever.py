"""Retriever module: top-k semantic retrieval using the EmbeddingStore."""

import re
from typing import List, Optional, Dict, Any, Iterable

from .embedding_store import EmbeddingStore


class Retriever:
    """Simple retriever that wraps the EmbeddingStore similarity search.

    Methods:
    - retrieve(query, k, collection_name) -> List[Dict]: returns chunks with metadata
    - build_context(chunks) -> str: assemble a context string suitable for prompting
    """

    def __init__(self, embedding_store: Optional[EmbeddingStore] = None, default_k: int = 5):
        self.store = embedding_store or EmbeddingStore()
        self.default_k = default_k

    def retrieve(self, query: str, k: Optional[int] = None, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return top-k relevant chunks (each with id, content, metadata, distance, relevance_score)."""
        k = k or self.default_k
        results = self.store.similarity_search(query=query, collection_name=collection_name, top_k=k)
        return self._rerank_by_document_hint(query, results)[:k]

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

    @classmethod
    def _query_terms(cls, query: str) -> List[str]:
        normalized = cls._normalize(query)
        return [term for term in normalized.split() if len(term) > 2]

    @classmethod
    def _document_score(cls, query: str, chunk: Dict[str, Any]) -> float:
        meta = chunk.get("metadata", {}) or {}
        doc_name = str(meta.get("doc_name") or meta.get("file_name") or meta.get("source") or "")
        normalized_doc = cls._normalize(doc_name)
        normalized_query = cls._normalize(query)
        terms = cls._query_terms(query)

        score = float(chunk.get("relevance_score", 0.0))

        # Strong boost when the query explicitly mentions the document name.
        if normalized_doc and normalized_doc in normalized_query:
            score += 0.5

        # Bigger boost for overlapping keywords between the query and the document name.
        if normalized_doc:
            for term in terms:
                if term in normalized_doc:
                    score += 0.12

        # Bigger boost when the chunk content itself repeats the query terms.
        content = cls._normalize(str(chunk.get("content") or ""))
        if content:
            overlap = sum(1 for term in terms[:12] if term in content)
            score += min(overlap * 0.05, 0.30)

        return score

    def _rerank_by_document_hint(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return chunks

        return sorted(
            chunks,
            key=lambda chunk: (
                self._document_score(query, chunk),
                float(chunk.get("relevance_score", 0.0)),
            ),
            reverse=True,
        )

    def build_context(self, chunks: List[Dict[str, Any]], max_chars: int = 4000) -> str:
        """Assemble retrieved chunks into a single context string for the LLM.

        Keeps simple source markers (source, page, chunk id) so the LLM can cite.
        """
        pieces = []
        for c in chunks:
            meta = c.get('metadata', {}) or {}
            # Prioritize doc_name (actual filename) over source (temp path)
            source = meta.get('doc_name') or meta.get('source') or meta.get('filename') or meta.get('source_file') or ''
            file_type = meta.get('file_type') or ''
            page = meta.get('page_number') or meta.get('page') or ''
            header = f"Source: {source}" + (f" | Type: {file_type}" if file_type else "") + (f" | Page: {page}" if page else "") + f" | ChunkID: {c.get('id')}"
            pieces.append(header + "\n" + (c.get('content') or "") + "\n")

        context = "\n---\n".join(pieces)
        if len(context) > max_chars:
            context = context[:max_chars]
        return context


__all__ = ["Retriever"]
