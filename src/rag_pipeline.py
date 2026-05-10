"""
RAG Pipeline Module
Orchestrates the complete RAG workflow: upload → parse → chunk → embed → retrieve → generate
Features session-based handling, efficient querying, and debugging logs.
"""

import os
import logging
import re
from typing import List, Tuple, Optional, Dict, Any

from src.document_loader import DocumentLoader
from src.text_chunker import TextChunker, ChunkValidator
from src.embedding_store import EmbeddingStore
from src.retriever import Retriever
from src.llm_handler import LLMHandler
from src.document_stats import DocumentStats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class RAGPipeline:
    """Complete RAG pipeline orchestration resembling NotebookLM behavior."""
    
    def __init__(self, 
                 persist_dir: str = "./data/vector_store",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 800,
                 chunk_overlap: int = 150):
        
        logger.info("Initializing RAG Pipeline components...")
        self.document_loader = DocumentLoader()
        self.text_chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_store = EmbeddingStore(persist_dir=persist_dir, model_name=embedding_model)
        retrieve_k = int(os.getenv("RETRIEVAL_TOP_K", "8"))
        self.retriever = Retriever(embedding_store=self.embedding_store, default_k=retrieve_k)
        self.llm_handler = LLMHandler()
        # Store document stats for deterministic answers to factual questions
        self.document_stats = {}

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

    def _get_session_doc_names(self, session_id: str) -> List[str]:
        names = []
        prefix = f"{session_id}:"
        for key, stats in self.document_stats.items():
            if key.startswith(prefix):
                names.append(stats.get("doc_name", key[len(prefix):]))
        return sorted(set(names))

    def _match_docs_from_question(self, question: str, doc_names: List[str]) -> List[str]:
        """Loose matching: rank documents by word overlap with the question."""
        q_norm = self._normalize_text(question)
        q_terms = set(t for t in q_norm.split() if len(t) > 2)
        scored: List[Tuple[int, str]] = []

        for name in doc_names:
            n_norm = self._normalize_text(name)
            n_terms = set(t for t in n_norm.split() if len(t) > 2)
            score = 0
            if n_norm and n_norm in q_norm:
                score += 5
            score += len(q_terms.intersection(n_terms))
            scored.append((score, name))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return []

        top_score = scored[0][0]
        if top_score <= 0:
            return []
        return [name for score, name in scored if score == top_score]

    def _looks_single_doc_ambiguous(self, question: str) -> bool:
        q = self._normalize_text(question)
        markers = [
            "this document", "the document", "this file", "the file",
            "this paper", "the paper", "this book", "the book",
        ]
        return any(marker in q for marker in markers)

    def _heuristic_intent_override(self, question: str) -> Optional[str]:
        """Lightweight heuristic to avoid false STAT classification for content questions.

        Examples where 'character' should be CONTENT: "Who is the main character...",
        "How many characters are introduced..." (refers to fictional characters), etc.
        """
        q = (question or "").lower()
        # If the question references fictional characters (main, introduced, who, name), prefer CONTENT
        if 'character' in q or 'characters' in q:
            if any(k in q for k in ['main', 'introduced', 'introduc', 'who', 'name', 'names', 'protagonist']):
                return 'CONTENT'
        return None

    def _clarification_response(self, doc_names: List[str]) -> Dict[str, Any]:
        docs_text = "\n".join(f"- {name}" for name in doc_names)
        return {
            "answer": (
                "I need a target document to answer this precisely. "
                "Please reply with one document name, a list of document names, or 'all'.\n\n"
                "Available documents:\n"
                f"{docs_text}"
            ),
            "sources": [],
            "citations": [],
        }

    @staticmethod
    def _relevance_strength(chunks: List[Dict[str, Any]]) -> float:
        if not chunks:
            return 0.0
        scores = []
        for chunk in chunks:
            try:
                scores.append(float(chunk.get("relevance_score", 0.0)))
            except Exception:
                continue
        return max(scores) if scores else 0.0

    @staticmethod
    def _document_summary_from_history(chat_history: Optional[List[Dict[str, str]]], max_turns: int = 6, max_chars: int = 900) -> str:
        if not chat_history:
            return ""

        pairs: List[Tuple[str, str]] = []
        pending_user: Optional[str] = None

        for message in chat_history:
            role = (message.get("role") or "").lower()
            content = (message.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                pending_user = content
            elif role == "assistant" and pending_user:
                pairs.append((pending_user, content))
                pending_user = None

        if not pairs:
            return ""

        def trim(text: str, limit: int) -> str:
            clean = re.sub(r"\s+", " ", text).strip()
            return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "…"

        lines: List[str] = []
        for user_text, assistant_text in pairs[-max_turns:]:
            lines.append(f"User asked: {trim(user_text, 90)}")
            lines.append(f"Assistant replied: {trim(assistant_text, 120)}")

        summary = "\n".join(lines)
        return summary if len(summary) <= max_chars else summary[-max_chars:]

    def ingest_document(self, file_path: str, session_id: str, original_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Full ingestion flow: upload → parse → chunk → embed → store.
        Configures an isolated collection for the given session.
        """
        logger.info(f"Starting ingestion: file='{file_path}', session='{session_id}'")
        
        # 1. Parse
        try:
            doc_name, content, metadata = self.document_loader.load_document(os.path.abspath(file_path))
            if original_filename:
                # Preserve original upload filename for better source attribution
                metadata["file_name"] = original_filename
                metadata["doc_name"] = os.path.splitext(original_filename)[0]
                metadata["source"] = original_filename
                doc_name = metadata["doc_name"]
        except Exception as e:
            logger.error(f"Failed to parse document: {e}")
            return {"status": "error", "message": f"Parse error: {e}"}

        # 2. Chunk
        chunks = self.text_chunker.chunk_text(content, metadata)
        if not chunks:
            logger.warning("No chunks generated from the document.")
            return {"status": "error", "message": "No text extracted."}
        
        # Validate chunks (optional strictness)
        valid_count, invalid_count, errors = ChunkValidator.validate_chunks(chunks)
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid chunks: {errors}")

        # 3. Embed & Store
        collection_name = f"session_{session_id}"
        try:
            added = self.embedding_store.add_documents(chunks=chunks, collection_name=collection_name)
            logger.info(f"Ingestion complete: {added} chunks stored in collection '{collection_name}'")
            safe_metadata = dict(metadata or {})
            safe_metadata["doc_name"] = doc_name
            
            # Store document stats for this session
            stats_key = f"{session_id}:{doc_name}"
            self.document_stats[stats_key] = {
                'doc_name': doc_name,
                'paragraph_count': metadata.get('paragraph_count', 0),
                'sentence_count': metadata.get('sentence_count', 0),
                'word_count': metadata.get('word_count', 0),
                'character_count': metadata.get('character_count', 0),
                'estimated_pages': metadata.get('estimated_pages', 0),
            }
            logger.info(f"Stored stats for {doc_name}: {self.document_stats[stats_key]}")
            
            return {
                "status": "success",
                "document": doc_name,
                "chunks_added": added,
                "collection": collection_name,
                "metadata": safe_metadata
            }
        except Exception as e:
            logger.error(f"Embedding/Storage error: {e}")
            return {"status": "error", "message": str(e)}

    def answer_query(
        self,
        question: str,
        session_id: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        conversation_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full query flow: question → check for stat-based answer → retrieve from session → augment prompt → generate.
        """
        collection_name = f"session_{session_id}"
        logger.info(f"Processing query: '{question}' for session '{session_id}'")

        session_docs = self._get_session_doc_names(session_id)
        matched_docs = self._match_docs_from_question(question, session_docs)

        # If multiple docs are loaded and the question is singular/ambiguous, ask for clarification.
        if len(session_docs) > 1 and self._looks_single_doc_ambiguous(question) and len(matched_docs) != 1:
            logger.info("Question appears ambiguous across multiple documents; requesting clarification.")
            return self._clarification_response(session_docs)
        
        # 0. Use LLM to decide if this is a stats query, then optionally use deterministic stats.
        session_stats = []
        for stats_key, stats in self.document_stats.items():
            if stats_key.startswith(f"{session_id}:"):
                session_stats.append(stats)

        if session_stats:
            # Allow a fast heuristic to override the LLM when appropriate
            heuristic = self._heuristic_intent_override(question)
            if heuristic:
                intent = heuristic
            else:
                intent = self.llm_handler.classify_query_intent(question)
            logger.info(f"Query intent classified as: {intent}")

            if intent == "STAT":
                stat_candidates = []
                for stats in session_stats:
                    doc_name = stats.get("doc_name", "")
                    if matched_docs and doc_name not in matched_docs:
                        continue
                    stat_answer = DocumentStats.answer_stat_question(question.lower(), stats)
                    if stat_answer:
                        stat_candidates.append((doc_name, stat_answer))

                if len(stat_candidates) == 1:
                    doc_name, stat_answer = stat_candidates[0]
                    logger.info(f"Answered from document stats: {stat_answer}")
                    return {
                        "answer": f"{stat_answer} (source: {doc_name})",
                        "sources": [],
                        "citations": [{"source": doc_name, "page": None}],
                    }

                if len(stat_candidates) > 1:
                    logger.info("Stat question matches multiple documents; requesting clarification.")
                    return self._clarification_response([doc for doc, _ in stat_candidates])
        
        # 1. Retrieve
        try:
            retrieved_chunks = self.retriever.retrieve(query=question, collection_name=collection_name)
            
            if not retrieved_chunks:
                logger.info("No relevant context found in vector store.")
            else:
                logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks.")
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return {"answer": "Error retrieving documents. Please try again.", "sources": [], "error": str(e)}

        retrieval_strength = self._relevance_strength(retrieved_chunks)
        weak_retrieval_threshold = float(os.getenv("WEAK_RETRIEVAL_THRESHOLD", "0.45"))
        allow_general_explanation = retrieval_strength < weak_retrieval_threshold
        if allow_general_explanation:
            logger.info(
                "Retrieval looks weak (strength=%.3f, threshold=%.3f); enabling general explanation mode.",
                retrieval_strength,
                weak_retrieval_threshold,
            )

        summary_text = conversation_summary or self._document_summary_from_history(chat_history)

        # 2. Augment & Generate
        try:
            response = self.llm_handler.answer(
                question=question,
                retrieved_chunks=retrieved_chunks,
                chat_history=chat_history,
                conversation_summary=summary_text,
                allow_general_explanation=allow_general_explanation,
            )

            if allow_general_explanation:
                answer_text = (response.get("answer") or "").strip()
                if answer_text and not answer_text.lower().startswith("general explanation:"):
                    response["answer"] = f"General explanation: {answer_text}"

            logger.info("Successfully generated answer.")
            return response
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {"answer": f"Error generating response: {str(e)}", "sources": [], "error": str(e)}

    def clear_session(self, session_id: str) -> bool:
        """
        Deletes the session-specific collection, clearing out uploaded data for this session.
        """
        collection_name = f"session_{session_id}"
        logger.info(f"Clearing session data for '{session_id}' (Collection: {collection_name})")
        success = self.embedding_store.delete_collection(collection_name)
        
        # Also clear stats for this session
        keys_to_delete = [k for k in self.document_stats.keys() if k.startswith(f"{session_id}:")]
        for key in keys_to_delete:
            del self.document_stats[key]
            logger.info(f"Cleared stats for {key}")
        
        if success:
            logger.info("Session cleared successfully.")
        else:
            logger.warning("Failed to clear session or session did not exist.")
        return success

