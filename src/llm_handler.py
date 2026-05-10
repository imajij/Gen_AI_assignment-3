"""LLM handler using an OpenAI-compatible API (supports Google Gen-LM OpenAI-compatible endpoint).

Behavior:
- Enforces strict grounding: "Answer ONLY from provided context."
- If answer not present, returns the exact message: "This information is not present in the uploaded document."
- Includes source citations extracted from chunk metadata.
"""

import os
import requests
from typing import List, Dict, Any, Optional


class LLMHandler:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("API key not found. Set API_KEY or OPENAI_API_KEY in env.")

        default_base_url = os.getenv("BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        default_model = os.getenv("MODEL_NAME") or os.getenv("MODEL", "gemma-4-26b-a4b-it")

        self.base_url = (base_url or default_base_url).rstrip("/")
        self.model = model or default_model
        # OpenAI-compatible chat completion path
        self._endpoint = f"{self.base_url}/chat/completions"

    def _system_instructions(self) -> str:
        return (
            "You are a helpful notebook assistant. Use the uploaded documents and the conversation history. "
            "Rules:\n"
            "1. For document questions, prefer the uploaded sources and cite them.\n"
            "2. You may synthesize or paraphrase when the answer is clearly supported by the text.\n"
            "3. If the question is a follow-up that depends on earlier chat turns, use that history for context.\n"
            "4. If the answer is not in the documents and not supported by the chat history, you may still give a brief general explanation when safe, but say when you are going beyond the documents.\n"
            "5. Cite sources in parentheses: (source: [filename], page: [page#]).\n"
            "6. Be concise, but do not be unnaturally terse; give the smallest useful explanation.\n"
            "Do NOT invent unsupported document facts."
        )

    def _general_response_instructions(self) -> str:
        return (
            "The retrieved document context appears weak or unrelated. "
            "Provide a short general explanation that helps the user, even if it is not directly grounded in the uploaded documents. "
            "Start the answer with 'General explanation:' and keep it to 1-3 sentences. "
            "If you do use any retrieved document context, make that explicit."
        )

    @staticmethod
    def _format_chat_history(chat_history: Optional[List[Dict[str, str]]], max_turns: int = 8) -> str:
        if not chat_history:
            return ""

        lines: List[str] = []
        for message in chat_history[-max_turns:]:
            role = (message.get("role") or "user").lower()
            content = (message.get("content") or "").strip()
            if not content:
                continue
            label = "User" if role == "user" else "Assistant" if role == "assistant" else role.title()
            lines.append(f"{label}: {content}")
        return "\n".join(lines)

    def _build_messages(self, question: str, context: str, chat_history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        system = self._system_instructions()
        history_text = self._format_chat_history(chat_history)
        history_block = f"Conversation history:\n{history_text}\n\n" if history_text else ""
        user = (
            history_block +
            "Context:\n" + context + "\n\n" +
            "Question: " + question + "\n\nAnswer concisely. Include citations to the sources in the context."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _build_general_messages(
        self,
        question: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        system = self._system_instructions() + "\n\n" + self._general_response_instructions()
        history_text = self._format_chat_history(chat_history)
        history_block = f"Conversation history:\n{history_text}\n\n" if history_text else ""
        user = (
            history_block +
            "Context:\n" + context + "\n\n" +
            "Question: " + question + "\n\nAnswer naturally and label the response as a general explanation."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def classify_query_intent(self, question: str) -> str:
        """Classify question intent as 'STAT' or 'CONTENT'.

        STAT: document-level metrics (word count, paragraph count, line count, page count, character count).
        CONTENT: semantic/document meaning questions (who/what/why/describe/story/entities).
        """
        system = (
            "Classify the user's question into one label only: STAT or CONTENT. "
            "Use STAT only for document-level metrics/counts/length questions such as word count, "
            "paragraph count, line count, page count, character count. "
            "Use CONTENT for semantic questions about entities, meaning, plot, definitions, or explanations. "
            "If unsure, choose CONTENT. Return only STAT or CONTENT."
        )
        user = (
            "Question: " + question + "\n\n"
            "Examples:\n"
            "- How many paragraphs are in this document? -> STAT\n"
            "- What is the character count? -> STAT\n"
            "- Who is the main character in Alice in Wonderland? -> CONTENT\n"
            "- How many characters are introduced in Alice in Wonderland? -> CONTENT"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": 8,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        try:
            resp = requests.post(self._endpoint, json=payload, headers=headers, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            if choices:
                msg = (choices[0].get("message") or {}).get("content", "")
                label = (msg or "").strip().upper()
                if "STAT" in label:
                    return "STAT"
        except Exception:
            # Safe fallback: avoid false stat shortcuts.
            return "CONTENT"

        return "CONTENT"

    def answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        chat_history: Optional[List[Dict[str, str]]] = None,
        conversation_summary: Optional[str] = None,
        allow_general_explanation: bool = False,
    ) -> Dict[str, Any]:
        """Return a grounded answer using retrieved chunks.

        Returns dict: {answer: str, sources: List[Dict], raw: Dict}
        """
        # Handle empty retrieval only when there is no conversational context to lean on.
        if not retrieved_chunks and not chat_history and not conversation_summary and not allow_general_explanation:
            return {
                "answer": "This information is not present in the uploaded document.",
                "sources": [],
                "citations": [],
                "raw": {}
            }

        if allow_general_explanation and not retrieved_chunks:
            context = "No strong document match was retrieved. Use the conversation context and provide a general explanation."
        else:
            context = ""

        # Build context string (ordered, small summary style)
        source_lines = []
        seen_sources = set()
        parts = []
        for c in retrieved_chunks:
            meta = c.get("metadata", {}) or {}
            # Prioritize doc_name (actual filename) over source (temp path)
            source = meta.get("doc_name") or meta.get("source") or meta.get("filename") or meta.get("source_file") or "unknown"
            file_type = meta.get("file_type") or "unknown"
            page = meta.get("page_number") or meta.get("page") or ""
            if source not in seen_sources:
                seen_sources.add(source)
                source_lines.append(f"- {source} ({file_type})")
            header = f"Source: {source}" + (f" | Page: {page}" if page else "") + f" | ChunkID: {c.get('id')}"
            parts.append(header + "\n" + (c.get("content") or ""))

        source_summary = "\n".join(source_lines)
        retrieved_context = "\n\n---\n\n".join(parts)
        if source_summary:
            retrieved_context = "Available sources:\n" + source_summary + "\n\nRetrieved context:\n" + retrieved_context

        summary_block = f"Conversation summary:\n{conversation_summary}\n\n" if conversation_summary else ""
        context = summary_block + (retrieved_context if retrieved_context else context)

        if allow_general_explanation:
            messages = self._build_general_messages(question, context, chat_history=chat_history)
        else:
            messages = self._build_messages(question, context, chat_history=chat_history)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        try:
            # Increased timeout to handle large context (60 seconds)
            resp = requests.post(self._endpoint, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.Timeout:
            raise RuntimeError("LLM request timeout (>60s): Context may be too large or API is slow. Try reducing document size or number of uploads.")
        except requests.RequestException as e:
            detail = ""
            response = getattr(e, "response", None)
            if response is not None:
                body = response.text.strip()
                if body:
                    # Keep the body short so it remains readable in the UI/logs.
                    detail = f" | Response body: {body[:500]}"
            debug = f"model={self.model}, endpoint={self._endpoint}, temperature={temperature}, max_tokens={max_tokens}"
            raise RuntimeError(f"LLM request failed: {e} | {debug}{detail}")

        # Try to read assistant content from OpenAI-compatible response
        answer_text = ""
        try:
            choices = data.get("choices") or []
            if choices:
                # prefer message.content if available
                first = choices[0]
                msg = first.get("message") or {}
                answer_text = msg.get("content") or first.get("text") or ""
        except Exception:
            answer_text = ""

        # Extract simple citations list from retrieved_chunks metadata
        seen = set()
        citations = []
        for c in retrieved_chunks:
            meta = c.get("metadata", {}) or {}
            # Prioritize doc_name (actual filename) over source (temp path)
            src = meta.get("doc_name") or meta.get("source") or meta.get("filename") or meta.get("source_file") or "unknown"
            page = meta.get("page_number") or meta.get("page") or None
            key = (src, page)
            if key not in seen:
                seen.add(key)
                citations.append({"source": src, "page": page})

        return {
            "answer": answer_text.strip(),
            "sources": retrieved_chunks,
            "citations": citations,
            "raw": data
        }


__all__ = ["LLMHandler"]
