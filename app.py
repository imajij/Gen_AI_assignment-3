"""
NotebookLM-Style RAG Application
Streamlit-based interface for Retrieval Augmented Generation with Collection Management
"""

import os
import re
import time
import streamlit as st
import tempfile
import uuid
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

from src.rag_pipeline import RAGPipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NotebookLM RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root tokens ── */
:root {
    --bg:        #0f0f11;
    --surface:   #18181c;
    --surface2:  #222228;
    --border:    #2e2e38;
    --accent:    #7c6af7;
    --accent2:   #a78bfa;
    --text:      #e8e8f0;
    --muted:     #8888a0;
    --success:   #4ade80;
    --warning:   #fbbf24;
    --danger:    #f87171;
    --radius:    12px;
    --radius-sm: 8px;
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

.stApp {
    background: var(--bg);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'DM Serif Display', serif;
    color: var(--text);
}

[data-testid="stSidebar"] .stCaption {
    color: var(--muted);
    font-size: 0.72rem;
    font-family: 'DM Sans', monospace;
    letter-spacing: 0.04em;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all 0.18s ease !important;
    padding: 0.45rem 0.9rem !important;
}

.stButton > button:hover {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #fff !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(124,106,247,0.35) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface2);
    border: 1.5px dashed var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent);
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 0 !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"] .stMarkdown,
.user-bubble {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius) var(--radius) 4px var(--radius);
    padding: 0.85rem 1.1rem;
    display: inline-block;
    max-width: 80%;
}

/* Assistant bubble */
[data-testid="stChatMessage"] .stMarkdown {
    line-height: 1.7;
}

/* ── Expanders (Sources / Thought) ── */
[data-testid="stExpander"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 0.5rem !important;
}

[data-testid="stExpander"] summary {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

[data-testid="stExpander"] summary:hover {
    color: var(--accent2) !important;
}

/* Source items inside expander */
.source-item {
    padding: 0.6rem 0.8rem;
    border-radius: var(--radius-sm);
    border-left: 3px solid var(--accent);
    background: rgba(124,106,247,0.07);
    margin-bottom: 0.5rem;
    font-size: 0.84rem;
}

.source-item .source-name {
    font-weight: 600;
    color: var(--accent2);
    margin-bottom: 0.2rem;
}

.source-item .source-score {
    font-size: 0.72rem;
    color: var(--muted);
    margin-bottom: 0.3rem;
}

.source-item .source-excerpt {
    color: var(--muted);
    font-style: italic;
    font-size: 0.80rem;
    line-height: 1.5;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,106,247,0.2) !important;
}

/* ── Info / Success / Error alerts ── */
.stAlert {
    border-radius: var(--radius) !important;
    font-size: 0.84rem !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: var(--accent) !important;
}

/* ── Dividers ── */
hr {
    border-color: var(--border) !important;
    margin: 1rem 0 !important;
}

/* ── Page title ── */
.rag-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--text);
    margin-bottom: 0;
    line-height: 1.2;
}
.rag-subtitle {
    color: var(--muted);
    font-size: 0.85rem;
    margin-top: 0.25rem;
    margin-bottom: 1.5rem;
}

/* ── Doc metadata expanders ── */
.doc-meta {
    font-size: 0.75rem;
    color: var(--muted);
}

/* ── Thinking animation ── */
@keyframes pulse-dot {
    0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
    40%           { opacity: 1;   transform: scale(1.0); }
}
.thinking-dots span {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent2);
    margin: 0 2px;
    animation: pulse-dot 1.2s infinite ease-in-out;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }

.thinking-label {
    font-size: 0.78rem;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0.4rem 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar       { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "pipeline" not in st.session_state:
    try:
        st.session_state.pipeline = RAGPipeline(
            persist_dir=os.getenv("VECTOR_STORE_PATH", "./data/vector_store"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
    except Exception as e:
        st.error(f"Error initialising pipeline: {str(e)}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

if "last_failed_prompt" not in st.session_state:
    st.session_state.last_failed_prompt = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_compact_conversation_summary(messages, max_pairs: int = 4, max_chars: int = 900) -> str:
    pairs, pending_user = [], None
    for message in messages:
        role = message.get("role", "").lower()
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
        compact = " ".join(text.split())
        return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"

    lines = []
    for user_text, assistant_text in pairs[-max_pairs:]:
        lines.append(f"User: {trim(user_text, 90)}")
        lines.append(f"Assistant: {trim(assistant_text, 120)}")

    summary = "\n".join(lines)
    return summary if len(summary) <= max_chars else summary[-max_chars:]


def split_thought_block(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    match = re.search(r"<thought>(.*?)</thought>", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return "", text.strip()
    thought = match.group(1).strip()
    cleaned = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    return thought, cleaned


def stream_text(text: str, delay: float = 0.008):
    """Yield text in small chunks for a typing effect."""
    if not text:
        return
    chunk_size = 20
    for i in range(0, len(text), chunk_size):
        yield text[i: i + chunk_size]
        time.sleep(delay)


def render_sources(sources: list, expanded: bool = False):
    """Render source documents in a styled expander."""
    if not sources:
        return
    with st.expander(f"📚  Sources  ·  {len(sources)} chunks", expanded=expanded):
        for idx, doc in enumerate(sources, 1):
            source_name = "Unknown"
            if "metadata" in doc:
                source_name = (
                    doc["metadata"].get("doc_name")
                    or doc["metadata"].get("file_name")
                    or doc["metadata"].get("source")
                    or "Unknown"
                )
            else:
                source_name = doc.get("source", "Unknown")
            if isinstance(source_name, str):
                source_name = os.path.basename(source_name)

            score = doc.get("relevance_score")
            score_str = f"Score: {score:.2f}" if score is not None else (
                f"Page {doc['page']}" if "page" in doc else ""
            )
            excerpt = doc.get("content", "")[:240] + ("…" if len(doc.get("content", "")) > 240 else "")

            st.markdown(f"""
            <div class="source-item">
                <div class="source-name">[{idx}] {source_name}</div>
                {"<div class='source-score'>" + score_str + "</div>" if score_str else ""}
                {"<div class='source-excerpt'>" + excerpt + "</div>" if excerpt else ""}
            </div>
            """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def display_sidebar():
    with st.sidebar:
        st.markdown("# 📚 Notebook")
        st.caption(f"Session · {st.session_state.session_id[:8]}…")
        st.divider()

        # Upload
        st.markdown("### Upload Documents")
        uploaded_files = st.file_uploader(
            "PDF · TXT · DOCX · MD · HTML",
            type=["pdf", "txt", "docx", "md", "html"],
            accept_multiple_files=True,
            label_visibility="visible",
        )

        if uploaded_files:
            if st.button("✨  Process Documents", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    try:
                        temp_path = None
                        try:
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=Path(uploaded_file.name).suffix
                            ) as tmp:
                                tmp.write(uploaded_file.read())
                                temp_path = tmp.name
                            result = st.session_state.pipeline.ingest_document(
                                file_path=temp_path,
                                session_id=st.session_state.session_id,
                                original_filename=uploaded_file.name,
                            )
                            if result.get("status") == "success":
                                st.session_state.uploaded_docs.append(result.get("metadata", {}))
                                st.success(f"✓ {result.get('document')} — {result.get('chunks_added')} chunks")
                            else:
                                st.error(f"Error: {uploaded_file.name}: {result.get('message')}")
                        finally:
                            if temp_path and os.path.exists(temp_path):
                                os.remove(temp_path)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                st.session_state.documents_loaded = True
                status_text.empty()
                progress_bar.empty()
                st.info("All documents processed!")

        st.divider()

        # Document metadata
        st.markdown("### Loaded Documents")
        if st.session_state.uploaded_docs:
            for idx, meta in enumerate(st.session_state.uploaded_docs, 1):
                name = meta.get("file_name") or meta.get("doc_name") or f"Document {idx}"
                with st.expander(f"{idx}. {name}"):
                    st.markdown(
                        "<div class='doc-meta'>" +
                        "<br>".join(f"<b>{k}</b>: {v}" for k, v in meta.items()) +
                        "</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("No documents loaded yet.")

        st.divider()

        # Actions
        st.markdown("### Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Sources", use_container_width=True):
                if st.session_state.pipeline.clear_session(st.session_state.session_id):
                    st.session_state.documents_loaded = False
                    st.session_state.uploaded_docs = []
                    st.success("Sources cleared!")
                    st.rerun()
                else:
                    st.warning("Nothing to clear.")
        with col2:
            if st.button("🔄 Reset Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        st.divider()
        st.markdown("""
        <div style="font-size:0.75rem; color: var(--muted); line-height:1.7">
        <b style="color:var(--accent2)">NotebookLM RAG v2.0</b><br>
        ChromaDB · SentenceTransformers<br>
        OpenAI-compatible API · Session isolation
        </div>
        """, unsafe_allow_html=True)


# ── Chat interface ────────────────────────────────────────────────────────────
def display_chat_interface():
    st.markdown('<p class="rag-title">Chat with your Notebook</p>', unsafe_allow_html=True)
    st.markdown('<p class="rag-subtitle">Ask questions about your uploaded documents</p>', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.info("📂 Upload and process documents in the sidebar, then ask away.")

    # ── Render history ────────────────────────────────────────────────────────
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"), st.container():
                render_sources(message.get("sources", []))
                if message.get("thought"):
                    with st.expander("💭  Thought process", expanded=False):
                        st.markdown(message["thought"])
                st.markdown(message["content"])
                st.empty()  # ghost fence per history message

    # ── Retry banner ──────────────────────────────────────────────────────────
    prompt = None
    if st.session_state.last_failed_prompt:
        st.warning(f"⚠️ Last request failed: *\"{st.session_state.last_failed_prompt[:80]}\"*")
        col1, col2 = st.columns([1, 6])
        with col1:
            if st.button("🔄 Retry", use_container_width=True):
                prompt = st.session_state.last_failed_prompt
                st.session_state.last_failed_prompt = None
        with col2:
            if st.button("✖ Dismiss", use_container_width=True):
                st.session_state.last_failed_prompt = None
                st.rerun()

    # ── Input — top-level so it keeps its sticky-bottom position ─────────────
    chat_prompt = st.chat_input(
        "Ask a question about your documents…",
        disabled=not st.session_state.documents_loaded,
    )
    if chat_prompt:
        prompt = chat_prompt

    if not prompt:
        return
    if not st.session_state.documents_loaded:
        st.warning("Please upload and process documents first.")
        return

    # ── Render new user message ───────────────────────────────────────────────
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # ── Generate & render response ────────────────────────────────────────────
    answer_text, thought_text, retrieved_chunks = None, "", []

    with st.chat_message("assistant", avatar="🤖"), st.container():
        result, error_msg = None, None

        # st.spinner instead of st.empty() placeholder — avoids ghost trigger
        with st.spinner("Retrieving & generating…"):
            try:
                result = st.session_state.pipeline.answer_query(
                    question=prompt,
                    session_id=st.session_state.session_id,
                    chat_history=st.session_state.messages[-8:],
                    conversation_summary=build_compact_conversation_summary(
                        st.session_state.messages[:-8]
                    ),
                )
            except Exception as e:
                error_msg = str(e)

        if error_msg:
            st.error(f"❌ Error: {error_msg}")
            st.caption("Click **Retry** above to try again, or rephrase your question.")
            st.session_state.last_failed_prompt = prompt
            st.rerun()
            return

        retrieved_chunks = result.get("sources", [])
        raw_answer = result.get("answer", "No answer returned.")
        thought_text, answer_text = split_thought_block(raw_answer)

        render_sources(retrieved_chunks, expanded=False)

        if thought_text:
            with st.expander("💭  Thought process", expanded=False):
                st.markdown(thought_text)

        if hasattr(st, "write_stream"):
            st.write_stream(stream_text(answer_text))
            st.empty()  # ghost fence after write_stream
        else:
            st.markdown(answer_text)

    # ── Persist AFTER rendering ───────────────────────────────────────────────
    if answer_text is not None:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "thought": thought_text,
            "sources": retrieved_chunks,
        })

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    display_sidebar()
    display_chat_interface()


if __name__ == "__main__":
    main()