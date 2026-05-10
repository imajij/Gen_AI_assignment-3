"""Prompt templates and helpers for grounded RAG interactions."""


def grounded_system_prompt() -> str:
    return (
        "You are a helpful assistant. Answer ONLY from the provided context. "
        "If the answer is not contained in the provided context, reply exactly: "
        "This information is not present in the uploaded document. "
        "Do not invent facts or hallucinate. Provide brief, friendly answers and include source citations."
    )


def user_prompt(question: str, context: str) -> str:
    return (
        "Context:\n" + context + "\n\n"
        "Question: " + question + "\n\n"
        "Answer concisely. Include citations to the sources in the context."
    )
