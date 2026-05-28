"""Prompt construction for RAG pipeline."""

import logging
from typing import Optional

from config.prompts import SYSTEM_PROMPT, QUERY_PROMPT

logger = logging.getLogger(__name__)


def build_context_string(
    retrieved_chunks: list[dict],
    max_context_chars: int = 16000,
) -> str:
    """Build a formatted context string from retrieved chunks.

    Args:
        retrieved_chunks: List of retrieval results with 'text', 'metadata'.
        max_context_chars: Maximum total context characters (~4000 tokens).
            Prevents exceeding Groq's TPM limits.

    Returns:
        Formatted context string with document citations.
    """
    context_parts = []
    seen_texts = set()
    total_chars = 0

    for i, result in enumerate(retrieved_chunks, 1):
        text = result["text"]

        # Deduplicate (especially important for hierarchical where
        # multiple children might map to same parent)
        text_key = text[:200]
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)

        metadata = result.get("metadata", {})
        paper_id = metadata.get("paper_id", "Unknown")
        section = metadata.get("section", "")

        # Format context chunk with source attribution
        header = f"[Source {i}: {paper_id}"
        if section:
            header += f", Section: {section}"
        header += "]"

        chunk_text = f"{header}\n{text}"

        # Truncate if adding this chunk would exceed the limit
        if total_chars + len(chunk_text) > max_context_chars:
            remaining = max_context_chars - total_chars
            if remaining > 200:
                chunk_text = chunk_text[:remaining] + "\n[...truncated]"
                context_parts.append(chunk_text)
            break

        context_parts.append(chunk_text)
        total_chars += len(chunk_text) + 10  # account for separator

    return "\n\n---\n\n".join(context_parts)


def build_rag_prompt(
    query: str,
    retrieved_chunks: list[dict],
    system_prompt: Optional[str] = None,
) -> list[dict]:
    """Build the chat messages list for the RAG pipeline."""
    context = build_context_string(retrieved_chunks)

    user_message = QUERY_PROMPT.format(
        context=context,
        query=query,
    )

    messages = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    logger.info(
        f"Built RAG prompt: {len(retrieved_chunks)} chunks, "
        f"{len(context)} chars context"
    )
    return messages


def build_context_with_metadata(
    retrieved_chunks: list[dict],
    include_scores: bool = False,
) -> str:
    """Build enriched context with retrieval metadata (for debugging/eval)."""
    parts = []
    for i, result in enumerate(retrieved_chunks, 1):
        meta = result.get("metadata", {})
        header_parts = [
            f"[{i}]",
            f"Document: {meta.get('paper_id', '?')}",
        ]
        if meta.get("section"):
            header_parts.append(f"Section: {meta['section']}")
        if include_scores:
            if "rrf_score" in result:
                header_parts.append(f"RRF: {result['rrf_score']:.4f}")
            if "distance" in result:
                header_parts.append(f"Dist: {result['distance']:.4f}")

        header = " | ".join(header_parts)
        parts.append(f"{header}\n{result['text']}")

    return f"\n\n{'='*60}\n\n".join(parts)
