"""Semantic chunking strategy.

Splits text at points where consecutive sentence embeddings
show a large cosine distance, indicating a topic shift.
"""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import (
    SEMANTIC_BREAKPOINT_PERCENTILE,
    SEMANTIC_BUFFER_SIZE,
    EMBEDDING_MODEL,
)
from src.chunking.base_chunker import BaseChunker, Chunk

logger = logging.getLogger(__name__)


class SemanticChunker(BaseChunker):
    """Semantic chunking based on embedding similarity breakpoints.

    Splits text at points where consecutive sentence embeddings show
    the largest cosine distance (semantic shift).
    """

    def __init__(
        self,
        breakpoint_percentile: int = SEMANTIC_BREAKPOINT_PERCENTILE,
        buffer_size: int = SEMANTIC_BUFFER_SIZE,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        super().__init__("semantic")
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re
        # Split on sentence-ending punctuation followed by space/newline
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter empty strings and very short fragments
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _compute_breakpoints(self, sentences: list[str]) -> list[int]:
        """Find semantic breakpoints between sentences.

        Returns indices where chunks should be split.
        """
        if len(sentences) <= 1:
            return []

        # Embed all sentences
        embeddings = self.embedding_model.encode(sentences, show_progress_bar=False)

        # Compute cosine distances between consecutive sentences
        distances = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            distances.append(1 - sim)

        if not distances:
            return []

        # Find breakpoints where distance exceeds the percentile threshold
        threshold = np.percentile(distances, self.breakpoint_percentile)
        breakpoints = [
            i + 1 for i, d in enumerate(distances)
            if d >= threshold
        ]

        return breakpoints

    def chunk_text(
        self,
        text: str,
        paper_id: str,
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        metadata = metadata or {}
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        breakpoints = self._compute_breakpoints(sentences)

        # Build chunks from breakpoints
        chunks = []
        start_idx = 0
        all_break_positions = sorted(set([0] + breakpoints + [len(sentences)]))

        for i in range(len(all_break_positions) - 1):
            chunk_start = all_break_positions[i]
            chunk_end = all_break_positions[i + 1]

            # Apply buffer: include adjacent sentences for context
            buf_start = max(0, chunk_start - self.buffer_size)
            buf_end = min(len(sentences), chunk_end + self.buffer_size)

            chunk_text = " ".join(sentences[buf_start:buf_end])

            if not chunk_text.strip():
                continue

            chunk = Chunk(
                chunk_id=f"{paper_id}_semantic_{len(chunks):04d}",
                text=chunk_text,
                paper_id=paper_id,
                strategy="semantic",
                section=metadata.get("section", ""),
                token_count=self._estimate_tokens(chunk_text),
                metadata={
                    **metadata,
                    "chunk_index": len(chunks),
                    "sentence_range": (chunk_start, chunk_end),
                    "breakpoint_percentile": self.breakpoint_percentile,
                },
            )
            chunks.append(chunk)

        logger.info(
            f"{paper_id}: Semantic chunking -> {len(chunks)} chunks "
            f"(avg {sum(c.token_count for c in chunks) / max(len(chunks), 1):.0f} tokens)"
        )
        return chunks
