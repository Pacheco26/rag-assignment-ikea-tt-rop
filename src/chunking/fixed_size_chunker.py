"""Fixed-size chunking strategy.

Splits text into chunks of a fixed token size with overlap.
Used as the baseline approach for comparison with semantic and hierarchical strategies.
"""

import logging
from typing import Optional

from llama_index.core.node_parser import SentenceSplitter

from config.settings import FIXED_CHUNK_SIZE, FIXED_CHUNK_OVERLAP
from src.chunking.base_chunker import BaseChunker, Chunk

logger = logging.getLogger(__name__)


class FixedSizeChunker(BaseChunker):
    """Fixed-size chunking with sentence-aware boundaries.

    Uses LlamaIndex SentenceSplitter which respects sentence boundaries
    while targeting the specified chunk size.
    """

    def __init__(
        self,
        chunk_size: int = FIXED_CHUNK_SIZE,
        chunk_overlap: int = FIXED_CHUNK_OVERLAP,
    ):
        super().__init__("fixed_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_text(
        self,
        text: str,
        paper_id: str,
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        metadata = metadata or {}
        # Use LlamaIndex splitter to get text splits
        splits = self.splitter.split_text(text)

        chunks = []
        offset = 0
        for i, split_text in enumerate(splits):
            # Find position in original text
            start = text.find(split_text[:50], offset)
            if start == -1:
                start = offset
            end = start + len(split_text)

            chunk = Chunk(
                chunk_id=f"{paper_id}_fixed_{i:04d}",
                text=split_text,
                paper_id=paper_id,
                strategy="fixed_size",
                section=metadata.get("section", ""),
                start_char=start,
                end_char=end,
                token_count=self._estimate_tokens(split_text),
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "chunk_size_setting": self.chunk_size,
                    "chunk_overlap_setting": self.chunk_overlap,
                },
            )
            chunks.append(chunk)
            offset = max(offset, start + 1)

        logger.info(
            f"{paper_id}: Fixed-size chunking -> {len(chunks)} chunks "
            f"(avg {sum(c.token_count for c in chunks) / max(len(chunks), 1):.0f} tokens)"
        )
        return chunks
