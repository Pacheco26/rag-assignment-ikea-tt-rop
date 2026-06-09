"""Hierarchical parent-child chunking strategy.

Creates two levels of chunks: small children for precise retrieval
matching, and larger parents for richer generation context.
"""

import logging
from typing import Optional

from llama_index.core.node_parser import SentenceSplitter

from config.settings import HIERARCHICAL_CHILD_SIZE, HIERARCHICAL_PARENT_SIZE
from src.chunking.base_chunker import BaseChunker, Chunk

logger = logging.getLogger(__name__)


class HierarchicalChunker(BaseChunker):
    """Hierarchical parent-child chunking.

    Creates two levels:
    - Parent chunks (1024 tokens): used as generation context
    - Child chunks (128 tokens): used for retrieval matching

    Children reference their parent so the retriever can
    match on child (precise) -> return parent (contextual).
    """

    def __init__(
        self,
        child_size: int = HIERARCHICAL_CHILD_SIZE,
        parent_size: int = HIERARCHICAL_PARENT_SIZE,
        child_overlap: int = 20,
        parent_overlap: int = 100,
    ):
        super().__init__("hierarchical")
        self.child_size = child_size
        self.parent_size = parent_size

        self.parent_splitter = SentenceSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
        )
        self.child_splitter = SentenceSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
        )

    def chunk_text(
        self,
        text: str,
        paper_id: str,
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        metadata = metadata or {}

        parent_texts = self.parent_splitter.split_text(text)

        all_chunks = []
        for p_idx, parent_text in enumerate(parent_texts):
            parent_id = f"{paper_id}_hier_parent_{p_idx:04d}"

            # split each parent into children
            child_texts = self.child_splitter.split_text(parent_text)

            for c_idx, child_text in enumerate(child_texts):
                child_chunk = Chunk(
                    chunk_id=f"{paper_id}_hier_child_{p_idx:04d}_{c_idx:04d}",
                    text=child_text,
                    paper_id=paper_id,
                    strategy="hierarchical",
                    section=metadata.get("section", ""),
                    token_count=self._estimate_tokens(child_text),
                    parent_id=parent_id,
                    parent_text=parent_text,
                    metadata={
                        **metadata,
                        "parent_index": p_idx,
                        "child_index": c_idx,
                        "child_size_setting": self.child_size,
                        "parent_size_setting": self.parent_size,
                        "is_child": True,
                    },
                )
                all_chunks.append(child_chunk)

        num_parents = len(parent_texts)
        num_children = len(all_chunks)
        avg_children_per_parent = num_children / max(num_parents, 1)

        logger.info(
            f"{paper_id}: Hierarchical chunking -> {num_parents} parents, "
            f"{num_children} children "
            f"(avg {avg_children_per_parent:.1f} children/parent, "
            f"avg child {sum(c.token_count for c in all_chunks) / max(num_children, 1):.0f} tokens)"
        )
        return all_chunks
