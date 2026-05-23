"""Abstract base class for all chunking strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """Represents a single text chunk with metadata."""
    chunk_id: str
    text: str
    paper_id: str
    strategy: str
    section: str = ""
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)
    # For hierarchical: reference to parent chunk
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    @abstractmethod
    def chunk_text(self, text: str, paper_id: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: The text to chunk.
            paper_id: Identifier for the source document.
            metadata: Optional metadata to attach to chunks.

        Returns:
            List of Chunk objects.
        """
        ...

    def chunk_paper(self, paper_data: dict) -> list[Chunk]:
        """Chunk an entire document (with section awareness).

        Args:
            paper_data: Dictionary with 'paper_id', 'full_text',
                       and optionally 'sections' keys.

        Returns:
            List of Chunk objects.
        """
        paper_id = paper_data["paper_id"]
        metadata = {
            "filename": paper_data.get("filename", ""),
        }

        # If sections are available, chunk per section for better metadata
        if "sections" in paper_data and paper_data["sections"]:
            all_chunks = []
            for section in paper_data["sections"]:
                section_meta = {**metadata, "section": section["section"]}
                chunks = self.chunk_text(
                    section["text"], paper_id, section_meta,
                )
                for chunk in chunks:
                    chunk.section = section["section"]
                all_chunks.extend(chunks)
            # Re-number chunk IDs to be globally unique across all sections
            if self.strategy_name == "hierarchical":
                # For hierarchical, re-map parent IDs and child IDs
                parent_id_map = {}
                parent_counter = 0
                for chunk in all_chunks:
                    if chunk.parent_id and chunk.parent_id not in parent_id_map:
                        new_parent_id = f"{paper_id}_hier_parent_{parent_counter:04d}"
                        parent_id_map[chunk.parent_id] = new_parent_id
                        parent_counter += 1
                for i, chunk in enumerate(all_chunks):
                    chunk.chunk_id = f"{paper_id}_hier_child_{i:04d}"
                    if chunk.parent_id:
                        chunk.parent_id = parent_id_map[chunk.parent_id]
                    chunk.metadata["chunk_index"] = i
            else:
                strategy_tag = "fixed" if "fixed" in self.strategy_name else self.strategy_name
                for i, chunk in enumerate(all_chunks):
                    chunk.chunk_id = f"{paper_id}_{strategy_tag}_{i:04d}"
                    chunk.metadata["chunk_index"] = i
            return all_chunks

        return self.chunk_text(paper_data["full_text"], paper_id, metadata)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)
