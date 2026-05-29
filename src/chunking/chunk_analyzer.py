"""Intrinsic chunk quality metrics.

Evaluates chunks before retrieval using embedding-based measures:
- ICC (Intra-Chunk Coherence)
- ICD (Inter-Chunk Distinctiveness)
- SC (Size Compliance)
- BC (Boundary Clarity)
"""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking.base_chunker import Chunk

logger = logging.getLogger(__name__)


class ChunkAnalyzer:
    """Computes intrinsic quality metrics for chunks."""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model_name)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def intra_chunk_coherence(self, chunk: Chunk) -> float:
        """ICC: Average cosine similarity between consecutive sentences in a chunk.

        High ICC = chunk is internally coherent (good).
        Higher values indicate better internal coherence.
        """
        import re
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk.text) if len(s.strip()) > 5]

        if len(sentences) < 2:
            return 1.0  # single sentence is perfectly coherent

        embeddings = self.model.encode(sentences, show_progress_bar=False)
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        return float(np.mean(similarities))

    def inter_chunk_distinctiveness(self, chunks: list[Chunk]) -> float:
        """ICD: 1 - average cosine similarity between adjacent chunks.

        High ICD = chunks are distinct from each other (good).
        Higher values indicate better internal coherence.
        """
        if len(chunks) < 2:
            return 1.0

        embeddings = self.model.encode(
            [c.text for c in chunks], show_progress_bar=False,
        )
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        return float(1.0 - np.mean(similarities))

    def size_compliance(self, chunks: list[Chunk]) -> float:
        """SC: Coefficient of variation of chunk sizes (lower = more uniform).

        Low SC = more uniform chunk sizes (good for fixed-size, less relevant for semantic).
        Higher values indicate better internal coherence.
        """
        if not chunks:
            return 0.0

        sizes = [c.token_count for c in chunks]
        mean_size = np.mean(sizes)
        if mean_size == 0:
            return 0.0

        return float(np.std(sizes) / mean_size)

    def boundary_clarity(self, chunks: list[Chunk]) -> float:
        """BC: Average semantic discontinuity at chunk boundaries.

        High BC = clear topic shifts at boundaries (good).
        Higher values indicate clearer topic transitions at boundaries.

        Computed as the average cosine distance between the last sentence
        of chunk_i and the first sentence of chunk_{i+1}.
        """
        if len(chunks) < 2:
            return 1.0

        import re
        boundary_distances = []

        for i in range(len(chunks) - 1):
            # Get last sentence of current chunk
            sents_curr = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunks[i].text) if len(s.strip()) > 5]
            # Get first sentence of next chunk
            sents_next = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunks[i + 1].text) if len(s.strip()) > 5]

            if not sents_curr or not sents_next:
                continue

            emb_last = self.model.encode([sents_curr[-1]], show_progress_bar=False)[0]
            emb_first = self.model.encode([sents_next[0]], show_progress_bar=False)[0]

            distance = 1.0 - self._cosine_similarity(emb_last, emb_first)
            boundary_distances.append(distance)

        if not boundary_distances:
            return 0.0

        return float(np.mean(boundary_distances))

    def analyze_chunks(self, chunks: list[Chunk]) -> dict:
        """Compute all intrinsic metrics for a set of chunks.

        Args:
            chunks: List of chunks from a single strategy/document.

        Returns:
            Dictionary with all metrics and per-chunk ICC scores.
        """
        if not chunks:
            return {
                "icc_mean": 0, "icc_std": 0,
                "icd": 0, "sc": 0, "bc": 0,
                "num_chunks": 0, "avg_tokens": 0,
            }

        # Per-chunk ICC
        icc_scores = [self.intra_chunk_coherence(c) for c in chunks]

        result = {
            "icc_mean": float(np.mean(icc_scores)),
            "icc_std": float(np.std(icc_scores)),
            "icc_per_chunk": icc_scores,
            "icd": self.inter_chunk_distinctiveness(chunks),
            "sc": self.size_compliance(chunks),
            "bc": self.boundary_clarity(chunks),
            "num_chunks": len(chunks),
            "avg_tokens": float(np.mean([c.token_count for c in chunks])),
            "total_tokens": sum(c.token_count for c in chunks),
        }

        logger.info(
            f"Chunk analysis: ICC={result['icc_mean']:.4f}, "
            f"ICD={result['icd']:.4f}, SC={result['sc']:.4f}, "
            f"BC={result['bc']:.4f} ({len(chunks)} chunks)"
        )
        return result
