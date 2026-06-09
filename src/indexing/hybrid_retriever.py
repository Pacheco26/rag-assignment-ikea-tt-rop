"""Hybrid retriever combining dense (ChromaDB) + sparse (BM25) via RRF.

For the hierarchical strategy, retrieval matches on children
but returns parent text for richer generation context.
"""

import logging
from typing import Optional

from config.settings import TOP_K, HYBRID_ALPHA, RRF_K
from src.chunking.base_chunker import Chunk
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Index

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever using Reciprocal Rank Fusion (dense + BM25)."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        strategy: str,
        alpha: float = HYBRID_ALPHA,
        rrf_k: int = RRF_K,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.strategy = strategy
        self.alpha = alpha
        self.rrf_k = rrf_k

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[dict],
        bm25_results: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Combine dense and BM25 results using RRF.

        RRF score = alpha * 1/(k + rank_dense) + (1-alpha) * 1/(k + rank_bm25)
        """
        scores = {}
        for rank, result in enumerate(dense_results, 1):
            doc_id = result["id"]
            rrf_score = self.alpha * (1.0 / (self.rrf_k + rank))
            scores[doc_id] = {
                "rrf_score": rrf_score,
                "text": result["text"],
                "metadata": result["metadata"],
                "dense_rank": rank,
                "dense_distance": result.get("distance", 0),
                "bm25_rank": None,
                "bm25_score": 0,
            }

        for result in bm25_results:
            chunk = result["chunk"]
            doc_id = chunk.chunk_id
            rank = result["rank"]
            bm25_rrf = (1 - self.alpha) * (1.0 / (self.rrf_k + rank))

            if doc_id in scores:
                scores[doc_id]["rrf_score"] += bm25_rrf
                scores[doc_id]["bm25_rank"] = rank
                scores[doc_id]["bm25_score"] = result["score"]
                # Transfer parent info from BM25 chunk if not already set
                if chunk.parent_id and "parent_text" not in scores[doc_id]:
                    scores[doc_id]["metadata"]["parent_id"] = chunk.parent_id
                    scores[doc_id]["parent_text"] = chunk.parent_text
            else:
                scores[doc_id] = {
                    "rrf_score": bm25_rrf,
                    "text": chunk.text,
                    "metadata": {
                        "paper_id": chunk.paper_id,
                        "strategy": chunk.strategy,
                        "section": chunk.section,
                        "token_count": chunk.token_count,
                    },
                    "dense_rank": None,
                    "dense_distance": None,
                    "bm25_rank": rank,
                    "bm25_score": result["score"],
                }
                if chunk.parent_id:
                    scores[doc_id]["metadata"]["parent_id"] = chunk.parent_id
                    scores[doc_id]["parent_text"] = chunk.parent_text

        sorted_results = sorted(
            scores.items(),
            key=lambda x: x[1]["rrf_score"],
            reverse=True,
        )

        return [
            {"id": doc_id, **data}
            for doc_id, data in sorted_results[:top_k]
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        dense_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
    ) -> list[dict]:
        """Retrieve chunks using dense + BM25 with RRF fusion.

        For hierarchical, matches on children but returns parent text.
        """
        dense_top_k = dense_top_k or (top_k * 2)
        bm25_top_k = bm25_top_k or (top_k * 2)

        dense_results = self.vector_store.query(
            query_text=query,
            strategy=self.strategy,
            top_k=dense_top_k,
        )

        bm25_results = self.bm25_index.query(query, top_k=bm25_top_k)
        fused = self._reciprocal_rank_fusion(dense_results, bm25_results, top_k)

        # For hierarchical: swap child text with parent text for generation
        if self.strategy == "hierarchical":
            for result in fused:
                parent_text = result.get("parent_text")
                if parent_text:
                    result["child_text"] = result["text"]
                    result["text"] = parent_text

        logger.info(
            f"Hybrid retrieval ({self.strategy}): "
            f"{len(dense_results)} dense + {len(bm25_results)} BM25 "
            f"-> {len(fused)} results (alpha={self.alpha})"
        )
        return fused

    def retrieve_dense_only(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Dense-only retrieval (for comparison/ablation)."""
        return self.vector_store.query(
            query_text=query,
            strategy=self.strategy,
            top_k=top_k,
        )

    def retrieve_bm25_only(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """BM25-only retrieval (for comparison/ablation)."""
        results = self.bm25_index.query(query, top_k=top_k)
        return [
            {
                "id": r["chunk"].chunk_id,
                "text": r["chunk"].text,
                "metadata": {
                    "paper_id": r["chunk"].paper_id,
                    "section": r["chunk"].section,
                },
                "bm25_score": r["score"],
            }
            for r in results
        ]
