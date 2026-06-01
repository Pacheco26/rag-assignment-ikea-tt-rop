"""Retrieval metrics: MRR, nDCG@k, Recall@k, Precision@k, SourceCoverage."""

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def mean_reciprocal_rank(
    retrieved_paper_ids: list[str],
    relevant_paper_ids: list[str],
) -> float:
    """Compute Mean Reciprocal Rank (MRR). Returns 0-1."""
    relevant_set = set(relevant_paper_ids)
    for rank, paper_id in enumerate(retrieved_paper_ids, 1):
        if paper_id in relevant_set:
            return 1.0 / rank
    return 0.0


def precision_at_k(
    retrieved_paper_ids: list[str],
    relevant_paper_ids: list[str],
    k: int = 5,
) -> float:
    """Precision@k: fraction of top-k results that are relevant."""
    relevant_set = set(relevant_paper_ids)
    top_k = retrieved_paper_ids[:k]
    if not top_k:
        return 0.0
    relevant_in_top_k = sum(1 for pid in top_k if pid in relevant_set)
    return relevant_in_top_k / len(top_k)


def recall_at_k(
    retrieved_paper_ids: list[str],
    relevant_paper_ids: list[str],
    k: int = 5,
) -> float:
    """Recall@k: fraction of relevant documents found in top-k.

    Uses unique document IDs to avoid inflating recall when multiple
    chunks from the same relevant document are retrieved.
    """
    relevant_set = set(relevant_paper_ids)
    if not relevant_set:
        return 0.0
    top_k = retrieved_paper_ids[:k]
    unique_relevant_found = set(pid for pid in top_k if pid in relevant_set)
    return len(unique_relevant_found) / len(relevant_set)


def ndcg_at_k(
    retrieved_paper_ids: list[str],
    relevant_paper_ids: list[str],
    k: int = 5,
) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Uses binary relevance at the document level. When a relevant document
    appears for the first time in the ranked list, it scores 1;
    subsequent occurrences of the same document score 0.
    """
    relevant_set = set(relevant_paper_ids)
    top_k = retrieved_paper_ids[:k]

    # DCG — only count first occurrence of each relevant document
    dcg = 0.0
    seen_relevant = set()
    for i, paper_id in enumerate(top_k):
        if paper_id in relevant_set and paper_id not in seen_relevant:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
            seen_relevant.add(paper_id)

    # Ideal DCG (all relevant documents at the top positions)
    ideal_rels = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_rels))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def source_coverage(
    retrieved_paper_ids: list[str],
    relevant_paper_ids: list[str],
) -> float:
    """Fraction of relevant source documents that appear in the retrieved set."""
    if not relevant_paper_ids:
        return 0.0
    retrieved_set = set(retrieved_paper_ids)
    relevant_set = set(relevant_paper_ids)
    covered = retrieved_set & relevant_set
    return len(covered) / len(relevant_set)


def compute_all_retrieval_metrics(
    retrieved_chunks: list[dict],
    relevant_paper_ids: list[str],
    k: int = 5,
) -> dict:
    """Compute all retrieval metrics for a single query.

    Args:
        retrieved_chunks: List of retrieval result dicts with 'metadata.paper_id'.
        relevant_paper_ids: Gold-standard relevant document IDs.
        k: Cutoff for @k metrics.

    Returns:
        Dict with all metric scores.
    """
    # Extract ordered document IDs from results
    retrieved_paper_ids = [
        r.get("metadata", {}).get("paper_id", "")
        for r in retrieved_chunks
    ]

    # Unique documents (maintaining order of first appearance)
    seen = set()
    unique_retrieved = []
    for pid in retrieved_paper_ids:
        if pid and pid not in seen:
            seen.add(pid)
            unique_retrieved.append(pid)

    return {
        "mrr": mean_reciprocal_rank(retrieved_paper_ids, relevant_paper_ids),
        f"precision@{k}": precision_at_k(retrieved_paper_ids, relevant_paper_ids, k),
        f"recall@{k}": recall_at_k(retrieved_paper_ids, relevant_paper_ids, k),
        f"ndcg@{k}": ndcg_at_k(retrieved_paper_ids, relevant_paper_ids, k),
        "source_coverage": source_coverage(unique_retrieved, relevant_paper_ids),
        "num_retrieved": len(retrieved_chunks),
        "num_unique_papers": len(unique_retrieved),
        "retrieved_papers": unique_retrieved,
    }


def aggregate_metrics(results: list[dict]) -> dict:
    """Aggregate metrics across multiple queries (mean + std)."""
    if not results:
        return {}

    metrics = {}
    numeric_keys = [k for k in results[0] if isinstance(results[0][k], (int, float))]

    for key in numeric_keys:
        values = [r[key] for r in results]
        metrics[f"{key}_mean"] = float(np.mean(values))
        metrics[f"{key}_std"] = float(np.std(values))

    metrics["num_queries"] = len(results)
    return metrics
