"""Wrapper around ChunkAnalyzer for per-strategy evaluation."""

import logging
from typing import Optional

import numpy as np

from src.chunking.base_chunker import Chunk
from src.chunking.chunk_analyzer import ChunkAnalyzer

logger = logging.getLogger(__name__)


def evaluate_chunking_strategy(
    chunks: list[Chunk],
    strategy_name: str,
    analyzer: Optional[ChunkAnalyzer] = None,
) -> dict:
    """Evaluate a chunking strategy's intrinsic quality (ICC, ICD, SC, BC)."""
    analyzer = analyzer or ChunkAnalyzer()
    metrics = analyzer.analyze_chunks(chunks)
    metrics["strategy"] = strategy_name
    return metrics


def compare_strategies(
    strategy_chunks: dict[str, list[Chunk]],
    analyzer: Optional[ChunkAnalyzer] = None,
) -> dict:
    """Compare intrinsic metrics across multiple strategies."""
    analyzer = analyzer or ChunkAnalyzer()
    results = {}

    for strategy, chunks in strategy_chunks.items():
        logger.info(f"Evaluating {strategy} ({len(chunks)} chunks)...")
        results[strategy] = evaluate_chunking_strategy(
            chunks, strategy, analyzer,
        )

    # Build comparison table
    comparison = {
        "strategies": list(results.keys()),
        "metrics": {},
    }
    for metric in ["icc_mean", "icd", "sc", "bc", "num_chunks", "avg_tokens"]:
        comparison["metrics"][metric] = {
            strategy: results[strategy].get(metric, 0)
            for strategy in results
        }

    # Determine winners per metric
    winners = {}
    for metric in ["icc_mean", "icd", "bc"]:
        # Higher is better for ICC, ICD, BC
        values = comparison["metrics"].get(metric, {})
        if values:
            winners[metric] = max(values, key=values.get)

    # Lower is better for SC (size compliance = coefficient of variation)
    sc_values = comparison["metrics"].get("sc", {})
    if sc_values:
        winners["sc"] = min(sc_values, key=sc_values.get)

    comparison["winners"] = winners

    return {
        "per_strategy": results,
        "comparison": comparison,
    }


def generate_metrics_table(comparison: dict) -> str:
    """Generate a formatted markdown comparison table from compare_strategies() output."""
    strategies = comparison["comparison"]["strategies"]
    metrics_data = comparison["comparison"]["metrics"]

    # Header
    header = "| Metric | " + " | ".join(strategies) + " | Winner |"
    separator = "|--------|" + "|".join(["--------"] * len(strategies)) + "|--------|"

    rows = [header, separator]
    winners = comparison["comparison"].get("winners", {})

    metric_labels = {
        "icc_mean": "ICC (Coherence)",
        "icd": "ICD (Distinctiveness)",
        "sc": "SC (Size Variance)",
        "bc": "BC (Boundary Clarity)",
        "num_chunks": "Num Chunks",
        "avg_tokens": "Avg Tokens",
    }

    for metric_key, label in metric_labels.items():
        values = metrics_data.get(metric_key, {})
        row_parts = [f"| {label} "]
        for strategy in strategies:
            val = values.get(strategy, 0)
            if isinstance(val, float):
                row_parts.append(f"| {val:.4f} ")
            else:
                row_parts.append(f"| {val} ")
        winner = winners.get(metric_key, "-")
        row_parts.append(f"| {winner} |")
        rows.append("".join(row_parts))

    return "\n".join(rows)
