"""CLI: Run full evaluation (30 queries x 3 strategies = 90 experiments)."""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CHUNK_STRATEGIES, CORPUS_PROCESSED
from src.indexing.embedding_service import EmbeddingService
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Index
from src.indexing.hybrid_retriever import HybridRetriever
from src.generation.rag_pipeline import RAGPipeline
from src.evaluation.strategy_comparator import StrategyComparator
from src.evaluation.test_queries import ALL_TEST_QUERIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def setup_pipelines() -> dict[str, RAGPipeline]:
    """Initialize RAG pipelines for all strategies."""
    embedding_service = EmbeddingService()
    vector_store = VectorStore(embedding_service=embedding_service)

    pipelines = {}
    for strategy in CHUNK_STRATEGIES:
        # Load BM25 index
        bm25 = BM25Index(strategy)
        if not bm25.load():
            logger.warning(
                f"BM25 index for '{strategy}' not found. "
                "Run build_index.py first."
            )
            continue

        # Create hybrid retriever
        retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_index=bm25,
            strategy=strategy,
        )

        # Create RAG pipeline
        pipeline = RAGPipeline(retriever=retriever)
        pipelines[strategy] = pipeline

    return pipelines


def main():
    logger.info("=" * 60)
    logger.info("FurnishRAG Full Evaluation")
    logger.info(f"30 queries x {len(CHUNK_STRATEGIES)} strategies = "
                f"{30 * len(CHUNK_STRATEGIES)} experiments")
    logger.info("=" * 60)

    # Setup pipelines
    pipelines = setup_pipelines()
    if not pipelines:
        logger.error("No pipelines initialized. Run build_index.py first.")
        sys.exit(1)

    logger.info(f"Initialized pipelines: {list(pipelines.keys())}")

    # Run evaluation
    comparator = StrategyComparator(pipelines=pipelines)
    results = comparator.run_full_evaluation()

    # Print comparison table
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS COMPARISON TABLE")
    logger.info("=" * 60)
    table = comparator.generate_comparison_table(results["summary"])
    print(table)

    # Print per-category breakdown
    logger.info("\n" + "=" * 60)
    logger.info("PER-CATEGORY BREAKDOWN")
    logger.info("=" * 60)
    for key, data in results["summary"]["by_strategy_category"].items():
        logger.info(
            f"  {data['strategy']} x {data['category']}: "
            f"correctness={data['answer'].get('correctness_mean', 0):.3f}, "
            f"MRR={data['retrieval'].get('mrr_mean', 0):.3f}"
        )

    # Identify failure cases
    logger.info("\n" + "=" * 60)
    logger.info("FAILURE CASES (correctness < 0.3)")
    logger.info("=" * 60)
    for exp in results["experiments"]:
        if "error" in exp:
            continue
        correctness = exp.get("answer_metrics", {}).get("correctness", 0)
        if correctness < 0.3:
            logger.info(
                f"  [{exp['query_id']} x {exp['strategy']}] "
                f"correctness={correctness:.2f} - "
                f"Query: {exp['query'][:80]}..."
            )

    logger.info("\nEvaluation complete! Results saved to results/")


if __name__ == "__main__":
    main()
