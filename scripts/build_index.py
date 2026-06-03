"""CLI: Chunk + embed + index (3 strategies).

Builds all vector store and BM25 indices for the three chunking strategies.
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CORPUS_PROCESSED, CHUNK_STRATEGIES
from src.chunking.chunker_factory import create_chunker
from src.chunking.chunk_analyzer import ChunkAnalyzer
from src.indexing.embedding_service import EmbeddingService
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_processed_docs() -> list[dict]:
    """Load all processed document JSONs."""
    docs = []
    for json_file in sorted(CORPUS_PROCESSED.glob("D*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            docs.append(json.load(f))
    return docs


def main():
    logger.info("=" * 60)
    logger.info("FurnishRAG Index Builder")
    logger.info("=" * 60)

    # Load processed documents
    papers = load_processed_docs()
    if not papers:
        logger.error(
            f"No processed documents found in {CORPUS_PROCESSED}. "
            "Run ingest_corpus.py first."
        )
        sys.exit(1)

    logger.info(f"Loaded {len(papers)} processed documents")

    # Initialize services
    embedding_service = EmbeddingService()
    vector_store = VectorStore(embedding_service=embedding_service)
    analyzer = ChunkAnalyzer()

    stats = {}

    for strategy in CHUNK_STRATEGIES:
        logger.info(f"\n{'='*60}")
        logger.info(f"Strategy: {strategy}")
        logger.info("=" * 60)

        start_time = time.time()

        chunker = create_chunker(strategy)
        all_chunks = []
        for paper in papers:
            chunks = chunker.chunk_paper(paper)
            all_chunks.extend(chunks)

        chunk_time = time.time() - start_time
        logger.info(
            f"Chunking: {len(all_chunks)} chunks in {chunk_time:.1f}s"
        )

        # Analyze chunk quality (intrinsic metrics)
        logger.info("Computing intrinsic metrics...")
        # Only analyze a sample if too many chunks
        sample_size = min(len(all_chunks), 100)
        sample_chunks = all_chunks[:sample_size]
        metrics = analyzer.analyze_chunks(sample_chunks)
        metrics.pop("icc_per_chunk", None)  # Remove per-chunk data for logging

        logger.info(f"Intrinsic metrics: {json.dumps(metrics, indent=2)}")

        # Index in vector store
        logger.info("Building vector store index...")
        vector_store.reset_collection(strategy)
        vector_store.add_chunks(all_chunks, strategy)

        # Build BM25 index
        logger.info("Building BM25 index...")
        bm25 = BM25Index(strategy)
        bm25.build_index(all_chunks)
        bm25.save()

        total_time = time.time() - start_time

        stats[strategy] = {
            "num_chunks": len(all_chunks),
            "avg_tokens": sum(c.token_count for c in all_chunks) / len(all_chunks),
            "min_tokens": min(c.token_count for c in all_chunks),
            "max_tokens": max(c.token_count for c in all_chunks),
            "intrinsic_metrics": metrics,
            "build_time_seconds": total_time,
        }

        logger.info(f"Strategy '{strategy}' complete in {total_time:.1f}s")

    # Save stats
    stats_path = Path("results") / "chunking_comparison.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Index Build Summary")
    logger.info("=" * 60)
    for strategy, s in stats.items():
        logger.info(
            f"  {strategy}: {s['num_chunks']} chunks, "
            f"avg {s['avg_tokens']:.0f} tokens, "
            f"ICC={s['intrinsic_metrics'].get('icc_mean', 0):.4f}, "
            f"ICD={s['intrinsic_metrics'].get('icd', 0):.4f}"
        )
    logger.info("\nDone! All indices built.")


if __name__ == "__main__":
    main()
