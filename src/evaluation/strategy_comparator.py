"""Strategy comparator: runs 30 queries x 3 strategies = 90 experiments."""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import CHUNK_STRATEGIES, TOP_K, RESULTS_DIR
from src.evaluation.test_queries import ALL_TEST_QUERIES, TestQuery
from src.evaluation.retrieval_metrics import compute_all_retrieval_metrics, aggregate_metrics
from src.evaluation.answer_metrics import AnswerMetrics
from src.generation.rag_pipeline import RAGPipeline
from src.indexing.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class StrategyComparator:
    """Runs full evaluation: 30 queries x 3 strategies = 90 experiments."""

    def __init__(
        self,
        pipelines: dict[str, RAGPipeline],
        answer_evaluator: Optional[AnswerMetrics] = None,
        top_k: int = TOP_K,
    ):
        """Initialize with pipeline dict and optional evaluator."""
        self.pipelines = pipelines
        self.answer_evaluator = answer_evaluator or AnswerMetrics()
        self.top_k = top_k

    def run_single_experiment(
        self,
        query: TestQuery,
        strategy: str,
    ) -> dict:
        """Run a single query-strategy experiment and collect all metrics."""
        pipeline = self.pipelines[strategy]
        start_time = time.time()

        # Run RAG pipeline
        rag_result = pipeline.query(
            query.query,
            top_k=self.top_k,
            return_context=True,
        )

        # Compute retrieval metrics
        retrieval_scores = compute_all_retrieval_metrics(
            rag_result.get("retrieved_chunks", []),
            query.relevant_docs,
            k=self.top_k,
        )

        # Compute answer metrics (LLM-as-judge)
        answer_scores = self.answer_evaluator.evaluate_answer(
            query=query.query,
            generated_answer=rag_result["answer"],
            gold_answer=query.gold_answer,
            relevant_paper_ids=query.relevant_docs,
        )

        total_time = time.time() - start_time

        return {
            "query_id": query.query_id,
            "category": query.category,
            "strategy": strategy,
            "query": query.query,
            "generated_answer": rag_result["answer"],
            "gold_answer": query.gold_answer,
            "retrieval_metrics": retrieval_scores,
            "answer_metrics": answer_scores,
            "rag_metadata": rag_result.get("metadata", {}),
            "confidence": rag_result.get("confidence", 0),
            "sources": rag_result.get("sources", []),
            "total_time": total_time,
        }

    def run_full_evaluation(
        self,
        queries: Optional[list[TestQuery]] = None,
        strategies: Optional[list[str]] = None,
        save_results: bool = True,
    ) -> dict:
        """Run full 30 x 3 evaluation.

        Args:
            queries: Test queries (default: ALL_TEST_QUERIES).
            strategies: Strategies to evaluate (default: all).
            save_results: Whether to save results to disk.

        Returns:
            Dict with all results, aggregated metrics, and comparison tables.
        """
        queries = queries or ALL_TEST_QUERIES
        strategies = strategies or list(self.pipelines.keys())

        all_results = []
        total = len(queries) * len(strategies)
        completed = 0

        logger.info(
            f"Starting evaluation: {len(queries)} queries x "
            f"{len(strategies)} strategies = {total} experiments"
        )

        for strategy in strategies:
            logger.info(f"\n{'='*60}\nEvaluating strategy: {strategy}\n{'='*60}")
            for query in queries:
                completed += 1
                logger.info(
                    f"[{completed}/{total}] {query.query_id} x {strategy}"
                )
                try:
                    result = self.run_single_experiment(query, strategy)
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Experiment failed: {query.query_id} x {strategy}: {e}")
                    all_results.append({
                        "query_id": query.query_id,
                        "category": query.category,
                        "strategy": strategy,
                        "error": str(e),
                    })

        # Aggregate results
        summary = self._aggregate_results(all_results)

        output = {
            "experiments": all_results,
            "summary": summary,
            "config": {
                "num_queries": len(queries),
                "strategies": strategies,
                "top_k": self.top_k,
            },
        }

        if save_results:
            self._save_results(output)

        return output

    def _aggregate_results(self, results: list[dict]) -> dict:
        summary = {"by_strategy": {}, "by_category": {}, "by_strategy_category": {}}

        # Group by strategy
        strategy_results = {}
        for r in results:
            if "error" in r:
                continue
            strategy = r["strategy"]
            if strategy not in strategy_results:
                strategy_results[strategy] = []
            strategy_results[strategy].append(r)

        # Per-strategy aggregation
        for strategy, strat_results in strategy_results.items():
            retrieval_metrics = [r["retrieval_metrics"] for r in strat_results]
            answer_metrics = [r["answer_metrics"] for r in strat_results]

            summary["by_strategy"][strategy] = {
                "retrieval": aggregate_metrics(retrieval_metrics),
                "answer": {
                    "correctness_mean": _safe_mean([m.get("correctness", 0) for m in answer_metrics]),
                    "faithfulness_mean": _safe_mean([m.get("faithfulness", 0) for m in answer_metrics]),
                    "completeness_mean": _safe_mean([m.get("completeness", 0) for m in answer_metrics]),
                    "citation_f1_mean": _safe_mean([m.get("citation_f1", 0) for m in answer_metrics]),
                },
                "num_experiments": len(strat_results),
            }

        # Per-category per-strategy
        for r in results:
            if "error" in r:
                continue
            key = f"{r['strategy']}_{r['category']}"
            if key not in summary["by_strategy_category"]:
                summary["by_strategy_category"][key] = {
                    "strategy": r["strategy"],
                    "category": r["category"],
                    "results": [],
                }
            summary["by_strategy_category"][key]["results"].append(r)

        # Aggregate per strategy-category
        for key, data in summary["by_strategy_category"].items():
            strat_results = data["results"]
            retrieval_metrics = [r["retrieval_metrics"] for r in strat_results]
            answer_metrics = [r["answer_metrics"] for r in strat_results]
            data["retrieval"] = aggregate_metrics(retrieval_metrics)
            data["answer"] = {
                "correctness_mean": _safe_mean([m.get("correctness", 0) for m in answer_metrics]),
                "faithfulness_mean": _safe_mean([m.get("faithfulness", 0) for m in answer_metrics]),
                "completeness_mean": _safe_mean([m.get("completeness", 0) for m in answer_metrics]),
                "citation_f1_mean": _safe_mean([m.get("citation_f1", 0) for m in answer_metrics]),
            }
            data["num_experiments"] = len(strat_results)
            del data["results"]  # remove raw results from summary

        return summary

    def _save_results(self, output: dict) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Full JSON results
        results_path = RESULTS_DIR / "evaluation_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Results saved to {results_path}")

        # Summary CSV
        rows = []
        for strategy, data in output["summary"]["by_strategy"].items():
            row = {"strategy": strategy}
            if "retrieval" in data:
                for k, v in data["retrieval"].items():
                    row[f"retrieval_{k}"] = v
            if "answer" in data:
                for k, v in data["answer"].items():
                    row[f"answer_{k}"] = v
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            csv_path = RESULTS_DIR / "evaluation_summary.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Summary saved to {csv_path}")

    def generate_comparison_table(self, summary: dict) -> str:
        """Generate a formatted markdown comparison table."""
        strategies = list(summary["by_strategy"].keys())

        header = "| Metric | " + " | ".join(strategies) + " |"
        sep = "|--------|" + "|".join(["--------"] * len(strategies)) + "|"

        rows = [header, sep]

        # Retrieval metrics
        for metric in ["mrr_mean", f"ndcg@{self.top_k}_mean", f"recall@{self.top_k}_mean",
                        f"precision@{self.top_k}_mean", "source_coverage_mean"]:
            label = metric.replace("_mean", "").upper()
            values = []
            for s in strategies:
                val = summary["by_strategy"].get(s, {}).get("retrieval", {}).get(metric, 0)
                values.append(f"{val:.4f}")
            rows.append(f"| {label} | " + " | ".join(values) + " |")

        # Answer metrics
        for metric in ["correctness_mean", "faithfulness_mean", "completeness_mean", "citation_f1_mean"]:
            label = metric.replace("_mean", "").title()
            values = []
            for s in strategies:
                val = summary["by_strategy"].get(s, {}).get("answer", {}).get(metric, 0)
                values.append(f"{val:.4f}")
            rows.append(f"| {label} | " + " | ".join(values) + " |")

        return "\n".join(rows)


def _safe_mean(values: list) -> float:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else 0.0
