"""Generate all evaluation figures for the report and presentation."""

import json
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --- plot style ---
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.figsize": (8, 5),
})

COLORS = {
    "fixed_size": "#4C72B0",
    "semantic": "#DD8452",
    "hierarchical": "#55A868",
}
LABELS = {
    "fixed_size": "Fixed-Size (512)",
    "semantic": "Semantic",
    "hierarchical": "Hierarchical",
}
STRATEGIES = ["fixed_size", "semantic", "hierarchical"]


def load_evaluation_summary():
    """Load the evaluation_summary.csv into a dict."""
    path = RESULTS_DIR / "evaluation_summary.csv"
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        data = {}
        for row in reader:
            data[row["strategy"]] = {k: float(v) if k != "strategy" else v
                                     for k, v in row.items() if k != "strategy"}
    return data


def load_chunking_comparison():
    """Load chunking_comparison.json."""
    path = RESULTS_DIR / "chunking_comparison.json"
    with open(path, "r") as f:
        return json.load(f)


def load_evaluation_results():
    """Load full evaluation_results.json."""
    path = RESULTS_DIR / "evaluation_results.json"
    with open(path, "r") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Figure 1: Retrieval Metrics Comparison (Grouped Bar)
# --------------------------------------------------------------------------
def fig_retrieval_metrics(summary):
    metrics = ["retrieval_mrr_mean", "retrieval_precision@5_mean",
               "retrieval_ndcg@5_mean", "retrieval_source_coverage_mean"]
    metric_labels = ["MRR", "Precision@5", "nDCG@5", "Source Coverage"]

    x = np.arange(len(metric_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, strat in enumerate(STRATEGIES):
        values = [summary[strat][m] for m in metrics]
        stds = [summary[strat].get(m.replace("_mean", "_std"), 0) for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=LABELS[strat],
                      color=COLORS[strat], yerr=stds, capsize=3, alpha=0.9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("Retrieval Metrics by Chunking Strategy")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIGURES_DIR / "retrieval_metrics_comparison.png")
    plt.close(fig)
    print("  -> retrieval_metrics_comparison.png")


# --------------------------------------------------------------------------
# Figure 2: Answer Quality Metrics (Grouped Bar)
# --------------------------------------------------------------------------
def fig_answer_metrics(summary):
    metrics = ["answer_correctness_mean", "answer_faithfulness_mean",
               "answer_completeness_mean", "answer_citation_f1_mean"]
    metric_labels = ["Correctness", "Faithfulness", "Completeness", "Citation F1"]

    x = np.arange(len(metric_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, strat in enumerate(STRATEGIES):
        values = [summary[strat][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=LABELS[strat],
                      color=COLORS[strat], alpha=0.9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("Answer Quality Metrics by Chunking Strategy")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIGURES_DIR / "answer_quality_comparison.png")
    plt.close(fig)
    print("  -> answer_quality_comparison.png")


# --------------------------------------------------------------------------
# Figure 3: Intrinsic Chunk Quality (Radar Chart)
# --------------------------------------------------------------------------
def fig_intrinsic_radar(chunking):
    metrics = ["icc_mean", "icd", "bc"]
    metric_labels = ["ICC\n(Intra-Chunk\nCoherence)", "ICD\n(Inter-Chunk\nDistinctiveness)",
                     "BC\n(Boundary\nClarity)"]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for strat in STRATEGIES:
        values = [chunking[strat]["intrinsic_metrics"][m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", label=LABELS[strat], color=COLORS[strat], linewidth=2)
        ax.fill(angles, values, color=COLORS[strat], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.0)
    ax.set_title("Intrinsic Chunk Quality Metrics", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.savefig(FIGURES_DIR / "intrinsic_quality_radar.png")
    plt.close(fig)
    print("  -> intrinsic_quality_radar.png")


# --------------------------------------------------------------------------
# Figure 4: Chunk Size Distribution
# --------------------------------------------------------------------------
def fig_chunk_statistics(chunking):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Chunk Statistics by Strategy", fontsize=14, y=1.02)

    for i, strat in enumerate(STRATEGIES):
        ax = axes[i]
        data = chunking[strat]
        stats = {
            "Count": data["num_chunks"],
            "Avg Tokens": data["avg_tokens"],
            "Max Tokens": data["max_tokens"],
        }
        bars = ax.bar(stats.keys(), stats.values(), color=COLORS[strat], alpha=0.85)
        for bar, val in zip(bars, stats.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(LABELS[strat])
        ax.set_ylim(0, max(stats.values()) * 1.2)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "chunk_statistics.png")
    plt.close(fig)
    print("  -> chunk_statistics.png")


# --------------------------------------------------------------------------
# Figure 5: Performance by Query Category (Heatmap-style)
# --------------------------------------------------------------------------
def fig_category_heatmap(results):
    categories = ["factual", "comparative", "thematic"]
    metrics_keys = ["correctness", "faithfulness", "completeness"]

    # Compute per-category means
    data = {}
    for strat in STRATEGIES:
        data[strat] = {}
        for cat in categories:
            exps = [e for e in results["experiments"]
                    if e["strategy"] == strat and e["category"] == cat
                    and "error" not in e]
            data[strat][cat] = {}
            for m in metrics_keys:
                vals = [e["answer_metrics"][m] for e in exps]
                data[strat][cat][m] = np.mean(vals) if vals else 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Answer Metrics by Query Category and Strategy", fontsize=14, y=1.02)

    for idx, metric in enumerate(metrics_keys):
        ax = axes[idx]
        matrix = np.zeros((len(STRATEGIES), len(categories)))
        for i, strat in enumerate(STRATEGIES):
            for j, cat in enumerate(categories):
                matrix[i, j] = data[strat][cat][metric]

        im = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([c.capitalize() for c in categories])
        ax.set_yticks(range(len(STRATEGIES)))
        ax.set_yticklabels([LABELS[s] for s in STRATEGIES])
        ax.set_title(metric.capitalize())

        for i in range(len(STRATEGIES)):
            for j in range(len(categories)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if matrix[i, j] > 0.6 else "black")

    fig.colorbar(im, ax=axes, shrink=0.6, label="Score")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "category_performance_heatmap.png")
    plt.close(fig)
    print("  -> category_performance_heatmap.png")


# --------------------------------------------------------------------------
# Figure 6: MRR by Query Category (Grouped Bar)
# --------------------------------------------------------------------------
def fig_mrr_by_category(results):
    categories = ["factual", "comparative", "thematic"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(categories))
    width = 0.25

    for i, strat in enumerate(STRATEGIES):
        values = []
        for cat in categories:
            exps = [e for e in results["experiments"]
                    if e["strategy"] == strat and e["category"] == cat
                    and "error" not in e]
            mrrs = [e["retrieval_metrics"]["mrr"] for e in exps]
            values.append(np.mean(mrrs) if mrrs else 0)

        bars = ax.bar(x + i * width, values, width, label=LABELS[strat],
                      color=COLORS[strat], alpha=0.9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("MRR")
    ax.set_title("Mean Reciprocal Rank by Query Category")
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIGURES_DIR / "mrr_by_category.png")
    plt.close(fig)
    print("  -> mrr_by_category.png")


# --------------------------------------------------------------------------
# Figure 7: Build Time Comparison
# --------------------------------------------------------------------------
def fig_build_time(chunking):
    fig, ax = plt.subplots(figsize=(7, 4))
    times = [chunking[s]["build_time_seconds"] for s in STRATEGIES]
    bars = ax.barh([LABELS[s] for s in STRATEGIES], times,
                   color=[COLORS[s] for s in STRATEGIES], alpha=0.85)
    for bar, val in zip(bars, times):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}s", ha="left", va="center", fontsize=10)
    ax.set_xlabel("Build Time (seconds)")
    ax.set_title("Index Build Time by Strategy")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(FIGURES_DIR / "build_time_comparison.png")
    plt.close(fig)
    print("  -> build_time_comparison.png")


# --------------------------------------------------------------------------
# Figure 8: Overall Score Summary (Spider/Pentagon)
# --------------------------------------------------------------------------
def fig_overall_summary(summary, chunking):
    """Combined radar with 5 key dimensions."""
    dims = ["Retrieval\n(MRR)", "Answer\nCorrectness", "Faithfulness",
            "Citation\nF1", "Chunk\nCoherence"]

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for strat in STRATEGIES:
        values = [
            summary[strat]["retrieval_mrr_mean"],
            summary[strat]["answer_correctness_mean"],
            summary[strat]["answer_faithfulness_mean"],
            summary[strat]["answer_citation_f1_mean"],
            chunking[strat]["intrinsic_metrics"]["icc_mean"],
        ]
        values += values[:1]
        ax.plot(angles, values, "o-", label=LABELS[strat], color=COLORS[strat], linewidth=2)
        ax.fill(angles, values, color=COLORS[strat], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title("Overall Strategy Comparison", y=1.08, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.savefig(FIGURES_DIR / "overall_strategy_summary.png",
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("  -> overall_strategy_summary.png")


# --------------------------------------------------------------------------
# Figure 9: Confidence Score Distribution
# --------------------------------------------------------------------------
def fig_confidence_distribution(results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    fig.suptitle("Confidence Score Distribution by Strategy", fontsize=14, y=1.02)

    for i, strat in enumerate(STRATEGIES):
        ax = axes[i]
        exps = [e for e in results["experiments"]
                if e["strategy"] == strat and "error" not in e]
        confs = [e.get("confidence", 0.5) for e in exps]
        ax.hist(confs, bins=10, color=COLORS[strat], alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(confs), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean: {np.mean(confs):.3f}")
        ax.set_title(LABELS[strat])
        ax.set_xlabel("Confidence")
        if i == 0:
            ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confidence_distribution.png")
    plt.close(fig)
    print("  -> confidence_distribution.png")


# --------------------------------------------------------------------------
# Figure 10: Documents Retrieved Distribution
# --------------------------------------------------------------------------
def fig_documents_retrieved(results):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(STRATEGIES))
    width = 0.35

    for i, strat in enumerate(STRATEGIES):
        exps = [e for e in results["experiments"]
                if e["strategy"] == strat and "error" not in e]
        num_docs = [e["retrieval_metrics"]["num_unique_papers"] for e in exps]
        ax.bar(i, np.mean(num_docs), width, label=LABELS[strat],
               color=COLORS[strat], alpha=0.85,
               yerr=np.std(num_docs), capsize=5)
        ax.text(i, np.mean(num_docs) + 0.15,
                f"{np.mean(num_docs):.1f}", ha="center", fontsize=10)

    ax.set_ylabel("Avg. Unique Documents Retrieved")
    ax.set_title("Source Diversity: Unique Documents per Query")
    ax.set_xticks(range(len(STRATEGIES)))
    ax.set_xticklabels([LABELS[s] for s in STRATEGIES])
    ax.set_ylim(0, 5)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIGURES_DIR / "documents_retrieved_diversity.png")
    plt.close(fig)
    print("  -> documents_retrieved_diversity.png")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("Loading data...")
    summary = load_evaluation_summary()
    chunking = load_chunking_comparison()
    results = load_evaluation_results()

    print(f"\nGenerating figures to {FIGURES_DIR}/\n")

    fig_retrieval_metrics(summary)
    fig_answer_metrics(summary)
    fig_intrinsic_radar(chunking)
    fig_chunk_statistics(chunking)
    fig_category_heatmap(results)
    fig_mrr_by_category(results)
    fig_build_time(chunking)
    fig_overall_summary(summary, chunking)
    fig_confidence_distribution(results)
    fig_documents_retrieved(results)

    print(f"\nDone! {len(list(FIGURES_DIR.glob('*.png')))} figures generated.")


if __name__ == "__main__":
    main()
