"""Evaluation Dashboard — View results from 30 x 3 experiments."""

import json
import sys
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import RESULTS_DIR

st.title("Evaluation Dashboard")
st.markdown("Results from 30 queries x 3 chunking strategies = 90 experiments.")


@st.cache_data
def load_results():
    """Load evaluation results."""
    results_path = RESULTS_DIR / "evaluation_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_chunking_comparison():
    """Load chunking comparison data."""
    path = RESULTS_DIR / "chunking_comparison.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


results = load_results()
chunking_data = load_chunking_comparison()

if not results:
    st.warning("No evaluation results found. Run `run_evaluation.py` first.")

    # Show chunking comparison if available
    if chunking_data:
        st.subheader("Chunking Strategy Comparison (Intrinsic Metrics)")
        rows = []
        for strategy, data in chunking_data.items():
            metrics = data.get("intrinsic_metrics", {})
            rows.append({
                "Strategy": strategy,
                "Num Chunks": data["num_chunks"],
                "Avg Tokens": f"{data['avg_tokens']:.0f}",
                "ICC": f"{metrics.get('icc_mean', 0):.4f}",
                "ICD": f"{metrics.get('icd', 0):.4f}",
                "SC": f"{metrics.get('sc', 0):.4f}",
                "BC": f"{metrics.get('bc', 0):.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.stop()

summary = results.get("summary", {})
experiments = results.get("experiments", [])

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Per-Category", "Failure Analysis", "Intrinsic Metrics",
])

with tab1:
    st.subheader("Strategy Comparison Overview")

    # Build comparison table
    strategies = list(summary.get("by_strategy", {}).keys())
    if strategies:
        rows = []
        for strategy in strategies:
            data = summary["by_strategy"][strategy]
            ret = data.get("retrieval", {})
            ans = data.get("answer", {})
            rows.append({
                "Strategy": strategy,
                "MRR": f"{ret.get('mrr_mean', 0):.4f}",
                "nDCG@5": f"{ret.get('ndcg@5_mean', 0):.4f}",
                "Recall@5": f"{ret.get('recall@5_mean', 0):.4f}",
                "SourceCov": f"{ret.get('source_coverage_mean', 0):.4f}",
                "Correctness": f"{ans.get('correctness_mean', 0):.4f}",
                "Faithfulness": f"{ans.get('faithfulness_mean', 0):.4f}",
                "Citation F1": f"{ans.get('citation_f1_mean', 0):.4f}",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Radar chart
        metric_names = ["MRR", "nDCG@5", "Recall@5", "Correctness", "Faithfulness"]
        fig = go.Figure()
        for strategy in strategies:
            data = summary["by_strategy"][strategy]
            ret = data.get("retrieval", {})
            ans = data.get("answer", {})
            values = [
                ret.get("mrr_mean", 0),
                ret.get("ndcg@5_mean", 0),
                ret.get("recall@5_mean", 0),
                ans.get("correctness_mean", 0),
                ans.get("faithfulness_mean", 0),
            ]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_names,
                fill="toself",
                name=strategy,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Strategy Comparison Radar",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Per-Category Breakdown")

    by_sc = summary.get("by_strategy_category", {})
    if by_sc:
        rows = []
        for key, data in by_sc.items():
            ret = data.get("retrieval", {})
            ans = data.get("answer", {})
            rows.append({
                "Strategy": data["strategy"],
                "Category": data["category"],
                "MRR": f"{ret.get('mrr_mean', 0):.4f}",
                "Correctness": f"{ans.get('correctness_mean', 0):.4f}",
                "N": data.get("num_experiments", 0),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        # Grouped bar chart
        fig = px.bar(
            df, x="Category", y="Correctness",
            color="Strategy", barmode="group",
            title="Correctness by Category and Strategy",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Failure Analysis")
    st.markdown(
        "Experiments where correctness < 0.3, showing where each "
        "strategy struggles."
    )

    failures = []
    for exp in experiments:
        if "error" in exp:
            failures.append({
                "Query": exp.get("query_id", "?"),
                "Strategy": exp.get("strategy", "?"),
                "Issue": f"Error: {exp['error'][:100]}",
            })
        elif exp.get("answer_metrics", {}).get("correctness", 1) < 0.3:
            failures.append({
                "Query": exp["query_id"],
                "Strategy": exp["strategy"],
                "Category": exp["category"],
                "Correctness": f"{exp['answer_metrics']['correctness']:.2f}",
                "Question": exp["query"][:80] + "...",
            })

    if failures:
        st.dataframe(pd.DataFrame(failures), use_container_width=True)
        st.info(
            f"Found {len(failures)} failure cases out of {len(experiments)} experiments "
            f"({len(failures)/max(len(experiments),1)*100:.1f}%)"
        )
    else:
        st.success("No major failures detected (all correctness >= 0.3)!")

with tab4:
    st.subheader("Intrinsic Chunk Quality Metrics")
    st.markdown("ICC (Coherence), ICD (Distinctiveness), SC (Size Compliance), BC (Boundary Clarity)")

    if chunking_data:
        rows = []
        for strategy, data in chunking_data.items():
            metrics = data.get("intrinsic_metrics", {})
            rows.append({
                "Strategy": strategy,
                "Chunks": data["num_chunks"],
                "Avg Tokens": f"{data['avg_tokens']:.0f}",
                "ICC (Coherence)": f"{metrics.get('icc_mean', 0):.4f}",
                "ICD (Distinctiveness)": f"{metrics.get('icd', 0):.4f}",
                "SC (Size Variance)": f"{metrics.get('sc', 0):.4f}",
                "BC (Boundary Clarity)": f"{metrics.get('bc', 0):.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Bar chart comparison (use numeric values, not formatted strings)
        chart_rows = []
        for strategy, data in chunking_data.items():
            metrics = data.get("intrinsic_metrics", {})
            chart_rows.append({
                "Strategy": strategy,
                "ICC (Coherence)": metrics.get("icc_mean", 0),
                "ICD (Distinctiveness)": metrics.get("icd", 0),
                "BC (Boundary Clarity)": metrics.get("bc", 0),
            })
        metric_cols = ["ICC (Coherence)", "ICD (Distinctiveness)", "BC (Boundary Clarity)"]
        chart_df = pd.DataFrame(chart_rows)
        fig = px.bar(
            chart_df.melt(id_vars="Strategy", value_vars=metric_cols,
                     var_name="Metric", value_name="Score"),
            x="Metric", y="Score", color="Strategy",
            barmode="group",
            title="Intrinsic Quality Metrics Comparison",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Run `build_index.py` to generate chunking comparison data.")
