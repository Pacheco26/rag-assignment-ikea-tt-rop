"""Corpus Explorer — Keywords, Topics, Clusters visualization."""

import json
import sys
from pathlib import Path

import streamlit as st
import plotly.express as px
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import CORPUS_PROCESSED, CORPUS_METADATA

st.title("Corpus Explorer")
st.markdown("Explore keywords, topics, and clusters across the 20 IKEA buying guides.")


@st.cache_data
def load_documents():
    """Load processed documents."""
    papers = []
    for f in sorted(CORPUS_PROCESSED.glob("D*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            papers.append(json.load(fh))
    return papers


@st.cache_data
def load_metadata():
    """Load document metadata."""
    meta_path = CORPUS_METADATA / "papers_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return []


# Load data
papers = load_documents()
metadata = load_metadata()

if not papers:
    st.warning("No processed documents found. Run `ingest_corpus.py` first.")
    st.stop()

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs(["Documents Overview", "Keywords", "Topics", "Clusters"])

with tab1:
    st.subheader("Documents Overview")

    if metadata:
        df = pd.DataFrame(metadata)
        st.dataframe(df, use_container_width=True)

        # Category distribution
        cat_counts = df["category"].value_counts()
        fig = px.pie(
            values=cat_counts.values,
            names=cat_counts.index,
            title="Documents by Category",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Word count per document
    word_counts = []
    for p in papers:
        wc = p.get("word_count", len(p.get("full_text", "").split()))
        word_counts.append({"Document": p["paper_id"], "Words": wc})

    if word_counts:
        wc_df = pd.DataFrame(word_counts)
        fig3 = px.bar(
            wc_df, x="Document", y="Words",
            title="Word Count per Document",
        )
        st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.subheader("Keyword Extraction")
    st.info("Run keyword extraction to populate this view.")

    selected_paper = st.selectbox(
        "Select Document",
        [p["paper_id"] for p in papers],
        key="kw_paper",
    )

    if st.button("Extract Keywords", key="extract_kw"):
        from src.text_mining.keyword_extractor import KeywordExtractor

        paper = next(p for p in papers if p["paper_id"] == selected_paper)
        extractor = KeywordExtractor()

        with st.spinner("Extracting keywords..."):
            result = extractor.extract_paper_keywords(paper)

        st.subheader(f"Keywords for {selected_paper}")
        for kw in result["global_keywords"]:
            st.write(f"- **{kw['keyword']}** (score: {kw['score']:.3f})")

        if result.get("section_keywords"):
            st.subheader("Per-Section Keywords")
            for section, kws in result["section_keywords"].items():
                with st.expander(section):
                    for kw in kws:
                        st.write(f"- {kw['keyword']} ({kw['score']:.3f})")

with tab3:
    st.subheader("Topic Modelling")
    st.info("Run topic modelling to discover latent topics (BERTopic on chunks).")

    if st.button("Run Topic Modelling", key="run_topics"):
        from src.text_mining.topic_modeler import TopicModeler
        from src.chunking.chunker_factory import create_chunker

        with st.spinner("Chunking documents for topic modelling..."):
            chunker = create_chunker("fixed_size")
            all_chunks = []
            for paper in papers:
                all_chunks.extend(chunker.chunk_paper(paper))

        with st.spinner(f"Running BERTopic on {len(all_chunks)} chunks..."):
            modeler = TopicModeler()
            result = modeler.fit_on_chunks(all_chunks)

        st.success(f"Discovered {result['num_topics']} topics!")
        st.write(f"Outliers: {result['outlier_count']}")

        # Taxonomy alignment
        alignment = result.get("taxonomy_alignment", {})
        if alignment:
            st.subheader("Taxonomy Alignment")
            st.write(alignment.get("interpretation", ""))
            st.metric(
                "Alignment Score",
                f"{alignment.get('overall_alignment_score', 0):.1%}",
            )

            # Detailed alignment table
            topic_alignment = alignment.get("topic_alignment", {})
            if topic_alignment:
                rows = []
                for tid, info in topic_alignment.items():
                    rows.append({
                        "Topic": int(tid),
                        "Best Match": info["best_matching_category"],
                        "Overlap": info["overlap_papers"],
                        "Total": info["total_papers_in_topic"],
                        "Documents": ", ".join(sorted(info["papers"])),
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # BERTopic visualizations
        st.subheader("Topic Visualizations")

        topic_viz = modeler.visualize_topics()
        if topic_viz is not None:
            st.plotly_chart(topic_viz, use_container_width=True)
        else:
            st.caption("Inter-topic distance map not available for this run.")

        hierarchy_viz = modeler.visualize_hierarchy()
        if hierarchy_viz is not None:
            st.plotly_chart(hierarchy_viz, use_container_width=True)
        else:
            st.caption("Topic hierarchy not available for this run.")

with tab4:
    st.subheader("Chunk Clustering")
    st.info("Visualize how chunks cluster across documents.")

    # Strategy selection OUTSIDE the button callback
    cluster_strategy = st.selectbox(
        "Strategy for clustering",
        ["fixed_size", "semantic", "hierarchical"],
        key="cluster_strategy",
    )

    if st.button("Run Clustering", key="run_clusters"):
        from src.text_mining.cluster_analyzer import ClusterAnalyzer
        from src.chunking.chunker_factory import create_chunker

        with st.spinner("Chunking and clustering..."):
            chunker = create_chunker(cluster_strategy)
            all_chunks = []
            for paper in papers:
                all_chunks.extend(chunker.chunk_paper(paper))

            analyzer = ClusterAnalyzer()
            texts = [c.text for c in all_chunks]
            paper_ids = [c.paper_id for c in all_chunks]

            cluster_result = analyzer.cluster_chunks(texts, paper_ids)
            coords = analyzer.reduce_dimensions(texts)

        # Plot 2D scatter
        plot_df = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "Document": paper_ids,
            "Cluster": [str(l) for l in cluster_result["labels"]],
        })

        fig = px.scatter(
            plot_df, x="x", y="y",
            color="Document",
            symbol="Cluster",
            title=f"Chunk Embeddings ({cluster_strategy}) — UMAP 2D",
            hover_data=["Document", "Cluster"],
        )
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Silhouette Score", f"{cluster_result['silhouette_score']:.3f}")
        st.metric("Num Clusters", cluster_result["num_clusters"])

        # Cluster detail table
        if cluster_result.get("cluster_info"):
            rows = []
            for cid, info in cluster_result["cluster_info"].items():
                rows.append({
                    "Cluster": cid,
                    "Size": info["size"],
                    "Documents": ", ".join(sorted(info["papers"])),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
