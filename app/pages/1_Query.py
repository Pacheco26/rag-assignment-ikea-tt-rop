"""Q&A Interface — Ask questions about IKEA buying guides.

Integrates RAG pipeline with text mining analysis on retrieved results:
keyword extraction, summarization, diversity scoring, taxonomy classification.
"""

import sys
from pathlib import Path

import streamlit as st
import plotly.express as px
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import CHUNK_STRATEGIES, TOP_K
from src.indexing.embedding_service import EmbeddingService
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Index
from src.indexing.hybrid_retriever import HybridRetriever
from src.generation.rag_pipeline import RAGPipeline

st.title("Query Interface")
st.markdown("Ask questions about IKEA products and buying guides.")


@st.cache_resource
def load_pipelines():
    """Load RAG pipelines (cached)."""
    embedding_service = EmbeddingService()
    vector_store = VectorStore(embedding_service=embedding_service)
    pipelines = {}

    for strategy in CHUNK_STRATEGIES:
        bm25 = BM25Index(strategy)
        if bm25.load():
            retriever = HybridRetriever(
                vector_store=vector_store,
                bm25_index=bm25,
                strategy=strategy,
            )
            pipelines[strategy] = RAGPipeline(retriever=retriever)

    return pipelines


@st.cache_resource
def load_text_mining_tools():
    """Load text mining tools (cached to avoid reloading models)."""
    from src.text_mining.keyword_extractor import KeywordExtractor
    from src.text_mining.cluster_analyzer import ClusterAnalyzer
    from src.text_mining.classifier import TaxonomyClassifier
    return {
        "keyword_extractor": KeywordExtractor(),
        "cluster_analyzer": ClusterAnalyzer(),
        "classifier": TaxonomyClassifier(),
    }


def render_text_mining_analysis(result: dict, query: str):
    """Render text mining analysis on retrieved chunks below the RAG answer.

    Performs: keyword extraction, diversity analysis, taxonomy classification,
    and optional chunk summarization on the retrieved results.
    """
    retrieved_chunks = result.get("retrieved_chunks", [])
    if not retrieved_chunks:
        return

    st.markdown("---")
    st.markdown("### Text Mining Analysis on Retrieved Results")

    tools = load_text_mining_tools()

    # --- keyword extraction ---
    with st.expander("Keywords from Retrieved Chunks", expanded=True):
        combined_text = " ".join(
            c.get("text", "") for c in retrieved_chunks
        )[:10000]
        keywords = tools["keyword_extractor"].extract_keywords(combined_text, top_n=10)

        if keywords:
            kw_df = pd.DataFrame(keywords)
            fig = px.bar(
                kw_df, x="score", y="keyword", orientation="h",
                title="Top Keywords in Retrieved Context",
                labels={"score": "Relevance Score", "keyword": ""},
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No keywords extracted.")

    # --- taxonomy classification ---
    with st.expander("Taxonomy Classification", expanded=True):
        cls_result = tools["classifier"].classify_retrieved_results(retrieved_chunks)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Dominant Category", cls_result["dominant_category"])

        with col_b:
            dist = cls_result["category_distribution"]
            if dist:
                dist_df = pd.DataFrame(
                    [{"Category": k, "Proportion": v} for k, v in dist.items()]
                )
                fig2 = px.pie(
                    dist_df, values="Proportion", names="Category",
                    title="Category Distribution of Retrieved Chunks",
                )
                st.plotly_chart(fig2, use_container_width=True)

    # --- retrieval diversity ---
    with st.expander("Retrieval Diversity"):
        diversity = tools["cluster_analyzer"].compute_diversity_score(retrieved_chunks)

        cols = st.columns(4)
        cols[0].metric("Diversity Score", f"{diversity['diversity_score']:.2f}")
        cols[1].metric("Document Coverage", f"{diversity['paper_coverage']:.2f}")
        cols[2].metric("Section Coverage", f"{diversity['section_coverage']:.2f}")
        cols[3].metric("Semantic Diversity", f"{diversity['semantic_diversity']:.2f}")

        st.caption(
            f"Documents: {', '.join(diversity.get('unique_papers', []))} | "
            f"Sections: {', '.join(diversity.get('unique_sections', []))}"
        )

    # --- chunk summarization ---
    with st.expander("Synthesized Summary of Retrieved Chunks"):
        if st.button("Generate Summary (requires LLM call)", key="gen_summary"):
            from src.text_mining.summarizer import Summarizer
            summarizer = Summarizer()
            with st.spinner("Summarizing retrieved chunks..."):
                summary = summarizer.summarize_retrieved_chunks(
                    retrieved_chunks, query,
                )
            st.markdown(summary)


# strategy selection
col1, col2 = st.columns([3, 1])
with col1:
    strategy = st.selectbox(
        "Chunking Strategy",
        CHUNK_STRATEGIES,
        format_func=lambda x: {
            "fixed_size": "Fixed-Size (512 tokens)",
            "semantic": "Semantic (breakpoint)",
            "hierarchical": "Hierarchical (child=128, parent=1024)",
        }.get(x, x),
    )
with col2:
    top_k = st.slider("Top-K", 3, 10, TOP_K)
    compare_all = st.checkbox("Compare all strategies")

# Initialize session state for example queries
if "example_query" not in st.session_state:
    st.session_state.example_query = ""

# examples
with st.expander("Example Queries"):
    examples = [
        "What frame sizes are available for the PAX wardrobe system?",
        "Compare KALLAX and BILLY for storage solutions.",
        "What materials does IKEA use for sustainability?",
        "How do I plan a SEKTION kitchen installation?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            st.session_state.example_query = ex

# query input
query = st.text_area(
    "Your Question",
    value=st.session_state.example_query,
    placeholder="e.g., What dimensions are available for the PAX wardrobe system?",
    height=100,
)

# Clear example after use
if st.session_state.example_query and query == st.session_state.example_query:
    st.session_state.example_query = ""

# run query
if st.button("Ask", type="primary", disabled=not query):
    pipelines = load_pipelines()

    if not pipelines:
        st.error("No pipelines loaded. Please run `build_index.py` first.")
    else:
        if compare_all:
            # Run on all strategies
            tabs = st.tabs(list(pipelines.keys()))
            for tab, (strat_name, pipeline) in zip(tabs, pipelines.items()):
                with tab:
                    with st.spinner(f"Querying {strat_name}..."):
                        result = pipeline.query(query, top_k=top_k, return_context=True)
                    st.markdown(f"### Answer ({strat_name})")
                    st.markdown(result["answer"])
                    st.metric("Confidence", f"{result['confidence']:.2f}")

                    with st.expander("Sources"):
                        for src in result.get("sources", []):
                            st.write(f"- {src['paper_id']} ({src.get('section', '')})")

                    render_text_mining_analysis(result, query)
        else:
            # Single strategy
            if strategy in pipelines:
                with st.spinner(f"Querying with {strategy}..."):
                    result = pipelines[strategy].query(
                        query, top_k=top_k, return_context=True,
                    )

                st.markdown("### Answer")
                st.markdown(result["answer"])

                col1, col2, col3 = st.columns(3)
                col1.metric("Confidence", f"{result['confidence']:.2f}")
                col2.metric("Guides Used", result["metadata"]["num_papers"])
                col3.metric("Time", f"{result['metadata']['total_time']:.2f}s")

                with st.expander("Retrieved Sources"):
                    for src in result.get("sources", []):
                        st.write(f"- **{src['paper_id']}** ({src.get('section', '')})")

                with st.expander("Retrieved Context"):
                    st.text(result.get("context", "N/A")[:3000])

                if "verification" in result:
                    with st.expander("Verification Details"):
                        st.json(result["verification"])

                # Text mining analysis on retrieved results
                render_text_mining_analysis(result, query)
            else:
                st.error(f"Strategy '{strategy}' not loaded.")
