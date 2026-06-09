"""FurnishRAG - Streamlit Demo Application.

Multi-page app with:
1. Q&A Interface (with strategy selection)
2. Corpus Explorer (keywords, topics, clusters)
3. Evaluation Dashboard
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="FurnishRAG",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("FurnishRAG")
st.markdown(
    """
    **RAG System for exploring IKEA buying guides and furniture documentation**

    This system lets you query 20 IKEA buying guides using 3 different
    chunking strategies:
    - **Fixed-Size** (512 tokens, overlap 50)
    - **Semantic** (embedding similarity breakpoints)
    - **Hierarchical** (child=128, parent=1024)

    ---

    Navigate using the sidebar pages:
    - **Query**: Ask questions about IKEA products
    - **Explorer**: Explore keywords, topics, and clusters
    - **Evaluation**: View evaluation results and comparisons
    """
)

# Sidebar info
st.sidebar.title("About")
st.sidebar.info(
    """
    **FurnishRAG** — MINTRI Phase 2

    Authors: Tatiana Teixeira & Rodrigo Pacheco
    Course: MEI, ISEP

    Corpus: 20 IKEA buying guides
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Quick Stats:**
    - 20 buying guides analyzed
    - 5 product categories
    - 3 chunking strategies compared
    - 30 gold-standard test queries
    """
)
