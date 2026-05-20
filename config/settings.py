"""Central configuration for FurnishRAG."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# -- Paths --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_RAW = PROJECT_ROOT / "corpus" / "raw"
CORPUS_PROCESSED = PROJECT_ROOT / "corpus" / "processed"
CORPUS_METADATA = PROJECT_ROOT / "corpus" / "metadata"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# -- LLM Provider (Groq — free) -----------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# -- Embeddings (local, free — sentence-transformers) --------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 outputs 384 dimensions

# -- Chunking ------------------------------------------------------------------
# Strategy 1: Fixed-Size
FIXED_CHUNK_SIZE = 512       # tokens
FIXED_CHUNK_OVERLAP = 50     # tokens

# Strategy 2: Semantic
SEMANTIC_BREAKPOINT_PERCENTILE = 95
SEMANTIC_BUFFER_SIZE = 1

# Strategy 3: Hierarchical Parent-Child
HIERARCHICAL_CHILD_SIZE = 128    # tokens — retrieval unit
HIERARCHICAL_PARENT_SIZE = 1024  # tokens — generation context

CHUNK_STRATEGIES = ["fixed_size", "semantic", "hierarchical"]

# -- Retrieval -----------------------------------------------------------------
TOP_K = 5                    # number of chunks to retrieve
HYBRID_ALPHA = 0.7           # weight for dense retrieval in RRF (1-alpha for BM25)
RRF_K = 60                   # constant for Reciprocal Rank Fusion

# -- Generation ----------------------------------------------------------------
MAX_TOKENS = 1024
TEMPERATURE = 0.1
CONFIDENCE_THRESHOLD = 0.3   # below this, trigger refusal

# -- Evaluation ----------------------------------------------------------------
EVAL_TOP_K_VALUES = [3, 5, 10]
NUM_TEST_QUERIES = 30

# -- Document metadata ---------------------------------------------------------
DOC_IDS = [f"D{str(i).zfill(2)}" for i in range(1, 21)]
NUM_DOCS = 20
