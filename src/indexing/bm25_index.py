"""BM25 sparse index for hybrid retrieval."""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from config.settings import PROJECT_ROOT
from src.chunking.base_chunker import Chunk

logger = logging.getLogger(__name__)

BM25_CACHE_DIR = PROJECT_ROOT / ".bm25_cache"


class BM25Index:
    """BM25 sparse retrieval index."""

    def __init__(self, strategy: str):
        self.strategy = strategy
        self.index: Optional[BM25Okapi] = None
        self.chunks: list[Chunk] = []
        self.tokenized_corpus: list[list[str]] = []

        BM25_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def build_index(self, chunks: list[Chunk]) -> None:
        """Build BM25 index from a list of chunks."""
        self.chunks = chunks
        self.tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self.index = BM25Okapi(self.tokenized_corpus)
        logger.info(
            f"Built BM25 index for '{self.strategy}': {len(chunks)} documents"
        )

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Return the top-k BM25 matches for a query."""
        if self.index is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")

        tokenized_query = self._tokenize(query_text)
        scores = self.index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(scores[idx]),
                    "rank": rank + 1,
                })
        return results

    def save(self, path: Optional[Path] = None) -> None:
        """Save the BM25 index to disk."""
        path = path or (BM25_CACHE_DIR / f"bm25_{self.strategy}.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "tokenized_corpus": self.tokenized_corpus,
            }, f)
        logger.info(f"Saved BM25 index to {path}")

    def load(self, path: Optional[Path] = None) -> bool:
        """Load BM25 index from disk. Returns True if loaded ok."""
        path = path or (BM25_CACHE_DIR / f"bm25_{self.strategy}.pkl")
        if not path.exists():
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.chunks = data["chunks"]
        self.tokenized_corpus = data["tokenized_corpus"]
        self.index = BM25Okapi(self.tokenized_corpus)
        logger.info(f"Loaded BM25 index from {path}: {len(self.chunks)} documents")
        return True
