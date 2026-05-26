"""Local embedding service using sentence-transformers (free, no API needed)."""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages text embeddings using local sentence-transformers model."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.dimension = EMBEDDING_DIM
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a vector."""
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in batches."""
        logger.info(f"Embedding {len(texts)} texts locally...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 50,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Alias for embed_text."""
        return self.embed_text(query)
