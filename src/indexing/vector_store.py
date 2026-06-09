"""ChromaDB vector store management — 3 collections (1 per strategy)."""

import logging
from pathlib import Path
from typing import Optional

import chromadb

from config.settings import CHROMA_DB_PATH, CHUNK_STRATEGIES
from src.chunking.base_chunker import Chunk
from src.indexing.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-based vector store with one collection per chunking strategy."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.persist_dir = persist_dir or CHROMA_DB_PATH
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.embedding_service = embedding_service or EmbeddingService()
        self._collections: dict[str, chromadb.Collection] = {}

    def _get_collection(self, strategy: str) -> chromadb.Collection:
        """Get or create a collection for a chunking strategy."""
        if strategy not in self._collections:
            self._collections[strategy] = self.client.get_or_create_collection(
                name=f"furnishrag_{strategy}",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[strategy]

    def add_chunks(self, chunks: list[Chunk], strategy: str) -> None:
        """Add chunks to the vector store for a given strategy."""
        if not chunks:
            return

        collection = self._get_collection(strategy)

        # Embed all chunk texts
        texts = [c.text for c in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        # Prepare metadata (ChromaDB requires flat dict with str/int/float values)
        ids = [c.chunk_id for c in chunks]
        metadatas = []
        for c in chunks:
            meta = {
                "paper_id": c.paper_id,
                "strategy": c.strategy,
                "section": c.section or "",
                "token_count": c.token_count,
            }
            if c.parent_id:
                meta["parent_id"] = c.parent_id
            metadatas.append(meta)

        # Add in batches (ChromaDB limit is ~5000 per batch)
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )

        logger.info(
            f"Added {len(chunks)} chunks to collection '{strategy}'"
        )

    def query(
        self,
        query_text: str,
        strategy: str,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        """Query the vector store for similar chunks.

        Args:
            query_text: The query string.
            strategy: Which strategy's collection to search.
            top_k: Number of results to return.
            filter_metadata: Optional ChromaDB where filter.

        Returns:
            List of dicts with 'id', 'text', 'metadata', 'distance'.
        """
        collection = self._get_collection(strategy)
        query_embedding = self.embedding_service.embed_query(query_text)

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata

        results = collection.query(**kwargs)

        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return formatted

    def get_collection_count(self, strategy: str) -> int:
        collection = self._get_collection(strategy)
        return collection.count()

    def reset_collection(self, strategy: str) -> None:
        collection_name = f"furnishrag_{strategy}"
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self._collections.pop(strategy, None)
        logger.info(f"Reset collection '{strategy}'")
