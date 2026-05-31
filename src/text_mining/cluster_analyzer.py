"""Chunk clustering and retrieval diversity analysis."""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


class ClusterAnalyzer:
    """Clustering and diversity analysis for chunks and retrieval results."""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model_name)

    def cluster_chunks(
        self,
        texts: list[str],
        paper_ids: list[str],
        n_clusters: Optional[int] = None,
        method: str = "kmeans",
    ) -> dict:
        """Cluster text chunks and return assignments + silhouette metrics."""
        if len(texts) < 3:
            return {
                "num_clusters": 1,
                "labels": [0] * len(texts),
                "silhouette_score": 0,
                "cluster_info": {},
            }

        # Embed texts
        embeddings = self.model.encode(texts, show_progress_bar=False)

        if method == "hdbscan":
            try:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
                labels = clusterer.fit_predict(embeddings)
            except ImportError:
                logger.warning("HDBSCAN not available, falling back to KMeans")
                method = "kmeans"

        if method == "kmeans":
            if n_clusters is None:
                # Auto-select n_clusters using silhouette score
                best_k, best_score = 2, -1
                max_k = min(len(texts) - 1, 10)
                for k in range(2, max_k + 1):
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    temp_labels = km.fit_predict(embeddings)
                    score = silhouette_score(embeddings, temp_labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                n_clusters = best_k

            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(embeddings)

        # Analyze clusters
        labels_list = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
        unique_labels = set(labels_list)
        if -1 in unique_labels:
            unique_labels.discard(-1)

        cluster_info = {}
        for label in unique_labels:
            indices = [i for i, l in enumerate(labels_list) if l == label]
            cluster_papers = [paper_ids[i] for i in indices]
            cluster_info[int(label)] = {
                "size": len(indices),
                "papers": list(set(cluster_papers)),
                "paper_distribution": {
                    p: cluster_papers.count(p)
                    for p in set(cluster_papers)
                },
            }

        # Compute silhouette score if valid
        sil_score = 0.0
        if len(unique_labels) > 1 and len(unique_labels) < len(texts):
            try:
                sil_score = float(silhouette_score(embeddings, labels))
            except ValueError:
                pass

        result = {
            "num_clusters": len(unique_labels),
            "labels": labels_list,
            "silhouette_score": sil_score,
            "cluster_info": cluster_info,
            "method": method,
        }

        logger.info(
            f"Clustering: {len(unique_labels)} clusters, "
            f"silhouette={sil_score:.3f} ({method})"
        )
        return result

    def compute_diversity_score(
        self,
        retrieved_chunks: list[dict],
    ) -> dict:
        """Compute diversity metrics for retrieved chunks (document/section/semantic coverage)."""
        if not retrieved_chunks:
            return {"diversity_score": 0, "paper_coverage": 0, "section_coverage": 0}

        # Document diversity
        papers = [r.get("metadata", {}).get("paper_id", "") for r in retrieved_chunks]
        unique_papers = set(p for p in papers if p)
        paper_coverage = len(unique_papers) / max(len(retrieved_chunks), 1)

        # Section diversity
        sections = [r.get("metadata", {}).get("section", "") for r in retrieved_chunks]
        unique_sections = set(s for s in sections if s)
        section_coverage = len(unique_sections) / max(len(retrieved_chunks), 1)

        # Embedding diversity (average pairwise distance)
        texts = [r["text"] for r in retrieved_chunks]
        if len(texts) >= 2:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            from sklearn.metrics.pairwise import cosine_distances
            distances = cosine_distances(embeddings)
            # Average of upper triangle
            n = len(embeddings)
            avg_distance = np.sum(np.triu(distances, k=1)) / (n * (n - 1) / 2)
        else:
            avg_distance = 0.0

        diversity_score = (paper_coverage + section_coverage + avg_distance) / 3

        return {
            "diversity_score": float(diversity_score),
            "paper_coverage": float(paper_coverage),
            "section_coverage": float(section_coverage),
            "semantic_diversity": float(avg_distance),
            "unique_papers": list(unique_papers),
            "unique_sections": list(unique_sections),
            "num_chunks": len(retrieved_chunks),
        }

    def reduce_dimensions(
        self,
        texts: list[str],
        n_components: int = 2,
    ) -> np.ndarray:
        """Reduce embedding dimensions for visualization (UMAP with PCA fallback)."""
        embeddings = self.model.encode(texts, show_progress_bar=False)

        try:
            from umap import UMAP
            reducer = UMAP(n_components=n_components, random_state=42, n_neighbors=min(15, len(texts) - 1))
            return reducer.fit_transform(embeddings)
        except ImportError:
            # Fallback to PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            return pca.fit_transform(embeddings)
