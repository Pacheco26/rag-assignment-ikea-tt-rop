"""BERTopic topic modelling with IKEA taxonomy comparison."""

import logging
from typing import Optional

import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from config.settings import BERTOPIC_MIN_TOPIC_SIZE
from src.chunking.base_chunker import Chunk

logger = logging.getLogger(__name__)

# IKEA document taxonomy
IKEA_TAXONOMY = {
    "Storage & Organization": ["D01", "D03", "D04", "D13"],
    "Bedroom & Wardrobe": ["D02", "D05", "D06", "D14", "D20"],
    "Kitchen & Dining": ["D07", "D11", "D19"],
    "Living & Outdoor": ["D08", "D09", "D10", "D16"],
    "Home & Sustainability": ["D12", "D15", "D17", "D18"],
}


class TopicModeler:
    """BERTopic-based topic modelling for the corpus."""

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        min_topic_size: int = BERTOPIC_MIN_TOPIC_SIZE,
        nr_topics: str = "auto",
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        self.topic_model: Optional[BERTopic] = None

    def fit_on_chunks(
        self,
        chunks: list[Chunk],
    ) -> dict:
        """Fit BERTopic on chunk texts.

        Using chunks (hundreds) instead of documents (20) gives BERTopic
        enough data points for meaningful topic discovery.

        Args:
            chunks: List of Chunk objects from any/all strategies.

        Returns:
            Dict with topics, assignments, and comparison to IKEA taxonomy.
        """
        texts = [c.text for c in chunks]
        paper_ids = [c.paper_id for c in chunks]

        # Custom vectorizer for product/furniture text
        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        )

        # Fit BERTopic
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            min_topic_size=self.min_topic_size,
            nr_topics=self.nr_topics if self.nr_topics != "auto" else None,
            vectorizer_model=vectorizer,
            verbose=False,
        )

        topics, probs = self.topic_model.fit_transform(texts)

        # Get topic info
        topic_info = self.topic_model.get_topic_info()

        # Map topics to documents
        topic_paper_mapping = {}
        for chunk, topic_id in zip(chunks, topics):
            if topic_id == -1:  # outlier topic
                continue
            if topic_id not in topic_paper_mapping:
                topic_paper_mapping[topic_id] = set()
            topic_paper_mapping[topic_id].add(chunk.paper_id)

        # Compare with IKEA taxonomy
        taxonomy_alignment = self._compare_with_taxonomy(topic_paper_mapping)

        result = {
            "num_topics": len(set(topics)) - (1 if -1 in topics else 0),
            "topic_info": topic_info.to_dict() if hasattr(topic_info, 'to_dict') else str(topic_info),
            "topics_per_chunk": list(zip([c.chunk_id for c in chunks], topics)),
            "topic_paper_mapping": {
                k: list(v) for k, v in topic_paper_mapping.items()
            },
            "taxonomy_alignment": taxonomy_alignment,
            "outlier_count": sum(1 for t in topics if t == -1),
        }

        logger.info(
            f"BERTopic: {result['num_topics']} topics discovered "
            f"from {len(chunks)} chunks ({result['outlier_count']} outliers)"
        )
        return result

    def _compare_with_taxonomy(
        self,
        topic_paper_mapping: dict[int, set],
    ) -> dict:
        """Compare discovered topics with IKEA taxonomy categories.

        Returns alignment score and mapping.
        """
        alignment = {}
        for topic_id, papers in topic_paper_mapping.items():
            best_category = None
            best_overlap = 0

            for category, category_papers in IKEA_TAXONOMY.items():
                overlap = len(papers & set(category_papers))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_category = category

            alignment[topic_id] = {
                "best_matching_category": best_category,
                "overlap_papers": best_overlap,
                "total_papers_in_topic": len(papers),
                "papers": list(papers),
            }

        # TODO: maybe weight by topic size for a more meaningful score
        total_correct = sum(
            a["overlap_papers"] for a in alignment.values()
        )
        total_papers = sum(
            a["total_papers_in_topic"] for a in alignment.values()
        )
        overall_score = total_correct / total_papers if total_papers > 0 else 0

        return {
            "topic_alignment": alignment,
            "overall_alignment_score": overall_score,
            "interpretation": (
                f"BERTopic topics align {overall_score:.1%} with IKEA taxonomy. "
                f"{'Strong' if overall_score > 0.7 else 'Moderate' if overall_score > 0.4 else 'Weak'} "
                f"independent validation of the 5-category taxonomy."
            ),
        }

    def get_topic_representation(self, topic_id: int) -> list[tuple]:
        """Get the top words for a specific topic."""
        if self.topic_model is None:
            return []
        return self.topic_model.get_topic(topic_id)

    def visualize_topics(self) -> Optional[object]:
        """Generate an interactive topic visualization.

        Returns a Plotly figure if BERTopic is fitted.
        """
        if self.topic_model is None:
            return None
        try:
            return self.topic_model.visualize_topics()
        except Exception as e:
            logger.warning(f"Topic visualization failed: {e}")
            return None

    def visualize_hierarchy(self) -> Optional[object]:
        """Generate a hierarchical topic visualization."""
        if self.topic_model is None:
            return None
        try:
            return self.topic_model.visualize_hierarchy()
        except Exception as e:
            logger.warning(f"Hierarchy visualization failed: {e}")
            return None
