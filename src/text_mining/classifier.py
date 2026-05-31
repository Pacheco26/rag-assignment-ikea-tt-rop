"""Classifies text into IKEA product taxonomy categories using embedding similarity."""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Category descriptions for IKEA product taxonomy
TAXONOMY_CATEGORIES = {
    "Storage & Organization": (
        "Shelving units, bookcases, modular storage, display cabinets, "
        "wall-mounted shelves, storage inserts, drawers, baskets, boxes, "
        "BILLY bookcase, KALLAX shelf unit, BESTA TV storage, EKET cabinet, "
        "organizing, compartments, room divider"
    ),
    "Bedroom & Wardrobe": (
        "Beds, bed frames, mattresses, pillows, wardrobe systems, closet organizers, "
        "PAX wardrobe, HEMNES bedroom furniture, nightstands, dressers, "
        "sleep quality, firmness, spring mattress, foam mattress, comfort zones"
    ),
    "Kitchen & Dining": (
        "Kitchen cabinets, kitchen planning, countertops, sinks, faucets, "
        "SEKTION cabinet system, kitchen fronts, kitchen island, drawer organizers, "
        "KUNGSFORS open storage, kitchen rail system, cooking, dining"
    ),
    "Living & Outdoor": (
        "Sofas, armchairs, coffee tables, living room furniture, outdoor furniture, "
        "KIVIK sofa, EKTORP sofa, HEMNES living room, BONDHOLMEN outdoor series, "
        "cushions, covers, patio, garden furniture, weather-resistant"
    ),
    "Home & Sustainability": (
        "Interior design, style guide, home decoration, sustainability, circular design, "
        "laundry solutions, bathroom installation, cleaning, washing, drying, "
        "renewable materials, recycled, FSC-certified, environmental impact"
    ),
}


class TaxonomyClassifier:
    """Embedding-based classifier for the 5-category IKEA taxonomy."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self._category_embeddings: Optional[np.ndarray] = None
        self._category_names: list[str] = []
        self._init_category_embeddings()

    def _init_category_embeddings(self):
        self._category_names = list(TAXONOMY_CATEGORIES.keys())
        descriptions = list(TAXONOMY_CATEGORIES.values())
        self._category_embeddings = self.model.encode(
            descriptions, show_progress_bar=False, normalize_embeddings=True,
        )
        logger.info(f"Initialized taxonomy classifier with {len(self._category_names)} categories")

    def classify_text(self, text: str) -> dict:
        """Classify a text into the most likely taxonomy category."""
        text_embedding = self.model.encode(
            [text], show_progress_bar=False, normalize_embeddings=True,
        )

        # Cosine similarity (embeddings are normalized, so dot product = cosine)
        similarities = np.dot(self._category_embeddings, text_embedding.T).flatten()

        # Softmax for interpretable scores
        exp_scores = np.exp(similarities - np.max(similarities))
        probabilities = exp_scores / exp_scores.sum()

        best_idx = int(np.argmax(probabilities))
        scores = {
            name: float(prob)
            for name, prob in zip(self._category_names, probabilities)
        }

        return {
            "category": self._category_names[best_idx],
            "confidence": float(probabilities[best_idx]),
            "scores": scores,
        }

    def classify_chunks(self, chunks: list[dict]) -> list[dict]:
        """Classify multiple chunks, returns one result per chunk."""
        results = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if not text:
                results.append({"category": "Unknown", "confidence": 0.0, "scores": {}})
                continue
            results.append(self.classify_text(text))
        return results

    def classify_retrieved_results(self, retrieved_chunks: list[dict]) -> dict:
        """Classify retrieved chunks and compute category distribution."""
        classifications = self.classify_chunks(retrieved_chunks)

        # Aggregate: category distribution across retrieved chunks
        category_counts = {}
        for cls in classifications:
            cat = cls["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        total = len(classifications)
        category_distribution = {
            cat: count / total for cat, count in category_counts.items()
        } if total > 0 else {}

        # Dominant category
        dominant = max(category_counts, key=category_counts.get) if category_counts else "Unknown"

        return {
            "per_chunk": classifications,
            "category_distribution": category_distribution,
            "dominant_category": dominant,
            "num_chunks": total,
        }
