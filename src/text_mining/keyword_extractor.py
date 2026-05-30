"""KeyBERT keyword extraction with MMR diversity."""

import logging
from typing import Optional

from keybert import KeyBERT

from config.settings import (
    KEYBERT_MODEL,
    KEYPHRASE_NGRAM_RANGE,
    KEYPHRASE_TOP_N,
    MMR_DIVERSITY,
)

logger = logging.getLogger(__name__)


class KeywordExtractor:
    """KeyBERT-based keyword extraction for documents and corpus."""

    def __init__(
        self,
        model_name: str = KEYBERT_MODEL,
        top_n: int = KEYPHRASE_TOP_N,
        ngram_range: tuple = KEYPHRASE_NGRAM_RANGE,
        diversity: float = MMR_DIVERSITY,
    ):
        self.model = KeyBERT(model_name)
        self.top_n = top_n
        self.ngram_range = ngram_range
        self.diversity = diversity

    def extract_keywords(
        self,
        text: str,
        top_n: Optional[int] = None,
        use_mmr: bool = True,
    ) -> list[dict]:
        """Extract keywords from text. Returns list of {keyword, score} dicts."""
        top_n = top_n or self.top_n

        keywords = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=self.ngram_range,
            stop_words="english",
            top_n=top_n,
            use_mmr=use_mmr,
            diversity=self.diversity if use_mmr else 0,
        )

        return [{"keyword": kw, "score": float(score)} for kw, score in keywords]

    def extract_paper_keywords(self, paper_data: dict) -> dict:
        """Extract global + per-section keywords from a single document."""
        paper_id = paper_data["paper_id"]

        # Global keywords from full text
        global_keywords = self.extract_keywords(paper_data["full_text"][:10000])

        # Per-section keywords
        section_keywords = {}
        for section in paper_data.get("sections", []):
            if len(section["text"]) < 100:
                continue
            kws = self.extract_keywords(section["text"][:5000], top_n=5)
            section_keywords[section["section"]] = kws

        logger.info(
            f"{paper_id}: {len(global_keywords)} global keywords, "
            f"{len(section_keywords)} sections analyzed"
        )

        return {
            "paper_id": paper_id,
            "global_keywords": global_keywords,
            "section_keywords": section_keywords,
        }

    def extract_corpus_keywords(
        self,
        papers: list[dict],
        top_n: int = 20,
    ) -> dict:
        """Extract keywords across the entire corpus.

        Args:
            papers: List of document data dicts.
            top_n: Number of corpus-level keywords.

        Returns:
            Dict with corpus keywords and per-document keywords.
        """
        # Corpus-level: concatenate abstracts/introductions
        corpus_text = ""
        for paper in papers:
            sections = paper.get("sections", [])
            for section in sections:
                if section["section"].lower() in ["abstract", "introduction"]:
                    corpus_text += " " + section["text"]

        if not corpus_text.strip():
            corpus_text = " ".join(p["full_text"][:2000] for p in papers)

        corpus_keywords = self.extract_keywords(corpus_text[:15000], top_n=top_n)

        # Per-document keywords
        paper_keywords = {}
        for paper in papers:
            result = self.extract_paper_keywords(paper)
            paper_keywords[paper["paper_id"]] = result

        # Find unique vs shared keywords across documents
        all_kw_sets = {}
        for pid, data in paper_keywords.items():
            kw_set = {kw["keyword"].lower() for kw in data["global_keywords"]}
            all_kw_sets[pid] = kw_set

        # Keywords unique to each document
        unique_keywords = {}
        for pid, kws in all_kw_sets.items():
            others = set()
            for other_pid, other_kws in all_kw_sets.items():
                if other_pid != pid:
                    others |= other_kws
            unique_keywords[pid] = list(kws - others)

        # Shared keywords (appear in 3+ documents)
        from collections import Counter
        all_kws = [kw for kws in all_kw_sets.values() for kw in kws]
        kw_counts = Counter(all_kws)
        shared_keywords = [kw for kw, count in kw_counts.most_common() if count >= 3]

        return {
            "corpus_keywords": corpus_keywords,
            "paper_keywords": paper_keywords,
            "unique_keywords": unique_keywords,
            "shared_keywords": shared_keywords,
        }
