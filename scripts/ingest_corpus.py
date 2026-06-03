"""CLI: Parse PDFs -> JSON. Ingestion pipeline for the 20 IKEA buying guides."""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CORPUS_RAW, CORPUS_PROCESSED
from src.ingestion.pdf_parser import parse_all_pdfs
from src.ingestion.text_cleaner import clean_paper_dict
from src.ingestion.section_detector import add_sections_to_paper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("FurnishRAG Corpus Ingestion Pipeline")
    logger.info("=" * 60)

    logger.info(f"\nParsing PDFs from {CORPUS_RAW}")
    raw_papers = parse_all_pdfs(CORPUS_RAW, CORPUS_PROCESSED)

    if not raw_papers:
        logger.error(
            f"No PDFs found in {CORPUS_RAW}. "
            f"Please copy the IKEA buying guides to this directory."
        )
        logger.info(
            f"Expected files: D01_*.pdf through D20_*.pdf"
        )
        sys.exit(1)

    logger.info(f"\nCleaning text ({len(raw_papers)} documents)")
    cleaned_papers = []
    for paper in raw_papers:
        cleaned = clean_paper_dict(paper)
        cleaned_papers.append(cleaned)

    logger.info(f"\nDetecting sections")
    processed_papers = []
    for paper in cleaned_papers:
        processed = add_sections_to_paper(paper)
        processed_papers.append(processed)

    logger.info(f"\nSaving processed documents to {CORPUS_PROCESSED}")
    CORPUS_PROCESSED.mkdir(parents=True, exist_ok=True)
    for paper in processed_papers:
        out_path = CORPUS_PROCESSED / f"{paper['paper_id']}.json"
        # Remove raw text fields to save space (keep cleaned only)
        save_data = {k: v for k, v in paper.items() if k != "full_text_raw"}
        for page in save_data.get("pages", []):
            page.pop("text_raw", None)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        logger.info(f"  Saved {out_path.name}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Ingestion Summary")
    logger.info("=" * 60)
    total_words = 0
    for paper in processed_papers:
        wc = paper.get("word_count", len(paper["full_text"].split()))
        sections = len(paper.get("sections", []))
        total_words += wc
        logger.info(
            f"  {paper['paper_id']}: {wc:,} words, {sections} sections, "
            f"{paper['num_pages']} pages"
        )
    logger.info(f"\nTotal: {len(processed_papers)} documents, {total_words:,} words")
    logger.info("Done!")


if __name__ == "__main__":
    main()
