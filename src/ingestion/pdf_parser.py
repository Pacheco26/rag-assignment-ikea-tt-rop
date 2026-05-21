"""PDF text extraction using PyMuPDF (fitz).

Handles IKEA buying guide layouts including multi-column and product pages.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> dict:
    """Extract text from a PDF file page by page."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Use sort=True to handle multi-column layouts correctly
        text = page.get_text("text", sort=True)
        pages.append({
            "page_number": page_num + 1,
            "text": text,
        })

    full_text = "\n\n".join(p["text"] for p in pages)

    # Extract document ID from filename (e.g., D01_billy_bookcase -> D01)
    doc_id = pdf_path.stem.split("_")[0]
    num_pages = len(doc)

    doc.close()

    result = {
        "paper_id": doc_id,
        "filename": pdf_path.name,
        "num_pages": num_pages,
        "pages": pages,
        "full_text": full_text,
    }

    logger.info(f"Extracted {num_pages} pages from {pdf_path.name}")
    return result


def parse_all_pdfs(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "D*.pdf",
) -> list[dict]:
    """Parse all PDFs in a directory and save as JSON.

    Args:
        input_dir: Directory containing PDF files.
        output_dir: Directory to save extracted JSON files.
        pattern: Glob pattern to match PDF files.

    Returns:
        List of extraction results.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob(pattern))
    if not pdf_files:
        logger.warning(f"No PDFs found matching {pattern} in {input_dir}")
        return []

    results = []
    for pdf_path in pdf_files:
        logger.info(f"Processing {pdf_path.name}...")
        try:
            result = extract_text_from_pdf(pdf_path)
            # Save individual JSON
            out_path = output_dir / f"{result['paper_id']}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            results.append(result)
            logger.info(
                f"  -> {result['num_pages']} pages, "
                f"{len(result['full_text'])} chars"
            )
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")

    logger.info(f"Processed {len(results)}/{len(pdf_files)} PDFs")
    return results
