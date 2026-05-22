"""Section detection for IKEA buying guide documents."""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# IKEA buying guide section names (order matters for matching)
STANDARD_SECTIONS = [
    # Product overview
    "Product Description",
    "Product Overview",
    "Product Range",
    "Product Series",
    # Specifications
    "Specifications",
    "Dimensions",
    "Measurements",
    "Size Guide",
    "Sizes",
    # Materials and care
    "Materials",
    "Materials and Care",
    "Care Instructions",
    "Maintenance",
    "Cleaning",
    # Planning and design
    "Planning Tips",
    "Planning Guide",
    "How to Plan",
    "Design Tips",
    "Room Planning",
    "Buying Tips",
    "How to Choose",
    "Getting Started",
    "Step by Step",
    # Assembly and installation
    "Assembly",
    "Assembly Instructions",
    "Installation",
    "Installation Guide",
    # Safety
    "Safety",
    "Safety Information",
    "Important Information",
    "Good to Know",
    "Things to Know",
    # Sustainability
    "Sustainability",
    "Environment",
    "Environmental Profile",
    "Circular Design",
    # Warranty
    "Warranty",
    "Guarantee",
    # Combinations
    "Combinations",
    "Accessories",
    "Compatible Products",
    "Storage Solutions",
    "Interior Fittings",
    "Fronts and Handles",
    "Lighting",
]

# Pattern for numbered sections like "1. Planning Tips"
NUMBERED_SECTION_RE = re.compile(
    r"^(?:\d{1,2}\.?\s+)"           # Arabic numerals only
    r"([A-Z][A-Za-z\s&\-:]+)",      # Section title
    re.MULTILINE,
)

# Pattern for unnumbered sections (all caps or title case on own line)
UNNUMBERED_SECTION_RE = re.compile(
    r"^((?:PRODUCT\s+(?:DESCRIPTION|OVERVIEW|RANGE|SERIES)|"
    r"SPECIFICATIONS?|DIMENSIONS?|MEASUREMENTS?|"
    r"SIZE(?:\s+GUIDE)?|SIZES|"
    r"MATERIALS?(?:\s+AND\s+CARE)?|SUSTAINABILITY|"
    r"PLANNING\s+(?:TIPS|GUIDE)|DESIGN\s+TIPS|ROOM\s+PLANNING|"
    r"BUYING\s+TIPS|GETTING\s+STARTED|STEP\s+BY\s+STEP|"
    r"HOW\s+TO\s+(?:PLAN|CHOOSE|BUY|ASSEMBLE|CARE)|"
    r"ASSEMBLY(?:\s+INSTRUCTIONS)?|INSTALLATION(?:\s+GUIDE)?|"
    r"CARE\s+INSTRUCTIONS|MAINTENANCE|CLEANING|"
    r"SAFETY(?:\s+INFORMATION)?|IMPORTANT\s+INFORMATION|"
    r"GOOD\s+TO\s+KNOW|THINGS\s+TO\s+KNOW|"
    r"WARRANTY|GUARANTEE|"
    r"STORAGE\s+SOLUTIONS|INTERIOR\s+FITTINGS|"
    r"FRONTS\s+AND\s+HANDLES|LIGHTING|"
    r"COMBINATIONS|ACCESSORIES|COMPATIBLE\s+PRODUCTS|"
    r"ENVIRONMENTAL?\s+PROFILE|CIRCULAR\s+DESIGN)"
    r")\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _normalize_section_name(name: str) -> str:
    name = name.strip().rstrip(".")
    # Handle all-caps
    if name.isupper():
        name = name.title()
    return name


def detect_sections(text: str) -> list[dict]:
    """Detect section boundaries in buying guide text."""
    # Find all section headers
    headers = []

    # Numbered sections (e.g., "1. Planning Tips")
    for m in NUMBERED_SECTION_RE.finditer(text):
        title = _normalize_section_name(m.group(1))
        headers.append({
            "section": title,
            "start": m.start(),
            "header_end": m.end(),
        })

    # Unnumbered sections (e.g., "SPECIFICATIONS", "GOOD TO KNOW")
    for m in UNNUMBERED_SECTION_RE.finditer(text):
        title = _normalize_section_name(m.group(1))
        # Avoid duplicates from numbered matches
        is_duplicate = any(
            abs(h["start"] - m.start()) < 20 for h in headers
        )
        if not is_duplicate:
            headers.append({
                "section": title,
                "start": m.start(),
                "header_end": m.end(),
            })

    # Sort by position
    headers.sort(key=lambda h: h["start"])

    if not headers:
        logger.warning("No sections detected, returning full text as one section")
        return [{
            "section": "Full Text",
            "start": 0,
            "end": len(text),
            "text": text,
        }]

    # Build sections with text content
    sections = []
    for i, header in enumerate(headers):
        start = header["header_end"]
        end = headers[i + 1]["start"] if i + 1 < len(headers) else len(text)
        section_text = text[start:end].strip()

        if section_text:  # skip empty sections
            sections.append({
                "section": header["section"],
                "start": header["start"],
                "end": end,
                "text": section_text,
            })

    logger.info(f"Detected {len(sections)} sections: {[s['section'] for s in sections]}")
    return sections


def add_sections_to_paper(paper_data: dict) -> dict:
    """Add detected sections to a document data dictionary."""
    updated = paper_data.copy()
    updated["sections"] = detect_sections(paper_data["full_text"])
    logger.info(
        f"{paper_data['paper_id']}: {len(updated['sections'])} sections detected"
    )
    return updated
