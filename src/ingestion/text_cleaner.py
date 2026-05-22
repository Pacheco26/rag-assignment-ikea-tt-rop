"""Text cleaning and normalization for IKEA buying guide text."""

import re
import unicodedata
import logging

logger = logging.getLogger(__name__)


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters (e.g., smart quotes, ligatures)."""
    text = unicodedata.normalize("NFKD", text)
    # Replace common problematic characters
    replacements = {
        "\u2018": "'", "\u2019": "'",  # smart single quotes
        "\u201c": '"', "\u201d": '"',  # smart double quotes
        "\u2013": "-", "\u2014": "--",  # en/em dashes
        "\u2026": "...",               # ellipsis
        "\ufb01": "fi", "\ufb02": "fl",  # ligatures
        "\u00a0": " ",                 # non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def remove_headers_footers(text: str) -> str:
    """Remove common IKEA buying guide headers, footers, and boilerplate."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip page numbers (standalone numbers)
        if re.match(r"^\d{1,3}$", stripped):
            continue
        # Skip IKEA branding headers/footers
        if re.match(r"^(Inter IKEA|www\.ikea\.|IKEA-USA\.com)", stripped, re.IGNORECASE):
            continue
        # Skip copyright lines
        if re.match(r"^(Copyright|\u00a9|\(c\))\s", stripped, re.IGNORECASE):
            continue
        # Skip "Available at" / "See what is available" boilerplate
        if re.match(r"^(See what is available|Visit IKEA|Shop online at|Find it at)\s", stripped, re.IGNORECASE):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def remove_boilerplate(text: str) -> str:
    """Remove recurring IKEA marketing boilerplate text."""
    patterns = [
        r"A high standard of accuracy has been sought in the preparation of this buying guide\.[^.]*\.",
        r"We apologize\s+for, but will not be bound by or responsible for, errors and omissions in this\s+buying guide\.",
        r"Not all products may be available in all stores\.[^.]*\.",
        r"See what is available at your local store by\s+visiting\s+\S+\.",
        r"For more detailed product information, see the price tag or visit\s+\S+\.",
        r"All products shown require assembly\.",
        r"All textiles shown are imported\.",
        r"All units require assembly\.",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    return text


def remove_price_references(text: str) -> str:
    """Remove price references since they're region/time-dependent."""
    # Match dollar prices: $49, $49.99, $1,299
    text = re.sub(r'\$\d{1,},?\d*(?:\.\d{2})?', '[PRICE]', text)
    return text


def remove_urls(text: str) -> str:
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'IKEA-USA\.com\S*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'IKEA\.com\S*', '', text, flags=re.IGNORECASE)
    return text


def clean_whitespace(text: str) -> str:
    # Collapse multiple blank lines into maximum 2
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    # Remove trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def clean_paper_text(raw_text: str, remove_refs: bool = True) -> str:
    """Full cleaning pipeline for extracted buying guide text."""
    text = normalize_unicode(raw_text)
    text = remove_headers_footers(text)
    text = remove_boilerplate(text)
    text = remove_urls(text)
    text = remove_price_references(text)
    text = clean_whitespace(text)
    return text


def clean_paper_dict(paper_data: dict, remove_refs: bool = True) -> dict:
    """Clean a document data dictionary (from pdf_parser output)."""
    cleaned = paper_data.copy()
    cleaned["full_text_raw"] = paper_data["full_text"]
    cleaned["full_text"] = clean_paper_text(paper_data["full_text"], remove_refs)

    # Also clean individual pages
    for page in cleaned.get("pages", []):
        page["text_raw"] = page["text"]
        page["text"] = clean_paper_text(page["text"], remove_refs=False)

    cleaned["char_count"] = len(cleaned["full_text"])
    cleaned["word_count"] = len(cleaned["full_text"].split())

    logger.info(
        f"{cleaned['paper_id']}: {cleaned['word_count']} words after cleaning"
    )
    return cleaned
