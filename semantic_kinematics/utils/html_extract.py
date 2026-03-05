"""
HTML extraction utilities.

Consolidates HTML→text extraction from a1, b2 scripts with
provenance preservation per ADR-001.
"""

from typing import Optional, Tuple
from bs4 import BeautifulSoup


def extract_text_from_html(
    html_content: str,
    preserve_structure: bool = True
) -> str:
    """
    Extract plain text from HTML content.

    Args:
        html_content: Raw HTML string
        preserve_structure: If True, preserve paragraph breaks

    Returns:
        Extracted plain text
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, 'html.parser')

    if not preserve_structure:
        return soup.get_text(separator=' ', strip=True)

    # Process block elements to preserve paragraph structure
    blocks = []
    for elem in soup.find_all(['p', 'ul', 'ol', 'div', 'blockquote', 'pre'], recursive=False):
        if elem.name in ('ul', 'ol'):
            # Format list items with bullets
            items = []
            for li in elem.find_all('li', recursive=False):
                items.append(f"• {li.get_text(strip=True)}")
            if items:
                blocks.append('\n\n'.join(items))
        else:
            text = elem.get_text(strip=True)
            if text:
                blocks.append(text)

    if blocks:
        return "\n\n".join(blocks)
    else:
        # Fallback for content without block tags
        return soup.get_text(separator='\n', strip=True)


def extract_with_provenance(
    html_content: str,
    source_field: str = "unknown"
) -> Tuple[str, dict]:
    """
    Extract text with provenance metadata.

    Per ADR-001, preserves both extracted text and source HTML
    for verification.

    Args:
        html_content: Raw HTML string
        source_field: JSON path to source (e.g., "safeHtmlItem[0].html")

    Returns:
        Tuple of (extracted_text, extraction_metadata)

    Example:
        text, meta = extract_with_provenance(html, "safeHtmlItem[0].html")
        # meta = {
        #     "version": "1.0",
        #     "method": "beautifulsoup_block_extraction",
        #     "source_field": "safeHtmlItem[0].html",
        #     "text_html": "<p>Original...</p>"
        # }
    """
    from datetime import datetime, timezone

    text = extract_text_from_html(html_content, preserve_structure=True)

    extraction_meta = {
        "version": "1.0",
        "method": "beautifulsoup_block_extraction",
        "source_field": source_field,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    return text, extraction_meta


def build_message_with_provenance(
    message_id: str,
    timestamp: str,
    speaker: str,
    html_content: Optional[str],
    plain_text: Optional[str] = None,
    source_field: str = "unknown",
    source_file: str = "unknown",
    source_format: str = "unknown",
    preserve_html: bool = True
) -> dict:
    """
    Build a message dict with full provenance.

    If html_content is provided, extracts text and stores both.
    If only plain_text is provided (e.g., for User messages), uses that directly.

    Args:
        message_id: Unique message identifier
        timestamp: ISO 8601 timestamp
        speaker: "User", "Gemini", or "Thinking"
        html_content: Raw HTML (for AI responses)
        plain_text: Pre-extracted text (for User messages or fallback)
        source_field: JSON path to HTML source
        source_file: Source filename
        source_format: "takeout_html", "takeout_json", "webui"
        preserve_html: Whether to store text_html field

    Returns:
        Message dict per ADR-001 schema
    """
    if html_content:
        text, extraction_meta = extract_with_provenance(html_content, source_field)
    else:
        text = plain_text or ""
        extraction_meta = None

    msg = {
        "message_id": message_id,
        "timestamp": timestamp,
        "timestamp_source": source_format,
        "speaker": speaker,
        "text": text,
        "source_file": source_file,
        "source_format": source_format,
        "has_thinking": False,
        "thinking_blocks": None,
    }

    # Add provenance fields for AI responses
    if html_content and preserve_html:
        msg["text_html"] = html_content
        msg["extraction"] = extraction_meta

    return msg
