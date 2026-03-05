"""
Text cleaning and analysis utilities.

Consolidates text processing from f1, f2, f3 scripts.
"""

import re
from typing import Dict


def clean_text_for_analysis(text: str) -> str:
    """
    Remove code blocks, quotes, and technical artifacts.

    Used for exclamation analysis and other pattern detection
    where code/markup should be excluded.
    """
    if not text:
        return ""

    # Remove code blocks (triple backticks)
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Remove inline code
    text = re.sub(r'`[^`]+`', '', text)
    # Remove indented code blocks
    text = re.sub(r'^    .*$', '', text, flags=re.MULTILINE)

    # Remove HTML/XML tags
    text = re.sub(r'<![^>]*>', '', text)  # Comments, DOCTYPE
    text = re.sub(r'<[^>]+>', '', text)   # Tags

    # Remove shebang lines
    text = re.sub(r'^#!/.*$', '', text, flags=re.MULTILINE)

    # Remove markdown quotes
    text = re.sub(r'>\s*[^<\n]+', '', text)

    # Remove bash history/variable artifacts that look like exclamations
    text = re.sub(r'\$\s*[!][!$^]', '', text)

    return text


def count_exclamations(text: str) -> Dict[str, int]:
    """
    Count exclamations with position classification.

    Returns:
        Dict with keys: total, opening, closing, mid
    """
    if not text:
        return {'total': 0, 'opening': 0, 'closing': 0, 'mid': 0}

    text = clean_text_for_analysis(text)
    total = text.count('!')

    # Opening: exclamation appears in first sentence
    opening_match = re.search(r'^[^.!?]*!', text.strip(), re.MULTILINE)
    opening = 1 if opening_match else 0

    # Closing: text ends with exclamation
    closing = 1 if text.rstrip().endswith('!') else 0

    # Mid: everything else
    mid = total - opening - (closing if opening != closing else 0)

    return {'total': total, 'opening': opening, 'closing': closing, 'mid': mid}


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace for text comparison.

    - Collapses multiple spaces/newlines to single space
    - Strips leading/trailing whitespace
    """
    if not text:
        return ""
    return ' '.join(text.split())


def extract_words(text: str) -> set:
    """
    Extract word set for Jaccard similarity.

    Lowercases and removes punctuation for fuzzy matching.
    """
    if not text:
        return set()
    # Remove punctuation, lowercase, split
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    return set(cleaned.split())


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts.

    Returns:
        Float 0-1 (1 = identical word sets)
    """
    words1 = extract_words(text1)
    words2 = extract_words(text2)

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0
