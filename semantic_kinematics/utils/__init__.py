"""Shared utilities for text extraction, timestamps, and cleaning."""

from .timestamps import (
    parse_timestamp_to_utc,
    parse_html_timestamp,
    to_iso8601,
    timestamp_to_month,
)

from .text_cleaning import (
    clean_text_for_analysis,
    count_exclamations,
    normalize_whitespace,
    extract_words,
    jaccard_similarity,
)

from .html_extract import (
    extract_text_from_html,
    extract_with_provenance,
    build_message_with_provenance,
)

__all__ = [
    # timestamps
    'parse_timestamp_to_utc',
    'parse_html_timestamp',
    'to_iso8601',
    'timestamp_to_month',
    # text_cleaning
    'clean_text_for_analysis',
    'count_exclamations',
    'normalize_whitespace',
    'extract_words',
    'jaccard_similarity',
    # html_extract
    'extract_text_from_html',
    'extract_with_provenance',
    'build_message_with_provenance',
]
