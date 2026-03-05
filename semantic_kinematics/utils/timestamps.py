"""
Timestamp parsing and normalization utilities.

Consolidates timestamp handling from f2, f3, h1, h2, a1 scripts.
"""

import re
from datetime import datetime, timezone
from typing import Optional


def parse_timestamp_to_utc(timestamp_str: Optional[str]) -> Optional[datetime]:
    """
    Parse timestamp and normalize to UTC.

    Handles various formats:
    - ISO 8601 with Z (UTC): "2023-10-30T02:19:40Z"
    - ISO 8601 with timezone: "2023-10-30T02:19:40+00:00"
    - Space-separated: "2023-10-30 02:19:40"

    Returns:
        datetime with UTC timezone, or None if parsing fails
    """
    if not timestamp_str:
        return None

    try:
        # ISO 8601 with Z or timezone
        if 'T' in timestamp_str:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            # Space-separated (assume UTC if no timezone)
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

        # Ensure timezone-aware (assume UTC if naive)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt
    except Exception:
        return None


def parse_html_timestamp(timestamp_str: str) -> str:
    """
    Parse Google Takeout HTML timestamp format to ISO 8601.

    Input:  "Sep 15, 2024, 6:08:45 PM MDT"
    Output: "2024-09-15T18:08:45Z"

    Note: Timezone is stripped (not converted to UTC).
    """
    # Remove timezone suffix (MDT, MST, etc.)
    timestamp_str = re.sub(r'\s+[A-Z]{3,4}$', '', timestamp_str.strip())

    try:
        dt = datetime.strptime(timestamp_str, "%b %d, %Y, %I:%M:%S %p")
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return "1970-01-01T00:00:00Z"


def to_iso8601(dt: datetime) -> str:
    """Convert datetime to ISO 8601 string with Z suffix."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    utc_dt = dt.astimezone(timezone.utc)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp_to_month(timestamp_str: str) -> Optional[str]:
    """
    Extract YYYY-MM from timestamp for monthly aggregation.

    Returns:
        "2024-03" or None if parsing fails
    """
    dt = parse_timestamp_to_utc(timestamp_str)
    if dt:
        return dt.strftime("%Y-%m")
    return None
