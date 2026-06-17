"""Lossless phrase segmentation for the bearing/displacement atom.

Splits raw text into :class:`Phrase` units at demarcators -- terminal/internal
punctuation, dashes/ellipses, bare newlines, and paired set-delimiters
(quotes/parens/brackets). Each phrase carries its trailing demarcator and
whitespace, so concatenating every ``raw`` reconstructs the input byte-for-byte:

    "".join(p.raw for p in segment(text)) == text   # for ANY input

This is a hard invariant. The classifier may make taxonomy choices (e.g. how it
labels an ambiguous apostrophe), but it MUST NOT mutate, drop, or reorder bytes.
When delimiter structure is ambiguous or unbalanced (lone quote, nested parens),
the scanner falls back to LINEAR handling -- it treats the character as ordinary
content rather than crash or break the invariant.

Classification normalizes ``\\r\\n`` -> ``\\n`` for deciding the whitespace_class
of a boundary, but the original characters are preserved verbatim in ``raw``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

# ---------------------------------------------------------------------------
# Demarcator class constants. The 7 boundary classes from the spec, plus NONE
# for a trailing fragment that ends the text with no boundary of its own --
# forcing such a tail into a boundary class would be a lie about its structure.
# ---------------------------------------------------------------------------
TERM_ISOLATED = "TERM_ISOLATED"  # terminal punct + trailing ws containing a newline
TERM_FLOW = "TERM_FLOW"          # terminal punct + space/tab only (no newline)
INTERNAL = "INTERNAL"            # , ; :
DASH_ELLIP = "DASH_ELLIP"        # em/en dash, unicode ellipsis, ASCII ... (3+)
BREAK_BARE = "BREAK_BARE"        # newline boundary with no preceding punctuation
SET_QUOTE = "SET_QUOTE"          # boundary at a " or clear ' quote (open or close)
SET_PAREN = "SET_PAREN"         # boundary at ( ) [ ]
NONE = "NONE"                    # no demarcator (trailing fragment / ws-only tail)

# whitespace_class values
WS_SPACE = "space"
WS_NEWLINE = "newline"
WS_PARAGRAPH = "paragraph"
WS_NONE = "none"

_TERMINAL = set(".!?")
_INTERNAL = set(",;:")
_DASHES = set("—–")          # em dash, en dash
_ELLIPSIS_CHAR = "…"              # single-codepoint ellipsis
_OPEN_PAREN = set("([")
_CLOSE_PAREN = set(")]")
_WS = set(" \t")  # horizontal whitespace; newlines handled explicitly


@dataclass
class Phrase:
    raw: str                # content + trailing demarcator + trailing whitespace (LOSSLESS unit)
    content: str            # phrase text, leading/trailing whitespace + trailing demarcator stripped
    demarcator_class: str   # one of the 7 class constants above
    whitespace_class: str   # "space" | "newline" | "paragraph" | "none"


# ---------------------------------------------------------------------------
# Whitespace classification (operates on a trailing-whitespace run).
# ---------------------------------------------------------------------------
def _classify_whitespace(ws: str) -> str:
    """Classify a trailing-whitespace run. ``\\r\\n`` normalized for counting."""
    if not ws:
        return WS_NONE
    norm = ws.replace("\r\n", "\n").replace("\r", "\n")
    if "\n\n" in norm:
        return WS_PARAGRAPH
    if "\n" in norm:
        return WS_NEWLINE
    return WS_SPACE


def _contains_newline(ws: str) -> bool:
    return "\n" in ws or "\r" in ws


# ---------------------------------------------------------------------------
# Apostrophe vs quote disambiguation for the single quote `'`.
# ---------------------------------------------------------------------------
def _is_quote_open(text: str, i: int) -> bool:
    """A `'` is a quote-OPEN only when clearly quotative.

    Preconditions: preceded by start-of-text, whitespace, or an open paren; and
    followed by a non-whitespace, non-EOF character (it opens a span). Otherwise
    it is an apostrophe (or trailing punctuation) -> treat as content.
    """
    prev = text[i - 1] if i > 0 else ""
    nxt = text[i + 1] if i + 1 < len(text) else ""
    if not nxt or nxt.isspace():
        return False
    return prev == "" or prev.isspace() or prev in _OPEN_PAREN


def _is_quote_close(text: str, i: int) -> bool:
    """A `'` is a quote-CLOSE only when clearly quotative.

    Preconditions: preceded by a non-whitespace character (closes a span) and
    followed by whitespace, terminal/internal punctuation, a close paren, or EOF.
    A `'` flanked by letters on both sides (it's, don't, Arthur's) is an
    apostrophe -> content.
    """
    prev = text[i - 1] if i > 0 else ""
    nxt = text[i + 1] if i + 1 < len(text) else ""
    if prev == "" or prev.isspace():
        return False
    return nxt == "" or nxt.isspace() or nxt in _TERMINAL or nxt in _INTERNAL or nxt in _CLOSE_PAREN


# ---------------------------------------------------------------------------
# Helpers for emitting phrases.
# ---------------------------------------------------------------------------
def _consume_trailing_ws(text: str, j: int) -> int:
    """From index j, consume a run of whitespace (spaces, tabs, newlines, \\r)."""
    n = len(text)
    while j < n and (text[j] in _WS or text[j] in "\r\n"):
        j += 1
    return j


def _build_phrase(raw: str, content_lo: int, content_hi: int,
                  text: str, demarc_class: str, ws: str) -> Phrase:
    """Construct a Phrase. content is text[content_lo:content_hi] with leading
    and trailing whitespace stripped and (for set-delimited spans) the leading
    open delimiter removed. The trailing close delimiter and trailing whitespace
    are already excluded by the caller's choice of content_hi; this also trims
    any intra-content whitespace that sat between the last word and the boundary
    (e.g. the space before an opening quote)."""
    content = text[content_lo:content_hi].strip()
    # Set-delimited spans keep their opening char in the slice; drop it so the
    # content is the enclosed text only.
    if demarc_class in (SET_QUOTE, SET_PAREN) and content[:1] in ('"', "'", "(", "["):
        content = content[1:].lstrip()
    return Phrase(
        raw=raw,
        content=content,
        demarcator_class=demarc_class,
        whitespace_class=_classify_whitespace(ws),
    )


# ---------------------------------------------------------------------------
# Core scanner.
# ---------------------------------------------------------------------------
def segment(text: str) -> List[Phrase]:
    """Segment ``text`` into lossless :class:`Phrase` units.

    Invariant: ``"".join(p.raw for p in segment(text)) == text``.
    """
    if text == "":
        return []

    n = len(text)
    phrases: List[Phrase] = []

    # `start` = index where the current phrase's raw began (includes any leading
    # whitespace, which attaches to the phrase that follows it -- so for the very
    # first phrase, leading ws of the whole text is part of raw).
    start = 0
    # `content_start` = where the textual content begins (we trim leading ws of
    # content for the `content` field, but raw keeps everything from `start`).
    i = 0

    def flush(boundary_end: int, content_hi: int, demarc_class: str, ws: str) -> int:
        """Emit a phrase covering text[start:boundary_end]. Returns new start."""
        nonlocal start
        raw = text[start:boundary_end]
        phrases.append(_build_phrase(raw, start, content_hi, text, demarc_class, ws))
        start = boundary_end
        return boundary_end

    while i < n:
        ch = text[i]

        # --- Set delimiters: split BEFORE an opener and AFTER a closer. ---
        if ch == '"' or (ch == "'" and _is_quote_open(text, i)):
            # OPEN quote. If there is content before it, close that phrase first
            # (boundary just before the quote, no demarcator of its own -> the
            # preceding phrase keeps whatever class it would naturally get; but
            # here the split is caused BY the quote, so use SET_QUOTE on the
            # opener side too). We treat the opener as starting a new phrase that
            # itself ends at its close quote.
            # Scan to a matching close quote FIRST. Only split if balanced; an
            # unbalanced opener must fall back to linear without having emitted a
            # spurious SET_QUOTE phrase for the preceding text.
            close_char = ch
            j = i + 1
            found = -1
            while j < n:
                cj = text[j]
                if cj == close_char:
                    if close_char == '"':
                        found = j
                        break
                    # single quote: only a clear close counts
                    if _is_quote_close(text, j):
                        found = j
                        break
                j += 1
            if found == -1:
                # Unbalanced -> fall back to linear: treat the opener as content.
                i += 1
                continue
            # Balanced. Emit any preceding text as its own phrase (split at quote).
            if i > start:
                flush(i, i, SET_QUOTE, "")
            # Emit the quoted span (opener..closer inclusive) + trailing ws as raw.
            close_end = found + 1
            ws_end = _consume_trailing_ws(text, close_end)
            ws = text[close_end:ws_end]
            # content excludes the surrounding quote chars.
            flush(ws_end, found, SET_QUOTE, ws)
            i = ws_end
            continue

        if ch in _OPEN_PAREN:
            close_char = ")" if ch == "(" else "]"
            depth = 1
            j = i + 1
            found = -1
            while j < n:
                cj = text[j]
                if cj == ch:
                    depth += 1
                elif cj == close_char:
                    depth -= 1
                    if depth == 0:
                        found = j
                        break
                j += 1
            if found == -1:
                # Unbalanced -> linear fallback: opener is content.
                i += 1
                continue
            if i > start:
                flush(i, i, SET_PAREN, "")
            close_end = found + 1
            ws_end = _consume_trailing_ws(text, close_end)
            ws = text[close_end:ws_end]
            flush(ws_end, found, SET_PAREN, ws)
            i = ws_end
            continue

        if ch in _CLOSE_PAREN or ch == '"':
            # A stray closer with no recorded opener (the opener-balanced path
            # consumed its own closer). Treat as content -> linear.
            i += 1
            continue

        # --- ASCII ellipsis / dot run: distinguish ... (3+) from . terminal. ---
        if ch == ".":
            j = i
            while j < n and text[j] == ".":
                j += 1
            run = j - i
            if run >= 3:
                # Ellipsis -> DASH_ELLIP.
                ws_end = _consume_trailing_ws(text, j)
                ws = text[j:ws_end]
                flush(ws_end, i, DASH_ELLIP, ws)
                i = ws_end
                continue
            # run is 1 or 2 dots -> terminal punctuation; fall through to terminal
            # handling by NOT advancing past the dots here (handled below).

        # --- Terminal punctuation run: . ! ? and combos (?! !? etc). ---
        if ch in _TERMINAL:
            # Gather the contiguous terminal run, but a 3+ dot run is ellipsis
            # (already handled above), so here a dot run is 1-2 dots.
            j = i
            while j < n and text[j] in _TERMINAL:
                # stop a dot sub-run at 3 to leave ellipsis to the branch above;
                # but we already returned for 3+ dots, so just gather all.
                j += 1
            ws_end = _consume_trailing_ws(text, j)
            ws = text[j:ws_end]
            cls = TERM_ISOLATED if _contains_newline(ws) else TERM_FLOW
            flush(ws_end, i, cls, ws)
            i = ws_end
            continue

        # --- Internal punctuation: , ; : ---
        if ch in _INTERNAL:
            j = i + 1
            ws_end = _consume_trailing_ws(text, j)
            ws = text[j:ws_end]
            flush(ws_end, i, INTERNAL, ws)
            i = ws_end
            continue

        # --- Unicode ellipsis char and dashes. ---
        if ch == _ELLIPSIS_CHAR or ch in _DASHES:
            j = i + 1
            ws_end = _consume_trailing_ws(text, j)
            ws = text[j:ws_end]
            flush(ws_end, i, DASH_ELLIP, ws)
            i = ws_end
            continue

        # --- Bare newline boundary (no preceding punctuation). ---
        # Detect \r?\n at this position; consume the whole following ws run.
        if ch == "\n" or (ch == "\r"):
            # Only a bare break if there is actual content before this on the line
            # (i.e. start..i has non-whitespace). If the phrase so far is only
            # whitespace, the newline is leading ws for the NEXT phrase -- just
            # keep scanning so it gets absorbed.
            preceding = text[start:i]
            if preceding.strip() == "":
                # No content yet -> this ws belongs to the upcoming phrase's raw.
                i += 1
                continue
            ws_end = _consume_trailing_ws(text, i)
            ws = text[i:ws_end]
            flush(ws_end, i, BREAK_BARE, ws)
            i = ws_end
            continue

        # ordinary content character
        i += 1

    # --- Trailing remainder (content with no closing demarcator, or trailing ws). ---
    if start < n:
        remainder = text[start:n]
        if remainder.strip() == "":
            # Pure trailing whitespace: attach to the previous phrase if any,
            # else emit a whitespace-only phrase to preserve the invariant.
            if phrases:
                prev = phrases[-1]
                merged_raw = prev.raw + remainder
                phrases[-1] = Phrase(
                    raw=merged_raw,
                    content=prev.content,
                    demarcator_class=prev.demarcator_class,
                    whitespace_class=_classify_whitespace(
                        merged_raw[len(merged_raw.rstrip()):]
                    ),
                )
            else:
                # Whitespace-only input with no phrases: one NONE phrase holds it.
                phrases.append(Phrase(
                    raw=remainder,
                    content="",
                    demarcator_class=NONE,
                    whitespace_class=_classify_whitespace(remainder),
                ))
        else:
            # Content with no trailing demarcator -> NONE class. Classify any
            # trailing whitespace for the ws field.
            stripped = remainder.rstrip()
            trailing_ws = remainder[len(stripped):]
            content_hi = start + len(stripped)
            phrases.append(Phrase(
                raw=remainder,
                content=text[start:content_hi].lstrip(),
                demarcator_class=NONE,
                whitespace_class=_classify_whitespace(trailing_ws),
            ))

    return phrases
