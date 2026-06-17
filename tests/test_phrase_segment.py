"""Tests for the lossless phrase segmenter.

All pure logic, NO network. The load-bearing property is the lossless invariant:
``"".join(p.raw for p in segment(text)) == text`` for any input. Every other
test asserts a taxonomy decision on top of that floor.

One test loads the real specimen ``data/absurdism/bypass_dialogue.txt`` only to
confirm segment() does not crash on it and the invariant still holds.
"""

import os

import pytest

from semantic_kinematics.bearing.phrase_segment import (
    BREAK_BARE,
    DASH_ELLIP,
    INTERNAL,
    NONE,
    Phrase,
    SET_PAREN,
    SET_QUOTE,
    TERM_FLOW,
    TERM_ISOLATED,
    segment,
)

SPECIMEN = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "data", "absurdism", "bypass_dialogue.txt",
)


def _lossless(text: str):
    phrases = segment(text)
    assert "".join(p.raw for p in phrases) == text
    return phrases


# ---------------------------------------------------------------------------
# The hard invariant: lossless on a spread of inputs incl. \r\n.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "\n\n",
        "Just one sentence with no terminal",
        "Hello, world. How are you?",
        "First.\nSecond.\n\nThird.",
        "Line one\nLine two\nLine three",
        'He said "stop" and left.',
        "An aside (like this) follows.",
        "It's Arthur's, don't you think?",
        "Wait... what?!",
        "Dash — here, and an en–dash too.",
        "Windows\r\nline\r\nendings\r\nhere.",
        "Trailing space then nothing   ",
        "Mixed\r\nand\nbare breaks.\n",
    ],
)
def test_lossless_invariant(text):
    _lossless(text)


def test_lossless_crlf_preserves_original_bytes():
    text = "First.\r\nSecond.\r\n\r\nThird."
    phrases = _lossless(text)
    # Original \r chars survive verbatim in raw.
    assert "\r\n" in "".join(p.raw for p in phrases)


# ---------------------------------------------------------------------------
# One-liner isolation: terminal punct + newline -> TERM_ISOLATED.
# ---------------------------------------------------------------------------
def test_oneliner_isolation_term_isolated():
    text = "Something happened.\nWith a torch.\nThen silence.\n"
    phrases = _lossless(text)
    torch = [p for p in phrases if p.content == "With a torch"]
    assert len(torch) == 1
    p = torch[0]
    assert p.demarcator_class == TERM_ISOLATED
    assert p.raw == "With a torch.\n"
    assert p.whitespace_class == "newline"


def test_terminal_flow_vs_isolated():
    # space-trailing terminal -> TERM_FLOW; newline-trailing -> TERM_ISOLATED.
    text = "One. Two.\nThree."
    phrases = _lossless(text)
    assert phrases[0].demarcator_class == TERM_FLOW
    assert phrases[1].demarcator_class == TERM_ISOLATED


# ---------------------------------------------------------------------------
# Bare newline split (BREAK_BARE) with no punctuation.
# ---------------------------------------------------------------------------
def test_bare_newline_split():
    text = "alpha\nbeta\ngamma"
    phrases = _lossless(text)
    assert phrases[0].content == "alpha"
    assert phrases[0].demarcator_class == BREAK_BARE
    assert phrases[1].content == "beta"
    assert phrases[1].demarcator_class == BREAK_BARE
    # last has no trailing demarcator
    assert phrases[2].content == "gamma"
    assert phrases[2].demarcator_class == NONE


def test_single_vs_double_newline_whitespace_class():
    single = _lossless("a\nb")
    assert single[0].whitespace_class == "newline"
    double = _lossless("a\n\nb")
    assert double[0].whitespace_class == "paragraph"


# ---------------------------------------------------------------------------
# Parenthetical aside as its own unit (SET_PAREN).
# ---------------------------------------------------------------------------
def test_parenthetical_is_own_unit():
    text = "The hero (a tired man) sighed."
    phrases = _lossless(text)
    aside = [p for p in phrases if p.demarcator_class == SET_PAREN]
    assert any("a tired man" in p.content for p in aside)
    # bracket form too
    text2 = "See [note] for details."
    phrases2 = _lossless(text2)
    assert any(p.demarcator_class == SET_PAREN and "note" in p.content
               for p in phrases2)


# ---------------------------------------------------------------------------
# Quoted span as its own unit (SET_QUOTE).
# ---------------------------------------------------------------------------
def test_quoted_span_is_own_unit():
    text = 'She whispered "go now" softly.'
    phrases = _lossless(text)
    quoted = [p for p in phrases if p.demarcator_class == SET_QUOTE]
    # content excludes the surrounding quote chars and bordering whitespace.
    assert any(p.content == "go now" for p in quoted)


def test_set_span_content_excludes_delimiters_and_ws():
    phrases = segment("The hero (a tired man) sighed.")
    paren = [p for p in phrases if p.demarcator_class == SET_PAREN]
    assert any(p.content == "a tired man" for p in paren)
    # the lead-in phrase has its trailing space stripped from content but kept in raw
    lead = [p for p in phrases if p.content == "The hero"]
    assert lead and lead[0].raw == "The hero "


# ---------------------------------------------------------------------------
# Apostrophe NOT split.
# ---------------------------------------------------------------------------
def test_apostrophe_not_split():
    text = "don't worry, it's Arthur's hat"
    phrases = _lossless(text)
    # No SET_QUOTE boundary should be created by the in-word apostrophes.
    assert all(p.demarcator_class != SET_QUOTE for p in phrases)
    # don't / it's / Arthur's survive intact within some phrase content.
    joined = " ".join(p.content for p in phrases)
    assert "don't" in joined
    assert "it's" in joined
    assert "Arthur's" in joined


def test_single_quote_clear_quotation_splits():
    text = "He said 'go now' loudly."
    phrases = _lossless(text)
    quoted = [p for p in phrases if p.demarcator_class == SET_QUOTE]
    assert any("go now" in p.content for p in quoted)


# ---------------------------------------------------------------------------
# Unbalanced delimiters: linear fallback, no crash, invariant holds.
# ---------------------------------------------------------------------------
def test_unbalanced_double_quote_falls_back():
    text = 'He said "stop and never closed it.'
    phrases = _lossless(text)  # asserts no crash + lossless
    assert all(p.demarcator_class != SET_QUOTE for p in phrases)


def test_unbalanced_paren_falls_back():
    text = "Open (paren but no close here."
    phrases = _lossless(text)
    assert all(p.demarcator_class != SET_PAREN for p in phrases)


def test_stray_close_delimiter_falls_back():
    text = "Stray close paren) in the middle."
    _lossless(text)  # just must not crash and stay lossless


# ---------------------------------------------------------------------------
# ASCII ... -> DASH_ELLIP; single . -> terminal.
# ---------------------------------------------------------------------------
def test_ascii_ellipsis_vs_terminal_dot():
    text = "Wait... what."
    phrases = _lossless(text)
    assert phrases[0].demarcator_class == DASH_ELLIP
    assert phrases[0].content == "Wait"
    # the final '.' is terminal, not ellipsis
    assert phrases[-1].demarcator_class in (TERM_FLOW, TERM_ISOLATED, NONE)
    assert "what" in phrases[-1].content


def test_four_dots_is_ellipsis():
    phrases = _lossless("done.... next.")
    assert phrases[0].demarcator_class == DASH_ELLIP


def test_unicode_ellipsis_and_dashes():
    phrases = _lossless("here… and — there.")
    classes = [p.demarcator_class for p in phrases]
    assert DASH_ELLIP in classes


def test_internal_punctuation():
    phrases = _lossless("first, second; third: done.")
    internal = [p for p in phrases if p.demarcator_class == INTERNAL]
    assert len(internal) == 3


# ---------------------------------------------------------------------------
# Real specimen: must not crash; invariant holds.
# ---------------------------------------------------------------------------
def test_real_specimen_runs_and_is_lossless():
    with open(SPECIMEN, "r", encoding="utf-8") as fh:
        text = fh.read()
    phrases = segment(text)
    assert "".join(p.raw for p in phrases) == text
    assert len(phrases) > 0
    # Specimen phrase count (informational): ~hundreds of phrases on this passage.
    # print(f"specimen phrase count = {len(phrases)}")
    # Every phrase carries one of the known class strings.
    known = {TERM_ISOLATED, TERM_FLOW, INTERNAL, DASH_ELLIP, BREAK_BARE,
             SET_QUOTE, SET_PAREN, NONE}
    assert all(p.demarcator_class in known for p in phrases)
