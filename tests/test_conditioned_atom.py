"""Unit tests for the context-conditioned embedding atom (ADR-SKMCP-0003, Phase 2).

No network: a deterministic ``FakeAdapter`` mimics the live llama.cpp
``--pooling none`` contract — ``tokenize_pieces`` returns leading-space-grouped
pieces that concatenate back to the text, and ``embed_tokens`` returns
``[BOS] + one-row-per-piece + [EOS]``. These encode the live findings from the
8083 validation (tail-span localization; the benign set-demarcator straddle that
keeps the opening delimiter in the target span; the BOS/EOS row invariant).
"""

import re

import numpy as np
import pytest

from semantic_kinematics.bearing.phrase_segment import Phrase
from semantic_kinematics.bearing.conditioned import (
    _piece_offsets,
    _target_row_indices,
    conditioned_step,
    conditioned_vectors,
    length_bucket,
)

DIM = 8


def _vec(s: str) -> np.ndarray:
    """Deterministic distinct vector per piece string."""
    rng = np.random.default_rng(abs(hash(s)) % (2**32))
    return rng.standard_normal(DIM)


class FakeAdapter:
    """Mimics the live per-token contract without a server."""

    def tokenize_pieces(self, text):
        # Match the real Gemma tokenizer's behavior validated live on :8083:
        # newlines are their OWN pieces; a leading *space/tab* groups with the
        # following non-space char (" (", " said"); space/tab runs stand alone.
        # Tiles `text` exactly.
        pieces = re.findall(r"\n+|[ \t]*\S|[ \t]+", text)
        assert "".join(pieces) == text  # lossless, like the real /tokenize
        return [{"id": i, "piece": p} for i, p in enumerate(pieces)]

    def embed_tokens(self, text):
        pieces = [d["piece"] for d in self.tokenize_pieces(text)]
        rows = [_vec("<BOS>")] + [_vec(p) for p in pieces] + [_vec("<EOS>")]
        return np.vstack(rows)


class BadInvariantAdapter(FakeAdapter):
    """Returns the wrong row count (no EOS) to trip the alignment guard."""

    def embed_tokens(self, text):
        pieces = [d["piece"] for d in self.tokenize_pieces(text)]
        return np.vstack([_vec("<BOS>")] + [_vec(p) for p in pieces])  # missing EOS


def _phrase(raw, content=None, demarc="TERM_FLOW", ws="space"):
    return Phrase(
        raw=raw,
        content=content if content is not None else raw.strip(),
        demarcator_class=demarc,
        whitespace_class=ws,
    )


# --- pure offset / span-localization logic ------------------------------------

def test_piece_offsets_tile_text():
    ad = FakeAdapter()
    pieces = ad.tokenize_pieces("He said (with a torch).")
    offs = _piece_offsets(pieces)
    assert offs[0][0] == 0
    assert offs[-1][1] == len("He said (with a torch).")
    # contiguous, no gaps
    for (a, b), (c, _d) in zip(offs, offs[1:]):
        assert b == c


def test_target_rows_are_the_tail():
    ad = FakeAdapter()
    text = "The men lifted torches.\nWith a torch.\n"
    boundary = len("The men lifted torches.\n")  # start of the target phrase
    pieces = ad.tokenize_pieces(text)
    idxs = _target_row_indices(_piece_offsets(pieces), boundary)
    recon = "".join(pieces[j]["piece"] for j in idxs)
    assert recon == text[boundary:]  # clean (newline) boundary → exact


def test_set_demarcator_straddle_keeps_opening_delimiter():
    # 'He whispered ' (trailing space) | '(with a torch).' — the tokenizer makes
    # ' (' one leading-space piece that straddles the boundary. It MUST be in the
    # target span (the '(' opening demarcator is part of the register-act), so the
    # reconstruction is the target text plus at most one leading whitespace char.
    ad = FakeAdapter()
    prefix, target = "He whispered ", "(with a torch)."
    text = prefix + target
    idxs = _target_row_indices(_piece_offsets(ad.tokenize_pieces(text)), len(prefix))
    recon = "".join(ad.tokenize_pieces(text)[j]["piece"] for j in idxs)
    assert recon.lstrip() == target          # the '(' is retained
    assert recon.endswith(target)
    assert len(recon) - len(target) <= 1     # at most one straddled whitespace char


# --- conditioned_step / conditioned_vectors -----------------------------------

def test_k0_pools_whole_phrase():
    ad = FakeAdapter()
    phrases = [_phrase("alpha beta. "), _phrase("gamma delta.")]
    step = conditioned_step(phrases, i=1, k=0, adapter=ad)
    assert step.actual_k == 0
    # k=0: boundary 0 → every content token is in the span
    assert step.span_tokens == len(ad.tokenize_pieces("gamma delta."))


def test_vector_is_unit_norm_and_right_dim():
    ad = FakeAdapter()
    phrases = [_phrase("one two. "), _phrase("three four. "), _phrase("five six.")]
    step = conditioned_step(phrases, i=2, k=2, adapter=ad)
    assert step.vector.shape == (DIM,)
    assert np.isclose(np.linalg.norm(step.vector), 1.0)


def test_actual_k_is_capped_by_available_context():
    ad = FakeAdapter()
    phrases = [_phrase("a a. "), _phrase("b b. "), _phrase("c c.")]
    assert conditioned_step(phrases, i=1, k=5, adapter=ad).actual_k == 1  # only 1 leading
    assert conditioned_step(phrases, i=2, k=5, adapter=ad).actual_k == 2


def test_bos_eos_invariant_raises():
    phrases = [_phrase("x y. "), _phrase("z w.")]
    with pytest.raises(ValueError, match="misalignment"):
        conditioned_step(phrases, i=1, k=1, adapter=BadInvariantAdapter())


def test_conditioned_vectors_matrix_shape_and_norm():
    ad = FakeAdapter()
    phrases = [_phrase("aa bb. "), _phrase("cc dd. "), _phrase("ee ff. "), _phrase("gg hh.")]
    matrix, steps = conditioned_vectors(phrases, k=2, adapter=ad)
    assert matrix.shape == (4, DIM)
    assert len(steps) == 4
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)


def test_conditioning_changes_representation():
    ad = FakeAdapter()
    phrases = [_phrase("setup one. "), _phrase("setup two. "), _phrase("the target here.")]
    v0 = conditioned_step(phrases, i=2, k=0, adapter=ad).vector
    v2 = conditioned_step(phrases, i=2, k=2, adapter=ad).vector
    assert not np.allclose(v0, v2)  # leading context shifts the pooled target rep


@pytest.mark.parametrize(
    "n,bucket", [(1, "1-3"), (3, "1-3"), (4, "4-7"), (7, "4-7"),
                 (8, "8-15"), (15, "8-15"), (16, "16+"), (40, "16+")],
)
def test_length_bucket_boundaries(n, bucket):
    assert length_bucket(n) == bucket
