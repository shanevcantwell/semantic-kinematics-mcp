"""Context-conditioned embedding atom (ADR-SKM-0003).

Embed a target phrase *inside* a ``[leading context + target]`` window so the
embedder's attention encodes the setup into the target's representation, then
pool **only the target span** (consistent across the context-ramp index ``k``).

Why this construction and not context-free embedding: the comedic jolt is a
*conditional* object — the punchline carries it only read after its setup. A
context-free atom (token/phrase/sentence) amputates that contrast; conditioning
restores it. See ``docs/ADRs/proposed/ADR-SKM-0003-context-conditioned-embedding-atom.md``.

Span-localization is by **character-offset mapping**, not by tokenizing the
target in isolation: isolated-T diverges from in-context-T by a token at set
demarcators, no-space boundaries, and trailing ``\\n`` — a one-token slip is a
large fraction of a ~4-token punchline. Because the target is always the *tail*
of its window, its token span is exactly "every content token whose character
range ends past the prefix/target boundary", which absorbs any seam token
cleanly.

Pooling is over the target's **content** tokens (BOS/EOS excluded); this is
consistent across all ``k``. Note: llama.cpp mean-pools over *all* rows
including BOS/EOS, so ``k=0`` here is a *behavioral* anchor (context-free
phrase), not an exact match to the ``/v1/embeddings`` pooled vector.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from semantic_kinematics.bearing.phrase_segment import Phrase


# Token ceiling for one embedding window. Matches the BulkEmbedder convention
# (issue #20): the server is n_ctx=2048 with physical batch == n_ctx, so a single
# input is held to ~1800 true tokens — ~250 of headroom under the 2048 hard
# ceiling, covering the BOS/EOS framing too.
MAX_CTX = 1800


class WindowTooLong(Exception):
    """A target phrase alone exceeds ``max_ctx`` tokens — cannot be embedded.

    Distinct from a transient failure: this is a deterministic content limit, so
    the caller should *skip* (e.g. the null builder drops the turn), not retry.
    """


@dataclass
class ConditionedStep:
    """One target phrase's conditioned representation + its null-stratum keys."""

    vector: np.ndarray      # (dim,), L2-unit
    label: str              # target phrase content (trimmed for readability)
    actual_k: int           # leading phrases actually used (min(k, i))
    span_tokens: int        # target token count (content + demarcator + newline)
    demarcator_class: str   # target phrase's demarcator class


def _piece_offsets(pieces: List[dict]) -> List[Tuple[int, int]]:
    """Cumulative ``[start, end)`` char offsets for each ``/tokenize`` piece.

    Pieces concatenate back to the tokenized text exactly (Gemma leading-space
    convention), so this reconstructs an exact char→token map.
    """
    offsets: List[Tuple[int, int]] = []
    acc = 0
    for p in pieces:
        piece = p["piece"]
        offsets.append((acc, acc + len(piece)))
        acc += len(piece)
    return offsets


def _target_row_indices(
    content_offsets: List[Tuple[int, int]], boundary_char: int
) -> List[int]:
    """Indices of content tokens belonging to the target (the window's tail).

    A token is in the target if its char range *ends past* the prefix/target
    boundary (``end > boundary_char``) — this includes a seam token that straddles
    the boundary, and at ``k=0`` (boundary 0) selects every token.
    """
    return [j for j, (_a, b) in enumerate(content_offsets) if b > boundary_char]


def conditioned_step(
    phrases: List[Phrase], i: int, k: int, adapter, max_ctx: int = MAX_CTX
) -> ConditionedStep:
    """Build the conditioned vector for target ``phrases[i]`` with ``k`` leading
    phrases of context (capped to what's available: ``actual_k = min(k, i)``).

    The window is reconstructed with original separators intact (line structure
    is signal). Raises if the ``[BOS] + content + [EOS]`` row invariant breaks —
    that would mean the embedder's special-token framing changed and silent
    misalignment would contaminate short punchlines.
    """
    actual_k = min(k, i)
    # Cap leading context so the window fits the embedder's token ceiling.
    # Tokenize first (cheap); shrink k until it fits, then embed.
    while True:
        window = phrases[i - actual_k : i + 1]
        window_text = "".join(p.raw for p in window)
        pieces = adapter.tokenize_pieces(window_text)   # content tokens only
        if len(pieces) + 2 <= max_ctx or actual_k == 0:
            break
        actual_k -= 1
    if len(pieces) + 2 > max_ctx:
        raise WindowTooLong(
            f"target phrase i={i} alone is {len(pieces)} tokens > max_ctx {max_ctx}"
        )
    prefix_text = "".join(p.raw for p in window[:-1])
    boundary_char = len(prefix_text)  # target.content starts here (raw has no leading ws)

    rows = adapter.embed_tokens(window_text)          # [BOS] + content + [EOS]
    if rows.shape[0] != len(pieces) + 2:
        raise ValueError(
            "embed/tokenize misalignment: expected rows == pieces + 2 "
            f"(BOS+EOS), got rows={rows.shape[0]} pieces={len(pieces)} "
            "— embedder special-token framing changed; span localization unsafe."
        )
    content_rows = rows[1:-1]
    content_offsets = _piece_offsets(pieces)
    idxs = _target_row_indices(content_offsets, boundary_char)
    if not idxs:
        raise ValueError(
            f"target span empty at i={i}, k={k} "
            f"(boundary={boundary_char}, window_len={len(window_text)})"
        )

    pooled = content_rows[idxs].mean(axis=0)
    norm = np.linalg.norm(pooled)
    if norm == 0.0:
        raise ValueError(f"zero-norm pooled target at i={i}, k={k}")
    vector = pooled / norm

    label = phrases[i].content.strip()
    return ConditionedStep(
        vector=vector,
        label=label if len(label) <= 80 else label[:79] + "…",
        actual_k=actual_k,
        span_tokens=len(idxs),
        demarcator_class=phrases[i].demarcator_class,
    )


def conditioned_vectors(
    phrases: List[Phrase], k: int, adapter, max_ctx: int = MAX_CTX
) -> Tuple[np.ndarray, List[ConditionedStep]]:
    """Conditioned trajectory at context length ``k`` over an ordered phrase list.

    Returns ``(matrix, steps)`` where ``matrix`` is ``(n_phrases, dim)`` unit-norm
    — the contract for ``TrajectoryAnalyzer.analyze_embeddings`` — and ``steps``
    carries per-phrase null-stratum keys (``actual_k``, ``span_tokens``,
    ``demarcator_class``) used by the per-(k×length×demarcator) null.

    This is the single construction shared by both the null builder and the look
    (ADR read-discipline guard: if the two diverged, the per-``k`` smoothing
    would stop cancelling).
    """
    steps = [
        conditioned_step(phrases, i, k, adapter, max_ctx=max_ctx)
        for i in range(len(phrases))
    ]
    matrix = np.vstack([s.vector for s in steps])
    return matrix, steps


# Length-bucket boundaries on target span token count (ADR Phase 3).
LENGTH_BUCKETS = ((1, 3), (4, 7), (8, 15), (16, 10**9))


def length_bucket(span_tokens: int) -> str:
    """Map a target token count to its stratum label, e.g. ``"4-7"``."""
    for lo, hi in LENGTH_BUCKETS:
        if lo <= span_tokens <= hi:
            return f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    return "0"  # span_tokens == 0 should not occur (guarded upstream)
