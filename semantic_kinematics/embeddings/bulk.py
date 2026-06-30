"""
Bulk embedding engine.

`BulkEmbedder` wraps any :class:`EmbeddingAdapter` and embeds large corpora
fast and resumably. The central efficiency win over a naive splitter is that
*most* texts pass through whole -- a text is split into sub-chunks only if its
estimated token count exceeds ``max_tokens_per_chunk``. Sub-chunks from many
texts are packed together into a single ``embed_batch`` call until a per-request
token budget is reached, so the adapter (and the server behind it) sees large
batched requests rather than one call per text.

Resume semantics mirror the thought-vault embedding bridge: the checkpoint is a
JSONL file with one ``{"chunk_id", "embedding", "_failed"?}`` entry per original
text. An id counts as done only if its vector is the right dimension, non-zero,
and not flagged ``_failed`` -- so zero/failed entries are retried on resume.
"""

import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from semantic_kinematics.embeddings.base import EmbeddingAdapter


# Sentence-boundary split: a ., !, or ? followed by whitespace, or a newline.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class BulkEmbedder:
    """
    Resumable, batched bulk embedder over an :class:`EmbeddingAdapter`.

    Args:
        adapter: The embedding backend. Sub-chunk vectors are L2-normalized
            before averaging, so the adapter need not return unit vectors; the
            stored direction is independent of the adapter's normalization.
        max_tokens_per_request: Token budget per ``embed_batch`` call. Sub-chunks
            are accumulated across texts until this budget is reached, then one
            batched call is issued. Keep under the server's batch_size with
            headroom.
        max_tokens_per_chunk: A single text is split into sub-chunks only if its
            true token count (``adapter.count_tokens``) exceeds this. Keep under
            the server's physical batch / ctx ceiling with headroom so a single
            sub-chunk always fits. Most texts fall under this and pass through
            whole.
        checkpoint_path: Optional JSONL path for crash-resume. If it exists it is
            loaded on ``embed_corpus`` and appended to as groups complete.

    Token knobs (derived against embeddinggemma-300M-F32 on llama.cpp, where the
    physical batch == n_ctx == 2048 *actual* tokens):

    - ``max_tokens_per_chunk=1800`` -- ~250 tokens of headroom under the 2048
      hard ceiling, so a single sub-chunk that counts at <=1800 true tokens
      always fits one request. (The old default of 1500 only had headroom under
      the chars/4 estimate that issue #20 showed undershoots ~3.5x on dense
      code/JSON, letting 5000+-token inputs slip through unsplit and 500.)
    - ``max_tokens_per_request=2000`` -- a packed batch stays at/under the 2048
      physical batch; an over-budget single item is still sent as its own group.
    """

    def __init__(
        self,
        adapter: EmbeddingAdapter,
        *,
        max_tokens_per_request: int = 2000,
        max_tokens_per_chunk: int = 1800,
        checkpoint_path: Optional[str] = None,
        prep_window: int = 256,
    ):
        self.adapter = adapter
        self.max_tokens_per_request = max_tokens_per_request
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.checkpoint_path = checkpoint_path
        # Items prepped (split/tokenized) per embed cycle. Bounds the prep that
        # must re-run after a crash: a restart re-preps only the not-yet-
        # checkpointed remainder, never the whole corpus. Small enough that the
        # first checkpoint lands within seconds; large enough to pack groups.
        self.prep_window = prep_window

    # ------------------------------------------------------------------
    # Token estimation and splitting
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        """True token count from the adapter's own tokenizer.

        Delegates to ``adapter.count_tokens``; there is no chars-per-token
        fallback by design (issue #20) -- a backend with no reachable tokenizer
        raises ``NotImplementedError`` so the split decision never rests on a
        fiction that silently drops dense content.
        """
        return self.adapter.count_tokens(text)

    def _hard_split_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """Char-split an oversized atomic sentence so each piece fits ``max_tokens``.

        We don't know the chars/token ratio for dense content a priori (that was
        the whole #20 failure), so we bisect on character length and verify each
        candidate piece against the real tokenizer, shrinking until it fits.
        """
        pieces: List[str] = []
        remaining = text
        while remaining:
            if self._count_tokens(remaining) <= max_tokens:
                pieces.append(remaining)
                break
            # Find the largest char prefix that fits max_tokens true tokens.
            lo, hi = 1, len(remaining)
            best = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if self._count_tokens(remaining[:mid]) <= max_tokens:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            pieces.append(remaining[:best])
            remaining = remaining[best:]
        return pieces

    def _split_text(self, text: str, max_tokens: int) -> List[str]:
        """
        Split ``text`` into sub-chunks each at <= ``max_tokens`` *true* tokens.

        Packs whole sentences greedily against the adapter's real tokenizer; any
        single sentence that is itself over the limit is hard-split by characters
        (verified against the tokenizer). Never returns an empty list for
        non-empty input.
        """
        if not text.strip():
            return [text]
        if self._count_tokens(text) <= max_tokens:
            return [text]

        sentences = [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s]

        sub_chunks: List[str] = []
        current = ""

        def flush() -> None:
            nonlocal current
            if current:
                sub_chunks.append(current)
                current = ""

        for sentence in sentences:
            if self._count_tokens(sentence) > max_tokens:
                # Oversized single sentence: flush, then hard token-split it.
                flush()
                sub_chunks.extend(self._hard_split_by_tokens(sentence, max_tokens))
                continue

            candidate = sentence if not current else current + " " + sentence
            if current and self._count_tokens(candidate) > max_tokens:
                flush()
                current = sentence
            else:
                current = candidate

        flush()

        if not sub_chunks:
            # Defensive: e.g. whitespace-only after stripping splits.
            return [text]
        return sub_chunks

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> Dict[str, np.ndarray]:
        """
        Load completed embeddings from the checkpoint JSONL for crash resume.

        Only entries with a non-zero vector of the right dimension and no
        ``_failed`` marker count as completed; zero vectors and ``_failed``
        entries are omitted so they are retried on resume. Corrupt lines are
        skipped with a warning; valid entries on other lines are kept.
        """
        completed: Dict[str, np.ndarray] = {}
        path = self.checkpoint_path
        if not (path and os.path.isfile(path)):
            return completed

        dim = self.adapter.dimensions
        with open(path, "r") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    cid = entry["chunk_id"]
                    vec = entry.get("embedding")
                except (json.JSONDecodeError, KeyError, TypeError):
                    print(
                        f"[bulk] skipping corrupt checkpoint line {lineno}",
                        file=sys.stderr,
                    )
                    continue
                if entry.get("_failed", False):
                    continue
                if not vec or len(vec) != dim:
                    continue
                arr = np.asarray(vec, dtype=float)
                if float(np.dot(arr, arr)) > 1e-10:
                    completed[cid] = arr
        return completed

    @property
    def _meta_path(self) -> Optional[str]:
        """Sidecar path recording the checkpoint's producing model (issue #16)."""
        return self.checkpoint_path + ".meta.json" if self.checkpoint_path else None

    def _reconcile_meta(self) -> None:
        """Make the checkpoint self-describing and guard resume by model identity.

        A bare ``{chunk_id, embedding}`` JSONL cannot identify its producer, and
        the embedding dimension was the only resume guard -- two different models
        at the same dimension (e.g. two 4096-d backends) would silently merge
        into one artifact (issue #16). This writes/validates a sidecar
        ``<checkpoint>.meta.json`` carrying ``model_name`` + ``dimensions``:

        - No meta yet (fresh run, or adopting a pre-#16 checkpoint): record the
          current adapter's identity.
        - Meta present: it MUST match the current adapter or we refuse to resume
          (PARSE-AT-THE-DOOR fail-loud) rather than mixing incompatible vectors.

        Sidecar, not a header line, so the checkpoint stays a pure
        one-record-per-line stream (no dual-shape parse).
        """
        path = self._meta_path
        if not path:
            return
        current = {
            "model_name": self.adapter.model_name,
            "dimensions": int(self.adapter.dimensions),
        }
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                raise ValueError(
                    f"[bulk] checkpoint metadata {path!r} is unreadable ({exc}); "
                    f"refusing to resume against an unidentifiable artifact"
                ) from exc
            mismatch = {
                key: {"existing": existing.get(key), "current": current[key]}
                for key in current
                if existing.get(key) != current[key]
            }
            if mismatch:
                raise ValueError(
                    f"[bulk] checkpoint {self.checkpoint_path!r} was produced by a "
                    f"different model: {mismatch}. Resuming would merge incompatible "
                    f"vectors -- use a fresh checkpoint path."
                )
            return
        # Atomic write-then-rename: a torn meta (crash mid-write) would be read
        # as "unreadable" next run and permanently block resume, so never leave
        # a partial file at ``path``.
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(current, f)
        os.replace(tmp_path, path)

    # ------------------------------------------------------------------
    # Vector helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return vec
        return vec / norm

    def _is_valid(self, vec: np.ndarray) -> bool:
        return (
            vec is not None
            and vec.shape == (self.adapter.dimensions,)
            and float(np.dot(vec, vec)) > 1e-10
            and bool(np.all(np.isfinite(vec)))
        )

    # ------------------------------------------------------------------
    # Corpus embedding
    # ------------------------------------------------------------------

    def embed_corpus(
        self, items: List[Tuple[str, str]]
    ) -> Dict[str, np.ndarray]:
        """
        Embed a corpus of ``(text, chunk_id)`` items, resumably.

        Returns a dict mapping ``chunk_id`` -> embedding for every item that was
        successfully embedded (or restored from checkpoint). Permanently failed
        items (group errored, or backend returned a bad vector) are written to
        the checkpoint flagged ``_failed`` and omitted from the result.
        """
        dim = self.adapter.dimensions
        zero_vec = [0.0] * dim
        total = len(items)

        # #16: write/validate the sidecar identity before touching the
        # checkpoint, so a model/dim mismatch fails loud instead of silently
        # merging incompatible vectors.
        self._reconcile_meta()

        completed = self._load_checkpoint()
        pending = [(text, cid) for (text, cid) in items if cid not in completed]
        skipped = total - len(pending)
        print(
            f"[bulk] {total} total -- {skipped} from checkpoint, "
            f"{len(pending)} pending",
            file=sys.stderr,
        )

        if not pending:
            # Nothing to embed: leave the checkpoint file untouched.
            return completed

        f_out = None
        start_time = time.time()
        done = skipped

        def mark_failed(cid: str) -> None:
            """Write a ``_failed`` marker for ``cid`` so resume retries it.

            Same checkpoint mechanism the embed-batch failure path uses: a
            permanently-failed item is recorded with a zero vector and the
            ``_failed`` flag, never silently dropped, so an idempotent
            re-invocation picks it back up.
            """
            if f_out is not None:
                f_out.write(
                    json.dumps(
                        {
                            "chunk_id": cid,
                            "embedding": zero_vec,
                            "_failed": True,
                        }
                    )
                    + "\n"
                )

        try:
            if self.checkpoint_path:
                f_out = open(self.checkpoint_path, "a")

            # Stream the work in windows: prep (split/tokenize) a window, embed
            # it, checkpoint, then advance. The previous design prepped EVERY
            # pending item before the first embed -- a long, UNRESUMABLE head:
            # a crash before any group embedded lost all that prep with no
            # on-disk breadcrumb (embedding resumed from the jsonl, prep did
            # not). Windowing makes the whole run reconstruct-from-checkpoint --
            # a restart re-preps only items not yet written, never the corpus.
            for w_start in range(0, len(pending), self.prep_window):
                batch = pending[w_start:w_start + self.prep_window]

                # Precompute sub-chunks + token cost per item in THIS window. A
                # count_tokens/_split_text failure for ONE item marks just that
                # item _failed and continues -- symmetric with the embed-batch
                # failure path below -- rather than aborting the whole run and
                # losing already-completed work with no resumable marker.
                prepared: List[Tuple[str, List[str], int]] = []
                for text, cid in batch:
                    try:
                        sub_chunks = self._split_text(text, self.max_tokens_per_chunk)
                        cost = sum(self._count_tokens(sc) for sc in sub_chunks)
                    except Exception as exc:  # noqa: BLE001 -- isolate per item.
                        print(
                            f"[bulk] preparation (tokenize/split) failed for item "
                            f"{cid!r} ({exc}); marking _failed and continuing",
                            file=sys.stderr,
                        )
                        mark_failed(cid)
                        continue
                    if cost > self.max_tokens_per_request:
                        print(
                            f"[bulk] warning: item {cid!r} estimated at {cost} tokens "
                            f"exceeds per-request budget {self.max_tokens_per_request};"
                            f" it will be sent as its own over-budget group",
                            file=sys.stderr,
                        )
                    prepared.append((cid, sub_chunks, cost))

                # Group THIS window into request-batches by max_tokens_per_request.
                groups: List[List[Tuple[str, List[str]]]] = []
                current: List[Tuple[str, List[str]]] = []
                current_tokens = 0
                for cid, sub_chunks, cost in prepared:
                    if current and current_tokens + cost > self.max_tokens_per_request:
                        groups.append(current)
                        current = []
                        current_tokens = 0
                    current.append((cid, sub_chunks))
                    current_tokens += cost
                if current:
                    groups.append(current)

                for group in groups:
                    # Flatten all sub-chunks across the group into one batch call.
                    flat: List[str] = []
                    counts: List[int] = []
                    for _cid, sub_chunks in group:
                        flat.extend(sub_chunks)
                        counts.append(len(sub_chunks))

                    try:
                        raw = self.adapter.embed_batch(flat)
                        embeddings = np.asarray(raw, dtype=float)
                        group_failed = embeddings.shape[0] != len(flat)
                    except Exception as exc:  # noqa: BLE001 -- isolate per group.
                        print(
                            f"[bulk] group embed failed ({exc}); marking "
                            f"{len(group)} texts _failed",
                            file=sys.stderr,
                        )
                        embeddings = None
                        group_failed = True

                    # Reassemble one vector per original text.
                    idx = 0
                    for (cid, _sub_chunks), k in zip(group, counts):
                        if group_failed or embeddings is None:
                            vec = None
                        elif k == 1:
                            # Single sub-chunk: exact passthrough up to L2-normalize,
                            # so all stored embeddings are unit-norm regardless of
                            # adapter behavior.
                            vec = self._l2_normalize(embeddings[idx])
                        else:
                            # Direction centroid (#17): L2-normalize EACH sub-chunk
                            # before averaging so larger-norm sub-chunks can't
                            # magnitude-dominate the direction, then normalize the
                            # mean. Independent of whether the adapter returns unit
                            # vectors.
                            normalized = np.array(
                                [self._l2_normalize(embeddings[idx + j]) for j in range(k)]
                            )
                            mean = normalized.mean(axis=0)
                            if float(np.linalg.norm(mean)) <= 1e-12:
                                # Antipodal sub-chunks cancelled to ~zero: the mean
                                # has no direction, so it'll fail _is_valid below and
                                # be marked _failed. Name the chunk so it's findable.
                                print(
                                    f"[bulk] item {cid!r}: {k} sub-chunk mean "
                                    f"collapsed to near-zero (antipodal); rejecting",
                                    file=sys.stderr,
                                )
                            vec = self._l2_normalize(mean)
                        idx += k

                        if vec is not None and self._is_valid(vec):
                            completed[cid] = vec
                            if f_out is not None:
                                f_out.write(
                                    json.dumps(
                                        {"chunk_id": cid, "embedding": vec.tolist()}
                                    )
                                    + "\n"
                                )
                            done += 1
                        else:
                            mark_failed(cid)

                    # Flush once per group, not per line.
                    if f_out is not None:
                        f_out.flush()

                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[bulk] {done}/{total} ({rate:.0f}/s)",
                        file=sys.stderr,
                    )

                # A window of only prep-failures produces no groups; flush its
                # _failed markers so they are durable before the next window.
                if f_out is not None:
                    f_out.flush()
        finally:
            if f_out is not None:
                f_out.close()

        return completed
