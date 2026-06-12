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
        adapter: The embedding backend. Must produce unit (L2-normalized)
            vectors of dimension ``adapter.dimensions``.
        max_tokens_per_request: Token budget per ``embed_batch`` call. Sub-chunks
            are accumulated across texts until this budget is reached, then one
            batched call is issued. Keep under the server's batch_size with
            headroom.
        max_tokens_per_chunk: A single text is split into sub-chunks only if its
            estimated tokens exceed this. Keep under the server's ctx_size with
            headroom. Most texts fall under this and pass through whole.
        checkpoint_path: Optional JSONL path for crash-resume. If it exists it is
            loaded on ``embed_corpus`` and appended to as groups complete.
    """

    def __init__(
        self,
        adapter: EmbeddingAdapter,
        *,
        max_tokens_per_request: int = 3000,
        max_tokens_per_chunk: int = 1500,
        checkpoint_path: Optional[str] = None,
    ):
        self.adapter = adapter
        self.max_tokens_per_request = max_tokens_per_request
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.checkpoint_path = checkpoint_path

    # ------------------------------------------------------------------
    # Token estimation and splitting
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        """Heuristic token count: ~4 chars per token."""
        return len(text) // 4 + 1

    def _split_text(self, text: str, max_tokens: int) -> List[str]:
        """
        Split ``text`` into sub-chunks each estimated at <= ``max_tokens`` tokens.

        Packs whole sentences greedily; any single sentence that is itself over
        the limit is hard-split by characters. Never returns an empty list for
        non-empty input.
        """
        if not text.strip():
            return [text]
        if self._estimate_tokens(text) <= max_tokens:
            return [text]

        max_chars = max(1, max_tokens * 4)
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s]

        sub_chunks: List[str] = []
        current = ""

        def flush() -> None:
            nonlocal current
            if current:
                sub_chunks.append(current)
                current = ""

        for sentence in sentences:
            if self._estimate_tokens(sentence) > max_tokens:
                # Oversized single sentence: flush, then hard char-split it.
                flush()
                for start in range(0, len(sentence), max_chars):
                    piece = sentence[start:start + max_chars]
                    if piece:
                        sub_chunks.append(piece)
                continue

            candidate = sentence if not current else current + " " + sentence
            if current and self._estimate_tokens(candidate) > max_tokens:
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

        # Precompute sub-chunks + token cost per pending text once.
        prepared: List[Tuple[str, List[str], int]] = []
        for text, cid in pending:
            sub_chunks = self._split_text(text, self.max_tokens_per_chunk)
            cost = sum(self._estimate_tokens(sc) for sc in sub_chunks)
            if cost > self.max_tokens_per_request:
                print(
                    f"[bulk] warning: item {cid!r} estimated at {cost} tokens "
                    f"exceeds per-request budget {self.max_tokens_per_request};"
                    f" it will be sent as its own over-budget group",
                    file=sys.stderr,
                )
            prepared.append((cid, sub_chunks, cost))

        # Group into request-batches by max_tokens_per_request.
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

        f_out = None
        start_time = time.time()
        done = skipped

        try:
            if self.checkpoint_path:
                f_out = open(self.checkpoint_path, "a")

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
                        # Normalize single sub-chunk vectors too, so all stored
                        # embeddings are unit-norm regardless of adapter behavior.
                        vec = self._l2_normalize(embeddings[idx])
                    else:
                        vec = self._l2_normalize(embeddings[idx:idx + k].mean(axis=0))
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

                # Flush once per group, not per line.
                if f_out is not None:
                    f_out.flush()

                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0.0
                print(
                    f"[bulk] {done}/{total} ({rate:.0f}/s)",
                    file=sys.stderr,
                )
        finally:
            if f_out is not None:
                f_out.close()

        return completed
