#!/usr/bin/env python
"""
CLI to bulk-embed a JSONL corpus with crash-resume.

Each input line is a JSON object; the text and id fields are configurable. The
corpus is embedded with :class:`BulkEmbedder` over an adapter from
``get_adapter``, checkpointing to a JSONL file so interrupted runs resume.

Example:
    python scripts/embed_corpus.py corpus.jsonl --checkpoint out.jsonl \\
        --backend lmstudio --base-url http://localhost:8082/v1 \\
        --model embeddinggemma-300M-F32

For the ``lmstudio`` backend, ``--model``/``--base-url`` may also come from the
``EMBEDDING_MODEL``/``EMBEDDING_SERVER_URL`` environment variables (same
resolution chain as the MCP server's StateManager: explicit arg -> env -> hard
fail). No baked model/endpoint default (Rule #14) -- a silently-wrong-model run
is an unacceptable failure class.
"""

import argparse
import json
import os
import sys
import time

# Allow running as a plain script from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from semantic_kinematics.embeddings import get_adapter  # noqa: E402
from semantic_kinematics.embeddings.bulk import BulkEmbedder  # noqa: E402


def _read_items(path, text_field, id_field):
    """Read (text, chunk_id) items from a JSONL file, skipping blank text."""
    items = []
    with open(path, "r") as f:
        for line_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(text_field)
            if not text or not str(text).strip():
                continue
            chunk_id = obj.get(id_field)
            if chunk_id is None:
                chunk_id = f"line-{line_index}"
            items.append((str(text), str(chunk_id)))
    return items


def _make_adapter(backend, model, base_url):
    """Build an embedding adapter with backend-appropriate constructor kwargs.

    The ``lmstudio`` backend is network-based and takes ``model_name`` +
    ``base_url``. The in-process backends (``nv_embed`` /
    ``sentence_transformers``) are path-based: the model is resolved from
    ``model_path`` or the env-driven default (``NV_EMBED_MODEL_PATH``), and the
    lmstudio-only kwargs do not apply — forwarding them raises ``TypeError`` in
    the adapter constructor (the bug this routing fixes). Point the in-process
    backends at a specific model via the ``NV_EMBED_MODEL_PATH`` env var.
    """
    if backend == "lmstudio":
        return get_adapter(backend, model_name=model, base_url=base_url)
    if backend == "nv_embed":
        # Bulk embedding wants the model resident for the whole run. The
        # adapter's per-call unload default reloads ~15GB of weights per
        # request-group, which makes a corpus-scale run prohibitively slow
        # (measured ~2.2s/chunk vs. a single up-front load when resident).
        return get_adapter(backend, unload_after_use=False)
    return get_adapter(backend)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bulk-embed a JSONL corpus.")
    parser.add_argument("corpus", help="Path to input JSONL corpus.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint JSONL path.")
    parser.add_argument("--backend", default="lmstudio", help="Embedding backend.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("EMBEDDING_SERVER_URL"),
        help="lmstudio endpoint (required for --backend lmstudio; falls back to "
             "EMBEDDING_SERVER_URL; no baked default -- Rule #14).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EMBEDDING_MODEL"),
        help="lmstudio model id (required for --backend lmstudio; falls back to "
             "EMBEDDING_MODEL; no baked default -- Rule #14).",
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="chunk_id")
    parser.add_argument("--max-tokens-per-request", type=int, default=3000)
    parser.add_argument("--max-tokens-per-chunk", type=int, default=1500)
    args = parser.parse_args(argv)

    if args.backend == "lmstudio" and not args.model:
        parser.error(
            "--model is required for --backend lmstudio (or set EMBEDDING_MODEL)"
        )
    if args.backend == "lmstudio" and not args.base_url:
        parser.error(
            "--base-url is required for --backend lmstudio (or set EMBEDDING_SERVER_URL)"
        )

    items = _read_items(args.corpus, args.text_field, args.id_field)
    print(f"[embed_corpus] loaded {len(items)} items from {args.corpus}", file=sys.stderr)

    adapter = _make_adapter(args.backend, args.model, args.base_url)
    embedder = BulkEmbedder(
        adapter,
        max_tokens_per_request=args.max_tokens_per_request,
        max_tokens_per_chunk=args.max_tokens_per_chunk,
        checkpoint_path=args.checkpoint,
    )

    start = time.time()
    results = embedder.embed_corpus(items)
    elapsed = time.time() - start

    print(
        f"[embed_corpus] embedded {len(results)} / {len(items)} items "
        f"in {elapsed:.1f}s -> {args.checkpoint}"
    )


if __name__ == "__main__":
    main()
