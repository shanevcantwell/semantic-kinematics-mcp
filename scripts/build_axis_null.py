#!/usr/bin/env python3
"""
Build a background null cache for analyze_axis_alignment.

Embeds a corpus once with the active backend and persists embeddings + a
manifest keyed by model name. The null corpus is YOUR data and stays local --
do not commit the generated cache.

Usage:
    # one segment per line
    python scripts/build_axis_null.py corpus.txt --out cache/null.npy

    # a directory of .txt files (each file is one segment)
    python scripts/build_axis_null.py corpus_dir/ --out cache/null.npy

The active backend is selected via EMBEDDING_BACKEND (see StateManager). The
generated manifest (cache/null.npy.json) is what you pass as background_ref /
AXIS_NULL_MANIFEST.
"""

import argparse
import os
import sys

# Allow running as a plain script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_kinematics.mcp.commands.axis_alignment import build_null_cache
from semantic_kinematics.mcp.state_manager import StateManager


def load_segments(path: str) -> list[str]:
    """Read corpus segments: one per line for a file, or one per .txt in a dir."""
    if os.path.isdir(path):
        segments = []
        for name in sorted(os.listdir(path)):
            if name.endswith(".txt"):
                with open(os.path.join(path, name)) as f:
                    text = f.read().strip()
                if text:
                    segments.append(text)
        return segments
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", help="Corpus file (one segment/line) or dir of .txt files")
    parser.add_argument("--out", required=True, help="Output .npy path (manifest written alongside)")
    args = parser.parse_args()

    segments = load_segments(args.corpus)
    if len(segments) < 2:
        print(f"error: need at least 2 segments, found {len(segments)}", file=sys.stderr)
        return 1

    manager = StateManager()
    adapter = manager.get_adapter()
    print(f"Embedding {len(segments)} segments with {adapter.model_name} ...", file=sys.stderr)
    manifest = build_null_cache(adapter, segments, args.out, source=os.path.abspath(args.corpus))
    print(f"Wrote {manifest['count']} x {manifest['dimensions']} -> {args.out}", file=sys.stderr)
    print(f"Manifest: {args.out}.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
