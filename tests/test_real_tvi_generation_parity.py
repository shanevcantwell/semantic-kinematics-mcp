"""Opt-in parity against the promoted real TVI generation.

This test reads text/IDs only.  It never constructs an embedding adapter and
never opens, mutates, or refreshes a vector checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from itertools import zip_longest
from pathlib import Path

import pytest

import scripts.build_conditioned_null as conditioned
import scripts.build_displacement_null as displacement
import scripts.tvi_canonical_input as canonical
from scripts import embed_corpus, smoke_jolt

RUN_ENV = "SK_RUN_REAL_TVI_PARITY"
TVI_EXECUTABLE = Path(
    os.environ.get(
        "TVI_CANONICAL_CORPUS_EXECUTABLE",
        "/home/node/worktrees/tvi-canonical-query-door/.venv/bin/tvi-canonical-corpus",
    )
)
CORPUS_DB = Path(
    os.environ.get(
        "THOUGHT_VAULT_CANONICAL_DB",
        "/home/node/worktrees/tvi-canonical-query-door/output/canonical/corpus.db",
    )
)
CANONICAL_ROOT = Path(
    os.environ.get(
        "THOUGHT_VAULT_CANONICAL_ROOT",
        "/home/node/worktrees/tvi-canonical-query-door/output/canonical",
    )
)
LEGACY_CHUNKS = Path(
    os.environ.get(
        "SK_REAL_TVI_LEGACY_CHUNKS",
        "/home/node/github/shanevcantwell/thought-vault-integration/output/vectors/chunks.jsonl",
    )
)
INPUT_KEYS = ("base",)
SAMPLE_SIZE = 3000
SAMPLE_SEED = 1729
_BUILDERS = (displacement, conditioned)
_SENTINEL = object()

pytestmark = pytest.mark.real_generation


def _require_real_generation() -> None:
    if os.environ.get(RUN_ENV) != "1":
        pytest.skip(f"set {RUN_ENV}=1 to run the real TVI generation parity test")
    missing = [
        str(path)
        for path in (TVI_EXECUTABLE, CORPUS_DB, CANONICAL_ROOT, LEGACY_CHUNKS)
        if not path.exists()
    ]
    if missing:
        pytest.fail(f"real-generation parity inputs are missing: {missing}")


def _legacy_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with LEGACY_CHUNKS.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            value = json.loads(line)
            chunk_id = value.get("chunk_id")
            text = value.get("text")
            assert isinstance(chunk_id, str) and chunk_id, line_number
            assert isinstance(text, str), line_number
            pairs.append((chunk_id, text))
    return pairs


def _canonical_pairs() -> tuple[list[tuple[str, str]], canonical.CanonicalQueryProvenance]:
    pairs: list[tuple[str, str]] = []
    with canonical.open_canonical_chunks(
        CORPUS_DB,
        canonical_root=CANONICAL_ROOT,
        input_keys=INPUT_KEYS,
        executable=TVI_EXECUTABLE,
    ) as stream:
        provenance = stream.provenance
        for row in stream:
            pairs.append((row["chunk_id"], row["text"]))
    return pairs, provenance


@pytest.fixture(scope="module")
def real_pairs() -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str]],
    canonical.CanonicalQueryProvenance,
]:
    _require_real_generation()
    canonical_pairs, provenance = _canonical_pairs()
    return canonical_pairs, _legacy_pairs(), provenance


def _rows(pairs: Iterable[tuple[str, str]]) -> Iterator[dict[str, str]]:
    for chunk_id, text in pairs:
        yield {"chunk_id": chunk_id, "text": text}


def _assert_identical_sequence(
    actual: Iterable[object], expected: Iterable[object], *, label: str
) -> int:
    count = 0
    for index, (actual_item, expected_item) in enumerate(
        zip_longest(actual, expected, fillvalue=_SENTINEL)
    ):
        assert actual_item == expected_item, f"{label} first differs at index {index}"
        count += 1
    return count


def _sequence_sha256(values: Iterable[str | tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for value in values:
        fields = (value,) if isinstance(value, str) else value
        for field in fields:
            encoded = field.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


@pytest.mark.parametrize("builder", _BUILDERS)
def test_real_generation_usable_text_sequence_and_sampling_are_identical(
    builder,
    real_pairs: tuple[
        list[tuple[str, str]],
        list[tuple[str, str]],
        canonical.CanonicalQueryProvenance,
    ],
) -> None:
    canonical_pairs, legacy_pairs, provenance = real_pairs
    canonical_usable = list(builder._iter_usable_texts(_rows(canonical_pairs)))
    legacy_usable = list(builder._iter_usable_texts(_rows(legacy_pairs)))

    usable_count = _assert_identical_sequence(
        canonical_usable,
        legacy_usable,
        label=f"{builder.__name__} usable text",
    )
    canonical_sample, canonical_seen = builder._reservoir_sample_turns(
        _rows(canonical_pairs), SAMPLE_SIZE, SAMPLE_SEED
    )
    legacy_sample, legacy_seen = builder._reservoir_sample_turns(
        _rows(legacy_pairs), SAMPLE_SIZE, SAMPLE_SEED
    )

    assert canonical_seen == legacy_seen == usable_count
    assert canonical_sample == legacy_sample
    print(
        f"{builder.__name__}: generation_id={provenance.generation_id} "
        f"rows={len(canonical_pairs)} usable={usable_count} "
        f"usable_sha256={_sequence_sha256(canonical_usable)} "
        f"sample_k={SAMPLE_SIZE} seed={SAMPLE_SEED} "
        f"sample_sha256={_sequence_sha256(canonical_sample)}"
    )


def test_real_generation_smoke_jolt_specimen_selection_order_and_count_are_identical() -> None:
    _require_real_generation()
    legacy_rows = []
    with LEGACY_CHUNKS.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("conversation_id") != smoke_jolt.TARGET_CONVERSATION_ID:
                continue
            if row.get("speaker") not in smoke_jolt.SPECIMEN_B_SPEAKERS:
                continue
            legacy_rows.append(row)

    legacy_turns = smoke_jolt._ordered_specimen_b_turns(legacy_rows)
    canonical_turns, provenance = smoke_jolt._load_specimen_b_turns(
        corpus_db=str(CORPUS_DB),
        canonical_root=str(CANONICAL_ROOT),
        executable=str(TVI_EXECUTABLE),
    )

    assert canonical_turns == legacy_turns
    assert len(canonical_turns) == 201
    assert _sequence_sha256(canonical_turns) == (
        "9ccfe2e83268b858cb27a7298187c43ca606492bf9fda93f26ab06d0950f5f07"
    )
    print(
        f"smoke_jolt: generation_id={provenance.generation_id} "
        f"turns={len(canonical_turns)} "
        f"query_fingerprint={provenance.query_fingerprint_sha256} "
        f"turn_sha256={_sequence_sha256(canonical_turns)}"
    )


def test_real_generation_embedding_chunk_id_text_inputs_are_identical() -> None:
    _require_real_generation()
    canonical_source = embed_corpus.load_corpus_source(
        corpus_db=str(CORPUS_DB),
        canonical_root=str(CANONICAL_ROOT),
        input_keys=INPUT_KEYS,
        tvi_executable=str(TVI_EXECUTABLE),
    )
    legacy_source = embed_corpus.load_corpus_source(corpus=str(LEGACY_CHUNKS))

    canonical_inputs = ((chunk_id, text) for text, chunk_id in canonical_source.items)
    legacy_inputs = ((chunk_id, text) for text, chunk_id in legacy_source.items)
    input_count = _assert_identical_sequence(
        canonical_inputs,
        legacy_inputs,
        label="embed_corpus (chunk_id, text) input",
    )
    canonical_ordered_inputs = [(chunk_id, text) for text, chunk_id in canonical_source.items]
    print(
        f"embed_corpus: generation_id={canonical_source.provenance['generation_id']} "
        f"inputs={input_count} "
        f"input_sha256={_sequence_sha256(canonical_ordered_inputs)}; "
        "no adapter constructed and no checkpoint opened"
    )
