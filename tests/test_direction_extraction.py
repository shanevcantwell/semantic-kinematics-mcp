"""Tests for the functional-direction probe: initialize_direction (ADR-SKM-008
Phase 2, D2/D3).

All fixture-based: small synthetic memmaps + synthetic seedsets with known
geometry (seeds displaced along a known axis so direction recovery / AUC are
analytically checkable). NO real corpus, NO network, NO real embedding
backend. Covers the acceptance criteria named in issue #57:

- identity-mismatch (seedset id != calibration id) refuses
- a typed_exemplars axis (axis_alignment.build_axis) and a seedset_centroids
  axis (direction.compute_direction) project identically given identical
  poles (ONE-DOOR test)
- an injected leakage case trips the leakage-suspected verdict
- bootstrap-cosine on a too-thin era refuses (surfaces as under-determined)
- era-scoped filtering
- server.py registration (list_tools/call_tool complete-surface test)
- refuse-on-mismatch round trip for the direction artifact (sha mutation)
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pytest

from semantic_kinematics.mcp.commands import direction as d
from semantic_kinematics.mcp.commands.axis_alignment import build_axis
from semantic_kinematics.mcp.commands.direction import (
    Calibration,
    DirectionRefusal,
    auc_score,
    bootstrap_stability,
    compute_direction,
    held_out_separation_auc,
    initialize_direction,
    initialize_direction_core,
    load_direction,
    load_seedset,
    mean_center,
    null_reference,
    read_memmap_rows,
    refuse_unless_seedset_matches_calibration,
    topic_control_check,
    verdict_from_diagnostics,
    write_direction_artifact,
)


DIM = 16
EMBEDDING_MODEL_ID = "nvidia/NV-Embed-v2"
MEMMAP_SHA = "deadbeef" * 8


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Fixtures: a synthetic calibration + memmap + seedset with known geometry.
# --------------------------------------------------------------------------- #

def _make_calibration(mu: np.ndarray, memmap_path: str, n_used: int, sha: str = MEMMAP_SHA) -> Calibration:
    return Calibration(
        embedding_model_id=EMBEDDING_MODEL_ID,
        dim=DIM,
        source_memmap_path=memmap_path,
        source_memmap_sha256=sha,
        n_used=n_used,
        convention_version="tvi-008-dedup-keep-last-zerofilter-v1",
        mu=mu,
        mu_norm=float(np.linalg.norm(mu)),
        eigvecs=None,
        eigvals=None,
        eigenbasis_k=None,
        header={
            "regime": "corpus-calibration",
            "embedding_model_id": EMBEDDING_MODEL_ID,
            "dim": DIM,
            "source_memmap_path": memmap_path,
            "source_memmap_sha256": sha,
            "n_used": n_used,
            "convention_version": "tvi-008-dedup-keep-last-zerofilter-v1",
        },
        manifest={"mu_norm": float(np.linalg.norm(mu))},
    )


def _write_memmap(tmp_path: Path, matrix: np.ndarray) -> str:
    path = tmp_path / "vectors_test.f32"
    matrix.astype(np.float32).tofile(path)
    return str(path)


def _make_corpus(
    rng: np.random.Generator,
    n_total: int,
    dim: int,
    mu: np.ndarray,
    true_axis: np.ndarray,
    n_seeds: int,
    n_negatives: int,
    displacement: float = 1.5,
    noise: float = 0.3,
):
    """Build a synthetic (n_total, dim) corpus where rows [0:n_seeds) are
    seeds displaced +displacement along true_axis, rows
    [n_seeds:n_seeds+n_negatives) are topic-matched negatives at baseline
    (paired 1:1 with seeds by index), and the remainder is unlabeled
    background corpus (for random-sample / null-reference reads).
    """
    x = rng.normal(size=(n_total, dim)) * noise + mu
    x[:n_seeds] += displacement * true_axis
    seed_rowids = list(range(n_seeds))
    negative_rowids = list(range(n_seeds, n_seeds + n_negatives))
    return x, seed_rowids, negative_rowids


def _seedset_dict(
    pattern_id: str,
    embedding_model_id: str,
    vector_memmap_sha256: str,
    seed_rowids,
    negative_rowids,
    eras=None,
):
    eras = eras or ([None] * len(seed_rowids))
    seeds = [
        {"chunk_id": f"seed-{i}", "era": eras[i % len(eras)], "rowid_mm": rid}
        for i, rid in enumerate(seed_rowids)
    ]
    negatives = [
        {
            "chunk_id": f"neg-{i}",
            "era": eras[i % len(eras)],
            "rowid_mm": rid,
            "paired_seed_chunk_id": f"seed-{i}",
        }
        for i, rid in enumerate(negative_rowids)
    ]
    return {
        "manifest": {
            "pattern_id": pattern_id,
            "corpus_snapshot": {
                "embedding_model_id": embedding_model_id,
                "vector_memmap_sha256": vector_memmap_sha256,
            },
            "counts": {"seeds": len(seeds), "matched_negatives": len(negatives)},
        },
        "seeds": seeds,
        "negatives": negatives,
    }


@pytest.fixture
def synthetic_scene(tmp_path):
    """A full synthetic scene: memmap + calibration + seedset with n=41 seeds
    (matching the real comparative-perception seedset's small-n scale) and a
    known true axis, cleanly separated (for happy-path / ONE-DOOR tests)."""
    rng = np.random.default_rng(7)
    mu = rng.normal(size=DIM) * 0.1
    true_axis = rng.normal(size=DIM)
    true_axis /= np.linalg.norm(true_axis)

    n_seeds, n_negs, n_background = 41, 41, 200
    n_total = n_seeds + n_negs + n_background
    x, seed_rowids, negative_rowids = _make_corpus(
        rng, n_total, DIM, mu, true_axis, n_seeds, n_negs, displacement=0.6, noise=0.5
    )
    memmap_path = _write_memmap(tmp_path, x)
    calibration = _make_calibration(mu, memmap_path, n_total)
    seedset = _seedset_dict(
        "comparative-perception", EMBEDDING_MODEL_ID, MEMMAP_SHA, seed_rowids, negative_rowids
    )
    return {
        "rng": rng,
        "mu": mu,
        "true_axis": true_axis,
        "x": x,
        "memmap_path": memmap_path,
        "calibration": calibration,
        "seedset": seedset,
        "seed_rowids": seed_rowids,
        "negative_rowids": negative_rowids,
        "n_total": n_total,
    }


# --------------------------------------------------------------------------- #
# Seedset loader: shape validation + refusal.
# --------------------------------------------------------------------------- #

def test_load_seedset_happy_path(tmp_path, synthetic_scene):
    path = tmp_path / "pattern.seedset.json"
    path.write_text(json.dumps(synthetic_scene["seedset"]))
    blob = load_seedset(str(path))
    assert blob["manifest"]["pattern_id"] == "comparative-perception"
    assert len(blob["seeds"]) == 41


def test_load_seedset_missing_file_refuses(tmp_path):
    with pytest.raises(DirectionRefusal, match="not found"):
        load_seedset(str(tmp_path / "nope.seedset.json"))


def test_load_seedset_malformed_json_refuses(tmp_path):
    path = tmp_path / "bad.seedset.json"
    path.write_text("{not json")
    with pytest.raises(DirectionRefusal, match="not valid JSON"):
        load_seedset(str(path))


def test_load_seedset_missing_manifest_refuses(tmp_path):
    path = tmp_path / "x.seedset.json"
    path.write_text(json.dumps({"seeds": [], "negatives": []}))
    with pytest.raises(DirectionRefusal, match="manifest"):
        load_seedset(str(path))


def test_load_seedset_missing_corpus_snapshot_refuses(tmp_path, synthetic_scene):
    blob = dict(synthetic_scene["seedset"])
    blob["manifest"] = {k: v for k, v in blob["manifest"].items() if k != "corpus_snapshot"}
    path = tmp_path / "x.seedset.json"
    path.write_text(json.dumps(blob))
    with pytest.raises(DirectionRefusal, match="corpus_snapshot"):
        load_seedset(str(path))


def test_load_seedset_empty_seeds_refuses(tmp_path, synthetic_scene):
    blob = json.loads(json.dumps(synthetic_scene["seedset"]))
    blob["seeds"] = []
    path = tmp_path / "x.seedset.json"
    path.write_text(json.dumps(blob))
    with pytest.raises(DirectionRefusal, match="seeds"):
        load_seedset(str(path))


def test_load_seedset_row_missing_rowid_mm_refuses(tmp_path, synthetic_scene):
    blob = json.loads(json.dumps(synthetic_scene["seedset"]))
    del blob["seeds"][0]["rowid_mm"]
    path = tmp_path / "x.seedset.json"
    path.write_text(json.dumps(blob))
    with pytest.raises(DirectionRefusal, match="rowid_mm"):
        load_seedset(str(path))


# --------------------------------------------------------------------------- #
# D2 step 1: identity-mismatch refusal (the acceptance criterion named in #57).
# --------------------------------------------------------------------------- #

def test_refuses_on_embedding_model_id_mismatch(synthetic_scene):
    seedset = json.loads(json.dumps(synthetic_scene["seedset"]))
    seedset["manifest"]["corpus_snapshot"]["embedding_model_id"] = "some-other-model"
    with pytest.raises(DirectionRefusal, match="embedding_model_id"):
        refuse_unless_seedset_matches_calibration(seedset, synthetic_scene["calibration"])


def test_refuses_on_memmap_sha_mismatch(synthetic_scene):
    seedset = json.loads(json.dumps(synthetic_scene["seedset"]))
    seedset["manifest"]["corpus_snapshot"]["vector_memmap_sha256"] = "mismatched-sha"
    with pytest.raises(DirectionRefusal, match="source_memmap_sha256"):
        refuse_unless_seedset_matches_calibration(seedset, synthetic_scene["calibration"])


def test_matching_identity_does_not_raise(synthetic_scene):
    refuse_unless_seedset_matches_calibration(synthetic_scene["seedset"], synthetic_scene["calibration"])


# --------------------------------------------------------------------------- #
# Memmap row reader.
# --------------------------------------------------------------------------- #

def test_read_memmap_rows_returns_correct_rows(synthetic_scene):
    rows = read_memmap_rows(
        synthetic_scene["memmap_path"], [0, 1, 2], DIM, synthetic_scene["n_total"]
    )
    np.testing.assert_allclose(rows, synthetic_scene["x"][[0, 1, 2]], rtol=1e-5)


def test_read_memmap_rows_rejects_out_of_range_rowid(synthetic_scene):
    with pytest.raises(DirectionRefusal, match="out of range"):
        read_memmap_rows(
            synthetic_scene["memmap_path"], [synthetic_scene["n_total"] + 5], DIM, synthetic_scene["n_total"]
        )


def test_read_memmap_rows_rejects_size_mismatch(synthetic_scene):
    with pytest.raises(DirectionRefusal, match="size"):
        read_memmap_rows(synthetic_scene["memmap_path"], [0], DIM, synthetic_scene["n_total"] + 1)


# --------------------------------------------------------------------------- #
# Centroid math correctness + direction recovery (analytically checkable).
# --------------------------------------------------------------------------- #

def test_mean_center_subtracts_mu():
    vecs = np.array([[1.0, 2.0], [3.0, 4.0]])
    mu = np.array([1.0, 1.0])
    np.testing.assert_allclose(mean_center(vecs, mu), [[0.0, 1.0], [2.0, 3.0]])


def test_compute_direction_recovers_known_axis(synthetic_scene):
    true_axis = synthetic_scene["true_axis"]
    seed_vecs = read_memmap_rows(
        synthetic_scene["memmap_path"], synthetic_scene["seed_rowids"], DIM, synthetic_scene["n_total"]
    )
    neg_vecs = read_memmap_rows(
        synthetic_scene["memmap_path"], synthetic_scene["negative_rowids"], DIM, synthetic_scene["n_total"]
    )
    centered_seeds = mean_center(seed_vecs, synthetic_scene["mu"])
    centered_negs = mean_center(neg_vecs, synthetic_scene["mu"])
    result = compute_direction(centered_seeds, centered_negs)
    cosine_to_true = float(np.dot(result["unit_axis"], true_axis))
    # At this fixture's n=41 (matching the real comparative-perception
    # seedset's scale) with noise comparable to the displacement, recovery is
    # clearly non-random but not exact -- the analytically checkable claim is
    # "well above chance" (a random unit vector in 16-d has cosine
    # concentrated near 0), not near-1.0 recovery.
    assert cosine_to_true > 0.5
    assert result["pole_separation"] > 0


def test_compute_direction_underdetermined_on_coincident_poles():
    centered_seeds = np.zeros((5, 4))
    centered_negs = np.zeros((5, 4))
    result = compute_direction(centered_seeds, centered_negs)
    assert result["error"] == "axis underdetermined"
    assert result["pole_separation"] == 0.0


# --------------------------------------------------------------------------- #
# ONE-DOOR conformance: typed_exemplars axis (build_axis) and seedset_centroids
# axis (compute_direction) project identically given identical poles.
# --------------------------------------------------------------------------- #

def test_one_door_seedset_centroid_axis_matches_typed_exemplar_axis():
    """The acceptance criterion named in issue #57 / ADR-SKM-008 line 186:
    'a typed_exemplars axis and a seedset_centroids axis with identical poles
    project identically.' compute_direction's only new work is *sourcing* the
    poles from corpus-row centroids; once poles are in hand it must reuse
    axis_alignment.build_axis verbatim -- so given the SAME pole vectors, both
    entry points must produce the identical unit axis and pole_separation."""
    rng = np.random.default_rng(3)
    dim = 12
    pos_vecs = rng.normal(size=(20, dim))  # "typed exemplar" embeddings
    neg_vecs = rng.normal(size=(20, dim))  # "typed exemplar" negative embeddings
    neg_pole = neg_vecs.mean(axis=0)

    # typed_exemplars path: axis_alignment.build_axis directly.
    typed_axis, typed_sep = build_axis(pos_vecs, neg_pole)

    # seedset_centroids path: direction.compute_direction, fed the IDENTICAL
    # pole inputs (pos_vecs standing in for "centered seeds", neg_vecs for
    # "centered negatives" -- centering is a preprocessing step upstream of
    # this shared kernel, not part of what's being compared here).
    seedset_result = compute_direction(pos_vecs, neg_vecs, min_pole_separation=0.0)

    np.testing.assert_allclose(seedset_result["unit_axis"], typed_axis)
    assert seedset_result["pole_separation"] == pytest.approx(typed_sep)

    # And the resulting projection of an arbitrary probe vector is identical.
    probe = rng.normal(size=dim)
    assert np.dot(probe, seedset_result["unit_axis"]) == pytest.approx(np.dot(probe, typed_axis))


def test_compute_direction_does_not_reimplement_build_axis():
    """Code-review-shaped assertion: compute_direction calls build_axis rather
    than duplicating the centroid-difference math inline."""
    import inspect

    source = inspect.getsource(compute_direction)
    assert "build_axis(" in source


# --------------------------------------------------------------------------- #
# AUC (Mann-Whitney) numeric correctness.
# --------------------------------------------------------------------------- #

def test_auc_perfect_separation_is_one():
    pos = np.array([10.0, 11.0, 12.0])
    neg = np.array([1.0, 2.0, 3.0])
    assert auc_score(pos, neg) == pytest.approx(1.0)


def test_auc_perfect_anti_separation_is_zero():
    pos = np.array([1.0, 2.0, 3.0])
    neg = np.array([10.0, 11.0, 12.0])
    assert auc_score(pos, neg) == pytest.approx(0.0)


def test_auc_chance_separation_is_near_half():
    rng = np.random.default_rng(11)
    pos = rng.normal(size=5000)
    neg = rng.normal(size=5000)
    assert auc_score(pos, neg) == pytest.approx(0.5, abs=0.05)


def test_auc_ties_score_one_half():
    pos = np.array([1.0, 1.0])
    neg = np.array([1.0, 1.0])
    assert auc_score(pos, neg) == pytest.approx(0.5)


def test_auc_requires_nonempty_inputs():
    with pytest.raises(ValueError):
        auc_score(np.array([]), np.array([1.0]))


# --------------------------------------------------------------------------- #
# D3 held-out AUC + injected-leakage alarm.
# --------------------------------------------------------------------------- #

def test_held_out_auc_reasonable_on_moderate_separation(synthetic_scene):
    seed_vecs = read_memmap_rows(
        synthetic_scene["memmap_path"], synthetic_scene["seed_rowids"], DIM, synthetic_scene["n_total"]
    )
    neg_vecs = read_memmap_rows(
        synthetic_scene["memmap_path"], synthetic_scene["negative_rowids"], DIM, synthetic_scene["n_total"]
    )
    centered_seeds = mean_center(seed_vecs, synthetic_scene["mu"])
    centered_negs = mean_center(neg_vecs, synthetic_scene["mu"])
    seed_chunk_ids = [r["chunk_id"] for r in synthetic_scene["seedset"]["seeds"]]
    paired_ids = [r["paired_seed_chunk_id"] for r in synthetic_scene["seedset"]["negatives"]]

    result = held_out_separation_auc(
        centered_seeds, centered_negs, paired_seed_chunk_ids=paired_ids, seed_chunk_ids=seed_chunk_ids,
        rng=np.random.default_rng(0),
    )
    assert "error" not in result
    assert 0.0 <= result["auc"] <= 1.0
    assert result["n_held_seeds"] > 0
    assert result["leakage_suspected"] is False


def test_held_out_auc_trips_leakage_alarm_on_injected_perfect_separation():
    """An injected leakage case: seeds and negatives are near-perfectly
    separable (as if the regex were also the embedding). AUC on the held-out
    split must exceed the 0.98 threshold and the alarm must fire."""
    rng = np.random.default_rng(5)
    dim = 10
    n = 30
    true_axis = rng.normal(size=dim)
    true_axis /= np.linalg.norm(true_axis)
    mu = np.zeros(dim)

    # Huge, clean displacement + tiny noise -> near-perfect separability.
    seeds = rng.normal(size=(n, dim)) * 0.01 + 20.0 * true_axis
    negs = rng.normal(size=(n, dim)) * 0.01

    centered_seeds = mean_center(seeds, mu)
    centered_negs = mean_center(negs, mu)
    seed_chunk_ids = [f"s{i}" for i in range(n)]
    paired_ids = [f"s{i}" for i in range(n)]

    result = held_out_separation_auc(
        centered_seeds, centered_negs, paired_seed_chunk_ids=paired_ids, seed_chunk_ids=seed_chunk_ids,
        rng=np.random.default_rng(0),
    )
    assert result["auc"] > d.DEFAULT_LEAKAGE_AUC_THRESHOLD
    assert result["leakage_suspected"] is True


def test_held_out_auc_refuses_on_too_few_pairs():
    dim = 8
    centered_seeds = np.random.default_rng(0).normal(size=(3, dim))
    centered_negs = np.random.default_rng(1).normal(size=(3, dim))
    seed_chunk_ids = ["s0", "s1", "s2"]
    paired_ids = ["s0", "s1", "s2"]  # only 3 pairs, below MIN_PAIRS_FOR_HELD_OUT_SPLIT (4)
    result = held_out_separation_auc(
        centered_seeds, centered_negs, paired_seed_chunk_ids=paired_ids, seed_chunk_ids=seed_chunk_ids
    )
    assert "error" in result
    assert result["n_pairs_available"] == 3


def test_held_out_auc_handles_unpaired_rows(synthetic_scene):
    """Seeds/negatives with no matching partner train-only, never straddle
    the split (per D3's 'a seed and its matched negative never straddle the
    split')."""
    seed_chunk_ids = [f"s{i}" for i in range(10)]
    paired_ids = [f"s{i}" for i in range(8)] + [None, "no-such-seed"]
    rng = np.random.default_rng(0)
    centered_seeds = rng.normal(size=(10, 6))
    centered_negs = rng.normal(size=(10, 6))
    result = held_out_separation_auc(
        centered_seeds, centered_negs, paired_seed_chunk_ids=paired_ids, seed_chunk_ids=seed_chunk_ids,
        rng=np.random.default_rng(0),
    )
    assert "error" not in result


# --------------------------------------------------------------------------- #
# D3 topic-control check.
# --------------------------------------------------------------------------- #

def test_topic_control_flags_functional_not_topical(synthetic_scene):
    seed_vecs = read_memmap_rows(
        synthetic_scene["memmap_path"], synthetic_scene["seed_rowids"], DIM, synthetic_scene["n_total"]
    )
    neg_vecs = read_memmap_rows(
        synthetic_scene["memmap_path"], synthetic_scene["negative_rowids"], DIM, synthetic_scene["n_total"]
    )
    background_rowids = list(range(82, 182))
    random_vecs = read_memmap_rows(synthetic_scene["memmap_path"], background_rowids, DIM, synthetic_scene["n_total"])

    centered_seeds = mean_center(seed_vecs, synthetic_scene["mu"])
    centered_negs = mean_center(neg_vecs, synthetic_scene["mu"])
    centered_random = mean_center(random_vecs, synthetic_scene["mu"])

    direction_result = compute_direction(centered_seeds, centered_negs)
    check = topic_control_check(
        direction_result["unit_axis"], held_out_negatives=centered_negs,
        held_out_seeds=centered_seeds, random_topic_sample=centered_random,
    )
    # Random background is drawn from the same isotropic noise as negatives
    # (no displacement along the axis) -> should sit near chance vs negatives.
    assert abs(check["random_vs_negative_auc"] - 0.5) < 0.2


def test_topic_control_reports_sample_size():
    dim = 6
    rng = np.random.default_rng(2)
    unit_axis = rng.normal(size=dim)
    unit_axis /= np.linalg.norm(unit_axis)
    held_neg = rng.normal(size=(10, dim))
    held_seed = rng.normal(size=(10, dim)) + 3 * unit_axis
    random_sample = rng.normal(size=(50, dim))
    check = topic_control_check(unit_axis, held_neg, held_seed, random_sample)
    assert check["random_topic_sample_n"] == 50


# --------------------------------------------------------------------------- #
# D3 bootstrap stability.
# --------------------------------------------------------------------------- #

def test_bootstrap_stability_high_cosine_on_strong_signal(synthetic_scene):
    seed_vecs = read_memmap_rows(
        synthetic_scene["memmap_path"], synthetic_scene["seed_rowids"], DIM, synthetic_scene["n_total"]
    )
    neg_vecs = read_memmap_rows(
        synthetic_scene["memmap_path"], synthetic_scene["negative_rowids"], DIM, synthetic_scene["n_total"]
    )
    centered_seeds = mean_center(seed_vecs, synthetic_scene["mu"])
    centered_negs = mean_center(neg_vecs, synthetic_scene["mu"])
    result = bootstrap_stability(centered_seeds, centered_negs, b=50, rng=np.random.default_rng(0))
    assert "error" not in result
    assert 0.0 <= result["mean_pairwise_cosine"] <= 1.0
    assert result["n_successful_resamples"] > 0


def test_bootstrap_stability_under_determined_on_thin_era():
    """A too-thin era (n=2 seeds/negatives) yields an unstable, near-random
    direction under resampling -- bootstrap cosine should be low, and the
    result flagged under_determined=True (refuse to promote)."""
    rng = np.random.default_rng(9)
    dim = 20
    # Pure noise, no shared displacement axis at all -- each bootstrap draw
    # from n=2 samples with replacement produces a near-arbitrary direction.
    centered_seeds = rng.normal(size=(2, dim)) * 5.0
    centered_negs = rng.normal(size=(2, dim)) * 5.0
    result = bootstrap_stability(
        centered_seeds, centered_negs, b=100, min_pole_separation=0.0, rng=np.random.default_rng(1)
    )
    assert "error" not in result
    assert result["under_determined"] is True
    assert result["mean_pairwise_cosine"] < d.DEFAULT_MIN_BOOTSTRAP_COSINE


def test_bootstrap_stability_reports_ci():
    rng = np.random.default_rng(4)
    dim = 8
    centered_seeds = rng.normal(size=(20, dim)) + np.array([3] + [0] * (dim - 1))
    centered_negs = rng.normal(size=(20, dim))
    result = bootstrap_stability(centered_seeds, centered_negs, b=50, rng=np.random.default_rng(0))
    lo, hi = result["pole_separation_ci_95"]
    assert lo <= hi


# --------------------------------------------------------------------------- #
# D3 null reference.
# --------------------------------------------------------------------------- #

def test_null_reference_mu0_near_zero_for_centered_input():
    rng = np.random.default_rng(6)
    dim = 5
    unit_axis = rng.normal(size=dim)
    unit_axis /= np.linalg.norm(unit_axis)
    # Already mean-centered synthetic corpus sample (mean ~0 by construction).
    centered_corpus = rng.normal(size=(500, dim))
    result = null_reference(centered_corpus, unit_axis)
    assert abs(result["mu0"]) < 0.1
    assert result["sigma0"] > 0
    assert result["n"] == 500


# --------------------------------------------------------------------------- #
# verdict_from_diagnostics precedence.
# --------------------------------------------------------------------------- #

def test_verdict_usable_when_all_diagnostics_clean():
    direction_result = {"unit_axis": np.zeros(4), "pole_separation": 1.0}
    held_out = {"auc": 0.8, "leakage_suspected": False}
    bootstrap = {"mean_pairwise_cosine": 0.9, "under_determined": False}
    assert verdict_from_diagnostics(direction_result, held_out, bootstrap) == "usable"


def test_verdict_leakage_suspected_overrides_bootstrap():
    direction_result = {"unit_axis": np.zeros(4), "pole_separation": 1.0}
    held_out = {"auc": 0.99, "leakage_suspected": True}
    bootstrap = {"mean_pairwise_cosine": 0.9, "under_determined": False}
    assert verdict_from_diagnostics(direction_result, held_out, bootstrap) == "leakage-suspected"


def test_verdict_under_determined_when_bootstrap_fails():
    direction_result = {"unit_axis": np.zeros(4), "pole_separation": 1.0}
    held_out = {"auc": 0.7, "leakage_suspected": False}
    bootstrap = {"mean_pairwise_cosine": 0.3, "under_determined": True}
    assert verdict_from_diagnostics(direction_result, held_out, bootstrap) == "under-determined"


def test_verdict_under_determined_when_direction_extraction_failed():
    direction_result = {"error": "axis underdetermined"}
    assert verdict_from_diagnostics(direction_result, {}, {}) == "under-determined"


def test_verdict_under_determined_when_held_out_split_failed_despite_clean_bootstrap():
    """A held-out split that could not be measured (too few paired groups)
    must not read as a clean pass. Even paired with a high-stability bootstrap
    and a successfully extracted axis, an unmeasured leakage diagnostic yields
    "under-determined" -- the error guard on held_out takes precedence over
    both the leakage check and the usable fall-through."""
    direction_result = {"unit_axis": np.zeros(4), "pole_separation": 1.0}
    held_out = {"error": "too few paired groups to split"}
    bootstrap = {"mean_pairwise_cosine": 0.95, "under_determined": False}
    assert verdict_from_diagnostics(direction_result, held_out, bootstrap) == "under-determined"


# --------------------------------------------------------------------------- #
# initialize_direction_core: orchestration + era-scoped filtering.
# --------------------------------------------------------------------------- #

def test_initialize_direction_core_happy_path(synthetic_scene):
    seed_vecs = read_memmap_rows(
        synthetic_scene["memmap_path"], synthetic_scene["seed_rowids"], DIM, synthetic_scene["n_total"]
    )
    neg_vecs = read_memmap_rows(
        synthetic_scene["memmap_path"], synthetic_scene["negative_rowids"], DIM, synthetic_scene["n_total"]
    )
    random_vecs = read_memmap_rows(synthetic_scene["memmap_path"], list(range(82, 182)), DIM, synthetic_scene["n_total"])
    null_vecs = read_memmap_rows(synthetic_scene["memmap_path"], list(range(82, 282)), DIM, synthetic_scene["n_total"])

    result = initialize_direction_core(
        calibration=synthetic_scene["calibration"],
        seedset=synthetic_scene["seedset"],
        seed_vecs=seed_vecs,
        negative_vecs=neg_vecs,
        random_topic_vecs=random_vecs,
        corpus_null_sample_vecs=null_vecs,
        rng=np.random.default_rng(0),
    )
    assert "error" not in result
    assert result["verdict"] in {"usable", "under-determined", "leakage-suspected"}
    assert result["n_seeds"] == 41
    assert result["n_negatives"] == 41


def test_initialize_direction_core_era_filter_matches_zero_rows(synthetic_scene):
    seedset = json.loads(json.dumps(synthetic_scene["seedset"]))
    # All rows have era=None in the fixture; a named era matches nothing.
    result = initialize_direction_core(
        calibration=synthetic_scene["calibration"],
        seedset=seedset,
        seed_vecs=np.zeros((1, DIM)),
        negative_vecs=np.zeros((1, DIM)),
        random_topic_vecs=np.zeros((1, DIM)),
        corpus_null_sample_vecs=np.zeros((1, DIM)),
        era="Opus 4.8",
    )
    assert "error" in result
    assert "era filter" in result["error"]


def test_initialize_direction_core_era_scoped_extraction(tmp_path):
    """Era-scoped extraction (D2 step 4): seeds/negatives are filtered by
    denormalized era; a per-era direction uses only that era's rows."""
    rng = np.random.default_rng(21)
    dim = 10
    mu = np.zeros(dim)
    true_axis = rng.normal(size=dim)
    true_axis /= np.linalg.norm(true_axis)

    n_per_era = 10
    eras = ["Opus 4.8"] * n_per_era + ["Opus 4.7"] * n_per_era
    n_seeds = n_negs = 2 * n_per_era
    n_background = 100
    n_total = n_seeds + n_negs + n_background
    x = rng.normal(size=(n_total, dim)) * 0.3
    x[:n_seeds] += 1.5 * true_axis  # all seeds displaced regardless of era
    seed_rowids = list(range(n_seeds))
    negative_rowids = list(range(n_seeds, n_seeds + n_negs))

    memmap_path = _write_memmap(tmp_path, x)
    calibration = _make_calibration(mu, memmap_path, n_total)
    seedset = _seedset_dict(
        "pattern", EMBEDDING_MODEL_ID, MEMMAP_SHA, seed_rowids, negative_rowids, eras=eras
    )

    seed_vecs = read_memmap_rows(memmap_path, seed_rowids, dim, n_total)
    neg_vecs = read_memmap_rows(memmap_path, negative_rowids, dim, n_total)
    background_rowids = list(range(n_seeds + n_negs, n_total))
    random_vecs = read_memmap_rows(memmap_path, background_rowids, dim, n_total)

    result = initialize_direction_core(
        calibration=calibration,
        seedset=seedset,
        seed_vecs=seed_vecs,
        negative_vecs=neg_vecs,
        random_topic_vecs=random_vecs,
        corpus_null_sample_vecs=random_vecs,
        era="Opus 4.8",
        rng=np.random.default_rng(0),
    )
    assert "error" not in result
    assert result["n_seeds"] == n_per_era
    assert result["n_negatives"] == n_per_era


# --------------------------------------------------------------------------- #
# Direction artifact: write/load round trip + refuse-on-mismatch (mutated sha).
# --------------------------------------------------------------------------- #

def test_write_and_load_direction_artifact_round_trip(tmp_path, synthetic_scene):
    unit_axis = np.ones(DIM) / np.sqrt(DIM)
    diagnostics = {
        "pole_separation": 1.23,
        "verdict": "usable",
        "n_seeds": 41,
        "n_negatives": 41,
        "held_out_auc": {"auc": 0.8},
        "topic_control": {"seed_vs_negative_auc": 0.8},
        "bootstrap": {"mean_pairwise_cosine": 0.9},
        "null_reference": {"mu0": 0.0, "sigma0": 1.0, "n": 100},
    }
    npz_path, json_path = write_direction_artifact(
        out_dir=str(tmp_path / "directions"),
        pattern_id="comparative-perception",
        era=None,
        unit_axis=unit_axis,
        calibration=synthetic_scene["calibration"],
        seedset_manifest=synthetic_scene["seedset"]["manifest"],
        diagnostics=diagnostics,
    )
    assert os.path.isfile(npz_path)
    assert os.path.isfile(json_path)

    loaded = load_direction(json_path)
    assert loaded.pattern_id == "comparative-perception"
    assert loaded.era is None
    assert loaded.verdict == "usable"
    assert loaded.embedding_model_id == EMBEDDING_MODEL_ID
    np.testing.assert_allclose(loaded.unit_axis, unit_axis)


def test_write_direction_artifact_era_scoped_filename(tmp_path, synthetic_scene):
    unit_axis = np.ones(DIM) / np.sqrt(DIM)
    diagnostics = {
        "pole_separation": 1.0, "verdict": "usable", "n_seeds": 5, "n_negatives": 5,
        "held_out_auc": {}, "topic_control": {}, "bootstrap": {}, "null_reference": {},
    }
    npz_path, json_path = write_direction_artifact(
        out_dir=str(tmp_path / "directions"),
        pattern_id="comparative-perception",
        era="Opus 4.8",
        unit_axis=unit_axis,
        calibration=synthetic_scene["calibration"],
        seedset_manifest=synthetic_scene["seedset"]["manifest"],
        diagnostics=diagnostics,
    )
    assert "comparative-perception.Opus_4.8.direction" in os.path.basename(json_path)


def test_load_direction_missing_file_hard_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_direction(str(tmp_path / "nope.direction.json"))


def test_load_direction_headerless_refused(tmp_path):
    path = tmp_path / "x.direction.json"
    path.write_text(json.dumps({"verdict": "usable"}))
    with pytest.raises(ValueError, match="header"):
        load_direction(str(path))


@pytest.mark.parametrize(
    "missing_key",
    ["regime", "embedding_model_id", "source_memmap_sha256", "pattern_id", "era", "dim"],
)
def test_load_direction_missing_required_header_key_refused(tmp_path, missing_key):
    header = {
        "regime": "functional-direction",
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "source_memmap_sha256": MEMMAP_SHA,
        "pattern_id": "p",
        "era": None,
        "dim": DIM,
    }
    del header[missing_key]
    path = tmp_path / "x.direction.json"
    path.write_text(json.dumps({"header": header}))
    with pytest.raises(ValueError, match="missing required keys"):
        load_direction(str(path))


def test_load_direction_wrong_regime_refused(tmp_path):
    header = {
        "regime": "corpus-calibration",  # wrong regime
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "source_memmap_sha256": MEMMAP_SHA,
        "pattern_id": "p",
        "era": None,
        "dim": DIM,
    }
    path = tmp_path / "x.direction.json"
    path.write_text(json.dumps({"header": header}))
    with pytest.raises(ValueError, match="regime"):
        load_direction(str(path))


def test_direction_artifact_round_trip_detects_mutated_sha(tmp_path, synthetic_scene):
    """Refuse-on-mismatch round trip (CONSERVE-DATA-BOUNDARY enforcement
    surface): build -> load -> a consumer checking the loaded artifact's
    source_memmap_sha256 against a freshly-recomputed corpus sha detects a
    since-mutated corpus, exactly as Calibration.refuse_unless_matches does
    for the calibration layer."""
    unit_axis = np.ones(DIM) / np.sqrt(DIM)
    diagnostics = {
        "pole_separation": 1.0, "verdict": "usable", "n_seeds": 5, "n_negatives": 5,
        "held_out_auc": {}, "topic_control": {}, "bootstrap": {}, "null_reference": {},
    }
    _npz_path, json_path = write_direction_artifact(
        out_dir=str(tmp_path / "directions"),
        pattern_id="p",
        era=None,
        unit_axis=unit_axis,
        calibration=synthetic_scene["calibration"],
        seedset_manifest=synthetic_scene["seedset"]["manifest"],
        diagnostics=diagnostics,
    )
    loaded = load_direction(json_path)
    assert loaded.source_memmap_sha256 == synthetic_scene["calibration"].source_memmap_sha256

    # A "rebuilt" calibration (mutated corpus) now has a different sha; a
    # consumer must be able to detect the direction artifact is stale.
    mutated_calibration = _make_calibration(
        synthetic_scene["mu"], synthetic_scene["memmap_path"], synthetic_scene["n_total"], sha="mutated-sha"
    )
    assert loaded.source_memmap_sha256 != mutated_calibration.source_memmap_sha256


# --------------------------------------------------------------------------- #
# MCP handler: end-to-end happy path + refusal via the async handler.
# --------------------------------------------------------------------------- #

def test_initialize_direction_handler_happy_path(tmp_path, synthetic_scene):
    seedset_path = tmp_path / "pattern.seedset.json"
    seedset_path.write_text(json.dumps(synthetic_scene["seedset"]))

    calibration_json = tmp_path / "cal.calibration.json"
    calibration_npz = tmp_path / "cal.calibration.npz"
    np.savez(calibration_npz, mu=synthetic_scene["mu"])
    calibration_json.write_text(json.dumps({"header": synthetic_scene["calibration"].header, "mu_norm": 0.1}))

    out_dir = tmp_path / "directions"
    result = _run(
        initialize_direction(
            manager=None,
            args={
                "seedset_ref": str(seedset_path),
                "calibration_ref": str(calibration_json),
                "out_dir": str(out_dir),
                "bootstrap_b": 20,
                "seed": 0,
            },
        )
    )
    assert "error" not in result, result
    assert result["verdict"] in {"usable", "under-determined", "leakage-suspected"}
    assert os.path.isfile(result["direction_ref"])
    assert os.path.isfile(result["npz_path"])


def test_initialize_direction_handler_refuses_on_identity_mismatch(tmp_path, synthetic_scene):
    seedset = json.loads(json.dumps(synthetic_scene["seedset"]))
    seedset["manifest"]["corpus_snapshot"]["embedding_model_id"] = "wrong-model"
    seedset_path = tmp_path / "pattern.seedset.json"
    seedset_path.write_text(json.dumps(seedset))

    calibration_json = tmp_path / "cal.calibration.json"
    calibration_npz = tmp_path / "cal.calibration.npz"
    np.savez(calibration_npz, mu=synthetic_scene["mu"])
    calibration_json.write_text(json.dumps({"header": synthetic_scene["calibration"].header, "mu_norm": 0.1}))

    result = _run(
        initialize_direction(
            manager=None,
            args={"seedset_ref": str(seedset_path), "calibration_ref": str(calibration_json)},
        )
    )
    assert "error" in result
    assert "embedding_model_id" in result["error"]


def test_initialize_direction_handler_requires_refs():
    result = _run(initialize_direction(manager=None, args={}))
    assert "error" in result
    assert "seedset_ref" in result["error"]

    result2 = _run(initialize_direction(manager=None, args={"seedset_ref": "x"}))
    assert "error" in result2
    assert "calibration_ref" in result2["error"]


# --------------------------------------------------------------------------- #
# server.py registration: complete-surface test (ARCHITECTURE Rule 2).
# --------------------------------------------------------------------------- #

_SERVER_PY_PATH = (
    Path(__file__).resolve().parent.parent
    / "semantic_kinematics"
    / "mcp"
    / "server.py"
)


def _read_server_source() -> str:
    """Read server.py's source as TEXT, never importing it.

    Importing semantic_kinematics.mcp.server against the installed ``mcp``
    package (2.0.0) raises AttributeError at module scope
    (``Server.list_tools`` decorator no longer exists) -- a pre-existing,
    out-of-scope F0 defect (sk-mcp#63-class API drift, confirmed present on
    clean origin/main with this worktree's changes stashed out; see PR body).
    The registration conformance this test needs -- "every tool has a
    call_tool dispatch arm" -- is checkable from source text alone, so the
    import is avoided entirely rather than working around the drift here.
    """
    return _SERVER_PY_PATH.read_text(encoding="utf-8")


def test_direction_tools_all_have_a_call_tool_dispatch_arm():
    """ARCHITECTURE.md 'no orphan tool' instant-fail check: every tool
    direction.get_tools() declares must be reachable from server.py's
    call_tool dispatch."""
    server_src = _read_server_source()
    for tool in d.get_tools():
        assert f'name == "{tool.name}"' in server_src, (
            f"tool {tool.name!r} has no call_tool dispatch arm in server.py"
        )


def test_direction_module_registered_in_list_tools():
    server_src = _read_server_source()
    assert "direction.get_tools()" in server_src
    assert "from semantic_kinematics.mcp.commands import (" in server_src
    assert "direction" in server_src


# --------------------------------------------------------------------------- #
# Stateless-core: same-in/same-out (ARCHITECTURE Rule 1).
# --------------------------------------------------------------------------- #

def test_initialize_direction_core_is_deterministic_same_in_same_out(synthetic_scene):
    seed_vecs = read_memmap_rows(
        synthetic_scene["memmap_path"], synthetic_scene["seed_rowids"], DIM, synthetic_scene["n_total"]
    )
    neg_vecs = read_memmap_rows(
        synthetic_scene["memmap_path"], synthetic_scene["negative_rowids"], DIM, synthetic_scene["n_total"]
    )
    random_vecs = read_memmap_rows(synthetic_scene["memmap_path"], list(range(82, 182)), DIM, synthetic_scene["n_total"])
    null_vecs = read_memmap_rows(synthetic_scene["memmap_path"], list(range(82, 282)), DIM, synthetic_scene["n_total"])

    kwargs = dict(
        calibration=synthetic_scene["calibration"],
        seedset=synthetic_scene["seedset"],
        seed_vecs=seed_vecs,
        negative_vecs=neg_vecs,
        random_topic_vecs=random_vecs,
        corpus_null_sample_vecs=null_vecs,
    )
    result1 = initialize_direction_core(**kwargs, rng=np.random.default_rng(42))
    result2 = initialize_direction_core(**kwargs, rng=np.random.default_rng(42))

    np.testing.assert_allclose(result1["unit_axis"], result2["unit_axis"])
    assert result1["verdict"] == result2["verdict"]
    assert result1["held_out_auc"] == result2["held_out_auc"]


# --------------------------------------------------------------------------- #
# Additional refusal / propagation branches (coverage-completing).
# --------------------------------------------------------------------------- #

def test_load_seedset_empty_negatives_refuses(tmp_path, synthetic_scene):
    blob = json.loads(json.dumps(synthetic_scene["seedset"]))
    blob["negatives"] = []
    path = tmp_path / "x.seedset.json"
    path.write_text(json.dumps(blob))
    with pytest.raises(DirectionRefusal, match="negatives"):
        load_seedset(str(path))


def test_held_out_separation_auc_propagates_train_direction_error():
    """If the TRAIN split's poles are coincident (underdetermined), the error
    from compute_direction propagates through held_out_separation_auc rather
    than being swallowed."""
    dim = 4
    n = 6
    centered_seeds = np.zeros((n, dim))
    centered_negatives = np.zeros((n, dim))
    seed_chunk_ids = [f"s{i}" for i in range(n)]
    paired_ids = [f"s{i}" for i in range(n)]
    result = held_out_separation_auc(
        centered_seeds, centered_negatives,
        paired_seed_chunk_ids=paired_ids, seed_chunk_ids=seed_chunk_ids,
        rng=np.random.default_rng(0),
    )
    assert result["error"] == "axis underdetermined"


def test_load_direction_missing_npz_hard_fails(tmp_path):
    header = {
        "regime": "functional-direction",
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "source_memmap_sha256": MEMMAP_SHA,
        "pattern_id": "p",
        "era": None,
        "dim": DIM,
    }
    path = tmp_path / "p.direction.json"
    path.write_text(json.dumps({"header": header}))
    # No sibling .npz written.
    with pytest.raises(FileNotFoundError, match="npz"):
        load_direction(str(path))


def test_load_direction_missing_unit_axis_array_refused(tmp_path):
    header = {
        "regime": "functional-direction",
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "source_memmap_sha256": MEMMAP_SHA,
        "pattern_id": "p",
        "era": None,
        "dim": DIM,
    }
    json_path = tmp_path / "p.direction.json"
    json_path.write_text(json.dumps({"header": header}))
    npz_path = tmp_path / "p.direction.npz"
    np.savez(npz_path, not_unit_axis=np.zeros(4))
    with pytest.raises(ValueError, match="unit_axis"):
        load_direction(str(json_path))


def test_initialize_direction_handler_era_matches_zero_seed_rows(tmp_path, synthetic_scene):
    seedset_path = tmp_path / "pattern.seedset.json"
    seedset_path.write_text(json.dumps(synthetic_scene["seedset"]))
    calibration_json = tmp_path / "cal.calibration.json"
    calibration_npz = tmp_path / "cal.calibration.npz"
    np.savez(calibration_npz, mu=synthetic_scene["mu"])
    calibration_json.write_text(json.dumps({"header": synthetic_scene["calibration"].header, "mu_norm": 0.1}))

    result = _run(
        initialize_direction(
            manager=None,
            args={
                "seedset_ref": str(seedset_path),
                "calibration_ref": str(calibration_json),
                "era": "nonexistent-era",
            },
        )
    )
    assert "error" in result
    assert "era filter" in result["error"]


def test_initialize_direction_handler_refuses_on_memmap_row_out_of_range(tmp_path, synthetic_scene):
    """A seedset whose rowid_mm exceeds the calibration's declared n_used
    (e.g. the memmap was rebuilt smaller) refuses via read_memmap_rows,
    propagated as a structured error rather than an uncaught IndexError."""
    seedset = json.loads(json.dumps(synthetic_scene["seedset"]))
    seedset["seeds"][0]["rowid_mm"] = synthetic_scene["n_total"] + 1000
    seedset_path = tmp_path / "pattern.seedset.json"
    seedset_path.write_text(json.dumps(seedset))

    calibration_json = tmp_path / "cal.calibration.json"
    calibration_npz = tmp_path / "cal.calibration.npz"
    np.savez(calibration_npz, mu=synthetic_scene["mu"])
    calibration_json.write_text(json.dumps({"header": synthetic_scene["calibration"].header, "mu_norm": 0.1}))

    result = _run(
        initialize_direction(
            manager=None,
            args={"seedset_ref": str(seedset_path), "calibration_ref": str(calibration_json)},
        )
    )
    assert "error" in result
    assert "out of range" in result["error"]


# --------------------------------------------------------------------------- #
# tvi_root path resolution: the calibration's source_memmap_path is recorded
# repo-root-relative (files-only boundary, ADR-SKM-008 Option E); sk-mcp has
# no notion of "the tvi repo root" except what the caller supplies.
# --------------------------------------------------------------------------- #

def _write_relative_path_scene(tmp_path, synthetic_scene):
    """Re-home the synthetic scene's memmap under a fake tvi_root, and write
    a calibration header carrying a REPO-ROOT-RELATIVE source_memmap_path
    (mirroring the real build_corpus_calibration.py convention), rather than
    the absolute path the shared fixture uses directly."""
    tvi_root = tmp_path / "fake-tvi-root"
    join_dir = tvi_root / "output" / "vectors" / "join"
    join_dir.mkdir(parents=True)
    relative_memmap = "output/vectors/join/vectors_nv-embed-v2.f32"
    (tvi_root / relative_memmap).write_bytes(
        synthetic_scene["x"].astype(np.float32).tobytes()
    )

    header = dict(synthetic_scene["calibration"].header)
    header["source_memmap_path"] = relative_memmap

    seedset_path = tmp_path / "pattern.seedset.json"
    seedset_path.write_text(json.dumps(synthetic_scene["seedset"]))
    calibration_json = tmp_path / "cal.calibration.json"
    calibration_npz = tmp_path / "cal.calibration.npz"
    np.savez(calibration_npz, mu=synthetic_scene["mu"])
    calibration_json.write_text(json.dumps({"header": header, "mu_norm": 0.1}))

    return str(tvi_root), str(seedset_path), str(calibration_json)


def test_initialize_direction_resolves_relative_memmap_path_via_tvi_root(tmp_path, synthetic_scene):
    tvi_root, seedset_path, calibration_json = _write_relative_path_scene(tmp_path, synthetic_scene)
    result = _run(
        initialize_direction(
            manager=None,
            args={
                "seedset_ref": seedset_path,
                "calibration_ref": calibration_json,
                "out_dir": str(tmp_path / "directions"),
                "tvi_root": tvi_root,
                "bootstrap_b": 10,
            },
        )
    )
    assert "error" not in result, result
    assert result["verdict"] in {"usable", "under-determined", "leakage-suspected"}


def test_initialize_direction_refuses_instructively_without_tvi_root(tmp_path, synthetic_scene, monkeypatch):
    """Without tvi_root, a relative source_memmap_path that does not happen
    to resolve against the current working directory refuses with an
    instructive error naming the missing 'tvi_root' argument -- never a bare
    FileNotFoundError from deep inside memmap opening."""
    tvi_root, seedset_path, calibration_json = _write_relative_path_scene(tmp_path, synthetic_scene)
    # Run from a directory where the relative path does NOT resolve (unlike
    # the tvi_root case above).
    monkeypatch.chdir(tmp_path)
    result = _run(
        initialize_direction(
            manager=None,
            args={
                "seedset_ref": seedset_path,
                "calibration_ref": calibration_json,
                "out_dir": str(tmp_path / "directions"),
            },
        )
    )
    assert "error" in result
    assert "tvi_root" in result["error"]
