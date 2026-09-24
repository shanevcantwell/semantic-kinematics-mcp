"""Synthetic contract tests for ADR-SKM-009's experiment-layer geometry.

These tests validate numerical mechanics only.  They are not runs or results of
preregistered experiments SKM-FUNNEL-EXP0A/0B.
"""

import numpy as np
import pytest

from semantic_kinematics import experiments
from semantic_kinematics.experiments import (
    compare_composition,
    dispersion_change,
    displacement_gram,
    paired_displacements,
    population_direction,
    project_population,
    turnwise_projection,
)

EXPECTED_EXPORTS = {
    "paired_displacements",
    "population_direction",
    "project_population",
    "displacement_gram",
    "compare_composition",
    "dispersion_change",
    "turnwise_projection",
}


def test_package_exports_exact_initial_seam():
    assert set(experiments.__all__) == EXPECTED_EXPORTS
    assert all(callable(getattr(experiments, name)) for name in EXPECTED_EXPORTS)


def test_paired_displacements_sign_shape_float64_and_centering_cancellation():
    baseline = np.array([[[1.0, 2.0], [3.0, 5.0]], [[-2.0, 4.0], [8.0, -1.0]]], dtype=np.float32)
    conditioned = baseline + np.array([2.0, -3.0], dtype=np.float32)

    displacement = paired_displacements(conditioned, baseline)
    assert displacement.shape == baseline.shape
    assert displacement.dtype == np.float64
    np.testing.assert_array_equal(displacement, np.broadcast_to([2.0, -3.0], baseline.shape))

    shared_center = np.array([100.0, -40.0])
    centered = paired_displacements(conditioned - shared_center, baseline - shared_center)
    np.testing.assert_allclose(centered, displacement)


def test_paired_displacements_refuses_unmatched_shapes():
    with pytest.raises(ValueError, match="identical matched shapes"):
        paired_displacements(np.zeros((2, 3)), np.zeros((3, 3)))


def test_population_direction_uses_raw_population_mean_and_reports_norm():
    displacements = np.array([[[3.0, 0.0], [0.0, 4.0]], [[3.0, 4.0], [2.0, 0.0]]])
    # Raw mean = [2, 2], not a mean of row-normalized vectors.
    result = population_direction(displacements)

    np.testing.assert_allclose(result.mean_displacement, [2.0, 2.0])
    assert result.pre_normalization_norm == pytest.approx(np.sqrt(8.0))
    np.testing.assert_allclose(result.direction, [1.0 / np.sqrt(2.0)] * 2)
    assert result.population_shape == (2, 2)
    assert result.direction.dtype == np.float64


def test_population_direction_refuses_zero_mean():
    with pytest.raises(ValueError, match="zero mean"):
        population_direction([[1.0, 0.0], [-1.0, 0.0]])


def test_project_population_signed_values_and_shape_preservation():
    population = np.array([[[1.0, 4.0], [-2.0, 5.0]], [[3.0, 6.0], [0.0, 7.0]]])
    projected = project_population(population, [1.0, 0.0])

    assert projected.shape == (2, 2)
    assert projected.dtype == np.float64
    np.testing.assert_array_equal(projected, [[1.0, -2.0], [3.0, 0.0]])


@pytest.mark.parametrize("axis", ([2.0, 0.0], [0.0, 0.0]))
def test_project_population_refuses_nonunit_axis(axis):
    with pytest.raises(ValueError, match="unit length"):
        project_population([[1.0, 2.0]], axis)


def test_project_population_refuses_dimension_mismatched_axis():
    with pytest.raises(ValueError, match="does not match"):
        project_population([[1.0, 2.0]], [1.0, 0.0, 0.0])


def test_project_population_refuses_nonzero_projection_that_underflows_to_zero():
    minimum_subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
    unit_axis = [0.5, np.sqrt(0.75)]

    with pytest.raises(ValueError, match=r"nonzero dot product outside float64 range"):
        project_population([[minimum_subnormal, 0.0]], unit_axis)


def test_project_population_refuses_mixed_scale_underflow_hidden_by_unit_scales():
    with pytest.raises(ValueError, match=r"nonzero dot product outside float64 range"):
        project_population([[1.0, 1e-300, 0.0]], [0.0, 1e-300, 1.0])


def test_project_population_preserves_exact_cancellation_with_overlapping_nonzero_terms():
    projection = project_population([[1e-300, 1e-300, 0.0]], [1e-300, -1e-300, 1.0])

    np.testing.assert_array_equal(projection, [0.0])


def test_displacement_gram_retains_aligned_orthogonal_and_opposed_entries():
    vectors = np.array([[2.0, 0.0], [4.0, 0.0], [0.0, 3.0], [-1.0, 0.0]])
    result = displacement_gram(vectors)

    expected_gram = vectors @ vectors.T
    expected_cosine = np.array(
        [
            [1.0, 1.0, 0.0, -1.0],
            [1.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(result.gram, expected_gram)
    np.testing.assert_allclose(result.cosine_alignment, expected_cosine)
    np.testing.assert_allclose(result.vector_norms, [2.0, 4.0, 3.0, 1.0])
    assert result.gram.shape == (4, 4)


def test_displacement_gram_refuses_zero_rows_instead_of_hiding_them():
    with pytest.raises(ValueError, match=r"zero vectors at rows 1"):
        displacement_gram([[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]])


def test_displacement_gram_accepts_non_axis_aligned_exact_orthogonality():
    vectors = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, -1.0]])
    result = displacement_gram(vectors)

    assert result.gram[0, 1] == 0.0
    assert result.gram[1, 0] == 0.0
    assert result.cosine_alignment[0, 1] == pytest.approx(0.0, abs=1e-15)
    assert result.cosine_alignment[1, 0] == pytest.approx(0.0, abs=1e-15)


def test_displacement_gram_handles_small_vectors_when_raw_gram_is_representable():
    small = 1e-160
    result = displacement_gram([[small, 0.0], [0.0, -small]])

    assert np.all(np.isfinite(result.gram))
    assert result.gram[0, 0] > 0.0
    assert result.gram[1, 1] > 0.0
    np.testing.assert_allclose(result.cosine_alignment, np.eye(2), atol=0.0)
    np.testing.assert_allclose(result.vector_norms, [small, small], rtol=1e-15)


def test_displacement_gram_refuses_unrepresentable_nonzero_raw_dot_products():
    with pytest.raises(ValueError, match=r"nonzero dot product outside float64 range"):
        displacement_gram([[1e-300, 0.0], [0.0, 1e-300]])


def test_displacement_gram_refuses_mixed_scale_off_diagonal_underflow():
    with pytest.raises(ValueError, match=r"nonzero dot product outside float64 range"):
        displacement_gram([[1.0, 1e-300, 0.0], [0.0, 1e-300, 1.0]])


def test_displacement_gram_recovers_representable_exact_cancellation_residual():
    above_one = np.nextafter(np.float64(1.0), np.float64(2.0))
    below_one = np.nextafter(np.float64(1.0), np.float64(0.0))
    first = np.array([above_one, 1.0])
    second = np.array([below_one, -1.0])
    assert np.dot(first, second) == 0.0

    result = displacement_gram([first, second])

    below_half_ulp = np.nextafter(np.float64(2.0**-53), np.float64(0.0))
    expected = np.nextafter(below_half_ulp, np.float64(0.0))
    assert result.gram[0, 1] == expected
    assert result.gram[1, 0] == expected


def test_compare_composition_reports_full_residual_and_axis_split():
    observed_population = np.array([[4.0, 3.0], [2.0, 1.0]])  # mean [3, 2]
    constituents = np.array([[1.0, 0.0], [1.0, 1.0]])  # additive [2, 1]
    result = compare_composition(observed_population, constituents, [1.0, 0.0])

    np.testing.assert_allclose(result.observed_mean, [3.0, 2.0])
    np.testing.assert_allclose(result.additive_prediction, [2.0, 1.0])
    np.testing.assert_allclose(result.interaction_residual, [1.0, 1.0])
    assert result.observed_target_projection == pytest.approx(3.0)
    assert result.predicted_target_projection == pytest.approx(2.0)
    assert result.interaction_target_projection == pytest.approx(1.0)
    assert result.observed_norm == pytest.approx(np.sqrt(13.0))
    assert result.predicted_norm == pytest.approx(np.sqrt(5.0))
    assert result.interaction_norm == pytest.approx(np.sqrt(2.0))
    assert result.off_axis_interaction_magnitude == pytest.approx(1.0)


def test_compare_composition_additive_example_reports_zero_interaction_only():
    result = compare_composition([3.0, 1.0], [[1.0, 0.0], [2.0, 1.0]], [1.0, 0.0])

    np.testing.assert_array_equal(result.interaction_residual, [0.0, 0.0])
    assert result.interaction_target_projection == pytest.approx(0.0)
    assert result.interaction_norm == pytest.approx(0.0)
    assert result.off_axis_interaction_magnitude == pytest.approx(0.0)
    assert not hasattr(result, "significance")
    assert not hasattr(result, "superadditive")


def test_compare_composition_smaller_observed_norm_reports_signed_interaction_not_a_claim():
    result = compare_composition([1.0, 0.0], [[1.0, 0.0], [1.0, 0.0]], [1.0, 0.0])

    assert result.observed_norm < result.predicted_norm
    assert result.interaction_target_projection == pytest.approx(-1.0)
    np.testing.assert_array_equal(result.interaction_residual, [-1.0, 0.0])
    assert not hasattr(result, "regime")


def test_compare_composition_refuses_constituent_dimension_mismatch():
    with pytest.raises(ValueError, match="does not match observed"):
        compare_composition([1.0, 0.0], [[1.0, 0.0, 0.0]], [1.0, 0.0])


def test_dispersion_change_uses_population_rms_and_is_translation_invariant():
    before = np.array([[-1.0, 0.0], [1.0, 0.0]])
    after = np.array([[-2.0, 0.0], [2.0, 0.0]])
    result = dispersion_change(before, after)
    translated = dispersion_change(before + [100.0, -50.0], after + [-20.0, 80.0])

    assert result.before == pytest.approx(1.0)
    assert result.after == pytest.approx(2.0)
    assert result.delta == pytest.approx(1.0)
    assert result.ratio == pytest.approx(2.0)
    assert result.space == "total"
    assert result.subspace_dimension == 2
    assert translated == result


def test_dispersion_change_projects_onto_validated_axis_or_subspace():
    before = np.array([[-3.0, -4.0], [3.0, 4.0]])
    after = before / 2.0

    along_x = dispersion_change(before, after, subspace=[1.0, 0.0])
    full_subspace = dispersion_change(before, after, subspace=np.eye(2))

    assert along_x.before == pytest.approx(3.0)
    assert along_x.after == pytest.approx(1.5)
    assert along_x.ratio == pytest.approx(0.5)
    assert along_x.space == "projected"
    assert along_x.subspace_dimension == 1
    assert full_subspace.before == pytest.approx(5.0)
    assert full_subspace.subspace_dimension == 2


def test_dispersion_change_zero_baseline_has_explicit_undefined_ratio():
    before = np.array([[4.0, -2.0], [4.0, -2.0]])
    after = np.array([[3.0, -2.0], [5.0, -2.0]])
    result = dispersion_change(before, after)

    assert result.before == 0.0
    assert result.after == pytest.approx(1.0)
    assert result.ratio is None


def test_small_nonzero_dispersion_retains_a_defined_ratio_without_norm_underflow():
    before = np.array([[-1e-300, 0.0], [1e-300, 0.0]])
    after = np.array([[-2e-300, 0.0], [2e-300, 0.0]])
    result = dispersion_change(before, after)

    assert result.before > 0.0
    assert result.after > 0.0
    assert result.delta > 0.0
    assert result.ratio == pytest.approx(2.0)


def test_dispersion_change_refuses_shape_mismatch_and_nonorthonormal_subspace():
    with pytest.raises(ValueError, match="identical matched shapes"):
        dispersion_change(np.zeros((2, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="orthonormal"):
        dispersion_change(np.zeros((2, 2)), np.ones((2, 2)), [[1.0, 0.0], [1.0, 0.0]])


def test_turnwise_projection_preserves_pair_and_turn_dimensions():
    turn_displacements = np.array(
        [
            [[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]],
            [[[5.0, 50.0], [6.0, 60.0]], [[7.0, 70.0], [8.0, 80.0]]],
        ],
        dtype=np.float32,
    )
    projected = turnwise_projection(turn_displacements, [0.0, 1.0])

    assert projected.shape == (2, 2, 2)
    assert projected.dtype == np.float64
    np.testing.assert_array_equal(projected, turn_displacements[..., 1])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_all_operations_refuse_nonfinite_numeric_inputs(bad_value):
    bad_population = np.array([[1.0, 0.0], [bad_value, 1.0]])
    good_population = np.array([[1.0, 0.0], [2.0, 1.0]])

    operations = [
        lambda: paired_displacements(bad_population, good_population),
        lambda: population_direction(bad_population),
        lambda: project_population(bad_population, [1.0, 0.0]),
        lambda: displacement_gram(bad_population),
        lambda: compare_composition(bad_population, [[1.0, 0.0]], [1.0, 0.0]),
        lambda: dispersion_change(bad_population, good_population),
        lambda: turnwise_projection(bad_population, [1.0, 0.0]),
    ]
    for operation in operations:
        with pytest.raises(ValueError, match="finite"):
            operation()


@pytest.mark.parametrize(
    "operation",
    [
        lambda: paired_displacements(np.empty((0, 2)), np.empty((0, 2))),
        lambda: population_direction(np.empty((0, 2))),
        lambda: project_population(np.empty((0, 2)), [1.0, 0.0]),
        lambda: displacement_gram(np.empty((0, 2))),
        lambda: compare_composition([1.0, 0.0], np.empty((0, 2)), [1.0, 0.0]),
        lambda: dispersion_change(np.empty((0, 2)), np.empty((0, 2))),
        lambda: turnwise_projection(np.empty((0, 3, 2)), [1.0, 0.0]),
    ],
)
def test_all_operations_refuse_empty_populations(operation):
    with pytest.raises(ValueError, match="nonempty"):
        operation()


def test_near_maximum_inputs_are_stable_when_representable_and_refused_when_not():
    maximum = np.finfo(np.float64).max

    direction = population_direction([[maximum, 0.0], [maximum, 0.0]])
    assert direction.pre_normalization_norm == maximum
    np.testing.assert_array_equal(direction.direction, [1.0, 0.0])

    overflowing_operations = [
        lambda: paired_displacements([[maximum, 0.0]], [[-maximum, 0.0]]),
        lambda: project_population([[maximum, maximum]], [np.sqrt(0.5), np.sqrt(0.5)]),
        lambda: displacement_gram([[maximum, 0.0], [0.0, maximum]]),
        lambda: compare_composition([maximum, 0.0], [[maximum, 0.0], [maximum, 0.0]], [1.0, 0.0]),
        lambda: dispersion_change(
            [[-1e-300, 0.0], [1e-300, 0.0]],
            [[-maximum, 0.0], [maximum, 0.0]],
        ),
    ]
    for operation in overflowing_operations:
        with pytest.raises(ValueError, match=r"finite float64|non-finite.*float64"):
            operation()


def test_very_small_population_direction_does_not_become_falsely_zero():
    result = population_direction([[1e-300, 0.0], [1e-300, 0.0]])

    assert result.pre_normalization_norm == pytest.approx(1e-300, rel=1e-15, abs=0.0)
    np.testing.assert_array_equal(result.direction, [1.0, 0.0])


def test_population_direction_refuses_nonzero_mean_below_float64_range():
    minimum_subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))

    with pytest.raises(ValueError, match=r"nonzero result below float64 range"):
        population_direction([[minimum_subnormal, 0.0], [0.0, 0.0]])


def test_rms_dispersion_refuses_nonzero_result_below_float64_range():
    minimum_subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
    population = np.zeros((10, 1), dtype=np.float64)
    population[0, 0] = minimum_subnormal
    population[1, 0] = -minimum_subnormal

    with pytest.raises(ValueError, match=r"nonzero result below float64 range"):
        dispersion_change(population, population)


def test_minimum_subnormal_raw_gram_is_refused_as_unrepresentable():
    minimum_subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))

    with pytest.raises(ValueError, match=r"nonzero dot product outside float64 range"):
        displacement_gram([[minimum_subnormal]])


def test_tiny_but_representable_projection_mean_gram_and_rms_continue_to_work():
    tiny = np.float64(1e-160)

    projection = project_population([[tiny, 0.0]], [1.0, 0.0])
    direction = population_direction([[tiny, 0.0], [tiny, 0.0]])
    gram = displacement_gram([[tiny, 0.0]])
    dispersion = dispersion_change([[-tiny], [tiny]], [[-tiny], [tiny]])

    assert projection[0] == tiny
    assert direction.mean_displacement[0] == tiny
    assert gram.gram[0, 0] > 0.0
    assert dispersion.before == pytest.approx(tiny, rel=1e-15, abs=0.0)
    assert dispersion.after == pytest.approx(tiny, rel=1e-15, abs=0.0)


def test_float32_inputs_produce_float64_structured_arrays():
    f32 = np.array([[1.0, 0.0], [1.0, 2.0]], dtype=np.float32)

    direction = population_direction(f32)
    gram = displacement_gram(f32)
    composition = compare_composition(f32, f32, np.array([1.0, 0.0], dtype=np.float32))

    assert direction.direction.dtype == np.float64
    assert direction.mean_displacement.dtype == np.float64
    assert gram.gram.dtype == np.float64
    assert gram.cosine_alignment.dtype == np.float64
    assert gram.vector_norms.dtype == np.float64
    assert composition.observed_mean.dtype == np.float64
    assert composition.additive_prediction.dtype == np.float64
    assert composition.interaction_residual.dtype == np.float64
