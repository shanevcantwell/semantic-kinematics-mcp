"""Deterministic geometry for ADR-SKM-009 funnel experiments.

This module is an experiment-layer numerical seam.  It accepts already-computed
vectors, performs no I/O or model calls, and makes no inferential or empirical
claims.  Every computation is NumPy-only and converted to ``float64``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

ArrayLike = np.ndarray | Sequence[float] | Sequence[Sequence[float]]
_UNIT_TOLERANCE = 1e-7


@dataclass(frozen=True)
class PopulationDirectionResult:
    """A population's raw mean displacement and its normalized direction."""

    direction: np.ndarray
    mean_displacement: np.ndarray
    pre_normalization_norm: float
    population_shape: tuple[int, ...]


@dataclass(frozen=True)
class DisplacementGramResult:
    """Complete O(n²) constituent dot-product and cosine matrices."""

    gram: np.ndarray
    cosine_alignment: np.ndarray
    vector_norms: np.ndarray


@dataclass(frozen=True)
class CompositionComparisonResult:
    """Observed composition compared with an additive constituent prediction.

    No field labels a result additive, subadditive, or superadditive.  Those are
    empirical interpretations requiring a preregistered comparison procedure.
    """

    observed_mean: np.ndarray
    additive_prediction: np.ndarray
    interaction_residual: np.ndarray
    observed_target_projection: float
    predicted_target_projection: float
    interaction_target_projection: float
    observed_norm: float
    predicted_norm: float
    interaction_norm: float
    off_axis_interaction_magnitude: float


@dataclass(frozen=True)
class DispersionChangeResult:
    """Population RMS dispersion change in total or projected space."""

    before: float
    after: float
    delta: float
    ratio: float | None
    space: str
    subspace_dimension: int


def _as_float64(name: str, value: ArrayLike, *, min_ndim: int) -> np.ndarray:
    """Convert a real numeric input to float64 and enforce basic validity."""
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real numeric values")

    array = np.asarray(value, dtype=np.float64)
    if array.ndim < min_ndim:
        raise ValueError(f"{name} must have at least {min_ndim} dimensions, got {array.shape}")
    if array.size == 0 or array.shape[-1] == 0:
        raise ValueError(f"{name} must be nonempty, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _population(name: str, value: ArrayLike) -> np.ndarray:
    """Validate an array with population axes followed by an embedding axis."""
    array = _as_float64(name, value, min_ndim=2)
    if int(np.prod(array.shape[:-1])) == 0:
        raise ValueError(f"{name} population must be nonempty, got shape {array.shape}")
    return array


def _finite_result(name: str, value: np.ndarray | np.floating | float) -> np.ndarray:
    """Validate that a derived float64 result stayed representable and finite."""
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} produced a non-finite result outside float64 range")
    return result


def _checked_multiply(
    name: str,
    left: np.ndarray | np.floating | float,
    right: np.ndarray | np.floating | float,
) -> np.ndarray:
    """Multiply while refusing a mathematically nonzero result rounded to zero."""
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            result = left_array * right_array
    except FloatingPointError as error:
        raise _calculation_error(name, error) from error
    result = _finite_result(name, result)
    if np.any((result == 0.0) & (left_array != 0.0) & (right_array != 0.0)):
        raise ValueError(f"{name} produced a nonzero result below float64 range")
    return result


def _checked_divide(
    name: str,
    numerator: np.ndarray | np.floating | float,
    denominator: np.ndarray | np.floating | float,
) -> np.ndarray:
    """Divide while refusing a mathematically nonzero result rounded to zero."""
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            result = numerator_array / denominator_array
    except FloatingPointError as error:
        raise _calculation_error(name, error) from error
    result = _finite_result(name, result)
    if np.any((result == 0.0) & (numerator_array != 0.0)):
        raise ValueError(f"{name} produced a nonzero result below float64 range")
    return result


def _checked_dot(name: str, left: np.ndarray, right: np.ndarray) -> float:
    """Compute one float64 dot product and validate suspicious zero results.

    NumPy remains the deterministic float64 implementation on the normal path.
    Only when it returns zero despite overlapping nonzero operands do we sum the
    exact rational products of the original binary64 values.  This rare refusal
    path distinguishes true cancellation from a nonzero result that NumPy
    rounded away, including terms at different exponent scales.
    """
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            result = np.float64(np.dot(left, right))
    except FloatingPointError as error:
        raise _calculation_error(name, error) from error
    _finite_result(name, result)
    if result != 0.0:
        return float(result)

    overlap = (left != 0.0) & (right != 0.0)
    if not np.any(overlap):
        return 0.0

    exact_result = sum(
        (
            Fraction.from_float(float(left_value)) * Fraction.from_float(float(right_value))
            for left_value, right_value in zip(left[overlap], right[overlap])
        ),
        start=Fraction(0),
    )
    if exact_result == 0:
        return 0.0

    try:
        rounded_result = float(exact_result)
    except OverflowError as error:
        raise ValueError(f"{name} produced a nonzero dot product outside float64 range") from error
    if rounded_result == 0.0 or not np.isfinite(rounded_result):
        raise ValueError(f"{name} produced a nonzero dot product outside float64 range")
    return rounded_result


def _calculation_error(name: str, error: FloatingPointError) -> ValueError:
    return ValueError(f"{name} could not be represented as a finite float64 result")


def _stable_norm(name: str, value: np.ndarray, *, axis: int | None = None) -> np.ndarray:
    """Compute a scaled Euclidean norm without raw squaring overflow/underflow."""
    scale = np.max(np.abs(value), axis=axis, keepdims=True)
    zero = scale == 0.0
    safe_scale = np.where(zero, 1.0, scale)
    # Underflow in a component that is tiny relative to the row scale cannot alter
    # the rounded float64 norm. Overflow/invalid results remain hard failures.
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            scaled = value / safe_scale
            squared_sum = np.sum(scaled * scaled, axis=axis, dtype=np.float64)
            norm_factor = np.sqrt(squared_sum)
    except FloatingPointError as error:
        raise _calculation_error(name, error) from error
    norm = _checked_multiply(name, np.squeeze(scale, axis=axis), norm_factor)
    norm = np.where(np.squeeze(zero, axis=axis), 0.0, norm)
    return _finite_result(name, norm)


def _stable_mean(name: str, value: np.ndarray) -> np.ndarray:
    """Average rows after scaling to prevent an overflowing accumulation."""
    scale = np.max(np.abs(value), axis=0)
    zero = scale == 0.0
    safe_scale = np.where(zero, 1.0, scale)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            normalized_mean = np.mean(value / safe_scale, axis=0, dtype=np.float64)
    except FloatingPointError as error:
        raise _calculation_error(name, error) from error
    mean = _checked_multiply(name, scale, normalized_mean)
    mean = np.where(zero, 0.0, mean)
    return _finite_result(name, mean)


def _unit_axis(target_axis: ArrayLike, dimension: int) -> np.ndarray:
    axis = _as_float64("target_axis", target_axis, min_ndim=1)
    if axis.ndim != 1:
        raise ValueError(f"target_axis must be one-dimensional, got shape {axis.shape}")
    if axis.shape[0] != dimension:
        raise ValueError(
            f"target_axis dimension {axis.shape[0]} does not match vector dimension {dimension}"
        )
    norm = float(_stable_norm("target_axis norm", axis))
    if not np.isclose(norm, 1.0, rtol=_UNIT_TOLERANCE, atol=_UNIT_TOLERANCE):
        raise ValueError(f"target_axis must be unit length, got norm {norm}")
    return axis


def _mean_vector(name: str, value: ArrayLike) -> np.ndarray:
    """Return a vector unchanged or average all leading population dimensions."""
    array = _as_float64(name, value, min_ndim=1)
    if array.ndim == 1:
        return array.copy()
    if int(np.prod(array.shape[:-1])) == 0:
        raise ValueError(f"{name} population must be nonempty, got shape {array.shape}")
    return _stable_mean(f"{name} mean", array.reshape(-1, array.shape[-1]))


def paired_displacements(conditioned_after: ArrayLike, baseline_before: ArrayLike) -> np.ndarray:
    """Return matched ``conditioned_after - baseline_before`` displacements.

    All population/pair axes are preserved.  Pair identity and ordering cannot
    be inferred from arrays, so exact shape equality is the mechanically
    enforceable matching requirement.
    """
    conditioned = _population("conditioned_after", conditioned_after)
    baseline = _population("baseline_before", baseline_before)
    if conditioned.shape != baseline.shape:
        raise ValueError(
            "conditioned_after and baseline_before must have identical matched shapes, "
            f"got {conditioned.shape} and {baseline.shape}"
        )
    try:
        with np.errstate(over="raise", invalid="raise"):
            displacement = conditioned - baseline
    except FloatingPointError as error:
        raise _calculation_error("paired displacement subtraction", error) from error
    return _finite_result("paired displacement subtraction", displacement)


def population_direction(displacements: ArrayLike) -> PopulationDirectionResult:
    """Normalize the raw population mean displacement into a unit direction.

    The raw mean is taken over every population axis (all axes except the last
    embedding dimension).  Its norm is retained before normalization.  Empty
    populations and exactly zero mean displacement are refused.
    """
    population = _population("displacements", displacements)
    mean = _stable_mean(
        "population mean displacement", population.reshape(-1, population.shape[-1])
    )
    norm = float(_stable_norm("population mean displacement norm", mean))
    if norm == 0.0:
        raise ValueError("displacements have a zero mean; population direction is undefined")
    direction = _checked_divide("population direction normalization", mean, norm)
    return PopulationDirectionResult(
        direction=direction,
        mean_displacement=mean,
        pre_normalization_norm=norm,
        population_shape=population.shape[:-1],
    )


def project_population(displacements: ArrayLike, target_axis: ArrayLike) -> np.ndarray:
    """Project a displacement population onto a validated unit target axis.

    The returned signed projections preserve every population axis and remove
    only the final embedding dimension.
    """
    population = _population("displacements", displacements)
    axis = _unit_axis(target_axis, population.shape[-1])
    flattened = population.reshape(-1, population.shape[-1])
    projection = np.array(
        [_checked_dot("population projection", row, axis) for row in flattened],
        dtype=np.float64,
    )
    return projection.reshape(population.shape[:-1])


def displacement_gram(vectors: ArrayLike) -> DisplacementGramResult:
    """Return complete constituent dot-product and cosine alignment matrices.

    Both time and returned matrix storage are explicitly O(n²) in the number of
    vectors.  A zero vector is refused rather than silently omitted or assigned
    a misleading cosine, so every returned alignment entry is defined.
    """
    matrix = _as_float64("vectors", vectors, min_ndim=2)
    if matrix.ndim != 2:
        raise ValueError(f"vectors must have shape [n_vectors, dimension], got {matrix.shape}")
    norms = _stable_norm("vector norms", matrix, axis=1)
    zero_rows = np.flatnonzero(norms == 0.0)
    if zero_rows.size:
        raise ValueError(
            "cosine alignment is undefined for zero vectors at rows "
            + ", ".join(str(int(index)) for index in zero_rows)
        )
    gram = np.empty((matrix.shape[0], matrix.shape[0]), dtype=np.float64)
    for row_index in range(matrix.shape[0]):
        for column_index in range(row_index, matrix.shape[0]):
            dot = _checked_dot("displacement Gram matrix", matrix[row_index], matrix[column_index])
            gram[row_index, column_index] = dot
            gram[column_index, row_index] = dot
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            normalized = matrix / norms[:, None]
            cosine = normalized @ normalized.T
    except FloatingPointError as error:
        raise _calculation_error("cosine alignment", error) from error
    normalized = _finite_result("normalized displacement rows", normalized)
    cosine = _finite_result("cosine alignment", cosine)
    # Raw zero is authoritative after the checked dot has ruled out vanished
    # nonzero products.  This preserves exact cancellation/orthogonality even
    # when independently normalized operands round to a tiny nonzero dot.
    cosine = np.where(gram == 0.0, 0.0, cosine)
    if np.any((cosine == 0.0) & (gram != 0.0)):
        raise ValueError("cosine alignment contains a nonzero result outside float64 range")
    return DisplacementGramResult(
        gram=gram,
        cosine_alignment=np.clip(cosine, -1.0, 1.0),
        vector_norms=norms,
    )


def compare_composition(
    observed_combined: ArrayLike,
    constituent_effects: ArrayLike,
    target_axis: ArrayLike,
) -> CompositionComparisonResult:
    """Compare an observed combined effect with independent additive effects.

    ``observed_combined`` may be one displacement vector or a population whose
    leading dimensions are averaged.  ``constituent_effects`` is the nonempty
    ``[n_constituents, dimension]`` matrix of independently measured mean
    effects.  The interaction is always ``observed - additive_prediction``.
    """
    observed = _mean_vector("observed_combined", observed_combined)
    constituents = _as_float64("constituent_effects", constituent_effects, min_ndim=2)
    if constituents.ndim != 2:
        raise ValueError(
            "constituent_effects must have shape [n_constituents, dimension], "
            f"got {constituents.shape}"
        )
    if constituents.shape[1] != observed.shape[0]:
        raise ValueError(
            f"constituent dimension {constituents.shape[1]} does not match observed "
            f"dimension {observed.shape[0]}"
        )
    axis = _unit_axis(target_axis, observed.shape[0])

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            prediction = constituents.sum(axis=0, dtype=np.float64)
            interaction = observed - prediction
    except FloatingPointError as error:
        raise _calculation_error("composition calculation", error) from error

    prediction = _finite_result("additive prediction", prediction)
    interaction = _finite_result("interaction residual", interaction)
    observed_projection = _checked_dot("observed target projection", observed, axis)
    predicted_projection = _checked_dot("predicted target projection", prediction, axis)
    interaction_projection = _checked_dot("interaction target projection", interaction, axis)
    projected_interaction = _checked_multiply(
        "off-axis interaction projection", interaction_projection, axis
    )
    try:
        with np.errstate(over="raise", invalid="raise"):
            off_axis = interaction - projected_interaction
    except FloatingPointError as error:
        raise _calculation_error("off-axis interaction residual", error) from error
    off_axis = _finite_result("off-axis interaction residual", off_axis)
    interaction_norm = float(_stable_norm("interaction norm", interaction))

    return CompositionComparisonResult(
        observed_mean=observed,
        additive_prediction=prediction,
        interaction_residual=interaction,
        observed_target_projection=observed_projection,
        predicted_target_projection=predicted_projection,
        interaction_target_projection=interaction_projection,
        observed_norm=float(_stable_norm("observed norm", observed)),
        predicted_norm=float(_stable_norm("predicted norm", prediction)),
        interaction_norm=interaction_norm,
        off_axis_interaction_magnitude=float(_stable_norm("off-axis interaction norm", off_axis)),
    )


def _projection_basis(subspace: ArrayLike, dimension: int) -> np.ndarray:
    """Validate one unit axis or an orthonormal row-basis for a subspace."""
    basis = _as_float64("subspace", subspace, min_ndim=1)
    if basis.ndim == 1:
        return _unit_axis(basis, dimension).reshape(1, -1)
    if basis.ndim != 2:
        raise ValueError(
            f"subspace must be a unit axis or a 2-D orthonormal row basis, got {basis.shape}"
        )
    if basis.shape[1] != dimension:
        raise ValueError(
            f"subspace dimension {basis.shape[1]} does not match vector dimension {dimension}"
        )
    gram = np.empty((basis.shape[0], basis.shape[0]), dtype=np.float64)
    for row_index in range(basis.shape[0]):
        for column_index in range(row_index, basis.shape[0]):
            dot = _checked_dot("subspace Gram matrix", basis[row_index], basis[column_index])
            gram[row_index, column_index] = dot
            gram[column_index, row_index] = dot
    identity = np.eye(basis.shape[0], dtype=np.float64)
    if not np.allclose(gram, identity, rtol=_UNIT_TOLERANCE, atol=_UNIT_TOLERANCE):
        raise ValueError("subspace rows must form an orthonormal basis")
    return basis


def _rms_dispersion(population: np.ndarray, basis: np.ndarray | None) -> float:
    flattened = population.reshape(-1, population.shape[-1])
    mean = _stable_mean("dispersion population mean", flattened)
    try:
        with np.errstate(over="raise", invalid="raise"):
            centered = flattened - mean
    except FloatingPointError as error:
        raise _calculation_error("dispersion centering", error) from error
    centered = _finite_result("dispersion centering", centered)
    if basis is not None:
        centered = np.array(
            [
                [
                    _checked_dot("dispersion subspace projection", row, basis_row)
                    for basis_row in basis
                ]
                for row in centered
            ],
            dtype=np.float64,
        )

    scale = float(np.max(np.abs(centered)))
    if scale == 0.0:
        return 0.0
    # Scaling before squaring makes both near-maximum and subnormal populations
    # safe. Mean over population members is the population denominator N.
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            scaled = centered / scale
            squared_distance = np.sum(scaled * scaled, axis=1, dtype=np.float64)
            mean_squared_distance = np.mean(squared_distance, dtype=np.float64)
            dispersion_factor = np.sqrt(mean_squared_distance)
    except FloatingPointError as error:
        raise _calculation_error("RMS dispersion", error) from error
    dispersion = _checked_multiply("RMS dispersion", scale, dispersion_factor)
    return float(dispersion)


def dispersion_change(
    before: ArrayLike,
    after: ArrayLike,
    subspace: ArrayLike | None = None,
) -> DispersionChangeResult:
    """Measure population RMS dispersion before and after an intervention.

    Without ``subspace`` this is explicitly total-space dispersion.  Supplying
    either a unit axis or an orthonormal row-basis computes behaviorally
    projected dispersion.  Populations must have identical matched shapes.
    A zero baseline yields ``ratio=None`` rather than division noise.
    """
    before_population = _population("before", before)
    after_population = _population("after", after)
    if before_population.shape != after_population.shape:
        raise ValueError(
            f"before and after must have identical matched shapes, got "
            f"{before_population.shape} and {after_population.shape}"
        )

    basis = None if subspace is None else _projection_basis(subspace, before_population.shape[-1])
    before_dispersion = _rms_dispersion(before_population, basis)
    after_dispersion = _rms_dispersion(after_population, basis)
    if before_dispersion == 0.0:
        ratio = None
    else:
        ratio = float(_checked_divide("dispersion ratio", after_dispersion, before_dispersion))
    try:
        with np.errstate(over="raise", invalid="raise"):
            delta = np.float64(after_dispersion) - np.float64(before_dispersion)
    except FloatingPointError as error:
        raise _calculation_error("dispersion delta", error) from error
    delta = float(_finite_result("dispersion delta", delta))
    return DispersionChangeResult(
        before=before_dispersion,
        after=after_dispersion,
        delta=delta,
        ratio=ratio,
        space="total" if basis is None else "projected",
        subspace_dimension=before_population.shape[-1] if basis is None else basis.shape[0],
    )


def turnwise_projection(turn_displacements: ArrayLike, target_axis: ArrayLike) -> np.ndarray:
    """Project turn-indexed displacements while preserving all leading axes."""
    return project_population(turn_displacements, target_axis)
