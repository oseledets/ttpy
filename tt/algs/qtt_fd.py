"""Conservative central finite differences for variable-coefficient QTT PDEs.

This module deliberately contains the ordinary nodal finite-difference
operator, not the multilevel Kazeev--Bachmayr representation from
``tt.algs.qtt_ell``.  On ``N_j = 2**d_j`` interior nodes of the unit cube it
builds the positive-definite matrix of

    -div(k(x) grad u) = f,       u = 0 on the boundary.

The coefficient is evaluated at cell faces.  If ``M_j`` is the backward
difference in direction ``j``, ``k_j^-`` contains the coefficient on every
left face, and ``b_j^+`` is the coefficient on the missing high-boundary face,
then the assembled operator is

    A = sum_j h_j^-2 (M_j.T diag(k_j^-) M_j + diag(b_j^+)).

The formula is a central flux difference, is symmetric by construction, and
never forms a full grid.  Smooth face coefficients are sampled by TT-cross;
the sparse high-boundary mask is inserted exactly rather than being entrusted
to cross.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Number

import numpy as np

from ..core.tools import diag, eye, kron, ones, qdiff, unit
from ..core.vector import vector
from .cross import cross

__all__ = ["QTTDivgradInfo", "qtt_divgrad", "qtt_divgrad_from_faces"]


@dataclass
class QTTDivgradInfo:
    """Diagnostics of coefficient sampling and operator assembly."""

    bits: list[int]
    grid_shape: list[int]
    spacing: list[float]
    coefficient_eps: float
    left_face_ranks: list[list[int]] = field(default_factory=list)
    high_face_ranks: list[list[int]] = field(default_factory=list)
    cross_histories: list[object] = field(default_factory=list)
    operator_ranks: list[int] = field(default_factory=list)


def _bits(d):
    values = np.asarray([d] if isinstance(d, (int, np.integer)) else d,
                        dtype=np.int64).ravel()
    if values.size == 0 or np.any(values < 1):
        raise ValueError(f"d must contain positive QTT level counts, got {d!r}")
    return [int(value) for value in values]


def _spacing(bits, spacing):
    if spacing is None:
        return [1.0 / (2 ** value + 1) for value in bits]
    values = np.asarray(spacing, dtype=float).ravel()
    if values.size == 1:
        values = np.repeat(values, len(bits))
    if values.size != len(bits) or np.any(~np.isfinite(values)) \
            or np.any(values <= 0.0):
        raise ValueError(
            f"spacing must contain {len(bits)} positive finite values"
        )
    return [float(value) for value in values]


def _kron_all(items):
    result = None
    for item in items:
        result = kron(result, item)
    return result


def _indices(indices, bits):
    """Decode LSB-first QTT bits into one integer index per dimension."""
    decoded = []
    offset = 0
    for levels in bits:
        weights = 1 << np.arange(levels, dtype=np.int64)
        decoded.append(indices[:, offset:offset + levels] @ weights)
        offset += levels
    return decoded


def _values(coefficient, points):
    values = np.asarray(coefficient(points))
    if values.ndim == 0:
        values = np.full(points.shape[0], values)
    values = values.reshape(-1)
    if values.size != points.shape[0]:
        raise ValueError(
            "coefficient must return one value per point: got "
            f"{values.size} values for {points.shape[0]} points"
        )
    if np.iscomplexobj(values):
        raise ValueError("coefficient values must be real")
    if not np.all(np.isfinite(values)):
        raise ValueError("coefficient returned a non-finite face value")
    if np.any(values <= 0.0):
        minimum = float(np.min(values))
        raise ValueError(
            "qtt_divgrad needs a positive coefficient for an SPD operator; "
            f"sampled minimum is {minimum:.3e}"
        )
    return values


def _sample_left_face(coefficient, bits, spacing, axis, options):
    ndim = len(bits)

    def evaluate(indices):
        integer = _indices(indices, bits)
        points = np.column_stack([
            (integer[q] + (0.5 if q == axis else 1.0)) * spacing[q]
            for q in range(ndim)
        ])
        return _values(coefficient, points)

    return cross(evaluate, n=2, d=sum(bits), seed=options["seed"] + axis,
                 eps=options["eps"], nswp=options["nswp"],
                 kickrank=options["kickrank"], rmax=options["rmax"],
                 n_check=options["n_check"], verbose=options["verbose"])


def _sample_high_face(coefficient, bits, spacing, axis, options):
    """Coefficient on the high face, embedded in an exact boundary mask."""
    ndim = len(bits)
    remaining = [value for q, value in enumerate(bits) if q != axis]
    high_coordinate = (2 ** bits[axis] + 0.5) * spacing[axis]

    if not remaining:
        value = _values(coefficient, np.array([[high_coordinate]]))[0]
        return float(value) * unit(2, bits[axis], j=2 ** bits[axis] - 1), None

    def evaluate(indices):
        integer = _indices(indices, remaining)
        points = np.empty((indices.shape[0], ndim), dtype=float)
        cursor = 0
        for q in range(ndim):
            if q == axis:
                points[:, q] = high_coordinate
            else:
                points[:, q] = (integer[cursor] + 1.0) * spacing[q]
                cursor += 1
        return _values(coefficient, points)

    trace = cross(
        evaluate,
        n=2,
        d=sum(remaining),
        seed=options["seed"] + len(bits) + axis,
        eps=options["eps"],
        nswp=options["nswp"],
        kickrank=options["kickrank"],
        rmax=options["rmax"],
        n_check=options["n_check"],
        verbose=options["verbose"],
    )
    cores = []
    cursor = 0
    for q, levels in enumerate(bits):
        if q == axis:
            boundary = unit(2, levels, j=2 ** levels - 1)
            cores.extend(boundary.cores)
        else:
            cores.extend(trace.cores[cursor:cursor + levels])
            cursor += levels
    return vector.from_list(cores), trace.history


def qtt_divgrad_from_faces(d, left_faces, high_faces, *, spacing=None,
                           round_eps=1e-13):
    """Assemble ``-div(k grad)`` from QTT face coefficients.

    Args:
        d: QTT levels per physical dimension.
        left_faces: One full-grid TT vector per dimension.  At node ``i`` its
            entry is ``k`` on the face halfway between that node and its left
            neighbour (or the low Dirichlet boundary).
        high_faces: One full-grid TT vector per dimension, zero away from the
            last node in that dimension and equal to ``k`` on the high
            Dirichlet boundary face there.
        spacing: Grid spacing per dimension.  The default is
            ``h_j = 1/(2**d_j + 1)``.
        round_eps: Relative TT rounding tolerance after every directional term
            and sum.  Set to zero to retain the algebraic product ranks.

    Returns:
        A symmetric QTT matrix for the positive operator ``-div(k grad)``.
    """
    bits = _bits(d)
    steps = _spacing(bits, spacing)
    ndim = len(bits)
    if len(left_faces) != ndim or len(high_faces) != ndim:
        raise ValueError(f"expected {ndim} left and high face tensors")
    modes = [2] * sum(bits)
    for name, values in (("left_faces", left_faces),
                         ("high_faces", high_faces)):
        for axis, value in enumerate(values):
            if not isinstance(value, vector) or list(value.n) != modes:
                raise ValueError(
                    f"{name}[{axis}] must be a QTT vector with modes {modes}"
                )
    tolerance = float(round_eps)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("round_eps must be finite and non-negative")

    identities = [eye(2, levels) for levels in bits]
    operator = None
    for axis, levels in enumerate(bits):
        factors = list(identities)
        factors[axis] = qdiff(levels)
        difference = _kron_all(factors)
        term = difference.T @ diag(left_faces[axis]) @ difference
        term = term + diag(high_faces[axis])
        term = term * (steps[axis] ** -2)
        if tolerance > 0.0:
            term = term.round(tolerance)
        operator = term if operator is None else operator + term
        if tolerance > 0.0 and axis > 0:
            operator = operator.round(tolerance)
    return operator


def qtt_divgrad(d, coefficient=1.0, *, spacing=None, coefficient_eps=1e-12,
                round_eps=1e-13, cross_nswp=12, cross_kickrank=2,
                cross_rmax=64, n_check=256, seed=0, verbose=False,
                return_info=False):
    """Build a central-difference QTT matrix for ``-div(k grad)``.

    ``coefficient(points)`` must be vectorized over a ``(batch, ndim)`` array
    of physical coordinates.  It is sampled directly at faces by TT-cross; no
    nodal averaging is used.  The grid consists of ``2**d_j`` interior nodes
    in each direction and has homogeneous Dirichlet boundary conditions.

    A positive scalar coefficient takes an exact path without TT-cross.  For a
    callable, positivity is checked at every point visited by cross, but (as
    with every sampling method) this is not a global positivity certificate.
    Request ``return_info=True`` to receive coefficient ranks and cross
    histories alongside the matrix.
    """
    bits = _bits(d)
    steps = _spacing(bits, spacing)
    tolerance = float(coefficient_eps)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("coefficient_eps must be positive and finite")
    if isinstance(coefficient, Number):
        if np.iscomplexobj(coefficient):
            raise ValueError("coefficient must be real")
        value = float(coefficient)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("coefficient must be positive and finite")
    elif not callable(coefficient):
        raise TypeError("coefficient must be a positive scalar or callable")

    info = QTTDivgradInfo(
        bits=bits,
        grid_shape=[2 ** value for value in bits],
        spacing=steps,
        coefficient_eps=tolerance,
    )
    options = {
        "eps": tolerance,
        "nswp": int(cross_nswp),
        "kickrank": int(cross_kickrank),
        "rmax": None if cross_rmax is None else int(cross_rmax),
        "n_check": int(n_check),
        "seed": int(seed),
        "verbose": bool(verbose),
    }
    if options["nswp"] < 1 or options["kickrank"] < 0 \
            or options["n_check"] < 0:
        raise ValueError(
            "expected cross_nswp >= 1, cross_kickrank >= 0, n_check >= 0"
        )
    if options["rmax"] is not None and options["rmax"] < 1:
        raise ValueError("cross_rmax must be positive or None")
    full_modes = [2] * sum(bits)
    left_faces = []
    high_faces = []
    if isinstance(coefficient, Number):
        whole = value * ones(full_modes)
        for axis, levels in enumerate(bits):
            mask_factors = [ones(2, count) for count in bits]
            mask_factors[axis] = unit(2, levels, j=2 ** levels - 1)
            left_faces.append(whole.copy())
            high_faces.append(value * _kron_all(mask_factors))
    else:
        for axis in range(len(bits)):
            left = _sample_left_face(
                coefficient, bits, steps, axis, options
            )
            high, high_history = _sample_high_face(
                coefficient, bits, steps, axis, options
            )
            left_faces.append(left)
            high_faces.append(high)
            info.cross_histories.append(left.history)
            if high_history is not None:
                info.cross_histories.append(high_history)

    info.left_face_ranks = [list(map(int, value.r)) for value in left_faces]
    info.high_face_ranks = [list(map(int, value.r)) for value in high_faces]
    operator = qtt_divgrad_from_faces(
        bits,
        left_faces,
        high_faces,
        spacing=steps,
        round_eps=round_eps,
    )
    info.operator_ranks = list(map(int, operator.r))
    operator.qtt_divgrad_info = info
    return (operator, info) if return_info else operator
