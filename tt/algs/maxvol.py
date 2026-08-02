"""Maximum-volume submatrix selection: square (1-volume) and rectangular (2-volume).

Given a tall matrix ``A`` of shape ``(N, r)`` with ``N >= r`` and full column rank,
pick a small set of rows ``piv`` such that ``A = C A[piv]`` with a *small*
coefficient matrix ``C``.  Everything that does cross approximation (``tt.cross``,
``rect_cross``, ``multifuncrs``) needs exactly this: the rows are the interpolation
indices and ``C`` is the interpolation operator.

* :func:`maxvol` -- square version, ``K = r`` rows, greedy maximisation of
  ``|det A[piv]|``.  On exit ``max |C| <= tol`` (Chebyshev norm), i.e. the selected
  submatrix is *dominant*: no single row swap grows the determinant by more than
  ``tol``.  Goreinov, Tyrtyshnikov (2001); Goreinov, Oseledets, Savostyanov,
  Tyrtyshnikov, Zamarashkin, "How to find a good submatrix", 2010.
* :func:`rect_maxvol` -- rectangular version, ``K >= r`` rows, greedy maximisation
  of the 2-volume ``sqrt(det(A[piv]^* A[piv]))``.  On exit every row of ``C`` has
  2-norm ``<= tol``.  Mikhalev, Oseledets, "Rectangular maximum-volume submatrices
  and their applications", Linear Algebra Appl. 538 (2018) 187-211,
  arXiv:1502.07838.

Implementation notes
--------------------
Both routines are **numpy/scipy** code, not backend-generic: they need LAPACK
``getrf`` pivots for the starting guess and in-place BLAS ``ger``/``geru`` rank-1
updates to avoid ``O(N K)`` temporaries per swap.  Non-numpy input (a torch
tensor) is moved to host memory, and the returned coefficient matrix is moved
back to the backend/device of the input; the pivots are always a numpy int array,
because they are index bookkeeping and are consumed by numpy code everywhere in
this package.

Complexity, ``N`` rows, ``r`` columns, ``K`` selected rows:

* :func:`maxvol`: ``O(N r^2)`` for the LU start and the triangular solve, then
  ``O(N r)`` per row swap (Sherman-Morrison rank-1 update of ``C``, never a fresh
  inverse), so ``O(N r^2 + iters * N r)`` in total, ``O(N r)`` memory.
* :func:`rect_maxvol`: the square start plus ``O(N K)`` per added row
  (Sherman-Woodbury-Morrison rank-1 update of the pseudo-inverse), so
  ``O(N r^2 + N K^2)`` in total, ``O(N K)`` memory.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import get_blas_funcs, get_lapack_funcs

from .. import backend as bk

__all__ = [
    "maxvol", "rect_maxvol",
    "maxvol_qr", "rect_maxvol_qr",
    "maxvol_svd", "rect_maxvol_svd",
]


# --------------------------------------------------------------------------- #
# small private helpers
# --------------------------------------------------------------------------- #

def _host_matrix(a):
    """Move ``a`` to a Fortran-ordered numpy 2d array; return it and a restorer.

    The restorer maps a numpy result back onto the backend/device of ``a`` so a
    torch caller gets a torch matrix back.  Integer input is promoted to float64
    -- maxvol is a floating point algorithm, silently doing integer arithmetic
    would be a wrong answer, not a fast one.  Nested sequences are accepted, as
    the legacy ``asanyarray`` wrapper did.

    A non-finite entry is rejected here and not further down: every comparison
    the algorithm makes (``|C[i, j]| > tol``, ``max |C[piv] - I| > 1e-3``,
    ``||C[i]||^2 > tol^2``) is *false* for a NaN, so a single NaN would turn the
    whole chain of loud checks into a chain of silent passes and the caller would
    get plausible-looking pivots with an all-NaN coefficient matrix.
    """
    if isinstance(a, (list, tuple)):
        a = np.asarray(a)
    if isinstance(a, np.ndarray):
        arr, restore = a, (lambda x: x)
    else:
        backend = bk.backend_of(a)  # raises for anything that is not a known array
        arr = bk.to_numpy(a)
        restore = lambda x: bk.asarray(x, backend=backend)  # noqa: E731
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"maxvol expects a 2d matrix, got shape {arr.shape}")
    if arr.dtype.kind not in "fc":
        arr = arr.astype(np.float64)
    bk.canon_dtype(arr.dtype)  # loud on float16 & friends
    if arr.size and not np.isfinite(arr).all():
        bad = np.argwhere(~np.isfinite(arr))
        raise ValueError(
            f"maxvol: the input matrix has {len(bad)} non-finite entries, the "
            f"first at index {tuple(bad[0])} (value {arr[tuple(bad[0])]}). "
            "Every stopping test in maxvol is a '>' comparison and would be "
            "silently false for a NaN, so this is refused rather than propagated.")
    return np.asfortranarray(arr), restore


def _ger(c, alpha, x, y):
    """In place ``c += alpha * outer(x, y)`` through BLAS, no ``(N, K)`` temporary.

    ``geru`` (not ``gerc``) for complex: the second vector is used as is, the
    caller conjugates when the formula asks for it.
    """
    ger = get_blas_funcs("geru" if np.iscomplexobj(c) else "ger", (c,))
    out = ger(alpha, x, y, a=c, overwrite_a=1)
    if not np.may_share_memory(out, c):
        # f2py silently copies when layout/dtype do not match, which would drop
        # the update on the floor.  Refuse rather than return a stale matrix.
        raise RuntimeError(
            "BLAS ger refused to update the coefficient matrix in place "
            f"(dtype={c.dtype}, f_contiguous={c.flags.f_contiguous}); this is a bug")
    return c


def _square_tol(tol):
    """Canonical Chebyshev bound for square maxvol.

    ``max |C| >= 1`` always (``C[piv] = I``), so a value below 1 can never be a
    bound: it is the legacy Fortran spelling ``tt.maxvol.maxvol(a, nswp, tol=5e-2)``
    where the threshold is ``1 + tol``.  The mapping is exact, not a guess.
    """
    tol = float(tol)
    if not tol > 0.0:
        raise ValueError(f"maxvol tol must be positive, got {tol}")
    return 1.0 + tol if tol < 1.0 else tol


def _check_rank(lu, r, rcond, dtype):
    """Raise when the LU of the input says the columns are (numerically) dependent."""
    diag = np.abs(np.diag(lu[:r, :r]))
    top = float(diag.max()) if diag.size else 0.0
    low = float(diag.min()) if diag.size else 0.0
    if rcond is None:
        rcond = max(r, 1) * bk.eps_of(bk.canon_dtype(dtype))
    if top == 0.0 or low <= rcond * top:
        raise np.linalg.LinAlgError(
            "maxvol: the matrix is rank deficient / numerically singular "
            f"(smallest |U_ii| = {low:.3e}, largest = {top:.3e}, "
            f"threshold = {rcond:.3e} * largest). Orthogonalise the columns "
            "(maxvol_qr) or drop the dependent ones before calling maxvol.")


def _check_identity(c, piv, r, dtype, stage, hard=1e-3):
    """Post-condition ``C[piv] == I``; return the measured deviation.

    It is the cheapest independent witness that the whole thing is meaningful:
    it fails if the pivots are wrong, and it grows like ``eps * cond(A[piv])``, so
    an ill-conditioned input shows up as a number instead of a plausible-looking
    garbage matrix.  A deviation of ``hard`` or more means ``C`` no longer
    interpolates and is an error; anything above ``sqrt(eps)`` is a warning.
    """
    err = float(np.abs(c[piv] - np.eye(r, dtype=dtype)).max())
    if not np.isfinite(err):
        raise np.linalg.LinAlgError(
            f"maxvol: after {stage} C[piv] contains non-finite entries "
            f"(max|C[piv] - I| = {err}); the factorisation is meaningless. "
            "Note that `err > hard` is False for a NaN, which is why this is a "
            "separate test and not a looser threshold.")
    if err > hard:
        raise np.linalg.LinAlgError(
            f"maxvol: after {stage} C[piv] deviates from the identity by {err:.3e}; "
            "the input is too ill-conditioned for the selected rows to mean anything "
            "(orthogonalise the columns first, e.g. with maxvol_qr)")
    if err > np.sqrt(bk.eps_of(bk.canon_dtype(dtype))):
        warnings.warn(
            f"maxvol: after {stage} max|C[piv] - I| = {err:.3e}; the input is badly "
            "conditioned and the selected rows are only that accurate.",
            RuntimeWarning, stacklevel=3)
    return err


def _clamp_top_k(top_k_index, n, r):
    if top_k_index == -1 or top_k_index > n:
        top_k_index = n
    if top_k_index < r:
        top_k_index = r
    return int(top_k_index)


# --------------------------------------------------------------------------- #
# square maxvol
# --------------------------------------------------------------------------- #

def maxvol(a, tol=1.05, max_iters=100, nswp=None, top_k_index=-1,
           rcond=None, info=None, warn_nonconvergence=True):
    """Square maximum-volume submatrix: ``r`` rows of locally maximal ``|det|``.

    Starts from the row set produced by LU with partial pivoting, then repeatedly
    swaps the row that gives the largest gain: with ``C = A A[piv]^{-1}``, replacing
    ``piv[j]`` by row ``i`` multiplies ``|det A[piv]|`` by exactly ``|C[i, j]|``.
    ``C`` is updated by the Sherman-Morrison formula

    ``C <- C - outer(C[:, j], C[i, :] - e_j) / C[i, j]``

    which costs ``O(N r)``; the inverse is never recomputed.  The iteration stops
    when ``max |C| <= tol``, i.e. no single swap can gain more than a factor
    ``tol``; the determinant strictly grows at every accepted swap, so the loop
    cannot cycle.

    Args:
        a: ``(N, r)`` real or complex matrix, ``N >= r``, full column rank.
            numpy array or torch tensor.
        tol: Upper bound for the Chebyshev norm of ``C``; must be ``>= 1``.
            A value ``< 1`` is read as the legacy Fortran spelling and means
            ``1 + tol`` (so the legacy default ``tol=5e-2`` is ``1.05``).
        max_iters: Maximum number of row swaps.
        nswp: Legacy alias for ``max_iters`` (``tt.maxvol.maxvol(a, nswp, tol)``).
            When given it overrides ``max_iters``.
        top_k_index: Only rows ``0 .. top_k_index - 1`` may be selected;
            ``-1`` means all ``N`` rows.
        rcond: Relative threshold on the LU diagonal used to declare the input
            rank deficient. Default ``r * eps(dtype)``.
        info: Optional dict, filled with ``iters``, ``converged``, ``max_abs_C``,
            ``tol`` and ``identity_error`` (drift of ``C[piv]`` from the identity)
            so a caller can explain what happened without re-deriving it.
        warn_nonconvergence: Emit the ``RuntimeWarning`` below.  Set to ``False``
            only when a truncated run is the point (:func:`rect_maxvol` uses a few
            swaps as a warm start); ``info['converged']`` still reports the truth.

    Returns:
        tuple: ``(piv, C)`` -- **both**, unlike the legacy Fortran wrapper
        ``tt.maxvol.maxvol`` which returned only the pivots.

        * ``piv``: ``(r,)`` numpy int array, the selected row numbers of ``a``.
        * ``C``: ``(N, r)`` matrix with ``a == C @ a[piv]`` up to roundoff,
          ``C[piv] == I`` and ``max |C| <= tol``.  Same backend/device as ``a``.

    Raises:
        numpy.linalg.LinAlgError: ``a`` is rank deficient, or the computed ``C``
            does not reproduce the identity on the selected rows (which means the
            input is too ill-conditioned for the result to mean anything).
        ValueError: ``a`` is not a 2d matrix, or ``tol <= 0``.

    Warns:
        RuntimeWarning: the swap loop hit ``max_iters`` with ``max |C| > tol``
            (the result is still a valid factorisation ``a = C a[piv]``, it is
            merely not dominant; ``info['converged']`` is ``False``), or
            ``max |C[piv] - I| > sqrt(eps)``, i.e. the input is so ill-conditioned
            that ``C`` only interpolates to that accuracy
            (``info['identity_error']``).

    Note:
        ``N <= r`` is not an error: all rows are returned with ``C = I``, matching
        the legacy behaviour.  No rank test is performed in that case.

    Reference:
        S. A. Goreinov, I. V. Oseledets, D. V. Savostyanov, E. E. Tyrtyshnikov,
        N. L. Zamarashkin, "How to find a good submatrix", in Matrix Methods:
        Theory, Algorithms, Applications, 2010, pp. 247-256.
    """
    if nswp is not None:
        max_iters = int(nswp)
    max_iters = int(max_iters)
    tol = _square_tol(tol)

    A, restore = _host_matrix(a)
    n, r = A.shape
    if n <= r:
        piv = np.arange(n, dtype=np.int64)
        eye = np.eye(n, dtype=A.dtype)
        if info is not None:
            info.update(iters=0, converged=True, max_abs_C=1.0, tol=tol,
                        identity_error=0.0, note="N <= r, all rows selected")
        return piv, restore(eye)

    top_k = _clamp_top_k(top_k_index, n, r)
    rdt = np.dtype(bk.real_dtype(bk.canon_dtype(A.dtype)))

    # --- starting row set: LU with partial pivoting on the first top_k rows ---
    B = np.array(A[:top_k], order="F", copy=True)
    getrf = get_lapack_funcs("getrf", (B,))
    lu, ipiv, lapack_info = getrf(B, overwrite_a=1)
    if lapack_info < 0:
        raise np.linalg.LinAlgError(f"LAPACK getrf reported illegal argument {-lapack_info}")
    if lapack_info > 0:
        raise np.linalg.LinAlgError(
            f"maxvol: the matrix is singular, U[{lapack_info - 1}, {lapack_info - 1}] = 0")
    _check_rank(lu, r, rcond, A.dtype)

    piv = np.arange(n, dtype=np.int64)  # scipy returns 0-based pivots
    for i in range(r):
        j = int(ipiv[i])
        piv[i], piv[j] = piv[j], piv[i]
    piv = piv[:r].copy()

    # --- C = A @ inv(A[piv]) --------------------------------------------------
    # solve A[piv]^T X = A^T, C = X^T.  O(N r^2), one BLAS call.
    C = np.asfortranarray(bk.solve(A[piv].T, A.T).T)

    _check_identity(C, piv, r, A.dtype, "the initial solve")

    # --- swap loop ------------------------------------------------------------
    absbuf = np.empty((top_k, r), dtype=rdt, order="F")

    def _argmax():
        np.abs(C[:top_k], out=absbuf)
        pos = int(absbuf.argmax())          # argmax flattens in C order
        i, j = divmod(pos, r)
        return i, j, float(absbuf[i, j])

    i, j, mx = _argmax()
    iters = 0
    while mx > tol and iters < max_iters:
        pivot = C[i, j]                     # |pivot| = mx > tol >= 1, safe to divide
        col = C[:, j].copy()
        row = C[i, :].copy()
        row[j] -= 1.0
        piv[j] = i
        _ger(C, -1.0 / pivot, col, row)
        iters += 1
        i, j, mx = _argmax()

    converged = mx <= tol
    drift = _check_identity(C, piv, r, A.dtype, f"{iters} rank-1 updates")
    if info is not None:
        info.update(iters=iters, converged=bool(converged), max_abs_C=mx, tol=tol,
                    identity_error=drift)
    if not converged and warn_nonconvergence:
        warnings.warn(
            f"maxvol did not converge: after {iters} swaps max|C| = {mx:.6g} > tol = "
            f"{tol:.6g}. The factorisation a = C a[piv] still holds, the submatrix "
            "is just not dominant.", RuntimeWarning, stacklevel=2)
    return piv, restore(C)


# --------------------------------------------------------------------------- #
# rectangular maxvol
# --------------------------------------------------------------------------- #

def rect_maxvol(a, tol=1.05, maxK=None, min_add_K=None, minK=None,
                start_maxvol_iters=10, identity_submatrix=True, top_k_index=-1,
                rcond=None, info=None):
    """Rectangular maximum-volume submatrix: ``K >= r`` rows, small ``||C[i]||_2``.

    Greedy maximisation of the 2-volume: start from the square maxvol row set and
    keep adding the row whose coefficient vector has the largest 2-norm, until
    every row of ``C = A pinv(A[piv])`` has ``||C[i]||_2 <= tol``.  Adding row
    ``i`` is a rank-1 update of ``(H^* H)^{-1}`` (Sherman-Woodbury-Morrison), so
    ``C`` is updated in ``O(N K)`` instead of a fresh pseudo-inverse: with
    ``c = C[i]``, ``v = C conj(c)``, ``l = 1 / (1 + v[i])``

    ``C <- [C - l outer(v, c) | l v]``  and  ``||C[j]||^2 <- ||C[j]||^2 - l |v_j|^2``.

    The row norms therefore decrease monotonically, which is what bounds the
    interpolation error of a cross approximation.

    Args:
        a: ``(N, r)`` real or complex matrix, ``N >= r``, full column rank.
        tol: Upper bound for the 2-norm of the rows of ``C``.  Unlike
            :func:`maxvol` values below 1 are meaningful here (they simply ask for
            more rows) and are *not* remapped.
        maxK: Maximum number of selected rows; default (``None``) is ``N``.
        min_add_K: Minimum number of rows to add on top of the square ``r``.
        minK: Minimum number of selected rows; ``max(minK, r + min_add_K)`` wins.
        start_maxvol_iters: Number of square-maxvol swaps done before the
            rectangular phase.
        identity_submatrix: Overwrite ``C[piv]`` with the exact identity.  The
            maintained ``C`` is ``A pinv(A[piv])`` whose ``piv`` rows form the
            orthogonal projector onto the row space of ``A[piv]``, not the
            identity; both choices satisfy ``A[piv] = C[piv] A[piv]``, so this is
            a convention, not an approximation.
        top_k_index: Only rows ``0 .. top_k_index - 1`` may be selected.
        rcond: Passed to :func:`maxvol` for the rank test.
        info: Optional dict, filled with ``K``, ``max_row_norm`` (over all ``N``
            rows), ``max_row_norm_bounded`` (over the ``top_k_index`` rows the
            tolerance actually applies to), ``top_k_index``, ``converged``,
            ``stop_reason`` and the nested square-maxvol ``info``.

    Returns:
        tuple: ``(piv, C)``.

        * ``piv``: ``(K,)`` numpy int array of selected rows, ``K >= r``.
        * ``C``: ``(N, K)`` matrix with ``a == C @ a[piv]`` up to roundoff and
          ``||C[i]||_2 <= tol`` for every row.  ``info['converged']`` is
          ``False`` exactly when that last statement fails, which happens when
          ``maxK``/``top_k_index`` stopped the growth early, and also whenever
          ``tol < 1``: the growth criterion only ever inspects the *unselected*
          rows, and the selected ones have norm ``1`` (``identity_submatrix``)
          or up to ``1`` (rows of the orthoprojector), so a bound below 1 is
          unreachable by construction.  Same backend/device as ``a``.

    Raises:
        numpy.linalg.LinAlgError: rank-deficient input (from :func:`maxvol`).
        ValueError: contradictory ``minK > maxK``, or non-matrix input.

    Note:
        ``N <= r``: all rows are returned with ``C = I``, as in the legacy code.
        The ``||C||_2`` bounds of the paper are proved for the *exact* 2-volume
        maximiser; what this greedy routine guarantees is the stopping criterion
        ``max_i ||C[i]||_2 <= tol``.

    Reference:
        A. Mikhalev, I. V. Oseledets, "Rectangular maximum-volume submatrices and
        their applications", Linear Algebra and its Applications 538 (2018)
        187-211, arXiv:1502.07838.
    """
    A, restore = _host_matrix(a)
    n, r = A.shape
    if n <= r:
        if info is not None:
            info.update(K=n, max_row_norm=1.0, converged=True,
                        stop_reason="N <= r, all rows selected")
        return np.arange(n, dtype=np.int64), restore(np.eye(n, dtype=A.dtype))

    tol = float(tol)
    if not tol > 0.0:
        raise ValueError(f"rect_maxvol tol must be positive, got {tol}")
    tol2 = tol * tol
    top_k = _clamp_top_k(top_k_index, n, r)

    # --- bounds on K ----------------------------------------------------------
    maxK_user, minK_user = maxK, minK
    maxK = top_k if maxK is None else int(maxK)
    minK = r if minK is None else int(minK)
    if min_add_K is not None:
        minK = max(minK, r + int(min_add_K))
    if maxK_user is not None and minK_user is not None and minK_user > maxK_user:
        raise ValueError(
            f"rect_maxvol: minK={minK_user} > maxK={maxK_user}, the request is "
            "contradictory")
    # clamp against the physical bounds: at least r rows, at most top_k available
    maxK = min(max(maxK, r), top_k)
    minK = min(max(minK, r), maxK)

    dt = A.dtype
    rdt = np.dtype(bk.real_dtype(bk.canon_dtype(dt)))

    # deliberately truncated square phase: it is a warm start, so its
    # non-convergence is expected and only recorded (info['maxvol_info']).
    sq_info = {}
    piv0, C0 = maxvol(A, tol=1.0, max_iters=start_maxvol_iters, top_k_index=top_k,
                      rcond=rcond, info=sq_info, warn_nonconvergence=False)

    piv = np.zeros(maxK, dtype=np.int64)
    piv[:r] = piv0

    # growable (N, cap) Fortran buffer: column slices stay F-contiguous, so BLAS
    # ger can update them in place; capacity doubles, so copying is amortised O(N K)
    cap = min(maxK, max(2 * r, r + 8))
    C = np.zeros((n, cap), dtype=dt, order="F")
    C[:, :r] = C0

    row_norm_sqr = (np.abs(C[:top_k, :r]) ** 2).sum(axis=1).astype(rdt)
    chosen = np.ones(top_k, dtype=rdt)
    chosen[piv[:r]] = 0.0
    row_norm_sqr *= chosen
    row_norm_sqr[piv[:r]] = -1.0     # never re-select a row, even when minK forces growth

    K = r
    i = int(row_norm_sqr.argmax())
    while (row_norm_sqr[i] > tol2 and K < maxK) or K < minK:
        if K == cap:
            cap = min(maxK, max(cap * 2, cap + 1))
            grown = np.zeros((n, cap), dtype=dt, order="F")
            grown[:, :K] = C[:, :K]
            C = grown
        piv[K] = i
        chosen[i] = 0.0
        c = C[i, :K].copy()
        v = C[:, :K] @ c.conj()
        # v[i] = ||C[i]||^2 is real by construction; .real drops the roundoff drift
        l = 1.0 / (1.0 + float(np.real(v[i])))
        _ger(C[:, :K], -l, v, c)
        C[:, K] = l * v
        row_norm_sqr -= l * (v[:top_k] * v[:top_k].conj()).real
        row_norm_sqr *= chosen
        row_norm_sqr[piv[:K + 1]] = -1.0
        K += 1
        i = int(row_norm_sqr.argmax())

    piv = piv[:K].copy()
    C = C[:, :K].copy()
    if identity_submatrix:
        C[piv] = np.eye(K, dtype=dt)

    all_row_norms = np.linalg.norm(C, axis=1)
    max_row_norm = float(all_row_norms.max())
    # the tolerance only ever applied to the rows top_k_index allowed us to pick,
    # so that -- and not the global maximum -- is what may be called converged
    max_row_norm_bounded = float(all_row_norms[:top_k].max())

    # Two separate facts, and both have to hold before we may claim success.
    #
    # (1) the greedy criterion: no *candidate* row is left above tol.  A row that
    #     is already selected carries -1 in row_norm_sqr, so when everything is
    #     selected the criterion is vacuously satisfied.
    # (2) the advertised post-condition: EVERY row of the returned C is within
    #     tol.  This is not implied by (1): the criterion only ever looks at the
    #     unselected rows, while the selected ones have norm exactly 1 (with
    #     identity_submatrix) or up to 1 (rows of the orthoprojector).  For
    #     tol >= 1 that is automatic; for tol < 1 it is false, and reporting
    #     converged=True there would be a plausible-looking lie.
    remaining = float(row_norm_sqr[i])
    exhausted = remaining < 0.0
    criterion_met = remaining <= tol2
    postcondition_met = max_row_norm_bounded <= tol * (1.0 + 1e-9)
    converged = bool(criterion_met and postcondition_met)
    if converged:
        stop_reason = "tolerance"
    elif not criterion_met:
        stop_reason = "maxK" if K >= maxK else "minK"
    else:
        stop_reason = "all rows selected" if exhausted else "tol below 1"
    if info is not None:
        info.update(K=K, max_row_norm=max_row_norm,
                    max_row_norm_bounded=max_row_norm_bounded,
                    top_k_index=top_k, converged=converged,
                    stop_reason=stop_reason, maxvol_info=sq_info)
    if not converged:
        if not criterion_met:
            detail = (f"stopped at K = {K} (limit maxK = {maxK}) with a remaining "
                      f"candidate row norm {np.sqrt(max(remaining, 0.0)):.6g}")
        else:
            detail = (f"stopped at K = {K} ({stop_reason}); no candidate row is "
                      f"above the tolerance, but the largest row norm of C is "
                      f"{max_row_norm_bounded:.6g} -- the selected rows themselves "
                      "are not below tol, which tol < 1 can never achieve")
        warnings.warn(f"rect_maxvol {detail} > tol = {tol:.6g}.",
                      RuntimeWarning, stacklevel=2)
    return piv, restore(C)


# --------------------------------------------------------------------------- #
# convenience wrappers (legacy public API of rect_maxvol.py)
# --------------------------------------------------------------------------- #

def _svd_basis(a, svd_tol):
    """One SVD, truncated at ``svd_tol * s[0]``; returns ``(U_r, V_r)``.

    ``V_r = vh[:rank].conj().T`` are the right singular vectors as *columns*,
    which is the tall matrix column selection has to run on.  A single
    ``np.linalg.svd`` call: computing the rank from a separate ``compute_uv=False``
    SVD would double the ``O(N M^2)`` cost, which dominates these wrappers.
    """
    u, s, vh = np.linalg.svd(np.asarray(a), full_matrices=False)
    rank = max(int(np.sum(s >= s[0] * svd_tol)), 1)
    return u[:, :rank], vh[:rank].conj().T


def _orthonormal_basis(A, rcond):
    """``Q`` from ``A = Q R``, refusing a rank-deficient ``A``.

    Householder QR happily returns an orthonormal ``Q`` for a rank-deficient
    ``A`` -- the extra columns are an arbitrary complement of the column space,
    perfectly conditioned, so :func:`maxvol` cannot notice anything.  The
    resulting factorisation ``a = C a[piv]`` is even exact, but ``a[piv]`` is
    singular and the "maximum volume" it maximises is zero, which is precisely
    the plausible-looking wrong answer the direct call refuses to give.  The
    rank is therefore tested on ``diag(R)`` here, with the same criterion
    :func:`maxvol` applies to ``diag(U)``.
    """
    q, r = bk.qr(A)
    diag = np.abs(np.diag(np.asarray(r)))
    top = float(diag.max()) if diag.size else 0.0
    low = float(diag.min()) if diag.size else 0.0
    if rcond is None:
        rcond = max(A.shape[1], 1) * bk.eps_of(bk.canon_dtype(A.dtype))
    if top == 0.0 or low <= rcond * top:
        raise np.linalg.LinAlgError(
            "maxvol_qr: the matrix is rank deficient / numerically singular "
            f"(smallest |R_ii| = {low:.3e}, largest = {top:.3e}, threshold = "
            f"{rcond:.3e} * largest). QR does not fix a rank deficiency: it "
            "would return rows whose submatrix has zero volume. Drop the "
            "dependent columns, or use maxvol_svd, which truncates them.")
    return q


def maxvol_qr(a, tol=1.05, max_iters=100, rcond=None, **kwargs):
    """QR first, then :func:`maxvol` on ``Q``.

    The safe way to call maxvol on a matrix that is merely tall and badly
    scaled, not orthogonal: ``Q`` spans the same column space and is perfectly
    conditioned, and the row set of maximal volume is the same, since ``A = Q R``
    differs from ``Q`` by an invertible right factor.  Returns ``(piv, C)`` with
    ``a == C @ a[piv]``.

    Raises:
        numpy.linalg.LinAlgError: ``a`` is rank deficient.  QR is not a cure for
            that -- see :func:`_orthonormal_basis`.
    """
    A, restore = _host_matrix(a)
    if A.shape[0] <= A.shape[1]:
        return np.arange(A.shape[0], dtype=np.int64), restore(np.eye(A.shape[0], dtype=A.dtype))
    q = _orthonormal_basis(A, rcond)
    piv, c = maxvol(q, tol=tol, max_iters=max_iters, **kwargs)
    return piv, restore(np.asarray(c))


def rect_maxvol_qr(a, tol=1.05, rcond=None, **kwargs):
    """QR first, then :func:`rect_maxvol` on ``Q``. See :func:`maxvol_qr`."""
    A, restore = _host_matrix(a)
    if A.shape[0] <= A.shape[1]:
        return np.arange(A.shape[0], dtype=np.int64), restore(np.eye(A.shape[0], dtype=A.dtype))
    q = _orthonormal_basis(A, rcond)
    piv, c = rect_maxvol(q, tol=tol, **kwargs)
    return piv, restore(np.asarray(c))


def maxvol_svd(a, svd_tol=1e-3, tol=1.05, max_iters=100, job="F", **kwargs):
    """Truncated SVD, then :func:`maxvol` on the singular vectors.

    Args:
        a: ``(N, M)`` matrix of any shape.
        svd_tol: Relative cut-off for the singular values.
        job: ``'R'`` rows (left singular vectors), ``'C'`` columns (right ones),
            ``'F'`` both, returning ``(piv_row, C_row, piv_col, C_col)``.  The
            default is ``'F'``, matching the legacy ``rect_maxvol.maxvol_svd``.

    Note:
        ``a`` is only *approximated* here, with an error of the order of the
        discarded singular values: ``a ~ C @ a[piv]`` for ``job='R'`` and
        ``a ~ a[:, piv] @ C.conj().T`` for ``job='C'``.  The conjugate is not
        cosmetic -- the legacy code selected columns on ``V.T`` instead of
        ``V.conj().T`` and therefore returned a transposed-not-adjoint ``C`` for
        complex input; here ``job='C'`` reproduces a to full accuracy for
        complex matrices too.
    """
    u_r, v_r = _svd_basis(_host_matrix(a)[0], svd_tol)
    if job == "R":
        return maxvol(u_r, tol=tol, max_iters=max_iters, **kwargs)
    if job == "C":
        return maxvol(v_r, tol=tol, max_iters=max_iters, **kwargs)
    if job == "F":
        return (*maxvol(u_r, tol=tol, max_iters=max_iters, **kwargs),
                *maxvol(v_r, tol=tol, max_iters=max_iters, **kwargs))
    raise ValueError(f"job must be 'R', 'C' or 'F', got {job!r}")


def rect_maxvol_svd(a, svd_tol=1e-3, tol=1.05, job="F", **kwargs):
    """Truncated SVD, then :func:`rect_maxvol` on the singular vectors.

    See :func:`maxvol_svd` for the ``job`` convention (default ``'F'``, as in the
    legacy ``rect_maxvol.rect_maxvol_svd``).
    """
    u_r, v_r = _svd_basis(_host_matrix(a)[0], svd_tol)
    if job == "R":
        return rect_maxvol(u_r, tol=tol, **kwargs)
    if job == "C":
        return rect_maxvol(v_r, tol=tol, **kwargs)
    if job == "F":
        return (*rect_maxvol(u_r, tol=tol, **kwargs),
                *rect_maxvol(v_r, tol=tol, **kwargs))
    raise ValueError(f"job must be 'R', 'C' or 'F', got {job!r}")
