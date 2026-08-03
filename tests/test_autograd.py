"""Gradients must flow through the TT operations on the torch backend.

This is a *contract* file, not an algorithm file: it asks whether a functional
built from ordinary ttpy2 calls can be differentiated by ``torch.autograd``,
against central finite differences as the oracle.

It needs torch but not a GPU -- the property has nothing to do with the device,
and pinning it to CUDA would mean it never runs where it is cheapest to run.

Why it exists.  ``TorchBackend.norm`` used to end in ``.item()``.  Nothing
failed: the value was right, ``requires_grad`` was still True further down the
expression, and the gradient of any term containing ``||x||`` was simply
missing.  A silent wrong number is the one outcome this project treats as worse
than a crash, so the property is nailed down here.
"""

from __future__ import annotations

import numpy as np
import pytest

import tt
from tt import backend as bk
from tt.core import _ops

torch = pytest.importorskip("torch")


def rand_cores(d=4, n=3, r=2, seed=0, requires_grad=True):
    """Seeded float64 CPU cores of a rank-``r`` TT tensor."""
    rng = np.random.default_rng(seed)
    ranks = [1] + [r] * (d - 1) + [1]
    return [torch.tensor(rng.standard_normal((ranks[k], n, ranks[k + 1])),
                         dtype=torch.float64, requires_grad=requires_grad)
            for k in range(d)]


def finite_difference(fn, cores, h=1e-6):
    """Central differences of ``fn`` w.r.t. every entry of every core."""
    out = []
    for k in range(len(cores)):
        g = np.zeros(cores[k].shape)
        flat = g.reshape(-1)
        for i in range(flat.size):
            for sign, store in ((+1.0, "plus"), (-1.0, "minus")):
                pert = [c.detach().clone() for c in cores]
                pert[k].reshape(-1)[i] += sign * h
                val = float(fn(pert))
                if store == "plus":
                    up = val
                else:
                    down = val
            flat[i] = (up - down) / (2 * h)
        out.append(g)
    return out


def test_norm_keeps_the_autograd_tape():
    """The defect itself: ``||x||`` contributed a value but no gradient."""
    cores = rand_cores(d=3, n=2, r=2, seed=1)

    def fn(cs):
        return tt.vector.from_list(cs).norm()

    value = fn(cores)
    assert isinstance(value, torch.Tensor), "norm() must not collapse to a float"
    assert value.requires_grad
    value.backward()

    fd = finite_difference(fn, cores)
    for k, c in enumerate(cores):
        err = np.abs(np.asarray(c.grad) - fd[k]).max()
        assert err < 1e-6, f"core {k}: max |AD - FD| = {err:.3E}"


def test_gradient_of_a_mixed_functional_is_complete():
    """``<x,x> + ||x - b||`` -- the shape of a completion objective.

    With ``.item()`` in the norm the second term contributed exactly nothing and
    the total gradient was off by 7.0e-01 while every value printed correctly.
    """
    cores = rand_cores(d=4, n=3, r=2, seed=0)
    bcores = rand_cores(d=4, n=3, r=2, seed=7, requires_grad=False)

    def fn(cs):
        x = tt.vector.from_list(cs)
        b = tt.vector.from_list(bcores)
        return tt.dot(x, x) + (x - b).norm()

    total = fn(cores)
    total.backward()

    fd = finite_difference(fn, cores)
    for k, c in enumerate(cores):
        err = np.abs(np.asarray(c.grad) - fd[k]).max()
        scale = max(1.0, np.abs(fd[k]).max())
        assert err / scale < 1e-6, f"core {k}: max |AD - FD| = {err:.3E}"

    # and the dropped term was not negligible: this is what used to vanish
    x = tt.vector.from_list([c.detach() for c in cores])
    b = tt.vector.from_list(bcores)
    assert float((x - b).norm()) > 1.0


def test_scaling_by_a_zero_dimensional_scalar_works_and_differentiates():
    """``x * (1 / x.norm())`` is the normalization every algorithm writes.

    Once ``norm()`` returns a 0-d tensor, scalar multiplication has to accept
    one, or the fix that kept the tape alive would break the very expression
    the tape exists for.
    """
    cores = rand_cores(d=3, n=2, r=2, seed=3)

    def fn(cs):
        x = tt.vector.from_list(cs)
        return (x * (1.0 / x.norm())).norm()

    val = fn(cores)
    assert isinstance(val, torch.Tensor)
    # a normalized tensor has norm 1 whatever the input, so the gradient is 0 --
    # a weak assertion on its own, which is why the value is checked too
    assert abs(float(val.detach()) - 1.0) < 1e-12
    val.backward()
    for k, c in enumerate(cores):
        assert np.abs(np.asarray(c.grad)).max() < 1e-8, f"core {k} moved"


def test_is_scalar_agrees_across_backends():
    """The predicate that decides whether something may scale a TT tensor."""
    assert bk.is_scalar(2.0) and bk.is_scalar(2) and bk.is_scalar(2 + 3j)
    assert bk.is_scalar(np.float64(2.0))
    assert bk.is_scalar(np.array(2.0))
    assert bk.is_scalar(torch.tensor(2.0))
    assert not bk.is_scalar(np.array([2.0]))
    assert not bk.is_scalar(torch.tensor([2.0]))
    assert not bk.is_scalar("2.0")
    # a plain real carries no dtype opinion; an array scalar does
    assert bk.scalar_dtype(2.0) is None
    assert bk.scalar_dtype(2 + 3j) == "complex128"
    assert bk.scalar_dtype(np.array(2.0, dtype=np.float32)) == "float32"


def test_numpy_backend_norm_is_unchanged():
    """The torch fix must not have moved numpy's return type."""
    a = np.arange(6.0).reshape(2, 3)
    n = bk.norm(a)
    assert np.ndim(n) == 0
    assert abs(float(n) - np.linalg.norm(a)) < 1e-15
    assert float(_ops.norm(tt.rand([2, 3, 2], r=2).cores)) > 0.0
