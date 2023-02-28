"""Module fft extends applicability of common Fourier-based transformations to
tensor train domain. The interface of the routines are designed to comply
:py:mod:`scipy.fft` package.

.. currentmodule:: tt.fft

Fast Fourier Transforms (FFTs)
==============================

.. autosummary::
   :toctree: generated/

   fft - Fast (discrete) Fourier Transform (FFT)
   fftn - N-D FFT
   ifft - Inverse FFT
   ifftn - N-D inverse FFT

Discrete Sin and Cosine Transforms (DST and DCT)
================================================

.. autosummary::
   :toctree: generated/

   dct - Discrete cosine transform
   dctn - N-D Discrete cosine transform
   dst - Discrete sine transform
   dstn - N-D Discrete sine transform
   idct - Inverse discrete cosine transform
   idctn - N-D Inverse discrete cosine transform
   idst - Inverse discrete sine transform
   idstn - N-D Inverse discrete sine transform
"""

from typing import Literal, Optional, Tuple

import numpy as np
import scipy as sp
import scipy.fft
from numpy.typing import ArrayLike

from tt.core.vector import TensorTrain

__all__ = ('DCTType', 'DSTType', 'NormType', 'dct', 'dctn', 'dst', 'dstn',
           'fft', 'fftn', 'idct', 'idctn', 'idst', 'idstn', 'ifft', 'ifftn')


NormType = Literal['backward', 'forward', 'ortho']

DCTType = Literal[1, 2, 3, 4]

DSTType = Literal[1, 2, 3, 4]

DOCSTRING_FFT_1D = """\
Return the {direct}{name} Transform ({abbrv}) of an tensor train.

:param ten: The input tensor train.
:param n: Length of the transform.
:param axis: Axis along which the transform is computed.
    Default is the lastaxis.
:param norm: Normalization mode. Default is `backward`.
:return: The transformed input tensor train.
"""

DOCSTRING_FFT_ND = """\
Compute the N-D {direct}{name} Transform ({abbrv}) of an tensor train.

:param ten: The input tensor train.
:param s: Shape (length of each transformed axis) of the transformed output. If
    `s` is not given then, then the shape of the input along the axes specified
    by `axes` is used.
:param axes: Axes over which to compute the transform. If not given, the last
    `len(s)` axes are used, or all axes if `s` is also not specified.
:param norm: Normalization mode. Default is `backward`.
:return: The transformed input tensor train.
"""

DOCSTRING_REAL_1D = """\
Return the {direct}{name} Transform ({abbrv}) of an tensor train.

:param ten: The input tensor train.
:param type: Type of the transformation.
:param n: Length of the transform.
:param axis: Axis along which the transform is computed. Default is the last
    axis which is `-1`.
:param norm: Normalization mode. Default is `backward`.
:return: The transformed input tensor train.
"""

DOCSTRING_REAL_ND = """\
Return multidimensional {direct}{name} Transform ({abbrv}) of an tensor train
along specified axes.

:param ten: The input tensor train.
:param type: Type of the transformation.
:param s: Shape (length of each transformed axis) of the transformed output. If
    `s` is not given then, then the shape of the input along the axes specified
    by `axes` is used.
:param axes: Axes over which to compute the transform. If not given, the last
    `len(s)` axes are used, or all axes if `s` is also not specified.
:param norm: Normalization mode. Default is `backward`.
:return: The transformed input tensor train.
"""


def annotate(docstring: str):
    """Function annotate is a decorator which adjust docstring pattern to a
    concrete function.
    """
    def annotated(fn):
        funcname = fn.__name__

        # Choose transformation direction.
        direct = ''
        if funcname.startswith('i'):
            direct = 'Inverse '
            funcname = funcname[1:]

        # Select between cosine, sine or fast.
        name = funcname
        if funcname.startswith('fft'):
            name = 'Discrete Fourier'
        elif funcname.startswith('dct'):
            name = 'Discrete Cosine'
        elif funcname.startswith('dst'):
            name = 'Discrete Sine'

        # Infer abbreviation for transformation.
        abbrv = 'i' + funcname[:3] if direct else funcname[:3]
        abbrv = abbrv.upper()

        fn.__doc__ = docstring.format(abbrv=abbrv, direct=direct, name=name)
        return fn
    return annotated


def apply_fast_1d(fn, ten: TensorTrain, n: Optional[int], axis: Optional[int],
                  norm: Optional[NormType], overwrite_x: bool) -> TensorTrain:
    axis = sanitize_axis(ten, axis)
    core = fn(ten.cores[axis], n, 1, norm, overwrite_x)
    cores = ten.cores[:axis] + [core] + ten.cores[axis + 1:]
    return TensorTrain.from_list(cores)


def apply_real_1d(fn, ten: TensorTrain, type: int, n: Optional[int],
                  axis: Optional[int], norm: Optional[NormType],
                  overwrite_x: bool) -> TensorTrain:
    axis = sanitize_axis(ten, axis)
    core = fn(ten.cores[axis], type, n, 1, norm, overwrite_x)
    cores = ten.cores[:axis] + [core] + ten.cores[axis + 1:]
    return TensorTrain.from_list(cores)


def apply_fast_nd(fn, ten: TensorTrain, s: Optional[ArrayLike],
                  axes: Optional[ArrayLike], norm: Optional[NormType],
                  overwrite_x: bool) -> TensorTrain:
    if axes is not None:
        axes = np.asarray(axes)
    if s is not None:
        s = np.asarray(s)
    shape, axes = sanitize_dims(ten, s, axes)

    cores = []
    for i, core in enumerate(ten.cores):
        if i in axes:
            size = shape[i]
            core = fn(core, size, 1, norm, overwrite_x)
        cores.append(core)
    return TensorTrain.from_list(cores)


def apply_real_nd(fn, ten: TensorTrain, type: int, s: Optional[ArrayLike],
                  axes: Optional[ArrayLike], norm: Optional[NormType],
                  overwrite_x: bool) -> TensorTrain:
    if axes is not None:
        axes = np.asarray(axes)
    if s is not None:
        s = np.asarray(s)
    shape, axes = sanitize_dims(ten, s, axes)

    cores = []
    for i, core in enumerate(ten.cores):
        if i in axes:
            size = shape[i]
            core = fn(core, type, size, 1, norm, overwrite_x)
        cores.append(core)
    return TensorTrain.from_list(cores)


def sanitize_axis(ten: TensorTrain, axis: Optional[int]) -> int:
    if axis is None:
        return ten.ndim - 1
    elif -ten.ndim <= axis < ten.ndim:
        return axis % ten.ndim
    else:
        raise ValueError('Wrong value of axis. '
                         f'It should be from 0 to {ten.ndim}: axis={axis}.')


def sanitize_dims(ten: TensorTrain, shape: Optional[np.ndarray],
                  axes: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if axes is not None:
        axes = np.asarray(axes)
        if axes.ndim == 0:
            axes = axes[None]
        elif axes.ndim != 1:
            raise ValueError('Value of axes should be either None, int or '
                             'array-like of ints.')

    if shape is not None:
        shape = np.asarray(shape)
        if shape.ndim == 0:
            shape = shape[None]
        elif shape.ndim != 1:
            raise ValueError('Value of output shape should be either None, '
                             'int or array-like of ints.')

    if axes is None and shape is None:
        valid_axes = np.arange(ten.ndim)
        valid_shape = np.asarray(ten.shape)
    elif axes is not None and shape is None:
        valid_axes = np.asarray(axes)
        valid_shape = np.take(ten.shape, axes)
    elif axes is None and shape is not None:
        valid_axes = np.arange(ten.ndim - shape.size)
        valid_shape = np.asarray(axes)
    else:
        valid_axes = np.asarray(axes)
        valid_shape = np.asarray(shape)

    if len(valid_axes) != len(valid_shape):
        raise ValueError('Number elements of shape and axes does not match.')

    # Negative axes becomes positive (e.g. -1 -> ndim - 1).
    valid_axes = valid_axes % ten.ndim
    return valid_shape, valid_axes


@annotate(DOCSTRING_REAL_1D)
def dct(ten: TensorTrain, type: DSTType = 2, n: Optional[int] = None,
        axis: Optional[int] = None, norm: Optional[NormType] = None,
        overwrite_x: bool = False) -> TensorTrain:
    return apply_real_1d(sp.fft.dct, **locals())


@annotate(DOCSTRING_REAL_ND)
def dctn(ten: TensorTrain, type: DCTType = 2, s: Optional[ArrayLike] = None,
         axes: Optional[ArrayLike] = None, norm: Optional[NormType] = None,
         overwrite_x: bool = False) -> TensorTrain:
    return apply_real_nd(sp.fft.dct, **locals())


@annotate(DOCSTRING_REAL_1D)
def dst(ten: TensorTrain, type: DSTType = 2, n: Optional[int] = None,
        axis: Optional[int] = None, norm: Optional[NormType] = None,
        overwrite_x: bool = False) -> TensorTrain:
    return apply_real_1d(sp.fft.dst, **locals())


@annotate(DOCSTRING_REAL_ND)
def dstn(ten: TensorTrain, type: DSTType = 2, s: Optional[ArrayLike] = None,
         axes: Optional[ArrayLike] = None, norm: Optional[NormType] = None,
         overwrite_x: bool = False) -> TensorTrain:
    return apply_real_nd(sp.fft.dst, **locals())


@annotate(DOCSTRING_FFT_1D)
def fft(ten: TensorTrain, n: Optional[int] = None, axis: Optional[int] = None,
        norm: Optional[NormType] = None,
        overwrite_x: bool = False) -> TensorTrain:
    return apply_fast_1d(sp.fft.fft, **locals())


@annotate(DOCSTRING_FFT_ND)
def fftn(ten: TensorTrain, s: Optional[ArrayLike] = None,
         axes: Optional[ArrayLike] = None, norm: Optional[NormType] = None,
         overwrite_x: bool = False) -> TensorTrain:
    return apply_fast_nd(sp.fft.fft, **locals())


@annotate(DOCSTRING_REAL_1D)
def idct(ten: TensorTrain, type: DCTType = 2, n: Optional[int] = None,
         axis: Optional[int] = None, norm: Optional[NormType] = None,
         overwrite_x: bool = False) -> TensorTrain:
    return apply_real_1d(sp.fft.idct, **locals())


@annotate(DOCSTRING_REAL_ND)
def idctn(ten: TensorTrain, type: DCTType = 2, s: Optional[ArrayLike] = None,
          axes: Optional[ArrayLike] = None, norm: Optional[NormType] = None,
          overwrite_x: bool = False) -> TensorTrain:
    return apply_real_nd(sp.fft.idct, **locals())


@annotate(DOCSTRING_REAL_1D)
def idst(ten: TensorTrain, type: DSTType = 2, n: Optional[int] = None,
         axis: Optional[int] = None, norm: Optional[NormType] = None,
         overwrite_x: bool = False) -> TensorTrain:
    return apply_real_1d(sp.fft.idst, **locals())


@annotate(DOCSTRING_REAL_ND)
def idstn(ten: TensorTrain, type: DCTType = 2, s: Optional[ArrayLike] = None,
          axes: Optional[ArrayLike] = None, norm: Optional[NormType] = None,
          overwrite_x: bool = False) -> TensorTrain:
    return apply_real_nd(sp.fft.idst, **locals())


@annotate(DOCSTRING_FFT_1D)
def ifft(ten: TensorTrain, n: Optional[int] = None, axis: Optional[int] = None,
         norm: Optional[NormType] = None,
         overwrite_x: bool = False) -> TensorTrain:
    return apply_fast_1d(sp.fft.ifft, **locals())


@annotate(DOCSTRING_FFT_ND)
def ifftn(ten: TensorTrain, s: Optional[ArrayLike] = None,
          axes: Optional[ArrayLike] = None, norm: Optional[NormType] = None,
          overwrite_x: bool = False) -> TensorTrain:
    return apply_fast_nd(sp.fft.ifft, **locals())
