from os import PathLike
from typing import IO, Optional

import numpy as np

from .core.tensor_train import TensorTrain

TT_MAGIC = b'TT      '


def read(filename_or_fileobj: IO | PathLike, endianess: str = '=',
         native: bool = False) -> TensorTrain:
    """Function load_tt reads tensor represented in TT-format. This function
    corresponds to the reading routine from `tt-fort` project but provides an
    implementation in Python.
    """
    if isinstance(filename_or_fileobj, PathLike):
        if native:
            return TensorTrain.read(filename_or_fileobj)
        with open(filename_or_fileobj, 'rb') as fin:
            return _read(fin, endianess)
    else:
        return _read(filename_or_fileobj, endianess)


def _read(fin: IO, endianess: str) -> TensorTrain:
    int32 = np.dtype(np.int32).newbyteorder(endianess)

    elem_dtypes = {
        0: np.dtype(np.float64).newbyteorder(endianess),
        1: np.dtype(np.complex128).newbyteorder(endianess),
        2: np.dtype(np.float32).newbyteorder(endianess),
        3: np.dtype(np.complex64).newbyteorder(endianess),
    }

    # Verify file format signature.
    if (magic := fin.read(8)) != TT_MAGIC:
        raise RuntimeError(f'Wrong file format signature: {magic}.')

    version = tuple(np.fromfile(fin, int32, 2))
    if version[0] != 1:
        raise RuntimeError(f'Unsupported file format version: {version}.')

    # Read portion of field.
    _, dtype_code, _, _ = np.fromfile(fin, int32, 4)

    # Infer element type.
    if (dtype_elem := elem_dtypes.get(dtype_code)) is None:
        raise ValueError(f'Unknown element type code: {dtype_code}.')

    # Read comment.
    fin.read(64)

    # Read (reserved?) field again.
    np.fromfile(fin, int32, 8)

    # Read interval bound of arrays which specifies tensor train.
    left, right = np.fromfile(fin, int32, 2)
    ndim = right - left + 1

    # Read shape and rank.
    shape = np.fromfile(fin, int32, ndim)
    rank = np.fromfile(fin, int32, ndim + 1)

    # Infer core shape and read them to list.
    cores = []
    core_shapes = np.stack([rank[:-1], shape, rank[1:]]).T
    core_sizes = core_shapes.prod(axis=1)
    for core_shape, core_size in zip(core_shapes, core_sizes):
        core = np \
            .fromfile(fin, dtype_elem, core_size) \
            .reshape(core_shape, order='F')
        cores.append(core)

    return TensorTrain.from_list(cores)


def write(filename_or_fileobj: IO | PathLike, tensor_train: TensorTrain,
          endianess: str = '=', comment: Optional[str] = None,
          native: bool = False):
    """Function write writes a tensor represented at TT-format to a binary
    file. This function corresponds to the writing routine from `tt-fort`
    project but provides an implementation in Python.
    """
    if isinstance(filename_or_fileobj, PathLike):
        if native:
            return tensor_train.write(filename_or_fileobj)
        with open(filename_or_fileobj, 'wb') as fout:
            return _write(fout, tensor_train, endianess, comment)
    else:
        return _write(filename_or_fileobj, tensor_train, endianess, comment)


def _write(fout: IO, tensor_train: TensorTrain, endianess: str,
           comment: Optional[str]):
    int32 = np.dtype(np.int32).newbyteorder(endianess)

    elem_dtypes = {
        np.dtype(np.float64).newbyteorder(endianess): 0,
        np.dtype(np.complex128).newbyteorder(endianess): 1,
        np.dtype(np.float32).newbyteorder(endianess): 2,
        np.dtype(np.complex64).newbyteorder(endianess): 3,
    }

    fout.write(TT_MAGIC)

    version = np.array([1, 0], int32)
    version.tofile(fout)

    # Round up reuired number of slots in arena to the closest power of two.
    limit = 2 ** int(np.ceil(np.log2(len(tensor_train.shape))))

    # Infer type code for elements
    dtype_elem = tensor_train.dtype.newbyteorder(endianess)
    dtype_code = elem_dtypes.get(dtype_elem)
    if dtype_code is None:
        raise ValueError(f'Unsupported elemen type {elem_dtypes}.')

    info = np.array([limit, dtype_code, 0, 0], int32)
    info.tofile(fout)

    # Write fixed-length comment string in UTF-8.
    padding = 64
    if comment:
        encoded = comment.encode('utf8')[:padding]
        padding -= len(encoded)
        fout.write(encoded)
    fout.write(b'\x00' * padding)

    # Write information about arena (interval in fixed buffer to store
    # information about TT-cores).
    left, right = 1, len(tensor_train.shape)
    arena = np.array([left, 0, right, 0, 0, 0, 0, 0], int32)
    arena.tofile(fout)

    interval = np.array([left, right], int32)
    interval.tofile(fout)

    # Write shape and rank.
    shape = np.array(tensor_train.shape, int32)
    shape.tofile(fout)
    ranks = np.array(tensor_train.ranks, int32)
    ranks.tofile(fout)

    # Write cores sequentially.
    for core in tensor_train.cores:
        data = core \
            .reshape(-1, order='F') \
            .tobytes(order='F')
        fout.write(data)
