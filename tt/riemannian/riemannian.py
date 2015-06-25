# References:
# [1] C. Lubich, I. Oseledets and B. Vandereycken, Time integration of
# tensor trains.

import tt
import numpy as np
from math import ceil


def reshape(a, sz):
    return np.reshape(a, sz, order="F")


# Three functions below are here for debugging. They help to implement
# the algorithms in a straightforward and inefficient manner.
def left(X, i):
    """Compute the orthogonal matrix Q_{\leq i} as defined in [1]."""
    if i < 0:
        return np.ones([1, 1])
    answ = np.ones([1, 1])
    cores = tt.tensor.to_list(X)
    for dim in range(i+1):
        answ = np.tensordot(answ, cores[dim], 1)
    answ = reshape(answ, (-1, X.r[i+1]))
    return answ


def right(X, i):
    """Compute the orthogonal matrix Q_{\geq i} as defined in [1]."""
    if i > X.d-1:
        return np.ones([1, 1])
    answ = np.ones([1, 1])
    cores = tt.tensor.to_list(X)
    for dim in range(X.d-1, i-1, -1):
        answ = np.tensordot(cores[dim], answ, 1)
    answ = reshape(answ, (X.r[i], -1))
    return answ.T


def unfolding(tens, i):
    """Compute the i-th unfolding of a tensor."""
    return reshape(tens.full(), (np.prod(tens.n[0:(i+1)]), -1))


def project(X, Z, clusterSize=30, debug=False):
    """ Project tensor Z on the tangent space of tensor X.

    X is a tensor in the TT format.
    Z can be a tensor in the TT format or a list of tensors (in this case
    the function computes projection of the sum off all tensors in the list:
        project(X, Z) = P_X(\sum_i Z_i)
    ).
    This function implements an algorithm from the paper [1], theorem 3.1.

    Returns a tensor in the TT format with the TT-ranks equal 2 * rank(Z).
    """
    zArr = None
    if isinstance(Z, tt.tensor):
        zArr = [Z]
    else:
        # Join small clusters of tensors to speed things up.
        # TODO: instead of just summing clusterSize tensors, sum until
        # the running sum ranks exceed a threshold.
        numClusters = int(ceil(float(len(Z)) / clusterSize))
        zArr = [0] * numClusters
        for clusterIdx in range(numClusters):
            zArr[clusterIdx] = Z[clusterIdx * clusterSize]
            start = clusterIdx * clusterSize + 1
            end = min((clusterIdx + 1) * clusterSize, len(Z))
            for idx in range(start, end):
                zArr[clusterIdx] += Z[idx]

    # Get rid of redundant ranks (they cause technical difficulties).
    X = X.round(eps=0)

    numDims = X.d
    modeSize = X.n
    coresX = tt.tensor.to_list(X)
    coresZ = [None] * len(zArr)
    for idx in range(len(zArr)):
        assert(modeSize == zArr[idx].n).all()
        coresZ[idx] = tt.tensor.to_list(zArr[idx])

    projRanks = np.concatenate(([1], 2 * X.r[1:-1], [1]))
    # Initialize the cores of the projection_X(sum z[i]).
    coresP = []
    for dim in range(numDims):
        r1 = 2 * X.r[dim]
        r2 = 2 * X.r[dim+1]
        if dim == 0:
            r1 = 1
        if dim == numDims - 1:
            r2 = 1
        coresP.append(np.zeros((r1, modeSize[dim], r2)))
    # rhs[idx][dim] is an (Z.rank_dim * X.rank_dim) x 1 vector
    rhs = [[0] * (numDims+1) for _ in range(len(zArr))]
    for idx in range(len(zArr)):
        rhs[idx][numDims] = np.ones([1, 1])
    # Right to left orthogonalization of X and preparation of the rhs vectors.
    for dim in range(numDims-1, 0, -1):
        # Orthogonalization.
        cc = coresX[dim].copy()
        r1, n, r2 = cc.shape
        if dim > 0:
            cc = reshape(cc, (r1, -1)).T
            cc, rr = np.linalg.qr(cc)
            r1 = cc.shape[1]
            # Warning: since ranks can change here, do not use X.r!
            # Use coresX[dim].shape instead.
            coresX[dim] = reshape(cc.T, (r1, n, r2)).copy()
            coresX[dim-1] = np.tensordot(coresX[dim-1], rr.T, 1)

        # Fill the right orthogonal part of the projection.
        for value in range(modeSize[dim]):
            coresP[dim][0:r1, value, 0:r2] = coresX[dim][:, value, :]
        # Compute rhs.
        for idx in range(len(zArr)):
            coreProd = np.tensordot(coresZ[idx][dim], coresX[dim], axes=(1, 1))
            coreProd = np.transpose(coreProd, (0, 2, 1, 3))
            coreProd = reshape(coreProd, (zArr[idx].r[dim]*r1, zArr[idx].r[dim+1]*r2))
            rhs[idx][dim] = np.dot(coreProd, rhs[idx][dim+1])
    if debug:
        assert(np.allclose(X.full(), tt.tensor.from_list(coresX).full()))

    # lsh[idx] is an X.rank_dim x zArr[idx].rank_dim matrix.
    lhs = [np.ones([1, 1]) for _ in range(len(zArr))]
    # Left to right sweep.
    for dim in range(numDims):
        cc = coresX[dim].copy()
        r1, n, r2 = cc.shape
        if dim < numDims-1:
            # Left to right orthogonalization.
            cc = reshape(cc, (-1, r2))
            cc, rr = np.linalg.qr(cc)
            r2 = cc.shape[1]
            # Warning: since ranks can change here, do not use X.r!
            # Use coresX[dim].shape instead.
            if debug:
                # Need to do it before the move non orthogonal part rr to
                # the coresX[dim+1].
                rightQ = right(tt.tensor.from_list(coresX), dim+1)
            coresX[dim] = reshape(cc, (r1, n, r2)).copy()
            coresX[dim+1] = np.tensordot(rr, coresX[dim+1], 1)

            for idx in range(len(zArr)):
                currZCore = reshape(coresZ[idx][dim], (zArr[idx].r[dim], -1))
                currPCore = np.dot(lhs[idx], currZCore)

                # TODO: consider using np.einsum.
                coreProd = np.tensordot(coresX[dim], coresZ[idx][dim], axes=(1, 1))
                coreProd = np.transpose(coreProd, (0, 2, 1, 3))
                coreProd = reshape(coreProd, (r1*zArr[idx].r[dim], r2*zArr[idx].r[dim+1]))
                lhs[idx] = reshape(lhs[idx], (1, -1))
                lhs[idx] = np.dot(lhs[idx], coreProd)
                lhs[idx] = reshape(lhs[idx], (r2, zArr[idx].r[dim+1]))

                currPCore = reshape(currPCore, (-1, zArr[idx].r[dim+1]))
                currPCore -= np.dot(cc, lhs[idx])
                rhs[idx][dim+1] = reshape(rhs[idx][dim+1], (zArr[idx].r[dim+1], r2))
                currPCore = np.dot(currPCore, rhs[idx][dim+1])
                currPCore = reshape(currPCore, (r1, modeSize[dim], r2))
                if dim == 0:
                    for value in range(modeSize[dim]):
                        coresP[dim][0:r1, value, 0:r2] += currPCore[:, value, :]
                else:
                    for value in range(modeSize[dim]):
                        coresP[dim][r1:, value, 0:r2] += currPCore[:, value, :]

                if debug:
                    leftQm1 = left(tt.tensor.from_list(coresX), dim-1)
                    leftQ = left(tt.tensor.from_list(coresX), dim)

                    first = np.tensordot(leftQm1.T, unfolding(zArr[idx], dim-1), 1)
                    second = reshape(first, (-1, np.prod(modeSize[dim+1:])))
                    if dim < numDims-1:
                        explicit = second.dot(rightQ)
                        orth_cc = reshape(coresX[dim], (-1, coresX[dim].shape[2]))
                        explicit -= orth_cc.dot(leftQ.T.dot(unfolding(zArr[idx], dim)).dot(rightQ))
                    else:
                        explicit = second
                    explicit = reshape(explicit, currPCore.shape)
                    assert(np.allclose(explicit, currPCore))

            if dim == 0:
                for value in range(modeSize[dim]):
                    coresP[dim][0:r1, value, r2:] = coresX[dim][:, value, :]
            else:
                for value in range(modeSize[dim]):
                    coresP[dim][r1:, value, r2:] = coresX[dim][:, value, :]

        if dim == numDims-1:
            for idx in range(len(zArr)):
                currZCore = reshape(coresZ[idx][dim], (zArr[idx].r[dim], -1))
                currPCore = np.dot(lhs[idx], currZCore)
                for value in range(modeSize[dim]):
                    coresP[dim][r1:, value, 0:r2] += reshape(currPCore[:, value], (r1, 1))

    if debug:
        assert(np.allclose(X.full(), tt.tensor.from_list(coresX).full()))
    return tt.tensor.from_list(coresP)


def projector_splitting_add(Y, delta, debug=False):
    """Compute Y + delta via projector splitting scheme.

    This function implements the projector splitting scheme (section 4.2 of [1]).

    The result is a TT-tensor with the TT-ranks equal to the TT-ranks of Y."""
    # Get rid of redundant ranks (they cause technical difficulties).
    delta = delta.round(eps=0)
    numDims = delta.d
    assert(numDims == Y.d)
    modeSize = delta.n
    assert(modeSize == Y.n).all()
    coresDelta = tt.tensor.to_list(delta)
    coresY = tt.tensor.to_list(Y)
    # rhs[dim] is an (delta.rank_dim * Y.rank_dim) x 1 vector
    rhs = [None] * (numDims+1)
    rhs[numDims] = np.ones([1, 1])
    # Right to left orthogonalization of Y and preparation of the rhs vectors.
    for dim in range(numDims-1, 0, -1):
        # Orthogonalization.
        cc = coresY[dim].copy()
        r1, n, r2 = cc.shape
        cc = reshape(cc, (r1, -1)).T
        cc, rr = np.linalg.qr(cc)
        r1 = cc.shape[1]
        # Warning: since ranks can change here, do not use Y.r!
        # Use coresY[dim].shape instead.
        coresY[dim] = reshape(cc.T, (r1, n, r2)).copy()
        if dim > 0:
            coresY[dim-1] = np.tensordot(coresY[dim-1], rr.T, 1)

        # rhs computation.
        coreProd = np.tensordot(coresDelta[dim], coresY[dim], axes=(1, 1))
        coreProd = np.transpose(coreProd, (0, 2, 1, 3))
        coreProd = reshape(coreProd, (delta.r[dim]*r1, delta.r[dim+1]*r2))
        rhs[dim] = np.dot(coreProd, rhs[dim+1])
    if debug:
        assert(np.allclose(Y.full(), tt.tensor.from_list(coresY).full()))

    # lsh is an Y.rank_dim x delta.rank_dim matrix.
    lhs = np.ones([1, 1])
    # s is an Y.rank_dim x Y.rank_dim matrix.
    s = np.ones([1, 1])
    # Left to right projector splitting sweep.
    for dim in range(numDims):
        # Y^+ (formula 4.10)
        cc = coresDelta[dim].copy()
        r1, n, r2 = coresY[dim].shape
        cc = np.tensordot(lhs, cc, 1)
        rhs[dim+1] = reshape(rhs[dim+1], (delta.r[dim+1], r2))
        cc = reshape(cc, (-1, delta.r[dim+1]))
        cc = np.dot(cc, rhs[dim+1])
        if debug:
            first = np.kron(np.eye(modeSize[dim]), left(tt.tensor.from_list(coresY), dim-1).T)
            second = np.dot(first, unfolding(delta, dim))
            explicit = np.dot(second, right(tt.tensor.from_list(coresY), dim+1))
            assert(np.allclose(explicit, cc))
        cc += reshape(np.tensordot(s, coresY[dim], 1), (-1, Y.r[dim+1]))
        if dim < numDims-1:
            cc, rr = np.linalg.qr(cc)
            # TODO: do we need to use r1 = cc.shape[1] here????
        cc = reshape(cc, coresY[dim].shape)
        coresY[dim] = cc.copy()

        if dim < numDims-1:
            coreProd = np.tensordot(coresY[dim], coresDelta[dim], axes=(1, 1))
            coreProd = np.transpose(coreProd, (0, 2, 1, 3))
            coreProd = reshape(coreProd, (r1*delta.r[dim], r2*delta.r[dim+1]))
            lhs = reshape(lhs, (1, -1))
            lhs = np.dot(lhs, coreProd)
            lhs = reshape(lhs, (r2, delta.r[dim+1]))

        if dim < numDims-1:
            # Y^- (fomula 4.7)
            s = rr - np.dot(lhs, rhs[dim+1])
            if debug:
                first = left(tt.tensor.from_list(coresY), dim).T
                second = np.dot(first, unfolding(delta, dim))
                explicit = np.dot(second, right(tt.tensor.from_list(coresY), dim+1))
                assert(np.allclose(explicit, np.dot(lhs, rhs[dim+1])))

    return tt.tensor.from_list(coresY)
