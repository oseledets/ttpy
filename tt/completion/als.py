from __future__ import print_function
try:
    xrange
except NameError:
    xrange = range
import numpy as np
import tt
import time
import copy

def reshape(x, shape):
    '''
    Reshape given numpy.array into new shape with Fortran-ordering
 
    Parameters:
        :np.array: x
            given numpy array
        :list, tuple, np.array: shape
            new shape
    '''
    return np.reshape(x, shape, order = 'F')

def getRow(leftU, rightV, jVec):
    '''
    Compute X_{\geq \mu}^T \otimes X_{leq \mu}
    X_{\geq \mu} = V_{\mu+1}(j_{\mu}) \ldots V_{d} (j_{d}) [left interface matrix]
    X_{\leq \mu} = U_{1} (j_{1}) \ldots U_{\mu-1}(j_{\mu-1}) [right interface matrix]
    
    Parameters:
        :list of numpy.arrays: leftU
            left-orthogonal cores from 1 to \mu-1
        :list of numpy.arrays: rightV
            right-orthogonal cores from \mu+1 to d
        :list, tuple, np.array: jVec
            indices for each dimension n[k]
    Returns:
        :numpy.array: result
            Kronecker product between left and right interface
            matrices. Left matrix is transposed.
    '''
    jLeft = None
    jRight = None
    if len(leftU) > 0:
        jLeft = jVec[:len(leftU)]
    if len(rightV) > 0:
        jRight = jVec[-len(rightV):]
    
    multU = np.ones([1,1])
    for k in xrange(len(leftU)):
        multU = np.dot(multU, leftU[k][:, jLeft[k], :])
    multV= np.ones([1,1])
    for k in xrange(len(rightV)-1, -1, -1):
        multV = np.dot(rightV[k][:, jRight[k], :], multV)
    
    result = np.kron(multV.T, multU)
    return result
    
def orthLRFull(coreList, mu, splitResult = True):
    '''
    Orthogonalize list of TT-cores.
    
    Parameters:
        :list: coreList
            list of TT-cores (stored as numpy arrays)
        :int: mu
            separating index for left and right orthogonalization.
            Output cores will be left-orthogonal for dimensions from 1 to \mu-1
            and right-orthogonal for dimensions from \mu+1 to d
        :boolean: splitResult = True
            Controls whether outut should be splitted into left-, non-, right-orthogonal
            parts or not.
    
    Returns:
        :list: resultU
            left-orthogonal cores with indices from 1 to \mu-1
        :np.array: W
            \mu-th core
        :list: reultV
            right-orthogonal cores with indices from \mu+1 to d
    OR
        :list: resultU + [W] + resultV
            concatenated list of cores
    '''
    d = len(coreList)
    assert (mu >= 0) and (mu <= d)
    resultU = []
    for k in xrange(mu):
        core = coreList[k].copy()
        if k > 0:
            core = np.einsum('ijk,li->ljk', core, R)
        [r1, n, r2] = core.shape
        if (k < mu-1):
            core = reshape(core, [r1*n, r2])
            Q, R = np.linalg.qr(core)
            rnew = Q.shape[1]
            core = reshape(Q, [r1, n, rnew])
            resultU = resultU + [core]
    if mu > 0:
        W = core.copy()
    resultV = []
    for k in xrange(d-1, mu, -1):
        core = coreList[k].copy()
        if (k < d-1):
            core = np.einsum('ijk,lk->ijl', core, R)
        [r1, n, r2] = core.shape
        if (k > mu+1):
            core = reshape(core, [r1, n*r2])
            Q, R = np.linalg.qr(core.T)
            rnew = Q.shape[1]
            core = reshape(Q.T, [rnew, n, r2])
        resultV = [core] + resultV
    if mu < d-1:
        if mu > 0:
            W = np.einsum('ijk,lk->ijl', W, R)
        else:
            W = np.einsum('ijk,lk->ijl', coreList[0], R)
    if splitResult:
        return resultU, W, resultV
    return resultU + [W] + resultV

def computeFunctional(x, cooP):
    '''
    Compute value of functional J(X) = ||PX - PA||^2_F,
    where P is projector into index subspace of known elements,
          X is our approximation,
          A is original tensor.
          
    Parameters:
        :tt.vector: x
            current approximation [X]
        :dict: cooP
            dictionary with two records
                - 'indices': numpy.array of P x d shape,
                contains index subspace of P known elements;
                each string is an index of one element.
                - 'values': numpy array of size P,
                contains P known values.
    
    Returns:
        :float: result
            value of functional
    '''
    indices = cooP['indices']
    values = cooP['values']
    
    [P, d] = indices.shape
    assert P == len(values)
    
    result = 0
    for p in xrange(P):
        index = tuple(indices[p, :])
        result += (x[index] - values[p])**2
    result *= 0.5
    return result


def ttSparseALS(cooP, shape, x0=None, ttRank=1, tol=1e-5, maxnsweeps=20, verbose=True, alpha=1e-2):
    '''
    TT completion via Alternating Least Squares algorithm.
    
    Parameters:
        :dict: cooP
            dictionary with two records
                - 'indices': numpy.array of P x d shape,
                contains index subspace of P known elements;
                each string is an index of one element.
                - 'values': numpy array of size P,
                contains P known values.   
        :list, numpy.array: shape
            full-format shape of tensor to be completed [dimensions]
        :tt.vector: x0 = None
            initial approximation of completed tensor
            If it is specified, parameters 'shape' and 'ttRank' will be ignored
        :int, numpy.array: ttRank = 1
            assumed rank of completed tensor
        :float: tol = 1e-5
            tolerance for functional value
        :int: maxnsweeps = 20
            maximal number of sweeps [sequential optimization of all d cores
            in right or left direction]
        :boolean: verbose = True
            switcher of messages from function
        :float: alpha: = 1e-2
            regularizer of least squares problem for each slice of current TT core.
            [rcond parameter for np.linalg.lstsq]
            
    Returns:
        :tt.vector: xNew
            completed TT vector
        :list: fit
            list of functional values at each sweep
    '''
    indices = cooP['indices']
    values = cooP['values']
    
    [P, d] = indices.shape
    assert P == len(values)
    
    timeVal = time.clock()
    if x0 is None:
        x = tt.rand(shape, r = ttRank)
        x = x.round(0.)
        x = (1./x.norm())*x
    else:
        x = copy.deepcopy(x0)
    assert d == x.d
    # TODO: also check if cooP indices are aligned with shape
    normP = np.linalg.norm(values)
    values /= normP
    fitList = []
    sweepTimeList = []
    initTime = time.clock() - timeVal
    
    timeVal = time.clock()
    coreList = tt.vector.to_list(x)
    #coreList = orthLRFull(coreList, mu = d, splitResult = False)
    # orthTime = time.clock() - timeVal
    
    if verbose:
        print("Initialization time: %.3f seconds (proc.time)" % (initTime))
        # print "Orthogonalizing time: %.3f seconds (proc.time)" % (orthTime)
    
    for sweep in xrange(maxnsweeps):
        sweepStart = time.clock()
        # list left + right
        [kStart, kEnd, kStep] = [0, d, 1]
        # select direction of sweep
        '''
        if sweep % 2 == 0: # left to rigth
            [kStart, kEnd, kStep] = [0, d, 1]
        else: # right to left
            [kStart, kEnd, kStep] = [d-1, -1, -1]
        '''
        # fix k-th core to update
        for k in xrange(kStart, kEnd, kStep):
            [r1, n, r2] = coreList[k].shape
            core = np.zeros([r1, n, r2])
            leftU = []
            rightV = []
            if k > 0:
                leftU = coreList[:k]
            if k < d-1:
                rightV = coreList[k+1:] 
            for i in xrange(n):
                thetaI = np.where(indices[:, k] == i)[0]
                if len(thetaI) > 0:
                    A = np.zeros([len(thetaI), r1*r2])
                    for j in xrange(len(thetaI)):
                        tmp = getRow(leftU, rightV, indices[thetaI[j], :])
                        A[j:j+1, :] += tmp   # .flatten(order = 'F')
                    vecCoreSlice, _, _, _ = np.linalg.lstsq(A, values[thetaI])#, rcond = alpha)
                    # 0.5*np.linalg.norm(np.dot(A, vecCoreSlice) - values[thetaI])**2.
                    core[:, i, :] += reshape(vecCoreSlice, [r1, r2]) ####
            '''
            if k < (d-1):
                core = reshape(core, [r1*n, r2])
                Q, R = np.linalg.qr(core)
                rnew = Q.shape[1]
                core = reshape(Q, [r1, n, rnew])
                coreList[k+1] = np.einsum('ijk,li->ljk', coreList[k+1], R)
            '''
            coreList[k] = core.copy()
            '''
            else:
                if (k > 0):
                    core = reshape(core, [r1, n*r2])
                    Q, R = np.linalg.qr(core.T)
                    rnew = Q.shape[1]
                    core = reshape(Q.T, [rnew, n, r2])
                    coreList[k-1] = np.einsum('ijk,lk->ijl', coreList[k-1], R)
            '''
            
        xNew = tt.vector.from_list(coreList)
        fit = computeFunctional(xNew, cooP)
        fitList.append(fit)
        if fit < tol:
            break
        if sweep > 0:
            if abs(fit - fitList[-2]) < tol:
                break
        sweepTimeList.append(time.clock() - sweepStart)
        if verbose:
            print("sweep %d/%d\t fit value: %.5e\t time: %.3f seconds (proc.time)" % (sweep+1, maxnsweeps, fit, sweepTimeList[-1]))
    if verbose:
        print("Total sweep time: %.3f seconds (proc.time)\t Total time: %.3f seconds (proc.time)" % (sum(sweepTimeList), sum(sweepTimeList) + initTime))# + orthTime)
    info = {'fit': fitList, 'initTime': initTime,  'sweepTime': sweepTimeList} # 'orthTime': orthTime,
    xNew *= normP
    values *= normP
    
    return xNew, info
