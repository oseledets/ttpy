""" Dynamical TT-approximation """
import numpy as np
import dyn_tt
import tt

def ksl(A, y0, tau, verb=1, scheme='symm', space=8, rmax=2000):
    """ Dynamical tensor-train approximation based on projector splitting
        This function performs one step of dynamical tensor-train approximation
        for the equation

        .. math ::
            \\frac{dy}{dt} = A y, \\quad y(0) = y_0

        and outputs approximation for :math:`y(\\tau)`

    :References:


        1. Christian Lubich, Ivan Oseledets, and Bart Vandereycken.
        Time integration of tensor trains. arXiv preprint 1407.2042, 2014.

        http://arxiv.org/abs/1407.2042

        2. Christian Lubich and Ivan V. Oseledets. A projector-splitting integrator
        for dynamical low-rank approximation. BIT, 54(1):171-188, 2014.

        http://dx.doi.org/10.1007/s10543-013-0454-0

    :param A: Matrix in the TT-format
    :type A: matrix
    :param y0: Initial condition in the TT-format,
    :type y0: tensor
    :param tau: Timestep
    :type tau: float
    :param scheme: The integration scheme, possible values: 'symm' -- second order, 'first' -- first order
    :type scheme: str
    :param space: Maximal dimension of the Krylov space for the local EXPOKIT solver.
    :type space: int
    :rtype: tensor

    :Example:


        >>> import tt
        >>> import tt.ksl
        >>> import numpy as np
        >>> d = 8
        >>> a = tt.qlaplace_dd([d, d, d])
        >>> y0, ev = tt.eigb.eigb(a, tt.rand(2 , 24, 2), 1e-6, verb=0)
        Solving a block eigenvalue problem
        Looking for 1 eigenvalues with accuracy 1E-06
        swp: 1 er = 1.1408 rmax:2
        swp: 2 er = 190.01 rmax:2
        swp: 3 er = 2.72582E-08 rmax:2
        Total number of matvecs: 0
        >>> y1 = tt.ksl.ksl(a, y0, 1e-2)
        Solving a real-valued dynamical problem with tau=1E-02
        >>> print tt.dot(y1, y0) / (y1.norm() * y0.norm()) - 1 #Eigenvectors should not change
        0.0
    """

    ry = y0.r.copy()
    if scheme is 'symm':
        tp = 2
    else:
        tp = 1
    # Check for dtype
    y = tt.vector()
    if np.iscomplex(A.tt.core).any() or np.iscomplex(y0.core).any():
        dyn_tt.dyn_tt.ztt_ksl(
            y0.d,
            A.n,
            A.m,
            A.tt.r,
            A.tt.core + 0j,
            y0.core + 0j,
            ry,
            tau,
            rmax,
            0,
            10,
            verb,
            tp,
            space)
        y.core = dyn_tt.dyn_tt.zresult_core.copy()
    else:
        A.tt.core = np.real(A.tt.core)
        y0.core = np.real(y0.core)
        dyn_tt.dyn_tt.tt_ksl(
            y0.d,
            A.n,
            A.m,
            A.tt.r,
            A.tt.core,
            y0.core,
            ry,
            tau,
            rmax,
            0,
            10,
            verb,
            tp,
            space
            )
        y.core = dyn_tt.dyn_tt.dresult_core.copy()
    dyn_tt.dyn_tt.deallocate_result()
    y.d = y0.d
    y.n = A.n.copy()
    y.r = ry
    y.get_ps()
    return y


def diag_ksl(A, y0, tau, verb=1, scheme='symm', space=8, rmax=2000):
    """ Dynamical tensor-train approximation based on projector splitting
        This function performs one step of dynamical tensor-train approximation with diagonal matrix, i.e. it solves the equation
        for the equation

        .. math ::
            \\frac{dy}{dt} = V y, \\quad y(0) = y_0

        and outputs approximation for :math:`y(\\tau)`

    :References:


        1. Christian Lubich, Ivan Oseledets, and Bart Vandereycken.
        Time integration of tensor trains. arXiv preprint 1407.2042, 2014.

        http://arxiv.org/abs/1407.2042

        2. Christian Lubich and Ivan V. Oseledets. A projector-splitting integrator
        for dynamical low-rank approximation. BIT, 54(1):171-188, 2014.

        http://dx.doi.org/10.1007/s10543-013-0454-0

    :param A: Matrix in the TT-format
    :type A: matrix
    :param y0: Initial condition in the TT-format,
    :type y0: tensor
    :param tau: Timestep
    :type tau: float
    :param scheme: The integration scheme, possible values: 'symm' -- second order, 'first' -- first order
    :type scheme: str
    :param space: Maximal dimension of the Krylov space for the local EXPOKIT solver.
    :type space: int
    :rtype: tensor

    :Example:


        >>> import tt
        >>> import tt.ksl
        >>> import numpy as np
        >>> d = 8
        >>> a = tt.qlaplace_dd([d, d, d])
        >>> y0, ev = tt.eigb.eigb(a, tt.rand(2 , 24, 2), 1e-6, verb=0)
        Solving a block eigenvalue problem
        Looking for 1 eigenvalues with accuracy 1E-06
        swp: 1 er = 1.1408 rmax:2
        swp: 2 er = 190.01 rmax:2
        swp: 3 er = 2.72582E-08 rmax:2
        Total number of matvecs: 0
        >>> y1 = tt.ksl.ksl(a, y0, 1e-2)
        Solving a real-valued dynamical problem with tau=1E-02
        >>> print tt.dot(y1, y0) / (y1.norm() * y0.norm()) - 1 #Eigenvectors should not change
        0.0
    """
    ry = y0.r.copy()
    if scheme is 'symm':
        tp = 2
    else:
        tp = 1
    # Check for dtype
    y = tt.vector()
    if np.iscomplex(A.core).any() or np.iscomplex(y0.core).any():
        dyn_tt.dyn_diag_tt.ztt_diag_ksl(
            y0.d,
            A.n,
            A.r,
            A.core + 0j,
            y0.core + 0j,
            ry,
            tau,
            rmax,
            0,
            10,
            verb,
            tp,
            space)
        y.core = dyn_tt.dyn_diag_tt.zresult_core.copy()
    else:
        A.core = np.real(A.core)
        y0.core = np.real(y0.core)
        dyn_tt.dyn_diag_tt.dtt_diag_ksl(
            y0.d,
            A.n,
            A.r,
            A.core,
            y0.core,
            ry,
            tau,
            rmax,
            0,
            10,
            verb,
            tp,
            space)
        y.core = dyn_tt.dyn_diag_tt.dresult_core.copy()
    dyn_tt.dyn_diag_tt.deallocate_result()
    y.d = y0.d
    y.n = A.n.copy()
    y.r = ry
    y.get_ps()
    return y
