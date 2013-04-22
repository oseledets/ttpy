import amen_f90
import tt

def amen_solve(A, f, x0, eps, kickrank=4, nswp=20, local_prec='n', local_iters=2, local_restart=40, trunc_norm=1, max_full_size=50, verb=1):
    """ Approximate linear system solution 
            X = amen_solve(A,F,X0,EPS), using AMR/DMRG algorithm.
        
        :param A: Coefficients matrix, QTT-decomposed.
        :type A: matrix
        :param f: Right-hand side, TT-decomposed.
        :type f: tensor
        :param x0: TT-tensor of initial guess.
        :type x0: tensor
        :param eps: Accuracy.
        :type eps: float
    """
    m = A.m.copy()
    rx0 = x0.r.copy()
    psx0 = x0.ps.copy()
    if A.is_complex or f.is_complex:
        amen_f90.amen_f90.ztt_amen_wrapper(f.d, A.n, m,               \
                                           A.tt.r, A.tt.ps, A.tt.core, \
                                           f.r, f.ps, f.core,           \
                                           rx0, psx0, x0.core,           \
                                           eps, kickrank, nswp, local_iters, local_restart, trunc_norm, max_full_size, verb, local_prec)
    else:
        if x0.is_complex:
            x0 = x0.real()
            rx0 = x0.r.copy()
            psx0 = x0.ps.copy()
        amen_f90.amen_f90.dtt_amen_wrapper(f.d, A.n, m,               \
                                           A.tt.r, A.tt.ps, A.tt.core, \
                                           f.r, f.ps, f.core,           \
                                           rx0, psx0, x0.core,           \
                                           eps, kickrank, nswp, local_iters, local_restart, trunc_norm, max_full_size, verb, local_prec)
    x = tt.tensor()
    x.d = f.d
    x.n = m.copy()
    x.r = rx0
    if A.is_complex or f.is_complex:
        x.core = amen_f90.amen_f90.zcore.copy()
    else:
        x.core = amen_f90.amen_f90.core.copy()
    amen_f90.amen_f90.deallocate_result()
    x.get_ps()
    return x

