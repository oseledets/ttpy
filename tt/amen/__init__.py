import amen_f90
import tt
from amen_mv import amen_mv


def amen_solve(A, f, x0, eps, kickrank=4, nswp=20, local_prec='n',
               local_iters=2, local_restart=40, trunc_norm=1, max_full_size=50, verb=1):
    """ Approximate linear system solution in the tensor-train (TT) format
        using Alternating minimal energy (AMEN approach)

    :References: Sergey Dolgov, Dmitry. Savostyanov

                 Paper 1: http://arxiv.org/abs/1301.6068

                 Paper 2: http://arxiv.org/abs/1304.1222

    :param A: Matrix in the TT-format
    :type A: matrix
    :param f: Right-hand side in the TT-format
    :type f: tensor
    :param x0: TT-tensor of initial guess.
    :type x0: tensor
    :param eps: Accuracy.
    :type eps: float

    :Example:

        >>> import tt
        >>> import tt.amen #Needed, not imported automatically
        >>> a = tt.qlaplace_dd([8, 8, 8]) #3D-Laplacian
        >>> rhs = tt.ones(2, 3 * 8) #Right-hand side of all ones
        >>> x = tt.amen.amen_solve(a, rhs, rhs, 1e-8)
        amen_solve: swp=1, max_dx= 9.766E-01, max_res= 3.269E+00, max_rank=5
        amen_solve: swp=2, max_dx= 4.293E-01, max_res= 8.335E+00, max_rank=9
        amen_solve: swp=3, max_dx= 1.135E-01, max_res= 5.341E+00, max_rank=13
        amen_solve: swp=4, max_dx= 9.032E-03, max_res= 5.908E-01, max_rank=17
        amen_solve: swp=5, max_dx= 9.500E-04, max_res= 7.636E-02, max_rank=21
        amen_solve: swp=6, max_dx= 4.002E-05, max_res= 5.573E-03, max_rank=25
        amen_solve: swp=7, max_dx= 4.949E-06, max_res= 8.418E-04, max_rank=29
        amen_solve: swp=8, max_dx= 9.618E-07, max_res= 2.599E-04, max_rank=33
        amen_solve: swp=9, max_dx= 2.792E-07, max_res= 6.336E-05, max_rank=37
        amen_solve: swp=10, max_dx= 4.730E-08, max_res= 1.663E-05, max_rank=41
        amen_solve: swp=11, max_dx= 1.508E-08, max_res= 5.463E-06, max_rank=45
        amen_solve: swp=12, max_dx= 3.771E-09, max_res= 1.847E-06, max_rank=49
        amen_solve: swp=13, max_dx= 7.797E-10, max_res= 6.203E-07, max_rank=53
        amen_solve: swp=14, max_dx= 1.747E-10, max_res= 2.058E-07, max_rank=57
        amen_solve: swp=15, max_dx= 8.150E-11, max_res= 8.555E-08, max_rank=61
        amen_solve: swp=16, max_dx= 2.399E-11, max_res= 4.215E-08, max_rank=65
        amen_solve: swp=17, max_dx= 7.871E-12, max_res= 1.341E-08, max_rank=69
        amen_solve: swp=18, max_dx= 3.053E-12, max_res= 6.982E-09, max_rank=73
        >>> print (tt.matvec(a, x) - rhs).norm() / rhs.norm()
        5.5152374305127345e-09
    """
    m = A.m.copy()
    rx0 = x0.r.copy()
    psx0 = x0.ps.copy()
    if A.is_complex or f.is_complex:
        amen_f90.amen_f90.ztt_amen_wrapper(f.d, A.n, m,
                                           A.tt.r, A.tt.ps, A.tt.core,
                                           f.r, f.ps, f.core,
                                           rx0, psx0, x0.core,
                                           eps, kickrank, nswp, local_iters, local_restart, trunc_norm, max_full_size, verb, local_prec)
    else:
        if x0.is_complex:
            x0 = x0.real()
            rx0 = x0.r.copy()
            psx0 = x0.ps.copy()
        amen_f90.amen_f90.dtt_amen_wrapper(f.d, A.n, m,
                                           A.tt.r, A.tt.ps, A.tt.core,
                                           f.r, f.ps, f.core,
                                           rx0, psx0, x0.core,
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
