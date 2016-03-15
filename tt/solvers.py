import numpy as np
import tt

# TT-GMRES


def GMRES(A, u_0, b, eps=1E-6, restart=20, verb=0):
    """GMRES linear systems solver based on TT techniques.

    A = A(x[, eps]) is a function that multiplies x by matrix.
    """
    do_restart = True
    while do_restart:
        r0 = b + A((-1) * u_0)
        r0 = r0.round(eps)
        beta = r0.norm()
        bnorm = b.norm()
        curr_beta = beta
        if verb:
            print "/ Initial  residual  norm: %lf; mean rank:" % beta, r0.rmean()
        m = restart
        V = np.zeros(m + 1, dtype=object)  # Krylov basis
        V[0] = r0 * (1.0 / beta)
        H = np.mat(np.zeros((m + 1, m), dtype=np.complex128, order='F'))
        j = 0
        while j < m and curr_beta / bnorm > eps:
            delta = eps / (curr_beta / beta)
            # print "Calculating new Krylov vector"
            w = A(V[j], delta)
            #w = w.round(delta)
            for i in range(j + 1):
                H[i, j] = tt.dot(w, V[i])
                w = w + (-H[i, j]) * V[i]
            w = w.round(delta)
            if verb > 1:
                print "|% 3d. New Krylov vector mean rank:" % (j + 1), w.rmean()
            H[j + 1, j] = w.norm()
            V[j + 1] = w * (1 / H[j + 1, j])

            Hj = H[:j + 2, :j + 1]
            betae = np.zeros(j + 2, dtype=np.complex128)
            betae[0] = beta
            # solving Hj * y = beta e_1
            y, curr_beta, rank, s = np.linalg.lstsq(Hj, betae)
            curr_beta = curr_beta[0]
            if verb:
                print "|% 3d. LSTSQ residual norm:" % (j + 1), curr_beta
            j += 1
        x = u_0
        for i in range(j):
            x = x + V[i] * y[i]
        x = x.round(eps)
        if verb:
            print "\\ Solution mean rank:", x.rmean()
        u_0 = x
        do_restart = (curr_beta / bnorm > eps)
    return x
