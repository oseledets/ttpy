
import tt
import tt.eigb
import tt.ksl
import ksl_deim
import numpy as np
d = 8
a = tt.qlaplace_dd([d, d, d])
y0, ev = tt.eigb.eigb(a, tt.rand(2 , 24, 2), 1e-6, verb=0)
        # Solving a block eigenvalue problem
        # Looking for 1 eigenvalues with accuracy 1E-06
        # swp: 1 er = 1.1408 rmax:2
        # swp: 2 er = 190.01 rmax:2
        # swp: 3 er = 2.72582E-08 rmax:2
        # Total number of matvecs: 0
y_ksl = tt.ksl.ksl(a, y0, 1e-2)

def Nf(x):
    return 0

y_ksl_deim = ksl_deim.ksl_deim(a, Nf, y0, 1e-2)
        # Solving a real-valued dynamical problem with tau=1E-02
print(tt.dot(y_ksl, y0) / (y_ksl.norm() * y0.norm()) - 1) # Eigenvectors should not change
        # 0.0
print(tt.dot(y_ksl_deim, y0) / (y_ksl_deim.norm() * y0.norm()) - 1) # Eigenvectors should not change

print('Difference between KSL and KSL_DEIM:')
diff = y_ksl-y_ksl_deim
print(diff.norm())