import tt
import tt.eigb
import tt.ksl
import ksl_deim

d = 8
a = tt.qlaplace_dd([d, d, d])
y0, ev = tt.eigb.eigb(a, tt.rand(2 , 24, 2), 1e-6, verb=0) # solve eigenvalue problem

# define a point-wise (nonlinear) function
def Nf(y):
    return 0

# integrate with ksl and ksl_deim
Nsteps = 10
y_ksl = y0
y_ksl_deim = y0
for i in range(Nsteps):
        y_ksl = tt.ksl.ksl(a, y_ksl, 1e-2)
        y_ksl_deim = ksl_deim.ksl_deim(a, Nf, y_ksl_deim, 1e-2)

# Eigenvectors should not change
# print('Change in eigenvector from KSL:')
# print(tt.dot(y_ksl, y0) / (y_ksl.norm() * y0.norm()) - 1)
print('Change in eigenvector from KSL_DEIM:')
print(tt.dot(y_ksl_deim, y0) / (y_ksl_deim.norm() * y0.norm()) - 1) 

print('Difference between KSL and KSL_DEIM:')
print((y_ksl-y_ksl_deim).norm())