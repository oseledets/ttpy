import tt
from tt.amr import *
d = 12
A = tt.qlaplace_dd([d])
x = tt.ones(2,d)
y = mvk4(A,x,x,1e-6)
y = amr_solve(A,x,x,1e-6)
