import tt
from tt.amr import amr_solve, mvk4
""" This program test two subroutines: matrix-by-vector multiplication
    and linear system solution via AMR scheme"""

d = 12
A = tt.qlaplace_dd([d])
x = tt.ones(2,d)
y = mvk4(A,x,x,1e-6)
y = amr_solve(A,x,x,1e-6)
