from tt_tensor2 import *
from amr import *
A = tt_qlaplace_dd([5])
x = tt_ones(2,5)
y = mvk4(A,x,x,1e-6)

