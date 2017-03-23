from __future__ import print_function, absolute_import, division
import sys
sys.path.append('../')
import numpy as np
import tt
from tt.eigb import *
import time

""" This code computes many eigenvalus of the Laplacian operator """

d = 8
f = 8
A = tt.qlaplace_dd([d]*f)
#A = (-1)*A
#A = tt.eye(2,d)
n = [2] *(d * f)
r = [8] *(d * f + 1)
r[0] = 1
r[d * f] = 8 #Number of eigenvalues sought
x = tt.rand(n, d * f, r)
#x = tt_ones(2,d)
t = time.time()
y, lam = eigb(A, x, 1e-6)

t1 = time.time()
print('Eigenvalues:', lam)
print('Time is:', t1-t)
