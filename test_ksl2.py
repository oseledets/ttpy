import numpy as np
from math import pi,sqrt
import tt
from tt.ksl import ksl
import time
from scipy.linalg import expm
d = 6
f = 1
#A = tt.qlaplace_dd([d]*f)
A = tt.eye(2,d)
n = [2]*(d*f)
n0 = 2**d
t = np.arange(1,n0+1)*1.0/(n0+1)
#x = np.exp(-10*(t-0.5)**2)
x = np.sin(pi*t)
x = tt.tensor(x.reshape([2]*d,order='F'),1e-8)
x = tt.ones(2,d)
x = x + 0 * tt.rand(2,d,1)
tau = 1
tau1 = 1
y = x.copy()
ns_fin = 8
tau0 = 1.0
tau_ref = tau0/2**ns_fin
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(np.real(y.full().flatten("F")))
for i in xrange(2**ns_fin):  
    y=ksl(1j*A,y,tau_ref)
    line1.set_ydata(np.real(y.full().flatten("F")))
    fig.canvas.draw()
