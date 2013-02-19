import numpy as np
from math import pi,sqrt
import tt
from tt.ksl import ksl
import time
from scipy.linalg import expm
d = 6
f = 1
A = tt.qlaplace_dd([d]*f)*((2**d+1))**2
#A = tt.eye(2,d)
n = [2]*(d*f)
n0 = 2**d
t = np.arange(1,n0+1)*1.0/(n0+1)
x = np.exp(-20*(t-0.5)**2)
#x = np.sin(sqrt(2.0)*pi*t)
x = tt.tensor(x.reshape([2]*d,order='F'),1e-2)
#x = tt.ones(2,d)
#x = x + 0 * tt.rand(2,d,2)
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
#tau_ref = 0.0
line1, = ax.plot(np.real(y.full().flatten("F")))
ax.hold("True")
plt.ylim([-1,1])
#The solution is the exp(i*lam*t) * sin(pi*x)
x0 = x.full().flatten("F")
line2, = ax.plot(np.real(x0))
B = expm(tau_ref*1j*A.full())

for i in xrange(2**ns_fin):  
    y1=ksl(1j*A,y,tau_ref)
    x0 = np.dot(B,x0)
    #print y1.full().flatten("F")/y.full().flatten("F")
    y = y1.copy()
    line1.set_ydata(np.real(y.full().flatten("F")))
    line2.set_ydata(np.real(x0))
    fig.canvas.draw()
    raw_input()
