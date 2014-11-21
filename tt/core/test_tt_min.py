import math
import numpy as np
from math import pi
from tt_min import tt_min
from scipy.optimize import rosen
import time
import itertools

#fun = lambda x, y: x ** 2 + y ** 2
#tt_min(fun, -2, 2, d = 2)
#fun = lambda x : x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2
class boxfun(object):
    a = None
    b = None
    fun = None
    name = None
    true_min = None
    d = None
    def __init__(self, name, fun, a, b, true_min = None, d = None):
        self.a = a
        self.b = b
        self.d = d
        self.fun = fun
        self.true_min = true_min
        self.name = name

def reshape(a, sz):
    return np.reshape(a, sz, order='F')
def my_rosen(x):
    return rosen(x.T)

A = 10

def rastrigin(x):
    d = x.T.shape[0]
    y =  A * d + np.sum(x.T ** 2 - A * np.cos(pi * x.T), axis=0)
    return y

def michalewicz(x):
    m = 10
    d = x.T.shape[0]
    y1 = np.arange(d) + 1  
    y1 = np.sin((y1 * (x ** 2) / math.pi))
    y = -np.sum(np.sin(x.T) * ((y1.T) ** (2 * m)), axis=0)
    return y / d

#s = 0.05
#g1 = 20
#g2 = 10
#def pinters(x):
#    p2 = np.sum(x.T - 12, axis = 0)
#    q = np.sum((x.T - 12) ** 2, axis = 0)
#    p1 = p2 + q
#    y = s * q + np.sin(g1 * p1) ** 2 + np.sin(g2 * p2) ** 2
#    return y

def schwefel(x):
    return -np.sum(x.T * np.sin(np.sqrt(np.abs(x.T))), axis=0) / (500 * d)

def grienwank(x):
    d = x.T.shape[0]
    y1 = np.arange(d) + 1
    y1 = np.cos(1.0 / np.sqrt(y1) * x)
    return 1.0/4000 * np.sum(x.T ** 2, axis=0) - np.prod(y1.T) + 1

def ackley(x):
    a = 20
    b = 0.2
    c = 2 * math.pi
    d = x.T.shape[0]
    q1 = np.sqrt(np.sum(x.T ** 2, axis=0) / d)
    q2 = np.sum(np.cos(c * x.T), axis=0) / d
    
    return -a * np.exp(-b * q1) - np.exp(q2) + a + np.exp(1.0)

def shubert(x):
    d = 2
    x0 = reshape(x, (-1, d))
    x1 = x0[:, 0]
    x2 = x0[:, 1]
    y1 = np.arange(5) + 1
    f = np.zeros(x1.shape)
    for j in xrange(len(x1)):
        a = np.sum(y1 * np.cos((y1 + 1) * x1[j] + 1))
        b = np.sum(y1 * np.cos((y1 + 1) * x2[j] + 1))
        f[j] = a * b
    return f
    
schwefel_fun = boxfun('Schwefel', schwefel, -500, 500, 
                      true_min = -418.9829 / 500)
grienwank_fun = boxfun('Grienwank', grienwank, -600, 600, 
                      true_min = 0)
ackley_fun = boxfun('Ackley', ackley, -32.768, 32.768, 
                      true_min = 0)
michalewicz_fun = boxfun('Michalewicz', michalewicz, 0, math.pi, 
                      true_min = -0.9374 )

shubert_fun = boxfun('Shubert', shubert, -5.12, 5.12, d = 2,
                     true_min = -186.7309)
rastrigin_fun = boxfun('Rastrigin', rastrigin, -5.12, 5.12, 
                       true_min = 0)

rosenbrock_fun = boxfun('Rosenbrock', my_rosen, -2.048, 2.048, true_min = 0)

class smf(object):
    fun = None
    name = None
    def __init__(self, fun=None, name=None):
        self.name = name
        self.fun = fun


smooth_fun1 = smf(fun=lambda p, lam: (math.pi/2 - np.arctan(p - lam)), name='Arctan with shift')
smooth_fun2 = smf(fun=lambda p, lam: (math.pi/2 - np.arctan(p)), name='Arctan with no shift')
smooth_fun3 = smf(fun=lambda p, lam: np.exp(-(p - lam)), name='Exp with shift')
smooth_fun4 = smf(fun=lambda p, lam: np.exp(-p), name='Exp without shift')

#smooth_all = [smooth_fun1, smooth_fun2]
#dall = [2, 3]
smooth_all = [smooth_fun1, smooth_fun2, smooth_fun3, smooth_fun4]
fun_all = [rosenbrock_fun, ackley_fun, rastrigin_fun, grienwank_fun,schwefel_fun,michalewicz_fun]
dall = [5, 10, 20]
res_all = {}
pall = 50 #Number of iterations
q = np.load('test_f.npz')
res_all = q['res_all']
res_all = res_all.item()
for curfun, d, smooth_fun in itertools.product(fun_all, dall, smooth_all):
    print("Doing minimization of %s function, d = %d, smooth_fun = %s" % (curfun.name, d, smooth_fun.name))
    if curfun.d is not None:
        d = curfun.d
    rc_all = np.zeros(pall)
    xall = {}
    t = time.time()
    if (curfun.name, d, smooth_fun.name) not in res_all.keys():
        for i in xrange(pall):
            val, x = tt_min(curfun.fun, curfun.a, curfun.b, d = d,
                            n0 = 512, rmax = 10, nswp = 20, radd = 7, smooth_fun=smooth_fun.fun)
            rc_all[i] = val
            xall[i] = x
            #print('%d/%d' % (i, pall))
        t = time.time() - t
        res = {"points" : xall, "values" : rc_all, "true_min" : curfun.true_min, "time" : t / pall}
        res_all[(curfun.name, d, smooth_fun.name)] = res
        print 'Saving  a data point'
        np.savez("test_all_results_improved", res_all=res_all)
np.savez("test_all_results_improved", res_all=res_all) #Just save everything in one big dictionary

    
        
            
