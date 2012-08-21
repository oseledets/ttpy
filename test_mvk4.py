from tt_tensor2 import *
import mvk4
import time


d = 80
r = 50
rmax = 100
eps = 1e-6
b = tt_rand(2,d,r)
#M = tt_qlaplace_dd(d) 
M = tt_rand(4,d,3)
M = tt_matrix(M)
M = M+M+M+M+M
a = tt_rand(2,d,r);
b = tt_rand(2,d,1)
rnew=b.r.copy()
t1=time.time()
mvk4.tt_adapt_als.tt_mvk4(M.n,M.m,a.r,M.tt.r,M.tt.core,a.core,b.core,rnew,1e-8,150,verb=2,nswp=20,kickrank=5)
res = tt_tensor()
res.n = M.n.copy()
res.r = rnew.copy()
res.d = a.d
res.core = mvk4.tt_adapt_als.result_core.copy()
res.get_ps()
t2=time.time()
print('Total time is %.1f sec' % (t2-t1))
#new_core=mvk4.tt_adapt_als.result.copy()

