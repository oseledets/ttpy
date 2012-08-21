import numpy as np
import tt_tensor2 as tt
from tt_tensor2 import tt_tensor, tt_matrix #These two lines seems
#I think we could just rename tt_tensor -> tensor, tt_matrix -> matrix and that would be perfect

q = np.load("test.npz")
c = tt_tensor()
c.d = q["d"]
c.r = q["r"]
c.n = q["n"]
c.ps = q["ps"]
c.core = q["core"]
c1 = tt_tensor()
c1.d = q["sd"]
c1.ps = q["sps"]
c1.r = q["sr"]
c1.n = q["sn"]
c1.core = q["score"]
print c,c1
print tt.dot(c,c1)
