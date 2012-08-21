import numpy as np
import tensor_util as tu
permute=tu.tensor_util_lib.permute
w=np.arange(128)
i1=np.arange(7,dtype=np.int32)+1
sz=2*np.ones((7,),dtype=np.int32)
permute(w,sz,i1)

