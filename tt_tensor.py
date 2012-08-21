import numpy as np
import copy
from numpy.linalg import svd,norm,qr
from numpy import prod, reshape, nonzero, size, dot, diag
from math import sqrt
#This is surely required
class tt_tensor:
   """The basic class"""
   def __init__(self,a=None,eps=None): 
     if a is None:
          self.core=0
          self.d=0
          self.n=0
          self.r=0
          return
     self.d=a.ndim
     self.n=a.shape                  
     self.core=np.empty(self.d,dtype=object)
     self.r=np.zeros((self.d+1,1),dtype=np.int64)
     #self.r = [0]*(self.d+1)
     #Simple approximation scheme
     n=self.n
     r=self.r
     d=self.d
     core=self.core
     r[0]=1 #To fix later for the "matrix case"
     
     b=np.asfortranarray(a);
     for i in range(0,d-1):
     #Compute the SVD
       m=r[i]*n[i];  b=np.asanyarray(b,order='F')
       b.resize((m,size(b)/m))#b=b.reshape((m,size(b)/m),order='F')
       
       u,s,v = svd(b,full_matrices=False)
       #Chop the s 
       r1=my_chop2(s,eps*norm(s)/sqrt(d-1))
       #r1=2
       u=u[:,0:r1]
       s=s[0:r1]
       v=v[0:r1,:]
       #appr=dot(dot(u,diag(s)),v)
       #print norm(appr-b)/norm(b)
       r[i+1]=r1
       u.resize((r[i],n[i],r[i+1]))
       core[i]=u
       #Dot in numpy was written by idiots - who uses C ordering?
       b=dot(diag(s),v)
    #end for
     r[d]=1
     b.resize(r[d-1],n[d-1],r[d])
     core[d-1]=b
   #Print statement       
   def __repr__(self):
     str="This is a %d-dimensional tt_tensor \n" % self.d
     r=self.r
     d=self.d
     n=self.n
     for i in range(0,d):
         str=str+("r(%d)=%d, n(%d)=%d \n" % (i, r[i],i,n[i]))
     str=str+("r(%d)=%d \n" % (d,r[d]))
     return str
   
   def __getitem__(self,ind):       #Get a specific element
      core=self.core
      r=self.r
      d=self.d
      n=self.n
      v=1
      for i in range (0,d):
        cur_core=core[i]
        cur_core=cur_core[:,ind[i],:]
        cur_core=cur_core.reshape(r[i],r[i+1])
        v=dot(v,cur_core)
      if size(v) == 1:
         v=v[0,0]
      return v
   #@profile
   def __add__(self,other): #Add
      n=self.n
      d=self.d
      core1=self.core
      core2=other.core
      r1=self.r
      r2=other.r
      c=tt_tensor()
      r=r1+r2
      if ( r1[0] == r2[0] ):
         r[0]=r1[0]
      else:
         print('Error in the size of mode 0!\n')
      if ( r1[d] == r2[d]): 
         r[d]=r1[d]
      else:
         print('Error in the size of the last mode!\n')
      core=np.empty(d,dtype=object)
      c.core=core
      c.d=d
      c.n=n
      c.r=r
      for i in range(0,d):
         cr=np.zeros((r[i],n[i],r[i+1]))
         cr[0:r1[i],:,0:r1[i+1]]=core1[i]
         cr[r[i]-r2[i]:r[i],:,r[i+1]-r2[i+1]:r[i+1]]=core2[i]
         core[i]=cr
      
      return c
   def __mul__(self,other): #Multiplication
      if isinstance(other,(float,int)):
         c=tt_tensor()
         c.core=copy.deepcopy(self.core)
         c.d=self.d
         c.n=copy.deepcopy(self.n)
         c.r=copy.deepcopy(self.r)
         c.core[0]*=other
         return c
      elif isinstance(other,tt_tensor):
         #Write the hadamard product
         pass
      else: 
         print('Error in __mul__ function!\n')
   def __rmul__(self,other): #Multiplication
      if isinstance(other,(float,int)):
         c=tt_tensor()
         c.core=copy.deepcopy(self.core)
         c.d=self.d
         c.n=copy.deepcopy(self.n)
         c.r=copy.deepcopy(self.r)
         c.core[0]*=other
         return c
      elif isinstance(other,tt_tensor):
         #Write the hadamard product
         pass
      else: 
         print('Error in __rmul__ function!\n')
    
   def __sub__(self,other): #Subtract
      return self
  
 
def full(a):           #Convert back to full array --- now the syntax is a.full(); 
   core=a.core
   r=a.r
   d=a.d
   n=a.n
   ret=np.zeros(n)
   ret=core[0]
   for i in range (1,d):
      ret=reshape(ret,(size(ret)/r[i],r[i]),order='F')  
      cur_core=core[i]
      cur_core=reshape(cur_core,(r[i],size(cur_core)/r[i]),order='F')
      ret=dot(ret,cur_core)
   ret.resize(n)
   return ret
#@profile
def round(a,eps,rmax=None): #Approximate with eps-quality (create a copy of the array?)
   if rmax is None:
      rmax=prod(a.n)
   
   b=tt_tensor()
   b.d=a.d
   b.n=a.n
   b.core=copy.deepcopy(a.core)
   b.r=copy.deepcopy(a.r)
   n=b.n
   d=b.d
   core=b.core
   r=b.r
   #Start the work
   nrm=np.zeros(d,dtype=float)
   ru=np.eye(r[0])
   for i in range(0,d-1):
      cur_core=core[i] 
      cur_core=reshape(cur_core,(r[i],size(cur_core)/r[i]),order='F')
      cur_core=dot(ru,cur_core)
      r[i]=cur_core.shape[0]
      cur_core=reshape(cur_core,(size(cur_core)/r[i+1],r[i+1]),order='F')
      cur_core,ru=qr(cur_core,mode='full')
      rnew=cur_core.shape[1]
      core[i]=reshape(cur_core,(r[i],n[i],rnew),order='F')
  
    #The last core
   cur_core=core[d-1]
   cur_core=reshape(cur_core,(r[d-1],size(cur_core)/r[d-1]),order='F')
   cur_core=dot(ru,cur_core)
   r[d-1]=cur_core.shape[0]  
   core[d-1]=reshape(cur_core,(r[d-1],n[d-1],r[d]),order='F')
   #Now right-to-left svd
   rv=np.eye(r[d]) #Will not work for the block format
   for i in range(d-1,0,-1):
      cur_core=core[i] 
  
      cur_core=reshape(cur_core,(size(cur_core)/r[i+1],r[i+1]),order='F')
      cur_core=dot(cur_core,rv)
      
      r[i+1]=cur_core.shape[1]
      cur_core=reshape(cur_core,(r[i],size(cur_core)/r[i]),order='F')
      rv,s,v=svd(cur_core,full_matrices=False)
      rnew=my_chop2(s,eps*norm(s)/sqrt(d-1))
      #print 'rnew=',rnew,r
      rv=rv[:,0:rnew]
      s=s[0:rnew]
      v=v[0:rnew,:]
      rv=dot(rv,diag(s))
      #r[i]=r1
      core[i]=reshape(v,(rnew,n[i],r[i+1]),order='F')
      
   #For cycle end
   cur_core=core[0]
   cur_core=reshape(cur_core,(size(cur_core)/r[1],r[1]),order='F')
   cur_core=dot(cur_core,rv)
   r[1]=cur_core.shape[1]
   core[0]=reshape(cur_core,(r[0],n[0],r[1]),order='F')
   return b


#@profile
def my_chop2(s,eps):  #This function should be rewritten using fast inline
   sv=s[::-1]
   sv=sv**2
   ff=nonzero(sv<eps)
   if size(ff) == 0:
     r=size(s)
   else:
     r=size(s)-ff[-1][-1]-1
   return r
#@profile
def randn(n,d,r): #Create a random tt_tensor
   a=tt_tensor()
   a.n=[n]*d
   a.d=d
   a.r=np.ones((d+1,1),dtype=np.int64)*r
  
   a.r[0]=1
   a.r[d]=1
   a.core=np.empty(d,dtype=object)
   for i in range(0,d):
      a.core[i]=np.random.randn(a.r[i],a.n[i],a.r[i+1])
   return a 
