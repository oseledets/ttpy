Core TT utils
*************

.. .. automodule:: tt.core

Tensor manipulation routines
============================

``tt.tensor`` is the main class for manipulating TT-compressed tensors. To compress a tensor given
as full-weight NumPy array ``arr``, simply call

>>> arr_tt = tt.tensor(arr)

Then the full-weight array can be recovered from TT-representation using

>>> arr = arr_tt.full()

TT-tensor objects support basic element-wise arithmetic:

>>> a_tt, b_tt = tt.tensor(a), tt_tensor(b)
>>> c_tt = 0.3 * a_tt + b_tt - a_tt * b_tt

Such operations do usually increase TT-ranks, thus applying TT-rounding in these situations is recommended:

>>> c_tt = c_tt.round(1E-14)

Further details are descripted below.

.. autoclass:: tt.tensor
   :members:


Basic operations in TT-format
-----------------------------

.. automodule:: tt
   :members: diag, kron, mkron, concatenate, sum, matvec

Generation of standard TT-tensors
-----------------------------------

.. automodule:: tt
   :members: ones, xfun, sin, cos, delta, stepfun, rand

Matrix manipulation routines
============================

``tt.matrix`` is a special class for manipulating TT-decomposed matrices, i. e. decompositions of the form

.. math::
   
   A(i_1,\ldots,i_d;j_1,\ldots,j_d) = \sum_{\alpha_1,\ldots,\alpha_{d-1}}A_1(i_1,j_1,\alpha_1)\cdot A_2(\alpha_1,i_2,j_2,\alpha_2) \cdot \ldots \cdot A_d(\alpha_{d-1},i_d, j_d),

which are just regular TT-decompositions of tensors :math:`A(i_1,j_1;i_2,j_2;\ldots;i_d,j_d)` with merged indices :math:`i_k,j_k`.

.. autoclass:: tt.matrix
   :members: 

Generation of standard TT-matrices
----------------------------------

.. automodule:: tt
   :members: eye, qlaplace_dd, Toeplitz

