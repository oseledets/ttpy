.. ttpy documentation master file, created by
   sphinx-quickstart on Fri Apr 12 16:14:47 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TTPY Documentation
******************

.. |ttpy| replace:: **ttpy**

|ttpy| is a powerful Python package designed for manipulating large highly multidimensional arrays 
also called **tensors**. It is developed in the `Institute of Numerical Mathematics of Russian Academy 
of Sciences <http://www.inm.ras.ru/inm_en_ver/index.htm>`_ and based on the so-called Tensor Train 
decomposition techniques.

The Tensor Train decomposition of a tensor :math:`A(i_1,\ldots,i_d)` of shape 
:math:`n_1\times\ldots\times n_d` is basically any it's representation in form

.. math::
   
   A(i_1,\ldots,i_d) = \sum_{\alpha_1,\ldots,\alpha_{d-1}}A_1(i_1,\alpha_1)\cdot A_2(\alpha_1,i_2,\alpha_2) \cdot \ldots \cdot A_d(\alpha_{d-1},i_d).


For more detailed information on tensor algebra basics and the Tensor Train decomposition theory, see :doc:`tensorintro` topic.

Documentation contents:

.. toctree::
   :maxdepth: 2
   
   tensorintro
   corettutils
   ttalgorithms
 
.. only:: html 

   ******************
   Indices and Tables
   ******************

   * :ref:`genindex`
   * :ref:`search`

