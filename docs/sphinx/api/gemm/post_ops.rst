
Post-Operations
===============

Post-operations framework for fusing operations with GEMM computations.

Data Structures
---------------
.. _struct-dlp-metadata-t:
.. doxygenstruct:: dlp_metadata_t
   :project: aocl-dlp
   :members:

Post-Op Building Blocks
-----------------------

.. doxygenstruct:: dlp_eltwise_algo_t
   :project: aocl-dlp
   :members:

.. doxygenstruct:: dlp_zp_t
   :project: aocl-dlp
   :members:

.. doxygenstruct:: dlp_sf_t
   :project: aocl-dlp
   :members:

.. doxygenstruct:: dlp_post_op_eltwise
   :project: aocl-dlp
   :members:

.. doxygenstruct:: dlp_scale_t
   :project: aocl-dlp
   :members:

.. doxygenstruct:: dlp_post_op_bias
   :project: aocl-dlp
   :members:

.. doxygenstruct:: dlp_post_op_matrix_add
   :project: aocl-dlp
   :members:

.. doxygenstruct:: dlp_post_op_matrix_mul
   :project: aocl-dlp
   :members:

.. doxygenstruct:: dlp_pre_op
   :project: aocl-dlp
   :members:

.. doxygenstruct:: dlp_group_post_op
   :project: aocl-dlp
   :members:

.. doxygenstruct:: DLP_SYMM_STAT_QUANT
   :project: aocl-dlp
   :members:

.. doxygenstruct:: dlp_quant_op
   :project: aocl-dlp
   :members:

Error Handling
--------------

.. doxygenenum:: dlp_clsc_err_t
   :project: aocl-dlp

.. doxygenstruct:: dlp_error_hndl_t
   :project: aocl-dlp
   :members:

Enums
-----

.. doxygenenum:: DLP_POST_OP_TYPE
   :project: aocl-dlp

.. doxygenenum:: DLP_ELT_ALGO_TYPE
   :project: aocl-dlp

.. doxygenenum:: DLP_TYPE
   :project: aocl-dlp

.. rubric:: See Also

* :doc:`../gemm/index` - GEMM operations
* :doc:`../eltwise/index` - Element-wise operations
* :doc:`../utils/index` - Utility functions
