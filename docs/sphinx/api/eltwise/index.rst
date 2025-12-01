.. _eltwise-operations:

Element-wise Operations
=======================

Element-wise operations API for applying mathematical functions to matrix elements.

.. _eltwise-function-reference:

Function Reference
------------------
.. _fn-aocl-gemm-eltwise-ops-bf16of32:
.. doxygenfunction:: aocl_gemm_eltwise_ops_bf16of32
   :project: aocl-dlp

.. doxygenfunction:: aocl_gemm_eltwise_ops_bf16obf16
   :project: aocl-dlp

.. doxygenfunction:: aocl_gemm_eltwise_ops_f32of32
   :project: aocl-dlp

.. doxygenfunction:: aocl_gemm_eltwise_ops_f32obf16
   :project: aocl-dlp

.. doxygenfunction:: aocl_gemm_eltwise_ops_f32os32
   :project: aocl-dlp

.. doxygenfunction:: aocl_gemm_eltwise_ops_f32os8
   :project: aocl-dlp

.. doxygenfunction:: aocl_gemm_eltwise_ops_f32ou8
   :project: aocl-dlp

.. rubric:: See Also

* :doc:`../gemm/post_ops` - Post-operations framework
* :doc:`../utils/index` - Utility functions
* :doc:`../gemm/index` - GEMM operations
