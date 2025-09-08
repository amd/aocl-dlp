GEMM Operations
===============

General Matrix Multiplication (GEMM) operations with support for multiple data types and optimizations.

.. note::
   If the function list below is empty, ensure Doxygen XML is generated and available to Sphinx. See `docs/TODO.md`.



GEMM
------------------

Float32
~~~~~~~~~~~~
.. _fn-aocl-gemm-f32f32f32of32:
.. doxygenfunction:: aocl_gemm_f32f32f32of32
   :project: aocl-dlp

BFloat16
~~~~~~~~~~~~~
.. _fn-aocl-gemm-bf16bf16f32of32:
.. doxygenfunction:: aocl_gemm_bf16bf16f32of32
   :project: aocl-dlp

.. doxygenfunction:: aocl_gemm_bf16bf16f32obf16
   :project: aocl-dlp

Int8
~~~~~~~~~~~~~~
.. _fn-aocl-gemm-u8s8s32os32:
.. doxygenfunction:: aocl_gemm_u8s8s32os32
   :project: aocl-dlp

.. doxygenfunction:: aocl_gemm_s8s8s32os32
   :project: aocl-dlp

.. _fn-aocl-gemm-s8s8s32os8:
.. doxygenfunction:: aocl_gemm_s8s8s32os8
   :project: aocl-dlp

.. doxygenfunction:: aocl_gemm_u8s8s32os8
   :project: aocl-dlp

.. doxygenfunction:: aocl_gemm_u8s8s32of32
   :project: aocl-dlp

.. doxygenfunction:: aocl_gemm_u8s8s32obf16
   :project: aocl-dlp

.. doxygenfunction:: aocl_gemm_u8s8s32ou8
   :project: aocl-dlp

.. doxygenfunction:: aocl_gemm_s8s8s32of32
   :project: aocl-dlp

.. doxygenfunction:: aocl_gemm_s8s8s32obf16
   :project: aocl-dlp

.. doxygenfunction:: aocl_gemm_s8s8s32ou8
   :project: aocl-dlp

Mixed Precision GEMM
~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: aocl_gemm_bf16s4f32of32
   :project: aocl-dlp

.. doxygenfunction:: aocl_gemm_bf16s4f32obf16
   :project: aocl-dlp

Symmetric Quantization GEMM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: aocl_gemm_s8s8s32of32_sym_quant
   :project: aocl-dlp

.. doxygenfunction:: aocl_gemm_s8s8s32obf16_sym_quant
   :project: aocl-dlp

Batch GEMM Operations
---------------------
.. _fn-aocl-batch-gemm-f32f32f32of32:
.. doxygenfunction:: aocl_batch_gemm_f32f32f32of32
   :project: aocl-dlp

.. doxygenfunction:: aocl_batch_gemm_bf16bf16f32of32
   :project: aocl-dlp

.. doxygenfunction:: aocl_batch_gemm_bf16bf16f32obf16
   :project: aocl-dlp

.. doxygenfunction:: aocl_batch_gemm_bf16s4f32of32
   :project: aocl-dlp

.. doxygenfunction:: aocl_batch_gemm_bf16s4f32obf16
   :project: aocl-dlp

.. doxygenfunction:: aocl_batch_gemm_u8s8s32os32
   :project: aocl-dlp

.. doxygenfunction:: aocl_batch_gemm_u8s8s32os8
   :project: aocl-dlp

.. doxygenfunction:: aocl_batch_gemm_u8s8s32of32
   :project: aocl-dlp

.. doxygenfunction:: aocl_batch_gemm_u8s8s32obf16
   :project: aocl-dlp

.. doxygenfunction:: aocl_batch_gemm_u8s8s32ou8
   :project: aocl-dlp

.. doxygenfunction:: aocl_batch_gemm_s8s8s32os32
   :project: aocl-dlp

.. doxygenfunction:: aocl_batch_gemm_s8s8s32os8
   :project: aocl-dlp

.. doxygenfunction:: aocl_batch_gemm_s8s8s32of32
   :project: aocl-dlp

.. doxygenfunction:: aocl_batch_gemm_s8s8s32obf16
   :project: aocl-dlp

.. doxygenfunction:: aocl_batch_gemm_s8s8s32ou8
   :project: aocl-dlp

Matrix Reordering
-----------------

Buffer Size Functions
~~~~~~~~~~~~~~~~~~~~~
.. _fn-aocl-get-reorder-buf-size-f32f32f32of32:
.. doxygenfunction:: aocl_get_reorder_buf_size_f32f32f32of32
   :project: aocl-dlp

.. doxygenfunction:: aocl_get_reorder_buf_size_u8s8s32os32
   :project: aocl-dlp

.. doxygenfunction:: aocl_get_reorder_buf_size_bf16bf16f32of32
   :project: aocl-dlp

.. doxygenfunction:: aocl_get_reorder_buf_size_s8s8s32os32
   :project: aocl-dlp

.. doxygenfunction:: aocl_get_reorder_buf_size_u8s4s32os32
   :project: aocl-dlp

.. doxygenfunction:: aocl_get_reorder_buf_size_bf16s4f32of32
   :project: aocl-dlp

.. doxygenfunction:: aocl_get_reorder_buf_size_s8s8s32os32_sym_quant
   :project: aocl-dlp

Reordering Functions
~~~~~~~~~~~~~~~~~~~~
.. _fn-aocl-reorder-f32f32f32of32:
.. doxygenfunction:: aocl_reorder_f32f32f32of32
   :project: aocl-dlp

.. doxygenfunction:: aocl_reorder_f32f32f32of32_reference
   :project: aocl-dlp

.. doxygenfunction:: aocl_reorder_u8s8s32os32
   :project: aocl-dlp

.. doxygenfunction:: aocl_reorder_bf16bf16f32of32
   :project: aocl-dlp

.. doxygenfunction:: aocl_reorder_bf16bf16f32of32_reference
   :project: aocl-dlp

.. doxygenfunction:: aocl_reorder_s8s8s32os32
   :project: aocl-dlp

.. doxygenfunction:: aocl_reorder_u8s4s32os32
   :project: aocl-dlp

.. doxygenfunction:: aocl_reorder_bf16s4f32of32
   :project: aocl-dlp

.. doxygenfunction:: aocl_reorder_s8s8s32os32_sym_quant
   :project: aocl-dlp

.. doxygenfunction:: aocl_reorder_f32obf16
   :project: aocl-dlp

Unreordering Functions
~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: aocl_unreorder_bf16bf16f32of32
   :project: aocl-dlp

.. doxygenfunction:: aocl_unreorder_bf16bf16f32of32_reference
   :project: aocl-dlp

.. doxygenfunction:: aocl_unreorder_f32f32f32of32_reference
   :project: aocl-dlp

.. doxygenfunction:: aocl_unreorder_s8s8s32os32_reference
   :project: aocl-dlp


.. include:: post_ops.rst

.. rubric:: See Also

* :doc:`../post_ops/index` - Post-operations framework
* :doc:`../eltwise/index` - Element-wise operations
* :doc:`../library/index` - Library configuration

.. doxygenfile:: aocl_gemm_interface_apis.h
   :project: aocl-dlp
   :preprocess:
