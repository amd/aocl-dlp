AOCL-DLP API Documentation
==========================

Welcome to the AMD Optimizing CPU Libraries - Deep Learning Primitives (AOCL-DLP) API documentation.

This site provides the API reference only. For user guides, tutorials, installation, and examples, please visit the project Wiki.

API Reference
=============

Complete API reference for AOCL-DLP (AMD Optimizing CPU Libraries - Deep Learning Primitives).

.. toctree::
   :maxdepth: 2
   :caption: API Categories

   api/overview
   api/gemm/index
   api/gemm/post_ops
   api/eltwise/index
   api/utils/index
   api/library/index
   api/types/index

Quick API Lookup
----------------

.. _index-core-gemm:

Core GEMM Operations
~~~~~~~~~~~~~~~~~~~~

.. list-table:: GEMM API Functions
   :header-rows: 1
   :widths: 40 60

   * - Function Pattern
     - Description
   * - ``aocl_gemm_f32f32f32of32``
     - Float32 precision GEMM
   * - ``aocl_gemm_bf16bf16f32of32``
     - BFloat16 inputs, float32 output
   * - ``aocl_gemm_u8s8s32os32``
     - Unsigned/signed 8-bit quantized GEMM
   * - ``aocl_gemm_s8s8s32os8``
     - Signed 8-bit quantized GEMM
   * - ``aocl_gemm_f16f16f16of16``
     - IEEE float16 precision GEMM
   * - ``aocl_gemm_bf16s4f32of32``
     - BFloat16 x int4 mixed precision
   * - ``aocl_gemm_bf16s8s32os32``
     - BFloat16 x int8 mixed precision
   * - ``aocl_gemm_f32s8s32os32``
     - Float32 x int8 mixed precision
   * - ``aocl_gemm_s8s8s32of32_sym_quant``
     - Symmetric quantization GEMM

.. _index-batch-operations:

Batch Operations
~~~~~~~~~~~~~~~~

.. list-table:: Batch GEMM Functions
   :header-rows: 1
   :widths: 40 60

   * - Function Pattern
     - Description
   * - ``aocl_batch_gemm_*``
     - Batch processing for multiple matrices

.. _index-post-operations:

Post-Operations
~~~~~~~~~~~~~~~

.. list-table:: Post-Operations Framework
   :header-rows: 1
   :widths: 40 60

   * - Type / Structure
     - Description
   * - ``dlp_metadata_t``
     - Main metadata structure for configuring post-ops
   * - ``DLP_POST_OP_TYPE``
     - Post-op types: BIAS, ELTWISE, SCALE, MATRIX_ADD, MATRIX_MUL
   * - ``DLP_ELT_ALGO_TYPE``
     - Activation functions: RELU, GELU, SWISH, TANH, SIGMOID, etc.

Matrix Utilities
~~~~~~~~~~~~~~~~

.. list-table:: Matrix Utility Functions
   :header-rows: 1
   :widths: 40 60

   * - Function Pattern
     - Description
   * - ``aocl_get_reorder_buf_size_*``
     - Get buffer size for matrix reordering
   * - ``aocl_reorder_*``
     - Reorder matrix for optimal performance
   * - ``aocl_unreorder_*``
     - Convert reordered matrix back to normal format

.. _index-element-wise:

Element-wise Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Element-wise Functions
   :header-rows: 1
   :widths: 40 60

   * - Function Pattern
     - Description
   * - ``aocl_gemm_eltwise_ops_*``
     - Apply element-wise operations to matrices

.. _index-utility-functions:

Utility Functions
~~~~~~~~~~~~~~~~~

.. list-table:: Utility Functions
   :header-rows: 1
   :widths: 40 60

   * - Function Pattern
     - Description
   * - ``aocl_gemm_gelu_*``
     - GELU activation functions
   * - ``aocl_gemm_softmax_*``
     - Softmax functions

.. _index-library-management:

Library Management
~~~~~~~~~~~~~~~~~~

.. list-table:: Library Functions
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``dlp_thread_set_num_threads``
     - Configure thread count
   * - ``dlp_thread_set_ways``
     - Configure parallelization strategy
   * - ``dlp_aocl_enable_instruction_query``
     - Query AOCL_DLP_ENABLE_INSTRUCTIONS environment setting
   * - ``dlp_version_query``
     - Query library version (major, minor, patch)

API Selection Guide
-------------------

Choose the Right GEMM Variant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**By Precision Requirements:**

1. **High Precision**: ``f32f32f32of32`` for maximum accuracy
2. **Balanced**: ``bf16bf16f32of32`` for good accuracy with reduced memory
3. **Quantized**: ``u8s8s32os32`` or ``s8s8s32os8`` for inference

**By Performance Needs:**

1. **Single Operation**: Standard GEMM functions
2. **Multiple Operations**: Batch GEMM functions
3. **Repeated Operations**: Use matrix reordering

.. _index-data-type-naming:

Data Type Naming Convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Function names follow the pattern: ``[input_A][input_B][accumulation]o[output]``

- ``f32`` = float32
- ``bf16`` = bfloat16
- ``u8`` = uint8
- ``s8`` = int8
- ``s32`` = int32

Example: ``bf16bf16f32of32`` = bfloat16 inputs, float32 accumulation and output

.. _index-see-also:

See Also
--------

* :doc:`api/overview` - API design principles and usage patterns
* :doc:`api/gemm/index` - GEMM operations documentation
* :doc:`api/gemm/post_ops` - Post-operations framework
* `Quick Start Guide <https://github.com/amd/aocl-dlp/wiki/Quick-Start>`_ - Get started in 5 minutes
* `Integration Guide <https://github.com/amd/aocl-dlp/wiki/Integration-Guide>`_ - CMake integration, linking, and troubleshooting
* `Examples and Tutorials <https://github.com/amd/aocl-dlp/wiki/Examples-and-Tutorials>`_ - Code examples and usage patterns

Indices and Tables
==================

* :ref:`genindex`
* :ref:`search`
