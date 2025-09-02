API Reference
=============

Complete API reference for AOCL-DLP (AMD Optimizing CPU Libraries - Deep Learning Primitives).

.. toctree::
   :maxdepth: 2
   :caption: API Categories

   overview
   gemm/index
   post_ops/index
   eltwise/index
   utils/index
   library/index
   classic/index

Quick API Lookup
----------------

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

Batch Operations
~~~~~~~~~~~~~~~~

.. list-table:: Batch GEMM Functions
   :header-rows: 1
   :widths: 40 60

   * - Function Pattern
     - Description
   * - ``aocl_batch_gemm_*``
     - Batch processing for multiple matrices

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

Element-wise Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Element-wise Functions
   :header-rows: 1
   :widths: 40 60

   * - Function Pattern
     - Description
   * - ``aocl_gemm_eltwise_ops_*``
     - Apply element-wise operations to matrices

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
     - Query hardware capabilities

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

Data Type Naming Convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Function names follow the pattern: ``[input_A][input_B][accumulation]o[output]``

- ``f32`` = float32
- ``bf16`` = bfloat16
- ``u8`` = uint8
- ``s8`` = int8
- ``s32`` = int32

Example: ``bf16bf16f32of32`` = bfloat16 inputs, float32 accumulation and output

See Also
--------

* :doc:`overview` - API design principles and usage patterns
* :doc:`gemm/index` - GEMM operations documentation
* :doc:`post_ops/index` - Post-operations framework
