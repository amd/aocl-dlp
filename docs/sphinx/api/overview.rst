API Overview
============

AOCL-DLP provides a comprehensive set of APIs for high-performance deep learning primitives optimized for AMD processors. This overview introduces the key concepts, design principles, and usage patterns that apply across all APIs.

Core Concepts
-------------

Matrix Operations
~~~~~~~~~~~~~~~~~

AOCL-DLP is built around optimized matrix operations, primarily General Matrix Multiplication (GEMM):

.. math::

   C = \text{post_ops} (\alpha \cdot op(A) \cdot op(B) + \beta \cdot C)

Where:

- :math:`op(X)` can be :math:`X` (no transpose) or :math:`X^T` (transpose)
- :math:`\alpha, \beta` are scalar multipliers
- :math:`\text{post_ops}` represents fused post-processing operations

Data Type Support
~~~~~~~~~~~~~~~~~~

AOCL-DLP supports multiple precision formats to balance accuracy and performance:

.. list-table:: Supported Data Types
   :header-rows: 1
   :widths: 15 20 15 50

   * - Type
     - Size (bits)
     - Range
     - Use Case
   * - ``float32``
     - 32
     - ±3.4×10³⁸
     - High-precision training and inference
   * - ``bfloat16``
     - 16
     - ±3.4×10³⁸
     - Memory-efficient inference with good range
   * - ``int8``
     - 8
     - -128 to 127
     - Quantized weights and activations
   * - ``uint8``
     - 8
     - 0 to 255
     - Quantized activations (unsigned)
   * - ``int4``
     - 4
     - -8 to 7
     - Extreme quantization (packed in int8)
   * - ``int32``
     - 32
     - ±2.1×10⁹
     - Accumulation and intermediate results

API Categories
--------------

.. _overview-core-gemm:

Core GEMM Operations
~~~~~~~~~~~~~~~~~~~~

The fundamental matrix multiplication APIs:

.. code-block:: c

   // Float32 precision
   aocl_gemm_f32f32f32of32(...)

   // BFloat16 with float32 accumulation
   aocl_gemm_bf16bf16f32of32(...)

   // Quantized integer operations
   aocl_gemm_u8s8s32os32(...)
   aocl_gemm_s8s8s32os8(...)

.. _overview-batch-operations:

Batch Operations
~~~~~~~~~~~~~~~~

For processing multiple GEMM operations efficiently:

.. code-block:: c

   // Batch processing
   aocl_batch_gemm_f32f32f32of32(...)
   aocl_batch_gemm_bf16bf16f32of32(...)

.. _api-overview-element-wise:

Element-wise Operations
~~~~~~~~~~~~~~~~~~~~~~~

For applying operations without matrix multiplication:

.. code-block:: c

   // Element-wise operations
   aocl_gemm_eltwise_ops_f32of32(...)
   aocl_gemm_eltwise_ops_bf16of32(...)

.. _overview-utility-functions:

Utility Functions
~~~~~~~~~~~~~~~~~

Standalone mathematical operations:

.. code-block:: c

   // Activation functions
   aocl_gemm_gelu_tanh_f32(...)
   aocl_gemm_gelu_erf_f32(...)
   aocl_gemm_softmax_f32(...)

.. _api-overview-matrix-reordering:

Matrix Reordering
~~~~~~~~~~~~~~~~~

For optimizing repeated operations:

.. code-block:: c

   // Get buffer size for reordering
   aocl_get_reorder_buf_size_f32f32f32of32(...)

   // Reorder matrix for optimal access
   aocl_reorder_f32f32f32of32(...)

.. _api-overview-library-management:

Library Management
~~~~~~~~~~~~~~~~~~

For configuration:

.. code-block:: c

   // Thread configuration
   dlp_thread_set_num_threads(...)
   dlp_thread_set_ways(...)

   // Validate an AOCL_DLP_ENABLE_INSTRUCTIONS override
   dlp_aocl_enable_instruction_query()

API Design Principles
---------------------

Consistent Naming
~~~~~~~~~~~~~~~~~

All APIs follow a systematic naming convention:

.. code-block:: text

   aocl_[operation]_[input_types]o[output_type]

Examples:
- ``aocl_gemm_f32f32f32of32``: GEMM with float32 inputs and output
- ``aocl_gemm_u8s8s32os8``: GEMM with uint8/int8 inputs, int32 accumulation, int8 output
- ``aocl_batch_gemm_bf16bf16f32of32``: Batch GEMM with bfloat16 inputs, float32 output

Memory Layout Flexibility
~~~~~~~~~~~~~~~~~~~~~~~~~~

GEMM APIs use ``order`` for matrix layout and ``mem_format_a`` /
``mem_format_b`` for storage format:

.. list-table:: Layout and Memory Format Options
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Value
     - Description
   * - ``order``
     - ``'R'``
     - Row-major (C-style) layout
   * - ``order``
     - ``'C'``
     - Column-major (Fortran-style) layout
   * - ``mem_format_a`` or ``mem_format_b``
     - ``'N'``
     - Normal (unpacked) matrix storage
   * - ``mem_format_a`` or ``mem_format_b``
     - ``'P'``
     - Packed matrix storage where supported by the API
   * - ``mem_format_b``
     - ``'R'``
     - Reordered B storage where supported; A reordering is not supported

.. _api-design-parameter-validation:

Parameter Validation
~~~~~~~~~~~~~~~~~~~~

AOCL-DLP APIs perform robust parameter validation:

- **Dimension checks**: Ensure matrix dimensions are compatible
- **Pointer validation**: Handle NULL pointers gracefully
- **Range validation**: Check for valid enumeration values
- **Memory format validation**: Verify supported layout combinations

Hardware Abstraction
~~~~~~~~~~~~~~~~~~~~~

The library selects implementations supported by the current processor:

- **Feature detection**: Runtime detection of CPU capabilities
- **Path selection**: Choose an available implementation for the requested API
- **Availability**: Unsupported ISA/API combinations follow the individual
  API's documented error or no-op behavior

Post-Operations Framework
-------------------------

AOCL-DLP supports fusing common operations with GEMM to improve performance:

Operation Types
~~~~~~~~~~~~~~~

.. list-table:: Post-Operation Categories
   :header-rows: 1
   :widths: 20 30 50

   * - Category
     - Operations
     - Performance Benefit
   * - Activation
     - ReLU, PReLU, GeLU, Tanh, Sigmoid, SWISH
     - Eliminates separate activation pass
   * - Scaling
     - Scale, Clip
     - Fuses quantization/normalization
   * - Addition
     - Bias, Matrix Add
     - Combines common DNN operations
   * - Multiplication
     - Matrix Multiply
     - Enables element-wise scaling

Usage Pattern
~~~~~~~~~~~~~

.. code-block:: c

   // Configure post-operations via dlp_metadata_t
   float bias_vector[N] = { /* ... */ };

   dlp_post_op_bias bias_op = {
       .bias = bias_vector, .stor_type = DLP_F32,
       .sf = NULL, .zp = NULL
   };

   dlp_post_op_eltwise relu_op = {
       .sf = NULL,
       .algo = { .alpha = NULL, .beta = NULL,
                 .algo_type = RELU, .stor_type = DLP_F32 }
   };

   DLP_POST_OP_TYPE seq[] = { BIAS, ELTWISE };

   dlp_metadata_t meta = {0};
   meta.seq_length  = 2;
   meta.seq_vector  = seq;
   meta.bias        = &bias_op;
   meta.eltwise     = &relu_op;
   meta.num_eltwise = 1;

   // Use in GEMM
   aocl_gemm_f32f32f32of32('R', 'N', 'N', m, n, k,
       1.0f, a, lda, 'N', b, ldb, 'N',
       0.0f, c, ldc, &meta);

Performance Optimization
------------------------

Hardware Utilization
~~~~~~~~~~~~~~~~~~~~~

AOCL-DLP automatically leverages available CPU features:

.. list-table:: Hardware Features
   :header-rows: 1
   :widths: 25 25 50

   * - Feature
     - Availability
     - Benefit
   * - AVX2/FMA3
     - AMD Zen 1+
     - Vectorized floating-point operations
   * - AVX512
     - AMD Zen 4+
     - Wider vector operations
   * - AVX512_VNNI
     - AMD Zen 4+
     - Accelerated integer GEMM
   * - AVX512_BF16
     - AMD Zen 4+
     - Native bfloat16 operations

Memory Optimization
~~~~~~~~~~~~~~~~~~~

Key strategies for optimal memory performance:

1. **Data Layout**: Use row-major layout when possible
2. **Alignment**: Align matrices to cache line boundaries (64 bytes)
3. **Reordering**: Reorder frequently-used matrices
4. **Batch Processing**: Group similar operations
   5. **Memory Bandwidth**: Consider bandwidth limitations for large matrices

.. _api-overview-threading-config:

Threading Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Optimize parallel execution:

.. code-block:: c

   // Set thread count (typically number of CPU cores)
   dlp_thread_set_num_threads(8);

   // Configure workload distribution
   dlp_thread_set_ways(2, 4);  // JC=2, IC=4 loop parallelization

Common Usage Patterns
---------------------

Neural Network Inference
~~~~~~~~~~~~~~~~~~~~~~~~~

Typical workflow for neural network layers:

.. code-block:: c

   // 1. Initialize weights (once)
   float *weights = load_weights();

   // 2. Reorder weights for optimal performance (once)
   size_t reorder_size = aocl_get_reorder_buf_size_f32f32f32of32(...);
   float *weights_reordered = malloc(reorder_size);
   aocl_reorder_f32f32f32of32(..., weights, weights_reordered, ...);

   // 3. Set up post-operations (bias + activation)
   dlp_metadata_t post_ops = {0};
   setup_post_ops(&post_ops, bias, activation_type);

   // 4. Process inputs (repeated)
   for (int batch = 0; batch < num_batches; batch++) {
       aocl_gemm_f32f32f32of32(
           'R', 'N', 'N', batch_size, output_dim, input_dim,
           1.0f, input[batch], input_dim, 'N',
           weights_reordered, output_dim, 'R',
           0.0f, output[batch], output_dim,
           &post_ops
       );
   }

Quantized Inference
~~~~~~~~~~~~~~~~~~~

Workflow for quantized neural networks:

.. code-block:: c

   // 1. Load quantized weights and scales
   int8_t *weights_q = load_quantized_weights();
   float *scales = load_scales();

   // 2. Set up quantization post-ops
   dlp_metadata_t post_ops = {0};
   setup_quantization_post_ops(&post_ops, scales, zero_points);

   // 3. Process quantized inputs
   aocl_gemm_u8s8s32os8(
       'R', 'N', 'N', m, n, k,
       1, input_q, k, 'N',
       weights_q, n, 'N',
       0, output_q, n,
       &post_ops
   );

Batch Processing
~~~~~~~~~~~~~~~~

The batch API takes ``group_count`` groups, each containing ``group_size``
matrix multiplications. The example below uses one group with one matrix
multiplication:

.. code-block:: c

   const char order[] = { 'R' }, transa[] = { 'N' }, transb[] = { 'N' };
   const md_t m[] = { rows }, n[] = { columns }, k[] = { inner };
   const float alpha[] = { 1.0f }, beta[] = { 0.0f };
   const float *a[] = { input }, *b[] = { weights };
   float *c[] = { output };
   const md_t lda[] = { inner }, ldb[] = { columns }, ldc[] = { columns };
   const md_t group_size[] = { 1 };
   const char mem_format_a[] = { 'N' }, mem_format_b[] = { 'N' };
   dlp_metadata_t *metadata[] = { NULL };

   aocl_batch_gemm_f32f32f32of32(
       order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
       1, group_size, mem_format_a, mem_format_b, metadata
   );

Error Handling
--------------

AOCL-DLP uses defensive programming practices:

.. _api-overview-error-param-validation:

Parameter Validation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

   // APIs validate parameters and handle gracefully
   if (m <= 0 || n <= 0 || k <= 0) {
       // No operation performed, function returns safely
       return;
   }

   if (a == NULL || b == NULL || c == NULL) {
       // NULL pointers handled without crash
       return;
   }

Hardware Compatibility
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

   // Check whether an instruction-preference override is recognized
   if (!dlp_aocl_enable_instruction_query()) {
       printf("AOCL_DLP_ENABLE_INSTRUCTIONS is unset or invalid\n");
   }

Best Practices
--------------

1. **Choose Appropriate Precision**
   - Use lowest precision that meets accuracy requirements
   - Consider mixed precision (e.g., bf16 inputs, f32 accumulation)

2. **Optimize Memory Access**
   - Prefer row-major layout
   - Align matrices to cache boundaries
   - Use reordering for repeated operations

3. **Leverage Hardware Features**
   - Use feature detection to select optimal algorithms
   - Test on target hardware for validation

4. **Fuse Operations**
   - Use post-operations to minimize memory traffic
   - Group related computations

5. **Profile and Validate**
   - Measure performance with representative workloads
   - Validate numerical accuracy for your use case

.. _overview-best-practices:

Migration Guide
---------------

From Other BLAS Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~

AOCL-DLP APIs are designed to be familiar to BLAS users:

.. list-table:: BLAS to AOCL-DLP Mapping
   :header-rows: 1
   :widths: 30 40 30

   * - BLAS Function
     - AOCL-DLP Equivalent
     - Key Differences
   * - ``sgemm``
     - ``aocl_gemm_f32f32f32of32``
     - Additional post-operations support
   * - ``dgemm``
     - Use ``f32f32f32of32`` variant
     - AOCL-DLP focuses on single precision
   * - Custom quantized GEMM
     - ``aocl_gemm_u8s8s32os8``
     - Built-in quantization support

From Previous AOCL-DLP Versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When upgrading:

1. **Check API compatibility**: Review function signatures
2. **Update post-operations**: New post-op framework may require changes
3. **Validate performance**: Re-benchmark with new version
4. **Test accuracy**: Verify numerical results remain acceptable

.. _api-overview-see-also:

See Also
--------

* :doc:`gemm/index` - Detailed GEMM API documentation
* :doc:`gemm/post_ops` - Post-operations reference
* :doc:`types/index` - Data types and structures reference
* `AOCL-DLP Examples and Tutorials <https://github.com/amd/aocl-dlp/wiki/Examples-and-Tutorials>`_
* `Quick Start Guide <https://github.com/amd/aocl-dlp/wiki/Quick-Start>`_
* `Integration Guide <https://github.com/amd/aocl-dlp/wiki/Integration-Guide>`_
