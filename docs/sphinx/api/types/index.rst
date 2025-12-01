Data Types and Structures
=========================

Data types, structures, and constants used throughout AOCL-DLP.

Overview
--------

AOCL-DLP defines various data types and structures to support different precision levels and operation configurations. Understanding these types is essential for effective use of the library.

Basic Data Types
----------------

Dimension Types
~~~~~~~~~~~~~~~

.. doxygentypedef:: md_t
   :project: aocl-dlp

BFloat16 Support
~~~~~~~~~~~~~~~~

.. doxygentypedef:: bfloat16
   :project: aocl-dlp

Constants and Enumerations
---------------------------

Memory Format Constants
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Memory Format Values
   :header-rows: 1
   :widths: 25 75

   * - Constant
     - Description
   * - ``'R'``
     - Row-major (C-style) memory layout
   * - ``'C'``
     - Column-major (Fortran-style) memory layout
   * - ``'N'``
     - Normal (unpacked) matrix format
   * - ``'R'``
     - Reordered (packed) matrix format

Transpose Options
~~~~~~~~~~~~~~~~~

.. list-table:: Transpose Values
   :header-rows: 1
   :widths: 25 75

   * - Constant
     - Description
   * - ``'N'``
     - No transpose
   * - ``'T'``
     - Transpose matrix

Data Type Naming Convention
---------------------------

Function names in AOCL-DLP follow a systematic naming convention that encodes the data types:

.. _types-data-type-naming:

Pattern: ``[input_A][input_B][accumulation]o[output]``

.. list-table:: Type Abbreviations
   :header-rows: 1
   :widths: 20 30 50

   * - Abbreviation
     - Data Type
     - Description
   * - ``f32``
     - float (32-bit)
     - IEEE 754 single precision
   * - ``bf16``
     - bfloat16 (16-bit)
     - Brain floating point format
   * - ``u8``
     - uint8_t (8-bit)
     - Unsigned 8-bit integer
   * - ``s8``
     - int8_t (8-bit)
     - Signed 8-bit integer
   * - ``s4``
     - int4 (4-bit)
     - Signed 4-bit integer (packed)
   * - ``s32``
     - int32_t (32-bit)
     - Signed 32-bit integer

Examples
~~~~~~~~

.. list-table:: Function Name Examples
   :header-rows: 1
   :widths: 40 60

   * - Function Name
     - Meaning
   * - ``aocl_gemm_f32f32f32of32``
     - float32 × float32 → float32 (with float32 accumulation)
   * - ``aocl_gemm_bf16bf16f32of32``
     - bfloat16 × bfloat16 → float32 (with float32 accumulation)
   * - ``aocl_gemm_u8s8s32os8``
     - uint8 × int8 → int8 (with int32 accumulation)
   * - ``aocl_gemm_bf16s4f32of32``
     - bfloat16 × int4 → float32 (with float32 accumulation)

Memory Layout Considerations
----------------------------

Row-Major vs Column-Major
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

   // Row-major (C-style): A[i][j] = A[i*lda + j]
   float A_row_major[M][N];

   // Column-major (Fortran-style): A[i][j] = A[j*lda + i]
   float A_col_major[N][M];

Leading Dimensions
~~~~~~~~~~~~~~~~~~

The leading dimension must be at least as large as the corresponding matrix dimension:

.. code-block:: c

   // For row-major matrices
   assert(lda >= k);  // For matrix A (m × k)
   assert(ldb >= n);  // For matrix B (k × n)
   assert(ldc >= n);  // For matrix C (m × n)

   // For column-major matrices
   assert(lda >= m);  // For matrix A (m × k)
   assert(ldb >= k);  // For matrix B (k × n)
   assert(ldc >= m);  // For matrix C (m × n)

BFloat16 Usage
--------------

BFloat16 Format
~~~~~~~~~~~~~~~

BFloat16 (Brain Floating Point) is a 16-bit floating point format:

* **1 bit**: Sign
* **8 bits**: Exponent (same as float32)
* **7 bits**: Mantissa (reduced from float32's 23 bits)

Usage Example
~~~~~~~~~~~~~

.. code-block:: c

   #include "aocl_dlp.h"

   // Convert float32 to bfloat16
   float f32_value = 3.14159f;
   aocl_bf16 bf16_value = (aocl_bf16)f32_value;

   // Use in GEMM operations
   aocl_bf16 *a_bf16, *b_bf16;
   float *c_f32;

   aocl_gemm_bf16bf16f32of32(
       'R', 'N', 'N', m, n, k,
       1.0f, a_bf16, lda, 'N',
       b_bf16, ldb, 'N',
       0.0f, c_f32, ldc, NULL
   );

Type Safety and Conversions
---------------------------

Implicit Conversions
~~~~~~~~~~~~~~~~~~~~

AOCL-DLP handles some implicit type conversions:

* **Quantized to float**: Automatic dequantization in mixed-precision operations
* **BFloat16 to float32**: Automatic promotion for accumulation
* **Integer widening**: Automatic promotion to prevent overflow

Explicit Conversions
~~~~~~~~~~~~~~~~~~~~

For explicit type conversions, use appropriate utility functions or element-wise operations:

.. code-block:: c

   // Convert float32 array to bfloat16
   aocl_gemm_eltwise_ops_f32obf16(
       'R', 'N', 'N', m, n,
       f32_array, lda,
       bf16_array, ldb,
       NULL  // No additional operations
   );

.. _types-best-practices:

Best Practices
--------------

1. **Choose appropriate precision**: Balance accuracy and performance requirements
2. **Understand memory layouts**: Use consistent layouts throughout your application
3. **Validate dimensions**: Ensure leading dimensions are correctly set
4. **Handle type conversions**: Be explicit about precision requirements
5. **Consider hardware support**: Some types may have better hardware acceleration

.. _types-see-also:

See Also
--------

* :doc:`../gemm/index` - GEMM operations using these types
* :doc:`../gemm/post_ops` - Post-operations metadata structures
* :doc:`../eltwise/index` - Element-wise type conversions
