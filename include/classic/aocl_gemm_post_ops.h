/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef AOCL_DLP_GEMM_POST_OPS_H
#define AOCL_DLP_GEMM_POST_OPS_H

#include "classic/dlp_base_types.h"

#define AOCL_DLP_MAX_POST_OPS 8
#define AOCL_DLP_MAX_PRE_OPS  2

/**
 * @brief Enumeration of element-wise algorithm types supported in
 * post-operations.
 *
 * This enum defines the various activation functions and element-wise
 * operations that can be applied as post-operations in GEMM computations.
 */
/**
 * @enum DLP_ELT_ALGO_TYPE
 * @brief Element-wise algorithm types for post-operations.
 *
 * Enumerates supported activation and element-wise functions for GEMM post-ops.
 */
typedef enum
{
    RELU      = 0, /**< Rectified Linear Unit activation: max(0, x) */
    PRELU     = 1, /**< Parametric ReLU activation: max(alpha*x, x) */
    GELU_TANH = 2, /**< GELU activation using tanh approximation */
    GELU_ERF  = 3, /**< GELU activation using error function */
    CLIP      = 4, /**< Clipping operation: min(max(x, min_val), max_val) */
    SWISH     = 5, /**< Swish activation: x * sigmoid(x) */
    TANH      = 6, /**< Hyperbolic tangent activation */
    SIGMOID   = 7, /**< Sigmoid activation: 1 / (1 + exp(-x)) */
    MISH      = 8, /**< Mish activation: x * tanh(softplus(x)) */
} DLP_ELT_ALGO_TYPE;

/**
 * @brief Enumeration of post-operation types that can be applied to GEMM
 * results.
 *
 * This enum defines the different types of operations that can be performed
 * on the output matrix after GEMM computation.
 */
/**
 * @enum DLP_POST_OP_TYPE
 * @brief Post-operation types for GEMM results.
 *
 * Enumerates supported post-operations that can be applied to GEMM output.
 */
typedef enum
{
    ELTWISE    = 1, /**< Element-wise operations (activations) */
    BIAS       = 2, /**< Bias addition operation */
    SCALE      = 3, /**< Scaling operation */
    MATRIX_ADD = 4, /**< Matrix addition operation */
    MATRIX_MUL = 5, /**< Matrix multiplication operation */
} DLP_POST_OP_TYPE;

/**
 * @brief Enumeration of supported data types for parameter storage.
 *
 * This enum defines the various data types that can be used for storing
 * parameters in GEMM operations and post-operations.
 */
/**
 * @enum DLP_TYPE
 * @brief Supported data types for GEMM and post-op parameters.
 *
 * Enumerates all valid data types for parameter storage in GEMM/post-ops.
 */
typedef enum
{
    DLP_INVALID = 0, /**< Invalid or unspecified type */
    DLP_S4,          /**< Signed 4-bit integer */
    DLP_U4,          /**< Unsigned 4-bit integer */
    DLP_F4,          /**< 4-bit floating point */
    DLP_S8,          /**< Signed 8-bit integer */
    DLP_U8,          /**< Unsigned 8-bit integer */
    DLP_S16,         /**< Signed 16-bit integer */
    DLP_U16,         /**< Unsigned 16-bit integer */
    DLP_F16,         /**< 16-bit floating point */
    DLP_BF16,        /**< Brain floating point 16-bit */
    DLP_S32,         /**< Signed 32-bit integer */
    DLP_U32,         /**< Unsigned 32-bit integer */
    DLP_F32,         /**< 32-bit floating point */
    DLP_MAX          /**< Maximum value (enum boundary) */
} DLP_TYPE;

/**
 * @brief Structure defining element-wise algorithm parameters.
 *
 * This structure contains the parameters needed for element-wise operations
 * such as activation functions in post-operations.
 */
/**
 * @struct dlp_eltwise_algo_t
 * @brief Parameters for element-wise algorithm in post-ops.
 *
 * Holds alpha, beta, and algorithm type for element-wise operations (e.g.,
 * activation functions).
 */
typedef struct
{
    void* alpha; /**< Alpha parameter for the algorithm (e.g., leak factor for
                    PReLU) */
    void* beta;  /**< Beta parameter for the algorithm (e.g., upper bound for
                    CLIP) */
    DLP_ELT_ALGO_TYPE algo_type; /**< Type of element-wise algorithm to apply */

    DLP_TYPE stor_type; /**< Storage type of alpha and beta values */
} dlp_eltwise_algo_t;

/**
 * @brief Structure defining zero-point parameters for quantization.
 *
 * This structure contains zero-point information used in quantized operations.
 * Zero-point represents the quantized value that corresponds to the real
 * value zero.
 */
/**
 * @struct dlp_zp_t
 * @brief Zero-point parameters for quantization.
 *
 * Contains zero-point values, their length, and type for quantized operations.
 */
typedef struct
{
    void* zero_point;     /**< Pointer to zero-point values */
    md_t  zero_point_len; /**< Length of zero-point array (1 for per-tensor, n
                             for per-channel) */
    DLP_TYPE zero_point_type; /**< Data type of zero-point values */
} dlp_zp_t;

/**
 * @brief Structure defining scale factor parameters for quantization.
 *
 * This structure contains scale factor information used in quantized
 * operations. Scale factor represents the scaling applied during
 * quantization/dequantization.
 */
/**
 * @struct dlp_sf_t
 * @brief Scale factor parameters for quantization.
 *
 * Contains scale factor values, their length, and type for quantized
 * operations.
 */
typedef struct
{
    void* scale_factor;     /**< Pointer to scale factor values */
    md_t  scale_factor_len; /**< Length of scale factor array (1 for per-tensor,
                               n for per-channel) */
    DLP_TYPE scale_factor_type; /**< Data type of scale factor values */
} dlp_sf_t;

/**
 * @brief Structure defining scale operation parameters.
 *
 * This structure contains parameters for scaling operations, which can be
 * applied as post-operations. It uses structured scale factor and zero-point
 * parameters for better organization and type safety.
 */

/**
 * @struct dlp_scale_t
 * @brief Scale operation parameters for post-ops.
 *
 * Contains pointers to scale factor and zero-point parameter structures.
 */
typedef struct
{
    dlp_sf_t* sf; /**< Scale factor parameters */
    dlp_zp_t* zp; /**< Zero-point parameters */
} dlp_scale_t;

/**
 * @brief Structure defining element-wise post-operation parameters.
 *
 * This structure contains parameters for element-wise post-operations
 * such as activation functions applied to the GEMM result.
 */
/**
 * @struct dlp_post_op_eltwise
 * @brief Element-wise post-operation parameters.
 *
 * Contains scale factor and algorithm parameters for element-wise post-ops.
 */
typedef struct
{
    dlp_sf_t*          sf;   /**< Scale factor parameters */
    dlp_eltwise_algo_t algo; /**< Element-wise algorithm parameters */
} dlp_post_op_eltwise;

/**
 * @brief Structure defining bias post-operation parameters.
 *
 * This structure contains parameters for bias addition post-operations,
 * which add a bias vector to the GEMM result.
 */
/**
 * @struct dlp_post_op_bias
 * @brief Bias post-operation parameters.
 *
 * Contains pointer to bias values, their type, and optional scale factor.
 */
typedef struct
{
    void*     bias;      /**< Pointer to bias values */
    DLP_TYPE  stor_type; /**< Storage type of bias values */
    dlp_sf_t* sf;        /**< Scale factor for dequantization */
    dlp_zp_t* zp;        /**< Zero point for dequantization */
    md_t      bias_len;  /**< Length of bias array.*/
} dlp_post_op_bias;

/**
 * @brief Structure defining matrix addition post-operation parameters.
 *
 * This structure contains parameters for matrix addition post-operations,
 * which add another matrix to the GEMM result.
 */
/**
 * @struct dlp_post_op_matrix_add
 * @brief Matrix addition post-operation parameters.
 *
 * Contains pointer to matrix, leading dimension, type, and scale factor for
 * addition.
 */
typedef struct
{
    void*     matrix;    /**< Pointer to matrix to be added */
    md_t      ldm;       /**< Leading dimension of the matrix */
    DLP_TYPE  stor_type; /**< Storage type of matrix values */
    dlp_sf_t* sf;        /**< Scale factor parameters */
} dlp_post_op_matrix_add;

/**
 * @brief Structure defining matrix multiplication post-operation parameters.
 *
 * This structure contains parameters for matrix multiplication post-operations,
 * which multiply the GEMM result with another matrix.
 */
/**
 * @struct dlp_post_op_matrix_mul
 * @brief Matrix multiplication post-operation parameters.
 *
 * Contains pointer to matrix, leading dimension, type, and scale factor for
 * multiplication.
 */
typedef struct
{
    void*     matrix;    /**< Pointer to matrix to be multiplied */
    md_t      ldm;       /**< Leading dimension of the matrix */
    DLP_TYPE  stor_type; /**< Storage type of matrix values */
    dlp_sf_t* sf;        /**< Scale factor parameters */
} dlp_post_op_matrix_mul;

/**
 * @brief Structure defining pre-operation parameters.
 *
 * This structure contains parameters for operations that are applied
 * before the main GEMM computation, typically for quantization adjustments.
 */
/**
 * @struct dlp_pre_op
 * @brief Pre-operation parameters for GEMM.
 *
 * Contains zero-point and scale factor for matrix B, sequence length, and group
 * size.
 */
typedef struct
{
    dlp_zp_t* b_zp;       /**< Zero-point parameters for matrix B */
    dlp_sf_t* b_scl;      /**< Scale factor parameters for matrix B */
    md_t      seq_length; /**< Sequence length for the operation */
    md_t      group_size; /**< Group size for grouped operations */
} dlp_pre_op;

/**
 * @brief Structure defining grouped post-operation parameters.
 *
 * This structure contains parameters for grouped post-operations,
 * which apply different quantization parameters to different groups
 * of the matrices involved in GEMM.
 */
/**
 * @struct dlp_group_post_op
 * @brief Grouped post-operation parameters for GEMM.
 *
 * Contains group size, sequence length, scale factors, and zero-points for
 * matrices A and B.
 */
typedef struct
{
    md_t      group_size; /**< Size of each group for grouped operations */
    md_t      seq_length; /**< Sequence length for the operation */
    dlp_sf_t* a_scl;      /**< Scale factor parameters for matrix A */
    dlp_sf_t* b_scl;      /**< Scale factor parameters for matrix B */
    dlp_zp_t* a_zp;       /**< Zero-point parameters for matrix A */
    dlp_zp_t* b_zp;       /**< Zero-point parameters for matrix B */
} dlp_group_post_op;

/**
 * @brief Quantization operation parameters for a single matrix.
 *
 * This structure defines the quantization/dequantization parameters for a
 * matrix involved in low-precision GEMM operations. It supports both symmetric
 * and asymmetric quantization via scale factors and zero-points.
 *
 * Quantization Formula:
 *   - Symmetric:  q = round(x * scale)
 *   - Asymmetric: q = round(x * scale) - zero_point
 *
 * Dequantization Formula:
 *   - Symmetric:  x = q / scale
 *   - Asymmetric: x = (q + zero_point) / scale
 *
 * Usage Context:
 *   - Can be applied as pre-operation (before GEMM) or post-operation (after
 * GEMM)
 *   - Examples: Converting BF16 to S8
 *   - Supports per-tensor (single value) or per-channel/per-row (array of
 * values) quantization
 *
 * Symmetric vs Asymmetric:
 *   - Symmetric: Zero-point = 0, quantization range is symmetric around zero
 *                Simpler and faster, suitable when data is centered around zero
 *   - Asymmetric: Non-zero zero-point, can represent arbitrary ranges
 *                 More accurate for non-centered distributions, requires
 * additional computation
 */
/**
 * @struct dlp_quant_op
 * @brief Quantization operation parameters.
 *
 * Contains all parameters needed for quantizing or dequantizing a matrix,
 * including scale factors, zero-points, and data type information.
 */
typedef struct
{
    md_t group_size; /**< Size of each group for grouped quantization operations
                      */

    DLP_TYPE src_type; /**< Source data type before quantization (e.g.,
                          DLP_BF16, DLP_F32) */

    DLP_TYPE dst_type; /**< Destination data type after quantization (e.g.,
                          DLP_S8, DLP_U8) */

    dlp_sf_t* scl; /**< Scale factor parameters for quantization/dequantization.
                        Length: 1 for per-tensor, m for per-row/per-channel */

    dlp_zp_t* zp; /**< Zero-point parameters for asymmetric quantization.
                       Set to NULL for symmetric quantization (zero-point = 0).
                       Length: 1 for per-tensor, m for per-row/per-channel */

    bool symmetric; /**< true: Symmetric quantization (zero-point = 0), centered
                         around zero. false: Asymmetric quantization (non-zero
                         zero-point), supports arbitrary value ranges */
} dlp_quant_op;

/**
 * @brief Structure defining symmetric static quantization parameters.
 *
 * This structure contains parameters for symmetric static quantization,
 * where the quantization is performed with symmetric range around zero.
 */
/**
 * @struct DLP_SYMM_STAT_QUANT
 * @brief Symmetric static quantization parameters.
 *
 * Contains group size for symmetric static quantization.
 */
typedef struct
{
    md_t group_size; /**< Group size for grouped quantization */
} DLP_SYMM_STAT_QUANT;

/**
 * @brief Main metadata structure containing all post-operation configurations.
 *
 * This structure serves as the main container for all post-operation metadata,
 * defining the sequence and parameters of operations to be applied after GEMM.
 * It supports multiple post-operations that can be chained together in a
 * specific order.
 */
/**
 * @struct dlp_metadata_t
 * @brief Main metadata structure for post-operation configurations.
 *
 * Contains all post-operation parameters, sequence, and group information for
 * GEMM.
 */
typedef struct
{
    dlp_scale_t* scale;                 /**< Scale post-operations
                                            (multiple allowed) */
    dlp_post_op_eltwise* eltwise;       /**< Element-wise post-operations
                                          (multiple allowed) */
    dlp_post_op_bias*       bias;       /**< Bias addition post-operation */
    dlp_post_op_matrix_add* matrix_add; /**< Matrix addition post-operation */
    dlp_post_op_matrix_mul* matrix_mul; /**< Matrix multiplication
                                             post-operation */

    md_t seq_length; /**< Number of operations in the sequence (e.g., 2) */

    DLP_POST_OP_TYPE* seq_vector; /**< Sequence of post-operations to
                                       apply in order
                                       (e.g., seq_vector[0]=BIAS,
                                       seq_vector[1]=ELTWISE means bias
                                       followed by element-wise operation) */

    // ========== START: DEPRECATED FIELDS ==========
    // TODO: Deprecate these fields as they will be unified in dlp_quant_op
    dlp_pre_op* pre_ops; /**< Pre-operations to be applied before GEMM */

    dlp_group_post_op* post_op_grp; /**< Grouped post-operations for
                                         different quantization groups */
    // ========== END: DEPRECATED FIELDS ==========

    // ========== START: QUANTIZED PARAMETERS ==========
    dlp_quant_op* a_pre_quant; /**< Pre-quantization operations for matrix A
                                    (applied before GEMM computation) */
    md_t a_pre_op_seq_length;  /**< Number of pre-quantization operations for
                                    matrix A */

    dlp_quant_op* b_pre_quant; /**< Pre-quantization operations for matrix B
                                    (applied before GEMM computation) */
    md_t b_pre_op_seq_length;  /**< Number of pre-quantization operations for
                                    matrix B */

    dlp_quant_op* a_post_quant; /**< Post-quantization operations for matrix A
                                     (applied after GEMM computation) */
    md_t a_post_op_seq_length;  /**< Number of post-quantization operations for
                                     matrix A */

    dlp_quant_op* b_post_quant; /**< Post-quantization operations for matrix B
                                     (applied after GEMM computation) */
    md_t b_post_op_seq_length;  /**< Number of post-quantization operations for
                                     matrix B */
    // ========== END: QUANTIZED PARAMETERS ==========

    md_t num_eltwise; /**< Number of element-wise operations to track */

    dlp_error_hndl_t error_hndl; /**< Error handle for the routine, currently
                                      wrapped as part of the metadata. */
} dlp_metadata_t;

#define DLP_METADATA_SET_ERROR(metadata, err_no)                               \
    if ((metadata) != NULL) {                                                  \
        ((metadata)->error_hndl).error_code = err_no;                          \
    }

#endif // DLP_GEMM_POST_OPS_H
