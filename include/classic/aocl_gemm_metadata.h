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

#ifndef AOCL_DLP_GEMM_METADATA_H
#define AOCL_DLP_GEMM_METADATA_H

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
 * @enum DLP_GLU_ALGO_TYPE
 * @brief Gated Linear Unit (GLU) variants supported as a fused, shape-changing
 * terminal post-op.
 *
 * Every variant operates on an interleaved gate/up tile of width `2I`
 * produced by the GEMM (columns alternate `g, u, g, u, ...`) and emits
 * a tile of width `I`. The split is `gate = X[..., 0::2]`,
 * `up = X[..., 1::2]`, matching vLLM's `swiglu_oai_and_mul` family.
 *
 * Only the P0 variants are listed today. Values are explicitly numbered so
 * future additions never re-number the existing entries.
 */
typedef enum
{
    GATED_SWIGLU         = 0, /**< SiLU(gate) * up (vLLM `silu_and_mul`) */
    GATED_SWIGLU_AND_MUL = 1, /**< gpt-oss clamped GLU (vLLM
                           `swigluoai_and_mul`): with g = min(gate, limit),
                           (g * sigmoid(alpha * g)) * (clip(up, -limit, limit)
                           + 1) */

    DLP_GLU_ALGO_MAX, /**< Sentinel — keep last */
} DLP_GLU_ALGO_TYPE;

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
 * @enum DLP_PARAM_DIM_TYPE
 * @brief Granularity at which a parameter varies: per-tensor, per-channel,
 *        per-token, or per-group.
 */
typedef enum
{
    DLP_PARAM_DIM_INVALID     = 0, /**< Sentinel — unspecified / no parameter */
    DLP_PARAM_DIM_PER_TENSOR  = 1, /**< Scalar / per-tensor (single value) */
    DLP_PARAM_DIM_PER_CHANNEL = 2, /**< N values — one per output column */
    DLP_PARAM_DIM_PER_TOKEN   = 3, /**< M values — one per output row */
    DLP_PARAM_DIM_PER_GROUP   = 4, /**< Values are indexed by quantization
                                      group along the K dimension */
} DLP_PARAM_DIM_TYPE;

/**
 * @enum DLP_QUANT_OP_KIND
 * @brief Quantization operation category described by dlp_quant_op_t.
 *
 * The operation kind also describes where the scale is consumed:
 * - @ref DLP_QUANT_OP_QUANTIZE is used when a source matrix is quantized at
 *   pre-op/pack time and dequantized/corrected after accumulation, or when the
 *   source operands are already quantized and only post-accumulation
 *   dequantization/correction is required. The s8s8 symmetric-quant GEMM path
 *   uses this kind.
 * - @ref DLP_QUANT_OP_DEQUANTIZE is used when low-bit or quantized operands
 *   are expanded/dequantized before accumulation, usually during packing or
 *   pre-processing. WoQ paths such as bf16s4/bf16u4 use this kind.
 * - @ref DLP_QUANT_OP_EXPAND is used for representation expansion without
 *   scale or zero-point semantics.
 */
typedef enum
{
    DLP_QUANT_OP_NONE     = 0,   /**< Sentinel: no quantization operation. */
    DLP_QUANT_OP_QUANTIZE = 1,   /**< Pre-op quantization with
                                    post-accumulation dequantization/correction,
                                    or only post-accumulation
                                    dequantization/correction. Examples:
                                    F32/BF16 A -> S8 with accumulator
                                    correction, and s8s8 symmetric-quant
                                    accumulator scaling. */
    DLP_QUANT_OP_DEQUANTIZE = 2, /**< Pre-op/pack-time dequantization or
                                    expansion using scale factors and optional
                                    zero-points. Examples: bf16s4/bf16u4
                                    weight-only quantization where packed
                                    low-bit B is dequantized while packing or
                                    before compute. */
    DLP_QUANT_OP_EXPAND = 3,     /**< Pure representation expansion/conversion;
                                    no scale or zero-point semantics are
                                    implied. Example: unpacking low-bit data
                                    when no scale/zero-point is consumed. */
} DLP_QUANT_OP_KIND;

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
 * @struct dlp_sf_t
 * @brief Structure defining scale factor parameters for quantization.
 *
 * This structure contains scale factor information used in quantized
 * operations. Scale factor represents the scaling applied during
 * quantization/dequantization.
 *
 * scale_factor_dim contract:
 *   SCALE                            : caller must set explicitly;
 *                                      (dim, len) is validated.
 *                                      PER_TENSOR/1, PER_CHANNEL/n, or
 *                                      PER_TOKEN/m.
 *   BIAS / ELTWISE / MATADD / MATMUL : ignored; dim is inferred from len
 *                                      (1 -> PER_TENSOR, n -> PER_CHANNEL).
 *                                      PER_TOKEN is SCALE-only.
 */
typedef struct
{
    void* scale_factor;         /**< Pointer to scale_factor_len contiguous
                                   elements of scale_factor_type. */
    md_t scale_factor_len;      /**< Number of scale-factor elements. For
                                   SCALE, must match scale_factor_dim
                                   (1/n/m). For other ops, must be 1 or n. */
    DLP_TYPE scale_factor_type; /**< Data type of scale factor values. */
    DLP_PARAM_DIM_TYPE scale_factor_dim; /**< Granularity. Required for the
                                            SCALE post-op only; ignored for
                                            BIAS / ELTWISE / MATADD / MATMUL
                                            (their dim is inferred from
                                            scale_factor_len). See
                                            ::DLP_PARAM_DIM_TYPE. */
} dlp_sf_t;

/**
 * @struct dlp_qparam_t
 * @brief Common scale-factor / zero-point parameter descriptor.
 *
 * Describes one quantization parameter buffer without tying it to a specific
 * operation. The pointed-to data is owned by the caller and must remain valid
 * for the duration of the GEMM or reorder call that consumes the surrounding
 * metadata.
 *
 * The @ref outer_dim field declares how @ref data is indexed:
 * - @ref DLP_PARAM_DIM_PER_TENSOR: one scalar value.
 * - @ref DLP_PARAM_DIM_PER_CHANNEL: one value per output channel/column, or
 *   the B-side grouped layout required by grouped quantization paths.
 * - @ref DLP_PARAM_DIM_PER_TOKEN: one value per input token/row, or the A-side
 *   grouped layout required by grouped quantization paths.
 * - @ref DLP_PARAM_DIM_PER_GROUP: values are arranged by quantization group
 *   along K, optionally combined with the matrix-specific outer dimension
 *   expected by grouped quantization paths.
 */
typedef struct
{
    void* data; /**< Pointer to parameter values; may be NULL only when the
                   parameter itself is absent. */
    md_t len; /**< Number of elements in @ref data. Scalar parameters use 1. */
    DLP_TYPE stor_type; /**< Storage type of each element in @ref data. */
    DLP_PARAM_DIM_TYPE
    outer_dim; /**< Granularity/layout used to index @ref data. */
} dlp_qparam_t;

/**
 * @struct dlp_quant_op_t
 * @brief Unified quantization metadata for one GEMM matrix.
 *
 * This structure describes all quantization metadata associated with either the
 * A or B matrix. The containing @ref dlp_metadata_t field determines which
 * matrix the operation applies to: @ref dlp_metadata_t::a_quant_op for A and
 * @ref dlp_metadata_t::b_quant_op for B.
 *
 * @ref quant_scale_factors is used when the source matrix is quantized before
 * compute (for example F32/BF16 A -> S8). @ref dequant_scale_factors is used
 * when quantized data must be interpreted back in a wider domain (for example
 * S4/U4 B -> BF16 WOQ, or grouped S8xS8 accumulator scaling). A NULL
 * @ref zero_point denotes symmetric quantization; a non-NULL value enables
 * asymmetric compensation where the API supports it.
 * When both quant and dequant scale factors are provided for the same logical
 * operation, callers are expected to provide reciprocal values
 * (quant_scale_factor = 1 / dequant_scale_factor) for matching elements.
 *
 * @ref quant_op_kind makes the intended operation explicit:
 * - @ref DLP_QUANT_OP_QUANTIZE for pre-op quantization with post-accumulation
 *   dequantization/correction, or only post-accumulation dequantization.
 * - @ref DLP_QUANT_OP_DEQUANTIZE for pre-op/pack-time dequantization into a
 *   wider compute type, such as WoQ.
 * - @ref DLP_QUANT_OP_EXPAND for pure low-bit expansion/conversion without
 *   scale/zp semantics.
 *
 * The structure is referenced by pointer from @ref dlp_metadata_t. The caller
 * owns the structure and all nested @ref dlp_qparam_t buffers, and they must
 * remain valid for the duration of the call.
 */
typedef struct
{
    DLP_QUANT_OP_KIND
    quant_op_kind;     /**< Operation kind for this quant metadata. */
    DLP_TYPE src_type; /**< Source element type before quantization or
                          dequantization. */
    DLP_TYPE dst_type; /**< Destination element type after quantization or
                          dequantization. */
    md_t group_size;   /**< Group size along K for grouped quantization. A value
                          of 0 means one group spanning the full K dimension. */
    dlp_qparam_t*
        quant_scale_factors; /**< Scale factors for source-to-quantized
                                conversion; NULL when not applicable. When
                                dequant_scale_factors is also supplied for the
                                same logical operation, matching elements are
                                expected to satisfy quant_scale_factor =
                                1 / dequant_scale_factor. */
    dlp_qparam_t* dequant_scale_factors; /**< Scale factors for interpreting
                                            quantized data in the destination
                                            domain; NULL when not applicable.
                                            When quant_scale_factors is also
                                            supplied for the same logical
                                            operation, matching elements are
                                            expected to satisfy
                                            dequant_scale_factor =
                                            1 / quant_scale_factor. */
    dlp_qparam_t*
        zero_point; /**< Zero-point parameters for asymmetric quantization; NULL
                       for symmetric quantization. */
} dlp_quant_op_t;

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
 * @struct dlp_term_op_glu
 * @brief Gated Linear Unit (GLU) terminal-operation parameters.
 *
 * GLU is a shape-changing, terminal post-op: it folds the `M x 2I` GEMM
 * accumulator (N-axis interleaved `g, u, g, u, ...`) into an `M x I` result.
 * `C` still receives the full raw `M x 2I` accumulator; the folded output is
 * written to the separate, caller-owned `D` buffer (`d`, `ld_d`).
 *
 * Constraints (enforced by the chain validator):
 *   - At most one GLU op per chain, and it must be the last op in the chain.
 *   - `n` (columns of B) must equal `2*I`, hence even (`I` itself may be odd).
 *   - `C` is `M x 2I`, so `ldc >= 2I` is required (else
 *     `DLP_CLSC_INVALID_LEADING_DIMENSION`).
 */
typedef struct
{
    DLP_GLU_ALGO_TYPE algo_type; /**< Which GLU variant to apply. */

    void* d;   /**< Compacted GLU output buffer (MANDATORY). Receives the folded
                    `M x I` result (I = n/2), output-typed (same DLP_TYPE as C),
                    in the caller's storage order. */
    md_t ld_d; /**< Leading dimension of `d`: row stride (>= I) for row-major,
                    column stride (>= m) for column-major. May exceed the
                    minimum for padded/embedded D. */
    /*
     * Scalar convention: neither current variant takes a runtime scalar.
     * GATED_SWIGLU is SiLU(gate)*up; GATED_SWIGLU_AND_MUL is the gpt-oss
     * clamped GLU with its constants (SiLU scale 1.702, clamp 7.0, +1
     * linear-branch bias) baked into the kernel. So `alpha`/`beta`/`stor_type`
     * are ignored — leave them `NULL`/`DLP_INVALID`. The slots remain for
     * future parameterized variants.
     */
    void* alpha; /**< Optional per-variant scalar parameter; no current GLU
                      variant uses it, so leave it `NULL`. */
    void* beta;  /**< Optional per-variant scalar parameter; no current GLU
                      variant uses it, so leave it `NULL`. */

    DLP_TYPE stor_type; /**< Storage type of `alpha`/`beta`; set `DLP_INVALID`
                             while both are `NULL`. */
} dlp_term_op_glu;

/**
 * @struct dlp_gemm_blocking_t
 * @brief Structure defining GEMM blocking parameters.
 */
typedef struct
{
    md_t MR; // Micro-kernel M dimension
    md_t NR; // Micro-kernel N dimension
    md_t MC; // Cache blocking M dimension (multiple of MR)
    md_t NC; // Cache blocking N dimension (multiple of NR)
    md_t KC; // Cache blocking K dimension
} dlp_gemm_blocking_t;

/**
 * @struct dlp_gemm_sup_threshold_t
 * @brief Structure defining GEMM packing thresholds. Only applicable to F32
 * APIs.
 */
typedef struct
{
    md_t MT; /**< M dim threshold to decide whether to enable packing or not */
    md_t NT; /**< N dim threshold to decide whether to enable packing or not */
    md_t KT; /**< K dim threshold to decide whether to enable packing or not */
} dlp_gemm_sup_threshold_t;

/**
 * @struct dlp_gemm_hints_t
 * @brief Structure defining GEMM hints for optimal kernel generation. A value
 * of 0 indicates no hint is provided.
 */
typedef struct
{
    md_t m_hint;  /**< Hint for the M dimension. For instance, in reorder api
                       m_hint=4 implies all the m values that uses this reorder
                       buffer will be 4 and correspondingly the NR value can be
                       modified to take advantage of this. */
    md_t nt_hint; /**< Hint for the number of threads that will be used. For
                      instance, in reorder api nt_hint=32 implies that 32
                      threads will be used in GEMM calls consuming this
                      reordered buffer. Subsequently NR can be modified to be
                      more work distribution friendly based on this nt. */
} dlp_gemm_hints_t;

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

    dlp_quant_op_t* a_quant_op; /**< Optional unified quantization metadata for
                                   matrix A. NULL means no A-side quantization
                                   metadata is supplied. */
    dlp_quant_op_t* b_quant_op; /**< Optional unified quantization metadata for
                                   matrix B. NULL means no B-side quantization
                                   metadata is supplied. */

    md_t num_eltwise; /**< Number of element-wise operations to track */

    dlp_error_hndl_t error_hndl; /**< Error handle for the routine, currently
                                      wrapped as part of the metadata. */
    dlp_gemm_blocking_t*
        block_params; /**< Blocking parameters for GEMM kernels */
    dlp_gemm_sup_threshold_t*
        sup_thresholds; /**< Threshold parameters to decide whether to enable
                            packing or not. Currently only applicable for f32
                            and fp16 APIs only. */
    dlp_gemm_hints_t* gemm_hints; /**< GEMM hints for optimal kernel generation.
                                      A value of 0 indicates no hint is
                                      provided. */
    dlp_term_op_glu* glu; /**< Gated Linear Unit terminal-op (shape-changing
                               2I -> I). When non-NULL, it is applied after all
                               seq_vector post-ops. Only 1 GLU op supported. */
} dlp_metadata_t;

#define DLP_METADATA_SET_ERROR(metadata, err_no)                               \
    if ((metadata) != NULL) {                                                  \
        ((metadata)->error_hndl).error_code = err_no;                          \
    }

#endif // AOCL_DLP_GEMM_METADATA_H
