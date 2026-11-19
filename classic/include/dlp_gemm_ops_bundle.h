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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

#ifndef DLP_GEMM_OPS_BUNDLE_H
#define DLP_GEMM_OPS_BUNDLE_H

#include "dlp_gemm_post_ops.h"

/**
 * @brief Unified operations bundle for DLP_GEMM.
 *
 * This structure bundles all operation lists (pre-ops, post-ops, quantization,
 * etc.) into a single parameter for simplified function signatures.
 *
 * Thread Safety: Thread-safe if all pointed-to structures are thread-safe
 * Memory: Typically stack-allocated (24 bytes on 64-bit systems: three
 * pointers) Performance: Negligible overhead (<0.01% for typical GEMMs)
 *
 * @note Unused fields must be set to NULL.
 * @note The structure is designed to be stack-allocated with minimal overhead.
 * @note Fields are ordered by typical usage frequency for cache efficiency.
 */
typedef struct dlp_gemm_ops_bundle_t
{
    dlp_gemm_post_op*
        post_op_list;           ///< Post-operations (most common, all variants)
    dlp_quant_op_t* a_quant_op; ///< A matrix quantization metadata
    dlp_quant_op_t* b_quant_op; ///< B matrix quantization metadata
} dlp_gemm_ops_bundle_t;

/**
 * @brief Initialize ops bundle for standard GEMM (post-ops only)
 * @param post_ops Pointer to post-operations list (can be NULL)
 */
#define DLP_GEMM_OPS_BUNDLE_INIT_STANDARD(post_ops)                            \
    { .post_op_list = (post_ops), .a_quant_op = NULL, .b_quant_op = NULL }

/**
 * @brief Initialize ops bundle for B-side weight-only quantization.
 * @param b_quant Pointer to validated B quantization metadata.
 * @param post_ops Pointer to post-operations list (can be NULL)
 */
#define DLP_GEMM_OPS_BUNDLE_INIT_WOQ(b_quant, post_ops)                        \
    { .post_op_list = (post_ops), .a_quant_op = NULL, .b_quant_op = (b_quant) }

/**
 * @brief A and B quant ops for grouped s8xs8 and s8xs4.
 *
 * The 5-loop reads these pointers and fills the group attribute from them.
 */
#define DLP_GEMM_OPS_BUNDLE_INIT_QUANT_AB(a_quant, b_quant, post_ops)          \
    { .post_op_list = (post_ops),                                              \
      .a_quant_op   = (a_quant),                                               \
      .b_quant_op   = (b_quant) }

/**
 * @brief Initialize ops bundle for quantization variant
 * @param quant_op Pointer to quantization operations (can be NULL)
 * @param post_ops Pointer to post-operations list (can be NULL)
 */
#define DLP_GEMM_OPS_BUNDLE_INIT_QUANT(quant_op, post_ops)                     \
    { .post_op_list = (post_ops), .a_quant_op = (quant_op), .b_quant_op = NULL }

/**
 * @brief Initialize empty ops bundle (no operations)
 */
#define DLP_GEMM_OPS_BUNDLE_INIT_EMPTY()                                       \
    { .post_op_list = NULL, .a_quant_op = NULL, .b_quant_op = NULL }

/**
 * @brief Extract operations from bundle into local non-const variables.
 *
 * Creates local variables with backward-compatible names, minimizing
 * changes inside rowvar functions.
 *
 * @param ops Pointer to dlp_gemm_ops_bundle_t
 *
 * Performance Note: Compiler will optimize these to direct struct access
 * when possible, resulting in zero overhead.
 *
 * Usage:
 * @code
 * void dlp_gemm_rowvar_XXX(..., const dlp_gemm_ops_bundle_t* ops, ...) {
 *     DLP_GEMM_OPS_EXTRACT(ops);
 *
 *     // Use existing code unchanged:
 *     if (post_op_list != NULL) {
 *         apply_post_ops(..., post_op_list);
 *     }
 * }
 * @endcode
 */
#define DLP_GEMM_OPS_EXTRACT(ops)                                              \
    dlp_gemm_post_op* post_op_list = (ops)->post_op_list;                      \
    dlp_quant_op_t*   a_quant_op   = (ops)->a_quant_op;                        \
    dlp_quant_op_t*   b_quant_op   = (ops)->b_quant_op

#endif // DLP_GEMM_OPS_BUNDLE_H
