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

#include "aocl_dlp_gemm_check.h"
#include "classic/aocl_gemm_interface_apis.h"
#include "classic/aocl_lib_interface_apis.h"
#include "classic/dlp_errors.h"
#include "config/dlp_gemm_config.h"
#include "dlp_gemm_5loop_interface_apis.h"
#include "dlp_gemm_ops_bundle.h"
#include "dlp_gemm_post_ops.h"
#include "dlp_gemm_types.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "logging/dlp_gemm_logger.h"
#include "runtime/dlp_runtime.h"
#include "threading/dlp_gemm_thread_decor_openmp.h"

#if defined(__F16C__) && defined(__GNUC__)
#include <immintrin.h>
#endif

/**
 * @brief Convert float32 to float16
 *
 * Uses compiler intrinsics when available (_cvtss_sh with F16C),
 * otherwise falls back to portable software bit manipulation.
 * Uses round-to-nearest-even rounding mode per IEEE-754.
 *
 * The software fallback correctly handles:
 * - Round-to-nearest-even rounding
 * - NaN propagation (preserves quiet NaN)
 * - Subnormal denormalization with proper rounding
 * - Overflow to infinity
 * - Underflow to zero or subnormal
 * - Rounding-induced exponent increment
 */
static inline float16
f32_to_fp16(float f32_val)
{
#if defined(__F16C__) && defined(__GNUC__)
    /* Use F16C intrinsic for hardware conversion */
    return (float16)_cvtss_sh(f32_val, 0);
#else
    /* Software conversion from float32 to float16 with IEEE-754 rounding */
    union
    {
        float    f;
        uint32_t u;
    } x;
    x.f = f32_val;

    /* Extract components */
    uint32_t sign   = (x.u & 0x80000000U) >> 16;   /* Bit 31 → 15 */
    int32_t  exp32  = ((x.u & 0x7F800000U) >> 23); /* Extract exponent */
    uint32_t mant32 = (x.u & 0x007FFFFFU);         /* Extract mantissa */

    /* Special case: FP32 zero or subnormal */
    if (exp32 == 0) {
        /* FP32 subnormals are too small for FP16, flush to signed zero */
        return (float16)(sign);
    }

    /* Special case: FP32 infinity or NaN */
    if (exp32 == 0xFF) {
        if (mant32 == 0) {
            /* Infinity */
            return (float16)(sign | 0x7C00U);
        } else {
            /* NaN: preserve some mantissa bits, ensure quiet NaN */
            uint16_t mant16 = (uint16_t)((mant32 >> 13) | 0x0200U);
            return (float16)(sign | 0x7C00U | (mant16 & 0x03FFU));
        }
    }

    /* Rebias exponent: FP32 bias=127, FP16 bias=15 */
    int32_t exp16 = exp32 - 112; /* exp32 - 127 + 15 = exp32 - 112 */

    /* Add implicit leading 1 to mantissa for calculations */
    mant32 |= 0x00800000U;

    /* Check for underflow (handle denormals) */
    if (exp16 <= 0) {
        if (exp16 < -10) {
            /* Too small, flush to zero */
            return (float16)(sign);
        }

        /*
         * Denormalize: shift mantissa right to align with FP16 denormal format.
         * For FP16 denormals, the value is: mantissa * 2^-24
         * We need to shift the 24-bit mantissa (with implicit 1) right by
         * (14 - exp32 + 127) = (141 - exp32) positions to get the 10-bit
         * result. This is equivalent to shifting by (1 - exp16 + 13) = (14 -
         * exp16).
         */
        int total_shift = 14 - exp16; /* Total shift to get 10-bit mantissa */

        /* Round to nearest even using the bits that will be shifted out */
        uint32_t round_bit   = (mant32 >> (total_shift - 1)) & 1;
        uint32_t sticky_mask = (1U << (total_shift - 1)) - 1;
        uint32_t sticky      = (mant32 & sticky_mask) != 0;
        uint32_t lsb         = (mant32 >> total_shift) & 1;

        /* Compute the shifted mantissa */
        uint32_t mant16 = mant32 >> total_shift;

        /* Apply round-to-nearest-even */
        if (round_bit && (sticky || lsb)) {
            mant16++;
        }

        /* Check if rounding caused normalization (overflow into bit 10) */
        if (mant16 >= 0x0400U) {
            return (float16)(sign | 0x0400U); /* Smallest normal */
        }

        return (float16)(sign | (uint16_t)mant16);
    }

    /* Check for overflow before rounding */
    if (exp16 >= 0x1F) {
        return (float16)(sign | 0x7C00U);
    }

    /* Normal value: Round mantissa from 23 to 10 bits */
    uint32_t round_bits = mant32 & 0x1FFFU; /* Bits 12-0 */
    uint32_t lsb        = (mant32 >> 13) & 1;

    /* Round to nearest even */
    if (round_bits > 0x1000U || (round_bits == 0x1000U && lsb)) {
        mant32 += 0x1000U;
    }

    /* Check for carry into exponent AFTER rounding */
    if (mant32 & 0x01000000U) {
        /* Mantissa overflowed into bit 24 */
        exp16++;
        mant32 = 0x00800000U; /* Reset to implicit 1 only */

        /* Check if exponent overflowed to infinity */
        if (exp16 >= 0x1F) {
            return (float16)(sign | 0x7C00U);
        }
    }

    /* Extract rounded 10-bit mantissa (remove implicit 1) */
    uint16_t mant16 = (uint16_t)((mant32 >> 13) & 0x03FFU);

    return (float16)(sign | ((uint16_t)exp16 << 10) | mant16);
#endif
}

void
aocl_gemm_f16f16f16of16(const char      order,
                        const char      transa,
                        const char      transb,
                        const md_t      m,
                        const md_t      n,
                        const md_t      k,
                        const float     alpha,
                        const float16*  a,
                        const md_t      lda,
                        const char      mem_format_a,
                        const float16*  b,
                        const md_t      ldb,
                        const char      mem_format_b,
                        const float     beta,
                        float16*        c,
                        const md_t      ldc,
                        dlp_metadata_t* metadata)
{
    DLP_GEMM_START_LOGGER();
    DLP_GEMM_WRITE_LOGGER("f16f16f16of16", order, transa, transb, m, n, k,
                          ((float)alpha), lda, mem_format_a, ldb, mem_format_b,
                          ((float)beta), ldc, metadata);

    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    // Check if AVX-512-FP16 ISA is supported, dlp_gemm fp16 matmul requires it.
    if (dlp_cpuid_is_avx512fp16_supported() == FALSE) {
        dlp_print_msg(" AVX-512-FP16 ISA not supported by processor, "
                      "cannot perform f16f16f16of16 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // Initialize dlp_gemm context.
    dlp_init_global_cntx();

    // check for validity of params.
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_GEMM_CHECK("f16f16f16of16", order, transa, transb, m, n, k, a, lda,
                        mem_format_a, b, ldb, mem_format_b, c, ldc, err_no);

    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        goto err_hndl;
    }

    dlp_trans_t dlp_transa;
    dlp_trans_t dlp_transb;
    /* Map BLAS chars to their corresponding DLP enumerated type value. */
    dlp_param_map_netlib_to_dlp_trans(transa, &dlp_transa);
    dlp_param_map_netlib_to_dlp_trans(transb, &dlp_transb);

    bool is_row_major    = ((order == 'r') || (order == 'R'));
    bool is_column_major = ((order == 'c') || (order == 'C'));

    md_t rs_a = lda;
    md_t cs_a = 1;

    if (dlp_is_trans(dlp_transa)) {
        rs_a = 1;
        cs_a = lda;
    }

    md_t rs_b = ldb;
    md_t cs_b = 1;

    if (dlp_is_trans(dlp_transb)) {
        rs_b = 1;
        cs_b = ldb;
    }

    const md_t rs_c = ldc;
    const md_t cs_c = 1;

    AOCL_DLP_MEMORY_TAG mtag_a;
    AOCL_DLP_MEMORY_TAG mtag_b;

    dlp_param_map_char_to_lpmtag(mem_format_a, &mtag_a);
    dlp_param_map_char_to_lpmtag(mem_format_b, &mtag_b);

    // Check for A-dequantization post-op (a_post_quant)
    if ((metadata != NULL) && (metadata->a_post_quant != NULL)) {
        dlp_print_msg(" A-dequantization post-op is not supported for "
                      "f16f16f16of16 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // Check for pre-ops
    if ((metadata != NULL) && (metadata->pre_ops != NULL)
        && (metadata->pre_ops->seq_length > 0)) {
        dlp_print_msg(" Pre-ops are not supported for f16f16f16of16 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // Check for group post-ops
    if ((metadata != NULL) && (metadata->post_op_grp != NULL)) {
        dlp_print_msg(
            " Group post-ops are not supported for f16f16f16of16 gemm.",
            __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // Reordered A not supported now.
    if ((is_row_major == TRUE) && (mtag_a == REORDERED)) {
        dlp_print_msg(" Reordering of A matrix is not supported.", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }
    // Inputs swapped in column major, A becomes B from kernel point of view.
    // Reorder is not supported for column major matrices.
    else if ((is_column_major == TRUE)
             && ((mtag_b == REORDERED) || (mtag_a == REORDERED))) {
        dlp_print_msg(" Reordering of column major matrices is not supported.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // From 5-loop function point of view,
    // B matrix when transposed needs to be packed to ensure correct memory
    // access. The kernel expects B in K-MAJOR layout for optimal performance.
    // When B is transposed but UNPACKED, the strides don't match the expected
    // layout, so we must pack it.
    if ((is_row_major == TRUE) && (dlp_is_trans(dlp_transb))
        && (mtag_b != REORDERED)) {
        mtag_b = PACK;
    }

    // A matrix when in column major storage needs to be packed to row-major
    // storage as kernel expects A matrix to be in row-major format.
    if ((is_row_major == TRUE) && (dlp_is_trans(dlp_transa))) {
        mtag_a = PACK;
    }
    // Inputs swapped in column major, A becomes B from kernel point of view.
    // When transposed in column-major, the matrix needs to be packed.
    if ((is_column_major == TRUE) && (dlp_is_trans(dlp_transb))) {
        mtag_b = PACK;
    }
    if ((is_column_major == TRUE) && (dlp_is_trans(dlp_transa))
        && (mtag_a != REORDERED)) {
        mtag_a = PACK;
    }

    // Force packing for unpacked column-major A.
    // In column-major mode, A (which becomes B after swap) needs to be packed
    // to ensure correct memory access patterns for the row-major kernel.
    if ((is_column_major == TRUE) && (mtag_a == UNPACKED)) {
        mtag_a = PACK;
    }

    // Convert post op struct to post op linked list format.
    dlp_gemm_post_op post_op_list[AOCL_DLP_MAX_POST_OPS];
    dlp_clsc_err_t   err = dlp_gemm_translate_to_post_ops_list(
        metadata, post_op_list, (void*)c, (void*)(&order), m, n);

    if (err != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }

    // Create copy variables to handle column major case
    md_t                m_use      = m;
    md_t                n_use      = n;
    md_t                rs_a_use   = rs_a;
    md_t                cs_a_use   = cs_a;
    md_t                rs_b_use   = rs_b;
    md_t                cs_b_use   = cs_b;
    AOCL_DLP_MEMORY_TAG mtag_a_use = mtag_a;
    AOCL_DLP_MEMORY_TAG mtag_b_use = mtag_b;
    const float16*      a_use      = a;
    const float16*      b_use      = b;

    // Swapping inputs to induce row major computation for column major inputs.
    if (is_column_major == TRUE) {
        m_use      = n;
        n_use      = m;
        rs_a_use   = rs_b;
        cs_a_use   = cs_b;
        rs_b_use   = rs_a;
        cs_b_use   = cs_a;
        mtag_a_use = mtag_b;
        mtag_b_use = mtag_a;
        a_use      = b;
        b_use      = a;
    }

    // Initialize a local runtime with global settings if necessary.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(F16F16F16OF16);
    dlp_gemm_cntx_t  lcntx_l;
    // Create local copy, since each thread in a multi-instance setup
    // modifies the context object.
    lcntx_l = *lcntx_g;

    // Get block size hints for JIT kernel generation
    md_t mr_hint = lcntx_l.blksz.MR;
    md_t nr_hint = lcntx_l.blksz.NR;
    md_t kc_hint = lcntx_l.blksz.KC;

    // Create copy of mtag variables for JIT kernel generation
    AOCL_DLP_MEMORY_TAG jit_mtag_a = mtag_a_use;
    AOCL_DLP_MEMORY_TAG jit_mtag_b = mtag_b_use;

    // Convert alpha and beta from float to float16 for JIT kernel.
    // The FP16 JIT kernel uses vpbroadcastw (16-bit broadcast) to load
    // alpha/beta, so we must pass FP16 addresses.
    float16 alpha_fp16 = f32_to_fp16(alpha);
    float16 beta_fp16  = f32_to_fp16(beta);

    // Initialize DLP Plus kernel path (JIT support)
    lcntx_l.dlp_kernel_hndl.kernel_base = NULL;

    dlp_init_and_get_kernel_hndl(
        DLP_KERNEL_F16F16F16OF16, order, jit_mtag_a, jit_mtag_b, m_use, n_use,
        k, rs_a_use, cs_a_use, rs_b_use, cs_b_use, rs_c, cs_c,
        (void*)&alpha_fp16, (void*)&beta_fp16, post_op_list, mr_hint, nr_hint,
        kc_hint, DLP_F16, &lcntx_l.dlp_kernel_hndl);

    // FP16 is JIT-only (no intrinsic fallback), so check if JIT succeeded
    if (lcntx_l.dlp_kernel_hndl.kernel_base == NULL) {
        dlp_print_msg(" FP16 JIT kernel generation failed.", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // Create ops bundle for standard GEMM (post-ops only)
    dlp_gemm_ops_bundle_t ops = DLP_GEMM_OPS_BUNDLE_INIT_STANDARD(post_op_list);

#ifdef DLP_ENABLE_OPENMP
    dlp_gemm_f16f16f16of16_openmp_thread_decorator(
        m_use, n_use, k, a_use, rs_a_use, cs_a_use, mtag_a_use, b_use, rs_b_use,
        cs_b_use, mtag_b_use, c, rs_c, cs_c, alpha_fp16, beta_fp16, &rntm_g,
        &lcntx_l, &ops, DLP_F16);
#else
    dlp_gemm_f16f16f16of16_thread_decorator(
        m_use, n_use, k, a_use, rs_a_use, cs_a_use, mtag_a_use, b_use, rs_b_use,
        cs_b_use, mtag_b_use, c, rs_c, cs_c, alpha_fp16, beta_fp16, &rntm_g,
        &lcntx_l, &ops, DLP_F16);
#endif

err_hndl:;
    DLP_GEMM_STOP_LOGGER();
}
