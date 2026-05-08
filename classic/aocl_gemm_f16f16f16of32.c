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
#include "classic/aocl_fp16_convert.h"
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

/*
 * FP16xFP16 GEMM with native FP16 accumulation and F32 output.
 *
 * Alpha and beta are float16 to match the of16 prototype so the API stays
 * uniform across both FP16 output rails. The 5-loop widens beta to float
 * once before each kernel call; the JIT post-ops then run the F32
 * beta-combine and F32 in-place store back to user C at full precision.
 * No scratch buffer is allocated for C - the kernel writes directly to
 * the user pointer through a c_downscale = DLP_F32 build-time branch
 * inside the JIT.
 *
 * The decision engine, decorator, and 5-loop are shared with the of16
 * path. The only dispatch differences are c_downscale = DLP_F32 (passed
 * to dlp_init_and_get_kernel_hndl) and the type-aware C-pointer
 * arithmetic the 5-loop performs on the of32 branch.
 */
void
aocl_gemm_f16f16f16of32(const char      order,
                        const char      transa,
                        const char      transb,
                        const md_t      m,
                        const md_t      n,
                        const md_t      k,
                        const float16   alpha,
                        const float16*  a,
                        const md_t      lda,
                        const char      mem_format_a,
                        const float16*  b,
                        const md_t      ldb,
                        const char      mem_format_b,
                        const float16   beta,
                        float*          c,
                        const md_t      ldc,
                        dlp_metadata_t* metadata)
{
    DLP_GEMM_START_LOGGER();
    // alpha/beta arrive as float16 (matching the of16 prototype). The
    // shared logger prints them as %f, so widen once at the call boundary
    // via fp16_to_f32. The widening is for printing only and never
    // propagates back into computation.
    DLP_GEMM_WRITE_LOGGER("f16f16f16of32", order, transa, transb, m, n, k,
                          fp16_to_f32(alpha), lda, mem_format_a, ldb,
                          mem_format_b, fp16_to_f32(beta), ldc, metadata);

    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    if (dlp_cpuid_is_avx512fp16_supported() == FALSE) {
        dlp_print_msg(" AVX-512-FP16 ISA not supported by processor, "
                      "cannot perform f16f16f16of32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_GEMM_CHECK("f16f16f16of32", order, transa, transb, m, n, k, a, lda,
                        mem_format_a, b, ldb, mem_format_b, c, ldc, err_no);

    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        goto err_hndl;
    }

    dlp_trans_t dlp_transa;
    dlp_trans_t dlp_transb;
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

    if ((metadata != NULL) && (metadata->a_post_quant != NULL)) {
        dlp_print_msg(" A-dequantization post-op is not supported for "
                      "f16f16f16of32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    if ((metadata != NULL) && (metadata->pre_ops != NULL)
        && (metadata->pre_ops->seq_length > 0)) {
        dlp_print_msg(" Pre-ops are not supported for f16f16f16of32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    if ((metadata != NULL) && (metadata->post_op_grp != NULL)) {
        dlp_print_msg(
            " Group post-ops are not supported for f16f16f16of32 gemm.",
            __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    if ((is_row_major == TRUE) && (mtag_a == REORDERED)) {
        dlp_print_msg(" Reordering of A matrix is not supported.", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    } else if ((is_column_major == TRUE)
               && ((mtag_b == REORDERED) || (mtag_a == REORDERED))) {
        dlp_print_msg(" Reordering of column major matrices is not supported.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    if ((is_row_major == TRUE) && (dlp_is_trans(dlp_transb))
        && (mtag_b != REORDERED)) {
        mtag_b = PACK;
    }

    if ((is_row_major == TRUE) && (dlp_is_trans(dlp_transa))) {
        mtag_a = PACK;
    }
    if ((is_column_major == TRUE) && (dlp_is_trans(dlp_transb))) {
        mtag_b = PACK;
    }
    if ((is_column_major == TRUE) && (dlp_is_trans(dlp_transa))
        && (mtag_a != REORDERED)) {
        mtag_a = PACK;
    }

    if ((is_column_major == TRUE) && (mtag_a == UNPACKED)) {
        mtag_a = PACK;
    }

    dlp_gemm_post_op post_op_list[AOCL_DLP_MAX_POST_OPS];
    dlp_clsc_err_t   err = dlp_gemm_translate_to_post_ops_list(
        metadata, post_op_list, (void*)c, (void*)(&order), m, n);

    if (err != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }

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

    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    // The of32 path shares its context configuration with of16 since the
    // underlying FP16 accumulator, JIT generator, and block sizes are the
    // same. Only c_downscale differs, keyed in dlp_init_and_get_kernel_hndl
    // and forwarded to the JIT post-ops rail.
    dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(F16F16F16OF16);
    dlp_gemm_cntx_t  lcntx_l;
    lcntx_l = *lcntx_g;

    md_t mr_hint = lcntx_l.blksz.MR;
    md_t nr_hint = lcntx_l.blksz.NR;
    md_t kc_hint = lcntx_l.blksz.KC;

    AOCL_DLP_MEMORY_TAG jit_mtag_a = mtag_a_use;
    AOCL_DLP_MEMORY_TAG jit_mtag_b = mtag_b_use;

    lcntx_l.dlp_kernel_hndl.kernel_base = NULL;

    // alpha/beta are FP16 (matching the of16 prototype). The decision
    // engine reads (void*)&alpha and (void*)&beta as float16* via
    // getScalingTypes<dlp::float16>; the JIT consumes alpha as FP16
    // natively and widens beta to F32 only inside the of32 post-ops rail
    // (vbroadcastss + vfmadd231ps).
    dlp_init_and_get_kernel_hndl(
        DLP_KERNEL_F16F16F16OF32, order, jit_mtag_a, jit_mtag_b, m_use, n_use,
        k, rs_a_use, cs_a_use, rs_b_use, cs_b_use, rs_c, cs_c, (void*)&alpha,
        (void*)&beta, post_op_list, mr_hint, nr_hint, kc_hint, DLP_F32,
        &lcntx_l.dlp_kernel_hndl);

    if (lcntx_l.dlp_kernel_hndl.kernel_base == NULL) {
        dlp_print_msg(" FP16 JIT kernel generation failed.", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    dlp_gemm_ops_bundle_t ops = DLP_GEMM_OPS_BUNDLE_INIT_STANDARD(post_op_list);

    // The decorator signature takes float16* for its C handle to match the
    // shared of16 path. On the of32 branch the 5-loop performs type-aware
    // C-pointer arithmetic (it casts back to float* before each JR offset),
    // and the JIT loads/stores at full precision through the
    // c_downscale = DLP_F32 build-time branch. No FP16 scratch buffer is
    // allocated; user C is written in place every KC.
#ifdef DLP_ENABLE_OPENMP
    dlp_gemm_f16f16f16of16_openmp_thread_decorator(
        m_use, n_use, k, a_use, rs_a_use, cs_a_use, mtag_a_use, b_use, rs_b_use,
        cs_b_use, mtag_b_use, (float16*)c, rs_c, cs_c, alpha, beta, &rntm_g,
        &lcntx_l, &ops, DLP_F32);
#else
    dlp_gemm_f16f16f16of16_thread_decorator(
        m_use, n_use, k, a_use, rs_a_use, cs_a_use, mtag_a_use, b_use, rs_b_use,
        cs_b_use, mtag_b_use, (float16*)c, rs_c, cs_c, alpha, beta, &rntm_g,
        &lcntx_l, &ops, DLP_F32);
#endif

err_hndl:;
    DLP_GEMM_STOP_LOGGER();
}
