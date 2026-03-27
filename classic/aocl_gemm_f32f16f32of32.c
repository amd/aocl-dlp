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

/**
 * @file aocl_gemm_f32f16f32of32.c
 * @brief C API entry point for F32×FP16→F32 mixed-precision GEMM
 *
 * A and alpha/beta are F32, B is FP16, C is F32. Accumulation is F32.
 * The JIT micro-kernel converts B from FP16 to F32 internally via vcvtph2ps.
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

void
aocl_gemm_f32f16f32of32(const char      order,
                        const char      transa,
                        const char      transb,
                        const md_t      m,
                        const md_t      n,
                        const md_t      k,
                        const float     alpha,
                        const float*    a,
                        const md_t      lda,
                        const char      mem_format_a,
                        const float16*  b,
                        const md_t      ldb,
                        const char      mem_format_b,
                        const float     beta,
                        float*          c,
                        const md_t      ldc,
                        dlp_metadata_t* metadata)
{
    DLP_GEMM_START_LOGGER();
    DLP_GEMM_WRITE_LOGGER("f32f16f32of32", order, transa, transb, m, n, k,
                          ((float)alpha), lda, mem_format_a, ldb, mem_format_b,
                          ((float)beta), ldc, metadata);

    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    /* F32×FP16→F32 requires:
     *   - AVX-512F (vfmadd231ps for F32 FMA)
     *   - AVX-512BW (vmovdqu16 for FP16 loads)
     *   - F16C (vcvtph2ps for FP16→F32 conversion)
     * On Zen4+, AVX-512 support implies all three are present.
     * Note: Does NOT require AVX-512-FP16 (not available on AMD Zen4/5). */
    if (dlp_cpuid_is_avx512_supported() == FALSE) {
        dlp_print_msg(" AVX-512 ISA not supported by processor, "
                      "cannot perform f32f16f32of32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_GEMM_CHECK("f32f16f32of32", order, transa, transb, m, n, k, a, lda,
                        mem_format_a, b, ldb, mem_format_b, c, ldc, err_no);

    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        goto err_hndl;
    }

    dlp_trans_t dlp_transa;
    dlp_trans_t dlp_transb;
    dlp_param_map_netlib_to_dlp_trans(transa, &dlp_transa);
    dlp_param_map_netlib_to_dlp_trans(transb, &dlp_transb);

    bool is_column_major = ((order == 'c') || (order == 'C'));

    if (is_column_major == TRUE) {
        dlp_print_msg(
            " Column-major layout is not supported for f32f16f32of32 gemm. ",
            __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

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

    /* PackA and ReorderA not supported for F32×FP16→F32 */
    if (mtag_a == REORDERED || mtag_a == PACK) {
        dlp_print_msg(" Packing/reordering of A matrix is not supported for "
                      "f32f16f32of32. "
                      "A must be unpacked.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    /* transA requires packA (as in bf16/fp16/u8s8 APIs), which is not
     * supported for F32×FP16→F32. Reject transA explicitly. */
    if (dlp_is_trans(dlp_transa)) {
        dlp_print_msg(" Transpose of A matrix is not supported for "
                      "f32f16f32of32 (requires packA).",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    /* Transposed B must be packed for correct K-MAJOR access */
    if (dlp_is_trans(dlp_transb) && (mtag_b != REORDERED)) {
        mtag_b = PACK;
    }

    dlp_gemm_post_op post_op_list[AOCL_DLP_MAX_POST_OPS];
    dlp_clsc_err_t   err = dlp_gemm_translate_to_post_ops_list(
        metadata, post_op_list, (void*)c, (void*)(&order), m, n);

    if (err != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }

    /* Row-major only - no swapping needed */
    md_t                m_use      = m;
    md_t                n_use      = n;
    md_t                rs_a_use   = rs_a;
    md_t                cs_a_use   = cs_a;
    md_t                rs_b_use   = rs_b;
    md_t                cs_b_use   = cs_b;
    AOCL_DLP_MEMORY_TAG mtag_a_use = mtag_a;
    AOCL_DLP_MEMORY_TAG mtag_b_use = mtag_b;
    const float*        a_use      = a;
    const float16*      b_use      = b;

    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(F32F16F32OF32);
    dlp_gemm_cntx_t  lcntx_l;
    lcntx_l = *lcntx_g;

    md_t mr_hint = lcntx_l.blksz.MR;
    md_t nr_hint = lcntx_l.blksz.NR;
    md_t kc_hint = lcntx_l.blksz.KC;

    AOCL_DLP_MEMORY_TAG jit_mtag_a = mtag_a_use;
    AOCL_DLP_MEMORY_TAG jit_mtag_b = mtag_b_use;

    /* Alpha and beta are already F32 - pass directly to JIT */
    lcntx_l.dlp_kernel_hndl.kernel_base = NULL;

    dlp_init_and_get_kernel_hndl(
        DLP_KERNEL_F32F16F32OF32, order, jit_mtag_a, jit_mtag_b, m_use, n_use,
        k, rs_a_use, cs_a_use, rs_b_use, cs_b_use, rs_c, cs_c, (void*)&alpha,
        (void*)&beta, post_op_list, mr_hint, nr_hint, kc_hint, DLP_F32,
        &lcntx_l.dlp_kernel_hndl);

    if (lcntx_l.dlp_kernel_hndl.kernel_base == NULL) {
        dlp_print_msg(" F32xFP16 JIT kernel generation failed.", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_INVALID_JIT_KERNEL);
        goto err_hndl;
    }

    dlp_gemm_ops_bundle_t ops = DLP_GEMM_OPS_BUNDLE_INIT_STANDARD(post_op_list);

#ifdef DLP_ENABLE_OPENMP
    dlp_gemm_f32f16f32of32_openmp_thread_decorator(
        m_use, n_use, k, a_use, rs_a_use, cs_a_use, mtag_a_use, b_use, rs_b_use,
        cs_b_use, mtag_b_use, c, rs_c, cs_c, alpha, beta, &rntm_g, &lcntx_l,
        &ops, DLP_F32);
#else
    dlp_gemm_f32f16f32of32_thread_decorator(
        m_use, n_use, k, a_use, rs_a_use, cs_a_use, mtag_a_use, b_use, rs_b_use,
        cs_b_use, mtag_b_use, c, rs_c, cs_c, alpha, beta, &rntm_g, &lcntx_l,
        &ops, DLP_F32);
#endif

err_hndl:;
    DLP_GEMM_STOP_LOGGER();
}
