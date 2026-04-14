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

/**
 * @file dlp_gemm_f32f16_tiny.c
 * @brief Tiny path for F32xFP16->F32 mixed-precision GEMM/GEMV
 *
 * Single-threaded fast path for small matrix sizes, bypassing the full 5-loop
 * GEMM framework overhead (thread decorator, ops bundle, runtime).
 * Runs a simple JR loop calling the JIT micro-kernel directly.
 *
 * A and C are F32, B is FP16. Alpha/beta are F32.
 * Requires AVX-512 (for vcvtph2ps F16->F32 conversion + vfmadd231ps FMA).
 */

#include "config/dlp_gemm_config.h"
#include "dlp_gemm_5loop_interface_apis.h"
#include "dlp_gemm_post_ops.h"
#include "dlp_gemm_types.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "kernels/dlp_gemm_kernels.h"
#include "kernels/f32f16f32/dlp_gemm_pack_f16_f32f16.h"
#include "kernels/fp16fp16fp16/dlp_gemm_pack_fp16.h"
#include "sys_utils/dlp_gemm_sys.h"

#ifdef DLP_KERNELS_ZEN4

/**
 * @brief Tiny GEMV path for F32xFP16->F32.
 *
 * Handles m==1 and n==1 cases. Only uses JIT kernels (no static fallback
 * since F32xFP16 is AVX-512-only and always JIT-generated).
 */
DLP_GEMV_TINY(float, float16, float, f32f16f32of32)
{
    const float* a_use    = (const float*)a;
    md_t         rs_a_use = rs_a;
    md_t         cs_a_use = cs_a;

    float16* b_use    = (float16*)b;
    md_t     rs_b_use = rs_b;
    md_t     cs_b_use = cs_b;

    float16*       pack_b_buffer_fp16 = NULL;
    dlp_clsc_err_t err                = DLP_CLSC_SUCCESS;

    dlp_gemm_post_op_attr post_ops_attr;
    post_ops_attr.c_stor_type       = c_downscale;
    post_ops_attr.rs_c_downscale    = rs_c;
    post_ops_attr.cs_c_downscale    = cs_c;
    post_ops_attr.is_first_k        = TRUE;
    post_ops_attr.is_last_k         = TRUE;
    post_ops_attr.b_sum_offset      = 0;
    post_ops_attr.b_col_sum_vec     = NULL;
    post_ops_attr.b_col_sum_vec_s16 = NULL;

    if (c_downscale < DLP_F32)
        post_ops_attr.buf_downscale = c;
    else
        post_ops_attr.buf_downscale = NULL;

    if (n == 1) {
        /*
         * GEMV N=1: y = A * x  (M x K F32  *  K x 1 FP16  =  M x 1 F32)
         * JIT kernel: MR=16, NR=1, converts FP16 B to F32 internally.
         */

        if (mtag_b == REORDERED) {
            rs_b_use = 1;
            cs_b_use = 1;
        } else if (rs_b != 1) {
            /* Pack strided B vector into contiguous FP16 buffer */
            msz_t mem_b_size_req = sizeof(float16) * k;
            pack_b_buffer_fp16 =
                (float16*)dlp_malloc_page_aligned(mem_b_size_req, &err);

            for (iter_t k0 = 0; k0 < k; k0++) {
                pack_b_buffer_fp16[k0] = b[k0 * rs_b];
            }

            b_use    = pack_b_buffer_fp16;
            rs_b_use = 1;
            cs_b_use = 1;
        }

        post_ops_attr.post_op_c_i = 0;
        post_ops_attr.post_op_c_j = 0;

        /* F32xFP16 always has a JIT kernel (AVX-512 required) */
        dlp_execute_kernel(&(lcntx->dlp_kernel_hndl), m, 1, k, (float*)a_use,
                           rs_a_use, cs_a_use, 1, (float16*)b_use, rs_b_use,
                           cs_b_use, 0, 0, c, rs_c, cs_c, (void*)&alpha,
                           (void*)&beta, post_op_list, post_ops_attr);

        if (pack_b_buffer_fp16 != NULL) {
            dlp_free_page_aligned(pack_b_buffer_fp16);
        }
    } else {
        /*
         * GEMV M=1: y = x * B  (1 x K F32  *  K x N FP16  =  1 x N F32)
         * JIT kernel: MR=1, NR=64, converts FP16 B to F32 internally.
         */
        md_t NR = lcntx->blksz.NR;

        /* Single-threaded: no panel boundary splits */
        md_t n_sub_updated = 0;
        md_t jc_loop_rem   = 0;

        md_t packb_min_NR = get_packb_f32f16f32of32_min_NR();

        if (mtag_b == REORDERED) {
            b_use    = (float16*)b;
            rs_b_use = NR;
            cs_b_use = 1;
        } else if (mtag_b == PACK) {
            md_t  nc0_updated    = dlp_make_multiple_of_n(n, packb_min_NR);
            msz_t mem_b_size_req = sizeof(float16) * nc0_updated * k;

            pack_b_buffer_fp16 =
                (float16*)dlp_malloc_page_aligned(mem_b_size_req, &err);

            n_sub_updated = nc0_updated;

            ((pack_fp16)lcntx->packb_fun_ptr)(pack_b_buffer_fp16, b, rs_b, cs_b,
                                              n, k, &rs_b_use, &cs_b_use);

            rs_b_use = NR;
            cs_b_use = 1;

            b_use = pack_b_buffer_fp16;
        } else {
            b_use = (float16*)b;
        }

        post_ops_attr.post_op_c_i = 0;
        post_ops_attr.post_op_c_j = 0;

        dlp_execute_kernel(&(lcntx->dlp_kernel_hndl), 1, n, k, (float*)a_use,
                           rs_a_use, cs_a_use, 1, (float16*)b_use, rs_b_use,
                           cs_b_use, n_sub_updated, jc_loop_rem, c, rs_c, cs_c,
                           (void*)&alpha, (void*)&beta, post_op_list,
                           post_ops_attr);

        if (pack_b_buffer_fp16 != NULL) {
            dlp_free_page_aligned(pack_b_buffer_fp16);
        }
    }
}

/**
 * @brief Tiny GEMM path for F32xFP16->F32.
 *
 * Bypasses 5-loop framework for small shapes. Runs a simple JR loop
 * over NR-wide column panels, calling the JIT micro-kernel directly.
 *
 * For GEMV cases (m==1 or n==1), delegates to the GEMV tiny path above.
 */
DLP_GEMM_TINY(float, float16, float, f32f16f32of32)
{
    /* Delegate GEMV cases */
    if ((m == 1) || (n == 1)) {
        dlp_gemv_rowvar_tiny_f32f16f32of32(
            m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c, rs_c,
            cs_c, alpha, beta, lcntx, post_op_list, c_downscale);
        return;
    }

    md_t NR = (lcntx->dlp_kernel_hndl.kernel_base != NULL)
                  ? lcntx->dlp_kernel_hndl.nr
                  : lcntx->blksz.NR;
    md_t MR = (lcntx->dlp_kernel_hndl.kernel_base != NULL)
                  ? lcntx->dlp_kernel_hndl.mr
                  : lcntx->blksz.MR;

    const float* a_use    = NULL;
    md_t         rs_a_use = rs_a;
    md_t         cs_a_use = cs_a;

    const float16* b_use    = NULL;
    md_t           rs_b_use = rs_b;
    md_t           cs_b_use = cs_b;

    float16* pack_b_buffer_fp16 = NULL;
    msz_t    mem_b_size_req     = 0;

    md_t ps_a_use;
    md_t ps_b_use;

    md_t packb_min_NR = get_packb_f32f16f32of32_min_NR();

    dlp_gemm_post_op_attr post_ops_attr;
    post_ops_attr.c_stor_type       = c_downscale;
    post_ops_attr.rs_c_downscale    = rs_c;
    post_ops_attr.cs_c_downscale    = cs_c;
    post_ops_attr.b_sum_offset      = 0;
    post_ops_attr.b_col_sum_vec     = NULL;
    post_ops_attr.b_col_sum_vec_s16 = NULL;
    post_ops_attr.post_op_c_i       = 0;
    post_ops_attr.post_op_c_j       = 0;
    post_ops_attr.is_first_k        = TRUE;
    post_ops_attr.is_last_k         = TRUE;

    if (c_downscale < DLP_F32) {
        post_ops_attr.buf_downscale = c;
    } else {
        post_ops_attr.buf_downscale = NULL;
    }

    /* k=1 fast path: single JIT call for the full m x n tile */
    if ((k == 1) && (cs_b == 1)
        && (lcntx->dlp_kernel_hndl.kernel_base != NULL)) {
        dlp_execute_kernel(&(lcntx->dlp_kernel_hndl), m, n, k, (float*)a, rs_a,
                           cs_a, MR * rs_a, (float16*)b, rs_b, cs_b, 0, 0, c,
                           rs_c, cs_c, (void*)&alpha, (void*)&beta,
                           post_op_list, post_ops_attr);
        return;
    }

    /*
     * B matrix setup:
     * - REORDERED: use pre-packed B directly (FP16 NR-wide panels)
     * - PACK + column-major (rs_b==1): pack B into NR-wide FP16 panels
     * - Otherwise: use B as-is (tiny sizes don't benefit from packing)
     */
    if (mtag_b == REORDERED) {
        b_use    = b;
        rs_b_use = NR;
        cs_b_use = 1;
        ps_b_use = k;
    } else if ((mtag_b == PACK) && (rs_b == 1)) {
        md_t nc0_updated = dlp_make_multiple_of_n(n, packb_min_NR);
        mem_b_size_req   = sizeof(float16) * nc0_updated * k;

        dlp_clsc_err_t err = DLP_CLSC_SUCCESS;
        pack_b_buffer_fp16 =
            (float16*)dlp_malloc_page_aligned(mem_b_size_req, &err);

        ((pack_fp16)lcntx->packb_fun_ptr)(pack_b_buffer_fp16, b, rs_b, cs_b, n,
                                          k, &rs_b_use, &cs_b_use);

        rs_b_use = NR;
        cs_b_use = 1;
        ps_b_use = k;

        b_use = pack_b_buffer_fp16;
    } else {
        b_use    = b;
        ps_b_use = cs_b_use;
    }

    /* A matrix setup (always F32, no packing for F32xFP16) */
    a_use    = a;
    ps_a_use = MR * rs_a;

    /* JR loop: iterate over N in NR-wide tiles */
    for (iter_t jr = 0; jr < n; jr += NR) {
        md_t nr0 = dlp_min((n - jr), NR);

        post_ops_attr.post_op_c_i    = 0;
        post_ops_attr.post_op_c_j    = jr;
        post_ops_attr.rs_c_downscale = rs_c;

        if (lcntx->dlp_kernel_hndl.kernel_base != NULL) {
            dlp_execute_kernel(
                &(lcntx->dlp_kernel_hndl), m, nr0, k, (float*)a_use, rs_a_use,
                cs_a_use, ps_a_use, (float16*)(b_use + (jr * ps_b_use)),
                rs_b_use, cs_b_use, 0, 0, (c + jr * cs_c), rs_c, cs_c,
                (void*)&alpha, (void*)&beta, post_op_list, post_ops_attr);
        }
        /* No static kernel fallback — F32xFP16 is always JIT-generated */
    }

    if (pack_b_buffer_fp16 != NULL) {
        dlp_free_page_aligned(pack_b_buffer_fp16);
    }
}

#endif /* DLP_KERNELS_ZEN4 */
