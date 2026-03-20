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

#include "config/dlp_gemm_config.h"
#include "dlp_gemm_5loop_interface_apis.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "kernels/bf16bf16f32/dlp_gemm_pack_bf16.h"
#include "kernels/dlp_gemm_kernels.h"
#include "sys_utils/dlp_gemm_sys.h"
#include "threading/dlp_gemm_thread_utils.h"

// Kernel function prototypes
typedef void (*dlp_gemm_rowvar_bf16)(const md_t,
                                     const md_t,
                                     const md_t,
                                     const bfloat16*,
                                     const md_t,
                                     const md_t,
                                     const md_t,
                                     const bfloat16*,
                                     const md_t,
                                     const md_t,
                                     float*,
                                     const md_t,
                                     const md_t,
                                     const float,
                                     const float,
                                     dlp_gemm_post_op*,
                                     dlp_gemm_post_op_attr);

#ifdef DLP_KERNELS_ZEN4
LPGEMV_TINY(bfloat16, bfloat16, float, bf16bf16f32of32)
{
    // Strides are updated based on matrix packing/reordering.
    bfloat16* a_use    = (bfloat16*)a;
    md_t      rs_a_use = rs_a;
    md_t      cs_a_use = cs_a;

    bfloat16* b_use    = (bfloat16*)b;
    md_t      rs_b_use = rs_b;
    md_t      cs_b_use = cs_b;

    dlp_gemm_post_op_attr post_ops_attr;
    post_ops_attr.c_stor_type       = c_downscale;
    post_ops_attr.rs_c_downscale    = rs_c;
    post_ops_attr.cs_c_downscale    = cs_c;
    post_ops_attr.is_first_k        = TRUE;
    post_ops_attr.is_last_k         = TRUE;
    post_ops_attr.b_sum_offset      = 0;
    post_ops_attr.b_col_sum_vec     = NULL;
    post_ops_attr.b_col_sum_vec_s16 = NULL;

    if (c_downscale < DLP_F32) {
        post_ops_attr.buf_downscale = c;
    } else {
        post_ops_attr.buf_downscale = NULL;
    }

    if (n == 1) {
        bfloat16*      pack_a_buffer_bf16 = NULL;
        bfloat16*      pack_b_buffer_bf16 = NULL;
        dlp_clsc_err_t err                = DLP_CLSC_SUCCESS;

        // Increased MR from 6 to 16 to make use of 32 ZMM registers
        md_t MR = 16;

        // pack B matrix if rs_b > 1
        if ((mtag_b == PACK) && (rs_b != 1)) {
            msz_t mem_b_size_req = sizeof(bfloat16) * k;
            pack_b_buffer_bf16 =
                (bfloat16*)dlp_malloc_page_aligned(mem_b_size_req, &err);

            for (iter_t k0 = 0; k0 < k; k0++) {
                pack_b_buffer_bf16[k0] = b[k0 * rs_b];
            }

            b_use    = pack_b_buffer_bf16;
            rs_b_use = 1;
            cs_b_use = 1;
        }

        post_ops_attr.post_op_c_i    = 0;
        post_ops_attr.post_op_c_j    = 0;
        post_ops_attr.rs_c_downscale = rs_c;

        if (mtag_a == PACK) {
            msz_t mem_a_size_req = sizeof(bfloat16) * m * k;
            pack_a_buffer_bf16 =
                (bfloat16*)dlp_malloc_page_aligned(mem_a_size_req, &err);

            ((pack_bf16)lcntx->packa_fun_ptr)(pack_a_buffer_bf16, a, rs_a, cs_a,
                                              m, k, &rs_a_use, &cs_a_use);
            a_use = pack_a_buffer_bf16;
        }
        // Call dlp_gemv_n_one kernel
        if (lcntx->dlp_kernel_hndl.kernel_base != NULL) {
            dlp_execute_kernel(&(lcntx->dlp_kernel_hndl), m, 1, k,
                               (bfloat16*)a_use, rs_a_use, cs_a_use, 1,
                               (bfloat16*)b_use, rs_b_use, cs_b_use, 0, 0, c,
                               rs_c, cs_c, (void*)&alpha, (void*)&beta,
                               post_op_list, post_ops_attr);

        } else {
            dlp_gemv_n_one_bf16bf16f32of32(m, k, a_use, rs_a_use, cs_a_use,
                                           mtag_a, b_use, rs_b_use, cs_b_use,
                                           mtag_b, c, rs_c, cs_c, alpha, beta,
                                           MR, k, post_op_list, &post_ops_attr);
        }

        // Release pack buffers.
        if (pack_a_buffer_bf16 != NULL) {
            dlp_free_page_aligned(pack_a_buffer_bf16);
        }
        if (pack_b_buffer_bf16 != NULL) {
            dlp_free_page_aligned(pack_b_buffer_bf16);
        }
    }
}
#endif

// B should always be packed.
DLP_GEMM_TINY(bfloat16, bfloat16, float, bf16bf16f32of32)
{

#if (defined(DLP_KERNELS_ZEN4) && (!defined(DLP_GEMM_BF16_JIT)))
    // Handle using LPGEMV when m or/and n equal to 1
    // The avx512 check will be removed when avx2 kernels added in future
    if (n == 1) {
        dlp_gemv_rowvar_tiny_bf16bf16f32of32(
            m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c, rs_c,
            cs_c, alpha, beta, lcntx, post_op_list, c_downscale);
        return;
    }
#endif

    md_t NR = lcntx->blksz.NR;

    const int16_t* a_use          = NULL;
    md_t           cs_a_use       = cs_a;
    md_t           rs_a_use       = rs_a;
    md_t           a_block_stride = 0;

    const int16_t* b_use    = NULL;
    md_t           rs_b_use = rs_b;
    md_t           cs_b_use = cs_b;

    md_t rs_c_use       = rs_c;
    md_t rs_c_downscale = rs_c;

    bfloat16*      pack_a_buffer_bf16 = NULL;
    bfloat16*      pack_b_buffer_bf16 = NULL;
    dlp_clsc_err_t err                = DLP_CLSC_SUCCESS;
    msz_t          mem_b_size_req     = 0;
    msz_t          mem_a_size_req     = 0;
    md_t           packb_min_NR       = 16;

    // kc needs to be a multiple of 2 so that it can be used with dpbf16_ps
    // instruction. Padding is added in cases this condition is not
    // satisfied, and therefore the k offset used for packed/reordered
    // buffer needs to be updated.
    md_t k_updated = k;
    k_updated += (k_updated & 0x1);

    dlp_gemm_post_op_attr post_ops_attr;
    post_ops_attr.c_stor_type       = c_downscale;
    post_ops_attr.rs_c_downscale    = rs_c;
    post_ops_attr.cs_c_downscale    = cs_c;
    post_ops_attr.b_sum_offset      = 0;
    post_ops_attr.b_col_sum_vec     = NULL;
    post_ops_attr.b_col_sum_vec_s16 = NULL;

    if (c_downscale < DLP_F32) {
        post_ops_attr.buf_downscale = c;
    } else {
        post_ops_attr.buf_downscale = NULL;
    }

    bool is_first_k          = TRUE;
    post_ops_attr.is_first_k = is_first_k;
    bool is_last_k           = TRUE;
    post_ops_attr.is_last_k  = is_last_k;

    // k needs to be a multiple of 2 so that it can be used with dpbf16_ps
    // instruction. Padding is added in cases this condition is not
    // satisfied, and therefore the k offsets used for packed/reordered
    // buffers needs to be updated.
    md_t k0_updated = k;
    k0_updated += (k0_updated & 0x1);

    if (mtag_b == PACK) {
        // nc0 needs to be a multiple of 16 since this gives maximum
        // vectorization. Packing B always results in buffers with width
        // which is a multiple of 16. Subsequently the nc0 offsets used
        // for packed/reordered buffers needs to be updated.
        md_t nc0_updated = make_multiple_of_n(n, packb_min_NR);
        mem_b_size_req   = sizeof(bfloat16) * nc0_updated * k0_updated;
        pack_b_buffer_bf16 =
            (bfloat16*)dlp_malloc_page_aligned(mem_b_size_req, &err);

        ((pack_bf16)lcntx->packb_fun_ptr)(pack_b_buffer_bf16, b, rs_b, cs_b, n,
                                          k, &rs_b_use, &cs_b_use);

        b_use = pack_b_buffer_bf16;
    } else if (mtag_b == REORDERED) {
        b_use = b;

        dlp_gemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);
    }

    if (mtag_a == UNPACKED) {
        a_use = a;

        // bf16 kernel reads 2 elements, totalling 4 bytes in a
        // single broadcast for use in bf16 instruction.
        // Non bf16 based kernel requires update to this code.
        cs_a_use       = 2;
        a_block_stride = rs_a;
        rs_a_use       = rs_a;
    } else if (mtag_a == PACK) {
        mem_a_size_req = sizeof(bfloat16) * m * k;
        pack_a_buffer_bf16 =
            (bfloat16*)dlp_malloc_page_aligned(mem_a_size_req, &err);

        ((pack_bf16)lcntx->packa_fun_ptr)(pack_a_buffer_bf16, a, rs_a, cs_a, m,
                                          k, &rs_a_use, &cs_a_use);

        a_use          = pack_a_buffer_bf16;
        a_block_stride = rs_a_use;
    }

    for (iter_t jr = 0; jr < n; jr += NR) {
        md_t nr0 = dlp_min((n - jr), NR);

        // Post ops meta attributes.
        post_ops_attr.post_op_c_i    = 0;
        post_ops_attr.post_op_c_j    = jr;
        post_ops_attr.rs_c_downscale = rs_c_downscale;

        // Reorder/Packed B, Reorder/Packed/Unpacked A call.
        if (lcntx->dlp_kernel_hndl.kernel_base != NULL) {
            dlp_execute_kernel(
                &(lcntx->dlp_kernel_hndl), m, nr0, k, (int16_t*)a_use, rs_a_use,
                cs_a_use, a_block_stride, (int16_t*)(b_use + (jr * k0_updated)),
                rs_b_use, cs_b_use, 0, 0, (c + jr), rs_c_use, 1, (void*)&alpha,
                (void*)&beta, post_op_list, post_ops_attr);
        } else {
            ((dlp_gemm_rowvar_bf16)lcntx->kern_fun_ptr)(
                m, nr0, k, a_use, rs_a_use, cs_a_use, a_block_stride,
                (b_use + (jr * k0_updated)), rs_b_use, cs_b_use, (c + jr),
                rs_c_use, 1, alpha, beta, post_op_list, post_ops_attr);
        }
    }

    // Release pack buffers.
    if (pack_a_buffer_bf16 != NULL) {
        dlp_free_page_aligned(pack_a_buffer_bf16);
    }
    if (pack_b_buffer_bf16 != NULL) {
        dlp_free_page_aligned(pack_b_buffer_bf16);
    }
}
