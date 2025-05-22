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

#include "config/lpgemm_config.h"
#include "gemm_utils/lpgemm_utils.h"
#include "kernels/f32f32f32/lpgemm_pack_f32.h"
#include "kernels/lpgemm_kernels.h"
#include "lpgemm_5loop_interface_apis.h"
#include "lpgemm_post_ops.h"
#include "lpgemm_types.h"
#include "sys_utils/lpgemm_sys.h"

// Kernel function prototypes
typedef void (*lpgemm_rowvar_f32)(const md_t,
                                  const md_t,
                                  const md_t,
                                  const float*,
                                  const md_t,
                                  const md_t,
                                  const md_t,
                                  const float*,
                                  const md_t,
                                  const md_t,
                                  float*,
                                  const md_t,
                                  const md_t,
                                  const float,
                                  const float,
                                  lpgemm_post_op*,
                                  lpgemm_post_op_attr);

typedef void (*lpgemv_n_one_ker_ft)(const md_t,
                                    const md_t,
                                    const float*,
                                    const md_t,
                                    const md_t,
                                    const AOCL_MEMORY_TAG,
                                    const float*,
                                    const md_t,
                                    const md_t,
                                    const AOCL_MEMORY_TAG,
                                    float*,
                                    const md_t,
                                    const md_t,
                                    const float,
                                    const float,
                                    const md_t,
                                    const md_t,
                                    lpgemm_post_op*,
                                    lpgemm_post_op_attr*);

typedef void (*lpgemv_n_one_a_pack_ft)(float*,
                                       const float*,
                                       const md_t,
                                       const md_t,
                                       const md_t,
                                       const md_t,
                                       md_t*,
                                       md_t*);

LPGEMV_TINY(float, float, float, f32f32f32of32)
{
    const float* a_use    = (float*)a;
    md_t         rs_a_use = rs_a;
    md_t         cs_a_use = cs_a;

    float* b_use    = (float*)b;
    md_t   rs_b_use = rs_b;
    md_t   cs_b_use = cs_b;

    lpgemm_post_op_attr post_ops_attr;
    post_ops_attr.c_stor_type = c_downscale;
    if (c_downscale < F32)
        post_ops_attr.buf_downscale = c;
    else
        post_ops_attr.buf_downscale = NULL;

    if (n == 1) {
        float*         pack_a_buffer_f32f32f32of32 = NULL;
        float*         pack_b_buffer_f32f32f32of32 = NULL;
        dlp_clsc_err_t err                         = DLP_CLSC_SUCCESS;

        md_t                   MR;
        lpgemv_n_one_ker_ft    ker_fp;
        lpgemv_n_one_a_pack_ft packa_fp;

        // Workaround to select right kernel and blocksizes based on arch
        // since GEMV parameters are not available in lpgemm context.
#ifdef DLP_KERNELS_ZEN4
        if (lpgemm_get_enabled_arch() == DLP_ARCH_ZEN3) {
            MR       = 16;
            ker_fp   = lpgemv_n_one_f32f32f32of32_avx512_256;
            packa_fp = packa_mr8_f32f32f32of32_col_major;
        } else {
            MR       = 16;
            ker_fp   = lpgemv_n_one_f32f32f32of32;
            packa_fp = packa_mr16_f32f32f32of32_col_major;
        }
#else
        //  Increased MR from 6 to 16 to make use of 32 ZMM registers
        MR       = 8;
        ker_fp   = lpgemv_n_one_f32f32f32of32_avx2;
        packa_fp = packa_mr8_f32f32f32of32_col_major;
#endif

        // Pack B matrix if rs_b > 1
        if ((mtag_b == PACK) && (rs_b != 1)) {
            msz_t mem_b_size_req = sizeof(float) * k;
            pack_b_buffer_f32f32f32of32 =
                (float*)dlp_malloc_page_aligned(mem_b_size_req, &err);

            for (md_t k0 = 0; k0 < k; k0++) {
                pack_b_buffer_f32f32f32of32[k0] = b[k0 * rs_b];
            }

            b_use    = pack_b_buffer_f32f32f32of32;
            rs_b_use = 1;
            cs_b_use = 1;
        }

        if ((mtag_a == PACK) && (cs_a != 1)) {
            msz_t mem_a_size_req = sizeof(float) * m * k;
            pack_a_buffer_f32f32f32of32 =
                (float*)dlp_malloc_page_aligned(mem_a_size_req, &err);

            packa_fp(pack_a_buffer_f32f32f32of32, a_use, rs_a, cs_a, m, k,
                     &rs_a_use, &cs_a_use);
            a_use = pack_a_buffer_f32f32f32of32;
        }

        post_ops_attr.post_op_c_i = 0;
        post_ops_attr.post_op_c_j = 0;
        ker_fp(m, k, a_use, rs_a_use, cs_a_use, mtag_a, b_use, rs_b_use,
               cs_b_use, mtag_b, c, rs_c, cs_c, alpha, beta, MR, k,
               post_op_list, &post_ops_attr);

        if (pack_a_buffer_f32f32f32of32 != NULL) {
            dlp_free_page_aligned(pack_a_buffer_f32f32f32of32);
        }
        if (pack_b_buffer_f32f32f32of32 != NULL) {
            dlp_free_page_aligned(pack_b_buffer_f32f32f32of32);
        }
    }
}

LPGEMM_TINY(float, float, float, f32f32f32of32)
{
#ifdef DLP_KERNELS_ZEN4
    // Handle using LPGEMV when m or/and n equal to 1
    // The avx512 check will be removed when avx2 kernels added in future
    if ((((m == 1) && (lpgemm_get_enabled_arch() != DLP_ARCH_ZEN3)) || (n == 1))
        && (dlp_cpuid_is_avx512_supported() == TRUE)) {
#else
    // m=1 case is not implemented yet for AVX2
    if (((n == 1)) && (dlp_cpuid_is_avx2fma3_supported() == TRUE)) {
#endif
        lpgemv_rowvar_tiny_f32f32f32of32(
            m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c, rs_c,
            cs_c, alpha, beta, lcntx, post_op_list, c_downscale);
        return;
    }

    const md_t NR = lcntx->blksz.NR;
    const md_t MR = lcntx->blksz.MR;

    // Strides are updated based on matrix packing/reordering.
    const float* a_use    = NULL;
    md_t         rs_a_use = rs_a;
    md_t         cs_a_use = cs_a;

    const float* b_use    = NULL;
    md_t         rs_b_use = rs_b;
    md_t         cs_b_use = cs_b;

    // Pack buffer for B.
    float* pack_b_buffer_f32f32f32of32 = NULL;
    msz_t  mem_b_size_req              = 0;

    md_t rs_c_downscale = rs_c;

    md_t ps_a_use;
    md_t ps_b_use;

    const md_t cs_c_use = 1;

    lpgemm_post_op_attr post_ops_attr;
    post_ops_attr.c_stor_type = c_downscale;

    if (c_downscale < F32) {
        post_ops_attr.buf_downscale = c;
    } else {
        post_ops_attr.buf_downscale = NULL;
    }

    bool is_first_k          = TRUE;
    post_ops_attr.is_first_k = is_first_k;
    bool is_last_k           = TRUE;
    post_ops_attr.is_last_k  = is_last_k;

    // Even if the mtag_b is set to PACK, for tiny sizes its better to
    // pack only if it affects output accuracy (like column major B),
    // else ignore it.
    if ((mtag_b == PACK) && (rs_b == 1)) {
        md_t nc0_updated = make_multiple_of_n(n, NR);
        mem_b_size_req   = sizeof(float) * nc0_updated * k;

        dlp_clsc_err_t err = DLP_CLSC_SUCCESS;
        pack_b_buffer_f32f32f32of32 =
            (float*)dlp_malloc_page_aligned(mem_b_size_req, &err);

        ((lpgemm_pack_f32)lcntx->packb_fun_ptr)(pack_b_buffer_f32f32f32of32, b,
                                                rs_b, cs_b, n, k, &rs_b_use,
                                                &cs_b_use);

        rs_b_use = NR;
        cs_b_use = 1;
        ps_b_use = k;

        b_use = pack_b_buffer_f32f32f32of32;
    } else if (mtag_b == REORDERED) {
        b_use    = b;
        rs_b_use = NR;
        cs_b_use = 1;
        ps_b_use = k;
    } else {
        b_use    = b;
        ps_b_use = 1;
    }

    if (mtag_a == REORDERED) {
        a_use    = a;
        rs_a_use = 1;
        cs_a_use = MR;
        ps_a_use = MR * k;
    } else {
        a_use    = a;
        ps_a_use = MR * rs_a;
    }

    for (md_t jr = 0; jr < n; jr += NR) {
        md_t nr0 = dlp_min((n - jr), NR);

        // Post ops meta attributes.
        post_ops_attr.post_op_c_i    = 0;
        post_ops_attr.post_op_c_j    = jr;
        post_ops_attr.rs_c_downscale = rs_c_downscale;

        ((lpgemm_rowvar_f32)lcntx->kern_fun_ptr)(
            m, nr0, k, (float*)a_use, rs_a_use, cs_a_use, ps_a_use,
            (float*)(b_use + (jr * ps_b_use)), rs_b_use, cs_b_use, (c + jr),
            rs_c, cs_c_use, alpha, beta, post_op_list, post_ops_attr);
    }

    if (pack_b_buffer_f32f32f32of32 != NULL) {
        dlp_free_page_aligned(pack_b_buffer_f32f32f32of32);
    }
}
