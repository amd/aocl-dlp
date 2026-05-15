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
#include "dlp_gemm_eltwise_ops_interface_apis.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "kernels/dlp_gemm_eltwise_ops_kernels.h"
#include "threading/dlp_gemm_thread_utils.h"

// Kernel function prototypes.
typedef void (*dlp_gemm_util_post_ops_kernel_f32)(const md_t,
                                                  const md_t,
                                                  const float*,
                                                  const md_t,
                                                  const md_t,
                                                  float*,
                                                  const md_t,
                                                  const md_t,
                                                  dlp_gemm_post_op*,
                                                  dlp_gemm_post_op_attr);

DLP_GEMM_ELTWISE_OPS_IFACE(float, float, f32of32)
{
    (void)rntm; /* Threading handled via thread object, not rntm. */
    md_t NR = lcntx->blksz.NR;
    md_t MR = lcntx->blksz.MR;

    dlp_gemm_post_op_attr post_ops_attr;
    post_ops_attr.c_stor_type   = c_downscale;
    post_ops_attr.buf_downscale = NULL;

    // Generate thrinfo objects for jc and ic loops from dlp_gemm_thrinfo_t.
    dlp_task_id_t thread_jc;
    dlp_task_id_t thread_ic;

    dlp_gemm_gen_dlp_task_ids(thread, &thread_jc, &thread_ic);

    // Compute the JC, IC loop thread range for the current thread.
    md_t jc_start, jc_end;
    dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

    md_t ic_start, ic_end;
    dlp_thread_task_range(&thread_ic, m, MR, FALSE, &ic_start, &ic_end);

    post_ops_attr.post_op_c_i       = ic_start;
    post_ops_attr.post_op_c_j       = jc_start;
    post_ops_attr.rs_c_downscale    = rs_b;
    post_ops_attr.cs_c_downscale    = cs_b;
    post_ops_attr.is_first_k        = FALSE;
    post_ops_attr.is_last_k         = TRUE; // Should always be TRUE here.
    post_ops_attr.b_sum_offset      = 0;
    post_ops_attr.b_col_sum_vec     = NULL;
    post_ops_attr.b_col_sum_vec_s16 = NULL;

    // Advance the matrix to the right positions based on thread id.
    // To note that float, bfloat16, int32_t, int8_t and uint8_t are
    // handled using this same frame, so the strides needs to be
    // updated on the actual b matrix datatype or the c_downscale value.
    md_t dsize = sizeof(float);
    if (post_ops_attr.c_stor_type == DLP_BF16) {
        dsize = sizeof(bfloat16);
    }
    if (post_ops_attr.c_stor_type == DLP_S8
        || post_ops_attr.c_stor_type == DLP_U8) {
        dsize = sizeof(int8_t);
    }
    int8_t* b_i = (int8_t*)b;

    ((dlp_gemm_util_post_ops_kernel_f32)(lcntx->eltwise_ops_kern_fun_ptr))(
        (ic_end - ic_start), (jc_end - jc_start),
        a + (rs_a * ic_start) + (cs_a * jc_start), rs_a, cs_a,
        (float*)(b_i + (dsize * ((rs_b * ic_start) + (cs_b * jc_start)))), rs_b,
        cs_b, post_op_list, post_ops_attr);
}
