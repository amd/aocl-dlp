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

#include <string.h>

#include "classic/aocl_gemm_interface_apis.h"
#include "config/lpgemm_config.h"
#include "gemm_utils/lpgemm_utils.h"
#include "kernels/f32f32f32/lpgemm_pack_f32.h"
#include "sys_utils/dlp_cpu_arch.h"

#ifdef DLP_ENABLE_OPENMP
#include <omp.h>
#endif

AOCL_GEMM_GET_REORDER_BUF_SIZE(f32f32f32of32)
{
    if ((k <= 0) || (n <= 0)) {
        return 0; // Error.
    }

    // Check if AVX2 ISA is supported, lpgemm fp32 matmul only works with it.
    if (dlp_cpuid_is_avx2fma3_supported() == FALSE) {
        dlp_print_msg(" AVX2 ISA not supported by processor, "
                      "cannot perform f32f32f32 gemm.",
                      __FILE__, __LINE__);
        return 0; // Error.
    }

    // Initialize lpgemm context.
    aocl_lpgemm_init_global_cntx();

    AOCL_MATRIX_TYPE input_mat_type;
    dlp_param_map_char_to_lpmat_type(mat_type, &input_mat_type);

    if (input_mat_type == A_MATRIX) {
        return 0; // A reorder not supported.
    }

    const md_t NR = lpgemm_get_block_size_NR_global_cntx(F32F32F32OF32);

    // Extra space since packing does width in multiples of NR.
    md_t n_reorder;
    if (n == 1) {
        // When n == 1, LPGEMV doesn't expect B to be reordered.
        n_reorder = 1;
    } else {
        n_reorder = ((n + NR - 1) / NR) * NR;
    }

    msz_t size_req = sizeof(float) * k * n_reorder;

    return size_req;
}

// Pack B into row stored column panels.
AOCL_GEMM_REORDER(float, f32f32f32of32)
{
    dlp_trans_t dlp_trans;
    /* Map BLAS chars to their corresponding DLP enumerated type value. */
    dlp_param_map_netlib_to_dlp_trans(trans, &dlp_trans);

    if ((input_buf_addr == NULL) || (reorder_buf_addr == NULL) || (k <= 0)
        || (n <= 0)) {
        return; // Error.
    }

    // Only supports row major packing now.
    md_t rs_b, cs_b;
    if ((order == 'r') || (order == 'R')) {
        if ((dlp_is_notrans(dlp_trans) && (ldb < n))
            || (dlp_is_trans(dlp_trans) && (ldb < k))) {
            return; // Error.
        } else {
            rs_b = dlp_is_notrans(dlp_trans) ? ldb : 1;
            cs_b = dlp_is_notrans(dlp_trans) ? 1 : ldb;
        }
    } else if ((order == 'c') || (order == 'C')) {
        if ((dlp_is_notrans(dlp_trans) && (ldb < k))
            || (dlp_is_trans(dlp_trans) && (ldb < n))) {
            return; // Error.
        } else {
            rs_b = dlp_is_notrans(dlp_trans) ? 1 : ldb;
            cs_b = dlp_is_notrans(dlp_trans) ? ldb : 1;
        }
    } else {
        return; // Error
    }

    // Check if AVX2 ISA is supported, lpgemm fp32 matmul only works with it.
    if (dlp_cpuid_is_avx2fma3_supported() == FALSE) {
        dlp_print_msg(" AVX2 ISA not supported by processor, "
                      "cannot perform f32f32f32 gemm.",
                      __FILE__, __LINE__);
        return; // Error.
    }

    // Initialize lpgemm context.
    aocl_lpgemm_init_global_cntx();

    AOCL_MATRIX_TYPE input_mat_type;
    dlp_param_map_char_to_lpmat_type(mat_type, &input_mat_type);

    if (input_mat_type == A_MATRIX) {
        return; // A reorder not supported.
    }

    // Query the context for various blocksizes.
    lpgemm_cntx_t* lcntx = lpgemm_get_global_cntx_obj(F32F32F32OF32);
    md_t           NC    = lcntx->blksz.NC;
    md_t           KC    = lcntx->blksz.KC;
    md_t           NR    = lcntx->blksz.NR;

    md_t rs_b_reorder = 0;
    md_t cs_b_reorder = 0;

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    md_t n_threads = rntm_g.num_threads;
    n_threads      = (n_threads > 0) ? n_threads : 1;

    // When n == 1, B marix becomes a vector.
    // Reordering is avoided so that LPGEMV can process it efficiently.
    if (n == 1) {
        if (rs_b == 1) {
            memcpy(reorder_buf_addr, input_buf_addr, (k * sizeof(float)));
        } else {
            for (md_t k0 = 0; k0 < k; k0++) {
                reorder_buf_addr[k0] = input_buf_addr[k0 * rs_b];
            }
        }
        return;
    }

#ifdef DLP_ENABLE_OPENMP
    _Pragma("omp parallel num_threads(n_threads)")
    {
        // Initialise a local thrinfo obj for work split across threads.
        dlp_task_id_t thread_jc;
        thread_jc.n_way   = n_threads;
        thread_jc.work_id = omp_get_thread_num();
#else
    {
        // Initialise a local thrinfo obj for work split across threads.
        dlp_task_id_t thread_jc;
        thread_jc.n_way   = 1;
        thread_jc.work_id = 0;
#endif
        // Compute the JC loop thread range for the current thread. Per thread
        // gets multiple of NR columns.
        md_t jc_start, jc_end;
        dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);
        for (md_t jc = jc_start; jc < jc_end; jc += NC) {
            md_t nc0 = dlp_min((jc_end - jc), NC);

            md_t jc_cur_loop     = jc;
            md_t jc_cur_loop_rem = 0;
            md_t n_sub_updated;

            get_B_panel_reordered_start_offset_width(
                jc, n, NC, NR, &jc_cur_loop, &jc_cur_loop_rem, &nc0,
                &n_sub_updated);

            for (md_t pc = 0; pc < k; pc += KC) {
                md_t kc0 = dlp_min((k - pc), KC);

                // The offsets are calculated in such a way that it resembles
                // the reorder buffer traversal in single threaded reordering.
                // The panel boundaries (KCxNC) remain as it is accessed in
                // single thread, and as a consequence a thread with jc_start
                // inside the panel cannot consider NC range for reorder. It
                // has to work with NC' < NC, and the offset is calulated using
                // prev NC panels spanning k dim + cur NC panel spaning pc loop
                // cur iteration + (NC - NC') spanning current kc0 (<= KC).
                //
                // Eg: Consider the following reordered buffer diagram:
                //          t1              t2
                //          |               |
                //          |           |..NC..|
                //          |           |      |
                //          |.NC. |.NC. |NC'|NC"
                //     pc=0-+-----+-----+---+--+
                //        KC|     |     |   |  |
                //          |  1  |  3  |   5  |
                //    pc=KC-+-----+-----+---st-+
                //        KC|     |     |   |  |
                //          |  2  |  4  | 6 | 7|
                // pc=k=2KC-+-----+-----+---+--+
                //          |jc=0 |jc=NC|jc=2NC|
                //
                // The numbers 1,2..6,7 denotes the order in which reordered
                // KCxNC blocks are stored in memory, ie: block 1 followed by 2
                // followed by 3, etc. Given two threads t1 and t2, and t2 needs
                // to acces point st in the reorder buffer to write the data:
                // The offset calulation logic will be:
                // jc_cur_loop = 2NC, jc_cur_loop_rem = NC', pc = KC,
                // n_sub_updated = NC, k = 2KC, kc0_updated = KC
                //
                // st = ( jc_cur_loop * k )    <traverse blocks 1,2,3,4>
                //    + ( n_sub_updated * pc ) <traverse block 5>
                //    + ( NC' * kc0_updated)   <traverse block 6>
                ((lpgemm_pack_f32)lcntx->packb_fun_ptr)(
                    reorder_buf_addr + (jc_cur_loop * k) + (n_sub_updated * pc)
                        + (jc_cur_loop_rem * kc0),
                    input_buf_addr + (rs_b * pc) + (cs_b * jc), rs_b, cs_b, nc0,
                    kc0, &rs_b_reorder, &cs_b_reorder);
            }

            adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }
}
