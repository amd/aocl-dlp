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

#include "fp16fp16fp16/dlp_gemm_reorder_fp16.h"
#include "classic/aocl_fp16_type.h"
#include "config/dlp_gemm_config.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "kernels/fp16fp16fp16/dlp_gemm_pack_fp16.h"

#ifdef DLP_ENABLE_OPENMP
#include <omp.h>
#endif

void
dlp_reorderb_nr128_f16f16f16of16(dlp_gemm_obj_t*  b,
                                 dlp_gemm_obj_t*  b_reorder,
                                 dlp_rntm_t*      rntm,
                                 dlp_gemm_cntx_t* lcntx)
{
    md_t NC = lcntx->blksz.NC;
    md_t KC = lcntx->blksz.KC;
    md_t NR = lcntx->blksz.NR;

    // Extracting the matrix properties from the dlp_gemm object
    md_t rs_b = b->rs;
    md_t cs_b = b->cs;
    md_t n    = b->width;
    md_t k    = b->length;

    md_t rs_b_reorder = rs_b;
    md_t cs_b_reorder = cs_b;

    // FP16 packing factor = 1 (no k-padding, unlike BF16's k%2 requirement)
    // Uses uniform NR=32 stride (one ZMM register = 32 FP16 elements)

    md_t n_threads = rntm->num_threads;
    n_threads      = (n_threads > 0) ? n_threads : 1;

#ifdef DLP_ENABLE_OPENMP
    _Pragma("omp parallel num_threads(n_threads)")
    {
        dlp_task_id_t thread_jc;
        thread_jc.n_way   = n_threads;
        thread_jc.work_id = omp_get_thread_num();
#else
    {
        dlp_task_id_t thread_jc;
        thread_jc.n_way   = 1;
        thread_jc.work_id = 0;
#endif

        md_t jc_start, jc_end;
        dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

        for (iter_t jc = jc_start; jc < jc_end; jc += NC) {
            md_t nc0 = dlp_min((jc_end - jc), NC);

            md_t jc_cur_loop     = jc;
            md_t jc_cur_loop_rem = 0;
            md_t n_sub_updated;

            // Helper function rounds n_sub_updated to multiples of 32
            // packb_min_NR=32 for buffer rounding granularity (one ZMM)
            get_B_panel_reordered_start_offset_width(
                jc, n, NC, 32, &jc_cur_loop, &jc_cur_loop_rem, &nc0,
                &n_sub_updated);

            // Pack KC blocks for this NC panel
            // Note: No JR loop - packer handles subdivision internally (matches
            // BF16/INT8)
            for (iter_t pc = 0; pc < k; pc += KC) {
                md_t kc0 = dlp_min((k - pc), KC);

                // Offset calculation matches BF16/INT8 exactly:
                // - jc_cur_loop * k: Skip complete NC blocks before this one
                // - n_sub_updated * pc: Skip complete KC slices within current
                // NC block
                // - jc_cur_loop_rem * kc0: Skip columns within current KC slice
                md_t offset = (jc_cur_loop * k) + (n_sub_updated * pc)
                              + (jc_cur_loop_rem * kc0);

                // Pass full nc0 to packer (dispatcher selects chunk size
                // internally)
                ((dlp_gemm_pack_fp16)lcntx->packb_fun_ptr)(
                    ((float16*)b_reorder->storage.aligned_buffer) + offset,
                    ((float16*)b->storage.aligned_buffer) + (rs_b * pc)
                        + (jc * cs_b),
                    rs_b, cs_b, nc0, kc0, &rs_b_reorder, &cs_b_reorder);
            }

            adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }

    b_reorder->rs   = rs_b_reorder;
    b_reorder->cs   = cs_b_reorder;
    b_reorder->mtag = REORDERED;
}

void
dlp_unreorderb_nr128_f16f16f16of16(dlp_gemm_obj_t*  b,
                                   dlp_gemm_obj_t*  b_reorder,
                                   dlp_rntm_t*      rntm,
                                   dlp_gemm_cntx_t* lcntx)
{
    md_t NC = lcntx->blksz.NC;
    md_t KC = lcntx->blksz.KC;
    md_t NR = lcntx->blksz.NR;

    // Extracting the matrix properties from the dlp_gemm object
    md_t rs_b = b->rs;
    md_t cs_b = b->cs;
    md_t n    = b->width;
    md_t k    = b->length;

    md_t rs_b_reorder = rs_b;
    md_t cs_b_reorder = cs_b;

    // FP16 packing factor = 1 (no k-padding)
    // Uses uniform NR=32 stride (one ZMM register)

    md_t n_threads = rntm->num_threads;
    n_threads      = (n_threads > 0) ? n_threads : 1;

#ifdef DLP_ENABLE_OPENMP
    _Pragma("omp parallel num_threads(n_threads)")
    {
        dlp_task_id_t thread_jc;
        thread_jc.n_way   = n_threads;
        thread_jc.work_id = omp_get_thread_num();
#else
    {
        dlp_task_id_t thread_jc;
        thread_jc.n_way   = 1;
        thread_jc.work_id = 0;
#endif

        md_t jc_start, jc_end;
        dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

        for (iter_t jc = jc_start; jc < jc_end; jc += NC) {
            md_t nc0 = dlp_min((jc_end - jc), NC);

            md_t jc_cur_loop     = jc;
            md_t jc_cur_loop_rem = 0;
            md_t n_sub_updated;

            // Same helper function as in reorder
            get_B_panel_reordered_start_offset_width(
                jc, n, NC, 32, &jc_cur_loop, &jc_cur_loop_rem, &nc0,
                &n_sub_updated);

            // Unpack KC blocks for this NC panel
            // Note: No JR loop - unpacker handles subdivision internally
            // (matches BF16/INT8)
            for (iter_t pc = 0; pc < k; pc += KC) {
                md_t kc0 = dlp_min((k - pc), KC);

                // Offset calculation matches BF16/INT8 exactly:
                // - jc_cur_loop * k: Skip complete NC blocks before this one
                // - n_sub_updated * pc: Skip complete KC slices within current
                // NC block
                // - jc_cur_loop_rem * kc0: Skip columns within current KC slice
                md_t offset = (jc_cur_loop * k) + (n_sub_updated * pc)
                              + (jc_cur_loop_rem * kc0);

                // Pass full nc0 to unpacker (dispatcher selects chunk size
                // internally)
                ((dlp_gemm_unpack_fp16)lcntx->unpackb_fun_ptr)(
                    ((float16*)b->storage.aligned_buffer) + (rs_b * pc)
                        + (jc * cs_b),
                    ((float16*)b_reorder->storage.aligned_buffer) + offset,
                    rs_b, cs_b, nc0, kc0, &rs_b_reorder, &cs_b_reorder);
            }

            adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }
}
