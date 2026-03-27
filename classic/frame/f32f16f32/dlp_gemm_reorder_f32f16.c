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

#include "f32f16f32/dlp_gemm_reorder_f32f16.h"
#include "classic/aocl_fp16_type.h"
#include "config/dlp_gemm_config.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "kernels/f32f16f32/dlp_gemm_pack_f16_f32f16.h"
#include "kernels/fp16fp16fp16/dlp_gemm_pack_fp16.h"

#ifdef DLP_ENABLE_OPENMP
#include <omp.h>
#endif

void
reorderb_nr64_f32f16f32of32(dlp_gemm_obj_t*  b,
                            dlp_gemm_obj_t*  b_reorder,
                            dlp_rntm_t*      rntm,
                            dlp_gemm_cntx_t* lcntx)
{
    md_t NC = lcntx->blksz.NC;
    md_t KC = lcntx->blksz.KC;
    md_t NR = lcntx->blksz.NR;

    md_t rs_b = b->rs;
    md_t cs_b = b->cs;
    md_t n    = b->width;
    md_t k    = b->length;

    md_t rs_b_reorder = rs_b;
    md_t cs_b_reorder = cs_b;

    md_t packb_min_NR = get_packb_f32f16f32of32_min_NR();

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

            dlp_gemm_get_B_panel_reordered_start_offset_width(
                jc, n, NC, packb_min_NR, &jc_cur_loop, &jc_cur_loop_rem, &nc0,
                &n_sub_updated);

            for (iter_t pc = 0; pc < k; pc += KC) {
                md_t kc0 = dlp_min((k - pc), KC);

                md_t offset = (jc_cur_loop * k) + (n_sub_updated * pc)
                              + (jc_cur_loop_rem * kc0);

                ((dlp_gemm_pack_fp16)lcntx->packb_fun_ptr)(
                    ((float16*)b_reorder->storage.aligned_buffer) + offset,
                    ((float16*)b->storage.aligned_buffer) + (rs_b * pc)
                        + (jc * cs_b),
                    rs_b, cs_b, nc0, kc0, &rs_b_reorder, &cs_b_reorder);
            }

            dlp_gemm_adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }

    b_reorder->rs   = rs_b_reorder;
    b_reorder->cs   = cs_b_reorder;
    b_reorder->mtag = REORDERED;
}
