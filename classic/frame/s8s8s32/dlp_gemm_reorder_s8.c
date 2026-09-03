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

#include "s8s8s32/dlp_gemm_reorder_s8.h"
#include "config/dlp_gemm_config.h"
#include "gemm_utils/dlp_gemm_reorder_layout.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "kernels/s8s8s32/dlp_gemm_packa_s8.h"
#include "kernels/s8s8s32/dlp_gemm_packb_s8.h"

#ifdef DLP_ENABLE_OPENMP
#include <omp.h>
#endif

void
dlp_reorderb_s8s8s32o32(dlp_gemm_obj_t*  b,
                        dlp_gemm_obj_t*  b_reorder,
                        dlp_rntm_t*      rntm,
                        dlp_gemm_cntx_t* lcntx)
{
    md_t NC = lcntx->blksz.NC;
    md_t KC = lcntx->blksz.KC;
    md_t NR = lcntx->blksz.NR;

    md_t rs_b         = b->rs;
    md_t cs_b         = b->cs;
    md_t rs_b_reorder = rs_b;
    md_t cs_b_reorder = cs_b;

    md_t n = b->width;
    md_t k = b->length;

    // k needs to be a multiple of 4 so that it can be used with vpdpbusd
    // instruction. Padding is added in cases this condition is not
    // satisfied, and therefore the k offset used for packed/reordered
    // buffer needs to be updated.
    md_t k_updated = dlp_make_multiple_of_n(k, 4);
    md_t n_updated = dlp_make_multiple_of_n(n, 16);

    md_t n_threads = rntm->num_threads;
    n_threads      = (n_threads > 0) ? n_threads : 1;

    int32_t* pack_b_column_sum =
        (int32_t*)((int8_t*)b_reorder->storage.aligned_buffer
                   + (sizeof(int8_t) * n_updated * k_updated));

    for (iter_t idx = 0; idx < n_updated; idx++) {
        *(pack_b_column_sum + idx) = 0;
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
        // Compute the JC loop thread range for the current thread.
        md_t jc_start, jc_end;
        dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

        for (iter_t jc = jc_start; jc < jc_end; jc += NC) {
            md_t nc0 = dlp_min((jc_end - jc), NC);

            md_t jc_cur_loop     = jc;
            md_t jc_cur_loop_rem = 0;
            md_t n_sub_updated;

            dlp_gemm_get_B_panel_reordered_start_offset_width(
                jc, n, NC, dlp_get_packb_s8s8s32o32_min_NR(), &jc_cur_loop,
                &jc_cur_loop_rem, &nc0, &n_sub_updated);

            for (iter_t pc = 0; pc < k; pc += KC) {
                md_t kc0 = dlp_min((k - pc), KC);

                // kc0 needs to be a multiple of 4 so that it can be used with
                // vpdpbusd instruction. Padding is added in cases this
                // condition is not satisfied, and therefore the kc0 offsets
                // used for packed/reordered buffers needs to be updated.
                md_t kc0_updated = dlp_make_multiple_of_n(kc0, 4);

                int8_t* pack_dst =
                    (((int8_t*)b_reorder->storage.aligned_buffer)
                     + (jc_cur_loop * k_updated) + (n_sub_updated * pc)
                     + (jc_cur_loop_rem * kc0_updated));
                const int8_t* pack_src = (((int8_t*)b->storage.aligned_buffer)
                                          + (rs_b * pc) + jc * cs_b);

                dlp_execute_packb_kernel(
                    lcntx->dlp_pack_kernel_hndl.pack_b_hndl, (void*)pack_src,
                    (void*)pack_dst, nc0, kc0, rs_b, cs_b, &rs_b_reorder,
                    &cs_b_reorder, pack_b_column_sum + jc);
            }
            dlp_gemm_adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }

    b_reorder->rs   = rs_b_reorder;
    b_reorder->cs   = cs_b_reorder;
    b_reorder->mtag = REORDERED;
}

void
dlp_reorderb_nr64_s8s8s32o32_sym_quant(dlp_gemm_obj_t*  b,
                                       dlp_gemm_obj_t*  b_reorder,
                                       dlp_rntm_t*      rntm,
                                       dlp_gemm_cntx_t* lcntx,
                                       md_t             group_size)
{

    md_t NC = lcntx->blksz.NC;
    md_t KC = lcntx->blksz.KC;
    md_t NR = lcntx->blksz.NR;

    // A quantization group must be processed entirely within one KC block so
    // that its int32 accumulation completes before the group scale is applied;
    // a group that straddles a KC boundary is split across two pc iterations
    // and scaled incorrectly. Keep KC aligned to the group size:
    //  - if group_size > KC, grow KC up to group_size;
    //  - if group_size < KC but does not divide KC, shrink KC down to the
    //    largest multiple of group_size (so KC boundaries fall on group
    //    boundaries).
    // The same adjustment is done in the GEMM 5-loop to keep the reordered-B
    // layout consistent with GEMM execution.
    //
    // NOTE: reorder and GEMM are separate API calls with no channel to hand a
    // chosen KC across, so each side recomputes the SAME adjusted KC here. This
    // is safe only because it is a pure function of (base KC, group_size): base
    // KC is the static S8S8S32OS32 block-size table value (identical for a
    // given arch/config -- already a precondition for reusing reordered B) and
    // group_size comes from the same metadata. If a runtime KC override is ever
    // enabled (dlp_gemm_upd_cntx_with_metadata() is currently a no-op), it MUST
    // be applied to blksz.KC BEFORE this rounding on BOTH sides, or the reorder
    // and GEMM panel boundaries diverge.
    if (group_size > KC) {
        KC = group_size;
    } else if ((KC % group_size) != 0) {
        KC = (KC / group_size) * group_size;
    }

    md_t rs_b         = b->rs;
    md_t cs_b         = b->cs;
    md_t rs_b_reorder = rs_b;
    md_t cs_b_reorder = cs_b;

    md_t n = b->width;
    md_t k = b->length;

    md_t num_groups = (k + group_size - 1) / group_size;

    // k needs to be a multiple of 4 so that it can be used with vpdpbusd
    // instruction. Padding is added in cases this condition is not
    // satisfied, and therefore the k offset used for packed/reordered
    // buffer needs to be updated.
    md_t k_updated = dlp_make_multiple_of_n(k, 4);
    md_t n_updated = dlp_make_multiple_of_n(n, 16);

    md_t n_threads = rntm->num_threads;
    n_threads      = (n_threads > 0) ? n_threads : 1;

    int32_t* pack_b_column_sum =
        (int32_t*)((int8_t*)b_reorder->storage.aligned_buffer
                   + (sizeof(int8_t) * n_updated * k_updated));

    for (iter_t idx = 0; idx < num_groups * n_updated; idx++) {
        *(pack_b_column_sum + idx) = 0;
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
        // Compute the JC loop thread range for the current thread.
        md_t jc_start, jc_end;
        dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

        for (iter_t jc = jc_start; jc < jc_end; jc += NC) {
            md_t nc0 = dlp_min((jc_end - jc), NC);

            md_t jc_cur_loop     = jc;
            md_t jc_cur_loop_rem = 0;
            md_t n_sub_updated   = 0;

            dlp_gemm_get_B_panel_reordered_start_offset_width(
                jc, n, NC, dlp_get_packb_s8s8s32o32_min_NR(), &jc_cur_loop,
                &jc_cur_loop_rem, &nc0, &n_sub_updated);

            for (iter_t pc = 0; pc < k; pc += KC) {
                md_t kc0 = dlp_min((k - pc), KC);

                md_t group_start = pc / group_size;
                md_t group_end   = (pc + kc0 - 1) / group_size;

                // kc0 needs to be a multiple of 4 so that it can be used with
                // vpdpbusd instruction. Padding is added in cases this
                // condition is not satisfied, and therefore the kc0 offsets
                // used for packed/reordered buffers needs to be updated.
                md_t kc0_updated = dlp_make_multiple_of_n(kc0, 4);

                int8_t* b_dst_pc =
                    (((int8_t*)b_reorder->storage.aligned_buffer)
                     + (jc_cur_loop * k_updated) + (n_sub_updated * pc)
                     + (jc_cur_loop_rem * kc0_updated));

                // packing kernels are designed in such a way assuming that
                // entire KCxNC block is packed at once and strides are set
                // based on KC value. In current scenario, we call kernel with
                // blocks of group_size x NC so kernel assumes that KC is
                // group_size and strides are set based on group_size. To avoid
                // this, we are calling kernel with blocks of group_size x NR,
                // so that we can take care of the pointer movement across the
                // reorder buffer in the framework itself.
                for (iter_t jr = 0; jr < nc0; jr += NR) {
                    md_t nr0 = dlp_min((nc0 - jr), NR);

                    int8_t*  b_dst_jr  = b_dst_pc + jr * kc0_updated;
                    int32_t* b_sum_ptr = pack_b_column_sum + jc + jr;
                    int8_t*  b_src_ptr = (((int8_t*)b->storage.aligned_buffer)
                                         + (jc + jr) * cs_b);

                    if (nr0 < NR) {
                        md_t nr_mult_16  = (nr0 / 16) * 16;
                        md_t nr0_rem     = nr0 % 16;
                        md_t nr0_updated = nr_mult_16;

                        if (nr_mult_16 > 0) {
                            // group loop
                            for (iter_t group = group_start; group <= group_end;
                                 group++) {
                                md_t k_start = dlp_max(group * group_size, pc);
                                md_t k_end =
                                    dlp_min(((group + 1) * group_size - 1),
                                            pc + kc0 - 1);
                                md_t kg0 = k_end - k_start + 1;

                                ((packb_s32_s8)lcntx->packb_fun_ptr)(
                                    b_dst_jr
                                        + ((group * group_size) - pc)
                                              * nr0_updated,
                                    b_sum_ptr + (group * n_updated),
                                    b_src_ptr + (rs_b * k_start), rs_b, cs_b,
                                    nr_mult_16, kg0, &rs_b_reorder,
                                    &cs_b_reorder);
                            }
                            b_dst_jr += nr_mult_16 * kc0_updated;
                            b_sum_ptr += nr_mult_16;
                            b_src_ptr += nr_mult_16 * cs_b;
                        }

                        if (nr0_rem > 0) {
                            md_t nr0_updated = 16;
                            // group loop
                            for (iter_t group = group_start; group <= group_end;
                                 group++) {
                                md_t k_start = dlp_max(group * group_size, pc);
                                md_t k_end =
                                    dlp_min(((group + 1) * group_size - 1),
                                            pc + kc0 - 1);
                                md_t kg0 = k_end - k_start + 1;

                                ((packb_s32_s8)lcntx->packb_fun_ptr)(
                                    b_dst_jr
                                        + ((group * group_size) - pc)
                                              * nr0_updated,
                                    b_sum_ptr + (group * n_updated),
                                    b_src_ptr + (rs_b * k_start), rs_b, cs_b,
                                    nr0_rem, kg0, &rs_b_reorder, &cs_b_reorder);
                            }
                        }
                        // no fringe after this point
                        continue;
                    }

                    md_t nr0_updated = NR;
                    // nr0 == NR
                    for (iter_t group = group_start; group <= group_end;
                         group++) {
                        md_t k_start = dlp_max(group * group_size, pc);
                        md_t k_end   = dlp_min(((group + 1) * group_size - 1),
                                               pc + kc0 - 1);
                        md_t kg0     = k_end - k_start + 1;

                        ((packb_s32_s8)lcntx->packb_fun_ptr)(
                            b_dst_jr
                                + ((group * group_size) - pc) * nr0_updated,
                            b_sum_ptr + (group * n_updated),
                            b_src_ptr + (rs_b * k_start), rs_b, cs_b, NR, kg0,
                            &rs_b_reorder, &cs_b_reorder);
                    }
                }
            }
            dlp_gemm_adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }

    b_reorder->rs   = rs_b_reorder;
    b_reorder->cs   = cs_b_reorder;
    b_reorder->mtag = REORDERED;
}

void
dlp_reordera_mr6_s8s8s32o32(dlp_gemm_obj_t*  a,
                            dlp_gemm_obj_t*  a_reorder,
                            dlp_rntm_t*      rntm,
                            dlp_gemm_cntx_t* lcntx)
{
    (void)rntm;
    md_t MC = lcntx->blksz.MC;
    md_t KC = lcntx->blksz.KC;

    md_t rs_a         = a->rs;
    md_t rs_a_reorder = rs_a;
    md_t cs_a_reorder = a->cs;

    md_t k = a->width;
    md_t m = a->length;

    for (iter_t pc = 0; pc < k; pc += KC) {
        md_t kc0 = dlp_min((k - pc), KC);

        // kc0 needs to be a multiple of 4 so that it can be used with
        // vpdpbusd instruction. Padding is added in cases this
        // condition is not satisfied, and therefore the kc0 offsets
        // used for packed/reordered buffers needs to be updated.
        md_t kc0_updated = dlp_make_multiple_of_n(kc0, 4);

        for (iter_t ic = 0; ic < m; ic += MC) {
            md_t mc0 = dlp_min((m - ic), MC);

            ((packa_s32_s8)lcntx->packa_fun_ptr)(
                (((int8_t*)a_reorder->storage.aligned_buffer) + (pc * m)
                 + (ic * kc0_updated)),
                (((int8_t*)a->storage.aligned_buffer) + (rs_a * ic) + pc), rs_a,
                mc0, kc0, &rs_a_reorder, &cs_a_reorder);
        }
    }

    a_reorder->rs   = rs_a_reorder;
    a_reorder->cs   = cs_a_reorder;
    a_reorder->mtag = REORDERED;
}
