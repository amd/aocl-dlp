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

#include "s8s8s32/lpgemm_reorder_s8.h"
#include "config/lpgemm_config.h"
#include "gemm_utils/lpgemm_utils.h"
#include "kernels/s8s8s32/lpgemm_packa_s8.h"
#include "kernels/s8s8s32/lpgemm_packb_s8.h"

#ifdef DLP_ENABLE_OPENMP
#include <omp.h>
#endif

void
unreorderb_nr64_s8s8s32os32_reference(lpgemm_obj_t*  b,
                                      lpgemm_obj_t*  b_unreorder,
                                      dlp_rntm_t*    rntm,
                                      lpgemm_cntx_t* lcntx)
{
    md_t NC = lcntx->blksz.NC;
    md_t KC = lcntx->blksz.KC;
    md_t NR = lcntx->blksz.NR;

    // Extracting the matrix properties from the lpgemm object
    md_t rs_b = b->rs;
    md_t cs_b = b->cs;
    md_t n    = b->width;
    md_t k    = b->length;

    md_t k_updated = make_multiple_of_n(k, 4);

    md_t n_threads = rntm->num_threads;
    n_threads      = (n_threads > 0) ? n_threads : 1;

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

            get_B_panel_reordered_start_offset_width(
                jc, n, NC, 16, &jc_cur_loop, &jc_cur_loop_rem, &nc0,
                &n_sub_updated);

            for (iter_t pc = 0; pc < k; pc += KC) {
                md_t kc0 = dlp_min((k - pc), KC);

                // k needs to be a multiple of 4 so that it can be used with
                // dpbf instruction. Padding is added in cases this condition is
                // not satisfied, and therefore the k offset used for
                // packed/reordered buffer needs to be updated.
                md_t kc0_updated = make_multiple_of_n(kc0, 4);

                unpackb_nr64_s8_reference(
                    ((int8_t*)b_unreorder->storage.aligned_buffer)
                        + (jc_cur_loop * k_updated) + (n_sub_updated * pc)
                        + (jc_cur_loop_rem * kc0_updated),
                    (((int8_t*)b->storage.aligned_buffer) + (rs_b * pc)
                     + (jc * cs_b)),
                    nc0, kc0, rs_b, cs_b);
            }

            adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }
}

void
reorderb_nr64_s8s8s32o32(lpgemm_obj_t*  b,
                         lpgemm_obj_t*  b_reorder,
                         dlp_rntm_t*    rntm,
                         lpgemm_cntx_t* lcntx)
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
    md_t k_updated = make_multiple_of_n(k, 4);
    md_t n_updated = make_multiple_of_n(n, 16);

    md_t n_threads = rntm->num_threads;
    n_threads      = (n_threads > 0) ? n_threads : 1;

    int32_t* pack_b_column_sum =
        (int32_t*)(b_reorder->storage.aligned_buffer
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

            get_B_panel_reordered_start_offset_width(
                jc, n, NC, get_packb_s8s8s32o32_min_NR(), &jc_cur_loop,
                &jc_cur_loop_rem, &nc0, &n_sub_updated);

            for (iter_t pc = 0; pc < k; pc += KC) {
                md_t kc0 = dlp_min((k - pc), KC);

                // kc0 needs to be a multiple of 4 so that it can be used with
                // vpdpbusd instruction. Padding is added in cases this
                // condition is not satisfied, and therefore the kc0 offsets
                // used for packed/reordered buffers needs to be updated.
                md_t kc0_updated = make_multiple_of_n(kc0, 4);

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
                ((packb_s32_s8)lcntx->packb_fun_ptr)(
                    (((int8_t*)b_reorder->storage.aligned_buffer)
                     + (jc_cur_loop * k_updated) + (n_sub_updated * pc)
                     + (jc_cur_loop_rem * kc0_updated)),
                    pack_b_column_sum + jc,
                    (((int8_t*)b->storage.aligned_buffer) + (rs_b * pc)
                     + jc * cs_b),
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
reorderb_nr64_s8s8s32o32_sym_quant(lpgemm_obj_t*  b,
                                   lpgemm_obj_t*  b_reorder,
                                   dlp_rntm_t*    rntm,
                                   lpgemm_cntx_t* lcntx,
                                   md_t           group_size)
{

    md_t NC = lcntx->blksz.NC;
    md_t KC = lcntx->blksz.KC;
    md_t NR = lcntx->blksz.NR;

    // Group size should always be <= KC to make sure that entire group is
    // processed within one micro-kernel call. If group size is greater than KC,
    // then KC will be updated to group size. This same change will be done in
    // GEMM 5-loop to maintain consistency between reorder and GEMM execution.
    if (group_size > KC) {
        KC = group_size;
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
    md_t k_updated = make_multiple_of_n(k, 4);
    md_t n_updated = make_multiple_of_n(n, 16);

    md_t n_threads = rntm->num_threads;
    n_threads      = (n_threads > 0) ? n_threads : 1;

    int32_t* pack_b_column_sum =
        (int32_t*)(b_reorder->storage.aligned_buffer
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

            get_B_panel_reordered_start_offset_width(
                jc, n, NC, get_packb_s8s8s32o32_min_NR(), &jc_cur_loop,
                &jc_cur_loop_rem, &nc0, &n_sub_updated);

            for (iter_t pc = 0; pc < k; pc += KC) {
                md_t kc0 = dlp_min((k - pc), KC);

                md_t group_start = pc / group_size;
                md_t group_end   = (pc + kc0 - 1) / group_size;

                // kc0 needs to be a multiple of 4 so that it can be used with
                // vpdpbusd instruction. Padding is added in cases this
                // condition is not satisfied, and therefore the kc0 offsets
                // used for packed/reordered buffers needs to be updated.
                md_t kc0_updated = make_multiple_of_n(kc0, 4);

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
            adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }

    b_reorder->rs   = rs_b_reorder;
    b_reorder->cs   = cs_b_reorder;
    b_reorder->mtag = REORDERED;
}

void
reordera_mr6_s8s8s32o32(lpgemm_obj_t*  a,
                        lpgemm_obj_t*  a_reorder,
                        dlp_rntm_t*    rntm,
                        lpgemm_cntx_t* lcntx)
{
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
        md_t kc0_updated = make_multiple_of_n(kc0, 4);

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
