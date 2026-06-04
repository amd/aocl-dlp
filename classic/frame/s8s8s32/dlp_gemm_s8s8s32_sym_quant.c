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
#include "kernels/dlp_gemm_kernels.h"
#include "kernels/s8s8s32/dlp_gemm_packa_s8.h"
#include "kernels/s8s8s32/dlp_gemm_packb_s8.h"
#include "kernels/u8s8s32/dlp_gemm_packa.h"
#include "sys_utils/dlp_gemm_sys.h"
#include "threading/dlp_gemm_thread_utils.h"

// NOTE
// 1. Mandatory for matrix B to be reordered, i.e., mtag_b == REORDERED.
// 2. K should be divisible by group_size.
#ifdef DLP_KERNELS_ZEN4
DLP_GEMV2(int8_t, int8_t, int32_t, s8s8s32o32_sym_quant)
{
    (void)rntm; /* Threading handled via thread object, not rntm. */
    md_t NC = lcntx->blksz.NC;
    md_t KC = lcntx->blksz.KC;
    md_t MC = lcntx->blksz.MC;
    md_t NR = lcntx->blksz.NR;

    // Group size should always be <= KC to make sure that entire group is
    // processed within one micro-kernel call. If group size is greater than KC,
    // then KC will be updated to group size. This same change is done in
    // reorder function to maintain consistency between reorder and GEMM
    // execution.
    if (grp_post_op_list->group_size > KC) {
        KC = grp_post_op_list->group_size;
    }

    // Strides are updated based on matrix packing/reordering.
    int8_t* a_use    = (int8_t*)a;
    md_t    rs_a_use = rs_a;
    md_t    cs_a_use = cs_a;

    int8_t* b_use    = (int8_t*)b;
    md_t    rs_b_use = rs_b;
    md_t    cs_b_use = cs_b;

    float* c_use = NULL;

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

    msz_t mem_a_size_req = 0;

    int8_t* pack_a_buffer_s8s8s32os32 = NULL;

    dlp_gemm_grp_post_op_attr grp_post_ops_attr;

    md_t group_size = grp_post_op_list->group_size;

    // Initialize group post ops attributes.
    grp_post_ops_attr.a_scale_factor     = grp_post_op_list->a_scale_factor;
    grp_post_ops_attr.a_scale_factor_len = grp_post_op_list->a_scale_factor_len;
    grp_post_ops_attr.b_scale_factor     = grp_post_op_list->b_scale_factor;
    grp_post_ops_attr.b_scale_factor_len = grp_post_op_list->b_scale_factor_len;
    grp_post_ops_attr.a_zp               = grp_post_op_list->a_zp;
    grp_post_ops_attr.b_zp               = grp_post_op_list->b_zp;
    grp_post_ops_attr.a_zp_len           = grp_post_op_list->a_zp_len;
    grp_post_ops_attr.b_zp_len           = grp_post_op_list->b_zp_len;
    grp_post_ops_attr.group_size         = group_size;
    grp_post_ops_attr.sf_stor_type       = grp_post_op_list->sf_stor_type;
    grp_post_ops_attr.zp_stor_type       = grp_post_op_list->zp_stor_type;

    md_t num_groups                   = (k + group_size - 1) / group_size;
    grp_post_ops_attr.grp_post_op_lda = num_groups;
    grp_post_ops_attr.grp_post_op_ldb = n;

    // Generate thrinfo objects for jc and ic loops from dlp_gemm_thrinfo_t.
    dlp_task_id_t thread_jc;
    dlp_task_id_t thread_ic;

    dlp_gemm_gen_dlp_task_ids(thread, &thread_jc, &thread_ic);

    if (n == 1) {
        // Increased MR from 6 to 16 to make use of 32 ZMM registers
        md_t MR = 16;

        if (mtag_b == REORDERED) {
            post_ops_attr.b_col_sum_vec = (int32_t*)(b + k);
        } else if (mtag_b == PACK) {
            // Unreordered B not supported.
            return;
        } else {
            // Unpacked B not supported.
            return;
        }

        // Compute the IC loop thread range for the current thread.
        md_t ic_start, ic_end;
        thread_ic.n_way   = (thread_ic.n_way == 1) ? (thread->n_threads)
                                                   : (thread_ic.n_way);
        thread_ic.work_id = thread->tid;
        dlp_thread_task_range(&thread_ic, m, MR, FALSE, &ic_start, &ic_end);

        grp_post_ops_attr.grp_post_op_k = 0;
        for (iter_t ic = ic_start; ic < ic_end; ic += MC) {
            grp_post_ops_attr.grp_post_op_i = ic;

            md_t mc0 = dlp_min((ic_end - ic), MC);

            const int8_t* a_use = a + ic * rs_a;
            c_use               = c + ic * rs_c;

            post_ops_attr.post_op_c_i    = ic;
            post_ops_attr.post_op_c_j    = 0;
            post_ops_attr.rs_c_downscale = rs_c;

            if (mtag_a == PACK) {
                mem_a_size_req = sizeof(int8_t) * mc0 * k;

                if (pack_a_buffer_s8s8s32os32 == NULL) {
                    dlp_clsc_err_t ret_err;
                    pack_a_buffer_s8s8s32os32 =
                        dlp_malloc_page_aligned(mem_a_size_req, &ret_err);
                }

                ((packa_s32)lcntx->packa_fun_ptr)(
                    (uint8_t*)pack_a_buffer_s8s8s32os32,
                    (uint8_t*)(a + (rs_a * ic)), rs_a, cs_a, mc0, k, &rs_a_use,
                    &cs_a_use);
                a_use = pack_a_buffer_s8s8s32os32;
            }

            // Call dlp_gemv_n_one kernel
            dlp_gemv_n_one_s8s8s32os32_sym_quant(
                mc0, k, a_use, rs_a_use, cs_a_use, mtag_a, b_use, rs_b_use,
                cs_b_use, mtag_b, c_use, rs_c, cs_c, alpha, beta, MR, KC,
                grp_post_ops_attr, post_op_list, &post_ops_attr);
        }

        // Release pack buffers
        if (mtag_a == PACK && (pack_a_buffer_s8s8s32os32 != NULL)) {
            dlp_free_page_aligned(pack_a_buffer_s8s8s32os32);
        }
    } else {
        md_t gemm_MR = lcntx->blksz.MR;

        md_t jc_start, jc_end;
        thread_jc.n_way   = (thread_jc.n_way == 1) ? (thread->n_threads)
                                                   : (thread_jc.n_way);
        thread_jc.work_id = thread->tid;
        dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

        md_t packb_min_NR = dlp_get_packb_s8s8s32o32_min_NR();

        // kc needs to be a multiple of 4 so that it can be used with vpdpbusd
        // instruction. Padding is added in cases this condition is not
        // satisfied, and therefore the k offset used for packed/reordered
        // buffer needs to be updated.
        md_t k_updated = dlp_make_multiple_of_n(k, 4);
        md_t n_updated = dlp_make_multiple_of_n(n, 16);

        rs_a_use = rs_a;
        cs_a_use = 4;

        if (mtag_a == PACK) {
            mem_a_size_req = sizeof(uint8_t) * k;

            if (pack_a_buffer_s8s8s32os32 == NULL) {
                dlp_clsc_err_t ret_err;
                pack_a_buffer_s8s8s32os32 =
                    dlp_malloc_page_aligned(mem_a_size_req, &ret_err);
            }

            ((packa_s32)lcntx->packa_fun_ptr)(
                (uint8_t*)pack_a_buffer_s8s8s32os32, (uint8_t*)a, rs_a, cs_a, 1,
                k, &rs_a_use, &cs_a_use);

            dlp_get_packa_strides_mfringe_u8s8s32os32(rs_a, cs_a, &rs_a_use,
                                                      &cs_a_use, gemm_MR, 1);

            a_use = pack_a_buffer_s8s8s32os32;
        }

        grp_post_ops_attr.grp_post_op_k = 0;
        for (iter_t jc = jc_start; jc < jc_end; jc += NC) {
            grp_post_ops_attr.grp_post_op_j = jc;

            md_t nc0 = dlp_min((jc_end - jc), NC);
            c_use    = c + jc;

            md_t jc_cur_loop     = jc;
            md_t jc_cur_loop_rem = 0;
            md_t n_sub_updated   = 0;

            if (mtag_b == REORDERED) {
                dlp_gemm_get_B_panel_reordered_start_offset_width(
                    jc, n, NC, packb_min_NR, &jc_cur_loop, &jc_cur_loop_rem,
                    &nc0, &n_sub_updated);

                b_use = (int8_t*)(b + (jc_cur_loop * k_updated));

                dlp_gemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);

                post_ops_attr.b_col_sum_vec =
                    ((int32_t*)(b + (k_updated * n_updated))) + jc;

                grp_post_ops_attr.grp_post_op_sum_ld = n_updated;
            } else if (mtag_b == PACK) {
                // Unreordered B not supported.
                return;
            } else {
                // Unpacked B not supported.
                return;
            }

            post_ops_attr.post_op_c_i    = 0;
            post_ops_attr.post_op_c_j    = jc;
            post_ops_attr.rs_c_downscale = rs_c;
            post_ops_attr.b_sum_offset   = 0;

            dlp_gemv_m_one_s8s8s32os32_sym_quant(
                nc0, k, a_use, rs_a_use, cs_a_use, mtag_a, b_use, rs_b_use,
                cs_b_use, mtag_b, c_use, rs_c, cs_c, alpha, beta, NR, KC,
                n_sub_updated, jc_cur_loop_rem, grp_post_ops_attr, post_op_list,
                &post_ops_attr);

            if (mtag_b == REORDERED) {
                dlp_gemm_adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
            }
        } // jc loop

        // Release pack buffers.
        if ((mtag_a == PACK) && (pack_a_buffer_s8s8s32os32 != NULL)) {
            dlp_free_page_aligned(pack_a_buffer_s8s8s32os32);
        }
    }
}
#endif

// B should always be packed.
DLP_GEMM_5LOOP_UNIFIED(
    int8_t, int8_t, int32_t, float, s8s8s32o32_sym_quant, const)
{
    // Extract operations from bundle into local variables
    DLP_GEMM_OPS_EXTRACT(ops);

    md_t NC = lcntx->blksz.NC;
    md_t KC = lcntx->blksz.KC;
    md_t MC = lcntx->blksz.MC;
    md_t NR = lcntx->blksz.NR;
    md_t MR = lcntx->blksz.MR;

    if (mtag_b == UNPACKED) {
        // Error: can only work with packed B now.
        return;
    }

#ifdef DLP_KERNELS_ZEN4
    // Invoke sym_quant gemv kernels for m = 1 or n = 1.
    // Quantized GEMV is supported iff K and KC are divisible by group_size.
    // Fall back to GEMM Path otherwise.
    if (((k % grp_post_op_list->group_size) == 0)
        && ((KC % grp_post_op_list->group_size) == 0) && (mtag_b == REORDERED)
        && ((m == 1) || (n == 1))) {

        dlp_gemv_rowvar_s8s8s32o32_sym_quant(
            m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c, rs_c,
            cs_c, alpha, beta, rntm, thread, lcntx, grp_post_op_list,
            post_op_list, c_downscale);

        return;
    }
#endif

    // Group size should always be <= KC to make sure that entire group is
    // processed within one micro-kernel call. If group size is greater than KC,
    // then KC will be updated to group size. This same change is done in
    // reorder function to maintain consistency between reorder and GEMM
    // execution.
    if (grp_post_op_list->group_size > KC) {
        KC = grp_post_op_list->group_size;
    }

    // Strides are updated based on matrix packing/reordering.
    const int8_t* a_use          = NULL;
    md_t          rs_a_use       = rs_a;
    md_t          cs_a_use       = cs_a;
    md_t          a_block_stride = 0;

    const int8_t* b_use    = NULL;
    md_t          rs_b_use = rs_b;
    md_t          cs_b_use = cs_b;

    float* c_use_jc       = NULL;
    float* c_use_ic       = NULL;
    md_t   rs_c_use       = rs_c;
    md_t   rs_c_downscale = rs_c;

    // Pack buffer for A.
    int8_t* pack_a_buffer_s8s8s32o32 = NULL;
    msz_t   mem_a_size_req           = 0;

    // Pack buffer for B.
    int8_t* pack_b_buffer_s8s8s32o32 = NULL;
    msz_t   mem_b_size_req           = 0;
    md_t    packb_min_NR             = dlp_get_packb_s8s8s32o32_min_NR();

    // Temporary buffer for C accumulation when downscaling is required.
    float* temp_scal_c_buffer_s8s8s32o32 = NULL;
    msz_t  mem_scale_c_size_req          = 0;

    // kc needs to be a multiple of 4 so that it can be used with vpdpbusd
    // instruction. Padding is added in cases this condition is not
    // satisfied, and therefore the k offset used for packed/reordered
    // buffer needs to be updated.
    md_t k_updated = dlp_make_multiple_of_n(k, 4);
    md_t n_updated = dlp_make_multiple_of_n(n, 16);

    // To decide whether to apply post ops or not.
    bool is_last_k = FALSE;

    // To decide whether to use original s8 C or temp buffer for beta scale.
    bool is_first_k = FALSE;

    dlp_gemm_post_op_attr     post_ops_attr;
    dlp_gemm_grp_post_op_attr grp_post_ops_attr;

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

    md_t group_size = grp_post_op_list->group_size;

    // Initialize group post ops attributes.
    grp_post_ops_attr.a_scale_factor     = grp_post_op_list->a_scale_factor;
    grp_post_ops_attr.a_scale_factor_len = grp_post_op_list->a_scale_factor_len;
    grp_post_ops_attr.b_scale_factor     = grp_post_op_list->b_scale_factor;
    grp_post_ops_attr.b_scale_factor_len = grp_post_op_list->b_scale_factor_len;
    grp_post_ops_attr.a_zp               = grp_post_op_list->a_zp;
    grp_post_ops_attr.b_zp               = grp_post_op_list->b_zp;
    grp_post_ops_attr.a_zp_len           = grp_post_op_list->a_zp_len;
    grp_post_ops_attr.b_zp_len           = grp_post_op_list->b_zp_len;
    grp_post_ops_attr.group_size         = group_size;
    grp_post_ops_attr.sf_stor_type       = grp_post_op_list->sf_stor_type;
    grp_post_ops_attr.zp_stor_type       = grp_post_op_list->zp_stor_type;

    md_t num_groups                   = (k + group_size - 1) / group_size;
    grp_post_ops_attr.grp_post_op_lda = num_groups;
    grp_post_ops_attr.grp_post_op_ldb = n;

    // Generate thrinfo objects for jc and ic loops from dlp_gemm_thrinfo_t.
    dlp_task_id_t thread_jc;
    dlp_task_id_t thread_ic;

    dlp_gemm_gen_dlp_task_ids(thread, &thread_jc, &thread_ic);

    // Compute the JC, IC loop thread range for the current thread.
    md_t jc_start, jc_end;
    dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

    md_t ic_start, ic_end;
    dlp_thread_task_range(&thread_ic, m, MR, FALSE, &ic_start, &ic_end);

    for (iter_t jc = jc_start; jc < jc_end; jc += NC) {
        md_t nc0 = dlp_min((jc_end - jc), NC);

        md_t jc_cur_loop     = jc;
        md_t jc_cur_loop_rem = 0;
        md_t n_sub_updated   = 0;

        if (mtag_b == REORDERED) {
            dlp_gemm_get_B_panel_reordered_start_offset_width(
                jc, n, NC, packb_min_NR, &jc_cur_loop, &jc_cur_loop_rem, &nc0,
                &n_sub_updated);
        }

        if (c_downscale == DLP_F32) {
            c_use_jc = c + jc;
        }
        // Temp accumulaton buffer for C allocation.
        else if (c_downscale < DLP_F32) {
            // Buffer memory is only required if output needs to be
            // persisted across iterations of the pc/KC loop.
            // It was observed that the locks used while checking out
            // a buffer from memory pool had an impact on performance
            // and is better to not checkout if k <= KC.
            if (k > KC) {
                // The nc block itself can be split in 2 if its on a reorder
                // boundary. In this scenario its possible the first block
                // is smaller the second block, and the temporary buffer
                // should be allocated for the largest block.
                md_t nc0_buf = dlp_min((jc_end - jc), NC);
                mem_scale_c_size_req =
                    sizeof(float) * nc0_buf * (ic_end - ic_start);

                if (temp_scal_c_buffer_s8s8s32o32 == NULL) {
                    dlp_clsc_err_t ret_err;
                    temp_scal_c_buffer_s8s8s32o32 =
                        dlp_malloc_page_aligned(mem_scale_c_size_req, &ret_err);
                }

                c_use_jc = (float*)temp_scal_c_buffer_s8s8s32o32;
            }

            // The temp c buffer stride is modified as opposed to original C
            // matrix.
            rs_c_use = nc0;
        }

        int32_t* pack_b_column_sum = NULL;

        for (iter_t pc = 0; pc < k; pc += KC) {
            int32_t beta0 = (pc == 0) ? beta : 1;
            md_t    kc0   = dlp_min((k - pc), KC);

            grp_post_ops_attr.grp_post_op_k = pc;

            // kc0 needs to be a multiple of 4 so that it can be
            // used with vpdpbusd instruction. Padding is added in
            // cases this condition is not satisfied, and therefore
            // the kc0 offsets used for packed/reordered buffers
            // needs to be updated.
            md_t kc0_updated = dlp_make_multiple_of_n(kc0, 4);

            // No parallelization in k dim, k always starts at 0.
            is_first_k               = (pc == 0) ? (TRUE) : (FALSE);
            post_ops_attr.is_first_k = is_first_k;

            is_last_k               = ((pc + KC) >= k) ? (TRUE) : (FALSE);
            post_ops_attr.is_last_k = is_last_k;

            if (mtag_b == PACK) {
                // Pack B chunks are based on jc work id.
                md_t jc_work_id = thread_jc.work_id;

                // Using child thrinfo (thread_ic) tid to decide chief thread
                // per B matrix chunk (jc work id group)
                md_t nc0_updated = dlp_make_multiple_of_n(nc0, packb_min_NR);

                md_t group_start = pc / group_size;
                md_t group_end   = (pc + kc0 - 1) / group_size;

                md_t total_groups = (k + group_size - 1) / group_size;

                if (dlp_thread_am_ochief(&thread_ic)) {
                    // nc0 needs to be a multiple of 16 since this gives maximum
                    // vectorization. Packing B always results in buffers with
                    // width which is a multiple of 16. Subsequently the nc0
                    // offsets used for packed/reordered buffers needs to be
                    // updated.pack

                    mem_b_size_req =
                        sizeof(int8_t) * nc0_updated * kc0_updated
                        + (total_groups * nc0_updated * sizeof(int32_t));

                    if (pack_b_buffer_s8s8s32o32 == NULL) {
                        dlp_clsc_err_t ret_err;
                        pack_b_buffer_s8s8s32o32 =
                            dlp_malloc_page_aligned(mem_b_size_req, &ret_err);
                    }

                    thread->comm[jc_work_id].sent_object =
                        pack_b_buffer_s8s8s32o32;
                }

                // All threads in work group should wait till chief thread has
                // finished allocating the packing buffers.
                dlp_atomic_barrier(thread_ic.ocomm_id,
                                   &thread->comm[jc_work_id]);

                pack_b_buffer_s8s8s32o32 =
                    (int8_t*)thread->comm[jc_work_id].sent_object;

                // Compute the B panel per thread loop range for parallel
                // packing using ic_ways number of threads. Since atmost only
                // ic_ways threads can be used, the thread_ic attributes are
                // used to split the loop range.
                md_t jc_packb_start, jc_packb_end;
                dlp_thread_task_range(&thread_ic, nc0, NR, FALSE,
                                      &jc_packb_start, &jc_packb_end);

                if (pc == 0) {
                    pack_b_column_sum =
                        (int32_t*)(pack_b_buffer_s8s8s32o32
                                   + (sizeof(int8_t) * nc0_updated
                                      * kc0_updated));
                }

                // Ensure thread ranges are valid, especially cases where no:
                // of threads available for parallelization are greater than
                // no: of B panel NR chunks.
                if ((jc_packb_end > jc_packb_start)
                    && (jc_packb_start < (jc + nc0))) {
                    md_t nc0_pack = jc_packb_end - jc_packb_start;
                    if (pc == 0) {
                        for (iter_t group = 0; group < total_groups; group++) {
                            for (iter_t idx = jc_packb_start;
                                 idx < jc_packb_end; idx++) {
                                *(pack_b_column_sum + (group * nc0_updated)
                                  + idx) = 0;
                            }
                        }
                    }
                    // packing kernels are designed in such a way assuming that
                    // entire KCxNC block is packed at once and strides are set
                    // based on KC value. In current scenario, we call kernel
                    // with blocks of group_size x NC so kernel assumes that KC
                    // is group_size and strides are set based on group_size. To
                    // avoid this, we are calling kernel with blocks of
                    // group_size x NR, so that we can take care of the pointer
                    // movement across the reorder buffer in the framework
                    // itself.
                    for (iter_t jr = 0; jr < nc0_pack; jr += NR) {
                        md_t nr0 = dlp_min((nc0_pack - jr), NR);

                        int8_t* b_dst_jr =
                            pack_b_buffer_s8s8s32o32
                            + ((jc_packb_start + jr) * kc0_updated);
                        int32_t* b_sum_ptr =
                            pack_b_column_sum + (jc_packb_start + jr);
                        int8_t* b_src_jr =
                            (int8_t*)b + (cs_b * (jc + jc_packb_start + jr));

                        if (nr0 < NR) {
                            md_t nr_mult_16  = (nr0 / 16) * 16;
                            md_t nr0_rem     = nr0 % 16;
                            md_t nr0_updated = nr_mult_16;

                            if (nr_mult_16 > 0) {
                                // group loop
                                for (iter_t group = group_start;
                                     group <= group_end; group++) {
                                    md_t k_start =
                                        dlp_max(group * group_size, pc);
                                    md_t k_end =
                                        dlp_min(((group + 1) * group_size - 1),
                                                pc + kc0 - 1);
                                    md_t kg0 = k_end - k_start + 1;

                                    ((packb_s32_s8)lcntx->packb_fun_ptr)(
                                        b_dst_jr
                                            + ((group * group_size) - pc)
                                                  * nr0_updated,
                                        b_sum_ptr + (group * nc0_updated),
                                        b_src_jr + (rs_b * k_start), rs_b, cs_b,
                                        nr_mult_16, kg0, &rs_b_use, &cs_b_use);
                                }
                                b_dst_jr += nr_mult_16 * kc0_updated;
                                b_sum_ptr += nr_mult_16;
                                b_src_jr += nr_mult_16 * cs_b;
                            }

                            if (nr0_rem > 0) {
                                md_t nr0_updated = 16;
                                // group loop
                                for (iter_t group = group_start;
                                     group <= group_end; group++) {
                                    md_t k_start =
                                        dlp_max(group * group_size, pc);
                                    md_t k_end =
                                        dlp_min(((group + 1) * group_size - 1),
                                                pc + kc0 - 1);
                                    md_t kg0 = k_end - k_start + 1;

                                    ((packb_s32_s8)lcntx->packb_fun_ptr)(
                                        b_dst_jr
                                            + ((group * group_size) - pc)
                                                  * nr0_updated,
                                        b_sum_ptr + (group * nc0_updated),
                                        b_src_jr + (rs_b * k_start), rs_b, cs_b,
                                        nr0_rem, kg0, &rs_b_use, &cs_b_use);
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
                            md_t k_end = dlp_min(((group + 1) * group_size - 1),
                                                 pc + kc0 - 1);
                            md_t kg0   = k_end - k_start + 1;

                            ((packb_s32_s8)lcntx->packb_fun_ptr)(
                                b_dst_jr
                                    + ((group * group_size) - pc) * nr0_updated,
                                b_sum_ptr + (group * nc0_updated),
                                b_src_jr + (rs_b * k_start), rs_b, cs_b, NR,
                                kg0, &rs_b_use, &cs_b_use);
                        }
                    }

                    rs_b_use = NR * 4;
                    cs_b_use = NR;
                } else {
                    dlp_gemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);
                }

                // All threads in work group should wait till B matrix packing
                // is completed by the participating threads.
                dlp_atomic_barrier(thread_ic.ocomm_id,
                                   &thread->comm[jc_work_id]);
                b_use = pack_b_buffer_s8s8s32o32;

                post_ops_attr.b_col_sum_vec          = pack_b_column_sum;
                grp_post_ops_attr.grp_post_op_sum_ld = nc0_updated;

            } else if (mtag_b == REORDERED) {
                // In multi-threaded scenarios, an extra offset into a given
                // packed B panel is required, since the jc loop split can
                // result in per thread start offset inside the panel, instead
                // of panel boundaries.
                b_use = b + (jc_cur_loop * k_updated) + (n_sub_updated * pc)
                        + (jc_cur_loop_rem * kc0_updated);

                dlp_gemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);

                post_ops_attr.b_col_sum_vec =
                    ((int32_t*)(b + (k_updated * n_updated))) + jc;
                grp_post_ops_attr.grp_post_op_sum_ld = n_updated;
            } else {
                // Unpacked B not supported.
                return;
            }

            for (iter_t ic = ic_start; ic < ic_end; ic += MC) {
                md_t mc0 = dlp_min((ic_end - ic), MC);

                grp_post_ops_attr.grp_post_op_i = ic;

                // Only per thread C matrix is stored in temp buffer, so both
                // per thread jc and ic start should be normalized to zero.
                if (c_downscale < DLP_F32) {
                    c_use_ic = c_use_jc + (rs_c_use * (ic - ic_start));
                } else {
                    c_use_ic = c_use_jc + (rs_c_use * ic);
                }

                // Matrix A packed and reordered code path is not triggerred
                // currently since we do not support it yet.
                if (mtag_a == PACK) {
                    mem_a_size_req = sizeof(uint8_t) * mc0 * kc0_updated;

                    if (pack_a_buffer_s8s8s32o32 == NULL) {
                        dlp_clsc_err_t ret_err;
                        pack_a_buffer_s8s8s32o32 =
                            dlp_malloc_page_aligned(mem_a_size_req, &ret_err);
                    }

                    ((packa_s32)lcntx->packa_fun_ptr)(
                        (uint8_t*)pack_a_buffer_s8s8s32o32,
                        (uint8_t*)(a + (rs_a * ic) + (cs_a * pc)), rs_a, cs_a,
                        mc0, kc0, &rs_a_use, &cs_a_use);
                    a_use = pack_a_buffer_s8s8s32o32;

                    if (cs_a == 1) {
                        a_block_stride = kc0_updated;
                    }

                    else {
                        a_block_stride = rs_a_use;
                    }
                } else {
                    a_use = a + (rs_a * ic) + (cs_a * pc);

                    // Int8 kernel reads 4 elements, totalling 4 bytes in a
                    // single broadcast for use in vnni instruction.
                    // Non vnni based kernel requires update to this code.
                    cs_a_use       = 4;
                    a_block_stride = rs_a;
                }

                post_ops_attr.b_sum_offset = 0;

                for (iter_t jr = 0; jr < nc0; jr += NR) {
                    md_t nr0 = dlp_min((nc0 - jr), NR);

                    // Post ops meta attributes.
                    post_ops_attr.post_op_c_i    = ic;
                    post_ops_attr.post_op_c_j    = (jc + jr);
                    post_ops_attr.rs_c_downscale = rs_c_downscale;

                    grp_post_ops_attr.grp_post_op_j = jc + jr;

// The kernels are defined in zen4 folder
#ifdef DLP_KERNELS_ZEN4
                    // Reorder/Packed B, Reorder/Packed/Unpacked A call.
                    dlp_gemm_rowvar_s8s8s32os32_6x64m_sym_quant(
                        mc0, nr0, kc0, a_use, rs_a_use, cs_a_use,
                        a_block_stride, (b_use + (jr * kc0_updated)), rs_b_use,
                        cs_b_use, (c_use_ic + jr), rs_c_use, 1, alpha, beta0,
                        grp_post_ops_attr, post_op_list, post_ops_attr);
#endif
                    post_ops_attr.b_sum_offset += NR;
                }
            }
        }
        if (mtag_b == REORDERED) {
            dlp_gemm_adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }

    // Release pack buffers.
    if (mtag_b == PACK) {
        // All threads in work group should wait till B matrix usage is
        // completed by the participating threads.
        dlp_atomic_barrier(thread_jc.ocomm_id,
                           &thread->comm[thread_jc.work_id]);

        if (dlp_thread_am_ochief(&thread_ic)) {
            if (pack_b_buffer_s8s8s32o32 != NULL) {
                dlp_free_page_aligned(pack_b_buffer_s8s8s32o32);
            }
        }
    }
    if (mtag_a == PACK) {
        if (pack_a_buffer_s8s8s32o32 != NULL) {
            dlp_free_page_aligned(pack_a_buffer_s8s8s32o32);
        }
    }
    if (c_downscale < DLP_F32) {
        if (temp_scal_c_buffer_s8s8s32o32 != NULL) {
            dlp_free_page_aligned(temp_scal_c_buffer_s8s8s32o32);
        }
    }
}
