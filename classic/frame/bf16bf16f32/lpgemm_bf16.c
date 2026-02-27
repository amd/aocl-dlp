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
#include "kernels/bf16bf16f32/lpgemm_pack_bf16.h"
#include "kernels/lpgemm_kernels.h"
#include "lpgemm_5loop_interface_apis.h"
#include "sys_utils/lpgemm_sys.h"
#include "threading/lpgemm_thread_utils.h"

// Kernel function prototypes
typedef void (*lpgemm_rowvar_bf16)(const md_t,
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
                                   lpgemm_post_op*,
                                   lpgemm_post_op_attr);

#ifdef DLP_KERNELS_ZEN4
LPGEMV(bfloat16, bfloat16, float, bf16bf16f32of32)
{
    md_t NC = lcntx->blksz.NC;
    md_t KC = lcntx->blksz.KC;
    md_t MC = lcntx->blksz.MC;
    md_t NR = lcntx->blksz.NR;

    // Strides are updated based on matrix packing/reordering.
    bfloat16* a_use    = (bfloat16*)a;
    md_t      rs_a_use = rs_a;
    md_t      cs_a_use = cs_a;

    bfloat16* b_use    = (bfloat16*)b;
    md_t      rs_b_use = rs_b;
    md_t      cs_b_use = cs_b;

    float*    c_use              = NULL;
    bfloat16* pack_a_buffer_bf16 = NULL;

    lpgemm_post_op_attr post_ops_attr;
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

    msz_t mem_a_size_req = 0;
    msz_t mem_b_size_req = 0;

    bfloat16* pack_b_buffer_bf16 = NULL;

    // Generate thrinfo objects for jc and ic loops from lpgemm_thrinfo_t.
    dlp_task_id_t thread_jc;
    dlp_task_id_t thread_ic;

    lpgemm_gen_dlp_task_ids(thread, &thread_jc, &thread_ic);

    if (n == 1) {
        // Increased MR from 6 to 16 to make use of 32 ZMM registers
        md_t MR = 16;

        // pack B matrix if rs_b > 1
        if ((mtag_b == PACK) && (rs_b != 1)) {
            mem_b_size_req = sizeof(bfloat16) * k;

            if (pack_b_buffer_bf16 == NULL) {
                dlp_clsc_err_t ret_err;
                pack_b_buffer_bf16 =
                    dlp_malloc_page_aligned(mem_b_size_req, &ret_err);
            }

            for (iter_t k0 = 0; k0 < k; k0++) {
                pack_b_buffer_bf16[k0] = b[k0 * rs_b];
            }

            b_use    = pack_b_buffer_bf16;
            rs_b_use = 1;
            cs_b_use = 1;
        }

        // Compute the IC loop thread range for the current thread.
        md_t ic_start, ic_end;
        thread_ic.n_way   = (thread_ic.n_way == 1) ? (thread->n_threads)
                                                   : (thread_ic.n_way);
        thread_ic.work_id = thread->tid;
        dlp_thread_task_range(&thread_ic, m, MR, FALSE, &ic_start, &ic_end);

        for (iter_t ic = ic_start; ic < ic_end; ic += MC) {
            md_t            mc0          = dlp_min((ic_end - ic), MC);
            const bfloat16* a_use        = a + ic * rs_a;
            c_use                        = c + ic * rs_c;
            post_ops_attr.post_op_c_i    = ic;
            post_ops_attr.post_op_c_j    = 0;
            post_ops_attr.rs_c_downscale = rs_c;

            if (mtag_a == PACK) {
                mem_a_size_req = sizeof(bfloat16) * mc0 * k;

                if (pack_a_buffer_bf16 == NULL) {
                    dlp_clsc_err_t ret_err;
                    pack_a_buffer_bf16 =
                        dlp_malloc_page_aligned(mem_a_size_req, &ret_err);
                }

                ((pack_bf16)lcntx->packa_fun_ptr)(pack_a_buffer_bf16,
                                                  (a + (rs_a * ic)), rs_a, cs_a,
                                                  mc0, k, &rs_a_use, &cs_a_use);
                a_use = pack_a_buffer_bf16;
            }
            // Call lpgemv_n_one kernel
            if (lcntx->dlp_kernel_hndl.kernel_base != NULL) {
                dlp_execute_kernel(&(lcntx->dlp_kernel_hndl), mc0, 1, k,
                                   (bfloat16*)a_use, rs_a_use, cs_a_use, 1,
                                   (bfloat16*)b_use, rs_b_use, cs_b_use, 0, 0,
                                   c_use, rs_c, cs_c, (void*)&alpha,
                                   (void*)&beta, post_op_list, post_ops_attr);

            } else {
                lpgemv_n_one_bf16bf16f32of32(
                    mc0, k, a_use, rs_a_use, cs_a_use, mtag_a, b_use, rs_b_use,
                    cs_b_use, mtag_b, c_use, rs_c, cs_c, alpha, beta, MR, KC,
                    post_op_list, &post_ops_attr);
            }
        }

        // Release pack buffers
        if (mtag_a == PACK && (pack_a_buffer_bf16 != NULL)) {
            dlp_free_page_aligned(pack_a_buffer_bf16);
        }
        if (mtag_b == PACK && (pack_b_buffer_bf16 != NULL)) {
            dlp_free_page_aligned(pack_b_buffer_bf16);
        }
    } else {
        // Compute the JC loop thread range for the current thread.
        md_t jc_start, jc_end;
        thread_jc.n_way   = (thread_jc.n_way == 1) ? (thread->n_threads)
                                                   : (thread_jc.n_way);
        thread_jc.work_id = thread->tid;
        dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

        md_t packb_min_NR = 16;

        md_t k_updated = k;
        k_updated += (k_updated & 0x1);

        md_t kc0 = dlp_min(k, KC);

        kc0 += (kc0 & 0x1);

        rs_a_use = rs_a;
        cs_a_use = 2;

        if (mtag_a == PACK) {
            mem_a_size_req = sizeof(bfloat16) * k;

            if (pack_a_buffer_bf16 == NULL) {
                dlp_clsc_err_t ret_err;
                pack_a_buffer_bf16 =
                    dlp_malloc_page_aligned(mem_a_size_req, &ret_err);
            }

            ((pack_bf16)lcntx->packa_fun_ptr)(pack_a_buffer_bf16, a, rs_a, cs_a,
                                              1, k, &rs_a_use, &cs_a_use);

            a_use = pack_a_buffer_bf16;
        }

        for (iter_t jc = jc_start; jc < jc_end; jc += NC) {
            md_t nc0 = dlp_min((jc_end - jc), NC);
            c_use    = c + jc * cs_c;

            md_t jc_cur_loop     = jc;
            md_t jc_cur_loop_rem = 0;
            md_t n_sub_updated   = 0;

            if (mtag_b == REORDERED) {

                get_B_panel_reordered_start_offset_width(
                    jc, n, NC, packb_min_NR, &jc_cur_loop, &jc_cur_loop_rem,
                    &nc0, &n_sub_updated);

                b_use = (bfloat16*)(b + (jc_cur_loop * k_updated));

                lpgemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);
            } else if (mtag_b == PACK) {

                md_t nc0_updated = make_multiple_of_n(nc0, packb_min_NR);
                mem_b_size_req   = sizeof(bfloat16) * nc0_updated * k_updated;

                n_sub_updated = nc0_updated;

                if (pack_b_buffer_bf16 == NULL) {
                    dlp_clsc_err_t ret_err;
                    pack_b_buffer_bf16 =
                        dlp_malloc_page_aligned(mem_b_size_req, &ret_err);
                }

                for (iter_t pc = 0; pc < k; pc += KC) {
                    md_t kc0 = dlp_min((k - pc), KC);

                    md_t kc0_updated = kc0;
                    kc0_updated += (kc0_updated & 0x1);

                    ((pack_bf16)lcntx->packb_fun_ptr)(
                        ((bfloat16*)pack_b_buffer_bf16) + (n_sub_updated * pc),
                        (((bfloat16*)b) + (rs_b * pc) + (jc * cs_b)), rs_b,
                        cs_b, nc0, kc0, &rs_b_use, &cs_b_use);
                }

                b_use = pack_b_buffer_bf16;
            }

            post_ops_attr.post_op_c_i    = 0;
            post_ops_attr.post_op_c_j    = jc;
            post_ops_attr.rs_c_downscale = rs_c;

            if (lcntx->dlp_kernel_hndl.kernel_base != NULL) {
                dlp_execute_kernel(
                    &(lcntx->dlp_kernel_hndl), 1, nc0, k, (bfloat16*)a_use,
                    rs_a_use, cs_a_use, 1, (bfloat16*)b_use, rs_b_use, cs_b_use,
                    n_sub_updated, jc_cur_loop_rem, c_use, rs_c, cs_c,
                    (void*)&alpha, (void*)&beta, post_op_list, post_ops_attr);
            } else {
                lpgemv_m_one_bf16bf16f32of32(
                    nc0, k, a_use, rs_a_use, cs_a_use, mtag_a, b_use, rs_b_use,
                    cs_b_use, mtag_b, c_use, rs_c, cs_c, alpha, beta, NR, KC,
                    n_sub_updated, jc_cur_loop_rem, post_op_list,
                    &post_ops_attr);
            }

            if (mtag_b == REORDERED) {
                adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
            }
        } // jc loop

        // Release pack buffers.
        if (mtag_b == PACK && (pack_b_buffer_bf16 != NULL)) {
            dlp_free_page_aligned(pack_b_buffer_bf16);
        }
        if (mtag_a == PACK && (pack_a_buffer_bf16 != NULL)) {
            dlp_free_page_aligned(pack_a_buffer_bf16);
        }
    }
}
#endif
// B should always be packed.
LPGEMM_5LOOP_AVX512BF16(bfloat16, bfloat16, float, bf16bf16f32of32)
{
#if (defined(DLP_KERNELS_ZEN4) && (!defined(LPGEMM_BF16_JIT)))
    // Handle using LPGEMV when m or/and n equal to 1
    // The avx512 check will be removed when avx2 kernels added in future
    if ((n == 1) || (m == 1)) {
        lpgemv_rowvar_bf16bf16f32of32(
            m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c, rs_c,
            cs_c, alpha, beta, rntm, thread, lcntx, post_op_list, c_downscale);
        return;
    }
#endif

    md_t NC = lcntx->blksz.NC;
    md_t KC = lcntx->blksz.KC;
    md_t MC = lcntx->blksz.MC;
    md_t NR = lcntx->blksz.NR;
    md_t MR = lcntx->blksz.MR;

    const int16_t* a_use          = NULL;
    md_t           cs_a_use       = cs_a;
    md_t           rs_a_use       = rs_a;
    md_t           a_block_stride = 0;

    const int16_t* b_use    = NULL;
    md_t           rs_b_use = rs_b;
    md_t           cs_b_use = cs_b;

    float* c_use_jc       = NULL;
    float* c_use_ic       = NULL;
    md_t   rs_c_use       = rs_c;
    md_t   rs_c_downscale = rs_c;

    // Pack buffer for B.
    bfloat16* pack_b_buffer_bf16 = NULL;
    bfloat16* pack_a_buffer_bf16 = NULL;
    msz_t     mem_b_size_req     = 0;
    msz_t     mem_a_size_req     = 0;
    md_t      packb_min_NR       = 16;

    // Temporary buffer for C accumulation when downscaling is required.
    float* temp_scal_c_buffer_bf16 = NULL;
    msz_t  mem_scale_c_size_req    = 0;

    // kc needs to be a multiple of 2 so that it can be used with dpbf16_ps
    // instruction. Padding is added in cases this condition is not
    // satisfied, and therefore the k offset used for packed/reordered
    // buffer needs to be updated.
    md_t k_updated = k;
    k_updated += (k_updated & 0x1);

    // To decide whether to apply post ops or not.
    bool is_last_k = FALSE;

    // To decide whether to use original s8 C or temp buffer for beta scale.
    bool is_first_k = FALSE;

    lpgemm_post_op_attr post_ops_attr;
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

    // Generate thrinfo objects for jc and ic loops from lpgemm_thrinfo_t.
    dlp_task_id_t thread_jc;
    dlp_task_id_t thread_ic;

    lpgemm_gen_dlp_task_ids(thread, &thread_jc, &thread_ic);

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
            get_B_panel_reordered_start_offset_width(
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

                if (temp_scal_c_buffer_bf16 == NULL) {
                    dlp_clsc_err_t ret_err;
                    temp_scal_c_buffer_bf16 =
                        dlp_malloc_page_aligned(mem_scale_c_size_req, &ret_err);
                }

                c_use_jc = (float*)temp_scal_c_buffer_bf16;
            }

            // The temp c buffer stride is modified as opposed to original C
            // matrix.
            rs_c_use = nc0;
        }

        for (iter_t pc = 0; pc < k; pc += KC) {
            float beta0 = (pc == 0) ? beta : 1;
            md_t  kc0   = dlp_min((k - pc), KC);

            // No parallelization in k dim, k always starts at 0.
            is_first_k               = (pc == 0) ? (TRUE) : (FALSE);
            post_ops_attr.is_first_k = is_first_k;

            is_last_k               = ((pc + KC) >= k) ? (TRUE) : (FALSE);
            post_ops_attr.is_last_k = is_last_k;

            // kc0 needs to be a multiple of 2 so that it can be
            // used with dpbf16_ps instruction. Padding is added in
            // cases this condition is not satisfied, and therefore
            // the kc0 offsets used for packed/reordered buffers
            // needs to be updated.
            md_t kc0_updated = kc0;
            kc0_updated += (kc0_updated & 0x1);

            if (mtag_b == PACK) {
                // Pack B chunks are based on jc work id.
                md_t jc_work_id = thread_jc.work_id;

                // Using child thrinfo (thread_ic) tid to decide chief thread
                // per B matrix chunk (jc work id group)
                if (dlp_thread_am_ochief(&thread_ic)) {
                    // nc0 needs to be a multiple of 16 since this gives maximum
                    // vectorization. Packing B always results in buffers with
                    // width which is a multiple of 16. Subsequently the nc0
                    // offsets used for packed/reordered buffers needs to be
                    // updated.
                    md_t nc0_updated = make_multiple_of_n(nc0, packb_min_NR);
                    mem_b_size_req =
                        sizeof(bfloat16) * nc0_updated * kc0_updated;

                    if (pack_b_buffer_bf16 == NULL) {
                        dlp_clsc_err_t ret_err;
                        pack_b_buffer_bf16 =
                            dlp_malloc_page_aligned(mem_b_size_req, &ret_err);
                    }

                    thread->comm[jc_work_id].sent_object = pack_b_buffer_bf16;
                }

                // All threads in work group should wait till chief thread has
                // finished allocating the packing buffers.
                dlp_atomic_barrier(thread_ic.ocomm_id,
                                   &thread->comm[jc_work_id]);

                pack_b_buffer_bf16 =
                    (bfloat16*)thread->comm[jc_work_id].sent_object;

                // Compute the B panel per thread loop range for parallel
                // packing using ic_ways number of threads. Since atmost only
                // ic_ways threads can be used, the thread_ic attributes are
                // used to split the loop range.
                md_t jc_packb_start, jc_packb_end;
                dlp_thread_task_range(&thread_ic, nc0, NR, FALSE,
                                      &jc_packb_start, &jc_packb_end);

                // Ensure thread ranges are valid, especially cases where no:
                // of threads available for parallelization are greater than
                // no: of B panel NR chunks.
                if ((jc_packb_end > jc_packb_start)
                    && (jc_packb_start < (jc + nc0))) {
                    ((pack_bf16)lcntx->packb_fun_ptr)(
                        pack_b_buffer_bf16 + (jc_packb_start * kc0_updated),
                        (b + (rs_b * pc) + (cs_b * jc)
                         + (cs_b * jc_packb_start)),
                        rs_b, cs_b, (jc_packb_end - jc_packb_start), kc0,
                        &rs_b_use, &cs_b_use);
                } else {
                    lpgemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);
                }

                // All threads in work group should wait till B matrix packing
                // is completed by the participating threads.
                dlp_atomic_barrier(thread_ic.ocomm_id,
                                   &thread->comm[jc_work_id]);
                b_use = pack_b_buffer_bf16;
            }
            // B part getting processed
            else if (mtag_b == REORDERED) {
                // In multi-threaded scenarios, an extra offset into a given
                // packed B panel is required, since the jc loop split can
                // result in per thread start offset inside the panel, instead
                // of panel boundaries.
                b_use = b + (jc_cur_loop * k_updated) + (n_sub_updated * pc)
                        + (jc_cur_loop_rem * kc0_updated);

                lpgemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);
            }

            for (iter_t ic = ic_start; ic < ic_end; ic += MC) {
                md_t mc0 = dlp_min((ic_end - ic), MC);

                // Only per thread C matrix is stored in temp buffer, so both
                // per thread jc and ic start should be normalized to zero.
                if (c_downscale < DLP_F32) {
                    c_use_ic = c_use_jc + (rs_c_use * (ic - ic_start));
                } else {
                    c_use_ic = c_use_jc + (rs_c_use * ic);
                }

                if (mtag_a == UNPACKED) {
                    a_use = a + (rs_a * ic) + (cs_a * pc);

                    // bf16 kernel reads 2 elements, totalling 4 bytes in a
                    // single broadcast for use in bf16 instruction.
                    // Non bf16 based kernel requires update to this code.
                    cs_a_use       = 2;
                    a_block_stride = rs_a;
                    rs_a_use       = rs_a;
                } else if (mtag_a == PACK) {

                    mem_a_size_req = sizeof(bfloat16) * mc0 * kc0;

                    if (pack_a_buffer_bf16 == NULL) {
                        dlp_clsc_err_t ret_err;
                        pack_a_buffer_bf16 =
                            dlp_malloc_page_aligned(mem_a_size_req, &ret_err);
                    }

                    ((pack_bf16)lcntx->packa_fun_ptr)(
                        pack_a_buffer_bf16, (a + (rs_a * ic) + (cs_a * pc)),
                        rs_a, cs_a, mc0, kc0, &rs_a_use, &cs_a_use);
                    a_use          = pack_a_buffer_bf16;
                    a_block_stride = rs_a_use;
                }

                for (iter_t jr = 0; jr < nc0; jr += NR) {
                    md_t nr0 = dlp_min((nc0 - jr), NR);

                    // Post ops meta attributes.
                    post_ops_attr.post_op_c_i    = ic;
                    post_ops_attr.post_op_c_j    = (jc + jr);
                    post_ops_attr.rs_c_downscale = rs_c_downscale;

                    // Reorder/Packed B, Reorder/Packed/Unpacked A call.
                    if (lcntx->dlp_kernel_hndl.kernel_base != NULL) {
                        dlp_execute_kernel(
                            &(lcntx->dlp_kernel_hndl), mc0, nr0, kc0,
                            (int16_t*)a_use, rs_a_use, cs_a_use, a_block_stride,
                            (int16_t*)(b_use + (jr * kc0_updated)), rs_b_use,
                            cs_b_use, 0, 0, (c_use_ic + jr), rs_c_use, 1,
                            (void*)&alpha, (void*)&beta0, post_op_list,
                            post_ops_attr);
                    } else {
                        ((lpgemm_rowvar_bf16)lcntx->kern_fun_ptr)(
                            mc0, nr0, kc0, a_use, rs_a_use, cs_a_use,
                            a_block_stride, (b_use + (jr * kc0_updated)),
                            rs_b_use, cs_b_use, (c_use_ic + jr), rs_c_use, 1,
                            alpha, beta0, post_op_list, post_ops_attr);
                    }
                }
            }
        }
        if (mtag_b == REORDERED) {
            adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }

    // Release pack buffers.
    if (mtag_b == PACK) {
        // All threads in work group should wait till B matrix usage is
        // completed by the participating threads.
        dlp_atomic_barrier(thread_jc.ocomm_id,
                           &thread->comm[thread_jc.work_id]);

        if (dlp_thread_am_ochief(&thread_ic)) {
            if (pack_b_buffer_bf16 != NULL) {
                dlp_free_page_aligned(pack_b_buffer_bf16);
            }
        }
    }
    if (mtag_a == PACK) {
        if (pack_a_buffer_bf16 != NULL) {
            dlp_free_page_aligned(pack_a_buffer_bf16);
        }
    }
    if (c_downscale < DLP_F32) {
        if (temp_scal_c_buffer_bf16 != NULL) {
            dlp_free_page_aligned(temp_scal_c_buffer_bf16);
        }
    }
}

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

typedef void (*lpgemv_m_one_ker_ft)(const md_t,
                                    const md_t,
                                    const float*,
                                    const md_t,
                                    const md_t,
                                    const AOCL_MEMORY_TAG,
                                    const float*,
                                    md_t,
                                    const md_t,
                                    const AOCL_MEMORY_TAG,
                                    float*,
                                    const md_t,
                                    const md_t,
                                    const float,
                                    const float,
                                    md_t,
                                    const md_t,
                                    const md_t,
                                    const md_t,
                                    lpgemm_post_op*,
                                    lpgemm_post_op_attr*);

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

LPGEMV_F32_FALLBACK(bfloat16, bfloat16, float, bf16bf16f32of32)
{
    // DLP_BF16 Contexts
    md_t NC = lcntx->blksz.NC;
    md_t KC = lcntx->blksz.KC;
    md_t MC = lcntx->blksz.MC;
    md_t NR = lcntx->blksz.NR;
    md_t MR = lcntx->blksz.MR;

    // DLP_F32 contexts for the GEMM
    lpgemm_cntx_t*  lcntx_f32 = lpgemm_get_global_cntx_obj(F32F32F32OF32);
    md_t            f32_MR; // This will be modified
    md_t            f32_NR = lcntx_f32->blksz.NR;
    AOCL_MEMORY_TAG f32_mtag_b;

    const float* a_use    = NULL;
    md_t         cs_a_use = cs_a;
    md_t         rs_a_use = rs_a;

    const float* b_use    = NULL;
    md_t         rs_b_use = rs_b;
    md_t         cs_b_use = cs_b;

    float* c_use = NULL;

    // Pack buffer for B.
    float* cvt_b_buffer_bf16_f32 = NULL;
    float* cvt_a_buffer_bf16_f32 = NULL;

    msz_t mem_b_size_req = 0;
    msz_t mem_a_size_req = 0;

    lpgemm_post_op_attr post_ops_attr;
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
    /* The thread calculations would still follow DLP_BF16 dimensions*/
    // Generate thrinfo objects for jc and ic loops from lpgemm_thrinfo_t.
    dlp_task_id_t thread_jc;
    dlp_task_id_t thread_ic;

    lpgemm_gen_dlp_task_ids(thread, &thread_jc, &thread_ic);

    if (n == 1) // n = 1 case.
    {
        lpgemv_n_one_ker_ft ker_fp;
#ifdef DLP_KERNELS_ZEN4
        if (dlp_cpuid_is_avx512_supported() == TRUE) {
            if (lpgemm_get_enabled_arch() == DLP_ARCH_ZEN3) {
                f32_MR = 16;
                ker_fp = lpgemv_n_one_f32f32f32of32_avx512_256;
            } else {
                f32_MR = 16;
                ker_fp = lpgemv_n_one_f32f32f32of32;
            }
        } else {
#endif
            // Increased MR from 6 to 16 to make use of 32 ZMM registers
            f32_MR = 8;
            ker_fp = lpgemv_n_one_f32f32f32of32_avx2;
#ifdef DLP_KERNELS_ZEN4
        }
#endif
        // for bf16 inputs no matter if it's packed/re-ordered and unpacked,
        // the matrix to be given to the kernels has to be in bf16.
        mem_b_size_req = sizeof(float) * k;

        if (cvt_b_buffer_bf16_f32 == NULL) {
            dlp_clsc_err_t ret_err;
            cvt_b_buffer_bf16_f32 =
                dlp_malloc_page_aligned(mem_b_size_req, &ret_err);
        }

        // Call the function to convert bf16 to f32
        rs_b_use = 1;
        cs_b_use = 1;

        if (mtag_b == REORDERED) {
            /* For n = 1 case, a re-ordered matrix would be stored contigously
             in memeory and hence need to be accessed likewise for conversion.*/
            // Direct call to optimized GEMV unpacking (N=1, contiguous)
            unpackb_nr64_bf16_f32_gemv(b, cvt_b_buffer_bf16_f32, k);
        } else {
            // Direct call to optimized GEMV conversion (K=1, contiguous output)
            cvt_bf16_f32_gemv_row_major(cvt_b_buffer_bf16_f32, b, rs_b, k);
        }
        b_use = cvt_b_buffer_bf16_f32;

        // Compute the IC loop thread range for the current thread.
        md_t ic_start, ic_end;
        thread_ic.n_way   = (thread_ic.n_way == 1) ? (thread->n_threads)
                                                   : (thread_ic.n_way);
        thread_ic.work_id = thread->tid;
        dlp_thread_task_range(&thread_ic, m, MR, FALSE, &ic_start, &ic_end);

        for (iter_t ic = ic_start; ic < ic_end; ic += MC) {
            md_t mc0 = dlp_min((ic_end - ic), MC);

            c_use                        = c + ic * rs_c;
            post_ops_attr.post_op_c_i    = ic;
            post_ops_attr.post_op_c_j    = 0;
            post_ops_attr.rs_c_downscale = rs_c;
            mem_a_size_req               = sizeof(float) * mc0 * k;

            // For packed or unpacked A matrix, the mc0 * kc0 block is
            // converted to DLP_F32, i.e., packing has to be done by default
            if (cvt_a_buffer_bf16_f32 == NULL) {
                dlp_clsc_err_t ret_err;
                cvt_a_buffer_bf16_f32 =
                    dlp_malloc_page_aligned(mem_a_size_req, &ret_err);
            }

            rs_a_use = k;
            cs_a_use = 1;

            cvt_bf16_f32((cvt_a_buffer_bf16_f32), (a + (rs_a * ic)), rs_a, cs_a,
                         mc0, k, rs_a_use, cs_a_use);

            a_use = cvt_a_buffer_bf16_f32;

            if (lcntx->dlp_kernel_hndl.kernel_base != NULL) {
                dlp_execute_kernel(&(lcntx->dlp_kernel_hndl), mc0, 1, k,
                                   (float*)a_use, rs_a_use, cs_a_use, 1,
                                   (float*)b_use, rs_b_use, cs_b_use, 0, 0,
                                   c_use, rs_c, cs_c, (void*)&alpha,
                                   (void*)&beta, post_op_list, post_ops_attr);

            } else {
                ker_fp(mc0, k, a_use, rs_a_use, cs_a_use, mtag_a, b_use,
                       rs_b_use, cs_b_use, mtag_b, c_use, rs_c, cs_c, alpha,
                       beta, f32_MR, KC, post_op_list, &post_ops_attr);
            }
        }

        if (cvt_a_buffer_bf16_f32 != NULL) {
            dlp_free_page_aligned(cvt_a_buffer_bf16_f32);
        }
        if (cvt_b_buffer_bf16_f32 != NULL) {
            dlp_free_page_aligned(cvt_b_buffer_bf16_f32);
        }
    } else // m = 1 case
    {
        lpgemv_m_one_ker_ft ker_fp;
        float*              b_unreorder = NULL;

#ifdef DLP_KERNELS_ZEN4
        if (dlp_cpuid_is_avx512_supported() == TRUE) {
            if (lpgemm_get_enabled_arch() == DLP_ARCH_ZEN3) {
                ker_fp = lpgemv_m_one_f32f32f32of32_avx512_256;
            } else {
                ker_fp = lpgemv_m_one_f32f32f32of32;
            }
        } else {
#endif
            ker_fp = lpgemv_m_one_f32f32f32of32_avx2;
#ifdef DLP_KERNELS_ZEN4
        }
#endif
        // Compute the JC loop thread range for the current thread.
        md_t jc_start, jc_end;
        thread_jc.n_way   = (thread_jc.n_way == 1) ? (thread->n_threads)
                                                   : (thread_jc.n_way);
        thread_jc.work_id = thread->tid;
        dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

        md_t packb_min_NR = 16;

        md_t k_updated = k;
        k_updated += (k_updated & 0x1);

        md_t kc0 = dlp_min(k, KC);

        kc0 += (kc0 & 0x1);
        mem_a_size_req = sizeof(float) * k;

        // For packed or unpacked A matrix, the mc0 * kc0 block is
        // converted to DLP_F32, i.e., packing has to be done by default
        if (cvt_a_buffer_bf16_f32 == NULL) {
            dlp_clsc_err_t ret_err;
            cvt_a_buffer_bf16_f32 =
                dlp_malloc_page_aligned(mem_a_size_req, &ret_err);
        }

        rs_a_use = k;
        cs_a_use = 1;

        cvt_bf16_f32((cvt_a_buffer_bf16_f32), a, rs_a, cs_a, 1, k, rs_a_use,
                     cs_a_use);

        a_use = cvt_a_buffer_bf16_f32;

        for (iter_t jc = jc_start; jc < jc_end; jc += NC) {
            md_t nc0 = dlp_min((jc_end - jc), NC);
            c_use    = c + jc;

            md_t jc_cur_loop     = jc;
            md_t jc_cur_loop_rem = 0;
            md_t n_sub_updated   = 0;

            md_t nc0_updated = make_multiple_of_n(nc0, packb_min_NR);
            mem_b_size_req   = sizeof(float) * nc0_updated * k_updated;

            if (mtag_b == REORDERED) {
                get_B_panel_reordered_start_offset_width(
                    jc, n, NC, packb_min_NR, &jc_cur_loop, &jc_cur_loop_rem,
                    &nc0, &n_sub_updated);
            }

            for (iter_t pc = 0; pc < k; pc += KC) {
                md_t kc0 = dlp_min((k - pc), KC);

                md_t kc0_updated = kc0;
                kc0_updated += (kc0_updated & 0x1);

                if (mtag_b == REORDERED) {
                    if (b_unreorder == NULL) {
                        dlp_clsc_err_t ret_err;
                        b_unreorder =
                            dlp_malloc_page_aligned(mem_b_size_req, &ret_err);
                    }

                    rs_b_use = nc0;
                    cs_b_use = 1;

                    unpackb_nr64_bf16_f32((b + (jc_cur_loop * k_updated)
                                           + (jc_cur_loop_rem * kc0_updated)
                                           + (n_sub_updated * pc)),
                                          (b_unreorder + (nc0 * pc)), kc0, nc0,
                                          rs_b_use, cs_b_use);
                    b_use = b_unreorder;
                } else {
                    if (cvt_b_buffer_bf16_f32 == NULL) {
                        dlp_clsc_err_t ret_err;
                        cvt_b_buffer_bf16_f32 =
                            dlp_malloc_page_aligned(mem_b_size_req, &ret_err);
                    }

                    rs_b_use = nc0;
                    cs_b_use = 1;

                    cvt_bf16_f32((cvt_b_buffer_bf16_f32 + (nc0 * pc)),
                                 (b + (rs_b * pc) + (cs_b * jc)), rs_b, cs_b,
                                 kc0, nc0, rs_b_use, cs_b_use);
                    b_use = cvt_b_buffer_bf16_f32;
                }
                /* DLP_BF16 inputs when converted to DLP_F32, from a re-ordered
                 or packed form gets back to the original matrix dimensions,
                 i.e., the block sizes of matrix will not be known while
                 unpacking/un-reordering. Hence, resetting mtag_b before
                 calling the kernels to ensure the strides are taken correctly.
               */
                f32_mtag_b = UNPACKED;
            }
            post_ops_attr.post_op_c_i    = 0;
            post_ops_attr.post_op_c_j    = jc;
            post_ops_attr.rs_c_downscale = rs_c;
            post_ops_attr.is_first_k     = TRUE;

            if (lcntx->dlp_kernel_hndl.kernel_base != NULL) {
                dlp_execute_kernel(
                    &(lcntx->dlp_kernel_hndl), 1, nc0, k, (float*)a_use,
                    rs_a_use, cs_a_use, 1, (float*)b_use, rs_b_use, cs_b_use,
                    n_sub_updated, jc_cur_loop_rem, c_use, rs_c, cs_c,
                    (void*)&alpha, (void*)&beta, post_op_list, post_ops_attr);
            } else {
                ker_fp(nc0, k, a_use, rs_a_use, cs_a_use, mtag_a, b_use,
                       rs_b_use, cs_b_use, f32_mtag_b, c_use, rs_c, cs_c, alpha,
                       beta, f32_NR, KC, n_sub_updated, jc_cur_loop_rem,
                       post_op_list, &post_ops_attr);
            }
        }

        if (b_unreorder != NULL) {
            dlp_free_page_aligned(b_unreorder);
        }
        if (cvt_b_buffer_bf16_f32 != NULL) {
            dlp_free_page_aligned(cvt_b_buffer_bf16_f32);
        }
        if (cvt_a_buffer_bf16_f32 != NULL) {
            dlp_free_page_aligned(cvt_a_buffer_bf16_f32);
        }
    }
}

LPGEMM_5LOOP_F32_FALLBACK(bfloat16, bfloat16, float, bf16bf16f32of32)
{
#if (!defined(LPGEMM_BF16_JIT))
    // Handle using LPGEMV when m or/and n equal to 1
    // The avx512 check will be removed when avx2 kernels added in future
    if ((n == 1) || (m == 1)) {
        lpgemv_rowvar_f32_fallback_bf16bf16f32of32(
            m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c, rs_c,
            cs_c, alpha, beta, rntm, thread, lcntx, post_op_list, c_downscale);
        return;
    }
#endif
    // DLP_BF16 Contexts
    md_t NC = lcntx->blksz.NC;
    md_t KC = lcntx->blksz.KC;
    md_t MC = lcntx->blksz.MC;
    md_t NR = lcntx->blksz.NR;
    md_t MR = lcntx->blksz.MR;

    // DLP_F32 contexts for the GEMM
    lpgemm_cntx_t* lcntx_f32 = lpgemm_get_global_cntx_obj(F32F32F32OF32);
    md_t           f32_MR    = lcntx_f32->blksz.MR;
    md_t           f32_NR    = lcntx_f32->blksz.NR;

    const float* a_use          = NULL;
    md_t         cs_a_use       = cs_a;
    md_t         rs_a_use       = rs_a;
    md_t         a_block_stride = 0;

    const float* b_use    = NULL;
    md_t         rs_b_use = rs_b;
    md_t         cs_b_use = cs_b;

    float* c_use_jc       = NULL;
    float* c_use_ic       = NULL;
    md_t   rs_c_use       = rs_c;
    md_t   rs_c_downscale = rs_c;

    // Pack buffer for B.
    float* cvt_b_buffer_bf16_f32 = NULL;
    float* cvt_a_buffer_bf16_f32 = NULL;
    msz_t  mem_b_size_req        = 0;
    msz_t  mem_a_size_req        = 0;
    md_t   packb_min_NR          = 16;

    // Temporary buffer for C accumulation when downscaling is required.
    float* temp_scal_c_buffer_bf16 = NULL;
    msz_t  mem_scale_c_size_req    = 0;

    // kc needs to be a multiple of 2 so that it can be used with dpbf16_ps
    // instruction. Padding is added in cases this condition is not
    // satisfied, and therefore the k offset used for packed/reordered
    // buffer needs to be updated.
    md_t k_updated = k;
    k_updated += (k_updated & 0x1);

    // To decide whether to apply post ops or not.
    bool is_last_k = FALSE;

    // To decide whether to use original s8 C or temp buffer for beta scale.
    bool is_first_k = FALSE;

    lpgemm_post_op_attr post_ops_attr;
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
    /* The thread calculations would still follow DLP_BF16 dimensions*/
    // Generate thrinfo objects for jc and ic loops from lpgemm_thrinfo_t.
    dlp_task_id_t thread_jc;
    dlp_task_id_t thread_ic;

    lpgemm_gen_dlp_task_ids(thread, &thread_jc, &thread_ic);

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
            get_B_panel_reordered_start_offset_width(
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

                if (temp_scal_c_buffer_bf16 == NULL) {
                    dlp_clsc_err_t ret_err;
                    temp_scal_c_buffer_bf16 =
                        dlp_malloc_page_aligned(mem_scale_c_size_req, &ret_err);
                }

                c_use_jc = (float*)temp_scal_c_buffer_bf16;
            }

            // The temp c buffer stride is modified as opposed to original C
            // matrix.
            rs_c_use = nc0;
        }

        for (iter_t pc = 0; pc < k; pc += KC) {
            float beta0 = (pc == 0) ? beta : 1;
            md_t  kc0   = dlp_min((k - pc), KC);

            // No parallelization in k dim, k always starts at 0.
            is_first_k               = (pc == 0) ? (TRUE) : (FALSE);
            post_ops_attr.is_first_k = is_first_k;

            is_last_k               = ((pc + KC) >= k) ? (TRUE) : (FALSE);
            post_ops_attr.is_last_k = is_last_k;

            // kc0 needs to be a multiple of 2 so that it can be
            // used with dpbf16_ps instruction. Padding is added in
            // cases this condition is not satisfied, and therefore
            // the kc0 offsets used for packed/reordered buffers
            // needs to be updated.
            md_t kc0_updated = kc0;
            kc0_updated += (kc0_updated & 0x1);

            // Pack B chunks are based on jc work id.
            md_t jc_work_id = thread_jc.work_id;

            // Using child thrinfo (thread_ic) tid to decide chief thread
            // per B matrix chunk (jc work id group)
            if (dlp_thread_am_ochief(&thread_ic)) {
                if (cvt_b_buffer_bf16_f32 == NULL) {

                    // Calculate the maximum nc0 across potential panel boundary
                    // splits (reordered case). When a thread's jc range crosses
                    // an NC panel boundary, the jc loop splits into multiple
                    // iterations. The first iteration processes up to the panel
                    // boundary (nc0 may be reduced), and subsequent iteration
                    // process the remainder. We allocate for max(current_nc0,
                    // remaining_work) to handle both iterations with a single
                    // buffer allocation.
                    md_t remaining_nc_after_current =
                        dlp_min((jc_end - jc), NC)
                        - nc0; // remaining work after current nc0
                    md_t nc0_buf = dlp_max(remaining_nc_after_current, nc0);

                    // nc0 needs to be a multiple of 16 since this gives maximum
                    // vectorization. Packing B always results in buffers with
                    // width which is a multiple of 16. Subsequently the nc0
                    // offsets used for packed/reordered buffers needs to be
                    // updated.
                    md_t nc0_updated =
                        make_multiple_of_n(nc0_buf, packb_min_NR);

                    mem_b_size_req = sizeof(float) * nc0_updated * kc0_updated;
                    dlp_clsc_err_t ret_err;
                    cvt_b_buffer_bf16_f32 =
                        dlp_malloc_page_aligned(mem_b_size_req, &ret_err);
                }

                thread->comm[jc_work_id].sent_object = cvt_b_buffer_bf16_f32;
            }
            // All threads in work group should wait till chief thread has
            // finished allocating the packing buffers.
            dlp_atomic_barrier(thread_ic.ocomm_id, &thread->comm[jc_work_id]);

            if (mtag_b == PACK) {
                cvt_b_buffer_bf16_f32 =
                    (float*)thread->comm[jc_work_id].sent_object;

                // Compute the B panel per thread loop range for parallel
                // packing using ic_ways number of threads. Since atmost only
                // ic_ways threads can be used, the thread_ic attributes are
                // used to split the loop range.
                md_t jc_packb_start, jc_packb_end;
                dlp_thread_task_range(&thread_ic, nc0, NR, FALSE,
                                      &jc_packb_start, &jc_packb_end);
                // Ensure thread ranges are valid, especially cases where no:
                // of threads available for parallelization are greater than
                // no: of B panel NR chunks.
                if ((jc_packb_end > jc_packb_start)
                    && (jc_packb_start < (jc + nc0))) {
                    rs_b_use = nc0;
                    cs_b_use = 1;
                    cvt_bf16_f32((cvt_b_buffer_bf16_f32 + (jc_packb_start)),
                                 (b + (rs_b * pc) + (cs_b * jc)
                                  + (cs_b * jc_packb_start)),
                                 rs_b, cs_b, kc0,
                                 (jc_packb_end - jc_packb_start), rs_b_use,
                                 cs_b_use);
                } else {
                    rs_b_use = nc0;
                    cs_b_use = 1;
                }

                // All threads in work group should wait till B matrix packing
                // is completed by the participating threads.
                dlp_atomic_barrier(thread_ic.ocomm_id,
                                   &thread->comm[jc_work_id]);

                b_use = cvt_b_buffer_bf16_f32;
            }
            // B part getting processed
            else if (mtag_b == REORDERED) {
                // In multi-threaded scenarios, an extra offset into a given
                // packed B panel is required, since the jc loop split can
                // result in per thread start offset inside the panel, instead
                // of panel boundaries.
                // If B is re-ordered, for DLP_F32 input, the DLP_BF16 data has
                // to be unreordered and coverted to DLP_F32.

                float* b_unreorder =
                    (float*)thread->comm[jc_work_id].sent_object;
                md_t jc_packb_start, jc_packb_end;

                dlp_thread_task_range(&thread_ic, nc0, NR, FALSE,
                                      &jc_packb_start, &jc_packb_end);

                rs_b_use = nc0;
                cs_b_use = 1;
                if ((jc_packb_end > jc_packb_start)
                    && (jc_packb_start < (jc + nc0))) {
                    unpackb_nr64_bf16_f32(
                        (b + (jc_cur_loop * k_updated) + (n_sub_updated * pc)
                         + ((jc_cur_loop_rem + jc_packb_start) * kc0_updated)),
                        (b_unreorder + jc_packb_start), kc0,
                        (jc_packb_end - jc_packb_start), rs_b_use, cs_b_use);
                }

                // All threads in work group should wait till B matrix packing
                // is completed by the participating threads.
                dlp_atomic_barrier(thread_ic.ocomm_id,
                                   &thread->comm[jc_work_id]);
                b_use = b_unreorder;
            }

            for (iter_t ic = ic_start; ic < ic_end; ic += MC) {
                md_t mc0 = dlp_min((ic_end - ic), MC);

                // Only per thread C matrix is stored in temp buffer, so both
                // per thread jc and ic start should be normalized to zero.
                if (c_downscale < DLP_F32) {
                    c_use_ic = c_use_jc + (rs_c_use * (ic - ic_start));
                } else {
                    c_use_ic = c_use_jc + (rs_c_use * ic);
                }

                mem_a_size_req = sizeof(float) * mc0 * kc0;

                // For packed or unpacked A matrix, the mc0 * kc0 block is
                // converted to DLP_F32, i.e., packing has to be done by default
                if (cvt_a_buffer_bf16_f32 == NULL) {
                    dlp_clsc_err_t ret_err;
                    cvt_a_buffer_bf16_f32 =
                        dlp_malloc_page_aligned(mem_a_size_req, &ret_err);
                }

                rs_a_use = kc0;
                cs_a_use = 1;

                cvt_bf16_f32((cvt_a_buffer_bf16_f32),
                             (a + (rs_a * ic) + (cs_a * pc)), rs_a, cs_a, mc0,
                             kc0, rs_a_use, cs_a_use);
                a_use          = cvt_a_buffer_bf16_f32;
                a_block_stride = f32_MR * kc0;

                /*The NR loop should use the DLP_F32 kernel dimesnions*/
                for (iter_t jr = 0; jr < nc0; jr += f32_NR) {
                    md_t nr0 = dlp_min((nc0 - jr), f32_NR);

                    // Post ops meta attributes.
                    post_ops_attr.post_op_c_i    = ic;
                    post_ops_attr.post_op_c_j    = (jc + jr);
                    post_ops_attr.rs_c_downscale = rs_c_downscale;

                    /*To support AVX2, the DLP_F32 kernels are called.*/
                    if (lcntx->dlp_kernel_hndl.kernel_base != NULL) {
                        dlp_execute_kernel(
                            &(lcntx->dlp_kernel_hndl), mc0, nr0, kc0,
                            (float*)a_use, rs_a_use, cs_a_use, a_block_stride,
                            (float*)(b_use + jr), rs_b_use, cs_b_use, 0, 0,
                            (c_use_ic + jr), rs_c_use, 1, (void*)&alpha,
                            (void*)&beta0, post_op_list, post_ops_attr);
                    } else {
                        ((lpgemm_rowvar_f32)lcntx_f32->kern_fun_ptr)(
                            mc0, nr0, kc0, a_use, rs_a_use, cs_a_use,
                            a_block_stride, (b_use + jr), rs_b_use, cs_b_use,
                            (c_use_ic + jr), rs_c_use, 1, alpha, beta0,
                            post_op_list, post_ops_attr);
                    }
                }
            }
        }
        if (mtag_b == REORDERED) {
            adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }

    // Release pack buffers.
    if ((mtag_b == PACK) || (mtag_b == REORDERED)) {
        // All threads in work group should wait till B matrix usage is
        // completed by the participating threads.
        dlp_atomic_barrier(thread_jc.ocomm_id,
                           &thread->comm[thread_jc.work_id]);

        if (dlp_thread_am_ochief(&thread_ic)) {
            if (cvt_b_buffer_bf16_f32 != NULL) {
                dlp_free_page_aligned(cvt_b_buffer_bf16_f32);
            }
        }
    }

    if (cvt_a_buffer_bf16_f32 != NULL) {
        dlp_free_page_aligned(cvt_a_buffer_bf16_f32);
    }

    if (c_downscale < DLP_F32) {
        if (temp_scal_c_buffer_bf16 != NULL) {
            dlp_free_page_aligned(temp_scal_c_buffer_bf16);
        }
    }
}

LPGEMM_5LOOP_UNIFIED(bfloat16, bfloat16, float, float, bf16bf16f32of32,
                     /* mutable */)
{
    // Extract operations from bundle into local variables
    LPGEMM_OPS_EXTRACT(ops);

    if (lcntx->kern_fun_ptr == NULL) {
        lpgemm_rowvar_f32_fallback_bf16bf16f32of32(
            m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c, rs_c,
            cs_c, alpha, beta, rntm, thread, lcntx, post_op_list, c_downscale);
    } else {
        lpgemm_rowvar_avx512bf16_bf16bf16f32of32(
            m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c, rs_c,
            cs_c, alpha, beta, rntm, thread, lcntx, post_op_list, c_downscale);
    }
}
