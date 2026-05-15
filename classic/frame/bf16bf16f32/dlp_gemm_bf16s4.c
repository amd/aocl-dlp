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

/*
 * Framework for GEMM with bfloat16 A and signed 4-bit (s4) quantized B
 * (symmetric weight-only quantization). B is dequantized to bf16 via pre-ops
 * (scale only) then the standard bf16*bf16 micro-kernel runs. B must be
 * provided in reordered format.
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

// B should always be packed.
DLP_GEMM_5LOOP_UNIFIED(bfloat16, int8_t, float, float, bf16s4f32of32, const)
{
    (void)rntm; /* Threading handled via thread object, not rntm. */
    // Extract operations from bundle into local variables
    DLP_GEMM_OPS_EXTRACT(ops);

    md_t NC = lcntx->blksz.NC;
    md_t KC = lcntx->blksz.KC;
    md_t MC = lcntx->blksz.MC;
    md_t NR = lcntx->blksz.NR;
    md_t MR = lcntx->blksz.MR;

    md_t group_size = pre_op_list->group_size;

    const int16_t* a_use          = NULL;
    md_t           cs_a_use       = cs_a;
    md_t           rs_a_use       = rs_a;
    md_t           a_block_stride = 0;

    const bfloat16* b_use     = NULL;
    int8_t*         b_reorder = NULL;
    md_t            rs_b_use  = rs_b;
    md_t            cs_b_use  = cs_b;

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

    dlp_gemm_post_op_attr post_ops_attr;
    dlp_gemm_pre_op_attr  pre_ops_attr;

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

    // Initialize group scaling and zero-point params
    pre_ops_attr.scale_factor      = pre_op_list->scale_factor;
    pre_ops_attr.scale_factor_len  = pre_op_list->scale_factor_len;
    pre_ops_attr.scale_factor_type = pre_op_list->scale_factor_type;
    pre_ops_attr.group_size        = group_size;
    pre_ops_attr.pre_op_ld         = n;

    // Generate thrinfo objects for jc and ic loops from dlp_gemm_thrinfo_t.
    dlp_task_id_t thread_jc;
    dlp_task_id_t thread_ic;

    dlp_gemm_gen_dlp_task_ids(thread, &thread_jc, &thread_ic);

    // Compute the JC, IC loop thread range for the current thread.
    md_t jc_start, jc_end;
    dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

    md_t ic_start, ic_end;
    dlp_thread_task_range(&thread_ic, m, MR, FALSE, &ic_start, &ic_end);

    /* Group quantization is not supported for this packing schema */
    if (mtag_b == PACK_NR) {
        /* Allocating private pack buffer of size KCxNR for each thread */
        mem_b_size_req = (KC * NR * sizeof(bfloat16));

        if (pack_b_buffer_bf16 == NULL) {
            dlp_clsc_err_t ret_err;
            pack_b_buffer_bf16 =
                dlp_malloc_page_aligned(mem_b_size_req, &ret_err);
        }
    }

    for (iter_t jc = jc_start; jc < jc_end; jc += NC) {
        md_t nc0         = dlp_min((jc_end - jc), NC);
        md_t nc0_updated = dlp_make_multiple_of_n(nc0, 16);

        md_t jc_cur_loop     = jc;
        md_t jc_cur_loop_rem = 0;
        md_t n_sub_updated   = 0;

        /* B should always be reordered */
        {
            dlp_gemm_get_B_panel_reordered_start_offset_width(
                jc, n, NC, packb_min_NR, &jc_cur_loop, &jc_cur_loop_rem, &nc0,
                &n_sub_updated);

            dlp_gemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);
        }

        // This is for c_downscale == DLP_F32. When c_downscale < DLP_F32,
        // this value will not be used.
        c_use_jc = c + jc;

        // Temp accumulaton buffer for C allocation.
        if (c_downscale < DLP_F32) {
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
                // Temp buffer layout: rows contiguous per thread (stride nc0).
                rs_c_use = nc0;
            }
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

            // B is always supposed to be reordered.
            b_reorder = (int8_t*)b
                        + ((jc_cur_loop * k_updated) + (n_sub_updated * pc)
                           + (jc_cur_loop_rem * kc0_updated))
                              / 2;

            // B matrix will always be packed.
            if (mtag_b == PACK_KC) {
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

                pre_ops_attr.pre_op_b_i = pc;
                pre_ops_attr.pre_op_b_j =
                    jc_cur_loop + jc_cur_loop_rem + jc_packb_start;

                // Ensure thread ranges are valid, especially cases where no:
                // of threads available for parallelization are greater than
                // no: of B panel NR chunks.
                if ((jc_packb_end > jc_packb_start)
                    && (jc_packb_start < (jc + nc0))) {
                    /* Use dedicated nr1 packer for single-column (gemv): output
                     * layout is contiguous (2 bf16 per k-pair), required by
                     * JIT gemvn1; generic packer uses NR-stride layout. */
                    if (n == 1) {
                        dlp_packsclb_nr1_bf16s4f32of32(
                            pack_b_buffer_bf16 + (jc_packb_start * kc0_updated),
                            b_reorder + (jc_packb_start * kc0_updated) / 2, kc0,
                            pre_ops_attr);
                    } else {
                        ((pack_s4bf16)lcntx->packsclb_fun_ptr)(
                            pack_b_buffer_bf16 + (jc_packb_start * kc0_updated),
                            b_reorder + (jc_packb_start * kc0_updated) / 2,
                            (jc_packb_end - jc_packb_start), kc0, &rs_b_use,
                            &cs_b_use, pre_ops_attr);
                    }
                } else {
                    dlp_gemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);
                }

                // All threads in work group should wait till B matrix packing
                // is completed by the participating threads.
                dlp_atomic_barrier(thread_ic.ocomm_id,
                                   &thread->comm[jc_work_id]);
                b_use = pack_b_buffer_bf16;
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

                    if (mtag_b == PACK_NR) {
                        int8_t* b_jr = b_reorder + (jr * kc0_updated) / 2;
                        pre_ops_attr.pre_op_b_i = pc;
                        pre_ops_attr.pre_op_b_j =
                            jc_cur_loop + jc_cur_loop_rem + jr;

                        bfloat16* b_use_jr = pack_b_buffer_bf16;

                        /* packing B at JR level */
                        /* n==1: use nr1 packer for contiguous layout (gemvn1).
                         */
                        if (n == 1) {
                            dlp_packsclb_nr1_bf16s4f32of32(b_use_jr, b_jr, kc0,
                                                           pre_ops_attr);
                        } else {
                            ((pack_s4bf16)lcntx->packsclb_fun_ptr)(
                                b_use_jr, b_jr, nr0, kc0, &rs_b_use, &cs_b_use,
                                pre_ops_attr);
                        }

                        /* packed B kernel */
                        if (lcntx->dlp_kernel_hndl.kernel_base != NULL) {
                            dlp_execute_kernel(
                                &(lcntx->dlp_kernel_hndl), mc0, nr0, kc0,
                                (bfloat16*)a_use, rs_a_use, cs_a_use,
                                a_block_stride, (bfloat16*)b_use_jr, rs_b_use,
                                cs_b_use, 0, 0, (c_use_ic + jr), rs_c_use, 1,
                                (void*)&alpha, (void*)&beta0, post_op_list,
                                post_ops_attr);
                        } else {
                            ((dlp_gemm_rowvar_bf16)lcntx->kern_fun_ptr)(
                                mc0, nr0, kc0, a_use, rs_a_use, cs_a_use,
                                a_block_stride, b_use_jr, rs_b_use, cs_b_use,
                                (c_use_ic + jr), rs_c_use, 1, alpha, beta0,
                                post_op_list, post_ops_attr);
                        }
                    } else if (mtag_b == PACK_KC) {
                        bfloat16* b_use_jr =
                            (bfloat16*)b_use + (jr * kc0_updated);

                        /* packed B kernel */
                        if (lcntx->dlp_kernel_hndl.kernel_base != NULL) {
                            dlp_execute_kernel(
                                &(lcntx->dlp_kernel_hndl), mc0, nr0, kc0,
                                (bfloat16*)a_use, rs_a_use, cs_a_use,
                                a_block_stride, (bfloat16*)b_use_jr, rs_b_use,
                                cs_b_use, 0, 0, (c_use_ic + jr), rs_c_use, 1,
                                (void*)&alpha, (void*)&beta0, post_op_list,
                                post_ops_attr);
                        } else {
                            ((dlp_gemm_rowvar_bf16)lcntx->kern_fun_ptr)(
                                mc0, nr0, kc0, a_use, rs_a_use, cs_a_use,
                                a_block_stride, b_use_jr, rs_b_use, cs_b_use,
                                (c_use_ic + jr), rs_c_use, 1, alpha, beta0,
                                post_op_list, post_ops_attr);
                        }
                    }
#if (defined(DLP_KERNELS_ZEN4))
                    else // mtag_b == UNPACKED
                    {
                        int8_t* b_jr = b_reorder + (jr * kc0_updated) / 2;

                        /* set offsets to determine scale factors and
                         * zero-point values */
                        pre_ops_attr.pre_op_b_i = pc;
                        pre_ops_attr.pre_op_b_j =
                            jc_cur_loop + jc_cur_loop_rem + jr;
                        /* bf16s4f32of32 kernel */
                        dlp_gemm_rowvar_bf16s4f32of32_6x64m(
                            mc0, nr0, kc0, a_use, rs_a_use, cs_a_use,
                            a_block_stride, b_jr, rs_b_use, cs_b_use,
                            (c_use_ic + jr), rs_c_use, 1, alpha, beta0,
                            post_op_list, post_ops_attr, pre_ops_attr);
                    }
#endif
                }
            }
        }
        /* B is always reordered */
        {
            dlp_gemm_adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }

    // Release pack buffers.
    if (mtag_b == PACK_KC) {
        // All threads in work group should wait till B matrix usage is
        // completed by the participating threads.
        dlp_atomic_barrier(thread_jc.ocomm_id,
                           &thread->comm[thread_jc.work_id]);

        if (dlp_thread_am_ochief(&thread_ic)) {
            if (pack_b_buffer_bf16 != NULL) {
                dlp_free_page_aligned(pack_b_buffer_bf16);
            }
        }
    } else if (mtag_b == PACK_NR) {
        /* releasing private B buffer */
        if (pack_b_buffer_bf16 != NULL) {
            dlp_free_page_aligned(pack_b_buffer_bf16);
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
