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

/**
 * @file dlp_gemm_f32f16f32.c
 * @brief F32×FP16→F32 mixed-precision GEMM framework (5-loop structure)
 *
 * A and C are F32, B is FP16. Accumulation is F32.
 * B supports PACK and REORDERED modes (packed as FP16 with NR=64).
 * JIT kernel handles vcvtph2ps (FP16→F32) conversion internally.
 */

#include "bindings/c_wrappers/capi_kernel_frame_wrappers.h"
#include "config/dlp_gemm_config.h"
#include "dlp_gemm_5loop_interface_apis.h"
#include "dlp_gemm_ops_bundle.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "kernels/dlp_gemm_kernels.h"
#include "kernels/f32f16f32/dlp_gemm_pack_f16_f32f16.h"
#include "kernels/f32f32f32/dlp_gemm_pack_f32.h"
#include "kernels/fp16fp16fp16/dlp_gemm_pack_fp16.h"
#include "sys_utils/dlp_gemm_sys.h"
#include "threading/dlp_gemm_thread_utils.h"

#ifdef DLP_KERNELS_ZEN4

DLP_GEMV(float, float16, float, f32f16f32of32)
{
    (void)rntm;        /* Threading handled via thread object, not rntm. */
    (void)mtag_a;      /* mtag_a not used in F32xFP16 GEMV path. */
    (void)c_downscale; /* F32xFP16 GEMV always outputs F32. */
    const md_t NC = lcntx->blksz.NC;
    const md_t KC = lcntx->blksz.KC;
    const md_t MC = lcntx->blksz.MC;
    const md_t NR = lcntx->blksz.NR;

    const float* a_use    = (const float*)a;
    md_t         rs_a_use = rs_a;
    md_t         cs_a_use = cs_a;

    float16* b_use    = (float16*)b;
    md_t     rs_b_use = rs_b;
    md_t     cs_b_use = cs_b;

    float*   c_use              = NULL;
    float16* pack_b_buffer_fp16 = NULL;

    msz_t mem_b_size_req = 0;

    md_t packb_min_NR = get_packb_f32f16f32of32_min_NR();

    dlp_gemm_post_op_attr post_ops_attr;
    post_ops_attr.c_stor_type       = DLP_F32;
    post_ops_attr.rs_c_downscale    = rs_c;
    post_ops_attr.cs_c_downscale    = cs_c;
    post_ops_attr.is_first_k        = TRUE;
    post_ops_attr.is_last_k         = TRUE;
    post_ops_attr.b_sum_offset      = 0;
    post_ops_attr.b_col_sum_vec     = NULL;
    post_ops_attr.b_col_sum_vec_s16 = NULL;
    post_ops_attr.buf_downscale     = NULL;

    dlp_task_id_t thread_jc;
    dlp_task_id_t thread_ic;

    dlp_gemm_gen_dlp_task_ids(thread, &thread_jc, &thread_ic);

    if (n == 1) {
        /*
         * n=1 case: y = A * x  (M×K F32 × K×1 FP16 = M×1 F32)
         * Uses dedicated GEMV N=1 kernel (MR=16, NR=1).
         * The GEMV kernel handles its own M-loop and K-loop internally.
         * Framework only does IC parallelization.
         *
         * dlp_execute_kernel creates gemvN1Params when kernel_hndl.nr==1.
         * The GEMV kernel expects:
         *   a = A matrix (F32), x = B vector (FP16), y = C output (F32)
         *   rsA, csA = A strides, rsB/csB = x strides, rsC/csC = y strides
         */
        md_t MR = 16;

        if (mtag_b == REORDERED) {
            rs_b_use = 1;
            cs_b_use = 1;
        } else if (rs_b != 1) {
            mem_b_size_req = sizeof(float16) * k;

            dlp_clsc_err_t ret_err;
            pack_b_buffer_fp16 =
                dlp_malloc_page_aligned(mem_b_size_req, &ret_err);

            for (iter_t k0 = 0; k0 < k; k0++) {
                pack_b_buffer_fp16[k0] = b[k0 * rs_b];
            }

            b_use    = pack_b_buffer_fp16;
            rs_b_use = 1;
            cs_b_use = 1;
        }

        md_t ic_start, ic_end;
        thread_ic.n_way   = (thread_ic.n_way == 1) ? thread->n_threads
                                                   : thread_ic.n_way;
        thread_ic.work_id = thread->tid;
        dlp_thread_task_range(&thread_ic, m, MR, FALSE, &ic_start, &ic_end);

        for (iter_t ic = ic_start; ic < ic_end; ic += MC) {
            md_t         mc0  = dlp_min((ic_end - ic), MC);
            const float* a_ic = a + ic * rs_a;
            c_use             = c + ic * rs_c;

            post_ops_attr.post_op_c_i    = ic;
            post_ops_attr.post_op_c_j    = 0;
            post_ops_attr.rs_c_downscale = rs_c;

            dlp_execute_kernel(&lcntx->dlp_kernel_hndl, mc0, 1, k, (float*)a_ic,
                               rs_a_use, cs_a_use, 1, (float16*)b_use, rs_b_use,
                               cs_b_use, 0, 0, c_use, rs_c, cs_c, (void*)&alpha,
                               (void*)&beta, post_op_list, post_ops_attr);
        }

        if (pack_b_buffer_fp16 != NULL) {
            dlp_free_page_aligned(pack_b_buffer_fp16);
        }

    } else {
        /*
         * m=1 case: y = x * B  (1×K F32 × K×N FP16 = 1×N F32)
         * Uses dedicated GEMV M=1 kernel (MR=1, NR=64).
         * The GEMV kernel handles its own N-loop and K-loop internally.
         * Framework does JC parallelization only; passes full K to kernel.
         *
         * dlp_execute_kernel creates gemvM1Params when kernel_hndl.mr==1.
         * The GEMV kernel expects:
         *   x = A vector (F32), b = B matrix (FP16), y = C output (F32)
         *   rsB = B row stride, n/k = dimensions
         *   n_sub_updated, jc_cur_loop_rem = REORDERED B offsets
         */

        md_t jc_start, jc_end;
        thread_jc.n_way   = (thread_jc.n_way == 1) ? thread->n_threads
                                                   : thread_jc.n_way;
        thread_jc.work_id = thread->tid;
        dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

        md_t k_updated = k;

        post_ops_attr.post_op_c_i = 0;
        post_ops_attr.is_first_k  = TRUE;
        post_ops_attr.is_last_k   = TRUE;

        for (iter_t jc = jc_start; jc < jc_end; jc += NC) {
            md_t nc0 = dlp_min((jc_end - jc), NC);

            md_t jc_cur_loop     = jc;
            md_t jc_cur_loop_rem = 0;
            md_t n_sub_updated   = 0;

            if (mtag_b == REORDERED) {
                dlp_gemm_get_B_panel_reordered_start_offset_width(
                    jc, n, NC, packb_min_NR, &jc_cur_loop, &jc_cur_loop_rem,
                    &nc0, &n_sub_updated);

                b_use = (float16*)(b + (jc_cur_loop * k_updated));
                dlp_gemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);

            } else if (mtag_b == PACK) {
                md_t nc0_updated = dlp_make_multiple_of_n(nc0, packb_min_NR);
                n_sub_updated    = nc0_updated;
                mem_b_size_req   = sizeof(float16) * nc0_updated * k;

                if (pack_b_buffer_fp16 == NULL) {
                    dlp_clsc_err_t ret_err;
                    pack_b_buffer_fp16 =
                        dlp_malloc_page_aligned(mem_b_size_req, &ret_err);
                }

                for (iter_t pc = 0; pc < k; pc += KC) {
                    md_t kc0 = dlp_min((k - pc), KC);

                    ((pack_fp16)lcntx->packb_fun_ptr)(
                        pack_b_buffer_fp16 + (n_sub_updated * pc),
                        (b + (rs_b * pc) + (cs_b * jc)), rs_b, cs_b, nc0, kc0,
                        &rs_b_use, &cs_b_use);
                }

                b_use = pack_b_buffer_fp16;

            } else {
                b_use    = (float16*)(b + jc * cs_b);
                rs_b_use = rs_b;
                cs_b_use = cs_b;
            }

            float* c_use_jc = c + jc;

            post_ops_attr.post_op_c_j    = jc;
            post_ops_attr.rs_c_downscale = rs_c;

            dlp_execute_kernel(&lcntx->dlp_kernel_hndl, 1, nc0, k, (float*)a,
                               rs_a_use, cs_a_use, 1, (float16*)b_use, rs_b_use,
                               cs_b_use, n_sub_updated, jc_cur_loop_rem,
                               c_use_jc, rs_c, 1, (void*)&alpha, (void*)&beta,
                               post_op_list, post_ops_attr);

            if (mtag_b == REORDERED) {
                dlp_gemm_adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
            }
        }

        if (mtag_b == PACK && pack_b_buffer_fp16 != NULL) {
            dlp_free_page_aligned(pack_b_buffer_fp16);
        }
    }
}

DLP_GEMM_5LOOP_UNIFIED(float, float16, float, float, f32f16f32of32,
                       /* mutable */)
{
    DLP_GEMM_OPS_EXTRACT(ops);

    if ((n == 1) || (m == 1)) {
        dlp_gemv_rowvar_f32f16f32of32(
            m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c, rs_c,
            cs_c, alpha, beta, rntm, thread, lcntx, post_op_list, c_downscale);
        return;
    }

    const md_t NC = lcntx->blksz.NC;
    const md_t KC = lcntx->blksz.KC;
    const md_t MC = lcntx->blksz.MC;
    const md_t NR = lcntx->blksz.NR;
    const md_t MR = lcntx->blksz.MR;

    const float* a_use    = NULL;
    md_t         cs_a_use = cs_a;
    md_t         rs_a_use = rs_a;
    md_t         ps_a_use = 0;

    const float16* b_use    = NULL;
    md_t           rs_b_use = rs_b;
    md_t           cs_b_use = cs_b;
    md_t           ps_b_use = 0;

    float* c_use_jc = NULL;
    float* c_use_ic = NULL;
    md_t   rs_c_use = rs_c;

    float*   pack_a_buffer_f32  = NULL;
    float16* pack_b_buffer_fp16 = NULL;
    msz_t    mem_a_size_req     = 0;
    msz_t    mem_b_size_req     = 0;
    md_t     packb_min_NR       = get_packb_f32f16f32of32_min_NR();

    md_t k_updated = k;

    float one_f32 = 1.0f;

    bool is_last_k  = FALSE;
    bool is_first_k = FALSE;

    dlp_gemm_post_op_attr post_ops_attr;
    post_ops_attr.c_stor_type       = DLP_F32;
    post_ops_attr.rs_c_downscale    = rs_c;
    post_ops_attr.cs_c_downscale    = cs_c;
    post_ops_attr.b_sum_offset      = 0;
    post_ops_attr.b_col_sum_vec     = NULL;
    post_ops_attr.b_col_sum_vec_s16 = NULL;
    post_ops_attr.buf_downscale     = NULL;

    dlp_task_id_t thread_jc;
    dlp_task_id_t thread_ic;

    dlp_gemm_gen_dlp_task_ids(thread, &thread_jc, &thread_ic);

    md_t jc_start, jc_end;
    dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

    md_t ic_start, ic_end;
    dlp_thread_task_range(&thread_ic, m, MR, FALSE, &ic_start, &ic_end);

    /* JC loop: iterate over N in NC blocks */
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

        c_use_jc = c + jc;
        rs_c_use = rs_c;

        /* PC loop: iterate over K in KC blocks */
        for (iter_t pc = 0; pc < k; pc += KC) {
            float beta0 = (pc == 0) ? beta : one_f32;
            md_t  kc0   = dlp_min((k - pc), KC);

            is_first_k               = (pc == 0) ? TRUE : FALSE;
            post_ops_attr.is_first_k = is_first_k;

            is_last_k               = ((pc + KC) >= k) ? TRUE : FALSE;
            post_ops_attr.is_last_k = is_last_k;

            md_t kc0_updated = kc0;

            if (mtag_b == REORDERED) {
                /*
                 * REORDERED: B is pre-packed in K-MAJOR layout (FP16).
                 * Offset into the pre-packed buffer.
                 */
                b_use = b + (jc_cur_loop * k_updated) + (n_sub_updated * pc)
                        + (jc_cur_loop_rem * kc0_updated);
                dlp_gemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);
                ps_b_use = kc0_updated;

            } else if (mtag_b == PACK) {
                /*
                 * PACK: Allocate pack buffer, pack B (FP16) into K-MAJOR
                 * layout. Packed strides: rs_b_use = NR, cs_b_use = 1
                 */
                md_t jc_work_id = thread_jc.work_id;

                if (dlp_thread_am_ochief(&thread_ic)) {
                    md_t nc0_updated =
                        dlp_make_multiple_of_n(nc0, packb_min_NR);
                    mem_b_size_req =
                        sizeof(float16) * nc0_updated * kc0_updated;

                    if (pack_b_buffer_fp16 == NULL) {
                        dlp_clsc_err_t ret_err;
                        pack_b_buffer_fp16 =
                            dlp_malloc_page_aligned(mem_b_size_req, &ret_err);
                    }
                    thread->comm[jc_work_id].sent_object = pack_b_buffer_fp16;
                }

                dlp_atomic_barrier(thread_ic.ocomm_id,
                                   &thread->comm[jc_work_id]);
                pack_b_buffer_fp16 =
                    (float16*)thread->comm[jc_work_id].sent_object;

                md_t jc_packb_start, jc_packb_end;
                dlp_thread_task_range(&thread_ic, nc0, NR, FALSE,
                                      &jc_packb_start, &jc_packb_end);

                if ((jc_packb_end > jc_packb_start)
                    && (jc_packb_start < (jc + nc0))) {
                    ((pack_fp16)lcntx->packb_fun_ptr)(
                        pack_b_buffer_fp16 + (jc_packb_start * kc0_updated),
                        (b + (rs_b * pc) + (cs_b * jc)
                         + (cs_b * jc_packb_start)),
                        rs_b, cs_b, (jc_packb_end - jc_packb_start), kc0,
                        &rs_b_use, &cs_b_use);
                } else {
                    dlp_gemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);
                }

                dlp_atomic_barrier(thread_ic.ocomm_id,
                                   &thread->comm[jc_work_id]);
                b_use    = pack_b_buffer_fp16;
                ps_b_use = kc0_updated;

            } else {
                /*
                 * UNPACKED: Access B matrix directly without packing.
                 * B is FP16 - JIT kernel handles vcvtph2ps conversion.
                 */
                b_use    = b + (rs_b * pc) + (cs_b * jc);
                rs_b_use = rs_b;
                cs_b_use = cs_b;
                ps_b_use = cs_b_use;
            }

            /* IC loop: iterate over M in MC blocks */
            for (iter_t ic = ic_start; ic < ic_end; ic += MC) {
                md_t mc0 = dlp_min((ic_end - ic), MC);

                c_use_ic = c_use_jc + (rs_c_use * ic);

                /* A is always UNPACKED for F32×FP16→F32 (no PackA support) */
                a_use    = a + (rs_a * ic) + (cs_a * pc);
                cs_a_use = cs_a;
                ps_a_use = MR * rs_a;
                rs_a_use = rs_a;

                /*
                 * JR loop: iterate over N in NR-sized chunks.
                 * JIT kernel receives:
                 *   - a (F32), b (FP16), c (F32)
                 *   - kernel converts B from FP16→F32 via vcvtph2ps
                 *   - FMA uses vfmadd231ps (F32)
                 */
                for (iter_t jr = 0; jr < nc0; jr += NR) {
                    md_t nr0 = dlp_min((nc0 - jr), NR);

                    post_ops_attr.post_op_c_i = ic;
                    post_ops_attr.post_op_c_j = (jc + jr);

                    const float16* b_jr = b_use + (jr * ps_b_use);

                    dlp_execute_kernel(
                        &lcntx->dlp_kernel_hndl, mc0, nr0, kc0, (float*)a_use,
                        rs_a_use, cs_a_use, ps_a_use, (float16*)b_jr, rs_b_use,
                        cs_b_use, 0, 0, (float*)c_use_ic + jr, rs_c_use, 1,
                        (void*)&alpha, (void*)&beta0, post_op_list,
                        post_ops_attr);
                }
            }
        }

        if (mtag_b == REORDERED) {
            dlp_gemm_adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }

    /* Release pack buffer for B (A is never packed) */
    if (mtag_b == PACK) {
        dlp_atomic_barrier(thread_jc.ocomm_id,
                           &thread->comm[thread_jc.work_id]);

        if (dlp_thread_am_ochief(&thread_ic)) {
            if (pack_b_buffer_fp16 != NULL) {
                dlp_free_page_aligned(pack_b_buffer_fp16);
            }
        }
    }
}

#endif /* DLP_KERNELS_ZEN4 */
