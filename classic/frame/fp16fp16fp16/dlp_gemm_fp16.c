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
 * @file dlp_gemm_fp16.c
 * @brief FP16 GEMM framework layer implementing the 5-loop structure
 *
 * FP16 supports three memory modes for matrix B:
 * - UNPACKED: B accessed directly with user-provided strides
 * - PACK: B packed into K-MAJOR layout in local buffer
 * - REORDERED: B is pre-packed externally in K-MAJOR layout
 *
 * Unlike BF16, FP16 can support true UNPACKED because:
 * - FP16 uses native vfmadd231ph instruction (no k-pairing required)
 * - Each FP16 element is independent (no 2-element grouping like BF16's
 * dpbf16_ps)
 */

#include "bindings/c_wrappers/capi_kernel_frame_wrappers.h"
#include "classic/aocl_fp16_convert.h"
#include "config/dlp_gemm_config.h"
#include "dlp_gemm_5loop_interface_apis.h"
#include "dlp_gemm_ops_bundle.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "kernels/dlp_gemm_kernels.h"
#include "kernels/fp16fp16fp16/dlp_gemm_pack_fp16.h"
#include "sys_utils/dlp_gemm_sys.h"
#include "threading/dlp_gemm_thread_utils.h"

#ifdef DLP_KERNELS_ZEN4

/*
 * DLP_GEMV: FP16 GEMV (General Matrix-Vector Multiplication)
 *
 * Handles two cases:
 * - n=1: y = A * x (M x K matrix * K x 1 vector = M x 1 vector)
 * - m=1: y = x * B (1 x K vector * K x N matrix = 1 x N vector)
 *
 * Optimizations:
 * - n=1: MR=16 for better register utilization, parallelize along M (IC loop)
 * - m=1: NR=128 from GEMM, parallelize along N (JC loop) with K-blocking
 */
DLP_GEMV(float16, float16, float16, f16f16f16of16)
{
    const md_t NC = lcntx->blksz.NC;
    const md_t KC = lcntx->blksz.KC;
    const md_t MC = lcntx->blksz.MC;
    const md_t NR = lcntx->blksz.NR;

    float16* a_use    = (float16*)a;
    md_t     rs_a_use = rs_a;
    md_t     cs_a_use = cs_a;

    float16* b_use    = (float16*)b;
    md_t     rs_b_use = rs_b;
    md_t     cs_b_use = cs_b;

    float16* c_use              = NULL;
    float16* pack_a_buffer_fp16 = NULL;
    float16* pack_b_buffer_fp16 = NULL;

    msz_t mem_a_size_req = 0;
    msz_t mem_b_size_req = 0;

    md_t packb_min_NR = 32;

    /* C element byte-size: 2 for of16 (float16), 4 for of32 (in-place
       float). The GEMV signature pins C to float16* (rail-shared) but on
       the of32 rail the user buffer is actually float*; advance pointers
       by element-count using this byte-size factor so identical math
       drives both rails (mirrors the GEMM 5-loop's c_elem_size). */
    const msz_t c_elem_size = (c_downscale == DLP_F32) ? sizeof(float)
                                                       : sizeof(float16);

    /* Beta widened once at entry — the GEMV M1/N1 of32 JIT path reads
       beta via vbroadcastss in F32 (just like the GEMM of32 rail). */
    const float beta_f32 = fp16_to_f32(beta);

    dlp_gemm_post_op_attr post_ops_attr;
    post_ops_attr.c_stor_type       = c_downscale;
    post_ops_attr.rs_c_downscale    = rs_c;
    post_ops_attr.cs_c_downscale    = cs_c;
    post_ops_attr.is_first_k        = TRUE;
    post_ops_attr.is_last_k         = TRUE;
    post_ops_attr.b_sum_offset      = 0;
    post_ops_attr.b_col_sum_vec     = NULL;
    post_ops_attr.b_col_sum_vec_s16 = NULL;

    post_ops_attr.buf_downscale = c;

    /* Generate thrinfo objects for jc and ic loops */
    dlp_task_id_t thread_jc;
    dlp_task_id_t thread_ic;

    dlp_gemm_gen_dlp_task_ids(thread, &thread_jc, &thread_ic);

    if (n == 1) {
        /*
         * n=1 case: y = A * x
         * Increased MR from 6 to 16 for better register utilization
         */
        md_t MR = 16;

        /*
         * Pack B if not contiguous (rs_b > 1).
         * For GEMV n=1, B is a column vector and the optimized kernel
         * expects contiguous data. Pack B into a temporary buffer when
         * elements are strided (rs_b != 1).
         */

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

        /* Thread partitioning along M (IC loop) */
        md_t ic_start, ic_end;
        thread_ic.n_way   = (thread_ic.n_way == 1) ? thread->n_threads
                                                   : thread_ic.n_way;
        thread_ic.work_id = thread->tid;
        dlp_thread_task_range(&thread_ic, m, MR, FALSE, &ic_start, &ic_end);

        /* IC loop */
        for (iter_t ic = ic_start; ic < ic_end; ic += MC) {
            md_t           mc0  = dlp_min((ic_end - ic), MC);
            const float16* a_ic = a + ic * rs_a;
            /* Type-aware C-pointer arithmetic — see c_elem_size note. */
            c_use = (float16*)((char*)c + ic * rs_c * c_elem_size);
            post_ops_attr.post_op_c_i    = ic;
            post_ops_attr.post_op_c_j    = 0;
            post_ops_attr.rs_c_downscale = rs_c;

            if (mtag_a == PACK) {
                mem_a_size_req = sizeof(float16) * mc0 * k;

                if (pack_a_buffer_fp16 == NULL) {
                    dlp_clsc_err_t ret_err;
                    pack_a_buffer_fp16 =
                        dlp_malloc_page_aligned(mem_a_size_req, &ret_err);
                }

                ((pack_fp16)lcntx->packa_fun_ptr)(pack_a_buffer_fp16,
                                                  (a + (rs_a * ic)), rs_a, cs_a,
                                                  mc0, k, &rs_a_use, &cs_a_use);
                a_ic = pack_a_buffer_fp16;
            }

            /* Beta-pointer is rail-aware: of32 reads f32 (vbroadcastss),
               of16 reads f16 (vpbroadcastw). Alpha stays as f16 because
               the kernel widens it internally on of32 (vcvtph2ps). */
            const void* beta_arg = (c_downscale == DLP_F32)
                                       ? (const void*)&beta_f32
                                       : (const void*)&beta;
            dlp_execute_kernel(&(lcntx->dlp_kernel_hndl), mc0, 1, k,
                               (float16*)a_ic, rs_a_use, cs_a_use, 1,
                               (float16*)b_use, rs_b_use, cs_b_use, 0, 0, c_use,
                               rs_c, cs_c, (void*)&alpha, (void*)beta_arg,
                               post_op_list, post_ops_attr);
        }

        /* Release pack buffers */
        if (mtag_a == PACK && pack_a_buffer_fp16 != NULL) {
            dlp_free_page_aligned(pack_a_buffer_fp16);
        }
        if (pack_b_buffer_fp16 != NULL) {
            dlp_free_page_aligned(pack_b_buffer_fp16);
        }

    } else {
        /*
         * m=1 case: y = x * B
         * Use NR=128 from GEMM, parallelize along N (JC loop)
         *
         * Design Pattern (matching F32):
         * - Framework does NOT do KC blocking
         * - Framework passes full k to kernel
         * - Kernel handles KC blocking internally with REORDERED/PACK/UNPACKED
         * - This ensures consistent framework design across data types
         */

        /* Thread partitioning along N (JC loop) */
        md_t jc_start, jc_end;
        thread_jc.n_way   = (thread_jc.n_way == 1) ? thread->n_threads
                                                   : thread_jc.n_way;
        thread_jc.work_id = thread->tid;
        dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

        md_t k_updated = k;

        /* Pack A (single row) if needed - pack full K at once */
        if (mtag_a == PACK) {
            mem_a_size_req = sizeof(float16) * k;

            dlp_clsc_err_t ret_err;
            pack_a_buffer_fp16 =
                dlp_malloc_page_aligned(mem_a_size_req, &ret_err);

            ((pack_fp16)lcntx->packa_fun_ptr)(pack_a_buffer_fp16, a, rs_a, cs_a,
                                              1, k, &rs_a_use, &cs_a_use);

            a_use = pack_a_buffer_fp16;
        }

        post_ops_attr.post_op_c_i = 0;
        post_ops_attr.is_first_k  = TRUE;
        post_ops_attr.is_last_k   = TRUE;

        /* JC loop */
        for (iter_t jc = jc_start; jc < jc_end; jc += NC) {
            md_t nc0 = dlp_min((jc_end - jc), NC);
            /* Type-aware C-pointer arithmetic — see c_elem_size note. */
            c_use = (float16*)((char*)c + jc * cs_c * c_elem_size);

            md_t jc_cur_loop     = jc;
            md_t jc_cur_loop_rem = 0;
            md_t n_sub_updated   = 0;

            if (mtag_b == REORDERED) {
                dlp_gemm_get_B_panel_reordered_start_offset_width(
                    jc, n, NC, packb_min_NR, &jc_cur_loop, &jc_cur_loop_rem,
                    &nc0, &n_sub_updated);

                /* For REORDERED, pass base pointer - kernel handles KC
                 * internally */
                b_use = (float16*)(b + (jc_cur_loop * k_updated));
                dlp_gemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);

            } else if (mtag_b == PACK) {
                /*
                 * PACK: Pack B for this JC panel (all of K at once)
                 * Following F32 pattern - pack all KC blocks before kernel call
                 */
                md_t nc0_updated = dlp_make_multiple_of_n(nc0, packb_min_NR);
                mem_b_size_req   = sizeof(float16) * nc0_updated * k;
                n_sub_updated    = nc0_updated;

                if (pack_b_buffer_fp16 == NULL) {
                    dlp_clsc_err_t ret_err;
                    pack_b_buffer_fp16 =
                        dlp_malloc_page_aligned(mem_b_size_req, &ret_err);
                }

                /* Pack all KC blocks for this JC panel */
                for (iter_t pc = 0; pc < k; pc += KC) {
                    md_t kc0 = dlp_min((k - pc), KC);

                    ((pack_fp16)lcntx->packb_fun_ptr)(
                        pack_b_buffer_fp16 + (n_sub_updated * pc),
                        (b + (rs_b * pc) + (jc * cs_b)), rs_b, cs_b, nc0, kc0,
                        &rs_b_use, &cs_b_use);
                }

                b_use = pack_b_buffer_fp16;

            } else {
                /* UNPACKED: Pass base pointer - kernel handles KC internally */
                b_use    = (float16*)(b + jc * cs_b);
                rs_b_use = rs_b;
                cs_b_use = cs_b;
            }

            post_ops_attr.post_op_c_j    = jc;
            post_ops_attr.rs_c_downscale = rs_c;

            /*
             * Call GEMV m=1 kernel with FULL k (no KC blocking in framework)
             * JIT only - no intrinsic fallback
             */
            const void* beta_arg = (c_downscale == DLP_F32)
                                       ? (const void*)&beta_f32
                                       : (const void*)&beta;
            dlp_execute_kernel(
                &(lcntx->dlp_kernel_hndl), 1, nc0, k, (float16*)a_use, rs_a_use,
                cs_a_use, 1, (float16*)b_use, rs_b_use, cs_b_use, n_sub_updated,
                jc_cur_loop_rem, c_use, rs_c, cs_c, (void*)&alpha,
                (void*)beta_arg, post_op_list, post_ops_attr);

            if (mtag_b == REORDERED) {
                dlp_gemm_adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
            }
        }

        /* Release pack buffers */
        if (mtag_b == PACK && pack_b_buffer_fp16 != NULL) {
            dlp_free_page_aligned(pack_b_buffer_fp16);
        }
        if (mtag_a == PACK && pack_a_buffer_fp16 != NULL) {
            dlp_free_page_aligned(pack_a_buffer_fp16);
        }
    }
}

DLP_GEMM_5LOOP_UNIFIED(float16, float16, float16, float16, f16f16f16of16,
                       /* mutable */)
{
    // Extract operations from bundle into local variables
    DLP_GEMM_OPS_EXTRACT(ops);

    // The of32 JIT post-ops read beta through vbroadcastss + vfmadd231ps,
    // which expects a 32-bit float; on the of16 rail beta stays FP16
    // (vpbroadcastw + vmulph). Widen beta once here for of32 and pick the
    // right pointer at the kernel call site below.
    const float beta_f32 = fp16_to_f32(beta);

    // GEMV-shaped redirect: the dedicated FP16 GEMV kernels handle both
    // output rails - see the c_downscale handling inside
    // dlp_gemv_rowvar_f16f16f16of16.
    if ((n == 1) || (m == 1)) {
        dlp_gemv_rowvar_f16f16f16of16(
            m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c, rs_c,
            cs_c, alpha, beta, rntm, thread, lcntx, post_op_list, c_downscale);
        return;
    }

    const md_t NC = lcntx->blksz.NC;
    const md_t KC = lcntx->blksz.KC;
    const md_t MC = lcntx->blksz.MC;
    const md_t NR = lcntx->blksz.NR;
    const md_t MR = lcntx->blksz.MR;

    const float16* a_use    = NULL;
    md_t           cs_a_use = cs_a;
    md_t           rs_a_use = rs_a;
    md_t           ps_a_use = 0;

    const float16* b_use    = NULL;
    md_t           rs_b_use = rs_b;
    md_t           cs_b_use = cs_b;
    md_t           ps_b_use = 0; /* Panel stride for JR loop B offset */

    /* C element byte-size: 2 for of16 (float16), 4 for of32 (float). The
       5-loop signature pins C to float16* but on the of32 rail the user
       buffer is actually float*; advance pointers by element-count using
       this byte-size factor so the math is identical for both rails and
       follows the same pattern as the f32f32f32 5-loop (cs_c_use). */
    const msz_t c_elem_size = (c_downscale == DLP_F32) ? sizeof(float)
                                                       : sizeof(float16);

    float16* c_use_jc = NULL;
    float16* c_use_ic = NULL;
    md_t     rs_c_use = rs_c;

    float16* pack_a_buffer_fp16 = NULL;
    float16* pack_b_buffer_fp16 = NULL;
    msz_t    mem_a_size_req     = 0;
    msz_t    mem_b_size_req     = 0;
    md_t     packb_min_NR       = 32;

    md_t k_updated = k;

    /* Per-rail KC-block-1 beta-suppress sentinels. On the of16 rail beta is
       fp16 and the kernel reads vpbroadcastw; on of32 beta is widened to
       f32 once at entry and the kernel reads vbroadcastss. Pre-compute one
       sentinel of each type and select per-pc below. */
    const float16 one_fp16 = FP16_ONE;
    const float   one_f32  = 1.0f;

    bool is_last_k  = FALSE;
    bool is_first_k = FALSE;

    dlp_gemm_post_op_attr post_ops_attr;
    post_ops_attr.c_stor_type       = c_downscale;
    post_ops_attr.rs_c_downscale    = rs_c;
    post_ops_attr.cs_c_downscale    = cs_c;
    post_ops_attr.b_sum_offset      = 0;
    post_ops_attr.b_col_sum_vec     = NULL;
    post_ops_attr.b_col_sum_vec_s16 = NULL;
    // The of32 post-ops rail writes user F32 C in place through regCPtr
    // (the kernel signature's float16* C handle, re-cast to float* by the
    // JIT under c_downscale = DLP_F32). buf_downscale is unused on this
    // rail - we forward the C pointer for symmetry with the of16 path and
    // for any downstream post-op helpers that read it.
    post_ops_attr.buf_downscale = c;

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

        /* C-pointer advance in C-elements: byte-scale by c_elem_size so
           the same math works for of16 (2B) and of32 (4B). */
        c_use_jc = (float16*)((char*)c + jc * c_elem_size);
        rs_c_use = rs_c;

        /* PC loop: iterate over K in KC blocks */
        for (iter_t pc = 0; pc < k; pc += KC) {
            /* Per-KC beta selection (mirrors f32f32f32 / bf16 5-loops):
               first KC uses the user beta, intermediate KCs use 1.0 so the
               accumulator carried in C across KC blocks is preserved
               (additive only, never re-scaled by user beta). */
            const float16 beta0_fp16 = (pc == 0) ? beta : one_fp16;
            const float   beta0_f32  = (pc == 0) ? beta_f32 : one_f32;
            md_t          kc0        = dlp_min((k - pc), KC);

            is_first_k               = (pc == 0) ? TRUE : FALSE;
            post_ops_attr.is_first_k = is_first_k;

            is_last_k               = ((pc + KC) >= k) ? TRUE : FALSE;
            post_ops_attr.is_last_k = is_last_k;

            md_t kc0_updated = kc0;

            if (mtag_b == REORDERED) {
                /*
                 * REORDERED: B is pre-packed in K-MAJOR layout.
                 * Use pre-packed buffer with strides from context.
                 */
                b_use = b + (jc_cur_loop * k_updated) + (n_sub_updated * pc)
                        + (jc_cur_loop_rem * kc0_updated);
                dlp_gemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);
                ps_b_use = kc0_updated;

            } else if (mtag_b == PACK) {
                /*
                 * PACK: Allocate pack buffer, pack B into K-MAJOR layout.
                 * Packed strides: rs_b_use = NR, cs_b_use = 1
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
                 * Use user-provided strides: rs_b_use = rs_b (ldb),
                 * cs_b_use = cs_b (1 for row-major).
                 * No pack buffer allocation needed.
                 */
                b_use    = b + (rs_b * pc) + (cs_b * jc);
                rs_b_use = rs_b;
                cs_b_use = cs_b;
                ps_b_use = cs_b_use; /* Panel stride for JR loop */
            }

            /* IC loop: iterate over M in MC blocks */
            for (iter_t ic = ic_start; ic < ic_end; ic += MC) {
                md_t mc0 = dlp_min((ic_end - ic), MC);

                /* IC step in C-elements (rs_c_use = ldc), byte-scaled by
                   c_elem_size for rail-agnostic pointer math. */
                c_use_ic =
                    (float16*)((char*)c_use_jc + (rs_c_use * ic) * c_elem_size);

                if (mtag_a == UNPACKED) {
                    a_use    = a + (rs_a * ic) + (cs_a * pc);
                    cs_a_use = cs_a;      /* Usually 1 for row-major */
                    ps_a_use = MR * rs_a; /* Stride to next MR-row block */
                    rs_a_use = rs_a;      /* lda */
                } else if (mtag_a == PACK) {
                    /*
                     * PACK: Allocate pack buffer, pack A into M-MAJOR layout.
                     * PackA uses MR=32 blocks and pads mc0 to MR boundary.
                     * Buffer size: ((mc0 + MR - 1) / MR) * MR * KC
                     */
                    md_t mc0_padded = ((mc0 + MR - 1) / MR) * MR;
                    mem_a_size_req  = sizeof(float16) * mc0_padded * kc0;

                    if (pack_a_buffer_fp16 == NULL) {
                        dlp_clsc_err_t ret_err;
                        pack_a_buffer_fp16 =
                            dlp_malloc_page_aligned(mem_a_size_req, &ret_err);
                    }

                    ((pack_fp16)lcntx->packa_fun_ptr)(
                        pack_a_buffer_fp16, (a + (rs_a * ic) + (cs_a * pc)),
                        rs_a, cs_a, mc0, kc0, &rs_a_use, &cs_a_use);
                    a_use = pack_a_buffer_fp16;
                    /*
                     * Panel stride for packed A: MR * rs_a_use (kc0)
                     * The kernel processes MR rows at a time, and each row has
                     * rs_a_use (= kc0) elements in M-MAJOR layout.
                     */
                    ps_a_use = MR * rs_a_use;
                }

                /*
                 * JR loop: iterate over N in NR-sized chunks.
                 * The pack function packs all columns in a panel together
                 * with stride = NR_out (returned as rs_b_use).
                 * For PACK/REORDERED: rs_b_use is the actual packed width
                 *   (128 for full panels, or smaller for partial panels)
                 * For UNPACKED: rs_b_use = rs_b (user's ldb)
                 */
                for (iter_t jr = 0; jr < nc0; jr += NR) {
                    md_t nr0 = dlp_min((nc0 - jr), NR);

                    post_ops_attr.post_op_c_i = ic;
                    post_ops_attr.post_op_c_j = (jc + jr);

                    const float16* b_jr = b_use + (jr * ps_b_use);

                    /* JR step in C-elements, byte-scaled. */
                    float16* c_jr =
                        (float16*)((char*)c_use_ic + jr * c_elem_size);

                    /* Per-rail beta pointer: kernel reads it via
                       vpbroadcastw (of16) or vbroadcastss (of32). Both
                       point at the per-KC-suppressed value. */
                    void* beta_arg = (c_downscale == DLP_F32)
                                         ? (void*)&beta0_f32
                                         : (void*)&beta0_fp16;

                    dlp_execute_kernel(&(lcntx->dlp_kernel_hndl), mc0, nr0, kc0,
                                       (float16*)a_use, rs_a_use, cs_a_use,
                                       ps_a_use, (float16*)b_jr, rs_b_use,
                                       cs_b_use, 0, 0, c_jr, rs_c_use, 1,
                                       (void*)&alpha, beta_arg, post_op_list,
                                       post_ops_attr);
                }
            }
        }

        if (mtag_b == REORDERED) {
            dlp_gemm_adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }

    /* Release pack buffers - only for PACK mode */
    if (mtag_b == PACK) {
        dlp_atomic_barrier(thread_jc.ocomm_id,
                           &thread->comm[thread_jc.work_id]);

        if (dlp_thread_am_ochief(&thread_ic)) {
            if (pack_b_buffer_fp16 != NULL) {
                dlp_free_page_aligned(pack_b_buffer_fp16);
            }
        }
    }

    if (mtag_a == PACK && pack_a_buffer_fp16 != NULL) {
        dlp_free_page_aligned(pack_a_buffer_fp16);
    }
}

#endif /* DLP_KERNELS_ZEN4 */
