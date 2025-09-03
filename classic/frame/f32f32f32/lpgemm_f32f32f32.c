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
#include "kernels/f32f32f32/lpgemm_pack_f32.h"
#include "kernels/lpgemm_kernels.h"
#include "lpgemm_5loop_interface_apis.h"
#include "lpgemm_types.h"
#include "sys_utils/lpgemm_sys.h"
#include "threading/lpgemm_thread_utils.h"

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

typedef void (*lpgemv_a_pack_ft)(float*,
                                 const float*,
                                 const md_t,
                                 const md_t,
                                 const md_t,
                                 const md_t,
                                 md_t*,
                                 md_t*);

LPGEMV(float, float, float, f32f32f32of32)
{
    // Ignoring mtag_a/b and should_pack_A/B for now .
    // Matrices are packed only when the storage format is not supported by the
    // kernel.

    const float* a_use    = (float*)a;
    md_t         rs_a_use = rs_a;
    md_t         cs_a_use = cs_a;

    float* b_use    = (float*)b;
    md_t   rs_b_use = rs_b;
    md_t   cs_b_use = cs_b;

    msz_t mem_a_size_req = 0;
    msz_t mem_b_size_req = 0;

    float* pack_a_buffer_f32f32f32of32 = NULL;
    float* pack_b_buffer_f32f32f32of32 = NULL;

    // Query the context for various blocksizes.
    const md_t NC = lcntx->blksz.NC;
    const md_t KC = lcntx->blksz.KC;
    const md_t MC = lcntx->blksz.MC;
    const md_t NR = lcntx->blksz.NR;

    // Strides are updated based on matrix packing/reordering.
    float* c_use = NULL;

    lpgemm_post_op_attr post_ops_attr;
    post_ops_attr.c_stor_type = c_downscale;
    if (c_downscale < DLP_F32)
        post_ops_attr.buf_downscale = c;
    else
        post_ops_attr.buf_downscale = NULL;

    // Generate thrinfo objects for jc and ic loops from lpgemm_thrinfo_t.
    dlp_task_id_t thread_jc;
    dlp_task_id_t thread_ic;
    lpgemm_gen_dlp_task_ids(thread, &thread_jc, &thread_ic);

    if (n == 1) {
        md_t                MR;
        lpgemv_n_one_ker_ft ker_fp;
        lpgemv_a_pack_ft    packa_fp;

        // Workaround to select right kernel and blocksizes based on arch
        // since GEMV parameters are not available in lpgemm context.
#ifdef DLP_KERNELS_ZEN4
        // NOTE : JIT kernels are not generated when AOCL_ENABLE_INSTRUCTIONS
        //        is set.
        // TODO : Support for detecting the value in AOCL_ENABLE_INSTRUCTIONS,
        //        and generating the appropriate kernels.
        if (dlp_cpuid_is_avx512_supported() == TRUE) {
            if (lpgemm_get_enabled_arch() == DLP_ARCH_ZEN3) {
                MR       = 16;
                ker_fp   = lpgemv_n_one_f32f32f32of32_avx512_256;
                packa_fp = packa_mr8_f32f32f32of32_col_major;
            } else {
                MR       = 16;
                ker_fp   = lpgemv_n_one_f32f32f32of32;
                packa_fp = packa_mr16_f32f32f32of32_col_major;
            }
        } else {
#endif
            //  Increased MR from 6 to 16 to make use of 32 ZMM registers
            MR       = 8;
            ker_fp   = lpgemv_n_one_f32f32f32of32_avx2;
            packa_fp = packa_mr8_f32f32f32of32_col_major;

#ifdef DLP_KERNELS_ZEN4
        }
#endif
        // The vector is already contiguous if reordered.
        if (mtag_b == REORDERED) {
            rs_b_use = 1;
            cs_b_use = 1;
        }
        // Pack B matrix if rs_b > 1
        else if (rs_b != 1) {
            mem_b_size_req = sizeof(float) * k;

            if (pack_b_buffer_f32f32f32of32 == NULL) {
                dlp_clsc_err_t ret_err;
                pack_b_buffer_f32f32f32of32 =
                    dlp_malloc_page_aligned(mem_b_size_req, &ret_err);
            }

            for (md_t k0 = 0; k0 < k; k0++) {
                pack_b_buffer_f32f32f32of32[k0] = b[k0 * rs_b];
            }

            b_use    = pack_b_buffer_f32f32f32of32;
            rs_b_use = 1;
            cs_b_use = 1;
        }
        post_ops_attr.post_op_c_j = 0;

        // Compute the IC loop thread range for the current thread.
        md_t ic_start, ic_end;
        thread_ic.n_way   = (thread_ic.n_way == 1) ? (thread->n_threads)
                                                   : (thread_ic.n_way);
        thread_ic.work_id = thread->tid;
        dlp_thread_task_range(&thread_ic, m, MR, FALSE, &ic_start, &ic_end);

        for (md_t ic = ic_start; ic < ic_end; ic += MC) {
            md_t mc0                  = dlp_min((ic_end - ic), MC);
            a_use                     = a + ic * rs_a;
            c_use                     = c + ic * rs_c;
            post_ops_attr.post_op_c_i = ic;

            // To-Do: pack A case needs to be handled for AVX2 case.
            if (cs_a != 1) {
                mem_a_size_req = sizeof(float) * mc0 * k;

                if (pack_a_buffer_f32f32f32of32 == NULL) {
                    dlp_clsc_err_t ret_err;
                    pack_a_buffer_f32f32f32of32 =
                        dlp_malloc_page_aligned(mem_a_size_req, &ret_err);
                }

                packa_fp(pack_a_buffer_f32f32f32of32, a_use, rs_a, cs_a, mc0, k,
                         &rs_a_use, &cs_a_use);
                a_use = pack_a_buffer_f32f32f32of32;
            }

            if (lcntx->dlp_kernel_hndl.kernel_base != NULL) {
                dlp_execute_kernel(lcntx->dlp_kernel_hndl, mc0, 1, k,
                                   (float*)a_use, rs_a_use, cs_a_use, 1,
                                   (float*)b_use, rs_b_use, cs_b_use, 0, 0,
                                   c_use, rs_c, cs_c, (void*)&alpha,
                                   (void*)&beta, post_op_list, post_ops_attr);
            } else {
                ker_fp(mc0, k, a_use, rs_a_use, cs_a_use, mtag_a, b_use,
                       rs_b_use, cs_b_use, mtag_b, c_use, rs_c, cs_c, alpha,
                       beta, MR, KC, post_op_list, &post_ops_attr);
            }
        }
        if (pack_a_buffer_f32f32f32of32 != NULL) {
            dlp_free_page_aligned(pack_a_buffer_f32f32f32of32);
        }
        if (pack_b_buffer_f32f32f32of32 != NULL) {
            dlp_free_page_aligned(pack_b_buffer_f32f32f32of32);
        }
    } else {
        // m = 1 case is not implemented yet for AVX2

        lpgemv_m_one_ker_ft ker_fp;
        lpgemv_a_pack_ft    packa_fp;

#ifdef DLP_KERNELS_ZEN4
        if (dlp_cpuid_is_avx512_supported() == TRUE) {
            if (lpgemm_get_enabled_arch() == DLP_ARCH_ZEN3) {
                ker_fp   = lpgemv_m_one_f32f32f32of32_avx512_256;
                packa_fp = packa_mr8_f32f32f32of32_col_major;
            } else {
                ker_fp   = lpgemv_m_one_f32f32f32of32;
                packa_fp = packa_mr16_f32f32f32of32_col_major;
            }
        } else {
#endif
            ker_fp   = lpgemv_m_one_f32f32f32of32_avx2;
            packa_fp = packa_mr8_f32f32f32of32_col_major;
#ifdef DLP_KERNELS_ZEN4
        }
#endif
        // Compute the JC loop thread range for the current thread.
        md_t jc_start, jc_end;
        thread_jc.n_way   = (thread_jc.n_way == 1) ? (thread->n_threads)
                                                   : (thread_jc.n_way);
        thread_jc.work_id = thread->tid;
        dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

        if (cs_a != 1) {
            mem_a_size_req = sizeof(float) * k;

            if (pack_a_buffer_f32f32f32of32 == NULL) {
                dlp_clsc_err_t ret_err;
                pack_a_buffer_f32f32f32of32 =
                    dlp_malloc_page_aligned(mem_a_size_req, &ret_err);
            }

            packa_fp(pack_a_buffer_f32f32f32of32, a_use, rs_a, cs_a, 1, k,
                     &rs_a_use, &cs_a_use);

            a_use = pack_a_buffer_f32f32f32of32;
        }
        post_ops_attr.post_op_c_i = 0;

        for (md_t jc = jc_start; jc < jc_end; jc += NC) {
            md_t nc0 = dlp_min((jc_end - jc), NC);
            c_use    = c + jc * cs_c;

            md_t jc_cur_loop     = jc;
            md_t jc_cur_loop_rem = 0;
            md_t n_sub_updated   = 0;

            if (mtag_b == REORDERED) {
                get_B_panel_reordered_start_offset_width(
                    jc, n, NC, NR, &jc_cur_loop, &jc_cur_loop_rem, &nc0,
                    &n_sub_updated);

                b_use = (float*)(b + (jc_cur_loop * k));

                rs_b_use = NR;
                cs_b_use = 1;
            } else if (mtag_b == PACK) {
                // nc0 needs to be a multiple of 16 since this gives maximum
                // vectorization. Packing B always results in buffers with width
                // which is a multiple of 16. Subsequently the nc0 offsets used
                // for packed/reordered buffers needs to be updated.
                md_t nc0_updated = make_multiple_of_n(nc0, NR);

                mem_b_size_req = sizeof(float) * nc0_updated * k;
                n_sub_updated  = nc0_updated;

                if (pack_b_buffer_f32f32f32of32 == NULL) {
                    dlp_clsc_err_t ret_err;
                    pack_b_buffer_f32f32f32of32 =
                        dlp_malloc_page_aligned(mem_b_size_req, &ret_err);
                }

                for (md_t pc = 0; pc < k; pc += KC) {
                    md_t kc0 = dlp_min((k - pc), KC);

                    // Set the strides for pack buffer.
                    rs_b_use = NR;
                    cs_b_use = 1;

                    ((lpgemm_pack_f32)lcntx->packb_fun_ptr)(
                        pack_b_buffer_f32f32f32of32 + (n_sub_updated * pc),
                        b + (rs_b * pc) + (cs_b * jc), rs_b, cs_b, nc0, kc0,
                        &rs_b_use, &cs_b_use);
                }
                b_use = pack_b_buffer_f32f32f32of32;
            } else {
                b_use = (float*)b + jc * cs_b;
            }

            // update post-op pointer
            post_ops_attr.post_op_c_j = jc;

            // Call kernel
            if (lcntx->dlp_kernel_hndl.kernel_base != NULL) {
                dlp_execute_kernel(
                    lcntx->dlp_kernel_hndl, 1, nc0, k, (float*)a_use, rs_a_use,
                    cs_a_use, 1, (float*)b_use, rs_b_use, cs_b_use,
                    n_sub_updated, jc_cur_loop_rem, c_use, rs_c, cs_c,
                    (void*)&alpha, (void*)&beta, post_op_list, post_ops_attr);
            } else {
                ker_fp(nc0, k, a_use, rs_a_use, cs_a_use, mtag_a, b_use,
                       rs_b_use, cs_b_use, mtag_b, c_use, rs_c, cs_c, alpha,
                       beta, NR, KC, n_sub_updated, jc_cur_loop_rem,
                       post_op_list, &post_ops_attr);
            }

            if (mtag_b == REORDERED) {
                adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
            }
        } // jc loop

        // Release pack buffers.
        if ((cs_a != 1) && (pack_a_buffer_f32f32f32of32 != NULL)) {
            dlp_free_page_aligned(pack_a_buffer_f32f32f32of32);
        }
        if ((mtag_b == PACK) && (pack_b_buffer_f32f32f32of32 != NULL)) {
            dlp_free_page_aligned(pack_b_buffer_f32f32f32of32);
        }
    }
}

LPGEMM_5LOOP(float, float, float, f32f32f32of32)
{
    // Handle using LPGEMV when m or/and n equal to 1
    if (((m == 1) || (n == 1))
        && ((dlp_cpuid_is_avx512_supported() == TRUE)
            || (dlp_cpuid_is_avx2fma3_supported() == TRUE))) {
        lpgemv_rowvar_f32f32f32of32(
            m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c, rs_c,
            cs_c, alpha, beta, rntm, thread, lcntx, post_op_list, c_downscale);
        return;
    }

    // Query the context for various blocksizes.
    const md_t NC = lcntx->blksz.NC;
    const md_t KC = lcntx->blksz.KC;
    const md_t MC = lcntx->blksz.MC;
    const md_t NR = (lcntx->dlp_kernel_hndl.kernel_base != NULL)
                        ? lcntx->dlp_kernel_hndl.nr
                        : lcntx->blksz.NR;
    const md_t MR = (lcntx->dlp_kernel_hndl.kernel_base != NULL)
                        ? lcntx->dlp_kernel_hndl.mr
                        : lcntx->blksz.MR;

    // Strides are updated based on matrix packing/reordering.
    const float* a_use    = NULL;
    md_t         rs_a_use = rs_a;
    md_t         cs_a_use = cs_a;

    const float* b_use    = NULL;
    md_t         rs_b_use = rs_b;
    md_t         cs_b_use = cs_b;

    float* c_use_jc = NULL;
    float* c_use_ic = NULL;

    md_t rs_c_downscale = rs_c;

    // Only supporting row major with unit column strided C for now.
    const md_t cs_c_use = 1;

    /* Compute partitioning step values for each matrix of each loop. */
    md_t ps_a_use;
    md_t ps_b_use;

    // Check if packing of A is required.
    // TODO: mtag_a for tranpose needs to be honored.
    bool should_pack_A = rntm->pack_a;

    // Pack buffer for A.
    float* pack_a_buffer_f32f32f32of32 = NULL;
    msz_t  mem_a_size_req              = 0;

    // Check if packing of B is required.
    bool should_pack_B = rntm->pack_b;

    // Pack buffer for B.
    float* pack_b_buffer_f32f32f32of32 = NULL;
    msz_t  mem_b_size_req              = 0;

    float one_local = 1.0f;

    // To decide whether to apply post ops or not.
    bool is_last_k = FALSE;

    // To decide whether to use original s8 C or temp buffer for beta scale.
    bool is_first_k = FALSE;

    lpgemm_post_op_attr post_ops_attr;
    post_ops_attr.c_stor_type = c_downscale;
    if (c_downscale < DLP_F32) {
        post_ops_attr.buf_downscale = c;
    } else {
        post_ops_attr.buf_downscale = NULL;
    }

    // Generate thrinfo objects for jc and ic loops from lpgemm_thrinfo_t.
    dlp_task_id_t thread_jc;
    dlp_task_id_t thread_ic;

    lpgemm_gen_dlp_task_ids(thread, &thread_jc, &thread_ic);

    // Compute the JC loop thread range for the current thread.
    md_t jc_start, jc_end;
    dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

    // Compute the IC loop thread range for the current thread.
    md_t ic_start, ic_end;
    dlp_thread_task_range(&thread_ic, m, MR, FALSE, &ic_start, &ic_end);

    // Update the kernel pointer with right kernel
    lpgemm_rowvar_f32 ker_ptr = (lpgemm_rowvar_f32)lcntx->kern_fun_ptr;

    // Avoid packing of B in transb cases where rd kernels performs
    // better than rv + pack. rv kernel calls rd when rs_b==1.
    bool invoke_rd = FALSE;
    if ((lpgemm_get_enabled_arch() != DLP_ARCH_ZEN3) && ((n < 48) || (m < 16))
        && (rs_b == 1) && (mtag_b == PACK) && (mtag_a == UNPACKED)) {
        invoke_rd     = TRUE;
        mtag_b        = UNPACKED;
        should_pack_A = FALSE;
    }

    for (md_t jc = jc_start; jc < jc_end; jc += NC) {
        md_t nc0 = dlp_min((jc_end - jc), NC);
        c_use_jc = c + jc;

        md_t jc_cur_loop     = jc;
        md_t jc_cur_loop_rem = 0;
        md_t n_sub_updated   = 0;

        if (mtag_b == REORDERED) {
            get_B_panel_reordered_start_offset_width(
                jc, n, NC, NR, &jc_cur_loop, &jc_cur_loop_rem, &nc0,
                &n_sub_updated);
        }

        for (md_t pc = 0; pc < k; pc += KC) {
            float beta0 = (pc == 0) ? beta : one_local;
            md_t  kc0   = dlp_min((k - pc), KC);

            // No parallelization in k dim, k always starts at 0.
            is_first_k               = (pc == 0) ? (TRUE) : (FALSE);
            post_ops_attr.is_first_k = is_first_k;

            is_last_k               = ((pc + KC) >= k) ? (TRUE) : (FALSE);
            post_ops_attr.is_last_k = is_last_k;

            if (mtag_b == REORDERED) {
                // In multi-threaded scenarios, an extra offset into a given
                // packed B panel is required, since the jc loop split can
                // result in per thread start offset inside the panel, instead
                // of panel boundaries.
                b_use = b + (jc_cur_loop * k) + (n_sub_updated * pc)
                        + (jc_cur_loop_rem * kc0);

                rs_b_use = NR;
                cs_b_use = 1;
                ps_b_use = kc0;
            } else if ((mtag_b == PACK) || (should_pack_B == TRUE)) {
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
                    md_t nc0_updated = make_multiple_of_n(nc0, NR);
                    mem_b_size_req   = sizeof(float) * nc0_updated * kc0;

                    // The largest value for mem_b_size_req will be the first
                    // time pack_b_buffer_f32f32f32of32 is allocated. So no
                    // realloc required.
                    if (pack_b_buffer_f32f32f32of32 == NULL) {
                        dlp_clsc_err_t ret_err;
                        pack_b_buffer_f32f32f32of32 =
                            dlp_malloc_page_aligned(mem_b_size_req, &ret_err);
                    }

                    thread->comm[jc_work_id].sent_object =
                        pack_b_buffer_f32f32f32of32;
                }

                // All threads in work group should wait till chief thread has
                // finished allocating the packing buffers.
                dlp_atomic_barrier(thread_ic.ocomm_id,
                                   &thread->comm[jc_work_id]);

                pack_b_buffer_f32f32f32of32 =
                    (float*)thread->comm[jc_work_id].sent_object;
                // Set the strides for pack buffer.
                rs_b_use = NR;
                cs_b_use = 1;
                ps_b_use = kc0;

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
                    ((lpgemm_pack_f32)lcntx->packb_fun_ptr)(
                        pack_b_buffer_f32f32f32of32 + (jc_packb_start * kc0),
                        b + (rs_b * pc) + (cs_b * jc) + (cs_b * jc_packb_start),
                        rs_b, cs_b, (jc_packb_end - jc_packb_start), kc0,
                        &rs_b_use, &cs_b_use);
                } else {
                    lpgemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);
                }

                // All threads in work group should wait till B matrix packing
                // is completed by the participating threads.
                dlp_atomic_barrier(thread_ic.ocomm_id,
                                   &thread->comm[jc_work_id]);
                b_use = pack_b_buffer_f32f32f32of32;
            } else {
                b_use    = b + (pc * rs_b) + (jc * cs_b);
                ps_b_use = 1;
                if (invoke_rd == TRUE) {
                    ps_b_use = cs_b_use;
                }
            }

            for (md_t ic = ic_start; ic < ic_end; ic += MC) {
                md_t mc0 = dlp_min((ic_end - ic), MC);
                c_use_ic = c_use_jc + (rs_c * ic);

                if (mtag_a == REORDERED) {
                    // Extra space since packing does width in multiples of MR.
                    const md_t m_updated = ((m + MR - 1) / MR) * MR;
                    a_use                = a + (pc * m_updated) + (kc0 * ic);

                    rs_a_use = 1;
                    cs_a_use = MR;
                    ps_a_use = MR * kc0;
                } else if (should_pack_A == TRUE) {
                    // Extra space since packing does width in multiples of MR.
                    const md_t mc0_updated = ((mc0 + MR - 1) / MR) * MR;
                    mem_a_size_req         = sizeof(float) * mc0_updated * kc0;

                    if (pack_a_buffer_f32f32f32of32 == NULL) {
                        dlp_clsc_err_t ret_err;
                        pack_a_buffer_f32f32f32of32 =
                            dlp_malloc_page_aligned(mem_a_size_req, &ret_err);
                    }

                    rs_a_use = 1;
                    cs_a_use = MR;
                    ps_a_use = MR * kc0;

                    ((lpgemm_pack_f32)lcntx->packa_fun_ptr)(
                        pack_a_buffer_f32f32f32of32,
                        (a + (rs_a * ic) + (pc * cs_a)), rs_a, cs_a, mc0, kc0,
                        &rs_a_use, &cs_a_use);

                    a_use = pack_a_buffer_f32f32f32of32;
                } else {
                    a_use    = a + (rs_a * ic) + (pc * cs_a);
                    ps_a_use = MR * rs_a;
                }

                for (md_t jr = 0; jr < nc0; jr += NR) {
                    md_t nr0 = dlp_min((nc0 - jr), NR);

                    // Post ops meta attributes.
                    post_ops_attr.post_op_c_i    = ic;
                    post_ops_attr.post_op_c_j    = (jc + jr);
                    post_ops_attr.rs_c_downscale = rs_c_downscale;

                    // Call the micro-kernel
                    // TODO: Remove this once the generation of rd kernels
                    // is supported in JIT.
                    if ((lcntx->dlp_kernel_hndl.kernel_base != NULL)
                        && (!invoke_rd)) {
                        dlp_execute_kernel(
                            lcntx->dlp_kernel_hndl, mc0, nr0, kc0,
                            (float*)a_use, rs_a_use, cs_a_use, ps_a_use,
                            (float*)(b_use + (jr * ps_b_use)), rs_b_use,
                            cs_b_use, 0, 0, (c_use_ic + jr), rs_c, cs_c_use,
                            (void*)&alpha, (void*)&beta0, post_op_list,
                            post_ops_attr);
                    } else {
                        ker_ptr(mc0, nr0, kc0, (float*)a_use, rs_a_use,
                                cs_a_use, ps_a_use,
                                (float*)(b_use + (jr * ps_b_use)), rs_b_use,
                                cs_b_use, (c_use_ic + jr), rs_c, cs_c_use,
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
    if ((mtag_b == PACK) || (should_pack_B == TRUE)) {
        // All threads in work group should wait till B matrix usage is
        // completed by the participating threads.
        dlp_atomic_barrier(thread_jc.ocomm_id,
                           &thread->comm[thread_jc.work_id]);

        if (dlp_thread_am_ochief(&thread_ic)) {
            if (pack_b_buffer_f32f32f32of32 != NULL) {
                dlp_free_page_aligned(pack_b_buffer_f32f32f32of32);
            }
        }
    }
    if (should_pack_A == TRUE) {
        if (pack_a_buffer_f32f32f32of32 != NULL) {
            dlp_free_page_aligned(pack_a_buffer_f32f32f32of32);
        }
    }
}
