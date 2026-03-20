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

#include <string.h>

#include "aocl_dlp_gemm_check.h"
#include "classic/aocl_gemm_interface_apis.h"
#include "config/dlp_gemm_config.h"
#include "f32f32f32/dlp_gemm_reorder_f32.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "kernels/f32f32f32/dlp_gemm_pack_f32.h"

#ifdef DLP_ENABLE_OPENMP
#include <omp.h>
#endif

msz_t
aocl_get_reorder_buf_size_f32f32f32of32(const char      order,
                                        const char      trans,
                                        const char      mat_type,
                                        const md_t      k,
                                        const md_t      n,
                                        dlp_metadata_t* metadata)
{
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    // Check if AVX2 ISA is supported, dlp_gemm fp32 matmul only works with it.
    if (dlp_cpuid_is_avx2fma3_supported() == FALSE) {
        dlp_print_msg(" AVX2 ISA not supported by processor, "
                      "cannot perform f32f32f32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return 0; // Error.
    }

    // Initialize dlp_gemm context.
    aocl_lpgemm_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_REORDER_BUF_SIZE_CHECK("f32f32f32of32", order, trans, mat_type, k,
                                    n, err_no);
    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return 0; // Error.
    }

    AOCL_DLP_MATRIX_TYPE input_mat_type;
    dlp_param_map_char_to_lpmat_type(mat_type, &input_mat_type);

    if (input_mat_type == A_MATRIX) {
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return 0; // A reorder not supported.
    }

    const md_t NR = dlp_gemm_get_block_size_NR_global_cntx(F32F32F32OF32);

    // Extra space since packing does width in multiples of NR.
    md_t n_reorder;
    if (n == 1) {
        // When n == 1, LPGEMV doesn't expect B to be reordered.
        n_reorder = 1;
    } else {
        n_reorder = ((n + NR - 1) / NR) * NR;
    }

    msz_t size_req = sizeof(float) * k * n_reorder;

    return size_req;
}

// Pack B into row stored column panels.
void
aocl_reorder_f32f32f32of32(const char      order,
                           const char      trans,
                           const char      mat_type,
                           const float*    input_buf_addr,
                           float*          reorder_buf_addr,
                           const md_t      k,
                           const md_t      n,
                           const md_t      ldb,
                           dlp_metadata_t* metadata)
{
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    // Check if AVX2 ISA is supported, dlp_gemm fp32 matmul only works with it.
    if (dlp_cpuid_is_avx2fma3_supported() == FALSE) {
        dlp_print_msg(" AVX2 ISA not supported by processor, "
                      "cannot perform f32f32f32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return; // Error.
    }

    // Initialize dlp_gemm context.
    aocl_lpgemm_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_REORDER_CHECK("f32f32f32of32", order, trans, mat_type,
                           input_buf_addr, reorder_buf_addr, k, n, ldb, err_no);
    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return; // Error.
    }

    dlp_trans_t dlp_trans;
    /* Map BLAS chars to their corresponding DLP enumerated type value. */
    dlp_param_map_netlib_to_dlp_trans(trans, &dlp_trans);

    md_t rs_b = 0, cs_b = 0;
    if ((order == 'r') || (order == 'R')) {
        rs_b = dlp_is_notrans(dlp_trans) ? ldb : 1;
        cs_b = dlp_is_notrans(dlp_trans) ? 1 : ldb;
    } else if ((order == 'c') || (order == 'C')) {
        rs_b = dlp_is_notrans(dlp_trans) ? 1 : ldb;
        cs_b = dlp_is_notrans(dlp_trans) ? ldb : 1;
    }

    AOCL_DLP_MATRIX_TYPE input_mat_type;
    dlp_param_map_char_to_lpmat_type(mat_type, &input_mat_type);

    if (input_mat_type == A_MATRIX) {
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return; // A reorder not supported.
    }

    // Query the context for various blocksizes.
    dlp_gemm_cntx_t* lcntx = dlp_gemm_get_global_cntx_obj(F32F32F32OF32);
    md_t             NC    = lcntx->blksz.NC;
    md_t             KC    = lcntx->blksz.KC;
    md_t             NR    = lcntx->blksz.NR;

    md_t rs_b_reorder = 0;
    md_t cs_b_reorder = 0;

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    md_t n_threads = rntm_g.num_threads;
    n_threads      = (n_threads > 0) ? n_threads : 1;

    // When n == 1, B marix becomes a vector.
    // Reordering is avoided so that LPGEMV can process it efficiently.
    if (n == 1) {
        if (rs_b == 1) {
            memcpy(reorder_buf_addr, input_buf_addr, (k * sizeof(float)));
        } else {
            for (iter_t k0 = 0; k0 < k; k0++) {
                reorder_buf_addr[k0] = input_buf_addr[k0 * rs_b];
            }
        }
        return;
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
        // Compute the JC loop thread range for the current thread. Per thread
        // gets multiple of NR columns.
        md_t jc_start, jc_end;
        dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);
        for (iter_t jc = jc_start; jc < jc_end; jc += NC) {
            md_t nc0 = dlp_min((jc_end - jc), NC);

            md_t jc_cur_loop     = jc;
            md_t jc_cur_loop_rem = 0;
            md_t n_sub_updated;

            get_B_panel_reordered_start_offset_width(
                jc, n, NC, NR, &jc_cur_loop, &jc_cur_loop_rem, &nc0,
                &n_sub_updated);

            for (iter_t pc = 0; pc < k; pc += KC) {
                md_t kc0 = dlp_min((k - pc), KC);

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
                ((dlp_gemm_pack_f32)lcntx->packb_fun_ptr)(
                    reorder_buf_addr + (jc_cur_loop * k) + (n_sub_updated * pc)
                        + (jc_cur_loop_rem * kc0),
                    input_buf_addr + (rs_b * pc) + (cs_b * jc), rs_b, cs_b, nc0,
                    kc0, &rs_b_reorder, &cs_b_reorder);
            }

            adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }
}

void
aocl_reorder_f32f32f32of32_reference(const char      order,
                                     const char      trans,
                                     const char      mat_type,
                                     const float*    input_buf_addr,
                                     float*          reorder_buf_addr,
                                     const md_t      k,
                                     const md_t      n,
                                     const md_t      ldb,
                                     dlp_metadata_t* metadata)
{
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    // Check if AVX2 ISA is supported, dlp_gemm fp32 matmul only works with it.
    if (dlp_cpuid_is_avx2fma3_supported() == FALSE) {
        dlp_print_msg(" AVX2 ISA not supported by processor, "
                      "cannot perform f32f32f32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return; // Error.
    }

    // Initialize dlp_gemm context.
    aocl_lpgemm_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_REORDER_CHECK("f32f32f32of32_reference", order, trans, mat_type,
                           input_buf_addr, reorder_buf_addr, k, n, ldb, err_no);
    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return; // Error.
    }

    dlp_trans_t dlp_trans;
    /* Map BLAS chars to their corresponding DLP enumerated type value. */
    dlp_param_map_netlib_to_dlp_trans(trans, &dlp_trans);

    md_t rs_b = 0, cs_b = 0;
    if ((order == 'r') || (order == 'R')) {
        rs_b = dlp_is_notrans(dlp_trans) ? ldb : 1;
        cs_b = dlp_is_notrans(dlp_trans) ? 1 : ldb;
    } else if ((order == 'c') || (order == 'C')) {
        rs_b = dlp_is_notrans(dlp_trans) ? 1 : ldb;
        cs_b = dlp_is_notrans(dlp_trans) ? ldb : 1;
    }

    AOCL_DLP_MATRIX_TYPE input_mat_type;
    dlp_param_map_char_to_lpmat_type(mat_type, &input_mat_type);

    if (input_mat_type == A_MATRIX) {
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return; // A reorder not supported.
    }

    // Query the context for various blocksizes.
    dlp_gemm_cntx_t* lcntx = dlp_gemm_get_global_cntx_obj(F32F32F32OF32);
    md_t             NC    = lcntx->blksz.NC;
    md_t             KC    = lcntx->blksz.KC;
    md_t             NR    = lcntx->blksz.NR;

    md_t rs_b_reorder = 0;
    md_t cs_b_reorder = 0;

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    md_t n_threads = rntm_g.num_threads;
    n_threads      = (n_threads > 0) ? n_threads : 1;

    // When n == 1, B marix becomes a vector.
    // Reordering is avoided so that LPGEMV can process it efficiently.
    if (n == 1) {
        if (rs_b == 1) {
            memcpy(reorder_buf_addr, input_buf_addr, (k * sizeof(float)));
        } else {
            for (iter_t k0 = 0; k0 < k; k0++) {
                reorder_buf_addr[k0] = input_buf_addr[k0 * rs_b];
            }
        }
        return;
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
        // Compute the JC loop thread range for the current thread. Per thread
        // gets multiple of NR columns.
        md_t jc_start, jc_end;
        dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);
        for (iter_t jc = jc_start; jc < jc_end; jc += NC) {
            md_t nc0 = dlp_min((jc_end - jc), NC);

            md_t jc_cur_loop     = jc;
            md_t jc_cur_loop_rem = 0;
            md_t n_sub_updated;

            get_B_panel_reordered_start_offset_width(
                jc, n, NC, NR, &jc_cur_loop, &jc_cur_loop_rem, &nc0,
                &n_sub_updated);

            for (iter_t pc = 0; pc < k; pc += KC) {
                md_t kc0 = dlp_min((k - pc), KC);

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
                dlp_packb_f32f32f32of32_reference(
                    reorder_buf_addr + (jc_cur_loop * k) + (n_sub_updated * pc)
                        + (jc_cur_loop_rem * kc0),
                    input_buf_addr + (rs_b * pc) + (cs_b * jc), rs_b, cs_b, nc0,
                    kc0, NR, &rs_b_reorder, &cs_b_reorder);
            }

            adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }
}

void
dlp_unreorderb_nr64_f32f32f32of32_reference(dlp_gemm_obj_t*  b,
                                            dlp_gemm_obj_t*  b_unreorder,
                                            dlp_rntm_t*      rntm_g,
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

    md_t n_threads = rntm_g->num_threads;
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
                jc, n, NC, NR, &jc_cur_loop, &jc_cur_loop_rem, &nc0,
                &n_sub_updated);

            for (iter_t pc = 0; pc < k; pc += KC) {
                md_t kc0 = dlp_min((k - pc), KC);

                dlp_unpackb_f32f32f32of32_reference(
                    ((float*)b_unreorder->storage.aligned_buffer)
                        + (jc_cur_loop * k) + (n_sub_updated * pc)
                        + (jc_cur_loop_rem * kc0),
                    (((float*)b->storage.aligned_buffer) + (rs_b * pc)
                     + (jc * cs_b)),
                    nc0, kc0, NR, rs_b, cs_b);
            }

            adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }
}

void
aocl_unreorder_f32f32f32of32_reference(const char      order,
                                       const char      mat_type,
                                       const float*    reorder_buf_addr,
                                       float*          output_buf_addr,
                                       const md_t      k,
                                       const md_t      n,
                                       const md_t      ldb,
                                       dlp_metadata_t* metadata)
{
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_UNREORDER_CHECK("f32f32f32of32_reference", order, mat_type,
                             reorder_buf_addr, output_buf_addr, k, n, ldb,
                             err_no);
    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return; // Error.
    }

    md_t rs_b = 0, cs_b = 0;

    // Check for the validity of strides.
    if ((order == 'r') || (order == 'R')) {
        rs_b = ldb;
        cs_b = 1;
    } else if ((order == 'c') || (order == 'C')) {
        rs_b = 1;
        cs_b = ldb;
    }

    AOCL_DLP_MATRIX_TYPE input_mat_type;
    dlp_param_map_char_to_lpmat_type(mat_type, &input_mat_type);

    if (input_mat_type == A_MATRIX) {
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return; // A reorder not supported.
    }

#ifdef DLP_KERNELS_ZEN4
    if (n == 1) {
        if (rs_b == 1) {
            memcpy(output_buf_addr, reorder_buf_addr, (k * sizeof(float)));
        } else {
            for (iter_t k0 = 0; k0 < k; k0++) {
                output_buf_addr[k0 * rs_b] = reorder_buf_addr[k0];
            }
        }
        return;
    }
#endif

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(F32F32F32OF32);

    // create dummy b_reorder obj.
    dlp_gemm_obj_t b_reorder;
    b_reorder.storage.aligned_buffer = (void*)reorder_buf_addr;

    // create dummy b obj.
    dlp_gemm_obj_t b;
    b.storage.aligned_buffer = (void*)output_buf_addr;
    b.rs                     = rs_b;
    b.cs                     = cs_b;
    b.width                  = n;
    b.length                 = k;

    dlp_unreorderb_nr64_f32f32f32of32_reference(&b, &b_reorder, &rntm_g,
                                                lcntx_g);
}
