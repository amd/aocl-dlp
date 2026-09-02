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
#include "gemm_utils/dlp_gemm_reorder_layout.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "kernels/f32f32f32/dlp_gemm_pack_f32.h"
#include "threading/dlp_gemm_thread_utils.h"

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

    // Size is layout arithmetic; the tuned reorder still requires AVX2.

    // Initialize dlp_gemm context.
    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_REORDER_BUF_SIZE_CHECK("f32f32f32of32", order, trans, mat_type, k,
                                    n, err_no);
    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return 0; // Error.
    }

    AOCL_DLP_MATRIX_TYPE input_mat_type;
    dlp_param_map_char_to_lpmat_type(mat_type, &input_mat_type);

    if (input_mat_type != B_MATRIX) {
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return 0; // Only B is supported.
    }

    dlp_gemm_cntx_t lcntx_l = *(dlp_gemm_get_global_cntx_obj(F32F32F32OF32));
    err_no = dlp_gemm_upd_cntx_with_metadata(F32F32F32OF32, &lcntx_l, metadata);
    if (err_no != DLP_CLSC_SUCCESS) {
        dlp_print_msg(" Failed to update context with metadata.", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return 0; // Error.
    }

    err_no = dlp_gemm_validate_metadata_with_lcntx(metadata, &lcntx_l);
    if (err_no != DLP_CLSC_SUCCESS) {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "Local cntx diverged or corrupted from metadata, "
                 "local cntx values -> MC: %ld, NC: %ld, KC: %ld, "
                 "MR: %ld, NR: %ld\n",
                 lcntx_l.blksz.MC, lcntx_l.blksz.NC, lcntx_l.blksz.KC,
                 lcntx_l.blksz.MR, lcntx_l.blksz.NR);
        dlp_print_msg(msg, __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return 0; // Error.
    }
    const md_t NR = lcntx_l.blksz.NR;
    if (!dlp_reorder_ref_blocks_legal(NR, NR, lcntx_l.blksz.KC, 1)) {
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_INVALID_BLOCK_PARAMS);
        return 0;
    }

    // Extra space since packing does width in multiples of NR.
    md_t n_reorder;
    if (n == 1) {
        // When n == 1, DLP_GEMV doesn't expect B to be reordered.
        n_reorder = 1;
    } else {
        n_reorder = ((n + NR - 1) / NR) * NR;
    }

    msz_t size_req = 0;
    if (!dlp_reorder_size_bytes(k, n_reorder, (md_t)sizeof(float), 0,
                                &size_req)) {
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_INVALID_MATRIX_DIMENSION);
        return 0;
    }

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
    dlp_init_global_cntx();

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
    dlp_gemm_cntx_t lcntx = *(dlp_gemm_get_global_cntx_obj(F32F32F32OF32));
    err_no = dlp_gemm_upd_cntx_with_metadata(F32F32F32OF32, &lcntx, metadata);
    if (err_no != DLP_CLSC_SUCCESS) {
        dlp_print_msg(" Failed to update context with metadata.", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return; // Error.
    }
    md_t rs_b_reorder = 0;
    md_t cs_b_reorder = 0;

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    md_t n_threads = rntm_g.num_threads;
    n_threads      = (n_threads > 0) ? n_threads : 1;

    // When n == 1, B marix becomes a vector.
    // Reordering is avoided so that DLP_GEMV can process it efficiently.
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

    // JIT pack B: required for F32 GEMM reorder path.
    ((lcntx.dlp_pack_kernel_hndl).pack_b_hndl).kernel_base = NULL;
    dlp_init_and_get_packb_kernel_hndl(DLP_KERNEL_F32F32F32OF32, n, k, rs_b,
                                       cs_b, &lcntx);

    if (((lcntx.dlp_pack_kernel_hndl).pack_b_hndl).kernel_base == NULL) {
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_INVALID_JIT_KERNEL);
        return;
    }

    // The init is done, so the panel width the reorder writes is final.
    dlp_upd_pack_strides(DLP_KERNEL_F32F32F32OF32, &lcntx);

    err_no = dlp_gemm_validate_metadata_with_lcntx(metadata, &lcntx);
    if (err_no != DLP_CLSC_SUCCESS) {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "Local cntx diverged or corrupted from metadata, "
                 "local cntx values -> MC: %ld, NC: %ld, KC: %ld, "
                 "MR: %ld, NR: %ld\n",
                 lcntx.blksz.MC, lcntx.blksz.NC, lcntx.blksz.KC, lcntx.blksz.MR,
                 lcntx.blksz.NR);
        dlp_print_msg(msg, __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return; // Error.
    }

    md_t NC = lcntx.blksz.NC;
    md_t KC = lcntx.blksz.KC;
    md_t NR = lcntx.blksz.NR;

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

            dlp_gemm_get_B_panel_reordered_start_offset_width(
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
                dlp_execute_packb_kernel(
                    (lcntx.dlp_pack_kernel_hndl).pack_b_hndl,
                    (void*)(input_buf_addr + (rs_b * pc) + (cs_b * jc)),
                    (void*)(reorder_buf_addr + (jc_cur_loop * k)
                            + (n_sub_updated * pc) + (jc_cur_loop_rem * kc0)),
                    nc0, kc0, rs_b, cs_b, &rs_b_reorder, &cs_b_reorder);
            }

            dlp_gemm_adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
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

    // No ISA gate: this is portable C, unlike aocl_reorder_f32f32f32of32.

    // Initialize dlp_gemm context.
    dlp_init_global_cntx();

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

    if (input_mat_type != B_MATRIX) {
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return; // Only B is supported.
    }

    // Query the context for various blocksizes.
    dlp_gemm_cntx_t lcntx = *(dlp_gemm_get_global_cntx_obj(F32F32F32OF32));
    err_no = dlp_gemm_upd_cntx_with_metadata(F32F32F32OF32, &lcntx, metadata);
    if (err_no != DLP_CLSC_SUCCESS) {
        dlp_print_msg(" Failed to update context with metadata.", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return; // Error.
    }

    err_no = dlp_gemm_validate_metadata_with_lcntx(metadata, &lcntx);
    if (err_no != DLP_CLSC_SUCCESS) {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "Local cntx diverged or corrupted from metadata, "
                 "local cntx values -> MC: %ld, NC: %ld, KC: %ld, "
                 "MR: %ld, NR: %ld\n",
                 lcntx.blksz.MC, lcntx.blksz.NC, lcntx.blksz.KC, lcntx.blksz.MR,
                 lcntx.blksz.NR);
        dlp_print_msg(msg, __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return; // Error.
    }
    md_t NC = lcntx.blksz.NC;
    md_t KC = lcntx.blksz.KC;
    md_t NR = lcntx.blksz.NR;

    if (!dlp_reorder_ref_blocks_legal(NR, NR, KC, 1)) {
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_INVALID_BLOCK_PARAMS);
        return;
    }

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    // When n == 1, B marix becomes a vector.
    // Reordering is avoided so that DLP_GEMV can process it efficiently.
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

    dlp_reorder_ref_reorderb(reorder_buf_addr, input_buf_addr, sizeof(float), 1,
                             n, k, NR, NR, NC, KC, rs_b, cs_b,
                             rntm_g.num_threads);
}

void
aocl_unreorder_f32f32f32of32_reference(const char      order,
                                       const char      trans,
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
    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_UNREORDER_CHECK("f32f32f32of32_reference", order, trans, mat_type,
                             reorder_buf_addr, output_buf_addr, k, n, ldb,
                             err_no);
    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return; // Error.
    }

    dlp_trans_t dlp_trans;
    dlp_param_map_netlib_to_dlp_trans(trans, &dlp_trans);

    md_t rs_b = 0, cs_b = 0;
    dlp_reorder_plain_strides(order, dlp_trans, ldb, &rs_b, &cs_b);

    AOCL_DLP_MATRIX_TYPE input_mat_type;
    dlp_param_map_char_to_lpmat_type(mat_type, &input_mat_type);

    if (input_mat_type != B_MATRIX) {
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return; // Only B is supported.
    }

    // n == 1 is a vector: GEMV wants it unreordered, and the size API
    // allocates k elements with no NR pad.
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

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    dlp_gemm_cntx_t lcntx_g = *(dlp_gemm_get_global_cntx_obj(F32F32F32OF32));
    err_no = dlp_gemm_upd_cntx_with_metadata(F32F32F32OF32, &lcntx_g, metadata);
    if (err_no != DLP_CLSC_SUCCESS) {
        dlp_print_msg(" Failed to update context with metadata.", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return; // Error.
    }

    err_no = dlp_gemm_validate_metadata_with_lcntx(metadata, &lcntx_g);
    if (err_no != DLP_CLSC_SUCCESS) {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "Local cntx diverged or corrupted from metadata, "
                 "local cntx values -> MC: %ld, NC: %ld, KC: %ld, "
                 "MR: %ld, NR: %ld\n",
                 lcntx_g.blksz.MC, lcntx_g.blksz.NC, lcntx_g.blksz.KC,
                 lcntx_g.blksz.MR, lcntx_g.blksz.NR);
        dlp_print_msg(msg, __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return; // Error.
    }

    if (!dlp_reorder_ref_blocks_legal(lcntx_g.blksz.NR, lcntx_g.blksz.NR,
                                      lcntx_g.blksz.KC, 1)) {
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_INVALID_BLOCK_PARAMS);
        return;
    }

    dlp_reorder_ref_unreorderb(output_buf_addr, reorder_buf_addr, sizeof(float),
                               1, n, k, lcntx_g.blksz.NR, lcntx_g.blksz.NR,
                               lcntx_g.blksz.NC, lcntx_g.blksz.KC, rs_b, cs_b,
                               rntm_g.num_threads);
}
