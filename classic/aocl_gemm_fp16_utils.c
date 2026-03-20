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

#include <string.h>

#include "aocl_dlp_gemm_check.h"
#include "classic/aocl_fp16_type.h"
#include "classic/aocl_gemm_interface_apis.h"
#include "config/dlp_gemm_config.h"
#include "fp16fp16fp16/dlp_gemm_reorder_fp16.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "kernels/fp16fp16fp16/dlp_gemm_pack_fp16.h"

#ifdef DLP_ENABLE_OPENMP
#include <omp.h>
#endif

msz_t
aocl_get_reorder_buf_size_f16f16f16of16(const char      order,
                                        const char      trans,
                                        const char      mat_type,
                                        const md_t      k,
                                        const md_t      n,
                                        dlp_metadata_t* metadata)
{
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    // Check if AVX-512-FP16 ISA is supported, dlp_gemm fp16 reorder requires
    // it.
    if (dlp_cpuid_is_avx512fp16_supported() == FALSE) {
        dlp_print_msg(" AVX-512-FP16 ISA not supported by processor, "
                      "cannot perform f16f16f16of16 reorder.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return 0;
    }

    // Initialize dlp_gemm context.
    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_REORDER_BUF_SIZE_CHECK("f16f16f16of16", order, trans, mat_type, k,
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

    const md_t NR = dlp_gemm_get_block_size_NR_global_cntx(F16F16F16OF16);

    // Extra space since packing does width in multiples of NR (packb_min_NR).
    // One ZMM register = 32 FP16 elements.
    md_t n_reorder;
    if (n == 1) {
        // When n == 1, LPGEMV doesn't expect B to be reordered.
        n_reorder = 1;
    } else {
        n_reorder = make_multiple_of_n(n, NR);
    }

    msz_t size_req = sizeof(float16) * k * n_reorder;
    return size_req;
}

// Pack B into row stored column panels.
void
aocl_reorder_f16f16f16of16(const char      order,
                           const char      trans,
                           const char      mat_type,
                           const float16*  input_buf_addr,
                           float16*        reorder_buf_addr,
                           const md_t      k,
                           const md_t      n,
                           const md_t      ldb,
                           dlp_metadata_t* metadata)
{
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    // Check if AVX-512-FP16 ISA is supported, dlp_gemm fp16 reorder requires
    // it.
    if (dlp_cpuid_is_avx512fp16_supported() == FALSE) {
        dlp_print_msg(" AVX-512-FP16 ISA not supported by processor, "
                      "cannot perform f16f16f16of16 reorder.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return;
    }

    // Initialize dlp_gemm context.
    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_REORDER_CHECK("f16f16f16of16", order, trans, mat_type,
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

    // When n == 1, B matrix becomes a vector.
    // Reordering is avoided so that LPGEMV can process it efficiently.
    if (n == 1) {
        if (rs_b == 1) {
            memcpy(reorder_buf_addr, input_buf_addr, (k * sizeof(float16)));
        } else {
            for (iter_t k0 = 0; k0 < k; k0++) {
                reorder_buf_addr[k0] = input_buf_addr[k0 * rs_b];
            }
        }
        return;
    }

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(F16F16F16OF16);

    // Create dummy b_reorder obj.
    dlp_gemm_obj_t b_reorder;
    b_reorder.storage.aligned_buffer = reorder_buf_addr;

    // Create dummy original b obj.
    dlp_gemm_obj_t b;
    b.storage.aligned_buffer = (void*)input_buf_addr;
    b.rs                     = rs_b;
    b.cs                     = cs_b;
    b.width                  = n;
    b.length                 = k;

    dlp_reorderb_nr128_f16f16f16of16(&b, &b_reorder, &rntm_g, lcntx_g);
}

void
aocl_unreorder_f16f16f16of16(const char      order,
                             const char      mat_type,
                             const float16*  reorder_buf_addr,
                             float16*        output_buf_addr,
                             const md_t      k,
                             const md_t      n,
                             const md_t      ldb,
                             dlp_metadata_t* metadata)
{
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    if (dlp_cpuid_is_avx512fp16_supported() == FALSE) {
        dlp_print_msg(" AVX-512-FP16 ISA not supported by processor, "
                      "cannot perform f16f16f16of16 unreorder.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return;
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_UNREORDER_CHECK("f16f16f16of16", order, mat_type, reorder_buf_addr,
                             output_buf_addr, k, n, ldb, err_no);
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

    // When n == 1, B matrix becomes a vector.
    // Reordering is avoided so that LPGEMV can process it efficiently.
    if (n == 1) {
        if (rs_b == 1) {
            memcpy(output_buf_addr, reorder_buf_addr, (k * sizeof(float16)));
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

    dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(F16F16F16OF16);

    // Create dummy b_reorder obj.
    dlp_gemm_obj_t b_reorder;
    b_reorder.storage.aligned_buffer = (void*)reorder_buf_addr;

    // Create dummy b obj.
    dlp_gemm_obj_t b;
    b.storage.aligned_buffer = output_buf_addr;
    b.rs                     = rs_b;
    b.cs                     = cs_b;
    b.width                  = n;
    b.length                 = k;

    dlp_unreorderb_nr128_f16f16f16of16(&b, &b_reorder, &rntm_g, lcntx_g);
}
