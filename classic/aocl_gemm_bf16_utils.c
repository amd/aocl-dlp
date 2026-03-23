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
#include "bf16bf16f32/dlp_gemm_reorder_bf16.h"
#include "classic/aocl_gemm_interface_apis.h"
#include "config/dlp_gemm_config.h"
#include "dlp_gemm_types.h"
#include "gemm_utils/dlp_gemm_utils.h"

void
aocl_reorder_bf16bf16f32of32_reference(const char      order,
                                       const char      trans,
                                       const char      mat_type,
                                       const bfloat16* input_buf_addr,
                                       bfloat16*       reorder_buf_addr,
                                       const md_t      k,
                                       const md_t      n,
                                       const md_t      ldb,
                                       dlp_metadata_t* metadata)
{
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_REORDER_CHECK("bf16bf16f32of32_reference", order, trans, mat_type,
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

#if (defined(DLP_KERNELS_ZEN4) && (!defined(DLP_GEMM_BF16_JIT)))
    if (n == 1) {
        if (rs_b == 1) {
            memcpy(reorder_buf_addr, input_buf_addr, (k * sizeof(bfloat16)));
        } else {
            for (iter_t k0 = 0; k0 < k; k0++) {
                reorder_buf_addr[k0] = input_buf_addr[k0 * rs_b];
            }
        }
        return;
    }
#endif
    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(BF16BF16F32OF32);

    // Create dummy b_reorder obj.
    dlp_gemm_obj_t b_reorder;
    b_reorder.storage.aligned_buffer = reorder_buf_addr;

    // Create dummy original b obj;
    dlp_gemm_obj_t b;
    b.storage.aligned_buffer = (void*)input_buf_addr;
    b.rs                     = rs_b;
    b.cs                     = cs_b;
    b.width                  = n;
    b.length                 = k;

    dlp_reorderb_nr64_bf16bf16f32of32_reference(&b, &b_reorder, &rntm_g,
                                                lcntx_g);
}

void
aocl_unreorder_bf16bf16f32of32_reference(const char      order,
                                         const char      mat_type,
                                         const bfloat16* reorder_buf_addr,
                                         bfloat16*       output_buf_addr,
                                         const md_t      k,
                                         const md_t      n,
                                         const md_t      ldb,
                                         dlp_metadata_t* metadata)
{
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_UNREORDER_CHECK("bf16bf16f32of32_reference", order, mat_type,
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

#if (defined(DLP_KERNELS_ZEN4) && (!defined(DLP_GEMM_BF16_JIT)))
    if (n == 1) {
        if (rs_b == 1) {
            memcpy(output_buf_addr, reorder_buf_addr, (k * sizeof(bfloat16)));
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

    dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(BF16BF16F32OF32);

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

    dlp_unreorderb_nr64_bf16bf16f32of32_reference(&b, &b_reorder, &rntm_g,
                                                  lcntx_g);
}

msz_t
aocl_get_reorder_buf_size_bf16bf16f32of32(const char      order,
                                          const char      trans,
                                          const char      mat_type,
                                          const md_t      k,
                                          const md_t      n,
                                          dlp_metadata_t* metadata)
{
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_REORDER_BUF_SIZE_CHECK("bf16bf16f32of32", order, trans, mat_type,
                                    k, n, err_no);
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

    // Extra space since packing does width in multiples of 16. The bf16
    // instruction can be used as long as at least one zmm register can be fully
    // loaded; and since k_dim needs to be at least 2, having n_dim at least 16
    // should give 2x16=32 elements, enough for 1 zmm register.The padding is
    // not rounded to NR (=64), since that would result in memory wastage.
#if (defined(DLP_KERNELS_ZEN4) && (!defined(DLP_GEMM_BF16_JIT)))
    md_t n_reorder;
    /*It is expected that while bf16 input is passed to AVX2 kernels,
      the unreorder/conversion of bf16->f32 is done, which expects the
      reordered matrix to be padded with n multiple of 16, k multiple of 2. */
    if (n == 1) {
        n_reorder = 1;
    } else {
        n_reorder = dlp_make_multiple_of_n(n, 16);
    }

    // Extra space since packing does length in multiples of 2.
    md_t k_reorder;
    if (n == 1) {
        k_reorder = k;
    } else {
        k_reorder = dlp_make_multiple_of_n(k, 2);
    }
#else
    md_t n_reorder = dlp_make_multiple_of_n(n, 16);
    md_t k_reorder = dlp_make_multiple_of_n(k, 2);
#endif
    msz_t size_req = sizeof(int16_t) * k_reorder * n_reorder;

    return size_req;
}

void
aocl_reorder_bf16bf16f32of32(const char      order,
                             const char      trans,
                             const char      mat_type,
                             const bfloat16* input_buf_addr,
                             bfloat16*       reorder_buf_addr,
                             const md_t      k,
                             const md_t      n,
                             const md_t      ldb,
                             dlp_metadata_t* metadata)
{
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    // Check if avx-2 or avx512_bf16 ISA is supported, dlp_gemm matmul only
    // works with it.
    if (dlp_cpuid_is_avx512bf16_supported() == FALSE) {
        if (dlp_cpuid_is_avx2fma3_supported() == FALSE) {
            dlp_print_msg(
                " AVX512_BF16 ISA  and AVX2 ISA not supported by processor, "
                "cannot perform bf16bf16f32/f32f32f32 gemm.",
                __FILE__, __LINE__);
            DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
            return; // Error.
        }

        aocl_reorder_bf16bf16f32of32_reference(order, trans, mat_type,
                                               input_buf_addr, reorder_buf_addr,
                                               k, n, ldb, metadata);

        return;
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_REORDER_CHECK("bf16bf16f32of32", order, trans, mat_type,
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

#if (defined(DLP_KERNELS_ZEN4) && (!defined(DLP_GEMM_BF16_JIT)))
    /*When AOCL_DLP_ENABLE_INSTRUCTIONS=AVX2, f32 kernels would be executed for
      bf16 input, for which re-ordered bf16 input is converted and unreordered
      to hold f32 values. The un-reorder/convert API considers the padded bf16
      input.*/
    if (dlp_gemm_get_enabled_arch() == DLP_ARCH_ZEN3) {
        aocl_reorder_bf16bf16f32of32_reference(order, trans, mat_type,
                                               input_buf_addr, reorder_buf_addr,
                                               k, n, ldb, metadata);
        return;
    }
    if (n == 1) {
        if (rs_b == 1) {
            memcpy(reorder_buf_addr, input_buf_addr, (k * sizeof(bfloat16)));
        } else {
            for (iter_t k0 = 0; k0 < k; k0++) {
                reorder_buf_addr[k0] = input_buf_addr[k0 * rs_b];
            }
        }
        return;
    }
#endif
    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(BF16BF16F32OF32);

    // Create dummy b_reorder obj.
    dlp_gemm_obj_t b_reorder;
    b_reorder.storage.aligned_buffer = reorder_buf_addr;

    // Create dummy original b obj;
    dlp_gemm_obj_t b;
    b.storage.aligned_buffer = (void*)input_buf_addr;
    b.rs                     = rs_b;
    b.cs                     = cs_b;
    b.width                  = n;
    b.length                 = k;

    dlp_reorderb_nr64_bf16bf16f32of32(&b, &b_reorder, &rntm_g, lcntx_g);
}

void
aocl_reorder_f32obf16(const char      order,
                      const char      trans,
                      const char      mat_type,
                      const float*    input_buf_addr,
                      bfloat16*       reorder_buf_addr,
                      const md_t      k,
                      const md_t      n,
                      const md_t      ldb,
                      dlp_metadata_t* metadata)
{
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

#ifdef DLP_GEMM_BF16_JIT
    dlp_print_msg(" f32obf16 is not supported by JIT kernels.", __FILE__,
                  __LINE__);
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
    return;
#endif
    // Check if avx512_bf16 ISA is supported, dlp_gemm matmul only works with
    // it.
    if (dlp_cpuid_is_avx512bf16_supported() == FALSE) {
        dlp_print_msg(" AVX512_BF16 ISA not supported by processor, "
                      "cannot perform bf16bf16f32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_REORDER_CHECK("f32obf16", order, trans, mat_type, input_buf_addr,
                           reorder_buf_addr, k, n, ldb, err_no);
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

#if (defined(DLP_KERNELS_ZEN4))
    if (n == 1) {
        if (rs_b == 1) {
            for (iter_t k0 = 0; k0 < k; k0++) {
                memcpy(&reorder_buf_addr[k0], (char*)(&input_buf_addr[k0]) + 2,
                       sizeof(bfloat16));
            }
        } else {
            for (iter_t k0 = 0; k0 < k; k0++) {
                memcpy(&reorder_buf_addr[k0],
                       (char*)(&input_buf_addr[k0 * rs_b]) + 2,
                       sizeof(bfloat16));
            }
        }
        return;
    }
#endif

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(F32OBF16);

    // Create dummy b_reorder obj.
    dlp_gemm_obj_t b_reorder;
    b_reorder.storage.aligned_buffer = reorder_buf_addr;

    // Create dummy original b obj;
    dlp_gemm_obj_t b;
    b.storage.aligned_buffer = (void*)input_buf_addr;
    b.rs                     = rs_b;
    b.cs                     = cs_b;
    b.width                  = n;
    b.length                 = k;

    dlp_reorderb_mxp_nr64_f32obf16(&b, &b_reorder, &rntm_g, lcntx_g);
}

void
aocl_unreorder_bf16bf16f32of32(const char      order,
                               const char      mat_type,
                               const bfloat16* reorder_buf_addr,
                               bfloat16*       output_buf_addr,
                               const md_t      k,
                               const md_t      n,
                               const md_t      ldb,
                               dlp_metadata_t* metadata)
{
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    // Check if avx512_bf16 ISA is supported, dlp_gemm matmul only works with
    // it.
    if (dlp_cpuid_is_avx512bf16_supported() == FALSE) {
        dlp_print_msg(" AVX512_BF16 ISA not supported by processor, "
                      "cannot perform bf16bf16f32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_UNREORDER_CHECK("bf16bf16f32of32", order, mat_type,
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
#if (defined(DLP_KERNELS_ZEN4) && (!defined(DLP_GEMM_BF16_JIT)))
    if (n == 1) {
        if (rs_b == 1) {
            memcpy(output_buf_addr, reorder_buf_addr, (k * sizeof(bfloat16)));
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

    dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(BF16BF16F32OF32);

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

    dlp_unreorderb_nr64_bf16bf16f32of32(&b, &b_reorder, &rntm_g, lcntx_g);
}

msz_t
aocl_get_reorder_buf_size_bf16s4f32of32(const char      order,
                                        const char      trans,
                                        const char      mat_type,
                                        const md_t      k,
                                        const md_t      n,
                                        dlp_metadata_t* metadata)
{
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    // Check if avx512_bf16 ISA is supported, dlp_gemm matmul only works with
    // it.
    if (dlp_cpuid_is_avx512bf16_supported() == FALSE) {
        dlp_print_msg(" AVX512_BF16 ISA not supported by processor, "
                      "cannot perform bf16bf16f32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return 0; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_REORDER_BUF_SIZE_CHECK("bf16s4f32of32", order, trans, mat_type, k,
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

    md_t n_reorder;

    /*if (n == 1)
    {
        n_reorder = 1;
    }
    else*/
    {
        n_reorder = dlp_make_multiple_of_n(n, 16);
    }

    // Extra space since packing does length in multiples of 2.
    md_t k_reorder;
    /*if (n == 1)
    {
        k_reorder = k;
    }
    else*/
    {
        k_reorder = dlp_make_multiple_of_n(k, 2);
    }

    msz_t size_req = (sizeof(int8_t) * k_reorder * n_reorder) / 2;
    return size_req;
}

void
aocl_reorder_bf16s4f32of32(const char      order,
                           const char      trans,
                           const char      mat_type,
                           const int8_t*   input_buf_addr,
                           int8_t*         reorder_buf_addr,
                           const md_t      k,
                           const md_t      n,
                           const md_t      ldb,
                           dlp_metadata_t* metadata)
{
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    // Check if avx512_bf16 ISA is supported, dlp_gemm matmul only works with
    // it.
    if (dlp_cpuid_is_avx512bf16_supported() == FALSE) {
        dlp_print_msg(" AVX512_BF16 ISA not supported by processor, "
                      "cannot perform bf16bf16f32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_REORDER_CHECK("bf16s4f32of32", order, trans, mat_type,
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
    } else {
        return; // Error
    }

    AOCL_DLP_MATRIX_TYPE input_mat_type;
    dlp_param_map_char_to_lpmat_type(mat_type, &input_mat_type);

    if (input_mat_type == A_MATRIX) {
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return; // A reorder not supported.
    }

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(BF16S4F32OF32);

    // Create dummy b_reorder obj.
    dlp_gemm_obj_t b_reorder;
    b_reorder.storage.aligned_buffer = reorder_buf_addr;

    // Create dummy original b obj;
    dlp_gemm_obj_t b;
    b.storage.aligned_buffer = (void*)input_buf_addr;
    b.rs                     = rs_b;
    b.cs                     = cs_b;
    b.width                  = n;
    b.length                 = k;
    b.mat_type               = input_mat_type;

    dlp_reorderb_nr64_bf16s4f32of32(&b, &b_reorder, &rntm_g, lcntx_g);
}
