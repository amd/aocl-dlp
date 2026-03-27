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
#include "classic/dlp_errors.h"
#include "config/dlp_gemm_config.h"
#include "dlp_gemm_types.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "s8s8s32/dlp_gemm_reorder_s8.h"

msz_t
aocl_get_reorder_buf_size_s8s8s32os32(const char      order,
                                      const char      trans,
                                      const char      mat_type,
                                      const md_t      k,
                                      const md_t      n,
                                      dlp_metadata_t* metadata)
{
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    // Check if avx512_vnni ISA is supported, dlp_gemm matmul only works with
    // it.
    if (dlp_cpuid_is_avx512vnni_supported() == FALSE) {
        dlp_print_msg(" AVX512_VNNI ISA not supported by processor, "
                      "cannot perform s8s8s32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return 0; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_REORDER_BUF_SIZE_CHECK("s8s8s32os32", order, trans, mat_type, k, n,
                                    err_no);
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

    // Extra space since packing does width in multiples of 16. The vnni
    // instruction can be used as long as atleast one zmm register can be fully
    // loaded; and since k_dim needs to be atleast 4, having n_dim atleast 16
    // should give 4x16=64 elements, enough for 1 zmm register.The padding is
    // not rounded to NR (=64), since that would result in memory wastage.
#ifdef DLP_KERNELS_ZEN4
    md_t n_reorder;
    if (n == 1) {
        n_reorder = 1;
    } else {
        n_reorder = dlp_make_multiple_of_n(n, 16);
    }

    // Extra space since packing does length in multiples of 4.
    md_t k_reorder;
    if (n == 1) {
        k_reorder = k;
    } else {
        k_reorder = dlp_make_multiple_of_n(k, 4);
    }
#else
    md_t n_reorder = dlp_make_multiple_of_n(n, 16);
    md_t k_reorder = dlp_make_multiple_of_n(k, 4);
#endif
    // extra memory of n_reorder * sizeof(int32_t) to store sum of every column
    // of B matrix buffer
    msz_t size_req =
        sizeof(int8_t) * k_reorder * n_reorder + (n_reorder * sizeof(int32_t));

    return size_req;
}

msz_t
aocl_get_reorder_buf_size_s8s8s32os32_sym_quant(
    const char           order,
    const char           trans,
    const char           mat_type,
    const md_t           k,
    const md_t           n,
    DLP_SYMM_STAT_QUANT* symq_meta_data,
    dlp_metadata_t*      metadata)
{
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    // Check if avx512_vnni ISA is supported, dlp_gemm matmul only works with
    // it.
    if (dlp_cpuid_is_avx512vnni_supported() == FALSE) {
        dlp_print_msg(" AVX512_VNNI ISA not supported by processor, "
                      "cannot perform s8s8s32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return 0; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_REORDER_BUF_SIZE_CHECK("s8s8s32os32_sym_quant", order, trans,
                                    mat_type, k, n, err_no);
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

    // Extra space since packing does width in multiples of 16. The vnni
    // instruction can be used as long as atleast one zmm register can be fully
    // loaded; and since k_dim needs to be atleast 4, having n_dim atleast 16
    // should give 4x16=64 elements, enough for 1 zmm register.The padding is
    // not rounded to NR (=64), since that would result in memory wastage.
#ifdef DLP_KERNELS_ZEN4
    md_t n_reorder;
#if 0
    if (n == 1) {
        n_reorder = 1;
    } else
#endif
    {
        n_reorder = dlp_make_multiple_of_n(n, 16);
    }

    // Extra space since packing does length in multiples of 4.
    md_t k_reorder;
#if 0
    if (n == 1) {
        k_reorder = k;
    } else
#endif
    {
        k_reorder = dlp_make_multiple_of_n(k, 4);
    }
#else
    md_t n_reorder = dlp_make_multiple_of_n(n, 16);
    md_t k_reorder = dlp_make_multiple_of_n(k, 4);
#endif
    md_t group_size = symq_meta_data->group_size;

    if (group_size & 3) {
        dlp_print_msg(
            " Group size should be multiple of 4 for s8s8s32os32_sym_quant",
            __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_INVALID_GROUP_DIMENSION);
        return 0; // Error.
    }

    md_t num_groups = (k + group_size - 1) / group_size;

    // extra memory to store sum of every column per group of B matrix buffer
    size_t extra_mem_req = num_groups * n_reorder * sizeof(int32_t);

    // extra memory of n_reorder * sizeof(int32_t) to store sum of every column
    // of B matrix buffer
    msz_t size_req = sizeof(int8_t) * k_reorder * n_reorder + extra_mem_req;

    return size_req;
}

void
aocl_reorder_s8s8s32os32(const char      order,
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

    // Check if avx512_vnni ISA is supported, dlp_gemm matmul only works with
    // it.
    if (dlp_cpuid_is_avx512vnni_supported() == FALSE) {
        dlp_print_msg(" AVX512_VNNI ISA not supported by processor, "
                      "cannot perform s8s8s32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_REORDER_CHECK("s8s8s32os32", order, trans, mat_type,
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
#ifdef DLP_KERNELS_ZEN4
    if (n == 1) {
        int32_t* pack_b_column_sum =
            (int32_t*)(reorder_buf_addr + (sizeof(int8_t) * n * k));

        *pack_b_column_sum = 0;

        for (iter_t k0 = 0; k0 < k; k0++) {
            reorder_buf_addr[k0] = input_buf_addr[k0 * rs_b];
            *pack_b_column_sum += reorder_buf_addr[k0];
        }
        *pack_b_column_sum *= 128;
        return;
    }
#endif
    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(S8S8S32OS32);

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

    dlp_reorderb_nr64_s8s8s32o32(&b, &b_reorder, &rntm_g, lcntx_g);
}

void
aocl_reorder_s8s8s32os32_sym_quant(const char           order,
                                   const char           trans,
                                   const char           mat_type,
                                   const int8_t*        input_buf_addr,
                                   int8_t*              reorder_buf_addr,
                                   const md_t           k,
                                   const md_t           n,
                                   const md_t           ldb,
                                   DLP_SYMM_STAT_QUANT* symq_meta_data,
                                   dlp_metadata_t*      metadata)
{
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    // Check if avx512_vnni ISA is supported, dlp_gemm matmul only works with
    // it.
    if (dlp_cpuid_is_avx512vnni_supported() == FALSE) {
        dlp_print_msg(" AVX512_VNNI ISA not supported by processor, "
                      "cannot perform s8s8s32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return; // Error.
    }

    md_t group_size = symq_meta_data->group_size;
    if (group_size & 3) {
        dlp_print_msg(
            " Group size should be multiple of 4 for s8s8s32os32_sym_quant",
            __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_INVALID_GROUP_DIMENSION);
        return; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_REORDER_CHECK("s8s8s32os32_sym_quant", order, trans, mat_type,
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

#ifdef DLP_KERNELS_ZEN4
#if 0
    if (n == 1) {
        // Calculate the address of the beginning of the column sum buffer that
        // is allocated after the reorder buffer.
        int32_t* pack_b_column_sum =
            (int32_t*)(reorder_buf_addr + (k * sizeof(int8_t)));

        // NOTE We're working under the assumption that group_size is a factor
        // of k.
        for (iter_t k0 = 0; k0 < k; k0 += group_size) {
            // Initialize the current column sum to 0.
            *pack_b_column_sum = 0;
            for (iter_t group = 0; group < group_size; group++) {
                reorder_buf_addr[k0 + group] =
                    input_buf_addr[(k0 + group) * rs_b];
                *pack_b_column_sum += reorder_buf_addr[k0 + group];
            }

            *pack_b_column_sum *= 128;
            // Move the pack_b_column_sum pointer one step to the next group.
            pack_b_column_sum += 1;
        }

        return;
    }
#endif
#endif
    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(S8S8S32OS32);

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

    dlp_reorderb_nr64_s8s8s32o32_sym_quant(&b, &b_reorder, &rntm_g, lcntx_g,
                                           group_size);
}

void
aocl_unreorder_s8s8s32os32_reference(const char      order,
                                     const char      mat_type,
                                     const int8_t*   reorder_buf_addr,
                                     int8_t*         output_buf_addr,
                                     const md_t      k,
                                     const md_t      n,
                                     const md_t      ldb,
                                     dlp_metadata_t* metadata)
{
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_UNREORDER_CHECK("s8s8s32os32_reference", order, mat_type,
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
            memcpy(output_buf_addr, reorder_buf_addr, (k * sizeof(int8_t)));
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

    dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(S8S8S32OS32);

    // Create dummy b_reorder obj.
    dlp_gemm_obj_t b_reorder;
    b_reorder.storage.aligned_buffer = (void*)reorder_buf_addr;

    // Create dummy original b obj;
    dlp_gemm_obj_t b;
    b.storage.aligned_buffer = (void*)output_buf_addr;
    b.rs                     = rs_b;
    b.cs                     = cs_b;
    b.width                  = n;
    b.length                 = k;

    dlp_unreorderb_nr64_s8s8s32os32_reference(&b, &b_reorder, &rntm_g, lcntx_g);
}
