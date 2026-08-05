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

// Extract the symmetric-quant group size from the unified quant metadata.
// Mirrors the s8s8 sym-quant reorder path: the group size is carried in
// metadata->b_quant_op (and, when present, must agree with a_quant_op).
// A group_size of 0 means "one group spanning the full K dimension".
static dlp_clsc_err_t
dlp_get_sym_quant_group_size_from_metadata(const dlp_metadata_t* metadata,
                                           const md_t            k,
                                           md_t*                 group_size)
{
    if ((metadata == NULL) || (group_size == NULL)) {
        return DLP_CLSC_NULL_POINTER;
    }

    if (((metadata->a_quant_op != NULL)
         && (metadata->a_quant_op->quant_op_kind != DLP_QUANT_OP_QUANTIZE))
        || ((metadata->b_quant_op != NULL)
            && (metadata->b_quant_op->quant_op_kind
                != DLP_QUANT_OP_QUANTIZE))) {
        return DLP_CLSC_NOT_SUPPORTED;
    }

    md_t a_group_size =
        (metadata->a_quant_op != NULL) ? metadata->a_quant_op->group_size : 0;
    md_t b_group_size =
        (metadata->b_quant_op != NULL) ? metadata->b_quant_op->group_size : 0;

    if (a_group_size == 0) {
        a_group_size = k;
    }
    if (b_group_size == 0) {
        b_group_size = k;
    }

    if ((metadata->a_quant_op != NULL) && (metadata->b_quant_op != NULL)
        && (a_group_size != b_group_size)) {
        return DLP_CLSC_INVALID_GROUP_DIMENSION;
    }

    md_t gs = (metadata->b_quant_op != NULL) ? b_group_size : a_group_size;

    if ((gs <= 0) || (gs > k)) {
        return DLP_CLSC_INVALID_GROUP_DIMENSION;
    }
    // s8s4 requires the group size to be a multiple of 4, except for the
    // full-K single-group case where k itself need not be aligned.
    if ((gs != k) && ((gs & 3) != 0)) {
        return DLP_CLSC_INVALID_GROUP_DIMENSION;
    }

    *group_size = gs;
    return DLP_CLSC_SUCCESS;
}

msz_t
aocl_get_reorder_buf_size_s8s4s32os32(const char      order,
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
                      "cannot perform s8s4s32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return 0; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_REORDER_BUF_SIZE_CHECK("s8s4s32os32", order, trans, mat_type, k, n,
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

    md_t group_size = 0;
    err_no =
        dlp_get_sym_quant_group_size_from_metadata(metadata, k, &group_size);
    if (err_no != DLP_CLSC_SUCCESS) {
        dlp_print_msg(
            " Invalid group size for s8s4s32os32; expected a multiple of 4 "
            "carried in metadata->b_quant_op.",
            __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return 0; // Error.
    }

    md_t num_groups = (k + group_size - 1) / group_size;

#ifdef DLP_KERNELS_ZEN4
    // GEMV n==1 fast path (iff group_size divides both k and KC): the reorder
    // emits the tight *s8* single column (no s4 compression, no 16-wide
    // padding) followed by the per-group column sums, matching the layout the
    // s8s8 sym-quant dlp_gemv_n_one kernel consumes. This mirrors the s8s8
    // reorder so the s8s4 GEMV path can reuse the s8s8 GEMV kernels directly.
    dlp_gemm_cntx_t lcntx_g = *(dlp_gemm_get_global_cntx_obj(S8S8S32OS32));
    md_t            KC      = lcntx_g.blksz.KC;

    if ((n == 1) && ((k % group_size) == 0) && ((KC % group_size) == 0)) {
        // k s8 weights (k is a multiple of 4, so the int32 colsums that follow
        // stay naturally aligned) + one int32 column sum per group.
        return (msz_t)(sizeof(int8_t) * k)
               + (msz_t)num_groups * sizeof(int32_t);
    }
#endif

    // The compact s4 reordered layout uses the general (multiple-of-16) padded
    // footprint reused from the s8 reorder worker.
    md_t n_reorder = dlp_make_multiple_of_n(n, 16);
    md_t k_reorder = dlp_make_multiple_of_n(k, 4);

    // Column sums (uncompressed int32) stored after the compact s4 weights.
    size_t extra_mem_req = (size_t)num_groups * n_reorder * sizeof(int32_t);

    // s4 weights occupy half the bytes of the equivalent s8 packed weights.
    msz_t size_req =
        (msz_t)((sizeof(int8_t) * k_reorder * n_reorder) / 2) + extra_mem_req;

    return size_req;
}

void
aocl_reorder_s8s4s32os32(const char      order,
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
                      "cannot perform s8s4s32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        return; // Error.
    }

    dlp_clsc_err_t err_no     = DLP_CLSC_SUCCESS;
    md_t           group_size = 0;
    err_no =
        dlp_get_sym_quant_group_size_from_metadata(metadata, k, &group_size);
    if (err_no != DLP_CLSC_SUCCESS) {
        dlp_print_msg(
            " Invalid group size for s8s4s32os32; expected a multiple of 4 "
            "carried in metadata->b_quant_op.",
            __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    AOCL_DLP_REORDER_CHECK("s8s4s32os32", order, trans, mat_type,
                           input_buf_addr, reorder_buf_addr, k, n, ldb, err_no);
    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return; // Error.
    }

    dlp_trans_t dlp_trans;
    /* Map BLAS chars to their corresponding DLP enumerated type value. */
    dlp_param_map_netlib_to_dlp_trans(trans, &dlp_trans);

    // Strides are measured in s4 element (nibble) units.
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

    dlp_gemm_cntx_t lcntx_g = *(dlp_gemm_get_global_cntx_obj(S8S8S32OS32));

#ifdef DLP_KERNELS_ZEN4
    // GEMV n==1 fast path (iff group_size divides both k and KC): emit the
    // tight s8 single column + per-group column sums (each pre-scaled by 128),
    // exactly like the s8s8 sym-quant n==1 reorder. The s4 nibbles are
    // sign-extended to s8 here so the GEMV path can consume this buffer without
    // a widen step.
    md_t KC = lcntx_g.blksz.KC;
    if ((n == 1) && ((k % group_size) == 0) && ((KC % group_size) == 0)) {
        int32_t* pack_b_column_sum =
            (int32_t*)(reorder_buf_addr + (k * sizeof(int8_t)));
        const uint8_t* src = (const uint8_t*)input_buf_addr;

        for (iter_t k0 = 0; k0 < k; k0 += group_size) {
            md_t gsz           = dlp_min(group_size, k - k0);
            *pack_b_column_sum = 0;
            for (iter_t g = 0; g < gsz; g++) {
                // s4 element (k0+g) of the single column lives at nibble index
                // (k0+g)*rs_b (rs_b in nibble units); low nibble first.
                md_t    nib_idx = (k0 + g) * rs_b;
                uint8_t nib = (src[nib_idx >> 1] >> ((nib_idx & 1) * 4)) & 0x0F;
                int8_t  v   = (nib & 0x08) ? (int8_t)(nib | 0xF0) : (int8_t)nib;

                reorder_buf_addr[k0 + g] = v;
                *pack_b_column_sum += v;
            }
            *pack_b_column_sum *= 128;
            pack_b_column_sum += 1;
        }
        return;
    }
#endif

    // Create dummy b_reorder obj.
    dlp_gemm_obj_t b_reorder;
    b_reorder.storage.aligned_buffer = reorder_buf_addr;

    // Create dummy original b obj (nibble-packed s4).
    dlp_gemm_obj_t b;
    b.storage.aligned_buffer = (void*)input_buf_addr;
    b.rs                     = rs_b;
    b.cs                     = cs_b;
    b.width                  = n;
    b.length                 = k;

    err_no = dlp_reorderb_nr64_s8s4s32o32(&b, &b_reorder, &rntm_g, &lcntx_g,
                                          group_size);
    if (err_no != DLP_CLSC_SUCCESS) {
        dlp_print_msg(" s8s4 reorder failed (scratch allocation).", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, err_no);
        return; // Error: b_reorder was not marked REORDERED.
    }
}
