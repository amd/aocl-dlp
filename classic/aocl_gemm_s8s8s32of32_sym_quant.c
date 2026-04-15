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

#include "aocl_dlp_gemm_check.h"
#include "classic/aocl_gemm_interface_apis.h"
#include "config/dlp_gemm_config.h"
#include "dlp_gemm_5loop_interface_apis.h"
#include "dlp_gemm_ops_bundle.h"
#include "dlp_gemm_post_ops.h"
#include "dlp_gemm_types.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "logging/dlp_gemm_logger.h"
#include "runtime/dlp_runtime.h"
#include "threading/dlp_gemm_thread_decor_openmp.h"

#include <stdlib.h>
#include <string.h>

// Utility function to transpose a 2D matrix
static void
dlp_transpose_scale_2d(
    void* dst, const void* src, md_t rows, md_t cols, size_t elem_size)
{
    const char* s = (const char*)src;
    char*       d = (char*)dst;
    for (md_t r = 0; r < rows; r++) {
        for (md_t c = 0; c < cols; c++) {
            memcpy(d + (c * rows + r) * elem_size,
                   s + (r * cols + c) * elem_size, elem_size);
        }
    }
}

void
aocl_gemm_s8s8s32of32_sym_quant(const char      order,
                                const char      transa,
                                const char      transb,
                                const md_t      m,
                                const md_t      n,
                                const md_t      k,
                                const int32_t   alpha,
                                const int8_t*   a,
                                const md_t      lda,
                                const char      mem_format_a,
                                const int8_t*   b,
                                const md_t      ldb,
                                const char      mem_format_b,
                                const int32_t   beta,
                                float*          c,
                                const md_t      ldc,
                                dlp_metadata_t* metadata)
{
    DLP_GEMM_START_LOGGER();
    DLP_GEMM_WRITE_LOGGER("s8s8s32of32_sym_quant", order, transa, transb, m, n,
                          k, ((float)alpha), lda, mem_format_a, ldb,
                          mem_format_b, ((float)beta), ldc, metadata);

    DLP_METADATA_SET_ERROR(metadata,
                           DLP_CLSC_SUCCESS); // Set default error to success.

    // Temporary buffers for transposed scale factors.
    // Declared early so they are in scope at err_hndl for cleanup.
    void* colmaj_a_scale_buf = NULL;
    void* colmaj_b_scale_buf = NULL;

    // Check if avx512_vnni ISA is supported, dlp_gemm matmul only works with
    // it.
    if (dlp_cpuid_is_avx512vnni_supported() == FALSE) {
        dlp_print_msg(" AVX512_VNNI ISA not supported by processor, "
                      "cannot perform s8s8s32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    // check for validity of params.
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_GEMM_CHECK("s8s8s32of32_sym_quant", order, transa, transb, m, n, k,
                        a, lda, mem_format_a, b, ldb, mem_format_b, c, ldc,
                        err_no);
    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        goto err_hndl;
    }

    dlp_trans_t dlp_transa;
    dlp_trans_t dlp_transb;
    /* Map BLAS chars to their corresponding DLP enumerated type value. */
    dlp_param_map_netlib_to_dlp_trans(transa, &dlp_transa);
    dlp_param_map_netlib_to_dlp_trans(transb, &dlp_transb);

    bool is_row_major    = ((order == 'r') || (order == 'R'));
    bool is_column_major = ((order == 'c') || (order == 'C'));

    // Column major support disabled for int API's till micro-kernel
    // post-ops are updated to account for column major.
    if ((is_column_major == TRUE) && (metadata != NULL)
        && (metadata->seq_length > 0)) {
        dlp_print_msg("Column major inputs not supported with Post-ops.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // The strides are set assuming a row major kernel.
    md_t rs_a = lda;
    md_t cs_a = 1;

    if (dlp_is_trans(dlp_transa)) {
        rs_a = 1;
        cs_a = lda;
    }

    md_t rs_b = ldb;
    md_t cs_b = 1;

    if (dlp_is_trans(dlp_transb)) {
        rs_b = 1;
        cs_b = ldb;
    }
    const md_t rs_c = ldc;
    const md_t cs_c = 1;

    AOCL_DLP_MEMORY_TAG mtag_a;
    AOCL_DLP_MEMORY_TAG mtag_b;

    dlp_param_map_char_to_lpmtag(mem_format_a, &mtag_a);
    dlp_param_map_char_to_lpmtag(mem_format_b, &mtag_b);

    // Reorder is not supported for A matrix
    if ((is_row_major == TRUE) && (mtag_a == REORDERED)) {
        dlp_print_msg(" Reordering of A matrix is not supported in "
                      " row major case.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }
    // Inputs swapped in column major, A becomes B from kernel point of view.
    // Reorder is not supported for column major matrices.
    else if ((is_column_major == TRUE)
             && ((mtag_b == REORDERED) || (mtag_a == REORDERED))) {
        dlp_print_msg(" Reordering of column major matrices is "
                      " not supported.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // From 5-loop function point of view
    // B matrix needs to be packed in a certain format in order to be loaded
    // and used in bf16 instrution. As such the mtag_b always needs to be either
    // packed or reordered. B matrix as it is (unpacked) cannot be used, and
    // the mtag_b is set to packed to enable runtime packing.
    if ((is_row_major == TRUE) && (mtag_b == UNPACKED)) {
        mtag_b = PACK;
    }
    // Inputs swapped in column major, A becomes B from kernel point of view.
    else if ((is_column_major == TRUE) && (mtag_a == UNPACKED)) {
        mtag_a = PACK;
    }

    // From 5-loop function point of view,
    // A matrix when in column major storage needs to be packed to row-major
    // storage as kernel expects A matrix to be in row-major format.
    if ((is_row_major == TRUE) && (dlp_is_trans(dlp_transa))) {
        mtag_a = PACK;
    }
    // Inputs swapped in column major, A becomes B from kernel point of view.
    else if ((is_column_major == TRUE) && (dlp_is_trans(dlp_transb))) {
        mtag_b = PACK;
    }

    // Add early returns for NULL group quantization parameters.
    if (metadata == NULL || metadata->post_op_grp == NULL
        || metadata->post_op_grp->a_scl == NULL
        || metadata->post_op_grp->b_scl == NULL) {
        dlp_print_msg(
            "Required parameters for symmetric quantized GEMM missing."
            " Exiting..",
            __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NULL_POINTER);
        goto err_hndl;
    }

    // convert group-level post-op struct to linked list format.
    dlp_gemm_group_post_op grp_post_op_list[AOCL_DLP_MAX_POST_OPS];
    dlp_clsc_err_t         err = dlp_gemm_translate_to_group_postops_list(
        metadata->post_op_grp, grp_post_op_list, m, n, k);

    if (err != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }

    // For column-major, the kernel swaps A and B matrices.
    // Consequentially, the scale factor arrays must also be swapped and
    // transposed for the kernel.
    // Original layouts: a_scale (m, num_groups), b_scale (num_groups, n)
    // Kernel expects:   a_scale (n, num_groups), b_scale (num_groups, m)
    // Each swap is essentially a matrix transpose.
    if (is_column_major == TRUE) {
        md_t group_size = grp_post_op_list[0].group_size;

        md_t num_groups = (k + group_size - 1) / group_size;

        // Element size depends on the scale factor storage type.
        size_t sf_elem_size = (grp_post_op_list[0].sf_stor_type == DLP_BF16)
                                  ? sizeof(int16_t)
                                  : sizeof(float);

        // Transpose original b_scale (num_groups, n) to a_scale (n, num_groups)
        // for the kernel.
        if (grp_post_op_list[0].b_scale_factor != NULL) {
            dlp_clsc_err_t ret_err;
            msz_t          mem_a_buf_size_req = num_groups * n * sf_elem_size;
            colmaj_a_scale_buf =
                dlp_malloc_page_aligned(mem_a_buf_size_req, &ret_err);

            if (colmaj_a_scale_buf == NULL)
                goto err_hndl;

            dlp_transpose_scale_2d(colmaj_a_scale_buf,
                                   grp_post_op_list[0].b_scale_factor,
                                   num_groups, n, sf_elem_size);
        }

        // Transpose original a_scale (m, num_groups) to b_scale (num_groups, m)
        // for the kernel.
        if (grp_post_op_list[0].a_scale_factor != NULL) {
            dlp_clsc_err_t ret_err;
            msz_t          mem_b_buf_size_req = num_groups * m * sf_elem_size;
            colmaj_b_scale_buf =
                dlp_malloc_page_aligned(mem_b_buf_size_req, &ret_err);

            if (colmaj_b_scale_buf == NULL)
                goto err_hndl;

            dlp_transpose_scale_2d(colmaj_b_scale_buf,
                                   grp_post_op_list[0].a_scale_factor, m,
                                   num_groups, sf_elem_size);
        }

        // Swap the pointers and lengths in grp_post_op_list.
        for (iter_t i = 0; i < metadata->post_op_grp->seq_length; ++i) {
            // Swap scale factor pointers to transposed buffers.
            grp_post_op_list[i].a_scale_factor = colmaj_a_scale_buf;
            grp_post_op_list[i].b_scale_factor = colmaj_b_scale_buf;

            // Swap scale factor lengths.
            md_t tmp_sf_len = grp_post_op_list[i].a_scale_factor_len;
            grp_post_op_list[i].a_scale_factor_len =
                grp_post_op_list[i].b_scale_factor_len;
            grp_post_op_list[i].b_scale_factor_len = tmp_sf_len;

            // Swap zero-point pointers and lengths.
            void* tmp_zp                 = grp_post_op_list[i].a_zp;
            md_t  tmp_zp_len             = grp_post_op_list[i].a_zp_len;
            grp_post_op_list[i].a_zp     = grp_post_op_list[i].b_zp;
            grp_post_op_list[i].a_zp_len = grp_post_op_list[i].b_zp_len;
            grp_post_op_list[i].b_zp     = tmp_zp;
            grp_post_op_list[i].b_zp_len = tmp_zp_len;
        }
    }

    // Convert post op struct to post op linked list format.
    dlp_gemm_post_op post_op_list[AOCL_DLP_MAX_POST_OPS];
    err = dlp_gemm_translate_to_post_ops_list(metadata, post_op_list, (void*)c,
                                              (void*)(&order), m, n);

    if (err != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(S8S8S32OS32);

    dlp_gemm_ops_bundle_t ops =
        DLP_GEMM_OPS_BUNDLE_INIT_GRP(grp_post_op_list, post_op_list);

#ifdef DLP_ENABLE_OPENMP
    // Swapping inputs to induce row major computation for column major inputs.
    if (is_column_major == TRUE) {
        dlp_gemm_s8s8s32o32_sym_quant_openmp_thread_decorator(
            n, m, k, b, rs_b, cs_b, mtag_b, a, rs_a, cs_a, mtag_a, (float*)c,
            rs_c, cs_c, alpha, beta, &rntm_g, lcntx_g, &ops, DLP_F32);
    } else {
        dlp_gemm_s8s8s32o32_sym_quant_openmp_thread_decorator(
            m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, (float*)c,
            rs_c, cs_c, alpha, beta, &rntm_g, lcntx_g, &ops, DLP_F32);
    }
#else
    // Swapping inputs to induce row major computation for column major inputs.
    if (is_column_major == TRUE) {
        dlp_gemm_s8s8s32o32_sym_quant_thread_decorator(
            n, m, k, b, rs_b, cs_b, mtag_b, a, rs_a, cs_a, mtag_a, (float*)c,
            rs_c, cs_c, alpha, beta, &rntm_g, lcntx_g, &ops, DLP_F32);
    } else {
        dlp_gemm_s8s8s32o32_sym_quant_thread_decorator(
            m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, (float*)c,
            rs_c, cs_c, alpha, beta, &rntm_g, lcntx_g, &ops, DLP_F32);
    }
#endif

err_hndl:;
    // Free temporarily allocated buffers used for transpose in case of
    // column-major inputs.
    if (colmaj_a_scale_buf != NULL)
        dlp_free_page_aligned(colmaj_a_scale_buf);
    if (colmaj_b_scale_buf != NULL)
        dlp_free_page_aligned(colmaj_b_scale_buf);

    DLP_GEMM_STOP_LOGGER();
}
