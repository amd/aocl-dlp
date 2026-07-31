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
#include "kernels/s8s8s32/dlp_gemm_packb_s8s4.h"
#include "logging/dlp_gemm_logger.h"
#include "runtime/dlp_runtime.h"
#include "sys_utils/dlp_gemm_sys.h"
#include "threading/dlp_gemm_thread_decor_openmp.h"

#include <stdlib.h>
#include <string.h>

void
aocl_gemm_s8s4s32obf16(const char      order,
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
                       bfloat16*       c,
                       const md_t      ldc,
                       dlp_metadata_t* metadata)
{
    DLP_GEMM_START_LOGGER();
    DLP_GEMM_WRITE_LOGGER("s8s4s32obf16", order, transa, transb, m, n, k,
                          ((float)alpha), lda, mem_format_a, ldb, mem_format_b,
                          ((float)beta), ldc, metadata);

    DLP_METADATA_SET_ERROR(metadata,
                           DLP_CLSC_SUCCESS); // Set default error to success.

    // Check if avx512_vnni ISA is supported, dlp_gemm matmul only works with
    // it.
    if (dlp_cpuid_is_avx512vnni_supported() == FALSE) {
        dlp_print_msg(" AVX512_VNNI ISA not supported by processor, "
                      "cannot perform s8s4s32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    // check for validity of params.
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_GEMM_CHECK("s8s4s32obf16", order, transa, transb, m, n, k, a, lda,
                        mem_format_a, b, ldb, mem_format_b, c, ldc, err_no);
    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        goto err_hndl;
    }

    dlp_trans_t dlp_transa;
    dlp_trans_t dlp_transb;
    /* Map BLAS chars to their corresponding DLP enumerated type value. */
    dlp_param_map_netlib_to_dlp_trans(transa, &dlp_transa);
    dlp_param_map_netlib_to_dlp_trans(transb, &dlp_transb);

    bool is_row_major = ((order == 'r') || (order == 'R'));

    // s8s4 sym-quant supports row-major only.
    if (is_row_major == FALSE) {
        dlp_print_msg(" Column major inputs are not supported for "
                      "s8s4s32 sym-quant GEMM.",
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

    // Reorder is not supported for A matrix.
    if (mtag_a == REORDERED) {
        dlp_print_msg(" Reordering of A matrix is not supported.", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // B is either pre-reordered ('R', the compact-memory fast path) or a raw
    // nibble-packed s4 matrix that is packed on the fly. A raw unpacked B ('N')
    // is promoted to runtime PACK; an explicit 'P' already requests runtime
    // packing. This mirrors the s8s8 sym-quant path. Runtime packing works for
    // both transb='N' and transb='T' since rs_b/cs_b (set above) drive the
    // s4->s8 widen.
    if (mtag_b == UNPACKED) {
        mtag_b = PACK;
    } else if (mtag_b != REORDERED && mtag_b != PACK) {
        dlp_print_msg(" Matrix B must be reordered ('R'), unpacked ('N'), or "
                      "packed ('P') for s8s4s32 sym-quant GEMM.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // A in column-major/transposed storage needs to be packed to row-major.
    if (dlp_is_trans(dlp_transa)) {
        mtag_a = PACK;
    }

    // Add early returns for NULL group quantization parameters.
    if (metadata == NULL || metadata->a_quant_op == NULL
        || metadata->b_quant_op == NULL
        || metadata->a_quant_op->dequant_scale_factors == NULL
        || metadata->b_quant_op->dequant_scale_factors == NULL) {
        dlp_print_msg(
            "Required parameters for symmetric quantized GEMM missing."
            " Exiting..",
            __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NULL_POINTER);
        goto err_hndl;
    }

    // Validate scale-factor granularity up front. A (M x K) varies over rows
    // and is grouped along K, so PER_TENSOR / PER_TOKEN / PER_GROUP are valid
    // but PER_CHANNEL (a column/N concept) is not. B (K x N) varies over
    // columns, so PER_TENSOR / PER_CHANNEL / PER_GROUP are valid but PER_TOKEN
    // (a row/M concept) is not. Reject cross-assigned dims instead of silently
    // coercing them to PER_GROUP. (s8s4 is row-major only, so the column-major
    // scale swap/transpose used by s8s8 is intentionally not needed here.)
    {
        DLP_PARAM_DIM_TYPE a_dim =
            metadata->a_quant_op->dequant_scale_factors->outer_dim;
        DLP_PARAM_DIM_TYPE b_dim =
            metadata->b_quant_op->dequant_scale_factors->outer_dim;
        if ((a_dim != DLP_PARAM_DIM_PER_TENSOR)
            && (a_dim != DLP_PARAM_DIM_PER_TOKEN)
            && (a_dim != DLP_PARAM_DIM_PER_GROUP)) {
            dlp_print_msg(" A scale factor outer_dim must be PER_TENSOR, "
                          "PER_TOKEN or PER_GROUP for sym_quant. Exiting..",
                          __FILE__, __LINE__);
            DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
            goto err_hndl;
        }
        if ((b_dim != DLP_PARAM_DIM_PER_TENSOR)
            && (b_dim != DLP_PARAM_DIM_PER_CHANNEL)
            && (b_dim != DLP_PARAM_DIM_PER_GROUP)) {
            dlp_print_msg(" B scale factor outer_dim must be PER_TENSOR, "
                          "PER_CHANNEL or PER_GROUP for sym_quant. Exiting..",
                          __FILE__, __LINE__);
            DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
            goto err_hndl;
        }
    }

    // convert group-level post-op struct to linked list format.
    dlp_gemm_group_post_op grp_post_op_list[AOCL_DLP_MAX_POST_OPS];
    dlp_clsc_err_t         err = dlp_gemm_translate_to_group_postops_list(
        metadata->a_quant_op, metadata->b_quant_op, grp_post_op_list, m, n, k);

    if (err != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }

    // Convert post op struct to post op linked list format.
    dlp_gemm_post_op post_op_list[AOCL_DLP_MAX_POST_OPS];
    err = dlp_gemm_translate_to_post_ops_list(metadata, post_op_list, (void*)c,
                                              (void*)(&order), m, n);

    if (err != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }

    /* Capability gate: sym_quant has no JIT alternative; post-ops with
     * op_code > DLP_CLASSIC_MAX_POST_OP_CODE have no entry in the
     * classic post_ops_labels[] dispatch table. Reject cleanly. */
    if (dlp_gemm_post_op_list_has_jit_only_op(post_op_list) == true) {
        dlp_print_msg(" Requested post-op is not supported in the "
                      "classic kernel.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    dlp_gemm_cntx_t lcntx_g = *(dlp_gemm_get_global_cntx_obj(S8S8S32OS32));
    err = dlp_gemm_upd_cntx_with_metadata(S8S8S32OS32, &lcntx_g, metadata);
    if (err != DLP_CLSC_SUCCESS) {
        dlp_print_msg(" Failed to update context with metadata.", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }

    dlp_gemm_ops_bundle_t ops =
        DLP_GEMM_OPS_BUNDLE_INIT_GRP(grp_post_op_list, post_op_list);

#ifdef DLP_KERNELS_ZEN4
    // GEMV fast path: for m==1 or n==1 with a reordered B and a group size that
    // divides both k and KC, reuse the specialized s8s8 sym-quant GEMV kernels
    // (dlp_gemv_{m,n}_one) instead of the generic 6x64 GEMM micro-kernel. The
    // s8s4 reorder mirrors the s8s8 reorder, so:
    //   * n==1 (divisible): the reorder already emitted the tight s8 column +
    //     per-group column sums (identical to s8s8) -- consumed directly.
    //   * m==1: the compact s4 general layout is widened once to the equivalent
    //     s8 reorder buffer and handed to the s8s8 sym-quant path.
    md_t gs = grp_post_op_list[0].group_size;
    md_t KC = lcntx_g.blksz.KC;
    if ((mtag_b == REORDERED) && (gs > 0) && ((k % gs) == 0) && ((KC % gs) == 0)
        && ((m == 1) || (n == 1))) {
        // n==1: reorder already emitted tight s8 columns + per-group column
        // sums, consumed directly. m==1 (n!=1): must widen the compact s4 to s8
        // first; leave b_gemv NULL until that succeeds so an allocation failure
        // falls through to the generic s8s4 path below instead of feeding the
        // s8s8 kernel a nibble-packed (half-size) buffer.
        const int8_t* b_gemv  = (n == 1) ? b : NULL;
        int8_t*       b_widen = NULL;

        if (n != 1) {
            md_t k_updated  = dlp_make_multiple_of_n(k, 4);
            md_t n_updated  = dlp_make_multiple_of_n(n, 16);
            md_t num_groups = (k + gs - 1) / gs;

            msz_t w_s8_bytes = (msz_t)k_updated * n_updated;
            msz_t colsum_bytes =
                (msz_t)num_groups * n_updated * sizeof(int32_t);

            dlp_clsc_err_t ret_err;
            b_widen =
                dlp_malloc_page_aligned(w_s8_bytes + colsum_bytes, &ret_err);
            if (b_widen != NULL) {
                // Widen the compact s4 weights back to the s8 packed layout and
                // copy the (uncompressed) per-group column sums that trail
                // them.
                dlp_cvt_s4_to_s8_linear_avx512(b_widen, (const uint8_t*)b,
                                               (md_t)w_s8_bytes);
                memcpy(b_widen + w_s8_bytes,
                       (const int8_t*)b + (w_s8_bytes / 2), colsum_bytes);
                b_gemv = b_widen;
            }
        }

        if (b_gemv != NULL) {
#ifdef DLP_ENABLE_OPENMP
            dlp_gemm_s8s8s32o32_sym_quant_openmp_thread_decorator(
                m, n, k, a, rs_a, cs_a, mtag_a, b_gemv, rs_b, cs_b, mtag_b,
                (float*)c, rs_c, cs_c, alpha, beta, &rntm_g, &lcntx_g, &ops,
                DLP_BF16);
#else
            dlp_gemm_s8s8s32o32_sym_quant_thread_decorator(
                m, n, k, a, rs_a, cs_a, mtag_a, b_gemv, rs_b, cs_b, mtag_b,
                (float*)c, rs_c, cs_c, alpha, beta, &rntm_g, &lcntx_g, &ops,
                DLP_BF16);
#endif
            if (b_widen != NULL) {
                dlp_free_page_aligned(b_widen);
            }
            goto err_hndl;
        }
    }
#endif

#ifdef DLP_ENABLE_OPENMP
    dlp_gemm_s8s4s32o32_openmp_thread_decorator(
        m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, (float*)c, rs_c,
        cs_c, alpha, beta, &rntm_g, &lcntx_g, &ops, DLP_BF16);
#else
    dlp_gemm_s8s4s32o32_thread_decorator(
        m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, (float*)c, rs_c,
        cs_c, alpha, beta, &rntm_g, &lcntx_g, &ops, DLP_BF16);
#endif

err_hndl:;
    DLP_GEMM_STOP_LOGGER();
}
