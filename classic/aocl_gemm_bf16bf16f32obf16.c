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

static inline bool
is_tiny_input_bf16obf16(md_t m, md_t n, md_t k, dlp_gemm_cntx_t* lcntx)
{
    const md_t NC = lcntx->blksz.NC;
    const md_t MC = lcntx->blksz.MC;
    const md_t KC = lcntx->blksz.KC;
    const md_t MR = lcntx->blksz.MR;
    const md_t NR = lcntx->blksz.NR;

    md_t       mnk           = m * n * k;
    const md_t mnk_magic_num = 36 * 128 * 256;
    const md_t m_thresh      = 6 * MR;
    const md_t n_thresh      = 6 * NR;
    const md_t k_thresh      = 1024;

    // Need to explicitly check for MC, NC boundaries for safety.
    if ((m <= MC) && (n < NC) && (k < KC)
        && ((m <= m_thresh) && (n <= n_thresh) && (k <= k_thresh)
            && (mnk < mnk_magic_num))) {
        return TRUE;
    }

    return FALSE;
}

void
aocl_gemm_bf16bf16f32obf16(const char      order,
                           const char      transa,
                           const char      transb,
                           const md_t      m,
                           const md_t      n,
                           const md_t      k,
                           const float     alpha,
                           const bfloat16* a,
                           const md_t      lda,
                           const char      mem_format_a,
                           const bfloat16* b,
                           const md_t      ldb,
                           const char      mem_format_b,
                           const float     beta,
                           bfloat16*       c,
                           const md_t      ldc,
                           dlp_metadata_t* metadata)
{
    DLP_GEMM_START_LOGGER();
    DLP_GEMM_WRITE_LOGGER("bf16bf16f32obf16", order, transa, transb, m, n, k,
                          ((float)alpha), lda, mem_format_a, ldb, mem_format_b,
                          ((float)beta), ldc, metadata);

    DLP_METADATA_SET_ERROR(metadata,
                           DLP_CLSC_SUCCESS); // Set default error to success.

    // Check if avx512_vnni ISA is supported, dlp_gemm matmul only works with
    // it.
    if (dlp_cpuid_is_avx2fma3_supported() == FALSE) {
        dlp_print_msg(" AVX2 ISA not supported by processor, "
                      "cannot perform bf16bf16f32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    // check for validity of params.
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_GEMM_CHECK("bf16bf16f32obf16", order, transa, transb, m, n, k, a,
                        lda, mem_format_a, b, ldb, mem_format_b, c, ldc,
                        err_no);
    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        goto err_hndl;
    }

#ifdef DLP_GEMM_BF16_JIT
    if (dlp_gemm_get_jit_kernels_generated() == FALSE) {
        dlp_print_msg(" Could not generate bf16bf16f32obf16 "
                      " kernels using JIT.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }
#endif

    dlp_trans_t dlp_transa;
    dlp_trans_t dlp_transb;
    /* Map BLAS chars to their corresponding DLP enumerated type value. */
    dlp_param_map_netlib_to_dlp_trans(transa, &dlp_transa);
    dlp_param_map_netlib_to_dlp_trans(transb, &dlp_transb);

    bool is_row_major    = ((order == 'r') || (order == 'R'));
    bool is_column_major = ((order == 'c') || (order == 'C'));

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
        dlp_print_msg(
            " Reordering of A matrix is not supported in row major case.",
            __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }
    // Inputs swapped in column major, A becomes B from kernel point of view.
    // Reorder is not supported for column major matrices.
    else if ((is_column_major == TRUE)
             && ((mtag_b == REORDERED) || (mtag_a == REORDERED))) {
        dlp_print_msg(" Reordering of column major matrices is not supported.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // From 5-loop function point of view,
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

    // Convert post op struct to post op linked list format.
    // +1 for GLU terminal ops.
    dlp_gemm_post_op post_op_list[AOCL_DLP_MAX_POST_OPS + 1];
    dlp_clsc_err_t   err = dlp_gemm_translate_to_post_ops_list(
        metadata, post_op_list, (void*)c, (void*)(&order), m, n);
    if (err != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }

    md_t term_glu_offset =
        (metadata != NULL)
            ? ((metadata->seq_length > 0) ? metadata->seq_length : 0)
            : 0;
    // GLU terminal ops.
    err =
        dlp_gemm_translate_glu_term_op(metadata, post_op_list + term_glu_offset,
                                       post_op_list, (void*)(&order), m, n);
    if (err != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }

    // Create copy variables to handle column major case
    md_t                m_use          = m;
    md_t                n_use          = n;
    md_t                rs_a_use       = rs_a;
    md_t                cs_a_use       = cs_a;
    md_t                rs_b_use       = rs_b;
    md_t                cs_b_use       = cs_b;
    md_t                rs_c_use       = rs_c;
    md_t                cs_c_use       = cs_c;
    AOCL_DLP_MEMORY_TAG mtag_a_use     = mtag_a;
    AOCL_DLP_MEMORY_TAG mtag_b_use     = mtag_b;
    dlp_trans_t         dlp_transa_use = dlp_transa;
    dlp_trans_t         dlp_transb_use = dlp_transb;
    const bfloat16*     a_use          = a;
    const bfloat16*     b_use          = b;

    // Swapping inputs to induce row major computation for column major inputs.
    if (is_column_major == TRUE) {
        m_use          = n;
        n_use          = m;
        rs_a_use       = rs_b;
        cs_a_use       = cs_b;
        rs_b_use       = rs_a;
        cs_b_use       = cs_a;
        mtag_a_use     = mtag_b;
        mtag_b_use     = mtag_a;
        dlp_transa_use = dlp_transb;
        dlp_transb_use = dlp_transa;
        a_use          = b;
        b_use          = a;
    }

    // GEMV-specific optimization for avoiding unnecessary packing.
    // This optimization is enabled only when post-ops are disabled and
    // k >= 256, below which the packing cost is too small to justify the
    // overhead of the operation transpose.
    // We perform an "operation transpose" to use a more efficient kernel path.

    // For GEMV_M1: If B is transposed and not reordered, swap to use
    // GEMV_N1 to avoid packing B matrix. GEMV_N1 kernels support both
    // unit/non-unit strided loads/stores for C vector.
    if (((m_use == 1) && (k >= 256) && (dlp_is_trans(dlp_transb_use))
         && (mtag_b_use != REORDERED))
        && (post_op_list[0].op_code == POST_OPS_DISABLE)) {

        // Store temporary values before potential operation transpose
        md_t            m_tmp    = m_use;
        md_t            n_tmp    = n_use;
        md_t            rs_a_tmp = rs_a_use;
        md_t            cs_a_tmp = cs_a_use;
        md_t            rs_b_tmp = rs_b_use;
        md_t            cs_b_tmp = cs_b_use;
        md_t            rs_c_tmp = rs_c_use;
        md_t            cs_c_tmp = cs_c_use;
        const bfloat16* a_tmp    = a_use;
        const bfloat16* b_tmp    = b_use;

        m_use      = n_tmp;
        n_use      = m_tmp;
        a_use      = b_tmp;
        rs_a_use   = cs_b_tmp;
        cs_a_use   = rs_b_tmp;
        b_use      = a_tmp;
        rs_b_use   = cs_a_tmp;
        cs_b_use   = rs_a_tmp;
        rs_c_use   = cs_c_tmp;
        cs_c_use   = rs_c_tmp;
        mtag_a_use = UNPACKED;
        // Setting mtag_b_use is purely a safety measure. The input
        // vector(after our operation transpose) will anyways be
        // contiguous, and the framework will not pack it further.
        mtag_b_use = PACK;
    }

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    // Create local copy, since each thread in a multi-instance setup
    // modified the context object.
    dlp_gemm_cntx_t lcntx_l = *(dlp_gemm_get_global_cntx_obj(BF16BF16F32OF32));

    // Hand the threading state to the DE: MR bounds how much work an IC way
    // holds and the thread count bounds how much work there is to size MR
    // against, so neither can be reasoned about without the other.
    //
    // The ways go in alongside the count, carrying the -1 sentinels unless the
    // application pinned them through DLP_IC_NT / DLP_JC_NT -- a pinned
    // partition is a constraint, not a starting point. The DE writes the split
    // it resolved back here.
    lcntx_l.thread_info.num_threads = rntm_g.num_threads;
    lcntx_l.thread_info.ic_ways     = rntm_g.ic_ways;
    lcntx_l.thread_info.jc_ways     = rntm_g.jc_ways;

    // The BF16 5 loop framework internally queries F32 cntx in case BF16
    // API is called on a non BF16 ISA machine. Any update to BF16 cntx
    // and block params here via metadata therefore wont be reflected in
    // the 5 loop and hence disabling it on non BF16 machines.
    if (dlp_cpuid_is_avx512bf16_supported() == TRUE) {
        err = dlp_gemm_upd_cntx_with_metadata(BF16BF16F32OF32, &lcntx_l,
                                              metadata);
        if (err != DLP_CLSC_SUCCESS) {
            dlp_print_msg(" Failed to update context with metadata.", __FILE__,
                          __LINE__);
            DLP_METADATA_SET_ERROR(metadata, err);
            goto err_hndl;
        }
    }

    // Initialize DLP Plus kernel path.
    // For machines without AVX512BF16, BF16 operations use F32 JIT kernels.
    // Use F32 block sizes (MR=6, NR=16 for AVX2) instead of BF16 block sizes
    // (MR=6, NR=64 for AVX512BF16) to ensure correct kernel generation.
    // However the bf16 cntx blksz are updated with the f32 cntx blksz, and
    // needs to be reverted post the JIT kernel init.
    md_t og_mr_hint = lcntx_l.blksz.MR;
    md_t og_nr_hint = lcntx_l.blksz.NR;
    md_t og_kc_hint = lcntx_l.blksz.KC;

    // Create copy of mtag variables to handle jit kernel generation for bf16 on
    // architectures without bf16 support.
    AOCL_DLP_MEMORY_TAG jit_mtag_a = mtag_a_use;
    AOCL_DLP_MEMORY_TAG jit_mtag_b = mtag_b_use;

    // Get the configured architecture from the context.
    // This would be set based on the AOCL_DLP_ENABLE_INSTRUCTIONS environment
    // variable and/or the underlying architecture.
    dlp_arch_t arch_id = dlp_get_arch();

    if ((dlp_cpuid_is_avx512bf16_supported() == FALSE)
        || (arch_id == DLP_ARCH_ZEN3) || (arch_id == DLP_ARCH_ZEN2)
        || (arch_id == DLP_ARCH_ZEN)) {
        // On these architectures BF16 reroutes to the F32 GEMM kernels, which
        // do not implement the shape-changing GLU compacted half-width store.
        // Return NOT_SUPPORTED early (a clean skip) instead of letting JIT
        // generation fail later with INVALID_JIT_KERNEL (a hard error).
        if (dlp_gemm_post_op_list_has_shape_changing_glu(post_op_list)) {
            DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
            goto err_hndl;
        }

        // No native BF16 support - will use F32 kernels
        // Get F32 context for proper block sizes
        dlp_gemm_cntx_t* lcntx_f32 =
            dlp_gemm_get_global_cntx_obj(F32F32F32OF32);
        lcntx_l.blksz.MR = lcntx_f32->blksz.MR;
        lcntx_l.blksz.NR = lcntx_f32->blksz.NR;
        lcntx_l.blksz.KC = lcntx_f32->blksz.KC;

        // For m=1 case B matrix is unpacked inside the framework before
        // calling f32 kernel, same should be provided for generating JIT
        // kernels
        if (m_use == 1) {
            jit_mtag_b = UNPACKED;
        }
    }

    // Reject a GEMM that cannot be served against the panel its reordered B
    // was packed into, before a kernel is generated for it. These are the
    // values the decision engine is about to fold on. A differing m is not
    // such a case and is served -- see dlp_gemm_validate_hints_with_call.
    err = dlp_gemm_validate_hints_with_call(&lcntx_l, jit_mtag_b,
                                            rntm_g.num_threads, rntm_g.ic_ways,
                                            rntm_g.jc_ways);
    if (err != DLP_CLSC_SUCCESS) {
        char msg[256];
        if (err == DLP_CLSC_INVALID_GEMM_HINTS) {
            snprintf(msg, sizeof(msg),
                     "GEMM hints must be zero (unset) or positive, got "
                     "m_hint: %ld nt_hint: %ld\n",
                     (lcntx_l.gemm_kernel_hints).m_hint,
                     (lcntx_l.gemm_kernel_hints).nt_hint);
        } else {
            // The pool as the runtime resolved it, which under pinned ways is
            // their product rather than rntm_g.num_threads -- a pinned run
            // leaves the count unset, and printing that -1 would report the
            // sentinel as though it were the disagreement.
            snprintf(msg, sizeof(msg),
                     "GEMM over a reordered B will run on a different thread "
                     "count than the buffer was reordered under, nt_hint: %ld "
                     "vs threads: %ld\n",
                     (lcntx_l.gemm_kernel_hints).nt_hint,
                     dlp_gemm_effective_thread_count(
                         rntm_g.num_threads, rntm_g.ic_ways, rntm_g.jc_ways));
        }
        dlp_print_msg(msg, __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }

    // Initialize DLP Plus kernel path.
    lcntx_l.dlp_kernel_hndl.kernel_base = NULL;

    dlp_init_and_get_kernel_hndl(
        DLP_KERNEL_BF16BF16F32OBF16, order, jit_mtag_a, jit_mtag_b, m_use,
        n_use, k, rs_a_use, cs_a_use, rs_b_use, cs_b_use, rs_c_use, cs_c_use,
        (void*)&alpha, (void*)&beta, post_op_list, &lcntx_l, DLP_BF16);

    // Invalid handle means that the jit kernel generation has failed. Do not
    // attempt to execute the kernel, and return an error instead.
    if (lcntx_l.dlp_kernel_hndl.kernel_base == NULL) {
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_INVALID_JIT_KERNEL);
        goto err_hndl;
    }

    // Take the split the DE resolved. The decorator finds these set and hands
    // them back rather than deriving its own, through the same arm that
    // honours DLP_IC_NT / DLP_JC_NT. Both routes run the same heuristic over
    // the same extents and tile, so this saves the second derivation rather
    // than changing the answer -- what it changes is that the split each
    // candidate was costed against is the one that executes.
    //
    // The DE leaves these as it found them wherever it resolved no split: an
    // architecture the model does not serve, an ineligible call, or a pinned
    // runtime over a reordered B, where the model cannot see the pin and must
    // not overrule it. In each of those the assignment is an identity and the
    // decorator partitions as before.
    //
    // The ways only. The decorator recovers the count as ic_ways * jc_ways, so
    // leaving rntm_g.num_threads alone keeps the dlp_is_single_thread checks
    // below on the count the runtime resolved; those select the execution path,
    // not the partition.
    rntm_g.ic_ways = lcntx_l.thread_info.ic_ways;
    rntm_g.jc_ways = lcntx_l.thread_info.jc_ways;

    // JIT pack B (BF16): the pack-B kernel (full NR + fringe ladder) for the
    // AVX512-BF16 path. Generated while the (possibly F32-swapped) block sizes
    // are still in effect, mirroring the GEMM kernel init above. cs_b_use == 1
    // selects the row-major packer; cs_b_use != 1 (with rs_b_use == 1, i.e.
    // transB) selects the column-major 16x16 transpose packer.
    lcntx_l.dlp_pack_kernel_hndl.pack_b_hndl.kernel_base = NULL;
    dlp_init_and_get_packb_kernel_hndl(DLP_KERNEL_BF16BF16F32OBF16, n_use, k,
                                       rs_b_use, cs_b_use, &lcntx_l);

    // A pack-B kernel is only generated when the configured arch has
    // AVX512-BF16. Without it, BF16 reroutes to the F32 kernels, which
    // convert/unreorder B instead, so a NULL handle is expected there and
    // execution continues. With it, a NULL handle means JIT generation failed,
    // which is an error. n=1 gemv copies B rather than packing it into an NR
    // panel (NR == 1), and therefore needs no pack-B kernel.
    if ((dlp_cpuid_is_avx512bf16_configured() == TRUE) && (lcntx_l.blksz.NR > 1)
        && (lcntx_l.dlp_pack_kernel_hndl.pack_b_hndl.kernel_base == NULL)) {
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_INVALID_JIT_KERNEL);
        goto err_hndl;
    }

    // Both inits are done, so the tile the packed buffers are written against
    // is final.
    dlp_upd_pack_strides(DLP_KERNEL_BF16BF16F32OBF16, &lcntx_l);

    err = dlp_gemm_validate_metadata_with_lcntx(metadata, &lcntx_l);
    if (err != DLP_CLSC_SUCCESS) {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "Local cntx diverged or corrupted from metadata, "
                 "local cntx values -> MC: %ld, NC: %ld, KC: %ld, "
                 "MR: %ld, NR: %ld\n",
                 lcntx_l.blksz.MC, lcntx_l.blksz.NC, lcntx_l.blksz.KC,
                 lcntx_l.blksz.MR, lcntx_l.blksz.NR);
        dlp_print_msg(msg, __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, err);
        return;
    }

#if (defined(DLP_KERNELS_ZEN4) && (!defined(DLP_GEMM_BF16_JIT)))
    /* While AOCL_DLP_ENABLE_INSTRUCTIONS=AVX2 is enabled in machines that
     * supports DLP_BF16/VNNI with only the ISA check the exeution could enter
     * tiny path and result in seg fault as the tiny path for DLP_BF16->FP32 is
     * not available. Hence the arch_id also has to be verified here.
     */
    if (((arch_id == DLP_ARCH_ZEN4) || (arch_id == DLP_ARCH_ZEN5)
         || (arch_id == DLP_ARCH_ZEN6))
        && (dlp_cpuid_is_avx512bf16_supported() == TRUE)
        && (is_tiny_input_bf16obf16(m_use, n_use, k, &lcntx_l) == TRUE)
        && (dlp_is_single_thread(&rntm_g) == TRUE) && (is_row_major == TRUE)) {
        dlp_gemm_rowvar_tiny_bf16bf16f32of32(
            m_use, n_use, k, a_use, rs_a_use, cs_a_use, mtag_a_use, b_use,
            rs_b_use, cs_b_use, mtag_b_use, (float*)c, rs_c_use, cs_c_use,
            alpha, beta, &lcntx_l, post_op_list, DLP_BF16);
        goto err_hndl;
    }
#endif

    // Create ops bundle for standard GEMM (post-ops only)
    dlp_gemm_ops_bundle_t ops = DLP_GEMM_OPS_BUNDLE_INIT_STANDARD(post_op_list);

#ifdef DLP_ENABLE_OPENMP
    if (dlp_is_single_thread(&rntm_g) == FALSE) {
        dlp_gemm_bf16bf16f32of32_openmp_thread_decorator(
            m_use, n_use, k, a_use, rs_a_use, cs_a_use, mtag_a_use, b_use,
            rs_b_use, cs_b_use, mtag_b_use, (float*)c, rs_c_use, cs_c_use,
            alpha, beta, &rntm_g, &lcntx_l, &ops, DLP_BF16);
    } else
#endif
    {
        dlp_gemm_bf16bf16f32of32_thread_decorator(
            m_use, n_use, k, a_use, rs_a_use, cs_a_use, mtag_a_use, b_use,
            rs_b_use, cs_b_use, mtag_b_use, (float*)c, rs_c_use, cs_c_use,
            alpha, beta, &rntm_g, &lcntx_l, &ops, DLP_BF16);
    }

err_hndl:;
    DLP_GEMM_STOP_LOGGER();
}
