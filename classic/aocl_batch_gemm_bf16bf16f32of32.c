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
#include "dlp_gemm_post_ops.h"
#include "dlp_gemm_types.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "logging/dlp_gemm_logger.h"
#include "threading/dlp_gemm_thread_decor_openmp.h"

void
aocl_batch_gemm_bf16bf16f32of32(const char*      order,
                                const char*      transa,
                                const char*      transb,
                                const md_t*      m,
                                const md_t*      n,
                                const md_t*      k,
                                const float*     alpha,
                                const bfloat16** a,
                                const md_t*      lda,
                                const bfloat16** b,
                                const md_t*      ldb,
                                const float*     beta,
                                float**          c,
                                const md_t*      ldc,
                                const md_t       group_count,
                                const md_t*      group_size,
                                const char*      mem_format_a,
                                const char*      mem_format_b,
                                dlp_metadata_t** metadata)
{
    AOCL_DLP_BATCH_GEMM_NULL_ARGS_CHECK("batch_bf16bf16f32of32", order, transa,
                                        transb, m, n, k, alpha, a, lda, b, ldb,
                                        beta, c, ldc, group_count, group_size,
                                        mem_format_a, mem_format_b, metadata);

    DLP_GEMM_START_LOGGER();
    BATCH_DLP_GEMM_WRITE_LOGGER("bf16bf16f32of32", order, transa, transb,
                                group_count, group_size, m, n, k,
                                ((float*)alpha), lda, mem_format_a, ldb,
                                mem_format_b, ((float*)beta), ldc, metadata);

    // Check if avx512_vnni ISA is supported, dlp_gemm matmul only works with
    // it.
    if (dlp_cpuid_is_avx512bf16_supported() == FALSE) {
        dlp_print_msg(" AVX512_BF16 ISA not supported by processor, "
                      "cannot perform bf16bf16f32 gemm.",
                      __FILE__, __LINE__);
        for (iter_t gc_i = 0; gc_i < group_count; gc_i++) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
        }
        goto err_hndl;
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

#ifdef DLP_GEMM_BF16_JIT
    if (dlp_gemm_get_jit_kernels_generated() == FALSE) {
        dlp_print_msg(" Could not generate bf16bf16f32of32 "
                      " kernels using JIT.",
                      __FILE__, __LINE__);
        for (iter_t gc_i = 0; gc_i < group_count; gc_i++) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
        }
        goto err_hndl;
    }
#endif

    // offset to get subsequent matrix when group_count > 1
    md_t mat_idx = 0;

    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;

    for (iter_t gc_i = 0; gc_i < group_count; gc_i++) {

        DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_SUCCESS);

        // check for validity of params.
        AOCL_DLP_BATCH_GEMM_CHECK(
            "batch_bf16bf16f32of32", order[gc_i], transa[gc_i], transb[gc_i],
            group_count, group_size[gc_i], m[gc_i], n[gc_i], k[gc_i], a[gc_i],
            lda[gc_i], mem_format_a[gc_i], b[gc_i], ldb[gc_i],
            mem_format_b[gc_i], c[gc_i], ldc[gc_i], err_no);
        AOCL_DLP_BATCH_GEMM_MATRIX_PTR_CHECK(a, b, c, mat_idx, group_size[gc_i],
                                             err_no);
        if (err_no != DLP_CLSC_SUCCESS) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], err_no);
            goto err_hndl;
        }

        // Group_size is used across
        md_t g_sz = group_size[gc_i];

        dlp_trans_t dlp_transa;
        dlp_trans_t dlp_transb;

        md_t rs_a;
        md_t cs_a;

        md_t rs_b;
        md_t cs_b;

        md_t rs_c;
        md_t cs_c;

        AOCL_DLP_MEMORY_TAG mtag_a;
        AOCL_DLP_MEMORY_TAG mtag_b;

        const bfloat16 **a_local, **b_local;
        md_t             m_local, n_local, k_local;

        // Convert post op struct to post op linked list format.
        dlp_gemm_post_op post_op_list[AOCL_DLP_MAX_POST_OPS];

        dlp_clsc_err_t err = dlp_gemm_translate_to_post_ops_list(
            metadata[gc_i], post_op_list, (void*)c[gc_i],
            (void*)((order + gc_i)), m[gc_i], n[gc_i]);

        if (err != DLP_CLSC_SUCCESS) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], err);
            goto err_hndl;
        }

        /* Map BLAS chars to their corresponding DLP enumerated type value. */
        dlp_param_map_netlib_to_dlp_trans(transa[gc_i], &dlp_transa);
        dlp_param_map_netlib_to_dlp_trans(transb[gc_i], &dlp_transb);

        bool is_column_major = ((order[gc_i] == 'c') || (order[gc_i] == 'C'));
        if (is_column_major == TRUE) {
            rs_a = ldb[gc_i];
            cs_a = 1;

            if (dlp_is_trans(dlp_transb)) {
                rs_a = 1;
                cs_a = ldb[gc_i];
            }

            rs_b = lda[gc_i];
            cs_b = 1;

            if (dlp_is_trans(dlp_transa)) {
                rs_b = 1;
                cs_b = lda[gc_i];
            }

            dlp_param_map_char_to_lpmtag(mem_format_a[gc_i], &(mtag_b));
            dlp_param_map_char_to_lpmtag(mem_format_b[gc_i], &(mtag_a));

            // Inputs swapped in column major, A becomes B from kernel point
            // of view. Reorder is not supported for column major matrices.
            if (((mtag_b == REORDERED) || (mtag_a == REORDERED))) {
                dlp_print_msg(" Reordering of column major matrices is not "
                              "supported.",
                              __FILE__, __LINE__);
                DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
                goto err_hndl;
            }
            // From 5-loop function point of view,
            // A matrix when in column major storage needs to be packed to
            // row-major storage as kernel expects A matrix to be in
            // row-major format. Inputs swapped in column major, A becomes B
            // from kernel point of view.
            if (dlp_is_trans(dlp_transb)) {
                mtag_a = PACK;
            }

            // swap m & n in case of col-major matrices
            m_local = n[gc_i];
            n_local = m[gc_i];

            // swap a & b pointers in case of col-major matrices
            a_local = (b + mat_idx);
            b_local = (a + mat_idx);
        } else // row-major
        {
            rs_a = lda[gc_i];
            cs_a = 1;

            if (dlp_is_trans(dlp_transa)) {
                rs_a = 1;
                cs_a = lda[gc_i];
            }

            rs_b = ldb[gc_i];
            cs_b = 1;

            if (dlp_is_trans(dlp_transb)) {
                rs_b = 1;
                cs_b = ldb[gc_i];
            }

            dlp_param_map_char_to_lpmtag(mem_format_a[gc_i], &(mtag_a));
            dlp_param_map_char_to_lpmtag(mem_format_b[gc_i], &(mtag_b));

            // Reorder is not supported for A matrix
            if (mtag_a == REORDERED) {
                dlp_print_msg(" Reordering of A matrix is not supported in row "
                              "major case.",
                              __FILE__, __LINE__);
                DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
                goto err_hndl;
            }
            // From 5-loop function point of view,
            // A matrix when in column major storage needs to be packed to
            // row-major storage as kernel expects A matrix to be in
            // row-major format.
            if (dlp_is_trans(dlp_transa)) {
                mtag_a = PACK;
            }

            // copy the values of m & n
            m_local = m[gc_i];
            n_local = n[gc_i];

            // copy the values of a & b pointers
            a_local = (a + mat_idx);
            b_local = (b + mat_idx);
        }

        k_local = k[gc_i];

        rs_c = ldc[gc_i];
        cs_c = 1;

        // From 5-loop function point of view
        // B matrix needs to be packed in a certain format in order to be
        // loaded and used in bf16 instrution. As such the mtag_b always
        // needs to be either packed or reordered. B matrix as it is
        // (unpacked) cannot be used, and the mtag_b is set to packed to
        // enable runtime packing.
        if (mtag_b == UNPACKED) {
            mtag_b = PACK;
        }

        // Initialize a local runtime with global settings if necessary. Note
        // that in the case that a runtime is passed in, we make a local copy.
        dlp_rntm_t rntm_g;
        dlp_rntm_init_from_global(&rntm_g);

        // Create local copy, since each thread in a multi-instance setup
        // modified the context object.
        dlp_gemm_cntx_t lcntx_l =
            *(dlp_gemm_get_global_cntx_obj(BF16BF16F32OF32));
        err = dlp_gemm_upd_cntx_with_metadata(BF16BF16F32OF32, &lcntx_l,
                                              metadata[gc_i]);
        if (err != DLP_CLSC_SUCCESS) {
            dlp_print_msg(" Failed to update context with metadata.", __FILE__,
                          __LINE__);
            DLP_METADATA_SET_ERROR(metadata[gc_i], err);
            goto err_hndl;
        }

        // A malformed hint is refused here rather than modelled against. Only
        // the negative test applies: the decorator splits the pool across the
        // GEMMs in the group, so no one of them runs on the count this runtime
        // states and there is nothing here to hold nt_hint to. The tile is
        // therefore costed against nt_hint rather than the count this GEMM
        // runs on; the panel width both ends derive from the same hints
        // either way.
        err = dlp_gemm_validate_hints(&lcntx_l);
        if (err != DLP_CLSC_SUCCESS) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "GEMM hints must be zero (unset) or positive, got "
                     "m_hint: %ld nt_hint: %ld\n",
                     (lcntx_l.gemm_kernel_hints).m_hint,
                     (lcntx_l.gemm_kernel_hints).nt_hint);
            dlp_print_msg(msg, __FILE__, __LINE__);
            DLP_METADATA_SET_ERROR(metadata[gc_i], err);
            goto err_hndl;
        }

        // Initialize DLP Plus kernel path.
        lcntx_l.dlp_kernel_hndl.kernel_base = NULL;
        // All the g_sz inputs in a given group will have the same matrix
        // dimensions/attributes. Therefore the DE and Jit generation in
        // DLP Plus can proceed with any 1 input from this group.
        dlp_init_and_get_kernel_hndl(DLP_KERNEL_BF16BF16F32OF32, order[gc_i],
                                     mtag_a, mtag_b, m_local, n_local, k_local,
                                     rs_a, cs_a, rs_b, cs_b, rs_c, cs_c,
                                     (void*)&alpha[gc_i], (void*)&beta[gc_i],
                                     post_op_list, &lcntx_l, DLP_F32);

        // Invalid handle means that the jit kernel generation has failed. Do
        // not attempt to execute the kernel, and return an error instead.
        if (lcntx_l.dlp_kernel_hndl.kernel_base == NULL) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_INVALID_JIT_KERNEL);
            goto err_hndl;
        }

        // JIT pack B (BF16): the pack-B kernel (full NR + fringe ladder) for
        // the AVX512-BF16 path, mirroring the GEMM kernel init above. cs_b == 1
        // selects the row-major packer; cs_b != 1 (with rs_b == 1, i.e. transB)
        // selects the column-major 16x16 transpose packer.
        lcntx_l.dlp_pack_kernel_hndl.pack_b_hndl.kernel_base = NULL;
        dlp_init_and_get_packb_kernel_hndl(DLP_KERNEL_BF16BF16F32OF32, n_local,
                                           k_local, rs_b, cs_b, &lcntx_l);

        // A pack-B kernel is only generated when the configured arch has
        // AVX512-BF16. Without it, BF16 reroutes to the F32 kernels, which
        // convert/unreorder B instead, so a NULL handle is expected there and
        // execution continues. With it, a NULL handle means JIT generation
        // failed, which is an error. n=1 gemv copies B rather than packing it
        // into an NR panel (NR == 1), and therefore needs no pack-B kernel.
        if ((dlp_cpuid_is_avx512bf16_configured() == TRUE)
            && (lcntx_l.blksz.NR > 1)
            && (lcntx_l.dlp_pack_kernel_hndl.pack_b_hndl.kernel_base == NULL)) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_INVALID_JIT_KERNEL);
            goto err_hndl;
        }

        // Both inits are done, so the tile the packed buffers are written
        // against is final.
        dlp_upd_pack_strides(DLP_KERNEL_BF16BF16F32OF32, &lcntx_l);

        err = dlp_gemm_validate_metadata_with_lcntx(metadata[gc_i], &lcntx_l);
        if (err != DLP_CLSC_SUCCESS) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "Local cntx diverged or corrupted from metadata, "
                     "local cntx values -> MC: %ld, NC: %ld, KC: %ld, "
                     "MR: %ld, NR: %ld\n",
                     lcntx_l.blksz.MC, lcntx_l.blksz.NC, lcntx_l.blksz.KC,
                     lcntx_l.blksz.MR, lcntx_l.blksz.NR);
            dlp_print_msg(msg, __FILE__, __LINE__);
            DLP_METADATA_SET_ERROR(metadata[gc_i], err);
            goto err_hndl;
        }

        // Create ops bundle for standard GEMM (post-ops only)
        dlp_gemm_ops_bundle_t ops =
            DLP_GEMM_OPS_BUNDLE_INIT_STANDARD(post_op_list);

#ifdef DLP_ENABLE_OPENMP
        batch_dlp_gemm_bf16bf16f32of32_openmp_thread_decorator(
            g_sz, &m_local, &n_local, &k_local, (const bfloat16**)a_local,
            &rs_a, &cs_a, &mtag_a, (const bfloat16**)b_local, &rs_b, &cs_b,
            &mtag_b, &c[mat_idx], &rs_c, &cs_c, alpha[gc_i], beta[gc_i],
            &rntm_g, &lcntx_l, &ops, DLP_F32);

#else
        batch_dlp_gemm_bf16bf16f32of32_thread_decorator(
            g_sz, &m_local, &n_local, &k_local, (const bfloat16**)a_local,
            &rs_a, &cs_a, &mtag_a, (const bfloat16**)b_local, &rs_b, &cs_b,
            &mtag_b, &c[mat_idx], &rs_c, &cs_c, alpha[gc_i], beta[gc_i],
            &rntm_g, &lcntx_l, &ops, DLP_F32);
#endif
        mat_idx += g_sz;
    }
err_hndl:;
    DLP_GEMM_STOP_LOGGER();
}

void
aocl_batch_gemm_bf16bf16f32obf16(const char*      order,
                                 const char*      transa,
                                 const char*      transb,
                                 const md_t*      m,
                                 const md_t*      n,
                                 const md_t*      k,
                                 const float*     alpha,
                                 const bfloat16** a,
                                 const md_t*      lda,
                                 const bfloat16** b,
                                 const md_t*      ldb,
                                 const float*     beta,
                                 bfloat16**       c,
                                 const md_t*      ldc,
                                 const md_t       group_count,
                                 const md_t*      group_size,
                                 const char*      mem_format_a,
                                 const char*      mem_format_b,
                                 dlp_metadata_t** metadata)
{
    AOCL_DLP_BATCH_GEMM_NULL_ARGS_CHECK("batch_bf16bf16f32obf16", order, transa,
                                        transb, m, n, k, alpha, a, lda, b, ldb,
                                        beta, c, ldc, group_count, group_size,
                                        mem_format_a, mem_format_b, metadata);

    DLP_GEMM_START_LOGGER();
    BATCH_DLP_GEMM_WRITE_LOGGER("bf16bf16f32obf16", order, transa, transb,
                                group_count, group_size, m, n, k,
                                ((float*)alpha), lda, mem_format_a, ldb,
                                mem_format_b, ((float*)beta), ldc, metadata);

    // Check if avx512_vnni ISA is supported, dlp_gemm matmul only works with
    // it.
    if (dlp_cpuid_is_avx512bf16_supported() == FALSE) {
        dlp_print_msg(" AVX512_BF16 ISA not supported by processor, "
                      "cannot perform bf16bf16f32 gemm.",
                      __FILE__, __LINE__);
        for (iter_t gc_i = 0; gc_i < group_count; gc_i++) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
        }
        goto err_hndl;
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

#ifdef DLP_GEMM_BF16_JIT
    if (dlp_gemm_get_jit_kernels_generated() == FALSE) {
        dlp_print_msg(" Could not generate bf16bf16f32of32 "
                      " kernels using JIT.",
                      __FILE__, __LINE__);
        for (iter_t gc_i = 0; gc_i < group_count; gc_i++) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
        }
        goto err_hndl;
    }
#endif
    // offset to get subsequent matrix when group_count > 1
    md_t mat_idx = 0;

    // check for validity of params.
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;

    for (iter_t gc_i = 0; gc_i < group_count; gc_i++) {

        DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_SUCCESS);

        // Group_size is used across
        md_t g_sz = group_size[gc_i];

        // check for validity of params.
        AOCL_DLP_BATCH_GEMM_CHECK(
            "batch_bf16bf16f32obf16", order[gc_i], transa[gc_i], transb[gc_i],
            group_count, g_sz, m[gc_i], n[gc_i], k[gc_i], a[gc_i], lda[gc_i],
            mem_format_a[gc_i], b[gc_i], ldb[gc_i], mem_format_b[gc_i], c[gc_i],
            ldc[gc_i], err_no);
        AOCL_DLP_BATCH_GEMM_MATRIX_PTR_CHECK(a, b, c, mat_idx, g_sz, err_no);

        if (err_no != DLP_CLSC_SUCCESS) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], err_no);
            goto err_hndl;
        }

        md_t rs_a;
        md_t cs_a;

        md_t rs_b;
        md_t cs_b;

        md_t rs_c;
        md_t cs_c;

        AOCL_DLP_MEMORY_TAG mtag_a;
        AOCL_DLP_MEMORY_TAG mtag_b;

        const bfloat16 **a_local, **b_local;
        md_t             m_local, n_local, k_local;

        // Convert post op struct to post op linked list format.
        dlp_gemm_post_op post_op_list[AOCL_DLP_MAX_POST_OPS];

        dlp_clsc_err_t err = dlp_gemm_translate_to_post_ops_list(
            metadata[gc_i], post_op_list, (void*)c[gc_i],
            (void*)((order + gc_i)), m[gc_i], n[gc_i]);

        if (err != DLP_CLSC_SUCCESS) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], err);
            goto err_hndl;
        }

        dlp_trans_t dlp_transa;
        dlp_trans_t dlp_transb;

        /* Map BLAS chars to their corresponding DLP enumerated type value. */
        dlp_param_map_netlib_to_dlp_trans(transa[gc_i], &dlp_transa);
        dlp_param_map_netlib_to_dlp_trans(transb[gc_i], &dlp_transb);

        bool is_column_major = ((order[gc_i] == 'c') || (order[gc_i] == 'C'));

        if (is_column_major == TRUE) {
            rs_a = ldb[gc_i];
            cs_a = 1;

            if (dlp_is_trans(dlp_transb)) {
                rs_a = 1;
                cs_a = ldb[gc_i];
            }

            rs_b = lda[gc_i];
            cs_b = 1;

            if (dlp_is_trans(dlp_transa)) {
                rs_b = 1;
                cs_b = lda[gc_i];
            }

            dlp_param_map_char_to_lpmtag(mem_format_a[gc_i], &(mtag_b));
            dlp_param_map_char_to_lpmtag(mem_format_b[gc_i], &(mtag_a));

            // Inputs swapped in column major, A becomes B from kernel point
            // of view. Reorder is not supported for column major matrices.
            if (((mtag_b == REORDERED) || (mtag_a == REORDERED))) {
                dlp_print_msg(" Reordering of column major matrices is not "
                              "supported.",
                              __FILE__, __LINE__);
                DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
                goto err_hndl;
            }
            // From 5-loop function point of view,
            // A matrix when in column major storage needs to be packed to
            // row-major storage as kernel expects A matrix to be in
            // row-major format. Inputs swapped in column major, A becomes B
            // from kernel point of view.
            if (dlp_is_trans(dlp_transb)) {
                mtag_a = PACK;
            }

            // swap m & n in case of col-major matrices
            m_local = n[gc_i];
            n_local = m[gc_i];

            // swap a & b pointers in case of col-major matrices
            a_local = (b + mat_idx);
            b_local = (a + mat_idx);
        } else // row-major
        {
            rs_a = lda[gc_i];
            cs_a = 1;

            if (dlp_is_trans(dlp_transa)) {
                rs_a = 1;
                cs_a = lda[gc_i];
            }

            rs_b = ldb[gc_i];
            cs_b = 1;

            if (dlp_is_trans(dlp_transb)) {
                rs_b = 1;
                cs_b = ldb[gc_i];
            }

            dlp_param_map_char_to_lpmtag(mem_format_a[gc_i], &(mtag_a));
            dlp_param_map_char_to_lpmtag(mem_format_b[gc_i], &(mtag_b));

            // Reorder is not supported for A matrix
            if (mtag_a == REORDERED) {
                dlp_print_msg(" Reordering of A matrix is not supported in row "
                              "major case.",
                              __FILE__, __LINE__);
                DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
                goto err_hndl;
            }
            // From 5-loop function point of view,
            // A matrix when in column major storage needs to be packed to
            // row-major storage as kernel expects A matrix to be in
            // row-major format.
            if (dlp_is_trans(dlp_transa)) {
                mtag_a = PACK;
            }

            // copy the values of m & n
            m_local = m[gc_i];
            n_local = n[gc_i];

            // copy the values of a & b pointers
            a_local = (a + mat_idx);
            b_local = (b + mat_idx);
        }

        k_local = k[gc_i];

        rs_c = ldc[gc_i];
        cs_c = 1;

        // From 5-loop function point of view
        // B matrix needs to be packed in a certain format in order to be
        // loaded and used in bf16 instrution. As such the mtag_b always
        // needs to be either packed or reordered. B matrix as it is
        // (unpacked) cannot be used, and the mtag_b is set to packed to
        // enable runtime packing.
        if (mtag_b == UNPACKED) {
            mtag_b = PACK;
        }

        // Initialize a local runtime with global settings if necessary. Note
        // that in the case that a runtime is passed in, we make a local copy.
        dlp_rntm_t rntm_g;
        dlp_rntm_init_from_global(&rntm_g);

        // Create local copy, since each thread in a multi-instance setup
        // modified the context object.
        dlp_gemm_cntx_t lcntx_l =
            *(dlp_gemm_get_global_cntx_obj(BF16BF16F32OF32));
        err = dlp_gemm_upd_cntx_with_metadata(BF16BF16F32OF32, &lcntx_l,
                                              metadata[gc_i]);
        if (err != DLP_CLSC_SUCCESS) {
            dlp_print_msg(" Failed to update context with metadata.", __FILE__,
                          __LINE__);
            DLP_METADATA_SET_ERROR(metadata[gc_i], err);
            goto err_hndl;
        }

        // A malformed hint is refused here rather than modelled against. Only
        // the negative test applies: the decorator splits the pool across the
        // GEMMs in the group, so no one of them runs on the count this runtime
        // states and there is nothing here to hold nt_hint to. The tile is
        // therefore costed against nt_hint rather than the count this GEMM
        // runs on; the panel width both ends derive from the same hints
        // either way.
        err = dlp_gemm_validate_hints(&lcntx_l);
        if (err != DLP_CLSC_SUCCESS) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "GEMM hints must be zero (unset) or positive, got "
                     "m_hint: %ld nt_hint: %ld\n",
                     (lcntx_l.gemm_kernel_hints).m_hint,
                     (lcntx_l.gemm_kernel_hints).nt_hint);
            dlp_print_msg(msg, __FILE__, __LINE__);
            DLP_METADATA_SET_ERROR(metadata[gc_i], err);
            goto err_hndl;
        }

        // Initialize DLP Plus kernel path.
        lcntx_l.dlp_kernel_hndl.kernel_base = NULL;
        // All the g_sz inputs in a given group will have the same matrix
        // dimensions/attributes. Therefore the DE and Jit generation in
        // DLP Plus can proceed with any 1 input from this group.
        dlp_init_and_get_kernel_hndl(DLP_KERNEL_BF16BF16F32OF32, order[gc_i],
                                     mtag_a, mtag_b, m_local, n_local, k_local,
                                     rs_a, cs_a, rs_b, cs_b, rs_c, cs_c,
                                     (void*)&alpha[gc_i], (void*)&beta[gc_i],
                                     post_op_list, &lcntx_l, DLP_BF16);

        // Invalid handle means that the jit kernel generation has failed. Do
        // not attempt to execute the kernel, and return an error instead.
        if (lcntx_l.dlp_kernel_hndl.kernel_base == NULL) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_INVALID_JIT_KERNEL);
            goto err_hndl;
        }

        // JIT pack B (BF16): the pack-B kernel (full NR + fringe ladder) for
        // the AVX512-BF16 path, mirroring the GEMM kernel init above. cs_b == 1
        // selects the row-major packer; cs_b != 1 (with rs_b == 1, i.e. transB)
        // selects the column-major 16x16 transpose packer. The B panel layout
        // does not depend on the GEMM output type.
        lcntx_l.dlp_pack_kernel_hndl.pack_b_hndl.kernel_base = NULL;
        dlp_init_and_get_packb_kernel_hndl(DLP_KERNEL_BF16BF16F32OBF16, n_local,
                                           k_local, rs_b, cs_b, &lcntx_l);

        // A pack-B kernel is only generated when the configured arch has
        // AVX512-BF16. Without it, BF16 reroutes to the F32 kernels, which
        // convert/unreorder B instead, so a NULL handle is expected there and
        // execution continues. With it, a NULL handle means JIT generation
        // failed, which is an error. n=1 gemv copies B rather than packing it
        // into an NR panel (NR == 1), and therefore needs no pack-B kernel.
        if ((dlp_cpuid_is_avx512bf16_configured() == TRUE)
            && (lcntx_l.blksz.NR > 1)
            && (lcntx_l.dlp_pack_kernel_hndl.pack_b_hndl.kernel_base == NULL)) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_INVALID_JIT_KERNEL);
            goto err_hndl;
        }

        // Both inits are done, so the tile the packed buffers are written
        // against is final.
        dlp_upd_pack_strides(DLP_KERNEL_BF16BF16F32OBF16, &lcntx_l);

        err = dlp_gemm_validate_metadata_with_lcntx(metadata[gc_i], &lcntx_l);
        if (err != DLP_CLSC_SUCCESS) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "Local cntx diverged or corrupted from metadata, "
                     "local cntx values -> MC: %ld, NC: %ld, KC: %ld, "
                     "MR: %ld, NR: %ld\n",
                     lcntx_l.blksz.MC, lcntx_l.blksz.NC, lcntx_l.blksz.KC,
                     lcntx_l.blksz.MR, lcntx_l.blksz.NR);
            dlp_print_msg(msg, __FILE__, __LINE__);
            DLP_METADATA_SET_ERROR(metadata[gc_i], err);
            goto err_hndl;
        }

        // Create ops bundle for standard GEMM (post-ops only)
        dlp_gemm_ops_bundle_t ops =
            DLP_GEMM_OPS_BUNDLE_INIT_STANDARD(post_op_list);

#ifdef DLP_ENABLE_OPENMP
        batch_dlp_gemm_bf16bf16f32of32_openmp_thread_decorator(
            g_sz, &m_local, &n_local, &k_local, (const bfloat16**)a_local,
            &rs_a, &cs_a, &mtag_a, (const bfloat16**)b_local, &rs_b, &cs_b,
            &mtag_b, (float**)&c[mat_idx], &rs_c, &cs_c, alpha[gc_i],
            beta[gc_i], &rntm_g, &lcntx_l, &ops, DLP_BF16);

#else
        batch_dlp_gemm_bf16bf16f32of32_thread_decorator(
            g_sz, &m_local, &n_local, &k_local, (const bfloat16**)a_local,
            &rs_a, &cs_a, &mtag_a, (const bfloat16**)b_local, &rs_b, &cs_b,
            &mtag_b, (float**)&c[mat_idx], &rs_c, &cs_c, alpha[gc_i],
            beta[gc_i], &rntm_g, &lcntx_l, &ops, DLP_BF16);
#endif
        mat_idx += g_sz;
    }
err_hndl:;
    DLP_GEMM_STOP_LOGGER();
}
