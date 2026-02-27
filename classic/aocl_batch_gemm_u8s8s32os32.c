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

#include "aocl_gemm_check.h"
#include "classic/aocl_gemm_interface_apis.h"
#include "config/lpgemm_config.h"
#include "gemm_utils/lpgemm_utils.h"
#include "logging/lpgemm_logger.h"
#include "lpgemm_5loop_interface_apis.h"
#include "lpgemm_post_ops.h"
#include "lpgemm_types.h"
#include "threading/lpgemm_thread_decor_openmp.h"

void
aocl_batch_gemm_u8s8s32os32(const char*      order,
                            const char*      transa,
                            const char*      transb,
                            const md_t*      m,
                            const md_t*      n,
                            const md_t*      k,
                            const int32_t*   alpha,
                            const uint8_t**  a,
                            const md_t*      lda,
                            const int8_t**   b,
                            const md_t*      ldb,
                            const int32_t*   beta,
                            int32_t**        c,
                            const md_t*      ldc,
                            const md_t       group_count,
                            const md_t*      group_size,
                            const char*      mem_format_a,
                            const char*      mem_format_b,
                            dlp_metadata_t** metadata)
{
    LPGEMM_START_LOGGER();
    BATCH_LPGEMM_WRITE_LOGGER("u8s8s32os32", order, transa, transb, group_count,
                              group_size, m, n, k, alpha, lda, mem_format_a,
                              ldb, mem_format_b, beta, ldc, metadata);

    // Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
    if (dlp_cpuid_is_avx512vnni_supported() == FALSE) {
        dlp_print_msg(" AVX512_VNNI ISA not supported by processor, "
                      "cannot perform u8s8s32 gemm.",
                      __FILE__, __LINE__);
        for (md_t gc_i = 0; gc_i < group_count; gc_i++) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
        }
        goto err_hndl; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    // offset to get subsequent matrix when group_count > 1
    md_t mat_idx = 0;

    // check for validity of params.
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;

    for (md_t gc_i = 0; gc_i < group_count; gc_i++) {

        DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_SUCCESS);

        md_t g_sz = group_size[gc_i];
        // check for validity of params.
        AOCL_BATCH_GEMM_CHECK("batch_u8s8s32os32", order[gc_i], transa[gc_i],
                              transb[gc_i], group_count, g_sz, m[gc_i], n[gc_i],
                              k[gc_i], a[gc_i], lda[gc_i], mem_format_a[gc_i],
                              b[gc_i], ldb[gc_i], mem_format_b[gc_i], c[gc_i],
                              ldc[gc_i], err_no);

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

        AOCL_MEMORY_TAG mtag_a;
        AOCL_MEMORY_TAG mtag_b;

        // Convert post op struct to post op linked list format.
        lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];

        dlp_clsc_err_t err = lpgemm_translate_to_post_ops_list(
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

        const uint8_t** a_local;
        const int8_t**  b_local;
        md_t            m_local, n_local, k_local;

        bool is_column_major = ((order[gc_i] == 'c') || (order[gc_i] == 'C'));
        // Column major isn't supported
        if (is_column_major == TRUE) {
            dlp_print_msg("Column major inputs not supported.", __FILE__,
                          __LINE__);
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
            goto err_hndl;
        }
        // Batch GEMM executes only for the Row major case.
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
        // A matrix packing is not supported in row major case.
        // If A matrix is packed and not transposed, set to Unpack
        // and proceed with GEMM.
        if ((mtag_a == PACK) && (!dlp_is_trans(dlp_transa))) {
            mtag_a = UNPACKED;
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

        lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj(U8S8S32OS32);
        lpgemm_cntx_t  lcntx_l;
        // Create local copy, since each thread in a multi-instance setup
        // modifies the context object.
        lcntx_l = *lcntx_g;

        // Initialize DLP Plus kernel path.
        lcntx_l.dlp_kernel_hndl.kernel_base = NULL;
        // All the g_sz inputs in a given group will have the same matrix
        // dimensions/attributes. Therefore the DE and Jit generation in
        // DLP Plus can proceed with any 1 input from this group.
        dlp_init_and_get_kernel_hndl(
            DLP_KERNEL_U8S8S32OS32, order[gc_i], mtag_a, mtag_b, m_local,
            n_local, k_local, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c,
            (void*)&alpha[gc_i], (void*)&beta[gc_i], post_op_list,
            lcntx_l.blksz.MR, lcntx_l.blksz.NR, lcntx_l.blksz.KC, DLP_S32,
            &lcntx_l.dlp_kernel_hndl);

        // Invalid handle means that the jit kernel generation has failed. Do
        // not attempt to execute the kernel, and return an error instead.
        if (lcntx_l.dlp_kernel_hndl.kernel_base == NULL) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_INVALID_JIT_KERNEL);
            goto err_hndl;
        }

        // Create ops bundle for standard GEMM (post-ops only)
        lpgemm_ops_bundle_t ops = LPGEMM_OPS_BUNDLE_INIT_STANDARD(post_op_list);

#ifdef DLP_ENABLE_OPENMP
        batch_lpgemm_u8s8s32o32_openmp_thread_decorator(
            g_sz, &m_local, &n_local, &k_local, (const uint8_t**)a_local, &rs_a,
            &cs_a, &mtag_a, (const int8_t**)b_local, &rs_b, &cs_b, &mtag_b,
            &c[mat_idx], &rs_c, &cs_c, alpha[gc_i], beta[gc_i], &rntm_g,
            &lcntx_l, &ops, DLP_S32);
#else
        batch_lpgemm_u8s8s32o32_thread_decorator(
            g_sz, &m_local, &n_local, &k_local, (const uint8_t**)a_local, &rs_a,
            &cs_a, &mtag_a, (const int8_t**)b_local, &rs_b, &cs_b, &mtag_b,
            &c[mat_idx], &rs_c, &cs_c, alpha[gc_i], beta[gc_i], &rntm_g,
            &lcntx_l, &ops, DLP_S32);
#endif
        mat_idx += g_sz;
    }
err_hndl:;
    LPGEMM_STOP_LOGGER();
}

void
aocl_batch_gemm_u8s8s32os8(const char*      order,
                           const char*      transa,
                           const char*      transb,
                           const md_t*      m,
                           const md_t*      n,
                           const md_t*      k,
                           const int32_t*   alpha,
                           const uint8_t**  a,
                           const md_t*      lda,
                           const int8_t**   b,
                           const md_t*      ldb,
                           const int32_t*   beta,
                           int8_t**         c,
                           const md_t*      ldc,
                           const md_t       group_count,
                           const md_t*      group_size,
                           const char*      mem_format_a,
                           const char*      mem_format_b,
                           dlp_metadata_t** metadata)
{
    LPGEMM_START_LOGGER();
    BATCH_LPGEMM_WRITE_LOGGER("u8s8s32os8", order, transa, transb, group_count,
                              group_size, m, n, k, alpha, lda, mem_format_a,
                              ldb, mem_format_b, beta, ldc, metadata);

    // Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
    if (dlp_cpuid_is_avx512vnni_supported() == FALSE) {
        dlp_print_msg(" AVX512_VNNI ISA not supported by processor, "
                      "cannot perform u8s8s32 gemm.",
                      __FILE__, __LINE__);
        for (md_t gc_i = 0; gc_i < group_count; gc_i++) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
        }
        goto err_hndl; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    // offset to get subsequent matrix when group_count > 1
    md_t mat_idx = 0;

    // check for validity of params.
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;

    for (md_t gc_i = 0; gc_i < group_count; gc_i++) {

        DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_SUCCESS);

        md_t g_sz = group_size[gc_i];
        // check for validity of params.
        AOCL_BATCH_GEMM_CHECK("batch_u8s8s32os8", order[gc_i], transa[gc_i],
                              transb[gc_i], gc_i, g_sz, m[gc_i], n[gc_i],
                              k[gc_i], a[gc_i], lda[gc_i], mem_format_a[gc_i],
                              b[gc_i], ldb[gc_i], mem_format_b[gc_i], c[gc_i],
                              ldc[gc_i], err_no);

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

        AOCL_MEMORY_TAG mtag_a;
        AOCL_MEMORY_TAG mtag_b;

        const uint8_t** a_local;
        const int8_t**  b_local;
        md_t            m_local, n_local, k_local;

        // Convert post op struct to post op linked list format.
        lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];

        dlp_clsc_err_t err = lpgemm_translate_to_post_ops_list(
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
        // Column major isn't supported
        if (is_column_major == TRUE) {
            dlp_print_msg("Column major inputs not supported.", __FILE__,
                          __LINE__);
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
            goto err_hndl;
        }
        // Batch GEMM executes only for the Row major case.
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
        // A matrix packing is not supported in row major case.
        // If A matrix is packed and not transposed, set to Unpack
        // and proceed with GEMM.
        if ((mtag_a == PACK) && (!dlp_is_trans(dlp_transa))) {
            mtag_a = UNPACKED;
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

        lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj(U8S8S32OS32);
        lpgemm_cntx_t  lcntx_l;
        // Create local copy, since each thread in a multi-instance setup
        // modifies the context object.
        lcntx_l = *lcntx_g;

        // Initialize DLP Plus kernel path.
        lcntx_l.dlp_kernel_hndl.kernel_base = NULL;
        // All the g_sz inputs in a given group will have the same matrix
        // dimensions/attributes. Therefore the DE and Jit generation in
        // DLP Plus can proceed with any 1 input from this group.
        dlp_init_and_get_kernel_hndl(
            DLP_KERNEL_U8S8S32OS8, order[gc_i], mtag_a, mtag_b, m_local,
            n_local, k_local, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c,
            (void*)&alpha[gc_i], (void*)&beta[gc_i], post_op_list,
            lcntx_l.blksz.MR, lcntx_l.blksz.NR, lcntx_l.blksz.KC, DLP_S8,
            &lcntx_l.dlp_kernel_hndl);

        // Invalid handle means that the jit kernel generation has failed. Do
        // not attempt to execute the kernel, and return an error instead.
        if (lcntx_l.dlp_kernel_hndl.kernel_base == NULL) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_INVALID_JIT_KERNEL);
            goto err_hndl;
        }

        // Create ops bundle for standard GEMM (post-ops only)
        lpgemm_ops_bundle_t ops = LPGEMM_OPS_BUNDLE_INIT_STANDARD(post_op_list);

#ifdef DLP_ENABLE_OPENMP
        batch_lpgemm_u8s8s32o32_openmp_thread_decorator(
            g_sz, &m_local, &n_local, &k_local, (const uint8_t**)a_local, &rs_a,
            &cs_a, &mtag_a, (const int8_t**)b_local, &rs_b, &cs_b, &mtag_b,
            (int32_t**)&c[mat_idx], &rs_c, &cs_c, alpha[gc_i], beta[gc_i],
            &rntm_g, &lcntx_l, &ops, DLP_S8);
#else
        batch_lpgemm_u8s8s32o32_thread_decorator(
            g_sz, &m_local, &n_local, &k_local, (const uint8_t**)a_local, &rs_a,
            &cs_a, &mtag_a, (const int8_t**)b_local, &rs_b, &cs_b, &mtag_b,
            (int32_t**)&c[mat_idx], &rs_c, &cs_c, alpha[gc_i], beta[gc_i],
            &rntm_g, &lcntx_l, &ops, DLP_S8);
#endif
        mat_idx += g_sz;
    }
err_hndl:;
    LPGEMM_STOP_LOGGER();
}

void
aocl_batch_gemm_u8s8s32of32(const char*      order,
                            const char*      transa,
                            const char*      transb,
                            const md_t*      m,
                            const md_t*      n,
                            const md_t*      k,
                            const int32_t*   alpha,
                            const uint8_t**  a,
                            const md_t*      lda,
                            const int8_t**   b,
                            const md_t*      ldb,
                            const int32_t*   beta,
                            float**          c,
                            const md_t*      ldc,
                            const md_t       group_count,
                            const md_t*      group_size,
                            const char*      mem_format_a,
                            const char*      mem_format_b,
                            dlp_metadata_t** metadata)
{
    LPGEMM_START_LOGGER();
    BATCH_LPGEMM_WRITE_LOGGER("u8s8s32of32", order, transa, transb, group_count,
                              group_size, m, n, k, alpha, lda, mem_format_a,
                              ldb, mem_format_b, beta, ldc, metadata);

    // Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
    if (dlp_cpuid_is_avx512vnni_supported() == FALSE) {
        dlp_print_msg(" AVX512_VNNI ISA not supported by processor, "
                      "cannot perform u8s8s32of32 gemm.",
                      __FILE__, __LINE__);
        for (md_t gc_i = 0; gc_i < group_count; gc_i++) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
        }
        goto err_hndl; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    // offset to get subsequent matrix when group_count > 1
    md_t mat_idx = 0;

    // check for validity of params.
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;

    for (md_t gc_i = 0; gc_i < group_count; gc_i++) {

        DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_SUCCESS);

        md_t g_sz = group_size[gc_i];

        // check for validity of params.
        AOCL_BATCH_GEMM_CHECK("batch_u8s8s32of32", order[gc_i], transa[gc_i],
                              transb[gc_i], gc_i, g_sz, m[gc_i], n[gc_i],
                              k[gc_i], a[gc_i], lda[gc_i], mem_format_a[gc_i],
                              b[gc_i], ldb[gc_i], mem_format_b[gc_i], c[gc_i],
                              ldc[gc_i], err_no);

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

        AOCL_MEMORY_TAG mtag_a;
        AOCL_MEMORY_TAG mtag_b;

        const uint8_t** a_local;
        const int8_t**  b_local;
        md_t            m_local, n_local, k_local;

        // Convert post op struct to post op linked list format.
        lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];

        dlp_clsc_err_t err = lpgemm_translate_to_post_ops_list(
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
        // Column major isn't supported
        if (is_column_major == TRUE) {
            dlp_print_msg("Column major inputs not supported.", __FILE__,
                          __LINE__);
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
            goto err_hndl;
        }
        // Batch GEMM executes only for the Row major case.
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
        // A matrix packing is not supported in row major case.
        // If A matrix is packed and not transposed, set to Unpack
        // and proceed with GEMM.
        if ((mtag_a == PACK) && (!dlp_is_trans(dlp_transa))) {
            mtag_a = UNPACKED;
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

        lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj(U8S8S32OS32);
        lpgemm_cntx_t  lcntx_l;
        // Create local copy, since each thread in a multi-instance setup
        // modifies the context object.
        lcntx_l = *lcntx_g;

        // Initialize DLP Plus kernel path.
        lcntx_l.dlp_kernel_hndl.kernel_base = NULL;
        // All the g_sz inputs in a given group will have the same matrix
        // dimensions/attributes. Therefore the DE and Jit generation in
        // DLP Plus can proceed with any 1 input from this group.
        dlp_init_and_get_kernel_hndl(
            DLP_KERNEL_U8S8S32OF32, order[gc_i], mtag_a, mtag_b, m_local,
            n_local, k_local, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c,
            (void*)&alpha[gc_i], (void*)&beta[gc_i], post_op_list,
            lcntx_l.blksz.MR, lcntx_l.blksz.NR, lcntx_l.blksz.KC, DLP_F32,
            &lcntx_l.dlp_kernel_hndl);

        // Invalid handle means that the jit kernel generation has failed. Do
        // not attempt to execute the kernel, and return an error instead.
        if (lcntx_l.dlp_kernel_hndl.kernel_base == NULL) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_INVALID_JIT_KERNEL);
            goto err_hndl;
        }

        // Create ops bundle for standard GEMM (post-ops only)
        lpgemm_ops_bundle_t ops = LPGEMM_OPS_BUNDLE_INIT_STANDARD(post_op_list);

#ifdef DLP_ENABLE_OPENMP
        batch_lpgemm_u8s8s32o32_openmp_thread_decorator(
            g_sz, &m_local, &n_local, &k_local, (const uint8_t**)a_local, &rs_a,
            &cs_a, &mtag_a, (const int8_t**)b_local, &rs_b, &cs_b, &mtag_b,
            (int32_t**)&c[mat_idx], &rs_c, &cs_c, alpha[gc_i], beta[gc_i],
            &rntm_g, &lcntx_l, &ops, DLP_F32);
#else
        batch_lpgemm_u8s8s32o32_thread_decorator(
            g_sz, &m_local, &n_local, &k_local, (const uint8_t**)a_local, &rs_a,
            &cs_a, &mtag_a, (const int8_t**)b_local, &rs_b, &cs_b, &mtag_b,
            (int32_t**)&c[mat_idx], &rs_c, &cs_c, alpha[gc_i], beta[gc_i],
            &rntm_g, &lcntx_l, &ops, DLP_F32);
#endif
        mat_idx += g_sz;
    }
err_hndl:;
    LPGEMM_STOP_LOGGER();
}

void
aocl_batch_gemm_u8s8s32obf16(const char*      order,
                             const char*      transa,
                             const char*      transb,
                             const md_t*      m,
                             const md_t*      n,
                             const md_t*      k,
                             const int32_t*   alpha,
                             const uint8_t**  a,
                             const md_t*      lda,
                             const int8_t**   b,
                             const md_t*      ldb,
                             const int32_t*   beta,
                             bfloat16**       c,
                             const md_t*      ldc,
                             const md_t       group_count,
                             const md_t*      group_size,
                             const char*      mem_format_a,
                             const char*      mem_format_b,
                             dlp_metadata_t** metadata)
{
    LPGEMM_START_LOGGER();
    BATCH_LPGEMM_WRITE_LOGGER(
        "u8s8s32obf16", order, transa, transb, group_count, group_size, m, n, k,
        alpha, lda, mem_format_a, ldb, mem_format_b, beta, ldc, metadata);

    // Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
    if (dlp_cpuid_is_avx512vnni_supported() == FALSE) {
        dlp_print_msg(" AVX512_VNNI ISA not supported by processor, "
                      "cannot perform u8s8s32obf16 gemm.",
                      __FILE__, __LINE__);
        for (md_t gc_i = 0; gc_i < group_count; gc_i++) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
        }
        goto err_hndl; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    md_t mat_idx = 0;

    // check for validity of params.
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;

    for (md_t gc_i = 0; gc_i < group_count; gc_i++) {

        DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_SUCCESS);

        md_t g_sz = group_size[gc_i];
        // check for validity of params.
        AOCL_BATCH_GEMM_CHECK("batch_u8s8s32obf16", order[gc_i], transa[gc_i],
                              transb[gc_i], gc_i, g_sz, m[gc_i], n[gc_i],
                              k[gc_i], a[gc_i], lda[gc_i], mem_format_a[gc_i],
                              b[gc_i], ldb[gc_i], mem_format_b[gc_i], c[gc_i],
                              ldc[gc_i], err_no);

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

        AOCL_MEMORY_TAG mtag_a;
        AOCL_MEMORY_TAG mtag_b;

        const uint8_t** a_local;
        const int8_t**  b_local;

        md_t m_local, n_local, k_local;

        // Convert post op struct to post op linked list format.
        lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];

        dlp_clsc_err_t err = lpgemm_translate_to_post_ops_list(
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
        // Column major isn't supported
        if (is_column_major == TRUE) {
            dlp_print_msg("Column major inputs not supported.", __FILE__,
                          __LINE__);
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
            goto err_hndl;
        }
        // Batch GEMM executes only for the Row major case.
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

        lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj(U8S8S32OS32);
        lpgemm_cntx_t  lcntx_l;
        // Create local copy, since each thread in a multi-instance setup
        // modifies the context object.
        lcntx_l = *lcntx_g;

        // Initialize DLP Plus kernel path.
        lcntx_l.dlp_kernel_hndl.kernel_base = NULL;
        // All the g_sz inputs in a given group will have the same matrix
        // dimensions/attributes. Therefore the DE and Jit generation in
        // DLP Plus can proceed with any 1 input from this group.
        dlp_init_and_get_kernel_hndl(
            DLP_KERNEL_U8S8S32OBF16, order[gc_i], mtag_a, mtag_b, m_local,
            n_local, k_local, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c,
            (void*)&alpha[gc_i], (void*)&beta[gc_i], post_op_list,
            lcntx_l.blksz.MR, lcntx_l.blksz.NR, lcntx_l.blksz.KC, DLP_BF16,
            &lcntx_l.dlp_kernel_hndl);

        // Invalid handle means that the jit kernel generation has failed. Do
        // not attempt to execute the kernel, and return an error instead.
        if (lcntx_l.dlp_kernel_hndl.kernel_base == NULL) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_INVALID_JIT_KERNEL);
            goto err_hndl;
        }

        // Create ops bundle for standard GEMM (post-ops only)
        lpgemm_ops_bundle_t ops = LPGEMM_OPS_BUNDLE_INIT_STANDARD(post_op_list);

#ifdef DLP_ENABLE_OPENMP
        batch_lpgemm_u8s8s32o32_openmp_thread_decorator(
            g_sz, &m_local, &n_local, &k_local, (const uint8_t**)a_local, &rs_a,
            &cs_a, &mtag_a, (const int8_t**)b_local, &rs_b, &cs_b, &mtag_b,
            (int32_t**)&c[mat_idx], &rs_c, &cs_c, alpha[gc_i], beta[gc_i],
            &rntm_g, &lcntx_l, &ops, DLP_BF16);
#else
        batch_lpgemm_u8s8s32o32_thread_decorator(
            g_sz, &m_local, &n_local, &k_local, (const uint8_t**)a_local, &rs_a,
            &cs_a, &mtag_a, (const int8_t**)b_local, &rs_b, &cs_b, &mtag_b,
            (int32_t**)&c[mat_idx], &rs_c, &cs_c, alpha[gc_i], beta[gc_i],
            &rntm_g, &lcntx_l, &ops, DLP_BF16);
#endif
        mat_idx += g_sz;
    }
err_hndl:;
    LPGEMM_STOP_LOGGER();
}

void
aocl_batch_gemm_u8s8s32ou8(const char*      order,
                           const char*      transa,
                           const char*      transb,
                           const md_t*      m,
                           const md_t*      n,
                           const md_t*      k,
                           const int32_t*   alpha,
                           const uint8_t**  a,
                           const md_t*      lda,
                           const int8_t**   b,
                           const md_t*      ldb,
                           const int32_t*   beta,
                           uint8_t**        c,
                           const md_t*      ldc,
                           const md_t       group_count,
                           const md_t*      group_size,
                           const char*      mem_format_a,
                           const char*      mem_format_b,
                           dlp_metadata_t** metadata)
{
    LPGEMM_START_LOGGER();
    BATCH_LPGEMM_WRITE_LOGGER("u8s8s32ou8", order, transa, transb, group_count,
                              group_size, m, n, k, alpha, lda, mem_format_a,
                              ldb, mem_format_b, beta, ldc, metadata);

    // Check if avx512_vnni ISA is supported, lpgemm matmul only works with it.
    if (dlp_cpuid_is_avx512vnni_supported() == FALSE) {
        dlp_print_msg(" AVX512_VNNI ISA not supported by processor, "
                      "cannot perform u8s8s32ou8 gemm.",
                      __FILE__, __LINE__);
        for (md_t gc_i = 0; gc_i < group_count; gc_i++) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
        }
        goto err_hndl; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    md_t mat_idx = 0;

    // check for validity of params.
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;

    for (md_t gc_i = 0; gc_i < group_count; gc_i++) {

        DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_SUCCESS);

        md_t g_sz = group_size[gc_i];
        // check for validity of params.
        AOCL_BATCH_GEMM_CHECK("batch_u8s8s32ou8", order[gc_i], transa[gc_i],
                              transb[gc_i], gc_i, g_sz, m[gc_i], n[gc_i],
                              k[gc_i], a[gc_i], lda[gc_i], mem_format_a[gc_i],
                              b[gc_i], ldb[gc_i], mem_format_b[gc_i], c[gc_i],
                              ldc[gc_i], err_no);

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

        AOCL_MEMORY_TAG mtag_a;
        AOCL_MEMORY_TAG mtag_b;

        const uint8_t** a_local;
        const int8_t**  b_local;

        md_t m_local, n_local, k_local;

        // Convert post op struct to post op linked list format.
        lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];

        dlp_clsc_err_t err = lpgemm_translate_to_post_ops_list(
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
        // Column major isn't supported
        if (is_column_major == TRUE) {
            dlp_print_msg("Column major inputs not supported.", __FILE__,
                          __LINE__);
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
            goto err_hndl;
        }
        // Batch GEMM executes only for the Row major case.
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
        // A matrix packing is not supported in row major case.
        // If A matrix is packed and not transposed, set to Unpack
        // and proceed with GEMM.
        if ((mtag_a == PACK) && (!dlp_is_trans(dlp_transa))) {
            mtag_a = UNPACKED;
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

        lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj(U8S8S32OS32);
        lpgemm_cntx_t  lcntx_l;
        // Create local copy, since each thread in a multi-instance setup
        // modifies the context object.
        lcntx_l = *lcntx_g;

        // Initialize DLP Plus kernel path.
        lcntx_l.dlp_kernel_hndl.kernel_base = NULL;
        // All the g_sz inputs in a given group will have the same matrix
        // dimensions/attributes. Therefore the DE and Jit generation in
        // DLP Plus can proceed with any 1 input from this group.
        dlp_init_and_get_kernel_hndl(
            DLP_KERNEL_U8S8S32OU8, order[gc_i], mtag_a, mtag_b, m_local,
            n_local, k_local, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c,
            (void*)&alpha[gc_i], (void*)&beta[gc_i], post_op_list,
            lcntx_l.blksz.MR, lcntx_l.blksz.NR, lcntx_l.blksz.KC, DLP_U8,
            &lcntx_l.dlp_kernel_hndl);

        // Invalid handle means that the jit kernel generation has failed. Do
        // not attempt to execute the kernel, and return an error instead.
        if (lcntx_l.dlp_kernel_hndl.kernel_base == NULL) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_INVALID_JIT_KERNEL);
            goto err_hndl;
        }

        // Create ops bundle for standard GEMM (post-ops only)
        lpgemm_ops_bundle_t ops = LPGEMM_OPS_BUNDLE_INIT_STANDARD(post_op_list);

#ifdef DLP_ENABLE_OPENMP
        batch_lpgemm_u8s8s32o32_openmp_thread_decorator(
            g_sz, &m_local, &n_local, &k_local, (const uint8_t**)a_local, &rs_a,
            &cs_a, &mtag_a, (const int8_t**)b_local, &rs_b, &cs_b, &mtag_b,
            (int32_t**)&c[mat_idx], &rs_c, &cs_c, alpha[gc_i], beta[gc_i],
            &rntm_g, &lcntx_l, &ops, DLP_U8);
#else
        batch_lpgemm_u8s8s32o32_thread_decorator(
            g_sz, &m_local, &n_local, &k_local, (const uint8_t**)a_local, &rs_a,
            &cs_a, &mtag_a, (const int8_t**)b_local, &rs_b, &cs_b, &mtag_b,
            (int32_t**)&c[mat_idx], &rs_c, &cs_c, alpha[gc_i], beta[gc_i],
            &rntm_g, &lcntx_l, &ops, DLP_U8);
#endif
        mat_idx += g_sz;
    }
err_hndl:;
    LPGEMM_STOP_LOGGER();
}
