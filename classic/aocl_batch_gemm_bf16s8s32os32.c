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

#include "aocl_dlp_gemm_check.h"
#include "classic/aocl_gemm_interface_apis.h"
#include "config/dlp_gemm_config.h"
#include "dlp_gemm_5loop_interface_apis.h"
#include "dlp_gemm_ops_bundle.h"
#include "dlp_gemm_post_ops.h"
#include "dlp_gemm_types.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "logging/dlp_gemm_logger.h"
#include "threading/dlp_gemm_thread_decor_openmp.h"

static void
aocl_batch_gemm_bf16s8s32_impl(const char*       order,
                               const char*       transa,
                               const char*       transb,
                               const md_t*       m,
                               const md_t*       n,
                               const md_t*       k,
                               const int32_t*    alpha,
                               const bfloat16**  a,
                               const md_t*       lda,
                               const int8_t**    b,
                               const md_t*       ldb,
                               const int32_t*    beta,
                               void**            c,
                               const md_t*       ldc,
                               const md_t        group_count,
                               const md_t*       group_size,
                               const char*       mem_format_a,
                               const char*       mem_format_b,
                               dlp_metadata_t**  metadata,
                               const char*       func_name,
                               kernel_datatype_t krnl_dtype,
                               DLP_TYPE          c_dtype)
{
    DLP_GEMM_START_LOGGER();
    BATCH_DLP_GEMM_WRITE_LOGGER(func_name, order, transa, transb, group_count,
                                group_size, m, n, k, alpha, lda, mem_format_a,
                                ldb, mem_format_b, beta, ldc, metadata);

    // Check if avx512_bf16 ISA is supported.
    if (dlp_cpuid_is_avx512bf16_supported() == FALSE) {
        dlp_print_msg(" AVX512_BF16 ISA not supported by processor, "
                      "cannot perform bf16s8s32 gemm.",
                      __FILE__, __LINE__);
        for (iter_t gc_i = 0; gc_i < group_count; gc_i++) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
        }
        goto err_hndl;
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    // offset to get subsequent matrix when group_count > 1
    md_t mat_idx = 0;

    // check for validity of params.
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;

    for (iter_t gc_i = 0; gc_i < group_count; gc_i++) {

        DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_SUCCESS);

        md_t g_sz = group_size[gc_i];

        // check for validity of params.
        AOCL_DLP_BATCH_GEMM_CHECK(
            func_name, order[gc_i], transa[gc_i], transb[gc_i], group_count,
            g_sz, m[gc_i], n[gc_i], k[gc_i], a[gc_i], lda[gc_i],
            mem_format_a[gc_i], b[gc_i], ldb[gc_i], mem_format_b[gc_i], c[gc_i],
            ldc[gc_i], err_no);

        if (err_no != DLP_CLSC_SUCCESS) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], err_no);
            goto err_hndl;
        }

        // Validate required quantization metadata.
        if (metadata[gc_i] == NULL || metadata[gc_i]->a_pre_quant == NULL
            || metadata[gc_i]->a_post_quant == NULL) {
            dlp_print_msg(
                "One or more required parameters (metadata, a_pre_quant, "
                "a_post_quant) are NULL. Exiting..",
                __FILE__, __LINE__);
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NULL_POINTER);
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

        const bfloat16** a_local;
        const int8_t**   b_local;
        md_t             m_local, n_local, k_local;

        // Convert post op struct to post op linked list format.
        dlp_gemm_post_op post_op_list[AOCL_DLP_MAX_POST_OPS + 1];

        // ADQUANTIZE first, then seq_vector ops via
        // translate_to_post_ops_list(post_op_list+1).
        dlp_clsc_err_t err = dlp_gemm_translate_adquantize_post_op(
            metadata[gc_i], post_op_list, (void*)((order + gc_i)), m[gc_i]);
        if (err != DLP_CLSC_SUCCESS) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], err);
            goto err_hndl;
        }
        err = dlp_gemm_translate_to_post_ops_list(
            metadata[gc_i], post_op_list + 1, (void*)c[gc_i],
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
        // Column major isn't supported for bf16s8s32.
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

        // A matrix reordered format not supported (quantization happens
        // on-the-fly).
        if (mtag_a == REORDERED) {
            dlp_print_msg(
                " Reordering of A matrix is not supported in row major case.",
                __FILE__, __LINE__);
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
            goto err_hndl;
        }

        // Treat UNPACKED A as PACK so that on-the-fly BF16->S8 quantization
        // of A can be implemented using the same packing path.
        if (mtag_a == UNPACKED) {
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

        // B matrix needs to be packed or reordered for kernel execution.
        if (mtag_b == UNPACKED) {
            mtag_b = PACK;
        }

        // Initialize a local runtime with global settings.
        dlp_rntm_t rntm_g;
        dlp_rntm_init_from_global(&rntm_g);

        dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(S8S8S32OS32);
        dlp_gemm_cntx_t  lcntx_l;
        // Create local copy, since each thread in a multi-instance setup
        // modifies the context object.
        lcntx_l = *lcntx_g;

        // Initialize DLP Plus kernel path.
        lcntx_l.dlp_kernel_hndl.kernel_base = NULL;
        dlp_init_and_get_kernel_hndl(
            krnl_dtype, order[gc_i], mtag_a, mtag_b, m_local, n_local, k_local,
            rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, (void*)&alpha[gc_i],
            (void*)&beta[gc_i], post_op_list, lcntx_l.blksz.MR,
            lcntx_l.blksz.NR, lcntx_l.blksz.KC, c_dtype,
            &lcntx_l.dlp_kernel_hndl);

        // Invalid handle means that the jit kernel generation has failed.
        if (lcntx_l.dlp_kernel_hndl.kernel_base == NULL) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_INVALID_JIT_KERNEL);
            goto err_hndl;
        }

        // Create ops bundle with quantization info for A matrix.
        dlp_gemm_ops_bundle_t ops = DLP_GEMM_OPS_BUNDLE_INIT_QUANT(
            metadata[gc_i]->a_pre_quant, post_op_list);

#ifdef DLP_ENABLE_OPENMP
        batch_dlp_gemm_bf16s8s32os32_openmp_thread_decorator(
            g_sz, &m_local, &n_local, &k_local, (const bfloat16**)a_local,
            &rs_a, &cs_a, &mtag_a, (const int8_t**)b_local, &rs_b, &cs_b,
            &mtag_b, (int32_t**)&c[mat_idx], &rs_c, &cs_c, alpha[gc_i],
            beta[gc_i], &rntm_g, &lcntx_l, &ops, c_dtype);
#else
        batch_dlp_gemm_bf16s8s32os32_thread_decorator(
            g_sz, &m_local, &n_local, &k_local, (const bfloat16**)a_local,
            &rs_a, &cs_a, &mtag_a, (const int8_t**)b_local, &rs_b, &cs_b,
            &mtag_b, (int32_t**)&c[mat_idx], &rs_c, &cs_c, alpha[gc_i],
            beta[gc_i], &rntm_g, &lcntx_l, &ops, c_dtype);
#endif
        mat_idx += g_sz;
    }
err_hndl:;
    DLP_GEMM_STOP_LOGGER();
}

void
aocl_batch_gemm_bf16s8s32os32(const char*      order,
                              const char*      transa,
                              const char*      transb,
                              const md_t*      m,
                              const md_t*      n,
                              const md_t*      k,
                              const int32_t*   alpha,
                              const bfloat16** a,
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
    aocl_batch_gemm_bf16s8s32_impl(
        order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, (void**)c,
        ldc, group_count, group_size, mem_format_a, mem_format_b, metadata,
        "batch_bf16s8s32os32", DLP_KERNEL_S8S8S32OS32, DLP_S32);
}

void
aocl_batch_gemm_bf16s8s32os8(const char*      order,
                             const char*      transa,
                             const char*      transb,
                             const md_t*      m,
                             const md_t*      n,
                             const md_t*      k,
                             const int32_t*   alpha,
                             const bfloat16** a,
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
    aocl_batch_gemm_bf16s8s32_impl(
        order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, (void**)c,
        ldc, group_count, group_size, mem_format_a, mem_format_b, metadata,
        "batch_bf16s8s32os8", DLP_KERNEL_S8S8S32OS8, DLP_S8);
}

void
aocl_batch_gemm_bf16s8s32of32(const char*      order,
                              const char*      transa,
                              const char*      transb,
                              const md_t*      m,
                              const md_t*      n,
                              const md_t*      k,
                              const int32_t*   alpha,
                              const bfloat16** a,
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
    aocl_batch_gemm_bf16s8s32_impl(
        order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, (void**)c,
        ldc, group_count, group_size, mem_format_a, mem_format_b, metadata,
        "batch_bf16s8s32of32", DLP_KERNEL_S8S8S32OF32, DLP_F32);
}

void
aocl_batch_gemm_bf16s8s32obf16(const char*      order,
                               const char*      transa,
                               const char*      transb,
                               const md_t*      m,
                               const md_t*      n,
                               const md_t*      k,
                               const int32_t*   alpha,
                               const bfloat16** a,
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
    aocl_batch_gemm_bf16s8s32_impl(
        order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, (void**)c,
        ldc, group_count, group_size, mem_format_a, mem_format_b, metadata,
        "batch_bf16s8s32obf16", DLP_KERNEL_S8S8S32OBF16, DLP_BF16);
}

void
aocl_batch_gemm_bf16s8s32ou8(const char*      order,
                             const char*      transa,
                             const char*      transb,
                             const md_t*      m,
                             const md_t*      n,
                             const md_t*      k,
                             const int32_t*   alpha,
                             const bfloat16** a,
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
    aocl_batch_gemm_bf16s8s32_impl(
        order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, (void**)c,
        ldc, group_count, group_size, mem_format_a, mem_format_b, metadata,
        "batch_bf16s8s32ou8", DLP_KERNEL_S8S8S32OU8, DLP_U8);
}
