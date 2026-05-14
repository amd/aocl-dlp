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
#include "classic/aocl_fp16_convert.h"
#include "classic/aocl_gemm_interface_apis.h"
#include "config/dlp_gemm_config.h"
#include "dlp_gemm_5loop_interface_apis.h"
#include "dlp_gemm_ops_bundle.h"
#include "dlp_gemm_post_ops.h"
#include "dlp_gemm_types.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "logging/dlp_gemm_logger.h"
#include "threading/dlp_gemm_thread_decor_openmp.h"

void
aocl_batch_gemm_f16f16f16of16(const char*      order,
                              const char*      transa,
                              const char*      transb,
                              const md_t*      m,
                              const md_t*      n,
                              const md_t*      k,
                              const float16*   alpha,
                              const float16**  a,
                              const md_t*      lda,
                              const float16**  b,
                              const md_t*      ldb,
                              const float16*   beta,
                              float16**        c,
                              const md_t*      ldc,
                              const md_t       group_count,
                              const md_t*      group_size,
                              const char*      mem_format_a,
                              const char*      mem_format_b,
                              dlp_metadata_t** metadata)
{
    DLP_GEMM_START_LOGGER();
    BATCH_DLP_GEMM_WRITE_LOGGER_FP16(
        "f16f16f16of16", order, transa, transb, group_count, group_size, m, n,
        k, alpha, lda, mem_format_a, ldb, mem_format_b, beta, ldc, metadata);

    // Check if AVX-512-FP16 ISA is supported, dlp_gemm fp16 matmul requires it.
    if (dlp_cpuid_is_avx512fp16_supported() == FALSE) {
        dlp_print_msg(" AVX-512-FP16 ISA not supported by processor, "
                      "cannot perform f16f16f16of16 gemm.",
                      __FILE__, __LINE__);
        for (iter_t gc_i = 0; gc_i < group_count; gc_i++) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
        }
        goto err_hndl;
    }

    // Set MC, NC, KC, NR, MR.
    dlp_init_global_cntx();

    // check for validity of params.
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;

    // offset to get subsequent matrix when group_count > 1
    md_t mat_idx = 0;

    for (iter_t gc_i = 0; gc_i < group_count; gc_i++) {

        DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_SUCCESS);

        // Group_size is used across
        md_t g_sz = group_size[gc_i];

        // check for validity of params.
        AOCL_DLP_BATCH_GEMM_CHECK(
            "batch_f16f16f16of16", order[gc_i], transa[gc_i], transb[gc_i],
            group_count, g_sz, m[gc_i], n[gc_i], k[gc_i], a[gc_i], lda[gc_i],
            mem_format_a[gc_i], b[gc_i], ldb[gc_i], mem_format_b[gc_i], c[gc_i],
            ldc[gc_i], err_no);

        if (err_no != DLP_CLSC_SUCCESS) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], err_no);
            goto err_hndl;
        }

        // Post-ops are not supported for FP16 GEMM.
        if ((metadata[gc_i] != NULL) && (metadata[gc_i]->seq_length > 0)) {
            dlp_print_msg(" Post-ops are not supported for f16f16f16of16 gemm.",
                          __FILE__, __LINE__);
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
            goto err_hndl;
        }

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

        const float16 **a_local, **b_local;
        md_t            m_local, n_local, k_local;

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

            if (dlp_is_trans(dlp_transb)) {
                mtag_a = PACK;
            }

            if (dlp_is_trans(dlp_transa)) {
                mtag_b = PACK;
            }

            // Force packing for unpacked column-major A.
            if (mtag_a == UNPACKED) {
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

            if (dlp_is_trans(dlp_transa)) {
                mtag_a = PACK;
            }

            // Transposed B must be packed for correct K-MAJOR access
            if (dlp_is_trans(dlp_transb) && (mtag_b != REORDERED)) {
                mtag_b = PACK;
            }

            // copy the values of m & n
            m_local = m[gc_i];
            n_local = n[gc_i];

            // copy the values of a & b pointers
            a_local = (a + mat_idx);
            b_local = (b + mat_idx);
        }
        // Copy the value of k.
        k_local = k[gc_i];

        rs_c = ldc[gc_i];
        cs_c = 1;

        // Initialize a local runtime with global settings if necessary. Note
        // that in the case that a runtime is passed in, we make a local copy.
        dlp_rntm_t rntm_g;
        dlp_rntm_init_from_global(&rntm_g);

        dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(F16F16F16OF16);
        dlp_gemm_cntx_t  lcntx_l;
        // Create local copy, since each thread in a multi-instance setup
        // modifies the context object.
        lcntx_l = *lcntx_g;

        float16 alpha_fp16 = alpha[gc_i];
        float16 beta_fp16  = beta[gc_i];

        // Initialize DLP Plus kernel path.
        lcntx_l.dlp_kernel_hndl.kernel_base = NULL;
        // All the g_sz inputs in a given group will have the same matrix
        // dimensions/attributes. Therefore the DE and Jit generation in
        // DLP Plus can proceed with any 1 input from this group.
        dlp_init_and_get_kernel_hndl(
            DLP_KERNEL_F16F16F16OF16, order[gc_i], mtag_a, mtag_b, m_local,
            n_local, k_local, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c,
            (void*)&alpha_fp16, (void*)&beta_fp16, post_op_list,
            lcntx_l.blksz.MR, lcntx_l.blksz.NR, lcntx_l.blksz.KC, DLP_F16,
            &lcntx_l.dlp_kernel_hndl);

        // FP16 is JIT-only (no intrinsic fallback), so check if JIT succeeded
        if (lcntx_l.dlp_kernel_hndl.kernel_base == NULL) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_INVALID_JIT_KERNEL);
            goto err_hndl;
        }

        // Create ops bundle for standard GEMM (post-ops only)
        dlp_gemm_ops_bundle_t ops =
            DLP_GEMM_OPS_BUNDLE_INIT_STANDARD(post_op_list);

#ifdef DLP_ENABLE_OPENMP
        batch_dlp_gemm_f16f16f16of16_openmp_thread_decorator(
            g_sz, &m_local, &n_local, &k_local, (const float16**)a_local, &rs_a,
            &cs_a, &mtag_a, (const float16**)b_local, &rs_b, &cs_b, &mtag_b,
            &c[mat_idx], &rs_c, &cs_c, alpha_fp16, beta_fp16, &rntm_g, &lcntx_l,
            &ops, DLP_F16);

#else
        batch_dlp_gemm_f16f16f16of16_thread_decorator(
            g_sz, &m_local, &n_local, &k_local, (const float16**)a_local, &rs_a,
            &cs_a, &mtag_a, (const float16**)b_local, &rs_b, &cs_b, &mtag_b,
            &c[mat_idx], &rs_c, &cs_c, alpha_fp16, beta_fp16, &rntm_g, &lcntx_l,
            &ops, DLP_F16);
#endif
        // Increment the matrix index to get the next matrix in the group.
        mat_idx += g_sz;
    }
err_hndl:;
    DLP_GEMM_STOP_LOGGER();
}

/*
 * Batch GEMM: FP16×FP16 → F32 (FP16 native accumulation, F32 output).
 *
 * Mirrors aocl_batch_gemm_f16f16f16of16 but with float** for the C
 * matrices. Shares the of16 context, decision engine, threading
 * decorator, and JIT generator -- the only dispatch difference is
 * c_downscale = DLP_F32 passed to dlp_init_and_get_kernel_hndl,
 * which causes the JIT to widen beta to F32 and store C at full
 * precision without any scratch buffer.
 *
 * Alpha and beta arrive as float16 (matching the single GEMM API).
 * The JIT kernel uses vpbroadcastw (16-bit broadcast) to load them.
 */
void
aocl_batch_gemm_f16f16f16of32(const char*      order,
                              const char*      transa,
                              const char*      transb,
                              const md_t*      m,
                              const md_t*      n,
                              const md_t*      k,
                              const float16*   alpha,
                              const float16**  a,
                              const md_t*      lda,
                              const float16**  b,
                              const md_t*      ldb,
                              const float16*   beta,
                              float**          c,
                              const md_t*      ldc,
                              const md_t       group_count,
                              const md_t*      group_size,
                              const char*      mem_format_a,
                              const char*      mem_format_b,
                              dlp_metadata_t** metadata)
{
    DLP_GEMM_START_LOGGER();
    BATCH_DLP_GEMM_WRITE_LOGGER_FP16(
        "f16f16f16of32", order, transa, transb, group_count, group_size, m, n,
        k, alpha, lda, mem_format_a, ldb, mem_format_b, beta, ldc, metadata);

    if (dlp_cpuid_is_avx512fp16_supported() == FALSE) {
        dlp_print_msg(" AVX-512-FP16 ISA not supported by processor, "
                      "cannot perform f16f16f16of32 batch gemm.",
                      __FILE__, __LINE__);
        for (iter_t gc_i = 0; gc_i < group_count; gc_i++) {
            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
        }
        goto err_hndl_of32;
    }

    dlp_init_global_cntx();

    {
        dlp_clsc_err_t err_no  = DLP_CLSC_SUCCESS;
        md_t           mat_idx = 0;

        for (iter_t gc_i = 0; gc_i < group_count; gc_i++) {

            DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_SUCCESS);

            md_t g_sz = group_size[gc_i];

            AOCL_DLP_BATCH_GEMM_CHECK(
                "batch_f16f16f16of32", order[gc_i], transa[gc_i], transb[gc_i],
                group_count, g_sz, m[gc_i], n[gc_i], k[gc_i], a[gc_i],
                lda[gc_i], mem_format_a[gc_i], b[gc_i], ldb[gc_i],
                mem_format_b[gc_i], c[gc_i], ldc[gc_i], err_no);

            if (err_no != DLP_CLSC_SUCCESS) {
                DLP_METADATA_SET_ERROR(metadata[gc_i], err_no);
                goto err_hndl_of32;
            }

            if ((metadata[gc_i] != NULL) && (metadata[gc_i]->seq_length > 0)) {
                dlp_print_msg(
                    " Post-ops are not supported for f16f16f16of32 batch gemm.",
                    __FILE__, __LINE__);
                DLP_METADATA_SET_ERROR(metadata[gc_i], DLP_CLSC_NOT_SUPPORTED);
                goto err_hndl_of32;
            }

            dlp_trans_t dlp_transa;
            dlp_trans_t dlp_transb;

            md_t rs_a, cs_a;
            md_t rs_b, cs_b;
            md_t rs_c, cs_c;

            AOCL_DLP_MEMORY_TAG mtag_a;
            AOCL_DLP_MEMORY_TAG mtag_b;

            const float16 **a_local, **b_local;
            md_t            m_local, n_local, k_local;

            dlp_gemm_post_op post_op_list[AOCL_DLP_MAX_POST_OPS];

            dlp_clsc_err_t err = dlp_gemm_translate_to_post_ops_list(
                metadata[gc_i], post_op_list, (void*)c[gc_i],
                (void*)((order + gc_i)), m[gc_i], n[gc_i]);

            if (err != DLP_CLSC_SUCCESS) {
                DLP_METADATA_SET_ERROR(metadata[gc_i], err);
                goto err_hndl_of32;
            }

            dlp_param_map_netlib_to_dlp_trans(transa[gc_i], &dlp_transa);
            dlp_param_map_netlib_to_dlp_trans(transb[gc_i], &dlp_transb);

            bool is_column_major =
                ((order[gc_i] == 'c') || (order[gc_i] == 'C'));

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

                if (((mtag_b == REORDERED) || (mtag_a == REORDERED))) {
                    dlp_print_msg(" Reordering of column major matrices is not "
                                  "supported.",
                                  __FILE__, __LINE__);
                    DLP_METADATA_SET_ERROR(metadata[gc_i],
                                           DLP_CLSC_NOT_SUPPORTED);
                    goto err_hndl_of32;
                }

                if (dlp_is_trans(dlp_transb)) {
                    mtag_a = PACK;
                }

                if (dlp_is_trans(dlp_transa)) {
                    mtag_b = PACK;
                }

                if (mtag_a == UNPACKED) {
                    mtag_a = PACK;
                }

                m_local = n[gc_i];
                n_local = m[gc_i];

                a_local = (b + mat_idx);
                b_local = (a + mat_idx);
            } else {
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

                if (mtag_a == REORDERED) {
                    dlp_print_msg(
                        " Reordering of A matrix is not supported in row "
                        "major case.",
                        __FILE__, __LINE__);
                    DLP_METADATA_SET_ERROR(metadata[gc_i],
                                           DLP_CLSC_NOT_SUPPORTED);
                    goto err_hndl_of32;
                }

                if (dlp_is_trans(dlp_transa)) {
                    mtag_a = PACK;
                }

                if (dlp_is_trans(dlp_transb) && (mtag_b != REORDERED)) {
                    mtag_b = PACK;
                }

                m_local = m[gc_i];
                n_local = n[gc_i];

                a_local = (a + mat_idx);
                b_local = (b + mat_idx);
            }
            k_local = k[gc_i];

            rs_c = ldc[gc_i];
            cs_c = 1;

            dlp_rntm_t rntm_g;
            dlp_rntm_init_from_global(&rntm_g);

            dlp_gemm_cntx_t* lcntx_g =
                dlp_gemm_get_global_cntx_obj(F16F16F16OF16);
            dlp_gemm_cntx_t lcntx_l;
            lcntx_l = *lcntx_g;

            float16 alpha_fp16 = alpha[gc_i];
            float16 beta_fp16  = beta[gc_i];

            lcntx_l.dlp_kernel_hndl.kernel_base = NULL;

            dlp_init_and_get_kernel_hndl(
                DLP_KERNEL_F16F16F16OF32, order[gc_i], mtag_a, mtag_b, m_local,
                n_local, k_local, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c,
                (void*)&alpha_fp16, (void*)&beta_fp16, post_op_list,
                lcntx_l.blksz.MR, lcntx_l.blksz.NR, lcntx_l.blksz.KC, DLP_F32,
                &lcntx_l.dlp_kernel_hndl);

            if (lcntx_l.dlp_kernel_hndl.kernel_base == NULL) {
                dlp_print_msg(" FP16 JIT kernel generation failed for "
                              "f16f16f16of32 batch gemm.",
                              __FILE__, __LINE__);
                DLP_METADATA_SET_ERROR(metadata[gc_i],
                                       DLP_CLSC_INVALID_JIT_KERNEL);
                goto err_hndl_of32;
            }

            dlp_gemm_ops_bundle_t ops =
                DLP_GEMM_OPS_BUNDLE_INIT_STANDARD(post_op_list);

#ifdef DLP_ENABLE_OPENMP
            batch_dlp_gemm_f16f16f16of16_openmp_thread_decorator(
                g_sz, &m_local, &n_local, &k_local, (const float16**)a_local,
                &rs_a, &cs_a, &mtag_a, (const float16**)b_local, &rs_b, &cs_b,
                &mtag_b, (float16**)&c[mat_idx], &rs_c, &cs_c, alpha_fp16,
                beta_fp16, &rntm_g, &lcntx_l, &ops, DLP_F32);
#else
            batch_dlp_gemm_f16f16f16of16_thread_decorator(
                g_sz, &m_local, &n_local, &k_local, (const float16**)a_local,
                &rs_a, &cs_a, &mtag_a, (const float16**)b_local, &rs_b, &cs_b,
                &mtag_b, (float16**)&c[mat_idx], &rs_c, &cs_c, alpha_fp16,
                beta_fp16, &rntm_g, &lcntx_l, &ops, DLP_F32);
#endif
            mat_idx += g_sz;
        }
    }
err_hndl_of32:;
    DLP_GEMM_STOP_LOGGER();
}
