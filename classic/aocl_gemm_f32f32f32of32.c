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

#include "aocl_gemm_check.h"
#include "classic/aocl_gemm_interface_apis.h"
#include "classic/aocl_lib_interface_apis.h"
#include "classic/dlp_errors.h"
#include "config/lpgemm_config.h"
#include "gemm_utils/lpgemm_utils.h"
#include "logging/lpgemm_logger.h"
#include "lpgemm_5loop_interface_apis.h"
#include "lpgemm_ops_bundle.h"
#include "lpgemm_post_ops.h"
#include "lpgemm_types.h"
#include "runtime/dlp_runtime.h"
#include "threading/lpgemm_thread_decor_openmp.h"

static inline bool
is_tiny_input_f32(md_t m, md_t n, md_t k, lpgemm_cntx_t* lcntx)
{
    // if k == 1, then we can use the single-threaded tiny input kernel.
    // Multi-threading is not beneficial for k = 1.
    if (k == 1) {
        return TRUE;
    }

    const md_t NC = lcntx->blksz.NC;
    const md_t MC = lcntx->blksz.MC;
    const md_t KC = lcntx->blksz.KC;
    const md_t MR = lcntx->blksz.MR;
    const md_t NR = lcntx->blksz.NR;

    md_t       mnk           = m * n * k;
    md_t       mk            = m * k;
    const md_t mnk_magic_num = 12 * 64 * 496;
    const md_t mk_thresh     = 12000;
    const md_t m_thresh      = 5 * MR;
    const md_t n_thresh      = 2 * NR;
    const md_t k_thresh      = 480;

    // Need to explicitly check for MC, NC boundaries for safety.
    if (((k < KC) && (m <= MC) && (n < NC))
        && (((m <= m_thresh) && (n <= n_thresh) && (k <= k_thresh))
            || ((mnk < mnk_magic_num && m != 1 && mk < mk_thresh)))) {
        return TRUE;
    }

    return FALSE;
}

void
aocl_gemm_f32f32f32of32(const char      order,
                        const char      transa,
                        const char      transb,
                        const md_t      m,
                        const md_t      n,
                        const md_t      k,
                        const float     alpha,
                        const float*    a,
                        const md_t      lda,
                        const char      mem_format_a,
                        const float*    b,
                        const md_t      ldb,
                        const char      mem_format_b,
                        const float     beta,
                        float*          c,
                        const md_t      ldc,
                        dlp_metadata_t* metadata)
{
    LPGEMM_START_LOGGER();
    LPGEMM_WRITE_LOGGER("f32f32f32of32", order, transa, transb, m, n, k,
                        ((float)alpha), lda, mem_format_a, ldb, mem_format_b,
                        ((float)beta), ldc, metadata);

    DLP_METADATA_SET_ERROR(metadata,
                           DLP_CLSC_SUCCESS); // Set default error to success.

    // Check if AVX2 ISA is supported, lpgemm fp32 matmul only works with it.
    if (dlp_cpuid_is_avx2fma3_supported() == FALSE) {
        dlp_print_msg(" AVX2 ISA not supported by processor, "
                      "cannot perform f32f32f32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // Initialize lpgemm context.
    aocl_lpgemm_init_global_cntx();

    // check for validity of params.
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_GEMM_CHECK("f32f32f32of32", order, transa, transb, m, n, k, a, lda,
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

    bool is_row_major    = ((order == 'r') || (order == 'R'));
    bool is_column_major = ((order == 'c') || (order == 'C'));

    // In case the inputs are column major, the matrices are swapped (A -> B',
    // B -> A'), and C' is computed instead of C. The strides are set so that
    // in the end post the swap, the strides correspond to a row major kernel.
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

    AOCL_MEMORY_TAG mtag_a;
    AOCL_MEMORY_TAG mtag_b;

    dlp_param_map_char_to_lpmtag(mem_format_a, &mtag_a);
    dlp_param_map_char_to_lpmtag(mem_format_b, &mtag_b);

    // Reordered A not supported now.
    if ((is_row_major == TRUE) && (mtag_a == REORDERED)) {
        dlp_print_msg(" Reordering of A matrix is not supported.", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // Inputs swapped in column major, A becomes B from kernel point of view.
    else if ((is_column_major == TRUE)
             && ((mtag_b == REORDERED) || (mtag_a == REORDERED))) {
        dlp_print_msg(" Reordering of column major matrices is not supported.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // Temporary variables to store/transform the input for kernel generation
    // and execution.
    md_t            m_use = m, n_use = n, k_use = k;
    const float*    a_use    = a;
    const float*    b_use    = b;
    float*          c_use    = c;
    md_t            rs_a_use = rs_a, cs_a_use = cs_a;
    md_t            rs_b_use = rs_b, cs_b_use = cs_b;
    md_t            rs_c_use = rs_c, cs_c_use = cs_c;
    AOCL_MEMORY_TAG mtag_a_use = mtag_a;
    AOCL_MEMORY_TAG mtag_b_use = mtag_b;
    char            order_use  = order;

    // Convert post op struct to post op linked list format.
    lpgemm_post_op post_op_list[AOCL_MAX_POST_OPS];
    dlp_clsc_err_t err;

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    lpgemm_cntx_t* lcntx_g = lpgemm_get_global_cntx_obj(F32F32F32OF32);
    lpgemm_cntx_t  lcntx_l;
    // Create local copy, since each thread in a multi-instance setup
    // modified the context object.
    lcntx_l = *lcntx_g;

    // By this point, global_lpgemm_enable_arch is set to the correct
    // architecture.

    // Induce operation transpose and/or swapped strides based on the input.
    // NOTE :
    // This logic is primarily used to decide what JIT kernels are to be
    // generated. Any logical induce that we perform(swapping strides/matrices)
    // would reflect in the DE when generating the kernel. Since the 5-loop
    // algorithm(framework) is detached from the CPP layer, it is expected
    // that the induced ordering is maintained when calling the 5-loop, which
    // would internally call the execute handler, thus mapping to the correct
    // kernel.

    if (!((m == 1) || (n == 1))) {

        bool avx512_on_non_zen3 =
            dlp_cpuid_is_avx512_supported()
            && (lpgemm_get_enabled_arch() != DLP_ARCH_ZEN3);
        bool has_valid_metadata =
            (metadata != NULL) && (metadata->seq_length != 0);

        // Handling row-major storage.
        if (is_row_major == TRUE) {
            if ((mtag_b != REORDERED)) {
                // If B is not reordered, and if both A and B are transposed
                // instead of packing both the matrices to make them row-major,
                // induce a transpose and store C in col-major format.
                // After inducing transpose, A, B matrices are row-major, and C
                // matrix is column-major. This optimization are currently not
                // supported in post-ops of avx2 kernels.

                if ((avx512_on_non_zen3 || !has_valid_metadata)
                    && dlp_is_trans(dlp_transb) && dlp_is_trans(dlp_transa)) {
                    // induce transpose here
                    m_use      = n;
                    n_use      = m;
                    a_use      = b;
                    rs_a_use   = cs_b;
                    cs_a_use   = rs_b;
                    b_use      = a;
                    rs_b_use   = cs_a;
                    cs_b_use   = rs_a;
                    rs_c_use   = cs_c;
                    cs_c_use   = rs_c;
                    mtag_a_use = mtag_b;
                    mtag_b_use = mtag_a;
                    order_use  = 'c';
                } else {
                    // If any of the above conditions fail, then we need to pack
                    // the matrices.
                    if (dlp_is_trans(dlp_transb)) {
                        // set mtag_b to PACK
                        mtag_b     = PACK;
                        mtag_b_use = PACK;
                    }
                    if (dlp_is_trans(dlp_transa)) {
                        // set mtag_a to PACK
                        mtag_a     = PACK;
                        mtag_a_use = PACK;
                    }
                }
            } else if (dlp_is_trans(dlp_transa)) {
                mtag_a     = PACK;
                mtag_a_use = PACK;
            }

        }
        // Handling column-major storage.
        else {

            // If both A and B are transposed, don't induce transpose here.
            // Instead, store C matrix in col-major format.
            // A and B matrices are row-major, and C matrix is column-major.
            // scatter-gather approach is not supported with avx2 instructions.

            if ((avx512_on_non_zen3 || !has_valid_metadata)
                && dlp_is_trans(dlp_transa) && (dlp_is_trans(dlp_transb))) {
                // don't induce transpose here
                rs_a_use   = cs_a;
                cs_a_use   = rs_a;
                rs_b_use   = cs_b;
                cs_b_use   = rs_b;
                rs_c_use   = cs_c;
                cs_c_use   = rs_c;
                mtag_a_use = mtag_a;
                mtag_b_use = mtag_b;
                order_use  = 'r';
            } else {
                // Inputs swapped in column major, A becomes B from kernel point
                // of view.
                if (dlp_is_trans(dlp_transa) && (mtag_a == UNPACKED)) {
                    mtag_a = PACK;
                }

                // Inputs swapped in column major, A becomes B from kernel point
                // of view.
                if (dlp_is_trans(dlp_transb)) {
                    mtag_b = PACK;
                }

                m_use      = n;
                n_use      = m;
                a_use      = b;
                rs_a_use   = rs_b;
                cs_a_use   = cs_b;
                b_use      = a;
                rs_b_use   = rs_a;
                cs_b_use   = cs_a;
                rs_c_use   = rs_c;
                cs_c_use   = cs_c;
                mtag_a_use = mtag_b;
                mtag_b_use = mtag_a;
                order_use  = order;
            }
        }

        err = lpgemm_translate_to_post_ops_list(
            metadata, post_op_list, (void*)c_use, (void*)(&order_use), m,
            n); // To use order_reuse in future, when we support post-ops on
                // row-major with transpose toggled.

        if (err != DLP_CLSC_SUCCESS) {
            DLP_METADATA_SET_ERROR(metadata, err);
            goto err_hndl;
        }

    } else {
        // Create the post-ops  linked list
        err = lpgemm_translate_to_post_ops_list(
            metadata, post_op_list, (void*)c_use, (void*)(&order_use), m,
            n); // To use order_reuse in future, when we support post-ops on
                // row-major with transpose toggled.

        if (err != DLP_CLSC_SUCCESS) {
            DLP_METADATA_SET_ERROR(metadata, err);
            goto err_hndl;
        }

        dlp_trans_t dlp_transa_tmp = dlp_transa;
        dlp_trans_t dlp_transb_tmp = dlp_transb;

        if (is_row_major == TRUE) {
            if ((mtag_b != REORDERED) && (dlp_is_trans(dlp_transb))) {
                // set mtag_b to PACK
                mtag_b     = PACK;
                mtag_b_use = PACK;
            }
            if (dlp_is_trans(dlp_transa)) {
                // set mtag_a to PACK
                mtag_a     = PACK;
                mtag_a_use = PACK;
            }
        } else {
            // Inputs swapped in column major, A becomes B from kernel point
            // of view.
            if (dlp_is_trans(dlp_transa) && (mtag_a == UNPACKED)) {
                mtag_a = PACK;
            }

            // Inputs swapped in column major, A becomes B from kernel point
            // of view.
            if (dlp_is_trans(dlp_transb)) {
                mtag_b = PACK;
            }

            // Induce operation transpose based on whether the incoming
            // storage is row-major or column-major.
            m_use      = n;
            n_use      = m;
            a_use      = b;
            rs_a_use   = rs_b;
            cs_a_use   = cs_b;
            b_use      = a;
            rs_b_use   = rs_a;
            cs_b_use   = cs_a;
            rs_c_use   = rs_c;
            cs_c_use   = cs_c;
            mtag_a_use = mtag_b;
            mtag_b_use = mtag_a;

            dlp_transa_tmp = dlp_transb;
            dlp_transb_tmp = dlp_transa;
        }

        // A set of temporary variables to hold the induced transpose
        // based on storage scheme. These are used for further optimizations
        // like avoiding packing in the required cases.

        md_t            m_tmp = m_use, n_tmp = n_use, k_tmp = k_use;
        const float*    a_tmp    = a_use;
        const float*    b_tmp    = b_use;
        float*          c_tmp    = c_use;
        md_t            rs_a_tmp = rs_a_use, cs_a_tmp = cs_a_use;
        md_t            rs_b_tmp = rs_b_use, cs_b_tmp = cs_b_use;
        md_t            rs_c_tmp = rs_c_use, cs_c_tmp = cs_c_use;
        AOCL_MEMORY_TAG mtag_a_tmp = mtag_a_use;
        AOCL_MEMORY_TAG mtag_b_tmp = mtag_b_use;

        // This optimization is currently enabled only when post-ops are
        // disabled.
        if (post_op_list[0].op_code == POST_OPS_DISABLE) {
            // For GEMV_M1
            if ((m_tmp == 1) && dlp_is_trans(dlp_transb_tmp)
                && (mtag_b_tmp != REORDERED)) {
                // NOTE : We will reorder the inputs such that we use the
                // GEMV_N1 kernel instead of GEMV_M1, in order to avoid
                // packing of B matrix(if not already reordered). The
                // GEMV_N1 kernels support both unit/non-unit strided
                // loads/stores for C vector. Thus, we would be packing the
                // input vector alone(if needed).
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
                mtag_b_use = mtag_a_tmp;
            }
            // For GEMV_N1
            // The library does not support reorder of A or column-major
            // matrices, thereby not needing an explicit check.
            else if ((n_tmp == 1) && dlp_is_trans(dlp_transa_tmp)
                     && (rs_c_tmp == 1)) {
                // NOTE : We will reorder the inputs such that we use the
                // GEMV_M1 kernel instead of GEMV_N1, in order to avoid
                // packing of A matrix. The GEMV_M1 kernels(both classic and
                // JIT) support only unit-strided C vector(row-stored).
                // Thus, we need an explicit check for that.
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
                mtag_a_use = mtag_b_tmp;
                mtag_b_use = UNPACKED;
            }
        }
    }

    // Initialize DLP Plus kernel path.
    lcntx_l.dlp_kernel_hndl.kernel_base = NULL;
    dlp_init_and_get_kernel_hndl(
        DLP_KERNEL_F32F32F32OF32, order, mtag_a_use, mtag_b_use, m_use, n_use,
        k_use, rs_a_use, cs_a_use, rs_b_use, cs_b_use, rs_c_use, cs_c_use,
        (void*)&alpha, (void*)&beta, post_op_list, lcntx_l.blksz.MR,
        lcntx_l.blksz.NR, lcntx_l.blksz.KC, DLP_F32, &lcntx_l.dlp_kernel_hndl);

    // if kernel_hndl is NULL and cs_c!=1, then we need to induce transpose
    // and set mtag of both matrices to PACK.
    if ((lcntx_l.dlp_kernel_hndl.kernel_base == NULL) && (cs_c_use != 1)) {
        if (order == 'r') {
            // copy the original values
            m_use     = m;
            n_use     = n;
            a_use     = a;
            b_use     = b;
            rs_a_use  = rs_a;
            cs_a_use  = cs_a;
            rs_b_use  = rs_b;
            cs_b_use  = cs_b;
            rs_c_use  = rs_c;
            cs_c_use  = cs_c;
            order_use = order;
        } else { // order == 'c'
            // induce transpose
            m_use     = n;
            n_use     = m;
            a_use     = b;
            b_use     = a;
            rs_a_use  = rs_b;
            cs_a_use  = cs_b;
            rs_b_use  = rs_a;
            cs_b_use  = cs_a;
            rs_c_use  = rs_c;
            cs_c_use  = cs_c;
            order_use = order;
        }

        // setting both the matrices to PACK
        mtag_a_use = PACK;
        mtag_b_use = PACK;

        // re-translate the post-ops list based on the new values
        err = lpgemm_translate_to_post_ops_list(
            metadata, post_op_list, (void*)c_use, (void*)(&order_use), m, n);

        if (err != DLP_CLSC_SUCCESS) {
            DLP_METADATA_SET_ERROR(metadata, err);
            goto err_hndl;
        }
    }

    if (is_single_thread(&rntm_g) == TRUE) {
        if (is_tiny_input_f32(m_use, n_use, k_use, &lcntx_l) == TRUE) {
            lpgemm_rowvar_tiny_f32f32f32of32(
                m_use, n_use, k_use, a_use, rs_a_use, cs_a_use, mtag_a_use,
                b_use, rs_b_use, cs_b_use, mtag_b_use, c_use, rs_c_use,
                cs_c_use, alpha, beta, &lcntx_l, post_op_list, DLP_F32);

            goto err_hndl;
        }
    }

    // Create ops bundle for standard GEMM (post-ops only)
    lpgemm_ops_bundle_t ops = LPGEMM_OPS_BUNDLE_INIT_STANDARD(post_op_list);

#ifdef DLP_ENABLE_OPENMP
    lpgemm_f32f32f32of32_openmp_thread_decorator(
        m_use, n_use, k_use, a_use, rs_a_use, cs_a_use, mtag_a_use, b_use,
        rs_b_use, cs_b_use, mtag_b_use, c_use, rs_c_use, cs_c_use, alpha, beta,
        &rntm_g, &lcntx_l, &ops, DLP_F32);
#else
    lpgemm_f32f32f32of32_thread_decorator(
        m_use, n_use, k_use, a_use, rs_a_use, cs_a_use, mtag_a_use, b_use,
        rs_b_use, cs_b_use, mtag_b_use, c_use, rs_c_use, cs_c_use, alpha, beta,
        &rntm_g, &lcntx_l, &ops, DLP_F32);
#endif

err_hndl:;
    LPGEMM_STOP_LOGGER();
}
