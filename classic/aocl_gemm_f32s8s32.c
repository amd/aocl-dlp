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

/**
 * aocl_gemm_f32s8s32o<bf16/s32/f32/s8/u8>
 *
 * Performs General Matrix Multiplication (GEMM) with mixed precision:
 *   C = alpha * op(A) * op(B) + beta * C
 *
 * Where:
 *   - Matrix A is in float (F32) format and is quantized to S8 on-the-fly
 * during computation.
 *   - Matrix B is in S8 (int8) format and should be pre-quantized.
 *   - The result matrix C can be in BF16, S32, F32, S8, or U8 format depending
 * on the function variant used.
 *   - Dequantization of results is handled during kernel execution as part of
 * the JIT kernel's post-processing steps.
 *   - Intermediate accumulation is performed in S32 precision, which is
 * subsequently downscaled to the output type through post-processing
 * operations.
 *
 * Function Variants (by output type):
 *   - aocl_gemm_f32s8s32obf16: Output in BF16 (bfloat16)
 *   - aocl_gemm_f32s8s32os32:  Output in S32 (int32_t)
 *   - aocl_gemm_f32s8s32of32:  Output in F32 (float)
 *   - aocl_gemm_f32s8s32os8:   Output in S8 (int8_t)
 *   - aocl_gemm_f32s8s32ou8:   Output in U8 (uint8_t)
 *
 * Hardware Requirements:
 *   - AVX512_VNNI: Required for S8xS8->S32 dot product instructions
 *   Function will fail gracefully if these ISA extensions are not available.
 *
 * Implementation Details:
 *   - Uses cache-optimized 5-loop blocked GEMM algorithm (JC-PC-IC-JR-IR)
 *   - A matrix quantization: float -> S8 per-tensor or per-row
 *   - B matrix must be packed/reordered for optimal kernel performance
 *   - Supports multi-threaded execution via OpenMP
 *   - JIT kernel generation for fused post-ops (dequantization, bias,
 * activation)
 *
 * Supported configurations:
 *   - Matrix layout: Row-major only (order='R')
 *   - A transpose: None (transa='N' only)
 *   - B transpose: None or transpose (transb='N' or 'T')
 *   - A memory format: Unpacked only (on-the-fly quantization)
 *   - B memory format: Packed, Reordered, or Unpacked (auto-packed at runtime)
 *
 * Quantization requirements (via metadata):
 *   - Pre-quantization (F32 -> S8): Requires a_pre_quant_sf scale factors
 *     Optional: zero_point for asymmetric quantization
 *   - Post-quantization (downscaling): Requires a_post_quant_sf scale factors
 *   - Supports per-tensor (scale_factor_len = 1) or per-row (scale_factor_len =
 * m)
 *   - See dlp_quant_op structure for complete quantization parameter details
 *
 * Post-Operations Support:
 *   - Dequantization (pre-quantization, post-quantization)
 *   - Post-ops are JIT generated
 *
 * Example Usage:
 * @code
 *   // Per-tensor symmetric quantization (F32 output)
 *   dlp_sf_t a_pre_quant_scl = {&a_pre_quant_sf, 1, DLP_F32};
 *   dlp_sf_t a_post_quant_scl = {&a_post_quant_sf, 1, DLP_F32};
 *   dlp_quant_op a_pre_quant = {0, DLP_F32, DLP_S8, &a_pre_quant_scl, NULL,
 * true}; dlp_quant_op a_post_quant = {0, DLP_F32, DLP_S8, &a_post_quant_scl,
 * NULL, true}; dlp_metadata_t metadata = {&a_pre_quant, &a_post_quant};
 *
 *   aocl_gemm_f32s8s32obf16('R', 'N', 'N', m, n, k, 1,
 *                            a, lda, 'N', b, ldb, 'N', 0, c_bf16, ldc,
 * &metadata);
 *
 *   // Per-row asymmetric quantization (use scale_factor_len = m, add
 * zero_point) float *sf_row = ..., *inv_sf_row = ..., *zp_row = ...; dlp_sf_t
 * pre_scl = {sf_row, m, DLP_F32}; dlp_sf_t post_scl = {inv_sf_row, m, DLP_F32};
 *   dlp_zp_t zp = {zp_row, m, DLP_F32};
 *   dlp_quant_op pre_quant = {0, DLP_F32, DLP_S8, &pre_scl, &zp, false};
 *   dlp_quant_op post_quant = {0, DLP_F32, DLP_S8, &post_scl, &zp, false};
 *   dlp_metadata_t meta = {&pre_quant, &post_quant, NULL, NULL};
 * @endcode
 *
 * See Also:
 *   - dlp_metadata_t: Metadata structure definition
 *   - dlp_quant_op: Quantization operation structure
 *   - dlp_sf_t: Scale factor structure
 *   - dlp_zp_t: Zero point structure
 */

/* =========================================================================
 * Internal Implementation Function
 * =========================================================================
 * aocl_gemm_f32s8s32_impl
 * Internal implementation function that provides the core GEMM logic for all
 * f32s8s32 variants.
 *
 * See aocl_gemm_f32s8s32obf16 documentation for parameter descriptions.
 * @param func_name Function name for logging and error reporting
 * @param krnl_dtype Kernel datatype selector for JIT kernel generation
 * @param c_dtype Output matrix C datatype (DLP_BF16, DLP_S32, DLP_F32, DLP_S8,
 * DLP_U8)
 */
static void
aocl_gemm_f32s8s32_impl(const char        order,
                        const char        transa,
                        const char        transb,
                        const md_t        m,
                        const md_t        n,
                        const md_t        k,
                        const int32_t     alpha,
                        const float*      a,
                        const md_t        lda,
                        const char        mem_format_a,
                        const int8_t*     b,
                        const md_t        ldb,
                        const char        mem_format_b,
                        const int32_t     beta,
                        void*             c,
                        const md_t        ldc,
                        dlp_metadata_t*   metadata,
                        const char*       func_name,
                        kernel_datatype_t krnl_dtype,
                        DLP_TYPE          c_dtype)
{
    DLP_GEMM_START_LOGGER();
    DLP_GEMM_WRITE_LOGGER(func_name, order, transa, transb, m, n, k,
                          ((float)alpha), lda, mem_format_a, ldb, mem_format_b,
                          ((float)beta), ldc, metadata);

    // Set default error status to success. Will be updated if any check fails.
    DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_SUCCESS);

    // =========================================================================
    // HARDWARE ISA VALIDATION
    // =========================================================================
    // Verify processor supports required instruction set extensions.
    if (dlp_cpuid_is_avx512vnni_supported() == FALSE) {
        dlp_print_msg(" AVX512_VNNI ISA not supported by processor, "
                      "cannot perform f32s8s32 gemm.",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // Initialize global context with cache-optimized block sizes (MC, NC, KC,
    // NR, MR).
    dlp_init_global_cntx();

    // Validate input parameters (dimensions, strides, pointers, etc.).
    dlp_clsc_err_t err_no = DLP_CLSC_SUCCESS;
    AOCL_DLP_GEMM_CHECK(func_name, order, transa, transb, m, n, k, a, lda,
                        mem_format_a, b, ldb, mem_format_b, c, ldc, err_no);
    if (err_no != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err_no);
        goto err_hndl;
    }

    // Add early returns for NULL pointers.
    if (metadata == NULL || metadata->a_pre_quant == NULL
        || metadata->a_post_quant == NULL) {
        dlp_print_msg("One or more required parameters (metadata, a_pre_quant, "
                      "a_post_quant) are NULL. Exiting..",
                      __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NULL_POINTER);
        goto err_hndl;
    }

    dlp_trans_t dlp_transa;
    dlp_trans_t dlp_transb;
    dlp_param_map_netlib_to_dlp_trans(transa, &dlp_transa);
    dlp_param_map_netlib_to_dlp_trans(transb, &dlp_transb);

    bool is_column_major = ((order == 'c') || (order == 'C'));

    // Current implementation limitations:
    // - Only row-major layout supported (column-major requires different
    // kernel)
    if (is_column_major == TRUE) {
        dlp_print_msg("Column major for A is not supported.", __FILE__,
                      __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // Set matrix strides for row-major layout.
    // Row stride (rs): distance between consecutive rows
    // Column stride (cs): distance between consecutive columns
    md_t rs_a = lda; // A: row-major, no transpose
    md_t cs_a = 1;
    md_t rs_b = ldb; // B: depends on transpose
    md_t cs_b = 1;

    // Adjust A strides if transposed (effectively column-major access).
    if (dlp_is_trans(dlp_transa)) {
        rs_a = 1;
        cs_a = lda;
    }
    // Adjust B strides if transposed (effectively column-major access).
    if (dlp_is_trans(dlp_transb)) {
        rs_b = 1;
        cs_b = ldb;
    }

    const md_t rs_c = ldc; // C: always row-major output
    const md_t cs_c = 1;

    AOCL_DLP_MEMORY_TAG mtag_a;
    AOCL_DLP_MEMORY_TAG mtag_b;

    dlp_param_map_char_to_lpmtag(mem_format_a, &mtag_a);
    dlp_param_map_char_to_lpmtag(mem_format_b, &mtag_b);

    // A matrix reordered format not supported (quantization happens
    // on-the-fly).
    if (mtag_a == REORDERED) {
        dlp_print_msg(
            " Reordering of A matrix is not supported in row major case.",
            __FILE__, __LINE__);
        DLP_METADATA_SET_ERROR(metadata, DLP_CLSC_NOT_SUPPORTED);
        goto err_hndl;
    }

    // Treat UNPACKED A as PACK so that on-the-fly F32->S8 quantization of A
    // can be implemented using the same packing path (quantization as packing).
    if (mtag_a == UNPACKED) {
        mtag_a = PACK;
    }

    // S8 B matrix MUST be packed or reordered for efficient kernel execution.
    if (mtag_b == UNPACKED) {
        mtag_b = PACK;
    }

    // Translate user-provided post-op metadata to internal linked-list format.
    // Post-ops includes dequantization of results.
    dlp_gemm_post_op post_op_list[AOCL_DLP_MAX_POST_OPS + 1];
    dlp_clsc_err_t   err = dlp_gemm_translate_to_post_ops_list(
        metadata, post_op_list, (void*)c, (void*)(&order), m, n);

    if (err != DLP_CLSC_SUCCESS) {
        DLP_METADATA_SET_ERROR(metadata, err);
        goto err_hndl;
    }
    // Initialize runtime configuration from global settings (thread count,
    // etc.).
    dlp_rntm_t rntm_g;
    dlp_rntm_init_from_global(&rntm_g);

    dlp_gemm_cntx_t* lcntx_g = dlp_gemm_get_global_cntx_obj(S8S8S32OS32);
    dlp_gemm_cntx_t  lcntx_l;

    // Create thread-local copy since context may be modified during execution
    lcntx_l = *lcntx_g;

    // Initialize DLP Plus kernel path.
    lcntx_l.dlp_kernel_hndl.kernel_base = NULL;

    dlp_init_and_get_kernel_hndl(
        krnl_dtype, order, mtag_a, mtag_b, m, n, k, rs_a, cs_a, rs_b, cs_b,
        rs_c, cs_c, (void*)&alpha, (void*)&beta, post_op_list, lcntx_l.blksz.MR,
        lcntx_l.blksz.NR, lcntx_l.blksz.KC, c_dtype, &lcntx_l.dlp_kernel_hndl);

    if (lcntx_l.dlp_kernel_hndl.kernel_base == NULL) {
        dlp_print_msg(" Kernel handle is not initialized. Only supported for "
                      "JIT generated kernels.",
                      __FILE__, __LINE__);
        goto err_hndl;
    }

    dlp_gemm_ops_bundle_t ops =
        DLP_GEMM_OPS_BUNDLE_INIT_QUANT(metadata->a_pre_quant, post_op_list);

#ifdef DLP_ENABLE_OPENMP
    // Multi-threaded execution using OpenMP parallel regions.
    dlp_gemm_f32s8s32os32_openmp_thread_decorator(
        m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, (int32_t*)c,
        rs_c, cs_c, alpha, beta, &rntm_g, &lcntx_l, &ops, c_dtype);
#else
    // Single-threaded or manually-threaded execution.
    dlp_gemm_f32s8s32os32_thread_decorator(
        m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, (int32_t*)c,
        rs_c, cs_c, alpha, beta, &rntm_g, &lcntx_l, &ops, c_dtype);
#endif

err_hndl:;
    DLP_GEMM_STOP_LOGGER();
}

// =========================================================================
// API Implementations for f32s8s32 variants
// =========================================================================

void
aocl_gemm_f32s8s32obf16(const char      order,
                        const char      transa,
                        const char      transb,
                        const md_t      m,
                        const md_t      n,
                        const md_t      k,
                        const int32_t   alpha,
                        const float*    a,
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
    aocl_gemm_f32s8s32_impl(order, transa, transb, m, n, k, alpha, a, lda,
                            mem_format_a, b, ldb, mem_format_b, beta, (void*)c,
                            ldc, metadata, "f32s8s32obf16",
                            DLP_KERNEL_S8S8S32OBF16, DLP_BF16);
}

void
aocl_gemm_f32s8s32os32(const char      order,
                       const char      transa,
                       const char      transb,
                       const md_t      m,
                       const md_t      n,
                       const md_t      k,
                       const int32_t   alpha,
                       const float*    a,
                       const md_t      lda,
                       const char      mem_format_a,
                       const int8_t*   b,
                       const md_t      ldb,
                       const char      mem_format_b,
                       const int32_t   beta,
                       int32_t*        c,
                       const md_t      ldc,
                       dlp_metadata_t* metadata)
{
    aocl_gemm_f32s8s32_impl(order, transa, transb, m, n, k, alpha, a, lda,
                            mem_format_a, b, ldb, mem_format_b, beta, (void*)c,
                            ldc, metadata, "f32s8s32os32",
                            DLP_KERNEL_S8S8S32OS32, DLP_S32);
}

void
aocl_gemm_f32s8s32of32(const char      order,
                       const char      transa,
                       const char      transb,
                       const md_t      m,
                       const md_t      n,
                       const md_t      k,
                       const int32_t   alpha,
                       const float*    a,
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
    aocl_gemm_f32s8s32_impl(order, transa, transb, m, n, k, alpha, a, lda,
                            mem_format_a, b, ldb, mem_format_b, beta, (void*)c,
                            ldc, metadata, "f32s8s32of32",
                            DLP_KERNEL_S8S8S32OF32, DLP_F32);
}

void
aocl_gemm_f32s8s32os8(const char      order,
                      const char      transa,
                      const char      transb,
                      const md_t      m,
                      const md_t      n,
                      const md_t      k,
                      const int32_t   alpha,
                      const float*    a,
                      const md_t      lda,
                      const char      mem_format_a,
                      const int8_t*   b,
                      const md_t      ldb,
                      const char      mem_format_b,
                      const int32_t   beta,
                      int8_t*         c,
                      const md_t      ldc,
                      dlp_metadata_t* metadata)
{
    aocl_gemm_f32s8s32_impl(order, transa, transb, m, n, k, alpha, a, lda,
                            mem_format_a, b, ldb, mem_format_b, beta, (void*)c,
                            ldc, metadata, "f32s8s32os8", DLP_KERNEL_S8S8S32OS8,
                            DLP_S8);
}

void
aocl_gemm_f32s8s32ou8(const char      order,
                      const char      transa,
                      const char      transb,
                      const md_t      m,
                      const md_t      n,
                      const md_t      k,
                      const int32_t   alpha,
                      const float*    a,
                      const md_t      lda,
                      const char      mem_format_a,
                      const int8_t*   b,
                      const md_t      ldb,
                      const char      mem_format_b,
                      const int32_t   beta,
                      uint8_t*        c,
                      const md_t      ldc,
                      dlp_metadata_t* metadata)
{
    aocl_gemm_f32s8s32_impl(order, transa, transb, m, n, k, alpha, a, lda,
                            mem_format_a, b, ldb, mem_format_b, beta, (void*)c,
                            ldc, metadata, "f32s8s32ou8", DLP_KERNEL_S8S8S32OU8,
                            DLP_U8);
}
