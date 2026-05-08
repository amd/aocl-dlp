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

#ifndef AOCL_DLP_GEMM_INTERFACE_H
#define AOCL_DLP_GEMM_INTERFACE_H

#include "classic/aocl_bf16_type.h"
#include "classic/aocl_fp16_type.h"
#include "classic/aocl_gemm_post_ops.h"
#include "classic/dlp_base_types.h"

/**
 * @brief Returns the size of the buffer (in bytes) required for the
 * reordered matrix.
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] trans Transpose option for the matrix.
 * @param[in] mat_type Type of the matrix (e.g., 'A' for matrix A, 'B' for
 * matrix B).
 * @param[in] k Number of rows in the matrix.
 * @param[in] n Number of columns in the matrix.
 * @param[in] metadata Metadata for the post-operations.
 * @return Size of the buffer in bytes.
 */
DLP_CLASSIC_EXPORT msz_t
aocl_get_reorder_buf_size_f32f32f32of32(const char      order,
                                        const char      trans,
                                        const char      mat_type,
                                        const md_t      k,
                                        const md_t      n,
                                        dlp_metadata_t* metadata);
/// @brief Returns buffer size (in bytes) for matrix reordering.
DLP_CLASSIC_EXPORT msz_t
aocl_get_reorder_buf_size_u8s8s32os32(const char      order,
                                      const char      trans,
                                      const char      mat_type,
                                      const md_t      k,
                                      const md_t      n,
                                      dlp_metadata_t* metadata);
/// @brief Returns buffer size (in bytes) for matrix reordering.
DLP_CLASSIC_EXPORT msz_t
aocl_get_reorder_buf_size_bf16bf16f32of32(const char      order,
                                          const char      trans,
                                          const char      mat_type,
                                          const md_t      k,
                                          const md_t      n,
                                          dlp_metadata_t* metadata);
/// @brief Returns buffer size (in bytes) for matrix reordering.
DLP_CLASSIC_EXPORT msz_t
aocl_get_reorder_buf_size_s8s8s32os32(const char      order,
                                      const char      trans,
                                      const char      mat_type,
                                      const md_t      k,
                                      const md_t      n,
                                      dlp_metadata_t* metadata);
/// @brief Returns buffer size (in bytes) for matrix reordering.
DLP_CLASSIC_EXPORT msz_t
aocl_get_reorder_buf_size_u8s4s32os32(const char      order,
                                      const char      trans,
                                      const char      mat_type,
                                      const md_t      k,
                                      const md_t      n,
                                      dlp_metadata_t* metadata);
/// @brief Returns buffer size (in bytes) for matrix reordering.
DLP_CLASSIC_EXPORT msz_t
aocl_get_reorder_buf_size_bf16s4f32of32(const char      order,
                                        const char      trans,
                                        const char      mat_type,
                                        const md_t      k,
                                        const md_t      n,
                                        dlp_metadata_t* metadata);
/// @brief Returns buffer size (in bytes) for matrix reordering.
DLP_CLASSIC_EXPORT msz_t
aocl_get_reorder_buf_size_f16f16f16of16(const char      order,
                                        const char      trans,
                                        const char      mat_type,
                                        const md_t      k,
                                        const md_t      n,
                                        dlp_metadata_t* metadata);
DLP_CLASSIC_EXPORT msz_t
aocl_get_reorder_buf_size_f32f16f32of32(const char      order,
                                        const char      trans,
                                        const char      mat_type,
                                        const md_t      k,
                                        const md_t      n,
                                        dlp_metadata_t* metadata);

/**
 * @brief Returns the size of the buffer (in bytes) required for the
 * reordered matrix with symmetric quantization.
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] trans Transpose option for the matrix.
 * @param[in] mat_type Type of the matrix (e.g., 'A' for matrix A, 'B' for
 * matrix B).
 * @param[in] k Number of rows in the matrix.
 * @param[in] n Number of columns in the matrix.
 * @param[in] symq_meta_data Metadata for symmetric quantization.
 * @param[in] metadata Metadata for the post-operations.
 * @return Size of the buffer in bytes.
 */
DLP_CLASSIC_EXPORT msz_t
aocl_get_reorder_buf_size_s8s8s32os32_sym_quant(
    const char           order,
    const char           trans,
    const char           mat_type,
    const md_t           k,
    const md_t           n,
    DLP_SYMM_STAT_QUANT* symq_meta_data,
    dlp_metadata_t*      metadata);

/**
 * @brief Performs reordering of the input matrix. Expanded from
 * AOCL_DLP_GEMM_REORDER macro.
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] trans Transpose option for the matrix.
 * @param[in] mat_type Type of the matrix (e.g., 'A' for matrix A, 'B' for
 * matrix B).
 * @param[in] input_buf_addr Pointer to the input matrix buffer.
 * @param[out] reorder_buf_addr Pointer to the reordered matrix buffer.
 * @param[in] k Number of rows in the matrix.
 * @param[in] n Number of columns in the matrix.
 * @param[in] ldb Leading dimension of the matrix.
 * @param[in] metadata Metadata for the post-operations.
 */
DLP_CLASSIC_EXPORT void
aocl_reorder_f32f32f32of32(const char      order,
                           const char      trans,
                           const char      mat_type,
                           const float*    input_buf_addr,
                           float*          reorder_buf_addr,
                           const md_t      k,
                           const md_t      n,
                           const md_t      ldb,
                           dlp_metadata_t* metadata);
/// @brief Reorders the input matrix into an optimized layout.
DLP_CLASSIC_EXPORT void
aocl_reorder_f32f32f32of32_reference(const char      order,
                                     const char      trans,
                                     const char      mat_type,
                                     const float*    input_buf_addr,
                                     float*          reorder_buf_addr,
                                     const md_t      k,
                                     const md_t      n,
                                     const md_t      ldb,
                                     dlp_metadata_t* metadata);
/// @brief Reorders the input matrix into an optimized layout.
DLP_CLASSIC_EXPORT void
aocl_reorder_u8s8s32os32(const char      order,
                         const char      trans,
                         const char      mat_type,
                         const int8_t*   input_buf_addr,
                         int8_t*         reorder_buf_addr,
                         const md_t      k,
                         const md_t      n,
                         const md_t      ldb,
                         dlp_metadata_t* metadata);
/// @brief Reorders the input matrix into an optimized layout.
DLP_CLASSIC_EXPORT void
aocl_reorder_bf16bf16f32of32(const char      order,
                             const char      trans,
                             const char      mat_type,
                             const bfloat16* input_buf_addr,
                             bfloat16*       reorder_buf_addr,
                             const md_t      k,
                             const md_t      n,
                             const md_t      ldb,
                             dlp_metadata_t* metadata);
/// @brief Reorders the input matrix into an optimized layout.
DLP_CLASSIC_EXPORT void
aocl_reorder_bf16bf16f32of32_reference(const char      order,
                                       const char      trans,
                                       const char      mat_type,
                                       const bfloat16* input_buf_addr,
                                       bfloat16*       reorder_buf_addr,
                                       const md_t      k,
                                       const md_t      n,
                                       const md_t      ldb,
                                       dlp_metadata_t* metadata);
/// @brief Reorders the input matrix into an optimized layout.
DLP_CLASSIC_EXPORT void
aocl_reorder_s8s8s32os32(const char      order,
                         const char      trans,
                         const char      mat_type,
                         const int8_t*   input_buf_addr,
                         int8_t*         reorder_buf_addr,
                         const md_t      k,
                         const md_t      n,
                         const md_t      ldb,
                         dlp_metadata_t* metadata);
/// @brief Reorders the input matrix into an optimized layout.
DLP_CLASSIC_EXPORT void
aocl_reorder_u8s4s32os32(const char      order,
                         const char      trans,
                         const char      mat_type,
                         const int8_t*   input_buf_addr,
                         int8_t*         reorder_buf_addr,
                         const md_t      k,
                         const md_t      n,
                         const md_t      ldb,
                         dlp_metadata_t* metadata);
/// @brief Reorders the input matrix into an optimized layout.
DLP_CLASSIC_EXPORT void
aocl_reorder_bf16s4f32of32(const char      order,
                           const char      trans,
                           const char      mat_type,
                           const int8_t*   input_buf_addr,
                           int8_t*         reorder_buf_addr,
                           const md_t      k,
                           const md_t      n,
                           const md_t      ldb,
                           dlp_metadata_t* metadata);
/// @brief Reorders the input matrix into an optimized layout.
DLP_CLASSIC_EXPORT void
aocl_reorder_f16f16f16of16(const char      order,
                           const char      trans,
                           const char      mat_type,
                           const float16*  input_buf_addr,
                           float16*        reorder_buf_addr,
                           const md_t      k,
                           const md_t      n,
                           const md_t      ldb,
                           dlp_metadata_t* metadata);
DLP_CLASSIC_EXPORT void
aocl_reorder_f32f16f32of32(const char      order,
                           const char      trans,
                           const char      mat_type,
                           const float16*  input_buf_addr,
                           float16*        reorder_buf_addr,
                           const md_t      k,
                           const md_t      n,
                           const md_t      ldb,
                           dlp_metadata_t* metadata);

/**
 * @brief Performs reordering of the input matrix for symmetric
 * quantization. Expanded from AOCL_DLP_GEMM_REORDER_SYM_QUANT macro.
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] trans Transpose option for the matrix.
 * @param[in] mat_type Type of the matrix (e.g., 'A' for matrix A, 'B' for
 * matrix B).
 * @param[in] input_buf_addr Pointer to the input matrix buffer.
 * @param[out] reorder_buf_addr Pointer to the reordered matrix buffer.
 * @param[in] k Number of rows in the matrix.
 * @param[in] n Number of columns in the matrix.
 * @param[in] ldb Leading dimension of the matrix.
 * @param[in] symq_meta_data Metadata for symmetric quantization.
 * @param[in] metadata Metadata for the post-operations.
 */
DLP_CLASSIC_EXPORT void
aocl_reorder_s8s8s32os32_sym_quant(const char           order,
                                   const char           trans,
                                   const char           mat_type,
                                   const int8_t*        input_buf_addr,
                                   int8_t*              reorder_buf_addr,
                                   const md_t           k,
                                   const md_t           n,
                                   const md_t           ldb,
                                   DLP_SYMM_STAT_QUANT* symq_meta_data,
                                   dlp_metadata_t*      metadata);

/**
 * @brief Performs reordering of the input matrix for mixed precision
 * DLP_GEMM. Expanded from AOCL_DLP_GEMM_REORDER_MXP macro.
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] trans Transpose option for the matrix.
 * @param[in] mat_type Type of the matrix (e.g., 'A' for matrix A, 'B' for
 * matrix B).
 * @param[in] input_buf_addr Pointer to the input matrix buffer.
 * @param[out] reorder_buf_addr Pointer to the reordered matrix buffer.
 * @param[in] k Number of rows in the matrix.
 * @param[in] n Number of columns in the matrix.
 * @param[in] ldb Leading dimension of the matrix.
 * @param[in] metadata Metadata for the post-operations.
 */
DLP_CLASSIC_EXPORT void
aocl_reorder_f32obf16(const char      order,
                      const char      trans,
                      const char      mat_type,
                      const float*    input_buf_addr,
                      bfloat16*       reorder_buf_addr,
                      const md_t      k,
                      const md_t      n,
                      const md_t      ldb,
                      dlp_metadata_t* metadata);

/**
 * @brief Converts a reordered matrix back to its original format.
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] mat_type Type of the matrix (e.g., 'A' for matrix A, 'B' for
 * matrix B).
 * @param[in] reorder_buf_addr Pointer to the reordered matrix buffer.
 * @param[out] output_buf_addr Pointer to the output matrix buffer.
 * @param[in] k Number of rows in the matrix.
 * @param[in] n Number of columns in the matrix.
 * @param[in] ldb Leading dimension of the matrix.
 * @param[in] metadata Metadata for the post-operations.
 */
DLP_CLASSIC_EXPORT void
aocl_unreorder_bf16bf16f32of32(const char      order,
                               const char      mat_type,
                               const bfloat16* reorder_buf_addr,
                               bfloat16*       output_buf_addr,
                               const md_t      k,
                               const md_t      n,
                               const md_t      ldb,
                               dlp_metadata_t* metadata);
/// @brief Converts a reordered matrix back to its original layout.
DLP_CLASSIC_EXPORT void
aocl_unreorder_bf16bf16f32of32_reference(const char      order,
                                         const char      mat_type,
                                         const bfloat16* reorder_buf_addr,
                                         bfloat16*       output_buf_addr,
                                         const md_t      k,
                                         const md_t      n,
                                         const md_t      ldb,
                                         dlp_metadata_t* metadata);
/// @brief Converts a reordered matrix back to its original layout.
DLP_CLASSIC_EXPORT void
aocl_unreorder_f32f32f32of32_reference(const char      order,
                                       const char      mat_type,
                                       const float*    reorder_buf_addr,
                                       float*          output_buf_addr,
                                       const md_t      k,
                                       const md_t      n,
                                       const md_t      ldb,
                                       dlp_metadata_t* metadata);
/// @brief Converts a reordered matrix back to its original layout.
DLP_CLASSIC_EXPORT void
aocl_unreorder_s8s8s32os32_reference(const char      order,
                                     const char      mat_type,
                                     const int8_t*   reorder_buf_addr,
                                     int8_t*         output_buf_addr,
                                     const md_t      k,
                                     const md_t      n,
                                     const md_t      ldb,
                                     dlp_metadata_t* metadata);
/// @brief Converts a reordered matrix back to its original layout.
DLP_CLASSIC_EXPORT void
aocl_unreorder_f16f16f16of16(const char      order,
                             const char      mat_type,
                             const float16*  reorder_buf_addr,
                             float16*        output_buf_addr,
                             const md_t      k,
                             const md_t      n,
                             const md_t      ldb,
                             dlp_metadata_t* metadata);

/**
 * @brief Performs GEMM (General Matrix Multiplication) with support
 * for fused post-operations.
 *
 * Computes C = post_ops(alpha * op(A) * op(B) + beta * C), where op(X)
 * is X or X^T depending on the transpose flag.
 *
 * @param[in] order Memory layout: 'R' for row-major, 'C' for column-major.
 * @param[in] transa Transpose option for matrix A: 'N' (no) or 'T' (yes).
 * @param[in] transb Transpose option for matrix B: 'N' (no) or 'T' (yes).
 * @param[in] m Number of rows in matrices A and C.
 * @param[in] n Number of columns in matrices B and C.
 * @param[in] k Number of columns in A / rows in B (inner dimension).
 * @param[in] alpha Scalar multiplier for the product of matrices A and B.
 * @param[in] a Pointer to matrix A.
 * @param[in] lda Leading dimension of matrix A.
 * @param[in] mem_format_a Memory format of matrix A: 'N' (normal),
 *            'P' (packed), or 'R' (reordered).
 * @param[in] b Pointer to matrix B.
 * @param[in] ldb Leading dimension of matrix B.
 * @param[in] mem_format_b Memory format of matrix B: 'N' (normal),
 *            'P' (packed), or 'R' (reordered).
 * @param[in] beta Scalar multiplier for matrix C.
 * @param[in,out] c Pointer to matrix C (output).
 * @param[in] ldc Leading dimension of matrix C.
 * @param[in] metadata Pointer to post-operation metadata, or NULL for
 *            no post-operations.
 */
DLP_CLASSIC_EXPORT void
aocl_gemm_u8s8s32os32(const char      order,
                      const char      transa,
                      const char      transb,
                      const md_t      m,
                      const md_t      n,
                      const md_t      k,
                      const int32_t   alpha,
                      const uint8_t*  a,
                      const md_t      lda,
                      const char      mem_format_a,
                      const int8_t*   b,
                      const md_t      ldb,
                      const char      mem_format_b,
                      const int32_t   beta,
                      int32_t*        c,
                      const md_t      ldc,
                      dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_u8s8s32os32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_u8s8s32os8(const char      order,
                     const char      transa,
                     const char      transb,
                     const md_t      m,
                     const md_t      n,
                     const md_t      k,
                     const int32_t   alpha,
                     const uint8_t*  a,
                     const md_t      lda,
                     const char      mem_format_a,
                     const int8_t*   b,
                     const md_t      ldb,
                     const char      mem_format_b,
                     const int32_t   beta,
                     int8_t*         c,
                     const md_t      ldc,
                     dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_u8s8s32os32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_u8s8s32of32(const char      order,
                      const char      transa,
                      const char      transb,
                      const md_t      m,
                      const md_t      n,
                      const md_t      k,
                      const int32_t   alpha,
                      const uint8_t*  a,
                      const md_t      lda,
                      const char      mem_format_a,
                      const int8_t*   b,
                      const md_t      ldb,
                      const char      mem_format_b,
                      const int32_t   beta,
                      float*          c,
                      const md_t      ldc,
                      dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_u8s8s32os32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_u8s8s32of16(const char      order,
                      const char      transa,
                      const char      transb,
                      const md_t      m,
                      const md_t      n,
                      const md_t      k,
                      const int32_t   alpha,
                      const uint8_t*  a,
                      const md_t      lda,
                      const char      mem_format_a,
                      const int8_t*   b,
                      const md_t      ldb,
                      const char      mem_format_b,
                      const int32_t   beta,
                      float16*        c,
                      const md_t      ldc,
                      dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_u8s8s32os32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_u8s8s32obf16(const char      order,
                       const char      transa,
                       const char      transb,
                       const md_t      m,
                       const md_t      n,
                       const md_t      k,
                       const int32_t   alpha,
                       const uint8_t*  a,
                       const md_t      lda,
                       const char      mem_format_a,
                       const int8_t*   b,
                       const md_t      ldb,
                       const char      mem_format_b,
                       const int32_t   beta,
                       bfloat16*       c,
                       const md_t      ldc,
                       dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_u8s8s32os32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_u8s8s32ou8(const char      order,
                     const char      transa,
                     const char      transb,
                     const md_t      m,
                     const md_t      n,
                     const md_t      k,
                     const int32_t   alpha,
                     const uint8_t*  a,
                     const md_t      lda,
                     const char      mem_format_a,
                     const int8_t*   b,
                     const md_t      ldb,
                     const char      mem_format_b,
                     const int32_t   beta,
                     uint8_t*        c,
                     const md_t      ldc,
                     dlp_metadata_t* metadata);

/**
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] transa Transpose option for matrix A.
 * @param[in] transb Transpose option for matrix B.
 * @param[in] m Row dimensions.
 * @param[in] n Column dimensions.
 * @param[in] k Inner dimensions.
 * @param[in] alpha Scalar multiplier for the product of matrices A and B.
 * @param[in] a Pointer to matrix A.
 * @param[in] lda Leading dimension of matrix A.
 * @param[in] mem_format_a Memory format of matrix A.
 * @param[in] b Pointer to matrix B.
 * @param[in] ldb Leading dimension of matrix B.
 * @param[in] mem_format_b Memory format of matrix B.
 * @param[in] beta Scalar multiplier for matrix C.
 * @param[in,out] c Pointer to matrix C.
 * @param[in] ldc Leading dimension of matrix C.
 * @param[in] metadata Pointer to post-operation structures.
 */
DLP_CLASSIC_EXPORT void
aocl_gemm_s8s8s32os32(const char      order,
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
                      int32_t*        c,
                      const md_t      ldc,
                      dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_s8s8s32os32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_s8s8s32os8(const char      order,
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
                     int8_t*         c,
                     const md_t      ldc,
                     dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_s8s8s32os32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_s8s8s32of32(const char      order,
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
                      dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_s8s8s32os32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_s8s8s32of16(const char      order,
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
                      float16*        c,
                      const md_t      ldc,
                      dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_s8s8s32os32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_s8s8s32obf16(const char      order,
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
                       dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_s8s8s32os32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_s8s8s32ou8(const char      order,
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
                     uint8_t*        c,
                     const md_t      ldc,
                     dlp_metadata_t* metadata);

/**
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] transa Transpose option for matrix A.
 * @param[in] transb Transpose option for matrix B.
 * @param[in] m Row dimensions.
 * @param[in] n Column dimensions.
 * @param[in] k Inner dimensions.
 * @param[in] alpha Scalar multiplier for the product of matrices A and B.
 * @param[in] a Pointer to matrix A.
 * @param[in] lda Leading dimension of matrix A.
 * @param[in] mem_format_a Memory format of matrix A.
 * @param[in] b Pointer to matrix B.
 * @param[in] ldb Leading dimension of matrix B.
 * @param[in] mem_format_b Memory format of matrix B.
 * @param[in] beta Scalar multiplier for matrix C.
 * @param[in,out] c Pointer to matrix C.
 * @param[in] ldc Leading dimension of matrix C.
 * @param[in] metadata Pointer to post-operation structures.
 */
DLP_CLASSIC_EXPORT void
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
                                dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_s8s8s32of32_sym_quant for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_s8s8s32obf16_sym_quant(const char      order,
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
                                 dlp_metadata_t* metadata);

/**
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] transa Transpose option for matrix A.
 * @param[in] transb Transpose option for matrix B.
 * @param[in] m Row dimensions.
 * @param[in] n Column dimensions.
 * @param[in] k Inner dimensions.
 * @param[in] alpha Scalar multiplier for the product of matrices A and B.
 * @param[in] a Pointer to matrix A.
 * @param[in] lda Leading dimension of matrix A.
 * @param[in] mem_format_a Memory format of matrix A.
 * @param[in] b Pointer to matrix B.
 * @param[in] ldb Leading dimension of matrix B.
 * @param[in] mem_format_b Memory format of matrix B.
 * @param[in] beta Scalar multiplier for matrix C.
 * @param[in,out] c Pointer to matrix C.
 * @param[in] ldc Leading dimension of matrix C.
 * @param[in] metadata Pointer to post-operation structures.
 */
DLP_CLASSIC_EXPORT void
aocl_gemm_bf16bf16f32of32(const char      order,
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
                          float*          c,
                          const md_t      ldc,
                          dlp_metadata_t* metadata);

/// User needs to pass Scale Factor for downscaling C Matrix to bfloat16.
/// Refer to @ref aocl_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
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
                           dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_bf16s4f32of32(const char      order,
                        const char      transa,
                        const char      transb,
                        const md_t      m,
                        const md_t      n,
                        const md_t      k,
                        const float     alpha,
                        const bfloat16* a,
                        const md_t      lda,
                        const char      mem_format_a,
                        const int8_t*   b,
                        const md_t      ldb,
                        const char      mem_format_b,
                        const float     beta,
                        float*          c,
                        const md_t      ldc,
                        dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_bf16u4f32of32(const char      order,
                        const char      transa,
                        const char      transb,
                        const md_t      m,
                        const md_t      n,
                        const md_t      k,
                        const float     alpha,
                        const bfloat16* a,
                        const md_t      lda,
                        const char      mem_format_a,
                        const uint8_t*  b,
                        const md_t      ldb,
                        const char      mem_format_b,
                        const float     beta,
                        float*          c,
                        const md_t      ldc,
                        dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_bf16u4f32obf16(const char      order,
                         const char      transa,
                         const char      transb,
                         const md_t      m,
                         const md_t      n,
                         const md_t      k,
                         const float     alpha,
                         const bfloat16* a,
                         const md_t      lda,
                         const char      mem_format_a,
                         const uint8_t*  b,
                         const md_t      ldb,
                         const char      mem_format_b,
                         const float     beta,
                         bfloat16*       c,
                         const md_t      ldc,
                         dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_bf16s4f32obf16(const char      order,
                         const char      transa,
                         const char      transb,
                         const md_t      m,
                         const md_t      n,
                         const md_t      k,
                         const float     alpha,
                         const bfloat16* a,
                         const md_t      lda,
                         const char      mem_format_a,
                         const int8_t*   b,
                         const md_t      ldb,
                         const char      mem_format_b,
                         const float     beta,
                         bfloat16*       c,
                         const md_t      ldc,
                         dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_bf16s8s32obf16(const char      order,
                         const char      transa,
                         const char      transb,
                         const md_t      m,
                         const md_t      n,
                         const md_t      k,
                         const int32_t   alpha,
                         const bfloat16* a,
                         const md_t      lda,
                         const char      mem_format_a,
                         const int8_t*   b,
                         const md_t      ldb,
                         const char      mem_format_b,
                         const int32_t   beta,
                         bfloat16*       c,
                         const md_t      ldc,
                         dlp_metadata_t* metadata);

/**
 * @brief F32xFP16->F32 mixed-precision GEMM: C = alpha * A * B + beta * C
 *
 * A is F32, B is FP16, C is F32. Accumulation is in F32.
 * B is converted from FP16 to F32 inside the JIT micro-kernel
 * using vcvtph2ps, then accumulated with vfmadd231ps.
 */
DLP_CLASSIC_EXPORT void
aocl_gemm_f32f16f32of32(const char      order,
                        const char      transa,
                        const char      transb,
                        const md_t      m,
                        const md_t      n,
                        const md_t      k,
                        const float     alpha,
                        const float*    a,
                        const md_t      lda,
                        const char      mem_format_a,
                        const float16*  b,
                        const md_t      ldb,
                        const char      mem_format_b,
                        const float     beta,
                        float*          c,
                        const md_t      ldc,
                        dlp_metadata_t* metadata);

/**
 * @brief FP16xFP16 GEMM with native FP16 accumulation and FP16 output
 *
 * Uses native FP16 FMA operations for both accumulation and output.
 * Provides maximum performance but with lower precision than F32
 * accumulation.
 *
 * Note: alpha and beta are float16 (NATIVE_FP16) so the JIT kernel can
 * consume them directly via vpbroadcastw + vmulph without a runtime widen.
 * This matches the FP16-end-to-end character of this API.
 *
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] transa Transpose option for matrix A.
 * @param[in] transb Transpose option for matrix B.
 * @param[in] m Row dimensions.
 * @param[in] n Column dimensions.
 * @param[in] k Inner dimensions.
 * @param[in] alpha Scalar multiplier for the product of matrices A and B
 * (FP16).
 * @param[in] a Pointer to matrix A (FP16).
 * @param[in] lda Leading dimension of matrix A.
 * @param[in] mem_format_a Memory format of matrix A.
 * @param[in] b Pointer to matrix B (FP16).
 * @param[in] ldb Leading dimension of matrix B.
 * @param[in] mem_format_b Memory format of matrix B.
 * @param[in] beta Scalar multiplier for matrix C (FP16).
 * @param[in,out] c Pointer to matrix C (FP16 output).
 * @param[in] ldc Leading dimension of matrix C.
 * @param[in] metadata Pointer to post-operation structures.
 */
DLP_CLASSIC_EXPORT void
aocl_gemm_f16f16f16of16(const char      order,
                        const char      transa,
                        const char      transb,
                        const md_t      m,
                        const md_t      n,
                        const md_t      k,
                        const float16   alpha,
                        const float16*  a,
                        const md_t      lda,
                        const char      mem_format_a,
                        const float16*  b,
                        const md_t      ldb,
                        const char      mem_format_b,
                        const float16   beta,
                        float16*        c,
                        const md_t      ldc,
                        dlp_metadata_t* metadata);

/**
 * @brief FP16xFP16 GEMM with native FP16 accumulation and F32 output
 *
 * Inputs A and B are FP16; C is float (32-bit). The JIT kernel accumulates
 * the AB product in native FP16 via vfmadd231ph, then converts the
 * accumulator to F32 at the post-ops boundary so beta*C (F32) and
 * alpha*(AB) (widened to F32) combine without precision loss on the C
 * side. The combine and store-back run in place every KC against user C
 * (no scratch buffer); alpha is FP16 in the public API, beta is widened
 * to float once before each kernel call.
 *
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] transa Transpose option for matrix A.
 * @param[in] transb Transpose option for matrix B.
 * @param[in] m Row dimensions.
 * @param[in] n Column dimensions.
 * @param[in] k Inner dimensions.
 * @param[in] alpha Scalar multiplier for the product of matrices A and B
 * (FP16).
 * @param[in] a Pointer to matrix A (FP16).
 * @param[in] lda Leading dimension of matrix A.
 * @param[in] mem_format_a Memory format of matrix A.
 * @param[in] b Pointer to matrix B (FP16).
 * @param[in] ldb Leading dimension of matrix B.
 * @param[in] mem_format_b Memory format of matrix B.
 * @param[in] beta Scalar multiplier for matrix C (FP16).
 * @param[in,out] c Pointer to matrix C (F32 output).
 * @param[in] ldc Leading dimension of matrix C.
 * @param[in] metadata Pointer to post-operation structures.
 */
DLP_CLASSIC_EXPORT void
aocl_gemm_f16f16f16of32(const char      order,
                        const char      transa,
                        const char      transb,
                        const md_t      m,
                        const md_t      n,
                        const md_t      k,
                        const float16   alpha,
                        const float16*  a,
                        const md_t      lda,
                        const char      mem_format_a,
                        const float16*  b,
                        const md_t      ldb,
                        const char      mem_format_b,
                        const float16   beta,
                        float*          c,
                        const md_t      ldc,
                        dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_bf16s8s32os32(const char      order,
                        const char      transa,
                        const char      transb,
                        const md_t      m,
                        const md_t      n,
                        const md_t      k,
                        const int32_t   alpha,
                        const bfloat16* a,
                        const md_t      lda,
                        const char      mem_format_a,
                        const int8_t*   b,
                        const md_t      ldb,
                        const char      mem_format_b,
                        const int32_t   beta,
                        int32_t*        c,
                        const md_t      ldc,
                        dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_bf16s8s32of32(const char      order,
                        const char      transa,
                        const char      transb,
                        const md_t      m,
                        const md_t      n,
                        const md_t      k,
                        const int32_t   alpha,
                        const bfloat16* a,
                        const md_t      lda,
                        const char      mem_format_a,
                        const int8_t*   b,
                        const md_t      ldb,
                        const char      mem_format_b,
                        const int32_t   beta,
                        float*          c,
                        const md_t      ldc,
                        dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_bf16s8s32os8(const char      order,
                       const char      transa,
                       const char      transb,
                       const md_t      m,
                       const md_t      n,
                       const md_t      k,
                       const int32_t   alpha,
                       const bfloat16* a,
                       const md_t      lda,
                       const char      mem_format_a,
                       const int8_t*   b,
                       const md_t      ldb,
                       const char      mem_format_b,
                       const int32_t   beta,
                       int8_t*         c,
                       const md_t      ldc,
                       dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_gemm_bf16s8s32ou8(const char      order,
                       const char      transa,
                       const char      transb,
                       const md_t      m,
                       const md_t      n,
                       const md_t      k,
                       const int32_t   alpha,
                       const bfloat16* a,
                       const md_t      lda,
                       const char      mem_format_a,
                       const int8_t*   b,
                       const md_t      ldb,
                       const char      mem_format_b,
                       const int32_t   beta,
                       uint8_t*        c,
                       const md_t      ldc,
                       dlp_metadata_t* metadata);
/// Refer to @ref aocl_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
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
                        dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
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
                       dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
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
                       dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
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
                      dlp_metadata_t* metadata);

/// Refer to @ref aocl_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
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
                      dlp_metadata_t* metadata);

/// @copydoc aocl_gemm_u8s8s32os32
DLP_CLASSIC_EXPORT void
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
                        dlp_metadata_t* metadata);

/**
 * @brief Batch GEMM (General Matrix Multiplication) with support for fused
 * post-operations.
 * @param[in] order Array of memory layouts (row-major or column-major).
 * @param[in] transa Array of transpose options for A matrices.
 * @param[in] transb Array of transpose options for B matrices.
 * @param[in] m Array of row dimensions for each matrix in the batch.
 * @param[in] n Array of column dimensions for each matrix in the batch.
 * @param[in] k Array of inner dimensions for each matrix in the batch.
 * @param[in] alpha Array of scalar multipliers for the product of matrices
 * A and B.
 * @param[in] a Array of pointers to A matrices.
 * @param[in] lda Array of leading dimensions for A matrices.
 * @param[in] b Array of pointers to B matrices.
 * @param[in] ldb Array of leading dimensions for B matrices.
 * @param[in] beta Array of scalar multipliers for C matrices.
 * @param[out] c Array of pointers to C matrices.
 * @param[in] ldc Array of leading dimensions for C matrices.
 * @param[in] group_count Number of groups in batch.
 * @param[in] group_size Array of group sizes.
 * @param[in] mem_format_a Array of memory formats for A matrices.
 * @param[in] mem_format_b Array of memory formats for B matrices.
 * @param[in] metadata Array of pointers to post-operation structures.
 */
DLP_CLASSIC_EXPORT void
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
                                dlp_metadata_t** metadata);

/// User needs to pass Scale Factor for downscaling C Matrix to bfloat16.
/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
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
                                 dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_batch_gemm_bf16s4f32of32(const char*      order,
                              const char*      transa,
                              const char*      transb,
                              const md_t*      m,
                              const md_t*      n,
                              const md_t*      k,
                              const float*     alpha,
                              const bfloat16** a,
                              const md_t*      lda,
                              const int8_t**   b,
                              const md_t*      ldb,
                              const float*     beta,
                              float**          c,
                              const md_t*      ldc,
                              const md_t       group_count,
                              const md_t*      group_size,
                              const char*      mem_format_a,
                              const char*      mem_format_b,
                              dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_batch_gemm_bf16s4f32obf16(const char*      order,
                               const char*      transa,
                               const char*      transb,
                               const md_t*      m,
                               const md_t*      n,
                               const md_t*      k,
                               const float*     alpha,
                               const bfloat16** a,
                               const md_t*      lda,
                               const int8_t**   b,
                               const md_t*      ldb,
                               const float*     beta,
                               bfloat16**       c,
                               const md_t*      ldc,
                               const md_t       group_count,
                               const md_t*      group_size,
                               const char*      mem_format_a,
                               const char*      mem_format_b,
                               dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_batch_gemm_f32f32f32of32(const char*      order,
                              const char*      transa,
                              const char*      transb,
                              const md_t*      m,
                              const md_t*      n,
                              const md_t*      k,
                              const float*     alpha,
                              const float**    a,
                              const md_t*      lda,
                              const float**    b,
                              const md_t*      ldb,
                              const float*     beta,
                              float**          c,
                              const md_t*      ldc,
                              const md_t       group_count,
                              const md_t*      group_size,
                              const char*      mem_format_a,
                              const char*      mem_format_b,
                              dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
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
                            dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
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
                           dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
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
                            dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
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
                             dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
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
                           dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_batch_gemm_s8s8s32os32(const char*      order,
                            const char*      transa,
                            const char*      transb,
                            const md_t*      m,
                            const md_t*      n,
                            const md_t*      k,
                            const int32_t*   alpha,
                            const int8_t**   a,
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
                            dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_batch_gemm_s8s8s32os8(const char*      order,
                           const char*      transa,
                           const char*      transb,
                           const md_t*      m,
                           const md_t*      n,
                           const md_t*      k,
                           const int32_t*   alpha,
                           const int8_t**   a,
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
                           dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_batch_gemm_s8s8s32of32(const char*      order,
                            const char*      transa,
                            const char*      transb,
                            const md_t*      m,
                            const md_t*      n,
                            const md_t*      k,
                            const int32_t*   alpha,
                            const int8_t**   a,
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
                            dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_batch_gemm_s8s8s32obf16(const char*      order,
                             const char*      transa,
                             const char*      transb,
                             const md_t*      m,
                             const md_t*      n,
                             const md_t*      k,
                             const int32_t*   alpha,
                             const int8_t**   a,
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
                             dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_batch_gemm_s8s8s32ou8(const char*      order,
                           const char*      transa,
                           const char*      transb,
                           const md_t*      m,
                           const md_t*      n,
                           const md_t*      k,
                           const int32_t*   alpha,
                           const int8_t**   a,
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
                           dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_batch_gemm_s8s8s32of32_sym_quant(const char*      order,
                                      const char*      transa,
                                      const char*      transb,
                                      const md_t*      m,
                                      const md_t*      n,
                                      const md_t*      k,
                                      const int32_t*   alpha,
                                      const int8_t**   a,
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
                                      dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_batch_gemm_s8s8s32obf16_sym_quant(const char*      order,
                                       const char*      transa,
                                       const char*      transb,
                                       const md_t*      m,
                                       const md_t*      n,
                                       const md_t*      k,
                                       const int32_t*   alpha,
                                       const int8_t**   a,
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
                                       dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
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
                              dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
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
                             dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
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
                              dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
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
                               dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
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
                             dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_batch_gemm_f32s8s32os32(const char*      order,
                             const char*      transa,
                             const char*      transb,
                             const md_t*      m,
                             const md_t*      n,
                             const md_t*      k,
                             const int32_t*   alpha,
                             const float**    a,
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
                             dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_batch_gemm_f32s8s32os8(const char*      order,
                            const char*      transa,
                            const char*      transb,
                            const md_t*      m,
                            const md_t*      n,
                            const md_t*      k,
                            const int32_t*   alpha,
                            const float**    a,
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
                            dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_batch_gemm_f32s8s32of32(const char*      order,
                             const char*      transa,
                             const char*      transb,
                             const md_t*      m,
                             const md_t*      n,
                             const md_t*      k,
                             const int32_t*   alpha,
                             const float**    a,
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
                             dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_batch_gemm_f32s8s32obf16(const char*      order,
                              const char*      transa,
                              const char*      transb,
                              const md_t*      m,
                              const md_t*      n,
                              const md_t*      k,
                              const int32_t*   alpha,
                              const float**    a,
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
                              dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_batch_gemm_f32s8s32ou8(const char*      order,
                            const char*      transa,
                            const char*      transb,
                            const md_t*      m,
                            const md_t*      n,
                            const md_t*      k,
                            const int32_t*   alpha,
                            const float**    a,
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
                            dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_batch_gemm_f16f16f16of16(const char*      order,
                              const char*      transa,
                              const char*      transb,
                              const md_t*      m,
                              const md_t*      n,
                              const md_t*      k,
                              const float*     alpha,
                              const float16**  a,
                              const md_t*      lda,
                              const float16**  b,
                              const md_t*      ldb,
                              const float*     beta,
                              float16**        c,
                              const md_t*      ldc,
                              const md_t       group_count,
                              const md_t*      group_size,
                              const char*      mem_format_a,
                              const char*      mem_format_b,
                              dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_batch_gemm_f32f16f32of32(const char*      order,
                              const char*      transa,
                              const char*      transb,
                              const md_t*      m,
                              const md_t*      n,
                              const md_t*      k,
                              const float*     alpha,
                              const float**    a,
                              const md_t*      lda,
                              const float16**  b,
                              const md_t*      ldb,
                              const float*     beta,
                              float**          c,
                              const md_t*      ldc,
                              const md_t       group_count,
                              const md_t*      group_size,
                              const char*      mem_format_a,
                              const char*      mem_format_b,
                              dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_batch_gemm_bf16u4f32of32(const char*      order,
                              const char*      transa,
                              const char*      transb,
                              const md_t*      m,
                              const md_t*      n,
                              const md_t*      k,
                              const float*     alpha,
                              const bfloat16** a,
                              const md_t*      lda,
                              const uint8_t**  b,
                              const md_t*      ldb,
                              const float*     beta,
                              float**          c,
                              const md_t*      ldc,
                              const md_t       group_count,
                              const md_t*      group_size,
                              const char*      mem_format_a,
                              const char*      mem_format_b,
                              dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
aocl_batch_gemm_bf16u4f32obf16(const char*      order,
                               const char*      transa,
                               const char*      transb,
                               const md_t*      m,
                               const md_t*      n,
                               const md_t*      k,
                               const float*     alpha,
                               const bfloat16** a,
                               const md_t*      lda,
                               const uint8_t**  b,
                               const md_t*      ldb,
                               const float*     beta,
                               bfloat16**       c,
                               const md_t*      ldc,
                               const md_t       group_count,
                               const md_t*      group_size,
                               const char*      mem_format_a,
                               const char*      mem_format_b,
                               dlp_metadata_t** metadata);

#endif // AOCL_DLP_GEMM_INTERFACE_H
