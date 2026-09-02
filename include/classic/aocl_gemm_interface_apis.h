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
#include "classic/aocl_gemm_metadata.h"
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
 * @param[in,out] metadata Metadata for the post-operations.
 * @return Size of the buffer in bytes. This is layout arithmetic and does
 * not require the ISA used by the matching tuned reorder; a zero return
 * with DLP_CLSC_NOT_SUPPORTED means the arguments were rejected, not that
 * the processor is missing an ISA.
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
 * @param[in,out] metadata Metadata containing B quantization group size.
 * @return Size of the buffer in bytes.
 */
DLP_CLASSIC_EXPORT msz_t
aocl_get_reorder_buf_size_s8s8s32os32_sym_quant(const char      order,
                                                const char      trans,
                                                const char      mat_type,
                                                const md_t      k,
                                                const md_t      n,
                                                dlp_metadata_t* metadata);

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
 * @param[in,out] metadata Metadata for the post-operations.
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
/// @brief Portable C reorder of B. Honours metadata->block_params.
DLP_CLASSIC_EXPORT void
aocl_reorder_u8s8s32os32_reference(const char      order,
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
/// @brief Portable C reorder of B. Honours metadata->block_params.
DLP_CLASSIC_EXPORT void
aocl_reorder_s8s8s32os32_reference(const char      order,
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
/// @brief Portable C reorder of B. Does not require AVX-512-FP16.
/// Honours metadata->block_params.
DLP_CLASSIC_EXPORT void
aocl_reorder_f16f16f16of16_reference(const char      order,
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
 * @param[in,out] metadata Metadata containing B quantization group size.
 */
DLP_CLASSIC_EXPORT void
aocl_reorder_s8s8s32os32_sym_quant(const char      order,
                                   const char      trans,
                                   const char      mat_type,
                                   const int8_t*   input_buf_addr,
                                   int8_t*         reorder_buf_addr,
                                   const md_t      k,
                                   const md_t      n,
                                   const md_t      ldb,
                                   dlp_metadata_t* metadata);

/**
 * @brief Returns the size (in bytes) of the reordered buffer required for the
 * symmetric-quantized s8s4 GEMM path. The input B matrix is a signed 4-bit
 * (nibble-packed) weight matrix. The reordered buffer stores the s8 VNNI-4
 * packed weights compressed 2:1 back to nibbles, followed by the per-group
 * int32 column sums (stored uncompressed).
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] trans Transpose option for the matrix.
 * @param[in] mat_type Type of the matrix ('B'/'W' only).
 * @param[in] k Number of rows in the matrix (inner dimension).
 * @param[in] n Number of columns in the matrix.
 * @param[in,out] metadata Metadata carrying the B-side quantization group size
 * (via metadata->b_quant_op) and used for error reporting.
 * @return Size of the buffer in bytes.
 */
DLP_CLASSIC_EXPORT msz_t
aocl_get_reorder_buf_size_s8s4s32os32(const char      order,
                                      const char      trans,
                                      const char      mat_type,
                                      const md_t      k,
                                      const md_t      n,
                                      dlp_metadata_t* metadata);

/**
 * @brief Reorders a signed 4-bit (nibble-packed) B matrix into the compact
 * reordered layout consumed by the s8s4 symmetric-quantized GEMM. Refer to
 * @ref aocl_reorder_s8s8s32os32_sym_quant for parameter semantics. The
 * @p input_buf_addr points to nibble-packed s4 data; @p reorder_buf_addr must
 * be sized via @ref aocl_get_reorder_buf_size_s8s4s32os32. The quantization
 * group size is read from @p metadata->b_quant_op.
 */
DLP_CLASSIC_EXPORT void
aocl_reorder_s8s4s32os32(const char      order,
                         const char      trans,
                         const char      mat_type,
                         const int8_t*   input_buf_addr,
                         int8_t*         reorder_buf_addr,
                         const md_t      k,
                         const md_t      n,
                         const md_t      ldb,
                         dlp_metadata_t* metadata);

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
 * @param[in,out] metadata Metadata for the post-operations.
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
 * @defgroup unreorder Un-reorder APIs
 *
 * An un-reorder API is the exact inverse of the matching aocl_reorder_* API:
 * it takes a buffer that aocl_reorder_<type>() produced and writes the
 * original B matrix back out in plain row-major or column-major form. The
 * round trip is bit-exact, so no floating point tolerance is needed when
 * comparing against the matrix that was reordered.
 *
 * This is useful when the reordered buffer is the only copy of the weights
 * that is still live, for example to serialise weights back out, to hand
 * them to another library, or to inspect them while debugging.
 *
 * Typical usage:
 * @code
 * // 1. Reorder B once and keep the packed buffer.
 * msz_t buf_sz = aocl_get_reorder_buf_size_bf16bf16f32of32('r', 'n', 'B',
 *                                                          k, n, NULL);
 * bfloat16* b_reordered = (bfloat16*)malloc(buf_sz);
 * aocl_reorder_bf16bf16f32of32('r', 'n', 'B', b, b_reordered, k, n, ldb,
 *                              NULL);
 *
 * // 2. Run as many GEMMs as needed with mem_format_b = 'R'.
 *
 * // 3. Recover B. trans selects the output storage (need not match the
 * //    reorder call). output_buf is strided by ldb; the API never allocates.
 * bfloat16* b_recovered = (bfloat16*)malloc(k * ldb * sizeof(bfloat16));
 * dlp_metadata_t md;
 * memset(&md, 0, sizeof(md));
 * aocl_unreorder_bf16bf16f32of32('r', 'n', 'B', b_reordered, b_recovered, k,
 *                                n, ldb, &md);
 * if (md.error_hndl.error_code != DLP_CLSC_SUCCESS) { ... }
 * @endcode
 *
 * The reordered buffer carries no header describing how it was produced, so
 * the caller must pass the same @p order, @p k, @p n and blocking that were
 * used for the reorder. @p trans and @p ldb describe the output storage and
 * need not match the reorder call. Passing a different order, k, n or
 * blocking silently decodes the buffer with the wrong layout.
 *
 * Functions carrying the @c _reference suffix are plain C implementations.
 * They run on any processor regardless of the ISA the reorder used, and are
 * the correctness oracle for the tuned variants. Size them with the matching
 * aocl_get_reorder_buf_size_* call, which is also ISA-independent. The
 * reorder/un-reorder variants without the suffix may require a specific ISA
 * and will set DLP_CLSC_NOT_SUPPORTED in @p metadata when it is unavailable.
 *
 * @{
 */

/**
 * @brief Converts a reordered matrix back to its original format.
 *
 * @param[in] order Memory layout to write @p output_buf_addr in: 'r' or 'R'
 * for row-major, 'c' or 'C' for column-major. Must match the order used when
 * the buffer was reordered.
 * @param[in] trans Output transpose: 'n'/'N' writes logical B, 't'/'T'
 * writes B stored as its transpose. Independent of the trans used to pack.
 * @param[in] mat_type Type of the matrix. Only 'B' is supported; passing 'A'
 * sets DLP_CLSC_NOT_SUPPORTED in @p metadata and writes nothing.
 * @param[in] reorder_buf_addr Reordered matrix buffer, as produced by the
 * matching aocl_reorder_* call.
 * @param[out] output_buf_addr Destination for the recovered matrix. The
 * caller owns this buffer. Size it from @p order and @p trans: at least
 * @p k * @p ldb elements for row-major no-transpose or column-major
 * transpose, and @p n * @p ldb otherwise.
 * @param[in] k Number of rows of the original matrix.
 * @param[in] n Number of columns of the original matrix.
 * @param[in] ldb Leading dimension of @p output_buf_addr, not of the
 * reordered buffer. Row-major: >= @p n if @p trans is 'n', >= @p k if 't'.
 * Column-major: >= @p k if @p trans is 'n', >= @p n if 't'.
 * @param[in,out] metadata Optional, may be NULL. On return
 * metadata->error_hndl.error_code reports success or the reason for failure.
 * If metadata->block_params is set, NC, KC and NR are taken from there
 * instead of the architecture defaults, in which case they must match the
 * blocking that was used to produce @p reorder_buf_addr.
 */
DLP_CLASSIC_EXPORT void
aocl_unreorder_bf16bf16f32of32(const char      order,
                               const char      trans,
                               const char      mat_type,
                               const bfloat16* reorder_buf_addr,
                               bfloat16*       output_buf_addr,
                               const md_t      k,
                               const md_t      n,
                               const md_t      ldb,
                               dlp_metadata_t* metadata);
/// @brief Inverse of aocl_reorder_bf16bf16f32of32(), in portable C.
DLP_CLASSIC_EXPORT void
aocl_unreorder_bf16bf16f32of32_reference(const char      order,
                                         const char      trans,
                                         const char      mat_type,
                                         const bfloat16* reorder_buf_addr,
                                         bfloat16*       output_buf_addr,
                                         const md_t      k,
                                         const md_t      n,
                                         const md_t      ldb,
                                         dlp_metadata_t* metadata);
/// @brief Inverse of aocl_reorder_f32f32f32of32(), in portable C.
DLP_CLASSIC_EXPORT void
aocl_unreorder_f32f32f32of32_reference(const char      order,
                                       const char      trans,
                                       const char      mat_type,
                                       const float*    reorder_buf_addr,
                                       float*          output_buf_addr,
                                       const md_t      k,
                                       const md_t      n,
                                       const md_t      ldb,
                                       dlp_metadata_t* metadata);
/// @brief Inverse of aocl_reorder_s8s8s32os32(), in portable C.
DLP_CLASSIC_EXPORT void
aocl_unreorder_s8s8s32os32_reference(const char      order,
                                     const char      trans,
                                     const char      mat_type,
                                     const int8_t*   reorder_buf_addr,
                                     int8_t*         output_buf_addr,
                                     const md_t      k,
                                     const md_t      n,
                                     const md_t      ldb,
                                     dlp_metadata_t* metadata);
/// @brief Inverse of aocl_reorder_u8s8s32os32(), in portable C.
DLP_CLASSIC_EXPORT void
aocl_unreorder_u8s8s32os32_reference(const char      order,
                                     const char      trans,
                                     const char      mat_type,
                                     const int8_t*   reorder_buf_addr,
                                     int8_t*         output_buf_addr,
                                     const md_t      k,
                                     const md_t      n,
                                     const md_t      ldb,
                                     dlp_metadata_t* metadata);
/// @brief Inverse of aocl_reorder_f16f16f16of16(), in portable C. Unlike
/// aocl_unreorder_f16f16f16of16() this does not require AVX-512-FP16.
DLP_CLASSIC_EXPORT void
aocl_unreorder_f16f16f16of16_reference(const char      order,
                                       const char      trans,
                                       const char      mat_type,
                                       const float16*  reorder_buf_addr,
                                       float16*        output_buf_addr,
                                       const md_t      k,
                                       const md_t      n,
                                       const md_t      ldb,
                                       dlp_metadata_t* metadata);
/// @brief Converts a reordered matrix back to its original layout.
/// Requires AVX-512-FP16; use aocl_unreorder_f16f16f16of16_reference()
/// on processors without it.
DLP_CLASSIC_EXPORT void
aocl_unreorder_f16f16f16of16(const char      order,
                             const char      trans,
                             const char      mat_type,
                             const float16*  reorder_buf_addr,
                             float16*        output_buf_addr,
                             const md_t      k,
                             const md_t      n,
                             const md_t      ldb,
                             dlp_metadata_t* metadata);
/** @} */

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
 * @param[in,out] metadata Pointer to post-operation metadata, or NULL for
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
 * @param[in,out] metadata Pointer to post-operation structures.
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
 * @param[in,out] metadata Pointer to post-operation structures.
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
 * @brief Symmetric-quantized GEMM with an s8 activation matrix A and a signed
 * 4-bit (s4) weight matrix B, accumulating in s32 and downscaling the output
 * to f32. Matrix B may be supplied either pre-reordered (mem_format_b == 'R',
 * via @ref aocl_reorder_s8s4s32os32, the compact-memory fast path) or as a raw
 * nibble-packed s4 matrix that is packed at runtime (mem_format_b == 'N', which
 * is promoted to packing, or 'P' to request packing explicitly). In all cases
 * the s4 weights are widened to s8 on the fly and consumed by the shared s8s8
 * micro-kernel.
 * Refer to @ref aocl_gemm_s8s8s32of32_sym_quant for the parameter semantics.
 */
DLP_CLASSIC_EXPORT void
aocl_gemm_s8s4s32of32(const char      order,
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

/// Refer to @ref aocl_gemm_s8s4s32of32 for info on parameters.
DLP_CLASSIC_EXPORT void
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
 * @param[in,out] metadata Pointer to post-operation structures.
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
 * @param[in,out] metadata Pointer to post-operation structures.
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
 * @param[in,out] metadata Pointer to post-operation structures.
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
 * @param[in,out] metadata Array of group_count metadata pointers; NULL causes
 *                         a clean return without computation, and non-NULL
 *                         elements may receive an error status.
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

/**
 * @copydoc aocl_batch_gemm_bf16bf16f32of32
 *
 * @details Processing stops at the first group that fails. On return, each
 * non-NULL element of @p metadata reports one of the following:
 * - @c DLP_CLSC_SUCCESS: the group was computed successfully.
 * - @c DLP_CLSC_FAILURE: the group was not processed because an earlier group
 *   failed, or the group was processed and encountered a generic failure.
 * - Any other non-success @c dlp_clsc_err_t value: the group was processed and
 *   encountered the reported error. Batch-level errors (for example,
 *   @c DLP_CLSC_NOT_SUPPORTED) may be reported even when no groups were
 * processed.
 *
 * Callers must not rely on the initial value of an error code to indicate that
 * its group was processed.
 *
 * @param[in,out] metadata Array of pointers to per-group post-operation
 * structures. On return, each non-NULL element contains its group status in
 * @c error_hndl.error_code as described above.
 */
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
                              dlp_metadata_t** metadata);

/// Refer to @ref aocl_batch_gemm_bf16bf16f32of32 for info on parameters.
/// FP16×FP16 batch GEMM with FP16 native accumulation and F32 output.
DLP_CLASSIC_EXPORT void
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
