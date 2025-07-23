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

#ifndef AOCL_GEMM_INTERFACE_H
#define AOCL_GEMM_INTERFACE_H

#include "classic/aocl_bf16_type.h"
#include "classic/aocl_gemm_post_ops.h"
#include "classic/dlp_base_types.h"

/**
 * @brief Returns the size of the buffer (in bytes) required for the
 * reordered matrix.
 *
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] trans Transpose option for the matrix.
 * @param[in] mat_type Type of the matrix (e.g., 'A' for matrix A, 'B' for
 * matrix B).
 * @param[in] k Number of rows in the matrix.
 * @param[in] n Number of columns in the matrix.
 * @return Size of the buffer in bytes.
 */
#define AOCL_GEMM_GET_REORDER_BUF_SIZE(LP_SFX)                                 \
    DLP_CLASSIC_EXPORT msz_t aocl_get_reorder_buf_size_##LP_SFX(               \
        const char order, const char trans, const char mat_type, const md_t k, \
        const md_t n)

AOCL_GEMM_GET_REORDER_BUF_SIZE(f32f32f32of32);
AOCL_GEMM_GET_REORDER_BUF_SIZE(u8s8s32os32);
AOCL_GEMM_GET_REORDER_BUF_SIZE(bf16bf16f32of32);
AOCL_GEMM_GET_REORDER_BUF_SIZE(s8s8s32os32);
AOCL_GEMM_GET_REORDER_BUF_SIZE(u8s4s32os32);
AOCL_GEMM_GET_REORDER_BUF_SIZE(bf16s4f32of32);

/**
 * @brief Returns the size of the buffer (in bytes) required for the
 * reordered matrix with symmetric quantization.
 *
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] trans Transpose option for the matrix.
 * @param[in] mat_type Type of the matrix (e.g., 'A' for matrix A, 'B' for
 * matrix B).
 * @param[in] k Number of rows in the matrix.
 * @param[in] n Number of columns in the matrix.
 * @param[in] meta_data Metadata for symmetric quantization.
 * @return Size of the buffer in bytes.
 */
#define AOCL_GEMM_GET_REORDER_BUF_SIZE_SYM_QUANT(LP_SFX)                       \
    DLP_CLASSIC_EXPORT msz_t aocl_get_reorder_buf_size_##LP_SFX(               \
        const char order, const char trans, const char mat_type, const md_t k, \
        const md_t n, AOCL_SYMM_STAT_QUANT* meta_data)

AOCL_GEMM_GET_REORDER_BUF_SIZE_SYM_QUANT(s8s8s32os32_sym_quant);

/**
 * @brief Performs reordering of the input matrix. Reordering is the process
 * of packing the entire matrix upfront, so that the benefits of packed matrix
 * is obtained without incurring the packing costs during matmul computation.
 *
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] trans Transpose option for the matrix.
 * @param[in] mat_type Type of the matrix (e.g., 'A' for matrix A, 'B' for
 * matrix B).
 * @param[in] input_buf_addr Pointer to the input matrix buffer.
 * @param[out] reorder_buf_addr Pointer to the reordered matrix buffer.
 * @param[in] k Number of rows in the matrix.
 * @param[in] n Number of columns in the matrix.
 * @param[in] ldb Leading dimension of the matrix.
 */
#define AOCL_GEMM_REORDER(B_type, LP_SFX)                                      \
    DLP_CLASSIC_EXPORT void aocl_reorder_##LP_SFX(                             \
        const char order, const char trans, const char mat_type,               \
        const B_type* input_buf_addr, B_type* reorder_buf_addr, const md_t k,  \
        const md_t n, const md_t ldb)

AOCL_GEMM_REORDER(float, f32f32f32of32);
AOCL_GEMM_REORDER(int8_t, u8s8s32os32);
AOCL_GEMM_REORDER(bfloat16, bf16bf16f32of32);
AOCL_GEMM_REORDER(bfloat16, bf16bf16f32of32_reference);
AOCL_GEMM_REORDER(int8_t, s8s8s32os32);
AOCL_GEMM_REORDER(int8_t, u8s4s32os32);
AOCL_GEMM_REORDER(int8_t, bf16s4f32of32);

/**
 * @brief Performs reordering of the input matrix for symmetric quantization.
 *
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] trans Transpose option for the matrix.
 * @param[in] mat_type Type of the matrix (e.g., 'A' for matrix A, 'B' for
 * matrix B).
 * @param[in] input_buf_addr Pointer to the input matrix buffer.
 * @param[out] reorder_buf_addr Pointer to the reordered matrix buffer.
 * @param[in] k Number of rows in the matrix.
 * @param[in] n Number of columns in the matrix.
 * @param[in] ldb Leading dimension of the matrix.
 * @param[in] meta_data Metadata for symmetric quantization.
 */
#define AOCL_GEMM_REORDER_SYM_QUANT(B_type, LP_SFX)                            \
    DLP_CLASSIC_EXPORT void aocl_reorder_##LP_SFX(                             \
        const char order, const char trans, const char mat_type,               \
        const B_type* input_buf_addr, B_type* reorder_buf_addr, const md_t k,  \
        const md_t n, const md_t ldb, AOCL_SYMM_STAT_QUANT* meta_data)

AOCL_GEMM_REORDER_SYM_QUANT(int8_t, s8s8s32os32_sym_quant);

/**
 * @brief Performs reordering of the input matrix for mixed precision LPGEMM.
 *
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] trans Transpose option for the matrix.
 * @param[in] mat_type Type of the matrix (e.g., 'A' for matrix A, 'B' for
 * matrix B).
 * @param[in] input_buf_addr Pointer to the input matrix buffer.
 * @param[out] reorder_buf_addr Pointer to the reordered matrix buffer.
 * @param[in] k Number of rows in the matrix.
 * @param[in] n Number of columns in the matrix.
 * @param[in] ldb Leading dimension of the matrix.
 */
#define AOCL_GEMM_REORDER_MXP(A_type, B_type, LP_SFX)                          \
    DLP_CLASSIC_EXPORT void aocl_reorder_##LP_SFX(                             \
        const char order, const char trans, const char mat_type,               \
        const A_type* input_buf_addr, B_type* reorder_buf_addr, const md_t k,  \
        const md_t n, const md_t ldb)

AOCL_GEMM_REORDER_MXP(float, bfloat16, f32obf16);

/**
 * @brief Converts a reordered matrix back to its original format.
 *
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] mat_type Type of the matrix (e.g., 'A' for matrix A, 'B' for
 * matrix B).
 * @param[in] reorder_buf_addr Pointer to the reordered matrix buffer.
 * @param[out] output_buf_addr Pointer to the output matrix buffer.
 * @param[in] k Number of rows in the matrix.
 * @param[in] n Number of columns in the matrix.
 * @param[in] ldb Leading dimension of the matrix.
 */
#define AOCL_GEMM_UNREORDER(B_type, LP_SFX)                                    \
    DLP_CLASSIC_EXPORT void aocl_unreorder_##LP_SFX(                           \
        const char order, const char mat_type, const B_type* reorder_buf_addr, \
        B_type* output_buf_addr, const md_t k, const md_t n, const md_t ldb)

AOCL_GEMM_UNREORDER(bfloat16, bf16bf16f32of32);
AOCL_GEMM_UNREORDER(bfloat16, bf16bf16f32of32_reference);
AOCL_GEMM_UNREORDER(int8_t, s8s8s32os32_reference);

/**
 * @brief GEMM (General Matrix Multiplication) with support for fused
 * post-operations.
 *
 * @param[in] order Memory layout (row-major or column-major).
 * @param[in] transa Transpose option for matrix A.
 * @param[in] transb Transpose option for matrix B.
 * @param[in] m Number of rows in matrix A and C.
 * @param[in] n Number of columns in matrix B and C.
 * @param[in] k Number of columns in matrix A and rows in matrix B.
 * @param[in] alpha Scalar multiplier for the product of matrices A and B.
 * @param[in] a Pointer to matrix A.
 * @param[in] lda Leading dimension of matrix A.
 * @param[in] mem_format_a Memory format of matrix A.
 * @param[in] b Pointer to matrix B.
 * @param[in] ldb Leading dimension of matrix B.
 * @param[in] mem_format_b Memory format of matrix B.
 * @param[in] beta Scalar multiplier for matrix C.
 * @param[out] c Pointer to matrix C.
 * @param[in] ldc Leading dimension of matrix C.
 * @param[in] post_op_unparsed Pointer to post-operation structures.
 */
#define AOCL_GEMM_MATMUL(A_type, B_type, C_type, Sum_type, LP_SFX)             \
    DLP_CLASSIC_EXPORT void aocl_gemm_##LP_SFX(                                \
        const char order, const char transa, const char transb, const md_t m,  \
        const md_t n, const md_t k, const Sum_type alpha, const A_type* a,     \
        const md_t lda, const char mem_format_a, const B_type* b,              \
        const md_t ldb, const char mem_format_b, const Sum_type beta,          \
        C_type* c, const md_t ldc, aocl_post_op* post_op_unparsed)

AOCL_GEMM_MATMUL(uint8_t, int8_t, int32_t, int32_t, u8s8s32os32);
AOCL_GEMM_MATMUL(uint8_t, int8_t, int8_t, int32_t, u8s8s32os8);
AOCL_GEMM_MATMUL(uint8_t, int8_t, bfloat16, int32_t, u8s8s32obf16);
AOCL_GEMM_MATMUL(uint8_t, int8_t, float, int32_t, u8s8s32of32);
AOCL_GEMM_MATMUL(uint8_t, int8_t, uint8_t, int32_t, u8s8s32ou8);

AOCL_GEMM_MATMUL(int8_t, int8_t, int32_t, int32_t, s8s8s32os32);
AOCL_GEMM_MATMUL(int8_t, int8_t, int8_t, int32_t, s8s8s32os8);
AOCL_GEMM_MATMUL(int8_t, int8_t, bfloat16, int32_t, s8s8s32obf16);
AOCL_GEMM_MATMUL(int8_t, int8_t, float, int32_t, s8s8s32of32);
AOCL_GEMM_MATMUL(int8_t, int8_t, uint8_t, int32_t, s8s8s32ou8);

// Symmetric static quantization GEMM API
AOCL_GEMM_MATMUL(int8_t, int8_t, float, int32_t, s8s8s32of32_sym_quant);
AOCL_GEMM_MATMUL(int8_t, int8_t, bfloat16, int32_t, s8s8s32obf16_sym_quant);

AOCL_GEMM_MATMUL(bfloat16, bfloat16, bfloat16, float, bf16bf16f32obf16);
AOCL_GEMM_MATMUL(bfloat16, bfloat16, float, float, bf16bf16f32of32);
AOCL_GEMM_MATMUL(bfloat16, int8_t, float, float, bf16s4f32of32);
AOCL_GEMM_MATMUL(bfloat16, int8_t, bfloat16, float, bf16s4f32obf16);

AOCL_GEMM_MATMUL(float, float, float, float, f32f32f32of32);

/**
 * @brief Batch GEMM (General Matrix Multiplication) with support for fused
 * post-operations.
 *
 * @param[in] order Array of memory layouts (row-major or column-major).
 * @param[in] transa Array of transpose options for A matrices.
 * @param[in] transb Array of transpose options for B matrices.
 * @param[in] batch_size Number of matrices in the batch.
 * @param[in] m Array of row dimensions for each matrix in the batch.
 * @param[in] n Array of column dimensions for each matrix in the batch.
 * @param[in] k Array of inner dimensions for each matrix in the batch.
 * @param[in] alpha Array of scalar multipliers for the product of matrices A
 * and B.
 * @param[in] a Array of pointers to A matrices.
 * @param[in] lda Array of leading dimensions for A matrices.
 * @param[in] mem_format_a Array of memory formats for A matrices.
 * @param[in] b Array of pointers to B matrices.
 * @param[in] ldb Array of leading dimensions for B matrices.
 * @param[in] mem_format_b Array of memory formats for B matrices.
 * @param[in] beta Array of scalar multipliers for C matrices.
 * @param[out] c Array of pointers to C matrices.
 * @param[in] ldc Array of leading dimensions for C matrices.
 * @param[in] post_op_unparsed Array of pointers to post-operation structures.
 */
#define AOCL_BGEMM_MATMUL(A_type, B_type, C_type, Sum_type, LP_SFX)            \
    DLP_CLASSIC_EXPORT void aocl_batch_gemm_##LP_SFX(                          \
        const char* order, const char* transa, const char* transb,             \
        const md_t* m, const md_t* n, const md_t* k, const Sum_type* alpha,    \
        const A_type** a, const md_t* lda, const B_type** b, const md_t* ldb,  \
        const Sum_type* beta, C_type** c, const md_t* ldc,                     \
        const md_t group_count, const md_t* group_size,                        \
        const char* mem_format_a, const char* mem_format_b,                    \
        aocl_post_op** post_op_unparsed)

// bf16 APIs
AOCL_BGEMM_MATMUL(bfloat16, bfloat16, float, float, bf16bf16f32of32);
AOCL_BGEMM_MATMUL(bfloat16, bfloat16, bfloat16, float, bf16bf16f32obf16);
AOCL_BGEMM_MATMUL(bfloat16, int8_t, float, float, bf16s4f32of32);
AOCL_BGEMM_MATMUL(bfloat16, int8_t, bfloat16, float, bf16s4f32obf16);
// f32 APIs
AOCL_BGEMM_MATMUL(float, float, float, float, f32f32f32of32);
// u8s8 APIs
AOCL_BGEMM_MATMUL(uint8_t, int8_t, int32_t, int32_t, u8s8s32os32);
AOCL_BGEMM_MATMUL(uint8_t, int8_t, int8_t, int32_t, u8s8s32os8);
AOCL_BGEMM_MATMUL(uint8_t, int8_t, float, int32_t, u8s8s32of32);
AOCL_BGEMM_MATMUL(uint8_t, int8_t, bfloat16, int32_t, u8s8s32obf16);
AOCL_BGEMM_MATMUL(uint8_t, int8_t, uint8_t, int32_t, u8s8s32ou8);
// s8s8 APIs
AOCL_BGEMM_MATMUL(int8_t, int8_t, int32_t, int32_t, s8s8s32os32);
AOCL_BGEMM_MATMUL(int8_t, int8_t, int8_t, int32_t, s8s8s32os8);
AOCL_BGEMM_MATMUL(int8_t, int8_t, float, int32_t, s8s8s32of32);
AOCL_BGEMM_MATMUL(int8_t, int8_t, bfloat16, int32_t, s8s8s32obf16);
AOCL_BGEMM_MATMUL(int8_t, int8_t, uint8_t, int32_t, s8s8s32ou8);

#endif // AOCL_GEMM_INTERFACE_H
