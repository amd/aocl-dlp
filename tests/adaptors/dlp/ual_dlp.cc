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

/**
 * @file ual_dlp.cc
 * @brief Implementation of the DLP Unified Abstraction Layer
 *
 * This file contains the implementation of the DLP-based UAL, providing
 * optimized matrix operations with support for various data types, memory
 * layouts, and virtual transposition.
 */

#include "adaptors/dlp/ual_dlp.hh"
#include "adaptors/dlp/operation_dlp.hh"
#include <cstddef>
#include <iostream>

#include "aocl_dlp.h"

using namespace dlp::testing::framework;

namespace dlp::testing::classic {

/**
 * @brief Constructor for UalDlp
 *
 * Initializes a DLP-based Unified Abstraction Layer implementation.
 */
UalDlp::UalDlp()
    : IUal(UALType::DLP)
{
}

/**
 * @brief Get the UAL implementation type
 *
 * @return UALType::DLP for this implementation
 */
UALType
UalDlp::getUALType() const
{
    return UALType::DLP;
}

/**
 * @brief Convert UAL type to human-readable string
 *
 * @param type The UAL type to convert
 * @return std::string Human-readable description
 */
std::string
UalDlp::toString(UALType type)
{
    switch (type) {
        case UALType::DLP:
            return "Deep Learning Primitives";
        case UALType::MKL:
            return "Intel MKL";
        case UALType::ONEDNN:
            return "OneDNN";
        case UALType::REF:
            return "Reference";
        default:
            return "Unknown UAL";
    }
}

/**
 * @brief Public reorder interface that unpacks Matrix object
 *
 * @param in Input matrix to reorder
 * @param out Output matrix to store reordered data
 * @param A_type Type of matrix A in GEMM context
 * @param B_type Type of matrix B in GEMM context
 * @param C_type Type of matrix C in GEMM context
 * @param accType Target accumulation type
 * @return bool Success status
 */
bool
UalDlp::reorder(const Matrix& in,
                Matrix&       out,
                MatrixType    A_type,
                MatrixType    B_type,
                MatrixType    C_type,
                MatrixType    accType)
{

    // Use effective (logical) dimensions for reordering
    md_t effective_rows = in.getEffectiveRows();
    md_t effective_cols = in.getEffectiveCols();

    // Determine appropriate reorder function based on input type and GEMM
    // context The A, B, C types provide context for optimal reordering strategy
    md_t alloc_bytes = 0;

    // Select reorder function based on input matrix type and GEMM context
    if (in.getMatrixType() == MatrixType::f32) {
        // For mixed precision scenarios, we might need different handling
        if (A_type == MatrixType::f32 && B_type == MatrixType::f32
            && C_type == MatrixType::f32) {
            alloc_bytes = aocl_get_reorder_buf_size_f32f32f32of32(
                in.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c',
                in.isTransposed() ? 't' : 'n', 'B', effective_rows,
                effective_cols, nullptr);
        } else {
            // Handle mixed precision cases - for now, fall back to standard f32
            // reorder
            alloc_bytes = aocl_get_reorder_buf_size_f32f32f32of32(
                in.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c',
                in.isTransposed() ? 't' : 'n', 'B', effective_rows,
                effective_cols, nullptr);
        }
    } else if (in.getMatrixType() == MatrixType::bf16) {
        // For bf16, consider the accumulation type and output type
        if (accType == MatrixType::f32) {
            alloc_bytes = aocl_get_reorder_buf_size_bf16bf16f32of32(
                in.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c',
                in.isTransposed() ? 't' : 'n', 'B', effective_rows,
                effective_cols, nullptr);
        } else {
            // Handle other accumulation types - for now, fall back to standard
            alloc_bytes = aocl_get_reorder_buf_size_bf16bf16f32of32(
                in.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c',
                in.isTransposed() ? 't' : 'n', 'B', effective_rows,
                effective_cols, nullptr);
        }
    } else {
        return false;
    }

    // Create output matrix with dimensions that preserve the logical operation
    // If input was transposed, the reordered output should have physical
    // dimensions that match the effective dimensions (since reordering handles
    // transposition)
    md_t out_rows       = effective_rows;
    md_t out_cols       = effective_cols;
    md_t out_leadingDim = in.getLayout() == MatrixLayout::ROW_MAJOR ? out_cols
                                                                    : out_rows;

    // Create output matrix using the new interface with external memory
    // allocation
    // Leading dimension needs to be recomputed as the matrix is copied.
    auto memory = MatrixMemory::allocateBytes(alloc_bytes);
    out = Matrix(out_rows, out_cols, in.getMatrixType(), std::move(memory),
                 alloc_bytes, in.getLayout(), out_leadingDim, false, true);

    char layout = in.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c';

    // Use the type information to select appropriate reorder function
    switch (in.getMatrixType()) {
        case MatrixType::f32:
            // Consider GEMM context for optimal reordering
            if (A_type == MatrixType::f32 && B_type == MatrixType::f32
                && C_type == MatrixType::f32) {
                aocl_reorder_f32f32f32of32(
                    layout, in.isTransposed() ? 't' : 'n', 'B',
                    reinterpret_cast<const float*>(
                        in.getMatrixData().getMatrixPtr()),
                    reinterpret_cast<float*>(
                        out.getMatrixData().getMatrixPtr()),
                    effective_rows, effective_cols, in.getLeadingDimension(),
                    nullptr);
            } else {
                // Handle mixed precision - for now, use standard f32 reorder
                aocl_reorder_f32f32f32of32(
                    layout, in.isTransposed() ? 't' : 'n', 'B',
                    reinterpret_cast<const float*>(
                        in.getMatrixData().getMatrixPtr()),
                    reinterpret_cast<float*>(
                        out.getMatrixData().getMatrixPtr()),
                    effective_rows, effective_cols, in.getLeadingDimension(),
                    nullptr);
            }
            break;
        case MatrixType::bf16:
            // Consider accumulation type for bf16 reordering
            aocl_reorder_bf16bf16f32of32(
                layout, in.isTransposed() ? 't' : 'n', 'B',
                reinterpret_cast<const bfloat16*>(
                    in.getMatrixData().getMatrixPtr()),
                reinterpret_cast<bfloat16*>(out.getMatrixData().getMatrixPtr()),
                effective_rows, effective_cols, in.getLeadingDimension(), NULL);
            break;
        case MatrixType::s8:
            aocl_reorder_s8s8s32os32(
                layout, in.isTransposed() ? 't' : 'n', 'B',
                reinterpret_cast<const int8_t*>(
                    in.getMatrixData().getMatrixPtr()),
                reinterpret_cast<int8_t*>(out.getMatrixData().getMatrixPtr()),
                effective_rows, effective_cols, in.getLeadingDimension(), NULL);
            break;
        default:
            return false;
    }

    return true;
}

/**
 * @brief Validate GEMM parameters for correctness
 *
 * This function follows the exact logic from AOCL_GEMM_CHECK macro in
 * aocl_gemm_check.h
 *
 * @param A First input matrix
 * @param B Second input matrix
 * @param C Output matrix
 * @return bool True if parameters are valid, false otherwise
 */
bool
UalDlp::checkValidGemmParams(const Matrix& A, const Matrix& B, const Matrix& C)
{
    // Get GEMM operation dimensions (logical dimensions)
    uint32_t m = A.getEffectiveRows(); // Rows of A (and C)
    uint32_t n = B.getEffectiveCols(); // Cols of B (and C)
    uint32_t k = A.getEffectiveCols(); // Cols of A, Rows of B

    bool col_stored = (A.getLayout() == MatrixLayout::COLUMN_MAJOR);
    bool row_stored = (A.getLayout() == MatrixLayout::ROW_MAJOR);

    bool nota = !A.isTransposed(); // not transposed A
    bool notb = !B.isTransposed(); // not transposed B
    bool ta   = A.isTransposed();  // transposed A
    bool tb   = B.isTransposed();  // transposed B

    // All matrices should have the same layout
    if (A.getLayout() != B.getLayout() || A.getLayout() != C.getLayout()) {
        return false;
    }

    // Check basic dimensions - must be positive (same as macro: m <= 0, n <= 0,
    // k <= 0)
    if (m <= 0) {
        return false; // info = 4 in macro
    }
    if (n <= 0) {
        return false; // info = 5 in macro
    }
    if (k <= 0) {
        return false; // info = 6 in macro
    }

    // Check dimension compatibility for matrix multiplication
    if (A.getEffectiveCols() != B.getEffectiveRows()) {
        return false;
    }

    // Matrix A leading dimension checks (info = 9 in macro)
    // For reordered matrices, skip the leading dimension check as they have
    // custom layouts
    if (!A.isReordered()) {
        if (row_stored
            && ((nota && (A.getLeadingDimension() < k))
                || (ta && (A.getLeadingDimension() < m)))) {
            return false;
        }
        if (col_stored
            && ((nota && (A.getLeadingDimension() < m))
                || (ta && (A.getLeadingDimension() < k)))) {
            return false;
        }
    }

    // Matrix B leading dimension checks (info = 12 in macro)
    // For reordered matrices, skip the leading dimension check as they have
    // custom layouts
    if (!B.isReordered()) {
        if (row_stored
            && ((notb && (B.getLeadingDimension() < n))
                || (tb && (B.getLeadingDimension() < k)))) {
            return false;
        }
        if (col_stored
            && ((notb && (B.getLeadingDimension() < k))
                || (tb && (B.getLeadingDimension() < n)))) {
            return false;
        }
    }

    // Matrix C leading dimension checks (info = 16 in macro)
    // C is never reordered, so always check
    if (row_stored && (C.getLeadingDimension() < n)) {
        return false;
    }
    if (col_stored && (C.getLeadingDimension() < m)) {
        return false;
    }

    return true;
}

// Depriciated function
/**
 * @brief Public GEMM interface that unpacks Matrix objects
 *
 * @param A First input matrix
 * @param B Second input matrix
 * @param C Output matrix
 * @param accType Accumulation type
 * @param alpha Scaling factor for A*B
 * @param beta Scaling factor for C
 * @return bool Success status
 */
bool
UalDlp::gemm(const Matrix& A,
             const Matrix& B,
             Matrix&       C,
             MatrixType    accType,
             double        alpha,
             double        beta)
{

    // Validate parameters first
    // NOTE: This client-side validation is not ideal - proper error handling
    // should be implemented at the library level to provide consistent
    // parameter validation across all UAL implementations.
    if (!checkValidGemmParams(A, B, C)) {
        return false;
    }

    uint64_t type = encode_types(A.getMatrixType(), B.getMatrixType(),
                                 C.getMatrixType(), accType);

    char transA = A.isTransposed() ? 't' : 'n';
    char transB = B.isReordered() ? 'n' : (B.isTransposed() ? 't' : 'n');

    char layoutA = A.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c';

    // Determine memory format for matrices A and B
    // Pack takes precedence over reorder since it includes optimization
    char memFormatA = A.isPacked() ? 'p' : (A.isReordered() ? 'r' : 'n');
    char memFormatB = B.isPacked() ? 'p' : (B.isReordered() ? 'r' : 'n');

    // Validate leading dimensions
    if (C.getLayout() == MatrixLayout::ROW_MAJOR) {
        if (C.getLeadingDimension() < C.getCols()) {
            return false;
        }
    } else {
        if (C.getLeadingDimension() < C.getRows()) {
            return false;
        }
    }

    switch (type) {
        case encode_types<MatrixType::f32, MatrixType::f32, MatrixType::f32,
                          MatrixType::f32>(): {
            // For f32 operations, alpha/beta are float type
            float alpha_f32 = static_cast<float>(alpha);
            float beta_f32  = static_cast<float>(beta);

            aocl_gemm_f32f32f32of32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_f32,
                reinterpret_cast<float*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<float*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_f32,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), nullptr);

            return true;
        }

        case encode_types<MatrixType::bf16, MatrixType::bf16, MatrixType::f32,
                          MatrixType::f32>(): {
            float alpha_f32 = static_cast<float>(alpha);
            float beta_f32  = static_cast<float>(beta);

            aocl_gemm_bf16bf16f32of32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_f32,
                reinterpret_cast<bfloat16*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<bfloat16*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_f32,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), nullptr);

            return true;
        }
        // Int8 GEMM cases
        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::s32,
                          MatrixType::s32>(): {
            // uint8 × int8 → int32 directly supported
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_u8s8s32os32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<uint8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<int32_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), nullptr);

            return true;
        }

        case encode_types<MatrixType::bf16, MatrixType::bf16, MatrixType::bf16,
                          MatrixType::f32>(): {
            float alpha_f32 = static_cast<float>(alpha);
            float beta_f32  = static_cast<float>(beta);

            aocl_gemm_bf16bf16f32obf16(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_f32,
                reinterpret_cast<bfloat16*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<bfloat16*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_f32,
                reinterpret_cast<bfloat16*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), nullptr);

            return true;
        }
        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::s32,
                          MatrixType::s32>(): {
            // int8 × int8 → int32 directly supported
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32os32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<int8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<int32_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), nullptr);

            return true;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::f32,
                          MatrixType::s32>(): {
            // uint8 × int8 → int32 directly supported
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_u8s8s32of32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<uint8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), nullptr);

            return true;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::s8,
                          MatrixType::s32>(): {
            // uint8 × int8 → int32 directly supported
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_u8s8s32os8(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<uint8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<int8_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), nullptr);

            return true;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::u8,
                          MatrixType::s32>(): {
            // uint8 × int8 → int32 directly supported
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_u8s8s32os8(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<uint8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<int8_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), nullptr);

            return true;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::bf16,
                          MatrixType::s32>(): {
            // uint8 × int8 → bf16 directly supported
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_u8s8s32obf16(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<uint8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<bfloat16*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), nullptr);

            return true;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::f32,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32of32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<int8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), nullptr);

            return true;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::s8,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32os8(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<int8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<int8_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), nullptr);

            return true;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::u8,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32ou8(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<int8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<uint8_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), nullptr);

            return true;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::bf16,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32obf16(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<int8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<bfloat16*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), nullptr);

            return true;
        }

        default:
            return false;
    }
}

/**
 * @brief Perform general matrix multiplication with post-operations: C =
 * alpha*A*B + beta*C + PostOps
 *
 * @param A First input matrix
 * @param B Second input matrix
 * @param C Output matrix
 * @param accType Accumulation type
 * @param postOps Post-operations to apply (nullptr for no post-ops)
 * @param alpha Scaling factor for A*B
 * @param beta Scaling factor for C
 * @return bool Success status
 */
bool
UalDlp::gemm(const Matrix&                      A,
             const Matrix&                      B,
             Matrix&                            C,
             MatrixType                         accType,
             const std::shared_ptr<IOperation>& postOps,
             double                             alpha,
             double                             beta)
{
    // For now, if no postOps are provided, delegate to the original gemm method
    if (!postOps) {
        return gemm(A, B, C, accType, alpha, beta);
    }

    // Validate that the postOps are for DLP backend
    if (postOps->getUALType() != UALType::DLP) {
        return false;
    }

    // Cast to DlpOperation and access the serialized dlp_metadata_t structure
    // using friend access
    const auto* dlpOp = dynamic_cast<const DlpOperation*>(postOps.get());
    if (!dlpOp) {
        return false;
    }

    // Get the serialized dlp_metadata_t structure using friend access
    dlp_metadata_t* aocl_postops = dlpOp->toAoclPostOp();

    // Validate parameters first
    if (!checkValidGemmParams(A, B, C)) {
        return false;
    }

    uint64_t type = encode_types(A.getMatrixType(), B.getMatrixType(),
                                 C.getMatrixType(), accType);

    char transA = A.isTransposed() ? 't' : 'n';
    char transB = B.isReordered() ? 'n' : (B.isTransposed() ? 't' : 'n');

    char layoutA = A.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c';

    // Determine memory format for matrices A and B
    // Pack takes precedence over reorder since it includes optimization
    char memFormatA = A.isPacked() ? 'p' : (A.isReordered() ? 'r' : 'n');
    char memFormatB = B.isPacked() ? 'p' : (B.isReordered() ? 'r' : 'n');

    // Validate leading dimensions
    if (C.getLayout() == MatrixLayout::ROW_MAJOR) {
        if (C.getLeadingDimension() < C.getCols()) {
            return false;
        }
    } else {
        if (C.getLeadingDimension() < C.getRows()) {
            return false;
        }
    }

    switch (type) {
        case encode_types<MatrixType::f32, MatrixType::f32, MatrixType::f32,
                          MatrixType::f32>(): {
            // For f32 operations, alpha/beta are float type
            float alpha_f32 = static_cast<float>(alpha);
            float beta_f32  = static_cast<float>(beta);

            aocl_gemm_f32f32f32of32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_f32,
                reinterpret_cast<float*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<float*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_f32,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops);

            return true;
        }

        case encode_types<MatrixType::bf16, MatrixType::bf16, MatrixType::f32,
                          MatrixType::f32>(): {
            float alpha_f32 = static_cast<float>(alpha);
            float beta_f32  = static_cast<float>(beta);

            aocl_gemm_bf16bf16f32of32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_f32,
                reinterpret_cast<bfloat16*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<bfloat16*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_f32,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops);

            return true;
        }

        case encode_types<MatrixType::bf16, MatrixType::bf16, MatrixType::bf16,
                          MatrixType::f32>(): {
            float alpha_f32 = static_cast<float>(alpha);
            float beta_f32  = static_cast<float>(beta);

            aocl_gemm_bf16bf16f32obf16(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_f32,
                reinterpret_cast<bfloat16*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<bfloat16*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_f32,
                reinterpret_cast<bfloat16*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops);

            return true;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::s32,
                          MatrixType::s32>(): {
            // For u8/s8 operations, alpha/beta are int32_t type
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);
            aocl_gemm_u8s8s32os32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<uint8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<int32_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops);
            return true;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::s32,
                          MatrixType::s32>(): {
            // For s8/s8 operations, alpha/beta are int32_t type
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32os32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<int8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<int32_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops);

            return true;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::s8,
                          MatrixType::s32>(): {
            // For u8/s8 operations, alpha/beta are int32_t type
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_u8s8s32os8(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<uint8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<int8_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops);

            return true;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::f32,
                          MatrixType::s32>(): {
            // For u8/s8 operations, alpha/beta are int32_t type
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_u8s8s32of32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<uint8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops);

            return true;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::u8,
                          MatrixType::s32>(): {
            // uint8 × int8 → uint8 directly supported
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_u8s8s32ou8(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<uint8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<uint8_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops);

            return true;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::bf16,
                          MatrixType::s32>(): {
            // uint8 × int8 → bf16 directly supported
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_u8s8s32obf16(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<uint8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<bfloat16*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops);

            return true;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::f32,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32of32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<int8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops);

            return true;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::s8,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32os8(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<int8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<int8_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops);

            return true;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::u8,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32ou8(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<int8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<uint8_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops);

            return true;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::bf16,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32obf16(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<int8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<bfloat16*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops);

            return true;
        }

        default:
            return false;
    }
}

bool
UalDlp::gemm(md_t         m,
             md_t         n,
             md_t         k,
             void*        matA,
             MatrixType   matA_type,
             MatrixLayout matA_layout,
             bool         matA_transposed,
             md_t         matA_leadingDim,
             void*        matB,
             MatrixType   matB_type,
             MatrixLayout matB_layout,
             bool         matB_transposed,
             md_t         matB_leadingDim,
             void*        matC,
             MatrixType   matC_type,
             MatrixLayout matC_layout,
             bool         matC_transposed,
             md_t         matC_leadingDim,
             MatrixType   accType,
             double       alpha,
             double       beta) const
{
    uint64_t type = encode_types(matA_type, matB_type, matC_type, accType);

    char transA  = matA_transposed ? 't' : 'n';
    char transB  = matB_transposed ? 't' : 'n';
    char layoutA = matA_layout == MatrixLayout::ROW_MAJOR ? 'r' : 'c';

    // For benchmarking, assume normal memory format (no packing/reordering)
    char memFormatA = 'n';
    char memFormatB = 'n';

    switch (type) {
        case encode_types<MatrixType::f32, MatrixType::f32, MatrixType::f32,
                          MatrixType::f32>(): {
            // For f32 operations, alpha/beta are float type
            float alpha_f32 = static_cast<float>(alpha);
            float beta_f32  = static_cast<float>(beta);

            aocl_gemm_f32f32f32of32(
                layoutA, transA, transB, m, n, k, alpha_f32,
                reinterpret_cast<float*>(matA), matA_leadingDim, memFormatA,
                reinterpret_cast<float*>(matB), matB_leadingDim, memFormatB,
                beta_f32, reinterpret_cast<float*>(matC), matC_leadingDim,
                nullptr);

            return true;
        }

        default:
            return false;
    }
}

} // namespace dlp::testing::classic
