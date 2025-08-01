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
 * @param accType Target accumulation type
 * @return bool Success status
 */
bool
UalDlp::reorder(const Matrix& in, Matrix& out, MatrixType accType)
{

    // Use effective (logical) dimensions for reordering
    md_t effective_rows = in.getEffectiveRows();
    md_t effective_cols = in.getEffectiveCols();

    md_t alloc_bytes = aocl_get_reorder_buf_size_f32f32f32of32(
        in.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c',
        in.isTransposed() ? 't' : 'n', 'B', effective_rows, effective_cols);

    // Create output matrix with dimensions that preserve the logical operation
    // If input was transposed, the reordered output should have physical
    // dimensions that match the effective dimensions (since reordering handles
    // transposition)
    md_t out_rows       = effective_rows;
    md_t out_cols       = effective_cols;
    md_t out_leadingDim = in.getLayout() == MatrixLayout::ROW_MAJOR ? out_cols
                                                                    : out_rows;

    // Create out1put matrix using the new interface with external memory
    // allocation
    // Leading dimension needs to be recomputed as the matrix is copied.
    auto memory = MatrixMemory::allocateBytes(alloc_bytes);
    out = Matrix(out_rows, out_cols, in.getMatrixType(), std::move(memory),
                 alloc_bytes, in.getLayout(), out_leadingDim, false, true);

    char layout = in.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c';
    switch (in.getMatrixType()) {
        case MatrixType::f32:
            aocl_reorder_f32f32f32of32(
                layout, in.isTransposed() ? 't' : 'n', 'B',
                reinterpret_cast<const float*>(
                    in.getMatrixData().getMatrixPtr()),
                reinterpret_cast<float*>(out.getMatrixData().getMatrixPtr()),
                effective_rows, effective_cols, in.getLeadingDimension());
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
    // char transC = C.isTransposed() ? 't' : 'n';

    char layoutA = A.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c';
    // char layoutB = B.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c';
    // char layoutC = C.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c';

    // Might have 'p' for packed
    char isAReordered = A.isReordered() ? 'r' : 'n';
    char isBReordered = B.isReordered() ? 'r' : 'n';

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
                A.getLeadingDimension(), isAReordered,
                reinterpret_cast<float*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), isBReordered, beta_f32,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
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

    // Cast to DlpOperation and access the serialized aocl_post_op structure
    // using friend access
    const auto* dlpOp = dynamic_cast<const DlpOperation*>(postOps.get());
    if (!dlpOp) {
        return false;
    }

    // Get the serialized aocl_post_op structure using friend access
    aocl_post_op* aocl_postops = dlpOp->toAoclPostOp();

    // Validate parameters first
    if (!checkValidGemmParams(A, B, C)) {
        return false;
    }

    uint64_t type = encode_types(A.getMatrixType(), B.getMatrixType(),
                                 C.getMatrixType(), accType);

    char transA = A.isTransposed() ? 't' : 'n';
    char transB = B.isReordered() ? 'n' : (B.isTransposed() ? 't' : 'n');

    char layoutA = A.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c';

    char isAReordered = A.isReordered() ? 'r' : 'n';
    char isBReordered = B.isReordered() ? 'r' : 'n';

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
                A.getLeadingDimension(), isAReordered,
                reinterpret_cast<float*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), isBReordered, beta_f32,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops);

            return true;
        }

        default:
            return false;
    }
}

} // namespace dlp::testing::classic
