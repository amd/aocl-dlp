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
 * @file ual_ref.cc
 * @brief Implementation of the Reference Unified Abstraction Layer
 *
 * This file contains the implementation of the reference-based UAL, providing
 * basic matrix operations with support for various data types, memory
 * layouts, and virtual transposition.
 */

#include "framework/ual_ref.hh"
#include "ref/gemm_ref.hh"
#include <iostream>

extern "C"
{
#include "aocl_dlp.h"
}

namespace dlp::testing::classic {

/**
 * @brief Constructor for UalRef
 *
 * Initializes a reference-based Unified Abstraction Layer implementation.
 */
UalRef::UalRef()
    : IUal(UALType::REF)
{
}

/**
 * @brief Get the UAL implementation type
 *
 * @return UALType::REF for this implementation
 */
UALType
UalRef::getUALType() const
{
    return UALType::REF;
}

/**
 * @brief Convert UAL type to human-readable string
 *
 * @param type The UAL type to convert
 * @return std::string Human-readable description
 */
std::string
UalRef::toString(UALType type)
{
    switch (type) {
        case UALType::DLP:
            return "Deep Learning Primitives";
        case UALType::MKL:
            return "Intel MKL";
        case UALType::ONEDNN:
            return "OneDNN";
        case UALType::REF:
            return "Reference Implementation";
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
UalRef::reorder(const Matrix& in, Matrix& out, MatrixType accType)
{
    /*
        Reordering operation in reference is
        1. Apply transpose if needed
        2. Copy the data to a new matrix with the leading dimension same as
             in row major if m x n matrix n is leading dim
             in column major if m x n matrix m is leading dim
        3. Set this new matrix as the output matrix
    */

    // Get input matrix properties
    md_t         input_rows = in.getRows();
    md_t         input_cols = in.getCols();
    MatrixType   input_type = in.getMatrixType();
    MatrixLayout layout     = in.getLayout();
    bool         transposed = in.isTransposed();

    // For reordering, we need to determine the output matrix dimensions
    // If the input matrix is transposed, we need to swap dimensions to get the
    // logical layout
    md_t output_rows, output_cols, min_leading_dim;

    if (transposed) {
        // Swap dimensions to match logical layout (like DLP does)
        output_rows = input_cols;
        output_cols = input_rows;
    } else {
        // Keep same dimensions
        output_rows = input_rows;
        output_cols = input_cols;
    }

    // Calculate minimum required leading dimension for the reordered matrix
    if (layout == MatrixLayout::ROW_MAJOR) {
        // For row-major: leading dimension should be >= number of columns
        min_leading_dim = output_cols;
    } else {
        // For column-major: leading dimension should be >= number of rows
        min_leading_dim = output_rows;
    }

    // Create output matrix with minimum required leading dimension
    // The reordered matrix is no longer transposed (data is physically
    // reordered)
    out = Matrix(output_rows, output_cols, input_type, layout, min_leading_dim,
                 false, // not transposed after reordering
                 true   // mark as reordered
    );

    // Copy data from input to output with proper layout
    if (input_type == MatrixType::f32) {
        const float* src_data = reinterpret_cast<const float*>(in.getData());
        float*       dst_data = reinterpret_cast<float*>(out.getData());

        md_t src_ld = in.getLeadingDimension();
        md_t dst_ld = out.getLeadingDimension();

        // Copy data with proper layout handling using templated helper
        auto copyMatrix = [&](auto getSrcIndex, auto getDstIndex) {
            for (md_t i = 0; i < input_rows; ++i) {
                for (md_t j = 0; j < input_cols; ++j) {
                    auto src_idx = getSrcIndex(i, j, src_ld);
                    auto dst_idx = getDstIndex(i, j, dst_ld);

                    // Bounds check to prevent access violations
                    if (static_cast<size_t>(src_idx)
                            < in.getDataSizeBytes() / sizeof(float)
                        && static_cast<size_t>(dst_idx)
                               < out.getDataSizeBytes() / sizeof(float)) {
                        dst_data[dst_idx] = src_data[src_idx];
                    }
                }
            }
        };

        if (layout == MatrixLayout::ROW_MAJOR) {
            if (transposed) {
                // Input is transposed: src[i][j] represents logical[j][i]
                // Output is not transposed: dst[i][j] represents logical[i][j]
                // So we need: dst[j][i] = src[i][j]
                copyMatrix([](md_t i, md_t j, md_t ld) { return i * ld + j; },
                           [](md_t i, md_t j, md_t ld) { return j * ld + i; });
            } else {
                // Direct copy for non-transposed matrices
                copyMatrix([](md_t i, md_t j, md_t ld) { return i * ld + j; },
                           [](md_t i, md_t j, md_t ld) { return i * ld + j; });
            }
        } else { // COLUMN_MAJOR
            if (transposed) {
                // Input is transposed: src[j][i] represents logical[j][i]
                // Output is not transposed: dst[j][i] represents logical[j][i]
                // So we need: dst[i][j] = src[j][i]
                copyMatrix([](md_t i, md_t j, md_t ld) { return j * ld + i; },
                           [](md_t i, md_t j, md_t ld) { return i * ld + j; });
            } else {
                // Direct copy for non-transposed matrices
                copyMatrix([](md_t i, md_t j, md_t ld) { return j * ld + i; },
                           [](md_t i, md_t j, md_t ld) { return j * ld + i; });
            }
        }
    } else {
        // For other data types, we'd need similar implementations
        // For now, just return false for unsupported types
        return false;
    }

    return true;
}

/**
 * @brief Internal implementation of reorder with direct parameter access
 *
 * @param A Pointer to matrix data
 * @param AType Type of matrix data
 * @param rows Number of rows
 * @param cols Number of columns
 * @param leadingDim Leading dimension
 * @param layout Memory layout
 * @param transposed Whether the matrix is transposed
 * @param accType Target accumulation type
 * @return bool Success status
 */
bool
UalRef::reorder(void*        A,
                MatrixType   AType,
                uint32_t     rows,
                uint32_t     cols,
                uint32_t     leadingDim,
                MatrixLayout layout,
                bool         transposed,
                MatrixType   accType)
{
    // Implementation would depend on reference library calls
    // This is a placeholder
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
UalRef::checkValidGemmParams(const Matrix& A, const Matrix& B, const Matrix& C)
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

/**
 * @brief Public GEMM interface that unpacks Matrix objects
 *
 * @param A First input matrix
 * @param B Second input matrix
 * @param C Output matrix
 * @param accType Accumulation type
 * @return bool Success status
 */
bool
UalRef::gemm(const Matrix& A, const Matrix& B, Matrix& C, MatrixType accType)
{

    // Validate parameters first
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

    switch (type) {
        case encode_types<MatrixType::f32, MatrixType::f32, MatrixType::f32,
                          MatrixType::f32>():

            dlp::testing::classic::ref::aocl_gemm_f32f32f32of32_ref(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), 1.0,
                reinterpret_cast<const float*>(
                    A.getMatrixData().getMatrixPtr()),
                static_cast<int>(A.getLeadingDimension()),
                reinterpret_cast<const float*>(
                    B.getMatrixData().getMatrixPtr()),
                static_cast<int>(B.getLeadingDimension()), 1.0,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                static_cast<int>(C.getLeadingDimension()), nullptr);

            return true;

        default:
            return false;
    }
}
} // namespace dlp::testing::classic
