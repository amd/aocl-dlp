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

#include "adaptors/ref/ual_ref.hh"
#include "adaptors/ref/gemm_ref.hh"
#include "adaptors/ref/operation_ref.hh"
#include "framework/operation.hh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <variant>

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
    } else if (input_type == MatrixType::bf16) {
        const bfloat16* src_data =
            reinterpret_cast<const bfloat16*>(in.getData());
        bfloat16* dst_data = reinterpret_cast<bfloat16*>(out.getData());

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
                            < in.getDataSizeBytes() / sizeof(bfloat16)
                        && static_cast<size_t>(dst_idx)
                               < out.getDataSizeBytes() / sizeof(bfloat16)) {
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
UalRef::gemm(const Matrix& A,
             const Matrix& B,
             Matrix&       C,
             MatrixType    accType,
             double        alpha,
             double        beta)
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
                          MatrixType::f32>(): {
            // For f32 operations, alpha/beta are float type
            float alpha_f32 = static_cast<float>(alpha);
            float beta_f32  = static_cast<float>(beta);

            dlp::testing::classic::ref::aocl_gemm_f32f32f32of32_ref(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_f32,
                reinterpret_cast<const float*>(
                    A.getMatrixData().getMatrixPtr()),
                static_cast<int>(A.getLeadingDimension()),
                reinterpret_cast<const float*>(
                    B.getMatrixData().getMatrixPtr()),
                static_cast<int>(B.getLeadingDimension()), beta_f32,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                static_cast<int>(C.getLeadingDimension()), nullptr);

            return true;
        }

        case encode_types<MatrixType::bf16, MatrixType::bf16, MatrixType::f32,
                          MatrixType::f32>(): {
            float alpha_f32 = static_cast<float>(alpha);
            float beta_f32  = static_cast<float>(beta);

            dlp::testing::classic::ref::aocl_gemm_bf16bf16f32of32_ref(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_f32,
                reinterpret_cast<const bfloat16*>(
                    A.getMatrixData().getMatrixPtr()),
                static_cast<int>(A.getLeadingDimension()),
                reinterpret_cast<const bfloat16*>(
                    B.getMatrixData().getMatrixPtr()),
                static_cast<int>(B.getLeadingDimension()), beta_f32,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                static_cast<int>(C.getLeadingDimension()), nullptr);

            return true;
        }

        case encode_types<MatrixType::bf16, MatrixType::bf16, MatrixType::bf16,
                          MatrixType::f32>(): {
            float alpha_f32 = static_cast<float>(alpha);
            float beta_f32  = static_cast<float>(beta);

            dlp::testing::classic::ref::aocl_gemm_bf16bf16f32obf16_ref(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_f32,
                reinterpret_cast<const bfloat16*>(
                    A.getMatrixData().getMatrixPtr()),
                static_cast<int>(A.getLeadingDimension()),
                reinterpret_cast<const bfloat16*>(
                    B.getMatrixData().getMatrixPtr()),
                static_cast<int>(B.getLeadingDimension()), beta_f32,
                reinterpret_cast<bfloat16*>(C.getMatrixData().getMatrixPtr()),
                static_cast<int>(C.getLeadingDimension()), nullptr);

            return true;
        }

        default:
            return false;
    }
}

/**
 * @brief Apply element-wise post-operation to a matrix
 */
template<>
void
UalRef::applyPostOperation(Matrix& matrix,
                           const dlp::testing::framework::ElementWiseParam& op)
{
    switch (op.getOperation()) {
        case dlp::testing::framework::ElementWiseOperation::Relu:
            applyRelu(matrix);
            break;
        case dlp::testing::framework::ElementWiseOperation::Prelu:
            applyPrelu(matrix, op.getAlpha());
            break;
        case dlp::testing::framework::ElementWiseOperation::Gelu_Tanh:
            applyGeluTanh(matrix);
            break;
        case dlp::testing::framework::ElementWiseOperation::Gelu_Erf:
            applyGeluErf(matrix);
            break;
        case dlp::testing::framework::ElementWiseOperation::Clip:
            applyClip(matrix, op.getAlpha(), op.getBeta());
            break;
        case dlp::testing::framework::ElementWiseOperation::Swish:
            applySwish(matrix);
            break;
        case dlp::testing::framework::ElementWiseOperation::Tanh:
            applyTanh(matrix);
            break;
        case dlp::testing::framework::ElementWiseOperation::Sigmoid:
            applySigmoid(matrix);
            break;
        default:
            break;
    }
}

/**
 * @brief Apply sum/scale post-operation to a matrix
 */
template<>
void
UalRef::applyPostOperation(Matrix&                                  matrix,
                           const dlp::testing::framework::SumParam& op)
{
    switch (op.getOperation()) {
        case dlp::testing::framework::SumOperation::Sum:
            applySum(matrix, op.getZeroPoint());
            break;
        case dlp::testing::framework::SumOperation::Scale:
            applyScale(matrix, op.getScaleFactor());
            break;
        default:
            break;
    }
}

/**
 * @brief Apply bias post-operation to a matrix
 */
template<>
void
UalRef::applyPostOperation(Matrix&                                   matrix,
                           const dlp::testing::framework::BiasParam& op)
{
    applyBias(matrix, op.getBias());
}

/**
 * @brief Apply matrix add post-operation to a matrix
 */
template<>
void
UalRef::applyPostOperation(Matrix& matrix,
                           const dlp::testing::framework::MatrixAddParam& op)
{
    applyMatrixAdd(matrix, op.getMatrix(), op.getScaleFactor());
}

/**
 * @brief Apply matrix multiply post-operation to a matrix
 */
template<>
void
UalRef::applyPostOperation(Matrix& matrix,
                           const dlp::testing::framework::MatrixMulParam& op)
{
    applyMatrixMul(matrix, op.getMatrix(), op.getScaleFactor());
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
UalRef::gemm(const Matrix&                      A,
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

    // Validate that the postOps are for REF backend
    if (postOps->getUALType() != UALType::REF) {
        return false;
    }

    // Cast to RefOperation and use the iterator pattern
    const auto* refOp = dynamic_cast<const RefOperation*>(postOps.get());
    if (!refOp) {
        return false;
    }

    // First perform the GEMM operation
    bool result = gemm(A, B, C, accType, alpha, beta);
    if (!result) {
        return false;
    }

    // Apply post-operations using the iterator pattern
    refOp->resetIterator();
    while (refOp->hasNextPostOp()) {
        auto postOp = refOp->getNextPostOp();
        if (postOp) {
            // Process the post-operation using std::visit
            std::visit([&](const auto& op) { applyPostOperation(C, op); },
                       *postOp);
        }
    }

    return true;
}

bool
UalRef::gemm(md_t         m,
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
    // For now, this is a stub implementation that returns false
    // indicating the operation is not supported in the reference implementation
    // TODO: Implement reference GEMM logic if needed for benchmarking
    return false;
}

// Helper methods for specific operations
void
UalRef::applyRelu(Matrix& matrix)
{
    if (matrix.getMatrixType() == MatrixType::f32) {
        float* data = reinterpret_cast<float*>(matrix.getData());
        size_t size = matrix.getDataSizeBytes() / sizeof(float);

        for (size_t i = 0; i < size; ++i) {
            data[i] = std::max(0.0f, data[i]);
        }
    }
}

void
UalRef::applyPrelu(Matrix& matrix, const Matrix* alpha)
{
    if (matrix.getMatrixType() == MatrixType::f32) {
        float* data = reinterpret_cast<float*>(matrix.getData());
        size_t size = matrix.getDataSizeBytes() / sizeof(float);

        float alpha_val = 0.01f; // Default alpha
        if (alpha && alpha->getMatrixType() == MatrixType::f32) {
            alpha_val = *reinterpret_cast<const float*>(alpha->getData());
        }

        for (size_t i = 0; i < size; ++i) {
            if (data[i] < 0.0f) {
                data[i] *= alpha_val;
            }
        }
    }
}

void
UalRef::applyGeluTanh(Matrix& matrix)
{
    if (matrix.getMatrixType() == MatrixType::f32) {
        float* data = reinterpret_cast<float*>(matrix.getData());
        size_t size = matrix.getDataSizeBytes() / sizeof(float);

        for (size_t i = 0; i < size; ++i) {
            float x = data[i];
            float tanh_arg =
                std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x);
            data[i] = 0.5f * x * (1.0f + std::tanh(tanh_arg));
        }
    }
}

void
UalRef::applyGeluErf(Matrix& matrix)
{
    if (matrix.getMatrixType() == MatrixType::f32) {
        float* data = reinterpret_cast<float*>(matrix.getData());
        size_t size = matrix.getDataSizeBytes() / sizeof(float);

        for (size_t i = 0; i < size; ++i) {
            float x = data[i];
            data[i] = 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
        }
    }
}

void
UalRef::applyClip(Matrix& matrix, const Matrix* lower, const Matrix* upper)
{
    if (matrix.getMatrixType() == MatrixType::f32) {
        float* data = reinterpret_cast<float*>(matrix.getData());
        size_t size = matrix.getDataSizeBytes() / sizeof(float);

        float lower_val = -6.0f; // Default lower bound
        float upper_val = 6.0f;  // Default upper bound

        if (lower && lower->getMatrixType() == MatrixType::f32) {
            lower_val = *reinterpret_cast<const float*>(lower->getData());
        }
        if (upper && upper->getMatrixType() == MatrixType::f32) {
            upper_val = *reinterpret_cast<const float*>(upper->getData());
        }

        for (size_t i = 0; i < size; ++i) {
            data[i] = std::max(lower_val, std::min(upper_val, data[i]));
        }
    }
}

void
UalRef::applySwish(Matrix& matrix)
{
    if (matrix.getMatrixType() == MatrixType::f32) {
        float* data = reinterpret_cast<float*>(matrix.getData());
        size_t size = matrix.getDataSizeBytes() / sizeof(float);

        for (size_t i = 0; i < size; ++i) {
            float x = data[i];
            data[i] = x / (1.0f + std::exp(-x));
        }
    }
}

void
UalRef::applyTanh(Matrix& matrix)
{
    if (matrix.getMatrixType() == MatrixType::f32) {
        float* data = reinterpret_cast<float*>(matrix.getData());
        size_t size = matrix.getDataSizeBytes() / sizeof(float);

        for (size_t i = 0; i < size; ++i) {
            data[i] = std::tanh(data[i]);
        }
    }
}

void
UalRef::applySigmoid(Matrix& matrix)
{
    if (matrix.getMatrixType() == MatrixType::f32) {
        float* data = reinterpret_cast<float*>(matrix.getData());
        size_t size = matrix.getDataSizeBytes() / sizeof(float);

        for (size_t i = 0; i < size; ++i) {
            data[i] = 1.0f / (1.0f + std::exp(-data[i]));
        }
    }
}

void
UalRef::applySum(Matrix& matrix, const Matrix* zeroPoint)
{
    // For sum operations, we would typically add another matrix
    // For now, this is a placeholder implementation
}

void
UalRef::applyScale(Matrix& matrix, const Matrix* scaleFactor)
{
    if (matrix.getMatrixType() == MatrixType::f32 && scaleFactor
        && scaleFactor->getMatrixType() == MatrixType::f32) {
        float* data = reinterpret_cast<float*>(matrix.getData());
        size_t size = matrix.getDataSizeBytes() / sizeof(float);

        float scale = *reinterpret_cast<const float*>(scaleFactor->getData());

        for (size_t i = 0; i < size; ++i) {
            data[i] *= scale;
        }
    }
}

void
UalRef::applyBias(Matrix& matrix, const Matrix& bias)
{
    if (matrix.getMatrixType() == MatrixType::f32
        && bias.getMatrixType() == MatrixType::f32) {
        float*       data      = reinterpret_cast<float*>(matrix.getData());
        const float* bias_data = reinterpret_cast<const float*>(bias.getData());

        md_t rows      = matrix.getRows();
        md_t cols      = matrix.getCols();
        md_t bias_size = bias.getRows() * bias.getCols();

        // Apply bias to each row/column depending on bias size
        for (md_t i = 0; i < rows; ++i) {
            for (md_t j = 0; j < cols; ++j) {
                size_t idx      = i * matrix.getLeadingDimension() + j;
                size_t bias_idx = j % bias_size; // Broadcast bias
                data[idx] += bias_data[bias_idx];
            }
        }
    }
}

void
UalRef::applyMatrixAdd(Matrix&       matrix,
                       const Matrix& addMatrix,
                       const Matrix* scaleFactor)
{
    if (matrix.getMatrixType() == MatrixType::f32
        && addMatrix.getMatrixType() == MatrixType::f32) {
        float*       data = reinterpret_cast<float*>(matrix.getData());
        const float* add_data =
            reinterpret_cast<const float*>(addMatrix.getData());

        float scale = 1.0f;
        if (scaleFactor && scaleFactor->getMatrixType() == MatrixType::f32) {
            scale = *reinterpret_cast<const float*>(scaleFactor->getData());
        }

        size_t size =
            std::min(matrix.getDataSizeBytes(), addMatrix.getDataSizeBytes())
            / sizeof(float);

        for (size_t i = 0; i < size; ++i) {
            data[i] += add_data[i] * scale;
        }
    }
}

void
UalRef::applyMatrixMul(Matrix&       matrix,
                       const Matrix& mulMatrix,
                       const Matrix* scaleFactor)
{
    // Matrix multiplication is more complex and would require proper GEMM
    // For now, just element-wise multiplication
    if (matrix.getMatrixType() == MatrixType::f32
        && mulMatrix.getMatrixType() == MatrixType::f32) {
        float*       data = reinterpret_cast<float*>(matrix.getData());
        const float* mul_data =
            reinterpret_cast<const float*>(mulMatrix.getData());

        float scale = 1.0f;
        if (scaleFactor && scaleFactor->getMatrixType() == MatrixType::f32) {
            scale = *reinterpret_cast<const float*>(scaleFactor->getData());
        }

        size_t size =
            std::min(matrix.getDataSizeBytes(), mulMatrix.getDataSizeBytes())
            / sizeof(float);

        for (size_t i = 0; i < size; ++i) {
            data[i] *= mul_data[i] * scale;
        }
    }
}

} // namespace dlp::testing::classic
