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
#include "utils/conversion_utils.hh"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <variant>

extern "C"
{
#include "aocl_dlp.h"
}

namespace dlp::testing::classic {

using dlp::testing::framework::UALError;
using dlp::testing::utils::bf16_to_f32;

// Type conversion utilities for multi-datatype support
namespace {

    /**
     * @brief Convert a single value from any MatrixType to float
     * @param src_ptr Pointer to source data
     * @param src_type Source data type
     * @param index Element index
     * @return float Converted value
     */
    float convertToFloat(const void* src_ptr, MatrixType src_type, size_t index)
    {
        switch (src_type) {
            case MatrixType::f32:
                return static_cast<const float*>(src_ptr)[index];
            case MatrixType::u8:
                return static_cast<float>(
                    static_cast<const uint8_t*>(src_ptr)[index]);
            case MatrixType::s8:
                return static_cast<float>(
                    static_cast<const int8_t*>(src_ptr)[index]);
            case MatrixType::u16:
                return static_cast<float>(
                    static_cast<const uint16_t*>(src_ptr)[index]);
            case MatrixType::s16:
                return static_cast<float>(
                    static_cast<const int16_t*>(src_ptr)[index]);
            case MatrixType::u32:
                return static_cast<float>(
                    static_cast<const uint32_t*>(src_ptr)[index]);
            case MatrixType::s32:
                return static_cast<float>(
                    static_cast<const int32_t*>(src_ptr)[index]);
            case MatrixType::u4: {
                const uint8_t* data     = static_cast<const uint8_t*>(src_ptr);
                size_t         byte_idx = index / 2;
                size_t         bit_offset = (index % 2) * 4;
                uint8_t        value = (data[byte_idx] >> bit_offset) & 0x0F;
                return static_cast<float>(value);
            }
            case MatrixType::s4: {
                const uint8_t* data     = static_cast<const uint8_t*>(src_ptr);
                size_t         byte_idx = index / 2;
                size_t         bit_offset = (index % 2) * 4;
                uint8_t        value = (data[byte_idx] >> bit_offset) & 0x0F;
                // Sign extend 4-bit to 8-bit
                if (value & 0x08)
                    value |= 0xF0;
                return static_cast<float>(static_cast<int8_t>(value));
            }
            case MatrixType::bf16: {
                const bfloat16* data = static_cast<const bfloat16*>(src_ptr);
                return bf16_to_f32(data[index]);
            }
            default:
                return 0.0f;
        }
    }

    /**
     * @brief Convert a float value to any MatrixType and store it
     * @param dst_ptr Pointer to destination data
     * @param dst_type Destination data type
     * @param index Element index
     * @param value Float value to convert
     */
    void convertFromFloat(void*      dst_ptr,
                          MatrixType dst_type,
                          size_t     index,
                          float      value)
    {
        switch (dst_type) {
            case MatrixType::f32:
                static_cast<float*>(dst_ptr)[index] = value;
                break;
            case MatrixType::u8:
                static_cast<uint8_t*>(dst_ptr)[index] = static_cast<uint8_t>(
                    std::max(0.0f, std::min(255.0f, std::round(value))));
                break;
            case MatrixType::s8:
                static_cast<int8_t*>(dst_ptr)[index] = static_cast<int8_t>(
                    std::max(-128.0f, std::min(127.0f, std::round(value))));
                break;
            case MatrixType::u16:
                static_cast<uint16_t*>(dst_ptr)[index] = static_cast<uint16_t>(
                    std::max(0.0f, std::min(65535.0f, std::round(value))));
                break;
            case MatrixType::s16:
                static_cast<int16_t*>(dst_ptr)[index] = static_cast<int16_t>(
                    std::max(-32768.0f, std::min(32767.0f, std::round(value))));
                break;
            case MatrixType::u32:
                static_cast<uint32_t*>(dst_ptr)[index] = static_cast<uint32_t>(
                    std::max(0.0f, std::min(static_cast<float>(UINT32_MAX),
                                            std::round(value))));
                break;
            case MatrixType::s32:
                static_cast<int32_t*>(dst_ptr)[index] = static_cast<int32_t>(
                    std::max(-2147483648.0f,
                             std::min(2147483647.0f, std::round(value))));
                break;
            case MatrixType::u4: {
                uint8_t* data       = static_cast<uint8_t*>(dst_ptr);
                size_t   byte_idx   = index / 2;
                size_t   bit_offset = (index % 2) * 4;
                uint8_t  value4bit =
                    static_cast<uint8_t>(
                        std::max(0.0f, std::min(15.0f, std::round(value))))
                    & 0x0F;
                if (bit_offset == 0) {
                    data[byte_idx] = (data[byte_idx] & 0xF0) | value4bit;
                } else {
                    data[byte_idx] = (data[byte_idx] & 0x0F) | (value4bit << 4);
                }
                break;
            }
            case MatrixType::s4: {
                uint8_t* data       = static_cast<uint8_t*>(dst_ptr);
                size_t   byte_idx   = index / 2;
                size_t   bit_offset = (index % 2) * 4;
                int8_t   value_s4   = static_cast<int8_t>(
                    std::max(-8.0f, std::min(7.0f, std::round(value))));
                uint8_t value4bit = static_cast<uint8_t>(value_s4) & 0x0F;
                if (bit_offset == 0) {
                    data[byte_idx] = (data[byte_idx] & 0xF0) | value4bit;
                } else {
                    data[byte_idx] = (data[byte_idx] & 0x0F) | (value4bit << 4);
                }
                break;
            }
            case MatrixType::bf16:
                // BF16 is marked as unsupported as requested
                break;
            default:
                break;
        }
    }

    /**
     * @brief Copy matrix data with type conversion to float
     * @param src Source matrix
     * @param dst_data Destination float array
     * @param dst_ld Destination leading dimension
     * @return bool Success status
     */
    bool copyMatrixToFloat(const Matrix& src, float* dst_data, md_t dst_ld)
    {
        const void*  src_data   = src.getData();
        MatrixType   src_type   = src.getMatrixType();
        md_t         src_rows   = src.getRows();
        md_t         src_cols   = src.getCols();
        md_t         src_ld     = src.getLeadingDimension();
        MatrixLayout layout     = src.getLayout();
        bool         transposed = src.isTransposed();

        // Handle BF16 as unsupported
        if (src_type == MatrixType::bf16) {
            return false;
        }

        for (md_t i = 0; i < src_rows; ++i) {
            for (md_t j = 0; j < src_cols; ++j) {
                size_t src_idx, dst_idx;

                // Calculate source index based on layout and transposition
                if (layout == MatrixLayout::ROW_MAJOR) {
                    src_idx = static_cast<size_t>(i) * src_ld + j;
                } else {
                    src_idx = static_cast<size_t>(j) * src_ld + i;
                }

                // Calculate destination index (always row-major for float
                // conversion)
                if (transposed) {
                    // If source is transposed, swap i,j for destination
                    dst_idx = static_cast<size_t>(j) * dst_ld + i;
                } else {
                    dst_idx = static_cast<size_t>(i) * dst_ld + j;
                }

                dst_data[dst_idx] = convertToFloat(src_data, src_type, src_idx);
            }
        }
        return true;
    }

    /**
     * @brief Copy float data back to matrix with type conversion
     * @param src_data Source float array
     * @param src_ld Source leading dimension
     * @param dst Destination matrix
     * @return bool Success status
     */
    bool copyFloatToMatrix(const float* src_data, md_t src_ld, Matrix& dst)
    {
        void*        dst_data = dst.getData();
        MatrixType   dst_type = dst.getMatrixType();
        md_t         dst_rows = dst.getRows();
        md_t         dst_cols = dst.getCols();
        md_t         dst_ld   = dst.getLeadingDimension();
        MatrixLayout layout   = dst.getLayout();

        // Handle BF16 as unsupported
        if (dst_type == MatrixType::bf16) {
            return false;
        }

        for (md_t i = 0; i < dst_rows; ++i) {
            for (md_t j = 0; j < dst_cols; ++j) {
                size_t src_idx, dst_idx;

                // Source is always row-major float
                src_idx = static_cast<size_t>(i) * src_ld + j;

                // Calculate destination index based on layout
                if (layout == MatrixLayout::ROW_MAJOR) {
                    dst_idx = static_cast<size_t>(i) * dst_ld + j;
                } else {
                    dst_idx = static_cast<size_t>(j) * dst_ld + i;
                }

                convertFromFloat(dst_data, dst_type, dst_idx,
                                 src_data[src_idx]);
            }
        }
        return true;
    }

} // anonymous namespace

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
 * @brief Templated helper function for matrix reordering
 *
 * @tparam T Data type of the matrix elements
 * @param in Input matrix
 * @param out Output matrix (already created)
 * @param input_rows Number of input rows
 * @param input_cols Number of input columns
 * @param layout Matrix layout (ROW_MAJOR or COLUMN_MAJOR)
 * @param transposed Whether input matrix is transposed
 * @return bool Success status
 */
template<typename T>
bool
reorderTyped(const Matrix& in,
             Matrix&       out,
             md_t          input_rows,
             md_t          input_cols,
             MatrixLayout  layout,
             bool          transposed)
{
    const T* src_data = reinterpret_cast<const T*>(in.getData());
    T*       dst_data = reinterpret_cast<T*>(out.getData());

    md_t src_ld = in.getLeadingDimension();
    md_t dst_ld = out.getLeadingDimension();

    // Pre-calculate bounds for efficiency
    size_t max_src_elements = in.getDataSizeBytes() / sizeof(T);
    size_t max_dst_elements = out.getDataSizeBytes() / sizeof(T);

    // Copy data with proper layout handling using templated helper
    auto copyMatrix = [&](auto getSrcIndex, auto getDstIndex) {
        for (md_t i = 0; i < input_rows; ++i) {
            for (md_t j = 0; j < input_cols; ++j) {
                auto src_idx = getSrcIndex(i, j, src_ld);
                auto dst_idx = getDstIndex(i, j, dst_ld);

                // Optimized bounds check - check once per row if possible
                if (static_cast<size_t>(src_idx) < max_src_elements
                    && static_cast<size_t>(dst_idx) < max_dst_elements) {
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
            // Input is transposed column-major: src[j][i] represents
            // logical[i][j] Output is non-transposed column-major: dst[i][j]
            // represents logical[i][j] So we need: dst[i][j] = src[j][i]
            copyMatrix([](md_t i, md_t j, md_t ld) { return j * ld + i; },
                       [](md_t i, md_t j, md_t ld) { return i * ld + j; });
        } else {
            // Direct copy for non-transposed matrices
            copyMatrix([](md_t i, md_t j, md_t ld) { return j * ld + i; },
                       [](md_t i, md_t j, md_t ld) { return j * ld + i; });
        }
    }

    return true;
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
 * @return UALError Error code indicating success or failure
 */
UALError
UalRef::reorder(const Matrix& in,
                Matrix&       out,
                MatrixType    A_type,
                MatrixType    B_type,
                MatrixType    C_type,
                MatrixType    accType)
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

    // The A_type, B_type, C_type parameters provide context about the GEMM
    // operation where this reordered matrix will be used. This enables
    // optimizations based on the full type combination, handling edge cases
    // where mixed precision is involved. For reference implementation, we use
    // the input type but could apply different reordering strategies based on
    // the GEMM context.

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

    // Templated reorder implementation to avoid code duplication
    bool success = false;
    switch (input_type) {
        case MatrixType::f32:
            success = reorderTyped<float>(in, out, input_rows, input_cols,
                                          layout, transposed);
            break;
        case MatrixType::bf16:
            success = reorderTyped<bfloat16>(in, out, input_rows, input_cols,
                                             layout, transposed);
            break;
        case MatrixType::u8:
            success = reorderTyped<uint8_t>(in, out, input_rows, input_cols,
                                            layout, transposed);
            break;
        case MatrixType::s8:
            success = reorderTyped<int8_t>(in, out, input_rows, input_cols,
                                           layout, transposed);
            break;
        default:
            // For other data types, return not supported
            return UALError::UAL_NOT_SUPPORTED;
    }

    return success ? UALError::UAL_SUCCESS : UALError::UAL_FAILURE;
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
UalRef::checkValidGemmParams(const Matrix& A,
                             const Matrix& B,
                             const Matrix& C,
                             bool          hasMetadata)
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

    if (col_stored) {
        switch (A.getMatrixType()) {
            case MatrixType::u8:
                return false;
            case MatrixType::s8:
                if (C.getMatrixType() == MatrixType::u8)
                    return false;
                if (hasMetadata)
                    return false;
                break;

            case MatrixType::f32:
            case MatrixType::bf16:
                break;
            default:
                // Unsupported matrix type
                break;
        }
    }

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
    } else {
        return false;
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
    } else {
        if (row_stored) {
            if (notb && (B.getLeadingDimension() < B.getCols())) {
                return false;
            }
            if (tb && (B.getLeadingDimension() < B.getRows())) {
                return false;
            }
        } else {
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

// Core GEMM
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
    // Validation Check Removed Intentionally
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

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::s32,
                          MatrixType::s32>(): {
            // For u8/s8 operations, alpha/beta are int32_t type
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            dlp::testing::classic::ref::aocl_gemm_u8s8s32os32_ref(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<const uint8_t*>(
                    A.getMatrixData().getMatrixPtr()),
                static_cast<int>(A.getLeadingDimension()),
                reinterpret_cast<const int8_t*>(
                    B.getMatrixData().getMatrixPtr()),
                static_cast<int>(B.getLeadingDimension()), beta_s32,
                reinterpret_cast<int32_t*>(C.getMatrixData().getMatrixPtr()),
                static_cast<int>(C.getLeadingDimension()), nullptr);

            return true;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::s32,
                          MatrixType::s32>(): {
            // For s8/s8 operations, alpha/beta are int32_t type
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            dlp::testing::classic::ref::aocl_gemm_s8s8s32os32_ref(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<const int8_t*>(
                    A.getMatrixData().getMatrixPtr()),
                static_cast<int>(A.getLeadingDimension()),
                reinterpret_cast<const int8_t*>(
                    B.getMatrixData().getMatrixPtr()),
                static_cast<int>(B.getLeadingDimension()), beta_s32,
                reinterpret_cast<int32_t*>(C.getMatrixData().getMatrixPtr()),
                static_cast<int>(C.getLeadingDimension()), nullptr);

            return true;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::f32,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            dlp::testing::classic::ref::aocl_gemm_u8s8s32of32_ref(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<const uint8_t*>(
                    A.getMatrixData().getMatrixPtr()),
                static_cast<int>(A.getLeadingDimension()),
                reinterpret_cast<const int8_t*>(
                    B.getMatrixData().getMatrixPtr()),
                static_cast<int>(B.getLeadingDimension()), beta_s32,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                static_cast<int>(C.getLeadingDimension()), nullptr);

            return true;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::s8,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            dlp::testing::classic::ref::aocl_gemm_u8s8s32os8_ref(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<const uint8_t*>(
                    A.getMatrixData().getMatrixPtr()),
                static_cast<int>(A.getLeadingDimension()),
                reinterpret_cast<const int8_t*>(
                    B.getMatrixData().getMatrixPtr()),
                static_cast<int>(B.getLeadingDimension()), beta_s32,
                reinterpret_cast<int8_t*>(C.getMatrixData().getMatrixPtr()),
                static_cast<int>(C.getLeadingDimension()), nullptr);

            return true;
        }
        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::u8,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            dlp::testing::classic::ref::aocl_gemm_u8s8s32ou8_ref(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<const uint8_t*>(
                    A.getMatrixData().getMatrixPtr()),
                static_cast<int>(A.getLeadingDimension()),
                reinterpret_cast<const int8_t*>(
                    B.getMatrixData().getMatrixPtr()),
                static_cast<int>(B.getLeadingDimension()), beta_s32,
                reinterpret_cast<uint8_t*>(C.getMatrixData().getMatrixPtr()),
                static_cast<int>(C.getLeadingDimension()), nullptr);

            return true;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::bf16,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            dlp::testing::classic::ref::aocl_gemm_u8s8s32obf16_ref(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<const uint8_t*>(
                    A.getMatrixData().getMatrixPtr()),
                static_cast<int>(A.getLeadingDimension()),
                reinterpret_cast<const int8_t*>(
                    B.getMatrixData().getMatrixPtr()),
                static_cast<int>(B.getLeadingDimension()), beta_s32,
                reinterpret_cast<bfloat16*>(C.getMatrixData().getMatrixPtr()),
                static_cast<int>(C.getLeadingDimension()), nullptr);

            return true;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::f32,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            dlp::testing::classic::ref::aocl_gemm_s8s8s32of32_ref(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<const int8_t*>(
                    A.getMatrixData().getMatrixPtr()),
                static_cast<int>(A.getLeadingDimension()),
                reinterpret_cast<const int8_t*>(
                    B.getMatrixData().getMatrixPtr()),
                static_cast<int>(B.getLeadingDimension()), beta_s32,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                static_cast<int>(C.getLeadingDimension()), nullptr);

            return true;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::s8,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            dlp::testing::classic::ref::aocl_gemm_s8s8s32os8_ref(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<const int8_t*>(
                    A.getMatrixData().getMatrixPtr()),
                static_cast<int>(A.getLeadingDimension()),
                reinterpret_cast<const int8_t*>(
                    B.getMatrixData().getMatrixPtr()),
                static_cast<int>(B.getLeadingDimension()), beta_s32,
                reinterpret_cast<int8_t*>(C.getMatrixData().getMatrixPtr()),
                static_cast<int>(C.getLeadingDimension()), nullptr);

            return true;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::bf16,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            dlp::testing::classic::ref::aocl_gemm_s8s8s32obf16_ref(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<const int8_t*>(
                    A.getMatrixData().getMatrixPtr()),
                static_cast<int>(A.getLeadingDimension()),
                reinterpret_cast<const int8_t*>(
                    B.getMatrixData().getMatrixPtr()),
                static_cast<int>(B.getLeadingDimension()), beta_s32,
                reinterpret_cast<bfloat16*>(C.getMatrixData().getMatrixPtr()),
                static_cast<int>(C.getLeadingDimension()), nullptr);

            return true;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::u8,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            dlp::testing::classic::ref::aocl_gemm_s8s8s32ou8_ref(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<const int8_t*>(
                    A.getMatrixData().getMatrixPtr()),
                static_cast<int>(A.getLeadingDimension()),
                reinterpret_cast<const int8_t*>(
                    B.getMatrixData().getMatrixPtr()),
                static_cast<int>(B.getLeadingDimension()), beta_s32,
                reinterpret_cast<uint8_t*>(C.getMatrixData().getMatrixPtr()),
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
 * @brief Apply scale post-operation to a matrix
 */
template<>
void
UalRef::applyPostOperation(Matrix&                                    matrix,
                           const dlp::testing::framework::ScaleParam& op)
{
    applyScale(matrix, op.getScaleFactor(), op.getZeroPoint());
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
UALError
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
        bool success = checkValidGemmParams(A, B, C, false)
                       && gemm(A, B, C, accType, alpha, beta);
        return success ? UALError::UAL_SUCCESS : UALError::UAL_FAILURE;
    }

    // Validate that the postOps are for REF backend
    if (postOps->getUALType() != UALType::REF) {
        return UALError::UAL_FAILURE;
    }

    // Cast to RefOperation and use the iterator pattern
    const auto* refOp = dynamic_cast<const RefOperation*>(postOps.get());
    if (!refOp) {
        return UALError::UAL_CAST_ERROR;
    }

    // First perform the GEMM operation
    bool result = checkValidGemmParams(A, B, C, true)
                  && gemm(A, B, C, accType, alpha, beta);
    if (!result) {
        return UALError::UAL_FAILURE;
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

    return UALError::UAL_SUCCESS;
}

UALError
UalRef::gemm(md_t         m,
             md_t         n,
             md_t         k,
             void*        matA,
             MatrixType   matA_type,
             MatrixLayout matA_layout,
             bool         matA_transposed,
             char         memFormatA,
             md_t         matA_leadingDim,
             void*        matB,
             MatrixType   matB_type,
             MatrixLayout matB_layout,
             bool         matB_transposed,
             char         memFormatB,
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
    // For now, this is a stub implementation that returns failure
    // indicating the operation is not supported in the reference implementation
    // TODO: Implement reference GEMM logic if needed for benchmarking
    return UALError::UAL_FAILURE;
}

// Unified postop helper that handles type conversion
void
UalRef::applyUnifiedPostOp(Matrix&                             matrix,
                           std::function<void(float*, size_t)> operation)
{
    MatrixType original_type = matrix.getMatrixType();

    // Handle BF16 as unsupported
    if (original_type == MatrixType::bf16) {
        return;
    }

    // If already f32, apply operation directly with proper indexing
    if (original_type == MatrixType::f32) {
        float* data = reinterpret_cast<float*>(matrix.getData());
        md_t   rows = matrix.getRows();
        md_t   cols = matrix.getCols();
        md_t   ld   = matrix.getLeadingDimension();

        // Apply operation element by element with proper leading dimension
        // handling
        for (md_t i = 0; i < rows; ++i) {
            for (md_t j = 0; j < cols; ++j) {
                size_t idx = (matrix.getLayout() == MatrixLayout::ROW_MAJOR)
                                 ? (static_cast<size_t>(i) * ld + j)
                                 : (static_cast<size_t>(j) * ld + i);
                float  temp_val = data[idx];
                operation(&temp_val, 1);
                data[idx] = temp_val;
            }
        }
        return;
    }

    // For other types, convert to float, apply operation, convert back
    md_t rows = matrix.getRows();
    md_t cols = matrix.getCols();
    md_t ld   = cols; // Use simple row-major layout for temp matrix

    // Create temporary float matrix
    std::unique_ptr<float[]> temp_data(new float[rows * ld]);

    // Convert to float
    if (!copyMatrixToFloat(matrix, temp_data.get(), ld)) {
        return;
    }

    // Apply operation in float with proper indexing
    for (md_t i = 0; i < rows; ++i) {
        for (md_t j = 0; j < cols; ++j) {
            size_t idx      = static_cast<size_t>(i) * ld + j;
            float  temp_val = temp_data[idx];
            operation(&temp_val, 1);
            temp_data[idx] = temp_val;
        }
    }

    // Convert back to original type
    copyFloatToMatrix(temp_data.get(), ld, matrix);
}

// Helper methods for specific operations
void
UalRef::applyRelu(Matrix& matrix)
{
    applyUnifiedPostOp(matrix, [](float* data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            data[i] = std::max(0.0f, data[i]);
        }
    });
}

void
UalRef::applyPrelu(Matrix& matrix, const Matrix* alpha)
{
    MatrixType original_type = matrix.getMatrixType();

    // Handle BF16 as unsupported
    if (original_type == MatrixType::bf16) {
        return;
    }

    // Get alpha value
    float alpha_val = 0.01f; // Default alpha
    if (alpha && alpha->getMatrixType() == MatrixType::f32) {
        alpha_val = *reinterpret_cast<const float*>(alpha->getData());
    }

    // If already f32, apply directly
    if (original_type == MatrixType::f32) {
        float* data = reinterpret_cast<float*>(matrix.getData());
        md_t   rows = matrix.getRows();
        md_t   cols = matrix.getCols();
        md_t   ld   = matrix.getLeadingDimension();

        // Apply PReLU element by element with proper leading dimension handling
        for (md_t i = 0; i < rows; ++i) {
            for (md_t j = 0; j < cols; ++j) {
                size_t idx = (matrix.getLayout() == MatrixLayout::ROW_MAJOR)
                                 ? (static_cast<size_t>(i) * ld + j)
                                 : (static_cast<size_t>(j) * ld + i);
                if (data[idx] < 0.0f) {
                    data[idx] *= alpha_val;
                }
            }
        }
        return;
    }

    // For other types, convert to float, apply operation, convert back
    md_t rows = matrix.getRows();
    md_t cols = matrix.getCols();
    md_t ld   = cols; // Use simple row-major layout for temp matrix

    // Create temporary float matrix
    std::unique_ptr<float[]> temp_data(new float[rows * ld]);

    // Convert to float
    if (!copyMatrixToFloat(matrix, temp_data.get(), ld)) {
        return;
    }

    // Apply PReLU in float
    size_t size = rows * cols;
    for (size_t i = 0; i < size; ++i) {
        if (temp_data[i] < 0.0f) {
            temp_data[i] *= alpha_val;
        }
    }

    // Convert back to original type
    copyFloatToMatrix(temp_data.get(), ld, matrix);
}

void
UalRef::applyGeluTanh(Matrix& matrix)
{
    applyUnifiedPostOp(matrix, [](float* data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            float x = data[i];
            float tanh_arg =
                std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x);
            data[i] = 0.5f * x * (1.0f + std::tanh(tanh_arg));
        }
    });
}

void
UalRef::applyGeluErf(Matrix& matrix)
{
    applyUnifiedPostOp(matrix, [](float* data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            float x = data[i];
            data[i] = 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
        }
    });
}

void
UalRef::applyClip(Matrix& matrix, const Matrix* lower, const Matrix* upper)
{
    float lower_val = -6.0f; // Default lower bound
    float upper_val = 6.0f;  // Default upper bound

    if (lower && lower->getMatrixType() == MatrixType::f32) {
        lower_val = *reinterpret_cast<const float*>(lower->getData());
    }
    if (upper && upper->getMatrixType() == MatrixType::f32) {
        upper_val = *reinterpret_cast<const float*>(upper->getData());
    }

    applyUnifiedPostOp(
        matrix, [lower_val, upper_val](float* data, size_t size) {
            for (size_t i = 0; i < size; ++i) {
                data[i] = std::max(lower_val, std::min(upper_val, data[i]));
            }
        });
}

void
UalRef::applySwish(Matrix& matrix)
{
    applyUnifiedPostOp(matrix, [](float* data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            float x = data[i];
            data[i] = x / (1.0f + std::exp(-x));
        }
    });
}

void
UalRef::applyTanh(Matrix& matrix)
{
    applyUnifiedPostOp(matrix, [](float* data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            data[i] = std::tanh(data[i]);
        }
    });
}

void
UalRef::applySigmoid(Matrix& matrix)
{
    applyUnifiedPostOp(matrix, [](float* data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            data[i] = 1.0f / (1.0f + std::exp(-data[i]));
        }
    });
}

void
UalRef::applyScale(Matrix&       matrix,
                   const Matrix* scaleFactor,
                   const Matrix* zeroPoint)
{
    if (!scaleFactor) {
        return;
    }

    // Handle BF16 matrix as unsupported (but allow bf16 scale factor)
    if (matrix.getMatrixType() == MatrixType::bf16) {
        return;
    }

    // Also check if zeroPoint has BF16 type
    if (zeroPoint && zeroPoint->getMatrixType() == MatrixType::bf16) {
        return;
    }

    // Determine scale length: 1 (per-tensor) or N (per-channel)
    bool per_channel    = (scaleFactor->getRows() * scaleFactor->getCols()) > 1;
    bool has_zero_point = (zeroPoint != nullptr);
    bool per_channel_zp =
        has_zero_point && ((zeroPoint->getRows() * zeroPoint->getCols()) > 1);

    if (!per_channel && !per_channel_zp) {
        // Single scale value and optionally single zero point - convert to
        // float
        float scale = convertToFloat(scaleFactor->getData(),
                                     scaleFactor->getMatrixType(), 0);
        float zp    = 0.0f;
        if (has_zero_point) {
            zp = convertToFloat(zeroPoint->getData(),
                                zeroPoint->getMatrixType(), 0);
        }
        applyUnifiedPostOp(matrix, [scale, zp](float* data, size_t size) {
            for (size_t i = 0; i < size; ++i) {
                data[i] = data[i] * scale + zp;
            }
        });
        return;
    }

    // For per-channel scaling, we need to handle layout properly
    md_t rows = matrix.getRows();
    md_t cols = matrix.getCols();
    md_t ld   = matrix.getLeadingDimension();

    // Convert scale factor to float array
    md_t scale_size = scaleFactor->getRows() * scaleFactor->getCols();
    std::unique_ptr<float[]> scale_data(new float[scale_size]);

    for (md_t i = 0; i < scale_size; ++i) {
        scale_data[i] = convertToFloat(scaleFactor->getData(),
                                       scaleFactor->getMatrixType(), i);
    }

    // Convert zero point to float array if provided
    md_t                     zp_size = 0;
    std::unique_ptr<float[]> zp_data;
    if (has_zero_point) {
        zp_size = zeroPoint->getRows() * zeroPoint->getCols();
        zp_data.reset(new float[zp_size]);
        for (md_t i = 0; i < zp_size; ++i) {
            zp_data[i] = convertToFloat(zeroPoint->getData(),
                                        zeroPoint->getMatrixType(), i);
        }
    }

    // If matrix is already f32, apply directly
    if (matrix.getMatrixType() == MatrixType::f32) {
        float* data = reinterpret_cast<float*>(matrix.getData());

        // Per-channel assumed along N (columns)
        for (md_t i = 0; i < rows; ++i) {
            for (md_t j = 0; j < cols; ++j) {
                float  scale = scale_data[j % scale_size];
                float  zp    = has_zero_point ? zp_data[j % zp_size] : 0.0f;
                size_t idx   = (matrix.getLayout() == MatrixLayout::ROW_MAJOR)
                                   ? (static_cast<size_t>(i) * ld + j)
                                   : (static_cast<size_t>(j) * ld + i);
                data[idx]    = data[idx] * scale + zp;
            }
        }
        return;
    }

    // For other types, convert to float, apply operation, convert back
    md_t temp_ld = cols; // Use simple row-major layout for temp matrix

    // Create temporary float matrix
    std::unique_ptr<float[]> temp_data(new float[rows * temp_ld]);

    // Convert to float
    if (!copyMatrixToFloat(matrix, temp_data.get(), temp_ld)) {
        return;
    }

    // Apply per-channel scaling in float
    for (md_t i = 0; i < rows; ++i) {
        for (md_t j = 0; j < cols; ++j) {
            float  scale   = scale_data[j % scale_size];
            float  zp      = has_zero_point ? zp_data[j % zp_size] : 0.0f;
            size_t idx     = static_cast<size_t>(i) * temp_ld + j;
            temp_data[idx] = temp_data[idx] * scale + zp;
        }
    }

    // Convert back to original type
    copyFloatToMatrix(temp_data.get(), temp_ld, matrix);
}

void
UalRef::applyBias(Matrix& matrix, const Matrix& bias)
{
    // Handle BF16 matrix as unsupported (but allow bf16 bias)
    if (matrix.getMatrixType() == MatrixType::bf16) {
        return;
    }

    md_t bias_size = bias.getRows() * bias.getCols();

    // Convert bias to float array
    std::unique_ptr<float[]> bias_data(new float[bias_size]);
    for (md_t i = 0; i < bias_size; ++i) {
        bias_data[i] = convertToFloat(bias.getData(), bias.getMatrixType(), i);
    }

    // If matrix is already f32, apply directly
    if (matrix.getMatrixType() == MatrixType::f32) {
        float* data = reinterpret_cast<float*>(matrix.getData());
        md_t   rows = matrix.getRows();
        md_t   cols = matrix.getCols();
        md_t   ld   = matrix.getLeadingDimension();

        // Apply bias to each row/column depending on bias size
        for (md_t i = 0; i < rows; ++i) {
            for (md_t j = 0; j < cols; ++j) {
                size_t idx = (matrix.getLayout() == MatrixLayout::ROW_MAJOR)
                                 ? (static_cast<size_t>(i) * ld + j)
                                 : (static_cast<size_t>(j) * ld + i);
                size_t bias_idx = j % bias_size; // Broadcast bias
                data[idx] += bias_data[bias_idx];
            }
        }
        return;
    }

    // For other types, convert to float, apply operation, convert back
    md_t rows = matrix.getRows();
    md_t cols = matrix.getCols();
    md_t ld   = cols; // Use simple row-major layout for temp matrix

    // Create temporary float matrix
    std::unique_ptr<float[]> temp_data(new float[rows * ld]);

    // Convert to float
    if (!copyMatrixToFloat(matrix, temp_data.get(), ld)) {
        return;
    }

    // Apply bias in float
    for (md_t i = 0; i < rows; ++i) {
        for (md_t j = 0; j < cols; ++j) {
            size_t idx      = static_cast<size_t>(i) * ld + j;
            size_t bias_idx = j % bias_size; // Broadcast bias
            temp_data[idx] += bias_data[bias_idx];
        }
    }

    // Convert back to original type
    copyFloatToMatrix(temp_data.get(), ld, matrix);
}

void
UalRef::applyMatrixAdd(Matrix&       matrix,
                       const Matrix& addMatrix,
                       const Matrix* scaleFactor)
{
    // Handle BF16 as unsupported
    if (matrix.getMatrixType() == MatrixType::bf16
        || addMatrix.getMatrixType() == MatrixType::bf16) {
        return;
    }

    float scale = 1.0f;
    if (scaleFactor && scaleFactor->getMatrixType() == MatrixType::f32) {
        scale = *reinterpret_cast<const float*>(scaleFactor->getData());
    }

    // If both matrices are f32, apply directly
    if (matrix.getMatrixType() == MatrixType::f32
        && addMatrix.getMatrixType() == MatrixType::f32) {
        float*       data = reinterpret_cast<float*>(matrix.getData());
        const float* add_data =
            reinterpret_cast<const float*>(addMatrix.getData());

        md_t rows          = matrix.getRows();
        md_t cols          = matrix.getCols();
        md_t matrix_ld     = matrix.getLeadingDimension();
        md_t add_matrix_ld = addMatrix.getLeadingDimension();

        // Apply matrix addition with proper leading dimension handling
        for (md_t i = 0; i < rows; ++i) {
            for (md_t j = 0; j < cols; ++j) {
                size_t matrix_idx =
                    (matrix.getLayout() == MatrixLayout::ROW_MAJOR)
                        ? (static_cast<size_t>(i) * matrix_ld + j)
                        : (static_cast<size_t>(j) * matrix_ld + i);
                size_t add_idx =
                    (addMatrix.getLayout() == MatrixLayout::ROW_MAJOR)
                        ? (static_cast<size_t>(i) * add_matrix_ld + j)
                        : (static_cast<size_t>(j) * add_matrix_ld + i);
                data[matrix_idx] += add_data[add_idx] * scale;
            }
        }
        return;
    }

    // For other types, convert to float, apply operation, convert back
    md_t rows = matrix.getRows();
    md_t cols = matrix.getCols();
    md_t ld   = cols; // Use simple row-major layout for temp matrix

    // Create temporary float matrices
    std::unique_ptr<float[]> temp_matrix(new float[rows * ld]);
    std::unique_ptr<float[]> temp_add(new float[rows * ld]);

    // Convert both matrices to float
    if (!copyMatrixToFloat(matrix, temp_matrix.get(), ld)
        || !copyMatrixToFloat(addMatrix, temp_add.get(), ld)) {
        return;
    }

    // Apply matrix addition in float with proper indexing
    for (md_t i = 0; i < rows; ++i) {
        for (md_t j = 0; j < cols; ++j) {
            size_t idx = static_cast<size_t>(i) * ld + j;
            temp_matrix[idx] += temp_add[idx] * scale;
        }
    }

    // Convert back to original type
    copyFloatToMatrix(temp_matrix.get(), ld, matrix);
}

void
UalRef::applyMatrixMul(Matrix&       matrix,
                       const Matrix& mulMatrix,
                       const Matrix* scaleFactor)
{
    // Handle BF16 as unsupported
    if (matrix.getMatrixType() == MatrixType::bf16
        || mulMatrix.getMatrixType() == MatrixType::bf16) {
        return;
    }

    float scale = 1.0f;
    if (scaleFactor && scaleFactor->getMatrixType() == MatrixType::f32) {
        scale = *reinterpret_cast<const float*>(scaleFactor->getData());
    }

    // If both matrices are f32, apply directly
    if (matrix.getMatrixType() == MatrixType::f32
        && mulMatrix.getMatrixType() == MatrixType::f32) {
        float*       data = reinterpret_cast<float*>(matrix.getData());
        const float* mul_data =
            reinterpret_cast<const float*>(mulMatrix.getData());

        md_t rows          = matrix.getRows();
        md_t cols          = matrix.getCols();
        md_t matrix_ld     = matrix.getLeadingDimension();
        md_t mul_matrix_ld = mulMatrix.getLeadingDimension();

        // Apply matrix multiplication with proper leading dimension handling
        for (md_t i = 0; i < rows; ++i) {
            for (md_t j = 0; j < cols; ++j) {
                size_t matrix_idx =
                    (matrix.getLayout() == MatrixLayout::ROW_MAJOR)
                        ? (static_cast<size_t>(i) * matrix_ld + j)
                        : (static_cast<size_t>(j) * matrix_ld + i);
                size_t mul_idx =
                    (mulMatrix.getLayout() == MatrixLayout::ROW_MAJOR)
                        ? (static_cast<size_t>(i) * mul_matrix_ld + j)
                        : (static_cast<size_t>(j) * mul_matrix_ld + i);
                data[matrix_idx] *= mul_data[mul_idx] * scale;
            }
        }
        return;
    }

    // For other types, convert to float, apply operation, convert back
    md_t rows = matrix.getRows();
    md_t cols = matrix.getCols();
    md_t ld   = cols; // Use simple row-major layout for temp matrix

    // Create temporary float matrices
    std::unique_ptr<float[]> temp_matrix(new float[rows * ld]);
    std::unique_ptr<float[]> temp_mul(new float[rows * ld]);

    // Convert both matrices to float
    if (!copyMatrixToFloat(matrix, temp_matrix.get(), ld)
        || !copyMatrixToFloat(mulMatrix, temp_mul.get(), ld)) {
        return;
    }

    // Apply element-wise multiplication in float with proper indexing
    for (md_t i = 0; i < rows; ++i) {
        for (md_t j = 0; j < cols; ++j) {
            size_t idx = static_cast<size_t>(i) * ld + j;
            temp_matrix[idx] *= temp_mul[idx] * scale;
        }
    }

    // Convert back to original type
    copyFloatToMatrix(temp_matrix.get(), ld, matrix);
}

} // namespace dlp::testing::classic
