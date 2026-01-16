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
 * @file matrix.cc
 * @brief Implementation of the Matrix class with C++17 aligned allocation
 * support
 *
 * This file contains the implementation of the Matrix class that handles
 * various data types with support for different memory layouts, virtual
 * transposition, and optional memory alignment. Memory is managed using
 * C++17 aligned_alloc for aligned memory or regular new[] for unaligned memory.
 */

#include "framework/matrix.hh"
#include "aocl_dlp_config.h"
#include "classic/aocl_bf16_type.h"
#include "classic/dlp_base_types.h"
#include "utils/conversion_utils.hh"
#include <algorithm> // For std::max
#include <any>
#include <chrono> // For time-based seeding
#include <cmath>  // For std::abs, std::isnan, std::isinf
#include <cstdint>
#include <cstdlib>  // For std::aligned_alloc, std::free
#include <cstring>  // For std::memcpy, std::memcmp
#include <iomanip>  // For std::setprecision, std::setw, std::setfill
#include <iostream> // For std::cout
#include <limits>   // For std::numeric_limits
#include <random>   // For random number generation
#include <sstream>
#include <stdexcept>

using dlp::testing::utils::bf16_to_f32;
using dlp::testing::utils::f32_to_bf16;

namespace dlp { namespace testing { namespace framework {
    using dlp::testing::utils::bf16_to_f32;
    using dlp::testing::utils::f32_to_bf16;

    /**
     * @brief Calculate the machine epsilon for bfloat16
     *
     * BF16 has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
     * Machine epsilon is the smallest value such that 1.0 + epsilon != 1.0
     * For bf16, this is 2^-7 = 0.0078125
     *
     * @return The machine epsilon for bfloat16 as a float
     */
    static float bf16_machine_epsilon()
    {
        // BF16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
        // Machine epsilon = 2^(-mantissa_bits) = 2^(-7) = 0.0078125
        // We can also calculate this by finding the next representable value
        // after 1.0
        constexpr uint16_t one_bits  = 0x3F80u; // bf16 encoding of 1.0f
        constexpr uint16_t next_bits = 0x3F81u; // next representable value

        float one  = bf16_to_f32(static_cast<bfloat16>(one_bits));
        float next = bf16_to_f32(static_cast<bfloat16>(next_bits));

        return next - one;
    }

    /**
     * @brief Default constructor implementation
     *
     * Creates an empty matrix with default values. Initializes all member
     * variables to default states but does not allocate any memory.
     */
    Matrix::Matrix()
        : m_rows(0)
        , m_cols(0)
        , m_k(std::numeric_limits<md_t>::max())
        , m_type(MatrixType::f32)
        , m_data(nullptr)
        , m_dataSizeBytes(0)
        , m_alignment(0)
        , m_layout(MatrixLayout::ROW_MAJOR)
        , m_leadingDim(0)
        , m_transposed(false)
        , m_reordered(false)
        , m_packed(false)
    {
    }

    /**
     * @brief Main constructor with external memory management
     *
     * Creates a matrix with specified dimensions, data type, layout, and
     * externally provided memory. Takes ownership of the provided raw pointer.
     * The caller must ensure the memory remains valid for the lifetime of this
     * matrix object.
     */
    Matrix::Matrix(md_t         rows,
                   md_t         cols,
                   MatrixType   type,
                   uint8_t*     data,
                   size_t       dataSizeBytes,
                   MatrixLayout layout,
                   md_t         leadingDim,
                   bool         transposed,
                   bool         reordered,
                   size_t       alignment)
        : m_rows(rows)
        , m_cols(cols)
        , m_k(std::numeric_limits<md_t>::max())
        , m_type(type)
        , m_data(data) // Take ownership from raw pointer
        , m_dataSizeBytes(dataSizeBytes)
        , m_alignment(alignment)
        , m_layout(layout)
        , m_transposed(transposed)
        , m_reordered(reordered)
        , m_packed(false)
    {
        // Calculate leading dimension if not specified
        if (leadingDim == 0) {
            m_leadingDim = (layout == MatrixLayout::ROW_MAJOR) ? cols : rows;
        } else {
            m_leadingDim = leadingDim;
        }

        // Validate that provided memory is sufficient
        size_t requiredBytes = MatrixMemory::calculateRequiredBytes(
            type, rows, cols, layout, m_leadingDim);

        if (dataSizeBytes < requiredBytes) {
            throw std::invalid_argument(
                "Provided memory size is insufficient for matrix dimensions");
        }
    }

    /**
     * @brief Convenience constructor with automatic memory allocation
     *
     * Creates a matrix with automatic memory allocation using C++17
     * aligned_alloc if alignment is specified, or regular new[] if no alignment
     * is needed.
     */
    Matrix::Matrix(md_t         rows,
                   md_t         cols,
                   MatrixType   type,
                   MatrixLayout layout,
                   md_t         leadingDim,
                   bool         transposed,
                   bool         reordered,
                   size_t       alignment)
        : m_rows(rows)
        , m_cols(cols)
        , m_k(std::numeric_limits<md_t>::max())
        , m_type(type)
        , m_alignment(alignment)
        , m_layout(layout)
        , m_transposed(transposed)
        , m_reordered(reordered)
        , m_packed(false)
    {
        // For negative test cases: store actual dimensions (even if <= 0)
        // but use safe dimensions for memory allocation
        md_t alloc_rows = (rows <= 0) ? 1 : rows;
        md_t alloc_cols = (cols <= 0) ? 1 : cols;

        // Calculate leading dimension if not specified
        if (leadingDim == 0) {
            m_leadingDim = (layout == MatrixLayout::ROW_MAJOR) ? alloc_cols
                                                               : alloc_rows;
        } else {
            m_leadingDim = leadingDim;
        }

        // Calculate required memory size using safe dimensions
        m_dataSizeBytes = MatrixMemory::calculateRequiredBytes(
            type, alloc_rows, alloc_cols, layout, m_leadingDim);

        // Allocate memory using helper method
        m_data = allocateAlignedMemory(m_dataSizeBytes, alignment);

        // Zero-initialize the allocated memory to avoid uninitialized values
        if (m_data && m_dataSizeBytes > 0) {
            std::fill(m_data, m_data + m_dataSizeBytes, 0);
        }

        // Update size to reflect actual allocated size if aligned
        if (alignment > 0) {
            size_t alignedSize =
                (m_dataSizeBytes + alignment - 1) & ~(alignment - 1);
            m_dataSizeBytes = alignedSize;
        }
    }

    /**
     * @brief Copy constructor implementation
     *
     * Creates a deep copy of the source matrix with newly allocated memory
     * using the same alignment as the source.
     */
    Matrix::Matrix(const Matrix& other)
        : m_rows(other.m_rows)
        , m_cols(other.m_cols)
        , m_k(other.m_k)
        , m_type(other.m_type)
        , m_dataSizeBytes(other.m_dataSizeBytes)
        , m_alignment(other.m_alignment)
        , m_layout(other.m_layout)
        , m_leadingDim(other.m_leadingDim)
        , m_transposed(other.m_transposed)
        , m_reordered(other.m_reordered)
        , m_packed(other.m_packed)
    {
        // Allocate new memory using helper method
        m_data = allocateAlignedMemory(m_dataSizeBytes, m_alignment);

        // Copy the data
        if (other.m_data && m_dataSizeBytes > 0) {
            std::memcpy(m_data, other.m_data, m_dataSizeBytes);
        }
    }

    /**
     * @brief Move constructor implementation
     *
     * Transfers ownership of memory from source matrix.
     */
    Matrix::Matrix(Matrix&& other) noexcept
        : m_rows(other.m_rows)
        , m_cols(other.m_cols)
        , m_k(other.m_k)
        , m_type(other.m_type)
        , m_data(other.m_data)
        , m_dataSizeBytes(other.m_dataSizeBytes)
        , m_alignment(other.m_alignment)
        , m_layout(other.m_layout)
        , m_leadingDim(other.m_leadingDim)
        , m_transposed(other.m_transposed)
        , m_reordered(other.m_reordered)
        , m_packed(other.m_packed)
    {
        // Reset other matrix
        other.m_data          = nullptr;
        other.m_rows          = 0;
        other.m_cols          = 0;
        other.m_dataSizeBytes = 0;
        other.m_alignment     = 0;
        other.m_leadingDim    = 0;
        other.m_transposed    = false;
        other.m_reordered     = false;
        other.m_packed        = false;
    }

    /**
     * @brief Destructor implementation
     *
     * Properly releases memory based on allocation type.
     */
    Matrix::~Matrix()
    {
        deallocateAlignedMemory(m_data, m_alignment);
    }

    /**
     * @brief Copy assignment operator implementation
     *
     * Creates a deep copy of the source matrix with newly allocated memory.
     */
    Matrix& Matrix::operator=(const Matrix& other)
    {
        // Self-assignment check
        if (this == &other) {
            return *this;
        }

        // Free existing memory using helper method
        deallocateAlignedMemory(m_data, m_alignment);

        // Copy metadata
        m_rows          = other.m_rows;
        m_cols          = other.m_cols;
        m_k             = other.m_k;
        m_type          = other.m_type;
        m_dataSizeBytes = other.m_dataSizeBytes;
        m_alignment     = other.m_alignment;
        m_layout        = other.m_layout;
        m_leadingDim    = other.m_leadingDim;
        m_transposed    = other.m_transposed;
        m_reordered     = other.m_reordered;
        m_packed        = other.m_packed;

        // Allocate new memory using helper method
        m_data = allocateAlignedMemory(m_dataSizeBytes, m_alignment);

        // Copy the data
        if (other.m_data && m_dataSizeBytes > 0) {
            std::memcpy(m_data, other.m_data, m_dataSizeBytes);
        }

        return *this;
    }

    /**
     * @brief Move assignment operator implementation
     *
     * Transfers ownership of memory from source matrix.
     */
    Matrix& Matrix::operator=(Matrix&& other) noexcept
    {
        // Self-assignment check
        if (this == &other) {
            return *this;
        }

        // Free existing memory using helper method
        deallocateAlignedMemory(m_data, m_alignment);

        // Transfer ownership
        m_rows          = other.m_rows;
        m_cols          = other.m_cols;
        m_k             = other.m_k;
        m_type          = other.m_type;
        m_data          = other.m_data;
        m_dataSizeBytes = other.m_dataSizeBytes;
        m_alignment     = other.m_alignment;
        m_layout        = other.m_layout;
        m_leadingDim    = other.m_leadingDim;
        m_transposed    = other.m_transposed;
        m_reordered     = other.m_reordered;
        m_packed        = other.m_packed;

        // Reset other matrix
        other.m_data          = nullptr;
        other.m_rows          = 0;
        other.m_cols          = 0;
        other.m_dataSizeBytes = 0;
        other.m_alignment     = 0;
        other.m_leadingDim    = 0;
        other.m_transposed    = false;
        other.m_reordered     = false;
        other.m_packed        = false;

        return *this;
    }

    /**
     * @brief Get the number of rows in the matrix
     */
    md_t Matrix::getRows() const
    {
        return m_rows;
    }

    /**
     * @brief Get the number of columns in the matrix
     */
    md_t Matrix::getCols() const
    {
        return m_cols;
    }

    /**
     * @brief Get the matrix data type
     */
    MatrixType Matrix::getMatrixType() const
    {
        return m_type;
    }

    /**
     * @brief Get the matrix memory layout
     */
    MatrixLayout Matrix::getLayout() const
    {
        return m_layout;
    }

    /**
     * @brief Check if the matrix is logically transposed
     */
    bool Matrix::isTransposed() const
    {
        return m_transposed;
    }

    /**
     * @brief Check if the matrix is reordered
     */
    bool Matrix::isReordered() const
    {
        return m_reordered;
    }

    /**
     * @brief Get the leading dimension of the matrix
     */
    md_t Matrix::getLeadingDimension() const
    {
        return m_leadingDim;
    }

    /**
     * @brief Get the effective number of rows after considering transposition
     */
    md_t Matrix::getEffectiveRows() const
    {
        return m_transposed ? m_cols : m_rows;
    }

    /**
     * @brief Get the effective number of columns after considering
     * transposition
     */
    md_t Matrix::getEffectiveCols() const
    {
        return m_transposed ? m_rows : m_cols;
    }

    /**
     * @brief Get the data size in bytes
     */
    size_t Matrix::getDataSizeBytes() const
    {
        return m_dataSizeBytes;
    }

    /**
     * @brief Get raw pointer to the matrix data
     */
    void* Matrix::getData() const
    {
        return m_data;
    }

    /**
     * @brief Get the matrix data container (for backward compatibility)
     */
    MatrixData Matrix::getMatrixData() const
    {
        return MatrixData(m_type, m_data);
    }

    /**
     * @brief Set the reordering flag
     */
    void Matrix::setReordered(bool reordered)
    {
        m_reordered = reordered;
    }

    /**
     * @brief Check if the matrix is packed
     */
    bool Matrix::isPacked() const
    {
        return m_packed;
    }

    /**
     * @brief Set the packing flag
     */
    void Matrix::setPacked(bool packed)
    {
        m_packed = packed;
    }

    /**
     * @brief Set the k dimension for tolerance calculation
     */
    void Matrix::setK(md_t k) const
    {
        m_k = k;
    }

    /**
     * @brief Get element size in bytes for the matrix type
     */
    size_t Matrix::getElementSizeBytes() const
    {
        return MatrixMemory::getElementSizeBytes(m_type);
    }

    /**
     * @brief Compare two matrices with configurable mode and diagnostics
     */
    MatrixCompareResult Matrix::compare(const Matrix&               other,
                                        const MatrixCompareOptions& opts) const
    {
        MatrixCompareResult result;

        // Check dimensions
        if (m_rows != other.m_rows || m_cols != other.m_cols) {
            result.equal             = false;
            result.dimensionMismatch = true;
            return result;
        }

        // Check type
        if (m_type != other.m_type) {
            result.equal        = false;
            result.typeMismatch = true;
            return result;
        }

        // Check layout
        if (m_layout != other.m_layout) {
            result.equal          = false;
            result.layoutMismatch = true;
            return result;
        }

        // Check other metadata
        if (m_leadingDim != other.m_leadingDim
            || m_transposed != other.m_transposed
            || m_reordered != other.m_reordered || m_packed != other.m_packed) {
            result.equal            = false;
            result.metadataMismatch = true;
            return result;
        }

        // Check data size
        if (m_dataSizeBytes != other.m_dataSizeBytes) {
            result.equal             = false;
            result.dimensionMismatch = true;
            return result;
        }

        // Handle null data
        if (m_data == nullptr && other.m_data == nullptr) {
            result.equal = true;
            return result;
        }

        if (m_data == nullptr || other.m_data == nullptr) {
            result.equal = false;
            return result;
        }

        // Determine tolerances
        double absTol, relTol;
        if (m_type == MatrixType::f32) {
            // Use k dimension for tolerance scaling if available
            // When k dimension is large (GEMM), accumulated errors increase
            double k_factor = (m_k != std::numeric_limits<md_t>::max())
                                  ? static_cast<double>(m_k)
                                  : 1.0;

            // Get multipliers from options, use default 50.0 if not specified
            double rel_multiplier = (opts.relToleranceMultiplier >= 0.0)
                                        ? opts.relToleranceMultiplier
                                        : 50.0;
            double abs_multiplier = (opts.absToleranceMultiplier >= 0.0)
                                        ? opts.absToleranceMultiplier
                                        : 50.0;

            double epsilon = std::numeric_limits<float>::epsilon();

            // Calculate tolerances: tolerance = epsilon * k * multiplier
            absTol = (opts.absToleranceOverride >= 0.0)
                         ? opts.absToleranceOverride
                         : epsilon * k_factor * abs_multiplier;
            relTol = (opts.relToleranceOverride >= 0.0)
                         ? opts.relToleranceOverride
                         : epsilon * k_factor * rel_multiplier;
        } else if (m_type == MatrixType::bf16) {
            // Use k dimension for tolerance scaling if available
            // When k dimension is large (GEMM), accumulated errors increase
            double k_factor = (m_k != std::numeric_limits<md_t>::max())
                                  ? static_cast<double>(m_k)
                                  : 1.0;

            // Get multipliers from options, use default 50.0 if not specified
            double rel_multiplier = (opts.relToleranceMultiplier >= 0.0)
                                        ? opts.relToleranceMultiplier
                                        : 50.0;
            double abs_multiplier = (opts.absToleranceMultiplier >= 0.0)
                                        ? opts.absToleranceMultiplier
                                        : 50.0;

            // Calculate bf16 machine epsilon
            double epsilon = static_cast<double>(bf16_machine_epsilon());

            // Calculate tolerances: tolerance = epsilon * k * multiplier
            absTol = (opts.absToleranceOverride >= 0.0)
                         ? opts.absToleranceOverride
                         : epsilon * k_factor * abs_multiplier;
            relTol = (opts.relToleranceOverride >= 0.0)
                         ? opts.relToleranceOverride
                         : epsilon * k_factor * rel_multiplier;
        } else {
            // Integer types: exact comparison
            absTol = 0.0;
            relTol = 0.0;
        }

        result.usedAbsTolerance = absTol;
        result.usedRelTolerance = relTol;

        // Fast mode for integer types: use memcmp
        if (!opts.verbose
            && (m_type != MatrixType::f32 && m_type != MatrixType::bf16)) {
            result.equal =
                (std::memcmp(m_data, other.m_data, m_dataSizeBytes) == 0);
            return result;
        }

        // Element-wise comparison
        size_t elementCount = 0;

        switch (m_type) {
            case MatrixType::f32: {
                const float* data1 = reinterpret_cast<const float*>(m_data);
                const float* data2 =
                    reinterpret_cast<const float*>(other.m_data);
                elementCount = m_dataSizeBytes / sizeof(float);

                for (size_t i = 0; i < elementCount; ++i) {
                    float val1 = data1[i];
                    float val2 = data2[i];

                    // NOTE: Special handling for NaN values.
                    // Even though NaN is mathematically undefined and
                    // NaN == NaN is false, but since we are testing for NaN
                    // propagation, we treat the two NaNs at the same position
                    // to be equal for the following scenarios:
                    // 1. If A/B matrices contain NaNs, the resulting C matrix
                    //    should also have NaNs.
                    // 2. Edge cases where operations result in NaN (0.0f/0.0f)
                    //    or beta * C(NaN).
                    if (val1 == val2 // Check for exact equality
                        || (std::isnan(val1) && std::isnan(val2))) {
                        continue;
                    }

                    // Handling the scenario where one value is a NaN. Such
                    // case is treated as a mismatch, hence a failure is
                    // reported.
                    if (std::isnan(val1) || std::isnan(val2)) {
                        result.equal = false;
                        if (!opts.verbose) {
                            return result;
                        }

                        double diff = std::abs(static_cast<double>(val1)
                                               - static_cast<double>(val2));
                        if (result.mismatches.size() < opts.maxMismatches) {
                            size_t       r = i / m_leadingDim;
                            size_t       c = i % m_leadingDim;
                            MismatchInfo info;
                            info.row          = r;
                            info.col          = c;
                            info.value1       = static_cast<double>(val1);
                            info.value2       = static_cast<double>(val2);
                            info.absDiff      = diff;
                            info.relativeDiff = 0.0;
                            result.mismatches.push_back(info);
                        }
                        result.mismatchCount++;
                        if (diff > result.maxAbsDiff) {
                            result.maxAbsDiff = diff;
                            result.maxDiffRow = i / m_leadingDim;
                            result.maxDiffCol = i % m_leadingDim;
                        }
                        continue;
                    }

                    // Checking for Infs.
                    // Above val1 == val2 check has failed, hence both values
                    // are not equal or Inf since Inf == Inf returns true.
                    // Now, if one of the values is Inf, the case is treated as
                    // a mismatch and failure is reported.
                    if (std::isinf(val1) || std::isinf(val2)) {
                        result.equal = false;
                        if (!opts.verbose) {
                            return result;
                        }

                        double diff = std::abs(static_cast<double>(val1)
                                               - static_cast<double>(val2));
                        if (result.mismatches.size() < opts.maxMismatches) {
                            size_t       r = i / m_leadingDim;
                            size_t       c = i % m_leadingDim;
                            MismatchInfo info;
                            info.row          = r;
                            info.col          = c;
                            info.value1       = static_cast<double>(val1);
                            info.value2       = static_cast<double>(val2);
                            info.absDiff      = diff;
                            info.relativeDiff = 0.0;
                            result.mismatches.push_back(info);
                        }
                        result.mismatchCount++;
                        if (diff > result.maxAbsDiff) {
                            result.maxAbsDiff = diff;
                            result.maxDiffRow = i / m_leadingDim;
                            result.maxDiffCol = i % m_leadingDim;
                        }
                        continue;
                    }

                    double diff = std::abs(static_cast<double>(val1)
                                           - static_cast<double>(val2));
                    double maxAbs =
                        std::max(std::abs(static_cast<double>(val1)),
                                 std::abs(static_cast<double>(val2)));
                    double relDiff = (maxAbs > 0.0) ? (diff / maxAbs) : 0.0;

                    if (diff > absTol && relDiff > relTol) {
                        result.equal = false;

                        if (!opts.verbose) {
                            return result;
                        }

                        if (result.mismatches.size() < opts.maxMismatches) {
                            size_t       r = i / m_leadingDim;
                            size_t       c = i % m_leadingDim;
                            MismatchInfo info;
                            info.row          = r;
                            info.col          = c;
                            info.value1       = static_cast<double>(val1);
                            info.value2       = static_cast<double>(val2);
                            info.absDiff      = diff;
                            info.relativeDiff = relDiff;
                            result.mismatches.push_back(info);
                        }

                        result.mismatchCount++;
                        if (diff > result.maxAbsDiff) {
                            result.maxAbsDiff = diff;
                            result.maxRelDiff = relDiff;
                            result.maxDiffRow = i / m_leadingDim;
                            result.maxDiffCol = i % m_leadingDim;
                        }
                    }
                }
                break;
            }

            case MatrixType::bf16: {
                const bfloat16* data1 =
                    reinterpret_cast<const bfloat16*>(m_data);
                const bfloat16* data2 =
                    reinterpret_cast<const bfloat16*>(other.m_data);
                elementCount = m_dataSizeBytes / sizeof(bfloat16);

                for (size_t i = 0; i < elementCount; ++i) {
                    bfloat16 bf16_val1 = data1[i];
                    bfloat16 bf16_val2 = data2[i];

                    // Convert to float for comparison
                    float val1 = bf16_to_f32(bf16_val1);
                    float val2 = bf16_to_f32(bf16_val2);

                    // NOTE: Special handling for NaN values.
                    // Even though NaN is mathematically undefined and
                    // NaN == NaN is false, but since we are testing for NaN
                    // propagation, we treat the two NaNs at the same position
                    // to be equal for the following scenarios:
                    // 1. If A/B matrices contain NaNs, the resulting C matrix
                    //    should also have NaNs.
                    // 2. Edge cases where operations result in NaN (0.0f/0.0f)
                    //    or beta * C(NaN).
                    if (val1 == val2 // Check for exact equality
                        || (std::isnan(val1) && std::isnan(val2))) {
                        continue;
                    }

                    // Handling the scenario where one value is a NaN. Such
                    // case is treated as a mismatch, hence a failure is
                    // reported.
                    if (std::isnan(val1) || std::isnan(val2)) {
                        result.equal = false;
                        if (!opts.verbose) {
                            return result;
                        }

                        double diff = std::abs(static_cast<double>(val1)
                                               - static_cast<double>(val2));
                        if (result.mismatches.size() < opts.maxMismatches) {
                            size_t       r = i / m_leadingDim;
                            size_t       c = i % m_leadingDim;
                            MismatchInfo info;
                            info.row          = r;
                            info.col          = c;
                            info.value1       = static_cast<double>(val1);
                            info.value2       = static_cast<double>(val2);
                            info.absDiff      = diff;
                            info.relativeDiff = 0.0;
                            result.mismatches.push_back(info);
                        }
                        result.mismatchCount++;
                        if (diff > result.maxAbsDiff) {
                            result.maxAbsDiff = diff;
                            result.maxDiffRow = i / m_leadingDim;
                            result.maxDiffCol = i % m_leadingDim;
                        }
                        continue;
                    }

                    // Checking for Infs.
                    // Above val1 == val2 check has failed, hence both values
                    // are not equal or Inf since Inf == Inf returns true.
                    // Now, if one of the values is Inf, the case is treated as
                    // a mismatch and failure is reported.
                    if (std::isinf(val1) || std::isinf(val2)) {
                        result.equal = false;
                        if (!opts.verbose) {
                            return result;
                        }

                        double diff = std::abs(static_cast<double>(val1)
                                               - static_cast<double>(val2));
                        if (result.mismatches.size() < opts.maxMismatches) {
                            size_t       r = i / m_leadingDim;
                            size_t       c = i % m_leadingDim;
                            MismatchInfo info;
                            info.row          = r;
                            info.col          = c;
                            info.value1       = static_cast<double>(val1);
                            info.value2       = static_cast<double>(val2);
                            info.absDiff      = diff;
                            info.relativeDiff = 0.0;
                            result.mismatches.push_back(info);
                        }
                        result.mismatchCount++;
                        if (diff > result.maxAbsDiff) {
                            result.maxAbsDiff = diff;
                            result.maxDiffRow = i / m_leadingDim;
                            result.maxDiffCol = i % m_leadingDim;
                        }
                        continue;
                    }

                    double diff = std::abs(static_cast<double>(val1)
                                           - static_cast<double>(val2));
                    double maxAbs =
                        std::max(std::abs(static_cast<double>(val1)),
                                 std::abs(static_cast<double>(val2)));
                    double relDiff = (maxAbs > 0.0) ? (diff / maxAbs) : 0.0;

                    if (diff > absTol && relDiff > relTol) {
                        result.equal = false;

                        if (!opts.verbose) {
                            return result;
                        }

                        if (result.mismatches.size() < opts.maxMismatches) {
                            size_t       r = i / m_leadingDim;
                            size_t       c = i % m_leadingDim;
                            MismatchInfo info;
                            info.row          = r;
                            info.col          = c;
                            info.value1       = static_cast<double>(val1);
                            info.value2       = static_cast<double>(val2);
                            info.absDiff      = diff;
                            info.relativeDiff = relDiff;
                            result.mismatches.push_back(info);
                        }

                        result.mismatchCount++;
                        if (diff > result.maxAbsDiff) {
                            result.maxAbsDiff = diff;
                            result.maxRelDiff = relDiff;
                            result.maxDiffRow = i / m_leadingDim;
                            result.maxDiffCol = i % m_leadingDim;
                        }
                    }
                }
                break;
            }

            case MatrixType::s32: {
                const int32_t* data1 = reinterpret_cast<const int32_t*>(m_data);
                const int32_t* data2 =
                    reinterpret_cast<const int32_t*>(other.m_data);
                elementCount = m_dataSizeBytes / sizeof(int32_t);

                for (size_t i = 0; i < elementCount; ++i) {
                    if (data1[i] != data2[i]) {
                        result.equal = false;

                        if (!opts.verbose) {
                            return result;
                        }

                        double diff = std::abs(static_cast<double>(data1[i])
                                               - static_cast<double>(data2[i]));

                        if (result.mismatches.size() < opts.maxMismatches) {
                            size_t       r = i / m_leadingDim;
                            size_t       c = i % m_leadingDim;
                            MismatchInfo info;
                            info.row          = r;
                            info.col          = c;
                            info.value1       = static_cast<double>(data1[i]);
                            info.value2       = static_cast<double>(data2[i]);
                            info.absDiff      = diff;
                            info.relativeDiff = 0.0;
                            result.mismatches.push_back(info);
                        }

                        result.mismatchCount++;
                        if (diff > result.maxAbsDiff) {
                            result.maxAbsDiff = diff;
                            result.maxDiffRow = i / m_leadingDim;
                            result.maxDiffCol = i % m_leadingDim;
                        }
                    }
                }
                break;
            }

            case MatrixType::u32: {
                const uint32_t* data1 =
                    reinterpret_cast<const uint32_t*>(m_data);
                const uint32_t* data2 =
                    reinterpret_cast<const uint32_t*>(other.m_data);
                elementCount = m_dataSizeBytes / sizeof(uint32_t);

                for (size_t i = 0; i < elementCount; ++i) {
                    if (data1[i] != data2[i]) {
                        result.equal = false;

                        if (!opts.verbose) {
                            return result;
                        }

                        double diff = std::abs(static_cast<double>(data1[i])
                                               - static_cast<double>(data2[i]));

                        if (result.mismatches.size() < opts.maxMismatches) {
                            size_t       r = i / m_leadingDim;
                            size_t       c = i % m_leadingDim;
                            MismatchInfo info;
                            info.row          = r;
                            info.col          = c;
                            info.value1       = static_cast<double>(data1[i]);
                            info.value2       = static_cast<double>(data2[i]);
                            info.absDiff      = diff;
                            info.relativeDiff = 0.0;
                            result.mismatches.push_back(info);
                        }

                        result.mismatchCount++;
                        if (diff > result.maxAbsDiff) {
                            result.maxAbsDiff = diff;
                            result.maxDiffRow = i / m_leadingDim;
                            result.maxDiffCol = i % m_leadingDim;
                        }
                    }
                }
                break;
            }

            case MatrixType::s16: {
                const int16_t* data1 = reinterpret_cast<const int16_t*>(m_data);
                const int16_t* data2 =
                    reinterpret_cast<const int16_t*>(other.m_data);
                elementCount = m_dataSizeBytes / sizeof(int16_t);

                for (size_t i = 0; i < elementCount; ++i) {
                    if (data1[i] != data2[i]) {
                        result.equal = false;

                        if (!opts.verbose) {
                            return result;
                        }

                        double diff = std::abs(static_cast<double>(data1[i])
                                               - static_cast<double>(data2[i]));

                        if (result.mismatches.size() < opts.maxMismatches) {
                            size_t       r = i / m_leadingDim;
                            size_t       c = i % m_leadingDim;
                            MismatchInfo info;
                            info.row          = r;
                            info.col          = c;
                            info.value1       = static_cast<double>(data1[i]);
                            info.value2       = static_cast<double>(data2[i]);
                            info.absDiff      = diff;
                            info.relativeDiff = 0.0;
                            result.mismatches.push_back(info);
                        }

                        result.mismatchCount++;
                        if (diff > result.maxAbsDiff) {
                            result.maxAbsDiff = diff;
                            result.maxDiffRow = i / m_leadingDim;
                            result.maxDiffCol = i % m_leadingDim;
                        }
                    }
                }
                break;
            }

            case MatrixType::u16: {
                const uint16_t* data1 =
                    reinterpret_cast<const uint16_t*>(m_data);
                const uint16_t* data2 =
                    reinterpret_cast<const uint16_t*>(other.m_data);
                elementCount = m_dataSizeBytes / sizeof(uint16_t);

                for (size_t i = 0; i < elementCount; ++i) {
                    if (data1[i] != data2[i]) {
                        result.equal = false;

                        if (!opts.verbose) {
                            return result;
                        }

                        double diff = std::abs(static_cast<double>(data1[i])
                                               - static_cast<double>(data2[i]));

                        if (result.mismatches.size() < opts.maxMismatches) {
                            size_t       r = i / m_leadingDim;
                            size_t       c = i % m_leadingDim;
                            MismatchInfo info;
                            info.row          = r;
                            info.col          = c;
                            info.value1       = static_cast<double>(data1[i]);
                            info.value2       = static_cast<double>(data2[i]);
                            info.absDiff      = diff;
                            info.relativeDiff = 0.0;
                            result.mismatches.push_back(info);
                        }

                        result.mismatchCount++;
                        if (diff > result.maxAbsDiff) {
                            result.maxAbsDiff = diff;
                            result.maxDiffRow = i / m_leadingDim;
                            result.maxDiffCol = i % m_leadingDim;
                        }
                    }
                }
                break;
            }

            case MatrixType::s8: {
                const int8_t* data1 = reinterpret_cast<const int8_t*>(m_data);
                const int8_t* data2 =
                    reinterpret_cast<const int8_t*>(other.m_data);
                elementCount = m_dataSizeBytes / sizeof(int8_t);

                for (size_t i = 0; i < elementCount; ++i) {
                    if (data1[i] != data2[i]) {
                        result.equal = false;

                        if (!opts.verbose) {
                            return result;
                        }

                        double diff = std::abs(static_cast<double>(data1[i])
                                               - static_cast<double>(data2[i]));

                        if (result.mismatches.size() < opts.maxMismatches) {
                            size_t       r = i / m_leadingDim;
                            size_t       c = i % m_leadingDim;
                            MismatchInfo info;
                            info.row          = r;
                            info.col          = c;
                            info.value1       = static_cast<double>(data1[i]);
                            info.value2       = static_cast<double>(data2[i]);
                            info.absDiff      = diff;
                            info.relativeDiff = 0.0;
                            result.mismatches.push_back(info);
                        }

                        result.mismatchCount++;
                        if (diff > result.maxAbsDiff) {
                            result.maxAbsDiff = diff;
                            result.maxDiffRow = i / m_leadingDim;
                            result.maxDiffCol = i % m_leadingDim;
                        }
                    }
                }
                break;
            }

            case MatrixType::u8: {
                const uint8_t* data1 = reinterpret_cast<const uint8_t*>(m_data);
                const uint8_t* data2 =
                    reinterpret_cast<const uint8_t*>(other.m_data);
                elementCount = m_dataSizeBytes / sizeof(uint8_t);

                for (size_t i = 0; i < elementCount; ++i) {
                    if (data1[i] != data2[i]) {
                        result.equal = false;

                        if (!opts.verbose) {
                            return result;
                        }

                        double diff = std::abs(static_cast<double>(data1[i])
                                               - static_cast<double>(data2[i]));

                        if (result.mismatches.size() < opts.maxMismatches) {
                            size_t       r = i / m_leadingDim;
                            size_t       c = i % m_leadingDim;
                            MismatchInfo info;
                            info.row          = r;
                            info.col          = c;
                            info.value1       = static_cast<double>(data1[i]);
                            info.value2       = static_cast<double>(data2[i]);
                            info.absDiff      = diff;
                            info.relativeDiff = 0.0;
                            result.mismatches.push_back(info);
                        }

                        result.mismatchCount++;
                        if (diff > result.maxAbsDiff) {
                            result.maxAbsDiff = diff;
                            result.maxDiffRow = i / m_leadingDim;
                            result.maxDiffCol = i % m_leadingDim;
                        }
                    }
                }
                break;
            }

            case MatrixType::s4:
            case MatrixType::u4: {
                // For packed 4-bit types, compare byte by byte
                const uint8_t* data1 = reinterpret_cast<const uint8_t*>(m_data);
                const uint8_t* data2 =
                    reinterpret_cast<const uint8_t*>(other.m_data);
                elementCount = m_dataSizeBytes;

                for (size_t i = 0; i < elementCount; ++i) {
                    if (data1[i] != data2[i]) {
                        result.equal = false;

                        if (!opts.verbose) {
                            return result;
                        }

                        double diff = std::abs(static_cast<double>(data1[i])
                                               - static_cast<double>(data2[i]));

                        if (result.mismatches.size() < opts.maxMismatches) {
                            size_t       r = (i * 2) / m_leadingDim;
                            size_t       c = (i * 2) % m_leadingDim;
                            MismatchInfo info;
                            info.row          = r;
                            info.col          = c;
                            info.value1       = static_cast<double>(data1[i]);
                            info.value2       = static_cast<double>(data2[i]);
                            info.absDiff      = diff;
                            info.relativeDiff = 0.0;
                            result.mismatches.push_back(info);
                        }

                        result.mismatchCount++;
                        if (diff > result.maxAbsDiff) {
                            result.maxAbsDiff = diff;
                            result.maxDiffRow = (i * 2) / m_leadingDim;
                            result.maxDiffCol = (i * 2) % m_leadingDim;
                        }
                    }
                }
                break;
            }

            default:
                throw std::runtime_error(
                    "Unsupported matrix type for comparison");
        }

        return result;
    }

    /**
     * @brief Compare two matrices for equality
     */
    bool Matrix::operator==(const Matrix& other) const
    {
        return compare(other, MatrixCompareOptions::Fast()).equal;
    }

    /**
     * @brief Compare two matrices for inequality
     */
    bool Matrix::operator!=(const Matrix& other) const
    {
        return !(*this == other);
    }

    /**
     * @brief Allocate aligned memory with proper size rounding
     */
    uint8_t* Matrix::allocateAlignedMemory(size_t sizeBytes, size_t alignment)
    {
        if (alignment > 0) {
            // Validate alignment requirements for std::aligned_alloc
            if ((alignment & (alignment - 1)) != 0) {
                throw std::invalid_argument("Alignment must be a power of 2");
            }
            if (alignment < sizeof(void*)) {
                throw std::invalid_argument(
                    "Alignment must be at least sizeof(void*)");
            }

            // Ensure size is a multiple of alignment for std::aligned_alloc
            size_t alignedSize = (sizeBytes + alignment - 1) & ~(alignment - 1);

            // Use C++17 aligned allocation for aligned memory
            uint8_t* data = static_cast<uint8_t*>(
                std::aligned_alloc(alignment, alignedSize));
            if (!data) {
                throw std::bad_alloc();
            }

            return data;
        } else {
            // Use regular new[] allocation for unaligned memory
            return new uint8_t[sizeBytes];
        }
    }

    /**
     * @brief Deallocate aligned memory
     */
    void Matrix::deallocateAlignedMemory(uint8_t* ptr, size_t alignment)
    {
        if (ptr) {
            if (alignment > 0) {
                // Memory was allocated with std::aligned_alloc
                std::free(ptr);
            } else {
                // Memory was allocated with new[]
                delete[] ptr;
            }
        }
    }

    /**
     * @brief Fill matrix with random values from a uniform distribution
     *
     * This is a convenience overload that delegates to the parameterized
     * version with default bounds and distribution.
     */
    void Matrix::fillRandom(unsigned int seed)
    {
        // Delegate to the parameterized version with default values
        fillRandom(seed, -5.0, 5.0, "uniform");
    }

    /**
     * @brief Fill matrix with random values from a specified distribution and
     * range
     */
    void Matrix::fillRandom(unsigned int       seed,
                            double             lb,
                            double             ub,
                            const std::string& dist,
                            bool               force_int_distribution)
    {
        if (!m_data || m_dataSizeBytes == 0) {
            return;
        }

        // Use time-based seed if not provided
        if (seed == 0) {
            seed = static_cast<unsigned int>(
                std::chrono::high_resolution_clock::now()
                    .time_since_epoch()
                    .count());
        }

        std::mt19937 gen(seed);

        // Validate distribution type
        bool use_normal = (dist == "normal");
        if (!use_normal && dist != "uniform") {
            throw std::invalid_argument(
                "Invalid distribution type. Use 'uniform' or 'normal'");
        }

        switch (m_type) {
            case MatrixType::f32: {
                float* data         = reinterpret_cast<float*>(m_data);
                size_t elementCount = m_dataSizeBytes / sizeof(float);

                if (use_normal) {
                    // For normal distribution, center mean at (lb+ub)/2
                    // Use stddev such that ~99.7% of values fall within [lb,
                    // ub] (mean ± 3*stddev covers 99.7% in normal distribution)
                    double                          mean   = (lb + ub) / 2.0;
                    double                          stddev = (ub - lb) / 6.0;
                    std::normal_distribution<float> dis(
                        static_cast<float>(mean), static_cast<float>(stddev));
                    for (size_t i = 0; i < elementCount; ++i) {
                        data[i] = dis(gen);
                    }
                } else if (force_int_distribution) {
                    // Use integer distribution for integral stress testing
                    int lb_i = static_cast<int>(std::floor(lb));
                    int ub_i = static_cast<int>(std::floor(ub));
                    std::uniform_int_distribution<int> dis(lb_i, ub_i);
                    for (size_t i = 0; i < elementCount; ++i) {
                        data[i] = static_cast<float>(dis(gen));
                    }
                } else {
                    std::uniform_real_distribution<float> dis(
                        static_cast<float>(lb), static_cast<float>(ub));
                    for (size_t i = 0; i < elementCount; ++i) {
                        data[i] = dis(gen);
                    }
                }
                break;
            }

            case MatrixType::bf16: {
                uint16_t* data         = reinterpret_cast<uint16_t*>(m_data);
                size_t    elementCount = m_dataSizeBytes / sizeof(uint16_t);

                if (use_normal) {
                    double                          mean   = (lb + ub) / 2.0;
                    double                          stddev = (ub - lb) / 6.0;
                    std::normal_distribution<float> dis(
                        static_cast<float>(mean), static_cast<float>(stddev));
                    for (size_t i = 0; i < elementCount; ++i) {
                        data[i] = f32_to_bf16(dis(gen));
                    }
                } else if (force_int_distribution) {
                    // Use integer distribution for integral stress testing
                    int lb_i = static_cast<int>(std::floor(lb));
                    int ub_i = static_cast<int>(std::floor(ub));
                    std::uniform_int_distribution<int> dis(lb_i, ub_i);
                    for (size_t i = 0; i < elementCount; ++i) {
                        data[i] = f32_to_bf16(static_cast<float>(dis(gen)));
                    }
                } else {
                    std::uniform_real_distribution<float> dis(
                        static_cast<float>(lb), static_cast<float>(ub));
                    for (size_t i = 0; i < elementCount; ++i) {
                        data[i] = f32_to_bf16(dis(gen));
                    }
                }
                break;
            }

            case MatrixType::u8: {
                uint8_t* data = m_data;
                int      lb_i = static_cast<int>(std::max(0.0, lb));
                int      ub_i = static_cast<int>(std::min(255.0, ub));

                if (use_normal) {
                    double                           mean   = (lb + ub) / 2.0;
                    double                           stddev = (ub - lb) / 6.0;
                    std::normal_distribution<double> dis(mean, stddev);
                    for (size_t i = 0; i < m_dataSizeBytes; ++i) {
                        double val = dis(gen);
                        data[i]    = static_cast<uint8_t>(
                            std::max(0.0, std::min(255.0, val)));
                    }
                } else {
                    std::uniform_int_distribution<int> dis(lb_i, ub_i);
                    for (size_t i = 0; i < m_dataSizeBytes; ++i) {
                        data[i] = static_cast<uint8_t>(dis(gen));
                    }
                }
                break;
            }

            case MatrixType::s8: {
                int8_t* data = reinterpret_cast<int8_t*>(m_data);
                int     lb_i = static_cast<int>(std::max(-128.0, lb));
                int     ub_i = static_cast<int>(std::min(127.0, ub));

                if (use_normal) {
                    double                           mean   = (lb + ub) / 2.0;
                    double                           stddev = (ub - lb) / 6.0;
                    std::normal_distribution<double> dis(mean, stddev);
                    for (size_t i = 0; i < m_dataSizeBytes; ++i) {
                        double val = dis(gen);
                        data[i]    = static_cast<int8_t>(
                            std::max(-128.0, std::min(127.0, val)));
                    }
                } else {
                    std::uniform_int_distribution<int> dis(lb_i, ub_i);
                    for (size_t i = 0; i < m_dataSizeBytes; ++i) {
                        data[i] = static_cast<int8_t>(dis(gen));
                    }
                }
                break;
            }

            case MatrixType::u16: {
                uint16_t* data         = reinterpret_cast<uint16_t*>(m_data);
                size_t    elementCount = m_dataSizeBytes / sizeof(uint16_t);
                int       lb_i         = static_cast<int>(std::max(0.0, lb));
                int       ub_i = static_cast<int>(std::min(65535.0, ub));

                if (use_normal) {
                    double                           mean   = (lb + ub) / 2.0;
                    double                           stddev = (ub - lb) / 6.0;
                    std::normal_distribution<double> dis(mean, stddev);
                    for (size_t i = 0; i < elementCount; ++i) {
                        double val = dis(gen);
                        data[i]    = static_cast<uint16_t>(
                            std::max(0.0, std::min(65535.0, val)));
                    }
                } else {
                    std::uniform_int_distribution<int> dis(lb_i, ub_i);
                    for (size_t i = 0; i < elementCount; ++i) {
                        data[i] = static_cast<uint16_t>(dis(gen));
                    }
                }
                break;
            }

            case MatrixType::s16: {
                int16_t* data         = reinterpret_cast<int16_t*>(m_data);
                size_t   elementCount = m_dataSizeBytes / sizeof(int16_t);
                int      lb_i = static_cast<int>(std::max(-32768.0, lb));
                int      ub_i = static_cast<int>(std::min(32767.0, ub));

                if (use_normal) {
                    double                           mean   = (lb + ub) / 2.0;
                    double                           stddev = (ub - lb) / 6.0;
                    std::normal_distribution<double> dis(mean, stddev);
                    for (size_t i = 0; i < elementCount; ++i) {
                        double val = dis(gen);
                        data[i]    = static_cast<int16_t>(
                            std::max(-32768.0, std::min(32767.0, val)));
                    }
                } else {
                    std::uniform_int_distribution<int> dis(lb_i, ub_i);
                    for (size_t i = 0; i < elementCount; ++i) {
                        data[i] = static_cast<int16_t>(dis(gen));
                    }
                }
                break;
            }

            case MatrixType::u32: {
                uint32_t* data         = reinterpret_cast<uint32_t*>(m_data);
                size_t    elementCount = m_dataSizeBytes / sizeof(uint32_t);
                uint32_t  lb_i = static_cast<uint32_t>(std::max(0.0, lb));
                uint32_t  ub_i = static_cast<uint32_t>(
                    std::min(static_cast<double>(UINT32_MAX), ub));

                if (use_normal) {
                    double                           mean   = (lb + ub) / 2.0;
                    double                           stddev = (ub - lb) / 6.0;
                    std::normal_distribution<double> dis(mean, stddev);
                    for (size_t i = 0; i < elementCount; ++i) {
                        double val = dis(gen);
                        data[i]    = static_cast<uint32_t>(std::max(
                            0.0,
                            std::min(static_cast<double>(UINT32_MAX), val)));
                    }
                } else {
                    std::uniform_int_distribution<uint32_t> dis(lb_i, ub_i);
                    for (size_t i = 0; i < elementCount; ++i) {
                        data[i] = dis(gen);
                    }
                }
                break;
            }

            case MatrixType::s32: {
                int32_t* data         = reinterpret_cast<int32_t*>(m_data);
                size_t   elementCount = m_dataSizeBytes / sizeof(int32_t);
                int32_t  lb_i         = static_cast<int32_t>(
                    std::max(static_cast<double>(INT32_MIN), lb));
                int32_t ub_i = static_cast<int32_t>(
                    std::min(static_cast<double>(INT32_MAX), ub));

                if (use_normal) {
                    double                           mean   = (lb + ub) / 2.0;
                    double                           stddev = (ub - lb) / 6.0;
                    std::normal_distribution<double> dis(mean, stddev);
                    for (size_t i = 0; i < elementCount; ++i) {
                        double val = dis(gen);
                        data[i]    = static_cast<int32_t>(std::max(
                            static_cast<double>(INT32_MIN),
                            std::min(static_cast<double>(INT32_MAX), val)));
                    }
                } else {
                    std::uniform_int_distribution<int32_t> dis(lb_i, ub_i);
                    for (size_t i = 0; i < elementCount; ++i) {
                        data[i] = dis(gen);
                    }
                }
                break;
            }

            case MatrixType::u4:
            case MatrixType::s4: {
                // For packed 4-bit types, generate values in 0-15 range or
                // -8-7 range
                uint8_t* data = m_data;
                int      lb_i = (m_type == MatrixType::s4)
                                    ? static_cast<int>(std::max(-8.0, lb))
                                    : static_cast<int>(std::max(0.0, lb));
                int      ub_i = (m_type == MatrixType::s4)
                                    ? static_cast<int>(std::min(7.0, ub))
                                    : static_cast<int>(std::min(15.0, ub));

                if (use_normal) {
                    double                           mean   = (lb + ub) / 2.0;
                    double                           stddev = (ub - lb) / 6.0;
                    std::normal_distribution<double> dis(mean, stddev);
                    for (size_t i = 0; i < m_dataSizeBytes; ++i) {
                        // Pack two 4-bit values per byte
                        double  val1 = dis(gen);
                        double  val2 = dis(gen);
                        uint8_t nibble1 =
                            static_cast<uint8_t>(std::max(
                                static_cast<double>(lb_i),
                                std::min(static_cast<double>(ub_i), val1)))
                            & 0x0F;
                        uint8_t nibble2 =
                            static_cast<uint8_t>(std::max(
                                static_cast<double>(lb_i),
                                std::min(static_cast<double>(ub_i), val2)))
                            & 0x0F;
                        data[i] = nibble1 | (nibble2 << 4);
                    }
                } else {
                    std::uniform_int_distribution<int> dis(lb_i, ub_i);
                    for (size_t i = 0; i < m_dataSizeBytes; ++i) {
                        uint8_t nibble1 = static_cast<uint8_t>(dis(gen)) & 0x0F;
                        uint8_t nibble2 = static_cast<uint8_t>(dis(gen)) & 0x0F;
                        data[i]         = nibble1 | (nibble2 << 4);
                    }
                }
                break;
            }

            default:
                throw std::runtime_error("Unsupported matrix type for random "
                                         "fill with custom range");
        }
    }

    /**
     * @brief Fill matrix with a single value
     */
    void Matrix::fillValue(double value)
    {
        if (!m_data || m_dataSizeBytes == 0) {
            return;
        }

        switch (m_type) {
            case MatrixType::f32: {
                float  fillValue    = static_cast<float>(value);
                float* data         = reinterpret_cast<float*>(m_data);
                size_t elementCount = m_dataSizeBytes / sizeof(float);
                for (size_t i = 0; i < elementCount; ++i) {
                    data[i] = fillValue;
                }
                break;
            }
            case MatrixType::u8: {
                uint8_t  fillValue = static_cast<uint8_t>(value);
                uint8_t* data      = m_data;
                for (size_t i = 0; i < m_dataSizeBytes; ++i) {
                    data[i] = fillValue;
                }
                break;
            }
            case MatrixType::s8: {
                int8_t  fillValue = static_cast<int8_t>(value);
                int8_t* data      = reinterpret_cast<int8_t*>(m_data);
                for (size_t i = 0; i < m_dataSizeBytes; ++i) {
                    data[i] = fillValue;
                }
                break;
            }
            case MatrixType::s32: {
                int32_t  fillValue = static_cast<int32_t>(value);
                int32_t* data      = reinterpret_cast<int32_t*>(m_data);
                for (size_t i = 0; i < m_dataSizeBytes / sizeof(int32_t); ++i) {
                    data[i] = fillValue;
                }
                break;
            }
            case MatrixType::bf16: {
                // For BF16, convert double to float first, then to bfloat16
                float     fillValue          = static_cast<float>(value);
                bfloat16  fillValue_bfloat16 = f32_to_bf16(fillValue);
                bfloat16* data         = reinterpret_cast<bfloat16*>(m_data);
                size_t    elementCount = m_dataSizeBytes / sizeof(bfloat16);
                for (size_t i = 0; i < elementCount; ++i) {
                    data[i] = fillValue_bfloat16;
                }
                break;
            }
            case MatrixType::u16: {
                uint16_t  fillValue    = static_cast<uint16_t>(value);
                uint16_t* data         = reinterpret_cast<uint16_t*>(m_data);
                size_t    elementCount = m_dataSizeBytes / sizeof(uint16_t);
                for (size_t i = 0; i < elementCount; ++i) {
                    data[i] = fillValue;
                }
                break;
            }
            case MatrixType::s16: {
                int16_t  fillValue    = static_cast<int16_t>(value);
                int16_t* data         = reinterpret_cast<int16_t*>(m_data);
                size_t   elementCount = m_dataSizeBytes / sizeof(int16_t);
                for (size_t i = 0; i < elementCount; ++i) {
                    data[i] = fillValue;
                }
                break;
            }
            case MatrixType::u32: {
                uint32_t  fillValue    = static_cast<uint32_t>(value);
                uint32_t* data         = reinterpret_cast<uint32_t*>(m_data);
                size_t    elementCount = m_dataSizeBytes / sizeof(uint32_t);
                for (size_t i = 0; i < elementCount; ++i) {
                    data[i] = fillValue;
                }
                break;
            }
            case MatrixType::s4:
            case MatrixType::u4: {
                // For packed 4-bit types, pack two values per byte
                uint8_t* data          = m_data;
                uint8_t  fillValue4bit = static_cast<uint8_t>(value) & 0x0F;
                uint8_t  packedValue   = fillValue4bit | (fillValue4bit << 4);
                for (size_t i = 0; i < m_dataSizeBytes; ++i) {
                    data[i] = packedValue;
                }
                break;
            }
            default:
                throw std::runtime_error(
                    "Unsupported matrix type for value fill");
        }
    }

    /**
     * @brief Convert matrix to string representation
     */
    std::string Matrix::matrixToString(VerbosityLevel verbosity_level) const
    {
        std::ostringstream oss;
        printToStream(oss, verbosity_level);
        return oss.str();
    }

    /**
     * @brief Core matrix printing implementation using std::ostream
     */
    void Matrix::printToStream(std::ostream&  os,
                               VerbosityLevel verbosity_level) const
    {
        if (verbosity_level < VerbosityLevel::PARTIAL_MATRIX) {
            return; // No output at low verbosity
        }

        // Print metadata
        os << "Matrix(" << m_rows << "x" << m_cols << ", type=" << m_type
           << ", layout=" << m_layout << ", ld=" << m_leadingDim;

        if (m_transposed)
            os << ", transposed";
        if (m_reordered)
            os << ", reordered";
        if (m_packed)
            os << ", packed";
        os << ")\n";

        // Format matrix data based on type
        formatMatrixData(os, verbosity_level);
        os << "\n";
    }

    /**
     * @brief Print matrix contents based on verbosity level
     */
    void Matrix::printMatrix(const std::string& name,
                             VerbosityLevel     verbosity_level) const
    {
        if (!name.empty()) {
            std::cout << "\n=== " << name << " ===\n";
        }
        printToStream(std::cout, verbosity_level);
    }
    /**
     * @brief Format matrix data based on type
     */
    void Matrix::formatMatrixData(std::ostream&  os,
                                  VerbosityLevel verbosity_level) const
    {
        switch (m_type) {
            case MatrixType::f32:
                formatNumericMatrix<float>(os, verbosity_level);
                break;
            case MatrixType::bf16:
                formatMatrixBF16(os, verbosity_level);
                break;
            case MatrixType::s32:
                formatNumericMatrix<int32_t>(os, verbosity_level);
                break;
            case MatrixType::u32:
                formatNumericMatrix<uint32_t>(os, verbosity_level);
                break;
            case MatrixType::s16:
                formatNumericMatrix<int16_t>(os, verbosity_level);
                break;
            case MatrixType::u16:
                formatNumericMatrix<uint16_t>(os, verbosity_level);
                break;
            case MatrixType::s8:
                formatNumericMatrix<int8_t>(os, verbosity_level);
                break;
            case MatrixType::u8:
                formatNumericMatrix<uint8_t>(os, verbosity_level);
                break;
            case MatrixType::s4:
            case MatrixType::u4:
                formatMatrix4Bit(os, verbosity_level);
                break;
            default:
                os << "[Matrix type not supported for printing]\n";
        }
    }

    /**
     * @brief Template formatter for numeric matrix types
     */
    template<typename T>
    void Matrix::formatNumericMatrix(std::ostream&  os,
                                     VerbosityLevel verbosity_level) const
    {
        md_t max_rows = (verbosity_level >= VerbosityLevel::FULL_MATRIX) ? 50
                                                                         : 5;
        md_t max_cols = (verbosity_level >= VerbosityLevel::FULL_MATRIX) ? 50
                                                                         : 5;

        md_t rows_to_print = std::min(m_rows, max_rows);
        md_t cols_to_print = std::min(m_cols, max_cols);

        const T* data = reinterpret_cast<const T*>(m_data);

        for (md_t i = 0; i < rows_to_print; ++i) {
            for (md_t j = 0; j < cols_to_print; ++j) {
                size_t idx = i * m_leadingDim + j;

                if constexpr (std::is_floating_point_v<T>) {
                    os << std::setw(10) << std::fixed << std::setprecision(4)
                       << data[idx];
                } else if constexpr (sizeof(T) == 1) {
                    // Byte types - show as integers to avoid char
                    // interpretation
                    os << std::setw(6) << static_cast<int>(data[idx]);
                } else {
                    os << std::setw(8) << data[idx];
                }
            }
            os << "\n";
        }

        if (m_rows > max_rows || m_cols > max_cols) {
            os << "[Showing " << rows_to_print << "x" << cols_to_print << " of "
               << m_rows << "x" << m_cols << " total";
            if (verbosity_level < VerbosityLevel::FULL_MATRIX) {
                os << " - use -vvv for full matrix";
            }
            os << "]\n";
        }
    }

    /**
     * @brief Specialized formatter for BF16 matrices
     */
    void Matrix::formatMatrixBF16(std::ostream&  os,
                                  VerbosityLevel verbosity_level) const
    {
        md_t max_rows = (verbosity_level >= VerbosityLevel::FULL_MATRIX) ? 50
                                                                         : 5;
        md_t max_cols = (verbosity_level >= VerbosityLevel::FULL_MATRIX) ? 50
                                                                         : 5;

        md_t rows_to_print = std::min(m_rows, max_rows);
        md_t cols_to_print = std::min(m_cols, max_cols);

        const bfloat16* data = reinterpret_cast<const bfloat16*>(m_data);

        for (md_t i = 0; i < rows_to_print; ++i) {
            for (md_t j = 0; j < cols_to_print; ++j) {
                size_t idx = i * m_leadingDim + j;
                float  f32_val =
                    bf16_to_f32(data[idx]); // Reuse existing conversion
                os << std::setw(10) << std::fixed << std::setprecision(4)
                   << f32_val;
            }
            os << "\n";
        }

        if (m_rows > max_rows || m_cols > max_cols) {
            os << "[Showing " << rows_to_print << "x" << cols_to_print << " of "
               << m_rows << "x" << m_cols << " total]\n";
        }
    }

    /**
     * @brief Specialized formatter for 4-bit packed matrices
     */
    void Matrix::formatMatrix4Bit(std::ostream&  os,
                                  VerbosityLevel verbosity_level) const
    {
        md_t max_rows = (verbosity_level >= VerbosityLevel::FULL_MATRIX) ? 50
                                                                         : 5;
        md_t max_cols = (verbosity_level >= VerbosityLevel::FULL_MATRIX) ? 50
                                                                         : 5;

        md_t rows_to_print = std::min(m_rows, max_rows);
        md_t cols_to_print = std::min(m_cols, max_cols);

        const uint8_t* data = m_data;

        for (md_t i = 0; i < rows_to_print; ++i) {
            for (md_t j = 0; j < cols_to_print; ++j) {
                // Unpack 4-bit value
                size_t elem_idx   = i * m_leadingDim + j;
                size_t byte_idx   = elem_idx / 2;
                size_t bit_offset = (elem_idx % 2) * 4;

                uint8_t nibble = (data[byte_idx] >> bit_offset) & 0x0F;

                // Convert to signed if s4
                if (m_type == MatrixType::s4 && (nibble & 0x08)) {
                    int8_t signed_val = nibble | 0xF0; // Sign extend
                    os << std::setw(5) << static_cast<int>(signed_val);
                } else {
                    os << std::setw(5) << static_cast<int>(nibble);
                }
            }
            os << "\n";
        }

        if (m_rows > max_rows || m_cols > max_cols) {
            os << "[4-bit packed - showing " << rows_to_print << "x"
               << cols_to_print << " of " << m_rows << "x" << m_cols
               << " total]\n";
        }
    }

    /**
     * @brief Fill matrix with a repeating pattern
     */
    void Matrix::fillPattern(const std::vector<double>& pattern)
    {
        if (pattern.empty()) {
            throw std::invalid_argument("Pattern cannot be empty");
        }
        if (!m_data || m_dataSizeBytes == 0) {
            return;
        }

        size_t pattern_size = pattern.size();

        switch (m_type) {
            case MatrixType::f32: {
                float* data         = reinterpret_cast<float*>(m_data);
                size_t elementCount = m_dataSizeBytes / sizeof(float);
                for (size_t i = 0; i < elementCount; ++i) {
                    data[i] = static_cast<float>(pattern[i % pattern_size]);
                }
                break;
            }
            case MatrixType::bf16: {
                uint16_t* data         = reinterpret_cast<uint16_t*>(m_data);
                size_t    elementCount = m_dataSizeBytes / sizeof(uint16_t);
                for (size_t i = 0; i < elementCount; ++i) {
                    float f32_val =
                        static_cast<float>(pattern[i % pattern_size]);
                    data[i] = f32_to_bf16(f32_val);
                }
                break;
            }
            case MatrixType::u8: {
                uint8_t* data = m_data;
                for (size_t i = 0; i < m_dataSizeBytes; ++i) {
                    double val = pattern[i % pattern_size];
                    data[i]    = static_cast<uint8_t>(
                        std::max(0.0, std::min(255.0, std::round(val))));
                }
                break;
            }
            case MatrixType::s8: {
                int8_t* data = reinterpret_cast<int8_t*>(m_data);
                for (size_t i = 0; i < m_dataSizeBytes; ++i) {
                    double val = pattern[i % pattern_size];
                    data[i]    = static_cast<int8_t>(
                        std::max(-128.0, std::min(127.0, std::round(val))));
                }
                break;
            }
            case MatrixType::u16: {
                uint16_t* data         = reinterpret_cast<uint16_t*>(m_data);
                size_t    elementCount = m_dataSizeBytes / sizeof(uint16_t);
                for (size_t i = 0; i < elementCount; ++i) {
                    double val = pattern[i % pattern_size];
                    data[i]    = static_cast<uint16_t>(
                        std::max(0.0, std::min(65535.0, std::round(val))));
                }
                break;
            }
            case MatrixType::s16: {
                int16_t* data         = reinterpret_cast<int16_t*>(m_data);
                size_t   elementCount = m_dataSizeBytes / sizeof(int16_t);
                for (size_t i = 0; i < elementCount; ++i) {
                    double val = pattern[i % pattern_size];
                    data[i]    = static_cast<int16_t>(
                        std::max(-32768.0, std::min(32767.0, std::round(val))));
                }
                break;
            }
            case MatrixType::u32: {
                uint32_t* data         = reinterpret_cast<uint32_t*>(m_data);
                size_t    elementCount = m_dataSizeBytes / sizeof(uint32_t);
                for (size_t i = 0; i < elementCount; ++i) {
                    double val = pattern[i % pattern_size];
                    data[i]    = static_cast<uint32_t>(
                        std::max(0.0, std::min(static_cast<double>(UINT32_MAX),
                                                  std::round(val))));
                }
                break;
            }
            case MatrixType::s32: {
                int32_t* data         = reinterpret_cast<int32_t*>(m_data);
                size_t   elementCount = m_dataSizeBytes / sizeof(int32_t);
                for (size_t i = 0; i < elementCount; ++i) {
                    double val = pattern[i % pattern_size];
                    data[i]    = static_cast<int32_t>(
                        std::max(static_cast<double>(INT32_MIN),
                                    std::min(static_cast<double>(INT32_MAX),
                                             std::round(val))));
                }
                break;
            }
            case MatrixType::s4:
            case MatrixType::u4: {
                // For packed 4-bit types, pack two values per byte
                uint8_t* data = m_data;
                for (size_t i = 0; i < m_dataSizeBytes; ++i) {
                    // Get two pattern values for this byte
                    double val1 = pattern[(i * 2) % pattern_size];
                    double val2 = pattern[(i * 2 + 1) % pattern_size];

                    uint8_t nibble1, nibble2;
                    if (m_type == MatrixType::s4) {
                        // Signed 4-bit: range [-8, 7]
                        nibble1 = static_cast<int8_t>(std::max(
                                      -8.0, std::min(7.0, std::round(val1))))
                                  & 0x0F;
                        nibble2 = static_cast<int8_t>(std::max(
                                      -8.0, std::min(7.0, std::round(val2))))
                                  & 0x0F;
                    } else {
                        // Unsigned 4-bit: range [0, 15]
                        nibble1 = static_cast<uint8_t>(std::max(
                                      0.0, std::min(15.0, std::round(val1))))
                                  & 0x0F;
                        nibble2 = static_cast<uint8_t>(std::max(
                                      0.0, std::min(15.0, std::round(val2))))
                                  & 0x0F;
                    }
                    data[i] = nibble1 | (nibble2 << 4);
                }
                break;
            }
            default:
                throw std::runtime_error(
                    "Unsupported matrix type for pattern fill");
        }
    }

    /**
     * @brief Format comparison result as a human-readable string
     */
    std::string FormatCompareResult(const MatrixCompareResult& result,
                                    const Matrix&              matrix1,
                                    const Matrix&              matrix2)
    {
        std::ostringstream output;

        // Header
        output << "Matrix Comparison Report\n";
        output << "========================\n\n";

        // Matrix information
        output << "Matrix 1: " << matrix1.getRows() << "x" << matrix1.getCols()
               << ", type=" << matrix1.getMatrixType()
               << ", layout=" << matrix1.getLayout()
               << ", ld=" << matrix1.getLeadingDimension() << "\n";
        output << "Matrix 2: " << matrix2.getRows() << "x" << matrix2.getCols()
               << ", type=" << matrix2.getMatrixType()
               << ", layout=" << matrix2.getLayout()
               << ", ld=" << matrix2.getLeadingDimension() << "\n\n";

        // Overall result
        output << "Result: " << (result.equal ? "EQUAL" : "NOT EQUAL")
               << "\n\n";

        // Mismatch categorization
        if (result.dimensionMismatch) {
            output << "Cause: Dimension mismatch\n";
            return output.str();
        }

        if (result.typeMismatch) {
            output << "Cause: Type mismatch\n";
            return output.str();
        }

        if (result.layoutMismatch) {
            output << "Cause: Layout mismatch\n";
            return output.str();
        }

        if (result.metadataMismatch) {
            output << "Cause: Metadata mismatch (leading dimension, "
                      "transposed, reordered, or packed flags)\n";
            return output.str();
        }

        // Tolerance information
        if (matrix1.getMatrixType() == MatrixType::f32
            || matrix1.getMatrixType() == MatrixType::bf16) {
            output << "Tolerances used:\n";
            output << "  Absolute tolerance: " << std::scientific
                   << std::setprecision(6) << result.usedAbsTolerance << "\n";
            output << "  Relative tolerance: " << std::scientific
                   << std::setprecision(6) << result.usedRelTolerance << "\n\n";
        }

        // Statistics
        if (result.mismatchCount > 0) {
            output << "Mismatch Statistics:\n";
            output << "  Total mismatches: " << result.mismatchCount << "\n";
            output << "  Maximum absolute difference: " << std::scientific
                   << std::setprecision(8) << result.maxAbsDiff << " at ["
                   << result.maxDiffRow << "," << result.maxDiffCol << "]\n";
            if (result.maxRelDiff > 0.0) {
                output << "  Maximum relative difference: " << std::scientific
                       << std::setprecision(8) << result.maxRelDiff << "\n";
            }
            output << "\n";

            // Detailed mismatches
            if (!result.mismatches.empty()) {
                output << "Detailed Mismatches (showing first "
                       << result.mismatches.size() << "):\n";
                output << "  Row  Col  Value1              Value2              "
                          "AbsDiff             RelDiff\n";
                output << "  "
                          "---  ---  ------------------  ------------------  "
                          "------------------  ------------------\n";

                for (const auto& mismatch : result.mismatches) {
                    output << "  " << std::setw(3) << mismatch.row << "  "
                           << std::setw(3) << mismatch.col << "  "
                           << std::scientific << std::setprecision(10)
                           << std::setw(18) << mismatch.value1 << "  "
                           << std::setw(18) << mismatch.value2 << "  "
                           << std::setw(18) << mismatch.absDiff << "  ";
                    if (mismatch.relativeDiff > 0.0) {
                        output << std::setw(18) << mismatch.relativeDiff;
                    } else {
                        output << std::setw(18) << "N/A";
                    }
                    output << "\n";
                }

                if (result.mismatchCount > result.mismatches.size()) {
                    output << "  ... ("
                           << (result.mismatchCount - result.mismatches.size())
                           << " more mismatches not shown)\n";
                }
            }
        } else {
            output << "No mismatches found (matrices are equal)\n";
        }

        return output.str();
    }

}}} // namespace dlp::testing::framework
