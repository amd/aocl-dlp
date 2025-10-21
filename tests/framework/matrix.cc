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
#include <any>
#include <chrono> // For time-based seeding
#include <cmath>  // For std::abs
#include <cstdint>
#include <cstdlib> // For std::aligned_alloc, std::free
#include <cstring> // For std::memcpy
#include <limits>  // For std::numeric_limits
#include <random>  // For random number generation
#include <stdexcept>

using dlp::testing::utils::bf16_to_f32;
using dlp::testing::utils::f32_to_bf16;

namespace dlp { namespace testing { namespace framework {
    using dlp::testing::utils::bf16_to_f32;
    using dlp::testing::utils::f32_to_bf16;

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
        // Calculate leading dimension if not specified
        if (leadingDim == 0) {
            m_leadingDim = (layout == MatrixLayout::ROW_MAJOR) ? cols : rows;
        } else {
            m_leadingDim = leadingDim;
        }

        // Calculate required memory size
        m_dataSizeBytes = MatrixMemory::calculateRequiredBytes(
            type, rows, cols, layout, m_leadingDim);

        // Allocate memory using helper method
        m_data = allocateAlignedMemory(m_dataSizeBytes, alignment);

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
    void Matrix::setK(md_t k)
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
     * @brief Compare two matrices for equality
     */
    bool Matrix::operator==(const Matrix& other) const
    {
        // Check dimensions and metadata
        if (m_rows != other.m_rows || m_cols != other.m_cols
            || m_type != other.m_type || m_layout != other.m_layout
            || m_leadingDim != other.m_leadingDim
            || m_transposed != other.m_transposed
            || m_reordered != other.m_reordered || m_packed != other.m_packed) {
            return false;
        }

        // Check data content
        if (m_dataSizeBytes != other.m_dataSizeBytes) {
            return false;
        }

        if (m_data == nullptr && other.m_data == nullptr) {
            return true;
        }

        if (m_data == nullptr || other.m_data == nullptr) {
            return false;
        }

        // For floating point types, use tolerance-based comparison
        if (m_type == MatrixType::f32 || m_type == MatrixType::bf16) {
            return compareFloatingPointData(other);
        } else if (m_type == MatrixType::s32) {
            for (size_t i = 0; i < m_dataSizeBytes / sizeof(int32_t); ++i) {
                if (reinterpret_cast<const int32_t*>(m_data)[i]
                    != reinterpret_cast<const int32_t*>(other.m_data)[i]) {
                    return false;
                }
            }
        }

        // For integer types, use exact comparison
        return std::memcmp(m_data, other.m_data, m_dataSizeBytes) == 0;
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
     * @brief Helper function for floating point data comparison
     */
    bool Matrix::compareFloatingPointData(const Matrix& other) const
    {
        if (m_type == MatrixType::f32) {
            const float* thisData = reinterpret_cast<const float*>(m_data);
            const float* otherData =
                reinterpret_cast<const float*>(other.m_data);
            size_t elementCount = m_dataSizeBytes / sizeof(float);

            // Base tolerance using machine epsilon with safety factor
            double baseTolerance = 10.0 * std::numeric_limits<float>::epsilon();

            for (size_t i = 0; i < elementCount; ++i) {
                if (thisData[i] == otherData[i]) {
                    continue;
                }

                float maxAbs =
                    std::max(std::abs(thisData[i]), std::abs(otherData[i]));
                if (maxAbs == 0.0f) {
                    continue; // Both are zero
                }

                float relativeDiff =
                    std::abs(thisData[i] - otherData[i]) / maxAbs;
                double tolerance = baseTolerance * maxAbs;
                if (relativeDiff > tolerance) {
                    return false;
                }
            }
        } else if (m_type == MatrixType::bf16) {
            const bfloat16* thisData =
                reinterpret_cast<const bfloat16*>(m_data);
            const bfloat16* otherData =
                reinterpret_cast<const bfloat16*>(other.m_data);
            size_t elementCount = m_dataSizeBytes / sizeof(bfloat16);

            // To approximate bf16's lower precision, we use a much larger
            // multiplier for float's epsilon. 100000 * 1.2e-7 ≈ 1.2e-2
            double baseTolerance =
                100000.0 * std::numeric_limits<float>::epsilon();

            for (size_t i = 0; i < elementCount; ++i) {
                float thisFloat  = bf16_to_f32(thisData[i]);
                float otherFloat = bf16_to_f32(otherData[i]);

                if (thisFloat == otherFloat) {
                    continue;
                }

                float maxAbs =
                    std::max(std::abs(thisFloat), std::abs(otherFloat));
                if (maxAbs == 0.0f) {
                    continue; // Both are zero, already handled by equality
                              // check
                }

                float  relativeDiff = std::abs(thisFloat - otherFloat) / maxAbs;
                double tolerance    = baseTolerance * maxAbs;

                if (relativeDiff > tolerance) {
                    return false;
                }
            }
        }

        return true;
    }

    /**
     * @brief Fill matrix with random values from a uniform distribution
     */
    void Matrix::fillRandom(unsigned int seed)
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

        switch (m_type) {
            case MatrixType::f32: {
#if DLP_TESTING_ENABLE_HIGH_PRECISION_FLOAT
                std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
                float* data         = reinterpret_cast<float*>(m_data);
                size_t elementCount = m_dataSizeBytes / sizeof(float);
                for (size_t i = 0; i < elementCount; ++i) {
                    float value = dis(gen);
                    data[i]     = value;
                }
                break;
#else
                std::uniform_real_distribution<float> dis(-20.0f, 20.0f);
                float* data         = reinterpret_cast<float*>(m_data);
                size_t elementCount = m_dataSizeBytes / sizeof(float);
                for (size_t i = 0; i < elementCount; ++i) {
                    // Generate float value and truncate to remove fractional
                    // part
                    float value = dis(gen);
                    data[i]     = std::trunc(value); // Truncates toward zero
                }
                break;
#endif
            }
            case MatrixType::u8: {
                std::uniform_int_distribution<uint8_t> dis(0, 255);
                uint8_t*                               data = m_data;
                for (size_t i = 0; i < m_dataSizeBytes; ++i) {
                    data[i] = dis(gen);
                }
                break;
            }
            case MatrixType::s8: {
                std::uniform_int_distribution<int8_t> dis(-128, 127);
                int8_t* data = reinterpret_cast<int8_t*>(m_data);
                for (size_t i = 0; i < m_dataSizeBytes; ++i) {
                    data[i] = dis(gen);
                }
                break;
            }
            case MatrixType::u16: {
                std::uniform_int_distribution<uint16_t> dis(0, 65535);
                uint16_t* data         = reinterpret_cast<uint16_t*>(m_data);
                size_t    elementCount = m_dataSizeBytes / sizeof(uint16_t);
                for (size_t i = 0; i < elementCount; ++i) {
                    data[i] = dis(gen);
                }
                break;
            }
            case MatrixType::s16: {
                std::uniform_int_distribution<int16_t> dis(-32768, 32767);
                int16_t* data         = reinterpret_cast<int16_t*>(m_data);
                size_t   elementCount = m_dataSizeBytes / sizeof(int16_t);
                for (size_t i = 0; i < elementCount; ++i) {
                    data[i] = dis(gen);
                }
                break;
            }
            case MatrixType::u32: {
                std::uniform_int_distribution<uint32_t> dis(0, UINT32_MAX);
                uint32_t* data         = reinterpret_cast<uint32_t*>(m_data);
                size_t    elementCount = m_dataSizeBytes / sizeof(uint32_t);
                for (size_t i = 0; i < elementCount; ++i) {
                    data[i] = dis(gen);
                }
                break;
            }
            case MatrixType::s32: {
                std::uniform_int_distribution<int32_t> dis(INT32_MIN,
                                                           INT32_MAX);
                int32_t* data         = reinterpret_cast<int32_t*>(m_data);
                size_t   elementCount = m_dataSizeBytes / sizeof(int32_t);
                for (size_t i = 0; i < elementCount; ++i) {
                    data[i] = dis(gen);
                }
                break;
            }
            case MatrixType::bf16: {
#if DLP_TESTING_ENABLE_HIGH_PRECISION_FLOAT // Enable this to get a proper test
                                            // case.
                // For BF16, fill with random uint16_t values
                std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
                uint16_t* data         = reinterpret_cast<uint16_t*>(m_data);
                size_t    elementCount = m_dataSizeBytes / sizeof(uint16_t);
                for (size_t i = 0; i < elementCount; ++i) {
                    data[i] = f32_to_bf16(dis(gen));
                }
                break;
#else
                // For BF16, fill with random uint16_t values
                std::uniform_int_distribution<int16_t> dis(-5, 5);
                uint16_t* data         = reinterpret_cast<uint16_t*>(m_data);
                size_t    elementCount = m_dataSizeBytes / sizeof(uint16_t);
                for (size_t i = 0; i < elementCount; ++i) {
                    data[i] = f32_to_bf16(static_cast<float>(dis(gen)));
                }
                break;
#endif
            }
            case MatrixType::u4:
            case MatrixType::s4: {
                // For packed 4-bit types, fill with random bytes
                std::uniform_int_distribution<uint8_t> dis(0, 255);
                uint8_t*                               data = m_data;
                for (size_t i = 0; i < m_dataSizeBytes; ++i) {
                    data[i] = dis(gen);
                }
                break;
            }
            default:
                throw std::runtime_error(
                    "Unsupported matrix type for random fill");
        }
    }

    /**
     * @brief Fill matrix with a single value
     */
    void Matrix::fillValue(std::any value)
    {
        if (!m_data || m_dataSizeBytes == 0) {
            return;
        }

        switch (m_type) {
            case MatrixType::f32: {
                float  fillValue    = std::any_cast<float>(value);
                float* data         = reinterpret_cast<float*>(m_data);
                size_t elementCount = m_dataSizeBytes / sizeof(float);
                for (size_t i = 0; i < elementCount; ++i) {
                    data[i] = fillValue;
                }
                break;
            }
            case MatrixType::u8: {
                uint8_t  fillValue = std::any_cast<uint8_t>(value);
                uint8_t* data      = m_data;
                for (size_t i = 0; i < m_dataSizeBytes; ++i) {
                    data[i] = fillValue;
                }
                break;
            }
            case MatrixType::s8: {
                int8_t  fillValue = std::any_cast<int8_t>(value);
                int8_t* data      = reinterpret_cast<int8_t*>(m_data);
                for (size_t i = 0; i < m_dataSizeBytes; ++i) {
                    data[i] = fillValue;
                }
                break;
            }
            case MatrixType::s32: {
                int32_t  fillValue = std::any_cast<int32_t>(value);
                int32_t* data      = reinterpret_cast<int32_t*>(m_data);
                for (size_t i = 0; i < m_dataSizeBytes / sizeof(int32_t); ++i) {
                    data[i] = fillValue;
                }
                break;
            }
            case MatrixType::bf16: {
                // For BF16, fill with random uint16_t values
                float     fillValue          = std::any_cast<float>(value);
                bfloat16  fillValue_bfloat16 = f32_to_bf16(fillValue);
                bfloat16* data         = reinterpret_cast<bfloat16*>(m_data);
                size_t    elementCount = m_dataSizeBytes / sizeof(bfloat16);
                for (size_t i = 0; i < elementCount; ++i) {
                    data[i] = fillValue_bfloat16;
                }
                break;
            }
            // Add other types as needed
            default:
                throw std::runtime_error(
                    "Unsupported matrix type for value fill");
        }
    }

}}} // namespace dlp::testing::framework
