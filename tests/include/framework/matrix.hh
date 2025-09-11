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

#pragma once

#include "classic/dlp_base_types.h"
#include "types.hh"
#include <any>
#include <limits>
#include <memory>
#include <vector>

namespace dlp { namespace testing { namespace framework {

    /**
     * @class Matrix
     * @brief Matrix class for handling various data types with layout and
     * transposition support
     *
     * The Matrix class provides a uniform interface for working with matrices
     * of different data types, memory layouts, and supports virtual
     * transposition without copying data. Memory is managed externally via
     * unique_ptr for better safety and flexibility.
     */
    class Matrix
    {
      public:
        /**
         * @brief Default constructor
         *
         * Creates an empty matrix with default values. The matrix will need to
         * be properly initialized before use (e.g., through assignment or by
         * providing external memory).
         */
        Matrix();

        /**
         * @brief Main constructor with external memory management
         *
         * Creates a matrix with specified dimensions, data type, layout, and
         * externally provided memory.
         *
         * @param rows Number of rows in the matrix
         * @param cols Number of columns in the matrix
         * @param type Data type of the matrix elements
         * @param data Unique pointer to pre-allocated memory
         * @param dataSizeBytes Size of the allocated memory in bytes
         * @param layout Memory layout (ROW_MAJOR or COLUMN_MAJOR)
         * @param leadingDim Leading dimension (0 for automatic calculation)
         * @param transposed Whether the matrix is logically transposed without
         * data movement
         * @param reordered Whether the matrix is reordered
         */
        Matrix(md_t         rows,
               md_t         cols,
               MatrixType   type,
               uint8_t*     data,
               size_t       dataSizeBytes,
               MatrixLayout layout     = MatrixLayout::ROW_MAJOR,
               md_t         leadingDim = 0,
               bool         transposed = false,
               bool         reordered  = false,
               size_t       alignment  = 0);

        /**
         * @brief Convenience constructor with automatic memory allocation
         *
         * Creates a matrix with automatic memory allocation. This is provided
         * for convenience but external memory management is preferred.
         *
         * @param rows Number of rows in the matrix
         * @param cols Number of columns in the matrix
         * @param type Data type of the matrix elements
         * @param layout Memory layout (ROW_MAJOR or COLUMN_MAJOR)
         * @param leadingDim Leading dimension (0 for automatic calculation)
         * @param transposed Whether the matrix is logically transposed
         * @param reordered Whether the matrix is reordered
         * @param alignment Memory alignment in bytes (0 for no special
         * alignment)
         */
        Matrix(md_t         rows,
               md_t         cols,
               MatrixType   type,
               MatrixLayout layout     = MatrixLayout::ROW_MAJOR,
               md_t         leadingDim = 0,
               bool         transposed = false,
               bool         reordered  = false,
               size_t       alignment  = 0);

        /**
         * @brief Copy constructor
         *
         * Creates a deep copy of the source matrix with newly allocated memory.
         *
         * @param other The matrix to copy from
         */
        Matrix(const Matrix& other);

        /**
         * @brief Move constructor
         *
         * Transfers ownership of memory from source matrix.
         *
         * @param other The matrix to move from
         */
        Matrix(Matrix&& other) noexcept;

        /**
         * @brief Destructor
         *
         * Memory is properly released based on allocation type.
         */
        ~Matrix();

        /**
         * @brief Copy assignment operator
         *
         * Creates a deep copy of the source matrix with newly allocated memory.
         *
         * @param other The matrix to copy from
         * @return Matrix& Reference to this matrix
         */
        Matrix& operator=(const Matrix& other);

        /**
         * @brief Move assignment operator
         *
         * Transfers ownership of memory from source matrix.
         *
         * @param other The matrix to move from
         * @return Matrix& Reference to this matrix
         */
        Matrix& operator=(Matrix&& other) noexcept;

        /**
         * @brief Get the number of rows in the matrix
         *
         * @return md_t The number of rows
         */
        md_t getRows() const;

        /**
         * @brief Get the number of columns in the matrix
         *
         * @return md_t The number of columns
         */
        md_t getCols() const;

        /**
         * @brief Get the matrix data type
         *
         * @return MatrixType The type of data stored in the matrix
         */
        MatrixType getMatrixType() const;

        /**
         * @brief Get the matrix memory layout
         *
         * @return MatrixLayout The memory layout (ROW_MAJOR or COLUMN_MAJOR)
         */
        MatrixLayout getLayout() const;

        /**
         * @brief Check if the matrix is logically transposed
         *
         * @return bool True if the matrix is transposed, false otherwise
         */
        bool isTransposed() const;

        /**
         * @brief Check if the matrix is reordered
         *
         * @return bool True if the matrix is reordered, false otherwise
         */
        bool isReordered() const;

        /**
         * @brief Check if the matrix is packed
         *
         * @return bool True if the matrix is packed, false otherwise
         */
        bool isPacked() const;

        /**
         * @brief Get the leading dimension of the matrix
         *
         * The leading dimension is the stride between consecutive rows (for
         * row-major) or consecutive columns (for column-major).
         *
         * @return md_t The leading dimension
         */
        md_t getLeadingDimension() const;

        /**
         * @brief Get the effective number of rows after considering
         * transposition
         *
         * @return md_t Number of rows (or columns if transposed)
         */
        md_t getEffectiveRows() const;

        /**
         * @brief Get the effective number of columns after considering
         * transposition
         *
         * @return md_t Number of columns (or rows if transposed)
         */
        md_t getEffectiveCols() const;

        /**
         * @brief Get the data size in bytes
         *
         * @return size_t The size of allocated memory in bytes
         */
        size_t getDataSizeBytes() const;

        /**
         * @brief Get raw pointer to the matrix data
         *
         * @return void* Pointer to the matrix data (type-erased)
         */
        void* getData() const;

        /**
         * @brief Get the matrix data container (for backward compatibility)
         *
         * @return MatrixData The matrix data structure with type information
         */
        MatrixData getMatrixData() const;

        /**
         * @brief Set the reordering flag
         *
         * @param reordered Whether the matrix is reordered
         */
        void setReordered(bool reordered);

        /**
         * @brief Set the packing flag
         *
         * @param packed Whether the matrix is packed
         */
        void setPacked(bool packed);

        /**
         * @brief Set the k dimension for tolerance calculation
         *
         * @param k The k dimension for tolerance calculation
         */
        void setK(md_t k);

        /**
         * @brief Compare two matrices for equality
         *
         * Checks if two matrices have the same dimensions, type, and content
         *
         * @param other The matrix to compare with
         * @return bool True if matrices are equal, false otherwise
         */
        bool operator==(const Matrix& other) const;

        /**
         * @brief Compare two matrices for inequality
         *
         * Checks if two matrices differ in dimensions, type, or content
         *
         * @param other The matrix to compare with
         * @return bool True if matrices are not equal, false otherwise
         */
        bool operator!=(const Matrix& other) const;

        /**
         * @brief Fill matrix with random values from a uniform distribution
         *
         * Fills the matrix with random values appropriate for its data type.
         * For integer types, values are within the type's range.
         * For floating point types, values are between -1.0 and 1.0.
         *
         * @param seed Optional seed for the random number generator (0 means
         * use time-based seed)
         */
        void fillRandom(unsigned int seed = 0);

        /**
         * @brief Fill matrix with a single value
         *
         * Fills the matrix with a single value of the appropriate data type.
         *
         * @param value The value to fill the matrix with
         */
        void fillValue(std::any value);

        /**
         * @brief Get element size in bytes for the matrix type
         *
         * @return size_t Size of a single element in bytes (0 for packed types)
         */
        size_t getElementSizeBytes() const;

        // TEMPLATED FACTORY METHODS FOR POSTOPS API

        // Type mapping traits
        template<typename T>
        struct DefaultMatrixType;

        // Enable if type is supported
        template<typename T>
        using EnableIfSupported = typename std::enable_if<
            std::is_same_v<T, float> || std::is_same_v<T, double>
                || std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>
                || std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>
                || std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>,
            int>::type;

        // Template method implementations are below

        // Template implementations (must be in header)
      public:
        template<typename T, EnableIfSupported<T> = 0>
        static Matrix fromValue(T          value,
                                MatrixType type = DefaultMatrixType<T>::value)
        {
            Matrix matrix(1, 1, type);
            writeDataToMatrix(matrix, &value, 1, 0);
            return matrix;
        }

        template<typename T, EnableIfSupported<T> = 0>
        static Matrix fromVector(const std::vector<T>& values,
                                 MatrixType type = DefaultMatrixType<T>::value,
                                 MatrixLayout layout = MatrixLayout::ROW_MAJOR)
        {
            if (values.empty()) {
                throw std::invalid_argument(
                    "Cannot create matrix from empty vector");
            }

            Matrix matrix(1, values.size(), type, layout);
            writeDataToMatrix(matrix, values.data(), values.size(), 0);
            return matrix;
        }

        template<typename T, EnableIfSupported<T> = 0>
        static Matrix fromData(const std::vector<std::vector<T>>& data,
                               MatrixType   type = DefaultMatrixType<T>::value,
                               MatrixLayout layout = MatrixLayout::ROW_MAJOR)
        {
            if (data.empty() || data[0].empty()) {
                throw std::invalid_argument(
                    "Cannot create matrix from empty data");
            }

            size_t rows = data.size();
            size_t cols = data[0].size();

            // Validate that all rows have the same number of columns
            for (size_t i = 1; i < rows; ++i) {
                if (data[i].size() != cols) {
                    throw std::invalid_argument(
                        "All rows must have the same number of columns");
                }
            }

            Matrix matrix(rows, cols, type, layout);

            // Write data row by row
            md_t leadingDim = matrix.getLeadingDimension();
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    size_t flatIndex;
                    if (layout == MatrixLayout::ROW_MAJOR) {
                        flatIndex = i * leadingDim + j;
                    } else {
                        flatIndex = j * leadingDim + i;
                    }
                    writeDataToMatrix(matrix, &data[i][j], 1, flatIndex);
                }
            }

            return matrix;
        }

        /**
         * @brief Convenience method to create a scalar matrix
         *
         * @param value The scalar value
         * @return Matrix A 1x1 f32 matrix containing the value
         */
        static Matrix scalar(float value)
        {
            return fromValue<float>(value, MatrixType::f32);
        }

        /**
         * @brief Convenience method to create a scalar matrix from integer
         *
         * @param value The scalar value
         * @return Matrix A 1x1 s32 matrix containing the value
         */
        static Matrix scalar(int32_t value)
        {
            return fromValue<int32_t>(value, MatrixType::s32);
        }

        /**
         * @brief Convenience method to create a vector matrix
         *
         * @param values Vector of float values
         * @return Matrix A row vector f32 matrix containing the values
         */
        static Matrix vector(const std::vector<float>& values)
        {
            return fromVector<float>(values, MatrixType::f32);
        }

        /**
         * @brief Convenience method to create a vector matrix from integers
         *
         * @param values Vector of integer values
         * @return Matrix A row vector s32 matrix containing the values
         */
        static Matrix vector(const std::vector<int32_t>& values)
        {
            return fromVector<int32_t>(values, MatrixType::s32);
        }

      private:
        template<typename SourceT>
        static void writeDataToMatrix(Matrix&        matrix,
                                      const SourceT* source,
                                      size_t         count,
                                      size_t         startIndex = 0)
        {
            MatrixType targetType = matrix.getMatrixType();
            void*      targetData = matrix.getData();

            switch (targetType) {
                case MatrixType::f32: {
                    float* data = reinterpret_cast<float*>(targetData);
                    for (size_t i = 0; i < count; ++i) {
                        data[startIndex + i] = static_cast<float>(source[i]);
                    }
                    break;
                }
                case MatrixType::bf16: {
                    uint16_t* data = reinterpret_cast<uint16_t*>(targetData);
                    for (size_t i = 0; i < count; ++i) {
                        float floatVal = static_cast<float>(source[i]);
                        // Convert float to DLP_BF16 by truncating mantissa
                        // Use union to avoid strict-aliasing issues
                        union
                        {
                            float    f;
                            uint32_t u;
                        } float_bits;
                        float_bits.f = floatVal;
                        data[startIndex + i] =
                            static_cast<uint16_t>(float_bits.u >> 16);
                    }
                    break;
                }
                case MatrixType::s32: {
                    int32_t* data = reinterpret_cast<int32_t*>(targetData);
                    for (size_t i = 0; i < count; ++i) {
                        data[startIndex + i] = static_cast<int32_t>(source[i]);
                    }
                    break;
                }
                case MatrixType::u32: {
                    uint32_t* data = reinterpret_cast<uint32_t*>(targetData);
                    for (size_t i = 0; i < count; ++i) {
                        data[startIndex + i] = static_cast<uint32_t>(source[i]);
                    }
                    break;
                }
                case MatrixType::s16: {
                    int16_t* data = reinterpret_cast<int16_t*>(targetData);
                    for (size_t i = 0; i < count; ++i) {
                        data[startIndex + i] = static_cast<int16_t>(source[i]);
                    }
                    break;
                }
                case MatrixType::u16: {
                    uint16_t* data = reinterpret_cast<uint16_t*>(targetData);
                    for (size_t i = 0; i < count; ++i) {
                        data[startIndex + i] = static_cast<uint16_t>(source[i]);
                    }
                    break;
                }
                case MatrixType::s8: {
                    int8_t* data = reinterpret_cast<int8_t*>(targetData);
                    for (size_t i = 0; i < count; ++i) {
                        data[startIndex + i] = static_cast<int8_t>(source[i]);
                    }
                    break;
                }
                case MatrixType::u8: {
                    uint8_t* data = reinterpret_cast<uint8_t*>(targetData);
                    for (size_t i = 0; i < count; ++i) {
                        data[startIndex + i] = static_cast<uint8_t>(source[i]);
                    }
                    break;
                }
                case MatrixType::s4:
                case MatrixType::u4: {
                    // Handle packed 4-bit types
                    uint8_t* data = reinterpret_cast<uint8_t*>(targetData);
                    for (size_t i = 0; i < count; ++i) {
                        size_t  packedIndex = (startIndex + i) / 2;
                        size_t  bitOffset   = ((startIndex + i) % 2) * 4;
                        uint8_t value4bit =
                            static_cast<uint8_t>(source[i]) & 0x0F;

                        if (bitOffset == 0) {
                            // Lower 4 bits
                            data[packedIndex] =
                                (data[packedIndex] & 0xF0) | value4bit;
                        } else {
                            // Upper 4 bits
                            data[packedIndex] =
                                (data[packedIndex] & 0x0F) | (value4bit << 4);
                        }
                    }
                    break;
                }
                default:
                    throw std::runtime_error(
                        "Unsupported matrix type for writeDataToMatrix");
            }
        }

      private:
        /**
         * @brief Helper function for floating point data comparison
         *
         * @param other The matrix to compare with
         * @return bool True if floating point data matches within tolerance
         */
        bool compareFloatingPointData(const Matrix& other) const;

        /**
         * @brief Allocate aligned memory with proper size rounding
         *
         * @param sizeBytes Requested size in bytes
         * @param alignment Required alignment (0 for no special alignment)
         * @return Pointer to allocated memory
         * @throws std::bad_alloc if allocation fails
         * @throws std::invalid_argument if alignment is invalid
         */
        uint8_t* allocateAlignedMemory(size_t sizeBytes, size_t alignment);

        /**
         * @brief Deallocate aligned memory
         *
         * @param ptr Pointer to memory to deallocate
         * @param alignment Alignment used during allocation (0 for regular
         * allocation)
         */
        void deallocateAlignedMemory(uint8_t* ptr, size_t alignment);

        md_t m_rows; ///< Number of rows in the matrix
        md_t m_cols; ///< Number of columns in the matrix
        md_t m_k = std::numeric_limits<md_t>::max(); ///< K Dim for tolerance
                                                     ///< calculation
        MatrixType m_type;                           ///< Matrix data type
        uint8_t*
               m_data; ///< Matrix data storage (raw pointer for aligned alloc)
        size_t m_dataSizeBytes; ///< Size of allocated memory in bytes
        size_t m_alignment;     ///< Memory alignment in bytes (0 for no special
                                ///< alignment)
        MatrixLayout m_layout;  ///< Memory layout
        md_t         m_leadingDim; ///< Leading dimension (stride)
        bool m_transposed; ///< Whether the matrix is logically transposed
        bool m_reordered;  ///< Whether the matrix is reordered
        bool m_packed;     ///< Whether the matrix is packed
    };

    // Template specializations for default type mapping (must be outside class)
    template<>
    struct Matrix::DefaultMatrixType<float>
    {
        static constexpr MatrixType value = MatrixType::f32;
    };

    template<>
    struct Matrix::DefaultMatrixType<double>
    {
        static constexpr MatrixType value = MatrixType::f32;
    };

    template<>
    struct Matrix::DefaultMatrixType<int8_t>
    {
        static constexpr MatrixType value = MatrixType::s8;
    };

    template<>
    struct Matrix::DefaultMatrixType<uint8_t>
    {
        static constexpr MatrixType value = MatrixType::u8;
    };

    template<>
    struct Matrix::DefaultMatrixType<int16_t>
    {
        static constexpr MatrixType value = MatrixType::s16;
    };

    template<>
    struct Matrix::DefaultMatrixType<uint16_t>
    {
        static constexpr MatrixType value = MatrixType::u16;
    };

    template<>
    struct Matrix::DefaultMatrixType<int32_t>
    {
        static constexpr MatrixType value = MatrixType::s32;
    };

    template<>
    struct Matrix::DefaultMatrixType<uint32_t>
    {
        static constexpr MatrixType value = MatrixType::u32;
    };

}}} // namespace dlp::testing::framework
