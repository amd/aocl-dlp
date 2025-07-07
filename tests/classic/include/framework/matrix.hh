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

namespace dlp { namespace testing {

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
        Matrix(md_t                       rows,
               md_t                       cols,
               MatrixType                 type,
               std::unique_ptr<uint8_t[]> data,
               size_t                     dataSizeBytes,
               MatrixLayout               layout     = MatrixLayout::ROW_MAJOR,
               md_t                       leadingDim = 0,
               bool                       transposed = false,
               bool                       reordered  = false);

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
         */
        Matrix(md_t         rows,
               md_t         cols,
               MatrixType   type,
               MatrixLayout layout     = MatrixLayout::ROW_MAJOR,
               md_t         leadingDim = 0,
               bool         transposed = false,
               bool         reordered  = false);

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
         * Memory is automatically released by unique_ptr.
         */
        ~Matrix() = default;

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

      private:
        /**
         * @brief Helper function for floating point data comparison
         *
         * @param other The matrix to compare with
         * @return bool True if floating point data matches within tolerance
         */
        bool compareFloatingPointData(const Matrix& other) const;

        md_t m_rows; ///< Number of rows in the matrix
        md_t m_cols; ///< Number of columns in the matrix
        md_t m_k = std::numeric_limits<md_t>::max(); ///< K Dim for tolerance
                                                     ///< calculation
        MatrixType                 m_type;           ///< Matrix data type
        std::unique_ptr<uint8_t[]> m_data;           ///< Matrix data storage
        size_t       m_dataSizeBytes; ///< Size of allocated memory in bytes
        MatrixLayout m_layout;        ///< Memory layout
        md_t         m_leadingDim;    ///< Leading dimension (stride)
        bool m_transposed; ///< Whether the matrix is logically transposed
        bool m_reordered;  ///< Whether the matrix is reordered
    };

}} // namespace dlp::testing
