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
#include <sstream>
#include <string>
#include <vector>

namespace dlp { namespace testing { namespace framework {

    // Forward declaration
    class Matrix;

    /**
     * @struct MismatchInfo
     * @brief Information about a single element mismatch between matrices
     */
    struct MismatchInfo
    {
        size_t row;          ///< Row index of the mismatch
        size_t col;          ///< Column index of the mismatch
        double value1;       ///< Value from first matrix (converted to double)
        double value2;       ///< Value from second matrix (converted to double)
        double absDiff;      ///< Absolute difference between values
        double relativeDiff; ///< Relative difference (absDiff / max(|val1|,
                             ///< |val2|))
    };

    /**
     * @struct MatrixCompareOptions
     * @brief Options controlling matrix comparison behavior
     *
     * This structure controls how Matrix::compare() operates:
     * - Fast mode (verbose=false): Boolean comparison with early exit,
     *   minimal allocations, optimized for performance
     * - Verbose mode (verbose=true): Collects detailed diagnostics including
     *   mismatch locations, differences, and statistics
     */
    struct MatrixCompareOptions
    {
        bool verbose = false; ///< Enable verbose mode with detailed diagnostics
        size_t maxMismatches = 10; ///< Maximum number of mismatches to record
                                   ///< in verbose mode
        double absToleranceOverride =
            -1.0; ///< Override default absolute tolerance (-1 = use default)
        double relToleranceOverride =
            -1.0; ///< Override default relative tolerance (-1 = use default)
        double relToleranceMultiplier =
            -1.0; ///< Relative tolerance multiplier (-1 = use default 50.0)
        double absToleranceMultiplier =
            -1.0; ///< Absolute tolerance multiplier (-1 = use default 50.0)

        /**
         * @brief Create options for fast mode (boolean comparison only)
         */
        static MatrixCompareOptions Fast()
        {
            MatrixCompareOptions opts;
            opts.verbose = false;
            return opts;
        }

        /**
         * @brief Create options for verbose mode with detailed diagnostics
         * @param maxMismatches Maximum number of mismatches to record (default:
         * 10)
         */
        static MatrixCompareOptions Verbose(size_t maxMismatches = 10)
        {
            MatrixCompareOptions opts;
            opts.verbose       = true;
            opts.maxMismatches = maxMismatches;
            return opts;
        }
    };

    /**
     * @struct MatrixCompareResult
     * @brief Result of matrix comparison with optional detailed diagnostics
     *
     * Contains:
     * - Boolean equality result
     * - Mismatch type (dimension, type, or data)
     * - Detailed mismatch information (in verbose mode)
     * - Statistics (max differences, counts)
     */
    struct MatrixCompareResult
    {
        bool equal = true; ///< True if matrices are equal within tolerance

        // Mismatch categorization
        bool dimensionMismatch = false; ///< True if dimensions differ
        bool typeMismatch      = false; ///< True if types differ
        bool layoutMismatch    = false; ///< True if layouts differ
        bool metadataMismatch  = false; ///< True if other metadata differs

        // Statistics
        size_t mismatchCount = 0;   ///< Total number of element mismatches
        double maxAbsDiff    = 0.0; ///< Maximum absolute difference found
        double maxRelDiff    = 0.0; ///< Maximum relative difference found
        size_t maxDiffRow    = 0;   ///< Row of maximum difference
        size_t maxDiffCol    = 0;   ///< Column of maximum difference

        // Tolerance information
        double usedAbsTolerance =
            0.0; ///< Absolute tolerance used for comparison
        double usedRelTolerance =
            0.0; ///< Relative tolerance used for comparison

        // Detailed mismatch list (only populated in verbose mode)
        std::vector<MismatchInfo> mismatches;

        /**
         * @brief Check if comparison was successful (matrices are equal)
         */
        operator bool() const { return equal; }
    };

    /**
     * @brief Format comparison result as a human-readable string
     *
     * Produces detailed diagnostic output suitable for logging or test failure
     * messages. Includes dimension/type information, tolerance values, mismatch
     * details, and statistics.
     *
     * @param result The comparison result to format
     * @param matrix1 First matrix (for context)
     * @param matrix2 Second matrix (for context)
     * @return Formatted string representation
     */
    std::string FormatCompareResult(const MatrixCompareResult& result,
                                    const Matrix&              matrix1,
                                    const Matrix&              matrix2);

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
         * This is marked const because k is metadata used for tolerance
         * calculations, not part of the matrix's mathematical value.
         * Setting k doesn't change the observable state of the matrix.
         *
         * @param k The k dimension for tolerance calculation
         */
        void setK(md_t k) const;

        /**
         * @brief Compare two matrices with configurable mode and diagnostics
         *
         * Unified comparison method supporting both fast mode (boolean only)
         * and verbose mode (detailed diagnostics). This is the primary
         * comparison interface.
         *
         * Fast mode (opts.verbose=false):
         * - Optimized for performance with early exit
         * - No allocations or detailed tracking
         * - Uses memcmp for integer types
         * - Returns boolean result only
         *
         * Verbose mode (opts.verbose=true):
         * - Collects detailed mismatch information
         * - Records element locations, values, and differences
         * - Computes statistics (max diff, mismatch count)
         * - Suitable for debugging and test diagnostics
         *
         * @param other The matrix to compare with
         * @param opts Comparison options (fast vs verbose, tolerances)
         * @return MatrixCompareResult containing equality status and optional
         * diagnostics
         *
         * @example Fast mode for boolean check:
         * @code
         *   if (matrix1.compare(matrix2, MatrixCompareOptions::Fast())) {
         *     // Matrices are equal
         *   }
         * @endcode
         *
         * @example Verbose mode for diagnostics:
         * @code
         *   auto result = matrix1.compare(matrix2,
         * MatrixCompareOptions::Verbose(20)); if (!result) {
         *     std::cout << FormatCompareResult(result, matrix1, matrix2);
         *   }
         * @endcode
         */
        MatrixCompareResult compare(const Matrix&               other,
                                    const MatrixCompareOptions& opts =
                                        MatrixCompareOptions::Fast()) const;

        /**
         * @brief Compare two matrices for equality
         *
         * Checks if two matrices have the same dimensions, type, and content.
         * This is a convenience wrapper around compare() in fast mode.
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
         * @brief Fill matrix with random values from a specified distribution
         * and range
         *
         * Fills the matrix with random values from a specified distribution
         * (uniform or normal) within the given lower and upper bounds.
         *
         * @param seed Seed for the random number generator (0 means use
         * time-based seed)
         * @param lb Lower bound for random values
         * @param ub Upper bound for random values
         * @param dist Distribution type: "uniform" or "normal" (default:
         * "uniform")
         * @param force_int_distribution Force integer-only values for
         * float/bf16 types (default: true)
         */
        void fillRandom(unsigned int       seed,
                        double             lb,
                        double             ub,
                        const std::string& dist                   = "uniform",
                        bool               force_int_distribution = true);

        /**
         * @brief Fill matrix with a single value
         *
         * Fills the matrix with a single value of the appropriate data type.
         *
         * @param value The value to fill the matrix with
         */
        void fillValue(double value);

        /**
         * @brief Fill matrix with a repeating pattern
         *
         * Fills the entire allocated matrix memory with values from the
         * pattern, cycling through pattern values using modulo indexing.
         * Properly converts pattern values to the matrix's data type with
         * bounds checking.
         *
         * @param pattern Vector of values to repeat (must not be empty)
         * @throws std::invalid_argument if pattern is empty
         */
        void fillPattern(const std::vector<double>& pattern);

        /**
         * @brief Convert matrix to string representation
         *
         * Converts the matrix to a formatted string with support for all matrix
         * types. The amount of output depends on the verbosity level:
         * - Level 0-1: Empty string (no matrix printing)
         * - Level 2: Print partial matrices (5x5 elements)
         * - Level 3+: Print full matrices (up to 20x20)
         *
         * @param verbosity_level Verbosity level (0=none, 2=partial, 3=full)
         * @return String containing formatted matrix representation
         */
        std::string matrixToString(int verbosity_level = 2) const;

        /**
         * @brief Core matrix printing implementation using std::ostream
         *
         * This is the unified implementation that all printing methods delegate
         * to. Works with any std::ostream (cout, ostringstream, file streams,
         * etc.) Supports all matrix types with appropriate formatting.
         *
         * @param os Output stream to write to
         * @param verbosity_level Verbosity level controlling output detail
         */
        void printToStream(std::ostream& os, int verbosity_level) const;

        /**
         * @brief Print matrix contents based on verbosity level
         *
         * Prints the matrix values in a formatted grid. Supports all matrix
         * types with appropriate formatting. The amount of output depends on
         * the verbosity level:
         * - Level 0-1: No matrix printing
         * - Level 2: Print partial matrices (5x5 elements)
         * - Level 3+: Print full matrices (up to 20x20)
         *
         * @param name Optional name/label to display above the matrix
         * @param verbosity_level Verbosity level (0=none, 1=basic, 2=partial,
         * 3=full)
         */
        void printMatrix(const std::string& name            = "",
                         int                verbosity_level = 2) const;

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
         * @brief Format matrix data based on type
         *
         * Dispatches to appropriate type-specific formatter based on m_type.
         *
         * @param os Output stream to write to
         * @param verbosity_level Verbosity level controlling output detail
         */
        void formatMatrixData(std::ostream& os, int verbosity_level) const;

        /**
         * @brief Template formatter for numeric matrix types
         *
         * Handles f32, s32, u32, s16, u16, s8, u8 with appropriate formatting.
         *
         * @tparam T Numeric type (float, int32_t, uint8_t, etc.)
         * @param os Output stream to write to
         * @param verbosity_level Verbosity level controlling output detail
         */
        template<typename T>
        void formatNumericMatrix(std::ostream& os, int verbosity_level) const;

        /**
         * @brief Specialized formatter for BF16 matrices
         *
         * Converts BF16 values to float for readable display.
         *
         * @param os Output stream to write to
         * @param verbosity_level Verbosity level controlling output detail
         */
        void formatMatrixBF16(std::ostream& os, int verbosity_level) const;

        /**
         * @brief Specialized formatter for 4-bit packed matrices
         *
         * Unpacks and displays s4/u4 values as integers.
         *
         * @param os Output stream to write to
         * @param verbosity_level Verbosity level controlling output detail
         */
        void formatMatrix4Bit(std::ostream& os, int verbosity_level) const;

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

        md_t         m_rows; ///< Number of rows in the matrix
        md_t         m_cols; ///< Number of columns in the matrix
        mutable md_t m_k =
            std::numeric_limits<md_t>::max(); ///< K Dim for tolerance
                                              ///< calculation (mutable:
                                              ///< metadata for comparison)
        MatrixType m_type;                    ///< Matrix data type
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
