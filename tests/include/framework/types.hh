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

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <ostream>

/**
 * @namespace dlp
 * @brief Deep Learning Primitives namespace
 */
namespace dlp {

/**
 * @namespace dlp::testing
 * @brief Testing framework for Deep Learning Primitives
 */
namespace testing {

    /**
     * @namespace dlp::testing::framework
     * @brief Core framework components for testing
     */
    namespace framework {

        /**
         * @enum MatrixType
         * @brief Enumeration of supported matrix data types
         */
        enum class MatrixType : uint16_t
        {
            u4 = 0, ///< Unsigned 4-bit integer
            u8,     ///< Unsigned 8-bit integer
            u16,    ///< Unsigned 16-bit integer
            u32,    ///< Unsigned 32-bit integer
            s4,     ///< Signed 4-bit integer
            s8,     ///< Signed 8-bit integer
            s16,    ///< Signed 16-bit integer
            s32,    ///< Signed 32-bit integer
            f32,    ///< 32-bit floating point
            bf16,   ///< Brain floating point 16-bit
            fp16,   ///< IEEE 754 half-precision 16-bit
        };

        /**
         * @brief Stream output operator for MatrixType
         * @param os Output stream
         * @param type MatrixType to output
         * @return Reference to the output stream
         */
        inline std::ostream& operator<<(std::ostream& os, MatrixType type)
        {
            switch (type) {
                case MatrixType::u4:
                    return os << "u4";
                case MatrixType::u8:
                    return os << "u8";
                case MatrixType::u16:
                    return os << "u16";
                case MatrixType::u32:
                    return os << "u32";
                case MatrixType::s4:
                    return os << "s4";
                case MatrixType::s8:
                    return os << "s8";
                case MatrixType::s16:
                    return os << "s16";
                case MatrixType::s32:
                    return os << "s32";
                case MatrixType::f32:
                    return os << "f32";
                case MatrixType::bf16:
                    return os << "bf16";
                case MatrixType::fp16:
                    return os << "fp16";
                default:
                    return os << "unknown";
            }
        }

        /**
         * @enum MatrixLayout
         * @brief Enumeration of supported matrix memory layouts
         */
        enum class MatrixLayout
        {
            ROW_MAJOR,   ///< Row-major layout (C/C++ style)
            COLUMN_MAJOR ///< Column-major layout (Fortran style)
        };

        /**
         * @brief Stream output operator for MatrixLayout
         * @param os Output stream
         * @param layout MatrixLayout to output
         * @return Reference to the output stream
         */
        inline std::ostream& operator<<(std::ostream& os, MatrixLayout layout)
        {
            switch (layout) {
                case MatrixLayout::ROW_MAJOR:
                    return os << "row-major";
                case MatrixLayout::COLUMN_MAJOR:
                    return os << "column-major";
                default:
                    return os << "unknown";
            }
        }

        /**
         * @enum VerbosityLevel
         * @brief Enumeration of verbosity levels for diagnostic output
         */
        enum class VerbosityLevel : uint8_t
        {
            SILENT = 0, ///< No verbosity (default)
            BASIC  = 1, ///< Basic verbosity (-v): Verbose comparison results
            PARTIAL_MATRIX = 2, ///< Print partial matrices (-vv): 5x5 elements
            FULL_MATRIX =
                3 ///< Print full matrices (-vvv): up to 50x50 elements
        };

        /**
         * @brief Stream output operator for VerbosityLevel
         * @param os Output stream
         * @param level VerbosityLevel to output
         * @return Reference to the output stream
         */
        inline std::ostream& operator<<(std::ostream& os, VerbosityLevel level)
        {
            switch (level) {
                case VerbosityLevel::SILENT:
                    return os << "SILENT";
                case VerbosityLevel::BASIC:
                    return os << "BASIC";
                case VerbosityLevel::PARTIAL_MATRIX:
                    return os << "PARTIAL_MATRIX";
                case VerbosityLevel::FULL_MATRIX:
                    return os << "FULL_MATRIX";
                default:
                    return os << "UNKNOWN";
            }
        }

        /**
         * @brief Matrix memory allocation utilities
         */
        namespace MatrixMemory {

            /**
             * @brief Calculate the size in bytes for a single element of the
             * given type
             * @param type The matrix data type
             * @return Size in bytes (0 for packed types like u4/s4)
             */
            inline size_t getElementSizeBytes(MatrixType type)
            {
                switch (type) {
                    case MatrixType::u4:
                    case MatrixType::s4:
                        return 0; // Special case: 2 elements per byte
                    case MatrixType::u8:
                    case MatrixType::s8:
                        return 1;
                    case MatrixType::u16:
                    case MatrixType::s16:
                    case MatrixType::bf16:
                    case MatrixType::fp16:
                        return 2;
                    case MatrixType::u32:
                    case MatrixType::s32:
                    case MatrixType::f32:
                        return 4;
                    default:
                        return 0;
                }
            }

            /**
             * @brief Calculate required bytes for matrix storage
             * @param type Matrix data type
             * @param rows Number of rows
             * @param cols Number of columns
             * @param layout Memory layout
             * @param leadingDim Leading dimension (0 for automatic)
             * @return Required bytes for allocation
             */
            inline size_t calculateRequiredBytes(MatrixType   type,
                                                 size_t       rows,
                                                 size_t       cols,
                                                 MatrixLayout layout,
                                                 size_t       leadingDim = 0)
            {
                // Calculate leading dimension if not specified
                if (leadingDim == 0) {
                    leadingDim = (layout == MatrixLayout::ROW_MAJOR) ? cols
                                                                     : rows;
                }

                // Calculate total elements based on layout
                size_t totalElements;
                if (layout == MatrixLayout::ROW_MAJOR) {
                    totalElements = rows * leadingDim;
                } else {
                    totalElements = cols * leadingDim;
                }

                // Handle packed types (u4/s4)
                if (type == MatrixType::u4 || type == MatrixType::s4) {
                    return (totalElements + 1) / 2; // 2 elements per byte
                }

                // Regular types
                return totalElements * getElementSizeBytes(type);
            }

            /**
             * @brief Allocate memory for matrix data
             * @param type Matrix data type
             * @param rows Number of rows
             * @param cols Number of columns
             * @param layout Memory layout
             * @param leadingDim Leading dimension (0 for automatic)
             * @return Unique pointer to allocated memory
             */
            inline std::unique_ptr<uint8_t[]> allocate(MatrixType   type,
                                                       size_t       rows,
                                                       size_t       cols,
                                                       MatrixLayout layout,
                                                       size_t leadingDim = 0)
            {
                size_t bytes = calculateRequiredBytes(type, rows, cols, layout,
                                                      leadingDim);
                return std::make_unique<uint8_t[]>(bytes);
            }

            /**
             * @brief Allocate memory with custom size in bytes
             * @param sizeBytes Size in bytes to allocate
             * @return Unique pointer to allocated memory
             */
            inline uint8_t* allocateBytes(size_t sizeBytes,
                                          size_t alignment = 0)
            {
                uint8_t* data = nullptr;

                // Allocate memory based on alignment requirements
                if (alignment > 0) {
                    // Validate alignment requirements for std::aligned_alloc
                    if ((alignment & (alignment - 1)) != 0) {
                        throw std::invalid_argument(
                            "Alignment must be a power of 2");
                    }
                    if (alignment < sizeof(void*)) {
                        throw std::invalid_argument(
                            "Alignment must be at least sizeof(void*)");
                    }

                    // Ensure size is a multiple of alignment for
                    // std::aligned_alloc
                    size_t alignedSize =
                        (sizeBytes + alignment - 1) & ~(alignment - 1);

                    data = static_cast<uint8_t*>(
                        std::aligned_alloc(alignment, alignedSize));
                    if (!data) {
                        throw std::bad_alloc();
                    }

                    // Update size to reflect actual allocated size
                    sizeBytes = alignedSize;
                } else {
                    // Use regular allocation
                    data = new uint8_t[sizeBytes];
                }
                return data;
            }
        } // namespace MatrixMemory

        /**
         * @brief Legacy MatrixData structure for backward compatibility
         *
         * This structure is maintained for compatibility with existing code
         * that expects the old interface. It wraps the new memory model.
         */
        struct MatrixData
        {
            MatrixType type;     ///< The type of data stored in the matrix
            void*      data_ptr; ///< Pointer to the matrix data

            /**
             * @brief Constructor
             * @param matrixType The type of data
             * @param dataPtr Pointer to the data
             */
            MatrixData(MatrixType matrixType = MatrixType::f32,
                       void*      dataPtr    = nullptr)
                : type(matrixType)
                , data_ptr(dataPtr)
            {
            }

            /**
             * @brief Get raw pointer to the matrix data
             * @return void* Pointer to the matrix data (type-erased)
             */
            void* getMatrixPtr() const { return data_ptr; }
        };

    } // namespace framework

} // namespace testing

} // namespace dlp
