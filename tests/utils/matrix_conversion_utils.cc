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

#include "utils/matrix_conversion_utils.hh"
#include "utils/conversion_utils.hh"
#include <algorithm>
#include <cmath>

namespace dlp { namespace testing { namespace utils {

    using dlp::testing::framework::MatrixLayout;

    /**
     * @brief Convert a single value from any MatrixType to intermediate type
     */
    template<typename IntermediateT>
    IntermediateT convertTo(const void* src_ptr,
                            MatrixType  src_type,
                            size_t      index)
    {
        switch (src_type) {
            case MatrixType::f32:
                return static_cast<IntermediateT>(
                    static_cast<const float*>(src_ptr)[index]);
            case MatrixType::u8:
                return static_cast<IntermediateT>(
                    static_cast<const uint8_t*>(src_ptr)[index]);
            case MatrixType::s8:
                return static_cast<IntermediateT>(
                    static_cast<const int8_t*>(src_ptr)[index]);
            case MatrixType::u16:
                return static_cast<IntermediateT>(
                    static_cast<const uint16_t*>(src_ptr)[index]);
            case MatrixType::s16:
                return static_cast<IntermediateT>(
                    static_cast<const int16_t*>(src_ptr)[index]);
            case MatrixType::u32:
                return static_cast<IntermediateT>(
                    static_cast<const uint32_t*>(src_ptr)[index]);
            case MatrixType::s32:
                return static_cast<IntermediateT>(
                    static_cast<const int32_t*>(src_ptr)[index]);
            case MatrixType::u4: {
                const uint8_t* data     = static_cast<const uint8_t*>(src_ptr);
                size_t         byte_idx = index / 2;
                size_t         bit_offset = (index % 2) * 4;
                uint8_t        value = (data[byte_idx] >> bit_offset) & 0x0F;
                return static_cast<IntermediateT>(value);
            }
            case MatrixType::s4: {
                const uint8_t* data     = static_cast<const uint8_t*>(src_ptr);
                size_t         byte_idx = index / 2;
                size_t         bit_offset = (index % 2) * 4;
                uint8_t        value = (data[byte_idx] >> bit_offset) & 0x0F;
                // Sign extend 4-bit to 8-bit
                if (value & 0x08)
                    value |= 0xF0;
                return static_cast<IntermediateT>(static_cast<int8_t>(value));
            }
            case MatrixType::bf16: {
                const bfloat16* data    = static_cast<const bfloat16*>(src_ptr);
                float           f32_val = bf16_to_f32(data[index]);
                return static_cast<IntermediateT>(f32_val);
            }
            default:
                return static_cast<IntermediateT>(0);
        }
    }

    /**
     * @brief Helper to apply rounding for float sources, no-op for integer
     * sources
     */
    template<typename T>
    inline T applyRounding(T value)
    {
        return value;
    }

    template<>
    inline float applyRounding<float>(float value)
    {
        return std::rint(value);
    }

    /**
     * @brief Convert an intermediate type value to any MatrixType and store it
     */
    template<typename IntermediateT>
    void convertFrom(void*         dst_ptr,
                     MatrixType    dst_type,
                     size_t        index,
                     IntermediateT value)
    {
        switch (dst_type) {
            case MatrixType::f32:
                static_cast<float*>(dst_ptr)[index] = static_cast<float>(value);
                break;
            case MatrixType::u8: {
                auto rounded = applyRounding(value);
                static_cast<uint8_t*>(dst_ptr)[index] =
                    static_cast<uint8_t>(std::max(
                        static_cast<IntermediateT>(0),
                        std::min(static_cast<IntermediateT>(255), rounded)));
                break;
            }
            case MatrixType::s8: {
                auto rounded = applyRounding(value);
                static_cast<int8_t*>(dst_ptr)[index] =
                    static_cast<int8_t>(std::max(
                        static_cast<IntermediateT>(-128),
                        std::min(static_cast<IntermediateT>(127), rounded)));
                break;
            }
            case MatrixType::u16: {
                auto rounded = applyRounding(value);
                static_cast<uint16_t*>(dst_ptr)[index] =
                    static_cast<uint16_t>(std::max(
                        static_cast<IntermediateT>(0),
                        std::min(static_cast<IntermediateT>(65535), rounded)));
                break;
            }
            case MatrixType::s16: {
                auto rounded = applyRounding(value);
                static_cast<int16_t*>(dst_ptr)[index] =
                    static_cast<int16_t>(std::max(
                        static_cast<IntermediateT>(-32768),
                        std::min(static_cast<IntermediateT>(32767), rounded)));
                break;
            }
            case MatrixType::u32: {
                auto rounded                           = applyRounding(value);
                static_cast<uint32_t*>(dst_ptr)[index] = static_cast<uint32_t>(
                    std::max(static_cast<IntermediateT>(0), rounded));
                break;
            }
            case MatrixType::s32: {
                auto rounded                          = applyRounding(value);
                static_cast<int32_t*>(dst_ptr)[index] = static_cast<int32_t>(
                    std::max(static_cast<IntermediateT>(-2147483648LL),
                             std::min(static_cast<IntermediateT>(2147483647),
                                      rounded)));
                break;
            }
            case MatrixType::u4: {
                auto     rounded    = applyRounding(value);
                uint8_t* data       = static_cast<uint8_t*>(dst_ptr);
                size_t   byte_idx   = index / 2;
                size_t   bit_offset = (index % 2) * 4;
                uint8_t  value4bit =
                    static_cast<uint8_t>(std::max(
                        static_cast<IntermediateT>(0),
                        std::min(static_cast<IntermediateT>(15), rounded)))
                    & 0x0F;
                if (bit_offset == 0) {
                    data[byte_idx] = (data[byte_idx] & 0xF0) | value4bit;
                } else {
                    data[byte_idx] = (data[byte_idx] & 0x0F) | (value4bit << 4);
                }
                break;
            }
            case MatrixType::s4: {
                auto     rounded    = applyRounding(value);
                uint8_t* data       = static_cast<uint8_t*>(dst_ptr);
                size_t   byte_idx   = index / 2;
                size_t   bit_offset = (index % 2) * 4;
                int8_t   value_s4   = static_cast<int8_t>(
                    std::max(static_cast<IntermediateT>(-8),
                                 std::min(static_cast<IntermediateT>(7), rounded)));
                uint8_t value4bit = static_cast<uint8_t>(value_s4) & 0x0F;
                if (bit_offset == 0) {
                    data[byte_idx] = (data[byte_idx] & 0xF0) | value4bit;
                } else {
                    data[byte_idx] = (data[byte_idx] & 0x0F) | (value4bit << 4);
                }
                break;
            }
            case MatrixType::bf16: {
                static_cast<bfloat16*>(dst_ptr)[index] =
                    f32_to_bf16(static_cast<float>(value));
                break;
            }
            default:
                break;
        }
    }

    /**
     * @brief Copy matrix data with type conversion to intermediate type
     */
    template<typename IntermediateT>
    bool copyMatrixTo(const Matrix&  src,
                      IntermediateT* dst_data,
                      md_t           dst_ld,
                      MatrixLayout   dst_layout)
    {
        const void*  src_data   = src.getData();
        MatrixType   src_type   = src.getMatrixType();
        md_t         src_rows   = src.getRows();
        md_t         src_cols   = src.getCols();
        md_t         src_ld     = src.getLeadingDimension();
        MatrixLayout src_layout = src.getLayout();
        bool         transposed = src.isTransposed();

        for (md_t i = 0; i < src_rows; ++i) {
            for (md_t j = 0; j < src_cols; ++j) {
                size_t src_idx, dst_idx;

                // Calculate source index based on layout and transposition
                if (src_layout == MatrixLayout::ROW_MAJOR) {
                    src_idx = static_cast<size_t>(i) * src_ld + j;
                } else {
                    src_idx = static_cast<size_t>(j) * src_ld + i;
                }

                // Calculate destination index based on destination layout
                if (transposed) {
                    // If source is transposed, swap i,j for destination
                    if (dst_layout == MatrixLayout::ROW_MAJOR) {
                        dst_idx = static_cast<size_t>(j) * dst_ld + i;
                    } else {
                        dst_idx = static_cast<size_t>(i) * dst_ld + j;
                    }
                } else {
                    if (dst_layout == MatrixLayout::ROW_MAJOR) {
                        dst_idx = static_cast<size_t>(i) * dst_ld + j;
                    } else {
                        dst_idx = static_cast<size_t>(j) * dst_ld + i;
                    }
                }

                dst_data[dst_idx] =
                    convertTo<IntermediateT>(src_data, src_type, src_idx);
            }
        }
        return true;
    }

    /**
     * @brief Copy intermediate type data back to matrix with type conversion
     */
    template<typename IntermediateT>
    bool copyToMatrix(const IntermediateT* src_data,
                      md_t                 src_ld,
                      Matrix&              dst,
                      MatrixLayout         src_layout)
    {
        void*        dst_data   = dst.getData();
        MatrixType   dst_type   = dst.getMatrixType();
        md_t         dst_rows   = dst.getRows();
        md_t         dst_cols   = dst.getCols();
        md_t         dst_ld     = dst.getLeadingDimension();
        MatrixLayout dst_layout = dst.getLayout();

        for (md_t i = 0; i < dst_rows; ++i) {
            for (md_t j = 0; j < dst_cols; ++j) {
                size_t src_idx, dst_idx;

                // Calculate source index based on source layout
                if (src_layout == MatrixLayout::ROW_MAJOR) {
                    src_idx = static_cast<size_t>(i) * src_ld + j;
                } else {
                    src_idx = static_cast<size_t>(j) * src_ld + i;
                }

                // Calculate destination index based on destination layout
                if (dst_layout == MatrixLayout::ROW_MAJOR) {
                    dst_idx = static_cast<size_t>(i) * dst_ld + j;
                } else {
                    dst_idx = static_cast<size_t>(j) * dst_ld + i;
                }

                convertFrom<IntermediateT>(dst_data, dst_type, dst_idx,
                                           src_data[src_idx]);
            }
        }
        return true;
    }

    // Explicit template instantiations
    template float   convertTo<float>(const void*, MatrixType, size_t);
    template int32_t convertTo<int32_t>(const void*, MatrixType, size_t);

    template void convertFrom<float>(void*, MatrixType, size_t, float);
    template void convertFrom<int32_t>(void*, MatrixType, size_t, int32_t);

    template bool copyMatrixTo<float>(const Matrix&,
                                      float*,
                                      md_t,
                                      MatrixLayout);
    template bool copyMatrixTo<int32_t>(const Matrix&,
                                        int32_t*,
                                        md_t,
                                        MatrixLayout);

    template bool copyToMatrix<float>(const float*,
                                      md_t,
                                      Matrix&,
                                      MatrixLayout);
    template bool copyToMatrix<int32_t>(const int32_t*,
                                        md_t,
                                        Matrix&,
                                        MatrixLayout);

}}} // namespace dlp::testing::utils
