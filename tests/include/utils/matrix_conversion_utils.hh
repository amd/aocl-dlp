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

#ifndef DLP_TESTING_UTILS_MATRIX_CONVERSION_UTILS_HH
#define DLP_TESTING_UTILS_MATRIX_CONVERSION_UTILS_HH

#include "classic/dlp_base_types.h"
#include "framework/matrix.hh"

namespace dlp { namespace testing { namespace utils {

    using dlp::testing::framework::Matrix;
    using dlp::testing::framework::MatrixLayout;
    using dlp::testing::framework::MatrixType;

    /**
     * @brief Convert a single value from any MatrixType to intermediate type
     * @tparam IntermediateT Target type (float, int32_t, etc.)
     * @param src_ptr Pointer to source data
     * @param src_type Source data type
     * @param index Element index
     * @return IntermediateT Converted value
     */
    template<typename IntermediateT>
    IntermediateT convertTo(const void* src_ptr,
                            MatrixType  src_type,
                            size_t      index);

    /**
     * @brief Convert an intermediate type value to any MatrixType and store it
     * @tparam IntermediateT Source type (float, int32_t, etc.)
     * @param dst_ptr Pointer to destination data
     * @param dst_type Destination data type
     * @param index Element index
     * @param value Intermediate value to convert
     */
    template<typename IntermediateT>
    void convertFrom(void*         dst_ptr,
                     MatrixType    dst_type,
                     size_t        index,
                     IntermediateT value);

    /**
     * @brief Copy matrix data with type conversion to intermediate type
     * @tparam IntermediateT Target type (float, int32_t, etc.)
     * @param src Source matrix
     * @param dst_data Destination array
     * @param dst_ld Destination leading dimension
     * @param dst_layout Destination layout (default: ROW_MAJOR)
     * @return bool Success status
     */
    template<typename IntermediateT>
    bool copyMatrixTo(const Matrix&  src,
                      IntermediateT* dst_data,
                      md_t           dst_ld,
                      MatrixLayout   dst_layout = MatrixLayout::ROW_MAJOR);

    /**
     * @brief Copy intermediate type data back to matrix with type conversion
     * @tparam IntermediateT Source type (float, int32_t, etc.)
     * @param src_data Source array
     * @param src_ld Source leading dimension
     * @param dst Destination matrix
     * @param src_layout Source layout (default: ROW_MAJOR)
     * @return bool Success status
     */
    template<typename IntermediateT>
    bool copyToMatrix(const IntermediateT* src_data,
                      md_t                 src_ld,
                      Matrix&              dst,
                      MatrixLayout src_layout = MatrixLayout::ROW_MAJOR);

}}} // namespace dlp::testing::utils

#endif // DLP_TESTING_UTILS_MATRIX_CONVERSION_UTILS_HH
