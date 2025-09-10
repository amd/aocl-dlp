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
 */

#ifndef DLP_TESTING_UTILS_CONVERSION_UTILS_HH
#define DLP_TESTING_UTILS_CONVERSION_UTILS_HH

#include "classic/aocl_bf16_type.h"

namespace dlp { namespace testing { namespace utils {

    /**
     * @brief Convert bfloat16 to float32
     *
     * Converts a bfloat16 value to a float32 value by placing the 16-bit
     * bfloat16 value in the upper 16 bits of a 32-bit float.
     *
     * @param bf16_val The bfloat16 value to convert
     * @return float The corresponding float32 value
     */
    float bf16_to_f32(bfloat16 bf16_val);

    /**
     * @brief Convert float32 to bfloat16 with round-to-nearest-even
     *
     * Converts a float32 value to bfloat16 using round-to-nearest-even
     * (banker's) rounding for better accuracy. This implementation handles ties
     * by rounding to the nearest even value.
     *
     * @param f32_val The float32 value to convert
     * @return bfloat16 The corresponding bfloat16 value
     */
    bfloat16 f32_to_bf16(float f32_val);

    /**
     * @brief Convert float32 to bfloat16 using VCVTNEPS2BF16 algorithm
     *
     * This function implements the conversion from float32 to bfloat16
     * following the VCVTNEPS2BF16 instruction semantics, which includes
     * specific handling for zero, denormals, infinities, NaNs, and normal
     * numbers.
     * @param f32_val The float32 value to convert
     * @return bfloat16 The corresponding bfloat16 value
     */
    bfloat16 f32_to_bf16_vcvtneps2bf16(float f32_val);

}}} // namespace dlp::testing::utils

#endif // DLP_TESTING_UTILS_CONVERSION_UTILS_HH
