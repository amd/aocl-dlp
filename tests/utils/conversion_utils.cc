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

#include "utils/conversion_utils.hh"
#include <cstdint>

namespace dlp { namespace testing { namespace utils {

    float bf16_to_f32(bfloat16 bf16_val)
    {
        union
        {
            float    f;
            uint32_t u;
        } float_bits;
        float_bits.u = static_cast<uint32_t>(static_cast<uint16_t>(bf16_val))
                       << 16U;
        return float_bits.f;
    }

    bfloat16 f32_to_bf16(float f32_val)
    {
        union
        {
            float    f;
            uint32_t u;
        } bits;
        bits.f = f32_val;
        // Extract LSB of BF16 part for ties-to-even
        uint32_t lsb = (bits.u >> 16U) & 1U;
        uint32_t rounding_bias =
            0x7FFFU + lsb; // 0x7FFF for round, +lsb for ties-to-even
        uint32_t rounded    = bits.u + rounding_bias;
        uint16_t bf16_upper = static_cast<uint16_t>(rounded >> 16U);
        return static_cast<bfloat16>(bf16_upper);
    }

    /**
     * @brief Convert float32 to bfloat16 using VCVTNEPS2BF16 instruction
     * behavior
     *
     * This function mimics the exact behavior of the Intel VCVTNEPS2BF16
     * instruction which converts packed single-precision floating-point values
     * to packed BF16 values using specific rounding and special case handling.
     *
     * Algorithm overview:
     * 1. Handle special cases: zero/denormal, infinity, NaN
     * 2. For normal numbers: apply round-to-nearest-even with bias
     * 3. Truncate to upper 16 bits to get BF16 result
     *
     * The instruction pseudocode reference:
     * FOR j := 0 to elements-1
     *     IF input[j] is zero or denormal THEN
     *         result[j] := sign-preserving zero
     *     ELSE IF input[j] is infinity THEN
     *         result[j] := input[j][31:16] (preserve sign and infinity)
     *     ELSE IF input[j] is NaN THEN
     *         result[j] := input[j][31:16] with MSB of mantissa set (force
     * QNAN) ELSE // normal number result[j] :=
     * round_to_nearest_even(input[j])[31:16] ENDIF ENDFOR
     */
    bfloat16 f32_to_bf16_vcvtneps2bf16(float f32_val)
    {
        union
        {
            float    f;
            uint32_t u;
        } x;
        x.f = f32_val;

        // Extract components
        uint32_t sign     = x.u & 0x80000000U;
        uint32_t exp      = x.u & 0x7F800000U;
        uint32_t mantissa = x.u & 0x007FFFFFU;

        uint16_t dest;

        // Check if x is zero or denormal
        if (exp == 0) {
            // Sign preserving zero (denormal go to zero)
            dest = static_cast<uint16_t>(sign >> 16U); // dest[15] := x[31]
            // dest[14:0] := 0 (already handled by the sign shift)
        }
        // Check if x is infinity
        else if (exp == 0x7F800000U && mantissa == 0) {
            dest = static_cast<uint16_t>(x.u >> 16U); // dest[15:0] := x[31:16]
        }
        // Check if x is NaN
        else if (exp == 0x7F800000U && mantissa != 0) {
            dest = static_cast<uint16_t>(
                x.u >> 16U); // dest[15:0] := x[31:16] (truncate)
            dest |= 0x0040U; // dest[6] := 1 (set MSB of mantissa to force QNAN)
        }
        // Normal number
        else {
            uint32_t lsb = (x.u >> 16U) & 1U; // LSB := x[16]
            uint32_t rounding_bias =
                0x00007FFFU + lsb; // rounding_bias := 0x00007FFF + LSB
            uint32_t temp = x.u + rounding_bias; // temp[31:0] := x[31:0] +
                                                 // rounding_bias (integer add)
            dest =
                static_cast<uint16_t>(temp >> 16U); // dest[15:0] := temp[31:16]
        }

        return static_cast<bfloat16>(dest);
    }

}}} // namespace dlp::testing::utils
