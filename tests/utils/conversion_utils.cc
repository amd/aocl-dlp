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

    /**
     * @brief Convert float16 to float32 (lossless conversion)
     *
     * Converts an FP16 value to FP32 by expanding the components:
     * - Sign: copied directly
     * - Exponent: adjusted from bias 15 to bias 127
     * - Mantissa: extended from 10 to 23 bits by appending zeros
     *
     * Special cases:
     * - Zero: preserved with sign
     * - Subnormal: normalized to FP32 normal
     * - Infinity: preserved
     * - NaN: preserved with mantissa bits
     */
    float fp16_to_f32(float16 fp16_val)
    {
        uint16_t h = fp16_val;

        // Extract components
        uint32_t sign   = ((uint32_t)h & 0x8000U) << 16; // Bit 15 → 31
        uint32_t exp16  = (h & 0x7C00U) >> 10;           // Bits 14-10
        uint32_t mant16 = h & 0x03FFU;                   // Bits 9-0

        uint32_t fp32_bits;

        if (exp16 == 0) {
            // Zero or subnormal
            if (mant16 == 0) {
                // Zero (preserve sign)
                fp32_bits = sign;
            } else {
                // Subnormal: normalize it
                // Find the position of the leading 1 in the 10-bit mantissa
                int lz = __builtin_clz(mant16) - 21; // 32 - 10 - 1 = 21

                // Normalize: shift left until leading 1 reaches past bit 9
                uint32_t normalized = mant16 << lz;

                // After shifting, the leading 1 is at bit (9 + lz)
                // Remove it by shifting right and taking lower 9 bits
                uint32_t mant_normalized = (normalized >> 1) & 0x1FF;

                // Shift to FP32 mantissa position (top 9 bits of 23)
                uint32_t mant32 = mant_normalized << 14;

                // Compute FP32 exponent
                uint32_t exp32 = 127 - 14 - lz;

                fp32_bits = sign | (exp32 << 23) | mant32;
            }
        } else if (exp16 == 0x1F) {
            // Infinity or NaN
            fp32_bits = sign | 0x7F800000U | (mant16 << 13);
        } else {
            // Normal number
            // Adjust exponent bias: FP16 bias=15, FP32 bias=127
            uint32_t exp32  = exp16 + 112;  // exp16 - 15 + 127 = exp16 + 112
            uint32_t mant32 = mant16 << 13; // Extend mantissa: 10 → 23 bits

            fp32_bits = sign | (exp32 << 23) | mant32;
        }

        // Type-pun bits to float
        union
        {
            float    f;
            uint32_t u;
        } result;
        result.u = fp32_bits;
        return result.f;
    }

    /**
     * @brief Convert float32 to float16 with round-to-nearest-even (lossy)
     *
     * Converts an FP32 value to FP16 using round-to-nearest-even rounding.
     * This conversion is lossy and may overflow or underflow.
     *
     * Algorithm:
     * 1. Extract sign, exponent, mantissa from FP32
     * 2. Adjust exponent bias (127 → 15)
     * 3. Check for overflow → Infinity
     * 4. Check for underflow → Zero or subnormal
     * 5. Round mantissa from 23 to 10 bits
     * 6. Handle rounding-induced carry
     */
    float16 f32_to_fp16(float f32_val)
    {
        union
        {
            float    f;
            uint32_t u;
        } x;
        x.f = f32_val;

        // Extract components
        uint32_t sign   = (x.u & 0x80000000U) >> 16;   // Bit 31 → 15
        int32_t  exp32  = ((x.u & 0x7F800000U) >> 23); // Extract exponent
        uint32_t mant32 = (x.u & 0x007FFFFFU);         // Extract mantissa

        // Special case: FP32 zero or subnormal
        if (exp32 == 0) {
            return (float16)(sign); // Return signed zero
        }

        // Special case: FP32 infinity or NaN
        if (exp32 == 0xFF) {
            if (mant32 == 0) {
                // Infinity
                return (float16)(sign | 0x7C00U);
            } else {
                // NaN: preserve some mantissa bits
                uint16_t mant16 =
                    (uint16_t)((mant32 >> 13) | 0x0200U); // Ensure NaN
                return (float16)(sign | 0x7C00U | mant16);
            }
        }

        // Rebias exponent: FP32 bias=127, FP16 bias=15
        int32_t exp16 = exp32 - 112; // exp32 - 127 + 15 = exp32 - 112

        // Add implicit leading 1 to mantissa
        mant32 |= 0x00800000U;

        // Check for underflow (handle denormals)
        if (exp16 <= 0) {
            if (exp16 < -10) {
                // Too small, flush to zero
                return (float16)(sign);
            }

            // Denormalize: compute total shift to get 10-bit mantissa directly
            // For exp16=0: total_shift=14 (normal would be >>13, +1 for denorm)
            // For exp16=-1: total_shift=15, etc.
            int total_shift = 14 - exp16;

            // Round to nearest even:
            // - round_bit is at position (total_shift - 1)
            // - sticky bits are positions 0 to (total_shift - 2)
            // - lsb is at position total_shift
            uint32_t round_bit = (mant32 >> (total_shift - 1)) & 1;
            uint32_t sticky =
                (total_shift > 1)
                && ((mant32 & ((1U << (total_shift - 1)) - 1)) != 0);
            uint32_t lsb = (mant32 >> total_shift) & 1;

            uint32_t mant16 = mant32 >> total_shift;

            if (round_bit && (sticky || lsb)) {
                mant16++;
            }

            // Check if rounding caused normalization
            if (mant16 >= 0x0400U) {
                return (float16)(sign | 0x0400U); // Smallest normal
            }

            return (float16)(sign | (uint16_t)mant16);
        }

        // Normal number: Check if exponent is already at/above infinity
        if (exp16 >= 0x1F) {
            // Exponent already at/above infinity threshold
            return (float16)(sign | 0x7C00U);
        }

        // Normal value: Round mantissa from 23 to 10 bits
        uint32_t round_bits = mant32 & 0x1FFFU; // Bits 12-0
        uint32_t lsb        = (mant32 >> 13) & 1;

        // Round to nearest even
        // NOTE: The overflow check was previously done here prematurely.
        // The correct approach is to perform rounding first, then check
        // if the mantissa overflowed, which naturally handles all cases.
        if (round_bits > 0x1000U || (round_bits == 0x1000U && lsb)) {
            mant32 += 0x1000U;
        }

        // Check for carry into exponent AFTER rounding
        if (mant32 & 0x01000000U) {
            // Mantissa overflowed into bit 24
            exp16++;
            mant32 = 0x00800000U; // Reset to implicit 1 only

            // Check if exponent overflowed to infinity
            if (exp16 >= 0x1F) {
                return (float16)(sign | 0x7C00U);
            }
        }

        // Extract rounded 10-bit mantissa (remove implicit 1)
        uint16_t mant16 = (uint16_t)((mant32 >> 13) & 0x03FFU);

        return (float16)(sign | ((uint16_t)exp16 << 10) | mant16);
    }

    /**
     * @brief Convert float32 to float16 using VCVTPS2PH algorithm
     *
     * This implements the F16C extension VCVTPS2PH instruction behavior
     * with round-to-nearest-even mode (imm8 = 0).
     *
     * Handles all special cases according to IEEE 754 standard:
     * - Zero/subnormal → signed zero
     * - Infinity → infinity (preserve sign)
     * - NaN → NaN (set MSB of mantissa for quiet NaN)
     * - Normal → rounded value
     */
    float16 f32_to_fp16_vcvtps2ph(float f32_val)
    {
        union
        {
            float    f;
            uint32_t u;
        } x;
        x.f = f32_val;

        // Extract components
        uint32_t sign = x.u & 0x80000000U;
        uint32_t exp  = x.u & 0x7F800000U;
        uint32_t mant = x.u & 0x007FFFFFU;

        uint16_t result;

        // Check if zero or subnormal
        if (exp == 0) {
            // Sign-preserving zero
            result = (uint16_t)(sign >> 16);
        }
        // Check if infinity
        else if (exp == 0x7F800000U && mant == 0) {
            result = (uint16_t)(x.u >> 16); // Preserve sign and exponent
        }
        // Check if NaN
        else if (exp == 0x7F800000U && mant != 0) {
            result = (uint16_t)(x.u >> 16); // Truncate
            result |= 0x0200U; // Set MSB of mantissa to force quiet NaN
        }
        // Normal number
        else {
            // Use the standard round-to-nearest-even conversion
            return f32_to_fp16(f32_val);
        }

        return (float16)result;
    }

}}} // namespace dlp::testing::utils
