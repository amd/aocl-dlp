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
#ifndef AOCL_DLP_FP16_CONVERT_H
#define AOCL_DLP_FP16_CONVERT_H

#include <stdint.h>

#include "classic/aocl_fp16_type.h"

#if defined(__F16C__) && defined(__GNUC__)
#include <immintrin.h>
#endif

/**
 * @brief Convert float32 to float16
 *
 * Uses compiler intrinsics when available (_cvtss_sh with F16C),
 * otherwise falls back to portable software bit manipulation.
 * Uses round-to-nearest-even rounding mode per IEEE-754.
 *
 * The software fallback correctly handles:
 * - Round-to-nearest-even rounding
 * - NaN propagation (preserves quiet NaN)
 * - Subnormal denormalization with proper rounding
 * - Overflow to infinity
 * - Underflow to zero or subnormal
 * - Rounding-induced exponent increment
 */
static inline float16
f32_to_fp16(float f32_val)
{
#if defined(__F16C__) && defined(__GNUC__)
    return (float16)_cvtss_sh(f32_val, 0);
#else
    union
    {
        float    f;
        uint32_t u;
    } x;
    x.f = f32_val;

    uint32_t sign   = (x.u & 0x80000000U) >> 16;
    int32_t  exp32  = ((x.u & 0x7F800000U) >> 23);
    uint32_t mant32 = (x.u & 0x007FFFFFU);

    if (exp32 == 0) {
        return (float16)(sign);
    }

    if (exp32 == 0xFF) {
        if (mant32 == 0) {
            return (float16)(sign | 0x7C00U);
        } else {
            uint16_t mant16 = (uint16_t)((mant32 >> 13) | 0x0200U);
            return (float16)(sign | 0x7C00U | (mant16 & 0x03FFU));
        }
    }

    int32_t exp16 = exp32 - 112;

    mant32 |= 0x00800000U;

    if (exp16 <= 0) {
        if (exp16 < -10) {
            return (float16)(sign);
        }

        int total_shift = 14 - exp16;

        uint32_t round_bit   = (mant32 >> (total_shift - 1)) & 1;
        uint32_t sticky_mask = (1U << (total_shift - 1)) - 1;
        uint32_t sticky      = (mant32 & sticky_mask) != 0;
        uint32_t lsb         = (mant32 >> total_shift) & 1;

        uint32_t mant16 = mant32 >> total_shift;

        if (round_bit && (sticky || lsb)) {
            mant16++;
        }

        if (mant16 >= 0x0400U) {
            return (float16)(sign | 0x0400U);
        }

        return (float16)(sign | (uint16_t)mant16);
    }

    if (exp16 >= 0x1F) {
        return (float16)(sign | 0x7C00U);
    }

    uint32_t round_bits = mant32 & 0x1FFFU;
    uint32_t lsb        = (mant32 >> 13) & 1;

    if (round_bits > 0x1000U || (round_bits == 0x1000U && lsb)) {
        mant32 += 0x1000U;
    }

    if (mant32 & 0x01000000U) {
        exp16++;
        mant32 = 0x00800000U;

        if (exp16 >= 0x1F) {
            return (float16)(sign | 0x7C00U);
        }
    }

    uint16_t mant16 = (uint16_t)((mant32 >> 13) & 0x03FFU);

    return (float16)(sign | ((uint16_t)exp16 << 10) | mant16);
#endif
}

/**
 * @brief Convert float16 to float32
 *
 * Uses compiler intrinsics when available (_cvtsh_ss with F16C),
 * otherwise falls back to a portable software path that is bit-identical
 * to the F16C VCVTPH2PS instruction for all 65536 FP16 bit patterns.
 *
 * The software fallback correctly handles:
 * - +0 / -0
 * - +inf / -inf
 * - FP16 subnormals (renormalized to F32 normals)
 * - Quiet NaN payload preservation
 * - Signaling NaN to quiet NaN conversion (forces F32 mantissa MSB per
 *   IEEE-754 default exception handling, matching VCVTPH2PS exactly)
 */
static inline float
fp16_to_f32(float16 h)
{
#if defined(__F16C__) && defined(__GNUC__)
    return _cvtsh_ss((unsigned short)h);
#else
    uint32_t bits   = (uint32_t)(uint16_t)h;
    uint32_t sign   = (bits & 0x8000U) << 16;
    uint32_t exp16  = (bits >> 10) & 0x1FU;
    uint32_t mant16 = bits & 0x3FFU;
    uint32_t f32_bits;

    if (exp16 == 0) {
        if (mant16 == 0) {
            f32_bits = sign;
        } else {
            uint32_t shifts = 0;
            while ((mant16 & 0x400U) == 0) {
                mant16 <<= 1;
                shifts++;
            }
            mant16 &= 0x3FFU;
            uint32_t exp32 = (uint32_t)(127 - 14) - shifts;
            f32_bits       = sign | (exp32 << 23) | (mant16 << 13);
        }
    } else if (exp16 == 0x1FU) {
        uint32_t mant32 = mant16 << 13;
        if (mant16 != 0) {
            mant32 |= 0x00400000U;
        }
        f32_bits = sign | 0x7F800000U | mant32;
    } else {
        uint32_t exp32 = exp16 + (127 - 15);
        f32_bits       = sign | (exp32 << 23) | (mant16 << 13);
    }

    union
    {
        uint32_t u;
        float    f;
    } u;
    u.u = f32_bits;
    return u.f;
#endif
}

#endif // AOCL_DLP_FP16_CONVERT_H
