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
#ifndef AOCL_DLP_FP16_TYPE_H
#define AOCL_DLP_FP16_TYPE_H

#include <stdint.h>

/**
 * @brief FP16 (IEEE 754 half-precision) type definition
 *
 * FP16 format: 1 sign bit + 5 exponent bits + 10 mantissa bits = 16 bits total
 *
 * Note: Using uint16_t (not int16_t) to avoid sign extension issues
 * during bit manipulation operations.
 *
 * Range: ±6.10×10⁻⁵ to ±65504 (normal numbers)
 * Precision: ~3-4 decimal digits
 */
typedef uint16_t float16;

/** @name FP16 special value bit patterns (IEEE 754 half-precision) */
/** @{ */
#define FP16_ZERO     ((float16)0x0000) /**< +0.0                           */
#define FP16_NEG_ZERO ((float16)0x8000) /**< -0.0                           */
#define FP16_ONE      ((float16)0x3C00) /**< 1.0                            */
#define FP16_NEG_ONE  ((float16)0xBC00) /**< -1.0                           */
#define FP16_POS_INF  ((float16)0x7C00) /**< +Infinity                      */
#define FP16_NEG_INF  ((float16)0xFC00) /**< -Infinity                      */
#define FP16_QNAN     ((float16)0x7E00) /**< Quiet NaN (canonical)          */
#define FP16_MAX      ((float16)0x7BFF) /**< Maximum normal value (65504)   */
#define FP16_MIN_NORM ((float16)0x0400) /**< Minimum positive normal        */
/** @} */

#endif // AOCL_DLP_FP16_TYPE_H
