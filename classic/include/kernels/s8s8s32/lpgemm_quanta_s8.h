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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
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

/**
 * @file lpgemm_quanta_s8.h
 * @brief Shared AVX-512 macros and intrinsics for S8 quantization kernels
 *
 * This header provides common AVX-512 VNNI optimized macros for quantizing
 * various input types (FP32, BF16) to signed 8-bit integers (S8).
 *
 * Key Features:
 * - Symmetric quantization: q = round(a * scale)
 * - Asymmetric quantization: q = round(a * scale) - zero_point
 * - Supports per-tensor and per-row quantization parameters
 * - Unrolled operations for 1, 2, 4, 8, and 16 elements
 * - Input-type-specific LOAD macros (FP32, BF16)
 * - Common QUANT and STORE macros for all input types
 * - Transpose macros for column-major input handling
 *
 * Target Architecture: AMD Zen4 with AVX-512 VNNI support
 *
 * Used by:
 * - lpgemm_quanta_f32s8_amd512vnni.c (FP32 → S8 quantization)
 * - lpgemm_quanta_bf16s8_amd512vnni.c (BF16 → S8 quantization)
 */

#ifndef LPGEMM_QUANTA_S8_H
#define LPGEMM_QUANTA_S8_H

#include <immintrin.h>

// ============================================================================
// BASIC AVX-512 INTRINSIC WRAPPERS - COMMON TO ALL INPUT TYPES
// ============================================================================

// Convert FP32 to INT8 with saturation and store
#define STORE_MASKED_INT8(buffer, mask, ptr)                                   \
    _mm512_mask_cvtsepi32_storeu_epi8(buffer, mask, _mm512_cvtps_epi32(ptr));

// Symmetric quantization: q = round(a * scale)
#define QUANT_SYM(reg, sfv)                                                    \
    reg = _mm512_mul_round_ps(                                                 \
        reg, sfv, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

// Asymmetric quantization: q = round(a * scale) - zero_point
#define QUANT_ASYM(reg, sfv, zpv)                                              \
    reg = _mm512_fmsub_round_ps(                                               \
        reg, sfv, zpv, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

// Broadcast scalar values to ZMM registers
#define SET_U8_F32(reg, ptr)                                                   \
    reg = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_set1_epi8(ptr)));

#define SET_S8_F32(reg, ptr)                                                   \
    reg = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_set1_epi8(ptr)));

#define SET_S32_F32(reg, ptr) reg = _mm512_cvtepi32_ps(_mm512_set1_epi32(ptr));

#define SET_BF16_F32(reg, ptr)                                                 \
    reg = (__m512)(_mm512_sllv_epi32(                                          \
        _mm512_cvtepi16_epi32(_mm256_set1_epi16(ptr)),                         \
        _mm512_set1_epi32(16)));

#define SET_F32(reg, ptr) reg = _mm512_set1_ps(ptr);

// ============================================================================
// OPERATION TEMPLATES
// ============================================================================

#define BCST_SFZP_OP(i, MACRO, dest_array, ptr, base_idx, stride)              \
    MACRO(dest_array[i], ptr[base_idx + (i * stride)])

#define QUANT_SYM_OP(i, dest, sf) QUANT_SYM(dest[i], sf[i])

#define QUANT_ASYM_OP(i, dest, sf, zp) QUANT_ASYM(dest[i], sf[i], zp[i])

#define STORE_INT8_OP(i, dst, ic, kr, KC, mask, src)                           \
    STORE_MASKED_INT8(dst + ((ic + i) * KC + kr), mask, src[i])

// ============================================================================
// PUBLIC API - Unrolled operations for 2/4/8/16 elements
// ============================================================================
// Note: LOAD operations are input-type specific and defined in individual
// files (bf16s8 and f32s8). QUANT, STORE, and transpose operations are common.

// --- Scale Factor / Zero Point Broadcast ---
// stride=0 broadcasts same value, stride=1 loads consecutive values
#define BCST_2_SF_ZP(idx, MACRO, dest, ptr, base, stride)                      \
    BCST_SFZP_OP(idx, MACRO, dest, ptr, base, stride)                          \
    BCST_SFZP_OP(idx + 1, MACRO, dest, ptr, base, stride)

#define BCST_4_SF_ZP(idx, MACRO, dest, ptr, base, stride)                      \
    BCST_2_SF_ZP(idx, MACRO, dest, ptr, base, stride)                          \
    BCST_2_SF_ZP(idx + 2, MACRO, dest, ptr, base, stride)

#define BCST_8_SF_ZP(idx, MACRO, dest, ptr, base, stride)                      \
    BCST_4_SF_ZP(idx, MACRO, dest, ptr, base, stride)                          \
    BCST_4_SF_ZP(idx + 4, MACRO, dest, ptr, base, stride)

#define BCST_16_SF_ZP(idx, MACRO, dest, ptr, base, stride)                     \
    BCST_8_SF_ZP(idx, MACRO, dest, ptr, base, stride)                          \
    BCST_8_SF_ZP(idx + 8, MACRO, dest, ptr, base, stride)

// --- Symmetric Quantization ---
// Apply q = round(a * scale) to N elements
#define QUANT_2_SYM(idx, dest, sf)                                             \
    QUANT_SYM_OP(idx, dest, sf)                                                \
    QUANT_SYM_OP(idx + 1, dest, sf)

#define QUANT_4_SYM(idx, dest, sf)                                             \
    QUANT_2_SYM(idx, dest, sf)                                                 \
    QUANT_2_SYM(idx + 2, dest, sf)

#define QUANT_8_SYM(idx, dest, sf)                                             \
    QUANT_4_SYM(idx, dest, sf)                                                 \
    QUANT_4_SYM(idx + 4, dest, sf)

#define QUANT_16_SYM(idx, dest, sf)                                            \
    QUANT_8_SYM(idx, dest, sf)                                                 \
    QUANT_8_SYM(idx + 8, dest, sf)

// --- Asymmetric Quantization ---
// Apply q = round(a * scale) - zero_point to N elements
#define QUANT_2_ASYM(idx, dest, sf, zp)                                        \
    QUANT_ASYM_OP(idx, dest, sf, zp)                                           \
    QUANT_ASYM_OP(idx + 1, dest, sf, zp)
#define QUANT_4_ASYM(idx, dest, sf, zp)                                        \
    QUANT_2_ASYM(idx, dest, sf, zp)                                            \
    QUANT_2_ASYM(idx + 2, dest, sf, zp)

#define QUANT_8_ASYM(idx, dest, sf, zp)                                        \
    QUANT_4_ASYM(idx, dest, sf, zp)                                            \
    QUANT_4_ASYM(idx + 4, dest, sf, zp)
#define QUANT_16_ASYM(idx, dest, sf, zp)                                       \
    QUANT_8_ASYM(idx, dest, sf, zp)                                            \
    QUANT_8_ASYM(idx + 8, dest, sf, zp)

// --- INT8 Stores ---
// Store N quantized FP32 values as saturated INT8
#define STORE_2_INT8(idx, dst, ic, kr, KC, mask, src)                          \
    STORE_INT8_OP(idx, dst, ic, kr, KC, mask, src)                             \
    STORE_INT8_OP(idx + 1, dst, ic, kr, KC, mask, src)
#define STORE_4_INT8(idx, dst, ic, kr, KC, mask, src)                          \
    STORE_2_INT8(idx, dst, ic, kr, KC, mask, src)                              \
    STORE_2_INT8(idx + 2, dst, ic, kr, KC, mask, src)
#define STORE_8_INT8(idx, dst, ic, kr, KC, mask, src)                          \
    STORE_4_INT8(idx, dst, ic, kr, KC, mask, src)                              \
    STORE_4_INT8(idx + 4, dst, ic, kr, KC, mask, src)
#define STORE_16_INT8(idx, dst, ic, kr, KC, mask, src)                         \
    STORE_8_INT8(idx, dst, ic, kr, KC, mask, src)                              \
    STORE_8_INT8(idx + 8, dst, ic, kr, KC, mask, src)

// --- Transpose macros for column-major input ---

// Unpack low and high 32-bit elements from pairs of registers
#define UNPACKLO_PS16                                                          \
    b_reg[0] = _mm512_unpacklo_ps(a_reg[0], a_reg[1]);                         \
    b_reg[1] = _mm512_unpacklo_ps(a_reg[2], a_reg[3]);                         \
    b_reg[2] = _mm512_unpacklo_ps(a_reg[4], a_reg[5]);                         \
    b_reg[3] = _mm512_unpacklo_ps(a_reg[6], a_reg[7]);                         \
    b_reg[4] = _mm512_unpacklo_ps(a_reg[8], a_reg[9]);                         \
    b_reg[5] = _mm512_unpacklo_ps(a_reg[10], a_reg[11]);                       \
    b_reg[6] = _mm512_unpacklo_ps(a_reg[12], a_reg[13]);                       \
    b_reg[7] = _mm512_unpacklo_ps(a_reg[14], a_reg[15]);

#define UNPACKHI_PS16                                                          \
    b_reg[8]  = _mm512_unpackhi_ps(a_reg[0], a_reg[1]);                        \
    b_reg[9]  = _mm512_unpackhi_ps(a_reg[2], a_reg[3]);                        \
    b_reg[10] = _mm512_unpackhi_ps(a_reg[4], a_reg[5]);                        \
    b_reg[11] = _mm512_unpackhi_ps(a_reg[6], a_reg[7]);                        \
    b_reg[12] = _mm512_unpackhi_ps(a_reg[8], a_reg[9]);                        \
    b_reg[13] = _mm512_unpackhi_ps(a_reg[10], a_reg[11]);                      \
    b_reg[14] = _mm512_unpackhi_ps(a_reg[12], a_reg[13]);                      \
    b_reg[15] = _mm512_unpackhi_ps(a_reg[14], a_reg[15]);

// Shuffle 64-bit (2x32-bit) elements within 128-bit lanes
#define SHUFFLE_64x2                                                           \
    a_reg[0] = _mm512_shuffle_ps(b_reg[0], b_reg[1], 0x44);                    \
    a_reg[1] = _mm512_shuffle_ps(b_reg[0], b_reg[1], 0xEE);                    \
    a_reg[2] = _mm512_shuffle_ps(b_reg[2], b_reg[3], 0x44);                    \
    a_reg[3] = _mm512_shuffle_ps(b_reg[2], b_reg[3], 0xEE);                    \
                                                                               \
    a_reg[4] = _mm512_shuffle_ps(b_reg[4], b_reg[5], 0x44);                    \
    a_reg[5] = _mm512_shuffle_ps(b_reg[4], b_reg[5], 0xEE);                    \
    a_reg[6] = _mm512_shuffle_ps(b_reg[6], b_reg[7], 0x44);                    \
    a_reg[7] = _mm512_shuffle_ps(b_reg[6], b_reg[7], 0xEE);                    \
                                                                               \
    a_reg[8]  = _mm512_shuffle_ps(b_reg[8], b_reg[9], 0x44);                   \
    a_reg[9]  = _mm512_shuffle_ps(b_reg[8], b_reg[9], 0xEE);                   \
    a_reg[10] = _mm512_shuffle_ps(b_reg[10], b_reg[11], 0x44);                 \
    a_reg[11] = _mm512_shuffle_ps(b_reg[10], b_reg[11], 0xEE);                 \
                                                                               \
    a_reg[12] = _mm512_shuffle_ps(b_reg[12], b_reg[13], 0x44);                 \
    a_reg[13] = _mm512_shuffle_ps(b_reg[12], b_reg[13], 0xEE);                 \
    a_reg[14] = _mm512_shuffle_ps(b_reg[14], b_reg[15], 0x44);                 \
    a_reg[15] = _mm512_shuffle_ps(b_reg[14], b_reg[15], 0xEE);

// Permute 4x4 blocks of 32-bit elements across 128-bit lanes
#define PERMUTE4x4(mask1, mask2)                                               \
    b_reg[0] = _mm512_permutex2var_ps(a_reg[0], mask1, a_reg[2]);              \
    b_reg[1] = _mm512_permutex2var_ps(a_reg[1], mask1, a_reg[3]);              \
    b_reg[2] = _mm512_permutex2var_ps(a_reg[8], mask1, a_reg[10]);             \
    b_reg[3] = _mm512_permutex2var_ps(a_reg[9], mask1, a_reg[11]);             \
                                                                               \
    b_reg[4] = _mm512_permutex2var_ps(a_reg[4], mask1, a_reg[6]);              \
    b_reg[5] = _mm512_permutex2var_ps(a_reg[5], mask1, a_reg[7]);              \
    b_reg[6] = _mm512_permutex2var_ps(a_reg[12], mask1, a_reg[14]);            \
    b_reg[7] = _mm512_permutex2var_ps(a_reg[13], mask1, a_reg[15]);            \
                                                                               \
    b_reg[8]  = _mm512_permutex2var_ps(a_reg[0], mask2, a_reg[2]);             \
    b_reg[9]  = _mm512_permutex2var_ps(a_reg[1], mask2, a_reg[3]);             \
    b_reg[10] = _mm512_permutex2var_ps(a_reg[8], mask2, a_reg[10]);            \
    b_reg[11] = _mm512_permutex2var_ps(a_reg[9], mask2, a_reg[11]);            \
                                                                               \
    b_reg[12] = _mm512_permutex2var_ps(a_reg[4], mask2, a_reg[6]);             \
    b_reg[13] = _mm512_permutex2var_ps(a_reg[5], mask2, a_reg[7]);             \
    b_reg[14] = _mm512_permutex2var_ps(a_reg[12], mask2, a_reg[14]);           \
    b_reg[15] = _mm512_permutex2var_ps(a_reg[13], mask2, a_reg[15]);

// Permute 8x8 blocks of 32-bit elements across 256-bit lanes (final transpose)
#define PERMUTE8x8(mask3, mask4)                                               \
    a_reg[0] = _mm512_permutex2var_ps(b_reg[0], mask3, b_reg[4]);              \
    a_reg[1] = _mm512_permutex2var_ps(b_reg[1], mask3, b_reg[5]);              \
    a_reg[2] = _mm512_permutex2var_ps(b_reg[2], mask3, b_reg[6]);              \
    a_reg[3] = _mm512_permutex2var_ps(b_reg[3], mask3, b_reg[7]);              \
                                                                               \
    a_reg[4] = _mm512_permutex2var_ps(b_reg[0], mask4, b_reg[4]);              \
    a_reg[5] = _mm512_permutex2var_ps(b_reg[1], mask4, b_reg[5]);              \
    a_reg[6] = _mm512_permutex2var_ps(b_reg[2], mask4, b_reg[6]);              \
    a_reg[7] = _mm512_permutex2var_ps(b_reg[3], mask4, b_reg[7]);              \
                                                                               \
    a_reg[8]  = _mm512_permutex2var_ps(b_reg[8], mask3, b_reg[12]);            \
    a_reg[9]  = _mm512_permutex2var_ps(b_reg[9], mask3, b_reg[13]);            \
    a_reg[10] = _mm512_permutex2var_ps(b_reg[10], mask3, b_reg[14]);           \
    a_reg[11] = _mm512_permutex2var_ps(b_reg[11], mask3, b_reg[15]);           \
                                                                               \
    a_reg[12] = _mm512_permutex2var_ps(b_reg[8], mask4, b_reg[12]);            \
    a_reg[13] = _mm512_permutex2var_ps(b_reg[9], mask4, b_reg[13]);            \
    a_reg[14] = _mm512_permutex2var_ps(b_reg[10], mask4, b_reg[14]);           \
    a_reg[15] = _mm512_permutex2var_ps(b_reg[11], mask4, b_reg[15]);

#include <stdint.h>

typedef void (*quanta_bf16s8)(int8_t*,
                              const bfloat16*,
                              const md_t,
                              const md_t,
                              const md_t,
                              const md_t,
                              const void*,
                              const DLP_TYPE,
                              const md_t,
                              const void*,
                              const DLP_TYPE,
                              const md_t,
                              const md_t);

typedef void (*quanta_f32s8)(int8_t*,
                             const float*,
                             const md_t,
                             const md_t,
                             const md_t,
                             const void*,
                             const DLP_TYPE,
                             const md_t,
                             const void*,
                             const DLP_TYPE,
                             const md_t,
                             const md_t);

void
quanta_mr16_bf16s8(int8_t*         quant_a_buffer,
                   const bfloat16* a,
                   const md_t      rs_a,
                   const md_t      cs_a,
                   const md_t      MC,
                   const md_t      KC,
                   const void*     scale_factor,
                   const DLP_TYPE  sf_type,
                   md_t            sf_len,
                   const void*     zero_point,
                   const DLP_TYPE  zp_type,
                   md_t            zp_len,
                   const md_t      ic_offset);

void
quanta_mr16_f32s8(int8_t*        quant_a_buffer,
                  const float*   a,
                  const md_t     rs_a,
                  const md_t     cs_a,
                  const md_t     MC,
                  const md_t     KC,
                  const void*    scale_factor,
                  const DLP_TYPE sf_type,
                  md_t           sf_len,
                  const void*    zero_point,
                  const DLP_TYPE zp_type,
                  md_t           zp_len,
                  const md_t     ic_offset);

#endif // LPGEMM_QUANTA_S8_H
