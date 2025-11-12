
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
#include "kernels/dlp_kernels.h"
#include "kernels/s8s8s32/lpgemm_quanta_bf16s8.h"
#include <immintrin.h>

// Load A BF16 to F32
#define LOAD_BF16_TO_F32(reg, in)                                              \
    reg = (__m512)(_mm512_sllv_epi32(                                          \
        _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i*)(in))),       \
        _mm512_set1_epi32(16)));

// Load A masked BF16 to F32
#define LOAD_MASKED_BF16_TO_F32(reg, mask, in)                                 \
    reg = (__m512)(_mm512_sllv_epi32(                                          \
        _mm512_cvtepi16_epi32(                                                 \
            _mm256_maskz_loadu_epi16(mask, (const __m256i*)(in))),             \
        _mm512_set1_epi32(16)));

// Store int 8
#define STORE_MASKED_INT8(buffer, mask, ptr)                                   \
    _mm512_mask_cvtsepi32_storeu_epi8(buffer, mask, _mm512_cvtps_epi32(ptr));

/*
Asymmetric Quantization
a_quant = round(a * scale_factor) - zero_point
*/
#define QUANT_ASYM(reg, sfv, zpv)                                              \
    reg = _mm512_fmsub_round_ps(                                               \
        reg, sfv, zpv, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
/*
Symmetric Quantization
a_quant = round(a * scale_factor)
*/
#define QUANT_SYM(reg, sfv)                                                    \
    reg = _mm512_mul_round_ps(                                                 \
        reg, sfv, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

// Set 8-bit unsigned integer to 32-bit float
#define SET_U8_F32(reg, ptr)                                                   \
    reg = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_set1_epi8(ptr)));

// Set 8-bit signed integer to 32-bit float
#define SET_S8_F32(reg, ptr)                                                   \
    reg = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_set1_epi8(ptr)));

// Set bfloat16 to 32-bit float
#define SET_BF16_F32(reg, ptr)                                                 \
    reg = (__m512)(_mm512_sllv_epi32(                                          \
        _mm512_cvtepi16_epi32(_mm256_set1_epi16(ptr)),                         \
        _mm512_set1_epi32(16)));

// Set 32-bit integer to 32-bit float
#define SET_S32_F32(reg, ptr) reg = _mm512_cvtepi32_ps(_mm512_set1_epi32(ptr));

// Broadcast bfloat16 to 32-bit float
#define BF16_F32_BCST(reg, ptr)                                                \
    reg = (__m512)(_mm512_sllv_epi32(                                          \
        _mm512_cvtepi16_epi32(_mm256_set1_epi16(ptr)),                         \
        _mm512_set1_epi32(16)));

void
quant_a_sym_bf16s8_row_major(int8_t*         quant_a_buffer,
                             const bfloat16* a,
                             const md_t      rs_a,
                             const md_t      cs_a,
                             const md_t      MC,
                             const md_t      KC,
                             const void*     scale_factor,
                             const DLP_TYPE  sf_type,
                             md_t            sf_len,
                             const md_t      ic_offset);

void
quant_a_asym_bf16s8_row_major(int8_t*         quant_a_buffer,
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
/**
 * quanta_mr16_bf16s8
 *
 * Entry point for BF16 to S8 quantization optimized for AMD Zen4 (AVX-512
 * VNNI). Dispatcher that routes to symmetric or asymmetric quantization based
 * on zero-point presence.
 *
 * @param quant_a_buffer Output buffer for quantized S8 values (MC x KC,
 * row-major with stride KC)
 * @param a              Input BF16 matrix (MC x KC)
 * @param rs_a           Row stride of input matrix a
 * @param cs_a           Column stride of input matrix a (must be 1 for
 * row-major)
 * @param MC             Number of rows
 * @param KC             Number of columns
 * @param scale_factor   Quantization scale factors (length 1 for per-tensor, MC
 * for per-row)
 * @param sf_type        Data type of scale_factor (U8/S8/S32/BF16/F32)
 * @param sf_len         Length of scale_factor array (1=per-tensor,
 * >=MC=per-row)
 * @param zero_point     Zero-point values (NULL for symmetric, non-NULL for
 * asymmetric)
 * @param zp_type        Data type of zero_point
 * @param zp_len         Length of zero_point array
 * @param ic_offset      Row offset for indexing
 *
 * Note: Only row-major layout (cs_a==1) is supported. Returns immediately for
 * other layouts.
 */
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
                   const md_t      ic_offset)
{
    // Only row-major input (cs_a == 1) is supported by these kernels.
    if (cs_a == 1) {
        if (zero_point) {
            // Asymmetric quantization: q = round(a * scale) - zero_point
            quant_a_asym_bf16s8_row_major(
                quant_a_buffer, a, rs_a, cs_a, MC, KC, scale_factor, sf_type,
                sf_len, zero_point, zp_type, zp_len, ic_offset);
        } else {
            // Symmetric quantization: q = round(a * scale)
            quant_a_sym_bf16s8_row_major(quant_a_buffer, a, rs_a, cs_a, MC, KC,
                                         scale_factor, sf_type, sf_len,
                                         ic_offset);
        }
    } else {
        dlp_print_msg(" Column-major or other layouts not supported.", __FILE__,
                      __LINE__);
        return;
    }
}

/*
 * quant_a_sym_bf16s8_row_major
 *
 * Convert an MC x KC BF16 matrix (row-major, cs_a==1) to int8 using symmetric
 * per-tensor (sf_len==1) or per-row (sf_len>=MC) scale factors:
 *   q = round( a * scale[row] )
 *
 * Algorithm:
 *   - Processes columns in 16-element AVX-512 blocks (vectorized K dimension)
 *   - Final tail < 16 elements uses masked operations
 *   - Row blocking: 16/8/4/2/1 unrolling to reduce scalar cleanup
 *   - Pipeline: Load BF16 -> FP32, multiply by scale, round, saturate store to
 * S8
 *
 * Output layout: Dense row-major (row stride = KC)
 */
void
quant_a_sym_bf16s8_row_major(int8_t*         quant_a_buffer,
                             const bfloat16* a,
                             const md_t      rs_a,
                             const md_t      cs_a,
                             const md_t      MC,
                             const md_t      KC,
                             const void*     scale_factor,
                             const DLP_TYPE  sf_type,
                             md_t            sf_len,
                             const md_t      ic_offset)
{
    // AVX-512 parameters: process 16 elements per vector register.
    md_t      MR       = 16;            // Max rows per unrolled block
    md_t      NUM_ELEM = 16;            // Elements per ZMM register
    md_t      kleft    = KC % NUM_ELEM; // Tail elements
    __mmask16 mask = 0xFFFF >> (NUM_ELEM - kleft); // Mask for tail processing

    __m512 a_reg[MR]; // Temporary registers for input data
    __m512 sf[MR];    // Scale factors broadcasted into registers

    md_t ic = 0, kr = 0;
    md_t sf_idx =
        0; // Index into scale_factor array (adjusted by ic_offset for per-row)

    // broadcast scale factor for tensor quantization
    if (sf_len == 1) {
        if (sf_type == DLP_U8) {
            SET_U8_F32(sf[0], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[1], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[2], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[3], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[4], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[5], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[6], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[7], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[8], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[9], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[10], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[11], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[12], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[13], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[14], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[15], ((const uint8_t*)scale_factor)[0])
        } else if (sf_type == DLP_S8) {
            SET_S8_F32(sf[0], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[1], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[2], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[3], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[4], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[5], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[6], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[7], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[8], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[9], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[10], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[11], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[12], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[13], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[14], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[15], ((const int8_t*)scale_factor)[0])
        } else if (sf_type == DLP_S32) {
            SET_S32_F32(sf[0], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[1], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[2], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[3], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[4], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[5], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[6], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[7], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[8], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[9], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[10], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[11], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[12], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[13], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[14], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[15], ((const int32_t*)scale_factor)[0])
        } else if (sf_type == DLP_BF16) {
            SET_BF16_F32(sf[0], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[1], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[2], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[3], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[4], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[5], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[6], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[7], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[8], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[9], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[10], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[11], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[12], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[13], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[14], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[15], ((const bfloat16*)scale_factor)[0])
        } else {
            sf[0]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[1]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[2]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[3]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[4]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[5]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[6]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[7]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[8]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[9]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[10] = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[11] = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[12] = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[13] = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[14] = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[15] = _mm512_set1_ps(((const float*)scale_factor)[0]);
        }
    }

    for (; (ic + MR - 1) < MC; ic += MR) {
        // broadcast scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                SET_U8_F32(sf[0], ((const uint8_t*)scale_factor)[sf_idx])
                SET_U8_F32(sf[1], ((const uint8_t*)scale_factor)[sf_idx + 1])
                SET_U8_F32(sf[2], ((const uint8_t*)scale_factor)[sf_idx + 2])
                SET_U8_F32(sf[3], ((const uint8_t*)scale_factor)[sf_idx + 3])
                SET_U8_F32(sf[4], ((const uint8_t*)scale_factor)[sf_idx + 4])
                SET_U8_F32(sf[5], ((const uint8_t*)scale_factor)[sf_idx + 5])
                SET_U8_F32(sf[6], ((const uint8_t*)scale_factor)[sf_idx + 6])
                SET_U8_F32(sf[7], ((const uint8_t*)scale_factor)[sf_idx + 7])
                SET_U8_F32(sf[8], ((const uint8_t*)scale_factor)[sf_idx + 8])
                SET_U8_F32(sf[9], ((const uint8_t*)scale_factor)[sf_idx + 9])
                SET_U8_F32(sf[10], ((const uint8_t*)scale_factor)[sf_idx + 10])
                SET_U8_F32(sf[11], ((const uint8_t*)scale_factor)[sf_idx + 11])
                SET_U8_F32(sf[12], ((const uint8_t*)scale_factor)[sf_idx + 12])
                SET_U8_F32(sf[13], ((const uint8_t*)scale_factor)[sf_idx + 13])
                SET_U8_F32(sf[14], ((const uint8_t*)scale_factor)[sf_idx + 14])
                SET_U8_F32(sf[15], ((const uint8_t*)scale_factor)[sf_idx + 15])
            } else if (sf_type == DLP_S8) {
                SET_S8_F32(sf[0], ((const int8_t*)scale_factor)[sf_idx])
                SET_S8_F32(sf[1], ((const int8_t*)scale_factor)[sf_idx + 1])
                SET_S8_F32(sf[2], ((const int8_t*)scale_factor)[sf_idx + 2])
                SET_S8_F32(sf[3], ((const int8_t*)scale_factor)[sf_idx + 3])
                SET_S8_F32(sf[4], ((const int8_t*)scale_factor)[sf_idx + 4])
                SET_S8_F32(sf[5], ((const int8_t*)scale_factor)[sf_idx + 5])
                SET_S8_F32(sf[6], ((const int8_t*)scale_factor)[sf_idx + 6])
                SET_S8_F32(sf[7], ((const int8_t*)scale_factor)[sf_idx + 7])
                SET_S8_F32(sf[8], ((const int8_t*)scale_factor)[sf_idx + 8])
                SET_S8_F32(sf[9], ((const int8_t*)scale_factor)[sf_idx + 9])
                SET_S8_F32(sf[10], ((const int8_t*)scale_factor)[sf_idx + 10])
                SET_S8_F32(sf[11], ((const int8_t*)scale_factor)[sf_idx + 11])
                SET_S8_F32(sf[12], ((const int8_t*)scale_factor)[sf_idx + 12])
                SET_S8_F32(sf[13], ((const int8_t*)scale_factor)[sf_idx + 13])
                SET_S8_F32(sf[14], ((const int8_t*)scale_factor)[sf_idx + 14])
                SET_S8_F32(sf[15], ((const int8_t*)scale_factor)[sf_idx + 15]);
            } else if (sf_type == DLP_BF16) {
                SET_BF16_F32(sf[0], ((const bfloat16*)scale_factor)[sf_idx])
                SET_BF16_F32(sf[1], ((const bfloat16*)scale_factor)[sf_idx + 1])
                SET_BF16_F32(sf[2], ((const bfloat16*)scale_factor)[sf_idx + 2])
                SET_BF16_F32(sf[3], ((const bfloat16*)scale_factor)[sf_idx + 3])
                SET_BF16_F32(sf[4], ((const bfloat16*)scale_factor)[sf_idx + 4])
                SET_BF16_F32(sf[5], ((const bfloat16*)scale_factor)[sf_idx + 5])
                SET_BF16_F32(sf[6], ((const bfloat16*)scale_factor)[sf_idx + 6])
                SET_BF16_F32(sf[7], ((const bfloat16*)scale_factor)[sf_idx + 7])
                SET_BF16_F32(sf[8], ((const bfloat16*)scale_factor)[sf_idx + 8])
                SET_BF16_F32(sf[9], ((const bfloat16*)scale_factor)[sf_idx + 9])
                SET_BF16_F32(sf[10],
                             ((const bfloat16*)scale_factor)[sf_idx + 10])
                SET_BF16_F32(sf[11],
                             ((const bfloat16*)scale_factor)[sf_idx + 11])
                SET_BF16_F32(sf[12],
                             ((const bfloat16*)scale_factor)[sf_idx + 12])
                SET_BF16_F32(sf[13],
                             ((const bfloat16*)scale_factor)[sf_idx + 13])
                SET_BF16_F32(sf[14],
                             ((const bfloat16*)scale_factor)[sf_idx + 14])
                SET_BF16_F32(sf[15],
                             ((const bfloat16*)scale_factor)[sf_idx + 15])
            } else if (sf_type == DLP_S32) {
                SET_S32_F32(sf[0], ((const int32_t*)scale_factor)[sf_idx])
                SET_S32_F32(sf[1], ((const int32_t*)scale_factor)[sf_idx + 1])
                SET_S32_F32(sf[2], ((const int32_t*)scale_factor)[sf_idx + 2])
                SET_S32_F32(sf[3], ((const int32_t*)scale_factor)[sf_idx + 3])
                SET_S32_F32(sf[4], ((const int32_t*)scale_factor)[sf_idx + 4])
                SET_S32_F32(sf[5], ((const int32_t*)scale_factor)[sf_idx + 5])
                SET_S32_F32(sf[6], ((const int32_t*)scale_factor)[sf_idx + 6])
                SET_S32_F32(sf[7], ((const int32_t*)scale_factor)[sf_idx + 7])
                SET_S32_F32(sf[8], ((const int32_t*)scale_factor)[sf_idx + 8])
                SET_S32_F32(sf[9], ((const int32_t*)scale_factor)[sf_idx + 9])
                SET_S32_F32(sf[10], ((const int32_t*)scale_factor)[sf_idx + 10])
                SET_S32_F32(sf[11], ((const int32_t*)scale_factor)[sf_idx + 11])
                SET_S32_F32(sf[12], ((const int32_t*)scale_factor)[sf_idx + 12])
                SET_S32_F32(sf[13], ((const int32_t*)scale_factor)[sf_idx + 13])
                SET_S32_F32(sf[14], ((const int32_t*)scale_factor)[sf_idx + 14])
                SET_S32_F32(sf[15], ((const int32_t*)scale_factor)[sf_idx + 15])
            } else {
                sf[0] = _mm512_set1_ps(((const float*)scale_factor)[sf_idx]);
                sf[1] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 1]);
                sf[2] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 2]);
                sf[3] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 3]);
                sf[4] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 4]);
                sf[5] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 5]);
                sf[6] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 6]);
                sf[7] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 7]);
                sf[8] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 8]);
                sf[9] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 9]);
                sf[10] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 10]);
                sf[11] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 11]);
                sf[12] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 12]);
                sf[13] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 13]);
                sf[14] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 14]);
                sf[15] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 15]);
            }
        }

        for (kr = 0; (kr + 16 - 1) < KC; kr += 16) {
            // convert a bf16 to f32 and load in a_reg[0-15]
            LOAD_BF16_TO_F32(a_reg[0], a + ((ic + 0) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[1], a + ((ic + 1) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[2], a + ((ic + 2) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[3], a + ((ic + 3) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[4], a + ((ic + 4) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[5], a + ((ic + 5) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[6], a + ((ic + 6) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[7], a + ((ic + 7) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[8], a + ((ic + 8) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[9], a + ((ic + 9) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[10], a + ((ic + 10) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[11], a + ((ic + 11) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[12], a + ((ic + 12) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[13], a + ((ic + 13) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[14], a + ((ic + 14) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[15], a + ((ic + 15) * rs_a + kr * cs_a))
            // QUANT_SYM a_reg[0-15]
            QUANT_SYM(a_reg[0], sf[0])
            QUANT_SYM(a_reg[1], sf[1])
            QUANT_SYM(a_reg[2], sf[2])
            QUANT_SYM(a_reg[3], sf[3])
            QUANT_SYM(a_reg[4], sf[4])
            QUANT_SYM(a_reg[5], sf[5])
            QUANT_SYM(a_reg[6], sf[6])
            QUANT_SYM(a_reg[7], sf[7])
            QUANT_SYM(a_reg[8], sf[8])
            QUANT_SYM(a_reg[9], sf[9])
            QUANT_SYM(a_reg[10], sf[10])
            QUANT_SYM(a_reg[11], sf[11])
            QUANT_SYM(a_reg[12], sf[12])
            QUANT_SYM(a_reg[13], sf[13])
            QUANT_SYM(a_reg[14], sf[14])
            QUANT_SYM(a_reg[15], sf[15])
            // Saturate FP32 to S8 range [-128,127] and store to output buffer.
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 0) * KC + kr), 0xFFFF,
                              a_reg[0]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 1) * KC + kr), 0xFFFF,
                              a_reg[1]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 2) * KC + kr), 0xFFFF,
                              a_reg[2]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 3) * KC + kr), 0xFFFF,
                              a_reg[3]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 4) * KC + kr), 0xFFFF,
                              a_reg[4]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 5) * KC + kr), 0xFFFF,
                              a_reg[5]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 6) * KC + kr), 0xFFFF,
                              a_reg[6]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 7) * KC + kr), 0xFFFF,
                              a_reg[7]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 8) * KC + kr), 0xFFFF,
                              a_reg[8]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 9) * KC + kr), 0xFFFF,
                              a_reg[9]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 10) * KC + kr), 0xFFFF,
                              a_reg[10]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 11) * KC + kr), 0xFFFF,
                              a_reg[11]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 12) * KC + kr), 0xFFFF,
                              a_reg[12]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 13) * KC + kr), 0xFFFF,
                              a_reg[13]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 14) * KC + kr), 0xFFFF,
                              a_reg[14]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 15) * KC + kr), 0xFFFF,
                              a_reg[15]);
        }

        if (kleft) {
            // Load masked BF16 values into FP32
            LOAD_MASKED_BF16_TO_F32(a_reg[0], mask, a + (ic * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[1], mask,
                                    a + ((ic + 1) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[2], mask,
                                    a + ((ic + 2) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[3], mask,
                                    a + ((ic + 3) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[4], mask,
                                    a + ((ic + 4) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[5], mask,
                                    a + ((ic + 5) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[6], mask,
                                    a + ((ic + 6) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[7], mask,
                                    a + ((ic + 7) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[8], mask,
                                    a + ((ic + 8) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[9], mask,
                                    a + ((ic + 9) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[10], mask,
                                    a + ((ic + 10) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[11], mask,
                                    a + ((ic + 11) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[12], mask,
                                    a + ((ic + 12) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[13], mask,
                                    a + ((ic + 13) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[14], mask,
                                    a + ((ic + 14) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[15], mask,
                                    a + ((ic + 15) * rs_a + kr * cs_a))
            // QUANT_SYM a_reg[0-15]
            QUANT_SYM(a_reg[0], sf[0])
            QUANT_SYM(a_reg[1], sf[1])
            QUANT_SYM(a_reg[2], sf[2])
            QUANT_SYM(a_reg[3], sf[3])
            QUANT_SYM(a_reg[4], sf[4])
            QUANT_SYM(a_reg[5], sf[5])
            QUANT_SYM(a_reg[6], sf[6])
            QUANT_SYM(a_reg[7], sf[7])
            QUANT_SYM(a_reg[8], sf[8])
            QUANT_SYM(a_reg[9], sf[9])
            QUANT_SYM(a_reg[10], sf[10])
            QUANT_SYM(a_reg[11], sf[11])
            QUANT_SYM(a_reg[12], sf[12])
            QUANT_SYM(a_reg[13], sf[13])
            QUANT_SYM(a_reg[14], sf[14])
            QUANT_SYM(a_reg[15], sf[15])
            // Convert to int 8 and store
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 0) * KC + kr), mask,
                              a_reg[0])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 1) * KC + kr), mask,
                              a_reg[1])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 2) * KC + kr), mask,
                              a_reg[2])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 3) * KC + kr), mask,
                              a_reg[3])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 4) * KC + kr), mask,
                              a_reg[4])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 5) * KC + kr), mask,
                              a_reg[5])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 6) * KC + kr), mask,
                              a_reg[6])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 7) * KC + kr), mask,
                              a_reg[7])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 8) * KC + kr), mask,
                              a_reg[8])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 9) * KC + kr), mask,
                              a_reg[9])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 10) * KC + kr), mask,
                              a_reg[10])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 11) * KC + kr), mask,
                              a_reg[11])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 12) * KC + kr), mask,
                              a_reg[12])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 13) * KC + kr), mask,
                              a_reg[13])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 14) * KC + kr), mask,
                              a_reg[14])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 15) * KC + kr), mask,
                              a_reg[15])
        }
    }

    for (; (ic + 8 - 1) < MC; ic += 8) {
        // broadcast scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                SET_U8_F32(sf[0], ((const uint8_t*)scale_factor)[sf_idx])
                SET_U8_F32(sf[1], ((const uint8_t*)scale_factor)[sf_idx + 1])
                SET_U8_F32(sf[2], ((const uint8_t*)scale_factor)[sf_idx + 2])
                SET_U8_F32(sf[3], ((const uint8_t*)scale_factor)[sf_idx + 3])
                SET_U8_F32(sf[4], ((const uint8_t*)scale_factor)[sf_idx + 4])
                SET_U8_F32(sf[5], ((const uint8_t*)scale_factor)[sf_idx + 5])
                SET_U8_F32(sf[6], ((const uint8_t*)scale_factor)[sf_idx + 6])
                SET_U8_F32(sf[7], ((const uint8_t*)scale_factor)[sf_idx + 7])
            } else if (sf_type == DLP_S8) {
                SET_S8_F32(sf[0], ((const int8_t*)scale_factor)[sf_idx])
                SET_S8_F32(sf[1], ((const int8_t*)scale_factor)[sf_idx + 1])
                SET_S8_F32(sf[2], ((const int8_t*)scale_factor)[sf_idx + 2])
                SET_S8_F32(sf[3], ((const int8_t*)scale_factor)[sf_idx + 3])
                SET_S8_F32(sf[4], ((const int8_t*)scale_factor)[sf_idx + 4])
                SET_S8_F32(sf[5], ((const int8_t*)scale_factor)[sf_idx + 5])
                SET_S8_F32(sf[6], ((const int8_t*)scale_factor)[sf_idx + 6])
                SET_S8_F32(sf[7], ((const int8_t*)scale_factor)[sf_idx + 7])
            } else if (sf_type == DLP_S32) {
                SET_S32_F32(sf[0], ((const int32_t*)scale_factor)[sf_idx])
                SET_S32_F32(sf[1], ((const int32_t*)scale_factor)[sf_idx + 1])
                SET_S32_F32(sf[2], ((const int32_t*)scale_factor)[sf_idx + 2])
                SET_S32_F32(sf[3], ((const int32_t*)scale_factor)[sf_idx + 3])
                SET_S32_F32(sf[4], ((const int32_t*)scale_factor)[sf_idx + 4])
                SET_S32_F32(sf[5], ((const int32_t*)scale_factor)[sf_idx + 5])
                SET_S32_F32(sf[6], ((const int32_t*)scale_factor)[sf_idx + 6])
                SET_S32_F32(sf[7], ((const int32_t*)scale_factor)[sf_idx + 7])
            } else if (sf_type == DLP_BF16) {
                SET_BF16_F32(sf[0], ((const bfloat16*)scale_factor)[sf_idx])
                SET_BF16_F32(sf[1], ((const bfloat16*)scale_factor)[sf_idx + 1])
                SET_BF16_F32(sf[2], ((const bfloat16*)scale_factor)[sf_idx + 2])
                SET_BF16_F32(sf[3], ((const bfloat16*)scale_factor)[sf_idx + 3])
                SET_BF16_F32(sf[4], ((const bfloat16*)scale_factor)[sf_idx + 4])
                SET_BF16_F32(sf[5], ((const bfloat16*)scale_factor)[sf_idx + 5])
                SET_BF16_F32(sf[6], ((const bfloat16*)scale_factor)[sf_idx + 6])
                SET_BF16_F32(sf[7], ((const bfloat16*)scale_factor)[sf_idx + 7])
            } else {
                sf[0] = _mm512_set1_ps(((const float*)scale_factor)[sf_idx]);
                sf[1] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 1]);
                sf[2] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 2]);
                sf[3] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 3]);
                sf[4] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 4]);
                sf[5] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 5]);
                sf[6] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 6]);
                sf[7] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 7]);
            }
        }

        for (kr = 0; (kr + 16 - 1) < KC; kr += 16) {
            // convert a bf16 to f32 and load in a_reg[0-7]
            LOAD_BF16_TO_F32(a_reg[0], a + ((ic + 0) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[1], a + ((ic + 1) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[2], a + ((ic + 2) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[3], a + ((ic + 3) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[4], a + ((ic + 4) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[5], a + ((ic + 5) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[6], a + ((ic + 6) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[7], a + ((ic + 7) * rs_a + kr * cs_a))
            // QUANT_SYM a_reg[0-7]
            QUANT_SYM(a_reg[0], sf[0])
            QUANT_SYM(a_reg[1], sf[1])
            QUANT_SYM(a_reg[2], sf[2])
            QUANT_SYM(a_reg[3], sf[3])
            QUANT_SYM(a_reg[4], sf[4])
            QUANT_SYM(a_reg[5], sf[5])
            QUANT_SYM(a_reg[6], sf[6])
            QUANT_SYM(a_reg[7], sf[7])
            // convert to int32 and store
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 0) * KC + kr), 0xFFFF,
                              a_reg[0])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 1) * KC + kr), 0xFFFF,
                              a_reg[1])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 2) * KC + kr), 0xFFFF,
                              a_reg[2])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 3) * KC + kr), 0xFFFF,
                              a_reg[3])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 4) * KC + kr), 0xFFFF,
                              a_reg[4])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 5) * KC + kr), 0xFFFF,
                              a_reg[5])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 6) * KC + kr), 0xFFFF,
                              a_reg[6])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 7) * KC + kr), 0xFFFF,
                              a_reg[7])
        }

        if (kleft) {
            // Load masked BF16 values into FP32
            LOAD_MASKED_BF16_TO_F32(a_reg[0], mask, a + (ic * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[1], mask,
                                    a + ((ic + 1) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[2], mask,
                                    a + ((ic + 2) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[3], mask,
                                    a + ((ic + 3) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[4], mask,
                                    a + ((ic + 4) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[5], mask,
                                    a + ((ic + 5) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[6], mask,
                                    a + ((ic + 6) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[7], mask,
                                    a + ((ic + 7) * rs_a + kr * cs_a))
            // QUANT_SYM a_reg[0-7]
            QUANT_SYM(a_reg[0], sf[0])
            QUANT_SYM(a_reg[1], sf[1])
            QUANT_SYM(a_reg[2], sf[2])
            QUANT_SYM(a_reg[3], sf[3])
            QUANT_SYM(a_reg[4], sf[4])
            QUANT_SYM(a_reg[5], sf[5])
            QUANT_SYM(a_reg[6], sf[6])
            QUANT_SYM(a_reg[7], sf[7])
            // Convert to int8 and store
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 0) * KC + kr), mask,
                              a_reg[0])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 1) * KC + kr), mask,
                              a_reg[1])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 2) * KC + kr), mask,
                              a_reg[2])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 3) * KC + kr), mask,
                              a_reg[3])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 4) * KC + kr), mask,
                              a_reg[4])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 5) * KC + kr), mask,
                              a_reg[5])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 6) * KC + kr), mask,
                              a_reg[6])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 7) * KC + kr), mask,
                              a_reg[7])
        }
    }

    for (; (ic + 4 - 1) < MC; ic += 4) {
        // broadcast scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                SET_U8_F32(sf[0], ((const uint8_t*)scale_factor)[sf_idx])
                SET_U8_F32(sf[1], ((const uint8_t*)scale_factor)[sf_idx + 1])
                SET_U8_F32(sf[2], ((const uint8_t*)scale_factor)[sf_idx + 2])
                SET_U8_F32(sf[3], ((const uint8_t*)scale_factor)[sf_idx + 3])
            } else if (sf_type == DLP_S8) {
                SET_S8_F32(sf[0], ((const int8_t*)scale_factor)[sf_idx])
                SET_S8_F32(sf[1], ((const int8_t*)scale_factor)[sf_idx + 1])
                SET_S8_F32(sf[2], ((const int8_t*)scale_factor)[sf_idx + 2])
                SET_S8_F32(sf[3], ((const int8_t*)scale_factor)[sf_idx + 3])
            } else if (sf_type == DLP_S32) {
                SET_S32_F32(sf[0], ((const int32_t*)scale_factor)[sf_idx])
                SET_S32_F32(sf[1], ((const int32_t*)scale_factor)[sf_idx + 1])
                SET_S32_F32(sf[2], ((const int32_t*)scale_factor)[sf_idx + 2])
                SET_S32_F32(sf[3], ((const int32_t*)scale_factor)[sf_idx + 3])
            } else if (sf_type == DLP_BF16) {
                SET_BF16_F32(sf[0], ((const bfloat16*)scale_factor)[sf_idx])
                SET_BF16_F32(sf[1], ((const bfloat16*)scale_factor)[sf_idx + 1])
                SET_BF16_F32(sf[2], ((const bfloat16*)scale_factor)[sf_idx + 2])
                SET_BF16_F32(sf[3], ((const bfloat16*)scale_factor)[sf_idx + 3])
            } else {
                sf[0] = _mm512_set1_ps(((const float*)scale_factor)[sf_idx]);
                sf[1] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 1]);
                sf[2] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 2]);
                sf[3] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 3]);
            }
        }

        for (kr = 0; (kr + 16 - 1) < KC; kr += 16) {
            // convert a bf16 to f32 and load in a_reg[0-3]
            LOAD_BF16_TO_F32(a_reg[0], a + ((ic + 0) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[1], a + ((ic + 1) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[2], a + ((ic + 2) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[3], a + ((ic + 3) * rs_a + kr * cs_a))
            // QUANT_SYM a_reg[0-3]
            QUANT_SYM(a_reg[0], sf[0])
            QUANT_SYM(a_reg[1], sf[1])
            QUANT_SYM(a_reg[2], sf[2])
            QUANT_SYM(a_reg[3], sf[3])
            // convert to int32 and store
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 0) * KC + kr), 0xFFFF,
                              a_reg[0])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 1) * KC + kr), 0xFFFF,
                              a_reg[1])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 2) * KC + kr), 0xFFFF,
                              a_reg[2])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 3) * KC + kr), 0xFFFF,
                              a_reg[3])
        }

        if (kleft) {
            // load masked bf16 values into fp32
            LOAD_MASKED_BF16_TO_F32(a_reg[0], mask, a + (ic * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[1], mask,
                                    a + ((ic + 1) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[2], mask,
                                    a + ((ic + 2) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[3], mask,
                                    a + ((ic + 3) * rs_a + kr * cs_a))
            // QUANT_SYM
            QUANT_SYM(a_reg[0], sf[0])
            QUANT_SYM(a_reg[1], sf[1])
            QUANT_SYM(a_reg[2], sf[2])
            QUANT_SYM(a_reg[3], sf[3])
            // convert to int8 and store
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 0) * KC + kr), mask,
                              a_reg[0])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 1) * KC + kr), mask,
                              a_reg[1])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 2) * KC + kr), mask,
                              a_reg[2])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 3) * KC + kr), mask,
                              a_reg[3])
        }
    }

    for (; (ic + 2 - 1) < MC; ic += 2) {
        // broadcast scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                SET_U8_F32(sf[0], ((const uint8_t*)scale_factor)[sf_idx])
                SET_U8_F32(sf[1], ((const uint8_t*)scale_factor)[sf_idx + 1])
            } else if (sf_type == DLP_S8) {
                SET_S8_F32(sf[0], ((const int8_t*)scale_factor)[sf_idx])
                SET_S8_F32(sf[1], ((const int8_t*)scale_factor)[sf_idx + 1])
            } else if (sf_type == DLP_S32) {
                SET_S32_F32(sf[0], ((const int32_t*)scale_factor)[sf_idx])
                SET_S32_F32(sf[1], ((const int32_t*)scale_factor)[sf_idx + 1])
            } else if (sf_type == DLP_BF16) {
                SET_BF16_F32(sf[0], ((const bfloat16*)scale_factor)[sf_idx])
                SET_BF16_F32(sf[1], ((const bfloat16*)scale_factor)[sf_idx + 1])
            } else {
                sf[0] = _mm512_set1_ps(((const float*)scale_factor)[sf_idx]);
                sf[1] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 1]);
            }
        }

        for (kr = 0; (kr + 16 - 1) < KC; kr += 16) {
            // convert a bf16 to f32 and load in a_reg[0-3]
            LOAD_BF16_TO_F32(a_reg[0], a + ((ic)*rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[1], a + ((ic + 1) * rs_a + kr * cs_a))
            // QUANT_SYM a_reg[0-3]
            QUANT_SYM(a_reg[0], sf[0])
            QUANT_SYM(a_reg[1], sf[1])
            // convert to int 32 and store
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 0) * KC + kr), 0xFFFF,
                              a_reg[0])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 1) * KC + kr), 0xFFFF,
                              a_reg[1])
        }

        if (kleft) {
            // load masked bf16 values into fp32
            LOAD_MASKED_BF16_TO_F32(a_reg[0], mask, a + (ic * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[1], mask,
                                    a + ((ic + 1) * rs_a + kr * cs_a))
            // QUANT_SYM
            QUANT_SYM(a_reg[0], sf[0])
            QUANT_SYM(a_reg[1], sf[1])
            // convert to int8 and store
            STORE_MASKED_INT8(quant_a_buffer + ((ic)*KC + kr), mask, a_reg[0])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 1) * KC + kr), mask,
                              a_reg[1])
        }
    }

    for (; ic < MC; ic += 1) {
        // broadcast scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                SET_U8_F32(sf[0], ((const uint8_t*)scale_factor)[sf_idx])
            } else if (sf_type == DLP_S8) {
                SET_S8_F32(sf[0], ((const int8_t*)scale_factor)[sf_idx])
            } else if (sf_type == DLP_S32) {
                SET_S32_F32(sf[0], ((const int32_t*)scale_factor)[sf_idx])
            } else if (sf_type == DLP_BF16) {
                SET_BF16_F32(sf[0], ((const bfloat16*)scale_factor)[sf_idx])
            } else {
                sf[0] = _mm512_set1_ps(((const float*)scale_factor)[sf_idx]);
            }
        }

        for (kr = 0; (kr + 16 - 1) < KC; kr += 16) {
            // convert a bf16 to f32 and load in a_reg[0]
            LOAD_BF16_TO_F32(a_reg[0], a + (ic * rs_a + kr * cs_a))
            // QUANT_SYM
            QUANT_SYM(a_reg[0], sf[0])
            // convert to int8 and store
            STORE_MASKED_INT8(quant_a_buffer + ((ic)*KC + kr), 0xFFFF, a_reg[0])
        }

        if (kleft) {
            // load masked bf16 values into fp32
            LOAD_MASKED_BF16_TO_F32(a_reg[0], mask, a + (ic * rs_a + kr * cs_a))
            // QUANT_SYM
            QUANT_SYM(a_reg[0], sf[0]);
            // convert to int8 and store
            STORE_MASKED_INT8(quant_a_buffer + (ic * KC + kr), mask, a_reg[0])
        }
    }
}
/*
 * quant_a_asym_bf16s8_row_major
 *
 * Convert an MC x KC BF16 matrix (row-major, cs_a==1) to int8 using asymmetric
 * per-tensor (sf_len==1) or per-row (sf_len>=MC) quantization:
 *   q = round( a * scale[row] ) - zero_point[row]
 *
 * Asymmetric quantization maps the input range [a_min, a_max] to [-128, 127]
 * with an arbitrary zero-point, providing better accuracy for non-centered
 * distributions.
 *
 * Algorithm: Same vectorization strategy as symmetric version, but with
 * additional zero-point subtraction. Uses FMA (fused multiply-subtract) for
 * efficiency: q = FMA(a, scale, -zero_point) = a * scale - zero_point
 *
 * Note: The QUANT_ASYM macro implements: fmsub(a, scale, zero_point) with
 * rounding.
 */
void
quant_a_asym_bf16s8_row_major(int8_t*         quant_a_buffer,
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
                              const md_t      ic_offset)
{
    md_t      MR       = 16;
    md_t      NUM_ELEM = 16;
    md_t      kleft    = KC % NUM_ELEM;
    __m512    a_reg[MR];
    __m512    sf[MR]; // Scale factors
    __m512    zp[MR]; // Zero-points
    __mmask16 mask = 0xFFFF >> (NUM_ELEM - kleft);

    md_t ic = 0, kr = 0;
    md_t sf_idx = 0; // Index into scale_factor array (adjusted by ic_offset)
    md_t zp_idx = 0; // Index into zero_point array (adjusted by ic_offset)

    // Initialize to avoid uninitialized register warnings.
    for (md_t i = 0; i < MR; i++) {
        sf[i] = _mm512_setzero_ps();
        zp[i] = _mm512_setzero_ps();
    }

    // broadcast scale factor for tensor quantization
    if (sf_len == 1) {
        if (sf_type == DLP_U8) {
            SET_U8_F32(sf[0], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[1], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[2], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[3], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[4], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[5], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[6], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[7], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[8], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[9], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[10], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[11], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[12], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[13], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[14], ((const uint8_t*)scale_factor)[0])
            SET_U8_F32(sf[15], ((const uint8_t*)scale_factor)[0])
        } else if (sf_type == DLP_S8) {
            SET_S8_F32(sf[0], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[1], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[2], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[3], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[4], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[5], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[6], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[7], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[8], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[9], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[10], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[11], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[12], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[13], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[14], ((const int8_t*)scale_factor)[0])
            SET_S8_F32(sf[15], ((const int8_t*)scale_factor)[0])
        } else if (sf_type == DLP_S32) {
            SET_S32_F32(sf[0], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[1], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[2], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[3], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[4], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[5], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[6], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[7], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[8], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[9], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[10], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[11], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[12], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[13], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[14], ((const int32_t*)scale_factor)[0])
            SET_S32_F32(sf[15], ((const int32_t*)scale_factor)[0])
        } else if (sf_type == DLP_BF16) {
            SET_BF16_F32(sf[0], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[1], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[2], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[3], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[4], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[5], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[6], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[7], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[8], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[9], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[10], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[11], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[12], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[13], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[14], ((const bfloat16*)scale_factor)[0])
            SET_BF16_F32(sf[15], ((const bfloat16*)scale_factor)[0])
        } else {
            sf[0]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[1]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[2]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[3]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[4]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[5]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[6]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[7]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[8]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[9]  = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[10] = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[11] = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[12] = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[13] = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[14] = _mm512_set1_ps(((const float*)scale_factor)[0]);
            sf[15] = _mm512_set1_ps(((const float*)scale_factor)[0]);
        }
    }

    // =========================================================================
    // ZERO-POINT SETUP: Per-tensor quantization (zp_len == 1)
    // =========================================================================
    if (zp_len == 1) {
        if (zp_type == DLP_U8) {
            SET_U8_F32(zp[0], ((const uint8_t*)zero_point)[0])
            SET_U8_F32(zp[1], ((const uint8_t*)zero_point)[0])
            SET_U8_F32(zp[2], ((const uint8_t*)zero_point)[0])
            SET_U8_F32(zp[3], ((const uint8_t*)zero_point)[0])
            SET_U8_F32(zp[4], ((const uint8_t*)zero_point)[0])
            SET_U8_F32(zp[5], ((const uint8_t*)zero_point)[0])
            SET_U8_F32(zp[6], ((const uint8_t*)zero_point)[0])
            SET_U8_F32(zp[7], ((const uint8_t*)zero_point)[0])
            SET_U8_F32(zp[8], ((const uint8_t*)zero_point)[0])
            SET_U8_F32(zp[9], ((const uint8_t*)zero_point)[0])
            SET_U8_F32(zp[10], ((const uint8_t*)zero_point)[0])
            SET_U8_F32(zp[11], ((const uint8_t*)zero_point)[0])
            SET_U8_F32(zp[12], ((const uint8_t*)zero_point)[0])
            SET_U8_F32(zp[13], ((const uint8_t*)zero_point)[0])
            SET_U8_F32(zp[14], ((const uint8_t*)zero_point)[0])
            SET_U8_F32(zp[15], ((const uint8_t*)zero_point)[0])
        } else if (zp_type == DLP_S8) {
            SET_S8_F32(zp[0], ((const int8_t*)zero_point)[0])
            SET_S8_F32(zp[1], ((const int8_t*)zero_point)[0])
            SET_S8_F32(zp[2], ((const int8_t*)zero_point)[0])
            SET_S8_F32(zp[3], ((const int8_t*)zero_point)[0])
            SET_S8_F32(zp[4], ((const int8_t*)zero_point)[0])
            SET_S8_F32(zp[5], ((const int8_t*)zero_point)[0])
            SET_S8_F32(zp[6], ((const int8_t*)zero_point)[0])
            SET_S8_F32(zp[7], ((const int8_t*)zero_point)[0])
            SET_S8_F32(zp[8], ((const int8_t*)zero_point)[0])
            SET_S8_F32(zp[9], ((const int8_t*)zero_point)[0])
            SET_S8_F32(zp[10], ((const int8_t*)zero_point)[0])
            SET_S8_F32(zp[11], ((const int8_t*)zero_point)[0])
            SET_S8_F32(zp[12], ((const int8_t*)zero_point)[0])
            SET_S8_F32(zp[13], ((const int8_t*)zero_point)[0])
            SET_S8_F32(zp[14], ((const int8_t*)zero_point)[0])
            SET_S8_F32(zp[15], ((const int8_t*)zero_point)[0])
        } else if (zp_type == DLP_S32) {
            SET_S32_F32(zp[0], ((const int32_t*)zero_point)[0])
            SET_S32_F32(zp[1], ((const int32_t*)zero_point)[0])
            SET_S32_F32(zp[2], ((const int32_t*)zero_point)[0])
            SET_S32_F32(zp[3], ((const int32_t*)zero_point)[0])
            SET_S32_F32(zp[4], ((const int32_t*)zero_point)[0])
            SET_S32_F32(zp[5], ((const int32_t*)zero_point)[0])
            SET_S32_F32(zp[6], ((const int32_t*)zero_point)[0])
            SET_S32_F32(zp[7], ((const int32_t*)zero_point)[0])
            SET_S32_F32(zp[8], ((const int32_t*)zero_point)[0])
            SET_S32_F32(zp[9], ((const int32_t*)zero_point)[0])
            SET_S32_F32(zp[10], ((const int32_t*)zero_point)[0])
            SET_S32_F32(zp[11], ((const int32_t*)zero_point)[0])
            SET_S32_F32(zp[12], ((const int32_t*)zero_point)[0])
            SET_S32_F32(zp[13], ((const int32_t*)zero_point)[0])
            SET_S32_F32(zp[14], ((const int32_t*)zero_point)[0])
            SET_S32_F32(zp[15], ((const int32_t*)zero_point)[0])
        } else if (zp_type == DLP_BF16) {
            SET_BF16_F32(zp[0], ((const bfloat16*)zero_point)[0])
            SET_BF16_F32(zp[1], ((const bfloat16*)zero_point)[0])
            SET_BF16_F32(zp[2], ((const bfloat16*)zero_point)[0])
            SET_BF16_F32(zp[3], ((const bfloat16*)zero_point)[0])
            SET_BF16_F32(zp[4], ((const bfloat16*)zero_point)[0])
            SET_BF16_F32(zp[5], ((const bfloat16*)zero_point)[0])
            SET_BF16_F32(zp[6], ((const bfloat16*)zero_point)[0])
            SET_BF16_F32(zp[7], ((const bfloat16*)zero_point)[0])
            SET_BF16_F32(zp[8], ((const bfloat16*)zero_point)[0])
            SET_BF16_F32(zp[9], ((const bfloat16*)zero_point)[0])
            SET_BF16_F32(zp[10], ((const bfloat16*)zero_point)[0])
            SET_BF16_F32(zp[11], ((const bfloat16*)zero_point)[0])
            SET_BF16_F32(zp[12], ((const bfloat16*)zero_point)[0])
            SET_BF16_F32(zp[13], ((const bfloat16*)zero_point)[0])
            SET_BF16_F32(zp[14], ((const bfloat16*)zero_point)[0])
            SET_BF16_F32(zp[15], ((const bfloat16*)zero_point)[0])
        } else {
            zp[0]  = _mm512_set1_ps(((const float*)zero_point)[0]);
            zp[1]  = _mm512_set1_ps(((const float*)zero_point)[0]);
            zp[2]  = _mm512_set1_ps(((const float*)zero_point)[0]);
            zp[3]  = _mm512_set1_ps(((const float*)zero_point)[0]);
            zp[4]  = _mm512_set1_ps(((const float*)zero_point)[0]);
            zp[5]  = _mm512_set1_ps(((const float*)zero_point)[0]);
            zp[6]  = _mm512_set1_ps(((const float*)zero_point)[0]);
            zp[7]  = _mm512_set1_ps(((const float*)zero_point)[0]);
            zp[8]  = _mm512_set1_ps(((const float*)zero_point)[0]);
            zp[9]  = _mm512_set1_ps(((const float*)zero_point)[0]);
            zp[10] = _mm512_set1_ps(((const float*)zero_point)[0]);
            zp[11] = _mm512_set1_ps(((const float*)zero_point)[0]);
            zp[12] = _mm512_set1_ps(((const float*)zero_point)[0]);
            zp[13] = _mm512_set1_ps(((const float*)zero_point)[0]);
            zp[14] = _mm512_set1_ps(((const float*)zero_point)[0]);
            zp[15] = _mm512_set1_ps(((const float*)zero_point)[0]);
        }
    }

    for (; (ic + MR - 1) < MC; ic += MR) {
        // broadcast scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                SET_U8_F32(sf[0], ((const uint8_t*)scale_factor)[sf_idx])
                SET_U8_F32(sf[1], ((const uint8_t*)scale_factor)[sf_idx + 1])
                SET_U8_F32(sf[2], ((const uint8_t*)scale_factor)[sf_idx + 2])
                SET_U8_F32(sf[3], ((const uint8_t*)scale_factor)[sf_idx + 3])
                SET_U8_F32(sf[4], ((const uint8_t*)scale_factor)[sf_idx + 4])
                SET_U8_F32(sf[5], ((const uint8_t*)scale_factor)[sf_idx + 5])
                SET_U8_F32(sf[6], ((const uint8_t*)scale_factor)[sf_idx + 6])
                SET_U8_F32(sf[7], ((const uint8_t*)scale_factor)[sf_idx + 7])
                SET_U8_F32(sf[8], ((const uint8_t*)scale_factor)[sf_idx + 8])
                SET_U8_F32(sf[9], ((const uint8_t*)scale_factor)[sf_idx + 9])
                SET_U8_F32(sf[10], ((const uint8_t*)scale_factor)[sf_idx + 10])
                SET_U8_F32(sf[11], ((const uint8_t*)scale_factor)[sf_idx + 11])
                SET_U8_F32(sf[12], ((const uint8_t*)scale_factor)[sf_idx + 12])
                SET_U8_F32(sf[13], ((const uint8_t*)scale_factor)[sf_idx + 13])
                SET_U8_F32(sf[14], ((const uint8_t*)scale_factor)[sf_idx + 14])
                SET_U8_F32(sf[15], ((const uint8_t*)scale_factor)[sf_idx + 15])
            } else if (sf_type == DLP_S8) {
                SET_S8_F32(sf[0], ((const int8_t*)scale_factor)[sf_idx])
                SET_S8_F32(sf[1], ((const int8_t*)scale_factor)[sf_idx + 1])
                SET_S8_F32(sf[2], ((const int8_t*)scale_factor)[sf_idx + 2])
                SET_S8_F32(sf[3], ((const int8_t*)scale_factor)[sf_idx + 3])
                SET_S8_F32(sf[4], ((const int8_t*)scale_factor)[sf_idx + 4])
                SET_S8_F32(sf[5], ((const int8_t*)scale_factor)[sf_idx + 5])
                SET_S8_F32(sf[6], ((const int8_t*)scale_factor)[sf_idx + 6])
                SET_S8_F32(sf[7], ((const int8_t*)scale_factor)[sf_idx + 7])
                SET_S8_F32(sf[8], ((const int8_t*)scale_factor)[sf_idx + 8])
                SET_S8_F32(sf[9], ((const int8_t*)scale_factor)[sf_idx + 9])
                SET_S8_F32(sf[10], ((const int8_t*)scale_factor)[sf_idx + 10])
                SET_S8_F32(sf[11], ((const int8_t*)scale_factor)[sf_idx + 11])
                SET_S8_F32(sf[12], ((const int8_t*)scale_factor)[sf_idx + 12])
                SET_S8_F32(sf[13], ((const int8_t*)scale_factor)[sf_idx + 13])
                SET_S8_F32(sf[14], ((const int8_t*)scale_factor)[sf_idx + 14])
                SET_S8_F32(sf[15], ((const int8_t*)scale_factor)[sf_idx + 15]);
            } else if (sf_type == DLP_BF16) {
                SET_BF16_F32(sf[0], ((const bfloat16*)scale_factor)[sf_idx])
                SET_BF16_F32(sf[1], ((const bfloat16*)scale_factor)[sf_idx + 1])
                SET_BF16_F32(sf[2], ((const bfloat16*)scale_factor)[sf_idx + 2])
                SET_BF16_F32(sf[3], ((const bfloat16*)scale_factor)[sf_idx + 3])
                SET_BF16_F32(sf[4], ((const bfloat16*)scale_factor)[sf_idx + 4])
                SET_BF16_F32(sf[5], ((const bfloat16*)scale_factor)[sf_idx + 5])
                SET_BF16_F32(sf[6], ((const bfloat16*)scale_factor)[sf_idx + 6])
                SET_BF16_F32(sf[7], ((const bfloat16*)scale_factor)[sf_idx + 7])
                SET_BF16_F32(sf[8], ((const bfloat16*)scale_factor)[sf_idx + 8])
                SET_BF16_F32(sf[9], ((const bfloat16*)scale_factor)[sf_idx + 9])
                SET_BF16_F32(sf[10],
                             ((const bfloat16*)scale_factor)[sf_idx + 10])
                SET_BF16_F32(sf[11],
                             ((const bfloat16*)scale_factor)[sf_idx + 11])
                SET_BF16_F32(sf[12],
                             ((const bfloat16*)scale_factor)[sf_idx + 12])
                SET_BF16_F32(sf[13],
                             ((const bfloat16*)scale_factor)[sf_idx + 13])
                SET_BF16_F32(sf[14],
                             ((const bfloat16*)scale_factor)[sf_idx + 14])
                SET_BF16_F32(sf[15],
                             ((const bfloat16*)scale_factor)[sf_idx + 15])
            } else if (sf_type == DLP_S32) {
                SET_S32_F32(sf[0], ((const int32_t*)scale_factor)[sf_idx])
                SET_S32_F32(sf[1], ((const int32_t*)scale_factor)[sf_idx + 1])
                SET_S32_F32(sf[2], ((const int32_t*)scale_factor)[sf_idx + 2])
                SET_S32_F32(sf[3], ((const int32_t*)scale_factor)[sf_idx + 3])
                SET_S32_F32(sf[4], ((const int32_t*)scale_factor)[sf_idx + 4])
                SET_S32_F32(sf[5], ((const int32_t*)scale_factor)[sf_idx + 5])
                SET_S32_F32(sf[6], ((const int32_t*)scale_factor)[sf_idx + 6])
                SET_S32_F32(sf[7], ((const int32_t*)scale_factor)[sf_idx + 7])
                SET_S32_F32(sf[8], ((const int32_t*)scale_factor)[sf_idx + 8])
                SET_S32_F32(sf[9], ((const int32_t*)scale_factor)[sf_idx + 9])
                SET_S32_F32(sf[10], ((const int32_t*)scale_factor)[sf_idx + 10])
                SET_S32_F32(sf[11], ((const int32_t*)scale_factor)[sf_idx + 11])
                SET_S32_F32(sf[12], ((const int32_t*)scale_factor)[sf_idx + 12])
                SET_S32_F32(sf[13], ((const int32_t*)scale_factor)[sf_idx + 13])
                SET_S32_F32(sf[14], ((const int32_t*)scale_factor)[sf_idx + 14])
                SET_S32_F32(sf[15], ((const int32_t*)scale_factor)[sf_idx + 15])
            } else {
                sf[0] = _mm512_set1_ps(((const float*)scale_factor)[sf_idx]);
                sf[1] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 1]);
                sf[2] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 2]);
                sf[3] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 3]);
                sf[4] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 4]);
                sf[5] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 5]);
                sf[6] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 6]);
                sf[7] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 7]);
                sf[8] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 8]);
                sf[9] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 9]);
                sf[10] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 10]);
                sf[11] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 11]);
                sf[12] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 12]);
                sf[13] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 13]);
                sf[14] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 14]);
                sf[15] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 15]);
            }
        }

        // broadcast zero point
        if (zp_len > 1) {
            zp_idx = ic_offset + ic;
            if (zp_type == DLP_U8) {
                SET_U8_F32(zp[0], ((const uint8_t*)zero_point)[zp_idx])
                SET_U8_F32(zp[1], ((const uint8_t*)zero_point)[zp_idx + 1])
                SET_U8_F32(zp[2], ((const uint8_t*)zero_point)[zp_idx + 2])
                SET_U8_F32(zp[3], ((const uint8_t*)zero_point)[zp_idx + 3])
                SET_U8_F32(zp[4], ((const uint8_t*)zero_point)[zp_idx + 4])
                SET_U8_F32(zp[5], ((const uint8_t*)zero_point)[zp_idx + 5])
                SET_U8_F32(zp[6], ((const uint8_t*)zero_point)[zp_idx + 6])
                SET_U8_F32(zp[7], ((const uint8_t*)zero_point)[zp_idx + 7])
                SET_U8_F32(zp[8], ((const uint8_t*)zero_point)[zp_idx + 8])
                SET_U8_F32(zp[9], ((const uint8_t*)zero_point)[zp_idx + 9])
                SET_U8_F32(zp[10], ((const uint8_t*)zero_point)[zp_idx + 10])
                SET_U8_F32(zp[11], ((const uint8_t*)zero_point)[zp_idx + 11])
                SET_U8_F32(zp[12], ((const uint8_t*)zero_point)[zp_idx + 12])
                SET_U8_F32(zp[13], ((const uint8_t*)zero_point)[zp_idx + 13])
                SET_U8_F32(zp[14], ((const uint8_t*)zero_point)[zp_idx + 14])
                SET_U8_F32(zp[15], ((const uint8_t*)zero_point)[zp_idx + 15])
            } else if (zp_type == DLP_S8) {
                SET_S8_F32(zp[0], ((const int8_t*)zero_point)[zp_idx])
                SET_S8_F32(zp[1], ((const int8_t*)zero_point)[zp_idx + 1])
                SET_S8_F32(zp[2], ((const int8_t*)zero_point)[zp_idx + 2])
                SET_S8_F32(zp[3], ((const int8_t*)zero_point)[zp_idx + 3])
                SET_S8_F32(zp[4], ((const int8_t*)zero_point)[zp_idx + 4])
                SET_S8_F32(zp[5], ((const int8_t*)zero_point)[zp_idx + 5])
                SET_S8_F32(zp[6], ((const int8_t*)zero_point)[zp_idx + 6])
                SET_S8_F32(zp[7], ((const int8_t*)zero_point)[zp_idx + 7])
                SET_S8_F32(zp[8], ((const int8_t*)zero_point)[zp_idx + 8])
                SET_S8_F32(zp[9], ((const int8_t*)zero_point)[zp_idx + 9])
                SET_S8_F32(zp[10], ((const int8_t*)zero_point)[zp_idx + 10])
                SET_S8_F32(zp[11], ((const int8_t*)zero_point)[zp_idx + 11])
                SET_S8_F32(zp[12], ((const int8_t*)zero_point)[zp_idx + 12])
                SET_S8_F32(zp[13], ((const int8_t*)zero_point)[zp_idx + 13])
                SET_S8_F32(zp[14], ((const int8_t*)zero_point)[zp_idx + 14])
                SET_S8_F32(zp[15], ((const int8_t*)zero_point)[zp_idx + 15])
            } else if (zp_type == DLP_S32) {
                SET_S32_F32(zp[0], ((const int32_t*)zero_point)[zp_idx])
                SET_S32_F32(zp[1], ((const int32_t*)zero_point)[zp_idx + 1])
                SET_S32_F32(zp[2], ((const int32_t*)zero_point)[zp_idx + 2])
                SET_S32_F32(zp[3], ((const int32_t*)zero_point)[zp_idx + 3])
                SET_S32_F32(zp[4], ((const int32_t*)zero_point)[zp_idx + 4])
                SET_S32_F32(zp[5], ((const int32_t*)zero_point)[zp_idx + 5])
                SET_S32_F32(zp[6], ((const int32_t*)zero_point)[zp_idx + 6])
                SET_S32_F32(zp[7], ((const int32_t*)zero_point)[zp_idx + 7])
                SET_S32_F32(zp[8], ((const int32_t*)zero_point)[zp_idx + 8])
                SET_S32_F32(zp[9], ((const int32_t*)zero_point)[zp_idx + 9])
                SET_S32_F32(zp[10], ((const int32_t*)zero_point)[zp_idx + 10])
                SET_S32_F32(zp[11], ((const int32_t*)zero_point)[zp_idx + 11])
                SET_S32_F32(zp[12], ((const int32_t*)zero_point)[zp_idx + 12])
                SET_S32_F32(zp[13], ((const int32_t*)zero_point)[zp_idx + 13])
                SET_S32_F32(zp[14], ((const int32_t*)zero_point)[zp_idx + 14])
                SET_S32_F32(zp[15], ((const int32_t*)zero_point)[zp_idx + 15])
            } else if (zp_type == DLP_BF16) {
                SET_BF16_F32(zp[0], ((const bfloat16*)zero_point)[zp_idx])
                SET_BF16_F32(zp[1], ((const bfloat16*)zero_point)[zp_idx + 1])
                SET_BF16_F32(zp[2], ((const bfloat16*)zero_point)[zp_idx + 2])
                SET_BF16_F32(zp[3], ((const bfloat16*)zero_point)[zp_idx + 3])
                SET_BF16_F32(zp[4], ((const bfloat16*)zero_point)[zp_idx + 4])
                SET_BF16_F32(zp[5], ((const bfloat16*)zero_point)[zp_idx + 5])
                SET_BF16_F32(zp[6], ((const bfloat16*)zero_point)[zp_idx + 6])
                SET_BF16_F32(zp[7], ((const bfloat16*)zero_point)[zp_idx + 7])
                SET_BF16_F32(zp[8], ((const bfloat16*)zero_point)[zp_idx + 8])
                SET_BF16_F32(zp[9], ((const bfloat16*)zero_point)[zp_idx + 9])
                SET_BF16_F32(zp[10], ((const bfloat16*)zero_point)[zp_idx + 10])
                SET_BF16_F32(zp[11], ((const bfloat16*)zero_point)[zp_idx + 11])
                SET_BF16_F32(zp[12], ((const bfloat16*)zero_point)[zp_idx + 12])
                SET_BF16_F32(zp[13], ((const bfloat16*)zero_point)[zp_idx + 13])
                SET_BF16_F32(zp[14], ((const bfloat16*)zero_point)[zp_idx + 14])
                SET_BF16_F32(zp[15], ((const bfloat16*)zero_point)[zp_idx + 15])
            } else {
                zp[0] = _mm512_set1_ps(((const float*)zero_point)[zp_idx]);
                zp[1] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 1]);
                zp[2] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 2]);
                zp[3] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 3]);
                zp[4] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 4]);
                zp[5] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 5]);
                zp[6] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 6]);
                zp[7] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 7]);
                zp[8] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 8]);
                zp[9] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 9]);
                zp[10] =
                    _mm512_set1_ps(((const float*)zero_point)[zp_idx + 10]);
                zp[11] =
                    _mm512_set1_ps(((const float*)zero_point)[zp_idx + 11]);
                zp[12] =
                    _mm512_set1_ps(((const float*)zero_point)[zp_idx + 12]);
                zp[13] =
                    _mm512_set1_ps(((const float*)zero_point)[zp_idx + 13]);
                zp[14] =
                    _mm512_set1_ps(((const float*)zero_point)[zp_idx + 14]);
                zp[15] =
                    _mm512_set1_ps(((const float*)zero_point)[zp_idx + 15]);
            }
        }

        for (kr = 0; (kr + 16 - 1) < KC; kr += 16) {
            // convert a bf16 to f32 and load in a_reg[0-15]
            LOAD_BF16_TO_F32(a_reg[0], a + ((ic + 0) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[1], a + ((ic + 1) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[2], a + ((ic + 2) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[3], a + ((ic + 3) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[4], a + ((ic + 4) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[5], a + ((ic + 5) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[6], a + ((ic + 6) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[7], a + ((ic + 7) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[8], a + ((ic + 8) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[9], a + ((ic + 9) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[10], a + ((ic + 10) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[11], a + ((ic + 11) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[12], a + ((ic + 12) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[13], a + ((ic + 13) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[14], a + ((ic + 14) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[15], a + ((ic + 15) * rs_a + kr * cs_a))
            // Apply asymmetric quantization: q = round(a * scale - zero_point)
            QUANT_ASYM(a_reg[0], sf[0], zp[0])
            QUANT_ASYM(a_reg[1], sf[1], zp[1])
            QUANT_ASYM(a_reg[2], sf[2], zp[2])
            QUANT_ASYM(a_reg[3], sf[3], zp[3])
            QUANT_ASYM(a_reg[4], sf[4], zp[4])
            QUANT_ASYM(a_reg[5], sf[5], zp[5])
            QUANT_ASYM(a_reg[6], sf[6], zp[6])
            QUANT_ASYM(a_reg[7], sf[7], zp[7])
            QUANT_ASYM(a_reg[8], sf[8], zp[8])
            QUANT_ASYM(a_reg[9], sf[9], zp[9])
            QUANT_ASYM(a_reg[10], sf[10], zp[10])
            QUANT_ASYM(a_reg[11], sf[11], zp[11])
            QUANT_ASYM(a_reg[12], sf[12], zp[12])
            QUANT_ASYM(a_reg[13], sf[13], zp[13])
            QUANT_ASYM(a_reg[14], sf[14], zp[14])
            QUANT_ASYM(a_reg[15], sf[15], zp[15])
            // Saturate to S8 and store.
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 0) * KC + kr), 0xFFFF,
                              a_reg[0]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 1) * KC + kr), 0xFFFF,
                              a_reg[1]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 2) * KC + kr), 0xFFFF,
                              a_reg[2]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 3) * KC + kr), 0xFFFF,
                              a_reg[3]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 4) * KC + kr), 0xFFFF,
                              a_reg[4]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 5) * KC + kr), 0xFFFF,
                              a_reg[5]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 6) * KC + kr), 0xFFFF,
                              a_reg[6]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 7) * KC + kr), 0xFFFF,
                              a_reg[7]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 8) * KC + kr), 0xFFFF,
                              a_reg[8]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 9) * KC + kr), 0xFFFF,
                              a_reg[9]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 10) * KC + kr), 0xFFFF,
                              a_reg[10]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 11) * KC + kr), 0xFFFF,
                              a_reg[11]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 12) * KC + kr), 0xFFFF,
                              a_reg[12]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 13) * KC + kr), 0xFFFF,
                              a_reg[13]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 14) * KC + kr), 0xFFFF,
                              a_reg[14]);
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 15) * KC + kr), 0xFFFF,
                              a_reg[15]);
        }

        if (kleft) {
            // load masked bf16 to f32
            LOAD_MASKED_BF16_TO_F32(a_reg[0], mask, a + (ic * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[1], mask,
                                    a + ((ic + 1) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[2], mask,
                                    a + ((ic + 2) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[3], mask,
                                    a + ((ic + 3) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[4], mask,
                                    a + ((ic + 4) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[5], mask,
                                    a + ((ic + 5) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[6], mask,
                                    a + ((ic + 6) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[7], mask,
                                    a + ((ic + 7) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[8], mask,
                                    a + ((ic + 8) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[9], mask,
                                    a + ((ic + 9) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[10], mask,
                                    a + ((ic + 10) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[11], mask,
                                    a + ((ic + 11) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[12], mask,
                                    a + ((ic + 12) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[13], mask,
                                    a + ((ic + 13) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[14], mask,
                                    a + ((ic + 14) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[15], mask,
                                    a + ((ic + 15) * rs_a + kr * cs_a))
            // asymmetric quantize
            QUANT_ASYM(a_reg[0], sf[0], zp[0])
            QUANT_ASYM(a_reg[1], sf[1], zp[1])
            QUANT_ASYM(a_reg[2], sf[2], zp[2])
            QUANT_ASYM(a_reg[3], sf[3], zp[3])
            QUANT_ASYM(a_reg[4], sf[4], zp[4])
            QUANT_ASYM(a_reg[5], sf[5], zp[5])
            QUANT_ASYM(a_reg[6], sf[6], zp[6])
            QUANT_ASYM(a_reg[7], sf[7], zp[7])
            QUANT_ASYM(a_reg[8], sf[8], zp[8])
            QUANT_ASYM(a_reg[9], sf[9], zp[9])
            QUANT_ASYM(a_reg[10], sf[10], zp[10])
            QUANT_ASYM(a_reg[11], sf[11], zp[11])
            QUANT_ASYM(a_reg[12], sf[12], zp[12])
            QUANT_ASYM(a_reg[13], sf[13], zp[13])
            QUANT_ASYM(a_reg[14], sf[14], zp[14])
            QUANT_ASYM(a_reg[15], sf[15], zp[15])
            // store quantized values as int8
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 0) * KC + kr), mask,
                              a_reg[0])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 1) * KC + kr), mask,
                              a_reg[1])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 2) * KC + kr), mask,
                              a_reg[2])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 3) * KC + kr), mask,
                              a_reg[3])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 4) * KC + kr), mask,
                              a_reg[4])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 5) * KC + kr), mask,
                              a_reg[5])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 6) * KC + kr), mask,
                              a_reg[6])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 7) * KC + kr), mask,
                              a_reg[7])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 8) * KC + kr), mask,
                              a_reg[8])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 9) * KC + kr), mask,
                              a_reg[9])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 10) * KC + kr), mask,
                              a_reg[10])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 11) * KC + kr), mask,
                              a_reg[11])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 12) * KC + kr), mask,
                              a_reg[12])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 13) * KC + kr), mask,
                              a_reg[13])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 14) * KC + kr), mask,
                              a_reg[14])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 15) * KC + kr), mask,
                              a_reg[15])
        }
    }

    for (; (ic + 8 - 1) < MC; ic += 8) {
        // broadcast scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                SET_U8_F32(sf[0], ((const uint8_t*)scale_factor)[sf_idx])
                SET_U8_F32(sf[1], ((const uint8_t*)scale_factor)[sf_idx + 1])
                SET_U8_F32(sf[2], ((const uint8_t*)scale_factor)[sf_idx + 2])
                SET_U8_F32(sf[3], ((const uint8_t*)scale_factor)[sf_idx + 3])
                SET_U8_F32(sf[4], ((const uint8_t*)scale_factor)[sf_idx + 4])
                SET_U8_F32(sf[5], ((const uint8_t*)scale_factor)[sf_idx + 5])
                SET_U8_F32(sf[6], ((const uint8_t*)scale_factor)[sf_idx + 6])
                SET_U8_F32(sf[7], ((const uint8_t*)scale_factor)[sf_idx + 7])
            } else if (sf_type == DLP_S8) {
                SET_S8_F32(sf[0], ((const int8_t*)scale_factor)[sf_idx])
                SET_S8_F32(sf[1], ((const int8_t*)scale_factor)[sf_idx + 1])
                SET_S8_F32(sf[2], ((const int8_t*)scale_factor)[sf_idx + 2])
                SET_S8_F32(sf[3], ((const int8_t*)scale_factor)[sf_idx + 3])
                SET_S8_F32(sf[4], ((const int8_t*)scale_factor)[sf_idx + 4])
                SET_S8_F32(sf[5], ((const int8_t*)scale_factor)[sf_idx + 5])
                SET_S8_F32(sf[6], ((const int8_t*)scale_factor)[sf_idx + 6])
                SET_S8_F32(sf[7], ((const int8_t*)scale_factor)[sf_idx + 7])
            } else if (sf_type == DLP_S32) {
                SET_S32_F32(sf[0], ((const int32_t*)scale_factor)[sf_idx])
                SET_S32_F32(sf[1], ((const int32_t*)scale_factor)[sf_idx + 1])
                SET_S32_F32(sf[2], ((const int32_t*)scale_factor)[sf_idx + 2])
                SET_S32_F32(sf[3], ((const int32_t*)scale_factor)[sf_idx + 3])
                SET_S32_F32(sf[4], ((const int32_t*)scale_factor)[sf_idx + 4])
                SET_S32_F32(sf[5], ((const int32_t*)scale_factor)[sf_idx + 5])
                SET_S32_F32(sf[6], ((const int32_t*)scale_factor)[sf_idx + 6])
                SET_S32_F32(sf[7], ((const int32_t*)scale_factor)[sf_idx + 7])
            } else if (sf_type == DLP_BF16) {
                SET_BF16_F32(sf[0], ((const bfloat16*)scale_factor)[sf_idx])
                SET_BF16_F32(sf[1], ((const bfloat16*)scale_factor)[sf_idx + 1])
                SET_BF16_F32(sf[2], ((const bfloat16*)scale_factor)[sf_idx + 2])
                SET_BF16_F32(sf[3], ((const bfloat16*)scale_factor)[sf_idx + 3])
                SET_BF16_F32(sf[4], ((const bfloat16*)scale_factor)[sf_idx + 4])
                SET_BF16_F32(sf[5], ((const bfloat16*)scale_factor)[sf_idx + 5])
                SET_BF16_F32(sf[6], ((const bfloat16*)scale_factor)[sf_idx + 6])
                SET_BF16_F32(sf[7], ((const bfloat16*)scale_factor)[sf_idx + 7])
            } else {
                sf[0] = _mm512_set1_ps(((const float*)scale_factor)[sf_idx]);
                sf[1] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 1]);
                sf[2] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 2]);
                sf[3] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 3]);
                sf[4] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 4]);
                sf[5] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 5]);
                sf[6] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 6]);
                sf[7] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 7]);
            }
        }

        // broadcast zero point
        if (zp_len > 1) {
            zp_idx = ic_offset + ic;
            if (zp_type == DLP_U8) {
                SET_U8_F32(zp[0], ((const uint8_t*)zero_point)[zp_idx])
                SET_U8_F32(zp[1], ((const uint8_t*)zero_point)[zp_idx + 1])
                SET_U8_F32(zp[2], ((const uint8_t*)zero_point)[zp_idx + 2])
                SET_U8_F32(zp[3], ((const uint8_t*)zero_point)[zp_idx + 3])
                SET_U8_F32(zp[4], ((const uint8_t*)zero_point)[zp_idx + 4])
                SET_U8_F32(zp[5], ((const uint8_t*)zero_point)[zp_idx + 5])
                SET_U8_F32(zp[6], ((const uint8_t*)zero_point)[zp_idx + 6])
                SET_U8_F32(zp[7], ((const uint8_t*)zero_point)[zp_idx + 7])
            } else if (zp_type == DLP_S8) {
                SET_S8_F32(zp[0], ((const int8_t*)zero_point)[zp_idx])
                SET_S8_F32(zp[1], ((const int8_t*)zero_point)[zp_idx + 1])
                SET_S8_F32(zp[2], ((const int8_t*)zero_point)[zp_idx + 2])
                SET_S8_F32(zp[3], ((const int8_t*)zero_point)[zp_idx + 3])
                SET_S8_F32(zp[4], ((const int8_t*)zero_point)[zp_idx + 4])
                SET_S8_F32(zp[5], ((const int8_t*)zero_point)[zp_idx + 5])
                SET_S8_F32(zp[6], ((const int8_t*)zero_point)[zp_idx + 6])
                SET_S8_F32(zp[7], ((const int8_t*)zero_point)[zp_idx + 7])
            } else if (zp_type == DLP_S32) {
                SET_S32_F32(zp[0], ((const int32_t*)zero_point)[zp_idx])
                SET_S32_F32(zp[1], ((const int32_t*)zero_point)[zp_idx + 1])
                SET_S32_F32(zp[2], ((const int32_t*)zero_point)[zp_idx + 2])
                SET_S32_F32(zp[3], ((const int32_t*)zero_point)[zp_idx + 3])
                SET_S32_F32(zp[4], ((const int32_t*)zero_point)[zp_idx + 4])
                SET_S32_F32(zp[5], ((const int32_t*)zero_point)[zp_idx + 5])
                SET_S32_F32(zp[6], ((const int32_t*)zero_point)[zp_idx + 6])
                SET_S32_F32(zp[7], ((const int32_t*)zero_point)[zp_idx + 7])
            } else if (zp_type == DLP_BF16) {
                SET_BF16_F32(zp[0], ((const bfloat16*)zero_point)[zp_idx])
                SET_BF16_F32(zp[1], ((const bfloat16*)zero_point)[zp_idx + 1])
                SET_BF16_F32(zp[2], ((const bfloat16*)zero_point)[zp_idx + 2])
                SET_BF16_F32(zp[3], ((const bfloat16*)zero_point)[zp_idx + 3])
                SET_BF16_F32(zp[4], ((const bfloat16*)zero_point)[zp_idx + 4])
                SET_BF16_F32(zp[5], ((const bfloat16*)zero_point)[zp_idx + 5])
                SET_BF16_F32(zp[6], ((const bfloat16*)zero_point)[zp_idx + 6])
                SET_BF16_F32(zp[7], ((const bfloat16*)zero_point)[zp_idx + 7])
            } else {
                zp[0] = _mm512_set1_ps(((const float*)zero_point)[zp_idx]);
                zp[1] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 1]);
                zp[2] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 2]);
                zp[3] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 3]);
                zp[4] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 4]);
                zp[5] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 5]);
                zp[6] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 6]);
                zp[7] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 7]);
            }
        }

        for (kr = 0; (kr + 16 - 1) < KC; kr += 16) {
            // convert a bf16 to f32 and load in a_reg[0-7]
            LOAD_BF16_TO_F32(a_reg[0], a + ((ic + 0) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[1], a + ((ic + 1) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[2], a + ((ic + 2) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[3], a + ((ic + 3) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[4], a + ((ic + 4) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[5], a + ((ic + 5) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[6], a + ((ic + 6) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[7], a + ((ic + 7) * rs_a + kr * cs_a))
            // quant_asym a_reg[0-7]
            QUANT_ASYM(a_reg[0], sf[0], zp[0])
            QUANT_ASYM(a_reg[1], sf[1], zp[1])
            QUANT_ASYM(a_reg[2], sf[2], zp[2])
            QUANT_ASYM(a_reg[3], sf[3], zp[3])
            QUANT_ASYM(a_reg[4], sf[4], zp[4])
            QUANT_ASYM(a_reg[5], sf[5], zp[5])
            QUANT_ASYM(a_reg[6], sf[6], zp[6])
            QUANT_ASYM(a_reg[7], sf[7], zp[7])
            // convert to int32 and store
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 0) * KC + kr), 0xFFFF,
                              a_reg[0])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 1) * KC + kr), 0xFFFF,
                              a_reg[1])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 2) * KC + kr), 0xFFFF,
                              a_reg[2])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 3) * KC + kr), 0xFFFF,
                              a_reg[3])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 4) * KC + kr), 0xFFFF,
                              a_reg[4])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 5) * KC + kr), 0xFFFF,
                              a_reg[5])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 6) * KC + kr), 0xFFFF,
                              a_reg[6])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 7) * KC + kr), 0xFFFF,
                              a_reg[7])
        }

        if (kleft) {
            // load masked bf16 to f32
            LOAD_MASKED_BF16_TO_F32(a_reg[0], mask, a + (ic * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[1], mask,
                                    a + ((ic + 1) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[2], mask,
                                    a + ((ic + 2) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[3], mask,
                                    a + ((ic + 3) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[4], mask,
                                    a + ((ic + 4) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[5], mask,
                                    a + ((ic + 5) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[6], mask,
                                    a + ((ic + 6) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[7], mask,
                                    a + ((ic + 7) * rs_a + kr * cs_a))
            // asymmetric quantize
            QUANT_ASYM(a_reg[0], sf[0], zp[0])
            QUANT_ASYM(a_reg[1], sf[1], zp[1])
            QUANT_ASYM(a_reg[2], sf[2], zp[2])
            QUANT_ASYM(a_reg[3], sf[3], zp[3])
            QUANT_ASYM(a_reg[4], sf[4], zp[4])
            QUANT_ASYM(a_reg[5], sf[5], zp[5])
            QUANT_ASYM(a_reg[6], sf[6], zp[6])
            QUANT_ASYM(a_reg[7], sf[7], zp[7])
            // store quantized values as int8
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 0) * KC + kr), mask,
                              a_reg[0])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 1) * KC + kr), mask,
                              a_reg[1])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 2) * KC + kr), mask,
                              a_reg[2])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 3) * KC + kr), mask,
                              a_reg[3])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 4) * KC + kr), mask,
                              a_reg[4])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 5) * KC + kr), mask,
                              a_reg[5])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 6) * KC + kr), mask,
                              a_reg[6])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 7) * KC + kr), mask,
                              a_reg[7])
        }
    }

    for (; (ic + 4 - 1) < MC; ic += 4) {
        // broadcast scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                SET_U8_F32(sf[0], ((const uint8_t*)scale_factor)[sf_idx])
                SET_U8_F32(sf[1], ((const uint8_t*)scale_factor)[sf_idx + 1])
                SET_U8_F32(sf[2], ((const uint8_t*)scale_factor)[sf_idx + 2])
                SET_U8_F32(sf[3], ((const uint8_t*)scale_factor)[sf_idx + 3])
            } else if (sf_type == DLP_S8) {
                SET_S8_F32(sf[0], ((const int8_t*)scale_factor)[sf_idx])
                SET_S8_F32(sf[1], ((const int8_t*)scale_factor)[sf_idx + 1])
                SET_S8_F32(sf[2], ((const int8_t*)scale_factor)[sf_idx + 2])
                SET_S8_F32(sf[3], ((const int8_t*)scale_factor)[sf_idx + 3])
            } else if (sf_type == DLP_S32) {
                SET_S32_F32(sf[0], ((const int32_t*)scale_factor)[sf_idx])
                SET_S32_F32(sf[1], ((const int32_t*)scale_factor)[sf_idx + 1])
                SET_S32_F32(sf[2], ((const int32_t*)scale_factor)[sf_idx + 2])
                SET_S32_F32(sf[3], ((const int32_t*)scale_factor)[sf_idx + 3])
            } else if (sf_type == DLP_BF16) {
                SET_BF16_F32(sf[0], ((const bfloat16*)scale_factor)[sf_idx])
                SET_BF16_F32(sf[1], ((const bfloat16*)scale_factor)[sf_idx + 1])
                SET_BF16_F32(sf[2], ((const bfloat16*)scale_factor)[sf_idx + 2])
                SET_BF16_F32(sf[3], ((const bfloat16*)scale_factor)[sf_idx + 3])
            } else {
                sf[0] = _mm512_set1_ps(((const float*)scale_factor)[sf_idx]);
                sf[1] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 1]);
                sf[2] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 2]);
                sf[3] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 3]);
            }
        }

        // broadcast zero point
        if (zp_len > 1) {
            zp_idx = ic_offset + ic;
            if (zp_type == DLP_U8) {
                SET_U8_F32(zp[0], ((const uint8_t*)zero_point)[zp_idx])
                SET_U8_F32(zp[1], ((const uint8_t*)zero_point)[zp_idx + 1])
                SET_U8_F32(zp[2], ((const uint8_t*)zero_point)[zp_idx + 2])
                SET_U8_F32(zp[3], ((const uint8_t*)zero_point)[zp_idx + 3])
            } else if (zp_type == DLP_S8) {
                SET_S8_F32(zp[0], ((const int8_t*)zero_point)[zp_idx])
                SET_S8_F32(zp[1], ((const int8_t*)zero_point)[zp_idx + 1])
                SET_S8_F32(zp[2], ((const int8_t*)zero_point)[zp_idx + 2])
                SET_S8_F32(zp[3], ((const int8_t*)zero_point)[zp_idx + 3])
            } else if (zp_type == DLP_S32) {
                SET_S32_F32(zp[0], ((const int32_t*)zero_point)[zp_idx])
                SET_S32_F32(zp[1], ((const int32_t*)zero_point)[zp_idx + 1])
                SET_S32_F32(zp[2], ((const int32_t*)zero_point)[zp_idx + 2])
                SET_S32_F32(zp[3], ((const int32_t*)zero_point)[zp_idx + 3])
            } else if (zp_type == DLP_BF16) {
                SET_BF16_F32(zp[0], ((const bfloat16*)zero_point)[zp_idx])
                SET_BF16_F32(zp[1], ((const bfloat16*)zero_point)[zp_idx + 1])
                SET_BF16_F32(zp[2], ((const bfloat16*)zero_point)[zp_idx + 2])
                SET_BF16_F32(zp[3], ((const bfloat16*)zero_point)[zp_idx + 3])
            } else {
                zp[0] = _mm512_set1_ps(((const float*)zero_point)[zp_idx]);
                zp[1] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 1]);
                zp[2] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 2]);
                zp[3] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 3]);
            }
        }

        for (kr = 0; (kr + 16 - 1) < KC; kr += 16) {
            // convert a bf16 to f32 and load in a_reg[0-3]
            LOAD_BF16_TO_F32(a_reg[0], a + ((ic + 0) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[1], a + ((ic + 1) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[2], a + ((ic + 2) * rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[3], a + ((ic + 3) * rs_a + kr * cs_a))
            // quant_asym a_reg[0-3]
            QUANT_ASYM(a_reg[0], sf[0], zp[0])
            QUANT_ASYM(a_reg[1], sf[1], zp[1])
            QUANT_ASYM(a_reg[2], sf[2], zp[2])
            QUANT_ASYM(a_reg[3], sf[3], zp[3])
            // convert to int32 and store
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 0) * KC + kr), 0xFFFF,
                              a_reg[0])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 1) * KC + kr), 0xFFFF,
                              a_reg[1])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 2) * KC + kr), 0xFFFF,
                              a_reg[2])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 3) * KC + kr), 0xFFFF,
                              a_reg[3])
        }

        if (kleft) {
            // load masked bf16 to f32
            LOAD_MASKED_BF16_TO_F32(a_reg[0], mask, a + (ic * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[1], mask,
                                    a + ((ic + 1) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[2], mask,
                                    a + ((ic + 2) * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[3], mask,
                                    a + ((ic + 3) * rs_a + kr * cs_a))
            // asymmetric quantize
            QUANT_ASYM(a_reg[0], sf[0], zp[0])
            QUANT_ASYM(a_reg[1], sf[1], zp[1])
            QUANT_ASYM(a_reg[2], sf[2], zp[2])
            QUANT_ASYM(a_reg[3], sf[3], zp[3])
            // store quantized values as int8
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 0) * KC + kr), mask,
                              a_reg[0])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 1) * KC + kr), mask,
                              a_reg[1])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 2) * KC + kr), mask,
                              a_reg[2])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 3) * KC + kr), mask,
                              a_reg[3])
        }
    }

    for (; (ic + 2 - 1) < MC; ic += 2) {
        // broadcast scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                SET_U8_F32(sf[0], ((const uint8_t*)scale_factor)[sf_idx])
                SET_U8_F32(sf[1], ((const uint8_t*)scale_factor)[sf_idx + 1])
            } else if (sf_type == DLP_S8) {
                SET_S8_F32(sf[0], ((const int8_t*)scale_factor)[sf_idx])
                SET_S8_F32(sf[1], ((const int8_t*)scale_factor)[sf_idx + 1])
            } else if (sf_type == DLP_S32) {
                SET_S32_F32(sf[0], ((const int32_t*)scale_factor)[sf_idx])
                SET_S32_F32(sf[1], ((const int32_t*)scale_factor)[sf_idx + 1])
            } else if (sf_type == DLP_BF16) {
                SET_BF16_F32(sf[0], ((const bfloat16*)scale_factor)[sf_idx])
                SET_BF16_F32(sf[1], ((const bfloat16*)scale_factor)[sf_idx + 1])
            } else {
                sf[0] = _mm512_set1_ps(((const float*)scale_factor)[sf_idx]);
                sf[1] =
                    _mm512_set1_ps(((const float*)scale_factor)[sf_idx + 1]);
            }
        }

        // broadcast zero point
        if (zp_len > 1) {
            zp_idx = ic_offset + ic;
            if (zp_type == DLP_U8) {
                SET_U8_F32(zp[0], ((const uint8_t*)zero_point)[zp_idx])
                SET_U8_F32(zp[1], ((const uint8_t*)zero_point)[zp_idx + 1])
            } else if (zp_type == DLP_S8) {
                SET_S8_F32(zp[0], ((const int8_t*)zero_point)[zp_idx])
                SET_S8_F32(zp[1], ((const int8_t*)zero_point)[zp_idx + 1])
            } else if (zp_type == DLP_S32) {
                SET_S32_F32(zp[0], ((const int32_t*)zero_point)[zp_idx])
                SET_S32_F32(zp[1], ((const int32_t*)zero_point)[zp_idx + 1])
            } else if (zp_type == DLP_BF16) {
                SET_BF16_F32(zp[0], ((const bfloat16*)zero_point)[zp_idx])
                SET_BF16_F32(zp[1], ((const bfloat16*)zero_point)[zp_idx + 1])
            } else {
                zp[0] = _mm512_set1_ps(((const float*)zero_point)[zp_idx]);
                zp[1] = _mm512_set1_ps(((const float*)zero_point)[zp_idx + 1]);
            }
        }

        for (kr = 0; (kr + 16 - 1) < KC; kr += 16) {
            // convert a bf16 to f32 and load in a_reg[0-3]
            LOAD_BF16_TO_F32(a_reg[0], a + ((ic)*rs_a + kr * cs_a))
            LOAD_BF16_TO_F32(a_reg[1], a + ((ic + 1) * rs_a + kr * cs_a))
            // quant_asym a_reg[0-3]
            QUANT_ASYM(a_reg[0], sf[0], zp[0])
            QUANT_ASYM(a_reg[1], sf[1], zp[1])
            // convert to int32 and store
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 0) * KC + kr), 0xFFFF,
                              a_reg[0])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 1) * KC + kr), 0xFFFF,
                              a_reg[1])
        }

        if (kleft) {
            // load masked bf16 to f32
            LOAD_MASKED_BF16_TO_F32(a_reg[0], mask, a + (ic * rs_a + kr * cs_a))
            LOAD_MASKED_BF16_TO_F32(a_reg[1], mask,
                                    a + ((ic + 1) * rs_a + kr * cs_a))
            // asymmetric quantize
            QUANT_ASYM(a_reg[0], sf[0], zp[0])
            QUANT_ASYM(a_reg[1], sf[1], zp[1])
            // store quantized values as int8
            STORE_MASKED_INT8(quant_a_buffer + ((ic)*KC + kr), mask, a_reg[0])
            STORE_MASKED_INT8(quant_a_buffer + ((ic + 1) * KC + kr), mask,
                              a_reg[1])
        }
    }

    for (; ic < MC; ic += 1) {
        // scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                SET_U8_F32(sf[0], ((const uint8_t*)scale_factor)[sf_idx])
            } else if (sf_type == DLP_S8) {
                SET_S8_F32(sf[0], ((const int8_t*)scale_factor)[sf_idx])
            } else if (sf_type == DLP_S32) {
                SET_S32_F32(sf[0], ((const int32_t*)scale_factor)[sf_idx])
            } else if (sf_type == DLP_BF16) {
                SET_BF16_F32(sf[0], ((const bfloat16*)scale_factor)[sf_idx])
            } else {
                sf[0] = _mm512_set1_ps(((const float*)scale_factor)[sf_idx]);
            }
        }

        // zero point
        if (zp_len > 1) {
            zp_idx = ic_offset + ic;
            if (zp_type == DLP_U8) {
                SET_U8_F32(zp[0], ((const uint8_t*)zero_point)[zp_idx])
            } else if (zp_type == DLP_S8) {
                SET_S8_F32(zp[0], ((const int8_t*)zero_point)[zp_idx])
            } else if (zp_type == DLP_S32) {
                SET_S32_F32(zp[0], ((const int32_t*)zero_point)[zp_idx])
            } else if (zp_type == DLP_BF16) {
                SET_BF16_F32(zp[0], ((const bfloat16*)zero_point)[zp_idx])
            } else {
                zp[0] = _mm512_set1_ps(((const float*)zero_point)[zp_idx]);
            }
        }

        for (kr = 0; (kr + 16 - 1) < KC; kr += 16) {
            // convert a bf16 to f32 and load in a_reg[0]
            LOAD_BF16_TO_F32(a_reg[0], a + (ic * rs_a + kr * cs_a))
            // asymmetric quantize
            QUANT_ASYM(a_reg[0], sf[0], zp[0])
            // store quantized values as int8
            STORE_MASKED_INT8(quant_a_buffer + ((ic)*KC + kr), 0xFFFF, a_reg[0])
        }

        if (kleft) {
            // load masked bf16 to f32
            LOAD_MASKED_BF16_TO_F32(a_reg[0], mask, a + (ic * rs_a + kr * cs_a))
            // asymmetric quantize
            QUANT_ASYM(a_reg[0], sf[0], zp[0]);
            // store quantized values as int8
            STORE_MASKED_INT8(quant_a_buffer + (ic * KC + kr), mask, a_reg[0])
        }
    }
}
