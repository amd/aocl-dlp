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

/*
 * FP16 AVX-512 Optimized Pack A Kernel (MR=32 configuration)
 *
 * Key characteristics:
 * - No k-padding required (packing_factor = 1, unlike BF16's factor of 2)
 * - Row-major: Simple vectorized copy (no transpose needed)
 * - Column-major: 32x32 transpose (matching PackB implementation)
 * - MR=32 uses 32 ZMM registers (32 FP16 elements each)
 * - Output layout: M-MAJOR (rs_p=KC, cs_p=1)
 *
 * Critical FP16 vs BF16 differences:
 * - No k-padding: kc0 = kc0 (not kc0 += (kc0 & 1))
 * - Column stride: cs_p = 1 (not cs_p = 2)
 * - Transpose size: 32x32 (not 16x16)
 */

#include "classic/aocl_fp16_type.h"
#include "fp16fp16fp16/lpgemm_reorder_fp16.h" /* For reference fallback */
#include "kernels/dlp_kernels.h"
#include <immintrin.h>

/* 32x32 transpose macro for FP16 (reused from PackB implementation)
 * Transposes 32 columns x 32 k-elements
 * Requires: a_reg[32] and b_reg[32] as __m512i arrays
 */
#define TRANSPOSE_32x32_FP16_AVX512                                            \
    /* Step 1: Unpack 16-bit pairs (within 128-bit lanes) */                   \
    b_reg[0]  = _mm512_unpacklo_epi16(a_reg[0], a_reg[1]);                     \
    b_reg[1]  = _mm512_unpackhi_epi16(a_reg[0], a_reg[1]);                     \
    b_reg[2]  = _mm512_unpacklo_epi16(a_reg[2], a_reg[3]);                     \
    b_reg[3]  = _mm512_unpackhi_epi16(a_reg[2], a_reg[3]);                     \
    b_reg[4]  = _mm512_unpacklo_epi16(a_reg[4], a_reg[5]);                     \
    b_reg[5]  = _mm512_unpackhi_epi16(a_reg[4], a_reg[5]);                     \
    b_reg[6]  = _mm512_unpacklo_epi16(a_reg[6], a_reg[7]);                     \
    b_reg[7]  = _mm512_unpackhi_epi16(a_reg[6], a_reg[7]);                     \
    b_reg[8]  = _mm512_unpacklo_epi16(a_reg[8], a_reg[9]);                     \
    b_reg[9]  = _mm512_unpackhi_epi16(a_reg[8], a_reg[9]);                     \
    b_reg[10] = _mm512_unpacklo_epi16(a_reg[10], a_reg[11]);                   \
    b_reg[11] = _mm512_unpackhi_epi16(a_reg[10], a_reg[11]);                   \
    b_reg[12] = _mm512_unpacklo_epi16(a_reg[12], a_reg[13]);                   \
    b_reg[13] = _mm512_unpackhi_epi16(a_reg[12], a_reg[13]);                   \
    b_reg[14] = _mm512_unpacklo_epi16(a_reg[14], a_reg[15]);                   \
    b_reg[15] = _mm512_unpackhi_epi16(a_reg[14], a_reg[15]);                   \
    b_reg[16] = _mm512_unpacklo_epi16(a_reg[16], a_reg[17]);                   \
    b_reg[17] = _mm512_unpackhi_epi16(a_reg[16], a_reg[17]);                   \
    b_reg[18] = _mm512_unpacklo_epi16(a_reg[18], a_reg[19]);                   \
    b_reg[19] = _mm512_unpackhi_epi16(a_reg[18], a_reg[19]);                   \
    b_reg[20] = _mm512_unpacklo_epi16(a_reg[20], a_reg[21]);                   \
    b_reg[21] = _mm512_unpackhi_epi16(a_reg[20], a_reg[21]);                   \
    b_reg[22] = _mm512_unpacklo_epi16(a_reg[22], a_reg[23]);                   \
    b_reg[23] = _mm512_unpackhi_epi16(a_reg[22], a_reg[23]);                   \
    b_reg[24] = _mm512_unpacklo_epi16(a_reg[24], a_reg[25]);                   \
    b_reg[25] = _mm512_unpackhi_epi16(a_reg[24], a_reg[25]);                   \
    b_reg[26] = _mm512_unpacklo_epi16(a_reg[26], a_reg[27]);                   \
    b_reg[27] = _mm512_unpackhi_epi16(a_reg[26], a_reg[27]);                   \
    b_reg[28] = _mm512_unpacklo_epi16(a_reg[28], a_reg[29]);                   \
    b_reg[29] = _mm512_unpackhi_epi16(a_reg[28], a_reg[29]);                   \
    b_reg[30] = _mm512_unpacklo_epi16(a_reg[30], a_reg[31]);                   \
    b_reg[31] = _mm512_unpackhi_epi16(a_reg[30], a_reg[31]);                   \
                                                                               \
    /* Step 2: Unpack 32-bit pairs */                                          \
    a_reg[0]  = _mm512_unpacklo_epi32(b_reg[0], b_reg[2]);                     \
    a_reg[1]  = _mm512_unpackhi_epi32(b_reg[0], b_reg[2]);                     \
    a_reg[2]  = _mm512_unpacklo_epi32(b_reg[1], b_reg[3]);                     \
    a_reg[3]  = _mm512_unpackhi_epi32(b_reg[1], b_reg[3]);                     \
    a_reg[4]  = _mm512_unpacklo_epi32(b_reg[4], b_reg[6]);                     \
    a_reg[5]  = _mm512_unpackhi_epi32(b_reg[4], b_reg[6]);                     \
    a_reg[6]  = _mm512_unpacklo_epi32(b_reg[5], b_reg[7]);                     \
    a_reg[7]  = _mm512_unpackhi_epi32(b_reg[5], b_reg[7]);                     \
    a_reg[8]  = _mm512_unpacklo_epi32(b_reg[8], b_reg[10]);                    \
    a_reg[9]  = _mm512_unpackhi_epi32(b_reg[8], b_reg[10]);                    \
    a_reg[10] = _mm512_unpacklo_epi32(b_reg[9], b_reg[11]);                    \
    a_reg[11] = _mm512_unpackhi_epi32(b_reg[9], b_reg[11]);                    \
    a_reg[12] = _mm512_unpacklo_epi32(b_reg[12], b_reg[14]);                   \
    a_reg[13] = _mm512_unpackhi_epi32(b_reg[12], b_reg[14]);                   \
    a_reg[14] = _mm512_unpacklo_epi32(b_reg[13], b_reg[15]);                   \
    a_reg[15] = _mm512_unpackhi_epi32(b_reg[13], b_reg[15]);                   \
    a_reg[16] = _mm512_unpacklo_epi32(b_reg[16], b_reg[18]);                   \
    a_reg[17] = _mm512_unpackhi_epi32(b_reg[16], b_reg[18]);                   \
    a_reg[18] = _mm512_unpacklo_epi32(b_reg[17], b_reg[19]);                   \
    a_reg[19] = _mm512_unpackhi_epi32(b_reg[17], b_reg[19]);                   \
    a_reg[20] = _mm512_unpacklo_epi32(b_reg[20], b_reg[22]);                   \
    a_reg[21] = _mm512_unpackhi_epi32(b_reg[20], b_reg[22]);                   \
    a_reg[22] = _mm512_unpacklo_epi32(b_reg[21], b_reg[23]);                   \
    a_reg[23] = _mm512_unpackhi_epi32(b_reg[21], b_reg[23]);                   \
    a_reg[24] = _mm512_unpacklo_epi32(b_reg[24], b_reg[26]);                   \
    a_reg[25] = _mm512_unpackhi_epi32(b_reg[24], b_reg[26]);                   \
    a_reg[26] = _mm512_unpacklo_epi32(b_reg[25], b_reg[27]);                   \
    a_reg[27] = _mm512_unpackhi_epi32(b_reg[25], b_reg[27]);                   \
    a_reg[28] = _mm512_unpacklo_epi32(b_reg[28], b_reg[30]);                   \
    a_reg[29] = _mm512_unpackhi_epi32(b_reg[28], b_reg[30]);                   \
    a_reg[30] = _mm512_unpacklo_epi32(b_reg[29], b_reg[31]);                   \
    a_reg[31] = _mm512_unpackhi_epi32(b_reg[29], b_reg[31]);                   \
                                                                               \
    /* Step 3: Unpack 64-bit pairs */                                          \
    b_reg[0]  = _mm512_unpacklo_epi64(a_reg[0], a_reg[4]);                     \
    b_reg[1]  = _mm512_unpackhi_epi64(a_reg[0], a_reg[4]);                     \
    b_reg[2]  = _mm512_unpacklo_epi64(a_reg[1], a_reg[5]);                     \
    b_reg[3]  = _mm512_unpackhi_epi64(a_reg[1], a_reg[5]);                     \
    b_reg[4]  = _mm512_unpacklo_epi64(a_reg[2], a_reg[6]);                     \
    b_reg[5]  = _mm512_unpackhi_epi64(a_reg[2], a_reg[6]);                     \
    b_reg[6]  = _mm512_unpacklo_epi64(a_reg[3], a_reg[7]);                     \
    b_reg[7]  = _mm512_unpackhi_epi64(a_reg[3], a_reg[7]);                     \
    b_reg[8]  = _mm512_unpacklo_epi64(a_reg[8], a_reg[12]);                    \
    b_reg[9]  = _mm512_unpackhi_epi64(a_reg[8], a_reg[12]);                    \
    b_reg[10] = _mm512_unpacklo_epi64(a_reg[9], a_reg[13]);                    \
    b_reg[11] = _mm512_unpackhi_epi64(a_reg[9], a_reg[13]);                    \
    b_reg[12] = _mm512_unpacklo_epi64(a_reg[10], a_reg[14]);                   \
    b_reg[13] = _mm512_unpackhi_epi64(a_reg[10], a_reg[14]);                   \
    b_reg[14] = _mm512_unpacklo_epi64(a_reg[11], a_reg[15]);                   \
    b_reg[15] = _mm512_unpackhi_epi64(a_reg[11], a_reg[15]);                   \
    b_reg[16] = _mm512_unpacklo_epi64(a_reg[16], a_reg[20]);                   \
    b_reg[17] = _mm512_unpackhi_epi64(a_reg[16], a_reg[20]);                   \
    b_reg[18] = _mm512_unpacklo_epi64(a_reg[17], a_reg[21]);                   \
    b_reg[19] = _mm512_unpackhi_epi64(a_reg[17], a_reg[21]);                   \
    b_reg[20] = _mm512_unpacklo_epi64(a_reg[18], a_reg[22]);                   \
    b_reg[21] = _mm512_unpackhi_epi64(a_reg[18], a_reg[22]);                   \
    b_reg[22] = _mm512_unpacklo_epi64(a_reg[19], a_reg[23]);                   \
    b_reg[23] = _mm512_unpackhi_epi64(a_reg[19], a_reg[23]);                   \
    b_reg[24] = _mm512_unpacklo_epi64(a_reg[24], a_reg[28]);                   \
    b_reg[25] = _mm512_unpackhi_epi64(a_reg[24], a_reg[28]);                   \
    b_reg[26] = _mm512_unpacklo_epi64(a_reg[25], a_reg[29]);                   \
    b_reg[27] = _mm512_unpackhi_epi64(a_reg[25], a_reg[29]);                   \
    b_reg[28] = _mm512_unpacklo_epi64(a_reg[26], a_reg[30]);                   \
    b_reg[29] = _mm512_unpackhi_epi64(a_reg[26], a_reg[30]);                   \
    b_reg[30] = _mm512_unpacklo_epi64(a_reg[27], a_reg[31]);                   \
    b_reg[31] = _mm512_unpackhi_epi64(a_reg[27], a_reg[31]);                   \
                                                                               \
    /* Step 4: First cross-lane shuffle */                                     \
    a_reg[0]  = _mm512_shuffle_i64x2(b_reg[0], b_reg[8], 0x44);                \
    a_reg[1]  = _mm512_shuffle_i64x2(b_reg[1], b_reg[9], 0x44);                \
    a_reg[2]  = _mm512_shuffle_i64x2(b_reg[2], b_reg[10], 0x44);               \
    a_reg[3]  = _mm512_shuffle_i64x2(b_reg[3], b_reg[11], 0x44);               \
    a_reg[4]  = _mm512_shuffle_i64x2(b_reg[4], b_reg[12], 0x44);               \
    a_reg[5]  = _mm512_shuffle_i64x2(b_reg[5], b_reg[13], 0x44);               \
    a_reg[6]  = _mm512_shuffle_i64x2(b_reg[6], b_reg[14], 0x44);               \
    a_reg[7]  = _mm512_shuffle_i64x2(b_reg[7], b_reg[15], 0x44);               \
    a_reg[8]  = _mm512_shuffle_i64x2(b_reg[0], b_reg[8], 0xEE);                \
    a_reg[9]  = _mm512_shuffle_i64x2(b_reg[1], b_reg[9], 0xEE);                \
    a_reg[10] = _mm512_shuffle_i64x2(b_reg[2], b_reg[10], 0xEE);               \
    a_reg[11] = _mm512_shuffle_i64x2(b_reg[3], b_reg[11], 0xEE);               \
    a_reg[12] = _mm512_shuffle_i64x2(b_reg[4], b_reg[12], 0xEE);               \
    a_reg[13] = _mm512_shuffle_i64x2(b_reg[5], b_reg[13], 0xEE);               \
    a_reg[14] = _mm512_shuffle_i64x2(b_reg[6], b_reg[14], 0xEE);               \
    a_reg[15] = _mm512_shuffle_i64x2(b_reg[7], b_reg[15], 0xEE);               \
    a_reg[16] = _mm512_shuffle_i64x2(b_reg[16], b_reg[24], 0x44);              \
    a_reg[17] = _mm512_shuffle_i64x2(b_reg[17], b_reg[25], 0x44);              \
    a_reg[18] = _mm512_shuffle_i64x2(b_reg[18], b_reg[26], 0x44);              \
    a_reg[19] = _mm512_shuffle_i64x2(b_reg[19], b_reg[27], 0x44);              \
    a_reg[20] = _mm512_shuffle_i64x2(b_reg[20], b_reg[28], 0x44);              \
    a_reg[21] = _mm512_shuffle_i64x2(b_reg[21], b_reg[29], 0x44);              \
    a_reg[22] = _mm512_shuffle_i64x2(b_reg[22], b_reg[30], 0x44);              \
    a_reg[23] = _mm512_shuffle_i64x2(b_reg[23], b_reg[31], 0x44);              \
    a_reg[24] = _mm512_shuffle_i64x2(b_reg[16], b_reg[24], 0xEE);              \
    a_reg[25] = _mm512_shuffle_i64x2(b_reg[17], b_reg[25], 0xEE);              \
    a_reg[26] = _mm512_shuffle_i64x2(b_reg[18], b_reg[26], 0xEE);              \
    a_reg[27] = _mm512_shuffle_i64x2(b_reg[19], b_reg[27], 0xEE);              \
    a_reg[28] = _mm512_shuffle_i64x2(b_reg[20], b_reg[28], 0xEE);              \
    a_reg[29] = _mm512_shuffle_i64x2(b_reg[21], b_reg[29], 0xEE);              \
    a_reg[30] = _mm512_shuffle_i64x2(b_reg[22], b_reg[30], 0xEE);              \
    a_reg[31] = _mm512_shuffle_i64x2(b_reg[23], b_reg[31], 0xEE);              \
                                                                               \
    /* Step 5: Second cross-lane shuffle */                                    \
    b_reg[0]  = _mm512_shuffle_i64x2(a_reg[0], a_reg[16], 0x88);               \
    b_reg[1]  = _mm512_shuffle_i64x2(a_reg[1], a_reg[17], 0x88);               \
    b_reg[2]  = _mm512_shuffle_i64x2(a_reg[2], a_reg[18], 0x88);               \
    b_reg[3]  = _mm512_shuffle_i64x2(a_reg[3], a_reg[19], 0x88);               \
    b_reg[4]  = _mm512_shuffle_i64x2(a_reg[4], a_reg[20], 0x88);               \
    b_reg[5]  = _mm512_shuffle_i64x2(a_reg[5], a_reg[21], 0x88);               \
    b_reg[6]  = _mm512_shuffle_i64x2(a_reg[6], a_reg[22], 0x88);               \
    b_reg[7]  = _mm512_shuffle_i64x2(a_reg[7], a_reg[23], 0x88);               \
    b_reg[8]  = _mm512_shuffle_i64x2(a_reg[0], a_reg[16], 0xDD);               \
    b_reg[9]  = _mm512_shuffle_i64x2(a_reg[1], a_reg[17], 0xDD);               \
    b_reg[10] = _mm512_shuffle_i64x2(a_reg[2], a_reg[18], 0xDD);               \
    b_reg[11] = _mm512_shuffle_i64x2(a_reg[3], a_reg[19], 0xDD);               \
    b_reg[12] = _mm512_shuffle_i64x2(a_reg[4], a_reg[20], 0xDD);               \
    b_reg[13] = _mm512_shuffle_i64x2(a_reg[5], a_reg[21], 0xDD);               \
    b_reg[14] = _mm512_shuffle_i64x2(a_reg[6], a_reg[22], 0xDD);               \
    b_reg[15] = _mm512_shuffle_i64x2(a_reg[7], a_reg[23], 0xDD);               \
    b_reg[16] = _mm512_shuffle_i64x2(a_reg[8], a_reg[24], 0x88);               \
    b_reg[17] = _mm512_shuffle_i64x2(a_reg[9], a_reg[25], 0x88);               \
    b_reg[18] = _mm512_shuffle_i64x2(a_reg[10], a_reg[26], 0x88);              \
    b_reg[19] = _mm512_shuffle_i64x2(a_reg[11], a_reg[27], 0x88);              \
    b_reg[20] = _mm512_shuffle_i64x2(a_reg[12], a_reg[28], 0x88);              \
    b_reg[21] = _mm512_shuffle_i64x2(a_reg[13], a_reg[29], 0x88);              \
    b_reg[22] = _mm512_shuffle_i64x2(a_reg[14], a_reg[30], 0x88);              \
    b_reg[23] = _mm512_shuffle_i64x2(a_reg[15], a_reg[31], 0x88);              \
    b_reg[24] = _mm512_shuffle_i64x2(a_reg[8], a_reg[24], 0xDD);               \
    b_reg[25] = _mm512_shuffle_i64x2(a_reg[9], a_reg[25], 0xDD);               \
    b_reg[26] = _mm512_shuffle_i64x2(a_reg[10], a_reg[26], 0xDD);              \
    b_reg[27] = _mm512_shuffle_i64x2(a_reg[11], a_reg[27], 0xDD);              \
    b_reg[28] = _mm512_shuffle_i64x2(a_reg[12], a_reg[28], 0xDD);              \
    b_reg[29] = _mm512_shuffle_i64x2(a_reg[13], a_reg[29], 0xDD);              \
    b_reg[30] = _mm512_shuffle_i64x2(a_reg[14], a_reg[30], 0xDD);              \
    b_reg[31] = _mm512_shuffle_i64x2(a_reg[15], a_reg[31], 0xDD);              \
                                                                               \
    /* Copy b_reg back to a_reg for stores */                                  \
    a_reg[0]  = b_reg[0];                                                      \
    a_reg[1]  = b_reg[1];                                                      \
    a_reg[2]  = b_reg[2];                                                      \
    a_reg[3]  = b_reg[3];                                                      \
    a_reg[4]  = b_reg[4];                                                      \
    a_reg[5]  = b_reg[5];                                                      \
    a_reg[6]  = b_reg[6];                                                      \
    a_reg[7]  = b_reg[7];                                                      \
    a_reg[8]  = b_reg[8];                                                      \
    a_reg[9]  = b_reg[9];                                                      \
    a_reg[10] = b_reg[10];                                                     \
    a_reg[11] = b_reg[11];                                                     \
    a_reg[12] = b_reg[12];                                                     \
    a_reg[13] = b_reg[13];                                                     \
    a_reg[14] = b_reg[14];                                                     \
    a_reg[15] = b_reg[15];                                                     \
    a_reg[16] = b_reg[16];                                                     \
    a_reg[17] = b_reg[17];                                                     \
    a_reg[18] = b_reg[18];                                                     \
    a_reg[19] = b_reg[19];                                                     \
    a_reg[20] = b_reg[20];                                                     \
    a_reg[21] = b_reg[21];                                                     \
    a_reg[22] = b_reg[22];                                                     \
    a_reg[23] = b_reg[23];                                                     \
    a_reg[24] = b_reg[24];                                                     \
    a_reg[25] = b_reg[25];                                                     \
    a_reg[26] = b_reg[26];                                                     \
    a_reg[27] = b_reg[27];                                                     \
    a_reg[28] = b_reg[28];                                                     \
    a_reg[29] = b_reg[29];                                                     \
    a_reg[30] = b_reg[30];                                                     \
    a_reg[31] = b_reg[31];

/* Forward declarations */
static void
packa_mr32_f16f16f16of16_row_major(float16*       pack_a_buffer,
                                   const float16* a,
                                   const md_t     rs_a,
                                   const md_t     cs_a,
                                   const md_t     MC,
                                   const md_t     KC,
                                   md_t*          rs_p,
                                   md_t*          cs_p);

static void
packa_mr32_f16f16f16of16_col_major(float16*       pack_a_buffer,
                                   const float16* a,
                                   const md_t     rs_a,
                                   const md_t     cs_a,
                                   const md_t     MC,
                                   const md_t     KC,
                                   md_t*          rs_p,
                                   md_t*          cs_p);

/* ============================================================================
 * Main dispatcher function for MR=32 FP16 Pack A
 * ============================================================================
 */
void
packa_mr32_f16f16f16of16(float16*       pack_a_buffer,
                         const float16* a,
                         const md_t     rs_a,
                         const md_t     cs_a,
                         const md_t     MC,
                         const md_t     KC,
                         md_t*          rs_p,
                         md_t*          cs_p)
{
    if (cs_a == 1) {
        /* Row-major: use optimized AVX-512 path (simple copy) */
        packa_mr32_f16f16f16of16_row_major(pack_a_buffer, a, rs_a, cs_a, MC, KC,
                                           rs_p, cs_p);
    } else {
        /* Column-major: use optimized AVX-512 path with 32x32 transpose */
        packa_mr32_f16f16f16of16_col_major(pack_a_buffer, a, rs_a, cs_a, MC, KC,
                                           rs_p, cs_p);
    }
}

/* ============================================================================
 * Row-major packing kernel (simple vectorized copy)
 * ============================================================================
 */

/*
 * Row-major packing for MR=32 panel
 *
 * Input:  A[M][K] row-major (rs_a = K stride, cs_a = 1)
 * Output: packed[m*KC + k] M-MAJOR layout
 *
 * For row-major input, we simply copy rows to packed format.
 * No transpose needed since both are row-major.
 */
static void
packa_mr32_f16f16f16of16_row_major(float16*       pack_a_buffer,
                                   const float16* a,
                                   const md_t     rs_a,
                                   const md_t     cs_a,
                                   const md_t     MC,
                                   const md_t     KC,
                                   md_t*          rs_p,
                                   md_t*          cs_p)
{
    md_t    MR = 32;
    __m512i a_reg[32];

    md_t ic = 0, kr = 0;

    /* Main loop: Process 32 M-rows at a time */
    for (ic = 0; (ic + MR - 1) < MC; ic += MR) {
        /* Process 32 K-elements at a time */
        for (kr = 0; (kr + 31) < KC; kr += 32) {
            /* Load and store 32 rows, each with 32 FP16 elements (interleaved)
             */
            for (md_t m = 0; m < 32; m++) {
                a_reg[m] = _mm512_loadu_si512(a + ((ic + m) * rs_a) + kr);
                _mm512_storeu_si512(pack_a_buffer + ((ic + m) * KC) + kr,
                                    a_reg[m]);
            }
        }

        /* K-fringe: Handle remaining K-elements with masked operations */
        if (kr < KC) {
            md_t      k_rem  = KC - kr;
            __mmask32 k_mask = (__mmask32)((1ULL << k_rem) - 1);

            /* Load 32 rows with masked load */
            for (md_t m = 0; m < 32; m++) {
                a_reg[m] = _mm512_maskz_loadu_epi16(k_mask,
                                                    a + ((ic + m) * rs_a) + kr);
            }

            /* Store with masked store */
            for (md_t m = 0; m < 32; m++) {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + m) * KC) + kr,
                                         k_mask, a_reg[m]);
            }
        }
    }

    /* M-fringe: Handle remaining M-rows (load/store already interleaved) */
    for (; (ic + 15) < MC; ic += 16) {
        for (kr = 0; (kr + 31) < KC; kr += 32) {
            for (md_t m = 0; m < 16; m++) {
                a_reg[m] = _mm512_loadu_si512(a + ((ic + m) * rs_a) + kr);
                _mm512_storeu_si512(pack_a_buffer + ((ic + m) * KC) + kr,
                                    a_reg[m]);
            }
        }
        if (kr < KC) {
            md_t      k_rem  = KC - kr;
            __mmask32 k_mask = (__mmask32)((1ULL << k_rem) - 1);
            for (md_t m = 0; m < 16; m++) {
                a_reg[m] = _mm512_maskz_loadu_epi16(k_mask,
                                                    a + ((ic + m) * rs_a) + kr);
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + m) * KC) + kr,
                                         k_mask, a_reg[m]);
            }
        }
    }

    for (; (ic + 7) < MC; ic += 8) {
        for (kr = 0; (kr + 31) < KC; kr += 32) {
            for (md_t m = 0; m < 8; m++) {
                a_reg[m] = _mm512_loadu_si512(a + ((ic + m) * rs_a) + kr);
                _mm512_storeu_si512(pack_a_buffer + ((ic + m) * KC) + kr,
                                    a_reg[m]);
            }
        }
        if (kr < KC) {
            md_t      k_rem  = KC - kr;
            __mmask32 k_mask = (__mmask32)((1ULL << k_rem) - 1);
            for (md_t m = 0; m < 8; m++) {
                a_reg[m] = _mm512_maskz_loadu_epi16(k_mask,
                                                    a + ((ic + m) * rs_a) + kr);
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + m) * KC) + kr,
                                         k_mask, a_reg[m]);
            }
        }
    }

    for (; (ic + 3) < MC; ic += 4) {
        for (kr = 0; (kr + 31) < KC; kr += 32) {
            for (md_t m = 0; m < 4; m++) {
                a_reg[m] = _mm512_loadu_si512(a + ((ic + m) * rs_a) + kr);
                _mm512_storeu_si512(pack_a_buffer + ((ic + m) * KC) + kr,
                                    a_reg[m]);
            }
        }
        if (kr < KC) {
            md_t      k_rem  = KC - kr;
            __mmask32 k_mask = (__mmask32)((1ULL << k_rem) - 1);
            for (md_t m = 0; m < 4; m++) {
                a_reg[m] = _mm512_maskz_loadu_epi16(k_mask,
                                                    a + ((ic + m) * rs_a) + kr);
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + m) * KC) + kr,
                                         k_mask, a_reg[m]);
            }
        }
    }

    for (; (ic + 1) < MC; ic += 2) {
        for (kr = 0; (kr + 31) < KC; kr += 32) {
            for (md_t m = 0; m < 2; m++) {
                a_reg[m] = _mm512_loadu_si512(a + ((ic + m) * rs_a) + kr);
                _mm512_storeu_si512(pack_a_buffer + ((ic + m) * KC) + kr,
                                    a_reg[m]);
            }
        }
        if (kr < KC) {
            md_t      k_rem  = KC - kr;
            __mmask32 k_mask = (__mmask32)((1ULL << k_rem) - 1);
            for (md_t m = 0; m < 2; m++) {
                a_reg[m] = _mm512_maskz_loadu_epi16(k_mask,
                                                    a + ((ic + m) * rs_a) + kr);
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + m) * KC) + kr,
                                         k_mask, a_reg[m]);
            }
        }
    }

    for (; ic < MC; ic += 1) {
        for (kr = 0; (kr + 31) < KC; kr += 32) {
            a_reg[0] = _mm512_loadu_si512(a + (ic * rs_a) + kr);
            _mm512_storeu_si512(pack_a_buffer + (ic * KC) + kr, a_reg[0]);
        }
        if (kr < KC) {
            md_t      k_rem  = KC - kr;
            __mmask32 k_mask = (__mmask32)((1ULL << k_rem) - 1);
            a_reg[0] = _mm512_maskz_loadu_epi16(k_mask, a + (ic * rs_a) + kr);
            _mm512_mask_storeu_epi16(pack_a_buffer + (ic * KC) + kr, k_mask,
                                     a_reg[0]);
        }
    }

    /* Output strides for M-MAJOR layout */
    *rs_p = KC;
    *cs_p = 1;
}

/* ============================================================================
 * Column-major packing kernel (with 32x32 transpose)
 * ============================================================================
 */

/*
 * Column-major packing for MR=32 panel
 *
 * Input:  A[M][K] column-major (rs_a = 1, cs_a = M stride)
 * Output: packed[m*KC + k] M-MAJOR layout
 *
 * For column-major input, we need to transpose 32 columns at a time.
 * Each column becomes a row in the M-MAJOR output layout.
 */
static void
packa_mr32_f16f16f16of16_col_major(float16*       pack_a_buffer,
                                   const float16* a,
                                   const md_t     rs_a,
                                   const md_t     cs_a,
                                   const md_t     MC,
                                   const md_t     KC,
                                   md_t*          rs_p,
                                   md_t*          cs_p)
{
    md_t    MR = 32;
    __m512i a_reg[32];
    __m512i b_reg[32];

    md_t ic = 0, kr = 0;

    /* Main loop: Process 32 M-rows at a time */
    for (ic = 0; (ic + MR - 1) < MC; ic += MR) {
        /* Process 32 K-elements at a time (32x32 transpose) */
        for (kr = 0; (kr + 31) < KC; kr += 32) {
            /* Load 32 columns (K-dim), each with 32 M-elements */
            for (md_t i = 0; i < 32; i++) {
                a_reg[i] = _mm512_loadu_si512(a + ic + ((kr + i) * cs_a));
            }

            /* Transpose 32x32 block (reuse macro from PackB) */
            TRANSPOSE_32x32_FP16_AVX512

                /* Store transposed result in M-MAJOR layout */
                for (md_t i = 0; i < 32; i++)
            {
                _mm512_storeu_si512(pack_a_buffer + ((ic + i) * KC) + kr,
                                    a_reg[i]);
            }
        }

        /* K-fringe: Handle remaining K-elements with masked transpose
         * Subdivide into chunks: 16, 8, 4, 2, 1 (matching PackB pattern)
         */
        /* K=16 fringe */
        for (; (kr + 15) < KC; kr += 16) {
            __mmask32 m_mask     = 0xFFFFFFFF; /* All 32 M-elements */
            md_t      k_cols     = 16;     /* Number of K columns to process */
            __mmask32 store_mask = 0xFFFF; /* Store only 16 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                /* Store ALL 32 transposed rows with masked store (only k_cols
                   K-elements each) */
                for (md_t i = 0; i < 32; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }

        /* K=8 fringe */
        for (; (kr + 7) < KC; kr += 8) {
            __mmask32 m_mask     = 0xFFFFFFFF; /* All 32 M-elements */
            md_t      k_cols     = 8;    /* Number of K columns to process */
            __mmask32 store_mask = 0xFF; /* Store only 8 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                /* Store ALL 32 transposed rows with masked store (only k_cols
                   K-elements each) */
                for (md_t i = 0; i < 32; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }

        /* K=4 fringe */
        for (; (kr + 3) < KC; kr += 4) {
            __mmask32 m_mask     = 0xFFFFFFFF; /* All 32 M-elements */
            md_t      k_cols     = 4;   /* Number of K columns to process */
            __mmask32 store_mask = 0xF; /* Store only 4 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                /* Store ALL 32 transposed rows with masked store (only k_cols
                   K-elements each) */
                for (md_t i = 0; i < 32; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }

        /* K=2 fringe */
        for (; (kr + 1) < KC; kr += 2) {
            __mmask32 m_mask     = 0xFFFFFFFF; /* All 32 M-elements */
            md_t      k_cols     = 2;   /* Number of K columns to process */
            __mmask32 store_mask = 0x3; /* Store only 2 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                /* Store ALL 32 transposed rows with masked store (only k_cols
                   K-elements each) */
                for (md_t i = 0; i < 32; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }

        /* K=1 fringe */
        for (; kr < KC; kr += 1) {
            __mmask32 m_mask     = 0xFFFFFFFF; /* All 32 M-elements */
            md_t      k_cols     = 1;   /* Number of K columns to process */
            __mmask32 store_mask = 0x1; /* Store only 1 element per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                /* Store ALL 32 transposed rows with masked store (only k_cols
                   K-elements each) */
                for (md_t i = 0; i < 32; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
    }

    /* M-fringe: Handle remaining M-rows */
    for (; (ic + 15) < MC; ic += 16) {
        for (kr = 0; (kr + 31) < KC; kr += 32) {
            /* Load 32 columns, each with 16 M-elements (zero-pad upper 16) */
            for (md_t i = 0; i < 32; i++) {
                __mmask32 m_mask = 0xFFFF; /* 16 elements valid */
                a_reg[i]         = _mm512_maskz_loadu_epi16(m_mask,
                                                            a + ic + ((kr + i) * cs_a));
            }

            TRANSPOSE_32x32_FP16_AVX512

                /* Store first 16 transposed rows */
                for (md_t i = 0; i < 16; i++)
            {
                _mm512_storeu_si512(pack_a_buffer + ((ic + i) * KC) + kr,
                                    a_reg[i]);
            }
        }
        /* K-fringe: Handle remaining K-elements with masked transpose */
        /* K=16 fringe */
        for (; (kr + 15) < KC; kr += 16) {
            __mmask32 m_mask     = 0xFFFF; /* 16 M-elements */
            md_t      k_cols     = 16;     /* Number of K columns to process */
            __mmask32 store_mask = 0xFFFF; /* Store only 16 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                /* Store 16 M-rows with masked store (only k_cols K-elements
                   each) */
                for (md_t i = 0; i < 16; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
        /* K=8 fringe */
        for (; (kr + 7) < KC; kr += 8) {
            __mmask32 m_mask     = 0xFFFF; /* 16 M-elements */
            md_t      k_cols     = 8;      /* Number of K columns to process */
            __mmask32 store_mask = 0xFF;   /* Store only 8 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 16; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
        /* K=4 fringe */
        for (; (kr + 3) < KC; kr += 4) {
            __mmask32 m_mask     = 0xFFFF; /* 16 M-elements */
            md_t      k_cols     = 4;      /* Number of K columns to process */
            __mmask32 store_mask = 0xF;    /* Store only 4 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 16; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
        /* K=2 fringe */
        for (; (kr + 1) < KC; kr += 2) {
            __mmask32 m_mask     = 0xFFFF; /* 16 M-elements */
            md_t      k_cols     = 2;      /* Number of K columns to process */
            __mmask32 store_mask = 0x3;    /* Store only 2 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 16; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
        /* K=1 fringe */
        for (; kr < KC; kr += 1) {
            __mmask32 m_mask     = 0xFFFF; /* 16 M-elements */
            md_t      k_cols     = 1;      /* Number of K columns to process */
            __mmask32 store_mask = 0x1;    /* Store only 1 element per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 16; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
    }

    for (; (ic + 7) < MC; ic += 8) {
        for (kr = 0; (kr + 31) < KC; kr += 32) {
            for (md_t i = 0; i < 32; i++) {
                __mmask32 m_mask = 0xFF; /* 8 elements valid */
                a_reg[i]         = _mm512_maskz_loadu_epi16(m_mask,
                                                            a + ic + ((kr + i) * cs_a));
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 8; i++)
            {
                _mm512_storeu_si512(pack_a_buffer + ((ic + i) * KC) + kr,
                                    a_reg[i]);
            }
        }
        /* K-fringe: Handle remaining K-elements */
        for (; (kr + 15) < KC; kr += 16) {
            __mmask32 m_mask     = 0xFF;   /* 8 M-elements */
            md_t      k_cols     = 16;     /* Number of K columns to process */
            __mmask32 store_mask = 0xFFFF; /* Store only 16 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 8; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
        for (; (kr + 7) < KC; kr += 8) {
            __mmask32 m_mask     = 0xFF; /* 8 M-elements */
            md_t      k_cols     = 8;    /* Number of K columns to process */
            __mmask32 store_mask = 0xFF; /* Store only 8 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 8; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
        for (; (kr + 3) < KC; kr += 4) {
            __mmask32 m_mask     = 0xFF; /* 8 M-elements */
            md_t      k_cols     = 4;    /* Number of K columns to process */
            __mmask32 store_mask = 0xF;  /* Store only 4 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 8; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
        for (; (kr + 1) < KC; kr += 2) {
            __mmask32 m_mask     = 0xFF; /* 8 M-elements */
            md_t      k_cols     = 2;    /* Number of K columns to process */
            __mmask32 store_mask = 0x3;  /* Store only 2 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 8; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
        for (; kr < KC; kr += 1) {
            __mmask32 m_mask     = 0xFF; /* 8 M-elements */
            md_t      k_cols     = 1;    /* Number of K columns to process */
            __mmask32 store_mask = 0x1;  /* Store only 1 element per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 8; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
    }

    for (; (ic + 3) < MC; ic += 4) {
        for (kr = 0; (kr + 31) < KC; kr += 32) {
            for (md_t i = 0; i < 32; i++) {
                __mmask32 m_mask = 0xF; /* 4 elements valid */
                a_reg[i]         = _mm512_maskz_loadu_epi16(m_mask,
                                                            a + ic + ((kr + i) * cs_a));
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 4; i++)
            {
                _mm512_storeu_si512(pack_a_buffer + ((ic + i) * KC) + kr,
                                    a_reg[i]);
            }
        }
        /* K-fringe: Handle remaining K-elements */
        for (; (kr + 15) < KC; kr += 16) {
            __mmask32 m_mask     = 0xF;    /* 4 M-elements */
            md_t      k_cols     = 16;     /* Number of K columns to process */
            __mmask32 store_mask = 0xFFFF; /* Store only 16 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 4; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
        for (; (kr + 7) < KC; kr += 8) {
            __mmask32 m_mask     = 0xF;  /* 4 M-elements */
            md_t      k_cols     = 8;    /* Number of K columns to process */
            __mmask32 store_mask = 0xFF; /* Store only 8 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 4; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
        for (; (kr + 3) < KC; kr += 4) {
            __mmask32 m_mask     = 0xF; /* 4 M-elements */
            md_t      k_cols     = 4;   /* Number of K columns to process */
            __mmask32 store_mask = 0xF; /* Store only 4 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 4; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
        for (; (kr + 1) < KC; kr += 2) {
            __mmask32 m_mask     = 0xF; /* 4 M-elements */
            md_t      k_cols     = 2;   /* Number of K columns to process */
            __mmask32 store_mask = 0x3; /* Store only 2 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 4; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
        for (; kr < KC; kr += 1) {
            __mmask32 m_mask     = 0xF; /* 4 M-elements */
            md_t      k_cols     = 1;   /* Number of K columns to process */
            __mmask32 store_mask = 0x1; /* Store only 1 element per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 4; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
    }

    for (; (ic + 1) < MC; ic += 2) {
        for (kr = 0; (kr + 31) < KC; kr += 32) {
            for (md_t i = 0; i < 32; i++) {
                __mmask32 m_mask = 0x3; /* 2 elements valid */
                a_reg[i]         = _mm512_maskz_loadu_epi16(m_mask,
                                                            a + ic + ((kr + i) * cs_a));
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 2; i++)
            {
                _mm512_storeu_si512(pack_a_buffer + ((ic + i) * KC) + kr,
                                    a_reg[i]);
            }
        }
        /* K-fringe: Handle remaining K-elements */
        for (; (kr + 15) < KC; kr += 16) {
            __mmask32 m_mask     = 0x3;    /* 2 M-elements */
            md_t      k_cols     = 16;     /* Number of K columns to process */
            __mmask32 store_mask = 0xFFFF; /* Store only 16 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 2; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
        for (; (kr + 7) < KC; kr += 8) {
            __mmask32 m_mask     = 0x3;  /* 2 M-elements */
            md_t      k_cols     = 8;    /* Number of K columns to process */
            __mmask32 store_mask = 0xFF; /* Store only 8 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 2; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
        for (; (kr + 3) < KC; kr += 4) {
            __mmask32 m_mask     = 0x3; /* 2 M-elements */
            md_t      k_cols     = 4;   /* Number of K columns to process */
            __mmask32 store_mask = 0xF; /* Store only 4 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 2; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
        for (; (kr + 1) < KC; kr += 2) {
            __mmask32 m_mask     = 0x3; /* 2 M-elements */
            md_t      k_cols     = 2;   /* Number of K columns to process */
            __mmask32 store_mask = 0x3; /* Store only 2 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 2; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
        for (; kr < KC; kr += 1) {
            __mmask32 m_mask     = 0x3; /* 2 M-elements */
            md_t      k_cols     = 1;   /* Number of K columns to process */
            __mmask32 store_mask = 0x1; /* Store only 1 element per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                for (md_t i = 0; i < 2; i++)
            {
                _mm512_mask_storeu_epi16(pack_a_buffer + ((ic + i) * KC) + kr,
                                         store_mask, a_reg[i]);
            }
        }
    }

    for (; ic < MC; ic += 1) {
        for (kr = 0; (kr + 31) < KC; kr += 32) {
            for (md_t i = 0; i < 32; i++) {
                __mmask32 m_mask = 0x1; /* 1 element valid */
                a_reg[i]         = _mm512_maskz_loadu_epi16(m_mask,
                                                            a + ic + ((kr + i) * cs_a));
            }

            TRANSPOSE_32x32_FP16_AVX512

                _mm512_storeu_si512(pack_a_buffer + (ic * KC) + kr, a_reg[0]);
        }
        /* K-fringe: Handle remaining K-elements */
        for (; (kr + 15) < KC; kr += 16) {
            __mmask32 m_mask     = 0x1;    /* 1 M-element */
            md_t      k_cols     = 16;     /* Number of K columns to process */
            __mmask32 store_mask = 0xFFFF; /* Store only 16 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                _mm512_mask_storeu_epi16(pack_a_buffer + (ic * KC) + kr,
                                         store_mask, a_reg[0]);
        }
        for (; (kr + 7) < KC; kr += 8) {
            __mmask32 m_mask     = 0x1;  /* 1 M-element */
            md_t      k_cols     = 8;    /* Number of K columns to process */
            __mmask32 store_mask = 0xFF; /* Store only 8 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                _mm512_mask_storeu_epi16(pack_a_buffer + (ic * KC) + kr,
                                         store_mask, a_reg[0]);
        }
        for (; (kr + 3) < KC; kr += 4) {
            __mmask32 m_mask     = 0x1; /* 1 M-element */
            md_t      k_cols     = 4;   /* Number of K columns to process */
            __mmask32 store_mask = 0xF; /* Store only 4 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                _mm512_mask_storeu_epi16(pack_a_buffer + (ic * KC) + kr,
                                         store_mask, a_reg[0]);
        }
        for (; (kr + 1) < KC; kr += 2) {
            __mmask32 m_mask     = 0x1; /* 1 M-element */
            md_t      k_cols     = 2;   /* Number of K columns to process */
            __mmask32 store_mask = 0x3; /* Store only 2 elements per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                _mm512_mask_storeu_epi16(pack_a_buffer + (ic * KC) + kr,
                                         store_mask, a_reg[0]);
        }
        for (; kr < KC; kr += 1) {
            __mmask32 m_mask     = 0x1; /* 1 M-element */
            md_t      k_cols     = 1;   /* Number of K columns to process */
            __mmask32 store_mask = 0x1; /* Store only 1 element per row */

            /* Load only valid columns, zero-pad the rest */
            for (md_t i = 0; i < 32; i++) {
                if (i < k_cols) {
                    a_reg[i] = _mm512_maskz_loadu_epi16(
                        m_mask, a + ic + ((kr + i) * cs_a));
                } else {
                    a_reg[i] = _mm512_setzero_si512();
                }
            }

            TRANSPOSE_32x32_FP16_AVX512

                _mm512_mask_storeu_epi16(pack_a_buffer + (ic * KC) + kr,
                                         store_mask, a_reg[0]);
        }
    }

    /* Output strides for M-MAJOR layout */
    *rs_p = KC;
    *cs_p = 1;
}
