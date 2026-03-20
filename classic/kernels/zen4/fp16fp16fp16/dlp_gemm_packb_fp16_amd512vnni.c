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
 * FP16 AVX-512 Optimized Pack B Kernel (NR=128 configuration)
 *
 * Key differences from BF16:
 * - No k-padding required (packing_factor = 1, unlike BF16's factor of 2)
 * - No interleaving required (BF16 interleaves for dpbf16_ps instruction)
 * - Simple K-MAJOR layout: packed[k*NR + n]
 *
 * NR=128 uses 4 ZMM registers (32 FP16 elements each).
 * Chunk hierarchy: 128 -> 96 -> 64 -> 32 -> lt32 (padded to 32)
 *
 * This kernel provides vectorized packing for both row-major and col-major
 * input matrices, outputting K-MAJOR packed format.
 */

#include "classic/aocl_fp16_type.h"
#include "fp16fp16fp16/dlp_gemm_reorder_fp16.h" /* For reference fallback */
#include "kernels/dlp_kernels.h"
#include <immintrin.h>
#include <string.h>

/* Forward declarations - Row-major kernels */
static void
dlp_packb_nr128_f16f16f16of16_row_major_kernel(float16*       pack_b_buffer,
                                               const float16* b,
                                               const md_t     ldb,
                                               const md_t     NC,
                                               const md_t     KC,
                                               md_t*          rs_b,
                                               md_t*          cs_b);

static void
dlp_packb_nr96_f16f16f16of16_row_major_kernel(float16*       pack_b_buffer,
                                              const float16* b,
                                              const md_t     ldb,
                                              const md_t     KC);

static void
dlp_packb_nr64_f16f16f16of16_row_major_kernel(float16*       pack_b_buffer,
                                              const float16* b,
                                              const md_t     ldb,
                                              const md_t     KC);

static void
dlp_packb_nr32_f16f16f16of16_row_major_kernel(float16*       pack_b_buffer,
                                              const float16* b,
                                              const md_t     ldb,
                                              const md_t     KC);

static void
dlp_packb_nrlt32_f16f16f16of16_row_major_kernel(float16*       pack_b_buffer,
                                                const float16* b,
                                                const md_t     ldb,
                                                const md_t     KC,
                                                const md_t     n0_partial_rem);

/* Forward declarations - Column-major kernels */
static void
dlp_packb_nr128_f16f16f16of16_col_major_kernel(float16*       pack_b_buffer,
                                               const float16* b,
                                               const md_t     ldb,
                                               const md_t     NC,
                                               const md_t     KC,
                                               md_t*          rs_b,
                                               md_t*          cs_b);

static void
dlp_packb_nr_mult_32_f16f16f16of16_col_major(float16*       pack_b_buffer,
                                             const float16* b,
                                             const md_t     NR,
                                             const md_t     ldb,
                                             const md_t     KC);

static void
dlp_packb_nrlt32_f16f16f16of16_col_major_kernel(float16*       pack_b_buffer,
                                                const float16* b,
                                                const md_t     ldb,
                                                const md_t     KC,
                                                const md_t     n0_partial_rem);

/* ============================================================================
 * Main dispatcher function for NR=128 FP16 Pack B
 * ============================================================================
 */
void
dlp_packb_nr128_f16f16f16of16(float16*       pack_b_buffer,
                              const float16* b,
                              const md_t     rs_b,
                              const md_t     cs_b,
                              const md_t     NC,
                              const md_t     KC,
                              md_t*          rs_p,
                              md_t*          cs_p)
{
    if (cs_b == 1) {
        /* Row-major: use optimized AVX-512 path
         * For row-major, ldb = rs_b (row stride = N) */
        dlp_packb_nr128_f16f16f16of16_row_major_kernel(pack_b_buffer, b, rs_b,
                                                       NC, KC, rs_p, cs_p);
    } else {
        /* Column-major: use optimized AVX-512 path with 32x32 transpose
         * For column-major, ldb = cs_b (column stride = K) */
        dlp_packb_nr128_f16f16f16of16_col_major_kernel(pack_b_buffer, b, cs_b,
                                                       NC, KC, rs_p, cs_p);
    }
}

/* ============================================================================
 * Row-major packing kernels
 * ============================================================================
 */

/*
 * Row-major packing for NR=128 panel (4 ZMM registers)
 *
 * Input:  B[K][N] row-major (ldb = N stride)
 * Output: packed[k*NR + n] K-MAJOR layout
 */
static void
dlp_packb_nr128_f16f16f16of16_row_major_kernel(float16*       pack_b_buffer,
                                               const float16* b,
                                               const md_t     ldb,
                                               const md_t     NC,
                                               const md_t     KC,
                                               md_t*          rs_b,
                                               md_t*          cs_b)
{
    md_t NR = 128;

    md_t n_full_pieces            = NC / NR;
    md_t n_full_pieces_loop_limit = n_full_pieces * NR;
    md_t n_partial_pieces         = NC % NR;

    __m512i a0, a1, a2, a3;

    /* Pack full NR=128 panels (4 ZMM loads per k-row) */
    for (iter_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR) {
        for (iter_t kr = 0; kr < KC; kr++) {
            /* Load 128 FP16 elements (4x ZMM registers, 32 elements each) */
            a0 = _mm512_loadu_si512(b + (ldb * kr) + jc);
            a1 = _mm512_loadu_si512(b + (ldb * kr) + jc + 32);
            a2 = _mm512_loadu_si512(b + (ldb * kr) + jc + 64);
            a3 = _mm512_loadu_si512(b + (ldb * kr) + jc + 96);

            /* Store directly in K-MAJOR layout: pack[k*NR + n] */
            _mm512_storeu_si512(pack_b_buffer + (jc * KC) + (kr * NR), a0);
            _mm512_storeu_si512(pack_b_buffer + (jc * KC) + (kr * NR) + 32, a1);
            _mm512_storeu_si512(pack_b_buffer + (jc * KC) + (kr * NR) + 64, a2);
            _mm512_storeu_si512(pack_b_buffer + (jc * KC) + (kr * NR) + 96, a3);
        }
    }

    /* Handle partial panel (NC % NR != 0) */
    if (n_partial_pieces > 0) {
        md_t n0_partial_rem  = n_partial_pieces % 32;
        md_t n0_partial_pack = 0;

        /* Split into optimal chunks: 96, 64, 32, or remainder */
        md_t n0_96 = n_partial_pieces / 96;
        md_t n0_64 = n_partial_pieces / 64;
        md_t n0_32 = n_partial_pieces / 32;

        if (n0_96 == 1) {
            dlp_packb_nr96_f16f16f16of16_row_major_kernel(
                (pack_b_buffer + (n_full_pieces_loop_limit * KC)),
                (b + n_full_pieces_loop_limit), ldb, KC);
            n0_partial_pack = 96;
        } else if (n0_64 == 1) {
            dlp_packb_nr64_f16f16f16of16_row_major_kernel(
                (pack_b_buffer + (n_full_pieces_loop_limit * KC)),
                (b + n_full_pieces_loop_limit), ldb, KC);
            n0_partial_pack = 64;
        } else if (n0_32 == 1) {
            dlp_packb_nr32_f16f16f16of16_row_major_kernel(
                (pack_b_buffer + (n_full_pieces_loop_limit * KC)),
                (b + n_full_pieces_loop_limit), ldb, KC);
            n0_partial_pack = 32;
        }

        if (n0_partial_rem > 0) {
            dlp_packb_nrlt32_f16f16f16of16_row_major_kernel(
                (pack_b_buffer + (n_full_pieces_loop_limit * KC)
                 + (n0_partial_pack * KC)),
                (b + n_full_pieces_loop_limit + n0_partial_pack), ldb, KC,
                n0_partial_rem);
        }
    }

    /* Output strides for K-MAJOR layout
     * rs_b = NR is the base stride for full panels.
     * The GEMM kernel adjusts this for fringe panels (similar to INT8).
     * For fringe panel at n0, kernel computes: (rs_b * n0_width) / NR
     */
    *rs_b = NR; /* Always return NR=128 as base stride */
    *cs_b = 1;  /* Stride to next n within same k-row */
}

/*
 * Row-major packing for NR=96 fringe panel (3 ZMM registers)
 */
static void
dlp_packb_nr96_f16f16f16of16_row_major_kernel(float16*       pack_b_buffer,
                                              const float16* b,
                                              const md_t     ldb,
                                              const md_t     KC)
{
    md_t NR = 96;

    __m512i a0, a1, a2;

    for (iter_t kr = 0; kr < KC; kr++) {
        /* Load 96 FP16 elements: 3x ZMM (32 each) */
        a0 = _mm512_loadu_si512(b + (ldb * kr));
        a1 = _mm512_loadu_si512(b + (ldb * kr) + 32);
        a2 = _mm512_loadu_si512(b + (ldb * kr) + 64);

        /* Store in K-MAJOR layout */
        _mm512_storeu_si512(pack_b_buffer + (kr * NR), a0);
        _mm512_storeu_si512(pack_b_buffer + (kr * NR) + 32, a1);
        _mm512_storeu_si512(pack_b_buffer + (kr * NR) + 64, a2);
    }
}

/*
 * Row-major packing for NR=64 fringe panel (2 ZMM registers)
 */
static void
dlp_packb_nr64_f16f16f16of16_row_major_kernel(float16*       pack_b_buffer,
                                              const float16* b,
                                              const md_t     ldb,
                                              const md_t     KC)
{
    md_t NR = 64;

    __m512i a0, a1;

    for (iter_t kr = 0; kr < KC; kr++) {
        /* Load 64 FP16 elements (2x ZMM registers, 32 elements each) */
        a0 = _mm512_loadu_si512(b + (ldb * kr));
        a1 = _mm512_loadu_si512(b + (ldb * kr) + 32);

        /* Store in K-MAJOR layout */
        _mm512_storeu_si512(pack_b_buffer + (kr * NR), a0);
        _mm512_storeu_si512(pack_b_buffer + (kr * NR) + 32, a1);
    }
}

/*
 * Row-major packing for NR=32 fringe panel (1 ZMM register)
 */
static void
dlp_packb_nr32_f16f16f16of16_row_major_kernel(float16*       pack_b_buffer,
                                              const float16* b,
                                              const md_t     ldb,
                                              const md_t     KC)
{
    md_t NR = 32;

    __m512i a0;

    for (iter_t kr = 0; kr < KC; kr++) {
        /* Load 32 FP16 elements (1x ZMM register) */
        a0 = _mm512_loadu_si512(b + (ldb * kr));

        /* Store in K-MAJOR layout */
        _mm512_storeu_si512(pack_b_buffer + (kr * NR), a0);
    }
}

/*
 * Row-major packing for NR<32 fringe panel with masking
 * Pads to NR=32 with zeros
 */
static void
dlp_packb_nrlt32_f16f16f16of16_row_major_kernel(float16*       pack_b_buffer,
                                                const float16* b,
                                                const md_t     ldb,
                                                const md_t     KC,
                                                const md_t     n0_partial_rem)
{
    md_t NR = 32;

    __m512i a0;

    /* Create mask for partial load (n0_partial_rem elements) */
    __mmask32 load_mask = (__mmask32)((1ULL << n0_partial_rem) - 1);

    for (iter_t kr = 0; kr < KC; kr++) {
        /* Masked load of partial elements (rest are zero) */
        a0 = _mm512_maskz_loadu_epi16(load_mask, b + (ldb * kr));

        /* Store full NR=32 with zero padding */
        _mm512_storeu_si512(pack_b_buffer + (kr * NR), a0);
    }
}

/* ============================================================================
 * Column-major packing kernels (require transpose)
 * ============================================================================
 */

/* Macros for 32-column transpose using 32 columns at a time */
#define LOAD_32_COLS_FP16_AVX512(b, ldb, kr)                                   \
    a_reg[0]  = _mm512_loadu_si512((b) + (ldb * 0) + (kr));                    \
    a_reg[1]  = _mm512_loadu_si512((b) + (ldb * 1) + (kr));                    \
    a_reg[2]  = _mm512_loadu_si512((b) + (ldb * 2) + (kr));                    \
    a_reg[3]  = _mm512_loadu_si512((b) + (ldb * 3) + (kr));                    \
    a_reg[4]  = _mm512_loadu_si512((b) + (ldb * 4) + (kr));                    \
    a_reg[5]  = _mm512_loadu_si512((b) + (ldb * 5) + (kr));                    \
    a_reg[6]  = _mm512_loadu_si512((b) + (ldb * 6) + (kr));                    \
    a_reg[7]  = _mm512_loadu_si512((b) + (ldb * 7) + (kr));                    \
    a_reg[8]  = _mm512_loadu_si512((b) + (ldb * 8) + (kr));                    \
    a_reg[9]  = _mm512_loadu_si512((b) + (ldb * 9) + (kr));                    \
    a_reg[10] = _mm512_loadu_si512((b) + (ldb * 10) + (kr));                   \
    a_reg[11] = _mm512_loadu_si512((b) + (ldb * 11) + (kr));                   \
    a_reg[12] = _mm512_loadu_si512((b) + (ldb * 12) + (kr));                   \
    a_reg[13] = _mm512_loadu_si512((b) + (ldb * 13) + (kr));                   \
    a_reg[14] = _mm512_loadu_si512((b) + (ldb * 14) + (kr));                   \
    a_reg[15] = _mm512_loadu_si512((b) + (ldb * 15) + (kr));                   \
    a_reg[16] = _mm512_loadu_si512((b) + (ldb * 16) + (kr));                   \
    a_reg[17] = _mm512_loadu_si512((b) + (ldb * 17) + (kr));                   \
    a_reg[18] = _mm512_loadu_si512((b) + (ldb * 18) + (kr));                   \
    a_reg[19] = _mm512_loadu_si512((b) + (ldb * 19) + (kr));                   \
    a_reg[20] = _mm512_loadu_si512((b) + (ldb * 20) + (kr));                   \
    a_reg[21] = _mm512_loadu_si512((b) + (ldb * 21) + (kr));                   \
    a_reg[22] = _mm512_loadu_si512((b) + (ldb * 22) + (kr));                   \
    a_reg[23] = _mm512_loadu_si512((b) + (ldb * 23) + (kr));                   \
    a_reg[24] = _mm512_loadu_si512((b) + (ldb * 24) + (kr));                   \
    a_reg[25] = _mm512_loadu_si512((b) + (ldb * 25) + (kr));                   \
    a_reg[26] = _mm512_loadu_si512((b) + (ldb * 26) + (kr));                   \
    a_reg[27] = _mm512_loadu_si512((b) + (ldb * 27) + (kr));                   \
    a_reg[28] = _mm512_loadu_si512((b) + (ldb * 28) + (kr));                   \
    a_reg[29] = _mm512_loadu_si512((b) + (ldb * 29) + (kr));                   \
    a_reg[30] = _mm512_loadu_si512((b) + (ldb * 30) + (kr));                   \
    a_reg[31] = _mm512_loadu_si512((b) + (ldb * 31) + (kr));

/* 32x32 transpose for FP16 (16-bit elements)
 * Transposes 32 columns x 32 k-elements into K-MAJOR format
 * Output: 32 consecutive k-values for each of the 32 n-positions
 *
 * 5-stage algorithm:
 *   Stage 1: unpacklo/hi epi16 - pairs adjacent 16-bit elements
 *   Stage 2: unpacklo/hi epi32 - pairs 32-bit groups
 *   Stage 3: unpacklo/hi epi64 - pairs 64-bit groups
 *   Stage 4: shuffle_i64x2 - first cross-lane shuffle (combines regs 0-7 with
 * 8-15) Stage 5: shuffle_i64x2 - second cross-lane shuffle (combines first half
 * with second half)
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
    /* Step 4: First cross-lane shuffle (combines regs 0-7 with 8-15, etc.)    \
     * shuffle_i64x2 selects 128-bit lanes:                                    \
     *   0x44 = 01|00|01|00 : lanes 0,1 from src1; lanes 0,1 from src2         \
     *   0xEE = 11|10|11|10 : lanes 2,3 from src1; lanes 2,3 from src2         \
     */                                                                        \
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
    /* Step 5: Second cross-lane shuffle (combines first 16 with second 16)    \
     * After stage 4, lanes 0,2 contain even k-rows and lanes 1,3 contain      \
     * odd k-rows. Use 0x88 to select lanes 0,2 and 0xDD to select lanes 1,3.  \
     */                                                                        \
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

/* Masked load macro for K-fringe handling (loads k_count elements per column)
 * k_mask: bitmask for which k-elements to load (e.g., 0xFFFF for 16 elements)
 */
#define MASK_LOAD_32_COLS_FP16_AVX512(b, ldb, kr, k_mask)                      \
    a_reg[0]  = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 0) + (kr));      \
    a_reg[1]  = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 1) + (kr));      \
    a_reg[2]  = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 2) + (kr));      \
    a_reg[3]  = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 3) + (kr));      \
    a_reg[4]  = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 4) + (kr));      \
    a_reg[5]  = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 5) + (kr));      \
    a_reg[6]  = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 6) + (kr));      \
    a_reg[7]  = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 7) + (kr));      \
    a_reg[8]  = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 8) + (kr));      \
    a_reg[9]  = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 9) + (kr));      \
    a_reg[10] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 10) + (kr));     \
    a_reg[11] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 11) + (kr));     \
    a_reg[12] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 12) + (kr));     \
    a_reg[13] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 13) + (kr));     \
    a_reg[14] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 14) + (kr));     \
    a_reg[15] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 15) + (kr));     \
    a_reg[16] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 16) + (kr));     \
    a_reg[17] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 17) + (kr));     \
    a_reg[18] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 18) + (kr));     \
    a_reg[19] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 19) + (kr));     \
    a_reg[20] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 20) + (kr));     \
    a_reg[21] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 21) + (kr));     \
    a_reg[22] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 22) + (kr));     \
    a_reg[23] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 23) + (kr));     \
    a_reg[24] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 24) + (kr));     \
    a_reg[25] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 25) + (kr));     \
    a_reg[26] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 26) + (kr));     \
    a_reg[27] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 27) + (kr));     \
    a_reg[28] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 28) + (kr));     \
    a_reg[29] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 29) + (kr));     \
    a_reg[30] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 30) + (kr));     \
    a_reg[31] = _mm512_maskz_loadu_epi16(k_mask, (b) + (ldb * 31) + (kr));

/* Store first 16 transposed rows (used for K=16 fringe) */
#define STORE_16_ROWS_FP16_AVX512(pack_b, kr, NR)                              \
    _mm512_storeu_si512(pack_b + ((kr + 0) * NR), a_reg[0]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 1) * NR), a_reg[1]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 2) * NR), a_reg[2]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 3) * NR), a_reg[3]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 4) * NR), a_reg[4]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 5) * NR), a_reg[5]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 6) * NR), a_reg[6]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 7) * NR), a_reg[7]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 8) * NR), a_reg[8]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 9) * NR), a_reg[9]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 10) * NR), a_reg[10]);                 \
    _mm512_storeu_si512(pack_b + ((kr + 11) * NR), a_reg[11]);                 \
    _mm512_storeu_si512(pack_b + ((kr + 12) * NR), a_reg[12]);                 \
    _mm512_storeu_si512(pack_b + ((kr + 13) * NR), a_reg[13]);                 \
    _mm512_storeu_si512(pack_b + ((kr + 14) * NR), a_reg[14]);                 \
    _mm512_storeu_si512(pack_b + ((kr + 15) * NR), a_reg[15]);

/* Store first 8 transposed rows (used for K=8 fringe) */
#define STORE_8_ROWS_FP16_AVX512(pack_b, kr, NR)                               \
    _mm512_storeu_si512(pack_b + ((kr + 0) * NR), a_reg[0]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 1) * NR), a_reg[1]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 2) * NR), a_reg[2]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 3) * NR), a_reg[3]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 4) * NR), a_reg[4]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 5) * NR), a_reg[5]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 6) * NR), a_reg[6]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 7) * NR), a_reg[7]);

/* Store first 4 transposed rows (used for K=4 fringe) */
#define STORE_4_ROWS_FP16_AVX512(pack_b, kr, NR)                               \
    _mm512_storeu_si512(pack_b + ((kr + 0) * NR), a_reg[0]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 1) * NR), a_reg[1]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 2) * NR), a_reg[2]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 3) * NR), a_reg[3]);

/* Store first 2 transposed rows (used for K=2 fringe) */
#define STORE_2_ROWS_FP16_AVX512(pack_b, kr, NR)                               \
    _mm512_storeu_si512(pack_b + ((kr + 0) * NR), a_reg[0]);                   \
    _mm512_storeu_si512(pack_b + ((kr + 1) * NR), a_reg[1]);

/* Store all 32 transposed rows */
#define STORE_32_ROWS_FP16_AVX512(pack_b, jr, kr, NR)                          \
    _mm512_storeu_si512(pack_b + jr + ((kr + 0) * NR), a_reg[0]);              \
    _mm512_storeu_si512(pack_b + jr + ((kr + 1) * NR), a_reg[1]);              \
    _mm512_storeu_si512(pack_b + jr + ((kr + 2) * NR), a_reg[2]);              \
    _mm512_storeu_si512(pack_b + jr + ((kr + 3) * NR), a_reg[3]);              \
    _mm512_storeu_si512(pack_b + jr + ((kr + 4) * NR), a_reg[4]);              \
    _mm512_storeu_si512(pack_b + jr + ((kr + 5) * NR), a_reg[5]);              \
    _mm512_storeu_si512(pack_b + jr + ((kr + 6) * NR), a_reg[6]);              \
    _mm512_storeu_si512(pack_b + jr + ((kr + 7) * NR), a_reg[7]);              \
    _mm512_storeu_si512(pack_b + jr + ((kr + 8) * NR), a_reg[8]);              \
    _mm512_storeu_si512(pack_b + jr + ((kr + 9) * NR), a_reg[9]);              \
    _mm512_storeu_si512(pack_b + jr + ((kr + 10) * NR), a_reg[10]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 11) * NR), a_reg[11]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 12) * NR), a_reg[12]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 13) * NR), a_reg[13]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 14) * NR), a_reg[14]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 15) * NR), a_reg[15]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 16) * NR), a_reg[16]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 17) * NR), a_reg[17]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 18) * NR), a_reg[18]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 19) * NR), a_reg[19]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 20) * NR), a_reg[20]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 21) * NR), a_reg[21]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 22) * NR), a_reg[22]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 23) * NR), a_reg[23]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 24) * NR), a_reg[24]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 25) * NR), a_reg[25]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 26) * NR), a_reg[26]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 27) * NR), a_reg[27]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 28) * NR), a_reg[28]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 29) * NR), a_reg[29]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 30) * NR), a_reg[30]);            \
    _mm512_storeu_si512(pack_b + jr + ((kr + 31) * NR), a_reg[31]);

/*
 * Column-major packing for NR=128 panel
 *
 * For col-major input, we need to transpose 32 columns at a time.
 * Each column becomes a row in the K-MAJOR output layout.
 */
static void
dlp_packb_nr128_f16f16f16of16_col_major_kernel(float16*       pack_b_buffer,
                                               const float16* b,
                                               const md_t     ldb,
                                               const md_t     NC,
                                               const md_t     KC,
                                               md_t*          rs_b,
                                               md_t*          cs_b)
{
    md_t NR = 128;

    md_t n_full_pieces            = NC / NR;
    md_t n_full_pieces_loop_limit = n_full_pieces * NR;
    md_t n_partial_pieces         = NC % NR;

    /* Pack full NR=128 panels (process 32 columns at a time) */
    for (iter_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR) {
        dlp_packb_nr_mult_32_f16f16f16of16_col_major(
            pack_b_buffer + (jc * KC), b + (jc * ldb), 128, ldb, KC);
    }

    /* Handle partial panel (NC % NR != 0) */
    if (n_partial_pieces > 0) {
        md_t n0_partial_rem  = n_partial_pieces % 32;
        md_t n0_partial_pack = 0;

        /* Split into optimal chunks: 96, 64, 32, or remainder */
        md_t n0_96 = n_partial_pieces / 96;
        md_t n0_64 = n_partial_pieces / 64;
        md_t n0_32 = n_partial_pieces / 32;

        if (n0_96 == 1) {
            dlp_packb_nr_mult_32_f16f16f16of16_col_major(
                (pack_b_buffer + (n_full_pieces_loop_limit * KC)),
                (b + n_full_pieces_loop_limit * ldb), 96, ldb, KC);
            n0_partial_pack = 96;
        } else if (n0_64 == 1) {
            dlp_packb_nr_mult_32_f16f16f16of16_col_major(
                (pack_b_buffer + (n_full_pieces_loop_limit * KC)),
                (b + n_full_pieces_loop_limit * ldb), 64, ldb, KC);
            n0_partial_pack = 64;
        } else if (n0_32 == 1) {
            dlp_packb_nr_mult_32_f16f16f16of16_col_major(
                (pack_b_buffer + (n_full_pieces_loop_limit * KC)),
                (b + n_full_pieces_loop_limit * ldb), 32, ldb, KC);
            n0_partial_pack = 32;
        }

        if (n0_partial_rem > 0) {
            dlp_packb_nrlt32_f16f16f16of16_col_major_kernel(
                (pack_b_buffer + (n_full_pieces_loop_limit * KC)
                 + (n0_partial_pack * KC)),
                (b + (n_full_pieces_loop_limit + n0_partial_pack) * ldb), ldb,
                KC, n0_partial_rem);
        }
    }

    /* Output strides for K-MAJOR layout
     * rs_b = NR is the base stride for full panels.
     * The GEMM kernel adjusts this for fringe panels (similar to INT8).
     * For fringe panel at n0, kernel computes: (rs_b * n0_width) / NR
     */
    *rs_b = NR; /* Always return NR=128 as base stride */
    *cs_b = 1;  /* Stride to next n within same k-row */
}

/*
 * Column-major packing for NR multiples of 32
 *
 * This function handles 32, 64, 96, and 128-width panels by processing
 * 32 columns at a time and transposing them to K-MAJOR format.
 *
 * Uses 32x32 transpose for main loop, with vectorized K-fringe handling
 * for K=16, K=8, K=4, K=2, K=1 remainder cases.
 */
static void
dlp_packb_nr_mult_32_f16f16f16of16_col_major(float16*       pack_b_buffer,
                                             const float16* b,
                                             const md_t     NR,
                                             const md_t     ldb,
                                             const md_t     KC)
{
    __m512i a_reg[32];
    __m512i b_reg[32];

    md_t kr = 0;

    /* Main loop: Process 32 k-elements at a time (32x32 transpose) */
    for (kr = 0; (kr + 31) < KC; kr += 32) {
        for (iter_t jr = 0; jr < NR; jr += 32) {
            /* Load 32 columns, each with 32 FP16 elements */
            LOAD_32_COLS_FP16_AVX512(b + (jr * ldb), ldb, kr);

            /* Transpose 32x32 block */
            TRANSPOSE_32x32_FP16_AVX512

                /* Store transposed result in K-MAJOR layout */
                STORE_32_ROWS_FP16_AVX512(pack_b_buffer, jr, kr, NR);
        }
    }

    /* K-fringe: Handle remaining k-elements with vectorized 32x32 transpose
     * The transpose still works correctly with masked loads (zeros in upper
     * k-positions will transpose to zeros in corresponding output positions).
     * We only store the valid k-rows.
     */

    /* K=16 fringe */
    for (; (kr + 15) < KC; kr += 16) {
        for (iter_t jr = 0; jr < NR; jr += 32) {
            /* Masked load: only 16 k-elements per column (lower 16 bits) */
            MASK_LOAD_32_COLS_FP16_AVX512(b + (jr * ldb), ldb, kr, 0xFFFF);

            /* Full 32x32 transpose (upper 16 positions are zero) */
            TRANSPOSE_32x32_FP16_AVX512

                /* Store only first 16 transposed rows */
                _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 0) * NR),
                                    a_reg[0]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 1) * NR), a_reg[1]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 2) * NR), a_reg[2]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 3) * NR), a_reg[3]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 4) * NR), a_reg[4]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 5) * NR), a_reg[5]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 6) * NR), a_reg[6]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 7) * NR), a_reg[7]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 8) * NR), a_reg[8]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 9) * NR), a_reg[9]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 10) * NR),
                                a_reg[10]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 11) * NR),
                                a_reg[11]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 12) * NR),
                                a_reg[12]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 13) * NR),
                                a_reg[13]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 14) * NR),
                                a_reg[14]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 15) * NR),
                                a_reg[15]);
        }
    }

    /* K=8 fringe */
    for (; (kr + 7) < KC; kr += 8) {
        for (iter_t jr = 0; jr < NR; jr += 32) {
            /* Masked load: only 8 k-elements per column */
            MASK_LOAD_32_COLS_FP16_AVX512(b + (jr * ldb), ldb, kr, 0xFF);

            TRANSPOSE_32x32_FP16_AVX512

                /* Store only first 8 transposed rows */
                _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 0) * NR),
                                    a_reg[0]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 1) * NR), a_reg[1]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 2) * NR), a_reg[2]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 3) * NR), a_reg[3]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 4) * NR), a_reg[4]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 5) * NR), a_reg[5]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 6) * NR), a_reg[6]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 7) * NR), a_reg[7]);
        }
    }

    /* K=4 fringe */
    for (; (kr + 3) < KC; kr += 4) {
        for (iter_t jr = 0; jr < NR; jr += 32) {
            /* Masked load: only 4 k-elements per column */
            MASK_LOAD_32_COLS_FP16_AVX512(b + (jr * ldb), ldb, kr, 0xF);

            TRANSPOSE_32x32_FP16_AVX512

                /* Store only first 4 transposed rows */
                _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 0) * NR),
                                    a_reg[0]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 1) * NR), a_reg[1]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 2) * NR), a_reg[2]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 3) * NR), a_reg[3]);
        }
    }

    /* K=2 fringe */
    for (; (kr + 1) < KC; kr += 2) {
        for (iter_t jr = 0; jr < NR; jr += 32) {
            /* Masked load: only 2 k-elements per column */
            MASK_LOAD_32_COLS_FP16_AVX512(b + (jr * ldb), ldb, kr, 0x3);

            TRANSPOSE_32x32_FP16_AVX512

                /* Store only first 2 transposed rows */
                _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 0) * NR),
                                    a_reg[0]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 1) * NR), a_reg[1]);
        }
    }

    /* K=1 fringe */
    for (; kr < KC; kr += 1) {
        for (iter_t jr = 0; jr < NR; jr += 32) {
            /* Masked load: only 1 k-element per column */
            MASK_LOAD_32_COLS_FP16_AVX512(b + (jr * ldb), ldb, kr, 0x1);

            TRANSPOSE_32x32_FP16_AVX512

                /* Store only first transposed row */
                _mm512_storeu_si512(pack_b_buffer + jr + (kr * NR), a_reg[0]);
        }
    }
}

/* Masked load macro for N-fringe columns (loads n_count columns, zeros rest)
 * Loads 32 k-elements per active column.
 */
#define LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n_count)                   \
    for (iter_t _jr = 0; _jr < n_count; _jr++) {                               \
        a_reg[_jr] = _mm512_loadu_si512(b + (ldb * _jr) + kr);                 \
    }                                                                          \
    for (iter_t _jr = n_count; _jr < 32; _jr++) {                              \
        a_reg[_jr] = _mm512_setzero_si512();                                   \
    }

/* Masked load for N-fringe columns with K-mask (for K-fringe within N-fringe)
 */
#define MASK_LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n_count, k_mask)      \
    for (iter_t _jr = 0; _jr < n_count; _jr++) {                               \
        a_reg[_jr] = _mm512_maskz_loadu_epi16(k_mask, b + (ldb * _jr) + kr);   \
    }                                                                          \
    for (iter_t _jr = n_count; _jr < 32; _jr++) {                              \
        a_reg[_jr] = _mm512_setzero_si512();                                   \
    }

/*
 * Column-major packing for NR<32 fringe panel
 * Pads to NR=32 with zeros
 *
 * Uses 32x32 transpose with zero-padded columns for N-fringe,
 * and masked loads for K-fringe handling.
 */
static void
dlp_packb_nrlt32_f16f16f16of16_col_major_kernel(float16*       pack_b_buffer,
                                                const float16* b,
                                                const md_t     ldb,
                                                const md_t     KC,
                                                const md_t     n0_partial_rem)
{
    md_t NR = 32;

    __m512i a_reg[32];
    __m512i b_reg[32];

    md_t kr = 0;

    /* Main loop: Process 32 k-elements at a time with N-fringe */
    for (kr = 0; (kr + 31) < KC; kr += 32) {
        /* Load partial columns, zero-pad the rest */
        LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n0_partial_rem);

        TRANSPOSE_32x32_FP16_AVX512

            _mm512_storeu_si512(pack_b_buffer + ((kr + 0) * NR), a_reg[0]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 1) * NR), a_reg[1]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 2) * NR), a_reg[2]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 3) * NR), a_reg[3]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 4) * NR), a_reg[4]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 5) * NR), a_reg[5]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 6) * NR), a_reg[6]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 7) * NR), a_reg[7]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 8) * NR), a_reg[8]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 9) * NR), a_reg[9]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 10) * NR), a_reg[10]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 11) * NR), a_reg[11]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 12) * NR), a_reg[12]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 13) * NR), a_reg[13]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 14) * NR), a_reg[14]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 15) * NR), a_reg[15]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 16) * NR), a_reg[16]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 17) * NR), a_reg[17]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 18) * NR), a_reg[18]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 19) * NR), a_reg[19]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 20) * NR), a_reg[20]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 21) * NR), a_reg[21]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 22) * NR), a_reg[22]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 23) * NR), a_reg[23]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 24) * NR), a_reg[24]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 25) * NR), a_reg[25]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 26) * NR), a_reg[26]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 27) * NR), a_reg[27]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 28) * NR), a_reg[28]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 29) * NR), a_reg[29]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 30) * NR), a_reg[30]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 31) * NR), a_reg[31]);
    }

    /* K=16 fringe with N-fringe */
    for (; (kr + 15) < KC; kr += 16) {
        MASK_LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n0_partial_rem,
                                             0xFFFF);

        TRANSPOSE_32x32_FP16_AVX512

            _mm512_storeu_si512(pack_b_buffer + ((kr + 0) * NR), a_reg[0]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 1) * NR), a_reg[1]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 2) * NR), a_reg[2]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 3) * NR), a_reg[3]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 4) * NR), a_reg[4]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 5) * NR), a_reg[5]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 6) * NR), a_reg[6]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 7) * NR), a_reg[7]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 8) * NR), a_reg[8]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 9) * NR), a_reg[9]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 10) * NR), a_reg[10]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 11) * NR), a_reg[11]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 12) * NR), a_reg[12]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 13) * NR), a_reg[13]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 14) * NR), a_reg[14]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 15) * NR), a_reg[15]);
    }

    /* K=8 fringe with N-fringe */
    for (; (kr + 7) < KC; kr += 8) {
        MASK_LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n0_partial_rem, 0xFF);

        TRANSPOSE_32x32_FP16_AVX512

            _mm512_storeu_si512(pack_b_buffer + ((kr + 0) * NR), a_reg[0]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 1) * NR), a_reg[1]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 2) * NR), a_reg[2]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 3) * NR), a_reg[3]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 4) * NR), a_reg[4]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 5) * NR), a_reg[5]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 6) * NR), a_reg[6]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 7) * NR), a_reg[7]);
    }

    /* K=4 fringe with N-fringe */
    for (; (kr + 3) < KC; kr += 4) {
        MASK_LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n0_partial_rem, 0xF);

        TRANSPOSE_32x32_FP16_AVX512

            _mm512_storeu_si512(pack_b_buffer + ((kr + 0) * NR), a_reg[0]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 1) * NR), a_reg[1]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 2) * NR), a_reg[2]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 3) * NR), a_reg[3]);
    }

    /* K=2 fringe with N-fringe */
    for (; (kr + 1) < KC; kr += 2) {
        MASK_LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n0_partial_rem, 0x3);

        TRANSPOSE_32x32_FP16_AVX512

            _mm512_storeu_si512(pack_b_buffer + ((kr + 0) * NR), a_reg[0]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 1) * NR), a_reg[1]);
    }

    /* K=1 fringe with N-fringe */
    for (; kr < KC; kr += 1) {
        MASK_LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n0_partial_rem, 0x1);

        TRANSPOSE_32x32_FP16_AVX512

            _mm512_storeu_si512(pack_b_buffer + (kr * NR), a_reg[0]);
    }
}
