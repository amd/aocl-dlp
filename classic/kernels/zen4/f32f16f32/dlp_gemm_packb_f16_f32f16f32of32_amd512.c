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
 * F32×FP16→F32 AVX-512 Optimized Pack B Kernel (NR=64 configuration)
 *
 * B matrix is FP16 (2 bytes per element). Packing keeps B in FP16 format.
 * Conversion to F32 happens in the JIT-generated micro-kernel via vcvtph2ps.
 *
 * NR=64 because output C is F32 (4 bytes), so 64 F32 outputs = 4 ZMM regs.
 * Each ZMM holds 32 FP16 elements, so NR=64 = 2 ZMM loads.
 *
 * K-MAJOR layout: packed[k*NR + n], no interleaving or k-padding.
 * Chunk hierarchy: 64 -> 32 -> lt32 (padded to 32)
 */

#include "classic/aocl_fp16_type.h"
#include "kernels/dlp_kernels.h"
#include <immintrin.h>
#include <string.h>

/* Forward declarations - Row-major kernels */
static void
dlp_packb_nr64_f32f16f32of32_row_major_kernel(float16*       pack_b_buffer,
                                              const float16* b,
                                              const md_t     ldb,
                                              const md_t     NC,
                                              const md_t     KC,
                                              md_t*          rs_b,
                                              md_t*          cs_b);

static void
dlp_packb_nr32_f32f16f32of32_row_major_kernel(float16*       pack_b_buffer,
                                              const float16* b,
                                              const md_t     ldb,
                                              const md_t     KC);

static void
dlp_packb_nrlt32_f32f16f32of32_row_major_kernel(float16*       pack_b_buffer,
                                                const float16* b,
                                                const md_t     ldb,
                                                const md_t     KC,
                                                const md_t     n0_partial_rem);

/* Forward declarations - Column-major kernels (for transposed B) */
static void
dlp_packb_nr64_f32f16f32of32_col_major_kernel(float16*       pack_b_buffer,
                                              const float16* b,
                                              const md_t     ldb,
                                              const md_t     NC,
                                              const md_t     KC,
                                              md_t*          rs_b,
                                              md_t*          cs_b);

static void
dlp_packb_nr_mult_32_f32f16f32of32_col_major(float16*       pack_b_buffer,
                                             const float16* b,
                                             const md_t     NR,
                                             const md_t     ldb,
                                             const md_t     KC);

static void
dlp_packb_nrlt32_f32f16f32of32_col_major_kernel(float16*       pack_b_buffer,
                                                const float16* b,
                                                const md_t     ldb,
                                                const md_t     KC,
                                                const md_t     n0_partial_rem);

static void
dlp_packb_nrlt32_f32f16f32of32_col_major_kernel_stride(
    float16*       pack_b_buffer,
    const float16* b,
    const md_t     ldb,
    const md_t     KC,
    const md_t     n0_partial_rem,
    const md_t     out_stride);

/* ============================================================================
 * Main dispatcher function for NR=64 F32×FP16→F32 Pack B
 * ============================================================================
 */
void
dlp_packb_nr64_f32f16f32of32(float16*       pack_b_buffer,
                             const float16* b,
                             const md_t     rs_b,
                             const md_t     cs_b,
                             const md_t     NC,
                             const md_t     KC,
                             md_t*          rs_p,
                             md_t*          cs_p)
{
    if (cs_b == 1) {
        dlp_packb_nr64_f32f16f32of32_row_major_kernel(pack_b_buffer, b, rs_b,
                                                      NC, KC, rs_p, cs_p);
    } else {
        dlp_packb_nr64_f32f16f32of32_col_major_kernel(pack_b_buffer, b, cs_b,
                                                      NC, KC, rs_p, cs_p);
    }
}

/* ============================================================================
 * Row-major packing kernels
 * ============================================================================
 */

/*
 * Row-major packing for NR=64 panel (2 ZMM registers)
 *
 * Input:  B[K][N] row-major (ldb = N stride), FP16 elements
 * Output: packed[k*NR + n] K-MAJOR layout, FP16 elements
 */
static void
dlp_packb_nr64_f32f16f32of32_row_major_kernel(float16*       pack_b_buffer,
                                              const float16* b,
                                              const md_t     ldb,
                                              const md_t     NC,
                                              const md_t     KC,
                                              md_t*          rs_b,
                                              md_t*          cs_b)
{
    md_t NR = 64;

    md_t n_full_pieces            = NC / NR;
    md_t n_full_pieces_loop_limit = n_full_pieces * NR;
    md_t n_partial_pieces         = NC % NR;

    __m512i a0, a1;

    /* Pack full NR=64 panels (2 ZMM loads per k-row) */
    for (iter_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR) {
        for (iter_t kr = 0; kr < KC; kr++) {
            a0 = _mm512_loadu_si512(b + (ldb * kr) + jc);
            a1 = _mm512_loadu_si512(b + (ldb * kr) + jc + 32);

            _mm512_storeu_si512(pack_b_buffer + (jc * KC) + (kr * NR), a0);
            _mm512_storeu_si512(pack_b_buffer + (jc * KC) + (kr * NR) + 32, a1);
        }
    }

    /*
     * Handle partial panel (NC % NR != 0).
     *
     * CRITICAL: Pack fringe panels at stride NR=64 (same as full panels).
     * The JIT kernel loads B in groups of 16 FP16 (converting to 16 F32 per
     * ZMM) and advances the B pointer by rsB (=NR=64 FP16 elements per k-row).
     * If we stored fringe panels at stride 32 (the old sub-panel approach),
     * the kernel would read incorrect data for any k-row after the first.
     *
     * Layout: each k-row always occupies NR=64 FP16 slots (128 bytes).
     *   - Columns 0..31:  loaded via ZMM (32 FP16) or masked
     *   - Columns 32..63: loaded via ZMM (32 FP16) or masked or zeroed
     *   - Unused columns are zero-padded (from maskz loads)
     */
    if (n_partial_pieces > 0) {
        float16* panel_buf = pack_b_buffer + (n_full_pieces_loop_limit * KC);
        const float16* b_panel = b + n_full_pieces_loop_limit;

        md_t n0_32          = n_partial_pieces / 32;
        md_t n0_partial_rem = n_partial_pieces % 32;

        if (n0_32 == 1 && n0_partial_rem == 0) {
            /* Exactly 32 columns: load first half, zero second half */
            __m512i zero = _mm512_setzero_si512();
            for (iter_t kr = 0; kr < KC; kr++) {
                a0 = _mm512_loadu_si512(b_panel + (ldb * kr));
                _mm512_storeu_si512(panel_buf + (kr * NR), a0);
                _mm512_storeu_si512(panel_buf + (kr * NR) + 32, zero);
            }
        } else if (n0_32 == 1 && n0_partial_rem > 0) {
            /* 33..63 columns: load first 32, masked load remainder */
            __mmask32 rem_mask = (__mmask32)((1ULL << n0_partial_rem) - 1);
            for (iter_t kr = 0; kr < KC; kr++) {
                a0 = _mm512_loadu_si512(b_panel + (ldb * kr));
                a1 = _mm512_maskz_loadu_epi16(rem_mask,
                                              b_panel + 32 + (ldb * kr));
                _mm512_storeu_si512(panel_buf + (kr * NR), a0);
                _mm512_storeu_si512(panel_buf + (kr * NR) + 32, a1);
            }
        } else {
            /* 1..31 columns: masked load first half, zero second half */
            __mmask32 rem_mask = (__mmask32)((1ULL << n0_partial_rem) - 1);
            __m512i   zero     = _mm512_setzero_si512();
            for (iter_t kr = 0; kr < KC; kr++) {
                a0 = _mm512_maskz_loadu_epi16(rem_mask, b_panel + (ldb * kr));
                _mm512_storeu_si512(panel_buf + (kr * NR), a0);
                _mm512_storeu_si512(panel_buf + (kr * NR) + 32, zero);
            }
        }
    }

    *rs_b = NR;
    *cs_b = 1;
}

/*
 * Row-major packing for NR=32 fringe panel (1 ZMM register)
 */
static void
dlp_packb_nr32_f32f16f32of32_row_major_kernel(float16*       pack_b_buffer,
                                              const float16* b,
                                              const md_t     ldb,
                                              const md_t     KC)
{
    md_t NR = 32;

    __m512i a0;

    for (iter_t kr = 0; kr < KC; kr++) {
        a0 = _mm512_loadu_si512(b + (ldb * kr));
        _mm512_storeu_si512(pack_b_buffer + (kr * NR), a0);
    }
}

/*
 * Row-major packing for NR<32 fringe panel with masking
 * Pads to NR=32 with zeros
 */
static void
dlp_packb_nrlt32_f32f16f32of32_row_major_kernel(float16*       pack_b_buffer,
                                                const float16* b,
                                                const md_t     ldb,
                                                const md_t     KC,
                                                const md_t     n0_partial_rem)
{
    md_t NR = 32;

    __m512i a0;

    __mmask32 load_mask = (__mmask32)((1ULL << n0_partial_rem) - 1);

    for (iter_t kr = 0; kr < KC; kr++) {
        a0 = _mm512_maskz_loadu_epi16(load_mask, b + (ldb * kr));
        _mm512_storeu_si512(pack_b_buffer + (kr * NR), a0);
    }
}

/* ============================================================================
 * Column-major packing kernels (for transposed B: rs_b=1, cs_b=ldb)
 *
 * When B is transposed, the layout is effectively column-major from the
 * packing perspective. These kernels perform a 32x32 FP16 transpose to
 * produce the same K-major packed layout as the row-major path.
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

#define TRANSPOSE_32x32_FP16_AVX512                                            \
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
    for (iter_t _i = 0; _i < 32; _i++) {                                       \
        a_reg[_i] = b_reg[_i];                                                 \
    }

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

#define LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n_count)                   \
    for (iter_t _jr = 0; _jr < n_count; _jr++) {                               \
        a_reg[_jr] = _mm512_loadu_si512(b + (ldb * _jr) + kr);                 \
    }                                                                          \
    for (iter_t _jr = n_count; _jr < 32; _jr++) {                              \
        a_reg[_jr] = _mm512_setzero_si512();                                   \
    }

#define MASK_LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n_count, k_mask)      \
    for (iter_t _jr = 0; _jr < n_count; _jr++) {                               \
        a_reg[_jr] = _mm512_maskz_loadu_epi16(k_mask, b + (ldb * _jr) + kr);   \
    }                                                                          \
    for (iter_t _jr = n_count; _jr < 32; _jr++) {                              \
        a_reg[_jr] = _mm512_setzero_si512();                                   \
    }

/*
 * Column-major packing for NR=64 panel
 */
static void
dlp_packb_nr64_f32f16f32of32_col_major_kernel(float16*       pack_b_buffer,
                                              const float16* b,
                                              const md_t     ldb,
                                              const md_t     NC,
                                              const md_t     KC,
                                              md_t*          rs_b,
                                              md_t*          cs_b)
{
    md_t NR = 64;

    md_t n_full_pieces            = NC / NR;
    md_t n_full_pieces_loop_limit = n_full_pieces * NR;
    md_t n_partial_pieces         = NC % NR;

    for (iter_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR) {
        dlp_packb_nr_mult_32_f32f16f32of32_col_major(
            pack_b_buffer + (jc * KC), b + (jc * ldb), 64, ldb, KC);
    }

    /*
     * Handle partial panel at stride NR=64, matching full panel layout.
     * See row-major version for detailed rationale.
     *
     * Zero-fill the entire partial panel first, then overwrite valid
     * columns. Each k-row is NR=64 FP16 elements wide in the output.
     * The first 32 columns go to offsets [0..31], the next <32 to [32..63].
     */
    if (n_partial_pieces > 0) {
        float16* panel_buf = pack_b_buffer + (n_full_pieces_loop_limit * KC);
        md_t     n0_32     = n_partial_pieces / 32;
        md_t     n0_partial_rem = n_partial_pieces % 32;

        /* Zero the entire partial panel buffer (NR=64 per k-row) */
        memset(panel_buf, 0, sizeof(float16) * NR * KC);

        if (n0_32 == 1) {
            /* Pack first 32 columns using stride-aware version */
            dlp_packb_nrlt32_f32f16f32of32_col_major_kernel_stride(
                panel_buf, (b + n_full_pieces_loop_limit * ldb), ldb, KC, 32,
                NR);
        }

        if (n0_partial_rem > 0) {
            /* Pack remaining <32 columns at offset +32 if first 32 exist */
            md_t col_offset = n0_32 * 32;
            dlp_packb_nrlt32_f32f16f32of32_col_major_kernel_stride(
                panel_buf + col_offset,
                (b + (n_full_pieces_loop_limit + col_offset) * ldb), ldb, KC,
                n0_partial_rem, NR);
        }
    }

    *rs_b = NR;
    *cs_b = 1;
}

/*
 * Column-major packing for NR multiples of 32
 */
static void
dlp_packb_nr_mult_32_f32f16f32of32_col_major(float16*       pack_b_buffer,
                                             const float16* b,
                                             const md_t     NR,
                                             const md_t     ldb,
                                             const md_t     KC)
{
    __m512i a_reg[32];
    __m512i b_reg[32];

    md_t kr = 0;

    for (kr = 0; (kr + 31) < KC; kr += 32) {
        for (iter_t jr = 0; jr < NR; jr += 32) {
            LOAD_32_COLS_FP16_AVX512(b + (jr * ldb), ldb, kr);
            TRANSPOSE_32x32_FP16_AVX512 STORE_32_ROWS_FP16_AVX512(pack_b_buffer,
                                                                  jr, kr, NR);
        }
    }

    for (; (kr + 15) < KC; kr += 16) {
        for (iter_t jr = 0; jr < NR; jr += 32) {
            MASK_LOAD_32_COLS_FP16_AVX512(b + (jr * ldb), ldb, kr, 0xFFFF);
            TRANSPOSE_32x32_FP16_AVX512 for (iter_t _i = 0; _i < 16; _i++)
            {
                _mm512_storeu_si512(pack_b_buffer + jr + ((kr + _i) * NR),
                                    a_reg[_i]);
            }
        }
    }

    for (; (kr + 7) < KC; kr += 8) {
        for (iter_t jr = 0; jr < NR; jr += 32) {
            MASK_LOAD_32_COLS_FP16_AVX512(b + (jr * ldb), ldb, kr, 0xFF);
            TRANSPOSE_32x32_FP16_AVX512 for (iter_t _i = 0; _i < 8; _i++)
            {
                _mm512_storeu_si512(pack_b_buffer + jr + ((kr + _i) * NR),
                                    a_reg[_i]);
            }
        }
    }

    for (; (kr + 3) < KC; kr += 4) {
        for (iter_t jr = 0; jr < NR; jr += 32) {
            MASK_LOAD_32_COLS_FP16_AVX512(b + (jr * ldb), ldb, kr, 0xF);
            TRANSPOSE_32x32_FP16_AVX512 for (iter_t _i = 0; _i < 4; _i++)
            {
                _mm512_storeu_si512(pack_b_buffer + jr + ((kr + _i) * NR),
                                    a_reg[_i]);
            }
        }
    }

    for (; (kr + 1) < KC; kr += 2) {
        for (iter_t jr = 0; jr < NR; jr += 32) {
            MASK_LOAD_32_COLS_FP16_AVX512(b + (jr * ldb), ldb, kr, 0x3);
            TRANSPOSE_32x32_FP16_AVX512 _mm512_storeu_si512(
                pack_b_buffer + jr + ((kr + 0) * NR), a_reg[0]);
            _mm512_storeu_si512(pack_b_buffer + jr + ((kr + 1) * NR), a_reg[1]);
        }
    }

    for (; kr < KC; kr += 1) {
        for (iter_t jr = 0; jr < NR; jr += 32) {
            MASK_LOAD_32_COLS_FP16_AVX512(b + (jr * ldb), ldb, kr, 0x1);
            TRANSPOSE_32x32_FP16_AVX512 _mm512_storeu_si512(
                pack_b_buffer + jr + (kr * NR), a_reg[0]);
        }
    }
}

/*
 * Column-major packing for NR<32 fringe panel
 */
static void
dlp_packb_nrlt32_f32f16f32of32_col_major_kernel(float16*       pack_b_buffer,
                                                const float16* b,
                                                const md_t     ldb,
                                                const md_t     KC,
                                                const md_t     n0_partial_rem)
{
    md_t NR = 32;

    __m512i a_reg[32];
    __m512i b_reg[32];

    md_t kr = 0;

    for (kr = 0; (kr + 31) < KC; kr += 32) {
        LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n0_partial_rem);
        TRANSPOSE_32x32_FP16_AVX512 for (iter_t _i = 0; _i < 32; _i++)
        {
            _mm512_storeu_si512(pack_b_buffer + ((kr + _i) * NR), a_reg[_i]);
        }
    }

    for (; (kr + 15) < KC; kr += 16) {
        MASK_LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n0_partial_rem,
                                             0xFFFF);
        TRANSPOSE_32x32_FP16_AVX512 for (iter_t _i = 0; _i < 16; _i++)
        {
            _mm512_storeu_si512(pack_b_buffer + ((kr + _i) * NR), a_reg[_i]);
        }
    }

    for (; (kr + 7) < KC; kr += 8) {
        MASK_LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n0_partial_rem, 0xFF);
        TRANSPOSE_32x32_FP16_AVX512 for (iter_t _i = 0; _i < 8; _i++)
        {
            _mm512_storeu_si512(pack_b_buffer + ((kr + _i) * NR), a_reg[_i]);
        }
    }

    for (; (kr + 3) < KC; kr += 4) {
        MASK_LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n0_partial_rem, 0xF);
        TRANSPOSE_32x32_FP16_AVX512 for (iter_t _i = 0; _i < 4; _i++)
        {
            _mm512_storeu_si512(pack_b_buffer + ((kr + _i) * NR), a_reg[_i]);
        }
    }

    for (; (kr + 1) < KC; kr += 2) {
        MASK_LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n0_partial_rem, 0x3);
        TRANSPOSE_32x32_FP16_AVX512 _mm512_storeu_si512(
            pack_b_buffer + ((kr + 0) * NR), a_reg[0]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 1) * NR), a_reg[1]);
    }

    for (; kr < KC; kr += 1) {
        MASK_LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n0_partial_rem, 0x1);
        TRANSPOSE_32x32_FP16_AVX512 _mm512_storeu_si512(
            pack_b_buffer + (kr * NR), a_reg[0]);
    }
}

/*
 * Column-major packing for NR<32 fringe panel with custom output stride.
 * Same as packb_nrlt32_f32f16f32of32_col_major_kernel but uses out_stride
 * instead of hardcoded NR=32. This allows packing <32 columns into a
 * wider NR=64 stride layout for the F32×FP16 kernel.
 */
static void
dlp_packb_nrlt32_f32f16f32of32_col_major_kernel_stride(
    float16*       pack_b_buffer,
    const float16* b,
    const md_t     ldb,
    const md_t     KC,
    const md_t     n0_partial_rem,
    const md_t     out_stride)
{
    __m512i a_reg[32];
    __m512i b_reg[32];

    md_t kr = 0;

    for (kr = 0; (kr + 31) < KC; kr += 32) {
        LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n0_partial_rem);
        TRANSPOSE_32x32_FP16_AVX512 for (iter_t _i = 0; _i < 32; _i++)
        {
            _mm512_storeu_si512(pack_b_buffer + ((kr + _i) * out_stride),
                                a_reg[_i]);
        }
    }

    for (; (kr + 15) < KC; kr += 16) {
        MASK_LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n0_partial_rem,
                                             0xFFFF);
        TRANSPOSE_32x32_FP16_AVX512 for (iter_t _i = 0; _i < 16; _i++)
        {
            _mm512_storeu_si512(pack_b_buffer + ((kr + _i) * out_stride),
                                a_reg[_i]);
        }
    }

    for (; (kr + 7) < KC; kr += 8) {
        MASK_LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n0_partial_rem, 0xFF);
        TRANSPOSE_32x32_FP16_AVX512 for (iter_t _i = 0; _i < 8; _i++)
        {
            _mm512_storeu_si512(pack_b_buffer + ((kr + _i) * out_stride),
                                a_reg[_i]);
        }
    }

    for (; (kr + 3) < KC; kr += 4) {
        MASK_LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n0_partial_rem, 0xF);
        TRANSPOSE_32x32_FP16_AVX512 for (iter_t _i = 0; _i < 4; _i++)
        {
            _mm512_storeu_si512(pack_b_buffer + ((kr + _i) * out_stride),
                                a_reg[_i]);
        }
    }

    for (; (kr + 1) < KC; kr += 2) {
        MASK_LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n0_partial_rem, 0x3);
        TRANSPOSE_32x32_FP16_AVX512 _mm512_storeu_si512(
            pack_b_buffer + ((kr + 0) * out_stride), a_reg[0]);
        _mm512_storeu_si512(pack_b_buffer + ((kr + 1) * out_stride), a_reg[1]);
    }

    for (; kr < KC; kr += 1) {
        MASK_LOAD_N_PARTIAL_COLS_FP16_AVX512(b, ldb, kr, n0_partial_rem, 0x1);
        TRANSPOSE_32x32_FP16_AVX512 _mm512_storeu_si512(
            pack_b_buffer + (kr * out_stride), a_reg[0]);
    }
}
