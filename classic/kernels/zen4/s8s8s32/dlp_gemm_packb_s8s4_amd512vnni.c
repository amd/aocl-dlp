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

/*
 * Fused s8s4 pack-B kernel (widen-on-load) for the symmetric-quant path.
 *
 * This is a fork of dlp_gemm_packb_s8_amd512vnni.c. The rearrange/VNNI layout,
 * per-group column-sum accumulation and store logic are byte-identical to the
 * proven s8 packer; the ONLY change is at the B loads: instead of reading s8
 * bytes, we load nibble-packed s4 and widen each row to s8 in-register (via the
 * int4 multishift utilities, sign-extended to [-8, 7]) before feeding the exact
 * same code. This removes the intermediate s8 scratch buffer entirely -- widen
 * and pack happen in a single pass.
 *
 * Nibble addressing: b is a signed-4-bit, nibble-packed matrix; rs_b/cs_b are
 * element strides measured in NIBBLE units. A source element at (row, col) is
 * nibble index (rs_b*row + cs_b*col); its byte is (idx >> 1) and it is the low
 * nibble when idx is even, the high nibble when odd. When a row's starting
 * nibble index is odd, the AVX-512 multishift "_ODD" variant is used to realign
 * the extracted stream (a masked +1 byte load supplies the trailing element;
 * for partial (<16-col) panels the trailing element is masked away, so that
 * extra load is skipped).
 *
 * Compiled with zen4 arch flags; callers live behind DLP_KERNELS_ZEN4.
 */

#include "kernels/dlp_kernels.h"
#include "kernels/s8s8s32/dlp_gemm_packb_s8s4.h"
#include <immintrin.h>
#include <string.h>

#include "../int4_utils_avx512.h"

/* ---- row-major widen-on-load helpers (produce the same reg the s8 packer's
 * plain load produced, but from nibble-packed s4). Rely on b_s4, the shift/
 * conv/sign selectors being in scope at the call site. ---- */

/* 64 s8 (<- 64 nibbles) into __m512i out. */
#define WIDEN64_ROW(out, nib)                                                  \
    do {                                                                       \
        md_t _nb = (md_t)(nib);                                                \
        if ((_nb & 1) == 0) {                                                  \
            __m256i _h =                                                       \
                _mm256_maskz_loadu_epi8(0xFFFFFFFF, b_s4 + (_nb >> 1));        \
            CVT_INT4_TO_INT8_64ELEM_MULTISHIFT(_h, out, shift_idx_64,          \
                                               sign_comp_64, TRUE);            \
        } else {                                                               \
            __m256i _h =                                                       \
                _mm256_maskz_loadu_epi8(0xFFFFFFFF, b_s4 + (_nb >> 1));        \
            __m256i _hl =                                                      \
                _mm256_maskz_loadu_epi8(0x80000000, b_s4 + (_nb >> 1) + 1);    \
            CVT_INT4_TO_INT8_64ELEM_MULTISHIFT_ODD(_h, _hl, out, shift_idx_64, \
                                                   conv_shift_64,              \
                                                   sign_comp_64, TRUE);        \
        }                                                                      \
    } while (0)

/* 32 s8 (<- 32 nibbles) into __m256i out. */
#define WIDEN32_ROW(out, nib)                                                  \
    do {                                                                       \
        md_t _nb = (md_t)(nib);                                                \
        if ((_nb & 1) == 0) {                                                  \
            __m128i _h = _mm_maskz_loadu_epi8(0xFFFF, b_s4 + (_nb >> 1));      \
            CVT_INT4_TO_INT8_32ELEM_MULTISHIFT(_h, out, shift_idx_32,          \
                                               sign_comp_32, TRUE);            \
        } else {                                                               \
            __m128i _h  = _mm_maskz_loadu_epi8(0xFFFF, b_s4 + (_nb >> 1));     \
            __m128i _hl = _mm_maskz_loadu_epi8(0x8000, b_s4 + (_nb >> 1) + 1); \
            CVT_INT4_TO_INT8_32ELEM_MULTISHIFT_ODD(_h, _hl, out, shift_idx_32, \
                                                   conv_shift_32,              \
                                                   sign_comp_32, TRUE);        \
        }                                                                      \
    } while (0)

/* 16 s8 (<- 16 nibbles) into __m128i out (full 16 columns present). */
#define WIDEN16_ROW(out, nib)                                                  \
    do {                                                                       \
        md_t _nb = (md_t)(nib);                                                \
        if ((_nb & 1) == 0) {                                                  \
            __m128i _h = _mm_maskz_loadu_epi8(0xFF, b_s4 + (_nb >> 1));        \
            CVT_INT4_TO_INT8_16ELEM_MULTISHIFT(_h, out, shift_idx_16,          \
                                               sign_comp_16, TRUE);            \
        } else {                                                               \
            __m128i _h  = _mm_maskz_loadu_epi8(0xFF, b_s4 + (_nb >> 1));       \
            __m128i _hl = _mm_maskz_loadu_epi8(0x80, b_s4 + (_nb >> 1) + 1);   \
            CVT_INT4_TO_INT8_16ELEM_MULTISHIFT_ODD(_h, _hl, out, shift_idx_16, \
                                                   conv_shift_16,              \
                                                   sign_comp_16, TRUE);        \
        }                                                                      \
    } while (0)

/* < 16 columns present: load only the bytes that hold the `rem` valid nibbles
 * (masked loads suppress faults past the row) and zero the rest. The trailing
 * ODD element (lane 15) is always masked off, so no +1 byte load is needed. */
#define WIDEN16_ROW_PARTIAL(out, nib, rem)                                      \
    do {                                                                        \
        md_t      _nb    = (md_t)(nib);                                         \
        md_t      _rem   = (md_t)(rem);                                         \
        __mmask16 _emask = _cvtu32_mask16((uint32_t)0xFFFF >> (16 - _rem));     \
        if ((_nb & 1) == 0) {                                                   \
            md_t      _nb2   = (_rem + 1) >> 1;                                 \
            __mmask16 _bmask = _cvtu32_mask16((uint32_t)(1u << _nb2) - 1u);     \
            __m128i   _h     = _mm_maskz_loadu_epi8(_bmask, b_s4 + (_nb >> 1)); \
            CVT_INT4_TO_INT8_16ELEM_MULTISHIFT(_h, out, shift_idx_16,           \
                                               sign_comp_16, TRUE);             \
        } else {                                                                \
            md_t      _nb2   = (_rem >> 1) + 1;                                 \
            __mmask16 _bmask = _cvtu32_mask16((uint32_t)(1u << _nb2) - 1u);     \
            __m128i   _h     = _mm_maskz_loadu_epi8(_bmask, b_s4 + (_nb >> 1)); \
            __m128i   _hl    = _mm_setzero_si128();                             \
            CVT_INT4_TO_INT8_16ELEM_MULTISHIFT_ODD(_h, _hl, out, shift_idx_16,  \
                                                   conv_shift_16,               \
                                                   sign_comp_16, TRUE);         \
        }                                                                       \
        out = _mm_maskz_mov_epi8(_emask, out);                                  \
    } while (0)

/* Declarations for the internal row-major fringe packers. */
static void
dlp_packb_nr48_s8s4_row(int8_t*        pack_b,
                        int32_t*       colsum,
                        const uint8_t* b_s4,
                        const md_t     rs_b,
                        const md_t     row0,
                        const md_t     col0,
                        const md_t     KC);
static void
dlp_packb_nr32_s8s4_row(int8_t*        pack_b,
                        int32_t*       colsum,
                        const uint8_t* b_s4,
                        const md_t     rs_b,
                        const md_t     row0,
                        const md_t     col0,
                        const md_t     KC);
static void
dlp_packb_nr16_s8s4_row(int8_t*        pack_b,
                        int32_t*       colsum,
                        const uint8_t* b_s4,
                        const md_t     rs_b,
                        const md_t     row0,
                        const md_t     col0,
                        const md_t     KC);
static void
dlp_packb_nrlt16_s8s4_row(int8_t*        pack_b,
                          int32_t*       colsum,
                          const uint8_t* b_s4,
                          const md_t     rs_b,
                          const md_t     row0,
                          const md_t     col0,
                          const md_t     KC,
                          const md_t     n0_partial_rem);

void
dlp_packb_nr64_s8s4s32os32_row_major(int8_t*        pack_b_buffer,
                                     int32_t*       pack_b_column_sum,
                                     const uint8_t* b_s4,
                                     const md_t     rs_b,
                                     const md_t     row0,
                                     const md_t     col0,
                                     const md_t     NC,
                                     const md_t     KC,
                                     md_t*          rs_b_out,
                                     md_t*          cs_b_out)
{
    md_t NR = 64;

    __m512i selector1 =
        _mm512_setr_epi64(0x0, 0x1, 0x8, 0x9, 0x2, 0x3, 0xA, 0xB);
    __m512i selector1_1 =
        _mm512_setr_epi64(0x4, 0x5, 0xC, 0xD, 0x6, 0x7, 0xE, 0xF);
    __m512i selector2 =
        _mm512_setr_epi64(0x0, 0x1, 0x2, 0x3, 0x8, 0x9, 0xA, 0xB);
    __m512i selector2_1 =
        _mm512_setr_epi64(0x4, 0x5, 0x6, 0x7, 0xC, 0xD, 0xE, 0xF);

    md_t n_full_pieces            = NC / NR;
    md_t n_full_pieces_loop_limit = n_full_pieces * NR;
    md_t n_partial_pieces         = NC % NR;

    md_t k_full_pieces_blks = KC / 4;
    md_t k_full_pieces      = k_full_pieces_blks * 4;
    md_t k_partial_pieces   = KC % 4;

    md_t KC_updated = KC;
    if (k_partial_pieces > 0) {
        KC_updated += (4 - k_partial_pieces);
    }

    __m512i sum1, sum2, sum3, sum4;
    __m512i mul_128 = _mm512_set1_epi32(7);

    // int4 -> int8 widen selectors.
    __m512i shift_idx_64;
    MULTISHIFT_32BIT_8_INT4_IDX_64ELEM(shift_idx_64);
    __m512i sign_comp_64 = _mm512_set1_epi8(0x08);
    CREATE_CVT_INT4_INT8_PERM_IDX_64ELEM_ODD_LD(conv_shift_arr_64);
    __m512i conv_shift_64 = _mm512_loadu_epi64(conv_shift_arr_64);

    __m512i a0, b0, c0, d0, a01, c01;

    for (iter_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR) {
        sum1 = _mm512_loadu_si512(pack_b_column_sum + jc);
        sum2 = _mm512_loadu_si512(pack_b_column_sum + 16 + jc);
        sum3 = _mm512_loadu_si512(pack_b_column_sum + 32 + jc);
        sum4 = _mm512_loadu_si512(pack_b_column_sum + 48 + jc);

        for (iter_t kr = 0; kr < k_full_pieces; kr += 4) {
            WIDEN64_ROW(a0, rs_b * (row0 + kr + 0) + (col0 + jc));
            WIDEN64_ROW(b0, rs_b * (row0 + kr + 1) + (col0 + jc));
            WIDEN64_ROW(c0, rs_b * (row0 + kr + 2) + (col0 + jc));
            WIDEN64_ROW(d0, rs_b * (row0 + kr + 3) + (col0 + jc));

            sum1 = _mm512_add_epi32(
                sum1,
                _mm512_sllv_epi32(
                    _mm512_add_epi32(
                        _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(a0, 0)),
                        _mm512_add_epi32(
                            _mm512_cvtepi8_epi32(
                                _mm512_extracti32x4_epi32(b0, 0)),
                            _mm512_add_epi32(
                                _mm512_cvtepi8_epi32(
                                    _mm512_extracti32x4_epi32(c0, 0)),
                                _mm512_cvtepi8_epi32(
                                    _mm512_extracti32x4_epi32(d0, 0))))),
                    mul_128));
            sum2 = _mm512_add_epi32(
                sum2,
                _mm512_sllv_epi32(
                    _mm512_add_epi32(
                        _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(a0, 1)),
                        _mm512_add_epi32(
                            _mm512_cvtepi8_epi32(
                                _mm512_extracti32x4_epi32(b0, 1)),
                            _mm512_add_epi32(
                                _mm512_cvtepi8_epi32(
                                    _mm512_extracti32x4_epi32(c0, 1)),
                                _mm512_cvtepi8_epi32(
                                    _mm512_extracti32x4_epi32(d0, 1))))),
                    mul_128));
            sum3 = _mm512_add_epi32(
                sum3,
                _mm512_sllv_epi32(
                    _mm512_add_epi32(
                        _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(a0, 2)),
                        _mm512_add_epi32(
                            _mm512_cvtepi8_epi32(
                                _mm512_extracti32x4_epi32(b0, 2)),
                            _mm512_add_epi32(
                                _mm512_cvtepi8_epi32(
                                    _mm512_extracti32x4_epi32(c0, 2)),
                                _mm512_cvtepi8_epi32(
                                    _mm512_extracti32x4_epi32(d0, 2))))),
                    mul_128));
            sum4 = _mm512_add_epi32(
                sum4,
                _mm512_sllv_epi32(
                    _mm512_add_epi32(
                        _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(a0, 3)),
                        _mm512_add_epi32(
                            _mm512_cvtepi8_epi32(
                                _mm512_extracti32x4_epi32(b0, 3)),
                            _mm512_add_epi32(
                                _mm512_cvtepi8_epi32(
                                    _mm512_extracti32x4_epi32(c0, 3)),
                                _mm512_cvtepi8_epi32(
                                    _mm512_extracti32x4_epi32(d0, 3))))),
                    mul_128));

            a01 = _mm512_unpacklo_epi8(a0, b0);
            a0  = _mm512_unpackhi_epi8(a0, b0);
            c01 = _mm512_unpacklo_epi8(c0, d0);
            c0  = _mm512_unpackhi_epi8(c0, d0);

            b0  = _mm512_unpacklo_epi16(a01, c01);
            a01 = _mm512_unpackhi_epi16(a01, c01);
            d0  = _mm512_unpacklo_epi16(a0, c0);
            c01 = _mm512_unpackhi_epi16(a0, c0);

            a0 = _mm512_permutex2var_epi64(b0, selector1, a01);
            c0 = _mm512_permutex2var_epi64(d0, selector1, c01);
            b0 = _mm512_permutex2var_epi64(b0, selector1_1, a01);
            d0 = _mm512_permutex2var_epi64(d0, selector1_1, c01);

            a01 = _mm512_permutex2var_epi64(a0, selector2, c0);
            c01 = _mm512_permutex2var_epi64(b0, selector2, d0);
            a0  = _mm512_permutex2var_epi64(a0, selector2_1, c0);
            c0  = _mm512_permutex2var_epi64(b0, selector2_1, d0);

            _mm512_storeu_si512(
                pack_b_buffer + ((jc * KC_updated) + ((kr + 0) * NR)), a01);
            _mm512_storeu_si512(
                pack_b_buffer + ((jc * KC_updated) + ((kr + 1) * NR)), a0);
            _mm512_storeu_si512(
                pack_b_buffer + ((jc * KC_updated) + ((kr + 2) * NR)), c01);
            _mm512_storeu_si512(
                pack_b_buffer + ((jc * KC_updated) + ((kr + 3) * NR)), c0);
        }
        if (k_partial_pieces > 0) {
            if (k_partial_pieces == 3) {
                WIDEN64_ROW(a0,
                            rs_b * (row0 + k_full_pieces + 0) + (col0 + jc));
                WIDEN64_ROW(b0,
                            rs_b * (row0 + k_full_pieces + 1) + (col0 + jc));
                WIDEN64_ROW(c0,
                            rs_b * (row0 + k_full_pieces + 2) + (col0 + jc));
                d0 = _mm512_setzero_si512();

                sum1 = _mm512_add_epi32(
                    sum1, _mm512_sllv_epi32(
                              _mm512_add_epi32(
                                  _mm512_cvtepi8_epi32(
                                      _mm512_extracti32x4_epi32(a0, 0)),
                                  _mm512_add_epi32(
                                      _mm512_cvtepi8_epi32(
                                          _mm512_extracti32x4_epi32(b0, 0)),
                                      _mm512_cvtepi8_epi32(
                                          _mm512_extracti32x4_epi32(c0, 0)))),
                              mul_128));
                sum2 = _mm512_add_epi32(
                    sum2, _mm512_sllv_epi32(
                              _mm512_add_epi32(
                                  _mm512_cvtepi8_epi32(
                                      _mm512_extracti32x4_epi32(a0, 1)),
                                  _mm512_add_epi32(
                                      _mm512_cvtepi8_epi32(
                                          _mm512_extracti32x4_epi32(b0, 1)),
                                      _mm512_cvtepi8_epi32(
                                          _mm512_extracti32x4_epi32(c0, 1)))),
                              mul_128));
                sum3 = _mm512_add_epi32(
                    sum3, _mm512_sllv_epi32(
                              _mm512_add_epi32(
                                  _mm512_cvtepi8_epi32(
                                      _mm512_extracti32x4_epi32(a0, 2)),
                                  _mm512_add_epi32(
                                      _mm512_cvtepi8_epi32(
                                          _mm512_extracti32x4_epi32(b0, 2)),
                                      _mm512_cvtepi8_epi32(
                                          _mm512_extracti32x4_epi32(c0, 2)))),
                              mul_128));
                sum4 = _mm512_add_epi32(
                    sum4, _mm512_sllv_epi32(
                              _mm512_add_epi32(
                                  _mm512_cvtepi8_epi32(
                                      _mm512_extracti32x4_epi32(a0, 3)),
                                  _mm512_add_epi32(
                                      _mm512_cvtepi8_epi32(
                                          _mm512_extracti32x4_epi32(b0, 3)),
                                      _mm512_cvtepi8_epi32(
                                          _mm512_extracti32x4_epi32(c0, 3)))),
                              mul_128));
            } else if (k_partial_pieces == 2) {
                WIDEN64_ROW(a0,
                            rs_b * (row0 + k_full_pieces + 0) + (col0 + jc));
                WIDEN64_ROW(b0,
                            rs_b * (row0 + k_full_pieces + 1) + (col0 + jc));
                c0 = _mm512_setzero_si512();
                d0 = _mm512_setzero_si512();

                sum1 = _mm512_add_epi32(
                    sum1,
                    _mm512_sllv_epi32(
                        _mm512_add_epi32(_mm512_cvtepi8_epi32(
                                             _mm512_extracti32x4_epi32(a0, 0)),
                                         _mm512_cvtepi8_epi32(
                                             _mm512_extracti32x4_epi32(b0, 0))),
                        mul_128));
                sum2 = _mm512_add_epi32(
                    sum2,
                    _mm512_sllv_epi32(
                        _mm512_add_epi32(_mm512_cvtepi8_epi32(
                                             _mm512_extracti32x4_epi32(a0, 1)),
                                         _mm512_cvtepi8_epi32(
                                             _mm512_extracti32x4_epi32(b0, 1))),
                        mul_128));
                sum3 = _mm512_add_epi32(
                    sum3,
                    _mm512_sllv_epi32(
                        _mm512_add_epi32(_mm512_cvtepi8_epi32(
                                             _mm512_extracti32x4_epi32(a0, 2)),
                                         _mm512_cvtepi8_epi32(
                                             _mm512_extracti32x4_epi32(b0, 2))),
                        mul_128));
                sum4 = _mm512_add_epi32(
                    sum4,
                    _mm512_sllv_epi32(
                        _mm512_add_epi32(_mm512_cvtepi8_epi32(
                                             _mm512_extracti32x4_epi32(a0, 3)),
                                         _mm512_cvtepi8_epi32(
                                             _mm512_extracti32x4_epi32(b0, 3))),
                        mul_128));
            } else {
                WIDEN64_ROW(a0,
                            rs_b * (row0 + k_full_pieces + 0) + (col0 + jc));
                b0 = _mm512_setzero_si512();
                c0 = _mm512_setzero_si512();
                d0 = _mm512_setzero_si512();

                sum1 = _mm512_add_epi32(
                    sum1,
                    _mm512_sllv_epi32(
                        _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(a0, 0)),
                        mul_128));
                sum2 = _mm512_add_epi32(
                    sum2,
                    _mm512_sllv_epi32(
                        _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(a0, 1)),
                        mul_128));
                sum3 = _mm512_add_epi32(
                    sum3,
                    _mm512_sllv_epi32(
                        _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(a0, 2)),
                        mul_128));
                sum4 = _mm512_add_epi32(
                    sum4,
                    _mm512_sllv_epi32(
                        _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(a0, 3)),
                        mul_128));
            }

            a01 = _mm512_unpacklo_epi8(a0, b0);
            a0  = _mm512_unpackhi_epi8(a0, b0);
            c01 = _mm512_unpacklo_epi8(c0, d0);
            c0  = _mm512_unpackhi_epi8(c0, d0);

            b0  = _mm512_unpacklo_epi16(a01, c01);
            a01 = _mm512_unpackhi_epi16(a01, c01);
            d0  = _mm512_unpacklo_epi16(a0, c0);
            c01 = _mm512_unpackhi_epi16(a0, c0);

            a0 = _mm512_permutex2var_epi64(b0, selector1, a01);
            c0 = _mm512_permutex2var_epi64(d0, selector1, c01);
            b0 = _mm512_permutex2var_epi64(b0, selector1_1, a01);
            d0 = _mm512_permutex2var_epi64(d0, selector1_1, c01);

            a01 = _mm512_permutex2var_epi64(a0, selector2, c0);
            c01 = _mm512_permutex2var_epi64(b0, selector2, d0);
            a0  = _mm512_permutex2var_epi64(a0, selector2_1, c0);
            c0  = _mm512_permutex2var_epi64(b0, selector2_1, d0);

            _mm512_storeu_si512(
                pack_b_buffer
                    + ((jc * KC_updated) + ((k_full_pieces + 0) * NR)),
                a01);
            _mm512_storeu_si512(
                pack_b_buffer
                    + ((jc * KC_updated) + ((k_full_pieces + 1) * NR)),
                a0);
            _mm512_storeu_si512(
                pack_b_buffer
                    + ((jc * KC_updated) + ((k_full_pieces + 2) * NR)),
                c01);
            _mm512_storeu_si512(
                pack_b_buffer
                    + ((jc * KC_updated) + ((k_full_pieces + 3) * NR)),
                c0);
        }
        _mm512_storeu_si512(pack_b_column_sum + jc, sum1);
        _mm512_storeu_si512(pack_b_column_sum + 16 + jc, sum2);
        _mm512_storeu_si512(pack_b_column_sum + 32 + jc, sum3);
        _mm512_storeu_si512(pack_b_column_sum + 48 + jc, sum4);
    }

    if (n_partial_pieces > 0) {
        md_t n0_partial_rem  = n_partial_pieces % 16;
        md_t n0_partial_pack = 0;

        md_t n0_48 = n_partial_pieces / 48;
        md_t n0_32 = n_partial_pieces / 32;
        md_t n0_16 = n_partial_pieces / 16;

        if (n0_48 == 1) {
            dlp_packb_nr48_s8s4_row(
                pack_b_buffer + (n_full_pieces_loop_limit * KC_updated),
                pack_b_column_sum + n_full_pieces_loop_limit, b_s4, rs_b, row0,
                col0 + n_full_pieces_loop_limit, KC);
            n0_partial_pack = 48;
        } else if (n0_32 == 1) {
            dlp_packb_nr32_s8s4_row(
                pack_b_buffer + (n_full_pieces_loop_limit * KC_updated),
                pack_b_column_sum + n_full_pieces_loop_limit, b_s4, rs_b, row0,
                col0 + n_full_pieces_loop_limit, KC);
            n0_partial_pack = 32;
        } else if (n0_16 == 1) {
            dlp_packb_nr16_s8s4_row(
                pack_b_buffer + (n_full_pieces_loop_limit * KC_updated),
                pack_b_column_sum + n_full_pieces_loop_limit, b_s4, rs_b, row0,
                col0 + n_full_pieces_loop_limit, KC);
            n0_partial_pack = 16;
        }

        if (n0_partial_rem > 0) {
            dlp_packb_nrlt16_s8s4_row(
                pack_b_buffer + (n_full_pieces_loop_limit * KC_updated)
                    + (n0_partial_pack * KC_updated),
                pack_b_column_sum + n_full_pieces_loop_limit + n0_partial_pack,
                b_s4, rs_b, row0,
                col0 + n_full_pieces_loop_limit + n0_partial_pack, KC,
                n0_partial_rem);
        }
    }
    *rs_b_out = NR * 4;
    *cs_b_out = NR;
}

/* ---- row-major fringe packers (fork of the s8 nr48/32/16/nrlt16). ---- */

static void
dlp_packb_nr48_s8s4_row(int8_t*        pack_b,
                        int32_t*       colsum,
                        const uint8_t* b_s4,
                        const md_t     rs_b,
                        const md_t     row0,
                        const md_t     col0,
                        const md_t     KC)
{
    md_t NR     = 64;
    md_t kr_new = 0;

    md_t k_full_pieces_blks = KC / 4;
    md_t k_full_pieces      = k_full_pieces_blks * 4;
    md_t k_partial_pieces   = KC % 4;

    __m256i a0_32, b0_32, c0_32, d0_32, a01_32, c01_32;
    __m512i a0_zmm, b0_zmm;
    __m128i a0_16, b0_16, c0_16, d0_16, a01_16, c01_16;

    __m512i sum1, sum2, sum3;
    __m512i mul_128 = _mm512_set1_epi32(7);

    __m256i shift_idx_32;
    MULTISHIFT_32BIT_8_INT4_IDX_32ELEM(shift_idx_32);
    __m256i sign_comp_32 = _mm256_set1_epi8(0x08);
    CREATE_CVT_INT4_INT8_PERM_IDX_32ELEM_ODD_LD(conv_shift_arr_32);
    __m256i conv_shift_32 =
        _mm256_maskz_loadu_epi64(_cvtu32_mask8(0xFF), conv_shift_arr_32);

    __m128i shift_idx_16;
    MULTISHIFT_32BIT_8_INT4_IDX_16ELEM(shift_idx_16);
    __m128i sign_comp_16 = _mm_set1_epi8(0x08);
    CREATE_CVT_INT4_INT8_PERM_IDX_16ELEM_ODD_LD(conv_shift_arr_16);
    __m128i conv_shift_16 =
        _mm_maskz_loadu_epi64(_cvtu32_mask8(0xFF), conv_shift_arr_16);

    sum1 = _mm512_loadu_si512(colsum);
    sum2 = _mm512_loadu_si512(colsum + 16);
    sum3 = _mm512_loadu_si512(colsum + 32);

    for (iter_t kr = 0; kr < k_full_pieces; kr += 4) {
        WIDEN32_ROW(a0_32, rs_b * (row0 + kr + 0) + col0);
        WIDEN32_ROW(b0_32, rs_b * (row0 + kr + 1) + col0);
        WIDEN32_ROW(c0_32, rs_b * (row0 + kr + 2) + col0);
        WIDEN32_ROW(d0_32, rs_b * (row0 + kr + 3) + col0);

        sum1 = _mm512_add_epi32(
            sum1,
            _mm512_sllv_epi32(
                _mm512_add_epi32(
                    _mm512_cvtepi8_epi32(_mm256_extracti32x4_epi32(a0_32, 0)),
                    _mm512_add_epi32(
                        _mm512_cvtepi8_epi32(
                            _mm256_extracti32x4_epi32(b0_32, 0)),
                        _mm512_add_epi32(
                            _mm512_cvtepi8_epi32(
                                _mm256_extracti32x4_epi32(c0_32, 0)),
                            _mm512_cvtepi8_epi32(
                                _mm256_extracti32x4_epi32(d0_32, 0))))),
                mul_128));
        sum2 = _mm512_add_epi32(
            sum2,
            _mm512_sllv_epi32(
                _mm512_add_epi32(
                    _mm512_cvtepi8_epi32(_mm256_extracti32x4_epi32(a0_32, 1)),
                    _mm512_add_epi32(
                        _mm512_cvtepi8_epi32(
                            _mm256_extracti32x4_epi32(b0_32, 1)),
                        _mm512_add_epi32(
                            _mm512_cvtepi8_epi32(
                                _mm256_extracti32x4_epi32(c0_32, 1)),
                            _mm512_cvtepi8_epi32(
                                _mm256_extracti32x4_epi32(d0_32, 1))))),
                mul_128));

        a01_32 = _mm256_unpacklo_epi8(a0_32, b0_32);
        a0_32  = _mm256_unpackhi_epi8(a0_32, b0_32);
        c01_32 = _mm256_unpacklo_epi8(c0_32, d0_32);
        c0_32  = _mm256_unpackhi_epi8(c0_32, d0_32);

        b0_32  = _mm256_unpacklo_epi16(a01_32, c01_32);
        a01_32 = _mm256_unpackhi_epi16(a01_32, c01_32);
        d0_32  = _mm256_unpacklo_epi16(a0_32, c0_32);
        c01_32 = _mm256_unpackhi_epi16(a0_32, c0_32);

        a0_32 = _mm256_shuffle_i32x4(b0_32, a01_32, 0x0);
        c0_32 = _mm256_shuffle_i32x4(b0_32, a01_32, 0x3);
        b0_32 = _mm256_shuffle_i32x4(d0_32, c01_32, 0x0);
        d0_32 = _mm256_shuffle_i32x4(d0_32, c01_32, 0x3);

        a0_zmm = _mm512_castsi256_si512(a0_32);
        a0_zmm = _mm512_inserti32x8(a0_zmm, b0_32, 0x1);
        b0_zmm = _mm512_castsi256_si512(c0_32);
        b0_zmm = _mm512_inserti32x8(b0_zmm, d0_32, 0x1);

        _mm512_storeu_si512(pack_b + ((kr_new + 0) * NR), a0_zmm);
        _mm512_storeu_si512(pack_b + ((kr_new + 1) * NR), b0_zmm);

        WIDEN16_ROW(a0_16, rs_b * (row0 + kr + 0) + col0 + 32);
        WIDEN16_ROW(b0_16, rs_b * (row0 + kr + 1) + col0 + 32);
        WIDEN16_ROW(c0_16, rs_b * (row0 + kr + 2) + col0 + 32);
        WIDEN16_ROW(d0_16, rs_b * (row0 + kr + 3) + col0 + 32);

        sum3 = _mm512_add_epi32(
            sum3, _mm512_sllv_epi32(
                      _mm512_add_epi32(
                          _mm512_cvtepi8_epi32(a0_16),
                          _mm512_add_epi32(
                              _mm512_cvtepi8_epi32(b0_16),
                              _mm512_add_epi32(_mm512_cvtepi8_epi32(c0_16),
                                               _mm512_cvtepi8_epi32(d0_16)))),
                      mul_128));

        a01_16 = _mm_unpacklo_epi8(a0_16, b0_16);
        a0_16  = _mm_unpackhi_epi8(a0_16, b0_16);
        c01_16 = _mm_unpacklo_epi8(c0_16, d0_16);
        c0_16  = _mm_unpackhi_epi8(c0_16, d0_16);

        b0_16  = _mm_unpacklo_epi16(a01_16, c01_16);
        a01_16 = _mm_unpackhi_epi16(a01_16, c01_16);
        d0_16  = _mm_unpacklo_epi16(a0_16, c0_16);
        c01_16 = _mm_unpackhi_epi16(a0_16, c0_16);

        a0_zmm = _mm512_castsi128_si512(b0_16);
        a0_zmm = _mm512_inserti32x4(a0_zmm, a01_16, 0x1);
        a0_zmm = _mm512_inserti32x4(a0_zmm, d0_16, 0x2);
        a0_zmm = _mm512_inserti32x4(a0_zmm, c01_16, 0x3);

        _mm512_storeu_si512(pack_b + ((kr_new + 2) * NR), a0_zmm);
        kr_new += 3;
    }
    if (k_partial_pieces > 0) {
        a0_32 = _mm256_setzero_si256();
        b0_32 = _mm256_setzero_si256();
        c0_32 = _mm256_setzero_si256();
        d0_32 = _mm256_setzero_si256();
        a0_16 = _mm_setzero_si128();
        b0_16 = _mm_setzero_si128();
        c0_16 = _mm_setzero_si128();
        d0_16 = _mm_setzero_si128();

        if (k_partial_pieces == 3) {
            WIDEN32_ROW(a0_32, rs_b * (row0 + k_full_pieces + 0) + col0);
            WIDEN32_ROW(b0_32, rs_b * (row0 + k_full_pieces + 1) + col0);
            WIDEN32_ROW(c0_32, rs_b * (row0 + k_full_pieces + 2) + col0);
            WIDEN16_ROW(a0_16, rs_b * (row0 + k_full_pieces + 0) + col0 + 32);
            WIDEN16_ROW(b0_16, rs_b * (row0 + k_full_pieces + 1) + col0 + 32);
            WIDEN16_ROW(c0_16, rs_b * (row0 + k_full_pieces + 2) + col0 + 32);
        } else if (k_partial_pieces == 2) {
            WIDEN32_ROW(a0_32, rs_b * (row0 + k_full_pieces + 0) + col0);
            WIDEN32_ROW(b0_32, rs_b * (row0 + k_full_pieces + 1) + col0);
            WIDEN16_ROW(a0_16, rs_b * (row0 + k_full_pieces + 0) + col0 + 32);
            WIDEN16_ROW(b0_16, rs_b * (row0 + k_full_pieces + 1) + col0 + 32);
        } else {
            WIDEN32_ROW(a0_32, rs_b * (row0 + k_full_pieces + 0) + col0);
            WIDEN16_ROW(a0_16, rs_b * (row0 + k_full_pieces + 0) + col0 + 32);
        }

        sum1 = _mm512_add_epi32(
            sum1,
            _mm512_sllv_epi32(
                _mm512_add_epi32(
                    _mm512_cvtepi8_epi32(_mm256_extracti32x4_epi32(a0_32, 0)),
                    _mm512_add_epi32(_mm512_cvtepi8_epi32(
                                         _mm256_extracti32x4_epi32(b0_32, 0)),
                                     _mm512_cvtepi8_epi32(
                                         _mm256_extracti32x4_epi32(c0_32, 0)))),
                mul_128));
        sum2 = _mm512_add_epi32(
            sum2,
            _mm512_sllv_epi32(
                _mm512_add_epi32(
                    _mm512_cvtepi8_epi32(_mm256_extracti32x4_epi32(a0_32, 1)),
                    _mm512_add_epi32(_mm512_cvtepi8_epi32(
                                         _mm256_extracti32x4_epi32(b0_32, 1)),
                                     _mm512_cvtepi8_epi32(
                                         _mm256_extracti32x4_epi32(c0_32, 1)))),
                mul_128));
        sum3 = _mm512_add_epi32(
            sum3,
            _mm512_sllv_epi32(
                _mm512_add_epi32(_mm512_cvtepi8_epi32(a0_16),
                                 _mm512_add_epi32(_mm512_cvtepi8_epi32(b0_16),
                                                  _mm512_cvtepi8_epi32(c0_16))),
                mul_128));

        a01_32 = _mm256_unpacklo_epi8(a0_32, b0_32);
        a0_32  = _mm256_unpackhi_epi8(a0_32, b0_32);
        c01_32 = _mm256_unpacklo_epi8(c0_32, d0_32);
        c0_32  = _mm256_unpackhi_epi8(c0_32, d0_32);

        b0_32  = _mm256_unpacklo_epi16(a01_32, c01_32);
        a01_32 = _mm256_unpackhi_epi16(a01_32, c01_32);
        d0_32  = _mm256_unpacklo_epi16(a0_32, c0_32);
        c01_32 = _mm256_unpackhi_epi16(a0_32, c0_32);

        a0_32 = _mm256_shuffle_i32x4(b0_32, a01_32, 0x0);
        c0_32 = _mm256_shuffle_i32x4(b0_32, a01_32, 0x3);
        b0_32 = _mm256_shuffle_i32x4(d0_32, c01_32, 0x0);
        d0_32 = _mm256_shuffle_i32x4(d0_32, c01_32, 0x3);

        a0_zmm = _mm512_castsi256_si512(a0_32);
        a0_zmm = _mm512_inserti32x8(a0_zmm, b0_32, 0x1);
        b0_zmm = _mm512_castsi256_si512(c0_32);
        b0_zmm = _mm512_inserti32x8(b0_zmm, d0_32, 0x1);

        _mm512_storeu_si512(pack_b + ((kr_new + 0) * NR), a0_zmm);
        _mm512_storeu_si512(pack_b + ((kr_new + 1) * NR), b0_zmm);

        a01_16 = _mm_unpacklo_epi8(a0_16, b0_16);
        a0_16  = _mm_unpackhi_epi8(a0_16, b0_16);
        c01_16 = _mm_unpacklo_epi8(c0_16, d0_16);
        c0_16  = _mm_unpackhi_epi8(c0_16, d0_16);

        b0_16  = _mm_unpacklo_epi16(a01_16, c01_16);
        a01_16 = _mm_unpackhi_epi16(a01_16, c01_16);
        d0_16  = _mm_unpacklo_epi16(a0_16, c0_16);
        c01_16 = _mm_unpackhi_epi16(a0_16, c0_16);

        a0_zmm = _mm512_castsi128_si512(b0_16);
        a0_zmm = _mm512_inserti32x4(a0_zmm, a01_16, 0x1);
        a0_zmm = _mm512_inserti32x4(a0_zmm, d0_16, 0x2);
        a0_zmm = _mm512_inserti32x4(a0_zmm, c01_16, 0x3);

        _mm512_storeu_si512(pack_b + ((kr_new + 2) * NR), a0_zmm);
    }
    _mm512_storeu_si512(colsum, sum1);
    _mm512_storeu_si512(colsum + 16, sum2);
    _mm512_storeu_si512(colsum + 32, sum3);
}

static void
dlp_packb_nr32_s8s4_row(int8_t*        pack_b,
                        int32_t*       colsum,
                        const uint8_t* b_s4,
                        const md_t     rs_b,
                        const md_t     row0,
                        const md_t     col0,
                        const md_t     KC)
{
    md_t NR     = 64;
    md_t kr_new = 0;

    md_t k_full_pieces_blks = KC / 4;
    md_t k_full_pieces      = k_full_pieces_blks * 4;
    md_t k_partial_pieces   = KC % 4;

    __m256i a0_32, b0_32, c0_32, d0_32, a01_32, c01_32;
    __m512i a0_zmm, b0_zmm;

    __m512i sum1, sum2;
    __m512i mul_128 = _mm512_set1_epi32(7);

    __m256i shift_idx_32;
    MULTISHIFT_32BIT_8_INT4_IDX_32ELEM(shift_idx_32);
    __m256i sign_comp_32 = _mm256_set1_epi8(0x08);
    CREATE_CVT_INT4_INT8_PERM_IDX_32ELEM_ODD_LD(conv_shift_arr_32);
    __m256i conv_shift_32 =
        _mm256_maskz_loadu_epi64(_cvtu32_mask8(0xFF), conv_shift_arr_32);

    sum1 = _mm512_loadu_si512(colsum);
    sum2 = _mm512_loadu_si512(colsum + 16);

    for (iter_t kr = 0; kr < k_full_pieces; kr += 4) {
        WIDEN32_ROW(a0_32, rs_b * (row0 + kr + 0) + col0);
        WIDEN32_ROW(b0_32, rs_b * (row0 + kr + 1) + col0);
        WIDEN32_ROW(c0_32, rs_b * (row0 + kr + 2) + col0);
        WIDEN32_ROW(d0_32, rs_b * (row0 + kr + 3) + col0);

        sum1 = _mm512_add_epi32(
            sum1,
            _mm512_sllv_epi32(
                _mm512_add_epi32(
                    _mm512_cvtepi8_epi32(_mm256_extracti32x4_epi32(a0_32, 0)),
                    _mm512_add_epi32(
                        _mm512_cvtepi8_epi32(
                            _mm256_extracti32x4_epi32(b0_32, 0)),
                        _mm512_add_epi32(
                            _mm512_cvtepi8_epi32(
                                _mm256_extracti32x4_epi32(c0_32, 0)),
                            _mm512_cvtepi8_epi32(
                                _mm256_extracti32x4_epi32(d0_32, 0))))),
                mul_128));
        sum2 = _mm512_add_epi32(
            sum2,
            _mm512_sllv_epi32(
                _mm512_add_epi32(
                    _mm512_cvtepi8_epi32(_mm256_extracti32x4_epi32(a0_32, 1)),
                    _mm512_add_epi32(
                        _mm512_cvtepi8_epi32(
                            _mm256_extracti32x4_epi32(b0_32, 1)),
                        _mm512_add_epi32(
                            _mm512_cvtepi8_epi32(
                                _mm256_extracti32x4_epi32(c0_32, 1)),
                            _mm512_cvtepi8_epi32(
                                _mm256_extracti32x4_epi32(d0_32, 1))))),
                mul_128));

        a01_32 = _mm256_unpacklo_epi8(a0_32, b0_32);
        a0_32  = _mm256_unpackhi_epi8(a0_32, b0_32);
        c01_32 = _mm256_unpacklo_epi8(c0_32, d0_32);
        c0_32  = _mm256_unpackhi_epi8(c0_32, d0_32);

        b0_32  = _mm256_unpacklo_epi16(a01_32, c01_32);
        a01_32 = _mm256_unpackhi_epi16(a01_32, c01_32);
        d0_32  = _mm256_unpacklo_epi16(a0_32, c0_32);
        c01_32 = _mm256_unpackhi_epi16(a0_32, c0_32);

        a0_32 = _mm256_shuffle_i32x4(b0_32, a01_32, 0x0);
        c0_32 = _mm256_shuffle_i32x4(b0_32, a01_32, 0x3);
        b0_32 = _mm256_shuffle_i32x4(d0_32, c01_32, 0x0);
        d0_32 = _mm256_shuffle_i32x4(d0_32, c01_32, 0x3);

        a0_zmm = _mm512_castsi256_si512(a0_32);
        a0_zmm = _mm512_inserti32x8(a0_zmm, b0_32, 0x1);
        b0_zmm = _mm512_castsi256_si512(c0_32);
        b0_zmm = _mm512_inserti32x8(b0_zmm, d0_32, 0x1);

        _mm512_storeu_si512(pack_b + ((kr_new + 0) * NR), a0_zmm);
        _mm512_storeu_si512(pack_b + ((kr_new + 1) * NR), b0_zmm);
        kr_new += 2;
    }
    if (k_partial_pieces > 0) {
        a0_32 = _mm256_setzero_si256();
        b0_32 = _mm256_setzero_si256();
        c0_32 = _mm256_setzero_si256();
        d0_32 = _mm256_setzero_si256();

        if (k_partial_pieces == 3) {
            WIDEN32_ROW(a0_32, rs_b * (row0 + k_full_pieces + 0) + col0);
            WIDEN32_ROW(b0_32, rs_b * (row0 + k_full_pieces + 1) + col0);
            WIDEN32_ROW(c0_32, rs_b * (row0 + k_full_pieces + 2) + col0);
        } else if (k_partial_pieces == 2) {
            WIDEN32_ROW(a0_32, rs_b * (row0 + k_full_pieces + 0) + col0);
            WIDEN32_ROW(b0_32, rs_b * (row0 + k_full_pieces + 1) + col0);
        } else {
            WIDEN32_ROW(a0_32, rs_b * (row0 + k_full_pieces + 0) + col0);
        }

        sum1 = _mm512_add_epi32(
            sum1,
            _mm512_sllv_epi32(
                _mm512_add_epi32(
                    _mm512_cvtepi8_epi32(_mm256_extracti32x4_epi32(a0_32, 0)),
                    _mm512_add_epi32(_mm512_cvtepi8_epi32(
                                         _mm256_extracti32x4_epi32(b0_32, 0)),
                                     _mm512_cvtepi8_epi32(
                                         _mm256_extracti32x4_epi32(c0_32, 0)))),
                mul_128));
        sum2 = _mm512_add_epi32(
            sum2,
            _mm512_sllv_epi32(
                _mm512_add_epi32(
                    _mm512_cvtepi8_epi32(_mm256_extracti32x4_epi32(a0_32, 1)),
                    _mm512_add_epi32(_mm512_cvtepi8_epi32(
                                         _mm256_extracti32x4_epi32(b0_32, 1)),
                                     _mm512_cvtepi8_epi32(
                                         _mm256_extracti32x4_epi32(c0_32, 1)))),
                mul_128));

        a01_32 = _mm256_unpacklo_epi8(a0_32, b0_32);
        a0_32  = _mm256_unpackhi_epi8(a0_32, b0_32);
        c01_32 = _mm256_unpacklo_epi8(c0_32, d0_32);
        c0_32  = _mm256_unpackhi_epi8(c0_32, d0_32);

        b0_32  = _mm256_unpacklo_epi16(a01_32, c01_32);
        a01_32 = _mm256_unpackhi_epi16(a01_32, c01_32);
        d0_32  = _mm256_unpacklo_epi16(a0_32, c0_32);
        c01_32 = _mm256_unpackhi_epi16(a0_32, c0_32);

        a0_32 = _mm256_shuffle_i32x4(b0_32, a01_32, 0x0);
        c0_32 = _mm256_shuffle_i32x4(b0_32, a01_32, 0x3);
        b0_32 = _mm256_shuffle_i32x4(d0_32, c01_32, 0x0);
        d0_32 = _mm256_shuffle_i32x4(d0_32, c01_32, 0x3);

        a0_zmm = _mm512_castsi256_si512(a0_32);
        a0_zmm = _mm512_inserti32x8(a0_zmm, b0_32, 0x1);
        b0_zmm = _mm512_castsi256_si512(c0_32);
        b0_zmm = _mm512_inserti32x8(b0_zmm, d0_32, 0x1);

        _mm512_storeu_si512(pack_b + ((kr_new + 0) * NR), a0_zmm);
        _mm512_storeu_si512(pack_b + ((kr_new + 1) * NR), b0_zmm);
    }
    _mm512_storeu_si512(colsum, sum1);
    _mm512_storeu_si512(colsum + 16, sum2);
}

static void
dlp_packb_nr16_s8s4_row(int8_t*        pack_b,
                        int32_t*       colsum,
                        const uint8_t* b_s4,
                        const md_t     rs_b,
                        const md_t     row0,
                        const md_t     col0,
                        const md_t     KC)
{
    md_t NR     = 64;
    md_t kr_new = 0;

    md_t k_full_pieces_blks = KC / 4;
    md_t k_full_pieces      = k_full_pieces_blks * 4;
    md_t k_partial_pieces   = KC % 4;

    __m128i a0_16, b0_16, c0_16, d0_16, a01_16, c01_16;
    __m512i a0_zmm;

    __m512i sum1;
    __m512i mul_128 = _mm512_set1_epi32(7);

    __m128i shift_idx_16;
    MULTISHIFT_32BIT_8_INT4_IDX_16ELEM(shift_idx_16);
    __m128i sign_comp_16 = _mm_set1_epi8(0x08);
    CREATE_CVT_INT4_INT8_PERM_IDX_16ELEM_ODD_LD(conv_shift_arr_16);
    __m128i conv_shift_16 =
        _mm_maskz_loadu_epi64(_cvtu32_mask8(0xFF), conv_shift_arr_16);

    sum1 = _mm512_loadu_si512(colsum);

    for (iter_t kr = 0; kr < k_full_pieces; kr += 4) {
        WIDEN16_ROW(a0_16, rs_b * (row0 + kr + 0) + col0);
        WIDEN16_ROW(b0_16, rs_b * (row0 + kr + 1) + col0);
        WIDEN16_ROW(c0_16, rs_b * (row0 + kr + 2) + col0);
        WIDEN16_ROW(d0_16, rs_b * (row0 + kr + 3) + col0);

        sum1 = _mm512_add_epi32(
            sum1, _mm512_sllv_epi32(
                      _mm512_add_epi32(
                          _mm512_cvtepi8_epi32(a0_16),
                          _mm512_add_epi32(
                              _mm512_cvtepi8_epi32(b0_16),
                              _mm512_add_epi32(_mm512_cvtepi8_epi32(c0_16),
                                               _mm512_cvtepi8_epi32(d0_16)))),
                      mul_128));

        a01_16 = _mm_unpacklo_epi8(a0_16, b0_16);
        a0_16  = _mm_unpackhi_epi8(a0_16, b0_16);
        c01_16 = _mm_unpacklo_epi8(c0_16, d0_16);
        c0_16  = _mm_unpackhi_epi8(c0_16, d0_16);

        b0_16  = _mm_unpacklo_epi16(a01_16, c01_16);
        a01_16 = _mm_unpackhi_epi16(a01_16, c01_16);
        d0_16  = _mm_unpacklo_epi16(a0_16, c0_16);
        c01_16 = _mm_unpackhi_epi16(a0_16, c0_16);

        a0_zmm = _mm512_castsi128_si512(b0_16);
        a0_zmm = _mm512_inserti32x4(a0_zmm, a01_16, 0x1);
        a0_zmm = _mm512_inserti32x4(a0_zmm, d0_16, 0x2);
        a0_zmm = _mm512_inserti32x4(a0_zmm, c01_16, 0x3);

        _mm512_storeu_si512(pack_b + ((kr_new + 0) * NR), a0_zmm);
        kr_new += 1;
    }
    if (k_partial_pieces > 0) {
        a0_16 = _mm_setzero_si128();
        b0_16 = _mm_setzero_si128();
        c0_16 = _mm_setzero_si128();
        d0_16 = _mm_setzero_si128();

        if (k_partial_pieces == 3) {
            WIDEN16_ROW(a0_16, rs_b * (row0 + k_full_pieces + 0) + col0);
            WIDEN16_ROW(b0_16, rs_b * (row0 + k_full_pieces + 1) + col0);
            WIDEN16_ROW(c0_16, rs_b * (row0 + k_full_pieces + 2) + col0);
        } else if (k_partial_pieces == 2) {
            WIDEN16_ROW(a0_16, rs_b * (row0 + k_full_pieces + 0) + col0);
            WIDEN16_ROW(b0_16, rs_b * (row0 + k_full_pieces + 1) + col0);
        } else {
            WIDEN16_ROW(a0_16, rs_b * (row0 + k_full_pieces + 0) + col0);
        }

        sum1 = _mm512_add_epi32(
            sum1,
            _mm512_sllv_epi32(
                _mm512_add_epi32(_mm512_cvtepi8_epi32(a0_16),
                                 _mm512_add_epi32(_mm512_cvtepi8_epi32(b0_16),
                                                  _mm512_cvtepi8_epi32(c0_16))),
                mul_128));

        a01_16 = _mm_unpacklo_epi8(a0_16, b0_16);
        a0_16  = _mm_unpackhi_epi8(a0_16, b0_16);
        c01_16 = _mm_unpacklo_epi8(c0_16, d0_16);
        c0_16  = _mm_unpackhi_epi8(c0_16, d0_16);

        b0_16  = _mm_unpacklo_epi16(a01_16, c01_16);
        a01_16 = _mm_unpackhi_epi16(a01_16, c01_16);
        d0_16  = _mm_unpacklo_epi16(a0_16, c0_16);
        c01_16 = _mm_unpackhi_epi16(a0_16, c0_16);

        a0_zmm = _mm512_castsi128_si512(b0_16);
        a0_zmm = _mm512_inserti32x4(a0_zmm, a01_16, 0x1);
        a0_zmm = _mm512_inserti32x4(a0_zmm, d0_16, 0x2);
        a0_zmm = _mm512_inserti32x4(a0_zmm, c01_16, 0x3);

        _mm512_storeu_si512(pack_b + ((kr_new + 0) * NR), a0_zmm);
    }
    _mm512_storeu_si512(colsum, sum1);
}

static void
dlp_packb_nrlt16_s8s4_row(int8_t*        pack_b,
                          int32_t*       colsum,
                          const uint8_t* b_s4,
                          const md_t     rs_b,
                          const md_t     row0,
                          const md_t     col0,
                          const md_t     KC,
                          const md_t     n0_partial_rem)
{
    md_t NR     = 64;
    md_t kr_new = 0;

    md_t k_full_pieces_blks = KC / 4;
    md_t k_full_pieces      = k_full_pieces_blks * 4;
    md_t k_partial_pieces   = KC % 4;

    __m128i a0_16, b0_16, c0_16, d0_16, a01_16, c01_16;
    __m512i a0_zmm;

    __m512i sum1;
    __m512i mul_128 = _mm512_set1_epi32(7);

    __m128i shift_idx_16;
    MULTISHIFT_32BIT_8_INT4_IDX_16ELEM(shift_idx_16);
    __m128i sign_comp_16 = _mm_set1_epi8(0x08);
    CREATE_CVT_INT4_INT8_PERM_IDX_16ELEM_ODD_LD(conv_shift_arr_16);
    __m128i conv_shift_16 =
        _mm_maskz_loadu_epi64(_cvtu32_mask8(0xFF), conv_shift_arr_16);

    sum1 = _mm512_loadu_si512(colsum);

    for (iter_t kr = 0; kr < k_full_pieces; kr += 4) {
        WIDEN16_ROW_PARTIAL(a0_16, rs_b * (row0 + kr + 0) + col0,
                            n0_partial_rem);
        WIDEN16_ROW_PARTIAL(b0_16, rs_b * (row0 + kr + 1) + col0,
                            n0_partial_rem);
        WIDEN16_ROW_PARTIAL(c0_16, rs_b * (row0 + kr + 2) + col0,
                            n0_partial_rem);
        WIDEN16_ROW_PARTIAL(d0_16, rs_b * (row0 + kr + 3) + col0,
                            n0_partial_rem);

        sum1 = _mm512_add_epi32(
            sum1, _mm512_sllv_epi32(
                      _mm512_add_epi32(
                          _mm512_cvtepi8_epi32(a0_16),
                          _mm512_add_epi32(
                              _mm512_cvtepi8_epi32(b0_16),
                              _mm512_add_epi32(_mm512_cvtepi8_epi32(c0_16),
                                               _mm512_cvtepi8_epi32(d0_16)))),
                      mul_128));

        a01_16 = _mm_unpacklo_epi8(a0_16, b0_16);
        a0_16  = _mm_unpackhi_epi8(a0_16, b0_16);
        c01_16 = _mm_unpacklo_epi8(c0_16, d0_16);
        c0_16  = _mm_unpackhi_epi8(c0_16, d0_16);

        b0_16  = _mm_unpacklo_epi16(a01_16, c01_16);
        a01_16 = _mm_unpackhi_epi16(a01_16, c01_16);
        d0_16  = _mm_unpacklo_epi16(a0_16, c0_16);
        c01_16 = _mm_unpackhi_epi16(a0_16, c0_16);

        a0_zmm = _mm512_castsi128_si512(b0_16);
        a0_zmm = _mm512_inserti32x4(a0_zmm, a01_16, 0x1);
        a0_zmm = _mm512_inserti32x4(a0_zmm, d0_16, 0x2);
        a0_zmm = _mm512_inserti32x4(a0_zmm, c01_16, 0x3);

        _mm512_storeu_si512(pack_b + ((kr_new + 0) * NR), a0_zmm);
        kr_new += 1;
    }
    if (k_partial_pieces > 0) {
        a0_16 = _mm_setzero_si128();
        b0_16 = _mm_setzero_si128();
        c0_16 = _mm_setzero_si128();
        d0_16 = _mm_setzero_si128();

        if (k_partial_pieces == 3) {
            WIDEN16_ROW_PARTIAL(a0_16, rs_b * (row0 + k_full_pieces + 0) + col0,
                                n0_partial_rem);
            WIDEN16_ROW_PARTIAL(b0_16, rs_b * (row0 + k_full_pieces + 1) + col0,
                                n0_partial_rem);
            WIDEN16_ROW_PARTIAL(c0_16, rs_b * (row0 + k_full_pieces + 2) + col0,
                                n0_partial_rem);
        } else if (k_partial_pieces == 2) {
            WIDEN16_ROW_PARTIAL(a0_16, rs_b * (row0 + k_full_pieces + 0) + col0,
                                n0_partial_rem);
            WIDEN16_ROW_PARTIAL(b0_16, rs_b * (row0 + k_full_pieces + 1) + col0,
                                n0_partial_rem);
        } else {
            WIDEN16_ROW_PARTIAL(a0_16, rs_b * (row0 + k_full_pieces + 0) + col0,
                                n0_partial_rem);
        }

        sum1 = _mm512_add_epi32(
            sum1,
            _mm512_sllv_epi32(
                _mm512_add_epi32(_mm512_cvtepi8_epi32(a0_16),
                                 _mm512_add_epi32(_mm512_cvtepi8_epi32(b0_16),
                                                  _mm512_cvtepi8_epi32(c0_16))),
                mul_128));

        a01_16 = _mm_unpacklo_epi8(a0_16, b0_16);
        a0_16  = _mm_unpackhi_epi8(a0_16, b0_16);
        c01_16 = _mm_unpacklo_epi8(c0_16, d0_16);
        c0_16  = _mm_unpackhi_epi8(c0_16, d0_16);

        b0_16  = _mm_unpacklo_epi16(a01_16, c01_16);
        a01_16 = _mm_unpackhi_epi16(a01_16, c01_16);
        d0_16  = _mm_unpacklo_epi16(a0_16, c0_16);
        c01_16 = _mm_unpackhi_epi16(a0_16, c0_16);

        a0_zmm = _mm512_castsi128_si512(b0_16);
        a0_zmm = _mm512_inserti32x4(a0_zmm, a01_16, 0x1);
        a0_zmm = _mm512_inserti32x4(a0_zmm, d0_16, 0x2);
        a0_zmm = _mm512_inserti32x4(a0_zmm, c01_16, 0x3);

        _mm512_storeu_si512(pack_b + ((kr_new + 0) * NR), a0_zmm);
    }
    _mm512_storeu_si512(colsum, sum1);
}

/* ============================ col-major (transb='T') ==================== *
 * Fork of the s8 col-major packer. For transb='T' a column of B is contiguous
 * in the k dimension (k-stride == 1 nibble; cs_b is the n-stride in nibbles).
 * Each of the 16 columns in a tile is widened (its k-run) to s8 in-register,
 * then the identical transpose ladder + VNNI store + column-sum reduction from
 * the s8 packer is applied. This removes the intermediate s8 scratch buffer. */

/* Widen a contiguous k-run of `kcount` (<=64) nibbles of a single column,
 * starting at nibble index `nib`, into the low `kcount` lanes of a __m512i
 * (higher lanes zeroed). Odd start nibbles use the multishift "_ODD" realign;
 * masked loads keep it fault-safe for short runs and partial columns. */
static inline __m512i
s8s4_widen_col64(const uint8_t* b_s4,
                 md_t           nib,
                 md_t           kcount,
                 __m512i        shift_idx_64,
                 __m512i        conv_shift_64,
                 __m512i        sign_comp_64)
{
    __m512i out;
    if ((nib & 1) == 0) {
        __mmask32 bmask = (kcount >= 64)
                              ? (__mmask32)0xFFFFFFFF
                              : (__mmask32)((1u << ((kcount + 1) >> 1)) - 1u);
        __m256i   h     = _mm256_maskz_loadu_epi8(bmask, b_s4 + (nib >> 1));
        CVT_INT4_TO_INT8_64ELEM_MULTISHIFT(h, out, shift_idx_64, sign_comp_64,
                                           TRUE);
    } else {
        __mmask32 bmask = (kcount >= 64)
                              ? (__mmask32)0xFFFFFFFF
                              : (__mmask32)((1u << ((kcount >> 1) + 1)) - 1u);
        __m256i   h     = _mm256_maskz_loadu_epi8(bmask, b_s4 + (nib >> 1));
        __m256i   hl =
            (kcount >= 64)
                  ? _mm256_maskz_loadu_epi8(0x80000000, b_s4 + (nib >> 1) + 1)
                  : _mm256_setzero_si256();
        CVT_INT4_TO_INT8_64ELEM_MULTISHIFT_ODD(
            h, hl, out, shift_idx_64, conv_shift_64, sign_comp_64, TRUE);
    }
    if (kcount < 64) {
        out = _mm512_maskz_mov_epi8((__mmask64)((1ULL << kcount) - 1ULL), out);
    }
    return out;
}

#define S8S4_UNPACKHILO32_AVX512                                               \
    b_reg[0]  = _mm512_unpacklo_epi32(a_reg[0], a_reg[1]);                     \
    b_reg[2]  = _mm512_unpacklo_epi32(a_reg[2], a_reg[3]);                     \
    b_reg[4]  = _mm512_unpacklo_epi32(a_reg[4], a_reg[5]);                     \
    b_reg[6]  = _mm512_unpacklo_epi32(a_reg[6], a_reg[7]);                     \
    b_reg[8]  = _mm512_unpacklo_epi32(a_reg[8], a_reg[9]);                     \
    b_reg[10] = _mm512_unpacklo_epi32(a_reg[10], a_reg[11]);                   \
    b_reg[12] = _mm512_unpacklo_epi32(a_reg[12], a_reg[13]);                   \
    b_reg[14] = _mm512_unpacklo_epi32(a_reg[14], a_reg[15]);                   \
    b_reg[1]  = _mm512_unpackhi_epi32(a_reg[0], a_reg[1]);                     \
    b_reg[3]  = _mm512_unpackhi_epi32(a_reg[2], a_reg[3]);                     \
    b_reg[5]  = _mm512_unpackhi_epi32(a_reg[4], a_reg[5]);                     \
    b_reg[7]  = _mm512_unpackhi_epi32(a_reg[6], a_reg[7]);                     \
    b_reg[9]  = _mm512_unpackhi_epi32(a_reg[8], a_reg[9]);                     \
    b_reg[11] = _mm512_unpackhi_epi32(a_reg[10], a_reg[11]);                   \
    b_reg[13] = _mm512_unpackhi_epi32(a_reg[12], a_reg[13]);                   \
    b_reg[15] = _mm512_unpackhi_epi32(a_reg[14], a_reg[15]);

#define S8S4_UNPACKHILO64_AVX512                                               \
    a_reg[0]  = _mm512_unpacklo_epi64(b_reg[0], b_reg[2]);                     \
    a_reg[1]  = _mm512_unpacklo_epi64(b_reg[4], b_reg[6]);                     \
    a_reg[2]  = _mm512_unpacklo_epi64(b_reg[8], b_reg[10]);                    \
    a_reg[3]  = _mm512_unpacklo_epi64(b_reg[12], b_reg[14]);                   \
    a_reg[4]  = _mm512_unpacklo_epi64(b_reg[1], b_reg[3]);                     \
    a_reg[5]  = _mm512_unpacklo_epi64(b_reg[5], b_reg[7]);                     \
    a_reg[6]  = _mm512_unpacklo_epi64(b_reg[9], b_reg[11]);                    \
    a_reg[7]  = _mm512_unpacklo_epi64(b_reg[13], b_reg[15]);                   \
    a_reg[8]  = _mm512_unpackhi_epi64(b_reg[0], b_reg[2]);                     \
    a_reg[9]  = _mm512_unpackhi_epi64(b_reg[4], b_reg[6]);                     \
    a_reg[10] = _mm512_unpackhi_epi64(b_reg[8], b_reg[10]);                    \
    a_reg[11] = _mm512_unpackhi_epi64(b_reg[12], b_reg[14]);                   \
    a_reg[12] = _mm512_unpackhi_epi64(b_reg[1], b_reg[3]);                     \
    a_reg[13] = _mm512_unpackhi_epi64(b_reg[5], b_reg[7]);                     \
    a_reg[14] = _mm512_unpackhi_epi64(b_reg[9], b_reg[11]);                    \
    a_reg[15] = _mm512_unpackhi_epi64(b_reg[13], b_reg[15]);

#define S8S4_PERMUTEX2_VAR64_AVX512                                            \
    b_reg[0]  = _mm512_permutex2var_epi64(a_reg[0], selector1, a_reg[1]);      \
    b_reg[1]  = _mm512_permutex2var_epi64(a_reg[2], selector1, a_reg[3]);      \
    b_reg[2]  = _mm512_permutex2var_epi64(a_reg[8], selector1, a_reg[9]);      \
    b_reg[3]  = _mm512_permutex2var_epi64(a_reg[10], selector1, a_reg[11]);    \
    b_reg[4]  = _mm512_permutex2var_epi64(a_reg[4], selector1, a_reg[5]);      \
    b_reg[5]  = _mm512_permutex2var_epi64(a_reg[6], selector1, a_reg[7]);      \
    b_reg[6]  = _mm512_permutex2var_epi64(a_reg[12], selector1, a_reg[13]);    \
    b_reg[7]  = _mm512_permutex2var_epi64(a_reg[14], selector1, a_reg[15]);    \
    b_reg[8]  = _mm512_permutex2var_epi64(a_reg[0], selector2, a_reg[1]);      \
    b_reg[9]  = _mm512_permutex2var_epi64(a_reg[2], selector2, a_reg[3]);      \
    b_reg[10] = _mm512_permutex2var_epi64(a_reg[8], selector2, a_reg[9]);      \
    b_reg[11] = _mm512_permutex2var_epi64(a_reg[10], selector2, a_reg[11]);    \
    b_reg[12] = _mm512_permutex2var_epi64(a_reg[4], selector2, a_reg[5]);      \
    b_reg[13] = _mm512_permutex2var_epi64(a_reg[6], selector2, a_reg[7]);      \
    b_reg[14] = _mm512_permutex2var_epi64(a_reg[12], selector2, a_reg[13]);    \
    b_reg[15] = _mm512_permutex2var_epi64(a_reg[14], selector2, a_reg[15]);

#define S8S4_SHUFFLE64x2_AVX512                                                \
    a_reg[0]  = _mm512_shuffle_i64x2(b_reg[0], b_reg[1], 0x44);                \
    a_reg[1]  = _mm512_shuffle_i64x2(b_reg[2], b_reg[3], 0x44);                \
    a_reg[2]  = _mm512_shuffle_i64x2(b_reg[4], b_reg[5], 0x44);                \
    a_reg[3]  = _mm512_shuffle_i64x2(b_reg[6], b_reg[7], 0x44);                \
    a_reg[4]  = _mm512_shuffle_i64x2(b_reg[8], b_reg[9], 0x44);                \
    a_reg[5]  = _mm512_shuffle_i64x2(b_reg[10], b_reg[11], 0x44);              \
    a_reg[6]  = _mm512_shuffle_i64x2(b_reg[12], b_reg[13], 0x44);              \
    a_reg[7]  = _mm512_shuffle_i64x2(b_reg[14], b_reg[15], 0x44);              \
    a_reg[8]  = _mm512_shuffle_i64x2(b_reg[0], b_reg[1], 0xEE);                \
    a_reg[9]  = _mm512_shuffle_i64x2(b_reg[2], b_reg[3], 0xEE);                \
    a_reg[10] = _mm512_shuffle_i64x2(b_reg[4], b_reg[5], 0xEE);                \
    a_reg[11] = _mm512_shuffle_i64x2(b_reg[6], b_reg[7], 0xEE);                \
    a_reg[12] = _mm512_shuffle_i64x2(b_reg[8], b_reg[9], 0xEE);                \
    a_reg[13] = _mm512_shuffle_i64x2(b_reg[10], b_reg[11], 0xEE);              \
    a_reg[14] = _mm512_shuffle_i64x2(b_reg[12], b_reg[13], 0xEE);              \
    a_reg[15] = _mm512_shuffle_i64x2(b_reg[14], b_reg[15], 0xEE);

#define S8S4_SUM_16_COLS_K64                                                   \
    for (iter_t i = 0; i < 16; i++) {                                          \
        __m512i sum0, sum1;                                                    \
        sum0 = _mm512_add_epi32(                                               \
            _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(a_reg[i], 0)),      \
            _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(a_reg[i], 1)));     \
        sum1 = _mm512_add_epi32(                                               \
            _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(a_reg[i], 2)),      \
            _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(a_reg[i], 3)));     \
        sum[i + jr] =                                                          \
            _mm512_add_epi32(sum[i + jr], _mm512_add_epi32(sum0, sum1));       \
    }

#define S8S4_SUM_16_COLS_K32                                                   \
    for (iter_t i = 0; i < 16; i++) {                                          \
        sum[i + jr] = _mm512_add_epi32(                                        \
            sum[i + jr],                                                       \
            _mm512_add_epi32(                                                  \
                _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(a_reg[i], 0)),  \
                _mm512_cvtepi8_epi32(                                          \
                    _mm512_extracti32x4_epi32(a_reg[i], 1))));                 \
    }

#define S8S4_SUM_16_COLS_K16                                                   \
    for (iter_t i = 0; i < 16; i++) {                                          \
        sum[i + jr] = _mm512_add_epi32(                                        \
            sum[i + jr],                                                       \
            _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(a_reg[i], 0)));     \
    }

/* Load+widen 16 full columns for a k-tile of `kcount` elements. */
#define S8S4_WIDEN_LOAD_16_COLS(kcount)                                        \
    for (iter_t _i = 0; _i < 16; _i++) {                                       \
        a_reg[_i] = s8s4_widen_col64(                                          \
            b_s4, cs_b * (col0 + jr + _i) + (row0 + kr), (kcount),             \
            shift_idx_64, conv_shift_64, sign_comp_64);                        \
    }

static void
dlp_packb_nr_mult16_s8s4_col(int8_t*        pack_b,
                             int32_t*       colsum,
                             const uint8_t* b_s4,
                             const md_t     cs_b,
                             const md_t     row0,
                             const md_t     col0,
                             const md_t     NR,
                             const md_t     KC)
{
    __m512i selector1 =
        _mm512_setr_epi64(0x0, 0x1, 0x8, 0x9, 0x4, 0x5, 0xC, 0xD);
    __m512i selector2 =
        _mm512_setr_epi64(0x2, 0x3, 0xA, 0xB, 0x6, 0x7, 0xE, 0xF);

    __m512i a_reg[16];
    __m512i b_reg[16];
    __m512i sum[64];
    __m512i mul_128 = _mm512_set1_epi32(7);

    __m512i shift_idx_64;
    MULTISHIFT_32BIT_8_INT4_IDX_64ELEM(shift_idx_64);
    __m512i sign_comp_64 = _mm512_set1_epi8(0x08);
    CREATE_CVT_INT4_INT8_PERM_IDX_64ELEM_ODD_LD(conv_shift_arr_64);
    __m512i conv_shift_64 = _mm512_loadu_epi64(conv_shift_arr_64);

    for (iter_t i = 0; i < 64; i++) {
        sum[i] = _mm512_setzero_si512();
    }

    md_t kr = 0;
    for (kr = 0; (kr + 63) < KC; kr += 64) {
        for (iter_t jr = 0; jr < NR; jr += 16) {
            S8S4_WIDEN_LOAD_16_COLS(64)
            S8S4_SUM_16_COLS_K64
            S8S4_UNPACKHILO32_AVX512
            S8S4_UNPACKHILO64_AVX512
            S8S4_PERMUTEX2_VAR64_AVX512
            S8S4_SHUFFLE64x2_AVX512 for (iter_t s = 0; s < 16; s++)
            {
                _mm512_storeu_si512(pack_b + (jr * 4) + ((kr + s * 4) * NR),
                                    a_reg[s]);
            }
        }
    }
    for (; (kr + 31) < KC; kr += 32) {
        for (iter_t jr = 0; jr < NR; jr += 16) {
            S8S4_WIDEN_LOAD_16_COLS(32)
            S8S4_SUM_16_COLS_K32
            S8S4_UNPACKHILO32_AVX512
            S8S4_UNPACKHILO64_AVX512
            S8S4_PERMUTEX2_VAR64_AVX512
            S8S4_SHUFFLE64x2_AVX512 for (iter_t s = 0; s < 8; s++)
            {
                _mm512_storeu_si512(pack_b + (jr * 4) + ((kr + s * 4) * NR),
                                    a_reg[s]);
            }
        }
    }
    for (; (kr + 15) < KC; kr += 16) {
        for (iter_t jr = 0; jr < NR; jr += 16) {
            S8S4_WIDEN_LOAD_16_COLS(16)
            S8S4_SUM_16_COLS_K16
            S8S4_UNPACKHILO32_AVX512
            S8S4_UNPACKHILO64_AVX512
            S8S4_PERMUTEX2_VAR64_AVX512
            S8S4_SHUFFLE64x2_AVX512 for (iter_t s = 0; s < 4; s++)
            {
                _mm512_storeu_si512(pack_b + (jr * 4) + ((kr + s * 4) * NR),
                                    a_reg[s]);
            }
        }
    }
    for (; (kr + 7) < KC; kr += 8) {
        for (iter_t jr = 0; jr < NR; jr += 16) {
            S8S4_WIDEN_LOAD_16_COLS(8)
            S8S4_SUM_16_COLS_K16
            S8S4_UNPACKHILO32_AVX512
            S8S4_UNPACKHILO64_AVX512
            S8S4_PERMUTEX2_VAR64_AVX512
            S8S4_SHUFFLE64x2_AVX512 for (iter_t s = 0; s < 2; s++)
            {
                _mm512_storeu_si512(pack_b + (jr * 4) + ((kr + s * 4) * NR),
                                    a_reg[s]);
            }
        }
    }
    for (; (kr + 3) < KC; kr += 4) {
        for (iter_t jr = 0; jr < NR; jr += 16) {
            S8S4_WIDEN_LOAD_16_COLS(4)
            S8S4_SUM_16_COLS_K16
            S8S4_UNPACKHILO32_AVX512
            S8S4_UNPACKHILO64_AVX512
            S8S4_PERMUTEX2_VAR64_AVX512
            S8S4_SHUFFLE64x2_AVX512 _mm512_storeu_si512(
                pack_b + (jr * 4) + (kr * NR), a_reg[0]);
        }
    }
    for (; (kr + 2) < KC; kr += 3) {
        for (iter_t jr = 0; jr < NR; jr += 16) {
            S8S4_WIDEN_LOAD_16_COLS(3)
            S8S4_SUM_16_COLS_K16
            S8S4_UNPACKHILO32_AVX512
            S8S4_UNPACKHILO64_AVX512
            S8S4_PERMUTEX2_VAR64_AVX512
            S8S4_SHUFFLE64x2_AVX512 _mm512_storeu_si512(
                pack_b + (jr * 4) + (kr * NR), a_reg[0]);
        }
    }
    for (; (kr + 1) < KC; kr += 2) {
        for (iter_t jr = 0; jr < NR; jr += 16) {
            S8S4_WIDEN_LOAD_16_COLS(2)
            S8S4_SUM_16_COLS_K16
            S8S4_UNPACKHILO32_AVX512
            S8S4_UNPACKHILO64_AVX512
            S8S4_PERMUTEX2_VAR64_AVX512
            S8S4_SHUFFLE64x2_AVX512 _mm512_storeu_si512(
                pack_b + (jr * 4) + (kr * NR), a_reg[0]);
        }
    }
    for (; kr < KC; kr += 1) {
        for (iter_t jr = 0; jr < NR; jr += 16) {
            S8S4_WIDEN_LOAD_16_COLS(1)
            S8S4_SUM_16_COLS_K16
            S8S4_UNPACKHILO32_AVX512
            S8S4_UNPACKHILO64_AVX512
            S8S4_PERMUTEX2_VAR64_AVX512
            S8S4_SHUFFLE64x2_AVX512 _mm512_storeu_si512(
                pack_b + (jr * 4) + (kr * NR), a_reg[0]);
        }
    }

    for (iter_t jr = 0; jr < NR; jr += 16) {
        __m512i sum0, sum1;
        sum0 = _mm512_set_epi32(_mm512_reduce_add_epi32(sum[jr + 15]),
                                _mm512_reduce_add_epi32(sum[jr + 14]),
                                _mm512_reduce_add_epi32(sum[jr + 13]),
                                _mm512_reduce_add_epi32(sum[jr + 12]),
                                _mm512_reduce_add_epi32(sum[jr + 11]),
                                _mm512_reduce_add_epi32(sum[jr + 10]),
                                _mm512_reduce_add_epi32(sum[jr + 9]),
                                _mm512_reduce_add_epi32(sum[jr + 8]),
                                _mm512_reduce_add_epi32(sum[jr + 7]),
                                _mm512_reduce_add_epi32(sum[jr + 6]),
                                _mm512_reduce_add_epi32(sum[jr + 5]),
                                _mm512_reduce_add_epi32(sum[jr + 4]),
                                _mm512_reduce_add_epi32(sum[jr + 3]),
                                _mm512_reduce_add_epi32(sum[jr + 2]),
                                _mm512_reduce_add_epi32(sum[jr + 1]),
                                _mm512_reduce_add_epi32(sum[jr + 0]));
        sum0 = _mm512_sllv_epi32(sum0, mul_128);
        sum1 = _mm512_loadu_si512(colsum + jr);
        sum1 = _mm512_add_epi32(sum0, sum1);
        _mm512_storeu_si512(colsum + jr, sum1);
    }
}

static void
dlp_packb_nrlt16_s8s4_col(int8_t*        pack_b,
                          int32_t*       colsum,
                          const uint8_t* b_s4,
                          const md_t     cs_b,
                          const md_t     row0,
                          const md_t     col0,
                          const md_t     KC,
                          const md_t     n0_partial_rem)
{
    md_t NR = 16;

    __m512i selector1 =
        _mm512_setr_epi64(0x0, 0x1, 0x8, 0x9, 0x4, 0x5, 0xC, 0xD);
    __m512i selector2 =
        _mm512_setr_epi64(0x2, 0x3, 0xA, 0xB, 0x6, 0x7, 0xE, 0xF);

    __m512i a_reg[16];
    __m512i b_reg[16];
    __m512i sum[16];
    __m512i mul_128 = _mm512_set1_epi32(7);

    __m512i shift_idx_64;
    MULTISHIFT_32BIT_8_INT4_IDX_64ELEM(shift_idx_64);
    __m512i sign_comp_64 = _mm512_set1_epi8(0x08);
    CREATE_CVT_INT4_INT8_PERM_IDX_64ELEM_ODD_LD(conv_shift_arr_64);
    __m512i conv_shift_64 = _mm512_loadu_epi64(conv_shift_arr_64);

    for (iter_t i = 0; i < 16; i++) {
        sum[i] = _mm512_setzero_si512();
    }

/* Load+widen the `n0_partial_rem` valid columns; zero the rest. */
#define S8S4_WIDEN_LOAD_LT16_COLS(kcount)                                      \
    for (jr = 0; jr < n0_partial_rem; jr += 1) {                               \
        a_reg[jr] =                                                            \
            s8s4_widen_col64(b_s4, cs_b * (col0 + jr) + (row0 + kr), (kcount), \
                             shift_idx_64, conv_shift_64, sign_comp_64);       \
    }                                                                          \
    for (; jr < NR; jr++) {                                                    \
        a_reg[jr] = _mm512_setzero_si512();                                    \
    }                                                                          \
    jr = 0;

    md_t kr = 0, jr = 0;
    for (kr = 0; (kr + 63) < KC; kr += 64) {
        S8S4_WIDEN_LOAD_LT16_COLS(64)
        S8S4_SUM_16_COLS_K64
        S8S4_UNPACKHILO32_AVX512
        S8S4_UNPACKHILO64_AVX512
        S8S4_PERMUTEX2_VAR64_AVX512
        S8S4_SHUFFLE64x2_AVX512 for (iter_t s = 0; s < 16; s++)
        {
            _mm512_storeu_si512(pack_b + ((kr + s * 4) * NR), a_reg[s]);
        }
    }
    for (; (kr + 31) < KC; kr += 32) {
        S8S4_WIDEN_LOAD_LT16_COLS(32)
        S8S4_SUM_16_COLS_K32
        S8S4_UNPACKHILO32_AVX512
        S8S4_UNPACKHILO64_AVX512
        S8S4_PERMUTEX2_VAR64_AVX512
        S8S4_SHUFFLE64x2_AVX512 for (iter_t s = 0; s < 8; s++)
        {
            _mm512_storeu_si512(pack_b + ((kr + s * 4) * NR), a_reg[s]);
        }
    }
    for (; (kr + 15) < KC; kr += 16) {
        S8S4_WIDEN_LOAD_LT16_COLS(16)
        S8S4_SUM_16_COLS_K16
        S8S4_UNPACKHILO32_AVX512
        S8S4_UNPACKHILO64_AVX512
        S8S4_PERMUTEX2_VAR64_AVX512
        S8S4_SHUFFLE64x2_AVX512 for (iter_t s = 0; s < 4; s++)
        {
            _mm512_storeu_si512(pack_b + ((kr + s * 4) * NR), a_reg[s]);
        }
    }
    for (; (kr + 7) < KC; kr += 8) {
        S8S4_WIDEN_LOAD_LT16_COLS(8)
        S8S4_SUM_16_COLS_K16
        S8S4_UNPACKHILO32_AVX512
        S8S4_UNPACKHILO64_AVX512
        S8S4_PERMUTEX2_VAR64_AVX512
        S8S4_SHUFFLE64x2_AVX512 for (iter_t s = 0; s < 2; s++)
        {
            _mm512_storeu_si512(pack_b + ((kr + s * 4) * NR), a_reg[s]);
        }
    }
    for (; (kr + 3) < KC; kr += 4) {
        S8S4_WIDEN_LOAD_LT16_COLS(4)
        S8S4_SUM_16_COLS_K16
        S8S4_UNPACKHILO32_AVX512
        S8S4_UNPACKHILO64_AVX512
        S8S4_PERMUTEX2_VAR64_AVX512
        S8S4_SHUFFLE64x2_AVX512 _mm512_storeu_si512(pack_b + (kr * NR),
                                                    a_reg[0]);
    }
    for (; (kr + 2) < KC; kr += 3) {
        S8S4_WIDEN_LOAD_LT16_COLS(3)
        S8S4_SUM_16_COLS_K16
        S8S4_UNPACKHILO32_AVX512
        S8S4_UNPACKHILO64_AVX512
        S8S4_PERMUTEX2_VAR64_AVX512
        S8S4_SHUFFLE64x2_AVX512 _mm512_storeu_si512(pack_b + (kr * NR),
                                                    a_reg[0]);
    }
    for (; (kr + 1) < KC; kr += 2) {
        S8S4_WIDEN_LOAD_LT16_COLS(2)
        S8S4_SUM_16_COLS_K16
        S8S4_UNPACKHILO32_AVX512
        S8S4_UNPACKHILO64_AVX512
        S8S4_PERMUTEX2_VAR64_AVX512
        S8S4_SHUFFLE64x2_AVX512 _mm512_storeu_si512(pack_b + (kr * NR),
                                                    a_reg[0]);
    }
    for (; kr < KC; kr += 1) {
        S8S4_WIDEN_LOAD_LT16_COLS(1)
        S8S4_SUM_16_COLS_K16
        S8S4_UNPACKHILO32_AVX512
        S8S4_UNPACKHILO64_AVX512
        S8S4_PERMUTEX2_VAR64_AVX512
        S8S4_SHUFFLE64x2_AVX512 _mm512_storeu_si512(pack_b + (kr * NR),
                                                    a_reg[0]);
    }
#undef S8S4_WIDEN_LOAD_LT16_COLS

    __m512i sum0, sum1;
    sum0 = _mm512_set_epi32(
        _mm512_reduce_add_epi32(sum[15]), _mm512_reduce_add_epi32(sum[14]),
        _mm512_reduce_add_epi32(sum[13]), _mm512_reduce_add_epi32(sum[12]),
        _mm512_reduce_add_epi32(sum[11]), _mm512_reduce_add_epi32(sum[10]),
        _mm512_reduce_add_epi32(sum[9]), _mm512_reduce_add_epi32(sum[8]),
        _mm512_reduce_add_epi32(sum[7]), _mm512_reduce_add_epi32(sum[6]),
        _mm512_reduce_add_epi32(sum[5]), _mm512_reduce_add_epi32(sum[4]),
        _mm512_reduce_add_epi32(sum[3]), _mm512_reduce_add_epi32(sum[2]),
        _mm512_reduce_add_epi32(sum[1]), _mm512_reduce_add_epi32(sum[0]));
    sum0 = _mm512_sllv_epi32(sum0, mul_128);
    sum1 = _mm512_loadu_epi32(colsum);
    sum1 = _mm512_add_epi32(sum0, sum1);
    _mm512_storeu_si512(colsum, sum1);
}

void
dlp_packb_nr64_s8s4s32os32_col_major(int8_t*        pack_b_buffer,
                                     int32_t*       pack_b_column_sum,
                                     const uint8_t* b_s4,
                                     const md_t     cs_b,
                                     const md_t     row0,
                                     const md_t     col0,
                                     const md_t     NC,
                                     const md_t     KC,
                                     md_t*          rs_b_out,
                                     md_t*          cs_b_out)
{
    md_t NR = 64;

    md_t n_full_pieces            = NC / NR;
    md_t n_full_pieces_loop_limit = n_full_pieces * NR;
    md_t n_partial_pieces         = NC % NR;

    md_t k_partial_pieces = KC % 4;
    md_t KC_updated       = KC;
    if (k_partial_pieces > 0) {
        KC_updated += (4 - k_partial_pieces);
    }

    for (iter_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR) {
        dlp_packb_nr_mult16_s8s4_col(pack_b_buffer + (jc * KC_updated),
                                     pack_b_column_sum + jc, b_s4, cs_b, row0,
                                     col0 + jc, 64, KC);
    }

    if (n_partial_pieces > 0) {
        md_t n0_partial_rem  = n_partial_pieces % 16;
        md_t n0_partial_pack = 0;

        md_t n0_48 = n_partial_pieces / 48;
        md_t n0_32 = n_partial_pieces / 32;
        md_t n0_16 = n_partial_pieces / 16;

        if (n0_48 == 1) {
            dlp_packb_nr_mult16_s8s4_col(
                pack_b_buffer + (n_full_pieces_loop_limit * KC_updated),
                pack_b_column_sum + n_full_pieces_loop_limit, b_s4, cs_b, row0,
                col0 + n_full_pieces_loop_limit, 48, KC);
            n0_partial_pack = 48;
        } else if (n0_32 == 1) {
            dlp_packb_nr_mult16_s8s4_col(
                pack_b_buffer + (n_full_pieces_loop_limit * KC_updated),
                pack_b_column_sum + n_full_pieces_loop_limit, b_s4, cs_b, row0,
                col0 + n_full_pieces_loop_limit, 32, KC);
            n0_partial_pack = 32;
        } else if (n0_16 == 1) {
            dlp_packb_nr_mult16_s8s4_col(
                pack_b_buffer + (n_full_pieces_loop_limit * KC_updated),
                pack_b_column_sum + n_full_pieces_loop_limit, b_s4, cs_b, row0,
                col0 + n_full_pieces_loop_limit, 16, KC);
            n0_partial_pack = 16;
        }

        if (n0_partial_rem > 0) {
            dlp_packb_nrlt16_s8s4_col(
                pack_b_buffer + (n_full_pieces_loop_limit * KC_updated)
                    + (n0_partial_pack * KC_updated),
                pack_b_column_sum + n_full_pieces_loop_limit + n0_partial_pack,
                b_s4, cs_b, row0,
                col0 + n_full_pieces_loop_limit + n0_partial_pack, KC,
                n0_partial_rem);
        }
    }

    *rs_b_out = NR * 4;
    *cs_b_out = NR;
}

/* ==================== linear s4<->s8 stream converters ==================== *
 * AVX-512 vectorized s4 <-> s8 converters for the s8s4 symmetric-quant path.
 * These are the vectorized counterparts of the scalar helpers in
 * kernels/s8s8s32/dlp_gemm_packb_s8s4.h and share the exact same nibble
 * convention (even linear index -> low nibble, odd -> high nibble; s4 range
 * [-8, 7] sign-extended). They operate on CONTIGUOUS, even-aligned streams
 * (the widened compute-panel and the packed weight region), which is why the
 * multishift widen / permute compress apply directly. Any leftover tail (or a
 * non-64/128 multiple) is finished with the scalar fallback so results are
 * bit-identical to the scalar helpers. */

/*
 * Widen n_elems contiguous s4 nibbles (from a byte-packed, even-aligned source)
 * into s8, sign-extending each nibble. 64 elements (32 source bytes) per vector
 * iteration via the multishift upscale; scalar tail for the remainder.
 */
void
dlp_cvt_s4_to_s8_linear_avx512(int8_t* dst, const uint8_t* src, md_t n_elems)
{
    __m512i shift_idx;
    MULTISHIFT_32BIT_8_INT4_IDX_64ELEM(shift_idx);
    __m512i sign_comp = _mm512_set1_epi8(0x08);

    md_t i = 0;
    for (; (i + 64) <= n_elems; i += 64) {
        __m256i in = _mm256_loadu_si256((const __m256i*)(src + (i >> 1)));
        __m512i out;
        CVT_INT4_TO_INT8_64ELEM_MULTISHIFT(in, out, shift_idx, sign_comp, TRUE);
        _mm512_storeu_si512((void*)(dst + i), out);
    }

    // Scalar tail (bit-identical to dlp_cvt_s4_to_s8_linear).
    for (; i < n_elems; i++) {
        dst[i] = dlp_s4_nibble_to_s8(src[i >> 1], (int)(i & 1));
    }
}

/*
 * Compress n_elems contiguous s8 values (each in [-8, 7]) into packed s4
 * nibbles. 128 elements (2 ZMM) -> 64 bytes per vector iteration; scalar tail
 * (including a possible 64-element remainder) for the rest.
 */
void
dlp_cvt_s8_to_s4_linear_avx512(uint8_t* dst, const int8_t* src, md_t n_elems)
{
    CREATE_CVT_INT8_INT4_PERM_IDX_64ELEM_2_ZMM_REG(even_idx_arr);
    __m512i even_perm_idx = _mm512_loadu_si512((const void*)even_idx_arr);
    __m512i all_1s        = _mm512_set1_epi8(0x01);
    __m512i odd_perm_idx  = _mm512_add_epi8(even_perm_idx, all_1s);
    __m512i clear_hi_bits = _mm512_set1_epi8(0x0F);

    md_t i = 0;
    for (; (i + 128) <= n_elems; i += 128) {
        __m512i in0 = _mm512_loadu_si512((const void*)(src + i));
        __m512i in1 = _mm512_loadu_si512((const void*)(src + i + 64));
        __m512i out;
        CVT_INT8_INT4_64ELEM_2_ZMM_REG(in0, in1, out, even_perm_idx,
                                       odd_perm_idx, clear_hi_bits);
        _mm512_storeu_si512((void*)(dst + (i >> 1)), out);
    }

    // Scalar tail (bit-identical to dlp_cvt_s8_to_s4_linear).
    for (; (i + 1) < n_elems; i += 2) {
        uint8_t lo  = (uint8_t)(src[i] & 0x0F);
        uint8_t hi  = (uint8_t)(src[i + 1] & 0x0F);
        dst[i >> 1] = (uint8_t)(lo | (hi << 4));
    }
    if (i < n_elems) {
        dst[i >> 1] = (uint8_t)(src[i] & 0x0F);
    }
}
