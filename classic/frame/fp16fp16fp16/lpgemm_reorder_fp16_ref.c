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

#include <immintrin.h>
#include <string.h>

#include "fp16fp16fp16/lpgemm_reorder_fp16.h"
/*
    Below are the reference packb functions which are
    varied based on block size NR (128, 96, 64, 32, lt32) and
    order (row / column (transpose)).

    NR=128 is the full panel (4 ZMM registers).
    Chunk hierarchy: 128 -> 96 -> 64 -> 32 -> lt32 (padded to 32)
*/

static void
packb_nr_f16f16f16of16_row_major_ref(float16*       pack_b,
                                     const float16* b,
                                     const md_t     NR,
                                     const md_t     ldb,
                                     const md_t     KC)
{
    // K-MAJOR packing: pack each k-row with NR elements contiguous
    for (iter_t kr = 0; kr < KC; kr++) {
        const float16* inp0  = (b + (ldb * kr));
        float16*       outp0 = (pack_b + (kr * NR));
        for (iter_t i = 0; i < NR; i++)
            *outp0++ = *inp0++;
    }
}

static void
packb_nrlt32_f16f16f16of16_row_major_ref(float16*       pack_b,
                                         const float16* b,
                                         const md_t     ldb,
                                         const md_t     KC,
                                         const md_t     n0_partial_rem)
{
    md_t NR = 32; // Pad to 32 (one ZMM register)
    // K-MAJOR packing with zero-padding for fringe
    for (iter_t kr = 0; kr < KC; kr++) {
        const float16* inp0  = (b + (ldb * kr));
        float16*       outp0 = (pack_b + (kr * NR));
        for (iter_t i = 0; i < n0_partial_rem; i++)
            *outp0++ = *inp0++;
        // Zero-pad remainder to NR=32
        for (iter_t i = n0_partial_rem; i < NR; i++)
            *outp0++ = 0;
    }
}

static void
packb_f16f16f16of16_row_major_ref(float16*       pack_b,
                                  const float16* b,
                                  const md_t     ldb,
                                  const md_t     NC,
                                  const md_t     KC,
                                  const md_t     NR,
                                  md_t*          rs_b,
                                  md_t*          cs_b)
{
    md_t n_full_pieces            = NC / NR;
    md_t n_full_pieces_loop_limit = n_full_pieces * NR;
    md_t n_partial_pieces         = NC % NR;

    // K-MAJOR packing: pack full NR-width chunks
    for (iter_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR) {
        packb_nr_f16f16f16of16_row_major_ref((pack_b + (jc * KC)), (b + jc), NR,
                                             ldb, KC);
    }

    if (n_partial_pieces > 0) {
        if (NR >= 32) {
            // Production path: subdivide into optimal chunks (96, 64, or 32)
            md_t n0_partial_rem  = n_partial_pieces % 32;
            md_t n0_partial_pack = 0;

            md_t n0_96 = n_partial_pieces / 96;
            md_t n0_64 = n_partial_pieces / 64;
            md_t n0_32 = n_partial_pieces / 32;

            if (n0_96 == 1) {
                packb_nr_f16f16f16of16_row_major_ref(
                    (pack_b + (n_full_pieces_loop_limit * KC)),
                    (b + n_full_pieces_loop_limit), 96, ldb, KC);
                n0_partial_pack = 96;
            } else if (n0_64 == 1) {
                packb_nr_f16f16f16of16_row_major_ref(
                    (pack_b + (n_full_pieces_loop_limit * KC)),
                    (b + n_full_pieces_loop_limit), 64, ldb, KC);
                n0_partial_pack = 64;
            } else if (n0_32 == 1) {
                packb_nr_f16f16f16of16_row_major_ref(
                    (pack_b + (n_full_pieces_loop_limit * KC)),
                    (b + n_full_pieces_loop_limit), 32, ldb, KC);
                n0_partial_pack = 32;
            }

            if (n0_partial_rem > 0) {
                packb_nrlt32_f16f16f16of16_row_major_ref(
                    (pack_b + (n_full_pieces_loop_limit * KC)
                     + (n0_partial_pack * KC)),
                    (b + n_full_pieces_loop_limit + n0_partial_pack), ldb, KC,
                    n0_partial_rem);
            }
        } else {
            // Test path: simple approach without subdivision
            packb_nr_f16f16f16of16_row_major_ref(
                (pack_b + (n_full_pieces_loop_limit * KC)),
                (b + n_full_pieces_loop_limit), NR, ldb, KC);
        }
    }

    *rs_b = NR; // Stride to next k-row
    *cs_b = 1;  // Stride to next n within same k-row
}

static void
packb_nr_f16f16f16of16_col_major_ref(float16*       pack_b_buffer,
                                     const float16* b,
                                     const md_t     NR,
                                     const md_t     ldb,
                                     const md_t     KC,
                                     const md_t     n0_partial_rem)
{
    // K-MAJOR packing: n-outer loop (cache-friendly for col-major input)
    // Match F32 implementation for optimal cache behavior
    for (iter_t i = 0; i < n0_partial_rem; i++) {
        const float16* inp  = (b + (ldb * i));   // Column i start
        float16*       outp = pack_b_buffer + i; // Output position i
        for (iter_t j = 0; j < KC; j++) {
            *(outp + (j * NR)) = *inp++; // Sequential read within column
        }
    }
    // Zero-pad remainder columns
    for (iter_t i = n0_partial_rem; i < NR; i++) {
        float16* outp = pack_b_buffer + i;
        for (iter_t j = 0; j < KC; j++) {
            *(outp + (j * NR)) = 0;
        }
    }
}

static void
packb_f16f16f16of16_col_major_ref(float16*       pack_b_buffer,
                                  const float16* b,
                                  const md_t     ldb,
                                  const md_t     NC,
                                  const md_t     KC,
                                  const md_t     NR,
                                  md_t*          rs_b,
                                  md_t*          cs_b)
{
    md_t n_full_pieces            = NC / NR;
    md_t n_full_pieces_loop_limit = n_full_pieces * NR;
    md_t n_partial_pieces         = NC % NR;

    // Pack full NR-width chunks
    for (iter_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR) {
        packb_nr_f16f16f16of16_col_major_ref(pack_b_buffer + (jc * KC),
                                             b + (jc * ldb), NR, ldb, KC, NR);
    }

    // Handle partial piece
    if (n_partial_pieces > 0) {
        if (NR >= 32) {
            // Production path: subdivide into optimal chunks (96, 64, or 32)
            md_t n0_partial_rem  = n_partial_pieces % 32;
            md_t n0_partial_pack = 0;

            md_t n0_96 = n_partial_pieces / 96;
            md_t n0_64 = n_partial_pieces / 64;
            md_t n0_32 = n_partial_pieces / 32;

            if (n0_96 == 1) {
                packb_nr_f16f16f16of16_col_major_ref(
                    (pack_b_buffer + (n_full_pieces_loop_limit * KC)),
                    (b + n_full_pieces_loop_limit * ldb), 96, ldb, KC, 96);
                n0_partial_pack = 96;
            } else if (n0_64 == 1) {
                packb_nr_f16f16f16of16_col_major_ref(
                    (pack_b_buffer + (n_full_pieces_loop_limit * KC)),
                    (b + n_full_pieces_loop_limit * ldb), 64, ldb, KC, 64);
                n0_partial_pack = 64;
            } else if (n0_32 == 1) {
                packb_nr_f16f16f16of16_col_major_ref(
                    (pack_b_buffer + (n_full_pieces_loop_limit * KC)),
                    (b + n_full_pieces_loop_limit * ldb), 32, ldb, KC, 32);
                n0_partial_pack = 32;
            }

            if (n0_partial_rem > 0) {
                packb_nr_f16f16f16of16_col_major_ref(
                    (pack_b_buffer + (n_full_pieces_loop_limit * KC)
                     + (n0_partial_pack * KC)),
                    (b + (n_full_pieces_loop_limit + n0_partial_pack) * ldb),
                    32, ldb, KC, n0_partial_rem);
            }
        } else {
            // Test path: simple approach without subdivision
            packb_nr_f16f16f16of16_col_major_ref(
                (pack_b_buffer + (n_full_pieces_loop_limit * KC)),
                (b + n_full_pieces_loop_limit * ldb), NR, ldb, KC,
                n_partial_pieces);
        }
    }

    *rs_b = NR; // Stride to next k-row (K-MAJOR layout)
    *cs_b = 1;  // Stride to next n within same k-row
}

void
packb_f16f16f16of16_reference(float16*       pack_b,
                              const float16* b,
                              const md_t     rs_b,
                              const md_t     cs_b,
                              const md_t     NC,
                              const md_t     KC,
                              const md_t     NR,
                              md_t*          rs_p,
                              md_t*          cs_p)
{
    if (cs_b == 1) {
        packb_f16f16f16of16_row_major_ref(pack_b, b, rs_b, NC, KC, NR, rs_p,
                                          cs_p);
    } else {
        packb_f16f16f16of16_col_major_ref(pack_b, b, cs_b, NC, KC, NR, rs_p,
                                          cs_p);
    }
}

static void
unpackb_nr_f16f16f16of16_ref(float16*       b,
                             const float16* unpack_b_buffer,
                             const md_t     NR,
                             const md_t     rs_b,
                             const md_t     cs_b,
                             const md_t     KC,
                             const md_t     n0_actual)
{
    // Reverse of K-MAJOR packing: packed[k*NR+n] → output B[k][n]
    for (iter_t kr = 0; kr < KC; kr++) {
        const float16* inp  = unpack_b_buffer + (kr * NR);
        float16*       outp = b + (rs_b * kr);

        for (iter_t i = 0; i < n0_actual; i++) {
            *outp = *inp++;
            outp += cs_b;
        }
    }
}

void
unpackb_f16f16f16of16_reference(float16*   b,
                                float16*   unpack_b_buffer,
                                const md_t NC,
                                const md_t KC,
                                const md_t NR,
                                md_t       rs_b,
                                md_t       cs_b)
{
    md_t n_full_pieces            = NC / NR;
    md_t n_full_pieces_loop_limit = n_full_pieces * NR;
    md_t n_partial_pieces         = NC % NR;

    // Unpack full NR-width chunks
    for (iter_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR) {
        unpackb_nr_f16f16f16of16_ref(b + (cs_b * jc),
                                     unpack_b_buffer + (jc * KC), NR, rs_b,
                                     cs_b, KC, NR);
    }

    // Handle partial piece (must match packer's subdivision logic)
    if (n_partial_pieces > 0) {
        if (NR >= 32) {
            // Production path: unpack subdivided chunks (96, 64, or 32)
            md_t n0_partial_rem    = n_partial_pieces % 32;
            md_t n0_partial_unpack = 0;

            md_t n0_96 = n_partial_pieces / 96;
            md_t n0_64 = n_partial_pieces / 64;
            md_t n0_32 = n_partial_pieces / 32;

            if (n0_96 == 1) {
                unpackb_nr_f16f16f16of16_ref(
                    b + (cs_b * n_full_pieces_loop_limit),
                    unpack_b_buffer + (n_full_pieces_loop_limit * KC), 96, rs_b,
                    cs_b, KC, 96);
                n0_partial_unpack = 96;
            } else if (n0_64 == 1) {
                unpackb_nr_f16f16f16of16_ref(
                    b + (cs_b * n_full_pieces_loop_limit),
                    unpack_b_buffer + (n_full_pieces_loop_limit * KC), 64, rs_b,
                    cs_b, KC, 64);
                n0_partial_unpack = 64;
            } else if (n0_32 == 1) {
                unpackb_nr_f16f16f16of16_ref(
                    b + (cs_b * n_full_pieces_loop_limit),
                    unpack_b_buffer + (n_full_pieces_loop_limit * KC), 32, rs_b,
                    cs_b, KC, 32);
                n0_partial_unpack = 32;
            }

            if (n0_partial_rem > 0) {
                unpackb_nr_f16f16f16of16_ref(
                    b + (cs_b * (n_full_pieces_loop_limit + n0_partial_unpack)),
                    unpack_b_buffer + (n_full_pieces_loop_limit * KC)
                        + (n0_partial_unpack * KC),
                    32, rs_b, cs_b, KC, n0_partial_rem);
            }
        } else {
            // Test path: simple approach without subdivision
            unpackb_nr_f16f16f16of16_ref(b + (cs_b * n_full_pieces_loop_limit),
                                         unpack_b_buffer
                                             + (n_full_pieces_loop_limit * KC),
                                         NR, rs_b, cs_b, KC, n_partial_pieces);
        }
    }
}
