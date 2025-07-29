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

#include <immintrin.h>
#include <string.h>

#include "f32f32f32/lpgemm_reorder_f32.h"
/*
    Below are the reference packb functions which are
    varied based on block size NR (64, 48, 32, 16, lt) and
    order (row / column (transpose)).
*/

static void
packb_f32f32f32of32_row_major_ref(float*       pack_b,
                                  const float* b,
                                  const md_t   ldb,
                                  const md_t   NC,
                                  const md_t   KC,
                                  const md_t   NR,
                                  md_t*        rs_b,
                                  md_t*        cs_b)
{
    md_t n_full_pieces            = NC / NR;
    md_t n_full_pieces_loop_limit = n_full_pieces * NR;
    md_t n_partial_pieces         = NC % NR;
    for (md_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR) {
        for (md_t kr = 0; kr < KC; kr++) {
            const float* inp0  = (b + (ldb * kr) + jc);
            float*       outp0 = (pack_b + (jc * KC) + (kr * NR));
            for (md_t i = 0; i < NR; i++)
                *outp0++ = *inp0++;
        }
    }

    if (n_partial_pieces > 0) {
        float*       pack_b_rem = (pack_b + (n_full_pieces_loop_limit * KC));
        const float* b_rem      = (b + n_full_pieces_loop_limit);
        for (md_t kr = 0; kr < KC; kr++) {
            const float* inp0  = (b_rem + (ldb * kr));
            float*       outp0 = (pack_b_rem + (kr * NR));
            for (md_t i = 0; i < n_partial_pieces; i++)
                *outp0++ = *inp0++;
        }
    }

    *rs_b = NR;
    *cs_b = 1;
}

static void
packb_nr_f32f32f32of32_col_major_ref(float*       pack_b_buffer,
                                     const float* b,
                                     const md_t   NR,
                                     const md_t   ldb,
                                     const md_t   KC,
                                     const md_t   n0_partial_rem)
{
    for (md_t i = 0; i < n0_partial_rem; i++) {
        const float* inp  = (b + (ldb * i));
        float*       outp = pack_b_buffer + i;
        for (md_t j = 0; j < KC; j++) {
            *(outp + (j * NR)) = *inp++;
        }
    }
    for (md_t i = n0_partial_rem; i < NR; i++) {
        float* outp = pack_b_buffer + i;
        for (md_t j = 0; j < KC; j++) {
            *(outp + (j * NR)) = 0;
        }
    }
}

static void
packb_f32f32f32of32_col_major_ref(float*       pack_b_buffer,
                                  const float* b,
                                  const md_t   ldb,
                                  const md_t   NC,
                                  const md_t   KC,
                                  const md_t   NR,
                                  md_t*        rs_b,
                                  md_t*        cs_b)
{
    md_t n_full_pieces            = NC / NR;
    md_t n_full_pieces_loop_limit = n_full_pieces * NR;
    md_t n_partial_pieces         = NC % NR;

    for (md_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR) {
        packb_nr_f32f32f32of32_col_major_ref(pack_b_buffer + (jc * KC),
                                             b + (jc * ldb), NR, ldb, KC, NR);
    }

    if (n_partial_pieces > 0) {
        packb_nr_f32f32f32of32_col_major_ref(
            (pack_b_buffer + (n_full_pieces_loop_limit * KC)),
            (b + n_full_pieces_loop_limit * ldb), NR, ldb, KC,
            n_partial_pieces);
    }

    *rs_b = NR;
    *cs_b = 1;
}

void
packb_f32f32f32of32_reference(float*       pack_b,
                              const float* b,
                              const md_t   rs_b,
                              const md_t   cs_b,
                              const md_t   NC,
                              const md_t   KC,
                              const md_t   NR,
                              md_t*        rs_p,
                              md_t*        cs_p)
{
    if (cs_b == 1) {
        packb_f32f32f32of32_row_major_ref(pack_b, b, rs_b, NC, KC, NR, rs_p,
                                          cs_p);
    } else {
        packb_f32f32f32of32_col_major_ref(pack_b, b, cs_b, NC, KC, NR, rs_p,
                                          cs_p);
    }
}
