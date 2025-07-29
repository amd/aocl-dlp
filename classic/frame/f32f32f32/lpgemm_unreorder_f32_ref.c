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

void
unpackb_f32f32f32of32_row_major_ref(float*     b,
                                    float*     unpack_b,
                                    const md_t NC,
                                    const md_t KC,
                                    const md_t NR,
                                    md_t       ldb)
{
    md_t n_full_pieces            = NC / NR;
    md_t n_full_pieces_loop_limit = n_full_pieces * NR;
    md_t n_partial_pieces         = NC % NR;

    for (md_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR) {
        for (md_t kr = 0; kr < KC; kr++) {
            float* outp = (unpack_b + (ldb * kr) + jc);
            float* inp  = (b + (jc * KC) + (kr * NR));

            for (md_t i = 0; i < NR; i++) {
                *outp++ = *inp++;
            }
        }
    }
    if (n_partial_pieces > 0) {
        md_t   nr0          = n_partial_pieces;
        float* b_rem        = (b + (n_full_pieces_loop_limit * KC));
        float* unpack_b_rem = (unpack_b + n_full_pieces_loop_limit);

        for (md_t kr = 0; kr < KC; kr++) {
            float* inp  = (b_rem + kr * NR);
            float* outp = (unpack_b_rem + (ldb * kr));

            for (md_t i = 0; i < nr0; i++) {
                *outp++ = *inp++;
            }
        }
    }
}

void
unpackb_f32f32f32of32_col_major_ref(float*     b,
                                    float*     unpack_b,
                                    const md_t NC,
                                    const md_t KC,
                                    const md_t NR,
                                    md_t       ldb)
{
    md_t n_full_pieces            = NC / NR;
    md_t n_full_pieces_loop_limit = n_full_pieces * NR;
    md_t n_partial_pieces         = NC % NR;

    for (md_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR) {
        for (md_t kr = 0; kr < KC; kr++) {
            float* outp = (unpack_b + jc * KC + kr);
            float* inp  = (b + (jc * ldb) + (kr * NR));

            for (md_t i = 0; i < NR; i++) {
                *(outp + i * ldb) = *inp++;
            }
        }
    }
    if (n_partial_pieces > 0) {
        float* b_rem        = (b + (n_full_pieces_loop_limit * KC));
        float* unpack_b_rem = (unpack_b + n_full_pieces_loop_limit * ldb);

        for (md_t kr = 0; kr < KC; kr++) {
            float* inp  = (b_rem + kr * NR);
            float* outp = (unpack_b_rem + kr);

            for (md_t i = 0; i < n_partial_pieces; i++) {
                *(outp + i * ldb) = *inp++;
            }
        }
    }
}

void
unpackb_f32f32f32of32_reference(float*     b,
                                float*     unpack_b,
                                const md_t NC,
                                const md_t KC,
                                const md_t NR,
                                md_t       rs_b,
                                md_t       cs_b)
{
    if (cs_b == 1) {
        unpackb_f32f32f32of32_row_major_ref(b, unpack_b, NC, KC, NR, rs_b);
    } else {
        unpackb_f32f32f32of32_col_major_ref(b, unpack_b, NC, KC, NR, cs_b);
    }
}
