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

#include "kernels/fp16fp16fp16/dlp_gemm_pack_fp16.h"
#include "fp16fp16fp16/dlp_gemm_reorder_fp16.h"
#include "gemm_utils/dlp_gemm_utils.h"

// ============================================================================
// Packing (dispatcher) functions for NR=128 configuration
// Chunk hierarchy: 128 -> 96 -> 64 -> 32 -> lt32
// ============================================================================

/* Reference implementation - used as fallback when AVX-512 kernel not available
 */
void
dlp_packb_nr128_f16f16f16of16_ref(float16*       pack_b_buffer,
                                  const float16* b,
                                  const md_t     rs_b,
                                  const md_t     cs_b,
                                  const md_t     NC,
                                  const md_t     KC,
                                  md_t*          rs_p,
                                  md_t*          cs_p)
{
    // Select panel chunk size based on NC
    // packb_min_NR=32 (one ZMM register = 32 FP16 elements)
    md_t packb_min_NR = 32;
    md_t chunk_size;

    // Round NC to nearest multiple of packb_min_NR
    md_t nc_rounded = (NC <= 32) ? NC : make_multiple_of_n(NC, packb_min_NR);

    // Select largest applicable chunk size
    if (nc_rounded >= 128) {
        chunk_size = 128;
    } else if (nc_rounded >= 96) {
        chunk_size = 96;
    } else if (nc_rounded >= 64) {
        chunk_size = 64;
    } else {
        chunk_size = 32;
    }

    // Pass selected chunk size to reference packer
    packb_f16f16f16of16_reference(pack_b_buffer, b, rs_b, cs_b, NC, KC,
                                  chunk_size, rs_p, cs_p);
}

void
dlp_packb_nr96_f16f16f16of16(float16*       pack_b_buffer,
                             const float16* b,
                             const md_t     rs_b,
                             const md_t     cs_b,
                             const md_t     NC,
                             const md_t     KC,
                             md_t*          rs_p,
                             md_t*          cs_p)
{
    packb_f16f16f16of16_reference(pack_b_buffer, b, rs_b, cs_b, NC, KC, 96,
                                  rs_p, cs_p);
}

void
dlp_packb_nr64_f16f16f16of16(float16*       pack_b_buffer,
                             const float16* b,
                             const md_t     rs_b,
                             const md_t     cs_b,
                             const md_t     NC,
                             const md_t     KC,
                             md_t*          rs_p,
                             md_t*          cs_p)
{
    packb_f16f16f16of16_reference(pack_b_buffer, b, rs_b, cs_b, NC, KC, 64,
                                  rs_p, cs_p);
}

void
dlp_packb_nr32_f16f16f16of16(float16*       pack_b_buffer,
                             const float16* b,
                             const md_t     rs_b,
                             const md_t     cs_b,
                             const md_t     NC,
                             const md_t     KC,
                             md_t*          rs_p,
                             md_t*          cs_p)
{
    packb_f16f16f16of16_reference(pack_b_buffer, b, rs_b, cs_b, NC, KC, 32,
                                  rs_p, cs_p);
}

void
dlp_packb_nrlt32_f16f16f16of16(float16*       pack_b_buffer,
                               const float16* b,
                               const md_t     rs_b,
                               const md_t     cs_b,
                               const md_t     NC,
                               const md_t     KC,
                               md_t*          rs_p,
                               md_t*          cs_p)
{
    // For NC < 32, pad to NR=32 with zeros
    packb_f16f16f16of16_reference(pack_b_buffer, b, rs_b, cs_b, NC, KC, 32,
                                  rs_p, cs_p);
}

// ============================================================================
// Unpacking (dispatcher) functions
// ============================================================================

void
dlp_unpackb_nr128_f16f16f16of16(float16*       b,
                                const float16* unpack_b_buffer,
                                const md_t     rs_b,
                                const md_t     cs_b,
                                const md_t     NC,
                                const md_t     KC,
                                md_t*          rs_p,
                                md_t*          cs_p)
{
    // Select panel chunk size based on NC (matches packb logic for symmetry)
    // packb_min_NR=32 (one ZMM register = 32 FP16 elements)
    md_t packb_min_NR = 32;
    md_t chunk_size;

    // Round NC to nearest multiple of packb_min_NR
    md_t nc_rounded = (NC <= 32) ? NC : make_multiple_of_n(NC, packb_min_NR);

    // Select largest applicable chunk size (must match packb for round-trip
    // correctness)
    if (nc_rounded >= 128) {
        chunk_size = 128;
    } else if (nc_rounded >= 96) {
        chunk_size = 96;
    } else if (nc_rounded >= 64) {
        chunk_size = 64;
    } else {
        chunk_size = 32;
    }

    // Pass selected chunk size to reference unpacker
    // Note: Reference function expects (b, unpack_b_buffer, NC, KC, NR, rs_b,
    // cs_b)
    unpackb_f16f16f16of16_reference(b, (float16*)unpack_b_buffer, NC, KC,
                                    chunk_size, rs_b, cs_b);

    // Set output strides (row-major output)
    if (rs_p)
        *rs_p = NC; // Row stride (next row)
    if (cs_p)
        *cs_p = 1; // Column stride (next column within row)
}
