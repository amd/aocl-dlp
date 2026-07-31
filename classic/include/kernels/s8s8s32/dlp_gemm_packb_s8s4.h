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

#ifndef DLP_GEMM_INT8_PACKB_S8S4
#define DLP_GEMM_INT8_PACKB_S8S4

#include "dlp_gemm_types.h"
#include <stdint.h>

/*
 * Signed 4-bit <-> signed 8-bit conversion helpers used by the s8s4 symmetric
 * quantized GEMM path.
 *
 * Nibble packing convention (identical to the u8s4/bf16s4 contract):
 *   - Two 4-bit values are packed per byte.
 *   - The element at even linear index  -> low  nibble (bits [3:0]).
 *   - The element at odd  linear index  -> high nibble (bits [7:4]).
 *   - So element i lives in byte (i >> 1); nibble selector is (i & 1).
 *
 * Because s4 values sign-extend into the range [-8, 7], they round-trip
 * losslessly through s8. The s8s4 reordered weight buffer stores the s8
 * VNNI-4 packed weights compressed 2:1 into nibbles using this exact linear
 * convention; the GEMM path re-expands them transiently back to s8 before the
 * shared s8s8 micro-kernel runs. Compression and expansion are pure linear
 * byte transforms over the packed region, so no knowledge of the internal
 * VNNI-4 layout is required here.
 */

/* Sign-extend a single 4-bit nibble (already isolated in bits [3:0]) to s8. */
static inline int8_t
dlp_s4_nibble_to_s8(uint8_t byte, int hi_nibble)
{
    uint8_t nib = (uint8_t)(hi_nibble ? ((byte >> 4) & 0x0F) : (byte & 0x0F));
    /* Two's-complement 4-bit -> 8-bit sign extension: (nib ^ 8) - 8. */
    return (int8_t)((int)(nib ^ 0x08) - 8);
}

/*
 * Expand n_elems s4 nibbles from a packed source buffer into s8 destination.
 * n_elems may be odd; the final (dangling) low nibble is still handled.
 */
static inline void
dlp_cvt_s4_to_s8_linear(int8_t* dst, const uint8_t* src, md_t n_elems)
{
    for (iter_t i = 0; i < n_elems; i++) {
        dst[i] = dlp_s4_nibble_to_s8(src[i >> 1], (int)(i & 1));
    }
}

/*
 * Compress n_elems s8 values (each guaranteed to be in [-8, 7]) into packed
 * s4 nibbles. n_elems is expected to be even for the s8s4 packed weight
 * region (product of multiples of 4 and 16); an odd tail element, if any, is
 * stored in the low nibble with the high nibble left zero.
 */
static inline void
dlp_cvt_s8_to_s4_linear(uint8_t* dst, const int8_t* src, md_t n_elems)
{
    md_t i = 0;
    for (; (i + 1) < n_elems; i += 2) {
        uint8_t lo  = (uint8_t)(src[i] & 0x0F);
        uint8_t hi  = (uint8_t)(src[i + 1] & 0x0F);
        dst[i >> 1] = (uint8_t)(lo | (hi << 4));
    }
    if (i < n_elems) {
        dst[i >> 1] = (uint8_t)(src[i] & 0x0F);
    }
}

/*
 * Widen a (rows x cols) s4 matrix, addressed via arbitrary element strides
 * (rs, cs) over a nibble-packed source buffer, into a contiguous row-major s8
 * matrix (dst[r * cols + c]). Used at reorder time to materialize a transient
 * s8 copy of the raw s4 B matrix so the existing s8 packer can be reused.
 */
static inline void
dlp_cvt_s4_to_s8_matrix(
    int8_t* dst, const uint8_t* src, md_t rows, md_t cols, md_t rs, md_t cs)
{
    for (iter_t r = 0; r < rows; r++) {
        for (iter_t c = 0; c < cols; c++) {
            md_t idx = (rs * r) + (cs * c);
            dst[(r * cols) + c] =
                dlp_s4_nibble_to_s8(src[idx >> 1], (int)(idx & 1));
        }
    }
}

/*
 * Widen a (rows x cols) interior sub-block of a strided, nibble-packed s4
 * source into a contiguous row-major s8 destination (dst[r * cols + c]). The
 * sub-block starts at source element (row0, col0), i.e. the source element
 * index is rs * (row0 + r) + cs * (col0 + c). This is the runtime-pack analogue
 * of dlp_cvt_s4_to_s8_matrix: it lets each packing thread widen only the block
 * it owns, from either B orientation (rs/cs carry transb). Nibble addressing is
 * done in element space (idx >> 1 / idx & 1), so an odd starting element that
 * lands mid-byte is handled correctly.
 */
static inline void
dlp_cvt_s4_to_s8_block(int8_t*        dst,
                       const uint8_t* src,
                       md_t           rows,
                       md_t           cols,
                       md_t           rs,
                       md_t           cs,
                       md_t           row0,
                       md_t           col0)
{
    for (iter_t r = 0; r < rows; r++) {
        for (iter_t c = 0; c < cols; c++) {
            md_t idx = (rs * (row0 + r)) + (cs * (col0 + c));
            dst[(r * cols) + c] =
                dlp_s4_nibble_to_s8(src[idx >> 1], (int)(idx & 1));
        }
    }
}

/*
 * Fused (widen-on-load) s8s4 pack-B kernels. These read the raw nibble-packed
 * s4 matrix directly (rs_b/cs_b are element strides in NIBBLE units; row0/col0
 * are the element origin of the sub-panel to pack), widen each row to s8
 * in-register and emit the exact s8 VNNI-4 packed layout + per-group int32
 * column sums that the s8 sym-quant micro-kernel consumes. No intermediate s8
 * scratch buffer is required.
 *
 * Defined in kernels/zen4/s8s8s32/dlp_gemm_packb_s8s4_amd512vnni.c (zen4 arch
 * flags); only called from DLP_KERNELS_ZEN4 code paths.
 */

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
                                     md_t*          cs_b_out);

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
                                     md_t*          cs_b_out);

/*
 * AVX-512 vectorized linear s4<->s8 stream converters for the s8s4 sym-quant
 * path. Bit-identical to the scalar contiguous converters above
 * (dlp_cvt_s4_to_s8_linear / dlp_cvt_s8_to_s4_linear), and defined alongside
 * the pack-B kernels in kernels/zen4/s8s8s32/dlp_gemm_packb_s8s4_amd512vnni.c
 * (zen4 arch flags). Only called from DLP_KERNELS_ZEN4 code paths (AVX-512-VNNI
 * hardware); src/dst must reference contiguous, even-aligned nibble/byte
 * streams.
 */
void
dlp_cvt_s4_to_s8_linear_avx512(int8_t* dst, const uint8_t* src, md_t n_elems);

void
dlp_cvt_s8_to_s4_linear_avx512(uint8_t* dst, const int8_t* src, md_t n_elems);

#endif // DLP_GEMM_INT8_PACKB_S8S4
