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

#include "classic/aocl_fp16_type.h"
#include "kernels/dlp_kernels.h"

/*
 * FP16 Reference Pack A Implementation (MR=32 configuration)
 *
 * Key characteristics:
 * - M-MAJOR layout: pack_a[m*KC + k] (rs_p=KC, cs_p=1)
 * - MR=32 packing block (32 M-rows per block)
 * - No chunk subdivision needed (simpler than PackB)
 * - Zero-padding for M-fringe to MR=32 boundary
 * - Row-major: Simple copy (no transpose)
 * - Column-major: Requires transpose (read column-wise, write row-wise)
 */

/*
 * Row-major packing for MR=32 panel
 *
 * Input:  A[M][K] row-major (rs_a = K stride, cs_a = 1)
 * Output: pack_a[m*KC + k] M-MAJOR layout
 *
 * For row-major input, we simply copy rows to packed format.
 * No transpose needed since both are row-major.
 */
static void
dlp_packa_mr32_f16f16f16of16_row_major_ref(float16*       pack_a,
                                           const float16* a,
                                           const md_t     MR,
                                           const md_t     lda,
                                           const md_t     KC,
                                           const md_t     m_actual)
{
    // Copy M-rows, each with KC elements
    for (iter_t m = 0; m < m_actual; m++) {
        const float16* inp  = a + m * lda;
        float16*       outp = pack_a + m * KC;
        for (iter_t k = 0; k < KC; k++) {
            *outp++ = *inp++;
        }
    }

    // Zero-pad remaining rows to MR
    for (iter_t m = m_actual; m < MR; m++) {
        float16* outp = pack_a + m * KC;
        for (iter_t k = 0; k < KC; k++) {
            *outp++ = 0;
        }
    }
}

/*
 * Column-major packing for MR=32 panel
 *
 * Input:  A[M][K] column-major (rs_a = 1, cs_a = M stride)
 * Output: pack_a[m*KC + k] M-MAJOR layout
 *
 * For column-major input, we need to transpose.
 * Read column-wise (k-dimension), write row-wise (m-dimension).
 */
static void
dlp_packa_mr32_f16f16f16of16_col_major_ref(float16*       pack_a,
                                           const float16* a,
                                           const md_t     MR,
                                           const md_t     lda,
                                           const md_t     KC,
                                           const md_t     m_actual)
{
    // Transpose: read column-wise, write row-wise
    for (iter_t m = 0; m < m_actual; m++) {
        float16* outp = pack_a + m * KC;
        for (iter_t k = 0; k < KC; k++) {
            *outp++ = a[k * lda + m]; // Transpose: read column-wise
        }
    }

    // Zero-pad remaining rows to MR
    for (iter_t m = m_actual; m < MR; m++) {
        float16* outp = pack_a + m * KC;
        for (iter_t k = 0; k < KC; k++) {
            *outp++ = 0;
        }
    }
}

/*
 * Main reference PackA function
 *
 * Args:
 *   pack_a: Output buffer (must be at least ((MC + 31) / 32) * 32 * KC
 * elements) a: Input matrix pointer rs_a: Row stride of A (in elements) cs_a:
 * Column stride of A (in elements) MC: Number of M-rows to pack KC: Number of
 * K-columns to pack rs_p: Output row stride (set to KC) cs_p: Output column
 * stride (set to 1)
 */
void
dlp_packa_f16f16f16of16_reference(float16*       pack_a,
                                  const float16* a,
                                  const md_t     rs_a,
                                  const md_t     cs_a,
                                  const md_t     MC,
                                  const md_t     KC,
                                  md_t*          rs_p,
                                  md_t*          cs_p)
{
    md_t MR     = 32; // Packing block size
    md_t m_full = MC / MR;
    md_t m_rem  = MC % MR;

    if (cs_a == 1) {
        // Row-major input
        // Pack full MR=32 blocks
        for (iter_t ic = 0; ic < m_full * MR; ic += MR) {
            dlp_packa_mr32_f16f16f16of16_row_major_ref(
                pack_a + ic * KC, a + ic * rs_a, MR, rs_a, KC, MR);
        }

        // Pack M-fringe (if any)
        if (m_rem > 0) {
            dlp_packa_mr32_f16f16f16of16_row_major_ref(
                pack_a + m_full * MR * KC, a + m_full * MR * rs_a, MR, rs_a, KC,
                m_rem);
        }
    } else {
        // Column-major input
        // Pack full MR=32 blocks
        for (iter_t ic = 0; ic < m_full * MR; ic += MR) {
            dlp_packa_mr32_f16f16f16of16_col_major_ref(pack_a + ic * KC, a + ic,
                                                       MR, cs_a, KC, MR);
        }

        // Pack M-fringe (if any)
        if (m_rem > 0) {
            dlp_packa_mr32_f16f16f16of16_col_major_ref(
                pack_a + m_full * MR * KC, a + m_full * MR, MR, cs_a, KC,
                m_rem);
        }
    }

    // Output strides for M-MAJOR layout
    *rs_p = KC; // Row stride = KC (each M-row is contiguous in K)
    *cs_p = 1;  // Column stride = 1
}

/*
 * Unpack A from M-MAJOR layout back to original format
 *
 * Input:  pack_a[m*KC + k] M-MAJOR layout
 * Output: a[m][k] in original layout (row-major or column-major)
 */
static void
dlp_unpacka_mr32_f16f16f16of16_row_major_ref(float16*       a,
                                             const float16* pack_a,
                                             const md_t     MR,
                                             const md_t     lda,
                                             const md_t     KC,
                                             const md_t     m_actual)
{
    // Copy M-rows from packed format back to row-major
    for (iter_t m = 0; m < m_actual; m++) {
        const float16* inp  = pack_a + m * KC;
        float16*       outp = a + m * lda;
        for (iter_t k = 0; k < KC; k++) {
            *outp++ = *inp++;
        }
    }
}

static void
dlp_unpacka_mr32_f16f16f16of16_col_major_ref(float16*       a,
                                             const float16* pack_a,
                                             const md_t     MR,
                                             const md_t     lda,
                                             const md_t     KC,
                                             const md_t     m_actual)
{
    // Transpose: read row-wise from packed, write column-wise to output
    for (iter_t m = 0; m < m_actual; m++) {
        const float16* inp = pack_a + m * KC;
        for (iter_t k = 0; k < KC; k++) {
            a[k * lda + m] = *inp++; // Transpose: write column-wise
        }
    }
}

/*
 * Main reference UnpackA function
 *
 * Args:
 *   a: Output matrix pointer
 *   pack_a: Packed buffer (M-MAJOR layout)
 *   rs_a: Row stride of output A (in elements)
 *   cs_a: Column stride of output A (in elements)
 *   MC: Number of M-rows to unpack
 *   KC: Number of K-columns to unpack
 */
void
dlp_unpacka_f16f16f16of16_reference(float16*       a,
                                    const float16* pack_a,
                                    const md_t     rs_a,
                                    const md_t     cs_a,
                                    const md_t     MC,
                                    const md_t     KC)
{
    md_t MR     = 32; // Packing block size
    md_t m_full = MC / MR;
    md_t m_rem  = MC % MR;

    if (cs_a == 1) {
        // Row-major output
        // Unpack full MR=32 blocks
        for (iter_t ic = 0; ic < m_full * MR; ic += MR) {
            dlp_unpacka_mr32_f16f16f16of16_row_major_ref(
                a + ic * rs_a, pack_a + ic * KC, MR, rs_a, KC, MR);
        }

        // Unpack M-fringe (if any)
        if (m_rem > 0) {
            dlp_unpacka_mr32_f16f16f16of16_row_major_ref(
                a + m_full * MR * rs_a, pack_a + m_full * MR * KC, MR, rs_a, KC,
                m_rem);
        }
    } else {
        // Column-major output
        // Unpack full MR=32 blocks
        for (iter_t ic = 0; ic < m_full * MR; ic += MR) {
            dlp_unpacka_mr32_f16f16f16of16_col_major_ref(
                a + ic, pack_a + ic * KC, MR, cs_a, KC, MR);
        }

        // Unpack M-fringe (if any)
        if (m_rem > 0) {
            dlp_unpacka_mr32_f16f16f16of16_col_major_ref(
                a + m_full * MR, pack_a + m_full * MR * KC, MR, cs_a, KC,
                m_rem);
        }
    }
}
