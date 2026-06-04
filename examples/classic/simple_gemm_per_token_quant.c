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
 * Example: Per-token (per-row) Downscale post-op for the s8s8 GEMM API
 *
 * Demonstrates the canonical W8A8 path on aocl_gemm_s8s8s32of32:
 *
 *   - The caller has already pre-quantized the activation matrix A from
 *     F32/BF16 to S8 with a per-row (per-token) scale factor. This is what
 *     an inference serving stack does outside the library.
 *   - The library runs the integer GEMM on (S8 × S8) accumulators.
 *   - The per-token A-dequantization is delivered to the kernel as a SCALE
 *     post-op with scale_factor_len == M and
 *     scale_factor_dim == DLP_PARAM_DIM_PER_TOKEN.
 *
 * This SCALE post-op is the pure-integer counterpart of ADQuantize's
 * a_post_quant in the F32×S8 / BF16×S8 APIs: same operation on the
 * accumulator (per-row dequant scale), different API surface because for
 * s8s8 / u8s8 the forward quant of A is already done by the caller.
 *
 * Two configurations are shown:
 *
 *   Example A: general M × N (here M = N = 128) with per-token Downscale.
 *   Example B: GEMV-N1 (N = 1) with per-token Downscale — the decoder
 *              hot-path for transformer inference. Before the PerM
 *              Downscale fix this configuration broadcast a single SF
 *              across 16 packed M-rows; with the fix it correctly loads
 *              16 contiguous PerM scales per accumulator register.
 *
 * The reference path is computed in plain C and the maximum absolute error
 * is reported alongside the top-left of each result matrix.
 */

#include "aocl_dlp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ───────────────────────────────────────────────────────────────────────────
 */
/* Helpers (init / print) */
/* ───────────────────────────────────────────────────────────────────────────
 */

static void
init_f32_matrix(float* matrix, int rows, int cols, int ld, float scale)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * ld + j] = scale * ((i + j + 1) % 100) / 100.0f;
        }
    }
}

static void
init_s8_matrix(int8_t* matrix, int rows, int cols, int ld, int base_value)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * ld + j] = (int8_t)((base_value * (i + j) % 255) - 127);
        }
    }
}

static void
print_f32_matrix_section(const char* name,
                         float*      matrix,
                         int         rows,
                         int         cols,
                         int         ld,
                         int         max_rows,
                         int         max_cols)
{
    printf("%s (%d x %d) - showing top-left %d x %d:\n", name, rows, cols,
           max_rows, max_cols);
    for (int i = 0; i < max_rows && i < rows; i++) {
        for (int j = 0; j < max_cols && j < cols; j++)
            printf("%10.4f ", matrix[i * ld + j]);
        printf("\n");
    }
    printf("\n");
}

static void
print_s8_matrix_section(const char* name,
                        int8_t*     matrix,
                        int         rows,
                        int         cols,
                        int         ld,
                        int         max_rows,
                        int         max_cols)
{
    printf("%s (%d x %d) - showing top-left %d x %d:\n", name, rows, cols,
           max_rows, max_cols);
    for (int i = 0; i < max_rows && i < rows; i++) {
        for (int j = 0; j < max_cols && j < cols; j++)
            printf("%4d ", matrix[i * ld + j]);
        printf("\n");
    }
    printf("\n");
}

/* ───────────────────────────────────────────────────────────────────────────
 */
/* Reference helpers */
/*                                                                             */
/* The kernel rounds with _MM_FROUND_TO_NEAREST_INT (round-half-to-even), */
/* so the reference forward-quant uses nearbyintf, not lroundf, to avoid */
/* tie-break disagreements at exact half-integer values. */
/* ───────────────────────────────────────────────────────────────────────────
 */

static int8_t
clip_q(long q)
{
    if (q > 127)
        return 127;
    if (q < -128)
        return -128;
    return (int8_t)q;
}

/*
 * Pre-quantize an F32 matrix per-row to S8 with a symmetric scheme.
 * This emulates what an inference serving stack does outside the library
 * before invoking aocl_gemm_s8s8s32of32: each row gets its own scale
 * = 127 / max(|row|), and the inverse (= max/127) is recorded as the
 * dequantization scale that will be passed back via the SCALE post-op.
 */
static void
quantize_per_row_to_s8(const float* A_src,
                       int8_t*      A_dst,
                       md_t         rows,
                       md_t         cols,
                       md_t         lda_src,
                       md_t         lda_dst,
                       float*       deq_sf)
{
    for (md_t i = 0; i < rows; i++) {
        float row_max = 0.0f;
        for (md_t j = 0; j < cols; j++) {
            float v = fabsf(A_src[i * lda_src + j]);
            if (v > row_max)
                row_max = v;
        }
        if (row_max < 1e-6f)
            row_max = 1e-6f;
        float fwd_sf = 127.0f / row_max;
        deq_sf[i]    = row_max / 127.0f;
        for (md_t j = 0; j < cols; j++) {
            long q = (long)nearbyintf(A_src[i * lda_src + j] * fwd_sf);
            A_dst[i * lda_dst + j] = clip_q(q);
        }
    }
}

/*
 * Reference for s8 × s8 → f32 with a per-token (PerM) scale on the
 * accumulator. Mirrors what aocl_gemm_s8s8s32of32 does after applying a
 * SCALE post-op with scale_factor_dim == DLP_PARAM_DIM_PER_TOKEN.
 */
static void
ref_s8s8(const int8_t* A,
         const int8_t* B,
         float*        C_ref,
         md_t          m,
         md_t          n,
         md_t          k,
         md_t          lda,
         md_t          ldb,
         md_t          ldc,
         const float*  perm_sf)
{
    for (md_t i = 0; i < m; i++) {
        for (md_t j = 0; j < n; j++) {
            int32_t acc = 0;
            for (md_t kk = 0; kk < k; kk++) {
                acc += (int32_t)A[i * lda + kk] * (int32_t)B[kk * ldb + j];
            }
            C_ref[i * ldc + j] = (float)acc * perm_sf[i];
        }
    }
}

static float
max_abs_err_f32(const float* a, const float* b, md_t n)
{
    float e = 0.0f;
    for (md_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > e)
            e = d;
    }
    return e;
}

/* ───────────────────────────────────────────────────────────────────────────
 */
/* Main */
/* ───────────────────────────────────────────────────────────────────────────
 */

int
main(void)
{
    /* Matrix dimensions for Example A (general M x N).
     * Small M (= 6) is chosen so the entire result matrix fits on screen. */
    md_t m = 6, n = 16, k = 128;
    md_t lda = k, ldb = n, ldc = n;

    float*  A_f32        = (float*)malloc(lda * m * sizeof(float));
    int8_t* A_s8         = (int8_t*)malloc(lda * m * sizeof(int8_t));
    int8_t* B            = (int8_t*)malloc(ldb * k * sizeof(int8_t));
    float*  C_f32        = (float*)malloc(ldc * m * sizeof(float));
    float*  C_f32_ref    = (float*)malloc(ldc * m * sizeof(float));
    float*  a_dequant_sf = (float*)malloc(m * sizeof(float));

    /* Example B (GEMV-N1) buffers, lazily allocated below. */
    int8_t* B_gv     = NULL;
    float*  C_f32_gv = NULL;
    float*  C_ref_gv = NULL;

    if (!A_f32 || !A_s8 || !B || !C_f32 || !C_f32_ref || !a_dequant_sf) {
        printf("Memory allocation failed\n");
        goto cleanup;
    }

    /* Build the original F32 activations and quantize them per-row to S8.
     * In a real inference pipeline the caller already holds A in S8 form;
     * we generate it here so the example is self-contained. */
    init_f32_matrix(A_f32, m, k, lda, 2.0f);
    init_s8_matrix(B, k, n, ldb, 5);
    quantize_per_row_to_s8(A_f32, A_s8, m, k, lda, lda, a_dequant_sf);

    /* Keep on-screen previews to a small top-left window so the example
     * output stays readable even when the actual matrices are large. */
    const int kPreviewRows = 6;
    const int kPreviewCols = 16;
    print_s8_matrix_section("Matrix A (S8, pre-quantized by caller)", A_s8, m,
                            k, lda, kPreviewRows, kPreviewCols);
    print_s8_matrix_section("Matrix B (S8)", B, k, n, ldb, kPreviewRows,
                            kPreviewCols);
    printf("Per-token A dequant scales (length M = %ld):\n  ", (long)m);
    for (md_t i = 0; i < m; i++)
        printf("%10.6f ", a_dequant_sf[i]);
    printf("\n\n");

    /* The same SCALE post-op metadata is reused across both examples — only
     * the n / leading-dimensions of the GEMM call change between them. */
    dlp_sf_t         a_dq_sf    = { .scale_factor      = a_dequant_sf,
                                    .scale_factor_len  = m,
                                    .scale_factor_type = DLP_F32,
                                    .scale_factor_dim  = DLP_PARAM_DIM_PER_TOKEN };
    dlp_scale_t      a_dq_scale = { .sf = &a_dq_sf, .zp = NULL };
    DLP_POST_OP_TYPE seq[1]     = { SCALE };

    /* ── Example A: general M x N + per-token Downscale ──────────────────── */
    printf("--- Example A: S8xS8 + per-token Downscale (A dequant via SCALE) "
           "---\n\n");
    memset(C_f32, 0, ldc * m * sizeof(float));

    dlp_metadata_t md_A = { 0 };
    md_A.scale          = &a_dq_scale;
    md_A.seq_length     = 1;
    md_A.seq_vector     = seq;

    aocl_gemm_s8s8s32of32('R', 'N', 'N', m, n, k, 1, A_s8, lda, 'N', B, ldb,
                          'N', 0, C_f32, ldc, &md_A);

    ref_s8s8(A_s8, B, C_f32_ref, m, n, k, lda, ldb, ldc, a_dequant_sf);
    print_f32_matrix_section("Result Matrix C", C_f32, m, n, ldc, kPreviewRows,
                             kPreviewCols);
    printf("Max abs error vs reference: %.6f\n",
           max_abs_err_f32(C_f32, C_f32_ref, m * ldc));

    /* ── Example B: GEMV-N1 (n = 1) + per-token Downscale ────────────────── */
    printf("\n--- Example B: S8xS8 GEMV-N1 (n=1, decoder hot path) + per-token "
           "Downscale ---\n\n");

    {
        md_t n_gv   = 1;
        md_t ldb_gv = n_gv;
        md_t ldc_gv = n_gv;

        B_gv     = (int8_t*)malloc(ldb_gv * k * sizeof(int8_t));
        C_f32_gv = (float*)malloc(ldc_gv * m * sizeof(float));
        C_ref_gv = (float*)malloc(ldc_gv * m * sizeof(float));

        if (!B_gv || !C_f32_gv || !C_ref_gv) {
            printf("Memory allocation failed for GEMV-N1 buffers\n");
            goto cleanup;
        }

        init_s8_matrix(B_gv, k, n_gv, ldb_gv, 5);
        memset(C_f32_gv, 0, ldc_gv * m * sizeof(float));

        dlp_metadata_t md_B = { 0 };
        md_B.scale          = &a_dq_scale;
        md_B.seq_length     = 1;
        md_B.seq_vector     = seq;

        aocl_gemm_s8s8s32of32('R', 'N', 'N', m, n_gv, k, 1, A_s8, lda, 'N',
                              B_gv, ldb_gv, 'N', 0, C_f32_gv, ldc_gv, &md_B);

        ref_s8s8(A_s8, B_gv, C_ref_gv, m, n_gv, k, lda, ldb_gv, ldc_gv,
                 a_dequant_sf);
        print_f32_matrix_section("Result Matrix C (n=1 decoder)", C_f32_gv, m,
                                 n_gv, ldc_gv, kPreviewRows, n_gv);
        printf("Max abs error vs reference: %.6f\n",
               max_abs_err_f32(C_f32_gv, C_ref_gv, m * ldc_gv));
    }

cleanup:
    free(A_f32);
    free(A_s8);
    free(B);
    free(C_f32);
    free(C_f32_ref);
    free(a_dequant_sf);
    free(B_gv);
    free(C_f32_gv);
    free(C_ref_gv);
    return 0;
}
