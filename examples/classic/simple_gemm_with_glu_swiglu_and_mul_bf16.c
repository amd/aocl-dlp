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
 * Example: BF16 GEMM fused with a Gated Linear Unit (GLU) post-op
 * ---------------------------------------------------------------
 * GLU is the gating used by the MoE / SwiGLU feed-forward block. For an input
 * row x the fused expression is:
 *
 *       f(x) = act(x . W_gate) (-) (x . W_up)
 *
 *   where (-) is element-wise multiply and act() is SiLU (GATED_SWIGLU) or the
 *   clamped gpt-oss/OpenAI variant (GATED_SWIGLU_AND_MUL). Instead of two
 * separate GEMMs, the gate and up projections are packed into ONE weight matrix
 * whose columns are INTERLEAVED  g0 u0 g1 u1 ...  so a single GEMM produces an
 *   M x 2I accumulator. The GLU post-op then consumes the (gate, up) pairs and
 *   emits the M x I gated result via a half-width store:
 *
 *       gate[m][j] = C_acc[m][2j]      up[m][j] = C_acc[m][2j+1]
 *       out [m][j] = act(gate) * up        (j = 0 .. I-1)
 *
 * Key API rules for the GLU post-op (see include/classic/aocl_gemm_metadata.h):
 *   - It is TERMINAL: it must be the last entry of seq_vector[] (it changes the
 *     shape from 2I to I, so nothing can run after it), and at most one GLU per
 *     chain. Earlier post-ops (e.g. BIAS) run on the M x 2I accumulator first,
 *     then GLU folds the (gate, up) pairs down to M x I -- see the second demo.
 *   - The compacted result goes to a SEPARATE caller-owned buffer D (MANDATORY
 *     for GLU), set via glu->d / glu->ld_d. D is logically M x I in C's storage
 *     order: row-major -> ld_d >= I, column-major -> ld_d >= m. out[m][j] lives
 *     at D[m*ld_d + j] (row-major). The kernel writes every element of D.
 *   - The GEMM output C stays an M x 2I workspace (ldc >= 2I) and receives the
 *     raw pre-GLU accumulator; the gated result is read from D, not C.
 *   - Neither variant takes runtime scalars: GATED_SWIGLU is plain
 * SiLU(gate)*up; GATED_SWIGLU_AND_MUL bakes its constants (alpha=1.702,
 * clamp=7.0, +1 bias), so alpha/beta/stor_type are left NULL/DLP_INVALID.
 *
 * This file shows two fused chains:
 *   1. Pure GATED_SWIGLU_AND_MUL            : seq_vector = { GLU }
 *   2. BIAS followed by GATED_SWIGLU_AND_MUL: seq_vector = { BIAS, GLU }  (GLU
 * terminal)
 *
 * NOTE: the fused GLU post-op is currently implemented only for the BF16 GEMM
 * path (aocl_gemm_bf16bf16f32of32), including the BF16 GEMV m == 1 fast path
 * (row-major via the M1 kernel, column-major via the N1 kernel). It is NOT
 * available on the other dtype GEMM APIs.
 */

#include "aocl_dlp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// float -> bfloat16 (round-toward-zero truncation is fine for an example)
static bfloat16
float_to_bfloat16(float v)
{
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return (bfloat16)(bits >> 16);
}

// bfloat16 -> float
static float
bfloat16_to_float(bfloat16 v)
{
    uint32_t bits = ((uint32_t)v) << 16;
    float    f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void
fill_bf16(bfloat16* mat, int rows, int cols, int ld, float lo, float hi)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float t = (float)((i * cols + j) % 17) / 16.0f; // deterministic
            mat[i * ld + j] = float_to_bfloat16(lo + t * (hi - lo));
        }
    }
}

static void
print_glu_output(const float* D, md_t m, md_t I, md_t ld_d)
{
    printf("GLU output  D  (M x I = %d x %d):\n", (int)m, (int)I);
    for (md_t i = 0; i < m; i++) {
        for (md_t j = 0; j < I; j++)
            printf("%8.4f ", D[i * ld_d + j]);
        printf("\n");
    }
    printf("\n");
}

// Reference GATED_SWIGLU_AND_MUL activation for one (gate, up) pair, applied on
// the post-BIAS accumulator values. Baked OAI constants: alpha = 1.702, limit
// = 7.
//   g   = min(gate, limit)                 (gate clamped to max only)
//   u   = clip(up, -limit, limit)
//   out = (u + 1) * g * sigmoid(alpha * g)
static float
ref_gated_swiglu_and_mul(float gate, float up)
{
    const float kAlpha = 1.702f, kLimit = 7.0f;
    const float g = gate < kLimit ? gate : kLimit;
    const float u = up < -kLimit ? -kLimit : (up > kLimit ? kLimit : up);
    return (u + 1.0f) * g / (1.0f + expf(-kAlpha * g));
}

// ===========================================================================
// Pure GATED_SWIGLU_AND_MUL  ->  seq_vector = { GLU }
// ===========================================================================
static int
run_gated_swiglu_and_mul_only(void)
{
    printf("=== GATED_SWIGLU_AND_MUL only (seq = { GLU }) ===\n");

    // I = half output width. The GEMM N is the interleaved width n = 2I.
    const md_t m = 4;     // tokens / rows
    const md_t I = 8;     // gated output width
    const md_t n = 2 * I; // interleaved gate/up width
    const md_t k = 32;    // contraction dim

    const md_t lda  = k; // row-major
    const md_t ldb  = n;
    const md_t ldc  = n; // C workspace: MUST be >= 2I
    const md_t ld_d = I; // D output (row-major): MUST be >= I

    bfloat16* A = (bfloat16*)malloc((size_t)m * lda * sizeof(bfloat16));
    bfloat16* B = (bfloat16*)malloc((size_t)k * ldb * sizeof(bfloat16));
    float*    C = (float*)malloc((size_t)m * ldc * sizeof(float));
    // Compacted GLU output buffer D (m x I), mandatory for the GLU post-op.
    float* D = (float*)malloc((size_t)m * ld_d * sizeof(float));
    if (!A || !B || !C || !D) {
        printf("allocation failed\n");
        free(A);
        free(B);
        free(C);
        free(D);
        return -1;
    }

    fill_bf16(A, m, k, lda, -1.0f, 1.0f);
    fill_bf16(B, k, n, ldb, -1.0f, 1.0f); // columns interleaved g,u,g,u,...

    const float alpha = 1.0f, beta = 0.0f;
    const char  order = 'R', transa = 'N', transb = 'N';
    const char  fmt_a = 'N', fmt_b = 'N';

    // ---- Build the GLU-only post-op metadata ----
    dlp_metadata_t* md = (dlp_metadata_t*)calloc(1, sizeof(dlp_metadata_t));
    md->seq_length     = 0;
    md->seq_vector     = NULL; // no pre-ops, GLU is terminal

    md->glu            = (dlp_term_op_glu*)calloc(1, sizeof(dlp_term_op_glu));
    md->glu->algo_type = GATED_SWIGLU_AND_MUL;
    md->glu->alpha     = NULL; // no runtime scalars (OAI consts baked in)
    md->glu->beta      = NULL;
    md->glu->stor_type = DLP_INVALID;
    md->glu->d         = D;    // mandatory compacted output buffer (m x I)
    md->glu->ld_d      = ld_d; // row-major D leading dim (>= I)

    // ---- Fused GEMM + GLU ----
    // D[:, 0:I] = GLU( alpha * A*B + beta*C ); C holds the raw 2I workspace.
    aocl_gemm_bf16bf16f32of32(order, transa, transb, m, n, k, alpha, A, lda,
                              fmt_a, B, ldb, fmt_b, beta, C, ldc, md);

    print_glu_output(D, m, I, ld_d);

    free(md->glu);
    free(md->seq_vector);
    free(md);

    // ---- Tiny scalar cross-check of row 0, col 0 ----
    // gate = (A*B)[0][0], up = (A*B)[0][1]; no bias in this demo.
    float gate = 0.0f, up = 0.0f;
    for (md_t p = 0; p < k; p++) {
        gate += bfloat16_to_float(A[0 * lda + p])
                * bfloat16_to_float(B[p * ldb + 0]);
        up += bfloat16_to_float(A[0 * lda + p])
              * bfloat16_to_float(B[p * ldb + 1]);
    }
    printf(
        "Reference out[0][0] = GATED_SWIGLU_AND_MUL(gate=%.4f, up=%.4f) = %.4f"
        "   (D[0][0] = %.4f)\n\n",
        gate, up, ref_gated_swiglu_and_mul(gate, up), D[0]);

    free(A);
    free(B);
    free(C);
    free(D);
    return 0;
}

// ===========================================================================
// BIAS then GATED_SWIGLU_AND_MUL  ->  seq_vector = { BIAS, GLU }
// ---------------------------------------------------------------------------
// BIAS runs first on the M x 2I accumulator (one bias value per interleaved
// column, so bias_len = n = 2I), then the terminal GLU folds each (gate, up)
// pair to a single gated output. This shows a post-op running *before* GLU.
// Uses different dimensions from the GLU-only chain on purpose.
// ===========================================================================
static int
run_bias_then_gated_swiglu_and_mul(void)
{
    printf("=== BIAS -> GATED_SWIGLU_AND_MUL (seq = { BIAS, GLU }) ===\n");

    const md_t m = 3;     // tokens / rows
    const md_t I = 6;     // gated output width
    const md_t n = 2 * I; // interleaved gate/up width
    const md_t k = 16;    // contraction dim

    const md_t lda  = k;
    const md_t ldb  = n;
    const md_t ldc  = n; // C workspace: MUST be >= 2I
    const md_t ld_d = I; // D output (row-major): MUST be >= I

    bfloat16* A = (bfloat16*)malloc((size_t)m * lda * sizeof(bfloat16));
    bfloat16* B = (bfloat16*)malloc((size_t)k * ldb * sizeof(bfloat16));
    float*    C = (float*)malloc((size_t)m * ldc * sizeof(float));
    float*    D = (float*)malloc((size_t)m * ld_d * sizeof(float));
    // Bias vector: one value per interleaved column of the 2I accumulator.
    float* bias = (float*)malloc((size_t)n * sizeof(float));
    if (!A || !B || !C || !D || !bias) {
        printf("allocation failed\n");
        free(A);
        free(B);
        free(C);
        free(D);
        free(bias);
        return -1;
    }

    fill_bf16(A, m, k, lda, -1.0f, 1.0f);
    fill_bf16(B, k, n, ldb, -1.0f, 1.0f);
    for (md_t j = 0; j < n; j++)
        bias[j] = 0.25f * (float)(j % 5) - 0.5f; // deterministic f32 bias

    const float alpha = 1.0f, beta = 0.0f;
    const char  order = 'R', transa = 'N', transb = 'N';
    const char  fmt_a = 'N', fmt_b = 'N';

    // ---- Build the BIAS -> GLU post-op chain ----
    dlp_metadata_t* md = (dlp_metadata_t*)calloc(1, sizeof(dlp_metadata_t));
    md->seq_length     = 1;
    md->seq_vector    = (DLP_POST_OP_TYPE*)malloc(1 * sizeof(DLP_POST_OP_TYPE));
    md->seq_vector[0] = BIAS; // runs on the M x 2I accumulator

    // BIAS post-op (f32 bias, no scale/zero-point needed).
    md->bias         = (dlp_post_op_bias*)calloc(1, sizeof(dlp_post_op_bias));
    md->bias[0].bias = bias;
    md->bias[0].stor_type = DLP_F32;
    md->bias[0].bias_len  = n; // one bias per interleaved column (2I)
    md->bias[0].sf        = NULL;
    md->bias[0].zp        = NULL;

    // GLU post-op (terminal).
    md->glu            = (dlp_term_op_glu*)calloc(1, sizeof(dlp_term_op_glu));
    md->glu->algo_type = GATED_SWIGLU_AND_MUL;
    md->glu->alpha     = NULL;
    md->glu->beta      = NULL;
    md->glu->stor_type = DLP_INVALID;
    md->glu->d         = D;
    md->glu->ld_d      = ld_d;

    // ---- Fused GEMM + BIAS + GLU ----
    // D[:, 0:I] = GLU( (alpha*A*B + beta*C) + bias ); C holds the 2I workspace.
    aocl_gemm_bf16bf16f32of32(order, transa, transb, m, n, k, alpha, A, lda,
                              fmt_a, B, ldb, fmt_b, beta, C, ldc, md);

    print_glu_output(D, m, I, ld_d);

    free(md->bias);
    free(md->glu);
    free(md->seq_vector);
    free(md);

    // ---- Tiny scalar cross-check of row 0, col 0 (with bias) ----
    // gate = (A*B)[0][0] + bias[0], up = (A*B)[0][1] + bias[1], then GLU.
    float gate = 0.0f, up = 0.0f;
    for (md_t p = 0; p < k; p++) {
        gate += bfloat16_to_float(A[0 * lda + p])
                * bfloat16_to_float(B[p * ldb + 0]);
        up += bfloat16_to_float(A[0 * lda + p])
              * bfloat16_to_float(B[p * ldb + 1]);
    }
    gate += bias[0];
    up += bias[1];
    printf("Reference out[0][0] = GATED_SWIGLU_AND_MUL(gate+bias=%.4f, "
           "up+bias=%.4f) "
           "= %.4f   (D[0][0] = %.4f)\n\n",
           gate, up, ref_gated_swiglu_and_mul(gate, up), D[0]);

    free(A);
    free(B);
    free(C);
    free(D);
    free(bias);
    return 0;
}

int
main(void)
{
    printf("BF16 GEMM fused with GLU post-op (GATED_SWIGLU_AND_MUL / MoE "
           "gating)\n\n");

    int rc = run_gated_swiglu_and_mul_only();
    if (rc != 0)
        return rc;

    rc = run_bias_then_gated_swiglu_and_mul();
    return rc;
}
