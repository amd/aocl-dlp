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
 * Quantized matrix multiplication: u8 activations x s8 weights.
 *
 * Part 1 dequantizes straight to F32   (aocl_gemm_u8s8s32of32).
 * Part 2 requantizes to S8             (aocl_gemm_u8s8s32os8).
 *
 * The one thing to take away
 * --------------------------
 * The GEMM accumulates in S32. That accumulator is already an integer count
 * of  a_scale * b_scale  units. The SCALE post-op multiplies it by
 * scale_factor. So scale_factor is a *ratio*, never a reciprocal scale:
 *
 *     F32 out : scale_factor = c_scale                 (= a_scale * b_scale)
 *     S8  out : scale_factor = c_scale / c_out_scale
 *
 * Passing 1/c_scale is the classic mistake. Here that would be 2^23, which
 * overflows the S32 accumulator long before it ever reaches the S8 clamp.
 *
 * Zero points
 * -----------
 * A is U8 with zero_point 0, which is the common post-ReLU activation case
 * and keeps the arithmetic exact. If A carried a non-zero zero point zp_a,
 * the kernel would still accumulate the raw  a_u8 * b_s8  products and the
 * caller would owe a per-column correction:
 *
 *     c = c_scale * ( acc - zp_a * sum_p B[p][j] )
 *
 * The A-side quantization metadata is not accepted by the U8S8 GEMM entry
 * points, so that correction has to be folded into a BIAS post-op or applied
 * by the caller. This example does not need it.
 *
 * Judging a quantized result
 * --------------------------
 * Compare in units of the output grid, not in percent. An S8 output is only
 * ever accurate to half a step, and every element smaller than half a step
 * rounds to zero -- a relative error of exactly 1.0 for a flawless kernel.
 * Mean relative error in particular says almost nothing: a fully saturated
 * output and a fully zeroed one both score close to 1.0.
 *
 * Why the checks below can demand exactness
 * ----------------------------------------
 * Both scales are powers of two and the inputs sit exactly on that grid, so
 * every product and every partial sum is representable in F32: the largest
 * accumulator here is 255 * 127 * K, well under 2^24. The quantized path and
 * the float reference must therefore agree bit for bit. With scales that come
 * out of real calibration this is not true and the F32 check needs a small
 * tolerance -- the S8 check in part 2 already shows that case.
 */

#include "aocl_dlp.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PREVIEW 4 /* rows/cols echoed to the screen */

/* Echo the top-left PREVIEW x PREVIEW corner of a row-major array with the
 * given leading dimension. Assumes the array is at least that large, which
 * every matrix in this example is. */
#define PRINT_TILE(label, mat, ld, fmt)                                        \
    do {                                                                       \
        printf("%s, top-left %dx%d:\n", (label), PREVIEW, PREVIEW);            \
        for (int r_ = 0; r_ < PREVIEW; r_++) {                                 \
            for (int c_ = 0; c_ < PREVIEW; c_++)                               \
                printf(fmt, (mat)[r_ * (ld) + c_]);                            \
            printf("\n");                                                      \
        }                                                                      \
        printf("\n");                                                          \
    } while (0)

/* Fill with values that land exactly on the quantization grid, so the round
 * trip below can be checked for exact equality. Sign and magnitude must not
 * correlate along K: a fixed pattern lets every term of the reduction
 * accumulate coherently, which masks whole classes of defect. Seeded once in
 * main() so a failure reproduces. */
static void
fill_grid(float* mat, md_t rows, md_t cols, md_t ld, int lo, int hi, float step)
{
    int span = hi - lo + 1;

    for (md_t i = 0; i < rows; i++)
        for (md_t j = 0; j < cols; j++)
            mat[i * ld + j] = (float)(lo + rand() % span) * step;
}

static void
quantize_u8(const float* src,
            uint8_t*     dst,
            md_t         rows,
            md_t         cols,
            md_t         ld,
            float        scale,
            int          zero_point)
{
    for (md_t i = 0; i < rows; i++)
        for (md_t j = 0; j < cols; j++) {
            float v = roundf(src[i * ld + j] / scale) + (float)zero_point;
            dst[i * ld + j] =
                (uint8_t)(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v));
        }
}

static void
quantize_s8(
    const float* src, int8_t* dst, md_t rows, md_t cols, md_t ld, float scale)
{
    for (md_t i = 0; i < rows; i++)
        for (md_t j = 0; j < cols; j++) {
            float v = roundf(src[i * ld + j] / scale);
            dst[i * ld + j] =
                (int8_t)(v < -128.0f ? -128.0f : (v > 127.0f ? 127.0f : v));
        }
}

static void
reference_gemm(const float* a,
               md_t         lda,
               const float* b,
               md_t         ldb,
               float*       c,
               md_t         ldc,
               md_t         m,
               md_t         n,
               md_t         k)
{
    for (md_t i = 0; i < m; i++)
        for (md_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (md_t p = 0; p < k; p++)
                sum += a[i * lda + p] * b[p * ldb + j];
            c[i * ldc + j] = sum;
        }
}

static float
max_abs(const float* mat, md_t rows, md_t cols, md_t ld)
{
    float peak = 0.0f;
    for (md_t i = 0; i < rows; i++)
        for (md_t j = 0; j < cols; j++)
            if (fabsf(mat[i * ld + j]) > peak)
                peak = fabsf(mat[i * ld + j]);
    return peak;
}

/* Compare against the float reference.
 *
 * The gate is absolute error measured in units of the output grid, not
 * relative error. A quantized output cannot do better than half a step, and
 * any element whose true value is below half a step lands on zero -- giving a
 * relative error of exactly 1.0 no matter how correct the kernel is. Gating
 * on relative error therefore reports failure for a perfect implementation,
 * and cannot tell a saturated result apart from one that was never computed.
 *
 * Returns 0 on success so main() can accumulate a real exit status -- an
 * example that always returns 0 cannot be used as a test.
 */
static int
check(const char*  label,
      const float* ref,
      const float* got,
      md_t         rows,
      md_t         cols,
      md_t         ld,
      float        step,
      float        tolerance_steps)
{
    float worst_abs = 0.0f;
    float worst_rel = 0.0f;
    md_t  offenders = 0;
    float limit     = tolerance_steps * step;

    for (md_t i = 0; i < rows; i++)
        for (md_t j = 0; j < cols; j++) {
            float r = ref[i * ld + j];
            float d = fabsf(r - got[i * ld + j]);

            if (d > worst_abs)
                worst_abs = d;
            if (fabsf(r) > 1e-6f && d / fabsf(r) > worst_rel)
                worst_rel = d / fabsf(r);
            if (d > limit)
                offenders++;
        }

    printf("%s\n", label);
    if (step > 0.0f)
        printf("  max error       : %.9g  (%.4f output steps, limit %.4f)\n",
               worst_abs, (double)(worst_abs / step), (double)tolerance_steps);
    else
        printf("  max error       : %.9g  (limit %.9g)\n", worst_abs, limit);
    printf("  max rel. error  : %.9g   (informational only)\n", worst_rel);
    printf("  elements outside tolerance : %lld of %lld\n",
           (long long)offenders, (long long)(rows * cols));

    if (offenders) {
        printf("  RESULT: MISMATCH\n\n");
        return 1;
    }
    printf("  RESULT: OK\n\n");
    return 0;
}

/* Report and classify the status the library left in the metadata.
 * Returns 0 to continue, 77 to skip (ctest SKIP_RETURN_CODE), 1 on error. */
static int
gemm_status(const char* what, const dlp_metadata_t* md)
{
    dlp_clsc_err_t code = md->error_hndl.error_code;

    if (code == DLP_CLSC_SUCCESS)
        return 0;

    if (code == DLP_CLSC_NOT_SUPPORTED) {
        /* Either the hardware lacks AVX512-VNNI, which U8S8 GEMM requires,
         * or some argument of this particular call is outside what the API
         * accepts. Both are legitimate reasons to skip rather than fail. */
        printf("%s: this configuration is not supported here. Skipping.\n",
               what);
        return 77;
    }

    printf("%s: FAILED, error code %d\n", what, (int)code);
    return 1;
}

int
main(void)
{
    const md_t m = 64, n = 64, k = 64;
    /* Deliberately larger than the row lengths: a leading dimension that
     * always equals its row length lets a mistake in the stride arithmetic
     * pass unnoticed. The trailing columns are never written or read. */
    const md_t lda = k + 4, ldb = n + 8, ldc = n + 8;

    /* Calibration is an offline step; a real pipeline arrives with these
     * already chosen. Powers of two here so the reference stays exact --
     * see the note at the top of the file. */
    const float a_scale      = 1.0f / 2048.0f; /* U8 activations, post-ReLU */
    const float b_scale      = 1.0f / 4096.0f; /* S8 weights, symmetric     */
    const int   a_zero_point = 0;
    const float c_scale      = a_scale * b_scale; /* units of the S32 acc */

    int failures = 0; /* mismatching comparisons */
    int status   = 0; /* process exit status */

    float*   a_f32 = (float*)malloc((size_t)lda * m * sizeof(float));
    float*   b_f32 = (float*)malloc((size_t)ldb * k * sizeof(float));
    float*   c_ref = (float*)malloc((size_t)ldc * m * sizeof(float));
    float*   c_f32 = (float*)malloc((size_t)ldc * m * sizeof(float));
    float*   c_deq = (float*)malloc((size_t)ldc * m * sizeof(float));
    uint8_t* a_u8  = (uint8_t*)malloc((size_t)lda * m);
    int8_t*  b_s8  = (int8_t*)malloc((size_t)ldb * k);
    int8_t*  c_s8  = (int8_t*)malloc((size_t)ldc * m);

    if (!a_f32 || !b_f32 || !c_ref || !c_f32 || !c_deq || !a_u8 || !b_s8
        || !c_s8) {
        printf("Memory allocation failed\n");
        status = 1;
        goto cleanup;
    }

    srand(0xC0FFEE);

    printf("Quantized GEMM: U8 x S8, %lldx%lldx%lld\n\n", (long long)m,
           (long long)n, (long long)k);

    fill_grid(a_f32, m, k, lda, 0, 255, a_scale); /* non-negative */
    fill_grid(b_f32, k, n, ldb, -127, 127, b_scale);

    PRINT_TILE("A (F32 activations)", a_f32, lda, "%9.5f ");
    PRINT_TILE("B (F32 weights)", b_f32, ldb, "%9.5f ");

    reference_gemm(a_f32, lda, b_f32, ldb, c_ref, ldc, m, n, k);
    PRINT_TILE("C reference (F32)", c_ref, ldc, "%9.5f ");

    quantize_u8(a_f32, a_u8, m, k, lda, a_scale, a_zero_point);
    quantize_s8(b_f32, b_s8, k, n, ldb, b_scale);

    PRINT_TILE("A (U8)", a_u8, lda, "%5d ");
    PRINT_TILE("B (S8)", b_s8, ldb, "%5d ");

    printf("a_scale %.9g   b_scale %.9g   c_scale %.9g\n\n", a_scale, b_scale,
           c_scale);

    /* ---- Part 1: dequantize to F32 -------------------------------------
     * scale_factor = c_scale converts the S32 accumulator to the original
     * float domain in one step. */
    {
        float            sf_value = c_scale;
        dlp_sf_t         sf       = { .scale_factor      = &sf_value,
                                      .scale_factor_len  = 1,
                                      .scale_factor_type = DLP_F32,
                                      .scale_factor_dim  = DLP_PARAM_DIM_PER_TENSOR };
        dlp_scale_t      scale    = { .sf = &sf, .zp = NULL };
        DLP_POST_OP_TYPE seq[1]   = { SCALE };
        dlp_metadata_t   md       = { 0 };

        md.scale      = &scale;
        md.seq_vector = seq;
        md.seq_length = 1;

        aocl_gemm_u8s8s32of32('R', 'N', 'N', m, n, k, 1, a_u8, lda, 'N', b_s8,
                              ldb, 'N', 0, c_f32, ldc, &md);

        int rc = gemm_status("u8s8s32of32", &md);
        if (rc) {
            status = (rc == 77 && failures == 0) ? 77 : 1;
            goto cleanup;
        }

        PRINT_TILE("C dequantized (F32)", c_f32, ldc, "%9.5f ");
        failures += check("Part 1 -- U8S8 -> F32, exact round trip", c_ref,
                          c_f32, m, n, ldc, 0.0f, 0.0f);
    }

    /* ---- Part 2: requantize to S8 --------------------------------------
     * The output needs its own scale. Calibration would supply it; here it is
     * derived from the reference so the example is self-contained. */
    {
        float c_out_scale = max_abs(c_ref, m, n, ldc) / 127.0f;

        /* An all-zero reference leaves no range to quantize into. That means
         * the generated inputs degenerated, which is a defect in this example
         * rather than a runtime condition to paper over: substituting a scale
         * would make part 2 vacuous and still report success, which is the
         * failure mode this rewrite exists to remove. Written as !(x > 0) so
         * a NaN is rejected too. */
        if (!(c_out_scale > 0.0f)) {
            printf("Reference result has no dynamic range; the generated "
                   "inputs are degenerate and part 2 cannot be scaled.\n");
            status = 1;
            goto cleanup;
        }

        float  sf_value     = c_scale / c_out_scale;
        int8_t c_zero_point = 0;

        dlp_sf_t         sf     = { .scale_factor      = &sf_value,
                                    .scale_factor_len  = 1,
                                    .scale_factor_type = DLP_F32,
                                    .scale_factor_dim  = DLP_PARAM_DIM_PER_TENSOR };
        dlp_zp_t         zp     = { .zero_point      = &c_zero_point,
                                    .zero_point_len  = 1,
                                    .zero_point_type = DLP_S8 };
        dlp_scale_t      scale  = { .sf = &sf, .zp = &zp };
        DLP_POST_OP_TYPE seq[1] = { SCALE };
        dlp_metadata_t   md     = { 0 };

        md.scale      = &scale;
        md.seq_vector = seq;
        md.seq_length = 1;

        printf("c_out_scale %.9g   scale_factor %.9g\n\n", c_out_scale,
               sf_value);

        aocl_gemm_u8s8s32os8('R', 'N', 'N', m, n, k, 1, a_u8, lda, 'N', b_s8,
                             ldb, 'N', 0, c_s8, ldc, &md);

        int rc = gemm_status("u8s8s32os8", &md);
        if (rc) {
            status = (rc == 77 && failures == 0) ? 77 : 1;
            goto cleanup;
        }

        PRINT_TILE("C quantized (S8)", c_s8, ldc, "%5d ");

        for (md_t i = 0; i < m; i++)
            for (md_t j = 0; j < n; j++)
                c_deq[i * ldc + j] =
                    (float)(c_s8[i * ldc + j] - c_zero_point) * c_out_scale;

        PRINT_TILE("C dequantized from S8 (F32)", c_deq, ldc, "%9.5f ");

        /* Half an output step is the information-theoretic floor for an S8
         * result; anything beyond it means the scale factor is wrong. The
         * bound is exact in real arithmetic, so an element landing squarely
         * on it can fall a fraction of an ULP outside once the scale-factor
         * product is rounded. The margin absorbs that without weakening the
         * check: a wrong scale factor misses by tens of steps, not by an
         * ULP. */
        failures += check("Part 2 -- U8S8 -> S8 -> F32, lossy round trip",
                          c_ref, c_deq, m, n, ldc, c_out_scale, 0.5f + 1e-3f);
    }

    printf("%s\n", failures ? "FAILED" : "PASSED");
    status = failures ? 1 : 0;

cleanup:
    free(a_f32);
    free(b_f32);
    free(c_ref);
    free(c_f32);
    free(c_deq);
    free(a_u8);
    free(b_s8);
    free(c_s8);
    return status;
}
