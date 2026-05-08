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

/**
 * @file gemm_fp16fp16fp16of32_ref.cc
 * @brief Reference for FP16xFP16 -> F32 GEMM
 *
 * Output C is F32; accumulation is native FP16 inside the K loop. At the
 * post-ops boundary (once per KC block, matching the JIT 5-loop) the FP16
 * accumulator is widened to F32, multiplied by alpha (FP16 alpha widened
 * to F32 here), combined with beta (FP16 widened to F32 once at entry)
 * times the user F32 C, and stored back into user F32 C. C never round-
 * trips through FP16 - that's the entire point of the F32 output rail.
 *
 * Two modes via USE_GOLD_STANDARD:
 *   1 : gold-standard (mathematically clean reference; sequential FP16
 *       FMA over the full K, single F32 post-op at the end). Use this
 *       to detect REAL bugs - any deviation by the kernel here is a
 *       correctness issue, not a tile-walk-order coincidence.
 *   0 : kernel-matching (KC-blocked walk that mirrors the JIT's
 *       intermediate F32 stores - this is the default and is the only
 *       mode that can hit FP16-rail tolerances tightly when K > KC).
 *
 * GEMV note: the dedicated FP16 GEMV kernels carry a c_downscale-aware
 * F32 store rail. The reference therefore mirrors the of16 GEMV
 * computation order (32-lane SIMD tree-reduce for n=1, K_SUB_ITER stride
 * for m=1) and applies the F32 post-op directly to user C without a
 * round-trip through FP16. GEMV does NOT do KC blocking - the framework
 * hands the full K to the kernel - so there is no per-KC beta=1 sentinel
 * in the GEMV branches below.
 */

#include "adaptors/ref/gemm_ref.hh"
#include "utils/conversion_utils.hh"
#include <cmath>

namespace dlp::testing::classic::ref {

using dlp::testing::utils::f32_to_fp16;
using dlp::testing::utils::fp16_to_f32;

#define USE_GOLD_STANDARD 0

void
aocl_gemm_f16f16f16of32_ref(const char     order,
                            const char     transa,
                            const char     transb,
                            const md_t     m,
                            const md_t     n,
                            const md_t     k,
                            float16        alpha,
                            const float16* A,
                            int            lda,
                            const float16* B,
                            int            ldb,
                            float16        beta,
                            float*         C,
                            int            ldc,
                            dlp_metadata_t* /*post_ops*/)
{
    bool isRowMajor = (order == 'r' || order == 'R');
    bool isTransA   = (transa == 't' || transa == 'T');
    bool isTransB   = (transb == 't' || transb == 'T');

    auto getA = [&](md_t i, md_t j) -> float16 {
        if (isRowMajor) {
            return !isTransA ? A[i * lda + j] : A[j * lda + i];
        } else {
            return !isTransA ? A[j * lda + i] : A[i * lda + j];
        }
    };

    auto getB = [&](md_t i, md_t j) -> float16 {
        if (isRowMajor) {
            return !isTransB ? B[i * ldb + j] : B[j * ldb + i];
        } else {
            return !isTransB ? B[j * ldb + i] : B[i * ldb + j];
        }
    };

    auto getC = [&](md_t i, md_t j) -> float& {
        if (isRowMajor) {
            return C[i * ldc + j];
        } else {
            return C[j * ldc + i];
        }
    };

    // Must match dlp_gemm_blksz_map.h::F16F16F16OF16 - the of32 rail
    // shares the of16 cntx including KC=2048 so the JIT's KC blocking
    // lines up across both rails.
    const md_t KC = 2048;

    // Widen alpha and beta to F32 once at the entry. This matches the
    // JIT's F32 post-ops rail: alpha is consumed as FP16 by the K loop's
    // FMA but is also widened to F32 inside the post-ops scaleAlpha when
    // kernel-ops emit F32 store-back (processF32TilePostOps). beta is
    // FP16 in the public API; the JIT widens it once per KC via
    // vbroadcastss before vfmadd231ps. Replicate that exact precision here.
    const float alpha_f32 = fp16_to_f32(alpha);
    const float beta_f32  = fp16_to_f32(beta);

    // BLAS spec: alpha == 0 collapses to C := beta * C. Keep this on the
    // F32 rail (no FP16 round-trip) since user C is F32 and the in-place
    // store would otherwise lose precision through fp16 needlessly.
    if (alpha_f32 == 0.0f) {
        for (iter_t i = 0; i < m; i++) {
            for (iter_t j = 0; j < n; j++) {
                float& c_ref = getC(i, j);
                if (beta_f32 != 0.0f) {
                    c_ref = beta_f32 * c_ref;
                } else {
                    c_ref = 0.0f;
                }
            }
        }
        return;
    }

    // GEMV path: the dedicated FP16 GEMV kernels run a single full-K loop
    // (no KC blocking), then widen + apply alpha/beta in F32 directly to
    // user C. After the column-major M/N swap inside the dispatcher, the
    // kernel-side n=1 lane is whichever dimension is 1 in row-major space.
    bool isGemv = (m == 1 || n == 1);
    if (isGemv) {
        bool isN1Kernel = isRowMajor ? (n == 1) : (m == 1);

#if USE_GOLD_STANDARD
        // Gold standard: simple sequential FP16 FMA; one F32 post-op.
        for (iter_t i = 0; i < m; i++) {
            for (iter_t j = 0; j < n; j++) {
                float& c_ref = getC(i, j);

                float16 sum = static_cast<float16>(0);
                for (iter_t p = 0; p < k; p++) {
                    float16 a_val = getA(i, p);
                    float16 b_val = getB(p, j);

                    float sum_f32_acc = fp16_to_f32(sum);
                    float a_f32       = fp16_to_f32(a_val);
                    float b_f32       = fp16_to_f32(b_val);
                    sum_f32_acc       = std::fmaf(a_f32, b_f32, sum_f32_acc);
                    sum               = f32_to_fp16(sum_f32_acc);
                }

                float sum_f32 = fp16_to_f32(sum);
                if (beta_f32 != 0.0f) {
                    c_ref = alpha_f32 * sum_f32 + beta_f32 * c_ref;
                } else {
                    c_ref = alpha_f32 * sum_f32;
                }
            }
        }
#else
        // Kernel-matching: 32-lane SIMD tree-reduce for n=1, K_SUB_ITER=4
        // for m=1. Both finish the kernel-side accumulation in FP16, then
        // hand the FP16 scalar to the F32 post-op (alpha and beta in F32).
        for (iter_t i = 0; i < m; i++) {
            for (iter_t j = 0; j < n; j++) {
                float& c_ref = getC(i, j);

                float16 sum;
                if (isN1Kernel) {
                    const md_t SIMD_WIDTH           = 32;
                    float16    lane_acc[SIMD_WIDTH] = { 0 };
                    for (iter_t p = 0; p < k; p++) {
                        md_t    lane  = p % SIMD_WIDTH;
                        float16 a_val = getA(i, p);
                        float16 b_val = getB(p, j);

                        float acc_f32  = fp16_to_f32(lane_acc[lane]);
                        float a_f32    = fp16_to_f32(a_val);
                        float b_f32    = fp16_to_f32(b_val);
                        acc_f32        = std::fmaf(a_f32, b_f32, acc_f32);
                        lane_acc[lane] = f32_to_fp16(acc_f32);
                    }
                    // Tree reduction (matches kernel pattern).
                    for (iter_t i2 = 0; i2 < 16; i2++) {
                        float f1     = fp16_to_f32(lane_acc[i2]);
                        float f2     = fp16_to_f32(lane_acc[i2 + 16]);
                        lane_acc[i2] = f32_to_fp16(f1 + f2);
                    }
                    for (iter_t i2 = 0; i2 < 8; i2++) {
                        float f1     = fp16_to_f32(lane_acc[i2]);
                        float f2     = fp16_to_f32(lane_acc[i2 + 8]);
                        lane_acc[i2] = f32_to_fp16(f1 + f2);
                    }
                    for (iter_t i2 = 0; i2 < 4; i2++) {
                        float f1     = fp16_to_f32(lane_acc[i2]);
                        float f2     = fp16_to_f32(lane_acc[i2 + 4]);
                        lane_acc[i2] = f32_to_fp16(f1 + f2);
                    }
                    for (iter_t i2 = 0; i2 < 2; i2++) {
                        float f1     = fp16_to_f32(lane_acc[i2]);
                        float f2     = fp16_to_f32(lane_acc[i2 + 2]);
                        lane_acc[i2] = f32_to_fp16(f1 + f2);
                    }
                    {
                        float f1 = fp16_to_f32(lane_acc[0]);
                        float f2 = fp16_to_f32(lane_acc[1]);
                        sum      = f32_to_fp16(f1 + f2);
                    }
                } else {
                    // M=1 K_SUB_ITER=4 pattern.
                    const md_t K_SUB_ITER        = 4;
                    float16    accum[K_SUB_ITER] = { 0 };
                    md_t       k_groups          = k / K_SUB_ITER;
                    md_t       k_left            = k % K_SUB_ITER;
                    for (iter_t g = 0; g < k_groups; g++) {
                        for (iter_t s = 0; s < K_SUB_ITER; s++) {
                            md_t    p     = g * K_SUB_ITER + s;
                            float16 a_val = getA(i, p);
                            float16 b_val = getB(p, j);

                            float acc_f32 = fp16_to_f32(accum[s]);
                            float a_f32   = fp16_to_f32(a_val);
                            float b_f32   = fp16_to_f32(b_val);
                            acc_f32       = std::fmaf(a_f32, b_f32, acc_f32);
                            accum[s]      = f32_to_fp16(acc_f32);
                        }
                    }
                    for (iter_t s = 0; s < k_left; s++) {
                        md_t    p     = k_groups * K_SUB_ITER + s;
                        float16 a_val = getA(i, p);
                        float16 b_val = getB(p, j);

                        float acc_f32 = fp16_to_f32(accum[s]);
                        float a_f32   = fp16_to_f32(a_val);
                        float b_f32   = fp16_to_f32(b_val);
                        acc_f32       = std::fmaf(a_f32, b_f32, acc_f32);
                        accum[s]      = f32_to_fp16(acc_f32);
                    }
                    sum = accum[0];
                    for (iter_t s = 1; s < K_SUB_ITER; s++) {
                        float sum_f32_t = fp16_to_f32(sum);
                        float acc_f32   = fp16_to_f32(accum[s]);
                        sum             = f32_to_fp16(sum_f32_t + acc_f32);
                    }
                }

                // F32 post-op: alpha and beta in F32, applied to user C.
                float sum_f32 = fp16_to_f32(sum);
                if (beta_f32 != 0.0f) {
                    c_ref = alpha_f32 * sum_f32 + beta_f32 * c_ref;
                } else {
                    c_ref = alpha_f32 * sum_f32;
                }
            }
        }
#endif
        return;
    }

#if USE_GOLD_STANDARD
    // ============================================================
    // GOLD STANDARD: sequential FP16 FMA over the full K, single
    // F32 post-op at the end. Useful to catch real correctness
    // bugs - if the kernel deviates from this beyond fp16 ULP
    // bounds, something is wrong with the kernel itself.
    // ============================================================
    for (iter_t i = 0; i < m; i++) {
        for (iter_t j = 0; j < n; j++) {
            float& c_ref = getC(i, j);

            float16 sum = static_cast<float16>(0);
            for (iter_t p = 0; p < k; p++) {
                float16 a_val = getA(i, p);
                float16 b_val = getB(p, j);

                float sum_f32_acc = fp16_to_f32(sum);
                float a_f32       = fp16_to_f32(a_val);
                float b_f32       = fp16_to_f32(b_val);
                sum_f32_acc       = std::fmaf(a_f32, b_f32, sum_f32_acc);
                sum               = f32_to_fp16(sum_f32_acc);
            }

            // alpha widened to F32 for the F32 post-op rail. Note: the
            // JIT first multiplies alpha (FP16) into the FP16 tile
            // (vmulph), then widens the FP16 product to F32 - we widen
            // alpha and the FP16 sum independently here and multiply in
            // F32. The two are not bit-identical (FP16 vmulph rounds the
            // product to FP16 first) - that's why USE_GOLD_STANDARD is
            // strictly stricter than the kernel and should be used as a
            // bug-detection ladder rather than a default.
            float sum_f32 = fp16_to_f32(sum);
            if (beta_f32 != 0.0f) {
                c_ref = alpha_f32 * sum_f32 + beta_f32 * c_ref;
            } else {
                c_ref = alpha_f32 * sum_f32;
            }
        }
    }
#else
    // ============================================================
    // KERNEL-MATCHING: KC-blocked walk that mirrors the JIT's
    // intermediate F32 stores. This is the default mode and is
    // what production tolerances are calibrated against.
    //
    // Per-KC contract (matches processF32TilePostOps):
    //   1. K loop:        FP16 FMA -> FP16 tile (sum_fp16)
    //   2. scaleAlpha:    FP16 vmulph: sum_fp16 *= alpha_fp16
    //   3. Widen FP16->F32 (vcvtph2ps): sum_f32 = (float) sum_fp16
    //   4. beta combine:  vfmadd231ps: tile_f32 = sum_f32 + beta_f32*C_f32
    //   5. Store F32 in-place to user C
    //
    // Beta on the second-and-later KC blocks is 1.0f (the kernel writes
    // back to its own running F32 C, summing across KC blocks at full
    // F32 precision).
    // ============================================================
    for (iter_t i = 0; i < m; i++) {
        for (iter_t j = 0; j < n; j++) {
            float& c_ref = getC(i, j);

            for (iter_t pc = 0; pc < k; pc += KC) {
                md_t kc0 = (k - pc) < KC ? (k - pc) : KC;

                // K-block: FP16 FMA -> FP16 sum
                float16 sum = static_cast<float16>(0);
                for (iter_t p = 0; p < kc0; p++) {
                    float16 a_val = getA(i, pc + p);
                    float16 b_val = getB(pc + p, j);

                    float sum_f32_acc = fp16_to_f32(sum);
                    float a_f32       = fp16_to_f32(a_val);
                    float b_f32       = fp16_to_f32(b_val);
                    sum_f32_acc       = std::fmaf(a_f32, b_f32, sum_f32_acc);
                    sum               = f32_to_fp16(sum_f32_acc);
                }

                // scaleAlpha (FP16) then widen to F32. The JIT does
                // exactly this sequence (vmulph -> vcvtph2ps).
                float alpha_sum_f32 = fp16_to_f32(
                    f32_to_fp16(fp16_to_f32(sum) * fp16_to_f32(alpha)));

                // Beta selection: first KC uses user beta_f32; later
                // KCs use 1.0 (the kernel sums into its own running F32
                // C across KC blocks).
                float beta_kc = (pc == 0) ? beta_f32 : 1.0f;

                if (beta_kc != 0.0f) {
                    c_ref = alpha_sum_f32 + beta_kc * c_ref;
                } else {
                    c_ref = alpha_sum_f32;
                }
            }
        }
    }
#endif
}

} // namespace dlp::testing::classic::ref
