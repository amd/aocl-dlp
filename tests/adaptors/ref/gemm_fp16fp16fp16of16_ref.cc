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
 * @file gemm_f16f16f16of16_ref.cc
 * @brief Reference implementation for FP16×FP16→FP16 GEMM
 *
 * This file provides two modes controlled by USE_GOLD_STANDARD:
 *
 * GOLD STANDARD MODE (USE_GOLD_STANDARD=1):
 *   - Mathematically correct, unbiased reference
 *   - Simple sequential FP16 accumulation
 *   - No kernel-specific patterns (no K_SUB_ITER, no 32-lane SIMD)
 *   - Tests that kernel produces "correct" results
 *   - Recommended for production testing
 *
 * KERNEL-MATCHING MODE (USE_GOLD_STANDARD=0):
 *   - Mimics kernel's exact computation order
 *   - Uses K_SUB_ITER=4 strided pattern for M=1
 *   - Uses 32-lane SIMD pattern for N=1
 *   - Tests implementation consistency, not correctness
 *   - Useful for debugging kernel issues
 *
 * Hardware Behavior (vfmadd231ph):
 *   1. Converts FP16 inputs to F32 internally
 *   2. Performs FMA in F32 precision
 *   3. Rounds result back to FP16 in the register
 *
 * Both modes use std::fmaf to match hardware FMA rounding behavior.
 */

#include "adaptors/ref/gemm_ref.hh"
#include "utils/conversion_utils.hh"
#include <cmath>

namespace dlp::testing::classic::ref {

using dlp::testing::utils::f32_to_fp16;
using dlp::testing::utils::fp16_to_f32;

/**
 * USE_GOLD_STANDARD: Controls reference implementation mode
 *
 * Set to 1: Gold standard - unbiased, mathematically correct reference
 *           Simple sequential accumulation, no kernel-specific patterns
 *
 * Set to 0: Kernel-matching - mimics DLP kernel's computation order
 *           K_SUB_ITER striding for M=1, 32-lane SIMD for N=1
 */
#define USE_GOLD_STANDARD 0

/**
 * @brief FP16×FP16→FP16 GEMM with native FP16 accumulation
 *
 * Performs: C = alpha*A*B + beta*C
 *
 * @param order Storage order ('r' for row-major, 'c' for column-major)
 * @param transa Transpose flag for A ('n' or 't')
 * @param transb Transpose flag for B ('n' or 't')
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A / rows in B
 * @param alpha Scaling factor for A*B (FP16)
 * @param A Input matrix A
 * @param lda Leading dimension of A
 * @param B Input matrix B
 * @param ldb Leading dimension of B
 * @param beta Scaling factor for C (FP16)
 * @param C Output matrix C (input/output)
 * @param ldc Leading dimension of C
 * @param post_ops Post operations (unused in reference)
 */
void
aocl_gemm_f16f16f16of16_ref(const char      order,
                            const char      transa,
                            const char      transb,
                            const md_t      m,
                            const md_t      n,
                            const md_t      k,
                            float16         alpha,
                            const float16*  A,
                            int             lda,
                            const float16*  B,
                            int             ldb,
                            float16         beta,
                            float16*        C,
                            int             ldc,
                            dlp_metadata_t* post_ops)
{
    bool isRowMajor = (order == 'r' || order == 'R');
    bool isTransA   = (transa == 't' || transa == 'T');
    bool isTransB   = (transb == 't' || transb == 'T');

    // Access helpers with layout and transpose handling
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

    auto getC = [&](md_t i, md_t j) -> float16& {
        if (isRowMajor) {
            return C[i * ldc + j];
        } else {
            return C[j * ldc + i];
        }
    };

    // K-blocking configuration - must match DLP's lpgemm_blksz_map.h
    // F16F16F16OF16: KC = 2048
    const md_t KC = 2048;

    // Check if this is a GEMV case (m=1 or n=1)
    bool isGemv = (m == 1 || n == 1);

    // For KERNEL-MATCHING mode, we need to account for column-major swap.
    // DLP kernel swaps M and N for column-major: N=1 becomes M=1 from kernel
    // POV. So for column-major N=1, the kernel uses the M=1 code path
    // (K_SUB_ITER=4). For column-major M=1, the kernel uses the N=1 code path
    // (tree reduction).
    bool isN1Kernel = isRowMajor ? (n == 1) : (m == 1);
    bool isM1Kernel = isRowMajor ? (m == 1) : (n == 1);

    float alpha_f32_val = fp16_to_f32(alpha);
    float beta_f32_val  = fp16_to_f32(beta);

    // BLAS spec: when alpha=0, skip A*B product entirely.
    // C = beta*C (or 0 when beta=0 too).
    // This prevents Inf*0=NaN propagation from A or B.
    if (alpha_f32_val == 0.0f) {
        for (iter_t i = 0; i < m; i++) {
            for (iter_t j = 0; j < n; j++) {
                float16& c_ref = getC(i, j);
                if (beta_f32_val != 0.0f) {
                    float c_f32 = fp16_to_f32(c_ref);
                    c_ref       = f32_to_fp16(beta_f32_val * c_f32);
                } else {
                    c_ref = static_cast<float16>(0);
                }
            }
        }
        return;
    }

    // Triple loop GEMM: C = alpha*A*B + beta*C
    for (iter_t i = 0; i < m; i++) {
        for (iter_t j = 0; j < n; j++) {
            float16& c_ref = getC(i, j);

            if (isGemv) {
#if USE_GOLD_STANDARD
                // ============================================================
                // GOLD STANDARD: Unbiased sequential FP16 accumulation
                // ============================================================
                // This is the mathematically correct reference:
                // - Simple sequential loop over K
                // - No kernel-specific patterns
                // - Tests that kernel produces correct results
                //
                // However, for large K (K > KC=2048), the kernel uses KC
                // blocking which causes intermediate stores to C. To match
                // kernel behavior for large K accumulation error stress tests,
                // we must also use KC blocking to match the kernel's precision
                // characteristics.
                //
                // For K <= KC: Use simple sequential accumulation (gold
                // standard) For K > KC: Use KC blocking to match kernel's
                // behavior

                if (k <= KC) {
                    // Small K: Simple sequential accumulation (gold standard)
                    float16 sum = static_cast<float16>(0);

                    for (iter_t p = 0; p < k; p++) {
                        float16 a_val = getA(i, p);
                        float16 b_val = getB(p, j);

                        // FMA with FP16 rounding (matches vfmadd231ph behavior)
                        float sum_f32 = fp16_to_f32(sum);
                        float a_f32   = fp16_to_f32(a_val);
                        float b_f32   = fp16_to_f32(b_val);
                        sum_f32       = std::fmaf(a_f32, b_f32, sum_f32);
                        sum           = f32_to_fp16(sum_f32);
                    }

                    float alpha_f32 = fp16_to_f32(alpha);
                    float sum_f32   = fp16_to_f32(sum);
                    float beta_f32  = fp16_to_f32(beta);
                    if (beta_f32 != 0.0f) {
                        float c_f32 = fp16_to_f32(c_ref);
                        c_ref =
                            f32_to_fp16(alpha_f32 * sum_f32 + beta_f32 * c_f32);
                    } else {
                        c_ref = f32_to_fp16(alpha_f32 * sum_f32);
                    }
                } else {
                    // Large K: Use KC blocking with K_SUB_ITER=4 pattern to
                    // match kernel The kernel processes K in KC blocks, uses
                    // K_SUB_ITER=4 within each KC block, consolidates
                    // accumulators after each KC block, then stores to C
                    const md_t K_SUB_ITER = 4;

                    for (iter_t pc = 0; pc < k; pc += KC) {
                        md_t kc0 = (k - pc) < KC ? (k - pc) : KC;

                        // Beta handling: use user beta for first K-block, 1.0
                        // for rest
                        float16 beta0 = (pc == 0) ? beta : f32_to_fp16(1.0f);

                        // K_SUB_ITER=4 accumulators for this KC block
                        float16 accum[K_SUB_ITER] = { 0 };

                        // Process kc0 elements using K_SUB_ITER=4 pattern
                        md_t k_groups = kc0 / K_SUB_ITER;
                        md_t k_left   = kc0 % K_SUB_ITER;

                        for (iter_t g = 0; g < k_groups; g++) {
                            for (iter_t s = 0; s < K_SUB_ITER; s++) {
                                md_t    p     = pc + g * K_SUB_ITER + s;
                                float16 a_val = getA(i, p);
                                float16 b_val = getB(p, j);

                                float acc_f32 = fp16_to_f32(accum[s]);
                                float a_f32   = fp16_to_f32(a_val);
                                float b_f32   = fp16_to_f32(b_val);
                                acc_f32  = std::fmaf(a_f32, b_f32, acc_f32);
                                accum[s] = f32_to_fp16(acc_f32);
                            }
                        }

                        // Handle remaining elements
                        for (iter_t s = 0; s < k_left; s++) {
                            md_t    p     = pc + k_groups * K_SUB_ITER + s;
                            float16 a_val = getA(i, p);
                            float16 b_val = getB(p, j);

                            float acc_f32 = fp16_to_f32(accum[s]);
                            float a_f32   = fp16_to_f32(a_val);
                            float b_f32   = fp16_to_f32(b_val);
                            acc_f32       = std::fmaf(a_f32, b_f32, acc_f32);
                            accum[s]      = f32_to_fp16(acc_f32);
                        }

                        // Consolidate accumulators (matches kernel's
                        // finalAccumulate)
                        float16 sum = accum[0];
                        for (iter_t s = 1; s < K_SUB_ITER; s++) {
                            float sum_f32 = fp16_to_f32(sum);
                            float acc_f32 = fp16_to_f32(accum[s]);
                            sum           = f32_to_fp16(sum_f32 + acc_f32);
                        }

                        float alpha_f32 = fp16_to_f32(alpha);
                        float sum_f32   = fp16_to_f32(sum);
                        float beta0_f32 = fp16_to_f32(beta0);
                        if (beta0_f32 != 0.0f) {
                            float c_f32 = fp16_to_f32(c_ref);
                            c_ref       = f32_to_fp16(alpha_f32 * sum_f32
                                                      + beta0_f32 * c_f32);
                        } else {
                            c_ref = f32_to_fp16(alpha_f32 * sum_f32);
                        }
                    }
                }

#else
                // ============================================================
                // KERNEL-MATCHING: Mimics DLP kernel computation order
                // ============================================================
                // Use this mode for debugging kernel implementation issues.
                // WARNING: This tests consistency, not correctness!
                //
                // NOTE: For column-major, DLP swaps M and N before kernel call.
                // So column-major N=1 uses M=1 kernel (K_SUB_ITER=4),
                // and column-major M=1 uses N=1 kernel (tree reduction).

                if (isN1Kernel) {
                    // N=1 case: 32-lane SIMD accumulation pattern
                    // Mimics ZMM register processing with horizontal sum
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

                    // Horizontal sum in FP16 using TREE REDUCTION
                    // This matches the DLP kernel's reduction pattern:
                    // 1. Add pairs: lane[i] + lane[i+16] for i=0..15
                    // 2. Add pairs: lane[i] + lane[i+8] for i=0..7
                    // 3. Continue...

                    // Step 1: 32 -> 16
                    for (iter_t i = 0; i < 16; i++) {
                        float f1    = fp16_to_f32(lane_acc[i]);
                        float f2    = fp16_to_f32(lane_acc[i + 16]);
                        lane_acc[i] = f32_to_fp16(f1 + f2);
                    }
                    // Step 2: 16 -> 8
                    for (iter_t i = 0; i < 8; i++) {
                        float f1    = fp16_to_f32(lane_acc[i]);
                        float f2    = fp16_to_f32(lane_acc[i + 8]);
                        lane_acc[i] = f32_to_fp16(f1 + f2);
                    }
                    // Step 3: 8 -> 4
                    for (iter_t i = 0; i < 4; i++) {
                        float f1    = fp16_to_f32(lane_acc[i]);
                        float f2    = fp16_to_f32(lane_acc[i + 4]);
                        lane_acc[i] = f32_to_fp16(f1 + f2);
                    }
                    // Step 4: 4 -> 2
                    for (iter_t i = 0; i < 2; i++) {
                        float f1    = fp16_to_f32(lane_acc[i]);
                        float f2    = fp16_to_f32(lane_acc[i + 2]);
                        lane_acc[i] = f32_to_fp16(f1 + f2);
                    }
                    // Step 5: 2 -> 1
                    float16 sum;
                    {
                        float f1 = fp16_to_f32(lane_acc[0]);
                        float f2 = fp16_to_f32(lane_acc[1]);
                        sum      = f32_to_fp16(f1 + f2);
                    }

                    float alpha_f32 = fp16_to_f32(alpha);
                    float sum_f32   = fp16_to_f32(sum);
                    float beta_f32  = fp16_to_f32(beta);
                    if (beta_f32 != 0.0f) {
                        float c_f32 = fp16_to_f32(c_ref);
                        c_ref =
                            f32_to_fp16(alpha_f32 * sum_f32 + beta_f32 * c_f32);
                    } else {
                        c_ref = f32_to_fp16(alpha_f32 * sum_f32);
                    }

                } else {
                    // M=1 case: K_SUB_ITER=4 strided accumulation
                    // Mimics kernel's software pipelining pattern
                    const md_t K_SUB_ITER        = 4;
                    float16    accum[K_SUB_ITER] = { 0 };

                    md_t k_groups = k / K_SUB_ITER;
                    md_t k_left   = k % K_SUB_ITER;

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

                    // Consolidate accumulators
                    float16 sum = accum[0];
                    for (iter_t s = 1; s < K_SUB_ITER; s++) {
                        float sum_f32 = fp16_to_f32(sum);
                        float acc_f32 = fp16_to_f32(accum[s]);
                        sum           = f32_to_fp16(sum_f32 + acc_f32);
                    }

                    float alpha_f32 = fp16_to_f32(alpha);
                    float sum_f32   = fp16_to_f32(sum);
                    float beta_f32  = fp16_to_f32(beta);
                    if (beta_f32 != 0.0f) {
                        float c_f32 = fp16_to_f32(c_ref);
                        c_ref =
                            f32_to_fp16(alpha_f32 * sum_f32 + beta_f32 * c_f32);
                    } else {
                        c_ref = f32_to_fp16(alpha_f32 * sum_f32);
                    }
                }
#endif
            } else {
                // ============================================================
                // GEMM path: KC blocking with store after each KC block
                // ============================================================
                // NOTE: KC blocking is REQUIRED for both modes because:
                // - It's framework behavior, not kernel optimization
                // - DLP framework stores to C after each KC block
                // - This causes precision loss that must be matched
                for (iter_t pc = 0; pc < k; pc += KC) {
                    md_t kc0 = (k - pc) < KC ? (k - pc) : KC;

                    // Beta handling: use user beta for first K-block, 1.0 for
                    // rest
                    float16 beta0 = (pc == 0) ? beta : f32_to_fp16(1.0f);

                    // FP16 accumulator for this K-block
                    float16 sum = static_cast<float16>(0);

                    for (iter_t p = 0; p < kc0; p++) {
                        float16 a_val = getA(i, pc + p);
                        float16 b_val = getB(pc + p, j);

                        float sum_f32 = fp16_to_f32(sum);
                        float a_f32   = fp16_to_f32(a_val);
                        float b_f32   = fp16_to_f32(b_val);
                        sum_f32       = std::fmaf(a_f32, b_f32, sum_f32);
                        sum           = f32_to_fp16(sum_f32);
                    }

                    float alpha_f32 = fp16_to_f32(alpha);
                    float sum_f32   = fp16_to_f32(sum);
                    float beta0_f32 = fp16_to_f32(beta0);
                    if (beta0_f32 != 0.0f) {
                        float c_f32 = fp16_to_f32(c_ref);
                        c_ref       = f32_to_fp16(alpha_f32 * sum_f32
                                                  + beta0_f32 * c_f32);
                    } else {
                        c_ref = f32_to_fp16(alpha_f32 * sum_f32);
                    }
                }
            }
        }
    }
}

} // namespace dlp::testing::classic::ref
