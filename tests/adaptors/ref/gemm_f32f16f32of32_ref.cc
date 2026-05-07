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
 * @file gemm_f32f16f32of32_ref.cc
 * @brief Reference implementation for F32×FP16→F32 mixed-precision GEMM
 *
 * Performs: C = alpha * A * B + beta * C
 * where A is float*, B is float16*, C is float*.
 * All accumulation in F32. B elements are converted to F32 via fp16_to_f32
 * before accumulation.
 */

#include "adaptors/ref/gemm_ref.hh"
#include "utils/conversion_utils.hh"

namespace dlp::testing::classic::ref {

using dlp::testing::utils::fp16_to_f32;

void
aocl_gemm_f32f16f32of32_ref(const char     order,
                            const char     transa,
                            const char     transb,
                            const md_t     m,
                            const md_t     n,
                            const md_t     k,
                            float          alpha,
                            const float*   A,
                            int            lda,
                            const float16* B,
                            int            ldb,
                            float          beta,
                            float*         C,
                            int            ldc,
                            dlp_metadata_t* /*post_ops*/)
{
    bool isRowMajor = (order == 'r' || order == 'R');
    bool isTransA   = (transa == 't' || transa == 'T');
    bool isTransB   = (transb == 't' || transb == 'T');

    auto getA = [&](md_t i, md_t j) -> float {
        if (isRowMajor) {
            return !isTransA ? A[i * lda + j] : A[j * lda + i];
        } else {
            return !isTransA ? A[j * lda + i] : A[i * lda + j];
        }
    };

    auto getB = [&](md_t i, md_t j) -> float {
        float16 b_val;
        if (isRowMajor) {
            b_val = !isTransB ? B[i * ldb + j] : B[j * ldb + i];
        } else {
            b_val = !isTransB ? B[j * ldb + i] : B[i * ldb + j];
        }
        return fp16_to_f32(b_val);
    };

    auto getC = [&](md_t i, md_t j) -> float& {
        if (isRowMajor) {
            return C[i * ldc + j];
        } else {
            return C[j * ldc + i];
        }
    };

    for (iter_t i = 0; i < m; i++) {
        for (iter_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (iter_t p = 0; p < k; p++) {
                sum += getA(i, p) * getB(p, j);
            }
            if (beta != 0.0f) {
                getC(i, j) = alpha * sum + beta * getC(i, j);
            } else {
                getC(i, j) = alpha * sum;
            }
        }
    }
}

} // namespace dlp::testing::classic::ref
