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
 */

#include "adaptors/ref/gemm_ref.hh"
#include "utils/conversion_utils.hh"
#include <cmath>

namespace dlp::testing::classic::ref {

using dlp::testing::utils::bf16_to_f32;
using dlp::testing::utils::f32_to_bf16_vcvtneps2bf16;

void
aocl_gemm_s8s8s32obf16_ref(const char      order,
                           const char      transa,
                           const char      transb,
                           const md_t      m,
                           const md_t      n,
                           const md_t      k,
                           int32_t         alpha,
                           const int8_t*   A,
                           int             lda,
                           const int8_t*   B,
                           int             ldb,
                           int32_t         beta,
                           bfloat16*       C,
                           int             ldc,
                           dlp_metadata_t* post_ops)
{
    // Reference implementation that mimics the VNNI vpdpbusd instruction
    // behavior. The vpdpbusd instruction performs: dst = dst + unsigned_A ×
    // signed_B (with accumulation).
    //
    // To compute s8×s8 using vpdpbusd (which expects unsigned×signed), we:
    // 1. Convert signed A to unsigned by adding 128: A_unsigned = A_signed +
    // 128
    // 2. Compute: (A_signed + 128) × B_signed + accumulator
    // 3. Apply bias correction to get the correct s8×s8 result
    //
    // Mathematical derivation:
    // (A_signed + 128) × B_signed = A_signed × B_signed + 128 × B_signed
    // Therefore: A_signed × B_signed = (A_signed + 128) × B_signed - 128 ×
    // B_signed Final result: A_signed × B_signed = dot_product - 128 ×
    // sum(B_signed)

    // Implementation of the reference kernel
    md_t i, j, l;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            int32_t       sum = 0;
            const int8_t *a_ptr, *b_ptr;
            int           a_stride, b_stride;

            if (order == 'R' || order == 'r') {
                if (transa == 'n' || transa == 'N') {
                    a_ptr    = A + i * lda;
                    a_stride = 1;
                } else {
                    a_ptr    = A + i;
                    a_stride = lda;
                }
                if (transb == 'n' || transb == 'N') {
                    b_ptr    = B + j;
                    b_stride = ldb;
                } else {
                    b_ptr    = B + j * ldb;
                    b_stride = 1;
                }
            } else {
                if (transa == 'n' || transa == 'N') {
                    a_ptr    = A + i;
                    a_stride = lda;
                } else {
                    a_ptr    = A + i * lda;
                    a_stride = 1;
                }
                if (transb == 'n' || transb == 'N') {
                    b_ptr    = B + j * ldb;
                    b_stride = 1;
                } else {
                    b_ptr    = B + j;
                    b_stride = ldb;
                }
            }
            const int8_t* a_k = a_ptr;
            const int8_t* b_k = b_ptr;

            int32_t dot_product = 0;
            int32_t b_sum       = 0; // Sum of B elements for bias correction

            // Loop over k dimension
            for (l = 0; l < k; l++) {
                // Convert signed A to unsigned (equivalent to adding 128)
                uint8_t a_unsigned = static_cast<uint8_t>(*a_k + 128);
                int8_t  b_signed   = *b_k;

                int32_t a_as_int32 =
                    static_cast<int32_t>(a_unsigned); // Always positive: 0-255
                int32_t b_as_int32 = static_cast<int32_t>(
                    b_signed); // Can be negative: -128 to 127
                dot_product += a_as_int32 * b_as_int32;

                // Accumulate B values for bias correction
                b_sum += b_as_int32;

                a_k += a_stride;
                b_k += b_stride;
            }

            // Apply bias correction: subtract 128 * sum(B) to get the correct
            // s8×s8 result This is because: (a+128)×b = a×b + 128×b, so we
            // subtract 128×sum(b)
            sum = dot_product - (128 * b_sum);

            // VNNI-style computation: Follow exact kernel typecast sequence
            // Step 1: Apply alpha scaling in int32 domain (matches kernel)
            int32_t alpha_times_sum = alpha * sum;

            // Step 2: Convert int32 result to float32 (matches kernel:
            // _mm512_cvtepi32_ps)
            int32_t result_i32 = alpha_times_sum;

            // Step 3: Handle beta*C term in float32 domain (matches kernel
            // post-ops flow)
            if (beta != 0) {
                float c_f32;
                if (order == 'R' || order == 'r')
                    c_f32 = bf16_to_f32(C[i * ldc + j]);
                else
                    c_f32 = bf16_to_f32(C[j * ldc + i]);

                int32_t c_int32 = static_cast<int32_t>(c_f32);
                int32_t betaC   = (beta * c_int32);
                result_i32 += betaC;
            }

            // Step 4: Convert final float32 result to bf16 (matches kernel
            // output)
            float    result_f32 = static_cast<float>(result_i32);
            bfloat16 bf_result  = f32_to_bf16_vcvtneps2bf16(result_f32);

            if (order == 'R' || order == 'r')
                C[i * ldc + j] = bf_result;
            else
                C[j * ldc + i] = bf_result;
        }
    }
}

} // namespace dlp::testing::classic::ref
