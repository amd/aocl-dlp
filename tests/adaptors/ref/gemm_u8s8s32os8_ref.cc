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
#include "adaptors/ref/gemm_ref.hh"

namespace dlp::testing::classic::ref {
void
aocl_gemm_u8s8s32os8_ref(const char      order,
                         const char      transa,
                         const char      transb,
                         const md_t      m,
                         const md_t      n,
                         const md_t      k,
                         int32_t         alpha,
                         const uint8_t*  A,
                         int             lda,
                         const int8_t*   B,
                         int             ldb,
                         int32_t         beta,
                         int8_t*         C,
                         int             ldc,
                         dlp_metadata_t* post_ops)
{

    // Implementation of the reference kernel
    md_t i, j, l;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            int32_t        sum = 0;
            const uint8_t* a_ptr;
            const int8_t*  b_ptr;
            int            a_stride, b_stride;

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
            const uint8_t* a_k = a_ptr;
            const int8_t*  b_k = b_ptr;

            // Loop over k dimension
            for (l = 0; l < k; l++) {
                uint8_t a_unsigned = *a_k; // Already unsigned: 0-255
                int8_t  b_signed   = *b_k; // Signed: -128 to 127

                int32_t a_as_int32 =
                    static_cast<int32_t>(a_unsigned); // Always positive: 0-255
                int32_t b_as_int32 = static_cast<int32_t>(
                    b_signed); // Can be negative: -128 to 127
                sum += a_as_int32 * b_as_int32;
                a_k += a_stride;
                b_k += b_stride;
            }

            // Compute the result and apply saturation (simulating hardware
            // behavior)
            int32_t result = 0;
            if (beta != 0) {
                result =
                    alpha * sum
                    + beta
                          * static_cast<int32_t>(
                              C[order == 'R' || order == 'r' ? i * ldc + j
                                                             : j * ldc + i]);
            } else {
                result = alpha * sum;
            }

            // Simulate hardware saturation to int8_t range [-128, 127]
            // This matches what CPU saturation instructions would do
            int8_t saturated_result;
            if (result > 127) {
                saturated_result = 127;
            } else if (result < -128) {
                saturated_result = -128;
            } else {
                saturated_result = static_cast<int8_t>(result);
            }

            if (order == 'R' || order == 'r')
                C[i * ldc + j] = saturated_result;
            else
                C[j * ldc + i] = saturated_result;
        }
    }
}
} // namespace dlp::testing::classic::ref
