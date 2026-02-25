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
#include "utils/conversion_utils.hh"
#include <cmath>
#include <iostream>

namespace dlp::testing::classic::ref {

using dlp::testing::utils::bf16_to_f32;

/**
 * Reference implementation of BF16×S8 GEMM with float output.
 *
 * This function computes C := alpha * op(A) * op(B) + beta * C where:
 * - A is bfloat16, quantized on-the-fly to int8
 * - B is int8
 * - C is float
 * - Intermediate accumulation is in int32
 *
 * Performs GEMM with on-the-fly quantization of A, VNNI-style multiplication,
 * bias correction, scaling, and float output.
 */
void
aocl_gemm_bf16s8s32of32_ref(const char            order,
                            const char            transa,
                            const char            transb,
                            const md_t            m,
                            const md_t            n,
                            const md_t            k,
                            int32_t               alpha,
                            const bfloat16*       A,
                            int                   lda,
                            const int8_t*         B,
                            int                   ldb,
                            int32_t               beta,
                            float*                C,
                            int                   ldc,
                            void*                 a_pre_quant_sf_data,
                            void*                 a_pre_quant_zp_data,
                            void*                 a_post_quant_sf_data,
                            void*                 a_post_quant_zp_data,
                            md_t                  sf_len,
                            md_t                  zp_len,
                            framework::MatrixType sf_type,
                            framework::MatrixType zp_type)
{
    // =========================================================================
    // Validate quantization metadata
    // =========================================================================
    if (a_pre_quant_sf_data == nullptr || a_post_quant_sf_data == nullptr) {
        std::cerr << "bf16s8s32of32_ref: Missing required scale factors"
                  << std::endl;
        return;
    }

    // =========================================================================
    // Helper lambdas to extract quantization parameters
    // =========================================================================

    auto getValueFromBuffer = [](void* data, framework::MatrixType data_type,
                                 md_t index) -> float {
        switch (data_type) {
            case framework::MatrixType::f32:
                return static_cast<const float*>(data)[index];
            case framework::MatrixType::bf16:
                return bf16_to_f32(static_cast<const bfloat16*>(data)[index]);
            case framework::MatrixType::s32:
                return static_cast<float>(
                    static_cast<const int32_t*>(data)[index]);
            case framework::MatrixType::s8:
                return static_cast<float>(
                    static_cast<const int8_t*>(data)[index]);
            case framework::MatrixType::u8:
                return static_cast<float>(
                    static_cast<const uint8_t*>(data)[index]);
            default:
                return static_cast<const float*>(data)[index];
        }
    };

    // =========================================================================
    // Extract quantization metadata
    // =========================================================================

    // Determine if we're using per-tensor or per-row quantization
    bool per_tensor_scale = (sf_len == 1);
    bool per_tensor_zp    = (zp_len == 1);
    bool is_symmetric     = (a_pre_quant_zp_data == nullptr);

    // =========================================================================
    // Main computation loop
    // =========================================================================
    md_t i, j, l;

    for (i = 0; i < m; i++) {
        // ---------------------------------------------------------------------
        // Get quantization parameters for current row
        // ---------------------------------------------------------------------
        md_t scale_idx = per_tensor_scale ? 0 : i;

        float a_pre_quant_sf =
            getValueFromBuffer(a_pre_quant_sf_data, sf_type, scale_idx);
        float a_post_quant_sf =
            getValueFromBuffer(a_post_quant_sf_data, sf_type, scale_idx);

        // Zero-point is only used for asymmetric quantization
        float a_pre_quant_zp  = 0.0f;
        float a_post_quant_zp = 0.0f;
        if (!is_symmetric) {
            md_t zp_idx = per_tensor_zp ? 0 : i;
            a_pre_quant_zp =
                getValueFromBuffer(a_pre_quant_zp_data, zp_type, zp_idx);
            a_post_quant_zp =
                getValueFromBuffer(a_post_quant_zp_data, zp_type, zp_idx);
        }

        for (j = 0; j < n; j++) {
            const bfloat16 *a_ptr, *a_k;
            const int8_t *  b_ptr, *b_k;
            int             a_stride, b_stride;

            // -----------------------------------------------------------------
            // Determine matrix access pattern based on layout and transpose
            // -----------------------------------------------------------------
            if (order == 'R' || order == 'r') {
                // Row-major layout
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
                // Column-major layout
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

            a_k = a_ptr;
            b_k = b_ptr;

            int32_t dot_product = 0; // Accumulator for (A+128)×B
            int32_t b_sum       = 0; // Sum of B elements for bias correction

            // -----------------------------------------------------------------
            // Inner product computation with on-the-fly quantization
            // -----------------------------------------------------------------
            for (l = 0; l < k; l++) {
                // Step 1: Quantize A element from BF16 to S8
                float a_f32 = bf16_to_f32(*a_k);

                // Apply quantization formula:
                // Symmetric: Q = round(A * scale)
                // Asymmetric: Q = round(A * scale - zp)
                float a_rounded;
                if (is_symmetric) {
                    a_rounded = std::nearbyint(a_f32 * a_pre_quant_sf);
                } else {
                    a_rounded =
                        std::nearbyint(a_f32 * a_pre_quant_sf - a_pre_quant_zp);
                }

                // Clamp to S8 range [-128, 127]
                if (a_rounded < -128.0f)
                    a_rounded = -128.0f;
                if (a_rounded > 127.0f)
                    a_rounded = 127.0f;
                int8_t a_s8 = static_cast<int8_t>(a_rounded);

                // Step 2: VNNI-style computation (vpdpbusd instruction)
                // VNNI instruction computes: unsigned_A × signed_B
                // Convert signed S8 to unsigned U8 by adding 128
                uint8_t a_unsigned = static_cast<uint8_t>(a_s8 + 128);
                int8_t  b_signed   = *b_k;

                // Compute: (A + 128) × B
                int32_t a_as_int32 =
                    static_cast<int32_t>(a_unsigned); // [0, 255]
                int32_t b_as_int32 =
                    static_cast<int32_t>(b_signed); // [-128, 127]
                dot_product += a_as_int32 * b_as_int32;

                // Accumulate B values for bias correction
                b_sum += b_as_int32;

                a_k += a_stride;
                b_k += b_stride;
            }

            // -----------------------------------------------------------------
            // Step 3: Bias correction
            // -----------------------------------------------------------------
            // Since we computed (A+128)×B instead of A×B, we need to subtract
            // the bias: (A+128)×B = A×B + 128×B
            // Therefore: A×B = (A+128)×B - 128×sum(B)
            int32_t sum = dot_product - (b_sum * 128);

            // -----------------------------------------------------------------
            // Step 4: Apply alpha scaling
            // -----------------------------------------------------------------
            int32_t alpha_times_sum = alpha * sum;

            // -----------------------------------------------------------------
            // Step 5: Add beta*C term
            // -----------------------------------------------------------------
            int32_t result_i32 = alpha_times_sum;
            if (beta != 0) {
                float c_f32;
                if (order == 'R' || order == 'r')
                    c_f32 = C[i * ldc + j];
                else
                    c_f32 = C[j * ldc + i];

                int32_t c_int32 = static_cast<int32_t>(c_f32);
                int32_t betaC   = (beta * c_int32);
                result_i32 += betaC;
            }

            float result_f32 = static_cast<float>(result_i32);

            // -----------------------------------------------------------------
            // Step 6: Dequantization
            // -----------------------------------------------------------------
            if (!is_symmetric) {
                // For asymmetric quantization, add back the zero-point
                // correction term: sum(B) × zero_point
                result_f32 = result_f32 + (b_sum)*a_post_quant_zp;
            }

            // Apply inverse scale factor
            result_f32 = result_f32 * a_post_quant_sf;

            // -----------------------------------------------------------------
            // Step 7: Store result
            // -----------------------------------------------------------------
            if (order == 'R' || order == 'r')
                C[i * ldc + j] = result_f32;
            else
                C[j * ldc + i] = result_f32;
        }
    }
}

} // namespace dlp::testing::classic::ref
