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
using dlp::testing::utils::f32_to_bf16_vcvtneps2bf16;

/** Unpack one u4 from packed uint8: row*ldb+col, low nibble first; u4 in
 * [0,15], unsigned. */
static inline uint8_t
unpack_u4(const uint8_t* packed, md_t ldb, md_t row, md_t col)
{
    size_t  linear_idx = static_cast<size_t>(row) * ldb + col;
    size_t  byte_idx   = linear_idx / 2;
    int     shift      = static_cast<int>((linear_idx % 2) * 4);
    uint8_t raw_4b     = (packed[byte_idx] >> shift) & 0x0F;
    return raw_4b;
}

/**
 * Reference: BF16×U4 GEMM, float output (WOQ). Asymmetric dequant; zero_point
 * required. C := alpha * A * dequant(B) + beta * C. A=bf16, B=packed u4 (2 per
 * byte), C=f32. reorder_b: B column-major (k,j), stride k_updated=(k+1)&~1,
 * index j*k_updated+k.
 *
 * Integer-domain ZP (zp_type s8, etc.):
 *   dequant = (u4 - zp) * scale (per column when zp_len == n).
 *
 * Float-domain ZP (zp_type bf16, matches DLP zero_point_type DLP_BF16):
 *   dequant = (u4 - 8) * scale + zp_f32; zp buffer holds bfloat16 ZPs with
 *   the same layout as scale (per-tensor / per-n).
 *
 * To match DLP: dequant B is rounded with VCVTNE (f32_to_bf16_vcvtneps2bf16),
 * then K is stepped two at a time using bf16_dot2_fma_accumulate (scalar
 * std::fma, same order as VDPBF16PS; no SIMD intrinsics). Odd K uses one FMA.
 */
void
aocl_gemm_bf16u4f32of32_ref(const char            order,
                            const char            transa,
                            const char            transb,
                            const md_t            m,
                            const md_t            n,
                            const md_t            k,
                            float                 alpha,
                            const bfloat16*       A,
                            int                   lda,
                            const uint8_t*        B,
                            int                   ldb,
                            float                 beta,
                            float*                C,
                            int                   ldc,
                            void*                 b_scale_data,
                            void*                 b_zp_data,
                            md_t                  sf_len,
                            md_t                  zp_len,
                            framework::MatrixType sf_type,
                            framework::MatrixType zp_type,
                            bool                  reorder_b)
{
    if (b_scale_data == nullptr) {
        std::cerr << "bf16u4f32of32_ref: Missing required B scale factors"
                  << std::endl;
        return;
    }
    if (b_zp_data == nullptr) {
        std::cerr
            << "bf16u4f32of32_ref: U4 requires zero_point (asymmetric WOQ)."
            << std::endl;
        return;
    }
    if (order != 'R' && order != 'r') {
        std::cerr << "bf16u4f32of32_ref: Only row-major order is supported"
                  << std::endl;
        return;
    }

    const md_t k_updated = (k + 1) & ~static_cast<md_t>(1);
    const md_t b_ldb     = reorder_b ? k_updated : static_cast<md_t>(ldb);

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

    bool per_tensor_scale = (sf_len == 1);
    bool per_tensor_zp    = (zp_len == 1);
    auto getScale         = [&](md_t j) -> float {
        return getValueFromBuffer(b_scale_data, sf_type,
                                  per_tensor_scale ? 0 : j);
    };
    auto getZp = [&](md_t j) -> float {
        return getValueFromBuffer(b_zp_data, zp_type, per_tensor_zp ? 0 : j);
    };

    const bool float_domain_zp = (zp_type == framework::MatrixType::bf16);

    md_t i, j, l;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            const bfloat16* a_ptr;
            int             a_stride;
            if (transa == 'N' || transa == 'n') {
                a_ptr    = A + i * lda;
                a_stride = 1;
            } else {
                a_ptr    = A + i;
                a_stride = lda;
            }

            float scale_j = getScale(j);
            float zp_j    = getZp(j);

            // get b_u4 at index lk
            auto b_u4_at = [&](md_t lk) -> uint8_t {
                if (transb == 'N' || transb == 'n') {
                    if (reorder_b)
                        return unpack_u4(B, b_ldb, j, lk);
                    return unpack_u4(B, b_ldb, lk, j);
                }
                if (reorder_b)
                    return unpack_u4(B, b_ldb, lk, j);
                return unpack_u4(B, b_ldb, j, lk);
            };

            // dequant B to f32
            auto dequant_b_f32 = [&](uint8_t b_u4) -> float {
                return float_domain_zp
                           ? (static_cast<float>(b_u4) - 8.0f) * scale_j + zp_j
                           : (static_cast<float>(b_u4) - zp_j) * scale_j;
            };

            float sum = 0.0f;
            for (l = 0; l + 1 < k; l += 2) {
                // dequant B to f32
                float b0 = dequant_b_f32(b_u4_at(l));
                float b1 = dequant_b_f32(b_u4_at(l + 1));
                // convert f32 to bf16
                float b_bf0 = bf16_to_f32(f32_to_bf16_vcvtneps2bf16(b0));
                float b_bf1 = bf16_to_f32(f32_to_bf16_vcvtneps2bf16(b1));
                // convert a_ptr to f32
                float a_f32   = bf16_to_f32(*a_ptr);
                float a_f32_1 = bf16_to_f32(*(a_ptr + a_stride));
                // accumulate fma
                sum = std::fma(a_f32_1, b_bf1, sum);
                sum = std::fma(a_f32, b_bf0, sum);
                a_ptr += 2 * a_stride;
            }
            if (l < k) {
                float b_bf = bf16_to_f32(
                    f32_to_bf16_vcvtneps2bf16(dequant_b_f32(b_u4_at(l))));
                float a_f32 = bf16_to_f32(*a_ptr);
                sum         = std::fma(a_f32, b_bf, sum);
            }

            if (beta != 0.0f)
                C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
            else
                C[i * ldc + j] = alpha * sum;
        }
    }
}

} // namespace dlp::testing::classic::ref
