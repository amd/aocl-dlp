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
#include <iostream>

namespace dlp::testing::classic::ref {

using dlp::testing::utils::bf16_to_f32;
using dlp::testing::utils::f32_to_bf16;

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
 * Reference: BF16×U4 GEMM, float output (WOQ). Asymmetric dequant: zero_point
 * required. C := alpha * A * dequant(B) + beta * C. A=bf16, B=packed u4 (2 per
 * byte), C=f32. dequant(B)[k,j] = (unpack_u4(B)[k,j] - zp[j]) * scale[j]
 * (per-tensor when sf_len/zp_len==1). reorder_b: B column-major (k,j), stride
 * k_updated=(k+1)&~1, index j*k_updated+k.
 *
 * To match DLP behavior: the kernel scales B in f32 then converts to bf16
 * before the BF16 matmul, so the effective dequant(B) is rounded to bf16. We
 * apply the same bf16 rounding here so ref matches for any scale (e.g.
 * sf=1.3f).
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
            float sum     = 0.0f;
            for (l = 0; l < k; l++) {
                uint8_t b_u4;
                if (transb == 'N' || transb == 'n') {
                    if (reorder_b)
                        b_u4 = unpack_u4(B, b_ldb, j, l);
                    else
                        b_u4 = unpack_u4(B, b_ldb, l, j);
                } else {
                    if (reorder_b)
                        b_u4 = unpack_u4(B, b_ldb, l, j);
                    else
                        b_u4 = unpack_u4(B, b_ldb, j, l);
                }
                float b_f32 = (static_cast<float>(b_u4) - zp_j) * scale_j;
                // Match DLP: kernel converts scaled B to bf16 before matmul
                b_f32 = bf16_to_f32(f32_to_bf16(b_f32));
                sum += bf16_to_f32(*a_ptr) * b_f32;
                a_ptr += a_stride;
            }

            if (beta != 0.0f)
                C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
            else
                C[i * ldc + j] = alpha * sum;
        }
    }
}

} // namespace dlp::testing::classic::ref
