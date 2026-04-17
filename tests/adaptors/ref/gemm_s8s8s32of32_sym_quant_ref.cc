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
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace dlp::testing::classic::ref {

namespace {

    using dlp::testing::framework::MatrixType;
    using dlp::testing::utils::bf16_to_f32;

    float read_scale_as_f32(const void* base, md_t idx, MatrixType stor_type)
    {
        if (stor_type == MatrixType::bf16) {
            return bf16_to_f32(static_cast<const bfloat16*>(base)[idx]);
        }
        return static_cast<const float*>(base)[idx];
    }

    /** VNNI-style dot over k indices [l_begin, l_end), plus sum of B in that
     * range. */
    void vnni_dot_segment(const char    order,
                          const char    transa,
                          const char    transb,
                          const md_t    m,
                          const md_t    n,
                          const md_t    k,
                          const md_t    i,
                          const md_t    j,
                          const md_t    l_begin,
                          const md_t    l_end,
                          const int8_t* A,
                          int           lda,
                          const int8_t* B,
                          int           ldb,
                          int32_t*      out_dot,
                          int32_t*      out_b_sum)
    {
        (void)m;
        (void)n;
        (void)k;
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

        const int8_t* a_k = a_ptr + static_cast<ptrdiff_t>(l_begin) * a_stride;
        const int8_t* b_k = b_ptr + static_cast<ptrdiff_t>(l_begin) * b_stride;

        int32_t dot_product = 0;
        int32_t b_sum       = 0;
        for (md_t l = l_begin; l < l_end; ++l) {
            uint8_t a_unsigned = static_cast<uint8_t>(*a_k + 128);
            int8_t  b_signed   = *b_k;
            int32_t a_as_int32 = static_cast<int32_t>(a_unsigned);
            int32_t b_as_int32 = static_cast<int32_t>(b_signed);
            dot_product += a_as_int32 * b_as_int32;
            b_sum += b_as_int32;
            a_k += a_stride;
            b_k += b_stride;
        }
        *out_dot   = dot_product;
        *out_b_sum = b_sum;
    }

    float c_elem(const char order, const float* C, int ldc, md_t i, md_t j)
    {
        if (order == 'R' || order == 'r')
            return C[i * ldc + j];
        return C[j * ldc + i];
    }

    void c_set(const char order, float* C, int ldc, md_t i, md_t j, float v)
    {
        if (order == 'R' || order == 'r')
            C[i * ldc + j] = v;
        else
            C[j * ldc + i] = v;
    }

} // namespace

void
aocl_gemm_s8s8s32of32_sym_quant_ref(const char    order,
                                    const char    transa,
                                    const char    transb,
                                    const md_t    m,
                                    const md_t    n,
                                    const md_t    k,
                                    int32_t       alpha,
                                    const int8_t* A,
                                    int           lda,
                                    const int8_t* B,
                                    int           ldb,
                                    int32_t       beta,
                                    float*        C,
                                    int           ldc,
                                    md_t          group_size,
                                    const void*   a_scale,
                                    const void*   b_scale,
                                    md_t          num_groups,
                                    MatrixType    scale_stor_type)
{
    md_t gs = group_size;
    if (gs == 0) {
        gs = k;
    } else if (gs > k) {
        // Match dlp_gemm_translate_to_group_postops_list: group_size may not
        // exceed k. Scale matrices are laid out for ceil(k / yaml_gs) groups;
        // treating the single partial tail as one group of size k matches DLP.
        gs = k;
    }

    const md_t ng = num_groups;

    // Match dlp_gemm_s8s8s32_sym_quant.c K-paneling: KC from S8S8S32OS32 Zen4
    // blksz (dlp_gemm_blksz_map.h), raised to at least group_size when needed.
    md_t KC = 2048;
    if (gs > KC) {
        KC = gs;
    }

    const float alpha_f = static_cast<float>(alpha);

    for (md_t i = 0; i < m; ++i) {
        for (md_t j = 0; j < n; ++j) {
            float c_run = 0.0f;
            if (beta != 0) {
                c_run = c_elem(order, C, ldc, i, j);
            }

            // Step pc by full KC each time (see dlp_gemm_s8s8s32_sym_quant.c).
            for (md_t pc = 0; pc < k; pc += KC) {
                const md_t kc0     = std::min(k - pc, KC);
                const md_t seg_end = pc + kc0;

                float chunk_acc = 0.0f;

                const md_t g_first = pc / gs;
                const md_t g_last  = (pc + kc0 - 1) / gs;

                for (md_t g = g_first; g <= g_last; ++g) {
                    md_t l0 = std::max(g * gs, pc);
                    md_t l1 = std::min(std::min(g * gs + gs, k), seg_end);
                    if (l0 >= l1) {
                        continue;
                    }

                    int32_t dot_raw, b_sum;
                    vnni_dot_segment(order, transa, transb, m, n, k, i, j, l0,
                                     l1, A, lda, B, ldb, &dot_raw, &b_sum);
                    const int32_t corr = dot_raw - (128 * b_sum);

                    const float sa =
                        read_scale_as_f32(a_scale, i * ng + g, scale_stor_type);
                    const float sb =
                        read_scale_as_f32(b_scale, g * n + j, scale_stor_type);
                    // Match Zen4 sym_quant kernels: (cvt_s32_to_f32(corr) *
                    // b_scl) * a_scl (CVT_ACCUM_REG_APPLY_SCALES_M_N).
                    chunk_acc += (static_cast<float>(corr) * sb) * sa;
                }

                const int32_t beta0 = (pc == 0) ? beta : 1;
                c_run = alpha_f * chunk_acc + static_cast<float>(beta0) * c_run;
            }

            c_set(order, C, ldc, i, j, c_run);
        }
    }
}

} // namespace dlp::testing::classic::ref
