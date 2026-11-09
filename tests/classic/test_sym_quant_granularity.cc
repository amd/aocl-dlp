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
 * Direct C-API tests for scale-factor granularity on the int8 symmetric
 * quantized GEMM entry points. PER_TENSOR is not supported on these APIs
 * and must early-return DLP_CLSC_NOT_SUPPORTED (or skip cleanly when the
 * ISA itself is not present).
 */

#include "aocl_dlp.h"
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

namespace {

struct QuantSetup
{
    dlp_metadata_t     md;
    dlp_quant_op_t     a_quant;
    dlp_quant_op_t     b_quant;
    dlp_qparam_t       a_sf;
    dlp_qparam_t       b_sf;
    std::vector<float> a_scale;
    std::vector<float> b_scale;

    QuantSetup(md_t               m,
               md_t               n,
               DLP_PARAM_DIM_TYPE a_dim,
               DLP_PARAM_DIM_TYPE b_dim)
        : a_scale(static_cast<size_t>(m), 1.0f)
        , b_scale(static_cast<size_t>(n), 1.0f)
    {
        std::memset(&md, 0, sizeof(md));
        std::memset(&a_quant, 0, sizeof(a_quant));
        std::memset(&b_quant, 0, sizeof(b_quant));
        std::memset(&a_sf, 0, sizeof(a_sf));
        std::memset(&b_sf, 0, sizeof(b_sf));

        a_sf.data      = a_scale.data();
        a_sf.len       = m;
        a_sf.stor_type = DLP_F32;
        a_sf.outer_dim = a_dim;

        b_sf.data      = b_scale.data();
        b_sf.len       = n;
        b_sf.stor_type = DLP_F32;
        b_sf.outer_dim = b_dim;

        a_quant.quant_op_kind         = DLP_QUANT_OP_QUANTIZE;
        a_quant.src_type              = DLP_S8;
        a_quant.dst_type              = DLP_S8;
        a_quant.group_size            = 0;
        a_quant.dequant_scale_factors = &a_sf;

        b_quant.quant_op_kind         = DLP_QUANT_OP_QUANTIZE;
        b_quant.src_type              = DLP_S8;
        b_quant.dst_type              = DLP_S8;
        b_quant.group_size            = 0;
        b_quant.dequant_scale_factors = &b_sf;

        md.a_quant_op            = &a_quant;
        md.b_quant_op            = &b_quant;
        md.error_hndl.error_code = DLP_CLSC_SUCCESS;
    }
};

bool
isIsaSkip(dlp_clsc_err_t err)
{
    return err == DLP_CLSC_NOT_SUPPORTED;
}

} // namespace

TEST(SymQuantGranularity, S8S8Of32RejectsBPerTensor)
{
    const md_t          m = 8, n = 16, k = 32;
    std::vector<int8_t> A((size_t)m * k, 1);
    std::vector<int8_t> B((size_t)k * n, 1);
    std::vector<float>  C((size_t)m * n, 0.0f);

    QuantSetup valid(m, n, DLP_PARAM_DIM_PER_GROUP, DLP_PARAM_DIM_PER_GROUP);
    aocl_gemm_s8s8s32of32_sym_quant('R', 'N', 'N', m, n, k, 1, A.data(), k, 'n',
                                    B.data(), n, 'n', 0, C.data(), n,
                                    &valid.md);
    if (isIsaSkip(valid.md.error_hndl.error_code)) {
        GTEST_SKIP() << "s8s8s32of32_sym_quant not supported on this processor";
    }

    QuantSetup bad(m, n, DLP_PARAM_DIM_PER_GROUP, DLP_PARAM_DIM_PER_TENSOR);
    aocl_gemm_s8s8s32of32_sym_quant('R', 'N', 'N', m, n, k, 1, A.data(), k, 'n',
                                    B.data(), n, 'n', 0, C.data(), n, &bad.md);
    EXPECT_EQ(bad.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST(SymQuantGranularity, S8S8Of32RejectsAPerTensor)
{
    const md_t          m = 8, n = 16, k = 32;
    std::vector<int8_t> A((size_t)m * k, 1);
    std::vector<int8_t> B((size_t)k * n, 1);
    std::vector<float>  C((size_t)m * n, 0.0f);

    QuantSetup valid(m, n, DLP_PARAM_DIM_PER_GROUP, DLP_PARAM_DIM_PER_GROUP);
    aocl_gemm_s8s8s32of32_sym_quant('R', 'N', 'N', m, n, k, 1, A.data(), k, 'n',
                                    B.data(), n, 'n', 0, C.data(), n,
                                    &valid.md);
    if (isIsaSkip(valid.md.error_hndl.error_code)) {
        GTEST_SKIP() << "s8s8s32of32_sym_quant not supported on this processor";
    }

    QuantSetup bad(m, n, DLP_PARAM_DIM_PER_TENSOR, DLP_PARAM_DIM_PER_GROUP);
    aocl_gemm_s8s8s32of32_sym_quant('R', 'N', 'N', m, n, k, 1, A.data(), k, 'n',
                                    B.data(), n, 'n', 0, C.data(), n, &bad.md);
    EXPECT_EQ(bad.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST(SymQuantGranularity, S8S8Obf16RejectsBPerTensor)
{
    const md_t            m = 8, n = 16, k = 32;
    std::vector<int8_t>   A((size_t)m * k, 1);
    std::vector<int8_t>   B((size_t)k * n, 1);
    std::vector<bfloat16> C((size_t)m * n, 0);

    QuantSetup valid(m, n, DLP_PARAM_DIM_PER_GROUP, DLP_PARAM_DIM_PER_GROUP);
    aocl_gemm_s8s8s32obf16_sym_quant('R', 'N', 'N', m, n, k, 1, A.data(), k,
                                     'n', B.data(), n, 'n', 0, C.data(), n,
                                     &valid.md);
    if (isIsaSkip(valid.md.error_hndl.error_code)) {
        GTEST_SKIP()
            << "s8s8s32obf16_sym_quant not supported on this processor";
    }

    QuantSetup bad(m, n, DLP_PARAM_DIM_PER_GROUP, DLP_PARAM_DIM_PER_TENSOR);
    aocl_gemm_s8s8s32obf16_sym_quant('R', 'N', 'N', m, n, k, 1, A.data(), k,
                                     'n', B.data(), n, 'n', 0, C.data(), n,
                                     &bad.md);
    EXPECT_EQ(bad.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST(SymQuantGranularity, S8S4Of32RejectsBPerTensor)
{
    const md_t          m = 8, n = 16, k = 32;
    std::vector<int8_t> A((size_t)m * k, 1);
    std::vector<int8_t> B((size_t)k * ((n + 1) / 2), 1);
    std::vector<float>  C((size_t)m * n, 0.0f);

    QuantSetup valid(m, n, DLP_PARAM_DIM_PER_GROUP, DLP_PARAM_DIM_PER_GROUP);
    aocl_gemm_s8s4s32of32('R', 'N', 'N', m, n, k, 1, A.data(), k, 'n', B.data(),
                          n, 'n', 0, C.data(), n, &valid.md);
    if (isIsaSkip(valid.md.error_hndl.error_code)) {
        GTEST_SKIP() << "s8s4s32of32 not supported on this processor";
    }

    QuantSetup bad(m, n, DLP_PARAM_DIM_PER_GROUP, DLP_PARAM_DIM_PER_TENSOR);
    aocl_gemm_s8s4s32of32('R', 'N', 'N', m, n, k, 1, A.data(), k, 'n', B.data(),
                          n, 'n', 0, C.data(), n, &bad.md);
    EXPECT_EQ(bad.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST(SymQuantGranularity, S8S4Obf16RejectsAPerTensor)
{
    const md_t            m = 8, n = 16, k = 32;
    std::vector<int8_t>   A((size_t)m * k, 1);
    std::vector<int8_t>   B((size_t)k * ((n + 1) / 2), 1);
    std::vector<bfloat16> C((size_t)m * n, 0);

    QuantSetup valid(m, n, DLP_PARAM_DIM_PER_GROUP, DLP_PARAM_DIM_PER_GROUP);
    aocl_gemm_s8s4s32obf16('R', 'N', 'N', m, n, k, 1, A.data(), k, 'n',
                           B.data(), n, 'n', 0, C.data(), n, &valid.md);
    if (isIsaSkip(valid.md.error_hndl.error_code)) {
        GTEST_SKIP() << "s8s4s32obf16 not supported on this processor";
    }

    QuantSetup bad(m, n, DLP_PARAM_DIM_PER_TENSOR, DLP_PARAM_DIM_PER_GROUP);
    aocl_gemm_s8s4s32obf16('R', 'N', 'N', m, n, k, 1, A.data(), k, 'n',
                           B.data(), n, 'n', 0, C.data(), n, &bad.md);
    EXPECT_EQ(bad.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}
