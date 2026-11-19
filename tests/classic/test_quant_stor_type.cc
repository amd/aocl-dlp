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
 * Direct C-API tests for quant-metadata storage types. Validation must reject
 * a type that the consumer kernels cannot load, with DLP_CLSC_NOT_SUPPORTED,
 * before any scale or zero-point buffer is read.
 *
 * - Symmetric quant (s8s8, s8s4): scale factors must be F32 or BF16.
 * - Dynamic quant (f32s8, bf16s8): quant scale factors and zero point cannot
 *   be F16. Dequant scale factors still accept F16.
 */

#include "aocl_dlp.h"
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

namespace {

constexpr uint16_t kF16One = 0x3C00;

bool
isIsaSkip(dlp_clsc_err_t err)
{
    return err == DLP_CLSC_NOT_SUPPORTED;
}

// Per-group scales over a single group (group_size 0), so A has m values and
// B has n values. The buffers are F32-sized, which covers every narrower type.
struct SymQuantSetup
{
    dlp_metadata_t     md;
    dlp_quant_op_t     a_quant;
    dlp_quant_op_t     b_quant;
    dlp_qparam_t       a_sf;
    dlp_qparam_t       b_sf;
    std::vector<float> a_scale;
    std::vector<float> b_scale;

    SymQuantSetup(md_t m, md_t n, DLP_TYPE sf_type)
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
        a_sf.stor_type = sf_type;
        a_sf.outer_dim = DLP_PARAM_DIM_PER_GROUP;

        b_sf.data      = b_scale.data();
        b_sf.len       = n;
        b_sf.stor_type = sf_type;
        b_sf.outer_dim = DLP_PARAM_DIM_PER_GROUP;

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

// Per-tensor dynamic A quantization. Each parameter has its own storage so a
// test can switch one of them to F16 with a correctly sized buffer.
struct DynQuantSetup
{
    dlp_metadata_t md;
    dlp_quant_op_t a_quant;
    dlp_qparam_t   quant_sf;
    dlp_qparam_t   dequant_sf;
    dlp_qparam_t   zp;
    float          quant_sf_f32   = 1.0f;
    float          dequant_sf_f32 = 1.0f;
    int8_t         zp_s8          = 0;
    uint16_t       quant_sf_f16   = kF16One;
    uint16_t       dequant_sf_f16 = kF16One;
    uint16_t       zp_f16         = 0;

    explicit DynQuantSetup(bool with_zp)
    {
        std::memset(&md, 0, sizeof(md));
        std::memset(&a_quant, 0, sizeof(a_quant));
        std::memset(&quant_sf, 0, sizeof(quant_sf));
        std::memset(&dequant_sf, 0, sizeof(dequant_sf));
        std::memset(&zp, 0, sizeof(zp));

        quant_sf.data      = &quant_sf_f32;
        quant_sf.len       = 1;
        quant_sf.stor_type = DLP_F32;
        quant_sf.outer_dim = DLP_PARAM_DIM_PER_TENSOR;

        dequant_sf.data      = &dequant_sf_f32;
        dequant_sf.len       = 1;
        dequant_sf.stor_type = DLP_F32;
        dequant_sf.outer_dim = DLP_PARAM_DIM_PER_TENSOR;

        zp.data      = &zp_s8;
        zp.len       = 1;
        zp.stor_type = DLP_S8;
        zp.outer_dim = DLP_PARAM_DIM_PER_TENSOR;

        a_quant.quant_op_kind         = DLP_QUANT_OP_QUANTIZE;
        a_quant.src_type              = DLP_F32;
        a_quant.dst_type              = DLP_S8;
        a_quant.group_size            = 0;
        a_quant.quant_scale_factors   = &quant_sf;
        a_quant.dequant_scale_factors = &dequant_sf;
        a_quant.zero_point            = with_zp ? &zp : nullptr;

        md.a_quant_op            = &a_quant;
        md.error_hndl.error_code = DLP_CLSC_SUCCESS;
    }

    void useF16QuantScale()
    {
        quant_sf.data      = &quant_sf_f16;
        quant_sf.stor_type = DLP_F16;
    }

    void useF16DequantScale()
    {
        dequant_sf.data      = &dequant_sf_f16;
        dequant_sf.stor_type = DLP_F16;
    }

    void useF16ZeroPoint()
    {
        zp.data      = &zp_f16;
        zp.stor_type = DLP_F16;
    }
};

class SymQuantStorType : public ::testing::TestWithParam<DLP_TYPE>
{};

class DynQuantStorType : public ::testing::Test
{
  protected:
    static constexpr md_t m = 8, n = 16, k = 32;

    std::vector<float>  A_f32 = std::vector<float>((size_t)m * k, 1.0f);
    std::vector<int8_t> B     = std::vector<int8_t>((size_t)k * n, 1);
    std::vector<float>  C     = std::vector<float>((size_t)m * n, 0.0f);

    dlp_clsc_err_t runF32S8(DynQuantSetup& s)
    {
        aocl_gemm_f32s8s32of32('R', 'N', 'N', m, n, k, 1, A_f32.data(), k, 'n',
                               B.data(), n, 'n', 0, C.data(), n, &s.md);
        return s.md.error_hndl.error_code;
    }

    void SetUp() override
    {
        DynQuantSetup valid(true);
        if (isIsaSkip(runF32S8(valid))) {
            GTEST_SKIP() << "f32s8s32of32 not supported on this processor";
        }
    }
};

} // namespace

TEST_P(SymQuantStorType, S8S8Of32RowMajorRejects)
{
    const md_t          m = 8, n = 16, k = 32;
    std::vector<int8_t> A((size_t)m * k, 1);
    std::vector<int8_t> B((size_t)k * n, 1);
    std::vector<float>  C((size_t)m * n, 0.0f);

    SymQuantSetup valid(m, n, DLP_F32);
    aocl_gemm_s8s8s32of32_sym_quant('R', 'N', 'N', m, n, k, 1, A.data(), k, 'n',
                                    B.data(), n, 'n', 0, C.data(), n,
                                    &valid.md);
    if (isIsaSkip(valid.md.error_hndl.error_code)) {
        GTEST_SKIP() << "s8s8s32of32_sym_quant not supported on this processor";
    }

    SymQuantSetup bad(m, n, GetParam());
    aocl_gemm_s8s8s32of32_sym_quant('R', 'N', 'N', m, n, k, 1, A.data(), k, 'n',
                                    B.data(), n, 'n', 0, C.data(), n, &bad.md);
    EXPECT_EQ(bad.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

// Column major transposes the scales before the kernel is generated, so the
// type has to be rejected before that copy.
TEST_P(SymQuantStorType, S8S8Obf16ColumnMajorRejects)
{
    const md_t            m = 8, n = 16, k = 32;
    std::vector<int8_t>   A((size_t)m * k, 1);
    std::vector<int8_t>   B((size_t)k * n, 1);
    std::vector<bfloat16> C((size_t)m * n, 0);

    SymQuantSetup valid(m, n, DLP_F32);
    aocl_gemm_s8s8s32obf16_sym_quant('C', 'N', 'N', m, n, k, 1, A.data(), m,
                                     'n', B.data(), k, 'n', 0, C.data(), m,
                                     &valid.md);
    if (isIsaSkip(valid.md.error_hndl.error_code)) {
        GTEST_SKIP()
            << "s8s8s32obf16_sym_quant not supported on this processor";
    }

    SymQuantSetup bad(m, n, GetParam());
    aocl_gemm_s8s8s32obf16_sym_quant('C', 'N', 'N', m, n, k, 1, A.data(), m,
                                     'n', B.data(), k, 'n', 0, C.data(), m,
                                     &bad.md);
    EXPECT_EQ(bad.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST_P(SymQuantStorType, S8S4Of32Rejects)
{
    const md_t          m = 8, n = 16, k = 32;
    std::vector<int8_t> A((size_t)m * k, 1);
    std::vector<int8_t> B((size_t)k * ((n + 1) / 2), 1);
    std::vector<float>  C((size_t)m * n, 0.0f);

    SymQuantSetup valid(m, n, DLP_F32);
    aocl_gemm_s8s4s32of32('R', 'N', 'N', m, n, k, 1, A.data(), k, 'n', B.data(),
                          n, 'n', 0, C.data(), n, &valid.md);
    if (isIsaSkip(valid.md.error_hndl.error_code)) {
        GTEST_SKIP() << "s8s4s32of32 not supported on this processor";
    }

    SymQuantSetup bad(m, n, GetParam());
    aocl_gemm_s8s4s32of32('R', 'N', 'N', m, n, k, 1, A.data(), k, 'n', B.data(),
                          n, 'n', 0, C.data(), n, &bad.md);
    EXPECT_EQ(bad.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

INSTANTIATE_TEST_SUITE_P(UnsupportedScaleTypes,
                         SymQuantStorType,
                         ::testing::Values(DLP_F16, DLP_S8, DLP_U8, DLP_S32));

TEST(SymQuantStorTypeAccepted, S8S8Of32AcceptsF32AndBF16)
{
    const md_t          m = 8, n = 16, k = 32;
    std::vector<int8_t> A((size_t)m * k, 1);
    std::vector<int8_t> B((size_t)k * n, 1);
    std::vector<float>  C((size_t)m * n, 0.0f);

    for (DLP_TYPE sf_type : { DLP_F32, DLP_BF16 }) {
        SymQuantSetup s(m, n, sf_type);
        aocl_gemm_s8s8s32of32_sym_quant('R', 'N', 'N', m, n, k, 1, A.data(), k,
                                        'n', B.data(), n, 'n', 0, C.data(), n,
                                        &s.md);
        if (isIsaSkip(s.md.error_hndl.error_code)) {
            GTEST_SKIP()
                << "s8s8s32of32_sym_quant not supported on this processor";
        }
        EXPECT_EQ(s.md.error_hndl.error_code, DLP_CLSC_SUCCESS)
            << "sf_type " << static_cast<int>(sf_type);
    }
}

TEST_F(DynQuantStorType, F32S8RejectsF16QuantScale)
{
    DynQuantSetup s(false);
    s.useF16QuantScale();
    EXPECT_EQ(runF32S8(s), DLP_CLSC_NOT_SUPPORTED);
}

TEST_F(DynQuantStorType, F32S8RejectsF16ZeroPoint)
{
    DynQuantSetup s(true);
    s.useF16ZeroPoint();
    EXPECT_EQ(runF32S8(s), DLP_CLSC_NOT_SUPPORTED);
}

TEST_F(DynQuantStorType, F32S8AcceptsF16DequantScale)
{
    DynQuantSetup s(false);
    s.useF16DequantScale();
    EXPECT_EQ(runF32S8(s), DLP_CLSC_SUCCESS);
}

TEST_F(DynQuantStorType, BF16S8RejectsF16QuantScale)
{
    std::vector<bfloat16> A_bf16((size_t)m * k, 0x3F80);

    DynQuantSetup s(false);
    s.a_quant.src_type = DLP_BF16;
    s.useF16QuantScale();
    aocl_gemm_bf16s8s32of32('R', 'N', 'N', m, n, k, 1, A_bf16.data(), k, 'n',
                            B.data(), n, 'n', 0, C.data(), n, &s.md);
    EXPECT_EQ(s.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST_F(DynQuantStorType, BF16S8RejectsF16ZeroPoint)
{
    std::vector<bfloat16> A_bf16((size_t)m * k, 0x3F80);

    DynQuantSetup s(true);
    s.a_quant.src_type = DLP_BF16;
    s.useF16ZeroPoint();
    aocl_gemm_bf16s8s32of32('R', 'N', 'N', m, n, k, 1, A_bf16.data(), k, 'n',
                            B.data(), n, 'n', 0, C.data(), n, &s.md);
    EXPECT_EQ(s.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}
