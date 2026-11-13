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
 * Direct C-API early-return tests for bf16s4 / bf16u4 weight-only GEMM.
 * Invalid granularity, missing/unsupported quant params, column-major layout,
 * and non-reordered B must fail before the kernel runs (or skip when the
 * AVX512_BF16 ISA is absent).
 */

#include "aocl_dlp.h"
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

namespace {

constexpr md_t kM = 8;
constexpr md_t kN = 16;
constexpr md_t kK = 32;

struct WoqSetup
{
    dlp_metadata_t      md;
    dlp_quant_op_t      b_quant;
    dlp_qparam_t        sf;
    dlp_qparam_t        zp;
    std::vector<float>  scale;
    std::vector<int8_t> zp_s8;
    std::vector<float>  zp_f32;

    WoqSetup(DLP_PARAM_DIM_TYPE scale_dim,
             md_t               scale_len,
             bool               with_zp,
             DLP_PARAM_DIM_TYPE zp_dim  = DLP_PARAM_DIM_PER_CHANNEL,
             md_t               zp_len  = 1,
             DLP_TYPE           zp_type = DLP_S8)
        : scale(static_cast<size_t>(scale_len > 0 ? scale_len : 1), 1.0f)
        , zp_s8(static_cast<size_t>(zp_len > 0 ? zp_len : 1), 0)
        , zp_f32(static_cast<size_t>(zp_len > 0 ? zp_len : 1), 0.0f)
    {
        std::memset(&md, 0, sizeof(md));
        std::memset(&b_quant, 0, sizeof(b_quant));
        std::memset(&sf, 0, sizeof(sf));
        std::memset(&zp, 0, sizeof(zp));

        sf.data      = scale.data();
        sf.len       = scale_len;
        sf.stor_type = DLP_F32;
        sf.outer_dim = scale_dim;

        b_quant.quant_op_kind         = DLP_QUANT_OP_DEQUANTIZE;
        b_quant.src_type              = DLP_BF16;
        b_quant.dst_type              = DLP_BF16;
        b_quant.group_size            = 0;
        b_quant.dequant_scale_factors = &sf;

        if (with_zp) {
            zp.len       = zp_len;
            zp.stor_type = zp_type;
            zp.outer_dim = zp_dim;
            zp.data = (zp_type == DLP_F32) ? static_cast<void*>(zp_f32.data())
                                           : static_cast<void*>(zp_s8.data());
            b_quant.zero_point = &zp;
        }

        md.b_quant_op            = &b_quant;
        md.error_hndl.error_code = DLP_CLSC_SUCCESS;
    }
};

bool
isIsaSkip(dlp_clsc_err_t err)
{
    return err == DLP_CLSC_NOT_SUPPORTED;
}

void
fillGemmBufs(std::vector<bfloat16>& A,
             std::vector<int8_t>&   B,
             std::vector<float>&    C)
{
    A.assign(static_cast<size_t>(kM) * kK, static_cast<bfloat16>(0));
    B.assign(static_cast<size_t>(kK) * ((kN + 1) / 2), 1);
    C.assign(static_cast<size_t>(kM) * kN, 0.0f);
}

void
callS4Of32(char            order,
           char            transa,
           char            mem_b,
           const bfloat16* A,
           const int8_t*   B,
           float*          C,
           dlp_metadata_t* md)
{
    aocl_gemm_bf16s4f32of32(order, transa, 'N', kM, kN, kK, 1.0f, A, kK, 'n', B,
                            kN, mem_b, 0.0f, C, kN, md);
}

void
callU4Of32(char            order,
           char            transa,
           char            mem_b,
           const bfloat16* A,
           const uint8_t*  B,
           float*          C,
           dlp_metadata_t* md)
{
    aocl_gemm_bf16u4f32of32(order, transa, 'N', kM, kN, kK, 1.0f, A, kK, 'n', B,
                            kN, mem_b, 0.0f, C, kN, md);
}

bool
isaSupportedS4()
{
    std::vector<bfloat16> A;
    std::vector<int8_t>   B;
    std::vector<float>    C;
    fillGemmBufs(A, B, C);
    dlp_metadata_t md;
    std::memset(&md, 0, sizeof(md));
    callS4Of32('R', 'N', 'r', A.data(), B.data(), C.data(), &md);
    return !isIsaSkip(md.error_hndl.error_code);
}

bool
isaSupportedU4()
{
    std::vector<bfloat16> A;
    std::vector<int8_t>   B;
    std::vector<float>    C;
    fillGemmBufs(A, B, C);
    dlp_metadata_t md;
    std::memset(&md, 0, sizeof(md));
    callU4Of32('R', 'N', 'r', A.data(),
               reinterpret_cast<const uint8_t*>(B.data()), C.data(), &md);
    return !isIsaSkip(md.error_hndl.error_code);
}

// GTEST_SKIP() expands to a return, so it must appear in the test body itself.
// Calling it from a helper marks the test skipped but lets the body run on.
#define SKIP_IF_NO_ISA_S4()                                                    \
    do {                                                                       \
        if (!isaSupportedS4()) {                                               \
            GTEST_SKIP() << "bf16s4f32of32 not supported on this processor";   \
        }                                                                      \
    } while (0)

#define SKIP_IF_NO_ISA_U4()                                                    \
    do {                                                                       \
        if (!isaSupportedU4()) {                                               \
            GTEST_SKIP() << "bf16u4f32of32 not supported on this processor";   \
        }                                                                      \
    } while (0)

// PER_CHANNEL + group_size < K + len == n: kernels must index by column, not
// group*n + col. kK=32, group_size=16 => two K-groups; scale buffer is n long.
// group_size must match the value later passed to GEMM (packed buffer has no
// header for KC alignment).
bool
reorderPackedB(std::vector<int8_t>& packed,
               std::vector<int8_t>& reordered,
               md_t                 group_size)
{
    packed.assign(static_cast<size_t>(kK) * ((kN + 1) / 2), 1);
    dlp_metadata_t md;
    dlp_quant_op_t b_quant;
    std::memset(&md, 0, sizeof(md));
    std::memset(&b_quant, 0, sizeof(b_quant));
    b_quant.quant_op_kind = DLP_QUANT_OP_DEQUANTIZE;
    b_quant.src_type      = DLP_S4;
    b_quant.dst_type      = DLP_BF16;
    b_quant.group_size    = group_size;
    md.b_quant_op         = &b_quant;
    msz_t nbytes =
        aocl_get_reorder_buf_size_bf16s4f32of32('R', 'N', 'B', kK, kN, &md);
    if (nbytes == 0 || isIsaSkip(md.error_hndl.error_code)) {
        return false;
    }
    reordered.assign(static_cast<size_t>(nbytes), 0);
    aocl_reorder_bf16s4f32of32('R', 'N', 'B', packed.data(), reordered.data(),
                               kK, kN, kN, &md);
    return md.error_hndl.error_code == DLP_CLSC_SUCCESS;
}

void
callBatchS4Of32(char            order,
                char            transa,
                char            mem_b,
                const bfloat16* A,
                const int8_t*   B,
                float*          C,
                dlp_metadata_t* md)
{
    const char      order_v = order, transa_v = transa, transb_v = 'N';
    const md_t      m_v = kM, n_v = kN, k_v = kK;
    const float     alpha_v = 1.0f, beta_v = 0.0f;
    const md_t      lda_v = kK, ldb_v = kN, ldc_v = kN;
    const md_t      group_count = 1, group_size = 1;
    const char      mem_a = 'n', mem_b_v = mem_b;
    const bfloat16* a_ptr  = A;
    const int8_t*   b_ptr  = B;
    float*          c_ptr  = C;
    dlp_metadata_t* md_ptr = md;

    aocl_batch_gemm_bf16s4f32of32(&order_v, &transa_v, &transb_v, &m_v, &n_v,
                                  &k_v, &alpha_v, &a_ptr, &lda_v, &b_ptr,
                                  &ldb_v, &beta_v, &c_ptr, &ldc_v, group_count,
                                  &group_size, &mem_a, &mem_b_v, &md_ptr);
}

void
callBatchU4Of32(char            order,
                char            transa,
                char            mem_b,
                const bfloat16* A,
                const uint8_t*  B,
                float*          C,
                dlp_metadata_t* md)
{
    const char      order_v = order, transa_v = transa, transb_v = 'N';
    const md_t      m_v = kM, n_v = kN, k_v = kK;
    const float     alpha_v = 1.0f, beta_v = 0.0f;
    const md_t      lda_v = kK, ldb_v = kN, ldc_v = kN;
    const md_t      group_count = 1, group_size = 1;
    const char      mem_a = 'n', mem_b_v = mem_b;
    const bfloat16* a_ptr  = A;
    const uint8_t*  b_ptr  = B;
    float*          c_ptr  = C;
    dlp_metadata_t* md_ptr = md;

    aocl_batch_gemm_bf16u4f32of32(&order_v, &transa_v, &transb_v, &m_v, &n_v,
                                  &k_v, &alpha_v, &a_ptr, &lda_v, &b_ptr,
                                  &ldb_v, &beta_v, &c_ptr, &ldc_v, group_count,
                                  &group_size, &mem_a, &mem_b_v, &md_ptr);
}

} // namespace

TEST(WoqGranularity, S4RejectsScalePerToken)
{
    SKIP_IF_NO_ISA_S4();
    std::vector<bfloat16> A;
    std::vector<int8_t>   B;
    std::vector<float>    C;
    fillGemmBufs(A, B, C);

    WoqSetup bad(DLP_PARAM_DIM_PER_TOKEN, kN, false);
    callS4Of32('R', 'N', 'r', A.data(), B.data(), C.data(), &bad.md);
    EXPECT_EQ(bad.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST(WoqGranularity, S4RejectsZeroPoint)
{
    SKIP_IF_NO_ISA_S4();
    std::vector<bfloat16> A;
    std::vector<int8_t>   B;
    std::vector<float>    C;
    fillGemmBufs(A, B, C);

    WoqSetup bad(DLP_PARAM_DIM_PER_CHANNEL, kN, true, DLP_PARAM_DIM_PER_CHANNEL,
                 kN, DLP_S8);
    callS4Of32('R', 'N', 'r', A.data(), B.data(), C.data(), &bad.md);
    EXPECT_EQ(bad.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST(WoqGranularity, S4RejectsColumnMajor)
{
    SKIP_IF_NO_ISA_S4();
    std::vector<bfloat16> A;
    std::vector<int8_t>   B;
    std::vector<float>    C;
    fillGemmBufs(A, B, C);

    WoqSetup setup(DLP_PARAM_DIM_PER_CHANNEL, kN, false);
    aocl_gemm_bf16s4f32of32('C', 'N', 'N', kM, kN, kK, 1.0f, A.data(), kM, 'n',
                            B.data(), kK, 'r', 0.0f, C.data(), kM, &setup.md);
    EXPECT_EQ(setup.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST(WoqGranularity, S4RejectsNonReorderedB)
{
    SKIP_IF_NO_ISA_S4();
    std::vector<bfloat16> A;
    std::vector<int8_t>   B;
    std::vector<float>    C;
    fillGemmBufs(A, B, C);

    WoqSetup setup(DLP_PARAM_DIM_PER_CHANNEL, kN, false);
    callS4Of32('R', 'N', 'n', A.data(), B.data(), C.data(), &setup.md);
    EXPECT_EQ(setup.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST(WoqGranularity, S4RejectsNullScale)
{
    SKIP_IF_NO_ISA_S4();
    std::vector<bfloat16> A;
    std::vector<int8_t>   B;
    std::vector<float>    C;
    fillGemmBufs(A, B, C);

    dlp_metadata_t md;
    dlp_quant_op_t b_quant;
    std::memset(&md, 0, sizeof(md));
    std::memset(&b_quant, 0, sizeof(b_quant));
    b_quant.quant_op_kind         = DLP_QUANT_OP_DEQUANTIZE;
    b_quant.dequant_scale_factors = nullptr;
    md.b_quant_op                 = &b_quant;

    callS4Of32('R', 'N', 'r', A.data(), B.data(), C.data(), &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NULL_POINTER);
}

TEST(WoqGranularity, S4RejectsNullQuantOp)
{
    SKIP_IF_NO_ISA_S4();
    std::vector<bfloat16> A;
    std::vector<int8_t>   B;
    std::vector<float>    C;
    fillGemmBufs(A, B, C);

    dlp_metadata_t md;
    std::memset(&md, 0, sizeof(md));
    callS4Of32('R', 'N', 'r', A.data(), B.data(), C.data(), &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NULL_POINTER);
}

TEST(WoqGranularity, U4RejectsScalePerToken)
{
    SKIP_IF_NO_ISA_U4();
    std::vector<bfloat16> A;
    std::vector<int8_t>   B;
    std::vector<float>    C;
    fillGemmBufs(A, B, C);

    WoqSetup bad(DLP_PARAM_DIM_PER_TOKEN, kN, true, DLP_PARAM_DIM_PER_CHANNEL,
                 kN, DLP_S8);
    callU4Of32('R', 'N', 'r', A.data(),
               reinterpret_cast<const uint8_t*>(B.data()), C.data(), &bad.md);
    EXPECT_EQ(bad.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST(WoqGranularity, U4RejectsZpPerToken)
{
    SKIP_IF_NO_ISA_U4();
    std::vector<bfloat16> A;
    std::vector<int8_t>   B;
    std::vector<float>    C;
    fillGemmBufs(A, B, C);

    WoqSetup bad(DLP_PARAM_DIM_PER_CHANNEL, kN, true, DLP_PARAM_DIM_PER_TOKEN,
                 kN, DLP_S8);
    callU4Of32('R', 'N', 'r', A.data(),
               reinterpret_cast<const uint8_t*>(B.data()), C.data(), &bad.md);
    EXPECT_EQ(bad.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST(WoqGranularity, U4RejectsMissingZp)
{
    SKIP_IF_NO_ISA_U4();
    std::vector<bfloat16> A;
    std::vector<int8_t>   B;
    std::vector<float>    C;
    fillGemmBufs(A, B, C);

    WoqSetup bad(DLP_PARAM_DIM_PER_CHANNEL, kN, false);
    callU4Of32('R', 'N', 'r', A.data(),
               reinterpret_cast<const uint8_t*>(B.data()), C.data(), &bad.md);
    EXPECT_EQ(bad.md.error_hndl.error_code, DLP_CLSC_NULL_POINTER);
}

TEST(WoqGranularity, U4RejectsF32ZpType)
{
    SKIP_IF_NO_ISA_U4();
    std::vector<bfloat16> A;
    std::vector<int8_t>   B;
    std::vector<float>    C;
    fillGemmBufs(A, B, C);

    WoqSetup bad(DLP_PARAM_DIM_PER_CHANNEL, kN, true, DLP_PARAM_DIM_PER_CHANNEL,
                 kN, DLP_F32);
    callU4Of32('R', 'N', 'r', A.data(),
               reinterpret_cast<const uint8_t*>(B.data()), C.data(), &bad.md);
    EXPECT_EQ(bad.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST(WoqGranularity, U4RejectsColumnMajor)
{
    SKIP_IF_NO_ISA_U4();
    std::vector<bfloat16> A;
    std::vector<int8_t>   B;
    std::vector<float>    C;
    fillGemmBufs(A, B, C);

    WoqSetup setup(DLP_PARAM_DIM_PER_CHANNEL, kN, true,
                   DLP_PARAM_DIM_PER_CHANNEL, kN, DLP_S8);
    aocl_gemm_bf16u4f32of32('C', 'N', 'N', kM, kN, kK, 1.0f, A.data(), kM, 'n',
                            reinterpret_cast<const uint8_t*>(B.data()), kK, 'r',
                            0.0f, C.data(), kM, &setup.md);
    EXPECT_EQ(setup.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST(WoqGranularity, U4RejectsNonReorderedB)
{
    SKIP_IF_NO_ISA_U4();
    std::vector<bfloat16> A;
    std::vector<int8_t>   B;
    std::vector<float>    C;
    fillGemmBufs(A, B, C);

    WoqSetup setup(DLP_PARAM_DIM_PER_CHANNEL, kN, true,
                   DLP_PARAM_DIM_PER_CHANNEL, kN, DLP_S8);
    callU4Of32('R', 'N', 'n', A.data(),
               reinterpret_cast<const uint8_t*>(B.data()), C.data(), &setup.md);
    EXPECT_EQ(setup.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST(WoqGranularity, S4Obf16RejectsScalePerToken)
{
    SKIP_IF_NO_ISA_S4();
    std::vector<bfloat16> A(static_cast<size_t>(kM) * kK, 0);
    std::vector<int8_t>   B(static_cast<size_t>(kK) * ((kN + 1) / 2), 1);
    std::vector<bfloat16> C(static_cast<size_t>(kM) * kN, 0);

    WoqSetup bad(DLP_PARAM_DIM_PER_TOKEN, kN, false);
    aocl_gemm_bf16s4f32obf16('R', 'N', 'N', kM, kN, kK, 1.0f, A.data(), kK, 'n',
                             B.data(), kN, 'r', 0.0f, C.data(), kN, &bad.md);
    EXPECT_EQ(bad.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST(WoqGranularity, U4Obf16RejectsMissingZp)
{
    SKIP_IF_NO_ISA_U4();
    std::vector<bfloat16> A(static_cast<size_t>(kM) * kK, 0);
    std::vector<int8_t>   B(static_cast<size_t>(kK) * ((kN + 1) / 2), 1);
    std::vector<bfloat16> C(static_cast<size_t>(kM) * kN, 0);

    WoqSetup bad(DLP_PARAM_DIM_PER_CHANNEL, kN, false);
    aocl_gemm_bf16u4f32obf16('R', 'N', 'N', kM, kN, kK, 1.0f, A.data(), kK, 'n',
                             reinterpret_cast<const uint8_t*>(B.data()), kN,
                             'r', 0.0f, C.data(), kN, &bad.md);
    EXPECT_EQ(bad.md.error_hndl.error_code, DLP_CLSC_NULL_POINTER);
}

TEST(WoqGranularity, S4PerChannelWithGroupSizeSucceeds)
{
    SKIP_IF_NO_ISA_S4();
    constexpr md_t      group_size = 16;
    std::vector<int8_t> packed, reordered;
    if (!reorderPackedB(packed, reordered, group_size)) {
        GTEST_SKIP() << "bf16s4 reorder not supported on this processor";
    }

    std::vector<bfloat16> A(static_cast<size_t>(kM) * kK, 0);
    std::vector<float>    C(static_cast<size_t>(kM) * kN, 0.0f);

    WoqSetup setup(DLP_PARAM_DIM_PER_CHANNEL, kN, false);
    setup.b_quant.group_size = group_size;
    callS4Of32('R', 'N', 'r', A.data(), reordered.data(), C.data(), &setup.md);
    EXPECT_EQ(setup.md.error_hndl.error_code, DLP_CLSC_SUCCESS);
}

TEST(WoqGranularity, U4PerChannelWithGroupSizeSucceeds)
{
    SKIP_IF_NO_ISA_U4();
    constexpr md_t      group_size = 16;
    std::vector<int8_t> packed, reordered;
    if (!reorderPackedB(packed, reordered, group_size)) {
        GTEST_SKIP() << "bf16u4 reorder not supported on this processor";
    }

    std::vector<bfloat16> A(static_cast<size_t>(kM) * kK, 0);
    std::vector<float>    C(static_cast<size_t>(kM) * kN, 0.0f);

    WoqSetup setup(DLP_PARAM_DIM_PER_CHANNEL, kN, true,
                   DLP_PARAM_DIM_PER_CHANNEL, kN, DLP_S8);
    setup.b_quant.group_size = group_size;
    callU4Of32('R', 'N', 'r', A.data(),
               reinterpret_cast<const uint8_t*>(reordered.data()), C.data(),
               &setup.md);
    EXPECT_EQ(setup.md.error_hndl.error_code, DLP_CLSC_SUCCESS);
}

TEST(WoqGranularity, BatchS4RejectsScalePerToken)
{
    SKIP_IF_NO_ISA_S4();
    std::vector<bfloat16> A;
    std::vector<int8_t>   B;
    std::vector<float>    C;
    fillGemmBufs(A, B, C);

    WoqSetup bad(DLP_PARAM_DIM_PER_TOKEN, kN, false);
    callBatchS4Of32('R', 'N', 'r', A.data(), B.data(), C.data(), &bad.md);
    EXPECT_EQ(bad.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST(WoqGranularity, BatchU4RejectsScalePerToken)
{
    SKIP_IF_NO_ISA_U4();
    std::vector<bfloat16> A;
    std::vector<int8_t>   B;
    std::vector<float>    C;
    fillGemmBufs(A, B, C);

    WoqSetup bad(DLP_PARAM_DIM_PER_TOKEN, kN, true, DLP_PARAM_DIM_PER_CHANNEL,
                 kN, DLP_S8);
    callBatchU4Of32('R', 'N', 'r', A.data(),
                    reinterpret_cast<const uint8_t*>(B.data()), C.data(),
                    &bad.md);
    EXPECT_EQ(bad.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST(WoqGranularity, BatchU4RejectsZpPerToken)
{
    SKIP_IF_NO_ISA_U4();
    std::vector<bfloat16> A;
    std::vector<int8_t>   B;
    std::vector<float>    C;
    fillGemmBufs(A, B, C);

    WoqSetup bad(DLP_PARAM_DIM_PER_CHANNEL, kN, true, DLP_PARAM_DIM_PER_TOKEN,
                 kN, DLP_S8);
    callBatchU4Of32('R', 'N', 'r', A.data(),
                    reinterpret_cast<const uint8_t*>(B.data()), C.data(),
                    &bad.md);
    EXPECT_EQ(bad.md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

// The reorder must not write beyond the size reported for the reorder buffer,
// including odd k where k is the contiguous dimension.
TEST(WoqGranularity, S4ReorderStaysWithinReportedSize)
{
    SKIP_IF_NO_ISA_S4();
    constexpr size_t  kGuardBytes = 64;
    constexpr uint8_t kSentinel   = 0xA5;
    const char        orders[]    = { 'R', 'C' };
    const char        transes[]   = { 'N', 'T' };
    const md_t        ns[]        = { 1, 15, 16, 17, 70 };

    for (char order : orders) {
        for (char trans : transes) {
            for (md_t k = 1; k <= 9; ++k) {
                for (md_t n : ns) {
                    const bool kContiguous = (order == 'R') == (trans == 'T');
                    const md_t ldb         = kContiguous ? k : n;

                    dlp_metadata_t md;
                    std::memset(&md, 0, sizeof(md));
                    const msz_t size = aocl_get_reorder_buf_size_bf16s4f32of32(
                        order, trans, 'B', k, n, &md);
                    ASSERT_GT(size, 0u);

                    std::vector<int8_t> in(
                        static_cast<size_t>(k) * static_cast<size_t>(n) + 16);
                    for (size_t i = 0; i < in.size(); ++i) {
                        in[i] = static_cast<int8_t>(i * 37 + 11);
                    }
                    std::vector<uint8_t> out(size + kGuardBytes, kSentinel);

                    aocl_reorder_bf16s4f32of32(
                        order, trans, 'B', in.data(),
                        reinterpret_cast<int8_t*>(out.data()), k, n, ldb, &md);

                    for (size_t i = size; i < out.size(); ++i) {
                        ASSERT_EQ(out[i], kSentinel)
                            << "write past reorder size at byte " << i
                            << " order=" << order << " trans=" << trans
                            << " k=" << k << " n=" << n;
                    }
                }
            }
        }
    }
}
