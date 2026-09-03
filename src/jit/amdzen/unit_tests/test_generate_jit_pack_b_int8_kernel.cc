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

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include "jit/amdzen/pack_b_amdzen_generator.hh"
#include "jit_generator_tests_utils.hh"

extern "C"
{
    void dlp_reorder_ref_packb(void*       pack_b,
                               const void* b,
                               md_t        elem_sz,
                               md_t        k_factor,
                               md_t        nc0,
                               md_t        kc0,
                               md_t        NR,
                               md_t        min_NR,
                               md_t        rs_b,
                               md_t        cs_b);
}

using namespace dlp::kernel_frame;
using namespace dlp::kernels;
using namespace dlp::jit;
using namespace amdzen::gen;
using namespace amdzen::utils;
using namespace test_jit_utils;

// Exact packed size for the cascade: full panels + base-width panel + a full
// 16-wide lt-block panel (storage is padded to 16 even when n%16 < 16).
static size_t
packedBufSizeINT8(md_t n, md_t k, md_t NR)
{
    md_t KC_updated = (k + 3) & ~static_cast<md_t>(3);
    md_t n_full     = (n / NR) * NR;
    md_t n_partial  = n - n_full;
    md_t baseW      = (n_partial / 16) * 16;
    md_t r          = n_partial - baseW;
    md_t widthTotal = n_full + baseW + (r > 0 ? 16 : 0);
    return static_cast<size_t>(widthTotal) * static_cast<size_t>(KC_updated);
}

class JitPackBINT8Test : public ::testing::Test
{
  protected:
    bool vnniSupported()
    {
        return !ArchBasedKernelTypes::getAllKernelTypesForU8S8().empty();
    }

    std::unique_ptr<packBJitGenerator> generate(md_t NR,
                                                bool colMajor,
                                                bool s8      = false,
                                                md_t kFactor = 4)
    {
        auto gen = std::make_unique<jitAmdZenPackBINT8>();

        packKernelInfo           packKI(NR, kFactor,
                                        kernelInstrPreference::avx512_zmm_favour,
                                        DataType::s8, DataType::s8, colMajor,
                                        /*accColSum=*/s8);
        packBJitGeneratorContext ctx(packKI);

        if ((*gen)(ctx) != jitGeneratorError::success) {
            return nullptr;
        }
        return gen;
    }

    // colMajor=false: row-major source (rs_src = N + ldPad, cs_src = 1).
    // colMajor=true : column-major source (rs_src = 1, cs_src = K + ldPad).
    // ldPad > 0 leaves 0x7F in the unused leading-dimension gap so a packer
    // that ignores cs_src / rs_src and assumes a tight matrix fails.
    void runCorrectness(md_t NR, md_t N, md_t K, bool colMajor, md_t ldPad = 0)
    {
        auto gen = generate(NR, colMajor);
        ASSERT_NE(gen, nullptr) << "generation failed NR=" << NR
                                << (colMajor ? " colMajor" : " rowMajor");

        const md_t rs_src = colMajor ? 1 : (N + ldPad);
        const md_t cs_src = colMajor ? (K + ldPad) : 1;

        const size_t srcCount =
            colMajor ? static_cast<size_t>(cs_src) * static_cast<size_t>(N)
                     : static_cast<size_t>(rs_src) * static_cast<size_t>(K);
        const size_t dstCount = packedBufSizeINT8(N, K, NR);

        std::vector<int8_t> srcBuf(srcCount, static_cast<int8_t>(0x7F));
        for (md_t ki = 0; ki < K; ++ki) {
            for (md_t ni = 0; ni < N; ++ni) {
                srcBuf[static_cast<size_t>(ki) * static_cast<size_t>(rs_src)
                       + static_cast<size_t>(ni)
                             * static_cast<size_t>(cs_src)] =
                    static_cast<int8_t>((ki * 13 + ni * 7 + 3) & 0xFF);
            }
        }

        std::vector<int8_t> refBuf(dstCount, 0);
        dlp_reorder_ref_packb(refBuf.data(), srcBuf.data(), sizeof(int8_t), 4,
                              N, K, NR, 16, rs_src, cs_src);

        constexpr size_t    kCanary = 64;
        std::vector<int8_t> dstBuf(dstCount + kCanary,
                                   static_cast<int8_t>(0x5A));

        packBParams params(srcBuf.data(), dstBuf.data(), N, K, rs_src, cs_src);
        ASSERT_EQ(gen->executeKernel(&params), kernelError::success);

        std::string label =
            std::string(colMajor ? "colMajor " : "rowMajor ")
            + "NR=" + std::to_string(NR) + " N=" + std::to_string(N)
            + " K=" + std::to_string(K) + " ldPad=" + std::to_string(ldPad);

        int mismatches = 0;
        for (size_t i = 0; i < dstCount && mismatches < 16; ++i) {
            if (dstBuf[i] != refBuf[i]) {
                ++mismatches;
                ADD_FAILURE() << label << " mismatch at " << i
                              << ": jit=" << static_cast<int>(dstBuf[i])
                              << " ref=" << static_cast<int>(refBuf[i]);
            }
        }
        EXPECT_TRUE(std::equal(refBuf.begin(), refBuf.end(), dstBuf.begin()))
            << label;
        EXPECT_TRUE(std::all_of(
            dstBuf.begin() + dstCount, dstBuf.end(),
            [](int8_t b) { return b == static_cast<int8_t>(0x5A); }))
            << label << " wrote past packed size";
        EXPECT_EQ(params.rs_dst, NR * 4);
        EXPECT_EQ(params.cs_dst, 64);
    }

    void runSameLogical(md_t NR, md_t N, md_t K)
    {
        auto genRow = generate(NR, /*colMajor=*/false);
        auto genCol = generate(NR, /*colMajor=*/true);
        ASSERT_NE(genRow, nullptr) << "NR=" << NR;
        ASSERT_NE(genCol, nullptr) << "NR=" << NR;

        std::vector<int8_t> rowSrc(static_cast<size_t>(K)
                                   * static_cast<size_t>(N));
        std::vector<int8_t> colSrc(rowSrc.size());
        for (md_t ki = 0; ki < K; ++ki) {
            for (md_t ni = 0; ni < N; ++ni) {
                const int8_t v =
                    static_cast<int8_t>((ki * 13 + ni * 7 + 3) & 0xFF);
                rowSrc[static_cast<size_t>(ki) * static_cast<size_t>(N)
                       + static_cast<size_t>(ni)]                          = v;
                colSrc[static_cast<size_t>(ki)
                       + static_cast<size_t>(ni) * static_cast<size_t>(K)] = v;
            }
        }

        const size_t        dstCount = packedBufSizeINT8(N, K, NR);
        std::vector<int8_t> dstRow(dstCount, static_cast<int8_t>(0x5A));
        std::vector<int8_t> dstCol(dstCount, static_cast<int8_t>(0xA5));

        packBParams pRow(rowSrc.data(), dstRow.data(), N, K, N, 1);
        packBParams pCol(colSrc.data(), dstCol.data(), N, K, 1, K);
        ASSERT_EQ(genRow->executeKernel(&pRow), kernelError::success);
        ASSERT_EQ(genCol->executeKernel(&pCol), kernelError::success);
        EXPECT_EQ(dstRow, dstCol) << "NR=" << NR << " N=" << N << " K=" << K;
        EXPECT_EQ(pRow.rs_dst, NR * 4);
        EXPECT_EQ(pCol.rs_dst, NR * 4);
        EXPECT_EQ(pRow.cs_dst, 64);
        EXPECT_EQ(pCol.cs_dst, 64);
    }

    // Fused s8s8 col-sum: packed bytes still match the generic reference,
    // and col_sum[n] += 128 * sum_k B[k,n] for live n only.
    void runColSumCorrectness(md_t    NR,
                              md_t    N,
                              md_t    K,
                              bool    colMajor,
                              md_t    ldPad = 0,
                              int32_t bias  = 0)
    {
        auto gen = generate(NR, colMajor, /*s8=*/true);
        ASSERT_NE(gen, nullptr) << "generation failed NR=" << NR
                                << (colMajor ? " colMajor" : " rowMajor");

        const md_t rs_src = colMajor ? 1 : (N + ldPad);
        const md_t cs_src = colMajor ? (K + ldPad) : 1;

        const size_t srcCount =
            colMajor ? static_cast<size_t>(cs_src) * static_cast<size_t>(N)
                     : static_cast<size_t>(rs_src) * static_cast<size_t>(K);
        const size_t dstCount = packedBufSizeINT8(N, K, NR);

        std::vector<int8_t> srcBuf(srcCount, static_cast<int8_t>(0x7F));
        for (md_t ki = 0; ki < K; ++ki) {
            for (md_t ni = 0; ni < N; ++ni) {
                srcBuf[static_cast<size_t>(ki) * static_cast<size_t>(rs_src)
                       + static_cast<size_t>(ni)
                             * static_cast<size_t>(cs_src)] =
                    static_cast<int8_t>((ki * 13 + ni * 7 + 3) & 0xFF);
            }
        }

        std::vector<int8_t>  refBuf(dstCount, 0);
        std::vector<int32_t> refSum(static_cast<size_t>(N), bias);
        dlp_reorder_ref_packb(refBuf.data(), srcBuf.data(), sizeof(int8_t), 4,
                              N, K, NR, 16, rs_src, cs_src);
        for (md_t ni = 0; ni < N; ++ni) {
            int32_t sum = 0;
            for (md_t ki = 0; ki < K; ++ki) {
                sum +=
                    srcBuf[static_cast<size_t>(ki) * static_cast<size_t>(rs_src)
                           + static_cast<size_t>(ni)
                                 * static_cast<size_t>(cs_src)];
            }
            refSum[static_cast<size_t>(ni)] += sum * 128;
        }

        constexpr size_t     kCanary = 16;
        std::vector<int8_t>  dstBuf(dstCount + 64, static_cast<int8_t>(0x5A));
        std::vector<int32_t> jitSum(static_cast<size_t>(N) + kCanary,
                                    0x7F7F7F7F);
        for (md_t ni = 0; ni < N; ++ni) {
            jitSum[static_cast<size_t>(ni)] = bias;
        }

        packBParams params(srcBuf.data(), dstBuf.data(), N, K, rs_src, cs_src);
        params.col_sum = jitSum.data();
        ASSERT_EQ(gen->executeKernel(&params), kernelError::success);

        std::string label =
            std::string(colMajor ? "colMajor " : "rowMajor ")
            + "NR=" + std::to_string(NR) + " N=" + std::to_string(N)
            + " K=" + std::to_string(K) + " ldPad=" + std::to_string(ldPad)
            + " bias=" + std::to_string(bias);

        EXPECT_TRUE(std::equal(refBuf.begin(), refBuf.end(), dstBuf.begin()))
            << label << " packed bytes";
        for (md_t ni = 0; ni < N; ++ni) {
            EXPECT_EQ(jitSum[static_cast<size_t>(ni)],
                      refSum[static_cast<size_t>(ni)])
                << label << " col_sum[" << ni << "]";
        }
        for (size_t i = static_cast<size_t>(N); i < jitSum.size(); ++i) {
            EXPECT_EQ(jitSum[i], 0x7F7F7F7F)
                << label << " wrote padded col_sum lane " << i;
        }
    }
};

TEST_F(JitPackBINT8Test, GenerateValidNR_RowMajor)
{
    if (!vnniSupported())
        GTEST_SKIP() << "AVX-512 VNNI not supported";

    for (md_t NR : { 16, 32, 48, 64, 80, 96, 112, 128, 144, 160 }) {
        SCOPED_TRACE("NR=" + std::to_string(NR));
        EXPECT_NE(generate(NR, false), nullptr);
    }
}

TEST_F(JitPackBINT8Test, InvalidNR_ReturnsError)
{
    if (!vnniSupported())
        GTEST_SKIP() << "AVX-512 VNNI not supported";

    for (md_t NR : { 0, 24, -1, 1, 8, 15, 17 }) {
        SCOPED_TRACE("NR=" + std::to_string(NR));
        EXPECT_EQ(generate(NR, false), nullptr);
        EXPECT_EQ(generate(NR, true), nullptr);
    }
}

TEST_F(JitPackBINT8Test, InvalidKFactor_ReturnsError)
{
    if (!vnniSupported())
        GTEST_SKIP() << "AVX-512 VNNI not supported";

    for (md_t kFactor : { 1, 2, 8 }) {
        SCOPED_TRACE("kFactor=" + std::to_string(kFactor));
        EXPECT_EQ(
            generate(/*NR=*/64, /*colMajor=*/false, /*s8=*/false, kFactor),
            nullptr);
    }
}

TEST_F(JitPackBINT8Test, S8NRAboveAccumulatorCapacity_ReturnsError)
{
    if (!vnniSupported())
        GTEST_SKIP() << "AVX-512 VNNI not supported";

    EXPECT_EQ(generate(/*NR=*/144, /*colMajor=*/false, /*s8=*/true), nullptr);
}

TEST_F(JitPackBINT8Test, GenerateValidNR_ColMajor)
{
    if (!vnniSupported())
        GTEST_SKIP() << "AVX-512 VNNI not supported";

    for (md_t NR : { 16, 32, 48, 64, 80, 96, 112, 128, 144, 160 }) {
        SCOPED_TRACE("NR=" + std::to_string(NR));
        EXPECT_NE(generate(NR, true), nullptr);
    }
}

TEST_F(JitPackBINT8Test, FullPanelsAndFringe_VsGenericReference)
{
    if (!vnniSupported())
        GTEST_SKIP() << "AVX-512 VNNI not supported";

    const md_t kVals[] = { 1, 2, 3, 4, 5, 7, 8, 16, 32, 63, 64, 65, 67, 96 };

    for (md_t NR : { 16, 32, 48, 64, 80, 96, 112, 128, 144, 160 }) {
        const std::vector<md_t> nVals = { 1,      16,        NR,      NR + 1,
                                          NR + 7, NR + 15,   NR + 16, NR + 17,
                                          2 * NR, 2 * NR + 3 };
        for (md_t N : nVals) {
            for (md_t K : kVals) {
                SCOPED_TRACE("NR=" + std::to_string(NR) + " N="
                             + std::to_string(N) + " K=" + std::to_string(K));
                runCorrectness(NR, N, K, /*colMajor=*/false);
            }
        }
    }
}

// Column-major 16x16 dword transpose. K values cover a 64-K tile, the
// masked K%64 fringe, and K%4 in {1,2,3} (zero-padded K-quad). N values
// cover every ladder rung: lt16, exact 16-multiple leftover, full panels.
TEST_F(JitPackBINT8Test, FullPanelsAndFringe_VsGenericReference_ColMajor)
{
    if (!vnniSupported())
        GTEST_SKIP() << "AVX-512 VNNI not supported";

    const md_t kVals[] = { 1, 2, 3, 4, 5, 7, 16, 32, 63, 64, 65, 67, 96 };

    for (md_t NR : { 16, 32, 48, 64, 80, 96, 112, 128, 144, 160 }) {
        const std::vector<md_t> nVals = { 1,      16,        NR,      NR + 1,
                                          NR + 7, NR + 15,   NR + 16, NR + 17,
                                          2 * NR, 2 * NR + 3 };
        for (md_t N : nVals) {
            for (md_t K : kVals) {
                SCOPED_TRACE("NR=" + std::to_string(NR) + " N="
                             + std::to_string(N) + " K=" + std::to_string(K));
                runCorrectness(NR, N, K, /*colMajor=*/true);
            }
        }
    }
}

TEST_F(JitPackBINT8Test, SameLogicalB_RowAndColMajorMatch)
{
    if (!vnniSupported())
        GTEST_SKIP() << "AVX-512 VNNI not supported";

    const md_t kVals[] = { 1, 3, 4, 17, 63, 64, 65, 67 };

    for (md_t NR : { 16, 32, 48, 64, 80, 96, 112, 128, 144, 160 }) {
        const std::vector<md_t> nVals = {
            1, 16, NR, NR + 7, NR + 16, 2 * NR + 3
        };
        for (md_t N : nVals) {
            for (md_t K : kVals) {
                SCOPED_TRACE("NR=" + std::to_string(NR) + " N="
                             + std::to_string(N) + " K=" + std::to_string(K));
                runSameLogical(NR, N, K);
            }
        }
    }
}

TEST_F(JitPackBINT8Test, PaddedLeadingDim_RowAndColMajor)
{
    if (!vnniSupported())
        GTEST_SKIP() << "AVX-512 VNNI not supported";

    constexpr md_t ldPad   = 11;
    const md_t     kVals[] = { 3, 17, 65, 67 };

    for (md_t NR : { 16, 32, 48, 64, 80, 96, 112, 128, 144, 160 }) {
        for (md_t N : { static_cast<md_t>(1), NR, NR + 7 }) {
            for (md_t K : kVals) {
                SCOPED_TRACE("NR=" + std::to_string(NR) + " N="
                             + std::to_string(N) + " K=" + std::to_string(K));
                runCorrectness(NR, N, K, /*colMajor=*/false, ldPad);
                runCorrectness(NR, N, K, /*colMajor=*/true, ldPad);
            }
        }
    }
}

TEST_F(JitPackBINT8Test, ZeroN_NoOp)
{
    if (!vnniSupported())
        GTEST_SKIP() << "AVX-512 VNNI not supported";

    md_t NR  = 64;
    auto gen = generate(NR, false);
    ASSERT_NE(gen, nullptr);

    int8_t      src[1] = { 1 };
    int8_t      dst[1] = { static_cast<int8_t>(0xA5) };
    packBParams params(src, dst, /*n=*/0, /*k=*/16, NR, 1);

    EXPECT_EQ(gen->executeKernel(&params), kernelError::success);
    EXPECT_EQ(dst[0], static_cast<int8_t>(0xA5))
        << "N=0 must not touch the destination";
}

TEST_F(JitPackBINT8Test, ColSum_VsGenericReference_RowAndColMajor)
{
    if (!vnniSupported())
        GTEST_SKIP() << "AVX-512 VNNI not supported";

    const md_t kVals[] = { 1, 3, 4, 17, 63, 64, 65, 67 };

    for (md_t NR : { 16, 32, 48, 64, 80, 96, 112, 128 }) {
        const std::vector<md_t> nVals = { 1,      16,      NR,        NR + 1,
                                          NR + 7, NR + 16, 2 * NR + 3 };
        for (md_t N : nVals) {
            for (md_t K : kVals) {
                SCOPED_TRACE("NR=" + std::to_string(NR) + " N="
                             + std::to_string(N) + " K=" + std::to_string(K));
                runColSumCorrectness(NR, N, K, /*colMajor=*/false);
                runColSumCorrectness(NR, N, K, /*colMajor=*/true);
            }
        }
    }
}

TEST_F(JitPackBINT8Test, ColSum_AccumulateAndPaddedLd)
{
    if (!vnniSupported())
        GTEST_SKIP() << "AVX-512 VNNI not supported";

    constexpr int32_t bias = 7;
    for (md_t NR : { 16, 32, 48, 64, 80, 96, 112, 128 }) {
        for (bool colMajor : { false, true }) {
            SCOPED_TRACE(std::string(colMajor ? "colMajor" : "rowMajor")
                         + " NR=" + std::to_string(NR));
            runColSumCorrectness(NR, NR + 7, 65, colMajor, /*ldPad=*/11, bias);
            runColSumCorrectness(NR, 1, 3, colMajor, /*ldPad=*/0, bias);
        }
    }
}

TEST_F(JitPackBINT8Test, U8GeneratedKernelDoesNotWriteColumnSum)
{
    if (!vnniSupported())
        GTEST_SKIP() << "AVX-512 VNNI not supported";

    constexpr int32_t kCanary = 0x5A5A5A5A;
    for (bool colMajor : { false, true }) {
        constexpr md_t NR = 64;
        constexpr md_t N  = NR + 7;
        constexpr md_t K  = 17;

        auto gen = generate(NR, colMajor);
        ASSERT_NE(gen, nullptr);

        const md_t   rs_src = colMajor ? 1 : N;
        const md_t   cs_src = colMajor ? K : 1;
        const size_t srcCount =
            colMajor ? static_cast<size_t>(cs_src) * static_cast<size_t>(N)
                     : static_cast<size_t>(rs_src) * static_cast<size_t>(K);

        std::vector<int8_t>  src(srcCount, 1);
        std::vector<int8_t>  dst(packedBufSizeINT8(N, K, NR), 0);
        std::vector<int32_t> colSum(static_cast<size_t>(N) + 16, kCanary);

        packBParams params(src.data(), dst.data(), N, K, rs_src, cs_src);
        params.col_sum = colSum.data();
        ASSERT_EQ(gen->executeKernel(&params), kernelError::success);
        EXPECT_TRUE(std::all_of(colSum.begin(), colSum.end(),
                                [](int32_t v) { return v == kCanary; }));
    }
}

TEST_F(JitPackBINT8Test, GeneratorKeepsSharedDatatypeCapabilities)
{
    jitAmdZenPackBINT8 allGen;
    ASSERT_EQ(allGen.getKernelDatatypes().size(), 12u);

    auto u8Gen = generate(/*NR=*/64, /*colMajor=*/false, /*s8=*/false);
    auto s8Gen = generate(/*NR=*/64, /*colMajor=*/false, /*s8=*/true);
    ASSERT_NE(u8Gen, nullptr);
    ASSERT_NE(s8Gen, nullptr);
    EXPECT_EQ(u8Gen->getKernelDatatypes().size(), 12u);
    EXPECT_EQ(s8Gen->getKernelDatatypes().size(), 12u);
}

TEST_F(JitPackBINT8Test, S8ExecuteRejectsNullColumnSum)
{
    if (!vnniSupported())
        GTEST_SKIP() << "AVX-512 VNNI not supported";

    auto gen = generate(/*NR=*/64, /*colMajor=*/false, /*s8=*/true);
    ASSERT_NE(gen, nullptr);

    int8_t      src[64] = {};
    int8_t      dst[64] = {};
    packBParams params(src, dst, /*n=*/1, /*k=*/1, /*rs_src=*/64,
                       /*cs_src=*/1);
    EXPECT_EQ(gen->executeKernel(&params), kernelError::badInputParams);
}

TEST_F(JitPackBINT8Test, ExecuteBeforeGenerate_ReturnsError)
{
    auto        gen    = std::make_unique<jitAmdZenPackBINT8>();
    int8_t      src[4] = {};
    int8_t      dst[4] = {};
    packBParams params(src, dst, 4, 4, 4, 1);
    EXPECT_EQ(gen->executeKernel(&params), kernelError::error);
}
