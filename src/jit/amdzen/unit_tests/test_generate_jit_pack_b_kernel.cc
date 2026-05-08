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

#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "jit/amdzen/pack_b_amdzen_generator.hh"
#include "jit_generator_tests_utils.hh"

using namespace dlp::kernel_frame;
using namespace dlp::kernels;
using namespace dlp::jit;
using namespace amdzen::gen;
using namespace amdzen::utils;
using namespace test_jit_utils;

// ============================================================================
// Valid NR lists per ISA
//
// Row-major constraints (f32_pack_b_generator.cc::allocateReg):
//   - NR > 0 and NR % simdWidth == 0
//   - numVecLoads = NR/simdWidth <= numRegs (full-NR kernel)
//   - lt-NR (mask) kernel:
//       AVX512: numVecLoads <= 7 (one opmask per block, k1..k7)
//       AVX2:   numVecLoads * 2 <= numRegs (data + mask YMMs)
//
// Column-major constraints (f32_col_major_pack_b_generator.cc):
//   - NR > 0 and NR % simdWidth == 0
//
// Row-major lt-NR maxima only (per-block mask/register budget; these do NOT
// apply to column-major packing):
//   AVX512: simdWidth=16, max lt-NR NR = 7*16 = 112
//   AVX2:   simdWidth=8,  max lt-NR NR = 8*8  = 64 (8 data + 8 mask YMMs)
// ============================================================================

static std::vector<md_t>
getValidNRs(kernelInstrType kType)
{
    if (kType == kernelInstrType::avx512_zmm_32_reg) {
        return { 16, 32, 48, 64, 80, 96, 112 };
    } else {
        return { 8, 16, 24, 32, 48, 64 };
    }
}

// NR values that should fail generation for both row-major and col-major:
//   - 0
//   - Not a multiple of simdWidth
static std::vector<md_t>
getInvalidNRs(kernelInstrType kType)
{
    if (kType == kernelInstrType::avx512_zmm_32_reg) {
        return { 0, 1, 7, 15, 17, 33, 63, 65 };
    } else {
        return { 0, 1, 3, 7, 9, 15, 17, 33 };
    }
}

// NR values that exceed the lt-NR per-block mask register budget.
// These fail for row-major (which needs numVecLoads masks) but succeed for
// col-major (which only uses a single mask in slot 0).
static std::vector<md_t>
getRowMajorOnlyInvalidNRs(kernelInstrType kType)
{
    if (kType == kernelInstrType::avx512_zmm_32_reg) {
        return { 128 };
    } else {
        return { 72, 80, 128 };
    }
}

// ============================================================================
// Reference (scalar) Pack B implementations
// ============================================================================

static void
referencePackBRowMajor(
    const float* src, float* dst, md_t n, md_t k, md_t NR, md_t ld_src)
{
    md_t n_full   = (n / NR) * NR;
    md_t n_remain = n - n_full;

    float* dstPtr = dst;

    for (md_t jj = 0; jj < n_full; jj += NR) {
        for (md_t i = 0; i < k; ++i) {
            for (md_t j = 0; j < NR; ++j) {
                *dstPtr++ = src[i * ld_src + (jj + j)];
            }
        }
    }

    if (n_remain > 0) {
        for (md_t i = 0; i < k; ++i) {
            for (md_t j = 0; j < NR; ++j) {
                if (j < n_remain) {
                    *dstPtr++ = src[i * ld_src + (n_full + j)];
                } else {
                    *dstPtr++ = 0.0f;
                }
            }
        }
    }
}

static void
referencePackBColMajor(
    const float* src, float* dst, md_t n, md_t k, md_t NR, md_t ld_src)
{
    // Column-major source: element B(k_idx, n_idx) = src[n_idx * ld_src +
    // k_idx]
    md_t n_full   = (n / NR) * NR;
    md_t n_remain = n - n_full;

    float* dstPtr = dst;

    for (md_t jj = 0; jj < n_full; jj += NR) {
        for (md_t i = 0; i < k; ++i) {
            for (md_t j = 0; j < NR; ++j) {
                *dstPtr++ = src[(jj + j) * ld_src + i];
            }
        }
    }

    if (n_remain > 0) {
        for (md_t i = 0; i < k; ++i) {
            for (md_t j = 0; j < NR; ++j) {
                if (j < n_remain) {
                    *dstPtr++ = src[(n_full + j) * ld_src + i];
                } else {
                    *dstPtr++ = 0.0f;
                }
            }
        }
    }
}

static size_t
packedBufSizeFloats(md_t n, md_t k, md_t NR)
{
    md_t n_panels = (n + NR - 1) / NR;
    return static_cast<size_t>(n_panels) * NR * k;
}

// ============================================================================
// ISA helpers
// ============================================================================

// Pack B only supports avx512_zmm and avx2_ymm (not avx512_ymm).
static std::vector<std::pair<kernelInstrType, kernelInstrPreference>>
getPackBSupportedISAs()
{
    std::vector<std::pair<kernelInstrType, kernelInstrPreference>> result;

    auto allKTypes = ArchBasedKernelTypes::getAllKernelTypesForF32();
    for (auto kType : allKTypes) {
        if (kType == kernelInstrType::avx512_zmm_32_reg) {
            result.emplace_back(kType,
                                kernelInstrPreference::avx512_zmm_favour);
        } else if (kType == kernelInstrType::avx2_ymm_16_reg) {
            result.emplace_back(kType, kernelInstrPreference::avx2_ymm_favour);
        }
    }
    return result;
}

static md_t
defaultNRForISA(kernelInstrType kType)
{
    return (kType == kernelInstrType::avx512_zmm_32_reg) ? 64 : 16;
}

// ============================================================================
// Test Fixture
// ============================================================================

class JitPackBTest : public JitGeneratorTestBase
{
  protected:
    struct PackBConfig
    {
        md_t                  NR;
        md_t                  k_factor;
        kernelInstrPreference kInstPref;
        bool                  isColMajor;
    };

    std::unique_ptr<jitAmdZenPackBFP32> generatePackBKernel(
        const PackBConfig& cfg)
    {
        auto gen = std::make_unique<jitAmdZenPackBFP32>();

        kernelInfo     dummyKI;
        packKernelInfo packKI(cfg.NR, cfg.k_factor, cfg.kInstPref,
                              DataType::f32, DataType::f32, cfg.isColMajor);

        jitGeneratorContext ctx(dummyKI, packKI);
        auto                err = (*gen)(ctx);

        if (err != jitGeneratorError::success) {
            return nullptr;
        }
        return gen;
    }

    void fillSource(float* buf, size_t count)
    {
        for (size_t i = 0; i < count; ++i) {
            buf[i] = static_cast<float>(i + 1);
        }
    }

    int compareBuffers(const float*       jitOut,
                       const float*       refOut,
                       size_t             count,
                       const std::string& label)
    {
        int mismatches = 0;
        for (size_t i = 0; i < count; ++i) {
            if (jitOut[i] != refOut[i]) {
                if (mismatches < 10) {
                    std::cerr << "  [" << label << "] mismatch at index " << i
                              << ": jit=" << jitOut[i] << " ref=" << refOut[i]
                              << std::endl;
                }
                ++mismatches;
            }
        }
        return mismatches;
    }

    kernelError executePackB(jitAmdZenPackBFP32* gen,
                             void*               src,
                             void*               dst,
                             md_t                n,
                             md_t                k,
                             md_t                rs_src,
                             md_t                cs_src)
    {
        packBParams params(src, dst, n, k, rs_src, cs_src);
        return gen->executeKernel(&params);
    }

    void runCorrectnessTest(const PackBConfig& cfg, md_t N, md_t K)
    {
        auto gen = generatePackBKernel(cfg);
        ASSERT_NE(gen, nullptr) << "Kernel generation failed for NR=" << cfg.NR
                                << " colMajor=" << cfg.isColMajor;

        md_t ld_src = cfg.isColMajor ? K : N;

        size_t srcCount = static_cast<size_t>(N) * K;
        size_t dstCount = packedBufSizeFloats(N, K, cfg.NR);

        std::vector<float> srcBuf(srcCount);
        fillSource(srcBuf.data(), srcCount);

        std::vector<float> refBuf(dstCount, 0.0f);
        if (cfg.isColMajor) {
            referencePackBColMajor(srcBuf.data(), refBuf.data(), N, K, cfg.NR,
                                   ld_src);
        } else {
            referencePackBRowMajor(srcBuf.data(), refBuf.data(), N, K, cfg.NR,
                                   ld_src);
        }

        std::vector<float> dstBuf(dstCount, 0.0f);

        md_t rs_src = cfg.isColMajor ? 1 : ld_src;
        md_t cs_src = cfg.isColMajor ? ld_src : 1;

        auto err = executePackB(gen.get(), srcBuf.data(), dstBuf.data(), N, K,
                                rs_src, cs_src);
        ASSERT_EQ(err, kernelError::success) << "executeKernel failed";

        std::string label =
            "NR=" + std::to_string(cfg.NR) + " N=" + std::to_string(N)
            + " K=" + std::to_string(K) + (cfg.isColMajor ? " col" : " row");

        int mismatches =
            compareBuffers(dstBuf.data(), refBuf.data(), dstCount, label);
        EXPECT_EQ(mismatches, 0)
            << label << ": " << mismatches << " mismatches out of " << dstCount
            << " elements";
    }
};

// ============================================================================
// 1. Generation Success Tests — all valid NRs
// ============================================================================

TEST_F(JitPackBTest, GenerateValidNR_RowMajor)
{
    auto isas = getPackBSupportedISAs();
    if (isas.empty()) {
        GTEST_SKIP() << "No pack-B-capable ISA detected";
    }

    for (auto [kType, kInstPref] : isas) {
        auto validNRs = getValidNRs(kType);

        for (md_t NR : validNRs) {
            SCOPED_TRACE(kTypeToArchString(kType) + " NR=" + std::to_string(NR)
                         + " row");

            PackBConfig cfg{ NR, 1, kInstPref, false };
            auto        gen = generatePackBKernel(cfg);
            EXPECT_NE(gen, nullptr) << "Row-major generation failed for "
                                    << kTypeToArchString(kType) << " NR=" << NR;
        }
    }
}

TEST_F(JitPackBTest, GenerateValidNR_ColMajor)
{
    auto isas = getPackBSupportedISAs();
    if (isas.empty()) {
        GTEST_SKIP() << "No pack-B-capable ISA detected";
    }

    for (auto [kType, kInstPref] : isas) {
        auto validNRs = getValidNRs(kType);

        for (md_t NR : validNRs) {
            SCOPED_TRACE(kTypeToArchString(kType) + " NR=" + std::to_string(NR)
                         + " col");

            PackBConfig cfg{ NR, 1, kInstPref, true };
            auto        gen = generatePackBKernel(cfg);
            EXPECT_NE(gen, nullptr) << "Col-major generation failed for "
                                    << kTypeToArchString(kType) << " NR=" << NR;
        }
    }
}

// ============================================================================
// 2. Generation Failure Tests — invalid NRs
// ============================================================================

TEST_F(JitPackBTest, GenerateInvalidNR_ReturnsError)
{
    auto isas = getPackBSupportedISAs();
    if (isas.empty()) {
        GTEST_SKIP() << "No pack-B-capable ISA detected";
    }

    for (auto [kType, kInstPref] : isas) {
        auto invalidNRs = getInvalidNRs(kType);

        for (md_t NR : invalidNRs) {
            for (bool colMajor : { false, true }) {
                std::string testName = kTypeToArchString(kType)
                                       + " NR=" + std::to_string(NR)
                                       + (colMajor ? " col" : " row");

                SCOPED_TRACE(testName);

                auto gen = std::make_unique<jitAmdZenPackBFP32>();

                kernelInfo          dummyKI;
                packKernelInfo      packKI(NR, 1, kInstPref, DataType::f32,
                                           DataType::f32, colMajor);
                jitGeneratorContext ctx(dummyKI, packKI);

                auto result = CrashIsolation::runIsolated(
                    [&]() { return static_cast<int>((*gen)(ctx)); });

                if (result.crashed) {
                    FAIL() << testName << " crashed with "
                           << CrashIsolation::signalName(result.crashSignal)
                           << " instead of returning error";
                } else {
                    EXPECT_NE(static_cast<jitGeneratorError>(result.returnCode),
                              jitGeneratorError::success)
                        << testName << " should have failed";
                }
            }
        }
    }
}

TEST_F(JitPackBTest, GenerateInvalidNR_RowMajorOnly_ReturnsError)
{
    auto isas = getPackBSupportedISAs();
    if (isas.empty()) {
        GTEST_SKIP() << "No pack-B-capable ISA detected";
    }

    for (auto [kType, kInstPref] : isas) {
        auto invalidNRs = getRowMajorOnlyInvalidNRs(kType);

        for (md_t NR : invalidNRs) {
            std::string testName =
                kTypeToArchString(kType) + " NR=" + std::to_string(NR) + " row";
            SCOPED_TRACE(testName);

            PackBConfig cfg{ NR, 1, kInstPref, false };
            auto        gen = generatePackBKernel(cfg);
            EXPECT_EQ(gen, nullptr)
                << testName << " should have failed for row-major";
        }
    }
}

TEST_F(JitPackBTest, GenerateInvalidNR_RowMajorOnly_ColMajorSucceeds)
{
    auto isas = getPackBSupportedISAs();
    if (isas.empty()) {
        GTEST_SKIP() << "No pack-B-capable ISA detected";
    }

    for (auto [kType, kInstPref] : isas) {
        auto invalidNRs = getRowMajorOnlyInvalidNRs(kType);

        for (md_t NR : invalidNRs) {
            std::string testName =
                kTypeToArchString(kType) + " NR=" + std::to_string(NR) + " col";
            SCOPED_TRACE(testName);

            PackBConfig cfg{ NR, 1, kInstPref, true };
            auto        gen = generatePackBKernel(cfg);
            EXPECT_NE(gen, nullptr)
                << testName << " should succeed for col-major";
        }
    }
}

// ============================================================================
// 3. Execution Correctness Tests — various N, K, NR shapes
// ============================================================================

struct PackBTestParams
{
    md_t N;
    md_t K;
    bool colMajor;
};

class JitPackBCorrectnessTest
    : public JitPackBTest
    , public ::testing::WithParamInterface<PackBTestParams>
{};

TEST_P(JitPackBCorrectnessTest, VerifyAgainstReference)
{
    auto params = GetParam();

    auto isas = getPackBSupportedISAs();
    if (isas.empty()) {
        GTEST_SKIP() << "No pack-B-capable ISA detected";
    }

    for (auto [kType, kInstPref] : isas) {
        auto validNRs = getValidNRs(kType);

        for (md_t NR : validNRs) {
            SCOPED_TRACE(kTypeToArchString(kType) + " NR=" + std::to_string(NR)
                         + " N=" + std::to_string(params.N)
                         + " K=" + std::to_string(params.K)
                         + (params.colMajor ? " col" : " row"));

            PackBConfig cfg{ NR, 1, kInstPref, params.colMajor };
            runCorrectnessTest(cfg, params.N, params.K);
        }
    }
}

static std::string
packBTestName(
    const ::testing::TestParamInfo<JitPackBCorrectnessTest::ParamType>& info)
{
    auto& p = info.param;
    return "N" + std::to_string(p.N) + "_K" + std::to_string(p.K)
           + (p.colMajor ? "_col" : "_row");
}

INSTANTIATE_TEST_SUITE_P(
    PackBShapes,
    JitPackBCorrectnessTest,
    ::testing::Values(
        // Minimal cases
        PackBTestParams{ 1, 1, false },
        PackBTestParams{ 1, 1, true },
        PackBTestParams{ 2, 2, false },
        PackBTestParams{ 2, 2, true },

        // Small N, small K (pure fringe)
        PackBTestParams{ 3, 5, false },
        PackBTestParams{ 3, 5, true },
        PackBTestParams{ 7, 3, false },
        PackBTestParams{ 7, 3, true },

        // N < simdWidth
        PackBTestParams{ 5, 16, false },
        PackBTestParams{ 5, 16, true },

        // N = simdWidth (exact, no mask needed)
        PackBTestParams{ 8, 16, false },
        PackBTestParams{ 8, 16, true },
        PackBTestParams{ 16, 16, false },
        PackBTestParams{ 16, 16, true },

        // N > simdWidth but < NR (fringe panel only for avx512)
        PackBTestParams{ 13, 32, false },
        PackBTestParams{ 13, 32, true },
        PackBTestParams{ 45, 17, false },
        PackBTestParams{ 45, 17, true },

        // Exact NR=64 (no fringe for avx512; multiple full panels for avx2)
        PackBTestParams{ 64, 32, false },
        PackBTestParams{ 64, 32, true },

        // N > NR: multiple full panels + fringe
        PackBTestParams{ 65, 16, false },
        PackBTestParams{ 65, 16, true },
        PackBTestParams{ 100, 48, false },
        PackBTestParams{ 100, 48, true },
        PackBTestParams{ 128, 64, false },
        PackBTestParams{ 128, 64, true },
        PackBTestParams{ 256, 13, false },
        PackBTestParams{ 256, 13, true },

        // K-fringe edge cases (K not a multiple of simdWidth)
        PackBTestParams{ 64, 1, false },
        PackBTestParams{ 64, 1, true },
        PackBTestParams{ 64, 7, false },
        PackBTestParams{ 64, 7, true },
        PackBTestParams{ 64, 15, false },
        PackBTestParams{ 64, 15, true },
        PackBTestParams{ 64, 17, false },
        PackBTestParams{ 64, 17, true },

        // Combined N + K fringe
        PackBTestParams{ 6, 13, false },
        PackBTestParams{ 6, 13, true },
        PackBTestParams{ 33, 7, false },
        PackBTestParams{ 33, 7, true },
        PackBTestParams{ 129, 31, false },
        PackBTestParams{ 129, 31, true },

        // Larger realistic shapes
        PackBTestParams{ 256, 256, false },
        PackBTestParams{ 256, 256, true },
        PackBTestParams{ 768, 768, false },
        PackBTestParams{ 768, 768, true }),
    packBTestName);

// ============================================================================
// 4. Negative Tests (bad inputs to generation / execution)
// ============================================================================

TEST_F(JitPackBTest, Negative_NullPackKI)
{
    auto gen = std::make_unique<jitAmdZenPackBFP32>();

    kernelInfo          dummyKI;
    jitGeneratorContext ctx(dummyKI);

    auto err = (*gen)(ctx);
    EXPECT_EQ(err, jitGeneratorError::badKernelInfo)
        << "Expected badKernelInfo for null packKI";
}

TEST_F(JitPackBTest, Negative_InvalidKInstPref)
{
    auto gen = std::make_unique<jitAmdZenPackBFP32>();

    kernelInfo     dummyKI;
    packKernelInfo packKI(64, 1, kernelInstrPreference::none, DataType::f32,
                          DataType::f32, false);

    jitGeneratorContext ctx(dummyKI, packKI);
    auto                err = (*gen)(ctx);
    EXPECT_EQ(err, jitGeneratorError::notSupported)
        << "Expected notSupported for kInstPref=none";
}

TEST_F(JitPackBTest, Negative_ExecuteBeforeGenerate)
{
    auto gen = std::make_unique<jitAmdZenPackBFP32>();

    float       src[16] = {};
    float       dst[16] = {};
    packBParams params(src, dst, 4, 4, 4, 1);

    auto err = gen->executeKernel(&params);
    EXPECT_EQ(err, kernelError::error)
        << "Expected error when executing without generation";
}

TEST_F(JitPackBTest, Negative_ZeroN)
{
    auto isas = getPackBSupportedISAs();
    if (isas.empty()) {
        GTEST_SKIP() << "No pack-B-capable ISA detected";
    }

    auto [kType, kInstPref] = isas.front();
    md_t NR                 = defaultNRForISA(kType);

    PackBConfig cfg{ NR, 1, kInstPref, false };
    auto        gen = generatePackBKernel(cfg);
    ASSERT_NE(gen, nullptr);

    float       src[1] = { 1.0f };
    float       dst[1] = { -1.0f };
    packBParams params(src, dst, /*n=*/0, /*k=*/16, NR, 1);

    auto err = gen->executeKernel(&params);
    EXPECT_EQ(err, kernelError::success) << "N=0 should succeed (no-op)";
    EXPECT_EQ(dst[0], -1.0f) << "N=0 should not touch destination buffer";
}

// ============================================================================

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
