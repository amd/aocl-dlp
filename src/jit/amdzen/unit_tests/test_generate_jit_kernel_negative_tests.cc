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

#include <functional>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "jit_generator_tests_utils.hh"

using namespace amdzen::gen;
using namespace amdzen::codegen;
using namespace amdzen::utils;
using namespace test_jit_utils;

namespace {

// ============================================================================
// Datatype Configuration
// ============================================================================

enum class Datatype
{
    F32,
    BF16,
    U8S8,
    S8
};

std::string
datatypeToString(Datatype dt)
{
    switch (dt) {
        case Datatype::F32:
            return "F32";
        case Datatype::BF16:
            return "BF16";
        case Datatype::U8S8:
            return "U8S8";
        case Datatype::S8:
            return "S8";
    }
    return "UNKNOWN";
}

int
datatypeToDownscale(Datatype dt)
{
    switch (dt) {
        case Datatype::F32:
            return DLP_F32;
        case Datatype::BF16:
            return DLP_BF16;
        case Datatype::U8S8:
        case Datatype::S8:
            return DLP_S32;
    }
    return DLP_F32;
}

int
datatypeToPrefetch(Datatype dt)
{
    return (dt == Datatype::U8S8) ? 0 : 8;
}

std::string
datatypeSkipMessage(Datatype dt)
{
    switch (dt) {
        case Datatype::F32:
            return "F32 tests require AVX2 or AVX512 support";
        case Datatype::BF16:
            return "BF16 tests require AVX512_BF16 support";
        case Datatype::U8S8:
            return "U8S8 tests require AVX512_VNNI support";
        case Datatype::S8:
            return "S8 tests require AVX512_VNNI support";
    }
    return "Unsupported datatype";
}

// ============================================================================
// GEMM generatorParams Construction Helpers
// ============================================================================

using GemmParamBuilder =
    std::function<generatorParams(kernelInfo&, kernelInstrType)>;

// Corrupt one field on kernelInfo, then build generatorParams from ki's fields
GemmParamBuilder
corruptParam(std::function<void(kernelInfo&)> corrupt)
{
    return [corrupt](kernelInfo& ki, kernelInstrType kType) {
        corrupt(ki);
        return generatorParams(ki.mr, ki.nr, ki.k_unroll, ki.prefetch_c_dist,
                               ki.c_downscale, ki.kOpsArrSize,
                               ki.genLtKrnlForAvailFullKrnl, true, ki.invokeRD,
                               ki.alphaScalingType, ki.betaScalingType, kType);
    };
}

// Override numMaskRegs (and optionally useMask) in generatorParams
GemmParamBuilder
corruptNumMaskRegs(int numMaskRegs, bool useMask = false)
{
    return [numMaskRegs, useMask](kernelInfo& ki, kernelInstrType kType) {
        return generatorParams(ki.mr, ki.nr, ki.k_unroll, ki.prefetch_c_dist,
                               ki.c_downscale, numMaskRegs, useMask, true,
                               false, ki.alphaScalingType, ki.betaScalingType,
                               kType);
    };
}

// Override c_downscale to an invalid value
GemmParamBuilder
corruptCDownscale(int c_downscale)
{
    return [c_downscale](kernelInfo& ki, kernelInstrType kType) {
        return generatorParams(ki.mr, ki.nr, ki.k_unroll, ki.prefetch_c_dist,
                               c_downscale, 0, false, true, false,
                               ki.alphaScalingType, ki.betaScalingType, kType);
    };
}

// ============================================================================
// GEMV N=1 gemvN1GeneratorParams Construction Helpers
// ============================================================================

using GemvN1ParamBuilder =
    std::function<gemvN1GeneratorParams(kernelInfo&, kernelInstrType)>;

GemvN1ParamBuilder
corruptGemvN1MR(int mr_value)
{
    return [mr_value](kernelInfo& ki, kernelInstrType kType) {
        ki.mr = mr_value;
        return gemvN1GeneratorParams(ki.mr, 15, ki.c_downscale, true, true,
                                     true, true, storageFormat::colMajor,
                                     ki.alphaScalingType, ki.betaScalingType,
                                     kType);
    };
}

GemvN1ParamBuilder
corruptGemvN1CDownscale(int c_downscale)
{
    return [c_downscale](kernelInfo& ki, kernelInstrType kType) {
        return gemvN1GeneratorParams(ki.mr, 15, c_downscale, true, true, true,
                                     true, storageFormat::colMajor,
                                     ki.alphaScalingType, ki.betaScalingType,
                                     kType);
    };
}

// ============================================================================
// GEMV M=1 gemvM1GeneratorParams Construction Helpers
// ============================================================================

using GemvM1ParamBuilder =
    std::function<gemvM1GeneratorParams(kernelInfo&, kernelInstrType)>;

GemvM1ParamBuilder
corruptGemvM1Field(std::function<void(kernelInfo&)> corrupt)
{
    return [corrupt](kernelInfo& ki, kernelInstrType kType) {
        corrupt(ki);
        gemvM1GeneratorParams gp(ki.c_downscale, ki.nr, 63, ki.kc, ki.k_unroll,
                                 PACK, true, true, true, true,
                                 storageFormat::rowMajor, ki.alphaScalingType,
                                 ki.betaScalingType, kType);
        gp.nfringe_main = true;
        gp.nfringe_left = true;
        gp.N_LEFT_16    = 48;
        gp.N_LEFT_LT16  = 15;
        return gp;
    };
}

GemvM1ParamBuilder
corruptGemvM1CDownscale(int c_downscale)
{
    return [c_downscale](kernelInfo& ki, kernelInstrType kType) {
        gemvM1GeneratorParams gp(c_downscale, ki.nr, 63, ki.kc, ki.k_unroll,
                                 PACK, true, true, true, true,
                                 storageFormat::rowMajor, ki.alphaScalingType,
                                 ki.betaScalingType, kType);
        gp.nfringe_main = true;
        gp.nfringe_left = true;
        gp.N_LEFT_16    = 48;
        gp.N_LEFT_LT16  = 15;
        return gp;
    };
}

// ============================================================================
// Parameter Structs (one per kernel shape)
// ============================================================================

struct GemmNegativeParam
{
    Datatype         dtype;
    std::string      testCase;
    GemmParamBuilder buildParams;
};

struct GemvN1NegativeParam
{
    Datatype           dtype;
    std::string        testCase;
    GemvN1ParamBuilder buildParams;
};

struct GemvM1NegativeParam
{
    Datatype           dtype;
    std::string        testCase;
    GemvM1ParamBuilder buildParams;
};

} // anonymous namespace

// ============================================================================
// Shared Base Fixture for Negative Tests
// ============================================================================

class JitNegativeTestBase : public JitGeneratorTestBase
{
  protected:
    void expectGenerationFailure(
        const std::string&                               testName,
        dlp::jit::jitGeneratorError                      expectedError,
        std::function<dlp::jit::jitGeneratorError(void)> testFunc)
    {
        auto result = CrashIsolation::runIsolated(
            [&]() { return static_cast<int>(testFunc()); });

        if (result.crashed) {
            FAIL() << testName << " crashed with "
                   << CrashIsolation::signalName(result.crashSignal)
                   << " (signal " << result.crashSignal << ")\n"
                   << "The generator should return an error code, not crash.";
        } else if (result.threw) {
            SUCCEED() << testName << " threw exception as expected: "
                      << result.exceptionMessage;
        } else {
            auto err =
                static_cast<dlp::jit::jitGeneratorError>(result.returnCode);
            EXPECT_NE(err, dlp::jit::jitGeneratorError::success)
                << testName << " should have failed but succeeded";
            if (expectedError != dlp::jit::jitGeneratorError::success) {
                EXPECT_EQ(err, expectedError)
                    << testName << " failed with unexpected error";
            }
        }
    }

    const std::vector<kernelInstrType>& getKTypes(Datatype dt)
    {
        switch (dt) {
            case Datatype::F32:
                return allKTypes_f32;
            case Datatype::BF16:
                return allKTypes_bf16;
            case Datatype::U8S8:
                return allKTypes_u8s8;
            case Datatype::S8:
                return allKTypes_s8;
        }
        return allKTypes_f32;
    }

    dlp::jit::jitGeneratorError dispatchGemmKernel(Datatype         dt,
                                                   kernelInstrType  kType,
                                                   generatorParams& p)
    {
        switch (dt) {
            case Datatype::F32:
                return generateF32GemmKernel(kType, p);
            case Datatype::BF16:
                return generateBF16GemmKernel(kType, p);
            case Datatype::U8S8:
                return generateU8S8GemmKernel(kType, p);
            case Datatype::S8:
                return generateS8GemmKernel(kType, p);
        }
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    dlp::jit::jitGeneratorError dispatchGemvN1Kernel(Datatype        dt,
                                                     kernelInstrType kType,
                                                     gemvN1GeneratorParams& p)
    {
        switch (dt) {
            case Datatype::F32:
                return generateF32GemvN1Kernel(kType, p);
            case Datatype::BF16:
                return generateBF16GemvN1Kernel(kType, p);
            case Datatype::U8S8:
                return generateU8S8GemvN1Kernel(kType, p);
            case Datatype::S8:
                return generateS8GemvN1Kernel(kType, p);
        }
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    dlp::jit::jitGeneratorError dispatchGemvM1Kernel(Datatype        dt,
                                                     kernelInstrType kType,
                                                     gemvM1GeneratorParams& p)
    {
        switch (dt) {
            case Datatype::F32:
                return generateF32GemvM1Kernel(kType, p);
            case Datatype::BF16:
                return generateBF16GemvM1Kernel(kType, p);
            case Datatype::U8S8:
                return generateU8S8GemvM1Kernel(kType, p);
            case Datatype::S8:
                return generateS8GemvM1Kernel(kType, p);
        }
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
};

// ============================================================================
// GEMM Negative Parameterized Tests
// ============================================================================

class JitGemmNegativeTest
    : public JitNegativeTestBase
    , public ::testing::WithParamInterface<GemmNegativeParam>
{};

TEST_P(JitGemmNegativeTest, InvalidParam)
{
    const auto& p      = GetParam();
    const auto& kTypes = getKTypes(p.dtype);
    if (kTypes.empty()) {
        GTEST_SKIP() << datatypeSkipMessage(p.dtype);
    }

    for (const auto& kType : kTypes) {
        kernelInfo ki  = MockKernelInfoGenerator::createGEMMKernelInfo(kType);
        ki.c_downscale = datatypeToDownscale(p.dtype);
        ki.prefetch_c_dist = datatypeToPrefetch(p.dtype);

        generatorParams gen_params = p.buildParams(ki, kType);

        std::string testName = datatypeToString(p.dtype) + "_GEMM_" + p.testCase
                               + " [" + kTypeToArchString(kType) + "]";

        expectGenerationFailure(
            testName, dlp::jit::jitGeneratorError::badKernelInfo,
            [&]() -> dlp::jit::jitGeneratorError {
                return dispatchGemmKernel(p.dtype, kType, gen_params);
            });
    }
}

INSTANTIATE_TEST_SUITE_P(
    GemmNegative,
    JitGemmNegativeTest,
    ::testing::Values(
        // --- F32 ---
        GemmNegativeParam{ Datatype::F32, "NegativeMR",
                           corruptParam([](kernelInfo& ki) { ki.mr = -1; }) },
        GemmNegativeParam{ Datatype::F32, "NegativeNR",
                           corruptParam([](kernelInfo& ki) { ki.nr = -1; }) },
        GemmNegativeParam{ Datatype::F32, "ZeroMR",
                           corruptParam([](kernelInfo& ki) { ki.mr = 0; }) },
        GemmNegativeParam{
            Datatype::F32, "NegativeKUnroll",
            corruptParam([](kernelInfo& ki) { ki.k_unroll = -1; }) },
        GemmNegativeParam{
            Datatype::F32, "ZeroKUnroll",
            corruptParam([](kernelInfo& ki) { ki.k_unroll = 0; }) },
        GemmNegativeParam{ Datatype::F32, "ExcessiveMR",
                           corruptParam([](kernelInfo& ki) { ki.mr = 1000; }) },
        GemmNegativeParam{ Datatype::F32, "ExcessiveNR",
                           corruptParam([](kernelInfo& ki) { ki.nr = 1000; }) },
        GemmNegativeParam{ Datatype::F32, "NegativeNumMaskRegs",
                           corruptNumMaskRegs(-1) },
        GemmNegativeParam{ Datatype::F32, "ExcessiveNumMaskRegs",
                           corruptNumMaskRegs(8, true) },
        GemmNegativeParam{ Datatype::F32, "InvalidCDownscale",
                           corruptCDownscale(-1) },
        // --- BF16 ---
        GemmNegativeParam{ Datatype::BF16, "NegativeMR",
                           corruptParam([](kernelInfo& ki) { ki.mr = -1; }) },
        GemmNegativeParam{ Datatype::BF16, "NegativeNR",
                           corruptParam([](kernelInfo& ki) { ki.nr = -1; }) },
        GemmNegativeParam{ Datatype::BF16, "ZeroMR",
                           corruptParam([](kernelInfo& ki) { ki.mr = 0; }) },
        GemmNegativeParam{
            Datatype::BF16, "NegativeKUnroll",
            corruptParam([](kernelInfo& ki) { ki.k_unroll = -1; }) },
        GemmNegativeParam{
            Datatype::BF16, "ZeroKUnroll",
            corruptParam([](kernelInfo& ki) { ki.k_unroll = 0; }) },
        GemmNegativeParam{ Datatype::BF16, "ExcessiveMR",
                           corruptParam([](kernelInfo& ki) { ki.mr = 1000; }) },
        GemmNegativeParam{ Datatype::BF16, "ExcessiveNR",
                           corruptParam([](kernelInfo& ki) { ki.nr = 1000; }) },
        GemmNegativeParam{ Datatype::BF16, "NegativeNumMaskRegs",
                           corruptNumMaskRegs(-1) },
        GemmNegativeParam{ Datatype::BF16, "ExcessiveNumMaskRegs",
                           corruptNumMaskRegs(8) },
        GemmNegativeParam{ Datatype::BF16, "InvalidCDownscale",
                           corruptCDownscale(-1) },
        // --- U8S8 ---
        GemmNegativeParam{ Datatype::U8S8, "NegativeMR",
                           corruptParam([](kernelInfo& ki) { ki.mr = -1; }) },
        GemmNegativeParam{ Datatype::U8S8, "NegativeNR",
                           corruptParam([](kernelInfo& ki) { ki.nr = -1; }) },
        GemmNegativeParam{ Datatype::U8S8, "ZeroMR",
                           corruptParam([](kernelInfo& ki) { ki.mr = 0; }) },
        GemmNegativeParam{
            Datatype::U8S8, "NegativeKUnroll",
            corruptParam([](kernelInfo& ki) { ki.k_unroll = -1; }) },
        GemmNegativeParam{
            Datatype::U8S8, "ZeroKUnroll",
            corruptParam([](kernelInfo& ki) { ki.k_unroll = 0; }) },
        GemmNegativeParam{ Datatype::U8S8, "ExcessiveMR",
                           corruptParam([](kernelInfo& ki) { ki.mr = 1000; }) },
        GemmNegativeParam{ Datatype::U8S8, "ExcessiveNR",
                           corruptParam([](kernelInfo& ki) { ki.nr = 1000; }) },
        GemmNegativeParam{ Datatype::U8S8, "NegativeNumMaskRegs",
                           corruptNumMaskRegs(-1) },
        GemmNegativeParam{ Datatype::U8S8, "ExcessiveNumMaskRegs",
                           corruptNumMaskRegs(8) },
        GemmNegativeParam{ Datatype::U8S8, "InvalidCDownscale",
                           corruptCDownscale(-1) },
        // --- S8 ---
        GemmNegativeParam{ Datatype::S8, "NegativeMR",
                           corruptParam([](kernelInfo& ki) { ki.mr = -1; }) },
        GemmNegativeParam{ Datatype::S8, "NegativeNR",
                           corruptParam([](kernelInfo& ki) { ki.nr = -1; }) },
        GemmNegativeParam{ Datatype::S8, "ZeroMR",
                           corruptParam([](kernelInfo& ki) { ki.mr = 0; }) },
        GemmNegativeParam{
            Datatype::S8, "NegativeKUnroll",
            corruptParam([](kernelInfo& ki) { ki.k_unroll = -1; }) },
        GemmNegativeParam{
            Datatype::S8, "ZeroKUnroll",
            corruptParam([](kernelInfo& ki) { ki.k_unroll = 0; }) },
        GemmNegativeParam{ Datatype::S8, "ExcessiveMR",
                           corruptParam([](kernelInfo& ki) { ki.mr = 1000; }) },
        GemmNegativeParam{ Datatype::S8, "ExcessiveNR",
                           corruptParam([](kernelInfo& ki) { ki.nr = 1000; }) },
        GemmNegativeParam{ Datatype::S8, "NegativeNumMaskRegs",
                           corruptNumMaskRegs(-1) },
        GemmNegativeParam{ Datatype::S8, "ExcessiveNumMaskRegs",
                           corruptNumMaskRegs(8) },
        GemmNegativeParam{ Datatype::S8, "InvalidCDownscale",
                           corruptCDownscale(-1) }),
    [](const ::testing::TestParamInfo<GemmNegativeParam>& info) {
        return datatypeToString(info.param.dtype) + "_GEMM_"
               + info.param.testCase;
    });

// ============================================================================
// GEMV N=1 Negative Parameterized Tests
// ============================================================================

class JitGemvN1NegativeTest
    : public JitNegativeTestBase
    , public ::testing::WithParamInterface<GemvN1NegativeParam>
{};

TEST_P(JitGemvN1NegativeTest, InvalidParam)
{
    const auto& p      = GetParam();
    const auto& kTypes = getKTypes(p.dtype);
    if (kTypes.empty()) {
        GTEST_SKIP() << datatypeSkipMessage(p.dtype);
    }

    for (const auto& kType : kTypes) {
        kernelInfo ki = MockKernelInfoGenerator::createGEMVN1KernelInfo(kType);

        gemvN1GeneratorParams gen_params = p.buildParams(ki, kType);

        std::string testName = datatypeToString(p.dtype) + "_GEMVN1_"
                               + p.testCase + " [" + kTypeToArchString(kType)
                               + "]";

        expectGenerationFailure(
            testName, dlp::jit::jitGeneratorError::badKernelInfo,
            [&]() -> dlp::jit::jitGeneratorError {
                return dispatchGemvN1Kernel(p.dtype, kType, gen_params);
            });
    }
}

INSTANTIATE_TEST_SUITE_P(
    GemvN1Negative,
    JitGemvN1NegativeTest,
    ::testing::Values(
        GemvN1NegativeParam{ Datatype::F32, "NegativeMR", corruptGemvN1MR(-1) },
        GemvN1NegativeParam{ Datatype::F32, "ZeroMR", corruptGemvN1MR(0) },
        GemvN1NegativeParam{ Datatype::BF16, "NegativeMR",
                             corruptGemvN1MR(-1) },
        GemvN1NegativeParam{ Datatype::U8S8, "ZeroMR", corruptGemvN1MR(0) },
        GemvN1NegativeParam{ Datatype::S8, "NegativeMR", corruptGemvN1MR(-1) },
        GemvN1NegativeParam{ Datatype::F32, "InvalidCDownscale",
                             corruptGemvN1CDownscale(-1) },
        GemvN1NegativeParam{ Datatype::BF16, "InvalidCDownscale",
                             corruptGemvN1CDownscale(-1) },
        GemvN1NegativeParam{ Datatype::U8S8, "InvalidCDownscale",
                             corruptGemvN1CDownscale(-1) },
        GemvN1NegativeParam{ Datatype::S8, "InvalidCDownscale",
                             corruptGemvN1CDownscale(-1) }),
    [](const ::testing::TestParamInfo<GemvN1NegativeParam>& info) {
        return datatypeToString(info.param.dtype) + "_GEMVN1_"
               + info.param.testCase;
    });

// ============================================================================
// GEMV M=1 Negative Parameterized Tests
// ============================================================================

class JitGemvM1NegativeTest
    : public JitNegativeTestBase
    , public ::testing::WithParamInterface<GemvM1NegativeParam>
{};

TEST_P(JitGemvM1NegativeTest, InvalidParam)
{
    const auto& p      = GetParam();
    const auto& kTypes = getKTypes(p.dtype);
    if (kTypes.empty()) {
        GTEST_SKIP() << datatypeSkipMessage(p.dtype);
    }

    for (const auto& kType : kTypes) {
        kernelInfo ki = MockKernelInfoGenerator::createGEMVM1KernelInfo(kType);

        gemvM1GeneratorParams gen_params = p.buildParams(ki, kType);

        std::string testName = datatypeToString(p.dtype) + "_GEMVM1_"
                               + p.testCase + " [" + kTypeToArchString(kType)
                               + "]";

        expectGenerationFailure(
            testName, dlp::jit::jitGeneratorError::badKernelInfo,
            [&]() -> dlp::jit::jitGeneratorError {
                return dispatchGemvM1Kernel(p.dtype, kType, gen_params);
            });
    }
}

INSTANTIATE_TEST_SUITE_P(
    GemvM1Negative,
    JitGemvM1NegativeTest,
    ::testing::Values(
        GemvM1NegativeParam{
            Datatype::F32, "NegativeNR",
            corruptGemvM1Field([](kernelInfo& ki) { ki.nr = -1; }) },
        GemvM1NegativeParam{
            Datatype::F32, "ZeroNR",
            corruptGemvM1Field([](kernelInfo& ki) { ki.nr = 0; }) },
        GemvM1NegativeParam{
            Datatype::F32, "ZeroKC",
            corruptGemvM1Field([](kernelInfo& ki) { ki.kc = 0; }) },
        GemvM1NegativeParam{
            Datatype::F32, "ZeroKSubIter",
            corruptGemvM1Field([](kernelInfo& ki) { ki.k_unroll = 0; }) },
        GemvM1NegativeParam{
            Datatype::BF16, "NegativeNR",
            corruptGemvM1Field([](kernelInfo& ki) { ki.nr = -1; }) },
        GemvM1NegativeParam{
            Datatype::U8S8, "ZeroNR",
            corruptGemvM1Field([](kernelInfo& ki) { ki.nr = 0; }) },
        GemvM1NegativeParam{
            Datatype::S8, "NegativeKSubIter",
            corruptGemvM1Field([](kernelInfo& ki) { ki.k_unroll = -1; }) },
        GemvM1NegativeParam{ Datatype::F32, "InvalidCDownscale",
                             corruptGemvM1CDownscale(-1) },
        GemvM1NegativeParam{ Datatype::BF16, "InvalidCDownscale",
                             corruptGemvM1CDownscale(-1) },
        GemvM1NegativeParam{ Datatype::U8S8, "InvalidCDownscale",
                             corruptGemvM1CDownscale(-1) },
        GemvM1NegativeParam{ Datatype::S8, "InvalidCDownscale",
                             corruptGemvM1CDownscale(-1) }),
    [](const ::testing::TestParamInfo<GemvM1NegativeParam>& info) {
        return datatypeToString(info.param.dtype) + "_GEMVM1_"
               + info.param.testCase;
    });

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
