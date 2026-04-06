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

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <iomanip>
#include <memory>

#include "jit_generator_tests_utils.hh" // JitGeneratorTestBase and utilities

using namespace amdzen::gen;     // amdzen generators
using namespace amdzen::codegen; // jit generate kernel function
using namespace amdzen::utils;   // generator params, kernel Instr type
using namespace test_jit_utils;  // Mock builders and signal handling utilities

// ============================================================================
// Memory Bound Test Fixture (extends JitGeneratorTestBase)
// ============================================================================

/**
 * @brief Test fixture for memory-bound JIT generator tests
 *
 * Extends JitGeneratorTestBase with utilities specific to memory bound testing:
 * - Code size verification
 * - Buffer utilization warnings
 */
class JitMemoryBoundTest : public test_jit_utils::JitGeneratorTestBase
{
  protected:
    /**
     * @brief Reusable helper for GEMM maximal post-ops tests across datatypes
     * @param datatypeName Name of the datatype being tested
     * @param allKTypes Vector of kernel instruction types to test
     * @param skipMessage Message to display when skipping unsupported types
     * @param downscaleTypes Vector of downscale type pairs for post-ops
     * @param generateFunc Callback to generate the JIT kernel
     * @param kiModifier Optional callback to tweak kernelInfo before param
     *                   construction (e.g., U8S8 sets prefetch_c_dist = 0)
     */
    void testGemmMaximalPostOps(
        const std::string&                              datatypeName,
        const std::vector<kernelInstrType>&             allKTypes,
        const std::string&                              skipMessage,
        const std::vector<std::pair<int, std::string>>& downscaleTypes,
        std::function<dlp::jit::jitGeneratorError(
            kernelInstrType, generatorParams&)>         generateFunc,
        std::function<void(kernelInfo&)>                kiModifier = nullptr)
    {
        if (allKTypes.empty()) {
            GTEST_SKIP() << skipMessage;
        }

        for (const auto& kType : allKTypes) {
            kernelInfo ki_gemm =
                MockKernelInfoGenerator::createGEMMKernelInfo(kType);

            if (kiModifier) {
                kiModifier(ki_gemm);
            }

            for (const auto& [c_downscale, typeName] : downscaleTypes) {
                generatorParams gen_params(
                    ki_gemm.mr, ki_gemm.nr, ki_gemm.k_unroll,
                    ki_gemm.prefetch_c_dist, c_downscale, ki_gemm.kOpsArrSize,
                    ki_gemm.genLtKrnlForAvailFullKrnl, true, ki_gemm.invokeRD,
                    ki_gemm.alphaScalingType, ki_gemm.betaScalingType, kType);

                gen_params.kernelOps =
                    KernelOpsBuilder::createMaximalConfig().build();

                std::string testName = datatypeName + "_GEMM_Out_" + typeName
                                       + " [" + kTypeToArchString(kType) + "]";
                expectGenerationSuccess(testName, [&]() {
                    return generateFunc(kType, gen_params);
                });
            }
        }
    }

    /**
     * @brief Reusable helper for GEMV N=1 maximal post-ops tests
     */
    void testGemvN1MaximalPostOps(
        const std::string&                              datatypeName,
        const std::vector<kernelInstrType>&             allKTypes,
        const std::string&                              skipMessage,
        const std::vector<std::pair<int, std::string>>& downscaleTypes,
        int                                             mLeft,
        std::function<dlp::jit::jitGeneratorError(
            kernelInstrType, gemvN1GeneratorParams&)>   generateFunc)
    {
        if (allKTypes.empty()) {
            GTEST_SKIP() << skipMessage;
        }

        for (const auto& kType : allKTypes) {
            kernelInfo ki_gemvn1 =
                MockKernelInfoGenerator::createGEMVN1KernelInfo(kType);

            for (const auto& [c_downscale, typeName] : downscaleTypes) {
                gemvN1GeneratorParams gen_params(
                    ki_gemvn1.mr, mLeft, c_downscale, true, true, true, true,
                    storageFormat::colMajor, ki_gemvn1.alphaScalingType,
                    ki_gemvn1.betaScalingType, kType);

                gen_params.kernelOps =
                    KernelOpsBuilder::createMaximalConfig().build();

                std::string testName = datatypeName + "_GEMVN1_Out_" + typeName
                                       + " [" + kTypeToArchString(kType) + "]";
                expectGenerationSuccess(testName, [&]() {
                    return generateFunc(kType, gen_params);
                });
            }
        }
    }

    /**
     * @brief Reusable helper for GEMV M=1 maximal post-ops tests
     */
    void testGemvM1MaximalPostOps(
        const std::string&                              datatypeName,
        const std::vector<kernelInstrType>&             allKTypes,
        const std::string&                              skipMessage,
        const std::vector<std::pair<int, std::string>>& downscaleTypes,
        int                                             nLeft,
        std::function<dlp::jit::jitGeneratorError(
            kernelInstrType, gemvM1GeneratorParams&)>   generateFunc)
    {
        if (allKTypes.empty()) {
            GTEST_SKIP() << skipMessage;
        }

        for (const auto& kType : allKTypes) {
            kernelInfo ki_gemvm1 =
                MockKernelInfoGenerator::createGEMVM1KernelInfo(kType);

            for (const auto& [c_downscale, typeName] : downscaleTypes) {
                gemvM1GeneratorParams gen_params(
                    c_downscale, ki_gemvm1.nr, nLeft, ki_gemvm1.kc,
                    ki_gemvm1.k_unroll, PACK, true, true, true, true,
                    storageFormat::rowMajor, ki_gemvm1.alphaScalingType,
                    ki_gemvm1.betaScalingType, kType);

                gen_params.nfringe_main = true;
                gen_params.nfringe_left = true;
                gen_params.N_LEFT_16    = 48;
                gen_params.N_LEFT_LT16  = 15;

                gen_params.kernelOps =
                    KernelOpsBuilder::createMaximalConfig().build();

                std::string testName = datatypeName + "_GEMVM1_Out_" + typeName
                                       + " [" + kTypeToArchString(kType) + "]";
                expectGenerationSuccess(testName, [&]() {
                    return generateFunc(kType, gen_params);
                });
            }
        }
    }
};

// ============================================================================
// GEMM Kernel Tests (MR > 1, NR > 1) - All Datatypes
// ============================================================================

TEST_F(JitMemoryBoundTest, F32_GEMM_AllOutputTypes_MaximalPostOps)
{
    testGemmMaximalPostOps("F32", allKTypes_f32,
                           "F32 tests require AVX2 or AVX512 support",
                           { { DLP_F32, "F32" } },
                           [this](kernelInstrType kType, generatorParams& p) {
                               return generateF32GemmKernel(kType, p);
                           });
}

TEST_F(JitMemoryBoundTest, BF16_GEMM_AllOutputTypes_MaximalPostOps)
{
    testGemmMaximalPostOps("BF16", allKTypes_bf16,
                           "BF16 tests require AVX512_BF16 support",
                           { { DLP_F32, "F32" }, { DLP_BF16, "BF16" } },
                           [this](kernelInstrType kType, generatorParams& p) {
                               return generateBF16GemmKernel(kType, p);
                           });
}

// GEMM All Output Types Tests (Integer Kernels)
TEST_F(JitMemoryBoundTest, U8S8_GEMM_AllOutputTypes_MaximalPostOps)
{
    testGemmMaximalPostOps(
        "U8S8", allKTypes_u8s8, "U8S8 tests require AVX512_VNNI support",
        { { DLP_S32, "S32" },
          { DLP_F32, "F32" },
          { DLP_BF16, "BF16" },
          { DLP_S8, "S8" },
          { DLP_U8, "U8" } },
        [this](kernelInstrType kType, generatorParams& p) {
            return generateU8S8GemmKernel(kType, p);
        },
        [](kernelInfo& ki) { ki.prefetch_c_dist = 0; });
}

TEST_F(JitMemoryBoundTest, S8_GEMM_AllOutputTypes_MaximalPostOps)
{
    testGemmMaximalPostOps("S8", allKTypes_s8,
                           "S8 tests require AVX512_VNNI support",
                           { { DLP_S32, "S32" },
                             { DLP_F32, "F32" },
                             { DLP_BF16, "BF16" },
                             { DLP_S8, "S8" },
                             { DLP_U8, "U8" } },
                           [this](kernelInstrType kType, generatorParams& p) {
                               return generateS8GemmKernel(kType, p);
                           });
}

// ============================================================================
// GEMV N=1 Kernel Tests - All Datatypes
// ============================================================================

TEST_F(JitMemoryBoundTest, F32_GEMVN1_MaximalPostOps)
{
    testGemvN1MaximalPostOps(
        "F32", allKTypes_f32, "F32 tests require AVX2 or AVX512 support",
        { { DLP_F32, "F32" } }, 7,
        [this](kernelInstrType kType, gemvN1GeneratorParams& p) {
            return generateF32GemvN1Kernel(kType, p);
        });
}

TEST_F(JitMemoryBoundTest, BF16_GEMVN1_AllOutputTypes_MaximalPostOps)
{
    testGemvN1MaximalPostOps(
        "BF16", allKTypes_bf16, "BF16 tests require AVX512_BF16 support",
        { { DLP_F32, "F32" }, { DLP_BF16, "BF16" } }, 15,
        [this](kernelInstrType kType, gemvN1GeneratorParams& p) {
            return generateBF16GemvN1Kernel(kType, p);
        });
}

// GEMV N=1 All Output Types Tests (Integer Kernels)
TEST_F(JitMemoryBoundTest, U8S8_GEMVN1_AllOutputTypes_MaximalPostOps)
{
    testGemvN1MaximalPostOps(
        "U8S8", allKTypes_u8s8, "U8S8 tests require AVX512_VNNI support",
        { { DLP_S32, "S32" },
          { DLP_F32, "F32" },
          { DLP_BF16, "BF16" },
          { DLP_S8, "S8" },
          { DLP_U8, "U8" } },
        15, [this](kernelInstrType kType, gemvN1GeneratorParams& p) {
            return generateU8S8GemvN1Kernel(kType, p);
        });
}

TEST_F(JitMemoryBoundTest, S8_GEMVN1_AllOutputTypes_MaximalPostOps)
{
    testGemvN1MaximalPostOps(
        "S8", allKTypes_s8, "S8 tests require AVX512_VNNI support",
        { { DLP_S32, "S32" },
          { DLP_F32, "F32" },
          { DLP_BF16, "BF16" },
          { DLP_S8, "S8" },
          { DLP_U8, "U8" } },
        15, [this](kernelInstrType kType, gemvN1GeneratorParams& p) {
            return generateS8GemvN1Kernel(kType, p);
        });
}

// ============================================================================
// GEMV M=1 Kernel Tests - All Datatypes
// ============================================================================

TEST_F(JitMemoryBoundTest, F32_GEMVM1_MaximalPostOps)
{
    testGemvM1MaximalPostOps(
        "F32", allKTypes_f32, "F32 tests require AVX2 or AVX512 support",
        { { DLP_F32, "F32" } }, 31,
        [this](kernelInstrType kType, gemvM1GeneratorParams& p) {
            return generateF32GemvM1Kernel(kType, p);
        });
}

TEST_F(JitMemoryBoundTest, BF16_GEMVM1_AllOutputTypes_MaximalPostOps)
{
    testGemvM1MaximalPostOps(
        "BF16", allKTypes_bf16, "BF16 tests require AVX512_BF16 support",
        { { DLP_F32, "F32" }, { DLP_BF16, "BF16" } }, 63,
        [this](kernelInstrType kType, gemvM1GeneratorParams& p) {
            return generateBF16GemvM1Kernel(kType, p);
        });
}

// GEMV M=1 All Output Types Tests (Integer Kernels)
TEST_F(JitMemoryBoundTest, U8S8_GEMVM1_AllOutputTypes_MaximalPostOps)
{
    testGemvM1MaximalPostOps(
        "U8S8", allKTypes_u8s8, "U8S8 tests require AVX512_VNNI support",
        { { DLP_S32, "S32" },
          { DLP_F32, "F32" },
          { DLP_BF16, "BF16" },
          { DLP_S8, "S8" },
          { DLP_U8, "U8" } },
        63, [this](kernelInstrType kType, gemvM1GeneratorParams& p) {
            return generateU8S8GemvM1Kernel(kType, p);
        });
}

TEST_F(JitMemoryBoundTest, S8_GEMVM1_AllOutputTypes_MaximalPostOps)
{
    testGemvM1MaximalPostOps(
        "S8", allKTypes_s8, "S8 tests require AVX512_VNNI support",
        { { DLP_S32, "S32" },
          { DLP_F32, "F32" },
          { DLP_BF16, "BF16" },
          { DLP_S8, "S8" },
          { DLP_U8, "U8" } },
        63, [this](kernelInstrType kType, gemvM1GeneratorParams& p) {
            return generateS8GemvM1Kernel(kType, p);
        });
}

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
