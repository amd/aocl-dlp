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

#include <gtest/gtest.h>
#include <vector>

#include "jit_generator_tests_utils.hh" // JitGeneratorTestBase and utilities

using namespace amdzen::gen;    // amdzen generators
using namespace amdzen::utils;  // generator params, kernel Instr type
using namespace test_jit_utils; // Mock builders and signal handling utilities

// ============================================================================
// Test Fixture for MR/NR Variant Tests
// ============================================================================

/**
 * @brief Test fixture for testing various MR/NR combinations
 *
 * Extends JitGeneratorTestBase to test kernel generation with different
 * MR (number of rows) and NR (number of columns) configurations.
 * Tests can be extended for different datatypes (F32, BF16, U8S8, S8).
 */
class JitMrNrVariantsTest : public test_jit_utils::JitGeneratorTestBase
{
  protected:
    /**
     * @brief Generic helper to test GEMM with a specific post-op across all
     * kernel types
     * @param datatypeName Name of datatype (e.g., "F32", "BF16", "U8S8", "S8")
     * @param postOpName Name of post-op (e.g., "GeLUTanh", "NoPostOps")
     * @param allKTypes Vector of all supported kernel types to test
     * @param downscaleType Downscale type (e.g., DLP_S32)
     * @param postOpBuilder Function to build the post-op
     * @param generateFunc Function that performs kernel generation given params
     */
    void testGemmWithPostOp(
        const std::string&                                  datatypeName,
        const std::string&                                  postOpName,
        const std::vector<kernelInstrType>&                 allKTypes,
        int                                                 downscaleType,
        std::function<KernelOpsBuilder&(KernelOpsBuilder&)> postOpBuilder,
        std::function<dlp::jit::jitGeneratorError(generatorParams&)>
            generateFunc)
    {
        if (allKTypes.empty()) {
            GTEST_SKIP();
        }

        for (const auto& kType : allKTypes) {
            std::vector<std::pair<int, int>> mrNrCombinations;

            if (kType == kernelInstrType::avx512_zmm_32_reg) {
                mrNrCombinations = { { 1, 240 }, { 1, 224 }, { 1, 208 },
                                     { 1, 192 }, { 1, 176 }, { 2, 160 },
                                     { 2, 144 }, { 2, 128 }, { 4, 96 },
                                     { 6, 64 },  { 9, 48 },  { 14, 32 },
                                     { 16, 16 } };
            } else if (kType == kernelInstrType::avx512_ymm_32_reg) {
                mrNrCombinations = { { 1, 120 }, { 1, 112 }, { 1, 104 },
                                     { 1, 96 },  { 1, 88 },  { 2, 80 },
                                     { 2, 72 },  { 2, 64 },  { 3, 56 },
                                     { 4, 48 },  { 5, 40 },  { 6, 32 },
                                     { 9, 24 },  { 14, 16 }, { 30, 8 } };
            } else if (kType == kernelInstrType::avx2_ymm_16_reg) {
                mrNrCombinations = { { 6, 16 }, { 5, 16 }, { 4, 16 },
                                     { 3, 24 }, { 2, 32 }, { 1, 48 } };
            }

            kernelInfo ki_gemm =
                MockKernelInfoGenerator::createGEMMKernelInfo(kType);

            for (const auto& [mr, nr] : mrNrCombinations) {
                std::string testName =
                    datatypeName + "_GEMM_MR" + std::to_string(mr) + "_NR"
                    + std::to_string(nr) + "_With" + postOpName + " ["
                    + kTypeToArchString(kType) + "]";

                generatorParams gen_params(
                    mr, nr, ki_gemm.k_unroll, ki_gemm.prefetch_c_dist,
                    downscaleType, ki_gemm.kOpsArrSize,
                    ki_gemm.genLtKrnlForAvailFullKrnl, true, ki_gemm.invokeRD,
                    ki_gemm.alphaScalingType, ki_gemm.betaScalingType, kType);

                KernelOpsBuilder builder;
                gen_params.kernelOps = postOpBuilder(builder).build();

                expectGenerationSuccess(
                    testName, [&]() { return generateFunc(gen_params); });
            }
        }
    }
};

// ============================================================================
// F32 GEMM MR/NR Variant Tests (Without Post-Ops)
// ============================================================================

TEST_F(JitMrNrVariantsTest, F32_GEMM_MR_NR_Variants)
{
    testGemmWithPostOp(
        "F32", "NoPostOps", allKTypes_f32, DLP_F32,
        [](KernelOpsBuilder& b) -> KernelOpsBuilder& { return b; },
        [this](generatorParams& p) {
            return generateF32GemmKernel(p.kType, p);
        });
}

// ============================================================================
// BF16 GEMM MR/NR Variant Tests (Without Post-Ops)
// ============================================================================

TEST_F(JitMrNrVariantsTest, BF16_GEMM_MR_NR_Variants)
{
    testGemmWithPostOp(
        "BF16", "NoPostOps", allKTypes_bf16, DLP_BF16,
        [](KernelOpsBuilder& b) -> KernelOpsBuilder& { return b; },
        [this](generatorParams& p) {
            return generateBF16GemmKernel(p.kType, p);
        });
}

// ============================================================================
// U8S8 GEMM MR/NR Variant Tests (Without Post-Ops)
// ============================================================================

TEST_F(JitMrNrVariantsTest, U8S8_GEMM_MR_NR_Variants)
{
    testGemmWithPostOp(
        "U8S8", "NoPostOps", allKTypes_u8s8, DLP_S32,
        [](KernelOpsBuilder& b) -> KernelOpsBuilder& { return b; },
        [this](generatorParams& p) {
            return generateU8S8GemmKernel(p.kType, p);
        });
}

// ============================================================================
// S8 GEMM MR/NR Variant Tests (Without Post-Ops)
// ============================================================================

TEST_F(JitMrNrVariantsTest, S8_GEMM_MR_NR_Variants)
{
    testGemmWithPostOp(
        "S8", "NoPostOps", allKTypes_s8, DLP_S32,
        [](KernelOpsBuilder& b) -> KernelOpsBuilder& { return b; },
        [this](generatorParams& p) {
            return generateS8GemmKernel(p.kType, p);
        });
}

// ============================================================================
// F32F16 GEMM MR/NR Variant Tests (Without Post-Ops)
// ============================================================================

TEST_F(JitMrNrVariantsTest, F32F16_GEMM_MR_NR_Variants)
{
    testGemmWithPostOp(
        "F32F16", "NoPostOps", allKTypes_f32f16, DLP_F32,
        [](KernelOpsBuilder& b) -> KernelOpsBuilder& { return b; },
        [this](generatorParams& p) {
            return generateF32F16GemmKernel(p.kType, p);
        });
}

// ============================================================================
// S8 GEMM MR/NR Variant Tests — Per-token (PerM) DOWNSCALE
//
// Exercises the PerM SF code path in getLoadMode/applyScaleFactor for both
// row-major and column-major C storage, ensuring JIT generation succeeds
// across all valid MR/NR pairs.
// ============================================================================

TEST_F(JitMrNrVariantsTest, S8_GEMM_Downscale_PerToken_RowMajor)
{
    using namespace dlp::kernel_frame;
    testGemmWithPostOp(
        "S8", "DownscalePerTokenRowMajor", allKTypes_s8, DLP_S32,
        [](KernelOpsBuilder& b) -> KernelOpsBuilder& {
            return b.addDownscale(DataType::f32, DataType::s8,
                                  storageFormat::rowMajor, ParamDim::PerM,
                                  ParamDim::Scalar);
        },
        [this](generatorParams& p) {
            return generateS8GemmKernel(p.kType, p);
        });
}

TEST_F(JitMrNrVariantsTest, S8_GEMM_Downscale_PerToken_ColMajor)
{
    using namespace dlp::kernel_frame;
    testGemmWithPostOp(
        "S8", "DownscalePerTokenColMajor", allKTypes_s8, DLP_S32,
        [](KernelOpsBuilder& b) -> KernelOpsBuilder& {
            return b.addDownscale(DataType::f32, DataType::s8,
                                  storageFormat::colMajor, ParamDim::PerM,
                                  ParamDim::Scalar);
        },
        [this](generatorParams& p) {
            return generateS8GemmKernel(p.kType, p);
        });
}

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
