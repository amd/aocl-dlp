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
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "jit_generator_tests_utils.hh"

using namespace amdzen::codegen;
using namespace amdzen::utils;
using namespace dlp::kernel_frame;
using namespace test_jit_utils;

namespace {

enum class MatrixOp
{
    Add,
    Mul
};

enum class ScaleMode
{
    Scalar,
    Vector
};

class JitGemvM1MatrixOpsTest : public JitGeneratorTestBase
{
  protected:
    template<kernelInstrType KType>
    void runCase(MatrixOp opType, storageFormat format, ScaleMode scaleMode)
    {
        constexpr int nr = (KType == kernelInstrType::avx512_zmm_32_reg) ? 16
                                                                         : 8;
        constexpr int fringe   = 3;
        constexpr int n        = nr + fringe;
        constexpr int startRow = 0;
        constexpr int startCol = 1;

        gemvM1GeneratorParams genParams(
            DLP_F32, nr, fringe, 4, 4, UNPACKED, true, false, true, false,
            format, scalingType::zero, scalingType::one, KType);

        kernelOpsMetaData op;
        op.type           = (opType == MatrixOp::Add) ? kernelOps::matAdd
                                                      : kernelOps::matMul;
        op.paramStorageDt = DataType::f32;
        op.scaleFactorDt  = DataType::f32;
        op.sfDim          = (scaleMode == ScaleMode::Scalar) ? ParamDim::Scalar
                                                             : ParamDim::PerN;
        op.cMatFormat     = format;
        genParams.kernelOps.push_back(op);

        jitF32GEMVM1<KType> generator(JIT_KERNEL_SIZE);
        ASSERT_EQ(generator.generateKernel(genParams),
                  dlp::jit::jitGeneratorError::success);
        generator.ready();
        auto kernel = generator.template getCode<jit_gemv_m1_kernel>();
        ASSERT_NE(kernel, nullptr);

        const int  logicalRows = startRow + 1 + 1;
        const int  logicalCols = startCol + n + 1;
        const md_t compactLdm =
            (format == storageFormat::rowMajor) ? logicalCols : logicalRows;
        const md_t paddedLdm = compactLdm + 5;

        std::vector<float> scaleFactors;
        float              scalarScale = 2.0f;
        if (scaleMode == ScaleMode::Vector) {
            const int scaleCount =
                (format == storageFormat::rowMajor) ? logicalCols : logicalRows;
            scaleFactors.resize(scaleCount);
            for (int i = 0; i < scaleCount; ++i) {
                scaleFactors[i] = static_cast<float>(i + 1);
            }
        }

        std::vector<float> compactOutput;
        for (const md_t ldm : { compactLdm, paddedLdm }) {
            md_t               runtimeLdm = ldm;
            const size_t       auxSize    = (format == storageFormat::rowMajor)
                                                ? static_cast<size_t>(logicalRows * ldm)
                                                : static_cast<size_t>(logicalCols * ldm);
            std::vector<float> auxiliary(auxSize, -1000.0f);

            for (int j = 0; j < n; ++j) {
                const size_t index =
                    (format == storageFormat::rowMajor)
                        ? static_cast<size_t>(startRow * ldm + startCol + j)
                        : static_cast<size_t>((startCol + j) * ldm + startRow);
                auxiliary[index] = static_cast<float>(j + 2);
            }

            char order = (format == storageFormat::rowMajor) ? 'r' : 'c';
            dlp_gemm_post_op runtimeOp{};
            runtimeOp.op_code  = (opType == MatrixOp::Add) ? POST_OPS_MATRIX_ADD
                                                           : POST_OPS_MATRIX_MUL;
            runtimeOp.op_args1 = auxiliary.data();
            runtimeOp.op_args2 = &order;
            runtimeOp.op_args3 = &runtimeLdm;
            runtimeOp.scale_factor =
                (scaleMode == ScaleMode::Scalar)
                    ? static_cast<void*>(&scalarScale)
                    : static_cast<void*>(scaleFactors.data());
            runtimeOp.scale_factor_len =
                (scaleMode == ScaleMode::Scalar)
                    ? 1
                    : static_cast<md_t>(scaleFactors.size());
            runtimeOp.stor_type        = DLP_F32;
            runtimeOp.sf_stor_type     = DLP_F32;
            runtimeOp.scale_factor_dim = (scaleMode == ScaleMode::Scalar)
                                             ? DLP_PARAM_DIM_PER_TENSOR
                                             : DLP_PARAM_DIM_PER_CHANNEL;

            dlp_gemm_post_op_attr attr{};
            attr.post_op_c_i = startRow;
            attr.post_op_c_j = startCol;
            attr.is_first_k  = 1;
            attr.is_last_k   = 1;
            attr.c_stor_type = DLP_F32;

            std::vector<float> x(1, 0.0f);
            std::vector<float> b(1, 0.0f);
            std::vector<float> y(n);
            for (int j = 0; j < n; ++j) {
                y[j] = static_cast<float>(10 + j);
            }
            const std::vector<float> inputY = y;
            float                    alpha  = 0.0f;
            float                    beta   = 1.0f;

            // Public col-major GEMV is induced to a row-major kernel with
            // csY==1. Direct M=1 col-major must use csY != 1 so colMajorPath
            // gathers along N with stride ldm (true 1xN column-major aux).
            const md_t csY = (format == storageFormat::rowMajor) ? 1
                                                                 : logicalRows;
            dlp::kernels::gemvM1Params params(x.data(), b.data(), y.data(), n,
                                              1, 1, 1, 1, 1, 1, csY, 0, 0,
                                              &alpha, &beta, &runtimeOp, attr);
            params.n_iter = 1;
            params.n_left = fringe;

            if constexpr (KType == kernelInstrType::avx512_zmm_32_reg) {
                params.nmask_avx512 =
                    static_cast<uint16_t>((1u << fringe) - 1u);
            } else if constexpr (KType == kernelInstrType::avx512_ymm_32_reg) {
                params.nmask_avx512_256 =
                    static_cast<uint8_t>((1u << fringe) - 1u);
            } else {
                std::fill_n(params.nmask_avx2.begin(), fringe, -1);
            }

            kernel(&params);

            for (int j = 0; j < n; ++j) {
                const float scale    = (scaleMode == ScaleMode::Scalar)
                                           ? scalarScale
                                           : ((format == storageFormat::rowMajor)
                                                  ? scaleFactors[startCol + j]
                                                  : scaleFactors[startRow]);
                const float operand  = static_cast<float>(j + 2) * scale;
                const float expected = (opType == MatrixOp::Add)
                                           ? inputY[j] + operand
                                           : inputY[j] * operand;
                EXPECT_FLOAT_EQ(y[j], expected)
                    << "col=" << j << " ldm=" << ldm;
            }

            if (compactOutput.empty()) {
                compactOutput = y;
            } else {
                EXPECT_EQ(y, compactOutput)
                    << "padded ldm changed the logical result";
            }
        }
    }

    template<kernelInstrType KType>
    void runAllCases()
    {
        for (const MatrixOp opType : { MatrixOp::Add, MatrixOp::Mul }) {
            for (const storageFormat format :
                 { storageFormat::rowMajor, storageFormat::colMajor }) {
                for (const ScaleMode scaleMode :
                     { ScaleMode::Scalar, ScaleMode::Vector }) {
                    SCOPED_TRACE(opType == MatrixOp::Add ? "MATRIX_ADD"
                                                         : "MATRIX_MUL");
                    SCOPED_TRACE(format == storageFormat::rowMajor
                                     ? "row-major auxiliary"
                                     : "column-major auxiliary");
                    SCOPED_TRACE(scaleMode == ScaleMode::Scalar
                                     ? "scalar scale"
                                     : "vector scale");
                    runCase<KType>(opType, format, scaleMode);
                }
            }
        }
    }
};

TEST_F(JitGemvM1MatrixOpsTest, DirectKernelHonorsAuxiliaryLeadingDimension)
{
    std::vector<kernelInstrType> kernelTypes = allKTypes_f32;
    const bool hasAvx2 = dlp::arch_utils::archConfigManager::getInstance()
                             .isAvx2Fma3SupportedByArch();
    if (hasAvx2
        && std::find(kernelTypes.begin(), kernelTypes.end(),
                     kernelInstrType::avx2_ymm_16_reg)
               == kernelTypes.end()) {
        kernelTypes.push_back(kernelInstrType::avx2_ymm_16_reg);
    }

    if (kernelTypes.empty()) {
        GTEST_SKIP() << "F32 JIT GEMV-M1 requires AVX2 or AVX-512";
    }

    for (const kernelInstrType kType : kernelTypes) {
        SCOPED_TRACE(kTypeToArchString(kType));
        switch (kType) {
            case kernelInstrType::avx512_zmm_32_reg:
                runAllCases<kernelInstrType::avx512_zmm_32_reg>();
                break;
            case kernelInstrType::avx512_ymm_32_reg:
                runAllCases<kernelInstrType::avx512_ymm_32_reg>();
                break;
            case kernelInstrType::avx2_ymm_16_reg:
                runAllCases<kernelInstrType::avx2_ymm_16_reg>();
                break;
            default:
                FAIL() << "Unexpected F32 kernel type";
        }
    }
}

} // namespace

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
