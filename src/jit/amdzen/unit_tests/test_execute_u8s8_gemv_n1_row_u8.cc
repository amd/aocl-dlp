/*******************************************************************************
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
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
 ******************************************************************************/

#include <array>
#include <cstdint>
#include <gtest/gtest.h>

#include "jit_generator_tests_utils.hh"

TEST(U8S8GemvN1Execution, RowStoredU8WithF32PostOps)
{
    using namespace amdzen::gen;
    using namespace amdzen::utils;
    using namespace dlp::kernel_frame;
    using namespace test_jit_utils;

    if (ArchBasedKernelTypes::getAllKernelTypesForU8S8().empty()) {
        GTEST_SKIP() << "Requires AVX-512 VNNI";
    }

    constexpr auto kType = kernelInstrType::avx512_zmm_32_reg;

    gemvN1GeneratorParams genParams(16, 2, DLP_U8, false, true, true, true,
                                    storageFormat::rowMajor, scalingType::one,
                                    scalingType::zero, kType);
    genParams.kernelOps =
        KernelOpsBuilder().addReLUScale(DataType::f32).build();

    jitU8S8VNNI_GEMVN1<kType> generator(JIT_KERNEL_SIZE);
    ASSERT_EQ(generator.generateKernel(genParams),
              dlp::jit::jitGeneratorError::success);
    generator.ready();

    // Two rows and a two-element K tail. The raw dots are {-12, 15}.
    std::array<uint8_t, 4> a{ 3, 0, 0, 5 };
    std::array<int8_t, 2>  x{ -4, 3 };
    std::array<int32_t, 2> yScratch{};
    std::array<uint8_t, 4> output{ 0xA5, 0xA5, 0xA5, 0xA5 };

    int32_t alpha = 1;
    int32_t beta  = 0;
    float   slope = -1.0f;

    dlp_gemm_post_op postOp{};
    postOp.op_code  = POST_OPS_RELU_SCALE;
    postOp.op_args2 = &slope;

    dlp_gemm_post_op_attr attr{};
    attr.post_op_c_i    = 0;
    attr.post_op_c_j    = 0;
    attr.rs_c_downscale = 1;
    attr.cs_c_downscale = 1;
    attr.buf_downscale  = output.data();
    attr.is_first_k     = 1;
    attr.is_last_k      = 1;
    attr.c_stor_type    = DLP_U8;

    dlp::kernels::gemvN1Params runtime(a.data(), x.data(), yScratch.data(), 2,
                                       2, 2, 1, 1, 1, 1, 1, &alpha, &beta,
                                       &postOp, attr);

    runtime.m_iter          = 0;
    runtime.m_left          = 2;
    runtime.k_iter          = 0;
    runtime.k_left          = 2;
    runtime.kmask_i8_avx512 = 0x3;
    runtime.mmask_avx512    = 0x3;

    auto kernel = generator.getCode<jit_gemv_n1_kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(&runtime);

    // ReLUScale(-1) maps -12 -> 12 and leaves 15 unchanged.
    EXPECT_EQ(output[0], 12);
    EXPECT_EQ(output[1], 15);
    EXPECT_EQ(output[2], 0xA5);
    EXPECT_EQ(output[3], 0xA5);
}

TEST(U8S8GemvN1Execution, ColumnStoredF16MultipleMBlocksUseRowStride)
{
    using namespace amdzen::gen;
    using namespace amdzen::utils;
    using namespace dlp::kernel_frame;
    using namespace test_jit_utils;

    if (ArchBasedKernelTypes::getAllKernelTypesForU8S8().empty()) {
        GTEST_SKIP() << "Requires AVX-512 VNNI";
    }

    constexpr auto kType  = kernelInstrType::avx512_zmm_32_reg;
    constexpr int  rows   = 33;
    constexpr int  stride = 3;

    gemvN1GeneratorParams genParams(16, 1, DLP_F16, true, false, true, false,
                                    storageFormat::colMajor, scalingType::zero,
                                    scalingType::zero, kType);
    jitU8S8VNNI_GEMVN1<kType> generator(JIT_KERNEL_SIZE);
    ASSERT_EQ(generator.generateKernel(genParams),
              dlp::jit::jitGeneratorError::success);
    generator.ready();

    std::array<uint8_t, rows>           a{};
    std::array<int8_t, 1>               x{};
    std::array<int32_t, rows>           yScratch{};
    std::array<uint16_t, rows * stride> output{};
    output.fill(0xA5A5);

    int32_t alpha = 0;
    int32_t beta  = 0;

    dlp_gemm_post_op_attr attr{};
    attr.rs_c_downscale = stride;
    attr.cs_c_downscale = 1;
    attr.buf_downscale  = output.data();
    attr.is_first_k     = 1;
    attr.is_last_k      = 1;
    attr.c_stor_type    = DLP_F16;

    dlp::kernels::gemvN1Params runtime(a.data(), x.data(), yScratch.data(),
                                       rows, 0, 1, 1, 1, 1, 1, 1, &alpha, &beta,
                                       nullptr, attr);
    runtime.m_iter       = 2;
    runtime.m_left       = 1;
    runtime.mmask_avx512 = 0x1;

    auto kernel = generator.getCode<jit_gemv_n1_kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(&runtime);

    for (int index = 0; index < rows * stride; ++index) {
        const bool written =
            index < 16 || (index >= 16 * stride && index < 16 * stride + 16)
            || index == 32 * stride;
        EXPECT_EQ(output[index], written ? 0 : 0xA5A5);
    }
}
