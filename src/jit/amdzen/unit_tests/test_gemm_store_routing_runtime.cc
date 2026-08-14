/*******************************************************************************
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
 ******************************************************************************/

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>

#include "jit_generator_tests_utils.hh"

namespace {

using amdzen::utils::generatorParams;
using amdzen::utils::kernelInstrType;
using dlp::kernel_frame::scalingType;

constexpr auto        kType  = kernelInstrType::avx512_zmm_32_reg;
constexpr std::size_t kLanes = 16;
constexpr uint8_t     kGuard = 0xA5;

template<std::size_t PayloadBytes>
class GuardedBuffer
{
  public:
    static constexpr std::size_t guardBytes = 32;

    GuardedBuffer() { bytes.fill(kGuard); }

    void*       data() { return bytes.data() + guardBytes; }
    const void* data() const { return bytes.data() + guardBytes; }

    template<class T>
    T* as()
    {
        return static_cast<T*>(data());
    }
    template<class T>
    const T* as() const
    {
        return static_cast<const T*>(data());
    }

    void fillPayload(uint8_t value)
    {
        std::fill_n(bytes.data() + guardBytes, PayloadBytes, value);
    }

    void expectGuardsIntact() const
    {
        EXPECT_TRUE(std::all_of(bytes.begin(), bytes.begin() + guardBytes,
                                [](uint8_t value) { return value == kGuard; }));
        EXPECT_TRUE(std::all_of(bytes.end() - guardBytes, bytes.end(),
                                [](uint8_t value) { return value == kGuard; }));
    }

    void expectPayloadTailIntact(std::size_t usedBytes) const
    {
        ASSERT_LE(usedBytes, PayloadBytes);
        EXPECT_TRUE(std::all_of(bytes.begin() + guardBytes + usedBytes,
                                bytes.begin() + guardBytes + PayloadBytes,
                                [](uint8_t value) { return value == kGuard; }));
    }

  private:
    alignas(64) std::array<uint8_t, PayloadBytes + 2 * guardBytes> bytes{};
};

using GemmKernel = void (*)(dlp::kernels::gemmParams*);

dlp::kernels::gemmParams
makeRuntime(void* c, void* downscale, int outputType, void* alpha, void* beta)
{
    static std::array<uint8_t, 64> a{};
    static std::array<int8_t, 64>  b{};

    dlp_gemm_post_op_attr attr{};
    attr.buf_downscale  = downscale;
    attr.rs_c_downscale = kLanes;
    attr.cs_c_downscale = 1;
    attr.c_stor_type    = outputType;

    dlp::kernels::gemmParams runtime(a.data(), b.data(), c, 1, kLanes, 0, 4, 4,
                                     4, 64, 1, kLanes, kLanes, 1, alpha, beta,
                                     nullptr, attr);
    runtime.maskS32    = 0xFFFF;
    runtime.maskF32[0] = 0xFFFF;
    return runtime;
}

void
fillTyped(void* output, int outputType, int value)
{
    if (outputType == DLP_S8) {
        std::fill_n(static_cast<int8_t*>(output), kLanes,
                    static_cast<int8_t>(value));
    } else if (outputType == DLP_U8) {
        std::fill_n(static_cast<uint8_t*>(output), kLanes,
                    static_cast<uint8_t>(value));
    } else if (outputType == DLP_S32) {
        std::fill_n(static_cast<int32_t*>(output), kLanes, value);
    } else if (outputType == DLP_F32) {
        std::fill_n(static_cast<float*>(output), kLanes,
                    static_cast<float>(value));
    } else {
        ASSERT_TRUE(value == 0 || value == 2 || value == 4);
        const uint16_t encoded =
            value == 0
                ? 0
                : (outputType == DLP_F16
                       ? static_cast<uint16_t>(value == 2 ? 0x4000 : 0x4400)
                       : static_cast<uint16_t>(value == 2 ? 0x4000 : 0x4080));
        std::fill_n(static_cast<uint16_t*>(output), kLanes, encoded);
    }
}

void
expectTyped(const void* output, int outputType, int expected)
{
    if (outputType == DLP_S8) {
        for (std::size_t i = 0; i < kLanes; ++i) {
            EXPECT_EQ(static_cast<const int8_t*>(output)[i], expected);
        }
    } else if (outputType == DLP_U8) {
        for (std::size_t i = 0; i < kLanes; ++i) {
            EXPECT_EQ(static_cast<const uint8_t*>(output)[i], expected);
        }
    } else if (outputType == DLP_S32) {
        for (std::size_t i = 0; i < kLanes; ++i) {
            EXPECT_EQ(static_cast<const int32_t*>(output)[i], expected);
        }
    } else if (outputType == DLP_F32) {
        for (std::size_t i = 0; i < kLanes; ++i) {
            EXPECT_EQ(static_cast<const float*>(output)[i], expected);
        }
    } else {
        ASSERT_TRUE(expected == 0 || expected == 2 || expected == 4);
        const uint16_t encoded =
            expected == 0
                ? 0
                : (outputType == DLP_F16
                       ? static_cast<uint16_t>(expected == 2 ? 0x4000 : 0x4400)
                       : static_cast<uint16_t>(expected == 2 ? 0x4000
                                                             : 0x4080));
        for (std::size_t i = 0; i < kLanes; ++i) {
            EXPECT_EQ(static_cast<const uint16_t*>(output)[i], encoded);
        }
    }
}

std::size_t
outputElementBytes(int outputType)
{
    if (outputType == DLP_S8 || outputType == DLP_U8) {
        return 1;
    }
    if (outputType == DLP_F16 || outputType == DLP_BF16) {
        return 2;
    }
    return 4;
}

TEST(GemmStoreRoutingRuntime, U8S8FirstIntermediateAndLastK)
{
    if (test_jit_utils::ArchBasedKernelTypes::getAllKernelTypesForU8S8()
            .empty()) {
        GTEST_SKIP() << "Requires AVX-512 VNNI";
    }

    for (int outputType :
         { DLP_S32, DLP_S8, DLP_U8, DLP_F32, DLP_F16, DLP_BF16 }) {
        SCOPED_TRACE(outputType);

        generatorParams params(1, kLanes, 1, 0, outputType, 0, false, false,
                               false, scalingType::zero, scalingType::generic,
                               kType);
        amdzen::gen::jitU8S8VNNI_GEMM<kType> generator(
            amdzen::utils::JIT_KERNEL_SIZE);
        ASSERT_EQ(generator.generateKernel(params),
                  dlp::jit::jitGeneratorError::success);
        generator.ready();
        auto kernel = generator.getCode<GemmKernel>();
        ASSERT_NE(kernel, nullptr);

        GuardedBuffer<kLanes * sizeof(int32_t)> c;
        GuardedBuffer<kLanes * sizeof(float)>   downscale;
        std::fill_n(c.as<int32_t>(), kLanes, 9);
        fillTyped(downscale.data(), outputType, 2);

        int32_t alpha = 0;
        int32_t beta  = 1;
        auto    runtime =
            makeRuntime(c.data(), downscale.data(), outputType, &alpha, &beta);

        runtime.kernelOpsAttr.is_first_k = 1;
        runtime.kernelOpsAttr.is_last_k  = 0;
        kernel(&runtime);
        if (outputType == DLP_S32) {
            expectTyped(c.data(), DLP_S32, 9);
            expectTyped(downscale.data(), DLP_S32, 2);
        } else {
            expectTyped(c.data(), DLP_S32, 2);
            expectTyped(downscale.data(), outputType, 2);
        }

        fillTyped(downscale.data(), outputType, 4);
        runtime.kernelOpsAttr.is_first_k = 0;
        kernel(&runtime);
        if (outputType == DLP_S32) {
            expectTyped(c.data(), DLP_S32, 9);
            expectTyped(downscale.data(), DLP_S32, 4);
        } else {
            expectTyped(c.data(), DLP_S32, 2);
            expectTyped(downscale.data(), outputType, 4);
        }

        runtime.kernelOpsAttr.is_last_k = 1;
        kernel(&runtime);
        if (outputType == DLP_S32) {
            expectTyped(c.data(), DLP_S32, 9);
            expectTyped(downscale.data(), DLP_S32, 4);
        } else {
            expectTyped(c.data(), DLP_S32, 2);
            expectTyped(downscale.data(), outputType, 2);
        }

        c.expectGuardsIntact();
        downscale.expectGuardsIntact();
        downscale.expectPayloadTailIntact(kLanes
                                          * outputElementBytes(outputType));
    }
}

TEST(GemmStoreRoutingRuntime, BF16NonNullDownscaleRouting)
{
    if (test_jit_utils::ArchBasedKernelTypes::getAllKernelTypesForBF16()
            .empty()) {
        GTEST_SKIP() << "Requires AVX-512 BF16";
    }

    generatorParams params(1, kLanes, 1, 0, DLP_BF16, 0, false, false, false,
                           scalingType::zero, scalingType::generic, kType);
    amdzen::GEMMcodeGenerator::jitGEMMBF16<kType> generator(
        amdzen::utils::JIT_KERNEL_SIZE);
    ASSERT_EQ(generator.generateKernel(params),
              dlp::jit::jitGeneratorError::success);
    generator.ready();
    auto kernel = generator.getCode<GemmKernel>();
    ASSERT_NE(kernel, nullptr);

    GuardedBuffer<kLanes * sizeof(float)>    c;
    GuardedBuffer<kLanes * sizeof(uint16_t)> downscale;
    float                                    alpha = 0.0f;
    float                                    beta  = 1.0f;

    std::fill_n(c.as<float>(), kLanes, 9.0f);
    fillTyped(downscale.data(), DLP_BF16, 2);
    auto runtime =
        makeRuntime(c.data(), downscale.data(), DLP_BF16, &alpha, &beta);
    runtime.kernelOpsAttr.is_first_k = 1;
    runtime.kernelOpsAttr.is_last_k  = 0;
    kernel(&runtime);
    expectTyped(c.data(), DLP_F32, 2);
    expectTyped(downscale.data(), DLP_BF16, 2);

    beta = 0.0f;
    std::fill_n(c.as<float>(), kLanes, 9.0f);
    fillTyped(downscale.data(), DLP_BF16, 4);
    runtime = makeRuntime(c.data(), downscale.data(), DLP_BF16, &alpha, &beta);
    runtime.kernelOpsAttr.is_first_k = 1;
    runtime.kernelOpsAttr.is_last_k  = 1;
    kernel(&runtime);
    expectTyped(c.data(), DLP_F32, 9);
    expectTyped(downscale.data(), DLP_BF16, 0);

    c.expectGuardsIntact();
    downscale.expectGuardsIntact();
}

TEST(GemmStoreRoutingRuntime, BF16NullDownscaleFallback)
{
    if (test_jit_utils::ArchBasedKernelTypes::getAllKernelTypesForBF16()
            .empty()) {
        GTEST_SKIP() << "Requires AVX-512 BF16";
    }

    generatorParams params(1, kLanes, 1, 0, DLP_BF16, 0, false, false, false,
                           scalingType::zero, scalingType::generic, kType);
    amdzen::GEMMcodeGenerator::jitGEMMBF16<kType> generator(
        amdzen::utils::JIT_KERNEL_SIZE);
    ASSERT_EQ(generator.generateKernel(params),
              dlp::jit::jitGeneratorError::success);
    generator.ready();
    auto kernel = generator.getCode<GemmKernel>();
    ASSERT_NE(kernel, nullptr);

    GuardedBuffer<kLanes * sizeof(float)> c;
    float                                 alpha = 0.0f;
    float                                 beta  = 0.0f;

    std::fill_n(c.as<float>(), kLanes, 9.0f);
    auto runtime = makeRuntime(c.data(), nullptr, DLP_BF16, &alpha, &beta);
    runtime.kernelOpsAttr.is_first_k = 1;
    runtime.kernelOpsAttr.is_last_k  = 1;
    kernel(&runtime);
    expectTyped(c.data(), DLP_F32, 0);

    beta = 1.0f;
    std::fill_n(c.as<float>(), kLanes, 9.0f);
    runtime = makeRuntime(c.data(), nullptr, DLP_BF16, &alpha, &beta);
    runtime.kernelOpsAttr.is_first_k = 1;
    runtime.kernelOpsAttr.is_last_k  = 0;
    kernel(&runtime);
    expectTyped(c.data(), DLP_F32, 9);
    c.expectGuardsIntact();
}

} // namespace
