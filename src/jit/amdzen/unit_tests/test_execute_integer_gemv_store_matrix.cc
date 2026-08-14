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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "jit_generator_tests_utils.hh"

namespace {

using amdzen::utils::gemvM1GeneratorParams;
using amdzen::utils::gemvN1GeneratorParams;
using amdzen::utils::jit_gemv_m1_kernel;
using amdzen::utils::jit_gemv_n1_kernel;
using amdzen::utils::kernelInstrType;
using dlp::kernel_frame::scalingType;
using dlp::kernel_frame::storageFormat;

constexpr auto        kType         = kernelInstrType::avx512_zmm_32_reg;
constexpr uint8_t     kSentinel     = 0xA5;
constexpr std::size_t kGuardBytes   = 64;
constexpr std::size_t kPayloadBytes = 320;

enum class Frontend
{
    u8s8,
    s8,
};

struct ExecutionCase
{
    Frontend frontend;
    int      destination;
    bool     withPostOp;
    bool     fringe;
    bool     rowStrided;
};

class GuardedBuffer
{
  public:
    GuardedBuffer() { bytes.fill(kSentinel); }

    void* data() { return bytes.data() + kGuardBytes; }

    void expectGuardsIntact() const
    {
        EXPECT_TRUE(
            std::all_of(bytes.begin(), bytes.begin() + kGuardBytes,
                        [](uint8_t value) { return value == kSentinel; }));
        EXPECT_TRUE(
            std::all_of(bytes.end() - kGuardBytes, bytes.end(),
                        [](uint8_t value) { return value == kSentinel; }));
    }

    void expectUnchangedPayload() const
    {
        EXPECT_TRUE(
            std::all_of(bytes.begin() + kGuardBytes, bytes.end() - kGuardBytes,
                        [](uint8_t value) { return value == kSentinel; }));
    }

    void expectZeroElements(int destination, int count, int stride) const
    {
        const std::size_t elementBytes =
            destination == DLP_S8 || destination == DLP_U8      ? 1
            : destination == DLP_F16 || destination == DLP_BF16 ? 2
                                                                : 4;
        const auto* payload = bytes.data() + kGuardBytes;
        for (int element = 0; element < count; ++element) {
            const int index = element * stride;
            if (destination == DLP_S32) {
                EXPECT_EQ(reinterpret_cast<const int32_t*>(payload)[index], 0);
            } else if (destination == DLP_S8) {
                EXPECT_EQ(reinterpret_cast<const int8_t*>(payload)[index], 0);
            } else if (destination == DLP_U8) {
                EXPECT_EQ(reinterpret_cast<const uint8_t*>(payload)[index], 0);
            } else if (destination == DLP_F32) {
                EXPECT_EQ(reinterpret_cast<const float*>(payload)[index], 0.0f);
            } else {
                EXPECT_EQ(reinterpret_cast<const uint16_t*>(payload)[index], 0);
            }
        }

        for (std::size_t byte = 0; byte < kPayloadBytes; ++byte) {
            bool written = false;
            for (int element = 0; element < count; ++element) {
                const std::size_t begin =
                    static_cast<std::size_t>(element * stride) * elementBytes;
                written = written
                          || (byte >= begin && byte < begin + elementBytes);
            }
            EXPECT_EQ(payload[byte], written ? 0 : kSentinel)
                << "payload byte " << byte;
        }
    }

  private:
    alignas(64) std::array<uint8_t, kPayloadBytes + 2 * kGuardBytes> bytes{};
};

const char*
destinationName(int destination)
{
    switch (destination) {
        case DLP_S32:
            return "S32";
        case DLP_S8:
            return "S8";
        case DLP_U8:
            return "U8";
        case DLP_F32:
            return "F32";
        case DLP_F16:
            return "F16";
        case DLP_BF16:
            return "BF16";
        default:
            return "Unknown";
    }
}

std::string
caseName(const ::testing::TestParamInfo<ExecutionCase>& info)
{
    const auto& config = info.param;
    return std::string(config.frontend == Frontend::u8s8 ? "U8S8" : "S8")
           + destinationName(config.destination)
           + (config.withPostOp ? "F32Source" : "S32Source")
           + (config.fringe ? "Fringe" : "Full")
           + (config.rowStrided ? "RowStrided" : "Contiguous");
}

std::vector<ExecutionCase>
makeCases(bool includeRowStrided)
{
    std::vector<ExecutionCase> cases;
    for (const auto frontend : { Frontend::u8s8, Frontend::s8 }) {
        for (const int destination :
             { DLP_S32, DLP_S8, DLP_U8, DLP_F32, DLP_F16, DLP_BF16 }) {
            for (const bool withPostOp : { false, true }) {
                for (const bool fringe : { false, true }) {
                    cases.push_back(
                        { frontend, destination, withPostOp, fringe, false });
                    if (includeRowStrided) {
                        cases.push_back({ frontend, destination, withPostOp,
                                          fringe, true });
                    }
                }
            }
        }
    }
    return cases;
}

dlp_gemm_post_op
makeReluPostOp()
{
    dlp_gemm_post_op postOp{};
    postOp.op_code = POST_OPS_RELU;
    return postOp;
}

template<class Generator>
void
executeM1(const ExecutionCase& config)
{
    constexpr int fullElements   = 64;
    constexpr int fringeElements = 7;
    const int activeElements = config.fringe ? fringeElements : fullElements;

    gemvM1GeneratorParams genParams(
        config.destination, fullElements, config.fringe ? fringeElements : 0, 4,
        4, PACK, !config.fringe, false, config.fringe, false,
        storageFormat::rowMajor, scalingType::zero, scalingType::zero, kType);
    genParams.N_LEFT_16    = 0;
    genParams.N_LEFT_LT16  = config.fringe ? fringeElements : 0;
    genParams.nfringe_main = false;
    genParams.nfringe_left = config.fringe;
    if (config.withPostOp) {
        genParams.kernelOps =
            test_jit_utils::KernelOpsBuilder().addReLU().build();
    }

    Generator generator(amdzen::utils::JIT_KERNEL_SIZE);
    ASSERT_EQ(generator.generateKernel(genParams),
              dlp::jit::jitGeneratorError::success);
    ASSERT_NO_THROW(generator.ready());
    const auto kernel = generator.template getCode<jit_gemv_m1_kernel>();
    ASSERT_NE(kernel, nullptr);

    std::array<uint8_t, 64> x{};
    std::array<int8_t, 256> b{};
    GuardedBuffer           scratch;
    GuardedBuffer           output;
    int32_t                 alpha  = 0;
    int32_t                 beta   = 0;
    auto                    postOp = makeReluPostOp();

    dlp_gemm_post_op_attr attr{};
    attr.buf_downscale = output.data();
    attr.is_first_k    = 1;
    attr.is_last_k     = 1;
    attr.c_stor_type   = config.destination;

    void* y = config.destination == DLP_S32 ? output.data() : scratch.data();
    dlp::kernels::gemvM1Params runtime(
        x.data(), b.data(), y, activeElements, 0, 1, 1, 64, 1, 1, 1, 0, 0,
        &alpha, &beta, config.withPostOp ? &postOp : nullptr, attr);
    runtime.n_iter       = config.fringe ? 0 : 1;
    runtime.n_left       = config.fringe ? fringeElements : 0;
    runtime.n_left_lt16  = runtime.n_left;
    runtime.nmask_avx512 = config.fringe ? 0x7F : 0xFFFF;

    ASSERT_NO_FATAL_FAILURE(kernel(&runtime));

    output.expectZeroElements(config.destination, activeElements, 1);
    output.expectGuardsIntact();
    scratch.expectGuardsIntact();
    if (config.destination != DLP_S32) {
        scratch.expectUnchangedPayload();
    }
}

template<class Generator>
void
executeN1(const ExecutionCase& config)
{
    constexpr int fullElements   = 16;
    constexpr int fringeElements = 7;
    constexpr int rowStride      = 3;
    const int  activeElements = config.fringe ? fringeElements : fullElements;
    const int  outputStride   = config.rowStrided ? rowStride : 1;
    const auto format         = config.rowStrided ? storageFormat::rowMajor
                                                  : storageFormat::colMajor;

    gemvN1GeneratorParams genParams(
        fullElements, config.fringe ? fringeElements : 0, config.destination,
        !config.fringe, false, config.fringe, false, format, scalingType::zero,
        scalingType::zero, kType);
    if (config.withPostOp) {
        genParams.kernelOps =
            test_jit_utils::KernelOpsBuilder().addReLU().build();
    }

    Generator generator(amdzen::utils::JIT_KERNEL_SIZE);
    ASSERT_EQ(generator.generateKernel(genParams),
              dlp::jit::jitGeneratorError::success);
    ASSERT_NO_THROW(generator.ready());
    const auto kernel = generator.template getCode<jit_gemv_n1_kernel>();
    ASSERT_NE(kernel, nullptr);

    std::array<uint8_t, 64> a{};
    std::array<int8_t, 64>  x{};
    GuardedBuffer           scratch;
    GuardedBuffer           output;
    int32_t                 alpha  = 0;
    int32_t                 beta   = 0;
    auto                    postOp = makeReluPostOp();

    dlp_gemm_post_op_attr attr{};
    attr.rs_c_downscale = outputStride;
    attr.cs_c_downscale = 1;
    attr.buf_downscale  = output.data();
    attr.is_first_k     = 1;
    attr.is_last_k      = 1;
    attr.c_stor_type    = config.destination;

    void* y = config.destination == DLP_S32 ? output.data() : scratch.data();
    dlp::kernels::gemvN1Params runtime(
        a.data(), x.data(), y, activeElements, 0, 64, 1, 1, 1, outputStride, 1,
        &alpha, &beta, config.withPostOp ? &postOp : nullptr, attr);
    runtime.m_iter          = config.fringe ? 0 : 1;
    runtime.m_left          = config.fringe ? fringeElements : 0;
    runtime.mmask_avx512    = config.fringe ? 0x7F : 0xFFFF;
    runtime.kmask_i8_avx512 = 0;

    ASSERT_NO_FATAL_FAILURE(kernel(&runtime));

    output.expectZeroElements(config.destination, activeElements, outputStride);
    output.expectGuardsIntact();
    scratch.expectGuardsIntact();
    if (config.destination != DLP_S32) {
        scratch.expectUnchangedPayload();
    }
}

class IntegerGemvM1Execution : public ::testing::TestWithParam<ExecutionCase>
{};

TEST_P(IntegerGemvM1Execution, ProductionGeneratorStoresTypedResults)
{
    if (test_jit_utils::ArchBasedKernelTypes::getAllKernelTypesForU8S8()
            .empty()) {
        GTEST_SKIP() << "Requires AVX-512 VNNI";
    }

    const auto& config = GetParam();
    if (config.frontend == Frontend::u8s8) {
        executeM1<amdzen::gen::jitU8S8VNNI_GEMVM1<kType>>(config);
    } else {
        executeM1<amdzen::gen::jitGEMVS8M1<kType>>(config);
    }
}

// The production M1 adapter exposes only contiguous stores along N. Strided
// output is therefore covered through the production N1 scalar-store path.
INSTANTIATE_TEST_SUITE_P(AllFrontendsDestinationsSourcesAndShapes,
                         IntegerGemvM1Execution,
                         ::testing::ValuesIn(makeCases(false)),
                         caseName);

class IntegerGemvN1Execution : public ::testing::TestWithParam<ExecutionCase>
{};

TEST_P(IntegerGemvN1Execution, ProductionGeneratorStoresTypedResults)
{
    if (test_jit_utils::ArchBasedKernelTypes::getAllKernelTypesForU8S8()
            .empty()) {
        GTEST_SKIP() << "Requires AVX-512 VNNI";
    }

    const auto& config = GetParam();
    if (config.frontend == Frontend::u8s8) {
        executeN1<amdzen::gen::jitU8S8VNNI_GEMVN1<kType>>(config);
    } else {
        executeN1<amdzen::gen::jitGEMVS8N1<kType>>(config);
    }
}

INSTANTIATE_TEST_SUITE_P(AllFrontendsDestinationsSourcesShapesAndLayouts,
                         IntegerGemvN1Execution,
                         ::testing::ValuesIn(makeCases(true)),
                         caseName);

} // namespace
