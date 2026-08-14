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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

#include "jit/amdzen/store/gemv_m1_store_emitter.hh"
#include "jit/amdzen/store/gemv_n1_store_emitter.hh"
#include "jit_generator_tests_utils.hh"

namespace {

using namespace amdzen::store;
using amdzen::utils::JIT_KERNEL_SIZE;
using amdzen::utils::kernelInstrType;
using dlp::kernel_frame::DataType;

constexpr auto kType = kernelInstrType::avx512_zmm_32_reg;

bool
hasAvx512Support()
{
    const auto types =
        test_jit_utils::ArchBasedKernelTypes::getAllKernelTypesForF32();
    return std::find(types.begin(), types.end(), kType) != types.end();
}

bool
hasAvx512Bf16Support()
{
    const auto types =
        test_jit_utils::ArchBasedKernelTypes::getAllKernelTypesForBF16();
    return std::find(types.begin(), types.end(), kType) != types.end();
}

uint16_t
toBf16Rne(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

template<bool Bf16>
void
runM1Store(SourceRegWidth view, int validValues)
{
    if (!hasAvx512Support() || (Bf16 && !hasAvx512Bf16Support())) {
        GTEST_SKIP() << (Bf16 ? "Requires AVX-512 BF16" : "Requires AVX-512");
    }

    using Output              = std::conditional_t<Bf16, uint16_t, float>;
    constexpr Output sentinel = static_cast<Output>(Bf16 ? 0xA5A5 : -99);
    const int        capacity = view == SourceRegWidth::zmm ? 16 : 8;

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 4, 1);
        const auto              sourcePtr    = frame.p[0];
        const auto              regYptr      = frame.p[1];
        const auto              yPtrResult   = frame.p[2];
        const auto              sourceResult = frame.p[3];
        const auto              maskValue    = frame.t[0];
        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);

        auto finalMask = StoreMask::none();
        if (validValues != capacity) {
            jit.mov(maskValue.cvt32(), (1u << validValues) - 1u);
            jit.kmovw(Xbyak::Opmask(1), maskValue.cvt32());
            finalMask = StoreMask::opmask(Xbyak::Opmask(1));
        }
        const GemvM1StoreRequest request{
            RegSpan{ 0, view, validValues },
            { regYptr, finalMask },
            Bf16 ? StoreSpec::nativeBf16(DataType::f32)
                 : StoreSpec::convert(DataType::f32, DataType::f32),
            StoreTemps::none()
        };
        GemvM1StoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.mov(Xbyak::util::ptr[yPtrResult], regYptr);
        if constexpr (Bf16) {
            jit.vmovdqu16(Xbyak::util::ptr[sourceResult], Xbyak::Ymm(0));
        } else {
            jit.vmovups(Xbyak::util::ptr[sourceResult], Xbyak::Zmm(0));
        }
    }
    jit.ready();

    std::array<float, 16> source{};
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<float>(i) + 0.3125f;
    }
    std::array<Output, 18> output{};
    output.fill(sentinel);
    std::array<Output, 16> sourceResult{};
    sourceResult.fill(sentinel);
    std::uintptr_t yPtrResult = 0;
    using Kernel = void (*)(const float*, Output*, std::uintptr_t*, Output*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data() + 1, &yPtrResult, sourceResult.data());

    EXPECT_EQ(output.front(), sentinel);
    EXPECT_EQ(output.back(), sentinel);
    for (int i = 0; i < capacity; ++i) {
        const Output expected = Bf16 ? static_cast<Output>(toBf16Rne(source[i]))
                                     : static_cast<Output>(source[i]);
        EXPECT_EQ(output[i + 1], i < validValues ? expected : sentinel);
        EXPECT_EQ(sourceResult[i], expected);
    }
    if constexpr (!Bf16) {
        for (size_t i = capacity; i < source.size(); ++i) {
            EXPECT_EQ(sourceResult[i], source[i]);
        }
    }

    auto* expectedYPtr = output.data() + 1;
    if constexpr (Bf16) {
        if (view == SourceRegWidth::zmm && validValues == capacity) {
            expectedYPtr += capacity;
        }
    }
    EXPECT_EQ(yPtrResult, reinterpret_cast<std::uintptr_t>(expectedYPtr));
}

template<bool Bf16>
void
runN1ContiguousStore()
{
    if (!hasAvx512Support() || (Bf16 && !hasAvx512Bf16Support())) {
        GTEST_SKIP() << (Bf16 ? "Requires AVX-512 BF16" : "Requires AVX-512");
    }

    using Output              = std::conditional_t<Bf16, uint16_t, float>;
    constexpr Output sentinel = static_cast<Output>(Bf16 ? 0xA5A5 : -99);

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 4, 0);
        const auto              sourcePtr    = frame.p[0];
        const auto              regYptr      = frame.p[1];
        const auto              yPtrResult   = frame.p[2];
        const auto              sourceResult = frame.p[3];
        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);

        const GemvN1StoreRequest request{
            GemvN1Source::packedLanes(RegSpan{ 0, SourceRegWidth::zmm, 16 }),
            GemvN1Destination::contiguousAdvanceComplete(regYptr, sourcePtr,
                                                         StoreMask::none()),
            Bf16 ? StoreSpec::nativeBf16(DataType::f32)
                 : StoreSpec::convert(DataType::f32, DataType::f32),
            StoreTemps::none()
        };
        GemvN1StoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.mov(Xbyak::util::ptr[yPtrResult], regYptr);
        if constexpr (Bf16) {
            jit.vmovdqu16(Xbyak::util::ptr[sourceResult], Xbyak::Ymm(0));
        } else {
            jit.vmovups(Xbyak::util::ptr[sourceResult], Xbyak::Zmm(0));
        }
    }
    jit.ready();

    std::array<float, 16> source{};
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<float>(i) + 0.4375f;
    }
    std::array<Output, 18> output{};
    output.fill(sentinel);
    std::array<Output, 16> sourceResult{};
    sourceResult.fill(sentinel);
    std::uintptr_t yPtrResult = 0;
    using Kernel = void (*)(const float*, Output*, std::uintptr_t*, Output*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data() + 1, &yPtrResult, sourceResult.data());

    EXPECT_EQ(output.front(), sentinel);
    EXPECT_EQ(output.back(), sentinel);
    for (size_t i = 0; i < source.size(); ++i) {
        const Output expected = Bf16 ? static_cast<Output>(toBf16Rne(source[i]))
                                     : static_cast<Output>(source[i]);
        EXPECT_EQ(output[i + 1], expected);
        EXPECT_EQ(sourceResult[i], expected);
    }
    EXPECT_EQ(yPtrResult, reinterpret_cast<std::uintptr_t>(output.data() + 17));
}

enum class PreservedContiguousType
{
    f32,
    bf16,
    s32,
};

template<PreservedContiguousType Type>
void
runN1PreservedContiguousTwoRegisterStore()
{
    constexpr bool isBf16 = Type == PreservedContiguousType::bf16;
    constexpr bool isS32  = Type == PreservedContiguousType::s32;
    if (!hasAvx512Support() || (isBf16 && !hasAvx512Bf16Support())) {
        GTEST_SKIP() << (isBf16 ? "Requires AVX-512 BF16" : "Requires AVX-512");
    }

    using Source = std::conditional_t<isS32, int32_t, float>;
    using Output =
        std::conditional_t<isBf16, uint16_t,
                           std::conditional_t<isS32, int32_t, float>>;
    constexpr Output sentinel = [] {
        if constexpr (isBf16) {
            return static_cast<Output>(0xA5A5);
        } else if constexpr (isS32) {
            return static_cast<Output>(0x5A5A5A5A);
        } else {
            return static_cast<Output>(-99.0f);
        }
    }();

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 3, 0);
        const auto              sourcePtr  = frame.p[0];
        const auto              regYptr    = frame.p[1];
        const auto              yPtrResult = frame.p[2];
        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.vmovups(Xbyak::Zmm(1), Xbyak::util::ptr[sourcePtr + 64]);

        const auto store = [&] {
            if constexpr (isBf16) {
                return StoreSpec::nativeBf16(DataType::f32);
            } else if constexpr (isS32) {
                return StoreSpec::convert(DataType::s32, DataType::s32);
            } else {
                return StoreSpec::convert(DataType::f32, DataType::f32);
            }
        }();
        const GemvN1StoreRequest request{
            GemvN1Source::packedLanes(RegSpan{ 0, SourceRegWidth::zmm, 32 }),
            GemvN1Destination::contiguous(regYptr, StoreMask::none()), store,
            StoreTemps::none()
        };
        GemvN1StoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.mov(Xbyak::util::ptr[yPtrResult], regYptr);
    }
    jit.ready();

    std::array<Source, 32> source{};
    for (size_t i = 0; i < source.size(); ++i) {
        if constexpr (isS32) {
            source[i] = static_cast<int32_t>(i * 17) - 100;
        } else {
            source[i] = static_cast<float>(i) + 0.375f;
        }
    }
    std::array<Output, 34> output{};
    output.fill(sentinel);
    std::uintptr_t yPtrResult = 0;
    using Kernel = void (*)(const Source*, Output*, std::uintptr_t*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data() + 1, &yPtrResult);

    EXPECT_EQ(output.front(), sentinel);
    EXPECT_EQ(output.back(), sentinel);
    for (size_t i = 0; i < source.size(); ++i) {
        const Output expected = [&] {
            if constexpr (isBf16) {
                return static_cast<Output>(toBf16Rne(source[i]));
            } else {
                return static_cast<Output>(source[i]);
            }
        }();
        EXPECT_EQ(output[i + 1], expected) << "element " << i;
    }
    EXPECT_EQ(yPtrResult, reinterpret_cast<std::uintptr_t>(output.data() + 1));
}

template<bool Bf16>
void
runN1PackedScalarStore(SourceRegWidth view)
{
    if (!hasAvx512Support() || (Bf16 && !hasAvx512Bf16Support())) {
        GTEST_SKIP() << (Bf16 ? "Requires AVX-512 BF16" : "Requires AVX-512");
    }

    using Output                 = std::conditional_t<Bf16, uint16_t, float>;
    constexpr Output sentinel    = static_cast<Output>(Bf16 ? 0xA5A5 : -99);
    constexpr int    validValues = 5;

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 4, 1);
        const auto              sourcePtr    = frame.p[0];
        const auto              regYptr      = frame.p[1];
        const auto              yPtrResult   = frame.p[2];
        const auto              sourceResult = frame.p[3];
        const auto              stride       = frame.t[0];
        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.mov(stride, 2 * sizeof(Output));

        const auto               scratch = StoreTemps::withZmm(1);
        const GemvN1StoreRequest request{
            GemvN1Source::packedLanes(RegSpan{ 0, view, validValues }),
            GemvN1Destination::scalarStrided(regYptr, stride),
            Bf16 ? StoreSpec::nativeBf16(DataType::f32)
                 : StoreSpec::convert(DataType::f32, DataType::f32),
            scratch
        };
        GemvN1StoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.mov(Xbyak::util::ptr[yPtrResult], regYptr);
        jit.vmovups(Xbyak::util::ptr[sourceResult], Xbyak::Zmm(0));
    }
    jit.ready();

    std::array<float, 16> source{};
    std::array<float, 16> sourceResult{};
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<float>(i) + 0.5625f;
    }
    sourceResult.fill(-1.0f);
    std::array<Output, 12> output{};
    output.fill(sentinel);
    std::uintptr_t yPtrResult = 0;
    using Kernel = void (*)(const float*, Output*, std::uintptr_t*, float*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data() + 1, &yPtrResult, sourceResult.data());

    EXPECT_EQ(output.front(), sentinel);
    EXPECT_EQ(output.back(), sentinel);
    for (int i = 0; i < validValues; ++i) {
        const Output expected = Bf16 ? static_cast<Output>(toBf16Rne(source[i]))
                                     : static_cast<Output>(source[i]);
        EXPECT_EQ(output[1 + i * 2], expected);
        EXPECT_EQ(output[2 + i * 2], sentinel);
    }
    EXPECT_EQ(sourceResult, source);
    EXPECT_EQ(yPtrResult, reinterpret_cast<std::uintptr_t>(output.data() + 11));
}

template<bool Bf16>
void
runGeneratedN1EmptyCompactFringe(int originalSize)
{
    if (!hasAvx512Bf16Support()) {
        GTEST_SKIP() << "Requires AVX-512 BF16";
    }

    using Output                   = std::conditional_t<Bf16, uint16_t, float>;
    constexpr Output sentinel      = static_cast<Output>(Bf16 ? 0xA5A5 : -99);
    constexpr int    stride        = 2;
    const int        compactValues = (originalSize / 16) * 8;

    amdzen::utils::gemvN1GeneratorParams params(
        originalSize == 1 ? 16 : 24, originalSize, Bf16 ? DLP_BF16 : DLP_F32,
        false, false, true, false, dlp::kernel_frame::storageFormat::colMajor,
        dlp::kernel_frame::scalingType::zero,
        dlp::kernel_frame::scalingType::zero, kType);
    params.storeHalfWidthResults = true;

    amdzen::codegen::jitBF16GEMVN1<kType> generator(JIT_KERNEL_SIZE);
    ASSERT_EQ(generator.generateKernel(params),
              dlp::jit::jitGeneratorError::success);
    ASSERT_NO_THROW(generator.ready());
    const auto kernel = generator.getCode<amdzen::utils::jit_gemv_n1_kernel>();
    ASSERT_NE(kernel, nullptr);

    std::array<uint16_t, 64> a{};
    std::array<uint16_t, 32> x{};
    std::array<float, 24>    y{};
    std::array<Output, 48>   output{};
    output.fill(sentinel);
    float alpha = 0.0f;
    float beta  = 0.0f;

    dlp_gemm_post_op_attr attr{};
    attr.post_op_c_i = 2;
    attr.post_op_c_j = 1;
    attr.is_last_k   = 1;
    attr.c_stor_type = Bf16 ? DLP_BF16 : DLP_F32;
    attr.buf_d       = output.data();
    attr.ld_d        = stride;
    dlp::kernels::gemvN1Params runtime(a.data(), x.data(), y.data(),
                                       originalSize, 0, 1, 1, 1, 1, 1, 1,
                                       &alpha, &beta, nullptr, attr);
    runtime.m_left       = originalSize;
    runtime.mmask_avx512 = originalSize == 1 ? 0x1 : 0x1FFFF;

    ASSERT_NO_FATAL_FAILURE(kernel(&runtime));

    const int base = 1 + stride;
    for (int i = 0; i < static_cast<int>(output.size()); ++i) {
        const bool written = i >= base && i < base + compactValues * stride
                             && (i - base) % stride == 0;
        EXPECT_EQ(output[i], written ? Output{} : sentinel) << "index " << i;
    }
}

template<bool Bf16>
void
runGeneratedM1EmptyCompactFringe(int originalSize)
{
    if (!hasAvx512Bf16Support()) {
        GTEST_SKIP() << "Requires AVX-512 BF16";
    }

    using Output                   = std::conditional_t<Bf16, uint16_t, float>;
    constexpr Output sentinel      = static_cast<Output>(Bf16 ? 0xA5A5 : -99);
    const int        compactValues = (originalSize / 16) * 8;

    amdzen::utils::gemvM1GeneratorParams params(
        Bf16 ? DLP_BF16 : DLP_F32, originalSize, 0, 4, 4, PACK, true, false,
        false, false, dlp::kernel_frame::storageFormat::rowMajor,
        dlp::kernel_frame::scalingType::zero,
        dlp::kernel_frame::scalingType::zero, kType);
    params.nfringe_main          = false;
    params.nfringe_left          = false;
    params.storeHalfWidthResults = true;

    amdzen::codegen::jitBF16GEMVM1<kType> generator(JIT_KERNEL_SIZE);
    ASSERT_EQ(generator.generateKernel(params),
              dlp::jit::jitGeneratorError::success);
    ASSERT_NO_THROW(generator.ready());
    const auto kernel = generator.getCode<amdzen::utils::jit_gemv_m1_kernel>();
    ASSERT_NE(kernel, nullptr);

    std::array<uint16_t, 8>  x{};
    std::array<uint16_t, 64> b{};
    std::array<float, 40>    y{};
    std::array<Output, 40>   output{};
    output.fill(sentinel);
    float alpha = 0.0f;
    float beta  = 0.0f;

    dlp_gemm_post_op_attr attr{};
    attr.post_op_c_j = 2;
    attr.is_last_k   = 1;
    attr.c_stor_type = Bf16 ? DLP_BF16 : DLP_F32;
    attr.buf_d       = output.data();
    attr.ld_d        = output.size();
    dlp::kernels::gemvM1Params runtime(
        x.data(), b.data(), y.data(), originalSize, 0, 1, 1, originalSize, 1, 1,
        1, 0, 0, &alpha, &beta, nullptr, attr);
    runtime.n_iter = 1;
    runtime.nmask_avx512 =
        originalSize % 16 == 0 ? 0xFFFF : (1u << (originalSize % 16)) - 1u;

    ASSERT_NO_FATAL_FAILURE(kernel(&runtime));

    constexpr int base = 1;
    for (int i = 0; i < static_cast<int>(output.size()); ++i) {
        const bool written = i >= base && i < base + compactValues;
        EXPECT_EQ(output[i], written ? Output{} : sentinel) << "index " << i;
    }
}

void
runN1F16ContiguousStore(int  validValues,
                        bool advance,
                        bool useRaxAdvance,
                        int  expectedCursorOffset)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }

    constexpr std::size_t strideBytes = 48;
    Xbyak::CodeGenerator  jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 4, 2);
        const auto              sourcePtr    = frame.p[0];
        const auto              cursor       = frame.p[1];
        const auto              cursorResult = frame.p[2];
        const auto              advanceBytes = frame.p[3];
        const auto              advanceTemp  = frame.t[0];
        const auto              maskTemp     = frame.t[1];
        jit.vmovups(Xbyak::Zmm(8), Xbyak::util::ptr[sourcePtr]);
        jit.vmovups(Xbyak::Zmm(9), Xbyak::util::ptr[sourcePtr + 64]);

        auto storeMask = StoreMask::none();
        if (validValues != 32) {
            jit.mov(maskTemp.cvt32(), (1u << (validValues - 16)) - 1u);
            jit.kmovw(Xbyak::Opmask(1), maskTemp.cvt32());
            storeMask = StoreMask::opmask(Xbyak::Opmask(1));
        }

        Xbyak::Reg64 advanceReg = advanceTemp;
        if (advance && useRaxAdvance) {
            advanceReg = Xbyak::Reg64(0);
        }
        jit.mov(advanceReg, advanceBytes);
        auto destination =
            advance ? GemvN1Destination::contiguousAdvanceComplete(
                          cursor, advanceReg, storeMask)
                    : GemvN1Destination::contiguous(cursor, storeMask);
        if (!advance) {
            destination.regAdvanceBytes = advanceReg;
        }

        const std::array<int, 1> scratchRegisters{ 0 };
        const auto scratch = StoreTemps::withZmm(scratchRegisters[0]);
        const GemvN1StoreRequest request{
            GemvN1Source::packedLanes(
                RegSpan{ 8, SourceRegWidth::zmm, validValues }),
            destination, StoreSpec::convert(DataType::f32, DataType::f16),
            scratch
        };
        GemvN1StoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.mov(Xbyak::util::ptr[cursorResult], cursor);
    }
    jit.ready();

    std::array<float, 32> source{};
    source.fill(1.0f);
    std::array<uint16_t, 64> output{};
    output.fill(0xA5A5);
    std::uintptr_t cursorResult = 0;
    using Kernel =
        void (*)(const float*, uint16_t*, std::uintptr_t*, std::size_t);
    auto kernel = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data(), &cursorResult, strideBytes);

    for (int value = 0; value < validValues; ++value) {
        const int byteOffset = advance ? (value / 16) * strideBytes
                                       : (value / 16) * 32;
        EXPECT_EQ(output[byteOffset / 2 + value % 16], 0x3C00);
    }
    for (std::size_t byteOffset = 0; byteOffset < output.size() * 2;
         byteOffset += 2) {
        const bool firstRegister = byteOffset < 32;
        const bool secondRegister =
            byteOffset >= (advance ? strideBytes : 32)
            && byteOffset
                   < (advance ? strideBytes : 32)
                         + static_cast<std::size_t>(validValues - 16) * 2;
        if (!firstRegister && !secondRegister) {
            EXPECT_EQ(output[byteOffset / 2], 0xA5A5);
        }
    }
    EXPECT_EQ(cursorResult, reinterpret_cast<std::uintptr_t>(output.data())
                                + expectedCursorOffset);
}

std::array<float, 16>
int8ClampInputs()
{
    return { (std::numeric_limits<float>::max)(),
             std::numeric_limits<float>::infinity(),
             std::numeric_limits<float>::quiet_NaN(),
             -(std::numeric_limits<float>::max)(),
             -std::numeric_limits<float>::infinity(),
             -129.0f,
             -128.0f,
             -0.5f,
             0.0f,
             0.5f,
             126.5f,
             127.0f,
             127.5f,
             254.5f,
             255.0f,
             255.5f };
}

template<bool U8>
auto
expectedInt8(float value)
{
    using Output        = std::conditional_t<U8, uint8_t, int8_t>;
    const float lower   = U8 ? 0.0f : -128.0f;
    const float upper   = U8 ? 255.0f : 127.0f;
    const float clipped = std::fmax(lower, std::fmin(upper, value));
    return static_cast<Output>(std::nearbyint(clipped));
}

template<bool U8, bool N1>
void
runGemvF32ToInt8Clamp()
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }

    using Output              = std::conditional_t<U8, uint8_t, int8_t>;
    constexpr Output sentinel = static_cast<Output>(U8 ? 0xA5 : 0x55);

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 2, 1);
        const auto              sourcePtr = frame.p[0];
        const auto              outputPtr = frame.p[1];
        const auto              immediate = frame.t[0];
        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);

        const auto store =
            StoreSpec::convert(DataType::f32, U8 ? DataType::u8 : DataType::s8);
        const auto temps =
            StoreTemps::withZmmAndGpr(1, Xbyak::Reg64(immediate.getIdx()));
        if constexpr (N1) {
            GemvN1StoreEmitter<kType> emitter(jit);
            const GemvN1StoreRequest request{ GemvN1Source::packedLanes(RegSpan{
                                                  0, SourceRegWidth::zmm, 16 }),
                                              GemvN1Destination::contiguous(
                                                  outputPtr, StoreMask::none()),
                                              store, temps };
            ASSERT_EQ(emitter.emit(request),
                      dlp::jit::jitGeneratorError::success);
        } else {
            GemvM1StoreEmitter<kType> emitter(jit);
            const GemvM1StoreRequest  request{ RegSpan{ 0, SourceRegWidth::zmm,
                                                       16 },
                                               { outputPtr, StoreMask::none() },
                                              store,
                                              temps };
            ASSERT_EQ(emitter.emit(request),
                      dlp::jit::jitGeneratorError::success);
        }
    }
    jit.ready();

    const auto             source = int8ClampInputs();
    std::array<Output, 18> output{};
    output.fill(sentinel);
    using Kernel      = void (*)(const float*, Output*);
    const auto kernel = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data() + 1);

    EXPECT_EQ(output.front(), sentinel);
    EXPECT_EQ(output.back(), sentinel);
    for (size_t i = 0; i < source.size(); ++i) {
        EXPECT_EQ(output[i + 1], expectedInt8<U8>(source[i])) << "lane " << i;
    }
}

} // namespace

TEST(GemvN1StoreEmitterDispatch, RejectsPackedOnlyExtractedConversions)
{
    using Error = dlp::jit::jitGeneratorError;
    Xbyak::CodeGenerator      jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    GemvN1StoreEmitter<kType> emitter(jit);
    const std::array<int, 2>  chunks{ 2, 3 };
    const auto                source =
        GemvN1Source::extractedXmm(chunks.data(), chunks.size(), 5);
    const auto destination =
        GemvN1Destination::scalarStrided(Xbyak::Reg64(0), Xbyak::Reg64(1));
    const auto expectUnsupported = [&](StoreSpec store) {
        const auto sizeBefore = jit.getSize();
        EXPECT_EQ(
            emitter.emit({ source, destination, store, StoreTemps::none() }),
            Error::notSupported);
        EXPECT_EQ(jit.getSize(), sizeBefore);
    };

    expectUnsupported(StoreSpec::convert(DataType::f32, DataType::s32));
    expectUnsupported(StoreSpec::convert(DataType::f32, DataType::s8));
    expectUnsupported(StoreSpec::convert(DataType::f32, DataType::f16));
    expectUnsupported(StoreSpec::softwareBf16(DataType::f32));
}

TEST(BF16GemvCompactStoreExecution, N1KeepsFullRegisterBeforeEmptyFringe)
{
    runGeneratedN1EmptyCompactFringe<false>(17);
    runGeneratedN1EmptyCompactFringe<true>(17);
    runGeneratedN1EmptyCompactFringe<false>(1);
    runGeneratedN1EmptyCompactFringe<true>(1);
}

TEST(BF16GemvCompactStoreExecution, M1KeepsFullRegistersBeforeEmptyFringe)
{
    for (const int originalSize : { 17, 33, 1 }) {
        SCOPED_TRACE(originalSize);
        runGeneratedM1EmptyCompactFringe<false>(originalSize);
        runGeneratedM1EmptyCompactFringe<true>(originalSize);
    }
}

TEST(GemvN1StoreEmitterDispatch, RejectsExtractedContiguousSource)
{
    using Error = dlp::jit::jitGeneratorError;
    Xbyak::CodeGenerator      jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    GemvN1StoreEmitter<kType> emitter(jit);
    const std::array<int, 1>  chunks{ 2 };
    const GemvN1StoreRequest  request{
        GemvN1Source::extractedXmm(chunks.data(), chunks.size(), 4),
        GemvN1Destination::contiguous(Xbyak::Reg64(0), StoreMask::none()),
        StoreSpec::convert(DataType::f32, DataType::f32), StoreTemps::none()
    };
    const auto sizeBefore = jit.getSize();
    EXPECT_EQ(emitter.emit(request), Error::notSupported);
    EXPECT_EQ(jit.getSize(), sizeBefore);
}

TEST(GemvM1StoreEmitterDispatch, RejectsYmmConversionWithoutEmitting)
{
    Xbyak::CodeGenerator      jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    GemvM1StoreEmitter<kType> emitter(jit);
    const GemvM1StoreRequest  request{ RegSpan{ 0, SourceRegWidth::ymm, 8 },
                                       { Xbyak::Reg64(0), StoreMask::none() },
                                      StoreSpec::convert(DataType::f32,
                                                          DataType::s32),
                                      StoreTemps::none() };
    const auto                sizeBefore = jit.getSize();

    EXPECT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::notSupported);
    EXPECT_EQ(jit.getSize(), sizeBefore);
}

TEST(GemvN1StoreEmitterDispatch, RejectsPackedYmmConversionWithoutEmitting)
{
    Xbyak::CodeGenerator      jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    GemvN1StoreEmitter<kType> emitter(jit);
    const GemvN1StoreRequest  request{
        GemvN1Source::packedLanes(RegSpan{ 0, SourceRegWidth::ymm, 8 }),
        GemvN1Destination::contiguous(Xbyak::Reg64(0), StoreMask::none()),
        StoreSpec::convert(DataType::f32, DataType::f16), StoreTemps::withZmm(1)
    };
    const auto sizeBefore = jit.getSize();

    EXPECT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::notSupported);
    EXPECT_EQ(jit.getSize(), sizeBefore);
}

TEST(GemvN1StoreEmitterEncoding, EmptyU8TailKeepsScratchSetup)
{
    Xbyak::CodeGenerator legacy(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    legacy.vpxord(Xbyak::Zmm(1), Xbyak::Zmm(1), Xbyak::Zmm(1));
    legacy.mov(Xbyak::Reg64(12), 255);
    legacy.vpbroadcastd(Xbyak::Zmm(2), Xbyak::Reg32(12));

    Xbyak::CodeGenerator     production(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    const std::array<int, 3> vectors{ 0, 1, 2 };
    const std::array<int, 1> gprs{ 12 };
    const auto               scratch =
        StoreTemps::withZmmAndGpr(vectors[0], Xbyak::Reg64(gprs[0]));
    const GemvN1StoreRequest request{
        GemvN1Source::packedLanes(RegSpan{ 16, SourceRegWidth::zmm, 0 }),
        GemvN1Destination::contiguous(Xbyak::Reg64(9), StoreMask::none()),
        StoreSpec::convert(DataType::s32, DataType::u8), scratch
    };
    GemvN1StoreEmitter<kType> emitter(production);
    ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
    ASSERT_EQ(production.getSize(), legacy.getSize());
    EXPECT_EQ(
        std::memcmp(production.getCode(), legacy.getCode(), legacy.getSize()),
        0);
}

TEST(GemvN1StoreEmitterExecution, S32ContiguousMaskedPreservesCursor)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }
    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 4, 1);
        const auto              sourcePtr    = frame.p[0];
        const auto              cursor       = frame.p[1];
        const auto              cursorResult = frame.p[2];
        const auto              maskValue    = frame.p[3];
        jit.vmovdqu32(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.kmovw(Xbyak::Opmask(2), maskValue.cvt32());
        const GemvN1StoreRequest request{
            GemvN1Source::packedLanes(RegSpan{ 0, SourceRegWidth::zmm, 5 }),
            GemvN1Destination::contiguous(cursor,
                                          StoreMask::opmask(Xbyak::Opmask(2))),
            StoreSpec::convert(DataType::s32, DataType::s32), StoreTemps::none()
        };
        GemvN1StoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.mov(Xbyak::util::ptr[cursorResult], cursor);
    }
    jit.ready();
    const std::array<int32_t, 16> source{ -8, -7, -6, -5, -4, -3, -2, -1,
                                          0,  1,  2,  3,  4,  5,  6,  7 };
    std::array<int32_t, 16>       output{};
    output.fill(-999);
    std::uintptr_t cursorResult = 0;
    using Kernel =
        void (*)(const int32_t*, int32_t*, std::uintptr_t*, uint16_t);
    auto kernel = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data(), &cursorResult, 0x001F);
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_EQ(output[i], i < 5 ? source[i] : -999);
    }
    EXPECT_EQ(cursorResult, reinterpret_cast<std::uintptr_t>(output.data()));
}

TEST(GemvN1StoreEmitterExecution, F32ToS32ScalarStridedIsDestructive)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }
    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 3, 1);
        const auto              sourcePtr    = frame.p[0];
        const auto              cursor       = frame.p[1];
        const auto              convertedPtr = frame.p[2];
        const auto              stride       = frame.t[0];
        jit.vmovups(Xbyak::Zmm(8), Xbyak::util::ptr[sourcePtr]);
        jit.mov(stride, 2 * sizeof(int32_t));
        const std::array<int, 4> scratchRegisters{ 0, 1, 2, 3 };
        const auto scratch = StoreTemps::withZmm(scratchRegisters[0]);
        const GemvN1StoreRequest request{
            GemvN1Source::packedLanes(RegSpan{ 8, SourceRegWidth::zmm, 5 }),
            GemvN1Destination::scalarStrided(cursor, stride),
            StoreSpec::convert(DataType::f32, DataType::s32), scratch
        };
        GemvN1StoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.vmovdqu32(Xbyak::util::ptr[convertedPtr], Xbyak::Zmm(8));
    }
    jit.ready();
    const std::array<float, 16> source{ -7.5f, -1.5f, -0.5f, 0.5f, 2.5f, 3.5f,
                                        4.5f,  5.5f,  6.5f,  7.5f, 8.5f, 9.5f,
                                        10.5f, 11.5f, 12.5f, 13.5f };
    std::array<int32_t, 11>     output{};
    std::array<int32_t, 16>     converted{};
    output.fill(-999);
    using Kernel = void (*)(const float*, int32_t*, int32_t*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data(), converted.data());
    for (size_t i = 0; i < source.size(); ++i) {
        const int32_t expected =
            static_cast<int32_t>(std::nearbyint(source[i]));
        EXPECT_EQ(converted[i], expected);
        if (i < 5) {
            EXPECT_EQ(output[i * 2], expected);
            EXPECT_EQ(output[i * 2 + 1], -999);
        }
    }
    EXPECT_EQ(output.back(), -999);
}

TEST(GemvN1StoreEmitterExecution, ScalarStridedF32PreservesGaps)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 2, 1);
        const auto              sourcePtr = frame.p[0];
        const auto              cursor    = frame.p[1];
        const auto              stride    = frame.t[0];
        jit.vmovups(Xbyak::Zmm(4), Xbyak::util::ptr[sourcePtr]);
        for (int chunk = 0; chunk < 4; ++chunk) {
            jit.vextractf32x4(Xbyak::Xmm(chunk), Xbyak::Zmm(4), chunk);
        }
        jit.mov(stride, 2 * sizeof(float));

        std::array<int, 2>       sourceChunks{ 0, 1 };
        const GemvN1StoreRequest request{
            GemvN1Source::extractedXmm(sourceChunks.data(), 2, 5),
            GemvN1Destination::scalarStrided(cursor, stride),
            StoreSpec::convert(DataType::f32, DataType::f32), StoreTemps::none()
        };
        GemvN1StoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
    }
    jit.ready();

    std::array<float, 16> source{};
    std::array<float, 11> output{};
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<float>(i) + 0.5f;
    }
    output.fill(-99.0f);
    using Kernel = void (*)(const float*, float*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data());

    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(output[i * 2], source[i]);
        EXPECT_EQ(output[i * 2 + 1], -99.0f);
    }
    EXPECT_EQ(output.back(), -99.0f);
}

TEST(GemvN1StoreEmitterExecution, F16FullRegistersAdvanceByRuntimeStride)
{
    runN1F16ContiguousStore(32, true, false, 96);
}

TEST(GemvN1StoreEmitterExecution, F16PartialTailDoesNotAdvanceCursor)
{
    runN1F16ContiguousStore(20, true, false, 48);
}

TEST(GemvN1StoreEmitterExecution, F16PreserveIgnoresInactiveAdvanceRegister)
{
    runN1F16ContiguousStore(32, false, false, 0);
}

TEST(GemvN1StoreEmitterExecution, F16AdvanceAcceptsRax)
{
    runN1F16ContiguousStore(32, true, true, 96);
}

TEST(GemvM1StoreEmitterExecution, F32PreservesCursor)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 3, 0);
        const auto              sourcePtr    = frame.p[0];
        const auto              cursor       = frame.p[1];
        const auto              cursorResult = frame.p[2];
        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);

        const GemvM1StoreRequest request{ RegSpan{ 0, SourceRegWidth::zmm, 16 },
                                          { cursor, StoreMask::none() },
                                          StoreSpec::convert(DataType::f32,
                                                             DataType::f32),
                                          StoreTemps::none() };
        GemvM1StoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.mov(Xbyak::util::ptr[cursorResult], cursor);
    }
    jit.ready();

    std::array<float, 16> source{};
    std::array<float, 16> output{};
    std::uintptr_t        cursorResult = 0;
    using Kernel = void (*)(const float*, float*, std::uintptr_t*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data(), &cursorResult);
    EXPECT_EQ(cursorResult, reinterpret_cast<std::uintptr_t>(output.data()));
}

TEST(GemvM1StoreEmitterExecution, FullBf16AdvancesCursor)
{
    if (!hasAvx512Bf16Support()) {
        GTEST_SKIP() << "Requires AVX-512 BF16";
    }

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 3, 0);
        const auto              sourcePtr    = frame.p[0];
        const auto              cursor       = frame.p[1];
        const auto              cursorResult = frame.p[2];
        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);

        const GemvM1StoreRequest request{ RegSpan{ 0, SourceRegWidth::zmm, 16 },
                                          { cursor, StoreMask::none() },
                                          StoreSpec::nativeBf16(DataType::f32),
                                          StoreTemps::none() };
        GemvM1StoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.mov(Xbyak::util::ptr[cursorResult], cursor);
    }
    jit.ready();

    std::array<float, 16>    source{};
    std::array<uint16_t, 16> output{};
    std::uintptr_t           cursorResult = 0;
    using Kernel = void (*)(const float*, uint16_t*, std::uintptr_t*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data(), &cursorResult);
    EXPECT_EQ(cursorResult,
              reinterpret_cast<std::uintptr_t>(output.data() + output.size()));
}

TEST(GemvM1StoreEmitterExecution, ReadOnlyS32MaskedPreservesCursor)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }
    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 4, 1);
        const auto              sourcePtr    = frame.p[0];
        const auto              cursor       = frame.p[1];
        const auto              cursorResult = frame.p[2];
        const auto              maskValue    = frame.p[3];
        jit.vmovdqu32(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.kmovw(Xbyak::Opmask(1), maskValue.cvt32());
        const GemvM1StoreRequest request{
            RegSpan{ 0, SourceRegWidth::zmm, 5 },
            { cursor, StoreMask::opmask(Xbyak::Opmask(1)) },
            StoreSpec::convert(DataType::s32, DataType::s32),
            StoreTemps::none()
        };
        GemvM1StoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.mov(Xbyak::util::ptr[cursorResult], cursor);
    }
    jit.ready();
    const std::array<int32_t, 16> source{ -8, -7, -6, -5, -4, -3, -2, -1,
                                          0,  1,  2,  3,  4,  5,  6,  7 };
    std::array<int32_t, 16>       output{};
    output.fill(-999);
    std::uintptr_t cursorResult = 0;
    using Kernel =
        void (*)(const int32_t*, int32_t*, std::uintptr_t*, uint16_t);
    auto kernel = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data(), &cursorResult, 0x001F);
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_EQ(output[i], i < 5 ? source[i] : -999);
    }
    EXPECT_EQ(cursorResult, reinterpret_cast<std::uintptr_t>(output.data()));
}

TEST(GemvM1StoreEmitterExecution, DestructiveF32ToS32Masked)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }
    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 4, 1);
        const auto              sourcePtr    = frame.p[0];
        const auto              cursor       = frame.p[1];
        const auto              convertedPtr = frame.p[2];
        const auto              maskValue    = frame.p[3];
        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.kmovw(Xbyak::Opmask(1), maskValue.cvt32());
        const GemvM1StoreRequest request{
            RegSpan{ 0, SourceRegWidth::zmm, 5 },
            { cursor, StoreMask::opmask(Xbyak::Opmask(1)) },
            StoreSpec::convert(DataType::f32, DataType::s32),
            StoreTemps::none()
        };
        GemvM1StoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.vmovdqu32(Xbyak::util::ptr[convertedPtr], Xbyak::Zmm(0));
    }
    jit.ready();
    const std::array<float, 16> source{ -7.5f, -1.5f, -0.5f, 0.5f, 2.5f, 3.5f,
                                        4.5f,  5.5f,  6.5f,  7.5f, 8.5f, 9.5f,
                                        10.5f, 11.5f, 12.5f, 13.5f };
    std::array<int32_t, 16>     output{};
    std::array<int32_t, 16>     converted{};
    output.fill(-999);
    using Kernel = void (*)(const float*, int32_t*, int32_t*, uint16_t);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data(), converted.data(), 0x001F);
    for (size_t i = 0; i < source.size(); ++i) {
        const int32_t expected =
            static_cast<int32_t>(std::nearbyint(source[i]));
        EXPECT_EQ(converted[i], expected);
        EXPECT_EQ(output[i], i < 5 ? expected : -999);
    }
}

TEST(GemvN1StoreEmitterExecution, ExtractedBf16OwnsAndConvertsSources)
{
    if (!hasAvx512Bf16Support()) {
        GTEST_SKIP() << "Requires AVX-512 BF16";
    }

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 2, 1);
        const auto              sourcePtr = frame.p[0];
        const auto              cursor    = frame.p[1];
        const auto              stride    = frame.t[0];
        jit.vmovups(Xbyak::Zmm(4), Xbyak::util::ptr[sourcePtr]);
        jit.vextractf32x4(Xbyak::Xmm(0), Xbyak::Zmm(4), 0);
        jit.vextractf32x4(Xbyak::Xmm(1), Xbyak::Zmm(4), 1);
        jit.mov(stride, 2 * sizeof(uint16_t));

        std::array<int, 2>       chunks{ 0, 1 };
        const GemvN1StoreRequest request{
            GemvN1Source::extractedXmm(chunks.data(), chunks.size(), 5),
            GemvN1Destination::scalarStrided(cursor, stride),
            StoreSpec::nativeBf16(DataType::f32), StoreTemps::none()
        };
        GemvN1StoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
    }
    jit.ready();

    std::array<float, 16>    source{};
    std::array<uint16_t, 11> output{};
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<float>(i) + 1.0f;
    }
    output.fill(0xFFFF);
    using Kernel = void (*)(const float*, uint16_t*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data());

    for (size_t i = 0; i < 5; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &source[i], sizeof(bits));
        EXPECT_EQ(output[i * 2], static_cast<uint16_t>(bits >> 16));
        EXPECT_EQ(output[i * 2 + 1], 0xFFFF);
    }
    EXPECT_EQ(output.back(), 0xFFFF);
}

TEST(GemvM1StoreEmitterExecution, MaskedF32Fringe)
{
    runM1Store<false>(SourceRegWidth::zmm, 5);
}

TEST(GemvM1StoreEmitterExecution, MaskedBf16Fringe)
{
    runM1Store<true>(SourceRegWidth::zmm, 5);
}

TEST(GemvM1StoreEmitterExecution, LowHalfCompactF32)
{
    runM1Store<false>(SourceRegWidth::ymm, 8);
}

TEST(GemvM1StoreEmitterExecution, LowHalfCompactBf16)
{
    runM1Store<true>(SourceRegWidth::ymm, 8);
}

TEST(GemvN1StoreEmitterExecution, ContiguousF32)
{
    runN1ContiguousStore<false>();
}

TEST(GemvN1StoreEmitterExecution, ContiguousBf16)
{
    runN1ContiguousStore<true>();
}

TEST(GemvM1StoreEmitterExecution, F32ToS8ClampsBeforeConversion)
{
    runGemvF32ToInt8Clamp<false, false>();
}

TEST(GemvM1StoreEmitterExecution, F32ToU8ClampsBeforeConversion)
{
    runGemvF32ToInt8Clamp<true, false>();
}

TEST(GemvN1StoreEmitterExecution, F32ToS8ClampsBeforeConversion)
{
    runGemvF32ToInt8Clamp<false, true>();
}

TEST(GemvN1StoreEmitterExecution, F32ToU8ClampsBeforeConversion)
{
    runGemvF32ToInt8Clamp<true, true>();
}

TEST(GemvN1StoreEmitterExecution, PreservedContiguousTwoRegisterF32)
{
    runN1PreservedContiguousTwoRegisterStore<PreservedContiguousType::f32>();
}

TEST(GemvN1StoreEmitterExecution, PreservedContiguousTwoRegisterBf16)
{
    runN1PreservedContiguousTwoRegisterStore<PreservedContiguousType::bf16>();
}

TEST(GemvN1StoreEmitterExecution, PreservedContiguousTwoRegisterS32)
{
    runN1PreservedContiguousTwoRegisterStore<PreservedContiguousType::s32>();
}

TEST(GemvN1StoreEmitterExecution, PackedScalarStridedF32)
{
    runN1PackedScalarStore<false>(SourceRegWidth::zmm);
}

TEST(GemvN1StoreEmitterExecution, PackedScalarStridedBf16)
{
    runN1PackedScalarStore<true>(SourceRegWidth::zmm);
}

TEST(GemvN1StoreEmitterExecution, PackedScalarStridedLowHalfF32)
{
    runN1PackedScalarStore<false>(SourceRegWidth::ymm);
}

TEST(GemvN1StoreEmitterExecution, PackedScalarStridedLowHalfBf16)
{
    runN1PackedScalarStore<true>(SourceRegWidth::ymm);
}
