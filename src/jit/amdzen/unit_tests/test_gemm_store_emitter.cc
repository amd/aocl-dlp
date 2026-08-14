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

#include "jit/amdzen/store/gemm_store_emitter.hh"
#include "jit/amdzen/store/store_emit.hh"
#include "jit_generator_tests_utils.hh"

namespace {

using amdzen::store::GemmStoreEmitter;
using amdzen::store::GemmStoreRequest;
using amdzen::store::SourceRegWidth;
using amdzen::store::StoreMask;
using amdzen::store::StoreMaskKind;
using amdzen::store::StoreSpec;
using amdzen::store::StoreTemps;
using amdzen::utils::JIT_KERNEL_SIZE;
using amdzen::utils::kernelInstrType;
using dlp::kernel_frame::DataType;

constexpr auto kType = kernelInstrType::avx512_zmm_32_reg;

GemmStoreRequest
makeRequest(Xbyak::Reg64   regCptr,
            Xbyak::Reg64   stride,
            StoreMask      finalMask       = StoreMask::none(),
            int            rows            = 1,
            int            registersPerRow = 1,
            SourceRegWidth sourceRegWidth  = SourceRegWidth::zmm)
{
    return { { 0, rows, registersPerRow, sourceRegWidth },
             { regCptr, stride, finalMask },
             StoreSpec::convert(DataType::f32, DataType::f32),
             StoreTemps::none() };
}

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
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

uint16_t
toF16Rne(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign     = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    uint32_t       mantissa = bits & 0x7fffffu;

    if (exponent == 0xffu) {
        return static_cast<uint16_t>(
            sign | 0x7c00u
            | (mantissa == 0 ? 0 : ((mantissa >> 13) | 0x0200u)));
    }
    if (exponent == 0) {
        return static_cast<uint16_t>(sign);
    }

    int32_t halfExponent = static_cast<int32_t>(exponent) - 112;
    mantissa |= 0x800000u;
    if (halfExponent <= 0) {
        if (halfExponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        const int      shift    = 14 - halfExponent;
        const uint32_t roundBit = (mantissa >> (shift - 1)) & 1u;
        const uint32_t sticky   = (mantissa & ((1u << (shift - 1)) - 1u)) != 0;
        uint32_t       halfMantissa = mantissa >> shift;
        halfMantissa += roundBit && (sticky || (halfMantissa & 1u));
        return static_cast<uint16_t>(
            sign | (halfMantissa >= 0x400u ? 0x400u : halfMantissa));
    }
    if (halfExponent >= 0x1f) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    const uint32_t roundBits = mantissa & 0x1fffu;
    if (roundBits > 0x1000u
        || (roundBits == 0x1000u && ((mantissa >> 13) & 1u))) {
        mantissa += 0x1000u;
    }
    if (mantissa & 0x1000000u) {
        ++halfExponent;
        mantissa = 0x800000u;
        if (halfExponent >= 0x1f) {
            return static_cast<uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<uint16_t>(sign
                                 | (static_cast<uint32_t>(halfExponent) << 10)
                                 | ((mantissa >> 13) & 0x3ffu));
}

float
floatFromBits(uint32_t bits)
{
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

} // namespace

TEST(StoreEmitDispatch, RejectsUnavailableIsaWithoutEmitting)
{
    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    const auto   request    = makeRequest(Xbyak::Reg64(0), Xbyak::Reg64(1));
    const size_t sizeBefore = jit.getSize();
    EXPECT_EQ(
        (amdzen::store::emit<kernelInstrType::avx2_ymm_16_reg>(jit, request)),
        dlp::jit::jitGeneratorError::notSupported);
    EXPECT_EQ(jit.getSize(), sizeBefore);
}

TEST(GemmStoreEmitterDispatch, RejectsYmmConversionWithoutEmitting)
{
    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    auto                 request = makeRequest(Xbyak::Reg64(0), Xbyak::Reg64(1),
                                               StoreMask::none(), 1, 1, SourceRegWidth::ymm);
    request.store = StoreSpec::convert(DataType::f32, DataType::s32);
    GemmStoreEmitter<kType> emitter(jit);
    const auto              sizeBefore = jit.getSize();

    EXPECT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::notSupported);
    EXPECT_EQ(jit.getSize(), sizeBefore);
}

TEST(GemmStoreEmitterExecution, DirectF32FullStore)
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

        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.mov(stride, 64);

        const auto              request = makeRequest(cursor, stride);
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
    }
    jit.ready();

    std::array<float, 16> source{};
    std::array<float, 18> output{};
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<float>(i) + 0.25f;
    }
    output.fill(-99.0f);

    using Kernel = void (*)(const float*, float*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data() + 1);

    EXPECT_EQ(output.front(), -99.0f);
    EXPECT_EQ(output.back(), -99.0f);
    for (size_t i = 0; i < source.size(); ++i) {
        EXPECT_EQ(output[i + 1], source[i]);
    }
}

TEST(GemmStoreEmitterExecution, DirectF32MaskedStore)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 3, 1);
        const auto              sourcePtr = frame.p[0];
        const auto              cursor    = frame.p[1];
        const auto              mask      = frame.p[2];
        const auto              stride    = frame.t[0];

        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.kmovw(Xbyak::Opmask(1), mask.cvt32());
        jit.mov(stride, 64);

        const auto request =
            makeRequest(cursor, stride, StoreMask::opmask(Xbyak::Opmask(1)));
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
    }
    jit.ready();

    std::array<float, 16> source{};
    std::array<float, 16> output{};
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<float>(i) + 0.5f;
    }
    output.fill(-99.0f);

    using Kernel = void (*)(const float*, float*, uint16_t);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data(), 0x001F);

    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_EQ(output[i], i < 5 ? source[i] : -99.0f);
    }
}

TEST(GemmStoreEmitterExecution, DirectF32GridWithRowPadding)
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

        for (int reg = 0; reg < 4; ++reg) {
            jit.vmovups(Xbyak::Zmm(reg),
                        Xbyak::util::ptr[sourcePtr + reg * 64]);
        }
        jit.mov(stride, 40 * sizeof(float));

        const auto request =
            makeRequest(cursor, stride, StoreMask::none(), 2, 2);
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
    }
    jit.ready();

    std::array<float, 64> source{};
    std::array<float, 82> output{};
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<float>(i) + 1.0f;
    }
    output.fill(-99.0f);

    using Kernel = void (*)(const float*, float*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data() + 1);

    EXPECT_EQ(output.front(), -99.0f);
    EXPECT_EQ(output.back(), -99.0f);
    for (size_t i = 0; i < 32; ++i) {
        EXPECT_EQ(output[i + 1], source[i]);
        EXPECT_EQ(output[i + 41], source[i + 32]);
    }
    for (size_t i = 33; i < 41; ++i) {
        EXPECT_EQ(output[i], -99.0f);
    }
    for (size_t i = 73; i < 81; ++i) {
        EXPECT_EQ(output[i], -99.0f);
    }
}

TEST(GemmStoreEmitterExecution, DirectF32LowHalfStore)
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

        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.mov(stride, 8 * sizeof(float));

        const auto request = makeRequest(cursor, stride, StoreMask::none(), 1,
                                         1, SourceRegWidth::ymm);
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
    }
    jit.ready();

    std::array<float, 16> source{};
    std::array<float, 10> output{};
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<float>(i) + 0.75f;
    }
    output.fill(-99.0f);

    using Kernel = void (*)(const float*, float*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data() + 1);

    EXPECT_EQ(output.front(), -99.0f);
    EXPECT_EQ(output.back(), -99.0f);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(output[i + 1], source[i]);
    }
}

TEST(GemmStoreEmitterExecution, NativeBf16FullStore)
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
        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.mov(stride, 16 * sizeof(uint16_t));

        const int               scratchRegister = 1;
        const auto              scratch = StoreTemps::withZmm(scratchRegister);
        const GemmStoreRequest  request{ { 0, 1, 1, SourceRegWidth::zmm },
                                         { cursor, stride, StoreMask::none() },
                                        StoreSpec::nativeBf16(DataType::f32),
                                        scratch };
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
    }
    jit.ready();

    std::array<float, 16>    source{};
    std::array<uint16_t, 18> output{};
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<float>(i) + 0.3125f;
    }
    output.fill(0xA5A5);

    using Kernel = void (*)(const float*, uint16_t*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data() + 1);

    EXPECT_EQ(output.front(), 0xA5A5);
    EXPECT_EQ(output.back(), 0xA5A5);
    for (size_t i = 0; i < source.size(); ++i) {
        EXPECT_EQ(output[i + 1], toBf16Rne(source[i]));
    }
}

TEST(GemmStoreEmitterExecution, NativeBf16MaskedStore)
{
    if (!hasAvx512Bf16Support()) {
        GTEST_SKIP() << "Requires AVX-512 BF16";
    }

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 3, 1);
        const auto              sourcePtr = frame.p[0];
        const auto              cursor    = frame.p[1];
        const auto              mask      = frame.p[2];
        const auto              stride    = frame.t[0];
        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.kmovw(Xbyak::Opmask(1), mask.cvt32());
        jit.mov(stride, 16 * sizeof(uint16_t));

        const int               scratchRegister = 1;
        const auto              scratch = StoreTemps::withZmm(scratchRegister);
        const GemmStoreRequest  request{ { 0, 1, 1, SourceRegWidth::zmm },
                                         { cursor, stride,
                                           StoreMask::opmask(Xbyak::Opmask(1)) },
                                        StoreSpec::nativeBf16(DataType::f32),
                                        scratch };
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
    }
    jit.ready();

    std::array<float, 16>    source{};
    std::array<uint16_t, 16> output{};
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<float>(i) + 0.4375f;
    }
    output.fill(0xA5A5);

    using Kernel = void (*)(const float*, uint16_t*, uint16_t);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data(), 0x001F);

    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_EQ(output[i], i < 5 ? toBf16Rne(source[i]) : 0xA5A5);
    }
}

TEST(GemmStoreEmitterExecution, NativeBf16LowHalfMaskedStore)
{
    if (!hasAvx512Bf16Support()) {
        GTEST_SKIP() << "Requires AVX-512 BF16";
    }

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 4, 1);
        const auto              sourcePtr    = frame.p[0];
        const auto              cursor       = frame.p[1];
        const auto              preservedPtr = frame.p[2];
        const auto              mask         = frame.p[3];
        const auto              stride       = frame.t[0];
        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.kmovw(Xbyak::Opmask(1), mask.cvt32());
        jit.mov(stride, 8 * sizeof(uint16_t));

        const int scratchRegister = 1;
        auto      request =
            makeRequest(cursor, stride, StoreMask::opmask(Xbyak::Opmask(1)), 1,
                        1, SourceRegWidth::ymm);
        request.store = StoreSpec::nativeBf16(DataType::f32);
        request.temps = StoreTemps::withZmm(scratchRegister);
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.vmovups(Xbyak::util::ptr[preservedPtr], Xbyak::Zmm(0));
    }
    jit.ready();

    const std::array<float, 16> source{ floatFromBits(0x3F808000),
                                        floatFromBits(0x3F818000),
                                        -1.0f,
                                        0.0f,
                                        65504.0f,
                                        -65504.0f,
                                        (std::numeric_limits<float>::max)(),
                                        -(std::numeric_limits<float>::max)(),
                                        101.0f,
                                        102.0f,
                                        103.0f,
                                        104.0f,
                                        105.0f,
                                        106.0f,
                                        107.0f,
                                        108.0f };
    std::array<uint16_t, 10>    output{};
    std::array<float, 16>       preserved{};
    output.fill(0xA5A5);
    using Kernel = void (*)(const float*, uint16_t*, float*, uint16_t);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data() + 1, preserved.data(), 0x00AD);

    EXPECT_EQ(output.front(), 0xA5A5);
    EXPECT_EQ(output.back(), 0xA5A5);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(output[i + 1],
                  (0x00AD & (1u << i)) ? toBf16Rne(source[i]) : 0xA5A5);
    }
    EXPECT_EQ(preserved, source);
}

TEST(GemmStoreEmitterExecution, DirectS32FullStore)
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

        jit.vmovdqu32(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.mov(stride, 16 * sizeof(int32_t));

        auto request  = makeRequest(cursor, stride);
        request.store = StoreSpec::convert(DataType::s32, DataType::s32);
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
    }
    jit.ready();

    std::array<int32_t, 16> source{};
    std::array<int32_t, 18> output{};
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<int32_t>(i) - 8;
    }
    output.fill(-999);

    using Kernel = void (*)(const int32_t*, int32_t*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data() + 1);

    EXPECT_EQ(output.front(), -999);
    EXPECT_EQ(output.back(), -999);
    for (size_t i = 0; i < source.size(); ++i) {
        EXPECT_EQ(output[i + 1], source[i]);
    }
}

TEST(GemmStoreEmitterExecution, DestructiveF32ToS32MaskedStore)
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
        const auto              mask         = frame.p[3];
        const auto              stride       = frame.t[0];

        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.kmovw(Xbyak::Opmask(1), mask.cvt32());
        jit.mov(stride, 16 * sizeof(int32_t));

        auto request =
            makeRequest(cursor, stride, StoreMask::opmask(Xbyak::Opmask(1)));
        request.store = StoreSpec::convert(DataType::f32, DataType::s32);
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.vmovdqu32(Xbyak::util::ptr[convertedPtr], Xbyak::Zmm(0));
    }
    jit.ready();

    std::array<float, 16>   source{};
    std::array<int32_t, 16> output{};
    std::array<int32_t, 16> converted{};
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<float>(static_cast<int>(i) - 8);
    }
    output.fill(-999);
    converted.fill(-999);

    using Kernel = void (*)(const float*, int32_t*, int32_t*, uint16_t);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data(), converted.data(), 0x001F);

    for (size_t i = 0; i < source.size(); ++i) {
        const int32_t expected = static_cast<int32_t>(source[i]);
        EXPECT_EQ(converted[i], expected);
        EXPECT_EQ(output[i], i < 5 ? expected : -999);
    }
}

TEST(GemmStoreEmitterExecution, ReadOnlyS32ToS8Saturates)
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
        jit.vmovdqu32(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.mov(stride, 16);

        auto request  = makeRequest(cursor, stride);
        request.store = StoreSpec::convert(DataType::s32, DataType::s8);
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
    }
    jit.ready();

    const std::array<int32_t, 16> source{
        (std::numeric_limits<int32_t>::min)(),
        -1024,
        -129,
        -128,
        -127,
        -1,
        0,
        1,
        126,
        127,
        128,
        255,
        1024,
        42,
        -42,
        (std::numeric_limits<int32_t>::max)()
    };
    std::array<int8_t, 18> output{};
    output.fill(0x55);
    using Kernel = void (*)(const int32_t*, int8_t*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data() + 1);

    EXPECT_EQ(output.front(), 0x55);
    EXPECT_EQ(output.back(), 0x55);
    for (size_t i = 0; i < source.size(); ++i) {
        EXPECT_EQ(output[i + 1],
                  static_cast<int8_t>(std::clamp(source[i], -128, 127)));
    }
}

TEST(GemmStoreEmitterExecution, DestructiveF32ToS8MaskedStore)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 4, 2);
        const auto              sourcePtr    = frame.p[0];
        const auto              cursor       = frame.p[1];
        const auto              convertedPtr = frame.p[2];
        const auto              mask         = frame.p[3];
        const auto              stride       = frame.t[0];
        const auto              immediate    = frame.t[1];
        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.kmovw(Xbyak::Opmask(1), mask.cvt32());
        jit.mov(stride, 16);

        auto request =
            makeRequest(cursor, stride, StoreMask::opmask(Xbyak::Opmask(1)));
        request.store = StoreSpec::convert(DataType::f32, DataType::s8);
        request.temps =
            StoreTemps::withZmmAndGpr(1, Xbyak::Reg64(immediate.getIdx()));
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.vmovdqu32(Xbyak::util::ptr[convertedPtr], Xbyak::Zmm(0));
    }
    jit.ready();

    const std::array<float, 16> source{ (std::numeric_limits<float>::max)(),
                                        std::numeric_limits<float>::infinity(),
                                        std::numeric_limits<float>::quiet_NaN(),
                                        -(std::numeric_limits<float>::max)(),
                                        -std::numeric_limits<float>::infinity(),
                                        1.0f,
                                        126.0f,
                                        127.0f,
                                        128.0f,
                                        200.0f,
                                        15.0f,
                                        -15.0f,
                                        63.0f,
                                        64.0f,
                                        -64.0f,
                                        -65.0f };
    std::array<int8_t, 16>      output{};
    std::array<int32_t, 16>     converted{};
    output.fill(0x55);
    converted.fill(-999);
    using Kernel = void (*)(const float*, int8_t*, int32_t*, uint16_t);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data(), converted.data(), 0x001F);

    for (size_t i = 0; i < source.size(); ++i) {
        const float clipped  = std::fmax(-128.0f, std::fmin(127.0f, source[i]));
        const auto  expected = static_cast<int32_t>(std::nearbyint(clipped));
        EXPECT_EQ(converted[i], expected);
        EXPECT_EQ(output[i],
                  i < 5 ? static_cast<int8_t>(std::clamp(expected, -128, 127))
                        : static_cast<int8_t>(0x55));
    }
}

TEST(GemmStoreEmitterExecution, ReadOnlyS32ToU8Clamps)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 2, 2);
        const auto              sourcePtr = frame.p[0];
        const auto              cursor    = frame.p[1];
        const auto              stride    = frame.t[0];
        const auto              immediate = frame.t[1];
        jit.vmovdqu32(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.mov(stride, 16);

        std::array<int, 3> vectors{ 1, 2, 3 };
        const int          gpr     = immediate.getIdx();
        auto               request = makeRequest(cursor, stride);
        request.store = StoreSpec::convert(DataType::s32, DataType::u8);
        request.temps =
            StoreTemps::withZmmAndGpr(vectors[0], Xbyak::Reg64(gpr));
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
    }
    jit.ready();

    const std::array<int32_t, 16> source{
        (std::numeric_limits<int32_t>::min)(),
        -1024,
        -1,
        0,
        1,
        127,
        254,
        255,
        256,
        1024,
        42,
        200,
        15,
        99,
        300,
        (std::numeric_limits<int32_t>::max)()
    };
    std::array<uint8_t, 18> output{};
    output.fill(0xA5);
    using Kernel = void (*)(const int32_t*, uint8_t*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data() + 1);

    EXPECT_EQ(output.front(), 0xA5);
    EXPECT_EQ(output.back(), 0xA5);
    for (size_t i = 0; i < source.size(); ++i) {
        EXPECT_EQ(output[i + 1],
                  static_cast<uint8_t>(std::clamp(source[i], 0, 255)));
    }
}

TEST(GemmStoreEmitterExecution, DestructiveF32ToU8MaskedStore)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 4, 2);
        const auto              sourcePtr    = frame.p[0];
        const auto              cursor       = frame.p[1];
        const auto              convertedPtr = frame.p[2];
        const auto              mask         = frame.p[3];
        const auto              stride       = frame.t[0];
        const auto              immediate    = frame.t[1];
        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.kmovw(Xbyak::Opmask(1), mask.cvt32());
        jit.mov(stride, 16);

        std::array<int, 3> vectors{ 1, 2, 3 };
        const int          gpr = immediate.getIdx();
        auto               request =
            makeRequest(cursor, stride, StoreMask::opmask(Xbyak::Opmask(1)));
        request.store = StoreSpec::convert(DataType::f32, DataType::u8);
        request.temps =
            StoreTemps::withZmmAndGpr(vectors[0], Xbyak::Reg64(gpr));
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.vmovdqu32(Xbyak::util::ptr[convertedPtr], Xbyak::Zmm(0));
    }
    jit.ready();

    const std::array<float, 16> source{ (std::numeric_limits<float>::max)(),
                                        std::numeric_limits<float>::infinity(),
                                        std::numeric_limits<float>::quiet_NaN(),
                                        -(std::numeric_limits<float>::max)(),
                                        -std::numeric_limits<float>::infinity(),
                                        256.0f,
                                        1000.0f,
                                        127.0f,
                                        128.0f,
                                        200.0f,
                                        15.0f,
                                        42.0f,
                                        99.0f,
                                        254.0f,
                                        300.0f,
                                        -300.0f };
    std::array<uint8_t, 16>     output{};
    std::array<int32_t, 16>     converted{};
    output.fill(0xA5);
    converted.fill(-999);
    using Kernel = void (*)(const float*, uint8_t*, int32_t*, uint16_t);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data(), converted.data(), 0x001F);

    for (size_t i = 0; i < source.size(); ++i) {
        const float clipped  = std::fmax(0.0f, std::fmin(255.0f, source[i]));
        const auto  expected = static_cast<int32_t>(std::nearbyint(clipped));
        EXPECT_EQ(converted[i], expected);
        EXPECT_EQ(output[i],
                  i < 5 ? static_cast<uint8_t>(std::clamp(expected, 0, 255))
                        : static_cast<uint8_t>(0xA5));
    }
}

TEST(GemmStoreEmitterExecution, DestructiveS32ToF32Converts)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 4, 2);
        const auto              sourcePtr    = frame.p[0];
        const auto              cursor       = frame.p[1];
        const auto              preservedPtr = frame.p[2];
        const auto              mask         = frame.p[3];
        const auto              stride       = frame.t[0];
        const auto              immediate    = frame.t[1];
        jit.vmovdqu32(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.kmovw(Xbyak::Opmask(1), mask.cvt32());
        jit.mov(stride, 16 * sizeof(float));

        auto request =
            makeRequest(cursor, stride, StoreMask::opmask(Xbyak::Opmask(1)));
        request.store = StoreSpec::convert(DataType::s32, DataType::f32);
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.vmovdqu32(Xbyak::util::ptr[preservedPtr], Xbyak::Zmm(0));
    }
    jit.ready();

    const std::array<int32_t, 16> source{ (std::numeric_limits<int32_t>::min)(),
                                          -16777217,
                                          -16777216,
                                          -1,
                                          0,
                                          1,
                                          16777216,
                                          16777217,
                                          (std::numeric_limits<int32_t>::max)(),
                                          -1024,
                                          1024,
                                          127,
                                          -127,
                                          255,
                                          -255,
                                          42 };
    std::array<float, 18>         output{};
    std::array<int32_t, 16>       preserved{};
    output.fill(-999.0f);
    using Kernel = void (*)(const int32_t*, float*, int32_t*, uint16_t);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    constexpr uint16_t mask = 0xA55B;
    kernel(source.data(), output.data() + 1, preserved.data(), mask);

    EXPECT_EQ(output.front(), -999.0f);
    EXPECT_EQ(output.back(), -999.0f);
    for (size_t i = 0; i < source.size(); ++i) {
        EXPECT_EQ(output[i + 1],
                  (mask & (1u << i)) ? static_cast<float>(source[i]) : -999.0f);
    }
    EXPECT_NE(preserved, source);
}

TEST(GemmStoreEmitterExecution, ReadOnlyF32ToF16Converts)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 4, 1);
        const auto              sourcePtr    = frame.p[0];
        const auto              cursor       = frame.p[1];
        const auto              preservedPtr = frame.p[2];
        const auto              mask         = frame.p[3];
        const auto              stride       = frame.t[0];
        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.kmovw(Xbyak::Opmask(1), mask.cvt32());
        jit.mov(stride, 16 * sizeof(uint16_t));

        const std::array<int, 1> vectors{ 1 };
        auto                     request =
            makeRequest(cursor, stride, StoreMask::opmask(Xbyak::Opmask(1)));
        request.store = StoreSpec::convert(DataType::f32, DataType::f16);
        request.temps = StoreTemps::withZmm(vectors[0]);
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.vmovups(Xbyak::util::ptr[preservedPtr], Xbyak::Zmm(0));
    }
    jit.ready();

    const std::array<float, 16> source{ 0.0f,
                                        -0.0f,
                                        1.0f + 0x1p-11f,
                                        1.0f + 0x3p-11f,
                                        -1.0f - 0x1p-11f,
                                        65504.0f,
                                        65519.0f,
                                        65520.0f,
                                        -65504.0f,
                                        -65520.0f,
                                        0x1p-24f,
                                        0x1p-25f,
                                        -0x1p-25f,
                                        std::numeric_limits<float>::infinity(),
                                        -std::numeric_limits<float>::infinity(),
                                        42.25f };
    std::array<uint16_t, 18>    output{};
    std::array<float, 16>       preserved{};
    output.fill(0xA5A5);
    using Kernel = void (*)(const float*, uint16_t*, float*, uint16_t);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    constexpr uint16_t mask = 0xD6B7;
    kernel(source.data(), output.data() + 1, preserved.data(), mask);

    EXPECT_EQ(output.front(), 0xA5A5);
    EXPECT_EQ(output.back(), 0xA5A5);
    for (size_t i = 0; i < source.size(); ++i) {
        EXPECT_EQ(output[i + 1],
                  (mask & (1u << i)) ? toF16Rne(source[i]) : 0xA5A5);
    }
    EXPECT_EQ(preserved, source);
}

TEST(GemmStoreEmitterExecution, DestructiveS32ToF16Converts)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 4, 1);
        const auto              sourcePtr    = frame.p[0];
        const auto              cursor       = frame.p[1];
        const auto              preservedPtr = frame.p[2];
        const auto              mask         = frame.p[3];
        const auto              stride       = frame.t[0];
        jit.vmovdqu32(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.kmovw(Xbyak::Opmask(1), mask.cvt32());
        jit.mov(stride, 16 * sizeof(uint16_t));

        const std::array<int, 1> vectors{ 1 };
        auto                     request =
            makeRequest(cursor, stride, StoreMask::opmask(Xbyak::Opmask(1)));
        request.store = StoreSpec::convert(DataType::s32, DataType::f16);
        request.temps = StoreTemps::withZmm(vectors[0]);
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.vmovdqu32(Xbyak::util::ptr[preservedPtr], Xbyak::Zmm(0));
    }
    jit.ready();

    const std::array<int32_t, 16> source{ (std::numeric_limits<int32_t>::min)(),
                                          -65521,
                                          -65520,
                                          -65519,
                                          -65504,
                                          -1,
                                          0,
                                          1,
                                          65504,
                                          65519,
                                          65520,
                                          65521,
                                          (std::numeric_limits<int32_t>::max)(),
                                          2049,
                                          -2049,
                                          42 };
    std::array<uint16_t, 18>      output{};
    std::array<int32_t, 16>       preserved{};
    output.fill(0xA5A5);
    using Kernel = void (*)(const int32_t*, uint16_t*, int32_t*, uint16_t);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    constexpr uint16_t mask = 0x6DB5;
    kernel(source.data(), output.data() + 1, preserved.data(), mask);

    EXPECT_EQ(output.front(), 0xA5A5);
    EXPECT_EQ(output.back(), 0xA5A5);
    for (size_t i = 0; i < source.size(); ++i) {
        EXPECT_EQ(output[i + 1], (mask & (1u << i))
                                     ? toF16Rne(static_cast<float>(source[i]))
                                     : 0xA5A5);
    }
    for (size_t i = 0; i < source.size(); ++i) {
        EXPECT_EQ(floatFromBits(static_cast<uint32_t>(preserved[i])),
                  static_cast<float>(source[i]));
    }
}

TEST(GemmStoreEmitterExecution, SoftwareBf16FromF32Converts)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 4, 2);
        const auto              sourcePtr    = frame.p[0];
        const auto              cursor       = frame.p[1];
        const auto              preservedPtr = frame.p[2];
        const auto              mask         = frame.p[3];
        const auto              stride       = frame.t[0];
        const auto              immediate    = frame.t[1];
        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.kmovw(Xbyak::Opmask(1), mask.cvt32());
        jit.mov(stride, 16 * sizeof(uint16_t));

        const std::array<int, 3> vectors{ 1, 2, 3 };
        const int                gpr = immediate.getIdx();
        auto                     request =
            makeRequest(cursor, stride, StoreMask::opmask(Xbyak::Opmask(1)));
        request.store = StoreSpec::softwareBf16(DataType::f32);
        request.temps =
            StoreTemps::withZmmAndGpr(vectors[0], Xbyak::Reg64(gpr));
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.vmovups(Xbyak::util::ptr[preservedPtr], Xbyak::Zmm(0));
    }
    jit.ready();

    const std::array<float, 16> source{
        floatFromBits(0x3F808000),
        floatFromBits(0x3F818000),
        floatFromBits(0xBF808000),
        floatFromBits(0xBF818000),
        0.0f,
        -0.0f,
        (std::numeric_limits<float>::min)(),
        std::numeric_limits<float>::denorm_min(),
        (std::numeric_limits<float>::max)(),
        -(std::numeric_limits<float>::max)(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        1.0f,
        -1.0f,
        65504.0f,
        -65504.0f
    };
    std::array<uint16_t, 18> output{};
    std::array<float, 16>    preserved{};
    output.fill(0xA5A5);
    using Kernel = void (*)(const float*, uint16_t*, float*, uint16_t);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    constexpr uint16_t mask = 0xB6AD;
    kernel(source.data(), output.data() + 1, preserved.data(), mask);

    EXPECT_EQ(output.front(), 0xA5A5);
    EXPECT_EQ(output.back(), 0xA5A5);
    for (size_t i = 0; i < source.size(); ++i) {
        EXPECT_EQ(output[i + 1],
                  (mask & (1u << i)) ? toBf16Rne(source[i]) : 0xA5A5);
    }
    EXPECT_NE(preserved, source);
}

TEST(GemmStoreEmitterExecution, SoftwareBf16FromS32Converts)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }

    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 4, 2);
        const auto              sourcePtr    = frame.p[0];
        const auto              cursor       = frame.p[1];
        const auto              preservedPtr = frame.p[2];
        const auto              mask         = frame.p[3];
        const auto              stride       = frame.t[0];
        const auto              immediate    = frame.t[1];
        jit.vmovdqu32(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.kmovw(Xbyak::Opmask(1), mask.cvt32());
        jit.mov(stride, 16 * sizeof(uint16_t));

        const std::array<int, 3> vectors{ 1, 2, 3 };
        const int                gpr = immediate.getIdx();
        auto                     request =
            makeRequest(cursor, stride, StoreMask::opmask(Xbyak::Opmask(1)));
        request.store = StoreSpec::softwareBf16(DataType::s32);
        request.temps =
            StoreTemps::withZmmAndGpr(vectors[0], Xbyak::Reg64(gpr));
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.vmovdqu32(Xbyak::util::ptr[preservedPtr], Xbyak::Zmm(0));
    }
    jit.ready();

    const std::array<int32_t, 16> source{
        (std::numeric_limits<int32_t>::min)(),
        -16777217,
        -16777216,
        -65535,
        -257,
        -256,
        -1,
        0,
        1,
        255,
        256,
        257,
        65535,
        16777216,
        16777217,
        (std::numeric_limits<int32_t>::max)()
    };
    std::array<uint16_t, 18> output{};
    std::array<int32_t, 16>  preserved{};
    output.fill(0xA5A5);
    using Kernel = void (*)(const int32_t*, uint16_t*, int32_t*, uint16_t);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    constexpr uint16_t mask = 0xDB6D;
    kernel(source.data(), output.data() + 1, preserved.data(), mask);

    EXPECT_EQ(output.front(), 0xA5A5);
    EXPECT_EQ(output.back(), 0xA5A5);
    for (size_t i = 0; i < source.size(); ++i) {
        EXPECT_EQ(output[i + 1], (mask & (1u << i))
                                     ? toBf16Rne(static_cast<float>(source[i]))
                                     : 0xA5A5);
    }
    EXPECT_NE(preserved, source);
}

TEST(GemmStoreEmitterExecution, DestructiveF32ToS8ConvertsSource)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }
    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 4, 2);
        const auto              sourcePtr    = frame.p[0];
        const auto              cursor       = frame.p[1];
        const auto              preservedPtr = frame.p[2];
        const auto              mask         = frame.p[3];
        const auto              stride       = frame.t[0];
        const auto              immediate    = frame.t[1];
        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.kmovw(Xbyak::Opmask(1), mask.cvt32());
        jit.mov(stride, 16);

        auto request =
            makeRequest(cursor, stride, StoreMask::opmask(Xbyak::Opmask(1)));
        request.store = StoreSpec::convert(DataType::f32, DataType::s8);
        request.temps =
            StoreTemps::withZmmAndGpr(1, Xbyak::Reg64(immediate.getIdx()));
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.vmovups(Xbyak::util::ptr[preservedPtr], Xbyak::Zmm(0));
    }
    jit.ready();

    const std::array<float, 16> source{ -200.0f, -127.6f, -1.5f,  0.5f,
                                        126.6f,  1.0f,    126.0f, 127.0f,
                                        128.0f,  200.0f,  15.0f,  -15.0f,
                                        63.0f,   64.0f,   -64.0f, -65.0f };
    std::array<int8_t, 16>      output{};
    std::array<int32_t, 16>     preserved{};
    output.fill(0x55);
    using Kernel = void (*)(const float*, int8_t*, int32_t*, uint16_t);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data(), preserved.data(), 0x001F);
    for (size_t i = 0; i < source.size(); ++i) {
        const float clipped = std::fmax(-128.0f, std::fmin(127.0f, source[i]));
        const int   converted = static_cast<int>(std::nearbyint(clipped));
        EXPECT_EQ(preserved[i], converted);
        EXPECT_EQ(output[i],
                  i < 5 ? static_cast<int8_t>(std::clamp(converted, -128, 127))
                        : static_cast<int8_t>(0x55));
    }
}

TEST(GemmStoreEmitterExecution, DestructiveS32ToF32InPlace)
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
        jit.vmovdqu32(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.mov(stride, 16 * sizeof(float));
        auto request  = makeRequest(cursor, stride);
        request.store = StoreSpec::convert(DataType::s32, DataType::f32);
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.vmovups(Xbyak::util::ptr[convertedPtr], Xbyak::Zmm(0));
    }
    jit.ready();

    const std::array<int32_t, 16> source{ -2048, -127, -1, 0, 1, 2,  15, 127,
                                          255,   512,  7,  8, 9, 10, 11, 12 };
    std::array<float, 16>         output{};
    std::array<float, 16>         converted{};
    using Kernel = void (*)(const int32_t*, float*, float*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data(), converted.data());
    for (size_t i = 0; i < source.size(); ++i) {
        const float expected = static_cast<float>(source[i]);
        EXPECT_EQ(output[i], expected);
        EXPECT_EQ(converted[i], expected);
    }
}

TEST(GemmStoreEmitterExecution, DirectF32ToF16OneScratch)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }
    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 3, 1);
        const auto              sourcePtr    = frame.p[0];
        const auto              cursor       = frame.p[1];
        const auto              preservedPtr = frame.p[2];
        const auto              stride       = frame.t[0];
        jit.vmovups(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.mov(stride, 16 * sizeof(uint16_t));
        const int packed  = 1;
        auto      request = makeRequest(cursor, stride);
        request.store     = StoreSpec::convert(DataType::f32, DataType::f16);
        request.temps     = StoreTemps::withZmm(packed);
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.vmovups(Xbyak::util::ptr[preservedPtr], Xbyak::Zmm(0));
    }
    jit.ready();
    const std::array<float, 16> source{ -100.5f, -3.25f, -1.0f, 0.0f,
                                        0.5f,    1.0f,   2.5f,  7.75f,
                                        16.0f,   32.0f,  64.0f, 128.0f,
                                        255.0f,  512.0f, 0.25f, -0.25f };
    std::array<uint16_t, 16>    output{};
    std::array<float, 16>       preserved{};
    using Kernel = void (*)(const float*, uint16_t*, float*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data(), preserved.data());
    for (size_t i = 0; i < source.size(); ++i) {
        EXPECT_EQ(output[i], toF16Rne(source[i]));
    }
    EXPECT_EQ(preserved, source);
}

TEST(GemmStoreEmitterExecution, DestructiveS32ToF16InPlace)
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
        jit.vmovdqu32(Xbyak::Zmm(0), Xbyak::util::ptr[sourcePtr]);
        jit.mov(stride, 16 * sizeof(uint16_t));
        const int packed  = 1;
        auto      request = makeRequest(cursor, stride);
        request.store     = StoreSpec::convert(DataType::s32, DataType::f16);
        request.temps     = StoreTemps::withZmm(packed);
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.vmovups(Xbyak::util::ptr[convertedPtr], Xbyak::Zmm(0));
    }
    jit.ready();
    const std::array<int32_t, 16> source{ -2048, -127, -1, 0, 1, 2,  15, 127,
                                          255,   512,  7,  8, 9, 10, 11, 12 };
    std::array<uint16_t, 16>      output{};
    std::array<float, 16>         converted{};
    using Kernel = void (*)(const int32_t*, uint16_t*, float*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data(), converted.data());
    for (size_t i = 0; i < source.size(); ++i) {
        const float expected = static_cast<float>(source[i]);
        EXPECT_EQ(converted[i], expected);
        EXPECT_EQ(output[i], toF16Rne(expected));
    }
}

TEST(GemmStoreEmitterExecution, SoftwareBf16FromF32CanonicalConverts)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }
    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 3, 2);
        const auto              sourcePtr    = frame.p[0];
        const auto              cursor       = frame.p[1];
        const auto              preservedPtr = frame.p[2];
        const auto              stride       = frame.t[0];
        const auto              immediate    = frame.t[1];
        jit.vmovups(Xbyak::Zmm(4), Xbyak::util::ptr[sourcePtr]);
        jit.mov(stride, 16 * sizeof(uint16_t));
        const std::array<int, 3> vectors{ 0, 1, 2 };
        const int                gpr     = immediate.getIdx();
        auto                     request = makeRequest(cursor, stride);
        request.tile.accumBaseIdx        = 4;
        request.store = StoreSpec::softwareBf16(DataType::f32);
        request.temps =
            StoreTemps::withZmmAndGpr(vectors[0], Xbyak::Reg64(gpr));
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.vmovups(Xbyak::util::ptr[preservedPtr], Xbyak::Zmm(4));
    }
    jit.ready();
    const std::array<float, 16> source{ -100.5f, -3.25f, -1.0f, 0.0f,
                                        0.5f,    1.0f,   2.5f,  7.75f,
                                        16.0f,   32.0f,  64.0f, 128.0f,
                                        255.0f,  512.0f, 0.25f, -0.25f };
    std::array<uint16_t, 16>    output{};
    std::array<float, 16>       preserved{};
    using Kernel = void (*)(const float*, uint16_t*, float*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data(), preserved.data());
    for (size_t i = 0; i < source.size(); ++i) {
        EXPECT_EQ(output[i], toBf16Rne(source[i]));
    }
    EXPECT_NE(preserved, source);
}

TEST(GemmStoreEmitterExecution, SoftwareBf16FromS32ConvertsInPlace)
{
    if (!hasAvx512Support()) {
        GTEST_SKIP() << "Requires AVX-512";
    }
    Xbyak::CodeGenerator jit(JIT_KERNEL_SIZE, Xbyak::AutoGrow);
    {
        Xbyak::util::StackFrame frame(&jit, 3, 2);
        const auto              sourcePtr    = frame.p[0];
        const auto              cursor       = frame.p[1];
        const auto              convertedPtr = frame.p[2];
        const auto              stride       = frame.t[0];
        const auto              immediate    = frame.t[1];
        jit.vmovdqu32(Xbyak::Zmm(4), Xbyak::util::ptr[sourcePtr]);
        jit.mov(stride, 16 * sizeof(uint16_t));
        const std::array<int, 3> vectors{ 0, 1, 2 };
        const int                gpr     = immediate.getIdx();
        auto                     request = makeRequest(cursor, stride);
        request.tile.accumBaseIdx        = 4;
        request.store = StoreSpec::softwareBf16(DataType::s32);
        request.temps =
            StoreTemps::withZmmAndGpr(vectors[0], Xbyak::Reg64(gpr));
        GemmStoreEmitter<kType> emitter(jit);
        ASSERT_EQ(emitter.emit(request), dlp::jit::jitGeneratorError::success);
        jit.vmovups(Xbyak::util::ptr[convertedPtr], Xbyak::Zmm(4));
    }
    jit.ready();
    const std::array<int32_t, 16> source{ -2048, -127, -1, 0, 1, 2,  15, 127,
                                          255,   512,  7,  8, 9, 10, 11, 12 };
    std::array<uint16_t, 16>      output{};
    std::array<float, 16>         converted{};
    using Kernel = void (*)(const int32_t*, uint16_t*, float*);
    auto kernel  = jit.getCode<Kernel>();
    ASSERT_NE(kernel, nullptr);
    kernel(source.data(), output.data(), converted.data());
    for (size_t i = 0; i < source.size(); ++i) {
        const float expected = static_cast<float>(source[i]);
        EXPECT_EQ(output[i], toBf16Rne(expected));
    }
    EXPECT_NE(converted.front(), static_cast<float>(source.front()));
}
