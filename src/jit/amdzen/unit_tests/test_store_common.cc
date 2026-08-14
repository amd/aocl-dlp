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

#include "jit/amdzen/store/store_common.hh"

#include <gtest/gtest.h>

namespace {

using amdzen::store::registerSizeBytes;
using amdzen::store::RegSpan;
using amdzen::store::SourceRegWidth;
using amdzen::store::StoreMask;
using amdzen::store::valuesPerRegister;
using dlp::kernel_frame::DataType;

TEST(CTypeUtils, DataTypeSizes)
{
    EXPECT_EQ(dlp::utils::dataTypeSizeBits(DataType::s4), 4);
    EXPECT_EQ(dlp::utils::dataTypeSizeBits(DataType::u4), 4);
    EXPECT_EQ(dlp::utils::dataTypeSizeBits(DataType::f4), 4);
    EXPECT_EQ(dlp::utils::dataTypeSizeBits(DataType::s8), 8);
    EXPECT_EQ(dlp::utils::dataTypeSizeBits(DataType::u8), 8);
    EXPECT_EQ(dlp::utils::dataTypeSizeBits(DataType::s16), 16);
    EXPECT_EQ(dlp::utils::dataTypeSizeBits(DataType::u16), 16);
    EXPECT_EQ(dlp::utils::dataTypeSizeBits(DataType::f16), 16);
    EXPECT_EQ(dlp::utils::dataTypeSizeBits(DataType::bf16), 16);
    EXPECT_EQ(dlp::utils::dataTypeSizeBits(DataType::s32), 32);
    EXPECT_EQ(dlp::utils::dataTypeSizeBits(DataType::u32), 32);
    EXPECT_EQ(dlp::utils::dataTypeSizeBits(DataType::f32), 32);
    EXPECT_EQ(dlp::utils::dataTypeSizeBits(DataType::invalid), 0);
    EXPECT_EQ(dlp::utils::dataTypeSizeBits(DataType::max_datatypes), 0);
}

TEST(StoreGeometry, PhysicalRegisterWidths)
{
    EXPECT_EQ(registerSizeBytes(SourceRegWidth::ymm), 32);
    EXPECT_EQ(registerSizeBytes(SourceRegWidth::zmm), 64);
    EXPECT_EQ(valuesPerRegister(SourceRegWidth::ymm, DataType::f32), 8);
    EXPECT_EQ(valuesPerRegister(SourceRegWidth::zmm, DataType::f32), 16);
}

TEST(StoreGeometry, ZmmCountsAndTails)
{
    const auto zero = RegSpan{ 4, SourceRegWidth::zmm, 0 };
    EXPECT_EQ(zero.registerCount(DataType::f32), 0);
    EXPECT_EQ(zero.validValuesInLastRegister(DataType::f32), 0);

    const auto one = RegSpan{ 4, SourceRegWidth::zmm, 1 };
    EXPECT_EQ(one.registerCount(DataType::f32), 1);
    EXPECT_EQ(one.validValuesInLastRegister(DataType::f32), 1);

    const auto exact = RegSpan{ 4, SourceRegWidth::zmm, 16 };
    EXPECT_EQ(exact.registerCount(DataType::f32), 1);
    EXPECT_EQ(exact.validValuesInLastRegister(DataType::f32), 16);

    const auto fullPlusOne = RegSpan{ 4, SourceRegWidth::zmm, 17 };
    EXPECT_EQ(fullPlusOne.registerCount(DataType::f32), 2);
    EXPECT_EQ(fullPlusOne.validValuesInLastRegister(DataType::f32), 1);

    const auto two = RegSpan{ 4, SourceRegWidth::zmm, 32 };
    EXPECT_EQ(two.registerCount(DataType::f32), 2);
    EXPECT_EQ(two.validValuesInLastRegister(DataType::f32), 16);
}

TEST(StoreGeometry, YmmAndSourceTypeSensitivity)
{
    const auto span = RegSpan{ 2, SourceRegWidth::ymm, 16 };
    EXPECT_EQ(span.valuesPerRegister(DataType::f32), 8);
    EXPECT_EQ(span.registerCount(DataType::f32), 2);
    EXPECT_EQ(span.valuesPerRegister(DataType::bf16), 16);
    EXPECT_EQ(span.registerCount(DataType::bf16), 1);
    EXPECT_EQ(span.valuesPerRegister(DataType::u8), 32);
    EXPECT_EQ(span.registerCount(DataType::u8), 1);
    EXPECT_EQ(span.valuesPerRegister(DataType::s4), 64);
    EXPECT_EQ(span.registerCount(DataType::s4), 1);
}

TEST(StoreGeometry, ValuesInRegisterChecksBounds)
{
    const auto span = RegSpan{ 7, SourceRegWidth::zmm, 17 };
    EXPECT_EQ(span.valuesInRegister(-1, DataType::f32), 0);
    EXPECT_EQ(span.valuesInRegister(0, DataType::f32), 16);
    EXPECT_EQ(span.valuesInRegister(1, DataType::f32), 1);
    EXPECT_EQ(span.valuesInRegister(2, DataType::f32), 0);
    EXPECT_EQ(span.valuesInRegister(0, DataType::invalid), 0);
}

TEST(StoreMask, AppliesOnlyRequestedOpmask)
{
    const auto address = Xbyak::util::ptr[Xbyak::Reg64(0)];
    const auto none    = StoreMask::none().apply(address);
    const auto masked  = StoreMask::opmask(Xbyak::Opmask(3)).apply(address);

    EXPECT_EQ(none.getOpmaskIdx(), 0);
    EXPECT_EQ(masked.getOpmaskIdx(), 3);
    EXPECT_EQ(masked.getRounding(), 0);
}

} // namespace
