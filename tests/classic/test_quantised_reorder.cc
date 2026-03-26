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

/**
 * @file test_quantised_reorder.cc
 * @brief Tests for symmetric static quantisation reorder APIs
 *
 * Direct C API tests for reorder operations with quantised matrix types.
 * These tests exercise the reorder path independently of GEMM.
 */

#include "adaptors/dlp/ual_dlp.hh"
#include "adaptors/ref/ual_ref.hh"
#include "framework/matrix.hh"
#include "framework/ual.hh"
#include <gtest/gtest.h>

using namespace dlp::testing::framework;
using namespace dlp::testing::classic;

class QuantisedReorderTest : public ::testing::Test
{
  protected:
    std::unique_ptr<UalDlp> ual_dlp;
    std::unique_ptr<UalRef> ual_ref;

    void SetUp() override
    {
        ual_dlp = std::make_unique<UalDlp>();
        ual_ref = std::make_unique<UalRef>();
    }
};

TEST_F(QuantisedReorderTest, ReorderS8MatrixBasic)
{
    md_t rows = 16;
    md_t cols = 16;

    Matrix B_normal(rows, cols, MatrixType::s8, MatrixLayout::ROW_MAJOR);
    B_normal.fillRandom();

    Matrix B_reordered(rows, cols, MatrixType::s8, MatrixLayout::ROW_MAJOR);
    B_reordered.setReordered(true);

    auto status =
        ual_dlp->reorder(B_normal, B_reordered, MatrixType::u8, MatrixType::s8,
                         MatrixType::s32, MatrixType::s32);

    // Reorder may return NOT_SUPPORTED on non-AMD hardware
    EXPECT_TRUE(status == UALError::UAL_SUCCESS
                || status == UALError::UAL_NOT_SUPPORTED);
}

TEST_F(QuantisedReorderTest, ReorderBf16MatrixBasic)
{
    md_t rows = 16;
    md_t cols = 16;

    Matrix B_normal(rows, cols, MatrixType::bf16, MatrixLayout::ROW_MAJOR);
    B_normal.fillRandom();

    Matrix B_reordered(rows, cols, MatrixType::bf16, MatrixLayout::ROW_MAJOR);
    B_reordered.setReordered(true);

    auto status =
        ual_dlp->reorder(B_normal, B_reordered, MatrixType::bf16,
                         MatrixType::bf16, MatrixType::f32, MatrixType::f32);

    EXPECT_TRUE(status == UALError::UAL_SUCCESS
                || status == UALError::UAL_NOT_SUPPORTED);
}

TEST_F(QuantisedReorderTest, ReorderF32MatrixBasic)
{
    md_t rows = 16;
    md_t cols = 16;

    Matrix B_normal(rows, cols, MatrixType::f32, MatrixLayout::ROW_MAJOR);
    B_normal.fillRandom();

    Matrix B_reordered(rows, cols, MatrixType::f32, MatrixLayout::ROW_MAJOR);
    B_reordered.setReordered(true);

    auto status =
        ual_dlp->reorder(B_normal, B_reordered, MatrixType::f32,
                         MatrixType::f32, MatrixType::f32, MatrixType::f32);

    EXPECT_TRUE(status == UALError::UAL_SUCCESS
                || status == UALError::UAL_NOT_SUPPORTED);
}

TEST_F(QuantisedReorderTest, ReorderNonSquareMatrix)
{
    md_t rows = 32;
    md_t cols = 64;

    Matrix B_normal(rows, cols, MatrixType::s8, MatrixLayout::ROW_MAJOR);
    B_normal.fillRandom();

    Matrix B_reordered(rows, cols, MatrixType::s8, MatrixLayout::ROW_MAJOR);
    B_reordered.setReordered(true);

    auto status =
        ual_dlp->reorder(B_normal, B_reordered, MatrixType::bf16,
                         MatrixType::s8, MatrixType::f32, MatrixType::s32);

    EXPECT_TRUE(status == UALError::UAL_SUCCESS
                || status == UALError::UAL_NOT_SUPPORTED);
}
