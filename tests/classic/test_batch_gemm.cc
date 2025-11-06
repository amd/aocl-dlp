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

#include "framework/matrix.hh"
#include "framework/ual.hh"
#include "framework/ual_factory.hh"
#include <gtest/gtest.h>

using namespace dlp::testing::framework;

namespace {

BatchGroup
makeF32Group(md_t         m,
             md_t         n,
             md_t         k,
             size_t       matrix_count,
             double       alpha,
             double       beta,
             uint32_t     seed_offset,
             MatrixLayout layoutA = MatrixLayout::ROW_MAJOR,
             MatrixLayout layoutB = MatrixLayout::ROW_MAJOR,
             MatrixLayout layoutC = MatrixLayout::ROW_MAJOR,
             bool         transA  = false,
             bool         transB  = false)
{
    BatchGroup group;
    group.m     = m;
    group.n     = n;
    group.k     = k;
    group.alpha = alpha;
    group.beta  = beta;

    for (size_t i = 0; i < matrix_count; ++i) {
        md_t a_rows = transA ? k : m;
        md_t a_cols = transA ? m : k;
        md_t b_rows = transB ? n : k;
        md_t b_cols = transB ? k : n;

        Matrix A(a_rows, a_cols, MatrixType::f32, layoutA, 0, transA);
        Matrix B(b_rows, b_cols, MatrixType::f32, layoutB, 0, transB);
        Matrix C(m, n, MatrixType::f32, layoutC, 0, false);

        A.fillRandom(static_cast<uint32_t>(42 + seed_offset + i));
        B.fillRandom(static_cast<uint32_t>(142 + seed_offset + i));
        C.fillRandom(static_cast<uint32_t>(242 + seed_offset + i));

        group.A_matrices.emplace_back(std::move(A));
        group.B_matrices.emplace_back(std::move(B));
        group.C_matrices.emplace_back(std::move(C));
    }

    group.memFormatA = deduce_mem_format(group.A_matrices.front());
    group.memFormatB = deduce_mem_format(group.B_matrices.front());
    return group;
}

std::vector<BatchGroup>
cloneGroups(const std::vector<BatchGroup>& groups)
{
    std::vector<BatchGroup> copies;
    copies.reserve(groups.size());
    for (const auto& group : groups) {
        copies.push_back(group);
    }
    return copies;
}

void
compareGroupResults(const std::vector<BatchGroup>& lhs,
                    const std::vector<BatchGroup>& rhs)
{
    ASSERT_EQ(lhs.size(), rhs.size());
    for (size_t g = 0; g < lhs.size(); ++g) {
        const auto& lhs_group = lhs[g];
        const auto& rhs_group = rhs[g];

        ASSERT_EQ(lhs_group.C_matrices.size(), rhs_group.C_matrices.size());

        for (size_t i = 0; i < lhs_group.C_matrices.size(); ++i) {
            const_cast<Matrix&>(lhs_group.C_matrices[i]).setK(lhs_group.k);
            const_cast<Matrix&>(rhs_group.C_matrices[i]).setK(rhs_group.k);

            auto result = lhs_group.C_matrices[i].compare(
                rhs_group.C_matrices[i], MatrixCompareOptions::Fast());
            EXPECT_TRUE(result.equal) << FormatCompareResult(
                result, lhs_group.C_matrices[i], rhs_group.C_matrices[i]);
        }
    }
}

} // namespace

TEST(BatchGemmTest, SingleGroupSingleMatrix)
{
    auto base_groups = std::vector<BatchGroup>{
        makeF32Group(/*m=*/8, /*n=*/8, /*k=*/8, /*matrix_count=*/1,
                     /*alpha=*/1.0, /*beta=*/0.0, /*seed_offset=*/0),
    };

    auto dlp_groups = cloneGroups(base_groups);
    auto ref_groups = cloneGroups(base_groups);

    auto ual_dlp = UalFactory::createUal(UALType::DLP);
    auto ual_ref = UalFactory::createUal(UALType::REF);
    ASSERT_NE(ual_dlp, nullptr);
    ASSERT_NE(ual_ref, nullptr);

    auto status_dlp = ual_dlp->batch_gemm(dlp_groups, MatrixType::f32);
    ASSERT_EQ(status_dlp, UALError::UAL_SUCCESS);

    auto status_ref = ual_ref->batch_gemm(ref_groups, MatrixType::f32);
    ASSERT_EQ(status_ref, UALError::UAL_SUCCESS);

    compareGroupResults(dlp_groups, ref_groups);
}

TEST(BatchGemmTest, MultipleGroupsMultipleMatrices)
{
    auto base_groups = std::vector<BatchGroup>{
        makeF32Group(/*m=*/6, /*n=*/4, /*k=*/5, /*matrix_count=*/2,
                     /*alpha=*/1.25, /*beta=*/0.1, /*seed_offset=*/10),
        makeF32Group(/*m=*/12, /*n=*/7, /*k=*/9, /*matrix_count=*/3,
                     /*alpha=*/0.75, /*beta=*/0.2, /*seed_offset=*/30),
    };

    auto dlp_groups = cloneGroups(base_groups);
    auto ref_groups = cloneGroups(base_groups);

    auto ual_dlp = UalFactory::createUal(UALType::DLP);
    auto ual_ref = UalFactory::createUal(UALType::REF);
    ASSERT_NE(ual_dlp, nullptr);
    ASSERT_NE(ual_ref, nullptr);

    auto status_dlp = ual_dlp->batch_gemm(dlp_groups, MatrixType::f32);
    ASSERT_EQ(status_dlp, UALError::UAL_SUCCESS);

    auto status_ref = ual_ref->batch_gemm(ref_groups, MatrixType::f32);
    ASSERT_EQ(status_ref, UALError::UAL_SUCCESS);

    compareGroupResults(dlp_groups, ref_groups);
}

TEST(BatchGemmTest, MixedGroupConfigurations)
{
    auto base_groups = std::vector<BatchGroup>{
        makeF32Group(/*m=*/5, /*n=*/3, /*k=*/4, /*matrix_count=*/1,
                     /*alpha=*/0.9, /*beta=*/0.3, /*seed_offset=*/50,
                     MatrixLayout::ROW_MAJOR, MatrixLayout::ROW_MAJOR,
                     MatrixLayout::ROW_MAJOR, /*transA=*/false,
                     /*transB=*/true),
        makeF32Group(/*m=*/16, /*n=*/16, /*k=*/8, /*matrix_count=*/4,
                     /*alpha=*/1.4, /*beta=*/-0.2, /*seed_offset=*/100,
                     MatrixLayout::ROW_MAJOR, MatrixLayout::ROW_MAJOR,
                     MatrixLayout::ROW_MAJOR, /*transA=*/true,
                     /*transB=*/false),
        makeF32Group(/*m=*/9, /*n=*/11, /*k=*/7, /*matrix_count=*/2,
                     /*alpha=*/0.6, /*beta=*/0.5, /*seed_offset=*/300,
                     MatrixLayout::ROW_MAJOR, MatrixLayout::ROW_MAJOR,
                     MatrixLayout::ROW_MAJOR, /*transA=*/false,
                     /*transB=*/false),
    };

    auto dlp_groups = cloneGroups(base_groups);
    auto ref_groups = cloneGroups(base_groups);

    auto ual_dlp = UalFactory::createUal(UALType::DLP);
    auto ual_ref = UalFactory::createUal(UALType::REF);
    ASSERT_NE(ual_dlp, nullptr);
    ASSERT_NE(ual_ref, nullptr);

    auto status_dlp = ual_dlp->batch_gemm(dlp_groups, MatrixType::f32);
    ASSERT_EQ(status_dlp, UALError::UAL_SUCCESS);

    auto status_ref = ual_ref->batch_gemm(ref_groups, MatrixType::f32);
    ASSERT_EQ(status_ref, UALError::UAL_SUCCESS);

    compareGroupResults(dlp_groups, ref_groups);
}
