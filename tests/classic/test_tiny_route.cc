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

#include <limits>

#include "jit/f32_tiny_gemm_kernel.h"
#include "tiny/dlp_gemm_f32_tiny_sgemm.h"

namespace {

struct Call
{
    char  order;
    bool  trans_a;
    bool  trans_b;
    md_t  m, n, k;
    float alpha, beta;
    md_t  lda, ldb, ldc;
};

// Calls the tiny router accepts under the AVX-512 and AVX2 tiers. Each row is
// a shape that the tiny path has a leaf for.
const Call kCalls[] = {
    { 'R', false, false, 8, 32, 32, 1.0f, 0.0f, 32, 32, 32 },
    { 'C', true, false, 8, 8, 32, 1.0f, 0.0f, 32, 32, 8 },
    { 'R', false, true, 6, 40, 128, 1.0f, 0.0f, 128, 128, 40 },
    { 'R', false, false, 12, 32, 128, 1.0f, 0.0f, 128, 32, 32 },
    { 'R', false, false, 1, 32, 32, 1.0f, 0.0f, 32, 32, 32 },
    { 'R', false, false, 8, 32, 32, 0.0f, 0.0f, 32, 32, 32 },
};

dlp_sgemm_tiny_leaf_t
route(const dlp_sgemm_tiny_isa_t& isa, const Call& c)
{
    dlp_sgemm_tiny_layout layout;
    return dlp_sgemm_tiny_route(&isa, c.order, c.trans_a, c.trans_b, c.m, c.n,
                                c.k, c.alpha, c.beta, c.lda, c.ldb, c.ldc,
                                &layout);
}

// Number of calls in kCalls that the route sends to a leaf rather than the
// framework.
int
routed_count(const dlp_sgemm_tiny_isa_t& isa)
{
    int count = 0;
    for (const Call& c : kCalls) {
        if (route(isa, c) != DLP_SGEMM_TINY_NONE) {
            ++count;
        }
    }
    return count;
}

} // namespace

// Rejection: a NaN alpha or beta always goes to the framework.
TEST(TinyRoute, NanScalarsReject)
{
    const dlp_sgemm_tiny_isa_t isa = { true, true, 2, 32 * 1024, 1024 * 1024 };
    const float                nan = std::numeric_limits<float>::quiet_NaN();
    Call                       c   = kCalls[0];
    c.alpha                        = nan;
    EXPECT_EQ(route(isa, c), DLP_SGEMM_TINY_NONE);
    c      = kCalls[0];
    c.beta = nan;
    EXPECT_EQ(route(isa, c), DLP_SGEMM_TINY_NONE);
}

// Rejection: with no ISA tier available, the route never selects a leaf.
TEST(TinyRoute, NoIsaRejectsEveryCall)
{
    const dlp_sgemm_tiny_isa_t isa = { false, false, 1, 32 * 1024,
                                       1024 * 1024 };
    EXPECT_EQ(routed_count(isa), 0);
}

// The AVX-512 tier routes at least one call in the set. A router that declines
// every call fails here instead of passing vacuously.
TEST(TinyRoute, Avx512RoutesSomeCalls)
{
    const dlp_sgemm_tiny_isa_t isa = { true, true, 2, 32 * 1024, 1024 * 1024 };
    EXPECT_GT(routed_count(isa), 0);
}

// The AVX2 tier routes at least one call, and never to the AVX-512 wide leaf.
TEST(TinyRoute, Avx2RoutesSomeCallsWithoutWideLeaf)
{
    const dlp_sgemm_tiny_isa_t isa = { false, true, 1, 32 * 1024, 1024 * 1024 };
    EXPECT_GT(routed_count(isa), 0);
    for (const Call& c : kCalls) {
        EXPECT_NE(route(isa, c), DLP_SGEMM_TINY_RD_WIDE);
    }
}

// Every leaf the route returns has its generated kernel.
TEST(TinyRoute, RoutedLeavesHaveKernels)
{
    const dlp_sgemm_tiny_isa_t isa = { true, true, 2, 32 * 1024, 1024 * 1024 };
    for (const Call& c : kCalls) {
        switch (route(isa, c)) {
            case DLP_SGEMM_TINY_RV:
                EXPECT_NE(dlp_sgemm_tiny_jit_rv_avx512(), nullptr);
                break;
            case DLP_SGEMM_TINY_RD:
                EXPECT_NE(dlp_sgemm_tiny_jit_rd_avx512(), nullptr);
                break;
            case DLP_SGEMM_TINY_RD_WIDE:
                EXPECT_NE(dlp_sgemm_tiny_jit_rd_avx512_wide(), nullptr);
                break;
            case DLP_SGEMM_TINY_NONE:
            case DLP_SGEMM_TINY_SCALE_C:
                break;
            default:
                ADD_FAILURE() << "unknown leaf";
        }
    }
}
