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

/*
 * Regression tests for the leading-dimension / matrix-dimension upper-bound
 * checks in the classic input validators. Each public entry point must reject
 * a dimension above DLP_MAX_GEMM_DIM via its error code rather than feeding
 * the out-of-range value into stride/offset math (CWE-190 -> CWE-787/125).
 *
 * These call the raw C API directly (rather than the UAL harness) so the
 * assertions land on the validator's error code. Note that a *missing* bound
 * does not fail cleanly: the oversized dimension reaches stride-indexed offset
 * math and the process takes SIGSEGV, which kills the whole binary and masks
 * every later case in this file. A crash here should therefore be read as
 * "a dimension bound regressed", not as test-harness flakiness.
 */

#include "classic/aocl_eltwise_ops_interface_apis.h"
#include "classic/aocl_gemm_interface_apis.h"
#include "classic/aocl_util_interface_apis.h"
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

namespace {

// Well above DLP_MAX_GEMM_DIM. md_t is int64_t, so this is a live value the
// bound must catch, not an overflowed constant.
constexpr md_t kOverMax = static_cast<md_t>(INT32_MAX) + 100;

// 2^30 used to pass the old INT32_MAX cap and was then consumed as a stride.
constexpr md_t kBillionScale = (md_t)1 << 30;

dlp_metadata_t
make_metadata()
{
    dlp_metadata_t md;
    std::memset(&md, 0, sizeof(md));
    md.error_hndl.error_code = DLP_CLSC_SUCCESS;
    return md;
}

} // namespace

// --- single GEMM ------------------------------------------------------------

TEST(DimBounds, GemmRejectsOversizedLda)
{
    const md_t         m = 4, n = 4, k = 4;
    std::vector<float> a(m * k, 1.0f), b(k * n, 1.0f), c(m * n, 0.0f);
    dlp_metadata_t     md = make_metadata();

    aocl_gemm_f32f32f32of32('r', 'n', 'n', m, n, k, 1.0f, a.data(), kOverMax,
                            'n', b.data(), n, 'n', 0.0f, c.data(), n, &md);

    if (md.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED)
        GTEST_SKIP() << "f32 GEMM not supported on this processor";
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_LEADING_DIMENSION);
}

TEST(DimBounds, GemmRejectsOversizedLdc)
{
    const md_t         m = 4, n = 4, k = 4;
    std::vector<float> a(m * k, 1.0f), b(k * n, 1.0f), c(m * n, 0.0f);
    dlp_metadata_t     md = make_metadata();

    aocl_gemm_f32f32f32of32('r', 'n', 'n', m, n, k, 1.0f, a.data(), k, 'n',
                            b.data(), n, 'n', 0.0f, c.data(), kOverMax, &md);

    if (md.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED)
        GTEST_SKIP() << "f32 GEMM not supported on this processor";
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_LEADING_DIMENSION);
}

TEST(DimBounds, GemmAcceptsInRangeLeadingDims)
{
    const md_t         m = 4, n = 4, k = 4;
    std::vector<float> a(m * k, 1.0f), b(k * n, 1.0f), c(m * n, 0.0f);
    dlp_metadata_t     md = make_metadata();

    aocl_gemm_f32f32f32of32('r', 'n', 'n', m, n, k, 1.0f, a.data(), k, 'n',
                            b.data(), n, 'n', 0.0f, c.data(), n, &md);

    // A valid, in-range call must never be rejected by a dimension bound.
    EXPECT_NE(md.error_hndl.error_code, DLP_CLSC_INVALID_LEADING_DIMENSION);
    EXPECT_NE(md.error_hndl.error_code, DLP_CLSC_INVALID_MATRIX_DIMENSION);
}

// --- reorder ----------------------------------------------------------------

TEST(DimBounds, ReorderRejectsOversizedLdb)
{
    const md_t         k = 8, n = 8;
    std::vector<float> in(k * n, 1.0f), out(k * n + 4096, 0.0f);
    dlp_metadata_t     md = make_metadata();

    aocl_reorder_f32f32f32of32('r', 'n', 'B', in.data(), out.data(), k, n,
                               kOverMax, &md);

    if (md.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED)
        GTEST_SKIP() << "f32 reorder not supported on this processor";
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_LEADING_DIMENSION);
}

// --- eltwise ops (added in response to PR #610 review) ----------------------

TEST(DimBounds, EltwiseRejectsOversizedM)
{
    const md_t         n = 4;
    std::vector<float> a(64, 1.0f), b(64, 0.0f);
    dlp_metadata_t     md = make_metadata();

    // kOverMax is passed as m -- the oversized dimension under test.
    aocl_gemm_eltwise_ops_f32of32('r', 'n', 'n', kOverMax, n, a.data(), n,
                                  b.data(), n, &md);

    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_MATRIX_DIMENSION);
}

TEST(DimBounds, EltwiseRejectsOversizedLda)
{
    const md_t         m = 4, n = 4;
    std::vector<float> a(64, 1.0f), b(64, 0.0f);
    dlp_metadata_t     md = make_metadata();

    // lda is >= n (passes the minimum-stride test) but exceeds INT32_MAX.
    aocl_gemm_eltwise_ops_f32of32('r', 'n', 'n', m, n, a.data(), kOverMax,
                                  b.data(), n, &md);

    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_LEADING_DIMENSION);
}

TEST(DimBounds, EltwiseRejectsOversizedLdb)
{
    const md_t         m = 4, n = 4;
    std::vector<float> a(64, 1.0f), b(64, 0.0f);
    dlp_metadata_t     md = make_metadata();

    aocl_gemm_eltwise_ops_f32of32('r', 'n', 'n', m, n, a.data(), n, b.data(),
                                  kOverMax, &md);

    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_LEADING_DIMENSION);
}

// --- batch GEMM -------------------------------------------------------------

TEST(DimBounds, BatchGemmRejectsOversizedLda)
{
    const md_t         m = 4, n = 4, k = 4;
    std::vector<float> a(m * k, 1.0f), b(k * n, 1.0f), c(m * n, 0.0f);
    const float*       ap = a.data();
    const float*       bp = b.data();
    float*             cp = c.data();

    char  order = 'r', ta = 'n', tb = 'n', mfa = 'n', mfb = 'n';
    float alpha = 1.0f, beta = 0.0f;
    md_t  bm = m, bn = n, bk = k, lda = kOverMax, ldb = n, ldc = n, gsize = 1;

    dlp_metadata_t  meta  = make_metadata();
    dlp_metadata_t* metap = &meta;

    aocl_batch_gemm_f32f32f32of32(&order, &ta, &tb, &bm, &bn, &bk, &alpha, &ap,
                                  &lda, &bp, &ldb, &beta, &cp, &ldc,
                                  /*group_count=*/1, &gsize, &mfa, &mfb,
                                  &metap);

    if (meta.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED)
        GTEST_SKIP() << "f32 batch GEMM not supported on this processor";
    EXPECT_EQ(meta.error_hndl.error_code, DLP_CLSC_INVALID_LEADING_DIMENSION);
}

// --- unreorder --------------------------------------------------------------

TEST(DimBounds, UnreorderRejectsOversizedLdb)
{
    const md_t         k = 8, n = 8;
    std::vector<float> in(k * n + 4096, 1.0f), out(k * n + 4096, 0.0f);
    dlp_metadata_t     md = make_metadata();

    aocl_unreorder_f32f32f32of32_reference('r', 'n', 'B', in.data(), out.data(),
                                           k, n, kOverMax, &md);

    if (md.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED)
        GTEST_SKIP() << "f32 unreorder not supported on this processor";
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_LEADING_DIMENSION);
}

// 2^30 is under the old INT32_MAX cap. It must now be rejected on its own,
// with the other dimensions small enough that only the argument under test
// trips the bound.
TEST(DimBounds, GemmRejectsBillionScaleN)
{
    const md_t         m = 4, n = kBillionScale, k = 4;
    std::vector<float> a((size_t)m * k, 1.0f), b((size_t)k * 4, 1.0f),
        c((size_t)m * 4, 0.0f);
    dlp_metadata_t md = make_metadata();

    aocl_gemm_f32f32f32of32('r', 'n', 'n', m, n, k, 1.0f, a.data(), k, 'n',
                            b.data(), 4, 'n', 0.0f, c.data(), 4, &md);

    if (md.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED)
        GTEST_SKIP() << "f32 GEMM not supported on this processor";
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_MATRIX_DIMENSION);
    EXPECT_EQ(c[0], 0.0f);
}

TEST(DimBounds, GemmRejectsBillionScaleLdb)
{
    // Column-major, B transposed: the minimum ldb is n, so 2^30 passes the
    // lower bound and has to be stopped by the upper bound.
    const md_t         m = 22, n = 16, k = 64;
    std::vector<float> a((size_t)m * k, 1.0f), b((size_t)k * n, 1.0f),
        c((size_t)m * n, 0.0f);
    dlp_metadata_t md = make_metadata();

    aocl_gemm_f32f32f32of32('c', 'n', 't', m, n, k, 1.0f, a.data(), m, 'n',
                            b.data(), kBillionScale, 'n', 0.0f, c.data(), m,
                            &md);

    if (md.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED)
        GTEST_SKIP() << "f32 GEMM not supported on this processor";
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_LEADING_DIMENSION);
    EXPECT_EQ(c[0], 0.0f);
}

// GELU / softmax return void. A rejected call must leave the buffer alone.
// A missing bound walks off this allocation and the process dies.
TEST(DimBounds, L1OpsRejectBillionScaleLengthAndStride)
{
    std::vector<float> x(8, 3.5f);

    aocl_gemm_gelu_tanh_f32(kBillionScale, x.data(), 1);
    aocl_gemm_gelu_erf_f32(kBillionScale, x.data(), 1);
    aocl_gemm_softmax_f32(kBillionScale, x.data(), 1);
    for (float v : x)
        EXPECT_EQ(v, 3.5f);

    aocl_gemm_gelu_tanh_f32(4, x.data(), kBillionScale);
    aocl_gemm_gelu_erf_f32(4, x.data(), kBillionScale);
    aocl_gemm_softmax_f32(4, x.data(), kBillionScale);
    for (float v : x)
        EXPECT_EQ(v, 3.5f);

    // Each of n and incx is under 2^28, but the last element index is not.
    const md_t wide = ((md_t)1 << 27) + 1;
    aocl_gemm_gelu_tanh_f32(3, x.data(), wide);
    aocl_gemm_gelu_erf_f32(3, x.data(), wide);
    aocl_gemm_softmax_f32(3, x.data(), wide);
    for (float v : x)
        EXPECT_EQ(v, 3.5f);
}

// The matrix_add operand's ldm is a leading dimension, so it gets the same
// upper bound as lda/ldb/ldc.
TEST(DimBounds, MatrixAddRejectsBillionScaleLdm)
{
    const md_t         m = 4, n = 4, k = 4;
    std::vector<float> a(m * k, 1.0f), b(k * n, 1.0f), c(m * n, 0.0f);
    std::vector<float> addend(m * n, 1.0f);

    dlp_post_op_matrix_add add;
    std::memset(&add, 0, sizeof(add));
    add.matrix    = addend.data();
    add.ldm       = kBillionScale;
    add.stor_type = DLP_F32;
    add.sf        = NULL;

    DLP_POST_OP_TYPE seq[1] = { MATRIX_ADD };
    dlp_metadata_t   md     = make_metadata();
    md.seq_vector           = seq;
    md.seq_length           = 1;
    md.matrix_add           = &add;

    aocl_gemm_f32f32f32of32('r', 'n', 'n', m, n, k, 1.0f, a.data(), k, 'n',
                            b.data(), n, 'n', 0.0f, c.data(), n, &md);

    if (md.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED)
        GTEST_SKIP() << "f32 GEMM not supported on this processor";
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_LEADING_DIMENSION);
}

TEST(DimBounds, GeluTanhStillRunsInRange)
{
    std::vector<float> x = { 1.0f, 2.0f, -1.0f, 0.5f };
    aocl_gemm_gelu_tanh_f32((md_t)x.size(), x.data(), 1);
    // AVX2 is the f32 baseline. A no-op here means the new bound rejected a
    // legal vector.
    EXPECT_NE(x[0], 1.0f);
}
