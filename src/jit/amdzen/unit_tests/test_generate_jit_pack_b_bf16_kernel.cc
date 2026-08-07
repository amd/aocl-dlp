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

#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include "jit/amdzen/pack_b_amdzen_generator.hh"
#include "jit_generator_tests_utils.hh"

using namespace dlp::kernel_frame;
using namespace dlp::kernels;
using namespace dlp::jit;
using namespace amdzen::gen;
using namespace amdzen::utils;
using namespace test_jit_utils;

// ============================================================================
// Reference: BF16 pack-B producing the K-paired interleaved layout.
//
// A panel of storage-width W holds W/16 contiguous 16-wide blocks per K-pair;
// block bj (n = 16*bj..16*bj+15) sits at kp*W + 32*bj, and within a block the
// two K rows are interleaved per n:
//   block[2*t + kk] = B[(kp+kk), c0 + 16*bj + t].
// Lanes beyond validCols (lt-block fringe) and K-rows beyond k (odd-K padding)
// are zero. The full buffer is the cascade of: full NR panels, one base-width
// fringe panel (largest 16-multiple <= n%NR), and one lt-block panel (always 16
// wide in storage, only n%16 lanes live) -- byte-for-byte the intrinsic layout.
//
// The packed output is identical for row- and column-major sources; only how
// B[row, col] is addressed differs, captured by (rs_src, cs_src):
//   * row-major B : rs_src = ld (= n), cs_src = 1
//   * col-major B : rs_src = 1,        cs_src = ld (= k)
// ============================================================================
static void
packPanelBF16(const uint16_t* src,
              uint16_t*       panel,
              md_t            c0,
              md_t            W,
              md_t            validCols,
              md_t            k,
              md_t            KC_updated,
              md_t            rs_src,
              md_t            cs_src)
{
    md_t numBlocks = W / 16;
    for (md_t kp = 0; kp < KC_updated; kp += 2) {
        for (md_t bj = 0; bj < numBlocks; ++bj) {
            uint16_t* block = panel + kp * W + 32 * bj;
            for (md_t t = 0; t < 16; ++t) {
                md_t localCol = 16 * bj + t;
                md_t col      = c0 + localCol;
                for (md_t kk = 0; kk < 2; ++kk) {
                    md_t row = kp + kk;
                    block[2 * t + kk] =
                        (row < k && localCol < validCols)
                            ? src[static_cast<size_t>(row) * rs_src
                                  + static_cast<size_t>(col) * cs_src]
                            : static_cast<uint16_t>(0);
                }
            }
        }
    }
}

static void
referencePackBBF16(const uint16_t* src,
                   uint16_t*       dst,
                   md_t            n,
                   md_t            k,
                   md_t            NR,
                   md_t            rs_src,
                   md_t            cs_src)
{
    md_t KC_updated = (k + 1) & ~static_cast<md_t>(1);

    md_t n_full = (n / NR) * NR;
    for (md_t jc = 0; jc < n_full; jc += NR) {
        packPanelBF16(src, dst + static_cast<size_t>(jc) * KC_updated, jc, NR,
                      NR, k, KC_updated, rs_src, cs_src);
    }

    md_t n_partial = n - n_full;
    md_t baseW     = (n_partial / 16) * 16;
    md_t off       = n_full;
    if (baseW > 0) {
        packPanelBF16(src, dst + static_cast<size_t>(off) * KC_updated, off,
                      baseW, baseW, k, KC_updated, rs_src, cs_src);
        off += baseW;
    }
    md_t r = n_partial - baseW;
    if (r > 0) {
        packPanelBF16(src, dst + static_cast<size_t>(off) * KC_updated, off, 16,
                      r, k, KC_updated, rs_src, cs_src);
    }
}

// Exact packed size for the cascade: full panels + base-width panel + a full
// 16-wide lt-block panel (storage is padded to 16 even when n%16 < 16).
static size_t
packedBufSizeBF16(md_t n, md_t k, md_t NR)
{
    md_t KC_updated = (k + 1) & ~static_cast<md_t>(1);
    md_t n_full     = (n / NR) * NR;
    md_t n_partial  = n - n_full;
    md_t baseW      = (n_partial / 16) * 16;
    md_t r          = n_partial - baseW;
    md_t widthTotal = n_full + baseW + (r > 0 ? 16 : 0);
    return static_cast<size_t>(widthTotal) * KC_updated;
}

// ============================================================================
// Fixture
// ============================================================================

class JitPackBBF16Test : public ::testing::Test
{
  protected:
    bool bf16Supported()
    {
        return !ArchBasedKernelTypes::getAllKernelTypesForBF16().empty();
    }

    std::unique_ptr<jitAmdZenPackBBF16> generate(md_t NR, bool colMajor)
    {
        auto gen = std::make_unique<jitAmdZenPackBBF16>();

        packKernelInfo           packKI(NR, /*k_factor=*/2,
                                        kernelInstrPreference::avx512_zmm_bf16_favour,
                                        DataType::bf16, DataType::bf16, colMajor);
        packBJitGeneratorContext ctx(packKI);

        if ((*gen)(ctx) != jitGeneratorError::success) {
            return nullptr;
        }
        return gen;
    }

    // colMajor=false: row-major source (rs_src = N, cs_src = 1).
    // colMajor=true : column-major source (rs_src = 1, cs_src = K), e.g. transB
    //                 or a column-major B operand. The packed output is
    //                 byte-for byte identical to the row-major case.
    void runCorrectness(md_t NR, md_t N, md_t K, bool colMajor)
    {
        auto gen = generate(NR, colMajor);
        ASSERT_NE(gen, nullptr) << "generation failed NR=" << NR;

        md_t rs_src = colMajor ? 1 : N;
        md_t cs_src = colMajor ? K : 1;

        size_t srcCount = static_cast<size_t>(K) * N;
        size_t dstCount = packedBufSizeBF16(N, K, NR);

        std::vector<uint16_t> srcBuf(srcCount);
        for (size_t i = 0; i < srcCount; ++i)
            srcBuf[i] = static_cast<uint16_t>((i + 1) & 0xFFFF);

        std::vector<uint16_t> refBuf(dstCount, 0);
        referencePackBBF16(srcBuf.data(), refBuf.data(), N, K, NR, rs_src,
                           cs_src);

        // Sentinel fill to catch any unwritten packed byte.
        std::vector<uint16_t> dstBuf(dstCount, 0xCCCC);

        packBParams params(srcBuf.data(), dstBuf.data(), N, K, rs_src, cs_src);
        ASSERT_EQ(gen->executeKernel(&params), kernelError::success);

        std::string label = std::string(colMajor ? "colMajor " : "rowMajor ")
                            + "NR=" + std::to_string(NR) + " N="
                            + std::to_string(N) + " K=" + std::to_string(K);

        int mismatches = 0;
        for (size_t i = 0; i < dstCount && mismatches < 10; ++i) {
            if (dstBuf[i] != refBuf[i]) {
                ++mismatches;
                ADD_FAILURE() << label << " mismatch at " << i
                              << ": jit=" << dstBuf[i] << " ref=" << refBuf[i];
            }
        }
        EXPECT_EQ(params.rs_dst, NR * 2);
        EXPECT_EQ(params.cs_dst, NR / 2);
    }
};

// ============================================================================
// Generation gating
// ============================================================================

TEST_F(JitPackBBF16Test, GenerateValidNR_RowMajor)
{
    if (!bf16Supported())
        GTEST_SKIP() << "AVX512-BF16 not supported";

    for (md_t NR : { 16, 32, 48, 64, 80, 96, 112 }) {
        SCOPED_TRACE("NR=" + std::to_string(NR));
        EXPECT_NE(generate(NR, false), nullptr);
    }
}

TEST_F(JitPackBBF16Test, GenerateValidNR_ColMajor)
{
    if (!bf16Supported())
        GTEST_SKIP() << "AVX512-BF16 not supported";

    for (md_t NR : { 16, 32, 64 }) {
        SCOPED_TRACE("NR=" + std::to_string(NR));
        EXPECT_NE(generate(NR, true), nullptr);
    }
}

TEST_F(JitPackBBF16Test, InvalidNR_ReturnsError)
{
    if (!bf16Supported())
        GTEST_SKIP() << "AVX512-BF16 not supported";

    for (md_t NR : { 0, 1, 8, 15, 17, 33 }) {
        SCOPED_TRACE("NR=" + std::to_string(NR));
        EXPECT_EQ(generate(NR, false), nullptr);
    }
}

// ============================================================================
// Execution correctness
// ============================================================================

TEST_F(JitPackBBF16Test, FullPanels_VariousShapes)
{
    if (!bf16Supported())
        GTEST_SKIP() << "AVX512-BF16 not supported";

    for (md_t NR : { 16, 32, 48, 64 }) {
        for (md_t panels : { 1, 2, 3 }) {
            md_t N = NR * panels;
            for (md_t K : { 1, 2, 3, 7, 8, 16, 17, 32, 64 }) {
                SCOPED_TRACE("NR=" + std::to_string(NR) + " N="
                             + std::to_string(N) + " K=" + std::to_string(K));
                runCorrectness(NR, N, K, /*colMajor=*/false);
            }
        }
    }
}

// Cascade fringe coverage: every n%NR remainder exercises a distinct
// base-width + lt-block combination against the byte-exact reference.
TEST_F(JitPackBBF16Test, NFringe_CascadeShapes)
{
    if (!bf16Supported())
        GTEST_SKIP() << "AVX512-BF16 not supported";

    md_t NR = 64;
    // Remainders touching each base width (48/32/16/0) and lt-block (1..15).
    for (md_t N : { 14, 16, 30, 32, 46, 48, 62, 64, 78, 100, 126, 191, 286 }) {
        for (md_t K : { 1, 2, 3, 7, 8, 16, 17, 32 }) {
            SCOPED_TRACE("NR=" + std::to_string(NR) + " N=" + std::to_string(N)
                         + " K=" + std::to_string(K));
            runCorrectness(NR, N, K, /*colMajor=*/false);
        }
    }
}

// Column-major source (the 16x16 transpose packer): same byte-exact reference,
// addressed column-major. K spans full 32-K tiles, the masked K-fringe, and
// odd-K (zero-padded K-pair partner).
TEST_F(JitPackBBF16Test, FullPanels_VariousShapes_ColMajor)
{
    if (!bf16Supported())
        GTEST_SKIP() << "AVX512-BF16 not supported";

    for (md_t NR : { 16, 32, 48, 64 }) {
        for (md_t panels : { 1, 2, 3 }) {
            md_t N = NR * panels;
            for (md_t K : { 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 64, 65 }) {
                SCOPED_TRACE("NR=" + std::to_string(NR) + " N="
                             + std::to_string(N) + " K=" + std::to_string(K));
                runCorrectness(NR, N, K, /*colMajor=*/true);
            }
        }
    }
}

// Column-major cascade fringe coverage: every n%NR remainder (base width +
// lt-block) against the byte-exact reference.
TEST_F(JitPackBBF16Test, NFringe_CascadeShapes_ColMajor)
{
    if (!bf16Supported())
        GTEST_SKIP() << "AVX512-BF16 not supported";

    md_t NR = 64;
    for (md_t N : { 14, 16, 30, 32, 46, 48, 62, 64, 78, 100, 126, 191, 286 }) {
        for (md_t K : { 1, 2, 3, 7, 16, 17, 31, 32, 33 }) {
            SCOPED_TRACE("NR=" + std::to_string(NR) + " N=" + std::to_string(N)
                         + " K=" + std::to_string(K));
            runCorrectness(NR, N, K, /*colMajor=*/true);
        }
    }
}

TEST_F(JitPackBBF16Test, ZeroN_NoOp)
{
    if (!bf16Supported())
        GTEST_SKIP() << "AVX512-BF16 not supported";

    md_t NR  = 64;
    auto gen = generate(NR, false);
    ASSERT_NE(gen, nullptr);

    uint16_t    src[1] = { 1 };
    uint16_t    dst[1] = { 0xAAAA };
    packBParams params(src, dst, /*n=*/0, /*k=*/16, NR, 1);

    EXPECT_EQ(gen->executeKernel(&params), kernelError::success);
    EXPECT_EQ(dst[0], 0xAAAA) << "N=0 must not touch the destination";
}

TEST_F(JitPackBBF16Test, ExecuteBeforeGenerate_ReturnsError)
{
    auto        gen    = std::make_unique<jitAmdZenPackBBF16>();
    uint16_t    src[4] = {};
    uint16_t    dst[4] = {};
    packBParams params(src, dst, 4, 4, 4, 1);
    EXPECT_EQ(gen->executeKernel(&params), kernelError::error);
}

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
