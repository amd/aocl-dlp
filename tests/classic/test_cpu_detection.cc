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
 * Regression tests for AOCL_DLP_ENABLE_INSTRUCTIONS=zen3 on VNNI / BF16 capable
 * hardware (CPUPL-9424).
 *
 * The test is vendor-neutral: it runs on any x86 CPU (AMD or Intel), because
 * the regression it guards is in the raw-hardware ISA gates, which reproduce
 * on any VNNI / BF16 capable part. Non-x86 builds skip the suite. The override
 * token "zen3" is accepted by the library regardless of CPU vendor, so setting
 * it here does not change behaviour beyond what the library already allows.
 *
 * With the override the configured arch has no AVX512-VNNI / AVX512-BF16, so
 * the int8 and bf16 s4/u4 weight-only-quant entry points must report
 * DLP_CLSC_NOT_SUPPORTED (for batch APIs: in every group's metadata). Before
 * the fix they gated on the raw hardware flags:
 *   - int8 passed the gate, found no JIT kernel and returned
 *     DLP_CLSC_INVALID_JIT_KERNEL;
 *   - bf16s4 passed the gate and (M == 1, N >= 32) silently returned wrong
 *     results, because B was reordered for the zen4 packer but read by the
 *     NR = 16 F32 JIT kernel.
 *
 * The architecture is fixed the first time the library is used, so the
 * override is set from a static initializer, before any DLP call, and this
 * file is its own executable. On x86 parts that really lack VNNI / BF16 the
 * APIs return NOT_SUPPORTED with or without the override, so the tests pass
 * trivially there; they guard the regression on VNNI / BF16 capable hardware
 * (AMD Zen4 / Zen5 and Intel AVX512-VNNI / BF16 parts).
 */
#include "aocl_dlp.h"
#include "framework/utils/process_env.hh"
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

namespace {

// Compile-time x86 check (covers MSVC, GCC and Clang on x86 and x86-64).
// Non-x86 builds skip the suite. No CPUID query is needed: the outcome under
// the override does not depend on the CPU vendor.
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__)                 \
    || defined(__i386__)
constexpr bool kIsX86 = true;
#else
constexpr bool kIsX86 = false;
#endif

// Runs before main() and before any library call.
const bool kZen3OverrideApplied =
    kIsX86
    && dlp::testing::utils::setEnvironmentVariable(
        "AOCL_DLP_ENABLE_INSTRUCTIONS", "zen3", true);

dlp_metadata_t
make_metadata()
{
    dlp_metadata_t md;
    std::memset(&md, 0, sizeof(md));
    md.error_hndl.error_code = DLP_CLSC_SUCCESS;
    return md;
}

// Per-channel f32 dequant scales for a 4-bit B matrix, as a WOQ caller
// supplies them. u4 is asymmetric and additionally needs per-channel zero
// points; without them the call fails argument validation (NULL_POINTER)
// instead of reaching the arch gate under test.
struct WoqParams
{
    std::vector<float>  scales;
    std::vector<int8_t> zero_points;
    dlp_qparam_t        sf;
    dlp_qparam_t        zp;
    dlp_quant_op_t      qop;

    WoqParams(md_t n, DLP_TYPE src_type)
        : scales(n, 1.0f)
        , zero_points(n, 0)
    {
        std::memset(&sf, 0, sizeof(sf));
        std::memset(&zp, 0, sizeof(zp));
        std::memset(&qop, 0, sizeof(qop));
        if (src_type == DLP_U4) {
            zp.data        = zero_points.data();
            zp.len         = n;
            zp.stor_type   = DLP_S8;
            zp.outer_dim   = DLP_PARAM_DIM_PER_CHANNEL;
            qop.zero_point = &zp;
        }
        sf.data                   = scales.data();
        sf.len                    = n;
        sf.stor_type              = DLP_F32;
        sf.outer_dim              = DLP_PARAM_DIM_PER_CHANNEL;
        qop.quant_op_kind         = DLP_QUANT_OP_DEQUANTIZE;
        qop.src_type              = src_type;
        qop.dst_type              = DLP_BF16;
        qop.group_size            = 0;
        qop.dequant_scale_factors = &sf;
    }
};

// Valid arguments for a two-group batch call (one matrix per group). Both
// groups must see DLP_CLSC_NOT_SUPPORTED, i.e. the per-group metadata loop in
// the arch gate has to run for every group.
template<typename AT, typename BT, typename CT, typename S>
struct BatchArgs
{
    static constexpr md_t kGroups = 2;

    md_t            m, n, k;
    std::vector<AT> a;
    std::vector<BT> b;
    std::vector<CT> c[kGroups];
    const AT*       ap[kGroups];
    const BT*       bp[kGroups];
    CT*             cp[kGroups];
    md_t            ms[kGroups], ns[kGroups], ks[kGroups];
    md_t            lda[kGroups], ldb[kGroups], ldc[kGroups];
    md_t            group_size[kGroups];
    char            order[kGroups], transa[kGroups], transb[kGroups];
    char            fmt_a[kGroups], fmt_b[kGroups];
    S               alpha[kGroups], beta[kGroups];
    dlp_metadata_t  md[kGroups];
    dlp_metadata_t* mdp[kGroups];

    // b_elems is the number of BT elements in B (packed for 4-bit types).
    BatchArgs(md_t m_, md_t n_, md_t k_, size_t b_elems, AT a_val, BT b_val)
        : m(m_)
        , n(n_)
        , k(k_)
        , a(static_cast<size_t>(m_ * k_), a_val)
        , b(b_elems, b_val)
    {
        for (md_t g = 0; g < kGroups; g++) {
            c[g].assign(static_cast<size_t>(m * n), CT{});
            ap[g]         = a.data();
            bp[g]         = b.data();
            cp[g]         = c[g].data();
            ms[g]         = m;
            ns[g]         = n;
            ks[g]         = k;
            lda[g]        = k;
            ldb[g]        = n;
            ldc[g]        = n;
            group_size[g] = 1;
            order[g]      = 'r';
            transa[g]     = 'n';
            transb[g]     = 'n';
            fmt_a[g]      = 'n';
            fmt_b[g]      = 'n';
            alpha[g]      = S(1);
            beta[g]       = S(0);
            md[g]         = make_metadata();
            mdp[g]        = &md[g];
        }
    }

    void set_quant_op(dlp_quant_op_t* qop)
    {
        for (md_t g = 0; g < kGroups; g++) {
            md[g].b_quant_op = qop;
        }
    }

    template<typename Fn>
    void run(Fn fn)
    {
        fn(order, transa, transb, ms, ns, ks, alpha, ap, lda, bp, ldb, beta, cp,
           ldc, kGroups, group_size, fmt_a, fmt_b, mdp);
    }

    void expect_not_supported() const
    {
        for (md_t g = 0; g < kGroups; g++) {
            EXPECT_EQ(md[g].error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED)
                << "group " << g;
        }
    }
};

constexpr uint16_t kBf16One = 0x3F80; // 1.0f

} // namespace

class CpuDetection : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        if (!kIsX86) {
            GTEST_SKIP()
                << "The configured-arch ISA gate is x86-only. Skipped: "
                   "this build is not x86.";
        }
        ASSERT_TRUE(kZen3OverrideApplied)
            << "Failed to set AOCL_DLP_ENABLE_INSTRUCTIONS=zen3";
    }
};

// --- int8: single GEMM ------------------------------------------------------

TEST_F(CpuDetection, U8S8S32OS32ReturnsNotSupported)
{
    const md_t           m = 16, n = 64, k = 32;
    std::vector<uint8_t> a(m * k, 1);
    std::vector<int8_t>  b(k * n, 1);
    std::vector<int32_t> c(m * n, 0);
    dlp_metadata_t       md = make_metadata();

    aocl_gemm_u8s8s32os32('r', 'n', 'n', m, n, k, 1, a.data(), k, 'n', b.data(),
                          n, 'n', 0, c.data(), n, &md);

    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST_F(CpuDetection, S8S8S32OS32ReturnsNotSupported)
{
    const md_t           m = 16, n = 64, k = 32;
    std::vector<int8_t>  a(m * k, 1), b(k * n, 1);
    std::vector<int32_t> c(m * n, 0);
    dlp_metadata_t       md = make_metadata();

    aocl_gemm_s8s8s32os32('r', 'n', 'n', m, n, k, 1, a.data(), k, 'n', b.data(),
                          n, 'n', 0, c.data(), n, &md);

    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

// --- int8: batch GEMM, unsigned and signed A --------------------------------

TEST_F(CpuDetection, BatchU8S8S32OS32ReturnsNotSupported)
{
    BatchArgs<uint8_t, int8_t, int32_t, int32_t> args(16, 64, 32, 32 * 64, 1,
                                                      1);
    args.run(aocl_batch_gemm_u8s8s32os32);
    args.expect_not_supported();
}

TEST_F(CpuDetection, BatchS8S8S32OS32ReturnsNotSupported)
{
    BatchArgs<int8_t, int8_t, int32_t, int32_t> args(16, 64, 32, 32 * 64, 1, 1);
    args.run(aocl_batch_gemm_s8s8s32os32);
    args.expect_not_supported();
}

// --- bf16 x s4 weight-only quant: single GEMM -------------------------------

// M = 1, N = 64: the shape that returned wrong results before the fix.
TEST_F(CpuDetection, Bf16S4F32OF32ReorderAndGemmReturnNotSupported)
{
    const md_t            m = 1, n = 64, k = 32;
    std::vector<bfloat16> a(m * k, kBf16One);
    std::vector<int8_t>   b(k * n / 2, 0x11);
    std::vector<float>    c(m * n, 0.0f);
    WoqParams             woq(n, DLP_S4);

    dlp_metadata_t md_size = make_metadata();
    md_size.b_quant_op     = &woq.qop;
    aocl_get_reorder_buf_size_bf16s4f32of32('r', 'n', 'B', k, n, &md_size);
    EXPECT_EQ(md_size.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);

    dlp_metadata_t md = make_metadata();
    md.b_quant_op     = &woq.qop;
    // B is never read once the arch gate rejects the call.
    aocl_gemm_bf16s4f32of32('r', 'n', 'n', m, n, k, 1.0f, a.data(), k, 'n',
                            b.data(), n, 'R', 0.0f, c.data(), n, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST_F(CpuDetection, Bf16S4F32OBF16ReturnsNotSupported)
{
    const md_t            m = 1, n = 64, k = 32;
    std::vector<bfloat16> a(m * k, kBf16One), c(m * n, 0);
    std::vector<int8_t>   b(k * n / 2, 0x11);
    WoqParams             woq(n, DLP_S4);

    dlp_metadata_t md = make_metadata();
    md.b_quant_op     = &woq.qop;
    aocl_gemm_bf16s4f32obf16('r', 'n', 'n', m, n, k, 1.0f, a.data(), k, 'n',
                             b.data(), n, 'R', 0.0f, c.data(), n, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

// --- bf16 x u4 weight-only quant: single GEMM, both output variants ---------

TEST_F(CpuDetection, Bf16U4F32OF32ReturnsNotSupported)
{
    const md_t            m = 1, n = 64, k = 32;
    std::vector<bfloat16> a(m * k, kBf16One);
    std::vector<uint8_t>  b(k * n / 2, 0x11);
    std::vector<float>    c(m * n, 0.0f);
    WoqParams             woq(n, DLP_U4);

    dlp_metadata_t md = make_metadata();
    md.b_quant_op     = &woq.qop;
    aocl_gemm_bf16u4f32of32('r', 'n', 'n', m, n, k, 1.0f, a.data(), k, 'n',
                            b.data(), n, 'R', 0.0f, c.data(), n, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST_F(CpuDetection, Bf16U4F32OBF16ReturnsNotSupported)
{
    const md_t            m = 1, n = 64, k = 32;
    std::vector<bfloat16> a(m * k, kBf16One), c(m * n, 0);
    std::vector<uint8_t>  b(k * n / 2, 0x11);
    WoqParams             woq(n, DLP_U4);

    dlp_metadata_t md = make_metadata();
    md.b_quant_op     = &woq.qop;
    aocl_gemm_bf16u4f32obf16('r', 'n', 'n', m, n, k, 1.0f, a.data(), k, 'n',
                             b.data(), n, 'R', 0.0f, c.data(), n, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

// --- bf16 x s4 / u4 weight-only quant: batch GEMM, both output variants -----

TEST_F(CpuDetection, BatchBf16S4F32OF32ReturnsNotSupported)
{
    BatchArgs<bfloat16, int8_t, float, float> args(1, 64, 32, 32 * 64 / 2,
                                                   kBf16One, 0x11);
    WoqParams                                 woq(64, DLP_S4);
    args.set_quant_op(&woq.qop);
    args.run(aocl_batch_gemm_bf16s4f32of32);
    args.expect_not_supported();
}

TEST_F(CpuDetection, BatchBf16S4F32OBF16ReturnsNotSupported)
{
    BatchArgs<bfloat16, int8_t, bfloat16, float> args(1, 64, 32, 32 * 64 / 2,
                                                      kBf16One, 0x11);
    WoqParams                                    woq(64, DLP_S4);
    args.set_quant_op(&woq.qop);
    args.run(aocl_batch_gemm_bf16s4f32obf16);
    args.expect_not_supported();
}

TEST_F(CpuDetection, BatchBf16U4F32OF32ReturnsNotSupported)
{
    BatchArgs<bfloat16, uint8_t, float, float> args(1, 64, 32, 32 * 64 / 2,
                                                    kBf16One, 0x11);
    WoqParams                                  woq(64, DLP_U4);
    args.set_quant_op(&woq.qop);
    args.run(aocl_batch_gemm_bf16u4f32of32);
    args.expect_not_supported();
}

TEST_F(CpuDetection, BatchBf16U4F32OBF16ReturnsNotSupported)
{
    BatchArgs<bfloat16, uint8_t, bfloat16, float> args(1, 64, 32, 32 * 64 / 2,
                                                       kBf16One, 0x11);
    WoqParams                                     woq(64, DLP_U4);
    args.set_quant_op(&woq.qop);
    args.run(aocl_batch_gemm_bf16u4f32obf16);
    args.expect_not_supported();
}
