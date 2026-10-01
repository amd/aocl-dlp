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
#include <cstring>

#include <gtest/gtest.h>

#include "arch_utils/arch_config_manager.hh"
#include "cpu_utils/cpu_features.hh"
#include "jit/amdzen/s8_gemm_quant_generator.hh"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

using amdzen::GEMMcodeGenerator::jitGEMMQuant;
using amdzen::utils::kernelInstrType;

namespace {

// Same nibble rule as dlp_s4_nibble_to_s8: even index is the low nibble,
// odd index is the high nibble, sign-extended from [-8, 7].
int8_t
refNibble(uint8_t byte, int hi)
{
    const uint8_t nib = hi ? static_cast<uint8_t>((byte >> 4) & 0x0F)
                           : static_cast<uint8_t>(byte & 0x0F);
    return static_cast<int8_t>(static_cast<int>(nib ^ 0x08) - 8);
}

// The three-instruction widen emitted by jitGEMMQuant::widenBLoad for gfni,
// using that class's constants so the matrix cannot drift from the kernel.
struct GfniWiden32 : Xbyak::CodeGenerator
{
    GfniWiden32()
        : Xbyak::CodeGenerator(4096, Xbyak::AutoGrow)
    {
        using Gen = jitGEMMQuant<kernelInstrType::avx512_zmm_32_reg>;
        Xbyak::Label pool;
        Xbyak::Label poolEnd;
        {
            Xbyak::util::StackFrame sf(this, 2, 0, 0);
            vpbroadcastq(Xbyak::Zmm(0),
                         Xbyak::util::qword[Xbyak::util::rip + pool]);
            vpmovzxdq(Xbyak::Zmm(1), Xbyak::util::yword[sf.p[0]]);
            vpmultishiftqb(Xbyak::Zmm(1), Xbyak::Zmm(0), Xbyak::Zmm(1));
            vgf2p8affineqb(Xbyak::Zmm(1), Xbyak::Zmm(1),
                           Xbyak::util::ptr_b[Xbyak::util::rip + pool + 8], 0);
            vmovdqu32(Xbyak::util::ptr[sf.p[1]], Xbyak::Zmm(1));
        }
        jmp(poolEnd, Xbyak::CodeGenerator::T_NEAR);
        align(64);
        L(pool);
        dq(Gen::kNibbleMultishiftCtl);
        dq(Gen::kGfniSignExtendMatrix);
        L(poolEnd);
        ready();
    }
};

bool
hostHasGfniWiden()
{
    auto& cpu = dlp::cpu_utils::cpuFeaturesInstance();
    // isAvx512SupportedByArch and avx512vbmi and gfni checks
    return dlp::arch_utils::archConfigManager::getInstance()
               .isAvx512SupportedByArch()
           && cpu.hasFeature(dlp::cpu_utils::isaFeature::avx512vbmi)
           && cpu.hasFeature(dlp::cpu_utils::isaFeature::gfni);
}

} // namespace

// Every packed byte value in every source lane. This is the placement the
// reorder buffer uses (low nibble = even element) and it includes the junk
// bits vpmultishiftqb leaves in bits 4..7 of each placed byte.
TEST(GfniNibbleWiden, MatchesScalarSignExtend)
{
    if (!hostHasGfniWiden()) {
        GTEST_SKIP() << "Requires AVX-512, AVX512-VBMI, and GFNI";
    }

    GfniWiden32 gen;
    using Fn            = void (*)(const uint8_t*, int8_t*);
    const Fn widen      = gen.getCode<Fn>();
    int      mismatches = 0;

    for (int pos = 0; pos < 32; ++pos) {
        for (int value = 0; value < 256; ++value) {
            alignas(64) uint8_t src[32] = {};
            alignas(64) int8_t  dst[64];
            src[pos] = static_cast<uint8_t>(value);
            std::memset(dst, 0x5A, sizeof(dst));
            widen(src, dst);
            for (int lane = 0; lane < 64; ++lane) {
                const int8_t exp = refNibble(src[lane >> 1], lane & 1);
                if (dst[lane] != exp) {
                    ++mismatches;
                }
            }
        }
    }

    EXPECT_EQ(mismatches, 0);
}
