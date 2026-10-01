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
#include <sstream>
#include <string>
#include <vector>

#include "jit/amdzen/s8_gemm_quant_generator.hh"
#include "jit_generator_tests_utils.hh"

using namespace amdzen::utils;
using namespace dlp::kernel_frame;
using amdzen::GEMMcodeGenerator::jitGEMMQuant;

namespace {

// The variant grid the orchestrator actually generates, from the GEMM branch of
// gemm_quant_amdzen_generator.cc: MR walks 1..6 (mr==0 meaning the full MR=6
// kernel with mLoop), and NR walks nr*16 for nr in 0..4, where NR==0 is the
// masked fringe kernel. Keeping the grid identical to production is the point
// -- these are exactly the shapes that must survive any register-budget change.
constexpr int kMaxMR       = 6;
constexpr int kNRVariants  = 5; // NR in {0, 16, 32, 48, 64}
constexpr int kElemsPerReg = 16;

struct GeneratedCode
{
    dlp::jit::jitGeneratorError err  = dlp::jit::jitGeneratorError::error;
    std::size_t                 size = 0;
    std::vector<uint8_t>        bytes;
};

// Mirrors the s8s4 symmetric group-quant metadata that
// gemmQuantDEBackendUtils fills in: both operands retain s8 compute metadata,
// while bQuant.mode distinguishes compact nibble loads from pre-widened B.
// In-kernel widening is the GFNI affine only. Neither operand carries a zero
// point, which reserves the +128 vec128 register.
quantGeneratorParams
makeS8S4Params(int MR, int nrIdx, opQuantMode bMode)
{
    generatorParams base(/*MR*/ 0, /*NR*/ 0, /*K_UNROLL*/ 4,
                         /*PREFETCH_C_DIST*/ 8, /*c_downscale*/ DLP_F32,
                         /*numMaskRegs*/ 1, /*useMask*/ false, /*mLoop*/ false,
                         /*is_k1*/ false, scalingType::generic,
                         scalingType::generic,
                         kernelInstrType::avx512_zmm_32_reg);

    quantGeneratorParams params(base);

    params.base.MR          = MR;
    params.base.mLoop       = (MR == kMaxMR);
    params.base.NR          = nrIdx * kElemsPerReg;
    params.base.useMask     = (nrIdx == 0);
    params.base.numMaskRegs = params.base.useMask ? 1 : 0;

    params.aQuant.src_type        = DataType::s8;
    params.aQuant.dst_type        = DataType::s32;
    params.aQuant.mode            = opQuantMode::dequantInKernel;
    params.aQuant.scale.storeDt   = DataType::f32;
    params.aQuant.scale.outerDim  = ParamDim::PerM;
    params.aQuant.scale.perGroupK = true;

    params.bQuant.src_type        = DataType::s8;
    params.bQuant.dst_type        = DataType::s32;
    params.bQuant.mode            = bMode;
    params.bQuant.scale.storeDt   = DataType::f32;
    params.bQuant.scale.outerDim  = ParamDim::PerGroup;
    params.bQuant.scale.perGroupK = true;

    return params;
}

GeneratedCode
generate(quantGeneratorParams& params)
{
    GeneratedCode out;

    jitGEMMQuant<kernelInstrType::avx512_zmm_32_reg> gen(JIT_KERNEL_SIZE);
    out.err = gen.generateKernel(params);
    if (out.err != dlp::jit::jitGeneratorError::success) {
        return out;
    }

    // AutoGrow may have relocated the buffer; ready() applies the pending
    // jump/branch fixups so getSize() matches what the orchestrator would run.
    gen.ready();
    out.size         = gen.getSize();
    const auto* code = gen.getCode();
    out.bytes.assign(code, code + out.size);
    return out;
}

// EVEX form of vgf2p8affineqb: 0x62, opcode map 0F3A in P0[2:0], opcode 0xCE.
// The legacy 0F 3A bytes are not present; the map lives in the prefix.
bool
containsGfniAffine(const std::vector<uint8_t>& bytes)
{
    if (bytes.size() < 5) {
        return false;
    }
    for (std::size_t i = 0; i + 5 <= bytes.size(); ++i) {
        if (bytes[i] == 0x62 && ((bytes[i + 1] & 0x07) == 0x03)
            && bytes[i + 4] == 0xCE) {
            return true;
        }
    }
    return false;
}

std::string
shapeName(int MR, int nrIdx)
{
    std::ostringstream os;
    os << "MR" << MR << "_NR" << (nrIdx * kElemsPerReg);
    if (nrIdx == 0) {
        os << "_masked";
    }
    return os.str();
}

class JitQuantGemmVariantsTest : public ::testing::Test
{
  protected:
    // Walks the production variant grid and asserts every shape generates.
    // generateKernel only encodes bytes, so this does not require a VNNI host.
    void sweepGrid(const std::string& label, opQuantMode bMode)
    {
        for (int MR = 1; MR <= kMaxMR; ++MR) {
            for (int nrIdx = 0; nrIdx < kNRVariants; ++nrIdx) {
                auto params = makeS8S4Params(MR, nrIdx, bMode);
                auto code   = generate(params);

                const std::string name = shapeName(MR, nrIdx);
                ASSERT_EQ(code.err, dlp::jit::jitGeneratorError::success)
                    << name << " failed to generate under " << label;
                EXPECT_GT(code.size, 0u) << name << " emitted no code";
            }
        }
    }
};

// The path in production today. Every shape must generate.
TEST_F(JitQuantGemmVariantsTest, AllShapesGenerateDequantInKernel)
{
    sweepGrid("bQuant.mode = dequantInKernel", opQuantMode::dequantInKernel);
}

// Reserving the vpmultishiftqb control costs one slot out of the A pool.
// This sweep proves every production shape still resolves.
TEST_F(JitQuantGemmVariantsTest, AllShapesGenerateWidenDequantInKernel)
{
    sweepGrid("bQuant.mode = widenDequantInKernel",
              opQuantMode::widenDequantInKernel);
}

// Widen mode replaces each plain B load with the GFNI placement sequence and
// appends the two-qword constant pool. Every shape must actually emit the
// affine.
TEST_F(JitQuantGemmVariantsTest, WidenModeEmitsGfniAffine)
{
    for (int MR = 1; MR <= kMaxMR; ++MR) {
        for (int nrIdx = 0; nrIdx < kNRVariants; ++nrIdx) {
            auto baseParams =
                makeS8S4Params(MR, nrIdx, opQuantMode::dequantInKernel);
            auto widenParams =
                makeS8S4Params(MR, nrIdx, opQuantMode::widenDequantInKernel);

            auto baseCode  = generate(baseParams);
            auto widenCode = generate(widenParams);

            const std::string name = shapeName(MR, nrIdx);
            ASSERT_EQ(baseCode.err, dlp::jit::jitGeneratorError::success)
                << name;
            ASSERT_EQ(widenCode.err, dlp::jit::jitGeneratorError::success)
                << name;
            EXPECT_GT(widenCode.size, baseCode.size)
                << name << " widen kernel is not larger than the plain B load";
            EXPECT_TRUE(containsGfniAffine(widenCode.bytes))
                << name << " widen kernel did not emit vgf2p8affineqb";
        }
    }
}

} // namespace
