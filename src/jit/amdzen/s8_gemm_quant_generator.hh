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

#pragma once

#include <cstdint>
#include <memory>

#include "jit/jit_generator_base.hh"
#include "jit_generator_utils.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "kernel_ops_handler.hh"
#include "kernels/kernel_base.hh"
#include "traits.hh"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace amdzen::GEMMcodeGenerator {

template<utils::kernelInstrType KType>
class jitGEMMQuant : public Xbyak::CodeGenerator
{
  public:
    jitGEMMQuant(size_t maxSize);
    ~jitGEMMQuant()                         = default;
    jitGEMMQuant(jitGEMMQuant&)             = delete;
    jitGEMMQuant& operator=(jitGEMMQuant&)  = delete;
    jitGEMMQuant(jitGEMMQuant&&)            = delete;
    jitGEMMQuant& operator=(jitGEMMQuant&&) = delete;

    dlp::jit::jitGeneratorError generateKernel(
        utils::quantGeneratorParams& params);

  private:
    using Traits  = amdzen::traits::ArchitectureTraits<KType>;
    using RegType = typename Traits::RegType;

    // Configuration and state.
    int  numRegs  = Traits::numRegs;
    int  RegSize  = Traits::regSize;
    int  RegBytes = Traits::regBytes;
    int  aReg, bReg, bFullReg, bMaskReg, cReg;
    int  vec128Reg;
    int  aRegIdx, bRegIdx, cRegIdx, vec128RegIdx;
    bool fBankInRegs = false; // F32 bank in fReg vs [rsp] stack bank
    int  fRegIdx     = 0;     // base ZMM index of the register-resident bank
    int  MR, NR;
    int  c_downscale = DLP_F32;

    dlp::kernel_frame::DataType aScaleType =
        dlp::kernel_frame::DataType::invalid;
    dlp::kernel_frame::DataType bScaleType =
        dlp::kernel_frame::DataType::invalid;

    // Whether each scale array is tiled along K. Mirrors a_grp_mul / b_grp_mul
    // in dlp_gemm_grp_post_op_attr_t: A PER_TOKEN holds one scale per row and B
    // PER_CHANNEL one per column, both of which collapse the K axis. When false
    // the K-group index contributes no offset to that operand's scale address
    // and its running pointer must not advance between groups. Baked in here
    // rather than branched on at run time; the kernel register keys on
    // perGroupK, so each granularity gets its own cached kernel.
    bool aPerGroupK = true;
    bool bPerGroupK = true;

    // Scale-factor element width, which drives every scale-pointer stride.
    int aScaleElemBytes() const
    {
        return (aScaleType == dlp::kernel_frame::DataType::bf16) ? 2 : 4;
    }
    int bScaleElemBytes() const
    {
        return (bScaleType == dlp::kernel_frame::DataType::bf16) ? 2 : 4;
    }

    Xbyak::Opmask mask_regs[utils::NUM_USABLE_MASKS];

    // Register allocations.
    Xbyak::Reg64 regTmpAptr, regBptr, regTmpCptr, regRsA, regCsA, regRsB,
        regRsC, regKIter;
    Xbyak::Reg64 regMiter;
    Xbyak::Reg64 regTmp1, regTmp2, regTmp3;
    Xbyak::Reg64 regCPtr, regAPtr;
    Xbyak::Reg64 stackPtr;
    Xbyak::Reg64 regBsumPtr, regBsclPtr, regAsclPtr;

    bool useMask = false; // Generate masked instructions for the NR fringe.

    dlp::jit::jitGeneratorError allocateReg(
        utils::quantGeneratorParams& params);
    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);
    void initializeParameters(bool addIrLoop);
    void initializeRegisters();
    dlp::jit::jitGeneratorError loadBValues();
    dlp::jit::jitGeneratorError BroadcastAVNNIB(bool isVNNIrem);
    dlp::jit::jitGeneratorError kLoop(int unroll, bool isVNNIrem);
    dlp::jit::jitGeneratorError loadBSumValues();
    dlp::jit::jitGeneratorError conversionCompensation();
    dlp::jit::jitGeneratorError applyGroupScalesFastPath();
    dlp::jit::jitGeneratorError scaleAlphaF32();
    dlp::jit::jitGeneratorError scaleBeta();
    dlp::jit::jitGeneratorError generatePostOps(utils::generatorParams& params);
    dlp::jit::jitGeneratorError storeResult();
    void                        updateCBufferPointers();
    void                        zeroStackBank();
    void                        loadStackBankToRegs();
    void                        zeroRegBank();
    void                        foldRegBankToRegs();
    dlp::jit::jitGeneratorError conversionCompensationGroup();
    dlp::jit::jitGeneratorError dequantAccumToStackGroup();
    dlp::jit::jitGeneratorError loadBScales(const Xbyak::Reg64& base);
    dlp::jit::jitGeneratorError broadcastAScale(int                 ar,
                                                const Xbyak::Reg64& base);

    void moveCPtr();

    dlp::jit::jitGeneratorError generateIrLoop(
        utils::quantGeneratorParams& qParams);
};

// Type alias for the supported instruction set.
using jitGemmGenerator_quant_avx512 =
    jitGEMMQuant<utils::kernelInstrType::avx512_zmm_32_reg>;

} // namespace amdzen::GEMMcodeGenerator
