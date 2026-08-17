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
#include <vector>

#include "jit/jit_generator_base.hh"
#include "jit_generator_utils.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "kernel_ops_handler.hh"
#include "traits.hh"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace amdzen::gen {

// s8s8s32 symmetric group-quantized GEMV, N == 1.
template<utils::kernelInstrType KType>
class jitGEMVQuantN1 : public Xbyak::CodeGenerator
{
  public:
    // Constructor that specifies the maximum size of generated JIT code.
    // Buffer allocation and AutoGrow behavior are managed internally by Xbyak.
    jitGEMVQuantN1(size_t maxSize);
    ~jitGEMVQuantN1()                           = default;
    jitGEMVQuantN1(jitGEMVQuantN1&)             = delete;
    jitGEMVQuantN1& operator=(jitGEMVQuantN1&)  = delete;
    jitGEMVQuantN1(jitGEMVQuantN1&&)            = delete;
    jitGEMVQuantN1& operator=(jitGEMVQuantN1&&) = delete;

    dlp::jit::jitGeneratorError generateKernel(
        utils::quantGemvN1GeneratorParams& params);

  private:
    using Traits  = amdzen::traits::ArchitectureTraits<KType>;
    using RegType = typename Traits::RegType;

    // Configuration and state.
    int numRegs  = Traits::numRegs;
    int RegSize  = Traits::regSize;
    int RegBytes = Traits::regBytes;
    int vnniWidth;

    int MR;     // Number of rows reduced at once
    int M_LEFT; // M-dimension left over elements

    dlp::kernel_frame::storageFormat yFormat;
    dlp::kernel_frame::scalingType   alphaScalingType;
    dlp::kernel_frame::scalingType   betaScalingType;
    int                              c_downscale = DLP_F32;

    // Scale-factor storage type, fixed at generation time and part of the
    // kernel cache key. Only f32 and bf16 are supported.
    dlp::kernel_frame::DataType aScaleType =
        dlp::kernel_frame::DataType::invalid;
    dlp::kernel_frame::DataType bScaleType =
        dlp::kernel_frame::DataType::invalid;

    // Whether each scale array is tiled along K. False means that operand
    // collapses the K axis (A PER_TOKEN / B PER_CHANNEL), so the group index
    // contributes no offset to its scale address and the scale can be hoisted
    // out of the group loop entirely.
    bool aPerGroupK = true;
    bool bPerGroupK = true;

    // Whether the +128 conversion register is reserved and used. vpdpbusd
    // wants an unsigned A operand, so on the symmetric path a signed int8 A is
    // biased by +128 and conversionCompensationGroup subtracts the 128*sum(B)
    // that introduces. The two are a matched pair and are gated together. An
    // operand carrying a zero point needs neither.
    bool useVec128 = true;

    int aScaleElemBytes() const
    {
        return (aScaleType == dlp::kernel_frame::DataType::bf16) ? 2 : 4;
    }
    int bScaleElemBytes() const
    {
        return (bScaleType == dlp::kernel_frame::DataType::bf16) ? 2 : 4;
    }

    // Register banks. accum holds the int32 per-row partial sums for one group;
    // fAcc holds the F32 total across groups.
    int accumReg, accumBaseIdx;
    int tmpReg, tmpBaseIdx;
    int xReg, xBaseIdx;
    int vec128Reg, vec128Idx;
    int fAccReg, fAccBaseIdx;
    int aSclReg, aSclBaseIdx;
    int bSclReg, bSclBaseIdx;
    int yReg, yBaseIdx;

    // Register allocations.
    Xbyak::Reg64 stackPtr;
    Xbyak::Reg64 regAptr, regTmpAptr;
    Xbyak::Reg64 regXptr;
    Xbyak::Reg64 regYptr, regTmpYptr;
    Xbyak::Reg64 regRsA;
    Xbyak::Reg64 regRsC;
    Xbyak::Reg64 regMIter;
    Xbyak::Reg64 regKIter;
    Xbyak::Reg64 regTmp1, regTmp2, regTmp3;
    Xbyak::Reg64 regGIter;

    // Setup and initialization.
    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);
    dlp::jit::jitGeneratorError allocateRegisters();
    void initializeParameters(utils::quantGemvN1GeneratorParams& params);
    void regInit(int baseIdx, int numRegs);

    // VNNI inner loop, unchanged from the non-quant kernel.
    dlp::jit::jitGeneratorError loadXValues(bool isFringe = false);
    dlp::jit::jitGeneratorError loadAValues(int aRegIdx, bool isFringe = false);
    dlp::jit::jitGeneratorError computeVNNI(int aRegIdx, int accumRegIdx);
    dlp::jit::jitGeneratorError processMRBlock(int  mSize,
                                               bool isFringe = false);
    dlp::jit::jitGeneratorError reduceToXmm(int startIdx,
                                            int tmpIdx,
                                            int blockSize);
    dlp::jit::jitGeneratorError reduceAccumulation(int mSize);

    // Group-quant specific.
    dlp::jit::jitGeneratorError groupLoop(int mSize);
    dlp::jit::jitGeneratorError conversionCompensationGroup(int mSize);
    dlp::jit::jitGeneratorError loadAScaleVector(int mSize);
    dlp::jit::jitGeneratorError loadBScaleBroadcast();
    dlp::jit::jitGeneratorError dequantAccumToFAcc(int mSize);
    void                        absoluteGroupIndex(const Xbyak::Reg64& dst);

    // F32 epilogue.
    dlp::jit::jitGeneratorError scaleAccByAlphaF32(int mSize);
    dlp::jit::jitGeneratorError scaleYByBetaF32(int mSize);
    dlp::jit::jitGeneratorError generatePostOps(
        utils::gemvN1GeneratorParams& params, int mSize);
    dlp::jit::jitGeneratorError storeYF32(int mSize);
    void                        updateCBufferPointers();
};

// s8s8s32 symmetric group-quantized GEMV, M == 1.
template<utils::kernelInstrType KType>
class jitGEMVQuantM1 : public Xbyak::CodeGenerator
{
  public:
    jitGEMVQuantM1(size_t maxSize);
    ~jitGEMVQuantM1()                           = default;
    jitGEMVQuantM1(jitGEMVQuantM1&)             = delete;
    jitGEMVQuantM1& operator=(jitGEMVQuantM1&)  = delete;
    jitGEMVQuantM1(jitGEMVQuantM1&&)            = delete;
    jitGEMVQuantM1& operator=(jitGEMVQuantM1&&) = delete;

    dlp::jit::jitGeneratorError generateKernel(
        utils::quantGemvM1GeneratorParams& params);

  private:
    using Traits  = amdzen::traits::ArchitectureTraits<KType>;
    using RegType = typename Traits::RegType;

    int numRegs  = Traits::numRegs;
    int RegSize  = Traits::regSize;
    int RegBytes = Traits::regBytes;
    int vnniWidth;

    int NR;
    int N_LEFT;
    int N_LEFT_16;
    int N_LEFT_LT16;
    int KC;
    int K_SUB_ITER;
    int c_downscale = DLP_F32;

    dlp::kernel_frame::storageFormat yFormat;
    dlp::kernel_frame::scalingType   alphaScalingType;
    dlp::kernel_frame::scalingType   betaScalingType;

    dlp::kernel_frame::DataType aScaleType =
        dlp::kernel_frame::DataType::invalid;
    dlp::kernel_frame::DataType bScaleType =
        dlp::kernel_frame::DataType::invalid;
    bool aPerGroupK = true;
    bool bPerGroupK = true;

    // Whether the +128 conversion register is reserved and used. vpdpbusd
    // wants an unsigned A operand, so on the symmetric path a signed int8 A is
    // biased by +128 and conversionCompensationGroup subtracts the 128*sum(B)
    // that introduces. The two are a matched pair and are gated together. An
    // operand carrying a zero point needs neither.
    bool useVec128 = true;

    int aScaleElemBytes() const
    {
        return (aScaleType == dlp::kernel_frame::DataType::bf16) ? 2 : 4;
    }
    int bScaleElemBytes() const
    {
        return (bScaleType == dlp::kernel_frame::DataType::bf16) ? 2 : 4;
    }

    // xReg holds the K_SUB_ITER A broadcasts, bReg the B tile (reused for the
    // B column sums and then the B scales), accum the int32 tile and fAcc the
    // F32 total across groups and KC panels.
    int xReg, xBaseIdx;
    int bReg, bBaseIdx;
    int accumReg, accumBaseIdx;
    int fAccReg, fAccBaseIdx;
    int aSclReg, aSclBaseIdx;
    int scratchIdx; // one spare vector register for beta / bf16 constants
    int tmpIdx;     // second spare
    int vec128Reg, vec128BaseIdx;

    Xbyak::Reg64 stackPtr;
    Xbyak::Reg64 regBptr;
    Xbyak::Reg64 regXptr;
    Xbyak::Reg64 regYptr, regTmpYptr;
    Xbyak::Reg64 regNIter, regKIter, regKSubIter;
    Xbyak::Reg64 regRsB;
    Xbyak::Reg64 regGAbs;
    Xbyak::Reg64 regTmp1, regTmp2;
    Xbyak::Reg64 regIncN, regIncK;

    // Number of 16-lane output bands this variant touches, and the index of
    // the partially masked one (-1 when every touched band is full).
    int activeBands(bool nMask) const;
    int maskedBand(bool nMask) const;

    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);
    dlp::jit::jitGeneratorError allocateRegisters();
    void initializeParameters(utils::quantGemvM1GeneratorParams& params);
    void regInit(int baseIdx, int numRegs);

    // VNNI inner steps, carried over from the non-quant M=1 kernel.
    dlp::jit::jitGeneratorError computeKxNR(bool nMask);
    dlp::jit::jitGeneratorError computeKxNfringe();
    dlp::jit::jitGeneratorError compute1xNR(bool nMask, bool isLastKGroup);
    dlp::jit::jitGeneratorError compute1xNfringe(bool isLastKGroup);
    dlp::jit::jitGeneratorError accumulateKSubIters(bool nMask);

    // Group-quant specific.
    dlp::jit::jitGeneratorError groupKLoop(bool nMask);
    dlp::jit::jitGeneratorError groupLoop(bool nMask, bool isLastPanel);
    dlp::jit::jitGeneratorError conversionCompensationGroup(bool nMask);
    dlp::jit::jitGeneratorError loadBScaleVectors(bool nMask);
    dlp::jit::jitGeneratorError loadAScaleBroadcast();
    dlp::jit::jitGeneratorError dequantAccumToFAcc(bool nMask);
    void                        setupPanelBBase(bool nMask, bool isLastPanel);

    dlp::jit::jitGeneratorError scaleAccByAlphaF32(bool nMask);
    dlp::jit::jitGeneratorError scaleYByBetaF32(bool nMask);
    dlp::jit::jitGeneratorError generatePostOps(
        utils::gemvM1GeneratorParams& params, bool nMask);
    dlp::jit::jitGeneratorError storeYF32(bool nMask);
    void                        updateYBufferPointers();
    dlp::jit::jitGeneratorError emitNTile(
        utils::quantGemvM1GeneratorParams& params, bool nMask);
};

using jitGemvGenerator_quant_n1_avx512 =
    jitGEMVQuantN1<utils::kernelInstrType::avx512_zmm_32_reg>;
using jitGemvGenerator_quant_m1_avx512 =
    jitGEMVQuantM1<utils::kernelInstrType::avx512_zmm_32_reg>;

} // namespace amdzen::gen
