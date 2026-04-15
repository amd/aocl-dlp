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
#include "kernel_ops_handler.hh"
#include "kernels/kernel_base.hh"
#include "traits.hh"
#include "xbyak/xbyak.h"
#include <iostream>

namespace amdzen::GEMMcodeGenerator {

template<utils::kernelInstrType KType>
class jitGEMMF32RD : public Xbyak::CodeGenerator
{
  public:
    // Constructor that specifies the maximum size of generated JIT code.
    // Buffer allocation and AutoGrow behavior are managed internally by Xbyak.
    jitGEMMF32RD(size_t maxSize);
    ~jitGEMMF32RD()                         = default;
    jitGEMMF32RD(jitGEMMF32RD&)             = delete;
    jitGEMMF32RD& operator=(jitGEMMF32RD&)  = delete;
    jitGEMMF32RD(jitGEMMF32RD&&)            = delete;
    jitGEMMF32RD& operator=(jitGEMMF32RD&&) = delete;

    dlp::jit::jitGeneratorError generateKernel(utils::generatorParams& params);

  private:
    using Traits      = amdzen::traits::ArchitectureTraits<KType>;
    using RegType     = typename Traits::RegType;
    using halfRegType = typename Traits::halfRegType;

    // Configuration and state
    const int numRegs      = Traits::numRegs;
    const int RegSize      = Traits::regSize;
    const int RegBytes     = Traits::regBytes;
    const int halfRegBytes = RegBytes / 2;
    const int nElemsPerReg = RegBytes / sizeof(float);
    const int nElemsPerXmm = 16 / sizeof(float);

    int nSubBlockSize;
    int MR, NR, useMask, numMaskRegs;
    int c_downscale;

    int aReg, bReg, cReg, accumReg;
    int aRegIdx, bRegIdx, cRegIdx, accumRegIdx;
    int nMaskRegIdx;

    // To store mask along the n-dimension.
    Xbyak::Opmask fringeMask[dlp::kernels::maxNumMasks - 1];
    // To store mask along the k-dimension.
    Xbyak::Opmask kLeftMask;

    Xbyak::Label label_store_result;

    Xbyak::Reg64 stackPtr, regAptr, regBptr, regCptr;
    Xbyak::Reg64 regRsA, regCsB, regRsC;
    // regTmp2 unused; regKIter aliases regTmpCptr, regRsC aliases regCsB3
    Xbyak::Reg64 regKIter, regMiter, regJJCounter, regTmp1, regTmp2;
    Xbyak::Reg64 regTmpAptr, regTmpBptr, regTmpCptr;
    Xbyak::Reg64 regCsB3, regRsA3, regRsA5;

    // Core kernel generation methods - simplified for F32 only
    dlp::jit::jitGeneratorError allocateReg();

    void initializeParameters(bool addIrLoop);

    void createMaskFromConstant(int value);

    dlp::jit::jitGeneratorError generateKrLoop(int  unrollFactor,
                                               bool isKFringe);

    dlp::jit::jitGeneratorError generateIrLoop(utils::generatorParams& params);

    dlp::jit::jitGeneratorError generateJrLoop(utils::generatorParams& params);
    // Memory operations
    dlp::jit::jitGeneratorError loadBValues();

    dlp::jit::jitGeneratorError kernelUnroll(int unroll);

    dlp::jit::jitGeneratorError storeResult(bool fuseBetaWithStore);

    // Setup and initialization
    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);
    void regInit();

    // Scaling operations
    dlp::jit::jitGeneratorError scaleAlpha();

    dlp::jit::jitGeneratorError scaleBetaColumnMajor();
    dlp::jit::jitGeneratorError scaleBetaRowMajor();

    dlp::jit::jitGeneratorError allocateMaskRegisters();

    dlp::jit::jitGeneratorError scaleBeta();

    dlp::jit::jitGeneratorError convertF32toBF16(int scratch1,
                                                 int scratch2,
                                                 int destIdx);

    // Helper methods
    dlp::jit::jitGeneratorError alphaScale();
    dlp::jit::jitGeneratorError betaScale();
    dlp::jit::jitGeneratorError storeResults(bool fuseBetaWithStore);
    dlp::jit::jitGeneratorError reduceAccumulation();
    dlp::jit::jitGeneratorError reduceToXmm(int startIdx,
                                            int tmpIdx,
                                            int blockSize);

    void calculateRowOffset(int           row,
                            Xbyak::Reg64& regTmp,
                            Xbyak::Reg64& regRsA);
    void loadRegF32Values(int                   regIdx,
                          const Xbyak::Address& address,
                          bool                  isFringe);
    void loadRegF32Xmm(int regIdx, const Xbyak::Address& address);
    void storeF32Xmm(int regIdx, const Xbyak::Address& address);
    void loadMRrowsOfA(bool isFringe);
};

// Type aliases for specific instruction sets
using jitGemmGenerator_f32_rd_avx512 =
    jitGEMMF32RD<utils::kernelInstrType::avx512_zmm_32_reg>;
using jitGemmGenerator_f32_rd_avx512_256 =
    jitGEMMF32RD<utils::kernelInstrType::avx512_ymm_32_reg>;
using jitGemmGenerator_f32_rd_avx2 =
    jitGEMMF32RD<utils::kernelInstrType::avx2_ymm_16_reg>;
} // namespace amdzen::GEMMcodeGenerator
