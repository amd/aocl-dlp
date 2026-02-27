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
#include "jit/xbyak/xbyak.h"
#include "jit_generator_utils.hh"
#include "kernel_ops_handler.hh"
#include "kernels/kernel_base.hh"
#include "traits.hh"

namespace amdzen::GEMMcodeGenerator {

template<utils::kernelInstrType KType>
class jitGEMMS8 : public Xbyak::CodeGenerator
{
  public:
    // Constructor that specifies the maximum size of generated JIT code.
    // Buffer allocation and AutoGrow behavior are managed internally by Xbyak.
    jitGEMMS8(size_t maxSize);
    ~jitGEMMS8()                      = default;
    jitGEMMS8(jitGEMMS8&)             = delete;
    jitGEMMS8& operator=(jitGEMMS8&)  = delete;
    jitGEMMS8(jitGEMMS8&&)            = delete;
    jitGEMMS8& operator=(jitGEMMS8&&) = delete;

    dlp::jit::jitGeneratorError generateKernel(utils::generatorParams& params);

  private:
    using Traits  = amdzen::traits::ArchitectureTraits<KType>;
    using RegType = typename Traits::RegType;

    // Configuration and state
    int numRegs  = Traits::numRegs;
    int RegSize  = Traits::regSize;
    int RegBytes = Traits::regBytes;
    int aReg, bReg, bFullReg, bMaskReg, cReg;
    int vec128Reg = 1; // Reserving one register for converting int8 to uint8.
    int aRegIdx, bRegIdx, cRegIdx, maskRegIdx, vec128RegIdx;
    int MR, NR;
    int c_downscale;

    // Register allocations
    Xbyak::Reg64 regTmpAptr, regBptr, regTmpCptr, regRsA, regCsA, regRsB,
        regRsC, regKIter;
    Xbyak::Reg64 regMiter;
    Xbyak::Reg64 regTmp1, regTmp2, regTmp3;
    Xbyak::Reg64 regCPtr, regAPtr;
    Xbyak::Reg64 stackPtr;

    Xbyak::Reg64 regkernelOpsList, regkernelOpsAttr;

    Xbyak::Label label_store_result;

    bool useMask = false; // Flag to indicate generation of masked instructions.

    // Setup and initialization
    dlp::jit::jitGeneratorError allocateReg();

    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);

    void initializeParameters(bool addIrLoop);

    void initializeRegisters();

    // Core operations
    dlp::jit::jitGeneratorError loadBValues();

    dlp::jit::jitGeneratorError BroadcastAVNNIB(bool isVNNIrem);

    dlp::jit::jitGeneratorError kLoop(int unroll, bool isVNNIrem);

    // Operations to compensate for A conversion to uint8 for VNNI
    dlp::jit::jitGeneratorError loadBSumValues();

    dlp::jit::jitGeneratorError conversionCompensation();

    // Scaling operations
    dlp::jit::jitGeneratorError scaleAlpha();

    dlp::jit::jitGeneratorError scaleBeta();

    // Memory operations
    dlp::jit::jitGeneratorError updateCBufferPointers();

    dlp::jit::jitGeneratorError storeResult(bool hasPostOps = false);

    void moveCPtr();

    // Kernel generation
    dlp::jit::jitGeneratorError generateIrLoop(utils::generatorParams& params);
};

// Type aliases for specific instruction sets
using jitGemmGenerator_s8_avx512 =
    jitGEMMS8<utils::kernelInstrType::avx512_zmm_32_reg>;
} // namespace amdzen::GEMMcodeGenerator
