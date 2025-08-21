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

namespace amdzen::avx512gen {

class jitAVX512 : public Xbyak::CodeGenerator
{
  public:
    // Constructor that takes buffer and its size for JIT code dumping
    jitAVX512(void* buffer, size_t bufferSize);
    ~jitAVX512()                      = default;
    jitAVX512(jitAVX512&)             = delete;
    jitAVX512& operator=(jitAVX512&)  = delete;
    jitAVX512(jitAVX512&&)            = delete;
    jitAVX512& operator=(jitAVX512&&) = delete;

    // Template function that takes the datatype as a template parameter
    template<dlp::kernel_frame::kernelDatatype KDT>
    dlp::jit::jitGeneratorError generateKernel(utils::generatorParams& params);

  private:
    using Traits = amdzen::traits::ArchitectureTraits<
        utils::kernelInstrType::avx512_zmm_32_reg>;

    // Configuration and state
    int numRegs  = Traits::numRegs;
    int RegSize  = Traits::regSize;
    int RegBytes = Traits::regBytes;
    int aReg, bReg, bFullReg, bMaskReg, cReg;
    int aRegIdx, bRegIdx, cRegIdx;
    int MR, NR;

    // Register allocations
    Xbyak::Reg64 regTmpAptr, regBptr, regTmpCptr, regRsA, regCsA, regRsB,
        regRsC, regKIter;
    Xbyak::Reg64 regMiter;
    Xbyak::Reg64 regTmp1, regTmp2, regTmp3;
    Xbyak::Reg64 regCPtr, regAPtr;
    Xbyak::Reg64 stackPtr;

    Xbyak::Reg64 regkernelOpsList, regkernelOpsAttr;

    Xbyak::Label label_store_result;

    bool useMask =
        false; // Flag to indicate if masked instructions are generated

    // Core kernel generation methods
    template<typename accumType>
    dlp::jit::jitGeneratorError allocateReg();

    template<typename aType, typename bType, typename cType>
    void initializeParameters(bool addIrLoop);

    template<typename aType, typename bType, typename cType, typename accumType>
    dlp::jit::jitGeneratorError generateIrLoop(utils::generatorParams& params);

    // Memory operations
    template<typename bType>
    dlp::jit::jitGeneratorError loadBValuesZmm();

    template<typename aType, typename bType, typename accumType>
    dlp::jit::jitGeneratorError BroadcastAFMAwithBZmm();

    template<typename aType, typename bType, typename accumType>
    dlp::jit::jitGeneratorError kernelUnrollZmm(int unroll);

    template<typename accumType, typename cType>
    dlp::jit::jitGeneratorError storeResult();

    // Setup and initialization
    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);
    void regInitZmm();
    void moveCPtr();

    // Scaling operations
    template<typename accumType>
    dlp::jit::jitGeneratorError scaleAlpha();

    template<typename accumType, typename cType>
    dlp::jit::jitGeneratorError scaleBeta();

    template<typename accumType>
    dlp::jit::jitGeneratorError cvtAccToFloat();
};

} // namespace amdzen::avx512gen
