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
class jitGEMMBF16 : public Xbyak::CodeGenerator
{
  public:
    // Constructor that takes buffer and its size for JIT code dumping
    jitGEMMBF16(size_t maxSize);
    ~jitGEMMBF16()                        = default;
    jitGEMMBF16(jitGEMMBF16&)             = delete;
    jitGEMMBF16& operator=(jitGEMMBF16&)  = delete;
    jitGEMMBF16(jitGEMMBF16&&)            = delete;
    jitGEMMBF16& operator=(jitGEMMBF16&&) = delete;

    dlp::jit::jitGeneratorError generateKernel(utils::generatorParams& params);

  private:
    using Traits      = amdzen::traits::ArchitectureTraits<KType>;
    using RegType     = typename Traits::RegType;
    using halfRegType = typename Traits::halfRegType;

    // Configuration and state
    int numRegs          = Traits::numRegs;
    int RegSize          = Traits::regSize;
    int RegBytes         = Traits::regBytes;
    int halfRegBytes     = RegBytes / 2;
    int nF32ElemsPerReg  = RegBytes / sizeof(float);
    int nBF16ElemsPerReg = RegBytes / sizeof(int16_t);
    int aReg, bReg, bFullReg, bMaskReg, cReg;
    int aRegIdx, bRegIdx, cRegIdx;
    int MR, NR, K_UNROLL, PREFETCH_C_DIST;
    int c_downscale;

    // Register allocations
    Xbyak::Reg64 regTmpAptr, regBptr, regTmpCptr, regRsA, regCsA, regRsB,
        regRsC, regKIter;
    Xbyak::Reg64 regMiter;
    Xbyak::Reg64 regTmp1, regTmp2;
    Xbyak::Reg64 regCPtr, regAPtr;
    Xbyak::Reg64 stackPtr;

    // Add mask register array (for AVX512)
    static constexpr int NUM_USABLE_MASKS = 7;        // k1-k7 available
    static constexpr int MASK_START_IDX   = 1;        // Start from k1
    Xbyak::Opmask        mask_regs[NUM_USABLE_MASKS]; // Array of usable masks

    bool useMask =
        false; // Flag to indicate if masked instructions are generated

    Xbyak::Label label_store_result;

    // Core kernel generation methods - BF16 specific
    dlp::jit::jitGeneratorError allocateReg();

    void initializeParameters(bool addIrLoop);

    dlp::jit::jitGeneratorError generateIrLoop(utils::generatorParams& params);

    // BF16 Memory operations
    dlp::jit::jitGeneratorError loadBValuesBF16();

    // BF16 computation: A (BF16) broadcast and multiply-add with B
    // (BF16) -> C (F32)
    dlp::jit::jitGeneratorError BroadcastABF16withB(bool isRemainder);

    dlp::jit::jitGeneratorError kLoopCompute(bool isRemainder, int kUnroll);

    // Setup and initialization
    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);

    void regInit();

    void moveCPtr();

    // Scaling operations - adapted for BF16->F32 accumulation
    dlp::jit::jitGeneratorError scaleAlpha();

    dlp::jit::jitGeneratorError scaleBeta();

    dlp::jit::jitGeneratorError storeResult();

    dlp::jit::jitGeneratorError loadMask();

    // dlp::jit::jitGeneratorError prefetchB();

    dlp::jit::jitGeneratorError prefetchC();
};

} // namespace amdzen::GEMMcodeGenerator
