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

#include "jit/jit_generator_base.hh"
#include "jit_generator_utils.hh"
#include "kernels/kernel_base.hh"
#include "traits.hh"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace amdzen::PackBcodeGenerator {

template<utils::kernelInstrType KType>
class jitPackBF32 : public Xbyak::CodeGenerator
{
  public:
    jitPackBF32();
    ~jitPackBF32()                        = default;
    jitPackBF32(jitPackBF32&)             = delete;
    jitPackBF32& operator=(jitPackBF32&)  = delete;
    jitPackBF32(jitPackBF32&&)            = delete;
    jitPackBF32& operator=(jitPackBF32&&) = delete;

    dlp::jit::jitGeneratorError generateKernel(
        utils::packBGeneratorParams& params);

  private:
    using Traits  = amdzen::traits::ArchitectureTraits<KType>;
    using RegType = typename Traits::RegType;

    static constexpr int numRegs        = Traits::numRegs;
    static constexpr int RegBytes       = Traits::regBytes;
    static constexpr int numElemsPerReg = RegBytes / sizeof(float);

    md_t NR_;
    int  numVecLoads;
    bool useMask_;

    Xbyak::Reg64 pParams;
    Xbyak::Reg64 regSrc;
    Xbyak::Reg64 regDst;
    Xbyak::Reg64 regK;
    Xbyak::Reg64 regLdbBytes;
    Xbyak::Reg64 regDstPanelStride;
    Xbyak::Reg64 regSrcEnd;
    Xbyak::Reg64 regNPartial;
    Xbyak::Reg64 regSrcBase;
    Xbyak::Reg64 regDstBase;
    Xbyak::Reg64 regSrcRow;
    Xbyak::Reg64 regDstRow;
    Xbyak::Reg64 regKr;

    dlp::jit::jitGeneratorError allocateReg();
    void initializeStackFrame(Xbyak::util::StackFrame& sf);
    void initializeParameters();
    void generateFullBlockLoop();
    void generateLtBlockLoop();
    void loadAndStoreRow();
    void loadPerBlockMasks();
    void maskedLoadAndStoreRow();
};

} // namespace amdzen::PackBcodeGenerator
