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
#include "kernel_ops_handler.hh"
#include "kernels/kernel_base.hh"
#include "traits.hh"
#include "xbyak/xbyak.h"

namespace amdzen::gen {

/**
 * @brief JIT generator for F32×FP16→F32 mixed-precision GEMM kernels
 *
 * B matrix is loaded as FP16 and converted to F32 via vcvtph2ps.
 * A matrix and C matrix are F32. FMA uses vfmadd231ps (F32).
 * Supports MR=6, NR=64 blocking with 16 F32 elements per ZMM.
 *
 * Register allocation for MR=6, NR=64:
 *   bFullReg = 64 / 16 = 4 (F32 ZMM registers for B)
 *   cReg = 6 × 4 = 24 (accumulator registers)
 *   aReg = 32 - 24 - 4 = 4 (scratch registers)
 */
template<utils::kernelInstrType KType>
class jitF32FP16_GEMM : public Xbyak::CodeGenerator
{
  public:
    jitF32FP16_GEMM(void* buffer, size_t bufferSize);
    ~jitF32FP16_GEMM()                            = default;
    jitF32FP16_GEMM(jitF32FP16_GEMM&)             = delete;
    jitF32FP16_GEMM& operator=(jitF32FP16_GEMM&)  = delete;
    jitF32FP16_GEMM(jitF32FP16_GEMM&&)            = delete;
    jitF32FP16_GEMM& operator=(jitF32FP16_GEMM&&) = delete;

    dlp::jit::jitGeneratorError generateKernel(utils::generatorParams& params);

  private:
    using Traits  = amdzen::traits::ArchitectureTraits<KType>;
    using RegType = typename Traits::RegType;

    // =================================================================
    // MIXED-PRECISION CONSTANTS
    // =================================================================
    static constexpr int FP16_ELEM_SIZE = 2;  // B is FP16 (2 bytes)
    static constexpr int F32_ELEM_SIZE  = 4;  // A and C are F32 (4 bytes)
    static constexpr int FP16_PER_YMM   = 16; // 32 bytes / 2 = 16 FP16 per YMM
    static constexpr int F32_PER_ZMM    = 16; // 64 bytes / 4 = 16 F32 per ZMM

    // =================================================================
    // KERNEL CONFIGURATION
    // =================================================================
    int  numRegs  = Traits::numRegs;
    int  RegSize  = Traits::regSize;
    int  RegBytes = Traits::regBytes;
    int  MR, NR;
    int  c_downscale;
    bool useMask     = false;
    int  numMaskRegs = 0;

    // =================================================================
    // REGISTER ALLOCATION
    // =================================================================
    int aReg, bReg, bFullReg, bMaskReg, cReg;
    int aRegIdx, bRegIdx, cRegIdx;

    // =================================================================
    // GENERAL PURPOSE REGISTERS
    // =================================================================
    Xbyak::Reg64 stackPtr;
    Xbyak::Reg64 regTmpAptr, regBptr, regTmpCptr;
    Xbyak::Reg64 regRsA, regCsA, regRsB, regRsC;
    Xbyak::Reg64 regKIter, regMiter;
    Xbyak::Reg64 regCPtr, regAPtr;
    Xbyak::Reg64 regTmp1, regTmp2, regTmp3;

    // =================================================================
    // CORE METHODS
    // =================================================================
    dlp::jit::jitGeneratorError allocateRegisters();
    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);
    void initializeParameters(bool mLoop = false);
    void initializeAccumulators(utils::generatorParams& params);

    dlp::jit::jitGeneratorError generateMLoop(utils::generatorParams& params);
    dlp::jit::jitGeneratorError generateIrLoop(utils::generatorParams& params);

    /**
     * @brief Load B as FP16 (YMM), convert to F32 (ZMM) via vcvtph2ps
     */
    dlp::jit::jitGeneratorError loadBValues();

    /**
     * @brief Broadcast A element (F32) and FMA with B (F32) using vfmadd231ps
     */
    dlp::jit::jitGeneratorError broadcastAFMAwithB(bool isKRemainder);

    /**
     * @brief K-loop unroll. B pointer advances by NR × FP16_ELEM_SIZE bytes.
     */
    dlp::jit::jitGeneratorError kUnroll(int unroll, bool isKRemainder);

    dlp::jit::jitGeneratorError scaleAlpha();
    dlp::jit::jitGeneratorError scaleBeta();
    dlp::jit::jitGeneratorError generatePostOps(utils::generatorParams& params);
    dlp::jit::jitGeneratorError storeResult();

    /**
     * @brief Store results as F32 using vmovups (not FP16)
     */
    dlp::jit::jitGeneratorError storeResultF32();

    dlp::jit::jitGeneratorError moveCPtr();
};

using jitGemmGenerator_f32f16_avx512 =
    jitF32FP16_GEMM<utils::kernelInstrType::avx512_zmm_32_reg>;

} // namespace amdzen::gen
