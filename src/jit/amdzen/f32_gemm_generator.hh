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

// To indicate the generation of kernel with IR loop outside and JR loop inside
// within the microkernel for k=1 case. This loop order performs better since it
// aligns with the memory access patterns of the data.
#define IR_JR_LOOP_ORDER_FOR_K1

// The implementation with JR loop outside and IR loop inside is not used
// currently. It is kept for future reference. #define JR_IR_LOOP_ORDER_FOR_K1

namespace amdzen::GEMMcodeGenerator {

template<utils::kernelInstrType KType>
class jitGEMMF32 : public Xbyak::CodeGenerator
{
  public:
    // Constructor that specifies the maximum size of generated JIT code.
    // Buffer allocation and AutoGrow behavior are managed internally by Xbyak.
    jitGEMMF32(size_t maxSize);
    ~jitGEMMF32()                       = default;
    jitGEMMF32(jitGEMMF32&)             = delete;
    jitGEMMF32& operator=(jitGEMMF32&)  = delete;
    jitGEMMF32(jitGEMMF32&&)            = delete;
    jitGEMMF32& operator=(jitGEMMF32&&) = delete;

    dlp::jit::jitGeneratorError generateKernel(utils::generatorParams& params);

  private:
    using Traits      = amdzen::traits::ArchitectureTraits<KType>;
    using RegType     = typename Traits::RegType;
    using halfRegType = typename Traits::halfRegType;

    // Configuration and state
    int numRegs        = Traits::numRegs;
    int RegSize        = Traits::regSize;
    int RegBytes       = Traits::regBytes;
    int halfRegBytes   = RegBytes / 2;
    int numElemsPerReg = RegBytes / sizeof(float);
    // maskVecReg is the number of YMM/ZMM registers used for mask storage.
    int aReg, bReg, bFullReg, bMaskReg, cReg, maskVecReg;
    int aRegIdx, bRegIdx, cRegIdx, maskRegIdx;
    int MR, NR;
    int c_downscale;

    // Register allocations
    Xbyak::Reg64 regTmpAptr, regBptr, regTmpCptr, regRsA, regCsA, regRsB,
        regRsC, regKIter;
    Xbyak::Reg64 regMiter;
    Xbyak::Reg64 regNiter, regNLeft; // For N-loop in k=1 kernels
    Xbyak::Reg64 regTmp1, regTmp2, regTmp3;
    Xbyak::Reg64 regCPtr, regAPtr;
    Xbyak::Reg64 stackPtr;

    Xbyak::Reg64 regkernelOpsList, regkernelOpsAttr;

    Xbyak::Label label_store_result;

    Xbyak::Label offsets;
    bool         dumpedOffsets = false;

    Xbyak::Opmask fringeMask[dlp::kernels::maxNumMasks];

    // 4 masks to handle transpose C operations
    Xbyak::Opmask mask0, mask1;

    bool useMask =
        false; // Flag to indicate if masked instructions are generated
    int numMaskRegs = 0;

    // Core kernel generation methods - simplified for F32 only
    dlp::jit::jitGeneratorError checkValidF32GemmParams(
        const utils::generatorParams& params);
    dlp::jit::jitGeneratorError allocateReg();

    void loadMasks();

    void initializeParameters(bool addIrLoop);

    dlp::jit::jitGeneratorError generateIrLoop(utils::generatorParams& params);

    // Memory operations
    dlp::jit::jitGeneratorError loadBValues();

    dlp::jit::jitGeneratorError BroadcastAFMAwithB();

    dlp::jit::jitGeneratorError kernelUnroll(int unroll);

    dlp::jit::jitGeneratorError storeResult(bool fuseBetaWithStore, bool mLoop);

    // this function uses scatter approach to store the result matrix
    // Not used currently.
    dlp::jit::jitGeneratorError storeResultColumnMajor(bool fuseBetaWithStore);
    dlp::jit::jitGeneratorError storeResultRowMajor(bool fuseBetaWithStore);

    // Setup and initialization
    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);
    void regInit();
    void moveCPtr(Xbyak::Reg64 regPtr, Xbyak::Reg64 regStride, int val);

    // Scaling operations
    dlp::jit::jitGeneratorError scaleAlpha();

    dlp::jit::jitGeneratorError scaleBetaColumnMajor();
    dlp::jit::jitGeneratorError scaleBetaRowMajor();

    void                        resetMasks(bool mask, int idx);
    dlp::jit::jitGeneratorError allocateMaskRegisters();

    dlp::jit::jitGeneratorError scaleBeta();

    dlp::jit::jitGeneratorError convertF32toBF16(int scratch1,
                                                 int scratch2,
                                                 int destIdx);

    // K=1 kernel generation methods
    // 2D label vectors: [numNRVariants][MR] - outer loop is NR, inner loop is
    // MR
    std::vector<std::vector<Xbyak::Label>> label_store_result_k1;

    int numNRVariants, numMRVariants;

    int currentMR;
    int currentNR;
    int currentNRIdx; // Index of current NR variant being generated

    dlp::jit::jitGeneratorError generateKernelK1(
        utils::generatorParams& params);
    dlp::jit::jitGeneratorError generateKernel_JR_IR(
        utils::generatorParams& params);
    dlp::jit::jitGeneratorError generateKernel_IR_JR(
        utils::generatorParams& params);
    dlp::jit::jitGeneratorError generateKernelBodyK1(
        utils::generatorParams& params,
        gen::kernelOpsHandler*  kernelOpsHandlerPtr);
};

// Type aliases for specific instruction sets
using jitGemmGenerator_f32_avx512 =
    jitGEMMF32<utils::kernelInstrType::avx512_zmm_32_reg>;
using jitGemmGenerator_f32_avx512_256 =
    jitGEMMF32<utils::kernelInstrType::avx512_ymm_32_reg>;
using jitGemmGenerator_f32_avx2 =
    jitGEMMF32<utils::kernelInstrType::avx2_ymm_16_reg>;
} // namespace amdzen::GEMMcodeGenerator
