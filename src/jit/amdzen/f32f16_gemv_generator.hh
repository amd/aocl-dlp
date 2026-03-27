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
#include "kernel_frame/kernel_frame_base.hh"
#include "kernel_ops_handler.hh"
#include "traits.hh"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace amdzen::gen {

/**
 * @brief JIT generator for F32×FP16→F32 GEMV N=1 kernels
 *
 * y = A * x where A is F32 (M×K), x is FP16 (K×1), y is F32 (M×1).
 * x (FP16) is loaded as YMM and converted to F32 via vcvtph2ps.
 * A (F32) rows are loaded as ZMM vectors.
 * FMA uses vfmadd231ps for F32 accumulation.
 * After K-loop, horizontal reduction 16 F32 → 1 scalar per row.
 * Uses MR=16 blocking: process 16 rows at a time.
 */
template<utils::kernelInstrType KType>
class jitF32FP16GEMVN1 : public Xbyak::CodeGenerator
{
  public:
    jitF32FP16GEMVN1(void* buffer, size_t bufferSize);
    ~jitF32FP16GEMVN1()                             = default;
    jitF32FP16GEMVN1(jitF32FP16GEMVN1&)             = delete;
    jitF32FP16GEMVN1& operator=(jitF32FP16GEMVN1&)  = delete;
    jitF32FP16GEMVN1(jitF32FP16GEMVN1&&)            = delete;
    jitF32FP16GEMVN1& operator=(jitF32FP16GEMVN1&&) = delete;

    dlp::jit::jitGeneratorError generateKernel(
        utils::gemvN1GeneratorParams& params);

  private:
    using Traits  = amdzen::traits::ArchitectureTraits<KType>;
    using RegType = typename Traits::RegType;

    static constexpr int FP16_ELEM_SIZE = 2;
    static constexpr int F32_ELEM_SIZE  = 4;
    static constexpr int F32_PER_ZMM    = 16;
    static constexpr int FP16_PER_YMM   = 16;

    int numRegs  = Traits::numRegs;
    int RegBytes = Traits::regBytes;

    int nElemsPerReg;
    int MR;
    int M_LEFT;

    dlp::kernel_frame::storageFormat yFormat;
    dlp::kernel_frame::scalingType   alphaScalingType;
    dlp::kernel_frame::scalingType   betaScalingType;

    static constexpr int NUM_USABLE_MASKS = 7;
    static constexpr int MASK_START_IDX   = 1;
    Xbyak::Opmask        mask_regs[NUM_USABLE_MASKS];

    /*
     * Register allocation for MR=16, F32 accumulation:
     *   accumReg = 16 (zmm16-zmm31, one per row)
     *   xReg     = 1  (zmm15, for converted B/x vector)
     *   tmpReg   = 4  (zmm0-zmm3, for A loading + reduction scratch)
     *   yReg     = 1  (zmm4, for Y load/beta scaling)
     */
    int xReg;
    int accumReg;
    int tmpReg;
    int yReg;

    int xBaseIdx;
    int accumBaseIdx;
    int tmpBaseIdx;
    int yBaseIdx;

    Xbyak::Reg64 stackPtr;
    Xbyak::Reg64 regAptr, regTmpAptr;
    Xbyak::Reg64 regXptr;
    Xbyak::Reg64 regYptr, regTmpYptr;
    Xbyak::Reg64 regRsA;
    Xbyak::Reg64 regCsA;
    Xbyak::Reg64 regRsC;
    Xbyak::Reg64 regMIter;
    Xbyak::Reg64 regKIter;
    Xbyak::Reg64 regTmp1;
    Xbyak::Reg64 regTmp2;
    Xbyak::Reg64 regTmp3;

    int simdWidth;

    dlp::jit::jitGeneratorError allocateRegisters();
    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);
    void initializeParameters();
    void regInit(int baseIdx, int numRegs);

    dlp::jit::jitGeneratorError generateMLoop(
        utils::gemvN1GeneratorParams& params);
    dlp::jit::jitGeneratorError generateIrLoop(
        int mSize, utils::gemvN1GeneratorParams& params);
    dlp::jit::jitGeneratorError processMRBlock(int  mSize,
                                               bool isFringe = false);

    dlp::jit::jitGeneratorError reduceToXmm(int startIdx,
                                            int tmpIdx,
                                            int blockSize);
    dlp::jit::jitGeneratorError reduceAccumulation(int mSize);

    dlp::jit::jitGeneratorError scaleAccumulationWithAlpha(int mSize);
    dlp::jit::jitGeneratorError scaleYWithBeta(int mSize);

    dlp::jit::jitGeneratorError storeYValues(int mSize);

    dlp::jit::jitGeneratorError loadMasks();

    std::unique_ptr<gen::kernelOpsHandler>            kernelOpsHandlerPtr;
    std::vector<dlp::kernel_frame::kernelOpsMetaData> kernelOpsVector;
};

/**
 * @brief JIT generator for F32×FP16→F32 GEMV M=1 kernels
 *
 * y = x * B where x is F32 (1×K), B is FP16 (K×N), y is F32 (1×N).
 * B (FP16) rows are loaded as YMM and converted to F32 via vcvtph2ps.
 * x (F32) is broadcast from scalar to ZMM.
 * FMA uses vfmadd231ps for F32 accumulation.
 * Uses 1×NR blocking (NR=64 for F32, 4 ZMMs of 16 F32 each).
 * K_SUB_ITER=4 for software pipelining.
 */
template<utils::kernelInstrType KType>
class jitF32FP16GEMVM1 : public Xbyak::CodeGenerator
{
  public:
    jitF32FP16GEMVM1(void* buffer, size_t bufferSize);
    ~jitF32FP16GEMVM1()                             = default;
    jitF32FP16GEMVM1(jitF32FP16GEMVM1&)             = delete;
    jitF32FP16GEMVM1& operator=(jitF32FP16GEMVM1&)  = delete;
    jitF32FP16GEMVM1(jitF32FP16GEMVM1&&)            = delete;
    jitF32FP16GEMVM1& operator=(jitF32FP16GEMVM1&&) = delete;

    dlp::jit::jitGeneratorError generateKernel(
        utils::gemvM1GeneratorParams& params);

  private:
    using Traits  = amdzen::traits::ArchitectureTraits<KType>;
    using RegType = typename Traits::RegType;

    static constexpr int FP16_ELEM_SIZE = 2;
    static constexpr int F32_ELEM_SIZE  = 4;
    static constexpr int F32_PER_ZMM    = 16;
    static constexpr int FP16_PER_YMM   = 16;

    int numRegs  = Traits::numRegs;
    int RegBytes = Traits::regBytes;

    int nElemsPerReg;
    int NR;
    int N_LEFT;
    int KC;
    int K_SUB_ITER;

    dlp::kernel_frame::storageFormat yFormat;
    dlp::kernel_frame::scalingType   alphaScalingType;
    dlp::kernel_frame::scalingType   betaScalingType;
    AOCL_DLP_MEMORY_TAG              mtag_b;

    static constexpr int NUM_USABLE_MASKS = 7;
    static constexpr int MASK_START_IDX   = 1;
    Xbyak::Opmask        mask_regs[NUM_USABLE_MASKS];

    /*
     * Register allocation for NR=64, F32 accumulation, K_SUB_ITER=4:
     *   NR/F32_PER_ZMM = 4 ZMMs for one panel
     *   accumReg = 4 * K_SUB_ITER = 16 (zmm16-zmm31)
     *   xReg     = 4  (zmm12-zmm15, one per K_SUB_ITER step)
     *   bReg     = 4  (zmm8-zmm11, B converted to F32)
     *   yReg     = 4  (zmm28-zmm31, for Y load/store - overlap with
     *                   upper accums after finalAccumulate)
     */
    int xReg;
    int bReg;
    int accumReg;
    int yReg;

    int xBaseIdx;
    int bBaseIdx;
    int accumBaseIdx;
    int yBaseIdx;

    Xbyak::Reg64 stackPtr;
    Xbyak::Reg64 regBptr;
    Xbyak::Reg64 regXptr;
    Xbyak::Reg64 regYptr, regTmpYptr;
    Xbyak::Reg64 regNIter;
    Xbyak::Reg64 regKIter;
    Xbyak::Reg64 regKSubIter;
    Xbyak::Reg64 regRsB;
    Xbyak::Reg64 regPsB;
    Xbyak::Reg64 regTmp1;
    Xbyak::Reg64 regTmp2;
    Xbyak::Reg64 regIncN;
    Xbyak::Reg64 regIncK;

    Xbyak::Label label_n_loop_start;
    Xbyak::Label label_n_loop_end;
    Xbyak::Label label_n_loop_k_loop_start;
    Xbyak::Label label_n_loop_k_loop_end;
    Xbyak::Label label_n_fringe_k_loop_start;
    Xbyak::Label label_n_fringe_k_loop_end;
    Xbyak::Label label_n_fringe_start;
    Xbyak::Label label_n_fringe_end;
    Xbyak::Label label_n_loop_k_fringe_start;
    Xbyak::Label label_n_loop_k_fringe_end;
    Xbyak::Label label_n_fringe_k_fringe_start;
    Xbyak::Label label_n_fringe_k_fringe_end;

    dlp::jit::jitGeneratorError allocateRegisters();
    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);
    void initializeParameters(utils::gemvM1GeneratorParams& params);
    void regInit(int baseIdx, int numRegs);
    dlp::jit::jitGeneratorError offsetBPtr(int temp);

    dlp::jit::jitGeneratorError loopKSubIter(bool kfringe, bool nfringe);
    dlp::jit::jitGeneratorError computeKxNR(bool nMask);
    dlp::jit::jitGeneratorError computeKxnfringe();
    dlp::jit::jitGeneratorError compute1xNR(bool nMask);
    dlp::jit::jitGeneratorError compute1xnfringe();
    dlp::jit::jitGeneratorError finalAccumulate();

    dlp::jit::jitGeneratorError scaleWithAlpha();
    dlp::jit::jitGeneratorError scaleYWithBeta(bool nMask);
    dlp::jit::jitGeneratorError scaleYWithBetaFringe();

    dlp::jit::jitGeneratorError storeYValues(bool nMask);
    dlp::jit::jitGeneratorError storeYValuesFringe();

    dlp::jit::jitGeneratorError loadMasks();

    std::unique_ptr<gen::kernelOpsHandler>            kernelOpsHandlerPtr;
    std::vector<dlp::kernel_frame::kernelOpsMetaData> kernelOpsVector;
};

} // namespace amdzen::gen
