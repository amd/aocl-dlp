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
#include "traits.hh"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace amdzen::gen {

/**
 * @brief JIT generator for FP16 GEMV N=1 kernels using Xbyak
 *
 * This class generates AVX512-FP16 optimized kernels for FP16 GEMV
 * operations where N=1 (matrix-vector multiplication y = A * x).
 * Uses native FP16 accumulation with vfmadd231ph instruction.
 * Uses MR x 1 blocking pattern (MR=16 for FP16).
 */
template<utils::kernelInstrType KType>
class jitFP16GEMVN1 : public Xbyak::CodeGenerator
{
  public:
    jitFP16GEMVN1(void* buffer, size_t bufferSize);
    ~jitFP16GEMVN1()                          = default;
    jitFP16GEMVN1(jitFP16GEMVN1&)             = delete;
    jitFP16GEMVN1& operator=(jitFP16GEMVN1&)  = delete;
    jitFP16GEMVN1(jitFP16GEMVN1&&)            = delete;
    jitFP16GEMVN1& operator=(jitFP16GEMVN1&&) = delete;

    /**
     * @brief Generate the complete FP16 GEMV N=1 kernel
     */
    dlp::jit::jitGeneratorError generateKernel(
        utils::gemvN1GeneratorParams& params);

  private:
    // =================================================================
    // TYPE DEFINITIONS AND ARCHITECTURE TRAITS
    // =================================================================
    using Traits    = amdzen::traits::ArchitectureTraits<KType>;
    using FP16Types = amdzen::traits::kernel_types<
        dlp::kernel_frame::kernelDatatype::f16f16f16of16>;
    using RegType = typename Traits::RegType;

    // =================================================================
    // FP16-SPECIFIC CONSTANTS
    // =================================================================
    static constexpr int FP16_PER_ZMM = FP16Types::elemsPerZmm;

    // =================================================================
    // KERNEL CONFIGURATION
    // =================================================================
    int numRegs  = Traits::numRegs;  // 32 for AVX512
    int RegBytes = Traits::regBytes; // 64 for ZMM

    int nElemsPerReg; // = 32 for FP16
    int MR;           // = 16 for FP16 GEMV N=1
    int M_LEFT;

    dlp::kernel_frame::storageFormat yFormat;
    dlp::kernel_frame::scalingType   alphaScalingType;
    dlp::kernel_frame::scalingType   betaScalingType;

    // Mask register array (k1-k7 available, k0 reserved)
    static constexpr int NUM_USABLE_MASKS = 7;
    static constexpr int MASK_START_IDX   = 1;
    Xbyak::Opmask        mask_regs[NUM_USABLE_MASKS];

    // =================================================================
    // REGISTER ALLOCATION
    // =================================================================
    int xReg;     // Register for X broadcast (1 register)
    int accumReg; // Accumulation registers (MR registers)
    int tmpReg;   // Temporary registers for reduction and A loading
    int yReg;     // Registers for Y load/store

    int xBaseIdx;
    int accumBaseIdx;
    int tmpBaseIdx;
    int yBaseIdx;

    // Scalar registers
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

    // =================================================================
    // CORE SETUP AND INITIALIZATION
    // =================================================================
    dlp::jit::jitGeneratorError allocateRegisters();
    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);
    void initializeParameters();
    void regInit(int baseIdx, int numRegs);

    // =================================================================
    // CORE COMPUTATION METHODS
    // =================================================================
    dlp::jit::jitGeneratorError generateMLoop(
        utils::gemvN1GeneratorParams& params);
    dlp::jit::jitGeneratorError generateIrLoop(int mSize);
    dlp::jit::jitGeneratorError processMRBlock(int  mSize,
                                               bool isFringe = false);

    // =================================================================
    // HORIZONTAL REDUCTION (32 FP16 -> 1 scalar)
    // =================================================================
    dlp::jit::jitGeneratorError reduceAccumulation(int mSize);
    dlp::jit::jitGeneratorError reduceAccToScalar(int accIdx, int tmpIdx);

    // =================================================================
    // SCALING AND POST-PROCESSING
    // =================================================================
    dlp::jit::jitGeneratorError scaleAlpha(int mSize);
    dlp::jit::jitGeneratorError scaleBeta(int mSize);
    dlp::jit::jitGeneratorError scaleYWithBeta_FP16(int  mSize,
                                                    bool isRowStored);

    // =================================================================
    // RESULT STORAGE
    // =================================================================
    dlp::jit::jitGeneratorError storeResult(int mSize);
    dlp::jit::jitGeneratorError storeY_rowStored_FP16(int mSize);
    dlp::jit::jitGeneratorError storeY_colStored_FP16(int mSize);

    // =================================================================
    // HELPER METHODS
    // =================================================================
    dlp::jit::jitGeneratorError loadMasks();
};

/**
 * @brief JIT generator for FP16 GEMV M=1 kernels using Xbyak
 *
 * This class generates AVX512-FP16 optimized kernels for FP16 GEMV
 * operations where M=1 (vector-matrix multiplication y = x * B).
 * Uses native FP16 accumulation with vfmadd231ph instruction.
 * Uses 1 x NR blocking pattern (NR=128 for FP16, 4 ZMMs).
 */
template<utils::kernelInstrType KType>
class jitFP16GEMVM1 : public Xbyak::CodeGenerator
{
  public:
    jitFP16GEMVM1(void* buffer, size_t bufferSize);
    ~jitFP16GEMVM1()                          = default;
    jitFP16GEMVM1(jitFP16GEMVM1&)             = delete;
    jitFP16GEMVM1& operator=(jitFP16GEMVM1&)  = delete;
    jitFP16GEMVM1(jitFP16GEMVM1&&)            = delete;
    jitFP16GEMVM1& operator=(jitFP16GEMVM1&&) = delete;

    /**
     * @brief Generate the complete FP16 GEMV M=1 kernel
     */
    dlp::jit::jitGeneratorError generateKernel(
        utils::gemvM1GeneratorParams& params);

  private:
    // =================================================================
    // TYPE DEFINITIONS AND ARCHITECTURE TRAITS
    // =================================================================
    using Traits    = amdzen::traits::ArchitectureTraits<KType>;
    using FP16Types = amdzen::traits::kernel_types<
        dlp::kernel_frame::kernelDatatype::f16f16f16of16>;
    using RegType = typename Traits::RegType;

    // =================================================================
    // FP16-SPECIFIC CONSTANTS
    // =================================================================
    static constexpr int FP16_PER_ZMM = FP16Types::elemsPerZmm;

    // =================================================================
    // KERNEL CONFIGURATION
    // =================================================================
    int numRegs  = Traits::numRegs;
    int RegBytes = Traits::regBytes;

    int nElemsPerReg; // = 32 for FP16
    int NR;           // = 128 for FP16 GEMV M=1 (4 ZMMs)
    int N_LEFT;
    int KC;
    int K_SUB_ITER;

    dlp::kernel_frame::storageFormat yFormat;
    dlp::kernel_frame::scalingType   alphaScalingType;
    dlp::kernel_frame::scalingType   betaScalingType;
    AOCL_MEMORY_TAG                  mtag_b;

    // Mask register array
    static constexpr int NUM_USABLE_MASKS = 7;
    static constexpr int MASK_START_IDX   = 1;
    Xbyak::Opmask        mask_regs[NUM_USABLE_MASKS];

    // =================================================================
    // REGISTER ALLOCATION (K_SUB_ITER pattern following U8S8)
    // =================================================================
    int xReg;     // K_SUB_ITER registers for X broadcast
    int bReg;     // NR/nElemsPerReg registers for B loads
    int accumReg; // (NR/nElemsPerReg) * K_SUB_ITER accumulation registers
    int yReg;     // NR/nElemsPerReg registers for Y load/store

    int xBaseIdx;     // zmm12-15 (for K_SUB_ITER=4)
    int bBaseIdx;     // zmm8-11
    int accumBaseIdx; // zmm16-31 (16 accumulators for software pipelining)
    int yBaseIdx;     // zmm28-31 (no conflict after finalAccumulate)

    // Scalar registers
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

    // Labels for code sections
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

    // =================================================================
    // CORE SETUP AND INITIALIZATION
    // =================================================================
    dlp::jit::jitGeneratorError allocateRegisters();
    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);
    void initializeParameters(utils::gemvM1GeneratorParams& params);
    void regInit(int baseIdx, int numRegs);

    // =================================================================
    // CORE COMPUTATION METHODS
    // =================================================================
    dlp::jit::jitGeneratorError loopKSubIter(bool kfringe, bool nfringe);
    dlp::jit::jitGeneratorError computeKxNR(bool nMask);
    dlp::jit::jitGeneratorError computeKxnfringe();
    dlp::jit::jitGeneratorError compute1xNR(bool nMask);
    dlp::jit::jitGeneratorError compute1xnfringe();
    dlp::jit::jitGeneratorError finalAccumulate();

    // =================================================================
    // SCALING AND POST-PROCESSING
    // =================================================================
    dlp::jit::jitGeneratorError scaleWithAlpha();
    dlp::jit::jitGeneratorError scaleYWithBeta(bool nMask);
    dlp::jit::jitGeneratorError scaleYWithBetaFringe();

    // =================================================================
    // RESULT STORAGE
    // =================================================================
    dlp::jit::jitGeneratorError storeYValues(bool nMask);
    dlp::jit::jitGeneratorError storeYValuesFringe();

    // =================================================================
    // HELPER METHODS
    // =================================================================
    dlp::jit::jitGeneratorError loadMasks();
};

} // namespace amdzen::gen
