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
 * @brief JIT generator for FP16 GEMM kernels using Xbyak
 *
 * This class generates AVX-512-FP16 optimized kernels for fp16 GEMM
 * operations. Uses native FP16 accumulation with vfmadd231ph instruction.
 * Supports configurable MR/NR blocking with 32 FP16 elements per ZMM.
 */
template<utils::kernelInstrType KType>
class jitFP16_GEMM : public Xbyak::CodeGenerator
{
  public:
    // Constructor that takes buffer and its size for JIT code dumping
    jitFP16_GEMM(void* buffer, size_t bufferSize);
    ~jitFP16_GEMM()                         = default;
    jitFP16_GEMM(jitFP16_GEMM&)             = delete;
    jitFP16_GEMM& operator=(jitFP16_GEMM&)  = delete;
    jitFP16_GEMM(jitFP16_GEMM&&)            = delete;
    jitFP16_GEMM& operator=(jitFP16_GEMM&&) = delete;

    /**
     * @brief Generate the complete FP16 GEMM kernel
     */
    dlp::jit::jitGeneratorError generateKernel(utils::generatorParams& params);

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
    static constexpr int FP16_ELEM_SIZE = FP16Types::elemSize;
    static constexpr int FP16_PER_ZMM   = FP16Types::elemsPerZmm;

    // =================================================================
    // KERNEL CONFIGURATION
    // =================================================================
    int  numRegs  = Traits::numRegs;  // 32 for AVX-512
    int  RegSize  = Traits::regSize;  // 512 bits
    int  RegBytes = Traits::regBytes; // 64 bytes
    int  MR, NR;
    int  c_downscale;
    bool useMask =
        false; // Flag to indicate if masked instructions are generated

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
    // CORE SETUP AND INITIALIZATION
    // =================================================================
    /**
     * @brief Allocate registers based on MR x NR tiling
     * For FP16: 32 elements per ZMM, so NR=128 uses 4 B registers
     */
    dlp::jit::jitGeneratorError allocateRegisters();

    /**
     * @brief Initialize stack frame registers
     */
    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);

    /**
     * @brief Initialize parameter pointers from stack
     */
    void initializeParameters(bool mLoop = false);

    /**
     * @brief Initialize accumulator registers to zero
     */
    void initializeAccumulators(utils::generatorParams& params);

    // =================================================================
    // CORE COMPUTATION METHODS
    // =================================================================
    /**
     * @brief Generate the M-loop that calls IR loop multiple times
     */
    dlp::jit::jitGeneratorError generateMLoop(utils::generatorParams& params);

    /**
     * @brief Generate the IR loop with K-loop, scaling and post-ops
     */
    dlp::jit::jitGeneratorError generateIrLoop(utils::generatorParams& params);

    // =================================================================
    // K-LOOP COMPUTATION
    // =================================================================
    /**
     * @brief Load B matrix values into registers using vmovdqu16
     */
    dlp::jit::jitGeneratorError loadBValues();

    /**
     * @brief Broadcast A element and FMA with B using vfmadd231ph
     */
    dlp::jit::jitGeneratorError broadcastAFMAwithB(bool isKRemainder);

    /**
     * @brief Unroll K-loop computation
     */
    dlp::jit::jitGeneratorError kUnroll(int unroll, bool isKRemainder);

    // =================================================================
    // SCALING OPERATIONS
    // =================================================================
    /**
     * @brief Apply alpha scaling using vmulph
     */
    dlp::jit::jitGeneratorError scaleAlpha();

    /**
     * @brief Apply beta scaling for existing C values
     */
    dlp::jit::jitGeneratorError scaleBeta();

    // =================================================================
    // POST-OPERATIONS AND RESULT STORAGE
    // =================================================================
    /**
     * @brief Generate post-operations sequence
     */
    dlp::jit::jitGeneratorError generatePostOps(utils::generatorParams& params);

    /**
     * @brief Dispatch to appropriate store function
     */
    dlp::jit::jitGeneratorError storeResult();

    /**
     * @brief Store results as FP16 using vmovdqu16
     */
    dlp::jit::jitGeneratorError storeResultFP16();

    /**
     * @brief Move C pointer for next M iteration
     */
    dlp::jit::jitGeneratorError moveCPtr();
};

// Type alias for AVX-512 FP16 GEMM generator
using jitGemmGenerator_fp16_avx512 =
    jitFP16_GEMM<utils::kernelInstrType::avx512_zmm_32_reg>;

} // namespace amdzen::gen
