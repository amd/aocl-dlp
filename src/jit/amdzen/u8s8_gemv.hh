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
#include "jit/xbyak/xbyak.h"
#include "jit/xbyak/xbyak_util.h"
#include "jit_generator_utils.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "kernel_ops_handler.hh"
#include "traits.hh"

namespace amdzen::gen {

/**
 * @brief JIT generator for u8s8s32os32 VNNI GEMV N=1 kernels using Xbyak
 *
 * This class generates VNNI optimized kernels for u8 * s8 -> s32 GEMV
 * operations where N=1 (matrix-vector multiplication). Supports AVX512_VNNI
 * instruction set. Uses MR x 1 blocking pattern (no NR tiling needed).
 */

template<utils::kernelInstrType KType>
class jitU8S8VNNI_GEMVN1 : public Xbyak::CodeGenerator
{
  public:
    // Constructor that takes buffer and its size for JIT code dumping
    jitU8S8VNNI_GEMVN1(void* buffer, size_t bufferSize);
    ~jitU8S8VNNI_GEMVN1()                               = default;
    jitU8S8VNNI_GEMVN1(jitU8S8VNNI_GEMVN1&)             = delete;
    jitU8S8VNNI_GEMVN1& operator=(jitU8S8VNNI_GEMVN1&)  = delete;
    jitU8S8VNNI_GEMVN1(jitU8S8VNNI_GEMVN1&&)            = delete;
    jitU8S8VNNI_GEMVN1& operator=(jitU8S8VNNI_GEMVN1&&) = delete;

    /**
     * @brief Generate the complete u8s8s32os32 VNNI GEMV kernel
     */
    dlp::jit::jitGeneratorError generateKernel(
        utils::gemvN1GeneratorParams& params);

  private:
    // =================================================================
    // TYPE DEFINITIONS AND ARCHITECTURE TRAITS
    // =================================================================
    using Traits  = amdzen::traits::ArchitectureTraits<KType>;
    using RegType = typename Traits::RegType;

    // =================================================================
    // VNNI-SPECIFIC CONSTANTS
    // =================================================================
    static constexpr int VNNI_GROUP_SIZE = 4; // VNNI groups 4 int8 elements

    // Note: Classic framework guarantees VNNI-packed buffers
    // No validation or configuration needed in JIT generator

    // =================================================================
    // KERNEL CONFIGURATION
    // =================================================================
    int numRegs  = Traits::numRegs;
    int RegBytes = Traits::regBytes;

    // GEMV-specific dimensions (adapted from F32 GEMV)
    int nElemsPerReg; // SIMD width for int32 accumulators
    int c_downscale;
    int MR;
    int M_LEFT;

    dlp::kernel_frame::storageFormat yFormat;
    dlp::kernel_frame::scalingType   alphaScalingType;
    dlp::kernel_frame::scalingType   betaScalingType;

    // Add mask register array (for AVX512)
    static constexpr int NUM_USABLE_MASKS = 7;        // k1-k7 available
    static constexpr int MASK_START_IDX   = 1;        // Start from k1
    Xbyak::Opmask        mask_regs[NUM_USABLE_MASKS]; // Array of usable masks

    // =================================================================
    // REGISTER ALLOCATION
    // =================================================================
    int aReg;     // Number of registers for matrix A (LOAD_UNROLL equivalent)
    int xReg;     // Number of registers for vector x (1 register needed)
    int accumReg; // Number of registers for accumulation (MR registers)
    int yReg;     // Number of registers for loading/storing Y (MR/simdWidth)
    int tmpReg;   // Number of registers for temporary use (reduction)
    int maskReg;  // Number of registers for mask

    // Register base indices
    int aRegIdx;      // Starting index for A registers (from beginning)
    int xRegIdx;      // Starting index for x register
    int xBaseIdx;     // Starting index for X registers
    int accumBaseIdx; // Starting index for accumulation registers (from end)
    int yBaseIdx;     // Starting index for Y registers
    int tmpBaseIdx;   // Starting index for temporary registers
    int maskBaseIdx;  // Starting index for mask registers

    // Matrix/Vector pointers and strides
    Xbyak::Reg64 stackPtr;            // Stack frame pointer
    Xbyak::Reg64 regAptr, regTmpAptr; // Pointer to matrix A and its temp
    Xbyak::Reg64 regXptr;             // Pointer to vector x
    Xbyak::Reg64 regYptr, regTmpYptr; // Pointer to vector y and its temp
    Xbyak::Reg64 regRsA;              // Row stride for A
    Xbyak::Reg64 regCsA;              // Column stride for A
    Xbyak::Reg64 regRsC;              // Row stride for C (Y)
    Xbyak::Reg64 regMIter;            // M-loop iterator
    Xbyak::Reg64 regKIter;            // K-loop iterator
    Xbyak::Reg64 regTmp1;             // General purpose temporary register 1
    Xbyak::Reg64 regTmp2;             // General purpose temporary register 2
    Xbyak::Reg64 regTmp3;             // General purpose temporary register 3

    // =================================================================
    // CORE SETUP AND INITIALIZATION
    // =================================================================

    /**
     * @brief Allocate registers based on MR x 1 GEMV tiling (no NR)
     */
    dlp::jit::jitGeneratorError allocateRegisters();

    /**
     * @brief Initialize stack frame registers
     */
    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);

    /**
     * @brief Initialize parameter pointers from stack
     */
    void initializeParameters();

    /**
     * @brief Initialize registers with zero values (utility function)
     */
    void regInit(int baseIdx, int numRegs);

    // =================================================================
    // CORE COMPUTATION METHODS
    // =================================================================
    /**
     * @brief Generate the M-loop that processes multiple M blocks
     * Follows same pattern as GEMM M-loop but calls GEMV-specific IR loop
     */
    dlp::jit::jitGeneratorError generateMLoop(
        utils::gemvN1GeneratorParams& params);

    /**
     * @brief Generate the IR loop with K-loop, scaling and post-ops
     * Adapted for GEMV: loads X vector, processes MR rows, accumulates
     */
    dlp::jit::jitGeneratorError generateIrLoop(int mSize);

    // =================================================================
    // GEMV-SPECIFIC COMPUTATION METHODS
    // ==================================================================

    /**
     * @brief Process MR block: main computation loop for M rows
     */
    dlp::jit::jitGeneratorError processMRBlock(int  mSize,
                                               bool isFringe = false);

    /**
     * @brief Reduce accumulation registers from wide SIMD to narrower results
     */
    dlp::jit::jitGeneratorError reduceAccumulation(int mSize);

    /**
     * @brief Reduce a single accumulation register to XMM
     */
    dlp::jit::jitGeneratorError reduceAccToXmm(int accIdx,
                                               int tmpBaseIdx,
                                               int subBlockSize);

    /**
     * @brief Rearrange Y vector for S32 row-major storage format
     */
    dlp::jit::jitGeneratorError rearrangeY_rowStored_S32(int mSize);

    /**
     * @brief Rearrange Y vector for U8 row-major storage format
     */
    dlp::jit::jitGeneratorError rearrangeY_rowStored_U8(int mSize);

    /**
     * @brief Rearrange Y vector for S8 row-major storage format
     */
    dlp::jit::jitGeneratorError rearrangeY_rowStored_S8(int mSize);

    /**
     * @brief Rearrange Y vector for F32 row-major storage format
     */
    dlp::jit::jitGeneratorError rearrangeY_rowStored_F32(int mSize);

    /**
     * @brief Rearrange Y vector for BF16 row-major storage format
     */
    dlp::jit::jitGeneratorError rearrangeY_rowStored_BF16(int mSize);

    // =================================================================
    // SCALING AND POST-PROCESSING
    // =================================================================
    /**
     * @brief Scale accumulation with alpha factor
     */
    dlp::jit::jitGeneratorError scaleAlpha(int mSize);

    /**
     * @brief Scale Y with beta factor and accumulate
     */
    dlp::jit::jitGeneratorError scaleBeta(int mSize);

    /**
     * @brief Scale Y for column-major storage format
     */
    dlp::jit::jitGeneratorError scaleYWithBeta_S32(int mSize, bool isRowStored);

    /**
     * @brief Scale Y for column-major storage format (uint8)
     */
    dlp::jit::jitGeneratorError scaleYWithBeta_U8(int mSize, bool isRowStored);

    /**
     * @brief Scale Y vector with beta factor for int8 data type
     */
    dlp::jit::jitGeneratorError scaleYWithBeta_S8(int mSize, bool isRowStored);

    /**
     * @brief Scale Y vector with beta factor for float32 data type
     */
    dlp::jit::jitGeneratorError scaleYWithBeta_F32(int mSize, bool isRowStored);

    /**
     * @brief Scale Y vector with beta factor for bfloat16 data type
     */
    dlp::jit::jitGeneratorError scaleYWithBeta_BF16(int  mSize,
                                                    bool isRowStored);

    /**
     * @brief Generate post-operations
     */
    dlp::jit::jitGeneratorError generatePostOps(
        utils::gemvN1GeneratorParams& params);

    /**
     * @brief Updating the C buffer pointers for the downscale operation
     */
    void updateCBufferPointers();

    // =================================================================
    // RESULT PROCESSING AND OUTPUT
    // =================================================================

    /**
     * @brief Store Result
     */
    dlp::jit::jitGeneratorError storeResult(int mSize);

    /**
     * @brief Store results for S32 data type (unified row/column storage)
     */
    dlp::jit::jitGeneratorError storeResult_S32(int mSize, bool isRowStored);

    /**
     * @brief Store results for U8 data type (unified row/column storage)
     */
    dlp::jit::jitGeneratorError storeResult_U8(int mSize, bool isRowStored);

    /**
     * @brief Store results for S8 data type (unified row/column storage)
     */
    dlp::jit::jitGeneratorError storeResult_S8(int mSize, bool isRowStored);

    /**
     * @brief Store results for F32 data type (unified row/column storage)
     */
    dlp::jit::jitGeneratorError storeResult_F32(int mSize, bool isRowStored);

    /**
     * @brief Store results for BF16 data type (unified row/column storage)
     */
    dlp::jit::jitGeneratorError storeResult_BF16(int mSize, bool isRowStored);

    /**
     * @brief Store results for S32 row-major storage format
     */
    dlp::jit::jitGeneratorError storeY_rowStored_S32(int mSize);

    /**
     * @brief Store results for U8 row-major storage format
     */
    dlp::jit::jitGeneratorError storeY_rowStored_U8(int mSize);

    /**
     * @brief Store results for S8 row-major storage format
     */
    dlp::jit::jitGeneratorError storeY_rowStored_S8(int mSize);

    /**
     * @brief Store results for F32 row-major storage format
     */
    dlp::jit::jitGeneratorError storeY_rowStored_F32(int mSize);

    /**
     * @brief Store results for BF16 row-major storage format
     */
    dlp::jit::jitGeneratorError storeY_rowStored_BF16(int mSize);

    // =================================================================
    // GEMV-SPECIFIC HELPER METHODS
    // =================================================================
    /**
     * @brief Load masks for fringe case handling (AVX512 only)
     */
    dlp::jit::jitGeneratorError loadMasks();
};
} // namespace amdzen::gen
