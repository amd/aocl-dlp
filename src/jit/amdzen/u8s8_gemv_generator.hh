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
    // Constructor that specifies the maximum size of generated JIT code.
    // Buffer allocation and AutoGrow behavior are managed internally by Xbyak.
    jitU8S8VNNI_GEMVN1(size_t maxSize);
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

    /**
     * @brief Generate constant data for BF16 conversion (Skylake compatibility)
     */
    void generateConstantData();

    // Labels for BF16 constant data (for Skylake compatibility)
    Xbyak::Label label_bf16_round_bias;
    Xbyak::Label label_bf16_lsb_mask;

    // Kernel operations handler for post-ops
    std::unique_ptr<gen::kernelOpsHandler>            kernelOpsHandlerPtr;
    std::vector<dlp::kernel_frame::kernelOpsMetaData> kernelOpsVector;
    bool                                              accumulatorsAreF32 =
        false; // Track if accumulators were converted to F32 for post-ops
};

/**
 * @brief JIT generator for u8s8s32os32 VNNI GEMV M=1 kernels using Xbyak
 *
 * This class generates VNNI optimized kernels for u8 * s8 -> s32 GEMV
 * operations where M=1 (vector-matrix multiplication). Supports AVX512_VNNI
 * instruction set. Uses 1 x NR blocking pattern with K_SUB_ITER optimization.
 */
template<utils::kernelInstrType KType>
class jitU8S8VNNI_GEMVM1 : public Xbyak::CodeGenerator
{
  public:
    // Constructor that specifies the maximum size of generated JIT code.
    // Buffer allocation and AutoGrow behavior are managed internally by Xbyak.
    jitU8S8VNNI_GEMVM1(size_t maxSize);
    ~jitU8S8VNNI_GEMVM1()                               = default;
    jitU8S8VNNI_GEMVM1(jitU8S8VNNI_GEMVM1&)             = delete;
    jitU8S8VNNI_GEMVM1& operator=(jitU8S8VNNI_GEMVM1&)  = delete;
    jitU8S8VNNI_GEMVM1(jitU8S8VNNI_GEMVM1&&)            = delete;
    jitU8S8VNNI_GEMVM1& operator=(jitU8S8VNNI_GEMVM1&&) = delete;

    /**
     * @brief Generate the complete u8s8s32os32 VNNI GEMV M=1 kernel
     */
    dlp::jit::jitGeneratorError generateKernel(
        utils::gemvM1GeneratorParams& params);

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

    // =================================================================
    // KERNEL CONFIGURATION
    // =================================================================
    int numRegs  = Traits::numRegs;
    int RegBytes = Traits::regBytes;

    // GEMV M=1 specific dimensions
    int nElemsPerReg; // SIMD width for int32 accumulators
    int c_downscale;
    int NR;
    int N_LEFT;
    int KC;
    int K_SUB_ITER;

    dlp::kernel_frame::storageFormat yFormat;
    dlp::kernel_frame::scalingType   alphaScalingType;
    dlp::kernel_frame::scalingType   betaScalingType;
    AOCL_DLP_MEMORY_TAG              mtag_b;

    // Add mask register array (for AVX512)
    static constexpr int NUM_USABLE_MASKS = 7;        // k1-k7 available
    static constexpr int MASK_START_IDX   = 1;        // Start from k1
    Xbyak::Opmask        mask_regs[NUM_USABLE_MASKS]; // Array of usable masks

    // =================================================================
    // REGISTER ALLOCATION
    // =================================================================
    int xReg;     // Number of registers for vector x (K_SUB_ITER registers)
    int bReg;     // Number of registers for matrix B (K_SUB_ITER registers)
    int accumReg; // Number of registers for accumulation ((NR/16)*K_SUB_ITER)
    int yReg;     // Number of registers for loading/storing Y (NR/16)
    int maskReg;  // Number of registers for mask

    // Register base indices
    int xBaseIdx;     // Starting index for X registers
    int bBaseIdx;     // Starting index for B registers
    int accumBaseIdx; // Starting index for accumulation registers
    int yBaseIdx;     // Starting index for Y registers
    int maskBaseIdx;  // Starting index for mask registers

    // Matrix/Vector pointers and strides
    Xbyak::Reg64 stackPtr;            // Stack frame pointer
    Xbyak::Reg64 regBptr;             // Pointer to matrix B
    Xbyak::Reg64 regXptr;             // Pointer to vector x
    Xbyak::Reg64 regYptr, regTmpYptr; // Pointer to vector y and its temp
    Xbyak::Reg64 regNIter;            // N-loop iterator
    Xbyak::Reg64 regKIter;            // K-loop iterator
    Xbyak::Reg64 regKSubIter;         // K sub-iteration iterator
    Xbyak::Reg64 regRsB;              // Row stride for B
    Xbyak::Reg64 regPsB;              // Panel stride for B
    Xbyak::Reg64 regTmp1;             // General purpose temporary register 1
    Xbyak::Reg64 regTmp2;             // General purpose temporary register 2
    Xbyak::Reg64 regIncN;             // N increment counter
    Xbyak::Reg64 regIncK;             // K increment counter
    Xbyak::Reg64 regDebugPtr;         // Debug buffer pointer register

    // Labels for code sections
    Xbyak::Label label_n_loop_start;            // Main n-dimension loop
    Xbyak::Label label_n_loop_end;              // End of n-dimension loop
    Xbyak::Label label_n_loop_k_loop_start;     // Main k-dimension loop
    Xbyak::Label label_n_loop_k_loop_end;       // End of k-dimension loop
    Xbyak::Label label_n_fringe_k_loop_start;   // Main k-dimension loop
    Xbyak::Label label_n_fringe_k_loop_end;     // End of k-dimension loop
    Xbyak::Label label_n_fringe_start;          // Handle n-dimension remainder
    Xbyak::Label label_n_fringe_end;            // End of n-dimension remainder
    Xbyak::Label label_n_loop_k_fringe_start;   // Handle k-dimension remainder
    Xbyak::Label label_n_loop_k_fringe_end;     // End of k-dimension remainder
    Xbyak::Label label_n_fringe_k_fringe_start; // Handle k-dimension remainder
    Xbyak::Label label_n_fringe_k_fringe_end;   // End of k-dimension remainder

    // =================================================================
    // CORE SETUP AND INITIALIZATION
    // =================================================================

    /**
     * @brief Allocate registers based on 1 x NR GEMV M=1 tiling
     */
    dlp::jit::jitGeneratorError allocateRegisters();

    /**
     * @brief Initialize stack frame registers
     */
    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);

    /**
     * @brief Initialize parameter pointers from stack
     */
    void initializeParameters(utils::gemvM1GeneratorParams& params);

    /**
     * @brief Initialize registers with zero values (utility function)
     */
    void regInit(int baseIdx, int numRegs);

    /**
     * @brief Calculate B pointer offset using power-of-2 decomposition
     */
    dlp::jit::jitGeneratorError offsetBPtr(int temp);

    // =================================================================
    // CORE COMPUTATION METHODS
    // =================================================================

    /**
     * @brief K sub-iteration loop handler
     */
    dlp::jit::jitGeneratorError loopKSubIter(bool kfringe, bool nfringe);

    /**
     * @brief Compute K_SUB_ITER x NR block (main compute function)
     */
    dlp::jit::jitGeneratorError computeKxNR(bool nMask);

    /**
     * @brief Compute K_SUB_ITER x n_fringe block (fringe compute function)
     */
    dlp::jit::jitGeneratorError computeKxnfringe();

    /**
     * @brief Compute 1 x NR block (single K iteration)
     */
    dlp::jit::jitGeneratorError compute1xNR(bool nMask, bool isLastKGroup);

    /**
     * @brief Compute 1 x n_fringe block (single K iteration, fringe)
     */
    dlp::jit::jitGeneratorError compute1xnfringe(bool isLastKGroup);

    /**
     * @brief Final accumulation across K_SUB_ITER accumulators
     */
    dlp::jit::jitGeneratorError finalAccumulate();

    // =================================================================
    // SCALING AND POST-PROCESSING
    // =================================================================

    /**
     * @brief Scale accumulation with alpha factor
     */
    dlp::jit::jitGeneratorError scaleWithAlpha();

    /**
     * @brief Scale Y with beta factor and accumulate
     */
    dlp::jit::jitGeneratorError scaleYWithBeta(bool nMask);

    /**
     * @brief Scale Y with beta for fringe case
     */
    dlp::jit::jitGeneratorError scaleYWithBetaFringe(bool isBetaOne);

    /**
     * @brief Scale Y with beta factor for S32 data type
     */
    dlp::jit::jitGeneratorError scaleYWithBeta_S32(bool nMask, bool isBetaOne);

    /**
     * @brief Scale Y with beta factor for U8 data type
     */
    dlp::jit::jitGeneratorError scaleYWithBeta_U8(bool nMask, bool isBetaOne);

    /**
     * @brief Scale Y with beta factor for S8 data type
     */
    dlp::jit::jitGeneratorError scaleYWithBeta_S8(bool nMask, bool isBetaOne);

    /**
     * @brief Scale Y with beta factor for F32 data type
     */
    dlp::jit::jitGeneratorError scaleYWithBeta_F32(bool nMask, bool isBetaOne);

    /**
     * @brief Scale Y with beta factor for BF16 data type
     */
    dlp::jit::jitGeneratorError scaleYWithBeta_BF16(bool nMask, bool isBetaOne);

    /**
     * @brief Scale Y with beta for fringe case (S32)
     */
    dlp::jit::jitGeneratorError scaleYWithBetaFringe_S32(bool isBetaOne);

    /**
     * @brief Scale Y with beta for fringe case (U8)
     */
    dlp::jit::jitGeneratorError scaleYWithBetaFringe_U8(bool isBetaOne);

    /**
     * @brief Scale Y with beta for fringe case (S8)
     */
    dlp::jit::jitGeneratorError scaleYWithBetaFringe_S8(bool isBetaOne);

    /**
     * @brief Scale Y with beta for fringe case (F32)
     */
    dlp::jit::jitGeneratorError scaleYWithBetaFringe_F32(bool isBetaOne);

    /**
     * @brief Scale Y with beta for fringe case (BF16)
     */
    dlp::jit::jitGeneratorError scaleYWithBetaFringe_BF16(bool isBetaOne);

    /**
     * @brief Update Y buffer pointers for downscale operations
     */
    void updateYBufferPointers();

    // =================================================================
    // RESULT PROCESSING AND OUTPUT
    // =================================================================

    /**
     * @brief Store Y values
     */
    dlp::jit::jitGeneratorError storeYValues(bool nMask);

    /**
     * @brief Store Y values for fringe case
     */
    dlp::jit::jitGeneratorError storeYValuesFringe();

    /**
     * @brief Store Y values for S32 data type
     */
    dlp::jit::jitGeneratorError storeYValues_S32(bool nMask);

    /**
     * @brief Store Y values for U8 data type
     */
    dlp::jit::jitGeneratorError storeYValues_U8(bool nMask);

    /**
     * @brief Store Y values for S8 data type
     */
    dlp::jit::jitGeneratorError storeYValues_S8(bool nMask);

    /**
     * @brief Store Y values for F32 data type
     */
    dlp::jit::jitGeneratorError storeYValues_F32(bool nMask);

    /**
     * @brief Store Y values for BF16 data type
     */
    dlp::jit::jitGeneratorError storeYValues_BF16(bool nMask);

    /**
     * @brief Store Y values for fringe case (S32)
     */
    dlp::jit::jitGeneratorError storeYValuesFringe_S32();

    /**
     * @brief Store Y values for fringe case (U8)
     */
    dlp::jit::jitGeneratorError storeYValuesFringe_U8();

    /**
     * @brief Store Y values for fringe case (S8)
     */
    dlp::jit::jitGeneratorError storeYValuesFringe_S8();

    /**
     * @brief Store Y values for fringe case (F32)
     */
    dlp::jit::jitGeneratorError storeYValuesFringe_F32();

    /**
     * @brief Store Y values for fringe case (BF16)
     */
    dlp::jit::jitGeneratorError storeYValuesFringe_BF16();

    // =================================================================
    // HELPER METHODS
    // =================================================================

    /**
     * @brief Load masks for fringe case handling (AVX512 only)
     */
    dlp::jit::jitGeneratorError loadMasks();

    /**
     * @brief Masked load for B matrix
     */
    dlp::jit::jitGeneratorError maskLoadB(int regIdx, int maskIdx);

    /**
     * @brief Generate constant data for BF16 conversion (Skylake compatibility)
     */
    void generateConstantData();

    // Labels for BF16 constant data (for Skylake compatibility)
    Xbyak::Label label_bf16_round_bias;
    Xbyak::Label label_bf16_lsb_mask;

    // Kernel operations handler for post-ops
    std::unique_ptr<gen::kernelOpsHandler>            kernelOpsHandlerPtr;
    std::vector<dlp::kernel_frame::kernelOpsMetaData> kernelOpsVector;
    bool                                              accumulatorsAreF32 =
        false; // Track if accumulators were converted to F32 for post-ops
};

} // namespace amdzen::gen
