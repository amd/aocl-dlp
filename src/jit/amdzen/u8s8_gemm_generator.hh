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
 * @brief JIT generator for u8s8s32 VNNI kernels using Xbyak
 *
 * This class generates VNNI optimized kernels for u8 * s8 -> s32 GEMM
 * operations. Supports AVX512_VNNI instruction set. It follows the classic
 * 6x64 blocking pattern but generates code at runtime.
 */
template<utils::kernelInstrType KType>
class jitU8S8VNNI_GEMM : public Xbyak::CodeGenerator
{
  public:
    // Constructor that specifies the maximum size of generated JIT code.
    // Buffer allocation and AutoGrow behavior are managed internally by Xbyak.
    jitU8S8VNNI_GEMM(size_t maxSize);
    ~jitU8S8VNNI_GEMM()                             = default;
    jitU8S8VNNI_GEMM(jitU8S8VNNI_GEMM&)             = delete;
    jitU8S8VNNI_GEMM& operator=(jitU8S8VNNI_GEMM&)  = delete;
    jitU8S8VNNI_GEMM(jitU8S8VNNI_GEMM&&)            = delete;
    jitU8S8VNNI_GEMM& operator=(jitU8S8VNNI_GEMM&&) = delete;

    /**
     * @brief Generate the complete u8s8s32 VNNI kernel
     */
    dlp::jit::jitGeneratorError generateKernel(utils::generatorParams& params);

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
    int  numRegs  = Traits::numRegs;
    int  RegSize  = Traits::regSize;
    int  RegBytes = Traits::regBytes;
    int  MR, NR;
    int  c_downscale;
    bool useMask =
        false; // Flag to indicate if masked instructions are generated
    bool accumulatorsAreF32 =
        false; // Track if accumulators were converted to F32 for post-ops

    // =================================================================
    // REGISTER ALLOCATION
    // =================================================================
    int aReg, bReg, bFullReg, bMaskReg, cReg;
    int aRegIdx, bRegIdx, cRegIdx, maskRegIdx;

    Xbyak::Opmask mask_regs[utils::NUM_USABLE_MASKS];

    // =================================================================
    // GENERAL PURPOSE REGISTERS
    // =================================================================
    Xbyak::Reg64 stackPtr;
    Xbyak::Reg64 regTmpAptr, regBptr, regTmpCptr;
    Xbyak::Reg64 regRsA, regCsA, regRsB, regRsC;
    Xbyak::Reg64 regKIter, regMiter;
    Xbyak::Reg64 regCPtr, regAPtr;
    Xbyak::Reg64 regTmp1, regTmp2, regTmp3;
    Xbyak::Reg64 regkernelOpsList, regkernelOpsAttr;

    // =================================================================
    // LABELS
    // =================================================================
    Xbyak::Label label_bf16_round_bias; // Constant data for BF16 conversion
    Xbyak::Label label_bf16_lsb_mask;   // Constant data for BF16 conversion

    // =================================================================
    // CORE SETUP AND INITIALIZATION
    // =================================================================
    /**
     * @brief Allocate registers based on MR x NR tiling
     */
    dlp::jit::jitGeneratorError allocateRegisters();

    /**
     * @brief Initialize stack frame registers (follows F32 pattern)
     */
    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);

    /**
     * @brief Initialize parameter pointers from stack
     */
    void initializeParameters(bool addIrLoop = false);

    // /**
    //  * @brief Generate function prologue with stack frame setup
    //  */
    // void generatePrologue(utils::generatorParams& params);

    // /**
    //  * @brief Generate function epilogue with stack cleanup
    //  */
    // void generateEpilogue();

    // /**
    //  * @brief Initialize accumulator registers to zero
    //  */
    void initializeAccumulators(utils::generatorParams& params);

    // =================================================================
    // CORE COMPUTATION METHODS
    // =================================================================
    /**
     * @brief Generate the M-loop that calls IR loop multiple times
     * Uses regMiter counter and moves C/A pointers between M blocks
     */
    dlp::jit::jitGeneratorError generateMLoop(utils::generatorParams& params);

    /**
     * @brief Generate the IR loop with K-loop, scaling and post-ops
     * Contains inLocalLabel/outLocalLabel for proper label scoping
     */
    dlp::jit::jitGeneratorError generateIrLoop(utils::generatorParams& params);

    // =================================================================
    // RESULT PROCESSING AND OUTPUT
    // =================================================================
    /**
     * @brief Store results from int32 accumulator to C matrix (dispatch
     * function)
     */
    dlp::jit::jitGeneratorError storeResult();

    /**
     * @brief Store results as int32 (original implementation)
     */
    dlp::jit::jitGeneratorError storeResultS32();

    /**
     * @brief Store results as int8 with signed saturation
     */
    dlp::jit::jitGeneratorError storeResultS8();

    /**
     * @brief Store results as uint8 with unsigned clipping
     */
    dlp::jit::jitGeneratorError storeResultU8();

    /**
     * @brief Store results as float32
     */
    dlp::jit::jitGeneratorError storeResultF32();

    /**
     * @brief Store results as float16
     */
    dlp::jit::jitGeneratorError storeResultF16();

    /**
     * @brief Store results as bfloat16
     */
    dlp::jit::jitGeneratorError storeResultBF16();

    // /**
    //  * @brief Generate saturation and packing for int8 output
    //  */
    // void generateInt32ToInt8Conversion(md_t mr, md_t nr_blocks);

    // =================================================================
    // SCALING OPERATIONS
    // =================================================================
    /**
     * @brief Apply alpha scaling to accumulator values
     * For u8s8: scale int32 accumulator before quantization/conversion
     */
    dlp::jit::jitGeneratorError scaleAlpha();

    /**
     * @brief Update C buffer pointers for next iteration
     * This function updates the C buffer pointers based on the current
     * iteration and any applied scaling factors.
     */
    void updateCBufferPointers();

    /**
     * @brief Apply beta scaling when accumulating with existing C values
     * For u8s8: scale existing C values before adding to computed results
     */
    dlp::jit::jitGeneratorError scaleBeta();

    // /**
    //  * @brief Apply quantization scaling for int8 output conversion
    //  * Handles per-tensor or per-channel quantization parameters
    //  */
    // dlp::jit::jitGeneratorError applyQuantizationScaling(md_t mr,
    //                                                      md_t nr_blocks);

    // =================================================================
    // POST-OPERATIONS AND ADVANCED FEATURES
    // =================================================================
    /**
     * @brief Generate post-operations sequence
     * Handle bias, scaling, activation functions, output conversion
     */
    dlp::jit::jitGeneratorError generatePostOps(utils::generatorParams& params);

    // =================================================================
    // CONSTANT DATA GENERATION
    // =================================================================
    /**
     * @brief Generate constant data tables used by the kernel
     * These are placed after the return instruction and accessed via
     * RIP-relative addressing
     */
    void generateConstantData();

    // =================================================================
    // NOTE: VNNI BUFFER HANDLING
    // =================================================================
    // Classic framework provides pre-packed VNNI buffers:
    // - B matrix: packed by dlp_packb_nr64_u8s8s32o32_row_major()
    // - A matrix: accessed with standard strides + VNNI broadcast
    // - All alignment requirements handled by classic packing
    // JIT generator only needs standard buffer addressing

    // =================================================================
    // INTERNAL HELPER METHODS
    // =================================================================
    /**
     * @brief Load B matrix values into registers
     */
    dlp::jit::jitGeneratorError loadBValues();

    /**
     * @brief Broadcast A and compute VNNI with B (follows F32
     * broadcastAFMAwithB pattern)
     */
    dlp::jit::jitGeneratorError broadcastAVNNIwithB(bool isVNNIrem);

    /**
     * @brief Unroll kernel computation loop
     */
    dlp::jit::jitGeneratorError kUnroll(int unroll, bool isVNNIrem);

    /**
     * @brief Move C pointer for next iteration
     */
    dlp::jit::jitGeneratorError moveCPtr();
};

using jitGemmGenerator_u8s8_avx512 =
    jitU8S8VNNI_GEMM<utils::kernelInstrType::avx512_zmm_32_reg>;

} // namespace amdzen::gen
