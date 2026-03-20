/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 notice,
 *    this list of conditions and the following disclaimer in the
 documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 IS"
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

#ifndef BF16_GEMV_GENERATOR_HH
#define BF16_GEMV_GENERATOR_HH

#include "jit/jit_generator_base.hh"
#include "jit_generator_utils.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "kernel_ops_handler.hh"
#include "traits.hh"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

typedef int16_t bfloat16;

namespace amdzen::codegen {

template<utils::kernelInstrType KType>
class jitBF16GEMVN1 : public Xbyak::CodeGenerator
{
  private:
    int                              RegBytes; // Size of ZMM register in bytes
    int                              numRegs;  // Number of ZMM registers
    int                              simdWidthF32;  // SIMD width for F32
    int                              simdWidthBF16; // SIMD width for BF16
    int                              MR; // Number of rows to process at once
    int                              M_LEFT; // M-dimension left over elements
    int                              c_downscale; // Downscale factor for C
    dlp::kernel_frame::storageFormat yFormat;     // Storage format of C matrix
    dlp::kernel_frame::scalingType   alphaScalingType;
    dlp::kernel_frame::scalingType   betaScalingType;

    int yReg;     // Number of registers for loading/storing from Y
    int aReg;     // Number of registers for matrix A
    int xReg;     // Number of registers for vector x
    int accumReg; // Number of registers for accumulation (partial dot products)
    int tmpReg;   // Number of registers for temporary use
    int yBaseIdx; // Starting index for Y registers
    int aBaseIdx; // Starting index for A registers
    int xBaseIdx; // Starting index for x registers
    int accumBaseIdx; // Starting index for accumulation registers
    int tmpBaseIdx;   // Starting index for temporary registers
    int maskBaseIdx;  // Starting index for mask registers

    Xbyak::Reg64 stackPtr;            // Stack frame pointer
    Xbyak::Reg64 regAptr, regTmpAptr; // Pointer to matrix A and its temp
    Xbyak::Reg64 regXptr;             // Pointer to vector x
    Xbyak::Reg64 regYptr, regTmpYptr; // Pointer to vector y and its temp
    Xbyak::Reg64 regRsA;              // Row stride for A
    Xbyak::Reg64 regCsA;              // Column stride for A
    Xbyak::Reg64 regRsC;              // Row stride for C
    Xbyak::Reg64 regMIter;            // M-loop iterator
    Xbyak::Reg64 regKIter;            // K-loop iterator
    Xbyak::Reg64 regTmp1;             // General purpose temporary register 1
    Xbyak::Reg64 regTmp2;             // General purpose temporary register 2
    Xbyak::Reg64 regTmp3;             // General purpose temporary register 3

    // Add mask register array (for AVX512)
    static constexpr int NUM_USABLE_MASKS = 7;        // k1-k7 available
    static constexpr int MASK_START_IDX   = 1;        // Start from k1
    Xbyak::Opmask        mask_regs[NUM_USABLE_MASKS]; // Array of usable masks

    Xbyak::Label label_m_loop_start;            // Main m-dimension loop
    Xbyak::Label label_m_loop_end;              // End of m-dimension loop
    Xbyak::Label label_m_loop_k_loop_start;     // Main k-dimension loop
    Xbyak::Label label_m_loop_k_loop_end;       // End of k-dimension loop
    Xbyak::Label label_m_fringe_k_loop_start;   // Main k-dimension loop
    Xbyak::Label label_m_fringe_k_loop_end;     // End of k-dimension loop
    Xbyak::Label label_m_fringe_start;          // Handle m-dimension remainder
    Xbyak::Label label_m_fringe_end;            // End of m-dimension remainder
    Xbyak::Label label_m_loop_k_fringe_start;   // Handle k-dimension remainder
    Xbyak::Label label_m_loop_k_fringe_end;     // End of k-dimension remainder
    Xbyak::Label label_m_fringe_k_fringe_start; // Handle k-dimension remainder
    Xbyak::Label label_m_fringe_k_fringe_end;   // End of k-dimension remainder
    Xbyak::Label label_reduce_start;            // Reduction operations
    Xbyak::Label label_store_result;            // Store final results

    // Defining the architecture specific type aliases
    using Traits  = traits::ArchitectureTraits<KType>;
    using RegType = typename Traits::RegType;

    //------------------------------------------------
    // ISA agnostic methods
    // Initializing the stack frame
    void initializeStackFrame(Xbyak::util::StackFrame&);

    // Initializing the parameters
    void initializeParameters(const utils::gemvN1GeneratorParams&);

    // Allocating the registers
    dlp::jit::jitGeneratorError allocateRegisters();

    //------------------------------------------------
    // ISA specific methods
    // Register initialization
    void regInit(int baseIdx, int numRegs);

    dlp::jit::jitGeneratorError loadMasks();

    dlp::jit::jitGeneratorError loadXValues(bool = false);

    dlp::jit::jitGeneratorError processMRBlock(int, bool = false);

    dlp::jit::jitGeneratorError loadAValues(int, bool = false);

    dlp::jit::jitGeneratorError computeDP(int, int);

    dlp::jit::jitGeneratorError reduceAccumulation(int);

    dlp::jit::jitGeneratorError scaleAccumulationWithAlpha(int);

    dlp::jit::jitGeneratorError scaleYWithBeta(int);

    dlp::jit::jitGeneratorError scaleYWithBetaRowStored(int, bool);

    dlp::jit::jitGeneratorError scaleYWithBetaColStored(int, bool);

    dlp::jit::jitGeneratorError storeYValues(int);

    dlp::jit::jitGeneratorError storeYValuesColStored(int);

    dlp::jit::jitGeneratorError storeYValuesRowStored(int);

    dlp::jit::jitGeneratorError reduceToXmm(int, int, int);

  public:
    jitBF16GEMVN1(size_t maxSize);
    ~jitBF16GEMVN1()                          = default;
    jitBF16GEMVN1(jitBF16GEMVN1&)             = delete;
    jitBF16GEMVN1(jitBF16GEMVN1&&)            = delete;
    jitBF16GEMVN1& operator=(jitBF16GEMVN1&)  = delete;
    jitBF16GEMVN1& operator=(jitBF16GEMVN1&&) = delete;

    // Main kernel generation interface
    dlp::jit::jitGeneratorError generateKernel(
        utils::gemvN1GeneratorParams& params);

    // Get the generated kernel function pointer
    // utils::jit_gemv_n1_kernel getKernel() {
    //     return getCode<utils::jit_gemv_n1_kernel>();
    // }
};

template<utils::kernelInstrType KType>
class jitBF16GEMVM1 : public Xbyak::CodeGenerator
{
  private:
    int                              RegBytes;  // Size of ZMM register in bytes
    int                              numRegs;   // Number of ZMM registers
    int                              simdWidth; // SIMD width for f32
    int                              NR;
    int                              N_LEFT;
    int                              N_LEFT_16;
    int                              N_LEFT_LT16;
    int                              RS_B_N_LEFT_16;
    int                              RS_B_N_LEFT_LT16;
    int                              KC;
    int                              K_SUB_ITER;
    int                              c_downscale;
    AOCL_DLP_MEMORY_TAG              mtag_b;
    dlp::kernel_frame::storageFormat yFormat;
    dlp::kernel_frame::scalingType   alphaScalingType;
    dlp::kernel_frame::scalingType   betaScalingType;

    int xReg;
    int bReg;
    int yReg;
    int maskReg;
    int accumReg;
    int tmpReg;
    int xBaseIdx;
    int bBaseIdx;
    int yBaseIdx;
    int accumBaseIdx;
    int maskBaseIdx;
    int tmpBaseIdx;

    Xbyak::Reg64 stackPtr;
    Xbyak::Reg64 regXptr;
    Xbyak::Reg64 regBptr;
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

    // Add mask register array (for AVX512)
    static constexpr int NUM_USABLE_MASKS = 7;        // k1-k7 available
    static constexpr int MASK_START_IDX   = 1;        // Start from k1
    Xbyak::Opmask        mask_regs[NUM_USABLE_MASKS]; // Array of usable masks

    Xbyak::Label label_n_loop_start;
    Xbyak::Label label_n_loop_end;
    Xbyak::Label label_n_loop_k_loop_start;
    Xbyak::Label label_n_loop_k_loop_end;
    Xbyak::Label label_n_loop_k_fringe_start;
    Xbyak::Label label_n_loop_k_fringe_end;
    Xbyak::Label label_n_fringe_start;
    Xbyak::Label label_n_fringe_end;
    Xbyak::Label label_n_fringe_k_loop_start;
    Xbyak::Label label_n_fringe_k_loop_end;
    Xbyak::Label label_n_fringe_k_fringe_start;
    Xbyak::Label label_n_fringe_k_fringe_end;

    Xbyak::Label label_n_fringe_main_start;
    Xbyak::Label label_n_fringe_main_end;

    Xbyak::Label label_n_fringe_left_start;
    Xbyak::Label label_n_fringe_left_end;

    Xbyak::Label label_n_fringe_main_k_loop_start;
    Xbyak::Label label_n_fringe_main_k_loop_end;
    Xbyak::Label label_n_fringe_main_k_fringe_start;
    Xbyak::Label label_n_fringe_main_k_fringe_end;

    Xbyak::Label label_n_fringe_left_k_loop_start;
    Xbyak::Label label_n_fringe_left_k_loop_end;
    Xbyak::Label label_n_fringe_left_k_fringe_start;
    Xbyak::Label label_n_fringe_left_k_fringe_end;

    Xbyak::Label label_accumulate_result;
    Xbyak::Label label_store_result;

    // Defining the architecture specific type aliases
    using Traits  = traits::ArchitectureTraits<KType>;
    using RegType = typename Traits::RegType;

    //------------------------------------------------
    // ISA agnostic methods
    // Initializing the stack frame
    void initializeStackFrame(Xbyak::util::StackFrame&);

    // Initializing the parameters
    void initializeParameters(utils::gemvM1GeneratorParams&);

    dlp::jit::jitGeneratorError checkValidBF16GemvM1NLeftParams(
        const utils::gemvM1GeneratorParams& params);
    // Allocating the registers
    dlp::jit::jitGeneratorError allocateRegisters();

    //------------------------------------------------
    // ISA specific methods
    // Register initialization
    void regInit(int baseIdx, int numRegs);

    // Core computation functions

    dlp::jit::jitGeneratorError maskLoadB(int, int);

    dlp::jit::jitGeneratorError offsetBPtr(int);

    dlp::jit::jitGeneratorError loadMasks();

    dlp::jit::jitGeneratorError updateMask(int, int);

    dlp::jit::jitGeneratorError restoreMask(int, int);

    dlp::jit::jitGeneratorError computeKxnfringe();

    dlp::jit::jitGeneratorError computeKxNR(bool = false);

    dlp::jit::jitGeneratorError compute1xnfringe(bool = false);

    dlp::jit::jitGeneratorError compute1xNR(bool = false, bool = false);

    dlp::jit::jitGeneratorError loopKSubIter(bool = false, bool = false);

    dlp::jit::jitGeneratorError finalAccumulate();

    dlp::jit::jitGeneratorError scaleWithAlpha();

    dlp::jit::jitGeneratorError scaleYWithBetaFringe(bool = false);

    dlp::jit::jitGeneratorError scaleYWithBeta(int);

    dlp::jit::jitGeneratorError storeYValuesFringe();

    dlp::jit::jitGeneratorError storeYValues(int n_size);

    //------------------------------------------------

  public:
    jitBF16GEMVM1(size_t maxSize);
    ~jitBF16GEMVM1()                          = default;
    jitBF16GEMVM1(jitBF16GEMVM1&)             = delete;
    jitBF16GEMVM1(jitBF16GEMVM1&&)            = delete;
    jitBF16GEMVM1& operator=(jitBF16GEMVM1&)  = delete;
    jitBF16GEMVM1& operator=(jitBF16GEMVM1&&) = delete;

    // Main kernel generation interface
    dlp::jit::jitGeneratorError generateKernel(
        utils::gemvM1GeneratorParams& params);

    // Get the generated kernel function pointer
    // utils::jit_gemv_m1_kernel getKernel() {
    //     return getCode<utils::jit_gemv_m1_kernel>();
    // }
};

} // namespace amdzen::codegen

#endif // BF16_GEMV_GENERATOR_HH

/*
    BF16 GEMV N1 Generator:
    Handles BF16 matrix-vector multiplication where N=1 (single column vector)

    The N1 case means we're computing: y = A * x where x is a single column
   vector

    Key characteristics:
    1. Input matrix A and vector x are BF16, output y is F32
    2. SIMD width calculations account for BF16 size (2 bytes per element)
    3. Uses BF16 dot product instructions (e.g., vdpbf16ps for AVX512)
    4. M-dimension loop for processing multiple rows
    5. K-dimension reduction within each row

    Supported datatypes:
    - bf16bf16f32of32: BF16 inputs, F32 computation, F32 output
*/
