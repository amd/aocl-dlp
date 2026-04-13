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

#ifndef F32_GEMV_GENERATOR_HH
#define F32_GEMV_GENERATOR_HH

#include "jit/jit_generator_base.hh"
#include "jit_generator_utils.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "kernel_ops_handler.hh"
#include "traits.hh"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace amdzen::codegen {

template<utils::kernelInstrType KType>
class jitF32GEMVN1 : public Xbyak::CodeGenerator
{
  private:
    int RegBytes;  // Size of ZMM register in bytes
    int numRegs;   // Number of ZMM registers
    int simdWidth; // SIMD width
    int c_downscale;
    int MR;          // Number of rows to process at once
    int M_LEFT;      // M-dimension left over elements
    int LOAD_UNROLL; // Number of regsiters to be used explicitly for loading
                     // from A
    dlp::kernel_frame::storageFormat yFormat; // Storage format of C matrix
    dlp::kernel_frame::scalingType alphaScalingType; // Type of kernel operation
    dlp::kernel_frame::scalingType betaScalingType;  // Type of beta scaling

    // Register counts and indices
    int yReg;     // Number of registers for loading/storing from Y
    int aReg;     // Number of registers for matrix A
    int xReg;     // Number of registers for vector x
    int accumReg; // Number of registers for accumulation (partial dot products)
    int tmpReg;   // Number of registers for temporary use
    int cvtReg;   // Number of registers to store converted output from f32 to
                  // bf16
    int maskReg;  // Number of registers for mask
    int yBaseIdx; // Starting index for accumulation registers (from end)
    int aBaseIdx; // Starting index for A registers (from beginning)
    int xBaseIdx; // Starting index for x registers (after A registers)
    int cvtBaseIdx;   // Starting index for conversion registers
    int accumBaseIdx; // Starting index for accumulation registers (after A and
                      // x
    int tmpBaseIdx;   // Starting index for temporary registers (after A and x
    int maskBaseIdx;  // Starting index for mask registers

    // Matrix/Vector pointers and strides
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

    Xbyak::Opmask mask_regs[utils::NUM_USABLE_MASKS];

    // Labels for code sections
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
    using Traits      = traits::ArchitectureTraits<KType>;
    using RegType     = typename Traits::RegType;
    using halfRegType = typename Traits::halfRegType;

    // Stack frame management
    void initializeStackFrame(Xbyak::util::StackFrame&);

    // Register initialization
    void regInit(int, int);

    // Implementation utilities
    dlp::jit::jitGeneratorError allocateRegisters();

    void initializeParameters(utils::gemvN1GeneratorParams&);

    // Core computation functions
    dlp::jit::jitGeneratorError loadAValues(int, bool = false);

    dlp::jit::jitGeneratorError loadXValues(bool = false);

    dlp::jit::jitGeneratorError loadYValues(int);

    dlp::jit::jitGeneratorError computeFMA(int, int);

    dlp::jit::jitGeneratorError computeLoadFMA(int, bool = false);

    dlp::jit::jitGeneratorError reduceToXmm(int, int, int);

    dlp::jit::jitGeneratorError reduceAccumulation(int);

    dlp::jit::jitGeneratorError scaleAccumulationWithAlpha(int);

    dlp::jit::jitGeneratorError scaleYWithBetaColStored(int, bool = false);

    dlp::jit::jitGeneratorError scaleYWithBetaRowStored(int, bool = false);

    dlp::jit::jitGeneratorError scaleYWithBeta(int);

    dlp::jit::jitGeneratorError convertF32toBF16(int, int, int);

    dlp::jit::jitGeneratorError storeYValuesColStored(int);

    dlp::jit::jitGeneratorError storeYValuesRowStored(int);

    dlp::jit::jitGeneratorError storeYValues(int);

    dlp::jit::jitGeneratorError processMRBlock(int, bool = false);

    dlp::jit::jitGeneratorError loadMasks();

  public:
    // Enforcing RAII, disallowing copy/move operations
    jitF32GEMVN1(size_t maxSize);
    ~jitF32GEMVN1()                         = default;
    jitF32GEMVN1(jitF32GEMVN1&)             = delete;
    jitF32GEMVN1& operator=(jitF32GEMVN1&)  = delete;
    jitF32GEMVN1(jitF32GEMVN1&&)            = delete;
    jitF32GEMVN1& operator=(jitF32GEMVN1&&) = delete;

    // Main kernel generation interface
    dlp::jit::jitGeneratorError generateKernel(
        utils::gemvN1GeneratorParams& params);

    // Get the generated kernel function pointer
    // This class will also contain the pointer type to the JIT kernel
    // That way, this typedef is available only when an instance of this class
    // is created.
    // utils::jit_gemv_n1_kernel getKernel()
    // {
    //     return getCode<jit_gemv_n1_kernel>();
    // }
};

template<utils::kernelInstrType KType>
class jitF32GEMVM1 : public Xbyak::CodeGenerator
{
  private:
    int                              RegBytes;  // Size of ZMM register in bytes
    int                              numRegs;   // Number of ZMM registers
    int                              simdWidth; // SIMD width
    int                              NR;
    int                              N_LEFT;
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

    Xbyak::Opmask mask_regs[utils::NUM_USABLE_MASKS];

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
    Xbyak::Label label_accumulate_result;
    Xbyak::Label label_store_result;

    // Defining the architecture specific type aliases
    using Traits      = traits::ArchitectureTraits<KType>;
    using RegType     = typename Traits::RegType;
    using halfRegType = typename Traits::halfRegType;

    //------------------------------------------------
    // ISA agnostic methods
    // Initializing the stack frame
    void initializeStackFrame(Xbyak::util::StackFrame&);

    // Initializing the parameters
    void initializeParameters(utils::gemvM1GeneratorParams&);

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

    dlp::jit::jitGeneratorError compute1xnfringe();

    dlp::jit::jitGeneratorError compute1xNR(bool = false);

    dlp::jit::jitGeneratorError loopKSubIter(bool = false, bool = false);

    dlp::jit::jitGeneratorError finalAccumulate();

    dlp::jit::jitGeneratorError scaleWithAlpha();

    dlp::jit::jitGeneratorError scaleYWithBetaFringe(bool = false);

    dlp::jit::jitGeneratorError scaleYWithBeta(bool = false);

    dlp::jit::jitGeneratorError convertF32toBF16(int, int, int);

    dlp::jit::jitGeneratorError storeYValuesFringe();

    dlp::jit::jitGeneratorError storeYValues(bool = false);

    //------------------------------------------------

  public:
    jitF32GEMVM1(size_t maxSize);
    ~jitF32GEMVM1()                         = default;
    jitF32GEMVM1(jitF32GEMVM1&)             = delete;
    jitF32GEMVM1(jitF32GEMVM1&&)            = delete;
    jitF32GEMVM1& operator=(jitF32GEMVM1&)  = delete;
    jitF32GEMVM1& operator=(jitF32GEMVM1&&) = delete;

    // Main kernel generation interface
    dlp::jit::jitGeneratorError generateKernel(
        utils::gemvM1GeneratorParams& params);

    // Get the generated kernel function pointer
    // utils::jit_gemv_m1_kernel getKernel() {
    //     return getCode<utils::jit_gemv_m1_kernel>();
    // }
};

} // namespace amdzen::codegen

#endif // F32_GEMV_HH

/*
    Any JIT generator :
    ISA agnostic code(loop structuring, pointer arithmetic, etc.)
    ISA specific code(reigster aliasing difference)
    ISA specific code(instruction difference)

    Case 1 does not require regType alias and templatization on regType
    Case 2 does not require templatization, but regType alias is required
    Case 3 requires templatization on regType and regType alias

    Define methods accordingly, and implement a unified JIT generator for all
    ISA. It would be only API specific !!
*/
