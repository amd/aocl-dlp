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

#ifndef F32_GEMV_HH
#define F32_GEMV_HH

#include "jit/jit_generator_base.hh"
#include "jit/xbyak/xbyak.h"
#include "jit/xbyak/xbyak_util.h"
#include "jit_generator_utils.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "kernel_ops_handler.hh"
#include "traits.hh"

namespace amdzen::codegen {

template<utils::kernelInstrType KType>
class jitF32GEMVM1 : public Xbyak::CodeGenerator
{
  private:
    int                              RegBytes;  // Size of ZMM register in bytes
    int                              numRegs;   // Number of ZMM registers
    int                              simdWidth; // SIMD width
    int                              NR;
    int                              KC;
    int                              K_SUB_ITER;
    AOCL_MEMORY_TAG                  mtag_b;
    dlp::kernel_frame::storageFormat yFormat;
    dlp::kernel_frame::scalingType   alphaScalingType;
    dlp::kernel_frame::scalingType   betaScalingType;

    int xReg;
    int bReg;
    int yReg;
    int maskReg;
    int accumReg;
    int xBaseIdx;
    int bBaseIdx;
    int yBaseIdx;
    int accumBaseIdx;
    int maskBaseIdx;

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
    void initializeParameters(const utils::gemvM1GeneratorParams&);

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

    dlp::jit::jitGeneratorError storeYValuesFringe();

    dlp::jit::jitGeneratorError storeYValues(bool = false);

    //------------------------------------------------

  public:
    jitF32GEMVM1(void* buffer = nullptr, size_t size = 0);
    ~jitF32GEMVM1()                         = default;
    jitF32GEMVM1(jitF32GEMVM1&)             = delete;
    jitF32GEMVM1(jitF32GEMVM1&&)            = delete;
    jitF32GEMVM1& operator=(jitF32GEMVM1&)  = delete;
    jitF32GEMVM1& operator=(jitF32GEMVM1&&) = delete;

    // Main kernel generation interface
    dlp::jit::jitGeneratorError generateKernel(
        const utils::gemvM1GeneratorParams& params);

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
