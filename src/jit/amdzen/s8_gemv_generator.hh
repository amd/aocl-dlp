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
#include "jit_generator_utils.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "kernel_ops_handler.hh"
#include "traits.hh"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace amdzen::gen {

template<utils::kernelInstrType KType>
class jitGEMVS8N1 : public Xbyak::CodeGenerator
{
  public:
    // Constructor that specifies the maximum size of generated JIT code.
    // Buffer allocation and AutoGrow behavior are managed internally by Xbyak.
    jitGEMVS8N1(size_t maxSize);
    ~jitGEMVS8N1()                        = default;
    jitGEMVS8N1(jitGEMVS8N1&)             = delete;
    jitGEMVS8N1& operator=(jitGEMVS8N1&)  = delete;
    jitGEMVS8N1(jitGEMVS8N1&&)            = delete;
    jitGEMVS8N1& operator=(jitGEMVS8N1&&) = delete;

    dlp::jit::jitGeneratorError generateKernel(
        utils::gemvN1GeneratorParams& params);

  private:
    using Traits  = amdzen::traits::ArchitectureTraits<KType>;
    using RegType = typename Traits::RegType;

    // Configuration and state
    int numRegs  = Traits::numRegs;
    int RegSize  = Traits::regSize;
    int RegBytes = Traits::regBytes;
    int vnniWidth; // VNNI width

    int MR;          // Number of rows to process at once
    int M_LEFT;      // M-dimension left over elements
    int LOAD_UNROLL; // Number of registers to be used explicitly for loading
                     // from A

    dlp::kernel_frame::storageFormat yFormat; // Storage format of C matrix
    dlp::kernel_frame::scalingType alphaScalingType; // Type of kernel operation
    dlp::kernel_frame::scalingType betaScalingType;  // Type of beta scaling

    int yReg;     // Number of registers for loading/storing from Y
    int aReg;     // Number of registers for matrix A
    int xReg;     // Number of registers for vector x
    int accumReg; // Number of registers for accumulation (partial dot products)
    int tmpReg;   // Number of registers for temporary use
    int maskReg;  // Number of registers for mask

    int yBaseIdx;     // Starting index for accumulation registers (from end)
    int aBaseIdx;     // Starting index for A registers (from beginning)
    int xBaseIdx;     // Starting index for x registers (after A registers)
    int accumBaseIdx; // Starting index for accumulation registers (after A and
                      // x)
    int tmpBaseIdx;   // Starting index for temporary registers (after A and x)
    int vec128Idx;    // Starting index for mask registers

    int vec128Reg; // Reserving one register for converting int8 to uint8.
    int c_downscale;

    // Register allocations
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

    Xbyak::Reg64 regkernelOpsList, regkernelOpsAttr;

    Xbyak::Label label_store_result;

    bool useMask = false; // Flag to indicate generation of masked instructions.

    // Setup and initialization
    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);

    dlp::jit::jitGeneratorError allocateRegisters();

    void initializeParameters(utils::gemvN1GeneratorParams& params);

    // Register initialization
    void regInit(int, int);

    // Memory operations
    dlp::jit::jitGeneratorError loadXValues(bool = false);

    dlp::jit::jitGeneratorError loadAValues(int, bool = false);

    dlp::jit::jitGeneratorError computeVNNI(int, int);

    dlp::jit::jitGeneratorError processMRBlock(int, bool = false);

    dlp::jit::jitGeneratorError reduceToXmm(int, int, int);

    dlp::jit::jitGeneratorError reduceAccumulation(int);

    // Compensating for conversion of A from int8 to uint8 for the VNNI
    // instruction.
    dlp::jit::jitGeneratorError conversionCompensation(int);

    dlp::jit::jitGeneratorError updateCBufferPointers();

    // Scaling operations
    dlp::jit::jitGeneratorError scaleAccByAlpha(int);

    dlp::jit::jitGeneratorError scaleYWithBetaColStored(int, bool = false);

    dlp::jit::jitGeneratorError scaleYWithBetaRowStored(int, bool = false);

    dlp::jit::jitGeneratorError scaleYByBeta(int mSize);

    dlp::jit::jitGeneratorError storeYColStored(int, bool = false);

    dlp::jit::jitGeneratorError storeYRowStored(int, bool = false);

    dlp::jit::jitGeneratorError storeY(int, bool = false);
};

template<utils::kernelInstrType KType>
class jitGEMVS8M1 : public Xbyak::CodeGenerator
{
  public:
    // Constructor that specifies the maximum size of generated JIT code.
    // Buffer allocation and AutoGrow behavior are managed internally by Xbyak.
    jitGEMVS8M1(size_t maxSize);
    ~jitGEMVS8M1()                        = default;
    jitGEMVS8M1(jitGEMVS8M1&)             = delete;
    jitGEMVS8M1& operator=(jitGEMVS8M1&)  = delete;
    jitGEMVS8M1(jitGEMVS8M1&&)            = delete;
    jitGEMVS8M1& operator=(jitGEMVS8M1&&) = delete;

    dlp::jit::jitGeneratorError generateKernel(
        utils::gemvM1GeneratorParams& params);

  private:
    using Traits  = amdzen::traits::ArchitectureTraits<KType>;
    using RegType = typename Traits::RegType;

    // Configuration and state
    int numRegs  = Traits::numRegs;
    int RegSize  = Traits::regSize;
    int RegBytes = Traits::regBytes;
    int vnniWidth; // VNNI width

    int NR;          // Number of columns to process at once
    int N_LEFT;      // N-dimension left over elements
    int N_LEFT_16;   // N-dimension left over elements when < 16
    int N_LEFT_LT16; // N-dimension left over elements when < 16
    int KC;          // Blocking factor for K-dimension
    int K_SUB_ITER;  // Number of K iterations per unrolled segment
    int c_downscale;

    dlp::kernel_frame::storageFormat yFormat; // Storage format of C matrix
    dlp::kernel_frame::scalingType alphaScalingType; // Type of kernel operation
    dlp::kernel_frame::scalingType betaScalingType;  // Type of beta scaling

    int xReg;     // Number of registers for vector x
    int bReg;     // Number of registers for matrix B
    int accumReg; // Number of registers for accumulation
    int yReg;     // Number of registers for loading/storing Y
    int maskReg;  // Number of registers for mask

    int vec128Reg = 1; // Reserving one register for converting int8 to uint8.
    int xBaseIdx;      // Starting index for X registers
    int bBaseIdx;      // Starting index for B registers
    int accumBaseIdx;  // Starting index for accumulation registers
    int yBaseIdx;      // Starting index for Y registers
    int maskBaseIdx;   // Starting index for mask registers
    int vec128BaseIdx; // Starting index for 128-bit vector register

    // Register allocations
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

    // Labels for code sections to prevent "duplicate label" errors
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
    Xbyak::Label label_skip_conversion_comp;    // End of k-dimension remainder

    // Setup and initialization
    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);

    dlp::jit::jitGeneratorError allocateRegisters();

    void initializeParameters(utils::gemvM1GeneratorParams& params);

    // Register initialization
    void regInit(int baseIdx, int numRegs);

    // Memory operations
    dlp::jit::jitGeneratorError offsetBPtr(int temp);

    dlp::jit::jitGeneratorError computeKxNR(bool nMask);

    dlp::jit::jitGeneratorError computeKxNfringe();

    dlp::jit::jitGeneratorError compute1xNR(bool nMask, bool isLastKGroup);

    dlp::jit::jitGeneratorError compute1xNfringe(bool isLastKGroup);

    dlp::jit::jitGeneratorError kLoop(bool kfringe = false,
                                      bool nfringe = false);

    dlp::jit::jitGeneratorError accumulateKSubIters();

    void updateYBufferPointers();

    dlp::jit::jitGeneratorError storeY(bool nMask, bool hasPostOps = false);

    // Scaling operations
    dlp::jit::jitGeneratorError scaleWithAlpha();

    dlp::jit::jitGeneratorError scaleYWithBeta(bool nMask);

    // Compensating for conversion of A from int8 to uint8 for the VNNI
    // instruction.
    dlp::jit::jitGeneratorError conversionCompensation(bool isFringe = false);
};

} // namespace amdzen::gen
