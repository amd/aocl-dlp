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

#ifndef AVX2_GENERATOR_HH
#define AVX2_GENERATOR_HH

#include <cstdint>
#include <vector>

#include "jit/jit_generator_base.hh"
#include "jit/xbyak/xbyak.h"
#include "jit/xbyak/xbyak_util.h"
#include "jit_generator_utils.hh"
#include "kernel_ops_handler.hh"
#include "kernels/kernel_base.hh"
#include "traits.hh"

namespace amdzen::avx2gen {

constexpr uint64_t JIT_KERNEL_SIZE = 4096;

typedef void (*jit_kernel)(dlp::kernels::gemmParams*);

/**
 * @brief AVX2 JIT GEMM kernel generator
 *
 * Generates optimized GEMM kernels using AVX2 instructions following the
 * 5-loop tiled algorithm with IR-loop implementation.
 *
 * Optimized Register allocation strategy:
 * - 2 YMM registers: A matrix broadcasting (alternating for optimal ILP)
 * - Variable YMM registers: B matrix loading (based on NR requirements)
 * - Remaining YMM registers: C accumulation (maximized for performance)
 *
 * The A register alternation (i % 2) provides optimal instruction-level
 * parallelism regardless of MR value, while freeing up registers for
 * additional C accumulation.
 */
class jitAVX2 : public Xbyak::CodeGenerator
{
  public:
    // Constructor that takes buffer and its size for JIT code dumping
    jitAVX2(void* buffer, size_t bufferSize);
    ~jitAVX2()                         = default;
    jitAVX2(const jitAVX2&)            = delete;
    jitAVX2& operator=(const jitAVX2&) = delete;
    jitAVX2(jitAVX2&&)                 = delete;
    jitAVX2& operator=(jitAVX2&&)      = delete;

    template<dlp::kernel_frame::kernelDatatype KDT>
    dlp::jit::jitGeneratorError generateKernel(utils::generatorParams& params);

  private:
    using Traits = amdzen::traits::ArchitectureTraits<
        utils::kernelInstrType::avx2_ymm_16_reg>;

    // Configuration and state
    int numRegs  = Traits::numRegs;
    int RegSize  = Traits::regSize;
    int RegBytes = Traits::regBytes;
    int aReg, bReg, bFullReg, bMaskReg, cReg;
    int aRegIdx, bRegIdx, cRegIdx, ymmMaskIdx;
    int MR, NR;

    Xbyak::Reg64 regTmpAptr, regBptr, regTmpCptr, regRsA, regCsA, regRsB,
        regRsC;
    Xbyak::Reg64 regMiter, regKIter;
    Xbyak::Reg64 regTmp1, regTmp2, regTmp3;
    Xbyak::Reg64 regCPtr, regAPtr;
    Xbyak::Reg64 stackPtr;

    Xbyak::Label m_loop;
    Xbyak::Label k_loop;
    Xbyak::Label k_rem_loop;
    Xbyak::Label label_store_result;

    bool useMask =
        false; // Flag to indicate if masked instructions are generated

    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);

    // Register initialization and utility methods
    void regInitYmm();

    // C buffer prefetching for performance optimization
    template<typename cType>
    void prefetchCBuffer();

    // Core kernel generation methods
    template<typename accumType>
    dlp::jit::jitGeneratorError allocateReg();

    template<typename aType, typename bType, typename cType>
    void initializeParameters(bool addIrLoop);

    template<typename aType, typename bType, typename cType, typename accumType>
    dlp::jit::jitGeneratorError generateIrLoop(utils::generatorParams& params);

    template<typename aType, typename bType, typename cType, typename accumType>
    dlp::jit::jitGeneratorError loadAValuesYmm(bool useMask = false);

    template<typename bType>
    dlp::jit::jitGeneratorError loadBValuesYmm();

    template<typename aType, typename bType, typename accumType>
    dlp::jit::jitGeneratorError BroadcastAandComputeFMAwithYmm();

    template<typename aType, typename bType, typename cType, typename accumType>
    dlp::jit::jitGeneratorError kernelUnrollYmm(int loopCount);

    template<typename accumType, typename cType>
    dlp::jit::jitGeneratorError storeResult();

    template<typename accumType>
    dlp::jit::jitGeneratorError scaleAlpha();

    template<typename accumType, typename cType>
    dlp::jit::jitGeneratorError scaleBeta();
#if 0 // TODO: Placeholder for beta = 1.0
    template<typename accumType, typename cType>
    dlp::jit::jitGeneratorError addCBufferBetaOne();
#endif
    void moveCPtr();
};

} // namespace amdzen::avx2gen

#endif // AVX2_GENERATOR_HH
