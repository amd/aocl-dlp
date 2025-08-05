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

#ifndef AVX512_GENERATOR_HH
#define AVX512_GENERATOR_HH

#include "jit/jit_generator_base.hh"
#include "jit/xbyak/xbyak.h"
#include "jit/xbyak/xbyak_util.h"
#include "kernel_op_handler.hh"
#include "kernels/kernel_base.hh"

#include <cstdint>
#include <vector>

namespace avx512gen::generator {

constexpr uint64_t JIT_KERNEL_SIZE = 8 * 4096;

typedef void (*jit_kernel)(dlp::kernels::gemmParams*);

struct generatorParams
{
    int MR; // This MR can be of either main kernel or fringe kernel
    int NR; // This NR can be of either main kernel or fringe kernel
    int K_UNROLL;
    // This will be used to generate NR + " < nElemsPerReg" kernels,
    // where NR is a multiple of nElemsPerReg including "0".
    bool useMask;
    bool mLoop;        // This will be set to true only for the main kernel
    bool is_beta_zero; // skip beta scaling if beta is 0
    bool is_alpha_one; // skip alpha scaling if alpha is 1
    std::vector<kernelOpsMetaData> kernelOps;

    generatorParams(md_t _MR,
                    md_t _NR,
                    int  _K_UNROLL,
                    bool _useMask,
                    bool _mLoop,
                    bool _is_beta_zero = false,
                    bool _is_alpha_one = false)
        : MR(_MR)
        , NR(_NR)
        , K_UNROLL(_K_UNROLL)
        , useMask(_useMask)
        , mLoop(_mLoop)
        , is_beta_zero(_is_beta_zero)
        , is_alpha_one(_is_alpha_one)
    {
    }

    generatorParams& operator=(const generatorParams& other)
    {
        MR           = other.MR;
        NR           = other.NR;
        K_UNROLL     = other.K_UNROLL;
        useMask      = other.useMask;
        mLoop        = other.mLoop;
        is_beta_zero = other.is_beta_zero;
        is_alpha_one = other.is_alpha_one;
        return *this;
    }

    generatorParams& operator=(generatorParams&& other)
    {
        *this = other;
        return *this;
    }

    ~generatorParams() = default;
};

class jitAVX512 : public Xbyak::CodeGenerator
{
  public:
    // Constructor that takes buffer and its size for JIT code dumping
    jitAVX512(void* buffer, size_t bufferSize);

    template<dlp::kernel_frame::kernelDatatype KDT>
    dlp::jit::jitGeneratorError generateKernel(generatorParams& params);

  private:
    // Configuration and state
    int numRegs  = 32;
    int RegSize  = 512;
    int RegBytes = RegSize / 8;
    int aReg, bReg, bFullReg, bMaskReg, cReg;
    int aRegIdx, bRegIdx, cRegIdx;
    int MR, NR;

    // Register allocations
    Xbyak::Reg64 regTmpAptr, regBptr, regTmpCptr, regRsA, regCsA, regRsB,
        regRsC, regKIter;
    Xbyak::Reg64 regMiter;
    Xbyak::Reg64 regTmp1, regTmp2, regTmp3;
    Xbyak::Reg64 regCPtr, regAPtr;
    Xbyak::Reg64 stackPtr;

    Xbyak::Reg64 regkernelOpsList, regkernelOpsAttr;

    Xbyak::Label label_store_result;

    bool useMask =
        false; // Flag to indicate if masked instructions are generated

    // Core kernel generation methods
    template<typename accumType>
    dlp::jit::jitGeneratorError allocateReg();

    template<typename aType, typename bType, typename cType>
    void initializeParameters(bool addIrLoop);

    template<typename aType, typename bType, typename cType, typename accumType>
    dlp::jit::jitGeneratorError generateIrLoop(generatorParams& params);

    // Memory operations
    template<typename bType>
    dlp::jit::jitGeneratorError loadBValuesZmm();

    template<typename aType, typename bType, typename accumType>
    dlp::jit::jitGeneratorError BroadcastAFMAwithBZmm();

    template<typename aType, typename bType, typename accumType>
    dlp::jit::jitGeneratorError kernelUnrollZmm(int unroll);

    template<typename accumType, typename cType>
    dlp::jit::jitGeneratorError storeResult();

    // Setup and initialization
    void initializeStackFrame(Xbyak::util::StackFrame& stackFrame);
    void regInitZmm();
    void moveCPtr();

    // Scaling operations
    template<typename accumType>
    dlp::jit::jitGeneratorError scaleAlpha();

    template<typename accumType, typename cType>
    dlp::jit::jitGeneratorError scaleBeta();

    template<typename accumType>
    dlp::jit::jitGeneratorError cvtAccToFloat();
};

class jitAVX512FP32 : public dlp::jit::jitGeneratorBase
{

    std::vector<dlp::kernel_frame::kernelDatatype> mKernelDatatypes;
    std::vector<dlp::cpu_utils::isaFeature>        mIsaFeaturesRequired;

  public:
    // jitAVX512 base;
    int                MR, NR;
    int                numMRVariants, numNRVariants;
    int                numKernelVariants;
    int                K_UNROLL;
    std::vector<void*> kernelCodeBlocks;

    jitAVX512FP32()
        : mKernelDatatypes({ dlp::kernel_frame::kernelDatatype::f32f32f32of32 })
        // TODO: Hardcoded for now, need to make it dynamic
        , mIsaFeaturesRequired{ dlp::cpu_utils::isaFeature::avx512f,
                                dlp::cpu_utils::isaFeature::avx512bw,
                                dlp::cpu_utils::isaFeature::avx512dq,
                                dlp::cpu_utils::isaFeature::avx512vl }
    {
    }

    ~jitAVX512FP32();

    dlp::jit::jitGeneratorError generateAllKernels(
        const dlp::kernel_frame::kernelInfo& kI);

    dlp::jit::jitGeneratorError operator()(
        const dlp::kernel_frame::kernelInfo& kI) override
    {
        return generateAllKernels(kI);
    }

    std::vector<dlp::kernel_frame::kernelDatatype>& getKernelDatatypes()
        override
    {
        return mKernelDatatypes;
    }

    std::vector<dlp::cpu_utils::isaFeature>& getIsaFeaturesRequired() override
    {
        return mIsaFeaturesRequired;
    }

    dlp::kernels::kernelError executeKernel(
        dlp::kernels::kernelParams* _params);

    std::unique_ptr<jitGeneratorBase> clone() override
    {
        return std::make_unique<jitAVX512FP32>();
    }
};

} // namespace avx512gen::generator

#endif // X86_JIT_FRAMEWORK_HPP
