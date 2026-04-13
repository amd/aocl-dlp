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

#include <memory>

#include "jit/jit_generator_base.hh"
#include "jit_generator_utils.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "kernels/kernel_base.hh"
#include "x86_kernel_ops_generator.hh"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace amdzen::gen {

/**
 * @brief Templated facade for post-op code generation.
 *
 * Directly owns kernelOpsGeneratorX86<KType>, eliminating virtual dispatch.
 * GEMM generators (already templated on KType) instantiate this directly.
 *
 * Usage:
 *   kernelOpsHandler<KType> handler(jit);
 *   handler.generateKernelOps(kernelOps, ..., vecPool, maskPool, maskOffset);
 */
template<utils::kernelInstrType KType>
class kernelOpsHandler
{
    using Traits = traits::ArchitectureTraits<KType>;
    using VecPoolType =
        utils::registerPool<typename Traits::RegType, Traits::numRegs>;
    using MaskPoolType =
        utils::registerPool<Xbyak::Opmask, Traits::numMaskRegs>;

    std::unique_ptr<x86gen::kernelOpsGeneratorX86<KType>> kOpsGen;

  public:
    explicit kernelOpsHandler(Xbyak::CodeGenerator* jit)
        : kOpsGen(std::make_unique<x86gen::kernelOpsGeneratorX86<KType>>(jit))
    {
    }

    dlp::jit::jitGeneratorError generateKernelOps(
        std::vector<dlp::kernel_frame::kernelOpsMetaData>& kernelOps,
        const Xbyak::Reg64&   postOpsArgWrapperPtrReg,
        dlp::jit::jitAlgoType algoType,
        int                   MR,
        int                   NR,
        bool                  useMask,
        int                   numMaskRegs,
        int                   cRegStartIdx,
        int                   cRegCount,
        VecPoolType&          vecPool,
        MaskPoolType&         maskPool,
        int                   maskOffset)
    {
        if (!kOpsGen)
            return dlp::jit::jitGeneratorError::notSupported;
        return kOpsGen->generateKernelOps(kernelOps, postOpsArgWrapperPtrReg,
                                          algoType, MR, NR, useMask,
                                          numMaskRegs, cRegStartIdx, cRegCount,
                                          vecPool, maskPool, maskOffset);
    }
};

} // namespace amdzen::gen

extern template class amdzen::gen::kernelOpsHandler<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
extern template class amdzen::gen::kernelOpsHandler<
    amdzen::utils::kernelInstrType::avx512_ymm_32_reg>;
extern template class amdzen::gen::kernelOpsHandler<
    amdzen::utils::kernelInstrType::avx2_ymm_16_reg>;
