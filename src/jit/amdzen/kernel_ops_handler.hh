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

#include <map>
#include <memory>

#include "jit/jit_generator_base.hh"
#include "jit/xbyak/xbyak.h"
#include "jit/xbyak/xbyak_util.h"
#include "jit_generator_utils.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "kernels/kernel_base.hh"
#include "x86_kernel_ops_generator.hh"

namespace amdzen::gen {

using namespace Xbyak;

// kernelOpHandler class that implements
class kernelOpsHandler
{
  public:
    kernelOpsHandler(Xbyak::CodeGenerator* jit, utils::kernelInstrType kType)
        : kOpsGen{ nullptr }
    {
        if (kType == utils::kernelInstrType::avx512_zmm_32_reg) {
            kOpsGen = std::make_unique<x86gen::kernelOpsGeneratorX86<
                utils::kernelInstrType::avx512_zmm_32_reg>>(jit);
        } else if (kType == utils::kernelInstrType::avx512_ymm_32_reg) {
            kOpsGen = std::make_unique<x86gen::kernelOpsGeneratorX86<
                utils::kernelInstrType::avx512_ymm_32_reg>>(jit);
        } else if (kType == utils::kernelInstrType::avx2_ymm_16_reg) {
            kOpsGen = std::make_unique<x86gen::kernelOpsGeneratorX86<
                utils::kernelInstrType::avx2_ymm_16_reg>>(jit);
        }
    }

    ~kernelOpsHandler()                                  = default;
    kernelOpsHandler(const kernelOpsHandler&)            = delete;
    kernelOpsHandler& operator=(const kernelOpsHandler&) = delete;
    kernelOpsHandler(kernelOpsHandler&&)                 = delete;
    kernelOpsHandler& operator=(kernelOpsHandler&&)      = delete;

    // Main post-op interface
    dlp::jit::jitGeneratorError generateKernelOps(
        std::vector<dlp::kernel_frame::kernelOpsMetaData>& kernelOps,
        const Xbyak::Reg64&   postOpsArgWrapperPtrReg,
        dlp::jit::jitAlgoType algoType,
        int                   MR,
        int                   NR,
        bool                  useMask,
        int                   numMaskRegs,
        int                   cRegStartIdx,
        int                   cRegCount)
    {
        if (kOpsGen) {
            return kOpsGen->generateKernelOps(
                kernelOps, postOpsArgWrapperPtrReg, algoType, MR, NR, useMask,
                numMaskRegs, cRegStartIdx, cRegCount);
        }
        return dlp::jit::jitGeneratorError::error;
    }

    // Function to generate the gelu const embeddings within the kernel.
    dlp::jit::jitGeneratorError generateKernelOpsAttributes()
    {
        if (kOpsGen) {
            return kOpsGen->embedKernelOpsAttributes();
        }
        return dlp::jit::jitGeneratorError::error;
    }

  private:
    std::unique_ptr<kernelOpsGeneratorInterface> kOpsGen;
};

} // namespace amdzen::gen
