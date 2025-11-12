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

#include <type_traits>

#include "jit/jit_generator_base.hh"
#include "jit/xbyak/xbyak.h"
#include "kernel_frame/kernel_frame_base.hh"

namespace amdzen::gen {

class kernelOpsGeneratorInterface
{
  public:
    virtual ~kernelOpsGeneratorInterface() = default;

    virtual dlp::jit::jitGeneratorError generateKernelOps(
        std::vector<dlp::kernel_frame::kernelOpsMetaData>& kernelOps,
        const Xbyak::Reg64&   postOpsArgWrapperPtrReg,
        dlp::jit::jitAlgoType algoType,
        int                   MR,
        int                   NR,
        bool                  useMask,
        int                   numMaskRegs,
        int                   cRegStartIdx,
        int                   cRegCount) = 0;
    virtual dlp::jit::jitGeneratorError bias(
        dlp::kernel_frame::kernelOpsMetaData& op) = 0;
    virtual dlp::jit::jitGeneratorError relu(
        dlp::kernel_frame::kernelOpsMetaData& op) = 0;
    virtual dlp::jit::jitGeneratorError reluScale(
        dlp::kernel_frame::kernelOpsMetaData& op) = 0;
    virtual dlp::jit::jitGeneratorError geluTanh(
        dlp::kernel_frame::kernelOpsMetaData& op) = 0;
    virtual dlp::jit::jitGeneratorError geluErf(
        dlp::kernel_frame::kernelOpsMetaData& op) = 0;
    virtual dlp::jit::jitGeneratorError clip(
        dlp::kernel_frame::kernelOpsMetaData& op) = 0;
    virtual dlp::jit::jitGeneratorError downscale(
        dlp::kernel_frame::kernelOpsMetaData& op) = 0;
    virtual dlp::jit::jitGeneratorError matadd(
        dlp::kernel_frame::kernelOpsMetaData& op) = 0;
    virtual dlp::jit::jitGeneratorError matmul(
        dlp::kernel_frame::kernelOpsMetaData& op) = 0;
    virtual dlp::jit::jitGeneratorError swish(
        dlp::kernel_frame::kernelOpsMetaData& op) = 0;
    virtual dlp::jit::jitGeneratorError tanh(
        dlp::kernel_frame::kernelOpsMetaData& op) = 0;
    virtual dlp::jit::jitGeneratorError sigmoid(
        dlp::kernel_frame::kernelOpsMetaData& op) = 0;
    virtual dlp::jit::jitGeneratorError aDQuantize(
        dlp::kernel_frame::kernelOpsMetaData& op)                  = 0;
    virtual dlp::jit::jitGeneratorError embedKernelOpsAttributes() = 0;

    virtual void advancePostOpsPtr() = 0;

  protected:
    // Template function for dispatching kernel operations - eliminates code
    // duplication.
    template<typename Derived>
    dlp::jit::jitGeneratorError dispatchKernelOps(
        std::vector<dlp::kernel_frame::kernelOpsMetaData>& kernelOps)
    {
        static_assert(
            std::is_base_of_v<kernelOpsGeneratorInterface, Derived>,
            "Dispatcher cannot work with non kernelOpsGeneratorInterface"
            " objects.");
        auto* impl = static_cast<Derived*>(this);

        for (auto& op : kernelOps) {
            dlp::jit::jitGeneratorError result;
            switch (op.type) {
                case dlp::kernel_frame::kernelOps::bias:
                    result = impl->bias(op);
                    break;
                case dlp::kernel_frame::kernelOps::relu:
                    result = impl->relu(op);
                    break;
                case dlp::kernel_frame::kernelOps::reluScale:
                    result = impl->reluScale(op);
                    break;
                case dlp::kernel_frame::kernelOps::geluTanh:
                    result = impl->geluTanh(op);
                    break;
                case dlp::kernel_frame::kernelOps::geluErf:
                    result = impl->geluErf(op);
                    break;
                case dlp::kernel_frame::kernelOps::clip:
                    result = impl->clip(op);
                    break;
                case dlp::kernel_frame::kernelOps::downscale:
                    result = impl->downscale(op);
                    break;
                case dlp::kernel_frame::kernelOps::matAdd:
                    result = impl->matadd(op);
                    break;
                case dlp::kernel_frame::kernelOps::matMul:
                    result = impl->matmul(op);
                    break;
                case dlp::kernel_frame::kernelOps::swish:
                    result = impl->swish(op);
                    break;
                case dlp::kernel_frame::kernelOps::tanh:
                    result = impl->tanh(op);
                    break;
                case dlp::kernel_frame::kernelOps::sigmoid:
                    result = impl->sigmoid(op);
                    break;
                case dlp::kernel_frame::kernelOps::aDQuantize:
                    result = impl->aDQuantize(op);
                    break;
                default:
                    return dlp::jit::jitGeneratorError::notSupported;
            }

            if (result != dlp::jit::jitGeneratorError::success) {
                return result;
            }

            impl->advancePostOpsPtr();
        }

        return dlp::jit::jitGeneratorError::success;
    }
};

} // namespace amdzen::gen
