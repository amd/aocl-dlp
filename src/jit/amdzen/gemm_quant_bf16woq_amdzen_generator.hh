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
#include <vector>

#include "amdzen_generator.hh"
#include "jit/jit_generator_base.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "kernels/kernel_base.hh"

namespace amdzen::gen {

// Quant-register wrapper that emits a real BF16 microkernel for the WOQ
// bf16×{s4,u4} paths. Frame packsclb dequants nibble→BF16; this generator owns
// the s4/u4 datatype buckets and delegates codegen/execute to jitAmdZenBF16.
class jitAmdZenGemmQuantBf16Woq : public dlp::jit::gemmQuantJitGenerator
{
    std::vector<dlp::kernel_frame::kernelDatatype> mKernelDatatypes;
    std::unique_ptr<jitAmdZenBF16>                 mBf16Gen;

  public:
    jitAmdZenGemmQuantBf16Woq();
    ~jitAmdZenGemmQuantBf16Woq() override                       = default;
    jitAmdZenGemmQuantBf16Woq(const jitAmdZenGemmQuantBf16Woq&) = delete;
    jitAmdZenGemmQuantBf16Woq& operator=(const jitAmdZenGemmQuantBf16Woq&) =
        delete;
    jitAmdZenGemmQuantBf16Woq(jitAmdZenGemmQuantBf16Woq&&)            = delete;
    jitAmdZenGemmQuantBf16Woq& operator=(jitAmdZenGemmQuantBf16Woq&&) = delete;

    dlp::jit::jitGeneratorError operator()(
        const dlp::jit::gemmQuantJitGeneratorContext& jI) override;

    std::vector<dlp::kernel_frame::kernelDatatype>& getKernelDatatypes()
        override;
    std::vector<dlp::cpu_utils::isaFeature>& getIsaFeaturesRequired() override;
    dlp::kernels::kernelError                executeKernel(
                       dlp::kernels::kernelParams* _params) override;
    std::unique_ptr<dlp::jit::gemmQuantJitGenerator> clone() override;
};

} // namespace amdzen::gen
