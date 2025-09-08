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

#include "jit/jit_generator_base.hh"
#include "jit_generator_utils.hh"
#include "kernels/kernel_base.hh"

namespace amdzen::gen {

// Single generator for both AVX512 and AVX2 micro-kernels.
class jitAmdZenFP32 : public dlp::jit::jitGeneratorBase
{

    std::vector<dlp::kernel_frame::kernelDatatype> mKernelDatatypes;
    std::vector<dlp::cpu_utils::isaFeature>        mIsaFeaturesRequired;
    bool                                           isZen4;
    bool                                           isZen;

    utils::kernelInstrType getGeneratorKernelType(
        dlp::kernel_frame::kernelInstrPreference kInstPref);

  public:
    // jitAVX512 base;
    md_t               MR, NR, KC;
    md_t               numMRVariants, numNRVariants;
    md_t               numKernelVariants;
    md_t               K_UNROLL;
    std::vector<void*> kernelCodeBlocks;

    jitAmdZenFP32();
    ~jitAmdZenFP32();
    jitAmdZenFP32(const jitAmdZenFP32&)            = delete;
    jitAmdZenFP32& operator=(const jitAmdZenFP32&) = delete;
    jitAmdZenFP32(jitAmdZenFP32&&)                 = delete;
    jitAmdZenFP32& operator=(jitAmdZenFP32&&)      = delete;

    dlp::jit::jitGeneratorError generateAllKernels(
        const dlp::jit::jitGeneratorContext& jI);

    dlp::jit::jitGeneratorError operator()(
        const dlp::jit::jitGeneratorContext& jI) override
    {
        return generateAllKernels(jI);
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
        dlp::kernels::kernelParams* _params) override;

    std::unique_ptr<jitGeneratorBase> clone() override
    {
        return std::make_unique<jitAmdZenFP32>();
    }
};

} // namespace amdzen::gen
