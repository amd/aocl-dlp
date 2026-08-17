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

#include "debug_utils/gdb_helper_utils.hh"
#include "debug_utils/jit_debug_utils.hh"
#include "jit/jit_generator_base.hh"
#include "jit_generator_utils.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "kernels/kernel_base.hh"
#include "xbyak/xbyak.h"

namespace amdzen::gen {

class jitAmdZenGemmQuant : public dlp::jit::gemmQuantJitGenerator
{
    std::vector<dlp::kernel_frame::kernelDatatype> mKernelDatatypes;
    std::vector<dlp::cpu_utils::isaFeature>        mIsaFeaturesRequired;
    utils::kernelInstrType                         kType;
    int                                            numElemsPerReg;

    // VNNI packs 4 int8s per 32-bit dword.
    int VNNI_CONST = 4;

    md_t MR = 0, NR = 0, KC = 0, K_UNROLL = 0, PREFETCH_C_DIST = 0;
    md_t c_downscale = 0;

    dlp::kernel_frame::DataType b_scale_type =
        dlp::kernel_frame::DataType::invalid;
    dlp::kernel_frame::DataType a_scale_type =
        dlp::kernel_frame::DataType::invalid;

    md_t numMRVariants = 0, numNRVariants = 0;
    md_t numKernelVariants = 0;

    std::vector<void*>                                 kernelCodeBlocks;
    std::vector<std::unique_ptr<Xbyak::CodeGenerator>> codeGenerators;

    int  getProcessBlockSize() const;
    void setGeneratorKernelMetaInfo(
        dlp::kernel_frame::kernelInstrPreference kInstPref);

    dlp::jit::jitGeneratorError generateAllKernels(
        const dlp::jit::gemmQuantJitGeneratorContext& jI);

  public:
    jitAmdZenGemmQuant();
    ~jitAmdZenGemmQuant();
    jitAmdZenGemmQuant(const jitAmdZenGemmQuant&)            = delete;
    jitAmdZenGemmQuant& operator=(const jitAmdZenGemmQuant&) = delete;
    jitAmdZenGemmQuant(jitAmdZenGemmQuant&&)                 = delete;
    jitAmdZenGemmQuant& operator=(jitAmdZenGemmQuant&&)      = delete;

    dlp::jit::jitGeneratorError operator()(
        const dlp::jit::gemmQuantJitGeneratorContext& jI) override
    {
        return generateAllKernels(jI);
    }

    std::vector<dlp::kernel_frame::kernelDatatype>& getKernelDatatypes()
        override;
    std::vector<dlp::cpu_utils::isaFeature>& getIsaFeaturesRequired() override;
    dlp::kernels::kernelError                executeKernel(
                       dlp::kernels::kernelParams* _params) override;
    std::unique_ptr<dlp::jit::gemmQuantJitGenerator> clone() override;
};

} // namespace amdzen::gen
