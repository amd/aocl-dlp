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

#include "gemm_quant_bf16woq_amdzen_generator.hh"

#include "jit_register/jit_register.hh"

namespace amdzen::gen {

jitAmdZenGemmQuantBf16Woq::jitAmdZenGemmQuantBf16Woq()
    : mKernelDatatypes({ dlp::kernel_frame::kernelDatatype::bf16s4f32of32,
                         dlp::kernel_frame::kernelDatatype::bf16s4f32obf16,
                         dlp::kernel_frame::kernelDatatype::bf16u4f32of32,
                         dlp::kernel_frame::kernelDatatype::bf16u4f32obf16 })
    , mBf16Gen(std::make_unique<jitAmdZenBF16>())
{
}

dlp::jit::jitGeneratorError
jitAmdZenGemmQuantBf16Woq::operator()(
    const dlp::jit::gemmQuantJitGeneratorContext& jI)
{
    // Generate the BF16 microkernel from the composed base kernelInfo. aQuant
    // / bQuant are cache-key metadata only; packsclb already applied scales.
    dlp::jit::gemmJitGeneratorContext bf16Ctx{ jI.kI.base };
    return mBf16Gen->operator()(bf16Ctx);
}

std::vector<dlp::kernel_frame::kernelDatatype>&
jitAmdZenGemmQuantBf16Woq::getKernelDatatypes()
{
    return mKernelDatatypes;
}

std::vector<dlp::cpu_utils::isaFeature>&
jitAmdZenGemmQuantBf16Woq::getIsaFeaturesRequired()
{
    return mBf16Gen->getIsaFeaturesRequired();
}

dlp::kernels::kernelError
jitAmdZenGemmQuantBf16Woq::executeKernel(dlp::kernels::kernelParams* _params)
{
    return mBf16Gen->executeKernel(_params);
}

std::unique_ptr<dlp::jit::gemmQuantJitGenerator>
jitAmdZenGemmQuantBf16Woq::clone()
{
    return std::make_unique<jitAmdZenGemmQuantBf16Woq>();
}

DLP_REGISTER_STATIC_GEMM_QUANT_JIT_GENERATOR(
    jitAmdZenGemmQuantBf16Woq, "dlp_amdzen_gemm_quant_bf16woq_jit");

} // namespace amdzen::gen
