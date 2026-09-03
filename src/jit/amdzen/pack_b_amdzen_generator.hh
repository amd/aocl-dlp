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

#include "debug_utils/gdb_helper_utils.hh"
#include "debug_utils/jit_debug_utils.hh"
#include "jit/jit_generator_base.hh"
#include "jit_generator_utils.hh"
#include "kernels/kernel_base.hh"

#include "xbyak/xbyak.h"

#include <memory>
#include <vector>

namespace amdzen::gen {

class jitAmdZenPackBFP32 : public dlp::jit::packBJitGenerator
{
    std::vector<dlp::kernel_frame::kernelDatatype> mKernelDatatypes;
    std::vector<dlp::cpu_utils::isaFeature>        mIsaFeaturesRequired;
    utils::kernelInstrType                         kType;
    int                                            numElemsPerReg;

    md_t NR;

    bool isColMajor_;

    // Exactly 2 kernels: [0] = NR (full), [1] = ltNR (masked).
    std::vector<void*>                                 kernelCodeBlocks;
    std::vector<std::unique_ptr<Xbyak::CodeGenerator>> codeGenerators;

    void setGeneratorKernelMetaInfo(
        dlp::kernel_frame::kernelInstrPreference kInstPref);

    dlp::jit::jitGeneratorError generateAllKernels(
        const dlp::jit::packBJitGeneratorContext& jI);

  public:
    jitAmdZenPackBFP32();
    ~jitAmdZenPackBFP32();
    jitAmdZenPackBFP32(const jitAmdZenPackBFP32&)            = delete;
    jitAmdZenPackBFP32& operator=(const jitAmdZenPackBFP32&) = delete;
    jitAmdZenPackBFP32(jitAmdZenPackBFP32&&)                 = delete;
    jitAmdZenPackBFP32& operator=(jitAmdZenPackBFP32&&)      = delete;

    dlp::jit::jitGeneratorError operator()(
        const dlp::jit::packBJitGeneratorContext& jI) override
    {
        return generateAllKernels(jI);
    }

    std::vector<dlp::kernel_frame::kernelDatatype>& getKernelDatatypes()
        override;
    std::vector<dlp::cpu_utils::isaFeature>& getIsaFeaturesRequired() override;
    dlp::kernels::kernelError                executeKernel(
                       dlp::kernels::kernelParams* _params) override;
    std::unique_ptr<dlp::jit::packBJitGenerator> clone() override;
};

// Orchestrator for JIT-generated BF16 pack-B kernels. Mirrors
// jitAmdZenPackBFP32: it owns ISA selection, generates the row-major fringe
// ladder, runs the per-panel cascade dispatch in executeKernel, and reports the
// packed-buffer strides.
//
// The ladder has numFull+1 kernels (numFull = NR / kernelWidth, kernelWidth =
// simdWidth / K_FACTOR = 16 bf16 lanes):
//   index 0            -> lt-block kernel (kernelWidth wide, runtime masked)
//   index i in [1..nf] -> looped kernel of width i*kernelWidth (nf == full NR)
// executeKernel issues the full-NR panels in one main call, then one base-width
// fringe panel, then one runtime-masked lt-block, matching the intrinsic
// packer's layout byte-for-byte. Both row-major and column-major sources are
// supported (column-major uses the shared 16x16 transpose path).
class jitAmdZenPackBBF16 : public dlp::jit::packBJitGenerator
{
    std::vector<dlp::kernel_frame::kernelDatatype> mKernelDatatypes;
    std::vector<dlp::cpu_utils::isaFeature>        mIsaFeaturesRequired;
    utils::kernelInstrType                         kType;
    int                                            numElemsPerReg;

    // BF16 fuses two consecutive K elements per lane (vdpbf16ps), so the packed
    // panel is laid out in K-pairs. K_FACTOR is fixed at 2 for BF16.
    static constexpr md_t K_FACTOR = 2;

    md_t NR;

    bool isColMajor_;

    // Fringe ladder: index 0 = lt-block (masked), index i = width
    // i*kernelWidth.
    std::vector<void*>                                 kernelCodeBlocks;
    std::vector<std::unique_ptr<Xbyak::CodeGenerator>> codeGenerators;

    void setGeneratorKernelMetaInfo(
        dlp::kernel_frame::kernelInstrPreference kInstPref);

    dlp::jit::jitGeneratorError generateAllKernels(
        const dlp::jit::packBJitGeneratorContext& jI);

  public:
    jitAmdZenPackBBF16();
    ~jitAmdZenPackBBF16();
    jitAmdZenPackBBF16(const jitAmdZenPackBBF16&)            = delete;
    jitAmdZenPackBBF16& operator=(const jitAmdZenPackBBF16&) = delete;
    jitAmdZenPackBBF16(jitAmdZenPackBBF16&&)                 = delete;
    jitAmdZenPackBBF16& operator=(jitAmdZenPackBBF16&&)      = delete;

    dlp::jit::jitGeneratorError operator()(
        const dlp::jit::packBJitGeneratorContext& jI) override
    {
        return generateAllKernels(jI);
    }

    std::vector<dlp::kernel_frame::kernelDatatype>& getKernelDatatypes()
        override;
    std::vector<dlp::cpu_utils::isaFeature>& getIsaFeaturesRequired() override;
    dlp::kernels::kernelError                executeKernel(
                       dlp::kernels::kernelParams* _params) override;
    std::unique_ptr<dlp::jit::packBJitGenerator> clone() override;
};

// Orchestrator for JIT-generated INT8 VNNI-4 pack-B kernels. Same ladder
// shape as jitAmdZenPackBBF16, with K_FACTOR = 4 (vpdpbusd) and kernelWidth
// = 16 n (one ZMM = 64 bytes).
//
//   index 0            -> lt-block kernel (kernelWidth wide, runtime masked)
//   index i in [1..nf] -> looped kernel of width i*kernelWidth (nf == full NR)
//
// executeKernel issues full-NR panels, then one base-width fringe panel,
// then one runtime-masked lt16 panel. packKernelInfo.accColSum selects
// packBGeneratorParams.accColSum: U8 emits pack-only code and S8 emits fused
// compensation. The execute ABI remains common. cs_dst is 64 (one VNNI ZMM),
// not NR/4; rs_dst is NR * 4. Column-major sources use a 16x16 dword transpose
// of 64-K tiles.
class jitAmdZenPackBINT8 : public dlp::jit::packBJitGenerator
{
    std::vector<dlp::kernel_frame::kernelDatatype> mKernelDatatypes;
    std::vector<dlp::cpu_utils::isaFeature>        mIsaFeaturesRequired;
    utils::kernelInstrType                         kType;
    int                                            numElemsPerReg;

    static constexpr md_t K_FACTOR = 4;

    md_t NR;

    bool isColMajor_;
    bool accColSum_;

    std::vector<void*>                                 kernelCodeBlocks;
    std::vector<std::unique_ptr<Xbyak::CodeGenerator>> codeGenerators;

    void setGeneratorKernelMetaInfo(
        dlp::kernel_frame::kernelInstrPreference kInstPref);

    dlp::jit::jitGeneratorError generateAllKernels(
        const dlp::jit::packBJitGeneratorContext& jI);

  public:
    jitAmdZenPackBINT8();
    ~jitAmdZenPackBINT8();
    jitAmdZenPackBINT8(const jitAmdZenPackBINT8&)            = delete;
    jitAmdZenPackBINT8& operator=(const jitAmdZenPackBINT8&) = delete;
    jitAmdZenPackBINT8(jitAmdZenPackBINT8&&)                 = delete;
    jitAmdZenPackBINT8& operator=(jitAmdZenPackBINT8&&)      = delete;

    dlp::jit::jitGeneratorError operator()(
        const dlp::jit::packBJitGeneratorContext& jI) override
    {
        return generateAllKernels(jI);
    }

    std::vector<dlp::kernel_frame::kernelDatatype>& getKernelDatatypes()
        override;
    std::vector<dlp::cpu_utils::isaFeature>& getIsaFeaturesRequired() override;
    dlp::kernels::kernelError                executeKernel(
                       dlp::kernels::kernelParams* _params) override;
    std::unique_ptr<dlp::jit::packBJitGenerator> clone() override;
};

} // namespace amdzen::gen
