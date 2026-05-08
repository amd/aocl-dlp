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

#include "pack_b_amdzen_generator.hh"
#include "f32_col_major_pack_b_generator.hh"
#include "f32_pack_b_generator.hh"
#include "jit_generator_utils.hh"
#include "jit_register/jit_register.hh"
#include "traits.hh"

namespace amdzen::gen {

jitAmdZenPackBFP32::jitAmdZenPackBFP32()
    : mKernelDatatypes({ dlp::kernel_frame::kernelDatatype::f32f32f32of32 })
    , mIsaFeaturesRequired({ dlp::cpu_utils::isaFeature::avx2 })
    , kType(utils::kernelInstrType::none)
    , numElemsPerReg(1)
    , NR(0)
    , isColMajor_(false)
{
}

jitAmdZenPackBFP32::~jitAmdZenPackBFP32()
{
    codeGenerators.clear();
    for (auto& block : kernelCodeBlocks) {
        block = nullptr;
    }
    kernelCodeBlocks.clear();
}

void
jitAmdZenPackBFP32::setGeneratorKernelMetaInfo(
    dlp::kernel_frame::kernelInstrPreference kInstPref)
{
    kType = utils::kernelInstrType::none;
    switch (kInstPref) {
        case dlp::kernel_frame::kernelInstrPreference::avx512_zmm_favour: {
            kType = utils::kernelInstrType::avx512_zmm_32_reg;
            numElemsPerReg =
                traits::ArchitectureTraits<
                    utils::kernelInstrType::avx512_zmm_32_reg>::regBytes
                / sizeof(float);
            break;
        }
        case dlp::kernel_frame::kernelInstrPreference::avx2_ymm_favour: {
            kType = utils::kernelInstrType::avx2_ymm_16_reg;
            numElemsPerReg =
                traits::ArchitectureTraits<
                    utils::kernelInstrType::avx2_ymm_16_reg>::regBytes
                / sizeof(float);
            break;
        }
        default:
            break;
    }
}

dlp::jit::jitGeneratorError
jitAmdZenPackBFP32::generateAllKernels(const dlp::jit::jitGeneratorContext& jI)
{
    if (jI.packKI == nullptr) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    setGeneratorKernelMetaInfo(jI.packKI->kInstPref);

    if (kType == utils::kernelInstrType::none) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    NR          = jI.packKI->panel_dim;
    isColMajor_ = jI.packKI->isColMajor;

    constexpr int numKernels = 2;
    kernelCodeBlocks.resize(numKernels, nullptr);

    utils::packBGeneratorParams genParams(0, jI.packKI->k_factor, kType, false,
                                          0);

    for (int ki = 0; ki < numKernels; ++ki) {
        bool useMask     = (ki == 1);
        int  numMaskRegs = useMask ? 1 : 0;

        genParams.NR          = NR;
        genParams.useMask     = useMask;
        genParams.numMaskRegs = numMaskRegs;

        dlp::jit::jitGeneratorError err = dlp::jit::jitGeneratorError::success;

        // Abstract the code further(moving gen declaration outside, and the
        // if-clause code as well. Maybe using lambda functions).
        if (isColMajor_) {
            switch (kType) {
                case utils::kernelInstrType::avx512_zmm_32_reg: {
                    auto gen = std::make_unique<
                        PackBcodeGenerator::jitPackBF32ColMajor<
                            utils::kernelInstrType::avx512_zmm_32_reg>>();
                    err = gen->generateKernel(genParams);
                    if (err == dlp::jit::jitGeneratorError::success) {
                        gen->ready();
                        kernelCodeBlocks[ki] = const_cast<void*>(
                            static_cast<const void*>(gen->getCode()));
                        codeGenerators.push_back(std::move(gen));
                    }
                    break;
                }
                case utils::kernelInstrType::avx2_ymm_16_reg: {
                    auto gen = std::make_unique<
                        PackBcodeGenerator::jitPackBF32ColMajor<
                            utils::kernelInstrType::avx2_ymm_16_reg>>();
                    err = gen->generateKernel(genParams);
                    if (err == dlp::jit::jitGeneratorError::success) {
                        gen->ready();
                        kernelCodeBlocks[ki] = const_cast<void*>(
                            static_cast<const void*>(gen->getCode()));
                        codeGenerators.push_back(std::move(gen));
                    }
                    break;
                }
                default:
                    err = dlp::jit::jitGeneratorError::notSupported;
                    break;
            }
        } else {
            switch (kType) {
                case utils::kernelInstrType::avx512_zmm_32_reg: {
                    auto gen = std::make_unique<PackBcodeGenerator::jitPackBF32<
                        utils::kernelInstrType::avx512_zmm_32_reg>>();
                    err      = gen->generateKernel(genParams);
                    if (err == dlp::jit::jitGeneratorError::success) {
                        gen->ready();
                        kernelCodeBlocks[ki] = const_cast<void*>(
                            static_cast<const void*>(gen->getCode()));
                        codeGenerators.push_back(std::move(gen));
                    }
                    break;
                }
                case utils::kernelInstrType::avx2_ymm_16_reg: {
                    auto gen = std::make_unique<PackBcodeGenerator::jitPackBF32<
                        utils::kernelInstrType::avx2_ymm_16_reg>>();
                    err      = gen->generateKernel(genParams);
                    if (err == dlp::jit::jitGeneratorError::success) {
                        gen->ready();
                        kernelCodeBlocks[ki] = const_cast<void*>(
                            static_cast<const void*>(gen->getCode()));
                        codeGenerators.push_back(std::move(gen));
                    }
                    break;
                }
                default:
                    err = dlp::jit::jitGeneratorError::notSupported;
                    break;
            }
        }

        // Proper cleanup code. Review once the codegen abstraction is improved.
        if (err != dlp::jit::jitGeneratorError::success) {
            codeGenerators.clear();
            kernelCodeBlocks.clear();
            return err;
        }

        DLP_ENABLE_JIT_DUMP_AND_MONITOR(
            kernelCodeBlocks[ki], utils::JIT_KERNEL_SIZE,
            isColMajor_ ? "jit_f32_pack_b_col_major"
                        : "jit_f32_pack_b_row_major",
            0, NR, useMask, ki);
    }

    return dlp::jit::jitGeneratorError::success;
}

std::vector<dlp::kernel_frame::kernelDatatype>&
jitAmdZenPackBFP32::getKernelDatatypes()
{
    return mKernelDatatypes;
}

std::vector<dlp::cpu_utils::isaFeature>&
jitAmdZenPackBFP32::getIsaFeaturesRequired()
{
    return mIsaFeaturesRequired;
}

dlp::kernels::kernelError
jitAmdZenPackBFP32::executeKernel(dlp::kernels::kernelParams* _params)
{
    if (kernelCodeBlocks.empty()) {
        return dlp::kernels::kernelError::error;
    }

    auto* params = static_cast<dlp::kernels::packBParams*>(_params);

    md_t n_total = params->n;

    // Source byte stride per column: for row-major, advancing by one
    // column is sizeof(float); for column-major, it is cs_src * sizeof(float).
    // Why a condition? In case it is row-major, the incoming cs_src will be one
    // anyways right?
    md_t srcColStrideBytes = isColMajor_ ? params->cs_src * sizeof(float)
                                         : sizeof(float);

    // ---- Dispatch: NR full panels via kernel [0], remainder via kernel [1]
    // ----

    // Pre-compute K-fringe masks for column-major transpose kernels.
    // Both the full NR kernel and ltNR kernel share the same K dimension,
    // so this mask is set once here and propagated to both calls.
    md_t k_tail = params->k % numElemsPerReg;
    if (kType == utils::kernelInstrType::avx512_zmm_32_reg) {
        params->k_fringe_mask =
            (k_tail > 0) ? static_cast<uint32_t>((1u << k_tail) - 1) : 0xFFFF;
    } else if (kType == utils::kernelInstrType::avx2_ymm_16_reg) {
        if (k_tail > 0) {
            // kMaskArray holds per-lane int32_t masks for AVX2. Active lanes
            // must be written as all-ones (-1), inactive lanes as zero.
            for (int i = 0; i < numElemsPerReg; ++i)
                params->kMaskArray[i] = (static_cast<md_t>(i) < k_tail) ? -1
                                                                        : 0;
        } else {
            params->kMaskArray.fill(-1);
        }
    }

    md_t n_full_pieces_limit = (n_total / NR) * NR;

    // Full NR panels
    if (n_full_pieces_limit > 0) {
        params->n_full_pieces_limit = n_full_pieces_limit;
        params->n_partial           = 0;

        auto mainKernel =
            reinterpret_cast<utils::jit_pack_b_kernel>(kernelCodeBlocks[0]);
        DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(mainKernel));
        mainKernel(params);
    }

    // Remainder (n_remainder < NR)
    md_t n_remainder = n_total - n_full_pieces_limit;

    if (n_remainder > 0) {
        params->src = static_cast<char*>(params->src)
                      + n_full_pieces_limit * srcColStrideBytes;
        params->dst = static_cast<char*>(params->dst)
                      + n_full_pieces_limit * params->k * sizeof(float);
        params->n = n_remainder;

        params->n_full_pieces_limit = 0;
        params->n_partial           = n_remainder;

        if (isColMajor_) {
            // Column-major: single tail mask stored in slot 0.
            md_t n_tail = n_remainder % numElemsPerReg;
            if (kType == utils::kernelInstrType::avx512_zmm_32_reg) {
                params->nFringeMaskPerBlock[0] =
                    (n_tail > 0) ? static_cast<uint32_t>((1u << n_tail) - 1)
                                 : 0xFFFF;
            } else if (kType == utils::kernelInstrType::avx2_ymm_16_reg) {
                if (n_tail > 0) {
                    for (int i = 0; i < numElemsPerReg; ++i)
                        params->nMaskPerBlock[0][i] =
                            (static_cast<md_t>(i) < n_tail) ? -1 : 0;
                } else {
                    params->nMaskPerBlock[0].fill(-1);
                }
            }
        } else {
            // Row-major: per-SIMD-block masks so each load uses a distinct
            // register/opmask.
            int  numBlocks = NR / numElemsPerReg;
            md_t remaining = n_remainder;

            if (kType == utils::kernelInstrType::avx512_zmm_32_reg) {
                for (int b = 0; b < numBlocks; ++b) {
                    if (remaining >= static_cast<md_t>(numElemsPerReg)) {
                        params->nFringeMaskPerBlock[b] = 0xFFFF;
                        remaining -= numElemsPerReg;
                    } else if (remaining > 0) {
                        params->nFringeMaskPerBlock[b] =
                            static_cast<uint32_t>((1u << remaining) - 1);
                        remaining = 0;
                    } else {
                        params->nFringeMaskPerBlock[b] = 0x0000;
                    }
                }
            } else if (kType == utils::kernelInstrType::avx2_ymm_16_reg) {
                for (int b = 0; b < numBlocks; ++b) {
                    if (remaining >= static_cast<md_t>(numElemsPerReg)) {
                        params->nMaskPerBlock[b].fill(-1);
                        remaining -= numElemsPerReg;
                    } else if (remaining > 0) {
                        for (int i = 0; i < numElemsPerReg; ++i)
                            params->nMaskPerBlock[b][i] =
                                (static_cast<md_t>(i) < remaining) ? -1 : 0;
                        remaining = 0;
                    } else {
                        params->nMaskPerBlock[b].fill(0);
                    }
                }
            }
        }

        auto ltKernel =
            reinterpret_cast<utils::jit_pack_b_kernel>(kernelCodeBlocks[1]);
        DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(ltKernel));
        ltKernel(params);
    }

    params->rs_dst = NR;
    params->cs_dst = 1;

    return dlp::kernels::kernelError::success;
}

std::unique_ptr<dlp::jit::jitGeneratorBase>
jitAmdZenPackBFP32::clone()
{
    return std::make_unique<jitAmdZenPackBFP32>();
}

DLP_REGISTER_STATIC_PACKB_JIT_GENERATOR(jitAmdZenPackBFP32, "f32_pack_b");

} // namespace amdzen::gen
