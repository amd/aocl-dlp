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
#include "bf16_col_major_pack_b_generator.hh"
#include "bf16_pack_b_generator.hh"
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

// ===========================================================================
// BF16 pack-B orchestrator (jitAmdZenPackBBF16)
// ===========================================================================

jitAmdZenPackBBF16::jitAmdZenPackBBF16()
    : mKernelDatatypes({ dlp::kernel_frame::kernelDatatype::bf16bf16f32of32,
                         dlp::kernel_frame::kernelDatatype::bf16bf16f32obf16 })
    , mIsaFeaturesRequired({ dlp::cpu_utils::isaFeature::avx512bf16 })
    , kType(utils::kernelInstrType::none)
    , numElemsPerReg(1)
    , NR(0)
    , isColMajor_(false)
{
}

jitAmdZenPackBBF16::~jitAmdZenPackBBF16()
{
    codeGenerators.clear();
    for (auto& block : kernelCodeBlocks) {
        block = nullptr;
    }
    kernelCodeBlocks.clear();
}

void
jitAmdZenPackBBF16::setGeneratorKernelMetaInfo(
    dlp::kernel_frame::kernelInstrPreference kInstPref)
{
    kType = utils::kernelInstrType::none;
    switch (kInstPref) {
        // BF16 pack-B requires AVX-512-BF16. Both the dedicated BF16 preference
        // and the generic AVX-512 ZMM preference map to the same 32x ZMM
        // backend; actual BF16 availability is enforced via the kernel's
        // required ISA features (avx512bf16), and the DE reroutes to the F32
        // pack-B path when native BF16 is unavailable.
        case dlp::kernel_frame::kernelInstrPreference::avx512_zmm_bf16_favour:
        case dlp::kernel_frame::kernelInstrPreference::avx512_zmm_favour: {
            kType = utils::kernelInstrType::avx512_zmm_32_reg;
            numElemsPerReg =
                traits::ArchitectureTraits<
                    utils::kernelInstrType::avx512_zmm_32_reg>::regBytes
                / sizeof(bfloat16);
            break;
        }
        default:
            break;
    }
}

dlp::jit::jitGeneratorError
jitAmdZenPackBBF16::generateAllKernels(const dlp::jit::jitGeneratorContext& jI)
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

    if (kType != utils::kernelInstrType::avx512_zmm_32_reg) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    // Fringe ladder (mirrors GEMM / the intrinsic packer). The pack block width
    // is kernelWidth = simdWidth / K_FACTOR (16 bf16 lanes for a ZMM). We build
    // looped kernels for every NR-multiple width down to one block, plus a
    // single runtime-masked lt-block kernel:
    //
    //   index 0            -> lt-block kernel (width = kernelWidth, useMask)
    //   index i in [1..nf] -> looped kernel of width i*kernelWidth
    //                         (i == nf is the main full-NR kernel)
    //
    // Each looped kernel packs as many panels of its width as requested via
    // packBParams.n_full_pieces_limit, so a single base-width fringe panel is
    // just that kernel invoked with the limit set to its width.
    const md_t kernelWidth = static_cast<md_t>(numElemsPerReg) / K_FACTOR;

    // The ladder only covers a positive NR that decomposes into whole
    // kernelWidth blocks. NR <= 0 is invalid, and a non-multiple NR would
    // silently drop its top (NR % kernelWidth) columns, so degrade to
    // notSupported and let the caller fall back to the intrinsic packer
    // instead.
    if (kernelWidth <= 0 || NR <= 0 || (NR % kernelWidth) != 0) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    const int numFull    = static_cast<int>(NR / kernelWidth);
    const int numKernels = numFull + 1;
    kernelCodeBlocks.resize(numKernels, nullptr);

    utils::packBGeneratorParams genParams(NR, jI.packKI->k_factor, kType,
                                          /*useMask=*/false, /*numMaskRegs=*/0);

    for (int ki = 0; ki < numKernels; ++ki) {
        const bool useMask = (ki == 0);
        const md_t width   = useMask ? kernelWidth
                                     : static_cast<md_t>(ki) * kernelWidth;

        genParams.NR          = width;
        genParams.useMask     = useMask;
        genParams.numMaskRegs = useMask ? 1 : 0;
        // Only the main full-NR kernel loops over N panels; fringe kernels pack
        // a single panel and skip the loop scaffolding.
        genParams.nLoop = (ki == numFull);

        dlp::jit::jitGeneratorError err = dlp::jit::jitGeneratorError::success;

        // Row-major and column-major share the same ladder shape and genParams;
        // only the per-panel codegen (interleave vs. transpose) differs.
        if (isColMajor_) {
            auto gen =
                std::make_unique<PackBcodeGenerator::jitPackBBF16ColMajor<
                    utils::kernelInstrType::avx512_zmm_32_reg>>();
            err = gen->generateKernel(genParams);
            if (err == dlp::jit::jitGeneratorError::success) {
                gen->ready();
                kernelCodeBlocks[ki] =
                    const_cast<void*>(static_cast<const void*>(gen->getCode()));
                codeGenerators.push_back(std::move(gen));
            }
        } else {
            auto gen = std::make_unique<PackBcodeGenerator::jitPackBBF16<
                utils::kernelInstrType::avx512_zmm_32_reg>>();
            err      = gen->generateKernel(genParams);
            if (err == dlp::jit::jitGeneratorError::success) {
                gen->ready();
                kernelCodeBlocks[ki] =
                    const_cast<void*>(static_cast<const void*>(gen->getCode()));
                codeGenerators.push_back(std::move(gen));
            }
        }

        if (err != dlp::jit::jitGeneratorError::success) {
            codeGenerators.clear();
            kernelCodeBlocks.clear();
            return err;
        }

        DLP_ENABLE_JIT_DUMP_AND_MONITOR(
            kernelCodeBlocks[ki], utils::JIT_KERNEL_SIZE,
            isColMajor_ ? "jit_bf16_pack_b_col_major"
                        : "jit_bf16_pack_b_row_major",
            0, width, useMask, ki);
    }

    return dlp::jit::jitGeneratorError::success;
}

std::vector<dlp::kernel_frame::kernelDatatype>&
jitAmdZenPackBBF16::getKernelDatatypes()
{
    return mKernelDatatypes;
}

std::vector<dlp::cpu_utils::isaFeature>&
jitAmdZenPackBBF16::getIsaFeaturesRequired()
{
    return mIsaFeaturesRequired;
}

dlp::kernels::kernelError
jitAmdZenPackBBF16::executeKernel(dlp::kernels::kernelParams* _params)
{
    if (kernelCodeBlocks.empty()) {
        return dlp::kernels::kernelError::error;
    }

    auto* params = static_cast<dlp::kernels::packBParams*>(_params);

    const md_t kernelWidth = static_cast<md_t>(numElemsPerReg) / K_FACTOR;
    const md_t n_total     = params->n;

    // A source n-step is one bf16 element row-major, or cs_src elements
    // column-major. The packed panel rounds K up to an even number of rows
    // (BF16 packs K in pairs); a width-W panel occupies KC_updated * W bf16
    // elements regardless of layout, matching the intrinsic packer's offsets.
    const md_t srcColStrideBytes =
        isColMajor_ ? params->cs_src * sizeof(bfloat16) : sizeof(bfloat16);
    const md_t KC_updated        = (params->k + 1) & ~static_cast<md_t>(1);
    const md_t dstPanelBytesPerW = KC_updated * sizeof(bfloat16);

    // Column-major kernels transpose runtime-K tiles, so the trailing K % 32
    // elements need an epi16 K-fringe mask (shared by the main, base-width, and
    // lt-block kernels). Odd K is covered too: the zero-masked partner lane
    // fills the dangling K-pair.
    if (isColMajor_) {
        const md_t k_tail     = params->k % numElemsPerReg;
        params->k_fringe_mask = (k_tail > 0)
                                    ? static_cast<uint32_t>((1u << k_tail) - 1)
                                    : 0xFFFFFFFFu;
    }

    auto advance = [&](md_t width) {
        params->src =
            static_cast<char*>(params->src) + width * srcColStrideBytes;
        params->dst =
            static_cast<char*>(params->dst) + width * dstPanelBytesPerW;
    };
    auto runKernel = [&](int idx) {
        auto kernel =
            reinterpret_cast<utils::jit_pack_b_kernel>(kernelCodeBlocks[idx]);
        DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(kernel));
        kernel(params);
    };

    const int numFull = static_cast<int>(NR / kernelWidth); // main kernel index

    // ---- Main: all full-NR panels in a single call (kernel[numFull]) --------
    const md_t n_full_pieces_limit = (n_total / NR) * NR;
    if (n_full_pieces_limit > 0) {
        params->n_full_pieces_limit = n_full_pieces_limit;
        params->n_partial           = 0;
        runKernel(numFull);
        advance(n_full_pieces_limit);
    }

    md_t n_partial = n_total - n_full_pieces_limit; // 0 .. NR-1

    // ---- Base fringe: one panel of the largest kernelWidth-multiple <=
    // n_partial (single-panel kernel: packs from src/dst directly, ignores the
    // loop limit).
    const md_t baseW = (n_partial / kernelWidth) * kernelWidth;
    if (baseW > 0) {
        runKernel(static_cast<int>(baseW / kernelWidth));
        advance(baseW);
        n_partial -= baseW;
    }

    // ---- lt-block fringe: remaining (< kernelWidth) columns, runtime masked
    // -- (single-panel masked kernel: one 16-lane block, low n_partial lanes
    // live.) Row-major masks the trailing 16-lane load; column-major instead
    // loads only the low n_partial columns (n_partial drives the load count).
    if (n_partial > 0) {
        if (isColMajor_) {
            params->n_partial = n_partial;
        } else {
            params->nFringeMaskPerBlock[0] =
                static_cast<uint32_t>((1u << n_partial) - 1);
        }
        runKernel(0);
    }

    params->rs_dst = NR * K_FACTOR;
    params->cs_dst = NR / K_FACTOR;

    return dlp::kernels::kernelError::success;
}

std::unique_ptr<dlp::jit::jitGeneratorBase>
jitAmdZenPackBBF16::clone()
{
    return std::make_unique<jitAmdZenPackBBF16>();
}

// ===========================================================================
// Static registration of pack-B JIT generators (all datatypes).
// ===========================================================================

DLP_REGISTER_STATIC_PACKB_JIT_GENERATOR(jitAmdZenPackBFP32, "f32_pack_b");
DLP_REGISTER_STATIC_PACKB_JIT_GENERATOR(jitAmdZenPackBBF16, "bf16_pack_b");

} // namespace amdzen::gen
