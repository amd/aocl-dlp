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

#include "gemm_quant_amdzen_generator.hh"

#include "aocl_dlp_config.h"

#include "jit_register/jit_register.hh"
#include "s8_gemm_quant_generator.hh"
#include "traits.hh"

namespace amdzen::gen {

jitAmdZenGemmQuant::jitAmdZenGemmQuant()
    : mKernelDatatypes({ dlp::kernel_frame::kernelDatatype::s8s8s32of32,
                         dlp::kernel_frame::kernelDatatype::s8s8s32obf16,
                         dlp::kernel_frame::kernelDatatype::u8s8s32of32,
                         dlp::kernel_frame::kernelDatatype::u8s8s32obf16 })
    , mIsaFeaturesRequired({ dlp::cpu_utils::isaFeature::avx512vnni })
    , kType(utils::kernelInstrType::none)
    , numElemsPerReg(1) // Initialise to 1 to avoid div-by-zero.
{
    mIsaFeaturesRequired.push_back(dlp::cpu_utils::isaFeature::avx512f);
    mIsaFeaturesRequired.push_back(dlp::cpu_utils::isaFeature::avx512bw);
}

jitAmdZenGemmQuant::~jitAmdZenGemmQuant()
{
    codeGenerators.clear();
    for (auto& block : kernelCodeBlocks) {
        block = nullptr;
    }
}

int
jitAmdZenGemmQuant::getProcessBlockSize() const
{
    switch (kType) {
        case utils::kernelInstrType::avx512_zmm_32_reg:
            return static_cast<int>(NR);
        default:
            return 0;
    }
}

void
jitAmdZenGemmQuant::setGeneratorKernelMetaInfo(
    dlp::kernel_frame::kernelInstrPreference kInstPref)
{
    kType = utils::kernelInstrType::none;
    switch (kInstPref) {
        case dlp::kernel_frame::kernelInstrPreference::avx512_zmm_favour: {
            kType = utils::kernelInstrType::avx512_zmm_32_reg;
            numElemsPerReg =
                traits::ArchitectureTraits<
                    utils::kernelInstrType::avx512_zmm_32_reg>::regBytes
                / VNNI_CONST; // 4 int8s per VNNI dword.
            break;
        }

        default:
            break;
    }
}

dlp::jit::jitGeneratorError
jitAmdZenGemmQuant::generateAllKernels(
    const dlp::jit::gemmQuantJitGeneratorContext& jI)
{
    dlp::jit::jitGeneratorError err = dlp::jit::jitGeneratorError::error;

    const dlp::kernel_frame::quantKernelInfo& qKI = jI.kI;

    MR              = qKI.base.mr;
    NR              = qKI.base.nr;
    KC              = qKI.base.kc;
    K_UNROLL        = qKI.base.k_unroll;
    PREFETCH_C_DIST = qKI.base.prefetch_c_dist;
    c_downscale     = qKI.base.c_downscale;
    a_scale_type    = qKI.aQuant.scale.storeDt;
    b_scale_type    = qKI.bQuant.scale.storeDt;

    setGeneratorKernelMetaInfo(qKI.base.kInstPref);

    int processBlockSize = getProcessBlockSize();
    if (processBlockSize <= 0) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    numNRVariants     = (processBlockSize / numElemsPerReg) + 1;
    numMRVariants     = MR;
    numKernelVariants = numMRVariants * numNRVariants;

    kernelCodeBlocks.resize(numKernelVariants);

    // Seed generatorParams (vanilla microkernel) from the composed base.
    utils::generatorParams baseParams(
        0, 0, static_cast<int>(qKI.base.k_unroll),
        static_cast<int>(PREFETCH_C_DIST), c_downscale, 1, false, false, false,
        qKI.base.alphaScalingType, qKI.base.betaScalingType, kType);

    // Carry the post-full-K ops (base.kOpsArr)
    for (std::size_t ii = 0; ii < qKI.base.kOpsArrSize; ++ii) {
        baseParams.kernelOps.push_back(qKI.base.kOpsArr[ii]);
    }

    for (iter_t mr = 0; mr < numMRVariants; mr++) {
        for (iter_t nr = 0; nr < numNRVariants; nr++) {
            utils::quantGeneratorParams params(baseParams);

            params.base.MR    = (mr == 0) ? MR : mr;
            params.base.mLoop = (mr == 0);

            params.base.NR      = nr * numElemsPerReg;
            params.base.useMask = (nr == 0);

            params.base.numMaskRegs = (params.base.useMask) ? 1 : 0;

            params.aQuant = qKI.aQuant;
            params.bQuant = qKI.bQuant;

            std::unique_ptr<Xbyak::CodeGenerator> gen;
            switch (kType) {
                case utils::kernelInstrType::avx512_zmm_32_reg: {
                    auto g = std::make_unique<GEMMcodeGenerator::jitGEMMQuant<
                        utils::kernelInstrType::avx512_zmm_32_reg>>(
                        utils::JIT_KERNEL_SIZE);
                    err = g->generateKernel(params);
                    gen = std::move(g);
                    break;
                }
                default:
                    err = dlp::jit::jitGeneratorError::error;
                    break;
            }
            if (err != dlp::jit::jitGeneratorError::success) {
                goto cleanup;
            }

            // Readjust jump/branch targets after any AutoGrow reallocation.
            gen->ready();
            kernelCodeBlocks[mr * numNRVariants + nr] =
                const_cast<void*>(static_cast<const void*>(gen->getCode()));
            codeGenerators.push_back(std::move(gen));
        }
    }

    return dlp::jit::jitGeneratorError::success;

cleanup:
    codeGenerators.clear();
    for (auto& block : kernelCodeBlocks) {
        block = nullptr;
    }
    return err;
}

dlp::kernels::kernelError
jitAmdZenGemmQuant::executeKernel(dlp::kernels::kernelParams* _params)
{
    auto params = static_cast<dlp::kernels::gemmParams*>(_params);

    int processBlockSize = getProcessBlockSize();
    if (processBlockSize <= 0) {
        return dlp::kernels::kernelError::error;
    }

    md_t mFullPieces    = params->m / MR;
    md_t mPartialPieces = params->m % MR;

    params->kIterBP   = params->k / (K_UNROLL * VNNI_CONST);
    params->kLeft     = params->k % (K_UNROLL * VNNI_CONST);
    params->kLeftIter = params->kLeft / VNNI_CONST;
    params->kLeftRem  = params->kLeft % VNNI_CONST;
    params->kLeftmask = (1 << params->kLeftRem) - 1;

    // Precompute grouped-quant loop counts
    md_t group_size = params->grpKernelOpsAttr.group_size;
    if (group_size == 0) {
        group_size = params->k; // defensive; frame already normalizes
    }
    params->nIter   = (params->k + group_size - 1) / group_size; // num_groups
    params->nLeft   = params->grpKernelOpsAttr.grp_post_op_k / group_size;
    params->kIterAP = group_size / VNNI_CONST;

    // Trailing group may be a partial K-remainder group when group_size does
    // not divide this tile's K. Its full-VNNI count is floor(lastGroupK / 4);
    // lastGroupK % 4 == k % 4 == kLeftRem, so kLeftmask already covers its
    // sub-4 tail (group_size is a multiple of 4 for VNNI packing).
    md_t lastGroupK     = params->k - (params->nIter - 1) * group_size;
    params->kIterAPLast = lastGroupK / VNNI_CONST;

    int8_t*  bPtr = static_cast<int8_t*>(params->b);
    int32_t* cPtr = static_cast<int32_t*>(params->c);
    int32_t* c_jr = cPtr;
    md_t     rsB  = params->rsB;

    md_t n = params->n;

    params->psA = MR * params->psA;

    md_t og_post_op_c_i   = (params->kernelOpsAttr).post_op_c_i;
    md_t og_grp_post_op_i = (params->grpKernelOpsAttr).grp_post_op_i;

    md_t fringe_row_offset = mFullPieces * MR;

    md_t nBlockSize  = (n >= processBlockSize) ? processBlockSize : n;
    md_t nFullpieces = nBlockSize / numElemsPerReg;
    md_t nRemainder  = nBlockSize % numElemsPerReg;

    int8_t* aPtr = static_cast<int8_t*>(params->a);

    if (nFullpieces > 0) {
        md_t elementsToProcess = nFullpieces * numElemsPerReg;
        params->rsB            = (nFullpieces * rsB) / VNNI_CONST;

        params->a = aPtr;
        params->c = c_jr;
        params->n = elementsToProcess;

        md_t kernel_n_idx = nFullpieces;
        if (params->m >= MR) {
            params->mIter             = mFullPieces;
            int               ker_idx = 0 * numNRVariants + kernel_n_idx;
            utils::jit_kernel kernel =
                reinterpret_cast<utils::jit_kernel>(kernelCodeBlocks[ker_idx]);
            kernel(params);
        }
        if (mPartialPieces) {
            (params->a) = (int8_t*)(params->a) + mFullPieces * params->psA;
            (params->c) =
                (int32_t*)(params->c) + mFullPieces * MR * params->rsC;
            (params->grpKernelOpsAttr).grp_post_op_i =
                og_grp_post_op_i + fringe_row_offset;
            (params->kernelOpsAttr).post_op_c_i =
                og_post_op_c_i + fringe_row_offset;
            int ker_idx = mPartialPieces * numNRVariants + kernel_n_idx;
            utils::jit_kernel kernel =
                reinterpret_cast<utils::jit_kernel>(kernelCodeBlocks[ker_idx]);
            kernel(params);
            (params->grpKernelOpsAttr).grp_post_op_i = og_grp_post_op_i;
        }

        md_t k_updated =
            ((params->k + VNNI_CONST - 1) / VNNI_CONST) * VNNI_CONST;
        params->b = (int8_t*)(params->b) + elementsToProcess * k_updated;

        c_jr = (int32_t*)(c_jr) + elementsToProcess;
        (params->kernelOpsAttr).post_op_c_j += elementsToProcess;
        (params->kernelOpsAttr).b_sum_offset += nFullpieces * 16;
        (params->kernelOpsAttr).post_op_c_i = og_post_op_c_i;
        // grp_post_op_j is the column base the kernel adds to b_scale_factor,
        // so it advances with the tile split just like post_op_c_j and
        // b_sum_offset. Without this the masked remainder below would
        // re-read the B scales of this tile's leading columns.
        (params->grpKernelOpsAttr).grp_post_op_j += elementsToProcess;

        n -= elementsToProcess;
    }

    if (nRemainder > 0) {
        params->a = aPtr;
        params->c = c_jr;
        params->n = nRemainder;

        params->rsB = numElemsPerReg * VNNI_CONST;

        if (kType == utils::kernelInstrType::avx512_zmm_32_reg) {
            params->maskS32    = 0xFFFF >> (numElemsPerReg - nRemainder);
            params->maskF32[0] = params->maskS32;
        }

        int kernel_n_idx = 0; // mask kernel
        if (params->m >= MR) {
            params->mIter             = mFullPieces;
            int               ker_idx = 0 * numNRVariants + kernel_n_idx;
            utils::jit_kernel kernel =
                reinterpret_cast<utils::jit_kernel>(kernelCodeBlocks[ker_idx]);
            kernel(params);
        }
        if (mPartialPieces) {
            (params->a) = (int8_t*)(params->a) + mFullPieces * params->psA;
            (params->c) =
                (int32_t*)(params->c) + mFullPieces * MR * params->rsC;
            (params->grpKernelOpsAttr).grp_post_op_i =
                og_grp_post_op_i + fringe_row_offset;
            (params->kernelOpsAttr).post_op_c_i =
                og_post_op_c_i + fringe_row_offset;
            int ker_idx = mPartialPieces * numNRVariants + kernel_n_idx;
            utils::jit_kernel kernel =
                reinterpret_cast<utils::jit_kernel>(kernelCodeBlocks[ker_idx]);
            kernel(params);
            (params->grpKernelOpsAttr).grp_post_op_i = og_grp_post_op_i;
            (params->kernelOpsAttr).post_op_c_i      = og_post_op_c_i;
        }
    }

    return dlp::kernels::kernelError::success;
}

std::vector<dlp::kernel_frame::kernelDatatype>&
jitAmdZenGemmQuant::getKernelDatatypes()
{
    return mKernelDatatypes;
}

std::vector<dlp::cpu_utils::isaFeature>&
jitAmdZenGemmQuant::getIsaFeaturesRequired()
{
    return mIsaFeaturesRequired;
}

std::unique_ptr<dlp::jit::gemmQuantJitGenerator>
jitAmdZenGemmQuant::clone()
{
    return std::make_unique<jitAmdZenGemmQuant>();
}

DLP_REGISTER_STATIC_GEMM_QUANT_JIT_GENERATOR(jitAmdZenGemmQuant,
                                             "dlp_amdzen_gemm_quant_jit");

} // namespace amdzen::gen
