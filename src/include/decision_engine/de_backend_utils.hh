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

#include <cctype>
#include <tuple>

#include "classic/dlp_macros.h"
#include "de_input.hh"
#include "de_shape_model.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "utils/ctype_utils.hh"
#include "utils/float16_types.hh"

namespace dlp::de {

// Leaf helpers for the shape model: the questions it asks about an input, and
// the tile that stands when it has nothing to choose.
//
// They sit here rather than with the model because they decide nothing. Each
// is a pure function of one input, with no candidate set and no dispatch, so a
// backend that overrides the model still asks these the same questions. What
// the model does with the answers is in optimizer/de_optimizer.hh.
class gemmShapeModelUtils
{
  public:
    // True when both halves of the tile are already settled, leaving the model
    // nothing to choose. One of the two is not enough: the other is still the
    // model's to pick, under the settled one as a constraint.
    DLP_ALWAYS_INLINE static bool isTilePinnedByCaller(
        const shape_model::gemmShapeModelInput& in)
    {
        constexpr md_t both = DLP_BLKSZ_SET_MR | DLP_BLKSZ_SET_NR;
        return (in.frozen & both) == both;
    }

    // True when the model may choose a tile from this input.
    //
    // The mask is the caller's say in the matter; there is no separate opt-in
    // flag. A bit is set either because the application authored that block
    // size or because an earlier init in the same call settled it, and both
    // mean the value is not the model's to choose. That is what keeps the
    // kernel init and the pack-B init from having to know which runs first:
    // whichever gets there first marks what it settled, and the other honours
    // the marks it finds.
    //
    // An m and a pool are also required, read straight off the input, which
    // already describes the GEMM the tile is being chosen for. A zero in either
    // is the application declining to describe it.
    //
    // The pool is stated one of two ways and either will do. A count is the
    // usual one. A pinned pair of ways is the other: ways outrank the count in
    // the runtime's precedence, so a pinned call arrives with the count at -1
    // and both ways filled, a half-stated pin having been completed to 1 x n
    // before it got here. Reading that -1 as an absent pool would refuse the
    // sweep exactly where the partition is already settled and a candidate can
    // be ranked against it directly.
    //
    // The memory tag of A is deliberately not part of this. The intrinsic
    // pack-A writes a plain row-major MC x KC image: every arm of its unroll
    // ladder stores to (ic + j) * KC + kr and reports rs_p = KC, so the packed
    // buffer carries no MR-shaped structure that a later MR could contradict.
    //
    // The architecture half of the screen is resolved once in the backend
    // constructor.
    DLP_ALWAYS_INLINE static bool isEligible(
        const shape_model::gemmShapeModelInput& in)
    {
        const bool poolStated = (in.num_threads > 0)
                                || ((in.ic_ways > 0) && (in.jc_ways > 0));

        return !isTilePinnedByCaller(in) && (in.m > 0) && poolStated;
    }

    // True when the application pinned MR through metadata. Honoured on every
    // path, modelled or not. This needs the provenance mask because by the time
    // a block size reaches the model, a stated 6 and a default 6 are the same
    // number.
    DLP_ALWAYS_INLINE static bool isMRFixedByCaller(
        const shape_model::gemmShapeModelInput& in)
    {
        return (in.frozen & DLP_BLKSZ_SET_MR) != 0;
    }

    // The same, for NR. Note what this does not cover: an NR that a Reorder
    // settled on. Nothing records that width, so it is not frozen in this
    // sense. See sweepCandidates for how the two ends agree on it instead.
    DLP_ALWAYS_INLINE static bool isNRFixedByCaller(
        const shape_model::gemmShapeModelInput& in)
    {
        return (in.frozen & DLP_BLKSZ_SET_NR) != 0;
    }

    // Whether the tunables admit this candidate into the sweep.
    //
    // A frozen dimension narrows the candidate set rather than being imposed
    // on the winner afterwards. Tiles are costed as pairs, so freezing one half
    // is a statement about which pairs are worth costing at all. Freezing both
    // leaves nothing to choose, and isEligible screens that out before the
    // sweep is reached.
    DLP_ALWAYS_INLINE static bool isCandidateAdmissible(
        const shape_model::gemmShapeModelInput& in,
        const shape_model::kernelDims&          cand)
    {
        if (isMRFixedByCaller(in) && (cand.mr != in.mr)) {
            return false;
        }
        if (isNRFixedByCaller(in) && (cand.nr != in.nr)) {
            return false;
        }
        return true;
    }

    // The tile that stands when the model has no say: every ineligible call,
    // and the fallback inside the modelled arm.
    //
    // It is just the context tile. NR comes back exactly as it was found, so
    // both ends of a Reorder fall back to the same width.
    //
    // Nothing here may size itself against an extent. This object may describe
    // the GEMM the hints characterise rather than the call, so a rule needing
    // the rows that will actually be loaded belongs in the kernel-info fold,
    // which has them.
    DLP_ALWAYS_INLINE static shape_model::kernelDims baselineKernelDims(
        const shape_model::gemmShapeModelInput& in)
    {
        return shape_model::kernelDims{ in.mr, in.nr };
    }
};

class gemmDEBackendUtils
{
    static void setKernelOps(kernel_frame::kernelOpsMetaData* metaData,
                             dlp_gemm_post_op*                post_op,
                             kernel_frame::kernelDatatype     k_dtype);

    DLP_ALWAYS_INLINE
    static std::pair<kernel_frame::scalingType, kernel_frame::scalingType>
    getScalingTypesF32(void* alpha, void* beta, md_t k, md_t kc_hint)
    {
        float                     alpha_val = *(static_cast<float*>(alpha));
        kernel_frame::scalingType alphaScalingType =
            kernel_frame::scalingType::generic;
        if (alpha_val == 1.0f) {
            alphaScalingType = kernel_frame::scalingType::one;
        } else if (alpha_val == 0.0f) {
            alphaScalingType = kernel_frame::scalingType::zero;
        }

        kernel_frame::scalingType betaScalingType =
            kernel_frame::scalingType::generic;
        if ((*(static_cast<float*>(beta)) == 0.0f) && (k <= kc_hint)) {
            betaScalingType = kernel_frame::scalingType::zero;
        }

        return std::make_pair(alphaScalingType, betaScalingType);
    }

    DLP_ALWAYS_INLINE
    static std::pair<kernel_frame::scalingType, kernel_frame::scalingType>
    getScalingTypesInt32(void* alpha, void* beta, md_t k, md_t kc_hint)
    {
        int32_t                   alpha_val = *(static_cast<int32_t*>(alpha));
        kernel_frame::scalingType alphaScalingType =
            kernel_frame::scalingType::generic;
        if (alpha_val == 1) {
            alphaScalingType = kernel_frame::scalingType::one;
        } else if (alpha_val == 0) {
            alphaScalingType = kernel_frame::scalingType::zero;
        }

        kernel_frame::scalingType betaScalingType =
            kernel_frame::scalingType::generic;
        if ((*(static_cast<int*>(beta)) == 0) && (k <= kc_hint)) {
            betaScalingType = kernel_frame::scalingType::zero;
        }

        return std::make_pair(alphaScalingType, betaScalingType);
    }

    DLP_ALWAYS_INLINE
    static std::pair<kernel_frame::scalingType, kernel_frame::scalingType>
    getScalingTypesFP16(void* alpha, void* beta, md_t k, md_t kc_hint)
    {
        uint16_t alpha_val = *(static_cast<uint16_t*>(alpha));

        kernel_frame::scalingType alphaScalingType =
            kernel_frame::scalingType::generic;
        if (alpha_val == FP16_ONE) {
            alphaScalingType = kernel_frame::scalingType::one;
        } else if (alpha_val == FP16_ZERO) {
            alphaScalingType = kernel_frame::scalingType::zero;
        }

        kernel_frame::scalingType betaScalingType =
            kernel_frame::scalingType::generic;
        uint16_t beta_val = *(static_cast<uint16_t*>(beta));
        if (beta_val == FP16_ONE) {
            /* Safe for both single-KC and multi-KC: user beta = 1.0 means
               every KC iteration has beta = 1.0 (the 5-loop passes 1.0 for
               intermediate KCs either way), so the same JIT `::one`
               fast-path is correct across all iterations. */
            betaScalingType = kernel_frame::scalingType::one;
        } else if ((beta_val == FP16_ZERO) && (k <= kc_hint)) {
            /* Only emit `::zero` for single-KC (k <= kc_hint). Multi-KC
               beta = 0 needs beta = 1.0 on intermediate iterations to
               preserve accumulation, which a JIT built with `::zero`
               would overwrite; keep it `::generic` in that case. */
            betaScalingType = kernel_frame::scalingType::zero;
        }

        return std::make_pair(alphaScalingType, betaScalingType);
    }

  public:
    template<typename T>
    DLP_ALWAYS_INLINE static std::pair<kernel_frame::scalingType,
                                       kernel_frame::scalingType>
    getScalingTypes(void* alpha, void* beta, md_t k, md_t kc_hint)
    {
        if constexpr (std::is_same_v<float, T>) {
            return gemmDEBackendUtils::getScalingTypesF32(alpha, beta, k,
                                                          kc_hint);
        } else if constexpr (std::is_same_v<int32_t, T>) {
            return gemmDEBackendUtils::getScalingTypesInt32(alpha, beta, k,
                                                            kc_hint);
        } else if constexpr (std::is_same_v<dlp::float16, T>) {
            return gemmDEBackendUtils::getScalingTypesFP16(alpha, beta, k,
                                                           kc_hint);
        }

        return std::make_pair(kernel_frame::scalingType::generic,
                              kernel_frame::scalingType::generic);
    }

    DLP_ALWAYS_INLINE
    static kernel_frame::kernelInfo checkPostOpsAndCreateKernelInfo(
        md_t                                mr,
        md_t                                nr,
        md_t                                term_fringe_nr,
        md_t                                k_unroll,
        md_t                                kc,
        md_t                                prefetch_c_dist,
        kernel_frame::scalingType           alphaScalingType,
        kernel_frame::scalingType           betaScalingType,
        AOCL_DLP_MEMORY_TAG                 mtag_a,
        AOCL_DLP_MEMORY_TAG                 mtag_b,
        bool                                allLtFringeKernels,
        bool                                invokeRD,
        bool                                anyKOpsOrder,
        kernel_frame::kernelInstrPreference kInstPref,
        md_t                                c_downscale,
        dlp::kernel_frame::kernelDatatype   k_dtype,
        [[maybe_unused]] md_t               rs_c,
        [[maybe_unused]] md_t               cs_c,
        dlp_gemm_post_op*                   metadata,
        bool                                skinnyN      = false,
        bool                                aliasMrSplit = false)
    {
        // Iterate over the post_ops list to get the number of post-ops.
        md_t              numPostOps    = 0;
        dlp_gemm_post_op* temp_post_ops = metadata;
        while ((temp_post_ops != nullptr)
               && (temp_post_ops->op_code != POST_OPS_DISABLE)) {
            temp_post_ops = temp_post_ops->next;
            numPostOps++;
        }

        if (numPostOps == 0) {
            return kernel_frame::kernelInfo{ mr,
                                             nr,
                                             term_fringe_nr,
                                             k_unroll,
                                             kc,
                                             prefetch_c_dist,
                                             alphaScalingType,
                                             betaScalingType,
                                             mtag_a,
                                             mtag_b,
                                             allLtFringeKernels,
                                             invokeRD,
                                             nullptr,
                                             0,
                                             anyKOpsOrder,
                                             kInstPref,
                                             c_downscale,
                                             skinnyN,
                                             aliasMrSplit };
        } else {
            kernel_frame::kernelInfo kI{ mr,
                                         nr,
                                         term_fringe_nr,
                                         k_unroll,
                                         kc,
                                         prefetch_c_dist,
                                         alphaScalingType,
                                         betaScalingType,
                                         mtag_a,
                                         mtag_b,
                                         allLtFringeKernels,
                                         invokeRD,
                                         nullptr,
                                         0,
                                         anyKOpsOrder,
                                         kInstPref,
                                         c_downscale,
                                         skinnyN,
                                         aliasMrSplit };
            kI.kOpsArrSize = numPostOps;
            kI.kOpsArr =
                kernel_frame::kernelInfo::allocateKernelOpsArray(numPostOps);

            md_t ii       = 0;
            temp_post_ops = metadata;
            while ((temp_post_ops != nullptr)
                   && (temp_post_ops->op_code != POST_OPS_DISABLE)) {
                setKernelOps(std::addressof(kI.kOpsArr[ii]), temp_post_ops,
                             k_dtype);
                temp_post_ops = temp_post_ops->next;
                ii++;
            }

            return kI;
        }
    }

    // Translate the C API's granularity enum into the kernel-frame one. The
    // two happen to share numeric values today, but a cast would silently
    // produce an out-of-range ParamDim the moment either enum gains a member,
    // so map every case explicitly and fall back to Invalid, which code-gen
    // already rejects with badKernelInfo.
    DLP_ALWAYS_INLINE
    static kernel_frame::ParamDim mapParamDim(DLP_PARAM_DIM_TYPE dim)
    {
        switch (dim) {
            case DLP_PARAM_DIM_PER_TENSOR:
                return kernel_frame::ParamDim::Scalar;
            case DLP_PARAM_DIM_PER_CHANNEL:
                return kernel_frame::ParamDim::PerN;
            case DLP_PARAM_DIM_PER_TOKEN:
                return kernel_frame::ParamDim::PerM;
            case DLP_PARAM_DIM_PER_GROUP:
                return kernel_frame::ParamDim::PerGroup;
            case DLP_PARAM_DIM_INVALID:
            default:
                return kernel_frame::ParamDim::Invalid;
        }
    }

    // Whether a parameter array is tiled along K, mirroring the a_grp_mul /
    // b_grp_mul derivation in dlp_gemm_s8s8s32_sym_quant.c exactly: A collapses
    // the K axis only when PER_TOKEN, B only when PER_CHANNEL, and every other
    // granularity (including PER_TENSOR) is stored fully tiled. Deriving this
    // from the array length instead would misclassify a PER_TOKEN A scale,
    // whose length is M rather than 1.
    DLP_ALWAYS_INLINE
    static bool isTiledAlongK(DLP_PARAM_DIM_TYPE dim, bool isBOperand)
    {
        const DLP_PARAM_DIM_TYPE collapsing =
            isBOperand ? DLP_PARAM_DIM_PER_CHANNEL : DLP_PARAM_DIM_PER_TOKEN;
        return dim != collapsing;
    }

    DLP_ALWAYS_INLINE
    static kernel_frame::quantKernelInfo checkPostOpsAndCreateQuantKernelInfo(
        md_t                                mr,
        md_t                                nr,
        md_t                                term_fringe_nr,
        md_t                                k_unroll,
        md_t                                kc,
        md_t                                prefetch_c_dist,
        kernel_frame::scalingType           alphaScalingType,
        kernel_frame::scalingType           betaScalingType,
        AOCL_DLP_MEMORY_TAG                 mtag_a,
        AOCL_DLP_MEMORY_TAG                 mtag_b,
        bool                                allLtFringeKernels,
        bool                                invokeRD,
        bool                                anyKOpsOrder,
        kernel_frame::kernelInstrPreference kInstPref,
        md_t                                c_downscale,
        dlp::kernel_frame::kernelDatatype   k_dtype,
        [[maybe_unused]] md_t               rs_c,
        [[maybe_unused]] md_t               cs_c,
        dlp_gemm_post_op*                   metadata,
        dlp_group_op*                       group_ops,
        bool                                skinnyN      = false,
        bool                                aliasMrSplit = false)
    {
        kernel_frame::quantKernelInfo qKI;

        // Quant kernel info creation needs base kernelInfo fields, aQuant and
        // bQuant fields. Reuse the kernelInfo creation to fill the base
        // kernelInfo fields.
        qKI.base = gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, term_fringe_nr, k_unroll, kc, prefetch_c_dist,
            alphaScalingType, betaScalingType, mtag_a, mtag_b,
            allLtFringeKernels, invokeRD, anyKOpsOrder, kInstPref, c_downscale,
            k_dtype, rs_c, cs_c, metadata, skinnyN, aliasMrSplit);

        // Handle group ops.
        // Fill the aQuant field.
        kernel_frame::opQuantInfo* aQuant = &qKI.aQuant;
        dlp_quant_op_t*            a_pqo  = group_ops->a_post_quant_op;
        aQuant->src_type = (kernel_frame::DataType)a_pqo->src_type;
        aQuant->dst_type = (kernel_frame::DataType)a_pqo->dst_type;
        // Hardcoded to dequantInKernel for group quantized kernels.
        aQuant->mode = kernel_frame::opQuantMode::dequantInKernel;
        aQuant->scale.storeDt =
            (kernel_frame::DataType)a_pqo->dequant_scale_factors->stor_type;
        aQuant->scale.outerDim =
            mapParamDim(a_pqo->dequant_scale_factors->outer_dim);
        aQuant->scale.perGroupK =
            isTiledAlongK(a_pqo->dequant_scale_factors->outer_dim, false);
        // Leave zeroPoint at its default(null) for symmetric quantization
        if (a_pqo->zero_point != nullptr) {
            aQuant->zeroPoint.storeDt =
                (kernel_frame::DataType)a_pqo->zero_point->stor_type;
            aQuant->zeroPoint.outerDim =
                mapParamDim(a_pqo->zero_point->outer_dim);
            aQuant->zeroPoint.perGroupK =
                isTiledAlongK(a_pqo->zero_point->outer_dim, false);
        }

        // Fill the bQuant field.
        kernel_frame::opQuantInfo* bQuant = &qKI.bQuant;
        dlp_quant_op_t*            b_pqo  = group_ops->b_post_quant_op;
        bQuant->src_type = (kernel_frame::DataType)b_pqo->src_type;
        bQuant->dst_type = (kernel_frame::DataType)b_pqo->dst_type;
        // Hardcoded to dequantInKernel for group quantized kernels.
        bQuant->mode = kernel_frame::opQuantMode::dequantInKernel;

        bQuant->scale.storeDt =
            (kernel_frame::DataType)b_pqo->dequant_scale_factors->stor_type;
        bQuant->scale.outerDim =
            mapParamDim(b_pqo->dequant_scale_factors->outer_dim);
        bQuant->scale.perGroupK =
            isTiledAlongK(b_pqo->dequant_scale_factors->outer_dim, true);
        // Leave zeroPoint at its default(null) for symmetric quantization
        if (b_pqo->zero_point != nullptr) {
            bQuant->zeroPoint.storeDt =
                (kernel_frame::DataType)b_pqo->zero_point->stor_type;
            bQuant->zeroPoint.outerDim =
                mapParamDim(b_pqo->zero_point->outer_dim);
            bQuant->zeroPoint.perGroupK =
                isTiledAlongK(b_pqo->zero_point->outer_dim, true);
        }

        return qKI;
    }
};

} // namespace dlp::de
