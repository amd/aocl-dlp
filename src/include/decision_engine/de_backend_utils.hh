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
#include "kernel_frame/kernel_frame_base.hh"
#include "utils/ctype_utils.hh"
#include "utils/float16_types.hh"

namespace dlp::de {

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

        // Reuse the kernelInfo creation to fill the base kernelInfo fields.
        qKI.base = gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, term_fringe_nr, k_unroll, kc, prefetch_c_dist,
            alphaScalingType, betaScalingType, mtag_a, mtag_b,
            allLtFringeKernels, invokeRD, anyKOpsOrder, kInstPref, c_downscale,
            k_dtype, rs_c, cs_c, metadata, skinnyN, aliasMrSplit);

        // Handle group ops.
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
