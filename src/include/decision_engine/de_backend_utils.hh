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
        kernel_frame::scalingType alphaScalingType =
            kernel_frame::scalingType::generic;
        if (*(static_cast<float*>(alpha)) == 1.0f) {
            alphaScalingType = kernel_frame::scalingType::one;
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
        kernel_frame::scalingType alphaScalingType =
            kernel_frame::scalingType::generic;
        if (*(static_cast<int*>(alpha)) == 1) {
            alphaScalingType = kernel_frame::scalingType::one;
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
        bool                                skinnyN = false)
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
                                             skinnyN };
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
                                         skinnyN };
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
};

} // namespace dlp::de
