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

namespace dlp::de {

class gemmDEBackendUtils
{
    static void setKernelOps(kernel_frame::kernelOpsMetaData* metaData,
                             lpgemm_post_op*                  post_op,
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
        AOCL_MEMORY_TAG                     mtag_a,
        AOCL_MEMORY_TAG                     mtag_b,
        bool                                allLtFringeKernels,
        bool                                invokeRD,
        bool                                anyKOpsOrder,
        kernel_frame::kernelInstrPreference kInstPref,
        md_t                                c_downscale,
        dlp::kernel_frame::kernelDatatype   k_dtype,
        md_t                                rs_c,
        md_t                                cs_c,
        lpgemm_post_op*                     metadata)
    {
        // Iterate over the post_ops list to get the number of post-ops.
        md_t            numPostOps    = 0;
        lpgemm_post_op* temp_post_ops = metadata;
        while ((temp_post_ops != nullptr)
               && (temp_post_ops->op_code != POST_OPS_DISABLE)) {
            temp_post_ops = temp_post_ops->next;
            numPostOps++;
        }

        if (numPostOps == 0) {
            return kernel_frame::kernelInfo{ mr,
                                             nr,
                                             0,
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
                                             c_downscale };
        } else {
            kernel_frame::kernelInfo kI{ mr,
                                         nr,
                                         0,
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
                                         c_downscale };
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
