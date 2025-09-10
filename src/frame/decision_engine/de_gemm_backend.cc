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

#include <cctype>

#include "arch_utils/arch_config_manager.hh"
#include "cpu_utils/cpu_features.hh"
#include "decision_engine/de_backend.hh"
#include "env_utils/env_var_manager.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "utils/ctype_utils.hh"

namespace dlp::de {

gemmF32DEBackend::gemmF32DEBackend()
    : isAvx512(false)
    , isAvx2(false)
    , canGenerateKernelInfo(true)
{
    // Use this eKernelInstPref to generate kernelInfo for the 32 Ymm register
    // based f32 kernel generation.
    eKernelInstPref =
        dlp::env_utils::EnvironmentVariableManager::getInstance()
            .getKernelInstructionPreferenceFromEnv("AOCL_ENABLE_INSTRUCTIONS");

    isAvx512 =
        arch_utils::archConfigManager::getInstance().isAvx512SupportedByArch();

    // isAvx2 is only checked for and set to true if machine is guaranteed
    // to not support Avx512 isa.
    if (!isAvx512) {
        isAvx2 = arch_utils::archConfigManager::getInstance()
                     .isAvx2Fma3SupportedByArch();
    }

    // Check if the DE can be enabled depending on the underlying ISA
    // and the configured arch.
    auto thisArch = arch_utils::archConfigManager::getInstance().getArch();
    if (isAvx512 && (thisArch != arch_utils::ArchitectureType::Zen5)
        && (thisArch != arch_utils::ArchitectureType::Zen4)
        && (thisArch != arch_utils::ArchitectureType::GenericAvx512Bf16)
        && (thisArch != arch_utils::ArchitectureType::GenericAvx512Vnni)
        && (thisArch != arch_utils::ArchitectureType::GenericAvx512)) {
        // Switch to Avx2 if the arch is selected as such.
        if ((thisArch != arch_utils::ArchitectureType::Generic)
            && (thisArch != arch_utils::ArchitectureType::Error)) {
            isAvx2   = true;
            isAvx512 = false;
        } else {
            // This is an invalid case, disable jit kernel generation.
            canGenerateKernelInfo = false;
        }
    }
    if (isAvx2
        && ((thisArch == arch_utils::ArchitectureType::Generic)
            || (thisArch == arch_utils::ArchitectureType::Error))) {
        canGenerateKernelInfo = false;
    }
}

void
gemmF32DEBackend::setKernelOps(kernel_frame::kernelOpsMetaData* metaData,
                               lpgemm_post_op*                  post_op,
                               kernel_frame::kernelDatatype     k_dtype)
{
    kernel_frame::kernelOps kOpsType = kernel_frame::kernelOps::invalid;
    switch (post_op->op_code) {
        case POST_OPS_BIAS: {
            metaData->type           = kernel_frame::kernelOps::bias;
            metaData->paramStorageDt = utils::getStorageDtFromAoclStorageType(
                static_cast<DLP_TYPE>(post_op->stor_type));
            char storFormatC =
                std::tolower(*(static_cast<char*>(post_op->op_args2)));
            metaData->cMatFormat = (storFormatC == 'c')
                                       ? kernel_frame::storageFormat::colMajor
                                       : kernel_frame::storageFormat::rowMajor;
            break;
        }
        case POST_OPS_RELU:
            metaData->type = kernel_frame::kernelOps::relu;
            break;
        case POST_OPS_RELU_SCALE:
            metaData->type = kernel_frame::kernelOps::reluScale;
            metaData->paramStorageDt =
                utils::getStorageDtFromDlpKernelDatatype(k_dtype);
            break;
        case POST_OPS_GELU_TANH:
            metaData->type = kernel_frame::kernelOps::geluTanh;
            break;
        case POST_OPS_GELU_ERF:
            metaData->type = kernel_frame::kernelOps::geluErf;
            break;
        case POST_OPS_CLIP:
            metaData->type = kernel_frame::kernelOps::clip;
            metaData->paramStorageDt =
                utils::getStorageDtFromDlpKernelDatatype(k_dtype);
            break;
        case POST_OPS_SWISH:
            metaData->type = kernel_frame::kernelOps::swish;
            metaData->paramStorageDt =
                utils::getStorageDtFromDlpKernelDatatype(k_dtype);
            break;
        case POST_OPS_TANH:
            metaData->type = kernel_frame::kernelOps::tanh;
            break;
        case POST_OPS_SIGMOID:
            metaData->type = kernel_frame::kernelOps::sigmoid;
            break;
        case POST_OPS_DOWNSCALE: {
            metaData->type = kernel_frame::kernelOps::downscale;
            char storFormatC =
                std::tolower(*(static_cast<char*>(post_op->op_args2)));
            metaData->cMatFormat    = (storFormatC == 'c')
                                          ? kernel_frame::storageFormat::colMajor
                                          : kernel_frame::storageFormat::rowMajor;
            metaData->scaleFactorDt = utils::getStorageDtFromAoclStorageType(
                static_cast<DLP_TYPE>(post_op->sf_stor_type));
            metaData->scalarScaleFactorRequired =
                (post_op->scale_factor_len == 1) ? true : false;
            metaData->vectorScaleFactorRequired =
                (post_op->scale_factor_len > 1) ? true : false;
            metaData->zeroPointDt = utils::getStorageDtFromAoclStorageType(
                static_cast<DLP_TYPE>(post_op->zp_stor_type));
            metaData->scalarZeroPointRequired =
                (*(static_cast<md_t*>(post_op->op_args3)) == 1) ? true : false;
            metaData->vectorZeroPointRequired =
                (*(static_cast<md_t*>(post_op->op_args3)) > 1) ? true : false;
            break;
        }
        case POST_OPS_MATRIX_ADD: {
            metaData->type           = kernel_frame::kernelOps::matAdd;
            metaData->paramStorageDt = utils::getStorageDtFromAoclStorageType(
                static_cast<DLP_TYPE>(post_op->stor_type));
            char storFormatC =
                std::tolower(*(static_cast<char*>(post_op->op_args2)));
            metaData->cMatFormat = (storFormatC == 'c')
                                       ? kernel_frame::storageFormat::colMajor
                                       : kernel_frame::storageFormat::rowMajor;
            metaData->scaleFactorDt =
                kernel_frame::DataType::f32; // TODO: Always F32 for mat add
            metaData->scalarScaleFactorRequired =
                (post_op->scale_factor_len == 1) ? true : false;
            metaData->vectorScaleFactorRequired =
                (post_op->scale_factor_len > 1) ? true : false;
            break;
        }
        case POST_OPS_MATRIX_MUL: {
            metaData->type           = kernel_frame::kernelOps::matMul;
            metaData->paramStorageDt = utils::getStorageDtFromAoclStorageType(
                static_cast<DLP_TYPE>(post_op->stor_type));
            char storFormatC =
                std::tolower(*(static_cast<char*>(post_op->op_args2)));
            metaData->cMatFormat = (storFormatC == 'c')
                                       ? kernel_frame::storageFormat::colMajor
                                       : kernel_frame::storageFormat::rowMajor;
            metaData->scaleFactorDt =
                kernel_frame::DataType::f32; // TODO: Always F32 for mat mul
            metaData->scalarScaleFactorRequired =
                (post_op->scale_factor_len == 1) ? true : false;
            metaData->vectorScaleFactorRequired =
                (post_op->scale_factor_len > 1) ? true : false;
            break;
        }
        case POST_OPS_DISABLE:
            metaData->type = kernel_frame::kernelOps::invalid;
            break;
        default:
            metaData->type = kernel_frame::kernelOps::invalid;
            break;
    }
}

std::optional<kernel_frame::kernelInfo>
gemmF32DEBackend::getKernelInfoForInput(iDEInput* in)
{
    auto gemmIn = static_cast<gemmDEInput*>(in);
    if ((gemmIn == nullptr) || (!canGenerateKernelInfo)) {
        return std::nullopt;
    }

    kernel_frame::scalingType alphaScalingType =
        kernel_frame::scalingType::generic;
    kernel_frame::scalingType betaScalingType =
        kernel_frame::scalingType::generic;

    md_t            mr           = gemmIn->mr_hint;
    md_t            nr           = gemmIn->nr_hint;
    md_t            k_unroll     = 1;
    md_t            kc           = gemmIn->kc_hint;
    AOCL_MEMORY_TAG mtag_a       = gemmIn->mtag_a;
    AOCL_MEMORY_TAG mtag_b       = gemmIn->mtag_b;
    bool            anyKOpsOrder = false;

    // Set the kernel instruction preference based on the CPU features.
    // NOTE : This could be overridden by the user in future
    //        Ex : Wanting to run AVX2 kernels on Zen4, based on
    //             AOCL_ENABLE_INSTRUCTIONS
    kernel_frame::kernelInstrPreference kInstPref =
        kernel_frame::kernelInstrPreference::none;
    if (isAvx512) {
        // TODO: Use eKernelInstPref here to set the right kernel type.
        kInstPref = kernel_frame::kernelInstrPreference::avx512_zmm_favour;
    } else if (isAvx2) {
        kInstPref = kernel_frame::kernelInstrPreference::avx2_ymm_favour;
    }

    if (gemmIn->n == 1) {

        if (isAvx2) {
            return std::nullopt; // We resort to using classic AVX2 kernels on
                                 // Zen machines(for now).
        }

        else if (isAvx512) {
            // NOTE : Since the standard interface send the post-ops
            //        list, even with no-post ops, we will still have
            //        one node which mentions the opcode as POST_OPS_DISABLE.
            //        The null-pointer check is purely defensive.

            mr       = 16;
            nr       = 1;
            k_unroll = 1; // k-unroll is 1 for GEMV N1

            if (gemmIn->metadata == nullptr) {
                kernel_frame::kernelInfo kI{ mr,
                                             nr,
                                             k_unroll,
                                             kc,
                                             alphaScalingType,
                                             betaScalingType,
                                             mtag_a,
                                             mtag_b,
                                             nullptr,
                                             0,
                                             anyKOpsOrder,
                                             kInstPref };
                return std::make_optional(kI);
            } else if (gemmIn->metadata[0].op_code == POST_OPS_DISABLE) {
                // This condition is not combined with the previous 'if' clause,
                // since we don't want unfriendly short-curcuiting.
                // Ex : Hypothetically, if gemmIn->metadata is NULL, then we
                //      should ensure that this condition is strictly evaluated
                //      after null check.
                kernel_frame::kernelInfo kI{ mr,
                                             nr,
                                             k_unroll,
                                             kc,
                                             alphaScalingType,
                                             betaScalingType,
                                             mtag_a,
                                             mtag_b,
                                             nullptr,
                                             0,
                                             anyKOpsOrder,
                                             kInstPref };
                return std::make_optional(kI);

            } else {
                // We generate JIT based AVX512 GEMV N1 kernels only when
                // post-ops is not required.
                return std::nullopt;
            }
        } else {
            // If the machine is neither Zen nor Zen4, then we don't support
            // GEMV N1 kernels.
            return std::nullopt;
        }
    } else if (gemmIn->m == 1) {
        if (isAvx2) {
            return std::nullopt;
        } else if (isAvx512) {
            // NOTE : Since the standard interface send the post-ops
            //        list, even with no-post ops, we will still have
            //        one node which mentions the opcode as POST_OPS_DISABLE.
            //        The null-pointer check is purely defensive.

            mr       = 1;
            nr       = 64;
            k_unroll = 1;   // k-unroll is 1 for GEMV N1
            kc       = 512; // This is harcoded from ZEN4 context.

            if (gemmIn->metadata == nullptr) {
                kernel_frame::kernelInfo kI{ mr,
                                             nr,
                                             k_unroll,
                                             kc,
                                             alphaScalingType,
                                             betaScalingType,
                                             mtag_a,
                                             mtag_b,
                                             nullptr,
                                             0,
                                             anyKOpsOrder,
                                             kInstPref };
                return std::make_optional(kI);
            } else if (gemmIn->metadata[0].op_code == POST_OPS_DISABLE) {
                // This condition is not combined with the previous 'if' clause,
                // since we don't want unfriendly short-curcuiting.
                // Ex : Hypothetically, if gemmIn->metadata is NULL, then we
                //      should ensure that this condition is strictly evaluated
                //      after null check.
                kernel_frame::kernelInfo kI{ mr,
                                             nr,
                                             k_unroll,
                                             kc,
                                             alphaScalingType,
                                             betaScalingType,
                                             mtag_a,
                                             mtag_b,
                                             nullptr,
                                             0,
                                             anyKOpsOrder,
                                             kInstPref };
                return std::make_optional(kI);
            } else {
                return std::nullopt;
            }
        } else {
            return std::nullopt;
        }
    } else {
        if (gemmIn->metadata == nullptr) {
            kernel_frame::kernelInfo kI{
                mr,     nr,     k_unroll, kc, alphaScalingType, betaScalingType,
                mtag_a, mtag_b, nullptr,  0,  anyKOpsOrder,     kInstPref
            };
            return std::make_optional(kI);
        } else {
            // Iterate over the post_ops list to get the number of post-ops.
            md_t            numPostOps    = 0;
            lpgemm_post_op* temp_post_ops = gemmIn->metadata;
            while ((temp_post_ops != NULL)
                   && (temp_post_ops->op_code != POST_OPS_DISABLE)) {
                temp_post_ops = temp_post_ops->next;
                numPostOps++;
            }

            if (numPostOps == 0) {
                kernel_frame::kernelInfo kI{ mr,
                                             nr,
                                             k_unroll,
                                             kc,
                                             alphaScalingType,
                                             betaScalingType,
                                             mtag_a,
                                             mtag_b,
                                             nullptr,
                                             0,
                                             anyKOpsOrder,
                                             kInstPref };
                return std::make_optional(kI);
            } else {
                kernel_frame::kernelInfo kI{ mr,
                                             nr,
                                             k_unroll,
                                             kc,
                                             alphaScalingType,
                                             betaScalingType,
                                             mtag_a,
                                             mtag_b,
                                             nullptr,
                                             0,
                                             anyKOpsOrder,
                                             kInstPref };
                kI.kOpsArrSize = numPostOps;
                kI.kOpsArr = kernel_frame::kernelInfo::allocateKernelOpsArray(
                    numPostOps);

                md_t ii       = 0;
                temp_post_ops = gemmIn->metadata;
                while ((temp_post_ops != NULL)
                       && (temp_post_ops->op_code != POST_OPS_DISABLE)) {
                    setKernelOps(std::addressof(kI.kOpsArr[ii]), temp_post_ops,
                                 gemmIn->k_dtype);
                    temp_post_ops = temp_post_ops->next;
                    ii++;
                }

                return std::make_optional(kI);
            }
        }
        return std::nullopt;
    }
    return std::nullopt;
}

} // namespace dlp::de
