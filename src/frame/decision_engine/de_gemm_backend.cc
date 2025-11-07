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
#include <memory>

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

    // This needs to be downgraded if the actual cpu is avx512 but is
    // downgraded via using AOCL_ENABLE_INSTRUCTIONS.
    numRegisters = cpu_utils::cpuFeaturesInstance().getNumVectorRegisters();

    // A hack to carry forward the inconsistency in the classic framework.
    // With this, we force the AVX512_256 path even when the env var is set to
    // AVX2. This hack should be removed in future, once we change the "classic"
    // framework.
    if (isAvx512
        && (eKernelInstPref
            == kernel_frame::kernelInstrPreference::avx2_ymm_favour)) {
        eKernelInstPref =
            kernel_frame::kernelInstrPreference::avx512_ymm_favour;
    }

    // Check if the DE can be enabled depending on the underlying ISA
    // and the configured arch.
    auto thisArch = arch_utils::archConfigManager::getInstance().getArch();

    // Checking if the underlying architecture supports AVX512, and if
    // thisArch(configured architecture) demands a different ISA not as part of
    // the default codepath.
    if (isAvx512 && (thisArch != arch_utils::ArchitectureType::Zen5)
        && (thisArch != arch_utils::ArchitectureType::Zen4)
        && (thisArch != arch_utils::ArchitectureType::GenericAvx512Bf16)
        && (thisArch != arch_utils::ArchitectureType::GenericAvx512Vnni)
        && (thisArch != arch_utils::ArchitectureType::GenericAvx512)) {
        // Switch to Avx2 if the arch is selected as such.
        if ((thisArch != arch_utils::ArchitectureType::Generic)
            && (thisArch != arch_utils::ArchitectureType::Error)) {

            // Double check if avx2 also is supported by machine.
            isAvx2 = arch_utils::archConfigManager::getInstance()
                         .isAvx2Fma3SupportedByArch();
            if (!isAvx2) {
                canGenerateKernelInfo = false;
            } else {
                numRegisters = numRegisters / 2;
            }
        } else {
            // This is an invalid case, disable jit kernel generation.
            canGenerateKernelInfo = false;
        }
    }
    // This condition can be taken in two cases :
    // - isAvx512 is false, but isAvx2 is true.
    // - Both isAvx512 and isAvx2 are true, but the configured arch demands that
    //   we run Avx2 kernels.
    if (isAvx2) {
        // This conditional block facilitates downgrading of eKernelInstPref on
        // an avx2 machine. For now, we do not downgrade to avx2 on avx512
        // machines, since the "classic" framework demands that we run the
        // avx512_256 path in such cases(refer to the code-section from line
        // 61-70).
        if ((thisArch != arch_utils::ArchitectureType::Generic)
            && (thisArch != arch_utils::ArchitectureType::Error)) {
            if (!isAvx512) {
                if ((eKernelInstPref
                     != dlp::kernel_frame::kernelInstrPreference::
                         avx2_xmm_favour)
                    && (eKernelInstPref
                        != dlp::kernel_frame::kernelInstrPreference::
                            avx2_ymm_favour)
                    && (eKernelInstPref
                        != dlp::kernel_frame::kernelInstrPreference::none)) {
                    eKernelInstPref =
                        dlp::kernel_frame::kernelInstrPreference::none;
                }
            }
            // This would be false already if it's an avx2 machine.
            // If it is originally an avx512 machine, we set it to false here,
            // since the configured hardware is avx2.
            isAvx512 = false;
        } else {
            canGenerateKernelInfo = false;
        }
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
            metaData->cMatFormat    = (storFormatC == 'c')
                                          ? kernel_frame::storageFormat::colMajor
                                          : kernel_frame::storageFormat::rowMajor;
            metaData->scaleFactorDt = utils::getStorageDtFromAoclStorageType(
                static_cast<DLP_TYPE>(post_op->sf_stor_type));
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
            metaData->cMatFormat    = (storFormatC == 'c')
                                          ? kernel_frame::storageFormat::colMajor
                                          : kernel_frame::storageFormat::rowMajor;
            metaData->scaleFactorDt = utils::getStorageDtFromAoclStorageType(
                static_cast<DLP_TYPE>(post_op->sf_stor_type));
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
    if (*(static_cast<float*>(gemmIn->alpha)) == 1.0f) {
        alphaScalingType = kernel_frame::scalingType::one;
    }
    kernel_frame::scalingType betaScalingType =
        kernel_frame::scalingType::generic;
    if ((*(static_cast<float*>(gemmIn->beta)) == 0.0f)
        && (gemmIn->k <= gemmIn->kc_hint)) {
        betaScalingType = kernel_frame::scalingType::zero;
    }

    md_t            mr              = gemmIn->mr_hint;
    md_t            nr              = gemmIn->nr_hint;
    md_t            k_unroll        = 1;
    md_t            kc              = gemmIn->kc_hint;
    md_t            prefetch_c_dist = 0; // Setting this to 0, until we use it.
    md_t            cs_c            = gemmIn->cs_c;
    AOCL_MEMORY_TAG mtag_a          = gemmIn->mtag_a;
    AOCL_MEMORY_TAG mtag_b          = gemmIn->mtag_b;
    md_t            c_downscale     = gemmIn->c_downscale;
    bool            anyKOpsOrder    = false;

    // Set the kernel instruction preference based on the CPU features.
    // NOTE : This could be overridden by the user in future
    //        Ex : Wanting to run AVX2 kernels on Zen4, based on
    //             AOCL_ENABLE_INSTRUCTIONS
    kernel_frame::kernelInstrPreference kInstPref = eKernelInstPref;

    // In case the environment variable was not set at all, resort to
    // setting it based on the native hardware support.
    if (kInstPref == kernel_frame::kernelInstrPreference::none) {
        if (isAvx512) {
            kInstPref = kernel_frame::kernelInstrPreference::avx512_zmm_favour;
        } else if (isAvx2) {
            kInstPref = kernel_frame::kernelInstrPreference::avx2_ymm_favour;
        } else {
            // This is an invalid case, disable jit kernel generation.
            return std::nullopt;
        }
    }

    if (gemmIn->n == 1) {
        if (isAvx2) {
            mr       = (kInstPref
                  == kernel_frame::kernelInstrPreference::avx2_ymm_favour)
                           ? 8
                           : 16;
            nr       = 1;
            k_unroll = 1; // k-unroll is 1 for GEMV N1
        } else if (isAvx512) {
            mr       = 16;
            nr       = 1;
            k_unroll = 1; // k-unroll is 1 for GEMV N1
        } else {
            return std::nullopt;
        }

        // NOTE : Since the standard interface send the post-ops
        //        list, even with no-post ops, we will still have
        //        one node which mentions the opcode as
        //        POST_OPS_DISABLE. The null-pointer check is purely
        //        defensive.
        if (gemmIn->metadata == nullptr) {
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
                                         false,
                                         nullptr,
                                         0,
                                         anyKOpsOrder,
                                         kInstPref,
                                         c_downscale };
            return std::make_optional(kI);
        } else if (gemmIn->metadata[0].op_code == POST_OPS_DISABLE) {
            // This condition is not combined with the previous 'if'
            // clause, since we don't want unfriendly short-curcuiting.
            // Ex : Hypothetically, if gemmIn->metadata is NULL, then we
            //      should ensure that this condition is strictly
            //      evaluated after null check.
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
                                         false,
                                         nullptr,
                                         0,
                                         anyKOpsOrder,
                                         kInstPref,
                                         c_downscale };
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
                                             0,
                                             k_unroll,
                                             kc,
                                             prefetch_c_dist,
                                             alphaScalingType,
                                             betaScalingType,
                                             mtag_a,
                                             mtag_b,
                                             false,
                                             nullptr,
                                             0,
                                             anyKOpsOrder,
                                             kInstPref,
                                             c_downscale };
                return std::make_optional(kI);
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
                                             false,
                                             nullptr,
                                             0,
                                             anyKOpsOrder,
                                             kInstPref,
                                             c_downscale };
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
    } else if (gemmIn->m == 1) {
        // The booleans isAvx2 and isAvx512 represent the configured
        // hardware Thus, on an AVX512 machine, if the env var is set to
        // avx2, isAvx2 would be set to true by the constructor.
        if (isAvx2) {
            mr = 1;
            // This setting is a follow-up of the hack we have defined in
            // the DE constructor. With this, we force the AVX512_256 path
            // even when the env var is set to AVX2.
            nr       = (kInstPref
                  == kernel_frame::kernelInstrPreference::avx2_ymm_favour)
                           ? 16
                           : 64;
            k_unroll = 2;
            kc       = 512; // This is harcoded from ZEN4 context.
        } else if (isAvx512) {
            mr = 1;
            nr = 64;
            k_unroll =
                (kInstPref
                 == kernel_frame::kernelInstrPreference::avx512_ymm_favour)
                    ? 2
                    : 4;
            kc = 512;
        } else {
            return std::nullopt;
        }

        // NOTE : Since the standard interface send the post-ops
        //        list, even with no-post ops, we will still have
        //        one node which mentions the opcode as POST_OPS_DISABLE.
        //        The null-pointer check is purely defensive.
        if (gemmIn->metadata == nullptr) {
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
                                         false,
                                         nullptr,
                                         0,
                                         anyKOpsOrder,
                                         kInstPref,
                                         c_downscale };
            return std::make_optional(kI);
        } else if (gemmIn->metadata[0].op_code == POST_OPS_DISABLE) {
            // This condition is not combined with the previous 'if' clause,
            // since we don't want unfriendly short-curcuiting.
            // Ex : Hypothetically, if gemmIn->metadata is NULL, then we
            //      should ensure that this condition is strictly evaluated
            //      after null check.
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
                                         false,
                                         nullptr,
                                         0,
                                         anyKOpsOrder,
                                         kInstPref,
                                         c_downscale };
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
                                             0,
                                             k_unroll,
                                             kc,
                                             prefetch_c_dist,
                                             alphaScalingType,
                                             betaScalingType,
                                             mtag_a,
                                             mtag_b,
                                             false,
                                             nullptr,
                                             0,
                                             anyKOpsOrder,
                                             kInstPref,
                                             c_downscale };
                return std::make_optional(kI);
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
                                             false,
                                             nullptr,
                                             0,
                                             anyKOpsOrder,
                                             kInstPref,
                                             c_downscale };
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
    } else {
        bool allLtFringeKernels = false;
        // For now masked lt fringes can only be enabled on avx512 machines
        // with zmm preferred instructions. Additionally to begin with,
        //  only enabling masked lt fringes when no post-ops are involved
        // for row-major C matrix.
        if (isAvx512
            && (kInstPref
                == kernel_frame::kernelInstrPreference::avx512_zmm_favour)) {
            md_t availableMasks =
                cpu_utils::cpuFeaturesInstance().getNumVectorMaskRegisters();
            md_t requiredMasks = nr / 16;
            // If masks required are more than available masks, we can't
            // generate all lt fringes. Also limit the mask usage to 4.
            if ((requiredMasks <= availableMasks) && (requiredMasks < 5)) {
                allLtFringeKernels = true;
            }
        }

        if (gemmIn->metadata == nullptr) {
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
                                         nullptr,
                                         0,
                                         anyKOpsOrder,
                                         kInstPref,
                                         c_downscale };
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
                                             0,
                                             k_unroll,
                                             kc,
                                             prefetch_c_dist,
                                             alphaScalingType,
                                             betaScalingType,
                                             mtag_a,
                                             mtag_b,
                                             allLtFringeKernels,
                                             nullptr,
                                             0,
                                             anyKOpsOrder,
                                             kInstPref,
                                             c_downscale };
                return std::make_optional(kI);
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
                                             cs_c == 1 ? false
                                                       : allLtFringeKernels,
                                             nullptr,
                                             0,
                                             anyKOpsOrder,
                                             kInstPref,
                                             c_downscale };
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

gemmBF16DEBackend::gemmBF16DEBackend()
    : isAvx512(false)
    , isAvx2(false)
    , isAvx512Bf16(false)
    , eKernelInstPref(kernel_frame::kernelInstrPreference::none)
    , canGenerateKernelInfo(true)
    , f32Backend(nullptr)
{
    // Check for AVX512_BF16 support using the archConfigManager.
    // If it doesn't exist, we reroute to use the F32 JIT path.
    // Env variable standard :
    // Whatever value is given for AOCL_ENABLE_INSTRUCTIONS, we should check
    // for BF16 support in the hardware. If it exists, we deploy the BF16 JIT
    // path, if not, we should use the F32 JIT path for compatibility with
    // AVX2 and non-BF16 AVX512 machines.
    // NOTE : Downscaling support on F32 JIT exists on it's AVX2 code-path
    //        Thus, when rerouting, we have to reroute only on machines that
    //        supports AVX2 exclusively(without any AVX512 support).
    isAvx512Bf16 = arch_utils::archConfigManager::getInstance()
                       .isAvx512Bf16SupportedByArch();
    isAvx512 =
        arch_utils::archConfigManager::getInstance().isAvx512SupportedByArch();

    isAvx2 = arch_utils::archConfigManager::getInstance()
                 .isAvx2Fma3SupportedByArch();

    eKernelInstPref =
        dlp::kernel_frame::kernelInstrPreference::avx512_zmm_favour;

    if (!isAvx512Bf16) {
        // Future-proof: reroute to F32 path for any machine without AVX512BF16
        // This enables BF16 JIT on AVX2 machines by using F32 computation
        if (isAvx2 && !isAvx512) {
            // NOTE : Given that the F32-DE tries to set the kernel instruction
            // preference
            //        based on the AOCL_ENABLE_INSTRUCTIONS and native hardware
            //        support, we can safely assume that we would strictly run
            //        the F32 AVX2 JIT.
            f32Backend = std::make_unique<gemmF32DEBackend>();
        } else {
            canGenerateKernelInfo = false;
        }
    }
}

void
gemmBF16DEBackend::setKernelOps(kernel_frame::kernelOpsMetaData* metaData,
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
            metaData->cMatFormat    = (storFormatC == 'c')
                                          ? kernel_frame::storageFormat::colMajor
                                          : kernel_frame::storageFormat::rowMajor;
            metaData->scaleFactorDt = utils::getStorageDtFromAoclStorageType(
                static_cast<DLP_TYPE>(post_op->sf_stor_type));
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
            metaData->cMatFormat    = (storFormatC == 'c')
                                          ? kernel_frame::storageFormat::colMajor
                                          : kernel_frame::storageFormat::rowMajor;
            metaData->scaleFactorDt = utils::getStorageDtFromAoclStorageType(
                static_cast<DLP_TYPE>(post_op->sf_stor_type));
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
gemmBF16DEBackend::getKernelInfoForInput(iDEInput* in)
{
    auto gemmIn = static_cast<gemmDEInput*>(in);
    if (gemmIn == nullptr) {
        return std::nullopt;
    }

    if (!canGenerateKernelInfo) {
        return std::nullopt;
    }

    // This rerouting currently happens only on machines without AVX512BF16 and
    // AVX512 support, that still have AVX2 support.
    if (f32Backend != nullptr) {
        return f32Backend->getKernelInfoForInput(in);
    }

    // At this point, we know that the underlying architecture supports
    // AVX512-BF16.
    kernel_frame::scalingType alphaScalingType =
        kernel_frame::scalingType::generic;
    if (*(static_cast<float*>(gemmIn->alpha)) == 1.0f) {
        alphaScalingType = kernel_frame::scalingType::one;
    }
    kernel_frame::scalingType betaScalingType =
        kernel_frame::scalingType::generic;
    if ((*(static_cast<float*>(gemmIn->beta)) == 0.0f)
        && (gemmIn->k <= gemmIn->kc_hint)) {
        betaScalingType = kernel_frame::scalingType::zero;
    }

    md_t mr       = gemmIn->mr_hint;
    md_t nr       = gemmIn->nr_hint;
    md_t k_unroll = 1;
    md_t kc       = gemmIn->kc_hint;
    md_t prefetch_c_dist =
        40; // Setting this to 40, which works for ZEN5. Should we set this in
            // the DE constructor, based on the underlying arch?
    AOCL_MEMORY_TAG mtag_a       = gemmIn->mtag_a;
    AOCL_MEMORY_TAG mtag_b       = gemmIn->mtag_b;
    md_t            c_downscale  = gemmIn->c_downscale;
    bool            anyKOpsOrder = false;

    // Set the kernel instruction preference based on the CPU features.
    // The DE constructor sets it to a safe value, based on the hardware
    // support.
    kernel_frame::kernelInstrPreference kInstPref = eKernelInstPref;

    if (gemmIn->n == 1) {
        mr       = 16;
        nr       = 1;
        k_unroll = 1; // k-unroll is 1 for GEMV N1
    } else if (gemmIn->m == 1) {
        nr       = 64;
        mr       = 1;
        k_unroll = 4;
        kc       = 4096;
    }

    if ((gemmIn->metadata != nullptr)
        && (gemmIn->metadata[0].op_code != POST_OPS_DISABLE)) {

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
                                         0,
                                         k_unroll,
                                         kc,
                                         prefetch_c_dist,
                                         alphaScalingType,
                                         betaScalingType,
                                         mtag_a,
                                         mtag_b,
                                         false,
                                         nullptr,
                                         0,
                                         anyKOpsOrder,
                                         kInstPref,
                                         c_downscale };
            return std::make_optional(kI);
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
                                         false,
                                         nullptr,
                                         0,
                                         anyKOpsOrder,
                                         kInstPref,
                                         c_downscale };
            kI.kOpsArrSize = numPostOps;
            kI.kOpsArr =
                kernel_frame::kernelInfo::allocateKernelOpsArray(numPostOps);

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
                                 false,
                                 nullptr,
                                 0,
                                 anyKOpsOrder,
                                 kInstPref,
                                 c_downscale };
    return std::make_optional(kI);
}

} // namespace dlp::de
