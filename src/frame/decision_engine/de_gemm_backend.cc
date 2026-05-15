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
    , canGeneratePackBKernelInfo(true)
{
    // Use this eKernelInstPref to generate kernelInfo for the 32 Ymm register
    // based f32 kernel generation.
    eKernelInstPref = dlp::env_utils::EnvironmentVariableManager::getInstance()
                          .getKernelInstructionPreferenceFromEnv(
                              "AOCL_DLP_ENABLE_INSTRUCTIONS");

    isAvx512 = arch_utils::archConfigManager::getInstance()
                   .isAvx512SupportedByConfiguredArch();
    bool isActualAvx512 =
        arch_utils::archConfigManager::getInstance().isAvx512SupportedByArch();

    // isAvx2 is only checked for and set to true if machine is guaranteed/
    // configured to not support Avx512 isa.
    if (!isAvx512) {
        isAvx2 = arch_utils::archConfigManager::getInstance()
                     .isAvx2Fma3SupportedByConfiguredArch();
    }

    // This needs to be downgraded if the actual cpu is avx512 but is
    // downgraded via using AOCL_DLP_ENABLE_INSTRUCTIONS.
    numRegisters = cpu_utils::cpuFeaturesInstance().getNumVectorRegisters();
    numVectorMaskRegisters =
        cpu_utils::cpuFeaturesInstance().getNumVectorMaskRegisters();

    // Checking if underlying architecture supports AVX512, and if thisArch
    // (configured architecture) demands a different ISA not as part of the
    // default codepath.
    if (!isAvx512) {
        if (isAvx2) {
            if (isActualAvx512) {
                // The underlying machine supports avx512, but the arch is
                // configured as avx2. In that case the previously queried
                // register details will be invalid since those would be based
                // on avx512 support. We need to downgrade the register
                // details to reflect the configured avx2 support.
                numRegisters           = numRegisters / 2;
                numVectorMaskRegisters = 0; // No mask registers in avx2
            }

            if ((eKernelInstPref
                 != dlp::kernel_frame::kernelInstrPreference::avx2_xmm_favour)
                && (eKernelInstPref
                    != dlp::kernel_frame::kernelInstrPreference::
                        avx2_ymm_favour)
                && (eKernelInstPref
                    != dlp::kernel_frame::kernelInstrPreference::none)) {
                eKernelInstPref =
                    dlp::kernel_frame::kernelInstrPreference::none;
            }
        } else {
            canGenerateKernelInfo      = false;
            canGeneratePackBKernelInfo = false;
        }
    }
}

std::optional<kernel_frame::kernelInfo>
gemmF32DEBackend::getKernelInfoForInput(iDEInput* in)
{
    auto gemmIn = static_cast<gemmDEInput*>(in);
    if (gemmIn == nullptr) {
        return std::nullopt;
    }

    if (!canGenerateKernelInfo) {
        return std::nullopt;
    }

    kernel_frame::kernelInfo kI;
    if (gemmIn->m == 1 || gemmIn->n == 1) {
        kI = getGemvKernelInfoForInputFastPath(
            gemmIn->k_dtype, gemmIn->m, gemmIn->n, gemmIn->k, gemmIn->rs_a,
            gemmIn->cs_a, gemmIn->rs_b, gemmIn->cs_b, gemmIn->rs_c,
            gemmIn->cs_c, gemmIn->alpha, gemmIn->beta, gemmIn->mtag_a,
            gemmIn->mtag_b, gemmIn->metadata, gemmIn->mr_hint, gemmIn->nr_hint,
            gemmIn->kc_hint, gemmIn->c_downscale, false);
    } else {
        kI = getGemmKernelInfoForInputFastPath(
            gemmIn->k_dtype, gemmIn->m, gemmIn->n, gemmIn->k, gemmIn->rs_a,
            gemmIn->cs_a, gemmIn->rs_b, gemmIn->cs_b, gemmIn->rs_c,
            gemmIn->cs_c, gemmIn->alpha, gemmIn->beta, gemmIn->mtag_a,
            gemmIn->mtag_b, gemmIn->metadata, gemmIn->mr_hint, gemmIn->nr_hint,
            gemmIn->kc_hint, gemmIn->c_downscale, false);
    }

    if ((kI.mr <= 0) || (kI.nr <= 0)) {
        return std::nullopt;
    } else {
        return std::make_optional(kI);
    }
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
    // On machines that support AVX512BF16, we use the BF16 path only
    // when the environment variable('AOCL_DLP_ENABLE_INSTRUCTIONS') is NOT
    // set to AVX2(or it's alternative, like ZEN3, ZEN2 or ZEN). This is the
    // standard that the legacy code enforces. NOTE : Downscaling support on F32
    // JIT exists on it's AVX2 code-path
    //        Thus, when rerouting, we have to reroute only on machines that
    //        supports AVX2 exclusively(without any AVX512 support).
    eKernelInstPref = dlp::env_utils::EnvironmentVariableManager::getInstance()
                          .getKernelInstructionPreferenceFromEnv(
                              "AOCL_DLP_ENABLE_INSTRUCTIONS");

    isAvx512Bf16 = arch_utils::archConfigManager::getInstance()
                       .isAvx512Bf16SupportedByArch();
    isAvx512 =
        arch_utils::archConfigManager::getInstance().isAvx512SupportedByArch();

    isAvx2 = arch_utils::archConfigManager::getInstance()
                 .isAvx2Fma3SupportedByArch();

    if (!isAvx512Bf16) {
        // Instantiate the F32 DE Backend for rerouting to F32 JIT
        // for any machine without AVX512BF16 support
        // where kernel instruction would be set to avx512_zmm_favour
        f32Backend = std::make_unique<gemmF32DEBackend>();
    } else if (eKernelInstPref
               == kernel_frame::kernelInstrPreference::avx2_ymm_favour) {
        // This would be the scenario where the machine supports AVX512BF16, but
        // we choose to run the F32(AVX2) code-path.
        f32Backend = std::make_unique<gemmF32DEBackend>();
    } else {
        // In scenarios where the AOCL_DLP_ENABLE_INSTRUCTIONS is set to AVX512
        // or AVX512_256, or not set at all, we generate the AVX512BF16 JIT
        // kernels. Once the hybrid version is implemented, we will avoid this
        // hardcoding.
        // kernel instruction preference is set avx512_zmm_bf16_favour when
        // underlying machine has avx512_bf16 support
        eKernelInstPref =
            kernel_frame::kernelInstrPreference::avx512_zmm_bf16_favour;
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

    // Only gemv n=1 supported in BF16 JIT path for now.
    kernel_frame::kernelInfo kI;
    if (gemmIn->m == 1 || gemmIn->n == 1) {
        kI = getGemvKernelInfoForInputFastPath(
            gemmIn->k_dtype, gemmIn->m, gemmIn->n, gemmIn->k, gemmIn->rs_a,
            gemmIn->cs_a, gemmIn->rs_b, gemmIn->cs_b, gemmIn->rs_c,
            gemmIn->cs_c, gemmIn->alpha, gemmIn->beta, gemmIn->mtag_a,
            gemmIn->mtag_b, gemmIn->metadata, gemmIn->mr_hint, gemmIn->nr_hint,
            gemmIn->kc_hint, gemmIn->c_downscale, false);
    } else {
        kI = getGemmKernelInfoForInputFastPath(
            gemmIn->k_dtype, gemmIn->m, gemmIn->n, gemmIn->k, gemmIn->rs_a,
            gemmIn->cs_a, gemmIn->rs_b, gemmIn->cs_b, gemmIn->rs_c,
            gemmIn->cs_c, gemmIn->alpha, gemmIn->beta, gemmIn->mtag_a,
            gemmIn->mtag_b, gemmIn->metadata, gemmIn->mr_hint, gemmIn->nr_hint,
            gemmIn->kc_hint, gemmIn->c_downscale, false);
    }

    if ((kI.mr <= 0) || (kI.nr <= 0)) {
        return std::nullopt;
    } else {
        return std::make_optional(kI);
    }
}

gemmU8S8DEBackend::gemmU8S8DEBackend()
    : isAvx512(false)
    , isAvx2(false)
    , isAvx512Vnni(false)
    , eKernelInstPref(kernel_frame::kernelInstrPreference::none)
    , canGenerateKernelInfo(true)
{
    // Use this eKernelInstPref to generate kernelInfo for kernel generation.
    eKernelInstPref = dlp::env_utils::EnvironmentVariableManager::getInstance()
                          .getKernelInstructionPreferenceFromEnv(
                              "AOCL_DLP_ENABLE_INSTRUCTIONS");

    isAvx512 =
        arch_utils::archConfigManager::getInstance().isAvx512SupportedByArch();

    // Check for VNNI support (required for u8s8 kernels)
    isAvx512Vnni = cpu_utils::cpuFeaturesInstance().hasFeature(
        cpu_utils::isaFeature::avx512vnni);

    // If Avx512 is not supported, we cannot generate kernel info.
    if (!isAvx512 || !isAvx512Vnni) {
        canGenerateKernelInfo = false;
    }
}

std::optional<kernel_frame::kernelInfo>
gemmU8S8DEBackend::getKernelInfoForInput(iDEInput* in)
{
    auto gemmIn = static_cast<gemmDEInput*>(in);
    if (gemmIn == nullptr) {
        return std::nullopt;
    }

    if (!canGenerateKernelInfo) {
        return std::nullopt;
    }

    kernel_frame::kernelInfo kI;
    if (gemmIn->m == 1 || gemmIn->n == 1) {
        kI = getGemvKernelInfoForInputFastPath(
            gemmIn->k_dtype, gemmIn->m, gemmIn->n, gemmIn->k, gemmIn->rs_a,
            gemmIn->cs_a, gemmIn->rs_b, gemmIn->cs_b, gemmIn->rs_c,
            gemmIn->cs_c, gemmIn->alpha, gemmIn->beta, gemmIn->mtag_a,
            gemmIn->mtag_b, gemmIn->metadata, gemmIn->mr_hint, gemmIn->nr_hint,
            gemmIn->kc_hint, gemmIn->c_downscale, false);
    } else {
        kI = getGemmKernelInfoForInputFastPath(
            gemmIn->k_dtype, gemmIn->m, gemmIn->n, gemmIn->k, gemmIn->rs_a,
            gemmIn->cs_a, gemmIn->rs_b, gemmIn->cs_b, gemmIn->rs_c,
            gemmIn->cs_c, gemmIn->alpha, gemmIn->beta, gemmIn->mtag_a,
            gemmIn->mtag_b, gemmIn->metadata, gemmIn->mr_hint, gemmIn->nr_hint,
            gemmIn->kc_hint, gemmIn->c_downscale, false);
    }

    if ((kI.mr <= 0) || (kI.nr <= 0)) {
        return std::nullopt;
    } else {
        return std::make_optional(kI);
    }
}

gemmS8DEBackend::gemmS8DEBackend()
    : isAvx512(false)
    , isAvx2(false)
    , isAvx512Bf16(false)
    , isAvx512Vnni(false)
    , eKernelInstPref(kernel_frame::kernelInstrPreference::none)
    , canGenerateKernelInfo(true)
{
    // Use this eKernelInstPref to generate kernelInfo for kernel generation.
    eKernelInstPref = dlp::env_utils::EnvironmentVariableManager::getInstance()
                          .getKernelInstructionPreferenceFromEnv(
                              "AOCL_DLP_ENABLE_INSTRUCTIONS");

    // Check for AVX512 support
    isAvx512 =
        arch_utils::archConfigManager::getInstance().isAvx512SupportedByArch();

    // Check for VNNI support, required for all s8 kernels
    isAvx512Vnni = cpu_utils::cpuFeaturesInstance().hasFeature(
        cpu_utils::isaFeature::avx512vnni);

    // If either of AVX512, VNNI or BF16 is unsupport, kernel info
    // cannot be generated.
    if (!isAvx512 || !isAvx512Vnni) {
        canGenerateKernelInfo = false;
    }
}

std::optional<kernel_frame::kernelInfo>
gemmS8DEBackend::getKernelInfoForInput(iDEInput* in)
{
    auto gemmIn = static_cast<gemmDEInput*>(in);
    if (gemmIn == nullptr) {
        return std::nullopt;
    }

    if (!canGenerateKernelInfo) {
        return std::nullopt;
    }

    kernel_frame::kernelInfo kI;
    if (gemmIn->m == 1 || gemmIn->n == 1) {
        kI = getGemvKernelInfoForInputFastPath(
            gemmIn->k_dtype, gemmIn->m, gemmIn->n, gemmIn->k, gemmIn->rs_a,
            gemmIn->cs_a, gemmIn->rs_b, gemmIn->cs_b, gemmIn->rs_c,
            gemmIn->cs_c, gemmIn->alpha, gemmIn->beta, gemmIn->mtag_a,
            gemmIn->mtag_b, gemmIn->metadata, gemmIn->mr_hint, gemmIn->nr_hint,
            gemmIn->kc_hint, gemmIn->c_downscale, false);
    } else {
        kI = getGemmKernelInfoForInputFastPath(
            gemmIn->k_dtype, gemmIn->m, gemmIn->n, gemmIn->k, gemmIn->rs_a,
            gemmIn->cs_a, gemmIn->rs_b, gemmIn->cs_b, gemmIn->rs_c,
            gemmIn->cs_c, gemmIn->alpha, gemmIn->beta, gemmIn->mtag_a,
            gemmIn->mtag_b, gemmIn->metadata, gemmIn->mr_hint, gemmIn->nr_hint,
            gemmIn->kc_hint, gemmIn->c_downscale, false);
    }

    if ((kI.mr <= 0) || (kI.nr <= 0)) {
        return std::nullopt;
    } else {
        return std::make_optional(kI);
    }
}

void
gemmDEBackendUtils::setKernelOps(
    kernel_frame::kernelOpsMetaData*              metaData,
    dlp_gemm_post_op*                             post_op,
    [[maybe_unused]] kernel_frame::kernelDatatype k_dtype)
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
            if (post_op->op_args3 != nullptr) {
                md_t bias_len          = *static_cast<md_t*>(post_op->op_args3);
                metaData->isScalarBias = (bias_len == 1);
            }
            // Dequantization support for BIAS
            if (post_op->scale_factor != nullptr) {
                // Set scale factor datatype from user-provided type
                metaData->scaleFactorDt =
                    utils::getStorageDtFromAoclStorageType(
                        static_cast<DLP_TYPE>(post_op->sf_stor_type));

                // Determine scalar vs vector scale factor based on length
                metaData->scalarScaleFactorRequired =
                    (post_op->scale_factor_len == 1) ? true : false;
                metaData->vectorScaleFactorRequired =
                    (post_op->scale_factor_len > 1) ? true : false;
            }

            if (post_op->bias_zp != nullptr) {
                // Set zero point datatype from user-provided type
                metaData->zeroPointDt = utils::getStorageDtFromAoclStorageType(
                    static_cast<DLP_TYPE>(post_op->zp_stor_type));

                // Determine scalar vs vector zero point based on length
                metaData->scalarZeroPointRequired =
                    (post_op->bias_zp_len == 1) ? true : false;
                metaData->vectorZeroPointRequired =
                    (post_op->bias_zp_len > 1) ? true : false;
            }
            break;
        }
        case POST_OPS_RELU:
            metaData->type = kernel_frame::kernelOps::relu;
            break;
        case POST_OPS_RELU_SCALE:
            metaData->type           = kernel_frame::kernelOps::reluScale;
            metaData->paramStorageDt = utils::getStorageDtFromAoclStorageType(
                static_cast<DLP_TYPE>(post_op->stor_type));
            break;
        case POST_OPS_GELU_TANH:
            metaData->type = kernel_frame::kernelOps::geluTanh;
            break;
        case POST_OPS_GELU_ERF:
            metaData->type = kernel_frame::kernelOps::geluErf;
            break;
        case POST_OPS_CLIP:
            metaData->type           = kernel_frame::kernelOps::clip;
            metaData->paramStorageDt = utils::getStorageDtFromAoclStorageType(
                static_cast<DLP_TYPE>(post_op->stor_type));
            break;
        case POST_OPS_SWISH: {
            metaData->type           = kernel_frame::kernelOps::swish;
            metaData->paramStorageDt = utils::getStorageDtFromAoclStorageType(
                static_cast<DLP_TYPE>(post_op->stor_type));
            break;
        }
        case POST_OPS_TANH:
            metaData->type = kernel_frame::kernelOps::tanh;
            break;
        case POST_OPS_SIGMOID:
            metaData->type = kernel_frame::kernelOps::sigmoid;
            break;
        case POST_OPS_MISH:
            metaData->type = kernel_frame::kernelOps::mish;
            break;
        case POST_OPS_DOWNSCALE: {
            metaData->type = kernel_frame::kernelOps::downscale;
            char storFormatC =
                std::tolower(*(static_cast<char*>(post_op->op_args2)));
            metaData->cMatFormat = (storFormatC == 'c')
                                       ? kernel_frame::storageFormat::colMajor
                                       : kernel_frame::storageFormat::rowMajor;
            if (post_op->scale_factor != nullptr) {
                metaData->scaleFactorDt =
                    utils::getStorageDtFromAoclStorageType(
                        static_cast<DLP_TYPE>(post_op->sf_stor_type));
                metaData->scalarScaleFactorRequired =
                    (post_op->scale_factor_len == 1) ? true : false;
                metaData->vectorScaleFactorRequired =
                    (post_op->scale_factor_len > 1) ? true : false;
            }
            if (post_op->op_args3 != nullptr) {
                metaData->zeroPointDt = utils::getStorageDtFromAoclStorageType(
                    static_cast<DLP_TYPE>(post_op->zp_stor_type));
                metaData->scalarZeroPointRequired =
                    (*(static_cast<md_t*>(post_op->op_args3)) == 1) ? true
                                                                    : false;
                metaData->vectorZeroPointRequired =
                    (*(static_cast<md_t*>(post_op->op_args3)) > 1) ? true
                                                                   : false;
            }
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
            // Only set scale factor metadata if scale_factor is provided
            if (post_op->scale_factor != nullptr) {
                metaData->scaleFactorDt =
                    utils::getStorageDtFromAoclStorageType(
                        static_cast<DLP_TYPE>(post_op->sf_stor_type));
                metaData->scalarScaleFactorRequired =
                    (post_op->scale_factor_len == 1) ? true : false;
                metaData->vectorScaleFactorRequired =
                    (post_op->scale_factor_len > 1) ? true : false;
            }
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
            // Only set scale factor metadata if scale_factor is provided
            if (post_op->scale_factor != nullptr) {
                metaData->scaleFactorDt =
                    utils::getStorageDtFromAoclStorageType(
                        static_cast<DLP_TYPE>(post_op->sf_stor_type));
                metaData->scalarScaleFactorRequired =
                    (post_op->scale_factor_len == 1) ? true : false;
                metaData->vectorScaleFactorRequired =
                    (post_op->scale_factor_len > 1) ? true : false;
            }
            break;
        }
        case POST_OPS_ADQUANTIZE: {
            // A-matrix dequantization: result = round((acc + b_col_sum * zp_A)
            // * inv_scale_A) Converts S32 accumulator back to original scale
            // after quantized GEMM.
            metaData->type = kernel_frame::kernelOps::aDQuantize;

            // A-dequantization uses per-row parameters
            metaData->cMatFormat = kernel_frame::storageFormat::rowMajor;

            // Extract inverse scale factor data type
            metaData->scaleFactorDt = utils::getStorageDtFromAoclStorageType(
                static_cast<DLP_TYPE>(post_op->sf_stor_type));
            metaData->scalarScaleFactorRequired =
                (post_op->scale_factor_len == 1) ? true : false;
            metaData->vectorScaleFactorRequired =
                (post_op->scale_factor_len > 1) ? true : false;

            // Extract zero-point data type
            metaData->zeroPointDt = utils::getStorageDtFromAoclStorageType(
                static_cast<DLP_TYPE>(post_op->zp_stor_type));
            if (post_op->op_args3 != nullptr) {
                metaData->scalarZeroPointRequired =
                    (*(static_cast<md_t*>(post_op->op_args3)) == 1) ? true
                                                                    : false;
                metaData->vectorZeroPointRequired =
                    (*(static_cast<md_t*>(post_op->op_args3)) > 1) ? true
                                                                   : false;
            }
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
gemmFP16DEBackend::gemmFP16DEBackend()
    : isAvx512(false)
    , isAvx512FP16(false)
    , eKernelInstPref(kernel_frame::kernelInstrPreference::none)
    , canGenerateKernelInfo(true)
{
    // Check for AVX512-FP16 support (required for FP16 JIT kernels)
    isAvx512FP16 = cpu_utils::cpuFeaturesInstance().hasFeature(
        cpu_utils::isaFeature::avx512fp16);

    isAvx512 =
        arch_utils::archConfigManager::getInstance().isAvx512SupportedByArch();

    // FP16 JIT requires AVX512-FP16 support
    if (!isAvx512 || !isAvx512FP16) {
        canGenerateKernelInfo = false;
    } else {
        // Set to AVX512 ZMM preference for FP16
        eKernelInstPref =
            kernel_frame::kernelInstrPreference::avx512_zmm_favour;
    }
}

std::optional<kernel_frame::kernelInfo>
gemmFP16DEBackend::getKernelInfoForInput(iDEInput* in)
{
    auto gemmIn = static_cast<gemmDEInput*>(in);
    if (gemmIn == nullptr) {
        return std::nullopt;
    }

    if (!canGenerateKernelInfo) {
        return std::nullopt;
    }

    kernel_frame::kernelInfo kI;
    if (gemmIn->m == 1 || gemmIn->n == 1) {
        kI = getGemvKernelInfoForInputFastPath(
            gemmIn->k_dtype, gemmIn->m, gemmIn->n, gemmIn->k, gemmIn->rs_a,
            gemmIn->cs_a, gemmIn->rs_b, gemmIn->cs_b, gemmIn->rs_c,
            gemmIn->cs_c, gemmIn->alpha, gemmIn->beta, gemmIn->mtag_a,
            gemmIn->mtag_b, gemmIn->metadata, gemmIn->mr_hint, gemmIn->nr_hint,
            gemmIn->kc_hint, gemmIn->c_downscale, false);
    } else {
        kI = getGemmKernelInfoForInputFastPath(
            gemmIn->k_dtype, gemmIn->m, gemmIn->n, gemmIn->k, gemmIn->rs_a,
            gemmIn->cs_a, gemmIn->rs_b, gemmIn->cs_b, gemmIn->rs_c,
            gemmIn->cs_c, gemmIn->alpha, gemmIn->beta, gemmIn->mtag_a,
            gemmIn->mtag_b, gemmIn->metadata, gemmIn->mr_hint, gemmIn->nr_hint,
            gemmIn->kc_hint, gemmIn->c_downscale, false);
    }

    if ((kI.mr <= 0) || (kI.nr <= 0)) {
        return std::nullopt;
    } else {
        return std::make_optional(kI);
    }
}

gemmF32FP16DEBackend::gemmF32FP16DEBackend()
    : isAvx512(false)
    , eKernelInstPref(kernel_frame::kernelInstrPreference::none)
    , canGenerateKernelInfo(true)
{
    isAvx512 =
        arch_utils::archConfigManager::getInstance().isAvx512SupportedByArch();

    // F32×FP16 only requires AVX-512F + AVX-512BW (for vcvtph2ps zmm)
    // Does NOT require avx512fp16 (that's for native FP16 arithmetic)
    if (!isAvx512) {
        canGenerateKernelInfo = false;
    } else {
        eKernelInstPref =
            kernel_frame::kernelInstrPreference::avx512_zmm_favour;
    }
}

std::optional<kernel_frame::kernelInfo>
gemmF32FP16DEBackend::getKernelInfoForInput(iDEInput* in)
{
    auto gemmIn = static_cast<gemmDEInput*>(in);
    if (gemmIn == nullptr) {
        return std::nullopt;
    }

    if (!canGenerateKernelInfo) {
        return std::nullopt;
    }

    kernel_frame::kernelInfo kI;
    if (gemmIn->m == 1 || gemmIn->n == 1) {
        kI = getGemvKernelInfoForInputFastPath(
            gemmIn->k_dtype, gemmIn->m, gemmIn->n, gemmIn->k, gemmIn->rs_a,
            gemmIn->cs_a, gemmIn->rs_b, gemmIn->cs_b, gemmIn->rs_c,
            gemmIn->cs_c, gemmIn->alpha, gemmIn->beta, gemmIn->mtag_a,
            gemmIn->mtag_b, gemmIn->metadata, gemmIn->mr_hint, gemmIn->nr_hint,
            gemmIn->kc_hint, gemmIn->c_downscale, false);
    } else {
        kI = getGemmKernelInfoForInputFastPath(
            gemmIn->k_dtype, gemmIn->m, gemmIn->n, gemmIn->k, gemmIn->rs_a,
            gemmIn->cs_a, gemmIn->rs_b, gemmIn->cs_b, gemmIn->rs_c,
            gemmIn->cs_c, gemmIn->alpha, gemmIn->beta, gemmIn->mtag_a,
            gemmIn->mtag_b, gemmIn->metadata, gemmIn->mr_hint, gemmIn->nr_hint,
            gemmIn->kc_hint, gemmIn->c_downscale, false);
    }

    if ((kI.mr <= 0) || (kI.nr <= 0)) {
        return std::nullopt;
    } else {
        return std::make_optional(kI);
    }
}

} // namespace dlp::de
