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
    numVectorMaskRegisters =
        cpu_utils::cpuFeaturesInstance().getNumVectorMaskRegisters();

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
            numVectorMaskRegisters = 0; // No mask registers in avx2
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
    eKernelInstPref =
        dlp::env_utils::EnvironmentVariableManager::getInstance()
            .getKernelInstructionPreferenceFromEnv("AOCL_ENABLE_INSTRUCTIONS");

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

void
gemmDEBackendUtils::setKernelOps(kernel_frame::kernelOpsMetaData* metaData,
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
} // namespace dlp::de
