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

#include <tuple>

#include "arch_utils/arch_config_manager.hh"
#include "bindings/c_wrappers/capi_cpu_features.h"
#include "bindings/c_wrappers/capi_env_config.h"
#include "cpu_utils/cpu_features.hh"

// The implementation for one of the C apis is in DLP Plus, so guarding
// its header file to avoid name mangling.
DLP_BEGIN_EXTERN_C
#include "classic/aocl_lib_interface_apis.h"
DLP_END_EXTERN_C

using namespace dlp::cpu_utils;
using namespace dlp::arch_utils;

// Determine if the CPU has support for AVX2 and FMA3.
bool
dlp_cpuid_is_avx2fma3_supported(void)
{
    return archConfigManager::getInstance().isAvx2Fma3SupportedByArch();
}

// Determine if the CPU has support for AVX512.
bool
dlp_cpuid_is_avx512_supported(void)
{
    return archConfigManager::getInstance().isAvx512SupportedByArch();
}

// Determine if the CPU has support for AVX512_VNNI.
bool
dlp_cpuid_is_avx512vnni_supported(void)
{
    return archConfigManager::getInstance().isAvx512VnniSupportedByArch();
}

// Determine if the CPU has support for AVX512_BF16.
bool
dlp_cpuid_is_avx512bf16_supported(void)
{
    return archConfigManager::getInstance().isAvx512Bf16SupportedByArch();
}

// Determine if the CPU has support for AVX512_FP16.
bool
dlp_cpuid_is_avx512fp16_supported(void)
{
    return archConfigManager::getInstance().isAvx512Fp16SupportedByArch();
}

uint32_t
dlp_cpuid_query_fp_datapath(void)
{
    return archConfigManager::getInstance().getFpDatapathWidthOfArch();
}

bool
dlp_cpuid_is_similar_zen5_arch()
{
    return archConfigManager::getInstance().isZen5SimilarArch();
}

bool
dlp_cpuid_is_similar_zen6_arch()
{
    return archConfigManager::getInstance().isZen6SimilarArch();
}

bool
dlp_cpuid_is_similar_zen4_arch()
{
    return archConfigManager::getInstance().isZen4SimilarArch();
}

bool
dlp_cpuid_is_similar_zen_arch()
{
    return archConfigManager::getInstance().isZenSimilarArch();
}

dlp_arch_t
dlp_get_arch(void)
{
    static const dlp_arch_t arch_id = []() -> dlp_arch_t {
        ArchitectureType thisArch =
            archConfigManager::getInstance().getConfiguredArch();
        // arch is generally used for setting block params and NOT to decide
        // micro-kernels. Hence the approximate mapping for generic archs to
        // zen similar archs is expected to work for now. To be revisited
        // when other vendor archs are added and requires custom handling.
        switch (thisArch) {
            case dlp::arch_utils::ArchitectureType::Zen6:
            case dlp::arch_utils::ArchitectureType::GenericAvx512Fp16:
                return DLP_ARCH_ZEN6;
            case dlp::arch_utils::ArchitectureType::Zen5:
            case dlp::arch_utils::ArchitectureType::GenericAvx512Bf16:
                return DLP_ARCH_ZEN5;
            case dlp::arch_utils::ArchitectureType::Zen4:
            case dlp::arch_utils::ArchitectureType::GenericAvx512Vnni:
            case dlp::arch_utils::ArchitectureType::GenericAvx512:
                return DLP_ARCH_ZEN4;
            case dlp::arch_utils::ArchitectureType::Zen3:
            case dlp::arch_utils::ArchitectureType::GenericAvx2:
                return DLP_ARCH_ZEN3;
            case dlp::arch_utils::ArchitectureType::Zen2:
                return DLP_ARCH_ZEN2;
            case dlp::arch_utils::ArchitectureType::Zen:
                return DLP_ARCH_ZEN;
            case dlp::arch_utils::ArchitectureType::Generic:
                return DLP_ARCH_GENERIC;
            default:
                return DLP_ARCH_ERROR;
        }
    }();
    return arch_id;
}

bool
dlp_aocl_enable_instruction_query(void)
{
    // Check whether the AOCL_DLP_ENABLE_INSTRUCTIONS environment variable
    // is set or not.
    static const bool aocl_e_i = []() -> bool {
        auto arch_id =
            dlp_env_get_var_arch_type("AOCL_DLP_ENABLE_INSTRUCTIONS");

        bool aocl_e_i = false;
        if (arch_id != DLP_ARCH_ERROR) {
            aocl_e_i = true;
        }

        return aocl_e_i;
    }();

    return aocl_e_i;
}
