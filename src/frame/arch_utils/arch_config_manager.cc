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

#include <vector>

#include "arch_utils/arch_config_manager.hh"
#include "bindings/c_wrappers/capi_cpu_features.h"
#include "classic/dlp_base_types.h"
#include "cpu_utils/cpu_features.hh"
#include "env_utils/env_var_manager.hh"

namespace dlp::arch_utils {

archConfigManager::archConfigManager()
{
    setIsAvx2Fma3Supported();
    setIsAvx512Supported();
    setIsAvx512VnniSupported();
    setIsAvx512Bf16Supported();
    setFpDatapathWidth();
    setIsZen5();
    setIsZen4();
    setIsZen();
    setArch();
}

archConfigManager&
archConfigManager::getInstance()
{
    // C++11 guarantees thread-safe static local initialization
    static archConfigManager instance;
    return instance;
}

bool
archConfigManager::isAvx2Fma3SupportedByArch() const noexcept
{
    return isAvx2Fma3Supported;
}

bool
archConfigManager::isAvx512SupportedByArch() const noexcept
{
    return isAvx512Supported;
}

bool
archConfigManager::isAvx512VnniSupportedByArch() const noexcept
{
    return isAvx512VnniSupported;
}

bool
archConfigManager::isAvx512Bf16SupportedByArch() const noexcept
{
    return isAvx512Bf16Supported;
}

std::uint32_t
archConfigManager::getFpDatapathWidthOfArch() const noexcept
{
    return fpDatapathWidth;
}

bool
archConfigManager::isZen5SimilarArch() const noexcept
{
    return isZen5;
}

bool
archConfigManager::isZen4SimilarArch() const noexcept
{
    return isZen4;
}

bool
archConfigManager::isZenSimilarArch() const noexcept
{
    return isZen;
}

ArchitectureType
archConfigManager::getArch() const noexcept
{
    return thisArch;
}

void
archConfigManager::setIsAvx2Fma3Supported()
{
    isAvx2Fma3Supported = [&]() -> bool {
        std::vector<dlp::cpu_utils::isaFeature> reqFeatures{
            dlp::cpu_utils::isaFeature::avx, dlp::cpu_utils::isaFeature::fma3,
            dlp::cpu_utils::isaFeature::avx2
        };
        return dlp::cpu_utils::cpuFeaturesInstance().hasFeatures(reqFeatures);
    }();
}

void
archConfigManager::setIsAvx512Supported()
{
    isAvx512Supported = [&]() -> bool {
        std::vector<dlp::cpu_utils::isaFeature> reqFeatures{
            dlp::cpu_utils::isaFeature::avx,
            dlp::cpu_utils::isaFeature::fma3,
            dlp::cpu_utils::isaFeature::avx2,
            dlp::cpu_utils::isaFeature::avx512f,
            dlp::cpu_utils::isaFeature::avx512dq,
            dlp::cpu_utils::isaFeature::avx512cd,
            dlp::cpu_utils::isaFeature::avx512bw,
            dlp::cpu_utils::isaFeature::avx512vl
        };
        return dlp::cpu_utils::cpuFeaturesInstance().hasFeatures(reqFeatures);
    }();
}

void
archConfigManager::setIsAvx512VnniSupported()
{
    isAvx512VnniSupported = [&]() -> bool {
        std::vector<dlp::cpu_utils::isaFeature> reqFeatures{
            dlp::cpu_utils::isaFeature::avx,
            dlp::cpu_utils::isaFeature::fma3,
            dlp::cpu_utils::isaFeature::avx2,
            dlp::cpu_utils::isaFeature::avx512f,
            dlp::cpu_utils::isaFeature::avx512dq,
            dlp::cpu_utils::isaFeature::avx512cd,
            dlp::cpu_utils::isaFeature::avx512bw,
            dlp::cpu_utils::isaFeature::avx512vl,
            dlp::cpu_utils::isaFeature::avx512vnni
        };
        return dlp::cpu_utils::cpuFeaturesInstance().hasFeatures(reqFeatures);
    }();
}

void
archConfigManager::setIsAvx512Bf16Supported()
{
    isAvx512Bf16Supported = [&]() -> bool {
        std::vector<dlp::cpu_utils::isaFeature> reqFeatures{
            dlp::cpu_utils::isaFeature::avx,
            dlp::cpu_utils::isaFeature::fma3,
            dlp::cpu_utils::isaFeature::avx2,
            dlp::cpu_utils::isaFeature::avx512f,
            dlp::cpu_utils::isaFeature::avx512dq,
            dlp::cpu_utils::isaFeature::avx512cd,
            dlp::cpu_utils::isaFeature::avx512bw,
            dlp::cpu_utils::isaFeature::avx512vl,
            dlp::cpu_utils::isaFeature::avx512vnni,
            dlp::cpu_utils::isaFeature::avx512bf16
        };
        return dlp::cpu_utils::cpuFeaturesInstance().hasFeatures(reqFeatures);
    }();
}

void
archConfigManager::setFpDatapathWidth()
{
    fpDatapathWidth = [&]() -> uint32_t {
        if (dlp::cpu_utils::cpuFeaturesInstance().getCpuVendor()
            == dlp::cpu_utils::cpuVendor::amd) {
            if (dlp::cpu_utils::cpuFeaturesInstance().hasFeature(
                    dlp::cpu_utils::isaFeature::datapath_fp512)) {
                return DATAPATH_FP512;
            }
            if (dlp::cpu_utils::cpuFeaturesInstance().hasFeature(
                    dlp::cpu_utils::isaFeature::datapath_fp256)) {
                return DATAPATH_FP256;
            }
            if (dlp::cpu_utils::cpuFeaturesInstance().hasFeature(
                    dlp::cpu_utils::isaFeature::datapath_fp128)) {
                return DATAPATH_FP128;
            }
        }
        return DATAPATH_INVALID;
    }();
}

void
archConfigManager::setIsZen5()
{
    isZen5 = [&]() -> bool {
        std::vector<dlp::cpu_utils::isaFeature> reqFeatures{
            dlp::cpu_utils::isaFeature::sse3,
            dlp::cpu_utils::isaFeature::ssse3,
            dlp::cpu_utils::isaFeature::sse41,
            dlp::cpu_utils::isaFeature::sse42,
            dlp::cpu_utils::isaFeature::avx,
            dlp::cpu_utils::isaFeature::fma3,
            dlp::cpu_utils::isaFeature::avx2,
            dlp::cpu_utils::isaFeature::avx512f,
            dlp::cpu_utils::isaFeature::avx512dq,
            dlp::cpu_utils::isaFeature::avx512cd,
            dlp::cpu_utils::isaFeature::avx512bw,
            dlp::cpu_utils::isaFeature::avx512vl,
            dlp::cpu_utils::isaFeature::avx512vnni,
            dlp::cpu_utils::isaFeature::avx512bf16,
            dlp::cpu_utils::isaFeature::movdiri,
            dlp::cpu_utils::isaFeature::movdir64b,
            dlp::cpu_utils::isaFeature::avx512vp2intersect,
            dlp::cpu_utils::isaFeature::avxvnni
        };
        return dlp::cpu_utils::cpuFeaturesInstance().hasFeatures(reqFeatures);
    }();
}

void
archConfigManager::setIsZen4()
{
    isZen4 = [&]() -> bool {
        std::vector<dlp::cpu_utils::isaFeature> reqFeatures{
            dlp::cpu_utils::isaFeature::sse3,
            dlp::cpu_utils::isaFeature::ssse3,
            dlp::cpu_utils::isaFeature::sse41,
            dlp::cpu_utils::isaFeature::sse42,
            dlp::cpu_utils::isaFeature::avx,
            dlp::cpu_utils::isaFeature::fma3,
            dlp::cpu_utils::isaFeature::avx2,
            dlp::cpu_utils::isaFeature::avx512f,
            dlp::cpu_utils::isaFeature::avx512dq,
            dlp::cpu_utils::isaFeature::avx512cd,
            dlp::cpu_utils::isaFeature::avx512bw,
            dlp::cpu_utils::isaFeature::avx512vl,
            dlp::cpu_utils::isaFeature::avx512vnni,
            dlp::cpu_utils::isaFeature::avx512bf16
        };
        return dlp::cpu_utils::cpuFeaturesInstance().hasFeatures(reqFeatures);
    }();
}

void
archConfigManager::setIsZen()
{
    isZen = [&]() -> bool {
        std::vector<dlp::cpu_utils::isaFeature> reqFeatures{
            dlp::cpu_utils::isaFeature::avx, dlp::cpu_utils::isaFeature::fma3,
            dlp::cpu_utils::isaFeature::avx2
        };
        return dlp::cpu_utils::cpuFeaturesInstance().hasFeatures(reqFeatures);
    }();
}

ArchitectureType
archConfigManager::queryUnderlyingArch(void)
{
    ArchitectureType archId = ArchitectureType::Generic;

    auto vendor = cpu_utils::cpuFeaturesInstance().getCpuVendor();
    if (vendor == cpu_utils::cpuVendor::intel) {
        // Check for each Intel configuration that is enabled, check for that
        // microarchitecture. We check from most recent to most dated.
#ifdef DLP_CONFIG_ZEN4
        // Even if not optimized for Intel processors, this should
        // generally perform better than skx codepath.
        // Currently only enabled for zen4 and amdzen configurations
        if (isAvx512Supported) {
            archId = ArchitectureType::Zen4;
        }
#endif
#ifdef DLP_CONFIG_ZEN3
        // Even if not optimized for Intel processors, this should
        // generally perform better than haswell codepath.
        // Currently only enabled for zen3 and amdzen configurations
        if (isAvx2Fma3Supported) {
            archId = ArchitectureType::Zen3;
        }
#endif
    } else if (vendor == cpu_utils::cpuVendor::amd) {
        // The ARCH is decided based on the dlp config set during compile
        // time and the ISA supported. The model and family id is NOT used
        // for determining the same.
#ifdef DLP_CONFIG_ZEN5
        if (isZen5) {
            archId = ArchitectureType::Zen5;
        } else if (isAvx512Bf16Supported) {
            // Fallback test for future AMD processors
            // Assume zen5 (if available) is preferable to zen4.
            archId = ArchitectureType::Zen5;
        }
#endif
#ifdef DLP_CONFIG_ZEN4
        if (isZen4) {
            archId = ArchitectureType::Zen4;
        } else if (isAvx512Bf16Supported) {
            // Fallback test for future AMD processors
            // Use zen4 if zen5 is not available.
            archId = ArchitectureType::Zen4;
        }
#endif
#ifdef DLP_CONFIG_ZEN3
        if (isZen) {
            archId = ArchitectureType::Zen3;
        } else if (isAvx2Fma3Supported) {
            // Fallback test for future AMD processors
            // Use zen3 if AVX512 support is not available but AVX2 is.
            archId = ArchitectureType::Zen3;
        }
#endif
#ifdef DLP_CONFIG_ZEN2
        if (isZen) {
            archId = ArchitectureType::Zen2;
        }
#endif
#ifdef DLP_CONFIG_ZEN
        if (isZen) {
            archId = ArchitectureType::Zen;
        }
#endif
    }

    return archId;
}

void
archConfigManager::setArch(void)
{
    // Get actual hardware arch and model ids.
    auto actualArch = queryUnderlyingArch();

    auto& manager = dlp::env_utils::EnvironmentVariableManager::getInstance();
    thisArch      = manager.getArchitectureFromEnv("AOCL_ENABLE_INSTRUCTIONS");

    bool aocl_e_i = false;
    if (thisArch != ArchitectureType::Error) {
        aocl_e_i = true;
    } else {
        // Architecture families.
#if defined DLP_FAMILY_INTEL64 || defined DLP_FAMILY_AMDZEN                    \
    || defined DLP_FAMILY_AMD64_LEGACY || defined DLP_FAMILY_X86_64
        thisArch = actualArch;
#endif
        // AMD microarchitectures.
#ifdef DLP_FAMILY_ZEN5
        thisArch = ArchitectureType::Zen5;
#endif
#ifdef DLP_FAMILY_ZEN4
        thisArch = ArchitectureType::Zen4;
#endif
#ifdef DLP_FAMILY_ZEN3
        thisArch = ArchitectureType::Zen3;
#endif
#ifdef DLP_FAMILY_ZEN2
        thisArch = ArchitectureType::Zen2;
#endif
#ifdef DLP_FAMILY_ZEN
        thisArch = ArchitectureType::Zen;
#endif
    }

    ArchitectureType origThisArch = thisArch;

    if ((origThisArch != ArchitectureType::Error) && (aocl_e_i == true)) {
        // If AVX2 test fails here we assume either:
        // 1. Config was either zen, zen2, zen3, zen4, zen5, haswell or skx,
        //    so there is no fallback code path, hence error checking
        //    above will fail.
        // 2. Config was amdzen, intel64 or x86_64, and will have
        //    generic code path.
        if (!isAvx2Fma3Supported) {
            switch (origThisArch) {
                case ArchitectureType::Zen5:
                case ArchitectureType::Zen4:
                case ArchitectureType::Zen3:
                case ArchitectureType::Zen2:
                case ArchitectureType::Zen:
                    thisArch = actualArch;
                    break;
                default:
                    break;
            }
        }
        // If AVX512 test fails here we assume either:
        // 1. Config was either zen5, zen4 or skx, so there is
        //    no fallback code path, hence error checking
        //    above will fail.
        // 2. Config was amdzen, intel64 or x86_64, and will have
        //    appropriate avx2 code path to try.
        if (!isAvx512Supported) {
            switch (origThisArch) {
                case ArchitectureType::Zen5:
                case ArchitectureType::Zen4:
                    thisArch = actualArch;
                    break;
                default:
                    break;
            }
        }
    }
}

} // namespace dlp::arch_utils
