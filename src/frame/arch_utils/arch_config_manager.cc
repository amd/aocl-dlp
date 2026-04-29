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

#include <unordered_map>
#include <vector>

#include "arch_utils/arch_config_manager.hh"
#include "bindings/c_wrappers/capi_cpu_features.h"
#include "classic/dlp_base_types.h"
#include "cpu_utils/cpu_features.hh"
#include "env_utils/env_var_manager.hh"

namespace dlp::arch_utils {

archConfigManager::archConfigManager()
{
    // Actual hardware support.
    setIsAvx2Fma3Supported();
    setIsAvx512Supported();
    setIsAvx512VnniSupported();
    setIsAvx512Bf16Supported();
    setIsAvx512Fp16Supported();
    setFpDatapathWidth();
    setIsZen6();
    setIsZen5();
    setIsZen4();
    setIsZen();

    // setArch should only be called post the above feature detections as it
    // relies on the results of those detections to set the arch correctly.
    // This function call sequence should not be modified. As a thumb rule any
    // actual hardware detection should be done before setArch is called, and
    // any configured arch determination should be done inside/after setArch.
    setArch();

    // Configured hardware support.
    setIsAvx2Fma3Configured();
    setIsAvx512Configured();
    setIsAvx512VnniConfigured();
    setIsAvx512Bf16Configured();
    setIsAvx512Fp16Configured();
    setIsZen6Configured();
    setIsZen5Configured();
    setIsZen4Configured();
    setIsZenConfigured();
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
archConfigManager::setIsAvx512Fp16Supported()
{
    isAvx512Fp16Supported = [&]() -> bool {
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
            dlp::cpu_utils::isaFeature::avx512bf16,
            dlp::cpu_utils::isaFeature::avx512fp16
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
archConfigManager::setIsZen6()
{
    isZen6 = [&]() -> bool {
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
            dlp::cpu_utils::isaFeature::avxvnni,
            dlp::cpu_utils::isaFeature::avx512fp16
        };
        return dlp::cpu_utils::cpuFeaturesInstance().hasFeatures(reqFeatures);
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
        // Map non amd x86_64 machines to generic variants that can be used to
        // convey arch ISA properties.
        if (isAvx512Fp16Supported) {
            archId = ArchitectureType::GenericAvx512Fp16;
        } else if (isAvx512Bf16Supported) {
            archId = ArchitectureType::GenericAvx512Bf16;
        } else if (isAvx512VnniSupported) {
            archId = ArchitectureType::GenericAvx512Vnni;
        } else if (isAvx512Supported) {
            archId = ArchitectureType::GenericAvx512;
        } else if (isAvx2Fma3Supported) {
            archId = ArchitectureType::GenericAvx2;
        }
    } else if (vendor == cpu_utils::cpuVendor::amd) {
        // The ARCH is decided based on the dlp config set during compile
        // time and the ISA supported. The model and family id is NOT used
        // for determining the same.
        if (isZen6) {
            archId = ArchitectureType::Zen6;
        } else if (isZen5) {
            archId = ArchitectureType::Zen5;
        } else if (isZen4) {
            archId = ArchitectureType::Zen4;
        } else if (isZen) {
            archId = ArchitectureType::Zen3;
#ifdef DLP_CONFIG_ZEN2
            archId = ArchitectureType::Zen2;
#endif
#ifdef DLP_CONFIG_ZEN
            archId = ArchitectureType::Zen;
#endif
        } else if (isAvx512Fp16Supported) {
            archId = ArchitectureType::Zen6;
        } else if (isAvx512Bf16Supported) {
            // Assume zen5 (if available) is preferable to zen4.
            archId = ArchitectureType::Zen5;
#ifdef DLP_CONFIG_ZEN4
            archId = ArchitectureType::Zen4;
#endif
        } else if (isAvx2Fma3Supported) {
            archId = ArchitectureType::Zen3;
#ifdef DLP_CONFIG_ZEN2
            archId = ArchitectureType::Zen2;
#endif
#ifdef DLP_CONFIG_ZEN
            archId = ArchitectureType::Zen;
#endif
        }
    }

    return archId;
}

// Explicit requirements table: each architecture declares exactly which
// ISA features it needs. If any required feature is absent on hardware,
// fall back to actualArch.
//
// To add a new architecture: add one row to the table below.
// To add a new feature dimension: add one bool column and one clause
// to the 'unsatisfied' check.
struct ArchFeatureRequirements
{
    bool requiresAvx2Fma3;
    bool requiresAvx512;
    bool requiresAvx512Vnni;
    bool requiresAvx512Bf16;
    bool requiresAvx512Fp16;
};

static const std::unordered_map<ArchitectureType, ArchFeatureRequirements>
    archRequirements = {
        // Format: ArchType req_avx2 req_avx512 req_vnni req_bf16 req_fp16
        { ArchitectureType::Zen6, { true, true, true, true, true } },
        { ArchitectureType::Zen5, { true, true, true, true, false } },
        { ArchitectureType::Zen4, { true, true, true, true, false } },
        { ArchitectureType::Zen3, { true, false, false, false, false } },
        { ArchitectureType::Zen2, { true, false, false, false, false } },
        { ArchitectureType::Zen, { true, false, false, false, false } },
        { ArchitectureType::GenericAvx512Fp16,
          { true, true, true, true, true } },
        { ArchitectureType::GenericAvx512Bf16,
          { true, true, true, true, false } },
        { ArchitectureType::GenericAvx512Vnni,
          { true, true, true, false, false } },
        { ArchitectureType::GenericAvx512,
          { true, true, false, false, false } },
        { ArchitectureType::GenericAvx2, { true, false, false, false, false } },
    };

void
archConfigManager::setArch(void)
{
    // Get actual hardware arch and model ids.
    actualArch = queryUnderlyingArch();

    auto& manager = dlp::env_utils::EnvironmentVariableManager::getInstance();
    thisArch = manager.getArchitectureFromEnv("AOCL_DLP_ENABLE_INSTRUCTIONS");

    bool aocl_e_i = false;
    if (thisArch != ArchitectureType::Error) {
        aocl_e_i = true;
    } else {
        thisArch = actualArch;
    }

    ArchitectureType origThisArch = thisArch;

    if ((origThisArch != ArchitectureType::Error) && (aocl_e_i == true)) {
        // Validate if the configured architecture's ISA requirements are
        // satisfied by the underlying hardware.
        auto it = archRequirements.find(origThisArch);
        if (it != archRequirements.end()) {
            const auto& req = it->second;
            bool        unsatisfied =
                (req.requiresAvx2Fma3 && !isAvx2Fma3Supported)
                || (req.requiresAvx512 && !isAvx512Supported)
                || (req.requiresAvx512Vnni && !isAvx512VnniSupported)
                || (req.requiresAvx512Bf16 && !isAvx512Bf16Supported)
                || (req.requiresAvx512Fp16 && !isAvx512Fp16Supported);
            if (unsatisfied) {
                thisArch = actualArch;
            }
        }
    }
}

void
archConfigManager::setIsAvx2Fma3Configured()
{
    isAvx2Fma3Configured =
        ((thisArch != ArchitectureType::Error)
         && (thisArch != ArchitectureType::Generic) && isAvx2Fma3Supported);
}

void
archConfigManager::setIsAvx512Configured()
{
    isAvx512Configured = (((thisArch == ArchitectureType::GenericAvx512)
                           || (thisArch == ArchitectureType::GenericAvx512Vnni)
                           || (thisArch == ArchitectureType::GenericAvx512Bf16)
                           || (thisArch == ArchitectureType::GenericAvx512Fp16)
                           || (thisArch == ArchitectureType::Zen6)
                           || (thisArch == ArchitectureType::Zen5)
                           || (thisArch == ArchitectureType::Zen4))
                          && isAvx512Supported);
}

void
archConfigManager::setIsAvx512VnniConfigured()
{
    isAvx512VnniConfigured =
        (((thisArch == ArchitectureType::GenericAvx512Vnni)
          || (thisArch == ArchitectureType::GenericAvx512Bf16)
          || (thisArch == ArchitectureType::GenericAvx512Fp16)
          || (thisArch == ArchitectureType::Zen6)
          || (thisArch == ArchitectureType::Zen5)
          || (thisArch == ArchitectureType::Zen4))
         && isAvx512VnniSupported);
}

void
archConfigManager::setIsAvx512Bf16Configured()
{
    isAvx512Bf16Configured =
        (((thisArch == ArchitectureType::GenericAvx512Bf16)
          || (thisArch == ArchitectureType::GenericAvx512Fp16)
          || (thisArch == ArchitectureType::Zen6)
          || (thisArch == ArchitectureType::Zen5)
          || (thisArch == ArchitectureType::Zen4))
         && isAvx512Bf16Supported);
}

void
archConfigManager::setIsAvx512Fp16Configured()
{
    isAvx512Fp16Configured = (((thisArch == ArchitectureType::GenericAvx512Fp16)
                               || (thisArch == ArchitectureType::Zen6))
                              && isAvx512Fp16Supported);
}

void
archConfigManager::setIsZen6Configured()
{
    isZen6Configured = (thisArch == ArchitectureType::Zen6);
}

void
archConfigManager::setIsZen5Configured()
{
    isZen5Configured = (thisArch == ArchitectureType::Zen5);
}

void
archConfigManager::setIsZen4Configured()
{
    isZen4Configured = (thisArch == ArchitectureType::Zen4);
}

void
archConfigManager::setIsZenConfigured()
{
    isZenConfigured = ((thisArch == ArchitectureType::Zen3)
                       || (thisArch == ArchitectureType::Zen2)
                       || (thisArch == ArchitectureType::Zen));
}

} // namespace dlp::arch_utils
