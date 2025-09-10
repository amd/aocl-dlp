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

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>

#include "env_utils/env_var_manager.hh"

namespace dlp::env_utils {

// Static lookup table for architecture string mappings
// Using std::array for compile-time initialization and better cache locality
constexpr std::array<std::pair<std::string_view, arch_utils::ArchitectureType>,
                     18>
    ARCH_STRING_MAP = {
        { // Primary AMD architecture names
          { "zen5", arch_utils::ArchitectureType::Zen5 },
          { "zen4", arch_utils::ArchitectureType::Zen4 },
          { "zen3", arch_utils::ArchitectureType::Zen3 },
          { "zen2", arch_utils::ArchitectureType::Zen2 },
          { "zen", arch_utils::ArchitectureType::Zen },
          { "zen1", arch_utils::ArchitectureType::Zen }, // Alias for zen

          // ISA aliases mapped to suitable architectures
          { "avx512", arch_utils::ArchitectureType::Zen4 },
          { "avx2", arch_utils::ArchitectureType::Zen3 },
          { "avx", arch_utils::ArchitectureType::Generic },

          // SSE variants - all map to generic
          { "sse4_2", arch_utils::ArchitectureType::Generic },
          { "sse4.2", arch_utils::ArchitectureType::Generic },
          { "sse4_1", arch_utils::ArchitectureType::Generic },
          { "sse4.1", arch_utils::ArchitectureType::Generic },
          { "sse4a", arch_utils::ArchitectureType::Generic },
          { "sse4", arch_utils::ArchitectureType::Generic },
          { "ssse3", arch_utils::ArchitectureType::Generic },
          { "sse3", arch_utils::ArchitectureType::Generic },
          { "sse2", arch_utils::ArchitectureType::Generic } }
    };

// Static lookup table for isa preference string mappings
// Using std::array for compile-time initialization and better cache locality
constexpr std::array<
    std::pair<std::string_view, kernel_frame::kernelInstrPreference>,
    9>
    INSTR_PREF_STRING_MAP = {
        { { "zen5", kernel_frame::kernelInstrPreference::avx512_zmm_favour },
          { "zen4", kernel_frame::kernelInstrPreference::avx512_zmm_favour },
          { "zen3", kernel_frame::kernelInstrPreference::avx2_ymm_favour },
          { "zen2", kernel_frame::kernelInstrPreference::avx2_ymm_favour },
          { "zen", kernel_frame::kernelInstrPreference::avx2_ymm_favour },
          { "zen1", kernel_frame::kernelInstrPreference::avx2_ymm_favour },
          { "avx512", kernel_frame::kernelInstrPreference::avx512_zmm_favour },
          { "avx512_ymm",
            kernel_frame::kernelInstrPreference::avx512_ymm_favour },
          { "avx2", kernel_frame::kernelInstrPreference::avx2_ymm_favour } }
    };

// Static lookup table for boolean string parsing
constexpr std::array<std::string_view, 4> TRUE_STRINGS  = { "1", "true", "yes",
                                                            "on" };
constexpr std::array<std::string_view, 4> FALSE_STRINGS = { "0", "false", "no",
                                                            "off" };

// Add all supported Environment variables here that are used by this manager
// and need to be initialized in the constructor.
constexpr std::array<std::string_view, 4> DLP_ENV_VAR_STRINGS = {
    "DLP_NUM_THREADS", "DLP_IC_NT", "DLP_JC_NT", "AOCL_ENABLE_LPGEMM_LOGGER"
};
constexpr std::array<std::string_view, 1> DLP_ARCH_ENV_VAR_STRINGS = {
    "AOCL_ENABLE_INSTRUCTIONS"
};

EnvironmentVariableManager::EnvironmentVariableManager()
{
    for (const auto& env_var_name : DLP_ENV_VAR_STRINGS) {
        auto optVal = getRawEnvVar(env_var_name);
        // Only add entry in env map if its defined.
        if (optVal.has_value()) {
            env_cache_[std::string{ env_var_name }] = optVal.value();
        }
    }
    for (const auto& env_var_name : DLP_ARCH_ENV_VAR_STRINGS) {
        auto optVal = getRawEnvVar(env_var_name);
        // Only add entry in env map if its defined.
        if (optVal.has_value()) {
            env_cache_[std::string{ env_var_name }] = optVal.value();

            // The arch env variable values are made case insensitive to
            // speed up its corresponding arch enum lookup.
            toLowerInPlace(env_cache_[std::string{ env_var_name }]);
        }
    }
}

EnvironmentVariableManager&
EnvironmentVariableManager::getInstance()
{
    // C++11 guarantees thread-safe static local initialization
    static EnvironmentVariableManager instance;
    return instance;
}

arch_utils::ArchitectureType
EnvironmentVariableManager::getArchitectureFromEnv(
    const std::string& env_var_name)
{
    // Check cache first
    auto cache_it = env_cache_.find(env_var_name);
    if (cache_it != env_cache_.end()) {
        return parsePredefinedString(cache_it->second, ARCH_STRING_MAP,
                                     arch_utils::ArchitectureType::Error);
    }

    return arch_utils::ArchitectureType::Error;
}

kernel_frame::kernelInstrPreference
EnvironmentVariableManager::getKernelInstructionPreferenceFromEnv(
    const std::string& env_var_name)
{
    // Check cache first
    auto cache_it = env_cache_.find(env_var_name);
    if (cache_it != env_cache_.end()) {
        return parsePredefinedString(cache_it->second, INSTR_PREF_STRING_MAP,
                                     kernel_frame::kernelInstrPreference::none);
    }

    return kernel_frame::kernelInstrPreference::none;
}

std::string
EnvironmentVariableManager::getStringFromEnv(const std::string& env_var_name,
                                             std::string        default_value)
{
    // Check cache first
    auto cache_it = env_cache_.find(env_var_name);
    if (cache_it != env_cache_.end()) {
        return cache_it->second;
    }

    return default_value;
}

md_t
EnvironmentVariableManager::getIntFromEnv(const std::string& env_var_name,
                                          md_t               default_value)
{
    auto str_value = getStringFromEnv(env_var_name);
    if (str_value.empty()) {
        return default_value;
    }

    // Use std::from_chars for efficient parsing (C++17)
    md_t        result    = default_value;
    const char* str_begin = str_value.data();
    const char* str_end   = str_begin + str_value.size();

    auto [ptr, ec] = std::from_chars(str_begin, str_end, result, 10);

    // Check if parsing was successful and consumed entire string
    if ((ec == std::errc{}) && (ptr == str_end)) {
        return result;
    }

    return default_value;
}

bool
EnvironmentVariableManager::getBoolFromEnv(const std::string& env_var_name,
                                           bool               default_value)
{
    auto str_value = getStringFromEnv(env_var_name);
    if (str_value.empty()) {
        return default_value;
    }

    // Convert to lowercase for case-insensitive comparison
    toLowerInPlace(str_value);

    // Check true values
    for (const auto& true_str : TRUE_STRINGS) {
        if (str_value == true_str) {
            return true;
        }
    }

    // Check false values
    for (const auto& false_str : FALSE_STRINGS) {
        if (str_value == false_str) {
            return false;
        }
    }

    return default_value;
}

void
EnvironmentVariableManager::toLowerInPlace(std::string& str) noexcept
{
    std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
}

std::optional<std::string>
EnvironmentVariableManager::getRawEnvVar(
    const std::string_view env_var_name) noexcept
{
    // It is guaranteed the string_view's passed here are valid
    // strings with null terminator.
    const char* env_value = std::getenv(env_var_name.data());
    if (env_value == nullptr) {
        return std::nullopt;
    }

    return std::string{ env_value };
}

} // namespace dlp::env_utils
