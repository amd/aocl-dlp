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

#include <algorithm>
#include <array>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "arch_utils/arch_config_manager.hh"
#include "bindings/c_wrappers/capi_kernel_frame_wrappers.h"
#include "classic/dlp_base_types.h"
#include "kernel_frame/kernel_frame_base.hh"

namespace dlp::env_utils {

/**
 * @brief Thread-safe environment variable configuration manager
 *
 * This class provides a centralized, thread-safe interface for managing
 * environment variable parsing and caching. It focuses on performance-critical
 * paths by minimizing string operations and providing efficient lookups.
 *
 * Key features:
 * - Thread-safe singleton access
 * - Efficient string-to-enum conversions with compile-time lookup tables
 * - Case-insensitive environment variable parsing
 * - Caching for frequently accessed values
 * - Type-safe architecture enumeration handling
 *
 * @note This class is designed for C++17 compatibility and follows RAII
 * principles. All memory management is handled automatically with no raw
 * pointers exposed.
 */
class EnvironmentVariableManager final
{
  public:
    /**
     * @brief Get the singleton instance of EnvironmentVariableManager
     * @return Reference to the singleton instance (thread-safe initialization)
     */
    static EnvironmentVariableManager& getInstance();

    /**
     * @brief Parse architecture type from environment variable
     * @param env_var_name Name of the environment variable to read
     * @return ArchitectureType enum value, or ArchitectureType::Error if
     * parsing fails
     *
     * Supports the following string mappings (case-insensitive):
     * - "zen6" -> ArchitectureType::Zen6
     * - "zen5" -> ArchitectureType::Zen5
     * - "zen4" -> ArchitectureType::Zen4
     * - "zen3" -> ArchitectureType::Zen3
     * - "zen2" -> ArchitectureType::Zen2
     * - "zen", "zen1" -> ArchitectureType::Zen
     * - "avx512" -> ArchitectureType::Zen4 (alias)
     * - "avx2" -> ArchitectureType::Zen3 (alias)
     * - "avx" -> ArchitectureType::Generic (alias)
     * - SSE variants -> ArchitectureType::Generic (alias)
     */
    arch_utils::ArchitectureType getArchitectureFromEnv(
        const std::string& env_var_name);

    /**
     * @brief Get kernel instruction preference from environment variable
     * @param env_var_name Name of the environment variable to read for kernel
     * instruction preference
     * @return kernel_frame::kernelInstrPreference enum value based on
     * environment variable setting
     *
     * Supports the following string mappings (case-insensitive):
     * - "zen5" -> kernelInstrPreference::avx512_zmm_favour
     * - "zen4" -> kernelInstrPreference::avx512_zmm_favour
     * - "zen3" -> kernelInstrPreference::avx2_ymm_favour
     * - "zen2" -> kernelInstrPreference::avx2_ymm_favour
     * - "zen", "zen1" -> kernelInstrPreference::avx2_ymm_favour
     * - "avx512" -> kernelInstrPreference::avx512_zmm_favour
     * - "avx512_ymm" -> kernelInstrPreference::avx512_ymm_favour (alias)
     * - "avx2" -> kernelInstrPreference::avx2_ymm_favour
     */
    kernel_frame::kernelInstrPreference getKernelInstructionPreferenceFromEnv(
        const std::string& env_var_name);

    /**
     * @brief Get string value from environment variable with optional
     * default
     * @param env_var_name Name of the environment variable to read
     * @param default_value Default value to return if environment variable
     * is not set
     * @return Environment variable value or default value
     *
     * Thread-safe and caches results for performance.
     */
    std::string getStringFromEnv(const std::string& env_var_name,
                                 std::string        default_value = "");

    /**
     * @brief Get integer value from environment variable with optional default
     * @param env_var_name Name of the environment variable to read
     * @param default_value Default value to return if parsing fails
     * @return Parsed integer value or default value
     *
     * Returns default_value if environment variable is not set or cannot be
     * parsed as integer.
     */
    md_t getIntFromEnv(const std::string& env_var_name, md_t default_value = 0);

    /**
     * @brief Get boolean value from environment variable with optional default
     * @param env_var_name Name of the environment variable to read
     * @param default_value Default value to return if parsing fails
     * @return Parsed boolean value or default value
     *
     * Recognizes the following as true (case-insensitive): "1", "true", "yes",
     * "on" Recognizes the following as false (case-insensitive): "0", "false",
     * "no", "off" Returns default_value for any other values or if environment
     * variable is not set.
     */
    bool getBoolFromEnv(const std::string& env_var_name,
                        bool               default_value = false);

    // Disable copy and move operations for singleton
    EnvironmentVariableManager(const EnvironmentVariableManager&) = delete;
    EnvironmentVariableManager& operator=(const EnvironmentVariableManager&) =
        delete;
    EnvironmentVariableManager(EnvironmentVariableManager&&) = delete;
    EnvironmentVariableManager& operator=(EnvironmentVariableManager&&) =
        delete;

  private:
    EnvironmentVariableManager();
    ~EnvironmentVariableManager() = default;

    void toLowerInPlace(std::string& str) noexcept;

    template<typename T, uint64_t N>
    T parsePredefinedString(
        const std::string&                                   keyStr,
        const std::array<std::pair<std::string_view, T>, N>& strMap,
        T                                                    errVal) noexcept
    {
        // Linear search through the sorted lookup table
        for (const auto& [preDefName, preDefType] : strMap) {
            if (keyStr == preDefName) {
                return preDefType;
            }
        }

        return errVal;
    }

    /**
     * @brief Get raw environment variable value
     * @param env_var_name Name of environment variable
     * @return Optional string value (empty if variable doesn't exist)
     */
    std::optional<std::string> getRawEnvVar(
        const std::string_view env_var_name) noexcept;

    /**
     * @brief Cache for string environment variable values
     * Key: environment variable name, Value: cached value
     * @note This will only be filled once from the constructor
     * and is not updated after that. This is thread-safe.
     */
    std::unordered_map<std::string, std::string> env_cache_;
};

} // namespace dlp::env_utils
