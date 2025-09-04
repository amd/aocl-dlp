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

#include <string>

#include "arch_utils/arch_config_manager.hh"
#include "bindings/c_wrappers/capi_env_config.h"
#include "env_utils/env_var_manager.hh"

dlp_arch_t
dlp_env_get_var_arch_type(const char* env_var_name)
{
    if (env_var_name == nullptr) {
        return DLP_ARCH_ERROR;
    }

    try {
        auto& manager =
            dlp::env_utils::EnvironmentVariableManager::getInstance();
        auto arch_type = manager.getArchitectureFromEnv(env_var_name);
        switch (arch_type) {
            case dlp::arch_utils::ArchitectureType::Zen5:
                return DLP_ARCH_ZEN5;
            case dlp::arch_utils::ArchitectureType::Zen4:
                return DLP_ARCH_ZEN4;
            case dlp::arch_utils::ArchitectureType::Zen3:
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
    } catch (...) {
        // Ensure no exceptions escape to C code
        return DLP_ARCH_ERROR;
    }
}

md_t
dlp_env_get_int(const char* env_var_name, int default_value)
{
    if (env_var_name == nullptr) {
        return default_value;
    }

    try {
        auto& manager =
            dlp::env_utils::EnvironmentVariableManager::getInstance();
        return manager.getIntFromEnv(env_var_name, default_value);
    } catch (...) {
        // Ensure no exceptions escape to C code
        return default_value;
    }
}

int
dlp_env_get_bool(const char* env_var_name, int default_value)
{
    if (env_var_name == nullptr) {
        return default_value;
    }

    try {
        auto& manager =
            dlp::env_utils::EnvironmentVariableManager::getInstance();
        bool result = manager.getBoolFromEnv(env_var_name, default_value != 0);
        return result ? 1 : 0;
    } catch (...) {
        // Ensure no exceptions escape to C code
        return default_value;
    }
}
