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

#ifndef CAPI_ENV_CONFIG_H
#define CAPI_ENV_CONFIG_H

#include "classic/dlp_base_types.h"
// Include the legacy type definition
#include "bindings/c_wrappers/capi_cpu_features.h"

DLP_BEGIN_EXTERN_C

/**
 * @brief C API wrapper for environment variable architecture parsing
 * @param env_var_name Name of the environment variable to parse
 * @return dlp_arch_t enum value, compatible with existing C code
 */
dlp_arch_t
dlp_env_get_var_arch_type(const char* env_var_name);

/**
 * @brief C API wrapper for kernel instruction preference parsing
 * @param env_var_name Name of the environment variable to parse
 * @return dlp_instr_pref_t enum value, compatible with existing C code
 */
dlp_instr_pref_t
dlp_env_get_kernel_instr_pref(const char* env_var_name);

/**
 * @brief C API wrapper for integer environment variable access
 * @param env_var_name Name of the environment variable to read
 * @param default_value Default value if parsing fails
 * @return Parsed integer value or default value
 */
md_t
dlp_env_get_int(const char* env_var_name, int default_value);

/**
 * @brief C API wrapper for boolean environment variable access
 * @param env_var_name Name of the environment variable to read
 * @param default_value Default value if parsing fails
 * @return 1 for true, 0 for false
 */
int
dlp_env_get_bool(const char* env_var_name, int default_value);

/**
 * @brief C API wrapper for checking if AOCL_DLP_ENABLE_LPGEMM_LOGGER
 * environment is set or not.
 * @return 1/true for enabled, 0/false otherwise.
 */
bool
dlp_env_is_logger_enabled();

DLP_END_EXTERN_C

#endif // CAPI_ENV_CONFIG_H
