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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
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

#ifndef AOCL_DLP_LIB_INTERFACE_H
#define AOCL_DLP_LIB_INTERFACE_H

#include "classic/dlp_base_types.h"

// Threading runtime setters.

/**
 * @brief Set library-global threading ways (ic_ways, jc_ways)
 *
 * Sets ways-based parallelism for all DLP operations in this process.
 * This configuration has higher precedence than num_threads.
 *
 * Atomicity Guarantee:
 * Three atomic stores (ways, nt, ext_mt) execute independently but form
 * logical transaction because all callers ALWAYS set: nt=-1, ext_mt=FALSE.
 *
 * @param jc JC loop parallelism (outer loop), must be > 0 or will be set to -1
 * @param ic IC loop parallelism (inner loop), must be > 0 or will be set to -1
 *
 * @note Setting ways automatically unsets num_threads (mutually exclusive
 * modes)
 * @note Value -1 means "unset"
 */
DLP_CLASSIC_EXPORT void
dlp_thread_set_ways_library(md_t jc, md_t ic);

/**
 * @brief Set thread-local threading ways (ic_ways, jc_ways)
 *
 * Sets ways-based parallelism for DLP operations ONLY in the calling thread.
 * Has HIGHEST precedence - overrides library-global and environment settings.
 *
 * @param jc JC loop parallelism (outer loop), must be > 0 or will be set to -1
 * @param ic IC loop parallelism (inner loop), must be > 0 or will be set to -1
 *
 * @note Thread-local settings are NOT inherited by spawned threads
 * @note Setting ways automatically unsets num_threads (mutually exclusive)
 * @note Value -1 means "unset"
 */
DLP_CLASSIC_EXPORT void
dlp_thread_set_ways_local(md_t jc, md_t ic);

/**
 * @brief Alias for dlp_thread_set_ways_local.
 */
DLP_CLASSIC_EXPORT void
dlp_thread_set_ways(md_t jc, md_t ic);

/**
 * @brief Get currently active IC ways parallelism count
 *
 * @return Active ic_ways value based on priority order, or -1 if unset
 */
DLP_CLASSIC_EXPORT md_t
dlp_thread_get_ic_ways_active(void);

/**
 * @brief Get currently active JC ways parallelism count
 *
 * @return Active jc_ways value based on priority order, or -1 if unset
 */
DLP_CLASSIC_EXPORT md_t
dlp_thread_get_jc_ways_active(void);

/**
 * @brief Set library-global thread count
 *
 * Sets thread count-based parallelism for all DLP operations in this process.
 * Note: Ways-based configuration (ic_ways, jc_ways) has HIGHER precedence.
 *
 * Interaction with Ways:
 * - Does NOT modify lib_packed_ways (ic/jc preserved)
 * - If ways are set (ic>0 or jc>0), they take precedence over num_threads
 * - Only effective when ways are unset (-1)
 *
 * @param n_threads Number of threads to use (> 0), or -1 to unset
 *
 * @note Value -1 means "unset"
 * @note Setting num_threads disables external threading control (OpenMP)
 */
DLP_CLASSIC_EXPORT void
dlp_thread_set_num_threads_library(md_t n_threads);

/**
 * @brief Set thread-local thread count
 *
 * Sets thread count-based parallelism for DLP operations ONLY in the calling
 * thread. Has HIGHEST precedence - overrides library-global and environment
 * settings. Note: Ways-based configuration still has higher precedence than
 * this.
 *
 * @param n_threads Number of threads to use (> 0), or -1 to unset
 *
 * @note Thread-local settings are NOT inherited by spawned threads
 * @note Value -1 means "unset"
 */
DLP_CLASSIC_EXPORT void
dlp_thread_set_num_threads_local(md_t n_threads);

/**
 * @brief Alias for dlp_thread_set_num_threads_local.
 */
DLP_CLASSIC_EXPORT void
dlp_thread_set_num_threads(md_t n_threads);

/**
 * @brief Get currently active thread count
 *
 * @return Active num_threads value based on priority order, or -1 if unset
 */
DLP_CLASSIC_EXPORT md_t
dlp_thread_get_num_threads_active(void);

/**
 * @brief Query whether AOCL_DLP_ENABLE_INSTRUCTIONS environment variable is
 * set.
 *
 * @return true if the environment variable is set, false otherwise.
 */
DLP_CLASSIC_EXPORT bool
dlp_aocl_enable_instruction_query(void);

/**
 * @brief Query AOCL-DLP library version
 *
 * @param[out] major Major version number (optional, pass NULL to skip)
 * @param[out] minor Minor version number (optional, pass NULL to skip)
 * @param[out] patch Patch version number (optional, pass NULL to skip)
 */
DLP_CLASSIC_EXPORT void
dlp_version_query(int* major, int* minor, int* patch);

#endif // AOCL_DLP_LIB_INTERFACE_H
