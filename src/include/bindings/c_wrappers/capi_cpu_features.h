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

#ifndef CAPI_CPU_FEATURES_H
#define CAPI_CPU_FEATURES_H

#include <stdbool.h>

#include "classic/dlp_base_types.h"
#include "classic/dlp_macros.h"

DLP_BEGIN_EXTERN_C

typedef enum
{
    DATAPATH_INVALID = -1,
    DATAPATH_FP128,
    DATAPATH_FP256,
    DATAPATH_FP512
} dlp_datapath_width;

// Content type of a cache at a given level. Mirrors dlp::cpu_utils::cacheType.
// A split level (typically L1) has separate DATA and INSTRUCTION caches; a
// unified level (typically L2 / L3) serves both, so either a DATA or an
// INSTRUCTION request is satisfied by the unified cache.
typedef enum
{
    DLP_CACHE_TYPE_INVALID = 0,
    DLP_CACHE_TYPE_DATA,
    DLP_CACHE_TYPE_INSTRUCTION,
    DLP_CACHE_TYPE_UNIFIED
} dlp_cache_type;

typedef enum
{
    DLP_ARCH_ERROR = 0,
    DLP_ARCH_GENERIC,

    // AMD
    DLP_ARCH_ZEN6,
    DLP_ARCH_ZEN5,
    DLP_ARCH_ZEN4,
    DLP_ARCH_ZEN3,
    DLP_ARCH_ZEN2,
    DLP_ARCH_ZEN,

    DLP_NUM_ARCHS
} dlp_arch_t;

typedef enum
{
    DLP_INSTR_PREF_NONE = 0,

    // x86_64 specific hints.
    DLP_INSTR_PREF_AVX2_XMM_FAVOUR,
    DLP_INSTR_PREF_AVX2_YMM_FAVOUR,
    DLP_INSTR_PREF_AVX512_XMM_FAVOUR,
    DLP_INSTR_PREF_AVX512_YMM_FAVOUR,
    DLP_INSTR_PREF_AVX512_ZMM_FAVOUR,

    MAX_KERNEL_INSTR_PREFERENCES
} dlp_instr_pref_t;

// API to check if AVX2 and FMA3 are supported or not on the current platform.
bool
dlp_cpuid_is_avx2fma3_supported(void);

// API to check if AVX512 is supported or not on the current platform.
bool
dlp_cpuid_is_avx512_supported(void);

// API to check if AVX512_VNNI is supported or not on the current platform.
bool
dlp_cpuid_is_avx512vnni_supported(void);

// API to check if AVX512_bf16 is supported or not on the current platform.
bool
dlp_cpuid_is_avx512bf16_supported(void);

// API to check if AVX512_bf16 is supported or not by the configured
// architecture, i.e. after any downgrade requested through
// AOCL_DLP_ENABLE_INSTRUCTIONS. This is the query the decision engine uses to
// decide whether BF16 kernels are generated at all, or whether BF16 is
// rerouted to the F32 kernels instead.
bool
dlp_cpuid_is_avx512bf16_configured(void);

// API to check if AVX512_fp16 is supported or not on the current platform.
bool
dlp_cpuid_is_avx512fp16_supported(void);

// API to get FP/SIMD execution datapath width.
// Returns a dlp_datapath_width enumerator: DATAPATH_FP128 (128-bit),
// DATAPATH_FP256 (256-bit) or DATAPATH_FP512 (512-bit) on supported AMD
// parts, or DATAPATH_INVALID when the width cannot be determined. The
// return value is an enum, not a literal bit-width integer.
dlp_datapath_width
dlp_cpuid_query_fp_datapath(void);

// API to check if cpu is zen5 arch.
bool
dlp_cpuid_is_similar_zen5_arch();

// API to check if cpu is zen6 arch.
bool
dlp_cpuid_is_similar_zen6_arch();

// API to check if cpu is zen4 arch.
bool
dlp_cpuid_is_similar_zen4_arch();

// API to check if cpu is zen arch.
bool
dlp_cpuid_is_similar_zen_arch();

// API to get underlying architecture (also modifiable
// via AOCL_DLP_ENABLE_INSTRUCTIONS)
dlp_arch_t
dlp_get_arch(void);

// API to get the number of physical cores per compute die. Assumes uniform
// compute-die topology and returns 0 if it cannot be determined.
int32_t
dlp_cpuid_get_num_cores_per_compute_die(void);

// API to get the number of hardware threads per compute die. This reports the
// processor's as-shipped SMT width: with SMT disabled it still counts every
// thread a core can run, not the online count. Assumes uniform compute-die
// topology and returns 0 if it cannot be determined.
int32_t
dlp_cpuid_get_num_hw_threads_per_compute_die(void);

// API to get the number of cache levels present on the current CPU (e.g. 3 =>
// L1, L2, L3). Returns 0 if the cache topology could not be determined.
int32_t
dlp_cpuid_get_num_cache_levels(void);

// API to get the size (in bytes) of the cache at the given 'level' that
// services the requested content 'type'. Valid values of 'level' are 1 (L1),
// 2 (L2), 3 (L3), etc. For a split level (typically L1) 'type' selects
// between the data and instruction caches; a unified level (typically L2/L3)
// satisfies either request. Returns 0 if the requested level/type does not
// exist or could not be determined.
int64_t
dlp_cpuid_get_cache_size(int32_t level, dlp_cache_type type);

DLP_END_EXTERN_C

#endif // CAPI_CPU_FEATURES_H
