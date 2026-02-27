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

typedef enum
{
    DLP_ARCH_ERROR = 0,
    DLP_ARCH_GENERIC,

    // AMD
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

// API to check if AVX512_fp16 is supported or not on the current platform.
bool
dlp_cpuid_is_avx512fp16_supported(void);

// API to get FP/SIMD execution datapath width.
uint32_t
dlp_cpuid_query_fp_datapath(void);

// API to check if cpu is zen5 arch.
bool
dlp_cpuid_is_similar_zen5_arch();

// API to check if cpu is zen4 arch.
bool
dlp_cpuid_is_similar_zen4_arch();

// API to check if cpu is zen arch.
bool
dlp_cpuid_is_similar_zen_arch();

// API to get underlying architecture (also modifiable
// via AOCL_ENABLE_INSTRUCTIONS)
dlp_arch_t
dlp_get_arch(void);

DLP_END_EXTERN_C

#endif // CAPI_CPU_FEATURES_H
