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

#include "aocl_dlp_config.h"

#if DLP_OS_WINDOWS
// Windows-specific CPUID implementation using intrinsics
#define NOMINMAX
#include <cstdint>
#include <intrin.h>

// Windows inline assembly replacement
static inline uint64_t
xgetbv(uint32_t xcr)
{
    return static_cast<uint64_t>(_xgetbv(xcr));
}

// Custom CPUID wrappers for Windows
static inline void
dlp_cpuid(
    uint32_t level, uint32_t& eax, uint32_t& ebx, uint32_t& ecx, uint32_t& edx)
{
    int regs[4];
    __cpuid(regs, (int)level);
    eax = (uint32_t)regs[0];
    ebx = (uint32_t)regs[1];
    ecx = (uint32_t)regs[2];
    edx = (uint32_t)regs[3];
}

static inline void
dlp_cpuid_count(uint32_t  level,
                uint32_t  count,
                uint32_t& eax,
                uint32_t& ebx,
                uint32_t& ecx,
                uint32_t& edx)
{
    int regs[4];
    __cpuidex(regs, (int)level, (int)count);
    eax = (uint32_t)regs[0];
    ebx = (uint32_t)regs[1];
    ecx = (uint32_t)regs[2];
    edx = (uint32_t)regs[3];
}

static inline uint32_t
dlp_get_cpuid_max(uint32_t level, uint32_t sig)
{
    int regs[4];
    __cpuid(regs, (int)level);
    return (uint32_t)regs[0];
}

#define __cpuid(level, eax, ebx, ecx, edx) dlp_cpuid(level, eax, ebx, ecx, edx)
#define __cpuid_count(level, count, eax, ebx, ecx, edx)                        \
    dlp_cpuid_count(level, count, eax, ebx, ecx, edx)
#define __get_cpuid_max(level, sig) dlp_get_cpuid_max(level, sig)

#else
#include "cpuid.h"
#endif
