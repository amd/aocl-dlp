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

#include <cstdio>
#include <mutex>

#include "classic/dlp_base_types.h"

#if (DLP_OS_LINUX != 0) && defined(DLP_JIT_DEBUG)

namespace amdzen::debug::perf_profile {

// JitDump file format structures (Linux perf standard)
struct jitDumpHeader
{
    uint32_t magic;   // 0x4A695444 "JiTD"
    uint32_t version; // 1
    uint32_t total_size;
    uint32_t elf_mach; // 62 for x86_64
    uint32_t pad1;
    uint32_t pid;
    uint64_t timestamp;
    uint64_t flags;
};

struct jitCodeLoadRecord
{
    uint32_t id; // 0 = JIT_CODE_LOAD
    uint32_t total_size;
    uint64_t timestamp;
    uint32_t pid;
    uint32_t tid;
    uint64_t vma;        // Virtual memory address
    uint64_t code_addr;  // Code address (same as vma)
    uint64_t code_size;  // Size of code
    uint64_t code_index; // Unique index
    // Followed by: null-terminated symbol name
    // Followed by: actual code bytes
};

class perfJitDumper
{
    // Using C style file io handling so to enable the usage of mmap.
    FILE*      jitDumpFile;
    FILE*      perfMap;
    void*      markerAddr; // mmap marker for perf detection
    std::mutex perfMtx;
    uint64_t   codeIndex;

    static uint64_t getTimestamp();
    static pid_t    gettid();

    perfJitDumper();
    ~perfJitDumper();

    perfJitDumper(const perfJitDumper&)            = delete;
    perfJitDumper& operator=(const perfJitDumper&) = delete;
    perfJitDumper(perfJitDumper&&)                 = delete;
    perfJitDumper& operator=(perfJitDumper&&)      = delete;

  public:
    static perfJitDumper& instance()
    {
        static perfJitDumper dumper;
        return dumper;
    }

    void registerCode(const void* code_addr,
                      size_t      code_size,
                      const char* symbol_name);
};

} // namespace amdzen::debug::perf_profile

#define DLP_ENABLE_PERF_PROFILE_FOR_JIT_KERNEL(code_addr, code_size,           \
                                               symbol_name)                    \
    amdzen::debug::perf_profile::perfJitDumper::instance().registerCode(       \
        (code_addr), (code_size), (symbol_name));

#else

#define DLP_ENABLE_PERF_PROFILE_FOR_JIT_KERNEL(code_addr, code_size,           \
                                               symbol_name)

#endif
