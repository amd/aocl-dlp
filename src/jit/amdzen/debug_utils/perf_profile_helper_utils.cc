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

#include "perf_profile_helper_utils.hh"

#if (DLP_OS_LINUX != 0) && defined(DLP_JIT_DEBUG)

#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace amdzen::debug::perf_profile {

uint64_t
perfJitDumper::getTimestamp()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

pid_t
perfJitDumper::gettid()
{
    return (pid_t)syscall(SYS_gettid);
}

perfJitDumper::perfJitDumper()
    : jitDumpFile(nullptr)
    , perfMap(nullptr)
    , markerAddr(nullptr)
    , codeIndex(0)
{
#define MAX_FNAME_LEN 128
    // Open JIT dump file in current directory.
    char dumpPath[MAX_FNAME_LEN];
    snprintf(dumpPath, sizeof(dumpPath), "jit-%d.dump", getpid());

    jitDumpFile = fopen(dumpPath, "a+");
    if (!jitDumpFile) {
        fprintf(stderr, "Warning: Failed to create jitdump file: %s\n",
                dumpPath);
        return;
    }

    char perfMapPath[MAX_FNAME_LEN];
    // Open JIT perf map file in current directory.
    snprintf(perfMapPath, sizeof(perfMapPath), "perf-%d.map", getpid());
    perfMap = fopen(perfMapPath, "a+");
    if (!perfMap) {
        fprintf(stderr, "Warning: Failed to create perf map file: %s\n",
                perfMapPath);
        return;
    }
#undef MAX_FNAME_LEN

    // Write header
    jitDumpHeader header = {};
    header.magic         = 0x4A695444; // "JiTD"
    header.version       = 1;
    header.total_size    = sizeof(jitDumpHeader);
    header.elf_mach      = 62; // EM_X86_64
    header.pid           = getpid();
    header.timestamp     = getTimestamp();
    header.flags         = 0;

    fwrite(&header, sizeof(header), 1, jitDumpFile);
    fflush(jitDumpFile);

    // CRITICAL: mmap the file so perf can detect it.
    // This is the "marker" that tells perf a jitdump file exists.
    markerAddr = mmap(nullptr, sysconf(_SC_PAGESIZE), PROT_READ | PROT_EXEC,
                      MAP_PRIVATE, fileno(jitDumpFile), 0);

    if (markerAddr == MAP_FAILED) {
        fprintf(stderr, "Warning: Failed to mmap jitdump marker\n");
        markerAddr = nullptr;
    }
}

perfJitDumper::~perfJitDumper()
{
    std::cout << "\nExecute the following steps to enable annotation of JIT"
              << " kernels in perf report." << std::endl;
    std::cout << "Verify perf-*.map and jit-*.dump files are created."
              << std::endl;
    std::cout << "Execute the following bash commands:" << std::endl;
    std::cout << "# perf inject -j -i perf.data -o perf.data.jitted"
              << std::endl;
    std::cout << "# perf report -i perf.data.jitted" << std::endl;
    std::cout << "In case the perf inject failed, please re run perf record"
              << " with -k 1 option, i.e.\n# perf record -k 1 ..." << std::endl;
    if (markerAddr && markerAddr != MAP_FAILED) {
        munmap(markerAddr, sysconf(_SC_PAGESIZE));
    }
    if (jitDumpFile) {
        fclose(jitDumpFile);
    }
}

void
perfJitDumper::registerCode(const void* code_addr,
                            size_t      code_size,
                            const char* symbol_name)
{
    if ((markerAddr == nullptr) || (!jitDumpFile) || (!perfMap)) {
        return; // Cannot generate dump in this case.
    }

    std::lock_guard<std::mutex> lock(perfMtx);

    // Write to perf map file with format: <address> <size> <symbol_name>
    fprintf(perfMap, "%lx %lx %s\n", (unsigned long)code_addr,
            (unsigned long)code_size, symbol_name);
    fflush(perfMap);

    size_t   name_len = std::strlen(symbol_name) + 1; // Include null terminator
    uint32_t total_size = sizeof(jitCodeLoadRecord) + name_len + code_size;

    jitCodeLoadRecord record = {};
    record.id                = 0; // JIT_CODE_LOAD
    record.total_size        = total_size;
    record.timestamp         = getTimestamp();
    record.pid               = getpid();
    record.tid               = gettid();
    record.vma               = (uint64_t)code_addr;
    record.code_addr         = (uint64_t)code_addr;
    record.code_size         = code_size;
    record.code_index        = codeIndex++;

    // Write record header
    fwrite(&record, sizeof(record), 1, jitDumpFile);

    // Write symbol name (null-terminated)
    fwrite(symbol_name, name_len, 1, jitDumpFile);

    // Write actual machine code bytes - THIS IS THE KEY!
    fwrite(code_addr, code_size, 1, jitDumpFile);

    fflush(jitDumpFile);
}

} // namespace amdzen::debug::perf_profile
#else

#endif
