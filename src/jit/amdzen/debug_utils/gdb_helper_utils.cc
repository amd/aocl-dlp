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

#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>

#if defined(__linux__)
#include <unistd.h>
#endif

#include "gdb_helper_utils.hh"

namespace amdzen::debug::gdb {

int gdbJitHelperUtils::aocl_jit_debug_break = 1;

void
gdbJitHelperUtils::helpDebugInGDB(void* kernelAddress)
{
    uint64_t thisPid = 0;
#if defined(__linux__)
    thisPid = (uint64_t)getpid();
#endif

    if (aocl_jit_debug_break == 1) {
        // Manual breakpoint - just infinite loop
        std::cout
            << "=== MANUAL BREAKPOINT JIT Debugging ===\n"
            << "--- Currently only works with single thread.\n"
            << "--- Press <ctrl> + C inside gdb to interrupt the default "
            << "sleep state the\n    manual breakpoint enters.\n"
            << "--- aocl_jit_debug_break=0 means continue and break at "
            << "next manual breakpoint.\n--- aocl_jit_debug_break=2 means "
            << "continue without breaking at any manual breakpoint.\n\n";
        std::cout << "=== Execute the following instructions in another "
                  << "terminal ===\n"
                  << "--- # denotes a command line instruction whereas "
                  << "(gdb) denotes a gdb instruction.\n\n";
        std::cout
            << "# gdb -p " << thisPid << "\n"
            << "(gdb) set language c++\n"
            << "(gdb) set variable "
            << "amdzen::debug::gdb::gdbJitHelperUtils::aocl_jit_debug_break"
            << " = 0\n"
            << "(gdb) break *" << kernelAddress << "\n"
            << "(gdb) display/i $pc\n"
            << "(gdb) continue\n";
    }

    while (aocl_jit_debug_break == 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (aocl_jit_debug_break == 0) {
        aocl_jit_debug_break = 1; // reset for next time
    }
}

} // namespace amdzen::debug::gdb
