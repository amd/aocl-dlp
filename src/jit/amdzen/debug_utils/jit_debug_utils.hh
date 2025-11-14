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
#include <string>

#include "classic/dlp_base_types.h"
#include "perf_profile_helper_utils.hh"

#ifdef DLP_JIT_DEBUG

namespace amdzen::debug::jit {

class jitDebugUtils
{
  public:
    // Function to dump JIT code to a file for debugging purposes. This
    // function will create a file with the name <code_name>_<m>x<n>.bin".
    // The code will be dumped in binary format.
    static void dumpAndMonitorJitCode(const void* code,
                                      int         code_size,
                                      const char* code_name,
                                      int         m,
                                      int         n,
                                      bool        isLtKernel,
                                      int         index)
    {
#define MAX_FNAME_LEN 256
        if (code) {
            std::string ltSfx{ "" };
            if (isLtKernel) {
                ltSfx = "_lt";
            }
            char fname[MAX_FNAME_LEN + 1];
            // TODO (Roma): support prefix for code / linux perf dumps
            snprintf(fname, MAX_FNAME_LEN, "idx%d_%s_%dx%s%d.bin", index,
                     code_name, m, ltSfx.c_str(), n);

            DLP_ENABLE_PERF_PROFILE_FOR_JIT_KERNEL(code, code_size, fname);

            FILE* fp = fopen(fname, "wb+");
            // Failure to dump code is not fatal
            if (fp) {
                int unused = fwrite(code, code_size, 1, fp);
                fclose(fp);
            }
        }
#undef MAX_FNAME_LEN
    }
};

} // namespace amdzen::debug::jit

#define DLP_ENABLE_JIT_DUMP_AND_MONITOR(code, code_size, code_name, m, n,      \
                                        isLtKernel, index)                     \
    amdzen::debug::jit::jitDebugUtils::dumpAndMonitorJitCode(                  \
        (code), (code_size), (code_name), (m), (n), (isLtKernel), (index));

#else

#define DLP_ENABLE_JIT_DUMP_AND_MONITOR(code, code_size, code_name, m, n,      \
                                        isLtKernel, index)

#endif
