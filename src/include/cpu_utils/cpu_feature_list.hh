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

#include <cstdint>

namespace dlp::cpu_utils {

enum class isaFeature : uint32_t
{
    invalid = 0,

    // x86_64 ISA features
    sse3,
    ssse3,
    sse41,
    sse42,
    avx,
    avx2,
    fma3,
    fma4,
    avx512f,
    avx512dq,
    avx512pf,
    avx512er,
    avx512cd,
    avx512bw,
    avx512vl,
    avx512vnni,
    avx512bf16,
    avx512fp16,
    avx512vbmi,
    avxvnni,
    avx512vp2intersect,
    movdiri,
    movdir64b,
    datapath_fp128,
    datapath_fp256,
    datapath_fp512,

    // Other ISA Features

    feature_limit
};

enum class cpuVendor : uint32_t
{
    invalid = 0,

    // x86_64 vendors
    amd,
    intel,

    // Other ISA vendors

    vendor_limit
};

} // namespace dlp::cpu_utils
