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

#include <gtest/gtest.h>

#include "jit/f32_tiny_gemm_kernel.h"

/*
 * Each tiny F32 leaf must generate a non-null kernel. A null accessor makes
 * dlp_sgemm_tiny_f32 fall back to the framework, so the YAML sweeps would
 * still pass without running the leaf. This test catches that case directly.
 */
TEST(TinySgemmJitGenerate, AllLeavesGenerate)
{
    EXPECT_NE(dlp_sgemm_tiny_jit_rv_avx512(), nullptr);
    EXPECT_NE(dlp_sgemm_tiny_jit_rd_avx512(), nullptr);
    EXPECT_NE(dlp_sgemm_tiny_jit_rd_avx512_wide(), nullptr);
    EXPECT_NE(dlp_sgemm_tiny_jit_rv_avx2(), nullptr);
    EXPECT_NE(dlp_sgemm_tiny_jit_rd_avx2(), nullptr);
}

/* Repeated calls must return the same cached kernel, not regenerate it. */
TEST(TinySgemmJitGenerate, AccessorsAreCached)
{
    EXPECT_EQ(dlp_sgemm_tiny_jit_rv_avx512(), dlp_sgemm_tiny_jit_rv_avx512());
    EXPECT_EQ(dlp_sgemm_tiny_jit_rd_avx2(), dlp_sgemm_tiny_jit_rd_avx2());
}
