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

#ifndef DLP_F32_TINY_GEMM_KERNEL_H
#define DLP_F32_TINY_GEMM_KERNEL_H

#include <stdint.h>

/*
 * Arguments of the f32 tiny SGEMM kernels emitted at library init.
 * Strides are in elements. cs_c must be 1; the leaves do not read it.
 * The compute loops of RV also require cs_b == 1. RD requires cs_a == 1
 * and rs_b == 1. alpha must be nonzero. beta == 0 does not read C.
 * No post-ops. nc is the RV column block: a positive multiple of the RV
 * panel width (64 on AVX-512, 16 on AVX2). RD does not read it.
 */
typedef struct
{
    int64_t      m;
    int64_t      n;
    int64_t      k;
    const float* a;
    int64_t      rs_a;
    int64_t      cs_a;
    const float* b;
    int64_t      rs_b;
    int64_t      cs_b;
    float*       c;
    int64_t      rs_c;
    int64_t      cs_c;
    float        alpha;
    float        beta;
    int64_t      nc;
} dlp_sgemm_tiny_jit_args;

typedef void (*dlp_sgemm_tiny_jit_fn)(const dlp_sgemm_tiny_jit_args* args);

#ifdef __cplusplus
extern "C"
{
#endif

    /* Each accessor emits its kernel once on first use and returns NULL if
     * generation fails. */
    dlp_sgemm_tiny_jit_fn dlp_sgemm_tiny_jit_rv_avx512(void);

    dlp_sgemm_tiny_jit_fn dlp_sgemm_tiny_jit_rd_avx512(void);

    dlp_sgemm_tiny_jit_fn dlp_sgemm_tiny_jit_rd_avx512_wide(void);

    dlp_sgemm_tiny_jit_fn dlp_sgemm_tiny_jit_rv_avx2(void);

    dlp_sgemm_tiny_jit_fn dlp_sgemm_tiny_jit_rd_avx2(void);

#ifdef __cplusplus
}
#endif

#endif
