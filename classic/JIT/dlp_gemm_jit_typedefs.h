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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
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

#ifndef JIT_TYPEDEFS_H
#define JIT_TYPEDEFS_H

typedef struct
{
    bool m_loop;
    bool alpha_scale;
    int  beta_scale;
    md_t MR;
    md_t NR;
    bool generate_mask;
} dlp_gemm_jit_inputs_t;

typedef struct
{
    uint64_t  m;
    uint64_t  n;
    uint64_t  k;
    uint64_t  rs_a;
    uint64_t  cs_a;
    uint64_t  rs_b;
    uint64_t  cs_b;
    uint64_t  rs_c;
    uint64_t  cs_c;
    bfloat16* a;
    bfloat16* b;
    float*    c;
    uint64_t  ps_a2;
    uint64_t  m_iter;
    uint64_t  k_iter_before_prefetch;
    uint64_t  k_iter_after_prefetch;
    uint64_t  k_left;
    float*    alpha;
    float*    beta;
    uint32_t  mask16;
    uint16_t  mask32;
#ifdef BPREFETCH_JIT
    uint64_t bprefetch_dist;
#endif
} dlp_gemm_jit_params_t;

typedef enum
{
    DLP_BETA_ZERO      = 0,
    DLP_BETA_ONE       = 1,
    DLP_BETA_MINUS_ONE = 2,
    DLP_BETA_GEN       = 3
} beta_val;
#endif
