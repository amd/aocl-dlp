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
#ifndef AOCL_DLP_GEMM_GELU_DEF_AVX512_H
#define AOCL_DLP_GEMM_GELU_DEF_AVX512_H

/* TANH GeLU (x) = 0.5* x * (1 + tanh ( 0.797884 * ( x + ( 0.044715 * x^3 ) ) )
 * )  */
#define GELU_TANH_F32_AVX512_DEF(reg, r, r2, x, z, dn, x_tanh, q)              \
                                                                               \
    r2     = _mm512_mul_ps(reg, reg);                                          \
    r2     = _mm512_mul_ps(r2, reg);                                           \
    x_tanh = _mm512_fmadd_ps(_mm512_set1_ps(0.044715), r2, reg);               \
    x_tanh = _mm512_mul_ps(x_tanh, _mm512_set1_ps(0.797884));                  \
                                                                               \
    /*x_tanh = tanhf(x_tanh) */                                                \
    TANHF_AVX512(x_tanh, r, r2, x, z, dn, q);                                  \
                                                                               \
    x_tanh = _mm512_add_ps(x_tanh, _mm512_set1_ps(1));                         \
    x_tanh = _mm512_mul_ps(x_tanh, reg);                                       \
    reg    = _mm512_mul_ps(x_tanh, _mm512_set1_ps(0.5));

/* ERF GeLU (x) = 0.5* x * (1 + erf (x * 0.707107 )) */
#define GELU_ERF_F32_AVX512_DEF(reg, y, r, r2)                                 \
    r = _mm512_mul_ps(reg, _mm512_set1_ps(0.70710678118654f));                 \
    y = _mm512_setzero_ps();                                                   \
    ERF_AOCL_DLP_AVX512(y, r);                                                 \
    r2  = _mm512_add_ps(y, _mm512_set1_ps(1));                                 \
    r2  = _mm512_mul_ps(r2, reg);                                              \
    reg = _mm512_mul_ps(r2, _mm512_set1_ps(0.5));

#endif // AOCL_DLP_GEMM_GELU_DEF_AVX512_H
