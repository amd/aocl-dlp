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

#ifndef DLP_GEMM_UTILS_KERN_H
#define DLP_GEMM_UTILS_KERN_H

#include "dlp_gemm_types.h"

typedef void (*dlp_gemm_util_l1_op_f32_kernel_t)(const md_t n,
                                                 float*     x,
                                                 const md_t incx);

#define DLP_GEMM_UTIL_L1_OP_KERNEL(V_type, OP_type)                            \
    void dlp_gemm_util_##OP_type##_kernel(const md_t n, V_type* x,             \
                                          const md_t incx)

// AVX512
DLP_GEMM_UTIL_L1_OP_KERNEL(float, f32_gelu_tanh_avx512);
DLP_GEMM_UTIL_L1_OP_KERNEL(float, f32_gelu_erf_avx512);
DLP_GEMM_UTIL_L1_OP_KERNEL(float, f32_softmax_avx512);

// AVX2
DLP_GEMM_UTIL_L1_OP_KERNEL(float, f32_gelu_tanh_avx2);
DLP_GEMM_UTIL_L1_OP_KERNEL(float, f32_gelu_erf_avx2);
DLP_GEMM_UTIL_L1_OP_KERNEL(float, f32_softmax_avx2);

#endif // DLP_GEMM_UTILS_KERN_H
