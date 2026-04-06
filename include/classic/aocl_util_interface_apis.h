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

#ifndef AOCL_DLP_UTIL_INTERFACE_H
#define AOCL_DLP_UTIL_INTERFACE_H

#include "classic/aocl_bf16_type.h"
#include "classic/dlp_base_types.h"

/**
 * @brief Performs GELU activation (tanh approximation) on a float vector.
 * @param[in] n Number of elements in the vector.
 * @param[inout] x Pointer to the vector.
 * @param[in] incx Stride between consecutive elements in the vector.
 */
DLP_CLASSIC_EXPORT void
aocl_gemm_gelu_tanh_f32(const md_t n, float* x, const md_t incx);

/**
 * @brief Performs GELU activation (erf approximation) on a float vector.
 * @param[in] n Number of elements in the vector.
 * @param[inout] x Pointer to the vector.
 * @param[in] incx Stride between consecutive elements in the vector.
 */
DLP_CLASSIC_EXPORT void
aocl_gemm_gelu_erf_f32(const md_t n, float* x, const md_t incx);

/**
 * @brief Performs softmax activation on a float vector.
 * @param[in] n Number of elements in the vector.
 * @param[inout] x Pointer to the vector.
 * @param[in] incx Stride between consecutive elements in the vector.
 */
DLP_CLASSIC_EXPORT void
aocl_gemm_softmax_f32(const md_t n, float* x, const md_t incx);

#endif // AOCL_DLP_UTIL_INTERFACE_H
