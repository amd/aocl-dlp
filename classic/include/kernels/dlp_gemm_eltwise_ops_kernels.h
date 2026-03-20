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

#ifndef DLP_GEMM_ELTWISE_OPS_KERN_H
#define DLP_GEMM_ELTWISE_OPS_KERN_H

#include "classic/aocl_bf16_type.h"
#include "dlp_gemm_post_ops.h"

#define DLP_GEMM_ELTWISE_OPS_KERNEL(A_type, B_type, LP_SFX)                    \
    void dlp_gemm_eltwise_ops_kernel_##LP_SFX(                                 \
        const md_t m0, const md_t n0, const A_type* a, const md_t rs_a,        \
        const md_t cs_a, B_type* b, const md_t rs_b, const md_t cs_b,          \
        dlp_gemm_post_op* post_ops_list, dlp_gemm_post_op_attr post_ops_attr)

DLP_GEMM_ELTWISE_OPS_KERNEL(bfloat16, float, bf16of32_6x64);
DLP_GEMM_ELTWISE_OPS_KERNEL(float, float, f32of32_6x64);

#define DLP_GEMM_ELTWISE_OPS_M_FRINGE_KERNEL(A_type, B_type, LP_SFX)           \
    void dlp_gemm_eltwise_ops_kernel_##LP_SFX(                                 \
        const md_t n0, const A_type* a, const md_t rs_a, const md_t cs_a,      \
        B_type* b, const md_t rs_b, const md_t cs_b,                           \
        dlp_gemm_post_op* post_ops_list, dlp_gemm_post_op_attr post_ops_attr)

DLP_GEMM_ELTWISE_OPS_M_FRINGE_KERNEL(bfloat16, float, bf16of32_5x64);
DLP_GEMM_ELTWISE_OPS_M_FRINGE_KERNEL(bfloat16, float, bf16of32_4x64);
DLP_GEMM_ELTWISE_OPS_M_FRINGE_KERNEL(bfloat16, float, bf16of32_3x64);
DLP_GEMM_ELTWISE_OPS_M_FRINGE_KERNEL(bfloat16, float, bf16of32_2x64);
DLP_GEMM_ELTWISE_OPS_M_FRINGE_KERNEL(bfloat16, float, bf16of32_1x64);
DLP_GEMM_ELTWISE_OPS_M_FRINGE_KERNEL(float, float, f32of32_5x64);
DLP_GEMM_ELTWISE_OPS_M_FRINGE_KERNEL(float, float, f32of32_4x64);
DLP_GEMM_ELTWISE_OPS_M_FRINGE_KERNEL(float, float, f32of32_3x64);
DLP_GEMM_ELTWISE_OPS_M_FRINGE_KERNEL(float, float, f32of32_2x64);
DLP_GEMM_ELTWISE_OPS_M_FRINGE_KERNEL(float, float, f32of32_1x64);

#endif // DLP_GEMM_ELTWISE_OPS_KERN_H
