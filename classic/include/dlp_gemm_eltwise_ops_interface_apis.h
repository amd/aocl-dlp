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

#ifndef DLP_GEMM_POSTOP_INTF_H
#define DLP_GEMM_POSTOP_INTF_H

#include "classic/aocl_bf16_type.h"
#include "dlp_gemm_post_ops.h"
#include "dlp_gemm_types.h"
#include "runtime/dlp_runtime.h"

#define DLP_GEMM_ELTWISE_OPS_IFACE(A_type, B_type, LP_SFX)                     \
    void dlp_gemm_eltwise_ops_interface_##LP_SFX(                              \
        const md_t m, const md_t n, const A_type* a, const md_t rs_a,          \
        const md_t cs_a, B_type* b, const md_t rs_b, const md_t cs_b,          \
        dlp_rntm_t* rntm, dlp_gemm_thrinfo_t* thread,                          \
        dlp_gemm_eltwise_ops_cntx_t* lcntx, dlp_gemm_post_op* post_op_list,    \
        DLP_TYPE c_downscale)

DLP_GEMM_ELTWISE_OPS_IFACE(bfloat16, float, bf16of32);
DLP_GEMM_ELTWISE_OPS_IFACE(float, float, f32of32);

#endif // DLP_GEMM_POSTOP_INTF_H
