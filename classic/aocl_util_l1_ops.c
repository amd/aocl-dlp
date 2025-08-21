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

#include "classic/aocl_util_interface_apis.h"
#include "config/lpgemm_config.h"
#include "gemm_utils/lpgemm_utils.h"
#include "kernels/lpgemm_utils_kernels.h"
#include "lpgemm_types.h"
#include "sys_utils/dlp_cpu_arch.h"

void
aocl_gemm_gelu_tanh_f32(const md_t n, float* x, const md_t incx)
{
    // Check if AVX2 ISA is supported, lpgemm u8s8s16os16 matmul only works with
    // it.
    if (dlp_cpuid_is_avx2fma3_supported() == FALSE) {
        dlp_print_msg(" AVX2 ISA not supported by processor, AOCL GEMM "
                      "utility l1 operations not supported.",
                      __FILE__, __LINE__);
        return; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    if ((n <= 0) || (x == NULL) || (incx <= 0)) {
        return; // Error.
    }

    lpgemm_util_cntx_t* lutil_cntx_g =
        lpgemm_util_get_global_cntx_obj(F32_GELU_TANH);
    ((lpgemm_util_l1_op_f32_kernel_t)lutil_cntx_g->kern_fun_ptr)(n, x, incx);
}

void
aocl_gemm_gelu_erf_f32(const md_t n, float* x, const md_t incx)
{
    // Check if AVX2 ISA is supported, lpgemm u8s8s16os16 matmul only works with
    // it.
    if (dlp_cpuid_is_avx2fma3_supported() == FALSE) {
        dlp_print_msg(" AVX2 ISA not supported by processor, AOCL GEMM "
                      "utility l1 operations not supported.",
                      __FILE__, __LINE__);
        return; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    if ((n <= 0) || (x == NULL) || (incx <= 0)) {
        return; // Error.
    }

    lpgemm_util_cntx_t* lutil_cntx_g =
        lpgemm_util_get_global_cntx_obj(F32_GELU_ERF);
    ((lpgemm_util_l1_op_f32_kernel_t)lutil_cntx_g->kern_fun_ptr)(n, x, incx);
}

void
aocl_gemm_softmax_f32(const md_t n, float* x, const md_t incx)
{
    // Check if AVX2 ISA is supported, lpgemm u8s8s16os16 matmul only works with
    // it.
    if (dlp_cpuid_is_avx2fma3_supported() == FALSE) {
        dlp_print_msg(" AVX2 ISA not supported by processor, AOCL GEMM "
                      "utility l1 operations not supported.",
                      __FILE__, __LINE__);
        return; // Error.
    }

    // Set MC, NC, KC, NR, MR.
    aocl_lpgemm_init_global_cntx();

    if ((n <= 0) || (x == NULL) || (incx <= 0)) {
        return; // Error.
    }

    lpgemm_util_cntx_t* lutil_cntx_g =
        lpgemm_util_get_global_cntx_obj(F32_SOFTMAX);
    ((lpgemm_util_l1_op_f32_kernel_t)lutil_cntx_g->kern_fun_ptr)(n, x, incx);
}
