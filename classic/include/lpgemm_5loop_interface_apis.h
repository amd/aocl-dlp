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

#ifndef LPGEMM_5LOOP_INTF_H
#define LPGEMM_5LOOP_INTF_H

#include "classic/aocl_bf16_type.h"
#include "classic/aocl_fp16_type.h"
#include "lpgemm_ops_bundle.h"
#include "lpgemm_post_ops.h"
#include "lpgemm_types.h"
#include "runtime/dlp_runtime.h"

#define LPGEMM_TINY(A_type, B_type, C_type, LP_SFX)                            \
    void lpgemm_rowvar_tiny_##LP_SFX(                                          \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        AOCL_MEMORY_TAG mtag_b, C_type* c, const md_t rs_c, const md_t cs_c,   \
        const C_type alpha, const C_type beta, lpgemm_cntx_t* lcntx,           \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale)

LPGEMM_TINY(float, float, float, f32f32f32of32);
LPGEMM_TINY(bfloat16, bfloat16, float, bf16bf16f32of32);

/**
 * @brief Unified 5-loop GEMM macro with lpgemm_ops_bundle_t interface.
 *
 * @param A_type          Type of A matrix elements
 * @param B_type          Type of B matrix elements
 * @param C_type          Type used for alpha/beta scalars
 * @param C_type_actual   Actual type of C matrix (may differ for sym_quant)
 * @param LP_SFX          Suffix for function name
 * @param MTAG_B_CONST    "const" for const mtag_b, empty for mutable
 */
#define LPGEMM_5LOOP_UNIFIED(A_type, B_type, C_type, C_type_actual, LP_SFX,    \
                             MTAG_B_CONST)                                     \
    void lpgemm_rowvar_##LP_SFX(                                               \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        MTAG_B_CONST AOCL_MEMORY_TAG mtag_b, C_type_actual* c,                 \
        const md_t rs_c, const md_t cs_c, const C_type alpha,                  \
        const C_type beta, dlp_rntm_t* rntm, lpgemm_thrinfo_t* thread,         \
        lpgemm_cntx_t* lcntx, const lpgemm_ops_bundle_t* ops,                  \
        DLP_TYPE c_downscale)

// BASE variants (mutable rs_b/cs_b/mtag_b for runtime modification)
LPGEMM_5LOOP_UNIFIED(uint8_t, int8_t, int32_t, int32_t, u8s8s32o32,
                     /* mutable */);
LPGEMM_5LOOP_UNIFIED(float, float, float, float, f32f32f32of32, /* mutable */);
LPGEMM_5LOOP_UNIFIED(bfloat16, bfloat16, float, float, bf16bf16f32of32,
                     /* mutable */);
LPGEMM_5LOOP_UNIFIED(int8_t, int8_t, int32_t, int32_t, s8s8s32o32,
                     /* mutable */);

// FP16 variant
LPGEMM_5LOOP_UNIFIED(float16, float16, float16, float16, f16f16f16of16,
                     /* mutable */);

// MP variant (const rs_b/cs_b/mtag_b)
LPGEMM_5LOOP_UNIFIED(bfloat16, int8_t, float, float, bf16s4f32of32, const);

// GRP variant (const rs_b/cs_b/mtag_b, C forced to float)
LPGEMM_5LOOP_UNIFIED(
    int8_t, int8_t, int32_t, float, s8s8s32o32_sym_quant, const);

// Q variants (const rs_b/cs_b/mtag_b)
LPGEMM_5LOOP_UNIFIED(bfloat16, int8_t, int32_t, int32_t, bf16s8s32os32, const);
LPGEMM_5LOOP_UNIFIED(float, int8_t, int32_t, int32_t, f32s8s32os32, const);

#define LPGEMM_5LOOP_F32_FALLBACK(A_type, B_type, C_type, LP_SFX)              \
    void lpgemm_rowvar_f32_fallback_##LP_SFX(                                  \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, md_t rs_b, md_t cs_b, AOCL_MEMORY_TAG mtag_b,         \
        C_type* c, const md_t rs_c, const md_t cs_c, const C_type alpha,       \
        const C_type beta, dlp_rntm_t* rntm, lpgemm_thrinfo_t* thread,         \
        lpgemm_cntx_t* lcntx, lpgemm_post_op* post_op_list,                    \
        DLP_TYPE c_downscale)

LPGEMM_5LOOP_F32_FALLBACK(bfloat16, bfloat16, float, bf16bf16f32of32);

#define LPGEMM_5LOOP_AVX512BF16(A_type, B_type, C_type, LP_SFX)                \
    void lpgemm_rowvar_avx512bf16_##LP_SFX(                                    \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, md_t rs_b, md_t cs_b, AOCL_MEMORY_TAG mtag_b,         \
        C_type* c, const md_t rs_c, const md_t cs_c, const C_type alpha,       \
        const C_type beta, dlp_rntm_t* rntm, lpgemm_thrinfo_t* thread,         \
        lpgemm_cntx_t* lcntx, lpgemm_post_op* post_op_list,                    \
        DLP_TYPE c_downscale)

LPGEMM_5LOOP_AVX512BF16(bfloat16, bfloat16, float, bf16bf16f32of32);

#define LPGEMV_TINY(A_type, B_type, C_type, LP_SFX)                            \
    void lpgemv_rowvar_tiny_##LP_SFX(                                          \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        const AOCL_MEMORY_TAG mtag_b, C_type* c, const md_t rs_c,              \
        const md_t cs_c, const C_type alpha, const C_type beta,                \
        lpgemm_cntx_t* lcntx, lpgemm_post_op* post_op_list,                    \
        DLP_TYPE c_downscale)

LPGEMV_TINY(float, float, float, f32f32f32of32);
LPGEMV_TINY(bfloat16, bfloat16, float, bf16bf16f32of32);

#define LPGEMV(A_type, B_type, C_type, LP_SFX)                                 \
    void lpgemv_rowvar_##LP_SFX(                                               \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        const AOCL_MEMORY_TAG mtag_b, C_type* c, const md_t rs_c,              \
        const md_t cs_c, const C_type alpha, const C_type beta,                \
        dlp_rntm_t* rntm, lpgemm_thrinfo_t* thread, lpgemm_cntx_t* lcntx,      \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale)

LPGEMV(float, float, float, f32f32f32of32);
LPGEMV(bfloat16, bfloat16, float, bf16bf16f32of32);
LPGEMV(uint8_t, int8_t, int32_t, u8s8s32os32);
LPGEMV(int8_t, int8_t, int32_t, s8s8s32os32);
LPGEMV(float16, float16, float16, f16f16f16of16);

#define LPGEMV_F32_FALLBACK(A_type, B_type, C_type, LP_SFX)                    \
    void lpgemv_rowvar_f32_fallback_##LP_SFX(                                  \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        const AOCL_MEMORY_TAG mtag_b, C_type* c, const md_t rs_c,              \
        const md_t cs_c, const C_type alpha, const C_type beta,                \
        dlp_rntm_t* rntm, lpgemm_thrinfo_t* thread, lpgemm_cntx_t* lcntx,      \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale)

LPGEMV_F32_FALLBACK(bfloat16, bfloat16, float, bf16bf16f32of32);

#define LPGEMV2(A_type, B_type, C_type, LP_SFX)                                \
    void lpgemv_rowvar_##LP_SFX(                                               \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        const AOCL_MEMORY_TAG mtag_b, float* c, const md_t rs_c,               \
        const md_t cs_c, const C_type alpha, const C_type beta,                \
        dlp_rntm_t* rntm, lpgemm_thrinfo_t* thread, lpgemm_cntx_t* lcntx,      \
        lpgemm_group_post_op* grp_post_op_list, lpgemm_post_op* post_op_list,  \
        DLP_TYPE c_downscale)

LPGEMV2(int8_t, int8_t, int32_t, s8s8s32os32_sym_quant);

#define LPGEMV3(A_type, B_type, C_type, LP_SFX)                                \
    void lpgemv_rowvar_##LP_SFX(                                               \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        const AOCL_MEMORY_TAG mtag_b, C_type* c, const md_t rs_c,              \
        const md_t cs_c, const C_type alpha, const C_type beta,                \
        dlp_rntm_t* rntm, lpgemm_thrinfo_t* thread, lpgemm_cntx_t* lcntx,      \
        dlp_quant_op* a_pre_quant, lpgemm_post_op* post_op_list,               \
        DLP_TYPE c_downscale)

LPGEMV3(bfloat16, int8_t, int32_t, bf16s8s32os32);
LPGEMV3(float, int8_t, int32_t, f32s8s32os32);

#endif // LPGEMM_5LOOP_INTF_H
