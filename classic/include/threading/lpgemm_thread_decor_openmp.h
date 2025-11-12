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

#ifndef LPGEMM_THREAD_DECOR_OPENMP_H
#define LPGEMM_THREAD_DECOR_OPENMP_H

#include "classic/aocl_bf16_type.h"
#include "lpgemm_post_ops.h"
#include "lpgemm_types.h"
#include "runtime/dlp_runtime.h"

#ifdef DLP_ENABLE_OPENMP

#define GEN_LPGEMM_OPENMP_DECORATOR_FN(A_type, B_type, C_type, LPGEMM_SFX)     \
    void lpgemm_##LPGEMM_SFX##_openmp_thread_decorator(                        \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        const AOCL_MEMORY_TAG mtag_b, C_type* c, const md_t rs_c,              \
        const md_t cs_c, const C_type alpha, const C_type beta,                \
        dlp_rntm_t* rntm_g, lpgemm_cntx_t* lcntx,                              \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale);

GEN_LPGEMM_OPENMP_DECORATOR_FN(uint8_t, int8_t, int32_t, u8s8s32o32)
GEN_LPGEMM_OPENMP_DECORATOR_FN(bfloat16, bfloat16, float, bf16bf16f32of32)
GEN_LPGEMM_OPENMP_DECORATOR_FN(float, float, float, f32f32f32of32)
GEN_LPGEMM_OPENMP_DECORATOR_FN(int8_t, int8_t, int32_t, s8s8s32o32)

#define GEN_BATCH_LPGEMM_OPENMP_DECORATOR_FN(A_type, B_type, C_type,           \
                                             LPGEMM_SFX)                       \
    void batch_lpgemm_##LPGEMM_SFX##_openmp_thread_decorator(                  \
        const md_t group_size, const md_t* m, const md_t* n, const md_t* k,    \
        const A_type** a, const md_t* rs_a, const md_t* cs_a,                  \
        const AOCL_MEMORY_TAG* mtag_a, const B_type** b, const md_t* rs_b,     \
        const md_t* cs_b, const AOCL_MEMORY_TAG* mtag_b, C_type** c,           \
        const md_t* rs_c, const md_t* cs_c, const C_type alpha,                \
        const C_type beta, dlp_rntm_t* rntm_g, lpgemm_cntx_t* lcntx,           \
        lpgemm_post_op(*post_op_list), DLP_TYPE c_downscale);

GEN_BATCH_LPGEMM_OPENMP_DECORATOR_FN(bfloat16, bfloat16, float, bf16bf16f32of32)
GEN_BATCH_LPGEMM_OPENMP_DECORATOR_FN(float, float, float, f32f32f32of32)
GEN_BATCH_LPGEMM_OPENMP_DECORATOR_FN(uint8_t, int8_t, int32_t, u8s8s32o32)
GEN_BATCH_LPGEMM_OPENMP_DECORATOR_FN(int8_t, int8_t, int32_t, s8s8s32o32)

#define GEN_BATCH_LPGEMM_OPENMP_DECORATOR_FN_MXP(A_type, B_type, C_type,       \
                                                 LPGEMM_SFX)                   \
    void batch_lpgemm_##LPGEMM_SFX##_openmp_thread_decorator(                  \
        const md_t group_size, const md_t* m, const md_t* n, const md_t* k,    \
        const A_type** a, const md_t* rs_a, const md_t* cs_a,                  \
        const AOCL_MEMORY_TAG* mtag_a, const B_type** b, const md_t* rs_b,     \
        const md_t* cs_b, AOCL_MEMORY_TAG* mtag_b, C_type** c,                 \
        const md_t* rs_c, const md_t* cs_c, const C_type alpha,                \
        const C_type beta, dlp_rntm_t* rntm_g, lpgemm_cntx_t* lcntx,           \
        lpgemm_pre_op(*pre_op_list), lpgemm_post_op(*post_op_list),            \
        DLP_TYPE c_downscale);

GEN_BATCH_LPGEMM_OPENMP_DECORATOR_FN_MXP(bfloat16, int8_t, float, bf16s4f32of32)

#define GEN_LPGEMM_OPENMP_DECORATOR_FN1(A_type, B_type, C_type, LPGEMM_SFX)    \
    void lpgemm_##LPGEMM_SFX##_openmp_thread_decorator(                        \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        AOCL_MEMORY_TAG mtag_b, C_type* c, const md_t rs_c, const md_t cs_c,   \
        const C_type alpha, const C_type beta, dlp_rntm_t* rntm_g,             \
        lpgemm_cntx_t* lcntx, lpgemm_pre_op* pre_op_list,                      \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale);

GEN_LPGEMM_OPENMP_DECORATOR_FN1(bfloat16, int8_t, float, bf16s4f32of32)

#define GEN_LPGEMM_OPENMP_DECORATOR_FN2(A_type, B_type, C_type, LPGEMM_SFX)    \
    void lpgemm_##LPGEMM_SFX##_openmp_thread_decorator(                        \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        AOCL_MEMORY_TAG mtag_b, float* c, const md_t rs_c, const md_t cs_c,    \
        const C_type alpha, const C_type beta, dlp_rntm_t* rntm_g,             \
        lpgemm_cntx_t* lcntx, lpgemm_group_post_op* grp_post_op_list,          \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale);

GEN_LPGEMM_OPENMP_DECORATOR_FN2(int8_t, int8_t, int32_t, s8s8s32o32_sym_quant)

#define GEN_LPGEMM_OPENMP_DECORATOR_FN3(A_type, B_type, C_type, LPGEMM_SFX)    \
    void lpgemm_##LPGEMM_SFX##_openmp_thread_decorator(                        \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        AOCL_MEMORY_TAG mtag_b, C_type* c, const md_t rs_c, const md_t cs_c,   \
        const C_type alpha, const C_type beta, dlp_rntm_t* rntm_g,             \
        lpgemm_cntx_t* lcntx, dlp_quant_op* a_pre_quant,                       \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale);

GEN_LPGEMM_OPENMP_DECORATOR_FN3(bfloat16, int8_t, int32_t, bf16s8s32os32)

#define GEN_UTIL_ELTWISE_OPS_OPENMP_DECORATOR_FN(A_type, B_type, LPGEMM_SFX)   \
    void lpgemm_eltwise_ops_##LPGEMM_SFX##_openmp_thread_decorator(            \
        const md_t m, const md_t n, const A_type* a, const md_t rs_a,          \
        const md_t cs_a, B_type* b, const md_t rs_b, const md_t cs_b,          \
        dlp_rntm_t* rntm_g, lpgemm_eltwise_ops_cntx_t* lcntx,                  \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale);

GEN_UTIL_ELTWISE_OPS_OPENMP_DECORATOR_FN(bfloat16, float, bf16of32)
GEN_UTIL_ELTWISE_OPS_OPENMP_DECORATOR_FN(float, float, f32of32)

#else

#define GEN_LPGEMM_DECORATOR_FN(A_type, B_type, C_type, LPGEMM_SFX)            \
    void lpgemm_##LPGEMM_SFX##_thread_decorator(                               \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        const AOCL_MEMORY_TAG mtag_b, C_type* c, const md_t rs_c,              \
        const md_t cs_c, const C_type alpha, const C_type beta,                \
        dlp_rntm_t* rntm_g, lpgemm_cntx_t* lcntx,                              \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale);

GEN_LPGEMM_DECORATOR_FN(uint8_t, int8_t, int32_t, u8s8s32o32)
GEN_LPGEMM_DECORATOR_FN(bfloat16, bfloat16, float, bf16bf16f32of32)
GEN_LPGEMM_DECORATOR_FN(float, float, float, f32f32f32of32)
GEN_LPGEMM_DECORATOR_FN(int8_t, int8_t, int32_t, s8s8s32o32)

#define GEN_BATCH_LPGEMM_DECORATOR_FN(A_type, B_type, C_type, LPGEMM_SFX)      \
    void batch_lpgemm_##LPGEMM_SFX##_thread_decorator(                         \
        const md_t group_size, const md_t* m, const md_t* n, const md_t* k,    \
        const A_type** a, const md_t* rs_a, const md_t* cs_a,                  \
        const AOCL_MEMORY_TAG* mtag_a, const B_type** b, const md_t* rs_b,     \
        const md_t* cs_b, const AOCL_MEMORY_TAG* mtag_b, C_type** c,           \
        const md_t* rs_c, const md_t* cs_c, const C_type alpha,                \
        const C_type beta, dlp_rntm_t* rntm_g, lpgemm_cntx_t* lcntx,           \
        lpgemm_post_op(*post_op_list), DLP_TYPE c_downscale);

GEN_BATCH_LPGEMM_DECORATOR_FN(bfloat16, bfloat16, float, bf16bf16f32of32)
GEN_BATCH_LPGEMM_DECORATOR_FN(float, float, float, f32f32f32of32)
GEN_BATCH_LPGEMM_DECORATOR_FN(uint8_t, int8_t, int32_t, u8s8s32o32)
GEN_BATCH_LPGEMM_DECORATOR_FN(int8_t, int8_t, int32_t, s8s8s32o32)

#define GEN_BATCH_LPGEMM_DECORATOR_FN_MP(A_type, B_type, C_type, LPGEMM_SFX)   \
    void batch_lpgemm_##LPGEMM_SFX##_thread_decorator(                         \
        const md_t group_size, const md_t* m, const md_t* n, const md_t* k,    \
        const A_type** a, const md_t* rs_a, const md_t* cs_a,                  \
        const AOCL_MEMORY_TAG* mtag_a, const B_type** b, const md_t* rs_b,     \
        const md_t* cs_b, const AOCL_MEMORY_TAG* mtag_b, C_type** c,           \
        const md_t* rs_c, const md_t* cs_c, const C_type alpha,                \
        const C_type beta, dlp_rntm_t* rntm_g, lpgemm_cntx_t* lcntx,           \
        lpgemm_pre_op(*pre_op_list), lpgemm_post_op(*post_op_list),            \
        DLP_TYPE c_downscale);

GEN_BATCH_LPGEMM_DECORATOR_FN_MP(bfloat16, int8_t, float, bf16s4f32of32)

#define GEN_LPGEMM_DECORATOR_FN1(A_type, B_type, C_type, LPGEMM_SFX)           \
    void lpgemm_##LPGEMM_SFX##_thread_decorator(                               \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        const AOCL_MEMORY_TAG mtag_b, C_type* c, const md_t rs_c,              \
        const md_t cs_c, const C_type alpha, const C_type beta,                \
        dlp_rntm_t* rntm_g, lpgemm_cntx_t* lcntx, lpgemm_pre_op* pre_op_list,  \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale);

GEN_LPGEMM_DECORATOR_FN1(bfloat16, int8_t, float, bf16s4f32of32)

#define GEN_LPGEMM_OPENMP_DECORATOR_FN2(A_type, B_type, C_type, LPGEMM_SFX)    \
    void lpgemm_##LPGEMM_SFX##_thread_decorator(                               \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        AOCL_MEMORY_TAG mtag_b, float* c, const md_t rs_c, const md_t cs_c,    \
        const C_type alpha, const C_type beta, dlp_rntm_t* rntm_g,             \
        lpgemm_cntx_t* lcntx, lpgemm_group_post_op* grp_post_op_list,          \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale);

GEN_LPGEMM_OPENMP_DECORATOR_FN2(int8_t, int8_t, int32_t, s8s8s32o32_sym_quant)

#define GEN_LPGEMM_DECORATOR_FN3(A_type, B_type, C_type, LPGEMM_SFX)           \
    void lpgemm_##LPGEMM_SFX##_thread_decorator(                               \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        AOCL_MEMORY_TAG mtag_b, C_type* c, const md_t rs_c, const md_t cs_c,   \
        const C_type alpha, const C_type beta, dlp_rntm_t* rntm_g,             \
        lpgemm_cntx_t* lcntx, dlp_quant_op* a_pre_quant,                       \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale);

GEN_LPGEMM_DECORATOR_FN3(bfloat16, int8_t, int32_t, bf16s8s32os32)

#define GEN_UTIL_ELTWISE_OPS_DECORATOR_FN(A_type, B_type, LPGEMM_SFX)          \
    void lpgemm_eltwise_ops_##LPGEMM_SFX##_thread_decorator(                   \
        const md_t m, const md_t n, const A_type* a, const md_t rs_a,          \
        const md_t cs_a, B_type* b, const md_t rs_b, const md_t cs_b,          \
        dlp_rntm_t* rntm_g, lpgemm_eltwise_ops_cntx_t* lcntx,                  \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale);

GEN_UTIL_ELTWISE_OPS_DECORATOR_FN(bfloat16, float, bf16of32)
GEN_UTIL_ELTWISE_OPS_DECORATOR_FN(float, float, f32of32)

#endif

#endif // LPGEMM_THREAD_DECOR_OPENMP_H
