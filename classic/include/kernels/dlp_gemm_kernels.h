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

#ifndef DLP_GEMM_KERN_INTF_H
#define DLP_GEMM_KERN_INTF_H

#include "classic/aocl_bf16_type.h"
#include "classic/aocl_fp16_type.h"
#include "dlp_gemm_post_ops.h"

// Disable DLP_BF16 kernel in cases where compilers support other avx 512
// features except DLP_BF16 ISA.
#if (defined(DLP_GCC)                                                          \
     && ((__GNUC__ < 11) || ((__GNUC__ == 11) && (__GNUC_MINOR__ < 2)))        \
     && defined(DLP_KERNELS_ZEN4))
#define DLP_GEMM_BF16_JIT
#define BPREFETCH_JIT
// #define DUMP_JIT_CODE
#endif

typedef void (*dlp_gemm_m_fringe_f32_ker_ft)(
    const md_t            k0,
    const float*          a,
    const md_t            rs_a,
    const md_t            cs_a,
    const float*          b,
    const md_t            rs_b,
    const md_t            cs_b,
    float*                c,
    const md_t            rs_c,
    const float           alpha,
    const float           beta,
    dlp_gemm_post_op*     post_ops_list,
    dlp_gemm_post_op_attr post_ops_attr);

typedef void (*dlp_gemm_mn_fringe_f32_mask_ker_ft)(
    const md_t            k0,
    const float*          a,
    const md_t            rs_a,
    const md_t            cs_a,
    const float*          b,
    const md_t            rs_b,
    const md_t            cs_b,
    float*                c,
    const md_t            rs_c,
    const float           alpha,
    const float           beta,
    const md_t            n0_rem,
    dlp_gemm_post_op*     post_ops_list,
    dlp_gemm_post_op_attr post_ops_attr);

#define DLP_GEMM_MAIN_KERN(A_type, B_type, C_type, LP_SFX)                     \
    void dlp_gemm_rowvar_##LP_SFX(                                             \
        const md_t m0, const md_t n0, const md_t k0, const A_type* a,          \
        const md_t rs_a, const md_t cs_a, const md_t ps_a, const B_type* b,    \
        const md_t rs_b, const md_t cs_b, C_type* c, const md_t rs_c,          \
        const md_t cs_c, const C_type alpha, const C_type beta,                \
        dlp_gemm_post_op* post_ops_list, dlp_gemm_post_op_attr post_ops_attr)

DLP_GEMM_MAIN_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_6x64);
DLP_GEMM_MAIN_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_6x64);
DLP_GEMM_MAIN_KERN(float, float, float, f32f32f32of32_6x16m);
DLP_GEMM_MAIN_KERN(float, float, float, f32f32f32of32_6x16m_rd);
DLP_GEMM_MAIN_KERN(float, float, float, f32f32f32of32_6x8m_rd);
DLP_GEMM_MAIN_KERN(float, float, float, f32f32f32of32_6x4m_rd);
DLP_GEMM_MAIN_KERN(float, float, float, f32f32f32of32_6x2m_rd);
DLP_GEMM_MAIN_KERN(float, float, float, f32f32f32of32_6x1m_rd);
DLP_GEMM_MAIN_KERN(float, float, float, f32f32f32of32_avx512_256_6x64m);
DLP_GEMM_MAIN_KERN(float, float, float, f32f32f32of32_avx512_6x64m);
DLP_GEMM_MAIN_KERN(float, float, float, f32f32f32of32_avx512_6x64m_rd);
DLP_GEMM_MAIN_KERN(float, float, float, f32f32f32of32_avx512_6x48m_rd);
DLP_GEMM_MAIN_KERN(float, float, float, f32f32f32of32_avx512_6x32m_rd);
DLP_GEMM_MAIN_KERN(int8_t, int8_t, int32_t, s8s8s32os32_6x64);

#define DLP_GEMM_MAIN_KERN1(A_type, B_type, C_type, LP_SFX)                    \
    void dlp_gemm_rowvar_##LP_SFX(                                             \
        const md_t m0, const md_t n0, const md_t k0, const A_type* a,          \
        const md_t rs_a, const md_t cs_a, const md_t ps_a, const B_type* b,    \
        const md_t rs_b, const md_t cs_b, C_type* c, const md_t rs_c,          \
        const md_t cs_c, const C_type alpha, const C_type beta,                \
        dlp_gemm_post_op* post_ops_list, dlp_gemm_post_op_attr post_ops_attr,  \
        dlp_gemm_pre_op_attr pre_ops_attr)

DLP_GEMM_MAIN_KERN1(bfloat16, int8_t, float, bf16s4f32of32_6x64m);
DLP_GEMM_MAIN_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_6x64m);

#define DLP_GEMM_MAIN_KERN2(A_type, B_type, C_type, LP_SFX)                    \
    void dlp_gemm_rowvar_##LP_SFX(                                             \
        const md_t m0, const md_t n0, const md_t k0, const A_type* a,          \
        const md_t rs_a, const md_t cs_a, const md_t ps_a, const B_type* b,    \
        const md_t rs_b, const md_t cs_b, float* c, const md_t rs_c,           \
        const md_t cs_c, const C_type alpha, const C_type beta,                \
        dlp_gemm_grp_post_op_attr grp_post_ops_attr,                           \
        dlp_gemm_post_op* post_ops_list, dlp_gemm_post_op_attr post_ops_attr)

DLP_GEMM_MAIN_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_6x64m_sym_quant);

#define DLP_GEMM_M_RD_FRINGE_KERN(A_type, B_type, C_type, LP_SFX)              \
    void dlp_gemm_rowvar_##LP_SFX(                                             \
        const md_t k0, const A_type* a, const md_t rs_a, const md_t cs_a,      \
        const B_type* b, const md_t rs_b, const md_t cs_b, C_type* c,          \
        const md_t rs_c, const md_t cs_c, const C_type alpha,                  \
        const C_type beta, dlp_gemm_post_op* post_ops_list,                    \
        dlp_gemm_post_op_attr post_ops_attr)

DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_5x64_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_4x64_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_3x64_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_2x64_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_1x64_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_5x48_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_4x48_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_3x48_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_2x48_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_1x48_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_5x32_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_4x32_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_3x32_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_2x32_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_1x32_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_2x16_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_1x16_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_2x8_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_1x8_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_2x4_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_1x4_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_2x2_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_2x1_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_1x2_rd);
DLP_GEMM_M_RD_FRINGE_KERN(float, float, float, f32f32f32of32_1x1_rd);

#define DLP_GEMM_M_FRINGE_KERN(A_type, B_type, C_type, LP_SFX)                 \
    void dlp_gemm_rowvar_##LP_SFX(                                             \
        const md_t k0, const A_type* a, const md_t rs_a, const md_t cs_a,      \
        const B_type* b, const md_t rs_b, const md_t cs_b, C_type* c,          \
        const md_t rs_c, const C_type alpha, const C_type beta,                \
        dlp_gemm_post_op* post_ops_list, dlp_gemm_post_op_attr post_ops_attr)

DLP_GEMM_M_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_5x64);
DLP_GEMM_M_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_4x64);
DLP_GEMM_M_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_3x64);
DLP_GEMM_M_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_2x64);
DLP_GEMM_M_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_1x64);

DLP_GEMM_M_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_5x64);
DLP_GEMM_M_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_4x64);
DLP_GEMM_M_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_3x64);
DLP_GEMM_M_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_2x64);
DLP_GEMM_M_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_1x64);

DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_5x64);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_4x64);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_3x64);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_2x64);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_1x64);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_5x48);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_4x48);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_3x48);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_2x48);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_1x48);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_5x32);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_4x32);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_3x32);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_2x32);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_1x32);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_5x16);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_4x16);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_3x16);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_2x16);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_1x16);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_5x16);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_4x16);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_3x16);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_2x16);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_1x16);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_5x8);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_4x8);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_3x8);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_2x8);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_1x8);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_5x4);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_4x4);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_3x4);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_2x4);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_1x4);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_5x2);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_4x2);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_3x2);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_2x2);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_1x2);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_5x1);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_4x1);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_3x1);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_2x1);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_1x1);

DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_256_5x32);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_256_4x32);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_256_3x32);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_256_2x32);
DLP_GEMM_M_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_256_1x32);

DLP_GEMM_M_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_5x64);
DLP_GEMM_M_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_4x64);
DLP_GEMM_M_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_3x64);
DLP_GEMM_M_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_2x64);
DLP_GEMM_M_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_1x64);

#define DLP_GEMM_M_FRINGE_KERN1(A_type, B_type, C_type, LP_SFX)                \
    void dlp_gemm_rowvar_##LP_SFX(                                             \
        const md_t k0, const A_type* a, const md_t rs_a, const md_t cs_a,      \
        const B_type* b, const md_t rs_b, const md_t cs_b, C_type* c,          \
        const md_t rs_c, const C_type alpha, const C_type beta,                \
        dlp_gemm_post_op* post_ops_list, dlp_gemm_post_op_attr post_ops_attr,  \
        dlp_gemm_pre_op_attr pre_ops_attr)

DLP_GEMM_M_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_5x64);
DLP_GEMM_M_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_5x64);
DLP_GEMM_M_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_4x64);
DLP_GEMM_M_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_4x64);
DLP_GEMM_M_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_3x64);
DLP_GEMM_M_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_3x64);
DLP_GEMM_M_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_2x64);
DLP_GEMM_M_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_2x64);
DLP_GEMM_M_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_1x64);
DLP_GEMM_M_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_1x64);

#define DLP_GEMM_M_FRINGE_KERN2(A_type, B_type, C_type, LP_SFX)                \
    void dlp_gemm_rowvar_##LP_SFX(                                             \
        const md_t k0, const A_type* a, const md_t rs_a, const md_t cs_a,      \
        const B_type* b, const md_t rs_b, const md_t cs_b, float* c,           \
        const md_t rs_c, const C_type alpha, const C_type beta,                \
        dlp_gemm_grp_post_op_attr grp_post_ops_attr,                           \
        dlp_gemm_post_op* post_ops_list, dlp_gemm_post_op_attr post_ops_attr)

DLP_GEMM_M_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_5x64_sym_quant);
DLP_GEMM_M_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_4x64_sym_quant);
DLP_GEMM_M_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_3x64_sym_quant);
DLP_GEMM_M_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_2x64_sym_quant);
DLP_GEMM_M_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_1x64_sym_quant);

#define DLP_GEMM_N_FRINGE_KERN(A_type, B_type, C_type, LP_SFX)                 \
    void dlp_gemm_rowvar_##LP_SFX(                                             \
        const md_t m0, const md_t k0, const A_type* a, const md_t rs_a,        \
        const md_t cs_a, const md_t ps_a, const B_type* b, const md_t rs_b,    \
        const md_t cs_b, C_type* c, const md_t rs_c, const C_type alpha,       \
        const C_type beta, dlp_gemm_post_op* post_ops_list,                    \
        dlp_gemm_post_op_attr post_ops_attr)

DLP_GEMM_N_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_6x16);
DLP_GEMM_N_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_12x16);
DLP_GEMM_N_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_6x32);
DLP_GEMM_N_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_9x32);
DLP_GEMM_N_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_6x48);

DLP_GEMM_N_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_6x16);
DLP_GEMM_N_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_6x32);
DLP_GEMM_N_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_6x48);

DLP_GEMM_N_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_6x48m);
DLP_GEMM_N_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_6x32m);
DLP_GEMM_N_FRINGE_KERN(float, float, float, f32f32f32of32_avx512_6x16m);
DLP_GEMM_N_FRINGE_KERN(float, float, float, f32f32f32of32_6x8m);
DLP_GEMM_N_FRINGE_KERN(float, float, float, f32f32f32of32_6x4m);
DLP_GEMM_N_FRINGE_KERN(float, float, float, f32f32f32of32_6x2m);
DLP_GEMM_N_FRINGE_KERN(float, float, float, f32f32f32of32_6x1m);

DLP_GEMM_N_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_6x16);
DLP_GEMM_N_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_6x32);
DLP_GEMM_N_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_6x48);

#define DLP_GEMM_N_FRINGE_KERN1(A_type, B_type, C_type, LP_SFX)                \
    void dlp_gemm_rowvar_##LP_SFX(                                             \
        const md_t m0, const md_t k0, const A_type* a, const md_t rs_a,        \
        const md_t cs_a, const md_t ps_a, const B_type* b, const md_t rs_b,    \
        const md_t cs_b, C_type* c, const md_t rs_c, const C_type alpha,       \
        const C_type beta, dlp_gemm_post_op* post_ops_list,                    \
        dlp_gemm_post_op_attr post_ops_attr,                                   \
        dlp_gemm_pre_op_attr  pre_ops_attr)

DLP_GEMM_N_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_6x16m);
DLP_GEMM_N_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_6x16m);
DLP_GEMM_N_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_6x32m);
DLP_GEMM_N_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_6x32m);
DLP_GEMM_N_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_6x48m);
DLP_GEMM_N_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_6x48m);

#define DLP_GEMM_N_FRINGE_KERN2(A_type, B_type, C_type, LP_SFX)                \
    void dlp_gemm_rowvar_##LP_SFX(                                             \
        const md_t m0, const md_t k0, const A_type* a, const md_t rs_a,        \
        const md_t cs_a, const md_t ps_a, const B_type* b, const md_t rs_b,    \
        const md_t cs_b, float* c, const md_t rs_c, const C_type alpha,        \
        const C_type beta, dlp_gemm_grp_post_op_attr grp_post_ops_attr,        \
        dlp_gemm_post_op* post_ops_list, dlp_gemm_post_op_attr post_ops_attr)

DLP_GEMM_N_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_6x48_sym_quant);
DLP_GEMM_N_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_6x32_sym_quant);
DLP_GEMM_N_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_6x16_sym_quant);

#define DLP_GEMM_N_LT_NR0_FRINGE_KERN2(A_type, B_type, C_type, LP_SFX)         \
    void dlp_gemm_rowvar_##LP_SFX(                                             \
        const md_t m0, const md_t k0, const A_type* a, const md_t rs_a,        \
        const md_t cs_a, const md_t ps_a, const B_type* b, const md_t rs_b,    \
        const md_t cs_b, float* c, const md_t rs_c, const C_type alpha,        \
        const C_type beta, const md_t n0_rem,                                  \
        dlp_gemm_grp_post_op_attr grp_post_ops_attr,                           \
        dlp_gemm_post_op* post_ops_list, dlp_gemm_post_op_attr post_ops_attr)

DLP_GEMM_N_LT_NR0_FRINGE_KERN2(int8_t,
                               int8_t,
                               int32_t,
                               s8s8s32os32_6xlt16_sym_quant);

#define DLP_GEMM_N_LT_NR0_FRINGE_KERN(A_type, B_type, C_type, LP_SFX)          \
    void dlp_gemm_rowvar_##LP_SFX(                                             \
        const md_t m0, const md_t k0, const A_type* a, const md_t rs_a,        \
        const md_t cs_a, const md_t ps_a, const B_type* b, const md_t rs_b,    \
        const md_t cs_b, C_type* c, const md_t rs_c, const C_type alpha,       \
        const C_type beta, const md_t n0_rem, dlp_gemm_post_op* post_ops_list, \
        dlp_gemm_post_op_attr post_ops_attr)

DLP_GEMM_N_LT_NR0_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_6xlt16);
DLP_GEMM_N_LT_NR0_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_12xlt16);

DLP_GEMM_N_LT_NR0_FRINGE_KERN(bfloat16,
                              bfloat16,
                              float,
                              bf16bf16f32of32_6xlt16);
DLP_GEMM_N_LT_NR0_FRINGE_KERN(float,
                              float,
                              float,
                              f32f32f32of32_avx512_6xlt16m);
DLP_GEMM_N_LT_NR0_FRINGE_KERN(float, float, float, f32f32f32of32_6xlt8m);

DLP_GEMM_N_LT_NR0_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_6xlt16);

#define DLP_GEMM_N_LT_NR0_FRINGE_KERN1(A_type, B_type, C_type, LP_SFX)         \
    void dlp_gemm_rowvar_##LP_SFX(                                             \
        const md_t m0, const md_t k0, const A_type* a, const md_t rs_a,        \
        const md_t cs_a, const md_t ps_a, const B_type* b, const md_t rs_b,    \
        const md_t cs_b, C_type* c, const md_t rs_c, const C_type alpha,       \
        const C_type beta, const md_t n0_rem, dlp_gemm_post_op* post_ops_list, \
        dlp_gemm_post_op_attr post_ops_attr,                                   \
        dlp_gemm_pre_op_attr  pre_ops_attr)

DLP_GEMM_N_LT_NR0_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_6xlt16m);
DLP_GEMM_N_LT_NR0_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_6xlt16m);

#define DLP_GEMM_MN_FRINGE_KERN(A_type, B_type, C_type, LP_SFX)                \
    void dlp_gemm_rowvar_##LP_SFX(                                             \
        const md_t k0, const A_type* a, const md_t rs_a, const md_t cs_a,      \
        const B_type* b, const md_t rs_b, const md_t cs_b, C_type* c,          \
        const md_t rs_c, const C_type alpha, const C_type beta,                \
        dlp_gemm_post_op* post_ops_list, dlp_gemm_post_op_attr post_ops_attr)

DLP_GEMM_MN_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_5x16);
DLP_GEMM_MN_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_4x16);
DLP_GEMM_MN_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_3x16);
DLP_GEMM_MN_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_2x16);
DLP_GEMM_MN_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_1x16);
DLP_GEMM_MN_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_5x32);
DLP_GEMM_MN_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_4x32);
DLP_GEMM_MN_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_3x32);
DLP_GEMM_MN_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_2x32);
DLP_GEMM_MN_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_1x32);
DLP_GEMM_MN_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_5x48);
DLP_GEMM_MN_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_4x48);
DLP_GEMM_MN_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_3x48);
DLP_GEMM_MN_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_2x48);
DLP_GEMM_MN_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_1x48);

DLP_GEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_5x16);
DLP_GEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_4x16);
DLP_GEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_3x16);
DLP_GEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_2x16);
DLP_GEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_1x16);
DLP_GEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_5x32);
DLP_GEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_4x32);
DLP_GEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_3x32);
DLP_GEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_2x32);
DLP_GEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_1x32);
DLP_GEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_5x48);
DLP_GEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_4x48);
DLP_GEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_3x48);
DLP_GEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_2x48);
DLP_GEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_1x48);

DLP_GEMM_MN_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_5x16);
DLP_GEMM_MN_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_4x16);
DLP_GEMM_MN_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_3x16);
DLP_GEMM_MN_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_2x16);
DLP_GEMM_MN_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_1x16);
DLP_GEMM_MN_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_5x32);
DLP_GEMM_MN_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_4x32);
DLP_GEMM_MN_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_3x32);
DLP_GEMM_MN_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_2x32);
DLP_GEMM_MN_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_1x32);
DLP_GEMM_MN_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_5x48);
DLP_GEMM_MN_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_4x48);
DLP_GEMM_MN_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_3x48);
DLP_GEMM_MN_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_2x48);
DLP_GEMM_MN_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_1x48);

#define DLP_GEMM_MN_FRINGE_KERN1(A_type, B_type, C_type, LP_SFX)               \
    void dlp_gemm_rowvar_##LP_SFX(                                             \
        const md_t k0, const A_type* a, const md_t rs_a, const md_t cs_a,      \
        const B_type* b, const md_t rs_b, const md_t cs_b, C_type* c,          \
        const md_t rs_c, const C_type alpha, const C_type beta,                \
        dlp_gemm_post_op* post_ops_list, dlp_gemm_post_op_attr post_ops_attr,  \
        dlp_gemm_pre_op_attr pre_ops_attr)

DLP_GEMM_MN_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_5x16);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_4x16);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_3x16);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_2x16);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_1x16);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_5x32);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_4x32);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_3x32);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_2x32);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_1x32);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_5x48);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_4x48);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_3x48);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_2x48);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_1x48);

DLP_GEMM_MN_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_5x16);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_4x16);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_3x16);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_2x16);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_1x16);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_5x32);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_4x32);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_3x32);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_2x32);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_1x32);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_5x48);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_4x48);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_3x48);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_2x48);
DLP_GEMM_MN_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_1x48);

#define DLP_GEMM_MN_FRINGE_KERN2(A_type, B_type, C_type, LP_SFX)               \
    void dlp_gemm_rowvar_##LP_SFX(                                             \
        const md_t k0, const A_type* a, const md_t rs_a, const md_t cs_a,      \
        const B_type* b, const md_t rs_b, const md_t cs_b, float* c,           \
        const md_t rs_c, const C_type alpha, const C_type beta,                \
        dlp_gemm_grp_post_op_attr grp_post_ops_attr,                           \
        dlp_gemm_post_op* post_ops_list, dlp_gemm_post_op_attr post_ops_attr)

DLP_GEMM_MN_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_5x48_sym_quant);
DLP_GEMM_MN_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_4x48_sym_quant);
DLP_GEMM_MN_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_3x48_sym_quant);
DLP_GEMM_MN_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_2x48_sym_quant);
DLP_GEMM_MN_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_1x48_sym_quant);
DLP_GEMM_MN_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_5x32_sym_quant);
DLP_GEMM_MN_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_4x32_sym_quant);
DLP_GEMM_MN_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_3x32_sym_quant);
DLP_GEMM_MN_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_2x32_sym_quant);
DLP_GEMM_MN_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_1x32_sym_quant);
DLP_GEMM_MN_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_5x16_sym_quant);
DLP_GEMM_MN_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_4x16_sym_quant);
DLP_GEMM_MN_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_3x16_sym_quant);
DLP_GEMM_MN_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_2x16_sym_quant);
DLP_GEMM_MN_FRINGE_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_1x16_sym_quant);

#define DLP_GEMM_MN_LT_NR0_FRINGE_KERN(A_type, B_type, C_type, LP_SFX)         \
    void dlp_gemm_rowvar_##LP_SFX(                                             \
        const md_t k0, const A_type* a, const md_t rs_a, const md_t cs_a,      \
        const B_type* b, const md_t rs_b, const md_t cs_b, C_type* c,          \
        const md_t rs_c, const C_type alpha, const C_type beta,                \
        const md_t n0_rem, dlp_gemm_post_op* post_ops_list,                    \
        dlp_gemm_post_op_attr post_ops_attr)

DLP_GEMM_MN_LT_NR0_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_5xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_4xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_3xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_2xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(uint8_t, int8_t, int32_t, u8s8s32o32_1xlt16);

DLP_GEMM_MN_LT_NR0_FRINGE_KERN(bfloat16,
                               bfloat16,
                               float,
                               bf16bf16f32of32_5xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(bfloat16,
                               bfloat16,
                               float,
                               bf16bf16f32of32_4xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(bfloat16,
                               bfloat16,
                               float,
                               bf16bf16f32of32_3xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(bfloat16,
                               bfloat16,
                               float,
                               bf16bf16f32of32_2xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(bfloat16,
                               bfloat16,
                               float,
                               bf16bf16f32of32_1xlt16);

DLP_GEMM_MN_LT_NR0_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_5xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_4xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_3xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_2xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(int8_t, int8_t, int32_t, s8s8s32os32_1xlt16);

DLP_GEMM_MN_LT_NR0_FRINGE_KERN(float,
                               float,
                               float,
                               f32f32f32of32_avx512_5xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(float,
                               float,
                               float,
                               f32f32f32of32_avx512_4xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(float,
                               float,
                               float,
                               f32f32f32of32_avx512_3xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(float,
                               float,
                               float,
                               f32f32f32of32_avx512_2xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(float,
                               float,
                               float,
                               f32f32f32of32_avx512_1xlt16);

DLP_GEMM_MN_LT_NR0_FRINGE_KERN(float, float, float, f32f32f32of32_5xlt8);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(float, float, float, f32f32f32of32_4xlt8);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(float, float, float, f32f32f32of32_3xlt8);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(float, float, float, f32f32f32of32_2xlt8);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN(float, float, float, f32f32f32of32_1xlt8);

#define DLP_GEMM_MN_LT_NR0_FRINGE_KERN1(A_type, B_type, C_type, LP_SFX)        \
    void dlp_gemm_rowvar_##LP_SFX(                                             \
        const md_t k0, const A_type* a, const md_t rs_a, const md_t cs_a,      \
        const B_type* b, const md_t rs_b, const md_t cs_b, C_type* c,          \
        const md_t rs_c, const C_type alpha, const C_type beta,                \
        const md_t n0_rem, dlp_gemm_post_op* post_ops_list,                    \
        dlp_gemm_post_op_attr post_ops_attr,                                   \
        dlp_gemm_pre_op_attr  pre_ops_attr)

DLP_GEMM_MN_LT_NR0_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_5xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_4xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_3xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_2xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN1(bfloat16, int8_t, float, bf16s4f32of32_1xlt16);

DLP_GEMM_MN_LT_NR0_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_5xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_4xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_3xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_2xlt16);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN1(bfloat16, uint8_t, float, bf16u4f32of32_1xlt16);

#define DLP_GEMM_MN_LT_NR0_FRINGE_KERN2(A_type, B_type, C_type, LP_SFX)        \
    void dlp_gemm_rowvar_##LP_SFX(                                             \
        const md_t k0, const A_type* a, const md_t rs_a, const md_t cs_a,      \
        const B_type* b, const md_t rs_b, const md_t cs_b, float* c,           \
        const md_t rs_c, const C_type alpha, const C_type beta,                \
        const md_t n0_rem, dlp_gemm_grp_post_op_attr grp_post_ops_attr,        \
        dlp_gemm_post_op* post_ops_list, dlp_gemm_post_op_attr post_ops_attr)

DLP_GEMM_MN_LT_NR0_FRINGE_KERN2(int8_t,
                                int8_t,
                                int32_t,
                                s8s8s32os32_5xlt16_sym_quant);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN2(int8_t,
                                int8_t,
                                int32_t,
                                s8s8s32os32_4xlt16_sym_quant);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN2(int8_t,
                                int8_t,
                                int32_t,
                                s8s8s32os32_3xlt16_sym_quant);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN2(int8_t,
                                int8_t,
                                int32_t,
                                s8s8s32os32_2xlt16_sym_quant);
DLP_GEMM_MN_LT_NR0_FRINGE_KERN2(int8_t,
                                int8_t,
                                int32_t,
                                s8s8s32os32_1xlt16_sym_quant);

#define LPGEMV_M_EQ1_KERN(A_type, B_type, C_type, LP_SFX)                      \
    void dlp_gemv_m_one_##LP_SFX(                                              \
        const md_t n0, const md_t k, const A_type* a, const md_t rs_a,         \
        const md_t cs_a, const AOCL_DLP_MEMORY_TAG mtag_a, const B_type* b,    \
        md_t rs_b, const md_t cs_b, const AOCL_DLP_MEMORY_TAG mtag_b,          \
        C_type* c, const md_t rs_c, const md_t cs_c, const C_type alpha,       \
        const C_type beta, md_t NR, const md_t KC, const md_t n_sub_updated,   \
        const md_t jc_cur_loop_rem, dlp_gemm_post_op* post_op,                 \
        dlp_gemm_post_op_attr* post_op_attr)

LPGEMV_M_EQ1_KERN(float, float, float, f32f32f32of32);
LPGEMV_M_EQ1_KERN(float, float, float, f32f32f32of32_avx2);
LPGEMV_M_EQ1_KERN(float, float, float, f32f32f32of32_avx512_256);
LPGEMV_M_EQ1_KERN(bfloat16, bfloat16, float, bf16bf16f32of32);
LPGEMV_M_EQ1_KERN(uint8_t, int8_t, int32_t, u8s8s32os32);
LPGEMV_M_EQ1_KERN(int8_t, int8_t, int32_t, s8s8s32os32);
LPGEMV_M_EQ1_KERN(float16, float16, float16, f16f16f16of16);

#define LPGEMV_M_EQ1_KERN2(A_type, B_type, C_type, LP_SFX)                     \
    void dlp_gemv_m_one_##LP_SFX(                                              \
        const md_t n0, const md_t k, const A_type* a, const md_t rs_a,         \
        const md_t cs_a, const AOCL_DLP_MEMORY_TAG mtag_a, const B_type* b,    \
        md_t rs_b, const md_t cs_b, const AOCL_DLP_MEMORY_TAG mtag_b,          \
        float* c, const md_t rs_c, const md_t cs_c, const C_type alpha,        \
        const C_type beta, md_t NR, const md_t KC, const md_t n_sub_updated,   \
        const md_t                jc_cur_loop_rem,                             \
        dlp_gemm_grp_post_op_attr grp_post_ops_attr,                           \
        dlp_gemm_post_op* post_op, dlp_gemm_post_op_attr* post_op_attr)

LPGEMV_M_EQ1_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_sym_quant);

#define LPGEMV_N_EQ1_KERN(A_type, B_type, C_type, LP_SFX)                      \
    void dlp_gemv_n_one_##LP_SFX(                                              \
        const md_t m0, const md_t k, const A_type* a, const md_t rs_a,         \
        const md_t cs_a, const AOCL_DLP_MEMORY_TAG mtag_a, const B_type* b,    \
        const md_t rs_b, const md_t cs_b, const AOCL_DLP_MEMORY_TAG mtag_b,    \
        C_type* c, const md_t rs_c, const md_t cs_c, const C_type alpha,       \
        const C_type beta, const md_t MR, const md_t KC,                       \
        dlp_gemm_post_op* post_op, dlp_gemm_post_op_attr* post_op_attr)

LPGEMV_N_EQ1_KERN(float, float, float, f32f32f32of32);
LPGEMV_N_EQ1_KERN(float, float, float, f32f32f32of32_avx2);
LPGEMV_N_EQ1_KERN(float, float, float, f32f32f32of32_avx512_256);
LPGEMV_N_EQ1_KERN(bfloat16, bfloat16, float, bf16bf16f32of32);
LPGEMV_N_EQ1_KERN(uint8_t, int8_t, int32_t, u8s8s32os32);
LPGEMV_N_EQ1_KERN(int8_t, int8_t, int32_t, s8s8s32os32);
LPGEMV_N_EQ1_KERN(float16, float16, float16, f16f16f16of16);

#define LPGEMV_N_EQ1_KERN2(A_type, B_type, C_type, LP_SFX)                     \
    void dlp_gemv_n_one_##LP_SFX(                                              \
        const md_t m0, const md_t k, const A_type* a, const md_t rs_a,         \
        const md_t cs_a, const AOCL_DLP_MEMORY_TAG mtag_a, const B_type* b,    \
        const md_t rs_b, const md_t cs_b, const AOCL_DLP_MEMORY_TAG mtag_b,    \
        float* c, const md_t rs_c, const md_t cs_c, const C_type alpha,        \
        const C_type beta, const md_t MR, const md_t KC,                       \
        dlp_gemm_grp_post_op_attr grp_post_ops_attr,                           \
        dlp_gemm_post_op* post_op, dlp_gemm_post_op_attr* post_op_attr)

LPGEMV_N_EQ1_KERN2(int8_t, int8_t, int32_t, s8s8s32os32_sym_quant);

/* ==========================================================================
 * F16F16F16OF16 GEMM kernels removed - JIT handles FP16 GEMM
 * GEMV kernels (dlp_gemv_m_one_f16f16f16of16, dlp_gemv_n_one_f16f16f16of16)
 * are still available and declared above.
 * ========================================================================== */

#endif // DLP_GEMM_KERN_INTF_H
