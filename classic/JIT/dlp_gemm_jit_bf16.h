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

#ifndef JIT_BF16_H
#define JIT_BF16_H

#include "xbyak/xbyak.h"
#include <cstring>
#include <functional>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "classic/dlp_base_types.h"

using namespace Xbyak;

class dlp_gemm_jit : public Xbyak::CodeGenerator
{

  private:
    void preamble();
    void postamble();
    void initialize_params(dlp_gemm_jit_inputs_t* params);
    void reg_init(md_t m_dim, md_t n_dim);
    void kernel_unroll(md_t m_dim, md_t n_dim);
    void prefetchC(md_t m_dim, md_t n_dim);
    void k_fringe_loop(md_t m_dim, md_t n_dim);
    void scale_alpha(md_t m_dim, md_t n_dim);
    // beta ops
    void bf16_f32_beta_op(md_t m_dim, md_t n_dim);
    void f32_f32_beta_op(md_t m_dim, md_t n_dim);
    // postops
    void clip_f32(md_t m_dim, md_t n_dim);
    void f32_f32_matrix_add(md_t m_dim, md_t n_dim);
    void bf16_f32_matrix_add(md_t m_dim, md_t n_dim);
    void bias_row_major(md_t m_dim, md_t n_dim);
    void bias_col_major(md_t m_dim, md_t n_dim);
    void relu(md_t m_dim, md_t n_dim);
    void relu_scale(md_t m_dim, md_t n_dim);
    void gelu_tanh(md_t m_dim, md_t n_dim);
    void POLY_EVAL_6_AVX512();
    void EXPF_AVX512();
    void TANHF_AVX512();
    void GELU_TANH_F32_AVX512_DEF(md_t reg);
    void TANHF_AVX512_DEF(md_t reg);
    void POLY_EVAL_HORNER_16_0_AVX512();
    void ERF_AVX512();
    void GELU_ERF_F32_AVX512_DEF(md_t reg);
    void gelu_erf(md_t m_dim, md_t n_dim);
    void SWISH_F32_AVX512_DEF(md_t reg);
    void swish(md_t m, md_t n);
    void downscale_row_major(md_t m_dim, md_t n_dim);
    void downscale_col_major(md_t m_dim, md_t n_dim);
    void tanh(md_t m_dim, md_t n_dim);
    void SIGMOID_AVX512_DEF(md_t reg);
    void sigmoid(md_t m_dim, md_t n_dim);

    void apply_post_ops_in_high_reg_pressure(const md_t num_post_op_regs,
                                             std::function<void(md_t)> op_fn);
    // C store functions
    void cvt_store_f32_bf16_mask(md_t m_dim, md_t n_dim);
    void store_f32(md_t m_dim, md_t n_dim);

    void post_op_label_lastk_safe_jump_with_next_ptr();
    void post_op_label_lastk_safe_jump();

    md_t num_elems_per_reg = 64 / sizeof(float);
    md_t n_rem;
    md_t num_fma_regs;
    md_t fma_start_idx  = 0;
    md_t load_start_idx = 0;
    md_t num_full_loads;
    md_t num_loads;
    md_t bcst_start_idx;
    md_t alpha_reg = fma_start_idx;
    md_t beta_reg;

    // registers used for gelu_tanh
    const md_t num_gelu_regs = 9;
    const md_t const1        = load_start_idx;
    const md_t const2        = load_start_idx + 1;
    const md_t x             = load_start_idx + 2;
    const md_t r             = load_start_idx + 3;
    const md_t r2            = load_start_idx + 4;
    const md_t z             = load_start_idx + 5;
    const md_t dn            = load_start_idx + 6;
    const md_t x_tanh        = load_start_idx + 7;
    const md_t q             = load_start_idx + 8;

    // registers for gelu_erf
    const md_t num_erf_regs = 5;
    const md_t x_erf        = load_start_idx + 4;

    // registers used for swish. Reusing the gelu_tanh registers.
    const md_t num_swish_regs = 9;

    const md_t stack_off_ps_a                   = 8;
    const md_t stack_off_k_iter_before_prefetch = 16;
    const md_t stack_off_k_iter_after_prefetch  = 24;
    const md_t stack_off_k_left                 = 32;
    const md_t stack_off_alpha                  = 40;
    const md_t stack_off_beta                   = 48;
    const md_t stack_off_b_ptr                  = 56;
    const md_t stack_off_postop                 = 64;
#ifdef BPREFETCH_JIT
    const md_t stack_off_bprefetch_dist = 72;
#endif
    const md_t stack_off_buf_downscale =
        stack_off_postop + offsetof(dlp_gemm_post_op_attr, buf_downscale);
    const md_t stack_off_temp_list =
        stack_off_postop + sizeof(dlp_gemm_post_op);

    const md_t stack_off_zmm_stack = stack_off_temp_list + 8;
    md_t       zmm_stack_top;

    void store_zmms_in_stack(md_t reg_start_idx, md_t num_regs, md_t stack_off);

    void get_zmms_from_stack(md_t reg_start_idx, md_t num_regs, md_t stack_off);

    float gelu_consts[7] = { 0.044715, 0.797884, -2, 0.5, -1, 2, 1 };
    float gelu_macros[6] = { 1.4426950408889634, 1.2582912E7, -88.0f, 88.0f,
                             (float)(1.0 / 0.0), -2147483648 };

    float dlp_gemm_exp[6] = { 1.0000000754895704,   0.6931472254087585,
                              0.2402210737432219,   0.05550297297702539,
                              0.009676036358193323, 0.001341000536524434 };

    float erf_consts[4] = { 0.707107, 1.0, 0.5, 3.553f };

    float dlp_gemm_erf[16] = { 1.1283793786592402,    2.5468861568875563E-5,
                               0.3756169877289898,    0.004025179163741976,
                               0.12947984300439994,   0.0412525204794885,
                               0.03918550001070417,   0.07104542913277255,
                               0.05717052146749476,   0.025310822854733135,
                               0.0067305713376882076, 0.0010410692067591445,
                               6.921588102382636E-5,  4.092409485758739E-6,
                               1.033131746125426E-6,  5.2927177513236435E-8 };

    const md_t gelu_consts_off  = 0;
    const md_t gelu_macros_off  = gelu_consts_off + sizeof(gelu_consts);
    const md_t dlp_gemm_exp_off = gelu_macros_off + sizeof(gelu_macros);
    const md_t erf_consts_off   = dlp_gemm_exp_off + sizeof(dlp_gemm_exp);
    const md_t dlp_gemm_erf_off = erf_consts_off + sizeof(erf_consts);

    Xbyak::Address get_constant(md_t table_off, md_t value_off)
    {
        return ptr[rip + tables + table_off + value_off * 4];
    }
    Xbyak::Label tables;

  public:
    dlp_gemm_jit(void* buffer, size_t bufferSize);
    void generate_kernel(dlp_gemm_jit_inputs_t* params);
    const void (*get_function() const)(dlp_gemm_jit_params_t*,
                                       dlp_gemm_post_op_attr*,
                                       dlp_gemm_post_op*);
    const void* get_code() const;
    md_t        get_size();
};
#endif
