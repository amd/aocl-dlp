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

#include <stdbool.h>

#include "bindings/c_wrappers/capi_cpu_features.h"
#include "config/dlp_gemm_config.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "tiny/dlp_gemm_f32_tiny_sgemm.h"

static dlp_sgemm_tiny_isa_t dlp_sgemm_tiny_isa_g = { false, false, 1, 0, 0 };

/*
 * AVX-512 when the CPU has it and AOCL_DLP_ENABLE_INSTRUCTIONS has neither
 * pinned the library to an AVX2 (or lower) tier nor asked for AVX-512 YMM
 * kernels. Otherwise the AVX2 kernels, which use YMM registers only. Runs
 * after the enabled arch and instruction preference are known.
 */
void
dlp_sgemm_tiny_init(void)
{
    bool avx512 = false;

    if (dlp_cpuid_is_avx512_supported() != FALSE) {
        const dlp_arch_t arch = dlp_gemm_get_enabled_arch();

        avx512 = (arch != DLP_ARCH_ZEN3) && (arch != DLP_ARCH_ZEN2)
                 && (arch != DLP_ARCH_ZEN) && (arch != DLP_ARCH_GENERIC)
                 && (arch != DLP_ARCH_ERROR)
                 && (dlp_gemm_get_enabled_instr_pref()
                     != DLP_INSTR_PREF_AVX512_YMM_FAVOUR);
    }
    dlp_sgemm_tiny_isa_g.avx512 = avx512;
    /* Same rule as the framework: AVX2 kernels need a configured arch that
     * is not Generic or Error, on a CPU with AVX2 and FMA3. */
    {
        const dlp_arch_t arch = dlp_gemm_get_enabled_arch();
        dlp_sgemm_tiny_isa_g.avx2 =
            avx512
            || ((dlp_cpuid_is_avx2fma3_supported() != FALSE)
                && (arch != DLP_ARCH_GENERIC) && (arch != DLP_ARCH_ERROR));
    }
    dlp_sgemm_tiny_isa_g.wide_rd_rows = avx512 ? 2 : 1;
    dlp_sgemm_tiny_isa_g.l1d_bytes =
        (md_t)dlp_cpuid_get_cache_size(1, DLP_CACHE_TYPE_DATA);
    dlp_sgemm_tiny_isa_g.l2_bytes =
        (md_t)dlp_cpuid_get_cache_size(2, DLP_CACHE_TYPE_UNIFIED);
}

/* C = beta * C; beta == 0 stores zeros so C may hold NaN or Inf on entry. */
static void
dlp_sgemm_tiny_scale_c(
    char order, md_t m, md_t n, float beta, float* c, md_t ldc)
{
    const bool row_major = (order == 'r') || (order == 'R');
    const md_t lines     = row_major ? m : n;
    const md_t len       = row_major ? n : m;

    for (md_t i = 0; i < lines; ++i) {
        float* line = c + i * ldc;
        for (md_t j = 0; j < len; ++j) {
            line[j] = (beta == 0.0f) ? 0.0f : beta * line[j];
        }
    }
}

static dlp_sgemm_tiny_jit_fn
dlp_sgemm_tiny_kernel(bool avx512, dlp_sgemm_tiny_leaf_t leaf)
{
    switch (leaf) {
        case DLP_SGEMM_TINY_RV:
            return avx512 ? dlp_sgemm_tiny_jit_rv_avx512()
                          : dlp_sgemm_tiny_jit_rv_avx2();
        case DLP_SGEMM_TINY_RD:
            return avx512 ? dlp_sgemm_tiny_jit_rd_avx512()
                          : dlp_sgemm_tiny_jit_rd_avx2();
        case DLP_SGEMM_TINY_RD_WIDE:
            return dlp_sgemm_tiny_jit_rd_avx512_wide();
        default:
            return NULL;
    }
}

bool
dlp_sgemm_tiny_f32(char         order,
                   dlp_trans_t  transa,
                   dlp_trans_t  transb,
                   md_t         m,
                   md_t         n,
                   md_t         k,
                   float        alpha,
                   const float* a,
                   md_t         lda,
                   const float* b,
                   md_t         ldb,
                   float        beta,
                   float*       c,
                   md_t         ldc)
{
    const dlp_sgemm_tiny_isa_t* isa = &dlp_sgemm_tiny_isa_g;
    dlp_sgemm_tiny_layout       l;
    const dlp_sgemm_tiny_leaf_t leaf = dlp_sgemm_tiny_route(
        isa, order, dlp_is_trans(transa), dlp_is_trans(transb), m, n, k, alpha,
        beta, lda, ldb, ldc, &l);
    if (leaf == DLP_SGEMM_TINY_NONE) {
        return false;
    }
    if (leaf == DLP_SGEMM_TINY_SCALE_C) {
        dlp_sgemm_tiny_scale_c(order, m, n, beta, c, ldc);
        return true;
    }
    const dlp_sgemm_tiny_jit_fn kernel =
        dlp_sgemm_tiny_kernel(isa->avx512, leaf);
    if (kernel == NULL) {
        return false;
    }

    dlp_sgemm_tiny_jit_args args;
    args.m     = (int64_t)l.m;
    args.n     = (int64_t)l.n;
    args.k     = (int64_t)l.k;
    args.a     = l.swapped ? b : a;
    args.rs_a  = (int64_t)l.rs_a;
    args.cs_a  = (int64_t)l.cs_a;
    args.b     = l.swapped ? a : b;
    args.rs_b  = (int64_t)l.rs_b;
    args.cs_b  = (int64_t)l.cs_b;
    args.c     = c;
    args.rs_c  = (int64_t)l.rs_c;
    args.cs_c  = (int64_t)l.cs_c;
    args.alpha = alpha;
    args.beta  = beta;
    args.nc = (leaf == DLP_SGEMM_TINY_RV) ? dlp_sgemm_tiny_rv_nc(isa, l.k) : 0;
    kernel(&args);
    return true;
}
