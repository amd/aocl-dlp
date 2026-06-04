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

/*
 * Multi-instance u8s8s32 GEMM example demonstrating all output variants:
 *   - aocl_gemm_u8s8s32os32  (int32_t output / accumulator exposed)
 *   - aocl_gemm_u8s8s32os8   (int8_t  downscaled output with saturation)
 *   - aocl_gemm_u8s8s32ou8   (uint8_t downscaled output with saturation)
 *   - aocl_gemm_u8s8s32of32  (float   converted output)
 *   - aocl_gemm_u8s8s32obf16 (bfloat16 converted output)
 *
 * Pattern is similar to multi_instance_gemm_f32.c but we maintain separate
 * output buffers per datatype and share (A, B) across the variants for each
 * instance to keep comparisons easy. Each variant applies the same alpha/beta
 * and optional post-ops metadata (bias + GELU_TANH as illustration) so users
 * can see uniform behavior across the output types.
 */

#define _POSIX_C_SOURCE 199309L
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef DLP_EXAMPLE_ENABLE_OPENMP
#include <omp.h>
#endif

#include "aocl_dlp.h"

// Forward declaration for cleanup helper (defined at end of file)
static void
free_all_instances(int        upto,
                   uint8_t**  A_,
                   int8_t**   B_,
                   int32_t**  C32_,
                   int8_t**   Cs8_,
                   uint8_t**  Cu8_,
                   float**    Cf32_,
                   bfloat16** Cbf16_);

// Utility to initialize unsigned A with a bounded pattern
static void
init_matrix_u8(uint8_t* A, int m, int k, int lda, uint8_t scale)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            A[i * lda + j] = (uint8_t)((i + 3 * j + 17) * scale + (j & 7));
        }
    }
}

// Utility to initialize signed B with small range values
static void
init_matrix_s8(int8_t* B, int k, int n, int ldb, int8_t scale)
{
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            B[i * ldb + j] =
                (int8_t)(((i - 2 * j) * scale + j) & 0x7F) - 64; // centered-ish
        }
    }
}

// Fill int32 C (seed) or zero
static void
init_matrix_s32(int32_t* C, int m, int n, int ldc, int pattern)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * ldc + j] = (pattern) ? (i - j) : 0;
        }
    }
}

// Float output initializer
static void
init_matrix_f32(float* C, int m, int n, int ldc, int pattern)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * ldc + j] = pattern ? (float)((i + j) % 13) : 0.0f;
        }
    }
}

// BF16 helper (store raw 16-bit BF16). We just zero or set small pattern.
static void
init_matrix_bf16(bfloat16* C, int m, int n, int ldc, int pattern)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            uint16_t val =
                (uint16_t)((pattern ? ((i + j) % 11) : 0) << 7); // crude
            C[i * ldc + j] = *(bfloat16*)&val;
        }
    }
}

// Metadata helpers (bias + simple GELU_TANH) similar to f32 example but single
// group.
static bool
init_bias_post_op(dlp_metadata_t* md, md_t n)
{
    md->bias = (dlp_post_op_bias*)malloc(sizeof(dlp_post_op_bias));
    if (!md->bias)
        return false;
    md->bias[0].bias =
        malloc(n * sizeof(int32_t)); // store int32 bias then cast where needed
    if (!md->bias[0].bias)
        return false;
    for (int i = 0; i < n; ++i)
        ((int32_t*)md->bias[0].bias)[i] = i % 23;
    md->bias[0].stor_type = DLP_S32; // raw bias type matches accumulator
    md->bias[0].bias_len  = n;
    md->bias[0].zp        = NULL;
    md->bias[0].sf        = (dlp_sf_t*)calloc(1, sizeof(dlp_sf_t));
    if (!md->bias[0].sf)
        return false;
    md->bias[0].sf->scale_factor = malloc(n * sizeof(float));
    if (!md->bias[0].sf->scale_factor)
        return false;
    for (int i = 0; i < n; ++i)
        ((float*)md->bias[0].sf->scale_factor)[i] = 1.0f;
    md->bias[0].sf->scale_factor_len  = n;
    md->bias[0].sf->scale_factor_type = DLP_F32;
    return true;
}

static bool
init_gelu_post_op(dlp_metadata_t* md, md_t n)
{
    md->eltwise = (dlp_post_op_eltwise*)malloc(sizeof(dlp_post_op_eltwise));
    if (!md->eltwise)
        return false;
    md->eltwise[0].algo.algo_type = GELU_TANH;
    md->eltwise[0].algo.alpha     = NULL;
    md->eltwise[0].algo.beta      = NULL;
    md->eltwise[0].sf             = (dlp_sf_t*)calloc(1, sizeof(dlp_sf_t));
    if (!md->eltwise[0].sf)
        return false;
    md->eltwise[0].sf->scale_factor = malloc(n * sizeof(float));
    if (!md->eltwise[0].sf->scale_factor)
        return false;
    for (int i = 0; i < n; ++i)
        ((float*)md->eltwise[0].sf->scale_factor)[i] = 1.0f;
    md->eltwise[0].sf->scale_factor_len  = n;
    md->eltwise[0].sf->scale_factor_type = DLP_F32;
    return true;
}

static void
init_metadata(dlp_metadata_t* md, md_t n)
{
    memset(md, 0, sizeof(*md));
    if (!init_bias_post_op(md, n))
        return;
    if (!init_gelu_post_op(md, n))
        return;
    md->seq_length  = 2;
    md->num_eltwise = 1;
    md->seq_vector =
        (DLP_POST_OP_TYPE*)malloc(md->seq_length * sizeof(DLP_POST_OP_TYPE));
    if (!md->seq_vector)
        return;
    md->seq_vector[0] = BIAS;
    md->seq_vector[1] = ELTWISE;
}

static void
free_metadata(dlp_metadata_t* md)
{
    if (!md)
        return;
    if (md->bias) {
        if (md->bias[0].bias)
            free(md->bias[0].bias);
        if (md->bias[0].sf) {
            if (md->bias[0].sf->scale_factor)
                free(md->bias[0].sf->scale_factor);
            free(md->bias[0].sf);
        }
        free(md->bias);
    }
    if (md->eltwise) {
        if (md->eltwise[0].sf) {
            if (md->eltwise[0].sf->scale_factor)
                free(md->eltwise[0].sf->scale_factor);
            free(md->eltwise[0].sf);
        }
        free(md->eltwise);
    }
    if (md->seq_vector)
        free(md->seq_vector);
}

static void
usage(const char* prog)
{
    printf("Usage: %s [-i num_instances] [-n repeats]\n", prog);
}

int
main(int argc, char** argv)
{
    int num_instances = 32; // fewer default than f32 to keep memory moderate
    int n_repeats     = 1;

    for (int arg = 1; arg < argc; ++arg) {
        if (!strcmp(argv[arg], "-i") && arg + 1 < argc) {
            num_instances = atoi(argv[++arg]);
        } else if (!strcmp(argv[arg], "-n") && arg + 1 < argc) {
            n_repeats = atoi(argv[++arg]);
        } else if (!strcmp(argv[arg], "-h")) {
            usage(argv[0]);
            return 0;
        }
    }
    if (num_instances <= 0)
        num_instances = 1;
    if (n_repeats <= 0)
        n_repeats = 1;

    srand(0xC0FFEE);

    uint8_t**  A     = (uint8_t**)calloc(num_instances, sizeof(uint8_t*));
    int8_t**   B     = (int8_t**)calloc(num_instances, sizeof(int8_t*));
    int32_t**  C32   = (int32_t**)calloc(num_instances, sizeof(int32_t*));
    int8_t**   Cs8   = (int8_t**)calloc(num_instances, sizeof(int8_t*));
    uint8_t**  Cu8   = (uint8_t**)calloc(num_instances, sizeof(uint8_t*));
    float**    Cf32  = (float**)calloc(num_instances, sizeof(float*));
    bfloat16** Cbf16 = (bfloat16**)calloc(num_instances, sizeof(bfloat16*));

    if (!A || !B || !C32 || !Cs8 || !Cu8 || !Cf32 || !Cbf16) {
        fprintf(stderr, "Allocation failure (pointer arrays)\n");
        free(A);
        free(B);
        free(C32);
        free(Cs8);
        free(Cu8);
        free(Cf32);
        free(Cbf16);
        return 1;
    }

    // Dimension sweep similar style (varied) but modest.
    for (int i = 0; i < num_instances; ++i) {
        md_t m = 64
                 + (i * (256 - 64))
                       / ((num_instances > 1) ? (num_instances - 1) : 1);
        md_t n = 64
                 + (i * (512 - 64))
                       / ((num_instances > 1) ? (num_instances - 1) : 1);
        md_t k = 64
                 + (i * (320 - 64))
                       / ((num_instances > 1) ? (num_instances - 1) : 1);
        md_t lda = k;
        md_t ldb = n;
        md_t ldc = n;

        size_t sizeA = (size_t)lda * m;
        size_t sizeB = (size_t)ldb * k;
        size_t sizeC = (size_t)ldc * m;

        A[i]     = (uint8_t*)malloc(sizeA);
        B[i]     = (int8_t*)malloc(sizeB);
        C32[i]   = (int32_t*)malloc(sizeC * sizeof(int32_t));
        Cs8[i]   = (int8_t*)malloc(sizeC);
        Cu8[i]   = (uint8_t*)malloc(sizeC);
        Cf32[i]  = (float*)malloc(sizeC * sizeof(float));
        Cbf16[i] = (bfloat16*)malloc(sizeC * sizeof(bfloat16));
        if (!A[i] || !B[i] || !C32[i] || !Cs8[i] || !Cu8[i] || !Cf32[i]
            || !Cbf16[i]) {
            fprintf(stderr, "Per-instance alloc failed at instance %d\n", i);
            free_all_instances(i + 1, A, B, C32, Cs8, Cu8, Cf32, Cbf16);
            return 1;
        }

        init_matrix_u8(A[i], m, k, lda, 1 + (i % 5));
        init_matrix_s8(B[i], k, n, ldb, 1 + (i % 3));
        init_matrix_s32(C32[i], m, n, ldc, 0);
        memset(Cs8[i], 0, sizeC);
        memset(Cu8[i], 0, sizeC);
        init_matrix_f32(Cf32[i], m, n, ldc, 0);
        init_matrix_bf16(Cbf16[i], m, n, ldc, 0);

        // Post-ops disabled: init_metadata(&meta[i], n);
    }

#ifdef DLP_EXAMPLE_ENABLE_OPENMP
    omp_set_max_active_levels(2);
#endif

    double          start_time, end_time;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    start_time = ts.tv_sec + ts.tv_nsec / 1e9;

    const char order  = 'R';
    const char transa = 'N';
    const char transb = 'N';

    for (int rep = 0; rep < n_repeats; ++rep) {
#ifdef DLP_EXAMPLE_ENABLE_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < num_instances; ++i) {
            dlp_thread_set_num_threads(1);
            md_t m = 64
                     + (i * (256 - 64))
                           / ((num_instances > 1) ? (num_instances - 1) : 1);
            md_t n = 64
                     + (i * (512 - 64))
                           / ((num_instances > 1) ? (num_instances - 1) : 1);
            md_t k = 64
                     + (i * (320 - 64))
                           / ((num_instances > 1) ? (num_instances - 1) : 1);
            md_t    lda = k, ldb = n, ldc = n;
            int32_t alpha = (i % 2) ? 2 : 1;
            int32_t beta =
                (i % 3) ? 1 : 0; // show both accumulation and overwrite cases
            const char mem_format_a = 'N';
            const char mem_format_b = (i % 2) ? 'N' : 'R';

            // Each output variant invoked independently using same inputs.
            aocl_gemm_u8s8s32os32(order, transa, transb, m, n, k, alpha, A[i],
                                  lda, mem_format_a, B[i], ldb, mem_format_b,
                                  beta, C32[i], ldc, NULL);
            aocl_gemm_u8s8s32os8(order, transa, transb, m, n, k, alpha, A[i],
                                 lda, mem_format_a, B[i], ldb, mem_format_b,
                                 beta, Cs8[i], ldc, NULL);
            aocl_gemm_u8s8s32ou8(order, transa, transb, m, n, k, alpha, A[i],
                                 lda, mem_format_a, B[i], ldb, mem_format_b,
                                 beta, Cu8[i], ldc, NULL);
            aocl_gemm_u8s8s32of32(order, transa, transb, m, n, k, alpha, A[i],
                                  lda, mem_format_a, B[i], ldb, mem_format_b,
                                  beta, Cf32[i], ldc, NULL);
            aocl_gemm_u8s8s32obf16(order, transa, transb, m, n, k, alpha, A[i],
                                   lda, mem_format_a, B[i], ldb, mem_format_b,
                                   beta, Cbf16[i], ldc, NULL);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &ts);
    end_time = ts.tv_sec + ts.tv_nsec / 1e9;
    printf("Completed %d repeats of %d u8s8s32 multi-output instances in %f s "
           "(avg %f s/repeat)\n",
           n_repeats, num_instances, end_time - start_time,
           (end_time - start_time) / n_repeats);

    // Basic spot check: print a few elements from first instance across
    // variants.
    if (num_instances > 0) {
        printf("Sample comparison (instance 0) CS32[0]=%d Cs8[0]=%d Cu8[0]=%u "
               "Cf32[0]=%.1f\n",
               C32[0][0], (int)Cs8[0][0], (unsigned)Cu8[0][0], Cf32[0][0]);
    }

    free_all_instances(num_instances, A, B, C32, Cs8, Cu8, Cf32, Cbf16);
    // metadata disabled -> no free(meta)
    return 0;
}

// Cleanup helper implementation
static void
free_all_instances(int        upto,
                   uint8_t**  A_,
                   int8_t**   B_,
                   int32_t**  C32_,
                   int8_t**   Cs8_,
                   uint8_t**  Cu8_,
                   float**    Cf32_,
                   bfloat16** Cbf16_)
{
    if (!A_)
        return; // if first failed all others likely NULL
    for (int i = 0; i < upto; ++i) {
        if (A_)
            free(A_[i]);
        if (B_)
            free(B_[i]);
        if (C32_)
            free(C32_[i]);
        if (Cs8_)
            free(Cs8_[i]);
        if (Cu8_)
            free(Cu8_[i]);
        if (Cf32_)
            free(Cf32_[i]);
        if (Cbf16_)
            free(Cbf16_[i]);
    }
    free(A_);
    free(B_);
    free(C32_);
    free(Cs8_);
    free(Cu8_);
    free(Cf32_);
    free(Cbf16_);
}
