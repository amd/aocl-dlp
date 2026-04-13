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
 * Simple example demonstrating the basic usage of AOCL-DLP's f32 GEMM operation
 *
 * This example shows how to:
 * 1. Initialize input matrices
 * 2. Perform a simple matrix multiplication C = alpha * A * B + beta * C
 * 3. Clean up resources
 */

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef DLP_EXAMPLE_ENABLE_OPENMP
#include <omp.h>
#endif

#include "aocl_dlp.h"

// Utility function to initialize a matrix with values
void
init_matrix(float* matrix, int rows, int cols, int ld, float value)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * ld + j] = value * (i + j + 1) / (rows * cols);
        }
    }
}

typedef struct
{
    char             order;
    md_t*            m;
    md_t*            n;
    md_t*            k;
    md_t*            lda;
    md_t*            ldb;
    md_t*            ldc;
    float**          a;
    float**          b;
    float**          c;
    float*           alpha;
    float*           beta;
    char*            transa;
    char*            transb;
    char*            mem_format_a;
    char*            mem_format_b;
    dlp_metadata_t** mData;
    int              repeat_post_ops;
} multi_instance_gemm_f32_args_t;

bool
allocate_memory(multi_instance_gemm_f32_args_t* args, int num_instances)
{
    // Allocate memory for all attributes
    args->m            = malloc(num_instances * sizeof(md_t));
    args->n            = malloc(num_instances * sizeof(md_t));
    args->k            = malloc(num_instances * sizeof(md_t));
    args->lda          = malloc(num_instances * sizeof(md_t));
    args->ldb          = malloc(num_instances * sizeof(md_t));
    args->ldc          = malloc(num_instances * sizeof(md_t));
    args->a            = malloc(num_instances * sizeof(float*));
    args->b            = malloc(num_instances * sizeof(float*));
    args->c            = malloc(num_instances * sizeof(float*));
    args->alpha        = malloc(num_instances * sizeof(float));
    args->beta         = malloc(num_instances * sizeof(float));
    args->transa       = malloc(num_instances * sizeof(char));
    args->transb       = malloc(num_instances * sizeof(char));
    args->mem_format_a = malloc(num_instances * sizeof(char));
    args->mem_format_b = malloc(num_instances * sizeof(char));
    args->mData        = malloc(num_instances * sizeof(dlp_metadata_t*));

    if (!args->m || !args->n || !args->k || !args->lda || !args->ldb
        || !args->ldc || !args->a || !args->b || !args->c || !args->alpha
        || !args->beta || !args->transa || !args->transb || !args->mem_format_a
        || !args->mem_format_b || !args->mData) {
        return false;
    }

    // Initialize all pointers to NULL
    for (int i = 0; i < num_instances; i++) {
        args->a[i]     = NULL;
        args->b[i]     = NULL;
        args->c[i]     = NULL;
        args->mData[i] = NULL;
    }

    return true;
}

void
cleanupBiasPostOp(dlp_metadata_t* mData, int repeat_post_ops)
{
    if (mData->bias) {
        for (int i = 0; i < repeat_post_ops; i++) {
            free((mData->bias[i]).bias);
            if ((mData->bias[i]).sf) {
                free(((mData->bias[i]).sf)->scale_factor);
            }
            free((mData->bias[i]).sf);
        }
        free(mData->bias);
    }
}

void
cleanupEltwisePostOp(dlp_metadata_t* mData)
{
    if (mData->eltwise) {
        for (int i = 0; i < mData->num_eltwise; i++) {
            // Free alpha and beta if they were allocated
            free((mData->eltwise[i]).algo.alpha);
            free((mData->eltwise[i]).algo.beta);
            if ((mData->eltwise[i]).sf) {
                free(((mData->eltwise[i]).sf)->scale_factor);
            }
            free((mData->eltwise[i]).sf);
        }
    }

    free(mData->eltwise);
}

void
cleanupMatAddPostOp(dlp_metadata_t* mData, int repeat_post_ops)
{
    if (mData->matrix_add) {
        for (int i = 0; i < repeat_post_ops; i++) {
            free((mData->matrix_add[i]).matrix);
            if ((mData->matrix_add[i]).sf) {
                free(((mData->matrix_add[i]).sf)->scale_factor);
            }
            free((mData->matrix_add[i]).sf);
        }
        free(mData->matrix_add);
    }
}

void
cleanupPostOps(dlp_metadata_t* mData, int repeat_post_ops)
{
    cleanupBiasPostOp(mData, repeat_post_ops);
    cleanupEltwisePostOp(mData);
    cleanupMatAddPostOp(mData, repeat_post_ops);
    free(mData->seq_vector);
}

void
free_memory(multi_instance_gemm_f32_args_t* args, int num_instances)
{
    // Free allocated memory
    free(args->m);
    free(args->n);
    free(args->k);
    free(args->lda);
    free(args->ldb);
    free(args->ldc);
    if (args->a) {
        for (int i = 0; i < num_instances; i++) {
            free(args->a[i]);
        }
    }
    if (args->b) {
        for (int i = 0; i < num_instances; i++) {
            free(args->b[i]);
        }
    }
    if (args->c) {
        for (int i = 0; i < num_instances; i++) {
            free(args->c[i]);
        }
    }
    free(args->a);
    free(args->b);
    free(args->c);
    free(args->alpha);
    free(args->beta);
    free(args->transa);
    free(args->transb);
    free(args->mem_format_a);
    free(args->mem_format_b);
    if (args->mData) {
        for (int i = 0; i < num_instances; i++) {
            if (args->mData[i]) {
                cleanupPostOps(args->mData[i], args->repeat_post_ops);
                free(args->mData[i]);
            }
        }
    }
    free(args->mData);
}

bool
initSingleBiasPostOp(dlp_metadata_t* mData, md_t m, md_t n, int index)
{
    (mData->bias[index]).bias = (float*)malloc(n * sizeof(float));
    if (!(mData->bias[index]).bias) {
        return false;
    }
    for (int i = 0; i < n; i++) {
        ((float*)(mData->bias[index]).bias)[i] =
            (float)i; // Initialize bias values
    }
    (mData->bias[index]).stor_type = DLP_F32;
    (mData->bias[index]).bias_len  = n;

    // Bias scale factor.
    (mData->bias[index]).sf = NULL;
    (mData->bias[index]).sf = (dlp_sf_t*)malloc(sizeof(dlp_sf_t));
    if (!(mData->bias[index]).sf) {
        return false;
    }
    ((mData->bias[index]).sf)->scale_factor = NULL;
    ((mData->bias[index]).sf)->scale_factor = (void*)malloc(n * sizeof(float));
    if (!((mData->bias[index]).sf)->scale_factor) {
        return false;
    }
    for (int i = 0; i < n; i++) {
        ((float*)((mData->bias[index]).sf)->scale_factor)[i] =
            1.5f; // Initialize scale factors
    }
    ((mData->bias[index]).sf)->scale_factor_len  = n;
    ((mData->bias[index]).sf)->scale_factor_type = DLP_F32;

    return true;
}

bool
initMultipleBiasPostOp(dlp_metadata_t* mData,
                       md_t            m,
                       md_t            n,
                       int             repeat_post_ops)
{
    mData->bias = NULL;
    mData->bias =
        (dlp_post_op_bias*)calloc(repeat_post_ops, sizeof(dlp_post_op_bias));
    if (!mData->bias) {
        return false;
    }
    bool ret = true;
    for (int i = 0; i < repeat_post_ops; ++i) {
        ret = initSingleBiasPostOp(mData, m, n, i);
        if (!ret) {
            break;
        }
    }
    return ret;
}

bool
initSingleGeluTanhPostOp(dlp_metadata_t* mData, md_t m, md_t n, int index)
{
    (mData->eltwise[index]).algo.algo_type = GELU_TANH;
    (mData->eltwise[index]).algo.alpha     = NULL;
    (mData->eltwise[index]).algo.beta      = NULL;

    // Eltwise scale factor.
    (mData->eltwise[index]).sf = NULL;
    (mData->eltwise[index]).sf = (dlp_sf_t*)malloc(sizeof(dlp_sf_t));
    if (!(mData->eltwise[index]).sf) {
        return false;
    }
    ((mData->eltwise[index]).sf)->scale_factor = NULL;
    ((mData->eltwise[index]).sf)->scale_factor =
        (void*)malloc(n * sizeof(float));
    if (!((mData->eltwise[index]).sf)->scale_factor) {
        return false;
    }
    for (int i = 0; i < n; i++) {
        ((float*)((mData->eltwise[index]).sf)->scale_factor)[i] =
            1.5f; // Initialize scale factors
    }
    ((mData->eltwise[index]).sf)->scale_factor_len  = n;
    ((mData->eltwise[index]).sf)->scale_factor_type = DLP_F32;

    return true;
}

bool
initMultipleGeluTanhPostOp(dlp_metadata_t* mData,
                           md_t            m,
                           md_t            n,
                           int             repeat_post_ops)
{
    mData->eltwise = NULL;
    mData->eltwise = (dlp_post_op_eltwise*)calloc(repeat_post_ops,
                                                  sizeof(dlp_post_op_eltwise));
    if (!mData->eltwise) {
        return false;
    }
    bool ret = true;
    for (int i = 0; i < repeat_post_ops; ++i) {
        ret = initSingleGeluTanhPostOp(mData, m, n, i);
        if (!ret) {
            break;
        }
    }
    return ret;
}

bool
initSingleMatAddPostOp(dlp_metadata_t* mData, md_t m, md_t n, int index)
{
    (mData->matrix_add[index]).matrix = NULL;
    (mData->matrix_add[index]).matrix = (float*)malloc(m * n * sizeof(float));
    if (!(mData->matrix_add[index]).matrix) {
        return false;
    }
    for (int i = 0; i < (m * n); i++) {
        ((float*)(mData->matrix_add[index]).matrix)[i] =
            (float)i; // Initialize matrix values
    }
    (mData->matrix_add[index]).ldm       = n;
    (mData->matrix_add[index]).stor_type = DLP_F32;

    // Matrix scale factor.
    (mData->matrix_add[index]).sf = NULL;
    (mData->matrix_add[index]).sf = (dlp_sf_t*)malloc(sizeof(dlp_sf_t));
    if (!(mData->matrix_add[index]).sf) {
        return false;
    }
    ((mData->matrix_add[index]).sf)->scale_factor = NULL;
    ((mData->matrix_add[index]).sf)->scale_factor =
        (void*)malloc(n * sizeof(float));
    if (!((mData->matrix_add[index]).sf)->scale_factor) {
        return false;
    }
    for (int i = 0; i < n; i++) {
        ((float*)((mData->matrix_add[index]).sf)->scale_factor)[i] =
            1.5f; // Initialize scale factors
    }
    ((mData->matrix_add[index]).sf)->scale_factor_len  = n;
    ((mData->matrix_add[index]).sf)->scale_factor_type = DLP_F32;

    return true;
}

bool
initMultipleMatAddPostOp(dlp_metadata_t* mData,
                         md_t            m,
                         md_t            n,
                         int             repeat_post_ops)
{
    mData->matrix_add = NULL;
    mData->matrix_add = (dlp_post_op_matrix_add*)calloc(
        repeat_post_ops, sizeof(dlp_post_op_matrix_add));
    if (!mData->matrix_add) {
        return false;
    }
    bool ret = true;
    for (int i = 0; i < repeat_post_ops; ++i) {
        ret = initSingleMatAddPostOp(mData, m, n, i);
        if (!ret) {
            break;
        }
    }
    return ret;
}

bool
initPostOps(dlp_metadata_t* mData, md_t m, md_t n, int repeat_post_ops)
{
    if (!initMultipleBiasPostOp(mData, m, n, repeat_post_ops)) {
        return false;
    }
    if (!initMultipleGeluTanhPostOp(mData, m, n, repeat_post_ops)) {
        return false;
    }
    if (!initMultipleMatAddPostOp(mData, m, n, repeat_post_ops)) {
        return false;
    }

    mData->post_op_grp = NULL;
    mData->pre_ops     = NULL;

    mData->seq_length  = 6; // Number of post-operations
    mData->num_eltwise = repeat_post_ops;
    mData->seq_vector  = NULL;
    mData->seq_vector =
        (DLP_POST_OP_TYPE*)calloc(mData->seq_length, sizeof(DLP_POST_OP_TYPE));
    if (!mData->seq_vector) {
        return false;
    }
    mData->seq_vector[0] = BIAS;
    mData->seq_vector[1] = BIAS;
    mData->seq_vector[2] = ELTWISE;
    mData->seq_vector[3] = ELTWISE;
    mData->seq_vector[4] = MATRIX_ADD;
    mData->seq_vector[5] = MATRIX_ADD;

    return true;
}

bool
initialize_memory(multi_instance_gemm_f32_args_t* args, int num_instances)
{
    // Initialize dimensions in specified ranges
    for (int i = 0; i < num_instances; i++) {
        args->m[i] = 244
                     + (i * (1024 - 244))
                           / ((num_instances > 1) ? (num_instances - 1)
                                                  : 1); // Range: 244 to 1024
        args->n[i] = 128
                     + (i * (4096 - 128))
                           / ((num_instances > 1) ? (num_instances - 1)
                                                  : 1); // Range: 128 to 4096
        args->k[i] = 200
                     + (i * (445 - 200))
                           / ((num_instances > 1) ? (num_instances - 1)
                                                  : 1); // Range: 200 to 445
        args->lda[i] = args->k[i];
        args->ldb[i] = args->n[i];
        args->ldc[i] = args->n[i];

        // Allocate individual matrices
        args->a[i] = (float*)malloc(args->lda[i] * args->m[i] * sizeof(float));
        args->b[i] = (float*)malloc(args->ldb[i] * args->k[i] * sizeof(float));
        args->c[i] = (float*)malloc(args->ldc[i] * args->m[i] * sizeof(float));

        if (!args->a[i] || !args->b[i] || !args->c[i]) {
            return false;
        }

        // Initialize matrices with some values
        init_matrix(args->a[i], args->m[i], args->k[i], args->lda[i], 1.0f);
        init_matrix(args->b[i], args->k[i], args->n[i], args->ldb[i], 0.5f);
        init_matrix(args->c[i], args->m[i], args->n[i], args->ldc[i],
                    0.0f); // Initialize C with zeros

        args->mData[i] = (dlp_metadata_t*)calloc(1, sizeof(dlp_metadata_t));
        if (!args->mData[i]) {
            return false;
        }
        args->repeat_post_ops = 2;
        if (!initPostOps(args->mData[i], args->m[i], args->n[i],
                         args->repeat_post_ops)) {
            return false;
        }

        args->alpha[i]        = (rand() % 2 == 0) ? 1.0f : 2.0f;
        args->beta[i]         = (rand() % 2 == 0) ? 1.0f : 9.0f;
        args->transa[i]       = 'N'; // No transpose for A
        args->transb[i]       = 'N'; // No transpose for B
        args->mem_format_a[i] = 'N'; // A is not reordered
        args->mem_format_b[i] = 'N'; // B is not reordered
    }

    return true;
}

// Only supporting row major for this example.
#define ZERO_INIT_AND_DEFINE_ARGS(var_name)                                    \
    multi_instance_gemm_f32_args_t var_name = { 'R',  NULL, NULL, NULL, NULL,  \
                                                NULL, NULL, NULL, NULL, NULL,  \
                                                NULL, NULL, NULL, NULL, NULL,  \
                                                NULL, NULL, 0 };

void
parse_cmd_line_args(int   argc,
                    char* argv[],
                    int*  num_instances,
                    int*  n_repeats,
                    bool* randomize_thread_per_instance)
{
    // Parse command line arguments for number of instances
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && ((i + 1) < argc)) {
            *num_instances = atoi(argv[i + 1]);
            if (*num_instances <= 0) {
                fprintf(stderr,
                        "Invalid number of instances: %d."
                        " Using default value of 64.\n",
                        *num_instances);
                *num_instances = 64;
            }
        } else if (strcmp(argv[i], "-n") == 0 && ((i + 1) < argc)) {
            *n_repeats = atoi(argv[i + 1]);
            if (*n_repeats <= 0) {
                fprintf(stderr,
                        "Invalid number of repeats: %d."
                        " Using default value of 1.\n",
                        *n_repeats);
                *n_repeats = 1;
            }
        } else if (strcmp(argv[i], "-r") == 0) {
            // This option is only to enable or disable randomization
            // of thread assignment per instance.
            *randomize_thread_per_instance = true;
        }
    }
}

int
main(int argc, char* argv[])
{
    int  num_instances                 = 64; // Number of instances.
    int  n_repeats                     = 1;  // Configure number of repetitions
    bool randomize_thread_per_instance = false;

    parse_cmd_line_args(argc, argv, &num_instances, &n_repeats,
                        &randomize_thread_per_instance);

    ZERO_INIT_AND_DEFINE_ARGS(args);

    if (!allocate_memory(&args, num_instances)) {
        fprintf(stderr, "Memory allocation failed\n");
        goto error_handle;
    }

    if (!initialize_memory(&args, num_instances)) {
        fprintf(stderr, "Memory initialization failed\n");
        goto error_handle;
    }

    const int threads_per_instance_len = 4;
    int       threads_per_instance[4]  = { 1, 3, 8, 29 };

#ifdef DLP_EXAMPLE_ENABLE_OPENMP
    // Expecting the api itself to be multi-threaded based on openmp.
    omp_set_max_active_levels(2);
#endif
    // Add timing variables
    double          start_time, end_time;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    start_time = ts.tv_sec + ts.tv_nsec / 1e9;

    for (int repeat = 0; repeat < n_repeats; repeat++) {
#ifdef DLP_EXAMPLE_ENABLE_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < num_instances; i++) {
            // Setting library internal thread count to a predetermined value.
            // Setting it outside the omp parallel does not work, since DLP
            // currently only updates a thread local rntm->num_threads value,
            // which will only be applicable to thread 0 and not others.
            int thread_idx = 0;
            if (randomize_thread_per_instance) {
                thread_idx = rand() % threads_per_instance_len;
            }
            dlp_thread_set_num_threads(threads_per_instance[thread_idx]);

            // Perform matrix multiplication: C = alpha * A * B + beta * C
            aocl_gemm_f32f32f32of32(
                args.order, args.transa[i], args.transb[i], args.m[i],
                args.n[i], args.k[i], args.alpha[i], args.a[i], args.lda[i],
                args.mem_format_a[i], args.b[i], args.ldb[i],
                args.mem_format_b[i], args.beta[i], args.c[i], args.ldc[i],
                args.mData[i]);
#ifdef DLP_EXAMPLE_ENABLE_OPENMP_DEBUG
#pragma omp critical
            printf("executed instance %d, repeat %d, tid: %d\n", i, repeat,
                   omp_get_thread_num());
#endif
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &ts);
    end_time            = ts.tv_sec + ts.tv_nsec / 1e9;
    double elapsed_time = end_time - start_time;

    printf("Total time for %d repeats: %f seconds, average time per repeat: %f "
           "seconds\n",
           n_repeats, elapsed_time, elapsed_time / n_repeats);

error_handle:
    free_memory(&args, num_instances);

    return 0;
}
