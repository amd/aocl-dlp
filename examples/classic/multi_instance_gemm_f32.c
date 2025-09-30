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
    char    order;
    md_t*   m;
    md_t*   n;
    md_t*   k;
    md_t*   lda;
    md_t*   ldb;
    md_t*   ldc;
    float** a;
    float** b;
    float** c;
    float*  alpha;
    float*  beta;
    char*   transa;
    char*   transb;
    char*   mem_format_a;
    char*   mem_format_b;
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

    if (!args->m || !args->n || !args->k || !args->lda || !args->ldb
        || !args->ldc || !args->a || !args->b || !args->c || !args->alpha
        || !args->beta || !args->transa || !args->transb || !args->mem_format_a
        || !args->mem_format_b) {
        return false;
    }

    // Initialize all pointers to NULL
    for (int i = 0; i < num_instances; i++) {
        args->a[i] = NULL;
        args->b[i] = NULL;
        args->c[i] = NULL;
    }

    return true;
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

        args->alpha[i]        = (rand() % 2 == 0) ? 1.0f : 2.0f;
        args->beta[i]         = (rand() % 2 == 0) ? 1.0f : 9.0f;
        args->transa[i]       = 'N'; // No transpose for A
        args->transb[i]       = 'N'; // No transpose for B
        args->mem_format_a[i] = 'N'; // A is not reordered
        args->mem_format_b[i] =
            (rand() % 2 == 0) ? 'N' : 'R'; // B is randomly reordered or not.
    }

    return true;
}

// Only supporting row major for this example.
#define ZERO_INIT_AND_DEFINE_ARGS(var_name)                                    \
    multi_instance_gemm_f32_args_t var_name = {                                \
        'R',  NULL, NULL, NULL, NULL, NULL, NULL, NULL,                        \
        NULL, NULL, NULL, NULL, NULL, NULL, NULL                               \
    };

void
parse_cmd_line_args(int argc, char* argv[], int* num_instances, int* n_repeats)
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
        }
    }
}

int
main(int argc, char* argv[])
{
    int num_instances = 64; // Number of instances.
    int n_repeats     = 1;  // Configure number of repetitions

    parse_cmd_line_args(argc, argv, &num_instances, &n_repeats);

    ZERO_INIT_AND_DEFINE_ARGS(args);

    if (!allocate_memory(&args, num_instances)) {
        fprintf(stderr, "Memory allocation failed\n");
        goto error_handle;
    }

    if (!initialize_memory(&args, num_instances)) {
        fprintf(stderr, "Memory initialization failed\n");
        goto error_handle;
    }

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
            // Setting library internal thread count to 1. Setting it outside
            // the omp parallel does not work, since DLP currently only updates
            // a thread local rntm->num_threads value, which will only be
            // applicable to thread 0 and not others.
            dlp_thread_set_num_threads(1);

            // Perform matrix multiplication: C = alpha * A * B + beta * C
            aocl_gemm_f32f32f32of32(
                args.order, args.transa[i], args.transb[i], args.m[i],
                args.n[i], args.k[i], args.alpha[i], args.a[i], args.lda[i],
                args.mem_format_a[i], args.b[i], args.ldb[i],
                args.mem_format_b[i], args.beta[i], args.c[i], args.ldc[i],
                NULL // No other operations.
            );
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
