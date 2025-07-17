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
 * Example demonstrating batch matrix multiplication
 *
 * This example shows how to:
 * 1. Initialize multiple sets of matrices
 * 2. Perform batch matrix multiplication
 * 3. Compare with sequential execution of individual GEMMs
 */

#ifndef _WIN32
/* Define _POSIX_C_SOURCE to access POSIX functions in strict C11 mode */
#define _POSIX_C_SOURCE 200809L
#endif

#include "aocl_dlp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

// Utility function to initialize a matrix with values
void
init_matrix(float* matrix, int rows, int cols, float value)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = value * (i + j + 1) / (rows * cols);
        }
    }
}

// Simple timing function for performance comparison
double
get_time_sec()
{
#ifdef _WIN32
    static LARGE_INTEGER freq = { 0 };
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9;
#endif
}

int
main()
{
    printf("Batch Matrix Multiplication Example\n\n");

    // Number of matrices in the batch
    const md_t batch_size = 10;

    // Matrix dimensions for each matrix in the batch
    // We'll use the same dimensions for all matrices in this example
    md_t m = 256; // Rows of A and C
    md_t n = 256; // Columns of B and C
    md_t k = 256; // Columns of A and rows of B

    // Arrays to hold dimensions for each matrix in the batch
    md_t* m_array = (md_t*)malloc(batch_size * sizeof(md_t));
    md_t* n_array = (md_t*)malloc(batch_size * sizeof(md_t));
    md_t* k_array = (md_t*)malloc(batch_size * sizeof(md_t));

    // Arrays to hold leading dimensions for each matrix in the batch
    md_t* lda_array = (md_t*)malloc(batch_size * sizeof(md_t));
    md_t* ldb_array = (md_t*)malloc(batch_size * sizeof(md_t));
    md_t* ldc_array = (md_t*)malloc(batch_size * sizeof(md_t));

    // Arrays to hold alpha and beta values for each operation
    float* alpha_array = (float*)malloc(batch_size * sizeof(float));
    float* beta_array  = (float*)malloc(batch_size * sizeof(float));

    // Arrays to hold memory format flags
    char* mem_format_a_array = (char*)malloc(batch_size * sizeof(char));
    char* mem_format_b_array = (char*)malloc(batch_size * sizeof(char));

    // Arrays to hold transpose flags
    char* transa_array = (char*)malloc(batch_size * sizeof(char));
    char* transb_array = (char*)malloc(batch_size * sizeof(char));

    // Array to hold order flags
    char* order_array = (char*)malloc(batch_size * sizeof(char));

    // Check memory allocation
    if (!m_array || !n_array || !k_array || !lda_array || !ldb_array
        || !ldc_array || !alpha_array || !beta_array || !mem_format_a_array
        || !mem_format_b_array || !transa_array || !transb_array
        || !order_array) {
        printf("Memory allocation for dimension arrays failed\n");
        return -1;
    }

    // Initialize dimension arrays with the same values for each matrix
    for (int i = 0; i < batch_size; i++) {
        m_array[i]            = m;
        n_array[i]            = n;
        k_array[i]            = k;
        lda_array[i]          = k;
        ldb_array[i]          = n;
        ldc_array[i]          = n;
        alpha_array[i]        = 1.0f;
        beta_array[i]         = 0.0f;
        mem_format_a_array[i] = 'N';
        mem_format_b_array[i] = 'N';
        transa_array[i]       = 'N';
        transb_array[i]       = 'N';
        order_array[i]        = 'R'; // Row-major storage for all matrices
    }

    // Allocate arrays to hold pointers to the matrices
    float** a_array      = (float**)malloc(batch_size * sizeof(float*));
    float** b_array      = (float**)malloc(batch_size * sizeof(float*));
    float** c_batch      = (float**)malloc(batch_size * sizeof(float*));
    float** c_sequential = (float**)malloc(batch_size * sizeof(float*));

    if (!a_array || !b_array || !c_batch || !c_sequential) {
        printf("Memory allocation for matrix pointer arrays failed\n");
        return -1;
    }

    // Allocate and initialize each matrix in the batch
    printf("Initializing matrices...\n");
    for (int i = 0; i < batch_size; i++) {
        a_array[i] = (float*)malloc(lda_array[i] * m_array[i] * sizeof(float));
        b_array[i] = (float*)malloc(ldb_array[i] * k_array[i] * sizeof(float));
        c_batch[i] = (float*)malloc(ldc_array[i] * m_array[i] * sizeof(float));
        c_sequential[i] =
            (float*)malloc(ldc_array[i] * m_array[i] * sizeof(float));

        if (!a_array[i] || !b_array[i] || !c_batch[i] || !c_sequential[i]) {
            printf("Memory allocation for matrices failed\n");
            return -1;
        }

        // Initialize with different values for each matrix in the batch
        init_matrix(a_array[i], m, k, 1.0f + 0.1f * i);
        init_matrix(b_array[i], k, n, 0.5f + 0.1f * i);
        memset(c_batch[i], 0, m * n * sizeof(float));
        memset(c_sequential[i], 0, m * n * sizeof(float));
    }

    // Array for post-operations (NULL for each operation in this example)
    aocl_post_op** post_ops_array =
        (aocl_post_op**)malloc(batch_size * sizeof(aocl_post_op*));
    if (!post_ops_array) {
        printf("Memory allocation for post-ops array failed\n");
        return -1;
    }
    for (int i = 0; i < batch_size; i++) {
        post_ops_array[i] = NULL;
    }

    // Method 1: Execute GEMMs sequentially
    printf("Running %lld GEMM operations sequentially...\n",
           (long long)batch_size);
    double sequential_start_time = get_time_sec();

    for (int i = 0; i < batch_size; i++) {
        aocl_gemm_f32f32f32of32('R', // Row-major storage
                                transa_array[i], transb_array[i], m_array[i],
                                n_array[i], k_array[i], alpha_array[i],
                                a_array[i], lda_array[i], mem_format_a_array[i],
                                b_array[i], ldb_array[i], mem_format_b_array[i],
                                beta_array[i], c_sequential[i], ldc_array[i],
                                post_ops_array[i]);
    }

    double sequential_end_time = get_time_sec();
    double sequential_time     = sequential_end_time - sequential_start_time;
    printf("Sequential GEMM time: %.6f seconds\n", sequential_time);

    // Method 2: Execute GEMMs in batch
    printf("Running %lld GEMM operations in batch...\n", (long long)batch_size);
    double batch_start_time = get_time_sec();

    aocl_batch_gemm_f32f32f32of32(
        order_array, // Array of row-major storage flags for all matrices
        transa_array, transb_array, batch_size, m_array, n_array, k_array,
        alpha_array, (const float**)a_array, lda_array, mem_format_a_array,
        (const float**)b_array, ldb_array, mem_format_b_array, beta_array,
        c_batch, ldc_array, post_ops_array);

    double batch_end_time = get_time_sec();
    double batch_time     = batch_end_time - batch_start_time;
    printf("Batch GEMM time: %.6f seconds\n", batch_time);

    // Calculate and print performance improvement
    double speedup = sequential_time / batch_time;
    printf("Speedup from batch execution: %.2fx\n\n", speedup);

    // Verify the results match
    int mismatch = 0;
    for (int b = 0; b < batch_size && !mismatch; b++) {
        for (int i = 0; i < m && !mismatch; i++) {
            for (int j = 0; j < n && !mismatch; j++) {
                if (fabsf(c_sequential[b][i * n + j] - c_batch[b][i * n + j])
                    > 1e-4) {
                    mismatch = 1;
                    printf("Mismatch found in matrix %d at position (%d, %d): "
                           "Sequential=%f, Batch=%f\n",
                           b, i, j, c_sequential[b][i * n + j],
                           c_batch[b][i * n + j]);
                }
            }
        }
    }

    if (!mismatch) {
        printf("Results match: Both methods produce the same output.\n");
    }

    // Free allocated memory
    for (int i = 0; i < batch_size; i++) {
        free(a_array[i]);
        free(b_array[i]);
        free(c_batch[i]);
        free(c_sequential[i]);
    }

    free(a_array);
    free(b_array);
    free(c_batch);
    free(c_sequential);
    free(m_array);
    free(n_array);
    free(k_array);
    free(lda_array);
    free(ldb_array);
    free(ldc_array);
    free(alpha_array);
    free(beta_array);
    free(mem_format_a_array);
    free(mem_format_b_array);
    free(transa_array);
    free(transb_array);
    free(order_array);
    free(post_ops_array);

    return 0;
}
