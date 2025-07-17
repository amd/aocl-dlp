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
 * Example demonstrating matrix reordering for optimized computation
 *
 * This example shows how to:
 * 1. Reorder a matrix for optimal computation
 * 2. Use the reordered matrix in GEMM operations
 * 3. Compare performance with and without reordering
 * 4. Perform multiple GEMM operations with the same reordered weights
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
    printf("Matrix Reordering Example for Optimized Computation\n\n");

    // Matrix dimensions
    md_t m = 2048; // Rows of A and C
    md_t n = 1024; // Columns of B and C
    md_t k = 1024; // Columns of A and rows of B

    // Leading dimensions (assuming row-major storage)
    md_t lda = k;
    md_t ldb = n;
    md_t ldc = n;

    // Number of iterations to test performance
    int num_iterations = 10;

    // Allocate memory for matrices
    float* a  = (float*)malloc(lda * m * sizeof(float));
    float* b  = (float*)malloc(ldb * k * sizeof(float));
    float* c1 = (float*)malloc(ldc * m * sizeof(float)); // For standard GEMM
    float* c2 =
        (float*)malloc(ldc * m * sizeof(float)); // For GEMM with reordered B

    if (!a || !b || !c1 || !c2) {
        printf("Memory allocation failed\n");
        return -1;
    }

    // Initialize matrices with some values
    printf("Initializing matrices...\n");
    init_matrix(a, m, k, 1.0f);
    init_matrix(b, k, n, 0.5f);
    memset(c1, 0, m * n * sizeof(float));
    memset(c2, 0, m * n * sizeof(float));

    // GEMM parameters
    float alpha        = 1.0f; // Scalar for A*B
    float beta         = 0.0f; // Scalar for C
    char  order        = 'R';  // Row-major storage
    char  transa       = 'N';  // No transpose for A
    char  transb       = 'N';  // No transpose for B
    char  mem_format_a = 'N';  // A is not reordered
    char  mem_format_b = 'N';  // B is not reordered for standard GEMM

    // Step 1: Determine the buffer size needed for the reordered matrix
    msz_t reorder_buffer_size =
        aocl_get_reorder_buf_size_f32f32f32of32(order, transb, 'B', k, n);

    printf("Reorder buffer size required: %zu bytes\n", reorder_buffer_size);

    // Step 2: Allocate memory for the reordered matrix
    float* b_reordered = (float*)malloc(reorder_buffer_size);
    if (!b_reordered) {
        printf("Memory allocation for reordered matrix failed\n");
        goto cleanup;
    }

    // Step 3: Reorder the B matrix
    printf("Reordering matrix B...\n");
    double reorder_start_time = get_time_sec();

    aocl_reorder_f32f32f32of32(order, transb, 'B', b, b_reordered, k, n, ldb);

    double reorder_end_time = get_time_sec();
    printf("Matrix reordering took %.6f seconds\n\n",
           reorder_end_time - reorder_start_time);

    // Step 4: Perform GEMM with standard (non-reordered) matrix
    printf("Performing %d iterations of standard GEMM...\n", num_iterations);
    double std_gemm_start_time = get_time_sec();

    for (int i = 0; i < num_iterations; i++) {
        // Reset C matrix for each iteration
        memset(c1, 0, m * n * sizeof(float));

        aocl_gemm_f32f32f32of32(order, transa, transb, m, n, k, alpha, a, lda,
                                mem_format_a, b, ldb, mem_format_b, beta, c1,
                                ldc,
                                NULL // No post-operations
        );
    }

    double std_gemm_end_time = get_time_sec();
    double std_gemm_avg_time =
        (std_gemm_end_time - std_gemm_start_time) / num_iterations;
    printf("Standard GEMM average time: %.6f seconds\n", std_gemm_avg_time);

    // Step 5: Perform GEMM with reordered matrix
    printf("Performing %d iterations of GEMM with reordered matrix...\n",
           num_iterations);
    double reordered_gemm_start_time = get_time_sec();

    // Use 'R' to indicate that B is reordered
    char reordered_format = 'R';

    for (int i = 0; i < num_iterations; i++) {
        // Reset C matrix for each iteration
        memset(c2, 0, m * n * sizeof(float));

        aocl_gemm_f32f32f32of32(order, transa, transb, m, n, k, alpha, a, lda,
                                mem_format_a, b_reordered, ldb,
                                reordered_format, beta, c2, ldc,
                                NULL // No post-operations
        );
    }

    double reordered_gemm_end_time = get_time_sec();
    double reordered_gemm_avg_time =
        (reordered_gemm_end_time - reordered_gemm_start_time) / num_iterations;
    printf("GEMM with reordered matrix average time: %.6f seconds\n",
           reordered_gemm_avg_time);

    // Calculate and print performance improvement
    double speedup = std_gemm_avg_time / reordered_gemm_avg_time;
    printf("Speedup from matrix reordering: %.2fx\n\n", speedup);

    // Calculate total time for multiple standard GEMMs vs. reordering +
    // multiple GEMMs
    double total_std_time       = std_gemm_avg_time * num_iterations;
    double total_reordered_time = (reorder_end_time - reorder_start_time)
                                  + (reordered_gemm_avg_time * num_iterations);

    printf("Total time comparison for %d GEMM operations:\n", num_iterations);
    printf("  Standard GEMM:           %.6f seconds\n", total_std_time);
    printf("  Reordering + GEMM:       %.6f seconds\n", total_reordered_time);

    if (total_reordered_time < total_std_time) {
        printf("  Reordering provided a %.2fx total speedup including "
               "reordering cost\n\n",
               total_std_time / total_reordered_time);
    } else {
        printf("  Reordering overhead exceeded the performance benefit for %d "
               "iterations\n",
               num_iterations);
        int break_even = (int)((reorder_end_time - reorder_start_time)
                               / (std_gemm_avg_time - reordered_gemm_avg_time));
        printf("  Break-even point: approximately %d iterations\n\n",
               break_even);
    }

    // Verify the results match
    int mismatch = 0;
    for (int i = 0; i < m && !mismatch; i++) {
        for (int j = 0; j < n && !mismatch; j++) {
            if (fabsf(c1[i * n + j] - c2[i * n + j]) > 1e-4) {
                mismatch = 1;
                printf("Mismatch found at position (%d, %d): Standard=%f, "
                       "Reordered=%f\n",
                       i, j, c1[i * n + j], c2[i * n + j]);
            }
        }
    }

    if (!mismatch) {
        printf("Results match: Both methods produce the same output.\n");
    }

cleanup:
    // Free allocated memory
    free(a);
    free(b);
    free(c1);
    free(c2);
    if (b_reordered)
        free(b_reordered);

    return 0;
}
