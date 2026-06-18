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
 * Example demonstrating batch matrix multiplication with multiple groups
 *
 * This example shows how to:
 * 1. Initialize multiple groups of matrices with different dimensions
 * 2. Perform grouped batch matrix multiplication
 * 3. Compare with sequential execution of individual GEMMs
 *
 * Grouped batch execution allows:
 * - Different matrix dimensions per group (e.g., 128x128, 256x256, 64x512)
 * - Different alpha/beta parameters per group
 * - Efficient batching of heterogeneous workloads
 * - Optimal resource utilization across diverse matrix operations
 *
 * This API is similar to Intel's CBLAS cblas_gemm_batch API which uses
 * group_count and group_size parameters for grouped batch operations.
 * Unlike single-group batching, this example demonstrates the full power
 * of the grouped interface with multiple groups of different configurations.
 */

#include "aocl_dlp.h"
#include "classic/dlp_compat.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int
main()
{
    printf("Batch Matrix Multiplication Example with Multiple Groups\n\n");

    // Define multiple groups with different configurations
    // Group 1: Small matrices (4 matrices of 128x128)
    // Group 2: Medium matrices (3 matrices of 256x256)
    // Group 3: Rectangular matrices (2 matrices of 64x512)

    const md_t num_groups     = 3;
    md_t       group_sizes[3] = { 4, 3, 2 }; // Number of matrices in each group

    // Calculate total number of matrices across all groups
    md_t total_matrices = 0;
    for (int i = 0; i < num_groups; i++) {
        total_matrices += group_sizes[i];
    }
    printf("Total matrices across %lld groups: %lld\n", (long long)num_groups,
           (long long)total_matrices);
    printf("Group 1: %lld matrices (128x128x128)\n", (long long)group_sizes[0]);
    printf("Group 2: %lld matrices (256x256x256)\n", (long long)group_sizes[1]);
    printf("Group 3: %lld matrices (64x512x256)\n", (long long)group_sizes[2]);
    printf("\n");

    // Group-specific matrix dimensions
    md_t group_m[3] = { 128, 256, 64 };  // Rows of A and C for each group
    md_t group_n[3] = { 128, 256, 512 }; // Columns of B and C for each group
    md_t group_k[3] = { 128, 256,
                        256 }; // Columns of A and rows of B for each group

    // Group-specific parameters
    float group_alpha[3] = { 1.0f, 1.5f,
                             0.8f }; // Different alpha for each group
    float group_beta[3] = { 0.0f, 0.0f, 0.0f }; // Different beta for each group

    // Arrays for group parameters (indexed by group number, not total matrix
    // count)
    md_t*  m_array            = (md_t*)malloc(num_groups * sizeof(md_t));
    md_t*  n_array            = (md_t*)malloc(num_groups * sizeof(md_t));
    md_t*  k_array            = (md_t*)malloc(num_groups * sizeof(md_t));
    md_t*  lda_array          = (md_t*)malloc(num_groups * sizeof(md_t));
    md_t*  ldb_array          = (md_t*)malloc(num_groups * sizeof(md_t));
    md_t*  ldc_array          = (md_t*)malloc(num_groups * sizeof(md_t));
    float* alpha_array        = (float*)malloc(num_groups * sizeof(float));
    float* beta_array         = (float*)malloc(num_groups * sizeof(float));
    char*  mem_format_a_array = (char*)malloc(num_groups * sizeof(char));
    char*  mem_format_b_array = (char*)malloc(num_groups * sizeof(char));
    char*  transa_array       = (char*)malloc(num_groups * sizeof(char));
    char*  transb_array       = (char*)malloc(num_groups * sizeof(char));
    char*  order_array        = (char*)malloc(num_groups * sizeof(char));

    // Check memory allocation
    if (!m_array || !n_array || !k_array || !lda_array || !ldb_array
        || !ldc_array || !alpha_array || !beta_array || !mem_format_a_array
        || !mem_format_b_array || !transa_array || !transb_array
        || !order_array) {
        printf("Memory allocation for parameter arrays failed\n");
        return -1;
    }

    // Fill group parameter arrays (indexed by group number)
    for (int group = 0; group < num_groups; group++) {
        m_array[group]   = group_m[group];
        n_array[group]   = group_n[group];
        k_array[group]   = group_k[group];
        lda_array[group] = group_k[group]; // Leading dimension for row-major A
        ldb_array[group] = group_n[group]; // Leading dimension for row-major B
        ldc_array[group] = group_n[group]; // Leading dimension for row-major C
        alpha_array[group]        = group_alpha[group];
        beta_array[group]         = group_beta[group];
        mem_format_a_array[group] = 'N';
        mem_format_b_array[group] = 'N';
        transa_array[group]       = 'N';
        transb_array[group]       = 'N';
        order_array[group]        = 'R'; // Row-major storage
    }

    // Allocate arrays to hold pointers to the matrices
    float** a_array      = (float**)malloc(total_matrices * sizeof(float*));
    float** b_array      = (float**)malloc(total_matrices * sizeof(float*));
    float** c_batch      = (float**)malloc(total_matrices * sizeof(float*));
    float** c_sequential = (float**)malloc(total_matrices * sizeof(float*));

    if (!a_array || !b_array || !c_batch || !c_sequential) {
        printf("Memory allocation for matrix pointer arrays failed\n");
        return -1;
    }

    // Allocate and initialize each matrix
    printf("Initializing matrices...\n");
    md_t matrix_idx = 0;
    for (int group = 0; group < num_groups; group++) {
        md_t m = m_array[group];
        md_t n = n_array[group];
        md_t k = k_array[group];

        for (int i = 0; i < group_sizes[group]; i++) {
            a_array[matrix_idx]      = (float*)malloc(m * k * sizeof(float));
            b_array[matrix_idx]      = (float*)malloc(k * n * sizeof(float));
            c_batch[matrix_idx]      = (float*)malloc(m * n * sizeof(float));
            c_sequential[matrix_idx] = (float*)malloc(m * n * sizeof(float));

            if (!a_array[matrix_idx] || !b_array[matrix_idx]
                || !c_batch[matrix_idx] || !c_sequential[matrix_idx]) {
                printf("Memory allocation for matrix %lld failed\n",
                       (long long)matrix_idx);
                return -1;
            }

            // Initialize with different values for each matrix
            init_matrix(a_array[matrix_idx], m, k, 1.0f + 0.1f * matrix_idx);
            init_matrix(b_array[matrix_idx], k, n, 0.5f + 0.1f * matrix_idx);
            memset(c_batch[matrix_idx], 0, m * n * sizeof(float));
            memset(c_sequential[matrix_idx], 0, m * n * sizeof(float));
            matrix_idx++;
        }
    }

    // Array for post-operations (NULL for each operation in this example)
    dlp_metadata_t** post_ops_array =
        (dlp_metadata_t**)malloc(total_matrices * sizeof(dlp_metadata_t*));
    if (!post_ops_array) {
        printf("Memory allocation for post-ops array failed\n");
        return -1;
    }
    for (int i = 0; i < total_matrices; i++) {
        post_ops_array[i] = NULL;
    }

    // Method 1: Execute GEMMs sequentially
    printf("Running %lld GEMM operations sequentially...\n",
           (long long)total_matrices);
    double sequential_start_time = dlp_get_time_sec();

    matrix_idx = 0;
    for (int group = 0; group < num_groups; group++) {
        md_t  m            = m_array[group];
        md_t  n            = n_array[group];
        md_t  k            = k_array[group];
        float alpha        = alpha_array[group];
        float beta         = beta_array[group];
        md_t  lda          = lda_array[group];
        md_t  ldb          = ldb_array[group];
        md_t  ldc          = ldc_array[group];
        char  transa       = transa_array[group];
        char  transb       = transb_array[group];
        char  mem_format_a = mem_format_a_array[group];
        char  mem_format_b = mem_format_b_array[group];

        for (int i = 0; i < group_sizes[group]; i++) {
            aocl_gemm_f32f32f32of32(
                'R', // Row-major storage
                transa, transb, m, n, k, alpha, a_array[matrix_idx], lda,
                mem_format_a, b_array[matrix_idx], ldb, mem_format_b, beta,
                c_sequential[matrix_idx], ldc, post_ops_array[matrix_idx]);
            matrix_idx++;
        }
    }

    double sequential_end_time = dlp_get_time_sec();
    double sequential_time     = sequential_end_time - sequential_start_time;
    printf("Sequential GEMM time: %.6f seconds\n", sequential_time);

    // Method 2: Execute GEMMs in grouped batch
    printf("Running %lld GEMM operations in %lld groups using batch API...\n",
           (long long)total_matrices, (long long)num_groups);
    double batch_start_time = dlp_get_time_sec();

    // For multiple groups, we pass:
    // - group_count: Number of groups (3 in this case)
    // - group_sizes: Array where group_sizes[i] = number of matrices in group i
    // - Group parameters: m_array[group], n_array[group], etc. (indexed by
    // group)
    // - Matrix arrays: a_array[matrix], b_array[matrix], etc. (indexed by total
    // matrix count)
    aocl_batch_gemm_f32f32f32of32(
        order_array, // Array of storage layouts for groups
        transa_array, transb_array, m_array, n_array, k_array, alpha_array,
        (const float**)a_array, lda_array, (const float**)b_array, ldb_array,
        beta_array, c_batch, ldc_array, num_groups,
        group_sizes, // group_count=3, group_sizes=[4,3,2]
        mem_format_a_array, mem_format_b_array, post_ops_array);

    double batch_end_time = dlp_get_time_sec();
    double batch_time     = batch_end_time - batch_start_time;
    printf("Batch GEMM time: %.6f seconds\n", batch_time);

    // Calculate and print performance improvement
    double speedup = sequential_time / batch_time;
    printf("Speedup from batch execution: %.2fx\n\n", speedup);

    // Verify the results match
    int mismatch = 0;
    matrix_idx   = 0;
    for (int group = 0; group < num_groups && !mismatch; group++) {
        md_t m = m_array[group];
        md_t n = n_array[group];

        for (int mat_in_group = 0;
             mat_in_group < group_sizes[group] && !mismatch; mat_in_group++) {
            for (int i = 0; i < m && !mismatch; i++) {
                for (int j = 0; j < n && !mismatch; j++) {
                    if (fabsf(c_sequential[matrix_idx][i * n + j]
                              - c_batch[matrix_idx][i * n + j])
                        > 1e-4) {
                        mismatch = 1;
                        printf("Mismatch found in matrix %lld (group %d, "
                               "matrix %d) at position (%d, %d): "
                               "Sequential=%f, Batch=%f\n",
                               (long long)matrix_idx, group, mat_in_group, i, j,
                               c_sequential[matrix_idx][i * n + j],
                               c_batch[matrix_idx][i * n + j]);
                    }
                }
            }
            matrix_idx++;
        }
    }

    if (!mismatch) {
        printf("Results match: Both methods produce the same output.\n");
        printf("\nGrouped batch execution allows:\n");
        printf("- Different matrix dimensions per group\n");
        printf("- Different alpha/beta values per group\n");
        printf("- Efficient batching of heterogeneous workloads\n");
    }

    // Free allocated memory
    for (int i = 0; i < total_matrices; i++) {
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
