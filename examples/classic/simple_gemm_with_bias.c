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
 * Example demonstrating the usage of AOCL-DLP GEMM with bias post-operation
 *
 * This example shows how to:
 * 1. Set up the post-operation structure for adding bias
 * 2. Perform matrix multiplication with fused bias addition
 * 3. Compare with separate bias addition
 */

#include "aocl_dlp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// Utility function to initialize a bias vector
void
init_bias(float* bias, int length, float value)
{
    for (int i = 0; i < length; i++) {
        bias[i] = value * (i + 1);
    }
}

// Utility function to print a small section of a matrix
void
print_matrix_section(const char* name,
                     float*      matrix,
                     int         rows,
                     int         cols,
                     int         max_rows,
                     int         max_cols)
{
    printf("%s (%d x %d) - showing top-left %d x %d:\n", name, rows, cols,
           max_rows, max_cols);
    for (int i = 0; i < max_rows && i < rows; i++) {
        for (int j = 0; j < max_cols && j < cols; j++) {
            printf("%8.4f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Utility function to print a vector
void
print_vector(const char* name, float* vector, int length, int max_elements)
{
    printf("%s (length %d) - showing first %d elements:\n", name, length,
           max_elements);
    for (int i = 0; i < max_elements && i < length; i++) {
        printf("%8.4f ", vector[i]);
    }
    printf("\n\n");
}

// Utility function to add bias to a matrix (separate operation)
void
add_bias(float* matrix, float* bias, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] += bias[j];
        }
    }
}

int
main()
{
    // Matrix dimensions
    md_t m = 128; // Rows of A and C
    md_t n = 64;  // Columns of B and C
    md_t k = 128; // Columns of A and rows of B

    // Leading dimensions (assuming row-major storage)
    md_t lda = k;
    md_t ldb = n;
    md_t ldc = n;

    // Allocate memory for matrices
    float* a  = (float*)malloc(lda * m * sizeof(float));
    float* b  = (float*)malloc(ldb * k * sizeof(float));
    float* c1 = (float*)malloc(ldc * m * sizeof(float)); // For fused operation
    float* c2 =
        (float*)malloc(ldc * m * sizeof(float));     // For separate operations
    float* bias = (float*)malloc(n * sizeof(float)); // Bias vector

    if (!a || !b || !c1 || !c2 || !bias) {
        printf("Memory allocation failed\n");
        return -1;
    }

    // Initialize matrices with some values
    init_matrix(a, m, k, lda, 1.0f);
    init_matrix(b, k, n, ldb, 0.5f);
    memset(c1, 0, ldc * m * sizeof(float));
    memset(c2, 0, ldc * m * sizeof(float));
    init_bias(bias, n, 1.5f);

    // Print a small section of the input matrices and bias
    print_matrix_section("Matrix A", a, m, k, 3, 3);
    print_matrix_section("Matrix B", b, k, n, 3, 3);
    print_vector("Bias Vector", bias, n, 6);

    // GEMM parameters
    float alpha        = 1.0f; // Scalar for A*B
    float beta         = 0.0f; // Scalar for C
    char  order        = 'R';  // Row-major storage
    char  transa       = 'N';  // No transpose for A
    char  transb       = 'N';  // No transpose for B
    char  mem_format_a = 'N';  // A is not reordered
    char  mem_format_b = 'N';  // B is not reordered

    // Method 1: Perform matrix multiplication with separate bias addition
    aocl_gemm_f32f32f32of32(order, transa, transb, m, n, k, alpha, a, lda,
                            mem_format_a, b, ldb, mem_format_b, beta, c2, ldc,
                            NULL // No post-operations
    );

    // Add bias separately
    add_bias(c2, bias, m, n);

    // Method 2: Set up post-op for bias addition
    aocl_post_op* post_ops = (aocl_post_op*)malloc(sizeof(aocl_post_op));
    if (!post_ops) {
        printf("Memory allocation for post-ops failed\n");
        goto cleanup;
    }
    memset(post_ops, 0, sizeof(aocl_post_op));

    // Initialize post-ops structure
    post_ops->seq_length = 1; // One operation: bias

    // Allocate sequence vector
    post_ops->seq_vector =
        (AOCL_POST_OP_TYPE*)malloc(sizeof(AOCL_POST_OP_TYPE));
    if (!post_ops->seq_vector) {
        printf("Memory allocation for sequence vector failed\n");
        goto cleanup;
    }
    post_ops->seq_vector[0] = BIAS; // First operation is bias addition

    // Allocate and set up bias post-op
    post_ops->bias = (aocl_post_op_bias*)malloc(sizeof(aocl_post_op_bias));
    if (!post_ops->bias) {
        printf("Memory allocation for bias post-op failed\n");
        goto cleanup;
    }

    // Set the bias vector and its storage type
    post_ops->bias->bias      = bias;
    post_ops->bias->stor_type = AOCL_GEMM_F32; // Bias is in f32 format

    // Perform matrix multiplication with fused bias addition
    aocl_gemm_f32f32f32of32(order, transa, transb, m, n, k, alpha, a, lda,
                            mem_format_a, b, ldb, mem_format_b, beta, c1, ldc,
                            post_ops // With bias post-operation
    );

    // Print results for comparison
    print_matrix_section("Result Matrix (separate bias)", c2, m, n, 3, 3);
    print_matrix_section("Result Matrix (fused bias)", c1, m, n, 3, 3);

    // Compare a few elements to ensure both methods produce the same results
    int mismatch = 0;
    for (int i = 0; i < m && !mismatch; i++) {
        for (int j = 0; j < n && !mismatch; j++) {
            if (fabsf(c1[i * n + j] - c2[i * n + j]) > 1e-4) {
                mismatch = 1;
                printf("Mismatch found at position (%d, %d): Fused=%f, "
                       "Separate=%f\n",
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
    free(bias);

    // Free post-ops memory
    if (post_ops) {
        if (post_ops->seq_vector)
            free(post_ops->seq_vector);
        if (post_ops->bias)
            free(post_ops->bias);
        free(post_ops);
    }

    return 0;
}
