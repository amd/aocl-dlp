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
 * Example demonstrating multiple post-operations in sequence
 *
 * This example shows how to:
 * 1. Set up multiple post-operations in sequence
 * 2. Perform GEMM with Bias + ReLU + Scale operations
 * 3. Compare with separate application of each operation
 */

#include "aocl_dlp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Utility function to initialize a matrix with values (positive and negative)
void
init_matrix(float* matrix, int rows, int cols, float value)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Alternating positive and negative values
            float sign           = ((i + j) % 2 == 0) ? 1.0f : -1.0f;
            matrix[i * cols + j] = sign * value * (i + j + 1) / (rows * cols);
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

// Utility function to apply bias separately
void
apply_bias(float* matrix, float* bias, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] += bias[j];
        }
    }
}

// Utility function to apply ReLU separately
void
apply_relu(float* matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (matrix[i * cols + j] < 0) {
                matrix[i * cols + j] = 0;
            }
        }
    }
}

// Utility function to apply scaling separately
void
apply_scale(float* matrix, int rows, int cols, float scale)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] *= scale;
        }
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

int
main()
{
    printf("Multiple Post-Operations Example: Bias + ReLU + Scale\n\n");

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
    float* c1 = (float*)malloc(ldc * m * sizeof(float)); // For fused operations
    float* c2 =
        (float*)malloc(ldc * m * sizeof(float));     // For separate operations
    float* bias = (float*)malloc(n * sizeof(float)); // Bias vector

    if (!a || !b || !c1 || !c2 || !bias) {
        printf("Memory allocation failed\n");
        return -1;
    }

    // Initialize matrices with some values
    init_matrix(a, m, k, 1.0f);
    init_matrix(b, k, n, 0.5f);
    memset(c1, 0, m * n * sizeof(float));
    memset(c2, 0, m * n * sizeof(float));
    init_bias(bias, n, 0.5f);

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

    // Scaling factor for the third operation
    float scale_factor = 0.5f;

    // Method 1: Perform matrix multiplication with separate post-operations
    aocl_gemm_f32f32f32of32(order, transa, transb, m, n, k, alpha, a, lda,
                            mem_format_a, b, ldb, mem_format_b, beta, c2, ldc,
                            NULL // No post-operations
    );

    // Apply operations separately
    apply_bias(c2, bias, m, n);
    apply_relu(c2, m, n);
    apply_scale(c2, m, n, scale_factor);

    // Method 2: Set up post-ops for combined operations: bias + ReLU + scale
    aocl_post_op* post_ops = (aocl_post_op*)malloc(sizeof(aocl_post_op));
    if (!post_ops) {
        printf("Memory allocation for post-ops failed\n");
        goto cleanup;
    }
    memset(post_ops, 0, sizeof(aocl_post_op));

    // Initialize post-ops structure for 3 operations
    post_ops->seq_length = 3; // Three operations: bias + ReLU + scale

    // Allocate sequence vector
    post_ops->seq_vector =
        (AOCL_POST_OP_TYPE*)malloc(3 * sizeof(AOCL_POST_OP_TYPE));
    if (!post_ops->seq_vector) {
        printf("Memory allocation for sequence vector failed\n");
        goto cleanup;
    }

    // Set operation sequence: bias, then ReLU, then scale
    post_ops->seq_vector[0] = BIAS;    // First operation
    post_ops->seq_vector[1] = ELTWISE; // Second operation
    post_ops->seq_vector[2] = SCALE;   // Third operation

    // 1. Set up bias operation
    post_ops->bias = (aocl_post_op_bias*)malloc(sizeof(aocl_post_op_bias));
    if (!post_ops->bias) {
        printf("Memory allocation for bias post-op failed\n");
        goto cleanup;
    }

    // Set the bias vector and its storage type
    post_ops->bias->bias      = bias;
    post_ops->bias->stor_type = AOCL_GEMM_F32; // Bias is in f32 format

    // 2. Set up ReLU operation (elementwise)
    post_ops->eltwise =
        (aocl_post_op_eltwise*)malloc(sizeof(aocl_post_op_eltwise));
    if (!post_ops->eltwise) {
        printf("Memory allocation for eltwise post-op failed\n");
        goto cleanup;
    }

    post_ops->eltwise->is_power_of_2    = 0;
    post_ops->eltwise->scale_factor     = NULL;
    post_ops->eltwise->scale_factor_len = 0;
    post_ops->eltwise->algo.alpha       = NULL;
    post_ops->eltwise->algo.beta        = NULL;
    post_ops->eltwise->algo.algo_type   = RELU; // Use ReLU activation

    // 3. Set up scale operation
    post_ops->sum = (aocl_post_op_sum*)malloc(sizeof(aocl_post_op_sum));
    if (!post_ops->sum) {
        printf("Memory allocation for sum/scale post-op failed\n");
        goto cleanup;
    }

    // Set up scaling factor
    post_ops->sum->is_power_of_2 = 0; // Not a power of 2
    post_ops->sum->scale_factor  = malloc(sizeof(float));
    if (!post_ops->sum->scale_factor) {
        printf("Memory allocation for scale factor failed\n");
        goto cleanup;
    }
    *((float*)post_ops->sum->scale_factor) = scale_factor;
    post_ops->sum->scale_factor_len        = 1;

    // Set up zero point parameters
    post_ops->sum->zero_point = malloc(sizeof(int8_t));
    if (!post_ops->sum->zero_point) {
        printf("Memory allocation for zero point failed\n");
        goto cleanup;
    }
    *((int8_t*)post_ops->sum->zero_point) = 0;
    post_ops->sum->zero_point_len         = 1;
    post_ops->sum->buff                   = NULL;

    // Perform matrix multiplication with fused operations
    aocl_gemm_f32f32f32of32(order, transa, transb, m, n, k, alpha, a, lda,
                            mem_format_a, b, ldb, mem_format_b, beta, c1, ldc,
                            post_ops // With fused operations
    );

    // Print results for comparison
    printf("Results after applying operations:\n");
    print_matrix_section(
        "Result with separate operations (bias + ReLU + scale)", c2, m, n, 3,
        3);
    print_matrix_section("Result with fused operations (bias + ReLU + scale)",
                         c1, m, n, 3, 3);

    // Compare results to ensure both methods produce the same results
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

    // Example of GeLU_Tanh activation instead of ReLU
    printf("\nExample with GeLU_Tanh activation instead of ReLU:\n");

    // Reset output matrices
    memset(c1, 0, m * n * sizeof(float));
    memset(c2, 0, m * n * sizeof(float));

    // Reset the post-ops structure for the new example
    post_ops->eltwise->algo.algo_type =
        GELU_TANH; // Change to GeLU_Tanh activation

    // Perform matrix multiplication with fused operations using GeLU_Tanh
    aocl_gemm_f32f32f32of32(order, transa, transb, m, n, k, alpha, a, lda,
                            mem_format_a, b, ldb, mem_format_b, beta, c1, ldc,
                            post_ops // With fused operations using GeLU_Tanh
    );

    // Print result with GeLU_Tanh
    print_matrix_section(
        "Result with fused operations (bias + GeLU_Tanh + scale)", c1, m, n, 3,
        3);

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
        if (post_ops->eltwise)
            free(post_ops->eltwise);
        if (post_ops->sum) {
            if (post_ops->sum->scale_factor)
                free(post_ops->sum->scale_factor);
            if (post_ops->sum->zero_point)
                free(post_ops->sum->zero_point);
            free(post_ops->sum);
        }
        free(post_ops);
    }

    return 0;
}
