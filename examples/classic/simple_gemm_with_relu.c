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
 * Example demonstrating the usage of AOCL-DLP GEMM with ReLU activation
 *
 * This example shows how to:
 * 1. Set up the post-operation structure for ReLU activation
 * 2. Perform matrix multiplication with fused ReLU
 * 3. Compare with separate ReLU application
 * 4. Demonstrate PReLU with custom scaling factor
 */

#include "aocl_dlp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Utility function to initialize a matrix with values (positive and negative)
void
init_matrix(float* matrix, int rows, int cols, int ld, float value)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Alternating positive and negative values
            float sign         = ((i + j) % 2 == 0) ? 1.0f : -1.0f;
            matrix[i * ld + j] = sign * value * (i + j + 1) / (rows * cols);
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

/* Reference utility functions for validation */

// Utility function to apply ReLU activation separately
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

// Utility function to apply PReLU activation separately
void
apply_prelu(float* matrix, int rows, int cols, float scale)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (matrix[i * cols + j] < 0) {
                matrix[i * cols + j] *= scale;
            }
        }
    }
}

int
main()
{
    // Initialize post-ops pointers to NULL to avoid uninitialized warnings
    aocl_post_op* relu_post_ops  = NULL;
    aocl_post_op* prelu_post_ops = NULL;

    // Matrix dimensions
    md_t m = 128; // Rows of A and C
    md_t n = 128; // Columns of B and C
    md_t k = 128; // Columns of A and rows of B

    // Leading dimensions (assuming row-major storage)
    md_t lda = k;
    md_t ldb = n;
    md_t ldc = n;

    // Allocate memory for matrices
    float* a  = (float*)malloc(lda * m * sizeof(float));
    float* b  = (float*)malloc(ldb * k * sizeof(float));
    float* c1 = (float*)malloc(ldc * m * sizeof(float)); // For fused ReLU
    float* c2 = (float*)malloc(ldc * m * sizeof(float)); // For separate ReLU
    float* c3 = (float*)malloc(ldc * m * sizeof(float)); // For PReLU

    if (!a || !b || !c1 || !c2 || !c3) {
        printf("Memory allocation failed\n");
        return -1;
    }

    // Initialize matrices with some values (including negative values)
    init_matrix(a, m, k, lda, 1.0f);
    init_matrix(b, k, n, ldb, 0.5f);
    memset(c1, 0, ldc * m * sizeof(float));
    memset(c2, 0, ldc * m * sizeof(float));
    memset(c3, 0, ldc * m * sizeof(float));

    // Print a small section of the input matrices
    print_matrix_section("Matrix A", a, m, k, 3, 3);
    print_matrix_section("Matrix B", b, k, n, 3, 3);

    // GEMM parameters
    float alpha        = 1.0f; // Scalar for A*B
    float beta         = 0.0f; // Scalar for C
    char  order        = 'R';  // Row-major storage
    char  transa       = 'N';  // No transpose for A
    char  transb       = 'N';  // No transpose for B
    char  mem_format_a = 'N';  // A is not reordered
    char  mem_format_b = 'N';  // B is not reordered

    // Method 1: Perform matrix multiplication with separate ReLU activation
    aocl_gemm_f32f32f32of32(order, transa, transb, m, n, k, alpha, a, lda,
                            mem_format_a, b, ldb, mem_format_b, beta, c2, ldc,
                            NULL // No post-operations
    );

    // Apply ReLU separately
    apply_relu(c2, m, n);

    // Method 2: Set up post-op for ReLU activation
    relu_post_ops = (aocl_post_op*)malloc(sizeof(aocl_post_op));
    if (!relu_post_ops) {
        printf("Memory allocation for post-ops failed\n");
        goto cleanup;
    }
    memset(relu_post_ops, 0, sizeof(aocl_post_op));

    // Initialize post-ops structure for ReLU
    relu_post_ops->seq_length = 1; // One operation: ReLU

    // Allocate sequence vector
    relu_post_ops->seq_vector =
        (AOCL_POST_OP_TYPE*)malloc(sizeof(AOCL_POST_OP_TYPE));
    if (!relu_post_ops->seq_vector) {
        printf("Memory allocation for sequence vector failed\n");
        goto cleanup;
    }
    relu_post_ops->seq_vector[0] = ELTWISE; // First operation is element-wise

    // Allocate and set up eltwise post-op for ReLU
    relu_post_ops->eltwise =
        (aocl_post_op_eltwise*)malloc(sizeof(aocl_post_op_eltwise));
    if (!relu_post_ops->eltwise) {
        printf("Memory allocation for eltwise post-op failed\n");
        goto cleanup;
    }

    relu_post_ops->eltwise->is_power_of_2    = 0;
    relu_post_ops->eltwise->scale_factor     = NULL;
    relu_post_ops->eltwise->scale_factor_len = 0;
    relu_post_ops->eltwise->algo.alpha       = NULL;
    relu_post_ops->eltwise->algo.beta        = NULL;
    relu_post_ops->eltwise->algo.algo_type   = RELU; // Use ReLU activation

    // Perform matrix multiplication with fused ReLU activation
    aocl_gemm_f32f32f32of32(order, transa, transb, m, n, k, alpha, a, lda,
                            mem_format_a, b, ldb, mem_format_b, beta, c1, ldc,
                            relu_post_ops // With ReLU post-operation
    );

    // Method 3: PReLU with custom scaling factor
    float prelu_scale = 0.1f; // Leaky ReLU with scale 0.1

    // First compute matrix multiplication separately
    aocl_gemm_f32f32f32of32(order, transa, transb, m, n, k, alpha, a, lda,
                            mem_format_a, b, ldb, mem_format_b, beta, c3, ldc,
                            NULL // No post-operations
    );

    // Apply PReLU separately
    apply_prelu(c3, m, n, prelu_scale);

    // Now set up post-op for PReLU activation
    prelu_post_ops = (aocl_post_op*)malloc(sizeof(aocl_post_op));
    if (!prelu_post_ops) {
        printf("Memory allocation for PReLU post-ops failed\n");
        goto cleanup;
    }
    memset(prelu_post_ops, 0, sizeof(aocl_post_op));

    // Initialize post-ops structure for PReLU
    prelu_post_ops->seq_length = 1; // One operation: PReLU

    // Allocate sequence vector
    prelu_post_ops->seq_vector =
        (AOCL_POST_OP_TYPE*)malloc(sizeof(AOCL_POST_OP_TYPE));
    if (!prelu_post_ops->seq_vector) {
        printf("Memory allocation for PReLU sequence vector failed\n");
        goto cleanup;
    }
    prelu_post_ops->seq_vector[0] = ELTWISE; // First operation is element-wise

    // Allocate and set up eltwise post-op for PReLU
    prelu_post_ops->eltwise =
        (aocl_post_op_eltwise*)malloc(sizeof(aocl_post_op_eltwise));
    if (!prelu_post_ops->eltwise) {
        printf("Memory allocation for PReLU eltwise post-op failed\n");
        goto cleanup;
    }

    // Set up PReLU parameters
    prelu_post_ops->eltwise->is_power_of_2    = 0;
    prelu_post_ops->eltwise->scale_factor     = NULL;
    prelu_post_ops->eltwise->scale_factor_len = 0;

    // Alpha parameter for PReLU (scaling factor for negative values)
    prelu_post_ops->eltwise->algo.alpha = malloc(sizeof(float));
    if (!prelu_post_ops->eltwise->algo.alpha) {
        printf("Memory allocation for PReLU alpha parameter failed\n");
        goto cleanup;
    }
    *((float*)prelu_post_ops->eltwise->algo.alpha) = prelu_scale;

    prelu_post_ops->eltwise->algo.beta      = NULL;
    prelu_post_ops->eltwise->algo.algo_type = PRELU; // Use PReLU activation

    // Print results for comparison
    print_matrix_section("Result Matrix (separate ReLU)", c2, m, n, 3, 3);
    print_matrix_section("Result Matrix (fused ReLU)", c1, m, n, 3, 3);
    print_matrix_section("Result Matrix (PReLU with scale 0.1)", c3, m, n, 3,
                         3);

    // Compare the ReLU results to ensure both methods produce the same results
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
        printf("Results match: Both ReLU methods produce the same output.\n");
    }

cleanup:
    // Free allocated memory
    free(a);
    free(b);
    free(c1);
    free(c2);
    free(c3);

    // Free ReLU post-ops memory
    if (relu_post_ops) {
        if (relu_post_ops->seq_vector)
            free(relu_post_ops->seq_vector);
        if (relu_post_ops->eltwise)
            free(relu_post_ops->eltwise);
        free(relu_post_ops);
    }

    // Free PReLU post-ops memory
    if (prelu_post_ops) {
        if (prelu_post_ops->seq_vector)
            free(prelu_post_ops->seq_vector);
        if (prelu_post_ops->eltwise) {
            if (prelu_post_ops->eltwise->algo.alpha)
                free(prelu_post_ops->eltwise->algo.alpha);
            free(prelu_post_ops->eltwise);
        }
        free(prelu_post_ops);
    }

    return 0;
}
