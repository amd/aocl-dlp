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
 * Example demonstrating the usage of AOCL-DLP GEMM with Mish activation
 *
 * Mish(x) = x * tanh(softplus(x))
 *
 * This example shows how to:
 * 1. Set up the post-operation structure for Mish activation
 * 2. Perform matrix multiplication with fused Mish
 * 3. Compare with separate Mish application
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
                     int         ld,
                     int         max_rows,
                     int         max_cols)
{
    printf("%s (%d x %d) - showing top-left %d x %d:\n", name, rows, cols,
           max_rows, max_cols);
    for (int i = 0; i < max_rows && i < rows; i++) {
        for (int j = 0; j < max_cols && j < cols; j++) {
            printf("%8.4f ", matrix[i * ld + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/* Reference utility function for validation */

// Utility function to apply Mish activation separately
// Uses the numerically stable softplus form: max(x, 0) + log1p(exp(-|x|))
void
apply_mish(float* matrix, int rows, int cols, int ld)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float x            = matrix[i * ld + j];
            float sp           = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
            matrix[i * ld + j] = x * tanhf(sp);
        }
    }
}

int
main()
{
    // Initialize post-ops pointer to NULL to avoid uninitialized warnings
    dlp_metadata_t* mish_post_ops = NULL;
    int             ret           = 0;

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
    float* c1 = (float*)malloc(ldc * m * sizeof(float)); // For fused Mish
    float* c2 = (float*)malloc(ldc * m * sizeof(float)); // For separate Mish

    if (!a || !b || !c1 || !c2) {
        printf("Memory allocation failed\n");
        ret = -1;
        goto cleanup;
    }

    // Initialize matrices with some values (including negative values)
    init_matrix(a, m, k, lda, 1.0f);
    init_matrix(b, k, n, ldb, 0.5f);
    memset(c1, 0, ldc * m * sizeof(float));
    memset(c2, 0, ldc * m * sizeof(float));

    // Print a small section of the input matrices
    print_matrix_section("Matrix A", a, m, k, lda, 3, 3);
    print_matrix_section("Matrix B", b, k, n, ldb, 3, 3);

    // GEMM parameters
    float alpha        = 1.0f; // Scalar for A*B
    float beta         = 0.0f; // Scalar for C
    char  order        = 'R';  // Row-major storage
    char  transa       = 'N';  // No transpose for A
    char  transb       = 'N';  // No transpose for B
    char  mem_format_a = 'N';  // A is not reordered
    char  mem_format_b = 'N';  // B is not reordered

    // Method 1: Perform matrix multiplication with separate Mish activation
    aocl_gemm_f32f32f32of32(order, transa, transb, m, n, k, alpha, a, lda,
                            mem_format_a, b, ldb, mem_format_b, beta, c2, ldc,
                            NULL // No post-operations
    );

    // Apply Mish separately
    apply_mish(c2, m, n, ldc);

    // Method 2: Set up post-op for Mish activation
    mish_post_ops = (dlp_metadata_t*)malloc(sizeof(dlp_metadata_t));
    if (!mish_post_ops) {
        printf("Memory allocation for post-ops failed\n");
        ret = -1;
        goto cleanup;
    }
    memset(mish_post_ops, 0, sizeof(dlp_metadata_t));

    // Initialize post-ops structure for Mish
    mish_post_ops->seq_length = 1; // One operation: Mish

    // Allocate sequence vector
    mish_post_ops->seq_vector =
        (DLP_POST_OP_TYPE*)malloc(sizeof(DLP_POST_OP_TYPE));
    if (!mish_post_ops->seq_vector) {
        printf("Memory allocation for sequence vector failed\n");
        ret = -1;
        goto cleanup;
    }
    mish_post_ops->seq_vector[0] = ELTWISE; // First operation is element-wise

    // Allocate and set up eltwise post-op for Mish
    mish_post_ops->eltwise =
        (dlp_post_op_eltwise*)malloc(sizeof(dlp_post_op_eltwise));
    if (!mish_post_ops->eltwise) {
        printf("Memory allocation for eltwise post-op failed\n");
        ret = -1;
        goto cleanup;
    }

    mish_post_ops->eltwise->sf         = NULL; // No scaling for this example
    mish_post_ops->eltwise->algo.alpha = NULL;
    mish_post_ops->eltwise->algo.beta  = NULL;
    mish_post_ops->eltwise->algo.algo_type = MISH; // Use Mish activation

    // Perform matrix multiplication with fused Mish activation
    aocl_gemm_f32f32f32of32(order, transa, transb, m, n, k, alpha, a, lda,
                            mem_format_a, b, ldb, mem_format_b, beta, c1, ldc,
                            mish_post_ops // With Mish post-operation
    );

    // Print results for comparison
    print_matrix_section("Result Matrix (separate Mish)", c2, m, n, ldc, 3, 3);
    print_matrix_section("Result Matrix (fused Mish)", c1, m, n, ldc, 3, 3);

    // Compare the Mish results to ensure both methods produce the same results
    int mismatch = 0;
    for (int i = 0; i < m && !mismatch; i++) {
        for (int j = 0; j < n && !mismatch; j++) {
            if (fabsf(c1[i * ldc + j] - c2[i * ldc + j]) > 1e-4) {
                mismatch = 1;
                printf("Mismatch found at position (%d, %d): Fused=%f, "
                       "Separate=%f\n",
                       i, j, c1[i * ldc + j], c2[i * ldc + j]);
            }
        }
    }

    if (!mismatch) {
        printf("Results match: Both Mish methods produce the same output.\n");
    }

cleanup:
    // Free allocated memory
    free(a);
    free(b);
    free(c1);
    free(c2);

    // Free Mish post-ops memory
    if (mish_post_ops) {
        if (mish_post_ops->seq_vector)
            free(mish_post_ops->seq_vector);
        if (mish_post_ops->eltwise)
            free(mish_post_ops->eltwise);
        free(mish_post_ops);
    }

    return ret;
}
