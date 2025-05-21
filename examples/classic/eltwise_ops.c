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
 * Example demonstrating element-wise operations without GEMM
 *
 * This example shows how to:
 * 1. Perform element-wise operations directly on matrices
 * 2. Apply different activation functions (GeLU, Tanh, ReLU)
 * 3. Convert between different data types (f32, bf16)
 */

#include "aocl_gemm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Utility function to initialize a matrix with values
void
init_matrix_f32(float* matrix, int rows, int cols, float value)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Generate values between -1 and 1
            float sign           = ((i + j) % 2 == 0) ? 1.0f : -1.0f;
            matrix[i * cols + j] = sign * value * (i + j + 1) / (rows * cols);
        }
    }
}

// Utility function to print a small section of a f32 matrix
void
print_matrix_section_f32(const char*  name,
                         const float* matrix,
                         int          rows,
                         int          cols,
                         int          max_rows,
                         int          max_cols)
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

// Utility function to print a small section of a bf16 matrix
void
print_matrix_section_bf16(const char*     name,
                          const bfloat16* matrix,
                          int             rows,
                          int             cols,
                          int             max_rows,
                          int             max_cols)
{
    printf("%s (%d x %d) - showing top-left %d x %d:\n", name, rows, cols,
           max_rows, max_cols);
    for (int i = 0; i < max_rows && i < rows; i++) {
        for (int j = 0; j < max_cols && j < cols; j++) {
            // Convert bfloat16 to float for printing
            uint32_t float_bits = ((uint32_t)matrix[i * cols + j]) << 16;
            float    float_val;
            memcpy(&float_val, &float_bits, sizeof(float));

            printf("%8.4f ", float_val);
        }
        printf("\n");
    }
    printf("\n");
}

// Convert float to bfloat16
bfloat16
f32_to_bf16(float f)
{
    uint32_t* float_bits = (uint32_t*)&f;
    return (bfloat16)(*float_bits >> 16);
}

// Utility function to manually convert float data to bfloat16
void
convert_f32_to_bf16(const float* src, bfloat16* dst, int size)
{
    for (int i = 0; i < size; i++) {
        dst[i] = f32_to_bf16(src[i]);
    }
}

// Naive implementation of GeLU for reference
void
apply_gelu_tanh_ref(float* matrix, int size)
{
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff          = 0.044715f;

    for (int i = 0; i < size; i++) {
        float x          = matrix[i];
        float x3         = x * x * x;
        float inner      = sqrt_2_over_pi * (x + coeff * x3);
        float tanh_inner = tanhf(inner);
        matrix[i]        = 0.5f * x * (1.0f + tanh_inner);
    }
}

// Naive implementation of Tanh for reference
void
apply_tanh_ref(float* matrix, int size)
{
    for (int i = 0; i < size; i++) {
        matrix[i] = tanhf(matrix[i]);
    }
}

// Function to check if values are close
int
is_close(float a, float b, float rtol, float atol)
{
    return fabsf(a - b) <= (atol + rtol * fabsf(b));
}

int
main()
{
    printf("Element-wise Operations Example (without GEMM)\n\n");

    // Matrix dimensions
    md_t m = 64; // Rows
    md_t n = 64; // Columns

    // Allocate memory for matrices
    float*    a_f32     = (float*)malloc(m * n * sizeof(float));
    float*    b_f32     = (float*)malloc(m * n * sizeof(float));
    float*    b_f32_ref = (float*)malloc(m * n * sizeof(float));
    bfloat16* a_bf16    = (bfloat16*)malloc(m * n * sizeof(bfloat16));
    bfloat16* b_bf16    = (bfloat16*)malloc(m * n * sizeof(bfloat16));

    if (!a_f32 || !b_f32 || !b_f32_ref || !a_bf16 || !b_bf16) {
        printf("Memory allocation failed\n");
        return -1;
    }

    // Initialize matrices with some values
    printf("Initializing matrices...\n");
    init_matrix_f32(a_f32, m, n, 2.0f);
    memset(b_f32, 0, m * n * sizeof(float));
    memset(b_f32_ref, 0, m * n * sizeof(float));

    // Print a small section of the input float matrix
    print_matrix_section_f32("Input Matrix A (float)", a_f32, m, n, 3, 3);

    // Example 1: Apply GeLU_Tanh activation to float matrix
    printf("Example 1: Applying GeLU_Tanh activation to float matrix\n");

    // Copy input to reference matrix for manual calculation
    memcpy(b_f32_ref, a_f32, m * n * sizeof(float));

    // Leading dimensions
    md_t lda = n;
    md_t ldb = n;

    // Set up post-op for GeLU_Tanh
    aocl_post_op* gelu_post_ops = (aocl_post_op*)malloc(sizeof(aocl_post_op));
    if (!gelu_post_ops) {
        printf("Memory allocation for post-ops failed\n");
        goto cleanup;
    }
    memset(gelu_post_ops, 0, sizeof(aocl_post_op));

    // Initialize post-ops structure
    gelu_post_ops->seq_length = 1; // One operation: GeLU_Tanh

    // Allocate sequence vector
    gelu_post_ops->seq_vector =
        (AOCL_POST_OP_TYPE*)malloc(sizeof(AOCL_POST_OP_TYPE));
    if (!gelu_post_ops->seq_vector) {
        printf("Memory allocation for sequence vector failed\n");
        goto cleanup;
    }
    gelu_post_ops->seq_vector[0] = ELTWISE; // Element-wise operation

    // Allocate and set up eltwise post-op for GeLU_Tanh
    gelu_post_ops->eltwise =
        (aocl_post_op_eltwise*)malloc(sizeof(aocl_post_op_eltwise));
    if (!gelu_post_ops->eltwise) {
        printf("Memory allocation for eltwise post-op failed\n");
        goto cleanup;
    }

    gelu_post_ops->eltwise->is_power_of_2    = 0;
    gelu_post_ops->eltwise->scale_factor     = NULL;
    gelu_post_ops->eltwise->scale_factor_len = 0;
    gelu_post_ops->eltwise->algo.alpha       = NULL;
    gelu_post_ops->eltwise->algo.beta        = NULL;
    gelu_post_ops->eltwise->algo.algo_type =
        GELU_TANH; // Use GeLU_Tanh activation

    // Apply GeLU_Tanh using element-wise operation API
    aocl_gemm_eltwise_ops_f32of32('R', 'N', 'N', // Row-major, no transposes
                                  m, n, a_f32, lda, b_f32, ldb, gelu_post_ops);

    // Apply reference GeLU_Tanh
    apply_gelu_tanh_ref(b_f32_ref, m * n);

    // Print result with GeLU_Tanh
    print_matrix_section_f32("Output Matrix B (float) with GeLU_Tanh", b_f32, m,
                             n, 3, 3);
    print_matrix_section_f32("Reference Matrix B (float) with GeLU_Tanh",
                             b_f32_ref, m, n, 3, 3);

    // Check if results are close
    int gelu_mismatch = 0;
    for (int i = 0; i < m * n && !gelu_mismatch; i++) {
        if (!is_close(b_f32[i], b_f32_ref[i], 1e-4f, 1e-4f)) {
            gelu_mismatch = 1;
            printf("GeLU Mismatch found at index %d: AOCL=%f, Reference=%f\n",
                   i, b_f32[i], b_f32_ref[i]);
        }
    }

    if (!gelu_mismatch) {
        printf("GeLU_Tanh results match between AOCL and reference "
               "implementation.\n\n");
    }

    // Example 2: Apply Tanh activation to float matrix
    printf("Example 2: Applying Tanh activation to float matrix\n");

    // Reset output matrices
    memset(b_f32, 0, m * n * sizeof(float));
    memcpy(b_f32_ref, a_f32, m * n * sizeof(float));

    // Update post-op for Tanh
    gelu_post_ops->eltwise->algo.algo_type = TANH; // Change to Tanh activation

    // Apply Tanh using element-wise operation API
    aocl_gemm_eltwise_ops_f32of32('R', 'N', 'N', // Row-major, no transposes
                                  m, n, a_f32, lda, b_f32, ldb, gelu_post_ops);

    // Apply reference Tanh
    apply_tanh_ref(b_f32_ref, m * n);

    // Print result with Tanh
    print_matrix_section_f32("Output Matrix B (float) with Tanh", b_f32, m, n,
                             3, 3);
    print_matrix_section_f32("Reference Matrix B (float) with Tanh", b_f32_ref,
                             m, n, 3, 3);

    // Example 3: Convert float to bfloat16, apply activation, and convert back
    printf("Example 3: Operations with BFloat16 data\n");

    // Convert float data to bfloat16
    convert_f32_to_bf16(a_f32, a_bf16, m * n);

    // Print the bfloat16 input matrix
    print_matrix_section_bf16("Input Matrix A (bfloat16)", a_bf16, m, n, 3, 3);

    // Set up post-op for ReLU
    gelu_post_ops->eltwise->algo.algo_type = RELU; // Change to ReLU activation

    // Apply ReLU using element-wise operation API with bfloat16 data
    aocl_gemm_eltwise_ops_bf16obf16('R', 'N', 'N', // Row-major, no transposes
                                    m, n, a_bf16, lda, b_bf16, ldb,
                                    gelu_post_ops);

    // Print result with ReLU in bfloat16
    print_matrix_section_bf16("Output Matrix B (bfloat16) with ReLU", b_bf16, m,
                              n, 3, 3);

    // Example 4: Convert bfloat16 to float during element-wise operation
    printf("Example 4: Converting BFloat16 to Float during operation\n");

    // Apply ReLU using element-wise operation API with bfloat16 input and float
    // output
    aocl_gemm_eltwise_ops_bf16of32('R', 'N', 'N', // Row-major, no transposes
                                   m, n, a_bf16, lda, b_f32, ldb,
                                   gelu_post_ops);

    // Print result
    print_matrix_section_f32("Output Matrix B (float from bfloat16) with ReLU",
                             b_f32, m, n, 3, 3);

    // Example 5: Demonstrate utility functions for single operations
    printf("Example 5: Using utility functions for vector operations\n");

    // Reset output matrices
    memcpy(b_f32, a_f32, m * n * sizeof(float));

    // Apply GeLU Tanh directly using the utility function
    aocl_gemm_gelu_tanh_f32(m * n, b_f32, 1);

    // Print result
    print_matrix_section_f32("Matrix after direct GeLU Tanh application", b_f32,
                             m, n, 3, 3);

    // Reset output matrices
    memcpy(b_f32, a_f32, m * n * sizeof(float));

    // Apply GeLU Erf directly using the utility function
    aocl_gemm_gelu_erf_f32(m * n, b_f32, 1);

    // Print result
    print_matrix_section_f32("Matrix after direct GeLU Erf application", b_f32,
                             m, n, 3, 3);

cleanup:
    // Free allocated memory
    free(a_f32);
    free(b_f32);
    free(b_f32_ref);
    free(a_bf16);
    free(b_bf16);

    // Free post-ops memory
    if (gelu_post_ops) {
        if (gelu_post_ops->seq_vector)
            free(gelu_post_ops->seq_vector);
        if (gelu_post_ops->eltwise)
            free(gelu_post_ops->eltwise);
        free(gelu_post_ops);
    }

    return 0;
}
