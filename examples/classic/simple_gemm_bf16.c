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
 * Example demonstrating the usage of AOCL-DLP's BFloat16 GEMM operation
 *
 * This example shows how to:
 * 1. Initialize BFloat16 matrices
 * 2. Perform matrix multiplication with bf16 input and f32 output
 * 3. Check hardware compatibility for AVX512-BF16 instructions
 */

#include "aocl_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Utility function to initialize a BFloat16 matrix from float values
void
init_bf16_matrix(bfloat16* matrix, int rows, int cols, float value)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Create a float value and convert it to bfloat16
            float float_val = value * (i + j + 1) / (rows * cols);

            // Convert float to bfloat16 using bit manipulation
            uint32_t* float_bits = (uint32_t*)&float_val;
            uint16_t  bf16_val   = (*float_bits >> 16);

            // Store the bfloat16 value
            matrix[i * cols + j] = bf16_val;
        }
    }
}

// Utility function to print a small section of a BFloat16 matrix
void
print_bf16_matrix_section(const char* name,
                          bfloat16*   matrix,
                          int         rows,
                          int         cols,
                          int         max_rows,
                          int         max_cols)
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

// Utility function to print a small section of a float matrix
void
print_float_matrix_section(const char* name,
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

int
main()
{
    // Matrix dimensions
    md_t m = 128; // Rows of A and C
    md_t n = 128; // Columns of B and C
    md_t k = 128; // Columns of A and rows of B

    // Leading dimensions (assuming row-major storage)
    md_t lda = k;
    md_t ldb = n;
    md_t ldc = n;

    // Allocate memory for matrices
    bfloat16* a = (bfloat16*)malloc(m * k * sizeof(bfloat16));
    bfloat16* b = (bfloat16*)malloc(k * n * sizeof(bfloat16));
    float*    c = (float*)malloc(m * n * sizeof(float));

    if (!a || !b || !c) {
        printf("Memory allocation failed\n");
        return -1;
    }

    // Initialize matrices with some values
    init_bf16_matrix(a, m, k, 1.0f);
    init_bf16_matrix(b, k, n, 0.5f);
    memset(c, 0, m * n * sizeof(float)); // Initialize C with zeros

    // Print a small section of the input matrices
    print_bf16_matrix_section("Matrix A (BF16)", a, m, k, 3, 3);
    print_bf16_matrix_section("Matrix B (BF16)", b, k, n, 3, 3);
    print_float_matrix_section("Initial Matrix C (F32)", c, m, n, 3, 3);

    // GEMM parameters
    float alpha        = 1.0f; // Scalar for A*B
    float beta         = 0.0f; // Scalar for C
    char  order        = 'R';  // Row-major storage
    char  transa       = 'N';  // No transpose for A
    char  transb       = 'N';  // No transpose for B
    char  mem_format_a = 'N';  // A is not reordered
    char  mem_format_b = 'N';  // B is not reordered

    // Perform matrix multiplication with BF16 inputs and F32 output: C = alpha
    // * A * B + beta * C
    aocl_gemm_bf16bf16f32of32(order, transa, transb, m, n, k, alpha, a, lda,
                              mem_format_a, b, ldb, mem_format_b, beta, c, ldc,
                              NULL // No post-operations
    );

    // Print a small section of the output matrix
    print_float_matrix_section("Result Matrix C (F32)", c, m, n, 3, 3);

    // Free allocated memory
    free(a);
    free(b);
    free(c);

    return 0;
}
