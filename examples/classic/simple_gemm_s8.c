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
 * Example demonstrating the usage of AOCL-DLP's 8-bit integer GEMM operation
 *
 * This example shows how to:
 * 1. Initialize 8-bit integer matrices
 * 2. Perform matrix multiplication with s8 input and s32 output
 * 3. Check hardware compatibility for AVX512-VNNI instructions
 */

#include "aocl_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Utility function to initialize a signed 8-bit integer matrix
void
init_s8_matrix(int8_t* matrix, int rows, int cols, int base_value)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Create a value between -127 and 127
            matrix[i * cols + j] = (int8_t)((base_value * (i + j) % 255) - 127);
        }
    }
}

// Utility function to initialize an unsigned 8-bit integer matrix
void
init_u8_matrix(uint8_t* matrix, int rows, int cols, int base_value)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Create a value between 0 and 255
            matrix[i * cols + j] = (uint8_t)((base_value * (i + j) % 256));
        }
    }
}

// Utility function to print a small section of a signed 8-bit integer matrix
void
print_s8_matrix_section(const char* name,
                        int8_t*     matrix,
                        int         rows,
                        int         cols,
                        int         max_rows,
                        int         max_cols)
{
    printf("%s (%d x %d) - showing top-left %d x %d:\n", name, rows, cols,
           max_rows, max_cols);
    for (int i = 0; i < max_rows && i < rows; i++) {
        for (int j = 0; j < max_cols && j < cols; j++) {
            printf("%4d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Utility function to print a small section of an unsigned 8-bit integer matrix
void
print_u8_matrix_section(const char* name,
                        uint8_t*    matrix,
                        int         rows,
                        int         cols,
                        int         max_rows,
                        int         max_cols)
{
    printf("%s (%d x %d) - showing top-left %d x %d:\n", name, rows, cols,
           max_rows, max_cols);
    for (int i = 0; i < max_rows && i < rows; i++) {
        for (int j = 0; j < max_cols && j < cols; j++) {
            printf("%4d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Utility function to print a small section of a signed 32-bit integer matrix
void
print_s32_matrix_section(const char* name,
                         int32_t*    matrix,
                         int         rows,
                         int         cols,
                         int         max_rows,
                         int         max_cols)
{
    printf("%s (%d x %d) - showing top-left %d x %d:\n", name, rows, cols,
           max_rows, max_cols);
    for (int i = 0; i < max_rows && i < rows; i++) {
        for (int j = 0; j < max_cols && j < cols; j++) {
            printf("%8d ", matrix[i * cols + j]);
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
    uint8_t* a = (uint8_t*)malloc(m * k * sizeof(uint8_t));
    int8_t*  b = (int8_t*)malloc(k * n * sizeof(int8_t));
    int32_t* c = (int32_t*)malloc(m * n * sizeof(int32_t));

    if (!a || !b || !c) {
        printf("Memory allocation failed\n");
        return -1;
    }

    // Initialize matrices with some values
    init_u8_matrix(a, m, k, 5);            // Initialize A with uint8 values
    init_s8_matrix(b, k, n, 3);            // Initialize B with int8 values
    memset(c, 0, m * n * sizeof(int32_t)); // Initialize C with zeros

    // Print a small section of the input matrices
    print_u8_matrix_section("Matrix A (U8)", a, m, k, 3, 3);
    print_s8_matrix_section("Matrix B (S8)", b, k, n, 3, 3);
    print_s32_matrix_section("Initial Matrix C (S32)", c, m, n, 3, 3);

    // GEMM parameters
    int32_t alpha        = 1;   // Scalar for A*B
    int32_t beta         = 0;   // Scalar for C
    char    order        = 'R'; // Row-major storage
    char    transa       = 'N'; // No transpose for A
    char    transb       = 'N'; // No transpose for B
    char    mem_format_a = 'N'; // A is not reordered
    char    mem_format_b = 'N'; // B is not reordered

    // Perform matrix multiplication with U8/S8 inputs and S32 output: C = alpha
    // * A * B + beta * C
    aocl_gemm_u8s8s32os32(order, transa, transb, m, n, k, alpha, a, lda,
                          mem_format_a, b, ldb, mem_format_b, beta, c, ldc,
                          NULL // No post-operations
    );

    // Print a small section of the output matrix
    print_s32_matrix_section("Result Matrix C (S32)", c, m, n, 3, 3);

    // Example 2: Using signed 8-bit for both input matrices
    printf("\n--- Example with signed 8-bit inputs (S8S8) ---\n\n");

    // Allocate memory for a signed 8-bit A matrix
    int8_t* a_s8 = (int8_t*)malloc(m * k * sizeof(int8_t));
    if (!a_s8) {
        printf("Memory allocation failed\n");
        goto cleanup;
    }

    // Initialize the signed 8-bit A matrix
    init_s8_matrix(a_s8, m, k, 7);

    // Reset the output matrix C
    memset(c, 0, m * n * sizeof(int32_t));

    // Print a small section of the input matrices
    print_s8_matrix_section("Matrix A (S8)", a_s8, m, k, 3, 3);
    print_s8_matrix_section("Matrix B (S8)", b, k, n, 3, 3);
    print_s32_matrix_section("Initial Matrix C (S32)", c, m, n, 3, 3);

    // Perform matrix multiplication with S8/S8 inputs and S32 output: C = alpha
    // * A * B + beta * C
    aocl_gemm_s8s8s32os32(order, transa, transb, m, n, k, alpha, a_s8, lda,
                          mem_format_a, b, ldb, mem_format_b, beta, c, ldc,
                          NULL // No post-operations
    );

    // Print a small section of the output matrix
    print_s32_matrix_section("Result Matrix C (S32)", c, m, n, 3, 3);

    // Free the additional matrix
    free(a_s8);

cleanup:
    // Free allocated memory
    free(a);
    free(b);
    free(c);

    return 0;
}
