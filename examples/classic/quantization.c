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
 * Example demonstrating quantized matrix multiplication
 *
 * This example shows how to:
 * 1. Quantize floating-point data to 8-bit integers
 * 2. Perform matrix multiplication with quantized data
 * 3. Apply scaling and zero-point adjustments
 * 4. Dequantize results back to floating-point
 */

#include "aocl_dlp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Utility function to find the min/max values in a float matrix
void
find_min_max(const float* matrix, int size, float* min_val, float* max_val)
{
    *min_val = matrix[0];
    *max_val = matrix[0];

    for (int i = 1; i < size; i++) {
        if (matrix[i] < *min_val) {
            *min_val = matrix[i];
        }
        if (matrix[i] > *max_val) {
            *max_val = matrix[i];
        }
    }
}

// Utility function to quantize float data to uint8_t
void
quantize_f32_to_u8(
    const float* src, uint8_t* dst, int size, float scale, uint8_t zero_point)
{
    for (int i = 0; i < size; i++) {
        float value = src[i] / scale + zero_point;
        // Clamp to [0, 255]
        if (value < 0)
            value = 0;
        if (value > 255)
            value = 255;
        dst[i] = (uint8_t)round(value);
    }
}

// Utility function to quantize float data to int8_t
void
quantize_f32_to_s8(
    const float* src, int8_t* dst, int size, float scale, int8_t zero_point)
{
    for (int i = 0; i < size; i++) {
        float value = src[i] / scale + zero_point;
        // Clamp to [-128, 127]
        if (value < -128)
            value = -128;
        if (value > 127)
            value = 127;
        dst[i] = (int8_t)round(value);
    }
}

// Utility function to dequantize uint8_t data to float
void
dequantize_u8_to_f32(
    const uint8_t* src, float* dst, int size, float scale, uint8_t zero_point)
{
    for (int i = 0; i < size; i++) {
        dst[i] = (float)(src[i] - zero_point) * scale;
    }
}

// Utility function to dequantize int8_t data to float
void
dequantize_s8_to_f32(
    const int8_t* src, float* dst, int size, float scale, int8_t zero_point)
{
    for (int i = 0; i < size; i++) {
        dst[i] = (float)(src[i] - zero_point) * scale;
    }
}

// Utility function to initialize a matrix with values
void
init_matrix(float* matrix, int rows, int cols, float value)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Generate values between -1 and 1
            float sign           = ((i + j) % 2 == 0) ? 1.0f : -1.0f;
            matrix[i * cols + j] = sign * value * (i + j + 1) / (rows * cols);
        }
    }
}

// Utility function to print a small section of a matrix
void
print_matrix_section(const char*  name,
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

// Utility function to print a small section of a uint8_t matrix
void
print_u8_matrix_section(const char*    name,
                        const uint8_t* matrix,
                        int            rows,
                        int            cols,
                        int            max_rows,
                        int            max_cols)
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

// Utility function to print a small section of a int8_t matrix
void
print_s8_matrix_section(const char*   name,
                        const int8_t* matrix,
                        int           rows,
                        int           cols,
                        int           max_rows,
                        int           max_cols)
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

// Utility function to perform reference floating-point matrix multiplication
void
reference_gemm(const float* a, const float* b, float* c, int m, int n, int k)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

int
main()
{
    printf("Quantized Matrix Multiplication Example\n\n");

    // Check if hardware supports int8 operations
    if (!dlp_aocl_enable_instruction_query()) {
        printf("Warning: This CPU may not fully support 8-bit integer "
               "operations.\n");
        printf("The library will use a slower fallback implementation.\n\n");
    } else {
        printf("Hardware support for 8-bit integer operations detected.\n\n");
    }

    // Matrix dimensions
    md_t m = 64; // Rows of A and C
    md_t n = 64; // Columns of B and C
    md_t k = 64; // Columns of A and rows of B

    // Leading dimensions (assuming row-major storage)
    md_t lda = k;
    md_t ldb = n;
    md_t ldc = n;

    // Allocate memory for float matrices (for reference calculation and initial
    // data)
    float* a_f32     = (float*)calloc(lda * m, sizeof(float));
    float* b_f32     = (float*)calloc(ldb * k, sizeof(float));
    float* c_ref     = (float*)calloc(ldc * m, sizeof(float));
    float* c_dequant = (float*)calloc(ldc * m, sizeof(float));

    // Allocate memory for quantized matrices
    uint8_t* a_u8  = (uint8_t*)calloc(lda * m, sizeof(uint8_t));
    int8_t*  b_s8  = (int8_t*)calloc(ldb * k, sizeof(int8_t));
    int32_t* c_s32 = (int32_t*)calloc(ldc * m, sizeof(int32_t));
    int8_t*  c_s8  = (int8_t*)calloc(ldc * m, sizeof(int8_t));

    if (!a_f32 || !b_f32 || !c_ref || !c_dequant || !a_u8 || !b_s8 || !c_s32
        || !c_s8) {
        printf("Memory allocation failed\n");
        return -1;
    }

    // Initialize float matrices with random values
    printf("Initializing matrices...\n");
    init_matrix(a_f32, m, k, 2.0f); // Values roughly in [-2, 2]
    init_matrix(b_f32, k, n, 1.0f); // Values roughly in [-1, 1]
    memset(c_ref, 0, m * n * sizeof(float));
    memset(c_dequant, 0, m * n * sizeof(float));
    memset(c_s32, 0, m * n * sizeof(int32_t));
    memset(c_s8, 0, m * n * sizeof(int8_t));

    // Print a small section of the input float matrices
    print_matrix_section("A (float)", a_f32, m, k, 3, 3);
    print_matrix_section("B (float)", b_f32, k, n, 3, 3);

    // Step 1: Compute reference result using floating-point
    printf("Computing reference result with floating-point...\n");
    reference_gemm(a_f32, b_f32, c_ref, m, n, k);
    print_matrix_section("C Reference (float)", c_ref, m, n, 3, 3);

    // Step 2: Determine quantization parameters for A (uint8_t)
    float a_min, a_max;
    find_min_max(a_f32, m * k, &a_min, &a_max);
    float a_range = fmaxf(fabsf(a_min), fabsf(a_max));
    float a_scale = a_range / 127.0f; // Scale = max(|min|, |max|) / 127, to map
                                      // the range to int8 [-128, 127]
    uint8_t a_zero_point = 128;       // Zero point for u8

    printf("A quantization parameters: scale = %f, zero_point = %d\n", a_scale,
           a_zero_point);

    // Step 3: Determine quantization parameters for B (int8_t)
    float b_min, b_max;
    find_min_max(b_f32, k * n, &b_min, &b_max);
    float  b_range      = fmaxf(fabsf(b_min), fabsf(b_max));
    float  b_scale      = b_range / 127.0f; // Scale to use full int8 range
    int8_t b_zero_point = 0; // Zero point for s8 (symmetric quantization)

    printf("B quantization parameters: scale = %f, zero_point = %d\n", b_scale,
           b_zero_point);

    // Step 4: Quantize input matrices
    quantize_f32_to_u8(a_f32, a_u8, m * k, a_scale, a_zero_point);
    quantize_f32_to_s8(b_f32, b_s8, k * n, b_scale, b_zero_point);

    // Print a small section of the quantized matrices
    print_u8_matrix_section("A (uint8)", a_u8, m, k, 3, 3);
    print_s8_matrix_section("B (int8)", b_s8, k, n, 3, 3);

    // Step 5: Determine output quantization parameters
    // To simplify, we'll use the product of input scales for the output scale
    float  c_scale      = a_scale * b_scale;
    int8_t c_zero_point = 0; // Zero point for output

    printf("C quantization parameters: scale = %f, zero_point = %d\n", c_scale,
           c_zero_point);

    // Step 6: Set up post-op for output quantization
    dlp_metadata_t* metadata =
        (dlp_metadata_t*)calloc(1, sizeof(dlp_metadata_t));
    if (!metadata) {
        printf("Memory allocation for post-ops failed\n");
        goto cleanup;
    }

    // Initialize post-ops structure
    metadata->seq_length = 1; // One operation: scale

    // Allocate sequence vector
    metadata->seq_vector =
        (DLP_POST_OP_TYPE*)calloc(1, sizeof(DLP_POST_OP_TYPE));
    if (!metadata->seq_vector) {
        printf("Memory allocation for sequence vector failed\n");
        goto cleanup;
    }
    metadata->seq_vector[0] = SCALE; // First operation is scaling

    // Allocate and set up sum post-op for scaling
    metadata->scale = (dlp_scale_t*)calloc(1, sizeof(dlp_scale_t));
    if (!metadata->scale) {
        printf("Memory allocation for sum post-op failed\n");
        goto cleanup;
    }

    // Allocate and set up scale factor structure
    metadata->scale->sf = calloc(1, sizeof(dlp_sf_t));
    if (!metadata->scale->sf) {
        printf("Memory allocation for scale factor structure failed\n");
        goto cleanup;
    }

    metadata->scale->sf->scale_factor = calloc(1, sizeof(float));
    if (!metadata->scale->sf->scale_factor) {
        printf("Memory allocation for scale factor data failed\n");
        goto cleanup;
    }

    // Set scalefactor for C to properly quantize the result.
    *((float*)metadata->scale->sf->scale_factor) = 1.0f / c_scale;
    metadata->scale->sf->scale_factor_len        = 1;
    metadata->scale->sf->scale_factor_type       = DLP_F32;
    metadata->scale->sf->scale_factor_dim        = DLP_PARAM_DIM_PER_TENSOR;

    // Allocate and set up zero point structure
    metadata->scale->zp = calloc(1, sizeof(dlp_zp_t));
    if (!metadata->scale->zp) {
        printf("Memory allocation for zero point structure failed\n");
        goto cleanup;
    }

    metadata->scale->zp->zero_point = calloc(1, sizeof(int8_t));
    if (!metadata->scale->zp->zero_point) {
        printf("Memory allocation for zero point data failed\n");
        goto cleanup;
    }

    *((int8_t*)metadata->scale->zp->zero_point) = c_zero_point;
    metadata->scale->zp->zero_point_len         = 1;
    metadata->scale->zp->zero_point_type        = DLP_S8;

    // Step 7: Perform quantized matrix multiplication with output as s8
    printf("Performing quantized matrix multiplication...\n");

    int32_t alpha = 1;
    int32_t beta  = 0;

    aocl_gemm_u8s8s32os8('R', 'N', 'N', // Row-major, no transposes
                         m, n, k, alpha, a_u8, lda, 'N', // Input A (uint8_t)
                         b_s8, ldb, 'N',                 // Input B (int8_t)
                         beta, c_s8, ldc,                // Output C (int8_t)
                         metadata // Scaling post-operation
    );

    // Print a small section of the quantized output
    print_s8_matrix_section("C Quantized (int8)", c_s8, m, n, 3, 3);

    // Step 8: Dequantize the result for comparison
    dequantize_s8_to_f32(c_s8, c_dequant, m * n, c_scale, c_zero_point);
    print_matrix_section("C Dequantized (float)", c_dequant, m, n, 3, 3);

    // Step 9: Compare the reference and dequantized results
    float max_diff = 0.0f;
    float avg_diff = 0.0f;

    for (int i = 0; i < m * n; i++) {
        float diff = fabsf(c_ref[i] - c_dequant[i]);
        avg_diff += diff;
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    avg_diff /= (m * n);

    printf("Comparison of results:\n");
    printf("  Maximum absolute difference: %f\n", max_diff);
    printf("  Average absolute difference: %f\n", avg_diff);

    // Compute relative error metrics
    float avg_rel_error = 0.0f;
    int   count         = 0;

    for (int i = 0; i < m * n; i++) {
        if (fabsf(c_ref[i]) > 1e-6) { // Avoid division by very small numbers
            float rel_error = fabsf(c_ref[i] - c_dequant[i]) / fabsf(c_ref[i]);
            avg_rel_error += rel_error;
            count++;
        }
    }

    if (count > 0) {
        avg_rel_error /= count;
        printf("  Average relative error: %f\n", avg_rel_error);
    }

cleanup:
    // Free allocated memory
    free(a_f32);
    free(b_f32);
    free(c_ref);
    free(c_dequant);
    free(a_u8);
    free(b_s8);
    free(c_s32);
    free(c_s8);

    // Free post-ops memory
    if (metadata) {
        if (metadata->seq_vector)
            free(metadata->seq_vector);
        if (metadata->scale) {
            if (metadata->scale->sf) {
                if (metadata->scale->sf->scale_factor)
                    free(metadata->scale->sf->scale_factor);
                free(metadata->scale->sf);
            }
            if (metadata->scale->zp) {
                if (metadata->scale->zp->zero_point)
                    free(metadata->scale->zp->zero_point);
                free(metadata->scale->zp);
            }
            free(metadata->scale);
        }
        free(metadata);
    }

    return 0;
}
