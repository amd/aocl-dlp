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
 * Example: BF16×S8→BF16 GEMM with on-the-fly quantization
 * Demonstrates symmetric (per-tensor) and asymmetric (per-row) quantization
 */

#include "aocl_dlp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void
init_bf16_matrix(bfloat16* matrix, int rows, int cols, int ld, float scale)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float     float_val  = scale * ((i + j + 1) % 100) / 100.0f;
            uint32_t* float_bits = (uint32_t*)&float_val;
            uint16_t  bf16_val   = (*float_bits >> 16);
            matrix[i * ld + j]   = bf16_val;
        }
    }
}

void
init_s8_matrix(int8_t* matrix, int rows, int cols, int ld, int base_value)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * ld + j] = (int8_t)((base_value * (i + j) % 255) - 127);
        }
    }
}

void
print_bf16_matrix_section(const char* name,
                          bfloat16*   matrix,
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
            uint32_t float_bits = ((uint32_t)matrix[i * ld + j]) << 16;
            float    float_val;
            memcpy(&float_val, &float_bits, sizeof(float));
            printf("%8.4f ", float_val);
        }
        printf("\n");
    }
    printf("\n");
}

void
print_s8_matrix_section(const char* name,
                        int8_t*     matrix,
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
            printf("%4d ", matrix[i * ld + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void
compute_per_row_quant_params(bfloat16* matrix,
                             md_t      rows,
                             md_t      cols,
                             md_t      ld,
                             float*    a_pre_quant_sf,
                             float*    a_post_quant_sf,
                             float*    zero_points)
{
    printf("Computing per-row quantization parameters:\n");

    for (md_t i = 0; i < rows; i++) {
        float min_val = INFINITY, max_val = -INFINITY;
        for (md_t j = 0; j < cols; j++) {
            uint32_t float_bits = ((uint32_t)matrix[i * ld + j]) << 16;
            float    val;
            memcpy(&val, &float_bits, sizeof(float));
            if (val < min_val)
                min_val = val;
            if (val > max_val)
                max_val = val;
        }

        float range = max_val - min_val;
        if (range < 1e-6f)
            range = 1e-6f;

        a_pre_quant_sf[i]  = 255.0f / range;
        a_post_quant_sf[i] = range / 255.0f;
        zero_points[i]     = -min_val * a_pre_quant_sf[i] - 128.0f;

        if (i < 3) {
            printf("  Row %3ld: min=%7.4f, max=%7.4f, a_pre_quant_sf=%7.6f, "
                   "a_post_quant_sf=%7.6f, zp=%7.4f\n",
                   (long)i, min_val, max_val, a_pre_quant_sf[i],
                   a_post_quant_sf[i], zero_points[i]);
        }
    }
    printf("  ... (computed for all %ld rows)\n\n", (long)rows);
}

int
main()
{
    // Matrix dimensions
    md_t m = 128, n = 128, k = 128;
    // Leading dimensions (assuming row-major storage)
    md_t lda = k, ldb = n, ldc = n;

    // Allocate memory for matrices
    bfloat16* a = (bfloat16*)malloc(lda * m * sizeof(bfloat16));
    int8_t*   b = (int8_t*)malloc(ldb * k * sizeof(int8_t));
    bfloat16* c = (bfloat16*)malloc(ldc * m * sizeof(bfloat16));

    // Initialize pointers for cleanup
    DLP_POST_OP_TYPE* seq_vector_1        = NULL;
    DLP_POST_OP_TYPE* seq_vector_2        = NULL;
    float*            a_pre_quant_sf_row  = NULL;
    float*            a_post_quant_sf_row = NULL;
    float*            zero_points         = NULL;

    if (!a || !b || !c) {
        printf("Memory allocation failed\n");
        return -1;
    }

    init_bf16_matrix(a, m, k, lda, 2.0f);
    init_s8_matrix(b, k, n, ldb, 5);
    memset(c, 0, ldc * m * sizeof(bfloat16));

    print_bf16_matrix_section("Matrix A (BF16)", a, m, k, lda, 3, 3);
    print_s8_matrix_section("Matrix B (S8)", b, k, n, ldb, 3, 3);
    print_bf16_matrix_section("Initial Matrix C (BF16)", c, m, n, ldc, 3, 3);

    // =======================================================================
    // Example 1: Symmetric Quantization (Per-Tensor)
    // =======================================================================
    printf("\n--- Example 1: Symmetric Quantization (Per-Tensor) ---\n\n");

    float a_pre_quant_sf  = 127.0f / 2.0f;
    float a_post_quant_sf = 2.0f / 127.0f;

    printf("Scale factors: pre-quantization=%.6f, post-quantization=%.6f\n\n",
           a_pre_quant_sf, a_post_quant_sf);

    dlp_sf_t a_pre_quant_scl  = { .scale_factor      = &a_pre_quant_sf,
                                  .scale_factor_len  = 1,
                                  .scale_factor_type = DLP_F32 };
    dlp_sf_t a_post_quant_scl = { .scale_factor      = &a_post_quant_sf,
                                  .scale_factor_len  = 1,
                                  .scale_factor_type = DLP_F32 };

    dlp_quant_op a_pre_quant = { .group_size = 0,
                                 .src_type   = DLP_BF16,
                                 .dst_type   = DLP_S8,
                                 .scl        = &a_pre_quant_scl,
                                 .zp         = NULL,
                                 .symmetric  = true };

    dlp_quant_op a_post_quant = { .group_size = 0,
                                  .src_type   = DLP_BF16,
                                  .dst_type   = DLP_S8,
                                  .scl        = &a_post_quant_scl,
                                  .zp         = NULL,
                                  .symmetric  = true };

    // Allocate and set up metadata with sequence vector
    dlp_metadata_t metadata = { 0 };
    metadata.a_pre_quant    = &a_pre_quant;
    metadata.a_post_quant   = &a_post_quant;
    metadata.seq_length     = 1;

    // Allocate sequence vector for post-quantization operation
    seq_vector_1 = (DLP_POST_OP_TYPE*)malloc(sizeof(DLP_POST_OP_TYPE));
    if (!seq_vector_1) {
        printf("Memory allocation for sequence vector failed\n");
        goto cleanup;
    }
    seq_vector_1[0]     = ADQUANTIZE;
    metadata.seq_vector = seq_vector_1;

    aocl_gemm_bf16s8s32obf16('R', 'N', 'N', m, n, k, 1.0f, a, lda, 'N', b, ldb,
                             'N', 0.0f, c, ldc, &metadata);

    print_bf16_matrix_section("Result Matrix C", c, m, n, ldc, 3, 3);

    // =======================================================================
    // Example 2: Asymmetric Quantization (Per-Row)
    // =======================================================================
    printf("\n--- Example 2: Asymmetric Quantization (Per-Row) ---\n\n");

    memset(c, 0, ldc * m * sizeof(bfloat16));

    a_pre_quant_sf_row  = (float*)malloc(m * sizeof(float));
    a_post_quant_sf_row = (float*)malloc(m * sizeof(float));
    zero_points         = (float*)malloc(m * sizeof(float));

    if (!a_pre_quant_sf_row || !a_post_quant_sf_row || !zero_points) {
        printf("Memory allocation for quantization parameters failed\n");
        goto cleanup;
    }

    compute_per_row_quant_params(a, m, k, lda, a_pre_quant_sf_row,
                                 a_post_quant_sf_row, zero_points);

    dlp_sf_t a_pre_quant_scl_row  = { .scale_factor      = a_pre_quant_sf_row,
                                      .scale_factor_len  = m,
                                      .scale_factor_type = DLP_F32 };
    dlp_sf_t a_post_quant_scl_row = { .scale_factor      = a_post_quant_sf_row,
                                      .scale_factor_len  = m,
                                      .scale_factor_type = DLP_F32 };
    dlp_zp_t zp_row               = { .zero_point      = zero_points,
                                      .zero_point_len  = m,
                                      .zero_point_type = DLP_F32 };

    dlp_quant_op a_pre_quant_row = { .group_size = 0,
                                     .src_type   = DLP_BF16,
                                     .dst_type   = DLP_S8,
                                     .scl        = &a_pre_quant_scl_row,
                                     .zp         = &zp_row,
                                     .symmetric  = false };

    dlp_quant_op a_post_quant_row = { .group_size = 0,
                                      .src_type   = DLP_BF16,
                                      .dst_type   = DLP_S8,
                                      .scl        = &a_post_quant_scl_row,
                                      .zp         = &zp_row,
                                      .symmetric  = false };

    // Allocate and set up metadata with sequence vector
    dlp_metadata_t metadata_row = { 0 };
    metadata_row.a_pre_quant    = &a_pre_quant_row;
    metadata_row.a_post_quant   = &a_post_quant_row;
    metadata_row.b_pre_quant    = NULL;
    metadata_row.b_post_quant   = NULL;
    metadata_row.seq_length     = 1;

    // Allocate sequence vector for post-quantization operation
    seq_vector_2 = (DLP_POST_OP_TYPE*)malloc(sizeof(DLP_POST_OP_TYPE));
    if (!seq_vector_2) {
        printf("Memory allocation for sequence vector failed\n");
        goto cleanup;
    }
    seq_vector_2[0]         = ADQUANTIZE;
    metadata_row.seq_vector = seq_vector_2;

    aocl_gemm_bf16s8s32obf16('R', 'N', 'N', m, n, k, 1.0f, a, lda, 'N', b, ldb,
                             'N', 0.0f, c, ldc, &metadata_row);

    print_bf16_matrix_section("Result Matrix C", c, m, n, ldc, 3, 3);

cleanup:
    // Free all allocated resources
    free(seq_vector_1);
    free(seq_vector_2);
    free(a_pre_quant_sf_row);
    free(a_post_quant_sf_row);
    free(zero_points);
    free(a);
    free(b);
    free(c);

    return 0;
}
