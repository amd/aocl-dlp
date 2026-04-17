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
 * Example: BF16×U4→F32 GEMM with asymmetric weight-only quantization (WOQ).
 *
 * B is packed (two u4 nibbles per byte), reordered with the same helper
 * as bf16s4; GEMM uses aocl_gemm_bf16u4f32of32 with mem_format_b 'R' and
 * metadata.pre_ops (b_scl, b_zp). Supported ZP types in the library include
 * DLP_S8 and DLP_BF16; this example uses s8 ZPs.
 */

#include "aocl_dlp.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

union bf16_to_float
{
    uint32_t u;
    float    f;
};

uint8_t
unpack_u4(const uint8_t* packed, md_t ldb, md_t row, md_t col)
{
    size_t linear = (size_t)row * (size_t)ldb + (size_t)col;
    size_t byte_i = linear / 2;
    int    shift  = (int)((linear % 2) * 4);
    return (uint8_t)((packed[byte_i] >> shift) & 0x0Fu);
}

void
pack_u4_at(uint8_t* packed, md_t ldb, md_t row, md_t col, uint8_t u4)
{
    size_t   linear = (size_t)row * (size_t)ldb + (size_t)col;
    size_t   bi     = linear / 2;
    int      sh     = (int)((linear % 2) * 4);
    uint8_t  nib    = (uint8_t)(u4 & 0x0Fu);
    uint8_t* p      = &packed[bi];
    *p              = (uint8_t)((*p & (uint8_t)~(0x0Fu << sh)) | (nib << sh));
}

void
init_bf16_matrix(bfloat16* matrix, int rows, int cols, int ld, float scale)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float float_val = scale * ((i + j + 1) % 100) / 100.0f;
            union bf16_to_float x;
            x.f                = float_val;
            uint16_t bf16_val  = (uint16_t)(x.u >> 16);
            matrix[i * ld + j] = bf16_val;
        }
    }
}

void
init_packed_u4_matrix(uint8_t* packed, md_t k, md_t n, md_t ldb)
{
    for (md_t i = 0; i < k; i++) {
        for (md_t j = 0; j < n; j++) {
            uint8_t v = (uint8_t)(((i + j) % 16));
            pack_u4_at(packed, ldb, i, j, v);
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
            union bf16_to_float x;
            x.u = ((uint32_t)matrix[i * ld + j]) << 16;
            printf("%8.4f ", x.f);
        }
        printf("\n");
    }
    printf("\n");
}

void
print_packed_u4_section(const char*    name,
                        const uint8_t* packed,
                        md_t           k,
                        md_t           n,
                        md_t           ldb,
                        int            max_rows,
                        int            max_cols)
{
    printf("%s (%d x %d) - showing top-left %d x %d:\n", name, (int)k, (int)n,
           max_rows, max_cols);
    for (int i = 0; i < max_rows && i < (int)k; i++) {
        for (int j = 0; j < max_cols && j < (int)n; j++) {
            printf("%4u ", (unsigned)unpack_u4(packed, ldb, (md_t)i, (md_t)j));
        }
        printf("\n");
    }
    printf("\n");
}

void
print_f32_matrix_section(const char* name,
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

int
main()
{
    printf("BF16×U4→F32 GEMM with asymmetric WOQ (B scale + B zero_point, "
           "DLP_S8 ZP)\n\n");

    md_t m = 128, n = 128, k = 128;
    md_t lda    = k;
    md_t ldc    = n;
    md_t ldb_in = n;

    dlp_metadata_t meta_reorder;
    memset(&meta_reorder, 0, sizeof(meta_reorder));

    // Get the size of the reorder buffer
    msz_t reorder_bytes = aocl_get_reorder_buf_size_bf16s4f32of32(
        'R', 'N', 'B', k, n, &meta_reorder);
    if (reorder_bytes == 0
        || meta_reorder.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        printf("Reorder buffer sizing failed (AVX512 BF16 required).\n");
        return -1;
    }

    // Get the size of the packed input matrix
    size_t packed_input_bytes = ((size_t)k * (size_t)ldb_in + 1) / 2;

    // Allocate memory for the matrices
    bfloat16* a           = (bfloat16*)malloc((size_t)lda * m * sizeof(*a));
    uint8_t*  b_packed    = (uint8_t*)malloc(packed_input_bytes);
    uint8_t*  b_reordered = (uint8_t*)malloc(reorder_bytes);
    float*    c           = (float*)malloc((size_t)ldc * m * sizeof(*c));

    if (!a || !b_packed || !b_reordered || !c) {
        printf("Memory allocation failed\n");
        free(a);
        free(b_packed);
        free(b_reordered);
        free(c);
        return -1;
    }

    // Initialize the matrices
    init_bf16_matrix(a, m, k, lda, 2.0f);
    memset(b_packed, 0, packed_input_bytes);
    init_packed_u4_matrix(b_packed, k, n, ldb_in);

    // Print the matrices
    print_bf16_matrix_section("Matrix A (BF16)", a, m, k, lda, 3, 3);
    print_packed_u4_section("Matrix B (U4 packed, logical view before reorder)",
                            b_packed, k, n, ldb_in, 3, 3);

    // Reorder the matrix
    aocl_reorder_bf16s4f32of32('R', 'N', 'B', (const int8_t*)b_packed,
                               (int8_t*)b_reordered, k, n, ldb_in,
                               &meta_reorder);
    if (meta_reorder.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        printf("aocl_reorder_bf16s4f32of32 failed, error_code=%d\n",
               (int)meta_reorder.error_hndl.error_code);
        free(a);
        free(b_packed);
        free(b_reordered);
        free(c);
        return -1;
    }
    printf("(B reordered via bf16s4 reorder API, same layout as U4; "
           "mem_format_b 'R')\n\n");

    // Initialize the result matrix
    memset(c, 0, (size_t)ldc * m * sizeof(float));
    print_f32_matrix_section("Initial Matrix C (F32)", c, m, n, ldc, 3, 3);

    md_t           ldb_gemm = n;
    dlp_metadata_t metadata;

    // =======================================================================
    // Example 1: per-tensor B scale and per-tensor ZP (len = 1)
    // =======================================================================
    printf("\n--- Example 1: WOQ per-tensor B scale and ZP ---\n\n");

    // Initialize the per-tensor B scale and ZP
    float  b_scale_tensor = 0.125f;
    int8_t b_zp_tensor    = 8;

    // Initialize the per-tensor B scale and ZP metadata
    dlp_sf_t   b_scl_tensor  = { .scale_factor      = &b_scale_tensor,
                                 .scale_factor_len  = 1,
                                 .scale_factor_type = DLP_F32 };
    dlp_zp_t   b_zp_t_tensor = { .zero_point      = &b_zp_tensor,
                                 .zero_point_len  = 1,
                                 .zero_point_type = DLP_S8 };
    dlp_pre_op pre_tensor    = { .b_zp       = &b_zp_t_tensor,
                                 .b_scl      = &b_scl_tensor,
                                 .seq_length = 1,
                                 .group_size = 0 };

    printf("B WOQ: scale_factor_len=1, value=%.6f; zero_point_len=1 (s8), "
           "value=%d\n\n",
           b_scale_tensor, (int)b_zp_tensor);

    // Initialize the metadata
    memset(&metadata, 0, sizeof(metadata));
    metadata.pre_ops = &pre_tensor;
    memset(c, 0, (size_t)ldc * (size_t)m * sizeof(float));

    // Run the GEMM
    aocl_gemm_bf16u4f32of32('R', 'N', 'N', m, n, k, 1.0f, a, lda, 'N',
                            b_reordered, ldb_gemm, 'R', 0.0f, c, ldc,
                            &metadata);
    if (metadata.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        printf("GEMM failed, error_code=%d\n",
               (int)metadata.error_hndl.error_code);
        goto cleanup;
    }

    print_f32_matrix_section("Result Matrix C (F32)", c, m, n, ldc, 3, 3);

    // =======================================================================
    // Example 2: per-channel B scale and ZP (len = n)
    // =======================================================================
    printf("\n--- Example 2: WOQ per-channel B scale and ZP (len=n) ---\n\n");

    float*  b_scale_ch = (float*)malloc((size_t)n * sizeof(float));
    int8_t* b_zp_ch    = (int8_t*)malloc((size_t)n * sizeof(int8_t));
    if (!b_scale_ch || !b_zp_ch) {
        printf("Channel WOQ allocation failed\n");
        free(b_scale_ch);
        free(b_zp_ch);
        goto cleanup;
    }
    // Initialize the per-channel B scale and ZP
    for (md_t j = 0; j < n; j++) {
        b_scale_ch[j] = 0.08f + 0.0005f * (float)j;
        b_zp_ch[j]    = (int8_t)(7 + (j % 3));
    }

    // Initialize the per-channel B scale and ZP metadata
    dlp_sf_t   b_scl_ch  = { .scale_factor      = b_scale_ch,
                             .scale_factor_len  = n,
                             .scale_factor_type = DLP_F32 };
    dlp_zp_t   b_zp_t_ch = { .zero_point      = b_zp_ch,
                             .zero_point_len  = n,
                             .zero_point_type = DLP_S8 };
    dlp_pre_op pre_ch    = { .b_zp       = &b_zp_t_ch,
                             .b_scl      = &b_scl_ch,
                             .seq_length = 1,
                             .group_size = 0 };

    printf("B WOQ: scale_factor_len=%ld, first 3: %.6f, %.6f, %.6f\n", (long)n,
           b_scale_ch[0], b_scale_ch[1], b_scale_ch[2]);
    printf("       zero_point_len=%ld (s8), first 3: %d, %d, %d\n", (long)n,
           (int)b_zp_ch[0], (int)b_zp_ch[1], (int)b_zp_ch[2]);
    printf("  ... (scales and ZPs for all %ld columns)\n\n", (long)n);

    // Initialize the metadata
    memset(&metadata, 0, sizeof(metadata));
    metadata.pre_ops = &pre_ch;
    memset(c, 0, (size_t)ldc * (size_t)m * sizeof(float));

    // Run the GEMM
    aocl_gemm_bf16u4f32of32('R', 'N', 'N', m, n, k, 1.0f, a, lda, 'N',
                            b_reordered, ldb_gemm, 'R', 0.0f, c, ldc,
                            &metadata);
    if (metadata.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        printf("GEMM failed, error_code=%d\n",
               (int)metadata.error_hndl.error_code);
        free(b_scale_ch);
        free(b_zp_ch);
        goto cleanup;
    }

    print_f32_matrix_section("Result Matrix C (F32)", c, m, n, ldc, 3, 3);

    free(b_scale_ch);
    free(b_zp_ch);

cleanup:
    free(a);
    free(b_packed);
    free(b_reordered);
    free(c);
    return 0;
}
