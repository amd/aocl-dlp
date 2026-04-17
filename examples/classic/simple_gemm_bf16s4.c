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
 * Example: BF16×S4→F32 GEMM with symmetric weight-only quantization (WOQ).
 *
 * B is packed (two s4 nibbles per byte), reordered for the kernel, with scales
 * in metadata.pre_ops->b_scl (per-tensor or per-channel on N).
 */

#include "aocl_dlp.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

union bf16_to_float
{
    uint32_t u;
    float    f;
};

int8_t
unpack_s4(const int8_t* packed, md_t ldb, md_t row, md_t col)
{
    size_t  linear = (size_t)row * (size_t)ldb + (size_t)col;
    size_t  byte_i = linear / 2;
    int     shift  = (int)((linear % 2) * 4);
    uint8_t bits4  = ((uint8_t)packed[byte_i] >> shift) & 0x0Fu;
    if (bits4 & 0x08u)
        return (int8_t)(bits4 | 0xF0u);
    return (int8_t)bits4;
}

void
pack_s4_at(int8_t* packed, md_t ldb, md_t row, md_t col, int8_t s4)
{
    size_t   linear = (size_t)row * (size_t)ldb + (size_t)col;
    size_t   bi     = linear / 2;
    int      sh     = (int)((linear % 2) * 4);
    uint8_t  nib    = (uint8_t)(((int)s4) & 0x0F);
    uint8_t* p      = (uint8_t*)&packed[bi];
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
init_packed_s4_matrix(int8_t* packed, md_t k, md_t n, md_t ldb)
{
    for (md_t i = 0; i < k; i++) {
        for (md_t j = 0; j < n; j++) {
            int v = (int)((i + j) % 16) - 8;
            pack_s4_at(packed, ldb, i, j, (int8_t)v);
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
print_packed_s4_section(const char*   name,
                        const int8_t* packed,
                        md_t          k,
                        md_t          n,
                        md_t          ldb,
                        int           max_rows,
                        int           max_cols)
{
    printf("%s (%d x %d) - showing top-left %d x %d:\n", name, (int)k, (int)n,
           max_rows, max_cols);
    for (int i = 0; i < max_rows && i < (int)k; i++) {
        for (int j = 0; j < max_cols && j < (int)n; j++) {
            printf("%4d ", (int)unpack_s4(packed, ldb, (md_t)i, (md_t)j));
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
    printf("BF16×S4→F32 GEMM with WOQ B scales (symmetric, no zero-point)\n\n");

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
    int8_t*   b_packed    = (int8_t*)malloc(packed_input_bytes);
    int8_t*   b_reordered = (int8_t*)malloc(reorder_bytes);
    float*    c           = (float*)malloc((size_t)ldc * m * sizeof(*c));

    if (!a || !b_packed || !b_reordered || !c) {
        printf("Memory allocation failed\n");
        free(a);
        free(b_packed);
        free(b_reordered);
        free(c);
        return -1;
    }

    init_bf16_matrix(a, m, k, lda, 2.0f);
    memset(b_packed, 0, packed_input_bytes);
    init_packed_s4_matrix(b_packed, k, n, ldb_in);

    print_bf16_matrix_section("Matrix A (BF16)", a, m, k, lda, 3, 3);
    print_packed_s4_section("Matrix B (S4 packed, logical view before reorder)",
                            b_packed, k, n, ldb_in, 3, 3);

    // Reorder the matrix
    aocl_reorder_bf16s4f32of32('R', 'N', 'B', b_packed, b_reordered, k, n,
                               ldb_in, &meta_reorder);
    if (meta_reorder.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        printf("aocl_reorder_bf16s4f32of32 failed, error_code=%d\n",
               (int)meta_reorder.error_hndl.error_code);
        free(a);
        free(b_packed);
        free(b_reordered);
        free(c);
        return -1;
    }
    printf("(B reordered in-place to separate buffer for GEMM; mem_format_b "
           "'R')\n\n");

    // Initialize the result matrix
    memset(c, 0, (size_t)ldc * m * sizeof(float));
    print_f32_matrix_section("Initial Matrix C (F32)", c, m, n, ldc, 3, 3);

    md_t           ldb_gemm = n;
    dlp_metadata_t metadata;

    // =======================================================================
    // Example 1: WOQ per-tensor B scale
    // =======================================================================
    printf("\n--- Example 1: WOQ per-tensor B scale ---\n\n");

    // Initialize the per-tensor B scale
    float      b_scale_tensor = 1.25f;
    dlp_sf_t   b_scl_tensor   = { .scale_factor      = &b_scale_tensor,
                                  .scale_factor_len  = 1,
                                  .scale_factor_type = DLP_F32 };
    dlp_pre_op pre_tensor     = { .b_zp       = NULL,
                                  .b_scl      = &b_scl_tensor,
                                  .seq_length = 1,
                                  .group_size = 0 };

    printf("B WOQ: scale_factor_len=1 (per-tensor), value=%.6f\n\n",
           b_scale_tensor);

    // Initialize the metadata
    memset(&metadata, 0, sizeof(metadata));
    metadata.pre_ops = &pre_tensor;
    memset(c, 0, (size_t)ldc * (size_t)m * sizeof(float));

    // Run the GEMM
    aocl_gemm_bf16s4f32of32('R', 'N', 'N', m, n, k, 1.0f, a, lda, 'N',
                            b_reordered, ldb_gemm, 'R', 0.0f, c, ldc,
                            &metadata);
    if (metadata.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        printf("GEMM failed, error_code=%d\n",
               (int)metadata.error_hndl.error_code);
        goto cleanup;
    }

    print_f32_matrix_section("Result Matrix C (F32)", c, m, n, ldc, 3, 3);

    // =======================================================================
    // Example 2: WOQ per-channel B scales (len = n)
    // =======================================================================
    printf("\n--- Example 2: WOQ per-channel B scales (len=n) ---\n\n");

    float* b_scale_channel = (float*)malloc((size_t)n * sizeof(float));
    if (!b_scale_channel) {
        printf("Channel scale allocation failed\n");
        goto cleanup;
    }
    // Initialize the per-channel B scale
    for (md_t j = 0; j < n; j++) {
        b_scale_channel[j] = 0.5f + 0.01f * (float)j;
    }

    // Initialize the per-channel B scale metadata
    dlp_sf_t   b_scl_channel = { .scale_factor      = b_scale_channel,
                                 .scale_factor_len  = n,
                                 .scale_factor_type = DLP_F32 };
    dlp_pre_op pre_channel   = { .b_zp       = NULL,
                                 .b_scl      = &b_scl_channel,
                                 .seq_length = 1,
                                 .group_size = 0 };

    printf("B WOQ: scale_factor_len=%ld (per output channel), first 3: "
           "%.6f, %.6f, %.6f\n",
           (long)n, b_scale_channel[0], b_scale_channel[1], b_scale_channel[2]);
    printf("  ... (scales for all %ld columns)\n\n", (long)n);

    // Initialize the metadata
    memset(&metadata, 0, sizeof(metadata));
    metadata.pre_ops = &pre_channel;
    memset(c, 0, (size_t)ldc * (size_t)m * sizeof(float));

    // Run the GEMM
    aocl_gemm_bf16s4f32of32('R', 'N', 'N', m, n, k, 1.0f, a, lda, 'N',
                            b_reordered, ldb_gemm, 'R', 0.0f, c, ldc,
                            &metadata);
    if (metadata.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        printf("GEMM failed, error_code=%d\n",
               (int)metadata.error_hndl.error_code);
        free(b_scale_channel);
        goto cleanup;
    }

    print_f32_matrix_section("Result Matrix C (F32)", c, m, n, ldc, 3, 3);

    free(b_scale_channel);

cleanup:
    free(a);
    free(b_packed);
    free(b_reordered);
    free(c);
    return 0;
}
