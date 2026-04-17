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
 * Example: S8×S8→F32 symmetric static quantization GEMM
 * (aocl_gemm_s8s8s32of32_sym_quant).
 *
 * Scales live in metadata->post_op_grp (dlp_group_post_op): a_scl, b_scl
 * (DLP_F32 here).
 */

#include "aocl_dlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
InitS8Matrix(int8_t* matrix, int rows, int cols, int ld, int base)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int v              = (base * (i + j)) % 251 - 125;
            matrix[i * ld + j] = (int8_t)v;
        }
    }
}

static void
PrintS8Section(const char* name,
               int8_t*     matrix,
               int         rows,
               int         cols,
               int         ld,
               int         max_r,
               int         max_c)
{
    printf("%s (%d x %d) - top-left %d x %d:\n", name, rows, cols, max_r,
           max_c);
    for (int i = 0; i < max_r && i < rows; i++) {
        for (int j = 0; j < max_c && j < cols; j++) {
            printf("%4d ", (int)matrix[i * ld + j]);
        }
        printf("\n");
    }
    printf("\n");
}

static void
PrintF32Section(const char* name,
                float*      matrix,
                int         rows,
                int         cols,
                int         ld,
                int         max_r,
                int         max_c)
{
    printf("%s (%d x %d) - top-left %d x %d:\n", name, rows, cols, max_r,
           max_c);
    for (int i = 0; i < max_r && i < rows; i++) {
        for (int j = 0; j < max_c && j < cols; j++) {
            printf("%8.4f ", matrix[i * ld + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int
main(void)
{
    printf("S8×S8→F32 GEMM with symmetric static quantization "
           "(post_op_grp scales)\n\n");

    md_t m   = 64;
    md_t n   = 64;
    md_t k   = 64;
    md_t lda = k;
    md_t ldb = n;
    md_t ldc = n;

    int8_t* a = (int8_t*)malloc((size_t)lda * (size_t)m * sizeof(*a));
    int8_t* b = (int8_t*)malloc((size_t)ldb * (size_t)k * sizeof(*b));
    float*  c = (float*)malloc((size_t)ldc * (size_t)m * sizeof(*c));

    if ((a == NULL) || (b == NULL) || (c == NULL)) {
        printf("Memory allocation failed\n");
        free(a);
        free(b);
        free(c);
        return -1;
    }

    InitS8Matrix(a, (int)m, (int)k, (int)lda, 3);
    InitS8Matrix(b, (int)k, (int)n, (int)ldb, 5);

    PrintS8Section("Matrix A (S8)", a, (int)m, (int)k, (int)lda, 3, 3);
    PrintS8Section("Matrix B (S8)", b, (int)k, (int)n, (int)ldb, 3, 3);

    const char    order        = 'R';
    const char    transa       = 'N';
    const char    transb       = 'N';
    const char    mem_format_a = 'N';
    const char    mem_format_b = 'N';
    const int32_t alpha        = 1;
    const int32_t beta0        = 0;

    dlp_metadata_t    metadata;
    dlp_group_post_op grp;
    dlp_sf_t          a_scl;
    dlp_sf_t          b_scl;

    /* Example 1: group_size 0 → ng = 1 */
    printf("--- Example 1: group_size=0 (full K) ---\n\n");

    md_t   ng1   = 1;
    float* a_sf1 = (float*)malloc((size_t)m * (size_t)ng1 * sizeof(float));
    float* b_sf1 = (float*)malloc((size_t)ng1 * (size_t)n * sizeof(float));
    if ((a_sf1 == NULL) || (b_sf1 == NULL)) {
        printf("Scale allocation failed\n");
        free(a_sf1);
        free(b_sf1);
        goto cleanup;
    }
    for (md_t i = 0; i < m; i++) {
        a_sf1[i] = 0.02f + 0.0001f * (float)i;
    }
    for (md_t j = 0; j < n; j++) {
        b_sf1[j] = 0.03f + 0.0002f * (float)j;
    }

    memset(&grp, 0, sizeof(grp));
    grp.group_size = 0;
    grp.seq_length = 1;
    grp.a_scl      = &a_scl;
    grp.b_scl      = &b_scl;
    grp.a_zp       = NULL;
    grp.b_zp       = NULL;

    a_scl.scale_factor      = a_sf1;
    a_scl.scale_factor_len  = m * ng1;
    a_scl.scale_factor_type = DLP_F32;
    b_scl.scale_factor      = b_sf1;
    b_scl.scale_factor_len  = ng1 * n;
    b_scl.scale_factor_type = DLP_F32;

    memset(&metadata, 0, sizeof(metadata));
    metadata.post_op_grp = &grp;
    memset(c, 0, (size_t)ldc * (size_t)m * sizeof(float));

    aocl_gemm_s8s8s32of32_sym_quant(order, transa, transb, m, n, k, alpha, a,
                                    lda, mem_format_a, b, ldb, mem_format_b,
                                    beta0, c, ldc, &metadata);
    if (metadata.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED) {
        printf("GEMM not supported (AVX512-VNNI required).\n");
        free(a_sf1);
        free(b_sf1);
        goto cleanup;
    }
    if (metadata.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        printf("GEMM failed, error_code=%d\n",
               (int)metadata.error_hndl.error_code);
        free(a_sf1);
        free(b_sf1);
        goto cleanup;
    }

    PrintF32Section("Result C (F32)", c, (int)m, (int)n, (int)ldc, 3, 3);
    free(a_sf1);
    free(b_sf1);

    /* Example 2: group_size 32, k=64 → ng=2 */
    printf("--- Example 2: group_size=32, ng=2 ---\n\n");

    md_t   ng2   = (k + 32 - 1) / 32;
    float* a_sf2 = (float*)malloc((size_t)m * (size_t)ng2 * sizeof(float));
    float* b_sf2 = (float*)malloc((size_t)ng2 * (size_t)n * sizeof(float));
    if ((a_sf2 == NULL) || (b_sf2 == NULL)) {
        printf("Scale allocation failed\n");
        free(a_sf2);
        free(b_sf2);
        goto cleanup;
    }
    for (md_t i = 0; i < m; i++) {
        for (md_t g = 0; g < ng2; g++) {
            a_sf2[i * ng2 + g] = 0.015f + 0.0003f * (float)(i + g);
        }
    }
    for (md_t g = 0; g < ng2; g++) {
        for (md_t j = 0; j < n; j++) {
            b_sf2[g * n + j] = 0.025f + 0.00015f * (float)(g * n + j);
        }
    }

    memset(&grp, 0, sizeof(grp));
    grp.group_size = 32;
    grp.seq_length = 1;
    grp.a_scl      = &a_scl;
    grp.b_scl      = &b_scl;
    grp.a_zp       = NULL;
    grp.b_zp       = NULL;

    a_scl.scale_factor      = a_sf2;
    a_scl.scale_factor_len  = m * ng2;
    a_scl.scale_factor_type = DLP_F32;
    b_scl.scale_factor      = b_sf2;
    b_scl.scale_factor_len  = ng2 * n;
    b_scl.scale_factor_type = DLP_F32;

    memset(&metadata, 0, sizeof(metadata));
    metadata.post_op_grp = &grp;
    memset(c, 0, (size_t)ldc * (size_t)m * sizeof(float));

    aocl_gemm_s8s8s32of32_sym_quant(order, transa, transb, m, n, k, alpha, a,
                                    lda, mem_format_a, b, ldb, mem_format_b,
                                    beta0, c, ldc, &metadata);
    if (metadata.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        printf("GEMM failed, error_code=%d\n",
               (int)metadata.error_hndl.error_code);
        free(a_sf2);
        free(b_sf2);
        goto cleanup;
    }

    PrintF32Section("Result C (F32)", c, (int)m, (int)n, (int)ldc, 3, 3);
    free(a_sf2);
    free(b_sf2);

cleanup:
    free(a);
    free(b);
    free(c);
    return 0;
}
