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
 * Experiment: U8S8S32 GEMM with a 12x32 kernel tile.
 *
 * B is packed first with aocl_reorder_u8s8s32os32 (NR-parameterized VNNI-4
 * packer) and then consumed as mem_format_b == 'R'. There is no runtime pack
 * in the GEMM call.
 */

#include "aocl_dlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
init_u8_matrix(uint8_t* matrix, int rows, int cols, int base_value)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (uint8_t)((base_value * (i + j) % 256));
        }
    }
}

static void
init_s8_matrix(int8_t* matrix, int rows, int cols, int base_value)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (int8_t)((base_value * (i + j) % 255) - 127);
        }
    }
}

static void
print_s32_section(const char*    name,
                  const int32_t* matrix,
                  int            rows,
                  int            cols,
                  int            max_rows,
                  int            max_cols)
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

static int
check_api(const char* what, const dlp_metadata_t* metadata)
{
    if (metadata->error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED) {
        printf("%s: AVX512-VNNI not supported on this processor.\n", what);
        return -1;
    }
    if (metadata->error_hndl.error_code != DLP_CLSC_SUCCESS) {
        printf("%s failed, error_code=%d\n", what,
               (int)metadata->error_hndl.error_code);
        return -1;
    }
    return 0;
}

int
main(void)
{
    const md_t    m     = 48;
    const md_t    n     = 96;
    const md_t    k     = 33;
    const md_t    lda   = k;
    const md_t    ldb   = n;
    const md_t    ldc   = n;
    const int32_t alpha = 1;
    const int32_t beta  = 0;

    dlp_gemm_blocking_t block_params;
    memset(&block_params, 0, sizeof(block_params));
    block_params.MR = 12;
    block_params.NR = 32;

    dlp_metadata_t metadata;
    memset(&metadata, 0, sizeof(metadata));
    metadata.block_params = &block_params;

    printf("U8S8S32 GEMM experiment: kernel tile 12x32, B packed then GEMM "
           "with mem_format_b='R'\n");
    printf("Problem size: m=%ld n=%ld k=%ld\n\n", (long)m, (long)n, (long)k);

    msz_t reorder_bytes =
        aocl_get_reorder_buf_size_u8s8s32os32('R', 'N', 'B', k, n, &metadata);
    if (check_api("aocl_get_reorder_buf_size_u8s8s32os32", &metadata) != 0) {
        return -1;
    }
    if (reorder_bytes == 0) {
        printf("Reorder buffer size is 0.\n");
        return -1;
    }

    uint8_t* a           = (uint8_t*)malloc((size_t)lda * (size_t)m);
    int8_t*  b           = (int8_t*)malloc((size_t)ldb * (size_t)k);
    int8_t*  b_reordered = (int8_t*)malloc(reorder_bytes);
    int32_t* c = (int32_t*)malloc((size_t)ldc * (size_t)m * sizeof(int32_t));
    int32_t* c_ref =
        (int32_t*)malloc((size_t)ldc * (size_t)m * sizeof(int32_t));

    if (!a || !b || !b_reordered || !c || !c_ref) {
        printf("Memory allocation failed\n");
        free(a);
        free(b);
        free(b_reordered);
        free(c);
        free(c_ref);
        return -1;
    }

    init_u8_matrix(a, (int)m, (int)k, 5);
    init_s8_matrix(b, (int)k, (int)n, 3);
    memset(c, 0, (size_t)m * (size_t)n * sizeof(int32_t));

    aocl_reorder_u8s8s32os32('R', 'N', 'B', b, b_reordered, k, n, ldb,
                             &metadata);
    if (check_api("aocl_reorder_u8s8s32os32", &metadata) != 0) {
        free(a);
        free(b);
        free(b_reordered);
        free(c);
        free(c_ref);
        return -1;
    }
    printf("Packed B with aocl_reorder_u8s8s32os32 (%zu bytes).\n\n",
           (size_t)reorder_bytes);

    aocl_gemm_u8s8s32os32('R', 'N', 'N', m, n, k, alpha, a, lda, 'N',
                          b_reordered, ldb, 'R', beta, c, ldc, &metadata);
    if (check_api("aocl_gemm_u8s8s32os32", &metadata) != 0) {
        free(a);
        free(b);
        free(b_reordered);
        free(c);
        free(c_ref);
        return -1;
    }

    print_s32_section("Result C (12x32, packed B)", c, (int)m, (int)n, 3, 3);

    for (md_t i = 0; i < m; i++) {
        for (md_t j = 0; j < n; j++) {
            int32_t acc = 0;
            for (md_t p = 0; p < k; p++) {
                acc += (int32_t)a[i * lda + p] * (int32_t)b[p * ldb + j];
            }
            c_ref[i * ldc + j] = alpha * acc;
        }
    }

    int mismatches = 0;
    for (md_t i = 0; i < m; i++) {
        for (md_t j = 0; j < n; j++) {
            if (c[i * ldc + j] != c_ref[i * ldc + j]) {
                if (mismatches < 4) {
                    printf("Mismatch at (%ld, %ld): gemm=%d naive=%d\n",
                           (long)i, (long)j, c[i * ldc + j],
                           c_ref[i * ldc + j]);
                }
                mismatches++;
            }
        }
    }

    if (mismatches == 0) {
        printf("Results match the naive reference.\n");
    } else {
        printf("%d mismatches versus the naive reference.\n", mismatches);
    }

    free(a);
    free(b);
    free(b_reordered);
    free(c);
    free(c_ref);
    return (mismatches == 0) ? 0 : 1;
}
