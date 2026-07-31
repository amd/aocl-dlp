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
 * Example: S8×S4→F32 symmetric static quantization GEMM
 * (aocl_gemm_s8s4s32of32).
 *
 * A is a standard s8 activation matrix. B is a signed 4-bit weight matrix,
 * nibble-packed (two 4-bit values per byte, even element in the low nibble).
 * This example pre-reorders B via aocl_reorder_s8s4s32os32 and invokes the GEMM
 * with mem_format_b == 'R' (the compact-memory fast path). Passing a raw
 * nibble-packed B with mem_format_b == 'N' (or 'P') is also supported and packs
 * B at runtime. Per-group static scales live in the unified quant metadata:
 * metadata->a_quant_op / metadata->b_quant_op, each carrying group_size and
 * dequant_scale_factors (DLP_F32 here).
 */

#include "aocl_dlp.h"
#include <math.h>
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

/* Fill a (rows x cols) signed 4-bit matrix (values in [-8, 7]) row-major and
 * nibble-pack it: element at linear index t occupies byte (t >> 1); even t in
 * the low nibble, odd t in the high nibble. Also returns the unpacked s4
 * values (as int8) in b_ref for reference computation. */
static void
InitS4MatrixPacked(uint8_t* packed, int8_t* b_ref, int rows, int cols, int base)
{
    int total = rows * cols;
    for (int t = 0; t < (total + 1) / 2; t++) {
        packed[t] = 0;
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int t       = i * cols + j;
            int v       = ((base * (i * 3 + j)) % 15) - 7; /* range [-7, 7] */
            b_ref[t]    = (int8_t)v;
            uint8_t nib = (uint8_t)(v & 0x0F);
            if ((t & 1) == 0) {
                packed[t >> 1] |= nib;
            } else {
                packed[t >> 1] |= (uint8_t)(nib << 4);
            }
        }
    }
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
            printf("%10.4f ", matrix[i * ld + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/* Reference: per-group static dequantized GEMM.
 * c[i,j] = sum_g a_sf[i,g] * b_sf[g,j] * (sum_{kk in group g} a[i,kk]*b[kk,j])
 */
static void
ReferenceSymQuant(const int8_t* a,
                  const int8_t* b_ref,
                  float*        c,
                  int           m,
                  int           n,
                  int           k,
                  int           lda,
                  int           ldc,
                  int           group_size,
                  int           ng,
                  const float*  a_sf,
                  const float*  b_sf)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int g = 0; g < ng; g++) {
                int k0 = g * group_size;
                int k1 = k0 + group_size;
                if (k1 > k) {
                    k1 = k;
                }
                int32_t isum = 0;
                for (int kk = k0; kk < k1; kk++) {
                    isum +=
                        (int32_t)a[i * lda + kk] * (int32_t)b_ref[kk * n + j];
                }
                acc += a_sf[i * ng + g] * b_sf[g * n + j] * (float)isum;
            }
            c[i * ldc + j] = acc;
        }
    }
}

int
main(void)
{
    printf("S8×S4→F32 GEMM with symmetric static quantization "
           "(reordered s4 B)\n\n");

    md_t m   = 64;
    md_t n   = 64;
    md_t k   = 64;
    md_t lda = k;
    md_t ldb = n;
    md_t ldc = n;

    const md_t group_size = 32;
    const md_t ng         = (k + group_size - 1) / group_size;

    int8_t*  a      = (int8_t*)malloc((size_t)lda * (size_t)m * sizeof(*a));
    uint8_t* b_pack = (uint8_t*)malloc(((size_t)k * (size_t)n + 1) / 2);
    int8_t*  b_ref  = (int8_t*)malloc((size_t)k * (size_t)n * sizeof(*b_ref));
    float*   c      = (float*)malloc((size_t)ldc * (size_t)m * sizeof(*c));
    float*   c_ref  = (float*)malloc((size_t)ldc * (size_t)m * sizeof(*c_ref));
    float*   a_sf   = (float*)malloc((size_t)m * (size_t)ng * sizeof(float));
    float*   b_sf   = (float*)malloc((size_t)ng * (size_t)n * sizeof(float));

    if ((a == NULL) || (b_pack == NULL) || (b_ref == NULL) || (c == NULL)
        || (c_ref == NULL) || (a_sf == NULL) || (b_sf == NULL)) {
        printf("Memory allocation failed\n");
        goto cleanup;
    }

    InitS8Matrix(a, (int)m, (int)k, (int)lda, 3);
    InitS4MatrixPacked(b_pack, b_ref, (int)k, (int)n, 5);

    for (md_t i = 0; i < m; i++) {
        for (md_t g = 0; g < ng; g++) {
            a_sf[i * ng + g] = 0.015f + 0.0003f * (float)(i + g);
        }
    }
    for (md_t g = 0; g < ng; g++) {
        for (md_t j = 0; j < n; j++) {
            b_sf[g * n + j] = 0.025f + 0.00015f * (float)(g * n + j);
        }
    }

    const char    order        = 'R';
    const char    transa       = 'N';
    const char    transb       = 'N';
    const char    mem_format_a = 'N';
    const int32_t alpha        = 1;
    const int32_t beta0        = 0;

    dlp_metadata_t metadata;
    dlp_qparam_t   a_scl_q;
    dlp_qparam_t   b_scl_q;
    dlp_quant_op_t a_quant_op;
    dlp_quant_op_t b_quant_op;

    /* Configure the unified symmetric-quant metadata once and reuse it for the
     * reorder (which reads only the B-side group size) and the GEMM (which also
     * reads the per-group dequant scale factors). This mirrors the s8s8
     * sym-quant example (simple_gemm_s8_sym_quant.c). */
    memset(&metadata, 0, sizeof(metadata));
    memset(&a_scl_q, 0, sizeof(a_scl_q));
    memset(&b_scl_q, 0, sizeof(b_scl_q));
    memset(&a_quant_op, 0, sizeof(a_quant_op));
    memset(&b_quant_op, 0, sizeof(b_quant_op));

    a_scl_q.data      = a_sf;
    a_scl_q.len       = m * ng;
    a_scl_q.stor_type = DLP_F32;
    a_scl_q.outer_dim = DLP_PARAM_DIM_PER_GROUP;

    b_scl_q.data      = b_sf;
    b_scl_q.len       = ng * n;
    b_scl_q.stor_type = DLP_F32;
    b_scl_q.outer_dim = DLP_PARAM_DIM_PER_GROUP;

    /* A is s8, B is nibble-packed s4 widened to s8 for the VNNI kernel. */
    a_quant_op.quant_op_kind         = DLP_QUANT_OP_QUANTIZE;
    a_quant_op.src_type              = DLP_S8;
    a_quant_op.dst_type              = DLP_S8;
    a_quant_op.group_size            = group_size;
    a_quant_op.dequant_scale_factors = &a_scl_q;

    b_quant_op.quant_op_kind         = DLP_QUANT_OP_QUANTIZE;
    b_quant_op.src_type              = DLP_S4;
    b_quant_op.dst_type              = DLP_S8;
    b_quant_op.group_size            = group_size;
    b_quant_op.dequant_scale_factors = &b_scl_q;

    metadata.a_quant_op = &a_quant_op;
    metadata.b_quant_op = &b_quant_op;

    /* Step 1: query the reordered buffer size and reorder the s4 B matrix. */
    msz_t reorder_size = aocl_get_reorder_buf_size_s8s4s32os32(
        order, transb, 'B', k, n, &metadata);
    if (metadata.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED) {
        printf("s8s4 reorder not supported (AVX512-VNNI required).\n");
        goto cleanup;
    }
    if ((reorder_size == 0)
        || (metadata.error_hndl.error_code != DLP_CLSC_SUCCESS)) {
        printf("Reorder buffer size query failed, error_code=%d\n",
               (int)metadata.error_hndl.error_code);
        goto cleanup;
    }

    int8_t* b_reorder = (int8_t*)malloc((size_t)reorder_size);
    if (b_reorder == NULL) {
        printf("Reorder buffer allocation failed\n");
        goto cleanup;
    }

    aocl_reorder_s8s4s32os32(order, transb, 'B', (const int8_t*)b_pack,
                             b_reorder, k, n, ldb, &metadata);
    if (metadata.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        printf("Reorder failed, error_code=%d\n",
               (int)metadata.error_hndl.error_code);
        free(b_reorder);
        goto cleanup;
    }

    /* Step 2: run the GEMM. The per-group dequant scale factors were already
     * attached to metadata (a_quant_op/b_quant_op) above. */
    memset(c, 0, (size_t)ldc * (size_t)m * sizeof(float));

    aocl_gemm_s8s4s32of32(order, transa, transb, m, n, k, alpha, a, lda,
                          mem_format_a, b_reorder, ldb, 'R', beta0, c, ldc,
                          &metadata);
    if (metadata.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        printf("GEMM failed, error_code=%d\n",
               (int)metadata.error_hndl.error_code);
        free(b_reorder);
        goto cleanup;
    }

    PrintF32Section("Result C (F32)", c, (int)m, (int)n, (int)ldc, 3, 3);

    /* Step 3: validate against a straightforward reference implementation. */
    ReferenceSymQuant(a, b_ref, c_ref, (int)m, (int)n, (int)k, (int)lda,
                      (int)ldc, (int)group_size, (int)ng, a_sf, b_sf);

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    for (md_t i = 0; i < m; i++) {
        for (md_t j = 0; j < n; j++) {
            double got   = (double)c[i * ldc + j];
            double ref   = (double)c_ref[i * ldc + j];
            double ae    = fabs(got - ref);
            double denom = fabs(ref) > 1e-6 ? fabs(ref) : 1e-6;
            double re    = ae / denom;
            if (ae > max_abs_err) {
                max_abs_err = ae;
            }
            if (re > max_rel_err) {
                max_rel_err = re;
            }
        }
    }
    printf("Validation vs reference: max_abs_err=%.6g, max_rel_err=%.6g\n",
           max_abs_err, max_rel_err);
    if (max_rel_err < 1e-3) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }

    free(b_reorder);

cleanup:
    free(a);
    free(b_pack);
    free(b_ref);
    free(c);
    free(c_ref);
    free(a_sf);
    free(b_sf);
    return 0;
}
