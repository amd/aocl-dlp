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
 * Example: portable reference reorder and un-reorder of B
 *
 * aocl_reorder_<type>_reference() packs B; aocl_unreorder_<type>_reference()
 * is the inverse. Both are plain C (no ISA gate on f32) and honour
 * metadata->block_params. Use them to recover a packed B, or as a fallback
 * when the tuned entry point is unavailable.
 *
 * This example shows how to:
 * 1. Query the packed-buffer size
 * 2. Pack with aocl_reorder_f32f32f32of32_reference()
 * 3. Recover with aocl_unreorder_f32f32f32of32_reference()
 * 4. Pass the same order, k, n and ldb to both sides
 * 5. Override NC/KC/NR through metadata (must match on pack and unpack)
 * 6. Inspect errors through dlp_metadata_t
 *
 * See matrix_reorder.c for the tuned packer used with GEMM mem_format_b='R',
 * and matrix_unreorder.c for recovering a buffer that the tuned packer wrote.
 */

#include "aocl_dlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    md_t k;
    md_t n;
    md_t ldb;
    char order; // 'r' or 'c'
    char trans; // 'n' or 't' — storage of the plain B, pack and unpack
} b_layout_t;

static void
init_matrix_f32(float* matrix, md_t count)
{
    for (md_t i = 0; i < count; i++) {
        matrix[i] = (float)((i % 251) - 125) * 0.125f;
    }
}

static const char*
err_name(dlp_clsc_err_t code)
{
    switch (code) {
        case DLP_CLSC_SUCCESS:
            return "DLP_CLSC_SUCCESS";
        case DLP_CLSC_NOT_SUPPORTED:
            return "DLP_CLSC_NOT_SUPPORTED";
        case DLP_CLSC_NULL_POINTER:
            return "DLP_CLSC_NULL_POINTER";
        case DLP_CLSC_INVALID_BLOCK_PARAMS:
            return "DLP_CLSC_INVALID_BLOCK_PARAMS";
        default:
            return "error";
    }
}

static md_t
output_elem_count(const b_layout_t* l)
{
    const int row = (l->order == 'r') || (l->order == 'R');
    const int nt  = (l->trans == 'n') || (l->trans == 'N');
    if (row) {
        return nt ? (l->k * l->ldb) : (l->n * l->ldb);
    }
    return nt ? (l->n * l->ldb) : (l->k * l->ldb);
}

// Pack then unpack. If bp is non-NULL it is attached to metadata on both
// calls so the two sides decode the same KC x NC panel layout.
static int
roundtrip_f32_reference(const b_layout_t* l, dlp_gemm_blocking_t* bp)
{
    dlp_metadata_t md;
    int            status  = -1;
    md_t           b_elems = output_elem_count(l);
    float*         b       = (float*)malloc((size_t)b_elems * sizeof(float));
    float*         b_out   = (float*)malloc((size_t)b_elems * sizeof(float));
    float*         packed  = NULL;

    if (!b || !b_out) {
        printf("  Memory allocation failed\n");
        goto cleanup;
    }
    init_matrix_f32(b, b_elems);
    memset(b_out, 0, (size_t)b_elems * sizeof(float));

    // Step 1: packed size. Never assume it; it depends on k, n and padding.
    memset(&md, 0, sizeof(md));
    md.block_params    = bp;
    msz_t packed_bytes = aocl_get_reorder_buf_size_f32f32f32of32(
        l->order, l->trans, 'B', l->k, l->n, &md);
    if (packed_bytes == 0) {
        printf("  Size query failed: %s\n", err_name(md.error_hndl.error_code));
        goto cleanup;
    }

    packed = (float*)malloc((size_t)packed_bytes);
    if (!packed) {
        printf("  Memory allocation failed\n");
        goto cleanup;
    }

    // Step 2: pack B. trans selects how the plain matrix is stored; the
    // packed buffer is always logical k x n.
    memset(&md, 0, sizeof(md));
    md.block_params = bp;
    aocl_reorder_f32f32f32of32_reference(l->order, l->trans, 'B', b, packed,
                                         l->k, l->n, l->ldb, &md);
    if (md.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        printf("  Reference reorder failed: %s\n",
               err_name(md.error_hndl.error_code));
        goto cleanup;
    }

    // Step 3: unpack with the same trans so the recovered storage matches
    // the input. trans on un-reorder is independent and can also write B^T.
    memset(&md, 0, sizeof(md));
    md.block_params = bp;
    aocl_unreorder_f32f32f32of32_reference(l->order, l->trans, 'B', packed,
                                           b_out, l->k, l->n, l->ldb, &md);
    if (md.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        printf("  Reference un-reorder failed: %s\n",
               err_name(md.error_hndl.error_code));
        goto cleanup;
    }

    if (memcmp(b, b_out, (size_t)b_elems * sizeof(float)) == 0) {
        printf("  k=%4ld n=%4ld order=%c trans=%c : recovered exactly "
               "(%zu byte packed buffer)\n",
               (long)l->k, (long)l->n, l->order, l->trans,
               (size_t)packed_bytes);
        status = 0;
    } else {
        printf("  k=%4ld n=%4ld order=%c trans=%c : MISMATCH\n", (long)l->k,
               (long)l->n, l->order, l->trans);
    }

cleanup:
    free(packed);
    free(b);
    free(b_out);
    return status;
}

static void
show_error_reporting(void)
{
    md_t           k = 64, n = 64;
    float          dummy_in[4]  = { 0 };
    float          dummy_out[4] = { 0 };
    dlp_metadata_t md;

    fflush(stdout);

    memset(&md, 0, sizeof(md));
    aocl_unreorder_f32f32f32of32_reference('r', 'n', 'A', dummy_in, dummy_out,
                                           k, n, n, &md);
    printf("  mat_type='A'     -> %s\n", err_name(md.error_hndl.error_code));
    fflush(stdout);

    memset(&md, 0, sizeof(md));
    aocl_unreorder_f32f32f32of32_reference('r', 'n', 'W', dummy_in, dummy_out,
                                           k, n, n, &md);
    printf("  mat_type='W'     -> %s\n", err_name(md.error_hndl.error_code));
    fflush(stdout);

    memset(&md, 0, sizeof(md));
    aocl_reorder_f32f32f32of32_reference('r', 'n', 'B', NULL, dummy_out, k, n,
                                         n, &md);
    printf("  input_buf=NULL   -> %s\n", err_name(md.error_hndl.error_code));
    fflush(stdout);
}

int
main(void)
{
    int failures = 0;

    printf("Reference Reorder / Un-reorder Example\n");
    printf("======================================\n\n");

    printf("Architecture-default blocking, row-major:\n");
    {
        b_layout_t cases[] = {
            { 256, 256, 256, 'r', 'n' },
            { 17, 33, 33, 'r', 'n' },
            { 17, 33, 17, 'r', 't' },
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            failures += (roundtrip_f32_reference(&cases[i], NULL) != 0);
        }
    }

    printf("\nArchitecture-default blocking, column-major "
           "(ldb is then the row count):\n");
    {
        b_layout_t cases[] = {
            { 128, 64, 128, 'c', 'n' },
            { 33, 17, 33, 'c', 'n' },
            { 33, 17, 17, 'c', 't' },
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            failures += (roundtrip_f32_reference(&cases[i], NULL) != 0);
        }
    }

    // Fields left 0 keep the architecture default. NC must be a multiple of
    // NR; for f32, KC has no extra grouping constraint.
    printf("\nCaller-supplied NR/NC/KC (same metadata on pack and unpack):\n");
    {
        dlp_gemm_blocking_t bp;
        memset(&bp, 0, sizeof(bp));
        bp.NR = 16;
        bp.NC = 64;
        bp.KC = 32;

        b_layout_t l = { 100, 80, 80, 'r', 'n' };
        failures += (roundtrip_f32_reference(&l, &bp) != 0);
    }

    printf("\nError reporting through metadata:\n");
    show_error_reporting();

    printf("\n%s\n",
           failures ? "Some round trips failed."
                    : "All round trips recovered the original matrix exactly.");
    return failures ? 1 : 0;
}
