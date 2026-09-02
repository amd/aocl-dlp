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
 * Example demonstrating how to recover a matrix from its reordered form
 *
 * aocl_unreorder_<type>() is the exact inverse of aocl_reorder_<type>(). It
 * is useful when the reordered buffer is the only live copy of the weights,
 * for example to serialise them back out, hand them to another library, or
 * inspect them while debugging.
 *
 * This example shows how to:
 * 1. Reorder B, run a GEMM with it, then recover the original B
 * 2. Check for errors through dlp_metadata_t
 * 3. Un-reorder a column-major matrix
 * 4. Fall back to the portable _reference variant when the tuned entry
 *    point is not supported on the current processor
 * 5. Size the output buffer correctly
 */

#include "aocl_dlp.h"
#include "dlp_example_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The reordered buffer stores no header describing how it was produced, so
// the caller has to remember the shape and layout it used at reorder time.
typedef struct
{
    md_t k;     // Rows of the original B
    md_t n;     // Columns of the original B
    md_t ldb;   // Leading dimension of the plain (un-reordered) B
    char order; // 'r' row-major or 'c' column-major
    char trans; // 'n' or 't' for the plain B storage
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
        default:
            return "error";
    }
}

// Number of elements the un-reorder API may write, which is what the output
// buffer has to be able to hold.
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

// Reorder B, recover it, and report whether the round trip was exact.
static int
roundtrip_f32(const b_layout_t* l)
{
    dlp_metadata_t md;
    int            status = -1;

    md_t   b_elems = output_elem_count(l);
    float* b       = (float*)malloc(b_elems * sizeof(float));
    float* b_out   = (float*)malloc(b_elems * sizeof(float));
    if (!b || !b_out) {
        printf("  Memory allocation failed\n");
        goto cleanup;
    }
    init_matrix_f32(b, b_elems);
    memset(b_out, 0, b_elems * sizeof(float));

    // Step 1: ask how large the reordered buffer has to be. Never assume a
    // size, it depends on the shape and on the target processor.
    memset(&md, 0, sizeof(md));
    msz_t reorder_buf_size = aocl_get_reorder_buf_size_f32f32f32of32(
        l->order, l->trans, 'B', l->k, l->n, &md);
    if (reorder_buf_size == 0) {
        printf("  Reorder buffer size query failed: %s\n",
               err_name(md.error_hndl.error_code));
        goto cleanup;
    }

    float* b_reordered = (float*)malloc(reorder_buf_size);
    if (!b_reordered) {
        printf("  Memory allocation failed\n");
        goto cleanup;
    }

    // Step 2: reorder B into the packed layout.
    memset(&md, 0, sizeof(md));
    aocl_reorder_f32f32f32of32(l->order, l->trans, 'B', b, b_reordered, l->k,
                               l->n, l->ldb, &md);
    if (md.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        printf("  Reorder failed: %s\n", err_name(md.error_hndl.error_code));
        free(b_reordered);
        goto cleanup;
    }

    // Step 3: recover B. trans is the output storage and need not match the
    // pack; here it does, so memcmp against the input is valid.
    memset(&md, 0, sizeof(md));
    aocl_unreorder_f32f32f32of32_reference(l->order, l->trans, 'B', b_reordered,
                                           b_out, l->k, l->n, l->ldb, &md);
    if (md.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        printf("  Un-reorder failed: %s\n", err_name(md.error_hndl.error_code));
        free(b_reordered);
        goto cleanup;
    }

    // The round trip is bit-exact, so no tolerance is needed here.
    if (memcmp(b, b_out, b_elems * sizeof(float)) == 0) {
        printf("  k=%4ld n=%4ld order=%c trans=%c : recovered exactly "
               "(%zu byte packed buffer)\n",
               (long)l->k, (long)l->n, l->order, l->trans,
               (size_t)reorder_buf_size);
        status = 0;
    } else {
        printf("  k=%4ld n=%4ld order=%c trans=%c : MISMATCH\n", (long)l->k,
               (long)l->n, l->order, l->trans);
    }

    free(b_reordered);

cleanup:
    free(b);
    free(b_out);
    return status;
}

// bf16 has both a tuned and a portable entry point. The tuned one needs
// AVX-512-BF16, so production code should fall back when it is missing.
static int
roundtrip_bf16(md_t k, md_t n)
{
    dlp_metadata_t md;
    int            status  = -1;
    md_t           ldb     = n; // row-major
    md_t           b_elems = k * n;

    bfloat16* b     = (bfloat16*)malloc(b_elems * sizeof(bfloat16));
    bfloat16* b_out = (bfloat16*)malloc(b_elems * sizeof(bfloat16));
    if (!b || !b_out) {
        printf("  Memory allocation failed\n");
        goto cleanup;
    }
    for (md_t i = 0; i < b_elems; i++) {
        b[i] = (bfloat16)(0x3f00 + (i % 128));
    }
    memset(b_out, 0, b_elems * sizeof(bfloat16));

    memset(&md, 0, sizeof(md));
    msz_t reorder_buf_size =
        aocl_get_reorder_buf_size_bf16bf16f32of32('r', 'n', 'B', k, n, &md);
    if (reorder_buf_size == 0) {
        printf("  Size query failed: %s\n", err_name(md.error_hndl.error_code));
        goto cleanup;
    }

    bfloat16* b_reordered = (bfloat16*)malloc(reorder_buf_size);
    if (!b_reordered) {
        printf("  Memory allocation failed\n");
        goto cleanup;
    }

    memset(&md, 0, sizeof(md));
    aocl_reorder_bf16bf16f32of32('r', 'n', 'B', b, b_reordered, k, n, ldb, &md);
    if (md.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        printf("  bf16 reorder not supported here, skipping\n");
        free(b_reordered);
        goto cleanup;
    }

    // Try the tuned entry point first and fall back to the portable one.
    memset(&md, 0, sizeof(md));
    aocl_unreorder_bf16bf16f32of32('r', 'n', 'B', b_reordered, b_out, k, n, ldb,
                                   &md);
    if (md.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED) {
        printf("  Tuned un-reorder unavailable, using the reference\n");
        memset(&md, 0, sizeof(md));
        aocl_unreorder_bf16bf16f32of32_reference('r', 'n', 'B', b_reordered,
                                                 b_out, k, n, ldb, &md);
    }

    if (md.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        printf("  Un-reorder failed: %s\n", err_name(md.error_hndl.error_code));
    } else if (memcmp(b, b_out, b_elems * sizeof(bfloat16)) == 0) {
        printf("  k=%4ld n=%4ld order=r : recovered exactly\n", (long)k,
               (long)n);
        status = 0;
    } else {
        printf("  k=%4ld n=%4ld order=r : MISMATCH\n", (long)k, (long)n);
    }

    free(b_reordered);

cleanup:
    free(b);
    free(b_out);
    return status;
}

// Un-reorder reports problems through metadata rather than a return value,
// so always inspect error_hndl.error_code.
static void
show_error_reporting(void)
{
    md_t  k = 64, n = 64;
    float dummy_in[4]  = { 0 };
    float dummy_out[4] = { 0 };

    dlp_metadata_t md;
    memset(&md, 0, sizeof(md));

    // The calls below fail on purpose. The library also writes a diagnostic
    // of its own to stderr, so flush stdout first to keep the two in order.
    fflush(stdout);

    // Only the B matrix can be un-reordered; 'A' is rejected.
    aocl_unreorder_f32f32f32of32_reference('r', 'n', 'A', dummy_in, dummy_out,
                                           k, n, n, &md);
    printf("  mat_type='A'      -> %s\n", err_name(md.error_hndl.error_code));
    fflush(stdout);

    // A NULL buffer is rejected rather than dereferenced.
    memset(&md, 0, sizeof(md));
    aocl_unreorder_f32f32f32of32_reference('r', 'n', 'B', NULL, dummy_out, k, n,
                                           n, &md);
    printf("  reorder_buf=NULL  -> %s\n", err_name(md.error_hndl.error_code));
    fflush(stdout);
}

int
main(void)
{
    int failures = 0;

    printf("Matrix Un-reorder Example\n");
    printf("=========================\n\n");

    printf("Round trip, row-major f32:\n");
    {
        b_layout_t cases[] = {
            { 1024, 1024, 1024, 'r', 'n' },
            { 256, 512, 512, 'r', 'n' },
            // Shapes that are not multiples of the internal blocking are
            // handled too.
            { 17, 33, 33, 'r', 'n' },
            { 17, 33, 17, 'r', 't' },
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            failures += (roundtrip_f32(&cases[i]) != 0);
        }
    }

    printf("\nRound trip, column-major f32 (ldb is then the row count):\n");
    {
        b_layout_t cases[] = {
            { 512, 256, 512, 'c', 'n' },
            { 33, 17, 33, 'c', 'n' },
            { 33, 17, 17, 'c', 't' },
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            failures += (roundtrip_f32(&cases[i]) != 0);
        }
    }

    printf("\nRound trip, bf16 with fallback to the reference:\n");
    roundtrip_bf16(512, 256);

    printf("\nError reporting through metadata:\n");
    show_error_reporting();

    printf("\n%s\n", failures ? "Some round trips failed."
                              : "All round trips recovered the original "
                                "matrix exactly.");
    return failures ? 1 : 0;
}
