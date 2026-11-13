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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
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

#ifndef DLP_GEMM_F32_TINY_SGEMM_H
#define DLP_GEMM_F32_TINY_SGEMM_H

#include <stdbool.h>

#include "classic/aocl_gemm_metadata.h"
#include "classic/dlp_base_types.h"
#include "classic/dlp_macros.h"
#include "jit/f32_tiny_gemm_kernel.h"

/* RD is profitable only after its horizontal-reduction setup is amortized. */
#ifndef DLP_F32_TINY_SGEMM_RD_MIN_K
#define DLP_F32_TINY_SGEMM_RD_MIN_K 8
#endif

#ifndef DLP_F32_TINY_SGEMM_MAX_M
#define DLP_F32_TINY_SGEMM_MAX_M 48
#endif

#ifndef DLP_F32_TINY_SGEMM_WORK_LIMIT
#define DLP_F32_TINY_SGEMM_WORK_LIMIT 256
#endif

/* One RV panel of columns; wider RD outputs use the wide-RD leaf. */
#ifndef DLP_F32_TINY_SGEMM_PANEL_N
#define DLP_F32_TINY_SGEMM_PANEL_N 64
#endif

/*
 * Longest K the wide-RD leaf takes. The leaf keeps the whole reduction in
 * registers and streams B once, which beats the framework while K is short.
 * Past that the framework's K panel wins instead, so the leaf declines rather
 * than taking a shape it would lose on.
 */
#ifndef DLP_F32_TINY_SGEMM_WIDE_RD_MAX_K
#define DLP_F32_TINY_SGEMM_WIDE_RD_MAX_K 112
#endif

/* What dlp_sgemm_tiny_f32 runs for an eligible call. */
typedef enum
{
    DLP_SGEMM_TINY_NONE = 0,
    /* B contiguous along n: broadcast A, load B rows. */
    DLP_SGEMM_TINY_RV,
    /* A and B contiguous along k: dot products, reduced per output. */
    DLP_SGEMM_TINY_RD,
    /* AVX-512 RD for one or two rows with B streamed, beyond one panel. */
    DLP_SGEMM_TINY_RD_WIDE,
    /* alpha == 0: C = beta * C without reading A or B. */
    DLP_SGEMM_TINY_SCALE_C
} dlp_sgemm_tiny_leaf_t;

/* ISA tier and cache sizes, fixed once the library is initialized. */
typedef struct
{
    bool avx512;       /* AVX-512 leaves; implies avx2 is also usable */
    bool avx2;         /* AVX2/FMA leaves allowed by the configured arch */
    md_t wide_rd_rows; /* max rows of the wide RD tile: 2 on AVX-512, else 1 */
    md_t l1d_bytes;
    md_t l2_bytes;
} dlp_sgemm_tiny_isa_t;

typedef struct
{
    md_t m;
    md_t n;
    md_t k;
    md_t rs_a;
    md_t cs_a;
    md_t rs_b;
    md_t cs_b;
    md_t rs_c;
    md_t cs_c;
    /* Set when C^T = B^T * A^T is computed: the leaf's A is the caller's B. */
    bool swapped;
} dlp_sgemm_tiny_layout;

static inline void
dlp_sgemm_tiny_gate_strides(
    bool row_major, bool trans, md_t ld, md_t* rs, md_t* cs)
{
    if (row_major) {
        *rs = trans ? 1 : ld;
        *cs = trans ? ld : 1;
    } else {
        *rs = trans ? ld : 1;
        *cs = trans ? 1 : ld;
    }
}

static inline dlp_sgemm_tiny_layout
dlp_sgemm_tiny_normalize(char order,
                         bool trans_a,
                         bool trans_b,
                         md_t m,
                         md_t n,
                         md_t k,
                         md_t lda,
                         md_t ldb,
                         md_t ldc)
{
    const bool            row_major = (order == 'r') || (order == 'R');
    dlp_sgemm_tiny_layout out       = { m, n, k, 0, 0, 0, 0, 0, 0, false };

    dlp_sgemm_tiny_gate_strides(row_major, trans_a, lda, &out.rs_a, &out.cs_a);
    dlp_sgemm_tiny_gate_strides(row_major, trans_b, ldb, &out.rs_b, &out.cs_b);
    dlp_sgemm_tiny_gate_strides(row_major, false, ldc, &out.rs_c, &out.cs_c);

    if ((out.cs_c != 1) || ((out.rs_c == 1) && (out.m > out.n))) {
        md_t tmp    = out.m;
        out.m       = out.n;
        out.n       = tmp;
        tmp         = out.rs_a;
        out.rs_a    = out.cs_b;
        out.cs_b    = tmp;
        tmp         = out.cs_a;
        out.cs_a    = out.rs_b;
        out.rs_b    = tmp;
        tmp         = out.rs_c;
        out.rs_c    = out.cs_c;
        out.cs_c    = tmp;
        out.swapped = true;
    }
    return out;
}

/* Preserve the previously measured alpha-zero coverage. */
static inline bool
dlp_sgemm_tiny_legacy_is_eligible(md_t m, md_t n, md_t k)
{
    return (m >= 2) && (n >= 2) && (m <= DLP_F32_TINY_SGEMM_MAX_M)
           && (m <= (DLP_F32_TINY_SGEMM_WORK_LIMIT / n))
           && ((m * n) <= (DLP_F32_TINY_SGEMM_WORK_LIMIT / k));
}

/*
 * Keep the route policy in terms of the work each leaf repeats.
 *
 * RV walks K once per 64-column output panel. The measured safe region needs
 * one such panel and no more than the rows/K-depth that keep the direct leaf
 * ahead of framework setup.
 *
 * RD produces four columns per reduction tile. Up to four column tiles can
 * carry four times as many rows and twice the K depth; up to two column tiles
 * can amortize four times the K depth. Wider outputs retain eight rows because
 * every additional four columns repeat the horizontal-reduction tree. A GEMV
 * keeps sixteen rows and the shallower K, beyond which the dedicated GEMV
 * kernels win.
 */
static inline bool
dlp_sgemm_tiny_rv_is_profitable(md_t m, md_t n, md_t k)
{
    return (m <= 16) && (n >= 3) && (n <= DLP_F32_TINY_SGEMM_PANEL_N)
           && (k <= 256);
}

static inline bool
dlp_sgemm_tiny_rd_is_profitable(md_t m, md_t n, md_t k)
{
    const md_t rd_tile_n        = 4;
    const md_t max_column_tiles = 16;

    if (n > max_column_tiles * rd_tile_n) {
        return false;
    }

    const bool gemv  = (m == 1) || (n == 1);
    const md_t max_m = (n > 4 * rd_tile_n) ? 8 : (gemv ? 16 : 32);
    const md_t max_k = (n <= 2 * rd_tile_n)              ? 1024
                       : ((n <= 4 * rd_tile_n) && !gemv) ? 512
                                                         : 256;
    return (m <= max_m) && (k <= max_k);
}

/* The direct RV/RD regions on a normalized layout. */
static inline bool
dlp_sgemm_tiny_layout_is_profitable(const dlp_sgemm_tiny_layout* layout)
{
    if (layout->cs_b == 1) {
        return dlp_sgemm_tiny_rv_is_profitable(layout->m, layout->n, layout->k);
    }
    return (layout->cs_a == 1) && (layout->rs_b == 1)
           && (layout->k > DLP_F32_TINY_SGEMM_RD_MIN_K)
           && dlp_sgemm_tiny_rd_is_profitable(layout->m, layout->n, layout->k);
}

/*
 * Cheap GEMV screen. It uses only the public dimensions and scales, so the
 * caller can skip metadata and stride normalization for K=1, scalar, long,
 * and oversized vectors. Public N=1 also needs K past the RD minimum: those
 * shorter reductions stay on the framework GEMV path.
 */
static inline bool
dlp_sgemm_tiny_gemv_worth_probing(
    md_t m, md_t n, md_t k, float alpha, float beta)
{
    if (((m == 1) == (n == 1)) || (k <= 1) || (k > 1024)
        || (((m == 1) ? n : m) > DLP_F32_TINY_SGEMM_PANEL_N) || (alpha == 0.0f)
        || (alpha != alpha) || (beta != beta)) {
        return false;
    }
    return (m == 1) || (k > DLP_F32_TINY_SGEMM_RD_MIN_K);
}

/*
 * Wide RD takes shapes the direct leaves do not cover: n spans more than one
 * packing panel, K-contiguous row-major A and B-transposed B (so both dot
 * products stream unit-stride), and K is short enough that the reduction
 * stays in registers. The row count is capped by the RD tile height
 * (wide_rd_row_capacity), so rows beyond the generator's MR are never routed
 * here.
 */
static inline bool
dlp_sgemm_tiny_wide_rd_layout_ok(const dlp_sgemm_tiny_layout* layout,
                                 md_t wide_rd_row_capacity)
{
    return (layout->m <= wide_rd_row_capacity)
           && (layout->n > DLP_F32_TINY_SGEMM_PANEL_N)
           && (layout->k > DLP_F32_TINY_SGEMM_RD_MIN_K)
           && (layout->k <= DLP_F32_TINY_SGEMM_WIDE_RD_MAX_K)
           && (layout->cs_a == 1) && (layout->rs_b == 1) && (layout->cs_b != 1)
           && (layout->k <= layout->n / 2);
}

static inline bool
dlp_sgemm_tiny_wide_rd_layout_fits(const dlp_sgemm_tiny_layout* layout,
                                   md_t                         l1d_bytes,
                                   md_t                         l2_bytes)
{
    if ((l1d_bytes < 4096) || (l2_bytes < 65536)) {
        return false;
    }

    const md_t a_elements = (l1d_bytes / 2) / (md_t)sizeof(float);
    const md_t b_elements = (l2_bytes - l2_bytes / 8) / (md_t)sizeof(float);

    return (layout->m > 0) && (layout->n > 0)
           && (layout->k <= a_elements / layout->m)
           && (layout->k <= b_elements / layout->n);
}

/*
 * The leaf for a call, from one normalization of its layout, or
 * DLP_SGEMM_TINY_NONE for the framework. This is the only production gate;
 * *layout is the normalized layout when a leaf is set.
 */
static inline dlp_sgemm_tiny_leaf_t
dlp_sgemm_tiny_route(const dlp_sgemm_tiny_isa_t* isa,
                     char                        order,
                     bool                        trans_a,
                     bool                        trans_b,
                     md_t                        m,
                     md_t                        n,
                     md_t                        k,
                     float                       alpha,
                     float                       beta,
                     md_t                        lda,
                     md_t                        ldb,
                     md_t                        ldc,
                     dlp_sgemm_tiny_layout*      layout)
{
    const bool matrix = (m >= 2) && (n >= 2);

    if ((alpha != alpha) || (beta != beta)) {
        return DLP_SGEMM_TINY_NONE;
    }
    if (alpha == 0.0f) {
        return (matrix && (DLP_F32_TINY_SGEMM_WORK_LIMIT != 0)
                && dlp_sgemm_tiny_legacy_is_eligible(m, n, k))
                   ? DLP_SGEMM_TINY_SCALE_C
                   : DLP_SGEMM_TINY_NONE;
    }
    if (!isa->avx512 && !isa->avx2) {
        return DLP_SGEMM_TINY_NONE;
    }
    if (!matrix && !dlp_sgemm_tiny_gemv_worth_probing(m, n, k, alpha, beta)) {
        return DLP_SGEMM_TINY_NONE;
    }

    *layout = dlp_sgemm_tiny_normalize(order, trans_a, trans_b, m, n, k, lda,
                                       ldb, ldc);
    bool eligible = (!matrix || (DLP_F32_TINY_SGEMM_WORK_LIMIT != 0))
                    && dlp_sgemm_tiny_layout_is_profitable(layout);
    if (!eligible && matrix && (DLP_F32_TINY_SGEMM_WORK_LIMIT != 0)) {
        eligible = dlp_sgemm_tiny_wide_rd_layout_ok(layout, isa->wide_rd_rows)
                   && dlp_sgemm_tiny_wide_rd_layout_fits(layout, isa->l1d_bytes,
                                                         isa->l2_bytes);
    }
    if (!eligible) {
        return DLP_SGEMM_TINY_NONE;
    }
    if (layout->cs_b == 1) {
        return DLP_SGEMM_TINY_RV;
    }
    return (isa->avx512 && (layout->n > DLP_F32_TINY_SGEMM_PANEL_N))
               ? DLP_SGEMM_TINY_RD_WIDE
               : DLP_SGEMM_TINY_RD;
}

/*
 * RV column block: the widest multiple of the panel (four zmm or two ymm)
 * whose k x nc slice of B fits half of L1, and at least one panel. k is
 * bounded by the RV regions, so the quotient fits 32 bits.
 */
static inline md_t
dlp_sgemm_tiny_rv_nc(const dlp_sgemm_tiny_isa_t* isa, md_t k)
{
    const md_t panel = isa->avx512 ? 64 : 16;
    const md_t nc    = (md_t)((uint32_t)(isa->l1d_bytes / 2)
                           / (uint32_t)(k * (md_t)sizeof(float)))
                    & ~(panel - 1);

    return (nc > panel) ? nc : panel;
}

/*
 * True when metadata carries nothing the framework would apply or validate.
 * GLU is held outside seq_vector, so seq_length == 0 alone is not enough.
 */
static inline bool
dlp_sgemm_tiny_metadata_ok(const dlp_metadata_t* metadata)
{
    return (metadata == NULL)
           || ((metadata->seq_length == 0) && (metadata->glu == NULL)
               && (metadata->a_quant_op == NULL)
               && (metadata->b_quant_op == NULL)
               && (metadata->block_params == NULL)
               && (metadata->sup_thresholds == NULL)
               && (metadata->gemm_hints == NULL));
}

DLP_BEGIN_EXTERN_C

/* Records the ISA tier and cache sizes. Called once at library init. */
void
dlp_sgemm_tiny_init(void);

/*
 * Runs a small f32 GEMM on the calling thread when dlp_sgemm_tiny_route
 * selects a leaf: the lazily emitted RV or RD kernel, with column-stored C
 * computed as C^T = B^T * A^T, or C = beta * C for alpha == 0. Returns false
 * without changing C when no leaf applies or kernel generation fails, so the
 * caller can use the framework path.
 */
bool
dlp_sgemm_tiny_f32(char         order,
                   dlp_trans_t  transa,
                   dlp_trans_t  transb,
                   md_t         m,
                   md_t         n,
                   md_t         k,
                   float        alpha,
                   const float* a,
                   md_t         lda,
                   const float* b,
                   md_t         ldb,
                   float        beta,
                   float*       c,
                   md_t         ldc);

DLP_END_EXTERN_C

#endif // DLP_GEMM_F32_TINY_SGEMM_H
