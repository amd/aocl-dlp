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

#ifndef DLP_GEMM_REORDER_LAYOUT_H
#define DLP_GEMM_REORDER_LAYOUT_H

#include "dlp_gemm_types.h"

/*
 * One description of the reordered-B panel layout, for any NR.
 *
 * The decision engine picks MR/NR/MC/KC/NC per kernel, so nothing downstream
 * of it may assume the 64-wide (or 128-wide, for fp16) micro panel the hand
 * written kernels were built around. The layout is not a family of special
 * cases though; the widths a packer subdivides a panel into, and where each
 * element lands inside a sub panel, both follow from (NR, min_NR, k_factor).
 * Stating that rule once here is what lets the reference packer and unpacker
 * agree at a width neither of them was written for.
 *
 * A KC x NC panel is cut along n into chunks:
 *
 *   |<--- NR --->|<--- NR --->|<-- w -->|<min_NR>|
 *   +------------+------------+---------+--------+
 *   |   full     |   full     | fringe  | tail   |
 *
 * Full NR-wide chunks come first. What is left over (rem = nc0 % NR) becomes
 * at most two more: a fringe of the largest whole multiple of min_NR that
 * fits, then a tail holding rem % min_NR columns. The tail still occupies a
 * whole min_NR-wide slot, zero filled, because min_NR is the granularity the
 * buffer size was rounded to. For f32 min_NR equals NR, so there is no fringe
 * and the tail is one NR-wide slot; for bf16 and int8 min_NR is 16, and for
 * fp16 it is 32, which is where the 48/32/16 and 96/64/32 ladders in the hand
 * written kernels come from.
 *
 * Inside a chunk of width `slot`, elements are k major with k_factor rows
 * interleaved per column, which is the VNNI grouping the dot-product
 * instructions need (k_factor 1 for f32/fp16, 2 for bf16, 4 for int8):
 *
 *   offset(kr, i) = (kr / kf) * (kf * slot) + (i * kf) + (kr % kf)
 *
 * Rows past kc0 and columns past the chunk's live width are zero, so a packed
 * buffer is fully defined even when k is not a multiple of k_factor.
 */

// One chunk of a panel: `cols` live columns starting at `col_start`, occupying
// a `slot`-wide sub panel that begins at `offset` elements into the panel.
typedef struct
{
    md_t col_start;
    md_t cols;
    md_t slot;
    md_t offset;
} dlp_reorder_chunk_t;

// Rounds `value` up to the next multiple of `to`. Same rule as
// dlp_make_multiple_of_n(); kept here so this header stays free of runtime
// includes.
DLP_INLINE md_t
dlp_reorder_round_up(md_t value, md_t to)
{
    if (to <= 0) {
        return 0;
    }
    return ((value + to - 1) / to) * to;
}

// Element offset within a chunk. See the layout note above.
DLP_INLINE md_t
dlp_reorder_elem_offset(md_t kr, md_t i, md_t slot, md_t k_factor)
{
    return ((kr / k_factor) * (k_factor * slot)) + (i * k_factor)
           + (kr % k_factor);
}

// Total elements a panel of nc0 columns and kc0 rows occupies once padded.
DLP_INLINE md_t
dlp_reorder_panel_elems(md_t nc0, md_t kc0, md_t min_NR, md_t k_factor)
{
    return dlp_reorder_round_up(nc0, min_NR)
           * dlp_reorder_round_up(kc0, k_factor);
}

// Seeds a walk over the chunks of an nc0-wide panel. Pair with
// dlp_reorder_chunk_next().
DLP_INLINE void
dlp_reorder_chunk_first(dlp_reorder_chunk_t* chunk)
{
    chunk->col_start = 0;
    chunk->cols      = 0;
    chunk->slot      = 0;
    chunk->offset    = 0;
}

// Advances to the next chunk, returning FALSE once the panel is covered.
// kc0_padded is kc0 rounded up to k_factor, i.e. the packed height of a chunk.
DLP_INLINE bool
dlp_reorder_chunk_next(
    dlp_reorder_chunk_t* chunk, md_t nc0, md_t NR, md_t min_NR, md_t kc0_padded)
{
    md_t next_col = chunk->col_start + chunk->cols;

    chunk->offset += chunk->slot * kc0_padded;
    chunk->col_start = next_col;

    md_t remaining = nc0 - next_col;
    if (remaining <= 0) {
        return FALSE;
    }
    // NR==0 would set cols=0 and never advance col_start.
    if ((NR <= 0) || (min_NR <= 0)) {
        return FALSE;
    }

    if (remaining >= NR) {
        // A full micro panel.
        chunk->cols = chunk->slot = NR;
    } else {
        md_t whole = (remaining / min_NR) * min_NR;
        if (whole > 0) {
            // The fringe: the widest whole multiple of min_NR that fits.
            chunk->cols = chunk->slot = whole;
        } else {
            // The tail: fewer than min_NR columns, in a full min_NR slot.
            chunk->cols = remaining;
            chunk->slot = min_NR;
        }
    }

    return TRUE;
}

// True when the blocking can be packed by the reference walk: NR and KC
// must be positive, NR a multiple of the tail granularity, and KC a
// multiple of the VNNI grouping. Without the positivity checks, NR==0
// (or KC==0) would pass because 0 % min_NR == 0, and a zero NR would
// stall dlp_reorder_chunk_next() on a 0-wide chunk.
DLP_INLINE bool
dlp_reorder_ref_blocks_legal(md_t NR, md_t min_NR, md_t KC, md_t k_factor)
{
    return (NR > 0) && (KC > 0) && (min_NR > 0) && (k_factor > 0)
           && ((NR % min_NR) == 0) && ((KC % k_factor) == 0);
}

// Rejects a product that would wrap msz_t (CWE-190). aocl_get_reorder_buf_size
// used to return a wrapped allocation for huge k*n (the reorder overflow in
// PR #609); this is that same guard. DLP_MSZ_MAX is UINT64_MAX / UINT32_MAX
// matching msz_t. a*b overflows iff a != 0 and b > DLP_MSZ_MAX / a.
DLP_INLINE bool
dlp_reorder_checked_mul_msz(msz_t a, msz_t b, msz_t* out)
{
    if ((a != 0) && (b > DLP_MSZ_MAX / a)) {
        return FALSE;
    }
    *out = a * b;
    return TRUE;
}

DLP_INLINE bool
dlp_reorder_size_bytes(
    md_t rows, md_t cols, md_t elem_sz, msz_t extra, msz_t* out)
{
    msz_t prod = 0;
    if ((rows < 0) || (cols < 0) || (elem_sz <= 0)) {
        return FALSE;
    }
    if (!dlp_reorder_checked_mul_msz((msz_t)rows, (msz_t)cols, &prod)) {
        return FALSE;
    }
    if (!dlp_reorder_checked_mul_msz(prod, (msz_t)elem_sz, &prod)) {
        return FALSE;
    }
    if (extra > DLP_MSZ_MAX - prod) {
        return FALSE;
    }
    *out = prod + extra;
    return TRUE;
}

/*
 * Width-agnostic reference pack/unpack for one KC x NC panel.
 *
 * elem_sz is the element size in bytes and k_factor the VNNI k grouping, so
 * one implementation serves every dtype. Strides are given as (rs_b, cs_b) on
 * the unpacked matrix, which makes row major and column major the same call.
 * The two are exact inverses; pack defines the padding, unpack never reads it.
 */
void
dlp_reorder_ref_packb(void*       pack_b,
                      const void* b,
                      md_t        elem_sz,
                      md_t        k_factor,
                      md_t        nc0,
                      md_t        kc0,
                      md_t        NR,
                      md_t        min_NR,
                      md_t        rs_b,
                      md_t        cs_b);

void
dlp_reorder_ref_unpackb(const void* pack_b,
                        void*       b,
                        md_t        elem_sz,
                        md_t        k_factor,
                        md_t        nc0,
                        md_t        kc0,
                        md_t        NR,
                        md_t        min_NR,
                        md_t        rs_b,
                        md_t        cs_b);

/*
 * Whole-B reorder / un-reorder: the same KC x NC walk, pack or unpack at
 * each tile. Public aocl_(un)reorder_*_reference APIs call these directly.
 * Both wrappers share dlp_reorder_ref_walk(); to_packed selects direction
 * so the two cannot drift. packb/unpackb remain the per-panel inverses.
 *
 * Panels are stored down k first, then across n. Numbers are storage order
 * (1 then 2 then 3 ...). Reorder writes that order; un-reorder reads it.
 * Thread count does not change the bytes: jc is split NR-wide, but offsets
 * are computed from the global (jc, n, NC), so a buffer packed at NT=1 is
 * unpacked correctly at any NT, and the other way around.
 *
 *              t1              t2
 *              |               |
 *              |           |..NC..|
 *              |           |      |
 *              |.NC. |.NC. |NC'|NC"|
 *         pc=0-+-----+-----+---+--+
 *            KC|  1  |  3  |   5  |
 *        pc=KC-+-----+-----+---st-+
 *            KC|  2  |  4  | 6 | 7|
 *     pc=k=2KC-+-----+-----+---+--+
 *              |jc=0 |jc=NC|jc=2NC|
 *
 * t2 starts inside the last NC panel, so it only owns NC' of that panel
 * (then NC" on the next jc step). Offset of st, with k_padded =
 * round_up(k, k_factor) and kc0_padded = round_up(kc0, k_factor):
 *
 *   st = jc_cur_loop * k_padded     // panels 1,2,3,4
 *      + n_sub_updated * pc         // panel 5
 *      + jc_cur_loop_rem * kc0_padded  // panel 6
 *
 * Inside a panel, n is cut into NR / fringe / min_NR chunks as above.
 * MR and MC are not used; k padding is round_up(k, k_factor).
 */
void
dlp_reorder_ref_reorderb(void*       pack_b,
                         const void* b,
                         md_t        elem_sz,
                         md_t        k_factor,
                         md_t        n,
                         md_t        k,
                         md_t        NR,
                         md_t        min_NR,
                         md_t        NC,
                         md_t        KC,
                         md_t        rs_b,
                         md_t        cs_b,
                         md_t        n_threads);

void
dlp_reorder_ref_unreorderb(void*       b,
                           const void* pack_b,
                           md_t        elem_sz,
                           md_t        k_factor,
                           md_t        n,
                           md_t        k,
                           md_t        NR,
                           md_t        min_NR,
                           md_t        NC,
                           md_t        KC,
                           md_t        rs_b,
                           md_t        cs_b,
                           md_t        n_threads);

#endif // DLP_GEMM_REORDER_LAYOUT_H
