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

/*
 * Width-agnostic reference packer and unpacker for reordered B panels.
 *
 * These are the portable oracles behind every dtype's reorder/un-reorder
 * reference. They are driven entirely by (NR, min_NR, k_factor) taken from the
 * runtime context, so a panel width the decision engine invents at runtime
 * needs no new code here. The layout rule they implement is documented in
 * dlp_gemm_reorder_layout.h.
 *
 * Element type is handled as a size in bytes rather than by generating a copy
 * per dtype. A reference path is not on any hot loop, and one implementation
 * that every dtype shares is worth far more than the copies it saves: the pack
 * and unpack below are literal inverses of each other, which is exactly the
 * property the round-trip tests assert.
 *
 * Source and destination strides are passed as (rs_b, cs_b), so row major
 * (rs_b = ldb, cs_b = 1) and column major (rs_b = 1, cs_b = ldb) are the same
 * code path rather than two hand-mirrored ones.
 */

#include <string.h>

#include "gemm_utils/dlp_gemm_reorder_layout.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "threading/dlp_gemm_thread_utils.h"

#ifdef DLP_ENABLE_OPENMP
#include <omp.h>
#endif

// Copies one element. elem_sz is 1, 2 or 4 bytes for the supported dtypes.
DLP_INLINE void
dlp_reorder_copy_elem(void* dst, const void* src, md_t elem_sz)
{
    memcpy(dst, src, (size_t)elem_sz);
}

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
                      md_t        cs_b)
{
    const md_t  kc0_padded = dlp_reorder_round_up(kc0, k_factor);
    char*       out        = (char*)pack_b;
    const char* in         = (const char*)b;

    // Zero first so that both the k rows past kc0 and the columns past a
    // chunk's live width are defined without the copy loops below having to
    // special-case them.
    memset(out, 0,
           (size_t)dlp_reorder_panel_elems(nc0, kc0, min_NR, k_factor)
               * (size_t)elem_sz);

    dlp_reorder_chunk_t chunk;
    dlp_reorder_chunk_first(&chunk);

    while (dlp_reorder_chunk_next(&chunk, nc0, NR, min_NR, kc0_padded)) {
        for (iter_t kr = 0; kr < kc0; kr++) {
            for (iter_t i = 0; i < chunk.cols; i++) {
                md_t src = ((kr * rs_b) + ((chunk.col_start + i) * cs_b));
                md_t dst =
                    chunk.offset
                    + dlp_reorder_elem_offset(kr, i, chunk.slot, k_factor);

                dlp_reorder_copy_elem(out + ((size_t)dst * (size_t)elem_sz),
                                      in + ((size_t)src * (size_t)elem_sz),
                                      elem_sz);
            }
        }
    }
}

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
                        md_t        cs_b)
{
    const md_t  kc0_padded = dlp_reorder_round_up(kc0, k_factor);
    const char* in         = (const char*)pack_b;
    char*       out        = (char*)b;

    dlp_reorder_chunk_t chunk;
    dlp_reorder_chunk_first(&chunk);

    // The exact inverse of the loop above: same chunk walk, same offsets, with
    // source and destination exchanged. Padding is simply never read, so the
    // caller's matrix keeps whatever it held outside the nc0 x kc0 window.
    while (dlp_reorder_chunk_next(&chunk, nc0, NR, min_NR, kc0_padded)) {
        for (iter_t kr = 0; kr < kc0; kr++) {
            for (iter_t i = 0; i < chunk.cols; i++) {
                md_t dst = ((kr * rs_b) + ((chunk.col_start + i) * cs_b));
                md_t src =
                    chunk.offset
                    + dlp_reorder_elem_offset(kr, i, chunk.slot, k_factor);

                dlp_reorder_copy_elem(out + ((size_t)dst * (size_t)elem_sz),
                                      in + ((size_t)src * (size_t)elem_sz),
                                      elem_sz);
            }
        }
    }
}

// One jc/NC x pc/KC walk. to_packed selects pack vs unpack at each tile so
// reorder and un-reorder cannot drift apart on offsets or k-padding.
// packed is the reordered buffer in both directions; to_packed chooses
// whether it is written or read.
//
// Storage order of KC x NC panels (numbers = write/read order). Un-reorder
// walks the same diagram in reverse at each element, not a different layout.
//
//              t1              t2
//              |               |
//              |           |..NC..|
//              |           |      |
//              |.NC. |.NC. |NC'|NC"|
//         pc=0-+-----+-----+---+--+
//            KC|  1  |  3  |   5  |
//        pc=KC-+-----+-----+---st-+
//            KC|  2  |  4  | 6 | 7|
//     pc=k=2KC-+-----+-----+---+--+
//              |jc=0 |jc=NC|jc=2NC|
//
// Offset of st (t2, pc=KC, inside the last NC panel):
//   jc_cur_loop * k_padded + n_sub_updated * pc + jc_cur_loop_rem * kc0_padded
//
// jc is split NR-wide across threads. n_way is the team that actually
// spawned (not the requested NT), so any thread count still covers [0, n).
static void
dlp_reorder_ref_walk(void* unpacked,
                     void* packed,
                     int   to_packed,
                     md_t  elem_sz,
                     md_t  k_factor,
                     md_t  n,
                     md_t  k,
                     md_t  NR,
                     md_t  min_NR,
                     md_t  NC,
                     md_t  KC,
                     md_t  rs_b,
                     md_t  cs_b,
                     md_t  n_threads)
{
    const md_t k_padded = dlp_reorder_round_up(k, k_factor);

    n_threads = (n_threads > 0) ? n_threads : 1;

    char*       unpacked_rw = (char*)unpacked;
    char*       packed_rw   = (char*)packed;
    const char* packed_ro   = (const char*)packed;

#ifdef DLP_ENABLE_OPENMP
    _Pragma("omp parallel num_threads(n_threads)")
    {
        dlp_task_id_t thread_jc;
        thread_jc.n_way   = omp_get_num_threads();
        thread_jc.work_id = omp_get_thread_num();
#else
    {
        dlp_task_id_t thread_jc;
        thread_jc.n_way   = 1;
        thread_jc.work_id = 0;
#endif

        md_t jc_start, jc_end;
        dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

        for (iter_t jc = jc_start; jc < jc_end; jc += NC) {
            md_t nc0 = dlp_min((jc_end - jc), NC);

            md_t jc_cur_loop     = jc;
            md_t jc_cur_loop_rem = 0;
            md_t n_sub_updated;

            dlp_gemm_get_B_panel_reordered_start_offset_width(
                jc, n, NC, min_NR, &jc_cur_loop, &jc_cur_loop_rem, &nc0,
                &n_sub_updated);

            for (iter_t pc = 0; pc < k; pc += KC) {
                const md_t kc0        = dlp_min((k - pc), KC);
                const md_t kc0_padded = dlp_reorder_round_up(kc0, k_factor);
                const md_t off = (jc_cur_loop * k_padded) + (n_sub_updated * pc)
                                 + (jc_cur_loop_rem * kc0_padded);
                const size_t packed_bytes = (size_t)off * (size_t)elem_sz;
                const size_t matrix_bytes =
                    (size_t)((rs_b * pc) + (jc * cs_b)) * (size_t)elem_sz;

                if (to_packed) {
                    dlp_reorder_ref_packb(
                        packed_rw + packed_bytes, unpacked_rw + matrix_bytes,
                        elem_sz, k_factor, nc0, kc0, NR, min_NR, rs_b, cs_b);
                } else {
                    dlp_reorder_ref_unpackb(
                        packed_ro + packed_bytes, unpacked_rw + matrix_bytes,
                        elem_sz, k_factor, nc0, kc0, NR, min_NR, rs_b, cs_b);
                }
            }

            dlp_gemm_adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }
}

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
                         md_t        n_threads)
{
    dlp_reorder_ref_walk((void*)b, pack_b, 1, elem_sz, k_factor, n, k, NR,
                         min_NR, NC, KC, rs_b, cs_b, n_threads);
}

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
                           md_t        n_threads)
{
    dlp_reorder_ref_walk(b, (void*)pack_b, 0, elem_sz, k_factor, n, k, NR,
                         min_NR, NC, KC, rs_b, cs_b, n_threads);
}
