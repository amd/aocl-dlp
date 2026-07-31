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

#include <string.h>

#include "config/dlp_gemm_config.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "kernels/s8s8s32/dlp_gemm_packb_s8.h"
#include "kernels/s8s8s32/dlp_gemm_packb_s8s4.h"
#include "s8s8s32/dlp_gemm_reorder_s8.h"
#include "sys_utils/dlp_gemm_sys.h"
#include "threading/dlp_gemm_thread_utils.h"

#ifdef DLP_ENABLE_OPENMP
#include <omp.h>
#endif

/*
 * Reorder worker for the s8s4 symmetric-quantized GEMM.
 *
 * The input matrix B is a signed 4-bit (nibble-packed) weight matrix addressed
 * via element strides (b->rs, b->cs) measured in nibble units.
 *
 * Compact reordered buffer layout (bytes):
 *   [0 .. compact_weight_bytes)                       : s4 packed weights
 *   [compact_weight_bytes .. + colsum_bytes)          : int32 per-group colsums
 * where
 *   compact_weight_bytes = (k_updated * n_updated) / 2
 *   colsum_bytes         = num_groups * n_updated * sizeof(int32_t)
 *   k_updated            = round_up(k, 4)
 *   n_updated            = round_up(n, 16)
 *
 * Because k_updated is a multiple of 4 and n_updated a multiple of 16, the
 * product is a multiple of 64, so compact_weight_bytes is a multiple of 32 and
 * the trailing int32 column sums remain naturally aligned.
 *
 * On zen4 the reorder is FUSED, mirroring the runtime-PACK path: each NR-wide
 * weight strip is packed to the s8 VNNI-4 layout directly from the raw s4 B via
 * the fused widen-on-load packers (row-major for transb='N', col-major for
 * transb='T'), its per-group column sums are written straight into the compact
 * colsum region, and the strip's s8 bytes are then compressed 2:1 into the
 * compact weight region. Only a small per-thread NR x KC s8 strip scratch is
 * used -- there is no full-matrix widen copy of B, no full s8 packed buffer,
 * and no reuse of the generic s8 reorder worker.
 *
 * Off-zen4 builds fall back to the portable scalar path: widen all of B to a
 * transient s8 matrix, reuse the s8 sym-quant reorder worker, then compress.
 */
#ifdef DLP_KERNELS_ZEN4
dlp_clsc_err_t
dlp_reorderb_nr64_s8s4s32o32(dlp_gemm_obj_t*  b,
                             dlp_gemm_obj_t*  b_reorder,
                             dlp_rntm_t*      rntm,
                             dlp_gemm_cntx_t* lcntx,
                             md_t             group_size)
{
    md_t NC = lcntx->blksz.NC;
    md_t KC = lcntx->blksz.KC;
    md_t NR = lcntx->blksz.NR;

    // A quantization group must be processed entirely within one KC block so
    // that the per-panel strip offset (group * group_size) - pc stays >= 0 and
    // the group's int32 column sums land in a single panel. Keep KC aligned to
    // the group size:
    //  - if group_size > KC, grow KC up to group_size;
    //  - if group_size < KC but does not divide KC, shrink KC down to the
    //    largest multiple of group_size (so KC boundaries fall on group
    //    boundaries).
    // Without the shrink, a group straddling the KC boundary makes
    // (group * group_size) - pc negative for the first group of the trailing
    // panel, underflowing the strip_s8 scratch below and corrupting the heap.
    // The GEMM 5-loop applies the SAME adjustment so reorder and compute agree
    // (see dlp_gemm_s8s4s32.c and the s8s8 sym-quant path).
    if (group_size > KC) {
        KC = group_size;
    } else if ((KC % group_size) != 0) {
        KC = (KC / group_size) * group_size;
    }

    // Element strides measured in nibble units. transb='N' => cs_b==1 (row
    // contiguous), transb='T' => rs_b==1 (col contiguous).
    md_t rs_b = b->rs;
    md_t cs_b = b->cs;

    md_t n = b->width;
    md_t k = b->length;

    md_t num_groups = (k + group_size - 1) / group_size;

    md_t k_updated = dlp_make_multiple_of_n(k, 4);
    md_t n_updated = dlp_make_multiple_of_n(n, 16);

    md_t n_threads = rntm->num_threads;
    n_threads      = (n_threads > 0) ? n_threads : 1;

    uint8_t* weights_compact      = (uint8_t*)b_reorder->storage.aligned_buffer;
    msz_t    compact_weight_bytes = (msz_t)(k_updated * n_updated) / 2;
    int32_t* pack_b_column_sum =
        (int32_t*)(weights_compact + compact_weight_bytes);

    for (iter_t idx = 0; idx < num_groups * n_updated; idx++) {
        pack_b_column_sum[idx] = 0;
    }

    // Per-thread s8 strip scratch, allocated ONCE up front (outside the OpenMP
    // region) so allocation failure is handled here with a plain serial return
    // -- there is no in-region malloc, and therefore no shared error flag /
    // data race to reason about. Each thread works on its own [work_id *
    // strip_stride .. +strip_stride) slice.
    //
    // strip_stride = NR * round_up(KC, 4). kc0_updated = round_up(kc0, 4) can
    // exceed KC by up to 3 when KC was grown to a group_size that is not a
    // multiple of 4 (e.g. single-group case where group_size == k); sizing to
    // the padded stride avoids memset/packing overrun. NR is 64 and
    // round_up(KC, 4) a multiple of 4, so strip_stride is a multiple of 256 and
    // every page-aligned-base slice stays AVX-512 (64-byte) aligned.
    msz_t          strip_stride = (msz_t)NR * dlp_make_multiple_of_n(KC, 4);
    dlp_clsc_err_t pool_err;
    int8_t*        strip_s8_pool = (int8_t*)dlp_malloc_page_aligned(
        strip_stride * (msz_t)n_threads, &pool_err);
    if (strip_s8_pool == NULL) {
        return DLP_CLSC_FAILURE;
    }

#ifdef DLP_ENABLE_OPENMP
    _Pragma("omp parallel num_threads(n_threads)")
    {
        dlp_task_id_t thread_jc;
        thread_jc.n_way   = n_threads;
        thread_jc.work_id = omp_get_thread_num();
#else
    {
        dlp_task_id_t thread_jc;
        thread_jc.n_way   = 1;
        thread_jc.work_id = 0;
#endif
        // This thread's slice of the pre-allocated scratch pool. Holds one
        // packed NR-wide weight strip (round_up(nr0,16) columns x kc0_updated
        // rows) before compression.
        int8_t* strip_s8 =
            strip_s8_pool + (msz_t)thread_jc.work_id * strip_stride;
        {
            md_t jc_start, jc_end;
            dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

            for (iter_t jc = jc_start; jc < jc_end; jc += NC) {
                md_t nc0 = dlp_min((jc_end - jc), NC);

                md_t jc_cur_loop     = jc;
                md_t jc_cur_loop_rem = 0;
                md_t n_sub_updated   = 0;

                dlp_gemm_get_B_panel_reordered_start_offset_width(
                    jc, n, NC, dlp_get_packb_s8s8s32o32_min_NR(), &jc_cur_loop,
                    &jc_cur_loop_rem, &nc0, &n_sub_updated);

                for (iter_t pc = 0; pc < k; pc += KC) {
                    md_t kc0         = dlp_min((k - pc), KC);
                    md_t kc0_updated = dlp_make_multiple_of_n(kc0, 4);

                    md_t group_start = pc / group_size;
                    md_t group_end   = (pc + kc0 - 1) / group_size;

                    // s8-equivalent byte offset of this KCxNC block's weights.
                    md_t g_pc_base = (jc_cur_loop * k_updated)
                                     + (n_sub_updated * pc)
                                     + (jc_cur_loop_rem * kc0_updated);

                    for (iter_t jr = 0; jr < nc0; jr += NR) {
                        md_t nr0 = dlp_min((nc0 - jr), NR);

                        // Global s8 weight byte offset of this strip; strip
                        // spans round_up(nr0,16) columns x kc0_updated rows.
                        md_t g_base      = g_pc_base + (jr * kc0_updated);
                        md_t strip_cols  = dlp_make_multiple_of_n(nr0, 16);
                        md_t strip_bytes = strip_cols * kc0_updated;

                        int32_t* b_sum_ptr = pack_b_column_sum + jc + jr;
                        md_t     col0      = jc + jr;
                        md_t     rs_o, cs_o;

                        memset(strip_s8, 0, (size_t)strip_bytes);

                        if (nr0 < NR) {
                            md_t nr_mult_16 = (nr0 / 16) * 16;
                            md_t nr0_rem    = nr0 % 16;

                            if (nr_mult_16 > 0) {
                                md_t nr0_updated = nr_mult_16;
                                for (iter_t group = group_start;
                                     group <= group_end; group++) {
                                    md_t k_start =
                                        dlp_max(group * group_size, pc);
                                    md_t k_end =
                                        dlp_min(((group + 1) * group_size - 1),
                                                pc + kc0 - 1);
                                    md_t kg0 = k_end - k_start + 1;

                                    if (cs_b == 1) {
                                        dlp_packb_nr64_s8s4s32os32_row_major(
                                            strip_s8
                                                + ((group * group_size) - pc)
                                                      * nr0_updated,
                                            b_sum_ptr + (group * n_updated),
                                            (const uint8_t*)
                                                b->storage.aligned_buffer,
                                            rs_b, k_start, col0, nr_mult_16,
                                            kg0, &rs_o, &cs_o);
                                    } else {
                                        dlp_packb_nr64_s8s4s32os32_col_major(
                                            strip_s8
                                                + ((group * group_size) - pc)
                                                      * nr0_updated,
                                            b_sum_ptr + (group * n_updated),
                                            (const uint8_t*)
                                                b->storage.aligned_buffer,
                                            cs_b, k_start, col0, nr_mult_16,
                                            kg0, &rs_o, &cs_o);
                                    }
                                }
                            }

                            if (nr0_rem > 0) {
                                md_t    nr0_updated = 16;
                                int8_t* strip_rem =
                                    strip_s8 + (nr_mult_16 * kc0_updated);
                                for (iter_t group = group_start;
                                     group <= group_end; group++) {
                                    md_t k_start =
                                        dlp_max(group * group_size, pc);
                                    md_t k_end =
                                        dlp_min(((group + 1) * group_size - 1),
                                                pc + kc0 - 1);
                                    md_t kg0 = k_end - k_start + 1;

                                    if (cs_b == 1) {
                                        dlp_packb_nr64_s8s4s32os32_row_major(
                                            strip_rem
                                                + ((group * group_size) - pc)
                                                      * nr0_updated,
                                            b_sum_ptr + nr_mult_16
                                                + (group * n_updated),
                                            (const uint8_t*)
                                                b->storage.aligned_buffer,
                                            rs_b, k_start, col0 + nr_mult_16,
                                            nr0_rem, kg0, &rs_o, &cs_o);
                                    } else {
                                        dlp_packb_nr64_s8s4s32os32_col_major(
                                            strip_rem
                                                + ((group * group_size) - pc)
                                                      * nr0_updated,
                                            b_sum_ptr + nr_mult_16
                                                + (group * n_updated),
                                            (const uint8_t*)
                                                b->storage.aligned_buffer,
                                            cs_b, k_start, col0 + nr_mult_16,
                                            nr0_rem, kg0, &rs_o, &cs_o);
                                    }
                                }
                            }
                        } else {
                            md_t nr0_updated = NR;
                            for (iter_t group = group_start; group <= group_end;
                                 group++) {
                                md_t k_start = dlp_max(group * group_size, pc);
                                md_t k_end =
                                    dlp_min(((group + 1) * group_size - 1),
                                            pc + kc0 - 1);
                                md_t kg0 = k_end - k_start + 1;

                                if (cs_b == 1) {
                                    dlp_packb_nr64_s8s4s32os32_row_major(
                                        strip_s8
                                            + ((group * group_size) - pc)
                                                  * nr0_updated,
                                        b_sum_ptr + (group * n_updated),
                                        (const uint8_t*)
                                            b->storage.aligned_buffer,
                                        rs_b, k_start, col0, NR, kg0, &rs_o,
                                        &cs_o);
                                } else {
                                    dlp_packb_nr64_s8s4s32os32_col_major(
                                        strip_s8
                                            + ((group * group_size) - pc)
                                                  * nr0_updated,
                                        b_sum_ptr + (group * n_updated),
                                        (const uint8_t*)
                                            b->storage.aligned_buffer,
                                        cs_b, k_start, col0, NR, kg0, &rs_o,
                                        &cs_o);
                                }
                            }
                        }

                        // Compress this strip's s8 weights 2:1 into the compact
                        // buffer. g_base is even (product of multiples of 4/16)
                        // so the halved destination offset is exact.
                        dlp_cvt_s8_to_s4_linear_avx512(weights_compact
                                                           + (g_base >> 1),
                                                       strip_s8, strip_bytes);
                    }
                }
                dlp_gemm_adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
            }
        }
    }

    dlp_free_page_aligned(strip_s8_pool);

    b_reorder->rs   = NR * 4;
    b_reorder->cs   = NR;
    b_reorder->mtag = REORDERED;
    return DLP_CLSC_SUCCESS;
}
#else  /* !DLP_KERNELS_ZEN4 : portable scalar fallback */
dlp_clsc_err_t
dlp_reorderb_nr64_s8s4s32o32(dlp_gemm_obj_t*  b,
                             dlp_gemm_obj_t*  b_reorder,
                             dlp_rntm_t*      rntm,
                             dlp_gemm_cntx_t* lcntx,
                             md_t             group_size)
{
    md_t n = b->width;
    md_t k = b->length;

    md_t k_updated = dlp_make_multiple_of_n(k, 4);
    md_t n_updated = dlp_make_multiple_of_n(n, 16);

    md_t num_groups = (k + group_size - 1) / group_size;

    msz_t s8_weight_bytes = (msz_t)sizeof(int8_t) * k_updated * n_updated;
    msz_t colsum_bytes    = (msz_t)num_groups * n_updated * sizeof(int32_t);

    dlp_clsc_err_t ret_err;

    // 1) Widen raw s4 B -> transient contiguous row-major s8 (k x n).
    int8_t* b_s8 = (int8_t*)dlp_malloc_page_aligned(
        (msz_t)k * n * sizeof(int8_t), &ret_err);
    if (b_s8 == NULL) {
        return DLP_CLSC_FAILURE;
    }
    dlp_cvt_s4_to_s8_matrix(b_s8, (const uint8_t*)b->storage.aligned_buffer, k,
                            n, b->rs, b->cs);

    // 2) Transient s8 reorder buffer (weights + colsums).
    int8_t* s8_reorder = (int8_t*)dlp_malloc_page_aligned(
        s8_weight_bytes + colsum_bytes, &ret_err);
    if (s8_reorder == NULL) {
        dlp_free_page_aligned(b_s8);
        return DLP_CLSC_FAILURE;
    }

    dlp_gemm_obj_t b_tmp         = *b;
    b_tmp.storage.aligned_buffer = b_s8;
    b_tmp.rs                     = n;
    b_tmp.cs                     = 1;
    b_tmp.width                  = n;
    b_tmp.length                 = k;

    dlp_gemm_obj_t b_reorder_tmp         = *b_reorder;
    b_reorder_tmp.storage.aligned_buffer = s8_reorder;

    dlp_reorderb_nr64_s8s8s32o32_sym_quant(&b_tmp, &b_reorder_tmp, rntm, lcntx,
                                           group_size);

    // 3) Compress packed s8 weights 2:1 -> nibbles, then copy colsums.
    uint8_t* dst                  = (uint8_t*)b_reorder->storage.aligned_buffer;
    msz_t    compact_weight_bytes = (msz_t)(s8_weight_bytes / 2);

    dlp_cvt_s8_to_s4_linear(dst, s8_reorder, (md_t)s8_weight_bytes);
    memcpy(dst + compact_weight_bytes, s8_reorder + s8_weight_bytes,
           colsum_bytes);

    b_reorder->rs   = b_reorder_tmp.rs;
    b_reorder->cs   = b_reorder_tmp.cs;
    b_reorder->mtag = REORDERED;

    dlp_free_page_aligned(s8_reorder);
    dlp_free_page_aligned(b_s8);
    return DLP_CLSC_SUCCESS;
}
#endif // DLP_KERNELS_ZEN4
