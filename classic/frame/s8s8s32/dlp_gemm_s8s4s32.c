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
#include "dlp_gemm_5loop_interface_apis.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "kernels/dlp_gemm_kernels.h"
#include "kernels/s8s8s32/dlp_gemm_packb_s8.h"
#include "kernels/s8s8s32/dlp_gemm_packb_s8s4.h"
#include "kernels/u8s8s32/dlp_gemm_packa.h"
#include "sys_utils/dlp_gemm_sys.h"
#include "threading/dlp_gemm_thread_utils.h"

#ifdef DLP_KERNELS_ZEN4
#include <immintrin.h>

// Transpose a 16x16 int8 tile with SSE2 (no lane crossing). Mirrors the helper
// in dlp_gemm_s8s8s32_sym_quant.c; used by the n == 1 GEMV path to gather a
// transposed (packed) A into a plain row-major buffer.
static inline void
dlp_transpose16x16_i8_s8s4(const int8_t* src,
                           md_t          src_rs,
                           int8_t*       dst,
                           md_t          dst_rs)
{
    __m128i x[16];
    for (int i = 0; i < 16; ++i) {
        x[i] = _mm_loadu_si128((const __m128i*)(src + i * src_rs));
    }
    __m128i b[16];
    for (int i = 0; i < 8; ++i) {
        b[2 * i]     = _mm_unpacklo_epi8(x[2 * i], x[2 * i + 1]);
        b[2 * i + 1] = _mm_unpackhi_epi8(x[2 * i], x[2 * i + 1]);
    }
    static const int p16[8][2] = {
        { 0, 2 },  { 1, 3 },  { 4, 6 },   { 5, 7 },
        { 8, 10 }, { 9, 11 }, { 12, 14 }, { 13, 15 }
    };
    __m128i c[16];
    for (int i = 0; i < 8; ++i) {
        c[2 * i]     = _mm_unpacklo_epi16(b[p16[i][0]], b[p16[i][1]]);
        c[2 * i + 1] = _mm_unpackhi_epi16(b[p16[i][0]], b[p16[i][1]]);
    }
    static const int p32[8][2] = {
        { 0, 4 },  { 1, 5 },  { 2, 6 },   { 3, 7 },
        { 8, 12 }, { 9, 13 }, { 10, 14 }, { 11, 15 }
    };
    __m128i d[16];
    for (int i = 0; i < 8; ++i) {
        d[2 * i]     = _mm_unpacklo_epi32(c[p32[i][0]], c[p32[i][1]]);
        d[2 * i + 1] = _mm_unpackhi_epi32(c[p32[i][0]], c[p32[i][1]]);
    }
    static const int p64[8][2] = { { 0, 8 },  { 1, 9 },  { 2, 10 }, { 3, 11 },
                                   { 4, 12 }, { 5, 13 }, { 6, 14 }, { 7, 15 } };
    for (int i = 0; i < 8; ++i) {
        _mm_storeu_si128((__m128i*)(dst + (2 * i) * dst_rs),
                         _mm_unpacklo_epi64(d[p64[i][0]], d[p64[i][1]]));
        _mm_storeu_si128((__m128i*)(dst + (2 * i + 1) * dst_rs),
                         _mm_unpackhi_epi64(d[p64[i][0]], d[p64[i][1]]));
    }
}

// Gather a (possibly transposed / strided) source A block into a plain
// row-major (mc0 x k) buffer with k contiguous, as required by the n == 1 GEMV
// micro-kernel. Mirrors the helper in dlp_gemm_s8s8s32_sym_quant.c.
static void
dlp_gather_a_rowmajor_s8_s8s4(
    int8_t* dst, const int8_t* a_src, md_t mc0, md_t k, md_t rs_a, md_t cs_a)
{
    if (cs_a == 1) {
        for (md_t r = 0; r < mc0; ++r) {
            memcpy(dst + r * k, a_src + r * rs_a, (size_t)k);
        }
        return;
    }
    if (rs_a == 1) {
        const md_t r16 = (mc0 / 16) * 16;
        const md_t k16 = (k / 16) * 16;
        for (md_t r0 = 0; r0 < r16; r0 += 16) {
            for (md_t kk0 = 0; kk0 < k16; kk0 += 16) {
                dlp_transpose16x16_i8_s8s4(a_src + r0 + kk0 * cs_a, cs_a,
                                           dst + r0 * k + kk0, k);
            }
            for (md_t r = r0; r < r0 + 16; ++r) {
                for (md_t kk = k16; kk < k; ++kk) {
                    dst[r * k + kk] = a_src[r + kk * cs_a];
                }
            }
        }
        for (md_t r = r16; r < mc0; ++r) {
            for (md_t kk = 0; kk < k; ++kk) {
                dst[r * k + kk] = a_src[r + kk * cs_a];
            }
        }
        return;
    }
    for (md_t r = 0; r < mc0; ++r) {
        for (md_t kk = 0; kk < k; ++kk) {
            dst[r * k + kk] = a_src[r * rs_a + kk * cs_a];
        }
    }
}

/*
 * Dedicated s8s4 sym-quant GEMV driver (m == 1 or n == 1), mirroring
 * dlp_gemv_rowvar_s8s8s32o32_sym_quant. The s8s4 reorder mirrors the s8s8
 * reorder, so the same specialized GEMV micro-kernels
 * (dlp_gemv_{n,m}_one_s8s8s32os32_sym_quant) are reused:
 *   * n == 1: the reorder already emitted the tight s8 column + per-group
 *     column sums (identical to s8s8), consumed directly.
 *   * m == 1: the compact s4 weights of each NC-wide B panel are widened once
 *     to the equivalent s8 reorder panel (per thread, in a small scratch) and
 *     handed to the s8s8 m == 1 kernel. The (uncompressed) per-group column
 *     sums that trail the compact weights are consumed in place.
 * Callers must guarantee mtag_b == REORDERED and that k / (unadjusted) KC are
 * divisible by group_size (both enforced by dlp_gemm_rowvar_s8s4s32o32).
 */
DLP_GEMV2(int8_t, int8_t, int32_t, s8s4s32o32)
{
    (void)rntm; /* Threading handled via thread object, not rntm. */
    md_t NC = lcntx->blksz.NC;
    md_t KC = lcntx->blksz.KC;
    md_t MC = lcntx->blksz.MC;
    md_t NR = lcntx->blksz.NR;

    // Reordered B is mandatory for this path (guaranteed by the caller).
    if (mtag_b != REORDERED) {
        return;
    }

    // Keep KC aligned to the group size so a quantization group never straddles
    // a KC boundary (same adjustment as the s8s8 sym-quant GEMV and the reorder
    // function).
    if (grp_post_op_list->group_size > KC) {
        KC = grp_post_op_list->group_size;
    } else if ((KC % grp_post_op_list->group_size) != 0) {
        KC = (KC / grp_post_op_list->group_size) * grp_post_op_list->group_size;
    }

    int8_t* a_use    = (int8_t*)a;
    md_t    rs_a_use = rs_a;
    md_t    cs_a_use = cs_a;

    int8_t* b_use    = (int8_t*)b;
    md_t    rs_b_use = rs_b;
    md_t    cs_b_use = cs_b;

    float* c_use = NULL;

    dlp_gemm_post_op_attr post_ops_attr = { 0 };

    post_ops_attr.c_stor_type       = c_downscale;
    post_ops_attr.rs_c_downscale    = rs_c;
    post_ops_attr.cs_c_downscale    = cs_c;
    post_ops_attr.is_first_k        = TRUE;
    post_ops_attr.is_last_k         = TRUE;
    post_ops_attr.b_sum_offset      = 0;
    post_ops_attr.b_col_sum_vec     = NULL;
    post_ops_attr.b_col_sum_vec_s16 = NULL;

    if (c_downscale < DLP_F32) {
        post_ops_attr.buf_downscale = c;
    } else {
        post_ops_attr.buf_downscale = NULL;
    }

    msz_t   mem_a_size_req = 0;
    int8_t* pack_a_buffer  = NULL;

    // Per-thread transient s8 scratch for the m == 1 path: it holds one widened
    // NC-wide reorder panel handed to the s8 GEMV kernel (n == 1 needs none).
    int8_t* b_panel_widen = NULL;

    dlp_gemm_grp_post_op_attr grp_post_ops_attr = { 0 };

    md_t group_size = grp_post_op_list->group_size;

    grp_post_ops_attr.a_scale_factor     = grp_post_op_list->a_scale_factor;
    grp_post_ops_attr.a_scale_factor_len = grp_post_op_list->a_scale_factor_len;
    grp_post_ops_attr.b_scale_factor     = grp_post_op_list->b_scale_factor;
    grp_post_ops_attr.b_scale_factor_len = grp_post_op_list->b_scale_factor_len;
    grp_post_ops_attr.a_zp               = grp_post_op_list->a_zp;
    grp_post_ops_attr.b_zp               = grp_post_op_list->b_zp;
    grp_post_ops_attr.a_zp_len           = grp_post_op_list->a_zp_len;
    grp_post_ops_attr.b_zp_len           = grp_post_op_list->b_zp_len;
    grp_post_ops_attr.group_size         = group_size;
    grp_post_ops_attr.sf_stor_type       = grp_post_op_list->sf_stor_type;
    grp_post_ops_attr.zp_stor_type       = grp_post_op_list->zp_stor_type;

    md_t num_groups = (k + group_size - 1) / group_size;
    grp_post_ops_attr.a_grp_mul =
        (grp_post_op_list->a_scale_factor_dim == DLP_PARAM_DIM_PER_TOKEN) ? 0
                                                                          : 1;
    grp_post_ops_attr.b_grp_mul =
        (grp_post_op_list->b_scale_factor_dim == DLP_PARAM_DIM_PER_CHANNEL) ? 0
                                                                            : 1;
    grp_post_ops_attr.grp_post_op_lda =
        (grp_post_op_list->a_scale_factor_dim == DLP_PARAM_DIM_PER_TOKEN)
            ? 1
            : num_groups;
    grp_post_ops_attr.grp_post_op_ldb = n;

    dlp_task_id_t thread_jc;
    dlp_task_id_t thread_ic;

    dlp_gemm_gen_dlp_task_ids(thread, &thread_jc, &thread_ic);

    if (n == 1) {
        // Increased MR from 6 to 16 to make use of 32 ZMM registers.
        md_t MR = 16;

        // n == 1: the reorder emitted the tight s8 column followed by the
        // per-group column sums, exactly like s8s8 -- consumed directly.
        post_ops_attr.b_col_sum_vec =
            (int32_t*)(b + dlp_gemm_col_sum_byte_offset(k));

        md_t ic_start, ic_end;
        thread_ic.n_way   = (thread_ic.n_way == 1) ? (thread->n_threads)
                                                   : (thread_ic.n_way);
        thread_ic.work_id = thread->tid;
        dlp_thread_task_range(&thread_ic, m, MR, FALSE, &ic_start, &ic_end);

        grp_post_ops_attr.grp_post_op_k = 0;
        for (iter_t ic = ic_start; ic < ic_end; ic += MC) {
            grp_post_ops_attr.grp_post_op_i = ic;

            md_t mc0 = dlp_min((ic_end - ic), MC);

            const int8_t* a_ic = a + ic * rs_a;
            c_use              = c + ic * rs_c;

            post_ops_attr.post_op_c_i    = ic;
            post_ops_attr.post_op_c_j    = 0;
            post_ops_attr.rs_c_downscale = rs_c;

            rs_a_use = rs_a;
            cs_a_use = cs_a;
            a_use    = (int8_t*)a_ic;

            if (mtag_a == PACK) {
                // The n == 1 kernel reads A as plain row-major with a
                // contiguous k dimension; it cannot consume the VNNI-packed
                // layout. Gather A into a row-major (mc0 x k) scratch buffer.
                mem_a_size_req = sizeof(int8_t) * mc0 * k;

                if (pack_a_buffer == NULL) {
                    dlp_clsc_err_t ret_err;
                    pack_a_buffer =
                        dlp_malloc_page_aligned(mem_a_size_req, &ret_err);
                    if (pack_a_buffer == NULL) {
                        return;
                    }
                }

                dlp_gather_a_rowmajor_s8_s8s4(pack_a_buffer, a + (rs_a * ic),
                                              mc0, k, rs_a, cs_a);
                a_use    = pack_a_buffer;
                rs_a_use = k;
                cs_a_use = 1;
            }

            dlp_gemv_n_one_s8s8s32os32_sym_quant(
                mc0, k, a_use, rs_a_use, cs_a_use, mtag_a, b_use, rs_b_use,
                cs_b_use, mtag_b, c_use, rs_c, cs_c, alpha, beta, MR, KC,
                grp_post_ops_attr, post_op_list, &post_ops_attr);
        }

        if ((mtag_a == PACK) && (pack_a_buffer != NULL)) {
            dlp_free_page_aligned(pack_a_buffer);
        }
    } else {
        // m == 1: parallelize over N (JC loop).
        md_t gemm_MR = lcntx->blksz.MR;

        md_t jc_start, jc_end;
        thread_jc.n_way   = (thread_jc.n_way == 1) ? (thread->n_threads)
                                                   : (thread_jc.n_way);
        thread_jc.work_id = thread->tid;
        dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

        md_t packb_min_NR = dlp_get_packb_s8s8s32o32_min_NR();

        // kc needs to be a multiple of 4 for vpdpbusd, and nc a multiple of 16;
        // reorder-buffer offsets are computed in these padded units.
        md_t k_updated = dlp_make_multiple_of_n(k, 4);
        md_t n_updated = dlp_make_multiple_of_n(n, 16);

        // Compact s4 weight region (bytes) preceding the per-group column sums.
        msz_t compact_weight_bytes = (msz_t)(k_updated * n_updated) / 2;

        rs_a_use = rs_a;
        cs_a_use = 4;

        if (mtag_a == PACK) {
            mem_a_size_req = sizeof(uint8_t) * k;

            if (pack_a_buffer == NULL) {
                dlp_clsc_err_t ret_err;
                pack_a_buffer =
                    dlp_malloc_page_aligned(mem_a_size_req, &ret_err);
                if (pack_a_buffer == NULL) {
                    return;
                }
            }

            ((packa_s32)lcntx->packa_fun_ptr)((uint8_t*)pack_a_buffer,
                                              (uint8_t*)a, rs_a, cs_a, 1, k,
                                              &rs_a_use, &cs_a_use);

            dlp_get_packa_strides_mfringe_u8s8s32os32(rs_a, cs_a, &rs_a_use,
                                                      &cs_a_use, gemm_MR, 1);

            a_use = pack_a_buffer;
        }

        // Widening scratch: at most one NC-wide s8 reorder panel
        // (n_sub_updated <= NC) times the padded k stride.
        {
            md_t  nc_pad      = dlp_make_multiple_of_n(NC, packb_min_NR);
            msz_t widen_bytes = (msz_t)nc_pad * k_updated * sizeof(int8_t);

            dlp_clsc_err_t ret_err;
            b_panel_widen = dlp_malloc_page_aligned(widen_bytes, &ret_err);
            if (b_panel_widen == NULL) {
                if ((mtag_a == PACK) && (pack_a_buffer != NULL)) {
                    dlp_free_page_aligned(pack_a_buffer);
                }
                return;
            }
        }

        grp_post_ops_attr.grp_post_op_k = 0;
        for (iter_t jc = jc_start; jc < jc_end; jc += NC) {
            grp_post_ops_attr.grp_post_op_j = jc;

            md_t nc0 = dlp_min((jc_end - jc), NC);
            c_use    = c + jc;

            md_t jc_cur_loop     = jc;
            md_t jc_cur_loop_rem = 0;
            md_t n_sub_updated   = 0;

            dlp_gemm_get_B_panel_reordered_start_offset_width(
                jc, n, NC, packb_min_NR, &jc_cur_loop, &jc_cur_loop_rem, &nc0,
                &n_sub_updated);

            // Widen this compact s4 panel back to the s8 reorder layout. The
            // panel base is (jc_cur_loop * k_updated) s8 elements into the
            // buffer; the compact buffer is that layout halved, so the offset
            // is provably even and /2 indexes it exactly. n_sub_updated *
            // k_updated s8 elements span the full panel the kernel walks.
            md_t panel_s8_base = jc_cur_loop * k_updated;
            md_t panel_elems   = n_sub_updated * k_updated;
            dlp_cvt_s4_to_s8_linear_avx512(
                b_panel_widen, ((const uint8_t*)b) + (panel_s8_base / 2),
                panel_elems);
            b_use = b_panel_widen;

            dlp_gemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);

            // Column sums live (uncompressed) after the compact weights.
            post_ops_attr.b_col_sum_vec =
                ((int32_t*)(((const int8_t*)b) + compact_weight_bytes)) + jc;
            grp_post_ops_attr.grp_post_op_sum_ld = n_updated;

            post_ops_attr.post_op_c_i    = 0;
            post_ops_attr.post_op_c_j    = jc;
            post_ops_attr.rs_c_downscale = rs_c;
            post_ops_attr.b_sum_offset   = 0;

            dlp_gemv_m_one_s8s8s32os32_sym_quant(
                nc0, k, a_use, rs_a_use, cs_a_use, mtag_a, b_use, rs_b_use,
                cs_b_use, mtag_b, c_use, rs_c, cs_c, alpha, beta, NR, KC,
                n_sub_updated, jc_cur_loop_rem, grp_post_ops_attr, post_op_list,
                &post_ops_attr);

            dlp_gemm_adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }

        if (b_panel_widen != NULL) {
            dlp_free_page_aligned(b_panel_widen);
        }
        if ((mtag_a == PACK) && (pack_a_buffer != NULL)) {
            dlp_free_page_aligned(pack_a_buffer);
        }
    }
}
#endif

/*
 * Forked 5-loop driver for the s8s4 symmetric-quantized GEMM.
 *
 * A is a standard s8 activation matrix. B is a signed-4-bit (nibble-packed)
 * weight matrix supplied in one of two forms:
 *
 *   REORDERED (mtag_b == REORDERED): the compact s4 reordered layout produced
 *     by dlp_reorderb_nr64_s8s4s32o32, i.e. the s8 VNNI-4 packed
 *     weights compressed 2:1 to nibbles, followed by the per-group int32 column
 *     sums. For each NR weight panel the compact s4 slice is transiently
 * widened back to s8 into a per-thread scratch buffer and handed to the shared
 * s8s8 micro-kernel. All reorder-buffer offsets are computed in s8-equivalent
 *     units and halved to index the compact nibble buffer; every such offset is
 *     provably even, so the halving is exact.
 *
 *   PACK (mtag_b == PACK): the raw nibble-packed s4 B (addressed via rs_b/cs_b,
 *     so both transb='N' and transb='T' are handled). B is packed on the fly,
 *     exactly like the s8s8 sym-quant PACK path: the packing is parallelized
 *     across the ic threads of each jc work group. Each packing thread emits
 * the s8 VNNI-4 packed layout + per-group column sums for the sub-block it owns
 *     directly from the raw s4, via fused widen-on-load packers (row-major for
 *     transb='N', col-major for transb='T') -- no intermediate s8 scratch. The
 *     compute micro-kernel is unchanged.
 */
DLP_GEMM_5LOOP_UNIFIED(int8_t, int8_t, int32_t, float, s8s4s32o32, const)
{
    // Extract operations from bundle into local variables.
    DLP_GEMM_OPS_EXTRACT(ops);
    (void)pre_op_list;
    (void)rntm;

    md_t NC = lcntx->blksz.NC;
    md_t KC = lcntx->blksz.KC;
    md_t MC = lcntx->blksz.MC;
    md_t NR = lcntx->blksz.NR;
    md_t MR = lcntx->blksz.MR;

    // Only reordered or runtime-packed B is supported for the s8s4 path.
    if (mtag_b != REORDERED && mtag_b != PACK) {
        return;
    }

#ifdef DLP_KERNELS_ZEN4
    // GEMV fast path: for m == 1 or n == 1 hand off to the dedicated s8s4
    // sym-quant GEMV driver, mirroring the s8s8 sym-quant frame. Quantized GEMV
    // is supported iff k and the (unadjusted) KC are divisible by group_size
    // and B is reordered; otherwise fall through to the GEMM path.
    if ((grp_post_op_list->group_size > 0)
        && ((k % grp_post_op_list->group_size) == 0)
        && ((KC % grp_post_op_list->group_size) == 0) && (mtag_b == REORDERED)
        && ((m == 1) || (n == 1))) {

        dlp_gemv_rowvar_s8s4s32o32(m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b,
                                   cs_b, mtag_b, c, rs_c, cs_c, alpha, beta,
                                   rntm, thread, lcntx, grp_post_op_list,
                                   post_op_list, c_downscale);

        return;
    }

    // A quantization group must be processed entirely within one KC block so
    // that its int32 accumulation completes before the group scale is applied
    // and the per-panel pack offset (group * group_size) - pc stays >= 0. Keep
    // KC aligned to the group size:
    //  - if group_size > KC, grow KC up to group_size;
    //  - if group_size < KC but does not divide KC, shrink KC down to the
    //    largest multiple of group_size (so KC boundaries fall on group
    //    boundaries).
    // Without the shrink, a group straddling a KC boundary is split across two
    // pc iterations (mis-scaled) and the trailing panel's first group yields a
    // negative pack offset -> out-of-bounds write. The reorder function applies
    // the SAME adjustment so the reordered-B layout matches GEMM execution
    // (mirrors the s8s8 sym-quant path).
    if (grp_post_op_list->group_size > KC) {
        KC = grp_post_op_list->group_size;
    } else if ((KC % grp_post_op_list->group_size) != 0) {
        KC = (KC / grp_post_op_list->group_size) * grp_post_op_list->group_size;
    }

    // Strides are updated based on matrix packing/reordering.
    const int8_t* a_use          = NULL;
    md_t          rs_a_use       = rs_a;
    md_t          cs_a_use       = cs_a;
    md_t          a_block_stride = 0;

    const int8_t* b_use    = NULL;
    md_t          rs_b_use = 0;
    md_t          cs_b_use = 0;

    float* c_use_jc       = NULL;
    float* c_use_ic       = NULL;
    md_t   rs_c_use       = rs_c;
    md_t   rs_c_downscale = rs_c;

    // Pack buffer for A.
    int8_t* pack_a_buffer_s8s4s32o32 = NULL;
    msz_t   mem_a_size_req           = 0;

    // Pack buffer for B (PACK path only, shared across the jc work group).
    int8_t* pack_b_buffer_s8s4s32o32 = NULL;
    msz_t   mem_b_size_req           = 0;

    md_t packb_min_NR = dlp_get_packb_s8s8s32o32_min_NR();

    // Temporary buffer for C accumulation when downscaling is required.
    float* temp_scal_c_buffer_s8s4s32o32 = NULL;
    msz_t  mem_scale_c_size_req          = 0;

    // Per-thread transient s8 scratch, REORDERED path only: it holds one
    // widened NR-wide weight panel handed to the s8 micro-kernel. The PACK path
    // packs directly from raw s4 via the fused widen-on-load packers (no
    // scratch).
    int8_t* b_panel_s8 = NULL;
    // kc0_updated = round_up(kc0, 4) can exceed KC by up to 3 when KC was grown
    // to a group_size that is not a multiple of 4 (e.g. single-group case where
    // group_size == k). Size the scratch to the padded stride to avoid overrun.
    msz_t b_panel_size =
        (msz_t)NR * dlp_make_multiple_of_n(KC, 4) * sizeof(int8_t);

    // kc needs to be a multiple of 4 so that it can be used with vpdpbusd
    // instruction. Padding is added when this is not satisfied, so the k offset
    // used for reordered buffer needs to be updated.
    md_t k_updated = dlp_make_multiple_of_n(k, 4);
    md_t n_updated = dlp_make_multiple_of_n(n, 16);

    // Compact s4 weight region size (in bytes) preceding the column sums.
    msz_t compact_weight_bytes = (msz_t)((k_updated * n_updated) / 2);

    bool is_last_k  = FALSE;
    bool is_first_k = FALSE;

    dlp_gemm_post_op_attr     post_ops_attr     = { 0 };
    dlp_gemm_grp_post_op_attr grp_post_ops_attr = { 0 };

    post_ops_attr.c_stor_type       = c_downscale;
    post_ops_attr.rs_c_downscale    = rs_c;
    post_ops_attr.cs_c_downscale    = cs_c;
    post_ops_attr.b_sum_offset      = 0;
    post_ops_attr.b_col_sum_vec     = NULL;
    post_ops_attr.b_col_sum_vec_s16 = NULL;

    if (c_downscale < DLP_F32) {
        post_ops_attr.buf_downscale = c;
    } else {
        post_ops_attr.buf_downscale = NULL;
    }

    md_t group_size = grp_post_op_list->group_size;

    grp_post_ops_attr.a_scale_factor     = grp_post_op_list->a_scale_factor;
    grp_post_ops_attr.a_scale_factor_len = grp_post_op_list->a_scale_factor_len;
    grp_post_ops_attr.b_scale_factor     = grp_post_op_list->b_scale_factor;
    grp_post_ops_attr.b_scale_factor_len = grp_post_op_list->b_scale_factor_len;
    grp_post_ops_attr.a_zp               = grp_post_op_list->a_zp;
    grp_post_ops_attr.b_zp               = grp_post_op_list->b_zp;
    grp_post_ops_attr.a_zp_len           = grp_post_op_list->a_zp_len;
    grp_post_ops_attr.b_zp_len           = grp_post_op_list->b_zp_len;
    grp_post_ops_attr.group_size         = group_size;
    grp_post_ops_attr.sf_stor_type       = grp_post_op_list->sf_stor_type;
    grp_post_ops_attr.zp_stor_type       = grp_post_op_list->zp_stor_type;

    // Derive per-matrix group strides from each matrix's scale-factor dim.
    //   A PER_TOKEN:   A has one scale per row    (lda=1, a_grp_mul=0).
    //   B PER_CHANNEL: B has one scale per column (b_grp_mul=0).
    //   otherwise:     stride by group (legacy per-group behaviour).
    // Mirrors dlp_gemm_s8s8s32_sym_quant.c; s8s4 reuses the same sym-quant
    // micro-kernel, which reads a_grp_mul/b_grp_mul/grp_post_op_lda.
    md_t num_groups = (k + group_size - 1) / group_size;
    grp_post_ops_attr.a_grp_mul =
        (grp_post_op_list->a_scale_factor_dim == DLP_PARAM_DIM_PER_TOKEN) ? 0
                                                                          : 1;
    grp_post_ops_attr.b_grp_mul =
        (grp_post_op_list->b_scale_factor_dim == DLP_PARAM_DIM_PER_CHANNEL) ? 0
                                                                            : 1;
    grp_post_ops_attr.grp_post_op_lda =
        (grp_post_op_list->a_scale_factor_dim == DLP_PARAM_DIM_PER_TOKEN)
            ? 1
            : num_groups;
    grp_post_ops_attr.grp_post_op_ldb = n;

    // Generate thrinfo objects for jc and ic loops.
    dlp_task_id_t thread_jc;
    dlp_task_id_t thread_ic;

    dlp_gemm_gen_dlp_task_ids(thread, &thread_jc, &thread_ic);

    md_t jc_start, jc_end;
    dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

    md_t ic_start, ic_end;
    dlp_thread_task_range(&thread_ic, m, MR, FALSE, &ic_start, &ic_end);

    // Allocate the per-thread transient widening scratch once (REORDERED only).
    if (mtag_b == REORDERED) {
        dlp_clsc_err_t ret_err;
        b_panel_s8 = dlp_malloc_page_aligned(b_panel_size, &ret_err);
        if (b_panel_s8 == NULL) {
            return;
        }
    }

    const uint8_t* b_compact = (const uint8_t*)b;

    for (iter_t jc = jc_start; jc < jc_end; jc += NC) {
        md_t nc0 = dlp_min((jc_end - jc), NC);

        md_t jc_cur_loop     = jc;
        md_t jc_cur_loop_rem = 0;
        md_t n_sub_updated   = 0;

        if (mtag_b == REORDERED) {
            dlp_gemm_get_B_panel_reordered_start_offset_width(
                jc, n, NC, packb_min_NR, &jc_cur_loop, &jc_cur_loop_rem, &nc0,
                &n_sub_updated);
        }

        if (c_downscale == DLP_F32) {
            c_use_jc = c + jc;
        } else if (c_downscale < DLP_F32) {
            if (k > KC) {
                md_t nc0_buf = dlp_min((jc_end - jc), NC);
                mem_scale_c_size_req =
                    sizeof(float) * nc0_buf * (ic_end - ic_start);

                if (temp_scal_c_buffer_s8s4s32o32 == NULL) {
                    dlp_clsc_err_t ret_err;
                    temp_scal_c_buffer_s8s4s32o32 =
                        dlp_malloc_page_aligned(mem_scale_c_size_req, &ret_err);
                }

                c_use_jc = (float*)temp_scal_c_buffer_s8s4s32o32;
            }
            rs_c_use = nc0;
        }

        // s8-equivalent base offset of this reordered B panel (in bytes).
        md_t b_s8_panel_base = (jc_cur_loop * k_updated);

        // Column-sum buffer for the PACK path (populated by the s8 packer at
        // pc == 0 and reused across the pc loop).
        int32_t* pack_b_column_sum = NULL;

        for (iter_t pc = 0; pc < k; pc += KC) {
            int32_t beta0 = (pc == 0) ? beta : 1;
            md_t    kc0   = dlp_min((k - pc), KC);

            grp_post_ops_attr.grp_post_op_k = pc;

            md_t kc0_updated = dlp_make_multiple_of_n(kc0, 4);

            is_first_k               = (pc == 0) ? (TRUE) : (FALSE);
            post_ops_attr.is_first_k = is_first_k;

            is_last_k               = ((pc + KC) >= k) ? (TRUE) : (FALSE);
            post_ops_attr.is_last_k = is_last_k;

            // s8-equivalent offset to the start of this KC sub-panel
            // (REORDERED path only).
            md_t b_s8_pc_base = 0;

            if (mtag_b == PACK) {
                // Runtime pack of the raw s4 B, parallelized across the ic
                // threads of this jc work group. Mirrors the s8s8 sym-quant
                // PACK path; the only s4-specific step is widening each owned
                // s4 sub-block to s8 before handing it to the s8 packer.
                md_t jc_work_id = thread_jc.work_id;

                md_t nc0_updated = dlp_make_multiple_of_n(nc0, packb_min_NR);

                md_t group_start  = pc / group_size;
                md_t group_end    = (pc + kc0 - 1) / group_size;
                md_t total_groups = (k + group_size - 1) / group_size;

                if (dlp_thread_am_ochief(&thread_ic)) {
                    mem_b_size_req =
                        sizeof(int8_t) * nc0_updated * kc0_updated
                        + (total_groups * nc0_updated * sizeof(int32_t));

                    if (pack_b_buffer_s8s4s32o32 == NULL) {
                        dlp_clsc_err_t ret_err;
                        pack_b_buffer_s8s4s32o32 =
                            dlp_malloc_page_aligned(mem_b_size_req, &ret_err);
                    }

                    thread->comm[jc_work_id].sent_object =
                        pack_b_buffer_s8s4s32o32;
                }

                dlp_atomic_barrier(thread_ic.ocomm_id,
                                   &thread->comm[jc_work_id]);

                pack_b_buffer_s8s4s32o32 =
                    (int8_t*)thread->comm[jc_work_id].sent_object;

                md_t jc_packb_start, jc_packb_end;
                dlp_thread_task_range(&thread_ic, nc0, NR, FALSE,
                                      &jc_packb_start, &jc_packb_end);

                if (pc == 0) {
                    pack_b_column_sum =
                        (int32_t*)(pack_b_buffer_s8s4s32o32
                                   + (sizeof(int8_t) * nc0_updated
                                      * kc0_updated));
                }

                if ((jc_packb_end > jc_packb_start)
                    && (jc_packb_start < (jc + nc0))) {
                    md_t nc0_pack = jc_packb_end - jc_packb_start;

                    if (pc == 0) {
                        for (iter_t group = 0; group < total_groups; group++) {
                            for (iter_t idx = jc_packb_start;
                                 idx < jc_packb_end; idx++) {
                                *(pack_b_column_sum + (group * nc0_updated)
                                  + idx) = 0;
                            }
                        }
                    }

                    for (iter_t jr = 0; jr < nc0_pack; jr += NR) {
                        md_t nr0 = dlp_min((nc0_pack - jr), NR);

                        int8_t* b_dst_jr =
                            pack_b_buffer_s8s4s32o32
                            + ((jc_packb_start + jr) * kc0_updated);
                        int32_t* b_sum_ptr =
                            pack_b_column_sum + (jc_packb_start + jr);

                        md_t col_sub = jc + jc_packb_start + jr;

                        // Both layouts are packed directly from the raw s4 B
                        // via fused widen-on-load packers (no s8 scratch):
                        // transb='N' (cs_b==1) uses the row-major packer,
                        // transb='T' (cs_b!=1) the col-major packer.
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
                                            b_dst_jr
                                                + ((group * group_size) - pc)
                                                      * nr0_updated,
                                            b_sum_ptr + (group * nc0_updated),
                                            (const uint8_t*)b, rs_b, k_start,
                                            col_sub, nr_mult_16, kg0, &rs_b_use,
                                            &cs_b_use);
                                    } else {
                                        dlp_packb_nr64_s8s4s32os32_col_major(
                                            b_dst_jr
                                                + ((group * group_size) - pc)
                                                      * nr0_updated,
                                            b_sum_ptr + (group * nc0_updated),
                                            (const uint8_t*)b, cs_b, k_start,
                                            col_sub, nr_mult_16, kg0, &rs_b_use,
                                            &cs_b_use);
                                    }
                                }
                                b_dst_jr += nr_mult_16 * kc0_updated;
                                b_sum_ptr += nr_mult_16;
                            }

                            if (nr0_rem > 0) {
                                md_t nr0_updated = 16;
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
                                            b_dst_jr
                                                + ((group * group_size) - pc)
                                                      * nr0_updated,
                                            b_sum_ptr + (group * nc0_updated),
                                            (const uint8_t*)b, rs_b, k_start,
                                            col_sub + nr_mult_16, nr0_rem, kg0,
                                            &rs_b_use, &cs_b_use);
                                    } else {
                                        dlp_packb_nr64_s8s4s32os32_col_major(
                                            b_dst_jr
                                                + ((group * group_size) - pc)
                                                      * nr0_updated,
                                            b_sum_ptr + (group * nc0_updated),
                                            (const uint8_t*)b, cs_b, k_start,
                                            col_sub + nr_mult_16, nr0_rem, kg0,
                                            &rs_b_use, &cs_b_use);
                                    }
                                }
                            }
                            continue;
                        }

                        md_t nr0_updated = NR;
                        for (iter_t group = group_start; group <= group_end;
                             group++) {
                            md_t k_start = dlp_max(group * group_size, pc);
                            md_t k_end = dlp_min(((group + 1) * group_size - 1),
                                                 pc + kc0 - 1);
                            md_t kg0   = k_end - k_start + 1;

                            if (cs_b == 1) {
                                dlp_packb_nr64_s8s4s32os32_row_major(
                                    b_dst_jr
                                        + ((group * group_size) - pc)
                                              * nr0_updated,
                                    b_sum_ptr + (group * nc0_updated),
                                    (const uint8_t*)b, rs_b, k_start, col_sub,
                                    NR, kg0, &rs_b_use, &cs_b_use);
                            } else {
                                dlp_packb_nr64_s8s4s32os32_col_major(
                                    b_dst_jr
                                        + ((group * group_size) - pc)
                                              * nr0_updated,
                                    b_sum_ptr + (group * nc0_updated),
                                    (const uint8_t*)b, cs_b, k_start, col_sub,
                                    NR, kg0, &rs_b_use, &cs_b_use);
                            }
                        }
                    }

                    rs_b_use = NR * 4;
                    cs_b_use = NR;
                } else {
                    dlp_gemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);
                }

                dlp_atomic_barrier(thread_ic.ocomm_id,
                                   &thread->comm[jc_work_id]);
                b_use = pack_b_buffer_s8s4s32o32;

                post_ops_attr.b_col_sum_vec          = pack_b_column_sum;
                grp_post_ops_attr.grp_post_op_sum_ld = nc0_updated;
            } else {
                // REORDERED: compact s4 weights, widened per-panel below.
                b_s8_pc_base = b_s8_panel_base + (n_sub_updated * pc)
                               + (jc_cur_loop_rem * kc0_updated);

                dlp_gemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);

                // Column sums live (uncompressed) after the compact weights.
                post_ops_attr.b_col_sum_vec =
                    ((int32_t*)(b_compact + compact_weight_bytes)) + jc;
                grp_post_ops_attr.grp_post_op_sum_ld = n_updated;
            }

            for (iter_t ic = ic_start; ic < ic_end; ic += MC) {
                md_t mc0 = dlp_min((ic_end - ic), MC);

                grp_post_ops_attr.grp_post_op_i = ic;

                if (c_downscale < DLP_F32) {
                    c_use_ic = dlp_offset_or_null_f32(
                        c_use_jc, (rs_c_use * (ic - ic_start)));
                } else {
                    c_use_ic = c_use_jc + (rs_c_use * ic);
                }

                if (mtag_a == PACK) {
                    mem_a_size_req = sizeof(uint8_t) * mc0 * kc0_updated;

                    if (pack_a_buffer_s8s4s32o32 == NULL) {
                        dlp_clsc_err_t ret_err;
                        pack_a_buffer_s8s4s32o32 =
                            dlp_malloc_page_aligned(mem_a_size_req, &ret_err);
                    }

                    ((packa_s32)lcntx->packa_fun_ptr)(
                        (uint8_t*)pack_a_buffer_s8s4s32o32,
                        (uint8_t*)(a + (rs_a * ic) + (cs_a * pc)), rs_a, cs_a,
                        mc0, kc0, &rs_a_use, &cs_a_use);
                    a_use = pack_a_buffer_s8s4s32o32;

                    if (cs_a == 1) {
                        a_block_stride = kc0_updated;
                    } else {
                        a_block_stride = rs_a_use;
                    }
                } else {
                    a_use = a + (rs_a * ic) + (cs_a * pc);

                    // Int8 kernel reads 4 elements (4 bytes) per broadcast.
                    cs_a_use       = 4;
                    a_block_stride = rs_a;
                }

                post_ops_attr.b_sum_offset = 0;

                for (iter_t jr = 0; jr < nc0; jr += NR) {
                    md_t nr0 = dlp_min((nc0 - jr), NR);

                    post_ops_attr.post_op_c_i    = ic;
                    post_ops_attr.post_op_c_j    = (jc + jr);
                    post_ops_attr.rs_c_downscale = rs_c_downscale;

                    grp_post_ops_attr.grp_post_op_j = jc + jr;

                    const int8_t* b_kernel;
                    if (mtag_b == REORDERED) {
                        // Transiently widen this compact s4 NR-panel back to
                        // s8. The panel occupies round_up(nr0,16) columns of
                        // the packed layout; the kernel never reads beyond
                        // that. The scratch tail is zeroed to guard against
                        // padded reads.
                        md_t widen_cols  = dlp_make_multiple_of_n(nr0, 16);
                        md_t widen_elems = widen_cols * kc0_updated;
                        md_t b_s8_jr_off = b_s8_pc_base + (jr * kc0_updated);

                        if (widen_elems < (NR * kc0_updated)) {
                            memset(b_panel_s8 + widen_elems, 0,
                                   (size_t)((NR * kc0_updated) - widen_elems));
                        }
                        dlp_cvt_s4_to_s8_linear_avx512(
                            b_panel_s8, b_compact + (b_s8_jr_off / 2),
                            widen_elems);
                        b_kernel = b_panel_s8;
                    } else {
                        // PACK: read the runtime-packed s8 weights directly.
                        b_kernel = b_use + (jr * kc0_updated);
                    }

                    dlp_gemm_rowvar_s8s8s32os32_6x64m_sym_quant(
                        mc0, nr0, kc0, a_use, rs_a_use, cs_a_use,
                        a_block_stride, b_kernel, rs_b_use, cs_b_use,
                        dlp_offset_or_null_f32(c_use_ic, jr), rs_c_use, 1,
                        alpha, beta0, grp_post_ops_attr, post_op_list,
                        post_ops_attr);

                    post_ops_attr.b_sum_offset += NR;
                }
            }
        }
        if (mtag_b == REORDERED) {
            dlp_gemm_adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }

    // Release buffers.
    if (mtag_b == PACK) {
        // All threads in the work group must finish using the shared packed B
        // before the chief thread frees it.
        dlp_atomic_barrier(thread_jc.ocomm_id,
                           &thread->comm[thread_jc.work_id]);

        if (dlp_thread_am_ochief(&thread_ic)) {
            if (pack_b_buffer_s8s4s32o32 != NULL) {
                dlp_free_page_aligned(pack_b_buffer_s8s4s32o32);
            }
        }
    }
    if (mtag_a == PACK) {
        if (pack_a_buffer_s8s4s32o32 != NULL) {
            dlp_free_page_aligned(pack_a_buffer_s8s4s32o32);
        }
    }
    if (c_downscale < DLP_F32) {
        if (temp_scal_c_buffer_s8s4s32o32 != NULL) {
            dlp_free_page_aligned(temp_scal_c_buffer_s8s4s32o32);
        }
    }
    if (b_panel_s8 != NULL) {
        dlp_free_page_aligned(b_panel_s8);
    }
#else
    (void)m;
    (void)n;
    (void)k;
    (void)a;
    (void)rs_a;
    (void)cs_a;
    (void)mtag_a;
    (void)b;
    (void)rs_b;
    (void)cs_b;
    (void)mtag_b;
    (void)c;
    (void)rs_c;
    (void)cs_c;
    (void)alpha;
    (void)beta;
    (void)rntm;
    (void)thread;
    (void)lcntx;
    (void)c_downscale;
    (void)post_op_list;
    (void)grp_post_op_list;
    (void)NC;
    (void)MC;
    (void)NR;
    (void)MR;
    (void)KC;
#endif
}
