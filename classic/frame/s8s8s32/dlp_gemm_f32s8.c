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

#include "config/dlp_gemm_config.h"
#include "dlp_gemm_5loop_interface_apis.h"
#include "gemm_utils/dlp_gemm_utils.h"
#include "kernels/dlp_gemm_kernels.h"
#include "kernels/s8s8s32/dlp_gemm_packa_s8.h"
#include "kernels/s8s8s32/dlp_gemm_packb_s8.h"
#include "kernels/s8s8s32/dlp_gemm_quanta_s8.h"
#include "kernels/u8s8s32/dlp_gemm_packa.h"
#include "sys_utils/dlp_gemm_sys.h"
#include "threading/dlp_gemm_thread_utils.h"

/**
 * Low-precision GEMV for F32 x S8 -> S32 with integrated dequantization.
 *
 * This function handles matrix-vector multiplication (n=1 case) where:
 *   - A is a F32 (float) input, quantized on-the-fly to S8
 *   - B is an S8 vector (packed or reordered for efficiency)
 *   - C is an S32 accumulator, which may be downscaled to S8, U8, BF16, or F32
 */
DLP_GEMV3(float, int8_t, int32_t, f32s8s32os32)
{
    md_t NC = lcntx->blksz.NC;
    md_t KC = lcntx->blksz.KC;
    md_t MC = lcntx->blksz.MC;
    md_t NR = lcntx->blksz.NR;

    // Scale factor parameters for A matrix quantization.
    void*    sf      = a_pre_quant->scl->scale_factor;
    md_t     sf_len  = a_pre_quant->scl->scale_factor_len;
    DLP_TYPE sf_type = a_pre_quant->scl->scale_factor_type;

    // Zero-point parameters for A matrix quantization.
    bool     is_symmetric = a_pre_quant->symmetric;
    void*    zp_val       = NULL;
    md_t     zp_len       = 0;
    DLP_TYPE zp_type      = DLP_INVALID;

    // Initialize zero-point values for asymmetric quantization.
    if (!is_symmetric) {
        if (a_pre_quant->zp == NULL) {
            dlp_print_msg(" Zero point parameters for A matrix quantization "
                          "are missing when asymmetric quantization is used.",
                          __FILE__, __LINE__);
            return;
        }
        zp_val  = a_pre_quant->zp->zero_point;
        zp_len  = a_pre_quant->zp->zero_point_len;
        zp_type = a_pre_quant->zp->zero_point_type;
    }

    // Strides are updated based on matrix packing/reordering.
    const int8_t* a_use    = NULL;
    md_t          rs_a_use = rs_a;
    md_t          cs_a_use = cs_a;

    const int8_t* b_use    = NULL;
    md_t          rs_b_use = rs_b;
    md_t          cs_b_use = cs_b;

    int32_t* c_use = NULL;

    int32_t* pack_b_column_sum = NULL;

    dlp_gemm_post_op_attr post_ops_attr;
    post_ops_attr.c_stor_type       = c_downscale;
    post_ops_attr.rs_c_downscale    = rs_c;
    post_ops_attr.cs_c_downscale    = cs_c;
    post_ops_attr.is_first_k        = TRUE;
    post_ops_attr.is_last_k         = TRUE;
    post_ops_attr.b_sum_offset      = 0;
    post_ops_attr.b_col_sum_vec     = NULL;
    post_ops_attr.b_col_sum_vec_s16 = NULL;

    if (c_downscale < DLP_S32 || c_downscale == DLP_F32)
        post_ops_attr.buf_downscale = c;
    else
        post_ops_attr.buf_downscale = NULL;

    msz_t mem_a_size_req    = 0;
    msz_t mem_b_size_req    = 0;
    msz_t mem_a_s8_size_req = 0;

    int8_t* pack_b_buffer_s8s8s32os32 = NULL;
    int8_t* quant_a_buffer_s8         = NULL;

    // Generate thrinfo objects for jc and ic loops from dlp_gemm_thrinfo_t.
    dlp_task_id_t thread_jc;
    dlp_task_id_t thread_ic;

    dlp_gemm_gen_dlp_task_ids(thread, &thread_jc, &thread_ic);

    if (n == 1) {
        // Increased MR from 6 to 16 to make use of 32 ZMM registers
        md_t MR = 16;

        // pack B matrix if needed
        if ((mtag_b == PACK)) {
            mem_b_size_req = sizeof(int8_t) * k + sizeof(int32_t);

            if (pack_b_buffer_s8s8s32os32 == NULL) {
                dlp_clsc_err_t ret_err;
                pack_b_buffer_s8s8s32os32 =
                    dlp_malloc_page_aligned(mem_b_size_req, &ret_err);
            }

            pack_b_column_sum =
                (int32_t*)(pack_b_buffer_s8s8s32os32 + (sizeof(int8_t) * k));

            *pack_b_column_sum = 0;

            for (iter_t k0 = 0; k0 < k; k0++) {
                pack_b_buffer_s8s8s32os32[k0] = b[k0 * rs_b];
                *pack_b_column_sum += pack_b_buffer_s8s8s32os32[k0];
            }

            *pack_b_column_sum *= 128;
            post_ops_attr.b_col_sum_vec = pack_b_column_sum;

            b_use    = pack_b_buffer_s8s8s32os32;
            rs_b_use = 1;
            cs_b_use = 1;
        } else if (mtag_b == REORDERED) {
            b_use                       = b;
            post_ops_attr.b_col_sum_vec = (int32_t*)(b + k);
        }

        // Compute the IC loop thread range for the current thread.
        md_t ic_start, ic_end;
        thread_ic.n_way   = (thread_ic.n_way == 1) ? (thread->n_threads)
                                                   : (thread_ic.n_way);
        thread_ic.work_id = thread->tid;
        dlp_thread_task_range(&thread_ic, m, MR, FALSE, &ic_start, &ic_end);

        for (iter_t ic = ic_start; ic < ic_end; ic += MR) {
            md_t mc0 = dlp_min((ic_end - ic), MR);

            c_use = c + ic * rs_c;

            post_ops_attr.post_op_c_i    = ic;
            post_ops_attr.post_op_c_j    = 0;
            post_ops_attr.rs_c_downscale = rs_c;

            // Allocate buffer for quantized A matrix (F32 -> S8 conversion).
            mem_a_s8_size_req = sizeof(int8_t) * mc0 * k;

            if (quant_a_buffer_s8 == NULL) {
                dlp_clsc_err_t ret_err;
                quant_a_buffer_s8 =
                    dlp_malloc_page_aligned(mem_a_s8_size_req, &ret_err);
            }

            // Perform on-the-fly quantization: F32 -> S8
            // Uses per-row scale factors and zero-points indexed by IC offset.
            dlp_quanta_mr16_f32s8(quant_a_buffer_s8, (float*)(a + (rs_a * ic)),
                                  rs_a, cs_a, mc0, k, sf, sf_type, sf_len,
                                  zp_val, zp_type, zp_len, ic);

            // Update A pointer and strides to use quantized buffer.
            a_use    = quant_a_buffer_s8;
            rs_a_use = k;
            cs_a_use = 4;

            dlp_execute_kernel(&(lcntx->dlp_kernel_hndl), mc0, 1, k,
                               (int8_t*)a_use, rs_a_use, cs_a_use, 1,
                               (int8_t*)(b_use), rs_b_use, cs_b_use, 0, 0,
                               c_use, rs_c, 1, (void*)&alpha, (void*)&beta,
                               post_op_list, post_ops_attr);
        }

        // Release quantized A buffer (per-thread allocation).
        if (quant_a_buffer_s8 != NULL) {
            dlp_free_page_aligned(quant_a_buffer_s8);
        }

        // Release pack buffers
        if (mtag_b == PACK && (pack_b_buffer_s8s8s32os32 != NULL)) {
            dlp_free_page_aligned(pack_b_buffer_s8s8s32os32);
        }
    } else {
        // m == 1 case
        md_t gemm_MR = lcntx->blksz.MR;

        md_t jc_start, jc_end;
        thread_jc.n_way   = (thread_jc.n_way == 1) ? (thread->n_threads)
                                                   : (thread_jc.n_way);
        thread_jc.work_id = thread->tid;
        dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

        md_t packb_min_NR = dlp_get_packb_s8s8s32o32_min_NR();

        md_t k_updated = dlp_make_multiple_of_n(k, 4);
        md_t n_updated = dlp_make_multiple_of_n(n, 16);

        rs_a_use = rs_a;
        cs_a_use = 4;

        // Allocate buffer for quantized A matrix (F32 -> S8 conversion).
        mem_a_s8_size_req = sizeof(int8_t) * k;

        if (quant_a_buffer_s8 == NULL) {
            dlp_clsc_err_t ret_err;
            quant_a_buffer_s8 =
                dlp_malloc_page_aligned(mem_a_s8_size_req, &ret_err);
        }

        // Perform on-the-fly quantization: F32 -> S8 for single row
        dlp_quanta_mr16_f32s8(quant_a_buffer_s8, (float*)a, rs_a, cs_a, 1, k,
                              sf, sf_type, sf_len, zp_val, zp_type, zp_len, 0);

        a_use = quant_a_buffer_s8;

        for (iter_t jc = jc_start; jc < jc_end; jc += NC) {
            md_t nc0 = dlp_min((jc_end - jc), NC);
            c_use    = c + jc;

            md_t jc_cur_loop     = jc;
            md_t jc_cur_loop_rem = 0;
            md_t n_sub_updated   = 0;

            if (mtag_b == REORDERED) {
                dlp_gemm_get_B_panel_reordered_start_offset_width(
                    jc, n, NC, packb_min_NR, &jc_cur_loop, &jc_cur_loop_rem,
                    &nc0, &n_sub_updated);

                b_use = (int8_t*)(b + (jc_cur_loop * k_updated));

                dlp_gemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);

                post_ops_attr.b_col_sum_vec =
                    ((int32_t*)(b + (k_updated * n_updated))) + jc;
            } else if (mtag_b == PACK) {
                md_t nc0_updated = dlp_make_multiple_of_n(nc0, packb_min_NR);

                mem_b_size_req = sizeof(int8_t) * nc0_updated * k_updated
                                 + (nc0_updated * sizeof(int32_t));

                n_sub_updated = nc0_updated;

                if (pack_b_buffer_s8s8s32os32 == NULL) {
                    dlp_clsc_err_t ret_err;
                    pack_b_buffer_s8s8s32os32 =
                        dlp_malloc_page_aligned(mem_b_size_req, &ret_err);
                }

                pack_b_column_sum =
                    (int32_t*)(pack_b_buffer_s8s8s32os32
                               + (sizeof(int8_t) * nc0_updated * k_updated));

                for (iter_t idx = 0; idx < nc0; idx++) {
                    *(pack_b_column_sum + idx) = 0;
                }

                for (iter_t pc = 0; pc < k; pc += KC) {
                    md_t kc0 = dlp_min((k - pc), KC);

                    ((packb_s32_s8)lcntx->packb_fun_ptr)(
                        (pack_b_buffer_s8s8s32os32) + (n_sub_updated * pc),
                        pack_b_column_sum, (b + (rs_b * pc) + (jc * cs_b)),
                        rs_b, cs_b, nc0, kc0, &rs_b_use, &cs_b_use);
                }

                b_use                       = pack_b_buffer_s8s8s32os32;
                post_ops_attr.b_col_sum_vec = pack_b_column_sum;
            }

            post_ops_attr.post_op_c_i    = 0;
            post_ops_attr.post_op_c_j    = jc;
            post_ops_attr.rs_c_downscale = rs_c;
            post_ops_attr.b_sum_offset   = 0;

            dlp_execute_kernel(
                &(lcntx->dlp_kernel_hndl), 1, nc0, k, (int8_t*)a_use, rs_a_use,
                cs_a_use, 1, (int8_t*)b_use, rs_b_use, cs_b_use, n_sub_updated,
                jc_cur_loop_rem, c_use, rs_c, cs_c, (void*)&alpha, (void*)&beta,
                post_op_list, post_ops_attr);

            if (mtag_b == REORDERED) {
                dlp_gemm_adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
            }
        } // jc loop

        // Release quantized A buffer
        if (quant_a_buffer_s8 != NULL) {
            dlp_free_page_aligned(quant_a_buffer_s8);
        }

        // Release pack buffers
        if (mtag_b == PACK && (pack_b_buffer_s8s8s32os32 != NULL)) {
            dlp_free_page_aligned(pack_b_buffer_s8s8s32os32);
        }
    }
}

/**
 * Low-precision GEMM for F32 x S8 -> S32 with integrated dequantization
 * post-ops.
 *
 * This function computes C = alpha * (A * B) + beta * C, where:
 *   - A is a F32 (float) input, quantized on-the-fly to S8 immediately
 * prior to compute.
 *   - B is an S8 matrix (pre-packed or reordered for efficiency).
 *   - C is an S32 accumulator, which may additionally be downscaled with
 * dequantization post-ops to S8, U8, BF16, or F32.
 *
 * The underlying implementation employs a cache-optimized, multi-threaded
 * 5-loop blocking (JC-PC-IC-JR-IR) algorithm. Quantization of A is performed
 * for each MC block.
 *
 * After the main GEMM computation (within the JIT kernel), post-processing
 * including dequantization is supported via post-ops. This enables direct
 * transformation of S32 accumulators into lower-precision formats (like
 * S8/U8/BF16/F32) using scale/shift parameters, all in the JIT kernel for
 * maximum performance.
 *
 * Loop/blocking hierarchy:
 *   JC: N dimension (NC blocking)
 *   PC: K dimension (KC blocking)
 *   IC: M dimension (MC blocking)
 *   JR: Inner N (NR micro-tile)
 *   IR: Inner M (MR micro-tile, handled by micro-kernel)
 */
DLP_GEMM_5LOOP_UNIFIED(float, int8_t, int32_t, int32_t, f32s8s32os32, const)
{
    // Extract operations from bundle into local variables
    DLP_GEMM_OPS_EXTRACT(ops);

    md_t NC = lcntx->blksz.NC;
    md_t KC = lcntx->blksz.KC;
    md_t MC = lcntx->blksz.MC;
    md_t NR = lcntx->blksz.NR;
    md_t MR = lcntx->blksz.MR;

    // Validate B matrix memory tag - only packed or reordered formats
    // supported.
    if (mtag_b == UNPACKED) {
        dlp_print_msg(" UNPACKED B matrix is not supported.", __FILE__,
                      __LINE__);
        return;
    }

    // Validate quantization parameters.
    if (a_pre_quant == NULL || a_pre_quant->scl == NULL
        || a_pre_quant->scl->scale_factor == NULL) {
        dlp_print_msg(
            " Scale factor parameters for A matrix quantization are missing.",
            __FILE__, __LINE__);
        return;
    }

    if (m == 1 || n == 1) {
        dlp_gemv_rowvar_f32s8s32os32(m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b,
                                     cs_b, mtag_b, c, rs_c, cs_c, alpha, beta,
                                     rntm, thread, lcntx, a_pre_quant,
                                     post_op_list, c_downscale);
        return;
    }

    // Scale factor parameters for A matrix quantization.
    void*    sf      = a_pre_quant->scl->scale_factor;
    md_t     sf_len  = a_pre_quant->scl->scale_factor_len;
    DLP_TYPE sf_type = a_pre_quant->scl->scale_factor_type;

    // Zero-point parameters for A matrix quantization.
    // Symmetric quantization: zp = 0 (no zero-point correction needed)
    // Asymmetric quantization: zp != 0 (requires zero-point subtraction)
    bool     is_symmetric = a_pre_quant->symmetric;
    void*    zp_val       = NULL;
    md_t     zp_len       = 0;
    DLP_TYPE zp_type      = DLP_INVALID;

    // Initialize zero-point values for asymmetric quantization.
    if (!is_symmetric) {
        if (a_pre_quant->zp == NULL) {
            dlp_print_msg(" Zero point parameters for A matrix quantization "
                          "are missing when asymmetric quantization is used.",
                          __FILE__, __LINE__);
            return;
        }
        zp_val  = a_pre_quant->zp->zero_point;
        zp_len  = a_pre_quant->zp->zero_point_len;
        zp_type = a_pre_quant->zp->zero_point_type;
    }

    // Matrix pointers and strides - updated based on packing/reordering status.
    // A matrix: Will point to quantized S8 buffer after on-the-fly
    // quantization.
    const int8_t* a_use          = NULL;
    md_t          rs_a_use       = rs_a; // Row stride for A
    md_t          cs_a_use       = cs_a; // Column stride for A
    md_t          a_block_stride = 0;    // Stride between MR blocks

    // B matrix: Points to packed or reordered buffer.
    const int8_t* b_use    = NULL;
    md_t          rs_b_use = rs_b; // Row stride for B
    md_t          cs_b_use = cs_b; // Column stride for B

    // C matrix: May use temporary buffer for accumulation across K iterations.
    int32_t* c_use_jc = NULL; // C pointer per JC iteration
    int32_t* c_use_ic = NULL; // C pointer per IC iteration
    md_t     rs_c_use = rs_c; // Row stride (may differ if using temp buffer)
    md_t     rs_c_downscale = rs_c; // Row stride for final downscaled output

    // Buffer for quantized A matrix (F32 -> S8 conversion).
    // Allocated per-thread, sized for one MC x KC block.
    int8_t* quant_a_buffer_s8 = NULL;
    msz_t   mem_a_s8_size_req = 0;

    // Buffer for packed B matrix (S8).
    int8_t* pack_b_buffer_s8s8s32o32 = NULL;
    msz_t   mem_b_size_req           = 0;
    md_t    packb_min_NR             = dlp_get_packb_s8s8s32o32_min_NR();

    // Temporary buffer for C accumulation when downscaling is required.
    int32_t* temp_scal_c_buffer_s8s8s32o32 = NULL;
    msz_t    mem_scale_c_size_req          = 0;

    // After quantizing f32 to s8, the s8 x s8 dot product kernel achieves
    // proper vectorization and alignment with the underlying SIMD instructions
    // (e.g., vpdpbusd) when the k dimension (kc) is a multiple of 4.
    // This function automatically pads k to a multiple of 4 if needed, so the
    // caller does not need to ensure this property. The k offset used for
    // packed/reordered buffers is updated accordingly.
    md_t k_updated = dlp_make_multiple_of_n(k, 4);
    md_t n_updated = dlp_make_multiple_of_n(n, 16);

    // Flags to control K-dimension accumulation behavior:
    // - is_last_k: TRUE on final KC iteration -> apply post-ops and downscaling
    // - is_first_k: TRUE on first KC iteration -> apply beta scaling to
    // original C
    bool is_last_k  = FALSE;
    bool is_first_k = FALSE;

    dlp_gemm_post_op_attr post_ops_attr;

    // Initialize post-operation attributes.
    post_ops_attr.c_stor_type    = c_downscale; // Output data type
    post_ops_attr.rs_c_downscale = rs_c; // Row stride for downscaled output
    post_ops_attr.cs_c_downscale = cs_c; // Column stride for downscaled output
    post_ops_attr.b_sum_offset   = 0;    // Offset into B column sum vector
    post_ops_attr.b_col_sum_vec =
        NULL; // B column sums for zero-point compensation
    post_ops_attr.b_col_sum_vec_s16 =
        NULL; // B column sums in S16 format (if needed)

    // Set downscaling buffer pointer based on output type.
    if (c_downscale < DLP_S32 || c_downscale == DLP_F32) {
        post_ops_attr.buf_downscale = c; // Downscale to smaller type or F32
    } else {
        post_ops_attr.buf_downscale = NULL; // No downscaling, output is S32
    }

    // Generate thrinfo objects for jc and ic loops from dlp_gemm_thrinfo_t.
    dlp_task_id_t thread_jc;
    dlp_task_id_t thread_ic;

    dlp_gemm_gen_dlp_task_ids(thread, &thread_jc, &thread_ic);

    // Compute the JC, IC loop thread range for the current thread.
    md_t jc_start, jc_end;
    dlp_thread_task_range(&thread_jc, n, NR, FALSE, &jc_start, &jc_end);

    md_t ic_start, ic_end;
    dlp_thread_task_range(&thread_ic, m, MR, FALSE, &ic_start, &ic_end);

    for (iter_t jc = jc_start; jc < jc_end; jc += NC) {
        md_t nc0 = dlp_min((jc_end - jc), NC); // Actual width of current panel

        md_t jc_cur_loop     = jc;
        md_t jc_cur_loop_rem = 0;
        md_t n_sub_updated   = 0;

        if (mtag_b == REORDERED) {
            dlp_gemm_get_B_panel_reordered_start_offset_width(
                jc, n, NC, packb_min_NR, &jc_cur_loop, &jc_cur_loop_rem, &nc0,
                &n_sub_updated);
        }

        // Set up C matrix pointer and stride based on output type.
        if (c_downscale == DLP_S32) {
            // Output is S32: write directly to final C matrix.
            c_use_jc = c + jc;
        }
        // Allocate temporary S32 accumulation buffer for downscaling scenarios.
        else if (c_downscale < DLP_S32 || c_downscale == DLP_F32) {
            // Temporary buffer is only needed when k > KC (multiple PC
            // iterations). For k <= KC, computation completes in one pass and
            // can write directly. Note: Avoiding unnecessary allocations
            // reduces memory pool lock contention.
            if (k > KC) {
                // Allocate buffer sized for the largest NC block this thread
                // handles. (NC blocks may split at reorder boundaries, so use
                // max size.)
                md_t nc0_buf = dlp_min((jc_end - jc), NC);
                mem_scale_c_size_req =
                    sizeof(int32_t) * nc0_buf * (ic_end - ic_start);

                if (temp_scal_c_buffer_s8s8s32o32 == NULL) {
                    dlp_clsc_err_t ret_err;
                    temp_scal_c_buffer_s8s8s32o32 =
                        dlp_malloc_page_aligned(mem_scale_c_size_req, &ret_err);
                }

                c_use_jc = (int32_t*)temp_scal_c_buffer_s8s8s32o32;
            }

            // The temp c buffer stride is modified as opposed to original C
            // matrix.
            rs_c_use = nc0;
        }

        // Column sum vector for B matrix (used for zero-point compensation).
        // Computed during packing: sum of each column to correct A's
        // zero-point.
        int32_t* pack_b_column_sum = NULL;

        for (iter_t pc = 0; pc < k; pc += KC) {
            int32_t beta0 = (pc == 0) ? beta : 1;
            md_t kc0 = dlp_min((k - pc), KC); // Actual depth of current panel

            // Align kc0 to multiple of 4 for SIMD instruction requirements.
            // Packed/reordered buffers include this padding.
            md_t kc0_updated = dlp_make_multiple_of_n(kc0, 4);

            // Set K iteration flags for proper accumulation and post-op
            // application.
            is_first_k               = (pc == 0) ? (TRUE) : (FALSE);
            post_ops_attr.is_first_k = is_first_k;

            is_last_k               = ((pc + KC) >= k) ? (TRUE) : (FALSE);
            post_ops_attr.is_last_k = is_last_k;

            // -----------------------------------------------------------------
            // B MATRIX PACKING/REORDERING SECTION
            // -----------------------------------------------------------------
            if (mtag_b == PACK) {
                {
                    // Pack B chunks are based on jc work id.
                    md_t jc_work_id = thread_jc.work_id;

                    // Using child thrinfo (thread_ic) tid to decide chief
                    // thread per B matrix chunk (jc work id group)
                    md_t nc0_updated =
                        dlp_make_multiple_of_n(nc0, packb_min_NR);

                    if (dlp_thread_am_ochief(&thread_ic)) {
                        // Buffer layout: [packed B matrix] + [column sum
                        // vector]
                        // - Packed B: nc0_updated x kc0_updated S8 elements
                        // - Column sums: nc0_updated S32 elements (for
                        // zero-point compensation)
                        mem_b_size_req =
                            sizeof(int8_t) * nc0_updated * kc0_updated
                            + (nc0_updated * sizeof(int32_t));

                        if (pack_b_buffer_s8s8s32o32 == NULL) {
                            dlp_clsc_err_t ret_err;
                            pack_b_buffer_s8s8s32o32 = dlp_malloc_page_aligned(
                                mem_b_size_req, &ret_err);
                        }

                        thread->comm[jc_work_id].sent_object =
                            pack_b_buffer_s8s8s32o32;
                    }

                    // All threads in work group should wait till chief thread
                    // has finished allocating the packing buffers.
                    dlp_atomic_barrier(thread_ic.ocomm_id,
                                       &thread->comm[jc_work_id]);

                    pack_b_buffer_s8s8s32o32 =
                        (int8_t*)thread->comm[jc_work_id].sent_object;

                    // Compute the B panel per thread loop range for parallel
                    // packing using ic_ways number of threads. Since atmost
                    // only ic_ways threads can be used, the thread_ic
                    // attributes are used to split the loop range.
                    md_t jc_packb_start, jc_packb_end;
                    dlp_thread_task_range(&thread_ic, nc0, NR, FALSE,
                                          &jc_packb_start, &jc_packb_end);

                    if (pc == 0) {
                        pack_b_column_sum =
                            (int32_t*)(pack_b_buffer_s8s8s32o32
                                       + (sizeof(int8_t) * nc0_updated
                                          * kc0_updated));
                    }

                    // Ensure thread ranges are valid, especially cases where
                    // no: of threads available for parallelization are greater
                    // than no: of B panel NR chunks.
                    if ((jc_packb_end > jc_packb_start)
                        && (jc_packb_start < (jc + nc0))) {
                        if (pc == 0) {
                            for (iter_t idx = jc_packb_start;
                                 idx < jc_packb_end; idx++) {
                                *(pack_b_column_sum + idx) = 0;
                            }
                        }

                        // Pack this thread's assigned columns of B matrix.
                        // Packing includes computing column sums for zero-point
                        // compensation.
                        ((packb_s32_s8)lcntx->packb_fun_ptr)(
                            pack_b_buffer_s8s8s32o32
                                + (jc_packb_start * kc0_updated),
                            pack_b_column_sum + (jc_packb_start),
                            (b + (rs_b * pc) + (cs_b * jc)
                             + (cs_b * jc_packb_start)),
                            rs_b, cs_b, (jc_packb_end - jc_packb_start), kc0,
                            &rs_b_use, &cs_b_use);
                    } else {
                        dlp_gemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);
                    }

                    // All threads in work group should wait till B matrix
                    // packing is completed by the participating threads.
                    dlp_atomic_barrier(thread_ic.ocomm_id,
                                       &thread->comm[jc_work_id]);
                    b_use = pack_b_buffer_s8s8s32o32;

                    post_ops_attr.b_col_sum_vec = pack_b_column_sum;
                }
            } else if (mtag_b == REORDERED) {
                // B matrix is pre-reordered: compute offset into reordered
                // buffer. Reordered layout: panels of size k_updated x
                // packb_min_NR Thread work ranges may start mid-panel,
                // requiring offset calculation.
                b_use = b + (jc_cur_loop * k_updated) + (n_sub_updated * pc)
                        + (jc_cur_loop_rem * kc0_updated);

                dlp_gemm_get_packb_strides(lcntx, &rs_b_use, &cs_b_use);

                // Column sum vector is stored after all reordered B data.
                post_ops_attr.b_col_sum_vec =
                    ((int32_t*)(b + (k_updated * n_updated))) + jc;
            } else {
                // Unpacked B not supported.
                dlp_print_msg(" B matrix in packed or reordered format are "
                              "only supported.",
                              __FILE__, __LINE__);
                return;
            }
            for (iter_t ic = ic_start; ic < ic_end; ic += MC) {
                md_t mc0 = dlp_min((ic_end - ic), MC);

                // Only per thread C matrix is stored in temp buffer, so both
                // per thread jc and ic start should be normalized to zero.
                if (c_downscale < DLP_S32 || c_downscale == DLP_F32) {
                    // Temp buffer: normalize IC offset to thread-local
                    // coordinates.
                    c_use_ic = c_use_jc + (rs_c_use * (ic - ic_start));
                } else {
                    // Direct S32 output: use global IC coordinates.
                    c_use_ic = c_use_jc + (rs_c_use * ic);
                }

                // Allocate buffer for quantized A matrix (F32 -> S8
                // conversion).
                mem_a_s8_size_req = sizeof(int8_t) * mc0 * kc0;

                if (quant_a_buffer_s8 == NULL) {
                    dlp_clsc_err_t ret_err;
                    quant_a_buffer_s8 =
                        dlp_malloc_page_aligned(mem_a_s8_size_req, &ret_err);
                }

                // Perform on-the-fly quantization: F32 -> S8
                // Uses per-row scale factors and zero-points indexed by IC
                // offset.
                dlp_quanta_mr16_f32s8(quant_a_buffer_s8,
                                      (float*)(a + (rs_a * ic) + (cs_a * pc)),
                                      rs_a, cs_a, mc0, kc0, sf, sf_type, sf_len,
                                      zp_val, zp_type, zp_len, ic);

                // Update A pointer and strides to use quantized buffer.
                // Quantized layout: row-major with row_stride = kc0
                a_use          = quant_a_buffer_s8;
                rs_a_use       = kc0;      // Elements per row
                cs_a_use       = 4;        // Stride within SIMD groups
                a_block_stride = rs_a_use; // Stride between MR blocks

                // Reset B column sum offset for new IC iteration.
                post_ops_attr.b_sum_offset = 0;

                for (iter_t jr = 0; jr < nc0; jr += NR) {
                    md_t nr0 =
                        dlp_min((nc0 - jr), NR); // Actual width of micro-tile

                    // Post ops meta attributes.
                    post_ops_attr.post_op_c_i    = ic;
                    post_ops_attr.post_op_c_j    = (jc + jr);
                    post_ops_attr.rs_c_downscale = rs_c_downscale;

                    // Run the micro-kernel to compute C += A * B for this
                    // micro-tile. This kernel accumulates S8 x S8 into S32,
                    // applying dequantization and optional post-operations
                    // (e.g., bias addition, ReLU activation, downscaling)
                    // during the final K step.
                    dlp_execute_kernel(
                        &(lcntx->dlp_kernel_hndl), mc0, nr0, kc0,
                        (int8_t*)a_use, rs_a_use, cs_a_use, a_block_stride,
                        (int8_t*)(b_use + (jr * kc0_updated)), rs_b_use,
                        cs_b_use, 0, 0, (c_use_ic + jr), rs_c_use, 1,
                        (void*)&alpha, (void*)&beta0, post_op_list,
                        post_ops_attr);

                    post_ops_attr.b_sum_offset += NR;
                }
            }
        }
        if (mtag_b == REORDERED) {
            dlp_gemm_adjust_B_panel_reordered_jc(&jc, jc_cur_loop);
        }
    }

    // Release pack buffers.
    if (mtag_b == PACK) {
        // All threads in work group should wait till B matrix usage is
        // completed by the participating threads.
        dlp_atomic_barrier(thread_jc.ocomm_id,
                           &thread->comm[thread_jc.work_id]);

        if (dlp_thread_am_ochief(&thread_ic)) {
            if (pack_b_buffer_s8s8s32o32 != NULL) {
                dlp_free_page_aligned(pack_b_buffer_s8s8s32o32);
            }
        }
    }

    // Release quantized A buffer (per-thread allocation).
    if (quant_a_buffer_s8 != NULL) {
        dlp_free_page_aligned(quant_a_buffer_s8);
    }

    // Release temporary C accumulation buffer (per-thread allocation).
    if (c_downscale < DLP_S32 || c_downscale == DLP_F32) {
        if (temp_scal_c_buffer_s8s8s32o32 != NULL) {
            dlp_free_page_aligned(temp_scal_c_buffer_s8s8s32o32);
        }
    }
}
