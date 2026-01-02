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

#include "lpgemm_post_ops.h"
#include "gemm_utils/lpgemm_utils.h"
#include "lpgemm_types.h"

static inline DLP_TYPE
get_stor_type(DLP_TYPE pstor_type)
{
    DLP_TYPE stor_type = DLP_INVALID;
    switch (pstor_type) {
        case DLP_F32:
            stor_type = DLP_F32;
            break;
        case DLP_BF16:
            stor_type = DLP_BF16;
            break;
        case DLP_S8:
            stor_type = DLP_S8;
            break;
        case DLP_U8:
            stor_type = DLP_U8;
            break;
        case DLP_S32:
            stor_type = DLP_S32;
            break;
        default:
            break;
    }

    return stor_type;
}

DLP_INLINE void
lpgemm_set_pre_ops_node_params(lpgemm_pre_op* pre_op_node,
                               md_t           group_size,
                               void*          zero_point,
                               void*          scale_factor,
                               md_t           zero_point_len,
                               md_t           scale_factor_len,
                               md_t           scale_factor_type)
{
    pre_op_node->group_size        = group_size;
    pre_op_node->scale_factor      = scale_factor;
    pre_op_node->scale_factor_len  = scale_factor_len;
    pre_op_node->zp                = zero_point;
    pre_op_node->zp_len            = zero_point_len;
    pre_op_node->scale_factor_type = scale_factor_type;
    pre_op_node->next              = NULL;
}

DLP_INLINE void
lpgemm_set_group_post_ops_node_params(lpgemm_group_post_op* post_op_node,
                                      md_t                  group_size,
                                      void*                 a_zero_point,
                                      void*                 a_scale_factor,
                                      md_t                  a_zero_point_len,
                                      md_t                  a_scale_factor_len,
                                      void*                 b_zero_point,
                                      void*                 b_scale_factor,
                                      md_t                  b_zero_point_len,
                                      md_t                  b_scale_factor_len,
                                      DLP_TYPE              sf_stor_type,
                                      DLP_TYPE              zp_stor_type)
{
    post_op_node->group_size         = group_size;
    post_op_node->a_zp               = a_zero_point;
    post_op_node->a_zp_len           = a_zero_point_len;
    post_op_node->a_scale_factor     = a_scale_factor;
    post_op_node->a_scale_factor_len = a_scale_factor_len;
    post_op_node->b_zp               = b_zero_point;
    post_op_node->b_zp_len           = b_zero_point_len;
    post_op_node->b_scale_factor     = b_scale_factor;
    post_op_node->b_scale_factor_len = b_scale_factor_len;
    post_op_node->sf_stor_type       = sf_stor_type;
    post_op_node->zp_stor_type       = zp_stor_type;
    post_op_node->next               = NULL;
}

dlp_clsc_err_t
lpgemm_translate_to_group_postops_list(dlp_group_post_op*    metadata,
                                       lpgemm_group_post_op* post_op_list,
                                       md_t                  m,
                                       md_t                  n,
                                       md_t                  k)
{
    if ((metadata == NULL) || (metadata->seq_length <= 0)) {
        lpgemm_set_group_post_ops_node_params(post_op_list, 0, NULL, NULL, 0, 0,
                                              NULL, NULL, 0, 0, DLP_INVALID,
                                              DLP_INVALID);

        return DLP_CLSC_SUCCESS;
    }

    for (md_t i = 0; i < metadata->seq_length; ++i) {
        /* group_size that is non-multiple of 4 is supported only when
         * group_size == k */
        md_t group_size = metadata->group_size;
        if ((group_size == 0) || (group_size > k) || (group_size == k))
            group_size = k;
        else if (metadata->group_size % 4 > 0)
            return DLP_CLSC_FAILURE;

        if (metadata->a_zp != NULL) {
            /* check for validity of pre-ops */
            if (((metadata->a_zp)->zero_point_len > 0)
                && ((metadata->a_zp)->zero_point == NULL))
                return DLP_CLSC_FAILURE;
        }

        if (metadata->a_scl != NULL) {
            if (((metadata->a_scl)->scale_factor_len > 0)
                && ((metadata->a_scl)->scale_factor == NULL))
                return DLP_CLSC_FAILURE;
        }

        if (metadata->b_zp != NULL) {
            /* check for validity of pre-ops */
            if (((metadata->b_zp)->zero_point_len > 0)
                && ((metadata->b_zp)->zero_point == NULL))
                return DLP_CLSC_FAILURE;
        }

        if (metadata->b_scl != NULL) {
            if (((metadata->b_scl)->scale_factor_len > 0)
                && ((metadata->b_scl)->scale_factor == NULL))
                return DLP_CLSC_FAILURE;
        }

        if (((metadata->a_scl)->scale_factor_type)
            != ((metadata->b_scl)->scale_factor_type)) {
            dlp_print_msg(" A and B scale factor type mismatch. Exiting..",
                          __FILE__, __LINE__);
            return DLP_CLSC_FAILURE;
        }

        // Not supporting zero-point for now.
        // if( ( ( metadata->a_zp )->zero_point_type ) != ( (
        // metadata->b_zp )->zero_point_type ) )
        // {
        // 	dlp_print_msg(" A and B zero point type mismatch. Exiting..",
        // __FILE__, __LINE__ ); 	return DLP_CLSC_FAILURE;
        // }

        DLP_TYPE tmp_zp_stor_type =
            DLP_INVALID; // get_stor_type( ( metadata->a_zp )->zero_point_type
                         // );

        // At this point we are sure that sf and zp types of both matrices
        // match.
        DLP_TYPE tmp_sf_stor_type = DLP_INVALID;
        if (metadata->a_scl != NULL) {
            tmp_sf_stor_type =
                get_stor_type((metadata->a_scl)->scale_factor_type);
        }

        lpgemm_set_group_post_ops_node_params(
            post_op_list, group_size,
            // A zero-point
            (metadata->a_zp == NULL) ? NULL : (metadata->a_zp)->zero_point,
            // A scale factor
            (metadata->a_scl == NULL) ? NULL : (metadata->a_scl)->scale_factor,
            // A zero-point length
            (metadata->a_zp == NULL) ? 0 : (metadata->a_zp)->zero_point_len,
            // A scale factor length
            (metadata->a_scl == NULL) ? 0 : (metadata->a_scl)->scale_factor_len,
            // B zero-point
            (metadata->b_zp == NULL) ? NULL : (metadata->b_zp)->zero_point,
            // B scale factor
            (metadata->b_scl == NULL) ? NULL : (metadata->b_scl)->scale_factor,
            // B zero-point length
            (metadata->b_zp == NULL) ? 0 : (metadata->b_zp)->zero_point_len,
            // B scale factor length
            (metadata->b_scl == NULL) ? 0 : (metadata->b_scl)->scale_factor_len,
            tmp_sf_stor_type, tmp_zp_stor_type);

        // Simulating linked link using an array.
        if (i < (metadata->seq_length - 1)) {
            (post_op_list + i)->next = (post_op_list + i + 1);
        }
    }

    return DLP_CLSC_SUCCESS;
}

dlp_clsc_err_t
lpgemm_translate_to_pre_ops_list(dlp_pre_op*    pre_op_unparsed,
                                 lpgemm_pre_op* pre_op_list,
                                 md_t           m,
                                 md_t           n,
                                 md_t           k)
{
    (void)(m); // Unused for now, potential to be used later.
    (void)(k); // Unused for now, potential to be used later.

    if ((pre_op_unparsed == NULL) || (pre_op_unparsed->seq_length <= 0)) {
        lpgemm_set_pre_ops_node_params(pre_op_list, 0, NULL, NULL, 0, 0,
                                       DLP_INVALID);

        return DLP_CLSC_SUCCESS;
    }

    if ((pre_op_unparsed->seq_length > AOCL_MAX_POST_OPS)) {
        lpgemm_set_pre_ops_node_params(pre_op_list, 0, NULL, NULL, 0, 0,
                                       DLP_INVALID);

        dlp_print_msg(" Max supported pre-ops is 2, supplied input pre-ops"
                      " are more. Exiting..",
                      __FILE__, __LINE__);
        return DLP_CLSC_UNEXPECTED_VECTOR_DIM; // Error, seq length exceeds max
                                               // pre ops permitted.
    }

    for (md_t i = 0; i < pre_op_unparsed->seq_length; ++i) {

        /* odd group_size is supported only when group_size == k */
        md_t group_size = pre_op_unparsed->group_size;
        if ((group_size == 0) || (group_size > k) || (group_size == k))
            group_size = k;
        else if (pre_op_unparsed->group_size % 2 == 1)
            return DLP_CLSC_FAILURE;

        if (pre_op_unparsed->b_zp != NULL) {
            /* check for validity of pre-ops */
            if (((pre_op_unparsed->b_zp)->zero_point_len > 0)
                && ((pre_op_unparsed->b_zp)->zero_point == NULL))
                return DLP_CLSC_FAILURE;
        }

        if (pre_op_unparsed->b_scl != NULL) {
            if (((pre_op_unparsed->b_scl)->scale_factor_len > 0)
                && ((pre_op_unparsed->b_scl)->scale_factor == NULL))
                return DLP_CLSC_FAILURE;
        }
        lpgemm_set_pre_ops_node_params(
            pre_op_list, group_size,
            (pre_op_unparsed->b_zp == NULL)
                ? NULL
                : (pre_op_unparsed->b_zp)->zero_point,
            (pre_op_unparsed->b_scl == NULL)
                ? NULL
                : (pre_op_unparsed->b_scl)->scale_factor,
            (pre_op_unparsed->b_zp == NULL)
                ? 0
                : (pre_op_unparsed->b_zp)->zero_point_len,
            (pre_op_unparsed->b_scl == NULL)
                ? 0
                : (pre_op_unparsed->b_scl)->scale_factor_len,
            (pre_op_unparsed->b_scl == NULL)
                ? DLP_INVALID
                : (((pre_op_unparsed->b_scl)->scale_factor_type == DLP_BF16)
                       ? DLP_BF16
                       : DLP_F32));

        // Simulating linked link using an array.
        if (i < (pre_op_unparsed->seq_length - 1)) {
            (pre_op_list + i)->next = (pre_op_list + i + 1);
        }
    }

    return DLP_CLSC_SUCCESS;
}

DLP_INLINE void
lpgemm_set_node_params(lpgemm_post_op*     post_op_node,
                       LPGEMM_POST_OP_CODE op_code,
                       void*               op1,
                       void*               op2,
                       void*               op3,
                       void*               scale_factor,
                       md_t                scale_factor_len,
                       void*               bias_zp,
                       md_t                bias_zp_len,
                       DLP_TYPE            stor_type,
                       DLP_TYPE            zp_stor_type,
                       DLP_TYPE            sf_stor_type)
{
    post_op_node->op_code          = op_code;
    post_op_node->op_args1         = op1;
    post_op_node->op_args2         = op2;
    post_op_node->op_args3         = op3;
    post_op_node->scale_factor     = scale_factor;
    post_op_node->scale_factor_len = scale_factor_len;
    post_op_node->bias_zp          = bias_zp;
    post_op_node->bias_zp_len      = bias_zp_len;
    post_op_node->stor_type        = stor_type;
    post_op_node->zp_stor_type     = zp_stor_type;
    post_op_node->sf_stor_type     = sf_stor_type;
    post_op_node->next             = NULL;
}

dlp_clsc_err_t
lpgemm_translate_to_post_ops_list(dlp_metadata_t* metadata,
                                  lpgemm_post_op* post_op_list,
                                  void*           scale_buffer,
                                  void*           meta_arg,
                                  md_t            m,
                                  md_t            n)
{
    (void)(scale_buffer); // Unused for now, potential to be used later.
    (void)(m);            // Unused for now, potential to be used later.

    if (metadata == NULL) {
        lpgemm_set_node_params(post_op_list, POST_OPS_DISABLE, NULL, NULL, NULL,
                               NULL, 0, NULL, 0, DLP_INVALID, DLP_INVALID,
                               DLP_INVALID);

        return DLP_CLSC_SUCCESS;
    }

    // Check if ADQUANTIZE is present (separate from seq_vector post-ops).
    bool has_adquantize = metadata->a_post_quant != NULL;

    // If there is no ADQUANTIZE and no other post-ops (seq_vector is empty),
    // disable post-ops and return success.
    if (!has_adquantize && metadata->seq_length <= 0) {
        lpgemm_set_node_params(post_op_list, POST_OPS_DISABLE, NULL, NULL, NULL,
                               NULL, 0, NULL, 0, DLP_INVALID, DLP_INVALID,
                               DLP_INVALID);
        return DLP_CLSC_SUCCESS;
    }

    if ((metadata->seq_length > AOCL_MAX_POST_OPS)) {
        lpgemm_set_node_params(post_op_list, POST_OPS_DISABLE, NULL, NULL, NULL,
                               NULL, 0, NULL, 0, DLP_INVALID, DLP_INVALID,
                               DLP_INVALID);

        dlp_print_msg(" Max supported post-ops is 5, supplied input post-ops"
                      " are more. Exiting..",
                      __FILE__, __LINE__);
        return DLP_CLSC_UNEXPECTED_VECTOR_DIM; // Error, seq length exceeds max
                                               // post ops permitted.
    }

    // ============================================================
    // A-DEQUANTIZATION POST-OP (Special handling - NOT in seq_vector)
    // ============================================================
    // NOTE: ADQUANTIZE is NOT part of the seq_vector array. It is
    // handled separately via the a_post_quant field in metadata.
    //
    // When a_post_quant is present, ADQUANTIZE is explicitly placed as
    // the FIRST node (index 0) of the post_op_list array. All regular
    // post-ops from seq_vector are then placed starting at index 1.
    //
    // Purpose: Since the A matrix was quantized (e.g., BF16 -> S8),
    // the GEMM produces scaled results. This dequantization operation
    // corrects the scaling using:
    //   result = round((acc + b_col_sum * zp_A) / scale_A)
    //
    // Quantization modes:
    // - Per-tensor: scale_factor_len = 1 (scalar shared across all rows)
    // - Per-row/per-channel: scale_factor_len = m (each row has its own
    //   scale and/or zero-point value)
    if (has_adquantize) {
        // --- Step 1: Validate a_pre_quant existence ---
        if (metadata->a_pre_quant == NULL) {
            dlp_print_msg(
                " a_post_quant exists but a_pre_quant is NULL. Exiting..",
                __FILE__, __LINE__);
            return DLP_CLSC_FAILURE;
        }

        // --- Step 2: Validate scale factor ---
        if (((metadata->a_post_quant)->scl
             && (metadata->a_post_quant)->scl->scale_factor_len > 0)
            && ((metadata->a_post_quant)->scl->scale_factor == NULL)) {
            dlp_print_msg(" a_post_quant.scl scale_factor is NULL. Exiting..",
                          __FILE__, __LINE__);
            return DLP_CLSC_NULL_POINTER;
        }
        if ((metadata->a_post_quant)->scl
            && ((metadata->a_post_quant)->scl->scale_factor_len != 1)
            && ((metadata->a_post_quant)->scl->scale_factor_len < m)) {
            dlp_print_msg(
                " a_post_quant.scl scale factor length is < m. Exiting..",
                __FILE__, __LINE__);
            return DLP_CLSC_NULL_POINTER;
        }

        // --- Step 3: Validate zero-point ---
        if (((metadata->a_post_quant)->zp
             && (metadata->a_post_quant)->zp->zero_point_len > 0)
            && ((metadata->a_post_quant)->zp->zero_point == NULL)) {
            dlp_print_msg(" a_post_quant.zp zero_point is NULL. Exiting..",
                          __FILE__, __LINE__);
            return DLP_CLSC_NULL_POINTER;
        }
        if ((metadata->a_post_quant)->zp
            && ((metadata->a_post_quant)->zp->zero_point_len != 1)
            && ((metadata->a_post_quant)->zp->zero_point_len < m)) {
            dlp_print_msg(
                " a_post_quant.zp zero point length is < m. Exiting..",
                __FILE__, __LINE__);
            return DLP_CLSC_NULL_POINTER;
        }

        // --- Step 4: Extract storage types ---
        DLP_TYPE tmp_zp_stor_type =
            (metadata->a_post_quant)->zp
                ? get_stor_type((metadata->a_post_quant)->zp->zero_point_type)
                : DLP_INVALID;
        DLP_TYPE tmp_sf_stor_type =
            (metadata->a_post_quant)->scl
                ? get_stor_type(
                      (metadata->a_post_quant)->scl->scale_factor_type)
                : DLP_INVALID;

        // --- Step 5: Setup zero-point length pointer ---
        // For symmetric quantization (no zero-point), use zero_zp_len = 0
        static md_t zero_zp_len = 0;
        md_t*       zero_point_len_ptr =
            (metadata->a_post_quant)->zp
                      ? &((metadata->a_post_quant)->zp->zero_point_len)
                      : &zero_zp_len;

        // --- Step 6: Create ADQUANTIZE node at post_op_list[0] ---
        lpgemm_set_node_params(
            (post_op_list), POST_OPS_ADQUANTIZE,
            (metadata->a_post_quant)->zp
                ? (metadata->a_post_quant)->zp->zero_point
                : NULL,
            meta_arg, zero_point_len_ptr,
            (metadata->a_post_quant)->scl
                ? (metadata->a_post_quant)->scl->scale_factor
                : NULL,
            (metadata->a_post_quant)->scl
                ? (metadata->a_post_quant)->scl->scale_factor_len
                : 0,
            NULL, 0, DLP_INVALID, tmp_zp_stor_type, tmp_sf_stor_type);

        // --- Step 7: Link to next post-op from seq_vector if present ---
        if (metadata->seq_length > 0) {
            (post_op_list)->next = (post_op_list + 1);
        }
    }

    // ADQUANTIZE (NOT in seq_vector) occupies post_op_list[0] when present.
    // Example with ADQUANTIZE + seq_length=2:
    //   post_op_list[0] = ADQUANTIZE, [1] = seq_vector[0], [2] = seq_vector[1]
    md_t post_op_offset = (has_adquantize) ? 1 : 0;
    md_t e_i            = 0; // Multiple eltwise supported.
    md_t s_i            = 0; // Multiple sum/scale supported.
    md_t b_i            = 0; // Multiple bias supported.
    md_t m_i            = 0; // Multiple matrix add supported.
    md_t mul_i          = 0; // Multiple matrix mul supported.
    for (md_t i = post_op_offset; i < metadata->seq_length + post_op_offset;
         ++i) {
        // Dispatcher code
        switch (*(metadata->seq_vector + i - post_op_offset)) {
            case ELTWISE: {
                LPGEMM_POST_OP_CODE tmp_code      = POST_OPS_DISABLE;
                DLP_TYPE            tmp_stor_type = DLP_INVALID;
                // Eltwise algo dispatcher.
                switch ((metadata->eltwise + e_i)->algo.algo_type) {
                    case RELU:
                        tmp_code = POST_OPS_RELU;
                        break;
                    case PRELU:
                        if ((metadata->eltwise + e_i)->algo.alpha == NULL) {
                            dlp_print_msg(" Post_op.alpha is NULL. Exiting..",
                                          __FILE__, __LINE__);
                            return DLP_CLSC_NULL_POINTER;
                        }
                        // NOTE: For PRELU, the alpha parameter can be of any
                        // type and this will be stored in the stor_type for now
                        // because the static kernel relies on the stor_type and
                        // not the scale factor type.
                        tmp_stor_type = get_stor_type(
                            (metadata->eltwise + e_i)->algo.stor_type);
                        tmp_code = POST_OPS_RELU_SCALE;
                        break;
                    case GELU_TANH:
                        tmp_code = POST_OPS_GELU_TANH;
                        break;
                    case GELU_ERF:
                        tmp_code = POST_OPS_GELU_ERF;
                        break;
                    case CLIP:
                        if (((metadata->eltwise + e_i)->algo.alpha == NULL)
                            || ((metadata->eltwise + e_i)->algo.beta == NULL)) {
                            dlp_print_msg(" Post_op.clip min or max value is "
                                          "NULL. Exiting..",
                                          __FILE__, __LINE__);
                            return DLP_CLSC_NULL_POINTER;
                        }
                        // Alpha and Beta should have same storage type for CLIP
                        tmp_stor_type = get_stor_type(
                            (metadata->eltwise + e_i)->algo.stor_type);
                        tmp_code = POST_OPS_CLIP;
                        break;
                    case SWISH:
                        if ((metadata->eltwise + e_i)->algo.alpha == NULL) {
                            dlp_print_msg(" Post_op.alpha is NULL. Exiting..",
                                          __FILE__, __LINE__);
                            return DLP_CLSC_NULL_POINTER;
                        }
                        tmp_code = POST_OPS_SWISH;
                        break;
                    case TANH:
                        tmp_code = POST_OPS_TANH;
                        break;
                    case SIGMOID:
                        tmp_code = POST_OPS_SIGMOID;
                        break;
                    default:
                        break;
                }
                lpgemm_set_node_params(
                    (post_op_list + i), tmp_code, NULL,
                    (metadata->eltwise + e_i)->algo.alpha,
                    (metadata->eltwise + e_i)->algo.beta,
                    (metadata->eltwise + e_i)->sf
                        ? (metadata->eltwise + e_i)->sf->scale_factor
                        : NULL,
                    (metadata->eltwise + e_i)->sf
                        ? (metadata->eltwise + e_i)->sf->scale_factor_len
                        : 0,
                    NULL, 0, tmp_stor_type, DLP_INVALID, DLP_INVALID);
                e_i += 1;
            } break;
            case BIAS: {
                if ((metadata->bias + b_i)->bias == NULL) {
                    dlp_print_msg(" Post_op.bias is NULL. Exiting..", __FILE__,
                                  __LINE__);
                    return DLP_CLSC_NULL_POINTER;
                }

                DLP_TYPE tmp_stor_type =
                    get_stor_type((metadata->bias + b_i)->stor_type);
                // Extract SF storage type
                DLP_TYPE tmp_sf_stor_type =
                    (metadata->bias + b_i)->sf
                        ? get_stor_type(
                              (metadata->bias + b_i)->sf->scale_factor_type)
                        : DLP_INVALID;

                DLP_TYPE tmp_zp_stor_type =
                    (metadata->bias + b_i)->zp
                        ? get_stor_type(
                              (metadata->bias + b_i)->zp->zero_point_type)
                        : DLP_INVALID;

                lpgemm_set_node_params(
                    (post_op_list + i), POST_OPS_BIAS,
                    (metadata->bias + b_i)->bias, meta_arg,
                    NULL, // op_args3 is NULL for BIAS
                    (metadata->bias + b_i)->sf
                        ? (metadata->bias + b_i)->sf->scale_factor
                        : NULL,
                    (metadata->bias + b_i)->sf
                        ? (metadata->bias + b_i)->sf->scale_factor_len
                        : 0,
                    (metadata->bias + b_i)->zp
                        ? (metadata->bias + b_i)->zp->zero_point
                        : NULL,
                    (metadata->bias + b_i)->zp
                        ? (metadata->bias + b_i)->zp->zero_point_len
                        : 0,
                    tmp_stor_type, tmp_zp_stor_type, tmp_sf_stor_type);

                b_i += 1;
            } break;
            case SCALE: {
                if (((metadata->scale + s_i)->sf
                     && (metadata->scale + s_i)->sf->scale_factor_len > 0)
                    && ((metadata->scale + s_i)->sf->scale_factor == NULL)) {
                    dlp_print_msg(
                        " Post_op.scale scale_factor is NULL. Exiting..",
                        __FILE__, __LINE__);
                    return DLP_CLSC_NULL_POINTER;
                }
                if (((metadata->scale + s_i)->zp
                     && (metadata->scale + s_i)->zp->zero_point_len > 0)
                    && ((metadata->scale + s_i)->zp->zero_point == NULL)) {
                    dlp_print_msg(
                        " Post_op.scale zero_point is NULL. Exiting..",
                        __FILE__, __LINE__);
                    return DLP_CLSC_NULL_POINTER;
                }
                if ((metadata->scale + s_i)->sf
                    && ((metadata->scale + s_i)->sf->scale_factor_len != 1)
                    && ((metadata->scale + s_i)->sf->scale_factor_len < n)) {
                    dlp_print_msg(" Post_op.scale scale factor length is < n."
                                  " Exiting..",
                                  __FILE__, __LINE__);
                    return DLP_CLSC_NULL_POINTER;
                }
                if ((metadata->scale + s_i)->zp
                    && ((metadata->scale + s_i)->zp->zero_point_len != 1)
                    && ((metadata->scale + s_i)->zp->zero_point_len < n)) {
                    dlp_print_msg(" Post_op.scale zero point length is < n."
                                  " Exiting..",
                                  __FILE__, __LINE__);
                    return DLP_CLSC_NULL_POINTER;
                }

                DLP_TYPE tmp_zp_stor_type =
                    (metadata->scale + s_i)->zp
                        ? get_stor_type(
                              (metadata->scale + s_i)->zp->zero_point_type)
                        : DLP_INVALID;
                DLP_TYPE tmp_sf_stor_type =
                    (metadata->scale + s_i)->sf
                        ? get_stor_type(
                              (metadata->scale + s_i)->sf->scale_factor_type)
                        : DLP_INVALID;

                md_t* zero_point_len_ptr =
                    (metadata->scale + s_i)->zp
                        ? &((metadata->scale + s_i)->zp->zero_point_len)
                        : NULL;

                lpgemm_set_node_params(
                    (post_op_list + i), POST_OPS_DOWNSCALE,
                    (metadata->scale + s_i)->zp
                        ? (metadata->scale + s_i)->zp->zero_point
                        : NULL,
                    meta_arg, zero_point_len_ptr,
                    (metadata->scale + s_i)->sf
                        ? (metadata->scale + s_i)->sf->scale_factor
                        : NULL,
                    (metadata->scale + s_i)->sf
                        ? (metadata->scale + s_i)->sf->scale_factor_len
                        : 0,
                    NULL, 0, DLP_INVALID, tmp_zp_stor_type, tmp_sf_stor_type);

                s_i += 1;
            } break;
            case MATRIX_ADD: {
                if (((metadata->matrix_add + m_i)->matrix == NULL)
                    || ((metadata->matrix_add + m_i)->ldm <= 0)) {
                    dlp_print_msg(
                        " Post_op.matrix_add attributes are invalid. Exiting..",
                        __FILE__, __LINE__);
                    return DLP_CLSC_NULL_POINTER;
                }
                DLP_TYPE tmp_stor_type =
                    get_stor_type((metadata->matrix_add + m_i)->stor_type);

                /* Get scale factor storage type */
                DLP_TYPE sf_stor_type =
                    (metadata->matrix_add + m_i)->sf
                        ? get_stor_type((metadata->matrix_add + m_i)
                                            ->sf->scale_factor_type)
                        : DLP_INVALID;

                lpgemm_set_node_params(
                    (post_op_list + i), POST_OPS_MATRIX_ADD,
                    (metadata->matrix_add + m_i)->matrix, meta_arg,
                    &((metadata->matrix_add + m_i)->ldm),
                    (metadata->matrix_add + m_i)->sf
                        ? (metadata->matrix_add + m_i)->sf->scale_factor
                        : NULL,
                    (metadata->matrix_add + m_i)->sf
                        ? (metadata->matrix_add + m_i)->sf->scale_factor_len
                        : 0,
                    NULL, 0, tmp_stor_type, DLP_INVALID, sf_stor_type);

                m_i += 1;
            } break;
            case MATRIX_MUL: {
                if (((metadata->matrix_mul + mul_i)->matrix == NULL)
                    || ((metadata->matrix_mul + mul_i)->ldm <= 0)) {
                    dlp_print_msg(
                        " Post_op.matrix_mul attributes are invalid. Exiting..",
                        __FILE__, __LINE__);
                    return DLP_CLSC_NULL_POINTER;
                }
                DLP_TYPE tmp_stor_type =
                    get_stor_type((metadata->matrix_mul + mul_i)->stor_type);

                /* Get scale factor storage type */
                DLP_TYPE sf_stor_type =
                    (metadata->matrix_mul + mul_i)->sf
                        ? get_stor_type((metadata->matrix_mul + mul_i)
                                            ->sf->scale_factor_type)
                        : DLP_INVALID;

                lpgemm_set_node_params(
                    (post_op_list + i), POST_OPS_MATRIX_MUL,
                    (metadata->matrix_mul + mul_i)->matrix, meta_arg,
                    &((metadata->matrix_mul + mul_i)->ldm),
                    (metadata->matrix_mul + mul_i)->sf
                        ? (metadata->matrix_mul + mul_i)->sf->scale_factor
                        : NULL,
                    (metadata->matrix_mul + mul_i)->sf
                        ? (metadata->matrix_mul + mul_i)->sf->scale_factor_len
                        : 0,
                    NULL, 0, tmp_stor_type, DLP_INVALID, sf_stor_type);

                mul_i += 1;
            } break;
            default:
                break;
        }

        // Simulating linked list using an array.
        // NOTE: ADQUANTIZE linking to next node is handled separately above
        if (i < (metadata->seq_length + post_op_offset - 1)) {
            (post_op_list + i)->next = (post_op_list + i + 1);
        }
    }
    return DLP_CLSC_SUCCESS;
}
