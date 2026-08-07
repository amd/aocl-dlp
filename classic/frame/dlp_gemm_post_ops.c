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

#include "dlp_gemm_post_ops.h"
#include "dlp_gemm_types.h"
#include "gemm_utils/dlp_gemm_utils.h"

bool
dlp_gemm_post_op_list_has_jit_only_op(const dlp_gemm_post_op* post_op_list)
{
    bool has_jit_only = false;
    for (const dlp_gemm_post_op* node                    = post_op_list;
         (node != NULL) && (has_jit_only == false); node = node->next) {
        if (node->op_code > DLP_CLASSIC_MAX_POST_OP_CODE) {
            has_jit_only = true;
        }
        /* Per-token (PER_TOKEN, len == m) DOWNSCALE is implemented only by
         * the JIT generator. The classic kernels' DOWNSCALE labels branch
         * on scale_factor_len in {1, n} and would index past the SF buffer
         * for a per-row vector. Force JIT routing to keep the classic path
         * safe from per-token inputs. */
        else if (node->op_code == POST_OPS_DOWNSCALE
                 && node->scale_factor_dim == DLP_PARAM_DIM_PER_TOKEN) {
            has_jit_only = true;
        }
    }
    return has_jit_only;
}

bool
dlp_gemm_post_op_list_has_shape_changing_glu(
    const dlp_gemm_post_op* post_op_list)
{
    return ((post_op_list != NULL) && (post_op_list->glu_d != NULL)
            && (post_op_list->glu_ld_d > 0));
}

static inline DLP_TYPE
dlp_gemm_get_stor_type(DLP_TYPE pstor_type)
{
    DLP_TYPE stor_type = DLP_INVALID;
    switch (pstor_type) {
        case DLP_F32:
            stor_type = DLP_F32;
            break;
        case DLP_BF16:
            stor_type = DLP_BF16;
            break;
        case DLP_F16:
            stor_type = DLP_F16;
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

static inline dlp_sf_t
dlp_gemm_qparam_to_sf(const dlp_qparam_t* qparam)
{
    dlp_sf_t sf = { 0 };
    if (qparam != NULL) {
        sf.scale_factor_len  = qparam->len;
        sf.scale_factor_type = qparam->stor_type;
        sf.scale_factor_dim  = qparam->outer_dim;
        // Keep len/type/dim so caller-side validation can report the error
        // when a non-empty qparam has no backing data.
        if (qparam->data != NULL) {
            sf.scale_factor = qparam->data;
        }
    }
    return sf;
}

static inline dlp_zp_t
dlp_gemm_qparam_to_zp(const dlp_qparam_t* qparam)
{
    dlp_zp_t zp = { 0 };
    if (qparam != NULL) {
        zp.zero_point_len  = qparam->len;
        zp.zero_point_type = qparam->stor_type;
        // Keep len/type so caller-side validation can report the error when a
        // non-empty qparam has no backing data.
        if (qparam->data != NULL) {
            zp.zero_point = qparam->data;
        }
    }
    return zp;
}

DLP_INLINE void
dlp_gemm_set_pre_ops_node_params(dlp_gemm_pre_op* pre_op_node,
                                 md_t             group_size,
                                 void*            zero_point,
                                 void*            scale_factor,
                                 md_t             zero_point_len,
                                 md_t             scale_factor_len,
                                 md_t             scale_factor_type,
                                 md_t             zero_point_type)
{
    pre_op_node->group_size        = group_size;
    pre_op_node->scale_factor      = scale_factor;
    pre_op_node->scale_factor_len  = scale_factor_len;
    pre_op_node->zp                = zero_point;
    pre_op_node->zp_len            = zero_point_len;
    pre_op_node->scale_factor_type = scale_factor_type;
    pre_op_node->zp_type           = zero_point_type;
    pre_op_node->next              = NULL;
}

DLP_INLINE void
dlp_gemm_set_group_post_ops_node_params(dlp_gemm_group_post_op* post_op_node,
                                        md_t                    group_size,
                                        DLP_PARAM_DIM_TYPE a_scale_factor_dim,
                                        DLP_PARAM_DIM_TYPE b_scale_factor_dim,
                                        void*              a_zero_point,
                                        void*              a_scale_factor,
                                        md_t               a_zero_point_len,
                                        md_t               a_scale_factor_len,
                                        void*              b_zero_point,
                                        void*              b_scale_factor,
                                        md_t               b_zero_point_len,
                                        md_t               b_scale_factor_len,
                                        DLP_TYPE           sf_stor_type,
                                        DLP_TYPE           zp_stor_type)
{
    post_op_node->group_size         = group_size;
    post_op_node->a_scale_factor_dim = a_scale_factor_dim;
    post_op_node->b_scale_factor_dim = b_scale_factor_dim;
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
dlp_gemm_translate_to_group_postops_list(dlp_quant_op_t*         a_quant_op,
                                         dlp_quant_op_t*         b_quant_op,
                                         dlp_gemm_group_post_op* post_op_list,
                                         md_t                    m,
                                         md_t                    n,
                                         md_t                    k)
{
    dlp_sf_t a_scl = dlp_gemm_qparam_to_sf(
        a_quant_op == NULL ? NULL : a_quant_op->dequant_scale_factors);
    dlp_sf_t b_scl = dlp_gemm_qparam_to_sf(
        b_quant_op == NULL ? NULL : b_quant_op->dequant_scale_factors);

    dlp_sf_t* a_scl_ptr =
        (a_quant_op == NULL || a_quant_op->dequant_scale_factors == NULL)
            ? NULL
            : &a_scl;
    dlp_sf_t* b_scl_ptr =
        (b_quant_op == NULL || b_quant_op->dequant_scale_factors == NULL)
            ? NULL
            : &b_scl;

    if ((a_quant_op == NULL) && (b_quant_op == NULL)) {
        dlp_gemm_set_group_post_ops_node_params(
            post_op_list, 0, DLP_PARAM_DIM_PER_GROUP, DLP_PARAM_DIM_PER_GROUP,
            NULL, NULL, 0, 0, NULL, NULL, 0, 0, DLP_INVALID, DLP_INVALID);

        return DLP_CLSC_SUCCESS;
    }

    if ((a_quant_op != NULL)
        && (a_quant_op->quant_op_kind != DLP_QUANT_OP_QUANTIZE)) {
        dlp_print_msg(
            " A grouped quant metadata must use DLP_QUANT_OP_QUANTIZE. "
            "Exiting..",
            __FILE__, __LINE__);
        return DLP_CLSC_NOT_SUPPORTED;
    }
    if ((b_quant_op != NULL)
        && (b_quant_op->quant_op_kind != DLP_QUANT_OP_QUANTIZE)) {
        dlp_print_msg(
            " B grouped quant metadata must use DLP_QUANT_OP_QUANTIZE. "
            "Exiting..",
            __FILE__, __LINE__);
        return DLP_CLSC_NOT_SUPPORTED;
    }

    if ((a_quant_op != NULL) && (a_quant_op->zero_point != NULL)) {
        dlp_print_msg(" A zero-point is not supported for grouped "
                      "symmetric quantization. Exiting..",
                      __FILE__, __LINE__);
        return DLP_CLSC_NOT_SUPPORTED;
    }
    if ((b_quant_op != NULL) && (b_quant_op->zero_point != NULL)) {
        dlp_print_msg(" B zero-point is not supported for grouped "
                      "symmetric quantization. Exiting..",
                      __FILE__, __LINE__);
        return DLP_CLSC_NOT_SUPPORTED;
    }

    md_t a_group_size = (a_quant_op != NULL) ? a_quant_op->group_size : 0;
    md_t b_group_size = (b_quant_op != NULL) ? b_quant_op->group_size : 0;

    if (a_group_size == 0) {
        a_group_size = k;
    }
    if (b_group_size == 0) {
        b_group_size = k;
    }

    if ((a_quant_op != NULL) && (b_quant_op != NULL)
        && (a_group_size != b_group_size)) {
        dlp_print_msg(" A and B group size mismatch. Exiting..", __FILE__,
                      __LINE__);
        return DLP_CLSC_INVALID_GROUP_DIMENSION;
    }

    md_t group_size = (a_quant_op != NULL) ? a_group_size : b_group_size;

    // Group ops may pass group_size 0 to mean "default"
    // (one group over full k).
    if ((group_size > k) || (group_size < 0)) {
        return DLP_CLSC_INVALID_GROUP_DIMENSION;
    } else if ((group_size != k) && (group_size % 4 != 0)) {
        return DLP_CLSC_INVALID_GROUP_DIMENSION;
    }

    DLP_PARAM_DIM_TYPE a_scale_factor_dim = (a_scl_ptr != NULL)
                                                ? a_scl_ptr->scale_factor_dim
                                                : DLP_PARAM_DIM_INVALID;
    DLP_PARAM_DIM_TYPE b_scale_factor_dim = (b_scl_ptr != NULL)
                                                ? b_scl_ptr->scale_factor_dim
                                                : DLP_PARAM_DIM_INVALID;

    if (a_scale_factor_dim == DLP_PARAM_DIM_INVALID
        || b_scale_factor_dim == DLP_PARAM_DIM_INVALID) {
        dlp_print_msg(" A or B scale factor dimension is invalid. Exiting..",
                      __FILE__, __LINE__);
        return DLP_CLSC_NOT_SUPPORTED;
    }

    // Number of K-groups per matrix.
    // A PER_TOKEN:   A has one scale per row (a_grp_mul=0)
    // B PER_CHANNEL: B has one scale per column (b_grp_mul=0)
    // otherwise:     per group (default)
    md_t num_groups = (k + group_size - 1) / group_size;
    md_t a_num_groups =
        (a_scale_factor_dim == DLP_PARAM_DIM_PER_TOKEN) ? 1 : num_groups;
    md_t b_num_groups =
        (b_scale_factor_dim == DLP_PARAM_DIM_PER_CHANNEL) ? 1 : num_groups;

    if (a_scl_ptr != NULL) {
        if ((a_scl_ptr->scale_factor_len > 0)
            && (a_scl_ptr->scale_factor == NULL))
            return DLP_CLSC_NULL_POINTER;

        if (a_scl_ptr->scale_factor_len < (m * a_num_groups))
            return DLP_CLSC_INVALID_SF_LEN;
    }

    if (b_scl_ptr != NULL) {
        if ((b_scl_ptr->scale_factor_len > 0)
            && (b_scl_ptr->scale_factor == NULL))
            return DLP_CLSC_NULL_POINTER;

        if (b_scl_ptr->scale_factor_len < (n * b_num_groups))
            return DLP_CLSC_INVALID_SF_LEN;
    }

    if ((a_scl_ptr != NULL) && (b_scl_ptr != NULL)
        && (a_scl_ptr->scale_factor_type != b_scl_ptr->scale_factor_type)) {
        dlp_print_msg(" A and B scale factor type mismatch. Exiting..",
                      __FILE__, __LINE__);
        return DLP_CLSC_TYPE_MISMATCH;
    }

    DLP_TYPE tmp_sf_stor_type = DLP_INVALID;
    if (a_scl_ptr != NULL) {
        tmp_sf_stor_type = dlp_gemm_get_stor_type(a_scl_ptr->scale_factor_type);
    }

    dlp_gemm_set_group_post_ops_node_params(
        post_op_list, group_size, a_scale_factor_dim, b_scale_factor_dim, NULL,
        (a_scl_ptr == NULL) ? NULL : a_scl_ptr->scale_factor, 0,
        (a_scl_ptr == NULL) ? 0 : a_scl_ptr->scale_factor_len, NULL,
        (b_scl_ptr == NULL) ? NULL : b_scl_ptr->scale_factor, 0,
        (b_scl_ptr == NULL) ? 0 : b_scl_ptr->scale_factor_len, tmp_sf_stor_type,
        DLP_INVALID);
    post_op_list->next = NULL;

    return DLP_CLSC_SUCCESS;
}

dlp_clsc_err_t
dlp_gemm_translate_to_group_op_list(const dlp_metadata_t* metadata,
                                    dlp_group_op*         group_op,
                                    md_t                  m,
                                    md_t                  n,
                                    md_t                  k)
{
    (void)(m);
    (void)(n);
    (void)(k);

    if ((group_op == NULL) || (metadata == NULL)) {
        return DLP_CLSC_NULL_POINTER;
    }

    // Both quant ops are validated by dlp_gemm_translate_to_group_postops_list,
    // which every caller runs (and error-checks) before reaching this point.
    group_op->a_post_quant_op = metadata->a_quant_op;
    group_op->b_post_quant_op = metadata->b_quant_op;

    return DLP_CLSC_SUCCESS;
}

dlp_clsc_err_t
dlp_gemm_translate_to_pre_ops_list(dlp_quant_op_t*  b_quant_op,
                                   dlp_gemm_pre_op* pre_op_list,
                                   md_t             m,
                                   md_t             n,
                                   md_t             k)
{
    (void)(m); // Unused for now, potential to be used later.
    (void)(n); // Unused for now, potential to be used later.

    if (b_quant_op == NULL) {
        dlp_gemm_set_pre_ops_node_params(pre_op_list, 0, NULL, NULL, 0, 0,
                                         DLP_INVALID, DLP_INVALID);

        return DLP_CLSC_SUCCESS;
    }

    if ((b_quant_op->quant_op_kind != DLP_QUANT_OP_DEQUANTIZE)
        && (b_quant_op->quant_op_kind != DLP_QUANT_OP_EXPAND)) {
        dlp_print_msg(
            " B pre-op quant metadata must use DLP_QUANT_OP_DEQUANTIZE or "
            "DLP_QUANT_OP_EXPAND. Exiting..",
            __FILE__, __LINE__);
        return DLP_CLSC_NOT_SUPPORTED;
    }

    dlp_sf_t b_scl = dlp_gemm_qparam_to_sf(b_quant_op->dequant_scale_factors);
    dlp_zp_t b_zp  = dlp_gemm_qparam_to_zp(b_quant_op->zero_point);

    dlp_sf_t* b_scl_ptr = (b_quant_op->dequant_scale_factors == NULL) ? NULL
                                                                      : &b_scl;
    dlp_zp_t* b_zp_ptr  = (b_quant_op->zero_point == NULL) ? NULL : &b_zp;

    md_t group_size = b_quant_op->group_size;

    // WOQ and similar pre-ops may pass group_size 0 to mean "default"
    // (one group over full k).
    if (group_size == 0) {
        group_size = k;
    }

    if ((group_size > k) || (group_size < 0)) {
        return DLP_CLSC_INVALID_GROUP_DIMENSION;
    } else if ((group_size != k) && (group_size % 2 != 0)) {
        return DLP_CLSC_INVALID_GROUP_DIMENSION;
    }

    for (iter_t i = 0; i < 1; ++i) {
        if (b_zp_ptr != NULL) {
            /* check for validity of pre-ops */
            if ((b_zp_ptr->zero_point_len > 0)
                && (b_zp_ptr->zero_point == NULL))
                return DLP_CLSC_NULL_POINTER;
        }

        if ((b_quant_op->quant_op_kind == DLP_QUANT_OP_DEQUANTIZE)
            && (b_scl_ptr == NULL)) {
            return DLP_CLSC_NULL_POINTER;
        }

        if (b_scl_ptr != NULL) {
            if ((b_scl_ptr->scale_factor_len > 0)
                && (b_scl_ptr->scale_factor == NULL))
                return DLP_CLSC_NULL_POINTER;
        }
        dlp_gemm_set_pre_ops_node_params(
            (pre_op_list + i), group_size,
            (b_zp_ptr == NULL) ? NULL : b_zp_ptr->zero_point,
            (b_scl_ptr == NULL) ? NULL : b_scl_ptr->scale_factor,
            (b_zp_ptr == NULL) ? 0 : b_zp_ptr->zero_point_len,
            (b_scl_ptr == NULL) ? 0 : b_scl_ptr->scale_factor_len,
            (b_scl_ptr == NULL)
                ? DLP_INVALID
                : ((b_scl_ptr->scale_factor_type == DLP_BF16) ? DLP_BF16
                                                              : DLP_F32),
            (b_zp_ptr == NULL) ? DLP_INVALID : b_zp_ptr->zero_point_type);

        (pre_op_list + i)->next = NULL;
    }

    return DLP_CLSC_SUCCESS;
}

// Identity scale used when a MATRIX_ADD / MATRIX_MUL post-op carries no scale
// factor. The grouped (sym_quant) micro-kernels only special-case
// scale_factor_len == 1 and otherwise dereference scale_factor directly, so a
// NULL/len-0 scale factor faults. Handing them a per-tensor 1.0f keeps the
// operand un-scaled (identity) while taking the safe scalar-broadcast path.
static const float dlp_matrix_op_identity_scale = 1.0f;

DLP_INLINE void
dlp_gemm_set_node_params(dlp_gemm_post_op*     post_op_node,
                         DLP_GEMM_POST_OP_CODE op_code,
                         void*                 op1,
                         void*                 op2,
                         void*                 op3,
                         void*                 scale_factor,
                         md_t                  scale_factor_len,
                         void*                 bias_zp,
                         md_t                  bias_zp_len,
                         DLP_TYPE              stor_type,
                         DLP_TYPE              zp_stor_type,
                         DLP_TYPE              sf_stor_type,
                         DLP_PARAM_DIM_TYPE    scale_factor_dim)
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
    post_op_node->scale_factor_dim = scale_factor_dim;
    post_op_node->glu_d            = NULL;
    post_op_node->glu_ld_d         = 0;
    post_op_node->next             = NULL;
}

dlp_clsc_err_t
dlp_gemm_translate_adquantize_post_op(dlp_metadata_t*   metadata,
                                      dlp_gemm_post_op* post_op_list,
                                      void*             meta_arg,
                                      md_t              m)
{
    // Step 1: Validate metadata and A quantization parameters.
    if (metadata == NULL || post_op_list == NULL || metadata->a_quant_op == NULL
        || metadata->a_quant_op->quant_scale_factors == NULL
        || metadata->a_quant_op->dequant_scale_factors == NULL) {
        dlp_print_msg("One or more required A quantization parameters are "
                      "NULL. Exiting..",
                      __FILE__, __LINE__);
        return DLP_CLSC_NULL_POINTER;
    }

    if (metadata->a_quant_op->quant_op_kind != DLP_QUANT_OP_QUANTIZE) {
        dlp_print_msg(
            " a_quant_op quant metadata must use DLP_QUANT_OP_QUANTIZE. "
            "Exiting..",
            __FILE__, __LINE__);
        return DLP_CLSC_NOT_SUPPORTED;
    }

    dlp_qparam_t* a_dequant_scl = metadata->a_quant_op->dequant_scale_factors;
    dlp_qparam_t* a_zp          = metadata->a_quant_op->zero_point;

    // --- Step 2: Validate scale factor ---
    if ((a_dequant_scl->len > 0) && (a_dequant_scl->data == NULL)) {
        dlp_print_msg(" a_quant_op.dequant_scale_factors data is NULL. "
                      "Exiting..",
                      __FILE__, __LINE__);
        return DLP_CLSC_NULL_POINTER;
    }
    if ((a_dequant_scl->len != 1) && (a_dequant_scl->len < m)) {
        dlp_print_msg(" a_quant_op.dequant_scale_factors len is < m. "
                      "Exiting..",
                      __FILE__, __LINE__);
        return DLP_CLSC_UNEXPECTED_VECTOR_DIM;
    }

    // --- Step 3: Validate zero-point ---
    if ((a_zp != NULL) && (a_zp->len > 0) && (a_zp->data == NULL)) {
        dlp_print_msg(" a_quant_op.zero_point data is NULL. Exiting..",
                      __FILE__, __LINE__);
        return DLP_CLSC_NULL_POINTER;
    }
    if ((a_zp != NULL) && (a_zp->len != 1) && (a_zp->len < m)) {
        dlp_print_msg(" a_quant_op.zero_point len is < m. Exiting..", __FILE__,
                      __LINE__);
        return DLP_CLSC_UNEXPECTED_VECTOR_DIM;
    }

    // --- Step 4: Extract storage types ---
    DLP_TYPE tmp_zp_stor_type = a_zp ? dlp_gemm_get_stor_type(a_zp->stor_type)
                                     : DLP_INVALID;
    DLP_TYPE tmp_sf_stor_type =
        dlp_gemm_get_stor_type(a_dequant_scl->stor_type);

    // --- Step 5: Validate outer_dim ---
    if ((a_dequant_scl->outer_dim != DLP_PARAM_DIM_PER_TENSOR)
        && (a_dequant_scl->outer_dim != DLP_PARAM_DIM_PER_TOKEN)) {
        dlp_print_msg(" a_quant_op.dequant_scale_factors outer_dim must be "
                      "PER_TENSOR or PER_TOKEN for ADQUANTIZE. Exiting..",
                      __FILE__, __LINE__);
        return DLP_CLSC_UNEXPECTED_VECTOR_DIM;
    }

    // --- Step 6: Setup zero-point length pointer ---
    // For symmetric quantization (no zero-point), use zero_zp_len = 0
    static md_t zero_zp_len        = 0;
    md_t*       zero_point_len_ptr = a_zp ? &(a_zp->len) : &zero_zp_len;

    // --- Step 7: Create ADQUANTIZE node at post_op_list[0] ---
    dlp_gemm_set_node_params(
        post_op_list, POST_OPS_ADQUANTIZE, a_zp ? a_zp->data : NULL, meta_arg,
        zero_point_len_ptr, a_dequant_scl->data, a_dequant_scl->len, NULL, 0,
        DLP_INVALID, tmp_zp_stor_type, tmp_sf_stor_type,
        a_dequant_scl->outer_dim);

    // --- Step 8: Link to seq_vector post-ops (filled at post_op_list+1) ---
    if (metadata->seq_length > 0) {
        (post_op_list)->next = (post_op_list + 1);
    }
    return DLP_CLSC_SUCCESS;
}

dlp_clsc_err_t
dlp_gemm_translate_to_post_ops_list(dlp_metadata_t*   metadata,
                                    dlp_gemm_post_op* post_op_list,
                                    void*             scale_buffer,
                                    void*             meta_arg,
                                    md_t              m,
                                    md_t              n)
{
    (void)(scale_buffer); // Unused for now, potential to be used later.

    if (post_op_list == NULL) {
        return DLP_CLSC_NULL_POINTER;
    }

    if (metadata == NULL || (metadata->seq_length <= 0)) {
        dlp_gemm_set_node_params(
            post_op_list, POST_OPS_DISABLE, NULL, NULL, NULL, NULL, 0, NULL, 0,
            DLP_INVALID, DLP_INVALID, DLP_INVALID, DLP_PARAM_DIM_INVALID);

        return DLP_CLSC_SUCCESS;
    }

    if ((metadata->seq_length > AOCL_DLP_MAX_POST_OPS)) {
        dlp_gemm_set_node_params(
            post_op_list, POST_OPS_DISABLE, NULL, NULL, NULL, NULL, 0, NULL, 0,
            DLP_INVALID, DLP_INVALID, DLP_INVALID, DLP_PARAM_DIM_INVALID);

        dlp_print_msg(" Max supported post-ops is 8, supplied input post-ops"
                      " are more. Exiting..",
                      __FILE__, __LINE__);
        return DLP_CLSC_UNEXPECTED_VECTOR_DIM; // Error, seq length exceeds max
                                               // post ops permitted.
    }

    if (metadata->seq_vector == NULL) {
        dlp_print_msg(" seq_vector is NULL. Exiting..", __FILE__, __LINE__);
        return DLP_CLSC_NULL_POINTER;
    }

    md_t        e_i         = 0; // Multiple eltwise supported.
    md_t        s_i         = 0; // Multiple sum/scale supported.
    md_t        b_i         = 0; // Multiple bias supported.
    md_t        m_i         = 0; // Multiple matrix add supported.
    md_t        mul_i       = 0; // Multiple matrix mul supported.
    static md_t zero_zp_len = 0;
    for (iter_t i = 0; i < metadata->seq_length; ++i) {
        // Dispatcher code
        switch (*(metadata->seq_vector + i)) {
            case ELTWISE: {
                if (metadata->eltwise == NULL) {
                    dlp_print_msg(" Post_op.eltwise is NULL. Exiting..",
                                  __FILE__, __LINE__);
                    return DLP_CLSC_NULL_POINTER;
                }
                DLP_GEMM_POST_OP_CODE tmp_code      = POST_OPS_DISABLE;
                DLP_TYPE              tmp_stor_type = DLP_INVALID;
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
                        tmp_stor_type = dlp_gemm_get_stor_type(
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
                        tmp_stor_type = dlp_gemm_get_stor_type(
                            (metadata->eltwise + e_i)->algo.stor_type);
                        tmp_code = POST_OPS_CLIP;
                        break;
                    case SWISH:
                        if ((metadata->eltwise + e_i)->algo.alpha == NULL) {
                            dlp_print_msg(" Post_op.alpha is NULL. Exiting..",
                                          __FILE__, __LINE__);
                            return DLP_CLSC_NULL_POINTER;
                        }
                        tmp_stor_type = dlp_gemm_get_stor_type(
                            (metadata->eltwise + e_i)->algo.stor_type);
                        tmp_code = POST_OPS_SWISH;
                        break;
                    case TANH:
                        tmp_code = POST_OPS_TANH;
                        break;
                    case SIGMOID:
                        tmp_code = POST_OPS_SIGMOID;
                        break;
                    case MISH:
                        tmp_code = POST_OPS_MISH;
                        break;
                    default:
                        break;
                }

                /* ELTWISE supports only per-tensor (len == 1) or
                 * per-channel (len == n) scale factors. */
                if ((metadata->eltwise + e_i)->sf
                    && ((metadata->eltwise + e_i)->sf->scale_factor_len != 1)
                    && ((metadata->eltwise + e_i)->sf->scale_factor_len != n)) {
                    dlp_print_msg(" ELTWISE scale_factor_len is != 1 and"
                                  " != n. Exiting..",
                                  __FILE__, __LINE__);
                    return DLP_CLSC_UNEXPECTED_VECTOR_DIM;
                }

                dlp_gemm_set_node_params(
                    (post_op_list + i), tmp_code, NULL,
                    (metadata->eltwise + e_i)->algo.alpha,
                    (metadata->eltwise + e_i)->algo.beta,
                    (metadata->eltwise + e_i)->sf
                        ? (metadata->eltwise + e_i)->sf->scale_factor
                        : NULL,
                    (metadata->eltwise + e_i)->sf
                        ? (metadata->eltwise + e_i)->sf->scale_factor_len
                        : 0,
                    NULL, 0, tmp_stor_type, DLP_INVALID, DLP_INVALID,
                    /* ELTWISE supports per-tensor or per-channel SF only;
                     * len fully determines the dim. */
                    (metadata->eltwise + e_i)->sf
                        ? (((metadata->eltwise + e_i)->sf->scale_factor_len
                            == 1)
                               ? DLP_PARAM_DIM_PER_TENSOR
                               : DLP_PARAM_DIM_PER_CHANNEL)
                        : DLP_PARAM_DIM_INVALID);
                e_i += 1;
            } break;
            case BIAS: {
                if (metadata->bias == NULL) {
                    dlp_print_msg(" Post_op.bias is NULL. Exiting..", __FILE__,
                                  __LINE__);
                    return DLP_CLSC_NULL_POINTER;
                }
                if ((metadata->bias + b_i)->bias == NULL) {
                    dlp_print_msg(
                        " Post_op.bias array pointer is NULL. Exiting..",
                        __FILE__, __LINE__);
                    return DLP_CLSC_NULL_POINTER;
                }

                if ((metadata->bias + b_i)->bias
                    && ((metadata->bias + b_i)->bias_len != 1)
                    && ((metadata->bias + b_i)->bias_len != n)) {
                    dlp_print_msg(" BIAS bias_len is != 1 and != n. Exiting..",
                                  __FILE__, __LINE__);
                    return DLP_CLSC_UNEXPECTED_VECTOR_DIM;
                }

                DLP_TYPE tmp_stor_type =
                    dlp_gemm_get_stor_type((metadata->bias + b_i)->stor_type);
                // Extract SF storage type
                DLP_TYPE tmp_sf_stor_type =
                    (metadata->bias + b_i)->sf
                        ? dlp_gemm_get_stor_type(
                              (metadata->bias + b_i)->sf->scale_factor_type)
                        : DLP_INVALID;

                DLP_TYPE tmp_zp_stor_type =
                    (metadata->bias + b_i)->zp
                        ? dlp_gemm_get_stor_type(
                              (metadata->bias + b_i)->zp->zero_point_type)
                        : DLP_INVALID;

                if (((metadata->bias + b_i)->sf
                     && (metadata->bias + b_i)->sf->scale_factor_len > 0)
                    && ((metadata->bias + b_i)->sf->scale_factor == NULL)) {
                    dlp_print_msg(
                        " BIAS scale_factor is NULL but length > 0. Exiting..",
                        __FILE__, __LINE__);
                    return DLP_CLSC_NULL_POINTER;
                }

                /* BIAS supports only per-tensor (len == 1) or per-channel
                 * (len == n) scale factors; reject anything else early so
                 * the kernel never sees a mid-sized vector and reads OOB. */
                if ((metadata->bias + b_i)->sf
                    && ((metadata->bias + b_i)->sf->scale_factor_len != 1)
                    && ((metadata->bias + b_i)->sf->scale_factor_len != n)) {
                    dlp_print_msg(
                        " BIAS scale_factor_len is != 1 and != n. Exiting..",
                        __FILE__, __LINE__);
                    return DLP_CLSC_UNEXPECTED_VECTOR_DIM;
                }

                if (((metadata->bias + b_i)->zp
                     && (metadata->bias + b_i)->zp->zero_point_len > 0)
                    && ((metadata->bias + b_i)->zp->zero_point == NULL)) {
                    dlp_print_msg(
                        " BIAS zero_point is NULL but length > 0. Exiting..",
                        __FILE__, __LINE__);
                    return DLP_CLSC_NULL_POINTER;
                }

                /* Same length contract for the BIAS dequant zero point. */
                if ((metadata->bias + b_i)->zp
                    && ((metadata->bias + b_i)->zp->zero_point_len != 1)
                    && ((metadata->bias + b_i)->zp->zero_point_len != n)) {
                    dlp_print_msg(
                        " BIAS zero_point_len is != 1 and != n. Exiting..",
                        __FILE__, __LINE__);
                    return DLP_CLSC_UNEXPECTED_VECTOR_DIM;
                }

                dlp_gemm_set_node_params(
                    (post_op_list + i), POST_OPS_BIAS,
                    (metadata->bias + b_i)->bias, meta_arg,
                    ((metadata->bias + b_i)->bias_len > 0)
                        ? &((metadata->bias + b_i)->bias_len)
                        : NULL,
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
                    tmp_stor_type, tmp_zp_stor_type, tmp_sf_stor_type,
                    /* BIAS supports per-tensor or per-channel SF only;
                     * len fully determines the dim. */
                    (metadata->bias + b_i)->sf
                        ? (((metadata->bias + b_i)->sf->scale_factor_len == 1)
                               ? DLP_PARAM_DIM_PER_TENSOR
                               : DLP_PARAM_DIM_PER_CHANNEL)
                        : DLP_PARAM_DIM_INVALID);

                b_i += 1;
            } break;
            case SCALE: {
                if (metadata->scale == NULL) {
                    dlp_print_msg(" Post_op.scale is NULL. Exiting..", __FILE__,
                                  __LINE__);
                    return DLP_CLSC_NULL_POINTER;
                }
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
                {
                    /* Strict (dim, len) coherence for SCALE SF.
                     *
                     *   PER_TENSOR  <=> len == 1
                     *   PER_TOKEN   <=> len == m   (per output row)
                     *   PER_CHANNEL <=> len == n   (per output column)
                     *
                     * Any other dim value (incl. INVALID) is rejected so
                     * the decision engine can map dim -> sfDim verbatim
                     * without having to second-guess the caller. */
                    const dlp_sf_t* sf = (metadata->scale + s_i)->sf;
                    if (sf) {
                        char* err = NULL;
                        switch (sf->scale_factor_dim) {
                            case DLP_PARAM_DIM_PER_TENSOR:
                                if (sf->scale_factor_len != 1)
                                    err = " Post_op.scale PER_TENSOR requires"
                                          " scale_factor_len == 1. Exiting..";
                                break;
                            case DLP_PARAM_DIM_PER_TOKEN:
                                if (sf->scale_factor_len != m)
                                    err = " Post_op.scale PER_TOKEN requires"
                                          " scale_factor_len == m. Exiting..";
                                break;
                            case DLP_PARAM_DIM_PER_CHANNEL:
                                if (sf->scale_factor_len != n)
                                    err = " Post_op.scale PER_CHANNEL requires"
                                          " scale_factor_len == n. Exiting..";
                                break;
                            default:
                                err = " Post_op.scale scale_factor_dim must be"
                                      " PER_TENSOR / PER_TOKEN / PER_CHANNEL."
                                      " Set sf->scale_factor_dim explicitly."
                                      " Exiting..";
                                break;
                        }
                        if (err) {
                            dlp_print_msg(err, __FILE__, __LINE__);
                            return DLP_CLSC_UNEXPECTED_VECTOR_DIM;
                        }
                    }
                }
                if ((metadata->scale + s_i)->zp
                    && ((metadata->scale + s_i)->zp->zero_point_len != 1)
                    && ((metadata->scale + s_i)->zp->zero_point_len < n)) {
                    dlp_print_msg(" Post_op.scale zero point length is < n."
                                  " Exiting..",
                                  __FILE__, __LINE__);
                    return DLP_CLSC_UNEXPECTED_VECTOR_DIM;
                }

                DLP_TYPE tmp_zp_stor_type =
                    (metadata->scale + s_i)->zp
                        ? dlp_gemm_get_stor_type(
                              (metadata->scale + s_i)->zp->zero_point_type)
                        : DLP_INVALID;
                DLP_TYPE tmp_sf_stor_type =
                    (metadata->scale + s_i)->sf
                        ? dlp_gemm_get_stor_type(
                              (metadata->scale + s_i)->sf->scale_factor_type)
                        : DLP_INVALID;

                md_t* zero_point_len_ptr =
                    (metadata->scale + s_i)->zp
                        ? &((metadata->scale + s_i)->zp->zero_point_len)
                        : &zero_zp_len;

                /* SCALE is the only post-op that supports per-token scale
                 * factors (in addition to per-tensor / per-channel), so we
                 * propagate sf->scale_factor_dim verbatim from the caller
                 * rather than inferring it from len. PerToken vs PerChannel
                 * cannot be distinguished from len alone (both are vectors)
                 * and is the user's choice. */
                dlp_gemm_set_node_params(
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
                    NULL, 0, DLP_INVALID, tmp_zp_stor_type, tmp_sf_stor_type,
                    (metadata->scale + s_i)->sf
                        ? (metadata->scale + s_i)->sf->scale_factor_dim
                        : DLP_PARAM_DIM_INVALID);

                s_i += 1;
            } break;
            case MATRIX_ADD: {
                if (metadata->matrix_add == NULL) {
                    dlp_print_msg(" Post_op.matrix_add is NULL. Exiting..",
                                  __FILE__, __LINE__);
                    return DLP_CLSC_NULL_POINTER;
                }
                if (((metadata->matrix_add + m_i)->matrix == NULL)
                    || ((metadata->matrix_add + m_i)->ldm <= 0)) {
                    dlp_print_msg(
                        " Post_op.matrix_add attributes are invalid. Exiting..",
                        __FILE__, __LINE__);
                    return DLP_CLSC_NULL_POINTER;
                }
                DLP_TYPE tmp_stor_type = dlp_gemm_get_stor_type(
                    (metadata->matrix_add + m_i)->stor_type);

                /* Get scale factor storage type (default f32 identity when the
                 * matrix operand carries no scale factor). */
                DLP_TYPE sf_stor_type =
                    (metadata->matrix_add + m_i)->sf
                        ? dlp_gemm_get_stor_type((metadata->matrix_add + m_i)
                                                     ->sf->scale_factor_type)
                        : DLP_F32;

                if (((metadata->matrix_add + m_i)->sf
                     && (metadata->matrix_add + m_i)->sf->scale_factor_len > 0)
                    && ((metadata->matrix_add + m_i)->sf->scale_factor
                        == NULL)) {
                    dlp_print_msg(" MATRIX_ADD scale_factor is NULL but length "
                                  "> 0. Exiting..",
                                  __FILE__, __LINE__);
                    return DLP_CLSC_NULL_POINTER;
                }

                /* MATRIX_ADD supports only per-tensor (len == 1) or
                 * per-channel (len == n) scale factors. */
                if ((metadata->matrix_add + m_i)->sf
                    && ((metadata->matrix_add + m_i)->sf->scale_factor_len != 1)
                    && ((metadata->matrix_add + m_i)->sf->scale_factor_len
                        != n)) {
                    dlp_print_msg(" MATRIX_ADD scale_factor_len is != 1 and"
                                  " != n. Exiting..",
                                  __FILE__, __LINE__);
                    return DLP_CLSC_UNEXPECTED_VECTOR_DIM;
                }

                dlp_gemm_set_node_params(
                    (post_op_list + i), POST_OPS_MATRIX_ADD,
                    (metadata->matrix_add + m_i)->matrix, meta_arg,
                    &((metadata->matrix_add + m_i)->ldm),
                    (metadata->matrix_add + m_i)->sf
                        ? (metadata->matrix_add + m_i)->sf->scale_factor
                        : (void*)&dlp_matrix_op_identity_scale,
                    (metadata->matrix_add + m_i)->sf
                        ? (metadata->matrix_add + m_i)->sf->scale_factor_len
                        : 1,
                    NULL, 0, tmp_stor_type, DLP_INVALID, sf_stor_type,
                    /* MATRIX_ADD supports per-tensor or per-channel SF only;
                     * len fully determines the dim. A missing SF is an implicit
                     * per-tensor identity (1.0). */
                    (metadata->matrix_add + m_i)->sf
                        ? (((metadata->matrix_add + m_i)->sf->scale_factor_len
                            == 1)
                               ? DLP_PARAM_DIM_PER_TENSOR
                               : DLP_PARAM_DIM_PER_CHANNEL)
                        : DLP_PARAM_DIM_PER_TENSOR);

                m_i += 1;
            } break;
            case MATRIX_MUL: {
                if (metadata->matrix_mul == NULL) {
                    dlp_print_msg(" Post_op.matrix_mul is NULL. Exiting..",
                                  __FILE__, __LINE__);
                    return DLP_CLSC_NULL_POINTER;
                }
                if (((metadata->matrix_mul + mul_i)->matrix == NULL)
                    || ((metadata->matrix_mul + mul_i)->ldm <= 0)) {
                    dlp_print_msg(
                        " Post_op.matrix_mul attributes are invalid. Exiting..",
                        __FILE__, __LINE__);
                    return DLP_CLSC_NULL_POINTER;
                }
                DLP_TYPE tmp_stor_type = dlp_gemm_get_stor_type(
                    (metadata->matrix_mul + mul_i)->stor_type);

                /* Get scale factor storage type (default f32 identity when the
                 * matrix operand carries no scale factor). */
                DLP_TYPE sf_stor_type =
                    (metadata->matrix_mul + mul_i)->sf
                        ? dlp_gemm_get_stor_type((metadata->matrix_mul + mul_i)
                                                     ->sf->scale_factor_type)
                        : DLP_F32;

                if (((metadata->matrix_mul + mul_i)->sf
                     && (metadata->matrix_mul + mul_i)->sf->scale_factor_len
                            > 0)
                    && ((metadata->matrix_mul + mul_i)->sf->scale_factor
                        == NULL)) {
                    dlp_print_msg(" MATRIX_MUL scale_factor is NULL but length "
                                  "> 0. Exiting..",
                                  __FILE__, __LINE__);
                    return DLP_CLSC_NULL_POINTER;
                }

                /* MATRIX_MUL supports only per-tensor (len == 1) or
                 * per-channel (len == n) scale factors. */
                if ((metadata->matrix_mul + mul_i)->sf
                    && ((metadata->matrix_mul + mul_i)->sf->scale_factor_len
                        != 1)
                    && ((metadata->matrix_mul + mul_i)->sf->scale_factor_len
                        != n)) {
                    dlp_print_msg(" MATRIX_MUL scale_factor_len is != 1 and"
                                  " != n. Exiting..",
                                  __FILE__, __LINE__);
                    return DLP_CLSC_UNEXPECTED_VECTOR_DIM;
                }

                dlp_gemm_set_node_params(
                    (post_op_list + i), POST_OPS_MATRIX_MUL,
                    (metadata->matrix_mul + mul_i)->matrix, meta_arg,
                    &((metadata->matrix_mul + mul_i)->ldm),
                    (metadata->matrix_mul + mul_i)->sf
                        ? (metadata->matrix_mul + mul_i)->sf->scale_factor
                        : (void*)&dlp_matrix_op_identity_scale,
                    (metadata->matrix_mul + mul_i)->sf
                        ? (metadata->matrix_mul + mul_i)->sf->scale_factor_len
                        : 1,
                    NULL, 0, tmp_stor_type, DLP_INVALID, sf_stor_type,
                    /* MATRIX_MUL supports per-tensor or per-channel SF only;
                     * len fully determines the dim. A missing SF is an implicit
                     * per-tensor identity (1.0). */
                    (metadata->matrix_mul + mul_i)->sf
                        ? (((metadata->matrix_mul + mul_i)->sf->scale_factor_len
                            == 1)
                               ? DLP_PARAM_DIM_PER_TENSOR
                               : DLP_PARAM_DIM_PER_CHANNEL)
                        : DLP_PARAM_DIM_PER_TENSOR);

                mul_i += 1;
            } break;
            default: {
                dlp_print_msg(
                    " Unsupported post-op type in seq_vector. Exiting..",
                    __FILE__, __LINE__);
                return DLP_CLSC_NOT_SUPPORTED;
                break;
            }
        }

        // Simulating linked list using an array.
        if (i < (metadata->seq_length - 1)) {
            (post_op_list + i)->next = (post_op_list + i + 1);
        }
    }

    return DLP_CLSC_SUCCESS;
}

/* GLU is shape-changing (2I -> I), so at most one is allowed per chain
 * at the very end of the post-op list.*/
dlp_clsc_err_t
dlp_gemm_translate_glu_term_op(dlp_metadata_t*   metadata,
                               dlp_gemm_post_op* post_op_list,
                               dlp_gemm_post_op* post_op_list_start,
                               void*             meta_arg,
                               md_t              m,
                               md_t              n)
{
    if (post_op_list_start == NULL || post_op_list == NULL) {
        return DLP_CLSC_NULL_POINTER;
    }

    /* Head-node list-level facts, initialized on every return path. */
    post_op_list_start->glu_d    = NULL;
    post_op_list_start->glu_ld_d = 0;

    if (metadata == NULL || metadata->glu == NULL) {
        dlp_gemm_set_node_params(
            post_op_list, POST_OPS_DISABLE, NULL, NULL, NULL, NULL, 0, NULL, 0,
            DLP_INVALID, DLP_INVALID, DLP_INVALID, DLP_PARAM_DIM_INVALID);
        // glu is optional, so if it's not present, just return success.
        return DLP_CLSC_SUCCESS;
    }

    /* The fused GLU writes its compacted (m x I) result
        into the caller's D buffer, so D is mandatory. */
    if (metadata->glu->d == NULL) {
        dlp_print_msg(" GLU requires a non-NULL output buffer D. "
                      "Exiting..",
                      __FILE__, __LINE__);
        return DLP_CLSC_NULL_POINTER;
    }

    /* n = 2I (interleaved gate/up); odd n has no pairing. */
    if ((n % 2) != 0) {
        dlp_print_msg(" GLU requires an even output width n (= 2I). "
                      "Exiting..",
                      __FILE__, __LINE__);
        return DLP_CLSC_INVALID_MATRIX_DIMENSION;
    }

    /* ld_d minimum: I (= n/2) for row-major D, m for column-major
        D. Order comes from the caller-resolved meta_arg. */
    const bool glu_col_major =
        (meta_arg != NULL)
        && ((*(const char*)meta_arg == 'c') || (*(const char*)meta_arg == 'C'));
    const md_t glu_n    = n / 2;
    const md_t min_ld_d = glu_col_major ? m : glu_n;
    if ((md_t)metadata->glu->ld_d < min_ld_d) {
        dlp_print_msg(" GLU ld_d is smaller than the D leading dimension "
                      "(I for row-major, m for column-major). Exiting..",
                      __FILE__, __LINE__);
        return DLP_CLSC_INVALID_LEADING_DIMENSION;
    }

    DLP_GEMM_POST_OP_CODE tmp_code = POST_OPS_DISABLE;
    DLP_TYPE tmp_stor_type = dlp_gemm_get_stor_type(metadata->glu->stor_type);

    switch (metadata->glu->algo_type) {
        case GATED_SWIGLU:
            tmp_code = POST_OPS_GATED_SWIGLU;
            break;
        case GATED_SWIGLU_AND_MUL:
            tmp_code = POST_OPS_GATED_SWIGLU_AND_MUL;
            break;
        default:
            dlp_print_msg(" Unknown GLU algo_type. Exiting..", __FILE__,
                          __LINE__);
            return DLP_CLSC_NOT_SUPPORTED;
    }

    dlp_gemm_set_node_params(post_op_list, tmp_code, metadata->glu->alpha,
                             meta_arg, metadata->glu->beta, NULL, 0, NULL, 0,
                             tmp_stor_type, DLP_INVALID, DLP_INVALID,
                             DLP_PARAM_DIM_INVALID);

    // Chain the last post-op with the terminal glu op. Only valid if the
    // caller has already filled post_op_list[0..seq_length-1].
    if (post_op_list_start != post_op_list) {
        (post_op_list - 1)->next = post_op_list;
    }

    /* Both current GLU variants are shape-changing (the default case above
     * rejects anything else). Record the list-level property on the head node
     * for O(1) lookup by consumers, along with the caller's D output buffer +
     * leading dim so the frame can point the half-width store at D. */
    post_op_list_start->glu_d    = metadata->glu->d;
    post_op_list_start->glu_ld_d = (uint64_t)metadata->glu->ld_d;
    post_op_list->glu_d          = metadata->glu->d;
    post_op_list->glu_ld_d       = (uint64_t)metadata->glu->ld_d;

    return DLP_CLSC_SUCCESS;
}
