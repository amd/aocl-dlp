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

#ifndef LPGEMM_POST_OPS_H
#define LPGEMM_POST_OPS_H

#include "classic/aocl_gemm_post_ops.h"
#include "lpgemm_types.h"

// Used as an internal structure.
typedef struct lpgemm_pre_op_t
{
    uint64_t                op_code;
    uint64_t                group_size;
    void*                   scale_factor;
    uint64_t                scale_factor_len;
    uint64_t                scale_factor_type;
    void*                   zp;
    uint64_t                zp_len;
    struct lpgemm_pre_op_t* next;
} lpgemm_pre_op;

typedef struct lpgemm_grp_post_op_attr_t
{
    void*             a_scale_factor;
    uint64_t          a_scale_factor_len;
    void*             a_zp;
    uint64_t          a_zp_len;
    void*             b_scale_factor;
    uint64_t          b_scale_factor_len;
    void*             b_zp;
    uint64_t          b_zp_len;
    uint64_t          group_size;
    uint64_t          grp_post_op_i;
    uint64_t          grp_post_op_j;
    uint64_t          grp_post_op_k;
    uint64_t          grp_post_op_lda;
    uint64_t          grp_post_op_ldb;
    uint64_t          grp_post_op_sum_ld;
    AOCL_STORAGE_TYPE sf_stor_type;
    AOCL_STORAGE_TYPE zp_stor_type;
} lpgemm_grp_post_op_attr;

// Used as an internal structure
typedef struct lpgemm_group_post_op_t
{
    uint64_t                       group_size;
    void*                          a_scale_factor;
    uint64_t                       a_scale_factor_len;
    void*                          a_zp;
    uint64_t                       a_zp_len;
    void*                          b_scale_factor;
    uint64_t                       b_scale_factor_len;
    void*                          b_zp;
    uint64_t                       b_zp_len;
    AOCL_STORAGE_TYPE              sf_stor_type;
    AOCL_STORAGE_TYPE              zp_stor_type;
    struct lpgemm_group_post_op_t* next;
} lpgemm_group_post_op;

typedef struct lpgemm_pre_op_attr_t
{
    void*    scale_factor;
    uint64_t scale_factor_len;
    uint64_t scale_factor_type;
    void*    zero_point;
    uint64_t zero_point_len;
    uint64_t pre_op_b_i;
    uint64_t pre_op_b_j;
    uint64_t group_size;
    uint64_t pre_op_ld;
} lpgemm_pre_op_attr;

dlp_clsc_err_t
lpgemm_translate_to_post_ops_list(aocl_post_op*   post_op_unparsed,
                                  lpgemm_post_op* post_op_list,
                                  void*           scale_buffer,
                                  void*           meta_arg,
                                  md_t            m,
                                  md_t            n);

dlp_clsc_err_t
lpgemm_translate_to_pre_ops_list(aocl_pre_op*   pre_op_unparsed,
                                 lpgemm_pre_op* pre_op_list,
                                 md_t           m,
                                 md_t           n,
                                 md_t           k);

dlp_clsc_err_t
lpgemm_translate_to_group_postops_list(aocl_group_post_op*   post_op_unparsed,
                                       lpgemm_group_post_op* post_op_list,
                                       md_t                  m,
                                       md_t                  n,
                                       md_t                  k);

#define POST_OP_LABEL_LASTK_SAFE_JUMP                                          \
    if ((post_ops_attr.is_last_k == TRUE) && (post_ops_list_temp != NULL)) {   \
        goto* post_ops_labels[post_ops_list_temp->op_code];                    \
    } else {                                                                   \
        goto* post_ops_labels[0];                                              \
    }

#define POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR                            \
    post_ops_list_temp = post_ops_list_temp->next;                             \
    if (post_ops_list_temp != NULL) {                                          \
        goto* post_ops_labels[post_ops_list_temp->op_code];                    \
    } else {                                                                   \
        goto* post_ops_labels[0];                                              \
    }

#endif // LPGEMM_POST_OPS_H
