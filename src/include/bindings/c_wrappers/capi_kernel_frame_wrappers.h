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

#ifndef CAPI_KERNEL_FRAME_WRAPPERS_H
#define CAPI_KERNEL_FRAME_WRAPPERS_H

#include <stdint.h>

#include "classic/aocl_gemm_post_ops.h"
#include "classic/dlp_base_types.h"
#include "classic/dlp_macros.h"

typedef enum
{
    UNPACKED  = 0,
    PACK      = 1,
    PACK_KC   = 2,
    PACK_NR   = 3,
    REORDERED = 4,
} AOCL_MEMORY_TAG;

// Post-ops codes.
typedef enum
{
    POST_OPS_DISABLE    = 0,
    POST_OPS_BIAS       = 1,
    POST_OPS_RELU       = 2,
    POST_OPS_RELU_SCALE = 3,
    POST_OPS_GELU_TANH  = 4,
    POST_OPS_GELU_ERF   = 5,
    POST_OPS_CLIP       = 6,
    POST_OPS_DOWNSCALE  = 7,
    POST_OPS_MATRIX_ADD = 8,
    POST_OPS_SWISH      = 9,
    POST_OPS_MATRIX_MUL = 10,
    POST_OPS_TANH       = 11,
    POST_OPS_SIGMOID    = 12,
    POST_OPS_SUM        = 13,
    POST_OPS_ADQUANTIZE = 14,
    POST_OPS_MAX
} LPGEMM_POST_OP_CODE;

// Used as an internal structure.
typedef struct lpgemm_post_op_t
{
    uint64_t op_code;
    void*    op_args1; // zero_point, bias, sum_buff
    void*    op_args2; // alpha, storage order, sum_zero_point
    void*    op_args3; // beta, zero_point_len
    void*    scale_factor;
    md_t     scale_factor_len;
    void*    bias_zp;     // Zero point pointer (for BIAS dequantization)
    md_t     bias_zp_len; // Zero point length (for BIAS dequantization)
    uint64_t stor_type;
    uint64_t zp_stor_type;
    uint64_t sf_stor_type; // Introduced for sf store type
    struct lpgemm_post_op_t* next;
} lpgemm_post_op;

// Used as an internal structure.
typedef struct lpgemm_post_op_attr_t
{
    uint64_t post_op_c_i;
    uint64_t post_op_c_j;
    uint64_t rs_c_downscale;
    uint64_t cs_c_downscale;
    void*    buf_downscale;
    uint64_t is_first_k;
    uint64_t is_last_k;
    uint64_t c_stor_type;
    uint64_t b_sum_offset;
    int32_t* b_col_sum_vec;
    int16_t* b_col_sum_vec_s16;
} lpgemm_post_op_attr;

// Type definitions that can be used by both C and C++ code. The enum tokens
// should follow the exact sequence as in kernelDatatype(kernel_frame_base.h).
typedef enum
{
    DLP_KERNEL_INVALID = 0,
    DLP_KERNEL_U8S8S32OS32,
    DLP_KERNEL_U8S8S32OF32,
    DLP_KERNEL_U8S8S32OBF16,
    DLP_KERNEL_U8S8S32OU8,
    DLP_KERNEL_U8S8S32OS8,
    DLP_KERNEL_S8S8S32OU8,
    DLP_KERNEL_S8S8S32OS8,
    DLP_KERNEL_S8S8S32OBF16,
    DLP_KERNEL_S8S8S32OF32,
    DLP_KERNEL_S8S8S32OS32,
    DLP_KERNEL_BF16BF16F32OBF16,
    DLP_KERNEL_BF16BF16F32OF32,
    DLP_KERNEL_F32F32F32OF32,
    DLP_KERNEL_DATATYPE_MAX
} kernel_datatype_t;

typedef struct
{
    void*             kernel_base;
    md_t              mr;
    md_t              nr;
    kernel_datatype_t kDtype;
    bool              invokeRD;
} dlp_kernel_hndl_t;

// C linkage for function declarations only
DLP_BEGIN_EXTERN_C

void
dlp_init_and_get_kernel_hndl(kernel_datatype_t  k_dtype,
                             char               storage_format,
                             AOCL_MEMORY_TAG    mtag_a,
                             AOCL_MEMORY_TAG    mtag_b,
                             md_t               m,
                             md_t               n,
                             md_t               k,
                             md_t               rs_a,
                             md_t               cs_a,
                             md_t               rs_b,
                             md_t               cs_b,
                             md_t               rs_c,
                             md_t               cs_c,
                             void*              alpha,
                             void*              beta,
                             lpgemm_post_op*    metadata,
                             md_t               mr_hint,
                             md_t               nr_hint,
                             md_t               kc_hint,
                             md_t               c_downscale,
                             dlp_kernel_hndl_t* kernel_hndl);

void
dlp_execute_kernel(dlp_kernel_hndl_t   kernel_hndl,
                   md_t                m,
                   md_t                n,
                   md_t                k,
                   void*               A,
                   md_t                rs_a,
                   md_t                cs_a,
                   md_t                ps_a,
                   void*               B,
                   md_t                rs_b,
                   md_t                cs_b,
                   md_t                n_sub_updated,
                   md_t                jc_cur_loop_rem,
                   void*               C,
                   md_t                rs_c,
                   md_t                cs_c,
                   void*               alpha,
                   void*               beta,
                   lpgemm_post_op*     post_ops_list,
                   lpgemm_post_op_attr post_ops_attr);

DLP_END_EXTERN_C

#endif // CAPI_KERNEL_FRAME_WRAPPERS_H
