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

#ifndef LPGEMM_TYPES_H
#define LPGEMM_TYPES_H

#include "classic/dlp_base_types.h"
#include "threading/lpgemm_thread_utils.h"

#include "bindings/c_wrappers/capi_cpu_features.h"
#include "bindings/c_wrappers/capi_env_config.h"
#include "bindings/c_wrappers/capi_kernel_frame_wrappers.h"

typedef enum
{
    LPGEMM_INT8  = 0,
    LPGEMM_INT16 = 1,
    LPGEMM_INT32 = 2
} AOCL_ARRAY_TYPE;

// Enum name template:A_mat_type ## B_mat_type ## Accumulate_type ## C_mat_type.
typedef enum
{
    U8S8S16OS16     = 0, // uint8_t - A, int8_t - B, int16_t - C
    U8S8S32OS32     = 1, // uint8_t - A, int8_t - B, int32_t - C
    F32F32F32OF32   = 2, // float - A, float - B, float - C
    BF16BF16F32OF32 = 3, // bf16 - A, bf16 - B, float - C
    S8S8S32OS32     = 4, // int8_t - A, int8_t - B, int32_t - C
    S8S8S16OS16     = 5, // int8_t - A, int8_t - B, int16_t - C
    U8S4S32OS32     = 6, // Only used for reordering int4_t B matrix.
    BF16S4F32OF32   = 7, // Only used for reordering int4_t B matrix.
    F32OBF16 = 8, // Only used for reordering input float matrix to bf16 reorder
    FP16FP16F32OF32 =
        9, // float16 - A, float16 - B, float - C (FP32 accumulator)
    F16F16F16OF16 =
        10, // float16 - A, float16 - B, float16 - C (FP16 accumulator)
    BF16U4F32OF32 = 11 // Only used for reordering uint4_t B matrix.
} AOCL_OPERATION_TYPE;
#define AOCL_OPERATION_TYPE_LEN 12

typedef enum
{
    F32_GELU_TANH = 0,
    F32_GELU_ERF  = 1,
    F32_SOFTMAX   = 2
} AOCL_UTIL_OPERATION_TYPE;
#define AOCL_UTIL_OPERATION_TYPE_LEN 3

typedef enum
{
    BF16OF32 = 0,
    F32OF32  = 1
} AOCL_ELTWISE_OPS_OPERATION_TYPE;
#define AOCL_ELTWISE_OPS_OPERATION_TYPE_LEN 2

typedef enum
{
    ROW_MAJOR    = 0,
    COLUMN_MAJOR = 1,
} AOCL_STOR_TAG;

typedef enum
{
    A_MATRIX     = 0,
    B_MATRIX     = 1,
    AWQ_B_MATRIX = 2,
} AOCL_MATRIX_TYPE;

typedef enum
{
    DEFAULT = 0,
    STRIDE2,
} AOCL_TID_DISTR_TYPE;

typedef struct
{
    void* aligned_buffer;
    void* origin_buffer;
} lpgemm_mem_t;

typedef struct
{
    md_t length;
    md_t width;

    md_t elem_size;

    md_t rs;
    md_t cs;

    AOCL_MEMORY_TAG  mtag;
    AOCL_MATRIX_TYPE mat_type;

    lpgemm_mem_t storage;
} lpgemm_obj_t;

typedef struct
{
    md_t MC;
    md_t NC;
    md_t KC;
    md_t NR;
    md_t MR;
} lpgemm_block_size_t;

typedef struct
{
    md_t packa_rs;
    md_t packa_cs;
    md_t packb_rs;
    md_t packb_cs;
} lpgemm_pack_strides_t;

typedef struct
{
    md_t MT;
    md_t NT;
    md_t KT;
} lpgemm_sup_thres_t;

typedef struct
{
    lpgemm_block_size_t   blksz;
    opaq_fp_t             kern_fun_ptr;
    opaq_fp_t             packa_fun_ptr;
    opaq_fp_t             packb_mxp_fun_ptr;
    opaq_fp_t             packb_fun_ptr;
    opaq_fp_t             unpackb_fun_ptr;
    opaq_fp_t             packsclb_fun_ptr;
    lpgemm_pack_strides_t pack_s;
    lpgemm_sup_thres_t    sup_thres;
    dlp_kernel_hndl_t     dlp_kernel_hndl;
} lpgemm_cntx_t;

typedef struct
{
    lpgemm_block_size_t blksz;
    opaq_fp_t           eltwise_ops_kern_fun_ptr;
} lpgemm_eltwise_ops_cntx_t;

typedef struct
{
    opaq_fp_t kern_fun_ptr;
} lpgemm_util_cntx_t;

#endif // LPGEMM_TYPES_H
