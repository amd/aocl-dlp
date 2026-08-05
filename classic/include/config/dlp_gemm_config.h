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

#ifndef DLP_GEMM_CONFIG_H
#define DLP_GEMM_CONFIG_H

#include "classic/aocl_gemm_metadata.h"
#include "dlp_gemm_types.h"

#define DLP_GEMM_BF16_MR 6
#define DLP_GEMM_BF16_NR 64
// num_f32_elems_per_zmm = zmm_width / sizeof( float )
#define NUM_F32_ELEMS_PER_ZMM (64 / sizeof(float))

void
dlp_init_global_cntx();

dlp_gemm_cntx_t*
dlp_gemm_get_global_cntx_obj(AOCL_DLP_OPERATION_TYPE op);

dlp_clsc_err_t
dlp_gemm_upd_cntx_with_metadata(AOCL_DLP_OPERATION_TYPE op,
                                dlp_gemm_cntx_t*        lcntx,
                                dlp_metadata_t*         metadata);

static inline dlp_clsc_err_t
dlp_gemm_validate_metadata_with_lcntx(dlp_metadata_t*  metadata,
                                      dlp_gemm_cntx_t* lcntx)
{
    if (metadata == NULL) {
        // Nothing to validate if either metadata or lcntx is NULL.
        return DLP_CLSC_SUCCESS;
    }

    if (lcntx == NULL) {
        // Invalid context pointer, return error.
        return DLP_CLSC_NULL_POINTER;
    }

    bool validateBoolBlockParams = TRUE;
    if (metadata->block_params != NULL) {
        // Block params updated only if possible.
        bool validateBoolMR =
            ((metadata->block_params)->MR > 0)
                ? ((lcntx->blksz).MR == (metadata->block_params)->MR)
                : TRUE;
        bool validateBoolNR =
            ((metadata->block_params)->NR > 0)
                ? ((lcntx->blksz).NR == (metadata->block_params)->NR)
                : TRUE;
        bool validateBoolMC =
            ((metadata->block_params)->MC > 0)
                ? ((lcntx->blksz).MC == (metadata->block_params)->MC)
                : TRUE;
        bool validateBoolKC =
            ((metadata->block_params)->KC > 0)
                ? ((lcntx->blksz).KC == (metadata->block_params)->KC)
                : TRUE;
        bool validateBoolNC =
            ((metadata->block_params)->NC > 0)
                ? ((lcntx->blksz).NC == (metadata->block_params)->NC)
                : TRUE;

        // Also add a check to ensure that the block sizes are multiples of
        // MR and NR on the lcntx. This will be based on the case where only
        // MC,NC were set in the metadata, and MR,NR were not set. In that
        // case, the lcntx will have the default MR,NR values, and the MC,NC
        // values will be set to the metadata values. Again this is for the
        // cases where DE was not involved, like reference reorder.
        bool validateBoolMC_multiple_of_MR =
            ((lcntx->blksz).MC % (lcntx->blksz).MR == 0);
        bool validateBoolNC_multiple_of_NR =
            ((lcntx->blksz).NC % (lcntx->blksz).NR == 0);

        validateBoolBlockParams =
            validateBoolMR && validateBoolNR && validateBoolMC && validateBoolKC
            && validateBoolNC && validateBoolMC_multiple_of_MR
            && validateBoolNC_multiple_of_NR;
    }

    bool validateBoolSupThresholds = TRUE;
    if (metadata->sup_thresholds != NULL) {
        // SUP thresholds updated only if possible.
        bool validateBoolMT =
            ((metadata->sup_thresholds)->MT >= 0)
                ? ((lcntx->sup_thres).MT == (metadata->sup_thresholds)->MT)
                : TRUE;
        bool validateBoolNT =
            ((metadata->sup_thresholds)->NT >= 0)
                ? ((lcntx->sup_thres).NT == (metadata->sup_thresholds)->NT)
                : TRUE;
        bool validateBoolKT =
            ((metadata->sup_thresholds)->KT >= 0)
                ? ((lcntx->sup_thres).KT == (metadata->sup_thresholds)->KT)
                : TRUE;
        validateBoolSupThresholds =
            validateBoolMT && validateBoolNT && validateBoolKT;
    }

    if (validateBoolBlockParams && validateBoolSupThresholds) {
        return DLP_CLSC_SUCCESS;
    } else {
        return DLP_CLSC_INVALID_BLOCK_PARAMS;
    }
}

dlp_gemm_util_cntx_t*
dlp_gemm_util_get_global_cntx_obj(AOCL_DLP_UTIL_OPERATION_TYPE op);

dlp_gemm_eltwise_ops_cntx_t*
dlp_gemm_eltwise_ops_get_global_cntx_obj(
    AOCL_DLP_ELTWISE_OPS_OPERATION_TYPE op);

md_t
dlp_gemm_get_block_size_MC_global_cntx(AOCL_DLP_OPERATION_TYPE op_type);

md_t
dlp_gemm_get_block_size_NC_global_cntx(AOCL_DLP_OPERATION_TYPE op_type);

md_t
dlp_gemm_get_block_size_KC_global_cntx(AOCL_DLP_OPERATION_TYPE op_type);

md_t
dlp_gemm_get_block_size_NR_global_cntx(AOCL_DLP_OPERATION_TYPE op_type);

md_t
dlp_gemm_get_block_size_MR_global_cntx(AOCL_DLP_OPERATION_TYPE op_type);

md_t
dlp_gemm_get_sup_thres_MT_global_cntx(AOCL_DLP_OPERATION_TYPE op_type);

md_t
dlp_gemm_get_sup_thres_NT_global_cntx(AOCL_DLP_OPERATION_TYPE op_type);

md_t
dlp_gemm_get_sup_thres_KT_global_cntx(AOCL_DLP_OPERATION_TYPE op_type);

dlp_arch_t
dlp_gemm_get_enabled_arch();

void
dlp_gemm_get_packa_strides(dlp_gemm_cntx_t* lcntx, md_t* rs, md_t* cs);

void
dlp_gemm_get_packb_strides(dlp_gemm_cntx_t* lcntx, md_t* rs, md_t* cs);

void
dlp_gemm_set_jit_kernel(void* kernel_fp, md_t m_index, md_t n_index);

void*
dlp_gemm_get_jit_kernel(md_t m_index, md_t n_index);

bool
dlp_gemm_get_jit_kernels_generated();

void
dlp_gemm_mod_block_size_s16(
    md_t m, md_t n, md_t k, md_t* MC, md_t* NC, md_t* KC);

#endif // DLP_GEMM_CONFIG_H
