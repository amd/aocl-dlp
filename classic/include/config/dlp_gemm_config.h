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

void
dlp_init_global_cntx();

dlp_gemm_cntx_t*
dlp_gemm_get_global_cntx_obj(AOCL_DLP_OPERATION_TYPE op);

dlp_clsc_err_t
dlp_gemm_upd_cntx_with_metadata(AOCL_DLP_OPERATION_TYPE op,
                                dlp_gemm_cntx_t*        lcntx,
                                dlp_metadata_t*         metadata);

// The size of the thread pool a call will actually run on.
//
// The runtime reports this two different ways and never both at once. Ways
// outrank counts in its precedence, and when ways win it leaves the count at
// -1, so a pinned run answers "no count" rather than answering wrongly. The
// product of the ways is that run's pool. A pin of one way alone is filled to
// 1 x n on the way out of dlp_init_threading; filling again here costs a
// compare and keeps this a function of its own arguments.
static inline md_t
dlp_gemm_effective_thread_count(md_t num_threads, md_t ic_ways, md_t jc_ways)
{
    if ((ic_ways > 0) || (jc_ways > 0)) {
        const md_t ic = (ic_ways > 0) ? ic_ways : 1;
        const md_t jc = (jc_ways > 0) ? jc_ways : 1;

        return ic * jc;
    }

    return num_threads;
}

// A hint that is not a number of anything: there is no m of -1 and no pool of
// -1 threads. That is malformed input rather than another spelling of the unset
// sentinel, and it is refused wherever it is stated -- over a reordered B, a
// packed one, or an untagged one. Nothing reads a hint under those other tags,
// but accepting a malformed one in silence would leave a caller believing it
// had said something.
//
// This is the whole of what a Reorder can check. It has no call in hand and no
// runtime describing one, so the thread count below is not its business.
static inline dlp_clsc_err_t
dlp_gemm_validate_hints(const dlp_gemm_cntx_t* lcntx)
{
    if (lcntx == NULL) {
        return DLP_CLSC_NULL_POINTER;
    }

    if (((lcntx->gemm_kernel_hints).m_hint < 0)
        || ((lcntx->gemm_kernel_hints).nt_hint < 0)) {
        return DLP_CLSC_INVALID_GEMM_HINTS;
    }

    return DLP_CLSC_SUCCESS;
}

// The rest of the contract, which only a GEMM can be held to, because only a
// GEMM knows the pool it will run on.
//
// Over a reordered B: zero is the sentinel for "unset" and leaves both ends on
// the context tile, which agrees. nt_hint must equal the pool this call will
// run on, because the Reorder resolved a partition from it and packed the panel
// to a width derived from that, which a differently sized pool cannot
// reproduce.
//
// The pool, not the partition: a caller pinning DLP_IC_NT / DLP_JC_NT to ways
// that multiply to nt_hint is served. The width was derived from the count
// alone, the ways never entered that derivation, and the pin is honoured when
// the kernel runs, so any factorization of the same count is expressible.
//
// m_hint is deliberately not compared against the call. One Reorder of the
// weights feeds many GEMMs at whatever m each batch brings, so a differing m is
// the case this API exists for; it is charged in shape rather than in errors,
// running a tile resolved for the characteristic m instead of its own.
//
// mtag_b must be the value handed to the decision engine, not the caller's
// original, since a column-major swap or a GEMV reroute can change it first.
static inline dlp_clsc_err_t
dlp_gemm_validate_hints_with_call(const dlp_gemm_cntx_t* lcntx,
                                  AOCL_DLP_MEMORY_TAG    mtag_b,
                                  md_t                   num_threads,
                                  md_t                   ic_ways,
                                  md_t                   jc_ways)
{
    const dlp_clsc_err_t err = dlp_gemm_validate_hints(lcntx);
    if (err != DLP_CLSC_SUCCESS) {
        return err;
    }

    if (mtag_b != REORDERED) {
        return DLP_CLSC_SUCCESS;
    }

    const md_t m_hint  = (lcntx->gemm_kernel_hints).m_hint;
    const md_t nt_hint = (lcntx->gemm_kernel_hints).nt_hint;

    // Not redundant with the comparison below, which is the reason to say so
    // here: reaching that with nt_hint at zero would refuse every call that
    // stated no hints at all.
    if ((m_hint == 0) || (nt_hint == 0)) {
        return DLP_CLSC_SUCCESS;
    }

    if (nt_hint
        != dlp_gemm_effective_thread_count(num_threads, ic_ways, jc_ways)) {
        return DLP_CLSC_HINT_MISMATCH;
    }

    return DLP_CLSC_SUCCESS;
}

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
dlp_gemm_mod_block_size_s16(
    md_t m, md_t n, md_t k, md_t* MC, md_t* NC, md_t* KC);

#endif // DLP_GEMM_CONFIG_H
