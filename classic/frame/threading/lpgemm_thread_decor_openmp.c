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

#include <stdio.h>
#include <stdlib.h>

#include "config/lpgemm_config.h"
#include "gemm_utils/lpgemm_utils.h"
#include "lpgemm_5loop_interface_apis.h"
#include "lpgemm_eltwise_ops_interface_apis.h"
#include "lpgemm_types.h"
#include "sys_utils/lpgemm_sys.h"
#include "threading/lpgemm_thread_decor_openmp.h"
#include "threading/lpgemm_thread_utils.h"

#ifdef DLP_ENABLE_OPENMP

#include <omp.h>

#define DLP_NUM_STATIC_COMMS 96

DLP_INLINE void
calculate_n_threads_per_gemm(md_t        group_size,
                             md_t*       n_threads,
                             md_t*       n_gemms_in_parallel,
                             md_t*       n_threads_per_gemm,
                             dlp_rntm_t* rntm_g)
{
    // Guard against group_size <= 0 to prevent divide-by-zero
    if (group_size <= 0) {
        *n_threads           = 1;
        *n_gemms_in_parallel = 1;
        *n_threads_per_gemm  = 1;
        return;
    }

    *n_threads           = rntm_g->num_threads;
    *n_gemms_in_parallel = -1;
    if (*n_threads == 1) {
        (*n_gemms_in_parallel) = 1;
    } else if (*n_gemms_in_parallel < 1) {
        (*n_gemms_in_parallel) = dlp_min((*n_threads), group_size);
    }
    /* ToDo: All the leftover thrads might go under-utilized. Could be optimized
     * further. */
    (*n_threads_per_gemm) = (*n_threads) / (*n_gemms_in_parallel);
}

DLP_INLINE md_t
next_factor(const md_t nt, const md_t part_nt)
{
    if (part_nt == nt) {
        return part_nt;
    }

    md_t nt_temp = part_nt + 1;
    while ((nt_temp <= nt) && ((nt % nt_temp) != 0)) {
        nt_temp++;
    }
    return nt_temp;
}

DLP_INLINE md_t
prev_factor(const md_t nt, const md_t part_nt)
{
    if (part_nt == 1) {
        return part_nt;
    }

    md_t nt_temp = part_nt - 1;
    while ((nt_temp >= 1) && ((nt % nt_temp) != 0)) {
        nt_temp--;
    }
    return nt_temp;
}

DLP_INLINE void
lpgemm_pnl_wrk_heur_adjust_ic_jc_ways(md_t  MR,
                                      md_t  NR,
                                      md_t  m,
                                      md_t  n,
                                      md_t* n_threads,
                                      md_t* ic_ways,
                                      md_t* jc_ways)
{
    // This function currently only increments ic and subsequently decrements
    // jc. Cannot proceed if all threads are allocated to ic.
    // The factorization adjustment here is based on improving the B NR panel
    // distribution among the jc threads.
    md_t mu = (m + MR - 1) / MR;
    md_t nu = (n + NR - 1) / NR;

    // The next 3 ic factors will be considered to see if it results in better
    // NR panel distribution and subsequently reduce the per thread panel work.
    md_t nu_mod_jc_ways = nu % (*jc_ways);
    if ((nu_mod_jc_ways != 0) && ((*ic_ways) < (*n_threads))) {
        md_t mu_ic_cur      = (mu + (*ic_ways) - 1) / (*ic_ways);
        md_t nu_jc_cur      = (nu + (*jc_ways) - 1) / (*jc_ways);
        md_t panel_work_cur = mu_ic_cur + nu_jc_cur;

        const md_t next_ic         = next_factor((*n_threads), (*ic_ways));
        const md_t prev_jc         = prev_factor((*n_threads), (*jc_ways));
        md_t       mu_ic_next      = (mu + next_ic - 1) / next_ic;
        md_t       nu_jc_prev      = (nu + prev_jc - 1) / prev_jc;
        md_t       panel_work_next = mu_ic_next + nu_jc_prev;

        if (panel_work_next < panel_work_cur) {
            panel_work_cur = panel_work_next;
            (*ic_ways)     = next_ic;
            (*jc_ways)     = prev_jc;
        }

        nu_mod_jc_ways = nu % (*jc_ways);
        if ((nu_mod_jc_ways != 0) && (next_ic < (*n_threads))) {
            const md_t next_next_ic    = next_factor((*n_threads), next_ic);
            const md_t prev_prev_jc    = prev_factor((*n_threads), prev_jc);
            md_t       mu_ic_next_next = (mu + next_next_ic - 1) / next_next_ic;
            md_t       nu_jc_prev_prev = (nu + prev_prev_jc - 1) / prev_prev_jc;
            md_t       panel_work_next_next = mu_ic_next_next + nu_jc_prev_prev;

            if (panel_work_next_next < panel_work_cur) {
                panel_work_cur = panel_work_next_next;
                (*ic_ways)     = next_next_ic;
                (*jc_ways)     = prev_prev_jc;
            }

            nu_mod_jc_ways = nu % (*jc_ways);
            if ((nu_mod_jc_ways != 0) && (next_next_ic < (*n_threads))) {
                const md_t next_next_next_ic =
                    next_factor((*n_threads), next_next_ic);
                const md_t prev_prev_prev_jc =
                    prev_factor((*n_threads), prev_prev_jc);
                md_t mu_ic_next_next_next =
                    (mu + next_next_next_ic - 1) / next_next_next_ic;
                md_t nu_jc_prev_prev_prev =
                    (nu + prev_prev_prev_jc - 1) / prev_prev_prev_jc;
                md_t panel_work_next_next_next =
                    mu_ic_next_next_next + nu_jc_prev_prev_prev;

                if (panel_work_next_next_next < panel_work_cur) {
                    (*ic_ways) = next_next_next_ic;
                    (*jc_ways) = prev_prev_prev_jc;
                }
            }
        }
    }
}

DLP_INLINE void
lpgemm_adjust_ic_jc_ways(const md_t m,
                         const md_t n,
                         const md_t k,
                         const md_t MC,
                         const md_t NC,
                         const md_t KC,
                         const md_t MR,
                         const md_t NR,
                         md_t*      n_threads,
                         md_t*      ic_ways,
                         md_t*      jc_ways,
                         md_t       m_boost)
{
    const md_t    m_ic                = m / (*ic_ways);
    const md_t    n_jc                = n / (*jc_ways);
    const int64_t cur_work_per_thread = m_ic + n_jc;

    const md_t next_ic = next_factor((*n_threads), (*ic_ways));
    const md_t prev_ic = prev_factor((*n_threads), (*ic_ways));
    const md_t next_jc = next_factor((*n_threads), (*jc_ways));
    const md_t prev_jc = prev_factor((*n_threads), (*jc_ways));

    const md_t m_next_ic = m / next_ic;
    const md_t m_prev_ic = m / prev_ic;
    const md_t n_next_jc = n / next_jc;
    const md_t n_prev_jc = n / prev_jc;

    const int64_t next_jc_work_per_thread = n_next_jc + m_prev_ic;
    const int64_t next_ic_work_per_thread = m_next_ic + n_prev_jc;

    const md_t MCx2                = MC * 2;
    const md_t k_factor            = k / KC;
    const md_t n_jc_modulo_NR      = n_jc % NR;
    const md_t n_prev_jc_modulo_NR = n_prev_jc % NR;

    bool can_increase_ic = FALSE;
    bool can_increase_jc = FALSE;

    if (((*ic_ways) > 1) && ((*jc_ways) < (*n_threads))) {
        if (next_jc_work_per_thread < cur_work_per_thread) {
            can_increase_jc = TRUE;
        }
        // Check whether m_prev_ic remains in good l2 load zone.
        else if ((((m_ic <= MC) && (m_prev_ic <= MC)) || (m_ic > MC))
                 && ((n_jc > NR) && (n_next_jc == NR))) {
            can_increase_jc = TRUE;
        }
    }
    if (((*ic_ways) < (*n_threads)) && ((*jc_ways) > 1)) {
        if (next_ic_work_per_thread <= cur_work_per_thread) {
            can_increase_ic = TRUE;
        }
        // ic adjustment towards next highest factor if it results in
        // m_next_ic <= MC. This helps in reducing number of A matrix
        // loads per thread to l2 from main memory.
        else if ((m_ic > MC) && (m_next_ic <= MC) && (m_next_ic >= MR)
                 && (k_factor > 4)) {
            can_increase_ic = TRUE;
        }
        // ic adjustment towards next highest factor resulted in better
        // performance when m is sufficiently larger than n.
        else if ((m > (m_boost * n)) && (m_ic >= MCx2) && (k_factor > 4)) {
            can_increase_ic = TRUE;
        }
        // Performance improvement also observed when n_jc is a multiple
        // of NR.
        else if ((n_jc_modulo_NR != 0) && (n_prev_jc_modulo_NR == 0)
                 && (k_factor > 4)) {
            can_increase_ic = TRUE;
        }
    }

    if (can_increase_ic) {
        (*ic_ways) = next_ic;
        (*jc_ways) = prev_jc;
    } else if (can_increase_jc) {
        // Giving priority to ic and m dimensions, if m >= n, jc must be < ic.
        if (((m >= n) && (prev_ic >= next_jc))
            || ((m < n) && (prev_ic <= next_jc))) {
            (*ic_ways) = prev_ic;
            (*jc_ways) = next_jc;
        }
    }
}

DLP_INLINE void
lpgemm_s32o32_get_threading(md_t*               n_threads,
                            md_t*               ic_ways,
                            md_t*               jc_ways,
                            md_t                m,
                            md_t                n,
                            md_t                k,
                            dlp_rntm_t*         rntm_g,
                            AOCL_OPERATION_TYPE op_type)
{
    *n_threads = rntm_g->num_threads;
    *jc_ways   = rntm_g->jc_ways;
    *ic_ways   = rntm_g->ic_ways;

    if (((*ic_ways) > 0) || ((*jc_ways) > 0)) {
        // If DLP_IC_NT or JC_NT are set.
        // Default cases.
        *ic_ways = ((*ic_ways) > 0) ? (*ic_ways) : 1;
        *jc_ways = ((*jc_ways) > 0) ? (*jc_ways) : 1;

        *n_threads = (*jc_ways) * (*ic_ways);
    } else if ((*n_threads) > 1) {

        md_t       NR         = lpgemm_get_block_size_NR_global_cntx(op_type);
        md_t       MR         = lpgemm_get_block_size_MR_global_cntx(op_type);
        md_t       mr_blks    = (m + MR - 1) / MR;
        md_t       nr_blks    = (n + NR - 1) / NR;
        md_t       mrxnr_blks = mr_blks * nr_blks;
        md_t       mr_blks_adj_n_threads = ((*n_threads) / mr_blks) * mr_blks;
        md_t       delta_mr_blks_adj     = (*n_threads) - mr_blks_adj_n_threads;
        const md_t low_freq_thres        = 6;

        if (n <= NR) {
            (*ic_ways)   = (*n_threads);
            (*jc_ways)   = 1;
            (*n_threads) = (*ic_ways) * (*jc_ways);
        } else if (m <= MR) {
            (*jc_ways)   = (*n_threads);
            (*ic_ways)   = 1;
            (*n_threads) = (*ic_ways) * (*jc_ways);
        } else if (((n % NR) == 0) && (mrxnr_blks <= (*n_threads))
                   && (delta_mr_blks_adj < low_freq_thres)) {
            (*ic_ways)   = mr_blks;
            (*jc_ways)   = (*n_threads) / (*ic_ways);
            (*n_threads) = (*ic_ways) * (*jc_ways);
        } else {
            // If DLP_NUM_THREADS are set, generate jc,ic from the same.
            dlp_thread_partition_2x2((*n_threads), m, n, ic_ways, jc_ways);
            if ((mr_blks >= (*ic_ways)) && (nr_blks >= (*jc_ways))) {
                lpgemm_pnl_wrk_heur_adjust_ic_jc_ways(MR, NR, m, n, n_threads,
                                                      ic_ways, jc_ways);
            }
        }
    } else {
        // Setting all the values to 1 in case n_threads <= 1. This ensures
        // the threading parameters are valid.
        *n_threads = 1;
        *jc_ways   = 1;
        *ic_ways   = 1;
    }
}

DLP_INLINE void
batch_lpgemm_s32o32_get_threading(md_t                group_size,
                                  md_t*               n_threads,
                                  md_t*               n_gemms_in_parallel,
                                  md_t*               n_threads_per_gemm,
                                  md_t*               ic_ways,
                                  md_t*               jc_ways,
                                  md_t                m,
                                  md_t                n,
                                  md_t                k,
                                  dlp_rntm_t*         rntm_g,
                                  AOCL_OPERATION_TYPE op_type)
{

    calculate_n_threads_per_gemm(group_size, n_threads, n_gemms_in_parallel,
                                 n_threads_per_gemm, rntm_g);

    if ((*n_threads_per_gemm) > 1) {

        md_t NR      = lpgemm_get_block_size_NR_global_cntx(op_type);
        md_t MR      = lpgemm_get_block_size_MR_global_cntx(op_type);
        md_t mr_blks = (m + MR - 1) / MR;
        md_t nr_blks = (n + NR - 1) / NR;

        if (n <= NR) {
            (*ic_ways)            = (*n_threads_per_gemm);
            (*jc_ways)            = 1;
            (*n_threads_per_gemm) = (*ic_ways) * (*jc_ways);
        } else if (m <= MR) {
            (*jc_ways)            = (*n_threads_per_gemm);
            (*ic_ways)            = 1;
            (*n_threads_per_gemm) = (*ic_ways) * (*jc_ways);
        } else {
            // If DLP_NUM_THREADS are set, generate jc,ic from the same.
            dlp_thread_partition_2x2((*n_threads_per_gemm), m, n, ic_ways,
                                     jc_ways);
            if ((mr_blks >= (*ic_ways)) && (nr_blks >= (*jc_ways))) {
                lpgemm_pnl_wrk_heur_adjust_ic_jc_ways(
                    MR, NR, m, n, n_threads_per_gemm, ic_ways, jc_ways);
            }
        }
        (*n_threads) = (*n_gemms_in_parallel) * (*ic_ways) * (*jc_ways);
    } else if ((*n_gemms_in_parallel) > 1) {
        // if n_threads_per_gemm <= 1 and n_gemms_in_parallel > 1
        // set ic, jc ways to 1 and n_threads to n_gemms_in_parallel
        *n_threads          = *n_gemms_in_parallel;
        *n_threads_per_gemm = 1;
        *jc_ways            = 1;
        *ic_ways            = 1;
    } else {
        // Setting all the values to 1 in case n_threads <= 1. This ensures
        // the threading parameters are valid.
        *n_threads           = 1;
        *n_gemms_in_parallel = 1;
        *n_threads_per_gemm  = 1;
        *jc_ways             = 1;
        *ic_ways             = 1;
    }
}

DLP_INLINE void
batch_lpgemm_u8s8s32o32_get_threading(md_t        group_size,
                                      md_t*       n_threads,
                                      md_t*       n_gemms_in_parallel,
                                      md_t*       n_threads_per_gemm,
                                      md_t*       ic_ways,
                                      md_t*       jc_ways,
                                      md_t        m,
                                      md_t        n,
                                      md_t        k,
                                      dlp_rntm_t* rntm_g)
{
    batch_lpgemm_s32o32_get_threading(
        group_size, n_threads, n_gemms_in_parallel, n_threads_per_gemm, ic_ways,
        jc_ways, m, n, k, rntm_g, U8S8S32OS32);
}

DLP_INLINE void
batch_lpgemm_s8s8s32o32_get_threading(md_t        group_size,
                                      md_t*       n_threads,
                                      md_t*       n_gemms_in_parallel,
                                      md_t*       n_threads_per_gemm,
                                      md_t*       ic_ways,
                                      md_t*       jc_ways,
                                      md_t        m,
                                      md_t        n,
                                      md_t        k,
                                      dlp_rntm_t* rntm_g)
{
    batch_lpgemm_s32o32_get_threading(
        group_size, n_threads, n_gemms_in_parallel, n_threads_per_gemm, ic_ways,
        jc_ways, m, n, k, rntm_g, S8S8S32OS32);
}

DLP_INLINE void
lpgemm_u8s8s32o32_get_threading(md_t*       n_threads,
                                md_t*       ic_ways,
                                md_t*       jc_ways,
                                md_t        m,
                                md_t        n,
                                md_t        k,
                                dlp_rntm_t* rntm_g)
{
    lpgemm_s32o32_get_threading(n_threads, ic_ways, jc_ways, m, n, k, rntm_g,
                                U8S8S32OS32);
}

DLP_INLINE void
lpgemm_s8s8s32o32_get_threading(md_t*       n_threads,
                                md_t*       ic_ways,
                                md_t*       jc_ways,
                                md_t        m,
                                md_t        n,
                                md_t        k,
                                dlp_rntm_t* rntm_g)
{
    lpgemm_s32o32_get_threading(n_threads, ic_ways, jc_ways, m, n, k, rntm_g,
                                S8S8S32OS32);
}

DLP_INLINE void
lpgemm_bf16bf16f32of32_get_threading(md_t*       n_threads,
                                     md_t*       ic_ways,
                                     md_t*       jc_ways,
                                     md_t        m,
                                     md_t        n,
                                     md_t        k,
                                     dlp_rntm_t* rntm_g)
{
    *n_threads = rntm_g->num_threads;
    *jc_ways   = rntm_g->jc_ways;
    *ic_ways   = rntm_g->ic_ways;

    if (((*ic_ways) > 0) || ((*jc_ways) > 0)) {
        // If DLP_IC_NT or JC_NT are set.
        // Default cases.
        *ic_ways = ((*ic_ways) > 0) ? (*ic_ways) : 1;
        *jc_ways = ((*jc_ways) > 0) ? (*jc_ways) : 1;

        *n_threads = (*jc_ways) * (*ic_ways);
    } else if ((*n_threads) > 1) {
        md_t       NR = lpgemm_get_block_size_NR_global_cntx(BF16BF16F32OF32);
        md_t       MR = lpgemm_get_block_size_MR_global_cntx(BF16BF16F32OF32);
        md_t       mr_blks               = (m + MR - 1) / MR;
        md_t       nr_blks               = (n + NR - 1) / NR;
        md_t       mrxnr_blks            = mr_blks * nr_blks;
        md_t       mr_blks_adj_n_threads = ((*n_threads) / mr_blks) * mr_blks;
        md_t       delta_mr_blks_adj     = (*n_threads) - mr_blks_adj_n_threads;
        const md_t low_freq_thres        = 6;

        if (n <= NR) {
            (*ic_ways)   = (*n_threads);
            (*jc_ways)   = 1;
            (*n_threads) = (*ic_ways) * (*jc_ways);
        } else if (m <= MR) {
            (*jc_ways)   = (*n_threads);
            (*ic_ways)   = 1;
            (*n_threads) = (*ic_ways) * (*jc_ways);
        } else if (((n % NR) == 0) && (mrxnr_blks <= (*n_threads))
                   && (delta_mr_blks_adj < low_freq_thres)) {
            (*ic_ways)   = mr_blks;
            (*jc_ways)   = (*n_threads) / (*ic_ways);
            (*n_threads) = (*ic_ways) * (*jc_ways);
        } else {
            // If DLP_NUM_THREADS are set, generate jc,ic from the same.
            dlp_thread_partition_2x2((*n_threads), m, n, ic_ways, jc_ways);
            if ((mr_blks >= (*ic_ways)) && (nr_blks >= (*jc_ways))) {
                lpgemm_pnl_wrk_heur_adjust_ic_jc_ways(MR, NR, m, n, n_threads,
                                                      ic_ways, jc_ways);
            }
        }
    } else {
        // Setting all the values to 1 in case n_threads <= 1. This ensures
        // the threading parameters are valid.
        *n_threads = 1;
        *jc_ways   = 1;
        *ic_ways   = 1;
    }
}

DLP_INLINE void
batch_lpgemm_bf16bf16f32of32_get_threading(md_t        group_size,
                                           md_t*       n_threads,
                                           md_t*       n_gemms_in_parallel,
                                           md_t*       n_threads_per_gemm,
                                           md_t*       ic_ways,
                                           md_t*       jc_ways,
                                           md_t        m,
                                           md_t        n,
                                           md_t        k,
                                           dlp_rntm_t* rntm_g)
{

    calculate_n_threads_per_gemm(group_size, n_threads, n_gemms_in_parallel,
                                 n_threads_per_gemm, rntm_g);

    /* The user is not allowed to set ic_ways or jc_ways */
    if ((*n_threads_per_gemm) > 1) {
        md_t NR      = lpgemm_get_block_size_NR_global_cntx(BF16BF16F32OF32);
        md_t MR      = lpgemm_get_block_size_MR_global_cntx(BF16BF16F32OF32);
        md_t mr_blks = (m + MR - 1) / MR;
        md_t nr_blks = (n + NR - 1) / NR;

        if (n <= NR) {
            (*ic_ways)            = (*n_threads_per_gemm);
            (*jc_ways)            = 1;
            (*n_threads_per_gemm) = (*ic_ways) * (*jc_ways);
        } else if (m <= MR) {
            (*jc_ways)            = (*n_threads_per_gemm);
            (*ic_ways)            = 1;
            (*n_threads_per_gemm) = (*ic_ways) * (*jc_ways);
        } else {
            // If DLP_NUM_THREADS are set, generate jc,ic from the same.
            dlp_thread_partition_2x2((*n_threads_per_gemm), m, n, ic_ways,
                                     jc_ways);
            if ((mr_blks >= (*ic_ways)) && (nr_blks >= (*jc_ways))) {
                lpgemm_pnl_wrk_heur_adjust_ic_jc_ways(
                    MR, NR, m, n, n_threads_per_gemm, ic_ways, jc_ways);
            }
        }
        (*n_threads) = (*n_gemms_in_parallel) * (*ic_ways) * (*jc_ways);
    } else if ((*n_gemms_in_parallel) > 1) {
        // if n_threads_per_gemm <= 1 and n_gemms_in_parallel > 1
        // set ic, jc ways to 1 and n_threads to n_gemms_in_parallel
        *n_threads          = *n_gemms_in_parallel;
        *n_threads_per_gemm = 1;
        *jc_ways            = 1;
        *ic_ways            = 1;
    } else {
        // Setting all the values to 1 in case n_threads <= 1. This ensures
        // the threading parameters are valid.
        *n_threads           = 1;
        *n_gemms_in_parallel = 1;
        *n_threads_per_gemm  = 1;
        *jc_ways             = 1;
        *ic_ways             = 1;
    }
}

// Some aspects of sgemm smart threading incorporated here. Eventually this
// will be redirected to the sgemm smart threading API.
DLP_INLINE void
lpgemm_f32f32f32of32_get_threading(md_t*       n_threads,
                                   md_t*       ic_ways,
                                   md_t*       jc_ways,
                                   md_t        m,
                                   md_t        n,
                                   md_t        k,
                                   dlp_rntm_t* rntm_g)
{
    // Query the context for SUP limits.
    const md_t MT = lpgemm_get_sup_thres_MT_global_cntx(F32F32F32OF32);
    const md_t NT = lpgemm_get_sup_thres_NT_global_cntx(F32F32F32OF32);
    const md_t KT = lpgemm_get_sup_thres_KT_global_cntx(F32F32F32OF32);

    // Query the context for various blocksizes.
    md_t NR = lpgemm_get_block_size_NR_global_cntx(F32F32F32OF32);
    md_t MR = lpgemm_get_block_size_MR_global_cntx(F32F32F32OF32);
    md_t MC = lpgemm_get_block_size_MC_global_cntx(F32F32F32OF32);
    md_t NC = lpgemm_get_block_size_NC_global_cntx(F32F32F32OF32);
    md_t KC = lpgemm_get_block_size_KC_global_cntx(F32F32F32OF32);

    const md_t MT_2 = MT / 2;

    *n_threads = rntm_g->num_threads;
    *jc_ways   = rntm_g->jc_ways;
    *ic_ways   = rntm_g->ic_ways;

    if (((*ic_ways) > 0) || ((*jc_ways) > 0)) {
        // If DLP_IC_NT or JC_NT are set.
        // Default cases.
        *ic_ways = ((*ic_ways) > 0) ? (*ic_ways) : 1;
        *jc_ways = ((*jc_ways) > 0) ? (*jc_ways) : 1;

        *n_threads = (*jc_ways) * (*ic_ways);
    } else if ((*n_threads) > 1) {
        md_t       mr_blks               = (m + MR - 1) / MR;
        md_t       nr_blks               = (n + NR - 1) / NR;
        md_t       mrxnr_blks            = mr_blks * nr_blks;
        md_t       mr_blks_adj_n_threads = ((*n_threads) / mr_blks) * mr_blks;
        md_t       delta_mr_blks_adj     = (*n_threads) - mr_blks_adj_n_threads;
        const md_t low_freq_thres        = 6;

        if (n <= NR) {
            (*ic_ways)   = (*n_threads);
            (*jc_ways)   = 1;
            (*n_threads) = (*ic_ways) * (*jc_ways);
        } else if (m <= MR) {
            (*jc_ways)   = (*n_threads);
            (*ic_ways)   = 1;
            (*n_threads) = (*ic_ways) * (*jc_ways);
        } else if (((n % NR) == 0) && (mrxnr_blks <= (*n_threads))
                   && (delta_mr_blks_adj < low_freq_thres)) {
            (*ic_ways)   = mr_blks;
            (*jc_ways)   = (*n_threads) / (*ic_ways);
            (*n_threads) = (*ic_ways) * (*jc_ways);
        } else {
            // If DLP_NUM_THREADS are set, generate jc,ic from the same.
            dlp_thread_partition_2x2((*n_threads), m, n, ic_ways, jc_ways);
            if ((mr_blks >= (*ic_ways)) && (nr_blks >= (*jc_ways))) {
                lpgemm_adjust_ic_jc_ways(m, n, k, MC, NC, KC, MR, NR, n_threads,
                                         ic_ways, jc_ways, 5);
            }
        }
    } else {
        // Setting all the values to 1 in case n_threads <= 1. This ensures
        // the threading parameters are valid.
        *n_threads = 1;
        *jc_ways   = 1;
        *ic_ways   = 1;
    }

    // Native -> SUP path.
    const md_t m_ic                = m / (*ic_ways);
    const md_t n_jc                = n / (*jc_ways);
    const md_t page_size           = dlp_get_page_size();
    const md_t page_size_b_floatx2 = 2 * (page_size / sizeof(float));

    if ((m >= MT) && (n >= NT) && (k >= KT)) {
        if (((k >= page_size_b_floatx2) && (m_ic > MT_2) && (n_jc >= NT))
            || ((dlp_cpuid_is_avx512_supported() == FALSE)
                && (k > page_size_b_floatx2))) {
            rntm_g->pack_b = TRUE;
            rntm_g->pack_a = TRUE;
        }
    }
}

DLP_INLINE void
batch_lpgemm_f32f32f32of32_get_threading(md_t        group_size,
                                         md_t*       n_threads,
                                         md_t*       n_gemms_in_parallel,
                                         md_t*       n_threads_per_gemm,
                                         md_t*       ic_ways,
                                         md_t*       jc_ways,
                                         md_t        m,
                                         md_t        n,
                                         md_t        k,
                                         dlp_rntm_t* rntm_g)
{

    calculate_n_threads_per_gemm(group_size, n_threads, n_gemms_in_parallel,
                                 n_threads_per_gemm, rntm_g);

    // Query the context for SUP limits.
    const md_t MT = lpgemm_get_sup_thres_MT_global_cntx(F32F32F32OF32);
    const md_t NT = lpgemm_get_sup_thres_NT_global_cntx(F32F32F32OF32);
    const md_t KT = lpgemm_get_sup_thres_KT_global_cntx(F32F32F32OF32);

    // Query the context for various blocksizes.
    md_t NR = lpgemm_get_block_size_NR_global_cntx(F32F32F32OF32);
    md_t MR = lpgemm_get_block_size_MR_global_cntx(F32F32F32OF32);
    md_t MC = lpgemm_get_block_size_MC_global_cntx(F32F32F32OF32);
    md_t NC = lpgemm_get_block_size_NC_global_cntx(F32F32F32OF32);
    md_t KC = lpgemm_get_block_size_KC_global_cntx(F32F32F32OF32);

    const md_t MT_2 = MT / 2;

    /* The user is not allowed to set ic_ways or jc_ways */
    if ((*n_threads_per_gemm) > 1) {
        md_t mr_blks = (m + MR - 1) / MR;
        md_t nr_blks = (n + NR - 1) / NR;

        if (n <= NR) {
            (*ic_ways)            = (*n_threads_per_gemm);
            (*jc_ways)            = 1;
            (*n_threads_per_gemm) = (*ic_ways) * (*jc_ways);
        } else if (m <= MR) {
            (*jc_ways)            = (*n_threads_per_gemm);
            (*ic_ways)            = 1;
            (*n_threads_per_gemm) = (*ic_ways) * (*jc_ways);
        } else {
            // If DLP_NUM_THREADS are set, generate jc,ic from the same.
            dlp_thread_partition_2x2((*n_threads_per_gemm), m, n, ic_ways,
                                     jc_ways);
            if ((mr_blks >= (*ic_ways)) && (nr_blks >= (*jc_ways))) {
                lpgemm_adjust_ic_jc_ways(m, n, k, MC, NC, KC, MR, NR,
                                         n_threads_per_gemm, ic_ways, jc_ways,
                                         5);
            }
        }
        (*n_threads) = (*n_gemms_in_parallel) * (*ic_ways) * (*jc_ways);
    } else if ((*n_gemms_in_parallel) > 1) {
        // if n_threads_per_gemm <= 1 and n_gemms_in_parallel > 1
        // set ic, jc ways to 1 and n_threads to n_gemms_in_parallel
        *n_threads          = *n_gemms_in_parallel;
        *n_threads_per_gemm = 1;
        *jc_ways            = 1;
        *ic_ways            = 1;
    } else {
        // Setting all the values to 1 in case n_threads <= 1. This ensures
        // the threading parameters are valid.
        *n_threads           = 1;
        *n_gemms_in_parallel = 1;
        *n_threads_per_gemm  = 1;
        *jc_ways             = 1;
        *ic_ways             = 1;
    }

    // Native -> SUP path.
    const md_t m_ic                = m / (*ic_ways);
    const md_t n_jc                = n / (*jc_ways);
    const md_t page_size           = dlp_get_page_size();
    const md_t page_size_b_floatx2 = 2 * (page_size / sizeof(float));

    if ((m >= MT) && (n >= NT) && (k >= KT)) {
        if (((k >= page_size_b_floatx2) && (m_ic > MT_2) && (n_jc >= NT))
            || ((dlp_cpuid_is_avx512_supported() == FALSE)
                && (k > page_size_b_floatx2))) {
            rntm_g->pack_b = TRUE;
            rntm_g->pack_a = TRUE;
        }
    }
}

DLP_INLINE AOCL_TID_DISTR_TYPE
lpgemm_get_tid_distr_type(md_t           m,
                          md_t           n,
                          md_t           k,
                          md_t           n_threads,
                          md_t           ic_ways,
                          md_t           jc_ways,
                          lpgemm_cntx_t* lcntx)
{
    AOCL_TID_DISTR_TYPE tid_distr = DEFAULT;

    md_t NR = lcntx->blksz.NR;
    md_t MR = lcntx->blksz.MR;

    md_t mr_blks      = (m + MR - 1) / MR;
    md_t nr_blks      = (n + NR - 1) / NR;
    md_t mr_x_nr_blks = mr_blks * nr_blks;

    md_t low_util_n_thread_thres = n_threads / 2;
    md_t mr_x_nr_blks_fringe     = mr_x_nr_blks % n_threads;

    lpgemm_thread_attrs_t* thr_attrs = lpgemm_get_thread_attrs();

    // Ensure nt for init and now are same.
    bool can_spread_tids = (thr_attrs->openmp_enabled == TRUE)
                           && (thr_attrs->tid_distr_nearly_seq == TRUE)
                           && (thr_attrs->tid_core_grp_load_high == TRUE);

    if ((mr_x_nr_blks <= low_util_n_thread_thres)
        && (can_spread_tids == TRUE)) {
        tid_distr = STRIDE2;
    } else if ((mr_x_nr_blks > n_threads) && ((mr_x_nr_blks / n_threads) <= 2)
               && (mr_x_nr_blks_fringe > 4)
               && (mr_x_nr_blks_fringe < low_util_n_thread_thres)
               && (can_spread_tids == TRUE)) {
        tid_distr = STRIDE2;
    }

    return tid_distr;
}

DLP_INLINE void
lpgemm_modify_tid_on_distr_type(md_t*               tid,
                                md_t                n_threads,
                                AOCL_TID_DISTR_TYPE tid_distr)
{
    if (tid_distr == STRIDE2)
    /* || ( tid_distr == STRIDE4 ) || ( tid_distr == STRIDE8 ) */
    {
        const md_t modif_factor    = 2; /* or 4 or 8 depending on distr type. */
        md_t       tid_modif_limit = (n_threads / modif_factor) * modif_factor;

        /* Distribute 0 - (n_threads/2 - 1) on even cores and remaining
         * (n_threads/2) - (n_threads - 1) on odd cores. Fringe tids,
         * ones that are beyond modif_factor multiple closest to n_threads
         * are ignored. */
        if ((*tid) < tid_modif_limit) {
            (*tid) = ((*tid) / modif_factor)
                     + (((*tid) % modif_factor) * (n_threads / modif_factor));
        }
    }

    /* 4 tid together. */
    /* thread.tid = ( ( ( thread.tid - ( ( thread.tid / 8 ) * 8 ) ) / 4 ) *
                   ( ( n_threads / 8 ) * 4 ) ) +
                 ( ( ( thread.tid / 8 ) * 4 ) + ( thread.tid % 4 ) ); */

    /* 2 tid together. */
    /* thread.tid = ( ( ( thread.tid - ( ( thread.tid / 8 ) * 8 ) ) / 2 ) *
                   ( ( n_threads / 8 ) * 2 ) ) +
                 ( ( ( thread.tid / 8 ) * 2 ) + ( thread.tid % 2 ) ); */
}

#define GEN_LPGEMM_OPENMP_DECORATOR(A_type, B_type, C_type, LPGEMM_SFX)        \
    void lpgemm_##LPGEMM_SFX##_openmp_thread_decorator(                        \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        const AOCL_MEMORY_TAG mtag_b, C_type* c, const md_t rs_c,              \
        const md_t cs_c, const C_type alpha, const C_type beta,                \
        dlp_rntm_t* rntm_g, lpgemm_cntx_t* lcntx,                              \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale)                    \
    {                                                                          \
        md_t n_threads;                                                        \
                                                                               \
        /* Factorization of threads along m and n dimension respectively.*/    \
        md_t ic_ways;                                                          \
        md_t jc_ways;                                                          \
                                                                               \
        lpgemm_##LPGEMM_SFX##_get_threading(&n_threads, &ic_ways, &jc_ways, m, \
                                            n, k, rntm_g);                     \
                                                                               \
        AOCL_TID_DISTR_TYPE tid_distr = lpgemm_get_tid_distr_type(             \
            m, n, k, n_threads, ic_ways, jc_ways, lcntx);                      \
                                                                               \
        dlp_task_comm_t  static_lpgemm_comms[DLP_NUM_STATIC_COMMS];            \
        dlp_task_comm_t* cur_lpgemm_comms = static_lpgemm_comms;               \
                                                                               \
        if (jc_ways > DLP_NUM_STATIC_COMMS) {                                  \
            cur_lpgemm_comms = malloc(jc_ways * sizeof(dlp_task_comm_t));      \
        }                                                                      \
        for (md_t i = 0; i < jc_ways; ++i) {                                   \
            dlp_task_comm_init(ic_ways, &cur_lpgemm_comms[i]);                 \
        }                                                                      \
                                                                               \
        _Pragma("omp parallel num_threads(n_threads)")                         \
        {                                                                      \
            /* Create a thread-local copy of the master thread's dlp_rntm_t.   \
             * This is necessary since we want each thread to be able to track \
             * its own small block pool_t as it executes down the function     \
             * stack.*/                                                        \
            dlp_rntm_t rntm_l = *rntm_g;                                       \
                                                                               \
            /* lpgemm_thrinfo_t object will be used to generate dlp_task_id_t  \
             * objects for use in dlp mt framework inside the respective mat   \
             * mul driver functions.*/                                         \
            lpgemm_thrinfo_t thread;                                           \
            thread.n_threads = n_threads;                                      \
            thread.tid       = omp_get_thread_num();                           \
            /* Performance improvement only observed when reordered buffers    \
             * are used. */                                                    \
            if (mtag_b == REORDERED) {                                         \
                lpgemm_modify_tid_on_distr_type(&thread.tid, n_threads,        \
                                                tid_distr);                    \
            }                                                                  \
                                                                               \
            thread.ic_ways = ic_ways;                                          \
            thread.jc_ways = jc_ways;                                          \
            thread.comm    = cur_lpgemm_comms;                                 \
                                                                               \
            lpgemm_rowvar_##LPGEMM_SFX(m, n, k, a, rs_a, cs_a, mtag_a, b,      \
                                       rs_b, cs_b, mtag_b, c, rs_c, cs_c,      \
                                       alpha, beta, &rntm_l, &thread, lcntx,   \
                                       post_op_list, c_downscale);             \
        }                                                                      \
        if (jc_ways > DLP_NUM_STATIC_COMMS) {                                  \
            free(cur_lpgemm_comms);                                            \
        }                                                                      \
    }

GEN_LPGEMM_OPENMP_DECORATOR(uint8_t, int8_t, int32_t, u8s8s32o32)
GEN_LPGEMM_OPENMP_DECORATOR(bfloat16, bfloat16, float, bf16bf16f32of32)
GEN_LPGEMM_OPENMP_DECORATOR(float, float, float, f32f32f32of32)
GEN_LPGEMM_OPENMP_DECORATOR(int8_t, int8_t, int32_t, s8s8s32o32)

#define GEN_BATCH_LPGEMM_OPENMP_DECORATOR(A_type, B_type, C_type, LPGEMM_SFX)  \
    void batch_lpgemm_##LPGEMM_SFX##_openmp_thread_decorator(                  \
        const md_t group_size, const md_t* m, const md_t* n, const md_t* k,    \
        const A_type** a, const md_t* rs_a, const md_t* cs_a,                  \
        const AOCL_MEMORY_TAG* mtag_a, const B_type** b, const md_t* rs_b,     \
        const md_t* cs_b, const AOCL_MEMORY_TAG* mtag_b, C_type** c,           \
        const md_t* rs_c, const md_t* cs_c, const C_type alpha,                \
        const C_type beta, dlp_rntm_t* rntm_g, lpgemm_cntx_t* lcntx,           \
        lpgemm_post_op(*post_op_list), DLP_TYPE c_downscale)                   \
    {                                                                          \
        /* For now, Assuming all the problems in GEMM are of same size.        \
         * To-Do: optimize work distribution for case where a batch contains   \
           GEMM problems of different sizes.                                   \
         */                                                                    \
        md_t n_threads;                                                        \
        /* Factorization of threads along m and n dimension respectively.*/    \
        md_t ic_ways;                                                          \
        md_t jc_ways;                                                          \
        md_t n_gemms_in_parallel;                                              \
        md_t n_threads_per_gemm;                                               \
                                                                               \
        /* Assuming all the problems in GEMM are of same size */               \
        batch_lpgemm_##LPGEMM_SFX##_get_threading(                             \
            group_size, &n_threads, &n_gemms_in_parallel, &n_threads_per_gemm, \
            &ic_ways, &jc_ways, m[0], n[0], k[0], rntm_g);                     \
                                                                               \
        dlp_task_comm_t  static_lpgemm_comms[DLP_NUM_STATIC_COMMS];            \
        dlp_task_comm_t* cur_lpgemm_comms = static_lpgemm_comms;               \
                                                                               \
        if (jc_ways * n_gemms_in_parallel > DLP_NUM_STATIC_COMMS) {            \
            cur_lpgemm_comms = malloc(jc_ways * n_gemms_in_parallel            \
                                      * sizeof(dlp_task_comm_t));              \
        }                                                                      \
        for (md_t i = 0; i < n_gemms_in_parallel * jc_ways; i++) {             \
            dlp_task_comm_init(ic_ways, &cur_lpgemm_comms[i]);                 \
        }                                                                      \
                                                                               \
        _Pragma("omp parallel num_threads(n_threads)")                         \
        {                                                                      \
            /* Create a thread-local copy of the master thread's dlp_rntm_t.   \
             * This is necessary since we want each thread to be able to track \
             * its own small block pool_t as it executes down the function     \
             * stack.*/                                                        \
            dlp_rntm_t rntm_l = *rntm_g;                                       \
                                                                               \
            /* lpgemm_thrinfo_t object will be used to generate dlp_task_id_t  \
             * objects for use in dlp mt framework inside the respective mat   \
             * mul driver functions.*/                                         \
            lpgemm_thrinfo_t thread;                                           \
            thread.n_threads = n_threads_per_gemm;                             \
            thread.tid       = omp_get_thread_num() % n_threads_per_gemm;      \
            thread.ic_ways   = ic_ways;                                        \
            thread.jc_ways   = jc_ways;                                        \
            thread.comm =                                                      \
                cur_lpgemm_comms                                               \
                + (omp_get_thread_num() / n_threads_per_gemm) * jc_ways;       \
                                                                               \
            md_t gemm_start;                                                   \
            md_t gemm_end;                                                     \
                                                                               \
            /* This structure is filled only to calculate workload             \
               distribution of GEMM problems among threads. This struct is not \
               passed to 5-loop.                                               \
            */                                                                 \
            dlp_task_id_t thrinfo;                                             \
            thrinfo.n_way   = n_gemms_in_parallel;                             \
            thrinfo.work_id = omp_get_thread_num() / n_threads_per_gemm;       \
            dlp_thread_task_range(&thrinfo, group_size, 1, FALSE, &gemm_start, \
                                  &gemm_end);                                  \
                                                                               \
            for (md_t i = gemm_start; i < gemm_end; i++) {                     \
                lpgemm_rowvar_##LPGEMM_SFX(                                    \
                    m[i], n[i], k[i], a[i], rs_a[i], cs_a[i], mtag_a[i], b[i], \
                    rs_b[i], cs_b[i], mtag_b[i], c[i], rs_c[i], cs_c[i],       \
                    alpha, beta, &rntm_l, &thread, lcntx, post_op_list,        \
                    c_downscale);                                              \
            }                                                                  \
        }                                                                      \
        if (jc_ways * n_gemms_in_parallel > DLP_NUM_STATIC_COMMS) {            \
            free(cur_lpgemm_comms);                                            \
        }                                                                      \
    }

GEN_BATCH_LPGEMM_OPENMP_DECORATOR(bfloat16, bfloat16, float, bf16bf16f32of32)
GEN_BATCH_LPGEMM_OPENMP_DECORATOR(float, float, float, f32f32f32of32)
GEN_BATCH_LPGEMM_OPENMP_DECORATOR(uint8_t, int8_t, int32_t, u8s8s32o32)
GEN_BATCH_LPGEMM_OPENMP_DECORATOR(int8_t, int8_t, int32_t, s8s8s32o32)

#define GEN_LPGEMM_OPENMP_DECORATOR_MP(A_type, B_type, C_type, LPGEMM_SFX)     \
    void lpgemm_##LPGEMM_SFX##_openmp_thread_decorator(                        \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        AOCL_MEMORY_TAG mtag_b, C_type* c, const md_t rs_c, const md_t cs_c,   \
        const C_type alpha, const C_type beta, dlp_rntm_t* rntm_g,             \
        lpgemm_cntx_t* lcntx, lpgemm_pre_op* pre_op_list,                      \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale)                    \
    {                                                                          \
        md_t n_threads;                                                        \
                                                                               \
        /* Factorization of threads along m and n dimension respectively.*/    \
        md_t ic_ways;                                                          \
        md_t jc_ways;                                                          \
                                                                               \
        lpgemm_bf16bf16f32of32_get_threading(&n_threads, &ic_ways, &jc_ways,   \
                                             m, n, k, rntm_g);                 \
                                                                               \
        AOCL_TID_DISTR_TYPE tid_distr = lpgemm_get_tid_distr_type(             \
            m, n, k, n_threads, ic_ways, jc_ways, lcntx);                      \
                                                                               \
        /* Decide whether to go with pack-based implementation                 \
           or kernel-level implementation */                                   \
        md_t MC = lpgemm_get_block_size_MC_global_cntx(BF16BF16F32OF32);       \
        if ((m / ic_ways) > MC)                                                \
            mtag_b = PACK_KC;                                                  \
        else                                                                   \
            mtag_b = UNPACKED;                                                 \
                                                                               \
        dlp_task_comm_t  static_lpgemm_comms[DLP_NUM_STATIC_COMMS];            \
        dlp_task_comm_t* cur_lpgemm_comms = static_lpgemm_comms;               \
                                                                               \
        if (jc_ways > DLP_NUM_STATIC_COMMS) {                                  \
            cur_lpgemm_comms = malloc(jc_ways * sizeof(dlp_task_comm_t));      \
        }                                                                      \
        for (md_t i = 0; i < jc_ways; ++i) {                                   \
            dlp_task_comm_init(ic_ways, &cur_lpgemm_comms[i]);                 \
        }                                                                      \
                                                                               \
        _Pragma("omp parallel num_threads(n_threads)")                         \
        {                                                                      \
            /* Create a thread-local copy of the master thread's dlp_rntm_t.   \
             * This is necessary since we want each thread to be able to track \
             * its own small block pool_t as it executes down the function     \
             * stack.*/                                                        \
            dlp_rntm_t rntm_l = *rntm_g;                                       \
                                                                               \
            /* lpgemm_thrinfo_t object will be used to generate dlp_task_id_t  \
             * objects for use in dlp mt framework inside the respective mat   \
             * mul driver functions.*/                                         \
            lpgemm_thrinfo_t thread;                                           \
            thread.n_threads = n_threads;                                      \
            thread.tid       = omp_get_thread_num();                           \
            if (mtag_b == REORDERED) {                                         \
                lpgemm_modify_tid_on_distr_type(&thread.tid, n_threads,        \
                                                tid_distr);                    \
            }                                                                  \
                                                                               \
            thread.ic_ways = ic_ways;                                          \
            thread.jc_ways = jc_ways;                                          \
            thread.comm    = cur_lpgemm_comms;                                 \
                                                                               \
            lpgemm_rowvar_##LPGEMM_SFX(                                        \
                m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c,      \
                rs_c, cs_c, alpha, beta, &rntm_l, &thread, lcntx, pre_op_list, \
                post_op_list, c_downscale);                                    \
        }                                                                      \
        if (jc_ways > DLP_NUM_STATIC_COMMS) {                                  \
            free(cur_lpgemm_comms);                                            \
        }                                                                      \
    }

GEN_LPGEMM_OPENMP_DECORATOR_MP(bfloat16, int8_t, float, bf16s4f32of32)

#define GEN_LPGEMM_OPENMP_DECORATOR_GRP(A_type, B_type, C_type, LPGEMM_SFX)    \
    void lpgemm_##LPGEMM_SFX##_openmp_thread_decorator(                        \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        AOCL_MEMORY_TAG mtag_b, float* c, const md_t rs_c, const md_t cs_c,    \
        const C_type alpha, const C_type beta, dlp_rntm_t* rntm_g,             \
        lpgemm_cntx_t* lcntx, lpgemm_group_post_op* grp_post_op_list,          \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale)                    \
    {                                                                          \
        md_t n_threads;                                                        \
                                                                               \
        /* Factorization of threads along m and n dimension respectively.*/    \
        md_t ic_ways;                                                          \
        md_t jc_ways;                                                          \
                                                                               \
        lpgemm_s8s8s32o32_get_threading(&n_threads, &ic_ways, &jc_ways, m, n,  \
                                        k, rntm_g);                            \
                                                                               \
        AOCL_TID_DISTR_TYPE tid_distr = lpgemm_get_tid_distr_type(             \
            m, n, k, n_threads, ic_ways, jc_ways, lcntx);                      \
                                                                               \
        dlp_task_comm_t  static_lpgemm_comms[DLP_NUM_STATIC_COMMS];            \
        dlp_task_comm_t* cur_lpgemm_comms = static_lpgemm_comms;               \
                                                                               \
        if (jc_ways > DLP_NUM_STATIC_COMMS) {                                  \
            cur_lpgemm_comms = malloc(jc_ways * sizeof(dlp_task_comm_t));      \
        }                                                                      \
        for (md_t i = 0; i < jc_ways; ++i) {                                   \
            dlp_task_comm_init(ic_ways, &cur_lpgemm_comms[i]);                 \
        }                                                                      \
                                                                               \
        _Pragma("omp parallel num_threads(n_threads)")                         \
        {                                                                      \
            /* Create a thread-local copy of the master thread's dlp_rntm_t.   \
             * This is necessary since we want each thread to be able to track \
             * its own small block pool_t as it executes down the function     \
             * stack.*/                                                        \
            dlp_rntm_t rntm_l = *rntm_g;                                       \
                                                                               \
            /* lpgemm_thrinfo_t object will be used to generate dlp_task_id_t  \
             * objects for use in dlp mt framework inside the respective mat   \
             * mul driver functions.*/                                         \
            lpgemm_thrinfo_t thread;                                           \
            thread.n_threads = n_threads;                                      \
            thread.tid       = omp_get_thread_num();                           \
            if (mtag_b == REORDERED) {                                         \
                lpgemm_modify_tid_on_distr_type(&thread.tid, n_threads,        \
                                                tid_distr);                    \
            }                                                                  \
                                                                               \
            thread.ic_ways = ic_ways;                                          \
            thread.jc_ways = jc_ways;                                          \
            thread.comm    = cur_lpgemm_comms;                                 \
                                                                               \
            lpgemm_rowvar_##LPGEMM_SFX(                                        \
                m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c,      \
                rs_c, cs_c, alpha, beta, &rntm_l, &thread, lcntx,              \
                grp_post_op_list, post_op_list, c_downscale);                  \
        }                                                                      \
        if (jc_ways > DLP_NUM_STATIC_COMMS) {                                  \
            free(cur_lpgemm_comms);                                            \
        }                                                                      \
    }

GEN_LPGEMM_OPENMP_DECORATOR_GRP(int8_t, int8_t, int32_t, s8s8s32o32_sym_quant)

#define GEN_LPGEMM_OPENMP_DECORATOR_Q(A_type, B_type, C_type, LPGEMM_SFX)      \
    void lpgemm_##LPGEMM_SFX##_openmp_thread_decorator(                        \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        AOCL_MEMORY_TAG mtag_b, C_type* c, const md_t rs_c, const md_t cs_c,   \
        const C_type alpha, const C_type beta, dlp_rntm_t* rntm_g,             \
        lpgemm_cntx_t* lcntx, dlp_quant_op* a_pre_quant,                       \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale)                    \
    {                                                                          \
        md_t n_threads;                                                        \
                                                                               \
        /* Factorization of threads along m and n dimension respectively.*/    \
        md_t ic_ways;                                                          \
        md_t jc_ways;                                                          \
                                                                               \
        lpgemm_s8s8s32o32_get_threading(&n_threads, &ic_ways, &jc_ways, m, n,  \
                                        k, rntm_g);                            \
                                                                               \
        AOCL_TID_DISTR_TYPE tid_distr = lpgemm_get_tid_distr_type(             \
            m, n, k, n_threads, ic_ways, jc_ways, lcntx);                      \
                                                                               \
        dlp_task_comm_t  static_lpgemm_comms[DLP_NUM_STATIC_COMMS];            \
        dlp_task_comm_t* cur_lpgemm_comms = static_lpgemm_comms;               \
                                                                               \
        if (jc_ways > DLP_NUM_STATIC_COMMS) {                                  \
            cur_lpgemm_comms = malloc(jc_ways * sizeof(dlp_task_comm_t));      \
        }                                                                      \
        for (md_t i = 0; i < jc_ways; ++i) {                                   \
            dlp_task_comm_init(ic_ways, &cur_lpgemm_comms[i]);                 \
        }                                                                      \
                                                                               \
        _Pragma("omp parallel num_threads(n_threads)")                         \
        {                                                                      \
            /* Create a thread-local copy of the master thread's dlp_rntm_t.   \
             * This is necessary since we want each thread to be able to track \
             * its own small block pool_t as it executes down the function     \
             * stack.*/                                                        \
            dlp_rntm_t rntm_l = *rntm_g;                                       \
                                                                               \
            /* lpgemm_thrinfo_t object will be used to generate dlp_task_id_t  \
             * objects for use in dlp mt framework inside the respective mat   \
             * mul driver functions.*/                                         \
            lpgemm_thrinfo_t thread;                                           \
            thread.n_threads = n_threads;                                      \
            thread.tid       = omp_get_thread_num();                           \
            if (mtag_b == REORDERED) {                                         \
                lpgemm_modify_tid_on_distr_type(&thread.tid, n_threads,        \
                                                tid_distr);                    \
            }                                                                  \
                                                                               \
            thread.ic_ways = ic_ways;                                          \
            thread.jc_ways = jc_ways;                                          \
            thread.comm    = cur_lpgemm_comms;                                 \
                                                                               \
            lpgemm_rowvar_##LPGEMM_SFX(                                        \
                m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b, cs_b, mtag_b, c,      \
                rs_c, cs_c, alpha, beta, &rntm_l, &thread, lcntx, a_pre_quant, \
                post_op_list, c_downscale);                                    \
        }                                                                      \
        if (jc_ways > DLP_NUM_STATIC_COMMS) {                                  \
            free(cur_lpgemm_comms);                                            \
        }                                                                      \
    }
GEN_LPGEMM_OPENMP_DECORATOR_Q(bfloat16, int8_t, int32_t, bf16s8s32os32)

#define GEN_BATCH_LPGEMM_OPENMP_DECORATOR_MP(A_type, B_type, C_type,           \
                                             LPGEMM_SFX, LPGEMM_PARENT_SFX)    \
    void batch_lpgemm_##LPGEMM_SFX##_openmp_thread_decorator(                  \
        const md_t group_size, const md_t* m, const md_t* n, const md_t* k,    \
        const A_type** a, const md_t* rs_a, const md_t* cs_a,                  \
        const AOCL_MEMORY_TAG* mtag_a, const B_type** b, const md_t* rs_b,     \
        const md_t* cs_b, AOCL_MEMORY_TAG* mtag_b, C_type** c,                 \
        const md_t* rs_c, const md_t* cs_c, const C_type alpha,                \
        const C_type beta, dlp_rntm_t* rntm_g, lpgemm_cntx_t* lcntx,           \
        lpgemm_pre_op(*pre_op_list), lpgemm_post_op(*post_op_list),            \
        DLP_TYPE c_downscale)                                                  \
    {                                                                          \
        /* For now, Assuming all the problems in GEMM are of same size.        \
         * To-Do: optimize work distribution for case where a batch contains   \
           GEMM problems of different sizes.                                   \
         */                                                                    \
        md_t n_threads;                                                        \
        /* Factorization of threads along m and n dimension respectively.*/    \
        md_t ic_ways;                                                          \
        md_t jc_ways;                                                          \
        md_t n_gemms_in_parallel;                                              \
        md_t n_threads_per_gemm;                                               \
                                                                               \
        /* Assuming all the problems in GEMM are of same size */               \
        batch_lpgemm_##LPGEMM_PARENT_SFX##_get_threading(                      \
            group_size, &n_threads, &n_gemms_in_parallel, &n_threads_per_gemm, \
            &ic_ways, &jc_ways, m[0], n[0], k[0], rntm_g);                     \
                                                                               \
        dlp_task_comm_t  static_lpgemm_comms[DLP_NUM_STATIC_COMMS];            \
        dlp_task_comm_t* cur_lpgemm_comms = static_lpgemm_comms;               \
                                                                               \
        if (jc_ways * n_gemms_in_parallel > DLP_NUM_STATIC_COMMS) {            \
            cur_lpgemm_comms = malloc(jc_ways * n_gemms_in_parallel            \
                                      * sizeof(dlp_task_comm_t));              \
        }                                                                      \
        for (md_t i = 0; i < n_gemms_in_parallel * jc_ways; i++) {             \
            dlp_task_comm_init(ic_ways, &cur_lpgemm_comms[i]);                 \
        }                                                                      \
                                                                               \
        md_t MC = lpgemm_get_block_size_MC_global_cntx(BF16BF16F32OF32);       \
                                                                               \
        _Pragma("omp parallel num_threads(n_threads)")                         \
        {                                                                      \
            /* Create a thread-local copy of the master thread's dlp_rntm_t.   \
             * This is necessary since we want each thread to be able to track \
             * its own small block pool_t as it executes down the function     \
             * stack.*/                                                        \
            dlp_rntm_t rntm_l = *rntm_g;                                       \
                                                                               \
            /* lpgemm_thrinfo_t object will be used to generate dlp_task_id_t  \
             * objects for use in dlp mt framework inside the respective mat   \
             * mul driver functions.*/                                         \
            lpgemm_thrinfo_t thread;                                           \
            thread.n_threads = n_threads_per_gemm;                             \
            thread.tid       = omp_get_thread_num() % n_threads_per_gemm;      \
            thread.ic_ways   = ic_ways;                                        \
            thread.jc_ways   = jc_ways;                                        \
            thread.comm =                                                      \
                cur_lpgemm_comms                                               \
                + (omp_get_thread_num() / n_threads_per_gemm) * jc_ways;       \
                                                                               \
            md_t gemm_start;                                                   \
            md_t gemm_end;                                                     \
                                                                               \
            /* This structure is filled only to calculate workload             \
               distribution of GEMM problems among threads. This struct is not \
               passed to 5-loop.                                               \
            */                                                                 \
            dlp_task_id_t thrinfo;                                             \
            thrinfo.n_way   = n_gemms_in_parallel;                             \
            thrinfo.work_id = omp_get_thread_num() / n_threads_per_gemm;       \
            dlp_thread_task_range(&thrinfo, group_size, 1, FALSE, &gemm_start, \
                                  &gemm_end);                                  \
                                                                               \
            for (md_t i = gemm_start; i < gemm_end; i++) {                     \
                /* Decide whether to go with pack-based implementation         \
                or kernel-level implementation */                              \
                if ((m[i] / ic_ways) > MC) {                                   \
                    mtag_b[i] = PACK_KC;                                       \
                } else {                                                       \
                    mtag_b[i] = UNPACKED;                                      \
                }                                                              \
                                                                               \
                lpgemm_rowvar_##LPGEMM_SFX(                                    \
                    m[i], n[i], k[i], a[i], rs_a[i], cs_a[i], mtag_a[i], b[i], \
                    rs_b[i], cs_b[i], mtag_b[i], c[i], rs_c[i], cs_c[i],       \
                    alpha, beta, &rntm_l, &thread, lcntx, pre_op_list,         \
                    post_op_list, c_downscale);                                \
            }                                                                  \
        }                                                                      \
        if (jc_ways * n_gemms_in_parallel > DLP_NUM_STATIC_COMMS) {            \
            free(cur_lpgemm_comms);                                            \
        }                                                                      \
    }

GEN_BATCH_LPGEMM_OPENMP_DECORATOR_MP(
    bfloat16, int8_t, float, bf16s4f32of32, bf16bf16f32of32)

DLP_INLINE void
lpgemm_eltwise_ops_bf16of32_get_threading(md_t*                      n_threads,
                                          md_t*                      ic_ways,
                                          md_t*                      jc_ways,
                                          md_t                       m,
                                          md_t                       n,
                                          dlp_rntm_t*                rntm_g,
                                          lpgemm_eltwise_ops_cntx_t* lcntx)
{
    *n_threads = rntm_g->num_threads;
    *jc_ways   = rntm_g->jc_ways;
    *ic_ways   = rntm_g->ic_ways;

    if (((*ic_ways) > 0) || ((*jc_ways) > 0)) {
        // If DLP_IC_NT or JC_NT are set.
        // Default cases.
        *ic_ways = ((*ic_ways) > 0) ? (*ic_ways) : 1;
        *jc_ways = ((*jc_ways) > 0) ? (*jc_ways) : 1;

        *n_threads = (*jc_ways) * (*ic_ways);
    } else if ((*n_threads) > 1) {
        md_t NR      = lcntx->blksz.NR;
        md_t MR      = lcntx->blksz.MR;
        md_t mr_blks = (m + MR - 1) / MR;
        md_t nr_blks = (n + NR - 1) / NR;

        if (n <= NR) {
            (*ic_ways)   = (mr_blks < (*n_threads)) ? mr_blks : (*n_threads);
            (*jc_ways)   = 1;
            (*n_threads) = (*ic_ways) * (*jc_ways);
        } else if (m <= MR) {
            (*jc_ways)   = (nr_blks < (*n_threads)) ? nr_blks : (*n_threads);
            (*ic_ways)   = 1;
            (*n_threads) = (*ic_ways) * (*jc_ways);
        } else if (mr_blks >= (*n_threads)) {
            (*ic_ways) = (*n_threads);
            (*jc_ways) = 1;
        } else if (mr_blks >= (md_t)((3.0 / 4.0) * (*n_threads))) {
            (*ic_ways)       = mr_blks;
            md_t rem_jc_ways = (md_t)((*n_threads) / (*ic_ways));
            (*jc_ways)       = (rem_jc_ways < nr_blks) ? rem_jc_ways : nr_blks;
            (*n_threads)     = (*ic_ways) * (*jc_ways);
        } else {
            // If DLP_NUM_THREADS are set, generate jc,ic from the same.
            dlp_thread_partition_2x2((*n_threads), m, n, ic_ways, jc_ways);
            if ((mr_blks < (*ic_ways)) && (nr_blks < (*jc_ways))) {
                (*ic_ways)   = mr_blks;
                (*jc_ways)   = nr_blks;
                (*n_threads) = (*ic_ways) * (*jc_ways);
            } else if (mr_blks < (*ic_ways)) {
                (*ic_ways)       = mr_blks;
                md_t rem_jc_ways = (md_t)((*n_threads) / (*ic_ways));
                (*jc_ways)   = (rem_jc_ways < nr_blks) ? rem_jc_ways : nr_blks;
                (*n_threads) = (*ic_ways) * (*jc_ways);
            } else if (nr_blks < (*jc_ways)) {
                (*jc_ways)       = nr_blks;
                md_t rem_ic_ways = (md_t)((*n_threads) / (*jc_ways));
                (*ic_ways)   = (rem_ic_ways < mr_blks) ? rem_ic_ways : mr_blks;
                (*n_threads) = (*ic_ways) * (*jc_ways);
            }
        }
    } else {
        // Setting all the values to 1 in case n_threads <= 1. This ensures
        // the threading parameters are valid.
        (*n_threads) = 1;
        (*jc_ways)   = 1;
        (*ic_ways)   = 1;
    }
}

DLP_INLINE void
lpgemm_eltwise_ops_f32of32_get_threading(md_t*                      n_threads,
                                         md_t*                      ic_ways,
                                         md_t*                      jc_ways,
                                         md_t                       m,
                                         md_t                       n,
                                         dlp_rntm_t*                rntm_g,
                                         lpgemm_eltwise_ops_cntx_t* lcntx)
{
    lpgemm_eltwise_ops_bf16of32_get_threading(n_threads, ic_ways, jc_ways, m, n,
                                              rntm_g, lcntx);
}

#define GEN_UTIL_ELTWISE_OPS_OPENMP_DECORATOR(A_type, B_type, LPGEMM_SFX)      \
    void lpgemm_eltwise_ops_##LPGEMM_SFX##_openmp_thread_decorator(            \
        const md_t m, const md_t n, const A_type* a, const md_t rs_a,          \
        const md_t cs_a, B_type* b, const md_t rs_b, const md_t cs_b,          \
        dlp_rntm_t* rntm_g, lpgemm_eltwise_ops_cntx_t* lcntx,                  \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale)                    \
    {                                                                          \
        md_t n_threads;                                                        \
                                                                               \
        /* Factorization of threads along m and n dimension respectively.*/    \
        md_t ic_ways;                                                          \
        md_t jc_ways;                                                          \
                                                                               \
        lpgemm_eltwise_ops_##LPGEMM_SFX##_get_threading(                       \
            &n_threads, &ic_ways, &jc_ways, m, n, rntm_g, lcntx);              \
                                                                               \
        dlp_task_comm_t  static_lpgemm_comms[DLP_NUM_STATIC_COMMS];            \
        dlp_task_comm_t* cur_lpgemm_comms = static_lpgemm_comms;               \
                                                                               \
        if (jc_ways > DLP_NUM_STATIC_COMMS) {                                  \
            cur_lpgemm_comms = malloc(jc_ways * sizeof(dlp_task_comm_t));      \
        }                                                                      \
        for (md_t i = 0; i < jc_ways; ++i) {                                   \
            dlp_task_comm_init(ic_ways, &cur_lpgemm_comms[i]);                 \
        }                                                                      \
                                                                               \
        _Pragma("omp parallel num_threads(n_threads)")                         \
        {                                                                      \
            /* Create a thread-local copy of the master thread's dlp_rntm_t.   \
             * This is necessary since we want each thread to be able to track \
             * its own small block pool_t as it executes down the function     \
             * stack.*/                                                        \
            dlp_rntm_t rntm_l = *rntm_g;                                       \
                                                                               \
            /* lpgemm_thrinfo_t object will be used to generate dlp_task_id_t  \
             * objects for use in dlp mt framework inside the respective mat   \
             * mul driver functions.*/                                         \
            lpgemm_thrinfo_t thread;                                           \
            thread.n_threads = n_threads;                                      \
            thread.tid       = omp_get_thread_num();                           \
            thread.ic_ways   = ic_ways;                                        \
            thread.jc_ways   = jc_ways;                                        \
            thread.comm      = cur_lpgemm_comms;                               \
                                                                               \
            lpgemm_eltwise_ops_interface_##LPGEMM_SFX(                         \
                m, n, a, rs_a, cs_a, b, rs_b, cs_b, &rntm_l, &thread, lcntx,   \
                post_op_list, c_downscale);                                    \
        }                                                                      \
        if (jc_ways > DLP_NUM_STATIC_COMMS) {                                  \
            free(cur_lpgemm_comms);                                            \
        }                                                                      \
    }

GEN_UTIL_ELTWISE_OPS_OPENMP_DECORATOR(bfloat16, float, bf16of32)
GEN_UTIL_ELTWISE_OPS_OPENMP_DECORATOR(float, float, f32of32)

#else

#define GEN_LPGEMM_DECORATOR(A_type, B_type, C_type, LPGEMM_SFX)               \
    void lpgemm_##LPGEMM_SFX##_thread_decorator(                               \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        const AOCL_MEMORY_TAG mtag_b, C_type* c, const md_t rs_c,              \
        const md_t cs_c, const C_type alpha, const C_type beta,                \
        dlp_rntm_t* rntm_g, lpgemm_cntx_t* lcntx,                              \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale)                    \
    {                                                                          \
        md_t n_threads = 1;                                                    \
                                                                               \
        /* Factorization of threads along m and n dimension respectively.*/    \
        md_t ic_ways = 1;                                                      \
        md_t jc_ways = 1;                                                      \
                                                                               \
        dlp_task_comm_t  static_lpgemm_comm;                                   \
        dlp_task_comm_t* cur_lpgemm_comm = &static_lpgemm_comm;                \
                                                                               \
        dlp_task_comm_init(ic_ways, cur_lpgemm_comm);                          \
                                                                               \
        /* lpgemm_thrinfo_t object will be used to generate dlp_task_id_t      \
         * objects for use in dlp mt framework inside the respective mat mul   \
         * driver functions.*/                                                 \
        lpgemm_thrinfo_t thread;                                               \
        thread.n_threads = n_threads;                                          \
        thread.tid       = 0;                                                  \
        thread.ic_ways   = ic_ways;                                            \
        thread.jc_ways   = jc_ways;                                            \
        thread.comm      = cur_lpgemm_comm;                                    \
                                                                               \
        lpgemm_rowvar_##LPGEMM_SFX(m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b,    \
                                   cs_b, mtag_b, c, rs_c, cs_c, alpha, beta,   \
                                   rntm_g, &thread, lcntx, post_op_list,       \
                                   c_downscale);                               \
    }

GEN_LPGEMM_DECORATOR(uint8_t, int8_t, int32_t, u8s8s32o32)
GEN_LPGEMM_DECORATOR(bfloat16, bfloat16, float, bf16bf16f32of32)
GEN_LPGEMM_DECORATOR(float, float, float, f32f32f32of32)
GEN_LPGEMM_DECORATOR(int8_t, int8_t, int32_t, s8s8s32o32)

#define GEN_LPGEMM_DECORATOR1(A_type, B_type, C_type, LPGEMM_SFX)              \
    void lpgemm_##LPGEMM_SFX##_thread_decorator(                               \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        AOCL_MEMORY_TAG mtag_b, C_type* c, const md_t rs_c, const md_t cs_c,   \
        const C_type alpha, const C_type beta, dlp_rntm_t* rntm_g,             \
        lpgemm_cntx_t* lcntx, lpgemm_pre_op* pre_op_list,                      \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale)                    \
    {                                                                          \
        md_t n_threads = 1;                                                    \
                                                                               \
        /* Factorization of threads along m and n dimension respectively.*/    \
        md_t ic_ways = 1;                                                      \
        md_t jc_ways = 1;                                                      \
                                                                               \
        /* Decide whether to go with pack-based implementation                 \
           or kernel-level implementation */                                   \
        md_t MC = lpgemm_get_block_size_MC_global_cntx(BF16BF16F32OF32);       \
        if ((m / ic_ways) > MC)                                                \
            mtag_b = PACK_KC;                                                  \
        else                                                                   \
            mtag_b = UNPACKED;                                                 \
                                                                               \
        dlp_task_comm_t  static_lpgemm_comm;                                   \
        dlp_task_comm_t* cur_lpgemm_comm = &static_lpgemm_comm;                \
                                                                               \
        dlp_task_comm_init(ic_ways, cur_lpgemm_comm);                          \
                                                                               \
        /* lpgemm_thrinfo_t object will be used to generate dlp_task_id_t      \
         * objects for use in dlp mt framework inside the respective mat mul   \
         * driver functions.*/                                                 \
        lpgemm_thrinfo_t thread;                                               \
        thread.n_threads = n_threads;                                          \
        thread.tid       = 0;                                                  \
        thread.ic_ways   = ic_ways;                                            \
        thread.jc_ways   = jc_ways;                                            \
        thread.comm      = cur_lpgemm_comm;                                    \
                                                                               \
        lpgemm_rowvar_##LPGEMM_SFX(m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b,    \
                                   cs_b, mtag_b, c, rs_c, cs_c, alpha, beta,   \
                                   rntm_g, &thread, lcntx, pre_op_list,        \
                                   post_op_list, c_downscale);                 \
    }

GEN_LPGEMM_DECORATOR1(bfloat16, int8_t, float, bf16s4f32of32)

#define GEN_LPGEMM_DECORATOR2(A_type, B_type, C_type, LPGEMM_SFX)              \
    void lpgemm_##LPGEMM_SFX##_thread_decorator(                               \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        AOCL_MEMORY_TAG mtag_b, float* c, const md_t rs_c, const md_t cs_c,    \
        const C_type alpha, const C_type beta, dlp_rntm_t* rntm_g,             \
        lpgemm_cntx_t* lcntx, lpgemm_group_post_op* grp_post_op_list,          \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale)                    \
    {                                                                          \
        md_t n_threads = 1;                                                    \
                                                                               \
        /* Factorization of threads along m and n dimension respectively.*/    \
        md_t ic_ways = 1;                                                      \
        md_t jc_ways = 1;                                                      \
                                                                               \
        dlp_task_comm_t  static_lpgemm_comm;                                   \
        dlp_task_comm_t* cur_lpgemm_comm = &static_lpgemm_comm;                \
                                                                               \
        dlp_task_comm_init(ic_ways, cur_lpgemm_comm);                          \
                                                                               \
        /* lpgemm_thrinfo_t object will be used to generate dlp_task_id_t      \
         * objects for use in dlp mt framework inside the respective mat mul   \
         * driver functions.*/                                                 \
        lpgemm_thrinfo_t thread;                                               \
        thread.n_threads = n_threads;                                          \
        thread.tid       = 0;                                                  \
        thread.ic_ways   = ic_ways;                                            \
        thread.jc_ways   = jc_ways;                                            \
        thread.comm      = cur_lpgemm_comm;                                    \
                                                                               \
        lpgemm_rowvar_##LPGEMM_SFX(m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b,    \
                                   cs_b, mtag_b, c, rs_c, cs_c, alpha, beta,   \
                                   rntm_g, &thread, lcntx, grp_post_op_list,   \
                                   post_op_list, c_downscale);                 \
    }

GEN_LPGEMM_DECORATOR2(int8_t, int8_t, int32_t, s8s8s32o32_sym_quant)

#define GEN_LPGEMM_DECORATOR_Q(A_type, B_type, C_type, LPGEMM_SFX)             \
    void lpgemm_##LPGEMM_SFX##_thread_decorator(                               \
        const md_t m, const md_t n, const md_t k, const A_type* a,             \
        const md_t rs_a, const md_t cs_a, const AOCL_MEMORY_TAG mtag_a,        \
        const B_type* b, const md_t rs_b, const md_t cs_b,                     \
        AOCL_MEMORY_TAG mtag_b, C_type* c, const md_t rs_c, const md_t cs_c,   \
        const C_type alpha, const C_type beta, dlp_rntm_t* rntm_g,             \
        lpgemm_cntx_t* lcntx, dlp_quant_op* a_pre_quant,                       \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale)                    \
    {                                                                          \
        md_t n_threads = 1;                                                    \
                                                                               \
        /* Factorization of threads along m and n dimension respectively.*/    \
        md_t ic_ways = 1;                                                      \
        md_t jc_ways = 1;                                                      \
                                                                               \
        dlp_task_comm_t  static_lpgemm_comm;                                   \
        dlp_task_comm_t* cur_lpgemm_comm = &static_lpgemm_comm;                \
                                                                               \
        dlp_task_comm_init(ic_ways, cur_lpgemm_comm);                          \
                                                                               \
        /* lpgemm_thrinfo_t object will be used to generate dlp_task_id_t      \
         * objects for use in dlp mt framework inside the respective mat mul   \
         * driver functions.*/                                                 \
        lpgemm_thrinfo_t thread;                                               \
        thread.n_threads = n_threads;                                          \
        thread.tid       = 0;                                                  \
        thread.ic_ways   = ic_ways;                                            \
        thread.jc_ways   = jc_ways;                                            \
        thread.comm      = cur_lpgemm_comm;                                    \
                                                                               \
        lpgemm_rowvar_##LPGEMM_SFX(m, n, k, a, rs_a, cs_a, mtag_a, b, rs_b,    \
                                   cs_b, mtag_b, c, rs_c, cs_c, alpha, beta,   \
                                   rntm_g, &thread, lcntx, a_pre_quant,        \
                                   post_op_list, c_downscale);                 \
    }
GEN_LPGEMM_DECORATOR_Q(bfloat16, int8_t, int32_t, bf16s8s32os32)

#define GEN_BATCH_LPGEMM_OPENMP_DECORATOR(A_type, B_type, C_type, LPGEMM_SFX)  \
    void batch_lpgemm_##LPGEMM_SFX##_thread_decorator(                         \
        const md_t group_size, const md_t* m, const md_t* n, const md_t* k,    \
        const A_type** a, const md_t* rs_a, const md_t* cs_a,                  \
        const AOCL_MEMORY_TAG* mtag_a, const B_type** b, const md_t* rs_b,     \
        const md_t* cs_b, const AOCL_MEMORY_TAG* mtag_b, C_type** c,           \
        const md_t* rs_c, const md_t* cs_c, const C_type alpha,                \
        const C_type beta, dlp_rntm_t* rntm_g, lpgemm_cntx_t* lcntx,           \
        lpgemm_post_op(*post_op_list), DLP_TYPE c_downscale)                   \
    {                                                                          \
        md_t n_threads = 1;                                                    \
                                                                               \
        /* Factorization of threads along m and n dimension respectively.*/    \
        md_t ic_ways = 1;                                                      \
        md_t jc_ways = 1;                                                      \
                                                                               \
        dlp_task_comm_t  static_lpgemm_comm;                                   \
        dlp_task_comm_t* cur_lpgemm_comm = &static_lpgemm_comm;                \
                                                                               \
        dlp_task_comm_init(ic_ways, cur_lpgemm_comm);                          \
        /* lpgemm_thrinfo_t object will be used to generate dlp_task_id_t      \
         * objects for use in dlp mt framework inside the respective mat mul   \
         * driver functions.*/                                                 \
        lpgemm_thrinfo_t thread;                                               \
        thread.n_threads = n_threads;                                          \
        thread.tid       = 0;                                                  \
        thread.ic_ways   = ic_ways;                                            \
        thread.jc_ways   = jc_ways;                                            \
        thread.comm      = cur_lpgemm_comm;                                    \
        md_t gemm_start  = 0;                                                  \
        md_t gemm_end    = group_size;                                         \
                                                                               \
        for (md_t i = gemm_start; i < gemm_end; i++) {                         \
            lpgemm_rowvar_##LPGEMM_SFX(                                        \
                m[i], n[i], k[i], a[i], rs_a[i], cs_a[i], mtag_a[i], b[i],     \
                rs_b[i], cs_b[i], mtag_b[i], c[i], rs_c[i], cs_c[i], alpha,    \
                beta, rntm_g, &thread, lcntx, post_op_list, c_downscale);      \
        }                                                                      \
    }

GEN_BATCH_LPGEMM_OPENMP_DECORATOR(bfloat16, bfloat16, float, bf16bf16f32of32)
GEN_BATCH_LPGEMM_OPENMP_DECORATOR(float, float, float, f32f32f32of32)
GEN_BATCH_LPGEMM_OPENMP_DECORATOR(uint8_t, int8_t, int32_t, u8s8s32o32)
GEN_BATCH_LPGEMM_OPENMP_DECORATOR(int8_t, int8_t, int32_t, s8s8s32o32)

#define GEN_BATCH_LPGEMM_OPENMP_DECORATOR_MP(A_type, B_type, C_type,           \
                                             LPGEMM_SFX)                       \
    void batch_lpgemm_##LPGEMM_SFX##_thread_decorator(                         \
        const md_t group_size, const md_t* m, const md_t* n, const md_t* k,    \
        const A_type** a, const md_t* rs_a, const md_t* cs_a,                  \
        const AOCL_MEMORY_TAG* mtag_a, const B_type** b, const md_t* rs_b,     \
        const md_t* cs_b, const AOCL_MEMORY_TAG* mtag_b, C_type** c,           \
        const md_t* rs_c, const md_t* cs_c, const C_type alpha,                \
        const C_type beta, dlp_rntm_t* rntm_g, lpgemm_cntx_t* lcntx,           \
        lpgemm_pre_op(*pre_op_list), lpgemm_post_op(*post_op_list),            \
        DLP_TYPE c_downscale)                                                  \
    {                                                                          \
        md_t n_threads = 1;                                                    \
                                                                               \
        /* Factorization of threads along m and n dimension respectively.*/    \
        md_t ic_ways = 1;                                                      \
        md_t jc_ways = 1;                                                      \
                                                                               \
        dlp_task_comm_t  static_lpgemm_comm;                                   \
        dlp_task_comm_t* cur_lpgemm_comm = &static_lpgemm_comm;                \
                                                                               \
        dlp_task_comm_init(ic_ways, cur_lpgemm_comm);                          \
        /* lpgemm_thrinfo_t object will be used to generate dlp_task_id_t      \
         * objects for use in dlp mt framework inside the respective mat mul   \
         * driver functions.*/                                                 \
        lpgemm_thrinfo_t thread;                                               \
        thread.n_threads = n_threads;                                          \
        thread.tid       = 0;                                                  \
        thread.ic_ways   = ic_ways;                                            \
        thread.jc_ways   = jc_ways;                                            \
        thread.comm      = cur_lpgemm_comm;                                    \
        md_t gemm_start  = 0;                                                  \
        md_t gemm_end    = group_size;                                         \
                                                                               \
        for (md_t i = gemm_start; i < gemm_end; i++) {                         \
            lpgemm_rowvar_##LPGEMM_SFX(                                        \
                m[i], n[i], k[i], a[i], rs_a[i], cs_a[i], mtag_a[i], b[i],     \
                rs_b[i], cs_b[i], mtag_b[i], c[i], rs_c[i], cs_c[i], alpha,    \
                beta, rntm_g, &thread, lcntx, pre_op_list, post_op_list,       \
                c_downscale);                                                  \
        }                                                                      \
    }

GEN_BATCH_LPGEMM_OPENMP_DECORATOR_MP(bfloat16, int8_t, float, bf16s4f32of32)

#define GEN_UTIL_ELTWISE_OPS_DECORATOR(A_type, B_type, LPGEMM_SFX)             \
    void lpgemm_eltwise_ops_##LPGEMM_SFX##_thread_decorator(                   \
        const md_t m, const md_t n, const A_type* a, const md_t rs_a,          \
        const md_t cs_a, B_type* b, const md_t rs_b, const md_t cs_b,          \
        dlp_rntm_t* rntm_g, lpgemm_eltwise_ops_cntx_t* lcntx,                  \
        lpgemm_post_op* post_op_list, DLP_TYPE c_downscale)                    \
    {                                                                          \
        md_t n_threads = 1;                                                    \
                                                                               \
        /* Factorization of threads along m and n dimension respectively.*/    \
        md_t ic_ways = 1;                                                      \
        md_t jc_ways = 1;                                                      \
                                                                               \
        dlp_task_comm_t  static_lpgemm_comm;                                   \
        dlp_task_comm_t* cur_lpgemm_comm = &static_lpgemm_comm;                \
                                                                               \
        dlp_task_comm_init(ic_ways, cur_lpgemm_comm);                          \
                                                                               \
        /* lpgemm_thrinfo_t object will be used to generate dlp_task_id_t      \
         * objects for use in dlp mt framework inside the respective mat mul   \
         * driver functions.*/                                                 \
        lpgemm_thrinfo_t thread;                                               \
        thread.n_threads = n_threads;                                          \
        thread.tid       = 0;                                                  \
        thread.ic_ways   = ic_ways;                                            \
        thread.jc_ways   = jc_ways;                                            \
        thread.comm      = cur_lpgemm_comm;                                    \
                                                                               \
        lpgemm_eltwise_ops_interface_##LPGEMM_SFX(                             \
            m, n, a, rs_a, cs_a, b, rs_b, cs_b, rntm_g, &thread, lcntx,        \
            post_op_list, c_downscale);                                        \
    }

GEN_UTIL_ELTWISE_OPS_DECORATOR(bfloat16, float, bf16of32)
GEN_UTIL_ELTWISE_OPS_DECORATOR(float, float, f32of32)

#endif
