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
 */

#include <stdlib.h>

#include "aocl_dlp_config.h"
#include "threading/dlp_omp_runtime.h"

bool
dlp_omp_collect_thread_topology(dlp_omp_topology_t* out)
{
    if (out == NULL) {
        return false;
    }

    out->available              = false;
    out->tid_cnt                = 0;
    out->tid_distr_nearly_seq   = false;
    out->tid_core_grp_load_high = false;
    out->tid_core_grp_id_list   = NULL;

    // The OpenMP place probe determines whether this path is usable. On
    // Windows, libomp's affinity policy—not processor-group count—determines
    // whether the returned places are meaningful.
#if DLP_OPENMP_ENABLED && DLP_OPENMP_HAS_PLACES
    int   nt_max                     = 0;
    int   num_procs                  = 0;
    int** thread_core_bind_list      = NULL;
    int*  adj_tid_cnt_for_core_grps  = NULL;
    int*  tid_cnt_for_core_grps      = NULL;
    int*  tid_core_grp_id_list       = NULL;
    int   core_grp_size              = 8;
    int   num_core_grps              = 0;
    bool  can_detect_topo            = true;
    int   core_grp_loaded_thres      = 3;
    int   core_grp_adj_tid_thres_cnt = 0;
    int   core_grp_adj_tid_cnt       = 0;
    int   core_grp_non_adj_tid_cnt   = 0;
    int   cur_core_grp_id            = 0;

    nt_max    = dlp_omp_get_max_threads();
    num_procs = dlp_omp_get_num_procs();
    if (nt_max <= 0 || num_procs <= 0 || nt_max > num_procs) {
        return false;
    }

    thread_core_bind_list =
        calloc((size_t)nt_max, sizeof(*thread_core_bind_list));
    if (thread_core_bind_list == NULL) {
        return false;
    }

#pragma omp parallel num_threads(nt_max)
    {
        int thread_num      = dlp_omp_get_thread_num();
        int thread_place    = omp_get_place_num();
        int place_num_procs = omp_get_place_num_procs(thread_place);

        if (place_num_procs > 0) {
            thread_core_bind_list[thread_num] =
                malloc((size_t)(place_num_procs + 1) * sizeof(int));
            if (thread_core_bind_list[thread_num] != NULL) {
                thread_core_bind_list[thread_num][0] = place_num_procs;
                omp_get_place_proc_ids(thread_place,
                                       &thread_core_bind_list[thread_num][1]);
            }
        }
    }

    tid_core_grp_id_list = malloc((size_t)nt_max * sizeof(int));
    if (tid_core_grp_id_list == NULL) {
        goto cleanup;
    }

    for (int ii = 0; ii < nt_max; ++ii) {
        tid_core_grp_id_list[ii] = -1;
        if (thread_core_bind_list[ii] == NULL) {
            can_detect_topo = false;
            break;
        }

        int st_core_grp_id =
            (thread_core_bind_list[ii][1] % num_procs) / core_grp_size;
        tid_core_grp_id_list[ii] = st_core_grp_id;
        for (int jj = 1; jj < thread_core_bind_list[ii][0]; ++jj) {
            int jj_core_grp_id =
                (thread_core_bind_list[ii][jj + 1] % num_procs) / core_grp_size;
            if (jj_core_grp_id != st_core_grp_id) {
                can_detect_topo = false;
                break;
            }
        }
        if (!can_detect_topo) {
            break;
        }
    }

    if (!can_detect_topo) {
        free(tid_core_grp_id_list);
        goto cleanup;
    }

    num_core_grps             = (num_procs + core_grp_size - 1) / core_grp_size;
    adj_tid_cnt_for_core_grps = calloc((size_t)num_core_grps, sizeof(int));
    tid_cnt_for_core_grps     = calloc((size_t)num_core_grps, sizeof(int));
    if (adj_tid_cnt_for_core_grps == NULL || tid_cnt_for_core_grps == NULL) {
        free(tid_core_grp_id_list);
        goto cleanup;
    }

    cur_core_grp_id = tid_core_grp_id_list[0];
    tid_cnt_for_core_grps[cur_core_grp_id] += 1;

    for (int ii = 1; ii < nt_max; ++ii) {
        if (tid_core_grp_id_list[ii] == cur_core_grp_id) {
            adj_tid_cnt_for_core_grps[cur_core_grp_id] += 1;
        } else {
            cur_core_grp_id = tid_core_grp_id_list[ii];
        }
        tid_cnt_for_core_grps[tid_core_grp_id_list[ii]] += 1;
    }

    for (int ii = 0; ii < num_core_grps; ++ii) {
        if (adj_tid_cnt_for_core_grps[ii] >= core_grp_loaded_thres) {
            core_grp_adj_tid_thres_cnt += 1;
            core_grp_adj_tid_cnt += 1;
        } else if (adj_tid_cnt_for_core_grps[ii] > 0) {
            core_grp_adj_tid_cnt += 1;
        } else if (tid_cnt_for_core_grps[ii] > 0) {
            core_grp_non_adj_tid_cnt += 1;
        }
    }

    out->available = true;
    out->tid_cnt   = nt_max;
    out->tid_distr_nearly_seq =
        core_grp_adj_tid_cnt > (2 * core_grp_non_adj_tid_cnt);
    out->tid_core_grp_load_high =
        core_grp_adj_tid_thres_cnt > 0
        && core_grp_adj_tid_thres_cnt
               >= (core_grp_adj_tid_cnt - core_grp_adj_tid_thres_cnt);
    out->tid_core_grp_id_list = tid_core_grp_id_list;

cleanup:
    free(tid_cnt_for_core_grps);
    free(adj_tid_cnt_for_core_grps);
    for (int ii = 0; ii < nt_max; ++ii) {
        free(thread_core_bind_list[ii]);
    }
    free(thread_core_bind_list);
    return out->available;
#else
    return false;
#endif
}
