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

#ifndef LPGEMM_OPENMP_THREAD_UTILS_H
#define LPGEMM_OPENMP_THREAD_UTILS_H

#include <stdio.h>

#include "classic/dlp_base_types.h"

typedef struct __attribute__((aligned(64)))
{
    int* tid_core_grp_id_list;
    int  tid_cnt;
    bool openmp_enabled;
    bool tid_distr_nearly_seq;
    bool tid_core_grp_load_high;
} lpgemm_thread_attrs_t;

void
lpgemm_init_thread_attrs();

// Should be called only after aocl_lpgemm_init_global_cntx.
lpgemm_thread_attrs_t*
lpgemm_get_thread_attrs();

typedef struct __attribute__((aligned(64)))
{
    void* sent_object;
    md_t  n_threads;
    md_t  barrier_sense;
    md_t  barrier_threads_arrived;
} dlp_task_comm_t;

typedef struct __attribute__((aligned(64)))
{
    md_t             n_threads;
    md_t             tid;
    md_t             ic_ways;
    md_t             jc_ways;
    dlp_task_comm_t* comm;
} lpgemm_thrinfo_t;

DLP_INLINE void
dlp_task_comm_init(md_t n_threads, dlp_task_comm_t* comm)
{
    if (comm == NULL)
        return;
    comm->sent_object             = NULL;
    comm->n_threads               = n_threads;
    comm->barrier_sense           = 0;
    comm->barrier_threads_arrived = 0;
}

typedef struct __attribute__((aligned(64)))
{
    // Our thread id within the task communicator.
    md_t ocomm_id;

    // The number of distinct threads used to parallelize the loop.
    md_t n_way;

    // What we're working on.
    md_t work_id;
} dlp_task_id_t;

// Parallelization only supported along jc and ic loops. Thus not reusing the
// existing thrinfo tree logic, since a light-weight work id generation will
// suffice. However the logic used for thread meta data generation, specific
// to jc and ic loops is borrowed.
DLP_INLINE void
lpgemm_gen_dlp_task_ids(lpgemm_thrinfo_t* thread,
                        dlp_task_id_t*    thread_jc,
                        dlp_task_id_t*    thread_ic)
{
    if (thread == NULL) {
        // Set n_ways=1 to ensure ST behaviour when thread is not initialized.
        // This is the case when DLP_ENABLE_OPENMP is not defined.
        thread_jc->ocomm_id = 0;
        thread_jc->n_way    = 1;
        thread_jc->work_id  = 0;

        thread_ic->ocomm_id = 0;
        thread_ic->n_way    = 1;
        thread_ic->work_id  = 0;
    } else {
        thread_jc->ocomm_id = thread->tid;
        thread_jc->n_way    = thread->jc_ways;
        md_t jc_work_id     = thread->tid / thread->ic_ways;
        thread_jc->work_id  = jc_work_id;

        md_t ic_comm_id     = thread->tid % thread->ic_ways;
        thread_ic->ocomm_id = ic_comm_id;
        thread_ic->n_way    = thread->ic_ways;
        thread_ic->work_id  = ic_comm_id;
    }
}

DLP_INLINE void
dlp_thread_task_range(dlp_task_id_t* thread,
                      md_t           n,
                      md_t           bf,
                      bool           handle_edge_low,
                      md_t*          start,
                      md_t*          end)
{
    md_t n_way = thread->n_way;

    if (n_way == 1) {
        *start = 0;
        *end   = n;
        return;
    }

    md_t work_id = thread->work_id;

    md_t all_start = 0;
    md_t all_end   = n;

    md_t size = all_end - all_start;

    md_t n_bf_whole = size / bf;
    md_t n_bf_left  = size % bf;

    md_t n_bf_lo = n_bf_whole / n_way;
    md_t n_bf_hi = n_bf_whole / n_way;

    // In this function, we partition the space between all_start and
    // all_end into n_way partitions, each a multiple of block_factor
    // with the exception of the one partition that receives the
    // "edge" case (if applicable).
    //
    // Here are examples of various thread partitionings, in units of
    // the block_factor, when n_way = 4. (A '+' indicates the thread
    // that receives the leftover edge case (ie: n_bf_left extra
    // rows/columns in its sub-range).
    //                                        (all_start ... all_end)
    // n_bf_whole  _left  hel  n_th_lo  _hi   thr0  thr1  thr2  thr3
    //         12     =0    f        0    4      3     3     3     3
    //         12     >0    f        0    4      3     3     3     3+
    //         13     >0    f        1    3      4     3     3     3+
    //         14     >0    f        2    2      4     4     3     3+
    //         15     >0    f        3    1      4     4     4     3+
    //         15     =0    f        3    1      4     4     4     3
    //
    //         12     =0    t        4    0      3     3     3     3
    //         12     >0    t        4    0      3+    3     3     3
    //         13     >0    t        3    1      3+    3     3     4
    //         14     >0    t        2    2      3+    3     4     4
    //         15     >0    t        1    3      3+    4     4     4
    //         15     =0    t        1    3      3     4     4     4

    // As indicated by the table above, load is balanced as equally
    // as possible, even in the presence of an edge case.

    // First, we must differentiate between cases where the leftover
    // "edge" case (n_bf_left) should be allocated to a thread partition
    // at the low end of the index range or the high end.

    if (handle_edge_low == FALSE) {
        // Notice that if all threads receive the same number of
        // block_factors, those threads are considered "high" and
        // the "low" thread group is empty.
        md_t n_th_lo = n_bf_whole % n_way;
        // md_t n_th_hi = n_way - n_th_lo;

        // If some partitions must have more block_factors than others
        // assign the slightly larger partitions to lower index threads.
        if (n_th_lo != 0)
            n_bf_lo += 1;

        // Compute the actual widths (in units of rows/columns) of
        // individual threads in the low and high groups.
        md_t size_lo = n_bf_lo * bf;
        md_t size_hi = n_bf_hi * bf;

        // Precompute the starting indices of the low and high groups.
        md_t lo_start = all_start;
        md_t hi_start = all_start + n_th_lo * size_lo;

        // Compute the start and end of individual threads' ranges
        // as a function of their work_ids and also the group to which
        // they belong (low or high).
        if (work_id < n_th_lo) {
            *start = lo_start + (work_id)*size_lo;
            *end   = lo_start + (work_id + 1) * size_lo;
        } else // if ( n_th_lo <= work_id )
        {
            *start = hi_start + (work_id - n_th_lo) * size_hi;
            *end   = hi_start + (work_id - n_th_lo + 1) * size_hi;

            // Since the edge case is being allocated to the high
            // end of the index range, we have to advance the last
            // thread's end.
            if (work_id == n_way - 1)
                *end += n_bf_left;
        }
    } else // if ( handle_edge_low == TRUE )
    {
        // Notice that if all threads receive the same number of
        // block_factors, those threads are considered "low" and
        // the "high" thread group is empty.
        md_t n_th_hi = n_bf_whole % n_way;
        md_t n_th_lo = n_way - n_th_hi;

        // If some partitions must have more block_factors than others
        // assign the slightly larger partitions to higher index threads.
        if (n_th_hi != 0)
            n_bf_hi += 1;

        // Compute the actual widths (in units of rows/columns) of
        // individual threads in the low and high groups.
        md_t size_lo = n_bf_lo * bf;
        md_t size_hi = n_bf_hi * bf;

        // Precompute the starting indices of the low and high groups.
        md_t lo_start = all_start;
        md_t hi_start = all_start + n_th_lo * size_lo + n_bf_left;

        // Compute the start and end of individual threads' ranges
        // as a function of their work_ids and also the group to which
        // they belong (low or high).
        if (work_id < n_th_lo) {
            *start = lo_start + (work_id)*size_lo;
            *end   = lo_start + (work_id + 1) * size_lo;

            // Since the edge case is being allocated to the low
            // end of the index range, we have to advance the
            // starts/ends accordingly.
            if (work_id == 0)
                *end += n_bf_left;
            else {
                *start += n_bf_left;
                *end += n_bf_left;
            }
        } else // if ( n_th_lo <= work_id )
        {
            *start = hi_start + (work_id - n_th_lo) * size_hi;
            *end   = hi_start + (work_id - n_th_lo + 1) * size_hi;
        }
    }
}

void
dlp_thread_partition_2x2(md_t n_thread,
                         md_t work1,
                         md_t work2,
                         md_t* restrict nt1,
                         md_t* restrict nt2);

DLP_INLINE bool
dlp_thread_am_ochief(dlp_task_id_t* t)
{
    return t->ocomm_id == 0;
}

void
dlp_atomic_barrier(md_t t_id, dlp_task_comm_t* comm);

#endif // LPGEMM_OPENMP_THREAD_UTILS_H
