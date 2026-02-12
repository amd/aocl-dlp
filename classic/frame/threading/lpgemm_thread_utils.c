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

#include <math.h>
#include <stdlib.h>

#include "sys_utils/lpgemm_sys.h"
#include "threading/lpgemm_thread_utils.h"

static dlp_pthread_once_t once_check_lpgemm_thread_topo_init =
    DLP_PTHREAD_ONCE_INIT;

static lpgemm_thread_attrs_t lpgemm_thread_attrs;

#ifdef DLP_ENABLE_OPENMP

#include <omp.h>

static void
lpgemm_detect_thread_topo()
{
    int nt_max    = omp_get_max_threads();
    int num_procs = omp_get_num_procs();

    if (nt_max > num_procs) {
        // Over subscription of threads, no more work distr.
        return;
    }

    lpgemm_thread_attrs.tid_cnt        = nt_max;
    lpgemm_thread_attrs.openmp_enabled = TRUE;

    int** thread_core_bind_list     = NULL;
    int*  adj_tid_cnt_for_core_grps = NULL;
    int*  tid_cnt_for_core_grps     = NULL;

    // Allocating memory for pointers, to track thread-core(s) binding
    // by OpenMP. The pointers are also initialized to NULL, in case we
    // actually do not spawn nt_max number of threads in the subsequent
    // parallel region.
    thread_core_bind_list = calloc(nt_max, sizeof(int*));
    if (thread_core_bind_list == NULL) {
        goto err_handle;
    }

// Launch max threads to determine the core bininding for all threads
// within the omp team.
#pragma omp parallel num_threads(nt_max)
    {
        int thread_num      = omp_get_thread_num();
        int thread_place    = omp_get_place_num();
        int place_num_procs = omp_get_place_num_procs(thread_place);

        // 1 extra int for storing num_procs value.
        thread_core_bind_list[thread_num] =
            malloc((place_num_procs + 1) * sizeof(int));
        if (thread_core_bind_list[thread_num] != NULL) {
            thread_core_bind_list[thread_num][0] = place_num_procs;
            omp_get_place_proc_ids(thread_place,
                                   &thread_core_bind_list[thread_num][1]);
        }
    }

    // When SMT is on, this should be 16. Need a way to dynamically retrieve it.
    const int core_grp_size   = 8;
    bool      can_detect_topo = TRUE;

    lpgemm_thread_attrs.tid_core_grp_id_list = malloc(nt_max * sizeof(int));
    if (lpgemm_thread_attrs.tid_core_grp_id_list == NULL) {
        goto err_handle;
    }

    // TIDs are assigned from 0 to nt_max - 1.
    // OpenMP for close distribution need not pin threads to sequential cores
    // in the presence of CCD architecture. Like tid 0-7 will be on core 0-7
    // but tid 8-15 could be on core 96-103. So just checking for increasing
    // core id for corresponding tid wont get accurate core group load.
    // GOMP_CPU_AFFINITY however assigns cores sequentially.
    for (int ii = 0; ii < nt_max; ++ii) {
        lpgemm_thread_attrs.tid_core_grp_id_list[ii] = -1;
        // Identify the core(s) in which thread would be bound
        // In case the thread was never spawned, this code-section is skipped.
        if (thread_core_bind_list[ii] != NULL) {
            // Wrap around the proc/core ids based on number of cores used.
            int st_core_grp_id =
                (thread_core_bind_list[ii][1] % num_procs) / core_grp_size;
            lpgemm_thread_attrs.tid_core_grp_id_list[ii] = st_core_grp_id;
            for (int jj = 1; jj < thread_core_bind_list[ii][0]; ++jj) {
                int cur_core_grp_id =
                    (thread_core_bind_list[ii][jj + 1] % num_procs)
                    / core_grp_size;
                if (cur_core_grp_id != st_core_grp_id) {
                    // Core binding spanning across core groups,
                    // cannot detect topo.
                    can_detect_topo = FALSE;
                    break;
                }
            }
        } else {
            // Thread was not spawned, cannot detect topo.
            // Break out of the current loop.
            can_detect_topo = FALSE;
            break;
        }
        // Check if the topo detection failed at any point.
        // If so, break out of the loop.
        if (can_detect_topo == FALSE) {
            break;
        }
    }

    int num_core_grps = num_procs / core_grp_size;

    // Get count of core groups that are loaded and not loaded with adj ranks.
    // This will give an approximation for thread pin distribution.
    if (can_detect_topo == TRUE) {

        adj_tid_cnt_for_core_grps = calloc(num_core_grps, sizeof(int));
        tid_cnt_for_core_grps     = calloc(num_core_grps, sizeof(int));
        if ((adj_tid_cnt_for_core_grps == NULL)
            || (tid_cnt_for_core_grps == NULL)) {
            goto err_handle;
        }

        const int core_grp_loaded_thres      = 3;
        int       core_grp_adj_tid_thres_cnt = 0;
        int       core_grp_adj_tid_cnt       = 0;
        int       core_grp_non_adj_tid_cnt   = 0;

        int cur_core_grp_id = lpgemm_thread_attrs.tid_core_grp_id_list[0];
        tid_cnt_for_core_grps[cur_core_grp_id] += 1;

        for (int ii = 1; ii < nt_max; ++ii) {
            if (lpgemm_thread_attrs.tid_core_grp_id_list[ii]
                == cur_core_grp_id) {
                adj_tid_cnt_for_core_grps[cur_core_grp_id] += 1;
            } else {
                cur_core_grp_id = lpgemm_thread_attrs.tid_core_grp_id_list[ii];
            }
            tid_cnt_for_core_grps[lpgemm_thread_attrs
                                      .tid_core_grp_id_list[ii]] += 1;
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

        if (core_grp_adj_tid_cnt > (2 * core_grp_non_adj_tid_cnt)) {
            lpgemm_thread_attrs.tid_distr_nearly_seq = TRUE;
        }

        if ((core_grp_adj_tid_thres_cnt > 0)
            && (core_grp_adj_tid_thres_cnt
                >= (core_grp_adj_tid_cnt - core_grp_adj_tid_thres_cnt))) {
            lpgemm_thread_attrs.tid_core_grp_load_high = TRUE;
        }
    }

err_handle:
    free(tid_cnt_for_core_grps);
    free(adj_tid_cnt_for_core_grps);

    if (thread_core_bind_list != NULL) {
        for (int ii = 0; ii < nt_max; ++ii) {
            free(thread_core_bind_list[ii]);
        }
    }
    free(thread_core_bind_list);
}

#else

static void
lpgemm_detect_thread_topo()
{
}

#endif // DLP_ENABLE_OPENMP

void
lpgemm_load_thread_attrs()
{
    lpgemm_thread_attrs.tid_core_grp_id_list   = NULL;
    lpgemm_thread_attrs.tid_cnt                = 0;
    lpgemm_thread_attrs.openmp_enabled         = FALSE;
    lpgemm_thread_attrs.tid_distr_nearly_seq   = FALSE;
    lpgemm_thread_attrs.tid_core_grp_load_high = FALSE;

    lpgemm_detect_thread_topo();
}

void
lpgemm_init_thread_attrs()
{
    dlp_pthread_once(&once_check_lpgemm_thread_topo_init,
                     lpgemm_load_thread_attrs);
}

// Should be called only after aocl_lpgemm_init_global_cntx.
lpgemm_thread_attrs_t*
lpgemm_get_thread_attrs()
{
    return &lpgemm_thread_attrs;
}

void
dlp_atomic_barrier(md_t t_id, dlp_task_comm_t* comm)
{
    if (comm == NULL || comm->n_threads == 1)
        return;

    // This orig_sense variable used as variable on which to spin for
    // waiting threads.
    md_t orig_sense = __atomic_load_n(&comm->barrier_sense, __ATOMIC_RELAXED);

    md_t my_threads_arrived =
        __atomic_add_fetch(&comm->barrier_threads_arrived, 1, __ATOMIC_ACQ_REL);

    // Last thread to arrive.
    if (my_threads_arrived == comm->n_threads) {
        comm->barrier_threads_arrived = 0;
        __atomic_fetch_xor(&comm->barrier_sense, 1, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&comm->barrier_sense, __ATOMIC_ACQUIRE)
               == orig_sense) {
        }
    }
}

typedef struct
{
    md_t n;
    md_t sqrt_n;
    md_t f;
} dlp_prime_factors_t;

static void
dlp_prime_factorization(md_t n, dlp_prime_factors_t* factors)
{
    factors->n      = n;
    factors->sqrt_n = (md_t)sqrt((double)n);
    factors->f      = 2;
}

static md_t
dlp_next_prime_factor(dlp_prime_factors_t* factors)
{
    // Sieve of eratosthenes style approach, but only for the factors of a
    // single number n, not for a range of numbers.
    while (factors->f <= factors->sqrt_n) {
        if (factors->f == 2) {
            if (factors->n % 2 == 0) {
                factors->n /= 2;
                return 2;
            }
            factors->f = 3;
        } else if (factors->f == 3) {
            if (factors->n % 3 == 0) {
                factors->n /= 3;
                return 3;
            }
            factors->f = 5;
        } else if (factors->f == 5) {
            if (factors->n % 5 == 0) {
                factors->n /= 5;
                return 5;
            }
            factors->f = 7;
        } else if (factors->f == 7) {
            if (factors->n % 7 == 0) {
                factors->n /= 7;
                return 7;
            }
            factors->f = 11;
        } else {
            if (factors->n % factors->f == 0) {
                factors->n /= factors->f;
                return factors->f;
            }
            factors->f++;
        }
    }

    // To get here we must be out of prime factors, leaving only n (if it is
    // prime) or an endless string of 1s.
    md_t tmp   = factors->n;
    factors->n = 1;
    return tmp;
}

bool
dlp_is_prime(md_t n)
{
    if (n < 2) {
        return FALSE;
    }

    dlp_prime_factors_t factors;
    dlp_prime_factorization(n, &factors);
    md_t f = dlp_next_prime_factor(&factors);

    return (f == n);
}

void
dlp_thread_partition_2x2(md_t n_thread,
                         md_t work1,
                         md_t work2,
                         md_t* restrict nt1,
                         md_t* restrict nt2)
{
    // Partition a number of threads into two factors nt1 and nt2 such that
    // nt1/nt2 ~= work1/work2. There is a fast heuristic algorithm and a
    // slower optimal algorithm (which minimizes |nt1*work2 - nt2*work1|).

    // Return early small prime numbers of threads.
    if (n_thread < 4) {
        *nt1 = (work1 >= work2 ? n_thread : 1);
        *nt2 = (work1 < work2 ? n_thread : 1);

        return;
    }

    // Compute with these local variables until the end of the function, at
    // which time we will save the values back to nt1 and nt2.
    md_t tn1 = 1;
    md_t tn2 = 1;

    // Both algorithms need the prime factorization of n_thread.
    dlp_prime_factors_t factors;
    dlp_prime_factorization(n_thread, &factors);

    // Fast algorithm: assign prime factors in increasing order to whichever
    // partition has more work to do. The work is divided by the number of
    // threads assigned at each iteration. This algorithm is sub-optimal in
    // some cases. We attempt to mitigate the cases that involve at least one
    // factor of 2. For example, in the partitioning of 12 with equal work
    // this algorithm tentatively finds 6x2. This factorization involves a
    // factor of 2 that can be reallocated, allowing us to convert it to the
    // optimal solution of 4x3. But some cases cannot be corrected this way
    // because they do not contain a factor of 2. For example, this algorithm
    // factors 105 (with equal work) into 21x5 whereas 7x15 would be optimal.
    md_t f;
    while ((f = dlp_next_prime_factor(&factors)) > 1) {
        if (work1 > work2) {
            work1 /= f;
            tn1 *= f;
        } else {
            work2 /= f;
            tn2 *= f;
        }
    }

    // Sometimes the last factor applied is prime. For example, on a square
    // matrix, we tentatively arrive (from the logic above) at:
    // - a 2x6 factorization when given 12 ways of parallelism
    // - a 2x10 factorization when given 20 ways of parallelism
    // - a 2x14 factorization when given 28 ways of parallelism
    // These factorizations are suboptimal under the assumption that we want
    // the parallelism to be as balanced as possible. Below, we make a final
    // attempt at rebalancing nt1 and nt2 by checking to see if the gap between
    // work1 and work2 is narrower if we reallocate a factor of 2.
    if (work1 > work2) {
        // Example: nt = 12
        //          w1 w2 (initial)   = 3600 3600; nt1 nt2 =  1 1
        //          w1 w2 (tentative) = 1800  600; nt1 nt2 =  2 6
        //          w1 w2 (ideal)     =  900 1200; nt1 nt2 =  4 3
        if (tn2 % 2 == 0) {
            md_t diff     = work1 - work2;
            md_t diff_mod = dlp_abs(work1 / 2 - work2 * 2);

            if (diff_mod < diff) {
                tn1 *= 2;
                tn2 /= 2;
            }
        }
    } else if (work1 < work2) {
        // Example: nt = 40
        //          w1 w2 (initial)   = 3600 3600; nt1 nt2 =  1 1
        //          w1 w2 (tentative) =  360  900; nt1 nt2 = 10 4
        //          w1 w2 (ideal)     =  720  450; nt1 nt2 =  5 8
        if (tn1 % 2 == 0) {
            md_t diff     = work2 - work1;
            md_t diff_mod = dlp_abs(work2 / 2 - work1 * 2);

            if (diff_mod < diff) {
                tn1 /= 2;
                tn2 *= 2;
            }
        }
    }

    // Save the final result.
    *nt1 = tn1;
    *nt2 = tn2;
}
