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

#ifndef DLP_OMP_RUNTIME_H
#define DLP_OMP_RUNTIME_H

/*
 * DLP owns this runtime boundary so OpenMP 3.0/4.0 queries can be expressed
 * using the OpenMP 2.0 surface available from MSVC's VCOMP runtime.  VCOMP
 * deliberately reports no capacity for an inner DLP team while already inside
 * an outer parallel region, even when nested execution is enabled, to avoid
 * assuming unsupported active-level semantics and oversubscribing the caller.
 */
#include <stdbool.h>

#if defined(__GNUC__) || defined(__clang__)
#define DLP_OMP_RUNTIME_LOCAL __attribute__((visibility("hidden")))
#else
#define DLP_OMP_RUNTIME_LOCAL
#endif

/* Keep omp.h outside the C-linkage block.  Some C++ OpenMP headers declare
 * templates and cannot be parsed with C linkage. */
#include "dlp_openmp_config.h"
#if DLP_OPENMP_ENABLED
#ifdef __cplusplus
extern "C++"
{
#endif
#include <omp.h>
#ifdef __cplusplus
}
#endif
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    static inline int dlp_omp_get_max_threads(void)
    {
#if DLP_OPENMP_ENABLED
        return omp_get_max_threads();
#else
    return 1;
#endif
    }

    static inline int dlp_omp_get_num_procs(void)
    {
#if DLP_OPENMP_ENABLED
        return omp_get_num_procs();
#else
    return 1;
#endif
    }

    static inline int dlp_omp_get_num_threads(void)
    {
#if DLP_OPENMP_ENABLED
        return omp_get_num_threads();
#else
    return 1;
#endif
    }

    static inline int dlp_omp_get_thread_num(void)
    {
#if DLP_OPENMP_ENABLED
        return omp_get_thread_num();
#else
    return 0;
#endif
    }

    static inline void dlp_omp_set_num_threads(int n)
    {
#if DLP_OPENMP_ENABLED
        omp_set_num_threads(n);
#else
    (void)n;
#endif
    }

    static inline bool dlp_omp_is_in_parallel_region(void)
    {
#if DLP_OPENMP_ENABLED
        return omp_in_parallel() != 0;
#else
    return false;
#endif
    }

    static inline bool dlp_omp_can_create_parallel_team(void)
    {
#if DLP_OPENMP_ENABLED
#if DLP_OPENMP_RUNTIME_VCOMP
        /* VCOMP nesting is a binary runtime setting; DLP remains conservative
         * and serializes an inner operation inside any outer region. */
        return !dlp_omp_is_in_parallel_region();
#elif DLP_OPENMP_HAS_ACTIVE_LEVELS
        return omp_get_active_level() < omp_get_max_active_levels();
#else
        return !dlp_omp_is_in_parallel_region();
#endif
#else
    return false;
#endif
    }

    static inline void dlp_omp_set_nested_parallelism(bool enabled)
    {
#if DLP_OPENMP_ENABLED
#if DLP_OPENMP_RUNTIME_VCOMP
#if DLP_OPENMP_HAS_NESTING
        omp_set_nested(enabled ? 1 : 0);
#else
        (void)enabled;
#endif
#elif DLP_OPENMP_HAS_ACTIVE_LEVELS
        omp_set_max_active_levels(enabled ? 2 : 1);
#elif DLP_OPENMP_HAS_NESTING
        omp_set_nested(enabled ? 1 : 0);
#else
        (void)enabled;
#endif
#else
    (void)enabled;
#endif
    }

    typedef struct dlp_omp_topology
    {
        /* True only when affinity/topology data was collected successfully. */
        bool available;
        int  tid_cnt;
        bool tid_distr_nearly_seq;
        bool tid_core_grp_load_high;
        int* tid_core_grp_id_list;
    } dlp_omp_topology_t;

    /*
     * This is deliberately the only out-of-line wrapper operation.  It is cold
     * topology discovery, and returns false whenever the selected runtime
     * cannot provide the DLP-specific information safely.
     */
    DLP_OMP_RUNTIME_LOCAL bool dlp_omp_collect_thread_topology(
        dlp_omp_topology_t* out);

#ifdef __cplusplus
}
#endif

#endif /* DLP_OMP_RUNTIME_H */
