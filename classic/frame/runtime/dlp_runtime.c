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

#include <stdatomic.h>
#include <stdlib.h>

#include "aocl_dlp_config.h"
#include "bindings/c_wrappers/capi_env_config.h"
#include "bindings/c_wrappers/capi_sys_utils.h"
#include "classic/aocl_lib_interface_apis.h"
#include "runtime/dlp_runtime.h"
#include "sys_utils/dlp_gemm_sys.h"

#ifdef DLP_ENABLE_OPENMP
#include <omp.h>
#endif

// The global rntm_t structure, which holds the global thread settings
// along with a few other key parameters.
static dlp_pthread_once_t once_init = DLP_PTHREAD_ONCE_INIT;

/**
 * ATOMIC THREADING CONFIGURATION GLOBALS
 *
 * Design Rationale:
 * - 8-byte atomics are GUARANTEED lock-free on all x86-64 platforms
 * - 16-byte atomics are NOT reliably lock-free (toolchain dependent)
 * - Solution: Pack multiple values into 8-byte atomics
 *
 * Memory Layout:
 * - lib_packed_ways: Packs ic_ways and jc_ways (both int32_t) into single
 * int64_t Bit layout: [63:32] = jc_ways, [31:0] = ic_ways
 * - lib_num_threads: Direct int64_t storage for thread count
 * - lib_ext_mt_ctr_var: Boolean flag for external threading control
 *
 * CRITICAL INVARIANT:
 * ic_ways and jc_ways MUST fit in int32_t (range: -2,147,483,648 to
 * 2,147,483,647). Current usage: Thread counts rarely exceed 1024, providing
 * 2,000,000x safety margin. If ever larger values needed, this packing scheme
 * must be redesigned.
 */
static __attribute__((aligned(64))) _Atomic int64_t lib_packed_ways    = -1;
static __attribute__((aligned(64))) _Atomic int64_t lib_num_threads    = -1;
static __attribute__((aligned(64))) _Atomic bool    lib_ext_mt_ctr_var = FALSE;

_Static_assert(sizeof(int32_t) == 4, "Pack optimization requires 32-bit ints");

/**
 * @brief Pack two int32_t values (ic_ways, jc_ways) into single int64_t
 *
 * Bit layout: [63:32] jc_ways (upper 32 bits)
 *             [31:0]  ic_ways (lower 32 bits)
 *
 * This allows atomic update of both values with a single 8-byte atomic store,
 * which is guaranteed lock-free on all x86-64 platforms.
 *
 * Implementation Note:
 * Casts to uint32_t before shifting to avoid undefined behavior with signed
 * shifts. Two's complement representation ensures negative values are preserved
 * correctly.
 *
 * @param ic IC loop parallelism way count (can be negative, e.g., -1 for unset)
 * @param jc JC loop parallelism way count (can be negative, e.g., -1 for unset)
 * @return Packed int64_t containing both values
 *
 * Example: dlp_pack_ways(-1, -1) = 0xFFFFFFFF_FFFFFFFF = -1LL
 */
static inline int64_t
dlp_pack_ways(int32_t ic, int32_t jc)
{
    // Cast to unsigned to avoid undefined behavior with signed left shifts
    uint32_t uic = (uint32_t)ic;
    uint32_t ujc = (uint32_t)jc;
    return ((uint64_t)ujc << 32) | uic;
}

/**
 * @brief Unpack int64_t into two int32_t values (ic_ways, jc_ways)
 *
 * Reverses the packing operation, extracting both values from packed int64_t.
 * Correctly handles negative values via two's complement representation.
 *
 * @param packed Packed int64_t containing ic and jc
 * @param[out] ic Pointer to receive ic_ways value (lower 32 bits)
 * @param[out] jc Pointer to receive jc_ways value (upper 32 bits)
 *
 * Example: dlp_unpack_ways(-1LL, &ic, &jc) → ic=-1, jc=-1
 */
static inline void
dlp_unpack_ways(int64_t packed, int32_t* ic, int32_t* jc)
{
    // Extract as unsigned to avoid sign extension issues, then cast to signed
    *ic = (int32_t)((uint64_t)packed & 0xFFFFFFFF);
    *jc = (int32_t)(((uint64_t)packed >> 32) & 0xFFFFFFFF);
}

// Make thread settings local to each thread calling DLP routines
DLP_CLASSIC_THREAD_LOCAL dlp_rntm_t dlp_tl_rntm = DLP_CLASSIC_RNTM_INITIALIZER;
DLP_CLASSIC_THREAD_LOCAL bool       dlp_init_tl_rntm = TRUE;

/**
 * @brief Initialize global threading configuration from environment variables
 *
 * Called exactly once via dlp_pthread_once. Sets global atomic variables that
 * control DLP threading behavior.
 *
 * Precedence order (highest to lowest):
 * 1. DLP_IC_NT/DLP_JC_NT environment variables (ways-based threading)
 * 2. DLP_NUM_THREADS environment variable (thread count-based)
 * 3. OpenMP settings via omp_get_max_threads() (if DLP_ENABLE_OPENMP)
 * 4. System core count (fallback)
 *
 * @note Special value -1 means "unset" for all threading parameters.
 */
void
dlp_init_threading(void)
{
    md_t jc = -1;
    md_t ic = -1;
    md_t nt = -1;

    // This will be used to decide who control threading during the
    // entirety of this dlp instance runtime.
    bool lcl_ext_mt_ctr_var = FALSE;

    bool openmp_enabled = FALSE;
#ifdef DLP_ENABLE_OPENMP
    openmp_enabled = TRUE;
#endif

    // Refer dlp_update_threading_priority_order function for the precedence
    // order of setting threading parameters.
    ic = dlp_env_get_int("DLP_IC_NT", -1);
    jc = dlp_env_get_int("DLP_JC_NT", -1);
    nt = dlp_env_get_int("DLP_NUM_THREADS", -1);

    if (jc > 0 || ic > 0) {
        ic = (ic <= 0) ? 1 : ic;
        jc = (jc <= 0) ? 1 : jc;
        nt = -1;
    } else if (nt > 0) {
        // It could be the case that ic and jc is initialized to < -1
        // via the env vars. Need to set to valid values.
        ic = -1;
        jc = -1;
    } else if (openmp_enabled == TRUE) {
#ifdef DLP_ENABLE_OPENMP
        md_t active_level = omp_get_active_level();
        md_t max_levels   = omp_get_max_active_levels();
        if (active_level < max_levels) {
            nt = omp_get_max_threads();
        } else {
            nt = 1;
        }
        lcl_ext_mt_ctr_var = TRUE;

        // It could be the case that ic and jc is initialized to < -1
        // via the env vars. Need to set to valid values.
        ic = -1;
        jc = -1;
#endif
    } else {
        // Reaching this branch implies openmp_enabled == FALSE (the
        // preceding else-if would have consumed the OpenMP case). In a
        // single-thread build (DLP_ENABLE_ST / threading model "none"),
        // the executor is hard-wired to a single thread, so advertising
        // num_cores here would be fictional. It would also have a real
        // downside: it makes dlp_is_single_thread() return FALSE and
        // silently disables the tiny-input fast path unless the caller
        // explicitly sets DLP_NUM_THREADS=1. Default to nt = 1 in that
        // build to match the executor. In other non-OpenMP builds, use
        // num_cores as the default.
        // Always safe to set nt = 1 instead of ic or jc. Otherwise if post
        // this the dlp_thread_set_num_threads_library API is called, the
        // library will ignore nt, since ic and jc are already set and have
        // higher precedence.
#ifdef DLP_ENABLE_ST
        nt = 1;
#else
        md_t num_cores = dlp_get_num_cores();
        nt             = (num_cores > 0) ? num_cores : 1;
#endif
        ic = -1;
        jc = -1;
    }

    // Only initialize library rntm and not the thread local rntm. Thread
    // local rntm should be explicitly opted into. Else the library rntm
    // will be used to determine threading. In case the library rntm is
    // zeroed out, then the env vars will be used. All this assuming the
    // ext_mt_ctr_var is FALSE. Else OpenMP will control threading.
    lib_num_threads    = nt;
    lib_packed_ways    = dlp_pack_ways((int32_t)ic, (int32_t)jc);
    lib_ext_mt_ctr_var = lcl_ext_mt_ctr_var;
}

/**
 * @brief Update runtime threading configuration based on priority order
 *
 * Determines active threading parameters by checking multiple sources in
 * priority order. The first non-empty source wins.
 *
 * Priority Order (highest to lowest):
 * 1. Thread-local settings (dlp_tl_rntm) - set via dlp_thread_set_*_local()
 * 2. Library-global settings (lib_*) - set via dlp_thread_set_*_library()
 * 3. Environment variables (DLP_IC_NT, DLP_JC_NT, DLP_NUM_THREADS)
 * 4. OpenMP settings (omp_get_max_threads) - if DLP_ENABLE_OPENMP
 * 5. System core count (fallback)
 *
 * Threading Mode Selection:
 * - Ways mode: If ic_ways > 0 OR jc_ways > 0, use ways (num_threads ignored)
 * - Thread count mode: If num_threads > 0 AND ways unset, use num_threads
 *
 * Atomic Consistency Strategy:
 * Three separate atomic loads (ext_mt, nt, ways) are NOT atomic as a group.
 * This is safe because API contracts ensure mutually exclusive configurations:
 * - set_ways_library(ic,jc) always sets nt=-1 and ext_mt=FALSE
 * - set_num_threads_library(nt) keeps ways unchanged, sets ext_mt=FALSE
 * - Ways have higher precedence, so temporal skew is harmless
 *
 * @param[out] rntm Runtime configuration to populate
 *
 * @note Value -1 means "unset" for threading parameters
 */
void
dlp_update_threading_priority_order(dlp_rntm_t* rntm)
{
    md_t act_ic = -1;
    md_t act_jc = -1;
    md_t act_nt = -1;

    // Thread local rntm gets highest priority.
    if (dlp_tl_rntm.ext_mt_ctr_var == FALSE) {
        // Extract threading data from rntm.
        md_t tl_nt = dlp_tl_rntm.num_threads;
        md_t tl_ic = dlp_tl_rntm.ic_ways;
        md_t tl_jc = dlp_tl_rntm.jc_ways;

        if (tl_jc > 0 || tl_ic > 0) {
            act_ic = (tl_ic <= 0) ? 1 : tl_ic;
            act_jc = (tl_jc <= 0) ? 1 : tl_jc;

            // Unset the value for nt.
            act_nt = -1;
        } else if (tl_nt > 0) {
            // If nt is also not set, then default to single threaded.
            act_nt = tl_nt;
            act_ic = -1;
            act_jc = -1;
        }
    }

    // Library level rntm gets second priority.
    if ((act_nt == -1) && (act_ic == -1) && (act_jc == -1)) {
        bool lcl_ext_mt_ctr_var =
            atomic_load_explicit(&lib_ext_mt_ctr_var, memory_order_acquire);
        // Even though it seems there is chance for TOCTOU issue here, it is
        // guaranteed not to happen. Typically race conditions can arise from
        // calling the dlp_thread_set_ways_library and dlp_thread_set_num_
        // threads_library APIs from multiple threads. However even in this
        // case ext_mt_ctr_var will always be set to false. Secondly the
        // values for ic,jc and nt will always be > 0 or -1 (unset). So the
        // eventual consistency should be good enough for most cases, and even
        // in worst case it will be a delayed update.
        if (lcl_ext_mt_ctr_var == FALSE) {
            // Extract threading data from rntm.
            md_t lib_nt =
                atomic_load_explicit(&lib_num_threads, memory_order_acquire);

            md_t packed_ways =
                atomic_load_explicit(&lib_packed_ways, memory_order_acquire);
            int32_t tmp_lib_ic = -1;
            int32_t tmp_lib_jc = -1;
            dlp_unpack_ways(packed_ways, &tmp_lib_ic, &tmp_lib_jc);
            md_t lib_ic = tmp_lib_ic;
            md_t lib_jc = tmp_lib_jc;

            if (lib_jc > 0 || lib_ic > 0) {
                act_ic = (lib_ic <= 0) ? 1 : lib_ic;
                act_jc = (lib_jc <= 0) ? 1 : lib_jc;

                // Unset the value for nt.
                act_nt = -1;
            } else if (lib_nt > 0) {
                // If nt is also not set, then default to single threaded.
                act_nt = lib_nt;
                act_ic = -1;
                act_jc = -1;
            }
        }
    }

    // DLP specific environment variables gets third priority.
    // It is guaranteed that threading settings are not set by external
    // entities (like OpenMP) if the ext_mt_ctr_var is FALSE for library
    // rntm (refer init function). In which case querying DLP thread
    // specific env vars is valid.
    if ((act_nt == -1) && (act_ic == -1) && (act_jc == -1)) {
        bool lcl_ext_mt_ctr_var =
            atomic_load_explicit(&lib_ext_mt_ctr_var, memory_order_acquire);
        if (lcl_ext_mt_ctr_var == FALSE) {
            md_t env_ic = dlp_env_get_int("DLP_IC_NT", -1);
            md_t env_jc = dlp_env_get_int("DLP_JC_NT", -1);
            md_t env_nt = dlp_env_get_int("DLP_NUM_THREADS", -1);

            // To be noted that the lib_num_threads and lib_packed_ways env
            // vars will not be set here even though the DLP threading env
            // vars are defined. This is because during init lib_num_threads
            // and lib_packed_ways would have been set based on the env vars.
            // However if this condition is still encountered, then it means
            // the lib threads and ways have been overridden by the library
            // APIs after init, and should be left as is.
            if (env_jc > 0 || env_ic > 0) {
                act_ic = (env_ic <= 0) ? 1 : env_ic;
                act_jc = (env_jc <= 0) ? 1 : env_jc;
                act_nt = -1;
            } else if (env_nt > 0) {
                act_nt = env_nt;
                act_ic = -1;
                act_jc = -1;
            }
        }
    }

#ifdef DLP_ENABLE_OPENMP
    // OpenMP specific environment variables or APIs gets fourth priority.
    // To be noted this will transfer the DLP threading control to an
    // external entity (OpenMP).
    if ((act_nt == -1) && (act_ic == -1) && (act_jc == -1)) {
        md_t omp_nt = omp_get_max_threads();
        if (omp_nt > 0) {
            act_nt                     = omp_nt;
            act_ic                     = -1;
            act_jc                     = -1;
            dlp_tl_rntm.ext_mt_ctr_var = TRUE;
            atomic_store_explicit(&lib_ext_mt_ctr_var, TRUE,
                                  memory_order_release);
        }
    }
#endif

    // If none of the thread control parameters have been set till now,
    // default to system cores as number of threads in OpenMP builds.
    // When built without OpenMP the executor can only run single-threaded
    // (see the non-OpenMP _thread_decorator in
    // dlp_gemm_thread_decor_openmp.c), so default to nt = 1 instead. This
    // keeps dlp_is_single_thread() == TRUE by default and prevents the
    // tiny-input fast path from being silently disabled in non-OpenMP
    // builds when the caller has not explicitly set DLP_NUM_THREADS=1.
    if ((act_nt == -1) && (act_ic == -1) && (act_jc == -1)) {
#ifdef DLP_ENABLE_ST
        act_nt = 1;
#else
        md_t num_cores = dlp_get_num_cores();
        act_nt         = (num_cores > 0) ? num_cores : 1;
#endif
        act_ic = -1;
        act_jc = -1;
    }

#ifdef DLP_ENABLE_OPENMP
    // Adjust the threading settings to be single threaded if the call is
    // from a nested omp parallel region and max active levels is not set
    // properly. This applies irrespective of the source of the threading
    // settings, as OpenMP will not be able to explicitly set number of
    // threads in this scenario and can lead to oversubscription.
    if ((act_nt > 0) || (act_ic > 0) || (act_jc > 0)) {
        md_t active_level = omp_get_active_level();
        md_t max_levels   = omp_get_max_active_levels();
        if (active_level >= max_levels) {
            act_nt = -1;
            act_ic = 1;
            act_jc = 1;
        }
    }
#endif

    rntm->num_threads = act_nt;
    rntm->ic_ways     = act_ic;
    rntm->jc_ways     = act_jc;
}

/**
 * @brief Ensure global threading initialized exactly once (internal helper)
 *
 * Uses pthread_once to guarantee dlp_init_threading() executes exactly once.
 * Also initializes thread-local ext_mt_ctr_var on first call per thread.
 *
 * Thread Safety: Thread-safe via pthread_once mechanism.
 */
DLP_INLINE void
dlp_init_threading_once()
{
    dlp_pthread_once(&once_init, dlp_init_threading);

    if (dlp_init_tl_rntm == TRUE) {
        // Only the external control variable is to be set for the thread
        // local rntm. The rest of the thread local rntm attributes should
        // only be set when the dlp_thread_set_ways_local or dlp_thread_set
        // _num_threads_local APIs are called.
        dlp_tl_rntm.ext_mt_ctr_var =
            atomic_load_explicit(&lib_ext_mt_ctr_var, memory_order_acquire);
        dlp_init_tl_rntm = FALSE;
    }
}

/**
 * @brief Initialize runtime configuration from global and thread-local settings
 *
 * Entry point for GEMM operations to obtain active threading configuration.
 * Combines global and thread-local settings based on priority order.
 *
 * @param[out] rntm Runtime configuration structure to populate
 *
 * @note Called at start of each GEMM operation
 * @note Resets rntm before populating to ensure clean state
 */
void
dlp_rntm_init_from_global(dlp_rntm_t* rntm)
{
    // We must ensure that dlp_lib_rntm and dlp_tl_rntm have been initialized
    dlp_init_threading_once();

    // TODO: Once the support for DLP_PACK_A and DLP_PACK_B env vars are
    // added, the rntm should be updated to account for those as well.
    DLP_CLASSIC_RNTM_RESET(*rntm);

    // This is to account for threading changes that might have happened
    // between DLP API calls.
    dlp_update_threading_priority_order(rntm);
}

void
dlp_thread_set_ways_library(md_t jc, md_t ic)
{
    // We must ensure that dlp_lib_rntm and dlp_tl_rntm have been initialized
    dlp_init_threading_once();

    if (ic <= 0) {
        ic = -1;
    }
    if (jc <= 0) {
        jc = -1;
    }

    int64_t packed_ways = dlp_pack_ways((int32_t)ic, (int32_t)jc);
    atomic_store_explicit(&lib_packed_ways, packed_ways, memory_order_release);

    // Also set the num_threads to -1 along with disabling the external control
    // flag. An interesting property here is that even though the ways and nt
    // are independently set using atomics, the combination is guaranteed to
    // act like an atomic transaction. Reason being any thread/instance calling
    // this function always implies nt is -1 and ext_mt_ctr_var is FALSE.
    // Unset num_threads when ways are set.
    atomic_store_explicit(&lib_num_threads, -1, memory_order_release);

    // DLP artifacts are used to set threading. Need to ensure OMP API or
    // env variables will not be of effect going forward.
    atomic_store_explicit(&lib_ext_mt_ctr_var, FALSE, memory_order_release);
}

void
dlp_thread_set_ways_local(md_t jc, md_t ic)
{
    dlp_init_threading_once();

    if (ic <= 0) {
        ic = -1;
    }
    if (jc <= 0) {
        jc = -1;
    }

    dlp_tl_rntm.ic_ways     = ic;
    dlp_tl_rntm.jc_ways     = jc;
    dlp_tl_rntm.num_threads = -1; // Unset num_threads when ways are set.

    // DLP artifacts are used to set threading. Need to ensure OMP API or
    // env variables will not be of effect going forward.
    dlp_tl_rntm.ext_mt_ctr_var = FALSE;
}

void
dlp_thread_set_ways(md_t jc, md_t ic)
{
    dlp_thread_set_ways_local(jc, ic);
}

md_t
dlp_thread_get_ic_ways_active(void)
{
    dlp_rntm_t rntm_act = DLP_CLASSIC_RNTM_INITIALIZER;
    dlp_rntm_init_from_global(&rntm_act);
    return rntm_act.ic_ways;
}

md_t
dlp_thread_get_jc_ways_active(void)
{
    dlp_rntm_t rntm_act = DLP_CLASSIC_RNTM_INITIALIZER;
    dlp_rntm_init_from_global(&rntm_act);
    return rntm_act.jc_ways;
}

void
dlp_thread_set_num_threads_library(md_t n_threads)
{
    dlp_init_threading_once();

    if (n_threads <= 0) {
        n_threads = -1;
    }

    // The dlp_rntm_ways_lib ic,jc are not modified in this case. The ic,jc
    // ways have higher priority, and in the event both the dlp_thread_set_
    // ways_library and dlp_thread_set_num_threads_library APIs are called,
    // the library will prioritise using the ic,jc ways for threading.
    atomic_store_explicit(&lib_num_threads, n_threads, memory_order_release);

    // DLP artifacts are used to set threading. Need to ensure OMP API or
    // env variables will not be of effect going forward.
    atomic_store_explicit(&lib_ext_mt_ctr_var, FALSE, memory_order_release);
}

void
dlp_thread_set_num_threads_local(md_t n_threads)
{
    dlp_init_threading_once();

    if (n_threads <= 0) {
        n_threads = -1;
    }

    // The dlp_rntm_ways_lib ic,jc are not modified in this case. The ic,jc
    // ways have higher priority, and in the event both the dlp_thread_set_
    // ways_library and dlp_thread_set_num_threads_library APIs are called,
    // the library will prioritise using the ic,jc ways for threading.
    dlp_tl_rntm.num_threads = n_threads;

    // DLP artifacts are used to set threading. Need to ensure OMP API or
    // env variables will not be of effect going forward.
    dlp_tl_rntm.ext_mt_ctr_var = FALSE;
}

void
dlp_thread_set_num_threads(md_t n_threads)
{
    dlp_thread_set_num_threads_local(n_threads);
}

md_t
dlp_thread_get_num_threads_active(void)
{
    dlp_rntm_t rntm_act = DLP_CLASSIC_RNTM_INITIALIZER;
    dlp_rntm_init_from_global(&rntm_act);
    return rntm_act.num_threads;
}

void
dlp_version_query(int* major, int* minor, int* patch)
{
    if (major)
        *major = AOCL_DLP_VERSION_MAJOR;
    if (minor)
        *minor = AOCL_DLP_VERSION_MINOR;
    if (patch)
        *patch = AOCL_DLP_VERSION_PATCH;
}
