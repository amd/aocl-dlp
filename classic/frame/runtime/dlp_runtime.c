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

#include <stdlib.h>

#include "aocl_dlp_config.h"
#include "bindings/c_wrappers/capi_env_config.h"
#include "classic/aocl_lib_interface_apis.h"
#include "runtime/dlp_runtime.h"
#include "sys_utils/lpgemm_sys.h"

#ifdef DLP_ENABLE_OPENMP
#include <omp.h>
#endif

// The global rntm_t structure, which holds the global thread settings
// along with a few other key parameters.
dlp_rntm_t                dlp_lib_rntm = DLP_CLASSIC_RNTM_INITIALIZER;
static dlp_pthread_once_t once_init    = DLP_PTHREAD_ONCE_INIT;

// Make thread settings local to each thread calling DLP routines
DLP_CLASSIC_THREAD_LOCAL dlp_rntm_t dlp_tl_rntm = DLP_CLASSIC_RNTM_INITIALIZER;
DLP_CLASSIC_THREAD_LOCAL bool       dlp_init_tl_rntm = TRUE;

void
dlp_init_threading(void)
{
    md_t jc, ic, nt;

    // But if nested parallelism is enabled - Then each application will launch
    // MT DLP.
    //
    // Order of precedence used for number of threads:
    // 0. valid value set using dlp_thread_set_num_threads(nt) by the
    // application
    // 1. valid value set for DLP_NUM_THREADS environment variable
    // 2. omp_set_num_threads(nt) issued by the application
    // 3. valid value set for OMP_NUM_THREADS environment variable
    // 4. Number of cores
    //
    // Try to read DLP_NUM_THREADS first.
    nt = dlp_env_get_int("DLP_NUM_THREADS", -1);

    // Mark flag to denote threading set by DLP and not external entities
    // like OpenMP. This will be used to decide who control threading
    // during the entirety of this dlp instance runtime.
    if (nt > 0) {
        dlp_lib_rntm.ext_mt_ctr_var = FALSE;
    } else {
#ifdef DLP_ENABLE_OPENMP
        md_t active_level = omp_get_active_level();
        md_t max_levels   = omp_get_max_active_levels();
        if (active_level < max_levels) {
            nt = omp_get_max_threads();
        } else {
            nt = 1;
        }
#else
        nt = 1;
#endif
        dlp_lib_rntm.ext_mt_ctr_var = TRUE;
    }

    // Read the environment variables for the number of threads (ways
    // of parallelism) for each individual loop.
    jc = dlp_env_get_int("DLP_JC_NT", -1);
    ic = dlp_env_get_int("DLP_IC_NT", -1);

    if (jc != -1 || ic != -1) {
        jc = (jc == -1) ? 1 : jc;
        ic = (ic == -1) ? 1 : ic;
        nt = -1;

        dlp_lib_rntm.ext_mt_ctr_var = FALSE;
    }

    dlp_lib_rntm.num_threads = nt;
    dlp_lib_rntm.ic_ways     = ic;
    dlp_lib_rntm.jc_ways     = jc;
}

void
dlp_update_threading(dlp_rntm_t* rntm)
{
    md_t jc, ic, nt;

    // Extract threading data from rntm.
    nt = rntm->num_threads;
    jc = rntm->jc_ways;
    ic = rntm->ic_ways;

    if (rntm->ext_mt_ctr_var == FALSE) {
        if (jc != -1 || ic != -1) {
            jc = (jc == -1) ? 1 : jc;
            ic = (ic == -1) ? 1 : ic;

            // Unset the value for nt.
            nt = -1;
        }

#ifdef DLP_ENABLE_OPENMP
        // If call is not from an active OpenMP level, then it will be
        // serial irrespective of DLP threading settings.
        md_t active_level = omp_get_active_level();
        md_t max_levels   = omp_get_max_active_levels();
        if (active_level >= max_levels) {
            nt = -1;
            jc = ic = 1;
        }
#endif
    } else {
#ifdef DLP_ENABLE_OPENMP
        md_t active_level = omp_get_active_level();
        md_t max_levels   = omp_get_max_active_levels();
        if (active_level < max_levels) {
            nt = omp_get_max_threads();
        } else {
            nt = 1;
        }
#else
        nt = 1;
#endif
    }

    rntm->num_threads = nt;
    rntm->ic_ways     = ic;
    rntm->jc_ways     = jc;
}

DLP_INLINE void
dlp_init_threading_once()
{
    dlp_pthread_once(&once_init, dlp_init_threading);
}

void
dlp_rntm_init_from_global(dlp_rntm_t* rntm)
{
    // We must ensure that dlp_lib_rntm and dlp_tl_rntm have been initialized
    dlp_init_threading_once();

    // Initialize dlp_tl_rntm as a copy of dlp_lib_rntm
    // Need to do this once per application thread
    if (dlp_init_tl_rntm == TRUE) {
        dlp_tl_rntm      = dlp_lib_rntm;
        dlp_init_tl_rntm = FALSE;
    }

    // Initialize supplied rntm from dlp_tl_rntm.
    *rntm = dlp_tl_rntm;

    // This is to account for threading changes that might have happened
    // between DLP API calls.
    dlp_update_threading(rntm);
}

void
dlp_thread_set_ways(md_t jc, md_t ic)
{
    // We must ensure that dlp_lib_rntm and dlp_tl_rntm have been initialized
    dlp_init_threading_once();

    // Update dlp_tl_rntm so any threads spawned after this call
    // inherit the values set here.
    dlp_tl_rntm.ic_ways = ic;
    dlp_tl_rntm.jc_ways = jc;

    // DLP artifacts is used to set threading. Need to ensure OMP API or
    // env variables will not be of effect going forward.
    dlp_tl_rntm.ext_mt_ctr_var = FALSE;
}

void
dlp_thread_set_num_threads(md_t n_threads)
{
    // We must ensure that dlp_lib_rntm and dlp_tl_rntm have been initialized
    dlp_init_threading_once();

    if (n_threads <= 0) {
        n_threads = 1;
    }

    // Update dlp_tl_rntm so any threads spawned after this call
    // inherit the value set here.
    dlp_tl_rntm.num_threads = n_threads;

    // DLP artifacts is used to set threading. Need to ensure OMP API or
    // env variables will not be of effect going forward.
    dlp_tl_rntm.ext_mt_ctr_var = FALSE;
}

void
dlp_version_query(int* major, int* minor, int* patch)
{
    char* AOCL_DLP_VERSION_MAJOR_STR = AOCL_DLP_VERSION_MAJOR;
    char* AOCL_DLP_VERSION_MINOR_STR = AOCL_DLP_VERSION_MINOR;
    char* AOCL_DLP_VERSION_PATCH_STR = AOCL_DLP_VERSION_PATCH;
    if (major)
        *major = atoi(AOCL_DLP_VERSION_MAJOR_STR);
    if (minor)
        *minor = atoi(AOCL_DLP_VERSION_MINOR_STR);
    if (patch)
        *patch = atoi(AOCL_DLP_VERSION_PATCH_STR);
}
