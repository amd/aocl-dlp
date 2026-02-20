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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

/**
 * @file test_runtime_threading.cc
 * @brief Unit tests for DLP runtime threading control APIs
 *
 * Tests the public threading control APIs defined in aocl_lib_interface_apis.h:
 * - dlp_thread_set_ways_library() / _local()
 * - dlp_thread_set_num_threads_library() / _local()
 * - dlp_thread_get_ic_ways_active() / jc_ways / num_threads
 *
 * Focus: API behavior, precedence order, and state management
 * Scope: Single-threaded tests (atomic consistency tested separately)
 *
 * PRECEDENCE ORDER (highest to lowest):
 * 1. Thread-local ways (tl_ic_ways, tl_jc_ways if > 0)
 * 2. Thread-local num_threads (tl_nt if > 0 AND tl_ways unset)
 * 3. Library ways (lib_ic_ways, lib_jc_ways if > 0 AND all tl unset)
 * 4. Library num_threads (lib_nt if > 0 AND all above unset)
 * 5. Environment variables / OpenMP / System cores (fallback)
 */

#include <cstdlib>
#include <gtest/gtest.h>

extern "C"
{
#include "classic/aocl_lib_interface_apis.h"
}

// ============================================================================
// TEST HELPERS
// ============================================================================

namespace {
/**
 * Check if DLP environment variable is set with positive value
 */
bool
isEnvVarDefined(const char* name)
{
    const char* value = std::getenv(name);
    if (!value) {
        return false;
    } else {
        return true;
    }
}

/**
 * Get environment variable value (or -1 if not set/invalid)
 */
long
getEnvVarValue(const char* name)
{
    const char* value = std::getenv(name);
    if (!value) {
        return -1;
    }

    long val = std::atol(value);
    return val > 0 ? val : -1;
}
} // namespace

// ============================================================================
// TEST FIXTURES
// ============================================================================

/**
 * Base test fixture for threading API tests
 * Resets threading configuration to known state before each test
 */
class ThreadingAPITest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // Reset all threading configurations to "unset" state
        // This ensures each test starts with clean slate
        dlp_thread_set_ways_library(-1, -1);
        dlp_thread_set_num_threads_library(-1);
        dlp_thread_set_ways_local(-1, -1);
        dlp_thread_set_num_threads_local(-1);
    }

    void TearDown() override
    {
        // Restore to defaults after each test
        dlp_thread_set_ways_library(-1, -1);
        dlp_thread_set_num_threads_library(-1);
        dlp_thread_set_ways_local(-1, -1);
        dlp_thread_set_num_threads_local(-1);
    }
};

/**
 * Test fixture for precedence order testing
 * Inherits from ThreadingAPITest for common setup/teardown
 */
class ThreadingPrecedenceTest : public ThreadingAPITest
{
  protected:
    // Additional setup specific to precedence tests if needed
};

// ============================================================================
// CATEGORY 1: BASIC SETTER/GETTER TESTS
// ============================================================================

/**
 * Test: Set and get library-global ways (ic, jc)
 * Verifies basic functionality of ways-based configuration
 */
TEST_F(ThreadingAPITest, SetLibraryWaysBasic)
{
    dlp_thread_set_ways_library(4, 8);

    EXPECT_EQ(dlp_thread_get_ic_ways_active(), 8);
    EXPECT_EQ(dlp_thread_get_jc_ways_active(), 4);
}

/**
 * Test: Set and get thread-local ways (ic, jc)
 * Verifies thread-local configuration works independently
 */
TEST_F(ThreadingAPITest, SetLocalWaysBasic)
{
    dlp_thread_set_ways_local(2, 16);

    EXPECT_EQ(dlp_thread_get_ic_ways_active(), 16);
    EXPECT_EQ(dlp_thread_get_jc_ways_active(), 2);
}

/**
 * Test: Set and get library-global thread count
 * Verifies basic thread count configuration
 */
TEST_F(ThreadingAPITest, SetLibraryThreadsBasic)
{
    dlp_thread_set_num_threads_library(16);

    EXPECT_EQ(dlp_thread_get_num_threads_active(), 16);
}

/**
 * Test: Set and get thread-local thread count
 * Verifies thread-local thread count works independently
 */
TEST_F(ThreadingAPITest, SetLocalThreadsBasic)
{
    dlp_thread_set_num_threads_local(8);

    EXPECT_EQ(dlp_thread_get_num_threads_active(), 8);
}

/**
 * Test: Reset library to reveal defaults/environment variables
 * When all configs unset, falls back to env vars or system defaults
 */
TEST_F(ThreadingPrecedenceTest, ResetAllRevealsDefaults)
{
    bool has_env_ic = isEnvVarDefined("DLP_IC_NT");
    bool has_env_jc = isEnvVarDefined("DLP_JC_NT");
    bool has_env_nt = isEnvVarDefined("DLP_NUM_THREADS");

    dlp_thread_set_num_threads_library(32);
    EXPECT_EQ(dlp_thread_get_num_threads_active(), 32);
    // Ways not set yet.
    EXPECT_EQ(dlp_thread_get_ic_ways_active(), -1);
    EXPECT_EQ(dlp_thread_get_jc_ways_active(), -1);

    dlp_thread_set_num_threads_library(-1);

    // Verify fallback based on environment.
    if (has_env_ic || has_env_jc) {
        // Falls back to environment ways.
        md_t env_ic = getEnvVarValue("DLP_IC_NT");
        md_t env_jc = getEnvVarValue("DLP_JC_NT");
        env_ic      = (env_ic <= 0) ? -1 : env_ic;
        env_jc      = (env_jc <= 0) ? -1 : env_jc;

        EXPECT_EQ(dlp_thread_get_ic_ways_active(), env_ic);
        EXPECT_EQ(dlp_thread_get_jc_ways_active(), env_jc);
    } else if (has_env_nt) {
        // Falls back to environment num_threads.
        md_t env_nt = getEnvVarValue("DLP_NUM_THREADS");
        env_nt      = (env_nt <= 0) ? -1 : env_nt;
        if (env_nt > 0) {
            EXPECT_EQ(dlp_thread_get_num_threads_active(), env_nt);
        } else {
            // Env var set but invalid - should fall back to omp if enabled
            // else system cores.
            md_t nt = dlp_thread_get_num_threads_active();
            EXPECT_GT(nt, 0);
        }
    } else {
        // Falls back to omp if enabled else system cores.
        md_t nt = dlp_thread_get_num_threads_active();
        EXPECT_GT(nt, 0);
    }
}

/**
 * Test: Alias functions work identically to _local variants
 * Verifies dlp_thread_set_ways() and dlp_thread_set_num_threads() are aliases
 */
TEST_F(ThreadingAPITest, AliasFunctionsEquivalent)
{
    dlp_thread_set_ways(4, 8);
    EXPECT_EQ(dlp_thread_get_ic_ways_active(), 8);
    EXPECT_EQ(dlp_thread_get_jc_ways_active(), 4);

    dlp_thread_set_ways_local(-1, -1);

    dlp_thread_set_num_threads(16);
    EXPECT_EQ(dlp_thread_get_num_threads_active(), 16);
}

/**
 * Test: Negative/zero values normalized to -1 (unset)
 * Note: Falls back to environment variables if DLP_IC_NT/DLP_JC_NT set
 */
TEST_F(ThreadingAPITest, NegativeValuesNormalizedToUnset)
{
    bool has_env_ic = isEnvVarDefined("DLP_IC_NT");
    bool has_env_jc = isEnvVarDefined("DLP_JC_NT");

    dlp_thread_set_ways_library(-5, -10);

    if (!has_env_ic && !has_env_jc) {
        // No env vars - should be -1
        EXPECT_EQ(dlp_thread_get_ic_ways_active(), -1);
        EXPECT_EQ(dlp_thread_get_jc_ways_active(), -1);
    } else {
        // Env vars set - should use those values
        md_t env_ic = getEnvVarValue("DLP_IC_NT");
        md_t env_jc = getEnvVarValue("DLP_JC_NT");
        env_ic      = (env_ic <= 0) ? -1 : env_ic;
        env_jc      = (env_jc <= 0) ? -1 : env_jc;

        EXPECT_EQ(dlp_thread_get_ic_ways_active(), env_ic);
        EXPECT_EQ(dlp_thread_get_jc_ways_active(), env_jc);
    }
}

/**
 * Test: Mixed valid/invalid ways values
 * When one value <= 0, it's coerced to 1 if the other is valid
 */
TEST_F(ThreadingAPITest, MixedValidInvalidWays)
{
    dlp_thread_set_ways_library(4, 0);
    EXPECT_EQ(dlp_thread_get_jc_ways_active(), 4);
    EXPECT_EQ(dlp_thread_get_ic_ways_active(), 1); // Coerced to 1
}

// ============================================================================
// CATEGORY 2: PRECEDENCE ORDER TESTS
// ============================================================================

/**
 * Test: Complete precedence chain
 * Sets configurations at all levels and verifies correct priority
 */
TEST_F(ThreadingPrecedenceTest, CompletePrecedenceChain)
{
    // Set library thread count (lowest precedence)
    dlp_thread_set_num_threads_library(64);
    EXPECT_EQ(dlp_thread_get_num_threads_active(), 64);
    // Ways not set yet.
    EXPECT_EQ(dlp_thread_get_ic_ways_active(), -1);
    EXPECT_EQ(dlp_thread_get_jc_ways_active(), -1);

    // Set library ways (higher than library threads)
    dlp_thread_set_ways_library(8, 16);
    EXPECT_EQ(dlp_thread_get_ic_ways_active(), 16);
    EXPECT_EQ(dlp_thread_get_jc_ways_active(), 8);
    // num_threads should be unset by ways setter
    EXPECT_EQ(dlp_thread_get_num_threads_active(), -1);

    // Set local thread count (higher than library ways)
    dlp_thread_set_num_threads_local(32);
    EXPECT_EQ(dlp_thread_get_num_threads_active(), 32);
    // Local threads win over library ways
    EXPECT_EQ(dlp_thread_get_ic_ways_active(), -1);
    EXPECT_EQ(dlp_thread_get_jc_ways_active(), -1);

    // Set local ways (should override local threads)
    dlp_thread_set_ways_local(4, 8);
    // Ways win at same level
    EXPECT_EQ(dlp_thread_get_ic_ways_active(), 8);
    EXPECT_EQ(dlp_thread_get_jc_ways_active(), 4);
    // num_threads unset by ways setter
    EXPECT_EQ(dlp_thread_get_num_threads_active(), -1);
}

/**
 * Test: Unsetting local restores library configuration
 * Verifies fallback to lower precedence level
 */
TEST_F(ThreadingPrecedenceTest, UnsetLocalRestoresLibrary)
{
    dlp_thread_set_ways_library(4, 8);
    dlp_thread_set_ways_local(2, 16);

    // Local active
    EXPECT_EQ(dlp_thread_get_ic_ways_active(), 16);
    EXPECT_EQ(dlp_thread_get_jc_ways_active(), 2);

    dlp_thread_set_ways_local(-1, -1);

    // Library ways now visible
    EXPECT_EQ(dlp_thread_get_ic_ways_active(), 8);
    EXPECT_EQ(dlp_thread_get_jc_ways_active(), 4);
}

// ============================================================================
// CATEGORY 3: OPENMP THREAD-LOCAL TESTS
// ============================================================================

#ifdef DLP_ENABLE_OPENMP

#include <atomic>
#include <omp.h>
#include <vector>

class ThreadingOpenMPTest : public ThreadingAPITest
{
  protected:
    // Additional setup specific to OpenMP tests if needed
    void SetUp() override
    {
        ThreadingAPITest::SetUp();
        // Enable nested parallelism - atleast 2 levels if DLP APIs are called
        // inside a omp loop. Else 1 level is sufficient.
        omp_set_max_active_levels(2);
    }

    void TearDown() override
    {
        ThreadingAPITest::TearDown();
        omp_set_max_active_levels(1);
    }
};

/**
 * Test: Thread-local num_threads independence in OpenMP parallel region
 * Each OpenMP thread sets different local configuration
 * Verifies true thread-local storage with no cross-contamination
 */
TEST_F(ThreadingOpenMPTest, OMPThreadLocalNumThreadsIndependence)
{
    // Set library default
    dlp_thread_set_num_threads_library(16);

    std::vector<md_t> observed_values(3, -999);

#pragma omp parallel num_threads(3)
    {
        int tid = omp_get_thread_num();

        // Each thread sets different value
        if (tid == 0) {
            dlp_thread_set_num_threads_local(4);
        } else if (tid == 1) {
            dlp_thread_set_num_threads_local(12);
        }
        // Thread 2: Don't set - should use library default (16)

#pragma omp barrier

        // Verify each thread sees its own setting
        observed_values[tid] = dlp_thread_get_num_threads_active();

        // Reset before leaving parallel region
        dlp_thread_set_num_threads_local(-1);
    }

    // Verify thread-local independence
    EXPECT_EQ(observed_values[0], 4);  // Thread 0 set to 4
    EXPECT_EQ(observed_values[1], 12); // Thread 1 set to 12
    EXPECT_EQ(observed_values[2], 16); // Thread 2 used library default
}

/**
 * Test: Thread-local ways independence in OpenMP parallel region
 * Verifies ways-based configuration is truly per-thread
 */
TEST_F(ThreadingOpenMPTest, OMPThreadLocalWaysIndependence)
{
    std::vector<md_t> ic_values(3, -999);
    std::vector<md_t> jc_values(3, -999);

#pragma omp parallel num_threads(3)
    {
        int tid = omp_get_thread_num();

        // Each thread sets different ways
        if (tid == 0) {
            dlp_thread_set_ways_local(2, 4);
        } else if (tid == 1) {
            dlp_thread_set_ways_local(8, 16);
        }
        // Thread 2: No local setting - falls back to library/env/default

#pragma omp barrier

        ic_values[tid] = dlp_thread_get_ic_ways_active();
        jc_values[tid] = dlp_thread_get_jc_ways_active();

        // Reset before leaving
        dlp_thread_set_ways_local(-1, -1);
    }

    // Verify independence
    EXPECT_EQ(ic_values[0], 4);
    EXPECT_EQ(jc_values[0], 2);
    EXPECT_EQ(ic_values[1], 16);
    EXPECT_EQ(jc_values[1], 8);

    bool has_env_ic = isEnvVarDefined("DLP_IC_NT");
    bool has_env_jc = isEnvVarDefined("DLP_JC_NT");

    if (!has_env_ic && !has_env_jc) {
        //  Thread 2 has no local setting, if DLP env vars are defined then
        // it need not be -1.
        EXPECT_EQ(ic_values[2], -1);
        EXPECT_EQ(jc_values[2], -1);
    }
}

/**
 * Test: Thread-local settings persist for OS thread lifetime
 * Thread-local values persist across OpenMP parallel regions
 * because they're tied to OS thread, not parallel region
 */
TEST_F(ThreadingOpenMPTest, OMPThreadLocalPersistsAcrossRegions)
{
    dlp_thread_set_num_threads_library(16);

// First parallel region - set thread-local
#pragma omp parallel num_threads(2)
    {
        dlp_thread_set_num_threads_local(99);
        EXPECT_EQ(dlp_thread_get_num_threads_active(), 99);
        // Intentionally NOT resetting
    }

    // Second parallel region - can see 1 or 2 99 from previous region.
    // Again the 2 depends on whether the same OS threads are reused as
    // part of omp internal thread pool, which is implementation defined.
    std::atomic<int> saw_99_count{ 0 };

#pragma omp parallel num_threads(2)
    {
        md_t nt = dlp_thread_get_num_threads_active();
        if (nt == 99) {
            saw_99_count.fetch_add(1);
            // reset if this is the case
            dlp_thread_set_num_threads_local(-1);
        }
    }

    // Verify no thread saw the leaked value
    EXPECT_GT(saw_99_count.load(), 0);
}

/**
 * Test: Test to check how open mp control APIs affect DLP threading in the
 * absence of DLP threading specific env vars or library API calls.
 */
TEST_F(ThreadingOpenMPTest, OMPThreadControlForLibrary)
{
    bool has_env_ic = isEnvVarDefined("DLP_IC_NT");
    bool has_env_jc = isEnvVarDefined("DLP_JC_NT");
    bool has_env_nt = isEnvVarDefined("DLP_NUM_THREADS");

    if (has_env_ic || has_env_jc || has_env_nt) {
        GTEST_SKIP() << "Environment variables for DLP threading are set. "
                     << "This test verifies OpenMP control in the absence of "
                     << "DLP threading env vars, so skipping.";
    }

    omp_set_num_threads(16);

    std::vector<md_t> observed_values_omp(3, -999);
    std::vector<md_t> observed_values_dlp(3, -999);

// First parallel region - set thread-local
#pragma omp parallel num_threads(3)
    {
        int tid = omp_get_thread_num();

        if (tid == 0) {
            omp_set_num_threads(4);
        } else if (tid == 1) {
            omp_set_num_threads(12);
        }

#pragma omp barrier
        observed_values_omp[tid] = omp_get_max_threads();
        observed_values_dlp[tid] = dlp_thread_get_num_threads_active();
    }

    EXPECT_EQ(observed_values_omp[0], 4);
    EXPECT_EQ(observed_values_omp[1], 12);
    EXPECT_EQ(observed_values_omp[2], 16);
    EXPECT_EQ(observed_values_dlp[0], observed_values_omp[0]);
    EXPECT_EQ(observed_values_dlp[1], observed_values_omp[1]);
    EXPECT_EQ(observed_values_dlp[2], observed_values_omp[2]);
}

#endif // DLP_ENABLE_OPENMP
