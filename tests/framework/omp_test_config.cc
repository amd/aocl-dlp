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

// ============================================================================
// OpenMP Wait Policy Configuration for Test Executables
// ============================================================================
//
// LLVM-based compilers (Clang, AOCC) ship with libomp, which by default uses
// active spin-waiting: OpenMP worker threads busy-spin after completing a
// parallel region, consuming CPU cycles while waiting for the next region.
// GCC's libgomp uses passive waiting (threads sleep) by default.
//
// In a test suite that executes thousands of short GEMM operations, this
// spin-waiting causes ~10x higher CPU time with AOCC/Clang compared to GCC,
// with no improvement in wall-clock time.
//
// Setting OMP_WAIT_POLICY=passive makes threads sleep between parallel
// regions, eliminating the wasted CPU cycles. The overwrite=0 flag ensures
// this default does not override an explicit user setting.
// ============================================================================

#include "framework/utils/process_env.hh"

namespace dlp::testing { namespace {

    struct OpenMPTestConfig
    {
        OpenMPTestConfig()
        {
            dlp::testing::utils::setEnvironmentVariable("OMP_WAIT_POLICY",
                                                        "passive", false);
        }
    } omp_test_config_instance;

}} // namespace dlp::testing
