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

#include "cold_cache.hh"

#include "aocl_dlp_config.h"
#include "classic/dlp_compat.h"
#include "framework/topology.hh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dlp { namespace bench {

    namespace {

        constexpr std::size_t kCacheLineBytes = 64;
        constexpr std::size_t kFallbackL3     = 256ULL * 1024 * 1024; // 256 MiB

        // Padded so per-thread sinks live on distinct cache lines (no false
        // sharing).
        struct alignas(64) Sink
        {
            volatile std::uint64_t v;
            char                   pad[64 - sizeof(std::uint64_t)];
        };

        bool              g_initialized   = false;
        bool              g_enabled       = false;
        char*             g_scratch       = nullptr;
        std::size_t       g_scratch_bytes = 0;
        std::size_t       g_l3_bytes      = 0;
        int               g_passes        = 1;
        std::vector<Sink> g_sinks;

        struct ScratchCleanup
        {
            ~ScratchCleanup()
            {
                if (g_scratch != nullptr)
                    dlp_aligned_free(g_scratch);
            }
        };

        // Keep the process-lifetime scratch allocation owned even though the
        // benchmark API itself needs no explicit shutdown step.
        ScratchCleanup g_scratch_cleanup;

    } // namespace

    bool coldCacheEnabled()
    {
        return g_enabled;
    }

    int coldCachePasses()
    {
        return g_passes;
    }

    std::size_t detectedL3Bytes()
    {
        return g_l3_bytes;
    }

    void initColdCache(bool enabled, int passes)
    {
        if (g_initialized)
            return;

        g_enabled = enabled;
        g_passes  = passes > 0 ? passes : 1;

#if DLP_OS_WINDOWS
        if (g_enabled) {
            std::cerr
                << "[cold_cache] WARN: cold-cache mode is not implemented "
                   "on Windows; ignoring --cold\n";
        }
        // FIXME(Windows): implement cache eviction with native Windows cache
        // and topology APIs instead of the Linux sysfs-based implementation.
        g_enabled     = false;
        g_initialized = true;
        return;
#endif

        if (!g_enabled) {
            // Don't allocate the scratch buffer if the user didn't ask for cold
            // cache. flushColdCache() will be a no-op.
            g_initialized = true;
            return;
        }

        g_l3_bytes = dlp::testing::framework::CpuTopology::get().l3Size();
        if (g_l3_bytes == 0) {
            std::cerr
                << "[cold_cache] WARN: could not read L3 size from sysfs, "
                   "falling back to "
                << (kFallbackL3 / (1024 * 1024)) << " MiB assumption\n";
            g_l3_bytes = kFallbackL3;
        }

        // Scratch sized 2x L3 to defeat any LRU/RRIP retention that might keep
        // a small fraction of operand lines warm across a single sweep. Round
        // up to a cache-line boundary.
        // FIXME: Eviction is not guaranteed on multi-L3-domain systems because
        // the scratch buffer is sized to 2x CPU0's L3 and partitioned across
        // all OpenMP workers; each domain may sweep less than its own L3
        // capacity. Make scratch sizing and worker partitioning domain-aware.
        g_scratch_bytes = 2 * g_l3_bytes;
        g_scratch_bytes =
            (g_scratch_bytes + kCacheLineBytes - 1) & ~(kCacheLineBytes - 1);

        g_scratch = static_cast<char*>(
            dlp_aligned_alloc(kCacheLineBytes, g_scratch_bytes));
        if (!g_scratch) {
            std::cerr << "[cold_cache] ERROR: failed to allocate "
                      << (g_scratch_bytes / (1024 * 1024))
                      << " MiB scratch buffer; cold cache disabled\n";
            g_scratch_bytes = 0;
            g_enabled       = false;
            g_initialized   = true;
            return;
        }

        int nt = 1;
#ifdef _OPENMP
        nt = omp_get_max_threads();
#endif
        g_sinks.assign(static_cast<std::size_t>(nt), Sink{});

        // Populate via a parallel sweep so each NUMA-local page is
        // first-touched on the thread that will sweep it during flush. Under
        // `numactl --cpunodebind=N --membind=N` this is moot (all pages live on
        // node N), but it doesn't hurt and is correct under interleaved
        // policies.
#ifdef _OPENMP
#pragma omp parallel
        {
            int         tid      = omp_get_thread_num();
            int         nt_inner = omp_get_num_threads();
            std::size_t chunk    = (g_scratch_bytes + nt_inner - 1) / nt_inner;
            std::size_t start    = static_cast<std::size_t>(tid) * chunk;
            std::size_t end      = std::min(start + chunk, g_scratch_bytes);
            if (start < end)
                std::memset(g_scratch + start, 1, end - start);
        }
#else
        std::memset(g_scratch, 1, g_scratch_bytes);
#endif

        std::cerr << "================================================"
                  << std::endl;
        std::cerr << "Cold cache: ENABLED" << std::endl;
        std::cerr << "  L3 detected: " << (g_l3_bytes / (1024 * 1024)) << " MiB"
                  << std::endl;
        std::cerr << "  Scratch:     " << (g_scratch_bytes / (1024 * 1024))
                  << " MiB" << std::endl;
        std::cerr << "  Passes/iter: " << g_passes << std::endl;
        std::cerr << "  Threads:     " << nt << std::endl;
        std::cerr << "================================================"
                  << std::endl;

        g_initialized = true;
    }

    void flushColdCache()
    {
        if (!g_enabled || g_scratch == nullptr || g_scratch_bytes == 0)
            return;

        for (int p = 0; p < g_passes; ++p) {
#ifdef _OPENMP
#pragma omp parallel
            {
                int         tid      = omp_get_thread_num();
                int         nt_inner = omp_get_num_threads();
                std::size_t chunk = (g_scratch_bytes + nt_inner - 1) / nt_inner;
                std::size_t start = static_cast<std::size_t>(tid) * chunk;
                std::size_t end   = std::min(start + chunk, g_scratch_bytes);
                std::uint64_t sum = 0;
                for (std::size_t i = start; i < end; i += kCacheLineBytes) {
                    sum += static_cast<unsigned char>(g_scratch[i]);
                }
                // Bounds check defensively in case OMP_NUM_THREADS changed
                // between init and flush (unusual but possible).
                if (tid < static_cast<int>(g_sinks.size()))
                    g_sinks[static_cast<std::size_t>(tid)].v = sum;
                else {
                    volatile std::uint64_t local = sum;
                    (void)local;
                }
            }
#else
            std::uint64_t sum = 0;
            for (std::size_t i = 0; i < g_scratch_bytes; i += kCacheLineBytes) {
                sum += static_cast<unsigned char>(g_scratch[i]);
            }
            if (!g_sinks.empty())
                g_sinks[0].v = sum;
#endif
        }
    }

}} // namespace dlp::bench
