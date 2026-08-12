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

#pragma once

#include <cstddef>

namespace dlp { namespace bench {

    /**
     * One-time setup for the cold-cache flush primitive.
     *
     * @param enabled  If true, allocate scratch and arm flushColdCache(). If
     *                 false, this is a no-op and flushColdCache() returns
     *                 immediately.
     * @param passes   Number of scratch read-passes per flushColdCache() call.
     *                 Default 1 is sufficient for Zen5/Zen6c LRU; raise only
     *                 for retention-resistant replacement policies.
     *
     * Windows currently does not implement cold-cache eviction. An enabled
     * request is ignored with a warning; native Windows cache/topology support
     * is tracked as a follow-up.
     *
     * When enabled, this:
     *   - reads L3 size from /sys via CpuTopology (falls back to 256 MiB),
     *   - allocates a 2x L3 scratch buffer aligned to a cache line,
     *   - first-touches it from every OMP thread so each NUMA-local slice is
     *     populated by its eventual flush worker,
     *   - prepares per-thread sinks to defeat dead-store elimination.
     *
     * Idempotent — second call returns immediately. Should be called once
     * from main() AFTER OMP initialization but BEFORE the benchmark loop
     * begins. Wire from ArgParser::getColdCache() / getColdPasses().
     */
    void initColdCache(bool enabled, int passes = 1);

    /**
     * Evict the working set from caches by sweeping a scratch buffer larger
     * than L3. Parallelized across OMP threads; each thread sweeps a disjoint
     * slice so every CCX's L3 cluster is touched. Read-only — relies on
     * cache replacement to evict prior data via LRU/RRIP. Triggers automatic
     * write-back of any dirty operand lines as a side effect.
     *
     * Cheap if cold cache is disabled (early return). Runs `passes` times
     * (set at initColdCache).
     */
    void flushColdCache();

    /// Query: was cold cache enabled at init time?
    bool coldCacheEnabled();

    /// Query: configured number of read-passes per flush call.
    int coldCachePasses();

    /// Diagnostic: L3 size detected during init, in bytes (0 if not
    /// initialised).
    std::size_t detectedL3Bytes();

}} // namespace dlp::bench
