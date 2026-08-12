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
#include <iosfwd>

namespace dlp { namespace testing { namespace framework {

    /**
     * @brief Snapshot of CPU topology relevant to test/benchmark code.
     *
     * Currently exposes cache geometry detected from sysfs. Designed to grow:
     * NUMA layout, CCX/cluster topology, and SMT info will be added as concrete
     * consumers materialise. Until then, this class is the canonical place for
     * any code that needs to ask "what does this machine look like?".
     *
     * Usage:
     * @code
     * auto& topo = dlp::testing::framework::CpuTopology::get();
     * std::size_t l3 = topo.l3Size();
     * @endcode
     *
     * Detection runs once on first call to get(); thread-safe under C++11 magic
     * statics. Returned values are read-only — no consumer should be able to
     * mutate the singleton's view of the world.
     */
    class CpuTopology
    {
      public:
        /**
         * @brief Lazy singleton accessor. Detection runs once on first call.
         *
         * Subsequent calls return the same instance with cached values. Safe
         * to call concurrently after C++11 magic-statics.
         */
        static const CpuTopology& get();

        /// L1 data cache size in bytes (CPU0's view). 0 if unavailable.
        std::size_t l1dSize() const { return l1d_; }

        /// L2 cache size in bytes (CPU0's view). 0 if unavailable.
        std::size_t l2Size() const { return l2_; }

        /**
         * @brief L3 cache size in bytes (CPU0's view).
         *
         * On AMD multi-CCX silicon (Zen2+, especially Zen6c) this is the size
         * of the L3 cluster shared by CPU0's CCX, NOT the socket-level
         * aggregate. Callers wanting "how much L3 must I evict to defeat
         * caching" want this value (per-cluster), not the sum across all
         * clusters.
         *
         * Returns 0 if the sysfs entry is unavailable; callers should fall back
         * to a sensible default rather than divide by zero.
         */
        std::size_t l3Size() const { return l3_; }

        /// Cache line size in bytes. 0 if unavailable.
        std::size_t cacheLineSize() const { return cache_line_; }

        /// Diagnostic dump of detected geometry, one field per line.
        void print(std::ostream& os) const;

        CpuTopology(const CpuTopology&)            = delete;
        CpuTopology& operator=(const CpuTopology&) = delete;

      private:
        CpuTopology(); // populates from /sys

        std::size_t l1d_        = 0;
        std::size_t l2_         = 0;
        std::size_t l3_         = 0;
        std::size_t cache_line_ = 0;
    };

}}} // namespace dlp::testing::framework
