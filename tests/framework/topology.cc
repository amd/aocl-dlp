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

#include "framework/topology.hh"

#include <cstdio>
#include <fstream>
#include <ostream>
#include <string>

namespace dlp { namespace testing { namespace framework {

    namespace {

        // Parse a sysfs "size" string of the form "32K", "1024K", "16M", "1G".
        // Returns 0 on parse failure or when the file isn't readable.
        std::size_t readSysfsSize(const char* path)
        {
            std::ifstream f(path);
            if (!f)
                return 0;
            std::string s;
            f >> s;
            if (s.empty())
                return 0;
            char        unit = s.back();
            std::size_t mul  = 1;
            if (unit == 'K' || unit == 'k') {
                mul = 1024;
                s.pop_back();
            } else if (unit == 'M' || unit == 'm') {
                mul = 1024ULL * 1024;
                s.pop_back();
            } else if (unit == 'G' || unit == 'g') {
                mul = 1024ULL * 1024 * 1024;
                s.pop_back();
            }
            try {
                return static_cast<std::size_t>(std::stoull(s)) * mul;
            } catch (...) {
                return 0;
            }
        }

        // Read a numeric field from sysfs (e.g., coherency_line_size).
        std::size_t readSysfsUInt(const char* path)
        {
            std::ifstream f(path);
            if (!f)
                return 0;
            std::size_t v = 0;
            f >> v;
            return v;
        }

        // Read a single-character file (e.g., cache 'type': "Data",
        // "Instruction", "Unified"). Returns the first character, or '\0' if
        // the file is missing.
        char readSysfsTypeFirstChar(const char* path)
        {
            std::ifstream f(path);
            if (!f)
                return '\0';
            std::string s;
            f >> s;
            return s.empty() ? '\0' : s[0];
        }

    } // namespace

    const CpuTopology& CpuTopology::get()
    {
        static const CpuTopology instance;
        return instance;
    }

    CpuTopology::CpuTopology()
    {
        // Walk /sys/devices/system/cpu/cpu0/cache/index{0..N}, dispatching by
        // {level, type} into l1d / l2 / l3 buckets. Index assignment varies
        // between kernels (some have separate index0=L1d, index1=L1i; some
        // collapse), so we trust the level+type fields rather than the index.
        constexpr int kMaxIndex = 8; // generous upper bound
        char          path[128];

        for (int i = 0; i < kMaxIndex; ++i) {
            std::snprintf(path, sizeof(path),
                          "/sys/devices/system/cpu/cpu0/cache/index%d/level",
                          i);
            std::ifstream level_f(path);
            if (!level_f)
                break; // no more cache levels enumerated

            int level = 0;
            level_f >> level;
            if (level <= 0)
                continue;

            std::snprintf(path, sizeof(path),
                          "/sys/devices/system/cpu/cpu0/cache/index%d/type", i);
            char type_first = readSysfsTypeFirstChar(path);

            std::snprintf(path, sizeof(path),
                          "/sys/devices/system/cpu/cpu0/cache/index%d/size", i);
            std::size_t size = readSysfsSize(path);

            // Cache line size is reported per index but is uniform across the
            // hierarchy on every architecture we care about. Latch the first
            // non-zero value we see.
            if (cache_line_ == 0) {
                std::snprintf(path, sizeof(path),
                              "/sys/devices/system/cpu/cpu0/cache/index%d/"
                              "coherency_line_size",
                              i);
                cache_line_ = readSysfsUInt(path);
            }

            switch (level) {
                case 1:
                    // L1d only — L1i is irrelevant for data-cache sizing in our
                    // workloads. type is "Data" ('D'), "Instruction" ('I'), or
                    // "Unified" ('U'). Treat 'D' and 'U' as data cache.
                    if ((type_first == 'D' || type_first == 'U') && l1d_ == 0)
                        l1d_ = size;
                    break;
                case 2:
                    if (l2_ == 0)
                        l2_ = size;
                    break;
                case 3:
                    if (l3_ == 0)
                        l3_ = size;
                    break;
                default:
                    // L4 (if present, e.g. eDRAM victim cache) is currently
                    // ignored. Add an l4_ field if a consumer needs it.
                    break;
            }
        }
    }

    void CpuTopology::print(std::ostream& os) const
    {
        auto fmt = [&](const char* label, std::size_t bytes) {
            os << "  " << label << ": ";
            if (bytes == 0) {
                os << "<unavailable>";
            } else if (bytes >= (1024ULL * 1024)) {
                os << (bytes / (1024 * 1024)) << " MiB";
            } else if (bytes >= 1024) {
                os << (bytes / 1024) << " KiB";
            } else {
                os << bytes << " B";
            }
            os << '\n';
        };
        os << "CpuTopology (CPU0 view):\n";
        fmt("L1d         ", l1d_);
        fmt("L2          ", l2_);
        fmt("L3          ", l3_);
        fmt("Cache line  ", cache_line_);
    }

}}} // namespace dlp::testing::framework
