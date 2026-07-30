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

#include <cstring>

#include "classic/dlp_base_types.h"
#include "cpu_utils/cpuid.hh"
#include "utils/type_utils.hh"
#include "x86_cpu_features.hh"

namespace dlp::cpu_utils {

enum x86_bit_positions : uint32_t
{
    // input register(s)     output register
    sse3_bit_pos               = (1u << 0),  // cpuid[eax=1]          :ecx[0]
    ssse3_bit_pos              = (1u << 9),  // cpuid[eax=1]          :ecx[9]
    sse41_bit_pos              = (1u << 19), // cpuid[eax=1]          :ecx[19]
    sse42_bit_pos              = (1u << 20), // cpuid[eax=1]          :ecx[20]
    avx_bit_pos                = (1u << 28), // cpuid[eax=1]          :ecx[28]
    avx2_bit_pos               = (1u << 5),  // cpuid[eax=7,ecx=0]    :ebx[5]
    fma3_bit_pos               = (1u << 12), // cpuid[eax=1]          :ecx[12]
    fma4_bit_pos               = (1u << 16), // cpuid[eax=0x80000001] :ecx[16]
    avx512f_bit_pos            = (1u << 16), // cpuid[eax=7,ecx=0]    :ebx[16]
    avx512dq_bit_pos           = (1u << 17), // cpuid[eax=7,ecx=0]    :ebx[17]
    avx512pf_bit_pos           = (1u << 26), // cpuid[eax=7,ecx=0]    :ebx[26]
    avx512er_bit_pos           = (1u << 27), // cpuid[eax=7,ecx=0]    :ebx[27]
    avx512cd_bit_pos           = (1u << 28), // cpuid[eax=7,ecx=0]    :ebx[28]
    avx512bw_bit_pos           = (1u << 30), // cpuid[eax=7,ecx=0]    :ebx[30]
    avx512vl_bit_pos           = (1u << 31), // cpuid[eax=7,ecx=0]    :ebx[31]
    avx512vnni_bit_pos         = (1u << 11), // cpuid[eax=7,ecx=0]    :ecx[11]
    movdiri_bit_pos            = (1u << 27), // cpuid[eax=7,ecx=0]    :ecx[27]
    movdir64b_bit_pos          = (1u << 28), // cpuid[eax=7,ecx=0]    :ecx[28]
    avx512vp2intersect_bit_pos = (1u << 8),  // cpuid[eax=7,ecx=0]    :edx[8]
    avx512fp16_bit_pos         = (1u << 23), // cpuid[eax=7,ecx=0]    :edx[23]
    avxvnni_bit_pos            = (1u << 4),  // cpuid[eax=7,ecx=1]    :eax[4]
    avx512bf16_bit_pos         = (1u << 5),  // cpuid[eax=7,ecx=1]    :eax[5]
    xgetbv_bit_pos     = (1u << 26) | (1u << 27), // cpuid[eax=1] :ecx[27:26]
    xgetbv_xmm_bit_pos = 0x02u,                   // xcr0[1]
    xgetbv_ymm_bit_pos = 0x04u,                   // xcr0[2]
    xgetbv_zmm_bit_pos = 0xe0u,                   // xcr0[7:5]
    datapath_fp128_bit_pos = (1u << 0), // cpuid[eax=0x8000001A] :eax[0]
    datapath_fp256_bit_pos = (1u << 2), // cpuid[eax=0x8000001A] :eax[2]
    datapath_fp512_bit_pos = (1u << 3)  // cpuid[eax=0x8000001A] :eax[3]
};

// CPUID leaf numbers used for deterministic cache-parameter enumeration.
// Both leaves share an identical output register bit-layout; only the leaf
// number differs between vendors.
enum x86_cache_leaf : uint32_t
{
    intel_cache_leaf = 0x00000004u, // Intel: Deterministic Cache Parameters
    amd_cache_leaf   = 0x8000001Du  // AMD (Zen+): Cache Topology Information
};

DLP_INLINE bool
dlp_cpuid_has_features(uint32_t have, uint32_t want)
{
    return (have & want) == want;
}

void
x86CpuFeatureDetector::detectx86IsaFeatures()
{
    uint32_t eax, ebx, ecx, edx;

    uint32_t cpuid_max     = __get_cpuid_max(0, 0);
    uint32_t cpuid_max_ext = __get_cpuid_max(0x80000000u, 0);

    if (cpuid_max < 1) {
        thisVendor = cpuVendor::invalid;
        return;
    }

    // The fourth '0' serves as the NULL-terminator for the vendor string.
    uint32_t vendor_string[4] = { 0, 0, 0, 0 };

    // This is actually a macro that modifies the last four operands,
    // hence why they are not passed by address.
    __cpuid(0, eax, vendor_string[0], vendor_string[2], vendor_string[1]);

    // Check extended feature bits for post-AVX2 features.
    if (cpuid_max >= 7) {
        // This is actually a macro that modifies the last four operands,
        // hence why they are not passed by address.
        __cpuid_count(7, 0, eax, ebx, ecx, edx);

        if (dlp_cpuid_has_features(ebx, avx2_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(isaFeature::avx2)] = 1;
        }
        if (dlp_cpuid_has_features(ebx, avx512f_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(isaFeature::avx512f)] =
                1;
        }
        if (dlp_cpuid_has_features(ebx, avx512dq_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(isaFeature::avx512dq)] =
                1;
        }
        if (dlp_cpuid_has_features(ebx, avx512pf_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(isaFeature::avx512pf)] =
                1;
        }
        if (dlp_cpuid_has_features(ebx, avx512er_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(isaFeature::avx512er)] =
                1;
        }
        if (dlp_cpuid_has_features(ebx, avx512cd_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(isaFeature::avx512cd)] =
                1;
        }
        if (dlp_cpuid_has_features(ebx, avx512bw_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(isaFeature::avx512bw)] =
                1;
        }
        if (dlp_cpuid_has_features(ebx, avx512vl_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(isaFeature::avx512vl)] =
                1;
        }

        if (dlp_cpuid_has_features(ecx, avx512vnni_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(
                isaFeature::avx512vnni)] = 1;
        }
        if (dlp_cpuid_has_features(ecx, movdiri_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(isaFeature::movdiri)] =
                1;
        }
        if (dlp_cpuid_has_features(ecx, movdir64b_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(isaFeature::movdir64b)] =
                1;
        }

        if (dlp_cpuid_has_features(edx, avx512vp2intersect_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(
                isaFeature::avx512vp2intersect)] = 1;
        }

        if (dlp_cpuid_has_features(edx, avx512fp16_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(
                isaFeature::avx512fp16)] = 1;
        }

        // This is actually a macro that modifies the last four operands,
        // hence why they are not passed by address.
        // This returns extended feature flags in EAX.
        // The availability of AVX512_BF16  can be found using the
        // 5th feature bit of the returned value
        __cpuid_count(7, 1, eax, ebx, ecx, edx);

        if (dlp_cpuid_has_features(eax, avxvnni_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(isaFeature::avxvnni)] =
                1;
        }
        if (dlp_cpuid_has_features(eax, avx512bf16_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(
                isaFeature::avx512bf16)] = 1;
        }
    }

    // Check extended processor info / features bits for AMD-specific features.
    if (cpuid_max_ext >= 0x80000001u) {
        // This is actually a macro that modifies the last four operands,
        // hence why they are not passed by address.
        __cpuid(0x80000001u, eax, ebx, ecx, edx);

        if (dlp_cpuid_has_features(ecx, fma4_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(isaFeature::fma4)] = 1;
        }
    }
    if (cpuid_max_ext >= 0x8000001Au) {
        // This is actually a macro that modifies the last four operands,
        // hence why they are not passed by address.
        // This returns extended feature flags in EAX.
        __cpuid(0x8000001A, eax, ebx, ecx, edx);

        if (dlp_cpuid_has_features(eax, datapath_fp128_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(
                isaFeature::datapath_fp128)] = 1;
        }
        if (dlp_cpuid_has_features(eax, datapath_fp256_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(
                isaFeature::datapath_fp256)] = 1;
        }
        if (dlp_cpuid_has_features(eax, datapath_fp512_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(
                isaFeature::datapath_fp512)] = 1;
        }
    }

    // Unconditionally check processor info / features bits.
    {
        // This is actually a macro that modifies the last four operands,
        // hence why they are not passed by address.
        __cpuid(1, eax, ebx, ecx, edx);

        // Check for SSE, AVX, and FMA3 features.
        if (dlp_cpuid_has_features(ecx, sse3_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(isaFeature::sse3)] = 1;
        }
        if (dlp_cpuid_has_features(ecx, ssse3_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(isaFeature::ssse3)] = 1;
        }
        if (dlp_cpuid_has_features(ecx, sse41_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(isaFeature::sse41)] = 1;
        }
        if (dlp_cpuid_has_features(ecx, sse42_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(isaFeature::sse42)] = 1;
        }
        if (dlp_cpuid_has_features(ecx, avx_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(isaFeature::avx)] = 1;
        }
        if (dlp_cpuid_has_features(ecx, fma3_bit_pos)) {
            featureMap[utils::getUnderlyingValueOfEnum(isaFeature::fma3)] = 1;
        }

        // Check whether the hardware supports xsave/xrestor/xsetbv/xgetbv AND
        // support for these is enabled by the OS. If so, then we proceed with
        // checking that various register-state saving features are available.
        if (dlp_cpuid_has_features(ecx, xgetbv_bit_pos)) {
            uint32_t xcr = 0;

            // Call xgetbv to get xcr0 (the extended control register) copied
            // to [edx:eax]. This encodes whether software supports various
            // register state-saving features.
#if DLP_OS_WINDOWS
            uint64_t xcr_result = xgetbv(xcr);
            eax = static_cast<uint32_t>(xcr_result & 0xFFFFFFFF);
            edx = static_cast<uint32_t>(xcr_result >> 32);
#else
            __asm__ __volatile__(".byte 0x0F, 0x01, 0xD0"
                                 : "=a"(eax), "=d"(edx)
                                 : "c"(xcr)
                                 : "cc");
#endif

            // The OS can manage the state of 512-bit zmm (AVX-512) registers
            // only if the xcr[7:5] bits are set. If they are not set, then
            // clear all feature bits related to AVX-512.
            if (!dlp_cpuid_has_features(eax, xgetbv_xmm_bit_pos
                                                 | xgetbv_ymm_bit_pos
                                                 | xgetbv_zmm_bit_pos)) {
                featureMap[utils::getUnderlyingValueOfEnum(
                    isaFeature::avx512f)]  = 0;
                featureMap[utils::getUnderlyingValueOfEnum(
                    isaFeature::avx512dq)] = 0;
                featureMap[utils::getUnderlyingValueOfEnum(
                    isaFeature::avx512cd)] = 0;
                featureMap[utils::getUnderlyingValueOfEnum(
                    isaFeature::avx512pf)] = 0;
                featureMap[utils::getUnderlyingValueOfEnum(
                    isaFeature::avx512er)] = 0;
                featureMap[utils::getUnderlyingValueOfEnum(
                    isaFeature::avx512bw)] = 0;
                featureMap[utils::getUnderlyingValueOfEnum(
                    isaFeature::avx512vl)] = 0;
            }

            // The OS can manage the state of 256-bit ymm (AVX) registers
            // only if the xcr[2] bit is set. If it is not set, then
            // clear all feature bits related to AVX.
            if (!dlp_cpuid_has_features(eax, xgetbv_xmm_bit_pos
                                                 | xgetbv_ymm_bit_pos)) {
                featureMap[utils::getUnderlyingValueOfEnum(isaFeature::avx)] =
                    0;
                featureMap[utils::getUnderlyingValueOfEnum(isaFeature::avx2)] =
                    0;
                featureMap[utils::getUnderlyingValueOfEnum(isaFeature::fma3)] =
                    0;
                featureMap[utils::getUnderlyingValueOfEnum(isaFeature::fma4)] =
                    0;
            }

            // The OS can manage the state of 128-bit xmm (SSE) registers
            // only if the xcr[1] bit is set. If it is not set, then
            // clear all feature bits related to SSE (which means the
            // entire bitfield is clear).
            if (!dlp_cpuid_has_features(eax, xgetbv_xmm_bit_pos)) {
                std::fill(featureMap.begin(), featureMap.end(), 0);
            }
        } else {
            // If the hardware does not support xsave/xrestor/xsetbv/xgetbv,
            // OR these features are not enabled by the OS, then we clear
            // the bitfield, because it means that not even xmm support is
            // present.

            std::fill(featureMap.begin(), featureMap.end(), 0);
        }
    }

    // Check the vendor string and return a value to indicate Intel or AMD.
    if (std::strcmp((char*)vendor_string, "AuthenticAMD") == 0) {
        thisVendor = cpuVendor::amd;
    } else if (std::strcmp((char*)vendor_string, "GenuineIntel") == 0) {
        thisVendor = cpuVendor::intel;
    } else {
        thisVendor = cpuVendor::invalid;
    }
}

void
x86CpuFeatureDetector::detectx86CacheInfo()
{
    // Cache topology derivation relies on knowing the vendor to pick the
    // correct (but bit-layout-identical) CPUID leaf. If the vendor could not
    // be determined, we cannot reliably enumerate caches, so bail out and
    // leave cacheHierarchy empty.
    if (thisVendor == cpuVendor::invalid) {
        return;
    }

    uint32_t cache_leaf;
    if (thisVendor == cpuVendor::amd) {
        // Ensure the AMD cache-topology leaf is available before using it.
        uint32_t cpuid_max_ext = __get_cpuid_max(0x80000000u, 0);
        if (cpuid_max_ext < amd_cache_leaf) {
            return;
        }
        cache_leaf = amd_cache_leaf;
    } else {
        // Intel (and any other vendor mapped here). Ensure leaf 0x4 exists.
        uint32_t cpuid_max = __get_cpuid_max(0, 0);
        if (cpuid_max < intel_cache_leaf) {
            return;
        }
        cache_leaf = intel_cache_leaf;
    }

    // Enumerate cache descriptors via successive sub-leaves. The same parsing
    // logic works for both vendors since the register bit-layout is identical.
    for (uint32_t sub_leaf = 0;; ++sub_leaf) {
        uint32_t eax, ebx, ecx, edx;

        // This is actually a macro that modifies the last four operands,
        // hence why they are not passed by address.
        __cpuid_count(cache_leaf, sub_leaf, eax, ebx, ecx, edx);

        // EAX[4:0] encodes the cache type. A value of 0 indicates that there
        // are no more cache descriptors to enumerate.
        uint32_t type_field = eax & 0x1Fu;
        if (type_field == 0) {
            break;
        }

        cacheInfo info;

        // EAX[7:5] encodes the cache level (1, 2, 3 ...).
        info.level = (eax >> 5) & 0x7u;

        switch (type_field) {
            case 1:
                info.type = cacheType::data;
                break;
            case 2:
                info.type = cacheType::instruction;
                break;
            case 3:
                info.type = cacheType::unified;
                break;
            default:
                continue; // Skip any other type that we do not support.
        }

        // Cache size = (ways+1) * (partitions+1) * (line_size+1) * (sets+1).
        // Each field is stored as (value - 1) in the respective register.
        //
        // This single formula is valid for every associativity type, which is
        // the whole point of the deterministic cache-parameters leaf:
        //   - Direct-mapped   : ways field is 0  => ways = 1.
        //   - Set-associative : ways and sets both > 1 (the general case).
        //   - Fully-associative: sets field (ECX) is 0 => sets = 1, and the
        //                        ways field carries the total number of lines.
        // In all three cases the product yields the correct total byte size, so
        // no associativity-specific guard is required. (EAX[9] separately flags
        // a fully-associative cache, but it is only needed to *report* the
        // associativity kind, not to compute the size.)
        //
        // All fields are widened to 64-bit before the "+ 1" so that the "sets"
        // field (a full 32-bit value from ECX) cannot overflow, and so the
        // final product is computed in 64-bit arithmetic.
        uint64_t line_size =
            static_cast<uint64_t>(ebx & 0xFFFu) + 1u; // EBX[11:0]
        uint64_t partitions =
            static_cast<uint64_t>((ebx >> 12) & 0x3FFu) + 1u; // EBX[21:12]
        uint64_t ways =
            static_cast<uint64_t>((ebx >> 22) & 0x3FFu) + 1u; // EBX[31:22]
        uint64_t sets = static_cast<uint64_t>(ecx) + 1u;      // ECX[31:0]

        info.sizeInBytes = ways * partitions * line_size * sets;

        cacheHierarchy.push_back(info);
    }
}

x86CpuFeatureDetector::x86CpuFeatureDetector()
{
    featureMap.resize(static_cast<std::size_t>(utils::getUnderlyingValueOfEnum(
                          isaFeature::feature_limit)),
                      0);

    detectx86IsaFeatures();
    detectx86CacheInfo();
}

bool
x86CpuFeatureDetector::hasFeatures(
    const std::vector<isaFeature>& featureList) const
{
    bool retVal = true;
    for (auto ele : featureList) {
        if (featureMap[utils::getUnderlyingValueOfEnum(ele)] == 0) {
            retVal = false;
            break;
        }
    }

    return retVal;
}

bool
x86CpuFeatureDetector::hasFeature(const isaFeature feature) const
{
    return (featureMap[utils::getUnderlyingValueOfEnum(feature)] == 1);
}

std::vector<isaFeature>
x86CpuFeatureDetector::getFeatures() const
{
    std::vector<isaFeature> features;
    for (uint32_t ii = utils::getUnderlyingValueOfEnum(isaFeature::invalid);
         ii < utils::getUnderlyingValueOfEnum(isaFeature::feature_limit);
         ++ii) {
        if (featureMap[ii] == 1) {
            features.push_back(
                utils::getEnumFromUnderlyingType<isaFeature, uint32_t>(ii));
        }
    }

    return features;
}

bool
x86CpuFeatureDetector::isCpuVendor(cpuVendor vendor) const
{
    return (thisVendor == vendor);
}

cpuVendor
x86CpuFeatureDetector::getCpuVendor() const
{
    return thisVendor;
}

int32_t
x86CpuFeatureDetector::getNumVectorRegisters() const
{
    // AVX-512 provides 32 vector registers, while AVX and SSE provide 16.
    if (hasFeature(isaFeature::avx512f)) {
        return 32;
    } else if (hasFeature(isaFeature::avx) || hasFeature(isaFeature::sse3)) {
        return 16;
    } else {
        return 0;
    }
}

int32_t
x86CpuFeatureDetector::getNumVectorMaskRegisters() const
{
    // AVX-512 provides 8 mask registers. However k0 cannot be used since it
    // acts as a special mask that represents no masking.
    if (hasFeature(isaFeature::avx512f)) {
        return 7;
    } else {
        // AVX and SSE provides none.
        return 0;
    }
}

int32_t
x86CpuFeatureDetector::getNumCacheLevels() const
{
    // The number of cache levels is the highest level reported across all the
    // detected cache descriptors (handles machines that lack, say, an L3).
    int32_t maxLevel = 0;
    for (const auto& info : cacheHierarchy) {
        if (static_cast<int32_t>(info.level) > maxLevel) {
            maxLevel = static_cast<int32_t>(info.level);
        }
    }

    return maxLevel;
}

cacheInfo
x86CpuFeatureDetector::getCacheInfo(int32_t level, cacheType type) const
{
    // Find the cache at 'level' that services the requested content 'type'.
    //
    // A split level (typically L1) has separate 'data' and 'instruction'
    // caches, so the caller's 'type' selects between them. A unified level
    // (typically L2 / L3) has a single cache that serves both, so a unified
    // descriptor satisfies either a 'data' or an 'instruction' request.
    //
    // If nothing matches (e.g. the level does not exist), a zero-initialized
    // cacheInfo is returned.
    cacheInfo unifiedMatch{};
    bool      haveUnified = false;

    for (const auto& info : cacheHierarchy) {
        if (static_cast<int32_t>(info.level) != level) {
            continue;
        }

        // Exact content-type match wins immediately.
        if (info.type == type) {
            return info;
        }

        // A unified cache can satisfy the request, but only as a fallback.
        // We do not return here: since the order of 'cacheHierarchy' entries is
        // not assumed, an exact-type match for this level may still appear
        // later. Record the unified cache and keep scanning so that an exact
        // match always takes priority.
        if (info.type == cacheType::unified && !haveUnified) {
            unifiedMatch = info;
            haveUnified  = true;
        }
    }

    return unifiedMatch;
}

int64_t
x86CpuFeatureDetector::getCacheSize(int32_t level, cacheType type) const
{
    return static_cast<int64_t>(getCacheInfo(level, type).sizeInBytes);
}

std::vector<cacheInfo>
x86CpuFeatureDetector::getAllCacheInfo() const
{
    return cacheHierarchy;
}

} // namespace dlp::cpu_utils
