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

#include "arch_utils/arch_config_manager.hh"
#include "classic/dlp_base_types.h"

namespace dlp::de::alias_detection {

// L1D way size on AMD Zen4 and Zen5: 64 sets * 64 B/line = 4096 B.
// When A's row stride (in bytes) is at (or within a few bytes of) a
// multiple of this size AND the kernel's unrolled row count exceeds
// the L1D set associativity for the configured architecture, every
// MR row of A maps to the same L1 cache set and triggers a conflict
// miss on each k-iteration.
inline constexpr md_t L1_WAY_SIZE_BYTES = 4096;

// Guard window around a multiple of the L1 way size for which we
// consider rsA "alias-prone". Set tightly to one quarter of a cache
// line (16 B) so that natural alignment offsets (e.g. lda + 16 floats
// = lda + 64 B) fall safely outside.
//
// The window is two-sided -- `r = rsA mod 4096 < 16` (just above a
// multiple) OR `r > 4080` (just below the next multiple) -- because
// aliasing depends on |distance to the nearest multiple of 4096|, not
// on which side of it the stride falls. The L1 set index is derived
// from bits [6:11] of the address, so two rows alias when their byte
// offsets differ by ~0 modulo 4096.
//
// Why "near multiples" must be caught, with an example:
//   * Exact multiple (r = 0), MR=16: row i sits at i*rsA. Each step is
//     a whole number of ways, so all 16 rows map to the same L1 set ->
//     full aliasing.
//   * Just above (r = 4, F32 lda where lda*4 mod 4096 = 4): row i is
//     offset i*4 bytes within the set; floor(15*4/64) = 0, so all 16
//     rows still land in one set.
//   * Just below (r = 4092, i.e. 4 bytes short of 4096): each row
//     wraps to -4 bytes relative to the previous multiple, so the 16
//     rows again collapse onto ~1 set -- identical aliasing to r = 4.
// For 16 <= r <= 4080 the rows scatter across at least 4 sets and stay
// within the associativity budget, so they are not flagged.
inline constexpr md_t ALIAS_GUARD_BYTES = 16;

// Returns the L1D set associativity (number of ways) for the
// configured AMD Zen architecture. The aliasing mitigation is only
// useful when the unrolled row count (MR) exceeds the associativity.
//
//   Zen / Zen2 / Zen3 / Zen4 (Genoa) : 32KB,  8-way set associative L1D
//   Zen5 (Turin)                     : 48KB, 12-way set associative L1D
//   Zen6 (and later, assumed)        : 12-way (conservative)
//   Generic / non-Zen                :  8-way (conservative)
inline int
getL1DAssociativityForCurrentArch()
{
    const auto& mgr = dlp::arch_utils::archConfigManager::getInstance();
    if (mgr.isZen6SimilarConfiguredArch()
        || mgr.isZen5SimilarConfiguredArch()) {
        return 12;
    }
    return 8;
}

// Returns true if the kernel selected by the DE would suffer L1
// conflict misses on every k-iteration, given the row stride
// `rs_a_elems` (in elements, not bytes), per-element size
// `elem_bytes`, and unrolled row count `mr`.
//
// Pure function of the inputs and the running arch. Safe to call from
// the DE hot path.
inline bool
shouldUseMrSplit(md_t rs_a_elems, md_t elem_bytes, md_t mr)
{
    if (mr <= getL1DAssociativityForCurrentArch()) {
        return false;
    }
    const md_t rs_a_bytes = rs_a_elems * elem_bytes;
    const md_t r          = rs_a_bytes % L1_WAY_SIZE_BYTES;
    return (r < ALIAS_GUARD_BYTES)
           || (r > (L1_WAY_SIZE_BYTES - ALIAS_GUARD_BYTES));
}

// Returns an MR cap that keeps an unpacked-A GEMM kernel safely below
// the L1D associativity threshold for the running arch. Used by the
// GEMM skinnyN path (which bumps mr to 16) to fall back to an
// alias-safe MR when DE detects an aliasing-prone `lda`. Capping at
// 8 keeps the kernel within reach on every Zen variant we ship to
// (Zen-family L1Ds are >= 8-way).
inline md_t
getAliasSafeMrCap()
{
    return getL1DAssociativityForCurrentArch();
}

} // namespace dlp::de::alias_detection
