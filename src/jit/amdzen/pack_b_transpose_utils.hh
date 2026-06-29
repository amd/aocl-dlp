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

#include "xbyak/xbyak.h"

// ───────────────────────────────────────────────────────────────────────────
// Shared in-register 16x16 32-bit-lane transpose for the column-major pack-B
// JIT generators.
//
// Both the F32 and the BF16 column-major packers read SIMD-width columns from a
// column-major source and need the same transpose to turn "16 columns of N
// 32-bit lanes" into "16 rows of 16 lanes". The transpose is pure data movement
// over 32-bit lanes, so it is dtype-agnostic:
//   * F32  : each lane is one float.
//   * BF16 : each lane is one bf16 K-pair (two consecutive bf16 packed into a
//            dword), so a 16x16 dword transpose yields the vdpbf16ps K-pair
//            layout directly.
//
// Only the loads (dtype-specific) and the stores (dtype-specific strides /
// masking) live in the individual generators; the permutation math is shared
// here so both stay in lock-step.
// ───────────────────────────────────────────────────────────────────────────

namespace amdzen::PackBcodeGenerator::transpose {

// After emitTranspose16x16(), output row r lives in Zmm(storeMap16[r]).
// (The 4-stage unpack/shuffle network leaves rows in this fixed permutation;
// callers index their stores through this map rather than copying rows back to
// a canonical order.)
inline constexpr int storeMap16[16] = { 0, 2, 8,  10, 1, 3, 9,  11,
                                        4, 6, 12, 14, 5, 7, 13, 15 };

// In-register 16x16 transpose of 32-bit lanes.
//
// Contract:
//   Input   : Zmm(0..15)  — column c (c = 0..15) holds 16 lanes to transpose.
//   Scratch : Zmm(16..31) — fully clobbered.
//   Output  : row r (r = 0..15) is left in Zmm(storeMap16[r]); the low 16 lanes
//             of that register hold column 0..15 of row r.
//
// The instructions are ps/pd-domain shuffles, but they perform no arithmetic
// (no FP exceptions / rounding), so the result is bit-exact for integer / bf16
// lane payloads as well.
inline void
emitTranspose16x16(Xbyak::CodeGenerator& cg)
{
    using Xbyak::Zmm;

    // Stage 1: vunpcklps / vunpckhps (pairs -> Zmm16..31)
    for (int i = 0; i < 16; i += 2) {
        cg.vunpcklps(Zmm(16 + i), Zmm(i), Zmm(i + 1));
        cg.vunpckhps(Zmm(16 + i + 1), Zmm(i), Zmm(i + 1));
    }

    // Stage 2: vunpcklpd / vunpckhpd (quads -> Zmm0..15)
    cg.vunpcklpd(Zmm(0), Zmm(16), Zmm(18));
    cg.vunpckhpd(Zmm(1), Zmm(16), Zmm(18));
    cg.vunpcklpd(Zmm(2), Zmm(20), Zmm(22));
    cg.vunpckhpd(Zmm(3), Zmm(20), Zmm(22));
    cg.vunpcklpd(Zmm(4), Zmm(24), Zmm(26));
    cg.vunpckhpd(Zmm(5), Zmm(24), Zmm(26));
    cg.vunpcklpd(Zmm(6), Zmm(28), Zmm(30));
    cg.vunpckhpd(Zmm(7), Zmm(28), Zmm(30));
    cg.vunpcklpd(Zmm(8), Zmm(17), Zmm(19));
    cg.vunpckhpd(Zmm(9), Zmm(17), Zmm(19));
    cg.vunpcklpd(Zmm(10), Zmm(21), Zmm(23));
    cg.vunpckhpd(Zmm(11), Zmm(21), Zmm(23));
    cg.vunpcklpd(Zmm(12), Zmm(25), Zmm(27));
    cg.vunpckhpd(Zmm(13), Zmm(25), Zmm(27));
    cg.vunpcklpd(Zmm(14), Zmm(29), Zmm(31));
    cg.vunpckhpd(Zmm(15), Zmm(29), Zmm(31));

    // Stage 3: vshuff32x4 0x44 / 0xEE (128-bit lane pairs -> Zmm16..31)
    static constexpr int s3a[8] = { 0, 4, 1, 5, 8, 12, 9, 13 };
    static constexpr int s3b[8] = { 2, 6, 3, 7, 10, 14, 11, 15 };
    for (int p = 0; p < 8; ++p) {
        cg.vshuff32x4(Zmm(16 + 2 * p), Zmm(s3a[p]), Zmm(s3b[p]), 0x44);
        cg.vshuff32x4(Zmm(16 + 2 * p + 1), Zmm(s3a[p]), Zmm(s3b[p]), 0xEE);
    }

    // Stage 4: vshuff32x4 0x88 / 0xDD (final placement -> Zmm0..15)
    static constexpr int s4a[8] = { 16, 20, 17, 21, 24, 28, 25, 29 };
    static constexpr int s4b[8] = { 18, 22, 19, 23, 26, 30, 27, 31 };
    for (int p = 0; p < 8; ++p) {
        cg.vshuff32x4(Zmm(2 * p), Zmm(s4a[p]), Zmm(s4b[p]), 0x88);
        cg.vshuff32x4(Zmm(2 * p + 1), Zmm(s4a[p]), Zmm(s4b[p]), 0xDD);
    }
}

} // namespace amdzen::PackBcodeGenerator::transpose
