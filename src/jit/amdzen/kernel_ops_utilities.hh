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

#include <cstdint>
#include <limits>

namespace amdzen::gen {

// ═══════════════════════════════════════════════════════════════════════════
// Lookup Tables for Post-Ops (embedded into JIT buffer as needed)
// ═══════════════════════════════════════════════════════════════════════════

namespace tables {

    // GELU constants
    inline constexpr float gelu_consts[8] = {
        0.044715f, 0.797884f, -2.0f, 0.5f, -1.0f, 2.0f, 1.0f, -0.0f
    };
    inline constexpr float gelu_macros[6] = {
        1.4426950408889634f,
        1.2582912E7f,
        -88.0f,
        88.0f,
        std::numeric_limits<float>::infinity(),
        -2147483648.0f
    };

    // EXP constants
    inline constexpr float dlp_gemm_exp[6] = {
        1.0000000754895704f,  0.6931472254087585f,   0.2402210737432219f,
        0.05550297297702539f, 0.009676036358193323f, 0.001341000536524434f
    };

    // ERF constants
    inline constexpr float  erf_consts[5]    = { 0.70710678118654f, 1.0f, 0.5f,
                                                 3.553f, 3.91920638084411621f };
    inline constexpr double dlp_gemm_erf[16] = {
        0x1.20dd7890d27e1cec99fce48c29cp0,
        -0x1.ab4bed70f238422edeeba9c558p-16,
        -0x1.80a1bd5878e0b0689c5ff4fcdd4p-2,
        -0x1.07cb4cde6a7d9528c8a732990e4p-8,
        0x1.092cba598f96f00ddc5854cf7cp-3,
        -0x1.51f0ce4ac87c55f11f685864714p-5,
        0x1.4101f320bf8bc4d41c228faaa6cp-5,
        -0x1.2300882a7d1b712726997de80ep-4,
        0x1.d45745fff0e4b6d0604a9ab6284p-5,
        -0x1.9eb1491956e31ded96176d7c8acp-6,
        0x1.b9183fc75d326b9044bc63c9694p-8,
        -0x1.10e8f8c89ad8645e7d769cd596cp-10,
        0x1.224ffc80cc19957a48ecedad6c8p-14,
        0x1.12a30f42c71308321e7e7cb0174p-18,
        -0x1.155445e2e006723066d72d22ddcp-20,
        0x1.c6a4181da4ef76f22bd39bb5dcp-25
    };

    // ERF coefficients for AVX-512 piecewise polynomial (6 degrees × 32
    // regions)
    inline constexpr uint32_t erf_f32_coeffs_hex[6][32] = {
        { // Degree 0
          0x31919200, 0x32807195, 0x3382b14e, 0x34505ab6, 0x350c5db6,
          0x35f2ff5c, 0x36f037e3, 0x37b9c4fc, 0x3870c660, 0x3942160a,
          0x3a2ab0e1, 0x3ae3c00e, 0x3b757566, 0x3c0825ed, 0x3c510b4f,
          0xbb54959c, 0xbd8aa4c0, 0xbe8a61b3, 0xbf0bf55a, 0xbecccf5d,
          0x3e0cff67, 0x3f461d8a, 0x3f7d51a9, 0x3f7ff731, 0x00000000,
          0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
          0x00000000, 0x00000000 },
        { // Degree 1
          0x3f4c4226, 0x3f4c421f, 0x3f4c4207, 0x3f4c41cc, 0x3f4c414e,
          0x3f4c3fae, 0x3f4c3a22, 0x3f4c2d24, 0x3f4c12e0, 0x3f4bc213,
          0x3f4acfdf, 0x3f48f77d, 0x3f46093b, 0x3f405752, 0x3f3b9cdd,
          0x3f48ee6b, 0x3f77b93f, 0x3fbad86d, 0x40018bfc, 0x3fe53efe,
          0x3f82f0de, 0x3e70a95c, 0x3c15f930, 0x38d32404, 0x00000000,
          0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
          0x00000000, 0x00000000 },
        { // Degree 2
          0x3691b95d, 0x3733f15f, 0x37f43886, 0x388b4cd9, 0x390cecf5,
          0x39ab0a08, 0x3a622cc3, 0x3afad85a, 0x3b74f8ce, 0x3c0b61be,
          0x3ca5eef7, 0x3d216fc4, 0x3d86408c, 0x3ddf37e7, 0x3e0f0b63,
          0x3d936854, 0xbe0ad73d, 0xbf1d4b56, 0xbf89c432, 0xbf6d9f4a,
          0xbefa57fe, 0xbdc89d25, 0xbb51e003, 0xb7fd21d0, 0x00000000,
          0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
          0x00000000, 0x00000000 },
        { // Degree 3
          0xbe083848, 0xbe084539, 0xbe0863d5, 0xbe0897b7, 0xbe08e8dd,
          0xbe09ab6f, 0xbe0b6a61, 0xbe0e45ae, 0xbe128735, 0xbe1bfd38,
          0xbe2f2738, 0xbe494387, 0xbe67db90, 0xbe89ae93, 0xbe96c166,
          0xbe802bd6, 0xbe0787ed, 0x3dcee55b, 0x3e94a722, 0x3e79387e,
          0x3df0e54b, 0x3ca7993e, 0x3a12efe6, 0x3697c013, 0x00000000,
          0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
          0x00000000, 0x00000000 },
        { // Degree 4
          0x398accca, 0x39ef1989, 0x3a58f003, 0x3ab1623b, 0x3b06f06d,
          0x3b653277, 0x3bcb8f35, 0x3c22886d, 0x3c703256, 0x3cc18958,
          0x3d1df607, 0x3d639359, 0x3d94cbcb, 0x3dbf6bcf, 0x3dd53028,
          0x3db7bc9d, 0x3d65fb7c, 0xba571897, 0xbd228f61, 0xbd041cef,
          0xbc6932ba, 0xbb0c49ca, 0xb84dd98d, 0xb4b5f4b3, 0x00000000,
          0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
          0x00000000, 0x00000000 },
        { // Degree 5
          0x3c9dd788, 0x3c9b68a5, 0x3c9786ce, 0x3c92f153, 0x3c8daa4e,
          0x3c848448, 0x3c6ca387, 0x3c4c4966, 0x3c28d0bd, 0x3bdf6be2,
          0x3b058174, 0xbb23ec7e, 0xbbd24ce4, 0xbc2c1681, 0xbc491abb,
          0xbc2a591c, 0xbbd760ce, 0xba832610, 0x3b0ff520, 0x3ae25578,
          0x3a35958a, 0x38bc345e, 0x35e6ce12, 0x322e8b39, 0x00000000,
          0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
          0x00000000, 0x00000000 }
    };

    // ERF constants for F32 AVX-512 implementation
    inline constexpr uint32_t erf_f32_constants_hex[4] = {
        0xc21fffff, // erf_idx_bias
        0x7fffffff, // abs_mask
        0x80000000, // sign_mask
        0x40e00000  // rbound (7.0f)
    };

} // namespace tables

} // namespace amdzen::gen
