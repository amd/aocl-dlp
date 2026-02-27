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

#ifndef LPGEMM_FP16_KERN_MACROS_H
#define LPGEMM_FP16_KERN_MACROS_H

#include <immintrin.h>

/**
 * @file lpgemm_fp16_kern_macros.h
 * @brief Helper macros for FP16 GEMM microkernels using AVX-512-FP16
 *
 * Key differences from BF16/F32:
 * - Each ZMM holds 32 FP16 elements (vs 16 F32 or 16 BF16)
 * - Uses _mm512_fmadd_ph (native FP16 FMA)
 * - No k-pairing (packing factor = 1)
 * - NR = 128 requires 4 ZMM registers per row
 */

/* Number of FP16 elements per ZMM register */
#define NUM_FP16_ELEMS_PER_ZMM 32

/* ==========================================================================
 * Accumulator initialization
 * ========================================================================== */

#define ZERO_ACC_ZMM_1_REG_FP16(r0) (r0) = _mm512_setzero_ph();

#define ZERO_ACC_ZMM_2_REG_FP16(r0, r1)                                        \
    (r0) = _mm512_setzero_ph();                                                \
    (r1) = _mm512_setzero_ph();

#define ZERO_ACC_ZMM_3_REG_FP16(r0, r1, r2)                                    \
    (r0) = _mm512_setzero_ph();                                                \
    (r1) = _mm512_setzero_ph();                                                \
    (r2) = _mm512_setzero_ph();

#define ZERO_ACC_ZMM_4_REG_FP16(r0, r1, r2, r3)                                \
    (r0) = _mm512_setzero_ph();                                                \
    (r1) = _mm512_setzero_ph();                                                \
    (r2) = _mm512_setzero_ph();                                                \
    (r3) = _mm512_setzero_ph();

/* ==========================================================================
 * B matrix loads (from packed K-MAJOR layout)
 * ========================================================================== */

/* Load 32 FP16 (1 ZMM) */
#define LOAD_B_1x32_FP16(b_ptr, b0) (b0) = _mm512_loadu_ph((b_ptr));

/* Load 64 FP16 (2 ZMM) */
#define LOAD_B_1x64_FP16(b_ptr, b0, b1)                                        \
    (b0) = _mm512_loadu_ph((b_ptr));                                           \
    (b1) = _mm512_loadu_ph((b_ptr) + 32);

/* Load 96 FP16 (3 ZMM) */
#define LOAD_B_1x96_FP16(b_ptr, b0, b1, b2)                                    \
    (b0) = _mm512_loadu_ph((b_ptr));                                           \
    (b1) = _mm512_loadu_ph((b_ptr) + 32);                                      \
    (b2) = _mm512_loadu_ph((b_ptr) + 64);

/* Load 128 FP16 (4 ZMM) */
#define LOAD_B_1x128_FP16(b_ptr, b0, b1, b2, b3)                               \
    (b0) = _mm512_loadu_ph((b_ptr));                                           \
    (b1) = _mm512_loadu_ph((b_ptr) + 32);                                      \
    (b2) = _mm512_loadu_ph((b_ptr) + 64);                                      \
    (b3) = _mm512_loadu_ph((b_ptr) + 96);

/* ==========================================================================
 * FMA operations: c += a_bcast * b
 * ========================================================================== */

#define FMA_1x32_FP16(a_bcast, b0, c0)                                         \
    (c0) = _mm512_fmadd_ph((a_bcast), (b0), (c0));

#define FMA_1x64_FP16(a_bcast, b0, b1, c0, c1)                                 \
    (c0) = _mm512_fmadd_ph((a_bcast), (b0), (c0));                             \
    (c1) = _mm512_fmadd_ph((a_bcast), (b1), (c1));

#define FMA_1x96_FP16(a_bcast, b0, b1, b2, c0, c1, c2)                         \
    (c0) = _mm512_fmadd_ph((a_bcast), (b0), (c0));                             \
    (c1) = _mm512_fmadd_ph((a_bcast), (b1), (c1));                             \
    (c2) = _mm512_fmadd_ph((a_bcast), (b2), (c2));

#define FMA_1x128_FP16(a_bcast, b0, b1, b2, b3, c0, c1, c2, c3)                \
    (c0) = _mm512_fmadd_ph((a_bcast), (b0), (c0));                             \
    (c1) = _mm512_fmadd_ph((a_bcast), (b1), (c1));                             \
    (c2) = _mm512_fmadd_ph((a_bcast), (b2), (c2));                             \
    (c3) = _mm512_fmadd_ph((a_bcast), (b3), (c3));

/* ==========================================================================
 * Alpha scaling
 * ========================================================================== */

#define SCALE_1x32_FP16(alpha_v, c0) (c0) = _mm512_mul_ph((c0), (alpha_v));

#define SCALE_1x64_FP16(alpha_v, c0, c1)                                       \
    (c0) = _mm512_mul_ph((c0), (alpha_v));                                     \
    (c1) = _mm512_mul_ph((c1), (alpha_v));

#define SCALE_1x96_FP16(alpha_v, c0, c1, c2)                                   \
    (c0) = _mm512_mul_ph((c0), (alpha_v));                                     \
    (c1) = _mm512_mul_ph((c1), (alpha_v));                                     \
    (c2) = _mm512_mul_ph((c2), (alpha_v));

#define SCALE_1x128_FP16(alpha_v, c0, c1, c2, c3)                              \
    (c0) = _mm512_mul_ph((c0), (alpha_v));                                     \
    (c1) = _mm512_mul_ph((c1), (alpha_v));                                     \
    (c2) = _mm512_mul_ph((c2), (alpha_v));                                     \
    (c3) = _mm512_mul_ph((c3), (alpha_v));

/* ==========================================================================
 * Beta * C accumulation (c_acc += beta * load(C))
 * ========================================================================== */

#define BETA_FMA_1x32_FP16(beta_v, c_ptr, c0)                                  \
    (c0) = _mm512_fmadd_ph((beta_v), _mm512_loadu_ph((c_ptr)), (c0));

#define BETA_FMA_1x64_FP16(beta_v, c_ptr, c0, c1)                              \
    (c0) = _mm512_fmadd_ph((beta_v), _mm512_loadu_ph((c_ptr)), (c0));          \
    (c1) = _mm512_fmadd_ph((beta_v), _mm512_loadu_ph((c_ptr) + 32), (c1));

#define BETA_FMA_1x96_FP16(beta_v, c_ptr, c0, c1, c2)                          \
    (c0) = _mm512_fmadd_ph((beta_v), _mm512_loadu_ph((c_ptr)), (c0));          \
    (c1) = _mm512_fmadd_ph((beta_v), _mm512_loadu_ph((c_ptr) + 32), (c1));     \
    (c2) = _mm512_fmadd_ph((beta_v), _mm512_loadu_ph((c_ptr) + 64), (c2));

#define BETA_FMA_1x128_FP16(beta_v, c_ptr, c0, c1, c2, c3)                     \
    (c0) = _mm512_fmadd_ph((beta_v), _mm512_loadu_ph((c_ptr)), (c0));          \
    (c1) = _mm512_fmadd_ph((beta_v), _mm512_loadu_ph((c_ptr) + 32), (c1));     \
    (c2) = _mm512_fmadd_ph((beta_v), _mm512_loadu_ph((c_ptr) + 64), (c2));     \
    (c3) = _mm512_fmadd_ph((beta_v), _mm512_loadu_ph((c_ptr) + 96), (c3));

/* ==========================================================================
 * Store to C
 * ========================================================================== */

#define STORE_C_1x32_FP16(c_ptr, c0) _mm512_storeu_ph((c_ptr), (c0));

#define STORE_C_1x64_FP16(c_ptr, c0, c1)                                       \
    _mm512_storeu_ph((c_ptr), (c0));                                           \
    _mm512_storeu_ph((c_ptr) + 32, (c1));

#define STORE_C_1x96_FP16(c_ptr, c0, c1, c2)                                   \
    _mm512_storeu_ph((c_ptr), (c0));                                           \
    _mm512_storeu_ph((c_ptr) + 32, (c1));                                      \
    _mm512_storeu_ph((c_ptr) + 64, (c2));

#define STORE_C_1x128_FP16(c_ptr, c0, c1, c2, c3)                              \
    _mm512_storeu_ph((c_ptr), (c0));                                           \
    _mm512_storeu_ph((c_ptr) + 32, (c1));                                      \
    _mm512_storeu_ph((c_ptr) + 64, (c2));                                      \
    _mm512_storeu_ph((c_ptr) + 96, (c3));

/* ==========================================================================
 * Masked operations for N < 32 fringe handling
 * ========================================================================== */

/* Masked load */
#define MASKZ_LOAD_FP16(mask, ptr) _mm512_maskz_loadu_epi16((mask), (ptr))

/* Masked store */
#define MASK_STORE_FP16(ptr, mask, val)                                        \
    _mm512_mask_storeu_epi16((ptr), (mask), (__m512i)(val))

/* Masked FMA: c += a * b (only for masked elements) */
#define MASK_FMA_FP16(mask, a_bcast, b_masked, c_masked)                       \
    (c_masked) = _mm512_mask_fmadd_ph((c_masked), (mask), (a_bcast), (b_masked))

#endif /* LPGEMM_FP16_KERN_MACROS_H */
