
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
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
#include "kernels/dlp_kernels.h"
#include "kernels/s8s8s32/lpgemm_quanta_s8.h"

// Load FP32 value
#define LOAD_F32(reg, in) reg = _mm512_loadu_ps((const float*)(in));
// Load FP32 with mask (for tail handling)
#define LOAD_MASKED_F32(reg, mask, in)                                         \
    reg = _mm512_maskz_loadu_ps(mask, (const float*)(in));

#define LOAD_F32_ROW_OP(i, dest, src, ic, kr, rs, cs)                          \
    LOAD_F32(dest[i], src + ((ic + i) * rs + kr * cs))

#define LOAD_F32_MASKED_ROW_OP(i, dest, mask, src, ic, kr, rs, cs)             \
    LOAD_MASKED_F32(dest[i], mask, src + ((ic + i) * rs + kr * cs))

#define LOAD_F32_COL_OP(i, dest, src, ic, kr, rs, cs)                          \
    LOAD_F32(dest[i], src + (ic * rs + (kr + i) * cs))

#define LOAD_F32_MASKED_COL_OP(i, dest, mask, src, ic, kr, rs, cs)             \
    LOAD_MASKED_F32(dest[i], mask, src + (ic * rs + (kr + i) * cs))

// --- FP32 Matrix Loads (Unmasked) ---
// Loads N rows from FP32 matrix
#define LOAD_2_F32_ROW(idx, dest, src, ic, kr, rs, cs)                         \
    LOAD_F32_ROW_OP(idx, dest, src, ic, kr, rs, cs)                            \
    LOAD_F32_ROW_OP(idx + 1, dest, src, ic, kr, rs, cs)
#define LOAD_4_F32_ROW(idx, dest, src, ic, kr, rs, cs)                         \
    LOAD_2_F32_ROW(idx, dest, src, ic, kr, rs, cs)                             \
    LOAD_2_F32_ROW(idx + 2, dest, src, ic, kr, rs, cs)
#define LOAD_8_F32_ROW(idx, dest, src, ic, kr, rs, cs)                         \
    LOAD_4_F32_ROW(idx, dest, src, ic, kr, rs, cs)                             \
    LOAD_4_F32_ROW(idx + 4, dest, src, ic, kr, rs, cs)
#define LOAD_16_F32_ROW(idx, dest, src, ic, kr, rs, cs)                        \
    LOAD_8_F32_ROW(idx, dest, src, ic, kr, rs, cs)                             \
    LOAD_8_F32_ROW(idx + 8, dest, src, ic, kr, rs, cs)

#define LOAD_2_F32_COL(idx, dest, src, ic, kr, rs, cs)                         \
    LOAD_F32_COL_OP(idx, dest, src, ic, kr, rs, cs)                            \
    LOAD_F32_COL_OP(idx + 1, dest, src, ic, kr, rs, cs)
#define LOAD_4_F32_COL(idx, dest, src, ic, kr, rs, cs)                         \
    LOAD_2_F32_COL(idx, dest, src, ic, kr, rs, cs)                             \
    LOAD_2_F32_COL(idx + 2, dest, src, ic, kr, rs, cs)
#define LOAD_8_F32_COL(idx, dest, src, ic, kr, rs, cs)                         \
    LOAD_4_F32_COL(idx, dest, src, ic, kr, rs, cs)                             \
    LOAD_4_F32_COL(idx + 4, dest, src, ic, kr, rs, cs)
#define LOAD_16_F32_COL(idx, dest, src, ic, kr, rs, cs)                        \
    LOAD_8_F32_COL(idx, dest, src, ic, kr, rs, cs)                             \
    LOAD_8_F32_COL(idx + 8, dest, src, ic, kr, rs, cs)

// --- FP32 Matrix Loads (Masked) ---
// Masked loads for handling tail cases (KC % 16 != 0)
#define LOAD_2_F32_MASKED_ROW(idx, dest, mask, src, ic, kr, rs, cs)            \
    LOAD_F32_MASKED_ROW_OP(idx, dest, mask, src, ic, kr, rs, cs)               \
    LOAD_F32_MASKED_ROW_OP(idx + 1, dest, mask, src, ic, kr, rs, cs)
#define LOAD_4_F32_MASKED_ROW(idx, dest, mask, src, ic, kr, rs, cs)            \
    LOAD_2_F32_MASKED_ROW(idx, dest, mask, src, ic, kr, rs, cs)                \
    LOAD_2_F32_MASKED_ROW(idx + 2, dest, mask, src, ic, kr, rs, cs)

#define LOAD_8_F32_MASKED_ROW(idx, dest, mask, src, ic, kr, rs, cs)            \
    LOAD_4_F32_MASKED_ROW(idx, dest, mask, src, ic, kr, rs, cs)                \
    LOAD_4_F32_MASKED_ROW(idx + 4, dest, mask, src, ic, kr, rs, cs)
#define LOAD_16_F32_MASKED_ROW(idx, dest, mask, src, ic, kr, rs, cs)           \
    LOAD_8_F32_MASKED_ROW(idx, dest, mask, src, ic, kr, rs, cs)                \
    LOAD_8_F32_MASKED_ROW(idx + 8, dest, mask, src, ic, kr, rs, cs)

#define LOAD_2_F32_MASKED_COL(idx, dest, mask, src, ic, kr, rs, cs)            \
    LOAD_F32_MASKED_COL_OP(idx, dest, mask, src, ic, kr, rs, cs)               \
    LOAD_F32_MASKED_COL_OP(idx + 1, dest, mask, src, ic, kr, rs, cs)
#define LOAD_4_F32_MASKED_COL(idx, dest, mask, src, ic, kr, rs, cs)            \
    LOAD_2_F32_MASKED_COL(idx, dest, mask, src, ic, kr, rs, cs)                \
    LOAD_2_F32_MASKED_COL(idx + 2, dest, mask, src, ic, kr, rs, cs)
#define LOAD_8_F32_MASKED_COL(idx, dest, mask, src, ic, kr, rs, cs)            \
    LOAD_4_F32_MASKED_COL(idx, dest, mask, src, ic, kr, rs, cs)                \
    LOAD_4_F32_MASKED_COL(idx + 4, dest, mask, src, ic, kr, rs, cs)
#define LOAD_16_F32_MASKED_COL(idx, dest, mask, src, ic, kr, rs, cs)           \
    LOAD_8_F32_MASKED_COL(idx, dest, mask, src, ic, kr, rs, cs)                \
    LOAD_8_F32_MASKED_COL(idx + 8, dest, mask, src, ic, kr, rs, cs)

void
quant_a_sym_f32s8_row_major(int8_t*        quant_a_buffer,
                            const float*   a,
                            const md_t     rs_a,
                            const md_t     cs_a,
                            const md_t     MC,
                            const md_t     KC,
                            const void*    scale_factor,
                            const DLP_TYPE sf_type,
                            md_t           sf_len,
                            const md_t     ic_offset);

void
quant_a_asym_f32s8_row_major(int8_t*        quant_a_buffer,
                             const float*   a,
                             const md_t     rs_a,
                             const md_t     cs_a,
                             const md_t     MC,
                             const md_t     KC,
                             const void*    scale_factor,
                             const DLP_TYPE sf_type,
                             md_t           sf_len,
                             const void*    zero_point,
                             const DLP_TYPE zp_type,
                             md_t           zp_len,
                             const md_t     ic_offset);

void
quant_a_sym_f32s8_col_major(int8_t*        quant_a_buffer,
                            const float*   a,
                            const md_t     rs_a,
                            const md_t     cs_a,
                            const md_t     MC,
                            const md_t     KC,
                            const void*    scale_factor,
                            const DLP_TYPE sf_type,
                            md_t           sf_len,
                            const md_t     ic_offset);

void
quant_a_asym_f32s8_col_major(int8_t*        quant_a_buffer,
                             const float*   a,
                             const md_t     rs_a,
                             const md_t     cs_a,
                             const md_t     MC,
                             const md_t     KC,
                             const void*    scale_factor,
                             const DLP_TYPE sf_type,
                             md_t           sf_len,
                             const void*    zero_point,
                             const DLP_TYPE zp_type,
                             md_t           zp_len,
                             const md_t     ic_offset);
/**
 * quanta_mr16_f32s8
 *
 * Entry point for F32 to S8 quantization optimized for AMD Zen4 (AVX-512
 * VNNI). Dispatcher that routes to symmetric or asymmetric quantization based
 * on zero-point presence.
 *
 * @param quant_a_buffer Output buffer for quantized S8 values (MC x KC,
 * row-major with stride KC)
 * @param a              Input F32 matrix (MC x KC)
 * @param rs_a           Row stride of input matrix a (1=column-major)
 * @param cs_a           Column stride of input matrix a (1=row-major)
 * @param MC             Number of rows
 * @param KC             Number of columns
 * @param scale_factor   Quantization scale factors (length 1 for per-tensor, MC
 * for per-row)
 * @param sf_type        Data type of scale_factor (U8/S8/S32/BF16/F32)
 * @param sf_len         Length of scale_factor array (1=per-tensor,
 * >=MC=per-row)
 * @param zero_point     Zero-point values (NULL for symmetric, non-NULL for
 * asymmetric)
 * @param zp_type        Data type of zero_point
 * @param zp_len         Length of zero_point array
 * @param ic_offset      Row offset for indexing
 *
 * Note: Supports both row-major (cs_a==1) and column-major (rs_a==1) input
 * layouts.
 */
void
quanta_mr16_f32s8(int8_t*        quant_a_buffer,
                  const float*   a,
                  const md_t     rs_a,
                  const md_t     cs_a,
                  const md_t     MC,
                  const md_t     KC,
                  const void*    scale_factor,
                  const DLP_TYPE sf_type,
                  md_t           sf_len,
                  const void*    zero_point,
                  const DLP_TYPE zp_type,
                  md_t           zp_len,
                  const md_t     ic_offset)
{
    // Input is column-major (rs_a == 1)
    if (rs_a == 1) {
        if (zero_point) {
            // Asymmetric quantization: q = round(a * scale) - zero_point
            quant_a_asym_f32s8_col_major(
                quant_a_buffer, a, rs_a, cs_a, MC, KC, scale_factor, sf_type,
                sf_len, zero_point, zp_type, zp_len, ic_offset);
        } else {
            // Symmetric quantization: q = round(a * scale)
            quant_a_sym_f32s8_col_major(quant_a_buffer, a, rs_a, cs_a, MC, KC,
                                        scale_factor, sf_type, sf_len,
                                        ic_offset);
        }
    } else {
        // Input is row-major (cs_a == 1)
        if (zero_point) {
            // Asymmetric quantization: q = round(a * scale) - zero_point
            quant_a_asym_f32s8_row_major(
                quant_a_buffer, a, rs_a, cs_a, MC, KC, scale_factor, sf_type,
                sf_len, zero_point, zp_type, zp_len, ic_offset);
        } else {
            // Symmetric quantization: q = round(a * scale)
            quant_a_sym_f32s8_row_major(quant_a_buffer, a, rs_a, cs_a, MC, KC,
                                        scale_factor, sf_type, sf_len,
                                        ic_offset);
        }
    }
}

/*
 * quant_a_sym_f32s8_row_major
 *
 * Convert an MC x KC F32 matrix (row-major, cs_a==1) to int8 using symmetric
 * per-tensor (sf_len==1) or per-row (sf_len>=MC) scale factors:
 *   q = round( a * scale[row] )
 *
 * Algorithm:
 *   - Processes columns in 16-element AVX-512 blocks (vectorized K dimension)
 *   - Final tail < 16 elements uses masked operations
 *   - Row blocking: 16/8/4/2/1 unrolling to reduce scalar cleanup
 *   - Pipeline: Load F32, multiply by scale, round, saturate store to S8
 *
 * Output layout: Dense row-major (row stride = KC)
 */
void
quant_a_sym_f32s8_row_major(int8_t*        quant_a_buffer,
                            const float*   a,
                            const md_t     rs_a,
                            const md_t     cs_a,
                            const md_t     MC,
                            const md_t     KC,
                            const void*    scale_factor,
                            const DLP_TYPE sf_type,
                            md_t           sf_len,
                            const md_t     ic_offset)
{
    // AVX-512 parameters: process 16 elements per vector register.
    md_t      MR       = 16;            // Max rows per unrolled block
    md_t      NUM_ELEM = 16;            // Elements per ZMM register
    md_t      kleft    = KC % NUM_ELEM; // Tail elements
    __mmask16 mask = 0xFFFF >> (NUM_ELEM - kleft); // Mask for tail processing

    __m512 a_reg[MR]; // Temporary registers for input data
    __m512 sf[MR];    // Scale factors broadcasted into registers

    // Initialize to avoid uninitialized register warnings
    for (iter_t i = 0; i < MR; i++) {
        sf[i] = _mm512_setzero_ps();
    }

    md_t ic = 0, kr = 0;
    md_t sf_idx =
        0; // Index into scale_factor array (adjusted by ic_offset for per-row)

    // broadcast scale factor for tensor quantization
    if (sf_len == 1) {
        if (sf_type == DLP_U8) {
            BCST_16_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor), 0,
                          0)
        } else if (sf_type == DLP_S8) {
            BCST_16_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor), 0,
                          0)
        } else if (sf_type == DLP_S32) {
            BCST_16_SF_ZP(0, SET_S32_F32, sf, ((const int32_t*)scale_factor), 0,
                          0)
        } else if (sf_type == DLP_BF16) {
            BCST_16_SF_ZP(0, SET_BF16_F32, sf, ((const bfloat16*)scale_factor),
                          0, 0)
        } else {
            BCST_16_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor), 0, 0)
        }
    }

    for (; (ic + MR - 1) < MC; ic += MR) {
        // broadcast scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                BCST_16_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor),
                              sf_idx, 1)
            } else if (sf_type == DLP_S8) {
                BCST_16_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor),
                              sf_idx, 1)
            } else if (sf_type == DLP_BF16) {
                BCST_16_SF_ZP(0, SET_BF16_F32, sf,
                              ((const bfloat16*)scale_factor), sf_idx, 1)
            } else if (sf_type == DLP_S32) {
                BCST_16_SF_ZP(0, SET_S32_F32, sf,
                              ((const int32_t*)scale_factor), sf_idx, 1)
            } else {
                BCST_16_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor),
                              sf_idx, 1)
            }
        }

        for (kr = 0; (kr + 15) < KC; kr += 16) {
            // load f32 in a_reg[0-15]
            LOAD_16_F32_ROW(0, a_reg, a, ic, kr, rs_a, cs_a)

            // QUANT_SYM a_reg[0-15]
            QUANT_16_SYM(0, a_reg, sf)

            // Saturate FP32 to S8 range [-128,127] and store to output buffer.
            STORE_16_INT8(0, quant_a_buffer, ic, kr, KC, 0xFFFF, a_reg)
        }

        if (kleft) {
            // Load masked FP32 values
            LOAD_16_F32_MASKED_ROW(0, a_reg, mask, a, ic, kr, rs_a, cs_a)
            // QUANT_SYM a_reg[0-15]
            QUANT_16_SYM(0, a_reg, sf)
            // Convert to int 8 and store
            STORE_16_INT8(0, quant_a_buffer, ic, kr, KC, mask, a_reg)
        }
    }

    for (; (ic + 7) < MC; ic += 8) {
        // broadcast scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                BCST_8_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_S8) {
                BCST_8_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_BF16) {
                BCST_8_SF_ZP(0, SET_BF16_F32, sf,
                             ((const bfloat16*)scale_factor), sf_idx, 1)
            } else if (sf_type == DLP_S32) {
                BCST_8_SF_ZP(0, SET_S32_F32, sf, ((const int32_t*)scale_factor),
                             sf_idx, 1)
            } else {
                BCST_8_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor),
                             sf_idx, 1)
            }
        }

        for (kr = 0; (kr + 15) < KC; kr += 16) {
            // load in a_reg[0-7]
            LOAD_8_F32_ROW(0, a_reg, a, ic, kr, rs_a, cs_a)
            // QUANT_SYM a_reg[0-7]
            QUANT_8_SYM(0, a_reg, sf)
            // convert to int8 and store
            STORE_8_INT8(0, quant_a_buffer, ic, kr, KC, 0xFFFF, a_reg)
        }

        if (kleft) {
            // Load masked FP32 values
            LOAD_8_F32_MASKED_ROW(0, a_reg, mask, a, ic, kr, rs_a, cs_a)
            // QUANT_SYM a_reg[0-7]
            QUANT_8_SYM(0, a_reg, sf)
            // Convert to int8 and store
            STORE_8_INT8(0, quant_a_buffer, ic, kr, KC, mask, a_reg)
        }
    }

    for (; (ic + 3) < MC; ic += 4) {
        // broadcast scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                BCST_4_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_S8) {
                BCST_4_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_S32) {
                BCST_4_SF_ZP(0, SET_S32_F32, sf, ((const int32_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_BF16) {
                BCST_4_SF_ZP(0, SET_BF16_F32, sf,
                             ((const bfloat16*)scale_factor), sf_idx, 1)
            } else {
                BCST_4_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor),
                             sf_idx, 1)
            }
        }

        for (kr = 0; (kr + 15) < KC; kr += 16) {
            // load in a_reg[0-3]
            LOAD_4_F32_ROW(0, a_reg, a, ic, kr, rs_a, cs_a)
            // QUANT_SYM a_reg[0-3]
            QUANT_4_SYM(0, a_reg, sf)
            // convert to int8 and store
            STORE_4_INT8(0, quant_a_buffer, ic, kr, KC, 0xFFFF, a_reg)
        }

        if (kleft) {
            // load masked FP32 values
            LOAD_4_F32_MASKED_ROW(0, a_reg, mask, a, ic, kr, rs_a, cs_a)
            // QUANT_SYM
            QUANT_4_SYM(0, a_reg, sf)
            // convert to int8 and store
            STORE_4_INT8(0, quant_a_buffer, ic, kr, KC, mask, a_reg)
        }
    }

    for (; (ic + 1) < MC; ic += 2) {
        // broadcast scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                BCST_2_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_S8) {
                BCST_2_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_S32) {
                BCST_2_SF_ZP(0, SET_S32_F32, sf, ((const int32_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_BF16) {
                BCST_2_SF_ZP(0, SET_BF16_F32, sf,
                             ((const bfloat16*)scale_factor), sf_idx, 1)
            } else {
                BCST_2_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor),
                             sf_idx, 1)
            }
        }

        for (kr = 0; (kr + 15) < KC; kr += 16) {
            // load in a_reg[0-1]
            LOAD_2_F32_ROW(0, a_reg, a, ic, kr, rs_a, cs_a)
            // QUANT_SYM a_reg[0-1]
            QUANT_2_SYM(0, a_reg, sf)
            // convert to int8 and store
            STORE_2_INT8(0, quant_a_buffer, ic, kr, KC, 0xFFFF, a_reg)
        }

        if (kleft) {
            // load masked FP32 values
            LOAD_2_F32_MASKED_ROW(0, a_reg, mask, a, ic, kr, rs_a, cs_a)
            // QUANT_SYM
            QUANT_2_SYM(0, a_reg, sf)
            // convert to int8 and store
            STORE_2_INT8(0, quant_a_buffer, ic, kr, KC, mask, a_reg)
        }
    }

    if (ic < MC) {
        // broadcast scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                SET_U8_F32(sf[0], ((const uint8_t*)scale_factor)[sf_idx])
            } else if (sf_type == DLP_S8) {
                SET_S8_F32(sf[0], ((const int8_t*)scale_factor)[sf_idx])
            } else if (sf_type == DLP_S32) {
                SET_S32_F32(sf[0], ((const int32_t*)scale_factor)[sf_idx])
            } else if (sf_type == DLP_BF16) {
                SET_BF16_F32(sf[0], ((const bfloat16*)scale_factor)[sf_idx])
            } else {
                sf[0] = _mm512_set1_ps(((const float*)scale_factor)[sf_idx]);
            }
        }

        for (kr = 0; (kr + 15) < KC; kr += 16) {
            // load in a_reg[0]
            LOAD_F32(a_reg[0], a + (ic * rs_a + kr * cs_a))
            // QUANT_SYM
            QUANT_SYM(a_reg[0], sf[0])
            // convert to int8 and store
            STORE_MASKED_INT8(quant_a_buffer + ((ic)*KC + kr), 0xFFFF, a_reg[0])
        }

        if (kleft) {
            // load masked FP32 values
            LOAD_MASKED_F32(a_reg[0], mask, a + (ic * rs_a + kr * cs_a))
            // QUANT_SYM
            QUANT_SYM(a_reg[0], sf[0]);
            // convert to int8 and store
            STORE_MASKED_INT8(quant_a_buffer + (ic * KC + kr), mask, a_reg[0])
        }
    }
}
/*
 * quant_a_asym_f32s8_row_major
 *
 * Convert an MC x KC F32 matrix (row-major, cs_a==1) to int8 using asymmetric
 * per-tensor (sf_len==1) or per-row (sf_len>=MC) quantization:
 *   q = round( a * scale[row] ) - zero_point[row]
 *
 * Asymmetric quantization maps the input range [a_min, a_max] to [-128, 127]
 * with an arbitrary zero-point, providing better accuracy for non-centered
 * distributions.
 *
 * Algorithm: Same vectorization strategy as symmetric version, but with
 * additional zero-point subtraction. Uses FMA (fused multiply-subtract) for
 * efficiency: q = FMA(a, scale, -zero_point) = a * scale - zero_point
 *
 * Note: The QUANT_ASYM macro implements: fmsub(a, scale, zero_point) with
 * rounding.
 */
void
quant_a_asym_f32s8_row_major(int8_t*        quant_a_buffer,
                             const float*   a,
                             const md_t     rs_a,
                             const md_t     cs_a,
                             const md_t     MC,
                             const md_t     KC,
                             const void*    scale_factor,
                             const DLP_TYPE sf_type,
                             md_t           sf_len,
                             const void*    zero_point,
                             const DLP_TYPE zp_type,
                             md_t           zp_len,
                             const md_t     ic_offset)
{
    md_t      MR       = 16;
    md_t      NUM_ELEM = 16;
    md_t      kleft    = KC % NUM_ELEM;
    __m512    a_reg[MR];
    __m512    sf[MR]; // Scale factors
    __m512    zp[MR]; // Zero-points
    __mmask16 mask = 0xFFFF >> (NUM_ELEM - kleft);

    md_t ic = 0, kr = 0;
    md_t sf_idx = 0; // Index into scale_factor array (adjusted by ic_offset)
    md_t zp_idx = 0; // Index into zero_point array (adjusted by ic_offset)

    // Initialize to avoid uninitialized register warnings.
    for (iter_t i = 0; i < MR; i++) {
        sf[i] = _mm512_setzero_ps();
        zp[i] = _mm512_setzero_ps();
    }

    // broadcast scale factor for tensor quantization
    if (sf_len == 1) {
        if (sf_type == DLP_U8) {
            BCST_16_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor), 0,
                          0)
        } else if (sf_type == DLP_S8) {
            BCST_16_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor), 0,
                          0)
        } else if (sf_type == DLP_S32) {
            BCST_16_SF_ZP(0, SET_S32_F32, sf, ((const int32_t*)scale_factor), 0,
                          0)
        } else if (sf_type == DLP_BF16) {
            BCST_16_SF_ZP(0, SET_BF16_F32, sf, ((const bfloat16*)scale_factor),
                          0, 0)
        } else {
            BCST_16_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor), 0, 0)
        }
    }

    // =========================================================================
    // ZERO-POINT SETUP: Per-tensor quantization (zp_len == 1)
    // =========================================================================
    if (zp_len == 1) {
        if (zp_type == DLP_U8) {
            BCST_16_SF_ZP(0, SET_U8_F32, zp, ((const uint8_t*)zero_point), 0, 0)
        } else if (zp_type == DLP_S8) {
            BCST_16_SF_ZP(0, SET_S8_F32, zp, ((const int8_t*)zero_point), 0, 0)
        } else if (zp_type == DLP_S32) {
            BCST_16_SF_ZP(0, SET_S32_F32, zp, ((const int32_t*)zero_point), 0,
                          0)
        } else if (zp_type == DLP_BF16) {
            BCST_16_SF_ZP(0, SET_BF16_F32, zp, ((const bfloat16*)zero_point), 0,
                          0)
        } else {
            BCST_16_SF_ZP(0, SET_F32, zp, ((const float*)zero_point), 0, 0)
        }
    }

    for (; (ic + MR - 1) < MC; ic += MR) {
        // broadcast scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                BCST_16_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor),
                              sf_idx, 1)
            } else if (sf_type == DLP_S8) {
                BCST_16_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor),
                              sf_idx, 1)
            } else if (sf_type == DLP_BF16) {
                BCST_16_SF_ZP(0, SET_BF16_F32, sf,
                              ((const bfloat16*)scale_factor), sf_idx, 1)
            } else if (sf_type == DLP_S32) {
                BCST_16_SF_ZP(0, SET_S32_F32, sf,
                              ((const int32_t*)scale_factor), sf_idx, 1)
            } else {
                BCST_16_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor),
                              sf_idx, 1)
            }
        }

        // broadcast zero point
        if (zp_len > 1) {
            zp_idx = ic_offset + ic;
            if (zp_type == DLP_U8) {
                BCST_16_SF_ZP(0, SET_U8_F32, zp, ((const uint8_t*)zero_point),
                              zp_idx, 1)
            } else if (zp_type == DLP_S8) {
                BCST_16_SF_ZP(0, SET_S8_F32, zp, ((const int8_t*)zero_point),
                              zp_idx, 1)
            } else if (zp_type == DLP_S32) {
                BCST_16_SF_ZP(0, SET_S32_F32, zp, ((const int32_t*)zero_point),
                              zp_idx, 1)
            } else if (zp_type == DLP_BF16) {
                BCST_16_SF_ZP(0, SET_BF16_F32, zp,
                              ((const bfloat16*)zero_point), zp_idx, 1)
            } else {
                BCST_16_SF_ZP(0, SET_F32, zp, ((const float*)zero_point),
                              zp_idx, 1)
            }
        }

        for (kr = 0; (kr + 15) < KC; kr += 16) {
            LOAD_16_F32_ROW(0, a_reg, a, ic, kr, rs_a, cs_a)
            QUANT_16_ASYM(0, a_reg, sf, zp)
            STORE_16_INT8(0, quant_a_buffer, ic, kr, KC, 0xFFFF, a_reg)
        }

        if (kleft) {
            LOAD_16_F32_MASKED_ROW(0, a_reg, mask, a, ic, kr, rs_a, cs_a)
            QUANT_16_ASYM(0, a_reg, sf, zp)
            STORE_16_INT8(0, quant_a_buffer, ic, kr, KC, mask, a_reg)
        }
    }

    for (; (ic + 7) < MC; ic += 8) {
        // broadcast scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                BCST_8_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_S8) {
                BCST_8_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_S32) {
                BCST_8_SF_ZP(0, SET_S32_F32, sf, ((const int32_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_BF16) {
                BCST_8_SF_ZP(0, SET_BF16_F32, sf,
                             ((const bfloat16*)scale_factor), sf_idx, 1)
            } else {
                BCST_8_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor),
                             sf_idx, 1)
            }
        }

        // broadcast zero point
        if (zp_len > 1) {
            zp_idx = ic_offset + ic;
            if (zp_type == DLP_U8) {
                BCST_8_SF_ZP(0, SET_U8_F32, zp, ((const uint8_t*)zero_point),
                             zp_idx, 1)
            } else if (zp_type == DLP_S8) {
                BCST_8_SF_ZP(0, SET_S8_F32, zp, ((const int8_t*)zero_point),
                             zp_idx, 1)
            } else if (zp_type == DLP_S32) {
                BCST_8_SF_ZP(0, SET_S32_F32, zp, ((const int32_t*)zero_point),
                             zp_idx, 1)
            } else if (zp_type == DLP_BF16) {
                BCST_8_SF_ZP(0, SET_BF16_F32, zp, ((const bfloat16*)zero_point),
                             zp_idx, 1)
            } else {
                BCST_8_SF_ZP(0, SET_F32, zp, ((const float*)zero_point), zp_idx,
                             1)
            }
        }

        for (kr = 0; (kr + 15) < KC; kr += 16) {
            LOAD_8_F32_ROW(0, a_reg, a, ic, kr, rs_a, cs_a)
            QUANT_8_ASYM(0, a_reg, sf, zp)
            STORE_8_INT8(0, quant_a_buffer, ic, kr, KC, 0xFFFF, a_reg)
        }

        if (kleft) {
            LOAD_8_F32_MASKED_ROW(0, a_reg, mask, a, ic, kr, rs_a, cs_a)
            QUANT_8_ASYM(0, a_reg, sf, zp)
            STORE_8_INT8(0, quant_a_buffer, ic, kr, KC, mask, a_reg)
        }
    }

    for (; (ic + 3) < MC; ic += 4) {
        // broadcast scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                BCST_4_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_S8) {
                BCST_4_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_S32) {
                BCST_4_SF_ZP(0, SET_S32_F32, sf, ((const int32_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_BF16) {
                BCST_4_SF_ZP(0, SET_BF16_F32, sf,
                             ((const bfloat16*)scale_factor), sf_idx, 1)
            } else {
                BCST_4_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor),
                             sf_idx, 1)
            }
        }

        // broadcast zero point
        if (zp_len > 1) {
            zp_idx = ic_offset + ic;
            if (zp_type == DLP_U8) {
                BCST_4_SF_ZP(0, SET_U8_F32, zp, ((const uint8_t*)zero_point),
                             zp_idx, 1)
            } else if (zp_type == DLP_S8) {
                BCST_4_SF_ZP(0, SET_S8_F32, zp, ((const int8_t*)zero_point),
                             zp_idx, 1)
            } else if (zp_type == DLP_S32) {
                BCST_4_SF_ZP(0, SET_S32_F32, zp, ((const int32_t*)zero_point),
                             zp_idx, 1)
            } else if (zp_type == DLP_BF16) {
                BCST_4_SF_ZP(0, SET_BF16_F32, zp, ((const bfloat16*)zero_point),
                             zp_idx, 1)
            } else {
                BCST_4_SF_ZP(0, SET_F32, zp, ((const float*)zero_point), zp_idx,
                             1)
            }
        }

        for (kr = 0; (kr + 15) < KC; kr += 16) {
            LOAD_4_F32_ROW(0, a_reg, a, ic, kr, rs_a, cs_a)
            QUANT_4_ASYM(0, a_reg, sf, zp)
            STORE_4_INT8(0, quant_a_buffer, ic, kr, KC, 0xFFFF, a_reg)
        }

        if (kleft) {
            LOAD_4_F32_MASKED_ROW(0, a_reg, mask, a, ic, kr, rs_a, cs_a)
            QUANT_4_ASYM(0, a_reg, sf, zp)
            STORE_4_INT8(0, quant_a_buffer, ic, kr, KC, mask, a_reg)
        }
    }

    for (; (ic + 1) < MC; ic += 2) {
        // broadcast scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                BCST_2_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_S8) {
                BCST_2_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_S32) {
                BCST_2_SF_ZP(0, SET_S32_F32, sf, ((const int32_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_BF16) {
                BCST_2_SF_ZP(0, SET_BF16_F32, sf,
                             ((const bfloat16*)scale_factor), sf_idx, 1)
            } else {
                BCST_2_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor),
                             sf_idx, 1)
            }
        }

        // broadcast zero point
        if (zp_len > 1) {
            zp_idx = ic_offset + ic;
            if (zp_type == DLP_U8) {
                BCST_2_SF_ZP(0, SET_U8_F32, zp, ((const uint8_t*)zero_point),
                             zp_idx, 1)
            } else if (zp_type == DLP_S8) {
                BCST_2_SF_ZP(0, SET_S8_F32, zp, ((const int8_t*)zero_point),
                             zp_idx, 1)
            } else if (zp_type == DLP_S32) {
                BCST_2_SF_ZP(0, SET_S32_F32, zp, ((const int32_t*)zero_point),
                             zp_idx, 1)
            } else if (zp_type == DLP_BF16) {
                BCST_2_SF_ZP(0, SET_BF16_F32, zp, ((const bfloat16*)zero_point),
                             zp_idx, 1)
            } else {
                BCST_2_SF_ZP(0, SET_F32, zp, ((const float*)zero_point), zp_idx,
                             1)
            }
        }

        for (kr = 0; (kr + 15) < KC; kr += 16) {
            LOAD_2_F32_ROW(0, a_reg, a, ic, kr, rs_a, cs_a)
            QUANT_2_ASYM(0, a_reg, sf, zp)
            STORE_2_INT8(0, quant_a_buffer, ic, kr, KC, 0xFFFF, a_reg)
        }

        if (kleft) {
            LOAD_2_F32_MASKED_ROW(0, a_reg, mask, a, ic, kr, rs_a, cs_a)
            QUANT_2_ASYM(0, a_reg, sf, zp)
            STORE_2_INT8(0, quant_a_buffer, ic, kr, KC, mask, a_reg)
        }
    }

    if (ic < MC) {
        // scale factor
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                SET_U8_F32(sf[0], ((const uint8_t*)scale_factor)[sf_idx])
            } else if (sf_type == DLP_S8) {
                SET_S8_F32(sf[0], ((const int8_t*)scale_factor)[sf_idx])
            } else if (sf_type == DLP_S32) {
                SET_S32_F32(sf[0], ((const int32_t*)scale_factor)[sf_idx])
            } else if (sf_type == DLP_BF16) {
                SET_BF16_F32(sf[0], ((const bfloat16*)scale_factor)[sf_idx])
            } else {
                sf[0] = _mm512_set1_ps(((const float*)scale_factor)[sf_idx]);
            }
        }

        // zero point
        if (zp_len > 1) {
            zp_idx = ic_offset + ic;
            if (zp_type == DLP_U8) {
                SET_U8_F32(zp[0], ((const uint8_t*)zero_point)[zp_idx])
            } else if (zp_type == DLP_S8) {
                SET_S8_F32(zp[0], ((const int8_t*)zero_point)[zp_idx])
            } else if (zp_type == DLP_S32) {
                SET_S32_F32(zp[0], ((const int32_t*)zero_point)[zp_idx])
            } else if (zp_type == DLP_BF16) {
                SET_BF16_F32(zp[0], ((const bfloat16*)zero_point)[zp_idx])
            } else {
                zp[0] = _mm512_set1_ps(((const float*)zero_point)[zp_idx]);
            }
        }

        for (kr = 0; (kr + 15) < KC; kr += 16) {
            // load in a_reg[0]
            LOAD_F32(a_reg[0], a + (ic * rs_a + kr * cs_a))
            // asymmetric quantize
            QUANT_ASYM(a_reg[0], sf[0], zp[0])
            // store quantized values as int8
            STORE_MASKED_INT8(quant_a_buffer + ((ic)*KC + kr), 0xFFFF, a_reg[0])
        }

        if (kleft) {
            // load masked f32
            LOAD_MASKED_F32(a_reg[0], mask, a + (ic * rs_a + kr * cs_a))
            // asymmetric quantize
            QUANT_ASYM(a_reg[0], sf[0], zp[0]);
            // store quantized values as int8
            STORE_MASKED_INT8(quant_a_buffer + (ic * KC + kr), mask, a_reg[0])
        }
    }
}

/*
 * quant_a_sym_f32s8_col_major
 *
 * Convert an MC x KC F32 matrix (column-major, rs_a==1) to int8 using
 * symmetric per-tensor (sf_len==1) or per-row (sf_len>=MC) scale factors:
 *   q = round( a * scale[row] )
 *
 * Algorithm:
 *   - For column-major: rows are stored contiguously (rs_a == 1)
 *   - Processes rows in blocks of 16/8/4/2/1
 *   - For each block, loads 16 columns (strided by cs_a) and transposes
 *   - Handles column tails (KC not multiple of 16/8/4/2) with partial loads
 *   - Pipeline: Load F32, transpose, multiply by scale, round, saturate store
 * to S8
 *
 * Output layout: Dense row-major (row stride = KC)
 */
void
quant_a_sym_f32s8_col_major(int8_t*        quant_a_buffer,
                            const float*   a,
                            const md_t     rs_a,
                            const md_t     cs_a,
                            const md_t     MC,
                            const md_t     KC,
                            const void*    scale_factor,
                            const DLP_TYPE sf_type,
                            md_t           sf_len,
                            const md_t     ic_offset)
{
    // AVX-512 parameters: process 16 elements per vector register.
    md_t MR       = 16;            // Max rows per unrolled block
    md_t NUM_ELEM = 16;            // Elements per ZMM register
    md_t kleft    = KC % NUM_ELEM; // Tail elements
    md_t m_left   = MC % 4;

    __m512 a_reg[16], b_reg[16];

    __m512i mask1 =
        _mm512_set_epi32(0x17, 0x16, 0x15, 0x14, 0x07, 0x06, 0x05, 0x04, 0x13,
                         0x12, 0x11, 0x10, 0x03, 0x02, 0x01, 0x00);

    __m512i mask2 =
        _mm512_set_epi32(0x1F, 0x1E, 0x1D, 0x1C, 0x0F, 0x0E, 0x0D, 0x0C, 0x1B,
                         0x1A, 0x19, 0x18, 0x0B, 0x0A, 0x09, 0x08);

    __m512i mask3 =
        _mm512_set_epi32(0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10, 0x07,
                         0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00);
    __m512i mask4 =
        _mm512_set_epi32(0x1F, 0x1E, 0x1D, 0x1C, 0x1B, 0x1A, 0x19, 0x18, 0x0F,
                         0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08);
    __m512 sf[MR]; // Scale factors broadcasted into registers

    // Initialize to avoid uninitialized register warnings
    for (iter_t i = 0; i < MR; i++) {
        a_reg[i] = _mm512_setzero_ps();
        sf[i]    = _mm512_setzero_ps();
    }
    md_t ic = 0, kr = 0;
    md_t sf_idx =
        0; // Index into scale_factor array (adjusted by ic_offset for per-row)

    if (sf_len == 1) {
        if (sf_type == DLP_U8) {
            BCST_16_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor), 0,
                          0)
        } else if (sf_type == DLP_S8) {
            BCST_16_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor), 0,
                          0)
        } else if (sf_type == DLP_S32) {
            BCST_16_SF_ZP(0, SET_S32_F32, sf, ((const int32_t*)scale_factor), 0,
                          0)
        } else if (sf_type == DLP_BF16) {
            BCST_16_SF_ZP(0, SET_BF16_F32, sf, ((const bfloat16*)scale_factor),
                          0, 0)
        } else {
            BCST_16_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor), 0, 0)
        }
    }

    for (; (ic + MR - 1) < MC; ic += MR) {

        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                BCST_16_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor),
                              sf_idx, 1)
            } else if (sf_type == DLP_S8) {
                BCST_16_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor),
                              sf_idx, 1)
            } else if (sf_type == DLP_BF16) {
                BCST_16_SF_ZP(0, SET_BF16_F32, sf,
                              ((const bfloat16*)scale_factor), sf_idx, 1)
            } else if (sf_type == DLP_S32) {
                BCST_16_SF_ZP(0, SET_S32_F32, sf,
                              ((const int32_t*)scale_factor), sf_idx, 1)
            } else {
                BCST_16_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor),
                              sf_idx, 1)
            }
        }

        for (kr = 0; (kr + 15) < KC; kr += 16) {
            LOAD_16_F32_COL(0, a_reg, a, ic, kr, rs_a, cs_a)
            // Transpose first: convert columns to rows
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2);
            PERMUTE8x8(mask3, mask4)
                // Now a_reg[i] contains row ic+i, apply per-row quantization
                QUANT_16_SYM(0, a_reg, sf)
                    STORE_16_INT8(0, quant_a_buffer, ic, kr, KC, 0xFFFF, a_reg)
        }

        for (; (kr + 7) < KC; kr += 8) {
            LOAD_8_F32_COL(0, a_reg, a, ic, kr, rs_a, cs_a)
            // Transpose first: convert columns to rows
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                // Now a_reg[i] contains row ic+i, apply per-row quantization
                QUANT_16_SYM(0, a_reg, sf)
                // Store with mask 0xFF (8 valid elements per row after 8-column
                // transpose)
                STORE_16_INT8(0, quant_a_buffer, ic, kr, KC, 0xFF, a_reg)
        }

        for (; (kr + 3) < KC; kr += 4) {
            LOAD_4_F32_COL(0, a_reg, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf)
                    STORE_16_INT8(0, quant_a_buffer, ic, kr, KC, 0xF, a_reg)
        }

        for (; (kr + 1) < KC; kr += 2) {
            LOAD_2_F32_COL(0, a_reg, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf)
                    STORE_16_INT8(0, quant_a_buffer, ic, kr, KC, 0x3, a_reg)
        }

        if (kr < KC) {
            LOAD_F32(a_reg[0], a + (ic * rs_a) + ((kr + 0) * cs_a))
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf)
                    STORE_16_INT8(0, quant_a_buffer, ic, kr, KC, 0x1, a_reg)
        }
    }
    for (; (ic + 7) < MC; ic += 8) {
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                BCST_8_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_S8) {
                BCST_8_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_BF16) {
                BCST_8_SF_ZP(0, SET_BF16_F32, sf,
                             ((const bfloat16*)scale_factor), sf_idx, 1)
            } else if (sf_type == DLP_S32) {
                BCST_8_SF_ZP(0, SET_S32_F32, sf, ((const int32_t*)scale_factor),
                             sf_idx, 1)
            } else {
                BCST_8_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor),
                             sf_idx, 1)
            }
        }

        for (kr = 0; (kr + 15) < KC; kr += 16) {
            LOAD_16_F32_MASKED_COL(0, a_reg, 0xFF, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf)
                    STORE_8_INT8(0, quant_a_buffer, ic, kr, KC, 0xFFFF, a_reg)
        }
        for (; (kr + 7) < KC; kr += 8) {
            LOAD_8_F32_MASKED_COL(0, a_reg, 0xFF, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf)
                    STORE_8_INT8(0, quant_a_buffer, ic, kr, KC, 0xFF, a_reg)
        }
        for (; (kr + 3) < KC; kr += 4) {
            LOAD_4_F32_MASKED_COL(0, a_reg, 0xFF, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf)
                    STORE_8_INT8(0, quant_a_buffer, ic, kr, KC, 0xF, a_reg)
        }
        for (; (kr + 1) < KC; kr += 2) {
            LOAD_2_F32_MASKED_COL(0, a_reg, 0xFF, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf)
                    STORE_8_INT8(0, quant_a_buffer, ic, kr, KC, 0x03, a_reg)
        }
        if (kr < KC) {
            LOAD_MASKED_F32(a_reg[0], 0xFF, a + (ic * rs_a) + ((kr + 0) * cs_a))
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf)
                    STORE_8_INT8(0, quant_a_buffer, ic, kr, KC, 0x01, a_reg)
        }
    }
    for (; (ic + 3) < MC; ic += 4) {
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                BCST_4_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_S8) {
                BCST_4_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_BF16) {
                BCST_4_SF_ZP(0, SET_BF16_F32, sf,
                             ((const bfloat16*)scale_factor), sf_idx, 1)
            } else if (sf_type == DLP_S32) {
                BCST_4_SF_ZP(0, SET_S32_F32, sf, ((const int32_t*)scale_factor),
                             sf_idx, 1)
            } else {
                BCST_4_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor),
                             sf_idx, 1)
            }
        }

        for (kr = 0; (kr + 15) < KC; kr += 16) {
            LOAD_16_F32_MASKED_COL(0, a_reg, 0xF, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf)
                    STORE_4_INT8(0, quant_a_buffer, ic, kr, KC, 0xFFFF, a_reg)
        }
        for (; (kr + 7) < KC; kr += 8) {
            LOAD_8_F32_MASKED_COL(0, a_reg, 0xF, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf)
                    STORE_4_INT8(0, quant_a_buffer, ic, kr, KC, 0xFF, a_reg)
        }
        for (; (kr + 3) < KC; kr += 4) {
            LOAD_4_F32_MASKED_COL(0, a_reg, 0xF, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf)
                    STORE_4_INT8(0, quant_a_buffer, ic, kr, KC, 0xF, a_reg)
        }
        for (; (kr + 1) < KC; kr += 2) {
            LOAD_2_F32_MASKED_COL(0, a_reg, 0xF, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf)
                    STORE_4_INT8(0, quant_a_buffer, ic, kr, KC, 0x03, a_reg)
        }
        if (kr < KC) {
            LOAD_MASKED_F32(a_reg[0], 0xF, a + (ic * rs_a) + ((kr + 0) * cs_a))
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf)
                    STORE_4_INT8(0, quant_a_buffer, ic, kr, KC, 0x01, a_reg)
        }
    }

    // Handle ic+2 (2 remaining rows)
    for (; (ic + 1) < MC; ic += 2) {
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                BCST_2_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_S8) {
                BCST_2_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_BF16) {
                BCST_2_SF_ZP(0, SET_BF16_F32, sf,
                             ((const bfloat16*)scale_factor), sf_idx, 1)
            } else if (sf_type == DLP_S32) {
                BCST_2_SF_ZP(0, SET_S32_F32, sf, ((const int32_t*)scale_factor),
                             sf_idx, 1)
            } else {
                BCST_2_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor),
                             sf_idx, 1)
            }
        }

        for (kr = 0; (kr + 15) < KC; kr += 16) {
            LOAD_16_F32_MASKED_COL(0, a_reg, 0x3, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf)
                    STORE_2_INT8(0, quant_a_buffer, ic, kr, KC, 0xFFFF, a_reg)
        }
        for (; (kr + 7) < KC; kr += 8) {
            LOAD_8_F32_MASKED_COL(0, a_reg, 0x3, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf)
                    STORE_2_INT8(0, quant_a_buffer, ic, kr, KC, 0xFF, a_reg)
        }
        for (; (kr + 3) < KC; kr += 4) {
            LOAD_4_F32_MASKED_COL(0, a_reg, 0x3, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf)
                    STORE_2_INT8(0, quant_a_buffer, ic, kr, KC, 0xF, a_reg)
        }
        for (; (kr + 1) < KC; kr += 2) {
            LOAD_2_F32_MASKED_COL(0, a_reg, 0x3, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf)
                    STORE_2_INT8(0, quant_a_buffer, ic, kr, KC, 0x03, a_reg)
        }
        if (kr < KC) {
            LOAD_MASKED_F32(a_reg[0], 0x3, a + (ic * rs_a) + ((kr + 0) * cs_a))
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf)
                    STORE_2_INT8(0, quant_a_buffer, ic, kr, KC, 0x01, a_reg)
        }
    }

    // Handle ic+1 (1 remaining row)
    if (ic < MC) {
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                SET_U8_F32(sf[0], ((const uint8_t*)scale_factor)[sf_idx])
            } else if (sf_type == DLP_S8) {
                SET_S8_F32(sf[0], ((const int8_t*)scale_factor)[sf_idx])
            } else if (sf_type == DLP_S32) {
                SET_S32_F32(sf[0], ((const int32_t*)scale_factor)[sf_idx])
            } else if (sf_type == DLP_BF16) {
                SET_BF16_F32(sf[0], ((const bfloat16*)scale_factor)[sf_idx])
            } else {
                sf[0] = _mm512_set1_ps(((const float*)scale_factor)[sf_idx]);
            }
        }

        for (kr = 0; (kr + 15) < KC; kr += 16) {
            LOAD_16_F32_MASKED_COL(0, a_reg, 0x1, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf) STORE_MASKED_INT8(
                    quant_a_buffer + (ic * KC + kr), 0xFFFF, a_reg[0])
        }
        for (; (kr + 7) < KC; kr += 8) {
            LOAD_8_F32_MASKED_COL(0, a_reg, 0x1, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf) STORE_MASKED_INT8(
                    quant_a_buffer + (ic * KC + kr), 0xFF, a_reg[0])
        }
        for (; (kr + 3) < KC; kr += 4) {
            LOAD_4_F32_MASKED_COL(0, a_reg, 0x1, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf) STORE_MASKED_INT8(
                    quant_a_buffer + (ic * KC + kr), 0xF, a_reg[0])
        }
        for (; (kr + 1) < KC; kr += 2) {
            LOAD_2_F32_MASKED_COL(0, a_reg, 0x1, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf) STORE_MASKED_INT8(
                    quant_a_buffer + (ic * KC + kr), 0x3, a_reg[0])
        }
        if (kr < KC) {
            LOAD_MASKED_F32(a_reg[0], 0x1, a + (ic * rs_a) + ((kr + 0) * cs_a))
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_SYM(0, a_reg, sf) STORE_MASKED_INT8(
                    quant_a_buffer + (ic * KC + kr), 0x1, a_reg[0])
        }
    }
}

/*
 * quant_a_asym_f32s8_col_major
 *
 * Convert an MC x KC F32 matrix (column-major, rs_a==1) to int8 using
 * asymmetric per-tensor (sf_len==1) or per-row (sf_len>=MC) quantization:
 *   q = round( a * scale[row] ) - zero_point[row]
 *
 * Asymmetric quantization maps the input range [a_min, a_max] to [-128, 127]
 * with an arbitrary zero-point, providing better accuracy for non-centered
 * distributions.
 *
 * Algorithm:
 *   - For column-major: rows are stored contiguously (rs_a == 1)
 *   - Processes rows in blocks of 16/8/4/2/1
 *   - For each block, loads 16 columns (strided by cs_a) and transposes
 *   - Handles column tails (KC not multiple of 16/8/4/2) with partial loads
 *   - Pipeline: Load F32, transpose, multiply by scale, subtract zero-point,
 * round, saturate store to S8
 *   - Uses FMA (fused multiply-subtract) for efficiency
 *
 * Output layout: Dense row-major (row stride = KC)
 */
void
quant_a_asym_f32s8_col_major(int8_t*        quant_a_buffer,
                             const float*   a,
                             const md_t     rs_a,
                             const md_t     cs_a,
                             const md_t     MC,
                             const md_t     KC,
                             const void*    scale_factor,
                             const DLP_TYPE sf_type,
                             md_t           sf_len,
                             const void*    zero_point,
                             const DLP_TYPE zp_type,
                             md_t           zp_len,
                             const md_t     ic_offset)
{
    // AVX-512 parameters: process 16 elements per vector register.
    md_t MR       = 16;            // Max rows per unrolled block
    md_t NUM_ELEM = 16;            // Elements per ZMM register
    md_t kleft    = KC % NUM_ELEM; // Tail elements
    md_t m_left   = MC % 4;

    __m512 a_reg[16], b_reg[16];

    __m512i mask1 =
        _mm512_set_epi32(0x17, 0x16, 0x15, 0x14, 0x07, 0x06, 0x05, 0x04, 0x13,
                         0x12, 0x11, 0x10, 0x03, 0x02, 0x01, 0x00);

    __m512i mask2 =
        _mm512_set_epi32(0x1F, 0x1E, 0x1D, 0x1C, 0x0F, 0x0E, 0x0D, 0x0C, 0x1B,
                         0x1A, 0x19, 0x18, 0x0B, 0x0A, 0x09, 0x08);

    __m512i mask3 =
        _mm512_set_epi32(0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10, 0x07,
                         0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00);
    __m512i mask4 =
        _mm512_set_epi32(0x1F, 0x1E, 0x1D, 0x1C, 0x1B, 0x1A, 0x19, 0x18, 0x0F,
                         0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08);

    __m512 sf[MR], zp[MR]; // Scale factors broadcasted into registers

    // Initialize to avoid uninitialized register warnings
    for (iter_t i = 0; i < MR; i++) {
        a_reg[i] = _mm512_setzero_ps();
        sf[i]    = _mm512_setzero_ps();
        zp[i]    = _mm512_setzero_ps();
    }
    md_t ic = 0, kr = 0;
    md_t sf_idx = 0, zp_idx = 0; // Index into scale_factor array (adjusted by
                                 // ic_offset for per-row)

    if (sf_len == 1) {
        if (sf_type == DLP_U8) {
            BCST_16_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor), 0,
                          0)
        } else if (sf_type == DLP_S8) {
            BCST_16_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor), 0,
                          0)
        } else if (sf_type == DLP_S32) {
            BCST_16_SF_ZP(0, SET_S32_F32, sf, ((const int32_t*)scale_factor), 0,
                          0)
        } else if (sf_type == DLP_BF16) {
            BCST_16_SF_ZP(0, SET_BF16_F32, sf, ((const bfloat16*)scale_factor),
                          0, 0)
        } else {
            BCST_16_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor), 0, 0)
        }
    }

    // =========================================================================
    // ZERO-POINT SETUP: Per-tensor quantization (zp_len == 1)
    // =========================================================================
    if (zp_len == 1) {
        if (zp_type == DLP_U8) {
            BCST_16_SF_ZP(0, SET_U8_F32, zp, ((const uint8_t*)zero_point), 0, 0)
        } else if (zp_type == DLP_S8) {
            BCST_16_SF_ZP(0, SET_S8_F32, zp, ((const int8_t*)zero_point), 0, 0)
        } else if (zp_type == DLP_S32) {
            BCST_16_SF_ZP(0, SET_S32_F32, zp, ((const int32_t*)zero_point), 0,
                          0)
        } else if (zp_type == DLP_BF16) {
            BCST_16_SF_ZP(0, SET_BF16_F32, zp, ((const bfloat16*)zero_point), 0,
                          0)
        } else {
            BCST_16_SF_ZP(0, SET_F32, zp, ((const float*)zero_point), 0, 0)
        }
    }

    for (; (ic + MR - 1) < MC; ic += MR) {

        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                BCST_16_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor),
                              sf_idx, 1)
            } else if (sf_type == DLP_S8) {
                BCST_16_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor),
                              sf_idx, 1)
            } else if (sf_type == DLP_BF16) {
                BCST_16_SF_ZP(0, SET_BF16_F32, sf,
                              ((const bfloat16*)scale_factor), sf_idx, 1)
            } else if (sf_type == DLP_S32) {
                BCST_16_SF_ZP(0, SET_S32_F32, sf,
                              ((const int32_t*)scale_factor), sf_idx, 1)
            } else {
                BCST_16_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor),
                              sf_idx, 1)
            }
        }

        // broadcast zero point
        if (zp_len > 1) {
            zp_idx = ic_offset + ic;
            if (zp_type == DLP_U8) {
                BCST_16_SF_ZP(0, SET_U8_F32, zp, ((const uint8_t*)zero_point),
                              zp_idx, 1)
            } else if (zp_type == DLP_S8) {
                BCST_16_SF_ZP(0, SET_S8_F32, zp, ((const int8_t*)zero_point),
                              zp_idx, 1)
            } else if (zp_type == DLP_S32) {
                BCST_16_SF_ZP(0, SET_S32_F32, zp, ((const int32_t*)zero_point),
                              zp_idx, 1)
            } else if (zp_type == DLP_BF16) {
                BCST_16_SF_ZP(0, SET_BF16_F32, zp,
                              ((const bfloat16*)zero_point), zp_idx, 1)
            } else {
                BCST_16_SF_ZP(0, SET_F32, zp, ((const float*)zero_point),
                              zp_idx, 1)
            }
        }

        for (kr = 0; (kr + 15) < KC; kr += 16) {
            LOAD_16_F32_COL(0, a_reg, a, ic, kr, rs_a, cs_a)
            // Transpose first: convert columns to rows
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2);
            PERMUTE8x8(mask3, mask4)
                // Now a_reg[i] contains row ic+i, apply per-row quantization
                QUANT_16_ASYM(0, a_reg, sf, zp)
                    STORE_16_INT8(0, quant_a_buffer, ic, kr, KC, 0xFFFF, a_reg)
        }

        for (; (kr + 7) < KC; kr += 8) {
            LOAD_8_F32_COL(0, a_reg, a, ic, kr, rs_a, cs_a)
            // Transpose first: convert columns to rows
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                // Now a_reg[i] contains row ic+i, apply per-row quantization
                QUANT_16_ASYM(0, a_reg, sf, zp)
                // Store with mask 0xFF (8 valid elements per row after 8-column
                // transpose)
                STORE_16_INT8(0, quant_a_buffer, ic, kr, KC, 0xFF, a_reg)
        }

        for (; (kr + 3) < KC; kr += 4) {
            LOAD_4_F32_COL(0, a_reg, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp)
                    STORE_16_INT8(0, quant_a_buffer, ic, kr, KC, 0xF, a_reg)
        }

        for (; (kr + 1) < KC; kr += 2) {
            LOAD_2_F32_COL(0, a_reg, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp)
                    STORE_16_INT8(0, quant_a_buffer, ic, kr, KC, 0x3, a_reg)
        }

        if (kr < KC) {
            LOAD_F32(a_reg[0], a + (ic * rs_a) + ((kr + 0) * cs_a))
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp)
                    STORE_16_INT8(0, quant_a_buffer, ic, kr, KC, 0x1, a_reg)
        }
    }
    for (; (ic + 7) < MC; ic += 8) {
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                BCST_8_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_S8) {
                BCST_8_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_BF16) {
                BCST_8_SF_ZP(0, SET_BF16_F32, sf,
                             ((const bfloat16*)scale_factor), sf_idx, 1)
            } else if (sf_type == DLP_S32) {
                BCST_8_SF_ZP(0, SET_S32_F32, sf, ((const int32_t*)scale_factor),
                             sf_idx, 1)
            } else {
                BCST_8_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor),
                             sf_idx, 1)
            }
        }
        // broadcast zero point
        if (zp_len > 1) {
            zp_idx = ic_offset + ic;
            if (zp_type == DLP_U8) {
                BCST_8_SF_ZP(0, SET_U8_F32, zp, ((const uint8_t*)zero_point),
                             zp_idx, 1)
            } else if (zp_type == DLP_S8) {
                BCST_8_SF_ZP(0, SET_S8_F32, zp, ((const int8_t*)zero_point),
                             zp_idx, 1)
            } else if (zp_type == DLP_S32) {
                BCST_8_SF_ZP(0, SET_S32_F32, zp, ((const int32_t*)zero_point),
                             zp_idx, 1)
            } else if (zp_type == DLP_BF16) {
                BCST_8_SF_ZP(0, SET_BF16_F32, zp, ((const bfloat16*)zero_point),
                             zp_idx, 1)
            } else {
                BCST_8_SF_ZP(0, SET_F32, zp, ((const float*)zero_point), zp_idx,
                             1)
            }
        }

        for (kr = 0; (kr + 15) < KC; kr += 16) {
            LOAD_16_F32_MASKED_COL(0, a_reg, 0xFF, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp)
                    STORE_8_INT8(0, quant_a_buffer, ic, kr, KC, 0xFFFF, a_reg)
        }
        for (; (kr + 7) < KC; kr += 8) {
            LOAD_8_F32_MASKED_COL(0, a_reg, 0xFF, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp)
                    STORE_8_INT8(0, quant_a_buffer, ic, kr, KC, 0xFF, a_reg)
        }
        for (; (kr + 3) < KC; kr += 4) {
            LOAD_4_F32_MASKED_COL(0, a_reg, 0xFF, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp)
                    STORE_8_INT8(0, quant_a_buffer, ic, kr, KC, 0xF, a_reg)
        }
        for (; (kr + 1) < KC; kr += 2) {
            LOAD_2_F32_MASKED_COL(0, a_reg, 0xFF, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp)
                    STORE_8_INT8(0, quant_a_buffer, ic, kr, KC, 0x03, a_reg)
        }
        if (kr < KC) {
            LOAD_MASKED_F32(a_reg[0], 0xFF, a + (ic * rs_a) + ((kr + 0) * cs_a))
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp)
                    STORE_8_INT8(0, quant_a_buffer, ic, kr, KC, 0x01, a_reg)
        }
    }
    for (; (ic + 3) < MC; ic += 4) {
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                BCST_4_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_S8) {
                BCST_4_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_BF16) {
                BCST_4_SF_ZP(0, SET_BF16_F32, sf,
                             ((const bfloat16*)scale_factor), sf_idx, 1)
            } else if (sf_type == DLP_S32) {
                BCST_4_SF_ZP(0, SET_S32_F32, sf, ((const int32_t*)scale_factor),
                             sf_idx, 1)
            } else {
                BCST_4_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor),
                             sf_idx, 1)
            }
        }

        // broadcast zero point
        if (zp_len > 1) {
            zp_idx = ic_offset + ic;
            if (zp_type == DLP_U8) {
                BCST_4_SF_ZP(0, SET_U8_F32, zp, ((const uint8_t*)zero_point),
                             zp_idx, 1)
            } else if (zp_type == DLP_S8) {
                BCST_4_SF_ZP(0, SET_S8_F32, zp, ((const int8_t*)zero_point),
                             zp_idx, 1)
            } else if (zp_type == DLP_S32) {
                BCST_4_SF_ZP(0, SET_S32_F32, zp, ((const int32_t*)zero_point),
                             zp_idx, 1)
            } else if (zp_type == DLP_BF16) {
                BCST_4_SF_ZP(0, SET_BF16_F32, zp, ((const bfloat16*)zero_point),
                             zp_idx, 1)
            } else {
                BCST_4_SF_ZP(0, SET_F32, zp, ((const float*)zero_point), zp_idx,
                             1)
            }
        }

        for (kr = 0; (kr + 15) < KC; kr += 16) {
            LOAD_16_F32_MASKED_COL(0, a_reg, 0xF, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp)
                    STORE_4_INT8(0, quant_a_buffer, ic, kr, KC, 0xFFFF, a_reg)
        }
        for (; (kr + 7) < KC; kr += 8) {
            LOAD_8_F32_MASKED_COL(0, a_reg, 0xF, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp)
                    STORE_4_INT8(0, quant_a_buffer, ic, kr, KC, 0xFF, a_reg)
        }
        for (; (kr + 3) < KC; kr += 4) {
            LOAD_4_F32_MASKED_COL(0, a_reg, 0xF, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp)
                    STORE_4_INT8(0, quant_a_buffer, ic, kr, KC, 0xF, a_reg)
        }
        for (; (kr + 1) < KC; kr += 2) {
            LOAD_2_F32_MASKED_COL(0, a_reg, 0xF, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp)
                    STORE_4_INT8(0, quant_a_buffer, ic, kr, KC, 0x03, a_reg)
        }
        if (kr < KC) {
            LOAD_MASKED_F32(a_reg[0], 0xF, a + (ic * rs_a) + ((kr + 0) * cs_a))
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp)
                    STORE_4_INT8(0, quant_a_buffer, ic, kr, KC, 0x01, a_reg)
        }
    }

    // Handle ic+2 (2 remaining rows)
    for (; (ic + 1) < MC; ic += 2) {
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                BCST_2_SF_ZP(0, SET_U8_F32, sf, ((const uint8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_S8) {
                BCST_2_SF_ZP(0, SET_S8_F32, sf, ((const int8_t*)scale_factor),
                             sf_idx, 1)
            } else if (sf_type == DLP_BF16) {
                BCST_2_SF_ZP(0, SET_BF16_F32, sf,
                             ((const bfloat16*)scale_factor), sf_idx, 1)
            } else if (sf_type == DLP_S32) {
                BCST_2_SF_ZP(0, SET_S32_F32, sf, ((const int32_t*)scale_factor),
                             sf_idx, 1)
            } else {
                BCST_2_SF_ZP(0, SET_F32, sf, ((const float*)scale_factor),
                             sf_idx, 1)
            }
        }
        // broadcast zero point
        if (zp_len > 1) {
            zp_idx = ic_offset + ic;
            if (zp_type == DLP_U8) {
                BCST_2_SF_ZP(0, SET_U8_F32, zp, ((const uint8_t*)zero_point),
                             zp_idx, 1)
            } else if (zp_type == DLP_S8) {
                BCST_2_SF_ZP(0, SET_S8_F32, zp, ((const int8_t*)zero_point),
                             zp_idx, 1)
            } else if (zp_type == DLP_S32) {
                BCST_2_SF_ZP(0, SET_S32_F32, zp, ((const int32_t*)zero_point),
                             zp_idx, 1)
            } else if (zp_type == DLP_BF16) {
                BCST_2_SF_ZP(0, SET_BF16_F32, zp, ((const bfloat16*)zero_point),
                             zp_idx, 1)
            } else {
                BCST_2_SF_ZP(0, SET_F32, zp, ((const float*)zero_point), zp_idx,
                             1)
            }
        }
        for (kr = 0; (kr + 15) < KC; kr += 16) {
            LOAD_16_F32_MASKED_COL(0, a_reg, 0x3, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp)
                    STORE_2_INT8(0, quant_a_buffer, ic, kr, KC, 0xFFFF, a_reg)
        }
        for (; (kr + 7) < KC; kr += 8) {
            LOAD_8_F32_MASKED_COL(0, a_reg, 0x3, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp)
                    STORE_2_INT8(0, quant_a_buffer, ic, kr, KC, 0xFF, a_reg)
        }
        for (; (kr + 3) < KC; kr += 4) {
            LOAD_4_F32_MASKED_COL(0, a_reg, 0x3, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp)
                    STORE_2_INT8(0, quant_a_buffer, ic, kr, KC, 0xF, a_reg)
        }
        for (; (kr + 1) < KC; kr += 2) {
            LOAD_2_F32_MASKED_COL(0, a_reg, 0x3, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp)
                    STORE_2_INT8(0, quant_a_buffer, ic, kr, KC, 0x03, a_reg)
        }
        if (kr < KC) {
            LOAD_MASKED_F32(a_reg[0], 0x3, a + (ic * rs_a) + ((kr + 0) * cs_a))
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp)
                    STORE_2_INT8(0, quant_a_buffer, ic, kr, KC, 0x01, a_reg)
        }
    }

    // Handle ic+1 (1 remaining row)
    if (ic < MC) {
        if (sf_len > 1) {
            sf_idx = ic_offset + ic;
            if (sf_type == DLP_U8) {
                SET_U8_F32(sf[0], ((const uint8_t*)scale_factor)[sf_idx])
            } else if (sf_type == DLP_S8) {
                SET_S8_F32(sf[0], ((const int8_t*)scale_factor)[sf_idx])
            } else if (sf_type == DLP_S32) {
                SET_S32_F32(sf[0], ((const int32_t*)scale_factor)[sf_idx])
            } else if (sf_type == DLP_BF16) {
                SET_BF16_F32(sf[0], ((const bfloat16*)scale_factor)[sf_idx])
            } else {
                sf[0] = _mm512_set1_ps(((const float*)scale_factor)[sf_idx]);
            }
        }
        // zero point
        if (zp_len > 1) {
            zp_idx = ic_offset + ic;
            if (zp_type == DLP_U8) {
                SET_U8_F32(zp[0], ((const uint8_t*)zero_point)[zp_idx])
            } else if (zp_type == DLP_S8) {
                SET_S8_F32(zp[0], ((const int8_t*)zero_point)[zp_idx])
            } else if (zp_type == DLP_S32) {
                SET_S32_F32(zp[0], ((const int32_t*)zero_point)[zp_idx])
            } else if (zp_type == DLP_BF16) {
                SET_BF16_F32(zp[0], ((const bfloat16*)zero_point)[zp_idx])
            } else {
                zp[0] = _mm512_set1_ps(((const float*)zero_point)[zp_idx]);
            }
        }

        for (kr = 0; (kr + 15) < KC; kr += 16) {
            LOAD_16_F32_MASKED_COL(0, a_reg, 0x1, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp) STORE_MASKED_INT8(
                    quant_a_buffer + (ic * KC + kr), 0xFFFF, a_reg[0])
        }
        for (; (kr + 7) < KC; kr += 8) {
            LOAD_8_F32_MASKED_COL(0, a_reg, 0x1, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp) STORE_MASKED_INT8(
                    quant_a_buffer + (ic * KC + kr), 0xFF, a_reg[0])
        }
        for (; (kr + 3) < KC; kr += 4) {
            LOAD_4_F32_MASKED_COL(0, a_reg, 0x1, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp) STORE_MASKED_INT8(
                    quant_a_buffer + (ic * KC + kr), 0xF, a_reg[0])
        }
        for (; (kr + 1) < KC; kr += 2) {
            LOAD_2_F32_MASKED_COL(0, a_reg, 0x1, a, ic, kr, rs_a, cs_a)
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp) STORE_MASKED_INT8(
                    quant_a_buffer + (ic * KC + kr), 0x3, a_reg[0])
        }
        if (kr < KC) {
            LOAD_MASKED_F32(a_reg[0], 0x1, a + (ic * rs_a) + ((kr + 0) * cs_a))
            UNPACKLO_PS16
            UNPACKHI_PS16
            SHUFFLE_64x2 PERMUTE4x4(mask1, mask2) PERMUTE8x8(mask3, mask4)
                QUANT_16_ASYM(0, a_reg, sf, zp) STORE_MASKED_INT8(
                    quant_a_buffer + (ic * KC + kr), 0x1, a_reg[0])
        }
    }
}
