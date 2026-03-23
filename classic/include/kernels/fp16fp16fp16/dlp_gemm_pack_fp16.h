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
#ifndef DLP_GEMM_FP16_PACKB
#define DLP_GEMM_FP16_PACKB

#include "classic/aocl_fp16_type.h"
#include "dlp_gemm_post_ops.h"

// Packing factor configuration
#define FP16_PACKING_FACTOR 1

// Minimum NR for fringe handling (one ZMM = 32 FP16 elements)
DLP_INLINE md_t
dlp_get_packb_fp16_min_NR()
{
    return 32; // Smallest fringe kernel = one ZMM register
}

// Function pointer types
typedef void (*dlp_gemm_pack_fp16)(float16*,
                                   const float16*,
                                   const md_t,
                                   const md_t,
                                   const md_t,
                                   const md_t,
                                   md_t*,
                                   md_t*);

typedef void (*dlp_gemm_unpack_fp16)(float16*,
                                     const float16*,
                                     const md_t,
                                     const md_t,
                                     const md_t,
                                     const md_t,
                                     md_t*,
                                     md_t*);

// Aliases for consistency with BF16 naming convention
typedef void (*pack_fp16)(float16*,
                          const float16*,
                          const md_t,
                          const md_t,
                          const md_t,
                          const md_t,
                          md_t*,
                          md_t*);

typedef void (*unpack_fp16)(float16*,
                            const float16*,
                            const md_t,
                            const md_t,
                            const md_t,
                            const md_t,
                            md_t*,
                            md_t*);

// Main packing functions (NR variants)
// NR=128: Full panel (4 ZMM registers)
void
dlp_packb_nr128_f16f16f16of16(float16*       pack_b_buffer,
                              const float16* b,
                              const md_t     rs_b,
                              const md_t     cs_b,
                              const md_t     NC,
                              const md_t     KC,
                              md_t*          rs_p,
                              md_t*          cs_p);

// NR=96: Fringe (3 ZMM registers)
void
dlp_packb_nr96_f16f16f16of16(float16*       pack_b_buffer,
                             const float16* b,
                             const md_t     rs_b,
                             const md_t     cs_b,
                             const md_t     NC,
                             const md_t     KC,
                             md_t*          rs_p,
                             md_t*          cs_p);

// NR=64: Fringe (2 ZMM registers)
void
dlp_packb_nr64_f16f16f16of16(float16*       pack_b_buffer,
                             const float16* b,
                             const md_t     rs_b,
                             const md_t     cs_b,
                             const md_t     NC,
                             const md_t     KC,
                             md_t*          rs_p,
                             md_t*          cs_p);

// NR=32: Minimum full chunk (1 ZMM register)
void
dlp_packb_nr32_f16f16f16of16(float16*       pack_b_buffer,
                             const float16* b,
                             const md_t     rs_b,
                             const md_t     cs_b,
                             const md_t     NC,
                             const md_t     KC,
                             md_t*          rs_p,
                             md_t*          cs_p);

// NR<32: Masked with zero-padding to 32
void
dlp_packb_nrlt32_f16f16f16of16(float16*       pack_b_buffer,
                               const float16* b,
                               const md_t     rs_b,
                               const md_t     cs_b,
                               const md_t     NC,
                               const md_t     KC,
                               md_t*          rs_p,
                               md_t*          cs_p);

// Unpacking functions
void
dlp_unpackb_nr128_f16f16f16of16(float16*       b,
                                const float16* unpack_b_buffer,
                                const md_t     rs_b,
                                const md_t     cs_b,
                                const md_t     NC,
                                const md_t     KC,
                                md_t*          rs_p,
                                md_t*          cs_p);

// Pack A functions (MR=32 configuration)
// MR=32 uses 32 ZMM registers (32 FP16 elements each)
// Output layout: M-MAJOR (rs_p=KC, cs_p=1)
void
dlp_packa_mr32_f16f16f16of16(float16*       pack_a_buffer,
                             const float16* a,
                             const md_t     rs_a,
                             const md_t     cs_a,
                             const md_t     MC,
                             const md_t     KC,
                             md_t*          rs_p,
                             md_t*          cs_p);

// Reference PackA implementation (ISA-independent, scalar)
void
dlp_packa_f16f16f16of16_reference(float16*       pack_a,
                                  const float16* a,
                                  const md_t     rs_a,
                                  const md_t     cs_a,
                                  const md_t     MC,
                                  const md_t     KC,
                                  md_t*          rs_p,
                                  md_t*          cs_p);

// Reference UnpackA implementation (ISA-independent, scalar)
void
dlp_unpacka_f16f16f16of16_reference(float16*       a,
                                    const float16* pack_a,
                                    const md_t     rs_a,
                                    const md_t     cs_a,
                                    const md_t     MC,
                                    const md_t     KC);

#endif // DLP_GEMM_FP16_PACKB
