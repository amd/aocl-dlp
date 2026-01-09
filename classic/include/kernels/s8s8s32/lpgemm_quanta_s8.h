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

#ifndef LPGEMM_QUANTA_S8_H
#define LPGEMM_QUANTA_S8_H

#include <stdint.h>

typedef void (*quanta_bf16s8)(int8_t*,
                              const bfloat16*,
                              const md_t,
                              const md_t,
                              const md_t,
                              const md_t,
                              const void*,
                              const DLP_TYPE,
                              const md_t,
                              const void*,
                              const DLP_TYPE,
                              const md_t,
                              const md_t);

typedef void (*quanta_f32s8)(int8_t*,
                             const float*,
                             const md_t,
                             const md_t,
                             const md_t,
                             const void*,
                             const DLP_TYPE,
                             const md_t,
                             const void*,
                             const DLP_TYPE,
                             const md_t,
                             const md_t);

void
quanta_mr16_bf16s8(int8_t*         quant_a_buffer,
                   const bfloat16* a,
                   const md_t      rs_a,
                   const md_t      cs_a,
                   const md_t      MC,
                   const md_t      KC,
                   const void*     scale_factor,
                   const DLP_TYPE  sf_type,
                   md_t            sf_len,
                   const void*     zero_point,
                   const DLP_TYPE  zp_type,
                   md_t            zp_len,
                   const md_t      ic_offset);

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
                  const md_t     ic_offset);

#endif // LPGEMM_QUANTA_S8_H
