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
#ifndef LPGEMM_F32_PACKAB
#define LPGEMM_F32_PACKAB

void
packa_mr16_f32f32f32of32_col_major(float*       pack_a_buffer,
                                   const float* a,
                                   const md_t   rs_a,
                                   const md_t   cs_a,
                                   const md_t   MC,
                                   const md_t   KC,
                                   md_t*        rs_p,
                                   md_t*        cs_p);

void
packa_mr8_f32f32f32of32_col_major(float*       pack_a_buffer,
                                  const float* a,
                                  const md_t   rs_a,
                                  const md_t   cs_a,
                                  const md_t   MC,
                                  const md_t   KC,
                                  md_t*        rs_p,
                                  md_t*        cs_p);

void
packa_mr6_f32f32f32of32_avx512(float*       pack_a_buf,
                               const float* a,
                               const md_t   rs,
                               const md_t   cs,
                               const md_t   MC,
                               const md_t   KC,
                               md_t*        rs_a,
                               md_t*        cs_a);

void
packa_mr6_f32f32f32of32_avx2(float*       pack_a_buf,
                             const float* a,
                             const md_t   rs,
                             const md_t   cs,
                             const md_t   MC,
                             const md_t   KC,
                             md_t*        rs_a,
                             md_t*        cs_a);

typedef void (*lpgemm_pack_f32)(float*,
                                const float*,
                                const md_t,
                                const md_t,
                                const md_t,
                                const md_t,
                                md_t*,
                                md_t*);

void
packb_nr64_f32f32f32of32(float*       pack_b_buffer,
                         const float* b,
                         const md_t   rs_b,
                         const md_t   cs_b,
                         const md_t   NC,
                         const md_t   KC,
                         md_t*        rs_p,
                         md_t*        cs_p);

void
packb_nr16_f32f32f32of32(float*       pack_b_buffer,
                         const float* b,
                         const md_t   rs_b,
                         const md_t   cs_b,
                         const md_t   NC,
                         const md_t   KC,
                         md_t*        rs_p,
                         md_t*        cs_p);

#endif // LPGEMM_F32_PACKAB
