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

#ifndef DLP_GEMM_INT8_PACKA
#define DLP_GEMM_INT8_PACKA

// The strides needs to be updated based on the m_fringe value to account
// for different schemas used to pack A fringe cases.
DLP_INLINE void
dlp_get_packa_strides_mfringe_u8s8s32os32(const md_t rs,
                                          const md_t cs,
                                          md_t*      rs_use,
                                          md_t*      cs_use,
                                          md_t       MR,
                                          md_t       m_fringe)
{
    // Only applicable for row major packing.
    if ((rs != 1) && (cs == 1) && ((*cs_use) > MR)) {
        (*rs_use) = 4;
        (*cs_use) = ((*cs_use) / MR) * m_fringe;
    }
}

typedef void (*packa_s32)(uint8_t*,
                          const uint8_t*,
                          const md_t,
                          const md_t,
                          const md_t,
                          const md_t,
                          md_t*,
                          md_t*);

void
dlp_packa_u8s8s32os32(uint8_t*       pack_a_buffer_u8s8s32o32,
                      const uint8_t* a,
                      const md_t     rs,
                      const md_t     cs,
                      const md_t     MC,
                      const md_t     KC,
                      md_t*          rs_a,
                      md_t*          cs_a);

#endif // DLP_GEMM_INT8_PACKA
