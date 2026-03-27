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

#ifndef DLP_GEMM_PACK_F16_F32F16_H
#define DLP_GEMM_PACK_F16_F32F16_H

#include "classic/aocl_fp16_type.h"
#include "classic/dlp_base_types.h"

/* F32×FP16→F32 Pack B: NR=64, FP16 elements, K-MAJOR layout */
void
dlp_packb_nr64_f32f16f32of32(float16*       pack_b_buffer,
                             const float16* b,
                             const md_t     rs_b,
                             const md_t     cs_b,
                             const md_t     NC,
                             const md_t     KC,
                             md_t*          rs_p,
                             md_t*          cs_p);

/*
 * Minimum NR for fringe handling.
 * Must be NR=64 because fringe panels are always packed at stride NR=64
 * (not 32) to match the JIT kernel's B pointer advancement. The JIT kernel
 * uses numElemsPerReg=16 (F32 per ZMM) so its rsB calculation divides by
 * NR/16=4, producing a stride of NR/4*2=32 bytes = 16 FP16 elements per
 * k-step. This requires the packed data to be at stride NR=64.
 */
DLP_INLINE md_t
get_packb_f32f16f32of32_min_NR()
{
    return 64;
}

#endif /* DLP_GEMM_PACK_F16_F32F16_H */
