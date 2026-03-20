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

#ifndef DLP_GEMM_INT8_PACKB_S8
#define DLP_GEMM_INT8_PACKB_S8

DLP_INLINE md_t
dlp_get_packb_s8s8s32o32_min_NR()
{
    // This is the minimum NR' required for use in u8s8s32 kernels. The idea
    // here is that since k needs to be a multiple of 4 (VNNI instr), NR'=16
    // results in total of 4 * NR' = 64 bytes to be loaded, which fits in 1 ZMM
    // register. Thus the smallest n fringe kernel dimension has n=16, and thus
    // any rounding for buffer sizes should be to 16.
    return 16;
}

typedef void (*packb_s32_s8)(int8_t*,
                             int32_t*,
                             const int8_t*,
                             const md_t,
                             const md_t,
                             const md_t,
                             const md_t,
                             md_t*,
                             md_t*);

void
dlp_packb_nr64_s8s8s32os32(int8_t*       pack_b_buffer_s8s8s32o32,
                           int32_t*      pack_b_column_sum,
                           const int8_t* b,
                           const md_t    rs_b,
                           const md_t    cs_b,
                           const md_t    NC,
                           const md_t    KC,
                           md_t*         rs_p,
                           md_t*         cs_p);

#endif // DLP_GEMM_INT8_PACKB_S8
