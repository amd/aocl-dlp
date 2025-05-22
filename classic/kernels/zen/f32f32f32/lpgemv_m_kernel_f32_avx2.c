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

#include <immintrin.h>

#include "kernels/dlp_kernels.h"
#include "lpgemm_kernel_macros_f32_avx2.h"

void
lpgemv_m_one_kernel_f32_avx2_ker_ft(const md_t            n0,
                                    const md_t            k,
                                    const float*          a,
                                    const md_t            rs_a,
                                    const md_t            cs_a,
                                    const AOCL_MEMORY_TAG mtag_a,
                                    const float*          b,
                                    const md_t            rs_b,
                                    const md_t            cs_b,
                                    const AOCL_MEMORY_TAG mtag_b,
                                    float*                c,
                                    const md_t            rs_c,
                                    const md_t            cs_c,
                                    const float           alpha,
                                    const float           beta,
                                    const md_t            NR,
                                    const md_t            KC,
                                    const md_t            n_sub_updated,
                                    const md_t            jc_cur_loop_rem,
                                    lpgemm_post_op*       post_op_list,
                                    lpgemm_post_op_attr*  post_op_attr)
{
    // TODO: Created dummy function as place holder.
    // AVX2 varient wil be implemented in next commits.
    // Code will take LPGEMM path for LPGEMV in AVX2 env
}
