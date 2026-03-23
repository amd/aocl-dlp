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

#include "dlp_gemm_jit_bf16.h"
#include "libjit_c_connector.h"

#ifdef __cplusplus
extern "C"
{
#endif

    static dlp_gemm_jit* dlp_gemm_jit_objs[DLP_GEMM_BF16_MR][DLP_GEMM_BF16_NR];

    void dlp_gemm_get_jit_kernel_inplace(dlp_gemm_jit_inputs_t* params,
                                         void*                  buffer,
                                         md_t                   bufferSize)
    {
        md_t m_idx                      = (params->MR) % DLP_GEMM_BF16_MR;
        md_t n_idx                      = (params->NR) / NUM_F32_ELEMS_PER_ZMM;
        dlp_gemm_jit_objs[m_idx][n_idx] = new dlp_gemm_jit(buffer, bufferSize);
        dlp_gemm_jit_objs[m_idx][n_idx]->generate_kernel(params);
    }

    void* dlp_gemm_get_jit_code(dlp_gemm_jit_inputs_t* params)
    {
        md_t m_idx = (params->MR) % DLP_GEMM_BF16_MR;
        md_t n_idx = (params->NR) / NUM_F32_ELEMS_PER_ZMM;
        return ((void*)dlp_gemm_jit_objs[m_idx][n_idx]->get_code());
    }

    md_t dlp_gemm_get_kernel_size(dlp_gemm_jit_inputs_t* params)
    {
        md_t m_idx = (params->MR) % DLP_GEMM_BF16_MR;
        md_t n_idx = (params->NR) / NUM_F32_ELEMS_PER_ZMM;
        return dlp_gemm_jit_objs[m_idx][n_idx]->get_size();
    }
#ifdef __cplusplus
}
#endif
