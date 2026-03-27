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

#include "classic/aocl_bf16_type.h"
#include "classic/aocl_fp16_type.h"
#include "classic/aocl_gemm_post_ops.h"
#include "classic/dlp_base_types.h"
#include "framework/types.hh"

namespace dlp::testing::classic::ref {
void
aocl_gemm_f32f32f32of32_ref(const char      order,
                            const char      transa,
                            const char      transb,
                            const md_t      m,
                            const md_t      n,
                            const md_t      k,
                            float           alpha,
                            const float*    A,
                            int             lda,
                            const float*    B,
                            int             ldb,
                            float           beta,
                            float*          C,
                            int             ldc,
                            dlp_metadata_t* post_ops);

void
aocl_gemm_f16f16f16of16_ref(const char      order,
                            const char      transa,
                            const char      transb,
                            const md_t      m,
                            const md_t      n,
                            const md_t      k,
                            float16         alpha,
                            const float16*  A,
                            int             lda,
                            const float16*  B,
                            int             ldb,
                            float16         beta,
                            float16*        C,
                            int             ldc,
                            dlp_metadata_t* post_ops);

void
aocl_gemm_f32f16f32of32_ref(const char      order,
                            const char      transa,
                            const char      transb,
                            const md_t      m,
                            const md_t      n,
                            const md_t      k,
                            float           alpha,
                            const float*    A,
                            int             lda,
                            const float16*  B,
                            int             ldb,
                            float           beta,
                            float*          C,
                            int             ldc,
                            dlp_metadata_t* post_ops);

void
aocl_gemm_bf16bf16f32of32_ref(const char      order,
                              const char      transa,
                              const char      transb,
                              const md_t      m,
                              const md_t      n,
                              const md_t      k,
                              float           alpha,
                              const bfloat16* A,
                              int             lda,
                              const bfloat16* B,
                              int             ldb,
                              float           beta,
                              float*          C,
                              int             ldc,
                              dlp_metadata_t* post_ops);

void
aocl_gemm_s8s8s32os32_ref(const char      order,
                          const char      transa,
                          const char      transb,
                          const md_t      m,
                          const md_t      n,
                          const md_t      k,
                          int32_t         alpha,
                          const int8_t*   A,
                          int             lda,
                          const int8_t*   B,
                          int             ldb,
                          int32_t         beta,
                          int32_t*        C,
                          int             ldc,
                          dlp_metadata_t* post_ops);

void
aocl_gemm_u8s8s32os32_ref(const char      order,
                          const char      transa,
                          const char      transb,
                          const md_t      m,
                          const md_t      n,
                          const md_t      k,
                          int32_t         alpha,
                          const uint8_t*  A,
                          int             lda,
                          const int8_t*   B,
                          int             ldb,
                          int32_t         beta,
                          int32_t*        C,
                          int             ldc,
                          dlp_metadata_t* post_ops);

void
aocl_gemm_u8s8s32of32_ref(const char      order,
                          const char      transa,
                          const char      transb,
                          const md_t      m,
                          const md_t      n,
                          const md_t      k,
                          int32_t         alpha,
                          const uint8_t*  A,
                          int             lda,
                          const int8_t*   B,
                          int             ldb,
                          int32_t         beta,
                          float*          C,
                          int             ldc,
                          dlp_metadata_t* post_ops);

void
aocl_gemm_s8s8s32of32_ref(const char      order,
                          const char      transa,
                          const char      transb,
                          const md_t      m,
                          const md_t      n,
                          const md_t      k,
                          int32_t         alpha,
                          const int8_t*   A,
                          int             lda,
                          const int8_t*   B,
                          int             ldb,
                          int32_t         beta,
                          float*          C,
                          int             ldc,
                          dlp_metadata_t* post_ops);

void
aocl_gemm_bf16s8s32of32_ref(const char            order,
                            const char            transa,
                            const char            transb,
                            const md_t            m,
                            const md_t            n,
                            const md_t            k,
                            int32_t               alpha,
                            const bfloat16*       A,
                            int                   lda,
                            const int8_t*         B,
                            int                   ldb,
                            int32_t               beta,
                            float*                C,
                            int                   ldc,
                            void*                 a_pre_quant_sf_data,
                            void*                 a_pre_quant_zp_data,
                            void*                 a_post_quant_sf_data,
                            void*                 a_post_quant_zp_data,
                            md_t                  sf_len,
                            md_t                  zp_len,
                            framework::MatrixType sf_type,
                            framework::MatrixType zp_type);

void
aocl_gemm_f32s8s32of32_ref(const char            order,
                           const char            transa,
                           const char            transb,
                           const md_t            m,
                           const md_t            n,
                           const md_t            k,
                           int32_t               alpha,
                           const float*          A,
                           int                   lda,
                           const int8_t*         B,
                           int                   ldb,
                           int32_t               beta,
                           float*                C,
                           int                   ldc,
                           void*                 a_pre_quant_sf_data,
                           void*                 a_pre_quant_zp_data,
                           void*                 a_post_quant_sf_data,
                           void*                 a_post_quant_zp_data,
                           md_t                  sf_len,
                           md_t                  zp_len,
                           framework::MatrixType sf_type,
                           framework::MatrixType zp_type);

void
aocl_gemm_bf16s4f32of32_ref(const char            order,
                            const char            transa,
                            const char            transb,
                            const md_t            m,
                            const md_t            n,
                            const md_t            k,
                            float                 alpha,
                            const bfloat16*       A,
                            int                   lda,
                            const int8_t*         B,
                            int                   ldb,
                            float                 beta,
                            float*                C,
                            int                   ldc,
                            void*                 b_scale_data,
                            md_t                  sf_len,
                            framework::MatrixType sf_type,
                            bool                  reorder_b = false);

/** BF16×U4 ref: zp_type s8 => (u4-zp)*scale; zp_type bf16 => (u4-8)*scale+zp.
 */
void
aocl_gemm_bf16u4f32of32_ref(const char            order,
                            const char            transa,
                            const char            transb,
                            const md_t            m,
                            const md_t            n,
                            const md_t            k,
                            float                 alpha,
                            const bfloat16*       A,
                            int                   lda,
                            const uint8_t*        B,
                            int                   ldb,
                            float                 beta,
                            float*                C,
                            int                   ldc,
                            void*                 b_scale_data,
                            void*                 b_zp_data,
                            md_t                  sf_len,
                            md_t                  zp_len,
                            framework::MatrixType sf_type,
                            framework::MatrixType zp_type,
                            bool                  reorder_b = false);
} // namespace dlp::testing::classic::ref
