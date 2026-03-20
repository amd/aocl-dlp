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

#ifndef DLP_GEMM_REORDER_BF16_H
#define DLP_GEMM_REORDER_BF16_H

#include "classic/aocl_bf16_type.h"
#include "dlp_gemm_types.h"
#include "runtime/dlp_runtime.h"

void
dlp_packb_nr64_bf16bf16f32of32_reference(bfloat16*       pack_b,
                                         const bfloat16* b,
                                         const md_t      rs_b,
                                         const md_t      cs_b,
                                         const md_t      NC,
                                         const md_t      KC,
                                         md_t*           rs_p,
                                         md_t*           cs_p);

void
dlp_unpackb_nr64_bf16bf16f32of32_reference(bfloat16*  b,
                                           bfloat16*  unpack_b_buffer,
                                           const md_t NC,
                                           const md_t KC,
                                           md_t       rs_b,
                                           md_t       cs_b);

void
dlp_unreorderb_nr64_bf16bf16f32of32_reference(dlp_gemm_obj_t*  b_reorder,
                                              dlp_gemm_obj_t*  b_unreorder,
                                              dlp_rntm_t*      rntm,
                                              dlp_gemm_cntx_t* lcntx);

void
dlp_reorderb_nr64_bf16bf16f32of32_reference(dlp_gemm_obj_t*  b,
                                            dlp_gemm_obj_t*  b_reorder,
                                            dlp_rntm_t*      rntm,
                                            dlp_gemm_cntx_t* lcntx);

void
dlp_reorderb_nr64_bf16bf16f32of32(dlp_gemm_obj_t*  b,
                                  dlp_gemm_obj_t*  b_reorder,
                                  dlp_rntm_t*      rntm,
                                  dlp_gemm_cntx_t* lcntx);

void
dlp_reorderb_nr64_bf16s4f32of32(dlp_gemm_obj_t*  b,
                                dlp_gemm_obj_t*  b_reorder,
                                dlp_rntm_t*      rntm,
                                dlp_gemm_cntx_t* lcntx);

void
dlp_reorderb_mxp_nr64_f32obf16(dlp_gemm_obj_t*  b,
                               dlp_gemm_obj_t*  b_reorder,
                               dlp_rntm_t*      rntm,
                               dlp_gemm_cntx_t* lcntx);

void
dlp_unreorderb_nr64_bf16bf16f32of32(dlp_gemm_obj_t*  b,
                                    dlp_gemm_obj_t*  b_reorder,
                                    dlp_rntm_t*      rntm,
                                    dlp_gemm_cntx_t* lcntx);

#endif // DLP_GEMM_REORDER_H
