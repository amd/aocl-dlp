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

#ifndef DLP_GEMM_POST_OPS_H
#define DLP_GEMM_POST_OPS_H

#include "classic/aocl_gemm_metadata.h"
#include "dlp_gemm_types.h"

struct dlp_gemm_quant_op_attr_t;
struct dlp_gemm_pre_op_attr_t;

/* Terminal GLU op (GatedSwiglu / GatedSwigluAndMul) translator. This should
 * be the last node in the post-op list, and its expected post_op_list[last]
 * is passed to this function. The GLU op is shape-changing (2I -> I) only
 * supported on BF16 GEMM APIs, and only when the AVX512-BF16 instruction set
 * is available. It is not supported on BF16 F32 fallback path.
 * There is this expectation that glu stor buffer info is part of the
 * first node in the post-op list for O(1) lookup of GLU buffer. Note that
 * post_op_list_start is the start node. */
dlp_clsc_err_t
dlp_gemm_translate_glu_term_op(dlp_metadata_t*   metadata,
                               dlp_gemm_post_op* post_op_list,
                               dlp_gemm_post_op* post_op_list_start,
                               void*             meta_arg,
                               md_t              m,
                               md_t              n);

/* Default post-op translator. */
dlp_clsc_err_t
dlp_gemm_translate_to_post_ops_list(dlp_metadata_t*   metadata,
                                    dlp_gemm_post_op* post_op_list,
                                    void*             scale_buffer,
                                    void*             meta_arg,
                                    md_t              m,
                                    md_t              n);

/* A-side dynamic quantization used by bf16s8/f32s8 APIs. */
dlp_clsc_err_t
dlp_gemm_validate_dynamic_quant_op(const dlp_quant_op_t* a_quant_op,
                                   md_t                  m,
                                   md_t                  k);

/* ADQUANTIZE at post_op_list[0]; validation must run first. */
void
dlp_gemm_fill_adquantize_post_op(const dlp_metadata_t* metadata,
                                 dlp_gemm_post_op*     post_op_list,
                                 void*                 meta_arg);

/* B-side WOQ validation. require_zero_point selects U4 (true) vs S4 (false). */
dlp_clsc_err_t
dlp_gemm_validate_woq_quant_op(const dlp_quant_op_t* b_quant_op,
                               md_t                  n,
                               md_t                  k,
                               bool                  require_zero_point);

/* Copies validated WOQ metadata into the existing pack-kernel attribute. */
void
dlp_gemm_fill_woq_pre_op_attr(struct dlp_gemm_pre_op_attr_t* attr,
                              const dlp_quant_op_t*          b_quant_op,
                              md_t                           n,
                              md_t                           k);

/* Source operand for s8 sym-quant. group_size 0 means full K and is left
 * unchanged. Passes dequant_scale_factors to dlp_gemm_validate_scale_factors,
 * then rejects PER_CHANNEL. Does not check kind or zero-point. */
dlp_clsc_err_t
dlp_gemm_validate_src_quant_op(const dlp_quant_op_t* src_quant_op,
                               md_t                  m,
                               md_t                  k);

/* Weight operand for s8 sym-quant. group_size 0 means full K and is left
 * unchanged. Passes dequant_scale_factors to dlp_gemm_validate_scale_factors,
 * then rejects PER_TOKEN. Does not check kind or zero-point. */
dlp_clsc_err_t
dlp_gemm_validate_weight_quant_op(const dlp_quant_op_t* weight_quant_op,
                                  md_t                  n,
                                  md_t                  k);

/* s8s8 sym-quant and s8s4. Calls the two operand checks, then applies
 * kind, zero-point, the multiple-of-4 group, and A/B agreement. */
dlp_clsc_err_t
dlp_gemm_validate_sym_quant_ops(const dlp_quant_op_t* a_quant_op,
                                const dlp_quant_op_t* b_quant_op,
                                md_t                  m,
                                md_t                  n,
                                md_t                  k);

/* 0 means full K. Returns the smaller of the two sizes. */
md_t
dlp_gemm_get_quant_group_size(md_t a_group_size, md_t b_group_size, md_t k);

/* Copies scale-array pointers into the execution attribute. */
void
dlp_gemm_fill_quant_op_attr(struct dlp_gemm_quant_op_attr_t* attr,
                            const dlp_quant_op_t*            a_quant_op,
                            const dlp_quant_op_t*            b_quant_op,
                            md_t                             group_size,
                            md_t                             k,
                            md_t                             n);

dlp_clsc_err_t
dlp_gemm_translate_to_group_op_list(const dlp_metadata_t* metadata,
                                    dlp_group_op*         group_op,
                                    md_t                  m,
                                    md_t                  n,
                                    md_t                  k);

/* Highest op_code with an entry in classic kernels' post_ops_labels[]
 * dispatch table. Codes above this must be routed through the JIT
 * micro-kernel. BUMP when a new label is added to ALL classic
 * post_ops_labels[] tables.
 *
 * This is a temporary workaround until the JIT kernels are completely
 * supported and the classic kernel dependency is removed. */
#define DLP_CLASSIC_MAX_POST_OP_CODE POST_OPS_SIGMOID

/* Returns true if the list contains any post-op whose op_code exceeds
 * DLP_CLASSIC_MAX_POST_OP_CODE (i.e. one that has no entry in the
 * classic post_ops_labels[] dispatch table and must be served by the
 * JIT micro-kernel). NULL list returns false. */
bool
dlp_gemm_post_op_list_has_jit_only_op(const dlp_gemm_post_op* post_op_list);

/* True if the list contains a shape-changing GLU post-op (GatedSwiglu /
 * GatedSwigluAndMul), which halves the output width via a half-width store.
 * O(1) read of the fact the translator cached on the head node. NULL list
 * returns false. The frame/decorator add the runtime engagement gate (beta/k +
 * split).
 */
bool
dlp_gemm_post_op_list_has_shape_changing_glu(
    const dlp_gemm_post_op* post_op_list);

/*
 * POST_OP_LABEL_LASTK_SAFE_JUMP, POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR,
 * and POST_OPS_DISABLE_LABEL are defined in dlp_compat.h (Section 15).
 */

#endif // DLP_GEMM_POST_OPS_H
