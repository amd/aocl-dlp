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

#pragma once

#include "framework/ual_plan.hh"
#include <oneapi/dnnl/dnnl.hpp>
#include <string>
#include <vector>

namespace dlp::testing::classic {

// Forward declaration
class UalOneDNN;

using namespace dlp::testing::framework;

/**
 * @class OneDNNUalPlan
 * @brief oneDNN backend implementation of IUalPlan
 *
 * prepare() creates the oneDNN matmul primitive (with alpha/beta configured
 * as output scales and sum post-op) and builds the native post-op chain
 * (eltwise, scale) plus the bias PD argument. execute() wraps the bound
 * buffers in oneDNN memory objects and runs the pre-built primitive.
 *
 * Post-ops are mapped to oneDNN's NATIVE primitive_attr / post_ops / bias-arg
 * API so this backend stays an independent correctness oracle. Configurations
 * with no native mapping (GLU, A-quant, GroupScale, WOQ, MatAdd/MatMul,
 * unsupported dtypes) defer to a UAL_NOT_SUPPORTED skip rather than throwing.
 */
class OneDNNUalPlan : public dlp::testing::framework::IUalPlan
{
  public:
    /**
     * @brief Construct with a reference to the owning UalOneDNN instance
     *
     * The plan needs access to the oneDNN engine, stream, and caches.
     */
    explicit OneDNNUalPlan(UalOneDNN& ual);
    ~OneDNNUalPlan() override = default;

    void     prepare() override;
    UALError execute() override;

  private:
    UalOneDNN& m_ual;

    // ─── Post-op dispatch (single extension point) ──────────────────
    // Centralised builder for the oneDNN post_ops chain. Appends the beta
    // sum-post-op first (ordering matters: sum must precede eltwise), then
    // iterates getPostOps() and dispatches per OperationType. Each case is a
    // FAMILY-SEAM that a family agent replaces with real native oneDNN calls.
    // Post-ops MUST go through oneDNN's native primitive_attr / post_ops API to
    // preserve the independent-oracle property of the backend.
    void buildPostOpChain(dnnl::post_ops& po, dnnl::primitive_attr& attr);

    // ─── Bias staging (matmul PD constructor arg, NOT a post_op) ─────
    // Inspects getPostOps() for a Bias entry and, when it maps natively,
    // builds m_bias_md ({1, N} f32, broadcast across M rows) and stages the
    // bias vector into owned m_bias_mem storage, setting m_has_bias so
    // prepare() selects the bias-carrying PD overload. Returns false (and calls
    // markNotSupported()) for sub-cases oneDNN cannot express natively: a bias
    // carrying a scale-factor/zero-point (REF dequantizes (bias-zp)*scale,
    // which DNNL_ARG_BIAS cannot do), a bias that is not the first post-op
    // (oneDNN always applies bias before every post-op), or more than one bias.
    // Must be called BEFORE the matmul primitive_desc is constructed.
    bool stageBias();

    // Eltwise family seam: append one native oneDNN eltwise post-op for the
    // given ElementWiseParam. Advances post_op_index for the appended op (so
    // the slot accounting stays symmetric with appendScalePostOp and any future
    // binary-registering family). Unsupported ops call markNotSupported().
    void appendEltwisePostOp(dnnl::post_ops&         po,
                             const ElementWiseParam& ew,
                             int&                    post_op_index);

    // Scale family seam: append an order-preserving native scale
    // (eltwise_linear for per-tensor, binary_mul[+binary_add] for vector
    // SF/ZP). post_op_index is advanced for every appended op so binary
    // operands register under the correct DNNL_ARG_ATTR_MULTIPLE_POST_OP slot.
    void appendScalePostOp(dnnl::post_ops&   po,
                           const ScaleParam& sp,
                           int&              post_op_index);

    // Pre-built primitive
    dnnl::matmul m_prim;

    // Pre-built memory objects (data handles swapped in execute())
    dnnl::memory m_a_mem;
    dnnl::memory m_b_mem;
    dnnl::memory m_c_mem;
    dnnl::memory m_alpha_mem;

    // ─── Post-op exec-arg storage (survives prepare()→execute()) ─────
    // Backing memory objects for any post-op that needs an execution argument
    // (e.g. binary post-op operands, DST scales). Families populate these in
    // their seam and register the matching DNNL_ARG_* entry into m_exec_args at
    // prepare() time so execute() stays allocation-free.
    std::vector<dnnl::memory> m_postop_mems;

    // Bias support. NOTE: in oneDNN, bias is a matmul PRIMITIVE-DESCRIPTOR
    // CONSTRUCTOR argument (an extra bias md passed to matmul::primitive_desc),
    // NOT a post_op. The bias family seam therefore does not build into
    // buildPostOpChain(); instead it sets m_has_bias + m_bias_md before the PD
    // is constructed so prepare() can pick the bias-carrying PD overload, and
    // binds m_bias_mem into m_exec_args under DNNL_ARG_BIAS.
    dnnl::memory       m_bias_mem;
    dnnl::memory::desc m_bias_md;
    bool               m_has_bias = false;

    // Pre-built execution arguments (zero allocation in execute())
    std::unordered_map<int, dnnl::memory> m_exec_args;

    bool  m_has_alpha_scale = false;
    bool  m_b_blocked       = false;
    float m_alpha_val       = 1.0f;

    // Set when this plan builds a custom post-op attr; when true the standard
    // path MUST NOT use the primitive cache (its key does not capture post-op
    // identity) — the primitive is built fresh instead.
    bool m_uses_postop_attr = false;

    // Deferred "not supported" signal. Some configurations (GLU, A-quant,
    // GroupScale, and any config oneDNN rejects at PD-build time) have no valid
    // native oneDNN mapping. The test harness only SKIPS a case when execute()
    // returns UAL_NOT_SUPPORTED; a throw from prepare() escapes uncaught and is
    // scored as a FAILURE. So prepare() must NOT throw for these — it records
    // the reason here and returns normally, and execute() reports
    // UAL_NOT_SUPPORTED. Families that cannot be supported set this via
    // markNotSupported() from their seam instead of throwing.
    bool        m_not_supported = false;
    std::string m_not_supported_reason;

    void markNotSupported(const std::string& reason)
    {
        m_not_supported        = true;
        m_not_supported_reason = reason;
    }
};

} // namespace dlp::testing::classic
