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
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

extern "C"
{
#include "classic/aocl_gemm_post_ops.h"
}

namespace dlp::testing::classic {

using namespace dlp::testing::framework;

/**
 * @class DlpUalPlan
 * @brief DLP backend implementation of IUalPlan
 *
 * prepare() builds ONE dlp_metadata_t with ALL fields (post-ops + quant),
 * resolves the dispatch function pointer, and pre-casts alpha/beta.
 * execute() calls the pre-resolved function with pre-built metadata.
 */
class DlpUalPlan : public dlp::testing::framework::IUalPlan
{
  public:
    DlpUalPlan();
    ~DlpUalPlan();

    void     prepare() override;
    UALError execute() override;

    dlp_metadata_t* getMetadata() const { return m_metadata; }

    void setGroupScale(std::unique_ptr<GroupScaleParam> param)
    {
        m_group_scale = std::move(param);
    }

    bool hasGroupScale() const { return m_group_scale != nullptr; }

  private:
    // Pre-built metadata (ONE struct for everything)
    dlp_metadata_t* m_metadata = nullptr;

    // Pre-resolved dispatch function
    // Takes (a_ptr, lda, b_ptr, ldb, c_ptr, ldc) - all other args captured
    std::function<void(void*, md_t, void*, md_t, void*, md_t)> m_dispatch;

    // Pre-resolved dispatch state
    char m_layout_char     = 'r';
    char m_transA_char     = 'n';
    char m_transB_char     = 'n';
    char m_transB_resolved = 'n';

    // Pre-cast alpha/beta. FP16 forms live in the base class (IUalPlan)
    // and are populated by setAlpha/setBeta via f32_to_fp16 once at
    // configuration time — no extra conversion in prepare().
    float   m_alpha_f32 = 1.0f;
    float   m_beta_f32  = 0.0f;
    int32_t m_alpha_s32 = 1;
    int32_t m_beta_s32  = 0;

    // Encoded type combo for dispatch
    uint64_t m_type_code = 0;

    // Typed post-op vectors (filled during prepare from m_post_ops)
    std::vector<std::unique_ptr<dlp::testing::framework::ElementWiseParam>>
        m_elementwise_ops;
    std::vector<std::unique_ptr<dlp::testing::framework::ScaleParam>>
        m_scale_ops;
    std::vector<std::unique_ptr<dlp::testing::framework::BiasParam>> m_bias_ops;
    std::vector<std::unique_ptr<dlp::testing::framework::MatrixAddParam>>
        m_matrix_add_ops;
    std::vector<std::unique_ptr<dlp::testing::framework::MatrixMulParam>>
        m_matrix_mul_ops;

    // Cleanup metadata
    void cleanupMetadata();

    // Post-op conversion (moved from DlpOperation)
    void convertElementWiseOperations();
    void convertScaleOperations();
    void convertBiasOperations();
    void convertMatrixAddOperations();
    void convertMatrixMulOperations();
    void buildSequenceVector();

    // Quant conversion (moved from DlpOperation)
    void convertA_QuantOperations();
    void convertB_QuantOperations();
    void convertWOQOperations();
    void convertGroupScaleOperations();

    // Helpers
    void*                    convertMatrixToPtr(const Matrix& matrix);
    static DLP_TYPE          getStorageType(MatrixType type);
    static DLP_ELT_ALGO_TYPE getElementWiseAlgoType(ElementWiseOperation op);
    static DLP_POST_OP_TYPE  getPostOpType(OperationType type);

    // Type encoding (from UalDlp)
    template<MatrixType A, MatrixType B, MatrixType C, MatrixType Acc>
    static constexpr uint64_t encodeTypes()
    {
        return (static_cast<uint64_t>(A) << 48)
               | (static_cast<uint64_t>(B) << 32)
               | (static_cast<uint64_t>(C) << 16) | static_cast<uint64_t>(Acc);
    }

    // Group-level symmetric quantization (group_scale)
    std::unique_ptr<GroupScaleParam> m_group_scale;

    // Broadcast buffers for scalar scale factors (kept alive for metadata
    // lifetime). When scale_factor_len=1 the kernel still indexes per-channel,
    // so we replicate the scalar to the expected buffer size.
    std::vector<uint8_t> m_broadcast_a_scale;
    std::vector<uint8_t> m_broadcast_b_scale;

    static uint64_t encodeTypes(MatrixType a,
                                MatrixType b,
                                MatrixType c,
                                MatrixType acc)
    {
        return (static_cast<uint64_t>(a) << 48)
               | (static_cast<uint64_t>(b) << 32)
               | (static_cast<uint64_t>(c) << 16) | static_cast<uint64_t>(acc);
    }
};

} // namespace dlp::testing::classic
