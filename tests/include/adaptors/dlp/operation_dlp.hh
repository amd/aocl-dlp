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

#include "framework/operation.hh"
#include <memory>
#include <vector>

extern "C"
{
#include "classic/aocl_gemm_post_ops.h"
}

namespace dlp::testing::classic {

using namespace dlp::testing::framework;

// Forward declaration for friend class
class UalDlp;

/**
 * @class DlpOperation
 * @brief DLP-specific implementation of post-operations
 *
 * This class implements the IOperation interface for the DLP backend,
 * converting high-level operation parameters to DLP's dlp_metadata_t structure.
 *
 * Supports multiple operations of the same type efficiently by:
 * 1. Collecting operations in vectors during addOperation()
 * 2. Allocating arrays once in finalize() based on actual counts
 * 3. Keeping toAoclPostOp() lightweight (just returns pointer)
 */
class DlpOperation : public dlp::testing::framework::IOperation
{
  private:
    std::shared_ptr<dlp_metadata_t> m_postops;
    bool                            m_finalized = false;

    // Collections for different operation types (filled during addOperation)
    std::vector<std::unique_ptr<dlp::testing::framework::ElementWiseParam>>
        m_elementwise_ops;
    std::vector<std::unique_ptr<dlp::testing::framework::ScaleParam>> m_sum_ops;
    std::vector<std::unique_ptr<dlp::testing::framework::BiasParam>> m_bias_ops;
    std::vector<std::unique_ptr<dlp::testing::framework::MatrixAddParam>>
        m_matrix_add_ops;
    std::vector<std::unique_ptr<dlp::testing::framework::MatrixMulParam>>
        m_matrix_mul_ops;

    // Friend class - UalDlp can access private members and methods
    friend class UalDlp;

    // Private method to get the backend-specific postops structure
    dlp_metadata_t* toAoclPostOp() { return m_postops.get(); }

    // Helper methods for batch conversion (called once in finalize)
    void convertElementWiseOperations();
    void convertSumOperations();
    void convertBiasOperations();
    void convertMatrixAddOperations();
    void convertMatrixMulOperations();
    void buildSequenceVector();

    // Helper to convert Matrix to appropriate DLP format
    void*    convertMatrixToPtr(const dlp::testing::framework::Matrix& matrix);
    DLP_TYPE getStorageType(dlp::testing::framework::MatrixType type);
    DLP_ELT_ALGO_TYPE getElementWiseAlgoType(
        dlp::testing::framework::ElementWiseOperation op);
    DLP_POST_OP_TYPE getPostOpType(dlp::testing::framework::OperationType type);

  public:
    DlpOperation();
    ~DlpOperation();

    // IOperation interface implementation
    void addOperations(
        const dlp::testing::framework::OperationParams& params) override;
    void addOperation(std::unique_ptr<dlp::testing::framework::IOperationParam>
                          param) override;
    void finalize() override;
};

} // namespace dlp::testing::classic
