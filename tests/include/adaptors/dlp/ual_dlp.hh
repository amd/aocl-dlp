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

#include "framework/ual.hh"
#include "framework/ual_plan.hh"

namespace dlp::testing::classic {

using dlp::testing::framework::BatchGroup;
using dlp::testing::framework::GroupScaleParam;
using dlp::testing::framework::IUal;
using dlp::testing::framework::Matrix;
using dlp::testing::framework::MatrixLayout;
using dlp::testing::framework::MatrixType;
using dlp::testing::framework::PreparedBatchGemmArgs;
using dlp::testing::framework::UALError;
using dlp::testing::framework::UALType;

/**
 * @class UalDlp
 * @brief Deep Learning Primitives implementation of UAL
 *
 * This class implements the IUAL interface using the DLP backend library,
 * providing optimized matrix operations for various data types.
 */
class UalDlp : public IUal
{
  public:
    /**
     * @brief Constructor
     *
     * Initializes a DLP-based UAL implementation.
     */
    UalDlp();

    /**
     * @brief Create a DLP execution plan
     * @return Unique pointer to a DlpUalPlan
     */
    std::unique_ptr<dlp::testing::framework::IUalPlan> createPlan() override;

    /**
     * @brief Get the UAL type
     *
     * @return UALType The UAL implementation type (DLP)
     */
    UALType getUALType() const override;

    // Helper function to encode type combinations for switch statement
    template<MatrixType A_type,
             MatrixType B_type,
             MatrixType C_type,
             MatrixType acc_type>
    static constexpr uint64_t encode_types()
    {
        return (static_cast<uint64_t>(A_type) << 48)
               | (static_cast<uint64_t>(B_type) << 32)
               | (static_cast<uint64_t>(C_type) << 16)
               | static_cast<uint64_t>(acc_type);
    }

    static uint64_t encode_types(MatrixType a,
                                 MatrixType b,
                                 MatrixType c,
                                 MatrixType acc)
    {
        return (static_cast<uint64_t>(a) << 48)
               | (static_cast<uint64_t>(b) << 32)
               | (static_cast<uint64_t>(c) << 16) | static_cast<uint64_t>(acc);
    }

    /**
     * @brief Get string representation of UAL type
     *
     * @param type UAL type
     * @return std::string Human-readable name
     */
    std::string toString(UALType type) override;

    /**
     * @brief Reorder matrix data to specified accumulation type
     *
     * @param in Input matrix to reorder
     * @param out Output matrix to store reordered data
     * @param A_type Type of matrix A in GEMM context
     * @param B_type Type of matrix B in GEMM context
     * @param C_type Type of matrix C in GEMM context
     * @param accType Accumulation type
     * @param group_scale Optional symmetric-quantization group-scale
     * parameters; when non-null selects the sym_quant reorder path
     * @return UALError Error code indicating success or failure
     */
    UALError reorder(const Matrix&          in,
                     Matrix&                out,
                     MatrixType             A_type,
                     MatrixType             B_type,
                     MatrixType             C_type,
                     MatrixType             accType,
                     const GroupScaleParam* group_scale = nullptr) override;

    UALError batch_gemm(std::vector<BatchGroup>& groups,
                        MatrixType               accType) override;

    /**
     * @brief Prepare backend-specific metadata for batch GEMM
     *
     * Extracts or creates dlp_metadata_t objects for each group and stores
     * them in the prepared arguments structure. This separates metadata
     * preparation overhead from the hot execution path.
     *
     * @param args Prepared batch GEMM arguments to populate with DLP metadata
     */
    void batch_prepare_metadata(PreparedBatchGemmArgs& args) override;

    /**
     * @brief Perform optimized batch GEMM using pre-prepared metadata
     *
     * This optimized variant eliminates allocation and casting overhead by
     * using pre-prepared metadata. Call batch_prepare_metadata() once before
     * using this method.
     *
     * NOTE: Assumes batch_prepare_metadata() was called successfully.
     * No redundant validation for maximum performance.
     *
     * @param prepared Pre-prepared batch GEMM arguments with DLP metadata
     * @return UALError Error code indicating success or failure
     */
    UALError batch_gemm(const PreparedBatchGemmArgs& prepared) override;
};

} // namespace dlp::testing::classic
