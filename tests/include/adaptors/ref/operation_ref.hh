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
#include <optional>
#include <variant>
#include <vector>

namespace dlp::testing::classic {

using dlp::testing::framework::BiasParam;
using dlp::testing::framework::ElementWiseParam;
using dlp::testing::framework::IOperation;
using dlp::testing::framework::IOperationParam;
using dlp::testing::framework::MatrixAddParam;
using dlp::testing::framework::MatrixMulParam;
using dlp::testing::framework::OperationParams;
using dlp::testing::framework::ScaleParam;

// Forward declaration for friend class
class UalRef;

/**
 * @class RefOperation
 * @brief Reference implementation of post-operations with iterator pattern
 *
 * This class implements the IOperation interface for the reference backend,
 * providing an iterator-style interface for processing post-operations.
 * The iterator pattern allows for flexible processing of operations either
 * on-demand (yield and use) or batch collection into vectors.
 */
class RefOperation : public IOperation
{
  public:
    /**
     * @brief Variant type for holding different post-operation parameters
     */
    using PostOpVariant = std::variant<ElementWiseParam,
                                       ScaleParam,
                                       BiasParam,
                                       MatrixAddParam,
                                       MatrixMulParam>;

  private:
    bool m_finalized = false;

    // Iterator state for getNextPostOp/hasNextPostOp
    mutable size_t m_current_index = 0;

    // Collections for different operation types (filled during addOperation)
    std::vector<std::unique_ptr<dlp::testing::framework::ElementWiseParam>>
        m_elementwise_ops;
    std::vector<std::unique_ptr<dlp::testing::framework::ScaleParam>> m_sum_ops;
    std::vector<std::unique_ptr<dlp::testing::framework::BiasParam>> m_bias_ops;
    std::vector<std::unique_ptr<dlp::testing::framework::MatrixAddParam>>
        m_matrix_add_ops;
    std::vector<std::unique_ptr<dlp::testing::framework::MatrixMulParam>>
        m_matrix_mul_ops;

    // Helper methods for batch conversion (called once in finalize)
    void convertElementWiseOperations();
    void convertSumOperations();
    void convertBiasOperations();
    void convertMatrixAddOperations();
    void convertMatrixMulOperations();
    void buildSequenceVector();

  public:
    RefOperation();
    ~RefOperation();

    // IOperation interface implementation
    void addOperations(
        const dlp::testing::framework::OperationParams& params) override;
    void addOperation(std::unique_ptr<dlp::testing::framework::IOperationParam>
                          param) override;
    void finalize() override;

    /**
     * @brief Check if there are more post-operations available
     *
     * @return true if there are more operations to process, false otherwise
     *
     * This method provides the iterator pattern capability to check if more
     * operations are available without advancing the iterator state.
     */
    bool hasNextPostOp() const;

    /**
     * @brief Get the next post-operation in the sequence
     *
     * @return std::optional<PostOpVariant> The next post-operation wrapped in
     * optional, or std::nullopt if no more operations are available
     *
     * This method implements the iterator pattern by returning the next
     * post-operation in the sequence. It advances the internal iterator state
     * and returns the operation as a variant. The method can be used in a
     * yield-and-use pattern:
     *
     * Example usage:
     * ```cpp
     * while (operation->hasNextPostOp()) {
     *     auto postOp = operation->getNextPostOp();
     *     if (postOp) {
     *         // Process the post-operation
     *         std::visit([&](const auto& op) {
     *             // Handle specific operation type
     *         }, *postOp);
     *     }
     * }
     * ```
     *
     * Or collect into a vector:
     * ```cpp
     * std::vector<PostOpVariant> postOps;
     * while (operation->hasNextPostOp()) {
     *     auto postOp = operation->getNextPostOp();
     *     if (postOp) {
     *         postOps.push_back(*postOp);
     *     }
     * }
     * ```
     */
    std::optional<PostOpVariant> getNextPostOp() const;

    /**
     * @brief Reset the iterator to the beginning
     *
     * This method resets the internal iterator state to the beginning,
     * allowing for multiple passes through the post-operations.
     */
    void resetIterator() const;

    /**
     * @brief Get the total number of post-operations
     *
     * @return size_t The total number of post-operations in the sequence
     */
    size_t getPostOpCount() const;

    // Friend class - UalRef can access private members and methods
    friend class UalRef;
};

} // namespace dlp::testing::classic
