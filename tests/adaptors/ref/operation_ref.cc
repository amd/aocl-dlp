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

#include "adaptors/ref/operation_ref.hh"
#include <iostream>
#include <stdexcept>
#include <vector>

namespace dlp::testing::classic {

RefOperation::RefOperation()
{
    m_ual_type = dlp::testing::framework::UALType::REF;
}

RefOperation::~RefOperation() = default;

void
RefOperation::addOperations(
    const dlp::testing::framework::OperationParams& params)
{
    for (const auto& param : params) {
        addOperation(param->clone());
    }
}

void
RefOperation::addOperation(
    std::unique_ptr<dlp::testing::framework::IOperationParam> param)
{
    if (!param) {
        return;
    }

    // Clone the parameter for sequence tracking BEFORE releasing it
    m_operation_params.push_back(param->clone());

    // Sort operations by type into separate vectors for efficient processing
    switch (param->getType()) {
        case dlp::testing::framework::OperationType::ElementWise: {
            auto ew_param =
                std::unique_ptr<dlp::testing::framework::ElementWiseParam>(
                    static_cast<dlp::testing::framework::ElementWiseParam*>(
                        param.release()));
            m_elementwise_ops.push_back(std::move(ew_param));
            break;
        }
        case dlp::testing::framework::OperationType::Scale: {
            auto sum_param =
                std::unique_ptr<dlp::testing::framework::ScaleParam>(
                    static_cast<dlp::testing::framework::ScaleParam*>(
                        param.release()));
            m_sum_ops.push_back(std::move(sum_param));
            break;
        }
        case dlp::testing::framework::OperationType::Bias: {
            auto bias_param =
                std::unique_ptr<dlp::testing::framework::BiasParam>(
                    static_cast<dlp::testing::framework::BiasParam*>(
                        param.release()));
            m_bias_ops.push_back(std::move(bias_param));
            break;
        }
        case dlp::testing::framework::OperationType::MatAdd: {
            auto mat_add_param =
                std::unique_ptr<dlp::testing::framework::MatrixAddParam>(
                    static_cast<dlp::testing::framework::MatrixAddParam*>(
                        param.release()));
            m_matrix_add_ops.push_back(std::move(mat_add_param));
            break;
        }
        case dlp::testing::framework::OperationType::MatMul: {
            auto mat_mul_param =
                std::unique_ptr<dlp::testing::framework::MatrixMulParam>(
                    static_cast<dlp::testing::framework::MatrixMulParam*>(
                        param.release()));
            m_matrix_mul_ops.push_back(std::move(mat_mul_param));
            break;
        }
        default:
            throw std::runtime_error("Unsupported operation type");
    }
}

void
RefOperation::finalize()
{
    if (m_finalized) {
        return;
    }

    if (m_operation_params.empty()) {
        m_finalized = true;
        return;
    }

    // Process operations - for reference implementation, we don't need to
    // convert to backend-specific format, but we can organize them for
    // efficient iteration

    if (!m_elementwise_ops.empty()) {
        convertElementWiseOperations();
    }

    if (!m_sum_ops.empty()) {
        convertSumOperations();
    }

    if (!m_bias_ops.empty()) {
        convertBiasOperations();
    }

    if (!m_matrix_add_ops.empty()) {
        convertMatrixAddOperations();
    }

    if (!m_matrix_mul_ops.empty()) {
        convertMatrixMulOperations();
    }

    // Build the sequence vector based on the original order
    buildSequenceVector();

    m_finalized = true;
}

bool
RefOperation::hasNextPostOp() const
{
    return m_current_index < m_operation_params.size();
}

std::optional<RefOperation::PostOpVariant>
RefOperation::getNextPostOp() const
{
    if (!hasNextPostOp()) {
        return std::nullopt;
    }

    const auto& param = m_operation_params[m_current_index];
    m_current_index++;

    // Convert the parameter to the appropriate variant type
    switch (param->getType()) {
        case dlp::testing::framework::OperationType::ElementWise: {
            const auto& ew_param =
                static_cast<const dlp::testing::framework::ElementWiseParam&>(
                    *param);
            return PostOpVariant{ ew_param };
        }
        case dlp::testing::framework::OperationType::Scale: {
            const auto& sum_param =
                static_cast<const dlp::testing::framework::ScaleParam&>(*param);
            return PostOpVariant{ sum_param };
        }
        case dlp::testing::framework::OperationType::Bias: {
            const auto& bias_param =
                static_cast<const dlp::testing::framework::BiasParam&>(*param);
            return PostOpVariant{ bias_param };
        }
        case dlp::testing::framework::OperationType::MatAdd: {
            const auto& mat_add_param =
                static_cast<const dlp::testing::framework::MatrixAddParam&>(
                    *param);
            return PostOpVariant{ mat_add_param };
        }
        case dlp::testing::framework::OperationType::MatMul: {
            const auto& mat_mul_param =
                static_cast<const dlp::testing::framework::MatrixMulParam&>(
                    *param);
            return PostOpVariant{ mat_mul_param };
        }
        default:
            throw std::runtime_error(
                "Unsupported operation type in getNextPostOp");
    }
}

void
RefOperation::resetIterator() const
{
    m_current_index = 0;
}

size_t
RefOperation::getPostOpCount() const
{
    return m_operation_params.size();
}

void
RefOperation::convertElementWiseOperations()
{
    // For reference implementation, we don't need to convert to backend format
    // This method is kept for consistency with the DLP implementation
    // and could be used for any reference-specific preprocessing
}

void
RefOperation::convertSumOperations()
{
    // For reference implementation, we don't need to convert to backend format
    // This method is kept for consistency with the DLP implementation
}

void
RefOperation::convertBiasOperations()
{
    // For reference implementation, we don't need to convert to backend format
    // This method is kept for consistency with the DLP implementation
}

void
RefOperation::convertMatrixAddOperations()
{
    // For reference implementation, we don't need to convert to backend format
    // This method is kept for consistency with the DLP implementation
}

void
RefOperation::convertMatrixMulOperations()
{
    // For reference implementation, we don't need to convert to backend format
    // This method is kept for consistency with the DLP implementation
}

void
RefOperation::buildSequenceVector()
{
    // For reference implementation, we don't need to build a sequence vector
    // like DLP does, but we can log the sequence for debugging

    for (size_t i = 0; i < m_operation_params.size(); ++i) {
        const auto& param = m_operation_params[i];
        std::string opName;

        switch (param->getType()) {
            case dlp::testing::framework::OperationType::ElementWise: {
                const auto& ew_param = static_cast<
                    const dlp::testing::framework::ElementWiseParam&>(*param);
                opName = "ElementWise(";
                switch (ew_param.getOperation()) {
                    case dlp::testing::framework::ElementWiseOperation::Relu:
                        opName += "Relu";
                        break;
                    case dlp::testing::framework::ElementWiseOperation::Prelu:
                        opName += "Prelu";
                        break;
                    case dlp::testing::framework::ElementWiseOperation::
                        Gelu_Tanh:
                        opName += "Gelu_Tanh";
                        break;
                    case dlp::testing::framework::ElementWiseOperation::
                        Gelu_Erf:
                        opName += "Gelu_Erf";
                        break;
                    case dlp::testing::framework::ElementWiseOperation::Clip:
                        opName += "Clip";
                        break;
                    case dlp::testing::framework::ElementWiseOperation::Swish:
                        opName += "Swish";
                        break;
                    case dlp::testing::framework::ElementWiseOperation::Tanh:
                        opName += "Tanh";
                        break;
                    case dlp::testing::framework::ElementWiseOperation::Sigmoid:
                        opName += "Sigmoid";
                        break;
                    default:
                        opName += "Unknown";
                        break;
                }
                opName += ")";
                break;
            }
            case dlp::testing::framework::OperationType::Scale:
                opName = "Scale";
                break;
            case dlp::testing::framework::OperationType::Bias:
                opName = "Bias";
                break;
            case dlp::testing::framework::OperationType::MatAdd:
                opName = "MatrixAdd";
                break;
            case dlp::testing::framework::OperationType::MatMul:
                opName = "MatrixMul";
                break;
            default:
                opName = "Unknown";
                break;
        }
    }
}

} // namespace dlp::testing::classic
