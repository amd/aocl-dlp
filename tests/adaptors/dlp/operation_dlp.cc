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

#include "adaptors/dlp/operation_dlp.hh"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace dlp::testing::classic {

DlpOperation::DlpOperation()
{
    m_ual_type = dlp::testing::framework::UALType::DLP;
    m_postops  = std::make_unique<aocl_post_op>();
    std::memset(m_postops.get(), 0, sizeof(aocl_post_op));
}

DlpOperation::~DlpOperation()
{
    // Clean up manually allocated memory for zero points in SUM operations
    if (m_postops && m_postops->sum) {
        for (size_t i = 0; i < m_sum_ops.size(); ++i) {
            // Check if this is a dynamically allocated zero point
            // (for SCALE operations where we allocated it ourselves)
            if (m_postops->sum[i].zero_point
                && m_postops->sum[i].zero_point_len == 1
                && m_postops->sum[i].zp_stor_type == AOCL_GEMM_INT8) {
                // Check if this was allocated by us (no zero point provided
                // originally)
                if (i < m_sum_ops.size() && !m_sum_ops[i]->hasZeroPoint()) {
                    // This was our dynamically allocated zero point
                    delete static_cast<int8_t*>(m_postops->sum[i].zero_point);
                }
            }
        }
    }
}

void
DlpOperation::addOperations(
    const dlp::testing::framework::OperationParams& params)
{
    if (m_finalized) {
        throw std::runtime_error("Cannot add operations after finalization");
    }

    for (const auto& param : params) {
        addOperation(param->clone());
    }
}

void
DlpOperation::addOperation(
    std::unique_ptr<dlp::testing::framework::IOperationParam> param)
{
    if (m_finalized) {
        throw std::runtime_error("Cannot add operations after finalization");
    }

    if (!param) {
        return;
    }

    // Clone the parameter for sequence tracking BEFORE releasing it
    m_operation_params.push_back(param->clone());

    // Sort operations by type into separate vectors for efficient array
    // allocation
    switch (param->getType()) {
        case dlp::testing::framework::OperationType::ElementWise: {
            auto ew_param =
                std::unique_ptr<dlp::testing::framework::ElementWiseParam>(
                    static_cast<dlp::testing::framework::ElementWiseParam*>(
                        param.release()));
            m_elementwise_ops.push_back(std::move(ew_param));
            break;
        }
        case dlp::testing::framework::OperationType::Sum: {
            auto sum_param = std::unique_ptr<dlp::testing::framework::SumParam>(
                static_cast<dlp::testing::framework::SumParam*>(
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
DlpOperation::finalize()
{
    if (m_finalized) {
        return;
    }

    if (m_operation_params.empty()) {
        m_finalized = true;
        return;
    }

    // Allocate arrays for each operation type based on actual counts
    // This happens only once, avoiding reallocations

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

void
DlpOperation::convertElementWiseOperations()
{
    size_t count = m_elementwise_ops.size();
    if (count == 0)
        return;

    // Allocate array once with exact size
    m_postops->eltwise = new aocl_post_op_eltwise[count];
    std::memset(m_postops->eltwise, 0, count * sizeof(aocl_post_op_eltwise));
    m_postops->num_eltwise = count;

    // Fill the array
    for (size_t i = 0; i < count; ++i) {
        const auto& param = *m_elementwise_ops[i];

        m_postops->eltwise[i].algo.algo_type =
            getElementWiseAlgoType(param.getOperation());

        // Set alpha and beta if provided
        if (param.hasAlpha()) {
            m_postops->eltwise[i].algo.alpha =
                convertMatrixToPtr(*param.getAlpha());
        }
        if (param.hasBeta()) {
            m_postops->eltwise[i].algo.beta =
                convertMatrixToPtr(*param.getBeta());
        }
    }
}

void
DlpOperation::convertSumOperations()
{
    size_t count = m_sum_ops.size();
    if (count == 0)
        return;

    // Allocate array once with exact size
    m_postops->sum = new aocl_post_op_sum[count];
    std::memset(m_postops->sum, 0, count * sizeof(aocl_post_op_sum));

    // Fill the array
    for (size_t i = 0; i < count; ++i) {
        const auto& param = *m_sum_ops[i];

        m_postops->sum[i].is_power_of_2 = param.getIsPowerOf2();

        // Initialize all fields to safe defaults
        m_postops->sum[i].scale_factor     = nullptr;
        m_postops->sum[i].scale_factor_len = 0;
        m_postops->sum[i].sf_stor_type = AOCL_PARAMS_STORAGE_TYPES::NULLTYPE;

        m_postops->sum[i].zero_point     = nullptr;
        m_postops->sum[i].zero_point_len = 0;
        m_postops->sum[i].zp_stor_type   = AOCL_PARAMS_STORAGE_TYPES::NULLTYPE;

        m_postops->sum[i].buff = nullptr;

        // Set scale factor if provided
        if (param.hasScaleFactor()) {
            m_postops->sum[i].scale_factor =
                convertMatrixToPtr(*param.getScaleFactor());
            m_postops->sum[i].scale_factor_len =
                param.getScaleFactor()->getCols();
            // For SCALE operations, we need to set storage types
            m_postops->sum[i].sf_stor_type =
                getStorageType(param.getScaleFactor()->getMatrixType());
        }

        // Set zero point if provided
        if (param.hasZeroPoint()) {
            m_postops->sum[i].zero_point =
                convertMatrixToPtr(*param.getZeroPoint());
            m_postops->sum[i].zero_point_len = param.getZeroPoint()->getCols();
            // For SCALE operations, we need to set storage types
            m_postops->sum[i].zp_stor_type =
                getStorageType(param.getZeroPoint()->getMatrixType());
        } else {
            // Based on DLP library validation, if we have a scale factor, we
            // need zero point
            if (param.hasScaleFactor()) {
                // Allocate and set default zero point for SCALE operations
                int8_t* zero_point_data          = new int8_t(0);
                m_postops->sum[i].zero_point     = zero_point_data;
                m_postops->sum[i].zero_point_len = 1;
                m_postops->sum[i].zp_stor_type   = AOCL_GEMM_INT8;
            } else {
                // Regular SUM operations without scale factors need a buffer
                // (tensor to add) Since we don't have a buffer, this operation
                // is invalid We should either provide a buffer or reject such
                // operations

                // For now, let's reject SUM operations without scale factors
                throw std::runtime_error(
                    "SUM operations without scale factors (and without "
                    "buffers) are not supported. Use SCALE operations "
                    "instead.");
            }
        }
    }
}

void
DlpOperation::convertBiasOperations()
{
    size_t count = m_bias_ops.size();
    if (count == 0)
        return;

    // Allocate array once with exact size
    m_postops->bias = new aocl_post_op_bias[count];
    std::memset(m_postops->bias, 0, count * sizeof(aocl_post_op_bias));

    // Fill the array
    for (size_t i = 0; i < count; ++i) {
        const auto& param = *m_bias_ops[i];

        m_postops->bias[i].bias = convertMatrixToPtr(param.getBias());
        m_postops->bias[i].stor_type =
            getStorageType(param.getBias().getMatrixType());
    }
}

void
DlpOperation::convertMatrixAddOperations()
{
    size_t count = m_matrix_add_ops.size();
    if (count == 0)
        return;

    // Allocate array once with exact size
    m_postops->matrix_add = new aocl_post_op_matrix_add[count];
    std::memset(m_postops->matrix_add, 0,
                count * sizeof(aocl_post_op_matrix_add));

    // Fill the array
    for (size_t i = 0; i < count; ++i) {
        const auto& param = *m_matrix_add_ops[i];

        m_postops->matrix_add[i].matrix = convertMatrixToPtr(param.getMatrix());
        m_postops->matrix_add[i].ldm = param.getMatrix().getLeadingDimension();
        m_postops->matrix_add[i].stor_type =
            getStorageType(param.getMatrix().getMatrixType());

        if (param.hasScaleFactor()) {
            m_postops->matrix_add[i].scale_factor =
                convertMatrixToPtr(*param.getScaleFactor());
            m_postops->matrix_add[i].scale_factor_len =
                param.getScaleFactor()->getCols();
        }
    }
}

void
DlpOperation::convertMatrixMulOperations()
{
    size_t count = m_matrix_mul_ops.size();
    if (count == 0)
        return;

    // Allocate array once with exact size
    m_postops->matrix_mul = new aocl_post_op_matrix_mul[count];
    std::memset(m_postops->matrix_mul, 0,
                count * sizeof(aocl_post_op_matrix_mul));

    // Fill the array
    for (size_t i = 0; i < count; ++i) {
        const auto& param = *m_matrix_mul_ops[i];

        m_postops->matrix_mul[i].matrix = convertMatrixToPtr(param.getMatrix());
        m_postops->matrix_mul[i].ldm = param.getMatrix().getLeadingDimension();
        m_postops->matrix_mul[i].stor_type =
            getStorageType(param.getMatrix().getMatrixType());

        if (param.hasScaleFactor()) {
            m_postops->matrix_mul[i].scale_factor =
                convertMatrixToPtr(*param.getScaleFactor());
            m_postops->matrix_mul[i].scale_factor_len =
                param.getScaleFactor()->getCols();
        }
    }
}

void
DlpOperation::buildSequenceVector()
{
    // Build sequence vector based on original operation order
    std::vector<AOCL_POST_OP_TYPE> sequence;
    sequence.reserve(m_operation_params.size());

    // Track array indices for each operation type
    size_t eltwise_idx = 0, sum_idx = 0, bias_idx = 0, matrix_add_idx = 0,
           matrix_mul_idx = 0;

    for (const auto& param : m_operation_params) {
        switch (param->getType()) {
            case dlp::testing::framework::OperationType::ElementWise:
                sequence.push_back(ELTWISE);
                eltwise_idx++;
                break;
            case dlp::testing::framework::OperationType::Sum: {
                const auto& sumParam =
                    static_cast<const dlp::testing::framework::SumParam&>(
                        *param);
                // Use SCALE in sequence if we have a scale factor, following
                // post_ops_combinations.c example
                if (sumParam.hasScaleFactor()) {
                    sequence.push_back(SCALE);
                } else {
                    sequence.push_back(SUM);
                }
                sum_idx++;
                break;
            }
            case dlp::testing::framework::OperationType::Bias:
                sequence.push_back(BIAS);
                bias_idx++;
                break;
            case dlp::testing::framework::OperationType::MatAdd:
                sequence.push_back(MATRIX_ADD);
                matrix_add_idx++;
                break;
            case dlp::testing::framework::OperationType::MatMul:
                sequence.push_back(MATRIX_MUL);
                matrix_mul_idx++;
                break;
            default:
                throw std::runtime_error(
                    "Unsupported operation type in sequence");
        }
    }

    // Set sequence information
    m_postops->seq_length = sequence.size();
    if (!sequence.empty()) {
        m_postops->seq_vector = new AOCL_POST_OP_TYPE[sequence.size()];
        std::memcpy(m_postops->seq_vector, sequence.data(),
                    sequence.size() * sizeof(AOCL_POST_OP_TYPE));
    }
}

void*
DlpOperation::convertMatrixToPtr(const dlp::testing::framework::Matrix& matrix)
{
    return matrix.getData();
}

AOCL_PARAMS_STORAGE_TYPES
DlpOperation::getStorageType(dlp::testing::framework::MatrixType type)
{
    switch (type) {
        case dlp::testing::framework::MatrixType::f32:
            return AOCL_GEMM_F32;
        case dlp::testing::framework::MatrixType::bf16:
            return AOCL_GEMM_BF16;
        case dlp::testing::framework::MatrixType::s8:
            return AOCL_GEMM_INT8;
        case dlp::testing::framework::MatrixType::u8:
            return AOCL_GEMM_UINT8;
        case dlp::testing::framework::MatrixType::s32:
            return AOCL_GEMM_INT32;
        default:
            return NULLTYPE;
    }
}

AOCL_ELT_ALGO_TYPE
DlpOperation::getElementWiseAlgoType(
    dlp::testing::framework::ElementWiseOperation op)
{
    switch (op) {
        case dlp::testing::framework::ElementWiseOperation::Relu:
            return RELU;
        case dlp::testing::framework::ElementWiseOperation::Prelu:
            return PRELU;
        case dlp::testing::framework::ElementWiseOperation::Gelu_Tanh:
            return GELU_TANH;
        case dlp::testing::framework::ElementWiseOperation::Gelu_Erf:
            return GELU_ERF;
        case dlp::testing::framework::ElementWiseOperation::Clip:
            return CLIP;
        case dlp::testing::framework::ElementWiseOperation::Swish:
            return SWISH;
        case dlp::testing::framework::ElementWiseOperation::Tanh:
            return TANH;
        case dlp::testing::framework::ElementWiseOperation::Sigmoid:
            return SIGMOID;
        default:
            throw std::runtime_error("Unsupported element-wise operation");
    }
}

AOCL_POST_OP_TYPE
DlpOperation::getPostOpType(dlp::testing::framework::OperationType type)
{
    switch (type) {
        case dlp::testing::framework::OperationType::Sum:
            return SUM;
        case dlp::testing::framework::OperationType::ElementWise:
            return ELTWISE;
        case dlp::testing::framework::OperationType::Bias:
            return BIAS;
        case dlp::testing::framework::OperationType::Scale:
            return SCALE;
        case dlp::testing::framework::OperationType::MatAdd:
            return MATRIX_ADD;
        case dlp::testing::framework::OperationType::MatMul:
            return MATRIX_MUL;
        default:
            throw std::runtime_error("Unsupported operation type");
    }
}

} // namespace dlp::testing::classic
