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
    m_postops  = std::make_shared<dlp_metadata_t>();
    std::memset(m_postops.get(), 0, sizeof(dlp_metadata_t));
}

DlpOperation::~DlpOperation()
{
    if (m_postops) {
        // Clean up manually allocated memory for zero points in Scale
        // operations
        if (m_postops->scale) {
            for (std::size_t i = 0; i < m_scale_ops.size(); ++i) {
                // Check if this is a dynamically allocated zero point
                // (for SCALE operations where we allocated it ourselves)
                if (m_postops->scale[i].zp
                    && m_postops->scale[i].zp->zero_point_len == 1
                    && m_postops->scale[i].zp->zero_point_type == DLP_S8) {
                    // Check if this was allocated by us (no zero point provided
                    // originally)
                    if (i < m_scale_ops.size()
                        && !m_scale_ops[i]->hasZeroPoint()) {
                        // This was our dynamically allocated zero point
                        delete static_cast<int8_t*>(
                            m_postops->scale[i].zp->zero_point);
                    }
                }

                // Clean up the dlp_sf_t and dlp_zp_t structures themselves
                delete m_postops->scale[i].sf;
                delete m_postops->scale[i].zp;
            }
        }

        // Clean up manually allocated memory for scale factors in MatrixAdd
        // operations
        if (m_postops->matrix_add) {
            for (std::size_t i = 0; i < m_matrix_add_ops.size(); ++i) {
                if (m_postops->matrix_add[i].sf) {
                    delete m_postops->matrix_add[i].sf;
                }
            }
        }

        // Clean up manually allocated memory for scale factors in MatrixMul
        // operations
        if (m_postops->matrix_mul) {
            for (std::size_t i = 0; i < m_matrix_mul_ops.size(); ++i) {
                if (m_postops->matrix_mul[i].sf) {
                    delete m_postops->matrix_mul[i].sf;
                }
            }
        }

        // Clean up manually allocated memory for scale factors and zero points
        // in Bias operations (both are optional user features)
        if (m_postops->bias) {
            for (std::size_t i = 0; i < m_bias_ops.size(); ++i) {
                if (m_postops->bias[i].sf) {
                    delete m_postops->bias[i].sf;
                }
                if (m_postops->bias[i].zp) {
                    delete m_postops->bias[i].zp;
                }
            }
        }

        // Clean up manually allocated memory for A matrix quantization scale
        // factors and zero points
        if (m_postops->a_pre_quant) {
            if (m_postops->a_pre_quant->scl) {
                delete m_postops->a_pre_quant->scl;
            }
            if (m_postops->a_pre_quant->zp) {
                delete m_postops->a_pre_quant->zp;
            }
            delete m_postops->a_pre_quant;
        }
        if (m_postops->a_post_quant) {
            if (m_postops->a_post_quant->scl) {
                delete m_postops->a_post_quant->scl;
            }
            if (m_postops->a_post_quant->zp) {
                delete m_postops->a_post_quant->zp;
            }
            delete m_postops->a_post_quant;
        }

        // Clean up manually allocated memory for B matrix quantization
        if (m_postops->pre_ops) {
            if (m_postops->pre_ops->b_scl) {
                delete m_postops->pre_ops->b_scl;
            }
            if (m_postops->pre_ops->b_zp) {
                delete m_postops->pre_ops->b_zp;
            }
            delete m_postops->pre_ops;
        }

        // Clean up all allocated arrays
        delete[] m_postops->eltwise;
        delete[] m_postops->scale;
        delete[] m_postops->bias;
        delete[] m_postops->matrix_add;
        delete[] m_postops->matrix_mul;
        delete[] m_postops->seq_vector;
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
        case dlp::testing::framework::OperationType::Scale: {
            auto scale_param =
                std::unique_ptr<dlp::testing::framework::ScaleParam>(
                    static_cast<dlp::testing::framework::ScaleParam*>(
                        param.release()));
            m_scale_ops.push_back(std::move(scale_param));
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
        case dlp::testing::framework::OperationType::A_Quant: {
            auto a_quant_param =
                std::unique_ptr<dlp::testing::framework::AQuantParam>(
                    static_cast<dlp::testing::framework::AQuantParam*>(
                        param.release()));
            m_a_quant_ops.push_back(std::move(a_quant_param));
            break;
        }
        case dlp::testing::framework::OperationType::WOQ: {
            auto woq_param = std::unique_ptr<dlp::testing::framework::WOQParam>(
                static_cast<dlp::testing::framework::WOQParam*>(
                    param.release()));
            m_woq_ops.push_back(std::move(woq_param));
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

    if (!m_scale_ops.empty()) {
        convertScaleOperations();
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

    if (!m_a_quant_ops.empty()) {
        convertA_QuantOperations();
    }

    if (!m_woq_ops.empty()) {
        convertWOQOperations();
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
    m_postops->eltwise = new dlp_post_op_eltwise[count];
    std::memset(m_postops->eltwise, 0, count * sizeof(dlp_post_op_eltwise));
    m_postops->num_eltwise = count;

    // Fill the array
    for (std::size_t i = 0; i < count; ++i) {
        const auto& param = *m_elementwise_ops[i];

        m_postops->eltwise[i].algo.algo_type =
            getElementWiseAlgoType(param.getOperation());

        // Set storage type for alpha and beta parameters.
        // For CLIP: alpha (lower bound) and beta (upper bound) must have the
        // same type. The parser layer ensures type consistency, so we can use
        // either parameter's type. Default to f32 if no parameters are
        // provided.
        m_postops->eltwise[i].algo.stor_type = DLP_F32;

        // Set alpha and beta if provided
        if (param.hasAlpha()) {
            m_postops->eltwise[i].algo.alpha =
                convertMatrixToPtr(*param.getAlpha());
            m_postops->eltwise[i].algo.stor_type =
                getStorageType(param.getAlpha()->getMatrixType());
        }
        if (param.hasBeta()) {
            m_postops->eltwise[i].algo.beta =
                convertMatrixToPtr(*param.getBeta());
            // If alpha wasn't present, use beta's type for stor_type
            // For CLIP, parser guarantees both types match if both are
            // specified
            if (!param.hasAlpha()) {
                m_postops->eltwise[i].algo.stor_type =
                    getStorageType(param.getBeta()->getMatrixType());
            }
        }
    }
}

void
DlpOperation::convertScaleOperations()
{
    size_t count = m_scale_ops.size();
    if (count == 0)
        return;

    // Allocate array once with exact size
    m_postops->scale = new dlp_scale_t[count];
    std::memset(m_postops->scale, 0, count * sizeof(dlp_scale_t));

    // Fill the array
    for (std::size_t i = 0; i < count; ++i) {
        const auto& param = *m_scale_ops[i];

        // Initialize all fields to safe defaults
        m_postops->scale[i].sf = nullptr;
        m_postops->scale[i].zp = nullptr;

        // Set scale factor if provided
        if (param.hasScaleFactor()) {
            m_postops->scale[i].sf = new dlp_sf_t;
            m_postops->scale[i].sf->scale_factor =
                convertMatrixToPtr(*param.getScaleFactor());
            m_postops->scale[i].sf->scale_factor_len =
                param.getScaleFactor()->getCols();
            m_postops->scale[i].sf->scale_factor_type =
                getStorageType(param.getScaleFactor()->getMatrixType());
        }

        // Set zero point if provided
        if (param.hasZeroPoint()) {
            m_postops->scale[i].zp = new dlp_zp_t;
            m_postops->scale[i].zp->zero_point =
                convertMatrixToPtr(*param.getZeroPoint());
            m_postops->scale[i].zp->zero_point_len =
                param.getZeroPoint()->getCols();
            m_postops->scale[i].zp->zero_point_type =
                getStorageType(param.getZeroPoint()->getMatrixType());
        } else {
            // Based on DLP library validation, if we have a scale factor, we
            // need zero point
            if (param.hasScaleFactor()) {
                // Allocate and set default zero point for SCALE operations
                int8_t* zero_point_data                 = new int8_t(0);
                m_postops->scale[i].zp                  = new dlp_zp_t;
                m_postops->scale[i].zp->zero_point      = zero_point_data;
                m_postops->scale[i].zp->zero_point_len  = 1;
                m_postops->scale[i].zp->zero_point_type = DLP_S8;
            } else {
                // No scale factor provided; invalid for SCALE op in testing.
                throw std::runtime_error(
                    "Scale operation requires scale factor");
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
    m_postops->bias = new dlp_post_op_bias[count];
    std::memset(m_postops->bias, 0, count * sizeof(dlp_post_op_bias));

    // Fill the array
    for (std::size_t i = 0; i < count; ++i) {
        const auto& param = *m_bias_ops[i];

        m_postops->bias[i].bias = convertMatrixToPtr(param.getBias());
        m_postops->bias[i].stor_type =
            getStorageType(param.getBias().getMatrixType());

        // Set scale factor if provided
        if (param.hasScaleFactor()) {
            m_postops->bias[i].sf = new dlp_sf_t;
            m_postops->bias[i].sf->scale_factor =
                convertMatrixToPtr(*param.getScaleFactor());
            m_postops->bias[i].sf->scale_factor_len =
                param.getScaleFactor()->getCols();
            m_postops->bias[i].sf->scale_factor_type =
                getStorageType(param.getScaleFactor()->getMatrixType());
        }

        // Set zero point if provided
        if (param.hasZeroPoint()) {
            m_postops->bias[i].zp = new dlp_zp_t;
            m_postops->bias[i].zp->zero_point =
                convertMatrixToPtr(*param.getZeroPoint());
            m_postops->bias[i].zp->zero_point_len =
                param.getZeroPoint()->getCols();
            m_postops->bias[i].zp->zero_point_type =
                getStorageType(param.getZeroPoint()->getMatrixType());
        }
    }
}

void
DlpOperation::convertMatrixAddOperations()
{
    size_t count = m_matrix_add_ops.size();
    if (count == 0)
        return;

    // Allocate array once with exact size
    m_postops->matrix_add = new dlp_post_op_matrix_add[count];
    std::memset(m_postops->matrix_add, 0,
                count * sizeof(dlp_post_op_matrix_add));

    // Fill the array
    for (std::size_t i = 0; i < count; ++i) {
        const auto& param = *m_matrix_add_ops[i];

        // Initialize sf pointer to null
        m_postops->matrix_add[i].sf = nullptr;

        m_postops->matrix_add[i].matrix = convertMatrixToPtr(param.getMatrix());
        m_postops->matrix_add[i].ldm = param.getMatrix().getLeadingDimension();
        m_postops->matrix_add[i].stor_type =
            getStorageType(param.getMatrix().getMatrixType());

        if (param.hasScaleFactor()) {
            m_postops->matrix_add[i].sf = new dlp_sf_t;
            m_postops->matrix_add[i].sf->scale_factor =
                convertMatrixToPtr(*param.getScaleFactor());
            m_postops->matrix_add[i].sf->scale_factor_len =
                param.getScaleFactor()->getCols();
            m_postops->matrix_add[i].sf->scale_factor_type =
                getStorageType(param.getScaleFactor()->getMatrixType());
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
    m_postops->matrix_mul = new dlp_post_op_matrix_mul[count];
    std::memset(m_postops->matrix_mul, 0,
                count * sizeof(dlp_post_op_matrix_mul));

    // Fill the array
    for (std::size_t i = 0; i < count; ++i) {
        const auto& param = *m_matrix_mul_ops[i];

        // Initialize sf pointer to null
        m_postops->matrix_mul[i].sf = nullptr;

        m_postops->matrix_mul[i].matrix = convertMatrixToPtr(param.getMatrix());
        m_postops->matrix_mul[i].ldm = param.getMatrix().getLeadingDimension();
        m_postops->matrix_mul[i].stor_type =
            getStorageType(param.getMatrix().getMatrixType());

        if (param.hasScaleFactor()) {
            m_postops->matrix_mul[i].sf = new dlp_sf_t;
            m_postops->matrix_mul[i].sf->scale_factor =
                convertMatrixToPtr(*param.getScaleFactor());
            m_postops->matrix_mul[i].sf->scale_factor_len =
                param.getScaleFactor()->getCols();
            m_postops->matrix_mul[i].sf->scale_factor_type =
                getStorageType(param.getScaleFactor()->getMatrixType());
        }
    }
}

void
DlpOperation::convertA_QuantOperations()
{
    size_t count = m_a_quant_ops.size();
    if (count == 0)
        return;

    // Allocate array once with exact size
    // Ensure the quant structure is allocated and zeroed
    if (!m_postops->a_pre_quant) {
        m_postops->a_pre_quant = new dlp_quant_op;
        std::memset(m_postops->a_pre_quant, 0, sizeof(dlp_quant_op));
        // Default to symmetric quantization (no zero-point)
        m_postops->a_pre_quant->symmetric = true;
    }

    // Support only quantization of matrix A (output)
    if (!m_postops->a_post_quant) {
        m_postops->a_post_quant = new dlp_quant_op;
        std::memset(m_postops->a_post_quant, 0, sizeof(dlp_quant_op));
        // Default to symmetric quantization (no zero-point)
        m_postops->a_post_quant->symmetric = true;
    }

    // Process each quant operation parameter (the last one overrides)
    for (std::size_t i = 0; i < count; ++i) {
        const auto& param = *m_a_quant_ops[i];

        // Scale factor assignment
        if (param.hasA_PreOpScaleFactor()) {
            if (!m_postops->a_pre_quant->scl) {
                m_postops->a_pre_quant->scl = new dlp_sf_t;
            }
            auto* scl = m_postops->a_pre_quant->scl;
            scl->scale_factor =
                convertMatrixToPtr(*param.getA_PreOpScaleFactor());
            scl->scale_factor_len = param.getA_PreOpScaleFactor()->getCols();
            scl->scale_factor_type =
                getStorageType(param.getA_PreOpScaleFactor()->getMatrixType());
        }

        // Zero point assignment
        if (param.hasA_PreOpZeroPoint()) {
            if (!m_postops->a_pre_quant->zp) {
                m_postops->a_pre_quant->zp = new dlp_zp_t;
            }
            auto* zp       = m_postops->a_pre_quant->zp;
            zp->zero_point = convertMatrixToPtr(*param.getA_PreOpZeroPoint());
            zp->zero_point_len = param.getA_PreOpZeroPoint()->getCols();
            zp->zero_point_type =
                getStorageType(param.getA_PreOpZeroPoint()->getMatrixType());
            m_postops->a_pre_quant->symmetric = false;
        }

        // Scale factor assignment
        if (param.hasA_PostOpScaleFactor()) {
            if (!m_postops->a_post_quant->scl) {
                m_postops->a_post_quant->scl = new dlp_sf_t;
            }
            auto* scl = m_postops->a_post_quant->scl;
            scl->scale_factor =
                convertMatrixToPtr(*param.getA_PostOpScaleFactor());
            scl->scale_factor_len = param.getA_PostOpScaleFactor()->getCols();
            scl->scale_factor_type =
                getStorageType(param.getA_PostOpScaleFactor()->getMatrixType());
        }

        // Zero point assignment
        if (param.hasA_PostOpZeroPoint()) {
            if (!m_postops->a_post_quant->zp) {
                m_postops->a_post_quant->zp = new dlp_zp_t;
            }
            auto* zp       = m_postops->a_post_quant->zp;
            zp->zero_point = convertMatrixToPtr(*param.getA_PostOpZeroPoint());
            zp->zero_point_len = param.getA_PostOpZeroPoint()->getCols();
            zp->zero_point_type =
                getStorageType(param.getA_PostOpZeroPoint()->getMatrixType());
            m_postops->a_post_quant->symmetric = false;
        }
    }
}

void
DlpOperation::convertWOQOperations()
{
    size_t count = m_woq_ops.size();
    if (count == 0)
        return;

    // Allocate and zero the pre_ops structure
    if (!m_postops->pre_ops) {
        m_postops->pre_ops = new dlp_pre_op;
        std::memset(m_postops->pre_ops, 0, sizeof(dlp_pre_op));
    }

    // Process each WOQ operation parameter (the last one overrides)
    for (std::size_t i = 0; i < count; ++i) {
        const auto& param = *m_woq_ops[i];

        // Scale factor assignment for B matrix
        if (param.hasB_ScaleFactor()) {
            if (!m_postops->pre_ops->b_scl) {
                m_postops->pre_ops->b_scl = new dlp_sf_t;
            }
            auto* scl         = m_postops->pre_ops->b_scl;
            scl->scale_factor = convertMatrixToPtr(*param.getB_ScaleFactor());
            scl->scale_factor_len = param.getB_ScaleFactor()->getCols();
            scl->scale_factor_type =
                getStorageType(param.getB_ScaleFactor()->getMatrixType());
        }

        // Zero point assignment for B matrix
        if (param.hasB_ZeroPoint()) {
            if (!m_postops->pre_ops->b_zp) {
                m_postops->pre_ops->b_zp = new dlp_zp_t;
            }
            auto* zp           = m_postops->pre_ops->b_zp;
            zp->zero_point     = convertMatrixToPtr(*param.getB_ZeroPoint());
            zp->zero_point_len = param.getB_ZeroPoint()->getCols();
            zp->zero_point_type =
                getStorageType(param.getB_ZeroPoint()->getMatrixType());
        }
    }

    // Set sequence information
    m_postops->pre_ops->seq_length = 1; // One WOQ operation
    m_postops->pre_ops->group_size =
        0; // 0 here indicates no explicit group size for WOQ
}

void
DlpOperation::buildSequenceVector()
{
    // Build sequence vector based on original operation order
    std::vector<DLP_POST_OP_TYPE> sequence;
    sequence.reserve(m_operation_params.size());

    // Track array indices for each operation type
    size_t eltwise_idx = 0, scale_idx = 0, bias_idx = 0, matrix_add_idx = 0,
           matrix_mul_idx = 0;

    for (const auto& param : m_operation_params) {
        switch (param->getType()) {
            case dlp::testing::framework::OperationType::ElementWise:
                sequence.push_back(ELTWISE);
                eltwise_idx++;
                break;
            case dlp::testing::framework::OperationType::Scale: {
                sequence.push_back(SCALE);
                scale_idx++;
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
            case dlp::testing::framework::OperationType::A_Quant:
                // A_Quant data already collected via convertA_QuantOperations()
                // Skip adding to sequence since it is not a regular post-op
                // like BIAS/RELU.
                break;
            case dlp::testing::framework::OperationType::WOQ:
                // WOQ data already collected via convertWOQOperations()
                // Skip adding to sequence since it is a pre-op, not a post-op.
                break;
            default:
                throw std::runtime_error(
                    "Unsupported operation type in sequence");
        }
    }

    // Set sequence information
    m_postops->seq_length = sequence.size();
    if (!sequence.empty()) {
        m_postops->seq_vector = new DLP_POST_OP_TYPE[sequence.size()];
        std::memcpy(m_postops->seq_vector, sequence.data(),
                    sequence.size() * sizeof(DLP_POST_OP_TYPE));
    }
}

void*
DlpOperation::convertMatrixToPtr(const dlp::testing::framework::Matrix& matrix)
{
    return matrix.getData();
}

DLP_TYPE
DlpOperation::getStorageType(dlp::testing::framework::MatrixType type)
{
    switch (type) {
        case dlp::testing::framework::MatrixType::f32:
            return DLP_F32;
        case dlp::testing::framework::MatrixType::fp16:
            return DLP_F16;
        case dlp::testing::framework::MatrixType::bf16:
            return DLP_BF16;
        case dlp::testing::framework::MatrixType::s8:
            return DLP_S8;
        case dlp::testing::framework::MatrixType::u8:
            return DLP_U8;
        case dlp::testing::framework::MatrixType::s32:
            return DLP_S32;
        default:
            return DLP_INVALID;
    }
}

DLP_ELT_ALGO_TYPE
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

DLP_POST_OP_TYPE
DlpOperation::getPostOpType(dlp::testing::framework::OperationType type)
{
    switch (type) {
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
