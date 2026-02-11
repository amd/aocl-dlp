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

#include "framework/utils/parser.hh"
#include "framework/matrix.hh"
#include "framework/operation.hh"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

using dlp::testing::framework::postops::createAQuant;
using dlp::testing::framework::postops::createBias;
using dlp::testing::framework::postops::createClip;
using dlp::testing::framework::postops::createGeluErf;
using dlp::testing::framework::postops::createGeluTanh;
using dlp::testing::framework::postops::createMatrixAdd;
using dlp::testing::framework::postops::createMatrixMul;
using dlp::testing::framework::postops::createPrelu;
using dlp::testing::framework::postops::createRelu;
using dlp::testing::framework::postops::createScale;
using dlp::testing::framework::postops::createSigmoid;
using dlp::testing::framework::postops::createSwish;
using dlp::testing::framework::postops::createTanh;
using dlp::testing::framework::postops::createWOQ;

namespace dlp::testing::utils {

// Constants for reproducible random quantization parameters
constexpr unsigned int RANDOM_SEED = 12345; // Fixed seed for reproducibility
constexpr float MIN_VALUE = 1.0f;  // Minimum value for random quant params
constexpr float MAX_VALUE = 20.0f; // Maximum value for random quant params

// Helper functions for parameter extraction
double
MicroTest::extractDoubleParam(
    const std::string&                                  param_name,
    const std::map<std::string, std::vector<std::any>>& params,
    double                                              default_value) const
{
    auto it = params.find(param_name);
    if (it != params.end() && !it->second.empty()) {
        auto param_str = std::any_cast<std::string>(it->second[0]);
        return std::stod(param_str);
    }
    return default_value;
}

MatrixType
MicroTest::extractMatrixTypeParam(
    const std::string&                                  param_name,
    const std::map<std::string, std::vector<std::any>>& params,
    MatrixType                                          default_type) const
{
    auto it = params.find(param_name);
    if (it != params.end() && !it->second.empty()) {
        auto type_str = std::any_cast<std::string>(it->second[0]);
        return stringToMatrixType(type_str);
    }
    return default_type;
}

bool
MicroTest::extractBoolParam(
    const std::string&                                  param_name,
    const std::map<std::string, std::vector<std::any>>& params,
    bool                                                default_value) const
{
    auto it = params.find(param_name);
    if (it != params.end() && !it->second.empty()) {
        auto bool_str = std::any_cast<std::string>(it->second[0]);
        return (bool_str == "true");
    }
    return default_value;
}

std::shared_ptr<IOperation>
MicroTest::getPostOp(UALType ual_type) const
{
    if (!m_has_postops) {
        return nullptr; // No PostOps for this test case
    }

    // Create the operation using OperationFactory
    auto operation = OperationFactory::createOperation(ual_type);

    // Get current combination from PostOpsIterator
    auto        combination = m_postops_iterator->getCurrentCombination();
    const auto& operations  = m_postops_iterator->getOperations();

    // Build each operation in the combination
    for (size_t op_index : combination) {
        const auto& op_config = operations[op_index];

        // Get parameter indices map for this operation
        const auto& param_indices =
            m_postops_iterator->getParameterIndices(op_index);

        // Create the appropriate operation parameter with correct indices
        auto param = createOperationParam(op_config, param_indices);

        // Add to operation
        operation->addOperation(std::move(param));
    }

    // Finalize and return
    operation->finalize();
    return operation;
}

std::unique_ptr<IOperationParam>
MicroTest::createOperationParam(
    const PostOpsIterator::PostOpConfig& config,
    const std::map<std::string, size_t>& param_indices) const
{
    if (config.type == "Elementwise-PRELU") {
        // PRELU: Parse alpha parameter using parameter indices
        double alpha_value = 0.1; // default
        auto   alpha_it    = config.params.find("alpha");
        if (alpha_it != config.params.end() && !alpha_it->second.empty()) {
            auto   idx_it = param_indices.find("alpha");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, alpha_it->second.size() - 1);
            alpha_value =
                std::stod(std::any_cast<std::string>(alpha_it->second[idx]));
        }

        MatrixType alpha_type    = MatrixType::f32; // default
        auto       alpha_type_it = config.params.find("alpha_type");
        if (alpha_type_it != config.params.end()
            && !alpha_type_it->second.empty()) {
            auto   idx_it = param_indices.find("alpha_type");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, alpha_type_it->second.size() - 1);
            auto type_str =
                std::any_cast<std::string>(alpha_type_it->second[idx]);
            alpha_type = stringToMatrixType(type_str);
        }

        auto alpha =
            Matrix::fromValue(static_cast<float>(alpha_value), alpha_type);
        return createPrelu().setAlpha(alpha).build();

    } else if (config.type == "Bias") {
        // Get bias_type and bias_dim parameters using parameter indices
        auto bias_type_it = config.params.find("bias_type");
        auto bias_dim_it  = config.params.find("bias_dim");

        if (bias_type_it == config.params.end()
            || bias_type_it->second.empty()) {
            throw std::runtime_error("Bias requires bias_type parameter");
        }
        if (bias_dim_it == config.params.end() || bias_dim_it->second.empty()) {
            throw std::runtime_error("Bias requires bias_dim parameter");
        }

        // Extract bias_type using parameter index
        auto   type_idx_it = param_indices.find("bias_type");
        size_t type_idx =
            (type_idx_it != param_indices.end()) ? type_idx_it->second : 0;
        type_idx = std::min(type_idx, bias_type_it->second.size() - 1);
        auto bias_type_str =
            std::any_cast<std::string>(bias_type_it->second[type_idx]);
        auto bias_type = stringToMatrixType(bias_type_str);

        // Extract bias_dim using parameter index
        auto   dim_idx_it = param_indices.find("bias_dim");
        size_t dim_idx =
            (dim_idx_it != param_indices.end()) ? dim_idx_it->second : 0;
        dim_idx = std::min(dim_idx, bias_dim_it->second.size() - 1);
        auto bias_dim_str =
            std::any_cast<std::string>(bias_dim_it->second[dim_idx]);

        // Create bias matrix based on dimension
        Matrix bias;
        if (bias_dim_str == "n") {
            // Create bias vector with size N
            std::vector<float> bias_data(getN(), 1.0f);
            bias = Matrix::fromVector(bias_data, bias_type);
        } else {
            // Single bias value
            bias = Matrix::fromValue(1.0f, bias_type);
        }

        // Start building bias operation
        BiasBuilder builder;
        builder.setBias(bias);

        // Optional scale factor for dequantization
        // Following same pattern as Scale post-op
        Matrix      scale_matrix;
        bool        has_sf   = false;
        std::string sf_len   = "";
        MatrixType  sf_type  = MatrixType::f32; // default
        double      sf_value = 2.5;             // default scale factor value

        auto sf_type_it = config.params.find("scale_factor_type");
        if (sf_type_it != config.params.end() && !sf_type_it->second.empty()) {
            auto   idx_it = param_indices.find("scale_factor_type");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, sf_type_it->second.size() - 1);
            auto sf_type_str =
                std::any_cast<std::string>(sf_type_it->second[idx]);
            sf_type = stringToMatrixType(sf_type_str);
            has_sf  = true;
        }

        auto sf_len_it = config.params.find("scale_factor_len");
        if (sf_len_it != config.params.end() && !sf_len_it->second.empty()) {
            auto   idx_it = param_indices.find("scale_factor_len");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, sf_len_it->second.size() - 1);
            sf_len     = std::any_cast<std::string>(sf_len_it->second[idx]);
            has_sf     = true;
        }

        // If user provided explicit scale_factor value, use it
        auto sf_it = config.params.find("scale_factor");
        if (sf_it != config.params.end() && !sf_it->second.empty()) {
            auto   idx_it = param_indices.find("scale_factor");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, sf_it->second.size() - 1);
            sf_value =
                std::stod(std::any_cast<std::string>(sf_it->second[idx]));
            has_sf = true;
        }

        if (has_sf) {
            if (sf_len == "n") {
                // Per-channel: create vector of length N with scale value
                std::vector<float> sf_data(getN(),
                                           static_cast<float>(sf_value));
                scale_matrix = Matrix::fromVector(sf_data, sf_type);
            } else {
                // Scalar (per-tensor) - default when sf_len not specified or
                // "1"
                scale_matrix =
                    Matrix::fromValue(static_cast<float>(sf_value), sf_type);
            }
            builder.setScaleFactor(scale_matrix);
        }

        // Optional zero point for dequantization
        // Following same pattern as Scale post-op
        Matrix      zero_point_matrix;
        bool        has_zp   = false;
        std::string zp_len   = "";
        MatrixType  zp_type  = MatrixType::f32; // default
        double      zp_value = 10.0;            // default zero point value

        auto zp_type_it = config.params.find("zero_point_type");
        if (zp_type_it != config.params.end() && !zp_type_it->second.empty()) {
            auto   idx_it = param_indices.find("zero_point_type");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, zp_type_it->second.size() - 1);
            auto zp_type_str =
                std::any_cast<std::string>(zp_type_it->second[idx]);
            zp_type = stringToMatrixType(zp_type_str);
            has_zp  = true;
        }

        auto zp_len_it = config.params.find("zero_point_len");
        if (zp_len_it != config.params.end() && !zp_len_it->second.empty()) {
            auto   idx_it = param_indices.find("zero_point_len");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, zp_len_it->second.size() - 1);
            zp_len     = std::any_cast<std::string>(zp_len_it->second[idx]);
            has_zp     = true;
        }

        // If user provided explicit zero_point value, use it
        auto zp_it = config.params.find("zero_point");
        if (zp_it != config.params.end() && !zp_it->second.empty()) {
            auto   idx_it = param_indices.find("zero_point");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, zp_it->second.size() - 1);
            zp_value =
                std::stod(std::any_cast<std::string>(zp_it->second[idx]));
            has_zp = true;
        }

        if (has_zp) {
            if (zp_len == "n") {
                // Per-channel: create vector of length N with zero point value
                std::vector<float> zp_data(getN(),
                                           static_cast<float>(zp_value));
                zero_point_matrix = Matrix::fromVector(zp_data, zp_type);
            } else {
                // Scalar (per-tensor) - default when zp_len not specified or
                // "1"
                zero_point_matrix =
                    Matrix::fromValue(static_cast<float>(zp_value), zp_type);
            }
            builder.setZeroPoint(zero_point_matrix);
        }

        return builder.build();

    } else if (config.type == "Elementwise-RELU") {
        // RELU requires no parameters
        return createRelu().build();

    } else if (config.type == "Elementwise-GELU-TANH") {
        // GELU-TANH requires no parameters
        return createGeluTanh().build();

    } else if (config.type == "Elementwise-GELU-ERF") {
        // GELU-ERF requires no parameters
        return createGeluErf().build();

    } else if (config.type == "Elementwise-SWISH") {
        // SWISH: Optional alpha parameter, default to 2.0f if not specified
        auto   alpha_type_it = config.params.find("alpha_type");
        double alpha_value   = extractDoubleParam("alpha", config.params, 2.0);

        if (alpha_type_it != config.params.end()
            && !alpha_type_it->second.empty()) {
            auto   idx_it = param_indices.find("alpha_type");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, alpha_type_it->second.size() - 1);
            auto alpha_type_str =
                std::any_cast<std::string>(alpha_type_it->second[idx]);
            auto alpha_type = stringToMatrixType(alpha_type_str);

            auto alpha =
                Matrix::fromValue(static_cast<float>(alpha_value), alpha_type);
            return createSwish().setAlpha(alpha).build();
        } else {
            // Use default alpha (will be set automatically in SwishBuilder)
            return createSwish().build();
        }

    } else if (config.type == "Elementwise-TANH") {
        // TANH requires no parameters
        return createTanh().build();

    } else if (config.type == "Elementwise-SIGMOID") {
        // SIGMOID requires no parameters
        return createSigmoid().build();

    } else if (config.type == "Elementwise-CLIP") {
        // CLIP: Parse alpha (lower bound) and beta (upper bound) parameters
        double alpha_value = -1.0; // default
        auto   alpha_it    = config.params.find("alpha");
        if (alpha_it != config.params.end() && !alpha_it->second.empty()) {
            auto   idx_it = param_indices.find("alpha");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, alpha_it->second.size() - 1);
            alpha_value =
                std::stod(std::any_cast<std::string>(alpha_it->second[idx]));
        }

        double beta_value = 6.0; // default
        auto   beta_it    = config.params.find("beta");
        if (beta_it != config.params.end() && !beta_it->second.empty()) {
            auto   idx_it = param_indices.find("beta");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, beta_it->second.size() - 1);
            beta_value =
                std::stod(std::any_cast<std::string>(beta_it->second[idx]));
        }

        // Check if alpha_type and beta_type are explicitly provided
        auto alpha_type_it = config.params.find("alpha_type");
        auto beta_type_it  = config.params.find("beta_type");

        bool has_alpha_type = (alpha_type_it != config.params.end()
                               && !alpha_type_it->second.empty());
        bool has_beta_type  = (beta_type_it != config.params.end()
                              && !beta_type_it->second.empty());

        MatrixType alpha_type;
        MatrixType beta_type;

        // Smart defaulting: if only one is specified, use it for both
        if (has_alpha_type && !has_beta_type) {
            // Only alpha specified, use it for both
            auto   idx_it = param_indices.find("alpha_type");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, alpha_type_it->second.size() - 1);
            auto type_str =
                std::any_cast<std::string>(alpha_type_it->second[idx]);
            alpha_type = stringToMatrixType(type_str);
            beta_type  = alpha_type;
        } else if (!has_alpha_type && has_beta_type) {
            // Only beta specified, use it for both
            auto   idx_it = param_indices.find("beta_type");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, beta_type_it->second.size() - 1);
            auto type_str =
                std::any_cast<std::string>(beta_type_it->second[idx]);
            beta_type  = stringToMatrixType(type_str);
            alpha_type = beta_type;
        } else {
            // Both specified OR neither specified - extract both
            alpha_type = MatrixType::f32; // default
            if (has_alpha_type) {
                auto   idx_it = param_indices.find("alpha_type");
                size_t idx    = (idx_it != param_indices.end()) ? idx_it->second
                                                                : 0;
                idx           = std::min(idx, alpha_type_it->second.size() - 1);
                auto type_str =
                    std::any_cast<std::string>(alpha_type_it->second[idx]);
                alpha_type = stringToMatrixType(type_str);
            }
            beta_type = MatrixType::f32; // default
            if (has_beta_type) {
                auto   idx_it = param_indices.find("beta_type");
                size_t idx    = (idx_it != param_indices.end()) ? idx_it->second
                                                                : 0;
                idx           = std::min(idx, beta_type_it->second.size() - 1);
                auto type_str =
                    std::any_cast<std::string>(beta_type_it->second[idx]);
                beta_type = stringToMatrixType(type_str);
            }

            // Warn only if both were explicitly specified and they differ
            if (has_alpha_type && has_beta_type && alpha_type != beta_type) {
                std::cerr << "Warning: Different datatypes for alpha \""
                          << matrixTypeToString(alpha_type) << "\" and beta \""
                          << matrixTypeToString(beta_type)
                          << "\" are not supported by dlp, translating beta "
                             "type to alpha type \""
                          << matrixTypeToString(alpha_type) << "\"."
                          << std::endl;
                beta_type = alpha_type; // Use alpha_type when both differ
            }
        }

        // Create matrices with parsed values and types (static cast to correct
        // type)
        auto lower_matrix =
            Matrix::fromValue(static_cast<float>(alpha_value), alpha_type);
        auto upper_matrix =
            Matrix::fromValue(static_cast<float>(beta_value), beta_type);

        return createClip()
            .setLowerBound(lower_matrix)
            .setUpperBound(upper_matrix)
            .build();

    } else if (config.type == "Scale") {
        // Scale: Parse scale_factor parameter
        double scale_value = 2.5; // default
        auto   sf_it       = config.params.find("scale_factor");
        if (sf_it != config.params.end() && !sf_it->second.empty()) {
            auto   idx_it = param_indices.find("scale_factor");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, sf_it->second.size() - 1);
            scale_value =
                std::stod(std::any_cast<std::string>(sf_it->second[idx]));
        }

        MatrixType scale_type = MatrixType::f32; // default
        auto       sf_type_it = config.params.find("scale_factor_type");
        if (sf_type_it != config.params.end() && !sf_type_it->second.empty()) {
            auto   idx_it = param_indices.find("scale_factor_type");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, sf_type_it->second.size() - 1);
            auto type_str = std::any_cast<std::string>(sf_type_it->second[idx]);
            scale_type    = stringToMatrixType(type_str);
        }

        bool is_power_of_2 = false; // default
        auto pow2_it       = config.params.find("is_power_of_2");
        if (pow2_it != config.params.end() && !pow2_it->second.empty()) {
            auto   idx_it = param_indices.find("is_power_of_2");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, pow2_it->second.size() - 1);
            auto bool_str = std::any_cast<std::string>(pow2_it->second[idx]);
            is_power_of_2 = (bool_str == "true");
        }

        // If is_power_of_2 is true but no scale_factor specified, use
        // default 2.0
        if (is_power_of_2
            && config.params.find("scale_factor") == config.params.end()) {
            scale_value = 2.0;
        }

        // Check scale_factor_len to determine scalar vs vector
        Matrix      scale_matrix;
        std::string sf_len    = "";
        auto        sf_len_it = config.params.find("scale_factor_len");
        if (sf_len_it != config.params.end() && !sf_len_it->second.empty()) {
            auto   idx_it = param_indices.find("scale_factor_len");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, sf_len_it->second.size() - 1);
            sf_len     = std::any_cast<std::string>(sf_len_it->second[idx]);
        }

        if (sf_len == "n") {
            // Per-channel: create vector of length N with scale value
            std::vector<float> sf_data(getN(), static_cast<float>(scale_value));
            scale_matrix = Matrix::fromVector(sf_data, scale_type);
        } else {
            // Scalar (per-tensor) - default when sf_len not specified or "1"
            scale_matrix =
                Matrix::fromValue(static_cast<float>(scale_value), scale_type);
        }

        // Optional zero-point
        Matrix      zero_point_matrix;
        bool        has_zp    = false;
        std::string zp_len    = "";
        MatrixType  zp_type   = MatrixType::s8; // default
        auto        zp_len_it = config.params.find("zero_point_len");
        if (zp_len_it != config.params.end() && !zp_len_it->second.empty()) {
            auto   idx_it = param_indices.find("zero_point_len");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, zp_len_it->second.size() - 1);
            zp_len     = std::any_cast<std::string>(zp_len_it->second[idx]);
            has_zp     = true;
        }
        auto zp_type_it = config.params.find("zero_point_type");
        if (zp_type_it != config.params.end() && !zp_type_it->second.empty()) {
            auto   idx_it = param_indices.find("zero_point_type");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, zp_type_it->second.size() - 1);
            auto zp_type_str =
                std::any_cast<std::string>(zp_type_it->second[idx]);
            zp_type = stringToMatrixType(zp_type_str);
            has_zp  = true;
        }
        if (has_zp) {
            if (zp_len == "n") {
                std::vector<int32_t> zp_data(getN(), 0);
                zero_point_matrix = Matrix::fromVector(zp_data, zp_type);
            } else {
                zero_point_matrix = Matrix::fromValue(0, zp_type);
            }
        }

        auto builder = createScale();
        builder.setScaleFactor(scale_matrix);
        if (has_zp) {
            builder.setZeroPoint(zero_point_matrix);
        }
        return builder.build();

    } else if (config.type == "Matrix-Add") {
        // Matrix-Add: Use C matrix dimensions (m×n) and proper leading dim
        auto rows = getM(); // Use current test case M dimension
        auto cols = getN(); // Use current test case N dimension

        // Extract matrix_type parameters (default to f32)
        MatrixType matrix_type    = MatrixType::f32; // default
        auto       matrix_type_it = config.params.find("matrix_type");
        if (matrix_type_it != config.params.end()
            && !matrix_type_it->second.empty()) {
            auto   idx_it = param_indices.find("matrix_type");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, matrix_type_it->second.size() - 1);
            auto type_str =
                std::any_cast<std::string>(matrix_type_it->second[idx]);
            matrix_type = stringToMatrixType(type_str);
        }

        // Get the storage format to match C matrix layout
        MatrixLayout layout = getStorageFormat();

        std::vector<std::vector<float>> matrix_data(
            rows, std::vector<float>(cols, 0.5f));
        auto matrix = Matrix::fromData(matrix_data, matrix_type, layout);

        // Check if scale_factor is explicitly provided (optional parameter)
        auto sf_it      = config.params.find("scale_factor");
        auto sf_type_it = config.params.find("scale_factor_type");
        auto sf_len_it  = config.params.find("scale_factor_len");

        // Scale factor is present if any of these params are specified
        bool has_sf =
            (sf_it != config.params.end() && !sf_it->second.empty())
            || (sf_type_it != config.params.end()
                && !sf_type_it->second.empty())
            || (sf_len_it != config.params.end() && !sf_len_it->second.empty());

        // If no scale factor, return without it
        if (!has_sf) {
            return createMatrixAdd().setMatrix(matrix).build();
        }

        // Build with scale factor
        double scale_value = 0.8; // default when sf params present
        if (sf_it != config.params.end() && !sf_it->second.empty()) {
            auto   idx_it = param_indices.find("scale_factor");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, sf_it->second.size() - 1);
            scale_value =
                std::stod(std::any_cast<std::string>(sf_it->second[idx]));
        }

        MatrixType scale_type = MatrixType::f32; // default
        if (sf_type_it != config.params.end() && !sf_type_it->second.empty()) {
            auto   idx_it = param_indices.find("scale_factor_type");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, sf_type_it->second.size() - 1);
            auto type_str = std::any_cast<std::string>(sf_type_it->second[idx]);
            scale_type    = stringToMatrixType(type_str);
        }

        // Check scale_factor_len to determine scalar vs vector
        Matrix      scale_matrix;
        std::string sf_len = "";
        if (sf_len_it != config.params.end() && !sf_len_it->second.empty()) {
            auto   idx_it = param_indices.find("scale_factor_len");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, sf_len_it->second.size() - 1);
            sf_len     = std::any_cast<std::string>(sf_len_it->second[idx]);
        }

        if (sf_len == "n") {
            // Per-channel: create vector of length N with scale value
            std::vector<float> sf_data(getN(), static_cast<float>(scale_value));
            scale_matrix = Matrix::fromVector(sf_data, scale_type);
        } else {
            // Scalar (per-tensor) - default when sf_len not specified or "1"
            scale_matrix =
                Matrix::fromValue(static_cast<float>(scale_value), scale_type);
        }

        return createMatrixAdd()
            .setMatrix(matrix)
            .setScaleFactor(scale_matrix)
            .build();

    } else if (config.type == "Matrix-Mul") {
        // Matrix-Mul: Use C matrix dimensions (m×n) and proper leading dim
        auto rows = getM(); // Use current test case M dimension
        auto cols = getN(); // Use current test case N dimension

        // Extract matrix-type parameters (default to f32)
        MatrixType matrix_type    = MatrixType::f32; // default
        auto       matrix_type_it = config.params.find("matrix_type");
        if (matrix_type_it != config.params.end()
            && !matrix_type_it->second.empty()) {
            auto   idx_it = param_indices.find("matrix_type");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, matrix_type_it->second.size() - 1);
            auto type_str =
                std::any_cast<std::string>(matrix_type_it->second[idx]);
            matrix_type = stringToMatrixType(type_str);
        }

        // Get the storage format to match C matrix layout
        MatrixLayout layout = getStorageFormat();

        std::vector<std::vector<float>> matrix_data(
            rows, std::vector<float>(cols, 1.0f));
        auto matrix = Matrix::fromData(matrix_data, matrix_type, layout);

        // Check if scale_factor is explicitly provided (optional parameter)
        auto sf_it      = config.params.find("scale_factor");
        auto sf_type_it = config.params.find("scale_factor_type");
        auto sf_len_it  = config.params.find("scale_factor_len");

        // Scale factor is present if any of these params are specified
        bool has_sf =
            (sf_it != config.params.end() && !sf_it->second.empty())
            || (sf_type_it != config.params.end()
                && !sf_type_it->second.empty())
            || (sf_len_it != config.params.end() && !sf_len_it->second.empty());

        // If no scale factor, return without it
        if (!has_sf) {
            return createMatrixMul().setMatrix(matrix).build();
        }

        // Build with scale factor
        double scale_value = 1.2; // default scale factor for Matrix-Mul
        if (sf_it != config.params.end() && !sf_it->second.empty()) {
            auto   idx_it = param_indices.find("scale_factor");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, sf_it->second.size() - 1);
            scale_value =
                std::stod(std::any_cast<std::string>(sf_it->second[idx]));
        }

        MatrixType scale_type = MatrixType::f32; // default
        if (sf_type_it != config.params.end() && !sf_type_it->second.empty()) {
            auto   idx_it = param_indices.find("scale_factor_type");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, sf_type_it->second.size() - 1);
            auto type_str = std::any_cast<std::string>(sf_type_it->second[idx]);
            scale_type    = stringToMatrixType(type_str);
        }

        // Check scale_factor_len to determine scalar vs vector
        Matrix      scale_matrix;
        std::string sf_len = "";
        if (sf_len_it != config.params.end() && !sf_len_it->second.empty()) {
            auto   idx_it = param_indices.find("scale_factor_len");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, sf_len_it->second.size() - 1);
            sf_len     = std::any_cast<std::string>(sf_len_it->second[idx]);
        }

        if (sf_len == "n") {
            // Per-channel: create vector of length N with scale value
            std::vector<float> sf_data(getN(), static_cast<float>(scale_value));
            scale_matrix = Matrix::fromVector(sf_data, scale_type);
        } else {
            // Scalar (per-tensor) - default when sf_len not specified or "1"
            scale_matrix =
                Matrix::fromValue(static_cast<float>(scale_value), scale_type);
        }

        return createMatrixMul()
            .setMatrix(matrix)
            .setScaleFactor(scale_matrix)
            .build();
    } else if (config.type == "A_Quant") {
        // A_Quant: Parse a_quant_sf_len and a_post_quant_sf_len
        std::string sf_len    = "1"; // default
        auto        sf_len_it = config.params.find("a_quant_sf_len");
        if (sf_len_it != config.params.end() && !sf_len_it->second.empty()) {
            auto   idx_it = param_indices.find("a_quant_sf_len");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, sf_len_it->second.size() - 1);
            sf_len     = std::any_cast<std::string>(sf_len_it->second[idx]);
        }

        // Parse a_quant_sf_type and a_post_quant_sf_type
        MatrixType sf_type    = MatrixType::f32; // default
        auto       sf_type_it = config.params.find("a_quant_sf_type");
        if (sf_type_it != config.params.end() && !sf_type_it->second.empty()) {
            auto   idx_it = param_indices.find("a_quant_sf_type");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, sf_type_it->second.size() - 1);
            auto type_str = std::any_cast<std::string>(sf_type_it->second[idx]);
            sf_type       = stringToMatrixType(type_str);
        }

        // Create scale factor matrix based on length specification
        Matrix pre_quant_sf_matrix;
        Matrix post_quant_sf_matrix;
        // Use fixed seed for reproducible random values across DLP and REF
        std::mt19937                          gen(RANDOM_SEED);
        std::uniform_real_distribution<float> dist(MIN_VALUE, MAX_VALUE);

        if (sf_len == "m") {
            std::vector<float> pre_quant_sf_data(getM());
            std::vector<float> post_quant_sf_data(getM());
            for (size_t i = 0; i < pre_quant_sf_data.size(); ++i) {
                // Generate random scale values between 0.1 and 10
                pre_quant_sf_data[i]  = dist(gen);
                post_quant_sf_data[i] = 1.0f / dist(gen);
            }
            pre_quant_sf_matrix =
                Matrix::fromVector(pre_quant_sf_data, sf_type);
            post_quant_sf_matrix =
                Matrix::fromVector(post_quant_sf_data, sf_type);
        } else {
            float sf_value       = dist(gen);
            pre_quant_sf_matrix  = Matrix::fromValue(sf_value, sf_type);
            post_quant_sf_matrix = Matrix::fromValue(1.0f / sf_value, sf_type);
        }

        // Parse optional zero_point for asymmetric quantization
        Matrix      zp_matrix;
        bool        has_zp = false;
        std::string zp_len = "1";

        // Parse a_quant_zp_len
        auto zp_len_it = config.params.find("a_quant_zp_len");
        if (zp_len_it != config.params.end() && !zp_len_it->second.empty()) {
            auto   idx_it = param_indices.find("a_quant_zp_len");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, zp_len_it->second.size() - 1);
            zp_len     = std::any_cast<std::string>(zp_len_it->second[idx]);
            has_zp     = true;
        }

        // Parse a_quant_zp_type
        MatrixType zp_type    = MatrixType::f32; // default
        auto       zp_type_it = config.params.find("a_quant_zp_type");
        if (zp_type_it != config.params.end() && !zp_type_it->second.empty()) {
            auto   idx_it = param_indices.find("a_quant_zp_type");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, zp_type_it->second.size() - 1);
            auto zp_type_str =
                std::any_cast<std::string>(zp_type_it->second[idx]);
            zp_type = stringToMatrixType(zp_type_str);
            has_zp  = true;
        }

        // Create zero point matrix based on length specification
        if (has_zp) {
            if (zp_len == "m") {
                std::vector<float> zp_data(getM());
                for (size_t idx = 0; idx < zp_data.size(); ++idx) {
                    // Generate random zero point values between 1 and 20
                    zp_data[idx] = dist(gen);
                }
                zp_matrix = Matrix::fromVector(zp_data, zp_type);
            } else {
                zp_matrix = Matrix::fromValue(dist(gen), zp_type);
            }

            // Asymmetric quantization: include zero-point
            return createAQuant()
                .setA_PreOpScaleFactor(pre_quant_sf_matrix)
                .setA_PostOpScaleFactor(post_quant_sf_matrix)
                .setA_PreOpZeroPoint(zp_matrix)
                .setA_PostOpZeroPoint(zp_matrix)
                .build();
        } else {
            // Symmetric quantization: no zero-point
            return createAQuant()
                .setA_PreOpScaleFactor(pre_quant_sf_matrix)
                .setA_PostOpScaleFactor(post_quant_sf_matrix)
                .build();
        }
    } else if (config.type == "WOQ") {
        // WOQ: Weight-Only Quantization for bf16s4
        // Parse scale_factor_len
        std::string sf_len    = "1"; // default
        auto        sf_len_it = config.params.find("scale_factor_len");
        if (sf_len_it != config.params.end() && !sf_len_it->second.empty()) {
            auto   idx_it = param_indices.find("scale_factor_len");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, sf_len_it->second.size() - 1);
            sf_len     = std::any_cast<std::string>(sf_len_it->second[idx]);
        }

        // Parse scale_factor_type
        MatrixType sf_type    = MatrixType::f32; // default
        auto       sf_type_it = config.params.find("scale_factor_type");
        if (sf_type_it != config.params.end() && !sf_type_it->second.empty()) {
            auto   idx_it = param_indices.find("scale_factor_type");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, sf_type_it->second.size() - 1);
            auto type_str = std::any_cast<std::string>(sf_type_it->second[idx]);
            sf_type       = stringToMatrixType(type_str);
        }

        // Create scale factor matrix based on length specification
        Matrix b_scale_factor_matrix;
        // Use fixed seed for reproducible random values
        std::mt19937                          gen(RANDOM_SEED);
        std::uniform_real_distribution<float> dist(MIN_VALUE, MAX_VALUE);

        if (sf_len == "n") {
            // Per-channel quantization: one scale per output channel
            std::vector<float> sf_data(getN());
            for (size_t i = 0; i < sf_data.size(); ++i) {
                // Generate random scale values between MIN_VALUE and MAX_VALUE
                sf_data[i] = dist(gen);
            }
            b_scale_factor_matrix = Matrix::fromVector(sf_data, sf_type);
        } else {
            // Per-tensor quantization: single scale for entire matrix
            float sf_value        = dist(gen);
            b_scale_factor_matrix = Matrix::fromValue(sf_value, sf_type);
        }

        // Parse zero_point_len
        std::string zp_len    = "1"; // default
        auto        zp_len_it = config.params.find("zero_point_len");
        if (zp_len_it != config.params.end() && !zp_len_it->second.empty()) {
            auto   idx_it = param_indices.find("zero_point_len");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, zp_len_it->second.size() - 1);
            zp_len     = std::any_cast<std::string>(zp_len_it->second[idx]);
        }

        // Parse zero_point_type
        MatrixType zp_type    = MatrixType::f32; // default
        auto       zp_type_it = config.params.find("zero_point_type");
        if (zp_type_it != config.params.end() && !zp_type_it->second.empty()) {
            auto   idx_it = param_indices.find("zero_point_type");
            size_t idx = (idx_it != param_indices.end()) ? idx_it->second : 0;
            idx        = std::min(idx, zp_type_it->second.size() - 1);
            auto zp_type_str =
                std::any_cast<std::string>(zp_type_it->second[idx]);
            zp_type = stringToMatrixType(zp_type_str);
        }

        // Create zero point matrix based on length specification
        Matrix b_zero_point_matrix;
        if (zp_len == "n") {
            // Per-channel: one zero point per output channel
            std::vector<float> zp_data(getN());
            for (size_t i = 0; i < zp_data.size(); ++i) {
                // Generate random zero point values using the configured
                // distribution
                zp_data[i] = dist(gen);
            }
            b_zero_point_matrix = Matrix::fromVector(zp_data, zp_type);
        } else {
            // Per-tensor: single zero point
            b_zero_point_matrix = Matrix::fromValue(dist(gen), zp_type);
        }

        // Create WOQ pre-operation using the same structure as A_Quant
        // but for B matrix (weights)
        return createWOQ()
            .setB_ScaleFactor(b_scale_factor_matrix)
            .setB_ZeroPoint(b_zero_point_matrix)
            .build();
    }
    throw std::runtime_error("Unknown operation type: " + config.type);
}

std::string
MicroTest::matrixTypeToString(MatrixType type) const
{
    switch (type) {
        case MatrixType::f32:
            return "f32";
        case MatrixType::bf16:
            return "bf16";
        case MatrixType::s8:
            return "s8";
        case MatrixType::u8:
            return "u8";
        case MatrixType::s16:
            return "s16";
        case MatrixType::u16:
            return "u16";
        case MatrixType::s32:
            return "s32";
        case MatrixType::u32:
            return "u32";
        case MatrixType::s4:
            return "s4";
        case MatrixType::u4:
            return "u4";
        default:
            throw std::runtime_error("Unknown matrix type enum value");
    }
}

MatrixType
MicroTest::stringToMatrixType(const std::string& str) const
{
    if (str == "f32")
        return MatrixType::f32;
    if (str == "bf16")
        return MatrixType::bf16;
    if (str == "s8" || str == "int8")
        return MatrixType::s8;
    if (str == "u8" || str == "uint8")
        return MatrixType::u8;
    if (str == "s16" || str == "int16")
        return MatrixType::s16;
    if (str == "u16" || str == "uint16")
        return MatrixType::u16;
    if (str == "s32" || str == "int32")
        return MatrixType::s32;
    if (str == "u32" || str == "uint32")
        return MatrixType::u32;
    if (str == "s4")
        return MatrixType::s4;
    if (str == "u4")
        return MatrixType::u4;

    throw std::runtime_error("Unknown matrix type: " + str);
}

// ============================================================================
// FILL PATTERN IMPLEMENTATION
// ============================================================================

/**
 * @brief Generate pattern values based on configuration
 */
std::vector<double>
FillPatternConfig::generatePattern() const
{
    std::vector<double> result;

    switch (type) {
        case PatternType::Static:
            return values; // Return as-is

        case PatternType::Modulo: {
            double range = ub - lb;
            if (range <= 0) {
                range = 1.0; // Safety fallback
            }

            // Use integer-based loop to avoid floating-point accumulation
            // errors
            size_t num_steps = static_cast<size_t>(std::ceil(range / step)) + 1;
            num_steps        = std::min(num_steps, MAX_PATTERN_SIZE);

            for (size_t i = 0; i < num_steps; ++i) {
                double x   = i * step;
                double val = std::fmod(x + offset, range) + lb;
                if (i == num_steps - 1) {
                    val = ub;
                }
                result.push_back(val);
            }
            break;
        }

        case PatternType::Linear: {
            // Use integer-based loop to avoid floating-point accumulation
            // errors
            size_t num_steps = static_cast<size_t>(std::ceil((ub - lb) / step));
            num_steps        = std::min(num_steps, MAX_PATTERN_SIZE);

            for (size_t i = 0; i < num_steps; ++i) {
                double x = lb + i * step;
                result.push_back(x * multiplier + offset);
            }
            break;
        }

        case PatternType::Sequence: {
            // Use integer-based loop to avoid floating-point accumulation
            // errors
            size_t num_steps = static_cast<size_t>(std::ceil((ub - lb) / step));
            num_steps        = std::min(num_steps, MAX_PATTERN_SIZE);

            for (size_t i = 0; i < num_steps; ++i) {
                result.push_back(lb + i * step);
            }
            break;
        }

        default:
            break;
    }

    // Safety: ensure non-empty pattern
    if (result.empty()) {
        result.push_back(0.0);
    }

    return result;
}

/**
 * @brief Validate pattern configuration
 */
bool
FillPatternConfig::validate(std::string& error_msg) const
{
    if (type == PatternType::Static) {
        if (values.empty()) {
            error_msg = "Static pattern requires non-empty values";
            return false;
        }
        return true;
    }

    // For expression types
    if (step <= 0.0) {
        error_msg = "Step must be positive";
        return false;
    }

    if (lb >= ub) {
        error_msg = "Lower bound must be less than upper bound";
        return false;
    }

    return true;
}

/**
 * @brief Output stream support for PatternType (for debugging)
 */
std::ostream&
operator<<(std::ostream& os, PatternType type)
{
    switch (type) {
        case PatternType::Static:
            return os << "Static";
        case PatternType::Modulo:
            return os << "Modulo";
        case PatternType::Linear:
            return os << "Linear";
        case PatternType::Sequence:
            return os << "Sequence";
        default:
            return os << "Unknown";
    }
}

/**
 * @brief Output stream support for FillPatternConfig (for debugging)
 */
std::ostream&
operator<<(std::ostream& os, const FillPatternConfig& cfg)
{
    os << "FillPatternConfig{type=" << cfg.type;

    if (cfg.type == PatternType::Static) {
        os << ", values=[";
        for (size_t i = 0; i < cfg.values.size(); ++i) {
            if (i > 0)
                os << ", ";
            os << cfg.values[i];
        }
        os << "]";
    } else {
        os << ", lb=" << cfg.lb << ", ub=" << cfg.ub << ", step=" << cfg.step;
        if (cfg.type == PatternType::Linear) {
            os << ", multiplier=" << cfg.multiplier;
        }
        if (cfg.type == PatternType::Modulo
            || cfg.type == PatternType::Linear) {
            os << ", offset=" << cfg.offset;
        }
    }

    os << "}";
    return os;
}

} // namespace dlp::testing::utils
