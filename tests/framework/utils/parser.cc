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

#include <stdexcept>

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

namespace dlp::testing::utils {

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

        // Create the appropriate operation parameter
        auto param = createOperationParam(op_config);

        // Add to operation
        operation->addOperation(std::move(param));
    }

    // Finalize and return
    operation->finalize();
    return operation;
}

std::unique_ptr<IOperationParam>
MicroTest::createOperationParam(
    const PostOpsIterator::PostOpConfig& config) const
{
    if (config.type == "Elementwise-PRELU") {
        // PRELU: Parse alpha parameter
        double alpha_value    = extractDoubleParam("alpha", config.params, 0.1);
        MatrixType alpha_type = extractMatrixTypeParam(
            "alpha_type", config.params, MatrixType::f32);

        // Create matrix with parsed values and type (static cast to correct
        // type)
        auto alpha =
            Matrix::fromValue(static_cast<float>(alpha_value), alpha_type);

        return createPrelu().setAlpha(alpha).build();

    } else if (config.type == "Bias") {
        // Get bias_type and bias_dim parameters
        auto bias_type_it = config.params.find("bias_type");
        auto bias_dim_it  = config.params.find("bias_dim");

        if (bias_type_it == config.params.end()
            || bias_type_it->second.empty()) {
            throw std::runtime_error("Bias requires bias_type parameter");
        }
        if (bias_dim_it == config.params.end() || bias_dim_it->second.empty()) {
            throw std::runtime_error("Bias requires bias_dim parameter");
        }

        auto bias_type_str =
            std::any_cast<std::string>(bias_type_it->second[0]);
        auto bias_dim_str = std::any_cast<std::string>(bias_dim_it->second[0]);
        auto bias_type    = stringToMatrixType(bias_type_str);

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

        return createBias().setBias(bias).build();

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
        // SWISH: Optional alpha parameter, default to 1.0f if not specified
        auto alpha_type_it = config.params.find("alpha_type");
        if (alpha_type_it != config.params.end()
            && !alpha_type_it->second.empty()) {
            auto alpha_type_str =
                std::any_cast<std::string>(alpha_type_it->second[0]);
            auto alpha_type = stringToMatrixType(alpha_type_str);
            auto alpha      = Matrix::fromValue(1.0f, alpha_type);
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
        double alpha_value = extractDoubleParam("alpha", config.params, -1.0);
        double beta_value  = extractDoubleParam("beta", config.params, 6.0);
        MatrixType alpha_type = extractMatrixTypeParam(
            "alpha_type", config.params, MatrixType::f32);
        MatrixType beta_type =
            extractMatrixTypeParam("beta_type", config.params, MatrixType::f32);

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
        double scale_value =
            extractDoubleParam("scale_factor", config.params, 2.5);
        MatrixType scale_type = extractMatrixTypeParam(
            "scale_factor_type", config.params, MatrixType::f32);
        bool is_power_of_2 =
            extractBoolParam("is_power_of_2", config.params, false);

        // If is_power_of_2 is true but no scale_factor specified, use
        // default 2.0
        if (is_power_of_2
            && config.params.find("scale_factor") == config.params.end()) {
            scale_value = 2.0;
        }

        // Create matrix with parsed values and type (static cast to correct
        // type)
        auto scale_matrix =
            Matrix::fromValue(static_cast<float>(scale_value), scale_type);

        // Optional zero-point
        Matrix      zero_point_matrix;
        bool        has_zp    = false;
        std::string zp_len    = "";
        MatrixType  zp_type   = MatrixType::s8; // default
        auto        zp_len_it = config.params.find("zero_point_len");
        if (zp_len_it != config.params.end() && !zp_len_it->second.empty()) {
            zp_len = std::any_cast<std::string>(zp_len_it->second[0]);
            has_zp = true;
        }
        auto zp_type_it = config.params.find("zero_point_type");
        if (zp_type_it != config.params.end() && !zp_type_it->second.empty()) {
            auto zp_type_str =
                std::any_cast<std::string>(zp_type_it->second[0]);
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

        // Create test matrix with C dimensions
        std::vector<std::vector<float>> matrix_data(
            rows, std::vector<float>(cols, 0.5f));
        auto matrix = Matrix::fromData(matrix_data);

        // Parse scale_factor parameter
        double scale_value =
            extractDoubleParam("scale_factor", config.params, 0.8);
        MatrixType scale_type = extractMatrixTypeParam(
            "scale_factor_type", config.params, MatrixType::f32);

        // Create matrix with parsed values and type (static cast to correct
        // type)
        auto scale_matrix =
            Matrix::fromValue(static_cast<float>(scale_value), scale_type);

        return createMatrixAdd()
            .setMatrix(matrix)
            .setScaleFactor(scale_matrix)
            .build();

    } else if (config.type == "Matrix-Mul") {
        // Matrix-Mul: Use C matrix dimensions (m×n) and proper leading dim
        auto rows = getM(); // Use current test case M dimension
        auto cols = getN(); // Use current test case N dimension

        // Create test matrix with C dimensions (identity-like for
        // multiplication)
        std::vector<std::vector<float>> matrix_data(
            rows, std::vector<float>(cols, 1.0f));
        auto matrix = Matrix::fromData(matrix_data);

        // Parse scale_factor parameter
        double scale_value =
            extractDoubleParam("scale_factor", config.params, 1.2);
        MatrixType scale_type = extractMatrixTypeParam(
            "scale_factor_type", config.params, MatrixType::f32);

        // Create matrix with parsed values and type (static cast to correct
        // type)
        auto scale_matrix =
            Matrix::fromValue(static_cast<float>(scale_value), scale_type);

        return createMatrixMul()
            .setMatrix(matrix)
            .setScaleFactor(scale_matrix)
            .build();
    }

    throw std::runtime_error("Unknown PostOp type: " + config.type);
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

} // namespace dlp::testing::utils
