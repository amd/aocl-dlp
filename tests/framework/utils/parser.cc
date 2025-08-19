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
        // Get alpha_type parameter (use first value from list)
        auto alpha_type_it = config.params.find("alpha_type");
        if (alpha_type_it == config.params.end()
            || alpha_type_it->second.empty()) {
            throw std::runtime_error(
                "Elementwise-PRELU requires alpha_type parameter");
        }

        auto alpha_type_str =
            std::any_cast<std::string>(alpha_type_it->second[0]);
        auto alpha_type = stringToMatrixType(alpha_type_str);
        auto alpha      = Matrix::fromValue(0.1f, alpha_type);

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
        // CLIP: Optional alpha (lower bound) and beta (upper bound) parameters
        Matrix lower_matrix, upper_matrix;

        // Check for alpha_type (lower bound)
        auto alpha_type_it = config.params.find("alpha_type");
        if (alpha_type_it != config.params.end()
            && !alpha_type_it->second.empty()) {
            auto alpha_type_str =
                std::any_cast<std::string>(alpha_type_it->second[0]);
            auto alpha_type = stringToMatrixType(alpha_type_str);

            // Check for alpha_value
            auto  alpha_value_it = config.params.find("alpha_value");
            float alpha_val      = -1.0f; // default
            if (alpha_value_it != config.params.end()
                && !alpha_value_it->second.empty()) {
                alpha_val = std::any_cast<float>(alpha_value_it->second[0]);
            }
            lower_matrix = Matrix::fromValue(alpha_val, alpha_type);
        } else {
            // Default lower bound
            lower_matrix = Matrix::fromValue(-1.0f, MatrixType::f32);
        }

        // Check for beta_type (upper bound)
        auto beta_type_it = config.params.find("beta_type");
        if (beta_type_it != config.params.end()
            && !beta_type_it->second.empty()) {
            auto beta_type_str =
                std::any_cast<std::string>(beta_type_it->second[0]);
            auto beta_type = stringToMatrixType(beta_type_str);

            // Check for beta_value
            auto  beta_value_it = config.params.find("beta_value");
            float beta_val      = 6.0f; // default
            if (beta_value_it != config.params.end()
                && !beta_value_it->second.empty()) {
                beta_val = std::any_cast<float>(beta_value_it->second[0]);
            }
            upper_matrix = Matrix::fromValue(beta_val, beta_type);
        } else {
            // Default upper bound
            upper_matrix = Matrix::fromValue(6.0f, MatrixType::f32);
        }

        return createClip()
            .setLowerBound(lower_matrix)
            .setUpperBound(upper_matrix)
            .build();

    } else if (config.type == "Scale") {
        // Parse scale_factor_len ("1" or "n"); default to "1"
        std::string scale_len_str = "1";
        auto        scale_len_it  = config.params.find("scale_factor_len");
        if (scale_len_it != config.params.end()
            && !scale_len_it->second.empty()) {
            scale_len_str = std::any_cast<std::string>(scale_len_it->second[0]);
        }

        // Get scale_factor_type if available, default to f32
        MatrixType scale_type    = MatrixType::f32;
        auto       scale_type_it = config.params.find("scale_factor_type");
        if (scale_type_it != config.params.end()
            && !scale_type_it->second.empty()) {
            auto scale_type_str =
                std::any_cast<std::string>(scale_type_it->second[0]);
            scale_type = stringToMatrixType(scale_type_str);
        }

        Matrix scale_factor;
        if (scale_len_str == "n") {
            std::vector<float> scale_data(getN(), 1.5f);
            scale_factor = Matrix::fromVector(scale_data, scale_type);
        } else {
            scale_factor = Matrix::fromValue(1.5f, scale_type);
        }

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
        builder.setScaleFactor(scale_factor);
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

        // Use hardcoded scale factor within f32 limits
        auto scale_matrix = Matrix::fromValue(0.8f, MatrixType::f32);

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

        // Use hardcoded scale factor within f32 limits
        auto scale_matrix = Matrix::fromValue(1.2f, MatrixType::f32);

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
