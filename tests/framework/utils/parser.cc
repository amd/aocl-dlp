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
            auto sf_type_str =
                std::any_cast<std::string>(sf_type_it->second[0]);
            sf_type = stringToMatrixType(sf_type_str);
            has_sf  = true;
        }

        auto sf_len_it = config.params.find("scale_factor_len");
        if (sf_len_it != config.params.end() && !sf_len_it->second.empty()) {
            sf_len = std::any_cast<std::string>(sf_len_it->second[0]);
            has_sf = true;
        }

        // If user provided explicit scale_factor value, use it
        auto sf_it = config.params.find("scale_factor");
        if (sf_it != config.params.end() && !sf_it->second.empty()) {
            sf_value = extractDoubleParam("scale_factor", config.params, 2.5);
            has_sf   = true;
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
            auto zp_type_str =
                std::any_cast<std::string>(zp_type_it->second[0]);
            zp_type = stringToMatrixType(zp_type_str);
            has_zp  = true;
        }

        auto zp_len_it = config.params.find("zero_point_len");
        if (zp_len_it != config.params.end() && !zp_len_it->second.empty()) {
            zp_len = std::any_cast<std::string>(zp_len_it->second[0]);
            has_zp = true;
        }

        // If user provided explicit zero_point value, use it
        auto zp_it = config.params.find("zero_point");
        if (zp_it != config.params.end() && !zp_it->second.empty()) {
            zp_value = extractDoubleParam("zero_point", config.params, 10.0);
            has_zp   = true;
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
            alpha_type = extractMatrixTypeParam("alpha_type", config.params);
            beta_type  = alpha_type;
        } else if (!has_alpha_type && has_beta_type) {
            // Only beta specified, use it for both
            beta_type  = extractMatrixTypeParam("beta_type", config.params);
            alpha_type = beta_type;
        } else {
            // Both specified OR neither specified - extract both
            alpha_type = extractMatrixTypeParam("alpha_type", config.params);
            beta_type  = extractMatrixTypeParam("beta_type", config.params);

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

        // Extract matrix_type parameters (default to f32)
        MatrixType matrix_type = extractMatrixTypeParam(
            "matrix_type", config.params, MatrixType::f32);

        // Create test matrix with C dimensions and specified type
        std::vector<std::vector<float>> matrix_data(
            rows, std::vector<float>(cols, 0.5f));
        auto matrix = Matrix::fromData(matrix_data, matrix_type);

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

        // Extract matrix-type parameters (default to f32)
        MatrixType matrix_type = extractMatrixTypeParam(
            "matrix_type", config.params, MatrixType::f32);

        // Create test matrix with C dimensions (identity-like for
        // multiplication)
        std::vector<std::vector<float>> matrix_data(
            rows, std::vector<float>(cols, 1.0f));
        auto matrix = Matrix::fromData(matrix_data, matrix_type);

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
