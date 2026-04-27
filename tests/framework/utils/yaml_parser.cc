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

#include <yaml-cpp/yaml.h>

// NOTE: When adding/removing/renaming fields, values, or operation types
// in this parser, update the validation tool at:
//   scripts/tools/validators/validate_yaml_configs.py
// That tool maintains a whitelist that must stay in sync with this parser
// (and with the post-op type names checked in parser.cc).

#include "classic/dlp_base_types.h"
#include "framework/iterator.hh"
#include "framework/range.hh"
#include "framework/types.hh"
#include "framework/utils/matrix_tag.hh"
#include "framework/utils/yaml_parser.hh"
#include "framework/value_iterable.hh"
#include "framework/vector_iterable.hh"
#include <iostream>

namespace dlp { namespace testing { namespace utils {
    class YamlParser::Impl
    {
      private:
        YAML::Node                 m_rootNode;
        int                        m_current_test          = 0;
        std::string                m_test_name             = {};
        std::string                m_current_test_set_name = {};
        YieldType                  m_yield_type = YieldType::CARTESIAN_PRODUCT;
        std::unique_ptr<MicroTest> m_current_micro_test;

        // Helper function to convert string to MatrixType
        MatrixType stringToMatrixType(const std::string& str)
        {
            if (str == "f32")
                return MatrixType::f32;
            if (str == "bf16")
                return MatrixType::bf16;
            if (str == "fp16")
                return MatrixType::fp16;
            if (str == "s8" || str == "int8")
                return MatrixType::s8;
            if (str == "s16" || str == "int16")
                return MatrixType::s16;
            if (str == "s32" || str == "int32")
                return MatrixType::s32;
            if (str == "u8" || str == "uint8")
                return MatrixType::u8;
            if (str == "u16" || str == "uint16")
                return MatrixType::u16;
            if (str == "u32" || str == "uint32")
                return MatrixType::u32;
            if (str == "s4")
                return MatrixType::s4;
            if (str == "u4")
                return MatrixType::u4;
            return MatrixType::f32; // default
        }

        // Helper function to convert string to MatrixLayout
        MatrixLayout stringToMatrixLayout(const std::string& str)
        {
            if (str == "row-major" || str == "row_major" || str == "ROW_MAJOR")
                return MatrixLayout::ROW_MAJOR;
            if (str == "column-major" || str == "column_major"
                || str == "COLUMN_MAJOR")
                return MatrixLayout::COLUMN_MAJOR;
            throw std::runtime_error(
                "Unknown storage_format: '" + str
                + "'. Valid: row-major, row_major, ROW_MAJOR, "
                  "column-major, column_major, COLUMN_MAJOR");
        }

        // Helper function to convert string to MatrixTag
        MatrixTag stringToMatrixTag(const std::string& str)
        {
            if (str == "none")
                return MatrixTag::NONE;
            if (str == "reorder")
                return MatrixTag::REORDER;
            if (str == "pack")
                return MatrixTag::PACK;
            return MatrixTag::NONE; // default
        }

        /**
         * @brief Parse a single fill pattern configuration from YAML node
         *
         * @param pattern_node YAML node containing pattern configuration
         * @param pattern_config Output FillPatternConfig to populate
         * @throws std::runtime_error if configuration is invalid
         */
        void parseFillPatternConfig(const YAML::Node&  pattern_node,
                                    FillPatternConfig& pattern_config)
        {
            // Determine pattern type (default to "static")
            std::string type_str = "static";
            if (pattern_node["type"]) {
                type_str = pattern_node["type"].as<std::string>();
            }

            // Parse based on type
            if (type_str == "static") {
                pattern_config.type = PatternType::Static;
                if (!pattern_node["values"]) {
                    throw std::runtime_error(
                        "Static pattern requires 'values' field");
                }
                for (const auto& val : pattern_node["values"]) {
                    pattern_config.values.push_back(val.as<double>());
                }
            } else if (type_str == "modulo") {
                pattern_config.type = PatternType::Modulo;
                if (!pattern_node["lb"] || !pattern_node["ub"]) {
                    throw std::runtime_error(
                        "Modulo pattern requires 'lb' and 'ub' fields");
                }
                pattern_config.lb = pattern_node["lb"].as<double>();
                pattern_config.ub = pattern_node["ub"].as<double>();
                if (pattern_node["step"]) {
                    pattern_config.step = pattern_node["step"].as<double>();
                }
                if (pattern_node["offset"]) {
                    pattern_config.offset = pattern_node["offset"].as<double>();
                }
            } else if (type_str == "linear") {
                pattern_config.type = PatternType::Linear;
                if (!pattern_node["lb"] || !pattern_node["ub"]) {
                    throw std::runtime_error(
                        "Linear pattern requires 'lb' and 'ub' fields");
                }
                pattern_config.lb = pattern_node["lb"].as<double>();
                pattern_config.ub = pattern_node["ub"].as<double>();
                if (pattern_node["step"]) {
                    pattern_config.step = pattern_node["step"].as<double>();
                }
                if (pattern_node["multiplier"]) {
                    pattern_config.multiplier =
                        pattern_node["multiplier"].as<double>();
                }
                if (pattern_node["offset"]) {
                    pattern_config.offset = pattern_node["offset"].as<double>();
                }
            } else if (type_str == "sequence") {
                pattern_config.type = PatternType::Sequence;
                if (!pattern_node["lb"]) {
                    throw std::runtime_error(
                        "Sequence pattern requires 'lb' field");
                }
                pattern_config.lb = pattern_node["lb"].as<double>();
                if (pattern_node["ub"]) {
                    pattern_config.ub = pattern_node["ub"].as<double>();
                } else {
                    pattern_config.ub = 1000.0; // Default large value
                }
                if (pattern_node["step"]) {
                    pattern_config.step = pattern_node["step"].as<double>();
                }
            } else {
                throw std::runtime_error("Unknown fill_pattern type: "
                                         + type_str
                                         + ". Allowed: static, modulo, "
                                           "linear, sequence");
            }

            // Validate configuration
            std::string error_msg;
            if (!pattern_config.validate(error_msg)) {
                throw std::runtime_error("Invalid fill_pattern config: "
                                         + error_msg);
            }
        }

        template<typename T>
        TypeErasedIterator get_value(
            YAML::Node node,
            YieldType  yield_type = YieldType::CARTESIAN_PRODUCT)
        {
            // Special handling for enum types - no range support, only lists or
            // single values
            if constexpr (std::is_same_v<T, MatrixType>) {
                if (node.IsSequence()) {
                    std::vector<MatrixType> values;
                    for (const auto& item : node) {
                        values.push_back(
                            stringToMatrixType(item.as<std::string>()));
                    }
                    return VectorIterable<MatrixType>(values).begin();
                } else {
                    // For SIMPLE_PRODUCT, single values should have infinite
                    // extent to repeat across iterations
                    bool report_inf = (yield_type == YieldType::SIMPLE_PRODUCT);
                    return ValueIterable<MatrixType>(
                               stringToMatrixType(node.as<std::string>()),
                               report_inf)
                        .begin();
                }
            } else if constexpr (std::is_same_v<T, MatrixLayout>) {
                if (node.IsSequence()) {
                    std::vector<MatrixLayout> values;
                    for (const auto& item : node) {
                        values.push_back(
                            stringToMatrixLayout(item.as<std::string>()));
                    }
                    return VectorIterable<MatrixLayout>(values).begin();
                } else {
                    // For SIMPLE_PRODUCT, single values should be infinite to
                    // extend
                    bool report_inf = (yield_type == YieldType::SIMPLE_PRODUCT);
                    return ValueIterable<MatrixLayout>(
                               stringToMatrixLayout(node.as<std::string>()),
                               report_inf)
                        .begin();
                }
            } else if constexpr (std::is_same_v<T, MatrixTag>) {
                if (node.IsSequence()) {
                    std::vector<MatrixTag> values;
                    for (const auto& item : node) {
                        values.push_back(
                            stringToMatrixTag(item.as<std::string>()));
                    }
                    return VectorIterable<MatrixTag>(values).begin();
                } else {
                    // For SIMPLE_PRODUCT, single values should be infinite to
                    // extend
                    bool report_inf = (yield_type == YieldType::SIMPLE_PRODUCT);
                    return ValueIterable<MatrixTag>(
                               stringToMatrixTag(node.as<std::string>()),
                               report_inf)
                        .begin();
                }
            } else {
                // Numerical types - support ranges, lists, and single values

                // Check if this is a range object (has ub/lb/step fields)
                if (node.IsMap()
                    && (node["ub"] || node["lb"] || node["step"])) {
                    // Set default values
                    T ub = T{}, lb = T{}, step = T{ 1 };

                    // Override with actual values if present
                    if (node["lb"]) {
                        lb = node["lb"].as<T>();
                    }
                    if (node["ub"]) {
                        ub = node["ub"].as<T>();
                    }
                    if (node["step"]) {
                        step = node["step"].as<T>();
                    }
                    // Make the range inclusive by adding step to upper bound
                    TypeErasedRange<T> range(lb, ub + step, step);
                    return range.begin();
                }

                // Check if this is a sequence (list/array)
                if (node.IsSequence()) {
                    if constexpr (std::is_same_v<T, md_t>) {
                        std::vector<md_t> values;
                        for (const auto& item : node) {
                            values.push_back(static_cast<md_t>(item.as<int>()));
                        }
                        return VectorIterable<md_t>(values).begin();
                    } else if constexpr (std::is_same_v<T, int>) {
                        std::vector<int> values;
                        for (const auto& item : node) {
                            values.push_back(item.as<int>());
                        }
                        return VectorIterable<int>(values).begin();
                    } else if constexpr (std::is_same_v<T, float>) {
                        std::vector<float> values;
                        for (const auto& item : node) {
                            values.push_back(item.as<float>());
                        }
                        return VectorIterable<float>(values).begin();
                    } else if constexpr (std::is_same_v<T, double>) {
                        std::vector<double> values;
                        for (const auto& item : node) {
                            values.push_back(item.as<double>());
                        }
                        return VectorIterable<double>(values).begin();
                    } else if constexpr (std::is_same_v<T, bool>) {
                        std::vector<bool> values;
                        for (const auto& item : node) {
                            values.push_back(item.as<bool>());
                        }
                        return VectorIterable<bool>(values).begin();
                    } else {
                        std::vector<T> values;
                        for (const auto& item : node) {
                            values.push_back(item.as<T>());
                        }
                        return VectorIterable<T>(values).begin();
                    }
                }

                // Otherwise, for SIMPLE_PRODUCT, make single values have
                // infinite extent to repeat across iterations
                bool report_inf = (yield_type == YieldType::SIMPLE_PRODUCT);
                if constexpr (std::is_same_v<T, md_t>) {
                    return ValueIterable<md_t>(
                               static_cast<md_t>(node.as<int>()), report_inf)
                        .begin();
                } else if constexpr (std::is_same_v<T, float>) {
                    return ValueIterable<float>(node.as<float>(), report_inf)
                        .begin();
                } else if constexpr (std::is_same_v<T, double>) {
                    return ValueIterable<double>(node.as<double>(), report_inf)
                        .begin();
                } else if constexpr (std::is_same_v<T, bool>) {
                    return ValueIterable<bool>(node.as<bool>(), report_inf)
                        .begin();
                } else if constexpr (std::is_same_v<T, std::string>) {
                    return ValueIterable<std::string>(node.as<std::string>(),
                                                      report_inf)
                        .begin();
                } else {
                    return ValueIterable<T>(node.as<T>(), report_inf).begin();
                }
            }
        }

        struct PostOpsConfig
        {
            std::vector<PostOpsIterator::PostOpConfig> operations;
            bool                                       cartesian;
        };

        PostOpsConfig parsePostOps(YAML::Node postops_node)
        {
            PostOpsConfig config;
            config.cartesian = postops_node["cartesian"].as<bool>(false);

            if (postops_node["operations"]) {
                for (const auto& op_node : postops_node["operations"]) {
                    PostOpsIterator::PostOpConfig op_config;
                    op_config.type = op_node["type"].as<std::string>();

                    if (op_node["params"]) {
                        for (const auto& param : op_node["params"]) {
                            std::string param_name =
                                param.first.as<std::string>();

                            // Parse parameter values (could be lists or single
                            // values)
                            std::vector<std::any> param_values;
                            if (param.second.IsSequence()) {
                                for (const auto& value : param.second) {
                                    param_values.push_back(
                                        value.as<std::string>());
                                }
                            } else {
                                param_values.push_back(
                                    param.second.as<std::string>());
                            }

                            op_config.params[param_name] = param_values;
                        }
                    }

                    config.operations.push_back(op_config);
                }
            }

            return config;
        }

        MicroTest getMicroTestFromNode(YAML::Node node)
        {
            TestCaseIterators iterators;

            // Store the test set name from the YAML node
            if (node["name"]) {
                m_current_test_set_name = node["name"].as<std::string>();
            } else {
                m_current_test_set_name =
                    "yaml_" + std::to_string(m_current_test);
            }

            // Determine yield type for this test (used for ValueIterable
            // behavior)
            YieldType yield_type_for_parsing = m_yield_type; // default
            if (node["product_type"]) {
                auto        product_type_node = node["product_type"];
                std::string product_type_str;

                if (product_type_node.IsSequence()) {
                    if (product_type_node.size() > 0) {
                        product_type_str =
                            product_type_node[0].as<std::string>();
                    }
                } else {
                    product_type_str = product_type_node.as<std::string>();
                }

                if (product_type_str == "simple") {
                    yield_type_for_parsing = YieldType::SIMPLE_PRODUCT;
                } else if (product_type_str == "cartesian") {
                    yield_type_for_parsing = YieldType::CARTESIAN_PRODUCT;
                }
            }

            // Required fields
            iterators.a_type =
                get_value<MatrixType>(node["a_type"], yield_type_for_parsing);
            iterators.b_type =
                get_value<MatrixType>(node["b_type"], yield_type_for_parsing);
            iterators.c_type =
                get_value<MatrixType>(node["c_type"], yield_type_for_parsing);
            iterators.acc_type =
                get_value<MatrixType>(node["acc_type"], yield_type_for_parsing);
            iterators.storage_format = get_value<MatrixLayout>(
                node["storage_format"], yield_type_for_parsing);
            iterators.m = get_value<md_t>(node["m"], yield_type_for_parsing);
            iterators.n = get_value<md_t>(node["n"], yield_type_for_parsing);
            iterators.k = get_value<md_t>(node["k"], yield_type_for_parsing);
            iterators.alpha =
                get_value<double>(node["alpha"], yield_type_for_parsing);
            iterators.beta =
                get_value<double>(node["beta"], yield_type_for_parsing);

            // Optional fields with defaults
            bool report_inf =
                (yield_type_for_parsing == YieldType::SIMPLE_PRODUCT);

            // Initialize transpose flags first (needed for LD calculation)
            iterators.trans_a =
                node["transA"]
                    ? get_value<bool>(node["transA"], yield_type_for_parsing)
                    : ValueIterable<bool>(false, report_inf).begin();
            iterators.trans_b =
                node["transB"]
                    ? get_value<bool>(node["transB"], yield_type_for_parsing)
                    : ValueIterable<bool>(false, report_inf).begin();

            // Now compute leading dimensions (requires transpose flags)
            if (node["lda"]) {
                iterators.lda =
                    get_value<md_t>(node["lda"], yield_type_for_parsing);
            } else {
                // LD not specified; use -1 sentinel so that 0 remains
                // available as an explicit boundary-test value.
                iterators.lda = ValueIterable<md_t>(-1, report_inf).begin();
            }

            if (node["ldb"]) {
                iterators.ldb =
                    get_value<md_t>(node["ldb"], yield_type_for_parsing);
            } else {
                // LD not specified; use -1 sentinel so that 0 remains
                // available as an explicit boundary-test value.
                iterators.ldb = ValueIterable<md_t>(-1, report_inf).begin();
            }

            if (node["ldc"]) {
                iterators.ldc =
                    get_value<md_t>(node["ldc"], yield_type_for_parsing);
            } else {
                // LD not specified; use -1 sentinel so that 0 remains
                // available as an explicit boundary-test value.
                iterators.ldc = ValueIterable<md_t>(-1, report_inf).begin();
            }

            // New MatrixTag-based parsing
            // Simplified design: single MatrixTag per matrix
            // Both reorder and pack getters check the same MatrixTag value
            iterators.mtag_a =
                node["mtagA"]
                    ? get_value<MatrixTag>(node["mtagA"],
                                           yield_type_for_parsing)
                    : ValueIterable<MatrixTag>(MatrixTag::NONE, report_inf)
                          .begin();
            iterators.mtag_b =
                node["mtagB"]
                    ? get_value<MatrixTag>(node["mtagB"],
                                           yield_type_for_parsing)
                    : ValueIterable<MatrixTag>(MatrixTag::NONE, report_inf)
                          .begin();

            // Parse group_size for batch GEMM tests (only if present)
            if (node["group_size"]) {
                iterators.has_group_size = true;
                iterators.group_size =
                    get_value<md_t>(node["group_size"], yield_type_for_parsing);
            } else {
                iterators.has_group_size = false;
                // Default: single matrix per group (safe for regular GEMM
                // tests)
                iterators.group_size =
                    ValueIterable<md_t>(1, report_inf).begin();
            }

            // Parse fill_value if present
            if (node["fill_value"]) {
                iterators.has_fill_value = true;
                auto fill_node           = node["fill_value"];

                // Extract lb (lower bound)
                if (fill_node["lb"]) {
                    iterators.fill_value.lb = fill_node["lb"].as<double>();
                }

                // Extract ub (upper bound)
                if (fill_node["ub"]) {
                    iterators.fill_value.ub = fill_node["ub"].as<double>();
                }

                // Extract dist (distribution type)
                if (fill_node["dist"]) {
                    iterators.fill_value.dist =
                        fill_node["dist"].as<std::string>();
                }

                // Extract force_int_distribution (default: true)
                if (fill_node["force_int_distribution"]) {
                    iterators.fill_value.force_int_distribution =
                        fill_node["force_int_distribution"].as<bool>();
                }
            }

            // Parse fill_pattern if present
            if (node["fill_pattern"]) {
                iterators.has_fill_pattern = true;
                auto fill_pattern_node     = node["fill_pattern"];

                // Check if fill_pattern is a list (multiple patterns) or a
                // single pattern
                if (fill_pattern_node.IsSequence()) {
                    // Multiple patterns with potential apply_matrix
                    // specification
                    for (const auto& pattern_node : fill_pattern_node) {
                        MatrixFillPattern matrix_pattern;

                        // Parse apply_matrix if present
                        if (pattern_node["apply_matrix"]) {
                            // Initialize all to false, then set specified ones
                            // to true
                            matrix_pattern.apply_to_A = false;
                            matrix_pattern.apply_to_B = false;
                            matrix_pattern.apply_to_C = false;

                            for (const auto& mat :
                                 pattern_node["apply_matrix"]) {
                                std::string mat_name = mat.as<std::string>();
                                if (mat_name == "A") {
                                    matrix_pattern.apply_to_A = true;
                                } else if (mat_name == "B") {
                                    matrix_pattern.apply_to_B = true;
                                } else if (mat_name == "C") {
                                    matrix_pattern.apply_to_C = true;
                                } else {
                                    throw std::runtime_error(
                                        "Invalid matrix name in apply_matrix: "
                                        + mat_name + ". Allowed: A, B, C");
                                }
                            }
                        }
                        // If no apply_matrix is specified, matrix_pattern keeps
                        // its default of applying to all matrices (see
                        // MatrixFillPattern initialization).

                        // Parse pattern configuration
                        parseFillPatternConfig(pattern_node,
                                               matrix_pattern.pattern);

                        iterators.fill_patterns.push_back(matrix_pattern);
                    }
                } else {
                    // Single pattern (backward compatibility) - applies to all
                    // matrices
                    MatrixFillPattern matrix_pattern;
                    parseFillPatternConfig(fill_pattern_node,
                                           matrix_pattern.pattern);
                    iterators.fill_patterns.push_back(matrix_pattern);
                }
            }

            // Parse tolerances if present
            if (node["tolerances"]) {
                iterators.has_tolerances = true;
                auto tol_node            = node["tolerances"];

                // Extract relative tolerance multiplier
                if (tol_node["relative"]) {
                    double rel_val = tol_node["relative"].as<double>();
                    if (rel_val < 0.0) {
                        throw std::runtime_error(
                            "Relative tolerance multiplier must be "
                            "non-negative, got: "
                            + std::to_string(rel_val));
                    }
                    // Reasonable upper bound check (500000 = ~10000x
                    // default 50.0)
                    if (rel_val > 500000.0) {
                        throw std::runtime_error(
                            "Relative tolerance multiplier is unreasonably "
                            "large: "
                            + std::to_string(rel_val)
                            + ". Consider using a value < 500000");
                    }
                    iterators.tolerances.relative = rel_val;
                }

                // Extract absolute tolerance multiplier
                if (tol_node["absolute"]) {
                    double abs_val = tol_node["absolute"].as<double>();
                    if (abs_val < 0.0) {
                        throw std::runtime_error(
                            "Absolute tolerance multiplier must be "
                            "non-negative, got: "
                            + std::to_string(abs_val));
                    }
                    // Reasonable upper bound check (500000 = ~10000x
                    // default 50.0)
                    if (abs_val > 500000.0) {
                        throw std::runtime_error(
                            "Absolute tolerance multiplier is unreasonably "
                            "large: "
                            + std::to_string(abs_val)
                            + ". Consider using a value < 500000");
                    }
                    iterators.tolerances.absolute = abs_val;
                }
            }

            // Parse PostOps and PreOps if present.
            // Pre-operations (e.g. WOQ for bf16s4) and post_operations are
            // combined into one iterator: pre_ops first, then post_ops, so
            // both appear in the benchmark name and in execution.
            std::unique_ptr<PostOpsIterator> postops_iterator = nullptr;
            bool has_post = (node["post_operations"] ? true : false);
            bool has_pre  = (node["pre_operations"] ? true : false);
            if (has_post && has_pre) {
                auto preops_config  = parsePostOps(node["pre_operations"]);
                auto postops_config = parsePostOps(node["post_operations"]);
                std::vector<PostOpsIterator::PostOpConfig> merged;
                merged.reserve(preops_config.operations.size()
                               + postops_config.operations.size());
                merged.insert(merged.end(), preops_config.operations.begin(),
                              preops_config.operations.end());
                merged.insert(merged.end(), postops_config.operations.begin(),
                              postops_config.operations.end());
                postops_iterator = std::make_unique<PostOpsIterator>(
                    merged, false); // cartesian false = one sequence
            } else if (has_pre) {
                auto preops_config = parsePostOps(node["pre_operations"]);
                postops_iterator   = std::make_unique<PostOpsIterator>(
                    preops_config.operations, preops_config.cartesian);
            } else if (has_post) {
                auto postops_config = parsePostOps(node["post_operations"]);
                postops_iterator    = std::make_unique<PostOpsIterator>(
                    postops_config.operations, postops_config.cartesian);
            }

            // ================================================================
            // VALIDATION: batch_gemm_tests specific rules
            // ================================================================
            if (m_test_name == "batch_gemm_tests" && iterators.has_group_size) {
                // Determine if multi-group mode by counting group_size values
                std::vector<md_t> group_size_values;
                auto              gs_copy = iterators.group_size;
                do {
                    md_t val = std::any_cast<md_t>(gs_copy.dereference());
                    group_size_values.push_back(val);
                    if (!gs_copy.has_next())
                        break;
                    gs_copy.increment();
                } while (true);

                bool is_multi_group = (group_size_values.size() > 1);

                if (is_multi_group) {
                    // Multi-group mode: enforce SIMPLE_PRODUCT
                    if (yield_type_for_parsing
                        == YieldType::CARTESIAN_PRODUCT) {
                        std::string test_name_str =
                            node["name"] ? node["name"].as<std::string>()
                                         : "unnamed";
                        throw std::runtime_error(
                            "ERROR: batch_gemm_tests with multi-group "
                            "(group_size list length > 1) "
                            "requires product_type: 'simple'. Cartesian "
                            "product "
                            "is only allowed "
                            "for single-group batch testing (scalar "
                            "group_size).\n"
                            "Test name: "
                            + test_name_str
                            + "\n"
                              "group_size has "
                            + std::to_string(group_size_values.size())
                            + " values.\n"
                              "Hint: Add 'product_type: \"simple\"' or use "
                              "scalar group_size.");
                    }

                    // Force simple product for multi-group
                    yield_type_for_parsing = YieldType::SIMPLE_PRODUCT;

                    // TODO: Add validation for parameter list lengths
                    // NOTE: Validation temporarily disabled due to
                    // TypeErasedIterator copy behavior causing infinite loops
                    // with scalar values. This needs to be fixed in the
                    // iterator implementation. For now, YAML tests will fail at
                    // runtime if parameters have incorrect lengths, which is
                    // acceptable for MVP.

                    // The validation should ensure that in multi-group mode:
                    // - All critical params (m, n, k, lda, ldb, ldc, alpha,
                    // beta)
                    //   have either 1 value (scalar) or N values (one per
                    //   group)
                    // - For now, users must ensure this manually in their YAML
                    // configs
                }
            }

            return MicroTest(iterators, yield_type_for_parsing,
                             std::move(postops_iterator));
        }

      public:
        Impl(const std::string& filename, std::string test_name)
            : m_test_name(test_name)
        {
            try {
                m_rootNode = YAML::LoadFile(filename);
                if (!m_rootNode || m_rootNode.IsNull()) {
                    throw std::runtime_error(
                        "Failed to load YAML file or file is empty");
                }
            } catch (const YAML::Exception& e) {
                throw std::runtime_error(
                    "Failed to load YAML configuration file: "
                    + std::string(e.what()));
            }
            // current_test is already initialized to 0 by default in the header
        }
        MicroTest& getMicroTest()
        {
            if (!m_current_micro_test) {
                m_current_micro_test =
                    std::make_unique<MicroTest>(getMicroTestFromNode(
                        m_rootNode[m_test_name][m_current_test]));
            }
            return *m_current_micro_test;
        }
        void next()
        {
            // Increment current_test
            m_current_test++;
            // Clear cached micro test so it gets regenerated
            m_current_micro_test.reset();
        }
        void reset()
        {
            m_current_test = 0;
            m_current_micro_test.reset();
        }
        size_t getMicroTestCount() const
        {
            return m_rootNode[m_test_name].size();
        }
        void setYieldType(YieldType yield_type) { m_yield_type = yield_type; }
        YieldType   getYieldType() const { return m_yield_type; }
        std::string getTestSuiteName() const { return m_test_name; }
        std::string getCurrentTestName() const
        {
            return m_current_test_set_name;
        }
    };

    YamlParser::YamlParser(const std::string& filename, std::string test_name)
        : m_impl(std::make_unique<Impl>(filename, test_name))
    {
        if (!m_impl) {
            throw std::runtime_error("Failed to create YamlParser");
        }
        m_impl->reset();
    }

    // Destructor needs to be defined in the source file for unique_ptr<Impl>
    YamlParser::~YamlParser() = default;

    MicroTest& YamlParser::getMicroTest()
    {
        return m_impl->getMicroTest();
    }

    void YamlParser::next()
    {
        m_impl->next();
    }

    void YamlParser::reset()
    {
        m_impl->reset();
    }

    size_t YamlParser::getMicroTestCount() const
    {
        return m_impl->getMicroTestCount();
    }

    void YamlParser::setYieldType(YieldType yield_type)
    {
        m_impl->setYieldType(yield_type);
    }

    YieldType YamlParser::getYieldType() const
    {
        return m_impl->getYieldType();
    }

    std::string YamlParser::getTestSuiteName() const
    {
        return m_impl->getTestSuiteName();
    }

    std::string YamlParser::getCurrentTestName() const
    {
        return m_impl->getCurrentTestName();
    }

    // Convenience methods to access current MicroTest properties
    // FIXME: Remove these methods and use the getMicroTest() method instead
    MatrixType YamlParser::getAType() const
    {
        return m_impl->getMicroTest().getAType();
    }

    MatrixType YamlParser::getBType() const
    {
        return m_impl->getMicroTest().getBType();
    }

    MatrixType YamlParser::getCType() const
    {
        return m_impl->getMicroTest().getCType();
    }

    MatrixType YamlParser::getAccType() const
    {
        return m_impl->getMicroTest().getAccType();
    }

    MatrixLayout YamlParser::getStorageFormat() const
    {
        return m_impl->getMicroTest().getStorageFormat();
    }

    md_t YamlParser::getM() const
    {
        return m_impl->getMicroTest().getM();
    }

    md_t YamlParser::getN() const
    {
        return m_impl->getMicroTest().getN();
    }

    md_t YamlParser::getK() const
    {
        return m_impl->getMicroTest().getK();
    }

    md_t YamlParser::getLDA() const
    {
        return m_impl->getMicroTest().getLDA();
    }

    md_t YamlParser::getLDB() const
    {
        return m_impl->getMicroTest().getLDB();
    }

    md_t YamlParser::getLDC() const
    {
        return m_impl->getMicroTest().getLDC();
    }

    double YamlParser::getAlpha() const
    {
        return m_impl->getMicroTest().getAlpha();
    }

    double YamlParser::getBeta() const
    {
        return m_impl->getMicroTest().getBeta();
    }

    bool YamlParser::getTransA() const
    {
        return m_impl->getMicroTest().getTransA();
    }

    bool YamlParser::getTransB() const
    {
        return m_impl->getMicroTest().getTransB();
    }

    bool YamlParser::getReorderA() const
    {
        return m_impl->getMicroTest().getReorderA();
    }

    bool YamlParser::getReorderB() const
    {
        return m_impl->getMicroTest().getReorderB();
    }

    bool YamlParser::getPackA() const
    {
        return m_impl->getMicroTest().getPackA();
    }

    bool YamlParser::getPackB() const
    {
        return m_impl->getMicroTest().getPackB();
    }

}}} // namespace dlp::testing::utils
