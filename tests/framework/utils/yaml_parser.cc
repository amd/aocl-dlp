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
        int                        m_current_test = 0;
        std::string                m_test_name    = {};
        YieldType                  m_yield_type = YieldType::CARTESIAN_PRODUCT;
        std::unique_ptr<MicroTest> m_current_micro_test;

        // Helper function to convert string to MatrixType
        MatrixType stringToMatrixType(const std::string& str)
        {
            if (str == "f32")
                return MatrixType::f32;
            if (str == "bf16")
                return MatrixType::bf16;
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
            return MatrixLayout::ROW_MAJOR; // default
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

        template<typename T>
        TypeErasedIterator get_value(YAML::Node node)
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
                    return ValueIterable<MatrixType>(
                               stringToMatrixType(node.as<std::string>()))
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
                    return ValueIterable<MatrixLayout>(
                               stringToMatrixLayout(node.as<std::string>()))
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
                    return ValueIterable<MatrixTag>(
                               stringToMatrixTag(node.as<std::string>()))
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

                // Otherwise, treat as a single value
                if constexpr (std::is_same_v<T, md_t>) {
                    return ValueIterable<md_t>(
                               static_cast<md_t>(node.as<int>()))
                        .begin();
                } else if constexpr (std::is_same_v<T, float>) {
                    return ValueIterable<float>(node.as<float>()).begin();
                } else if constexpr (std::is_same_v<T, double>) {
                    return ValueIterable<double>(node.as<double>()).begin();
                } else if constexpr (std::is_same_v<T, bool>) {
                    return ValueIterable<bool>(node.as<bool>()).begin();
                } else if constexpr (std::is_same_v<T, std::string>) {
                    return ValueIterable<std::string>(node.as<std::string>())
                        .begin();
                } else {
                    return ValueIterable<T>(node.as<T>()).begin();
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

            // Required fields
            iterators.a_type   = get_value<MatrixType>(node["a_type"]);
            iterators.b_type   = get_value<MatrixType>(node["b_type"]);
            iterators.c_type   = get_value<MatrixType>(node["c_type"]);
            iterators.acc_type = get_value<MatrixType>(node["acc_type"]);
            iterators.storage_format =
                get_value<MatrixLayout>(node["storage_format"]);
            iterators.m     = get_value<md_t>(node["m"]);
            iterators.n     = get_value<md_t>(node["n"]);
            iterators.k     = get_value<md_t>(node["k"]);
            iterators.alpha = get_value<double>(node["alpha"]);
            iterators.beta  = get_value<double>(node["beta"]);

            // Optional fields with defaults
            if (node["lda"]) {
                iterators.lda = get_value<md_t>(node["lda"]);
            } else {
                // Default lda to m value - use the first value from m iterator
                auto m_val = iterators.m.dereference();
                iterators.lda =
                    ValueIterable<md_t>(std::any_cast<md_t>(m_val)).begin();
            }

            if (node["ldb"]) {
                iterators.ldb = get_value<md_t>(node["ldb"]);
            } else {
                // Default ldb to k value - use the first value from k iterator
                auto k_val = iterators.k.dereference();
                iterators.ldb =
                    ValueIterable<md_t>(std::any_cast<md_t>(k_val)).begin();
            }

            if (node["ldc"]) {
                iterators.ldc = get_value<md_t>(node["ldc"]);
            } else {
                // Default ldc to m value - use the first value from m iterator
                auto m_val = iterators.m.dereference();
                iterators.ldc =
                    ValueIterable<md_t>(std::any_cast<md_t>(m_val)).begin();
            }

            iterators.trans_a = node["transA"]
                                    ? get_value<bool>(node["transA"])
                                    : ValueIterable<bool>(false).begin();
            iterators.trans_b = node["transB"]
                                    ? get_value<bool>(node["transB"])
                                    : ValueIterable<bool>(false).begin();

            // New MatrixTag-based parsing - replaces
            // reorderA/reorderB/packA/packB
            iterators.reorder_a =
                node["mtagA"]
                    ? get_value<MatrixTag>(node["mtagA"])
                    : ValueIterable<MatrixTag>(MatrixTag::NONE).begin();
            iterators.reorder_b =
                node["mtagB"]
                    ? get_value<MatrixTag>(node["mtagB"])
                    : ValueIterable<MatrixTag>(MatrixTag::NONE).begin();
            iterators.pack_a =
                node["mtagA"]
                    ? get_value<MatrixTag>(node["mtagA"])
                    : ValueIterable<MatrixTag>(MatrixTag::NONE).begin();
            iterators.pack_b =
                node["mtagB"]
                    ? get_value<MatrixTag>(node["mtagB"])
                    : ValueIterable<MatrixTag>(MatrixTag::NONE).begin();

            // Parse PostOps if present
            std::unique_ptr<PostOpsIterator> postops_iterator = nullptr;
            if (node["post_operations"]) {
                auto postops_config = parsePostOps(node["post_operations"]);
                postops_iterator    = std::make_unique<PostOpsIterator>(
                    postops_config.operations, postops_config.cartesian);
            }

            return MicroTest(iterators, m_yield_type,
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
        const MicroTest& getMicroTest()
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
        YieldType getYieldType() const { return m_yield_type; }
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

    const MicroTest& YamlParser::getMicroTest()
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
