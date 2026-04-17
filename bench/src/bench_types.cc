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

#include "bench_types.hh"
#include "framework/operation.hh"
#include "framework/utils/parser.hh"
#include "framework/utils/postops_iterator.hh"
#include "framework/utils/yaml_parser.hh"
#include <functional>
#include <iostream>
#include <sstream>

using namespace dlp::testing::utils;
using namespace dlp::testing::framework;

namespace dlp::benchmarking {

size_t
GemmBenchConfig::hash() const
{
    size_t h = 0;
    h ^= std::hash<md_t>{}(this->m) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<md_t>{}(this->n) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<md_t>{}(this->k) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(static_cast<int>(this->a_type)) + 0x9e3779b9
         + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(static_cast<int>(this->b_type)) + 0x9e3779b9
         + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(static_cast<int>(this->c_type)) + 0x9e3779b9
         + (h << 6) + (h >> 2);
    h ^= std::hash<bool>{}(this->transA) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<bool>{}(this->transB) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<bool>{}(this->reorderA) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<bool>{}(this->reorderB) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<bool>{}(this->packA) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<bool>{}(this->packB) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
}

std::string
extractPostOpsDescription(
    const std::shared_ptr<std::vector<std::unique_ptr<IOperationParam>>>&
        post_op_params)
{
    if (!post_op_params || post_op_params->empty()) {
        return "";
    }

    std::ostringstream       postops_desc;
    std::vector<std::string> op_names;

    for (const auto& param : *post_op_params) {
        switch (param->getType()) {
            case OperationType::ElementWise: {
                const auto& ew_param =
                    static_cast<const ElementWiseParam&>(*param);
                std::string op_name;
                switch (ew_param.getOperation()) {
                    case ElementWiseOperation::Relu:
                        op_name = "Relu";
                        break;
                    case ElementWiseOperation::Prelu:
                        op_name = "Prelu";
                        break;
                    case ElementWiseOperation::Gelu_Tanh:
                        op_name = "GeluTanh";
                        break;
                    case ElementWiseOperation::Gelu_Erf:
                        op_name = "GeluErf";
                        break;
                    case ElementWiseOperation::Clip:
                        op_name = "Clip";
                        break;
                    case ElementWiseOperation::Swish:
                        op_name = "Swish";
                        break;
                    case ElementWiseOperation::Tanh:
                        op_name = "Tanh";
                        break;
                    case ElementWiseOperation::Sigmoid:
                        op_name = "Sigmoid";
                        break;
                    default:
                        op_name = "UnknownEltwise";
                        break;
                }
                op_names.push_back(op_name);
                break;
            }
            case OperationType::Bias:
                op_names.push_back("Bias");
                break;
            case OperationType::MatAdd:
                op_names.push_back("MatAdd");
                break;
            case OperationType::MatMul:
                op_names.push_back("MatMul");
                break;
            case OperationType::Scale:
                op_names.push_back("Scale");
                break;
            case OperationType::A_Quant:
                op_names.push_back("A_Quant");
                break;
            case OperationType::WOQ:
                op_names.push_back("WOQ");
                break;
            case OperationType::SymQuant:
                op_names.push_back("SymQuant");
                break;
            default:
                op_names.push_back("UnknownOp");
                break;
        }
    }

    if (!op_names.empty()) {
        postops_desc << "_PostOps";
        for (std::size_t i = 0; i < op_names.size(); ++i) {
            postops_desc << "_" << op_names[i];
        }
    }

    return postops_desc.str();
}

std::string
generateBenchmarkName(const GemmBenchConfig& config)
{
    std::ostringstream name;

    // Add data type information
    name << config.a_type << config.b_type << config.acc_type << "o"
         << config.c_type;

    // Add dimensions
    name << ",M:" << config.m << ",N:" << config.n << ",K:" << config.k;

    // Add storage format
    name << ",stor:"
         << (config.storage_format == MatrixLayout::ROW_MAJOR ? "r" : "c");

    // Add transposition
    name << (config.transA ? ",trA:t" : ",trA:n");
    name << (config.transB ? ",trB:t" : ",trB:n");

    // Add memory tags (pack takes precedence over reorder)
    std::string mtagA = config.packA ? "p" : (config.reorderA ? "r" : "n");
    std::string mtagB = config.packB ? "p" : (config.reorderB ? "r" : "n");
    name << ",mtagA:" << mtagA;
    name << ",mtagB:" << mtagB;

    if (config.sym_quant_param) {
        name << ",sym_quant";
    }

    // setAQuant / setWOQ (not in post_op_params; see
    // MicroTest::getPostOpParams)
    if (config.a_quant_param) {
        name << "_AQuant";
    }
    if (config.woq_param) {
        name << "_WOQ";
    }

    // Add detailed post_ops information
    std::string postops_desc;
    if (config.has_post_ops && config.post_op_params) {
        postops_desc = extractPostOpsDescription(config.post_op_params);
    }

    if (!postops_desc.empty()) {
        name << postops_desc;
    }

    return name.str();
}

std::string
generateBatchBenchmarkName(const BatchGemmBenchConfig& config)
{
    std::ostringstream name;

    // Add data type information (same format as GEMM)
    name << config.a_type << config.b_type << config.acc_type << "o"
         << config.c_type;

    // Add dimensions
    name << ",M:" << config.m << ",N:" << config.n << ",K:" << config.k;

    // Add storage format
    name << ",stor:"
         << (config.storage_format == MatrixLayout::ROW_MAJOR ? "r" : "c");

    // Add transposition
    name << (config.transA ? ",trA:t" : ",trA:n");
    name << (config.transB ? ",trB:t" : ",trB:n");

    // Add batch info (this is what makes it different from GEMM)
    name << ",batch:" << config.group_size;

    // Note: batch_gemm doesn't currently support reordering in benchmark
    // but we could add mtagA/mtagB here if needed in future

    return name.str();
}

std::vector<GemmBenchConfig>
loadBenchmarkConfigs(const std::string& yaml_path)
{
    std::vector<GemmBenchConfig> configs;
    const size_t                 MAX_CONFIGS = 50000; // Reasonable limit

    try {
        YamlParser parser(yaml_path, "gemm_tests");
        // Don't override yield type - let each test's product_type control it

        size_t microTestCount = parser.getMicroTestCount();

        for (std::size_t i = 0; i < microTestCount; ++i) {
            MicroTest& microTest =
                const_cast<MicroTest&>(parser.getMicroTest());
            YieldType yield_type         = microTest.getYieldType();
            size_t    total_combinations = microTest.getSize();
            size_t    test_count = std::min(total_combinations, MAX_CONFIGS);

            const char* mode_str = (yield_type == YieldType::SIMPLE_PRODUCT)
                                       ? "(simple product)"
                                       : "(cartesian product)";
            std::cout << "Test set " << i << ": Using " << test_count
                      << " out of " << total_combinations
                      << " total combinations " << mode_str << std::endl;

            // Use do-while pattern to properly iterate through MicroTest
            size_t j = 0;
            do {
                if (j >= test_count) {
                    break; // Respect the max limit
                }

                GemmBenchConfig config;
                config.a_type         = microTest.getAType();
                config.b_type         = microTest.getBType();
                config.c_type         = microTest.getCType();
                config.acc_type       = microTest.getAccType();
                config.storage_format = microTest.getStorageFormat();
                config.m              = microTest.getM();
                config.n              = microTest.getN();
                config.k              = microTest.getK();
                config.lda            = microTest.getLDA();
                config.ldb            = microTest.getLDB();
                config.ldc            = microTest.getLDC();
                config.alpha          = microTest.getAlpha();
                config.beta           = microTest.getBeta();
                config.transA         = microTest.getTransA();
                config.transB         = microTest.getTransB();
                config.reorderA       = microTest.getReorderA();
                config.reorderB       = microTest.getReorderB();
                config.packA          = microTest.getPackA();
                config.packB          = microTest.getPackB();

                // Extract fill_value if present
                if (microTest.hasFillValue()) {
                    config.has_fill_value = true;
                    const auto& fill_val  = microTest.getFillValue();
                    config.fill_lb        = fill_val.lb;
                    config.fill_ub        = fill_val.ub;
                    config.fill_dist      = fill_val.dist;
                    config.force_int_distribution =
                        fill_val.force_int_distribution;
                }

                // Extract post_ops if present
                auto post_op_params_vec = std::make_shared<
                    std::vector<std::unique_ptr<IOperationParam>>>(
                    microTest.getPostOpParams());
                if (post_op_params_vec && !post_op_params_vec->empty()) {
                    config.has_post_ops   = true;
                    config.post_op_params = post_op_params_vec;
                }

                // Same as MicroTest::configurePlan(): explicit quant setters.
                if (auto aq = microTest.getAQuantParam()) {
                    config.a_quant_param = std::make_shared<AQuantParam>(*aq);
                }
                if (auto wq = microTest.getWOQParam()) {
                    config.woq_param = std::make_shared<WOQParam>(*wq);
                }
                if (auto sq = microTest.getSymQuantParam()) {
                    config.sym_quant_param =
                        std::make_shared<SymQuantParam>(*sq);
                }

                // Generate name after populating config
                config.name = generateBenchmarkName(config);

                configs.push_back(config);
                j++;

                // Move to next iteration using standard pattern
                if (microTest.hasNext()) {
                    microTest.next();
                } else {
                    break; // No more iterations
                }
            } while (true);

            if (i < microTestCount - 1) {
                parser.next();
            }
        }

        std::cout << "Loaded " << configs.size() << " benchmark configurations"
                  << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error loading YAML configuration: " << e.what()
                  << std::endl;
    }

    return configs;
}

std::vector<BatchGemmBenchConfig>
loadBatchGemmBenchmarkConfigs(const std::string& yaml_path)
{
    std::vector<BatchGemmBenchConfig> configs;

    try {
        YamlParser parser(yaml_path, "batch_gemm_tests");

        size_t microTestCount = parser.getMicroTestCount();

        for (std::size_t i = 0; i < microTestCount; ++i) {
            MicroTest& microTest =
                const_cast<MicroTest&>(parser.getMicroTest());

            // For batch_gemm benchmarks, each YAML entry is ONE benchmark
            // We just take the first iteration's values
            BatchGemmBenchConfig config;
            config.a_type         = microTest.getAType();
            config.b_type         = microTest.getBType();
            config.c_type         = microTest.getCType();
            config.acc_type       = microTest.getAccType();
            config.storage_format = microTest.getStorageFormat();
            config.m              = microTest.getM();
            config.n              = microTest.getN();
            config.k              = microTest.getK();
            config.alpha          = microTest.getAlpha();
            config.beta           = microTest.getBeta();
            config.transA         = microTest.getTransA();
            config.transB         = microTest.getTransB();
            config.group_size     = microTest.getGroupSize();

            // Generate name using same convention as bench_gemm
            config.name = generateBatchBenchmarkName(config);

            configs.push_back(config);

            if (i < microTestCount - 1) {
                parser.next();
            }
        }

        std::cout << "Loaded " << configs.size()
                  << " batch GEMM benchmark configurations from YAML"
                  << std::endl;
        return configs;

    } catch (const std::exception& e) {
        std::cerr << "ERROR loading batch GEMM YAML: " << e.what() << std::endl;
    }

    return configs;
}

} // namespace dlp::benchmarking
