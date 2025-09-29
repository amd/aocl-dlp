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

// ============================================================================
// INCLUDES AND DEPENDENCIES
// ============================================================================

#include "classic/aocl_bf16_type.h"
#include "framework/operation.hh"
#include "framework/ual.hh"
#include "framework/ual_factory.hh"
#include "framework/utils/arg_parser.hh"
#include "framework/utils/yaml_parser.hh"
#include "test_config.hh"
#include "utils/conversion_utils.hh"

#include <algorithm>
#include <filesystem>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace dlp::testing::utils;
using namespace dlp::testing::framework;
using namespace dlp::testing::framework::postops;

// ============================================================================
// GLOBAL CONFIGURATION VARIABLES
// ============================================================================

// Global variable to store configurable YAML file path
// This can be set via command line arguments or defaults to the built-in config
static std::string g_yaml_config_file = TEST_CONFIG_DIR
    "/gemm_test_config.yaml";

// ============================================================================
// CONFIGURATION STRUCTURES AND TYPES
// ============================================================================

// Configuration structure for YAML-driven tests
struct GemmTestConfig
{
    std::string name   = "";
    MatrixType  a_type = MatrixType::f32, b_type = MatrixType::f32,
               c_type = MatrixType::f32, acc_type = MatrixType::f32;
    MatrixLayout                storage_format = MatrixLayout::ROW_MAJOR;
    md_t                        m = 0, n = 0, k = 0;
    md_t                        lda = 0, ldb = 0, ldc = 0;
    double                      alpha = 1.0, beta = 0.0;
    bool                        transA = false, transB = false;
    bool                        reorderA = false, reorderB = false;
    bool                        packA = false, packB = false;
    std::shared_ptr<IOperation> postops_dlp = nullptr;
    std::shared_ptr<IOperation> postops_ref = nullptr;
    bool                        has_postops = false;

    // Default constructor (required by GoogleTest)
    GemmTestConfig() = default;

    // Constructor for easy initialization
    GemmTestConfig(const std::string&          test_name,
                   MatrixType                  a_type_,
                   MatrixType                  b_type_,
                   MatrixType                  c_type_,
                   MatrixType                  acc_type_,
                   MatrixLayout                storage_format_,
                   md_t                        m_,
                   md_t                        n_,
                   md_t                        k_,
                   md_t                        lda_,
                   md_t                        ldb_,
                   md_t                        ldc_,
                   double                      alpha_,
                   double                      beta_,
                   bool                        transA_,
                   bool                        transB_,
                   bool                        reorderA_,
                   bool                        reorderB_,
                   bool                        packA_,
                   bool                        packB_,
                   std::shared_ptr<IOperation> postops_dlp_ = nullptr,
                   std::shared_ptr<IOperation> postops_ref_ = nullptr)
        : name(test_name)
        , a_type(a_type_)
        , b_type(b_type_)
        , c_type(c_type_)
        , acc_type(acc_type_)
        , storage_format(storage_format_)
        , m(m_)
        , n(n_)
        , k(k_)
        , lda(lda_)
        , ldb(ldb_)
        , ldc(ldc_)
        , alpha(alpha_)
        , beta(beta_)
        , transA(transA_)
        , transB(transB_)
        , reorderA(reorderA_)
        , reorderB(reorderB_)
        , packA(packA_)
        , packB(packB_)
        , postops_dlp(postops_dlp_)
        , postops_ref(postops_ref_)
        , has_postops(postops_dlp_ != nullptr || postops_ref_ != nullptr)
    {
    }

    // Equality operator for Google Test
    bool operator==(const GemmTestConfig& other) const
    {
        return name == other.name && m == other.m && n == other.n
               && k == other.k && lda == other.lda && ldb == other.ldb
               && ldc == other.ldc && alpha == other.alpha && beta == other.beta
               && transA == other.transA && transB == other.transB
               && reorderA == other.reorderA && reorderB == other.reorderB
               && packA == other.packA && packB == other.packB
               && has_postops == other.has_postops
               && postops_dlp == other.postops_dlp
               && postops_ref == other.postops_ref;
    }
};

// ============================================================================
// HELPER FUNCTIONS - VALIDATION
// ============================================================================

bool
check_valid_params(const GemmTestConfig& config)
{
    // This function follows the exact logic from AOCL_GEMM_CHECK macro in
    // aocl_gemm_check.h with additional handling for reordered matrices
    // (which have custom memory layouts)
    bool col_stored = (config.storage_format == MatrixLayout::COLUMN_MAJOR);
    bool row_stored = (config.storage_format == MatrixLayout::ROW_MAJOR);

    bool nota = !config.transA; // not transposed A
    bool notb = !config.transB; // not transposed B
    bool ta   = config.transA;  // transposed A
    bool tb   = config.transB;  // transposed B

    // Check basic dimensions - must be positive (same as macro: m <= 0, n <= 0,
    // k <= 0)
    if (config.m <= 0) {
        return false; // info = 4 in macro
    }
    if (config.n <= 0) {
        return false; // info = 5 in macro
    }
    if (config.k <= 0) {
        return false; // info = 6 in macro
    }

    // Column Major API support checks
    if (col_stored) {
        switch (config.a_type) {
            // u8 api does not support column major
            case MatrixType::u8:
                return false;

            // s8s8 api supports column major without post-ops
            // s8s8ou8 does not support column major
            case MatrixType::s8:
                if (config.c_type == MatrixType::u8)
                    return false;
                if (config.has_postops)
                    return false;
                break;

            case MatrixType::f32:
            case MatrixType::bf16:
                break;

            // Add cases for other data types as needed
            default:
                break;
        }
    }

    // Leading dimension checks for matrix A (info = 9 in macro)
    // Skip leading dimension checks for reordered matrices as they have custom
    // layouts
    if (!config.reorderA) {
        if (row_stored
            && ((nota && (config.lda < config.k))
                || (ta && (config.lda < config.m)))) {
            return false;
        }
        if (col_stored
            && ((nota && (config.lda < config.m))
                || (ta && (config.lda < config.k)))) {
            return false;
        }
    } else {
        return false;
    }

    // Leading dimension checks for matrix B (info = 12 in macro)
    // Skip leading dimension checks for reordered matrices as they have custom
    // layouts
    if (!config.reorderB) {
        if (row_stored
            && ((notb && (config.ldb < config.n))
                || (tb && (config.ldb < config.k)))) {
            return false;
        }
        if (col_stored
            && ((notb && (config.ldb < config.k))
                || (tb && (config.ldb < config.n)))) {
            return false;
        }
    } else {
        if (row_stored) {
            if (notb && (config.ldb < config.n)) {
                return false;
            }
            if (tb && (config.ldb < config.k)) {
                return false;
            }
        } else {
            return false;
        }
    }

    // Leading dimension checks for matrix C (info = 16 in macro)
    // Matrix C is never reordered, so always check
    if (row_stored && (config.ldc < config.n)) {
        return false;
    }
    if (col_stored && (config.ldc < config.m)) {
        return false;
    }

    return true;
}

// ============================================================================
// HELPER FUNCTIONS - PRINTING AND OUTPUT
// ============================================================================

// Helper function to print detailed configuration parameters
std::string
printConfigDetails(const GemmTestConfig& config)
{
    std::ostringstream details;
    details << "\n=== GEMM Test Configuration Details ===\n";
    details << "Test Name: " << config.name << "\n";
    details << "Matrix Dimensions: M=" << config.m << ", N=" << config.n
            << ", K=" << config.k << "\n";
    details << "Data Types: A=" << config.a_type << ", B=" << config.b_type
            << ", C=" << config.c_type << ", ACC=" << config.acc_type << "\n";
    details << "Storage Format: " << config.storage_format << "\n";
    details << "Transposition: transA=" << (config.transA ? "true" : "false")
            << ", transB=" << (config.transB ? "true" : "false") << "\n";
    details << "Leading Dimensions: lda=" << config.lda
            << ", ldb=" << config.ldb << ", ldc=" << config.ldc << "\n";
    details << "Alpha/Beta: alpha=" << config.alpha << ", beta=" << config.beta
            << "\n";
    details << "Reordering: reorderA=" << (config.reorderA ? "true" : "false")
            << ", reorderB=" << (config.reorderB ? "true" : "false") << "\n";
    details << "Packing: packA=" << (config.packA ? "true" : "false")
            << ", packB=" << (config.packB ? "true" : "false") << "\n";
    details << "PostOps: has_postops="
            << (config.has_postops ? "true" : "false");
    if (config.has_postops) {
        details << ", DLP PostOps=" << (config.postops_dlp ? "present" : "null")
                << ", REF PostOps="
                << (config.postops_ref ? "present" : "null");
    }
    details << "\n";

    // Calculate effective matrix dimensions after transposition
    md_t a_rows = config.transA ? config.k : config.m;
    md_t a_cols = config.transA ? config.m : config.k;
    md_t b_rows = config.transB ? config.n : config.k;
    md_t b_cols = config.transB ? config.k : config.n;

    details << "Effective Matrix Sizes:\n";
    details << "  Matrix A: " << a_rows << "x" << a_cols
            << (config.transA ? " (transposed)" : " (not transposed)") << "\n";
    details << "  Matrix B: " << b_rows << "x" << b_cols
            << (config.transB ? " (transposed)" : " (not transposed)") << "\n";
    details << "  Matrix C: " << config.m << "x" << config.n
            << " (never transposed)\n";
    details << "========================================\n";

    return details.str();
}

// Custom printer for better test output
void
PrintTo(const GemmTestConfig& config, std::ostream* os)
{
    *os << config.name << "_(" << config.m << "x" << config.n << "x"
        << config.k;
    if (config.transA || config.transB) {
        *os << "_trans" << (config.transA ? "A" : "")
            << (config.transB ? "B" : "");
    }
    if (config.storage_format == MatrixLayout::COLUMN_MAJOR) {
        *os << "_colMajor";
    } else {
        *os << "_rowMajor";
    }
    if (config.reorderA || config.reorderB) {
        *os << "_reorder" << (config.reorderA ? "A" : "")
            << (config.reorderB ? "B" : "");
    }
    if (config.has_postops) {
        *os << "_withPostOps";
    }
    *os << ")";
}

// Custom Google Test printer for Matrix class to show actual values instead of
// memory dumps
void
PrintTo(const Matrix& matrix, std::ostream* os)
{
    *os << "Matrix(" << matrix.getRows() << "x" << matrix.getCols()
        << ", type=" << matrix.getMatrixType()
        << ", layout=" << matrix.getLayout()
        << ", ld=" << matrix.getLeadingDimension();

    if (matrix.isTransposed())
        *os << ", transposed";
    if (matrix.isReordered())
        *os << ", reordered";
    if (matrix.isPacked())
        *os << ", packed";

    *os << ")";

    // Show actual matrix values if it's f32 and not too large
    if (matrix.getMatrixType() == MatrixType::f32 && matrix.getRows() <= 10
        && matrix.getCols() <= 10) {

        const float* data = reinterpret_cast<const float*>(matrix.getData());
        *os << "\nValues:\n";

        for (md_t i = 0; i < matrix.getRows(); ++i) {
            for (md_t j = 0; j < matrix.getCols(); ++j) {
                size_t index = i * matrix.getLeadingDimension() + j;
                *os << std::fixed << std::setprecision(6) << data[index]
                    << "\t";
            }
            *os << "\n";
        }
    } else {
        *os << " [values not shown - too large or not f32]";
    }
}

// ============================================================================
// HELPER FUNCTIONS - POSTOPS UTILITIES
// ============================================================================

// Helper function to extract operation names from PostOps
std::string
extractPostOpsDescription(const std::shared_ptr<IOperation>& postops)
{
    if (!postops) {
        return "";
    }

    std::ostringstream       postops_desc;
    std::vector<std::string> op_names;

    // Get the operation parameters
    const auto& params = postops->getParams();

    for (const auto& param : params) {
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
            default:
                op_names.push_back("UnknownOp");
                break;
        }
    }

    if (!op_names.empty()) {
        postops_desc << "_PostOps";
        for (size_t i = 0; i < op_names.size(); ++i) {
            postops_desc << "_" << op_names[i];
        }
    }

    return postops_desc.str();
}

// ============================================================================
// HELPER FUNCTIONS - TEST CONFIGURATION GENERATION
// ============================================================================

// Helper function to generate meaningful test names
std::string
generateTestName(const MicroTest& microTest,
                 size_t           testSetIndex,
                 size_t           configIndex)
{
    std::ostringstream name;

    // Start with base information
    name << "yaml_" << testSetIndex << "_M" << microTest.getM() << "_N"
         << microTest.getN() << "_K" << microTest.getK();

    // Add data type information if not default f32
    if (microTest.getAType() != MatrixType::f32
        || microTest.getBType() != MatrixType::f32
        || microTest.getCType() != MatrixType::f32) {
        name << "_" << microTest.getAType() << microTest.getBType()
             << microTest.getCType();
    }

    // Add storage format if not row-major (which is typically default)
    if (microTest.getStorageFormat() == MatrixLayout::COLUMN_MAJOR) {
        name << "_ColMajor";
    }

    // Add transposition information
    if (microTest.getTransA())
        name << "_transA";
    if (microTest.getTransB())
        name << "_transB";

    // Add reordering information
    if (microTest.getReorderA())
        name << "_reorderA";
    if (microTest.getReorderB())
        name << "_reorderB";

    // Add packing information
    if (microTest.getPackA())
        name << "_packA";
    if (microTest.getPackB())
        name << "_packB";

    // Add alpha/beta information if non-standard values
    if (microTest.getAlpha() != 1.0) {
        // Convert decimal to GoogleTest-friendly format (replace . with p)
        std::ostringstream alpha_stream;
        alpha_stream << std::fixed << std::setprecision(1)
                     << microTest.getAlpha();
        std::string alpha_str = alpha_stream.str();
        std::replace(alpha_str.begin(), alpha_str.end(), '.', 'p');
        std::replace(alpha_str.begin(), alpha_str.end(), '-',
                     'n'); // negative sign
        name << "_alpha" << alpha_str;
    }
    if (microTest.getBeta() != 0.0) {
        // Convert decimal to GoogleTest-friendly format (replace . with p)
        std::ostringstream beta_stream;
        beta_stream << std::fixed << std::setprecision(1)
                    << microTest.getBeta();
        std::string beta_str = beta_stream.str();
        std::replace(beta_str.begin(), beta_str.end(), '.', 'p');
        std::replace(beta_str.begin(), beta_str.end(), '-',
                     'n'); // negative sign
        name << "_beta" << beta_str;
    }

    auto postops_dlp = microTest.getPostOp(UALType::DLP);
    auto postops_ref = microTest.getPostOp(UALType::REF);

    std::string postops_desc;
    if (postops_dlp) {
        postops_desc = extractPostOpsDescription(postops_dlp);
    } else if (postops_ref) {
        postops_desc = extractPostOpsDescription(postops_ref);
    }

    if (!postops_desc.empty()) {
        name << postops_desc;
    }

    // Add config index for uniqueness
    name << "_" << configIndex;

    return name.str();
}

// ============================================================================
// HELPER FUNCTIONS - CONFIGURATION LOADING
// ============================================================================

// Function to load test configurations from YAML
std::vector<GemmTestConfig>
loadTestConfigurations(const std::string& yaml_file)
{
    std::vector<GemmTestConfig> configs;
    const size_t                MAX_CONFIGS_PER_TEST_SET =
        50000; // Reasonable limit to prevent millions of tests

    try {
        YamlParser parser(yaml_file, "gemm_tests");
        parser.setYieldType(YieldType::CARTESIAN_PRODUCT);

        size_t microTestCount = parser.getMicroTestCount();

        for (size_t i = 0; i < microTestCount; ++i) {
            // Get a mutable reference to the MicroTest for iteration
            // Note: This design could be improved by making MicroTest iteration
            // const-correct
            MicroTest& microTest =
                const_cast<MicroTest&>(parser.getMicroTest());
            size_t total_combinations = microTest.getSize();
            size_t test_count =
                std::min(total_combinations, MAX_CONFIGS_PER_TEST_SET);

            std::cout << "Test set " << i << ": Using " << test_count
                      << " out of " << total_combinations
                      << " total combinations" << std::endl;

            for (size_t j = 0; j < test_count; ++j) {
                // Generate meaningful test name
                std::string testName = generateTestName(microTest, i, j);

                // Extract PostOps from MicroTest
                auto postops_dlp = microTest.getPostOp(UALType::DLP);
                auto postops_ref = microTest.getPostOp(UALType::REF);

                // Extract configuration from parser
                configs.emplace_back(
                    testName,                     // name
                    microTest.getAType(),         // a_type
                    microTest.getBType(),         // b_type
                    microTest.getCType(),         // c_type
                    microTest.getAccType(),       // acc_type
                    microTest.getStorageFormat(), // storage_format
                    microTest.getM(),             // m
                    microTest.getN(),             // n
                    microTest.getK(),             // k
                    microTest.getLDA(),           // lda
                    microTest.getLDB(),           // ldb
                    microTest.getLDC(),           // ldc
                    microTest.getAlpha(),         // alpha
                    microTest.getBeta(),          // beta
                    microTest.getTransA(),        // transA
                    microTest.getTransB(),        // transB
                    microTest.getReorderA(),      // reorderA
                    microTest.getReorderB(),      // reorderB
                    microTest.getPackA(),         // packA
                    microTest.getPackB(),         // packB
                    postops_dlp,                  // postops_dlp
                    postops_ref                   // postops_ref
                );
                if (j < test_count - 1) {
                    microTest.next();
                }
            }
            // Move to next test configuration
            if (i < microTestCount - 1) {
                parser.next();
            }
        }

        std::cout << "Loaded " << configs.size() << " test configurations from "
                  << yaml_file << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error loading YAML configuration: " << e.what()
                  << std::endl;
        // Use Google Test's failure mechanism instead of silent failure
        ADD_FAILURE() << "Failed to load YAML configuration: " << e.what();
        return configs; // Return empty vector but with explicit test failure
    }

    return configs;
}

// Function to create additional test configurations programmatically
std::vector<GemmTestConfig>
createAdditionalTestConfigs()
{
    std::vector<GemmTestConfig> configs;

    // Add some programmatically generated test cases
    configs.emplace_back("small_square",          // name
                         MatrixType::f32,         // a_type
                         MatrixType::f32,         // b_type
                         MatrixType::f32,         // c_type
                         MatrixType::f32,         // acc_type
                         MatrixLayout::ROW_MAJOR, // storage_format
                         16, 16, 16,              // m, n, k
                         16, 16, 16,              // lda, ldb, ldc
                         1.0, 0.0,                // alpha, beta
                         false, false,            // transA, transB
                         false, false,            // reorderA, reorderB
                         false, false,            // packA, packB
                         nullptr, nullptr         // postops_dlp, postops_ref
    );

    configs.emplace_back("rectangular_1",         // name
                         MatrixType::f32,         // a_type
                         MatrixType::f32,         // b_type
                         MatrixType::f32,         // c_type
                         MatrixType::f32,         // acc_type
                         MatrixLayout::ROW_MAJOR, // storage_format
                         32, 16, 24,              // m, n, k
                         24, 16, 16,              // lda, ldb, ldc
                         1.5, 0.5,                // alpha, beta
                         false, false,            // transA, transB
                         false, false,            // reorderA, reorderB
                         false, false,            // packA, packB
                         nullptr, nullptr         // postops_dlp, postops_ref
    );

    configs.emplace_back("transposed_A",          // name
                         MatrixType::f32,         // a_type
                         MatrixType::f32,         // b_type
                         MatrixType::f32,         // c_type
                         MatrixType::f32,         // acc_type
                         MatrixLayout::ROW_MAJOR, // storage_format
                         20, 20, 20,              // m, n, k
                         20, 20, 20,              // lda, ldb, ldc
                         2.0, 1.0,                // alpha, beta
                         false, false,            // transA, transB
                         false, false,            // reorderA, reorderB
                         false, false,            // packA, packB
                         nullptr, nullptr         // postops_dlp, postops_ref
    );

    configs.emplace_back("edge_case_1x1",         // name
                         MatrixType::f32,         // a_type
                         MatrixType::f32,         // b_type
                         MatrixType::f32,         // c_type
                         MatrixType::f32,         // acc_type
                         MatrixLayout::ROW_MAJOR, // storage_format
                         1, 1, 1,                 // m, n, k
                         1, 1, 1,                 // lda, ldb, ldc
                         1.0, 0.0,                // alpha, beta
                         false, false,            // transA, transB
                         false, false,            // reorderA, reorderB
                         false, false,            // packA, packB
                         nullptr, nullptr         // postops_dlp, postops_ref
    );

    configs.emplace_back("edge_case_100x100_transposedB_reorderedB", // name
                         MatrixType::f32,                            // a_type
                         MatrixType::f32,                            // b_type
                         MatrixType::f32,                            // c_type
                         MatrixType::f32,                            // acc_type
                         MatrixLayout::ROW_MAJOR, // storage_format
                         100, 100, 100,           // m, n, k
                         100, 100, 100,           // lda, ldb, ldc
                         1.0, 0.0,                // alpha, beta
                         false, true,             // transA, transB
                         false, true,             // reorderA, reorderB
                         false, false,            // packA, packB
                         nullptr, nullptr         // postops_dlp, postops_ref
    );

    return configs;
}

// ============================================================================
// HELPER FUNCTIONS - DETAILED MATRIX COMPARISON
// ============================================================================

#if DLP_ENABLE_DETAILED_DEBUG || DLP_TESTING_ENABLE_DETAILED_DEBUG

// Helper function to compare matrices and report detailed differences
std::string
compareMatricesDetailed(const Matrix& matrix1,
                        const Matrix& matrix2,
                        size_t        maxMismatches = 10)
{
    std::ostringstream result;

    // Basic dimension check
    if (matrix1.getRows() != matrix2.getRows()
        || matrix1.getCols() != matrix2.getCols()) {
        result << "Matrix dimension mismatch: "
               << "Matrix1(" << matrix1.getRows() << "x" << matrix1.getCols()
               << ") vs "
               << "Matrix2(" << matrix2.getRows() << "x" << matrix2.getCols()
               << ")\n";
        return result.str();
    }

    if (matrix1.getMatrixType() != matrix2.getMatrixType()) {
        result << "Matrix type mismatch: "
               << "Matrix1(" << matrix1.getMatrixType() << ") vs "
               << "Matrix2(" << matrix2.getMatrixType() << ")\n";
        return result.str();
    }

    size_t rows       = matrix1.getRows();
    size_t cols       = matrix1.getCols();
    size_t mismatches = 0;
    double maxDiff    = 0.0;
    size_t maxDiffRow = 0, maxDiffCol = 0;

    // Calculate tolerance based on k dimension (similar to
    // Matrix::compareFloatingPointData)
    double tolerance = 1e-10;

    result << "Using maximum tolerance for debugging: " << tolerance << "\n";
    result << "Matrix dimensions: " << rows << "x" << cols << "\n";

    if (matrix1.getMatrixType() == MatrixType::f32) {
        const float* data1 = reinterpret_cast<const float*>(matrix1.getData());
        const float* data2 = reinterpret_cast<const float*>(matrix2.getData());
        size_t       elementCount = matrix1.getDataSizeBytes() / sizeof(float);

        for (size_t i = 0; i < elementCount; ++i) {
            float  val1 = data1[i];
            float  val2 = data2[i];
            double diff = std::abs(val1 - val2);

            if (diff > tolerance) {
                if (mismatches < maxMismatches) {
                    // Calculate row and column for reporting (based on leading
                    // dimension)
                    size_t r = i / matrix1.getLeadingDimension();
                    size_t c = i % matrix1.getLeadingDimension();
                    result << "Mismatch at [" << r << "," << c << "]: "
                           << "DLP=" << std::scientific << std::setprecision(10)
                           << val1 << ", REF=" << val2 << ", diff=" << diff
                           << "\n";
                }
                mismatches++;

                if (diff > maxDiff) {
                    maxDiff    = diff;
                    maxDiffRow = i / matrix1.getLeadingDimension();
                    maxDiffCol = i % matrix1.getLeadingDimension();
                }
            }
        }
    } else if (matrix1.getMatrixType() == MatrixType::bf16) {
        const bfloat16* data1 =
            reinterpret_cast<const bfloat16*>(matrix1.getData());
        const bfloat16* data2 =
            reinterpret_cast<const bfloat16*>(matrix2.getData());
        size_t elementCount = matrix1.getDataSizeBytes() / sizeof(bfloat16);

        for (size_t i = 0; i < elementCount; ++i) {
            bfloat16 bf16_val1 = data1[i];
            bfloat16 bf16_val2 = data2[i];

            // Convert to float for comparison
            float  val1 = bf16_to_f32(bf16_val1);
            float  val2 = bf16_to_f32(bf16_val2);
            double diff = std::abs(val1 - val2);

            if (diff > tolerance) {
                if (mismatches < maxMismatches) {
                    // Calculate row and column for reporting (based on leading
                    // dimension)
                    size_t r = i / matrix1.getLeadingDimension();
                    size_t c = i % matrix1.getLeadingDimension();
                    result << "Mismatch at [" << r << "," << c << "]: "
                           << "DLP=0x" << std::hex << std::setw(4)
                           << std::setfill('0')
                           << static_cast<uint16_t>(bf16_val1) << "("
                           << std::scientific << std::setprecision(8) << val1
                           << "), "
                           << "REF=0x" << std::hex << std::setw(4)
                           << std::setfill('0')
                           << static_cast<uint16_t>(bf16_val2) << "(" << val2
                           << "), " << std::dec << "diff=" << diff << "\n";
                }
                mismatches++;

                if (diff > maxDiff) {
                    maxDiff    = diff;
                    maxDiffRow = i / matrix1.getLeadingDimension();
                    maxDiffCol = i % matrix1.getLeadingDimension();
                }
            }
        }
    } else if (matrix1.getMatrixType() == MatrixType::u8) {
        const uint8_t* data1 =
            reinterpret_cast<const uint8_t*>(matrix1.getData());
        const uint8_t* data2 =
            reinterpret_cast<const uint8_t*>(matrix2.getData());
        size_t elementCount = matrix1.getDataSizeBytes() / sizeof(uint8_t);

        for (size_t i = 0; i < elementCount; ++i) {
            uint8_t u8_val1 = data1[i];
            uint8_t u8_val2 = data2[i];
            double  diff    = std::abs(static_cast<double>(u8_val1)
                                       - static_cast<double>(u8_val2));

            if (diff > tolerance) {
                if (mismatches < maxMismatches) {
                    // Calculate row and column for reporting (based on leading
                    // dimension)
                    size_t r = i / matrix1.getLeadingDimension();
                    size_t c = i % matrix1.getLeadingDimension();
                    result << "Mismatch at [" << r << "," << c << "]: "
                           << "DLP=0x" << std::hex << std::setw(2)
                           << std::setfill('0')
                           << static_cast<uint16_t>(u8_val1) << "(" << std::dec
                           << static_cast<int>(u8_val1) << "), "
                           << "REF=0x" << std::hex << std::setw(2)
                           << std::setfill('0')
                           << static_cast<uint16_t>(u8_val2) << "(" << std::dec
                           << static_cast<int>(u8_val2) << "), "
                           << "diff=" << diff << "\n";
                }
                mismatches++;

                if (diff > maxDiff) {
                    maxDiff    = diff;
                    maxDiffRow = i / matrix1.getLeadingDimension();
                    maxDiffCol = i % matrix1.getLeadingDimension();
                }
            }
        }
    } else {
        result << "Unsupported matrix type for detailed comparison: "
               << matrix1.getMatrixType() << "\n";
        return result.str();
    }

    result << "\nTotal mismatches: " << mismatches << "\n";
    if (mismatches > maxMismatches) {
        result << "... (showing first " << maxMismatches << " mismatches)\n";
    }

    if (maxDiff > 0.0) {
        result << "Maximum difference: " << std::scientific
               << std::setprecision(8) << maxDiff << " at position ["
               << maxDiffRow << "," << maxDiffCol << "]\n";
    }

    return result.str();
}
#endif

// ============================================================================
// GLOBAL INITIALIZATION
// ============================================================================

// Static initialization of test configurations
// This runs before main() and loads all configurations for parameterized tests
static std::vector<GemmTestConfig>
initializeTestConfigurations()
{
    std::vector<GemmTestConfig> all_configs;

    // Load YAML configurations
    auto yaml_configs = loadTestConfigurations(g_yaml_config_file);

    // Load programmatic configurations
    auto prog_configs = createAdditionalTestConfigs();

    // Combine all configurations
    all_configs.reserve(yaml_configs.size() + prog_configs.size());
    all_configs.insert(all_configs.end(), yaml_configs.begin(),
                       yaml_configs.end());
    all_configs.insert(all_configs.end(), prog_configs.begin(),
                       prog_configs.end());

    std::cout << "Total test configurations initialized: " << all_configs.size()
              << std::endl;

    return all_configs;
}

// Function to get test configurations (initialized on first call)
static const std::vector<GemmTestConfig>&
getTestConfigurations()
{
    static std::vector<GemmTestConfig> all_test_configs =
        initializeTestConfigurations();
    return all_test_configs;
}

// ============================================================================
// TEST FIXTURE CLASSES
// ============================================================================

// PARAMETERIZED TEST FIXTURE
class GemmParameterizedTest : public ::testing::TestWithParam<GemmTestConfig>
{
  protected:
    void SetUp() override { config_ = GetParam(); }

    void TearDown() override
    {
        // Optional cleanup per test
    }

    // Helper method to run the actual GEMM test
    void RunGemmTest()
    {
        // Create test matrices
        MatrixLayout layout = config_.storage_format;

        // Determine effective dimensions considering transposition
        md_t a_rows = config_.transA ? config_.k : config_.m;
        md_t a_cols = config_.transA ? config_.m : config_.k;
        md_t b_rows = config_.transB ? config_.n : config_.k;
        md_t b_cols = config_.transB ? config_.k : config_.n;

        Matrix A(a_rows, a_cols, config_.a_type, layout, config_.lda,
                 config_.transA);
        Matrix B(b_rows, b_cols, config_.b_type, layout, config_.ldb,
                 config_.transB);
        Matrix C(config_.m, config_.n, config_.c_type, layout, config_.ldc,
                 false);
        Matrix A_ref(a_rows, a_cols, config_.a_type, layout, config_.lda,
                     config_.transA);
        Matrix B_ref(b_rows, b_cols, config_.b_type, layout, config_.ldb,
                     config_.transB);
        Matrix C_ref(config_.m, config_.n, config_.c_type, layout, config_.ldc,
                     false);

// Initialize matrices with deterministic random values
#if 1
        A.fillRandom(42 + config_.m); // Use configuration to vary seed
        B.fillRandom(43 + config_.n);
        C.fillRandom(44 + config_.k);
        // C.fillValue(234);
        A_ref = A;
        B_ref = B;
        C_ref = C;
        // Reset packing state for reference matrices since reference
        // implementation doesn't support packing optimizations
        A_ref.setPacked(false);
        B_ref.setPacked(false);
        C_ref.setPacked(false);
#else
        A.fillValue((int8_t)10);
        B.fillValue((int8_t)-5);
        C.fillValue((float)20);
        A_ref.fillValue((int8_t)10);
        B_ref.fillValue((int8_t)-5);
        C_ref.fillValue((float)20);
#endif

        // Make a copy of C for reference computation
        C_ref = C;
        // Reset packing state after all matrix copies
        C_ref.setPacked(false);

        // TODO: The current UAL interface doesn't support alpha/beta parameters
        // Future enhancement: Add alpha/beta support to the GEMM interface
        // For now, alpha/beta values are stored in config but not used in
        // computation

        // Perform GEMM with DLP implementation
        std::unique_ptr<IUal> ual_dlp = UalFactory::createUal(UALType::DLP);

        // Apply reordering if specified
        if (config_.reorderA) {
            A.setReordered(true);
            A_ref.setReordered(true);
        }
        bool     params_valid       = check_valid_params(config_);
        UALError dlp_reorder_status = UALError::UAL_SUCCESS;
        UALError ref_reorder_status = UALError::UAL_SUCCESS;
        if (config_.reorderB) {
            // Create reordered matrix with custom allocation size in bytes
            Matrix B_reordered;
            Matrix B_ref_reordered;

            dlp_reorder_status =
                ual_dlp->reorder(B, B_reordered, config_.a_type, config_.b_type,
                                 config_.c_type, config_.acc_type);

            // Skip test if DLP reorder is not supported
            if (dlp_reorder_status == UALError::UAL_NOT_SUPPORTED) {
                GTEST_SKIP()
                    << "DLP reorder not supported for this configuration";
            }

            if (dlp_reorder_status == UALError::UAL_SUCCESS) {
                B = B_reordered;
            }

            // Also apply reordering to reference matrix to ensure both have
            // same parameters
            std::unique_ptr<IUal> ual_ref_for_reorder =
                UalFactory::createUal(UALType::REF);

            // Check if the config is valid then reorder
            if (params_valid) {
                ref_reorder_status = ual_ref_for_reorder->reorder(
                    B_ref, B_ref_reordered, config_.a_type, config_.b_type,
                    config_.c_type, config_.acc_type);

                if (ref_reorder_status == UALError::UAL_SUCCESS) {
                    B_ref = B_ref_reordered;
                    // Reset packing state after reordering assignment
                    B_ref.setPacked(false);
                }
            } else {
                ref_reorder_status = UALError::UAL_FAILURE;
            }
        }

        // Apply packing if specified
        // Note: Pack parameters are DLP optimization and are handled
        // via mem_format parameters in the DLP GEMM call.
        // Reference implementation ignores pack parameters completely as
        // pack is an on-the-fly optimization that doesn't change mathematical
        // results.
        if (config_.packA) {
            A.setPacked(true);
            // Reference implementation: pack is no-op, A_ref remains unchanged
        }
        if (config_.packB) {
            B.setPacked(true);
            // Reference implementation: pack is no-op, B_ref remains unchanged
        }

        ASSERT_TRUE(ual_dlp != nullptr) << "Failed to create DLP UAL";

        UALError dlp_status =
            ual_dlp->gemm(A, B, C, config_.acc_type, config_.postops_dlp,
                          config_.alpha, config_.beta);

        if (dlp_status == UALError::UAL_NOT_SUPPORTED) {
            GTEST_SKIP() << "DLP GEMM not supported for this configuration";
        }

        bool dlp_result = (dlp_status == UALError::UAL_SUCCESS);

        // Perform GEMM with reference implementation
        std::unique_ptr<IUal> ual_ref = UalFactory::createUal(UALType::REF);
        ASSERT_TRUE(ual_ref != nullptr) << "Failed to create REF UAL";

        UALError ref_status =
            ual_ref->gemm(A_ref, B_ref, C_ref, config_.acc_type,
                          config_.postops_ref, config_.alpha, config_.beta);

        bool ref_result = (ref_status == UALError::UAL_SUCCESS);

        // Check if parameters are valid

        if (params_valid) {
            // For valid parameters, both implementations should succeed
            EXPECT_TRUE(dlp_result)
                << "DLP GEMM should succeed with valid parameters:"
                << printConfigDetails(config_);
            EXPECT_TRUE(ref_result)
                << "Reference GEMM should succeed with valid parameters:"
                << printConfigDetails(config_);

            // And produce the same results
            C.setK(config_.k);
            // Ensure both result matrices have the same packing state for
            // comparison Since packing is an optimization hint that shouldn't
            // affect results
            C_ref.setPacked(C.isPacked());
            // Detailed comparison with mismatch reporting
            if (!(C == C_ref)) {
                std::ostringstream detailed_error;
                detailed_error << "DLP and Reference results mismatch for "
                                  "valid parameters:\n"
                               << printConfigDetails(config_) << "\n";

#if DLP_ENABLE_DETAILED_DEBUG || DLP_TESTING_ENABLE_DETAILED_DEBUG
                detailed_error << compareMatricesDetailed(C, C_ref);
#endif
                FAIL() << detailed_error.str();
            }
        } else {
            // For invalid parameters, both implementations should fail
            // gracefully
            EXPECT_FALSE(dlp_result
                         && (dlp_reorder_status == UALError::UAL_SUCCESS))
                << "DLP GEMM should fail gracefully with invalid parameters:"
                << printConfigDetails(config_);
            EXPECT_FALSE(ref_result
                         && (ref_reorder_status == UALError::UAL_SUCCESS))
                << "Reference GEMM should fail gracefully "
                   "with invalid parameters:"
                << printConfigDetails(config_);

            // No need to compare results when both operations failed
            std::cout << "Test passed: Both implementations correctly rejected "
                         "invalid parameters"
                      << std::endl;
        }
    }

  protected:
    GemmTestConfig config_;
};

// ============================================================================
// PARAMETERIZED TESTS
// ============================================================================

// The actual parameterized test - each configuration becomes a separate test
// case
TEST_P(GemmParameterizedTest, CompareImplementations)
{
    RunGemmTest();
}

// Register all configurations as individual test cases
INSTANTIATE_TEST_SUITE_P(
    YamlConfigurations,
    GemmParameterizedTest,
    ::testing::ValuesIn(getTestConfigurations()),
    [](const ::testing::TestParamInfo<GemmTestConfig>& info) {
        return info.param.name;
    });

// ============================================================================
// BASIC TESTS
// ============================================================================

// Original basic test - kept as an example
TEST(GEMMTest, Basic)
{
    // Test setup
    md_t m = 2;
    md_t n = 3;
    md_t k = 4;

    // Create matrices
    Matrix A(m, k, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix B(k, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix C(m, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix C_ref(m, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);

    // Initialize matrices with random values
    A.fillRandom(42); // Using fixed seed for reproducibility
    B.fillRandom(43);
    C.fillRandom(44);
    // Make a copy of C for reference
    C_ref = C;

    // Perform GEMM
    std::unique_ptr<IUal> ual = UalFactory::createUal(UALType::DLP);
    UALError dlp_status       = ual->gemm(A, B, C, MatrixType::f32, nullptr);
    EXPECT_EQ(dlp_status, UALError::UAL_SUCCESS);

    // Perform GEMM with reference implementation
    ual                 = UalFactory::createUal(UALType::REF);
    UALError ref_status = ual->gemm(A, B, C_ref, MatrixType::f32, nullptr);
    EXPECT_EQ(ref_status, UALError::UAL_SUCCESS);

    // To compare with reference, we need to set the k dimension for tolerance
    C.setK(k);

    EXPECT_EQ(C, C_ref);
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

// Separate test for matrix creation edge cases
TEST(GEMMTest, EdgeCases)
{
    // Test 1x1 matrices - use explicit constructor with layout parameter to
    // avoid ambiguity
    Matrix A_small(1u, 1u, MatrixType::f32, MatrixLayout::ROW_MAJOR);
    Matrix B_small(1u, 1u, MatrixType::f32, MatrixLayout::ROW_MAJOR);
    Matrix C_small(1u, 1u, MatrixType::f32, MatrixLayout::ROW_MAJOR);
    Matrix C_small_ref(1u, 1u, MatrixType::f32, MatrixLayout::ROW_MAJOR);

    A_small.fillRandom(1);
    B_small.fillRandom(2);
    C_small.fillRandom(3);
    C_small_ref = C_small;

    std::unique_ptr<IUal> ual = UalFactory::createUal(UALType::DLP);
    UALError              dlp_status =
        ual->gemm(A_small, B_small, C_small, MatrixType::f32, nullptr);
    ASSERT_EQ(dlp_status, UALError::UAL_SUCCESS);

    ual = UalFactory::createUal(UALType::REF);
    UALError ref_status =
        ual->gemm(A_small, B_small, C_small_ref, MatrixType::f32, nullptr);
    ASSERT_EQ(ref_status, UALError::UAL_SUCCESS);

    // To compare with reference, we need to set the k dimension for tolerance
    C_small.setK(1);

    EXPECT_EQ(C_small, C_small_ref);
}

// ============================================================================
// DEBUG TESTS
// ============================================================================

// Test for debugging double-free issue
TEST(GEMMTest, DebugDoubleFree)
{
    // Recreate the problematic sequence
    MatrixLayout layout = MatrixLayout::ROW_MAJOR;

    // Create matrices with invalid parameters (similar to the failing test)
    Matrix B(320, 512, MatrixType::f32, layout, 256,
             true); // transB=true, invalid ldb

    // Create default-constructed matrix for reordering
    Matrix B_reordered;

    // Try reordering
    std::unique_ptr<IUal> ual_dlp = UalFactory::createUal(UALType::DLP);
    UALError              reorder_result =
        ual_dlp->reorder(B, B_reordered, MatrixType::f32, MatrixType::f32,
                         MatrixType::f32, MatrixType::f32);

    // Reordering should fail and ldb is lower than min needed ldb.
    EXPECT_TRUE(reorder_result == UALError::UAL_FAILURE
                || reorder_result == UALError::UAL_NOT_SUPPORTED)
        << "Reordering should fail with invalid parameters";
}

// ============================================================================
// POSTOPS TESTS
// ============================================================================

// Test for PostOps integration
TEST(GEMMTest, PostOpsIntegration)
{
    // Test setup
    md_t m = 4;
    md_t n = 4;
    md_t k = 4;

    // Create matrices
    Matrix A(m, k, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix B(k, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix C(m, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix C_ref(m, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);

    // Initialize matrices with simple values for easier verification
    A.fillValue(0.5f);
    B.fillValue(0.2f);
    C.fillValue(0.0f);
    C_ref = C;

    // Create PostOps using the builder pattern
    auto operation_dlp = OperationFactory::createOperation(UALType::DLP);

    // Add a ReLU operation
    auto relu_dlp = createRelu().build();
    operation_dlp->addOperation(std::move(relu_dlp));

    // Add a bias operation
    auto bias_dlp =
        Matrix::fromVector(std::vector<float>{ 1.0f, 2.0f, 3.0f, 4.0f });
    auto biasOp_dlp = createBias().setBias(bias_dlp).build();
    operation_dlp->addOperation(std::move(biasOp_dlp));

    // Finalize the operations
    operation_dlp->finalize();

    // Perform GEMM with PostOps using DLP
    std::unique_ptr<IUal> ual_dlp = UalFactory::createUal(UALType::DLP);
    UALError              dlp_status =
        ual_dlp->gemm(A, B, C, MatrixType::f32, operation_dlp);

    // For now, just verify that the operation doesn't crash
    // In a real implementation, we would verify the actual results
    EXPECT_EQ(dlp_status, UALError::UAL_SUCCESS)
        << "DLP GEMM with PostOps should succeed";

    auto operation_ref = OperationFactory::createOperation(UALType::REF);

    // Add a ReLU operation
    auto relu_ref = createRelu().build();
    operation_ref->addOperation(std::move(relu_ref));

    // Add a bias operation
    auto bias_ref =
        Matrix::fromVector(std::vector<float>{ 1.0f, 2.0f, 3.0f, 4.0f });
    auto biasOp_ref = createBias().setBias(bias_ref).build();
    operation_ref->addOperation(std::move(biasOp_ref));

    // Finalize the operations
    operation_ref->finalize();

    // Test with reference implementation (should ignore PostOps)
    std::unique_ptr<IUal> ual_ref = UalFactory::createUal(UALType::REF);
    UALError              ref_status =
        ual_ref->gemm(A, B, C_ref, MatrixType::f32, operation_ref);

    EXPECT_EQ(ref_status, UALError::UAL_SUCCESS)
        << "Reference GEMM with PostOps should succeed";

    // Compare the results
    C_ref.setK(k);
    EXPECT_EQ(C_ref, C);

    // Test with nullptr PostOps (should behave like original gemm)
    Matrix C_no_postops(m, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0,
                        false);
    C_no_postops.fillValue(0.0f);

    UALError no_postops_status =
        ual_dlp->gemm(A, B, C_no_postops, MatrixType::f32, nullptr);
    EXPECT_EQ(no_postops_status, UALError::UAL_SUCCESS)
        << "DLP GEMM with nullptr PostOps should succeed";
}

// Test for PostOps Builder Pattern
TEST(GEMMTest, PostOpsBuilderPattern)
{
    // Test the builder pattern for different operations

    // Test ReLU builder
    auto relu = createRelu().build();
    EXPECT_EQ(relu->getType(), OperationType::ElementWise);

    // Test Bias builder
    auto bias   = Matrix::fromVector(std::vector<float>{ 1.0f, 2.0f, 3.0f });
    auto biasOp = createBias().setBias(bias).build();
    EXPECT_EQ(biasOp->getType(), OperationType::Bias);

    // Test Scale builder
    auto scale   = Matrix::scalar(2.0f);
    auto scaleOp = createScale().setScaleFactor(scale).build();
    EXPECT_EQ(scaleOp->getType(), OperationType::Scale);

    // Test Matrix Add builder
    auto matrix = Matrix::fromData(
        std::vector<std::vector<float>>{ { 1.0f, 2.0f }, { 3.0f, 4.0f } });
    auto matAddOp = createMatrixAdd().setMatrix(matrix).build();
    EXPECT_EQ(matAddOp->getType(), OperationType::MatAdd);

    // Test operation creation for different UAL types
    auto dlp_operation = OperationFactory::createOperation(UALType::DLP);
    EXPECT_EQ(dlp_operation->getUALType(), UALType::DLP);

    auto ref_operation = OperationFactory::createOperation(UALType::REF);
    EXPECT_EQ(ref_operation->getUALType(), UALType::REF);
}

// Test for multiple PostOps of the same type
TEST(GEMMTest, MultiplePostOpsOfSameType)
{
    // Test setup
    md_t m = 4;
    md_t n = 4;
    md_t k = 4;

    // Create matrices
    Matrix A(m, k, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix B(k, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix C(m, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);

    // Initialize matrices with simple values
    A.fillValue(0.5f);
    B.fillValue(0.2f);
    C.fillValue(0.0f);

    // Create PostOps with multiple operations of the same type
    auto operation = OperationFactory::createOperation(UALType::DLP);

    // Add multiple RELU operations
    auto relu1 = createRelu().build();
    operation->addOperation(std::move(relu1));

    auto relu2 = createRelu().build();
    operation->addOperation(std::move(relu2));

    // Add multiple BIAS operations
    auto bias1 =
        Matrix::fromVector(std::vector<float>{ 1.0f, 2.0f, 3.0f, 4.0f });
    auto biasOp1 = createBias().setBias(bias1).build();
    operation->addOperation(std::move(biasOp1));

    auto bias2 =
        Matrix::fromVector(std::vector<float>{ 0.5f, 1.0f, 1.5f, 2.0f });
    auto biasOp2 = createBias().setBias(bias2).build();
    operation->addOperation(std::move(biasOp2));

    // Add multiple SCALE operations
    auto scale1   = Matrix::fromVector(std::vector<float>{ 1.5f });
    auto scaleOp1 = createScale().setScaleFactor(scale1).build();
    operation->addOperation(std::move(scaleOp1));

    auto scale2   = Matrix::fromVector(std::vector<float>{ 2.0f });
    auto scaleOp2 = createScale().setScaleFactor(scale2).build();
    operation->addOperation(std::move(scaleOp2));

    // Finalize the operations
    operation->finalize();

    // Perform GEMM with multiple PostOps using DLP
    std::unique_ptr<IUal> ual_dlp = UalFactory::createUal(UALType::DLP);
    UALError dlp_status = ual_dlp->gemm(A, B, C, MatrixType::f32, operation);

    // Verify that the operation doesn't crash and handles multiple ops
    // correctly
    EXPECT_EQ(dlp_status, UALError::UAL_SUCCESS)
        << "DLP GEMM with multiple PostOps of same type should succeed";

    std::cout << "Multiple PostOps of same type test completed successfully!"
              << std::endl;
    std::cout
        << "Operation sequence: RELU -> RELU -> BIAS -> BIAS -> SCALE -> SCALE"
        << std::endl;
}

// Simple test for debugging multiple operations
TEST(GEMMTest, DebugMultipleOps)
{
    // Test setup
    md_t m = 2;
    md_t n = 2;
    md_t k = 2;

    // Create matrices
    Matrix A(m, k, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix B(k, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix C(m, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);

    // Initialize matrices with simple values
    A.fillValue(1.0f);
    B.fillValue(1.0f);
    C.fillValue(0.0f);

    // Create PostOps with just two RELU operations
    auto operation = OperationFactory::createOperation(UALType::DLP);

    std::cout << "Adding first RELU operation..." << std::endl;
    auto relu1 = createRelu().build();
    operation->addOperation(std::move(relu1));

    std::cout << "Adding second RELU operation..." << std::endl;
    auto relu2 = createRelu().build();
    operation->addOperation(std::move(relu2));

    std::cout << "Finalizing operations..." << std::endl;
    operation->finalize();

    std::cout << "Creating UAL..." << std::endl;
    std::unique_ptr<IUal> ual_dlp = UalFactory::createUal(UALType::DLP);

    std::cout << "Calling GEMM..." << std::endl;
    UALError dlp_status = ual_dlp->gemm(A, B, C, MatrixType::f32, operation);

    std::cout << "GEMM result: "
              << (dlp_status == UALError::UAL_SUCCESS ? "SUCCESS" : "FAILED")
              << std::endl;

    EXPECT_EQ(dlp_status, UALError::UAL_SUCCESS)
        << "DLP GEMM with two RELU operations should succeed";
}

// Test to isolate the segmentation fault issue
TEST(GEMMTest, IsolateSegFault)
{
    // Test setup
    md_t m = 2;
    md_t n = 2;
    md_t k = 2;

    // Create matrices
    Matrix A(m, k, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix B(k, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix C(m, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);

    // Initialize matrices
    A.fillValue(1.0f);
    B.fillValue(1.0f);
    C.fillValue(0.0f);

    std::cout << "Testing BIAS operations..." << std::endl;

    // Test 1: Just BIAS operations
    {
        auto operation = OperationFactory::createOperation(UALType::DLP);

        try {
            std::cout << "Creating bias matrix..." << std::endl;
            auto bias1 = Matrix::fromVector(std::vector<float>{ 1.0f, 2.0f });
            std::cout << "Creating bias operation..." << std::endl;
            auto biasOp1 = createBias().setBias(bias1).build();
            std::cout << "Adding bias operation..." << std::endl;
            operation->addOperation(std::move(biasOp1));

            std::cout << "Finalizing..." << std::endl;
            operation->finalize();

            std::cout << "Creating UAL..." << std::endl;
            std::unique_ptr<IUal> ual_dlp = UalFactory::createUal(UALType::DLP);

            std::cout << "Calling GEMM with BIAS..." << std::endl;
            UALError status =
                ual_dlp->gemm(A, B, C, MatrixType::f32, operation);

            std::cout << "BIAS test result: "
                      << (status == UALError::UAL_SUCCESS ? "SUCCESS"
                                                          : "FAILED")
                      << std::endl;
            EXPECT_EQ(status, UALError::UAL_SUCCESS);
        } catch (const std::exception& e) {
            std::cout << "Exception in BIAS test: " << e.what() << std::endl;
            FAIL() << "Exception in BIAS test: " << e.what();
        }
    }

    std::cout << "Testing SCALE operations..." << std::endl;

    // Test 2: SCALE operations with scale factor
    {
        auto operation = OperationFactory::createOperation(UALType::DLP);

        try {
            std::cout << "Creating scale operation with scale factor..."
                      << std::endl;
            auto scale = Matrix::fromVector(std::vector<float>{ 1.5f });
            auto sc1   = createScale().setScaleFactor(scale).build();
            std::cout << "Adding scale operation..." << std::endl;
            operation->addOperation(std::move(sc1));

            std::cout << "Finalizing..." << std::endl;
            operation->finalize();

            std::cout << "Creating UAL..." << std::endl;
            std::unique_ptr<IUal> ual_dlp = UalFactory::createUal(UALType::DLP);

            std::cout << "Calling GEMM with SCALE..." << std::endl;
            UALError status =
                ual_dlp->gemm(A, B, C, MatrixType::f32, operation);

            std::cout << "Scale test result: "
                      << (status == UALError::UAL_SUCCESS ? "SUCCESS"
                                                          : "FAILED")
                      << std::endl;
            EXPECT_EQ(status, UALError::UAL_SUCCESS);
        } catch (const std::exception& e) {
            std::cout << "Exception in Scale test: " << e.what() << std::endl;
            FAIL() << "Exception in Scale test: " << e.what();
        }
    }
}

// Comprehensive PostOps comparison test
TEST(GEMMTest, PostOpsComprehensiveComparison)
{
    // Test setup with consistent initialization
    md_t m = 4;
    md_t n = 4;
    md_t k = 4;

    // Create matrices for DLP
    Matrix A_dlp(m, k, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix B_dlp(k, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix C_dlp(m, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);

    // Create matrices for REF (identical to DLP)
    Matrix A_ref(m, k, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix B_ref(k, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix C_ref(m, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);

    // Initialize matrices with consistent values
    A_dlp.fillValue(0.5f);
    B_dlp.fillValue(0.2f);
    C_dlp.fillValue(0.0f);

    A_ref.fillValue(0.5f);
    B_ref.fillValue(0.2f);
    C_ref.fillValue(0.0f);

    // Create PostOps for DLP
    auto operation_dlp = OperationFactory::createOperation(UALType::DLP);

    // Add ReLU operation to DLP
    auto relu_dlp = createRelu().build();
    operation_dlp->addOperation(std::move(relu_dlp));

    // Add bias operation to DLP
    auto bias_dlp =
        Matrix::fromVector(std::vector<float>{ 1.0f, 2.0f, 3.0f, 4.0f });
    auto biasOp_dlp = createBias().setBias(bias_dlp).build();
    operation_dlp->addOperation(std::move(biasOp_dlp));

    operation_dlp->finalize();

    // Create PostOps for REF (identical operations)
    auto operation_ref = OperationFactory::createOperation(UALType::REF);

    // Add ReLU operation to REF
    auto relu_ref = createRelu().build();
    operation_ref->addOperation(std::move(relu_ref));

    // Add bias operation to REF
    auto bias_ref =
        Matrix::fromVector(std::vector<float>{ 1.0f, 2.0f, 3.0f, 4.0f });
    auto biasOp_ref = createBias().setBias(bias_ref).build();
    operation_ref->addOperation(std::move(biasOp_ref));

    operation_ref->finalize();

    // Test DLP implementation
    std::unique_ptr<IUal> ual_dlp = UalFactory::createUal(UALType::DLP);
    ASSERT_TRUE(ual_dlp != nullptr) << "Failed to create DLP UAL";

    UALError dlp_status =
        ual_dlp->gemm(A_dlp, B_dlp, C_dlp, MatrixType::f32, operation_dlp);
    EXPECT_EQ(dlp_status, UALError::UAL_SUCCESS)
        << "DLP GEMM with PostOps should succeed";

    // Test REF implementation
    std::unique_ptr<IUal> ual_ref = UalFactory::createUal(UALType::REF);
    ASSERT_TRUE(ual_ref != nullptr) << "Failed to create REF UAL";

    UALError ref_status =
        ual_ref->gemm(A_ref, B_ref, C_ref, MatrixType::f32, operation_ref);
    EXPECT_EQ(ref_status, UALError::UAL_SUCCESS)
        << "REF GEMM with PostOps should succeed";

    // Compare results (both should have applied the same PostOps)
    C_dlp.setK(k);
    C_ref.setK(k);
    EXPECT_EQ(C_dlp, C_ref)
        << "DLP and REF should produce identical results with same PostOps";

    std::cout << "Comprehensive PostOps comparison test completed!"
              << std::endl;
}

// Test for UAL type validation
TEST(GEMMTest, PostOpsUALTypeValidation)
{
    // Test setup
    md_t m = 2;
    md_t n = 2;
    md_t k = 2;

    Matrix A(m, k, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix B(k, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix C(m, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);

    A.fillValue(1.0f);
    B.fillValue(1.0f);
    C.fillValue(0.0f);

    // Create DLP PostOps
    auto dlp_operation = OperationFactory::createOperation(UALType::DLP);
    auto relu_dlp      = createRelu().build();
    dlp_operation->addOperation(std::move(relu_dlp));
    dlp_operation->finalize();

    // Create REF PostOps
    auto ref_operation = OperationFactory::createOperation(UALType::REF);
    auto relu_ref      = createRelu().build();
    ref_operation->addOperation(std::move(relu_ref));
    ref_operation->finalize();

    // Test: DLP UAL with DLP PostOps (should succeed)
    std::unique_ptr<IUal> ual_dlp = UalFactory::createUal(UALType::DLP);
    UALError              dlp_with_dlp_status =
        ual_dlp->gemm(A, B, C, MatrixType::f32, dlp_operation);
    EXPECT_EQ(dlp_with_dlp_status, UALError::UAL_SUCCESS)
        << "DLP UAL with DLP PostOps should succeed";

    // Test: REF UAL with REF PostOps (should succeed)
    std::unique_ptr<IUal> ual_ref = UalFactory::createUal(UALType::REF);
    UALError              ref_with_ref_status =
        ual_ref->gemm(A, B, C, MatrixType::f32, ref_operation);
    EXPECT_EQ(ref_with_ref_status, UALError::UAL_SUCCESS)
        << "REF UAL with REF PostOps should succeed";

    // Test: DLP UAL with REF PostOps (should fail due to type mismatch)
    Matrix C_mismatch1(m, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0,
                       false);
    C_mismatch1.fillValue(0.0f);
    UALError dlp_with_ref_status =
        ual_dlp->gemm(A, B, C_mismatch1, MatrixType::f32, ref_operation);
    EXPECT_NE(dlp_with_ref_status, UALError::UAL_SUCCESS)
        << "DLP UAL with REF PostOps should fail due to type mismatch";

    // Test: REF UAL with DLP PostOps (should fail due to type mismatch)
    Matrix C_mismatch2(m, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0,
                       false);
    C_mismatch2.fillValue(0.0f);
    UALError ref_with_dlp_status =
        ual_ref->gemm(A, B, C_mismatch2, MatrixType::f32, dlp_operation);
    EXPECT_NE(ref_with_dlp_status, UALError::UAL_SUCCESS)
        << "REF UAL with DLP PostOps should fail due to type mismatch";

    std::cout << "UAL type validation test completed!" << std::endl;
}

// ============================================================================
// MAIN FUNCTION WITH ARGUMENT PARSING
// ============================================================================

int
main(int argc, char** argv)
{
    // Parse custom arguments before GoogleTest processes them
    auto parser = dlp::testing::utils::ArgParser::parseTestArgs(argc, argv);

    // Handle help request - show our custom help first, then let GoogleTest
    // show its help
    if (parser.helpRequested()) {
        parser.printUsage(argv[0]);
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "GoogleTest Help:" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        // Don't return here - let GoogleTest handle --help naturally below
    }

    // Update global YAML configuration file path if specified
    std::string yaml_file = parser.getYamlFile();
    if (!yaml_file.empty()) {
        g_yaml_config_file = yaml_file;
        std::cout << "Using YAML configuration file: " << g_yaml_config_file
                  << std::endl;
    } else {
        std::cout << "Using default YAML configuration file: "
                  << g_yaml_config_file << std::endl;
    }

    // Check if specified file exists (if custom file was provided)
    if (parser.getYamlFile().empty() == false
        && !std::filesystem::exists(g_yaml_config_file)) {
        std::cerr << "Error: YAML configuration file '" << g_yaml_config_file
                  << "' does not exist!" << std::endl;
        std::cerr << "Please check the file path or run with -h for usage "
                     "information."
                  << std::endl;
        return 1;
    }

    // Initialize GoogleTest with remaining arguments
    ::testing::InitGoogleTest(&argc, argv);

    // If help was requested, GoogleTest has already shown its help, so we can
    // exit
    if (parser.helpRequested()) {
        return 0;
    }

    // Run all tests
    return RUN_ALL_TESTS();
}
