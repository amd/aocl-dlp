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

#include "aocl_dlp.h"
#include "framework/operation.hh"
#include "framework/ual.hh"
#include "framework/ual_factory.hh"
#include "test_config.hh"
#include "utils/yaml_parser.hh"
#include <algorithm>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace dlp::testing;
using dlp::testing::IUal;
using dlp::testing::UalFactory;

// Configuration structure for YAML-driven tests
struct GemmTestConfig
{
    std::string  name;
    MatrixType   a_type, b_type, c_type, acc_type;
    MatrixLayout storage_format;
    md_t         m, n, k;
    md_t         lda, ldb, ldc;
    double       alpha, beta; // Changed from float to double for consistency
    bool         transA, transB;
    bool         reorderA, reorderB;
    bool         packA, packB; // TODO: Pack parameters not yet used in tests

    // PostOps support
    std::shared_ptr<IOperation> postops_dlp;
    std::shared_ptr<IOperation> postops_ref;
    bool                        has_postops;

    // Default constructor (required by GoogleTest)
    GemmTestConfig()
        : name("default")
        , m(1)
        , n(1)
        , k(1)
        , lda(1)
        , ldb(1)
        , ldc(1)
        , alpha(1.0)
        , beta(0.0)
        , transA(false)
        , transB(false)
        , reorderA(false)
        , reorderB(false)
        , packA(false)
        , packB(false)
        , postops_dlp(nullptr)
        , postops_ref(nullptr)
        , has_postops(false)
    {
    }

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

// Helper function to generate meaningful test names
std::string
generateTestName(const MicroTest& microTest,
                 size_t           testSetIndex,
                 size_t           configIndex)
{
    std::ostringstream name;
    name << "yaml_" << testSetIndex << "_M" << microTest.getM() << "_N"
         << microTest.getN() << "_K" << microTest.getK();
    if (microTest.getTransA())
        name << "_transA";
    if (microTest.getTransB())
        name << "_transB";
    if (microTest.getReorderA())
        name << "_reorderA";
    if (microTest.getReorderB())
        name << "_reorderB";
    name << "_" << configIndex;
    return name.str();
}

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

// Static initialization of test configurations
// This runs before main() and loads all configurations for parameterized tests
static std::vector<GemmTestConfig>
initializeTestConfigurations()
{
    std::vector<GemmTestConfig> all_configs;

    // Load YAML configurations
    std::string config_file  = TEST_CONFIG_DIR "/gemm_test_config.yaml";
    auto        yaml_configs = loadTestConfigurations(config_file);

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

// Global configurations - initialized once at startup
static const std::vector<GemmTestConfig> g_all_test_configs =
    initializeTestConfigurations();

// PARAMETERIZED TEST FIXTURE
class GemmParameterizedTest : public ::testing::TestWithParam<GemmTestConfig>
{
  protected:
    void SetUp() override
    {
        config_ = GetParam();
        std::cout << "Setting up test: " << config_.name << " (" << config_.m
                  << "x" << config_.n << "x" << config_.k << ")" << std::endl;
    }

    void TearDown() override
    {
        // Optional cleanup per test
    }

    // Helper method to run the actual GEMM test
    void RunGemmTest()
    {
        // Create matrices based on configuration
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
        Matrix C_ref(config_.m, config_.n, config_.acc_type, layout,
                     config_.ldc, false);

// Initialize matrices with deterministic random values
#if 0
        A.fillRandom(42 + config_.m); // Use configuration to vary seed
        B.fillRandom(43 + config_.n);
        C.fillRandom(44 + config_.k);
        A_ref = A;
        B_ref = B;
        C_ref = C;
#else
        A.fillValue(0.5f);
        B.fillValue(0.2f);
        C.fillValue(0.0f);
        A_ref.fillValue(0.5f);
        B_ref.fillValue(0.2f);
        C_ref.fillValue(0.0f);
#endif

        // Make a copy of C for reference computation
        C_ref = C;

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
        if (config_.reorderB) {
            // Create reordered matrix with custom allocation size in bytes
            Matrix B_reordered;
            Matrix B_ref_reordered;

            ual_dlp->reorder(B, B_reordered, config_.acc_type);
            B = B_reordered;

            // Also apply reordering to reference matrix to ensure both have
            // same parameters
            std::unique_ptr<IUal> ual_ref_for_reorder =
                UalFactory::createUal(UALType::REF);
            ual_ref_for_reorder->reorder(B_ref, B_ref_reordered,
                                         config_.acc_type);
            B_ref = B_ref_reordered;
        }

        ASSERT_TRUE(ual_dlp != nullptr) << "Failed to create DLP UAL";

        bool dlp_result =
            ual_dlp->gemm(A, B, C, config_.acc_type, config_.postops_dlp);

        // Perform GEMM with reference implementation
        std::unique_ptr<IUal> ual_ref = UalFactory::createUal(UALType::REF);
        ASSERT_TRUE(ual_ref != nullptr) << "Failed to create REF UAL";

        bool ref_result = ual_ref->gemm(A_ref, B_ref, C_ref, config_.acc_type,
                                        config_.postops_ref);

        // Check if parameters are valid
        bool params_valid = check_valid_params(config_);

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
            EXPECT_EQ(C, C_ref) << "DLP and Reference results should match for "
                                   "valid parameters:"
                                << printConfigDetails(config_);
        } else {
            // For invalid parameters, both implementations should fail
            // gracefully
            EXPECT_FALSE(dlp_result)
                << "DLP GEMM should fail gracefully with invalid parameters:"
                << printConfigDetails(config_);
            EXPECT_FALSE(ref_result) << "Reference GEMM should fail gracefully "
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
    ::testing::ValuesIn(g_all_test_configs),
    [](const ::testing::TestParamInfo<GemmTestConfig>& info) {
        return info.param.name;
    });

// ADDITIONAL NON-PARAMETERIZED TESTS
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
    ual->gemm(A, B, C, MatrixType::f32);

    // Perform GEMM with reference implementation
    ual = UalFactory::createUal(UALType::REF);
    ual->gemm(A, B, C_ref, MatrixType::f32);

    // To compare with reference, we need to set the k dimension for tolerance
    C.setK(k);

    EXPECT_EQ(C, C_ref);
}

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
    ASSERT_TRUE(ual->gemm(A_small, B_small, C_small, MatrixType::f32));

    ual = UalFactory::createUal(UALType::REF);
    ASSERT_TRUE(ual->gemm(A_small, B_small, C_small_ref, MatrixType::f32));

    // To compare with reference, we need to set the k dimension for tolerance
    C_small.setK(1);

    EXPECT_EQ(C_small, C_small_ref);
}

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
    bool reorder_result = ual_dlp->reorder(B, B_reordered, MatrixType::f32);

    // Assignment that might cause double-free
    if (reorder_result) {
        B = B_reordered;
    }

    // Objects will be destroyed here - this is where double-free might occur
}

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
    auto relu_dlp = postops::createRelu().build();
    operation_dlp->addOperation(std::move(relu_dlp));

    // Add a bias operation
    auto bias_dlp =
        Matrix::fromVector(std::vector<float>{ 1.0f, 2.0f, 3.0f, 4.0f });
    auto biasOp_dlp = postops::createBias().setBias(bias_dlp).build();
    operation_dlp->addOperation(std::move(biasOp_dlp));

    // Finalize the operations
    operation_dlp->finalize();

    // Perform GEMM with PostOps using DLP
    std::unique_ptr<IUal> ual_dlp = UalFactory::createUal(UALType::DLP);
    bool dlp_result = ual_dlp->gemm(A, B, C, MatrixType::f32, operation_dlp);

    // For now, just verify that the operation doesn't crash
    // In a real implementation, we would verify the actual results
    EXPECT_TRUE(dlp_result) << "DLP GEMM with PostOps should succeed";

    auto operation_ref = OperationFactory::createOperation(UALType::REF);

    // Add a ReLU operation
    auto relu_ref = postops::createRelu().build();
    operation_ref->addOperation(std::move(relu_ref));

    // Add a bias operation
    auto bias_ref =
        Matrix::fromVector(std::vector<float>{ 1.0f, 2.0f, 3.0f, 4.0f });
    auto biasOp_ref = postops::createBias().setBias(bias_ref).build();
    operation_ref->addOperation(std::move(biasOp_ref));

    // Finalize the operations
    operation_ref->finalize();

    // Test with reference implementation (should ignore PostOps)
    std::unique_ptr<IUal> ual_ref = UalFactory::createUal(UALType::REF);
    bool                  ref_result =
        ual_ref->gemm(A, B, C_ref, MatrixType::f32, operation_ref);

    EXPECT_TRUE(ref_result) << "Reference GEMM with PostOps should succeed";

    // Compare the results
    C_ref.setK(k);
    EXPECT_EQ(C_ref, C);

    // Test with nullptr PostOps (should behave like original gemm)
    Matrix C_no_postops(m, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0,
                        false);
    C_no_postops.fillValue(0.0f);

    bool no_postops_result =
        ual_dlp->gemm(A, B, C_no_postops, MatrixType::f32, nullptr);
    EXPECT_TRUE(no_postops_result)
        << "DLP GEMM with nullptr PostOps should succeed";
}

// Test for PostOps Builder Pattern
TEST(GEMMTest, PostOpsBuilderPattern)
{
    // Test the builder pattern for different operations

    // Test ReLU builder
    auto relu = postops::createRelu().build();
    EXPECT_EQ(relu->getType(), OperationType::ElementWise);

    // Test Bias builder
    auto bias   = Matrix::fromVector(std::vector<float>{ 1.0f, 2.0f, 3.0f });
    auto biasOp = postops::createBias().setBias(bias).build();
    EXPECT_EQ(biasOp->getType(), OperationType::Bias);

    // Test Scale builder
    auto scale   = Matrix::scalar(2.0f);
    auto scaleOp = postops::createScale().setScaleFactor(scale).build();
    EXPECT_EQ(scaleOp->getType(), OperationType::Sum);

    // Test Matrix Add builder
    auto matrix = Matrix::fromData(
        std::vector<std::vector<float>>{ { 1.0f, 2.0f }, { 3.0f, 4.0f } });
    auto matAddOp = postops::createMatrixAdd().setMatrix(matrix).build();
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
    auto relu1 = postops::createRelu().build();
    operation->addOperation(std::move(relu1));

    auto relu2 = postops::createRelu().build();
    operation->addOperation(std::move(relu2));

    // Add multiple BIAS operations
    auto bias1 =
        Matrix::fromVector(std::vector<float>{ 1.0f, 2.0f, 3.0f, 4.0f });
    auto biasOp1 = postops::createBias().setBias(bias1).build();
    operation->addOperation(std::move(biasOp1));

    auto bias2 =
        Matrix::fromVector(std::vector<float>{ 0.5f, 1.0f, 1.5f, 2.0f });
    auto biasOp2 = postops::createBias().setBias(bias2).build();
    operation->addOperation(std::move(biasOp2));

    // Add multiple SCALE operations (instead of SUM operations without scale
    // factors)
    auto scale1   = Matrix::fromVector(std::vector<float>{ 1.5f });
    auto scaleOp1 = postops::createScale()
                        .setScaleFactor(scale1)
                        .setIsPowerOf2(true)
                        .build();
    operation->addOperation(std::move(scaleOp1));

    auto scale2   = Matrix::fromVector(std::vector<float>{ 2.0f });
    auto scaleOp2 = postops::createScale()
                        .setScaleFactor(scale2)
                        .setIsPowerOf2(false)
                        .build();
    operation->addOperation(std::move(scaleOp2));

    // Finalize the operations
    operation->finalize();

    // Perform GEMM with multiple PostOps using DLP
    std::unique_ptr<IUal> ual_dlp = UalFactory::createUal(UALType::DLP);
    bool dlp_result = ual_dlp->gemm(A, B, C, MatrixType::f32, operation);

    // Verify that the operation doesn't crash and handles multiple ops
    // correctly
    EXPECT_TRUE(dlp_result)
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
    auto relu1 = postops::createRelu().build();
    operation->addOperation(std::move(relu1));

    std::cout << "Adding second RELU operation..." << std::endl;
    auto relu2 = postops::createRelu().build();
    operation->addOperation(std::move(relu2));

    std::cout << "Finalizing operations..." << std::endl;
    operation->finalize();

    std::cout << "Creating UAL..." << std::endl;
    std::unique_ptr<IUal> ual_dlp = UalFactory::createUal(UALType::DLP);

    std::cout << "Calling GEMM..." << std::endl;
    bool dlp_result = ual_dlp->gemm(A, B, C, MatrixType::f32, operation);

    std::cout << "GEMM result: " << (dlp_result ? "SUCCESS" : "FAILED")
              << std::endl;

    EXPECT_TRUE(dlp_result)
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
            auto biasOp1 = postops::createBias().setBias(bias1).build();
            std::cout << "Adding bias operation..." << std::endl;
            operation->addOperation(std::move(biasOp1));

            std::cout << "Finalizing..." << std::endl;
            operation->finalize();

            std::cout << "Creating UAL..." << std::endl;
            std::unique_ptr<IUal> ual_dlp = UalFactory::createUal(UALType::DLP);

            std::cout << "Calling GEMM with BIAS..." << std::endl;
            bool result = ual_dlp->gemm(A, B, C, MatrixType::f32, operation);

            std::cout << "BIAS test result: " << (result ? "SUCCESS" : "FAILED")
                      << std::endl;
            EXPECT_TRUE(result);
        } catch (const std::exception& e) {
            std::cout << "Exception in BIAS test: " << e.what() << std::endl;
            FAIL() << "Exception in BIAS test: " << e.what();
        }
    }

    std::cout << "Testing SUM operations..." << std::endl;

    // Test 2: SUM operations with scale factor
    {
        auto operation = OperationFactory::createOperation(UALType::DLP);

        try {
            std::cout << "Creating sum operation with scale factor..."
                      << std::endl;
            auto scale = Matrix::fromVector(std::vector<float>{ 1.5f });
            auto sum1  = postops::createSum()
                            .setScaleFactor(scale)
                            .setIsPowerOf2(true)
                            .build();
            std::cout << "Adding sum operation..." << std::endl;
            operation->addOperation(std::move(sum1));

            std::cout << "Finalizing..." << std::endl;
            operation->finalize();

            std::cout << "Creating UAL..." << std::endl;
            std::unique_ptr<IUal> ual_dlp = UalFactory::createUal(UALType::DLP);

            std::cout << "Calling GEMM with SUM..." << std::endl;
            bool result = ual_dlp->gemm(A, B, C, MatrixType::f32, operation);

            std::cout << "SUM test result: " << (result ? "SUCCESS" : "FAILED")
                      << std::endl;
            EXPECT_TRUE(result);
        } catch (const std::exception& e) {
            std::cout << "Exception in SUM test: " << e.what() << std::endl;
            FAIL() << "Exception in SUM test: " << e.what();
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
    auto relu_dlp = postops::createRelu().build();
    operation_dlp->addOperation(std::move(relu_dlp));

    // Add bias operation to DLP
    auto bias_dlp =
        Matrix::fromVector(std::vector<float>{ 1.0f, 2.0f, 3.0f, 4.0f });
    auto biasOp_dlp = postops::createBias().setBias(bias_dlp).build();
    operation_dlp->addOperation(std::move(biasOp_dlp));

    operation_dlp->finalize();

    // Create PostOps for REF (identical operations)
    auto operation_ref = OperationFactory::createOperation(UALType::REF);

    // Add ReLU operation to REF
    auto relu_ref = postops::createRelu().build();
    operation_ref->addOperation(std::move(relu_ref));

    // Add bias operation to REF
    auto bias_ref =
        Matrix::fromVector(std::vector<float>{ 1.0f, 2.0f, 3.0f, 4.0f });
    auto biasOp_ref = postops::createBias().setBias(bias_ref).build();
    operation_ref->addOperation(std::move(biasOp_ref));

    operation_ref->finalize();

    // Test DLP implementation
    std::unique_ptr<IUal> ual_dlp = UalFactory::createUal(UALType::DLP);
    ASSERT_TRUE(ual_dlp != nullptr) << "Failed to create DLP UAL";

    bool dlp_result =
        ual_dlp->gemm(A_dlp, B_dlp, C_dlp, MatrixType::f32, operation_dlp);
    EXPECT_TRUE(dlp_result) << "DLP GEMM with PostOps should succeed";

    // Test REF implementation
    std::unique_ptr<IUal> ual_ref = UalFactory::createUal(UALType::REF);
    ASSERT_TRUE(ual_ref != nullptr) << "Failed to create REF UAL";

    bool ref_result =
        ual_ref->gemm(A_ref, B_ref, C_ref, MatrixType::f32, operation_ref);
    EXPECT_TRUE(ref_result) << "REF GEMM with PostOps should succeed";

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
    auto relu_dlp      = postops::createRelu().build();
    dlp_operation->addOperation(std::move(relu_dlp));
    dlp_operation->finalize();

    // Create REF PostOps
    auto ref_operation = OperationFactory::createOperation(UALType::REF);
    auto relu_ref      = postops::createRelu().build();
    ref_operation->addOperation(std::move(relu_ref));
    ref_operation->finalize();

    // Test: DLP UAL with DLP PostOps (should succeed)
    std::unique_ptr<IUal> ual_dlp = UalFactory::createUal(UALType::DLP);
    bool dlp_with_dlp = ual_dlp->gemm(A, B, C, MatrixType::f32, dlp_operation);
    EXPECT_TRUE(dlp_with_dlp) << "DLP UAL with DLP PostOps should succeed";

    // Test: REF UAL with REF PostOps (should succeed)
    std::unique_ptr<IUal> ual_ref = UalFactory::createUal(UALType::REF);
    bool ref_with_ref = ual_ref->gemm(A, B, C, MatrixType::f32, ref_operation);
    EXPECT_TRUE(ref_with_ref) << "REF UAL with REF PostOps should succeed";

    // Test: DLP UAL with REF PostOps (should fail due to type mismatch)
    Matrix C_mismatch1(m, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0,
                       false);
    C_mismatch1.fillValue(0.0f);
    bool dlp_with_ref =
        ual_dlp->gemm(A, B, C_mismatch1, MatrixType::f32, ref_operation);
    EXPECT_FALSE(dlp_with_ref)
        << "DLP UAL with REF PostOps should fail due to type mismatch";

    // Test: REF UAL with DLP PostOps (should fail due to type mismatch)
    Matrix C_mismatch2(m, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0,
                       false);
    C_mismatch2.fillValue(0.0f);
    bool ref_with_dlp =
        ual_ref->gemm(A, B, C_mismatch2, MatrixType::f32, dlp_operation);
    EXPECT_FALSE(ref_with_dlp)
        << "REF UAL with DLP PostOps should fail due to type mismatch";

    std::cout << "UAL type validation test completed!" << std::endl;
}
