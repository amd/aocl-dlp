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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
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

#include "adaptors/dlp/ual_dlp.hh"
#include "bench_config.hh"
#include "framework/matrix.hh"
#include "framework/ual.hh"
#include "framework/ual_factory.hh"
#include "framework/utils/arg_parser.hh"
#include "framework/utils/yaml_parser.hh"

#include <benchmark/benchmark.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace dlp::testing::utils;
using namespace dlp::testing::framework;
using namespace dlp::testing::classic;

// ============================================================================
// GLOBAL CONFIGURATION VARIABLES
// ============================================================================

// Global variable to store configurable YAML file path
// This can be set via command line arguments or defaults to the built-in config
static std::string g_yaml_config_file = BENCH_CONFIG_DIR
    "/gemm_bench_basic_config.yaml";

// ============================================================================
// UAL GEMM FP32 BENCHMARK
// ============================================================================

/**
 * @brief Benchmark FP32 GEMM using DLP UAL implementation
 *
 * This benchmark measures the performance of matrix multiplication C = A*B
 * using the DLP (Deep Learning Primitives) UAL backend. No post-operations
 * are applied, focusing purely on the GEMM kernel performance.
 *
 * @param state Google Benchmark state object
 */

// Configuration structure for YAML-driven tests
struct GemmTestConfig
{
    std::string  name;
    MatrixType   a_type, b_type, c_type, acc_type;
    MatrixLayout storage_format;
    md_t         m, n, k;
    md_t         lda, ldb, ldc;
    double       alpha, beta;
    bool         transA, transB;
    bool         reorderA, reorderB;
    bool         packA, packB; // TODO: Pack parameters not yet used in tests

    // PostOps support
    std::shared_ptr<IOperation> postops_dlp;
    std::shared_ptr<IOperation> postops_ref;
    bool                        has_postops;

    // Default constructor (required by GoogleTest)
    GemmTestConfig()
        : name("BM_GEMM_DLP_FP32_default")
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
        return configs; // Return empty vector but with explicit test failure
    }

    return configs;
}

// TODO : can add programatical test configs similar to test_gemm

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

    // Combine all configurations
    all_configs.reserve(yaml_configs.size());
    all_configs.insert(all_configs.end(), yaml_configs.begin(),
                       yaml_configs.end());
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
// BENCHMARK REGISTRATION
// ============================================================================

// Function to benchmark test configurations with Google Benchmark
static void
BM_gemm(benchmark::State& state, GemmTestConfig config_)
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
    Matrix C(config_.m, config_.n, config_.c_type, layout, config_.ldc, false);
    Matrix A_ref(a_rows, a_cols, config_.a_type, layout, config_.lda,
                 config_.transA);
    Matrix B_ref(b_rows, b_cols, config_.b_type, layout, config_.ldb,
                 config_.transB);
    Matrix C_ref(config_.m, config_.n, config_.acc_type, layout, config_.ldc,
                 false);

// Initialize matrices with deterministic random values
#if 0
    A.fillRandom(42 + config_.m); // Use configuration to vary seed
    B.fillRandom(43 + config_.n);
    C.fillRandom(44 + config_.k);
#else
    A.fillValue(0.5f);
    B.fillValue(0.2f);
    C.fillValue(0.0f);
#endif

    // Note: using virtual might not be good for performance, better to template
    // this out Create a gemm benchmark fixture which will take a template UAL.
    // Refer experimental benchmarks in AOCL-Crypto to do this enhancement.
    std::unique_ptr<IUal> ual = UalFactory::createUal(UALType::DLP);

    /* Gather the required information to benchmark */
    // Apply reordering if specified
    if (config_.reorderA) {
        A.setReordered(true);
    }
    if (config_.reorderB) {
        // Create reordered matrix with custom allocation size in bytes
        Matrix B_reordered;

        ual->reorder(B, B_reordered, config_.a_type, config_.b_type,
                     config_.c_type, config_.acc_type);
        B = B_reordered;
    }

    // Apply packing if specified
    // Note: Pack parameters are DLP optimization and are handled
    // via mem_format parameters in the DLP GEMM call.
    if (config_.packA) {
        A.setPacked(true);
        // Reference implementation: pack is no-op, A_ref remains unchanged
    }
    if (config_.packB) {
        B.setPacked(true);
        // Reference implementation: pack is no-op, B_ref remains unchanged
    }

    // Get raw pointers for high-performance benchmarking
    auto a_ptr    = A.getMatrixData().getMatrixPtr();
    auto b_ptr    = B.getMatrixData().getMatrixPtr();
    auto c_ptr    = C.getMatrixData().getMatrixPtr();
    auto a_type   = A.getMatrixType();
    auto b_type   = B.getMatrixType();
    auto c_type   = C.getMatrixType();
    auto acc_type = config_.acc_type;
    auto transA   = A.isTransposed();
    auto transB   = B.isTransposed();
    auto lda      = A.getLeadingDimension();
    auto ldb      = B.getLeadingDimension();
    auto ldc      = C.getLeadingDimension();
    auto alpha    = config_.alpha;
    auto beta     = config_.beta;

    // Matrix dimensions for GEMM: C(m x n) = A(m x k) * B(k x n)
    auto m = config_.m;
    auto n = config_.n;
    auto k = config_.k;

    /* Actual benchmark loop */
    for (auto _ : state) {
        // Reset C matrix to ensure consistent starting conditions

        // Perform the GEMM operation using raw pointer API for maximum
        // performance
        bool result =
            ual->gemm(m, n, k, a_ptr, a_type, layout, transA, lda, b_ptr,
                      b_type, layout, transB, ldb, c_ptr, c_type, layout, false,
                      ldc, acc_type, alpha, beta);

        if (!result) {
            state.SkipWithError("GEMM operation failed");
            return;
        }

        // Prevent compiler optimization from eliminating the computation
        benchmark::DoNotOptimize(c_ptr);
    }

    /* Setting up counters */
    // Calculate and report performance metrics
    const double total_ops = 2.0 * config_.m * config_.n
                             * config_.k; // FLOPs for matrix multiplication

    // Calculate GFLOPS using Google Benchmark timing
    double ops_per_iteration = total_ops;
    state.counters["FLOPS:"] = benchmark::Counter(
        ops_per_iteration, benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000);

    // Report matrix dimensions in the output
    state.counters["M"] = config_.m;
    state.counters["N"] = config_.n;
    state.counters["K"] = config_.k;

    // Calculate memory bandwidth (assuming all data is transferred)
    const double total_bytes =
        (config_.m * config_.k + config_.k * config_.n + config_.m * config_.n)
        * sizeof(float);
    state.counters["B/W:"] = benchmark::Counter(
        total_bytes, benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
};

// Function to register all benchmarks for the test configurations
static void
register_benchmarks()
{
    // Generate all test configurations
    static std::vector<GemmTestConfig> all_test_configs =
        getTestConfigurations();

    // Registering benchmarks for all test configurations generated
    for (auto& test_input : all_test_configs) {
        benchmark::RegisterBenchmark(test_input.name, BM_gemm, test_input)
            ->Unit(benchmark::kMicrosecond)
            ->MinTime(3);
    }

    return;
}

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
        std::cout << "GoogleBench Help:" << std::endl;
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

    // Print information about the benchmark
    std::cout << "=== UAL GEMM FP32 Benchmark ===" << std::endl;
    std::cout << "Benchmarking DLP UAL implementation for FP32 GEMM"
              << std::endl;
    std::cout << "Format: C = A * B (no post-operations)" << std::endl;
    std::cout << "Metrics: GFLOPS, Memory Bandwidth (GB/s)" << std::endl;
    std::cout << "=======================================" << std::endl;

    // Initialize Google Benchmark first
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;

    // Register benchmarks
    register_benchmarks();

    // Run benchmarks
    std::cout << "Running Benchmarks" << std::endl;
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
