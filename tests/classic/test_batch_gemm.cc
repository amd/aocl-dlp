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
#include "framework/matrix.hh"
#include "framework/operation.hh"
#include "framework/ual.hh"
#include "framework/ual_factory.hh"
#include "framework/utils/arg_parser.hh"
#include "framework/utils/yaml_parser.hh"
#include "test_config.hh"
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif
#include <tuple>
#include <vector>

using namespace dlp::testing::framework;
using namespace dlp::testing::framework::postops;
using namespace dlp::testing::utils;

constexpr size_t MAX_CONFIGS_PER_TEST_SET = 50000;

// Global variable to store configurable YAML file paths. These can be set via
// command line arguments (one or more -f/--file flags, and/or comma-separated
// values) or default to the single built-in config file.
static std::vector<std::string> g_batch_gemm_yaml_files = {
    TEST_CONFIG_DIR "/batch_gemm_test_config.yaml"
};

// Helper to derive a sanitized, GoogleTest-safe identifier from a YAML file
// path (uses the file stem, replacing any non-alphanumeric characters with
// underscores). Used to keep test names unique across multiple YAML files.
static std::string
sanitizeBatchFileStem(const std::string& yaml_file)
{
    std::string stem = std::filesystem::path(yaml_file).stem().string();
    for (char& c : stem) {
        if (!std::isalnum(static_cast<unsigned char>(c))) {
            c = '_';
        }
    }
    if (stem.empty()) {
        stem = "file";
    }
    return stem;
}

// ============================================================================
// CONFIGURATION STRUCTURE
// ============================================================================

/**
 * @brief Configuration for a single batch GEMM test case
 *
 * For single-group mode: represents one test with one group (cartesian
 * variations) For multi-group mode: represents one test with multiple groups
 * (simple product)
 */
struct BatchGemmTestConfig
{
    std::string name = "";

    // Matrix types (same for all groups in a test)
    MatrixType   a_type         = MatrixType::f32;
    MatrixType   b_type         = MatrixType::f32;
    MatrixType   c_type         = MatrixType::f32;
    MatrixType   acc_type       = MatrixType::f32;
    MatrixLayout storage_format = MatrixLayout::ROW_MAJOR;

    // Group parameters (vectors support multi-group configurations)
    std::vector<md_t>   m_values;
    std::vector<md_t>   n_values;
    std::vector<md_t>   k_values;
    std::vector<md_t>   lda_values;
    std::vector<md_t>   ldb_values;
    std::vector<md_t>   ldc_values;
    std::vector<double> alpha_values;
    std::vector<double> beta_values;
    std::vector<bool>   transA_values;
    std::vector<bool>   transB_values;
    std::vector<bool>   reorderA_values;
    std::vector<bool>   reorderB_values;
    std::vector<bool>   packA_values;
    std::vector<bool>   packB_values;
    std::vector<md_t>   group_size_values;

    // Tolerance configuration
    bool   has_tolerances     = false;
    double tolerance_relative = -1.0;
    double tolerance_absolute = -1.0;

    // Optional NaN-equality opt-in for deliberate NaN-propagation tests.
    // Default (false) treats a NaN in the output as a mismatch.
    bool treat_nan_equal = false;

    // PostOps support (per-group to handle dimension-dependent PostOps)
    bool has_postops = false;
    std::vector<std::vector<std::unique_ptr<IOperationParam>>>
        post_op_params_per_group;

    // Test identification
    size_t config_index = 0;

    // Default constructor
    BatchGemmTestConfig() = default;

    // Move constructor and assignment (defaulted)
    BatchGemmTestConfig(BatchGemmTestConfig&&)            = default;
    BatchGemmTestConfig& operator=(BatchGemmTestConfig&&) = default;

    // Copy constructor - needed for GTest parameterized tests
    BatchGemmTestConfig(const BatchGemmTestConfig& other)
        : name(other.name)
        , a_type(other.a_type)
        , b_type(other.b_type)
        , c_type(other.c_type)
        , acc_type(other.acc_type)
        , storage_format(other.storage_format)
        , m_values(other.m_values)
        , n_values(other.n_values)
        , k_values(other.k_values)
        , lda_values(other.lda_values)
        , ldb_values(other.ldb_values)
        , ldc_values(other.ldc_values)
        , alpha_values(other.alpha_values)
        , beta_values(other.beta_values)
        , transA_values(other.transA_values)
        , transB_values(other.transB_values)
        , reorderA_values(other.reorderA_values)
        , reorderB_values(other.reorderB_values)
        , packA_values(other.packA_values)
        , packB_values(other.packB_values)
        , group_size_values(other.group_size_values)
        , has_tolerances(other.has_tolerances)
        , tolerance_relative(other.tolerance_relative)
        , tolerance_absolute(other.tolerance_absolute)
        , treat_nan_equal(other.treat_nan_equal)
        , has_postops(other.has_postops)
        , config_index(other.config_index)
    {
        post_op_params_per_group.reserve(other.post_op_params_per_group.size());
        for (const auto& group_params : other.post_op_params_per_group) {
            std::vector<std::unique_ptr<IOperationParam>> cloned;
            cloned.reserve(group_params.size());
            for (const auto& p : group_params) {
                if (p) {
                    cloned.push_back(p->clone());
                }
            }
            post_op_params_per_group.push_back(std::move(cloned));
        }
    }

    // Copy assignment
    BatchGemmTestConfig& operator=(const BatchGemmTestConfig& other)
    {
        if (this != &other) {
            BatchGemmTestConfig tmp(other);
            *this = std::move(tmp);
        }
        return *this;
    }

    /**
     * @brief Get number of groups in this test configuration
     */
    size_t getGroupCount() const
    {
        return group_size_values.empty() ? 0 : group_size_values.size();
    }

    /**
     * @brief Check if this is a multi-group test
     */
    bool isMultiGroup() const { return getGroupCount() > 1; }
};

namespace {

BatchGroup
makeF32Group(md_t         m,
             md_t         n,
             md_t         k,
             size_t       matrix_count,
             double       alpha,
             double       beta,
             uint32_t     seed_offset,
             MatrixLayout layoutA = MatrixLayout::ROW_MAJOR,
             MatrixLayout layoutB = MatrixLayout::ROW_MAJOR,
             MatrixLayout layoutC = MatrixLayout::ROW_MAJOR,
             bool         transA  = false,
             bool         transB  = false)
{
    BatchGroup group;
    group.m     = m;
    group.n     = n;
    group.k     = k;
    group.alpha = alpha;
    group.beta  = beta;

    for (std::size_t i = 0; i < matrix_count; ++i) {
        md_t a_rows = transA ? k : m;
        md_t a_cols = transA ? m : k;
        md_t b_rows = transB ? n : k;
        md_t b_cols = transB ? k : n;

        Matrix A(a_rows, a_cols, MatrixType::f32, layoutA, -1, transA);
        Matrix B(b_rows, b_cols, MatrixType::f32, layoutB, -1, transB);
        Matrix C(m, n, MatrixType::f32, layoutC, -1, false);

        A.fillRandom(static_cast<uint32_t>(42 + seed_offset + i));
        B.fillRandom(static_cast<uint32_t>(142 + seed_offset + i));
        C.fillRandom(static_cast<uint32_t>(242 + seed_offset + i));

        group.A_matrices.emplace_back(std::move(A));
        group.B_matrices.emplace_back(std::move(B));
        group.C_matrices.emplace_back(std::move(C));
    }

    group.memFormatA = deduce_mem_format(group.A_matrices.front());
    group.memFormatB = deduce_mem_format(group.B_matrices.front());
    return group;
}

std::vector<BatchGroup>
cloneGroups(const std::vector<BatchGroup>& groups)
{
    std::vector<BatchGroup> copies;
    copies.reserve(groups.size());
    for (const auto& src : groups) {
        BatchGroup dst;
        dst.A_matrices = src.A_matrices;
        dst.B_matrices = src.B_matrices;
        dst.C_matrices = src.C_matrices;
        dst.m          = src.m;
        dst.n          = src.n;
        dst.k          = src.k;
        dst.alpha      = src.alpha;
        dst.beta       = src.beta;
        dst.memFormatA = src.memFormatA;
        dst.memFormatB = src.memFormatB;
        for (const auto& p : src.post_op_params) {
            if (p) {
                dst.post_op_params.push_back(p->clone());
            }
        }
        if (src.a_quant) {
            dst.a_quant = std::make_unique<AQuantParam>(*src.a_quant);
        }
        if (src.group_scale) {
            dst.group_scale =
                std::make_unique<GroupScaleParam>(*src.group_scale);
        }
        copies.push_back(std::move(dst));
    }
    return copies;
}

void
compareGroupResults(const std::vector<BatchGroup>& lhs,
                    const std::vector<BatchGroup>& rhs)
{
    ASSERT_EQ(lhs.size(), rhs.size());
    for (std::size_t g = 0; g < lhs.size(); ++g) {
        const auto& lhs_group = lhs[g];
        const auto& rhs_group = rhs[g];

        ASSERT_EQ(lhs_group.C_matrices.size(), rhs_group.C_matrices.size());

        for (std::size_t i = 0; i < lhs_group.C_matrices.size(); ++i) {
            lhs_group.C_matrices[i].setK(lhs_group.k);
            rhs_group.C_matrices[i].setK(rhs_group.k);

            auto result = lhs_group.C_matrices[i].compare(
                rhs_group.C_matrices[i], MatrixCompareOptions::Fast());
            EXPECT_TRUE(result.equal) << FormatCompareResult(
                result, lhs_group.C_matrices[i], rhs_group.C_matrices[i]);
        }
    }
}

// ============================================================================
// CONFIGURATION LOADING
// ============================================================================

/**
 * @brief Load batch GEMM test configurations from YAML file
 *
 * Supports both single-group (cartesian) and multi-group (simple) modes.
 * Validation is handled by the YAML parser.
 *
 * For CARTESIAN_PRODUCT mode: Each MicroTest iteration = separate test
 * For SIMPLE_PRODUCT mode: All MicroTest iterations = groups in ONE test
 *
 * @param yaml_file   Path to the YAML config file to load.
 * @param namePrefix  Optional prefix prepended to every generated test name so
 *                    that names remain unique when multiple YAML files are
 *                    loaded in a single run. Empty for the single-file case
 *                    (preserving the original naming scheme).
 */
std::vector<BatchGemmTestConfig>
loadBatchGemmTestConfigurations(const std::string& yaml_file,
                                const std::string& namePrefix = "")
{
    std::vector<BatchGemmTestConfig> configs;

    // Build the prefix fragment applied to each generated test name.
    const std::string prefix = namePrefix.empty() ? std::string()
                                                  : (namePrefix + "_");

    try {
        YamlParser parser(yaml_file, "batch_gemm_tests");
        // Don't override yield type - let each test's product_type control it

        size_t microTestCount = parser.getMicroTestCount();
        std::cout << "Loading batch GEMM tests from: " << yaml_file
                  << std::endl;
        std::cout << "Found " << microTestCount << " test set(s)" << std::endl;

        size_t total_configs = 0;

        for (std::size_t i = 0; i < microTestCount; ++i) {
            MicroTest& microTest          = parser.getMicroTest();
            YieldType  yield_type         = microTest.getYieldType();
            size_t     total_combinations = microTest.getSize();
            size_t     test_count =
                std::min(total_combinations, MAX_CONFIGS_PER_TEST_SET);

            if (yield_type == YieldType::SIMPLE_PRODUCT) {
                // SIMPLE_PRODUCT: All iterations = groups in ONE test
                std::cout << "Test set " << i << ": Multi-group test with "
                          << test_count << " groups" << std::endl;

                BatchGemmTestConfig config;

                // Get the current test set name from YAML (e.g.,
                // "batch_test_name")
                std::string currentTestName = parser.getCurrentTestName();

                // Generate test name (with per-file prefix so names stay
                // unique across multiple YAML files).
                config.name = prefix + "yaml_" + std::to_string(i) + "_"
                              + currentTestName + "_MultiGroup";
                config.config_index = total_configs;

                // Extract common parameters from first iteration
                config.a_type         = microTest.getAType();
                config.b_type         = microTest.getBType();
                config.c_type         = microTest.getCType();
                config.acc_type       = microTest.getAccType();
                config.storage_format = microTest.getStorageFormat();

                // Extract tolerances if present
                if (microTest.hasTolerances()) {
                    config.has_tolerances     = true;
                    const auto& tol           = microTest.getTolerances();
                    config.tolerance_relative = tol.relative;
                    config.tolerance_absolute = tol.absolute;
                }

                config.treat_nan_equal = microTest.getTreatNaNEqual();

                // Collect all iterations as groups
                size_t j = 0;
                do {
                    if (j >= test_count) {
                        break; // Respect the max limit
                    }

                    // Each iteration is a group with its own dimensions
                    config.m_values.push_back(microTest.getM());
                    config.n_values.push_back(microTest.getN());
                    config.k_values.push_back(microTest.getK());
                    config.lda_values.push_back(microTest.getLDA());
                    config.ldb_values.push_back(microTest.getLDB());
                    config.ldc_values.push_back(microTest.getLDC());
                    config.alpha_values.push_back(microTest.getAlpha());
                    config.beta_values.push_back(microTest.getBeta());
                    config.transA_values.push_back(microTest.getTransA());
                    config.transB_values.push_back(microTest.getTransB());
                    config.reorderA_values.push_back(microTest.getReorderA());
                    config.reorderB_values.push_back(microTest.getReorderB());
                    config.packA_values.push_back(microTest.getPackA());
                    config.packB_values.push_back(microTest.getPackB());
                    config.group_size_values.push_back(
                        microTest.getGroupSize());

                    // Extract PostOps for THIS group (sized for current
                    // dimensions)
                    auto params  = microTest.getPostOpParams();
                    auto a_quant = microTest.getAQuantParam();
                    if (a_quant) {
                        params.push_back(std::move(a_quant));
                    }
                    auto group_scale = microTest.getGroupScaleParam();
                    if (group_scale) {
                        params.push_back(std::move(group_scale));
                    }
                    bool has_ops = !params.empty();
                    config.post_op_params_per_group.emplace_back(
                        std::move(params));

                    config.has_postops = has_ops;

                    j++;

                    // Move to next iteration using standard pattern
                    if (microTest.hasNext()) {
                        microTest.next();
                    } else {
                        break; // No more iterations
                    }
                } while (true);

                configs.push_back(std::move(config));
                total_configs++;

            } else {
                // CARTESIAN_PRODUCT: Each iteration = separate test
                std::cout << "Test set " << i << ": Processing " << test_count
                          << " out of " << total_combinations << " combinations"
                          << std::endl;

                size_t j = 0;
                do {
                    if (j >= test_count) {
                        break; // Respect the max limit
                    }

                    BatchGemmTestConfig config;

                    // Get the current test set name from YAML
                    std::string currentTestName = parser.getCurrentTestName();

                    // Generate test name (with per-file prefix so names stay
                    // unique across multiple YAML files).
                    config.name = prefix + "yaml_" + std::to_string(i) + "_"
                                  + currentTestName + "_" + std::to_string(j);
                    config.config_index = total_configs;

                    // Extract common parameters
                    config.a_type         = microTest.getAType();
                    config.b_type         = microTest.getBType();
                    config.c_type         = microTest.getCType();
                    config.acc_type       = microTest.getAccType();
                    config.storage_format = microTest.getStorageFormat();

                    // Extract group parameters (single group per test)
                    config.m_values.push_back(microTest.getM());
                    config.n_values.push_back(microTest.getN());
                    config.k_values.push_back(microTest.getK());
                    config.lda_values.push_back(microTest.getLDA());
                    config.ldb_values.push_back(microTest.getLDB());
                    config.ldc_values.push_back(microTest.getLDC());
                    config.alpha_values.push_back(microTest.getAlpha());
                    config.beta_values.push_back(microTest.getBeta());
                    config.transA_values.push_back(microTest.getTransA());
                    config.transB_values.push_back(microTest.getTransB());
                    config.reorderA_values.push_back(microTest.getReorderA());
                    config.reorderB_values.push_back(microTest.getReorderB());
                    config.packA_values.push_back(microTest.getPackA());
                    config.packB_values.push_back(microTest.getPackB());
                    config.group_size_values.push_back(
                        microTest.getGroupSize());

                    // Extract tolerances if present
                    if (microTest.hasTolerances()) {
                        config.has_tolerances     = true;
                        const auto& tol           = microTest.getTolerances();
                        config.tolerance_relative = tol.relative;
                        config.tolerance_absolute = tol.absolute;
                    }

                    config.treat_nan_equal = microTest.getTreatNaNEqual();

                    // CARTESIAN: Extract PostOps once (single group per test)
                    auto params  = microTest.getPostOpParams();
                    auto a_quant = microTest.getAQuantParam();
                    if (a_quant) {
                        params.push_back(std::move(a_quant));
                    }
                    auto group_scale = microTest.getGroupScaleParam();
                    if (group_scale) {
                        params.push_back(std::move(group_scale));
                    }
                    bool has_ops = !params.empty();
                    config.post_op_params_per_group.emplace_back(
                        std::move(params));
                    config.has_postops = has_ops;

                    configs.push_back(std::move(config));
                    total_configs++;
                    j++;

                    // Move to next iteration using standard pattern
                    if (microTest.hasNext()) {
                        microTest.next();
                    } else {
                        break; // No more iterations
                    }
                } while (true);
            }

            if (i < microTestCount - 1) {
                parser.next();
            }
        }

        std::cout << "Loaded " << configs.size()
                  << " batch GEMM test configurations" << std::endl;

    } catch (const std::exception& e) {
        // Let all exceptions propagate to gtest - no graceful handling
        std::cerr << "Error loading batch GEMM YAML configuration: " << e.what()
                  << std::endl;
        ADD_FAILURE() << "Failed to load YAML: " << e.what();
        throw; // Re-throw to ensure test failure is visible
    }

    return configs;
}
/**
 * @brief Convert BatchGemmTestConfig to BatchGroup vector for execution
 *
 * Creates the actual matrix data and group structures from config.
 *
 * @param config Test configuration
 * @param postops_per_group PostOps per group (dimension-aware), can be empty
 * @return Vector of BatchGroup objects ready for execution
 */
std::vector<BatchGroup>
configToGroups(const BatchGemmTestConfig& config,
               const std::vector<std::vector<std::unique_ptr<IOperationParam>>>&
                   postops_per_group = {})
{
    std::vector<BatchGroup> groups;
    groups.reserve(config.getGroupCount());

    for (std::size_t g = 0; g < config.getGroupCount(); ++g) {
        BatchGroup group;

        // Set group dimensions
        group.m     = config.m_values[g];
        group.n     = config.n_values[g];
        group.k     = config.k_values[g];
        group.alpha = config.alpha_values[g];
        group.beta  = config.beta_values[g];

        bool transA = config.transA_values[g];
        bool transB = config.transB_values[g];

        // Create matrices for this group
        md_t group_size = config.group_size_values[g];

        // Support group_size=0 for empty groups (valid test case)
        // Only create matrices if group_size > 0
        for (iter_t mat_idx = 0; mat_idx < group_size; ++mat_idx) {
            // Calculate matrix dimensions based on transpose flags
            md_t a_rows = transA ? group.k : group.m;
            md_t a_cols = transA ? group.m : group.k;
            md_t b_rows = transB ? group.n : group.k;
            md_t b_cols = transB ? group.k : group.n;

            // Matrix constructor now handles zero/negative dimensions
            // gracefully (stores actual dims, allocates safe memory)
            Matrix A(a_rows, a_cols, config.a_type, config.storage_format,
                     config.lda_values[g], transA);
            Matrix B(b_rows, b_cols, config.b_type, config.storage_format,
                     config.ldb_values[g], transB);
            Matrix C(group.m, group.n, config.c_type, config.storage_format,
                     config.ldc_values[g], false);

            // Fill with random data (use group + matrix index for unique seeds)
            uint32_t seed_offset = g * 1000 + mat_idx + config.config_index;
            A.fillRandom(42 + seed_offset);
            B.fillRandom(142 + seed_offset);
            C.fillRandom(242 + seed_offset);

            // FIXME: Have to take a decision on reorder outside or inside
            // Apply matrix tags (reorder/pack) based on config
            // Note: reorderA just sets a flag; actual reordering happens in the
            // GEMM call
            if (config.reorderA_values[g]) {
                A.setReordered(true);
            }

            // For reorderB, we set the flag. The actual reordering can be done
            // by UAL or handled via memory format in the batch GEMM call. For
            // batch GEMM, we just set the flag and let the memory format deduce
            // it.
            if (config.reorderB_values[g]) {
                B.setReordered(true);
            }

            // Pack flags are optimization hints handled via mem_format in GEMM
            // call
            if (config.packA_values[g]) {
                A.setPacked(true);
            }

            if (config.packB_values[g]) {
                B.setPacked(true);
            }

            group.A_matrices.emplace_back(std::move(A));
            group.B_matrices.emplace_back(std::move(B));
            group.C_matrices.emplace_back(std::move(C));
        }

        // Deduce memory formats (only if matrices were created)
        if (!group.A_matrices.empty() && !group.B_matrices.empty()) {
            group.memFormatA = deduce_mem_format(group.A_matrices.front());
            group.memFormatB = deduce_mem_format(group.B_matrices.front());
        }

        // Assign PostOps (per-group to handle dimension-dependent PostOps)
        // Separate A_Quant params from regular post-ops
        if (!postops_per_group.empty() && g < postops_per_group.size()) {
            for (const auto& p : postops_per_group[g]) {
                if (p) {
                    if (p->getType() == OperationType::A_Quant) {
                        // A_Quant goes to group.a_quant
                        auto* aq = dynamic_cast<const AQuantParam*>(p.get());
                        if (aq) {
                            group.a_quant = std::make_unique<AQuantParam>(*aq);
                        }
                    } else if (p->getType() == OperationType::GroupScale) {
                        auto* gs =
                            dynamic_cast<const GroupScaleParam*>(p.get());
                        if (gs) {
                            group.group_scale =
                                std::make_unique<GroupScaleParam>(*gs);
                        }
                    } else {
                        group.post_op_params.push_back(p->clone());
                    }
                }
            }
        }

        groups.push_back(std::move(group));
    }

    return groups;
}

// ============================================================================
// HELPER FUNCTIONS - VALIDATION
// ============================================================================

bool
check_valid_batch_params(const BatchGemmTestConfig& config)
{
    // Validate each group's parameters following AOCL_DLP_BATCH_GEMM_CHECK
    // macro logic
    for (std::size_t g = 0; g < config.getGroupCount(); ++g) {
        md_t m          = config.m_values[g];
        md_t n          = config.n_values[g];
        md_t k          = config.k_values[g];
        md_t lda        = config.lda_values[g];
        md_t ldb        = config.ldb_values[g];
        md_t ldc        = config.ldc_values[g];
        md_t group_size = config.group_size_values[g];
        bool transA     = config.transA_values[g];
        bool transB     = config.transB_values[g];
        bool reorderA   = config.reorderA_values[g];
        bool reorderB   = config.reorderB_values[g];

        bool col_stored = (config.storage_format == MatrixLayout::COLUMN_MAJOR);
        bool row_stored = (config.storage_format == MatrixLayout::ROW_MAJOR);

        // Check basic dimensions - must be positive
        if (m <= 0 || n <= 0 || k <= 0) {
            return false;
        }

        // Check group_size - must be positive (0 is invalid, causes FPE in
        // thread decorator)
        if (group_size <= 0) {
            return false;
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
                case MatrixType::fp16:
                    break;

                default:
                    break;
            }
        }

        // Physical matrix dimensions
        md_t a_rows = transA ? k : m;
        md_t a_cols = transA ? m : k;
        md_t b_rows = transB ? n : k;
        md_t b_cols = transB ? k : n;

        // LD < 0 means not specified in YAML; compute from dimensions.
        // Row-major: LD = cols, Col-major: LD = rows.
        if (lda == -1) {
            lda = row_stored ? a_cols : a_rows;
        }
        if (ldb == -1) {
            ldb = row_stored ? b_cols : b_rows;
        }
        if (ldc == -1) {
            ldc = row_stored ? n : m;
        }

        // Leading dimension checks for matrix A
        // Skip for reordered matrices as they have custom layouts
        if (!reorderA) {
            if (row_stored
                && ((!transA && (lda < k)) || (transA && (lda < m)))) {
                return false;
            }
            if (col_stored
                && ((!transA && (lda < m)) || (transA && (lda < k)))) {
                return false;
            }
        } else {
            // Reordering of A matrix not supported in row major case
            if (row_stored) {
                return false;
            }
        }

        // Leading dimension checks for matrix B
        // Skip for reordered matrices as they have custom layouts
        if (!reorderB) {
            if (row_stored
                && ((!transB && (ldb < n)) || (transB && (ldb < k)))) {
                return false;
            }
            if (col_stored
                && ((!transB && (ldb < k)) || (transB && (ldb < n)))) {
                return false;
            }
        } else {
            if (row_stored) {
                if (!transB && (ldb < n)) {
                    return false;
                }
                if (transB && (ldb < k)) {
                    return false;
                }
            } else {
                // Reordering column major matrices not supported
                return false;
            }
        }

        // Leading dimension checks for matrix C (never reordered)
        if (row_stored && (ldc < n)) {
            return false;
        }
        if (col_stored && (ldc < m)) {
            return false;
        }
    }

    return true;
}

} // namespace

TEST(BatchGemmTest, SingleGroupSingleMatrix)
{
    std::vector<BatchGroup> base_groups;
    base_groups.push_back(
        makeF32Group(/*m=*/8, /*n=*/8, /*k=*/8, /*matrix_count=*/1,
                     /*alpha=*/1.0, /*beta=*/0.0, /*seed_offset=*/0));

    auto dlp_groups = cloneGroups(base_groups);
    auto ref_groups = cloneGroups(base_groups);

    auto ual_dlp = UalFactory::createUal(UALType::DLP);
    auto ual_ref = UalFactory::createUal(UALType::REF);
    ASSERT_NE(ual_dlp, nullptr);
    ASSERT_NE(ual_ref, nullptr);

    auto status_dlp = ual_dlp->batch_gemm(dlp_groups, MatrixType::f32);
    ASSERT_EQ(status_dlp, UALError::UAL_SUCCESS);

    auto status_ref = ual_ref->batch_gemm(ref_groups, MatrixType::f32);
    ASSERT_EQ(status_ref, UALError::UAL_SUCCESS);

    compareGroupResults(dlp_groups, ref_groups);
}

TEST(BatchGemmTest, MultipleGroupsMultipleMatrices)
{
    std::vector<BatchGroup> base_groups;
    base_groups.push_back(
        makeF32Group(/*m=*/6, /*n=*/4, /*k=*/5, /*matrix_count=*/2,
                     /*alpha=*/1.25, /*beta=*/0.1, /*seed_offset=*/10));
    base_groups.push_back(
        makeF32Group(/*m=*/12, /*n=*/7, /*k=*/9, /*matrix_count=*/3,
                     /*alpha=*/0.75, /*beta=*/0.2, /*seed_offset=*/30));

    auto dlp_groups = cloneGroups(base_groups);
    auto ref_groups = cloneGroups(base_groups);

    auto ual_dlp = UalFactory::createUal(UALType::DLP);
    auto ual_ref = UalFactory::createUal(UALType::REF);
    ASSERT_NE(ual_dlp, nullptr);
    ASSERT_NE(ual_ref, nullptr);

    auto status_dlp = ual_dlp->batch_gemm(dlp_groups, MatrixType::f32);
    ASSERT_EQ(status_dlp, UALError::UAL_SUCCESS);

    auto status_ref = ual_ref->batch_gemm(ref_groups, MatrixType::f32);
    ASSERT_EQ(status_ref, UALError::UAL_SUCCESS);

    compareGroupResults(dlp_groups, ref_groups);
}

TEST(BatchGemmTest, MixedGroupConfigurations)
{
    std::vector<BatchGroup> base_groups;
    base_groups.push_back(
        makeF32Group(/*m=*/5, /*n=*/3, /*k=*/4, /*matrix_count=*/1,
                     /*alpha=*/0.9, /*beta=*/0.3, /*seed_offset=*/50,
                     MatrixLayout::ROW_MAJOR, MatrixLayout::ROW_MAJOR,
                     MatrixLayout::ROW_MAJOR, /*transA=*/false,
                     /*transB=*/true));
    base_groups.push_back(
        makeF32Group(/*m=*/16, /*n=*/16, /*k=*/8, /*matrix_count=*/4,
                     /*alpha=*/1.4, /*beta=*/-0.2, /*seed_offset=*/100,
                     MatrixLayout::ROW_MAJOR, MatrixLayout::ROW_MAJOR,
                     MatrixLayout::ROW_MAJOR, /*transA=*/true,
                     /*transB=*/false));
    base_groups.push_back(
        makeF32Group(/*m=*/9, /*n=*/11, /*k=*/7, /*matrix_count=*/2,
                     /*alpha=*/0.6, /*beta=*/0.5, /*seed_offset=*/300,
                     MatrixLayout::ROW_MAJOR, MatrixLayout::ROW_MAJOR,
                     MatrixLayout::ROW_MAJOR, /*transA=*/false,
                     /*transB=*/false));

    auto dlp_groups = cloneGroups(base_groups);
    auto ref_groups = cloneGroups(base_groups);

    auto ual_dlp = UalFactory::createUal(UALType::DLP);
    auto ual_ref = UalFactory::createUal(UALType::REF);
    ASSERT_NE(ual_dlp, nullptr);
    ASSERT_NE(ual_ref, nullptr);

    auto status_dlp = ual_dlp->batch_gemm(dlp_groups, MatrixType::f32);
    ASSERT_EQ(status_dlp, UALError::UAL_SUCCESS);

    auto status_ref = ual_ref->batch_gemm(ref_groups, MatrixType::f32);
    ASSERT_EQ(status_ref, UALError::UAL_SUCCESS);

    compareGroupResults(dlp_groups, ref_groups);
}

TEST(BatchGemmTest, GlobalPostOpsBasic)
{
    // Create a simple batch group
    std::vector<BatchGroup> base_groups;
    base_groups.push_back(
        makeF32Group(/*m=*/8, /*n=*/8, /*k=*/8, /*matrix_count=*/2,
                     /*alpha=*/1.0, /*beta=*/0.0, /*seed_offset=*/100));

    auto dlp_groups = cloneGroups(base_groups);
    auto ref_groups = cloneGroups(base_groups);

    // Add PostOps manually (RELU)
    // Attach PostOps to all groups
    for (auto& group : dlp_groups) {
        group.post_op_params.push_back(createRelu().build());
    }
    for (auto& group : ref_groups) {
        group.post_op_params.push_back(createRelu().build());
    }

    auto ual_dlp = UalFactory::createUal(UALType::DLP);
    auto ual_ref = UalFactory::createUal(UALType::REF);
    ASSERT_NE(ual_dlp, nullptr);
    ASSERT_NE(ual_ref, nullptr);

    auto status_dlp = ual_dlp->batch_gemm(dlp_groups, MatrixType::f32);
    ASSERT_EQ(status_dlp, UALError::UAL_SUCCESS)
        << "DLP batch GEMM with PostOps should succeed";

    auto status_ref = ual_ref->batch_gemm(ref_groups, MatrixType::f32);
    ASSERT_EQ(status_ref, UALError::UAL_SUCCESS)
        << "REF batch GEMM with PostOps should succeed";

    // Compare results (with PostOps tolerance)
    ASSERT_EQ(dlp_groups.size(), ref_groups.size());
    for (std::size_t g = 0; g < dlp_groups.size(); ++g) {
        const auto& dlp_group = dlp_groups[g];
        const auto& ref_group = ref_groups[g];

        ASSERT_EQ(dlp_group.C_matrices.size(), ref_group.C_matrices.size());

        for (std::size_t i = 0; i < dlp_group.C_matrices.size(); ++i) {
            dlp_group.C_matrices[i].setK(dlp_group.k);
            ref_group.C_matrices[i].setK(ref_group.k);

            MatrixCompareOptions opts = MatrixCompareOptions::Fast();
            opts.relToleranceOverride = 100.0;
            opts.absToleranceOverride = 100.0;

            auto result =
                dlp_group.C_matrices[i].compare(ref_group.C_matrices[i], opts);
            EXPECT_TRUE(result.equal)
                << "PostOps GEMM comparison failed"
                << FormatCompareResult(result, dlp_group.C_matrices[i],
                                       ref_group.C_matrices[i]);
        }
    }
}

TEST(BatchGemmTest, GlobalPostOpsMultipleGroups)
{
    // Create multiple groups with different dimensions
    std::vector<BatchGroup> base_groups;
    base_groups.push_back(
        makeF32Group(/*m=*/8, /*n=*/8, /*k=*/8, /*matrix_count=*/2,
                     /*alpha=*/1.0, /*beta=*/0.0, /*seed_offset=*/200));
    base_groups.push_back(
        makeF32Group(/*m=*/16, /*n=*/16, /*k=*/16, /*matrix_count=*/3,
                     /*alpha=*/1.0, /*beta=*/0.0, /*seed_offset=*/300));

    auto dlp_groups = cloneGroups(base_groups);
    auto ref_groups = cloneGroups(base_groups);

    // Per-group bias vectors sized to match each group's n dimension
    auto bias_8 = Matrix::fromVector(
        std::vector<float>{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f });
    auto                 bias_16        = Matrix::fromVector(std::vector<float>{
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f, 16.0f });
    std::vector<Matrix*> bias_per_group = { &bias_8, &bias_16 };

    // Attach PostOps to each group with matching bias dimension
    for (std::size_t g = 0; g < dlp_groups.size(); ++g) {
        dlp_groups[g].post_op_params.push_back(createRelu().build());
        dlp_groups[g].post_op_params.push_back(
            createBias().setBias(*bias_per_group[g]).build());
    }
    for (std::size_t g = 0; g < ref_groups.size(); ++g) {
        ref_groups[g].post_op_params.push_back(createRelu().build());
        ref_groups[g].post_op_params.push_back(
            createBias().setBias(*bias_per_group[g]).build());
    }

    auto ual_dlp = UalFactory::createUal(UALType::DLP);
    auto ual_ref = UalFactory::createUal(UALType::REF);
    ASSERT_NE(ual_dlp, nullptr);
    ASSERT_NE(ual_ref, nullptr);

    auto status_dlp = ual_dlp->batch_gemm(dlp_groups, MatrixType::f32);
    ASSERT_EQ(status_dlp, UALError::UAL_SUCCESS)
        << "DLP batch GEMM with multi-group PostOps should succeed";

    auto status_ref = ual_ref->batch_gemm(ref_groups, MatrixType::f32);
    ASSERT_EQ(status_ref, UALError::UAL_SUCCESS)
        << "REF batch GEMM with multi-group PostOps should succeed";

    // Compare results with higher tolerance for PostOps
    ASSERT_EQ(dlp_groups.size(), ref_groups.size());
    for (std::size_t g = 0; g < dlp_groups.size(); ++g) {
        const auto& dlp_group = dlp_groups[g];
        const auto& ref_group = ref_groups[g];

        ASSERT_EQ(dlp_group.C_matrices.size(), ref_group.C_matrices.size());

        for (std::size_t i = 0; i < dlp_group.C_matrices.size(); ++i) {
            dlp_group.C_matrices[i].setK(dlp_group.k);
            ref_group.C_matrices[i].setK(ref_group.k);

            MatrixCompareOptions opts = MatrixCompareOptions::Fast();
            opts.relToleranceOverride = 150.0;
            opts.absToleranceOverride = 150.0;

            auto result =
                dlp_group.C_matrices[i].compare(ref_group.C_matrices[i], opts);
            EXPECT_TRUE(result.equal)
                << "PostOps multi-group GEMM comparison failed"
                << FormatCompareResult(result, dlp_group.C_matrices[i],
                                       ref_group.C_matrices[i]);
        }
    }
}

// When a group fails parameter validation the batch aborts before the later
// groups are computed. Callers routinely zero-initialise their metadata and
// DLP_CLSC_SUCCESS is zero, so a group the routine never processes must be
// left flagged as failed rather than inheriting an implicit success over
// output that was never written. This drives the C API directly because the
// per-group error codes are not observable through the UAL wrapper.
TEST(BatchGemmTest, SkippedGroupsReportFailureNotSuccess)
{
    constexpr int   group_count   = 4;
    constexpr int   dim           = 8; // square M = N = K
    constexpr int   invalid_group = 1; // made invalid via lda = 0
    constexpr float sentinel      = -999.0f;
    constexpr float expected_c0   = static_cast<float>(dim); // ones·ones over K

    // One matrix per group. A and B are all-ones so a correctly computed
    // C[0] equals K (== dim); C starts at a sentinel so untouched output is
    // recognisable.
    std::vector<std::vector<float>> a_data(group_count,
                                           std::vector<float>(dim * dim, 1.0f));
    std::vector<std::vector<float>> b_data(group_count,
                                           std::vector<float>(dim * dim, 1.0f));
    std::vector<std::vector<float>> c_data(
        group_count, std::vector<float>(dim * dim, sentinel));

    std::vector<char> order(group_count, 'r');
    std::vector<char> transa(group_count, 'n');
    std::vector<char> transb(group_count, 'n');
    std::vector<char> mem_format_a(group_count, 'n');
    std::vector<char> mem_format_b(group_count, 'n');

    std::vector<md_t> m(group_count, dim);
    std::vector<md_t> n(group_count, dim);
    std::vector<md_t> k(group_count, dim);
    std::vector<md_t> lda(group_count, dim);
    std::vector<md_t> ldb(group_count, dim);
    std::vector<md_t> ldc(group_count, dim);
    std::vector<md_t> group_size(group_count, 1);

    std::vector<float> alpha(group_count, 1.0f);
    std::vector<float> beta(group_count, 0.0f);

    std::vector<const float*> a_ptrs(group_count);
    std::vector<const float*> b_ptrs(group_count);
    std::vector<float*>       c_ptrs(group_count);

    std::vector<dlp_metadata_t>  metadata(group_count);
    std::vector<dlp_metadata_t*> metadata_ptrs(group_count);

    for (int g = 0; g < group_count; ++g) {
        a_ptrs[g] = a_data[g].data();
        b_ptrs[g] = b_data[g].data();
        c_ptrs[g] = c_data[g].data();

        // Zero-init mirrors the common caller pattern where an unset error
        // code equals DLP_CLSC_SUCCESS.
        std::memset(&metadata[g], 0, sizeof(dlp_metadata_t));
        metadata_ptrs[g] = &metadata[g];
    }

    // Invalidate a middle group: lda < k fails the leading-dimension check and
    // aborts the shared validate/compute loop before the later groups run.
    lda[invalid_group] = 0;

    aocl_batch_gemm_f32f32f32of32(
        order.data(), transa.data(), transb.data(), m.data(), n.data(),
        k.data(), alpha.data(), a_ptrs.data(), lda.data(), b_ptrs.data(),
        ldb.data(), beta.data(), c_ptrs.data(), ldc.data(), group_count,
        group_size.data(), mem_format_a.data(), mem_format_b.data(),
        metadata_ptrs.data());

    if (metadata[0].error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED) {
        GTEST_SKIP() << "F32 batch GEMM not supported on this processor "
                        "(missing AVX2)";
    }

    // The group before the failure is computed and reports success.
    EXPECT_EQ(metadata[0].error_hndl.error_code, DLP_CLSC_SUCCESS);
    EXPECT_EQ(c_data[0][0], expected_c0);
    // The invalid group reports its specific validation error.
    EXPECT_NE(metadata[invalid_group].error_hndl.error_code, DLP_CLSC_SUCCESS);
    EXPECT_EQ(metadata[invalid_group].error_hndl.error_code,
              DLP_CLSC_INVALID_LEADING_DIMENSION);

    // Groups after the failure are never processed: they must report failure
    // rather than an implicit success, and their output must be untouched.
    for (int g = invalid_group + 1; g < group_count; ++g) {
        EXPECT_EQ(metadata[g].error_hndl.error_code, DLP_CLSC_FAILURE)
            << "skipped group " << g << " must not report success";
        EXPECT_EQ(c_data[g][0], sentinel)
            << "skipped group " << g << " output was unexpectedly written";
    }
}

template<typename AType, typename BatchFn>
void
verifyInt8SkippedGroupsReportFailure(BatchFn batch_gemm)
{
    constexpr int     group_count   = 4;
    constexpr int     dim           = 8;
    constexpr int     invalid_group = 1;
    constexpr int32_t sentinel      = -999;

    std::vector<std::vector<AType>> a_data(
        group_count, std::vector<AType>(dim * dim, static_cast<AType>(1)));
    std::vector<std::vector<int8_t>>  b_data(group_count,
                                             std::vector<int8_t>(dim * dim, 1));
    std::vector<std::vector<int32_t>> c_data(
        group_count, std::vector<int32_t>(dim * dim, sentinel));

    std::vector<char> order(group_count, 'r');
    std::vector<char> transa(group_count, 'n');
    std::vector<char> transb(group_count, 'n');
    std::vector<char> mem_format_a(group_count, 'n');
    std::vector<char> mem_format_b(group_count, 'n');

    std::vector<md_t> m(group_count, dim);
    std::vector<md_t> n(group_count, dim);
    std::vector<md_t> k(group_count, dim);
    std::vector<md_t> lda(group_count, dim);
    std::vector<md_t> ldb(group_count, dim);
    std::vector<md_t> ldc(group_count, dim);
    std::vector<md_t> group_size(group_count, 1);

    std::vector<int32_t> alpha(group_count, 1);
    std::vector<int32_t> beta(group_count, 0);

    std::vector<const AType*>  a_ptrs(group_count);
    std::vector<const int8_t*> b_ptrs(group_count);
    std::vector<int32_t*>      c_ptrs(group_count);

    std::vector<dlp_metadata_t>  metadata(group_count);
    std::vector<dlp_metadata_t*> metadata_ptrs(group_count);

    for (int g = 0; g < group_count; ++g) {
        a_ptrs[g] = a_data[g].data();
        b_ptrs[g] = b_data[g].data();
        c_ptrs[g] = c_data[g].data();
        std::memset(&metadata[g], 0, sizeof(dlp_metadata_t));
        metadata_ptrs[g] = &metadata[g];
    }

    lda[invalid_group] = 0;

    batch_gemm(order.data(), transa.data(), transb.data(), m.data(), n.data(),
               k.data(), alpha.data(), a_ptrs.data(), lda.data(), b_ptrs.data(),
               ldb.data(), beta.data(), c_ptrs.data(), ldc.data(), group_count,
               group_size.data(), mem_format_a.data(), mem_format_b.data(),
               metadata_ptrs.data());

    if (metadata[0].error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED) {
        GTEST_SKIP() << "INT8 batch GEMM requires AVX512-VNNI";
    }

    EXPECT_EQ(metadata[0].error_hndl.error_code, DLP_CLSC_SUCCESS);
    EXPECT_EQ(c_data[0][0], dim);
    EXPECT_EQ(metadata[invalid_group].error_hndl.error_code,
              DLP_CLSC_INVALID_LEADING_DIMENSION);
    for (int g = invalid_group + 1; g < group_count; ++g) {
        EXPECT_EQ(metadata[g].error_hndl.error_code, DLP_CLSC_FAILURE)
            << "skipped group " << g << " must not report success";
        EXPECT_EQ(c_data[g][0], sentinel)
            << "skipped group " << g << " output was unexpectedly written";
    }
}

TEST(BatchGemmTest, U8S8SkippedGroupsReportFailureNotSuccess)
{
    verifyInt8SkippedGroupsReportFailure<uint8_t>(aocl_batch_gemm_u8s8s32os32);
}

TEST(BatchGemmTest, S8S8SkippedGroupsReportFailureNotSuccess)
{
    verifyInt8SkippedGroupsReportFailure<int8_t>(aocl_batch_gemm_s8s8s32os32);
}

// ============================================================================
// INPUT VALIDATION (NEGATIVE PATHS)
// ============================================================================

/*
 * Negative-path coverage for the batch entry points. These drive the C API
 * directly because the per-group error codes are not observable through the UAL
 * wrapper, and because the reject path is not reachable from the YAML configs.
 *
 * Both axes are table-driven, so covering a new entry point or a new invalid
 * input is a single line:
 *
 *   BatchApiValidation   every entry point x every invalid parameter value
 *   BatchNullArrayArgs   the array arguments themselves being NULL (CPUPL-9033)
 *   BatchGroupCount      the by-value group_count bound
 *
 * Per-group error reporting is already pinned by the skipped-group tests above
 * and is not repeated.
 */

namespace validation {

constexpr md_t kM = 4, kN = 4, kK = 4;
constexpr md_t kOverMax = static_cast<md_t>(INT32_MAX) + 100;

/*
 * Which top-level array argument to pass as NULL. The batch APIs take every
 * per-group parameter as an array and index all of them (order[i],
 * group_size[i], metadata[i], ...), so each array has to be rejected at entry
 * rather than dereferenced.
 */
enum class NullArg
{
    None,
    Order,
    Transa,
    Transb,
    M,
    N,
    K,
    Alpha,
    A,
    Lda,
    B,
    Ldb,
    Beta,
    C,
    Ldc,
    GroupSize,
    MemFormatA,
    MemFormatB,
    Metadata
};

// Defaults describe a valid single-group batch; each case perturbs one field.
struct Knobs
{
    char order = 'r', transa = 'n', transb = 'n';
    md_t m = kM, n = kN, k = kK;
    md_t lda = kK, ldb = kN, ldc = kN;
    char mem_format_a = 'n', mem_format_b = 'n';
    md_t group_size = 1, group_count = 1;
    // A NULL matrix within an otherwise valid array, as opposed to null_arg
    // below which nulls the array itself.
    bool    null_a = false, null_b = false, null_c = false;
    NullArg null_arg = NullArg::None;
    // Pre-set on the metadata before the call. A sentinel no validator produces
    // makes "never written" detectable.
    dlp_clsc_err_t initial_code = DLP_CLSC_SUCCESS;
};

// The one shape every aocl_batch_gemm_* entry point has.
template<typename AType, typename BType, typename CType, typename ScalarT>
using BatchGemmFn = void (*)(const char*,
                             const char*,
                             const char*,
                             const md_t*,
                             const md_t*,
                             const md_t*,
                             const ScalarT*,
                             const AType**,
                             const md_t*,
                             const BType**,
                             const md_t*,
                             const ScalarT*,
                             CType**,
                             const md_t*,
                             md_t,
                             const md_t*,
                             const char*,
                             const char*,
                             dlp_metadata_t**);

// Passes ptr, or NULL when this case is the one nulling that argument.
template<typename T>
T*
arg_or_null(const Knobs& kn, NullArg which, T* ptr)
{
    return kn.null_arg == which ? nullptr : ptr;
}

/*
 * Runs a single-group batch call and returns the code left on that group's
 * metadata. Buffers are sized generously so a buffer bound is never what makes
 * a case fail: the only thing under test is the validator.
 */
template<typename AType, typename BType, typename CType, typename ScalarT>
dlp_clsc_err_t
invoke_batch(BatchGemmFn<AType, BType, CType, ScalarT> fn, const Knobs& kn)
{
    std::vector<AType> a_buf(4096, AType{});
    std::vector<BType> b_buf(4096, BType{});
    std::vector<CType> c_buf(4096, CType{});

    const AType* ap = kn.null_a ? nullptr : a_buf.data();
    const BType* bp = kn.null_b ? nullptr : b_buf.data();
    CType*       cp = kn.null_c ? nullptr : c_buf.data();

    ScalarT alpha = static_cast<ScalarT>(1);
    ScalarT beta  = static_cast<ScalarT>(0);

    dlp_metadata_t md;
    std::memset(&md, 0, sizeof(md));
    md.error_hndl.error_code = kn.initial_code;
    dlp_metadata_t* mdp      = &md;

    fn(arg_or_null(kn, NullArg::Order, &kn.order),
       arg_or_null(kn, NullArg::Transa, &kn.transa),
       arg_or_null(kn, NullArg::Transb, &kn.transb),
       arg_or_null(kn, NullArg::M, &kn.m), arg_or_null(kn, NullArg::N, &kn.n),
       arg_or_null(kn, NullArg::K, &kn.k),
       arg_or_null(kn, NullArg::Alpha, &alpha),
       arg_or_null(kn, NullArg::A, &ap), arg_or_null(kn, NullArg::Lda, &kn.lda),
       arg_or_null(kn, NullArg::B, &bp), arg_or_null(kn, NullArg::Ldb, &kn.ldb),
       arg_or_null(kn, NullArg::Beta, &beta), arg_or_null(kn, NullArg::C, &cp),
       arg_or_null(kn, NullArg::Ldc, &kn.ldc), kn.group_count,
       arg_or_null(kn, NullArg::GroupSize, &kn.group_size),
       arg_or_null(kn, NullArg::MemFormatA, &kn.mem_format_a),
       arg_or_null(kn, NullArg::MemFormatB, &kn.mem_format_b),
       arg_or_null(kn, NullArg::Metadata, &mdp));

    return md.error_hndl.error_code;
}

/*
 * Runs a single-group batch with a NULL matrix pointer in the group's flat
 * range. This is intentionally a safety probe for every public batch entry
 * point; datatype-specific metadata may cause a supported API to reject the
 * call for another reason after the common pointer check. The exact cumulative
 * offset/status contract is covered by the f32-specific test below.
 */
template<typename AType, typename BType, typename CType, typename ScalarT>
dlp_clsc_err_t
invoke_batch_interior_null(BatchGemmFn<AType, BType, CType, ScalarT> fn,
                           int null_matrix)
{
    constexpr md_t        group_sizes[] = { 2 };
    constexpr std::size_t matrix_count  = 2;

    std::vector<std::vector<AType>> a_storage(
        matrix_count, std::vector<AType>(kM * kK, AType{}));
    std::vector<std::vector<BType>> b_storage(
        matrix_count, std::vector<BType>(kK * kN, BType{}));
    std::vector<std::vector<CType>> c_storage(
        matrix_count, std::vector<CType>(kM * kN, CType{}));

    const AType* a_ptrs[matrix_count];
    const BType* b_ptrs[matrix_count];
    CType*       c_ptrs[matrix_count];
    for (std::size_t i = 0; i < matrix_count; ++i) {
        a_ptrs[i] = a_storage[i].data();
        b_ptrs[i] = b_storage[i].data();
        c_ptrs[i] = c_storage[i].data();
    }

    if (null_matrix == 0) {
        a_ptrs[1] = nullptr;
    } else if (null_matrix == 1) {
        b_ptrs[1] = nullptr;
    } else {
        c_ptrs[1] = nullptr;
    }

    const char    order[2]        = { 'r', 'r' };
    const char    transa[2]       = { 'n', 'n' };
    const char    transb[2]       = { 'n', 'n' };
    const char    mem_format_a[2] = { 'n', 'n' };
    const char    mem_format_b[2] = { 'n', 'n' };
    const md_t    m[2]            = { kM, kM };
    const md_t    n[2]            = { kN, kN };
    const md_t    k[2]            = { kK, kK };
    const md_t    lda[2]          = { kK, kK };
    const md_t    ldb[2]          = { kN, kN };
    const md_t    ldc[2]          = { kN, kN };
    const ScalarT alpha[2]        = { static_cast<ScalarT>(1),
                                      static_cast<ScalarT>(1) };
    const ScalarT beta[2]         = { static_cast<ScalarT>(0),
                                      static_cast<ScalarT>(0) };

    dlp_metadata_t metadata[2];
    std::memset(metadata, 0, sizeof(metadata));
    metadata[0].error_hndl.error_code = DLP_CLSC_FAILURE;
    dlp_metadata_t* metadata_ptrs[1]  = { &metadata[0] };

    fn(order, transa, transb, m, n, k, alpha, a_ptrs, lda, b_ptrs, ldb, beta,
       c_ptrs, ldc, 1, group_sizes, mem_format_a, mem_format_b, metadata_ptrs);

    return metadata[0].error_hndl.error_code;
}

struct BatchApi
{
    const char* name;
    dlp_clsc_err_t (*invoke)(const Knobs&);
    dlp_clsc_err_t (*invoke_interior_null)(int);
};

#define DLP_BATCH_API(api, AType, BType, CType, ScalarT)                       \
    BatchApi                                                                   \
    {                                                                          \
        #api,                                                                  \
            [](const Knobs& kn) -> dlp_clsc_err_t {                            \
                return invoke_batch<AType, BType, CType, ScalarT>(             \
                    &aocl_batch_gemm_##api, kn);                               \
            },                                                                 \
            [](int null_matrix) -> dlp_clsc_err_t {                            \
                return invoke_batch_interior_null<AType, BType, CType,         \
                                                  ScalarT>(                    \
                    &aocl_batch_gemm_##api, null_matrix);                      \
            }                                                                  \
    }

// All 32 public batch entry points.
std::vector<BatchApi>
all_batch_apis()
{
    return {
        DLP_BATCH_API(f32f32f32of32, float, float, float, float),

        DLP_BATCH_API(bf16bf16f32of32, bfloat16, bfloat16, float, float),
        DLP_BATCH_API(bf16bf16f32obf16, bfloat16, bfloat16, bfloat16, float),

        DLP_BATCH_API(bf16s4f32of32, bfloat16, int8_t, float, float),
        DLP_BATCH_API(bf16s4f32obf16, bfloat16, int8_t, bfloat16, float),
        DLP_BATCH_API(bf16u4f32of32, bfloat16, uint8_t, float, float),
        DLP_BATCH_API(bf16u4f32obf16, bfloat16, uint8_t, bfloat16, float),

        DLP_BATCH_API(bf16s8s32os32, bfloat16, int8_t, int32_t, int32_t),
        DLP_BATCH_API(bf16s8s32os8, bfloat16, int8_t, int8_t, int32_t),
        DLP_BATCH_API(bf16s8s32ou8, bfloat16, int8_t, uint8_t, int32_t),
        DLP_BATCH_API(bf16s8s32of32, bfloat16, int8_t, float, int32_t),
        DLP_BATCH_API(bf16s8s32obf16, bfloat16, int8_t, bfloat16, int32_t),

        DLP_BATCH_API(u8s8s32os32, uint8_t, int8_t, int32_t, int32_t),
        DLP_BATCH_API(u8s8s32os8, uint8_t, int8_t, int8_t, int32_t),
        DLP_BATCH_API(u8s8s32ou8, uint8_t, int8_t, uint8_t, int32_t),
        DLP_BATCH_API(u8s8s32of32, uint8_t, int8_t, float, int32_t),
        DLP_BATCH_API(u8s8s32obf16, uint8_t, int8_t, bfloat16, int32_t),

        DLP_BATCH_API(s8s8s32os32, int8_t, int8_t, int32_t, int32_t),
        DLP_BATCH_API(s8s8s32os8, int8_t, int8_t, int8_t, int32_t),
        DLP_BATCH_API(s8s8s32ou8, int8_t, int8_t, uint8_t, int32_t),
        DLP_BATCH_API(s8s8s32of32, int8_t, int8_t, float, int32_t),
        DLP_BATCH_API(s8s8s32obf16, int8_t, int8_t, bfloat16, int32_t),

        DLP_BATCH_API(s8s8s32of32_sym_quant, int8_t, int8_t, float, int32_t),
        DLP_BATCH_API(s8s8s32obf16_sym_quant, int8_t, int8_t, bfloat16,
                      int32_t),

        DLP_BATCH_API(f32s8s32os32, float, int8_t, int32_t, int32_t),
        DLP_BATCH_API(f32s8s32os8, float, int8_t, int8_t, int32_t),
        DLP_BATCH_API(f32s8s32ou8, float, int8_t, uint8_t, int32_t),
        DLP_BATCH_API(f32s8s32of32, float, int8_t, float, int32_t),
        DLP_BATCH_API(f32s8s32obf16, float, int8_t, bfloat16, int32_t),

        DLP_BATCH_API(f16f16f16of16, float16, float16, float16, float16),
        DLP_BATCH_API(f16f16f16of32, float16, float16, float, float16),
        DLP_BATCH_API(f32f16f32of32, float, float16, float, float),
    };
}

struct Scenario
{
    const char*    name;
    Knobs          knobs;
    dlp_clsc_err_t expected;
};

std::vector<Scenario>
all_scenarios()
{
    std::vector<Scenario> s;
    auto                  add = [&s](const char* name, dlp_clsc_err_t expected,
                    void (*mutate)(Knobs&)) {
        Knobs k;
        mutate(k);
        s.push_back({ name, k, expected });
    };

    add("NullA", DLP_CLSC_NULL_POINTER, [](Knobs& k) { k.null_a = true; });
    add("NullB", DLP_CLSC_NULL_POINTER, [](Knobs& k) { k.null_b = true; });
    add("NullC", DLP_CLSC_NULL_POINTER, [](Knobs& k) { k.null_c = true; });

    add("InvalidOrder", DLP_CLSC_INVALID_ORDER,
        [](Knobs& k) { k.order = 'x'; });
    add("InvalidTransA", DLP_CLSC_INVALID_TRANSPOSE,
        [](Knobs& k) { k.transa = 'x'; });
    add("InvalidTransB", DLP_CLSC_INVALID_TRANSPOSE,
        [](Knobs& k) { k.transb = 'x'; });

    add("InvalidMemFormatA", DLP_CLSC_INVALID_MEMORY_TAG,
        [](Knobs& k) { k.mem_format_a = 'x'; });
    add("InvalidMemFormatB", DLP_CLSC_INVALID_MEMORY_TAG,
        [](Knobs& k) { k.mem_format_b = 'x'; });

    add("ZeroM", DLP_CLSC_INVALID_MATRIX_DIMENSION, [](Knobs& k) { k.m = 0; });
    add("ZeroN", DLP_CLSC_INVALID_MATRIX_DIMENSION, [](Knobs& k) { k.n = 0; });
    add("ZeroK", DLP_CLSC_INVALID_MATRIX_DIMENSION, [](Knobs& k) { k.k = 0; });
    add("NegativeM", DLP_CLSC_INVALID_MATRIX_DIMENSION,
        [](Knobs& k) { k.m = -1; });
    add("OverMaxN", DLP_CLSC_INVALID_MATRIX_DIMENSION,
        [](Knobs& k) { k.n = kOverMax; });

    add("LdaBelowMin", DLP_CLSC_INVALID_LEADING_DIMENSION,
        [](Knobs& k) { k.lda = kK - 1; });
    add("LdbBelowMin", DLP_CLSC_INVALID_LEADING_DIMENSION,
        [](Knobs& k) { k.ldb = kN - 1; });
    add("LdcBelowMin", DLP_CLSC_INVALID_LEADING_DIMENSION,
        [](Knobs& k) { k.ldc = kN - 1; });
    add("LdcOverMax", DLP_CLSC_INVALID_LEADING_DIMENSION,
        [](Knobs& k) { k.ldc = kOverMax; });

    // Batch-only: the per-group size guard. No single-GEMM equivalent exists.
    add("ZeroGroupSize", DLP_CLSC_INVALID_GROUP_DIMENSION,
        [](Knobs& k) { k.group_size = 0; });
    add("NegativeGroupSize", DLP_CLSC_INVALID_GROUP_DIMENSION,
        [](Knobs& k) { k.group_size = -1; });
    add("OverMaxGroupSize", DLP_CLSC_INVALID_GROUP_DIMENSION,
        [](Knobs& k) { k.group_size = kOverMax; });

    return s;
}

// Outcome of running a probe in a forked child.
struct ChildResult
{
    int signal    = 0;
    int exit_code = -1;
};

/*
 * Runs `probe` in a forked child and reports how the child terminated, exiting
 * with the value `probe` returns so a caller can assert on an error code and on
 * survival in one test. Used where the library might dereference an argument
 * before validating it: a fault stays in the child and becomes a normal
 * assertion failure instead of taking the whole binary down and masking every
 * later case. fork() is used directly because death-test support is not enabled
 * in this build's GoogleTest configuration.
 */
template<typename Fn>
ChildResult
run_isolated(Fn&& probe)
{
#if defined(_WIN32)
    ChildResult result;
    result.exit_code = probe();
    return result;
#else
    const pid_t pid = fork();
    if (pid == 0) {
        _exit(probe());
    }

    ChildResult result;
    if (pid < 0) {
        return result; // fork failed; reported by the caller's assertion
    }

    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        result.signal = WTERMSIG(status);
    }
    return result;
#endif
}

} // namespace validation

using namespace validation;

class BatchInteriorNullEveryApi : public ::testing::TestWithParam<BatchApi>
{};

TEST_P(BatchInteriorNullEveryApi, RejectsInteriorNullWithoutCrashing)
{
#if defined(DLP_ENABLE_OPENMP)
    GTEST_SKIP() << "Interior-NULL fork probes are run in the non-OpenMP build";
#endif

    const BatchApi& api = GetParam();

    for (int null_matrix = 0; null_matrix < 3; ++null_matrix) {
        const ChildResult r = run_isolated([&api, null_matrix] {
            return static_cast<int>(api.invoke_interior_null(null_matrix));
        });

        ASSERT_EQ(r.signal, 0)
            << "aocl_batch_gemm_" << api.name << " crashed on an interior "
            << (null_matrix == 0 ? "A" : (null_matrix == 1 ? "B" : "C"))
            << " NULL pointer";

        if (r.exit_code == DLP_CLSC_NOT_SUPPORTED) {
            continue;
        }

        EXPECT_EQ(r.exit_code, DLP_CLSC_NULL_POINTER)
            << "aocl_batch_gemm_" << api.name
            << " did not report DLP_CLSC_NULL_POINTER for an interior "
               "NULL pointer in "
            << (null_matrix == 0 ? "A" : (null_matrix == 1 ? "B" : "C"));
    }
}

INSTANTIATE_TEST_SUITE_P(AllApis,
                         BatchInteriorNullEveryApi,
                         ::testing::ValuesIn(all_batch_apis()),
                         [](const ::testing::TestParamInfo<BatchApi>& info) {
                             return std::string(info.param.name);
                         });

TEST(BatchGemmTest, RejectsInteriorNullMatrixPointers)
{
    enum class NullMatrix
    {
        A,
        B,
        C
    };

    const md_t m = 2, n = 2, k = 2;

    auto run = [&](NullMatrix null_matrix, const std::vector<md_t>& group_sizes,
                   std::size_t null_index) {
        std::size_t matrix_count = 0;
        for (const md_t group_size : group_sizes) {
            matrix_count += static_cast<std::size_t>(group_size);
        }

        std::vector<std::vector<float>> a_storage(
            matrix_count, std::vector<float>(m * k, 1.0f));
        std::vector<std::vector<float>> b_storage(
            matrix_count, std::vector<float>(k * n, 1.0f));
        std::vector<std::vector<float>> c_storage(
            matrix_count, std::vector<float>(m * n, 0.0f));

        std::vector<const float*> a_ptrs(matrix_count);
        std::vector<const float*> b_ptrs(matrix_count);
        std::vector<float*>       c_ptrs(matrix_count);
        for (std::size_t i = 0; i < matrix_count; ++i) {
            a_ptrs[i] = a_storage[i].data();
            b_ptrs[i] = b_storage[i].data();
            c_ptrs[i] = c_storage[i].data();
        }

        if (null_matrix == NullMatrix::A) {
            a_ptrs[null_index] = nullptr;
        } else if (null_matrix == NullMatrix::B) {
            b_ptrs[null_index] = nullptr;
        } else {
            c_ptrs[null_index] = nullptr;
        }

        const std::size_t group_count = group_sizes.size();
        std::vector<char> order(group_count, 'r');
        std::vector<char> transa(group_count, 'n');
        std::vector<char> transb(group_count, 'n');
        std::vector<char> mem_format_a(group_count, 'n');
        std::vector<char> mem_format_b(group_count, 'n');

        std::vector<md_t>  m_values(group_count, m);
        std::vector<md_t>  n_values(group_count, n);
        std::vector<md_t>  k_values(group_count, k);
        std::vector<md_t>  lda(group_count, k);
        std::vector<md_t>  ldb(group_count, n);
        std::vector<md_t>  ldc(group_count, n);
        std::vector<float> alpha(group_count, 1.0f);
        std::vector<float> beta(group_count, 0.0f);

        std::vector<dlp_metadata_t>  metadata(group_count);
        std::vector<dlp_metadata_t*> metadata_ptrs(group_count);
        for (std::size_t i = 0; i < group_count; ++i) {
            std::memset(&metadata[i], 0, sizeof(dlp_metadata_t));
            metadata_ptrs[i] = &metadata[i];
        }

        aocl_batch_gemm_f32f32f32of32(
            order.data(), transa.data(), transb.data(), m_values.data(),
            n_values.data(), k_values.data(), alpha.data(), a_ptrs.data(),
            lda.data(), b_ptrs.data(), ldb.data(), beta.data(), c_ptrs.data(),
            ldc.data(), static_cast<md_t>(group_count), group_sizes.data(),
            mem_format_a.data(), mem_format_b.data(), metadata_ptrs.data());

        std::vector<dlp_clsc_err_t> statuses;
        statuses.reserve(group_count);
        for (const auto& md : metadata) {
            statuses.push_back(md.error_hndl.error_code);
        }
        return statuses;
    };

    // Keep the child exit status small enough for POSIX wait-status encoding:
    // SUCCESS=0, NULL_POINTER=1, NOT_SUPPORTED=2, all other errors=3.
    const auto compact_status = [](dlp_clsc_err_t status) {
        if (status == DLP_CLSC_SUCCESS) {
            return 0;
        }
        if (status == DLP_CLSC_NULL_POINTER) {
            return 1;
        }
        if (status == DLP_CLSC_NOT_SUPPORTED) {
            return 2;
        }
        return 3;
    };

    const auto run_isolated_case = [&](NullMatrix               null_matrix,
                                       const std::vector<md_t>& group_sizes,
                                       std::size_t              null_index) {
        return run_isolated([&] {
            const auto statuses = run(null_matrix, group_sizes, null_index);
            int        result   = compact_status(statuses[0]);
            if (statuses.size() > 1) {
                result |= compact_status(statuses[1]) << 2;
            }
            return result;
        });
    };

    for (const auto null_matrix :
         { NullMatrix::A, NullMatrix::B, NullMatrix::C }) {
        const char* null_name =
            null_matrix == NullMatrix::A
                ? "A"
                : (null_matrix == NullMatrix::B ? "B" : "C");
        SCOPED_TRACE(null_name);

        // The group-indexed checks see slot 0, but slot 1 is consumed by the
        // group and must also be validated.
        const ChildResult single_group =
            run_isolated_case(null_matrix, { 2 }, 1);
        ASSERT_EQ(single_group.signal, 0)
            << "interior NULL in " << null_name
            << " crashed the batch GEMM test process";
        if (single_group.exit_code == 2) {
            GTEST_SKIP() << "F32 batch GEMM is not supported on this processor";
        }
        ASSERT_EQ(single_group.exit_code, 1)
            << "interior NULL in the single-group " << null_name
            << " group was not rejected";
    }

#if defined(DLP_ENABLE_OPENMP)
    GTEST_SKIP()
        << "Cumulative interior-NULL fork probes are run in the non-OpenMP "
           "build";
#else
    for (const auto null_matrix :
         { NullMatrix::A, NullMatrix::B, NullMatrix::C }) {
        const char* null_name =
            null_matrix == NullMatrix::A
                ? "A"
                : (null_matrix == NullMatrix::B ? "B" : "C");
        SCOPED_TRACE(null_name);

        // The second group's first matrix is at cumulative flat index 2;
        // validating only a[gc_i]/b[gc_i]/c[gc_i] would inspect slot 1.
        const ChildResult multiple_groups =
            run_isolated_case(null_matrix, { 2, 1 }, 2);
        ASSERT_EQ(multiple_groups.signal, 0)
            << "cumulative interior NULL in batch GEMM crashed the test "
               "process";
        EXPECT_EQ(multiple_groups.exit_code & 0x3, 0)
            << "the valid first group did not complete successfully";
        EXPECT_EQ((multiple_groups.exit_code >> 2) & 0x3, 1)
            << "the second group did not report DLP_CLSC_NULL_POINTER";
    }
#endif
}

// --- every entry point x every invalid parameter
// ------------------------------

using ApiScenario = std::tuple<BatchApi, Scenario>;

class BatchApiValidation : public ::testing::TestWithParam<ApiScenario>
{};

TEST_P(BatchApiValidation, RejectsInvalidInput)
{
    const auto& api      = std::get<0>(GetParam());
    const auto& scenario = std::get<1>(GetParam());

    const dlp_clsc_err_t got = api.invoke(scenario.knobs);

    if (got == DLP_CLSC_NOT_SUPPORTED) {
        GTEST_SKIP() << api.name
                     << " is not supported on this processor; the validation "
                        "result is unobservable";
    }

    EXPECT_EQ(got, scenario.expected)
        << "aocl_batch_gemm_" << api.name << " did not reject scenario '"
        << scenario.name << "' with the documented error code. Expected "
        << static_cast<int>(scenario.expected) << ", got "
        << static_cast<int>(got)
        << ". DLP_CLSC_SUCCESS means the bad input reached the kernel; "
           "DLP_CLSC_FAILURE means the group was left in its default failed "
           "state without the specific reason being recorded.";
}

INSTANTIATE_TEST_SUITE_P(AllApis,
                         BatchApiValidation,
                         ::testing::ValuesIn([] {
                             std::vector<ApiScenario> out;
                             for (const auto& api : all_batch_apis()) {
                                 for (const auto& sc : all_scenarios()) {
                                     out.emplace_back(api, sc);
                                 }
                             }
                             return out;
                         }()),
                         [](const ::testing::TestParamInfo<ApiScenario>& info) {
                             return std::string(std::get<0>(info.param).name)
                                    + "_" + std::get<1>(info.param).name;
                         });

// --- NULL array arguments (CPUPL-9033) ---------------------------------------

/*
 * AOCL_DLP_BATCH_GEMM_NULL_ARGS_CHECK validates the array arguments at function
 * entry. AOCL_DLP_BATCH_GEMM_CHECK cannot cover this: it is handed the
 * already-indexed values (order[gc_i], a[gc_i]), so on a NULL array the fault
 * happens while its arguments are evaluated, before its body runs.
 *
 * The guard sits ahead of the ISA check, so these cases report
 * DLP_CLSC_NULL_POINTER on every processor and never skip.
 */
struct NullArgCase
{
    const char* name;
    NullArg     arg;
    // The metadata array is the one argument with nowhere to record an error,
    // so its contract is a clean return rather than a reported code.
    bool reports_code;
};

class BatchNullArrayArgs : public ::testing::TestWithParam<NullArgCase>
{};

TEST_P(BatchNullArrayArgs, RejectedAtEntry)
{
    const NullArgCase& c = GetParam();

    const ChildResult r = run_isolated([&c] {
        Knobs kn;
        kn.null_arg = c.arg;
        return static_cast<int>(invoke_batch<float, float, float, float>(
            &aocl_batch_gemm_f32f32f32of32, kn));
    });

    ASSERT_EQ(r.signal, 0) << "aocl_batch_gemm_f32f32f32of32 died (signal "
                           << r.signal << ") on a NULL " << c.name
                           << " array instead of returning";

    if (c.reports_code) {
        EXPECT_EQ(r.exit_code, static_cast<int>(DLP_CLSC_NULL_POINTER))
            << "a NULL " << c.name
            << " array was not reported as DLP_CLSC_NULL_POINTER";
    }
}

INSTANTIATE_TEST_SUITE_P(EveryArrayArg,
                         BatchNullArrayArgs,
                         ::testing::ValuesIn(std::vector<NullArgCase>{
                             { "order", NullArg::Order, true },
                             { "transa", NullArg::Transa, true },
                             { "transb", NullArg::Transb, true },
                             { "m", NullArg::M, true },
                             { "n", NullArg::N, true },
                             { "k", NullArg::K, true },
                             { "alpha", NullArg::Alpha, true },
                             { "a", NullArg::A, true },
                             { "lda", NullArg::Lda, true },
                             { "b", NullArg::B, true },
                             { "ldb", NullArg::Ldb, true },
                             { "beta", NullArg::Beta, true },
                             { "c", NullArg::C, true },
                             { "ldc", NullArg::Ldc, true },
                             { "group_size", NullArg::GroupSize, true },
                             { "mem_format_a", NullArg::MemFormatA, true },
                             { "mem_format_b", NullArg::MemFormatB, true },
                             { "metadata", NullArg::Metadata, false },
                         }),
                         [](const ::testing::TestParamInfo<NullArgCase>& info) {
                             return std::string(info.param.name);
                         });

/*
 * The guard is one shared macro, but it has to be invoked in each entry point,
 * and a missed insertion is a live crash rather than a wrong error code. This
 * drives every entry point with a NULL A array to prove the call site is there.
 */
class BatchNullArrayEveryApi : public ::testing::TestWithParam<BatchApi>
{};

TEST_P(BatchNullArrayEveryApi, GuardIsPresent)
{
    const BatchApi& api = GetParam();

    const ChildResult r = run_isolated([&api] {
        Knobs kn;
        kn.null_arg = NullArg::A;
        return static_cast<int>(api.invoke(kn));
    });

    ASSERT_EQ(r.signal, 0)
        << "aocl_batch_gemm_" << api.name << " died (signal " << r.signal
        << ") on a NULL A array, so it is missing the entry-level NULL guard";
    EXPECT_EQ(r.exit_code, static_cast<int>(DLP_CLSC_NULL_POINTER))
        << "aocl_batch_gemm_" << api.name
        << " did not report a NULL A array as DLP_CLSC_NULL_POINTER";
}

INSTANTIATE_TEST_SUITE_P(AllApis,
                         BatchNullArrayEveryApi,
                         ::testing::ValuesIn(all_batch_apis()),
                         [](const ::testing::TestParamInfo<BatchApi>& info) {
                             return std::string(info.param.name);
                         });

// --- group_count -------------------------------------------------------------

/*
 * group_count is a by-value loop bound, so AOCL_DLP_BATCH_GEMM_CHECK never sees
 * it: the macro validates the loop variable, which requires the loop to already
 * be running. A count of zero or less therefore skips every loop in the
 * function, including the one that seeds the metadata, so the call does nothing
 * and reports nothing. The assertion is only that it returns, since with no
 * group to report on there is no documented error code to demand. Forked
 * because a bad bound would fault rather than return.
 */
class BatchGroupCount : public ::testing::TestWithParam<md_t>
{};

TEST_P(BatchGroupCount, NonPositiveCountReturnsWithoutCrashing)
{
    const md_t count = GetParam();

    const ChildResult r = run_isolated([count] {
        Knobs kn;
        kn.group_count = count;
        invoke_batch<float, float, float, float>(&aocl_batch_gemm_f32f32f32of32,
                                                 kn);
        return 0;
    });

    EXPECT_EQ(r.signal, 0)
        << "aocl_batch_gemm_f32f32f32of32 died on group_count=" << count
        << " (signal " << r.signal << ")";
}

INSTANTIATE_TEST_SUITE_P(NonPositive,
                         BatchGroupCount,
                         ::testing::Values(md_t{ 0 }, md_t{ -1 }),
                         [](const ::testing::TestParamInfo<md_t>& info) {
                             return info.param == 0 ? "Zero" : "Negative";
                         });

// A zero group_count means no group's metadata is ever touched, so the caller's
// error code stays exactly as it was initialised. Pinning this documents that a
// zero-count batch is silent: it reports neither success nor failure, and a
// caller that pre-set SUCCESS cannot tell the work was skipped.
TEST(BatchGroupCountContract, ZeroCountLeavesMetadataUntouched)
{
    Knobs kn;
    kn.group_count = 0;
    kn.initial_code =
        DLP_CLSC_INVALID_MATRIX_TYPE; // no validator produces this

    const dlp_clsc_err_t got = invoke_batch<float, float, float, float>(
        &aocl_batch_gemm_f32f32f32of32, kn);

    EXPECT_EQ(got, DLP_CLSC_INVALID_MATRIX_TYPE)
        << "a group_count of 0 wrote an error code even though there is no "
           "group to report on";
}

// ============================================================================
// YAML-DRIVEN PARAMETERIZED TESTS
// ============================================================================

/**
 * @brief Parameterized test fixture for YAML-driven batch GEMM tests
 *
 * Each test instance runs DLP implementation against reference and compares
 * results. Supports both single-group and multi-group configurations.
 */
class BatchGemmYamlTest : public ::testing::TestWithParam<BatchGemmTestConfig>
{
  protected:
    void SetUp() override
    {
        // Setup can be used for common initialization
    }

    void TearDown() override
    {
        // Cleanup if needed
    }

    /**
     * @brief Execute batch GEMM test for a given configuration
     */
    void executeBatchGemmTest(const BatchGemmTestConfig& config)
    {
        // Check if parameters are valid
        bool params_valid = check_valid_batch_params(config);

        // Convert config to groups (with per-group PostOps if present)
        auto dlp_groups =
            configToGroups(config, config.post_op_params_per_group);
        auto ref_groups =
            configToGroups(config, config.post_op_params_per_group);

        // Create UAL instances
        auto ual_dlp = UalFactory::createUal(UALType::DLP);
        auto ual_ref = UalFactory::createUal(UALType::REF);
        ASSERT_NE(ual_dlp, nullptr)
            << "Failed to create DLP UAL for test: " << config.name;
        ASSERT_NE(ual_ref, nullptr)
            << "Failed to create REF UAL for test: " << config.name;

        // Execute DLP implementation
        auto status_dlp = ual_dlp->batch_gemm(dlp_groups, config.acc_type);

        // Skip test if ISA not supported (e.g., AVX512_BF16 for BF16,
        // AVX512_VNNI for INT8)
        if (status_dlp == UALError::UAL_NO_MATCHING_API) {
            GTEST_SKIP()
                << "No classic aocl_batch_gemm_* API for this type combination";
        }
        if (status_dlp == UALError::UAL_NOT_SUPPORTED) {
            GTEST_SKIP()
                << "DLP batch_gemm not supported for this configuration "
                << "(ISA not available on this hardware)";
        }

        // Execute reference implementation
        auto status_ref = ual_ref->batch_gemm(ref_groups, config.acc_type);

        if (params_valid) {
            // For valid parameters, both implementations should succeed
            EXPECT_EQ(status_dlp, UALError::UAL_SUCCESS)
                << "DLP batch_gemm should succeed with valid parameters for "
                   "test: "
                << config.name;
            EXPECT_EQ(status_ref, UALError::UAL_SUCCESS)
                << "REF batch_gemm should succeed with valid parameters for "
                   "test: "
                << config.name;

            // Compare results only if both succeeded
            if (status_dlp == UALError::UAL_SUCCESS
                && status_ref == UALError::UAL_SUCCESS) {
                ASSERT_EQ(dlp_groups.size(), ref_groups.size())
                    << "Group count mismatch for test: " << config.name;

                for (std::size_t g = 0; g < dlp_groups.size(); ++g) {
                    const auto& dlp_group = dlp_groups[g];
                    const auto& ref_group = ref_groups[g];

                    ASSERT_EQ(dlp_group.C_matrices.size(),
                              ref_group.C_matrices.size())
                        << "Matrix count mismatch in group " << g
                        << " for test: " << config.name;

                    for (std::size_t i = 0; i < dlp_group.C_matrices.size();
                         ++i) {
                        // Set k dimension for comparison (needed for certain
                        // matrix formats)
                        dlp_group.C_matrices[i].setK(dlp_group.k);
                        ref_group.C_matrices[i].setK(ref_group.k);

                        // Prepare comparison options
                        MatrixCompareOptions compare_opts =
                            MatrixCompareOptions::Fast();
                        if (config.has_tolerances) {
                            compare_opts.relToleranceOverride =
                                config.tolerance_relative;
                            compare_opts.absToleranceOverride =
                                config.tolerance_absolute;
                        }
                        compare_opts.treatNaNEqual = config.treat_nan_equal;

                        // Compare matrices
                        auto result = dlp_group.C_matrices[i].compare(
                            ref_group.C_matrices[i], compare_opts);

                        EXPECT_TRUE(result.equal)
                            << "Matrix comparison failed for test: "
                            << config.name << "\n  Group: " << g
                            << ", Matrix: " << i
                            << "\n  Dimensions: m=" << config.m_values[g]
                            << ", n=" << config.n_values[g]
                            << ", k=" << config.k_values[g] << "\n"
                            << FormatCompareResult(result,
                                                   dlp_group.C_matrices[i],
                                                   ref_group.C_matrices[i]);
                    }
                }
            }
        } else {
            // For invalid parameters, both implementations should fail
            // gracefully
            EXPECT_NE(status_dlp, UALError::UAL_SUCCESS)
                << "DLP batch_gemm should fail gracefully with invalid "
                   "parameters for test: "
                << config.name;
            EXPECT_NE(status_ref, UALError::UAL_SUCCESS)
                << "REF batch_gemm should fail gracefully with invalid "
                   "parameters for test: "
                << config.name;

            // No need to compare results when both operations should have
            // failed
            std::cout << "Test passed: Both implementations correctly rejected "
                         "invalid parameters for test: "
                      << config.name << std::endl;
        }
    }
};

/**
 * @brief Main parameterized test that executes all YAML configurations
 */
TEST_P(BatchGemmYamlTest, YamlDrivenTest)
{
    const auto& config = GetParam();
    executeBatchGemmTest(config);
}

// ============================================================================
// TEST INSTANTIATION
// ============================================================================

/**
 * @brief Instantiate parameterized tests from YAML configuration
 *
 * Loads all test configurations from batch_gemm_test_config.yaml and creates
 * a separate test case for each configuration. Test names are derived from
 * the "name" field in the YAML.
 */
// Load batch GEMM configurations from all specified YAML files. When multiple
// files are provided, a per-file prefix is applied to generated test names to
// keep them globally unique. For a single file, the prefix is left empty to
// preserve the original naming scheme.
static std::vector<BatchGemmTestConfig>
loadBatchGemmTestConfigurationsFromFiles(
    const std::vector<std::string>& yaml_files)
{
    std::vector<BatchGemmTestConfig> all_configs;

    // The prefix combines a file index with the sanitized file stem
    // ("f<idx>_<stem>") so that even files that share the same filename (in
    // different directories) never produce duplicate test names.
    const bool multiple_files = yaml_files.size() > 1;
    for (size_t fi = 0; fi < yaml_files.size(); ++fi) {
        const std::string& yaml_file = yaml_files[fi];
        std::string        prefix    = multiple_files
                                           ? ("f" + std::to_string(fi) + "_"
                                    + sanitizeBatchFileStem(yaml_file))
                                           : "";
        auto configs = loadBatchGemmTestConfigurations(yaml_file, prefix);
        // Reserve to avoid repeated reallocations across files (each file can
        // contribute a large number of configs).
        all_configs.reserve(all_configs.size() + configs.size());
        all_configs.insert(all_configs.end(),
                           std::make_move_iterator(configs.begin()),
                           std::make_move_iterator(configs.end()));
    }

    return all_configs;
}

// Function to get batch GEMM test configurations (initialized on first call).
//
// IMPORTANT (multi-file -f/--file support):
// The configurations are built lazily via a function-local static, so the
// vector is populated the FIRST time this function is invoked -- NOT at
// static-initialization time. GoogleTest evaluates the ValuesIn(...) parameter
// generator during test registration (inside InitGoogleTest / RUN_ALL_TESTS),
// which runs from main() AFTER command-line arguments have been parsed and
// g_batch_gemm_yaml_files has been updated from -f/--file. Therefore the
// -f/--file selection (including multiple files) DOES affect which tests are
// instantiated. Do not convert this to a namespace-scope global initializer,
// which would run before main() and freeze the default config.
static const std::vector<BatchGemmTestConfig>&
getBatchGemmTestConfigurations()
{
    static std::vector<BatchGemmTestConfig> all_test_configs =
        loadBatchGemmTestConfigurationsFromFiles(g_batch_gemm_yaml_files);
    return all_test_configs;
}

INSTANTIATE_TEST_SUITE_P(
    YamlDriven,
    BatchGemmYamlTest,
    ::testing::ValuesIn(getBatchGemmTestConfigurations()),
    [](const ::testing::TestParamInfo<BatchGemmTestConfig>& info) {
        return info.param.name;
    });

// ============================================================================
// MAIN FUNCTION WITH ARGUMENT PARSING
// ============================================================================

// Custom main function to handle command-line arguments
int
main(int argc, char** argv)
{
    // Parse custom arguments before GoogleTest processes them
    auto parser = dlp::testing::utils::ArgParser::parseTestArgs(argc, argv);

    // Handle help request
    if (parser.helpRequested()) {
        parser.printUsage(argv[0]);
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "GoogleTest Help:" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    }

    // Resolve the YAML configuration file paths. Multiple files may be provided
    // via repeated -f/--file flags and/or comma-separated values. The built-in
    // default (g_batch_gemm_yaml_files) is threaded through getYamlFiles() so
    // it is only used when no -f/--file flag was given. Non-existent files
    // passed via -f are skipped with a warning.
    const std::string        default_yaml = g_batch_gemm_yaml_files.empty()
                                                ? std::string()
                                                : g_batch_gemm_yaml_files[0];
    std::vector<std::string> yaml_files   = parser.getYamlFiles(default_yaml);

    if (yaml_files.empty()) {
        // Reaching here means either the user explicitly passed -f but every
        // supplied path was invalid, or the default itself is missing. In
        // either case fail loudly rather than silently running a different
        // suite (e.g. a mistyped -f path in CI must not pass by running the
        // default configuration).
        std::cerr << "Error: No valid YAML configuration file(s) provided!"
                  << std::endl;
        if (parser.hasYamlFileArg()) {
            std::cerr << "All paths given via -f/--file were invalid."
                      << std::endl;
        }
        std::cerr << "Please check the file path(s) or run with -h for usage "
                     "information."
                  << std::endl;
        return 1;
    }

    g_batch_gemm_yaml_files = yaml_files;
    if (parser.hasYamlFileArg()) {
        std::cout << "Using YAML configuration file(s):" << std::endl;
    } else {
        std::cout << "Using default YAML configuration file(s):" << std::endl;
    }
    for (const auto& f : g_batch_gemm_yaml_files) {
        std::cout << "  - " << f << std::endl;
    }

    // Initialize GoogleTest
    ::testing::InitGoogleTest(&argc, argv);

    // Exit if help was requested
    if (parser.helpRequested()) {
        return 0;
    }

    // Run all tests
    return RUN_ALL_TESTS();
}
