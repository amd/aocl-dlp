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

#include "framework/matrix.hh"
#include "framework/operation.hh"
#include "framework/ual.hh"
#include "framework/ual_factory.hh"
#include "framework/utils/arg_parser.hh"
#include "framework/utils/yaml_parser.hh"
#include "test_config.hh"
#include <filesystem>
#include <gtest/gtest.h>

using namespace dlp::testing::framework;
using namespace dlp::testing::framework::postops;
using namespace dlp::testing::utils;

constexpr size_t MAX_CONFIGS_PER_TEST_SET = 50000;

// Global variable to store configurable YAML file path
static std::string g_batch_gemm_yaml_file = TEST_CONFIG_DIR
    "/batch_gemm_test_config.yaml";

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
 */
std::vector<BatchGemmTestConfig>
loadBatchGemmTestConfigurations(const std::string& yaml_file)
{
    std::vector<BatchGemmTestConfig> configs;

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

                // Generate test name
                config.name = "yaml_" + std::to_string(i) + "_"
                              + currentTestName + +"_MultiGroup";
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

                    // Generate test name
                    config.name = "yaml_" + std::to_string(i) + "_"
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
// Function to get batch GEMM test configurations (initialized on first call)
static const std::vector<BatchGemmTestConfig>&
getBatchGemmTestConfigurations()
{
    static std::vector<BatchGemmTestConfig> all_test_configs =
        loadBatchGemmTestConfigurations(g_batch_gemm_yaml_file);
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

    // Update YAML configuration file path if specified
    std::string yaml_file = parser.getYamlFile();
    if (!yaml_file.empty()) {
        g_batch_gemm_yaml_file = yaml_file;
        std::cout << "Using custom YAML configuration: "
                  << g_batch_gemm_yaml_file << std::endl;
    } else {
        std::cout << "Using default YAML configuration: "
                  << g_batch_gemm_yaml_file << std::endl;
    }

    // Check if specified file exists
    if (!parser.getYamlFile().empty()
        && !std::filesystem::exists(g_batch_gemm_yaml_file)) {
        std::cerr << "Error: YAML file '" << g_batch_gemm_yaml_file
                  << "' does not exist!" << std::endl;
        return 1;
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
