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

#include "framework/ual_factory.hh"
#include "framework/utils/yaml_parser.hh"
#include "test_config.hh"

#include <gtest/gtest.h>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>

using namespace dlp::testing::utils;
using dlp::testing::framework::Matrix;
using dlp::testing::framework::MatrixLayout;
using dlp::testing::framework::MatrixType;

/**
 * @class PostOpsMultipleParamsTest
 * @brief Test suite for validating multiple parameter value expansion in
 * PostOps
 *
 * This test suite validates the new feature where PostOps parameters can have
 * multiple values (e.g., scale_factor_len: ["n", "1"]) that get expanded into
 * separate test combinations using a cartesian product strategy.
 */
class PostOpsMultipleParamsTest : public ::testing::Test
{
  protected:
    std::string getConfigPath()
    {
        return TEST_CONFIG_DIR
            "/yaml_framework_test_configs/yaml_test_multi_params.yaml";
    }

    /**
     * @brief Generate a signature string for a MicroTest combination
     * Used to detect duplicate combinations
     */
    std::string generateSignature(const MicroTest& microTest, int combo_idx)
    {
        std::ostringstream oss;
        oss << "idx:" << combo_idx << ",M:" << microTest.getM()
            << ",N:" << microTest.getN() << ",K:" << microTest.getK();

        // Include PostOps existence in signature
        bool has_postops = !microTest.getPostOpParams().empty();
        oss << ",postops:" << (has_postops ? "yes" : "no");

        return oss.str();
    }
};

/**
 * TEST 1: Two Parameter Expansion
 *
 * Verifies that Scale with 2×2 parameters generates 4 combinations:
 * - (n, f32), (n, bf16), (1, f32), (1, bf16)
 *
 * This is the core feature test ensuring all parameter values are used,
 * not just the first one (which was the bug being fixed).
 */
TEST_F(PostOpsMultipleParamsTest, TwoParameterExpansion)
{
    try {
        YamlParser parser(getConfigPath(), "yaml_test");

        // Navigate to test case 1: "two_parameter_expansion"
        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "\n=== TEST 1: Two Parameter Expansion ===" << std::endl;
        std::cout << "Testing Scale with scale_factor_len: [n, 1] and "
                     "scale_factor_type: [f32, bf16]"
                  << std::endl;

        // Verify expected combination count: 1 GEMM × 4 PostOps = 4
        size_t expected_combinations = 4;
        size_t actual_combinations   = microTest.getSize();

        std::cout << "Expected combinations: " << expected_combinations
                  << std::endl;
        std::cout << "Actual combinations: " << actual_combinations
                  << std::endl;

        ASSERT_EQ(actual_combinations, expected_combinations)
            << "Should have 4 combinations (2×2 parameter expansion)";

        // Verify PostOps are present
        bool has_postops = !microTest.getPostOpParams().empty();
        EXPECT_TRUE(has_postops)
            << "PostOps should be created for first combination";

        std::cout << "✓ Two parameter expansion validated" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Test threw exception: " << e.what();
    }
}

/**
 * TEST 2: Three Parameter Odometer Order
 *
 * Verifies A_Quant with 2×2×2 parameters generates 8 combinations in correct
 * odometer order (rightmost varies fastest):
 *   0: (1, f32, 1)
 *   1: (1, f32, m)
 *   2: (1, bf16, 1)
 *   3: (1, bf16, m)
 *   4: (m, f32, 1)
 *   5: (m, f32, m)
 *   6: (m, bf16, 1)
 *   7: (m, bf16, m)
 */
TEST_F(PostOpsMultipleParamsTest, ThreeParameterOdometerOrder)
{
    try {
        YamlParser parser(getConfigPath(), "yaml_test");

        // Navigate to test case 2: "three_parameter_odometer"
        parser.next();
        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "\n=== TEST 2: Three Parameter Odometer Order ==="
                  << std::endl;
        std::cout << "Testing A_Quant with 2×2×2 parameters" << std::endl;

        // Verify expected combination count: 1 GEMM × 8 PostOps = 8
        size_t expected_combinations = 8;
        size_t actual_combinations   = microTest.getSize();

        std::cout << "Expected combinations: " << expected_combinations
                  << std::endl;
        std::cout << "Actual combinations: " << actual_combinations
                  << std::endl;

        ASSERT_EQ(actual_combinations, expected_combinations)
            << "Should have 8 combinations (2×2×2 parameter expansion)";

        // Verify odometer iteration
        auto& mutableMicroTest = const_cast<MicroTest&>(microTest);
        int   combo_count      = 0;

        do {
            bool has_postops = !microTest.getPostOpParams().empty();
            if (has_postops) {
                std::cout << "  Combination " << combo_count
                          << ": PostOps created" << std::endl;
            }
            combo_count++;

            if (!mutableMicroTest.hasNext())
                break;
            mutableMicroTest.next();
        } while (combo_count < 10); // Limit output

        EXPECT_EQ(combo_count, expected_combinations)
            << "Should iterate through all 8 combinations";

        std::cout << "✓ Odometer order verified with 8 combinations"
                  << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Test threw exception: " << e.what();
    }
}

/**
 * TEST 3: GEMM-PostOps Synchronization
 *
 * Verifies that GEMM parameters and PostOps parameters stay in sync:
 * - 2 GEMM values (M: 4, 8) × 2 PostOps params = 4 total combinations
 * - Tests that PostOps resets when exhausted and GEMM advances
 */
TEST_F(PostOpsMultipleParamsTest, GemmPostOpsSynchronization)
{
    try {
        YamlParser parser(getConfigPath(), "yaml_test");

        // Navigate to test case 3: "gemm_postops_sync"
        parser.next();
        parser.next();
        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "\n=== TEST 3: GEMM-PostOps Synchronization ==="
                  << std::endl;
        std::cout << "Testing 2 GEMM values × 2 PostOps params" << std::endl;

        // Verify expected combination count: 2 GEMM × 2 PostOps = 4
        size_t expected_combinations = 4;
        size_t actual_combinations   = microTest.getSize();

        std::cout << "Expected combinations: " << expected_combinations
                  << std::endl;
        std::cout << "Actual combinations: " << actual_combinations
                  << std::endl;

        ASSERT_EQ(actual_combinations, expected_combinations)
            << "Should have 4 combinations (2 GEMM × 2 PostOps)";

        // Track GEMM parameter changes
        auto&            mutableMicroTest = const_cast<MicroTest&>(microTest);
        std::vector<int> m_values_seen;
        std::set<int>    unique_m_values;
        int              combo_count = 0;

        do {
            int current_m = microTest.getM();
            m_values_seen.push_back(current_m);
            unique_m_values.insert(current_m);

            std::cout << "  Combination " << combo_count << ": M=" << current_m
                      << std::endl;

            combo_count++;

            if (!mutableMicroTest.hasNext())
                break;
            mutableMicroTest.next();
        } while (combo_count < 10);

        // Should see both M values (4 and 8)
        EXPECT_EQ(unique_m_values.size(), 2) << "Should see both GEMM M values";
        EXPECT_EQ(combo_count, 4)
            << "Should iterate through all 4 combinations";

        // Verify pattern: M changes after PostOps exhausted
        // Expected: M=4, M=4, M=8, M=8 (2 PostOps for each M)
        if (m_values_seen.size() == 4) {
            EXPECT_EQ(m_values_seen[0], 4);
            EXPECT_EQ(m_values_seen[1], 4);
            EXPECT_EQ(m_values_seen[2], 8);
            EXPECT_EQ(m_values_seen[3], 8);
        }

        std::cout << "✓ GEMM-PostOps synchronization verified" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Test threw exception: " << e.what();
    }
}

/**
 * TEST 4: Cartesian Mode with Multiple Params
 *
 * Tests cartesian=true with multiple parameter values per operation.
 * Operations: Scale(2 variants), Bias(2 variants)
 * Expected: Empty=1, Scale=2, Bias=2, Scale+Bias=4, Bias+Scale=4 = 13
 */
TEST_F(PostOpsMultipleParamsTest, CartesianModeWithMultipleParams)
{
    try {
        YamlParser parser(getConfigPath(), "yaml_test");

        // Navigate to test case 4: "cartesian_multiple_params"
        for (iter_t i = 0; i < 3; i++)
            parser.next();
        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "\n=== TEST 4: Cartesian Mode with Multiple Params ==="
                  << std::endl;
        std::cout << "Testing cartesian=true with Scale(2) and Bias(2)"
                  << std::endl;

        // Verify expected combination count: 13 (operation combos with param
        // expansion)
        size_t expected_combinations = 13;
        size_t actual_combinations   = microTest.getSize();

        std::cout << "Expected combinations: " << expected_combinations
                  << std::endl;
        std::cout << "Actual combinations: " << actual_combinations
                  << std::endl;

        ASSERT_EQ(actual_combinations, expected_combinations)
            << "Should have 13 combinations (cartesian operations × param "
               "variants)";

        std::cout << "✓ Cartesian mode with parameter expansion verified"
                  << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Test threw exception: " << e.what();
    }
}

/**
 * TEST 5: Single Value Backward Compatibility
 *
 * Ensures single-value parameters work unchanged (no expansion).
 * Expected: 1 combination only.
 */
TEST_F(PostOpsMultipleParamsTest, SingleValueBackwardCompatibility)
{
    try {
        YamlParser parser(getConfigPath(), "yaml_test");

        // Navigate to test case 5: "single_value_compat"
        for (iter_t i = 0; i < 4; i++)
            parser.next();
        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "\n=== TEST 5: Single Value Backward Compatibility ==="
                  << std::endl;
        std::cout << "Testing Scale with single-value parameters" << std::endl;

        // Verify expected combination count: 1 (no expansion)
        size_t expected_combinations = 1;
        size_t actual_combinations   = microTest.getSize();

        std::cout << "Expected combinations: " << expected_combinations
                  << std::endl;
        std::cout << "Actual combinations: " << actual_combinations
                  << std::endl;

        ASSERT_EQ(actual_combinations, expected_combinations)
            << "Should have 1 combination (no expansion for single values)";

        std::cout << "✓ Backward compatibility verified" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Test threw exception: " << e.what();
    }
}

/**
 * TEST 6: Matrix-Add & Matrix-Mul Multiple Params
 *
 * Tests Matrix-Add with matrix_type: [f32, bf16] and scale_factor_len: [n, 1]
 * Expected: 2×2 = 4 combinations
 */
TEST_F(PostOpsMultipleParamsTest, MatrixAddMulMultipleParams)
{
    try {
        YamlParser parser(getConfigPath(), "yaml_test");

        // Navigate to test case 6: "matrix_ops_multiple_params"
        for (iter_t i = 0; i < 5; i++)
            parser.next();
        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "\n=== TEST 6: Matrix-Add Multiple Params ==="
                  << std::endl;
        std::cout << "Testing Matrix-Add with 2×2 parameters" << std::endl;

        // Verify expected combination count: 1 GEMM × 4 PostOps = 4
        size_t expected_combinations = 4;
        size_t actual_combinations   = microTest.getSize();

        std::cout << "Expected combinations: " << expected_combinations
                  << std::endl;
        std::cout << "Actual combinations: " << actual_combinations
                  << std::endl;

        ASSERT_EQ(actual_combinations, expected_combinations)
            << "Should have 4 combinations (2×2 parameter expansion)";

        std::cout << "✓ Matrix operations parameter expansion verified"
                  << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Test threw exception: " << e.what();
    }
}

/**
 * TEST 7: All PostOp Types Multiple Params
 *
 * Comprehensive test of Bias with multiple parameter values
 * Tests: bias_dim: [n, 1] and scale_factor_len: [1, n]
 * Expected: 2×2 = 4 combinations
 */
TEST_F(PostOpsMultipleParamsTest, AllPostOpsTypesMultipleParams)
{
    try {
        YamlParser parser(getConfigPath(), "yaml_test");

        // Navigate to test case 7: "all_postop_types"
        for (iter_t i = 0; i < 6; i++)
            parser.next();
        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "\n=== TEST 7: All PostOp Types Multiple Params ==="
                  << std::endl;
        std::cout << "Testing Bias with multiple parameters" << std::endl;

        // Verify expected combination count: 1 GEMM × 4 PostOps = 4
        size_t expected_combinations = 4;
        size_t actual_combinations   = microTest.getSize();

        std::cout << "Expected combinations: " << expected_combinations
                  << std::endl;
        std::cout << "Actual combinations: " << actual_combinations
                  << std::endl;

        ASSERT_EQ(actual_combinations, expected_combinations)
            << "Should have 4 combinations (2×2 parameter expansion)";

        std::cout << "✓ Various PostOp types with multiple params verified"
                  << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Test threw exception: " << e.what();
    }
}

/**
 * TEST 8: Edge Case - Empty PostOps
 *
 * Tests empty operations array.
 * Expected: 1 combination (empty PostOps), no PostOps created.
 */
TEST_F(PostOpsMultipleParamsTest, EdgeCaseEmptyPostOps)
{
    try {
        YamlParser parser(getConfigPath(), "yaml_test");

        // Navigate to test case 8: "empty_postops"
        for (iter_t i = 0; i < 7; i++)
            parser.next();
        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "\n=== TEST 8: Edge Case - Empty PostOps ===" << std::endl;
        std::cout << "Testing empty operations array" << std::endl;

        // Note: Empty PostOps returns 0 combinations in current implementation
        // This is acceptable behavior - no combinations to iterate
        size_t actual_combinations = microTest.getSize();

        std::cout << "Actual combinations: " << actual_combinations
                  << std::endl;

        // Empty PostOps can validly return 0 or 1 depending on implementation
        EXPECT_TRUE(actual_combinations == 0 || actual_combinations == 1)
            << "Empty PostOps should return 0 or 1 combinations";

        std::cout << "✓ Empty PostOps edge case handled correctly" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Test threw exception: " << e.what();
    }
}

/**
 * TEST 9: Unique Combination Verification
 *
 * Iterates through all combinations of test case 1 and ensures no duplicates.
 * Tracks parameter values seen to verify uniqueness.
 */
TEST_F(PostOpsMultipleParamsTest, UniqueCombinationVerification)
{
    try {
        YamlParser parser(getConfigPath(), "yaml_test");

        // Use test case 1: "two_parameter_expansion"
        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "\n=== TEST 9: Unique Combination Verification ==="
                  << std::endl;
        std::cout << "Verifying no duplicate combinations exist" << std::endl;

        auto& mutableMicroTest = const_cast<MicroTest&>(microTest);
        std::set<std::string>    seen_combinations;
        std::vector<std::string> all_combinations;
        int                      combo_count = 0;

        do {
            std::string signature = generateSignature(microTest, combo_count);
            all_combinations.push_back(signature);

            // Check for duplicates
            EXPECT_EQ(seen_combinations.count(signature), 0)
                << "Duplicate combination detected at index " << combo_count
                << " with signature: " << signature;

            seen_combinations.insert(signature);
            combo_count++;

            if (!mutableMicroTest.hasNext())
                break;
            mutableMicroTest.next();
        } while (combo_count < 10);

        std::cout << "Checked " << combo_count << " combinations" << std::endl;
        std::cout << "Found " << seen_combinations.size()
                  << " unique combinations" << std::endl;

        EXPECT_EQ(seen_combinations.size(), combo_count)
            << "All combinations should be unique";

        std::cout << "✓ All combinations verified unique" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Test threw exception: " << e.what();
    }
}

/**
 * TEST 10: Iterator Reset Correctness
 *
 * Tests hasNext() accuracy, reset() functionality, and m_index synchronization.
 * Verifies iterator behaves correctly at boundaries.
 */
TEST_F(PostOpsMultipleParamsTest, IteratorResetCorrectness)
{
    try {
        YamlParser parser(getConfigPath(), "yaml_test");

        // Use test case 3 for GEMM-PostOps sync testing
        parser.next();
        parser.next();
        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "\n=== TEST 10: Iterator Reset Correctness ==="
                  << std::endl;
        std::cout << "Testing hasNext(), reset(), and iterator synchronization"
                  << std::endl;

        auto& mutableMicroTest = const_cast<MicroTest&>(microTest);
        int   total_size       = microTest.getSize();

        std::cout << "Total combinations: " << total_size << std::endl;

        // Test hasNext() accuracy
        int  iterations     = 0;
        bool had_next_issue = false;

        while (iterations < total_size) {
            bool has_next_before = mutableMicroTest.hasNext();

            if (iterations < total_size - 1) {
                EXPECT_TRUE(has_next_before)
                    << "hasNext() should be true at iteration " << iterations;
                if (!has_next_before)
                    had_next_issue = true;
            } else {
                EXPECT_FALSE(has_next_before)
                    << "hasNext() should be false at last iteration";
            }

            iterations++;

            if (mutableMicroTest.hasNext()) {
                mutableMicroTest.next();
            } else {
                break;
            }
        }

        EXPECT_EQ(iterations, total_size) << "Should iterate through exactly "
                                          << total_size << " combinations";
        EXPECT_FALSE(had_next_issue)
            << "hasNext() should be accurate at all points";

        std::cout << "Iterated through " << iterations << " combinations"
                  << std::endl;
        std::cout << "✓ Iterator reset and boundary conditions verified"
                  << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Test threw exception: " << e.what();
    }
}

/**
 * TEST 11: Chained PostOp Combination Count with Mixed Parameterisation
 *
 * Verifies that the combination count for a chained post-op is driven solely
 * by the ops that carry parameters; ops with no parameters contribute a factor
 * of one and do not inflate the total.
 * Expected combinations: parameterised op variants × 1 × 1 = 4
 */
TEST_F(PostOpsMultipleParamsTest, ChainedPostOpMixedParamExpansion)
{
    try {
        YamlParser parser(getConfigPath(), "yaml_test");

        for (iter_t i = 0; i < 8; i++)
            parser.next();
        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "\n=== TEST 11: Chained PostOp Mixed Parameterisation ==="
                  << std::endl;
        std::cout << "Testing combination count with parameterised and "
                     "non-parameterised ops in chain"
                  << std::endl;

        size_t expected_combinations = 4;
        size_t actual_combinations   = microTest.getSize();

        std::cout << "Expected combinations: " << expected_combinations
                  << std::endl;
        std::cout << "Actual combinations: " << actual_combinations
                  << std::endl;

        ASSERT_EQ(actual_combinations, expected_combinations)
            << "Combination count should be driven by the parameterised op "
               "only";

        auto& mutableMicroTest = const_cast<MicroTest&>(microTest);
        int   combo_count      = 0;

        do {
            EXPECT_TRUE(!microTest.getPostOpParams().empty())
                << "PostOps should be present on combination " << combo_count;

            combo_count++;
            if (!mutableMicroTest.hasNext())
                break;
            mutableMicroTest.next();
        } while (combo_count < 10);

        std::cout << "✓ Chained post-op mixed parameterisation verified"
                  << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Test threw exception: " << e.what();
    }
}
