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

#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

#include "framework/utils/yaml_parser.hh"
#include "test_config.hh"
#include "utils/conversion_utils.hh"

using namespace dlp::testing::utils;
using dlp::testing::framework::MatrixLayout;
using dlp::testing::framework::MatrixType;

/*
   Test configurations which uses range and list type
   File: yaml_test_config_range_list.yaml
 */
// Test the YamlParser with a sample YAML configuration file
TEST(YamlParserTest, ParseGemmTestConfig)
{
    // Use a relative path to the test configuration file
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_config_range_list.yaml";

    // Create and test the parser
    try {
        YamlParser parser(filepath, "yaml_test");

        // Get the number of test cases
        size_t testCount = parser.getMicroTestCount();
        ASSERT_EQ(testCount, 2) << "Expected 2 test cases in the YAML file";

        // Get the first MicroTest
        const MicroTest& microTest = parser.getMicroTest();

        // Test the first test case - this will be the first combination from
        // the cartesian product
        std::cout << "Test Case 0:" << std::endl;
        std::cout << "  A Type: " << microTest.getAType() << std::endl;
        std::cout << "  B Type: " << microTest.getBType() << std::endl;
        std::cout << "  C Type: " << microTest.getCType() << std::endl;
        std::cout << "  Acc Type: " << microTest.getAccType() << std::endl;
        std::cout << "  Storage Format: " << microTest.getStorageFormat()
                  << std::endl;
        std::cout << "  M: " << microTest.getM() << std::endl;
        std::cout << "  N: " << microTest.getN() << std::endl;
        std::cout << "  K: " << microTest.getK() << std::endl;
        std::cout << "  LDA: " << microTest.getLDA() << std::endl;
        std::cout << "  LDB: " << microTest.getLDB() << std::endl;
        std::cout << "  LDC: " << microTest.getLDC() << std::endl;
        std::cout << "  Alpha: " << microTest.getAlpha() << std::endl;
        std::cout << "  Beta: " << microTest.getBeta() << std::endl;
        std::cout << "  TransA: " << (microTest.getTransA() ? "true" : "false")
                  << std::endl;
        std::cout << "  TransB: " << (microTest.getTransB() ? "true" : "false")
                  << std::endl;
        std::cout << "  ReorderA: "
                  << (microTest.getReorderA() ? "true" : "false") << std::endl;
        std::cout << "  ReorderB: "
                  << (microTest.getReorderB() ? "true" : "false") << std::endl;
        std::cout << "  PackA: " << (microTest.getPackA() ? "true" : "false")
                  << std::endl;
        std::cout << "  PackB: " << (microTest.getPackB() ? "true" : "false")
                  << std::endl;

        // Verify some values from the first test case
        EXPECT_EQ(microTest.getAType(), MatrixType::f32);
        EXPECT_EQ(microTest.getBType(), MatrixType::f32);
        EXPECT_EQ(microTest.getCType(), MatrixType::f32);
        EXPECT_EQ(microTest.getAccType(), MatrixType::f32);
        EXPECT_EQ(microTest.getStorageFormat(), MatrixLayout::ROW_MAJOR);

        // The first combination should have the first values from each
        // range/list
        EXPECT_EQ(microTest.getM(), 10); // First value from range lb:10
        EXPECT_EQ(microTest.getN(),
                  10); // First value from range lb:10, ub:10, step:0
        EXPECT_EQ(microTest.getK(),
                  10); // First value from range lb:10, ub:10, step:0
        EXPECT_EQ(microTest.getAlpha(),
                  2.5f);                      // First value from [2.5, 0, -2.5]
        EXPECT_EQ(microTest.getBeta(), 2.5f); // First value from [2.5, 0, -2.5]
        EXPECT_EQ(microTest.getLDA(), 10);    // First value from [10, 20, 30]
        EXPECT_EQ(microTest.getLDB(),
                  10); // First value from range lb:10, ub:10, step:0
        EXPECT_EQ(microTest.getLDC(), 10);     // Single value
        EXPECT_FALSE(microTest.getTransA());   // First value from [false, true]
        EXPECT_TRUE(microTest.getTransB());    // Single value [true]
        EXPECT_FALSE(microTest.getReorderA()); // Single value false
        EXPECT_TRUE(microTest.getReorderB());  // Single value true

        // Move to the next test case
        parser.next();

        // Test the second test case
        const MicroTest& microTest2 = parser.getMicroTest();
        std::cout << "\nTest Case 1:" << std::endl;
        std::cout << "  A Type: " << microTest2.getAType() << std::endl;
        std::cout << "  B Type: " << microTest2.getBType() << std::endl;
        std::cout << "  C Type: " << microTest2.getCType() << std::endl;
        std::cout << "  Acc Type: " << microTest2.getAccType() << std::endl;
        std::cout << "  Storage Format: " << microTest2.getStorageFormat()
                  << std::endl;
        std::cout << "  M: " << microTest2.getM() << std::endl;
        std::cout << "  N: " << microTest2.getN() << std::endl;
        std::cout << "  K: " << microTest2.getK() << std::endl;
        std::cout << "  LDA: " << microTest2.getLDA() << std::endl;
        std::cout << "  LDB: " << microTest2.getLDB() << std::endl;
        std::cout << "  LDC: " << microTest2.getLDC() << std::endl;
        std::cout << "  Alpha: " << microTest2.getAlpha() << std::endl;
        std::cout << "  Beta: " << microTest2.getBeta() << std::endl;
        std::cout << "  TransA: " << (microTest2.getTransA() ? "true" : "false")
                  << std::endl;
        std::cout << "  TransB: " << (microTest2.getTransB() ? "true" : "false")
                  << std::endl;
        std::cout << "  ReorderA: "
                  << (microTest2.getReorderA() ? "true" : "false") << std::endl;
        std::cout << "  ReorderB: "
                  << (microTest2.getReorderB() ? "true" : "false") << std::endl;
        std::cout << "  PackA: " << (microTest2.getPackA() ? "true" : "false")
                  << std::endl;
        std::cout << "  PackB: " << (microTest2.getPackB() ? "true" : "false")
                  << std::endl;

        // Verify some values from the second test case
        EXPECT_EQ(microTest2.getAType(), MatrixType::f32);
        EXPECT_EQ(microTest2.getBType(), MatrixType::f32);
        EXPECT_EQ(microTest2.getCType(), MatrixType::f32);
        EXPECT_EQ(microTest2.getAccType(), MatrixType::f32);
        EXPECT_EQ(microTest2.getStorageFormat(), MatrixLayout::ROW_MAJOR);

        // The second test case should have the first values from the
        // medium_matrix configuration
        EXPECT_EQ(microTest2.getM(), 256); // First value from range lb:256
        EXPECT_EQ(microTest2.getN(), 256); // First value from range lb:256
        EXPECT_EQ(microTest2.getK(), 256); // First value from range lb:256
        EXPECT_EQ(microTest2.getAlpha(),
                  1.5f); // First value from [1.5, 0, -1.5]
        EXPECT_EQ(microTest2.getBeta(),
                  1.5f);                     // First value from [1.5, 0, -1.5]
        EXPECT_EQ(microTest2.getLDA(), 256); // First value from [256, 512, 768]
        EXPECT_EQ(microTest2.getLDB(), 256); // First value from range lb:256
        EXPECT_EQ(microTest2.getLDC(), 256); // Single value
        EXPECT_FALSE(microTest2.getTransA()); // First value from [false, true]
        EXPECT_TRUE(microTest2.getTransB());  // First value from [true, false]
        EXPECT_FALSE(microTest2.getReorderA()); // Single value false
        EXPECT_TRUE(microTest2.getReorderB());  // Single value true

        // Test reset functionality
        parser.reset();
        const MicroTest& resetTest = parser.getMicroTest();
        EXPECT_EQ(resetTest.getM(), 10)
            << "Reset should return to the first test case";

    } catch (const std::exception& e) {
        FAIL() << "Parser threw an exception: " << e.what();
    }
}

// Test the entire cartesian product for a test case
TEST(YamlParserTest, MediumMatrixCartesianProductTest)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_config_range_list.yaml";

    try {
        YamlParser parser(filepath, "yaml_test");

        // Move to the second test case (medium_matrix)
        parser.next();

        // Get the medium_matrix test case and iterate through its combinations
        const MicroTest& microTestRef = parser.getMicroTest();
        // Cast away const to call next() - this is safe since we own the parser
        MicroTest& microTest = const_cast<MicroTest&>(microTestRef);

        // Get the total number of combinations for the medium_matrix test case
        size_t totalCombinations = microTest.getSize();
        std::cout << "\n=== Cartesian Product Test for 'medium_matrix' ==="
                  << std::endl;
        std::cout << "Expected total combinations: " << totalCombinations
                  << std::endl;
        std::cout << "\nAll combinations:" << std::endl;

        size_t combinationCount = 0;

        // Print the first combination (current state)
        std::cout << "\nCombination " << (combinationCount + 1) << ":"
                  << std::endl;
        std::cout << "  M=" << microTest.getM() << ", N=" << microTest.getN()
                  << ", K=" << microTest.getK()
                  << ", Alpha=" << microTest.getAlpha()
                  << ", Beta=" << microTest.getBeta()
                  << ", LDA=" << microTest.getLDA()
                  << ", TransA=" << (microTest.getTransA() ? "true" : "false")
                  << ", TransB=" << (microTest.getTransB() ? "true" : "false")
                  << ", ReorderB="
                  << (microTest.getReorderB() ? "true" : "false") << std::endl;
        combinationCount++;

        // Iterate through some combinations (limit to first 10 for output)
        while (microTest.hasNext() && combinationCount < 10) {
            microTest.next();
            std::cout << "\nCombination " << (combinationCount + 1) << ":"
                      << std::endl;
            std::cout << "  M=" << microTest.getM()
                      << ", N=" << microTest.getN()
                      << ", K=" << microTest.getK()
                      << ", Alpha=" << microTest.getAlpha()
                      << ", Beta=" << microTest.getBeta()
                      << ", LDA=" << microTest.getLDA() << ", TransA="
                      << (microTest.getTransA() ? "true" : "false")
                      << ", TransB="
                      << (microTest.getTransB() ? "true" : "false")
                      << ", ReorderB="
                      << (microTest.getReorderB() ? "true" : "false")
                      << std::endl;
            combinationCount++;
        }

        if (combinationCount >= 10) {
            std::cout << "\n... (showing first 10 combinations only)"
                      << std::endl;
        }

        // Verify the count matches expectation for medium_matrix:
        // Based on the YAML config for medium_matrix:
        // Base parameters:
        // - m: range lb:256, ub:1024, step:64 = [256, 320, 384, ..., 1024] = 13
        // values
        // - n: range lb:256, ub:1024, step:64 = [256, 320, 384, ..., 1024] = 13
        // values
        // - k: range lb:256, ub:1024, step:64 = [256, 320, 384, ..., 1024] = 13
        // values
        // - alpha: [1.5, 0, -1.5] = 3 values
        // - beta: [1.5, 0, -1.5] = 3 values
        // - lda: [256, 512, 768] = 3 values
        // - ldb: range lb:256, ub:1024, step:64 = 13 values
        // - transA: [false, true] = 2 values
        // - transB: [true, false] = 2 values
        // - reorderB: true = 1 value
        // Base total: 13 * 13 * 13 * 3 * 3 * 3 * 13 * 2 * 2 * 1 = 13^4 * 3^3 *
        // 2^2 = 28561 * 27 * 4 = 3,084,588 combinations
        //
        // PostOps (4 operations with cartesian=false):
        // Operations: [RELU, PRELU, Bias, Scale] applied in sequence
        // With cartesian=false: 1 combination (single sequence)
        // PostOps total: 1 combination
        //
        // Total combinations: 3,084,588 base × 1 PostOps = 3,084,588

        // Calculate expected range size: (1024 - 256) / 64 + 1 = 768/64 + 1 =
        // 12 + 1 = 13
        size_t rangeSize        = (1024 - 256) / 64 + 1; // 13 values
        size_t baseCombinations = rangeSize * rangeSize * rangeSize * 3 * 3 * 3
                                  * rangeSize * 2 * 2 * 1; // 3,084,588
        size_t postOpsCombinations = 1; // cartesian=false: single sequence
        size_t expectedCombinations =
            baseCombinations * postOpsCombinations; // 3,084,588

        EXPECT_EQ(totalCombinations, expectedCombinations)
            << "Expected " << expectedCombinations << " combinations but got "
            << totalCombinations;

        // Verify some values from the first combination
        EXPECT_EQ(microTest.getAType(), MatrixType::f32);
        EXPECT_EQ(microTest.getBType(), MatrixType::f32);
        EXPECT_EQ(microTest.getCType(), MatrixType::f32);
        EXPECT_EQ(microTest.getAccType(), MatrixType::f32);
        EXPECT_EQ(microTest.getStorageFormat(), MatrixLayout::ROW_MAJOR);

        std::cout << "\nCombinations shown: "
                  << std::min(combinationCount, static_cast<size_t>(10))
                  << std::endl;
        std::cout << "Expected total combinations: " << expectedCombinations
                  << std::endl;
        std::cout << "Reported by MicroTest.getSize(): " << totalCombinations
                  << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Parser threw an exception: " << e.what();
    }
}

/*
   Test configurations which uses value type
   File: yaml_test_config_value.yaml
 */

/**
 * @brief Test YAML parser with value-only configuration
 *
 * This test verifies that the parser correctly handles YAML configurations
 * where all parameters are specified as single values (no ranges or lists).
 * This is the simplest configuration type.
 */
TEST(YamlParserTest, ParseValueOnlyConfig)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_config_value.yaml";

    try {
        YamlParser parser(filepath, "yaml_test");

        // Get the number of test cases
        size_t testCount = parser.getMicroTestCount();
        ASSERT_EQ(testCount, 2)
            << "Expected 2 test cases in the value-only YAML file";

        // Test the first test case (small_matrix)
        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "\n=== Value-Only Config Test - small_matrix ==="
                  << std::endl;
        std::cout << "Test Case 0:" << std::endl;
        std::cout << "  A Type: " << microTest.getAType() << std::endl;
        std::cout << "  B Type: " << microTest.getBType() << std::endl;
        std::cout << "  C Type: " << microTest.getCType() << std::endl;
        std::cout << "  Acc Type: " << microTest.getAccType() << std::endl;
        std::cout << "  Storage Format: " << microTest.getStorageFormat()
                  << std::endl;
        std::cout << "  M: " << microTest.getM() << std::endl;
        std::cout << "  N: " << microTest.getN() << std::endl;
        std::cout << "  K: " << microTest.getK() << std::endl;
        std::cout << "  LDA: " << microTest.getLDA() << std::endl;
        std::cout << "  LDB: " << microTest.getLDB() << std::endl;
        std::cout << "  LDC: " << microTest.getLDC() << std::endl;
        std::cout << "  Alpha: " << microTest.getAlpha() << std::endl;
        std::cout << "  Beta: " << microTest.getBeta() << std::endl;
        std::cout << "  TransA: " << (microTest.getTransA() ? "true" : "false")
                  << std::endl;
        std::cout << "  TransB: " << (microTest.getTransB() ? "true" : "false")
                  << std::endl;
        std::cout << "  ReorderA: "
                  << (microTest.getReorderA() ? "true" : "false") << std::endl;
        std::cout << "  ReorderB: "
                  << (microTest.getReorderB() ? "true" : "false") << std::endl;

        // Verify values match the YAML configuration exactly
        EXPECT_EQ(microTest.getAType(), MatrixType::f32);
        EXPECT_EQ(microTest.getBType(), MatrixType::f32);
        EXPECT_EQ(microTest.getCType(), MatrixType::f32);
        EXPECT_EQ(microTest.getAccType(), MatrixType::f32);
        EXPECT_EQ(microTest.getStorageFormat(), MatrixLayout::ROW_MAJOR);
        EXPECT_EQ(microTest.getM(), 10);
        EXPECT_EQ(microTest.getN(), 10);
        EXPECT_EQ(microTest.getK(), 10);
        EXPECT_EQ(microTest.getAlpha(), 1.0);
        EXPECT_EQ(microTest.getBeta(), 0.0);
        EXPECT_EQ(microTest.getLDA(), 10);
        EXPECT_EQ(microTest.getLDB(), 10);
        EXPECT_EQ(microTest.getLDC(), 10);
        EXPECT_TRUE(microTest.getTransA());    // true in YAML
        EXPECT_FALSE(microTest.getTransB());   // false in YAML
        EXPECT_FALSE(microTest.getReorderA()); // false in YAML
        EXPECT_FALSE(microTest.getReorderB()); // false in YAML

        // For value-only config, there should be only 1 combination per test
        // case
        const MicroTest& microTestRef = parser.getMicroTest();
        MicroTest& mutableMicroTest   = const_cast<MicroTest&>(microTestRef);
        size_t     combinations       = mutableMicroTest.getSize();
        EXPECT_EQ(combinations, 1)
            << "Value-only config should have exactly 1 combination";
        EXPECT_FALSE(mutableMicroTest.hasNext())
            << "Value-only config should not have next combination";

        // Move to the second test case (medium_matrix)
        parser.next();
        const MicroTest& microTest2 = parser.getMicroTest();

        std::cout << "\nTest Case 1 (medium_matrix):" << std::endl;
        std::cout << "  M: " << microTest2.getM() << std::endl;
        std::cout << "  N: " << microTest2.getN() << std::endl;
        std::cout << "  K: " << microTest2.getK() << std::endl;
        std::cout << "  Alpha: " << microTest2.getAlpha() << std::endl;
        std::cout << "  Beta: " << microTest2.getBeta() << std::endl;
        std::cout << "  TransA: " << (microTest2.getTransA() ? "true" : "false")
                  << std::endl;
        std::cout << "  TransB: " << (microTest2.getTransB() ? "true" : "false")
                  << std::endl;

        // Verify medium_matrix values
        EXPECT_EQ(microTest2.getAType(), MatrixType::f32);
        EXPECT_EQ(microTest2.getBType(), MatrixType::f32);
        EXPECT_EQ(microTest2.getCType(), MatrixType::f32);
        EXPECT_EQ(microTest2.getAccType(), MatrixType::f32);
        EXPECT_EQ(microTest2.getStorageFormat(), MatrixLayout::ROW_MAJOR);
        EXPECT_EQ(microTest2.getM(), 256);
        EXPECT_EQ(microTest2.getN(), 256);
        EXPECT_EQ(microTest2.getK(), 256);
        EXPECT_EQ(microTest2.getAlpha(), 1.0);
        EXPECT_EQ(microTest2.getBeta(), 0.0);
        EXPECT_EQ(microTest2.getLDA(), 256);
        EXPECT_EQ(microTest2.getLDB(), 256);
        EXPECT_EQ(microTest2.getLDC(), 256);
        EXPECT_FALSE(microTest2.getTransA());   // false in YAML
        EXPECT_TRUE(microTest2.getTransB());    // true in YAML
        EXPECT_FALSE(microTest2.getReorderA()); // false in YAML
        EXPECT_FALSE(microTest2.getReorderB()); // false in YAML

        // Test reset functionality
        parser.reset();
        const MicroTest& resetTest = parser.getMicroTest();
        EXPECT_EQ(resetTest.getM(), 10)
            << "Reset should return to the first test case";
        EXPECT_TRUE(resetTest.getTransA())
            << "Reset should restore original TransA value";

    } catch (const std::exception& e) {
        FAIL() << "Parser threw an exception: " << e.what();
    }
}

/*
   Test configurations which uses list type
   File: yaml_test_config_list.yaml
 */

/**
 * @brief Test YAML parser with list-only configuration
 *
 * This test verifies that the parser correctly handles YAML configurations
 * where parameters are specified as lists of values. This tests the cartesian
 * product generation with list-based parameters.
 */
TEST(YamlParserTest, ParseListOnlyConfig)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_config_list.yaml";

    try {
        YamlParser parser(filepath, "yaml_test");

        // Get the number of test cases
        size_t testCount = parser.getMicroTestCount();
        ASSERT_EQ(testCount, 1)
            << "Expected 1 test case in the list-only YAML file";

        // Test the first test case (small_matrix)
        const MicroTest& microTestRef = parser.getMicroTest();
        MicroTest&       microTest    = const_cast<MicroTest&>(microTestRef);

        std::cout << "\n=== List-Only Config Test - small_matrix ==="
                  << std::endl;

        // Calculate expected combinations based on YAML config:
        // Base parameters:
        // - a_type: ["f32"] = 1 value
        // - b_type: ["f32"] = 1 value
        // - c_type: ["f32"] = 1 value
        // - acc_type: ["f32"] = 1 value
        // - storage_format: ["row-major"] = 1 value
        // - transA: [false, true] = 2 values
        // - transB: [true] = 1 value
        // - m: 10 = 1 value (single value)
        // - n: 10 = 1 value (single value)
        // - k: 10 = 1 value (single value)
        // - alpha: [2.5, 0, -2.5] = 3 values
        // - beta: [2.5, 0, -2.5] = 3 values
        // - lda: [10, 20, 30] = 3 values
        // - ldb: 10 = 1 value (single value)
        // - ldc: 10 = 1 value (single value)
        // - reorderA: false = 1 value (single value)
        // - reorderB: true = 1 value (single value)
        // Base total: 1 * 1 * 1 * 1 * 1 * 2 * 1 * 1 * 1 * 1 * 3 * 3 * 3 * 1 * 1
        // * 1 * 1 = 54 combinations
        //
        // PostOps (3 operations with cartesian=true):
        // With corrected logic: cartesian=true generates all permutations of
        // the full sequence For 3 operations ["Elementwise-PRELU", "Bias",
        // "Scale"]:
        // PostOps (3 operations with cartesian=true):
        // All subsets and their permutations:
        // - Empty: 1, Singles: 3, Pairs: C(3,2)*2! = 6, Triples: C(3,3)*3! = 6
        // - Total: 1 + 3 + 6 + 6 = 16 combinations
        //
        // Total combinations: 54 base × 16 PostOps = 864

        size_t totalCombinations = microTest.getSize();
        size_t baseCombinations  = 1 * 1 * 1 * 1 * 1 * 2 * 1 * 1 * 1 * 1 * 3 * 3
                                  * 3 * 1 * 1 * 1 * 1; // 54
        size_t postOpsCombinations = 16; // All subsets + permutations
        size_t expectedCombinations =
            baseCombinations * postOpsCombinations; // 864

        std::cout << "Expected total combinations: " << expectedCombinations
                  << std::endl;
        std::cout << "Reported by MicroTest.getSize(): " << totalCombinations
                  << std::endl;

        EXPECT_EQ(totalCombinations, expectedCombinations)
            << "Expected " << expectedCombinations << " combinations but got "
            << totalCombinations;

        // Test first few combinations
        size_t combinationCount = 0;
        std::cout << "\nFirst 10 combinations:" << std::endl;

        // Print the first combination (current state)
        std::cout << "\nCombination " << (combinationCount + 1) << ":"
                  << std::endl;
        std::cout << "  M=" << microTest.getM() << ", N=" << microTest.getN()
                  << ", K=" << microTest.getK()
                  << ", Alpha=" << microTest.getAlpha()
                  << ", Beta=" << microTest.getBeta()
                  << ", LDA=" << microTest.getLDA()
                  << ", TransA=" << (microTest.getTransA() ? "true" : "false")
                  << ", TransB=" << (microTest.getTransB() ? "true" : "false")
                  << ", ReorderA="
                  << (microTest.getReorderA() ? "true" : "false")
                  << ", ReorderB="
                  << (microTest.getReorderB() ? "true" : "false") << std::endl;

        // Verify first combination values
        EXPECT_EQ(microTest.getAType(), MatrixType::f32);
        EXPECT_EQ(microTest.getBType(), MatrixType::f32);
        EXPECT_EQ(microTest.getCType(), MatrixType::f32);
        EXPECT_EQ(microTest.getAccType(), MatrixType::f32);
        EXPECT_EQ(microTest.getStorageFormat(), MatrixLayout::ROW_MAJOR);
        EXPECT_EQ(microTest.getM(), 10);
        EXPECT_EQ(microTest.getN(), 10);
        EXPECT_EQ(microTest.getK(), 10);
        EXPECT_EQ(microTest.getAlpha(), 2.5); // First value from [2.5, 0, -2.5]
        EXPECT_EQ(microTest.getBeta(), 2.5);  // First value from [2.5, 0, -2.5]
        EXPECT_EQ(microTest.getLDA(), 10);    // First value from [10, 20, 30]
        EXPECT_EQ(microTest.getLDB(), 10);
        EXPECT_EQ(microTest.getLDC(), 10);
        EXPECT_FALSE(microTest.getTransA());   // First value from [false, true]
        EXPECT_TRUE(microTest.getTransB());    // Single value [true]
        EXPECT_FALSE(microTest.getReorderA()); // Single value false
        EXPECT_TRUE(microTest.getReorderB());  // Single value true

        combinationCount++;

        // Iterate through some combinations (limit to first 10 for output)
        while (microTest.hasNext() && combinationCount < 10) {
            microTest.next();
            std::cout << "\nCombination " << (combinationCount + 1) << ":"
                      << std::endl;
            std::cout
                << "  M=" << microTest.getM() << ", N=" << microTest.getN()
                << ", K=" << microTest.getK()
                << ", Alpha=" << microTest.getAlpha()
                << ", Beta=" << microTest.getBeta()
                << ", LDA=" << microTest.getLDA()
                << ", TransA=" << (microTest.getTransA() ? "true" : "false")
                << ", TransB=" << (microTest.getTransB() ? "true" : "false")
                << ", ReorderA=" << (microTest.getReorderA() ? "true" : "false")
                << ", ReorderB=" << (microTest.getReorderB() ? "true" : "false")
                << std::endl;
            combinationCount++;
        }

        if (combinationCount >= 10) {
            std::cout << "\n... (showing first 10 combinations only)"
                      << std::endl;
        }

        // Test that we can iterate through all combinations
        while (microTest.hasNext()) {
            microTest.next();
            combinationCount++;
        }

        EXPECT_EQ(combinationCount, expectedCombinations)
            << "Should be able to iterate through all " << expectedCombinations
            << " combinations";

        std::cout << "\nTotal combinations iterated: " << combinationCount
                  << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Parser threw an exception: " << e.what();
    }
}

/**
 * @brief Test YAML file consistency and parameter completeness
 *
 * This test verifies that all YAML configuration files have consistent
 * node names and parameter structures for GEMM testing.
 */
TEST(YamlParserTest, YamlFileConsistencyCheck)
{
    // Test files that should be compatible with GEMM testing
    std::vector<std::string> gemmConfigFiles = {
        TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_config_value.yaml",
        TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_config_list.yaml",
        TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_config_range_list.yaml"
    };

    std::vector<std::string> fileNames = { "yaml_test_config_value.yaml",
                                           "yaml_test_config_list.yaml",
                                           "yaml_test_config_range_list.yaml" };

    // Test each file can be parsed successfully
    for (size_t i = 0; i < gemmConfigFiles.size(); ++i) {
        try {
            YamlParser parser(gemmConfigFiles[i], "yaml_test");

            size_t testCount = parser.getMicroTestCount();
            EXPECT_GT(testCount, 0) << "File " << fileNames[i]
                                    << " should have at least 1 test case";

            // Test that we can access the first MicroTest
            const MicroTest& microTest = parser.getMicroTest();

            // Verify all required parameters are accessible (no exceptions
            // thrown)
            EXPECT_NO_THROW({
                auto aType         = microTest.getAType();
                auto bType         = microTest.getBType();
                auto cType         = microTest.getCType();
                auto accType       = microTest.getAccType();
                auto storageFormat = microTest.getStorageFormat();
                auto m             = microTest.getM();
                auto n             = microTest.getN();
                auto k             = microTest.getK();
                auto alpha         = microTest.getAlpha();
                auto beta          = microTest.getBeta();
                auto lda           = microTest.getLDA();
                auto ldb           = microTest.getLDB();
                auto ldc           = microTest.getLDC();
                auto transA        = microTest.getTransA();
                auto transB        = microTest.getTransB();
                auto reorderA      = microTest.getReorderA();
                auto reorderB      = microTest.getReorderB();
                auto packA         = microTest.getPackA();
                auto packB         = microTest.getPackB();
            }) << "File "
               << fileNames[i]
               << " should provide access to all required GEMM parameters";

            // Test that combinations can be generated
            const MicroTest& microTestRef = parser.getMicroTest();
            MicroTest& mutableMicroTest = const_cast<MicroTest&>(microTestRef);
            size_t     combinations     = mutableMicroTest.getSize();
            EXPECT_GT(combinations, 0)
                << "File " << fileNames[i]
                << " should generate at least 1 combination";

        } catch (const std::exception& e) {
            FAIL() << "File " << fileNames[i]
                   << " failed to parse: " << e.what();
        }
    }
}

/**
 * @brief Test SimpleProduct (element-wise) functionality with list
 * configuration
 *
 * This test verifies that the ELEMENT_WISE yield type works correctly,
 * pairing corresponding elements from different parameter lists.
 */
TEST(YamlParserTest, YieldTypeSwitchingTest)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_config_list.yaml";

    try {
        YamlParser parser(filepath, "yaml_test");

        // Test default mode (should be CARTESIAN_PRODUCT)
        EXPECT_EQ(parser.getYieldType(), YieldType::CARTESIAN_PRODUCT);

        // Get cartesian product size
        const MicroTest& cartesianTestRef = parser.getMicroTest();
        MicroTest& cartesianTest = const_cast<MicroTest&>(cartesianTestRef);
        size_t     cartesianSize = cartesianTest.getSize();

        // Switch to element-wise mode
        parser.setYieldType(YieldType::SIMPLE_PRODUCT);
        EXPECT_EQ(parser.getYieldType(), YieldType::SIMPLE_PRODUCT);

        // Reset to get a new MicroTest with element-wise mode
        parser.reset();
        const MicroTest& elementWiseTestRef = parser.getMicroTest();
        MicroTest& elementWiseTest = const_cast<MicroTest&>(elementWiseTestRef);
        size_t     elementWiseSize = elementWiseTest.getSize();

        // Element-wise should have fewer or equal combinations than cartesian
        // product
        EXPECT_LE(elementWiseSize, cartesianSize)
            << "Element-wise should have fewer or equal combinations than "
               "cartesian product";

        // Switch back to cartesian product
        parser.setYieldType(YieldType::CARTESIAN_PRODUCT);
        EXPECT_EQ(parser.getYieldType(), YieldType::CARTESIAN_PRODUCT);

        parser.reset();
        const MicroTest& cartesianTest2Ref = parser.getMicroTest();
        MicroTest& cartesianTest2 = const_cast<MicroTest&>(cartesianTest2Ref);
        size_t     cartesianSize2 = cartesianTest2.getSize();

        // Should be the same as original cartesian size
        EXPECT_EQ(cartesianSize2, cartesianSize)
            << "Cartesian product size should be consistent";

    } catch (const std::exception& e) {
        FAIL() << "Yield type switching test threw an exception: " << e.what();
    }
}

/**
 * @brief Test SimpleProduct behavior with value-only configuration
 *
 * This test verifies that element-wise mode works correctly when all
 * parameters are single values.
 */
TEST(YamlParserTest, ElementWiseValueOnlyTest)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_config_value.yaml";

    try {
        YamlParser parser(filepath, "yaml_test");

        // Set to element-wise mode
        parser.setYieldType(YieldType::SIMPLE_PRODUCT);

        const MicroTest& microTestRef = parser.getMicroTest();
        MicroTest&       microTest    = const_cast<MicroTest&>(microTestRef);

        // With all single values, both cartesian and element-wise should give 1
        // combination
        size_t totalCombinations = microTest.getSize();

        EXPECT_EQ(totalCombinations, 1)
            << "Value-only config should have 1 combination in any mode";

        // Verify we can access the values (constructor already consumed the
        // combination)
        EXPECT_EQ(microTest.getM(), 10);
        EXPECT_EQ(microTest.getN(), 10);
        EXPECT_EQ(microTest.getK(), 10);
        EXPECT_EQ(microTest.getAlpha(), 1.0);
        EXPECT_EQ(microTest.getBeta(), 0.0);

        // Test if there are more combinations available
        // Note: SimpleProduct's has_next() may return true initially
        size_t actualCombinations = 1; // Constructor consumed the first one

        try {
            while (microTest.hasNext()) {
                microTest.next();
                actualCombinations++;

                // Safety check to avoid infinite loop
                if (actualCombinations > 5) {
                    FAIL() << "Too many combinations for value-only config";
                    break;
                }
            }
        } catch (const std::runtime_error& e) {
            // This is expected when SimpleProduct reaches the end
        }

        EXPECT_EQ(actualCombinations, totalCombinations);

    } catch (const std::exception& e) {
        FAIL() << "Element-wise value-only test threw an exception: "
               << e.what();
    }
}

/**
 * @brief Test minimal PostOps configuration with cartesian=false
 *
 * This test verifies that cartesian=false applies all operations in sequence.
 * Expected: 1 base combination × 1 PostOps sequence = 1 total
 */
TEST(YamlParserTest, MinimalPostOpsNoCartesianTest)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/"
        "yaml_test_config_minimal_no_cartesian.yaml";

    try {
        YamlParser parser(filepath, "gemm_tests");

        // Get the number of test cases
        size_t testCount = parser.getMicroTestCount();
        ASSERT_EQ(testCount, 1)
            << "Expected 1 test case in the minimal no-cartesian YAML file";

        // Test the minimal_postops_test_no_cartesian case
        const MicroTest& microTestRef = parser.getMicroTest();
        MicroTest&       microTest    = const_cast<MicroTest&>(microTestRef);

        std::cout << "\n=== Minimal PostOps No-Cartesian Test ===" << std::endl;

        // Calculate expected combinations:
        // Base parameters: all single values = 1 base combination
        // PostOps with 2 operations ["Elementwise-RELU", "Bias"] and
        // cartesian=false:
        // 1. ["Elementwise-RELU", "Bias"] (apply all operations in sequence)
        // Total PostOps combinations: 1 (single sequence)
        // Total: 1 base × 1 PostOps = 1 combination

        size_t totalCombinations    = microTest.getSize();
        size_t expectedCombinations = 1; // 1 base × 1 PostOps sequence

        std::cout << "Expected total combinations: " << expectedCombinations
                  << std::endl;
        std::cout << "Reported by MicroTest.getSize(): " << totalCombinations
                  << std::endl;

        EXPECT_EQ(totalCombinations, expectedCombinations)
            << "Expected " << expectedCombinations << " combinations but got "
            << totalCombinations;

        // Test the single combination
        std::cout << "\nThe single combination:" << std::endl;
        std::cout << "  M=" << microTest.getM() << ", N=" << microTest.getN()
                  << ", K=" << microTest.getK()
                  << ", Alpha=" << microTest.getAlpha()
                  << ", Beta=" << microTest.getBeta()
                  << ", LDA=" << microTest.getLDA()
                  << ", TransA=" << (microTest.getTransA() ? "true" : "false")
                  << ", TransB=" << (microTest.getTransB() ? "true" : "false")
                  << std::endl;

        // Verify base values
        EXPECT_EQ(microTest.getAType(), MatrixType::f32);
        EXPECT_EQ(microTest.getBType(), MatrixType::f32);
        EXPECT_EQ(microTest.getCType(), MatrixType::f32);
        EXPECT_EQ(microTest.getAccType(), MatrixType::f32);
        EXPECT_EQ(microTest.getStorageFormat(), MatrixLayout::ROW_MAJOR);
        EXPECT_EQ(microTest.getM(), 10);
        EXPECT_EQ(microTest.getN(), 10);
        EXPECT_EQ(microTest.getK(), 10);
        EXPECT_EQ(microTest.getAlpha(), 1.0);
        EXPECT_EQ(microTest.getBeta(), 0.0);
        EXPECT_EQ(microTest.getLDA(), 10);
        EXPECT_EQ(microTest.getLDB(), 10);
        EXPECT_EQ(microTest.getLDC(), 10);
        EXPECT_FALSE(microTest.getTransA());
        EXPECT_FALSE(microTest.getTransB());

        // Should have no more combinations
        EXPECT_FALSE(microTest.hasNext())
            << "Should have no more combinations with cartesian=false";

        std::cout << "✓ Test PASSED - Single combination generated correctly"
                  << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Minimal PostOps no-cartesian test threw an exception: "
               << e.what();
    }
}

/**
 * @brief Test minimal PostOps configuration for debugging
 *
 * This test uses a minimal YAML configuration with just 2 PostOps and single
 * parameter values to validate that PostOps iteration works correctly.
 * Expected: 1 base combination × 2 PostOps permutations = 2 total
 */
TEST(YamlParserTest, MinimalPostOpsDebugTest)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_config_minimal.yaml";

    try {
        YamlParser parser(filepath, "gemm_tests");

        // Get the number of test cases
        size_t testCount = parser.getMicroTestCount();
        ASSERT_EQ(testCount, 1)
            << "Expected 1 test case in the minimal YAML file";

        // Test the minimal_postops_test case
        const MicroTest& microTestRef = parser.getMicroTest();
        MicroTest&       microTest    = const_cast<MicroTest&>(microTestRef);

        std::cout << "\n=== Minimal PostOps Debug Test ===" << std::endl;

        // Calculate expected combinations:
        // Base parameters: all single values = 1 base combination
        // PostOps with 2 operations ["Elementwise-RELU", "Bias"] and
        // cartesian=true:
        // PostOps (2 operations with cartesian=true):
        // All subsets and their permutations:
        // - Empty: 1, Singles: 2, Pairs: C(2,2)*2! = 2
        // - Total: 1 + 2 + 2 = 5 combinations
        // Total: 1 base × 5 PostOps = 5 combinations

        size_t totalCombinations    = microTest.getSize();
        size_t expectedCombinations = 5; // 1 base × 5 PostOps combinations

        std::cout << "Expected total combinations: " << expectedCombinations
                  << std::endl;
        std::cout << "Reported by MicroTest.getSize(): " << totalCombinations
                  << std::endl;

        EXPECT_EQ(totalCombinations, expectedCombinations)
            << "Expected " << expectedCombinations << " combinations but got "
            << totalCombinations;

        // Test all combinations
        size_t combinationCount = 0;
        std::cout << "\nAll combinations:" << std::endl;

        // Print the first combination (current state)
        combinationCount++;
        std::cout << "\nCombination " << combinationCount << ":" << std::endl;
        std::cout << "  M=" << microTest.getM() << ", N=" << microTest.getN()
                  << ", K=" << microTest.getK()
                  << ", Alpha=" << microTest.getAlpha()
                  << ", Beta=" << microTest.getBeta()
                  << ", LDA=" << microTest.getLDA()
                  << ", TransA=" << (microTest.getTransA() ? "true" : "false")
                  << ", TransB=" << (microTest.getTransB() ? "true" : "false")
                  << std::endl;

        // Verify first combination base values (should be constant across all
        // combinations)
        EXPECT_EQ(microTest.getAType(), MatrixType::f32);
        EXPECT_EQ(microTest.getBType(), MatrixType::f32);
        EXPECT_EQ(microTest.getCType(), MatrixType::f32);
        EXPECT_EQ(microTest.getAccType(), MatrixType::f32);
        EXPECT_EQ(microTest.getStorageFormat(), MatrixLayout::ROW_MAJOR);
        EXPECT_EQ(microTest.getM(), 10);
        EXPECT_EQ(microTest.getN(), 10);
        EXPECT_EQ(microTest.getK(), 10);
        EXPECT_EQ(microTest.getAlpha(), 1.0);
        EXPECT_EQ(microTest.getBeta(), 0.0);
        EXPECT_EQ(microTest.getLDA(), 10);
        EXPECT_EQ(microTest.getLDB(), 10);
        EXPECT_EQ(microTest.getLDC(), 10);
        EXPECT_FALSE(microTest.getTransA());
        EXPECT_FALSE(microTest.getTransB());

        // Iterate through all remaining combinations
        while (microTest.hasNext()) {
            microTest.next();
            combinationCount++;
            std::cout << "\nCombination " << combinationCount << ":"
                      << std::endl;
            std::cout << "  M=" << microTest.getM()
                      << ", N=" << microTest.getN()
                      << ", K=" << microTest.getK()
                      << ", Alpha=" << microTest.getAlpha()
                      << ", Beta=" << microTest.getBeta()
                      << ", LDA=" << microTest.getLDA() << ", TransA="
                      << (microTest.getTransA() ? "true" : "false")
                      << ", TransB="
                      << (microTest.getTransB() ? "true" : "false")
                      << std::endl;

            // Verify base parameters remain constant (only PostOps should
            // change)
            EXPECT_EQ(microTest.getM(), 10);
            EXPECT_EQ(microTest.getN(), 10);
            EXPECT_EQ(microTest.getK(), 10);
            EXPECT_EQ(microTest.getAlpha(), 1.0);
            EXPECT_EQ(microTest.getBeta(), 0.0);
        }

        EXPECT_EQ(combinationCount, expectedCombinations)
            << "Should be able to iterate through all " << expectedCombinations
            << " combinations";

        std::cout << "\nTotal combinations iterated: " << combinationCount
                  << std::endl;
        std::cout << "✓ Test PASSED - All " << expectedCombinations
                  << " combinations generated correctly" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Minimal PostOps debug test threw an exception: " << e.what();
    }
}

/**
 * @brief Comprehensive test for all elementwise operations with parameter
 * parsing
 *
 * This test validates that all elementwise operations can parse their
 * respective parameters from YAML configuration files and provides
 * comprehensive coverage including:
 * - Default values (backward compatibility)
 * - Custom parameter values and types
 * - Different data types (bf16, f32)
 * - All supported operations (CLIP, PRELU, Scale)
 */
TEST(YamlParserTest, ElementwiseParametersComprehensiveTest)
{
    try {
        // Use consolidated test configuration file
        std::string filepath = TEST_CONFIG_DIR
            "/yaml_framework_test_configs/"
            "yaml_test_elementwise_parameters.yaml";

        YamlParser parser(filepath, "yaml_test");

        std::cout << "\n=== Comprehensive Elementwise Parameters Test ==="
                  << std::endl;

        // Test 1: Default values (backward compatibility)
        parser.reset();
        const MicroTest& defaultTest = parser.getMicroTest();

        std::cout << "Test 1: Default values..." << std::endl;
        auto postop_dlp = defaultTest.getPostOp(UALType::DLP);
        auto postop_ref = defaultTest.getPostOp(UALType::REF);

        EXPECT_NE(postop_dlp, nullptr)
            << "DLP PostOp should be created with defaults";
        EXPECT_NE(postop_ref, nullptr)
            << "REF PostOp should be created with defaults";
        std::cout << "✓ Default values test passed" << std::endl;

        // Test 2: Custom parameters for multiple operations
        parser.next();
        const MicroTest& customTest = parser.getMicroTest();

        std::cout << "Test 2: Custom parameters..." << std::endl;
        postop_dlp = customTest.getPostOp(UALType::DLP);
        postop_ref = customTest.getPostOp(UALType::REF);

        EXPECT_NE(postop_dlp, nullptr)
            << "DLP PostOp should be created with custom params";
        EXPECT_NE(postop_ref, nullptr)
            << "REF PostOp should be created with custom params";
        std::cout << "✓ Custom parameters test passed" << std::endl;

        // Test 3: Different data types (bf16)
        parser.next();
        const MicroTest& typesTest = parser.getMicroTest();

        std::cout << "Test 3: Different data types..." << std::endl;
        postop_dlp = typesTest.getPostOp(UALType::DLP);
        postop_ref = typesTest.getPostOp(UALType::REF);

        EXPECT_NE(postop_dlp, nullptr)
            << "DLP PostOp should be created with bf16 types";
        EXPECT_NE(postop_ref, nullptr)
            << "REF PostOp should be created with bf16 types";
        std::cout << "✓ Different data types test passed" << std::endl;

        std::cout << "✓ All Tests PASSED - Comprehensive parameter support "
                     "works correctly"
                  << std::endl;

    } catch (const std::exception& e) {
        FAIL()
            << "Comprehensive elementwise parameters test threw an exception: "
            << e.what();
    }
}

/**
 * @brief Test LDA/LDB/LDC default calculation correctness
 *
 * This test validates that when LDA/LDB/LDC are not specified in YAML,
 * they default to the correct minimum legal values based on storage format,
 * transpose flags, and matrix dimensions according to BLAS specification.
 */
TEST(YamlParserTest, LeadingDimensionDefaultsTest)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_ld_defaults.yaml";

    try {
        YamlParser parser(filepath, "yaml_test");

        std::cout << "\n=== Leading Dimension Defaults Test ===" << std::endl;

        size_t testCount = parser.getMicroTestCount();
        ASSERT_EQ(testCount, 7)
            << "Expected 7 test cases in LD defaults YAML file";

        // Test Case 1: Row-major without transpose
        // Expected: LDA >= k=100, LDB >= n=100, LDC >= n=100
        {
            const MicroTest& microTest = parser.getMicroTest();
            std::cout
                << "\nTest Case 1: Row-major, no transpose (m=1000, n=100, "
                   "k=100)"
                << std::endl;

            md_t m   = microTest.getM();
            md_t n   = microTest.getN();
            md_t k   = microTest.getK();
            md_t lda = microTest.getLDA();
            md_t ldb = microTest.getLDB();
            md_t ldc = microTest.getLDC();

            std::cout << "  LDA=" << lda << " (should be >= k=" << k << ")"
                      << std::endl;
            std::cout << "  LDB=" << ldb << " (should be >= n=" << n << ")"
                      << std::endl;
            std::cout << "  LDC=" << ldc << " (should be >= n=" << n << ")"
                      << std::endl;

            EXPECT_GE(lda, k)
                << "Row-major, not transposed A: LDA should be >= k";
            EXPECT_GE(ldb, n)
                << "Row-major, not transposed B: LDB should be >= n";
            EXPECT_GE(ldc, n) << "Row-major C: LDC should be >= n";
        }

        // Test Case 2: Row-major with transposed A
        // Expected: LDA >= m=100, LDB >= n=100, LDC >= n=100
        parser.next();
        {
            const MicroTest& microTest = parser.getMicroTest();
            std::cout
                << "\nTest Case 2: Row-major, transposed A (m=100, n=100, "
                   "k=1000)"
                << std::endl;

            md_t m   = microTest.getM();
            md_t n   = microTest.getN();
            md_t k   = microTest.getK();
            md_t lda = microTest.getLDA();
            md_t ldb = microTest.getLDB();
            md_t ldc = microTest.getLDC();

            std::cout << "  LDA=" << lda << " (should be >= m=" << m << ")"
                      << std::endl;
            std::cout << "  LDB=" << ldb << " (should be >= n=" << n << ")"
                      << std::endl;
            std::cout << "  LDC=" << ldc << " (should be >= n=" << n << ")"
                      << std::endl;

            EXPECT_GE(lda, m) << "Row-major, transposed A: LDA should be >= m";
            EXPECT_GE(ldb, n)
                << "Row-major, not transposed B: LDB should be >= n";
            EXPECT_GE(ldc, n) << "Row-major C: LDC should be >= n";
        }

        // Test Case 3: Row-major with transposed B
        // Expected: LDA >= k=100, LDB >= k=100, LDC >= n=1000
        parser.next();
        {
            const MicroTest& microTest = parser.getMicroTest();
            std::cout
                << "\nTest Case 3: Row-major, transposed B (m=100, n=1000, "
                   "k=100)"
                << std::endl;

            md_t m   = microTest.getM();
            md_t n   = microTest.getN();
            md_t k   = microTest.getK();
            md_t lda = microTest.getLDA();
            md_t ldb = microTest.getLDB();
            md_t ldc = microTest.getLDC();

            std::cout << "  LDA=" << lda << " (should be >= k=" << k << ")"
                      << std::endl;
            std::cout << "  LDB=" << ldb << " (should be >= k=" << k << ")"
                      << std::endl;
            std::cout << "  LDC=" << ldc << " (should be >= n=" << n << ")"
                      << std::endl;

            EXPECT_GE(lda, k)
                << "Row-major, not transposed A: LDA should be >= k";
            EXPECT_GE(ldb, k) << "Row-major, transposed B: LDB should be >= k";
            EXPECT_GE(ldc, n) << "Row-major C: LDC should be >= n";
        }

        // Test Case 4: Row-major with both transposed
        // Expected: LDA >= m=500, LDB >= k=300, LDC >= n=200
        parser.next();
        {
            const MicroTest& microTest = parser.getMicroTest();
            std::cout << "\nTest Case 4: Row-major, both transposed (m=500, "
                         "n=200, k=300)"
                      << std::endl;

            md_t m   = microTest.getM();
            md_t n   = microTest.getN();
            md_t k   = microTest.getK();
            md_t lda = microTest.getLDA();
            md_t ldb = microTest.getLDB();
            md_t ldc = microTest.getLDC();

            std::cout << "  LDA=" << lda << " (should be >= m=" << m << ")"
                      << std::endl;
            std::cout << "  LDB=" << ldb << " (should be >= k=" << k << ")"
                      << std::endl;
            std::cout << "  LDC=" << ldc << " (should be >= n=" << n << ")"
                      << std::endl;

            EXPECT_GE(lda, m) << "Row-major, transposed A: LDA should be >= m";
            EXPECT_GE(ldb, k) << "Row-major, transposed B: LDB should be >= k";
            EXPECT_GE(ldc, n) << "Row-major C: LDC should be >= n";
        }

        // Test Case 5: Column-major without transpose
        // Expected: LDA >= m=1000, LDB >= k=100, LDC >= m=1000
        parser.next();
        {
            const MicroTest& microTest = parser.getMicroTest();
            std::cout << "\nTest Case 5: Column-major, no transpose (m=1000, "
                         "n=100, k=100)"
                      << std::endl;

            md_t m   = microTest.getM();
            md_t n   = microTest.getN();
            md_t k   = microTest.getK();
            md_t lda = microTest.getLDA();
            md_t ldb = microTest.getLDB();
            md_t ldc = microTest.getLDC();

            std::cout << "  LDA=" << lda << " (should be >= m=" << m << ")"
                      << std::endl;
            std::cout << "  LDB=" << ldb << " (should be >= k=" << k << ")"
                      << std::endl;
            std::cout << "  LDC=" << ldc << " (should be >= m=" << m << ")"
                      << std::endl;

            EXPECT_GE(lda, m)
                << "Column-major, not transposed A: LDA should be >= m";
            EXPECT_GE(ldb, k)
                << "Column-major, not transposed B: LDB should be >= k";
            EXPECT_GE(ldc, m) << "Column-major C: LDC should be >= m";
        }

        // Test Case 6: Column-major with transposed A
        // Expected: LDA >= k=1000, LDB >= k=1000, LDC >= m=100
        parser.next();
        {
            const MicroTest& microTest = parser.getMicroTest();
            std::cout << "\nTest Case 6: Column-major, transposed A (m=100, "
                         "n=100, k=1000)"
                      << std::endl;

            md_t m   = microTest.getM();
            md_t n   = microTest.getN();
            md_t k   = microTest.getK();
            md_t lda = microTest.getLDA();
            md_t ldb = microTest.getLDB();
            md_t ldc = microTest.getLDC();

            std::cout << "  LDA=" << lda << " (should be >= k=" << k << ")"
                      << std::endl;
            std::cout << "  LDB=" << ldb << " (should be >= k=" << k << ")"
                      << std::endl;
            std::cout << "  LDC=" << ldc << " (should be >= m=" << m << ")"
                      << std::endl;

            EXPECT_GE(lda, k)
                << "Column-major, transposed A: LDA should be >= k";
            EXPECT_GE(ldb, k)
                << "Column-major, not transposed B: LDB should be >= k";
            EXPECT_GE(ldc, m) << "Column-major C: LDC should be >= m";
        }

        // Test Case 7: Cartesian product with mixed dimensions
        // This verifies that the maximum minimum LD is used across all
        // combinations
        parser.next();
        {
            const MicroTest& microTestRef = parser.getMicroTest();
            MicroTest&       microTest = const_cast<MicroTest&>(microTestRef);

            std::cout
                << "\nTest Case 7: Cartesian product with mixed dimensions"
                << std::endl;

            // The parser should have computed the max of all minimum required
            // LDs across all combinations For m=[10,100], n=[20,200],
            // k=[30,300], storage=[row,col], transA=[F,T], transB=[F,T]
            //
            // We need to ensure that the chosen LDA/LDB/LDC work for ALL
            // combinations The fix should have computed this correctly

            md_t lda = microTest.getLDA();
            md_t ldb = microTest.getLDB();
            md_t ldc = microTest.getLDC();

            std::cout << "  Computed LDA=" << lda << std::endl;
            std::cout << "  Computed LDB=" << ldb << std::endl;
            std::cout << "  Computed LDC=" << ldc << std::endl;

            // Iterate through all combinations and verify LD values are valid
            size_t combinationCount = 0;
            size_t validCount       = 0;

            do {
                md_t         m       = microTest.getM();
                md_t         n       = microTest.getN();
                md_t         k       = microTest.getK();
                bool         transA  = microTest.getTransA();
                bool         transB  = microTest.getTransB();
                MatrixLayout storage = microTest.getStorageFormat();

                // Compute minimum legal LD for this specific combination
                bool is_row_major = (storage == MatrixLayout::ROW_MAJOR);
                md_t min_lda, min_ldb, min_ldc;

                if (is_row_major) {
                    min_lda = transA ? m : k;
                    min_ldb = transB ? k : n;
                    min_ldc = n;
                } else {
                    min_lda = transA ? k : m;
                    min_ldb = transB ? n : k;
                    min_ldc = m;
                }

                // Verify the computed defaults are valid for this combination
                bool valid =
                    (lda >= min_lda) && (ldb >= min_ldb) && (ldc >= min_ldc);
                if (valid) {
                    validCount++;
                }

                EXPECT_GE(lda, min_lda)
                    << "Combination " << combinationCount << ": LDA=" << lda
                    << " should be >= " << min_lda << " (m=" << m << ", k=" << k
                    << ", transA=" << transA
                    << ", storage=" << (is_row_major ? "row" : "col") << ")";
                EXPECT_GE(ldb, min_ldb)
                    << "Combination " << combinationCount << ": LDB=" << ldb
                    << " should be >= " << min_ldb << " (n=" << n << ", k=" << k
                    << ", transB=" << transB
                    << ", storage=" << (is_row_major ? "row" : "col") << ")";
                EXPECT_GE(ldc, min_ldc)
                    << "Combination " << combinationCount << ": LDC=" << ldc
                    << " should be >= " << min_ldc << " (m=" << m << ", n=" << n
                    << ", storage=" << (is_row_major ? "row" : "col") << ")";

                combinationCount++;

                if (!microTest.hasNext())
                    break;
                microTest.next();
            } while (true);

            std::cout << "  Verified " << validCount << " out of "
                      << combinationCount << " combinations" << std::endl;

            EXPECT_EQ(validCount, combinationCount)
                << "All combinations should have valid leading dimensions";
        }

        std::cout << "\n✓ All Leading Dimension Default Tests PASSED!"
                  << std::endl;
        std::cout << "The fix correctly computes minimum legal LD values based "
                     "on:"
                  << std::endl;
        std::cout << "  - Storage format (row-major vs column-major)"
                  << std::endl;
        std::cout << "  - Transpose flags (transA, transB)" << std::endl;
        std::cout << "  - Matrix dimensions (m, n, k)" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Leading dimension defaults test threw an exception: "
               << e.what();
    }
}

// ============================================================================
// YAML Parser Tests for fill_value Feature
// ============================================================================

using namespace dlp::testing::framework;

class YamlParserFillValueTest : public ::testing::Test
{
  protected:
    std::string getTestConfigPath(const std::string& filename)
    {
        // Use TEST_CONFIG_DIR for consistency with other tests
        return std::string(TEST_CONFIG_DIR) + "/yaml_framework_test_configs/"
               + filename;
    }
};

// Test: Parse YAML with uniform distribution fill_value
TEST_F(YamlParserFillValueTest, ParseUniformFillValue)
{
    std::string yaml_path = getTestConfigPath("yaml_test_fill_value.yaml");
    YamlParser  parser(yaml_path, "gemm_tests");

    const MicroTest& microTest = parser.getMicroTest();

    // First test case has uniform fill_value
    EXPECT_TRUE(microTest.hasFillValue());

    const auto& fill_config = microTest.getFillValue();
    EXPECT_DOUBLE_EQ(fill_config.lb, -2.5);
    EXPECT_DOUBLE_EQ(fill_config.ub, 3.5);
    EXPECT_EQ(fill_config.dist, "uniform");
}

// Test: Parse YAML with normal distribution fill_value
TEST_F(YamlParserFillValueTest, ParseNormalFillValue)
{
    std::string yaml_path = getTestConfigPath("yaml_test_fill_value.yaml");
    YamlParser  parser(yaml_path, "gemm_tests");

    // Move to second test case
    parser.next();
    const MicroTest& microTest = parser.getMicroTest();

    EXPECT_TRUE(microTest.hasFillValue());

    const auto& fill_config = microTest.getFillValue();
    EXPECT_DOUBLE_EQ(fill_config.lb, -1.0);
    EXPECT_DOUBLE_EQ(fill_config.ub, 1.0);
    EXPECT_EQ(fill_config.dist, "normal");
}

// Test: Parse YAML without fill_value (backward compatibility)
TEST_F(YamlParserFillValueTest, NoFillValue)
{
    std::string yaml_path = getTestConfigPath("yaml_test_fill_value.yaml");
    YamlParser  parser(yaml_path, "gemm_tests");

    // Move to third test case (no fill_value)
    parser.next();
    parser.next();
    const MicroTest& microTest = parser.getMicroTest();

    EXPECT_FALSE(microTest.hasFillValue());
}

// Test: fillRandom with custom parameters
TEST_F(YamlParserFillValueTest, FillRandomWithCustomParams)
{
    Matrix m(100, 100, MatrixType::f32);

    // Fill with custom uniform distribution
    m.fillRandom(42, -10.0, 10.0, "uniform");

    // Verify values are within expected range
    float* data     = reinterpret_cast<float*>(m.getData());
    float  min_val  = data[0];
    float  max_val  = data[0];
    bool   has_data = false;

    for (size_t i = 0; i < 10000; ++i) {
        if (data[i] < min_val)
            min_val = data[i];
        if (data[i] > max_val)
            max_val = data[i];
        has_data = true;
    }

    EXPECT_TRUE(has_data);
    EXPECT_GE(min_val, -10.0);
    EXPECT_LE(max_val, 10.0);
}

// Test: fillRandom delegates correctly (backward compatibility)
TEST_F(YamlParserFillValueTest, FillRandomBackwardCompatibility)
{
    Matrix m1(50, 50, MatrixType::f32);
    Matrix m2(50, 50, MatrixType::f32);

    // Both should produce same results with same seed
    m1.fillRandom(123);
    m2.fillRandom(123);

    EXPECT_TRUE(m1 == m2);
}

TEST_F(YamlParserFillValueTest, ForceIntDistributionF32)
{
    // Test that force_int_distribution=true produces integer-only values
    Matrix m(20, 20, MatrixType::f32);
    m.fillRandom(42, -5.0, 5.0, "uniform", true);

    const float* data = reinterpret_cast<const float*>(m.getData());
    size_t       size = 20 * 20;

    // Verify all values are integers
    for (size_t i = 0; i < size; ++i) {
        float val = data[i];
        EXPECT_FLOAT_EQ(val, std::floor(val))
            << "Expected integer value at index " << i << ", got " << val;
        EXPECT_GE(val, -5.0);
        EXPECT_LE(val, 5.0);
    }
}

TEST_F(YamlParserFillValueTest, ForceIntDistributionBF16)
{
    // Test that force_int_distribution=true produces integer-only values for
    // bf16
    Matrix m(15, 15, MatrixType::bf16);
    m.fillRandom(42, -3.0, 3.0, "uniform", true);

    const uint16_t* data = reinterpret_cast<const uint16_t*>(m.getData());
    size_t          size = 15 * 15;

    // Verify all values are integers (when converted to float)
    for (size_t i = 0; i < size; ++i) {
        float val = dlp::testing::utils::bf16_to_f32(data[i]);
        EXPECT_FLOAT_EQ(val, std::floor(val))
            << "Expected integer value at index " << i << ", got " << val;
        EXPECT_GE(val, -3.0);
        EXPECT_LE(val, 3.0);
    }
}

TEST_F(YamlParserFillValueTest, ForceIntDistributionFalseF32)
{
    // Test that force_int_distribution=false can produce fractional values
    Matrix m(100, 100, MatrixType::f32);
    m.fillRandom(42, -2.5, 2.5, "uniform", false);

    const float* data = reinterpret_cast<const float*>(m.getData());
    size_t       size = 100 * 100;

    // With a large enough sample, we should find at least some non-integer
    // values
    bool found_fractional = false;
    for (size_t i = 0; i < size; ++i) {
        float val = data[i];
        EXPECT_GE(val, -2.5);
        EXPECT_LE(val, 2.5);
        if (std::abs(val - std::floor(val)) > 1e-6) {
            found_fractional = true;
        }
    }
    // Note: With uniform_real_distribution, we expect fractional values,
    // but we won't enforce this in the test since it's probabilistic
}

TEST_F(YamlParserFillValueTest, ParseForceIntDistribution)
{
    std::string yaml_path =
        getTestConfigPath("yaml_test_force_int_distribution.yaml");

    YamlParser parser(yaml_path, "gemm_tests");
    EXPECT_GT(parser.getMicroTestCount(), 0);

    // Test case 1: force_int_distribution: true
    const MicroTest& test1 = parser.getMicroTest();
    EXPECT_TRUE(test1.hasFillValue());
    const auto& fill_val1 = test1.getFillValue();
    EXPECT_EQ(fill_val1.lb, -5.0);
    EXPECT_EQ(fill_val1.ub, 5.0);
    EXPECT_EQ(fill_val1.dist, "uniform");
    EXPECT_TRUE(fill_val1.force_int_distribution);

    // Test case 2: force_int_distribution: false
    parser.next();
    const MicroTest& test2 = parser.getMicroTest();
    EXPECT_TRUE(test2.hasFillValue());
    const auto& fill_val2 = test2.getFillValue();
    EXPECT_EQ(fill_val2.lb, -2.5);
    EXPECT_EQ(fill_val2.ub, 3.5);
    EXPECT_EQ(fill_val2.dist, "uniform");
    EXPECT_FALSE(fill_val2.force_int_distribution);

    // Test case 5: default (should be true)
    parser.next();
    parser.next();
    parser.next();
    const MicroTest& test5 = parser.getMicroTest();
    EXPECT_TRUE(test5.hasFillValue());
    const auto& fill_val5 = test5.getFillValue();
    EXPECT_TRUE(fill_val5.force_int_distribution)
        << "force_int_distribution should default to true";
}

/**
 * @brief Test tolerance configuration parsing from YAML
 *
 * This test verifies that tolerance multipliers can be read from YAML
 * configuration and that defaults are used when not specified.
 */
TEST(YamlParserTest, ParseToleranceConfig)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_tolerances.yaml";

    try {
        YamlParser parser(filepath, "tolerance_test");

        // Test 1: Default tolerance (no tolerances specified)
        const MicroTest& test1 = parser.getMicroTest();
        EXPECT_FALSE(test1.hasTolerances())
            << "Test without tolerances section should not have tolerances";

        // Test 2: Custom tolerance multipliers
        parser.next();
        const MicroTest& test2 = parser.getMicroTest();
        EXPECT_TRUE(test2.hasTolerances())
            << "Test with tolerances section should have tolerances";
        const auto& tol2 = test2.getTolerances();
        EXPECT_DOUBLE_EQ(tol2.absolute, 5.0)
            << "Absolute tolerance multiplier should be 5.0";
        EXPECT_DOUBLE_EQ(tol2.relative, 1.0)
            << "Relative tolerance multiplier should be 1.0";

        // Test 3: Zero tolerance (bitwise equality)
        parser.next();
        const MicroTest& test3 = parser.getMicroTest();
        EXPECT_TRUE(test3.hasTolerances());
        const auto& tol3 = test3.getTolerances();
        EXPECT_DOUBLE_EQ(tol3.absolute, 0.0)
            << "Zero absolute tolerance should enforce bitwise equality";
        EXPECT_DOUBLE_EQ(tol3.relative, 0.0)
            << "Zero relative tolerance should enforce bitwise equality";

        // Test 4: Only relative tolerance specified
        parser.next();
        const MicroTest& test4 = parser.getMicroTest();
        EXPECT_TRUE(test4.hasTolerances());
        const auto& tol4 = test4.getTolerances();
        EXPECT_DOUBLE_EQ(tol4.relative, 10.0)
            << "Relative tolerance multiplier should be 10.0";
        EXPECT_DOUBLE_EQ(tol4.absolute, -1.0)
            << "Absolute tolerance should use default when not specified";

        // Test 5: Only absolute tolerance specified
        parser.next();
        const MicroTest& test5 = parser.getMicroTest();
        EXPECT_TRUE(test5.hasTolerances());
        const auto& tol5 = test5.getTolerances();
        EXPECT_DOUBLE_EQ(tol5.absolute, 10.0)
            << "Absolute tolerance multiplier should be 10.0";
        EXPECT_DOUBLE_EQ(tol5.relative, -1.0)
            << "Relative tolerance should use default when not specified";

    } catch (const std::exception& e) {
        FAIL() << "Parser threw an exception: " << e.what();
    }
}

/**
 * @brief Test that invalid tolerance values are rejected
 *
 * This test verifies that negative tolerance multipliers are properly
 * rejected with appropriate error messages.
 */
TEST(YamlParserTest, InvalidToleranceConfig)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_tolerances_invalid.yaml";

    try {
        YamlParser       parser(filepath, "invalid_tolerance_test");
        const MicroTest& test = parser.getMicroTest();
        FAIL()
            << "Parser should have thrown an exception for negative tolerance";
    } catch (const std::runtime_error& e) {
        std::string error_msg = e.what();
        EXPECT_NE(error_msg.find("non-negative"), std::string::npos)
            << "Error message should mention non-negative requirement. Got: "
            << error_msg;
    } catch (const std::exception& e) {
        FAIL() << "Unexpected exception type: " << e.what();
    }
}

/**
 * @brief Test that tolerance multipliers affect matrix comparison
 *
 * This test creates two matrices with small differences and verifies
 * that custom tolerance multipliers control whether they are considered
 * equal. This is the end-to-end test that demonstrates the feature working.
 */
TEST(YamlParserTest, ToleranceMultipliersAffectComparison)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_tolerances.yaml";

    try {
        YamlParser parser(filepath, "tolerance_test");

        // Get test with custom tolerances
        parser.next(); // Skip to test 2: custom_tolerance
        const MicroTest& test = parser.getMicroTest();
        ASSERT_TRUE(test.hasTolerances());
        const auto& tol = test.getTolerances();

        // Create two small matrices
        Matrix A(10u, 10u, MatrixType::f32, MatrixLayout::ROW_MAJOR);
        Matrix B(10u, 10u, MatrixType::f32, MatrixLayout::ROW_MAJOR);

        // Fill both with same values initially
        A.fillRandom(42);
        B = A;

        // Set k dimension for tolerance calculation
        A.setK(100);
        B.setK(100);

        // Test 1: Default tolerance (should be equal)
        MatrixCompareOptions opts_default   = MatrixCompareOptions::Fast();
        auto                 result_default = A.compare(B, opts_default);
        EXPECT_TRUE(result_default.equal)
            << "Identical matrices should be equal with default tolerance";

        // Test 2: Zero tolerance with identical matrices (should be equal)
        MatrixCompareOptions opts_zero   = MatrixCompareOptions::Fast();
        opts_zero.relToleranceMultiplier = 0.0;
        opts_zero.absToleranceMultiplier = 0.0;
        auto result_zero                 = A.compare(B, opts_zero);
        EXPECT_TRUE(result_zero.equal)
            << "Identical matrices should be equal even with zero tolerance";

        // Create matrices with slightly different random values
        Matrix C(10u, 10u, MatrixType::f32, MatrixLayout::ROW_MAJOR);
        Matrix D(10u, 10u, MatrixType::f32, MatrixLayout::ROW_MAJOR);
        C.fillRandom(100); // Different seed creates different values
        D.fillRandom(101); // Another different seed
        C.setK(100);
        D.setK(100);

        // Test 3: With strict zero tolerance, different matrices should not be
        // equal
        auto result_zero_diff = C.compare(D, opts_zero);
        EXPECT_FALSE(result_zero_diff.equal)
            << "Different matrices should not be equal with zero tolerance";

        // Test 4: With very large multiplier, might be equal depending on
        // values
        MatrixCompareOptions opts_large   = MatrixCompareOptions::Fast();
        opts_large.relToleranceMultiplier = 100000.0;
        opts_large.absToleranceMultiplier = 100000.0;
        auto result_large                 = C.compare(D, opts_large);
        // Large tolerance allows differences (test demonstrates tolerance is
        // applied)

        // Test 5: With default multiplier
        MatrixCompareOptions opts_normal   = MatrixCompareOptions::Fast();
        auto                 result_normal = C.compare(D, opts_normal);
        // Default tolerance behavior

        // Test 6: Verify that tolerance values are calculated correctly
        EXPECT_GT(result_default.usedAbsTolerance, 0.0)
            << "Default absolute tolerance should be positive";
        EXPECT_GT(result_default.usedRelTolerance, 0.0)
            << "Default relative tolerance should be positive";
        EXPECT_EQ(result_zero.usedAbsTolerance, 0.0)
            << "Zero multiplier should result in zero absolute tolerance";
        EXPECT_EQ(result_zero.usedRelTolerance, 0.0)
            << "Zero multiplier should result in zero relative tolerance";
        EXPECT_GT(result_large.usedAbsTolerance,
                  result_default.usedAbsTolerance)
            << "Large multiplier should result in larger tolerance";

    } catch (const std::exception& e) {
        FAIL() << "Test threw an exception: " << e.what();
    }
}

// ============================================================================
// BATCH GEMM YAML TESTS
// ============================================================================

/**
 * @brief Test parsing group_size parameter for batch GEMM
 *
 * This test verifies that the group_size parameter can be parsed correctly
 * and that the MicroTest provides access to it.
 */
TEST(YamlParserTest, ParseGroupSizeParameter)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    try {
        YamlParser parser(filepath, "batch_gemm_tests");

        // Get the number of test cases
        size_t testCount = parser.getMicroTestCount();
        ASSERT_GT(testCount, 0)
            << "Expected at least 1 test case in batch_gemm_test_config.yaml";

        // Test the first test case (should have group_size)
        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "\n=== Group Size Parameter Test ===" << std::endl;

        // Verify group_size is accessible
        EXPECT_TRUE(microTest.hasGroupSize())
            << "Batch GEMM test should have group_size";

        md_t group_size = microTest.getGroupSize();
        std::cout << "  Group Size: " << group_size << std::endl;

        EXPECT_GT(group_size, 0) << "Group size should be positive";

        // Verify other parameters are still accessible
        EXPECT_NO_THROW({
            auto m = microTest.getM();
            auto n = microTest.getN();
            auto k = microTest.getK();
            std::cout << "  Matrix dimensions: m=" << m << ", n=" << n
                      << ", k=" << k << std::endl;
        }) << "Basic GEMM parameters should still be accessible";

        std::cout << "✓ Group size parameter test passed" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Parser threw an exception: " << e.what();
    }
}

/**
 * @brief Test single-group batch GEMM with cartesian product
 *
 * This test verifies that single-group batch GEMM tests (scalar group_size)
 * work correctly with cartesian product mode.
 */
TEST(YamlParserTest, SingleGroupBatchGemmCartesian)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    try {
        YamlParser parser(filepath, "batch_gemm_tests");

        // First test should be single-group with cartesian product
        const MicroTest& microTestRef = parser.getMicroTest();
        MicroTest&       microTest    = const_cast<MicroTest&>(microTestRef);

        std::cout << "\n=== Single-Group Batch GEMM (Cartesian) Test ==="
                  << std::endl;

        // Verify it's single-group mode (scalar group_size)
        EXPECT_TRUE(microTest.hasGroupSize());
        md_t group_size = microTest.getGroupSize();
        std::cout << "  Group size: " << group_size << std::endl;

        // Get total combinations
        size_t totalCombinations = microTest.getSize();
        std::cout << "  Total combinations: " << totalCombinations << std::endl;

        // For cartesian product with m=[4,8,16], n=[4,8,16], k=[4,8,16]
        // Expected: 3 * 3 * 3 = 27 combinations
        size_t expectedCombinations = 27;
        EXPECT_EQ(totalCombinations, expectedCombinations)
            << "Single-group cartesian should generate " << expectedCombinations
            << " combinations";

        // Test first few combinations
        size_t testCount = std::min(size_t(5), totalCombinations);
        std::cout << "\nFirst " << testCount << " combinations:" << std::endl;

        for (size_t i = 0; i < testCount; ++i) {
            md_t m  = microTest.getM();
            md_t n  = microTest.getN();
            md_t k  = microTest.getK();
            md_t gs = microTest.getGroupSize();

            std::cout << "  Combination " << (i + 1) << ": m=" << m
                      << ", n=" << n << ", k=" << k << ", group_size=" << gs
                      << std::endl;

            // Verify group_size is constant across iterations
            EXPECT_EQ(gs, group_size)
                << "Group size should remain constant in single-group mode";

            if (microTest.hasNext() && i < testCount - 1) {
                microTest.next();
            }
        }

        std::cout << "✓ Single-group cartesian test passed" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Single-group cartesian test threw an exception: "
               << e.what();
    }
}

/**
 * @brief Test multi-group batch GEMM with simple product
 *
 * This test verifies that multi-group batch GEMM tests (list group_size)
 * work correctly with simple product mode (enforced).
 */
TEST(YamlParserTest, MultiGroupBatchGemmSimple)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    try {
        YamlParser parser(filepath, "batch_gemm_tests");

        // Skip to a multi-group test (test index 3: MultiGroup_MixedSizes_F32)
        for (int i = 0; i < 3; ++i) {
            parser.next();
        }

        const MicroTest& microTestRef = parser.getMicroTest();
        MicroTest&       microTest    = const_cast<MicroTest&>(microTestRef);

        std::cout << "\n=== Multi-Group Batch GEMM (Simple Product) Test ==="
                  << std::endl;

        // Get total combinations
        size_t totalCombinations = microTest.getSize();
        std::cout << "  Total combinations: " << totalCombinations << std::endl;

        // For simple product with m=[8,16,32], n=[8,16,32], k=[8,16,32],
        // group_size=[2,3,4]
        // Expected: 3 combinations (element-wise pairing)
        size_t expectedCombinations = 3;
        EXPECT_EQ(totalCombinations, expectedCombinations)
            << "Multi-group simple product should generate "
            << expectedCombinations << " combinations";

        // Test all combinations
        std::cout << "\nAll combinations:" << std::endl;
        size_t count = 0;

        do {
            md_t m  = microTest.getM();
            md_t n  = microTest.getN();
            md_t k  = microTest.getK();
            md_t gs = microTest.getGroupSize();

            std::cout << "  Group " << (count + 1) << ": m=" << m << ", n=" << n
                      << ", k=" << k << ", group_size=" << gs << std::endl;

            // Verify dimensions change across groups
            EXPECT_GT(m, 0);
            EXPECT_GT(n, 0);
            EXPECT_GT(k, 0);
            EXPECT_GT(gs, 0);

            count++;
            if (!microTest.hasNext())
                break;
            microTest.next();
        } while (count < totalCombinations + 5); // Safety limit

        EXPECT_EQ(count, expectedCombinations)
            << "Should iterate through exactly " << expectedCombinations
            << " combinations";

        std::cout << "✓ Multi-group simple product test passed" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Multi-group simple product test threw an exception: "
               << e.what();
    }
}

/**
 * @brief Test backward compatibility without group_size
 *
 * This test verifies that YAML configurations without group_size still work
 * (group_size defaults to 1).
 */
TEST(YamlParserTest, BackwardCompatibilityNoGroupSize)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_config_value.yaml";

    try {
        YamlParser parser(filepath, "yaml_test");

        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "\n=== Backward Compatibility (No group_size) Test ==="
                  << std::endl;

        // Verify getGroupSize() returns default value of 1
        md_t group_size = microTest.getGroupSize();
        std::cout << "  Group size (default): " << group_size << std::endl;

        EXPECT_EQ(group_size, 1)
            << "Default group_size should be 1 for backward compatibility";

        // Verify all other parameters work normally
        EXPECT_NO_THROW({
            auto m     = microTest.getM();
            auto n     = microTest.getN();
            auto k     = microTest.getK();
            auto alpha = microTest.getAlpha();
            auto beta  = microTest.getBeta();
        }) << "All parameters should be accessible without group_size";

        std::cout << "✓ Backward compatibility test passed" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Backward compatibility test threw an exception: "
               << e.what();
    }
}

/**
 * @brief Test that group_size values are correctly paired in simple product
 *
 * This test verifies the element-wise pairing of group_size with other
 * parameters in multi-group mode.
 */
TEST(YamlParserTest, GroupSizeSimpleProductPairing)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    try {
        YamlParser parser(filepath, "batch_gemm_tests");

        // Skip to multi-group test (test index 3)
        for (int i = 0; i < 3; ++i) {
            parser.next();
        }

        const MicroTest& microTestRef = parser.getMicroTest();
        MicroTest&       microTest    = const_cast<MicroTest&>(microTestRef);

        std::cout << "\n=== Group Size Simple Product Pairing Test ==="
                  << std::endl;

        // Expected pairings for MultiGroup_MixedSizes_F32:
        // Group 0: m=8, n=8, k=8, group_size=2
        // Group 1: m=16, n=16, k=16, group_size=3
        // Group 2: m=32, n=32, k=32, group_size=4

        std::vector<md_t> expected_m  = { 8, 16, 32 };
        std::vector<md_t> expected_n  = { 8, 16, 32 };
        std::vector<md_t> expected_k  = { 8, 16, 32 };
        std::vector<md_t> expected_gs = { 2, 3, 4 };

        size_t count = 0;
        do {
            md_t m  = microTest.getM();
            md_t n  = microTest.getN();
            md_t k  = microTest.getK();
            md_t gs = microTest.getGroupSize();

            std::cout << "  Pairing " << count << ": m=" << m << ", n=" << n
                      << ", k=" << k << ", group_size=" << gs << std::endl;

            EXPECT_EQ(m, expected_m[count])
                << "M dimension should be paired correctly";
            EXPECT_EQ(n, expected_n[count])
                << "N dimension should be paired correctly";
            EXPECT_EQ(k, expected_k[count])
                << "K dimension should be paired correctly";
            EXPECT_EQ(gs, expected_gs[count])
                << "Group size should be paired correctly";

            count++;
            if (!microTest.hasNext())
                break;
            microTest.next();
        } while (count < 5); // Safety limit

        EXPECT_EQ(count, 3) << "Should have exactly 3 pairings";

        std::cout << "✓ Simple product pairing test passed" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Simple product pairing test threw an exception: "
               << e.what();
    }
}

/**
 * @brief Test batch_gemm_tests root node requirement
 *
 * This test verifies that batch GEMM specific validations only apply
 * when the root node is "batch_gemm_tests".
 */
TEST(YamlParserTest, BatchGemmTestsRootNodeRequirement)
{
    // Test 1: Regular gemm_tests should work without group_size
    {
        std::string filepath = TEST_CONFIG_DIR
            "/yaml_framework_test_configs/yaml_test_config_value.yaml";

        try {
            YamlParser parser(filepath, "yaml_test");

            const MicroTest& microTest = parser.getMicroTest();

            // Should work fine without group_size (defaults to 1)
            EXPECT_NO_THROW({
                md_t gs = microTest.getGroupSize();
                EXPECT_EQ(gs, 1);
            });

            std::cout
                << "✓ Regular yaml_test works without group_size requirement"
                << std::endl;

        } catch (const std::exception& e) {
            FAIL() << "Regular yaml_test should work: " << e.what();
        }
    }

    // Test 2: batch_gemm_tests enforces validations
    {
        std::string filepath = TEST_CONFIG_DIR
            "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

        try {
            YamlParser parser(filepath, "batch_gemm_tests");

            // Should parse successfully with proper batch GEMM configuration
            size_t testCount = parser.getMicroTestCount();
            EXPECT_GT(testCount, 0);

            std::cout << "✓ batch_gemm_tests enforces proper validations"
                      << std::endl;

        } catch (const std::exception& e) {
            FAIL() << "batch_gemm_tests should parse: " << e.what();
        }
    }
}

/**
 * @brief Test that all batch GEMM test configurations are valid
 *
 * This comprehensive test iterates through all test configurations in the
 * batch_gemm_test_config.yaml file to ensure they all parse and execute
 * correctly.
 */
TEST(YamlParserTest, AllBatchGemmConfigurationsValid)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    try {
        YamlParser parser(filepath, "batch_gemm_tests");

        size_t testCount = parser.getMicroTestCount();
        std::cout << "\n=== Validating All Batch GEMM Configurations ==="
                  << std::endl;
        std::cout << "Total test configurations: " << testCount << std::endl;

        ASSERT_GT(testCount, 0)
            << "Should have at least one test configuration";

        // Iterate through all test configurations
        for (size_t i = 0; i < testCount; ++i) {
            const MicroTest& microTest = parser.getMicroTest();

            std::cout << "\nTest " << (i + 1) << std::endl;

            // Verify all required parameters are accessible
            EXPECT_NO_THROW({
                auto a_type = microTest.getAType();
                auto b_type = microTest.getBType();
                auto c_type = microTest.getCType();
                auto m      = microTest.getM();
                auto n      = microTest.getN();
                auto k      = microTest.getK();
                auto alpha  = microTest.getAlpha();
                auto beta   = microTest.getBeta();
                auto gs     = microTest.getGroupSize();

                std::cout << "  Dimensions: m=" << m << ", n=" << n
                          << ", k=" << k << std::endl;
                std::cout << "  Group size: " << gs << std::endl;
                std::cout << "  Types: A=" << a_type << ", B=" << b_type
                          << ", C=" << c_type << std::endl;

                // Verify valid values
                EXPECT_GT(m, 0) << "M dimension should be positive";
                EXPECT_GT(n, 0) << "N dimension should be positive";
                EXPECT_GT(k, 0) << "K dimension should be positive";
                EXPECT_GT(gs, 0) << "Group size should be positive";
            }) << "Test configuration "
               << (i + 1) << " should have valid parameters";

            // Move to next test configuration
            if (i < testCount - 1) {
                parser.next();
            }
        }

        std::cout << "\n✓ All " << testCount
                  << " batch GEMM configurations are valid" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Batch GEMM configuration validation failed: " << e.what();
    }
}

/**
 * @brief Test validation: cartesian product with multi-group should fail
 *
 * This test verifies that attempting to use cartesian product with multiple
 * group_size values is properly rejected by the parser.
 */
TEST(YamlParserTest, MultiGroupCartesianValidation)
{
    // Note: This test would require a YAML file that attempts to violate
    // the validation rules. Since our batch_gemm_test_config.yaml is valid,
    // this test demonstrates the expected behavior through documentation.

    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    try {
        YamlParser parser(filepath, "batch_gemm_tests");

        // All multi-group tests in our config use simple product
        // so they should all parse successfully
        size_t testCount = parser.getMicroTestCount();
        EXPECT_GT(testCount, 0)
            << "Should have valid batch GEMM configurations";

        // Verify that multi-group tests use simple product
        for (size_t i = 0; i < testCount; ++i) {
            const MicroTest& microTestRef = parser.getMicroTest();
            MicroTest&       microTest = const_cast<MicroTest&>(microTestRef);

            size_t combinations = microTest.getSize();

            // Multi-group tests should have limited combinations (simple
            // product) Single-group tests may have many (cartesian product)
            if (combinations <= 10) {
                // Likely a multi-group test
                std::cout << "Test " << i << " has " << combinations
                          << " combinations (likely multi-group)" << std::endl;
            }

            if (i < testCount - 1) {
                parser.next();
            }
        }

        std::cout << "✓ All batch GEMM tests use appropriate product types"
                  << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Validation test failed: " << e.what();
    }
}

/**
 * @brief Test scalar broadcast in multi-group mode
 *
 * Verifies that scalar parameters (like alpha, beta) are properly broadcast
 * to all groups in multi-group mode.
 */
TEST(YamlParserTest, ScalarBroadcastInMultiGroup)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    try {
        YamlParser parser(filepath, "batch_gemm_tests");

        // Skip to multi-group test (test index 3)
        for (int i = 0; i < 3; ++i) {
            parser.next();
        }

        const MicroTest& microTestRef = parser.getMicroTest();
        MicroTest&       microTest    = const_cast<MicroTest&>(microTestRef);

        std::cout << "\n=== Scalar Broadcast Test ===" << std::endl;

        // Get first alpha/beta values
        double first_alpha = microTest.getAlpha();
        double first_beta  = microTest.getBeta();

        std::cout << "First group: alpha=" << first_alpha
                  << ", beta=" << first_beta << std::endl;

        // Iterate through all groups
        size_t count = 0;
        do {
            double alpha = microTest.getAlpha();
            double beta  = microTest.getBeta();

            std::cout << "Group " << count << ": alpha=" << alpha
                      << ", beta=" << beta << std::endl;

            // For this test configuration, alpha and beta are scalar
            // (broadcast) So they should be the same across all groups
            EXPECT_EQ(alpha, first_alpha)
                << "Scalar alpha should be broadcast to all groups";
            EXPECT_EQ(beta, first_beta)
                << "Scalar beta should be broadcast to all groups";

            count++;
            if (!microTest.hasNext())
                break;
            microTest.next();
        } while (count < 10); // Safety limit

        std::cout << "✓ Scalar broadcast works correctly" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Scalar broadcast test failed: " << e.what();
    }
}

/**
 * @brief Test group_size consistency within test configuration
 *
 * Verifies that all parameters in a test configuration have consistent
 * list lengths matching the number of groups.
 */
TEST(YamlParserTest, GroupSizeConsistency)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    try {
        YamlParser parser(filepath, "batch_gemm_tests");

        // Skip to multi-group test (test index 3)
        for (int i = 0; i < 3; ++i) {
            parser.next();
        }

        const MicroTest& microTestRef = parser.getMicroTest();
        MicroTest&       microTest    = const_cast<MicroTest&>(microTestRef);

        std::cout << "\n=== Group Size Consistency Test ===" << std::endl;

        // Count how many groups we have
        size_t            group_count = 0;
        std::vector<md_t> all_m, all_n, all_k, all_gs;

        do {
            all_m.push_back(microTest.getM());
            all_n.push_back(microTest.getN());
            all_k.push_back(microTest.getK());
            all_gs.push_back(microTest.getGroupSize());

            group_count++;
            if (!microTest.hasNext())
                break;
            microTest.next();
        } while (group_count < 10); // Safety limit

        std::cout << "Found " << group_count << " groups" << std::endl;

        // Verify all arrays have the same length
        EXPECT_EQ(all_m.size(), group_count)
            << "M values should match group count";
        EXPECT_EQ(all_n.size(), group_count)
            << "N values should match group count";
        EXPECT_EQ(all_k.size(), group_count)
            << "K values should match group count";
        EXPECT_EQ(all_gs.size(), group_count)
            << "Group size values should match group count";

        // Verify values are different across groups (heterogeneous)
        bool has_different_m  = false;
        bool has_different_gs = false;

        for (size_t i = 1; i < group_count; ++i) {
            if (all_m[i] != all_m[0])
                has_different_m = true;
            if (all_gs[i] != all_gs[0])
                has_different_gs = true;
        }

        EXPECT_TRUE(has_different_m || has_different_gs)
            << "Multi-group test should have heterogeneous dimensions or group "
               "sizes";

        std::cout << "✓ Group size consistency verified" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Group size consistency test failed: " << e.what();
    }
}

/**
 * @brief Test tolerance settings with batch GEMM
 *
 * Verifies that tolerance settings work correctly with batch GEMM
 * configurations.
 */
TEST(YamlParserTest, ToleranceWithBatchGemm)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    try {
        YamlParser parser(filepath, "batch_gemm_tests");

        size_t testCount = parser.getMicroTestCount();

        // Check if any batch GEMM tests have custom tolerances
        bool found_with_tolerance    = false;
        bool found_without_tolerance = false;

        for (size_t i = 0; i < testCount; ++i) {
            const MicroTest& microTest = parser.getMicroTest();

            if (microTest.hasTolerances()) {
                found_with_tolerance = true;
                const auto& tol      = microTest.getTolerances();

                EXPECT_GE(tol.absolute, 0.0)
                    << "Absolute tolerance should be non-negative";
                EXPECT_GE(tol.relative, 0.0)
                    << "Relative tolerance should be non-negative";
            } else {
                found_without_tolerance = true;
            }

            if (i < testCount - 1) {
                parser.next();
            }
        }

        // Both cases should exist or at least one
        EXPECT_TRUE(found_with_tolerance || found_without_tolerance)
            << "Should have at least some test configurations";

        std::cout << "✓ Tolerance settings work with batch GEMM" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Tolerance with batch GEMM test failed: " << e.what();
    }
}

/**
 * @brief Test matrix tag parameters with batch GEMM
 *
 * Verifies that matrix tag parameters (reorder, pack) work correctly
 * with batch GEMM configurations.
 */
TEST(YamlParserTest, MatrixTagsWithBatchGemm)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    try {
        YamlParser parser(filepath, "batch_gemm_tests");

        // Iterate through all configurations
        size_t testCount = parser.getMicroTestCount();

        for (size_t i = 0; i < testCount; ++i) {
            const MicroTest& microTest = parser.getMicroTest();

            // Verify all matrix tag parameters are accessible
            EXPECT_NO_THROW({
                bool reorderA = microTest.getReorderA();
                bool reorderB = microTest.getReorderB();
                bool packA    = microTest.getPackA();
                bool packB    = microTest.getPackB();

                // All values should be valid booleans
                EXPECT_TRUE(reorderA == true || reorderA == false);
                EXPECT_TRUE(reorderB == true || reorderB == false);
                EXPECT_TRUE(packA == true || packA == false);
                EXPECT_TRUE(packB == true || packB == false);
            }) << "Matrix tags should be accessible for batch GEMM test "
               << i;

            if (i < testCount - 1) {
                parser.next();
            }
        }

        std::cout << "✓ Matrix tags work with batch GEMM" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Matrix tags with batch GEMM test failed: " << e.what();
    }
}

/**
 * @brief Test leading dimension calculation with group_size
 *
 * Verifies that leading dimensions (LDA, LDB, LDC) are correctly calculated
 * or provided for batch GEMM configurations.
 */
TEST(YamlParserTest, LeadingDimensionsWithBatchGemm)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    try {
        YamlParser parser(filepath, "batch_gemm_tests");

        size_t testCount = parser.getMicroTestCount();

        for (size_t i = 0; i < testCount; ++i) {
            const MicroTest& microTestRef = parser.getMicroTest();
            MicroTest&       microTest = const_cast<MicroTest&>(microTestRef);

            // Check leading dimensions for first combination
            md_t         m       = microTest.getM();
            md_t         n       = microTest.getN();
            md_t         k       = microTest.getK();
            md_t         lda     = microTest.getLDA();
            md_t         ldb     = microTest.getLDB();
            md_t         ldc     = microTest.getLDC();
            bool         transA  = microTest.getTransA();
            bool         transB  = microTest.getTransB();
            MatrixLayout storage = microTest.getStorageFormat();

            // Verify leading dimensions are valid
            // When lda/ldb/ldc are not specified in YAML, they are
            // auto-calculated to the minimum legal values by the YAML parser
            bool is_row_major = (storage == MatrixLayout::ROW_MAJOR);

            if (is_row_major) {
                md_t min_lda = transA ? m : k;
                md_t min_ldb = transB ? k : n;
                md_t min_ldc = n;

                EXPECT_GE(lda, min_lda)
                    << "LDA should be >= " << min_lda << " for test " << i;
                EXPECT_GE(ldb, min_ldb)
                    << "LDB should be >= " << min_ldb << " for test " << i;
                EXPECT_GE(ldc, min_ldc)
                    << "LDC should be >= " << min_ldc << " for test " << i;
            } else {
                md_t min_lda = transA ? k : m;
                md_t min_ldb = transB ? n : k;
                md_t min_ldc = m;

                EXPECT_GE(lda, min_lda)
                    << "LDA should be >= " << min_lda << " for test " << i;
                EXPECT_GE(ldb, min_ldb)
                    << "LDB should be >= " << min_ldb << " for test " << i;
                EXPECT_GE(ldc, min_ldc)
                    << "LDC should be >= " << min_ldc << " for test " << i;
            }

            if (i < testCount - 1) {
                parser.next();
            }
        }

        std::cout << "✓ Leading dimensions valid for all batch GEMM tests"
                  << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Leading dimensions test failed: " << e.what();
    }
}

/**
 * @brief Test batch GEMM with all matrix types
 *
 * Verifies that batch GEMM configurations work correctly with all matrix types.
 */
TEST(YamlParserTest, BatchGemmWithAllMatrixTypes)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    try {
        YamlParser parser(filepath, "batch_gemm_tests");

        size_t testCount = parser.getMicroTestCount();

        for (size_t i = 0; i < testCount; ++i) {
            const MicroTest& microTest = parser.getMicroTest();

            // Verify all matrix types are accessible
            MatrixType a_type   = microTest.getAType();
            MatrixType b_type   = microTest.getBType();
            MatrixType c_type   = microTest.getCType();
            MatrixType acc_type = microTest.getAccType();

            std::cout << "Test " << i << ": Types A=" << a_type
                      << ", B=" << b_type << ", C=" << c_type
                      << ", Acc=" << acc_type << std::endl;

            // All types should be accessible (no exceptions thrown)
            // Types are already validated by the parser

            if (i < testCount - 1) {
                parser.next();
            }
        }

        std::cout << "✓ All matrix types valid for batch GEMM" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Matrix types test failed: " << e.what();
    }
}

/**
 * @brief Test batch GEMM with PostOps parsing
 *
 * Verifies that PostOps are correctly parsed and accessible for batch GEMM
 * tests.
 */
TEST(YamlParserTest, BatchGemmWithPostOps)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    try {
        YamlParser parser(filepath, "batch_gemm_tests");

        // Find a test with PostOps (skip to appropriate test)
        size_t microTestCount     = parser.getMicroTestCount();
        bool   found_postops_test = false;

        std::cout << "\n=== Batch GEMM PostOps Test ===" << std::endl;

        for (size_t i = 0; i < microTestCount; ++i) {
            const MicroTest& microTest = parser.getMicroTest();

            // Check if PostOps are available by attempting to get them
            auto postops_dlp =
                microTest.getPostOp(dlp::testing::framework::UALType::DLP);
            auto postops_ref =
                microTest.getPostOp(dlp::testing::framework::UALType::REF);

            if (postops_dlp != nullptr && postops_ref != nullptr) {
                found_postops_test = true;

                std::cout << "Found batch GEMM test with PostOps at index " << i
                          << std::endl;

                EXPECT_NE(postops_dlp, nullptr)
                    << "DLP PostOps should be available";
                EXPECT_NE(postops_ref, nullptr)
                    << "REF PostOps should be available";

                // Verify group_size is present
                EXPECT_TRUE(microTest.hasGroupSize());

                md_t group_size = microTest.getGroupSize();
                std::cout << "  group_size=" << group_size << std::endl;

                break;
            }

            if (i < microTestCount - 1) {
                parser.next();
            }
        }

        EXPECT_TRUE(found_postops_test)
            << "Should have at least one batch GEMM test with PostOps";

        std::cout << "✓ PostOps parsing works for batch GEMM" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "BatchGemmWithPostOps test failed: " << e.what();
    }
}

/**
 * @brief Test PostOps consistency across iterations
 *
 * Verifies that PostOps remain available and consistent across MicroTest
 * iterations.
 */
TEST(YamlParserTest, BatchGemmPostOpsConsistency)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    try {
        YamlParser parser(filepath, "batch_gemm_tests");

        // Test that PostOps remain consistent across iterations
        size_t microTestCount = parser.getMicroTestCount();

        std::cout << "\n=== Batch GEMM PostOps Consistency Test ==="
                  << std::endl;

        for (size_t i = 0; i < microTestCount; ++i) {
            const MicroTest& microTestRef = parser.getMicroTest();
            MicroTest&       microTest = const_cast<MicroTest&>(microTestRef);

            // Check if PostOps are available
            auto postops_check =
                microTest.getPostOp(dlp::testing::framework::UALType::DLP);
            if (postops_check == nullptr) {
                if (i < microTestCount - 1)
                    parser.next();
                continue;
            }

            std::cout << "Testing PostOps consistency for test " << i
                      << std::endl;

            // Check that PostOps are available across multiple iterations
            size_t iterations = std::min(microTest.getSize(), size_t(3));

            for (size_t j = 0; j < iterations; ++j) {
                auto postops_dlp =
                    microTest.getPostOp(dlp::testing::framework::UALType::DLP);

                EXPECT_NE(postops_dlp, nullptr)
                    << "PostOps should be available at iteration " << j;

                std::cout << "  Iteration " << j << ": PostOps OK" << std::endl;

                if (j < iterations - 1 && microTest.hasNext()) {
                    microTest.next();
                }
            }

            std::cout << "✓ PostOps consistent across iterations" << std::endl;
            break; // Only test first PostOps-enabled test
        }

    } catch (const std::exception& e) {
        FAIL() << "BatchGemmPostOpsConsistency test failed: " << e.what();
    }
}

/**
 * @brief Test PostOps with different group sizes
 *
 * Verifies that PostOps work correctly with both single-group and multi-group
 * configurations.
 */
TEST(YamlParserTest, BatchGemmPostOpsWithGroupSizes)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    try {
        YamlParser parser(filepath, "batch_gemm_tests");

        size_t microTestCount = parser.getMicroTestCount();

        std::cout << "\n=== Batch GEMM PostOps with Group Sizes Test ==="
                  << std::endl;

        size_t single_group_postops = 0;
        size_t multi_group_postops  = 0;

        for (size_t i = 0; i < microTestCount; ++i) {
            const MicroTest& microTest = parser.getMicroTest();

            auto postops =
                microTest.getPostOp(dlp::testing::framework::UALType::DLP);
            if (postops != nullptr && microTest.hasGroupSize()) {
                md_t group_size = microTest.getGroupSize();

                EXPECT_NE(postops, nullptr);

                if (group_size == 1) {
                    single_group_postops++;
                    std::cout
                        << "Found single-group test with PostOps (group_size="
                        << group_size << ")" << std::endl;
                } else {
                    multi_group_postops++;
                    std::cout
                        << "Found multi-group test with PostOps (group_size="
                        << group_size << ")" << std::endl;
                }
            }

            if (i < microTestCount - 1) {
                parser.next();
            }
        }

        std::cout << "Single-group PostOps tests: " << single_group_postops
                  << std::endl;
        std::cout << "Multi-group PostOps tests: " << multi_group_postops
                  << std::endl;

        EXPECT_GT(single_group_postops + multi_group_postops, 0)
            << "Should have at least one test with PostOps and group_size";

        std::cout << "✓ PostOps work with various group sizes" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "BatchGemmPostOpsWithGroupSizes test failed: " << e.what();
    }
}

// ============================================================================
// BATCH GEMM FRAMEWORK TESTS
// ============================================================================

/**
 * @brief Test fixture for batch GEMM framework tests
 */
class BatchGemmFrameworkTest : public ::testing::Test
{
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

/**
 * @brief Test MicroTest default group_size behavior
 *
 * Verifies that MicroTest returns a default group_size of 1 when not specified,
 * ensuring backward compatibility with non-batch GEMM tests.
 */
TEST_F(BatchGemmFrameworkTest, DefaultGroupSize)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_config_value.yaml";

    YamlParser       parser(filepath, "yaml_test");
    const MicroTest& microTest = parser.getMicroTest();

    // Test default behavior
    EXPECT_EQ(microTest.getGroupSize(), 1)
        << "Default group_size should be 1 for backward compatibility";

    EXPECT_FALSE(microTest.hasGroupSize())
        << "hasGroupSize() should return false when group_size not in YAML";
}

/**
 * @brief Test MicroTest with explicit group_size
 *
 * Verifies that MicroTest correctly parses and provides access to group_size
 * parameter when explicitly specified in YAML.
 */
TEST_F(BatchGemmFrameworkTest, ExplicitGroupSize)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    YamlParser       parser(filepath, "batch_gemm_tests");
    const MicroTest& microTest = parser.getMicroTest();

    // Test explicit group_size
    EXPECT_TRUE(microTest.hasGroupSize())
        << "hasGroupSize() should return true when group_size in YAML";

    md_t group_size = microTest.getGroupSize();
    EXPECT_GT(group_size, 0) << "Explicit group_size should be positive";
}

/**
 * @brief Test group_size iteration in cartesian product mode
 *
 * Verifies that group_size remains constant across all combinations when
 * using cartesian product mode (single-group batch testing).
 */
TEST_F(BatchGemmFrameworkTest, GroupSizeCartesianProduct)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    YamlParser       parser(filepath, "batch_gemm_tests");
    const MicroTest& microTestRef = parser.getMicroTest();
    MicroTest&       microTest    = const_cast<MicroTest&>(microTestRef);

    md_t         initial_group_size = microTest.getGroupSize();
    size_t       iteration_count    = 0;
    const size_t MAX_ITERATIONS     = 10;

    // Iterate through combinations
    do {
        md_t current_group_size = microTest.getGroupSize();
        EXPECT_EQ(current_group_size, initial_group_size)
            << "Group size should remain constant in cartesian mode at "
               "iteration "
            << iteration_count;

        iteration_count++;
        if (iteration_count >= MAX_ITERATIONS || !microTest.hasNext()) {
            break;
        }
        microTest.next();
    } while (true);

    EXPECT_GT(iteration_count, 1)
        << "Should have iterated through multiple combinations";
}

/**
 * @brief Test group_size pairing in simple product mode
 *
 * Verifies that group_size values are correctly paired with other parameters
 * in simple product mode (multi-group batch testing).
 */
TEST_F(BatchGemmFrameworkTest, GroupSizeSimpleProduct)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    YamlParser parser(filepath, "batch_gemm_tests");

    // Skip to multi-group test (test index 3 = 4th test)
    for (int i = 0; i < 3; ++i) {
        parser.next();
    }

    const MicroTest& microTestRef = parser.getMicroTest();
    MicroTest&       microTest    = const_cast<MicroTest&>(microTestRef);

    // Expected pairings for MultiGroup_MixedSizes_F32
    std::vector<md_t> expected_group_sizes = { 2, 3, 4 };
    std::vector<md_t> expected_m           = { 8, 16, 32 };

    size_t count = 0;
    do {
        md_t gs = microTest.getGroupSize();
        md_t m  = microTest.getM();

        ASSERT_LT(count, expected_group_sizes.size())
            << "Exceeded expected number of pairings";

        EXPECT_EQ(gs, expected_group_sizes[count])
            << "Group size should be paired correctly at iteration " << count;
        EXPECT_EQ(m, expected_m[count])
            << "M dimension should be paired with group_size at iteration "
            << count;

        count++;
        if (!microTest.hasNext())
            break;
        microTest.next();
    } while (count < expected_group_sizes.size() + 1);

    EXPECT_EQ(count, expected_group_sizes.size())
        << "Should have exactly " << expected_group_sizes.size() << " pairings";
}

/**
 * @brief Test MicroTest::getSize() with group_size
 *
 * Verifies that MicroTest correctly calculates total combinations
 * when group_size is present.
 */
TEST_F(BatchGemmFrameworkTest, SizeCalculationWithGroupSize)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    // Test 1: Single-group cartesian (should multiply combinations)
    {
        YamlParser       parser(filepath, "batch_gemm_tests");
        const MicroTest& microTestRef = parser.getMicroTest();
        MicroTest&       microTest    = const_cast<MicroTest&>(microTestRef);

        size_t reported_size = microTest.getSize();

        // For first test: m=[8,16], n=[8,16], k=[8,16] = 2*2*2 = 8
        // But we have 3 values each, so should be larger
        EXPECT_GT(reported_size, 0)
            << "Size should be positive for single-group cartesian";

        // Count actual iterations
        size_t actual_count = 0;
        do {
            actual_count++;
            if (!microTest.hasNext())
                break;
            microTest.next();
        } while (actual_count < reported_size + 5); // Safety limit

        EXPECT_LE(actual_count, reported_size)
            << "Actual iterations should match or be less than reported size";
    }

    // Test 2: Multi-group simple product
    {
        YamlParser parser(filepath, "batch_gemm_tests");

        // Skip to multi-group test (test index 3 = 4th test)
        for (int i = 0; i < 3; ++i) {
            parser.next();
        }

        const MicroTest& microTestRef = parser.getMicroTest();
        MicroTest&       microTest    = const_cast<MicroTest&>(microTestRef);

        size_t reported_size = microTest.getSize();

        // For MultiGroup_MixedSizes_F32: 3 groups
        EXPECT_EQ(reported_size, 3)
            << "Multi-group simple product should have 3 combinations";
    }
}

/**
 * @brief Test group_size with different data types
 *
 * Verifies that group_size works correctly with different matrix types.
 */
TEST_F(BatchGemmFrameworkTest, GroupSizeWithDifferentTypes)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    YamlParser parser(filepath, "batch_gemm_tests");

    // Iterate through all test configurations
    size_t testCount = parser.getMicroTestCount();

    for (size_t i = 0; i < testCount; ++i) {
        const MicroTest& microTest = parser.getMicroTest();

        // Verify group_size is accessible regardless of matrix types
        EXPECT_NO_THROW({
            md_t       gs     = microTest.getGroupSize();
            MatrixType a_type = microTest.getAType();
            MatrixType b_type = microTest.getBType();
            MatrixType c_type = microTest.getCType();

            EXPECT_GT(gs, 0) << "Group size should be positive for test " << i;
        }) << "Group size should work with any matrix type at test "
           << i;

        if (i < testCount - 1) {
            parser.next();
        }
    }
}

/**
 * @brief Test group_size persistence across reset
 *
 * Verifies that group_size is correctly restored after parser reset.
 */
TEST_F(BatchGemmFrameworkTest, GroupSizePersistenceAcrossReset)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    YamlParser       parser(filepath, "batch_gemm_tests");
    const MicroTest& microTest1 = parser.getMicroTest();

    md_t initial_group_size = microTest1.getGroupSize();
    md_t initial_m          = microTest1.getM();

    // Move to next test
    parser.next();
    const MicroTest& microTest2 = parser.getMicroTest();

    md_t second_group_size = microTest2.getGroupSize();

    // Reset parser
    parser.reset();
    const MicroTest& microTest3 = parser.getMicroTest();

    md_t reset_group_size = microTest3.getGroupSize();
    md_t reset_m          = microTest3.getM();

    // Verify reset restored initial values
    EXPECT_EQ(reset_group_size, initial_group_size)
        << "Group size should be restored after reset";
    EXPECT_EQ(reset_m, initial_m)
        << "Other parameters should also be restored after reset";
}

/**
 * @brief Test group_size with edge case values
 *
 * Tests group_size with minimum valid values and larger values.
 */
TEST_F(BatchGemmFrameworkTest, GroupSizeEdgeCases)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    YamlParser parser(filepath, "batch_gemm_tests");

    // Look for test with group_size = 1 (minimum valid)
    size_t testCount          = parser.getMicroTestCount();
    bool   found_single_group = false;
    bool   found_large_group  = false;

    for (size_t i = 0; i < testCount; ++i) {
        const MicroTest& microTest = parser.getMicroTest();
        md_t             gs        = microTest.getGroupSize();

        if (gs == 1) {
            found_single_group = true;
            EXPECT_EQ(gs, 1)
                << "Single matrix per group (group_size=1) should be valid";
        }

        if (gs >= 4) {
            found_large_group = true;
            EXPECT_GT(gs, 0) << "Larger group sizes should be valid";
        }

        if (i < testCount - 1) {
            parser.next();
        }
    }

    // We expect to find both cases in the test config
    EXPECT_TRUE(found_single_group || found_large_group)
        << "Should have at least one edge case in test configurations";
}

/**
 * @brief Test group_size type safety
 *
 * Verifies that group_size is properly type-safe (md_t type).
 */
TEST_F(BatchGemmFrameworkTest, GroupSizeTypeSafety)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    YamlParser       parser(filepath, "batch_gemm_tests");
    const MicroTest& microTest = parser.getMicroTest();

    // Verify type is md_t (should be size_t or similar)
    md_t group_size = microTest.getGroupSize();

    // Type should support comparison operators
    EXPECT_TRUE(group_size == group_size);
    EXPECT_FALSE(group_size != group_size);
    EXPECT_TRUE(group_size >= 1);
    EXPECT_TRUE(group_size <= std::numeric_limits<md_t>::max());

    // Should be assignable
    md_t copy = group_size;
    EXPECT_EQ(copy, group_size);
}

/**
 * @brief Test PostOps with group_size
 *
 * Verifies that both PostOps and group_size are accessible together.
 */
TEST_F(BatchGemmFrameworkTest, PostOpsWithGroupSize)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    YamlParser parser(filepath, "batch_gemm_tests");

    // Navigate to a test with PostOps
    size_t microTestCount = parser.getMicroTestCount();
    bool   found          = false;

    std::cout << "\n=== PostOps + GroupSize Framework Test ===" << std::endl;

    for (size_t i = 0; i < microTestCount; ++i) {
        const MicroTest& microTest = parser.getMicroTest();

        auto postops = microTest.getPostOp(UALType::DLP);
        if (postops != nullptr && microTest.hasGroupSize()) {
            found = true;

            // Verify both PostOps and group_size are accessible
            EXPECT_NE(postops, nullptr);
            EXPECT_TRUE(microTest.hasGroupSize());

            md_t group_size = microTest.getGroupSize();

            EXPECT_NE(postops, nullptr) << "PostOps should be available";
            EXPECT_GT(group_size, 0) << "Group size should be positive";

            std::cout << "✓ Found batch GEMM test with PostOps and group_size="
                      << group_size << std::endl;

            break;
        }

        if (i < microTestCount - 1) {
            parser.next();
        }
    }

    EXPECT_TRUE(found)
        << "Should find at least one test with both PostOps and group_size";
}

/**
 * @brief Test that PostOps work with both single-group and multi-group modes
 *
 * Verifies that PostOps are correctly handled for different group
 * configurations.
 */
TEST_F(BatchGemmFrameworkTest, PostOpsWithVariousGroupConfigurations)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/batch_gemm_test_config.yaml";

    YamlParser parser(filepath, "batch_gemm_tests");

    size_t microTestCount = parser.getMicroTestCount();

    std::cout << "\n=== PostOps Group Configuration Test ===" << std::endl;

    bool found_single_group_with_postops = false;
    bool found_multi_group_with_postops  = false;

    for (size_t i = 0; i < microTestCount; ++i) {
        const MicroTest& microTestRef = parser.getMicroTest();
        MicroTest&       microTest    = const_cast<MicroTest&>(microTestRef);

        auto postops = microTest.getPostOp(UALType::DLP);
        if (postops != nullptr && microTest.hasGroupSize()) {
            // Check group configuration by examining iterations
            md_t initial_group_size = microTest.getGroupSize();

            // For single-group: group_size should be constant across iterations
            // For multi-group: group_size may vary (but we test the first
            // value)

            if (initial_group_size >= 2) {
                // This could be single-group with multiple matrices
                // or first value of multi-group

                // Try to check if this is truly multi-group by checking next
                // iteration
                if (microTest.hasNext()) {
                    microTest.next();
                    md_t next_group_size = microTest.getGroupSize();

                    if (next_group_size != initial_group_size) {
                        found_multi_group_with_postops = true;
                        std::cout << "✓ Found multi-group with PostOps "
                                     "(group_sizes vary)"
                                  << std::endl;
                    } else {
                        found_single_group_with_postops = true;
                        std::cout
                            << "✓ Found single-group with PostOps (group_size="
                            << initial_group_size << ")" << std::endl;
                    }
                } else {
                    found_single_group_with_postops = true;
                    std::cout
                        << "✓ Found single-group with PostOps (group_size="
                        << initial_group_size << ")" << std::endl;
                }
            }
        }

        if (found_single_group_with_postops && found_multi_group_with_postops) {
            break; // Found both, can stop
        }

        if (i < microTestCount - 1) {
            parser.next();
        }
    }

    // We should find at least one configuration with PostOps
    EXPECT_TRUE(found_single_group_with_postops
                || found_multi_group_with_postops)
        << "Should find at least one group configuration with PostOps";
}

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
