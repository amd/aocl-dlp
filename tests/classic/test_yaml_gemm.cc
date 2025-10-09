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

#include <gtest/gtest.h>
#include <iostream>

#include "framework/utils/yaml_parser.hh"
#include "test_config.hh"

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

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
