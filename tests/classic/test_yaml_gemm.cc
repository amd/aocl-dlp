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

#include "test_config.hh"
#include "utils/yaml_parser.hh"

using namespace dlp::testing;

/*
   Test configurations which uses range and list type
   File: yaml_test_config_range_list.yaml
 */
// Test the YamlParser with a sample YAML configuration file
TEST(YamlParserTest, ParseGemmTestConfig)
{
    // Use a relative path to the test configuration file
    std::string filepath = TEST_CONFIG_DIR
        "/test_configs/yaml_test_config_range_list.yaml";

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
TEST(YamlParserTest, CartesianProductTest)
{
    std::string filepath = TEST_CONFIG_DIR
        "/test_configs/yaml_test_config_range_list.yaml";

    try {
        YamlParser parser(filepath, "yaml_test");

        // Get the first test case (small_matrix) and iterate through all
        // combinations We'll use a reference to the MicroTest and call next()
        // on it
        const MicroTest& microTestRef = parser.getMicroTest();
        // Cast away const to call next() - this is safe since we own the parser
        MicroTest& microTest = const_cast<MicroTest&>(microTestRef);

        // Get the total number of combinations for this test case
        size_t totalCombinations = microTest.getSize();
        std::cout << "\n=== Cartesian Product Test for 'small_matrix' ==="
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

        // Iterate through all remaining combinations
        while (microTest.hasNext()) {
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

            // Limit output to first 10 combinations to avoid overwhelming
            // output
            if (combinationCount >= 10) {
                std::cout << "\n... (showing first 10 combinations only)"
                          << std::endl;
                break;
            }
        }

        // Verify the count matches expectation
        // Based on the YAML config for small_matrix:
        // Base parameters:
        // - m: range lb:10, ub:20, step:5 = [10, 15, 20] = 3 values
        // - n: range lb:10, ub:10, step:0 = [10] = 1 value
        // - k: range lb:10, ub:10, step:0 = [10] = 1 value
        // - alpha: [2.5, 0, -2.5] = 3 values
        // - beta: [2.5, 0, -2.5] = 3 values
        // - lda: [10, 20, 30] = 3 values
        // - transA: [false, true] = 2 values
        // - transB: [true] = 1 value
        // - reorderB: true = 1 value
        // Base total: 3 * 1 * 1 * 3 * 3 * 3 * 2 * 1 * 1 = 162 combinations
        //
        // PostOps (6 operations with cartesian=true):
        // Operations: [SIGMOID, Sum, Scale, Bias, Matrix-Add, Matrix-Mul]
        // With cartesian=true: generates all permutations of the full sequence
        // PostOps total: 6! = 720 permutations
        //
        // Total combinations: 162 base × 720 PostOps = 116,640

        size_t baseCombinations = 3 * 1 * 1 * 3 * 3 * 3 * 2 * 1 * 1; // 162
        size_t postOpsCombinations =
            720; // 6! permutations of the full sequence
        size_t expectedCombinations =
            baseCombinations * postOpsCombinations; // 116,640
        EXPECT_EQ(totalCombinations, expectedCombinations)
            << "Expected " << expectedCombinations << " combinations but got "
            << totalCombinations;

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

// Test moving to the medium_matrix test set and its cartesian product
TEST(YamlParserTest, MediumMatrixCartesianProductTest)
{
    std::string filepath = TEST_CONFIG_DIR
        "/test_configs/yaml_test_config_range_list.yaml";

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
        "/test_configs/yaml_test_config_value.yaml";

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
        "/test_configs/yaml_test_config_list.yaml";

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
        // "Sum"]:
        // 1. [PRELU, Bias, Sum]
        // 2. [PRELU, Sum, Bias]
        // 3. [Bias, PRELU, Sum]
        // 4. [Bias, Sum, PRELU]
        // 5. [Sum, PRELU, Bias]
        // 6. [Sum, Bias, PRELU]
        // PostOps total: 3! = 6 permutations
        //
        // Total combinations: 54 base × 6 PostOps = 324

        size_t totalCombinations = microTest.getSize();
        size_t baseCombinations  = 1 * 1 * 1 * 1 * 1 * 2 * 1 * 1 * 1 * 1 * 3 * 3
                                  * 3 * 1 * 1 * 1 * 1; // 54
        size_t postOpsCombinations = 6; // 3! permutations of the full sequence
        size_t expectedCombinations =
            baseCombinations * postOpsCombinations; // 324

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
        TEST_CONFIG_DIR "/test_configs/yaml_test_config_value.yaml",
        TEST_CONFIG_DIR "/test_configs/yaml_test_config_list.yaml",
        TEST_CONFIG_DIR "/test_configs/yaml_test_config_range_list.yaml"
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
TEST(YamlParserTest, ElementWiseSimpleProductTest)
{
    std::string filepath = TEST_CONFIG_DIR
        "/test_configs/yaml_test_config_list.yaml";

    try {
        YamlParser parser(filepath, "yaml_test");

        // Set to element-wise mode instead of cartesian product
        parser.setYieldType(YieldType::SIMPLE_PRODUCT);
        EXPECT_EQ(parser.getYieldType(), YieldType::SIMPLE_PRODUCT);

        // Get the test case and cast to mutable for iteration
        const MicroTest& microTestRef = parser.getMicroTest();
        MicroTest&       microTest    = const_cast<MicroTest&>(microTestRef);

        // In element-wise mode, the size should be the minimum of all list
        // sizes From yaml_test_config_list.yaml:
        // Base parameters:
        // - a_type: ["f32"] = 1 value (single-element list, NOT expandable)
        // - transA: [false, true] = 2 values (list)
        // - alpha: [2.5, 0, -2.5] = 3 values (list)
        // - beta: [2.5, 0, -2.5] = 3 values (list)
        // - lda: [10, 20, 30] = 3 values (list)
        // - m: 10 = single value (expandable)
        // Base minimum size = min(1, 2, 3, 3, 3) = 1 (limited by a_type
        // single-element list)
        //
        // PostOps (3 operations with cartesian=true):
        // 3 operations ["Elementwise-PRELU", "Bias", "Sum"] generate 3! = 6
        // permutations
        //
        // Total: 1 base × 6 PostOps = 6 combinations

        size_t totalCombinations = microTest.getSize();
        size_t baseCombinations =
            1; // Limited by a_type: ["f32"] single-element list
        size_t postOpsCombinations = 6; // 3! permutations
        size_t expectedCombinations =
            baseCombinations * postOpsCombinations; // 6

        EXPECT_EQ(totalCombinations, expectedCombinations)
            << "Element-wise should have " << expectedCombinations
            << " combinations";

        // Should be first elements from each list: a_type=f32, transA=false,
        // alpha=2.5, beta=2.5, lda=10
        EXPECT_EQ(microTest.getAType(),
                  MatrixType::f32);           // First (and only) from ["f32"]
        EXPECT_FALSE(microTest.getTransA());  // First from [false, true]
        EXPECT_EQ(microTest.getAlpha(), 2.5); // First from [2.5, 0, -2.5]
        EXPECT_EQ(microTest.getBeta(), 2.5);  // First from [2.5, 0, -2.5]
        EXPECT_EQ(microTest.getLDA(), 10);    // First from [10, 20, 30]

        // Note: SimpleProduct's has_next() uses CartesianProduct::has_next()
        // which doesn't know about SimpleProduct's m_finished flag until next()
        // is called. So we need to actually try calling next() to see if it
        // throws an exception.
        size_t actualCombinations = 1; // Constructor consumed the first one
        try {
            while (microTest.hasNext()) {
                microTest.next();
                actualCombinations++;

                // Safety check to avoid infinite loop
                if (actualCombinations > 10) {
                    FAIL() << "Too many combinations, something is wrong";
                    break;
                }
            }
        } catch (const std::runtime_error& e) {
            // This is expected when SimpleProduct reaches the end
        }

        EXPECT_EQ(actualCombinations, expectedCombinations);

    } catch (const std::exception& e) {
        FAIL() << "Element-wise test threw an exception: " << e.what();
    }
}

/**
 * @brief Test SimpleProduct with range configuration
 *
 * This test verifies element-wise functionality with range-based parameters.
 */
TEST(YamlParserTest, ElementWiseWithRangesTest)
{
    std::string filepath = TEST_CONFIG_DIR
        "/test_configs/yaml_test_config_range_list.yaml";

    try {
        YamlParser parser(filepath, "yaml_test");

        // Set to element-wise mode
        parser.setYieldType(YieldType::SIMPLE_PRODUCT);

        // Get the first test case (small_matrix)
        const MicroTest& microTestRef = parser.getMicroTest();
        MicroTest&       microTest    = const_cast<MicroTest&>(microTestRef);

        // From yaml_test_config_range_list.yaml small_matrix:
        // Base parameters in ElementWise mode:
        // ALL parameters are considered for minimum:
        // - Single-element lists: a_type[1], b_type[1], c_type[1], acc_type[1],
        // storage_format[1], transB[1] = 1 each
        // - Multi-element lists: alpha[3], beta[3], lda[3], transA[2]
        // - Ranges: m[3], n[1], k[1], ldb[1]
        // - Single values: ldc, mtagA, mtagB = 1 each
        // Base ElementWise combinations: min(1,1,1,1,1,2,1,3,1,1,3,3,3,1,1,1) =
        // 1 (limited by single-element lists)
        //
        // PostOps (6 operations with cartesian=true):
        // 6! = 720 permutations
        //
        // Total: 1 base × 720 PostOps = 720 combinations

        size_t totalCombinations = microTest.getSize();
        size_t baseCombinations =
            1; // ElementWise: limited by single-element lists
        size_t postOpsCombinations = 720; // 6! permutations
        size_t expectedCombinations =
            baseCombinations * postOpsCombinations; // 720

        // Verify we can access all parameters without exceptions
        EXPECT_NO_THROW({
            auto m      = microTest.getM();
            auto n      = microTest.getN();
            auto k      = microTest.getK();
            auto alpha  = microTest.getAlpha();
            auto beta   = microTest.getBeta();
            auto lda    = microTest.getLDA();
            auto transA = microTest.getTransA();
        });

        // Test if there are more combinations available
        size_t combinationCount = 1; // Constructor consumed the first one

        try {
            while (microTest.hasNext()) {
                microTest.next();
                combinationCount++;

                // Safety check to avoid infinite loop - allow for PostOps
                // multiplication
                if (combinationCount > expectedCombinations + 10) {
                    FAIL() << "Too many combinations, expected around "
                           << expectedCombinations;
                    break;
                }
            }
        } catch (const std::runtime_error& e) {
            // This is expected when SimpleProduct reaches the end
        }

        EXPECT_EQ(combinationCount, totalCombinations);
        EXPECT_EQ(totalCombinations, expectedCombinations)
            << "Expected " << expectedCombinations
            << " combinations for ElementWise with PostOps";

    } catch (const std::exception& e) {
        FAIL() << "Element-wise with ranges test threw an exception: "
               << e.what();
    }
}

/**
 * @brief Test switching between CartesianProduct and SimpleProduct modes
 *
 * This test verifies that the YieldType can be changed and affects
 * the number of combinations generated.
 */
TEST(YamlParserTest, YieldTypeSwitchingTest)
{
    std::string filepath = TEST_CONFIG_DIR
        "/test_configs/yaml_test_config_list.yaml";

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
        "/test_configs/yaml_test_config_value.yaml";

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
 * @brief Comprehensive test demonstrating CartesianProduct vs SimpleProduct
 * differences
 *
 * This test creates a scenario where CartesianProduct and SimpleProduct produce
 * significantly different numbers of combinations, clearly showing the
 * difference between the two approaches.
 */
TEST(YamlParserTest, CartesianVsElementWiseComparisonTest)
{
    std::string filepath = TEST_CONFIG_DIR
        "/test_configs/yaml_test_config_list.yaml";

    try {
        // Test Cartesian Product first
        {
            YamlParser parser(filepath, "yaml_test");
            parser.setYieldType(YieldType::CARTESIAN_PRODUCT);

            const MicroTest& microTestRef = parser.getMicroTest();
            MicroTest&       microTest = const_cast<MicroTest&>(microTestRef);

            size_t cartesianSize = microTest.getSize();
            EXPECT_GT(cartesianSize, 1)
                << "Cartesian product should have multiple combinations";
        }

        // Test Element-Wise (SimpleProduct)
        {
            YamlParser parser(filepath, "yaml_test");
            parser.setYieldType(YieldType::SIMPLE_PRODUCT);

            const MicroTest& microTestRef = parser.getMicroTest();
            MicroTest&       microTest = const_cast<MicroTest&>(microTestRef);

            size_t elementWiseSize = microTest.getSize();

            // Count actual combinations
            size_t combinationCount = 1;
            try {
                while (microTest.hasNext()) {
                    microTest.next();
                    combinationCount++;

                    if (combinationCount > 10)
                        break; // Safety check
                }
            } catch (const std::runtime_error& e) {
                // Expected when SimpleProduct reaches the end
            }

            EXPECT_EQ(combinationCount, elementWiseSize);
            EXPECT_GE(elementWiseSize, 1)
                << "Element-wise should have at least 1 combination";
        }

    } catch (const std::exception& e) {
        FAIL()
            << "Cartesian vs Element-Wise comparison test threw an exception: "
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
        "/test_configs/yaml_test_config_minimal_no_cartesian.yaml";

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
        "/test_configs/yaml_test_config_minimal.yaml";

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
        // 1. ["Elementwise-RELU", "Bias"] (original order)
        // 2. ["Bias", "Elementwise-RELU"] (reverse order)
        // Total PostOps combinations: 2 (2! permutations)
        // Total: 1 base × 2 PostOps = 2 combinations

        size_t totalCombinations    = microTest.getSize();
        size_t expectedCombinations = 2; // 1 base × 2 PostOps permutations

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

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
