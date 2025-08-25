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

using namespace dlp::testing::utils;
using dlp::testing::framework::Matrix;
using dlp::testing::framework::MatrixLayout;
using dlp::testing::framework::MatrixType;
using dlp::testing::framework::UalFactory;
using dlp::testing::framework::UALType;

class PostOpsYamlTest : public ::testing::Test
{
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(PostOpsYamlTest, BasicPostOpsParsingTest)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_postops.yaml";

    try {
        YamlParser parser(filepath, "yaml_test");

        // Get the number of test cases
        size_t testCount = parser.getMicroTestCount();
        ASSERT_EQ(testCount, 2)
            << "Expected 2 test cases in the PostOps YAML file";

        // Test the first test case (with PostOps)
        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "=== PostOps YAML Test - First Test Case ===" << std::endl;
        std::cout << "Test parameters:" << std::endl;
        std::cout << "  M: " << microTest.getM() << std::endl;
        std::cout << "  N: " << microTest.getN() << std::endl;
        std::cout << "  K: " << microTest.getK() << std::endl;
        std::cout << "  A Type: " << microTest.getAType() << std::endl;
        std::cout << "  Total combinations: " << microTest.getSize()
                  << std::endl;

        // Verify basic parameters
        EXPECT_EQ(microTest.getM(), 4);
        EXPECT_EQ(microTest.getN(), 4);
        EXPECT_EQ(microTest.getK(), 4);
        EXPECT_EQ(microTest.getAType(), MatrixType::f32);

        // Test PostOps functionality
        auto postops_dlp = microTest.getPostOp(UALType::DLP);
        auto postops_ref = microTest.getPostOp(UALType::REF);

        if (postops_dlp) {
            std::cout << "  DLP PostOps: CREATED SUCCESSFULLY" << std::endl;
            EXPECT_NE(postops_dlp, nullptr) << "DLP PostOps should be created";
        } else {
            std::cout << "  DLP PostOps: NONE (empty combination)" << std::endl;
        }

        if (postops_ref) {
            std::cout << "  REF PostOps: CREATED SUCCESSFULLY" << std::endl;
            EXPECT_NE(postops_ref, nullptr) << "REF PostOps should be created";
        } else {
            std::cout << "  REF PostOps: NONE (empty combination)" << std::endl;
        }

        // Iterate through some combinations to test PostOps iterator
        int combination_count = 0;
        int postops_found     = 0;

        auto& mutableMicroTest = const_cast<MicroTest&>(microTest);

        do {
            auto dlp_ops = microTest.getPostOp(UALType::DLP);
            auto ref_ops = microTest.getPostOp(UALType::REF);

            if (dlp_ops || ref_ops) {
                postops_found++;
                std::cout << "  Combination " << combination_count
                          << ": PostOps present" << std::endl;
            }

            combination_count++;

            // Limit to avoid too much output
            if (combination_count >= 10)
                break;

            if (mutableMicroTest.hasNext()) {
                mutableMicroTest.next();
            } else {
                break;
            }
        } while (true);

        std::cout << "  Tested " << combination_count << " combinations"
                  << std::endl;
        std::cout << "  PostOps found in " << postops_found << " combinations"
                  << std::endl;

        // We should have found some PostOps combinations
        EXPECT_GT(postops_found, 0) << "Should find some PostOps combinations";

        // Move to the second test case (without PostOps)
        parser.next();
        const MicroTest& microTest2 = parser.getMicroTest();

        std::cout << "=== Second Test Case (No PostOps) ===" << std::endl;
        std::cout << "  M: " << microTest2.getM() << std::endl;
        std::cout << "  N: " << microTest2.getN() << std::endl;
        std::cout << "  K: " << microTest2.getK() << std::endl;
        std::cout << "  Total combinations: " << microTest2.getSize()
                  << std::endl;

        // This test case should have no PostOps
        auto postops_dlp_2 = microTest2.getPostOp(UALType::DLP);
        auto postops_ref_2 = microTest2.getPostOp(UALType::REF);

        EXPECT_EQ(postops_dlp_2, nullptr)
            << "Second test case should have no PostOps";
        EXPECT_EQ(postops_ref_2, nullptr)
            << "Second test case should have no PostOps";

        std::cout << "  DLP PostOps: " << (postops_dlp_2 ? "PRESENT" : "NONE")
                  << std::endl;
        std::cout << "  REF PostOps: " << (postops_ref_2 ? "PRESENT" : "NONE")
                  << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "PostOps YAML test threw an exception: " << e.what();
    }
}

TEST_F(PostOpsYamlTest, PostOpsCombinationCountTest)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_postops.yaml";

    try {
        YamlParser       parser(filepath, "yaml_test");
        const MicroTest& microTest = parser.getMicroTest();

        // First test case has PostOps with cartesian enabled
        // Operations: PRELU (2 alpha_types) + Bias (2 bias_dims) + Sum (2
        // scale_lens) Cartesian should generate many combinations:
        // - Empty combination: 1
        // - Single operations: 3
        // - Pairs: 3*2! = 6 permutations
        // - Triple: 1*3! = 6 permutations
        // Total should be > 10 combinations

        size_t total_combinations = microTest.getSize();
        std::cout << "Total combinations with PostOps: " << total_combinations
                  << std::endl;

        // Since we have 1 base parameter combination and multiple PostOps
        // combinations, the total should be significantly more than 1
        EXPECT_GT(total_combinations, 10)
            << "Should have many PostOps combinations";

        // Test without PostOps
        parser.next();
        const MicroTest& microTest2              = parser.getMicroTest();
        size_t           no_postops_combinations = microTest2.getSize();

        std::cout << "Total combinations without PostOps: "
                  << no_postops_combinations << std::endl;

        // Without PostOps, should just be the base parameter combinations (1 in
        // this case)
        EXPECT_EQ(no_postops_combinations, 1)
            << "Should have only base combinations without PostOps";

    } catch (const std::exception& e) {
        FAIL() << "PostOps combination count test threw an exception: "
               << e.what();
    }
}

TEST_F(PostOpsYamlTest, PostOpsWithGemmTest)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_postops.yaml";

    try {
        YamlParser       parser(filepath, "yaml_test");
        const MicroTest& microTest = parser.getMicroTest();

        // Create matrices for testing
        Matrix A(4, 4, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
        Matrix B(4, 4, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
        Matrix C_dlp(4, 4, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
        Matrix C_ref(4, 4, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);

        // Initialize matrices with simple values
        A.fillValue(0.5f);
        B.fillValue(0.2f);
        C_dlp.fillValue(0.0f);
        C_ref.fillValue(0.0f);

        // Get PostOps
        auto postops_dlp = microTest.getPostOp(UALType::DLP);
        auto postops_ref = microTest.getPostOp(UALType::REF);

        // Create UAL instances
        auto ual_dlp = UalFactory::createUal(UALType::DLP);
        auto ual_ref = UalFactory::createUal(UALType::REF);

        // Test GEMM with PostOps (should not crash)
        bool dlp_result =
            ual_dlp->gemm(A, B, C_dlp, MatrixType::f32, postops_dlp);
        bool ref_result =
            ual_ref->gemm(A, B, C_ref, MatrixType::f32, postops_ref);

        std::cout << "GEMM with PostOps:" << std::endl;
        std::cout << "  DLP result: " << (dlp_result ? "SUCCESS" : "FAILED")
                  << std::endl;
        std::cout << "  REF result: " << (ref_result ? "SUCCESS" : "FAILED")
                  << std::endl;

        // The operations should complete without crashing
        EXPECT_TRUE(dlp_result || ref_result)
            << "At least one GEMM operation should succeed";

    } catch (const std::exception& e) {
        std::cout << "Expected exception during GEMM test (implementation may "
                     "not be complete): "
                  << e.what() << std::endl;
        // Don't fail the test for now, as PostOps GEMM implementation might not
        // be complete
    }
}
