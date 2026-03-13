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

class ComprehensivePostOpsTest : public ::testing::Test
{
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ComprehensivePostOpsTest, AllPostOpsTypesRangeListTest)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_config_range_list.yaml";

    try {
        YamlParser parser(filepath, "yaml_test");

        // Test the first test case (comprehensive PostOps)
        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "=== Range-List PostOps Test - Small Matrix ==="
                  << std::endl;
        std::cout << "Test parameters:" << std::endl;
        std::cout << "  M: " << microTest.getM() << std::endl;
        std::cout << "  N: " << microTest.getN() << std::endl;
        std::cout << "  K: " << microTest.getK() << std::endl;
        std::cout << "  Total combinations: " << microTest.getSize()
                  << std::endl;

        // Test PostOps functionality
        int combination_count = 0;
        int postops_found     = 0;

        auto& mutableMicroTest = const_cast<MicroTest&>(microTest);

        // Track PostOps types found
        std::set<std::string> postops_types_found;

        do {
            auto dlp_ops = microTest.getPostOp(UALType::DLP);
            auto ref_ops = microTest.getPostOp(UALType::REF);

            if (dlp_ops || ref_ops) {
                postops_found++;

                // Log first few combinations to see variety
                if (combination_count < 20) {
                    std::cout << "  Combination " << combination_count
                              << ": PostOps present" << std::endl;
                }
            }

            combination_count++;

            // Limit iterations for this test
            if (combination_count >= 100)
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

        // Should have many PostOps combinations due to cartesian=true
        EXPECT_GT(postops_found, 50)
            << "Should find many PostOps combinations with cartesian=true";

        // Test medium matrix (simpler PostOps)
        parser.next();
        const MicroTest& microTest2 = parser.getMicroTest();

        std::cout << "=== Medium Matrix Test (Simpler PostOps) ==="
                  << std::endl;
        std::cout << "  M: " << microTest2.getM() << std::endl;
        std::cout << "  N: " << microTest2.getN() << std::endl;
        std::cout << "  Total combinations: " << microTest2.getSize()
                  << std::endl;

        auto postops_dlp_2 = microTest2.getPostOp(UALType::DLP);
        auto postops_ref_2 = microTest2.getPostOp(UALType::REF);

        std::cout << "  DLP PostOps: " << (postops_dlp_2 ? "PRESENT" : "NONE")
                  << std::endl;
        std::cout << "  REF PostOps: " << (postops_ref_2 ? "PRESENT" : "NONE")
                  << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Range-List PostOps test threw an exception: " << e.what();
    }
}

TEST_F(ComprehensivePostOpsTest, AllPostOpsTypesListTest)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_config_list.yaml";

    try {
        YamlParser       parser(filepath, "yaml_test");
        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "=== List-Based PostOps Test ===" << std::endl;
        std::cout << "  M: " << microTest.getM() << std::endl;
        std::cout << "  N: " << microTest.getN() << std::endl;
        std::cout << "  Total combinations: " << microTest.getSize()
                  << std::endl;

        // Verify PostOps are working
        auto postops_dlp = microTest.getPostOp(UALType::DLP);
        auto postops_ref = microTest.getPostOp(UALType::REF);

        EXPECT_TRUE(postops_dlp || postops_ref)
            << "Should have PostOps in list configuration";

        // Test a few combinations to verify variety
        int   tested           = 0;
        int   with_postops     = 0;
        auto& mutableMicroTest = const_cast<MicroTest&>(microTest);

        do {
            auto dlp_ops = microTest.getPostOp(UALType::DLP);
            auto ref_ops = microTest.getPostOp(UALType::REF);

            if (dlp_ops || ref_ops) {
                with_postops++;
            }

            tested++;
            if (tested >= 50)
                break; // Limit test scope

            if (mutableMicroTest.hasNext()) {
                mutableMicroTest.next();
            } else {
                break;
            }
        } while (true);

        std::cout << "  Tested " << tested << " combinations" << std::endl;
        std::cout << "  PostOps present in " << with_postops << " combinations"
                  << std::endl;

        EXPECT_GT(with_postops, 10)
            << "Should find significant PostOps combinations";

    } catch (const std::exception& e) {
        FAIL() << "List PostOps test threw an exception: " << e.what();
    }
}

TEST_F(ComprehensivePostOpsTest, AllPostOpsTypesValueTest)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_config_value.yaml";

    try {
        YamlParser parser(filepath, "yaml_test");

        // Test first case
        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "=== Value-Based PostOps Test - Small Matrix ==="
                  << std::endl;
        std::cout << "  M: " << microTest.getM() << std::endl;
        std::cout << "  N: " << microTest.getN() << std::endl;
        std::cout << "  Total combinations: " << microTest.getSize()
                  << std::endl;

        auto postops_dlp = microTest.getPostOp(UALType::DLP);
        auto postops_ref = microTest.getPostOp(UALType::REF);

        std::cout << "  DLP PostOps: " << (postops_dlp ? "PRESENT" : "NONE")
                  << std::endl;
        std::cout << "  REF PostOps: " << (postops_ref ? "PRESENT" : "NONE")
                  << std::endl;

        // Test medium matrix
        parser.next();
        const MicroTest& microTest2 = parser.getMicroTest();

        std::cout << "=== Value-Based PostOps Test - Medium Matrix ==="
                  << std::endl;
        std::cout << "  M: " << microTest2.getM() << std::endl;
        std::cout << "  N: " << microTest2.getN() << std::endl;
        std::cout << "  Total combinations: " << microTest2.getSize()
                  << std::endl;

        auto postops_dlp_2 = microTest2.getPostOp(UALType::DLP);
        auto postops_ref_2 = microTest2.getPostOp(UALType::REF);

        std::cout << "  DLP PostOps: " << (postops_dlp_2 ? "PRESENT" : "NONE")
                  << std::endl;
        std::cout << "  REF PostOps: " << (postops_ref_2 ? "PRESENT" : "NONE")
                  << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Value PostOps test threw an exception: " << e.what();
    }
}

TEST_F(ComprehensivePostOpsTest, PostOpsWithGemmIntegrationTest)
{
    std::string filepath = TEST_CONFIG_DIR
        "/yaml_framework_test_configs/yaml_test_config_list.yaml";

    try {
        YamlParser       parser(filepath, "yaml_test");
        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "=== PostOps GEMM Integration Test ===" << std::endl;

        // Create matrices for testing
        Matrix A(10, 10, MatrixType::f32, MatrixLayout::ROW_MAJOR, -1, false);
        Matrix B(10, 10, MatrixType::f32, MatrixLayout::ROW_MAJOR, -1, false);
        Matrix C_dlp(10, 10, MatrixType::f32, MatrixLayout::ROW_MAJOR, -1,
                     false);
        Matrix C_ref(10, 10, MatrixType::f32, MatrixLayout::ROW_MAJOR, -1,
                     false);

        // Initialize matrices
        A.fillValue(0.5f);
        B.fillValue(0.2f);
        C_dlp.fillValue(0.0f);
        C_ref.fillValue(0.0f);

        // Test multiple PostOps combinations
        int   successful_gemm_ops = 0;
        int   total_tested        = 0;
        auto& mutableMicroTest    = const_cast<MicroTest&>(microTest);

        // Create UAL instances once
        auto ual_dlp = UalFactory::createUal(UALType::DLP);
        auto ual_ref = UalFactory::createUal(UALType::REF);

        do {
            auto postops_dlp = microTest.getPostOp(UALType::DLP);
            auto postops_ref = microTest.getPostOp(UALType::REF);

            if (postops_dlp || postops_ref) {
                try {
                    // Test GEMM with PostOps
                    UALError dlp_status = UALError::UAL_FAILURE;
                    UALError ref_status = UALError::UAL_FAILURE;

                    if (postops_dlp) {
                        dlp_status = ual_dlp->gemm(A, B, C_dlp, MatrixType::f32,
                                                   postops_dlp);
                        // Skip test if ISA not supported
                        if (dlp_status == UALError::UAL_NOT_SUPPORTED) {
                            GTEST_SKIP() << "DLP GEMM not supported on this "
                                            "processor (ISA not available)";
                        }
                    }

                    if (postops_ref) {
                        ref_status = ual_ref->gemm(A, B, C_ref, MatrixType::f32,
                                                   postops_ref);
                    }

                    if (dlp_status == UALError::UAL_SUCCESS
                        || ref_status == UALError::UAL_SUCCESS) {
                        successful_gemm_ops++;

                        if (total_tested < 5) {
                            std::cout << "  GEMM " << total_tested + 1
                                      << ": DLP="
                                      << (dlp_status == UALError::UAL_SUCCESS
                                              ? "SUCCESS"
                                              : "SKIPPED")
                                      << ", REF="
                                      << (ref_status == UALError::UAL_SUCCESS
                                              ? "SUCCESS"
                                              : "SKIPPED")
                                      << std::endl;
                        }
                    }
                } catch (const std::exception& e) {
                    // Some PostOps combinations might not be supported yet
                    std::cout << "  GEMM " << total_tested + 1
                              << ": Expected exception: " << e.what()
                              << std::endl;
                }
            }

            total_tested++;
            if (total_tested >= 20)
                break; // Limit test scope

            if (mutableMicroTest.hasNext()) {
                mutableMicroTest.next();
            } else {
                break;
            }
        } while (true);

        std::cout << "  Tested " << total_tested << " PostOps combinations"
                  << std::endl;
        std::cout << "  Successful GEMM operations: " << successful_gemm_ops
                  << std::endl;

        // Expect at least some successful operations
        EXPECT_GT(successful_gemm_ops, 0)
            << "Should have some successful GEMM operations with PostOps";

    } catch (const std::exception& e) {
        FAIL() << "GEMM integration test threw an exception: " << e.what();
    }
}

TEST_F(ComprehensivePostOpsTest, BackwardCompatibilityTest)
{
    // Ensure that YAML files without PostOps still work
    try {
        // This test uses our custom PostOps file without the second test case
        std::string filepath = TEST_CONFIG_DIR
            "/yaml_framework_test_configs/yaml_test_postops.yaml";
        YamlParser parser(filepath, "yaml_test");

        // Skip to the second test case (no PostOps)
        parser.next();
        const MicroTest& microTest = parser.getMicroTest();

        std::cout << "=== Backward Compatibility Test ===" << std::endl;
        std::cout << "  Test without PostOps should work normally" << std::endl;
        std::cout << "  M: " << microTest.getM() << std::endl;
        std::cout << "  N: " << microTest.getN() << std::endl;
        std::cout << "  Total combinations: " << microTest.getSize()
                  << std::endl;

        auto postops_dlp = microTest.getPostOp(UALType::DLP);
        auto postops_ref = microTest.getPostOp(UALType::REF);

        EXPECT_EQ(postops_dlp, nullptr) << "Should have no PostOps";
        EXPECT_EQ(postops_ref, nullptr) << "Should have no PostOps";

        std::cout << "  PostOps: CORRECTLY ABSENT" << std::endl;

        // Test normal GEMM operation
        Matrix A(4, 4, MatrixType::f32, MatrixLayout::ROW_MAJOR, -1, false);
        Matrix B(4, 4, MatrixType::f32, MatrixLayout::ROW_MAJOR, -1, false);
        Matrix C(4, 4, MatrixType::f32, MatrixLayout::ROW_MAJOR, -1, false);

        A.fillValue(0.5f);
        B.fillValue(0.2f);
        C.fillValue(0.0f);

        auto     ual_dlp = UalFactory::createUal(UALType::DLP);
        UALError status  = ual_dlp->gemm(A, B, C, MatrixType::f32, postops_dlp);

        // Skip test if ISA not supported
        if (status == UALError::UAL_NOT_SUPPORTED) {
            GTEST_SKIP() << "DLP GEMM not supported on this processor "
                         << "(ISA not available)";
        }

        EXPECT_EQ(status, UALError::UAL_SUCCESS)
            << "GEMM without PostOps should work";
        std::cout << "  GEMM without PostOps: SUCCESS" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Backward compatibility test threw an exception: "
               << e.what();
    }
}
