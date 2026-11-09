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
 * Test to demonstrate PostOps validation in SIMPLE_PRODUCT mode
 */

#include "framework/utils/yaml_parser.hh"
#include "test_config.hh"
#include <gtest/gtest.h>
#include <iostream>

using namespace dlp::testing::utils;

TEST(PostOpsValidation, ValidBroadcastPattern)
{
    // 3 GEMM groups, 1 PostOps combo → should pass (broadcast)
    std::cout << "\n=== Testing Valid Broadcast Pattern ===" << std::endl;
    std::cout << "3 GEMM groups, 1 PostOps combo (broadcast)" << std::endl;

    EXPECT_NO_THROW({
        YamlParser parser(TEST_CONFIG_DIR
                          "/yaml_framework_test_configs/validation_test.yaml",
                          "batch_gemm_tests");

        // Get first test (Valid_BroadcastPattern)
        MicroTest& microTest = parser.getMicroTest();
        std::cout << "✓ Validation passed: Broadcast pattern accepted"
                  << std::endl;
        std::cout << "  GEMM groups: " << microTest.getSize() << std::endl;
    });
}

TEST(PostOpsValidation, ValidExactMatch)
{
    // 2 GEMM groups, 2 PostOps combos → should pass (exact match)
    std::cout << "\n=== Testing Valid Exact Match ===" << std::endl;
    std::cout << "2 GEMM groups, 2 PostOps combos (exact match)" << std::endl;

    EXPECT_NO_THROW({
        YamlParser parser(TEST_CONFIG_DIR
                          "/yaml_framework_test_configs/validation_test.yaml",
                          "batch_gemm_tests");

        // Get second test (Valid_ExactMatch)
        parser.next();
        MicroTest& microTest = parser.getMicroTest();
        std::cout << "✓ Validation passed: Exact match accepted" << std::endl;
        std::cout << "  GEMM groups: " << microTest.getSize() << std::endl;
    });
}

TEST(PostOpsValidation, InvalidExtraPostOps)
{
    // 2 GEMM groups, 3 PostOps combos → should fail
    std::cout << "\n=== Testing Invalid Extra PostOps ===" << std::endl;
    std::cout << "2 GEMM groups, 3 PostOps combos (should fail)" << std::endl;

    try {
        YamlParser parser(TEST_CONFIG_DIR
                          "/yaml_framework_test_configs/validation_test.yaml",
                          "batch_gemm_tests");

        // Get third test (Invalid_ExtraPostOps)
        parser.next();
        parser.next();
        MicroTest& microTest = parser.getMicroTest();

        FAIL() << "Expected validation error but none was thrown!";
    } catch (const std::runtime_error& e) {
        std::cout << "✓ Validation correctly caught the error:" << std::endl;
        std::cout << e.what() << std::endl;
        EXPECT_TRUE(
            std::string(e.what()).find("SIMPLE_PRODUCT mode validation failed")
            != std::string::npos);
    }
}

TEST(PostOpsValidation, InvalidMissingPostOps)
{
    // 3 GEMM groups, 2 PostOps combos → should fail
    std::cout << "\n=== Testing Invalid Missing PostOps ===" << std::endl;
    std::cout << "3 GEMM groups, 2 PostOps combos (should fail)" << std::endl;

    try {
        YamlParser parser(TEST_CONFIG_DIR
                          "/yaml_framework_test_configs/validation_test.yaml",
                          "batch_gemm_tests");

        // Get fourth test (Invalid_MissingPostOps)
        parser.next();
        parser.next();
        parser.next();
        MicroTest& microTest = parser.getMicroTest();

        FAIL() << "Expected validation error but none was thrown!";
    } catch (const std::runtime_error& e) {
        std::cout << "✓ Validation correctly caught the error:" << std::endl;
        std::cout << e.what() << std::endl;
        EXPECT_TRUE(
            std::string(e.what()).find("SIMPLE_PRODUCT mode validation failed")
            != std::string::npos);
    }
}

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
