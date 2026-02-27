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

/**
 * @file test_bench_utilities.cc
 * @brief Unit tests for benchmark framework utility functions
 */

#include "bench_config.hh"
#include "bench_test_config.hh"
#include "bench_types.hh"
#include "framework/matrix.hh"

#include <gtest/gtest.h>
#include <string>
#include <unordered_set>
#include <vector>

using namespace dlp::benchmarking;
using namespace dlp::testing::framework;

// Forward declaration of BenchmarkMetrics class to test getMatrixTypeSize
// without pulling in Google Benchmark dependencies
namespace dlp::benchmarking {
class BenchmarkMetrics
{
  public:
    static size_t getMatrixTypeSize(MatrixType type);
};
} // namespace dlp::benchmarking

// ============================================================================
// Test Suite: GemmBenchConfig Hash Function
// ============================================================================

class BenchConfigHashTest : public ::testing::Test
{
  protected:
    GemmBenchConfig createDefaultConfig()
    {
        GemmBenchConfig config;
        config.m        = 64;
        config.n        = 64;
        config.k        = 64;
        config.a_type   = MatrixType::f32;
        config.b_type   = MatrixType::f32;
        config.c_type   = MatrixType::f32;
        config.acc_type = MatrixType::f32;
        config.transA   = false;
        config.transB   = false;
        config.reorderA = false;
        config.reorderB = false;
        return config;
    }
};

TEST_F(BenchConfigHashTest, HashDeterminism)
{
    GemmBenchConfig config1 = createDefaultConfig();
    GemmBenchConfig config2 = createDefaultConfig();

    // Same configuration should produce same hash
    EXPECT_EQ(config1.hash(), config2.hash());
}

TEST_F(BenchConfigHashTest, HashUniquenessForDimensions)
{
    GemmBenchConfig config1 = createDefaultConfig();
    GemmBenchConfig config2 = createDefaultConfig();
    GemmBenchConfig config3 = createDefaultConfig();

    config1.m = 64;
    config2.m = 128;
    config3.m = 256;

    size_t hash1 = config1.hash();
    size_t hash2 = config2.hash();
    size_t hash3 = config3.hash();

    // Different dimensions should produce different hashes
    EXPECT_NE(hash1, hash2);
    EXPECT_NE(hash2, hash3);
    EXPECT_NE(hash1, hash3);
}

TEST_F(BenchConfigHashTest, HashSensitivityToDataTypes)
{
    GemmBenchConfig config_f32  = createDefaultConfig();
    GemmBenchConfig config_bf16 = createDefaultConfig();

    config_f32.a_type  = MatrixType::f32;
    config_bf16.a_type = MatrixType::bf16;

    EXPECT_NE(config_f32.hash(), config_bf16.hash());
}

TEST_F(BenchConfigHashTest, HashSensitivityToTranspose)
{
    GemmBenchConfig config1 = createDefaultConfig();
    GemmBenchConfig config2 = createDefaultConfig();

    config1.transA = false;
    config2.transA = true;

    EXPECT_NE(config1.hash(), config2.hash());
}

TEST_F(BenchConfigHashTest, HashSensitivityToReordering)
{
    GemmBenchConfig config1 = createDefaultConfig();
    GemmBenchConfig config2 = createDefaultConfig();

    config1.reorderB = false;
    config2.reorderB = true;

    EXPECT_NE(config1.hash(), config2.hash());
}

TEST_F(BenchConfigHashTest, NoCollisionsForCommonConfigs)
{
    std::unordered_set<size_t>   hashes;
    std::vector<GemmBenchConfig> configs;

    // Generate several common configurations
    for (iter_t m : { 64, 128, 256, 512 }) {
        for (iter_t n : { 64, 128, 256, 512 }) {
            for (iter_t k : { 64, 128, 256, 512 }) {
                GemmBenchConfig config = createDefaultConfig();
                config.m               = m;
                config.n               = n;
                config.k               = k;
                configs.push_back(std::move(config));
            }
        }
    }

    // Check for collisions
    for (const auto& config : configs) {
        size_t h = config.hash();
        EXPECT_EQ(hashes.count(h), 0)
            << "Hash collision detected for config with m=" << config.m
            << ", n=" << config.n << ", k=" << config.k;
        hashes.insert(h);
    }
}

// ============================================================================
// Test Suite: Benchmark Name Generation
// ============================================================================

class BenchmarkNameGenerationTest : public ::testing::Test
{
  protected:
    GemmBenchConfig createDefaultConfig()
    {
        GemmBenchConfig config;
        config.m              = 64;
        config.n              = 128;
        config.k              = 256;
        config.a_type         = MatrixType::f32;
        config.b_type         = MatrixType::f32;
        config.c_type         = MatrixType::f32;
        config.acc_type       = MatrixType::f32;
        config.storage_format = MatrixLayout::ROW_MAJOR;
        config.transA         = false;
        config.transB         = false;
        config.reorderA       = false;
        config.reorderB       = false;
        return config;
    }
};

TEST_F(BenchmarkNameGenerationTest, BasicNameFormat)
{
    GemmBenchConfig config = createDefaultConfig();
    std::string     name   = generateBenchmarkName(config);

    // Name should contain dimensions
    EXPECT_NE(name.find("M:64"), std::string::npos);
    EXPECT_NE(name.find("N:128"), std::string::npos);
    EXPECT_NE(name.find("K:256"), std::string::npos);
}

TEST_F(BenchmarkNameGenerationTest, DataTypeEncoding)
{
    GemmBenchConfig config = createDefaultConfig();
    config.a_type          = MatrixType::f32;
    config.b_type          = MatrixType::f32;
    config.acc_type        = MatrixType::f32;
    config.c_type          = MatrixType::f32;

    std::string name = generateBenchmarkName(config);

    // Should start with data type information
    EXPECT_EQ(name.substr(0, 7), "f32f32f");
}

TEST_F(BenchmarkNameGenerationTest, TransposeFlags)
{
    GemmBenchConfig config = createDefaultConfig();

    // Test no transpose
    config.transA     = false;
    config.transB     = false;
    std::string name1 = generateBenchmarkName(config);
    EXPECT_NE(name1.find("trA:n"), std::string::npos);
    EXPECT_NE(name1.find("trB:n"), std::string::npos);

    // Test transpose A
    config.transA     = true;
    config.transB     = false;
    std::string name2 = generateBenchmarkName(config);
    EXPECT_NE(name2.find("trA:t"), std::string::npos);
    EXPECT_NE(name2.find("trB:n"), std::string::npos);

    // Test transpose B
    config.transA     = false;
    config.transB     = true;
    std::string name3 = generateBenchmarkName(config);
    EXPECT_NE(name3.find("trA:n"), std::string::npos);
    EXPECT_NE(name3.find("trB:t"), std::string::npos);
}

TEST_F(BenchmarkNameGenerationTest, ReorderingFlags)
{
    GemmBenchConfig config = createDefaultConfig();

    // Test no reordering
    config.reorderA   = false;
    config.reorderB   = false;
    std::string name1 = generateBenchmarkName(config);
    EXPECT_NE(name1.find("mtagA:n"), std::string::npos);
    EXPECT_NE(name1.find("mtagB:n"), std::string::npos);

    // Test reorder B
    config.reorderA   = false;
    config.reorderB   = true;
    std::string name2 = generateBenchmarkName(config);
    EXPECT_NE(name2.find("mtagA:n"), std::string::npos);
    EXPECT_NE(name2.find("mtagB:r"), std::string::npos);
}

TEST_F(BenchmarkNameGenerationTest, StorageFormat)
{
    GemmBenchConfig config = createDefaultConfig();

    // Test row major
    config.storage_format = MatrixLayout::ROW_MAJOR;
    std::string name1     = generateBenchmarkName(config);
    EXPECT_NE(name1.find("stor:r"), std::string::npos);

    // Test column major
    config.storage_format = MatrixLayout::COLUMN_MAJOR;
    std::string name2     = generateBenchmarkName(config);
    EXPECT_NE(name2.find("stor:c"), std::string::npos);
}

TEST_F(BenchmarkNameGenerationTest, UniquenessForDifferentConfigs)
{
    GemmBenchConfig config1 = createDefaultConfig();
    GemmBenchConfig config2 = createDefaultConfig();

    config1.m = 64;
    config2.m = 128;

    std::string name1 = generateBenchmarkName(config1);
    std::string name2 = generateBenchmarkName(config2);

    EXPECT_NE(name1, name2);
}

// ============================================================================
// Test Suite: Matrix Type Size
// ============================================================================

class MatrixTypeSizeTest : public ::testing::Test
{};

TEST_F(MatrixTypeSizeTest, U4TypeSize)
{
    EXPECT_EQ(BenchmarkMetrics::getMatrixTypeSize(MatrixType::u4), 1);
}

TEST_F(MatrixTypeSizeTest, S4TypeSize)
{
    EXPECT_EQ(BenchmarkMetrics::getMatrixTypeSize(MatrixType::s4), 1);
}

TEST_F(MatrixTypeSizeTest, U8TypeSize)
{
    EXPECT_EQ(BenchmarkMetrics::getMatrixTypeSize(MatrixType::u8), 1);
}

TEST_F(MatrixTypeSizeTest, S8TypeSize)
{
    EXPECT_EQ(BenchmarkMetrics::getMatrixTypeSize(MatrixType::s8), 1);
}

TEST_F(MatrixTypeSizeTest, U16TypeSize)
{
    EXPECT_EQ(BenchmarkMetrics::getMatrixTypeSize(MatrixType::u16), 2);
}

TEST_F(MatrixTypeSizeTest, S16TypeSize)
{
    EXPECT_EQ(BenchmarkMetrics::getMatrixTypeSize(MatrixType::s16), 2);
}

TEST_F(MatrixTypeSizeTest, BF16TypeSize)
{
    EXPECT_EQ(BenchmarkMetrics::getMatrixTypeSize(MatrixType::bf16), 2);
}

TEST_F(MatrixTypeSizeTest, U32TypeSize)
{
    EXPECT_EQ(BenchmarkMetrics::getMatrixTypeSize(MatrixType::u32), 4);
}

TEST_F(MatrixTypeSizeTest, S32TypeSize)
{
    EXPECT_EQ(BenchmarkMetrics::getMatrixTypeSize(MatrixType::s32), 4);
}

TEST_F(MatrixTypeSizeTest, F32TypeSize)
{
    EXPECT_EQ(BenchmarkMetrics::getMatrixTypeSize(MatrixType::f32), 4);
}

// ============================================================================
// Test Suite: YAML Configuration Loading
// ============================================================================

class YamlConfigLoadingTest : public ::testing::Test
{
  protected:
    std::string getTestConfigPath()
    {
        return std::string(BENCH_TEST_CONFIG_DIR) + "/bench_test_minimal.yaml";
    }
};

TEST_F(YamlConfigLoadingTest, LoadValidYaml)
{
    std::string yaml_path = getTestConfigPath();
    auto        configs   = loadBenchmarkConfigs(yaml_path);

    // Should load successfully
    EXPECT_FALSE(configs.empty());
}

TEST_F(YamlConfigLoadingTest, ConfigCount)
{
    std::string yaml_path = getTestConfigPath();
    auto        configs   = loadBenchmarkConfigs(yaml_path);

    // Our minimal config has 2 test cases, each with 1 simple product config
    EXPECT_EQ(configs.size(), 2);
}

TEST_F(YamlConfigLoadingTest, FirstConfigFields)
{
    std::string yaml_path = getTestConfigPath();
    auto        configs   = loadBenchmarkConfigs(yaml_path);

    ASSERT_GE(configs.size(), 1);

    const auto& config = configs[0];

    // Verify first config values
    EXPECT_EQ(config.m, 64);
    EXPECT_EQ(config.n, 64);
    EXPECT_EQ(config.k, 64);
    EXPECT_EQ(config.a_type, MatrixType::f32);
    EXPECT_EQ(config.b_type, MatrixType::f32);
    EXPECT_EQ(config.c_type, MatrixType::f32);
    EXPECT_EQ(config.acc_type, MatrixType::f32);
    EXPECT_EQ(config.transA, false);
    EXPECT_EQ(config.transB, false);
    EXPECT_EQ(config.reorderA, false);
    EXPECT_EQ(config.reorderB, false);
}

TEST_F(YamlConfigLoadingTest, SecondConfigFields)
{
    std::string yaml_path = getTestConfigPath();
    auto        configs   = loadBenchmarkConfigs(yaml_path);

    ASSERT_GE(configs.size(), 2);

    const auto& config = configs[1];

    // Verify second config values
    EXPECT_EQ(config.m, 128);
    EXPECT_EQ(config.n, 128);
    EXPECT_EQ(config.k, 128);
    EXPECT_EQ(config.a_type, MatrixType::bf16);
    EXPECT_EQ(config.b_type, MatrixType::bf16);
    EXPECT_EQ(config.c_type, MatrixType::f32);
    EXPECT_EQ(config.acc_type, MatrixType::f32);
    EXPECT_EQ(config.transA, true);
    EXPECT_EQ(config.transB, false);
    EXPECT_EQ(config.reorderA, false);
    EXPECT_EQ(config.reorderB, true);
}

TEST_F(YamlConfigLoadingTest, ConfigNameGeneration)
{
    std::string yaml_path = getTestConfigPath();
    auto        configs   = loadBenchmarkConfigs(yaml_path);

    ASSERT_GE(configs.size(), 1);

    // Each config should have a generated name
    for (const auto& config : configs) {
        EXPECT_FALSE(config.name.empty());
        // Name should contain dimension info
        EXPECT_NE(config.name.find("M:"), std::string::npos);
        EXPECT_NE(config.name.find("N:"), std::string::npos);
        EXPECT_NE(config.name.find("K:"), std::string::npos);
    }
}

TEST_F(YamlConfigLoadingTest, NonexistentFileHandling)
{
    std::string yaml_path = "/nonexistent/path/to/config.yaml";
    auto        configs   = loadBenchmarkConfigs(yaml_path);

    // Should return empty vector on error
    EXPECT_TRUE(configs.empty());
}

// ============================================================================
// Main
// ============================================================================

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
