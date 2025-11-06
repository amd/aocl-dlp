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

#pragma once

#include "bench_config.hh" // For BENCH_CONFIG_DIR macro
#include "framework/matrix.hh"
#include <string>
#include <vector>

// Forward declaration outside the benchmarking namespace
namespace dlp { namespace testing { namespace utils {
    struct FillValueConfig;
}}} // namespace dlp::testing::utils

namespace dlp::benchmarking {

using namespace dlp::testing::framework;

/**
 * @brief Configuration for a single GEMM benchmark
 */
struct GemmBenchConfig
{
    std::string  name;
    MatrixType   a_type, b_type, c_type, acc_type;
    MatrixLayout storage_format;
    md_t         m, n, k;
    md_t         lda, ldb, ldc;
    double       alpha, beta;
    bool         transA, transB;
    bool         reorderA, reorderB;

    // Optional fill_value configuration
    bool        has_fill_value         = false;
    double      fill_lb                = -5.0;
    double      fill_ub                = 5.0;
    std::string fill_dist              = "uniform";
    bool        force_int_distribution = true;

    // Default constructor
    GemmBenchConfig()
        : name("default")
        , a_type(MatrixType::f32)
        , b_type(MatrixType::f32)
        , c_type(MatrixType::f32)
        , acc_type(MatrixType::f32)
        , storage_format(MatrixLayout::ROW_MAJOR)
        , m(1)
        , n(1)
        , k(1)
        , lda(1)
        , ldb(1)
        , ldc(1)
        , alpha(1.0)
        , beta(0.0)
        , transA(false)
        , transB(false)
        , reorderA(false)
        , reorderB(false)
    {
    }

    /**
     * @brief Generate hash for caching
     */
    size_t hash() const;
};

/**
 * @brief Load benchmark configurations from YAML file
 * @param yaml_path Path to YAML configuration file
 * @return Vector of benchmark configurations
 */
std::vector<GemmBenchConfig>
loadBenchmarkConfigs(const std::string& yaml_path);

/**
 * @brief Generate meaningful benchmark name from config
 */
std::string
generateBenchmarkName(const GemmBenchConfig& config);

} // namespace dlp::benchmarking
