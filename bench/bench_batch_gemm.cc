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
 * @file bench_batch_gemm.cc
 * @brief Optimized Batch GEMM Benchmark with Metadata Preparation Optimization
 *
 * KEY OPTIMIZATION:
 * This benchmark implements a critical optimization that separates metadata
 * preparation from the hot execution path:
 *
 * 1. batch_prepare_metadata() called ONCE in constructor
 *    - Performs all allocations (std::make_shared)
 *    - Performs all type casts (dynamic_cast)
 *    - Stores metadata pointers in prepared_args_
 *
 * 2. batch_gemm(prepared_args_) in hot loop
 *    - ZERO allocations
 *    - ZERO casts
 *    - Pure kernel execution
 *
 * RESULT: Accurate GFLOPS measurement without framework overhead distortion
 */

#include "bench_metrics.hh"
#include "bench_types.hh"
#include "cold_cache.hh"

#include "adaptors/dlp/ual_dlp.hh"
#include "framework/batch_gemm_args.hh"
#include "framework/matrix.hh"
#include "framework/ual.hh"
#include "framework/ual_factory.hh"
#include "framework/utils/arg_parser.hh"
#include "framework/utils/yaml_parser.hh"

// OpenMP for detailed threading info
#ifdef DLP_ENABLE_OPENMP
#include <omp.h>
#endif

#include <benchmark/benchmark.h>

#include <cctype>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dlp::testing::framework;
using namespace dlp::testing::classic;
using namespace dlp::testing::utils;
using namespace dlp::benchmarking;

// ============================================================================
// BATCH GEMM CONFIG TO GROUPS CONVERSION
// ============================================================================

/**
 * @brief Convert BatchGemmBenchConfig to vector of BatchGroup
 *
 * This creates the actual Matrix objects with allocated memory
 */
std::vector<BatchGroup>
configToGroups(const BatchGemmBenchConfig& config)
{
    std::vector<BatchGroup> groups;

    const size_t alignment = 4096;

    // Handle both single-group and multi-group configurations
    size_t num_groups = config.group_sizes.empty() ? 1
                                                   : config.group_sizes.size();

    for (std::size_t g = 0; g < num_groups; ++g) {
        BatchGroup group;

        // Get dimensions for this group (scalar values broadcast to all groups)
        md_t m = config.ms.empty()
                     ? config.m
                     : (config.ms.size() == 1 ? config.ms[0] : config.ms[g]);
        md_t n = config.ns.empty()
                     ? config.n
                     : (config.ns.size() == 1 ? config.ns[0] : config.ns[g]);
        md_t k = config.ks.empty()
                     ? config.k
                     : (config.ks.size() == 1 ? config.ks[0] : config.ks[g]);

        group.m = m;
        group.n = n;
        group.k = k;

        // Get alpha/beta for this group
        group.alpha = config.alphas.empty()
                          ? config.alpha
                          : (config.alphas.size() == 1 ? config.alphas[0]
                                                       : config.alphas[g]);
        group.beta  = config.betas.empty()
                          ? config.beta
                          : (config.betas.size() == 1 ? config.betas[0]
                                                      : config.betas[g]);

        // Get group size
        size_t group_size = config.group_sizes.empty() ? config.group_size
                                                       : config.group_sizes[g];

        // Create matrices for this group
        for (std::size_t i = 0; i < group_size; ++i) {
            bool transA = config.transAs.empty() ? config.transA
                                                 : (config.transAs.size() == 1
                                                        ? config.transAs[0]
                                                        : config.transAs[g]);
            bool transB = config.transBs.empty() ? config.transB
                                                 : (config.transBs.size() == 1
                                                        ? config.transBs[0]
                                                        : config.transBs[g]);

            md_t a_rows = transA ? k : m;
            md_t a_cols = transA ? m : k;
            md_t b_rows = transB ? n : k;
            md_t b_cols = transB ? k : n;

            Matrix A(a_rows, a_cols, config.a_type, config.storage_format,
                     a_cols, transA, false, alignment);
            Matrix B(b_rows, b_cols, config.b_type, config.storage_format,
                     b_cols, transB, false, alignment);
            Matrix C(m, n, config.c_type, config.storage_format, n, false,
                     false, alignment);

            // Initialize with random data
            A.fillRandom(42 + g * 100 + i);
            B.fillRandom(43 + g * 100 + i);
            C.fillRandom(44 + g * 100 + i);

            group.A_matrices.push_back(std::move(A));
            group.B_matrices.push_back(std::move(B));
            group.C_matrices.push_back(std::move(C));
        }

        groups.push_back(std::move(group));
    }

    return groups;
}

// ============================================================================
// OPTIMIZED BATCH GEMM BENCHMARK WITH METADATA PREPARATION
// ============================================================================

/**
 * @brief Optimized batch GEMM benchmark using template inheritance
 *
 * CRITICAL OPTIMIZATION:
 * - batch_prepare_metadata() called ONCE in constructor
 * - batch_gemm(prepared_args_) in hot loop uses pre-computed metadata
 * - Eliminates per-iteration allocation and casting overhead
 */
template<typename ConcreteUAL>
class OptimizedBatchGemmBenchmark : public ConcreteUAL
{
  public:
    // Constructor: allocate matrices and prepare metadata ONCE
    OptimizedBatchGemmBenchmark(const BatchGemmBenchConfig& config)
        : config_(config)
    {
        // Step 1: Create BatchGroups with allocated matrices
        groups_ = configToGroups(config_);

        // Step 2: Prepare arguments (pre-calculate pointers, dimensions, etc.)
        UALError prep_status =
            prepare_batch_gemm_args(groups_, config_.acc_type, prepared_args_);
        if (prep_status != UALError::UAL_SUCCESS) {
            throw std::runtime_error("Failed to prepare batch GEMM arguments");
        }

        // Step 3: CRITICAL OPTIMIZATION - Prepare backend-specific metadata
        // ONCE This call performs all std::make_shared and dynamic_cast
        // operations The metadata pointers are cached in
        // prepared_args_.backend_metadata
        this->batch_prepare_metadata(prepared_args_);

        // Trial-execute once so missing classic APIs and runtime/ISA rejects
        // are skipped at registration (same as GEMM bench / GTest), not
        // reported as a warmup or measured-loop ERROR.
        m_probe_status = this->batch_gemm(prepared_args_);

        // Calculate total operations for GFLOPS
        total_flops_ = 0;
        for (std::size_t g = 0; g < groups_.size(); ++g) {
            md_t   m                 = groups_[g].m;
            md_t   n                 = groups_[g].n;
            md_t   k                 = groups_[g].k;
            size_t matrices_in_group = groups_[g].size();

            // Each matrix multiply: 2*M*N*K FLOPs
            total_flops_ += matrices_in_group * 2.0 * m * n * k;
        }
    }

    UALError probeStatus() const { return m_probe_status; }

    // Benchmark execution
    void run(benchmark::State& state)
    {
        const bool cold = dlp::bench::coldCacheEnabled();

        // WARMUP: 5 iterations to stabilize CPU/cache (skip when measuring
        // cold-cache behaviour — warming defeats the point).
        if (!cold) {
            for (iter_t i = 0; i < 5; ++i) {
                UALError status = this->batch_gemm(prepared_args_);
                if (status != UALError::UAL_SUCCESS) {
                    if (const char* reason = ualRejectionMessage(status)) {
                        state.SkipWithError(reason);
                    } else {
                        state.SkipWithError("Warmup batch_gemm failed");
                    }
                    return;
                }
            }
        }

        // MEASURED LOOP: time only batch_gemm() via std::chrono and submit
        // to gbench through SetIterationTime() (UseManualTime). The optional
        // cold-cache flush runs before the timed region so it is excluded
        // from the reported time.
        bool status = true;
        for (auto _ : state) {
            if (cold)
                dlp::bench::flushColdCache();

            const auto t0 = std::chrono::steady_clock::now();
            status &= this->batch_gemm(prepared_args_) == UALError::UAL_SUCCESS;
            const auto t1 = std::chrono::steady_clock::now();

            const double iter_seconds =
                std::chrono::duration<double>(t1 - t0).count();
            state.SetIterationTime(iter_seconds);

            // Prevent compiler from optimizing away memory writes
            benchmark::ClobberMemory();
        }
        if (status != true) {
            state.SkipWithError("Measured batch_gemm failed");
            return;
        }

        // Calculate and report metrics using Google Benchmark Counter API
        // The counter automatically handles rate calculation
        state.counters["GFLOPS"] = benchmark::Counter(
            total_flops_, benchmark::Counter::kIsIterationInvariantRate,
            benchmark::Counter::kIs1000);

        // Report configuration details
        state.counters["Groups"] = static_cast<double>(groups_.size());
        state.counters["TotalMatrices"] =
            static_cast<double>(prepared_args_.total_matrices);

        // Report dimensions of first group
        if (!groups_.empty()) {
            state.counters["M"] = static_cast<double>(groups_[0].m);
            state.counters["N"] = static_cast<double>(groups_[0].n);
            state.counters["K"] = static_cast<double>(groups_[0].k);
        }
    }

  private:
    const BatchGemmBenchConfig& config_;

    // Storage (allocated once in constructor)
    std::vector<BatchGroup> groups_;
    PreparedBatchGemmArgs   prepared_args_;
    double                  total_flops_;
    UALError                m_probe_status = UALError::UAL_SUCCESS;
};

// Typedef for DLP backend
using OptimizedBatchGemmBenchmarkDlp = OptimizedBatchGemmBenchmark<UalDlp>;

// ============================================================================
// BENCHMARK REGISTRATION
// ============================================================================

/**
 * @brief Register benchmarks with optimized fixture management
 *
 * KEY: Create ONE fixture instance per benchmark configuration,
 * stored in a vector to keep them alive for the entire benchmark run.
 *
 * @param configs Vector of benchmark configurations to register
 * @param iterations Number of iterations to run (-1 for default MinTime
 * behavior)
 */
void
registerOptimizedBenchmarks(const std::vector<BatchGemmBenchConfig>& configs,
                            int64_t iterations     = -1,
                            double  bench_min_time = 3.0)
{
    // Store fixture instances to keep them alive
    static std::vector<std::unique_ptr<OptimizedBatchGemmBenchmarkDlp>>
        fixtures;

    std::cerr << "================================================"
              << std::endl;
    if (iterations > 0) {
        std::cerr << "Benchmark mode: Fixed iterations (" << iterations << ")"
                  << std::endl;
    } else {
        std::cerr << "Benchmark mode: MinTime (" << bench_min_time
                  << " seconds)" << std::endl;
    }
    std::cerr << "================================================"
              << std::endl;

    for (const auto& config : configs) {
        try {
            // Create fixture ONCE per benchmark
            auto fixture =
                std::make_unique<OptimizedBatchGemmBenchmarkDlp>(config);

            const auto probe_status = fixture->probeStatus();
            if (probe_status != UALError::UAL_SUCCESS) {
                if (const char* reason = ualRejectionMessage(probe_status)) {
                    std::cerr << "Skipping (" << reason << "): " << config.name
                              << std::endl;
                } else {
                    std::cerr << "Skipping (probe failed; status="
                              << static_cast<int>(probe_status)
                              << "): " << config.name << std::endl;
                }
                continue;
            }

            // Capture raw pointer (fixture lifetime managed by static vector)
            auto* fixture_ptr = fixture.get();
            fixtures.push_back(std::move(fixture));

            // Register benchmark with lambda that uses existing fixture
            // Use Iterations() if specified, otherwise use MinTime()
            if (iterations > 0) {
                benchmark::RegisterBenchmark(
                    config.name.c_str(),
                    [fixture_ptr](benchmark::State& st) {
                        fixture_ptr->run(st);
                    })
                    ->Unit(benchmark::kMillisecond)
                    ->UseManualTime()
                    ->Iterations(
                        static_cast<benchmark::IterationCount>(iterations));
            } else {
                benchmark::RegisterBenchmark(
                    config.name.c_str(),
                    [fixture_ptr](benchmark::State& st) {
                        fixture_ptr->run(st);
                    })
                    ->Unit(benchmark::kMillisecond)
                    ->UseManualTime()
                    ->MinTime(bench_min_time);
            }
        } catch (const std::exception& ex) {
            std::cerr << "Skipping benchmark '" << config.name
                      << "' due to exception during fixture construction: "
                      << ex.what() << std::endl;
        } catch (...) {
            std::cerr
                << "Skipping benchmark '" << config.name
                << "' due to unknown exception during fixture construction."
                << std::endl;
        }
    }
}

// ============================================================================
// MAIN
// ============================================================================

int
main(int argc, char** argv)
{
    // Parse custom arguments before Google Benchmark processes them
    auto parser = ArgParser::parseTestArgs(argc, argv);

    // Handle help request
    if (parser.helpRequested()) {
        parser.printUsage(argv[0]);
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Google Benchmark Help:" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    }

    // Get YAML configuration file path(s). Multiple files may be provided via
    // repeated -f/--file flags and/or comma-separated values. The built-in
    // default is threaded through getYamlFiles() so it is only used when no
    // -f/--file flag was given.
    std::vector<std::string> yaml_files =
        parser.getYamlFiles(BENCH_CONFIG_DIR "/batch_gemm_bench_config.yaml");

    if (yaml_files.empty()) {
        // The user explicitly passed -f/--file but every supplied path was
        // invalid (or the default itself is missing). Fail loudly rather than
        // silently benchmarking a different configuration (a mistyped -f path
        // in CI must not pass by running the default suite).
        std::cerr << "Error: No valid YAML configuration file(s) provided!"
                  << std::endl;
        if (parser.hasYamlFileArg()) {
            std::cerr << "All paths given via -f/--file were invalid."
                      << std::endl;
        }
        std::cerr << "Please check the file path(s) or run with -h for usage "
                     "information."
                  << std::endl;
        return 1;
    }

    if (parser.hasYamlFileArg()) {
        std::cout << "Using YAML configuration file(s):" << std::endl;
    } else {
        std::cout << "Using default YAML configuration file: "
                  << yaml_files.front() << std::endl;
    }
    for (const auto& f : yaml_files) {
        std::cout << "  - " << f << std::endl;
    }

    // Load batch GEMM configurations from every file. When multiple files are
    // provided a per-file prefix (the sanitized file stem) is added to each
    // benchmark name so results remain distinguishable across files.
    std::vector<BatchGemmBenchConfig> configs;
    const bool                        multiple_files = yaml_files.size() > 1;
    for (size_t fi = 0; fi < yaml_files.size(); ++fi) {
        const std::string& yaml_file = yaml_files[fi];
        // Paths returned by getYamlFiles() are already validated to exist, so
        // no further existence check is required here.
        auto file_configs = loadBatchGemmBenchmarkConfigs(yaml_file);
        if (multiple_files) {
            std::string stem = std::filesystem::path(yaml_file).stem().string();
            for (char& c : stem) {
                if (!std::isalnum(static_cast<unsigned char>(c))) {
                    c = '_';
                }
            }
            // Combine a file index with the sanitized stem ("f<idx>_<stem>/")
            // so names are guaranteed unique even when two files share the
            // same filename in different directories.
            const std::string prefix =
                "f" + std::to_string(fi) + "_" + stem + "/";
            for (auto& cfg : file_configs) {
                cfg.name = prefix + cfg.name;
            }
        }
        // Reserve to avoid repeated reallocations when loading multiple files.
        configs.reserve(configs.size() + file_configs.size());
        configs.insert(configs.end(),
                       std::make_move_iterator(file_configs.begin()),
                       std::make_move_iterator(file_configs.end()));
    }

    if (configs.empty()) {
        std::cerr << "No batch GEMM configurations loaded!" << std::endl;
        return 1;
    }

    std::cout << "=== AOCL-DLP Batch GEMM Benchmark ===" << std::endl;
    std::cout << "Loaded " << configs.size() << " configurations from "
              << yaml_files.size() << " file(s)" << std::endl;

#ifdef DLP_ENABLE_OPENMP
    std::cout << "OpenMP: Enabled" << std::endl;
    std::cout << "  OMP_NUM_THREADS = " << omp_get_max_threads() << std::endl;
    char* omp_proc_bind = std::getenv("OMP_PROC_BIND");
    if (omp_proc_bind) {
        std::cout << "  OMP_PROC_BIND   = " << omp_proc_bind << std::endl;
    } else {
        std::cout << "  OMP_PROC_BIND   = (not set)" << std::endl;
    }
    char* omp_places = std::getenv("OMP_PLACES");
    if (omp_places) {
        std::cout << "  OMP_PLACES      = " << omp_places << std::endl;
    } else {
        std::cout << "  OMP_PLACES      = (not set)" << std::endl;
    }
#else
    std::cout << "OpenMP: Disabled" << std::endl;
#endif

    std::cout << "\n=== OPTIMIZATION ENABLED ===" << std::endl;
    std::cout << "Metadata preparation: ONCE in constructor" << std::endl;
    std::cout << "Hot loop: Pure kernel execution (no overhead)" << std::endl;

    // Get iteration count from command line (-n flag)
    int64_t iterations = parser.getIterations();

    // Register all benchmarks
    // Resolve MinTime: --benchmark_min_time CLI > BENCH_MIN_TIME env > 3.0
    double bench_min_time = parser.getBenchMinTime();

    // One-time cold-cache init (no-op unless --cold was passed).
    dlp::bench::initColdCache(parser.getColdCache(), parser.getColdPasses());

    registerOptimizedBenchmarks(configs, iterations, bench_min_time);

    // Initialize and run Google Benchmark
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}
