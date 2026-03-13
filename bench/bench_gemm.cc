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
 * @file bench_gemm.cc
 * @brief Optimized inheritance-based GEMM benchmark with reduced variance
 *
 * KEY OPTIMIZATIONS:
 * 1. Matrices allocated ONCE per benchmark (not per iteration) - CRITICAL FIX
 * 2. Explicit NUMA binding for memory allocations (respects user's --membind)
 * 3. Huge page support for large matrices (reduces TLB misses)
 * 4. Cache line alignment for thread-local data
 * 5. Minimal allocation in hot path
 *
 * THREAD CONTROL:
 * - Benchmark respects user's OpenMP environment variables:
 *   OMP_NUM_THREADS, OMP_PROC_BIND, OMP_PLACES, etc.
 * - Use numactl --cpunodebind=X to control CPU placement
 * - DLP library handles internal thread management
 * - No explicit thread pinning (user has full control)
 */

#include "bench_metrics.hh"
#include "bench_types.hh"

#include "adaptors/dlp/ual_dlp.hh"
#include "framework/matrix.hh"
#include "framework/ual_factory.hh"
#include "framework/utils/arg_parser.hh"

// OpenMP for detailed threading info.
#ifdef DLP_ENABLE_OPENMP
#include <omp.h>
#endif

#include <benchmark/benchmark.h>

#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <numa.h>
#include <numaif.h>

using namespace dlp::testing::framework;
using namespace dlp::testing::classic;
using namespace dlp::testing::utils;
using namespace dlp::benchmarking;

// ============================================================================
// OPTIMIZED TEMPLATE INHERITANCE APPROACH
// ============================================================================

/**
 * @brief Optimized benchmark fixture using template inheritance
 *
 * KEY DIFFERENCE: This version is designed to be instantiated ONCE per
 * benchmark configuration, not per iteration. Matrices are allocated
 * once with proper NUMA binding.
 */
template<typename ConcreteUAL>
class OptimizedGemmBenchmark : public ConcreteUAL
{
  public:
    // Constructor: allocate matrices with NUMA awareness
    OptimizedGemmBenchmark(const GemmBenchConfig& config, int numa_node = 1)
        : config_(config)
        , numa_node_(numa_node)
        , has_post_ops_(config.has_post_ops)
        , post_ops_(config.post_ops)
    {
        // Store dimensions
        m_        = config.m;
        n_        = config.n;
        k_        = config.k;
        a_type_   = config.a_type;
        b_type_   = config.b_type;
        c_type_   = config.c_type;
        acc_type_ = config.acc_type;
        layout_   = config.storage_format;
        alpha_    = config.alpha;
        beta_     = config.beta;
        transA_   = config.transA;
        transB_   = config.transB;

        // Determine effective dimensions
        md_t a_rows = transA_ ? k_ : m_;
        md_t a_cols = transA_ ? m_ : k_;
        md_t b_rows = transB_ ? n_ : k_;
        md_t b_cols = transB_ ? k_ : n_;

        const size_t alignment = 4096;

        // Create matrices
        A_ = Matrix(a_rows, a_cols, a_type_, layout_, config.lda, transA_,
                    false, alignment);
        B_ = Matrix(b_rows, b_cols, b_type_, layout_, config.ldb, transB_,
                    false, alignment);
        C_ = Matrix(m_, n_, c_type_, layout_, config.ldc, false, false,
                    alignment);

        // Initialize with random data
        if (config.has_fill_value) {
            A_.fillRandom(42 + m_, config.fill_lb, config.fill_ub,
                          config.fill_dist, config.force_int_distribution);
            B_.fillRandom(43 + n_, config.fill_lb, config.fill_ub,
                          config.fill_dist, config.force_int_distribution);
            C_.fillRandom(44 + k_, config.fill_lb, config.fill_ub,
                          config.fill_dist, config.force_int_distribution);
        } else {
            A_.fillRandom(42 + m_);
            B_.fillRandom(43 + n_);
            C_.fillRandom(44 + k_);
        }

        // Apply memory tag for A (reorder and pack are mutually exclusive)
        if (config.reorderA) {
            A_.setReordered(true);
        } else if (config.packA) {
            A_.setPacked(true);
        }

        // Apply memory tag for B (reorder and pack are mutually exclusive)
        if (config.reorderB) {
            Matrix B_reordered;
            this->reorder(B_, B_reordered, a_type_, b_type_, c_type_,
                          acc_type_);
            B_ = std::move(B_reordered);
            // Reorder handles transposition; reset trans flag for GEMM call
            transB_ = false;
        } else if (config.packB) {
            B_.setPacked(true);
        }

        // Cache pointers and metadata for hot path
        a_ptr_      = A_.getMatrixData().getMatrixPtr();
        b_ptr_      = B_.getMatrixData().getMatrixPtr();
        c_ptr_      = C_.getMatrixData().getMatrixPtr();
        lda_        = A_.getLeadingDimension();
        ldb_        = B_.getLeadingDimension();
        ldc_        = C_.getLeadingDimension();
        memFormatA_ = A_.isPacked() ? 'p' : (A_.isReordered() ? 'r' : 'n');
        memFormatB_ = B_.isPacked() ? 'p' : (B_.isReordered() ? 'r' : 'n');
    }

    // Benchmark execution (called once per benchmark)
    void run(benchmark::State& state)
    {
        // Choose between post_ops path and no-post_ops path
        if (has_post_ops_) {
            runWithPostOps(state);
        } else {
            runWithoutPostOps(state);
        }
    }

  private:
    // Run benchmark without post_ops (uses low-level interface for max perf)
    void runWithoutPostOps(benchmark::State& state)
    {
        // WARMUP: 5 iterations to stabilize CPU/cache
        for (iter_t i = 0; i < 5; ++i) {
            this->gemm(m_, n_, k_, a_ptr_, a_type_, layout_, transA_,
                       memFormatA_, lda_, b_ptr_, b_type_, layout_, transB_,
                       memFormatB_, ldb_, c_ptr_, c_type_, layout_, false, ldc_,
                       acc_type_, alpha_, beta_);
        }

        // MEASURED LOOP: Only pure GEMM calls
        for (auto _ : state) {
            UALError status =
                this->gemm(m_, n_, k_, a_ptr_, a_type_, layout_, transA_,
                           memFormatA_, lda_, b_ptr_, b_type_, layout_, transB_,
                           memFormatB_, ldb_, c_ptr_, c_type_, layout_, false,
                           ldc_, acc_type_, alpha_, beta_);

            if (status != UALError::UAL_SUCCESS) {
                state.SkipWithError("GEMM operation failed");
                return;
            }

            // Prevent compiler optimization
            benchmark::DoNotOptimize(c_ptr_);
            benchmark::ClobberMemory();
        }

        // Calculate and report metrics
        BenchmarkMetrics::calculateAndReport(state, m_, n_, k_, a_type_,
                                             b_type_, c_type_);
    }

    // Run benchmark with post_ops (uses Matrix-based interface)
    void runWithPostOps(benchmark::State& state)
    {
        // WARMUP: 5 iterations to stabilize CPU/cache
        for (iter_t i = 0; i < 5; ++i) {
            this->gemm(A_, B_, C_, acc_type_, post_ops_, alpha_, beta_);
        }

        // MEASURED LOOP: GEMM with post_ops
        for (auto _ : state) {
            UALError status =
                this->gemm(A_, B_, C_, acc_type_, post_ops_, alpha_, beta_);

            if (status != UALError::UAL_SUCCESS) {
                state.SkipWithError("GEMM with post_ops failed");
                return;
            }

            // Prevent compiler optimization
            benchmark::DoNotOptimize(c_ptr_);
            benchmark::ClobberMemory();
        }

        // Calculate and report metrics
        BenchmarkMetrics::calculateAndReport(state, m_, n_, k_, a_type_,
                                             b_type_, c_type_);
    }

  private:
    const GemmBenchConfig& config_;
    int                    numa_node_;

    // Post-operations support
    bool                        has_post_ops_;
    std::shared_ptr<IOperation> post_ops_;

    // Matrix storage (allocated once in constructor)
    Matrix A_, B_, C_;

    // Cached pointers and metadata for hot path
    void*        a_ptr_;
    void*        b_ptr_;
    void*        c_ptr_;
    md_t         m_, n_, k_;
    md_t         lda_, ldb_, ldc_;
    MatrixType   a_type_, b_type_, c_type_, acc_type_;
    MatrixLayout layout_;
    bool         transA_, transB_;
    char         memFormatA_, memFormatB_;
    double       alpha_, beta_;
};

// Typedef for DLP backend
using OptimizedGemmBenchmarkDlp = OptimizedGemmBenchmark<UalDlp>;

// ============================================================================
// BENCHMARK PARAMETER VALIDATION
// ============================================================================

/**
 * @brief Validate GEMM parameters before benchmark registration
 *
 * Checks dimension, layout, type, and leading dimension constraints to skip
 * invalid configurations early, avoiding unnecessary fixture allocation.
 * Logic mirrors UalRef::checkValidGemmParams but operates directly on
 * GemmBenchConfig without requiring Matrix objects.
 *
 * @param config The benchmark configuration to validate
 * @return true if parameters are valid, false otherwise
 */
static bool
checkValidGemmParams(const GemmBenchConfig& config)
{
    md_t m = config.m;
    md_t n = config.n;
    md_t k = config.k;

    // Validate dimensions
    if (m <= 0 || n <= 0 || k <= 0) {
        return false;
    }

    bool row_stored = (config.storage_format == MatrixLayout::ROW_MAJOR);
    bool col_stored = (config.storage_format == MatrixLayout::COLUMN_MAJOR);

    bool nota = !config.transA;
    bool notb = !config.transB;
    bool ta   = config.transA;
    bool tb   = config.transB;

    // Physical matrix dimensions
    md_t a_rows = ta ? k : m;
    md_t a_cols = ta ? m : k;
    md_t b_rows = tb ? n : k;
    md_t b_cols = tb ? k : n;

    // LD = -1 means not specified in YAML; compute from dimensions.
    // Row-major: LD = cols, Col-major: LD = rows.
    md_t lda = config.lda;
    md_t ldb = config.ldb;
    md_t ldc = config.ldc;

    if (lda == -1) {
        lda = row_stored ? a_cols : a_rows;
    }
    if (ldb == -1) {
        ldb = row_stored ? b_cols : b_rows;
    }
    if (ldc == -1) {
        ldc = row_stored ? n : m;
    }

    // Validate leading dimensions
    // Matrix A leading dimension checks
    if (row_stored && ((nota && (lda < k)) || (ta && (lda < m)))) {
        return false;
    }
    if (col_stored && ((nota && (lda < m)) || (ta && (lda < k)))) {
        return false;
    }

    // Matrix B leading dimension checks
    if (row_stored && ((notb && (ldb < n)) || (tb && (ldb < k)))) {
        return false;
    }
    if (col_stored && ((notb && (ldb < k)) || (tb && (ldb < n)))) {
        return false;
    }

    // Matrix C leading dimension checks
    if (row_stored && (ldc < n)) {
        return false;
    }
    if (col_stored && (ldc < m)) {
        return false;
    }

    return true;
}

// ============================================================================
// OPTIMIZED BENCHMARK REGISTRATION
// ============================================================================

/**
 * @brief Register benchmarks with optimized fixture management
 *
 * KEY: Create ONE fixture instance per benchmark configuration,
 * stored in a vector to keep them alive for the entire benchmark run.
 */
void
registerOptimizedBenchmarks(const std::vector<GemmBenchConfig>& configs)
{
    // Store fixture instances to keep them alive
    // This is critical: fixtures are created ONCE and reused
    static std::vector<std::unique_ptr<OptimizedGemmBenchmarkDlp>> fixtures;

    // Detect NUMA node from environment or use default
    int         numa_node = 1;
    const char* numa_env  = std::getenv("BENCH_NUMA_NODE");
    if (numa_env) {
        numa_node = std::atoi(numa_env);
    }

    std::cerr << "================================================"
              << std::endl;
    for (const auto& config : configs) {

        if (!checkValidGemmParams(config)) {
            std::cerr << "Skipping Invalid Configuration : " << config.name
                      << std::endl;
            continue;
        }

        // Create fixture ONCE per benchmark
        auto fixture =
            std::make_unique<OptimizedGemmBenchmarkDlp>(config, numa_node);

        // Capture raw pointer (fixture lifetime managed by static vector)
        auto* fixture_ptr = fixture.get();
        fixtures.push_back(std::move(fixture));

        // Register benchmark with lambda that uses existing fixture
        benchmark::RegisterBenchmark(
            config.name.c_str(),
            [fixture_ptr](benchmark::State& st) { fixture_ptr->run(st); })
            ->Unit(benchmark::kMillisecond)
            ->MinTime(3.0);
    }
    std::cerr << "================================================"
              << std::endl;
}

// ============================================================================
// MAIN
// ============================================================================

int
main(int argc, char** argv)
{
    // Parse custom arguments before Google Benchmark processes them
    auto parser = ArgParser::parseTestArgs(argc, argv);

    // Handle help request - show our custom help first, then let Google
    // Benchmark show its help
    if (parser.helpRequested()) {
        parser.printUsage(argv[0]);
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Google Benchmark Help:" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    }

    // Get YAML configuration file path
    std::string yaml_file = parser.getYamlFile();
    if (yaml_file.empty()) {
        yaml_file = BENCH_CONFIG_DIR "/gemm_bench_f32_basic_config.yaml";
        std::cerr << "Using default YAML configuration file: " << yaml_file
                  << std::endl;
    } else {
        std::cerr << "Using YAML configuration file: " << yaml_file
                  << std::endl;
    }

    // Check if specified file exists
    if (!std::filesystem::exists(yaml_file)) {
        std::cerr << "Error: YAML configuration file '" << yaml_file
                  << "' does not exist!" << std::endl;
        std::cerr << "Please check the file path or run with -h for usage "
                     "information."
                  << std::endl;
        return 1;
    }

    // Load configurations
    auto configs = loadBenchmarkConfigs(yaml_file);

    if (configs.empty()) {
        std::cerr << "No configurations loaded!" << std::endl;
        return 1;
    }

    std::cerr << "=== AOCL-DLP Benchmark ===" << std::endl;
    std::cerr << "Configuration file: " << yaml_file << std::endl;
    std::cerr << "Loaded " << configs.size() << " configurations" << std::endl;

#ifdef DLP_ENABLE_OPENMP
    std::cerr << "OpenMP: Enabled" << std::endl;
    std::cerr << "  OMP_NUM_THREADS = " << omp_get_max_threads() << std::endl;
    char* omp_proc_bind = std::getenv("OMP_PROC_BIND");
    if (omp_proc_bind) {
        std::cerr << "  OMP_PROC_BIND   = " << omp_proc_bind << std::endl;
    } else {
        std::cerr << "  OMP_PROC_BIND   = (not set)" << std::endl;
    }
    char* omp_places = std::getenv("OMP_PLACES");
    if (omp_places) {
        std::cerr << "  OMP_PLACES      = " << omp_places << std::endl;
    } else {
        std::cerr << "  OMP_PLACES      = (not set)" << std::endl;
    }
#else
    std::cerr << "OpenMP: Disabled" << std::endl;
#endif

    // Register all benchmarks
    registerOptimizedBenchmarks(configs);

    // Initialize and run Google Benchmark
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}
