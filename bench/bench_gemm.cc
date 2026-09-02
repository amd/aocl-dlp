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

#include "aocl_dlp_config.h"
#include "bench_metrics.hh"
#include "bench_types.hh"
#include "cold_cache.hh"

#include "adaptors/dlp/ual_dlp.hh"
#include "framework/matrix.hh"
#include "framework/ual_factory.hh"
#include "framework/ual_plan.hh"
#include "framework/utils/arg_parser.hh"

// OpenMP for detailed threading info.
#ifdef DLP_ENABLE_OPENMP
#include <omp.h>
#endif

#include <benchmark/benchmark.h>

#include <cctype>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#if !DLP_OS_WINDOWS
#include <numa.h>
#include <numaif.h>
#endif

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

        // Feed tuning knobs (blocking params / SUP thresholds / GEMM hints) to
        // the DLP UAL
        // BEFORE reorder so the buffer-size query and packing use the same
        // block sizes as the GEMM (the "set before reorder" use case). Only the
        // DLP backend honors these; absent knobs leave library defaults.
        if (config.has_blocking || config.has_sup_thresholds
            || config.has_gemm_hints) {
            this->setTuningKnobs(
                config.blk_MR, config.blk_NR, config.blk_MC, config.blk_NC,
                config.blk_KC, config.sup_MT, config.sup_NT, config.sup_KT,
                config.has_blocking, config.has_sup_thresholds, config.m_hint,
                config.nt_hint, config.has_gemm_hints);
        } else {
            this->clearTuningKnobs();
        }

        // Apply memory tag for A (reorder and pack are mutually exclusive)
        if (config.reorderA) {
            A_.setReordered(true);
        } else if (config.packA) {
            A_.setPacked(true);
        }

        // Apply memory tag for B (reorder and pack are mutually exclusive)
        if (config.reorderB) {
            Matrix   B_reordered;
            UALError rstat =
                this->reorder(B_, B_reordered, a_type_, b_type_, c_type_,
                              acc_type_, config.group_scale_param.get());
            if (rstat != UALError::UAL_SUCCESS) {
                // Kernel REJECTED this configuration during reorder (e.g. an
                // unsupported tuning-knob combination). Mark the fixture as
                // not-runnable; registration will SKIP it (mirrors the GTest
                // harness) instead of reporting a benchmark ERROR later.
                m_probe_status = rstat;
                return;
            }
            B_ = std::move(B_reordered);
            // Reorder handles transposition; reset trans flag for GEMM call
            transB_ = false;
        } else if (config.packB) {
            B_.setPacked(true);
        }

        // Cache C pointer for DoNotOptimize in hot path
        c_ptr_ = C_.getMatrixData().getMatrixPtr();

        // Create and configure the execution plan
        plan_ = this->createPlan();
        plan_->configureFrom(A_, B_, C_, acc_type_, alpha_, beta_);

        // Match MicroTest::configurePlan(): fusion ops, then quant setters.
        if (config.has_post_ops && config.post_op_params) {
            for (const auto& p : *config.post_op_params) {
                plan_->addPostOp(p->clone());
            }
        }
        if (config.a_quant_param) {
            plan_->setAQuant(
                std::make_unique<AQuantParam>(*config.a_quant_param));
        }
        if (config.woq_param) {
            plan_->setWOQ(std::make_unique<WOQParam>(*config.woq_param));
        }
        if (config.group_scale_param) {
            plan_->setGroupScale(
                std::make_unique<GroupScaleParam>(*config.group_scale_param));
        }
        // Feed the SAME tuning knobs to the GEMM path so the kernel uses the
        // same block sizes as the reorder above. Only the DLP plan honors
        // these; absent knobs leave the library defaults.
        if (config.has_blocking) {
            plan_->setBlocking(config.blk_MR, config.blk_NR, config.blk_MC,
                               config.blk_NC, config.blk_KC);
        }
        if (config.has_sup_thresholds) {
            plan_->setSupThresholds(config.sup_MT, config.sup_NT,
                                    config.sup_KT);
        }
        if (config.has_gemm_hints) {
            plan_->setGemmHints(config.m_hint, config.nt_hint);
        }

        // Pre-build all backend state
        plan_->prepare();

        // Trial-execute ONCE at setup so rejections are SKIPPED at
        // registration instead of a benchmark ERROR in the hot loop.
        // Covers missing classic APIs (UAL_NO_MATCHING_API), runtime/ISA
        // rejects (UAL_NOT_SUPPORTED), and invalid tuning knobs.
        plan_->setBuffers(A_, B_, C_);
        m_probe_status = plan_->execute();
    }

    // Whether the fixture is runnable (kernel accepted the config at setup).
    // UAL_SUCCESS means it can be benchmarked; anything else => skip.
    UALError probeStatus() const { return m_probe_status; }

    // Benchmark execution (called once per benchmark)
    void run(benchmark::State& state)
    {
        // Bind buffers once before the hot loop
        plan_->setBuffers(A_, B_, C_);

        const bool cold = dlp::bench::coldCacheEnabled();

        // WARMUP: 5 iterations to stabilize CPU/cache (skip when measuring
        // cold-cache behaviour — warming defeats the point).
        if (!cold) {
            for (iter_t i = 0; i < 5; ++i) {
                plan_->execute();
            }
        }

        // MEASURED LOOP: time only plan_->execute() via std::chrono and
        // submit to gbench through SetIterationTime() (UseManualTime).
        // This excludes the optional cold-cache flush from the reported time.
        for (auto _ : state) {
            if (cold)
                dlp::bench::flushColdCache();

            const auto t0     = std::chrono::steady_clock::now();
            UALError   status = plan_->execute();
            const auto t1     = std::chrono::steady_clock::now();

            if (status != UALError::UAL_SUCCESS) {
                if (const char* reason = ualRejectionMessage(status)) {
                    state.SkipWithError(reason);
                } else {
                    state.SkipWithError("GEMM operation failed");
                }
                return;
            }

            const double iter_seconds =
                std::chrono::duration<double>(t1 - t0).count();
            state.SetIterationTime(iter_seconds);

            // Prevent compiler optimization
            benchmark::DoNotOptimize(c_ptr_);
            benchmark::ClobberMemory();
        }

        md_t group_size = config_.group_scale_param
                              ? config_.group_scale_param->getGroupSize()
                              : 0;
        BenchmarkMetrics::calculateAndReport(state, m_, n_, k_, a_type_,
                                             b_type_, c_type_, group_size);
    }

  private:
    const GemmBenchConfig& config_;
    int                    numa_node_;

    // Execution plan (replaces old gemm() call paths)
    std::unique_ptr<IUalPlan> plan_;

    // Matrix storage (allocated once in constructor)
    Matrix A_, B_, C_;

    // Cached pointer for DoNotOptimize
    void* c_ptr_;

    // Setup-time kernel status: UAL_SUCCESS if the config was accepted (and the
    // trial execute succeeded), otherwise the rejection code. Used to skip
    // registration of configs the kernel does not support (e.g. unsupported
    // tuning knobs) so they are reported as SKIPPED, not ERROR.
    UALError m_probe_status = UALError::UAL_SUCCESS;

    // Cached metadata for metrics reporting
    md_t         m_, n_, k_;
    MatrixType   a_type_, b_type_, c_type_, acc_type_;
    MatrixLayout layout_;
    bool         transA_, transB_;
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
 *
 * @param configs Vector of benchmark configurations to register
 * @param iterations Number of iterations to run (-1 for default MinTime
 * behavior)
 */
void
registerOptimizedBenchmarks(const std::vector<GemmBenchConfig>& configs,
                            int64_t                             iterations = -1,
                            double bench_min_time = 3.0)
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

        if (!checkValidGemmParams(config)) {
            std::cerr << "Skipping Invalid Configuration : " << config.name
                      << std::endl;
            continue;
        }

        // Create fixture ONCE per benchmark
        auto fixture =
            std::make_unique<OptimizedGemmBenchmarkDlp>(config, numa_node);

        // If probing this config failed, SKIP it instead of registering — so
        // it is reported as skipped, not as a benchmark ERROR (mirrors
        // GTest).
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
            continue; // fixture destructed here; not registered
        }

        // Capture raw pointer (fixture lifetime managed by static vector)
        auto* fixture_ptr = fixture.get();
        fixtures.push_back(std::move(fixture));

        // Register benchmark with lambda that uses existing fixture
        // Use Iterations() if specified, otherwise use MinTime()
        if (iterations > 0) {
            benchmark::RegisterBenchmark(
                config.name.c_str(),
                [fixture_ptr](benchmark::State& st) { fixture_ptr->run(st); })
                ->Unit(benchmark::kMillisecond)
                ->UseManualTime()
                ->Iterations(
                    static_cast<benchmark::IterationCount>(iterations));
        } else {
            benchmark::RegisterBenchmark(
                config.name.c_str(),
                [fixture_ptr](benchmark::State& st) { fixture_ptr->run(st); })
                ->Unit(benchmark::kMillisecond)
                ->UseManualTime()
                ->MinTime(bench_min_time);
        }
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

    // Get YAML configuration file path(s). Multiple files may be provided via
    // repeated -f/--file flags and/or comma-separated values. The built-in
    // default is threaded through getYamlFiles() so it is only used when no
    // -f/--file flag was given.
    std::vector<std::string> yaml_files = parser.getYamlFiles(
        BENCH_CONFIG_DIR "/gemm_bench_f32_basic_config.yaml");

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
        std::cerr << "Using YAML configuration file(s):" << std::endl;
    } else {
        std::cerr << "Using default YAML configuration file: "
                  << yaml_files.front() << std::endl;
    }
    for (const auto& f : yaml_files) {
        std::cerr << "  - " << f << std::endl;
    }

    // Load configurations from every file. When multiple files are provided a
    // per-file prefix is added to each benchmark name so results remain
    // distinguishable. The prefix combines a file index with the sanitized
    // file stem ("f<idx>_<stem>/") so names are guaranteed unique even when
    // two files share the same filename in different directories.
    std::vector<GemmBenchConfig> configs;
    const bool                   multiple_files = yaml_files.size() > 1;
    for (size_t fi = 0; fi < yaml_files.size(); ++fi) {
        const std::string& yaml_file = yaml_files[fi];
        // Paths returned by getYamlFiles() are already validated to exist, so
        // no further existence check is required here.
        auto file_configs = loadBenchmarkConfigs(yaml_file);
        if (multiple_files) {
            std::string stem = std::filesystem::path(yaml_file).stem().string();
            for (char& c : stem) {
                if (!std::isalnum(static_cast<unsigned char>(c))) {
                    c = '_';
                }
            }
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
        std::cerr << "No configurations loaded!" << std::endl;
        return 1;
    }

    std::cerr << "=== AOCL-DLP Benchmark ===" << std::endl;
    std::cerr << "Loaded " << configs.size() << " configurations from "
              << yaml_files.size() << " file(s)" << std::endl;

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

    // Get iteration count from command line (-n flag)
    int64_t iterations = parser.getIterations();

    // Resolve MinTime: --benchmark_min_time CLI > BENCH_MIN_TIME env > 3.0
    double bench_min_time = parser.getBenchMinTime();

    // One-time cold-cache init (no-op unless --cold was passed).
    // Must come AFTER OMP setup so omp_get_max_threads() returns the right
    // value for the per-thread sink array.
    dlp::bench::initColdCache(parser.getColdCache(), parser.getColdPasses());

    // Register all benchmarks
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
