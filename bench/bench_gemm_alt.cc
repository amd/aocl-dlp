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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
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

#include "framework/matrix.hh"
#include "framework/ual.hh"
#include "framework/ual_factory.hh"

#include <benchmark/benchmark.h>
#include <iostream>
#include <memory>

using namespace dlp::testing::framework;

// ============================================================================
// UAL GEMM FP32 BENCHMARK
// ============================================================================

/**
 * @brief Benchmark FP32 GEMM using DLP UAL implementation
 *
 * This benchmark measures the performance of matrix multiplication C = A*B
 * using the DLP (Deep Learning Primitives) UAL backend. No post-operations
 * are applied, focusing purely on the GEMM kernel performance.
 *
 * @param state Google Benchmark state object
 */
static void
BM_GEMM_DLP_FP32(benchmark::State& state)
{
    // Extract matrix dimensions from benchmark parameters
    const md_t m = state.range(0);
    const md_t n = state.range(1);
    const md_t k = state.range(2);

    // Create matrices with FP32 precision and row-major layout
    Matrix A(m, k, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix B(k, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);
    Matrix C(m, n, MatrixType::f32, MatrixLayout::ROW_MAJOR, 0, false);

    // Initialize matrices with random data for realistic benchmarking
    A.fillRandom(42); // Fixed seed for reproducible results
    B.fillRandom(43);
    C.fillRandom(44);

    // Create DLP UAL instance for high-performance computation
    std::unique_ptr<IUal> ual = UalFactory::createUal(UALType::DLP);
    if (!ual) {
        state.SkipWithError("Failed to create DLP UAL instance");
        return;
    }

    // Benchmark loop - measures the time for repeated GEMM operations
    for (auto _ : state) {
        // Reset C matrix to ensure consistent starting conditions
        C.fillRandom(44);

        // Perform the GEMM operation: C = A * B
        bool result = ual->gemm(A, B, C, MatrixType::f32);

        if (!result) {
            state.SkipWithError("GEMM operation failed");
            return;
        }

        // Prevent compiler optimization from eliminating the computation
        benchmark::DoNotOptimize(C);
    }

    // Calculate and report performance metrics
    const double total_ops = 2.0 * m * n * k; // FLOPs for matrix multiplication
    state.counters["GFLOPS"] = benchmark::Counter(
        total_ops, benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);

    // Report matrix dimensions in the output
    state.counters["M"] = m;
    state.counters["N"] = n;
    state.counters["K"] = k;

    // Calculate memory bandwidth (assuming all data is transferred)
    const double total_bytes = (m * k + k * n + m * n) * sizeof(float);
    state.counters["GB/s"]   = benchmark::Counter(
        total_bytes, benchmark::Counter::kIsRate, benchmark::Counter::kIs1024);
}

// ============================================================================
// BENCHMARK REGISTRATION
// ============================================================================

// Register benchmarks for common matrix sizes
// Format: (M, N, K) dimensions

// Small matrices (good for debugging and quick tests)
BENCHMARK(BM_GEMM_DLP_FP32)
    ->Args({ 64, 64, 64 })
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GEMM_DLP_FP32)
    ->Args({ 128, 128, 128 })
    ->Unit(benchmark::kMicrosecond);

// Medium matrices (typical for many ML workloads)
BENCHMARK(BM_GEMM_DLP_FP32)
    ->Args({ 256, 256, 256 })
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GEMM_DLP_FP32)
    ->Args({ 512, 512, 512 })
    ->Unit(benchmark::kMicrosecond);

// Large matrices (memory-bound scenarios)
BENCHMARK(BM_GEMM_DLP_FP32)
    ->Args({ 1024, 1024, 1024 })
    ->Unit(benchmark::kMillisecond);

// Rectangular matrices (common in neural networks)
BENCHMARK(BM_GEMM_DLP_FP32)
    ->Args({ 1024, 512, 256 })
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GEMM_DLP_FP32)
    ->Args({ 512, 1024, 256 })
    ->Unit(benchmark::kMicrosecond);

// Batch sizes common in inference
BENCHMARK(BM_GEMM_DLP_FP32)
    ->Args({ 1, 1024, 1024 })
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GEMM_DLP_FP32)
    ->Args({ 32, 1024, 1024 })
    ->Unit(benchmark::kMicrosecond);

int
main(int argc, char** argv)
{
    // Print information about the benchmark
    std::cout << "=== UAL GEMM FP32 Benchmark ===" << std::endl;
    std::cout << "Benchmarking DLP UAL implementation for FP32 GEMM"
              << std::endl;
    std::cout << "Format: C = A * B (no post-operations)" << std::endl;
    std::cout << "Metrics: GFLOPS, Memory Bandwidth (GB/s)" << std::endl;
    std::cout << "=======================================" << std::endl;

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
