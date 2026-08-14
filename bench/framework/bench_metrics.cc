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

#include "bench_metrics.hh"
#include <fstream>
#include <sstream>

namespace dlp::benchmarking {

// FIXME(#544): Duplicates dlp::testing::framework::getElementSizeBytes() in
// tests/include/framework/types.hh, which returns 0 for u4/s4 and 0 by
// default where this returns 1 and 4. Packed accounting is now handled by
// getMatrixSizeBytes(); the duplicate helper still needs to be reconciled.
size_t
BenchmarkMetrics::getMatrixTypeSize(MatrixType type)
{
    switch (type) {
        case MatrixType::u4:
        case MatrixType::s4:
            return 1; // Memory is byte-addressable; packed arrays are special.
        case MatrixType::u8:
        case MatrixType::s8:
            return 1;
        case MatrixType::u16:
        case MatrixType::s16:
        case MatrixType::bf16:
        case MatrixType::fp16:
            return 2;
        case MatrixType::u32:
        case MatrixType::s32:
        case MatrixType::f32:
            return 4;
        default:
            return 4; // Default to 4 bytes
    }
}

size_t
BenchmarkMetrics::getMatrixSizeBytes(MatrixType type, md_t rows, md_t cols)
{
    if (rows <= 0 || cols <= 0) {
        return 0;
    }

    const size_t elements = static_cast<size_t>(rows) * cols;
    if (type == MatrixType::u4 || type == MatrixType::s4) {
        // Two 4-bit elements share one byte; round up for an odd element
        // count rather than reporting a fractional physical byte.
        return (elements / 2) + (elements % 2);
    }

    return elements * getMatrixTypeSize(type);
}

void
BenchmarkMetrics::calculateAndReport(benchmark::State& state,
                                     md_t              m,
                                     md_t              n,
                                     md_t              k,
                                     MatrixType        a_type,
                                     MatrixType        b_type,
                                     MatrixType        c_type,
                                     md_t              group_size)
{
    // Calculate GFLOPS
    double ops = 2.0 * static_cast<double>(m) * n * k;
    state.counters["GFLOPS"] =
        benchmark::Counter(ops, benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::kIs1000);

    // For bandwidth calculation, we need bytes and will let Google Benchmark
    // calculate the rate
    const size_t bytes_A     = getMatrixSizeBytes(a_type, m, k);
    const size_t bytes_B     = getMatrixSizeBytes(b_type, k, n);
    const size_t bytes_C     = getMatrixSizeBytes(c_type, m, n);
    const double total_bytes = static_cast<double>(bytes_A)
                               + static_cast<double>(bytes_B)
                               + static_cast<double>(bytes_C);

    // Bandwidth in GB/s (Google Benchmark will calculate the rate)
    state.counters["Bandwidth_GB/s"] = benchmark::Counter(
        total_bytes, benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000);

    // Note: Efficiency calculation would require knowing actual GFLOPS achieved
    // which we can't calculate here without timing information
    // We'll leave it out for now or calculate in post-processing

    // Configuration metadata (useful for filtering and analysis)
    state.counters["M"] = static_cast<double>(m);
    state.counters["N"] = static_cast<double>(n);
    state.counters["K"] = static_cast<double>(k);

    // Quantization group size along K (0 = full K)
    state.counters["group_size"] = static_cast<double>(group_size);

    // Matrix size in MB
    double matrix_size_mb            = total_bytes / (1024.0 * 1024.0);
    state.counters["Matrix_Size_MB"] = matrix_size_mb;
}

} // namespace dlp::benchmarking
