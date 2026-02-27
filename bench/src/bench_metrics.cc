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

size_t
BenchmarkMetrics::getMatrixTypeSize(MatrixType type)
{
    switch (type) {
        case MatrixType::u4:
        case MatrixType::s4:
            return 1; // 4-bit types packed, but count as 1 byte min
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

void
BenchmarkMetrics::calculateAndReport(benchmark::State& state,
                                     md_t              m,
                                     md_t              n,
                                     md_t              k,
                                     MatrixType        a_type,
                                     MatrixType        b_type,
                                     MatrixType        c_type)
{
    // Calculate GFLOPS
    double ops = 2.0 * static_cast<double>(m) * n * k;
    state.counters["GFLOPS"] =
        benchmark::Counter(ops, benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::kIs1000);

    // For bandwidth calculation, we need bytes and will let Google Benchmark
    // calculate the rate
    size_t size_a = getMatrixTypeSize(a_type);
    size_t size_b = getMatrixTypeSize(b_type);
    size_t size_c = getMatrixTypeSize(c_type);

    double bytes_A     = static_cast<double>(m) * k * size_a;
    double bytes_B     = static_cast<double>(k) * n * size_b;
    double bytes_C     = static_cast<double>(m) * n * size_c;
    double total_bytes = bytes_A + bytes_B + bytes_C;

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

    // Matrix size in MB
    double matrix_size_mb =
        (size_a * m * k + size_b * k * n + size_c * m * n) / (1024.0 * 1024.0);
    state.counters["Matrix_Size_MB"] = matrix_size_mb;
}

} // namespace dlp::benchmarking
