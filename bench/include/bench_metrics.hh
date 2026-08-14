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

#include "framework/matrix.hh"
#include <benchmark/benchmark.h>
#include <string>

namespace dlp::benchmarking {

using namespace dlp::testing::framework;

/**
 * @brief Utility class for calculating and reporting benchmark metrics
 */
class BenchmarkMetrics
{
  public:
    /**
     * @brief Calculate and report all metrics for a GEMM benchmark
     * @param state Google Benchmark state object
     * @param m, n, k Matrix dimensions
     * @param a_type, b_type, c_type Matrix data types
     * @param group_size Quantization group size along K (0 = full K)
     */
    static void calculateAndReport(benchmark::State& state,
                                   md_t              m,
                                   md_t              n,
                                   md_t              k,
                                   MatrixType        a_type,
                                   MatrixType        b_type,
                                   MatrixType        c_type,
                                   md_t              group_size);

    /**
     * @brief Get the addressable storage-unit size for a matrix type.
     *
     * Packed 4-bit types return one byte because memory is byte-addressable.
     * Use getMatrixSizeBytes() for the exact aggregate size of packed data.
     */
    static size_t getMatrixTypeSize(MatrixType type);

    /**
     * @brief Get the dense (unpadded) byte footprint of a rows x cols matrix.
     *
     * Packed 4-bit types are counted as ceil(rows * cols / 2), matching the
     * framework's linear nibble packing. This ignores leading-dimension
     * padding and is intended for bandwidth and metric accounting only. Use
     * MatrixMemory::calculateRequiredBytes() to size an allocation.
     */
    static size_t getMatrixSizeBytes(MatrixType type, md_t rows, md_t cols);
};

} // namespace dlp::benchmarking
