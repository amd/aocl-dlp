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

/*
 * Numerical tests for post-ops applied directly to a vector, without a GEMM:
 * the aocl_gemm_<op>_f32 entry points in aocl_util_interface_apis.h. Each is
 * checked against a scalar double-precision reference, so the assertions land
 * on the activation's own arithmetic rather than on a GEMM result it was fused
 * into. This is the place to add coverage as more standalone post-ops appear.
 *
 * The sibling test_postops_*.cc files cover the YAML/UAL post-op plumbing --
 * parsing, parameter expansion, and combination validation -- not the numerics
 * of the ops themselves.
 *
 * Currently covered: softmax, GELU tanh, GELU erf.
 *
 * These had no numerical coverage anywhere in the suite, which let a silently
 * wrong softmax ship: its reduction pass exponentiated x only into a register,
 * so the division pass re-read the untouched input from memory and returned
 * x_i / sum(exp(x_j)) instead of exp(x_i) / sum(exp(x_j)).
 *
 * The n sweeps below deliberately cover every vector/remainder branch of both
 * kernels: 16 elements per AVX-512 iteration and 8 + 4 + scalar-tail for AVX2,
 * so n = 1..64 exercises all of them plus their interactions.
 */

#include "aocl_dlp.h"
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

// Budgets for the vectorised expf approximation plus the horizontal
// double-precision denominator sum. The AVX2, SSE and scalar incx paths each
// reach the result differently, so none of them agrees bit-for-bit with a
// scalar reference; this is the spread across all three, not a rounding
// allowance for one.
constexpr float kTol      = 1e-5f;
constexpr float kSentinel = -12345.0f;

// Scalar double-precision reference: exp(x_i) / sum_j(exp(x_j)).
std::vector<float>
ref_softmax(const std::vector<float>& x, md_t n, md_t incx)
{
    double sum = 0.0;
    for (md_t i = 0; i < n; i++)
        sum += std::exp(static_cast<double>(x[i * incx]));

    std::vector<float> out = x;
    for (md_t i = 0; i < n; i++)
        out[i * incx] = static_cast<float>(
            std::exp(static_cast<double>(x[i * incx])) / sum);
    return out;
}

// Runs the library softmax over a strided vector padded with a sentinel, then
// checks it against the reference, that the result sums to 1, and that nothing
// outside the addressed lanes was written.
void
check_softmax(md_t n, md_t incx, const std::vector<float>& in)
{
    SCOPED_TRACE("n=" + std::to_string(n) + " incx=" + std::to_string(incx));

    std::vector<float> got = in;
    std::vector<float> ref = ref_softmax(in, n, incx);

    aocl_gemm_softmax_f32(n, got.data(), incx);

    double sum = 0.0;
    for (md_t i = 0; i < n; i++) {
        EXPECT_NEAR(got[i * incx], ref[i * incx], kTol) << "at element " << i;
        sum += got[i * incx];
    }
    EXPECT_NEAR(sum, 1.0, 1e-4) << "softmax output must sum to 1";

    for (std::size_t i = 0; i < got.size(); i++) {
        const bool addressed = (i % static_cast<std::size_t>(incx) == 0)
                               && (i / static_cast<std::size_t>(incx)
                                   < static_cast<std::size_t>(n));
        if (!addressed) {
            EXPECT_FLOAT_EQ(got[i], kSentinel) << "wrote outside lane " << i;
        }
    }
}

// n*incx lanes of data plus a sentinel-filled tail, so an over-write past the
// logical end of the vector is caught rather than silently tolerated.
std::vector<float>
make_input(md_t n, md_t incx, float (*gen)(md_t))
{
    std::vector<float> v(static_cast<std::size_t>(n * incx) + 8, kSentinel);
    for (md_t i = 0; i < n; i++)
        v[i * incx] = gen(i);
    return v;
}

} // namespace

// A softmax over an all-equal input is forced by symmetry to return 1/n in
// every lane. No formula that skips the numerator's exp() can produce that:
// the buggy x_i / sum(exp(x_j)) returns 0 here, for any n.
TEST(PostOpsDirect, SoftmaxAllZeroIsUniform)
{
    for (md_t n = 1; n <= 64; n++) {
        SCOPED_TRACE("n=" + std::to_string(n));
        std::vector<float> x(static_cast<std::size_t>(n), 0.0f);

        aocl_gemm_softmax_f32(n, x.data(), 1);

        for (md_t i = 0; i < n; i++)
            EXPECT_NEAR(x[i], 1.0f / static_cast<float>(n), kTol)
                << "at element " << i;
    }
}

TEST(PostOpsDirect, SoftmaxMatchesReferenceUnitStride)
{
    for (md_t n = 1; n <= 64; n++) {
        check_softmax(n, 1, make_input(n, 1, [](md_t i) {
                          return (static_cast<float>(i % 7) - 3.0f) * 0.5f;
                      }));
    }
    // Beyond the remainder branches, into repeated main-loop iterations.
    for (md_t n : { 128, 129, 255, 300 }) {
        check_softmax(n, 1, make_input(n, 1, [](md_t i) {
                          return std::sin(static_cast<float>(i)) * 3.0f;
                      }));
    }
}

TEST(PostOpsDirect, SoftmaxMatchesReferenceNonUnitStride)
{
    for (md_t incx : { 2, 3, 4 }) {
        for (md_t n = 1; n <= 20; n++) {
            check_softmax(n, incx, make_input(n, incx, [](md_t i) {
                              return static_cast<float>(i % 5) - 2.0f;
                          }));
        }
    }
}

// GELU shares the file and the block structure the softmax bug lived in, so
// pin its numerics here too.
TEST(PostOpsDirect, GeluTanhMatchesReference)
{
    for (md_t n = 1; n <= 40; n++) {
        SCOPED_TRACE("n=" + std::to_string(n));
        std::vector<float> x(static_cast<std::size_t>(n));
        for (md_t i = 0; i < n; i++)
            x[i] = (static_cast<float>(i % 9) - 4.0f) * 0.75f;
        const std::vector<float> in = x;

        aocl_gemm_gelu_tanh_f32(n, x.data(), 1);

        for (md_t i = 0; i < n; i++) {
            const double v = in[i];
            const double expected =
                0.5 * v
                * (1.0 + std::tanh(0.797884 * (v + 0.044715 * v * v * v)));
            EXPECT_NEAR(x[i], static_cast<float>(expected), 1e-4)
                << "at element " << i;
        }
    }
}

TEST(PostOpsDirect, GeluErfMatchesReference)
{
    for (md_t n = 1; n <= 40; n++) {
        SCOPED_TRACE("n=" + std::to_string(n));
        std::vector<float> x(static_cast<std::size_t>(n));
        for (md_t i = 0; i < n; i++)
            x[i] = (static_cast<float>(i % 9) - 4.0f) * 0.75f;
        const std::vector<float> in = x;

        aocl_gemm_gelu_erf_f32(n, x.data(), 1);

        for (md_t i = 0; i < n; i++) {
            const double v        = in[i];
            const double expected = 0.5 * v * (1.0 + std::erf(v * 0.707107));
            EXPECT_NEAR(x[i], static_cast<float>(expected), 1e-4)
                << "at element " << i;
        }
    }
}
