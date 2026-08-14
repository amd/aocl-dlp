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

#include <cmath>

#include "classic/dlp_base_types.h"
#include "classic/dlp_macros.h"

// Number theory the thread partitioner needs and nothing about threads.
//
// These were part of de_thread_partition.hh, where they read as though the
// answers were about work distribution. They are not: a divisor of n is a
// divisor of n whatever n counts. Keeping them here means the partitioner holds
// only the part that knows what a thread is.
//
// The prime helpers port dlp_prime_factors_t and its two functions from
// dlp_gemm_thread_utils.c, which are file-static there and so unreachable from
// C++. The two copies must agree and nothing enforces that, so the
// transcription is faithful.
namespace dlp::utils::math {

struct primeFactors
{
    md_t n;
    md_t sqrtN;
    md_t f;
};

DLP_ALWAYS_INLINE primeFactors
primeFactorization(md_t n)
{
    return primeFactors{ n,
                         static_cast<md_t>(std::sqrt(static_cast<double>(n))),
                         2 };
}

DLP_ALWAYS_INLINE md_t
nextPrimeFactor(primeFactors& factors)
{
    // Sieve of eratosthenes style approach, but only for the factors of a
    // single number n, not for a range of numbers.
    while (factors.f <= factors.sqrtN) {
        if (factors.f == 2) {
            if (factors.n % 2 == 0) {
                factors.n /= 2;
                return 2;
            }
            factors.f = 3;
        } else if (factors.f == 3) {
            if (factors.n % 3 == 0) {
                factors.n /= 3;
                return 3;
            }
            factors.f = 5;
        } else if (factors.f == 5) {
            if (factors.n % 5 == 0) {
                factors.n /= 5;
                return 5;
            }
            factors.f = 7;
        } else if (factors.f == 7) {
            if (factors.n % 7 == 0) {
                factors.n /= 7;
                return 7;
            }
            factors.f = 11;
        } else {
            if (factors.n % factors.f == 0) {
                factors.n /= factors.f;
                return factors.f;
            }
            factors.f++;
        }
    }

    // To get here we must be out of prime factors, leaving only n (if it is
    // prime) or an endless string of 1s.
    md_t tmp  = factors.n;
    factors.n = 1;
    return tmp;
}

DLP_ALWAYS_INLINE bool
isPrime(md_t n)
{
    if (n < 2) {
        return false;
    }

    primeFactors factors = primeFactorization(n);
    return nextPrimeFactor(factors) == n;
}

// The next divisor of nt strictly above partNt, and the previous one strictly
// below it.
//
// These walk the divisor lattice of nt in step, which is what lets a caller
// move one factor up and the other down while holding their product fixed:
// divisors are symmetric about the square root.
DLP_ALWAYS_INLINE md_t
nextFactor(const md_t nt, const md_t partNt)
{
    if (partNt == nt) {
        return partNt;
    }

    md_t ntTemp = partNt + 1;
    while ((ntTemp <= nt) && ((nt % ntTemp) != 0)) {
        ntTemp++;
    }
    return ntTemp;
}

DLP_ALWAYS_INLINE md_t
prevFactor(const md_t nt, const md_t partNt)
{
    if (partNt == 1) {
        return partNt;
    }

    md_t ntTemp = partNt - 1;
    while ((ntTemp >= 1) && ((nt % ntTemp) != 0)) {
        ntTemp--;
    }
    return ntTemp;
}

} // namespace dlp::utils::math
