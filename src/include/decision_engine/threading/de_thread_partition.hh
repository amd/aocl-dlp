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

#include <cstdlib>

#include "classic/dlp_base_types.h"
#include "classic/dlp_macros.h"
#include "utils/math_utils.hh"

// The 2-D thread factorization, available to the DE before it settles on a
// tile.
//
// The tile and the split are one decision. ic_ways bounds the M rows a way
// holds, which MR divides, and jc_ways the N panels, which NR divides, so
// scoring a tile without the split mismeasures the work. Running it this early
// is safe because the dependency goes one way: this reads MR and NR and feeds
// nothing back.
//
// The C original in classic/frame/threading still serves every other datatype,
// so there are two copies of this and nothing enforces that they agree. Keep
// the transcription below line-for-line faithful, comments included; the
// equivalence is checked against the C functions over an exhaustive sweep.
// Sharing one definition would mean moving it under src/include, since
// CLASSIC_INCLUDES carries src/include but DLP_PLUS_INCLUDES does not carry
// classic/include.
namespace dlp::de::thread_partition {

// Partition a thread count into two factors nt1 and nt2 such that
// nt1/nt2 ~= work1/work2.
//
// This depends on the thread count and the two work extents, not on the tile,
// so a candidate sweep can hoist it out of the scoring loop and redo only the
// tile-dependent adjustments per candidate.
DLP_ALWAYS_INLINE void
partition2x2(md_t nThread, md_t work1, md_t work2, md_t& nt1, md_t& nt2)
{
    // Return early small prime numbers of threads.
    if (nThread < 4) {
        nt1 = (work1 >= work2 ? nThread : 1);
        nt2 = (work1 < work2 ? nThread : 1);

        return;
    }

    md_t tn1 = 1;
    md_t tn2 = 1;

    utils::math::primeFactors factors =
        utils::math::primeFactorization(nThread);

    // Fast algorithm: assign prime factors in increasing order to whichever
    // partition has more work to do. The work is divided by the number of
    // threads assigned at each iteration. This algorithm is sub-optimal in
    // some cases. We attempt to mitigate the cases that involve at least one
    // factor of 2. For example, in the partitioning of 12 with equal work
    // this algorithm tentatively finds 6x2. This factorization involves a
    // factor of 2 that can be reallocated, allowing us to convert it to the
    // optimal solution of 4x3. But some cases cannot be corrected this way
    // because they do not contain a factor of 2. For example, this algorithm
    // factors 105 (with equal work) into 21x5 whereas 7x15 would be optimal.
    md_t f;
    while ((f = utils::math::nextPrimeFactor(factors)) > 1) {
        if (work1 > work2) {
            work1 /= f;
            tn1 *= f;
        } else {
            work2 /= f;
            tn2 *= f;
        }
    }

    // Sometimes the last factor applied is prime. For example, on a square
    // matrix, we tentatively arrive (from the logic above) at a 2x6
    // factorization when given 12 ways of parallelism. Below, we make a final
    // attempt at rebalancing nt1 and nt2 by checking to see if the gap between
    // work1 and work2 is narrower if we reallocate a factor of 2.
    if (work1 > work2) {
        if (tn2 % 2 == 0) {
            md_t diff    = work1 - work2;
            md_t diffMod = std::abs(work1 / 2 - work2 * 2);

            if (diffMod < diff) {
                tn1 *= 2;
                tn2 /= 2;
            }
        }
    } else if (work1 < work2) {
        if (tn1 % 2 == 0) {
            md_t diff    = work2 - work1;
            md_t diffMod = std::abs(work2 / 2 - work1 * 2);

            if (diffMod < diff) {
                tn1 /= 2;
                tn2 *= 2;
            }
        }
    }

    nt1 = tn1;
    nt2 = tn2;
}

// Rebalance a seed partition to spread the B NR panels more evenly. Unlike
// partition2x2 above, this depends on the tile, so a candidate sweep has to
// redo it per candidate.
DLP_ALWAYS_INLINE void
adjustIcJcWays(md_t  mr,
               md_t  nr,
               md_t  m,
               md_t  n,
               md_t& nThreads,
               md_t& icWays,
               md_t& jcWays)
{
    // This function currently only increments ic and subsequently decrements
    // jc. Cannot proceed if all threads are allocated to ic.
    // The factorization adjustment here is based on improving the B NR panel
    // distribution among the jc threads.
    md_t mu = (m + mr - 1) / mr;
    md_t nu = (n + nr - 1) / nr;

    // The next 3 ic factors will be considered to see if it results in better
    // NR panel distribution and subsequently reduce the per thread panel work.
    md_t nuModJcWays = nu % jcWays;
    if ((nuModJcWays != 0) && (icWays < nThreads)) {
        md_t muIcCur      = (mu + icWays - 1) / icWays;
        md_t nuJcCur      = (nu + jcWays - 1) / jcWays;
        md_t panelWorkCur = muIcCur + nuJcCur;

        const md_t nextIc        = utils::math::nextFactor(nThreads, icWays);
        const md_t prevJc        = utils::math::prevFactor(nThreads, jcWays);
        md_t       muIcNext      = (mu + nextIc - 1) / nextIc;
        md_t       nuJcPrev      = (nu + prevJc - 1) / prevJc;
        md_t       panelWorkNext = muIcNext + nuJcPrev;

        if (panelWorkNext < panelWorkCur) {
            panelWorkCur = panelWorkNext;
            icWays       = nextIc;
            jcWays       = prevJc;
        }

        nuModJcWays = nu % jcWays;
        if ((nuModJcWays != 0) && (nextIc < nThreads)) {
            const md_t nextNextIc   = utils::math::nextFactor(nThreads, nextIc);
            const md_t prevPrevJc   = utils::math::prevFactor(nThreads, prevJc);
            md_t       muIcNextNext = (mu + nextNextIc - 1) / nextNextIc;
            md_t       nuJcPrevPrev = (nu + prevPrevJc - 1) / prevPrevJc;
            md_t       panelWorkNextNext = muIcNextNext + nuJcPrevPrev;

            if (panelWorkNextNext < panelWorkCur) {
                panelWorkCur = panelWorkNextNext;
                icWays       = nextNextIc;
                jcWays       = prevPrevJc;
            }

            nuModJcWays = nu % jcWays;
            if ((nuModJcWays != 0) && (nextNextIc < nThreads)) {
                const md_t nextNextNextIc =
                    utils::math::nextFactor(nThreads, nextNextIc);
                const md_t prevPrevPrevJc =
                    utils::math::prevFactor(nThreads, prevPrevJc);
                md_t muIcNextNextNext =
                    (mu + nextNextNextIc - 1) / nextNextNextIc;
                md_t nuJcPrevPrevPrev =
                    (nu + prevPrevPrevJc - 1) / prevPrevPrevJc;
                md_t panelWorkNextNextNext =
                    muIcNextNextNext + nuJcPrevPrevPrev;

                if (panelWorkNextNextNext < panelWorkCur) {
                    icWays = nextNextNextIc;
                    jcWays = prevPrevPrevJc;
                }
            }
        }
    }
}

DLP_ALWAYS_INLINE void
adjustForPrimeThreadCount(md_t  mr,
                          md_t  nr,
                          md_t  m,
                          md_t  n,
                          md_t& nThreads,
                          md_t& icWays,
                          md_t& jcWays)
{
    md_t mrBlks = (m + mr - 1) / mr;
    md_t nrBlks = (n + nr - 1) / nr;

    // With prime numbers, only two pairs of factors are possible
    // (1, prime number) and (prime number, 1). Both of these scenarios can
    // lead to suboptimal threading performance depending on the input. Hence,
    // we try to adjust the factors to get a better balance by reducing the
    // number of threads by 1 and checking if it results in more favourable
    // factors.
    md_t reducedIc       = 1;
    md_t reducedJc       = 1;
    md_t reducedNThreads = nThreads - 1;
    partition2x2(reducedNThreads, m, n, reducedIc, reducedJc);
    if (mrBlks >= reducedIc) {
        adjustIcJcWays(mr, nr, m, n, reducedNThreads, reducedIc, reducedJc);
    }

    md_t mrBlksIc     = (mrBlks + icWays - 1) / icWays;
    md_t nrBlksJc     = (nrBlks + jcWays - 1) / jcWays;
    md_t panelWorkOrg = mrBlksIc + nrBlksJc;

    md_t mrBlksReducedIc = (mrBlks + reducedIc - 1) / reducedIc;
    md_t nrBlksReducedJc = (nrBlks + reducedJc - 1) / reducedJc;
    md_t panelWorkRed    = mrBlksReducedIc + nrBlksReducedJc;

    if (panelWorkRed < panelWorkOrg) {
        icWays   = reducedIc;
        jcWays   = reducedJc;
        nThreads = reducedNThreads;
    }
}

// The factorizer, as a base that a decision engine inherits.
//
// The skeleton below is shared because every classic per-datatype factorizer
// already agrees on it. Each one honours a caller-supplied way count first,
// collapses to a single direction when the problem is degenerate in M or N,
// and otherwise seeds from partition2x2. They differ only in what happens
// after that seed: BF16 rebalances on panel work and again for a prime thread
// count, while F32 works from MC, NC and KC and does neither.
//
// So adjustWays is the seam. A datatype that needs its own rebalancing
// overrides that and reaches for the free functions above, rather than
// restating the skeleton. A backend that overrides nothing keeps the
// unadjusted seed, which is a valid partition.
class gemmThreadPartitioner
{
  public:
    virtual ~gemmThreadPartitioner() = default;

    // Resolve a thread count into icWays x jcWays for a given tile.
    //
    // All three of nThreads, icWays and jcWays are in/out. On entry the ways
    // carry the caller's DLP_IC_NT / DLP_JC_NT override, non-positive where
    // unset.
    //
    // Every route out of here satisfies icWays * jcWays == nThreads, which the
    // threading decorator relies on: it recovers the count from the product.
    // nThreads can come back lower than it went in, because dropping a thread
    // sometimes divides the work better.
    DLP_ALWAYS_INLINE void resolveWays(const md_t mr,
                                       const md_t nr,
                                       const md_t m,
                                       const md_t n,
                                       md_t&      nThreads,
                                       md_t&      icWays,
                                       md_t&      jcWays) const
    {
        if ((icWays > 0) || (jcWays > 0)) {
            // If DLP_IC_NT or JC_NT are set.
            // Default cases.
            icWays = (icWays > 0) ? icWays : 1;
            jcWays = (jcWays > 0) ? jcWays : 1;

            nThreads = jcWays * icWays;
        } else if (nThreads > 1) {
            md_t       mrBlks            = (m + mr - 1) / mr;
            md_t       nrBlks            = (n + nr - 1) / nr;
            md_t       mrxnrBlks         = mrBlks * nrBlks;
            md_t       mrBlksAdjNThreads = (nThreads / mrBlks) * mrBlks;
            md_t       deltaMrBlksAdj    = nThreads - mrBlksAdjNThreads;
            const md_t lowFreqThres      = 6;

            if (n <= nr) {
                icWays   = nThreads;
                jcWays   = 1;
                nThreads = icWays * jcWays;
            } else if (m <= mr) {
                jcWays   = nThreads;
                icWays   = 1;
                nThreads = icWays * jcWays;
            } else if (((n % nr) == 0) && (mrxnrBlks <= nThreads)
                       && (deltaMrBlksAdj < lowFreqThres)) {
                icWays   = mrBlks;
                jcWays   = nThreads / icWays;
                nThreads = icWays * jcWays;
            } else {
                // If DLP_NUM_THREADS are set, generate jc,ic from the same.
                partition2x2(nThreads, m, n, icWays, jcWays);

                adjustWays(mr, nr, m, n, nThreads, icWays, jcWays);
            }
        } else {
            // Setting all the values to 1 in case nThreads <= 1. This ensures
            // the threading parameters are valid.
            nThreads = 1;
            jcWays   = 1;
            icWays   = 1;
        }
    }

  protected:
    // What a datatype does to the seed partition. The default is to accept it
    // unchanged.
    //
    // Only the general arm above reaches this. The caller-override and
    // degenerate arms are not a datatype's to reinterpret, and a thread count
    // of one leaves nothing to balance.
    DLP_ALWAYS_INLINE virtual void adjustWays(md_t  mr,
                                              md_t  nr,
                                              md_t  m,
                                              md_t  n,
                                              md_t& nThreads,
                                              md_t& icWays,
                                              md_t& jcWays) const
    {
        (void)mr;
        (void)nr;
        (void)m;
        (void)n;
        (void)nThreads;
        (void)icWays;
        (void)jcWays;
    }
};

} // namespace dlp::de::thread_partition
