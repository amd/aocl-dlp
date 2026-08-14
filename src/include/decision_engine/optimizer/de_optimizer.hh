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

#include <cstddef>

#include "classic/dlp_base_types.h"
#include "classic/dlp_macros.h"
#include "decision_engine/de_backend_utils.hh"
#include "decision_engine/de_shape_model.hh"
#include "decision_engine/threading/de_thread_partition.hh"

// The analytical block-shape model: which microkernel tile to run, and the
// thread split that goes with it.
//
// This is a base class rather than a member of the decision engine, so that a
// Reorder engine can inherit the same search a GEMM backend does. Both ends
// need it. A packed B panel does not record the NR it was packed at, so the
// only way a later GEMM recovers that width is to re-run the identical search
// over identical inputs.
//
// It derives the factorizer instead of sitting beside it. A tile can only be
// costed against the split it would run under, so the search needs to be able
// to factorize.
namespace dlp::de::optimizer {

class gemmOptimizer : public thread_partition::gemmThreadPartitioner
{
  public:
    // The entry point. The backend picks the arm, because eligibility depends
    // on state the optimizer does not hold, namely the architecture it runs on.
    //
    // One object, describing one GEMM, and which one was settled by
    // makeModelInput. Neither arm has a second to reach for.
    //
    // The default answer is the tile the context already holds, which is the
    // right answer for any backend with no candidate set of its own.
    DLP_ALWAYS_INLINE virtual shape_model::gemmShapeModelResult
    resolveGemmShape(const shape_model::gemmShapeModelInput& in) const
    {
        return baselineShape(in);
    }

    // The tile for a call the model has no say over. The modelled arm falls
    // back to the same tile where nothing is admissible, but reaches it through
    // the sweep, which resolves a split for it.
    //
    // Nothing here reads an extent. The context tile is the answer whatever the
    // object describes, which is what lets this arm serve a reordered B whose
    // hints were never stated: an object carrying zeros for m and the thread
    // count, with no call to fall back to. Rules needing the extents that will
    // actually run belong in the kernel-info fold, which has them.
    //
    // It must never resolve a split. Factorizing is the modelled arm's job, and
    // the split that comes back from it is the one a candidate was costed
    // against.
    DLP_ALWAYS_INLINE virtual shape_model::gemmShapeModelResult baselineShape(
        const shape_model::gemmShapeModelInput& in) const
    {
        return deferSplit(in, gemmShapeModelUtils::baselineKernelDims(in));
    }

    // The modelled arm. The base has no candidate set, so there is nothing to
    // search and it takes the baseline. A backend with tiles overrides this
    // and calls sweepCandidates below with its own set.
    DLP_ALWAYS_INLINE virtual shape_model::gemmShapeModelResult proposeShape(
        const shape_model::gemmShapeModelInput& in) const
    {
        return baselineShape(in);
    }

    // What a candidate is worth. Lower is better.
    //
    // It scores a shape rather than a tile, so the objective can see the split
    // the tile would run under. A makespan is a property of that pair.
    //
    // The split passed in must be the modelled GEMM's, which is what
    // resolveShape produces. Over a reordered B that is not this call's split,
    // and only the modelled one is a value both ends can compute. Ranking by
    // anything else would let a GEMM outvote the Reorder that already
    // committed the panel width.
    //
    // Nothing calls this while each backend offers a single tile, which leaves
    // the sweep nothing to rank.
    DLP_ALWAYS_INLINE virtual md_t costEval(
        const shape_model::gemmShapeModelInput&  in,
        const shape_model::gemmShapeModelResult& shape) const
    {
        (void)in;
        (void)shape;
        return 0;
    }

  protected:
    // A tile, with the split left to the threading decorator.
    //
    // The ways pass through untouched rather than being zeroed, but nothing
    // downstream may read them: they describe whichever GEMM the input was
    // about, and this arm did not resolve a partition for it. Echoing the
    // input keeps this a pure tile decision, which is what "deferred" means
    // here. The kernel-info fold enforces the rest by publishing only where
    // the factorizer ran.
    DLP_ALWAYS_INLINE static shape_model::gemmShapeModelResult deferSplit(
        const shape_model::gemmShapeModelInput& in,
        const shape_model::kernelDims           dims)
    {
        return shape_model::gemmShapeModelResult{ dims.mr, dims.nr,
                                                  in.num_threads, in.ic_ways,
                                                  in.jc_ways };
    }

    // The same tile, with the split the modelled GEMM would run it under. The
    // counterpart to deferSplit above, and the only caller of the factorizer
    // in this class.
    //
    // It sizes the split from the input as it stands, which over a reordered B
    // is the hinted GEMM. That is what a cost model must score against, and
    // also what must never be published as the split to execute.
    //
    // Called once per candidate by the sweep.
    DLP_ALWAYS_INLINE shape_model::gemmShapeModelResult resolveShape(
        const shape_model::gemmShapeModelInput& in,
        const shape_model::kernelDims           dims) const
    {
        shape_model::gemmShapeModelResult out = deferSplit(in, dims);

        resolveWays(dims.mr, dims.nr, in.m, in.n, out.num_threads, out.ic_ways,
                    out.jc_ways);

        return out;
    }

    // The search. It keeps the first admissible candidate; ranking them by
    // costEval is what turns this into a choice once a backend offers more
    // than one tile.
    //
    // A tile is chosen as a pair, never one dimension at a time: MR and NR
    // trade against the same 32 ZMMs. That is also what the Reorder contract
    // needs. A Reorder cannot hand the width it chose to a later GEMM, so the
    // two agree by running this same sweep over the same inputs, which is why
    // the caller folds the hints into the input.
    //
    // The candidate set arrives as an argument. It belongs to the backend,
    // which owns the JIT generator that decides which tiles exist, and a base
    // cannot name a member of the class that derives it. A virtual accessor
    // would work and is the wrong tool: an objective runs this loop once per
    // candidate, and a vtable load there returns a pointer the compiler cannot
    // fold. Taking the array by reference keeps the extent a compile-time
    // constant, and N deduces from the call.
    //
    // Each candidate is paired with the split it would run under, because a
    // tile's cost is a property of the pair rather than of the tile. The
    // winner's split is the one the backend publishes, so the partition a tile
    // was chosen under is the partition that executes. That is sound only
    // because the input is the GEMM that will run: over a reordered B the
    // classic layer holds the call to the hints before the model is reached.
    //
    // The register budget is not checked. setupRegisterConfig in the JIT
    // backend already counts accumulators, B panel and mask registers against
    // cRegCount and rejects what does not fit, and a second copy of that test
    // would drift from it.
    //
    // An empty admissible set falls back to the baseline, which is what a
    // tunable naming a tile the generator does not offer produces. That
    // fallback returns in.nr on either end, so a Reorder and the GEMM after it
    // fall back to the same width.
    template<std::size_t N>
    DLP_ALWAYS_INLINE shape_model::gemmShapeModelResult sweepCandidates(
        const shape_model::gemmShapeModelInput& in,
        const shape_model::kernelDims (&candidates)[N]) const
    {
        static_assert(N > 0, "a backend must offer at least one tile");

        for (const shape_model::kernelDims& cand : candidates) {
            if (gemmShapeModelUtils::isCandidateAdmissible(in, cand)) {
                return resolveShape(in, cand);
            }
        }

        return resolveShape(in, gemmShapeModelUtils::baselineKernelDims(in));
    }
};

} // namespace dlp::de::optimizer
