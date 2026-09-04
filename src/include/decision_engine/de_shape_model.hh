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

#include "bindings/c_wrappers/capi_kernel_frame_wrappers.h"
#include "classic/dlp_base_types.h"
#include "classic/dlp_macros.h"

// The vocabulary of the analytical block-shape model: what it is asked, what
// it answers with, and what it chooses between.
//
// Only the types live here, because the three parts of the model vary
// independently: the questions asked of an input are pure and sit in
// de_backend_utils.hh, the tiles that exist are a fact about one datatype's JIT
// generator so each backend names its own set, and the search that ranks them
// is in optimizer/de_optimizer.hh. Keeping the types apart from all three lets
// the helpers and the backends name the same input without including each
// other.
namespace dlp::de::shape_model {

// Everything the model is a function of. A struct rather than a parameter list
// so later terms, cache capacities or MC and NC, can be added without touching
// call sites; k is carried for that reason and nothing reads it.
//
// Strides and memory tags are absent by design. Strides belong to the L1 alias
// guard, which runs after this model. The tag is read once, by makeModelInput
// below, to decide which GEMM these fields describe.
//
// Every field describes one GEMM, and they all describe the same one. An object
// is never a mixture of the call in hand and the GEMM the hints characterise: a
// reader taking m from one and the thread count from the other would derive a
// width no packed panel was built at.
struct gemmShapeModelInput
{
    // The rows of the GEMM this object describes. Over a reordered B the hints
    // stand in for it, and a zero there is the application declining to
    // describe a GEMM at all, which makes the call ineligible and leaves the
    // context tile standing on both ends alike.
    md_t m;
    md_t n;
    md_t k;

    // The tile on entry: the context default, i.e. what applies when no rule
    // fires.
    md_t mr;
    md_t nr;

    // DLP_BLKSZ_SET_* bits marking which of those two the application
    // authored. A set bit is a constraint the model reports back rather than
    // searching around.
    md_t frozen;

    // Threads this GEMM runs on, and the DLP_IC_NT / DLP_JC_NT override,
    // non-positive where unset. These are inputs to a partition, never the
    // partition itself.
    //
    // Over a reordered B the ways arrive unset whatever the runtime was pinned
    // to. A cost model ranks candidates by the split they would run under, and
    // a Reorder has no runtime to read a pin from, so a GEMM that fed one in
    // would rank against a partition its Reorder could not have seen. The pin
    // reaches the threading decorator by its own route regardless.
    //
    // A pinned call leaves num_threads at -1, because the runtime answers with
    // ways or with a count and never both.
    md_t num_threads;
    md_t ic_ways;
    md_t jc_ways;

    // Whether this shape decision is for a reordered/packed-B panel (i.e., B is
    // packed once and then reused). The cost model has no packing term, so it
    // is only consulted in this case rather than on pack-on-the-fly GEMMs.
    bool b_reordered;
};

// The one place an input is built, and the one place the memory tag is read.
//
// Which GEMM the model is asked about is decided here, once, from the tag:
//
//   B reordered   the GEMM the hints characterise, not this call. A packed
//                 panel does not record the width it was built at, so a later
//                 GEMM recovers that width only by re-running the Reorder's
//                 search over the Reorder's inputs, and all the Reorder had
//                 was the hints. Feeding this call's own extents in would let
//                 a GEMM outvote a panel that cannot change. Unset hints
//                 travel through as zeros and come out ineligible, leaving
//                 both ends on the context tile.
//
//   anything else this call. No panel was packed, so there is no earlier
//                 decision to reproduce. Hints stated here describe nothing
//                 and are not read, which is why they need not be cleared
//                 upstream to be ignored.
//
// The thread fields follow from that. The hinted object takes nt_hint and no
// ways: a Reorder has no runtime to read a pin from, so ways would be a field
// only one end could fill, and both ends have to rank alike. The call's object
// takes the runtime's answer as the runtime spelled it, a count or a pair of
// ways, either of which describes a pool that isEligible accepts.
//
// Deciding this once lets everything downstream read one object without asking
// which GEMM it holds. Both ends also reach here from different translation
// units, a Reorder through the pack-B init and a GEMM through the kernel init,
// and the search is only as identical as its input, so assembling it in one
// place is what keeps the two field lists from drifting apart.
DLP_ALWAYS_INLINE gemmShapeModelInput
makeModelInput(md_t m,
               md_t n,
               md_t k,
               md_t mr,
               md_t nr,
               md_t frozen,
               md_t m_hint,
               md_t nt_hint,
               md_t num_threads,
               md_t ic_ways,
               md_t jc_ways,
               bool b_reordered)
{
    if (b_reordered) {
        return gemmShapeModelInput{ m_hint,
                                    n,
                                    k,
                                    mr,
                                    nr,
                                    frozen,
                                    nt_hint,
                                    /*ic_ways=*/0,
                                    /*jc_ways=*/0,
                                    /*b_reordered=*/true };
    }

    return gemmShapeModelInput{
        m,      n,           k,       mr,      nr,
        frozen, num_threads, ic_ways, jc_ways, /*b_reordered=*/false
    };
}

// The tile, and the split to run it under where the model resolved one.
//
// The ways are an answer only where the factorizer ran. There num_threads is
// part of that answer rather than an echo of the input, and can come back lower
// than offered when that divides the work more evenly; ic_ways * jc_ways ==
// num_threads holds. Everywhere else they are whatever the input carried and
// mean nothing, which is why the kernel-info fold publishes a split only where
// the model resolved one.
struct gemmShapeModelResult
{
    md_t mr;
    md_t nr;
    md_t num_threads;
    md_t ic_ways;
    md_t jc_ways;
};

// A microkernel tile. One type throughout, for both a tile the generator can
// emit and the tile the model settled on. Which of the two a given kernelDims
// is depends only on where it sits, so the array a backend offers is named for
// its role instead.
struct kernelDims
{
    md_t mr;
    md_t nr;
};

} // namespace dlp::de::shape_model
