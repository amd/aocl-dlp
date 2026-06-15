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

#include "xbyak/xbyak.h"

namespace amdzen::codegen::alias_mitigation {

// Codegen helper used by the two-pass MR/2 alias-mitigation path in
// the GEMV N=1 generators (F32, BF16). Emits
// `regBasePtr += numRows * regRowStride` using `regScratch` as the
// multiplication target. Used between the two MR/2 passes to advance
// the A row cursor to the second half of the rows.
//
// The decision to emit the two-pass body (and call this helper) is
// made by the DE (see decision_engine/alias_detection_utils.hh) and
// communicated to the generator via `kernelInfo::aliasMrSplit`. The
// kernel emits no runtime stride check; one variant emits the
// single-pass MR body, the other emits the two-pass MR/2 body.
//
// Clobbers: `regScratch`.
template<typename Gen>
inline void
emitAdvanceRows(Gen*                g,
                int                 numRows,
                const Xbyak::Reg64& regBasePtr,
                const Xbyak::Reg64& regRowStride,
                const Xbyak::Reg64& regScratch)
{
    g->mov(regScratch, numRows);
    g->imul(regScratch, regRowStride);
    g->add(regBasePtr, regScratch);
}

} // namespace amdzen::codegen::alias_mitigation
