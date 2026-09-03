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

#include "jit/jit_generator_base.hh"
#include "jit_generator_utils.hh"
#include "kernels/kernel_base.hh"
#include "pack_b_transpose_utils.hh"
#include "traits.hh"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

#include <cstdint>

namespace amdzen::PackBcodeGenerator {

// JIT code generator for the column-major INT8 VNNI-4 pack-B kernel
// (AVX-512 only).
//
// The packed output is byte-for-byte identical to the row-major INT8 packer:
// four consecutive K bytes are interleaved per n (vpdpbusd), so the panel is
// laid out in K-quads with rs_dst = NR*4 and cs_dst = 64 (one VNNI ZMM). Only
// the read path differs — the source is column-major, so a 16x16 dword
// transpose reorders 16 columns of 16 K-quads into that layout instead of the
// XMM unpack used row-major.
//
// Algorithm (one 16-wide n sub-block, one 64-K tile):
//   * Load 16 columns, each 64 int8 (= 16 dwords) contiguous down the column,
//     into Zmm(0..15). Treating each K-quad as a 32-bit lane, this is "16
//     columns of 16 dwords".
//   * Apply the shared 16x16 dword transpose (pack_b_transpose_utils.hh).
//     Output row r (in Zmm(storeMap16[r])) then holds, for the 16 columns, the
//     dword (B[4r..4r+3, n]) in lane n — exactly K-quad r's packed block.
//   * Store the 16 K-quad rows at rs_dst = NR*4 bytes.
//
// K is tiled in steps of 64 real-K (= one full ZMM = 16 K-quads). The trailing
// K % 64 elements are handled by a runtime epi8-masked load; K % 4 in {1,2,3}
// falls out for free because the zero-masked partner bytes fill the dangling
// K-quad. The GEMM reads KC_updated = (K+3)&~3, so that padded quad is stored.
//
// One instance generates a single panel-width kernel (params.NR). The
// orchestrator builds the same fringe ladder as the row-major path: a looped
// kernel per NR-multiple width plus a runtime-masked lt-block kernel
// (params.useMask) whose sub-block loads only n_partial columns.
//
// Register budget (enforced in allocateReg): 16 input ZMMs (the 16 loaded
// columns) + 16 scratch ZMMs (transpose) = 32 ZMM, which AVX-512 provides
// exactly. GP registers come from the Xbyak StackFrame.
//
// When packBGeneratorParams.accColSum is set by the S8 generator identity,
// each live column is horizontally summed before the transpose and added as
// (hsum << 7) into col_sum[n]. U8 generation leaves the flag false and emits
// no trailer code. The S8 pointer lives in rax and is advanced by NR int32s
// per nLoop panel.
template<utils::kernelInstrType KType>
class jitPackBINT8ColMajor : public Xbyak::CodeGenerator
{
  public:
    jitPackBINT8ColMajor();
    ~jitPackBINT8ColMajor()                                 = default;
    jitPackBINT8ColMajor(jitPackBINT8ColMajor&)             = delete;
    jitPackBINT8ColMajor& operator=(jitPackBINT8ColMajor&)  = delete;
    jitPackBINT8ColMajor(jitPackBINT8ColMajor&&)            = delete;
    jitPackBINT8ColMajor& operator=(jitPackBINT8ColMajor&&) = delete;

    dlp::jit::jitGeneratorError generateKernel(
        utils::packBGeneratorParams& params);

  private:
    using Traits  = amdzen::traits::ArchitectureTraits<KType>;
    using RegType = typename Traits::RegType;

    static constexpr int numRegs    = Traits::numRegs;
    static constexpr int RegBytes   = Traits::regBytes;
    static constexpr int ELEM_BYTES = 1; // int8
    static constexpr int K_FACTOR   = 4; // K-quads for vpdpbusd

    // Logical int8 elements per ZMM (64 for AVX-512). One full K-tile loads
    // this many real-K values per column.
    static constexpr int numElemsPerReg = RegBytes / ELEM_BYTES;
    // Transpose tile width: 16 columns / 16 dword-rows.
    static constexpr int BLOCK_N = 16;
    // Vector registers needed: 16 input columns + 16 transpose scratch.
    static constexpr int kNumDataRegs    = BLOCK_N;
    static constexpr int kNumScratchRegs = BLOCK_N;
    static constexpr int kNumVecRegs     = kNumDataRegs + kNumScratchRegs;

    md_t NR_;
    int  numSubBlocks_; // NR_ / BLOCK_N
    bool useMask_;      // lt-block kernel (partial-N sub-block)
    bool nLoop_;        // wrap body in the N-panel (jc) loop
    bool accColSum_;

    // Opmask for the runtime K-fringe (epi8, up to 64 lanes -> kmovq).
    static constexpr int MASK_KFRINGE = 1; // k1

    Xbyak::Reg64 pParams;
    Xbyak::Reg64 regK;
    Xbyak::Reg64 regKFull;    // K rounded down to a multiple of 64
    Xbyak::Reg64 regLdbBytes; // cs_src * sizeof(int8): column-to-column
    Xbyak::Reg64 regSrcEnd;   // exclusive end of the full-NR source region
    Xbyak::Reg64 regSrcPanel; // current panel source base
    Xbyak::Reg64 regDstPanel; // current panel packed base
    Xbyak::Reg64 regSrcSb;    // per-tile / sub-block source base
    Xbyak::Reg64 regDstSb;    // per-tile / sub-block packed base
    Xbyak::Reg64 regKr;       // K-loop counter (real-K units)
    Xbyak::Reg64 regSbCount;  // sub-block loop count / partial-N column
                              // count (lt-block kernel)
    Xbyak::Reg64 regColPtr;   // scratch address register
    Xbyak::Reg64 regTmp;
    // Not a StackFrame temp: rax is free (Xbyak never assigns it to p[]/t[]).
    Xbyak::Reg64 regColSum;

    static constexpr int COL_SUM_SHIFT = 7; // *128
    // Horizontal-sum scratch after emitLoadCols, before transpose.
    static constexpr int ZMM_HSUM0 = 16;

    dlp::jit::jitGeneratorError allocateReg();
    void initializeStackFrame(Xbyak::util::StackFrame& sf);
    void initializeParameters();

    void generateFullBlockLoop();
    void generateLtBlockLoop();

    // Pack one panel with K as the outer loop and the NR/16 sub-blocks as the
    // inner loop, matching the intrinsic write order: for each 64-K tile the
    // sub-blocks are swept in turn, so the packed buffer is written
    // contiguously in K-quad-major order. partialNCols selects the lt-block
    // behaviour (a single sub-block of regSbCount columns).
    void emitKLoop(bool partialNCols);

    void emitTileSweep(bool kMasked, bool partialNCols);
    void emitLoadCols(bool kMasked, bool partialNCols);
    void emitStoreRows(bool kMasked);
    void emitColSumFromLoadedCols(bool partialNCols);
    void emitHsumColToMem(int colZmm, int byteOff);
    void advanceColSum();
};

} // namespace amdzen::PackBcodeGenerator
