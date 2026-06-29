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

// JIT code generator for the column-major BF16 pack-B kernel (AVX-512 only).
//
// The packed output is byte-for-byte identical to the row-major BF16 packer:
// two consecutive K rows are interleaved per lane (vdpbf16ps), so the panel is
// laid out in K-pairs with rs_dst = NR*2 bf16 and cs_dst = NR/2. Only the read
// path differs — the source is column-major, so a 16x16 transpose reorders the
// data into that layout instead of the masked-ZMM interleave used row-major.
//
// Algorithm (one 16-wide n sub-block, one 32-K tile):
//   * Load 16 columns, each 32 bf16 (= 16 dwords) contiguous down the column,
//     into Zmm(0..15). Treating each bf16 pair as a 32-bit lane, this is "16
//     columns of 16 dwords".
//   * Apply the shared 16x16 dword transpose (pack_b_transpose_utils.hh).
//   Output
//     row r (in Zmm(storeMap16[r])) then holds, for the 16 columns, the dword
//     (B[2r, n], B[2r+1, n]) in lane n — exactly K-pair r's packed block.
//   * Store the 16 K-pair rows at rs_dst = NR*2 bf16 stride.
//
// K is tiled in steps of 32 real-K (= one full ZMM = 16 K-pairs). The trailing
// K % 32 elements are handled by a runtime epi16-masked load; odd K falls out
// for free because the zero-masked partner lane fills the dangling K-pair.
//
// One instance generates a single panel-width kernel (params.NR). The
// orchestrator builds the same fringe ladder as the row-major path: a looped
// kernel per NR-multiple width plus a runtime-masked lt-block kernel
// (params.useMask) whose sub-block loads only n_partial columns.
//
// Register budget (enforced in allocateReg): 16 input ZMMs (the 16 loaded
// columns) + 16 scratch ZMMs (transpose) = 32 ZMM, which AVX-512 provides
// exactly. GP registers come from the Xbyak StackFrame.
template<utils::kernelInstrType KType>
class jitPackBBF16ColMajor : public Xbyak::CodeGenerator
{
  public:
    jitPackBBF16ColMajor();
    ~jitPackBBF16ColMajor()                                 = default;
    jitPackBBF16ColMajor(jitPackBBF16ColMajor&)             = delete;
    jitPackBBF16ColMajor& operator=(jitPackBBF16ColMajor&)  = delete;
    jitPackBBF16ColMajor(jitPackBBF16ColMajor&&)            = delete;
    jitPackBBF16ColMajor& operator=(jitPackBBF16ColMajor&&) = delete;

    dlp::jit::jitGeneratorError generateKernel(
        utils::packBGeneratorParams& params);

  private:
    using Traits  = amdzen::traits::ArchitectureTraits<KType>;
    using RegType = typename Traits::RegType;

    static constexpr int numRegs    = Traits::numRegs;
    static constexpr int RegBytes   = Traits::regBytes;
    static constexpr int ELEM_BYTES = 2; // bf16
    static constexpr int K_FACTOR   = 2; // K-pairing for vdpbf16ps

    // Logical bf16 elements per ZMM (32 for AVX-512). One full K-tile loads
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

    // Opmask for the runtime K-fringe (epi16, up to 32 lanes -> kmovd).
    static constexpr int MASK_KFRINGE = 1; // k1

    Xbyak::Reg64 pParams;
    Xbyak::Reg64 regK;
    Xbyak::Reg64 regKFull;    // K rounded down to a multiple of 32
    Xbyak::Reg64 regLdbBytes; // cs_src * sizeof(bf16): column-to-column
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

    dlp::jit::jitGeneratorError allocateReg();
    void initializeStackFrame(Xbyak::util::StackFrame& sf);
    void initializeParameters();

    // Top-level loop generators.
    void generateFullBlockLoop();
    void generateLtBlockLoop();

    // Pack one panel with K as the outer loop and the NR/16 sub-blocks as the
    // inner loop, matching the intrinsic / F32 col-major write order: for each
    // 32-K tile the sub-blocks are swept in turn, so the packed buffer is
    // written contiguously in K-pair-major order. partialNCols selects the
    // lt-block behaviour (a single sub-block of regSbCount columns). Expects
    // regSrcPanel / regDstPanel to point at the panel start and leaves them
    // unchanged for the caller's panel advance.
    void emitKLoop(bool partialNCols);

    // One 32-K tile: derive the per-tile source / packed bases (regSrcSb /
    // regDstSb) from regKr, then run the sub-block(s) — load 16 columns,
    // transpose, store the K-pair rows.
    void emitTileSweep(bool kMasked, bool partialNCols);

    // Load BLOCK_N columns (each a full / K-masked ZMM) from regSrcSb into
    // Zmm(0..15).
    void emitLoadCols(bool kMasked, bool partialNCols);
    // Store the transposed K-pair rows to regDstSb. When kMasked, only the
    // runtime number of K-pair rows in the K-fringe tile are written.
    void emitStoreRows(bool kMasked);
};

} // namespace amdzen::PackBcodeGenerator
