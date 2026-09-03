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
#include "traits.hh"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

#include <cstdint>

namespace amdzen::PackBcodeGenerator {

// JIT code generator for the row-major INT8 VNNI-4 pack-B kernel.
//
// INT8 GEMM fuses four consecutive K elements per lane (vpdpbusd), so the
// packed panel is laid out in K-quads: four source rows (k..k+3) are
// interleaved per n into 16-wide ZMM blocks (one VNNI dword per n). For an
// NR-wide panel the per-K-quad block at offset kr*NR + 64*bj holds n =
// 16*bj..16*bj+15.
//
// The transform matches the nr64 intrinsic when NR >= 64: one 64-n chunk is
//   4x ZMM loads -> vpunpcklbw/hbw -> vpunpcklwd/hwd -> vpermt2q (the four
//   nr64 selectors) -> four 64-byte stores.
// Remainder 16/32/48-n (NR=16/32/48, or the tail of NR=96) keeps the nr16
// primitive: 4x 16-byte loads -> unpack -> vinserti32x4 -> one ZMM store.
// Those two sequences are byte-identical; the ZMM path is the one that
// matches intrinsic throughput on the default 6x64 tile.
//
// cs_dst is 64, not NR/4: it is the n-step between adjacent 16-n VNNI ZMMs
// (16 n * 4 k bytes). The nr64 intrinsic's *cs_p = NR is only correct because
// NR is hardcoded 64 there. rs_dst is NR * 4 (bytes to the next K-quad).
//
// When packBGeneratorParams.accColSum is set by the S8 generator identity,
// the kernel also accumulates compensation:
// col_sum[n] += 128 * sum_k B[k, n]. U8 generation leaves the flag false and
// emits no trailer code. The pointer is kept in rax (not a StackFrame temp)
// and is required non-null for the S8 specialization.
//
// One instance generates a single panel-width kernel (params.NR). The
// orchestrator builds the fringe ladder: every NR-multiple width plus a
// runtime-masked lt16 variant (params.useMask), whose 16-byte loads take
// their opmask from packBParams.nFringeMaskPerBlock[0].
template<utils::kernelInstrType KType>
class jitPackBINT8 : public Xbyak::CodeGenerator
{
  public:
    jitPackBINT8();
    ~jitPackBINT8()                         = default;
    jitPackBINT8(jitPackBINT8&)             = delete;
    jitPackBINT8& operator=(jitPackBINT8&)  = delete;
    jitPackBINT8(jitPackBINT8&&)            = delete;
    jitPackBINT8& operator=(jitPackBINT8&&) = delete;

    dlp::jit::jitGeneratorError generateKernel(
        utils::packBGeneratorParams& params);

  private:
    using Traits  = amdzen::traits::ArchitectureTraits<KType>;
    using RegType = typename Traits::RegType;

    static constexpr int numRegs        = Traits::numRegs;
    static constexpr int RegBytes       = Traits::regBytes;
    static constexpr int ELEM_BYTES     = 1; // int8
    static constexpr int numElemsPerReg = RegBytes / ELEM_BYTES;
    // Output block width in n (16 VNNI dwords per ZMM).
    static constexpr int BLOCK_N  = 16;
    static constexpr int K_FACTOR = 4;

    md_t NR_;
    int  numBlocks; // NR_ / BLOCK_N
    bool useMask_;
    bool nLoop_;    // wrap the panel body in the N-panel (jc) loop
    bool useZmm64_; // emit the nr64 ZMM chunk when NR>=64 and not lt16
    bool accColSum_;

    // nr64 vpermt2q index vectors (epi64), same constants as
    // dlp_packb_nr64_u8s8s32o32_row_major.
    int64_t      selector1[8]    = { 0x0, 0x1, 0x8, 0x9, 0x2, 0x3, 0xA, 0xB };
    int64_t      selector1_1[8]  = { 0x4, 0x5, 0xC, 0xD, 0x6, 0x7, 0xE, 0xF };
    int64_t      selector2[8]    = { 0x0, 0x1, 0x2, 0x3, 0x8, 0x9, 0xA, 0xB };
    int64_t      selector2_1[8]  = { 0x4, 0x5, 0x6, 0x7, 0xC, 0xD, 0xE, 0xF };
    const md_t   selector1_off   = 0;
    const md_t   selector1_1_off = selector1_off + sizeof(selector1);
    const md_t   selector2_off   = selector1_1_off + sizeof(selector1_1);
    const md_t   selector2_1_off = selector2_off + sizeof(selector2);
    Xbyak::Label selectors;

    Xbyak::Address get_selector(md_t off) { return ptr[rip + selectors + off]; }

    // Persistent XMM/ZMM assignments (matches the nr16 unpack reuse).
    static constexpr int XMM_ZERO = 0; // zero row for K-tail / unused loads
    static constexpr int XMM_A    = 1; // row k0, then unpackhi k0/k1
    static constexpr int XMM_B    = 2; // row k1, then n0 (ZMM store)
    static constexpr int XMM_C    = 3; // row k2, then unpackhi k2/k3
    static constexpr int XMM_D    = 4; // row k3, then n2
    static constexpr int XMM_ABLO = 5; // unpacklo k0/k1, then n1
    static constexpr int XMM_CDLO = 6; // unpacklo k2/k3, then n3

    // 64-n path lives in zmm16-31 (volatile on Windows x64).
    static constexpr int ZMM_SEL1   = 16;
    static constexpr int ZMM_SEL1_1 = 17;
    static constexpr int ZMM_SEL2   = 18;
    static constexpr int ZMM_SEL2_1 = 19;
    static constexpr int ZMM_A64    = 20;
    static constexpr int ZMM_B64    = 21;
    static constexpr int ZMM_C64    = 22;
    static constexpr int ZMM_D64    = 23;
    static constexpr int ZMM_T0     = 24;
    static constexpr int ZMM_T1     = 25;
    static constexpr int ZMM_P0     = 26;
    static constexpr int ZMM_P1     = 27;
    static constexpr int ZMM_P2     = 28;
    static constexpr int ZMM_P3     = 29;
    static constexpr int ZMM_S0     = 30;
    static constexpr int ZMM_S1     = 31;

    // s8s8 col-sum: vpmovsxbd scratch (Windows-volatile) and one ZMM acc
    // per 16-n block. NR=96 uses zmm8..zmm13. Scale is <<7 (= *128).
    static constexpr int ZMM_CVT       = 7;
    static constexpr int ZMM_ACC0      = 8;
    static constexpr int COL_SUM_SHIFT = 7; // *128

    static constexpr int MASK_LT16 = 1; // k1

    Xbyak::Reg64 pParams;
    Xbyak::Reg64 regSrc;
    Xbyak::Reg64 regDst;
    Xbyak::Reg64 regK;
    Xbyak::Reg64 regLdbBytes;
    Xbyak::Reg64 regLdb3; // 3 * rs_src bytes, for the k+3 row
    Xbyak::Reg64 regDstPanelStride;
    Xbyak::Reg64 regSrcEnd;
    Xbyak::Reg64 regSrcBase;
    Xbyak::Reg64 regDstBase;
    Xbyak::Reg64 regSrcRow;
    Xbyak::Reg64 regDstRow;
    Xbyak::Reg64 regKFull;
    Xbyak::Reg64 regKr;
    // Not a StackFrame temp: rax is free (Xbyak never assigns it to p[]/t[]).
    Xbyak::Reg64 regColSum;

    dlp::jit::jitGeneratorError allocateReg();
    void initializeStackFrame(Xbyak::util::StackFrame& sf);
    void initializeParameters();
    void embedSelectors();
    void generateFullPanelLoop();
    // Emit one NR-wide panel: the K-quad loop plus the K%4 tail. Expects
    // regSrcRow / regDstRow to point at the panel start; both are advanced.
    void generatePanelBody();
    // Emit one NR-wide K-quad transform. nLiveRows in {1,2,3,4} is the number
    // of source K-rows that exist; missing rows are taken as zero.
    void interleaveKQuad(int nLiveRows);
    // One 64-n chunk starting at 16-n block index bj (0, 4, ...).
    void interleaveKQuad64(int nLiveRows, int bj);
    void loadRow16(const Xbyak::Xmm&     dst,
                   const Xbyak::Address& addr,
                   bool                  live);
    void loadRow64(const Xbyak::Zmm&     dst,
                   const Xbyak::Address& addr,
                   bool                  live);
    void zeroColSumAcc();
    void accXmm16(int bj, const Xbyak::Xmm& row, bool live);
    void accZmm64(int               bj,
                  const Xbyak::Zmm& a,
                  const Xbyak::Zmm& b,
                  const Xbyak::Zmm& c,
                  const Xbyak::Zmm& d,
                  int               nLiveRows);
    void flushColSum();
    void advanceColSum();
};

} // namespace amdzen::PackBcodeGenerator
