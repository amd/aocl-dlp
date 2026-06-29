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

// JIT code generator for the row-major BF16 pack-B kernel.
//
// BF16 GEMM fuses two consecutive K elements per lane (vdpbf16ps), so the
// packed panel is laid out in K-pairs: two source rows (k, k+1) are interleaved
// element-wise (k0n0,k1n0,k0n1,k1n1,...) into 16-wide ZMM blocks. For an
// NR-wide panel the per-K-pair block at offset kr*NR + 32*bj holds n =
// 16*bj..16*bj+15.
//
// The transform is built from a single masked-ZMM primitive:
//   2 loads (rows k, k+1) -> vpunpcklwd/vpunpckhwd -> vpermt2q(selector1) [+
//   vpermt2q(selector1_1)] -> store. A full 32-n chunk uses full ZMM loads and
//   emits two output blocks; a trailing 16-n block uses 16-wide masked loads
//   and emits one output block (the selector1_1 half would be all-zero and is
//   dropped). This keeps every load/store on ZMM, matching the micro-kernel.
//
// One instance generates a single panel-width kernel (params.NR). The
// orchestrator builds the fringe ladder by generating this kernel for each
// NR-multiple width plus a runtime-masked lt-block variant (params.useMask),
// whose trailing 16-lane load takes its opmask from packBParams at run time.
// This generator handles row-major sources; column-major sources are handled by
// the dedicated jitPackBBF16ColMajor generator.
template<utils::kernelInstrType KType>
class jitPackBBF16 : public Xbyak::CodeGenerator
{
  public:
    jitPackBBF16();
    ~jitPackBBF16()                         = default;
    jitPackBBF16(jitPackBBF16&)             = delete;
    jitPackBBF16& operator=(jitPackBBF16&)  = delete;
    jitPackBBF16(jitPackBBF16&&)            = delete;
    jitPackBBF16& operator=(jitPackBBF16&&) = delete;

    dlp::jit::jitGeneratorError generateKernel(
        utils::packBGeneratorParams& params);

  private:
    using Traits  = amdzen::traits::ArchitectureTraits<KType>;
    using RegType = typename Traits::RegType;

    static constexpr int numRegs    = Traits::numRegs;
    static constexpr int RegBytes   = Traits::regBytes;
    static constexpr int ELEM_BYTES = 2; // bf16
    // Logical bf16 elements per ZMM (32 for AVX-512).
    static constexpr int numElemsPerReg = RegBytes / ELEM_BYTES;
    // Output block width in bf16 elements (16 n-values per ZMM block).
    static constexpr int BLOCK_N = 16;

    md_t NR_;
    int  numFullChunks; // number of 32-n chunks (full ZMM loads)
    bool hasTail16;     // true when NR % 32 == 16 (one trailing masked block)
    bool useMask_;
    bool nLoop_; // wrap the panel body in the N-panel (jc) loop

    // ── vpermt2q index vectors (epi64 selectors), shared with the intrinsic ──
    int64_t      selector1[8]    = { 0x0, 0x1, 0x8, 0x9, 0x2, 0x3, 0xA, 0xB };
    int64_t      selector1_1[8]  = { 0x4, 0x5, 0xC, 0xD, 0x6, 0x7, 0xE, 0xF };
    const md_t   selector1_off   = 0;
    const md_t   selector1_1_off = selector1_off + sizeof(selector1);
    Xbyak::Label selectors;

    Xbyak::Address get_selector(md_t off) { return ptr[rip + selectors + off]; }

    // Persistent ZMM assignments.
    static constexpr int ZMM_SEL1 = 0; // selector1 (vpermt2q indices)
    static constexpr int ZMM_SEL2 = 1; // selector1_1
    static constexpr int ZMM_ZERO = 2; // zero row for odd-K tail
    static constexpr int ZMM_A    = 3; // row k load
    static constexpr int ZMM_C    = 4; // row k+1 load
    static constexpr int ZMM_LO   = 5; // unpacklo result
    static constexpr int ZMM_HI   = 6; // unpackhi result
    static constexpr int ZMM_O0   = 7; // output block (low half)
    static constexpr int ZMM_O1   = 8; // output block (high half)

    // Opmask for the trailing 16-n masked load (low 16 bf16).
    static constexpr int MASK_TAIL16 = 1; // k1

    Xbyak::Reg64 pParams;
    Xbyak::Reg64 regSrc;
    Xbyak::Reg64 regDst;
    Xbyak::Reg64 regK;
    Xbyak::Reg64 regLdbBytes;
    Xbyak::Reg64 regDstPanelStride;
    Xbyak::Reg64 regSrcEnd;
    Xbyak::Reg64 regSrcBase;
    Xbyak::Reg64 regDstBase;
    Xbyak::Reg64 regSrcRow;
    Xbyak::Reg64 regDstRow;
    Xbyak::Reg64 regKFull;
    Xbyak::Reg64 regKr;

    dlp::jit::jitGeneratorError allocateReg();
    void initializeStackFrame(Xbyak::util::StackFrame& sf);
    void initializeParameters();
    void embedSelectors();
    void generateFullPanelLoop();
    // Emit one NR-wide panel: the K-pair loop plus the odd-K tail. Expects
    // regSrcRow / regDstRow to point at the panel start; both are advanced.
    void generatePanelBody();
    // Emit one NR-wide K-pair transform. When zeroSecondRow is true the second
    // row (k+1) is taken as zero (odd-K tail).
    void interleaveKPair(bool zeroSecondRow);
};

} // namespace amdzen::PackBcodeGenerator
