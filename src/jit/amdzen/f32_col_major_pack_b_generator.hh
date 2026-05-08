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

namespace amdzen::PackBcodeGenerator {

// Column-major F32 pack B JIT generator (AVX-512 and AVX2).
//
// Generates per-variant kernels:
//   Full kernel (useMask=false): runtime N-panel, sub-block, K loops.
//   Lt kernel (useMask=true): runtime n_sub loop with optional masked
//     sub-block, runtime K-loop.
//
// AVX-512: load simdWidth(16) cols -> 4-stage 16x16 transpose -> store.
// AVX2:    load simdWidth(8)  cols -> 3-stage 8x8  transpose  -> store.
template<utils::kernelInstrType KType>
class jitPackBF32ColMajor : public Xbyak::CodeGenerator
{
  public:
    jitPackBF32ColMajor();
    ~jitPackBF32ColMajor()                                = default;
    jitPackBF32ColMajor(jitPackBF32ColMajor&)             = delete;
    jitPackBF32ColMajor& operator=(jitPackBF32ColMajor&)  = delete;
    jitPackBF32ColMajor(jitPackBF32ColMajor&&)            = delete;
    jitPackBF32ColMajor& operator=(jitPackBF32ColMajor&&) = delete;

    dlp::jit::jitGeneratorError generateKernel(
        utils::packBGeneratorParams& params);

  private:
    using Traits  = amdzen::traits::ArchitectureTraits<KType>;
    using RegType = typename Traits::RegType;

    static constexpr int numRegs   = Traits::numRegs;
    static constexpr int RegBytes  = Traits::regBytes;
    static constexpr int simdWidth = RegBytes / sizeof(float);

    // log2(simdWidth): AVX-512 -> 4 (16 floats), AVX2 -> 3 (8 floats)
    static constexpr int simdShift = (simdWidth == 16) ? 4 : 3;

    // AVX-512 16x16: post-transpose output row r is in Zmm(storeMap16[r]).
    static constexpr int storeMap16[16] = { 0, 2, 8,  10, 1, 3, 9,  11,
                                            4, 6, 12, 14, 5, 7, 13, 15 };

    md_t NR_;
    int  numSubBlocks_;
    bool useMask_;

    Xbyak::Reg64 pParams;
    Xbyak::Reg64 regSrcPanel;
    Xbyak::Reg64 regDstPanel;
    Xbyak::Reg64 regK;
    Xbyak::Reg64 regLdbBytes;
    Xbyak::Reg64 regNrBytes;
    Xbyak::Reg64 regKFull;
    Xbyak::Reg64 regSrcSb;
    Xbyak::Reg64 regDstSb;
    Xbyak::Reg64 regKr;
    Xbyak::Reg64 regSbCount;
    Xbyak::Reg64 regColPtr;
    Xbyak::Reg64 regTmp;
    Xbyak::Reg64 regTmp2;

    // AVX-512: K-fringe mask and N-fringe store mask (opmask registers).
    Xbyak::Opmask nFringeMask;
    Xbyak::Opmask nStoreMask;

    // AVX2: mask registers stored in Ymm for vmaskmovps.
    int avx2KMaskRegIdx;
    int avx2NMaskRegIdx;

    // Register budget validation
    dlp::jit::jitGeneratorError allocateReg();

    // Top-level loop generators
    void generateFullBlockLoop();
    void generateLtBlockLoop();

    // K-loop and sub-block advance
    void emitKLoop(bool nFringe);
    void advanceSrcBySW();

    // ISA-specific initialization of mask registers
    void initMaskRegs();
    void loadNFringeMask();

    // ISA-specific K-fringe mask setup
    void emitKFringeMask();

    // ISA-specific load / transpose / store
    void emitLoadCols(bool kMasked, bool partialNCols);
    void emitStoreRows(bool kMasked, bool nMaskedStore);

    // Specialized transpose routines
    void emitTranspose16x16();
    void emitTranspose8x8();
    void emitTranspose();

    // ISA-specific helpers for emitLoadCols / emitStoreRows
    void emitLoadOne(int regIdx, bool kMasked);
    void emitZeroDataRegs();
    void emitStoreOne(int row, int srcReg, int nrBytes, bool nMasked);
};

} // namespace amdzen::PackBcodeGenerator
