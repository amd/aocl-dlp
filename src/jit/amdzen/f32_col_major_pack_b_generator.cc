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

#include "f32_col_major_pack_b_generator.hh"
#include "traits.hh"

namespace amdzen::PackBcodeGenerator {

// ──────────────────────────────────────────────────────────────────────
// Construction
// ──────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
jitPackBF32ColMajor<KType>::jitPackBF32ColMajor()
    : Xbyak::CodeGenerator(utils::JIT_KERNEL_SIZE, Xbyak::AutoGrow)
    , NR_(0)
    , numSubBlocks_(0)
    , useMask_(false)
    , avx2KMaskRegIdx(0)
    , avx2NMaskRegIdx(0)
{
}

// ──────────────────────────────────────────────────────────────────────
// ISA-specific: mask register initialization
// ──────────────────────────────────────────────────────────────────────

template<>
void
jitPackBF32ColMajor<utils::kernelInstrType::avx512_zmm_32_reg>::initMaskRegs()
{
    nFringeMask = Xbyak::Opmask(1);
    nStoreMask  = Xbyak::Opmask(2);
}

template<>
void
jitPackBF32ColMajor<utils::kernelInstrType::avx2_ymm_16_reg>::initMaskRegs()
{
    avx2KMaskRegIdx = 8;
    avx2NMaskRegIdx = 9;
}

// ──────────────────────────────────────────────────────────────────────
// ISA-specific: load N-fringe store mask from packBParams
// ──────────────────────────────────────────────────────────────────────

template<>
void
jitPackBF32ColMajor<
    utils::kernelInstrType::avx512_zmm_32_reg>::loadNFringeMask()
{
    kmovw(nStoreMask,
          ptr[pParams
              + offsetof(dlp::kernels::packBParams, nFringeMaskPerBlock)]);
}

template<>
void
jitPackBF32ColMajor<utils::kernelInstrType::avx2_ymm_16_reg>::loadNFringeMask()
{
    vmovdqu(Xbyak::Ymm(avx2NMaskRegIdx),
            ptr[pParams + offsetof(dlp::kernels::packBParams, nMaskPerBlock)]);
}

// ──────────────────────────────────────────────────────────────────────
// ISA-specific: build K-fringe mask (k_tail is in regTmp)
// ──────────────────────────────────────────────────────────────────────

template<>
void
jitPackBF32ColMajor<
    utils::kernelInstrType::avx512_zmm_32_reg>::emitKFringeMask()
{
    kmovw(nFringeMask,
          ptr[pParams + offsetof(dlp::kernels::packBParams, k_fringe_mask)]);
}

template<>
void
jitPackBF32ColMajor<utils::kernelInstrType::avx2_ymm_16_reg>::emitKFringeMask()
{
    vmovdqu(Xbyak::Ymm(avx2KMaskRegIdx),
            ptr[pParams + offsetof(dlp::kernels::packBParams, kMaskArray)]);
}

// ──────────────────────────────────────────────────────────────────────
// ISA-specific: zero all data registers before partial-N loads
// ──────────────────────────────────────────────────────────────────────

template<>
void
jitPackBF32ColMajor<
    utils::kernelInstrType::avx512_zmm_32_reg>::emitZeroDataRegs()
{
    for (int i = 0; i < simdWidth; ++i)
        vpxord(Xbyak::Zmm(i), Xbyak::Zmm(i), Xbyak::Zmm(i));
}

template<>
void
jitPackBF32ColMajor<utils::kernelInstrType::avx2_ymm_16_reg>::emitZeroDataRegs()
{
    for (int i = 0; i < simdWidth; ++i)
        vpxor(Xbyak::Ymm(i), Xbyak::Ymm(i), Xbyak::Ymm(i));
}

// ──────────────────────────────────────────────────────────────────────
// ISA-specific: load one column vector (optionally K-masked)
// ──────────────────────────────────────────────────────────────────────

template<>
void
jitPackBF32ColMajor<utils::kernelInstrType::avx512_zmm_32_reg>::emitLoadOne(
    int regIdx, bool kMasked)
{
    if (kMasked)
        vmovups(Xbyak::Zmm(regIdx) | nFringeMask | T_z, ptr[regColPtr]);
    else
        vmovups(Xbyak::Zmm(regIdx), ptr[regColPtr]);
}

template<>
void
jitPackBF32ColMajor<utils::kernelInstrType::avx2_ymm_16_reg>::emitLoadOne(
    int regIdx, bool kMasked)
{
    if (kMasked)
        vmaskmovps(Xbyak::Ymm(regIdx), Xbyak::Ymm(avx2KMaskRegIdx),
                   ptr[regColPtr]);
    else
        vmovups(Xbyak::Ymm(regIdx), ptr[regColPtr]);
}

// ──────────────────────────────────────────────────────────────────────
// ISA-specific: store one transposed row (optionally N-masked)
// ──────────────────────────────────────────────────────────────────────

template<>
void
jitPackBF32ColMajor<utils::kernelInstrType::avx512_zmm_32_reg>::emitStoreOne(
    int row, int srcReg, int nrBytes, bool nMasked)
{
    if (nMasked)
        vmovups(ptr[regColPtr + row * nrBytes] | nStoreMask,
                Xbyak::Zmm(srcReg));
    else
        vmovups(ptr[regColPtr + row * nrBytes], Xbyak::Zmm(srcReg));
}

template<>
void
jitPackBF32ColMajor<utils::kernelInstrType::avx2_ymm_16_reg>::emitStoreOne(
    int row, int srcReg, int nrBytes, bool nMasked)
{
    if (nMasked)
        vmaskmovps(ptr[regColPtr + row * nrBytes], Xbyak::Ymm(avx2NMaskRegIdx),
                   Xbyak::Ymm(srcReg));
    else
        vmovups(ptr[regColPtr + row * nrBytes], Xbyak::Ymm(srcReg));
}

// ──────────────────────────────────────────────────────────────────────
// ISA-specific: in-register transpose
// ──────────────────────────────────────────────────────────────────────

template<>
void
jitPackBF32ColMajor<
    utils::kernelInstrType::avx512_zmm_32_reg>::emitTranspose16x16()
{
    // Stage 1: vunpcklps / vunpckhps (pairs -> Zmm16..31)
    for (int i = 0; i < 16; i += 2) {
        vunpcklps(Xbyak::Zmm(16 + i), Xbyak::Zmm(i), Xbyak::Zmm(i + 1));
        vunpckhps(Xbyak::Zmm(16 + i + 1), Xbyak::Zmm(i), Xbyak::Zmm(i + 1));
    }

    // Stage 2: vunpcklpd / vunpckhpd (quads -> Zmm0..15)
    vunpcklpd(Xbyak::Zmm(0), Xbyak::Zmm(16), Xbyak::Zmm(18));
    vunpckhpd(Xbyak::Zmm(1), Xbyak::Zmm(16), Xbyak::Zmm(18));
    vunpcklpd(Xbyak::Zmm(2), Xbyak::Zmm(20), Xbyak::Zmm(22));
    vunpckhpd(Xbyak::Zmm(3), Xbyak::Zmm(20), Xbyak::Zmm(22));
    vunpcklpd(Xbyak::Zmm(4), Xbyak::Zmm(24), Xbyak::Zmm(26));
    vunpckhpd(Xbyak::Zmm(5), Xbyak::Zmm(24), Xbyak::Zmm(26));
    vunpcklpd(Xbyak::Zmm(6), Xbyak::Zmm(28), Xbyak::Zmm(30));
    vunpckhpd(Xbyak::Zmm(7), Xbyak::Zmm(28), Xbyak::Zmm(30));
    vunpcklpd(Xbyak::Zmm(8), Xbyak::Zmm(17), Xbyak::Zmm(19));
    vunpckhpd(Xbyak::Zmm(9), Xbyak::Zmm(17), Xbyak::Zmm(19));
    vunpcklpd(Xbyak::Zmm(10), Xbyak::Zmm(21), Xbyak::Zmm(23));
    vunpckhpd(Xbyak::Zmm(11), Xbyak::Zmm(21), Xbyak::Zmm(23));
    vunpcklpd(Xbyak::Zmm(12), Xbyak::Zmm(25), Xbyak::Zmm(27));
    vunpckhpd(Xbyak::Zmm(13), Xbyak::Zmm(25), Xbyak::Zmm(27));
    vunpcklpd(Xbyak::Zmm(14), Xbyak::Zmm(29), Xbyak::Zmm(31));
    vunpckhpd(Xbyak::Zmm(15), Xbyak::Zmm(29), Xbyak::Zmm(31));

    // Stage 3: vshuff32x4 0x44 / 0xEE (128-bit lane pairs -> Zmm16..31)
    static constexpr int s3a[8] = { 0, 4, 1, 5, 8, 12, 9, 13 };
    static constexpr int s3b[8] = { 2, 6, 3, 7, 10, 14, 11, 15 };
    for (int p = 0; p < 8; ++p) {
        vshuff32x4(Xbyak::Zmm(16 + 2 * p), Xbyak::Zmm(s3a[p]),
                   Xbyak::Zmm(s3b[p]), 0x44);
        vshuff32x4(Xbyak::Zmm(16 + 2 * p + 1), Xbyak::Zmm(s3a[p]),
                   Xbyak::Zmm(s3b[p]), 0xEE);
    }

    // Stage 4: vshuff32x4 0x88 / 0xDD (final placement -> Zmm0..15)
    static constexpr int s4a[8] = { 16, 20, 17, 21, 24, 28, 25, 29 };
    static constexpr int s4b[8] = { 18, 22, 19, 23, 26, 30, 27, 31 };
    for (int p = 0; p < 8; ++p) {
        vshuff32x4(Xbyak::Zmm(2 * p), Xbyak::Zmm(s4a[p]), Xbyak::Zmm(s4b[p]),
                   0x88);
        vshuff32x4(Xbyak::Zmm(2 * p + 1), Xbyak::Zmm(s4a[p]),
                   Xbyak::Zmm(s4b[p]), 0xDD);
    }
}

template<>
void
jitPackBF32ColMajor<utils::kernelInstrType::avx2_ymm_16_reg>::emitTranspose8x8()
{
    // Standard 8x8 F32 transpose using AVX2 (3 stages).
    // Input:  Ymm(0..7) = 8 columns of up to 8 floats each.
    // Output: Ymm(0..7) = 8 transposed rows.
    //
    // The transpose uses Ymm8..15 as scratch. Mask registers (K-mask in
    // Ymm8, N-mask in Ymm9) are only needed for the pre-transpose loads
    // and are not required after transpose (stores are unmasked), so no
    // save/restore is needed.

    // Stage 1: interleave 32-bit pairs (Ymm0..7 -> Ymm8..15)
    for (int i = 0; i < 8; i += 2) {
        vunpcklps(Xbyak::Ymm(8 + i), Xbyak::Ymm(i), Xbyak::Ymm(i + 1));
        vunpckhps(Xbyak::Ymm(8 + i + 1), Xbyak::Ymm(i), Xbyak::Ymm(i + 1));
    }

    // Stage 2: interleave 64-bit pairs (Ymm8..15 -> Ymm0..7)
    vunpcklpd(Xbyak::Ymm(0), Xbyak::Ymm(8), Xbyak::Ymm(10));
    vunpckhpd(Xbyak::Ymm(1), Xbyak::Ymm(8), Xbyak::Ymm(10));
    vunpcklpd(Xbyak::Ymm(2), Xbyak::Ymm(9), Xbyak::Ymm(11));
    vunpckhpd(Xbyak::Ymm(3), Xbyak::Ymm(9), Xbyak::Ymm(11));
    vunpcklpd(Xbyak::Ymm(4), Xbyak::Ymm(12), Xbyak::Ymm(14));
    vunpckhpd(Xbyak::Ymm(5), Xbyak::Ymm(12), Xbyak::Ymm(14));
    vunpcklpd(Xbyak::Ymm(6), Xbyak::Ymm(13), Xbyak::Ymm(15));
    vunpckhpd(Xbyak::Ymm(7), Xbyak::Ymm(13), Xbyak::Ymm(15));

    // Stage 3: swap 128-bit lanes (Ymm0..7 -> Ymm8..15)
    vperm2f128(Xbyak::Ymm(8), Xbyak::Ymm(0), Xbyak::Ymm(4), 0x20);
    vperm2f128(Xbyak::Ymm(9), Xbyak::Ymm(1), Xbyak::Ymm(5), 0x20);
    vperm2f128(Xbyak::Ymm(10), Xbyak::Ymm(2), Xbyak::Ymm(6), 0x20);
    vperm2f128(Xbyak::Ymm(11), Xbyak::Ymm(3), Xbyak::Ymm(7), 0x20);
    vperm2f128(Xbyak::Ymm(12), Xbyak::Ymm(0), Xbyak::Ymm(4), 0x31);
    vperm2f128(Xbyak::Ymm(13), Xbyak::Ymm(1), Xbyak::Ymm(5), 0x31);
    vperm2f128(Xbyak::Ymm(14), Xbyak::Ymm(2), Xbyak::Ymm(6), 0x31);
    vperm2f128(Xbyak::Ymm(15), Xbyak::Ymm(3), Xbyak::Ymm(7), 0x31);

    // Final placement: Ymm(8..15) -> Ymm(0..7).
    for (int r = 0; r < 8; ++r)
        vmovaps(Xbyak::Ymm(r), Xbyak::Ymm(8 + r));
}

// ──────────────────────────────────────────────────────────────────────
// Top-level entry point
// ──────────────────────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────────────────
// Register budget validation
//
// The column-major transpose operates on one simdWidth-wide sub-block
// at a time, so vector register usage is fixed (does not grow with NR):
//
//   AVX-512: Zmm(0..15) for data, Zmm(16..31) for transpose scratch.
//            Masks use opmask registers (k1, k2) — no vector conflict.
//            Requires: numRegs >= 2 * simdWidth (32 >= 32).
//
//   AVX2:    Ymm(0..7) for data, Ymm(8..15) for transpose scratch.
//            Masks in Ymm(8) and Ymm(9) — overlap with scratch,
//            saved/restored on the stack around the transpose.
//            Requires: numRegs >= 2 * simdWidth (16 >= 16),
//            and mask indices must be valid (< numRegs, >= simdWidth).
// ──────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitPackBF32ColMajor<KType>::allocateReg()
{
    if (NR_ <= 0 || (NR_ % simdWidth) != 0)
        return dlp::jit::jitGeneratorError::badKernelInfo;

    // Transpose needs 2 * simdWidth vector registers.
    if (numRegs < 2 * simdWidth)
        return dlp::jit::jitGeneratorError::badKernelInfo;

    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        if (avx2KMaskRegIdx >= numRegs || avx2KMaskRegIdx < simdWidth)
            return dlp::jit::jitGeneratorError::badKernelInfo;
        if (avx2NMaskRegIdx >= numRegs || avx2NMaskRegIdx < simdWidth)
            return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitPackBF32ColMajor<KType>::generateKernel(utils::packBGeneratorParams& params)
{
    NR_           = params.NR;
    numSubBlocks_ = NR_ / simdWidth;
    useMask_      = params.useMask;

    initMaskRegs();

    RETURN_IF_ERROR(allocateReg());

    Xbyak::util::StackFrame sf(this, 1, 12 | Xbyak::util::UseRBPAsFramePointer,
                               0);

    pParams     = sf.p[0];
    regSrcPanel = sf.t[0];
    regDstPanel = sf.t[1];
    regK        = sf.t[2];
    regLdbBytes = sf.t[3];
    regKFull    = sf.t[4];
    regSrcSb    = sf.t[5];
    regDstSb    = sf.t[6];
    regKr       = sf.t[7];
    regSbCount  = sf.t[8];
    regColPtr   = sf.t[9];
    regTmp      = sf.t[10];
    regTmp2     = sf.t[11];

    mov(regK, ptr[pParams + offsetof(dlp::kernels::packBParams, k)]);
    mov(regLdbBytes,
        ptr[pParams + offsetof(dlp::kernels::packBParams, cs_src)]);
    shl(regLdbBytes, 2);

    mov(regKFull, regK);
    and_(regKFull, ~(simdWidth - 1));

    if (useMask_)
        generateLtBlockLoop();
    else
        generateFullBlockLoop();

    mov(regTmp, NR_);
    mov(ptr[pParams + offsetof(dlp::kernels::packBParams, rs_dst)], regTmp);
    mov(regTmp, 1);
    mov(ptr[pParams + offsetof(dlp::kernels::packBParams, cs_dst)], regTmp);

    vzeroupper();
    return dlp::jit::jitGeneratorError::success;
}

// ──────────────────────────────────────────────────────────────────────
// Advance source pointer by simdWidth columns (simdWidth * ldb_bytes)
// ──────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
void
jitPackBF32ColMajor<KType>::advanceSrcBySW()
{
    mov(regKr, regLdbBytes);
    shl(regKr, simdShift);
    add(regSrcSb, regKr);
    add(regDstSb, static_cast<int>(simdWidth * sizeof(float)));
}

// ──────────────────────────────────────────────────────────────────────
// Full kernel: N-panel loop -> sub-block loop -> K-loop
// ──────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
void
jitPackBF32ColMajor<KType>::generateFullBlockLoop()
{
    Xbyak::Label l_panel_done, l_panel_loop, l_sb_loop;

    mov(regSrcPanel, ptr[pParams + offsetof(dlp::kernels::packBParams, src)]);
    mov(regDstPanel, ptr[pParams + offsetof(dlp::kernels::packBParams, dst)]);

    mov(regTmp,
        ptr[pParams
            + offsetof(dlp::kernels::packBParams, n_full_pieces_limit)]);
    imul(regTmp, regLdbBytes);
    add(regTmp, regSrcPanel);

    cmp(regSrcPanel, regTmp);
    jge(l_panel_done, T_NEAR);

    L(l_panel_loop);

    mov(regSrcSb, regSrcPanel);
    xor_(regDstSb, regDstSb);
    mov(regSbCount, numSubBlocks_);

    L(l_sb_loop);

    emitKLoop(false);
    advanceSrcBySW();

    dec(regSbCount);
    jnz(l_sb_loop, T_NEAR);

    // Advance to next NR-panel
    mov(regKr, regLdbBytes);
    imul(regKr, regKr, static_cast<int>(NR_));
    add(regSrcPanel, regKr);

    mov(regKr, regK);
    imul(regKr, regKr, static_cast<int>(NR_ * sizeof(float)));
    add(regDstPanel, regKr);

    mov(regTmp, ptr[pParams + offsetof(dlp::kernels::packBParams, src)]);
    mov(regKr, ptr[pParams
                   + offsetof(dlp::kernels::packBParams, n_full_pieces_limit)]);
    imul(regKr, regLdbBytes);
    add(regTmp, regKr);

    cmp(regSrcPanel, regTmp);
    jb(l_panel_loop, T_NEAR);

    L(l_panel_done);
}

// ──────────────────────────────────────────────────────────────────────
// Lt kernel: runtime n_sub loop + optional masked tail sub-block
// ──────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
void
jitPackBF32ColMajor<KType>::generateLtBlockLoop()
{
    mov(regSrcSb, ptr[pParams + offsetof(dlp::kernels::packBParams, src)]);
    mov(regDstPanel, ptr[pParams + offsetof(dlp::kernels::packBParams, dst)]);
    xor_(regDstSb, regDstSb);

    mov(regTmp, ptr[pParams + offsetof(dlp::kernels::packBParams, n_partial)]);

    // regSbCount = n_partial / simdWidth
    mov(regSbCount, regTmp);
    shr(regSbCount, simdShift);

    // regTmp2 = n_partial % simdWidth
    mov(regTmp2, regTmp);
    and_(regTmp2, simdWidth - 1);

    // Full sub-block loop
    {
        Xbyak::Label l_sub_done, l_sub_loop;

        test(regSbCount, regSbCount);
        jz(l_sub_done, T_NEAR);

        L(l_sub_loop);

        emitKLoop(false);
        advanceSrcBySW();

        dec(regSbCount);
        jnz(l_sub_loop, T_NEAR);

        L(l_sub_done);
    }

    // Masked tail sub-block
    {
        Xbyak::Label l_tail_done;

        test(regTmp2, regTmp2);
        jz(l_tail_done, T_NEAR);

        mov(regSbCount, regTmp2);
        emitKLoop(true);

        L(l_tail_done);
    }
}

// ──────────────────────────────────────────────────────────────────────
// K-loop for one sub-block
// ──────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
void
jitPackBF32ColMajor<KType>::emitKLoop(bool nFringe)
{
    Xbyak::Label l_kfull_done, l_kfull_loop, l_kfringe_done;

    xor_(regKr, regKr);

    test(regKFull, regKFull);
    jz(l_kfull_done, T_NEAR);

    L(l_kfull_loop);

    emitLoadCols(false, nFringe);
    emitTranspose();
    emitStoreRows(false, false);

    add(regKr, simdWidth);
    cmp(regKr, regKFull);
    jb(l_kfull_loop, T_NEAR);

    L(l_kfull_done);

    // K-fringe: partial tile of k_tail rows
    mov(regTmp, regK);
    and_(regTmp, simdWidth - 1);
    test(regTmp, regTmp);
    jz(l_kfringe_done, T_NEAR);

    emitKFringeMask();

    mov(regKr, regKFull);

    emitLoadCols(true, nFringe);
    emitTranspose();
    emitStoreRows(true, false);

    L(l_kfringe_done);
}

// ──────────────────────────────────────────────────────────────────────
// Transpose dispatch
// ──────────────────────────────────────────────────────────────────────

template<>
void
jitPackBF32ColMajor<utils::kernelInstrType::avx512_zmm_32_reg>::emitTranspose()
{
    emitTranspose16x16();
}

template<>
void
jitPackBF32ColMajor<utils::kernelInstrType::avx2_ymm_16_reg>::emitTranspose()
{
    emitTranspose8x8();
}

// ──────────────────────────────────────────────────────────────────────
// Load up to simdWidth columns into vector registers
// ──────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
void
jitPackBF32ColMajor<KType>::emitLoadCols(bool kMasked, bool partialNCols)
{
    lea(regColPtr, ptr[regSrcSb + regKr * 4]);

    if (partialNCols) {
        emitZeroDataRegs();

        for (int i = 0; i < simdWidth; ++i) {
            Xbyak::Label l_skip;
            cmp(regSbCount, i + 1);
            jl(l_skip, T_NEAR);

            emitLoadOne(i, kMasked);

            L(l_skip);
            if (i < simdWidth - 1)
                add(regColPtr, regLdbBytes);
        }
    } else {
        for (int i = 0; i < simdWidth; ++i) {
            emitLoadOne(i, kMasked);

            if (i < simdWidth - 1)
                add(regColPtr, regLdbBytes);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Store simdWidth transposed rows to output buffer
// ──────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
void
jitPackBF32ColMajor<KType>::emitStoreRows(bool kMasked, bool nMaskedStore)
{
    int nrBytes = static_cast<int>(NR_ * sizeof(float));

    mov(regColPtr, regKr);
    imul(regColPtr, regColPtr, nrBytes);
    add(regColPtr, regDstPanel);
    add(regColPtr, regDstSb);

    if (!kMasked) {
        for (int row = 0; row < simdWidth; ++row) {
            int srcReg = (simdWidth == 16) ? storeMap16[row] : row;
            emitStoreOne(row, srcReg, nrBytes, nMaskedStore);
        }
    } else {
        mov(regTmp, regK);
        and_(regTmp, simdWidth - 1);

        for (int row = 0; row < simdWidth; ++row) {
            Xbyak::Label l_skip;
            cmp(regTmp, row + 1);
            jb(l_skip, T_NEAR);

            int srcReg = (simdWidth == 16) ? storeMap16[row] : row;
            emitStoreOne(row, srcReg, nrBytes, nMaskedStore);

            L(l_skip);
        }
    }
}

} // namespace amdzen::PackBcodeGenerator

template class amdzen::PackBcodeGenerator::jitPackBF32ColMajor<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
template class amdzen::PackBcodeGenerator::jitPackBF32ColMajor<
    amdzen::utils::kernelInstrType::avx2_ymm_16_reg>;
