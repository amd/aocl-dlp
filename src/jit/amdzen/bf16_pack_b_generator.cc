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

#include "bf16_pack_b_generator.hh"
#include "traits.hh"

namespace amdzen::PackBcodeGenerator {

template<utils::kernelInstrType KType>
jitPackBBF16<KType>::jitPackBBF16()
    : Xbyak::CodeGenerator(utils::JIT_KERNEL_SIZE, Xbyak::AutoGrow)
    , NR_(0)
    , numFullChunks(0)
    , hasTail16(false)
    , useMask_(false)
    , nLoop_(true)
{
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitPackBBF16<KType>::generateKernel(utils::packBGeneratorParams& params)
{
    NR_      = params.NR;
    useMask_ = params.useMask;
    nLoop_   = params.nLoop;
    // A full chunk spans one SIMD register (numElemsPerReg bf16 lanes = two
    // 16-lane output blocks); a trailing 16-lane block remains when NR_ is an
    // odd multiple of BLOCK_N.
    numFullChunks = static_cast<int>(NR_ / numElemsPerReg);
    hasTail16     = ((NR_ % numElemsPerReg) == BLOCK_N);

    RETURN_IF_ERROR(allocateReg());

    Xbyak::util::StackFrame stackFrame(
        this, 1, 12 | Xbyak::util::UseRBPAsFramePointer, 0);
    initializeStackFrame(stackFrame);
    initializeParameters();

    // Load the vpermt2q index vectors and prepare the zero / tail-mask state.
    vmovdqu64(Xbyak::Zmm(ZMM_SEL1), get_selector(selector1_off));
    vmovdqu64(Xbyak::Zmm(ZMM_SEL2), get_selector(selector1_1_off));
    vpxorq(Xbyak::Zmm(ZMM_ZERO), Xbyak::Zmm(ZMM_ZERO), Xbyak::Zmm(ZMM_ZERO));
    if (hasTail16) {
        // Low 16 of 32 epi16 lanes -> upper 16 stay zero (kmovw writes 16
        // bits). For the lt16 fringe kernel the active-lane count is a runtime
        // value, so the opmask is taken from packBParams; otherwise the whole
        // 16-wide block is live (0xFFFF).
        if (useMask_) {
            mov(regKr.cvt32(), ptr[pParams
                                   + offsetof(dlp::kernels::packBParams,
                                              nFringeMaskPerBlock)]);
        } else {
            mov(regKr.cvt32(), 0xFFFF);
        }
        kmovw(Xbyak::Opmask(MASK_TAIL16), regKr.cvt32());
    }

    generateFullPanelLoop();

    // Packed strides for this kernel width (the orchestrator overwrites these
    // with the full-NR values after the cascade; kept for parity with the F32
    // generator and standalone use).
    mov(regKr, static_cast<int>(NR_ * 2));
    mov(ptr[pParams + offsetof(dlp::kernels::packBParams, rs_dst)], regKr);
    mov(regKr, static_cast<int>(NR_ / 2));
    mov(ptr[pParams + offsetof(dlp::kernels::packBParams, cs_dst)], regKr);

    vzeroupper();
    embedSelectors();
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitPackBBF16<KType>::allocateReg()
{
    // BF16 pack-B is AVX-512 (ZMM) only and packs in 16-wide blocks.
    if constexpr (KType != utils::kernelInstrType::avx512_zmm_32_reg) {
        return dlp::jit::jitGeneratorError::notSupported;
    }
    if (NR_ <= 0 || (NR_ % BLOCK_N) != 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitPackBBF16<KType>::initializeStackFrame(Xbyak::util::StackFrame& sf)
{
    pParams           = sf.p[0];
    regSrc            = sf.t[0];
    regDst            = sf.t[1];
    regK              = sf.t[2];
    regLdbBytes       = sf.t[3];
    regDstPanelStride = sf.t[4];
    regSrcEnd         = sf.t[5];
    regSrcBase        = sf.t[6];
    regDstBase        = sf.t[7];
    regSrcRow         = sf.t[8];
    regDstRow         = sf.t[9];
    regKFull          = sf.t[10];
    regKr             = sf.t[11];
}

template<utils::kernelInstrType KType>
void
jitPackBBF16<KType>::initializeParameters()
{
    mov(regSrc, ptr[pParams + offsetof(dlp::kernels::packBParams, src)]);
    mov(regDst, ptr[pParams + offsetof(dlp::kernels::packBParams, dst)]);
    mov(regK, ptr[pParams + offsetof(dlp::kernels::packBParams, k)]);

    // Source row stride (bytes): one K-row step is rs_src bf16 elements.
    mov(regLdbBytes,
        ptr[pParams + offsetof(dlp::kernels::packBParams, rs_src)]);
    shl(regLdbBytes, 1); // * sizeof(bf16)

    // The packed n-panel stride and the source-region end are only consumed by
    // the N-panel (jc) loop; single-panel fringe kernels skip both.
    if (nLoop_) {
        // Packed n-panel stride (bytes) = KC_updated * NR * sizeof(bf16), where
        // KC_updated rounds K up to an even number of K-rows (BF16 packs in
        // pairs).
        mov(regDstPanelStride, regK);
        add(regDstPanelStride, 1);
        and_(regDstPanelStride, -2); // clear bit 0 -> round up to even
        imul(regDstPanelStride, regDstPanelStride,
             static_cast<int>(NR_ * ELEM_BYTES));

        // End of the full-NR source region (exclusive), in bytes.
        mov(regSrcEnd,
            ptr[pParams
                + offsetof(dlp::kernels::packBParams, n_full_pieces_limit)]);
        shl(regSrcEnd, 1); // * sizeof(bf16)
        add(regSrcEnd, regSrc);
    }

    // K rounded down to a whole number of K-pairs.
    mov(regKFull, regK);
    and_(regKFull, -2); // clear bit 0 -> round down to even
}

template<utils::kernelInstrType KType>
void
jitPackBBF16<KType>::generatePanelBody()
{
    Xbyak::Label l_kr_loop, l_kr_done, l_no_ktail;

    // ── K-pair loop: process two source rows per iteration ──
    xor_(regKr, regKr);
    cmp(regKr, regKFull);
    jge(l_kr_done, T_NEAR);

    L(l_kr_loop);

    interleaveKPair(false);

    lea(regSrcRow, ptr[regSrcRow + regLdbBytes * 2]); // advance two source rows
    add(regDstRow, static_cast<int>(NR_ * 2 * ELEM_BYTES)); // rs_dst bytes
    add(regKr, 2);
    cmp(regKr, regKFull);
    jb(l_kr_loop, T_NEAR);

    L(l_kr_done);

    // ── Odd-K tail: one remaining source row, second row zero-filled ──
    test(regK, 1);
    jz(l_no_ktail, T_NEAR);
    interleaveKPair(true);
    L(l_no_ktail);
}

template<utils::kernelInstrType KType>
void
jitPackBBF16<KType>::generateFullPanelLoop()
{
    // Single-panel fringe kernel: pack one NR-wide panel starting at src/dst,
    // with no N-panel loop scaffolding.
    if (!nLoop_) {
        mov(regSrcRow, regSrc);
        mov(regDstRow, regDst);
        generatePanelBody();
        return;
    }

    // Main kernel: iterate every full NR panel in [src, srcEnd).
    Xbyak::Label l_jc_done, l_jc_loop;

    mov(regSrcBase, regSrc);
    mov(regDstBase, regDst);

    cmp(regSrcBase, regSrcEnd);
    jge(l_jc_done, T_NEAR);

    L(l_jc_loop);

    mov(regSrcRow, regSrcBase);
    mov(regDstRow, regDstBase);

    generatePanelBody();

    add(regSrcBase, static_cast<int>(NR_ * ELEM_BYTES)); // next n-panel
    add(regDstBase, regDstPanelStride);

    cmp(regSrcBase, regSrcEnd);
    jb(l_jc_loop, T_NEAR);

    L(l_jc_done);
}

// Emit one NR-wide K-pair transform: interleave rows k and k+1 into NR/16
// contiguous 16-wide ZMM blocks. Full 32-n chunks use full ZMM loads and emit
// two blocks each; a trailing 16-n block (when NR % 32 == 16) uses 16-wide
// masked loads and emits one block.
template<utils::kernelInstrType KType>
void
jitPackBBF16<KType>::interleaveKPair(bool zeroSecondRow)
{
    using Xbyak::Zmm;

    Zmm zero = Zmm(ZMM_ZERO);
    Zmm a    = Zmm(ZMM_A);
    Zmm c    = Zmm(ZMM_C);
    Zmm lo   = Zmm(ZMM_LO);
    Zmm hi   = Zmm(ZMM_HI);
    Zmm o0   = Zmm(ZMM_O0);
    Zmm o1   = Zmm(ZMM_O1);
    Zmm sel1 = Zmm(ZMM_SEL1);
    Zmm sel2 = Zmm(ZMM_SEL2);

    for (int ch = 0; ch < numFullChunks; ++ch) {
        int srcOff = ch * RegBytes;     // 64-byte (32 bf16) chunk
        int dstOff = ch * 2 * RegBytes; // two output blocks per chunk

        vmovdqu64(a, ptr[regSrcRow + srcOff]);
        Zmm cReg = zero;
        if (!zeroSecondRow) {
            vmovdqu64(c, ptr[regSrcRow + regLdbBytes + srcOff]);
            cReg = c;
        }

        vpunpcklwd(lo, a, cReg);
        vpunpckhwd(hi, a, cReg);

        vmovdqa64(o0, lo);
        vpermt2q(o0, sel1, hi);
        vmovdqa64(o1, lo);
        vpermt2q(o1, sel2, hi);

        vmovdqu64(ptr[regDstRow + dstOff], o0);
        vmovdqu64(ptr[regDstRow + dstOff + RegBytes], o1);
    }

    if (hasTail16) {
        int srcOff = numFullChunks * RegBytes; // (NR-16) bf16
        int dstOff = (static_cast<int>(NR_ / BLOCK_N) - 1) * RegBytes;

        vmovdqu16(a | Xbyak::Opmask(MASK_TAIL16) | T_z,
                  ptr[regSrcRow + srcOff]);
        Zmm cReg = zero;
        if (!zeroSecondRow) {
            vmovdqu16(c | Xbyak::Opmask(MASK_TAIL16) | T_z,
                      ptr[regSrcRow + regLdbBytes + srcOff]);
            cReg = c;
        }

        vpunpcklwd(lo, a, cReg);
        vpunpckhwd(hi, a, cReg);

        vmovdqa64(o0, lo);
        vpermt2q(o0, sel1, hi);

        vmovdqu64(ptr[regDstRow + dstOff], o0);
    }
}

template<utils::kernelInstrType KType>
void
jitPackBBF16<KType>::embedSelectors()
{
    Xbyak::Label selectorsEnd;
    jmp(selectorsEnd, T_NEAR);
    {
        size_t remain = getSize() % 64;
        if (remain)
            nop(64 - remain);
    }
    L(selectors);
    db(reinterpret_cast<uint8_t*>(selector1), sizeof(selector1));
    db(reinterpret_cast<uint8_t*>(selector1_1), sizeof(selector1_1));
    L(selectorsEnd);
}

} // namespace amdzen::PackBcodeGenerator

template class amdzen::PackBcodeGenerator::jitPackBBF16<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
