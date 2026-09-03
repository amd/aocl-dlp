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

#include "int8_col_major_pack_b_generator.hh"
#include "traits.hh"

namespace amdzen::PackBcodeGenerator {

template<utils::kernelInstrType KType>
jitPackBINT8ColMajor<KType>::jitPackBINT8ColMajor()
    : Xbyak::CodeGenerator(utils::JIT_KERNEL_SIZE, Xbyak::AutoGrow)
    , NR_(0)
    , numSubBlocks_(0)
    , useMask_(false)
    , nLoop_(true)
    , accColSum_(false)
{
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitPackBINT8ColMajor<KType>::allocateReg()
{
    if constexpr (KType != utils::kernelInstrType::avx512_zmm_32_reg) {
        return dlp::jit::jitGeneratorError::notSupported;
    }
    if (NR_ <= 0 || (NR_ % BLOCK_N) != 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
    if (numRegs < kNumVecRegs) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitPackBINT8ColMajor<KType>::initializeStackFrame(Xbyak::util::StackFrame& sf)
{
    pParams     = sf.p[0];
    regK        = sf.t[0];
    regKFull    = sf.t[1];
    regLdbBytes = sf.t[2];
    regSrcEnd   = sf.t[3];
    regSrcPanel = sf.t[4];
    regDstPanel = sf.t[5];
    regSrcSb    = sf.t[6];
    regDstSb    = sf.t[7];
    regKr       = sf.t[8];
    regSbCount  = sf.t[9];
    regColPtr   = sf.t[10];
    regTmp      = sf.t[11];
    regColSum   = rax; // not a StackFrame temp; see header
}

template<utils::kernelInstrType KType>
void
jitPackBINT8ColMajor<KType>::initializeParameters()
{
    mov(regK, ptr[pParams + offsetof(dlp::kernels::packBParams, k)]);

    // Column-to-column source stride (bytes): one n-step is cs_src elements.
    mov(regLdbBytes,
        ptr[pParams + offsetof(dlp::kernels::packBParams, cs_src)]);
    // ELEM_BYTES == 1, no shift.

    if (nLoop_) {
        mov(regSrcEnd,
            ptr[pParams
                + offsetof(dlp::kernels::packBParams, n_full_pieces_limit)]);
        imul(regSrcEnd, regLdbBytes);
        add(regSrcEnd, ptr[pParams + offsetof(dlp::kernels::packBParams, src)]);
    }

    // K rounded down to a whole number of 64-K tiles. The remainder (K % 64,
    // including any K%4 in {1,2,3}) is handled by the masked K-fringe tile.
    mov(regKFull, regK);
    and_(regKFull, ~(numElemsPerReg - 1));

    if (accColSum_) {
        mov(regColSum,
            ptr[pParams + offsetof(dlp::kernels::packBParams, col_sum)]);
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitPackBINT8ColMajor<KType>::generateKernel(utils::packBGeneratorParams& params)
{
    NR_           = params.NR;
    useMask_      = params.useMask;
    nLoop_        = params.nLoop;
    numSubBlocks_ = static_cast<int>(NR_ / BLOCK_N);
    accColSum_    = params.accColSum;

    RETURN_IF_ERROR(allocateReg());

    Xbyak::util::StackFrame stackFrame(
        this, 1, 12 | Xbyak::util::UseRBPAsFramePointer, 0);
    initializeStackFrame(stackFrame);

    // Preserve callee-saved xmm6-15 across the kernel call on Windows x64.
    // No-op on SysV (Linux). This 16x16 transpose packer uses Zmm(0..15) as
    // scratch, so the xmm halves of regs 6..15 must be saved/restored.
    utils::winAbiVectorGuard winAbiGuard(this);

    initializeParameters();

    if (useMask_)
        generateLtBlockLoop();
    else
        generateFullBlockLoop();

    // Packed strides for this kernel width (the orchestrator overwrites these
    // with the full-NR values after the cascade).
    //
    // cs_dst is 64, not NR_/4: one VNNI ZMM is 16 n * 4 k bytes. The nr64
    // col-major intrinsic writes *cs_p = NR only because NR is hardcoded 64.
    mov(regTmp, static_cast<int>(NR_ * K_FACTOR));
    mov(ptr[pParams + offsetof(dlp::kernels::packBParams, rs_dst)], regTmp);
    mov(regTmp, static_cast<int>(RegBytes));
    mov(ptr[pParams + offsetof(dlp::kernels::packBParams, cs_dst)], regTmp);

    vzeroupper();
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitPackBINT8ColMajor<KType>::generateFullBlockLoop()
{
    if (!nLoop_) {
        mov(regSrcPanel,
            ptr[pParams + offsetof(dlp::kernels::packBParams, src)]);
        mov(regDstPanel,
            ptr[pParams + offsetof(dlp::kernels::packBParams, dst)]);
        emitKLoop(/*partialNCols=*/false);
        return;
    }

    Xbyak::Label l_jc_done, l_jc_loop;

    mov(regSrcPanel, ptr[pParams + offsetof(dlp::kernels::packBParams, src)]);
    mov(regDstPanel, ptr[pParams + offsetof(dlp::kernels::packBParams, dst)]);

    cmp(regSrcPanel, regSrcEnd);
    jge(l_jc_done, T_NEAR);

    L(l_jc_loop);

    emitKLoop(/*partialNCols=*/false);

    mov(regTmp, regLdbBytes);
    imul(regTmp, regTmp, static_cast<int>(NR_));
    add(regSrcPanel, regTmp);

    // Packed panel = KC_updated * NR bytes, KC_updated = (K+3)&~3.
    mov(regTmp, regK);
    add(regTmp, K_FACTOR - 1);
    and_(regTmp, -K_FACTOR);
    imul(regTmp, regTmp, static_cast<int>(NR_ * ELEM_BYTES));
    add(regDstPanel, regTmp);
    if (accColSum_) {
        advanceColSum();
    }

    cmp(regSrcPanel, regSrcEnd);
    jb(l_jc_loop, T_NEAR);

    L(l_jc_done);
}

template<utils::kernelInstrType KType>
void
jitPackBINT8ColMajor<KType>::generateLtBlockLoop()
{
    mov(regSrcPanel, ptr[pParams + offsetof(dlp::kernels::packBParams, src)]);
    mov(regDstPanel, ptr[pParams + offsetof(dlp::kernels::packBParams, dst)]);
    mov(regSbCount,
        ptr[pParams + offsetof(dlp::kernels::packBParams, n_partial)]);

    emitKLoop(/*partialNCols=*/true);
}

template<utils::kernelInstrType KType>
void
jitPackBINT8ColMajor<KType>::emitKLoop(bool partialNCols)
{
    Xbyak::Label l_kfull_done, l_kfull_loop, l_kfringe_done;

    xor_(regKr, regKr);

    test(regKFull, regKFull);
    jz(l_kfull_done, T_NEAR);

    L(l_kfull_loop);

    emitTileSweep(/*kMasked=*/false, partialNCols);

    add(regKr, numElemsPerReg);
    cmp(regKr, regKFull);
    jb(l_kfull_loop, T_NEAR);

    L(l_kfull_done);

    // K-fringe: trailing K % 64 elements (covers K%4 via the zero-mask).
    mov(regTmp, regK);
    and_(regTmp, numElemsPerReg - 1);
    test(regTmp, regTmp);
    jz(l_kfringe_done, T_NEAR);

    // k1 = (1 << k_tail) - 1 as a 64-bit epi8 mask. k_fringe_mask in
    // packBParams is uint32_t (enough for BF16's 32 epi16 lanes, not for 64
    // int8 lanes), so the mask is built here with BMI2 bzhi.
    mov(regColPtr, -1);
    bzhi(regColPtr, regColPtr, regTmp);
    kmovq(Xbyak::Opmask(MASK_KFRINGE), regColPtr);

    mov(regKr, regKFull);

    emitTileSweep(/*kMasked=*/true, partialNCols);

    L(l_kfringe_done);
}

template<utils::kernelInstrType KType>
void
jitPackBINT8ColMajor<KType>::emitTileSweep(bool kMasked, bool partialNCols)
{
    lea(regSrcSb, ptr[regSrcPanel + regKr * ELEM_BYTES]);

    // Packed base for K-quad (regKr/4) of sub-block 0: (regKr/4) * (NR*4)
    // = regKr * NR bytes from the panel start.
    mov(regDstSb, regKr);
    imul(regDstSb, regDstSb, static_cast<int>(NR_));
    add(regDstSb, regDstPanel);

    if (partialNCols) {
        emitLoadCols(kMasked, /*partialNCols=*/true);
        if (accColSum_) {
            emitColSumFromLoadedCols(/*partialNCols=*/true);
        }
        transpose::emitTranspose16x16(*this);
        emitStoreRows(kMasked);
        return;
    }

    Xbyak::Label l_sb_loop;
    mov(regSbCount, numSubBlocks_);

    L(l_sb_loop);

    emitLoadCols(kMasked, /*partialNCols=*/false);
    if (accColSum_) {
        emitColSumFromLoadedCols(/*partialNCols=*/false);
    }
    transpose::emitTranspose16x16(*this);
    emitStoreRows(kMasked);

    mov(regTmp, regLdbBytes);
    imul(regTmp, regTmp, BLOCK_N);
    add(regSrcSb, regTmp);
    add(regDstSb, static_cast<int>(BLOCK_N * K_FACTOR * ELEM_BYTES));

    dec(regSbCount);
    jnz(l_sb_loop, T_NEAR);
}

template<utils::kernelInstrType KType>
void
jitPackBINT8ColMajor<KType>::emitLoadCols(bool kMasked, bool partialNCols)
{
    using Xbyak::Zmm;

    mov(regColPtr, regSrcSb);

    auto loadOne = [&](int i) {
        if (kMasked)
            vmovdqu8(Zmm(i) | Xbyak::Opmask(MASK_KFRINGE) | T_z,
                     ptr[regColPtr]);
        else
            vmovdqu64(Zmm(i), ptr[regColPtr]);
    };

    if (partialNCols) {
        for (int i = 0; i < BLOCK_N; ++i)
            vpxorq(Zmm(i), Zmm(i), Zmm(i));

        for (int i = 0; i < BLOCK_N; ++i) {
            Xbyak::Label l_skip;
            cmp(regSbCount, i + 1);
            jl(l_skip, T_NEAR);

            loadOne(i);

            L(l_skip);
            if (i < BLOCK_N - 1)
                add(regColPtr, regLdbBytes);
        }
    } else {
        for (int i = 0; i < BLOCK_N; ++i) {
            loadOne(i);
            if (i < BLOCK_N - 1)
                add(regColPtr, regLdbBytes);
        }
    }
}

template<utils::kernelInstrType KType>
void
jitPackBINT8ColMajor<KType>::emitStoreRows(bool kMasked)
{
    using Xbyak::Zmm;

    // rs_dst in bytes = NR * 4 (K-quad stride).
    const int nrBytes = static_cast<int>(NR_ * K_FACTOR * ELEM_BYTES);

    if (!kMasked) {
        for (int r = 0; r < BLOCK_N; ++r) {
            const int srcReg = transpose::storeMap16[r];
            vmovdqu64(ptr[regDstSb + r * nrBytes], Zmm(srcReg));
        }
        return;
    }

    // K-fringe: store ceil(k_tail / 4) K-quad rows, k_tail = K % 64.
    mov(regTmp, regK);
    and_(regTmp, numElemsPerReg - 1);
    add(regTmp, K_FACTOR - 1);
    shr(regTmp, 2); // ceil(k_tail / 4)

    for (int r = 0; r < BLOCK_N; ++r) {
        Xbyak::Label l_skip;
        cmp(regTmp, r + 1);
        jb(l_skip, T_NEAR);

        const int srcReg = transpose::storeMap16[r];
        vmovdqu64(ptr[regDstSb + r * nrBytes], Zmm(srcReg));

        L(l_skip);
    }
}

template<utils::kernelInstrType KType>
void
jitPackBINT8ColMajor<KType>::emitHsumColToMem(int colZmm, int byteOff)
{
    using Xbyak::Xmm;
    using Xbyak::Ymm;
    using Xbyak::Zmm;

    // Signed horizontal sum of 64 int8s (vpsadbw is unsigned). Four
    // vpmovsxbd of 16-byte lanes, then a dword reduction. Scale <<7 (=*128)
    // matches DLP_INT8_S8_COL_SUM_SCALE.
    vpmovsxbd(Zmm(ZMM_HSUM0), Xmm(colZmm));
    vextracti32x4(Xmm(ZMM_HSUM0 + 1), Zmm(colZmm), 1);
    vpmovsxbd(Zmm(ZMM_HSUM0 + 1), Xmm(ZMM_HSUM0 + 1));
    vextracti32x4(Xmm(ZMM_HSUM0 + 2), Zmm(colZmm), 2);
    vpmovsxbd(Zmm(ZMM_HSUM0 + 2), Xmm(ZMM_HSUM0 + 2));
    vextracti32x4(Xmm(ZMM_HSUM0 + 3), Zmm(colZmm), 3);
    vpmovsxbd(Zmm(ZMM_HSUM0 + 3), Xmm(ZMM_HSUM0 + 3));
    vpaddd(Zmm(ZMM_HSUM0), Zmm(ZMM_HSUM0), Zmm(ZMM_HSUM0 + 1));
    vpaddd(Zmm(ZMM_HSUM0 + 2), Zmm(ZMM_HSUM0 + 2), Zmm(ZMM_HSUM0 + 3));
    vpaddd(Zmm(ZMM_HSUM0), Zmm(ZMM_HSUM0), Zmm(ZMM_HSUM0 + 2));
    vextracti32x8(Ymm(ZMM_HSUM0 + 1), Zmm(ZMM_HSUM0), 1);
    vpaddd(Ymm(ZMM_HSUM0), Ymm(ZMM_HSUM0), Ymm(ZMM_HSUM0 + 1));
    vextracti32x4(Xmm(ZMM_HSUM0 + 1), Ymm(ZMM_HSUM0), 1);
    vpaddd(Xmm(ZMM_HSUM0), Xmm(ZMM_HSUM0), Xmm(ZMM_HSUM0 + 1));
    // vphaddd has no EVEX form, so xmm16+ cannot use it. Shuffle-add is EVEX.
    vpshufd(Xmm(ZMM_HSUM0 + 1), Xmm(ZMM_HSUM0), 0x4E);
    vpaddd(Xmm(ZMM_HSUM0), Xmm(ZMM_HSUM0), Xmm(ZMM_HSUM0 + 1));
    vpshufd(Xmm(ZMM_HSUM0 + 1), Xmm(ZMM_HSUM0), 0xB1);
    vpaddd(Xmm(ZMM_HSUM0), Xmm(ZMM_HSUM0), Xmm(ZMM_HSUM0 + 1));
    vpextrd(regTmp.cvt32(), Xmm(ZMM_HSUM0), 0);
    shl(regTmp.cvt32(), COL_SUM_SHIFT);
    add(dword[regColPtr + byteOff], regTmp.cvt32());
}

template<utils::kernelInstrType KType>
void
jitPackBINT8ColMajor<KType>::emitColSumFromLoadedCols(bool partialNCols)
{
    if (partialNCols) {
        mov(regColPtr, regColSum);
        for (int i = 0; i < BLOCK_N; ++i) {
            Xbyak::Label l_col_skip;
            cmp(regSbCount, i + 1);
            jl(l_col_skip, T_NEAR);
            emitHsumColToMem(i, i * static_cast<int>(sizeof(int32_t)));
            L(l_col_skip);
        }
    } else {
        // Sub-block index = numSubBlocks_ - remaining; *64 bytes of int32s.
        mov(regColPtr, numSubBlocks_);
        sub(regColPtr, regSbCount);
        shl(regColPtr, 6);
        add(regColPtr, regColSum);
        for (int i = 0; i < BLOCK_N; ++i) {
            emitHsumColToMem(i, i * static_cast<int>(sizeof(int32_t)));
        }
    }
}

template<utils::kernelInstrType KType>
void
jitPackBINT8ColMajor<KType>::advanceColSum()
{
    add(regColSum, static_cast<int>(NR_ * static_cast<md_t>(sizeof(int32_t))));
}

} // namespace amdzen::PackBcodeGenerator

template class amdzen::PackBcodeGenerator::jitPackBINT8ColMajor<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
