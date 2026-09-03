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

#include "int8_pack_b_generator.hh"
#include "traits.hh"

namespace amdzen::PackBcodeGenerator {

template<utils::kernelInstrType KType>
jitPackBINT8<KType>::jitPackBINT8()
    : Xbyak::CodeGenerator(utils::JIT_KERNEL_SIZE, Xbyak::AutoGrow)
    , NR_(0)
    , numBlocks(0)
    , useMask_(false)
    , nLoop_(true)
    , useZmm64_(false)
    , accColSum_(false)
{
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitPackBINT8<KType>::generateKernel(utils::packBGeneratorParams& params)
{
    NR_        = params.NR;
    useMask_   = params.useMask;
    nLoop_     = params.nLoop;
    numBlocks  = static_cast<int>(NR_ / BLOCK_N);
    useZmm64_  = (!useMask_) && (numBlocks >= 4);
    accColSum_ = params.accColSum;

    RETURN_IF_ERROR(allocateReg());

    // 1 param + 13 temps (14 GPRs). No UseRBPAsFramePointer so the 13th temp
    // (3*ldb) is available; matches the INT8 GEMM packer's StackFrame budget.
    Xbyak::util::StackFrame stackFrame(this, 1, 13, 0);
    initializeStackFrame(stackFrame);

    // Preserve callee-saved xmm6-15 across the kernel call on Windows x64.
    // No-op on SysV (Linux). Unpack uses XMM_CDLO(6); col-sum accumulators
    // start at zmm8 and scale with the generated panel width.
    const int winHi = accColSum_ ? (ZMM_ACC0 + numBlocks - 1) : XMM_CDLO;
    utils::winAbiVectorGuard winAbiGuard(this, XMM_CDLO, winHi);

    initializeParameters();

    vpxorq(Xbyak::Xmm(XMM_ZERO), Xbyak::Xmm(XMM_ZERO), Xbyak::Xmm(XMM_ZERO));
    if (useZmm64_) {
        vmovdqu64(Xbyak::Zmm(ZMM_SEL1), get_selector(selector1_off));
        vmovdqu64(Xbyak::Zmm(ZMM_SEL1_1), get_selector(selector1_1_off));
        vmovdqu64(Xbyak::Zmm(ZMM_SEL2), get_selector(selector2_off));
        vmovdqu64(Xbyak::Zmm(ZMM_SEL2_1), get_selector(selector2_1_off));
    }
    if (useMask_) {
        // Active-lane count is a runtime value (n%16); opmask from packBParams.
        mov(regKr.cvt32(),
            ptr[pParams
                + offsetof(dlp::kernels::packBParams, nFringeMaskPerBlock)]);
        kmovw(Xbyak::Opmask(MASK_LT16), regKr.cvt32());
    }

    generateFullPanelLoop();

    // Packed strides for this kernel width (the orchestrator overwrites these
    // with the full-NR values after the cascade).
    //
    // cs_dst is 64, not NR_/4: one VNNI ZMM is 16 n * 4 k bytes, and that is
    // the GEMM n-step between 16-n blocks. The nr64 intrinsic writes
    // *cs_p = NR only because NR is hardcoded 64 there. rs_dst is NR * 4.
    mov(regKr, static_cast<int>(NR_ * K_FACTOR));
    mov(ptr[pParams + offsetof(dlp::kernels::packBParams, rs_dst)], regKr);
    mov(regKr, static_cast<int>(RegBytes));
    mov(ptr[pParams + offsetof(dlp::kernels::packBParams, cs_dst)], regKr);

    vzeroupper();
    if (useZmm64_) {
        embedSelectors();
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitPackBINT8<KType>::allocateReg()
{
    if constexpr (KType != utils::kernelInstrType::avx512_zmm_32_reg) {
        return dlp::jit::jitGeneratorError::notSupported;
    }
    constexpr md_t maxNR = (ZMM_SEL1 - ZMM_ACC0) * BLOCK_N;
    if (NR_ <= 0 || (accColSum_ && NR_ > maxNR) || (NR_ % BLOCK_N) != 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
    // The masked kernel is the lt16 panel only (orchestrator index 0).
    if (useMask_ && NR_ != BLOCK_N) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitPackBINT8<KType>::initializeStackFrame(Xbyak::util::StackFrame& sf)
{
    pParams           = sf.p[0];
    regSrc            = sf.t[0];
    regDst            = sf.t[1];
    regK              = sf.t[2];
    regLdbBytes       = sf.t[3];
    regLdb3           = sf.t[4];
    regDstPanelStride = sf.t[5];
    regSrcEnd         = sf.t[6];
    regSrcBase        = sf.t[7];
    regDstBase        = sf.t[8];
    regSrcRow         = sf.t[9];
    regDstRow         = sf.t[10];
    regKFull          = sf.t[11];
    regKr             = sf.t[12];
    regColSum         = rax; // not a StackFrame temp; see header
}

template<utils::kernelInstrType KType>
void
jitPackBINT8<KType>::initializeParameters()
{
    mov(regSrc, ptr[pParams + offsetof(dlp::kernels::packBParams, src)]);
    mov(regDst, ptr[pParams + offsetof(dlp::kernels::packBParams, dst)]);
    mov(regK, ptr[pParams + offsetof(dlp::kernels::packBParams, k)]);

    // Source row stride (bytes): one K-row step is rs_src int8 elements.
    mov(regLdbBytes,
        ptr[pParams + offsetof(dlp::kernels::packBParams, rs_src)]);
    // ELEM_BYTES == 1, no shift.
    lea(regLdb3, ptr[regLdbBytes + regLdbBytes * 2]); // 3 * ldb

    if (nLoop_) {
        // Packed n-panel stride (bytes) = KC_updated * NR, where KC_updated
        // rounds K up to a multiple of 4 (INT8 packs in K-quads).
        mov(regDstPanelStride, regK);
        add(regDstPanelStride, K_FACTOR - 1);
        and_(regDstPanelStride, -K_FACTOR);
        imul(regDstPanelStride, regDstPanelStride,
             static_cast<int>(NR_ * ELEM_BYTES));

        mov(regSrcEnd,
            ptr[pParams
                + offsetof(dlp::kernels::packBParams, n_full_pieces_limit)]);
        add(regSrcEnd, regSrc); // * sizeof(int8_t)
    }

    // K rounded down to a whole number of K-quads.
    mov(regKFull, regK);
    and_(regKFull, -K_FACTOR);

    if (accColSum_) {
        mov(regColSum,
            ptr[pParams + offsetof(dlp::kernels::packBParams, col_sum)]);
    }
}

template<utils::kernelInstrType KType>
void
jitPackBINT8<KType>::generatePanelBody()
{
    Xbyak::Label l_kr_loop, l_kr_done, l_no_ktail, l_rem1, l_rem2;

    xor_(regKr, regKr);
    cmp(regKr, regKFull);
    jge(l_kr_done, T_NEAR);

    L(l_kr_loop);

    interleaveKQuad(K_FACTOR);

    lea(regSrcRow, ptr[regSrcRow + regLdbBytes * 4]);
    add(regDstRow, static_cast<int>(NR_ * K_FACTOR * ELEM_BYTES));
    add(regKr, K_FACTOR);
    cmp(regKr, regKFull);
    jb(l_kr_loop, T_NEAR);

    L(l_kr_done);

    // K remainder in {1,2,3}: missing rows are zero (GEMM reads KC_updated).
    mov(regKr, regK);
    and_(regKr, K_FACTOR - 1);
    jz(l_no_ktail, T_NEAR);
    cmp(regKr, 1);
    je(l_rem1, T_NEAR);
    cmp(regKr, 2);
    je(l_rem2, T_NEAR);
    interleaveKQuad(3);
    jmp(l_no_ktail, T_NEAR);
    L(l_rem2);
    interleaveKQuad(2);
    jmp(l_no_ktail, T_NEAR);
    L(l_rem1);
    interleaveKQuad(1);
    L(l_no_ktail);
}

template<utils::kernelInstrType KType>
void
jitPackBINT8<KType>::generateFullPanelLoop()
{
    if (!nLoop_) {
        mov(regSrcRow, regSrc);
        mov(regDstRow, regDst);
        if (accColSum_) {
            zeroColSumAcc();
        }
        generatePanelBody();
        if (accColSum_) {
            flushColSum();
        }
        return;
    }

    Xbyak::Label l_jc_done, l_jc_loop;

    mov(regSrcBase, regSrc);
    mov(regDstBase, regDst);

    cmp(regSrcBase, regSrcEnd);
    jge(l_jc_done, T_NEAR);

    L(l_jc_loop);

    mov(regSrcRow, regSrcBase);
    mov(regDstRow, regDstBase);

    if (accColSum_) {
        zeroColSumAcc();
    }
    generatePanelBody();
    if (accColSum_) {
        flushColSum();
    }

    add(regSrcBase, static_cast<int>(NR_ * ELEM_BYTES));
    add(regDstBase, regDstPanelStride);
    if (accColSum_) {
        advanceColSum();
    }

    cmp(regSrcBase, regSrcEnd);
    jb(l_jc_loop, T_NEAR);

    L(l_jc_done);
}

template<utils::kernelInstrType KType>
void
jitPackBINT8<KType>::loadRow16(const Xbyak::Xmm&     dst,
                               const Xbyak::Address& addr,
                               bool                  live)
{
    if (!live) {
        vmovdqa64(dst, Xbyak::Xmm(XMM_ZERO));
        return;
    }
    if (useMask_) {
        vmovdqu8(dst | Xbyak::Opmask(MASK_LT16) | T_z, addr);
    } else {
        vmovdqu8(dst, addr);
    }
}

template<utils::kernelInstrType KType>
void
jitPackBINT8<KType>::interleaveKQuad(int nLiveRows)
{
    using Xbyak::Xmm;
    using Xbyak::Zmm;

    Xmm a    = Xmm(XMM_A);
    Xmm b    = Xmm(XMM_B);
    Xmm c    = Xmm(XMM_C);
    Xmm d    = Xmm(XMM_D);
    Xmm ablo = Xmm(XMM_ABLO);
    Xmm cdlo = Xmm(XMM_CDLO);
    Zmm out  = Zmm(XMM_B); // n0 in xmm2 becomes the store ZMM

    const bool live1 = nLiveRows >= 2;
    const bool live2 = nLiveRows >= 3;
    const bool live3 = nLiveRows >= 4;

    int bj = 0;
    if (useZmm64_) {
        for (; bj + 4 <= numBlocks; bj += 4) {
            interleaveKQuad64(nLiveRows, bj);
        }
    }

    for (; bj < numBlocks; ++bj) {
        const int srcOff = bj * BLOCK_N;
        const int dstOff = bj * RegBytes;

        loadRow16(a, ptr[regSrcRow + srcOff], true);
        loadRow16(b, ptr[regSrcRow + regLdbBytes + srcOff], live1);
        loadRow16(c, ptr[regSrcRow + regLdbBytes * 2 + srcOff], live2);
        loadRow16(d, ptr[regSrcRow + regLdb3 + srcOff], live3);

        if (accColSum_) {
            accXmm16(bj, a, true);
            accXmm16(bj, b, live1);
            accXmm16(bj, c, live2);
            accXmm16(bj, d, live3);
        }

        // nr16 primitive: unpack bytes then words, insert 4 XMM lanes into ZMM.
        vpunpcklbw(ablo, a, b);
        vpunpckhbw(a, a, b);
        vpunpcklbw(cdlo, c, d);
        vpunpckhbw(c, c, d);

        vpunpcklwd(b, ablo, cdlo);    // n0-3
        vpunpckhwd(ablo, ablo, cdlo); // n4-7
        vpunpcklwd(d, a, c);          // n8-11
        vpunpckhwd(cdlo, a, c);       // n12-15

        vinserti32x4(out, out, ablo, 1);
        vinserti32x4(out, out, d, 2);
        vinserti32x4(out, out, cdlo, 3);

        vmovdqu64(ptr[regDstRow + dstOff], out);
    }
}

template<utils::kernelInstrType KType>
void
jitPackBINT8<KType>::loadRow64(const Xbyak::Zmm&     dst,
                               const Xbyak::Address& addr,
                               bool                  live)
{
    if (!live) {
        vmovdqa64(dst, Xbyak::Zmm(XMM_ZERO));
        return;
    }
    vmovdqu64(dst, addr);
}

template<utils::kernelInstrType KType>
void
jitPackBINT8<KType>::interleaveKQuad64(int nLiveRows, int bj)
{
    using Xbyak::Zmm;

    const bool live1 = nLiveRows >= 2;
    const bool live2 = nLiveRows >= 3;
    const bool live3 = nLiveRows >= 4;

    const int srcOff = bj * BLOCK_N;
    const int dstOff = bj * RegBytes;

    Zmm a     = Zmm(ZMM_A64);
    Zmm b     = Zmm(ZMM_B64);
    Zmm c     = Zmm(ZMM_C64);
    Zmm d     = Zmm(ZMM_D64);
    Zmm t0    = Zmm(ZMM_T0);
    Zmm t1    = Zmm(ZMM_T1);
    Zmm p0    = Zmm(ZMM_P0);
    Zmm p1    = Zmm(ZMM_P1);
    Zmm p2    = Zmm(ZMM_P2);
    Zmm p3    = Zmm(ZMM_P3);
    Zmm s0    = Zmm(ZMM_S0);
    Zmm s1    = Zmm(ZMM_S1);
    Zmm sel1  = Zmm(ZMM_SEL1);
    Zmm sel11 = Zmm(ZMM_SEL1_1);
    Zmm sel2  = Zmm(ZMM_SEL2);
    Zmm sel21 = Zmm(ZMM_SEL2_1);

    loadRow64(a, ptr[regSrcRow + srcOff], true);
    loadRow64(b, ptr[regSrcRow + regLdbBytes + srcOff], live1);
    loadRow64(c, ptr[regSrcRow + regLdbBytes * 2 + srcOff], live2);
    loadRow64(d, ptr[regSrcRow + regLdb3 + srcOff], live3);

    if (accColSum_) {
        accZmm64(bj, a, b, c, d, nLiveRows);
    }

    // Same unpack + permute as dlp_packb_nr64_*_row_major.
    vpunpcklbw(t0, a, b); // a01
    vpunpckhbw(a, a, b);
    vpunpcklbw(t1, c, d); // c01
    vpunpckhbw(c, c, d);

    vpunpcklwd(b, t0, t1);  // U0
    vpunpckhwd(t0, t0, t1); // U1
    vpunpcklwd(d, a, c);    // U2
    vpunpckhwd(t1, a, c);   // U3

    vmovdqa64(p0, b);
    vpermt2q(p0, sel1, t0);
    vmovdqa64(p1, d);
    vpermt2q(p1, sel1, t1);
    vmovdqa64(p2, b);
    vpermt2q(p2, sel11, t0);
    vmovdqa64(p3, d);
    vpermt2q(p3, sel11, t1);

    vmovdqa64(s0, p0);
    vpermt2q(s0, sel2, p1);
    vmovdqa64(s1, p2);
    vpermt2q(s1, sel2, p3);
    vmovdqa64(a, p0); // S2
    vpermt2q(a, sel21, p1);
    vmovdqa64(b, p2); // S3
    vpermt2q(b, sel21, p3);

    vmovdqu64(ptr[regDstRow + dstOff], s0);
    vmovdqu64(ptr[regDstRow + dstOff + RegBytes], a);
    vmovdqu64(ptr[regDstRow + dstOff + 2 * RegBytes], s1);
    vmovdqu64(ptr[regDstRow + dstOff + 3 * RegBytes], b);
}

template<utils::kernelInstrType KType>
void
jitPackBINT8<KType>::zeroColSumAcc()
{
    for (int i = 0; i < numBlocks; ++i) {
        vpxord(Xbyak::Zmm(ZMM_ACC0 + i), Xbyak::Zmm(ZMM_ACC0 + i),
               Xbyak::Zmm(ZMM_ACC0 + i));
    }
}

template<utils::kernelInstrType KType>
void
jitPackBINT8<KType>::accXmm16(int bj, const Xbyak::Xmm& row, bool live)
{
    if (!live) {
        return;
    }
    vpmovsxbd(Xbyak::Zmm(ZMM_CVT), row);
    vpaddd(Xbyak::Zmm(ZMM_ACC0 + bj), Xbyak::Zmm(ZMM_ACC0 + bj),
           Xbyak::Zmm(ZMM_CVT));
}

template<utils::kernelInstrType KType>
void
jitPackBINT8<KType>::accZmm64(int               bj,
                              const Xbyak::Zmm& a,
                              const Xbyak::Zmm& b,
                              const Xbyak::Zmm& c,
                              const Xbyak::Zmm& d,
                              int               nLiveRows)
{
    using Xbyak::Xmm;
    using Xbyak::Zmm;

    auto accRow = [&](const Zmm& row, bool live) {
        if (!live) {
            return;
        }
        for (int lane = 0; lane < 4; ++lane) {
            if (lane == 0) {
                vpmovsxbd(Zmm(ZMM_CVT), Xmm(row.getIdx()));
            } else {
                vextracti32x4(Xmm(ZMM_CVT), row, lane);
                vpmovsxbd(Zmm(ZMM_CVT), Xmm(ZMM_CVT));
            }
            vpaddd(Zmm(ZMM_ACC0 + bj + lane), Zmm(ZMM_ACC0 + bj + lane),
                   Zmm(ZMM_CVT));
        }
    };

    accRow(a, true);
    accRow(b, nLiveRows >= 2);
    accRow(c, nLiveRows >= 3);
    accRow(d, nLiveRows >= 4);
}

template<utils::kernelInstrType KType>
void
jitPackBINT8<KType>::flushColSum()
{
    using Xbyak::Zmm;

    for (int i = 0; i < numBlocks; ++i) {
        vpslld(Zmm(ZMM_CVT), Zmm(ZMM_ACC0 + i), COL_SUM_SHIFT);
        if (useMask_) {
            // Live n < 16: do not RMW padded lanes (5-loop only zeros nc0).
            vmovdqu32(Zmm(ZMM_ACC0 + i) | Xbyak::Opmask(MASK_LT16) | T_z,
                      ptr[regColSum + i * RegBytes]);
            vpaddd(Zmm(ZMM_CVT), Zmm(ZMM_CVT), Zmm(ZMM_ACC0 + i));
            vmovdqu32(ptr[regColSum + i * RegBytes] | Xbyak::Opmask(MASK_LT16),
                      Zmm(ZMM_CVT));
        } else {
            vpaddd(Zmm(ZMM_CVT), Zmm(ZMM_CVT), ptr[regColSum + i * RegBytes]);
            vmovdqu32(ptr[regColSum + i * RegBytes], Zmm(ZMM_CVT));
        }
    }
}

template<utils::kernelInstrType KType>
void
jitPackBINT8<KType>::advanceColSum()
{
    add(regColSum, static_cast<int>(NR_ * static_cast<md_t>(sizeof(int32_t))));
}

template<utils::kernelInstrType KType>
void
jitPackBINT8<KType>::embedSelectors()
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
    db(reinterpret_cast<uint8_t*>(selector2), sizeof(selector2));
    db(reinterpret_cast<uint8_t*>(selector2_1), sizeof(selector2_1));
    L(selectorsEnd);
}

} // namespace amdzen::PackBcodeGenerator

template class amdzen::PackBcodeGenerator::jitPackBINT8<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
