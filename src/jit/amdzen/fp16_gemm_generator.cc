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

#include "fp16_gemm_generator.hh"

namespace amdzen::gen {

using namespace Xbyak;

template<utils::kernelInstrType KType>
jitFP16_GEMM<KType>::jitFP16_GEMM(void* buffer, size_t bufferSize)
    : Xbyak::CodeGenerator(bufferSize, buffer)
{
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::allocateRegisters()
{
    if (MR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Each ZMM holds FP16_PER_ZMM (=32) FP16 elements (64 bytes / 2 bytes).
    bFullReg = (NR / FP16_PER_ZMM);
    bMaskReg = (useMask ? 1 : 0);
    bReg     = bFullReg + bMaskReg;
    cReg     = MR * bReg;
    aReg     = numRegs - cReg - bReg;

    if (aReg < 1) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // C is anchored at the top of the file, B sits immediately below it,
    // and A takes index 0 (the FP16 generator only uses one broadcast
    // slot). Same anchor-from-top pattern as the BF16/U8S8 generators.
    cRegIdx = numRegs - cReg;
    bRegIdx = cRegIdx - bReg;
    aRegIdx = 0;

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitFP16_GEMM<KType>::initializeParameters(bool mLoop)
{
    mov(regTmpAptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);
    if (mLoop) {
        mov(regAPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);
        mov(regMiter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, mIter)]);
        mov(regTmp3, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, psA)]);
        // Scale psA for FP16 (2 bytes per element)
        lea(regTmp3, ptr[regTmp3 * FP16_ELEM_SIZE]);
    }

    // Load post_op_c_i for downscale buffer addressing
    mov(regTmp2,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);

    // Initialize parameter pointers from gemmParams structure
    mov(regCPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, c)]);
    mov(regRsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsA)]);
    mov(regCsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, csA)]);
    mov(regRsB, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsB)]);
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsC)]);

    // Scale strides for FP16 (2 bytes per element)
    // A matrix strides
    lea(regRsA, ptr[regRsA * FP16_ELEM_SIZE]);
    lea(regCsA, ptr[regCsA * FP16_ELEM_SIZE]);
    // B matrix stride (rs_b)
    lea(regRsB, ptr[regRsB * FP16_ELEM_SIZE]);
    /* Byte-scale rsC once per kernel build so every downstream consumer
       (storeResult, scaleBeta, the of32 F32 store/combine, moveCPtr)
       treats regRsC uniformly as a byte stride. The rail is a build-time
       constant (one kernel per c_downscale), so this is a codegen-time
       branch. */
    if (c_downscale == DLP_F32) {
        lea(regRsC, ptr[regRsC * F32_ELEM_SIZE]);
    } else {
        lea(regRsC, ptr[regRsC * FP16_ELEM_SIZE]);
    }

    mov(regTmpCptr, regCPtr);

    /* Bind mask_regs[i] -> kN aliases (k1..k7). MASK_START_IDX = 1 keeps
       k0 reserved by hardware. The fp16 generator uses three slots:
         mask_regs[0] = k1 - NR-fringe mask (FP16 B-load, FP16 C-store)
         mask_regs[1] = k2 - F32 lane mask (low half) for the of32 rail
         mask_regs[2] = k3 - F32 lane mask (high half) for the of32 rail
       The two F32 masks double as the partial-spill path's tail mask. */
    for (iter_t i = 0; i < utils::NUM_USABLE_MASKS; i++) {
        mask_regs[i] = Xbyak::Opmask(utils::MASK_START_IDX + i);
    }

    if (useMask) {
        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            kmovd(mask_regs[0],
                  ptr[stackPtr + offsetof(dlp::kernels::gemmParams, maskFP16)]);
        }
    }
}

template<utils::kernelInstrType KType>
void
jitFP16_GEMM<KType>::initializeAccumulators(utils::generatorParams& params)
{
    // Zero out accumulator registers for FP16 results
    if constexpr (Traits::isAVX512) {
        vpxord(RegType(cRegIdx), RegType(cRegIdx), RegType(cRegIdx));
    }

    for (iter_t i = 1; i < cReg; i++) {
        vmovdqa32(RegType(cRegIdx + i), RegType(cRegIdx));
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::generateIrLoop(utils::generatorParams& params)
{
    initializeAccumulators(params);

    inLocalLabel();

    if (params.alphaScalingType != dlp::kernel_frame::scalingType::zero) {
        // Load B matrix pointer
        mov(regBptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, b)]);

        // Generate K-loop
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kIterBP)]);
        test(regKIter, regKIter);
        je(".FP16_CONSIDKLEFT", T_NEAR);

        // Main unrolled K-loop
        L(".FP16_LOOPKITER");
        RETURN_IF_ERROR(kUnroll(params.K_UNROLL, false));
        sub(regKIter, 1);
        jne(".FP16_LOOPKITER", T_NEAR);

        L(".FP16_CONSIDKLEFT");
        // Handle remaining K iterations
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeft)]);
        test(regKIter, regKIter);
        je(".FP16_POSTACCUM", T_NEAR);

        // Process remaining K one at a time
        L(".FP16_KLEFTLOOP");
        RETURN_IF_ERROR(kUnroll(1, false));
        sub(regKIter, 1);
        jne(".FP16_KLEFTLOOP", T_NEAR);

        L(".FP16_POSTACCUM");
    }

    // Generate post-ops and store
    RETURN_IF_ERROR(generatePostOps(params));

    vzeroupper();
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::loadBValues()
{
    // Load B matrix values using vmovdqu16 for FP16
    for (iter_t i = 0; i < bFullReg; i++) {
        if constexpr (Traits::isAVX512) {
            vmovdqu16(RegType(bRegIdx + i), ptr[regBptr + i * RegBytes]);
        }
    }

    if (useMask) {
        int maskRegIndex = bRegIdx + bFullReg;
        if (maskRegIndex >= numRegs) {
            return dlp::jit::jitGeneratorError::badKernelInfo;
        }

        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            vmovdqu16(RegType(maskRegIndex) | mask_regs[0] | T_z,
                      ptr[regBptr + bFullReg * RegBytes]);
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::storeResult()
{
    mov(regTmpCptr, regCPtr);

    // Default: store as FP16
    return storeResultFP16();
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::storeResultFP16()
{
    // Store FP16 accumulator results using vmovdqu16
    for (iter_t i = 0; i < MR; i++) {
        for (iter_t j = 0; j < bFullReg; j++) {
            vmovdqu16(ptr[regTmpCptr + j * RegBytes],
                      RegType(cRegIdx + i * bReg + j));
        }
        if (bMaskReg > 0) {
            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                vmovdqu16(ptr[regTmpCptr + bFullReg * RegBytes] | mask_regs[0],
                          RegType(cRegIdx + i * bReg + bFullReg));
            }
        }
        add(regTmpCptr, regRsC);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::scaleAlpha()
{
    int alphaRegIdx = aRegIdx;

    // Load alpha scaling factor (FP16)
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, alpha)]);
    // Broadcast 16-bit FP16 value to all lanes
    vpbroadcastw(RegType(alphaRegIdx), ptr[regTmp1]);

    // Scale all accumulator registers with alpha using vmulph
    for (iter_t i = 0; i < cReg; i++) {
        vmulph(Zmm(cRegIdx + i), Zmm(cRegIdx + i), Zmm(alphaRegIdx));
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::scaleBeta(bool betaIsOne)
{
    int betaRegIdx = aRegIdx;

    Xbyak::Label betaZeroEnd;

    if (!betaIsOne) {
        mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, beta)]);
        vpbroadcastw(RegType(betaRegIdx), ptr[regTmp1]);

        // The Decision Engine passes betaScalingType as generic for k > KC
        // even when beta = 0. Check at runtime to avoid C accesses when
        // beta = 0, preventing NaN propagation from uninitialized C.
        vpxord(RegType(bRegIdx), RegType(bRegIdx), RegType(bRegIdx));
        vucomish(Xmm(betaRegIdx), Xmm(bRegIdx));
        je(betaZeroEnd, T_NEAR);
    }
    mov(regTmpCptr, regCPtr);

    for (iter_t i = 0; i < MR; i++) {
        for (iter_t j = 0; j < bFullReg; j++) {
            vmovdqu16(RegType(bRegIdx + j), ptr[regTmpCptr + j * RegBytes]);
            if (!betaIsOne) {
                vmulph(Zmm(bRegIdx + j), Zmm(bRegIdx + j), Zmm(betaRegIdx));
            }
            vaddph(Zmm(cRegIdx + i * bReg + j), Zmm(cRegIdx + i * bReg + j),
                   Zmm(bRegIdx + j));
        }
        if (bMaskReg > 0) {
            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                vmovdqu16(RegType(bRegIdx + bFullReg) | mask_regs[0] | T_z,
                          ptr[regTmpCptr + bFullReg * RegBytes]);
                if (!betaIsOne) {
                    vmulph(Zmm(bRegIdx + bFullReg), Zmm(bRegIdx + bFullReg),
                           Zmm(betaRegIdx));
                }
                vaddph(Zmm(cRegIdx + i * bReg + bFullReg),
                       Zmm(cRegIdx + i * bReg + bFullReg),
                       Zmm(bRegIdx + bFullReg));
            }
        }
        add(regTmpCptr, regRsC);
    }

    L(betaZeroEnd);
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::generatePostOps(utils::generatorParams& params)
{
    /* Apply FP16 alpha (both rails consume alpha as FP16; the public
       APIs / 5-loop narrow it before dispatch). */
    if (params.alphaScalingType != dlp::kernel_frame::scalingType::one
        && params.alphaScalingType != dlp::kernel_frame::scalingType::zero) {
        RETURN_IF_ERROR(scaleAlpha());
    }

    /* Tile shape for the partial-spill 64-col chunks (NR=96 / NR=128). The
       head chunk widens FIRST_FP16_REGS_ROW = 2 FP16 ZMMs per row to
       F32_REGS_PER_ROW = 4 F32 ZMMs per row; the tail chunk reuses the
       same F32 row stride and zero-pads for NR=96 (tailFp16Regs == 1) so
       the masked store/combine emission is geometry-uniform. */
    static constexpr int FIRST_FP16_REGS_ROW = 2;
    static constexpr int CHUNK_COLS       = FIRST_FP16_REGS_ROW * FP16_PER_ZMM;
    static constexpr int F32_REGS_PER_ROW = FIRST_FP16_REGS_ROW * 2;

    const bool needsPartialSpill =
        (!useMask && (bFullReg == 3 || bFullReg == 4));
    const bool hasKernelOps = !params.kernelOps.empty();

    /* dstRegStart anchors the F32 tile at ZMM 0..N-1. scratchRegStart
       reserves the top two ZMMs for the of32 beta broadcast and the
       user-C tmp load. allocateRegisters has verified the FP16
       accumulator (cRegIdx..numRegs-1) does not overlap these slots. */
    const int dstRegStart     = 0;
    const int scratchRegStart = numRegs - 2;

    /* Of16 short-circuits before the chunk path:
         - no kernelOps:           scaleBeta -> storeResult, return
         - kernelOps + !is_last_k: scaleBeta -> storeResult, return
         - kernelOps + is_last_k:  scaleBeta -> chunk path
       Of32 enters the chunk path every KC and folds beta + the F32
       store inside processChunk; kernelOps is is_last_k-gated within
       the chunk. */
    Xbyak::Label local_intermediate_store;
    Xbyak::Label local_postops_end;

    if (c_downscale != DLP_F32) {
        if (params.betaScalingType != dlp::kernel_frame::scalingType::zero) {
            const bool betaIsOne =
                (params.betaScalingType == dlp::kernel_frame::scalingType::one);
            RETURN_IF_ERROR(scaleBeta(betaIsOne));
        }
        if (!hasKernelOps) {
            return storeResult();
        }

        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
        test(regTmp1, regTmp1);
        je(local_intermediate_store, T_NEAR);
    }

    /* Build chunk descriptors. The partial-spill path emits a shared
       kernelOps subroutine after the chunks (if hasKernelOps) and points
       both chunk configs at it; the register-only path emits the
       handler body inline inside processChunk. */
    Xbyak::Label  kernelOpsSub;
    Xbyak::Label* sharedSub =
        (needsPartialSpill && hasKernelOps) ? &kernelOpsSub : nullptr;

    auto buildHeadChunk = [&]() {
        ChunkConfig cfg;
        cfg.source           = ChunkConfig::Source::Registers;
        cfg.fp16SrcOffset    = 0;
        cfg.fp16RegsPerRow   = needsPartialSpill ? FIRST_FP16_REGS_ROW
                                                 : (useMask ? 1 : bFullReg);
        cfg.dstRegStart      = dstRegStart;
        cfg.scratchRegStart  = scratchRegStart;
        cfg.numFullF32Pairs  = needsPartialSpill ? 1 : (useMask ? 0 : bFullReg);
        cfg.hasMaskedF32Pair = needsPartialSpill || useMask;
        cfg.numFullColsF32   = needsPartialSpill ? (CHUNK_COLS / 2)
                                                 : (useMask ? 0 : NR);
        cfg.colElemOffset    = 0;
        cfg.maskedFp16Store  = useMask;
        cfg.zeroUpperF32Pair = false;
        if (needsPartialSpill) {
            cfg.maskSource = ChunkConfig::MaskSource::ConstantAllOnes;
        } else if (useMask) {
            cfg.maskSource = ChunkConfig::MaskSource::FromFp16Mask;
        } else {
            cfg.maskSource = ChunkConfig::MaskSource::None;
        }
        cfg.kernelOpsSubroutine = sharedSub;
        return cfg;
    };

    auto buildTailChunk = [&](int tailFp16Regs, int tailCols) {
        ChunkConfig cfg;
        cfg.source              = ChunkConfig::Source::Stack;
        cfg.fp16SrcOffset       = 0;
        cfg.fp16RegsPerRow      = tailFp16Regs;
        cfg.dstRegStart         = dstRegStart;
        cfg.scratchRegStart     = scratchRegStart;
        cfg.numFullF32Pairs     = 1;
        cfg.hasMaskedF32Pair    = true;
        cfg.numFullColsF32      = CHUNK_COLS / 2;
        cfg.colElemOffset       = CHUNK_COLS;
        cfg.maskedFp16Store     = false;
        cfg.zeroUpperF32Pair    = (tailFp16Regs < FIRST_FP16_REGS_ROW);
        cfg.maskSource          = (tailCols == CHUNK_COLS)
                                      ? ChunkConfig::MaskSource::ConstantAllOnes
                                      : ChunkConfig::MaskSource::ConstantTailHalfOnly;
        cfg.kernelOpsSubroutine = sharedSub;
        return cfg;
    };

    /* Run the chunk(s). Register-only paths emit one chunk; partial-spill
       paths emit two chunks bracketed by stack spill/restore and the
       post_op_c_j +/- CHUNK_COLS dance the kernelOpsHandler needs. */
    if (!needsPartialSpill) {
        RETURN_IF_ERROR(processChunk(params, buildHeadChunk()));
    } else {
        const int tailFp16Regs = bFullReg - FIRST_FP16_REGS_ROW;
        const int tailCols     = tailFp16Regs * FP16_PER_ZMM;
        const int tailStackSz  = MR * tailFp16Regs * RegBytes;

        sub(rsp, tailStackSz);
        for (iter_t row = 0; row < MR; row++) {
            int rowBase = cRegIdx + row * bReg + FIRST_FP16_REGS_ROW;
            for (int j = 0; j < tailFp16Regs; j++) {
                int stackOff = (row * tailFp16Regs + j) * RegBytes;
                vmovups(ptr[rsp + stackOff], RegType(rowBase + j));
            }
        }

        RETURN_IF_ERROR(processChunk(params, buildHeadChunk()));

        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
        lea(regTmp1, ptr[regTmp1 + CHUNK_COLS]);
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_j)],
            regTmp1);

        RETURN_IF_ERROR(
            processChunk(params, buildTailChunk(tailFp16Regs, tailCols)));

        add(rsp, tailStackSz);

        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
        lea(regTmp1, ptr[regTmp1 - CHUNK_COLS]);
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_j)],
            regTmp1);
    }

    /* Of16 join: the chunk path falls through here on is_last_k; the
       intermediate-KC FP16 spill jumps in below. */
    if (c_downscale != DLP_F32) {
        jmp(local_postops_end, T_NEAR);
        L(local_intermediate_store);
        RETURN_IF_ERROR(storeResult());
        L(local_postops_end);
    }

    /* Emit the shared kernelOps subroutine after the chunks (jumped over
       at runtime). Only the partial-spill path needs it - the
       register-only chunk emits the handler body inline. */
    if (sharedSub != nullptr) {
        const int chunkNumCRegs = MR * F32_REGS_PER_ROW;
        RETURN_IF_ERROR(emitKernelOpsSubroutine(
            params, kernelOpsSub, dstRegStart, chunkNumCRegs, CHUNK_COLS / 2,
            /*hasMaskedF32Pair=*/true));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::processChunk(utils::generatorParams& params,
                                  const ChunkConfig&      cfg)
{
    /* Step 0: seed the of32 lane masks (maskF32[0..1]) and load
       mask_regs[1..2] from them. The seed source comes from
       cfg.maskSource:
         - FromFp16Mask:         split the FP16 NR-fringe mask in two
         - ConstantAllOnes:      full 32-col F32 chunk
         - ConstantTailHalfOnly: 32-col tail (NR=96), upper half disabled
         - None:                 no masked F32 pair, masks unused
       k2/k3 are loaded only on the of32 rail because applyBetaCombineF32
       and storeResultF32_inplace read them directly; the
       kernelOpsHandler loads them itself on of16 when it needs them. */
    if (cfg.hasMaskedF32Pair) {
        switch (cfg.maskSource) {
            case ChunkConfig::MaskSource::FromFp16Mask:
                populateF32MasksFromFP16();
                break;
            case ChunkConfig::MaskSource::ConstantAllOnes:
                mov(dword[stackPtr
                          + offsetof(dlp::kernels::gemmParams, maskF32[0])],
                    0xFFFFFFFF);
                break;
            case ChunkConfig::MaskSource::ConstantTailHalfOnly:
                mov(dword[stackPtr
                          + offsetof(dlp::kernels::gemmParams, maskF32[0])],
                    0x00000000);
                break;
            case ChunkConfig::MaskSource::None:
                break;
        }
        if (c_downscale == DLP_F32) {
            kmovw(mask_regs[1],
                  ptr[stackPtr + offsetof(dlp::kernels::gemmParams, maskF32)]);
            kmovw(mask_regs[2],
                  ptr[stackPtr + offsetof(dlp::kernels::gemmParams, maskF32)
                      + sizeof(uint16_t)]);
        }
    }

    /* Step 1: widen the FP16 chunk into the F32 tile starting at
       cfg.dstRegStart. F32 row stride is f32RegsPerRow; for the
       NR=96 partial-spill tail we have fp16RegsPerRow == 1 but
       f32RegsPerRow == 2, and zeroUpperF32Pair zero-pads the upper F32
       pair so the masked store path is geometry-uniform. */
    const int fp16RegsPerRow = cfg.fp16RegsPerRow;
    const int totalPairs = cfg.numFullF32Pairs + (cfg.hasMaskedF32Pair ? 1 : 0);
    const int f32RegsPerRow = totalPairs * 2;

    if (cfg.source == ChunkConfig::Source::Registers) {
        /* convertChunkFromRegsToF32 uses fp16RegsPerRow*2 as its row
           stride, which equals f32RegsPerRow on every register-source
           chunk (head chunks always widen FIRST_FP16_REGS_ROW; the
           register-only single chunk has fp16RegsPerRow*2 == f32RegsPerRow
           by construction). */
        convertChunkFromRegsToF32(cfg.fp16SrcOffset, fp16RegsPerRow,
                                  cfg.dstRegStart);
    } else {
        for (iter_t row = 0; row < MR; row++) {
            int dstBase = cfg.dstRegStart + row * f32RegsPerRow;
            for (int j = 0; j < fp16RegsPerRow; j++) {
                int stackOff = (row * fp16RegsPerRow + j) * RegBytes;
                int dstLo    = dstBase + j * 2;
                int dstHi    = dstLo + 1;

                vmovdqu16(Ymm(dstLo), ptr[rsp + stackOff]);
                vcvtph2ps(Zmm(dstLo), Ymm(dstLo));
                vmovdqu16(Ymm(dstHi), ptr[rsp + stackOff + 32]);
                vcvtph2ps(Zmm(dstHi), Ymm(dstHi));
            }
            if (cfg.zeroUpperF32Pair) {
                for (int j = fp16RegsPerRow * 2; j < f32RegsPerRow; j++) {
                    vpxord(Zmm(dstBase + j), Zmm(dstBase + j),
                           Zmm(dstBase + j));
                }
            }
        }
    }

    /* Step 2: of32 F32 beta-combine with user C from regCPtr. No-op on
       of16 (FP16 scaleBeta already ran inside generatePostOps). */
    if (c_downscale == DLP_F32
        && params.betaScalingType != dlp::kernel_frame::scalingType::zero) {
        RETURN_IF_ERROR(applyBetaCombineF32(
            params, cfg.numFullF32Pairs, cfg.hasMaskedF32Pair, cfg.dstRegStart,
            cfg.scratchRegStart, cfg.colElemOffset));
    }

    /* Step 3: kernelOpsHandler chain (gated on is_last_k at runtime so
       intermediate-KC of32 chunks skip it). The shared subroutine path
       CALLs cfg.kernelOpsSubroutine; the inline path emits the body in
       place. */
    if (!params.kernelOps.empty()) {
        Xbyak::Label local_skip;

        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
        test(regTmp1, regTmp1);
        je(local_skip, T_NEAR);

        if (cfg.kernelOpsSubroutine != nullptr) {
            call(*cfg.kernelOpsSubroutine);
        } else {
            const int chunkNumCRegs = MR * f32RegsPerRow;
            RETURN_IF_ERROR(emitKernelOpsBody(params, cfg.dstRegStart,
                                              chunkNumCRegs, cfg.numFullColsF32,
                                              cfg.hasMaskedF32Pair));
        }

        L(local_skip);
    }

    /* Step 4: store. Of32 streams the F32 tile in place to user C; of16
       narrows row-by-row back to FP16 and stores. */
    if (c_downscale == DLP_F32) {
        RETURN_IF_ERROR(
            storeResultF32_inplace(cfg.numFullF32Pairs, cfg.hasMaskedF32Pair,
                                   cfg.dstRegStart, cfg.colElemOffset));
    } else {
        for (iter_t row = 0; row < MR; row++) {
            convertF32ChunkToFP16AndStoreRow(
                cfg.dstRegStart + row * f32RegsPerRow, row, cfg.colElemOffset,
                fp16RegsPerRow, cfg.maskedFp16Store);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::emitKernelOpsBody(utils::generatorParams& params,
                                       int                     dstRegStart,
                                       int                     numCRegs,
                                       int                     numFullColsF32,
                                       bool                    hasMaskedF32Pair)
{
    using VecPoolType =
        utils::registerPool<typename Traits::RegType, Traits::numRegs>;
    using MaskPoolType =
        utils::registerPool<Xbyak::Opmask, Traits::numMaskRegs>;

    VecPoolType vecPool;
    vecPool.setAccumulators(dstRegStart, numCRegs);
    RETURN_IF_ERROR(vecPool.init(this, Traits::regBytes));

    /* Preserve the masks we own across this generator (k1 for the FP16
       fringe and, on the of32 rail, k2/k3 for the F32 lane masks loaded
       in applyBetaCombineF32) so the handler's fringe-mask allocator
       skips them - that's how every other GEMM/GEMV generator in the
       suite hands locally-loaded masks to the post-op chain (see
       u8s8/s8/bf16/f32f16). */
    MaskPoolType maskPool;
    int          numPreserved = 1 + (hasMaskedF32Pair ? 2 : 0);
    maskPool.addPreserve(utils::MASK_START_IDX, numPreserved);
    RETURN_IF_ERROR(maskPool.init(this, utils::maskSaveWidth<KType>(),
                                  Traits::reservedMaskBits));

    gen::kernelOpsHandler<KType> kernelOpsHandler(this);
    int                          numMaskRegsF32 = hasMaskedF32Pair ? 2 : 0;
    int                          maskOffset =
        hasMaskedF32Pair
                                     ? static_cast<int>(offsetof(dlp::kernels::gemmParams, maskF32[0]))
                                     : -1;

    return kernelOpsHandler.generateKernelOps(
        params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemm, MR,
        numFullColsF32, hasMaskedF32Pair, numMaskRegsF32, dstRegStart, numCRegs,
        vecPool, maskPool, maskOffset);
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::emitKernelOpsSubroutine(utils::generatorParams& params,
                                             Xbyak::Label&           label,
                                             int  dstRegStart,
                                             int  numCRegs,
                                             int  numFullColsF32,
                                             bool hasMaskedF32Pair)
{
    Xbyak::Label endLabel;
    jmp(endLabel, T_NEAR);
    L(label);
    RETURN_IF_ERROR(emitKernelOpsBody(params, dstRegStart, numCRegs,
                                      numFullColsF32, hasMaskedF32Pair));
    ret();
    L(endLabel);
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::applyBetaCombineF32(utils::generatorParams& params,
                                         int  numFullF32Pairs,
                                         bool hasMaskedF32Pair,
                                         int  dstRegStart,
                                         int  scratchRegStart,
                                         int  colElemOffsetF32)
{
    /* Of32 F32 beta-combine: load user C row -> Zmm(tmpReg), broadcast
       beta -> Zmm(betaRegIdx), then combine with the F32 accumulator
       (vfmadd231ps, or vaddps when beta == 1.0). Both scratch ZMMs live
       above the F32 tile and are dead after this function returns. */
    int tmpReg     = scratchRegStart + 1;
    int betaRegIdx = scratchRegStart;

    Xbyak::Label betaZeroEnd;

    if (params.betaScalingType != dlp::kernel_frame::scalingType::one) {
        mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, beta)]);
        vbroadcastss(Zmm(betaRegIdx), ptr[regTmp1]);

        // Runtime beta=0 check: the Decision Engine passes betaScalingType
        // as generic for k > KC even when beta = 0. Skip C accesses to
        // avoid NaN/Inf propagation from uninitialized C.
        vpxord(Zmm(tmpReg), Zmm(tmpReg), Zmm(tmpReg));
        vucomiss(Xmm(betaRegIdx), Xmm(tmpReg));
        je(betaZeroEnd, T_NEAR);
    }

    int totalPairs    = numFullF32Pairs + (hasMaskedF32Pair ? 1 : 0);
    int f32RegsPerRow = totalPairs * 2;
    int baseColBytes  = colElemOffsetF32 * F32_ELEM_SIZE;

    mov(regTmpCptr, regCPtr);
    if (colElemOffsetF32 > 0) {
        add(regTmpCptr, baseColBytes);
    }

    for (iter_t i = 0; i < MR; i++) {
        int rowDst = dstRegStart + i * f32RegsPerRow;
        for (iter_t j = 0; j < f32RegsPerRow; j++) {
            int colByteOff = j * RegBytes;
            int dstReg     = rowDst + j;
            int pairIdx    = j / 2; // which F32 pair
            int laneInPair = j & 1; // lo (0) or hi (1)

            bool inMaskedPair =
                hasMaskedF32Pair && (pairIdx == numFullF32Pairs);

            if (inMaskedPair) {
                Xbyak::Opmask kMask = (laneInPair == 0) ? mask_regs[1]
                                                        : mask_regs[2];
                vmovups(Zmm(tmpReg) | kMask | T_z,
                        ptr[regTmpCptr + colByteOff]);
            } else {
                vmovups(Zmm(tmpReg), ptr[regTmpCptr + colByteOff]);
            }

            if (params.betaScalingType == dlp::kernel_frame::scalingType::one) {
                vaddps(Zmm(dstReg), Zmm(dstReg), Zmm(tmpReg));
            } else {
                vfmadd231ps(Zmm(dstReg), Zmm(tmpReg), Zmm(betaRegIdx));
            }
        }
        add(regTmpCptr, regRsC);
    }

    L(betaZeroEnd);
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::storeResultF32_inplace(int  numFullF32Pairs,
                                            bool hasMaskedF32Pair,
                                            int  dstRegStart,
                                            int  colElemOffsetF32)
{
    /* mask_regs[1..2] are still live from applyBetaCombineF32 (and
       preserved across kernelOpsHandler by the mask-pool's preserve
       list), so no reload is needed here. */
    int totalPairs    = numFullF32Pairs + (hasMaskedF32Pair ? 1 : 0);
    int f32RegsPerRow = totalPairs * 2;
    int baseColBytes  = colElemOffsetF32 * F32_ELEM_SIZE;

    mov(regTmpCptr, regCPtr);
    if (colElemOffsetF32 > 0) {
        add(regTmpCptr, baseColBytes);
    }

    for (iter_t i = 0; i < MR; i++) {
        int rowDst = dstRegStart + i * f32RegsPerRow;
        for (iter_t j = 0; j < f32RegsPerRow; j++) {
            int colByteOff = j * RegBytes;
            int dstReg     = rowDst + j;
            int pairIdx    = j / 2;
            int laneInPair = j & 1;

            bool inMaskedPair =
                hasMaskedF32Pair && (pairIdx == numFullF32Pairs);

            if (inMaskedPair) {
                Xbyak::Opmask kMask = (laneInPair == 0) ? mask_regs[1]
                                                        : mask_regs[2];
                vmovups(ptr[regTmpCptr + colByteOff] | kMask, Zmm(dstReg));
            } else {
                vmovups(ptr[regTmpCptr + colByteOff], Zmm(dstReg));
            }
        }
        add(regTmpCptr, regRsC);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::generateKernel(utils::generatorParams& params)
{
    RETURN_IF_ERROR(utils::jitGeneratorUtils::checkValidGemmParams(params));

    MR          = params.MR;
    NR          = params.NR;
    useMask     = params.useMask;
    c_downscale = params.c_downscale;

    RETURN_IF_ERROR(allocateRegisters());

    // StackFrame manages general purpose registers
    {
        Xbyak::util::StackFrame stackFrame(this, 1, 13, 0);
        initializeStackFrame(stackFrame);
        initializeParameters(params.mLoop);

        // Generate M-loop if needed, otherwise just IR loop
        if (params.mLoop) {
            RETURN_IF_ERROR(generateMLoop(params));
        } else {
            RETURN_IF_ERROR(generateIrLoop(params));
        }
    } // StackFrame destructor inserts 'ret' here

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::generateMLoop(utils::generatorParams& params)
{
    // Check if M iterations are needed
    test(regMiter, regMiter);
    je(".FP16_MLOOP_END", T_NEAR);

    L(".FP16_MLOOP_START");

    // Generate the inner IR loop
    RETURN_IF_ERROR(generateIrLoop(params));

    // Move to next M block
    RETURN_IF_ERROR(moveCPtr());

    // Update A pointer for next M block
    mov(regTmpAptr, regAPtr);
    lea(regTmpAptr, ptr[regTmpAptr + regTmp3]);
    mov(regAPtr, regTmpAptr);

    lea(regTmp2, ptr[regTmp2 + MR]);
    mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, post_op_c_i)],
        regTmp2);

    // Decrement M counter
    sub(regMiter, 1);
    jne(".FP16_MLOOP_START", T_NEAR);

    L(".FP16_MLOOP_END");

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::broadcastAFMAwithB(bool isKRemainder)
{
    for (iter_t i = 0; i < MR; i++) {
        // Broadcast 16-bit FP16 A element to all lanes using vpbroadcastw
        vpbroadcastw(RegType(aRegIdx), ptr[regTmpAptr]);

        // FMA: C += A * B using vfmadd231ph (native FP16 FMA)
        for (iter_t j = 0; j < bReg; j++) {
            vfmadd231ph(RegType(cRegIdx + i * bReg + j), RegType(aRegIdx),
                        RegType(bRegIdx + j));
        }

        // Advance A pointer to next row using rs_a (already scaled by element
        // size)
        add(regTmpAptr, regRsA);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::kUnroll(int unroll, bool isKRemainder)
{
    // Unroll the FP16 kernel loop
    for (iter_t p = 0; p < unroll; p++) {
        // Save A pointer
        mov(regTmp1, regTmpAptr);

        // Load B registers
        RETURN_IF_ERROR(loadBValues());
        add(regBptr, regRsB);

        // Perform FP16 FMA compute
        RETURN_IF_ERROR(broadcastAFMAwithB(isKRemainder));

        // Advance A pointer to next column (cs_a = 1 element = 2 bytes)
        lea(regTmpAptr, ptr[regTmp1 + regCsA]);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitFP16_GEMM<KType>::initializeStackFrame(Xbyak::util::StackFrame& stackFrame)
{
    stackPtr = stackFrame.p[0];

    regTmpAptr = stackFrame.t[0];
    regBptr    = stackFrame.t[1];
    regTmpCptr = stackFrame.t[2];
    regRsA     = stackFrame.t[3];
    regCsA     = stackFrame.t[4];
    regRsB     = stackFrame.t[5];
    regRsC     = stackFrame.t[6];
    regKIter   = stackFrame.t[7];
    regCPtr    = stackFrame.t[8];
    regAPtr    = stackFrame.t[9];
    regTmp1    = stackFrame.t[10];
    regTmp2    = stackFrame.t[11];
    regTmp3    = stackFrame.t[12];
}

template<utils::kernelInstrType KType>
void
jitFP16_GEMM<KType>::spillChunkToStack(int srcRegOffset, int fp16RegsPerRow)
{
    sub(rsp, MR * fp16RegsPerRow * RegBytes);
    for (iter_t row = 0; row < MR; row++) {
        int rowBase = cRegIdx + row * bReg + srcRegOffset;
        for (iter_t j = 0; j < fp16RegsPerRow; j++) {
            int stackOff = (row * fp16RegsPerRow + j) * RegBytes;
            vmovups(ptr[rsp + stackOff], RegType(rowBase + j));
        }
    }
}

template<utils::kernelInstrType KType>
void
jitFP16_GEMM<KType>::restoreStackAfterChunkSpill(int fp16RegsPerRow)
{
    add(rsp, MR * fp16RegsPerRow * RegBytes);
}

template<utils::kernelInstrType KType>
void
jitFP16_GEMM<KType>::convertChunkFromRegsToF32(int srcRegOffset,
                                               int fp16RegsPerRow,
                                               int dstRegStart)
{
    for (iter_t row = 0; row < MR; row++) {
        int srcBase = cRegIdx + row * bReg + srcRegOffset;
        int dstBase = dstRegStart + row * fp16RegsPerRow * 2;
        for (iter_t j = 0; j < fp16RegsPerRow; j++) {
            int srcReg = srcBase + j;
            int dstLo  = dstBase + j * 2;
            int dstHi  = dstLo + 1;

            vextractf32x8(Ymm(dstHi), Zmm(srcReg), 1);
            vcvtph2ps(Zmm(dstLo), Ymm(srcReg));
            vcvtph2ps(Zmm(dstHi), Ymm(dstHi));
        }
    }
}

template<utils::kernelInstrType KType>
void
jitFP16_GEMM<KType>::convertF32ChunkToFP16AndStoreRow(int  f32RegStart,
                                                      int  rowIdx,
                                                      int  colElemOffset,
                                                      int  fp16RegsPerRow,
                                                      bool maskedStore)
{
    mov(regTmpCptr, regCPtr);
    if (rowIdx > 0) {
        mov(regTmp1, regRsC);
        imul(regTmp1, regTmp1, rowIdx);
        add(regTmpCptr, regTmp1);
    }
    if (colElemOffset > 0) {
        add(regTmpCptr, colElemOffset * FP16_ELEM_SIZE);
    }

    for (iter_t j = 0; j < fp16RegsPerRow; j++) {
        int srcLo = f32RegStart + j * 2;
        int srcHi = f32RegStart + j * 2 + 1;

        vcvtps2ph(Ymm(srcLo), Zmm(srcLo), 0x04);
        vcvtps2ph(Ymm(srcHi), Zmm(srcHi), 0x04);
        vinserti32x8(Zmm(srcLo), Zmm(srcLo), Ymm(srcHi), 1);

        if (maskedStore && (j == (fp16RegsPerRow - 1))) {
            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                vmovdqu16(ptr[regTmpCptr + j * RegBytes] | mask_regs[0],
                          Zmm(srcLo));
            }
        } else {
            vmovdqu16(ptr[regTmpCptr + j * RegBytes], Zmm(srcLo));
        }
    }
}

template<utils::kernelInstrType KType>
void
jitFP16_GEMM<KType>::populateF32MasksFromFP16()
{
    mov(Reg32(regTmp1.getIdx()),
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, maskFP16)]);
    mov(word[stackPtr + offsetof(dlp::kernels::gemmParams, maskF32)],
        Reg16(regTmp1.getIdx()));
    shr(Reg32(regTmp1.getIdx()), 16);
    mov(word[stackPtr + offsetof(dlp::kernels::gemmParams, maskF32)
             + sizeof(uint16_t)],
        Reg16(regTmp1.getIdx()));
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::moveCPtr()
{
    if (MR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Compute regCPtr += MR * regRsC
    // x86 LEA can only have ONE index register, so we must use imul
    mov(regTmp1, regRsC);
    imul(regTmp1, regTmp1, MR);
    add(regCPtr, regTmp1);

    return dlp::jit::jitGeneratorError::success;
}

} // namespace amdzen::gen

// Explicit template instantiation
template class amdzen::gen::jitFP16_GEMM<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
