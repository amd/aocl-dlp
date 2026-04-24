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
    // Check if MR is valid
    if (MR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // For FP16: each ZMM register holds 32 FP16 elements (64 bytes / 2)
    int nElemsPerReg = FP16_PER_ZMM; // 32
    bFullReg         = (NR / nElemsPerReg);
    bMaskReg         = (useMask ? 1 : 0);
    bReg             = bFullReg + bMaskReg;
    cReg             = MR * bReg;

    // Calculate available A registers
    aReg = numRegs - cReg - bReg;

    // Check if we have enough registers
    if (aReg < 1) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Register index assignment
    aRegIdx = 0;           // A registers start at index 0
    bRegIdx = aReg;        // B registers follow A registers
    cRegIdx = aReg + bReg; // C registers follow B registers

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
    // C matrix row stride
    lea(regRsC, ptr[regRsC * FP16_ELEM_SIZE]);

    mov(regTmpCptr, regCPtr);

    if (useMask) {
        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            for (int i = 0; i < utils::NUM_USABLE_MASKS; i++) {
                mask_regs[i] = Xbyak::Opmask(utils::MASK_START_IDX + i);
            }
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
            // Masked load for FP16 fringe
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
jitFP16_GEMM<KType>::scaleBeta()
{
    int betaRegIdx = aRegIdx;

    // Load beta scaling factor (FP16)
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, beta)]);
    vpbroadcastw(RegType(betaRegIdx), ptr[regTmp1]);
    mov(regTmpCptr, regCPtr);

    // Load existing C values, scale by beta, and add to accumulators
    for (iter_t i = 0; i < MR; i++) {
        for (iter_t j = 0; j < bFullReg; j++) {
            // Load existing C values
            vmovdqu16(RegType(bRegIdx + j), ptr[regTmpCptr + j * RegBytes]);
            // Scale by beta using vmulph
            vmulph(Zmm(bRegIdx + j), Zmm(bRegIdx + j), Zmm(betaRegIdx));
            // Add to accumulator using vaddph
            vaddph(Zmm(cRegIdx + i * bReg + j), Zmm(cRegIdx + i * bReg + j),
                   Zmm(bRegIdx + j));
        }
        if (bMaskReg > 0) {
            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                vmovdqu16(RegType(bRegIdx + bFullReg) | mask_regs[0] | T_z,
                          ptr[regTmpCptr + bFullReg * RegBytes]);
                vmulph(Zmm(bRegIdx + bFullReg), Zmm(bRegIdx + bFullReg),
                       Zmm(betaRegIdx));
                vaddph(Zmm(cRegIdx + i * bReg + bFullReg),
                       Zmm(cRegIdx + i * bReg + bFullReg),
                       Zmm(bRegIdx + bFullReg));
            }
        }
        add(regTmpCptr, regRsC);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::generatePostOps(utils::generatorParams& params)
{
    Xbyak::Label local_store_fp16;
    Xbyak::Label local_end_postops;

    // Handle alpha scaling (skip for both alpha=1 and alpha=0)
    if (params.alphaScalingType != dlp::kernel_frame::scalingType::one
        && params.alphaScalingType != dlp::kernel_frame::scalingType::zero) {
        RETURN_IF_ERROR(scaleAlpha());
    }

    // Handle beta scaling
    if (params.betaScalingType != dlp::kernel_frame::scalingType::zero) {
        RETURN_IF_ERROR(scaleBeta());
    }

    if (params.kernelOps.empty()) {
        RETURN_IF_ERROR(storeResult());
        return dlp::jit::jitGeneratorError::success;
    }

    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
    test(regTmp1, regTmp1);
    je(local_store_fp16, T_NEAR);

    if (useMask || bFullReg <= 2) {
        RETURN_IF_ERROR(generatePostOpsRegisterOnly(params));
    } else if (bFullReg == 3 || bFullReg == 4) {
        RETURN_IF_ERROR(generatePostOpsPartialSpill(params));

    } else {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    jmp(local_end_postops, T_NEAR);

    L(local_store_fp16);
    RETURN_IF_ERROR(storeResult());

    L(local_end_postops);

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::generatePostOpsRegisterOnly(utils::generatorParams& params)
{
    using VecPoolType =
        utils::registerPool<typename Traits::RegType, Traits::numRegs>;
    using MaskPoolType =
        utils::registerPool<Xbyak::Opmask, Traits::numMaskRegs>;

    int fp16RegsPerRow = useMask ? 1 : bFullReg;
    int f32RegsPerRow  = fp16RegsPerRow * 2;
    int numMaskRegsF32 = useMask ? 2 : 0;
    int dstRegStart    = 0;
    int numCRegs       = MR * f32RegsPerRow;
    int effectiveNR    = useMask ? 0 : NR;

    if (useMask) {
        populateF32MasksFromFP16();
    }

    convertChunkFromRegsToF32(0, fp16RegsPerRow, dstRegStart);

    VecPoolType vecPool;
    vecPool.setAccumulators(dstRegStart, numCRegs);
    RETURN_IF_ERROR(vecPool.init(this, Traits::regBytes));

    MaskPoolType maskPool;
    maskPool.addPreserve(utils::MASK_START_IDX, useMask ? 1 : 0);
    RETURN_IF_ERROR(maskPool.init(this, utils::maskSaveWidth<KType>(),
                                  Traits::reservedMaskBits));

    gen::kernelOpsHandler<KType> kernelOpsHandler(this);
    int                          maskOffset =
        useMask
                                     ? static_cast<int>(offsetof(dlp::kernels::gemmParams, maskF32[0]))
                                     : -1;

    RETURN_IF_ERROR(kernelOpsHandler.generateKernelOps(
        params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemm, MR,
        effectiveNR, useMask, numMaskRegsF32, dstRegStart, numCRegs, vecPool,
        maskPool, maskOffset));

    for (iter_t row = 0; row < MR; row++) {
        convertF32ChunkToFP16AndStoreRow(dstRegStart + row * f32RegsPerRow, row,
                                         0, fp16RegsPerRow, useMask);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16_GEMM<KType>::generatePostOpsPartialSpill(utils::generatorParams& params)
{
    // First chunk is always 64 cols (2 FP16 ZMMs), processed from registers.
    // Tail chunk is the remainder (32 or 64 cols), spilled to stack.
    // A single 64-col subroutine with masking on the upper 2 F32 ZMMs
    // handles both: mask=0xFFFF for full 64-col, mask=0x0000 for 32-col tail.
    static constexpr int FIRST_FP16_REGS_ROW = 2;
    static constexpr int CHUNK_COLS       = FIRST_FP16_REGS_ROW * FP16_PER_ZMM;
    static constexpr int F32_REGS_PER_ROW = FIRST_FP16_REGS_ROW * 2; // 4
    static constexpr int NUM_MASK_F32     = 2; // masked F32 ZMMs per row

    using VecPoolType =
        utils::registerPool<typename Traits::RegType, Traits::numRegs>;
    using MaskPoolType =
        utils::registerPool<Xbyak::Opmask, Traits::numMaskRegs>;

    int dstRegStart = 0;
    int numCRegs    = MR * F32_REGS_PER_ROW;
    int tailFp16Regs =
        bFullReg - FIRST_FP16_REGS_ROW; // 1 for NR=96, 2 for NR=128
    int tailCols      = tailFp16Regs * FP16_PER_ZMM; // 32 or 64
    int tailStackSize = MR * tailFp16Regs * RegBytes;

    Xbyak::Label postOpsSubroutine;
    Xbyak::Label postOpsSubroutineEnd;

    // Step 1: Spill only the tail chunk to stack.
    sub(rsp, tailStackSize);
    for (iter_t row = 0; row < MR; row++) {
        int rowBase = cRegIdx + row * bReg + FIRST_FP16_REGS_ROW;
        for (int j = 0; j < tailFp16Regs; j++) {
            int stackOff = (row * tailFp16Regs + j) * RegBytes;
            vmovups(ptr[rsp + stackOff], RegType(rowBase + j));
        }
    }

    // Step 2: Process first chunk (64 cols) from accumulator registers.
    // Set masks to 0xFFFF (all lanes active) for the upper 2 F32 ZMMs.
    mov(dword[stackPtr + offsetof(dlp::kernels::gemmParams, maskF32[0])],
        0xFFFFFFFF);

    convertChunkFromRegsToF32(0, FIRST_FP16_REGS_ROW, dstRegStart);
    call(postOpsSubroutine);

    for (iter_t row = 0; row < MR; row++) {
        convertF32ChunkToFP16AndStoreRow(dstRegStart + row * F32_REGS_PER_ROW,
                                         row, 0, FIRST_FP16_REGS_ROW);
    }

    // Step 3: Process tail chunk from stack.
    // Advance post_op_c_j by 64.
    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
    lea(regTmp1, ptr[regTmp1 + CHUNK_COLS]);
    mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, post_op_c_j)],
        regTmp1);

    // Set masks based on tail width.
    // For 64-col tail (NR=128): mask = 0xFFFF,0xFFFF (all active)
    // For 32-col tail (NR=96):  mask = 0x0000,0x0000 (upper half inactive)
    if (tailCols == CHUNK_COLS) {
        mov(dword[stackPtr + offsetof(dlp::kernels::gemmParams, maskF32[0])],
            0xFFFFFFFF);
    } else {
        mov(dword[stackPtr + offsetof(dlp::kernels::gemmParams, maskF32[0])],
            0x00000000);
    }

    // Load tail chunk from stack and convert to F32.
    // For 32-col tail: only the first 2 F32 ZMMs per row get real data,
    // the upper 2 are zeroed (will be masked out by the subroutine).
    for (iter_t row = 0; row < MR; row++) {
        int dstBase = dstRegStart + row * F32_REGS_PER_ROW;
        for (int j = 0; j < tailFp16Regs; j++) {
            int stackOff = (row * tailFp16Regs + j) * RegBytes;
            int dstLo    = dstBase + j * 2;
            int dstHi    = dstLo + 1;

            vmovdqu16(Ymm(dstLo), ptr[rsp + stackOff]);
            vcvtph2ps(Zmm(dstLo), Ymm(dstLo));

            vmovdqu16(Ymm(dstHi), ptr[rsp + stackOff + 32]);
            vcvtph2ps(Zmm(dstHi), Ymm(dstHi));
        }
        // Zero the upper F32 ZMMs if tail is narrower than 64 cols
        if (tailFp16Regs < FIRST_FP16_REGS_ROW) {
            for (int j = tailFp16Regs * 2; j < F32_REGS_PER_ROW; j++) {
                vpxord(Zmm(dstBase + j), Zmm(dstBase + j), Zmm(dstBase + j));
            }
        }
    }

    call(postOpsSubroutine);

    // Store only the valid tail columns
    for (iter_t row = 0; row < MR; row++) {
        convertF32ChunkToFP16AndStoreRow(dstRegStart + row * F32_REGS_PER_ROW,
                                         row, CHUNK_COLS, tailFp16Regs);
    }

    // Step 4: Restore stack and post_op_c_j
    add(rsp, tailStackSize);

    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
    lea(regTmp1, ptr[regTmp1 - CHUNK_COLS]);
    mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, post_op_c_j)],
        regTmp1);

    // Jump over the subroutine body
    jmp(postOpsSubroutineEnd, T_NEAR);

    // Post-ops subroutine: 64-col with masking on upper 2 F32 ZMMs.
    // Emitted once, called for each chunk. Masks loaded from maskF32[0..1].
    L(postOpsSubroutine);
    {
        VecPoolType vecPool;
        vecPool.setAccumulators(dstRegStart, numCRegs);
        RETURN_IF_ERROR(vecPool.init(this, Traits::regBytes));

        // The partial-spill path is only reached when useMask is false
        // (dispatcher in generatePostOps), so mask_regs[0] is not live here
        // and nothing needs preserving. The addPreserve() below is written
        // defensively in case the dispatch conditions change.
        MaskPoolType maskPool;
        maskPool.addPreserve(utils::MASK_START_IDX, useMask ? 1 : 0);
        RETURN_IF_ERROR(maskPool.init(this, utils::maskSaveWidth<KType>(),
                                      Traits::reservedMaskBits));

        gen::kernelOpsHandler<KType> kernelOpsHandler(this);

        int maskOffset =
            static_cast<int>(offsetof(dlp::kernels::gemmParams, maskF32[0]));

        RETURN_IF_ERROR(kernelOpsHandler.generateKernelOps(
            params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemm, MR,
            CHUNK_COLS / 2, true, NUM_MASK_F32, dstRegStart, numCRegs, vecPool,
            maskPool, maskOffset));
    }
    ret();
    L(postOpsSubroutineEnd);

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
jitFP16_GEMM<KType>::convertChunkFromStackToF32(int fp16RegsPerRow,
                                                int dstRegStart)
{
    for (iter_t row = 0; row < MR; row++) {
        int dstBase = dstRegStart + row * fp16RegsPerRow * 2;
        for (iter_t j = 0; j < fp16RegsPerRow; j++) {
            int stackOff = (row * fp16RegsPerRow + j) * RegBytes;
            int dstLo    = dstBase + j * 2;
            int dstHi    = dstLo + 1;

            vmovdqu16(Ymm(dstLo), ptr[rsp + stackOff]);
            vcvtph2ps(Zmm(dstLo), Ymm(dstLo));

            vmovdqu16(Ymm(dstHi), ptr[rsp + stackOff + 32]);
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
