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

#include "f32f16_gemm_generator.hh"

namespace amdzen::gen {

using namespace Xbyak;

template<utils::kernelInstrType KType>
jitF32FP16_GEMM<KType>::jitF32FP16_GEMM(void* buffer, size_t bufferSize)
    : Xbyak::CodeGenerator(bufferSize, buffer)
{
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16_GEMM<KType>::allocateRegisters()
{
    if (MR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Output is F32: each ZMM holds 16 F32 elements
    // NR=64 → bFullReg = 64 / 16 = 4
    bFullReg = (NR / F32_PER_ZMM);
    bMaskReg = (useMask ? 1 : 0);
    bReg     = bFullReg + bMaskReg;
    cReg     = MR * bReg;

    aReg = numRegs - cReg - bReg;

    if (aReg < 1) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    aRegIdx = 0;
    bRegIdx = aReg;
    cRegIdx = aReg + bReg;

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitF32FP16_GEMM<KType>::initializeParameters(bool mLoop)
{
    mov(regTmpAptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);
    if (mLoop) {
        mov(regAPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);
        mov(regMiter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, mIter)]);
        mov(regTmp3, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, psA)]);
        // Scale psA for F32 (4 bytes per element) — A is F32
        lea(regTmp3, ptr[regTmp3 * F32_ELEM_SIZE]);
    }

    mov(regTmp2,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);

    mov(regCPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, c)]);
    mov(regRsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsA)]);
    mov(regCsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, csA)]);
    mov(regRsB, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsB)]);
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsC)]);

    // Scale strides: A is F32 (4 bytes), B is FP16 (2 bytes), C is F32 (4
    // bytes)
    lea(regRsA, ptr[regRsA * F32_ELEM_SIZE]);  // A stride: × 4 bytes
    lea(regCsA, ptr[regCsA * F32_ELEM_SIZE]);  // A col stride: × 4 bytes
    lea(regRsB, ptr[regRsB * FP16_ELEM_SIZE]); // B stride: × 2 bytes (FP16!)
    lea(regRsC, ptr[regRsC * F32_ELEM_SIZE]);  // C stride: × 4 bytes

    mov(regTmpCptr, regCPtr);

    if (useMask) {
        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            // For F32×FP16: mask is 16-bit (16 F32 elements per ZMM)
            kmovw(k3,
                  ptr[stackPtr + offsetof(dlp::kernels::gemmParams, maskFP16)]);
        }
    }
}

template<utils::kernelInstrType KType>
void
jitF32FP16_GEMM<KType>::initializeAccumulators(utils::generatorParams& params)
{
    // Zero out F32 accumulator registers
    if constexpr (Traits::isAVX512) {
        vpxord(RegType(cRegIdx), RegType(cRegIdx), RegType(cRegIdx));
    }

    for (iter_t i = 1; i < cReg; i++) {
        vmovdqa32(RegType(cRegIdx + i), RegType(cRegIdx));
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16_GEMM<KType>::generateIrLoop(utils::generatorParams& params)
{
    initializeAccumulators(params);

    inLocalLabel();

    if (params.alphaScalingType != dlp::kernel_frame::scalingType::zero) {
        mov(regBptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, b)]);

        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kIterBP)]);
        test(regKIter, regKIter);
        je(".F32FP16_CONSIDKLEFT", T_NEAR);

        L(".F32FP16_LOOPKITER");
        RETURN_IF_ERROR(kUnroll(params.K_UNROLL, false));
        dec(regKIter);
        jne(".F32FP16_LOOPKITER", T_NEAR);

        L(".F32FP16_CONSIDKLEFT");
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeft)]);
        test(regKIter, regKIter);
        je(".F32FP16_POSTACCUM", T_NEAR);

        L(".F32FP16_KLEFTLOOP");
        RETURN_IF_ERROR(kUnroll(1, false));
        dec(regKIter);
        jne(".F32FP16_KLEFTLOOP", T_NEAR);

        L(".F32FP16_POSTACCUM");
    }

    RETURN_IF_ERROR(generatePostOps(params));

    vzeroupper();
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16_GEMM<KType>::loadBValues()
{
    // Load B as FP16 (YMM, 16 FP16 = 32 bytes), convert to F32 (ZMM, 16 F32)
    // Packed B layout: each k-row has NR=64 FP16 elements = 128 bytes
    // We load 4 groups of 16 FP16 → 4 ZMM of F32
    for (iter_t i = 0; i < bFullReg; i++) {
        // Load 16 FP16 values (32 bytes) into YMM using temporary aRegIdx
        vmovdqu16(Ymm(aRegIdx),
                  ptr[regBptr + i * FP16_PER_YMM * FP16_ELEM_SIZE]);

        // Convert 16 FP16 → 16 F32 into ZMM B register
        vcvtph2ps(Zmm(bRegIdx + i), Ymm(aRegIdx));
    }

    if (useMask && bMaskReg > 0) {
        int maskRegIndex = bRegIdx + bFullReg;
        if (maskRegIndex >= numRegs) {
            return dlp::jit::jitGeneratorError::badKernelInfo;
        }

        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            // Masked load of FP16, convert to F32
            vmovdqu16(Ymm(aRegIdx) | k3 | T_z,
                      ptr[regBptr + bFullReg * FP16_PER_YMM * FP16_ELEM_SIZE]);
            vcvtph2ps(Zmm(maskRegIndex), Ymm(aRegIdx));
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16_GEMM<KType>::storeResult()
{
    mov(regTmpCptr, regCPtr);
    return storeResultF32();
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16_GEMM<KType>::storeResultF32()
{
    // Store F32 accumulators using vmovups
    for (iter_t i = 0; i < MR; i++) {
        for (iter_t j = 0; j < bFullReg; j++) {
            vmovups(ptr[regTmpCptr + j * RegBytes],
                    RegType(cRegIdx + i * bReg + j));
        }
        if (bMaskReg > 0) {
            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                vmovups(ptr[regTmpCptr + bFullReg * RegBytes] | k3,
                        RegType(cRegIdx + i * bReg + bFullReg));
            }
        }
        add(regTmpCptr, regRsC);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16_GEMM<KType>::scaleAlpha()
{
    int alphaRegIdx = aRegIdx;

    // Load F32 alpha and broadcast
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, alpha)]);
    vbroadcastss(RegType(alphaRegIdx), ptr[regTmp1]);

    // Scale all accumulators with alpha using vmulps (F32)
    for (iter_t i = 0; i < cReg; i++) {
        vmulps(Zmm(cRegIdx + i), Zmm(cRegIdx + i), Zmm(alphaRegIdx));
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16_GEMM<KType>::scaleBeta()
{
    int betaRegIdx = aRegIdx;

    // Load F32 beta and broadcast
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, beta)]);
    vbroadcastss(RegType(betaRegIdx), ptr[regTmp1]);
    mov(regTmpCptr, regCPtr);

    // Load existing F32 C values, scale by beta, add to accumulators
    for (iter_t i = 0; i < MR; i++) {
        for (iter_t j = 0; j < bFullReg; j++) {
            vmovups(RegType(bRegIdx + j), ptr[regTmpCptr + j * RegBytes]);
            vmulps(Zmm(bRegIdx + j), Zmm(bRegIdx + j), Zmm(betaRegIdx));
            vaddps(Zmm(cRegIdx + i * bReg + j), Zmm(cRegIdx + i * bReg + j),
                   Zmm(bRegIdx + j));
        }
        if (bMaskReg > 0) {
            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                vmovups(RegType(bRegIdx + bFullReg) | k3 | T_z,
                        ptr[regTmpCptr + bFullReg * RegBytes]);
                vmulps(Zmm(bRegIdx + bFullReg), Zmm(bRegIdx + bFullReg),
                       Zmm(betaRegIdx));
                vaddps(Zmm(cRegIdx + i * bReg + bFullReg),
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
jitF32FP16_GEMM<KType>::generatePostOps(utils::generatorParams& params)
{
    inLocalLabel();

    if (params.alphaScalingType != dlp::kernel_frame::scalingType::one) {
        RETURN_IF_ERROR(scaleAlpha());
    }

    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
    test(regTmp1, regTmp1);
    je(".F32FP16_STORE_NO_POSTOPS", T_NEAR);

    if (!params.kernelOps.empty()) {
        gen::kernelOpsHandler kOpsHandler(this, params.kType);

        if (params.betaScalingType != dlp::kernel_frame::scalingType::zero) {
            RETURN_IF_ERROR(scaleBeta());
        }

        RETURN_IF_ERROR((kOpsHandler.generateKernelOps(
            params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemm, MR, NR,
            useMask, numMaskRegs, cRegIdx, cReg)));

        kOpsHandler.generateKernelOpsAttributes();

        RETURN_IF_ERROR(storeResult());
        jmp(".F32FP16_AFTER_STORE", T_NEAR);
    }

    L(".F32FP16_STORE_NO_POSTOPS");

    if (params.betaScalingType != dlp::kernel_frame::scalingType::zero) {
        RETURN_IF_ERROR(scaleBeta());
    }

    RETURN_IF_ERROR(storeResult());

    L(".F32FP16_AFTER_STORE");

    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16_GEMM<KType>::generateKernel(utils::generatorParams& params)
{
    MR          = params.MR;
    NR          = params.NR;
    useMask     = params.useMask;
    numMaskRegs = params.numMaskRegs;
    c_downscale = params.c_downscale;

    RETURN_IF_ERROR(allocateRegisters());

    {
        Xbyak::util::StackFrame stackFrame(this, 1, 13, 0);
        initializeStackFrame(stackFrame);
        initializeParameters(params.mLoop);

        if (params.mLoop) {
            RETURN_IF_ERROR(generateMLoop(params));
        } else {
            RETURN_IF_ERROR(generateIrLoop(params));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16_GEMM<KType>::generateMLoop(utils::generatorParams& params)
{
    test(regMiter, regMiter);
    je(".F32FP16_MLOOP_END", T_NEAR);

    L(".F32FP16_MLOOP_START");

    RETURN_IF_ERROR(generateIrLoop(params));

    RETURN_IF_ERROR(moveCPtr());

    mov(regTmpAptr, regAPtr);
    lea(regTmpAptr, ptr[regTmpAptr + regTmp3]);
    mov(regAPtr, regTmpAptr);

    lea(regTmp2, ptr[regTmp2 + MR]);
    mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, post_op_c_i)],
        regTmp2);

    dec(regMiter);
    jne(".F32FP16_MLOOP_START", T_NEAR);

    L(".F32FP16_MLOOP_END");

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16_GEMM<KType>::broadcastAFMAwithB(bool isKRemainder)
{
    for (iter_t i = 0; i < MR; i++) {
        // Broadcast F32 A element to all lanes using vbroadcastss
        vbroadcastss(RegType(aRegIdx), ptr[regTmpAptr]);

        // FMA: C += A * B using vfmadd231ps (F32 FMA)
        for (iter_t j = 0; j < bReg; j++) {
            vfmadd231ps(RegType(cRegIdx + i * bReg + j), RegType(aRegIdx),
                        RegType(bRegIdx + j));
        }

        // Advance A pointer by rs_a (already scaled to bytes)
        add(regTmpAptr, regRsA);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16_GEMM<KType>::kUnroll(int unroll, bool isKRemainder)
{
    for (iter_t p = 0; p < unroll; p++) {
        mov(regTmp1, regTmpAptr);

        RETURN_IF_ERROR(loadBValues());

        // CRITICAL: B stride is FP16 (2 bytes per element)
        // NR=64 elements × 2 bytes = 128 bytes per k-row
        add(regBptr, regRsB);

        RETURN_IF_ERROR(broadcastAFMAwithB(isKRemainder));

        // Advance A pointer to next column (cs_a, already scaled to bytes)
        lea(regTmpAptr, ptr[regTmp1 + regCsA]);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitF32FP16_GEMM<KType>::initializeStackFrame(
    Xbyak::util::StackFrame& stackFrame)
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
dlp::jit::jitGeneratorError
jitF32FP16_GEMM<KType>::moveCPtr()
{
    if (MR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    mov(regTmp1, regRsC);
    imul(regTmp1, regTmp1, MR);
    add(regCPtr, regTmp1);

    return dlp::jit::jitGeneratorError::success;
}

} // namespace amdzen::gen

// Explicit template instantiation
template class amdzen::gen::jitF32FP16_GEMM<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
