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
            + offsetof(lpgemm_post_op_attr, post_op_c_i)]);

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
            // For AVX512 ZMM: load 32-bit mask for FP16 elements (32 per ZMM)
            kmovd(k3,
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
            vmovdqu16(RegType(maskRegIndex) | k3 | T_z,
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
                vmovdqu16(ptr[regTmpCptr + bFullReg * RegBytes] | k3,
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
                vmovdqu16(RegType(bRegIdx + bFullReg) | k3 | T_z,
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
    // Handle alpha scaling (skip for both alpha=1 and alpha=0)
    if (params.alphaScalingType != dlp::kernel_frame::scalingType::one
        && params.alphaScalingType != dlp::kernel_frame::scalingType::zero) {
        RETURN_IF_ERROR(scaleAlpha());
    }

    // Handle beta scaling
    if (params.betaScalingType != dlp::kernel_frame::scalingType::zero) {
        RETURN_IF_ERROR(scaleBeta());
    }

    // Store results
    RETURN_IF_ERROR(storeResult());

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
            + offsetof(lpgemm_post_op_attr, post_op_c_i)],
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
