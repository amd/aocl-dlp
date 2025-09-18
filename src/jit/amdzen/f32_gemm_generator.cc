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

#include <functional>
#include <memory>

#include "aocl_dlp_config.h"

#include "f32_gemm_generator.hh"
#include "jit_register/jit_register.hh"

namespace amdzen::GEMMcodeGenerator {

using namespace Xbyak;

template<utils::kernelInstrType KType>
jitGEMMF32<KType>::jitGEMMF32(void* buffer, size_t bufferSize)
    : Xbyak::CodeGenerator(bufferSize, buffer)
{
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::allocateReg()
{
    // check if MR, NR, k_unroll are valid
    if (!((MR > 0))) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
    int nElemsPerReg = RegBytes / sizeof(float);
    bFullReg         = ((NR) / nElemsPerReg);
    bMaskReg         = (useMask ? 1 : 0);
    bReg             = bFullReg + bMaskReg;
    cReg             = MR * bReg;
    aReg             = numRegs - cReg - bReg;

    // Check if we have enough registers
    if (aReg < 1) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
    cRegIdx = numRegs - cReg;
    bRegIdx = cRegIdx - bReg;
    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        aReg = 1;
        if (useMask) {
            maskRegIdx = 0;
            aRegIdx    = maskRegIdx + 1;
        } else {
            aRegIdx = 0;
        }

        int totalRegsNeeded = cReg + bReg + aReg + (useMask ? 1 : 0);
        if (totalRegsNeeded > numRegs || aReg < 1) {
            return dlp::jit::jitGeneratorError::badKernelInfo;
        }
    } else {
        aRegIdx = 0;
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMMF32<KType>::initializeParameters(bool addIrLoop)
{
    if (addIrLoop) {
        // Move A and C pointers to stack so
        // they can be accessed at the start or end of IR-loop
        mov(regAPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);
        mov(regMiter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, mIter)]);

    } else {
        mov(regTmpAptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);
    }
    mov(regCPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, c)]);
    mov(regRsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsA)]);
    mov(regCsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, csA)]);
    mov(regRsB, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsB)]);
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsC)]);

    lea(regRsA, ptr[regRsA * sizeof(float)]);
    lea(regCsA, ptr[regCsA * sizeof(float)]);
    lea(regRsB, ptr[regRsB * sizeof(float)]);
    lea(regRsC, ptr[regRsC * sizeof(float)]);

    mov(regTmpCptr, regCPtr);

    if (useMask) {
        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            kmovw(k3,
                  ptr[stackPtr + offsetof(dlp::kernels::gemmParams, maskF32)]);
        } else if constexpr (KType
                             == utils::kernelInstrType::avx512_ymm_32_reg) {
            kmovb(
                k3,
                ptr[stackPtr + offsetof(dlp::kernels::gemmParams, maskF32_8)]);
        } else if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            lea(regTmp1,
                ptr[stackPtr + offsetof(dlp::kernels::gemmParams, maskArray)]);
            vmovdqu(Ymm(maskRegIdx), ptr[regTmp1]);
        } else {
            // Kept for future use
        }
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::loadBValues()
{
    using RegType = typename Traits::RegType;
    for (int i = 0; i < bFullReg; i++) {
        vmovups(RegType(bRegIdx + i), ptr[regBptr + i * RegBytes]);
    }
    if (bMaskReg > 0) {
        if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            vmaskmovps(Ymm(bRegIdx + bFullReg), Ymm(maskRegIdx),
                       ptr[regBptr + bFullReg * RegBytes]);
        } else {
            // For AVX512_zmm and _ymm, mask instruction is same.
            vmovups(RegType(bRegIdx + bFullReg) | k3,
                    ptr[regBptr + bFullReg * RegBytes]);
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::BroadcastAFMAwithB()
{
    using RegType = typename Traits::RegType;
    for (int i = 0; i < MR; i++) {
        vbroadcastss(RegType(aRegIdx), ptr[regTmpAptr]);
        add(regTmpAptr, regRsA);
        for (int j = 0; j < bReg; j++) {
            vfmadd231ps(RegType(cRegIdx + i * bReg + j), RegType(bRegIdx + j),
                        RegType(aRegIdx));
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::kernelUnroll(int unroll)
{
    dlp::jit::jitGeneratorError err = dlp::jit::jitGeneratorError::error;
    // Unroll the kernel loop
    for (int p = 0; p < unroll; p++) {
        // copy A ptr to another register
        mov(regTmp1, regTmpAptr);
        // load B regs
        RETURN_IF_ERROR(loadBValues());
        add(regBptr, regRsB);
        RETURN_IF_ERROR(BroadcastAFMAwithB());
        lea(regTmpAptr, ptr[regTmp1 + regCsA]);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::storeResult()
{
    using RegType = typename Traits::RegType;
    mov(regTmpCptr, regCPtr);
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < bFullReg; j++) {
            // Regular store
            vmovups(ptr[regTmpCptr + j * RegBytes],
                    RegType(cRegIdx + i * bReg + j));
        }
        if (bMaskReg > 0) {
            if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
                vmaskmovps(ptr[regTmpCptr + bFullReg * RegBytes],
                           Ymm(maskRegIdx), Ymm(cRegIdx + i * bReg + bFullReg));
            } else {
                vmovups(ptr[regTmpCptr + bFullReg * RegBytes] | k3,
                        RegType(cRegIdx + i * bReg + bFullReg));
            }
        }
        add(regTmpCptr, regRsC);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMMF32<KType>::initializeStackFrame(Xbyak::util::StackFrame& stackFrame)
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
jitGEMMF32<KType>::regInit()
{
    using RegType = typename Traits::RegType;
    vxorps(RegType(cRegIdx), RegType(cRegIdx), RegType(cRegIdx));
    for (int i = 1; i < cReg; i++) {
        vmovaps(RegType(cRegIdx + i), RegType(cRegIdx));
    }
}

template<utils::kernelInstrType KType>
void
jitGEMMF32<KType>::moveCPtr()
{
    // Update C pointer for next row : cbuf += m * MR * rs_c
    // lea() doesn't work with non-power-of-2 values, and avx2
    // doesn't support imul. Hence, decomposing m value to power
    // of 2 to handle all the cases commonly. Let's say if MR = 13
    // then we can represent it as 8 + 4 + 1
    int m_val       = MR;
    int power2scale = 1;
    while (m_val > 0) {
        if (m_val & 1) {
            lea(regCPtr, ptr[regCPtr + power2scale * regRsC]);
        }
        m_val >>= 1;
        power2scale <<= 1;
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::scaleAlpha()
{
    using RegType            = typename Traits::RegType;
    int          alphaRegIdx = aRegIdx;
    Xbyak::Reg64 alphaReg    = regTmp1;
    mov(alphaReg, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, alpha)]);
    vbroadcastss(RegType(alphaRegIdx), ptr[alphaReg]);
    for (int i = 0; i < cReg; i++) {
        vmulps(RegType(cRegIdx + i), RegType(cRegIdx + i),
               RegType(alphaRegIdx));
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::scaleBeta()
{
    using RegType           = typename Traits::RegType;
    Xbyak::Reg64 betaReg    = regTmp1;
    int          betaRegIdx = aRegIdx;
    mov(betaReg, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, beta)]);
    vbroadcastss(RegType(betaRegIdx), ptr[betaReg]);
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < bFullReg; j++) {
            vmovups(RegType(bRegIdx + j), ptr[regTmpCptr + j * RegBytes]);
            vfmadd231ps(RegType(cRegIdx + i * bReg + j), RegType(betaRegIdx),
                        RegType(bRegIdx + j));
        }
        if (bMaskReg > 0) {
            if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
                // Handle masked C values for fringe elements using dedicated
                // mask register
                vmaskmovps(Ymm(bRegIdx + bFullReg), Ymm(maskRegIdx),
                           ptr[regTmpCptr + bFullReg * RegBytes]);
                vfmadd231ps(Ymm(cRegIdx + i * bReg + bFullReg),
                            Ymm(bRegIdx + bFullReg), Ymm(betaRegIdx));
            } else {
                vmovups(RegType(bRegIdx + bFullReg) | k3,
                        ptr[regTmpCptr + bFullReg * RegBytes]);
                vfmadd231ps(RegType(cRegIdx + i * bReg + bFullReg),
                            RegType(betaRegIdx), RegType(bRegIdx + bFullReg));
            }
        }
        add(regTmpCptr, regRsC);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::generateIrLoop(utils::generatorParams& params)
{
    inLocalLabel();
    // calculate and load pointers
    if (params.mLoop) {
        L(".BLOOP6X64I");
        // Move A and C pointers to stack so
        // they can be accessed at the start or end of IR-loop
        mov(regTmpAptr, regAPtr);
    }
    mov(regBptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, b)]);

    // zero out accumulators
    regInit();

    // Generate K-loop
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kIter)]);
    test(regKIter, regKIter);
    je(".BCONSIDKLEFT", T_NEAR);

    // Kernel unroll loop
    L(".BLOOPKITER");
    RETURN_IF_ERROR(kernelUnroll(params.K_UNROLL));
    dec(regKIter); // i -= 1
    jne(".BLOOPKITER", T_NEAR);

    L(".BCONSIDKLEFT");
    // load k_left
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeft)]);
    test(regKIter, regKIter);
    je(".BPOSTACCUM", T_NEAR);

    RETURN_IF_ERROR(kernelUnroll(1));

    L(".BPOSTACCUM");

    if (params.alphaScalingType == dlp::kernel_frame::scalingType::one) {
        // skip alpha scaling if alpha is 1
    } else {
        // alpha scaling
        RETURN_IF_ERROR(scaleAlpha());
    }

    // To-Do: add support for beta scaling if beta is 1 using vaddps
    if (params.betaScalingType == dlp::kernel_frame::scalingType::zero) {
        // skip beta scaling if beta is 0
    } else {
        // beta scaling
        RETURN_IF_ERROR(scaleBeta());
    }

    // check if is_last_k is set
    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(lpgemm_post_op_attr, is_last_k)]);
    test(regTmp1, regTmp1);
    je(label_store_result, T_NEAR);

    // Create and set up kernelOphandler if there are post-ops
    if (!params.kernelOps.empty()) {
        gen::kernelOpsHandler kernelOpsHandler(this, params.MR, params.NR,
                                               params.useMask, cRegIdx, cReg,
                                               params.kType);

        RETURN_IF_ERROR(
            (kernelOpsHandler.generateKernelOps(params.kernelOps, stackPtr)));

        // The gelu constants are embedded within the generated JIT kernel.
        // Otherwise a bug was observed whereby the address of gelu constants
        // inside the class turned out to be beyond what JIT can access.
        kernelOpsHandler.generateKernelOpsAttributes();
    }

    L(label_store_result);
    // store C
    RETURN_IF_ERROR(storeResult());

    if (params.mLoop) {
        // get A pointer from stack
        // add psA to A pointer
        mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, psA)]);
        lea(regTmp1, ptr[regTmp1 * sizeof(float)]);
        lea(regAPtr, ptr[regAPtr + regTmp1]);

        // Update post_op_attr c_i. kernelOpsAttr is not a pointer, so adding
        // two offsets at the same time is safe.
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
        add(regTmp1, MR);
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, post_op_c_i)],
            regTmp1);

        moveCPtr();

        // decrement m_iter
        dec(regMiter);

        jne(".BLOOP6X64I", T_NEAR);
    }
    vzeroupper();
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

// Generate kernel for F32 operations
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::generateKernel(utils::generatorParams& params)
{
    MR      = params.MR;
    NR      = params.NR;
    useMask = params.useMask;

    RETURN_IF_ERROR(allocateReg());

    // There are 14 general purpose(64 bit) registers.
    // StackFrame manages these registers, since we are using
    // one register for the input parameter of the function,
    // the rest are used as scratch registers to store variables like
    // pointers, strides, counters, etc.
    // Note that all the scratch registers allocated by the stack frame
    // need not be used by the kernel.
    // Putting inside a scope so that some tables can be generated post
    // the ret instr. StackFrame inserts a ret instr in its destructor.
    Xbyak::util::StackFrame stackFrame(this, 1, 13, 0);
    initializeStackFrame(stackFrame);
    initializeParameters(params.mLoop);

    RETURN_IF_ERROR(generateIrLoop(params));

    return dlp::jit::jitGeneratorError::success;
}

} // namespace amdzen::GEMMcodeGenerator

// Explicit template instantiations for classes
template class amdzen::GEMMcodeGenerator::jitGEMMF32<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
template class amdzen::GEMMcodeGenerator::jitGEMMF32<
    amdzen::utils::kernelInstrType::avx512_ymm_32_reg>;
template class amdzen::GEMMcodeGenerator::jitGEMMF32<
    amdzen::utils::kernelInstrType::avx2_ymm_16_reg>;
