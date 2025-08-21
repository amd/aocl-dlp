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

#include "avx512_generator.hh"
#include "jit_register/jit_register.hh"

namespace amdzen::avx512gen {

using namespace Xbyak;

jitAVX512::jitAVX512(void* buffer, size_t bufferSize)
    : Xbyak::CodeGenerator(bufferSize,
                           buffer) // Call the base class constructor
{
}

template<typename accumType>
dlp::jit::jitGeneratorError
jitAVX512::allocateReg()
{
    // check if MR, NR, k_unroll are valid
    if (!((MR > 0))) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
    int nElemsPerReg = RegBytes / sizeof(accumType);
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
    aRegIdx = 0;

    return dlp::jit::jitGeneratorError::success;
}

template<typename aType, typename bType, typename cType>
void
jitAVX512::initializeParameters(bool addIrLoop)
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

    lea(regRsA, ptr[regRsA * sizeof(aType)]);
    lea(regCsA, ptr[regCsA * sizeof(aType)]);
    lea(regRsB, ptr[regRsB * sizeof(bType)]);
    lea(regRsC, ptr[regRsC * sizeof(cType)]);

    mov(regTmpCptr, regCPtr);

    if (useMask) {
        kmovw(k3, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, maskF32)]);
    }
}

template<typename bType>
dlp::jit::jitGeneratorError
jitAVX512::loadBValuesZmm()
{
    return dlp::jit::jitGeneratorError::notSupported;
}

template<typename aType, typename bType, typename accumType>
dlp::jit::jitGeneratorError
jitAVX512::BroadcastAFMAwithBZmm()
{
    return dlp::jit::jitGeneratorError::notSupported;
}

template<typename aType, typename bType, typename accumType>
dlp::jit::jitGeneratorError
jitAVX512::kernelUnrollZmm(int unroll)
{
    dlp::jit::jitGeneratorError err = dlp::jit::jitGeneratorError::error;
    // Unroll the kernel loop
    for (int p = 0; p < unroll; p++) {
        // copy A ptr to another register
        mov(regTmp1, regTmpAptr);
        // load B regs
        RETURN_IF_ERROR((loadBValuesZmm<bType>()));
        add(regBptr, regRsB);
        RETURN_IF_ERROR((BroadcastAFMAwithBZmm<aType, bType, accumType>()));
        lea(regTmpAptr, ptr[regTmp1 + regCsA]);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<typename accumType>
dlp::jit::jitGeneratorError
jitAVX512::cvtAccToFloat()
{
    return dlp::jit::jitGeneratorError::notSupported;
}

template<typename accumType, typename cType>
dlp::jit::jitGeneratorError
jitAVX512::storeResult()
{
    return dlp::jit::jitGeneratorError::notSupported;
}

void
jitAVX512::initializeStackFrame(Xbyak::util::StackFrame& stackFrame)
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

void
jitAVX512::regInitZmm()
{
    vxorps(Zmm(cRegIdx), Zmm(cRegIdx), Zmm(cRegIdx));
    for (int i = 1; i < cReg; i++) {
        vmovaps(Zmm(cRegIdx + i), Zmm(cRegIdx));
    }
}

void
jitAVX512::moveCPtr()
{
    // get C pointer from stack
    // add MR to C pointer

    int temp = MR;
    // split MR into powers of 2 and then multiply with regRsC
    // and add to C pointer
    // for example MR = 13 = 8 + 4 + 1
    // MR = 2^3 + 2^2 + 2^0
    int power = 1;
    while (temp > 0) {
        if (temp & 1) {
            lea(regCPtr, ptr[regCPtr + power * regRsC]);
        }
        temp >>= 1;
        power <<= 1;
    }
}

template<typename accumType>
dlp::jit::jitGeneratorError
jitAVX512::scaleAlpha()
{
    // placeholder for alpha scaling
    return dlp::jit::jitGeneratorError::notSupported;
}

template<typename accumType, typename cType>
dlp::jit::jitGeneratorError
jitAVX512::scaleBeta()
{
    // placeholder for beta scaling
    return dlp::jit::jitGeneratorError::notSupported;
}

template<typename aType, typename bType, typename cType, typename accumType>
dlp::jit::jitGeneratorError
jitAVX512::generateIrLoop(utils::generatorParams& params)
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
    regInitZmm();

    // Generate K-loop
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kIter)]);
    test(regKIter, regKIter);
    je(".BCONSIDKLEFT", T_NEAR);

    // Kernel unroll loop
    L(".BLOOPKITER");
    RETURN_IF_ERROR(
        (kernelUnrollZmm<aType, bType, accumType>(params.K_UNROLL)));
    dec(regKIter); // i -= 1
    jne(".BLOOPKITER", T_NEAR);

    L(".BCONSIDKLEFT");
    // load k_left
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeft)]);
    test(regKIter, regKIter);
    je(".BPOSTACCUM", T_NEAR);

    RETURN_IF_ERROR((kernelUnrollZmm<aType, bType, accumType>(1)));

    L(".BPOSTACCUM");

    if (params.alphaScalingType == dlp::kernel_frame::scalingType::one) {
        // skip alpha scaling if alpha is 1
    } else {
        // alpha scaling
        RETURN_IF_ERROR((scaleAlpha<accumType>()));
    }

    // To-Do: add support for beta scaling if beta is 1 using vaddps
    if (params.betaScalingType == dlp::kernel_frame::scalingType::zero) {
        // skip beta scaling if beta is 0
    } else {
        // beta scaling
        RETURN_IF_ERROR((scaleBeta<accumType, cType>()));
    }

    // convert acc type to post-op type
    RETURN_IF_ERROR((cvtAccToFloat<accumType>()));

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
    RETURN_IF_ERROR((storeResult<accumType, cType>()));

    if (params.mLoop) {
        // get A pointer from stack
        // add psA to A pointer
        mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, psA)]);
        lea(regTmp1, ptr[regTmp1 * sizeof(aType)]);
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

// Template function that takes the datatype as a template parameter
template<dlp::kernel_frame::kernelDatatype KDT>
dlp::jit::jitGeneratorError
jitAVX512::generateKernel(utils::generatorParams& params)
{
    MR      = params.MR;
    NR      = params.NR;
    useMask = params.useMask;

    using types     = amdzen::traits::kernel_types<KDT>;
    using aType     = typename types::aType;
    using bType     = typename types::bType;
    using cType     = typename types::cType;
    using accumType = typename types::accumType;

    // Assuming post-ops are always applied on float type

    RETURN_IF_ERROR((allocateReg<accumType>()));

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
    initializeParameters<aType, bType, cType>(params.mLoop);

    RETURN_IF_ERROR((generateIrLoop<aType, bType, cType, accumType>(params)));

    return dlp::jit::jitGeneratorError::success;
}

// Instantiating for a concrete type, to enable header and .cc file
// separation of templated function.
template dlp::jit::jitGeneratorError
jitAVX512::generateKernel<dlp::kernel_frame::kernelDatatype::f32f32f32of32>(
    utils::generatorParams& params);

template<>
dlp::jit::jitGeneratorError
jitAVX512::loadBValuesZmm<float>()
{
    for (int i = 0; i < bFullReg; i++) {
        vmovups(Zmm(bRegIdx + i), ptr[regBptr + i * RegBytes]);
    }
    if (bMaskReg > 0) {
        vmovups(Zmm(bRegIdx + bFullReg) | k3,
                ptr[regBptr + bFullReg * RegBytes]);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512::BroadcastAFMAwithBZmm<float, float, float>()
{
    for (int i = 0; i < MR; i++) {
        vbroadcastss(Zmm(aRegIdx), ptr[regTmpAptr]);
        add(regTmpAptr, regRsA);
        for (int j = 0; j < bReg; j++) {
            vfmadd231ps(Zmm(cRegIdx + i * bReg + j), Zmm(bRegIdx + j),
                        Zmm(aRegIdx));
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512::cvtAccToFloat<float>()
{
    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512::storeResult<float, float>()
{
    mov(regTmpCptr, regCPtr);
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < bFullReg; j++) {
            // Regular store
            vmovups(ptr[regTmpCptr + j * RegBytes],
                    Zmm(cRegIdx + i * bReg + j));
        }
        if (bMaskReg > 0) {
            vmovups(ptr[regTmpCptr + bFullReg * RegBytes] | k3,
                    Zmm(cRegIdx + i * bReg + bFullReg));
        }
        add(regTmpCptr, regRsC);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512::scaleAlpha<float>()
{
    int          alphaRegIdx = aRegIdx;
    Xbyak::Reg64 alphaReg    = regTmp1;
    mov(alphaReg, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, alpha)]);
    vbroadcastss(Zmm(alphaRegIdx), ptr[alphaReg]);
    for (int i = 0; i < cReg; i++) {
        vmulps(Zmm(cRegIdx + i), Zmm(cRegIdx + i), Zmm(alphaRegIdx));
    }
    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512::scaleBeta<float, float>()
{
    Xbyak::Reg64 betaReg    = regTmp1;
    int          betaRegIdx = aRegIdx;
    mov(betaReg, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, beta)]);
    vbroadcastss(Zmm(betaRegIdx), ptr[betaReg]);
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < bFullReg; j++) {
            vmovups(Zmm(bRegIdx + j), ptr[regTmpCptr + j * RegBytes]);
            vfmadd231ps(Zmm(cRegIdx + i * bReg + j), Zmm(betaRegIdx),
                        Zmm(bRegIdx + j));
        }
        if (bMaskReg > 0) {
            vmovups(Zmm(bRegIdx + bFullReg) | k3,
                    ptr[regTmpCptr + bFullReg * RegBytes]);
            vfmadd231ps(Zmm(cRegIdx + i * bReg + bFullReg), Zmm(betaRegIdx),
                        Zmm(bRegIdx + bFullReg));
        }
        add(regTmpCptr, regRsC);
    }
    return dlp::jit::jitGeneratorError::success;
}

} // namespace amdzen::avx512gen
