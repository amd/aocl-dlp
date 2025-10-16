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
#include <iostream>
#include <memory>
#include <string>

#include "avx2_generator.hh"
#include "classic/dlp_base_types.h"
#include "jit_register/jit_register.hh"

namespace amdzen::avx2gen {

using namespace Xbyak;

jitAVX2::jitAVX2(void* buffer, size_t bufferSize)
    : Xbyak::CodeGenerator(bufferSize, buffer)
{
}

template<typename accumType>
dlp::jit::jitGeneratorError
jitAVX2::allocateReg()
{
    // Check if MR is valid
    if (MR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    /* AVX2 Register Allocation Strategy (16 YMM registers available:
     * YMM0-YMM15)
     *
     * Allocation Priority (from highest to lowest register indices):
     * 1. C-registers (accumulation): MR×bReg registers for result accumulation
     * 2. B-registers (data): bReg registers for B matrix values (NR elements)
     * 3. A-registers (broadcast): 2 registers max for A matrix broadcasting
     * 4. Mask register: 1 register for fringe case masking (when needed)
     *
     * Rationale:
     * - C-registers need highest count (MR×bReg) and use high indices for
     * better encoding
     * - B-registers reused across MR rows, allocated just before C-registers
     * - A-registers reuse with alternating pattern (only 2 needed), use low
     * indices
     * - Mask register positioned after A-registers for fringe case handling
     */

    int nElemsPerReg = RegBytes / sizeof(accumType); // 8 floats per YMM256
    bFullReg         = ((NR) / nElemsPerReg); // Full registers for NR elements
    bMaskReg         = (useMask ? 1 : 0);     // +1 register for fringe masking
    bReg             = bFullReg + bMaskReg;   // Total B registers needed
    cReg             = (MR * bReg); // C registers: MR rows × bReg cols

    // Allocate from high to low indices: C-regs use YMM15 down, B-regs before
    // C-regs
    cRegIdx = numRegs - cReg; // C-registers: highest indices
    bRegIdx = cRegIdx - bReg; // B-registers: just below C-regs

    // A-registers: minimal count for broadcast reuse, use lowest indices
    // TODO: Needs a better register allocation strategy, due to non
    // availability of mask registers in AVX2.
    aReg = 1;

    // Mask register: positioned after A-registers for fringe N-dimension
    // handling
    if (useMask) {
        ymmMaskIdx = 0;
    }
    aRegIdx = ymmMaskIdx + useMask;

    // Verify sufficient registers: must fit A + B + C + mask (if needed)
    int totalRegsNeeded = cReg + bReg + aReg + (useMask ? 1 : 0);
    if (totalRegsNeeded > numRegs || aReg < 1) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    return dlp::jit::jitGeneratorError::success;
}

template<typename aType, typename bType, typename cType>
void
jitAVX2::initializeParameters(bool isMLoop)
{
    if (isMLoop) {
        // Move A and C pointers to stack for IR-loop access
        mov(regAPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);
        mov(regMiter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, mIter)]);
    } else {
        // For non-M-loop case, initialize regTmpAptr directly from gemmParams
        mov(regTmpAptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);
    }

    mov(regCPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, c)]);
    mov(regRsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsA)]);
    mov(regCsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, csA)]);
    mov(regRsB, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsB)]);
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsC)]);

    // Scale strides by element size for future uses
    lea(regRsA, ptr[regRsA * sizeof(aType)]);
    lea(regCsA, ptr[regCsA * sizeof(aType)]);
    lea(regRsB, ptr[regRsB * sizeof(bType)]);
    lea(regRsC, ptr[regRsC * sizeof(cType)]);

    // Save C pointer value
    mov(regTmpCptr, regCPtr);

    if (useMask) {
        lea(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, maskArray)]);
        vmovdqu(Ymm(ymmMaskIdx), ptr[regTmp1]); // Load all 8 floats (32 bytes)
    }
}

template<typename bType>
dlp::jit::jitGeneratorError
jitAVX2::loadBValuesYmm()
{
    // Load B matrix values into ymm registers calculated in allocateReg().
    for (int i = 0; i < bFullReg; i++) {
        // Load B values into ymm registers
        vmovups(Ymm(bRegIdx + i), ptr[regBptr + i * RegBytes]);
    }

    if (bMaskReg > 0) {
        // Load B values with masking using vmaskmovps
        vmaskmovps(Ymm(bRegIdx + bFullReg), Ymm(ymmMaskIdx),
                   ptr[regBptr + bFullReg * RegBytes]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<typename aType, typename bType, typename accumType>
dlp::jit::jitGeneratorError
jitAVX2::BroadcastAandComputeFMAwithYmm()
{
    // Atmost 2 registers can be used to broadcast A values with same B.
    for (int i = 0; i < MR; i++) {
        vbroadcastss(Ymm(aRegIdx), ptr[regTmpAptr]);
        // Move a to next row : (float *)a = a + (i * rs_a)
        add(regTmpAptr, regRsA);
        for (int j = 0; j < bReg; j++) {
            vfmadd231ps(Ymm(cRegIdx + i * bReg + j), Ymm(aRegIdx),
                        Ymm(bRegIdx + j));
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<typename aType, typename bType, typename cType, typename accumType>
dlp::jit::jitGeneratorError
jitAVX2::kernelUnrollYmm(int unroll)
{
    // Unroll the kernel loop
    for (int k = 0; k < unroll; ++k) {
        // Load B values for this K iteration
        RETURN_IF_ERROR((loadBValuesYmm<bType>()));
        // Move to next K row in B : (float*)b += rsB
        // TODO: This accounts only for F32, when bf16 inputs are
        // used need to account for the correct strides there.
        add(regBptr, regRsB);
        // Save A pointer to regTmp1
        mov(regTmp2, regTmpAptr);
        // Broadcast A and perform FMA
        RETURN_IF_ERROR(
            (BroadcastAandComputeFMAwithYmm<aType, bType, accumType>()));
        // Update A pointer to next column
        lea(regTmpAptr, ptr[regTmp2 + regCsA]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<typename aType, typename bType, typename cType, typename accumType>
dlp::jit::jitGeneratorError
jitAVX2::generateIrLoop(utils::generatorParams& params)
{
    inLocalLabel();

    // Main M loop entry - only create m_loop if main Loop(6x16)
    if (params.mLoop) {
        // Beginning of M-Loop
        L(m_loop);
        // Save A pointer for future use
        mov(regTmpAptr, regAPtr);
    }

    // Save B pointer to regBPtr
    mov(regBptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, b)]);

    // Prefetch C buffer before starting computation
    prefetchCBuffer<cType>();

    // Zero accumulation registers
    regInitYmm();

    // Extract the K-loop param
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, k)]);
    test(regKIter, regKIter);
    // if k < k/k_unroll, jump to k_left
    jz(k_rem_loop, T_NEAR);

    L(k_loop); // Start of K-loop : K_iter = k / k_unroll.
    RETURN_IF_ERROR(
        (kernelUnrollYmm<aType, bType, cType, accumType>(params.K_UNROLL)));
    // Decrement K iterator
    dec(regKIter); // k -= 1
    jnz(k_loop, T_NEAR);

    L(k_rem_loop);
    // Perform any necessary operations for the remaining K iterations.
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeft)]);
    test(regKIter, regKIter);
    jz(".alphabetaScale", T_NEAR); // If no k_rem, move to alpha scaling

    RETURN_IF_ERROR((kernelUnrollYmm<aType, bType, cType, accumType>(1)));

    // TODO: To be replaced with the appropriate label that handles
    // post-ops for the GEMM.
    L(".alphabetaScale");
    if (params.alphaScalingType == dlp::kernel_frame::scalingType::one) {
        // skip alpha scaling if alpha is 1
    } else {
        // alpha scaling
        RETURN_IF_ERROR((scaleAlpha<accumType>()));
    }

    if (params.betaScalingType != dlp::kernel_frame::scalingType::zero) {
        // beta scaling
        RETURN_IF_ERROR((scaleBeta<accumType, cType>()));
    }
#if 0 // TODO: Placeholder for beta = 1.0
    if (params.betaScalingType == dlp::kernel_frame::scalingType::one) {
        // Beta = 1.0: Use optimized addition path
        RETURN_IF_ERROR((addCBufferBetaOne<accumType, cType>()));
    }
#endif

    // check if is_last_k is set
    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(lpgemm_post_op_attr, is_last_k)]);
    test(regTmp1, regTmp1);
    je(label_store_result, T_NEAR);

    // Create and set up kernelOphandler if there are post-ops
    if (!params.kernelOps.empty()) {
        gen::kernelOpsHandler kernelOpsHandler(this, params.kType);

        RETURN_IF_ERROR((kernelOpsHandler.generateKernelOps(
            params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemm, params.MR,
            params.NR, params.useMask, cRegIdx, cReg)));

        // The gelu constants are embedded within the generated JIT kernel.
        // Otherwise a bug was observed whereby the address of gelu constants
        // inside the class turned out to be beyond what JIT can access.
        kernelOpsHandler.generateKernelOpsAttributes();
    }

    L(label_store_result);
    RETURN_IF_ERROR((storeResult<accumType, cType>()));

    // Only handle M-loop iteration if mLoop is enabled
    if (params.mLoop) {
        // Update the pointers for the next iteration
        // Update a pointer as abuf = (float*)a + m * ps_a;
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
        // Decrement M iterator
        dec(regMiter); // m -= 1
        jnz(m_loop, T_NEAR);
    }

    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

template<typename accumType, typename cType>
dlp::jit::jitGeneratorError
jitAVX2::storeResult()
{
    return dlp::jit::jitGeneratorError::notSupported;
}

void
jitAVX2::initializeStackFrame(Xbyak::util::StackFrame& stackFrame)
{
    stackPtr = stackFrame.p[0];

    // Assign register variables using StackFrame's scratch registers
    regTmpAptr = stackFrame.t[0];
    regBptr    = stackFrame.t[1];
    regTmpCptr = stackFrame.t[2];
    regRsA     = stackFrame.t[3];
    regCsA     = stackFrame.t[4];
    regRsB     = stackFrame.t[5];
    regRsC     = stackFrame.t[6];
    regAPtr    = stackFrame.t[7];
    regCPtr    = stackFrame.t[8];
    regMiter   = stackFrame.t[9];
    regTmp1    = stackFrame.t[10];
    regTmp2    = stackFrame.t[11];
    regTmp3    = stackFrame.t[12];
}

void
jitAVX2::regInitYmm()
{
    // Zero out all C accumulation registers (YMM4-YMM15)
    vpxor(Ymm(cRegIdx), Ymm(cRegIdx), Ymm(cRegIdx));
    for (int i = 1; i < cReg; ++i) {
        // Last cReg registers are used for storing C,
        // hence checking bounds.
        if (cRegIdx + i <= 15) {
            vmovaps(Ymm(cRegIdx + i), Ymm(cRegIdx));
        }
    }
}

template<typename cType>
void
jitAVX2::prefetchCBuffer()
{
    /* C Buffer Prefetch Strategy
     *
     * Purpose: Prefetch C matrix data into cache before K-loop computation
     * Pattern: Prefetch all MR rows sequentially using T0 hint (all cache
     * levels) Timing: Called right after B pointer setup, before regInitYmm()
     *
     * This matches the static kernel pattern:
     * _mm_prefetch((cbuf + i * rs_c), _MM_HINT_T0) for i = 0 to MR-1
     */

    // Prefetch all MR rows of C buffer
    // regCPtr points to current C block base address
    // Each row is separated by regRsC (row stride in bytes)
    mov(regTmpCptr, regCPtr); // Start from base C pointer
    for (int i = 0; i < MR; i++) {
        prefetcht0(ptr[regTmpCptr]); // Prefetch current row
        if (i < MR - 1) {            // Don't add after last iteration
            add(regTmpCptr, regRsC); // Move to next row
        }
    }
}

void
jitAVX2::moveCPtr()
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

template<typename accumType>
dlp::jit::jitGeneratorError
jitAVX2::scaleAlpha()
{
    int alphaRegIdx = aRegIdx;
    // Load alpha value
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, alpha)]);

    vbroadcastss(Ymm(alphaRegIdx), ptr[regTmp1]);
    // Scale C registers by alpha
    for (int i = 0; i < cReg; i++) {
        vmulps(Ymm(cRegIdx + i), Ymm(cRegIdx + i), Ymm(alphaRegIdx));
    }
    return dlp::jit::jitGeneratorError::success;
}

template<typename accumType, typename cType>
dlp::jit::jitGeneratorError
jitAVX2::scaleBeta()
{
    return dlp::jit::jitGeneratorError::notSupported;
}
#if 0
// TODO: Placeholder support for beta = 1.0
template<typename accumType, typename cType>
dlp::jit::jitGeneratorError
jitAVX2::addCBufferBetaOne()
{
    /* C Buffer Addition Optimization (Beta = 1.0 case)
     *
     * Operation: C_acc = C_acc + C_old (no multiplication needed)
     * This saves the beta broadcast and uses faster VADDPS instead of
     * VFMADD231PS
     */

    mov(regTmpCptr, regCPtr);
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < bFullReg; j++) {
            // Load existing C values
            vmovups(Ymm(bRegIdx + j), ptr[regTmpCptr + j * RegBytes]);
            // C_acc = C_acc + C_old (direct addition, no scaling)
            vaddps(Ymm(cRegIdx + i * bReg + j), Ymm(cRegIdx + i * bReg + j),
                   Ymm(bRegIdx + j));
        }
        if (bMaskReg > 0) {
            // Handle masked C values for fringe elements
            vmaskmovps(Ymm(bRegIdx + bFullReg), Ymm(ymmMaskIdx),
                       ptr[regTmpCptr + bFullReg * RegBytes]);
            vaddps(Ymm(cRegIdx + i * bReg + bFullReg),
                   Ymm(cRegIdx + i * bReg + bFullReg), Ymm(bRegIdx + bFullReg));
        }
        // Move to next row in C matrix
        add(regTmpCptr, regRsC);
    }
    return dlp::jit::jitGeneratorError::success;
}
#endif

template<dlp::kernel_frame::kernelDatatype KDT>
dlp::jit::jitGeneratorError
jitAVX2::generateKernel(utils::generatorParams& params)
{
    MR      = params.MR;
    NR      = params.NR;
    useMask = params.useMask;

    using types     = amdzen::traits::kernel_types<KDT>;
    using aType     = typename types::aType;
    using bType     = typename types::bType;
    using cType     = typename types::cType;
    using accumType = typename types::accumType;

    RETURN_IF_ERROR((allocateReg<accumType>()));

    // StackFrame scope for proper register management
    {
        Xbyak::util::StackFrame stackFrame(this, 1, 13, 0);
        initializeStackFrame(stackFrame);
        // Load parameters from gemmParams structure
        initializeParameters<aType, bType, cType>(params.mLoop);

        // Generate IR loop
        RETURN_IF_ERROR(
            (generateIrLoop<aType, bType, cType, accumType>(params)));
    }

    return dlp::jit::jitGeneratorError::success;
}

// Explicit template instantiations for FP32
template dlp::jit::jitGeneratorError
jitAVX2::generateKernel<dlp::kernel_frame::kernelDatatype::f32f32f32of32>(
    utils::generatorParams& params);

template<>
dlp::jit::jitGeneratorError
jitAVX2::scaleBeta<float, float>()
{
    int betaRegIdx = aRegIdx;

    // Load beta value
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, beta)]);
    vbroadcastss(Ymm(betaRegIdx), ptr[regTmp1]);

    mov(regTmpCptr, regCPtr);
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < bFullReg; j++) {
            // Load existing C values
            vmovups(Ymm(bRegIdx + j), ptr[regTmpCptr + j * RegBytes]);
            // C_acc = C_acc + beta * C_old
            vfmadd231ps(Ymm(cRegIdx + i * bReg + j), Ymm(bRegIdx + j),
                        Ymm(betaRegIdx));
        }
        if (bMaskReg > 0) {
            // Handle masked C values for fringe elements using dedicated mask
            // register
            vmaskmovps(Ymm(bRegIdx + bFullReg), Ymm(ymmMaskIdx),
                       ptr[regTmpCptr + bFullReg * RegBytes]);
            vfmadd231ps(Ymm(cRegIdx + i * bReg + bFullReg),
                        Ymm(bRegIdx + bFullReg), Ymm(betaRegIdx));
        }
        // Move to next row in C matrix
        add(regTmpCptr, regRsC);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX2::storeResult<float, float>()
{
    mov(regTmpCptr, regCPtr);
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < bFullReg; j++) {
            // Store the result from the accumulation register to the C matrix
            vmovups(ptr[regTmpCptr + j * RegBytes],
                    Ymm(cRegIdx + i * bReg + j));
        }
        if (bMaskReg > 0) {
            // Store the masked register for fringe elements
            vmaskmovps(ptr[regTmpCptr + bFullReg * RegBytes], Ymm(ymmMaskIdx),
                       Ymm(cRegIdx + i * bReg + bFullReg));
        }
        // Move to next row in C matrix : cbuf += rs_c
        add(regTmpCptr, regRsC);
    }

    return dlp::jit::jitGeneratorError::success;
}

} // namespace amdzen::avx2gen
