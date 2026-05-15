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

#include "f32f16_gemv_generator.hh"

namespace amdzen::gen {

using namespace Xbyak;

// =============================================================================
// jitF32FP16GEMVM1: M=1 GEMV (y = x * B) - Vector-Matrix multiplication
//
// x is F32 (1×K), B is FP16 (K×N), y is F32 (1×N).
// Strategy: broadcast x[k] as F32, load B row as FP16→F32, FMA in F32.
// NR = 64 (4 ZMMs of 16 F32 elements each).
// K_SUB_ITER = 4 for software pipelining (4 k-steps accumulated separately).
// =============================================================================

template<utils::kernelInstrType KType>
jitF32FP16GEMVM1<KType>::jitF32FP16GEMVM1(void* buffer, size_t bufferSize)
    : Xbyak::CodeGenerator(bufferSize, buffer)
{
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVM1<KType>::allocateRegisters()
{
    nElemsPerReg = F32_PER_ZMM;

    int regsPerPanel = NR / nElemsPerReg; // 64 / 16 = 4

    xReg     = K_SUB_ITER;                // 4 (broadcast x values)
    bReg     = regsPerPanel;              // 4 (B row converted to F32)
    accumReg = regsPerPanel * K_SUB_ITER; // 16 accumulators

    accumBaseIdx = numRegs - accumReg;  // zmm16-zmm31
    xBaseIdx     = accumBaseIdx - xReg; // zmm12-zmm15
    bBaseIdx     = xBaseIdx - bReg;     // zmm8-zmm11
    yBaseIdx     = numRegs - bReg;      // zmm28-zmm31 (post finalAccumulate)

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitF32FP16GEMVM1<KType>::initializeStackFrame(Xbyak::util::StackFrame& frame)
{
    stackPtr    = frame.p[0];
    regBptr     = frame.t[0];
    regXptr     = frame.t[1];
    regYptr     = frame.t[2];
    regTmpYptr  = frame.t[3];
    regNIter    = frame.t[4];
    regKIter    = frame.t[5];
    regKSubIter = frame.t[6];
    regRsB      = frame.t[7];
    regPsB      = frame.t[8];
    regTmp1     = frame.t[9];
    regTmp2     = frame.t[10];
    regIncN     = frame.t[11];
    regIncK     = frame.t[12];
}

template<utils::kernelInstrType KType>
void
jitF32FP16GEMVM1<KType>::initializeParameters(
    [[maybe_unused]] utils::gemvM1GeneratorParams& params)
{
    nElemsPerReg = F32_PER_ZMM;

    mov(regYptr, ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, y)]);
    mov(regRsB, ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, rsB)]);

    // Scale rsB to bytes: B is FP16, 2 bytes per element
    lea(regRsB, ptr[regRsB * FP16_ELEM_SIZE]);
}

template<utils::kernelInstrType KType>
void
jitF32FP16GEMVM1<KType>::regInit(int baseIdx, int numRegs)
{
    vpxord(Zmm(baseIdx), Zmm(baseIdx), Zmm(baseIdx));
    for (iter_t i = 1; i < numRegs; i++) {
        vmovdqa32(Zmm(baseIdx + i), Zmm(baseIdx));
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVM1<KType>::loadMasks()
{
    for (iter_t i = 0; i < utils::NUM_USABLE_MASKS; i++) {
        mask_regs[i] = Xbyak::Opmask(utils::MASK_START_IDX + i);
    }

    // Load N-dimension mask for F32 (16-bit mask for 16 F32 elements per ZMM)
    kmovw(mask_regs[0],
          ptr[stackPtr
              + offsetof(dlp::kernels::gemvM1Params, nmask_fp16_avx512)]);

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVM1<KType>::computeKxNR([[maybe_unused]] bool nMask)
{
    int regsPerPanel = NR / nElemsPerReg;

    for (iter_t ksub = 0; ksub < K_SUB_ITER; ksub++) {
        vbroadcastss(Zmm(xBaseIdx + ksub), ptr[regXptr]);
        add(regXptr, F32_ELEM_SIZE);

        for (iter_t i = 0; i < regsPerPanel; i++) {
            vmovdqu16(Ymm(bBaseIdx + i),
                      ptr[regBptr + i * FP16_PER_YMM * FP16_ELEM_SIZE]);
            vcvtph2ps(Zmm(bBaseIdx + i), Ymm(bBaseIdx + i));

            vfmadd231ps(Zmm(accumBaseIdx + K_SUB_ITER * i + ksub),
                        Zmm(xBaseIdx + ksub), Zmm(bBaseIdx + i));
        }

        add(regBptr, regRsB);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVM1<KType>::compute1xNR([[maybe_unused]] bool nMask)
{
    int regsPerPanel = NR / nElemsPerReg;

    vbroadcastss(Zmm(xBaseIdx), ptr[regXptr]);
    add(regXptr, F32_ELEM_SIZE);

    for (iter_t i = 0; i < regsPerPanel; i++) {
        vmovdqu16(Ymm(bBaseIdx + i),
                  ptr[regBptr + i * FP16_PER_YMM * FP16_ELEM_SIZE]);
        vcvtph2ps(Zmm(bBaseIdx + i), Ymm(bBaseIdx + i));

        vfmadd231ps(Zmm(accumBaseIdx + K_SUB_ITER * i), Zmm(xBaseIdx),
                    Zmm(bBaseIdx + i));
    }

    add(regBptr, regRsB);

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVM1<KType>::computeKxnfringe()
{
    int fullRegs    = N_LEFT / nElemsPerReg;
    int partialLeft = N_LEFT % nElemsPerReg;

    for (iter_t ksub = 0; ksub < K_SUB_ITER; ksub++) {
        vbroadcastss(Zmm(xBaseIdx + ksub), ptr[regXptr]);
        add(regXptr, F32_ELEM_SIZE);

        for (iter_t i = 0; i < fullRegs; i++) {
            vmovdqu16(Ymm(bBaseIdx + i),
                      ptr[regBptr + i * FP16_PER_YMM * FP16_ELEM_SIZE]);
            vcvtph2ps(Zmm(bBaseIdx + i), Ymm(bBaseIdx + i));
            vfmadd231ps(Zmm(accumBaseIdx + K_SUB_ITER * i + ksub),
                        Zmm(xBaseIdx + ksub), Zmm(bBaseIdx + i));
        }

        if (partialLeft > 0) {
            vmovdqu16(Ymm(bBaseIdx) | mask_regs[0] | T_z,
                      ptr[regBptr + fullRegs * FP16_PER_YMM * FP16_ELEM_SIZE]);
            vcvtph2ps(Zmm(bBaseIdx), Ymm(bBaseIdx));
            vfmadd231ps(Zmm(accumBaseIdx + K_SUB_ITER * fullRegs + ksub),
                        Zmm(xBaseIdx + ksub), Zmm(bBaseIdx));
        }

        add(regBptr, regRsB);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVM1<KType>::compute1xnfringe()
{
    int fullRegs    = N_LEFT / nElemsPerReg;
    int partialLeft = N_LEFT % nElemsPerReg;

    vbroadcastss(Zmm(xBaseIdx), ptr[regXptr]);
    add(regXptr, F32_ELEM_SIZE);

    for (iter_t i = 0; i < fullRegs; i++) {
        vmovdqu16(Ymm(bBaseIdx + i),
                  ptr[regBptr + i * FP16_PER_YMM * FP16_ELEM_SIZE]);
        vcvtph2ps(Zmm(bBaseIdx + i), Ymm(bBaseIdx + i));
        vfmadd231ps(Zmm(accumBaseIdx + K_SUB_ITER * i), Zmm(xBaseIdx),
                    Zmm(bBaseIdx + i));
    }

    if (partialLeft > 0) {
        vmovdqu16(Ymm(bBaseIdx) | mask_regs[0] | T_z,
                  ptr[regBptr + fullRegs * FP16_PER_YMM * FP16_ELEM_SIZE]);
        vcvtph2ps(Zmm(bBaseIdx), Ymm(bBaseIdx));
        vfmadd231ps(Zmm(accumBaseIdx + K_SUB_ITER * fullRegs), Zmm(xBaseIdx),
                    Zmm(bBaseIdx));
    }

    add(regBptr, regRsB);

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVM1<KType>::loopKSubIter(bool kfringe, bool nfringe)
{
    if (kfringe) {
        if (nfringe) {
            mov(regKSubIter,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvM1Params, k_left_sub_iter)]);
            test(regKSubIter, regKSubIter);
            jz(".K_FRINGE_NFRINGE_LEFT", T_NEAR);

            L(".K_FRINGE_NFRINGE_MAIN");
            computeKxnfringe();
            dec(regKSubIter);
            jnz(".K_FRINGE_NFRINGE_MAIN", T_NEAR);

            L(".K_FRINGE_NFRINGE_LEFT");
            mov(regKSubIter,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvM1Params, k_left_sub_left)]);
            test(regKSubIter, regKSubIter);
            jz(".K_FRINGE_NFRINGE_DONE", T_NEAR);

            L(".K_FRINGE_NFRINGE_LEFT_LOOP");
            compute1xnfringe();
            dec(regKSubIter);
            jnz(".K_FRINGE_NFRINGE_LEFT_LOOP", T_NEAR);

            L(".K_FRINGE_NFRINGE_DONE");
        } else {
            mov(regKSubIter,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvM1Params, k_left_sub_iter)]);
            test(regKSubIter, regKSubIter);
            jz(".K_FRINGE_FULL_LEFT", T_NEAR);

            L(".K_FRINGE_FULL_MAIN");
            computeKxNR(false);
            dec(regKSubIter);
            jnz(".K_FRINGE_FULL_MAIN", T_NEAR);

            L(".K_FRINGE_FULL_LEFT");
            mov(regKSubIter,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvM1Params, k_left_sub_left)]);
            test(regKSubIter, regKSubIter);
            jz(".K_FRINGE_FULL_DONE", T_NEAR);

            L(".K_FRINGE_FULL_LOOP");
            compute1xNR(false);
            dec(regKSubIter);
            jnz(".K_FRINGE_FULL_LOOP", T_NEAR);

            L(".K_FRINGE_FULL_DONE");
        }
    } else {
        if (nfringe) {
            mov(regKSubIter,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvM1Params, k_iter_sub_iter)]);
            test(regKSubIter, regKSubIter);
            jz(".K_MAIN_NFRINGE_LEFT", T_NEAR);

            L(".K_MAIN_NFRINGE_MAIN");
            computeKxnfringe();
            dec(regKSubIter);
            jnz(".K_MAIN_NFRINGE_MAIN", T_NEAR);

            L(".K_MAIN_NFRINGE_LEFT");
            mov(regKSubIter,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvM1Params, k_iter_sub_left)]);
            test(regKSubIter, regKSubIter);
            jz(".K_MAIN_NFRINGE_DONE", T_NEAR);

            L(".K_MAIN_NFRINGE_LEFT_LOOP");
            compute1xnfringe();
            dec(regKSubIter);
            jnz(".K_MAIN_NFRINGE_LEFT_LOOP", T_NEAR);

            L(".K_MAIN_NFRINGE_DONE");
        } else {
            mov(regKSubIter,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvM1Params, k_iter_sub_iter)]);
            test(regKSubIter, regKSubIter);
            jz(".K_MAIN_FULL_LEFT", T_NEAR);

            L(".K_MAIN_FULL_MAIN");
            computeKxNR(false);
            dec(regKSubIter);
            jnz(".K_MAIN_FULL_MAIN", T_NEAR);

            L(".K_MAIN_FULL_LEFT");
            mov(regKSubIter,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvM1Params, k_iter_sub_left)]);
            test(regKSubIter, regKSubIter);
            jz(".K_MAIN_FULL_DONE", T_NEAR);

            L(".K_MAIN_FULL_LOOP");
            compute1xNR(false);
            dec(regKSubIter);
            jnz(".K_MAIN_FULL_LOOP", T_NEAR);

            L(".K_MAIN_FULL_DONE");
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVM1<KType>::finalAccumulate()
{
    int regsPerPanel = NR / nElemsPerReg;

    for (iter_t i = 0; i < regsPerPanel; i++) {
        for (iter_t ksub = 1; ksub < K_SUB_ITER; ksub++) {
            vaddps(Zmm(accumBaseIdx + K_SUB_ITER * i),
                   Zmm(accumBaseIdx + K_SUB_ITER * i),
                   Zmm(accumBaseIdx + K_SUB_ITER * i + ksub));
        }
        vmovaps(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + K_SUB_ITER * i));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVM1<KType>::scaleWithAlpha()
{
    if (alphaScalingType != dlp::kernel_frame::scalingType::one) {
        int regsPerPanel = NR / nElemsPerReg;

        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, alpha)]);
        vbroadcastss(Zmm(bBaseIdx), ptr[regTmp1]);

        for (iter_t i = 0; i < regsPerPanel; i++) {
            vmulps(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i), Zmm(bBaseIdx));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVM1<KType>::scaleYWithBeta([[maybe_unused]] bool nMask)
{
    int  regsPerPanel = NR / nElemsPerReg;
    bool isBetaOne = (betaScalingType == dlp::kernel_frame::scalingType::one);

    if (betaScalingType == dlp::kernel_frame::scalingType::zero) {
        return dlp::jit::jitGeneratorError::success;
    }

    Xbyak::Label betaZeroEnd;

    if (!isBetaOne) {
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, beta)]);
        vbroadcastss(Zmm(bBaseIdx + 1), ptr[regTmp1]);

        // NOTE: The Decision Engine will pass betaScalingType as generic
        // for k > KC even when beta = 0. Hence, broadcasting beta and
        // checking if it is actually zero during run-time. This conforms
        // to the standard of avoiding accesses to Y when beta = 0.
        vpxord(Zmm(bBaseIdx), Zmm(bBaseIdx), Zmm(bBaseIdx));
        vucomiss(Xmm(bBaseIdx + 1), Xmm(bBaseIdx));
        je(betaZeroEnd, T_NEAR);
    }

    // Load Y values (F32), scale by beta, add to accumulators
    for (iter_t i = 0; i < regsPerPanel; i++) {
        vmovups(Zmm(yBaseIdx + i), ptr[regYptr + i * RegBytes]);

        if (isBetaOne) {
            vaddps(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i),
                   Zmm(yBaseIdx + i));
        } else {
            vfmadd231ps(Zmm(accumBaseIdx + i), Zmm(bBaseIdx + 1),
                        Zmm(yBaseIdx + i));
        }
    }

    L(betaZeroEnd);
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVM1<KType>::scaleYWithBetaFringe()
{
    int  fullRegs    = N_LEFT / nElemsPerReg;
    int  partialLeft = N_LEFT % nElemsPerReg;
    bool isBetaOne   = (betaScalingType == dlp::kernel_frame::scalingType::one);

    if (betaScalingType == dlp::kernel_frame::scalingType::zero) {
        return dlp::jit::jitGeneratorError::success;
    }

    Xbyak::Label betaZeroEnd;

    if (!isBetaOne) {
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, beta)]);
        vbroadcastss(Zmm(bBaseIdx + 1), ptr[regTmp1]);

        vpxord(Zmm(bBaseIdx), Zmm(bBaseIdx), Zmm(bBaseIdx));
        vucomiss(Xmm(bBaseIdx + 1), Xmm(bBaseIdx));
        je(betaZeroEnd, T_NEAR);
    }

    for (iter_t i = 0; i < fullRegs; i++) {
        vmovups(Zmm(yBaseIdx + i), ptr[regYptr + i * RegBytes]);
        if (isBetaOne) {
            vaddps(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i),
                   Zmm(yBaseIdx + i));
        } else {
            vfmadd231ps(Zmm(accumBaseIdx + i), Zmm(bBaseIdx + 1),
                        Zmm(yBaseIdx + i));
        }
    }

    if (partialLeft > 0) {
        vmovups(Zmm(yBaseIdx) | mask_regs[0] | T_z,
                ptr[regYptr + fullRegs * RegBytes]);
        if (isBetaOne) {
            vaddps(Zmm(accumBaseIdx + fullRegs), Zmm(accumBaseIdx + fullRegs),
                   Zmm(yBaseIdx));
        } else {
            vfmadd231ps(Zmm(accumBaseIdx + fullRegs), Zmm(bBaseIdx + 1),
                        Zmm(yBaseIdx));
        }
    }

    L(betaZeroEnd);
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVM1<KType>::storeYValues([[maybe_unused]] bool nMask)
{
    int regsPerPanel = NR / nElemsPerReg;

    for (iter_t i = 0; i < regsPerPanel; i++) {
        vmovups(ptr[regYptr + i * RegBytes], Zmm(accumBaseIdx + i));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVM1<KType>::storeYValuesFringe()
{
    int fullRegs    = N_LEFT / nElemsPerReg;
    int partialLeft = N_LEFT % nElemsPerReg;

    for (iter_t i = 0; i < fullRegs; i++) {
        vmovups(ptr[regYptr + i * RegBytes], Zmm(accumBaseIdx + i));
    }

    if (partialLeft > 0) {
        vmovups(ptr[regYptr + fullRegs * RegBytes] | mask_regs[0],
                Zmm(accumBaseIdx + fullRegs));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVM1<KType>::generateKernel(utils::gemvM1GeneratorParams& params)
{
    NR               = params.NR;
    N_LEFT           = params.N_LEFT;
    KC               = params.KC;
    K_SUB_ITER       = params.K_SUB_ITER;
    mtag_b           = params.mtag_b;
    alphaScalingType = params.alphaScalingType;
    betaScalingType  = params.betaScalingType;

    RETURN_IF_ERROR(allocateRegisters());

    {
        Xbyak::util::StackFrame stackFrame(this, 1, 13, 0);
        initializeStackFrame(stackFrame);
        initializeParameters(params);

        loadMasks();

        inLocalLabel();

        if (!params.kernelOps.empty()) {
            kernelOpsHandlerPtr =
                std::make_unique<gen::kernelOpsHandler<KType>>(this);
        }

        xor_(regIncN, regIncN);
        xor_(regIncK, regIncK);

        int yRegsNR = NR / nElemsPerReg;

        // N-loop: process NR elements at a time
        if (params.nloop) {
            mov(regNIter,
                ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, n_iter)]);
            test(regNIter, regNIter);
            jz(label_n_loop_end, T_NEAR);

            L(label_n_loop_start);

            regInit(accumBaseIdx, accumReg);
            xor_(regIncK, regIncK);

            if (params.alphaScalingType
                != dlp::kernel_frame::scalingType::zero) {
                mov(regXptr,
                    ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, x)]);

                if (params.kloop) {
                    mov(regKIter,
                        ptr[stackPtr
                            + offsetof(dlp::kernels::gemvM1Params, k_iter)]);
                    test(regKIter, regKIter);
                    jz(label_n_loop_k_loop_end, T_NEAR);

                    L(label_n_loop_k_loop_start);
                    mov(regBptr,
                        ptr[stackPtr
                            + offsetof(dlp::kernels::gemvM1Params, b)]);

                    if ((mtag_b == REORDERED) || (mtag_b == PACK)) {
                        mov(regPsB, KC);
                        lea(regPsB, ptr[regPsB * FP16_ELEM_SIZE]);
                        mov(regTmpYptr,
                            ptr[stackPtr
                                + offsetof(dlp::kernels::gemvM1Params,
                                           jc_cur_loop_rem)]);
                        mov(regTmp2, ptr[stackPtr
                                         + offsetof(dlp::kernels::gemvM1Params,
                                                    n_sub_updated)]);
                        imul(regTmpYptr, regPsB);
                        imul(regTmp2, regIncK);
                        lea(regTmp2, ptr[regTmp2 * FP16_ELEM_SIZE]);

                        lea(regBptr, ptr[regBptr + regTmpYptr]);
                        lea(regBptr, ptr[regBptr + regTmp2]);
                    } else {
                        mov(regPsB, 1);
                        lea(regPsB, ptr[regPsB * FP16_ELEM_SIZE]);
                        mov(regTmp2, regRsB);
                        imul(regTmp2, regIncK);
                        add(regBptr, regTmp2);
                    }

                    mov(regTmp2, regIncN);
                    imul(regTmp2, regPsB);
                    add(regBptr, regTmp2);

                    loopKSubIter(false, false);

                    mov(regTmp2, KC);
                    add(regIncK, regTmp2);
                    dec(regKIter);
                    jnz(label_n_loop_k_loop_start, T_NEAR);
                }

                L(label_n_loop_k_loop_end);

                if (params.kfringe) {
                    mov(regKIter,
                        ptr[stackPtr
                            + offsetof(dlp::kernels::gemvM1Params, k_left)]);
                    test(regKIter, regKIter);
                    jz(label_n_loop_k_fringe_end, T_NEAR);

                    L(label_n_loop_k_fringe_start);
                    mov(regBptr,
                        ptr[stackPtr
                            + offsetof(dlp::kernels::gemvM1Params, b)]);

                    if ((mtag_b == REORDERED) || (mtag_b == PACK)) {
                        mov(regPsB, ptr[stackPtr
                                        + offsetof(dlp::kernels::gemvM1Params,
                                                   k_left)]);
                        lea(regPsB, ptr[regPsB * FP16_ELEM_SIZE]);
                        mov(regTmpYptr,
                            ptr[stackPtr
                                + offsetof(dlp::kernels::gemvM1Params,
                                           jc_cur_loop_rem)]);
                        mov(regTmp2, ptr[stackPtr
                                         + offsetof(dlp::kernels::gemvM1Params,
                                                    n_sub_updated)]);
                        imul(regTmpYptr, regPsB);
                        imul(regTmp2, regIncK);
                        lea(regTmp2, ptr[regTmp2 * FP16_ELEM_SIZE]);

                        lea(regBptr, ptr[regBptr + regTmpYptr]);
                        lea(regBptr, ptr[regBptr + regTmp2]);
                    } else {
                        mov(regPsB, 1);
                        lea(regPsB, ptr[regPsB * FP16_ELEM_SIZE]);
                        mov(regTmp2, regRsB);
                        imul(regTmp2, regIncK);
                        add(regBptr, regTmp2);
                    }

                    mov(regTmp2, regIncN);
                    imul(regTmp2, regPsB);
                    add(regBptr, regTmp2);

                    loopKSubIter(true, false);
                }

                L(label_n_loop_k_fringe_end);

                finalAccumulate();
                scaleWithAlpha();
            }

            scaleYWithBeta(false);

            if (kernelOpsHandlerPtr) {
                using VecPoolType =
                    utils::registerPool<typename Traits::RegType,
                                        Traits::numRegs>;
                using MaskPoolType =
                    utils::registerPool<Xbyak::Opmask, Traits::numMaskRegs>;

                int numCRegs = yRegsNR;

                VecPoolType vecPool;
                vecPool.setAccumulators(accumBaseIdx, numCRegs);
                RETURN_IF_ERROR(vecPool.init(this, Traits::regBytes));

                MaskPoolType maskPool;
                maskPool.addPreserve(utils::MASK_START_IDX, 1);
                RETURN_IF_ERROR(maskPool.init(this,
                                              utils::maskSaveWidth<KType>(),
                                              Traits::reservedMaskBits));

                int maskOffset =
                    offsetof(dlp::kernels::gemvM1Params, nmask_fp16_avx512);

                RETURN_IF_ERROR((kernelOpsHandlerPtr->generateKernelOps(
                    params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemv_m1,
                    1, params.NR, false, 1, accumBaseIdx, numCRegs, vecPool,
                    maskPool, maskOffset)));
            }

            storeYValues(false);

            // Advance Y by NR F32 elements
            mov(regTmp2, NR);
            add(regIncN, regTmp2);
            lea(regYptr, ptr[regYptr + regTmp2 * F32_ELEM_SIZE]);

            if (!params.kernelOps.empty()) {
                mov(regTmp1,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                        + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
                add(regTmp1, NR);
                mov(ptr[stackPtr
                        + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                        + offsetof(dlp_gemm_post_op_attr, post_op_c_j)],
                    regTmp1);
            }

            dec(regNIter);
            jnz(label_n_loop_start, T_NEAR);
        }

        L(label_n_loop_end);

        // N-fringe: process remaining N_LEFT elements
        if (params.nfringe) {
            mov(regNIter,
                ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, n_left)]);
            test(regNIter, regNIter);
            jz(label_n_fringe_end, T_NEAR);

            L(label_n_fringe_start);

            regInit(accumBaseIdx, accumReg);
            xor_(regIncK, regIncK);

            if (params.alphaScalingType
                != dlp::kernel_frame::scalingType::zero) {
                mov(regXptr,
                    ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, x)]);

                if (params.kloop) {
                    mov(regKIter,
                        ptr[stackPtr
                            + offsetof(dlp::kernels::gemvM1Params, k_iter)]);
                    test(regKIter, regKIter);
                    jz(label_n_fringe_k_loop_end, T_NEAR);

                    L(label_n_fringe_k_loop_start);
                    mov(regBptr,
                        ptr[stackPtr
                            + offsetof(dlp::kernels::gemvM1Params, b)]);

                    if ((mtag_b == REORDERED) || (mtag_b == PACK)) {
                        mov(regPsB, KC);
                        lea(regPsB, ptr[regPsB * FP16_ELEM_SIZE]);
                        mov(regTmpYptr,
                            ptr[stackPtr
                                + offsetof(dlp::kernels::gemvM1Params,
                                           jc_cur_loop_rem)]);
                        mov(regTmp2, ptr[stackPtr
                                         + offsetof(dlp::kernels::gemvM1Params,
                                                    n_sub_updated)]);
                        imul(regTmpYptr, regPsB);
                        imul(regTmp2, regIncK);
                        lea(regTmp2, ptr[regTmp2 * FP16_ELEM_SIZE]);

                        lea(regBptr, ptr[regBptr + regTmpYptr]);
                        lea(regBptr, ptr[regBptr + regTmp2]);
                    } else {
                        mov(regPsB, 1);
                        lea(regPsB, ptr[regPsB * FP16_ELEM_SIZE]);
                        mov(regTmp2, regRsB);
                        imul(regTmp2, regIncK);
                        add(regBptr, regTmp2);
                    }

                    mov(regTmp2, regIncN);
                    imul(regTmp2, regPsB);
                    add(regBptr, regTmp2);

                    loopKSubIter(false, true);

                    mov(regTmp2, KC);
                    add(regIncK, regTmp2);
                    dec(regKIter);
                    jnz(label_n_fringe_k_loop_start, T_NEAR);
                }

                L(label_n_fringe_k_loop_end);

                if (params.kfringe) {
                    mov(regKIter,
                        ptr[stackPtr
                            + offsetof(dlp::kernels::gemvM1Params, k_left)]);
                    test(regKIter, regKIter);
                    jz(label_n_fringe_k_fringe_end, T_NEAR);

                    L(label_n_fringe_k_fringe_start);
                    mov(regBptr,
                        ptr[stackPtr
                            + offsetof(dlp::kernels::gemvM1Params, b)]);

                    if ((mtag_b == REORDERED) || (mtag_b == PACK)) {
                        mov(regPsB, ptr[stackPtr
                                        + offsetof(dlp::kernels::gemvM1Params,
                                                   k_left)]);
                        lea(regPsB, ptr[regPsB * FP16_ELEM_SIZE]);
                        mov(regTmpYptr,
                            ptr[stackPtr
                                + offsetof(dlp::kernels::gemvM1Params,
                                           jc_cur_loop_rem)]);
                        mov(regTmp2, ptr[stackPtr
                                         + offsetof(dlp::kernels::gemvM1Params,
                                                    n_sub_updated)]);
                        imul(regTmpYptr, regPsB);
                        imul(regTmp2, regIncK);
                        lea(regTmp2, ptr[regTmp2 * FP16_ELEM_SIZE]);

                        lea(regBptr, ptr[regBptr + regTmpYptr]);
                        lea(regBptr, ptr[regBptr + regTmp2]);
                    } else {
                        mov(regPsB, 1);
                        lea(regPsB, ptr[regPsB * FP16_ELEM_SIZE]);
                        mov(regTmp2, regRsB);
                        imul(regTmp2, regIncK);
                        add(regBptr, regTmp2);
                    }

                    mov(regTmp2, regIncN);
                    imul(regTmp2, regPsB);
                    add(regBptr, regTmp2);

                    loopKSubIter(true, true);
                }

                L(label_n_fringe_k_fringe_end);

                finalAccumulate();
                scaleWithAlpha();
            }

            scaleYWithBetaFringe();

            if (kernelOpsHandlerPtr) {
                using VecPoolType =
                    utils::registerPool<typename Traits::RegType,
                                        Traits::numRegs>;
                using MaskPoolType =
                    utils::registerPool<Xbyak::Opmask, Traits::numMaskRegs>;

                int fringeRegs = (N_LEFT + nElemsPerReg - 1) / nElemsPerReg;

                VecPoolType vecPool;
                vecPool.setAccumulators(accumBaseIdx, fringeRegs);
                RETURN_IF_ERROR(vecPool.init(this, Traits::regBytes));

                MaskPoolType maskPool;
                maskPool.addPreserve(utils::MASK_START_IDX, 1);
                RETURN_IF_ERROR(maskPool.init(this,
                                              utils::maskSaveWidth<KType>(),
                                              Traits::reservedMaskBits));

                int maskOffset =
                    offsetof(dlp::kernels::gemvM1Params, nmask_fp16_avx512);

                RETURN_IF_ERROR((kernelOpsHandlerPtr->generateKernelOps(
                    params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemv_m1,
                    1, params.N_LEFT, true, 1, accumBaseIdx, fringeRegs,
                    vecPool, maskPool, maskOffset)));
            }

            storeYValuesFringe();
        }

        L(label_n_fringe_end);
        outLocalLabel();
    }

    return dlp::jit::jitGeneratorError::success;
}

// =============================================================================
// jitF32FP16GEMVN1: N=1 GEMV (y = A * x) - Matrix-Vector multiplication
//
// A is F32 (M×K), x is FP16 (K×1), y is F32 (M×1).
// Strategy: load x[k:k+16] as FP16 YMM, convert to F32 ZMM.
//           For each row: load A[i][k:k+16] as F32 ZMM, FMA.
//           After K-loop: reduce 16 F32 → 1 scalar per row.
// MR = 16 (process 16 rows at a time).
// =============================================================================

template<utils::kernelInstrType KType>
jitF32FP16GEMVN1<KType>::jitF32FP16GEMVN1(void* buffer, size_t bufferSize)
    : Xbyak::CodeGenerator(bufferSize, buffer)
{
}

template<utils::kernelInstrType KType>
void
jitF32FP16GEMVN1<KType>::initializeStackFrame(Xbyak::util::StackFrame& frame)
{
    stackPtr   = frame.p[0];
    regAptr    = frame.t[0];
    regTmpAptr = frame.t[1];
    regXptr    = frame.t[2];
    regYptr    = frame.t[3];
    regTmpYptr = frame.t[4];
    regRsA     = frame.t[5];
    regCsA     = frame.t[6];
    regRsC     = frame.t[7];
    regMIter   = frame.t[8];
    regKIter   = frame.t[9];
    regTmp1    = frame.t[10];
    regTmp2    = frame.t[11];
    regTmp3    = frame.t[12];
}

template<utils::kernelInstrType KType>
void
jitF32FP16GEMVN1<KType>::initializeParameters()
{
    nElemsPerReg = F32_PER_ZMM;

    mov(regRsA, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, rsA)]);
    mov(regCsA, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, csA)]);
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, rsC)]);

    // A is F32: scale strides by 4 bytes. C/Y is F32: scale by 4 bytes.
    lea(regRsA, ptr[regRsA * F32_ELEM_SIZE]);
    lea(regCsA, ptr[regCsA * F32_ELEM_SIZE]);
    lea(regRsC, ptr[regRsC * F32_ELEM_SIZE]);

    mov(regAptr, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, a)]);
    mov(regYptr, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, y)]);
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVN1<KType>::allocateRegisters()
{
    if (MR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    simdWidth = F32_PER_ZMM;

    xReg     = 1;
    accumReg = MR;
    tmpReg   = 4;
    yReg     = MR / simdWidth;
    if (yReg == 0)
        yReg = 1;

    accumBaseIdx = numRegs - accumReg;
    xBaseIdx     = accumBaseIdx - xReg;
    tmpBaseIdx   = 0;
    yBaseIdx     = numRegs - yReg;

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitF32FP16GEMVN1<KType>::regInit(int baseIdx, int numRegs)
{
    vpxord(Zmm(baseIdx), Zmm(baseIdx), Zmm(baseIdx));
    for (iter_t i = 1; i < numRegs; i++) {
        vmovdqa32(Zmm(baseIdx + i), Zmm(baseIdx));
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVN1<KType>::loadMasks()
{
    for (iter_t i = 0; i < utils::NUM_USABLE_MASKS; i++) {
        mask_regs[i] = Xbyak::Opmask(utils::MASK_START_IDX + i);
    }

    // K-mask: 16-bit for 16 F32 elements per ZMM (k_left F32 elements)
    // BUT x is FP16, so we load 16 FP16 into YMM, convert to 16 F32 in ZMM.
    // The mask applies to the F32 ZMM (16-bit mask).
    // A is F32: also uses same 16-bit mask.
    kmovw(mask_regs[0],
          ptr[stackPtr
              + offsetof(dlp::kernels::gemvN1Params, kmask_fp16_avx512)]);
    kmovw(mask_regs[1],
          ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, mmask_avx512)]);

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVN1<KType>::processMRBlock(int mSize, bool isFringe)
{
    /*
     * For each row i (0..mSize-1):
     *   Load A[i][k:k+16] as F32 vector (ZMM)
     *   FMA: acc[i] += A[i][k:k+16] * x_f32[k:k+16]  (element-wise F32)
     *
     * x has already been loaded into xBaseIdx as F32 (converted from FP16).
     * A is natively F32.
     */

    int mLeft = mSize % 4;
    xor_(regTmp1, regTmp1);
    regInit(tmpBaseIdx, tmpReg);

    // Process rows in groups of 4 for ILP
    for (iter_t i = 0; i < mSize / 4; i++) {
        for (iter_t j = 0; j < 4; j++) {
            if (isFringe) {
                vmovups(Zmm(tmpBaseIdx + j) | mask_regs[0] | T_z,
                        ptr[regTmpAptr + regTmp1]);
            } else {
                vmovups(Zmm(tmpBaseIdx + j), ptr[regTmpAptr + regTmp1]);
            }
            vfmadd231ps(Zmm(accumBaseIdx + i * 4 + j), Zmm(xBaseIdx),
                        Zmm(tmpBaseIdx + j));
            add(regTmp1, regRsA);
        }
    }

    for (iter_t j = 0; j < mLeft; j++) {
        if (isFringe) {
            vmovups(Zmm(tmpBaseIdx + j) | mask_regs[0] | T_z,
                    ptr[regTmpAptr + regTmp1]);
        } else {
            vmovups(Zmm(tmpBaseIdx + j), ptr[regTmpAptr + regTmp1]);
        }
        vfmadd231ps(Zmm(accumBaseIdx + (mSize / 4) * 4 + j), Zmm(xBaseIdx),
                    Zmm(tmpBaseIdx + j));
        add(regTmp1, regRsA);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVN1<KType>::reduceToXmm(int startIdx, int tmpIdx, int blockSize)
{
    if (blockSize > 4) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    for (iter_t i = 0; i < 4; i++) {
        vxorps(Ymm(tmpIdx + i), Ymm(tmpIdx + i), Ymm(tmpIdx + i));
    }

    for (iter_t i = 0; i < blockSize; i++) {
        vextractf32x8(Ymm(tmpIdx + i), Zmm(startIdx + i), 1);
        vaddps(Ymm(tmpIdx + i), Ymm(tmpIdx + i), Ymm(startIdx + i));
    }

    vhaddps(Ymm(tmpIdx), Ymm(tmpIdx), Ymm(tmpIdx + 1));
    vhaddps(Ymm(tmpIdx + 2), Ymm(tmpIdx + 2), Ymm(tmpIdx + 3));
    vhaddps(Ymm(tmpIdx), Ymm(tmpIdx), Ymm(tmpIdx + 2));

    vextractf128(Xmm(tmpIdx + 1), Ymm(tmpIdx), 1);
    vaddps(Xmm(tmpIdx), Xmm(tmpIdx + 1), Xmm(tmpIdx));

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVN1<KType>::reduceAccumulation(int mSize)
{
    for (iter_t i = 0; i < mSize; i += simdWidth) {
        int blockSize = (mSize - i) < simdWidth ? (mSize - i) : simdWidth;

        for (iter_t j = 0; j < blockSize; j += 4) {
            int subBlockSize = (blockSize - j) < 4 ? (blockSize - j) : 4;

            RETURN_IF_ERROR(
                (reduceToXmm(accumBaseIdx + i + j, tmpBaseIdx, subBlockSize)));

            vinsertf32x4(Zmm(accumBaseIdx + i / simdWidth),
                         Zmm(accumBaseIdx + i / simdWidth), Xmm(tmpBaseIdx),
                         j / 4);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVN1<KType>::scaleAccumulationWithAlpha(int mSize)
{
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, alpha)]);
    vbroadcastss(Zmm(tmpBaseIdx), ptr[regKIter]);
    for (iter_t i = 0; i < (mSize + simdWidth - 1) / simdWidth; i += 1) {
        vmulps(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i), Zmm(tmpBaseIdx));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVN1<KType>::scaleYWithBeta(int mSize)
{
    bool isBetaOne   = (betaScalingType == dlp::kernel_frame::scalingType::one);
    bool isRowStored = (yFormat == dlp::kernel_frame::storageFormat::rowMajor);
    int  mLeft       = mSize % simdWidth;

    Xbyak::Label betaZeroEnd;

    if (!isBetaOne) {
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, beta)]);
        vbroadcastss(Zmm(xBaseIdx), ptr[regKIter]);

        // NOTE: The Decision Engine will pass betaScalingType as generic
        // for k > KC even when beta = 0. Hence, broadcasting beta and
        // checking if it is actually zero during run-time. This conforms
        // to the standard of avoiding accesses to Y when beta = 0.
        vpxord(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx), Zmm(tmpBaseIdx));
        vucomiss(Xmm(xBaseIdx), Xmm(tmpBaseIdx));
        je(betaZeroEnd, T_NEAR);
    }

    if (isRowStored) {
        mov(regTmpYptr, regYptr);

        for (iter_t i = 0; i < mSize / simdWidth; i += 1) {
            // Gather MR scalars from Y (row-stored: stride = rsC)
            // Build packed vector in tmpBaseIdx
            for (iter_t j = 0; j < simdWidth; j += 4) {
                int subSize = ((simdWidth - j) < 4) ? (simdWidth - j) : 4;
                for (iter_t s = 0; s < subSize; s++) {
                    int elemIdx = i * simdWidth + j + s;
                    mov(regTmp1, elemIdx);
                    imul(regTmp1, regRsC);
                    vmovss(Xmm(tmpBaseIdx + 1), ptr[regTmpYptr + regTmp1]);
                    if (s == 0) {
                        vmovss(Xmm(tmpBaseIdx + 2), Xmm(tmpBaseIdx + 1));
                    } else {
                        vinsertps(Xmm(tmpBaseIdx + 2), Xmm(tmpBaseIdx + 2),
                                  Xmm(tmpBaseIdx + 1), s << 4);
                    }
                }
                vinsertf32x4(Zmm(yBaseIdx), Zmm(yBaseIdx), Xmm(tmpBaseIdx + 2),
                             j / 4);
            }

            if (isBetaOne) {
                vaddps(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i),
                       Zmm(yBaseIdx));
            } else {
                vfmadd231ps(Zmm(accumBaseIdx + i), Zmm(xBaseIdx),
                            Zmm(yBaseIdx));
            }
        }

        if (mLeft) {
            for (iter_t j = 0; j < mLeft; j += 4) {
                int subSize = ((mLeft - j) < 4) ? (mLeft - j) : 4;
                for (iter_t s = 0; s < subSize; s++) {
                    int elemIdx = (mSize / simdWidth) * simdWidth + j + s;
                    mov(regTmp1, elemIdx);
                    imul(regTmp1, regRsC);
                    vmovss(Xmm(tmpBaseIdx + 1), ptr[regTmpYptr + regTmp1]);
                    if (s == 0) {
                        vmovss(Xmm(tmpBaseIdx + 2), Xmm(tmpBaseIdx + 1));
                    } else {
                        vinsertps(Xmm(tmpBaseIdx + 2), Xmm(tmpBaseIdx + 2),
                                  Xmm(tmpBaseIdx + 1), s << 4);
                    }
                }
                vinsertf32x4(Zmm(yBaseIdx), Zmm(yBaseIdx), Xmm(tmpBaseIdx + 2),
                             j / 4);
            }

            if (isBetaOne) {
                vaddps(Zmm(accumBaseIdx + mSize / simdWidth) | mask_regs[1],
                       Zmm(accumBaseIdx + mSize / simdWidth), Zmm(yBaseIdx));
            } else {
                vfmadd231ps(Zmm(accumBaseIdx + mSize / simdWidth)
                                | mask_regs[1],
                            Zmm(xBaseIdx), Zmm(yBaseIdx));
            }
        }
    } else {
        mov(regTmpYptr, regYptr);

        for (iter_t i = 0; i < mSize / simdWidth; i += 1) {
            if (isBetaOne) {
                vaddps(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i),
                       ptr[regTmpYptr]);
            } else {
                vfmadd231ps(Zmm(accumBaseIdx + i), Zmm(xBaseIdx),
                            ptr[regTmpYptr]);
            }
            lea(regTmpYptr, ptr[regTmpYptr + simdWidth * F32_ELEM_SIZE]);
        }
        if (mLeft) {
            if (isBetaOne) {
                vaddps(Zmm(accumBaseIdx + mSize / simdWidth) | mask_regs[1],
                       Zmm(accumBaseIdx + mSize / simdWidth), ptr[regTmpYptr]);
            } else {
                vfmadd231ps(Zmm(accumBaseIdx + mSize / simdWidth)
                                | mask_regs[1],
                            Zmm(xBaseIdx), ptr[regTmpYptr]);
            }
        }
    }

    L(betaZeroEnd);
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVN1<KType>::storeYValues(int mSize)
{
    bool isRowStored = (yFormat == dlp::kernel_frame::storageFormat::rowMajor);
    int  mLeft       = mSize % simdWidth;

    mov(regTmpYptr, regYptr);

    if (isRowStored) {
        for (iter_t i = 0; i < mSize / simdWidth; i += 1) {
            for (iter_t j = 0; j < simdWidth; j++) {
                int elemIdx = i * simdWidth + j;
                mov(regTmp1, elemIdx);
                imul(regTmp1, regRsC);
                vextractf32x4(Xmm(tmpBaseIdx), Zmm(accumBaseIdx + i), j / 4);
                int lane = j % 4;
                if (lane != 0) {
                    vpermilps(Xmm(tmpBaseIdx + 1), Xmm(tmpBaseIdx), lane);
                    vmovss(ptr[regTmpYptr + regTmp1], Xmm(tmpBaseIdx + 1));
                } else {
                    vmovss(ptr[regTmpYptr + regTmp1], Xmm(tmpBaseIdx));
                }
            }
        }
        if (mLeft) {
            for (iter_t j = 0; j < mLeft; j++) {
                int elemIdx = (mSize / simdWidth) * simdWidth + j;
                mov(regTmp1, elemIdx);
                imul(regTmp1, regRsC);
                vextractf32x4(Xmm(tmpBaseIdx),
                              Zmm(accumBaseIdx + mSize / simdWidth), j / 4);
                int lane = j % 4;
                if (lane != 0) {
                    vpermilps(Xmm(tmpBaseIdx + 1), Xmm(tmpBaseIdx), lane);
                    vmovss(ptr[regTmpYptr + regTmp1], Xmm(tmpBaseIdx + 1));
                } else {
                    vmovss(ptr[regTmpYptr + regTmp1], Xmm(tmpBaseIdx));
                }
            }
        }
    } else {
        for (iter_t i = 0; i < mSize / simdWidth; i += 1) {
            vmovups(ptr[regTmpYptr], Zmm(accumBaseIdx + i));
            lea(regTmpYptr, ptr[regTmpYptr + simdWidth * F32_ELEM_SIZE]);
        }
        if (mLeft) {
            vmovups(ptr[regTmpYptr] | mask_regs[1],
                    Zmm(accumBaseIdx + mSize / simdWidth));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVN1<KType>::generateIrLoop(
    int mSize, [[maybe_unused]] utils::gemvN1GeneratorParams& params)
{
    inLocalLabel();

    mov(regTmpAptr, regAptr);
    mov(regTmpYptr, regAptr);

    mov(regXptr, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, x)]);

    regInit(accumBaseIdx, MR);

    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, k_iter)]);
    test(regKIter, regKIter);
    jz(".KLOOP_FRINGE", T_NEAR);

    L(".KLOOP_START");

    vmovdqu16(Ymm(xBaseIdx), ptr[regXptr]);
    vcvtph2ps(Zmm(xBaseIdx), Ymm(xBaseIdx));

    RETURN_IF_ERROR(processMRBlock(mSize, false));

    add(regTmpYptr, RegBytes);
    mov(regTmpAptr, regTmpYptr);
    add(regXptr, FP16_PER_YMM * FP16_ELEM_SIZE);

    dec(regKIter);
    jnz(".KLOOP_START", T_NEAR);

    L(".KLOOP_FRINGE");

    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, k_left)]);
    test(regKIter, regKIter);
    jz(".KLOOP_FRINGE_END", T_NEAR);

    vmovdqu16(Ymm(xBaseIdx) | mask_regs[0] | T_z, ptr[regXptr]);
    vcvtph2ps(Zmm(xBaseIdx), Ymm(xBaseIdx));
    processMRBlock(mSize, true);

    L(".KLOOP_FRINGE_END");

    RETURN_IF_ERROR(reduceAccumulation(mSize));

    if (alphaScalingType != dlp::kernel_frame::scalingType::one) {
        RETURN_IF_ERROR(scaleAccumulationWithAlpha(mSize));
    }

    if (betaScalingType != dlp::kernel_frame::scalingType::zero) {
        RETURN_IF_ERROR(scaleYWithBeta(mSize));
    }

    outLocalLabel();
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVN1<KType>::generateMLoop(utils::gemvN1GeneratorParams& params)
{
    inLocalLabel();

    mov(regMIter, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, m_iter)]);
    test(regMIter, regMIter);
    jz(".M_FRINGE", T_NEAR);

    L(".MLOOP_START");

    if (params.alphaScalingType != dlp::kernel_frame::scalingType::zero) {
        RETURN_IF_ERROR(generateIrLoop(MR, params));
    }

    if (kernelOpsHandlerPtr) {
        using VecPoolType =
            utils::registerPool<typename Traits::RegType, Traits::numRegs>;
        using MaskPoolType =
            utils::registerPool<Xbyak::Opmask, Traits::numMaskRegs>;

        VecPoolType vecPool;
        vecPool.setAccumulators(accumBaseIdx, yReg);
        RETURN_IF_ERROR(vecPool.init(this, Traits::regBytes));

        MaskPoolType maskPool;
        maskPool.addPreserve(utils::MASK_START_IDX, 2);
        RETURN_IF_ERROR(maskPool.init(this, utils::maskSaveWidth<KType>(),
                                      Traits::reservedMaskBits));

        int maskOffset = offsetof(dlp::kernels::gemvN1Params, mmask_avx512);

        RETURN_IF_ERROR((kernelOpsHandlerPtr->generateKernelOps(
            params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemv_n1,
            params.MR, 1, false, 1, accumBaseIdx, yReg, vecPool, maskPool,
            maskOffset)));
    }

    RETURN_IF_ERROR(storeYValues(MR));

    // Advance A by MR rows
    mov(regTmp1, MR);
    imul(regTmp1, regRsA);
    add(regAptr, regTmp1);

    // Advance Y by MR elements
    mov(regTmp1, MR);
    imul(regTmp1, regRsC);
    add(regYptr, regTmp1);

    if (!params.kernelOps.empty()) {
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
        add(regTmp1, MR);
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_i)],
            regTmp1);
    }

    dec(regMIter);
    jnz(".MLOOP_START", T_NEAR);

    L(".MLOOP_END");
    L(".M_FRINGE");

    if (params.M_LEFT > 0) {
        if (params.alphaScalingType != dlp::kernel_frame::scalingType::zero) {
            RETURN_IF_ERROR(generateIrLoop(params.M_LEFT, params));
        }

        if (kernelOpsHandlerPtr) {
            using VecPoolType =
                utils::registerPool<typename Traits::RegType, Traits::numRegs>;
            using MaskPoolType =
                utils::registerPool<Xbyak::Opmask, Traits::numMaskRegs>;

            int fringeRegs = (params.M_LEFT + simdWidth - 1) / simdWidth;

            VecPoolType vecPool;
            vecPool.setAccumulators(accumBaseIdx, fringeRegs);
            RETURN_IF_ERROR(vecPool.init(this, Traits::regBytes));

            MaskPoolType maskPool;
            maskPool.addPreserve(utils::MASK_START_IDX, 2);
            RETURN_IF_ERROR(maskPool.init(this, utils::maskSaveWidth<KType>(),
                                          Traits::reservedMaskBits));

            int maskOffset = offsetof(dlp::kernels::gemvN1Params, mmask_avx512);

            RETURN_IF_ERROR((kernelOpsHandlerPtr->generateKernelOps(
                params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemv_n1,
                params.M_LEFT, 1, true, 1, accumBaseIdx, fringeRegs, vecPool,
                maskPool, maskOffset)));
        }

        RETURN_IF_ERROR(storeYValues(params.M_LEFT));
    }

    outLocalLabel();
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32FP16GEMVN1<KType>::generateKernel(utils::gemvN1GeneratorParams& params)
{
    MR               = params.MR;
    M_LEFT           = params.M_LEFT;
    yFormat          = params.yFormat;
    alphaScalingType = params.alphaScalingType;
    betaScalingType  = params.betaScalingType;

    RETURN_IF_ERROR(allocateRegisters());

    {
        Xbyak::util::StackFrame stackFrame(this, 1, 13, 0);
        initializeStackFrame(stackFrame);
        initializeParameters();

        loadMasks();

        if (!params.kernelOps.empty()) {
            kernelOpsHandlerPtr =
                std::make_unique<gen::kernelOpsHandler<KType>>(this);
        }

        if (params.mloop) {
            RETURN_IF_ERROR(generateMLoop(params));
        } else {
            if (params.alphaScalingType
                != dlp::kernel_frame::scalingType::zero) {
                RETURN_IF_ERROR(generateIrLoop(params.M_LEFT, params));
            }

            if (kernelOpsHandlerPtr) {
                using VecPoolType =
                    utils::registerPool<typename Traits::RegType,
                                        Traits::numRegs>;
                using MaskPoolType =
                    utils::registerPool<Xbyak::Opmask, Traits::numMaskRegs>;

                int fringeRegs = (params.M_LEFT + simdWidth - 1) / simdWidth;

                VecPoolType vecPool;
                vecPool.setAccumulators(accumBaseIdx, fringeRegs);
                RETURN_IF_ERROR(vecPool.init(this, Traits::regBytes));

                MaskPoolType maskPool;
                maskPool.addPreserve(utils::MASK_START_IDX, 2);
                RETURN_IF_ERROR(maskPool.init(this,
                                              utils::maskSaveWidth<KType>(),
                                              Traits::reservedMaskBits));

                int maskOffset =
                    offsetof(dlp::kernels::gemvN1Params, mmask_avx512);

                RETURN_IF_ERROR((kernelOpsHandlerPtr->generateKernelOps(
                    params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemv_n1,
                    params.M_LEFT, 1, true, 1, accumBaseIdx, fringeRegs,
                    vecPool, maskPool, maskOffset)));
            }

            RETURN_IF_ERROR(storeYValues(params.M_LEFT));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

} // namespace amdzen::gen

// Explicit template instantiation
template class amdzen::gen::jitF32FP16GEMVM1<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
template class amdzen::gen::jitF32FP16GEMVN1<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
