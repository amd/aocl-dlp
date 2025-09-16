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

#include "f32_gemv.hh"
#include "jit_register/jit_register.hh"

namespace amdzen::codegen {

template<utils::kernelInstrType KType>
jitF32GEMVM1<KType>::jitF32GEMVM1(void* buffer, size_t size)
    : Xbyak::CodeGenerator(size, buffer) // Call base class constructor
{
}

template<utils::kernelInstrType KType>
void
jitF32GEMVM1<KType>::initializeStackFrame(Xbyak::util::StackFrame& frame)
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
jitF32GEMVM1<KType>::initializeParameters(
    const utils::gemvM1GeneratorParams& params)
{
    NR               = params.NR;
    KC               = params.KC;
    K_SUB_ITER       = params.K_SUB_ITER;
    yFormat          = params.yFormat;
    alphaScalingType = params.alphaScalingType;
    betaScalingType  = params.betaScalingType;
    mtag_b           = params.mtag_b;

    RegBytes = Traits::regBytes;
    numRegs  = Traits::numRegs;

    simdWidth = RegBytes / sizeof(float); // For f32

    if ((mtag_b == REORDERED) || (mtag_b == PACK)) {
        mov(regRsB, NR);
        lea(regRsB, ptr[regRsB * sizeof(float)]); // rsB = NR * sizeof(float)
    } else {

        mov(regRsB, ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, rsB)]);
        lea(regRsB, ptr[regRsB * sizeof(float)]);
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::allocateRegisters()
{
    if (NR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Allocate registers according to the following rules:
    // x Registers : K_SUB_ITER(hardset to 4 for now)
    // y Registers : NR/simdWidth
    // Accumulation registers : (NR/simdWidth) * K_SUB_ITER

    // NOTE : For now, the generator only supports NR being a multiple of
    // simdWidth. This makes sense from a performance perspective.
    yReg     = NR / simdWidth;
    xReg     = K_SUB_ITER;
    bReg     = K_SUB_ITER;
    accumReg = (NR / simdWidth) * K_SUB_ITER;
    maskReg  = 0; // Set this only when AVX512 codepath is disabled.

    // Direct addressing mode on FMA instructions are avoided here, since
    // we could initiate loads eariler with explicit loads.
    // Thus, both x and B loads are done into registers.
    accumBaseIdx = numRegs - accumReg;
    xBaseIdx     = accumBaseIdx - xReg;
    yBaseIdx     = numRegs - yReg;
    bBaseIdx     = xBaseIdx - bReg;
    maskBaseIdx  = bBaseIdx; // Set this only when AVX512 codepath is disabled.

    if (!Traits::hasMaskSupport) { // Native mask register-file is not supported
                                   // by the architecture.
        maskReg     = NR / simdWidth;
        maskBaseIdx = bBaseIdx - maskReg;
    }

    if (maskBaseIdx < 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitF32GEMVM1<KType>::regInit(int baseIdx, int numRegs)
{
    for (int i = 0; i < numRegs; i++) {
        vxorps(RegType(baseIdx + i), RegType(baseIdx + i),
               RegType(baseIdx + i));
    }
}

// Calculates the address by breaking it into powers of 2
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::offsetBPtr(int temp)
{
    // Offset register, to be added with base in the caller method.
    xor_(regTmp1, regTmp1);
    int power = 1;
    while (temp > 0) {
        if (temp & 1) {
            lea(regTmp1, ptr[regTmp1 + power * regRsB]);
        }
        temp >>= 1;
        power <<= 1;
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::loadMasks()
{
    // Ensuring mapping only from k1 to k7(to avoid k0 usage internally)
    for (int i = 0; i < NUM_USABLE_MASKS; i++) {
        mask_regs[i] = Xbyak::Opmask(MASK_START_IDX + i);
    }

    // Load the masks until the required number, capping on the available
    // registers
    // In case of AVX512, required number is 4
    // In case of AVX512_256, required number is 8
    // The cap is 7(since we can't use k0).
    for (int i = 0; i < NR / simdWidth && i < NUM_USABLE_MASKS; i++) {

        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            kmovw(mask_regs[i],
                  ptr[stackPtr
                      + offsetof(dlp::kernels::gemvM1Params, nmask_avx512)
                      + (i * sizeof(uint16_t))]);
        } else if constexpr (KType
                             == utils::kernelInstrType::avx512_ymm_32_reg) {
            kmovb(mask_regs[i],
                  ptr[stackPtr
                      + offsetof(dlp::kernels::gemvM1Params, nmask_avx512_256)
                      + (i * sizeof(uint8_t))]);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitF32GEMVM1<utils::kernelInstrType::avx2_ymm_16_reg>::loadMasks()
{
    for (int i = 0; i < NR / simdWidth; i++) {
        lea(regNIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, nmask_avx2)
                + (i * sizeof(std::array<int32_t, 8>))]);
        vmovdqu(Xbyak::Ymm(maskBaseIdx + i), ptr[regNIter]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::maskLoadB(int regIdx, int maskIdx)
{
    vmovups(RegType(bBaseIdx + regIdx) | mask_regs[maskIdx],
            ptr[regTmp2 + regTmp1]);

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitF32GEMVM1<utils::kernelInstrType::avx2_ymm_16_reg>::maskLoadB(int regIdx,
                                                                 int maskIdx)
{
    vmaskmovps(Xbyak::Ymm(bBaseIdx + regIdx), Xbyak::Ymm(maskBaseIdx + maskIdx),
               ptr[regTmp2 + regTmp1]);

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::updateMask(int regIdx, int maskIdx)
{
    // Reuse the masks in a round-robin fashion, if we exceed the set limit
    if (regIdx >= NUM_USABLE_MASKS) {
        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            kmovw(mask_regs[maskIdx],
                  ptr[stackPtr
                      + offsetof(dlp::kernels::gemvM1Params, nmask_avx512)
                      + (regIdx * sizeof(uint16_t))]);
        } else if constexpr (KType
                             == utils::kernelInstrType::avx512_ymm_32_reg) {
            kmovb(mask_regs[maskIdx],
                  ptr[stackPtr
                      + offsetof(dlp::kernels::gemvM1Params, nmask_avx512_256)
                      + (regIdx * sizeof(uint8_t))]);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitF32GEMVM1<utils::kernelInstrType::avx2_ymm_16_reg>::updateMask(int regIdx,
                                                                  int maskIdx)
{
    // An empty function in case of AVX2
    return dlp::jit::jitGeneratorError::notSupported;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::restoreMask(int regIdx, int maskIdx)
{
    // Retrieve the original mask
    if (regIdx >= NUM_USABLE_MASKS) {
        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            kmovw(mask_regs[maskIdx],
                  ptr[stackPtr
                      + offsetof(dlp::kernels::gemvM1Params, nmask_avx512)
                      + (maskIdx * sizeof(uint16_t))]);
        } else if constexpr (KType
                             == utils::kernelInstrType::avx512_ymm_32_reg) {
            kmovb(mask_regs[maskIdx],
                  ptr[stackPtr
                      + offsetof(dlp::kernels::gemvM1Params, nmask_avx512_256)
                      + (maskIdx * sizeof(uint8_t))]);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitF32GEMVM1<utils::kernelInstrType::avx2_ymm_16_reg>::restoreMask(int regIdx,
                                                                   int maskIdx)
{
    // An empty function in case of AVX2
    return dlp::jit::jitGeneratorError::notSupported;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::computeKxnfringe()
{
    for (int j = 0; j < K_SUB_ITER; j++) {
        vbroadcastss(RegType(xBaseIdx + j), ptr[regXptr + j * sizeof(float)]);
    }

    int m = 0;
    for (int i = 0; i < NR / simdWidth; i++) {
        m = i % NUM_USABLE_MASKS;
        updateMask(i, m);
        for (int j = 0; j < K_SUB_ITER; j++) {
            offsetBPtr(j); // Calculated into regTmp1
            maskLoadB(j, m);
            vfmadd231ps(RegType(accumBaseIdx + K_SUB_ITER * i + j),
                        RegType(xBaseIdx + j), RegType(bBaseIdx + j));
        }
        restoreMask(i, m);

        // Update the pointer for B
        add(regTmp2, simdWidth * sizeof(float));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::computeKxNR(bool nMask)
{
    mov(regTmp2, regBptr);
    if (!nMask) {
        for (int i = 0; i < K_SUB_ITER; i += 1) {
            vbroadcastss(RegType(xBaseIdx + i),
                         ptr[regXptr + i * sizeof(float)]);
        }
        for (int i = 0; i < NR / simdWidth; i += 1) {
            for (int j = 0; j < K_SUB_ITER; j += 1) {
                offsetBPtr(j); // Calculated into regTmp1
                vmovups(RegType(bBaseIdx + j), ptr[regTmp2 + regTmp1]);
                vfmadd231ps(RegType(accumBaseIdx + K_SUB_ITER * i + j),
                            RegType(xBaseIdx + j), RegType(bBaseIdx + j));
            }

            // Update the pointer for B
            add(regTmp2, simdWidth * sizeof(float));
        }
    } else {
        computeKxnfringe();
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::compute1xnfringe()
{
    vbroadcastss(RegType(xBaseIdx), ptr[regXptr]);

    int m = 0;
    int j = 0;
    for (int i = 0; i < NR / simdWidth; i++) {
        m = i % NUM_USABLE_MASKS;
        j = i % K_SUB_ITER;
        updateMask(i, m);
        xor_(regTmp1, regTmp1);
        lea(regTmp1, ptr[regTmp1 + i * simdWidth * sizeof(float)]);
        maskLoadB(j, m);
        vfmadd231ps(RegType(accumBaseIdx + K_SUB_ITER * i), RegType(xBaseIdx),
                    RegType(bBaseIdx + j));
        restoreMask(i, m);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::compute1xNR(bool nMask)
{
    mov(regTmp2, regBptr);

    if (!nMask) {
        vbroadcastss(RegType(xBaseIdx), ptr[regXptr]);

        int j = 0;
        for (int i = 0; i < NR / simdWidth; i += 1) {
            j = i % K_SUB_ITER;
            vmovups(RegType(bBaseIdx + j),
                    ptr[regTmp2 + i * simdWidth * sizeof(float)]);
            vfmadd231ps(RegType(accumBaseIdx + K_SUB_ITER * i),
                        RegType(xBaseIdx), RegType(bBaseIdx + j));
        }
    } else {
        compute1xnfringe();
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::loopKSubIter(bool kfringe, bool nfringe)
{
    // Defining labels locally to avoid redifinition issues
    Xbyak::Label sub_loop_kc_main_loop_start;
    Xbyak::Label sub_loop_kc_main_loop_end;
    Xbyak::Label sub_loop_kc_fringe_loop_start;
    Xbyak::Label sub_loop_kc_fringe_loop_end;
    Xbyak::Label sub_loop_kf_main_loop_start;
    Xbyak::Label sub_loop_kf_main_loop_end;
    Xbyak::Label sub_loop_kf_fringe_loop_start;
    Xbyak::Label sub_loop_kf_fringe_loop_end;

    // We would receive a k-value, representing the blocksize inside
    // the k-loop. If it is possible to iterate over this in blocks of 4,
    // we do it

    // Pointers to be used : x and B.
    if (!kfringe) {
        mov(regKSubIter,
            ptr[stackPtr
                + offsetof(dlp::kernels::gemvM1Params, k_iter_sub_iter)]);
        test(regKSubIter, regKSubIter);
        jz(sub_loop_kc_main_loop_end, T_NEAR);
        L(sub_loop_kc_main_loop_start);

        computeKxNR(nfringe);

        // Update the pointers for next k iteration
        lea(regXptr, ptr[regXptr + K_SUB_ITER * sizeof(float)]);
        lea(regBptr, ptr[regBptr + regRsB * K_SUB_ITER]);

        dec(regKSubIter);
        jnz(sub_loop_kc_main_loop_start, T_NEAR);

        L(sub_loop_kc_main_loop_end);
        mov(regKSubIter,
            ptr[stackPtr
                + offsetof(dlp::kernels::gemvM1Params, k_iter_sub_left)]);
        test(regKSubIter, regKSubIter);
        jz(sub_loop_kc_fringe_loop_end, T_NEAR);
        L(sub_loop_kc_fringe_loop_start);

        compute1xNR(nfringe);

        // Update the pointers for next k iteration
        lea(regXptr, ptr[regXptr + sizeof(float)]);
        lea(regBptr, ptr[regBptr + regRsB]);

        dec(regKSubIter);
        jnz(sub_loop_kc_fringe_loop_start, T_NEAR);

        L(sub_loop_kc_fringe_loop_end);
    } else {
        mov(regKSubIter,
            ptr[stackPtr
                + offsetof(dlp::kernels::gemvM1Params, k_left_sub_iter)]);
        test(regKSubIter, regKSubIter);
        jz(sub_loop_kf_main_loop_end, T_NEAR);
        L(sub_loop_kf_main_loop_start);

        computeKxNR(nfringe);

        // Update the pointers for next k iteration
        lea(regXptr, ptr[regXptr + K_SUB_ITER * sizeof(float)]);
        lea(regBptr, ptr[regBptr + regRsB * K_SUB_ITER]);

        dec(regKSubIter);
        jnz(sub_loop_kf_main_loop_start, T_NEAR);

        L(sub_loop_kf_main_loop_end);
        mov(regKSubIter,
            ptr[stackPtr
                + offsetof(dlp::kernels::gemvM1Params, k_left_sub_left)]);
        test(regKSubIter, regKSubIter);
        jz(sub_loop_kf_fringe_loop_end, T_NEAR);
        L(sub_loop_kf_fringe_loop_start);

        compute1xNR(nfringe);

        // Update the pointers for next k iteration
        lea(regXptr, ptr[regXptr + sizeof(float)]);
        lea(regBptr, ptr[regBptr + regRsB]);

        dec(regKSubIter);
        jnz(sub_loop_kf_fringe_loop_start, T_NEAR);

        L(sub_loop_kf_fringe_loop_end);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::finalAccumulate()
{
    for (int i = 0; i < NR / simdWidth; i += 1) {
        for (int j = 1; j < K_SUB_ITER; j += 1) {
            vaddps(RegType(accumBaseIdx + K_SUB_ITER * i),
                   RegType(accumBaseIdx + K_SUB_ITER * i),
                   RegType(accumBaseIdx + K_SUB_ITER * i + j));
        }
        vmovaps(RegType(accumBaseIdx + i),
                RegType(accumBaseIdx + K_SUB_ITER * i));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::scaleWithAlpha()
{
    if (alphaScalingType != dlp::kernel_frame::scalingType::one) {
        mov(regKSubIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, alpha)]);
        vbroadcastss(RegType(xBaseIdx), ptr[regKSubIter]);
        for (int i = 0; i < NR / simdWidth; i += 1) {
            vmulps(RegType(accumBaseIdx + i), RegType(xBaseIdx),
                   RegType(accumBaseIdx + i));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::scaleYWithBetaFringe(bool isBetaOne)
{
    int m = 0;
    if (!isBetaOne) {
        for (int i = 0; i < NR / simdWidth; i++) {
            m = i % NUM_USABLE_MASKS;
            updateMask(i, m);
            vfmadd231ps(RegType(accumBaseIdx + i) | mask_regs[m],
                        RegType(xBaseIdx),
                        ptr[regTmpYptr + i * simdWidth * sizeof(float)]);
            restoreMask(i, m);
        }

    } else {
        for (int i = 0; i < NR / simdWidth; i++) {
            m = i % NUM_USABLE_MASKS;
            updateMask(i, m);
            vaddps(RegType(accumBaseIdx + i) | mask_regs[m],
                   RegType(accumBaseIdx + i),
                   ptr[regTmpYptr + i * simdWidth * sizeof(float)]);
            restoreMask(i, m);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitF32GEMVM1<utils::kernelInstrType::avx2_ymm_16_reg>::scaleYWithBetaFringe(
    bool isBetaOne)
{
    if (!isBetaOne) {
        for (int i = 0; i < NR / simdWidth; i++) {
            vmaskmovps(Xbyak::Ymm(yBaseIdx + i), Xbyak::Ymm(maskBaseIdx + i),
                       ptr[regTmpYptr + i * simdWidth * sizeof(float)]);
            vfmadd231ps(Xbyak::Ymm(accumBaseIdx + i), Xbyak::Ymm(xBaseIdx),
                        Xbyak::Ymm(yBaseIdx + i));
        }
    } else {
        for (int i = 0; i < NR / simdWidth; i++) {
            vmaskmovps(Xbyak::Ymm(yBaseIdx + i), Xbyak::Ymm(maskBaseIdx + i),
                       ptr[regTmpYptr + i * simdWidth * sizeof(float)]);
            vaddps(Xbyak::Ymm(accumBaseIdx + i), Xbyak::Ymm(accumBaseIdx + i),
                   Xbyak::Ymm(yBaseIdx + i));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::scaleYWithBeta(bool nMask)
{
    mov(regTmpYptr, regYptr);

    bool isBetaZero = (betaScalingType == dlp::kernel_frame::scalingType::zero);
    bool isBetaOne  = (betaScalingType == dlp::kernel_frame::scalingType::one);

    if (!isBetaZero) {
        if (!isBetaOne) {
            mov(regKSubIter,
                ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, beta)]);
            vbroadcastss(RegType(xBaseIdx), ptr[regKSubIter]);
        }
        if (!nMask) {
            if (!isBetaOne) {
                for (int i = 0; i < NR / simdWidth; i += 1) {
                    vfmadd231ps(
                        RegType(accumBaseIdx + i), RegType(xBaseIdx),
                        ptr[regTmpYptr + i * simdWidth * sizeof(float)]);
                }
            } else {
                for (int i = 0; i < NR / simdWidth; i += 1) {
                    vaddps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                           ptr[regTmpYptr + i * simdWidth * sizeof(float)]);
                }
            }

        } else {
            scaleYWithBetaFringe();
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::storeYValuesFringe()
{
    int m = 0;
    for (int i = 0; i < NR / simdWidth; i++) {
        m = i % NUM_USABLE_MASKS;
        updateMask(i, m);
        vmovups(ptr[regTmpYptr + i * simdWidth * sizeof(float)] | mask_regs[m],
                RegType(accumBaseIdx + i));
        restoreMask(i, m);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitF32GEMVM1<utils::kernelInstrType::avx2_ymm_16_reg>::storeYValuesFringe()
{
    for (int i = 0; i < NR / simdWidth; i++) {
        vmaskmovps(ptr[regTmpYptr + i * simdWidth * sizeof(float)],
                   Xbyak::Ymm(maskBaseIdx + i), Xbyak::Ymm(accumBaseIdx + i));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::storeYValues(bool nMask)
{
    mov(regTmpYptr, regYptr);

    if (!nMask) {
        for (int i = 0; i < NR / simdWidth; i += 1) {
            vmovups(ptr[regTmpYptr + i * simdWidth * sizeof(float)],
                    RegType(accumBaseIdx + i));
        }
    } else {
        storeYValuesFringe();
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::generateKernel(const utils::gemvM1GeneratorParams& params)
{
    // Defining type based aliases
    using bType     = float;
    using xType     = float;
    using yType     = float;
    using accumType = float;

    // Using Xbyak's utility for managing the stack frame
    Xbyak::util::StackFrame frame(this, 1, 13, 0);
    initializeStackFrame(frame);

    // Initializing the parameters
    initializeParameters(params);

    // Allocating valid ranges for register usage
    RETURN_IF_ERROR(allocateRegisters());

    inLocalLabel();

    mov(regYptr, ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, y)]);
    xor_(regIncN, regIncN); // regIncN is used to increment the
    // pointer for N dimension(zeroed before the nloop)

    if (params.nloop) {
        mov(regNIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, n_iter)]);
        test(regNIter, regNIter);
        jz(label_n_loop_end, T_NEAR);
        L(label_n_loop_start);

        // Y prefetch, before the k-loop
        if (betaScalingType != dlp::kernel_frame::scalingType::zero) {
            for (int i = 0; i < NR / simdWidth; i++) {
                prefetcht0(ptr[regYptr + i * simdWidth * sizeof(float)]);
            }
        }

        // Zero out accumulator registers for this n iteration
        regInit(accumBaseIdx, accumReg);
        xor_(regIncK,
             regIncK); // regIncK is used to increment
                       // the pointer for K dimension(zeroed before the
                       // kloop)

        // K-loop is not needed if alpha is zero
        if (params.alphaScalingType != dlp::kernel_frame::scalingType::zero) {
            // Vector x
            mov(regXptr,
                ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, x)]);

            lea(regTmp1, ptr[regRsB + regRsB * 2]); // 3*rsB

            if (params.kloop) {

                mov(regKIter,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvM1Params, k_iter)]);
                test(regKIter, regKIter);
                jz(label_n_loop_k_loop_end, T_NEAR);

                L(label_n_loop_k_loop_start);
                mov(regBptr,
                    ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, b)]);

                // The base pointer to B is should be updated based on
                // whether the matrix is packed/reordered or not This logic
                // is ported from the static kernels, which requires us to
                // update it inside the k-loop.
                if ((mtag_b == REORDERED) || (mtag_b == PACK)) {
                    mov(regPsB, KC);
                    lea(regPsB, ptr[regPsB * sizeof(float)]);
                    mov(regTmpYptr, ptr[stackPtr
                                        + offsetof(dlp::kernels::gemvM1Params,
                                                   jc_cur_loop_rem)]);
                    mov(regTmp2, ptr[stackPtr
                                     + offsetof(dlp::kernels::gemvM1Params,
                                                n_sub_updated)]);
                    imul(regTmpYptr, regPsB);
                    imul(regTmp2, regIncK);
                    lea(regTmp2, ptr[regTmp2 * sizeof(float)]);

                    lea(regBptr, ptr[regBptr + regTmpYptr]);
                    lea(regBptr, ptr[regBptr + regTmp2]);
                } else {
                    mov(regPsB, 1);
                    lea(regPsB, ptr[regPsB * sizeof(float)]);
                    mov(regTmp2, regRsB);
                    imul(regTmp2, regIncK);

                    add(regBptr, regTmp2);
                }

                // Set the base pointer for the iteration
                mov(regTmp2, regIncN);
                imul(regTmp2, regPsB);

                add(regBptr, regTmp2);

                prefetcht0(ptr[regBptr + K_SUB_ITER * regRsB]);

                // This is a sub-loop over the k-dimension
                // This is intended to utlize more registers(it is not just
                // a code-unroll). The block size is KC(since it is the main
                // loop). The booleans indicate which runtime parameter we
                // have to use for iteration. The pointer to x is
                // automatically updated inside.
                loopKSubIter(false, false);

                // Decrement the k-loop iterator
                // Also, increment the pointer offset
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
                    ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, b)]);

                if ((mtag_b == REORDERED) || (mtag_b == PACK)) {
                    mov(regPsB,
                        ptr[stackPtr
                            + offsetof(dlp::kernels::gemvM1Params, k_left)]);
                    lea(regPsB, ptr[regPsB * sizeof(float)]);
                    mov(regTmpYptr, ptr[stackPtr
                                        + offsetof(dlp::kernels::gemvM1Params,
                                                   jc_cur_loop_rem)]);
                    mov(regTmp2, ptr[stackPtr
                                     + offsetof(dlp::kernels::gemvM1Params,
                                                n_sub_updated)]);
                    imul(regTmpYptr, regPsB);
                    imul(regTmp2, regIncK);
                    lea(regTmp2, ptr[regTmp2 * sizeof(float)]);

                    lea(regBptr, ptr[regBptr + regTmpYptr]);
                    lea(regBptr, ptr[regBptr + regTmp2]);
                } else {
                    mov(regPsB, 1);
                    lea(regPsB, ptr[regPsB * sizeof(float)]);

                    mov(regTmp2, regRsB);
                    imul(regTmp2, regIncK);

                    add(regBptr, regTmp2);
                }

                mov(regTmp2, regIncN);
                imul(regTmp2, regPsB);

                add(regBptr, regTmp2);

                prefetcht0(ptr[regBptr + K_SUB_ITER * regRsB]);

                // This is a sub-loop over the k-dimension
                // This is intended to utlize more registers(it is not just
                // a code-unroll) The block size is K_LEFT(since it is the
                // fringe loop) The boolean indicates which runtime
                // parameter we have to use for iteration. The pointers are
                // updated as part of the loopKSubIter function.
                loopKSubIter(true, false);
            }

            L(label_n_loop_k_fringe_end);

            // Final accumulattion of the result
            finalAccumulate();

            // Scale with alpha
            scaleWithAlpha();
        }

        // Scale the result by beta, and store it accordingly
        scaleYWithBeta(false);
        storeYValues(false);

        // Update the pointers for next n iteration(NOTE : B pointer is set
        // inside the kloop, owing to the implementation in static kernels)
        mov(regTmp2, NR);
        add(regIncN, regTmp2);
        lea(regYptr, ptr[regYptr + regTmp2 * sizeof(float)]);
        dec(regNIter);
        jnz(label_n_loop_start, T_NEAR);
    }
    L(label_n_loop_end);
    if (params.nfringe) {

        loadMasks();

        mov(regNIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, n_left)]);
        test(regNIter, regNIter);
        jz(label_n_fringe_end, T_NEAR);
        L(label_n_fringe_start);

        // Zero out accumulator registers for this n iteration
        regInit(accumBaseIdx, accumReg);
        xor_(regIncK, regIncK);

        // K-loop is not needed if alpha is zero
        if (params.alphaScalingType != dlp::kernel_frame::scalingType::zero) {
            // Vector x
            mov(regXptr,
                ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, x)]);

            lea(regTmp1, ptr[regRsB + regRsB * 2]); // 3*rsB

            if (params.kloop) {

                mov(regKIter,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvM1Params, k_iter)]);
                test(regKIter, regKIter);
                jz(label_n_fringe_k_loop_end, T_NEAR);

                L(label_n_fringe_k_loop_start);
                mov(regBptr,
                    ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, b)]);

                if ((mtag_b == REORDERED) || (mtag_b == PACK)) {
                    mov(regPsB, KC);
                    lea(regPsB, ptr[regPsB * sizeof(float)]);
                    mov(regTmpYptr, ptr[stackPtr
                                        + offsetof(dlp::kernels::gemvM1Params,
                                                   jc_cur_loop_rem)]);
                    mov(regTmp2, ptr[stackPtr
                                     + offsetof(dlp::kernels::gemvM1Params,
                                                n_sub_updated)]);
                    imul(regTmpYptr, regPsB);
                    imul(regTmp2, regIncK);
                    lea(regTmp2, ptr[regTmp2 * sizeof(float)]);

                    lea(regBptr, ptr[regBptr + regTmpYptr]);
                    lea(regBptr, ptr[regBptr + regTmp2]);
                } else {
                    mov(regPsB, 1);
                    lea(regPsB, ptr[regPsB * sizeof(float)]);

                    mov(regTmp2, regRsB);
                    imul(regTmp2, regIncK);

                    add(regBptr, regTmp2);
                }

                mov(regTmp2, regIncN);
                imul(regTmp2, regPsB);

                add(regBptr, regTmp2);

                prefetcht0(ptr[regBptr + K_SUB_ITER * regRsB]);

                loopKSubIter(false, true);

                // Decrement the k-loop iterator
                // Also, increment the pointer offset
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
                    ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, b)]);

                if ((mtag_b == REORDERED) || (mtag_b == PACK)) {
                    mov(regPsB,
                        ptr[stackPtr
                            + offsetof(dlp::kernels::gemvM1Params, k_left)]);
                    lea(regPsB, ptr[regPsB * sizeof(float)]);
                    mov(regTmpYptr, ptr[stackPtr
                                        + offsetof(dlp::kernels::gemvM1Params,
                                                   jc_cur_loop_rem)]);
                    mov(regTmp2, ptr[stackPtr
                                     + offsetof(dlp::kernels::gemvM1Params,
                                                n_sub_updated)]);
                    imul(regTmpYptr, regPsB);
                    imul(regTmp2, regIncK);
                    lea(regTmp2, ptr[regTmp2 * sizeof(float)]);

                    lea(regBptr, ptr[regBptr + regTmpYptr]);
                    lea(regBptr, ptr[regBptr + regTmp2]);
                } else {
                    mov(regPsB, 1);
                    lea(regPsB, ptr[regPsB * sizeof(float)]);

                    mov(regTmp2, regRsB);
                    imul(regTmp2, regIncK);

                    add(regBptr, regTmp2);
                }

                mov(regTmp2, regIncN);
                imul(regTmp2, regPsB);

                add(regBptr, regTmp2);

                prefetcht0(ptr[regBptr + K_SUB_ITER * regRsB]);

                loopKSubIter(true, true);
            }

            L(label_n_fringe_k_fringe_end);

            // Final accumulattion of the result
            finalAccumulate();

            // Scale with alpha
            scaleWithAlpha();
        }

        // Scale the result by beta, and store it accordingly
        scaleYWithBeta(true);
        storeYValues(true);
    }

    L(label_n_fringe_end);
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

} // namespace amdzen::codegen

template class amdzen::codegen::jitF32GEMVM1<
    amdzen::utils::kernelInstrType::avx2_ymm_16_reg>;
template class amdzen::codegen::jitF32GEMVM1<
    amdzen::utils::kernelInstrType::avx512_ymm_32_reg>;
template class amdzen::codegen::jitF32GEMVM1<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
