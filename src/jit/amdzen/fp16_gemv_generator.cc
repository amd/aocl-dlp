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

#include "fp16_gemv_generator.hh"

namespace amdzen::gen {

using namespace Xbyak;

// =============================================================================
// jitFP16GEMVM1: M=1 GEMV (y = x * B) - Vector-Matrix multiplication
// =============================================================================

template<utils::kernelInstrType KType>
jitFP16GEMVM1<KType>::jitFP16GEMVM1(void* buffer, size_t bufferSize)
    : Xbyak::CodeGenerator(bufferSize, buffer)
{
}

template<utils::kernelInstrType KType>
void
jitFP16GEMVM1<KType>::initializeStackFrame(Xbyak::util::StackFrame& frame)
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
jitFP16GEMVM1<KType>::initializeParameters(utils::gemvM1GeneratorParams& params)
{
    NR               = params.NR;
    N_LEFT           = params.N_LEFT;
    KC               = params.KC;
    K_SUB_ITER       = params.K_SUB_ITER;
    yFormat          = params.yFormat;
    alphaScalingType = params.alphaScalingType;
    betaScalingType  = params.betaScalingType;
    mtag_b           = params.mtag_b;

    RegBytes     = Traits::regBytes;
    numRegs      = Traits::numRegs;
    nElemsPerReg = FP16_PER_ZMM; // 32 FP16 elements per ZMM

    loadMasks();

    // Set rsB based on mtag_b (following F32 pattern)
    // For packed/reordered: rsB = NR * sizeof(uint16_t) = 128 * 2 = 256 bytes
    // For unpacked: load from params
    if ((mtag_b == REORDERED) || (mtag_b == PACK)) {
        mov(regRsB, NR * sizeof(uint16_t)); // rsB = NR * 2 = 256 bytes
    } else {
        mov(regRsB, ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, rsB)]);
        lea(regRsB, ptr[regRsB * sizeof(uint16_t)]);
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVM1<KType>::allocateRegisters()
{
    if (NR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // FP16 GEMV M=1 register allocation (following U8S8 K_SUB_ITER pattern):
    // NR = 128 (4 ZMMs of 32 FP16 elements each)
    // K_SUB_ITER = 4 (software pipelining factor)
    //
    // xReg: K_SUB_ITER registers for broadcasting x[k+0..k+3]
    // bReg: NR/32 = 4 registers for loading B row
    // accumReg: (NR/32) * K_SUB_ITER = 16 registers for accumulation
    // yReg: NR/32 = 4 registers for Y load/store

    xReg     = K_SUB_ITER;                       // 4
    bReg     = NR / nElemsPerReg;                // 128 / 32 = 4
    accumReg = (NR / nElemsPerReg) * K_SUB_ITER; // 4 * 4 = 16
    yReg     = NR / nElemsPerReg;                // 4

    // Register index assignment:
    // zmm0-7:   available for scratch
    // zmm8-11:  bReg (B matrix loads)
    // zmm12-15: xReg (X broadcasts)
    // zmm16-31: accumReg (16 accumulators for software pipelining)
    // After finalAccumulate(): zmm16-19 hold consolidated results
    // zmm28-31: yReg (Y load/store, no conflict post-finalAccumulate)

    accumBaseIdx = numRegs - accumReg;  // 32 - 16 = 16 (zmm16-31)
    xBaseIdx     = accumBaseIdx - xReg; // 16 - 4 = 12 (zmm12-15)
    bBaseIdx     = xBaseIdx - bReg;     // 12 - 4 = 8 (zmm8-11)
    yBaseIdx     = numRegs - yReg;      // 32 - 4 = 28 (zmm28-31)

    if (bBaseIdx < 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitFP16GEMVM1<KType>::regInit(int baseIdx, int numRegs)
{
    // Zero out FP16 registers using vpxord
    vpxord(Zmm(baseIdx), Zmm(baseIdx), Zmm(baseIdx));
    for (iter_t i = 1; i < numRegs; i++) {
        vmovdqa32(Zmm(baseIdx + i), Zmm(baseIdx));
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVM1<KType>::loadMasks()
{
    // Initialize mask register array (k1-k7)
    for (iter_t i = 0; i < NUM_USABLE_MASKS; i++) {
        mask_regs[i] = Xbyak::Opmask(MASK_START_IDX + i);
    }

    // Load N-dimension mask for FP16 (32-bit mask for 32 elements per ZMM)
    // FP16 uses nmask_fp16_avx512 (uint32_t) since there are 32 FP16 elements
    // per ZMM
    kmovd(mask_regs[0],
          ptr[stackPtr
              + offsetof(dlp::kernels::gemvM1Params, nmask_fp16_avx512)]);

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVM1<KType>::compute1xNR(bool nMask)
{
    mov(regTmp2, regBptr);

    if (!nMask) {
        // Broadcast x[k] to all 32 lanes (16-bit broadcast)
        vpbroadcastw(Zmm(xBaseIdx), ptr[regXptr]);
        // Load B row (128 FP16 = 4 ZMMs) and FMA
        // Note: Use K_SUB_ITER * i for accumulator index (matches U8S8 pattern)
        // This ensures single-K elements go to correct accumulators for
        // finalAccumulate
        for (iter_t i = 0; i < NR / nElemsPerReg; i++) {
            vmovdqu16(Zmm(bBaseIdx + i),
                      ptr[regTmp2 + i * nElemsPerReg * sizeof(uint16_t)]);
            vfmadd231ph(Zmm(accumBaseIdx + K_SUB_ITER * i), Zmm(xBaseIdx),
                        Zmm(bBaseIdx + i));
        }
    } else {
        compute1xnfringe();
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVM1<KType>::compute1xnfringe()
{
    // Broadcast x[k] - single row version
    vpbroadcastw(Zmm(xBaseIdx), ptr[regXptr]);

    int n_iter = N_LEFT / nElemsPerReg;
    int n_left = N_LEFT % nElemsPerReg;

    bool isPacked = ((mtag_b == REORDERED) || (mtag_b == PACK));

    // Load and process full ZMMs (primary panel)
    // Note: Use K_SUB_ITER * i for accumulator index (matches U8S8 pattern)
    // This ensures single-K elements go to correct accumulators for
    // finalAccumulate
    for (iter_t i = 0; i < n_iter; i++) {
        vmovdqu16(Zmm(bBaseIdx + i),
                  ptr[regTmp2 + i * nElemsPerReg * sizeof(uint16_t)]);
        vfmadd231ph(Zmm(accumBaseIdx + K_SUB_ITER * i), Zmm(xBaseIdx),
                    Zmm(bBaseIdx + i));
    }

    // Load and process partial ZMM (secondary panel or remaining in row)
    if (n_left) {
        if (isPacked && (N_LEFT > nElemsPerReg)) {
            // For packed matrices with N_LEFT > nElemsPerReg:
            // Secondary panel is stored separately, use regTmpYptr
            vmovdqu16(Zmm(bBaseIdx + n_iter) | mask_regs[0] | T_z,
                      ptr[regTmpYptr]);
        } else {
            // For unpacked matrices OR packed with N_LEFT <= 32:
            // n_left elements are contiguous after n_iter full ZMMs
            vmovdqu16(Zmm(bBaseIdx + n_iter) | mask_regs[0] | T_z,
                      ptr[regTmp2 + n_iter * nElemsPerReg * sizeof(uint16_t)]);
        }
        vfmadd231ph(Zmm(accumBaseIdx + K_SUB_ITER * n_iter), Zmm(xBaseIdx),
                    Zmm(bBaseIdx + n_iter));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVM1<KType>::computeKxNR(bool nMask)
{
    mov(regTmp2, regBptr);

    if (!nMask) {
        // Pre-broadcast ALL K_SUB_ITER x values for better ILP
        for (iter_t k = 0; k < K_SUB_ITER; k++) {
            vpbroadcastw(Zmm(xBaseIdx + k),
                         ptr[regXptr + k * sizeof(uint16_t)]);
        }

        // Process K_SUB_ITER rows with pre-broadcast x values
        for (iter_t j = 0; j < K_SUB_ITER; j++) {
            // Load B row and FMA for each of 4 ZMMs
            for (iter_t i = 0; i < NR / nElemsPerReg; i++) {
                vmovdqu16(Zmm(bBaseIdx + i),
                          ptr[regTmp2 + i * nElemsPerReg * sizeof(uint16_t)]);
                vfmadd231ph(Zmm(accumBaseIdx + K_SUB_ITER * i + j),
                            Zmm(xBaseIdx + j), Zmm(bBaseIdx + i));
            }
            add(regTmp2, regRsB);
        }
    } else {
        computeKxnfringe();
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVM1<KType>::computeKxnfringe()
{
    // Pre-broadcast K_SUB_ITER x values for better ILP
    for (iter_t j = 0; j < K_SUB_ITER; j++) {
        vpbroadcastw(Zmm(xBaseIdx + j), ptr[regXptr + j * sizeof(uint16_t)]);
    }

    int n_iter = N_LEFT / nElemsPerReg; // Number of full ZMMs (primary panel)
    int n_left = N_LEFT % nElemsPerReg; // Remaining elements (secondary panel)

    bool isPacked = ((mtag_b == REORDERED) || (mtag_b == PACK));

    // Process K_SUB_ITER rows
    for (iter_t k = 0; k < K_SUB_ITER; k++) {
        // Process primary panel (full ZMMs) from regTmp2
        for (iter_t i = 0; i < n_iter; i++) {
            vmovdqu16(Zmm(bBaseIdx + i),
                      ptr[regTmp2 + i * nElemsPerReg * sizeof(uint16_t)]);
            vfmadd231ph(Zmm(accumBaseIdx + K_SUB_ITER * i + k),
                        Zmm(xBaseIdx + k), Zmm(bBaseIdx + i));
        }

        // Process remaining elements (partial ZMM with mask)
        if (n_left) {
            if (isPacked && (N_LEFT > nElemsPerReg)) {
                // For packed matrices with N_LEFT > nElemsPerReg:
                // Secondary panel is stored separately, use regTmpYptr
                // Secondary panel row stride is RegBytes (1 ZMM per row)
                vmovdqu16(Zmm(bBaseIdx + n_iter) | mask_regs[0] | T_z,
                          ptr[regTmpYptr + k * RegBytes]);
            } else {
                // For unpacked matrices OR packed with N_LEFT <= 32:
                // n_left elements are contiguous after n_iter full ZMMs
                vmovdqu16(
                    Zmm(bBaseIdx + n_iter) | mask_regs[0] | T_z,
                    ptr[regTmp2 + n_iter * nElemsPerReg * sizeof(uint16_t)]);
            }
            vfmadd231ph(Zmm(accumBaseIdx + K_SUB_ITER * n_iter + k),
                        Zmm(xBaseIdx + k), Zmm(bBaseIdx + n_iter));
        }

        // Advance primary panel pointer by rsB
        add(regTmp2, regRsB);
    }

    // Advance secondary panel pointer for packed matrices with N_LEFT >
    // nElemsPerReg
    if (n_left && isPacked && (N_LEFT > nElemsPerReg)) {
        lea(regTmpYptr, ptr[regTmpYptr + K_SUB_ITER * RegBytes]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVM1<KType>::loopKSubIter(bool kfringe, bool nfringe)
{
    Xbyak::Label sub_loop_main_start;
    Xbyak::Label sub_loop_main_end;
    Xbyak::Label sub_loop_fringe_start;
    Xbyak::Label sub_loop_fringe_end;

    // For N-fringe with packed matrices:
    // Initialize regTmpYptr to point to B base, then calculate offset to
    // secondary (lt nElemsPerReg) panel if N_LEFT > nElemsPerReg
    if (nfringe && ((mtag_b == REORDERED) || (mtag_b == PACK))) {
        mov(regTmpYptr, regBptr); // Initialize to B base

        if (N_LEFT > nElemsPerReg) {
            // Calculate offset to secondary panel:
            // offset = (rsB / 2) * psB = (rsB/2 elements) * (panel_stride
            // bytes)
            mov(regTmp2, regRsB);
            shr(regTmp2, 1);       // rsB / 2 (bytes to FP16 elements)
            imul(regTmp2, regPsB); // * panel_stride
            lea(regTmpYptr,
                ptr[regTmpYptr + regTmp2]); // Point to secondary panel
        }
    }

    if (!kfringe) {
        mov(regKSubIter,
            ptr[stackPtr
                + offsetof(dlp::kernels::gemvM1Params, k_iter_sub_iter)]);
        test(regKSubIter, regKSubIter);
        jz(sub_loop_main_end, T_NEAR);

        L(sub_loop_main_start);
        computeKxNR(nfringe);

        // Update pointers
        lea(regXptr, ptr[regXptr + K_SUB_ITER * sizeof(uint16_t)]);
        lea(regBptr, ptr[regBptr + regRsB * K_SUB_ITER]);

        dec(regKSubIter);
        jnz(sub_loop_main_start, T_NEAR);

        L(sub_loop_main_end);

        // Handle sub-iteration remainder
        mov(regKSubIter,
            ptr[stackPtr
                + offsetof(dlp::kernels::gemvM1Params, k_iter_sub_left)]);
        test(regKSubIter, regKSubIter);
        jz(sub_loop_fringe_end, T_NEAR);

        L(sub_loop_fringe_start);
        compute1xNR(nfringe);

        lea(regXptr, ptr[regXptr + sizeof(uint16_t)]);
        lea(regBptr, ptr[regBptr + regRsB]);
        // Advance secondary panel pointer (following U8S8 pattern)
        // Secondary panel row stride is RegBytes (1 ZMM)
        lea(regTmpYptr, ptr[regTmpYptr + RegBytes]);

        dec(regKSubIter);
        jnz(sub_loop_fringe_start, T_NEAR);

        L(sub_loop_fringe_end);
    } else {
        // K-fringe handling
        mov(regKSubIter,
            ptr[stackPtr
                + offsetof(dlp::kernels::gemvM1Params, k_left_sub_iter)]);
        test(regKSubIter, regKSubIter);
        jz(sub_loop_main_end, T_NEAR);

        L(sub_loop_main_start);
        computeKxNR(nfringe);

        lea(regXptr, ptr[regXptr + K_SUB_ITER * sizeof(uint16_t)]);
        lea(regBptr, ptr[regBptr + regRsB * K_SUB_ITER]);

        dec(regKSubIter);
        jnz(sub_loop_main_start, T_NEAR);

        L(sub_loop_main_end);

        mov(regKSubIter,
            ptr[stackPtr
                + offsetof(dlp::kernels::gemvM1Params, k_left_sub_left)]);
        test(regKSubIter, regKSubIter);
        jz(sub_loop_fringe_end, T_NEAR);

        L(sub_loop_fringe_start);
        compute1xNR(nfringe);

        lea(regXptr, ptr[regXptr + sizeof(uint16_t)]);
        lea(regBptr, ptr[regBptr + regRsB]);
        // Advance secondary panel pointer (following U8S8 pattern)
        // Secondary panel row stride is RegBytes (1 ZMM)
        lea(regTmpYptr, ptr[regTmpYptr + RegBytes]);

        dec(regKSubIter);
        jnz(sub_loop_fringe_start, T_NEAR);

        L(sub_loop_fringe_end);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVM1<KType>::finalAccumulate()
{
    // FP16 uses K_SUB_ITER=4, creating 16 accumulators (4 N-chunks x 4
    // K-sub-iterations) These must be consolidated after the K-loop. For each
    // N-chunk i, sum accumulators [K_SUB_ITER*i + 0..K_SUB_ITER-1] into
    // accumBaseIdx+i
    //
    // Following U8S8 pattern: always consolidate NR/nElemsPerReg chunks.
    // This works for both N-loop and N-fringe because:
    // - regInit zeros ALL accumulators
    // - Unused accumulators stay 0, consolidating them is harmless
    // - Only storeYValues uses N_LEFT-based bounds
    for (iter_t i = 0; i < NR / nElemsPerReg; i++) {
        // Sum accumulators K_SUB_ITER*i + 1..K_SUB_ITER-1 into K_SUB_ITER*i
        for (iter_t j = 1; j < K_SUB_ITER; j++) {
            vaddph(Zmm(accumBaseIdx + K_SUB_ITER * i),
                   Zmm(accumBaseIdx + K_SUB_ITER * i),
                   Zmm(accumBaseIdx + K_SUB_ITER * i + j));
        }
        // Move consolidated result to accumBaseIdx + i
        vmovdqa32(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + K_SUB_ITER * i));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVM1<KType>::scaleWithAlpha()
{
    // When beta != 0, alpha scaling is combined with beta scaling in
    // scaleYWithBeta() to avoid intermediate FP16 rounding.
    // Only apply alpha here when beta == 0 (no Y accumulation).
    if (betaScalingType == dlp::kernel_frame::scalingType::zero) {
        if (alphaScalingType != dlp::kernel_frame::scalingType::one) {
            // Load alpha and broadcast
            mov(regKSubIter,
                ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, alpha)]);
            vpbroadcastw(Zmm(xBaseIdx), ptr[regKSubIter]);

            // Scale accumulators - following U8S8 pattern: always scale all
            // NR/nElemsPerReg accumulators. This works because:
            // - regInit zeros ALL accumulators
            // - Unused accumulators are 0, so 0 * alpha = 0
            // - Only storeYValues uses N_LEFT-based bounds
            for (iter_t i = 0; i < NR / nElemsPerReg; i++) {
                vmulph(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i),
                       Zmm(xBaseIdx));
            }
        }
    }
    // When beta != 0, alpha will be applied in scaleYWithBeta() using FMA

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVM1<KType>::scaleYWithBeta(bool nMask)
{
    bool isBetaZero = (betaScalingType == dlp::kernel_frame::scalingType::zero);
    bool isBetaOne  = (betaScalingType == dlp::kernel_frame::scalingType::one);
    bool isAlphaOne = (alphaScalingType == dlp::kernel_frame::scalingType::one);

    if (isBetaZero) {
        return dlp::jit::jitGeneratorError::success;
    }

    mov(regTmpYptr, regYptr);

    // Combined alpha*acc + beta*Y computation to avoid intermediate FP16
    // rounding. Uses vfmadd132ph: dest = dest * src1 + src2
    // This computes: acc = acc * alpha + betaY in a single FMA operation.

    // Load alpha into xBaseIdx (used for FMA source)
    if (!isAlphaOne) {
        mov(regKSubIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, alpha)]);
        vpbroadcastw(Zmm(xBaseIdx), ptr[regKSubIter]);
    }

    // Load beta into xBaseIdx+1 (unless beta==1)
    if (!isBetaOne) {
        mov(regKSubIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, beta)]);
        vpbroadcastw(Zmm(xBaseIdx + 1), ptr[regKSubIter]);
    }

    if (!nMask) {
        for (iter_t i = 0; i < NR / nElemsPerReg; i++) {
            // Load Y
            vmovdqu16(Zmm(yBaseIdx + i),
                      ptr[regTmpYptr + i * nElemsPerReg * sizeof(uint16_t)]);

            // Scale Y by beta if needed: betaY = Y * beta
            if (!isBetaOne) {
                vmulph(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i), Zmm(xBaseIdx + 1));
            }

            // Combined alpha*acc + betaY operation
            if (isAlphaOne) {
                // acc = acc + betaY (alpha=1)
                vaddph(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i),
                       Zmm(yBaseIdx + i));
            } else {
                // acc = acc * alpha + betaY (single FMA, one rounding)
                // vfmadd132ph(a, b, c) = a*c + b
                vfmadd132ph(Zmm(accumBaseIdx + i), Zmm(yBaseIdx + i),
                            Zmm(xBaseIdx));
            }
        }
    } else {
        scaleYWithBetaFringe();
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVM1<KType>::scaleYWithBetaFringe()
{
    bool isBetaOne  = (betaScalingType == dlp::kernel_frame::scalingType::one);
    bool isAlphaOne = (alphaScalingType == dlp::kernel_frame::scalingType::one);

    int n_iter = N_LEFT / nElemsPerReg;
    int n_left = N_LEFT % nElemsPerReg;

    for (iter_t i = 0; i < n_iter; i++) {
        vmovdqu16(Zmm(yBaseIdx + i),
                  ptr[regTmpYptr + i * nElemsPerReg * sizeof(uint16_t)]);

        // Scale Y by beta if needed: betaY = Y * beta
        if (!isBetaOne) {
            vmulph(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i), Zmm(xBaseIdx + 1));
        }

        // Combined alpha*acc + betaY operation
        if (isAlphaOne) {
            // acc = acc + betaY (alpha=1)
            vaddph(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i),
                   Zmm(yBaseIdx + i));
        } else {
            // acc = acc * alpha + betaY (single FMA, one rounding)
            vfmadd132ph(Zmm(accumBaseIdx + i), Zmm(yBaseIdx + i),
                        Zmm(xBaseIdx));
        }
    }

    if (n_left) {
        vmovdqu16(Zmm(yBaseIdx + n_iter) | mask_regs[0] | T_z,
                  ptr[regTmpYptr + n_iter * nElemsPerReg * sizeof(uint16_t)]);

        // Scale Y by beta if needed
        if (!isBetaOne) {
            vmulph(Zmm(yBaseIdx + n_iter), Zmm(yBaseIdx + n_iter),
                   Zmm(xBaseIdx + 1));
        }

        // Combined alpha*acc + betaY operation
        if (isAlphaOne) {
            vaddph(Zmm(accumBaseIdx + n_iter), Zmm(accumBaseIdx + n_iter),
                   Zmm(yBaseIdx + n_iter));
        } else {
            vfmadd132ph(Zmm(accumBaseIdx + n_iter), Zmm(yBaseIdx + n_iter),
                        Zmm(xBaseIdx));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVM1<KType>::storeYValues(bool nMask)
{
    mov(regTmpYptr, regYptr);

    if (!nMask) {
        for (iter_t i = 0; i < NR / nElemsPerReg; i++) {
            vmovdqu16(ptr[regTmpYptr + i * nElemsPerReg * sizeof(uint16_t)],
                      Zmm(accumBaseIdx + i));
        }
    } else {
        storeYValuesFringe();
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVM1<KType>::storeYValuesFringe()
{
    int n_iter = N_LEFT / nElemsPerReg;
    int n_left = N_LEFT % nElemsPerReg;

    for (iter_t i = 0; i < n_iter; i++) {
        vmovdqu16(ptr[regTmpYptr + i * nElemsPerReg * sizeof(uint16_t)],
                  Zmm(accumBaseIdx + i));
    }

    if (n_left) {
        vmovdqu16(ptr[regTmpYptr + n_iter * nElemsPerReg * sizeof(uint16_t)]
                      | mask_regs[0],
                  Zmm(accumBaseIdx + n_iter));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVM1<KType>::generateKernel(utils::gemvM1GeneratorParams& params)
{
    RETURN_IF_ERROR(utils::jitGeneratorUtils::checkValidGemvM1Params(params));

    {
        Xbyak::util::StackFrame frame(this, 1, 13, 0);
        initializeStackFrame(frame);
        initializeParameters(params);
        RETURN_IF_ERROR(allocateRegisters());

        inLocalLabel();

        mov(regYptr, ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, y)]);
        xor_(regIncN, regIncN);

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

                    // B pointer calculation depends on packing status
                    if ((mtag_b == REORDERED) || (mtag_b == PACK)) {
                        // Packed/reordered: use panel stride based on KC
                        // Following BF16 pattern with sizeof scaling for 2-byte
                        // elements
                        mov(regPsB, KC);
                        lea(regPsB,
                            ptr[regPsB
                                * sizeof(uint16_t)]); // psB = KC * 2 bytes
                        mov(regTmpYptr,
                            ptr[stackPtr
                                + offsetof(dlp::kernels::gemvM1Params,
                                           jc_cur_loop_rem)]);
                        mov(regTmp2, ptr[stackPtr
                                         + offsetof(dlp::kernels::gemvM1Params,
                                                    n_sub_updated)]);
                        imul(regTmpYptr,
                             regPsB); // jc_cur_loop_rem * psB (bytes)
                        imul(regTmp2,
                             regIncK); // n_sub_updated * regIncK (elements)
                        lea(regTmp2,
                            ptr[regTmp2 * sizeof(uint16_t)]); // Scale to bytes

                        lea(regBptr, ptr[regBptr + regTmpYptr]);
                        lea(regBptr, ptr[regBptr + regTmp2]);
                    } else {
                        // Unpacked: use row stride
                        // FIX: regPsB must be in bytes for correct N offset
                        // calculation
                        mov(regPsB, sizeof(uint16_t)); // psB = 2 bytes
                        mov(regTmp2, regRsB);
                        imul(regTmp2, regIncK); // rsB * regIncK (bytes, rsB
                                                // already scaled)

                        add(regBptr, regTmp2);
                    }

                    mov(regTmp2, regIncN);
                    imul(regTmp2,
                         regPsB); // regIncN * psB (now correctly in bytes)
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

                    // B pointer calculation depends on packing status
                    if ((mtag_b == REORDERED) || (mtag_b == PACK)) {
                        // Packed/reordered: use panel stride based on k_left
                        mov(regPsB, ptr[stackPtr
                                        + offsetof(dlp::kernels::gemvM1Params,
                                                   k_left)]);
                        lea(regPsB,
                            ptr[regPsB
                                * sizeof(uint16_t)]); // psB = k_left * 2 bytes
                        mov(regTmpYptr,
                            ptr[stackPtr
                                + offsetof(dlp::kernels::gemvM1Params,
                                           jc_cur_loop_rem)]);
                        mov(regTmp2, ptr[stackPtr
                                         + offsetof(dlp::kernels::gemvM1Params,
                                                    n_sub_updated)]);
                        imul(regTmpYptr,
                             regPsB); // jc_cur_loop_rem * psB (bytes)
                        imul(regTmp2,
                             regIncK); // n_sub_updated * regIncK (elements)
                        lea(regTmp2,
                            ptr[regTmp2 * sizeof(uint16_t)]); // Scale to bytes

                        lea(regBptr, ptr[regBptr + regTmpYptr]);
                        lea(regBptr, ptr[regBptr + regTmp2]);
                    } else {
                        // Unpacked: use row stride
                        mov(regPsB, 1);
                        lea(regPsB,
                            ptr[regPsB * sizeof(uint16_t)]); // psB = 2 bytes
                        mov(regTmp2, regRsB);
                        imul(regTmp2, regIncK); // rsB * regIncK (bytes)

                        add(regBptr, regTmp2);
                    }

                    mov(regTmp2, regIncN);
                    imul(regTmp2, regPsB); // regIncN * psB
                    add(regBptr, regTmp2);

                    loopKSubIter(true, false);
                }

                L(label_n_loop_k_fringe_end);

                finalAccumulate();
                scaleWithAlpha();
            }

            scaleYWithBeta(false);
            storeYValues(false);

            mov(regTmp2, NR);
            add(regIncN, regTmp2);
            lea(regYptr, ptr[regYptr + regTmp2 * sizeof(uint16_t)]);

            dec(regNIter);
            jnz(label_n_loop_start, T_NEAR);
        }

        L(label_n_loop_end);

        if (params.nfringe) {
            mov(regNIter,
                ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, n_left)]);
            test(regNIter, regNIter);
            jz(label_n_fringe_end, T_NEAR);

            // Only adjust rsB for packed/reordered matrices
            // For unpacked, keep the original rsB = ldb * sizeof(uint16_t)
            // rsB matches packB greedy selection for PRIMARY panel:
            // Calculate rsB based on how many full ZMMs fit in N_LEFT
            // Each ZMM holds nElemsPerReg elements (32 FP16)
            // rsB = number_of_full_ZMMs * RegBytes (or at least 1 ZMM for
            // fringe)
            if ((mtag_b == REORDERED) || (mtag_b == PACK)) {
                int numFullZmms = N_LEFT / nElemsPerReg;
                if (numFullZmms == 0) {
                    // N_LEFT < nElemsPerReg: only secondary panel, rsB = 1 ZMM
                    mov(regRsB, RegBytes);
                } else {
                    // rsB = numFullZmms * RegBytes
                    mov(regRsB, numFullZmms * RegBytes);
                }
            }

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

                    // B pointer calculation depends on packing status
                    if ((mtag_b == REORDERED) || (mtag_b == PACK)) {
                        // Packed/reordered: use panel stride based on KC
                        mov(regPsB, KC);
                        lea(regPsB,
                            ptr[regPsB
                                * sizeof(uint16_t)]); // psB = KC * 2 bytes
                        mov(regTmpYptr,
                            ptr[stackPtr
                                + offsetof(dlp::kernels::gemvM1Params,
                                           jc_cur_loop_rem)]);
                        mov(regTmp2, ptr[stackPtr
                                         + offsetof(dlp::kernels::gemvM1Params,
                                                    n_sub_updated)]);
                        imul(regTmpYptr,
                             regPsB); // jc_cur_loop_rem * psB (bytes)
                        imul(regTmp2,
                             regIncK); // n_sub_updated * regIncK (elements)
                        lea(regTmp2,
                            ptr[regTmp2 * sizeof(uint16_t)]); // Scale to bytes

                        lea(regBptr, ptr[regBptr + regTmpYptr]);
                        lea(regBptr, ptr[regBptr + regTmp2]);
                    } else {
                        // Unpacked: use row stride
                        mov(regPsB, 1);
                        lea(regPsB,
                            ptr[regPsB * sizeof(uint16_t)]); // psB = 2 bytes
                        mov(regTmp2, regRsB);
                        imul(regTmp2, regIncK); // rsB * regIncK (bytes)

                        add(regBptr, regTmp2);
                    }

                    mov(regTmp2, regIncN);
                    imul(regTmp2, regPsB); // regIncN * psB
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

                    // B pointer calculation depends on packing status
                    if ((mtag_b == REORDERED) || (mtag_b == PACK)) {
                        // Packed/reordered: use panel stride based on k_left
                        mov(regPsB, ptr[stackPtr
                                        + offsetof(dlp::kernels::gemvM1Params,
                                                   k_left)]);
                        lea(regPsB,
                            ptr[regPsB
                                * sizeof(uint16_t)]); // psB = k_left * 2 bytes
                        mov(regTmpYptr,
                            ptr[stackPtr
                                + offsetof(dlp::kernels::gemvM1Params,
                                           jc_cur_loop_rem)]);
                        mov(regTmp2, ptr[stackPtr
                                         + offsetof(dlp::kernels::gemvM1Params,
                                                    n_sub_updated)]);
                        imul(regTmpYptr,
                             regPsB); // jc_cur_loop_rem * psB (bytes)
                        imul(regTmp2,
                             regIncK); // n_sub_updated * regIncK (elements)
                        lea(regTmp2,
                            ptr[regTmp2 * sizeof(uint16_t)]); // Scale to bytes

                        lea(regBptr, ptr[regBptr + regTmpYptr]);
                        lea(regBptr, ptr[regBptr + regTmp2]);
                    } else {
                        // Unpacked: use row stride
                        mov(regPsB, 1);
                        lea(regPsB,
                            ptr[regPsB * sizeof(uint16_t)]); // psB = 2 bytes
                        mov(regTmp2, regRsB);
                        imul(regTmp2, regIncK); // rsB * regIncK (bytes)

                        add(regBptr, regTmp2);
                    }

                    mov(regTmp2, regIncN);
                    imul(regTmp2, regPsB); // regIncN * psB
                    add(regBptr, regTmp2);

                    loopKSubIter(true, true);
                }

                L(label_n_fringe_k_fringe_end);

                finalAccumulate();
                scaleWithAlpha();
            }

            scaleYWithBeta(true);
            storeYValues(true);
        }

        L(label_n_fringe_end);
        outLocalLabel();

        vzeroupper();
    } // StackFrame destructor inserts 'ret' here

    return dlp::jit::jitGeneratorError::success;
}

// =============================================================================
// jitFP16GEMVN1: N=1 GEMV (y = A * x) - Matrix-Vector multiplication
// =============================================================================

template<utils::kernelInstrType KType>
jitFP16GEMVN1<KType>::jitFP16GEMVN1(void* buffer, size_t bufferSize)
    : Xbyak::CodeGenerator(bufferSize, buffer)
{
}

template<utils::kernelInstrType KType>
void
jitFP16GEMVN1<KType>::initializeStackFrame(Xbyak::util::StackFrame& frame)
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
jitFP16GEMVN1<KType>::initializeParameters()
{
    nElemsPerReg = FP16_PER_ZMM; // 32 FP16 elements per ZMM

    mov(regRsA, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, rsA)]);
    mov(regCsA, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, csA)]);
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, rsC)]);

    // Scale strides by FP16 element size (2 bytes per element)
    // Following F32 pattern where strides are scaled by sizeof(float)
    lea(regRsA, ptr[regRsA * sizeof(uint16_t)]);
    lea(regCsA, ptr[regCsA * sizeof(uint16_t)]);
    lea(regRsC, ptr[regRsC * sizeof(uint16_t)]);

    mov(regAptr, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, a)]);
    mov(regYptr, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, y)]);
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVN1<KType>::allocateRegisters()
{
    if (MR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // FP16 GEMV N=1 register allocation:
    // MR = 16 (process 16 rows at a time)
    // xReg: 1 register for broadcasting x[k]
    // accumReg: MR registers (one per row)
    // tmpReg: 4 for A loading and reduction
    // yReg: 1 (for reduction results)

    xReg     = 1;
    accumReg = MR; // 16
    tmpReg   = 4;
    yReg     = 1;

    // Register index assignment
    accumBaseIdx = numRegs - accumReg;  // zmm16-zmm31
    xBaseIdx     = accumBaseIdx - xReg; // zmm15
    tmpBaseIdx   = 0;                   // zmm0-zmm3
    yBaseIdx     = tmpBaseIdx + tmpReg; // zmm4

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitFP16GEMVN1<KType>::regInit(int baseIdx, int numRegs)
{
    vpxord(Zmm(baseIdx), Zmm(baseIdx), Zmm(baseIdx));
    for (iter_t i = 1; i < numRegs; i++) {
        vmovdqa32(Zmm(baseIdx + i), Zmm(baseIdx));
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVN1<KType>::loadMasks()
{
    for (iter_t i = 0; i < NUM_USABLE_MASKS; i++) {
        mask_regs[i] = Xbyak::Opmask(MASK_START_IDX + i);
    }

    // Load M-dimension mask (32-bit for FP16)
    kmovd(mask_regs[0],
          ptr[stackPtr
              + offsetof(dlp::kernels::gemvN1Params, kmask_fp16_avx512)]);
    kmovw(mask_regs[1],
          ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, mmask_avx512)]);

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVN1<KType>::processMRBlock(int mSize, bool isFringe)
{
    // GEMV N=1 algorithm: y[i] = Σ A[i,k] * x[k] for all k
    // For each row i, load A[i][k:k+32] as a VECTOR and FMA with x[k:k+32]
    // This performs element-wise multiply-accumulate, NOT broadcast!
    //
    // Following F32/U8S8 pattern:
    // 1. Load A row segment as vector: A[i][k:k+simdWidth]
    // 2. FMA: acc[i] += A[i,k:k+simd] * x[k:k+simd] (element-wise)

    int mLeft = mSize % 4;
    xor_(regTmp1, regTmp1);
    regInit(tmpBaseIdx, tmpReg);

    // Process rows in groups of 4 for better ILP (following F32/U8S8 pattern)
    for (iter_t i = 0; i < mSize / 4; i++) {
        for (iter_t j = 0; j < 4; j++) {
            // Load A[row][k:k+32] as a VECTOR (32 FP16 elements)
            if (isFringe) {
                vmovdqu16(Zmm(tmpBaseIdx + j) | mask_regs[0] | T_z,
                          ptr[regTmpAptr + regTmp1]);
            } else {
                vmovdqu16(Zmm(tmpBaseIdx + j), ptr[regTmpAptr + regTmp1]);
            }
            // Element-wise FMA: acc[row] += A[row,k:k+32] * x[k:k+32]
            vfmadd231ph(Zmm(accumBaseIdx + i * 4 + j), Zmm(xBaseIdx),
                        Zmm(tmpBaseIdx + j));
            add(regTmp1, regRsA); // Move to next row
        }
    }

    // Handle remaining rows (mSize % 4)
    for (iter_t j = 0; j < mLeft; j++) {
        if (isFringe) {
            vmovdqu16(Zmm(tmpBaseIdx + j) | mask_regs[0] | T_z,
                      ptr[regTmpAptr + regTmp1]);
        } else {
            vmovdqu16(Zmm(tmpBaseIdx + j), ptr[regTmpAptr + regTmp1]);
        }
        vfmadd231ph(Zmm(accumBaseIdx + (mSize / 4) * 4 + j), Zmm(xBaseIdx),
                    Zmm(tmpBaseIdx + j));
        add(regTmp1, regRsA);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVN1<KType>::reduceAccToScalar(int accIdx, int tmpIdx)
{
    // Horizontal reduction: 32 FP16 -> 1 scalar
    // Using EVEX-encoded instructions throughout (required for FP16 ops)
    //
    // Step 1: ZMM (32 FP16) -> YMM (16 FP16)
    vextractf32x8(Ymm(tmpIdx), Zmm(accIdx), 1);
    vaddph(Ymm(accIdx), Ymm(accIdx), Ymm(tmpIdx));

    // Step 2: YMM (16 FP16) -> XMM (8 FP16)
    // Use vextractf32x4 (EVEX) instead of vextractf128 (VEX)
    vextractf32x4(Xmm(tmpIdx), Ymm(accIdx), 1);
    vaddph(Xmm(accIdx), Xmm(accIdx), Xmm(tmpIdx));

    // Step 3: XMM (8 FP16) -> 4 FP16
    // Shuffle upper 4 FP16 to lower position: [7,6,5,4,3,2,1,0] -> [7,6,5,4]
    // Using vpermilps with EVEX encoding to avoid VEX/EVEX mixing
    vpermilps(Xmm(tmpIdx), Xmm(accIdx), 0xEE);
    vaddph(Xmm(accIdx), Xmm(accIdx), Xmm(tmpIdx));

    // Step 4: 4 FP16 -> 2 FP16
    // Shuffle: [3,2,1,0] -> [3,2]
    vpermilps(Xmm(tmpIdx), Xmm(accIdx), 0x55);
    vaddph(Xmm(accIdx), Xmm(accIdx), Xmm(tmpIdx));

    // Step 5: 2 FP16 -> 1 FP16
    // Shift right by 2 bytes to move element 1 to element 0 position
    vpsrldq(Xmm(tmpIdx), Xmm(accIdx), 2);
    vaddph(Xmm(accIdx), Xmm(accIdx), Xmm(tmpIdx));

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVN1<KType>::reduceAccumulation(int mSize)
{
    // Reduce each accumulator to a single FP16 scalar
    for (iter_t i = 0; i < mSize; i++) {
        RETURN_IF_ERROR(reduceAccToScalar(accumBaseIdx + i, tmpBaseIdx));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVN1<KType>::scaleAlpha(int mSize)
{
    if (alphaScalingType != dlp::kernel_frame::scalingType::one) {
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, alpha)]);
        vpbroadcastw(Zmm(tmpBaseIdx + 1), ptr[regKIter]);

        for (iter_t i = 0; i < mSize; i++) {
            vmulph(Xmm(accumBaseIdx + i), Xmm(accumBaseIdx + i),
                   Xmm(tmpBaseIdx + 1));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVN1<KType>::scaleYWithBeta_FP16(int mSize, bool isRowStored)
{
    bool isBetaOne = (betaScalingType == dlp::kernel_frame::scalingType::one);

    if (!isBetaOne) {
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, beta)]);
        vpbroadcastw(Zmm(tmpBaseIdx + 2), ptr[regKIter]);
    }

    if (isRowStored) {
        // Load Y elements scattered in row-major format
        for (iter_t i = 0; i < mSize; i++) {
            mov(regTmp1, i);
            imul(regTmp1, regRsC);

            vpbroadcastw(Xmm(yBaseIdx), ptr[regTmpYptr + regTmp1]);

            if (isBetaOne) {
                vaddph(Xmm(accumBaseIdx + i), Xmm(accumBaseIdx + i),
                       Xmm(yBaseIdx));
            } else {
                vfmadd231ph(Xmm(accumBaseIdx + i), Xmm(tmpBaseIdx + 2),
                            Xmm(yBaseIdx));
            }
        }
    } else {
        // Column-major: Y elements are contiguous
        for (iter_t i = 0; i < mSize; i++) {
            vpbroadcastw(Xmm(yBaseIdx), ptr[regTmpYptr + i * sizeof(uint16_t)]);

            if (isBetaOne) {
                vaddph(Xmm(accumBaseIdx + i), Xmm(accumBaseIdx + i),
                       Xmm(yBaseIdx));
            } else {
                vfmadd231ph(Xmm(accumBaseIdx + i), Xmm(tmpBaseIdx + 2),
                            Xmm(yBaseIdx));
            }
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVN1<KType>::scaleBeta(int mSize)
{
    mov(regTmpYptr, regYptr);

    bool isRowStored = (yFormat == dlp::kernel_frame::storageFormat::rowMajor);
    RETURN_IF_ERROR(scaleYWithBeta_FP16(mSize, isRowStored));

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVN1<KType>::storeY_rowStored_FP16(int mSize)
{
    // Store each scalar result to row-major Y
    for (iter_t i = 0; i < mSize; i++) {
        mov(regTmp1, i);
        imul(regTmp1, regRsC);
        // Extract lowest 16 bits and store
        vpextrw(ptr[regTmpYptr + regTmp1], Xmm(accumBaseIdx + i), 0);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVN1<KType>::storeY_colStored_FP16(int mSize)
{
    // Store each scalar result to column-major Y (contiguous)
    for (iter_t i = 0; i < mSize; i++) {
        vpextrw(ptr[regTmpYptr + i * sizeof(uint16_t)], Xmm(accumBaseIdx + i),
                0);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVN1<KType>::storeResult(int mSize)
{
    mov(regTmpYptr, regYptr);

    if (yFormat == dlp::kernel_frame::storageFormat::rowMajor) {
        RETURN_IF_ERROR(storeY_rowStored_FP16(mSize));
    } else {
        RETURN_IF_ERROR(storeY_colStored_FP16(mSize));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVN1<KType>::generateIrLoop(int mSize)
{
    // GEMV N=1 K-loop: For each K-chunk of simdWidth (32 FP16 elements):
    //   1. Load x[k:k+32] as a vector
    //   2. For each row i: load A[i][k:k+32] as vector, FMA with x
    //   3. Advance A column pointer by simdWidth*csA, X pointer by RegBytes
    // After K-loop: reduce each accumulator from 32 FP16 to 1 scalar

    inLocalLabel();

    mov(regTmpAptr, regAptr);
    mov(regTmpYptr, regAptr); // Track column position for A matrix (like U8S8)

    mov(regXptr, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, x)]);

    regInit(accumBaseIdx, MR);

    if (alphaScalingType != dlp::kernel_frame::scalingType::zero) {
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, k_iter)]);
        test(regKIter, regKIter);
        jz(".KLOOP_FRINGE", T_NEAR);

        L(".KLOOP_START");

        // Load x[k:k+32] as a VECTOR (32 FP16 elements)
        vmovdqu16(Zmm(xBaseIdx), ptr[regXptr]);

        RETURN_IF_ERROR(processMRBlock(mSize, false));

        // Advance to next K chunk (32 FP16 elements = 64 bytes)
        // Column advancement: regTmpYptr tracks column position
        add(regTmpYptr, RegBytes);   // Move column pointer by 64 bytes
        mov(regTmpAptr, regTmpYptr); // Reset row pointer to new column
        add(regXptr, RegBytes);      // Advance X by 64 bytes

        dec(regKIter);
        jnz(".KLOOP_START", T_NEAR);

        L(".KLOOP_FRINGE");

        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, k_left)]);
        test(regKIter, regKIter);
        jz(".KLOOP_FRINGE_END", T_NEAR);

        // Load remaining X elements with mask (k_left FP16 elements)
        vmovdqu16(Zmm(xBaseIdx) | mask_regs[0] | T_z, ptr[regXptr]);
        RETURN_IF_ERROR(processMRBlock(mSize, true));

        L(".KLOOP_FRINGE_END");

        // After K-loop: reduce each accumulator from 32 FP16 to 1 scalar
        RETURN_IF_ERROR(reduceAccumulation(mSize));
    }

    // Skip alpha scaling when alpha=0 (accumulators already zeroed by regInit)
    if (alphaScalingType != dlp::kernel_frame::scalingType::one
        && alphaScalingType != dlp::kernel_frame::scalingType::zero) {
        RETURN_IF_ERROR(scaleAlpha(mSize));
    }

    if (betaScalingType != dlp::kernel_frame::scalingType::zero) {
        RETURN_IF_ERROR(scaleBeta(mSize));
    }

    RETURN_IF_ERROR(storeResult(mSize));

    outLocalLabel();
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVN1<KType>::generateMLoop(utils::gemvN1GeneratorParams& params)
{
    inLocalLabel();

    mov(regMIter, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, m_iter)]);
    test(regMIter, regMIter);
    jz(".M_FRINGE", T_NEAR);

    L(".MLOOP_START");

    RETURN_IF_ERROR(generateIrLoop(MR));

    mov(regTmp1, MR);
    imul(regTmp1, regRsA);
    add(regAptr, regTmp1);

    mov(regTmp1, MR);
    imul(regTmp1, regRsC);
    add(regYptr, regTmp1);

    dec(regMIter);
    jnz(".MLOOP_START", T_NEAR);

    L(".MLOOP_END");
    L(".M_FRINGE");

    RETURN_IF_ERROR(generateIrLoop(params.M_LEFT));

    outLocalLabel();
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitFP16GEMVN1<KType>::generateKernel(utils::gemvN1GeneratorParams& params)
{
    RETURN_IF_ERROR(utils::jitGeneratorUtils::checkValidGemvN1Params(params));

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

        if (params.mloop) {
            RETURN_IF_ERROR(generateMLoop(params));
        } else {
            RETURN_IF_ERROR(generateIrLoop(params.M_LEFT));
        }

        vzeroupper();
    } // StackFrame destructor inserts 'ret' here

    return dlp::jit::jitGeneratorError::success;
}

} // namespace amdzen::gen

// Explicit template instantiation
template class amdzen::gen::jitFP16GEMVM1<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
template class amdzen::gen::jitFP16GEMVN1<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
