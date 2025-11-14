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
jitF32GEMVN1<KType>::jitF32GEMVN1(void* buffer, size_t size)
    : Xbyak::CodeGenerator(size, buffer) // Call base class constructor
{
}

template<utils::kernelInstrType KType>
void
jitF32GEMVN1<KType>::initializeStackFrame(Xbyak::util::StackFrame& frame)
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
jitF32GEMVN1<KType>::initializeParameters(utils::gemvN1GeneratorParams& params)
{
    // Set dimensions from params
    MR               = params.MR; // Number of rows to process
    M_LEFT           = params.M_LEFT;
    yFormat          = params.yFormat;          // Storage format of C matrix
    alphaScalingType = params.alphaScalingType; // Type of alpha scaling
    betaScalingType  = params.betaScalingType;  // Type of beta scaling
    c_downscale      = params.c_downscale;

    RegBytes = Traits::regBytes;
    numRegs  = Traits::numRegs;

    simdWidth = RegBytes / sizeof(float); // For f32

    if (c_downscale < DLP_F32) {
        // Initialize F32→BF16 conversion constants on stack
        // Stack layout: [0-15]: final result, [16-47]: constants

        // Store constant 0x00010000 at [rsp + 16] for bit 16 extraction
        mov(regTmp1.cvt32(), 0x00010000);
        mov(dword[rsp + 16], regTmp1.cvt32());

        // Store constant 0x00007FFF at [rsp + 20] for rounding
        mov(regTmp1.cvt32(), 0x00007FFF);
        mov(dword[rsp + 20], regTmp1.cvt32());
    }

    // Load pointers and strides from the stack
    mov(regRsA, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, rsA)]);
    mov(regCsA, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, csA)]);
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, rsC)]);

    // Scale strides by data type size
    lea(regRsA, ptr[regRsA * sizeof(float)]);
    lea(regCsA, ptr[regCsA * sizeof(float)]);
    lea(regRsC, ptr[regRsC * sizeof(float)]);
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVN1<KType>::allocateRegisters()
{
    // Check if MR is valid
    if (MR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Allocate registers according to the rules:
    maskReg = 0; // Set this only when AVX512 codepath is disabled.

    // 1. Accumulation registers : MR registers for partial dot products
    accumReg     = MR;
    accumBaseIdx = numRegs - accumReg; // Start from the end

    yReg     = MR / simdWidth;
    yBaseIdx = numRegs - yReg; // Start from the end

    // NOTE : Before loading from y, we would be using MR registers from the end
    //        for accumulating alpha*A*B. This would then be reduced to MR/16
    //        registers, starting from accumBaseIdx. We would still have
    //        15*MR/16 registers left, which we would use for storing the
    //        result, indexed from yBaseIdx(which would be the last MR/16
    //        registers).

    // Ex : If MR is 16
    //      accumReg = 16
    //      accumBaseIdx = 32 - 16 = 16
    //      yReg = 16 / (64 / 4) = 1
    //      yBaseIdx = 31
    //      tmpReg = 4
    //      tmpBaseIdx = 0
    //      xReg = 31 - 16 - 4 = 11
    //      xBaseIdx = 11

    // registers to store and use for downscaling of f32 values
    if (c_downscale < DLP_F32) {
        cvtReg     = 3;
        cvtBaseIdx = yBaseIdx - cvtReg;
    }

    // Temporary registers (tmpReg): Use remaining registers for reduction
    tmpReg     = 4;
    tmpBaseIdx = 0; // To make sure we index YMM greater than 16

    // X registers (xReg): Use remaining registers for vector x
    xReg     = 1;
    xBaseIdx = tmpReg;

    maskBaseIdx = xBaseIdx + xReg;

    if (!Traits::hasMaskSupport) { // Native mask register-file is not supported
        maskReg = 2;
    }

    // We need to only consider accumReg, tmpReg, cvtReg and maskReg(in case of
    // ymm_16) for total register count. Check if we have enough registers
    if (maskBaseIdx + maskReg > accumBaseIdx) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitF32GEMVN1<KType>::regInit(int baseIdx, int numRegs)
{
    // Zero out accumulation registers
    vxorps(RegType(baseIdx), RegType(baseIdx), RegType(baseIdx));
    for (int i = 1; i < numRegs; i++) {
        vmovaps(RegType(baseIdx + i), RegType(baseIdx));
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVN1<KType>::loadAValues(int aRegIdx, bool isFringe)
{
    if (isFringe) {
        vmovups(RegType(tmpBaseIdx + aRegIdx) | mask_regs[0],
                ptr[regTmpAptr + regTmp1]);
    } else {
        vmovups(RegType(tmpBaseIdx + aRegIdx), ptr[regTmpAptr + regTmp1]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitF32GEMVN1<utils::kernelInstrType::avx2_ymm_16_reg>::loadAValues(
    int aRegIdx, bool isFringe)
{
    if (isFringe) {
        vmaskmovps(Xbyak::Ymm(tmpBaseIdx + aRegIdx), Xbyak::Ymm(maskBaseIdx),
                   ptr[regTmpAptr + regTmp1]);
    } else {
        vmovups(Xbyak::Ymm(tmpBaseIdx + aRegIdx), ptr[regTmpAptr + regTmp1]);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVN1<KType>::loadXValues(bool isFringe)
{
    if (isFringe) {
        vmovups(RegType(xBaseIdx) | mask_regs[0], ptr[regXptr]);
    } else {
        vmovups(RegType(xBaseIdx), ptr[regXptr]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitF32GEMVN1<utils::kernelInstrType::avx2_ymm_16_reg>::loadXValues(
    bool isFringe)
{
    if (isFringe) {
        vmaskmovps(Xbyak::Ymm(xBaseIdx), Xbyak::Ymm(maskBaseIdx), ptr[regXptr]);
    } else {
        vmovups(Xbyak::Ymm(xBaseIdx), ptr[regXptr]);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVN1<KType>::computeFMA(int aRegIdx, int accumRegIdx)
{
    vfmadd231ps(RegType(accumBaseIdx + accumRegIdx), RegType(xBaseIdx),
                RegType(tmpBaseIdx + aRegIdx));

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVN1<KType>::computeLoadFMA(int rowIdx, bool isFringe)
{
    if (isFringe) {
        vfmadd231ps(RegType(accumBaseIdx + rowIdx) | mask_regs[0],
                    RegType(xBaseIdx), ptr[regTmpAptr + regTmp1]);
    } else {
        vfmadd231ps(RegType(accumBaseIdx + rowIdx), RegType(xBaseIdx),
                    ptr[regTmpAptr + regTmp1]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVN1<KType>::processMRBlock(int mSize, bool isFringe)
{
    // Perform the compute over the MR rows
    int mLeft = mSize % 4;
    xor_(regTmp1, regTmp1);
    regInit(tmpBaseIdx, tmpReg);
    for (int i = 0; i < mSize / 4; i++) {

        for (int j = 0; j < 4; j++) {
            RETURN_IF_ERROR((loadAValues(j, isFringe)));
            RETURN_IF_ERROR((computeFMA(j, i * 4 + j)));

            add(regTmp1, regRsA);
        }
        // RETURN_IF_ERROR((computeLoadFMA(i, isFringe)));
        // add(regTmp1, regRsA);
    }

    for (int j = 0; j < mLeft; j++) {
        RETURN_IF_ERROR((loadAValues(j, isFringe)));
        RETURN_IF_ERROR((computeFMA(j, (mSize / 4) * 4 + j)));
        add(regTmp1, regRsA);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVN1<KType>::reduceToXmm(int startIdx, int tmpIdx, int blockSize)
{
    // Function only handles blocks of 4 or fewer ZMMs
    if (blockSize > 4) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Zero out the temporary registers we'll need
    for (int i = 0; i < 4; i++) {
        vxorps(Xbyak::Ymm(tmpIdx + i), Xbyak::Ymm(tmpIdx + i),
               Xbyak::Ymm(tmpIdx + i));
    }

    // Extract upper 256-bits and add to lower 256-bits for valid inputs
    // This extact + add logic is specific to AVX512 ISA, when using ZMM
    // registers. In case of using YMM registers, just move it onto temporary
    // registers.
    for (int i = 0; i < blockSize; i++) {
        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            // Extract upper 256-bits to temp YMM
            vextractf32x8(Xbyak::Ymm(tmpIdx + i), Xbyak::Zmm(startIdx + i), 1);
            // Add to lower 256-bits of input ZMM, storing in original ZMM's YMM
            // part
            vaddps(Xbyak::Ymm(tmpIdx + i), Xbyak::Ymm(tmpIdx + i),
                   Xbyak::Ymm(startIdx + i));
        } else {
            vmovups(Xbyak::Ymm(tmpIdx + i), Xbyak::Ymm(startIdx + i));
        }
    }

    // First round of horizontal adds
    vhaddps(Xbyak::Ymm(tmpIdx), Xbyak::Ymm(tmpIdx),
            Xbyak::Ymm(tmpIdx + 1)); // First pair (with zero if blockSize=1)

    // Second round of horizontal adds
    vhaddps(Xbyak::Ymm(tmpIdx + 2), Xbyak::Ymm(tmpIdx + 2),
            Xbyak::Ymm(tmpIdx + 3));

    // Third round of horizontal adds
    vhaddps(Xbyak::Ymm(tmpIdx), Xbyak::Ymm(tmpIdx), Xbyak::Ymm(tmpIdx + 2));

    // Final reduction from YMM to XMM
    vextractf128(Xbyak::Xmm(tmpIdx + 1), Xbyak::Ymm(tmpIdx),
                 1); // Extract upper 128-bits
    vaddps(Xbyak::Xmm(tmpIdx), Xbyak::Xmm(tmpIdx + 1),
           Xbyak::Xmm(tmpIdx)); // Add to lower 128-bits

    // Result is now in the XMM portion of startIdx
    // - For blockSize=1: Only first float is valid
    // - For blockSize=2: First two floats are valid
    // - For blockSize=3: First three floats are valid
    // - For blockSize=4: All four floats are valid
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVN1<KType>::reduceAccumulation(int mSize)
{
    // Process mSize registers in blocks of the simdWidth
    for (int i = 0; i < mSize; i += simdWidth) {
        // Number of registers to process in this ZMM block
        int blockSize = (mSize - i) < simdWidth ? (mSize - i) : simdWidth;

        // Process this block in groups of 4 registers
        for (int j = 0; j < blockSize; j += 4) {
            int subBlockSize = (blockSize - j) < 4 ? (blockSize - j) : 4;

            // Reduce 4 (or fewer) ZMMs to one XMM
            RETURN_IF_ERROR(
                (reduceToXmm(accumBaseIdx + i + j, tmpBaseIdx, subBlockSize)));

            // Insert the resulting XMM
            // into the appropriate
            // position in destination
            // ZMM
            if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
                vinsertf128(Xbyak::Ymm(accumBaseIdx + i / simdWidth),
                            Xbyak::Ymm(accumBaseIdx + i / simdWidth),
                            Xbyak::Xmm(tmpBaseIdx), j / 4);
            } else {
                vinsertf32x4(RegType(accumBaseIdx + i / simdWidth),
                             RegType(accumBaseIdx + i / simdWidth),
                             Xbyak::Xmm(tmpBaseIdx), j / 4);
            }
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVN1<KType>::scaleAccumulationWithAlpha(int mSize)
{
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, alpha)]);
    vbroadcastss(RegType(tmpBaseIdx), ptr[regKIter]);
    for (int i = 0; i < (mSize + simdWidth - 1) / simdWidth; i += 1) {
        vmulps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
               RegType(tmpBaseIdx));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVN1<KType>::scaleYWithBetaColStored(int mSize, bool betaOne)
{
    inLocalLabel();
    Xbyak::Label label_betaop_col, label_betaop_col_end;
    if (!betaOne) {
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, beta)]);
        vbroadcastss(RegType(xBaseIdx), ptr[regKIter]);
    }
    int mLeft = mSize % simdWidth;

    // beta scaling when output is bf16, where it is converted to accum type f32
    if (c_downscale < DLP_F32) {
        // Check for is_first_k
        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, is_first_k)]);
        test(regTmp2, regTmp2);
        je(label_betaop_col, T_NEAR);

        mov(regTmpYptr,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, buf_downscale)]);

        // NULL check
        cmp(regTmpYptr, 0);
        je(label_betaop_col, T_NEAR);

        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
        lea(regTmp2, ptr[regTmp2 * sizeof(bfloat16)]);

        add(regTmpYptr, regTmp2);

        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, rs_c_downscale)]);
        lea(regTmp2, ptr[regTmp2 * sizeof(bfloat16)]); // BF16 stride

        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, post_op_c_i)]);

        imul(regKIter, regTmp2);
        add(regTmpYptr, regKIter);

        // Store complete SIMD-width chunks
        for (int i = 0; i < mSize / simdWidth; i += 1) {
            movdqu(Xbyak::Xmm(tmpBaseIdx), ptr[regTmpYptr]);
            vpmovsxwd(Xbyak::Ymm(tmpBaseIdx), Xbyak::Xmm(tmpBaseIdx));
            vpslld(Xbyak::Ymm(tmpBaseIdx), Xbyak::Ymm(tmpBaseIdx), 16);
            vfmadd231ps(Xbyak::Ymm(accumBaseIdx + i), Xbyak::Ymm(xBaseIdx),
                        Xbyak::Ymm(tmpBaseIdx));
            lea(regTmpYptr, ptr[regTmpYptr + simdWidth * sizeof(bfloat16)]);
        }
        if (mLeft) {
            mov(regTmp2, mLeft);

            // Loop: Load n_remainder BF16 elements from C matrix to stack
            xor_(regKIter.cvt32(), regKIter.cvt32()); // elem_idx = 0

            Xbyak::Label loop_load_start, loop_load_end;
            L(loop_load_start);

            // Check if elem_idx < n_remainder
            cmp(regKIter.cvt32(), regTmp2.cvt32());
            jge(loop_load_end, T_NEAR);

            // Load BF16 value from C matrix to stack
            mov(regXptr.cvt16(), word[regTmpYptr + regKIter * sizeof(int16_t)]);
            mov(word[rsp + regKIter * sizeof(int16_t)], regXptr.cvt16());

            inc(regKIter.cvt32()); // elem_idx++
            jmp(loop_load_start, T_NEAR);

            L(loop_load_end);

            movdqu(Xbyak::Xmm(tmpBaseIdx), ptr[rsp]);
            vpmovsxwd(Xbyak::Ymm(tmpBaseIdx), Xbyak::Xmm(tmpBaseIdx));
            vpslld(Xbyak::Ymm(tmpBaseIdx), Xbyak::Ymm(tmpBaseIdx), 16);
            vfmadd231ps(Xbyak::Ymm(accumBaseIdx + (mSize / simdWidth)),
                        Xbyak::Ymm(xBaseIdx), Xbyak::Ymm(tmpBaseIdx));
        }

        jmp(label_betaop_col_end, T_NEAR);
        L(label_betaop_col);
    }

    for (int i = 0; i < mSize / simdWidth; i += 1) {
        if (betaOne) {
            vaddps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                   ptr[regTmpYptr]);
        } else {
            vfmadd231ps(RegType(accumBaseIdx + i), RegType(xBaseIdx),
                        ptr[regTmpYptr]);
        }
        lea(regTmpYptr, ptr[regTmpYptr + simdWidth * sizeof(float)]);
    }
    if (mLeft) {
        if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            vmaskmovps(Xbyak::Ymm(yBaseIdx), Xbyak::Ymm(maskBaseIdx + 1),
                       ptr[regTmpYptr]);
            if (betaOne) {
                vaddps(Xbyak::Ymm(accumBaseIdx + (mSize / simdWidth)),
                       Xbyak::Ymm(accumBaseIdx + (mSize / simdWidth)),
                       Xbyak::Ymm(yBaseIdx));
            } else {
                vfmadd231ps(Xbyak::Ymm(accumBaseIdx + (mSize / simdWidth)),
                            Xbyak::Ymm(xBaseIdx), Xbyak::Ymm(yBaseIdx));
            }
        } else {
            if (betaOne) {
                vaddps(RegType(accumBaseIdx + (mSize / simdWidth))
                           | mask_regs[1],
                       RegType(accumBaseIdx + (mSize / simdWidth)),
                       ptr[regTmpYptr]);
            } else {
                vfmadd231ps(RegType(accumBaseIdx + (mSize / simdWidth))
                                | mask_regs[1],
                            RegType(xBaseIdx), ptr[regTmpYptr]);
            }
        }
    }

    L(label_betaop_col_end);
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVN1<KType>::scaleYWithBetaRowStored(int mSize, bool betaOne)
{
    if (!betaOne) {
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, beta)]);
        vbroadcastss(RegType(xBaseIdx), ptr[regKIter]);
    }

    inLocalLabel();
    Xbyak::Label label_betaop_row, label_betaop_row_end;

    // beta scaling when output is bf16, where it is converted to accum type f32
    if (c_downscale < DLP_F32) {
        // Check for is_first_k
        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, is_first_k)]);
        test(regTmp2, regTmp2);
        je(label_betaop_row, T_NEAR);

        mov(regTmpYptr,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, buf_downscale)]);

        // NULL check
        cmp(regTmpYptr, 0);
        je(label_betaop_row, T_NEAR);

        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
        lea(regTmp2, ptr[regTmp2 * sizeof(bfloat16)]);

        add(regTmpYptr, regTmp2);

        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, rs_c_downscale)]);
        lea(regTmp2, ptr[regTmp2 * sizeof(bfloat16)]); // BF16 stride

        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, post_op_c_i)]);

        imul(regKIter, regTmp2);
        add(regTmpYptr, regKIter);

        // regTmp2 now contains BF16 stride, regTmpYptr points to downscale
        // buffer
        lea(regTmp3, ptr[regTmp2 + 2 * regTmp2]); // regTmp3 = stride + 2*stride

        for (int i = 0; i < (mSize + simdWidth - 1) / simdWidth; i += 1) {
            int blockSize  = ((mSize - i * simdWidth) < simdWidth)
                                 ? (mSize - i * simdWidth)
                                 : simdWidth;
            int num_blocks = blockSize / 4;
            int rem_block  = blockSize % 4;
            regInit(tmpBaseIdx, tmpReg);

            for (int j = 0; j < num_blocks; j += 1) {
                // Load 4 BF16 values and convert to F32 by placing in upper 16
                // bits
                vxorps(Xbyak::Xmm(tmpBaseIdx), Xbyak::Xmm(tmpBaseIdx),
                       Xbyak::Xmm(tmpBaseIdx));
                vpinsrw(Xbyak::Xmm(tmpBaseIdx), Xbyak::Xmm(tmpBaseIdx),
                        ptr[regTmpYptr], 1); // position 1 = bits 16-31
                vbroadcastss(RegType(tmpBaseIdx), Xbyak::Xmm(tmpBaseIdx));

                vxorps(Xbyak::Xmm(tmpBaseIdx + 1), Xbyak::Xmm(tmpBaseIdx + 1),
                       Xbyak::Xmm(tmpBaseIdx + 1));
                vpinsrw(Xbyak::Xmm(tmpBaseIdx + 1), Xbyak::Xmm(tmpBaseIdx + 1),
                        ptr[regTmpYptr + regTmp2], 1);
                vbroadcastss(RegType(tmpBaseIdx + 1),
                             Xbyak::Xmm(tmpBaseIdx + 1));

                vxorps(Xbyak::Xmm(tmpBaseIdx + 2), Xbyak::Xmm(tmpBaseIdx + 2),
                       Xbyak::Xmm(tmpBaseIdx + 2));
                vpinsrw(Xbyak::Xmm(tmpBaseIdx + 2), Xbyak::Xmm(tmpBaseIdx + 2),
                        ptr[regTmpYptr + 2 * regTmp2], 1);
                vbroadcastss(RegType(tmpBaseIdx + 2),
                             Xbyak::Xmm(tmpBaseIdx + 2));

                vxorps(Xbyak::Xmm(tmpBaseIdx + 3), Xbyak::Xmm(tmpBaseIdx + 3),
                       Xbyak::Xmm(tmpBaseIdx + 3));
                vpinsrw(Xbyak::Xmm(tmpBaseIdx + 3), Xbyak::Xmm(tmpBaseIdx + 3),
                        ptr[regTmpYptr + regTmp3], 1);
                vbroadcastss(RegType(tmpBaseIdx + 3),
                             Xbyak::Xmm(tmpBaseIdx + 3));

                // Now continue with EXACT same logic as F32 path
                vunpcklps(RegType(tmpBaseIdx), RegType(tmpBaseIdx),
                          RegType(tmpBaseIdx + 1));
                vunpcklps(RegType(tmpBaseIdx + 2), RegType(tmpBaseIdx + 2),
                          RegType(tmpBaseIdx + 3));
                vshufps(RegType(tmpBaseIdx), RegType(tmpBaseIdx),
                        RegType(tmpBaseIdx + 2), 0x44);

                vinsertf128(Xbyak::Ymm(yBaseIdx + i), Xbyak::Ymm(yBaseIdx + i),
                            Xbyak::Xmm(tmpBaseIdx), j);
                lea(regTmpYptr, ptr[regTmpYptr + regTmp2 * 4]);
            }

            if (rem_block) {
                // Handle remaining elements with same pattern
                switch (rem_block) {
                    case 3:
                        vxorps(Xbyak::Xmm(tmpBaseIdx + 2),
                               Xbyak::Xmm(tmpBaseIdx + 2),
                               Xbyak::Xmm(tmpBaseIdx + 2));
                        vpinsrw(Xbyak::Xmm(tmpBaseIdx + 2),
                                Xbyak::Xmm(tmpBaseIdx + 2),
                                ptr[regTmpYptr + regTmp2 * 2], 1);
                        vbroadcastss(RegType(tmpBaseIdx + 2),
                                     Xbyak::Xmm(tmpBaseIdx + 2));
                    case 2:
                        vxorps(Xbyak::Xmm(tmpBaseIdx + 1),
                               Xbyak::Xmm(tmpBaseIdx + 1),
                               Xbyak::Xmm(tmpBaseIdx + 1));
                        vpinsrw(Xbyak::Xmm(tmpBaseIdx + 1),
                                Xbyak::Xmm(tmpBaseIdx + 1),
                                ptr[regTmpYptr + regTmp2], 1);
                        vbroadcastss(RegType(tmpBaseIdx + 1),
                                     Xbyak::Xmm(tmpBaseIdx + 1));
                    case 1:
                        vxorps(Xbyak::Xmm(tmpBaseIdx), Xbyak::Xmm(tmpBaseIdx),
                               Xbyak::Xmm(tmpBaseIdx));
                        vpinsrw(Xbyak::Xmm(tmpBaseIdx), Xbyak::Xmm(tmpBaseIdx),
                                ptr[regTmpYptr], 1);
                        vbroadcastss(RegType(tmpBaseIdx),
                                     Xbyak::Xmm(tmpBaseIdx));
                    case 0:
                        break;
                }

                // Same unpack/shuffle logic as F32
                vunpcklps(RegType(tmpBaseIdx), RegType(tmpBaseIdx),
                          RegType(tmpBaseIdx + 1));
                vunpcklps(RegType(tmpBaseIdx + 2), RegType(tmpBaseIdx + 2),
                          RegType(tmpBaseIdx + 3));
                vshufps(RegType(tmpBaseIdx), RegType(tmpBaseIdx),
                        RegType(tmpBaseIdx + 2), 0x44);

                vinsertf128(Xbyak::Ymm(yBaseIdx + i), Xbyak::Ymm(yBaseIdx + i),
                            Xbyak::Xmm(tmpBaseIdx), num_blocks);
            }

            if (betaOne) {
                vaddps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(yBaseIdx + i));
            } else {
                vfmadd231ps(RegType(accumBaseIdx + i), RegType(xBaseIdx),
                            RegType(yBaseIdx + i));
            }
        }

        jmp(label_betaop_row_end, T_NEAR);
        L(label_betaop_row);
    }

    // F32 path (original code)
    // Store offsets for Y, using it's row-stride
    lea(regTmp3, ptr[regRsC + 2 * regRsC]); // regTmp3 = rsC + 2*rsC
    for (int i = 0; i < (mSize + simdWidth - 1) / simdWidth; i += 1) {
        int blockSize  = (mSize - i * simdWidth) < simdWidth
                             ? (mSize - i * simdWidth)
                             : simdWidth;
        int num_blocks = blockSize / 4;
        int rem_block  = blockSize % 4;
        regInit(tmpBaseIdx, tmpReg);
        for (int j = 0; j < num_blocks; j += 1) {
            vbroadcastss(RegType(tmpBaseIdx), ptr[regTmpYptr]);
            vbroadcastss(RegType(tmpBaseIdx + 1), ptr[regTmpYptr + regRsC]);
            vbroadcastss(RegType(tmpBaseIdx + 2), ptr[regTmpYptr + 2 * regRsC]);
            vbroadcastss(RegType(tmpBaseIdx + 3), ptr[regTmpYptr + regTmp3]);
            vunpcklps(RegType(tmpBaseIdx), RegType(tmpBaseIdx),
                      RegType(tmpBaseIdx + 1));
            vunpcklps(RegType(tmpBaseIdx + 2), RegType(tmpBaseIdx + 2),
                      RegType(tmpBaseIdx + 3));
            vshufps(RegType(tmpBaseIdx), RegType(tmpBaseIdx),
                    RegType(tmpBaseIdx + 2), 0x44);

            if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
                vinsertf128(Xbyak::Ymm(yBaseIdx + i), Xbyak::Ymm(yBaseIdx + i),
                            Xbyak::Xmm(tmpBaseIdx), j);
            } else {
                vinsertf32x4(RegType(yBaseIdx + i), RegType(yBaseIdx + i),
                             Xbyak::Xmm(tmpBaseIdx), j);
            }
            lea(regTmpYptr, ptr[regTmpYptr + regRsC * 4]);
        }
        if (rem_block) {
            switch (rem_block) {
                case 3:
                    vbroadcastss(RegType(tmpBaseIdx + 2),
                                 ptr[regTmpYptr + regRsC * 2]);
                case 2:
                    vbroadcastss(RegType(tmpBaseIdx + 1),
                                 ptr[regTmpYptr + regRsC]);
                case 1:
                    vbroadcastss(RegType(tmpBaseIdx), ptr[regTmpYptr]);
                case 0:
                    break;
            }
            vunpcklps(RegType(tmpBaseIdx), RegType(tmpBaseIdx),
                      RegType(tmpBaseIdx + 1));
            vunpcklps(RegType(tmpBaseIdx + 2), RegType(tmpBaseIdx + 2),
                      RegType(tmpBaseIdx + 3));
            vshufps(RegType(tmpBaseIdx), RegType(tmpBaseIdx),
                    RegType(tmpBaseIdx + 2), 0x44);
            if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
                vinsertf128(Xbyak::Ymm(yBaseIdx + i), Xbyak::Ymm(yBaseIdx + i),
                            Xbyak::Xmm(tmpBaseIdx), num_blocks);
            } else {
                vinsertf32x4(RegType(yBaseIdx + i), RegType(yBaseIdx + i),
                             Xbyak::Xmm(tmpBaseIdx), num_blocks);
            }
        }

        if (betaOne) {
            vaddps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                   RegType(yBaseIdx + i));
        } else {
            vfmadd231ps(RegType(accumBaseIdx + i), RegType(xBaseIdx),
                        RegType(yBaseIdx + i));
        }
    }

    L(label_betaop_row_end);
    outLocalLabel();
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVN1<KType>::scaleYWithBeta(int mSize)
{
    bool is_beta_one = (betaScalingType == dlp::kernel_frame::scalingType::one);
    if (betaScalingType != dlp::kernel_frame::scalingType::zero) {
        mov(regTmpYptr, regYptr);
        if (yFormat == dlp::kernel_frame::storageFormat::colMajor) {
            RETURN_IF_ERROR((scaleYWithBetaColStored(mSize, is_beta_one)));
        } else {
            RETURN_IF_ERROR((scaleYWithBetaRowStored(mSize, is_beta_one)));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVN1<KType>::convertF32toBF16(int scratch1, int scratch2, int destIdx)
{
    return dlp::jit::jitGeneratorError::notSupported;
}

template<>
dlp::jit::jitGeneratorError
jitF32GEMVN1<utils::kernelInstrType::avx2_ymm_16_reg>::convertF32toBF16(
    int scratch1, int scratch2, int destIdx)
{
    vbroadcastss(Xbyak::Ymm(scratch1),
                 ptr[rsp + 16]); // Load 0x00010000
    vpand(Xbyak::Ymm(scratch1), Xbyak::Ymm(destIdx),
          Xbyak::Ymm(scratch1)); // Extract bit 16
    vpsrld(Xbyak::Ymm(scratch1), Xbyak::Ymm(scratch1),
           16); // Shift to position 0 → tlsb

    vbroadcastss(Xbyak::Ymm(scratch2),
                 ptr[rsp + 20]); // Load 0x00007FFF
    vpaddd(Xbyak::Ymm(scratch2), Xbyak::Ymm(destIdx),
           Xbyak::Ymm(scratch2)); // Add rounding to original

    vpaddd(Xbyak::Ymm(scratch2), Xbyak::Ymm(scratch2),
           Xbyak::Ymm(scratch1)); // Add tlsb → rounded

    vpsrld(Xbyak::Ymm(scratch2), Xbyak::Ymm(scratch2),
           16); // Shift right 16 bits

    // Extract upper 128 bits of YMM → XMM
    vextracti128(Xbyak::Xmm(scratch1), Xbyak::Ymm(scratch2), 1);

    // Pack 8×32-bit to 8×16-bit
    vpackusdw(Xbyak::Xmm(scratch2), Xbyak::Xmm(scratch2), Xbyak::Xmm(scratch1));

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVN1<KType>::storeYValuesColStored(int mSize)
{
    int mLeft = mSize % simdWidth;
    inLocalLabel();
    Xbyak::Label label_storeop_col, label_storeop_col_end;

    // downscaling accum to bf16 before store, when output is of type bf16
    if (c_downscale < DLP_F32) {
        // Check for is_last_k
        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, is_last_k)]);
        test(regTmp2, regTmp2);
        je(label_storeop_col, T_NEAR);

        mov(regTmpYptr,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, buf_downscale)]);

        // NULL check
        cmp(regTmpYptr, 0);
        je(label_storeop_col, T_NEAR);

        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
        lea(regTmp2, ptr[regTmp2 * sizeof(bfloat16)]);

        add(regTmpYptr, regTmp2);

        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, rs_c_downscale)]);
        lea(regTmp2, ptr[regTmp2 * sizeof(bfloat16)]); // BF16 stride

        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, post_op_c_i)]);

        imul(regKIter, regTmp2);
        add(regTmpYptr, regKIter);

        // Store complete SIMD-width chunks
        for (int i = 0; i < mSize / simdWidth; i += 1) {
            RETURN_IF_ERROR(
                convertF32toBF16(tmpBaseIdx, tmpBaseIdx + 1, accumBaseIdx + i));
            movdqu(ptr[regTmpYptr], Xbyak::Xmm(tmpBaseIdx + 1));
            lea(regTmpYptr, ptr[regTmpYptr + simdWidth * sizeof(bfloat16)]);
        }
        if (mLeft) {
            RETURN_IF_ERROR(
                convertF32toBF16(tmpBaseIdx, tmpBaseIdx + 1,
                                 accumBaseIdx + (mSize / simdWidth)));

            // Now Xmm(scratch2) contains 8×BF16 values
            // Store the BF16 result to stack for element-wise access
            movdqu(ptr[rsp + 0],
                   Xbyak::Xmm(tmpBaseIdx + 1)); // 8×16-bit to stack

            // Get n_remainder: n % 8
            mov(regTmp2, mLeft);
            and_(regTmp2.cvt32(), 7); // n % 8

            // Calculate destination base address
            // lea(regKIter, ptr[regTmpCptr + bFullReg * halfRegBytes]);

            // Loop: copy n_remainder elements from stack to destination
            xor_(regKIter.cvt32(), regKIter.cvt32()); // elem_idx = 0

            Xbyak::Label loop_start, loop_end;
            L(loop_start);

            // Check if elem_idx < n_remainder
            cmp(regKIter.cvt32(), regTmp2.cvt32());
            jge(loop_end, T_NEAR);

            // Load BF16 value from stack and store to destination
            // Use regYptr as temporary (done with B matrix access)
            mov(regXptr.cvt16(), word[rsp + regKIter * sizeof(int16_t)]);
            mov(word[regTmpYptr + regKIter * sizeof(int16_t)], regXptr.cvt16());

            inc(regKIter.cvt32()); // elem_idx++
            jmp(loop_start, T_NEAR);

            L(loop_end);
        }

        jmp(label_storeop_col_end, T_NEAR);
        L(label_storeop_col);
    }

    for (int i = 0; i < mSize / simdWidth; i += 1) {
        vmovups(ptr[regTmpYptr], RegType(accumBaseIdx + i));
        lea(regTmpYptr, ptr[regTmpYptr + simdWidth * sizeof(float)]);
    }
    if (mLeft) {
        if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            vmaskmovps(ptr[regTmpYptr], Xbyak::Ymm(maskBaseIdx + 1),
                       Xbyak::Ymm(accumBaseIdx + (mSize / simdWidth)));
        } else {
            vmovups(ptr[regTmpYptr] | mask_regs[1],
                    RegType(accumBaseIdx + (mSize / simdWidth)));
        }
    }

    L(label_storeop_col_end);
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVN1<KType>::storeYValuesRowStored(int mSize)
{

    inLocalLabel();
    for (int i = 0; i < (mSize + simdWidth - 1) / simdWidth; i++) {
        int elements_in_reg = (i < mSize / simdWidth) ? simdWidth
                                                      : (mSize % simdWidth);
        if (elements_in_reg == 0)
            break;

        // Extract 4 chunks of 128-bits (4 floats each) from the ZMM
        for (int j = 0; j < elements_in_reg; j += 4) {
            if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
                vextractf128(Xbyak::Xmm(tmpBaseIdx + j / 4),
                             RegType(accumBaseIdx + i), j / 4);
            } else {
                vextractf32x4(Xbyak::Xmm(tmpBaseIdx + j / 4), // ISA specific
                              RegType(accumBaseIdx + i), j / 4);
            }
        }

        Xbyak::Label label_storeop_row, label_storeop_row_end;

        // downscaling accum to bf16 before store, when output is of type bf16
        if (c_downscale < DLP_F32) {
            mov(regTmp2,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                    + offsetof(lpgemm_post_op_attr, is_last_k)]);
            test(regTmp2, regTmp2);
            je(label_storeop_row, T_NEAR);

            mov(regTmpYptr,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                    + offsetof(lpgemm_post_op_attr, buf_downscale)]);

            // NULL check
            cmp(regTmpYptr, 0);
            je(label_storeop_row, T_NEAR);

            mov(regTmp2,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                    + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
            lea(regTmp2, ptr[regTmp2 * sizeof(bfloat16)]);

            add(regTmpYptr, regTmp2);

            mov(regTmp2,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                    + offsetof(lpgemm_post_op_attr, rs_c_downscale)]);
            lea(regTmp2, ptr[regTmp2 * sizeof(bfloat16)]); // BF16 stride

            mov(regKIter,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                    + offsetof(lpgemm_post_op_attr, post_op_c_i)]);

            imul(regKIter, regTmp2);
            add(regTmpYptr, regKIter);

            for (int j = 0; j < (elements_in_reg + 3) / 4; j++) {
                RETURN_IF_ERROR(convertF32toBF16(
                    cvtBaseIdx, (cvtBaseIdx + 1 + j), tmpBaseIdx + j));
            }

            // Now store each extracted value to its proper row-strided location
            for (int j = 0; j < elements_in_reg; j++) {
                int tmp_reg    = j / 4; // Which temp register has our value
                int pos_in_reg = j % 4; // Position within that temp register

                vpextrw(ptr[regTmpYptr], Xbyak::Xmm(cvtBaseIdx + 1 + tmp_reg),
                        pos_in_reg);

                // Move to next row
                add(regTmpYptr, regTmp2);
            }

            jmp(label_storeop_row_end, T_NEAR);
            L(label_storeop_row);
        }

        // Now store each extracted value to its proper row-strided location
        for (int j = 0; j < elements_in_reg; j++) {
            int tmp_reg    = j / 4; // Which temp register has our value
            int pos_in_reg = j % 4; // Position within that temp register

            if (pos_in_reg == 0) {
                // First element in XMM can be stored directly
                vmovss(ptr[regTmpYptr], Xbyak::Xmm(tmpBaseIdx + tmp_reg));
            } else {
                // Extract the 32-bit float to memory directly
                vpextrd(ptr[regTmpYptr], Xbyak::Xmm(tmpBaseIdx + tmp_reg),
                        pos_in_reg);
            }

            // Move to next row
            add(regTmpYptr, regRsC);
        }

        L(label_storeop_row_end);
    }
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVN1<KType>::storeYValues(int mSize)
{
    // Store values from Y
    mov(regTmpYptr, regYptr);
    if (yFormat == dlp::kernel_frame::storageFormat::colMajor) {
        RETURN_IF_ERROR((storeYValuesColStored(mSize)));
    } else {
        RETURN_IF_ERROR((storeYValuesRowStored(mSize)));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVN1<KType>::loadMasks()
{
    // Ensuring mapping only from k1 to k7(to avoid k0 usage internally)
    for (int i = 0; i < NUM_USABLE_MASKS; i++) {
        mask_regs[i] = Xbyak::Opmask(MASK_START_IDX + i);
    }

    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        // Loading the masks
        kmovw(
            mask_regs[0],
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kmask_avx512)]);
        kmovw(
            mask_regs[1],
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, mmask_avx512)]);
    } else if constexpr (KType == utils::kernelInstrType::avx512_ymm_32_reg) {
        kmovb(mask_regs[0],
              ptr[stackPtr
                  + offsetof(dlp::kernels::gemvN1Params, kmask_avx512_256)]);
        kmovb(mask_regs[1],
              ptr[stackPtr
                  + offsetof(dlp::kernels::gemvN1Params, mmask_avx512_256)]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitF32GEMVN1<utils::kernelInstrType::avx2_ymm_16_reg>::loadMasks()
{
    lea(regKIter,
        ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kmask_avx2)]);
    vmovdqu(Xbyak::Ymm(maskBaseIdx), ptr[regKIter]);
    lea(regKIter,
        ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, mmask_avx2)]);
    vmovdqu(Xbyak::Ymm(maskBaseIdx + 1), ptr[regKIter]);
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVN1<KType>::generateKernel(utils::gemvN1GeneratorParams& params)
{

    Xbyak::util::StackFrame frame(this, 1, 13, 48);
    initializeStackFrame(frame);

    initializeParameters(params);
    RETURN_IF_ERROR((allocateRegisters()));

    // Acquire the addresses of A and Y
    mov(regAptr, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, a)]);
    mov(regYptr, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, y)]);

    inLocalLabel();

    loadMasks();

    // Create kernel ops handler once for the entire kernel
    std::unique_ptr<gen::kernelOpsHandler> kernelOpsHandlerPtr;
    if (!params.kernelOps.empty()) {
        kernelOpsHandlerPtr =
            std::make_unique<gen::kernelOpsHandler>(this, params.kType);
    }

    // Set the for-loop sequence for m-dimension
    if (params.mloop) {
        mov(regMIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, m_iter)]);
        test(regMIter, regMIter);
        jz(label_m_loop_end, T_NEAR);
        L(label_m_loop_start);
        // }

        // Zero out accumulator registers for this m iteration
        regInit(accumBaseIdx, MR);

        // Y prefetch, before the k-loop
        if (betaScalingType != dlp::kernel_frame::scalingType::zero) {
            prefetcht0(ptr[regYptr]);
        }

        // K-loop is not needed if alpha is zero
        if (params.alphaScalingType != dlp::kernel_frame::scalingType::zero) {
            // Acquire the pointers for A
            // One is used in the m-loop, while other in the k-loop
            mov(regTmpAptr, regAptr);
            mov(regTmpYptr, regAptr);

            // Acquire the address of X
            mov(regXptr,
                ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, x)]);

            // Set the for-loop sequence for k-dimension
            if (params.kloop) {
                mov(regKIter,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvN1Params, k_iter)]);
                test(regKIter, regKIter);
                jz(label_m_loop_k_loop_end, T_NEAR);
                L(label_m_loop_k_loop_start);

                // Load the X vector
                RETURN_IF_ERROR((loadXValues()));

                // Process all rows including fringe
                RETURN_IF_ERROR((processMRBlock(MR)));

                // Save current A pointer and update pointers for next k
                // iteration
                mov(regTmp1, simdWidth);
                imul(regTmp1, regCsA);
                add(regTmpYptr, regTmp1);
                mov(regTmpAptr, regTmpYptr);
                add(regXptr, RegBytes); // Since B will be unit-strided

                dec(regKIter);
                jnz(label_m_loop_k_loop_start, T_NEAR);
            }
            L(label_m_loop_k_loop_end);

            if (params.kfringe) {
                mov(regKIter,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvN1Params, k_left)]);
                test(regKIter, regKIter);
                jz(label_m_loop_k_fringe_end, T_NEAR);
                L(label_m_loop_k_fringe_start);

                RETURN_IF_ERROR((loadXValues(true)));
                RETURN_IF_ERROR((processMRBlock(MR, true)));
            }

            L(label_m_loop_k_fringe_end);

            // Reduce the accumulation registers to XMMs, and put it in
            // ZMMs
            reduceAccumulation(MR);

            // Alpha scaling
            if (params.alphaScalingType
                != dlp::kernel_frame::scalingType::one) {
                scaleAccumulationWithAlpha(MR);
            }
        }

        // Working good for element-wise loads/stores for C.
        scaleYWithBeta(MR);

        if (kernelOpsHandlerPtr) {
            RETURN_IF_ERROR((kernelOpsHandlerPtr->generateKernelOps(
                params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemv_n1,
                params.MR, 1, false, 1, accumBaseIdx, yReg)));

            kernelOpsHandlerPtr->generateKernelOpsAttributes();

            // For avx2 config, we use YMM registers for masks
            // these mask registers may have been used as scratch registers
            // by the post-ops handler. Thus, we need to reload them.
            if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
                loadMasks();
            }
        }

        storeYValues(MR);

        // if (params.mloop) {
        // Update pointers for next m iteration(for A and y)
        mov(regTmp2, MR);
        imul(regTmp2, regRsA);
        add(regAptr, regTmp2);
        mov(regTmp1, MR);
        imul(regTmp1, regRsC);
        add(regYptr, regTmp1);

        // Update post_op_c_i for the next m-iteration (similar to GEMM pattern)
        // This ensures each iteration uses the correct offset for post-ops
        if (c_downscale < DLP_F32 || !params.kernelOps.empty()) {
            mov(regTmp1,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                    + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
            add(regTmp1, MR);
            mov(ptr[stackPtr
                    + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                    + offsetof(lpgemm_post_op_attr, post_op_c_i)],
                regTmp1);
        }

        dec(regMIter);
        jnz(label_m_loop_start, T_NEAR);
    }

    L(label_m_loop_end);
    if (params.mfringe) {

        mov(regMIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, m_left)]);
        test(regMIter, regMIter);
        jz(label_m_fringe_end, T_NEAR);
        L(label_m_fringe_start);

        regInit(accumBaseIdx, M_LEFT);

        // K-loop is not needed if alpha is zero
        if (params.alphaScalingType != dlp::kernel_frame::scalingType::zero) {
            // Acquire the pointers for A
            mov(regTmpAptr, regAptr);
            mov(regTmpYptr, regAptr);

            // Acquire the address of X
            mov(regXptr,
                ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, x)]);

            // Set the for-loop sequence for k-dimension
            if (params.kloop) {
                mov(regKIter,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvN1Params, k_iter)]);
                test(regKIter, regKIter);
                jz(label_m_fringe_k_loop_end, T_NEAR);
                L(label_m_fringe_k_loop_start);

                // Load the X vector
                RETURN_IF_ERROR((loadXValues()));

                // Process all rows including fringe
                RETURN_IF_ERROR((processMRBlock(M_LEFT)));

                // Update pointers for next k iteration
                mov(regTmp1, simdWidth);
                imul(regTmp1, regCsA);
                add(regTmpYptr, regTmp1);
                mov(regTmpAptr, regTmpYptr);
                add(regXptr, RegBytes); // Since B will be unit-strided

                dec(regKIter);
                jnz(label_m_fringe_k_loop_start, T_NEAR);
            }
            L(label_m_fringe_k_loop_end);
            if (params.kfringe) {
                mov(regKIter,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvN1Params, k_left)]);
                test(regKIter, regKIter);
                jz(label_m_fringe_k_fringe_end, T_NEAR);
                L(label_m_fringe_k_fringe_start);

                RETURN_IF_ERROR((loadXValues(true)));

                RETURN_IF_ERROR((processMRBlock(M_LEFT, true)));
            }
            L(label_m_fringe_k_fringe_end);

            // Reduce the accumulation registers to XMMs, and put it in
            // ZMMs
            reduceAccumulation(M_LEFT);
            // Alpha scaling
            if (params.alphaScalingType
                != dlp::kernel_frame::scalingType::one) {
                scaleAccumulationWithAlpha(M_LEFT);
            }
        }

        scaleYWithBeta(M_LEFT);

        if (kernelOpsHandlerPtr) {
            RETURN_IF_ERROR((kernelOpsHandlerPtr->generateKernelOps(
                params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemv_n1,
                params.M_LEFT, 1, true, 1, accumBaseIdx, M_LEFT / simdWidth)));

            // This call will skip embedding tables (already done in main loop)
            kernelOpsHandlerPtr->generateKernelOpsAttributes();

            // For avx2 config, we use YMM registers for masks
            // these mask registers may have been used as scratch registers
            // by the post-ops handler. Thus, we need to reload them.
            if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
                loadMasks();
            }
        }

        storeYValues(M_LEFT);
    }
    L(label_m_fringe_end);
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

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
jitF32GEMVM1<KType>::initializeParameters(utils::gemvM1GeneratorParams& params)
{
    NR               = params.NR;
    N_LEFT           = params.N_LEFT;
    KC               = params.KC;
    K_SUB_ITER       = params.K_SUB_ITER;
    yFormat          = params.yFormat;
    alphaScalingType = params.alphaScalingType;
    betaScalingType  = params.betaScalingType;
    mtag_b           = params.mtag_b;

    c_downscale = params.c_downscale;

    RegBytes = Traits::regBytes;
    numRegs  = Traits::numRegs;

    simdWidth = RegBytes / sizeof(float); // For f32

    if (c_downscale < DLP_F32) {
        // Initialize F32→BF16 conversion constants on stack
        // Stack layout: [0-15]: final result, [16-47]: constants

        // Store constant 0x00010000 at [rsp + 16] for bit 16 extraction
        mov(regTmp1.cvt32(), 0x00010000);
        mov(dword[rsp + 16], regTmp1.cvt32());

        // Store constant 0x00007FFF at [rsp + 20] for rounding
        mov(regTmp1.cvt32(), 0x00007FFF);
        mov(dword[rsp + 20], regTmp1.cvt32());
    }

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
    tmpReg   = yReg;
    maskReg  = 0; // Set this only when AVX512 codepath is disabled.

    // Direct addressing mode on FMA instructions are avoided here, since
    // we could initiate loads eariler with explicit loads.
    // Thus, both x and B loads are done into registers.
    accumBaseIdx = numRegs - accumReg;
    xBaseIdx     = accumBaseIdx - xReg;
    yBaseIdx     = numRegs - yReg;
    bBaseIdx     = xBaseIdx - bReg;
    maskBaseIdx  = bBaseIdx; // Set this only when AVX512 codepath is disabled.

    // allocting temp reg for storing downscaled values
    if (c_downscale < DLP_F32) {
        tmpBaseIdx  = bBaseIdx - tmpReg;
        maskBaseIdx = tmpBaseIdx;
    }

    if (!Traits::hasMaskSupport) { // Native mask register-file is not supported
                                   // by the architecture.
        maskReg     = 1;
        maskBaseIdx = maskBaseIdx - maskReg;
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

    // Load the masks
    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        kmovw(
            mask_regs[0],
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, nmask_avx512)]);
    } else if constexpr (KType == utils::kernelInstrType::avx512_ymm_32_reg) {
        kmovb(mask_regs[0],
              ptr[stackPtr
                  + offsetof(dlp::kernels::gemvM1Params, nmask_avx512_256)]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitF32GEMVM1<utils::kernelInstrType::avx2_ymm_16_reg>::loadMasks()
{
    lea(regNIter,
        ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, nmask_avx2)]);
    vmovdqu(Xbyak::Ymm(maskBaseIdx), ptr[regNIter]);

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
jitF32GEMVM1<KType>::computeKxnfringe()
{
    for (int j = 0; j < K_SUB_ITER; j++) {
        vbroadcastss(RegType(xBaseIdx + j), ptr[regXptr + j * sizeof(float)]);
    }

    int n_iter = N_LEFT / simdWidth;
    int n_left = N_LEFT % simdWidth;
    for (int i = 0; i < n_iter; i++) {
        for (int j = 0; j < K_SUB_ITER; j++) {
            offsetBPtr(j); // Calculated into regTmp1
            vmovups(RegType(bBaseIdx + j), ptr[regTmp2 + regTmp1]);
            vfmadd231ps(RegType(accumBaseIdx + K_SUB_ITER * i + j),
                        RegType(xBaseIdx + j), RegType(bBaseIdx + j));
        }
        add(regTmp2, simdWidth * sizeof(float));
    }
    if (n_left) {
        for (int j = 0; j < K_SUB_ITER; j++) {
            offsetBPtr(j); // Calculated into regTmp1
            maskLoadB(j, 0);
            vfmadd231ps(RegType(accumBaseIdx + K_SUB_ITER * n_iter + j),
                        RegType(xBaseIdx + j), RegType(bBaseIdx + j));
        }
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

    int j      = 0;
    int n_iter = N_LEFT / simdWidth;
    int n_left = N_LEFT % simdWidth;
    for (int i = 0; i < n_iter; i++) {
        j = i % K_SUB_ITER;
        xor_(regTmp1, regTmp1);
        lea(regTmp1, ptr[regTmp1 + i * simdWidth * sizeof(float)]);
        vmovups(RegType(bBaseIdx + j), ptr[regTmp2 + regTmp1]);
        vfmadd231ps(RegType(accumBaseIdx + K_SUB_ITER * i), RegType(xBaseIdx),
                    RegType(bBaseIdx + j));
    }
    if (n_left) {
        j = 0;
        xor_(regTmp1, regTmp1);
        lea(regTmp1, ptr[regTmp1 + n_iter * simdWidth * sizeof(float)]);
        maskLoadB(j, 0);
        vfmadd231ps(RegType(accumBaseIdx + K_SUB_ITER * n_iter),
                    RegType(xBaseIdx), RegType(bBaseIdx + j));
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
    int n_iter = N_LEFT / simdWidth;
    int n_left = N_LEFT % simdWidth;
    if (!isBetaOne) {
        for (int i = 0; i < n_iter; i++) {
            vfmadd231ps(RegType(accumBaseIdx + i), RegType(xBaseIdx),
                        ptr[regTmpYptr + i * simdWidth * sizeof(float)]);
        }
        if (n_left) {
            vfmadd231ps(RegType(accumBaseIdx + n_iter) | mask_regs[0],
                        RegType(xBaseIdx),
                        ptr[regTmpYptr + n_iter * simdWidth * sizeof(float)]);
        }
    } else {
        for (int i = 0; i < n_iter; i++) {
            vaddps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                   ptr[regTmpYptr + i * simdWidth * sizeof(float)]);
        }
        if (n_left) {
            vaddps(RegType(accumBaseIdx + n_iter) | mask_regs[0],
                   RegType(accumBaseIdx + n_iter),
                   ptr[regTmpYptr + n_iter * simdWidth * sizeof(float)]);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitF32GEMVM1<utils::kernelInstrType::avx2_ymm_16_reg>::scaleYWithBetaFringe(
    bool isBetaOne)
{

    inLocalLabel();
    Xbyak::Label label_beta_scale_fringe, label_beta_scale_fringe_end;

    int n_iter = N_LEFT / simdWidth;
    int n_left = N_LEFT % simdWidth;

    // Handle downscaling of Y and scale beta
    if (c_downscale < DLP_F32) {
        // Check for is_first_k
        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, is_first_k)]);
        test(regTmp2, regTmp2);
        je(label_beta_scale_fringe, T_NEAR);

        // Get the downscale buffer pointer
        mov(regTmpYptr,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, buf_downscale)]);

        // NULL check
        cmp(regTmpYptr, 0);
        je(label_beta_scale_fringe, T_NEAR);

        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
        lea(regTmp2, ptr[regTmp2 * sizeof(bfloat16)]);

        add(regTmpYptr, regTmp2);

        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, rs_c_downscale)]);
        lea(regTmp2, ptr[regTmp2 * sizeof(bfloat16)]); // BF16 stride

        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, post_op_c_i)]);

        imul(regKIter, regTmp2);
        add(regTmpYptr, regKIter);

        // Store complete SIMD-width chunks
        for (int i = 0; i < n_iter; i += 1) {
            movdqu(Xbyak::Xmm(tmpBaseIdx), ptr[regTmpYptr]);
            vpmovsxwd(Xbyak::Ymm(tmpBaseIdx), Xbyak::Xmm(tmpBaseIdx));
            vpslld(Xbyak::Ymm(tmpBaseIdx), Xbyak::Ymm(tmpBaseIdx), 16);
            vfmadd231ps(Xbyak::Ymm(accumBaseIdx + i), Xbyak::Ymm(xBaseIdx),
                        Xbyak::Ymm(tmpBaseIdx));
            lea(regTmpYptr, ptr[regTmpYptr + simdWidth * sizeof(bfloat16)]);
        }
        if (n_left) {
            mov(regTmp2, n_left);
            // and_(regTmp2.cvt32(), 7); // n % 8

            // Calculate destination base address for fringe BF16 elements
            // lea(regKIter, ptr[regTmpCptr + bFullReg * halfRegBytes]);

            // Loop: Load n_remainder BF16 elements from C matrix to stack
            xor_(regKIter.cvt32(), regKIter.cvt32()); // elem_idx = 0

            Xbyak::Label loop_load_start, loop_load_end;
            L(loop_load_start);

            // Check if elem_idx < n_remainder
            cmp(regKIter.cvt32(), regTmp2.cvt32());
            jge(loop_load_end, T_NEAR);

            // Load BF16 value from C matrix to stack
            mov(regBptr.cvt16(), word[regTmpYptr + regKIter * sizeof(int16_t)]);
            mov(word[rsp + regKIter * sizeof(int16_t)], regBptr.cvt16());

            inc(regKIter.cvt32()); // elem_idx++
            jmp(loop_load_start, T_NEAR);

            L(loop_load_end);

            movdqu(Xbyak::Xmm(tmpBaseIdx), ptr[rsp]);
            vpmovsxwd(Xbyak::Ymm(tmpBaseIdx), Xbyak::Xmm(tmpBaseIdx));
            vpslld(Xbyak::Ymm(tmpBaseIdx), Xbyak::Ymm(tmpBaseIdx), 16);
            vfmadd231ps(Xbyak::Ymm(accumBaseIdx + n_iter), Xbyak::Ymm(xBaseIdx),
                        Xbyak::Ymm(tmpBaseIdx));
        }

        jmp(label_beta_scale_fringe_end, T_NEAR);
        L(label_beta_scale_fringe);
    }

    if (!isBetaOne) {
        for (int i = 0; i < n_iter; i++) {
            vfmadd231ps(Xbyak::Ymm(accumBaseIdx + i), Xbyak::Ymm(xBaseIdx),
                        ptr[regTmpYptr + i * simdWidth * sizeof(float)]);
        }
        if (n_left) {
            vmaskmovps(Xbyak::Ymm(yBaseIdx + n_iter), Xbyak::Ymm(maskBaseIdx),
                       ptr[regTmpYptr + n_iter * simdWidth * sizeof(float)]);
            vfmadd231ps(Xbyak::Ymm(accumBaseIdx + n_iter), Xbyak::Ymm(xBaseIdx),
                        Xbyak::Ymm(yBaseIdx + n_iter));
        }
    } else {
        for (int i = 0; i < n_iter; i++) {
            vaddps(Xbyak::Ymm(accumBaseIdx + i), Xbyak::Ymm(accumBaseIdx + i),
                   ptr[regTmpYptr + i * simdWidth * sizeof(float)]);
        }
        if (n_left) {
            vmaskmovps(Xbyak::Ymm(yBaseIdx + n_iter), Xbyak::Ymm(maskBaseIdx),
                       ptr[regTmpYptr + n_iter * simdWidth * sizeof(float)]);
            vaddps(Xbyak::Ymm(accumBaseIdx + n_iter),
                   Xbyak::Ymm(accumBaseIdx + n_iter),
                   Xbyak::Ymm(yBaseIdx + n_iter));
        }
    }

    L(label_beta_scale_fringe_end);
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::scaleYWithBeta(bool nMask)
{
    inLocalLabel();
    Xbyak::Label label_beta_scale, label_beta_scale_end;

    bool isBetaZero = (betaScalingType == dlp::kernel_frame::scalingType::zero);
    bool isBetaOne  = (betaScalingType == dlp::kernel_frame::scalingType::one);

    mov(regTmpYptr, regYptr);

    if (!isBetaZero) {
        if (!isBetaOne) {
            mov(regKSubIter,
                ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, beta)]);
            vbroadcastss(RegType(xBaseIdx), ptr[regKSubIter]);
        }

        if (!nMask) {
            // Handle downscaling of Y and scale beta
            if (c_downscale < DLP_F32) {
                // Check for is_first_k
                mov(regTmp2,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                        + offsetof(lpgemm_post_op_attr, is_first_k)]);
                test(regTmp2, regTmp2);
                je(label_beta_scale, T_NEAR);

                // Get the downscale buffer pointer
                mov(regTmpYptr,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                        + offsetof(lpgemm_post_op_attr, buf_downscale)]);

                // NULL check
                cmp(regTmpYptr, 0);
                je(label_beta_scale, T_NEAR);

                mov(regTmp2,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                        + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
                lea(regTmp2, ptr[regTmp2 * sizeof(bfloat16)]);

                add(regTmpYptr, regTmp2);

                mov(regTmp2,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                        + offsetof(lpgemm_post_op_attr, rs_c_downscale)]);
                lea(regTmp2, ptr[regTmp2 * sizeof(bfloat16)]); // BF16 stride

                mov(regKIter,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                        + offsetof(lpgemm_post_op_attr, post_op_c_i)]);

                imul(regKIter, regTmp2);
                add(regTmpYptr, regKIter);

                // Store complete SIMD-width chunks
                for (int i = 0; i < NR / simdWidth; i += 1) {
                    movdqu(Xbyak::Xmm(tmpBaseIdx), ptr[regTmpYptr]);
                    vpmovsxwd(Xbyak::Ymm(tmpBaseIdx), Xbyak::Xmm(tmpBaseIdx));
                    vpslld(Xbyak::Ymm(tmpBaseIdx), Xbyak::Ymm(tmpBaseIdx), 16);
                    vfmadd231ps(Xbyak::Ymm(accumBaseIdx + i),
                                Xbyak::Ymm(xBaseIdx), Xbyak::Ymm(tmpBaseIdx));
                    lea(regTmpYptr,
                        ptr[regTmpYptr + simdWidth * sizeof(bfloat16)]);
                }

                jmp(label_beta_scale_end, T_NEAR);
                L(label_beta_scale);
            }

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

        L(label_beta_scale_end);
    }

    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::convertF32toBF16(int scratch1, int scratch2, int destIdx)
{
    return dlp::jit::jitGeneratorError::notSupported;
}

template<>
dlp::jit::jitGeneratorError
jitF32GEMVM1<utils::kernelInstrType::avx2_ymm_16_reg>::convertF32toBF16(
    int scratch1, int scratch2, int destIdx)
{
    vbroadcastss(Xbyak::Ymm(scratch1),
                 ptr[rsp + 16]); // Load 0x00010000
    vpand(Xbyak::Ymm(scratch1), Xbyak::Ymm(destIdx),
          Xbyak::Ymm(scratch1)); // Extract bit 16
    vpsrld(Xbyak::Ymm(scratch1), Xbyak::Ymm(scratch1),
           16); // Shift to position 0 → tlsb

    vbroadcastss(Xbyak::Ymm(scratch2),
                 ptr[rsp + 20]); // Load 0x00007FFF
    vpaddd(Xbyak::Ymm(scratch2), Xbyak::Ymm(destIdx),
           Xbyak::Ymm(scratch2)); // Add rounding to original

    vpaddd(Xbyak::Ymm(scratch2), Xbyak::Ymm(scratch2),
           Xbyak::Ymm(scratch1)); // Add tlsb → rounded

    vpsrld(Xbyak::Ymm(scratch2), Xbyak::Ymm(scratch2),
           16); // Shift right 16 bits

    // Extract upper 128 bits of YMM → XMM
    vextracti128(Xbyak::Xmm(scratch1), Xbyak::Ymm(scratch2), 1);

    // Pack 8×32-bit to 8×16-bit
    vpackusdw(Xbyak::Xmm(scratch2), Xbyak::Xmm(scratch2), Xbyak::Xmm(scratch1));

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::storeYValuesFringe()
{
    int n_iter = N_LEFT / simdWidth;
    int n_left = N_LEFT % simdWidth;
    for (int i = 0; i < n_iter; i++) {
        vmovups(ptr[regTmpYptr + i * simdWidth * sizeof(float)],
                RegType(accumBaseIdx + i));
    }
    if (n_left) {
        vmovups(ptr[regTmpYptr + n_iter * simdWidth * sizeof(float)]
                    | mask_regs[0],
                RegType(accumBaseIdx + n_iter));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitF32GEMVM1<utils::kernelInstrType::avx2_ymm_16_reg>::storeYValuesFringe()
{
    inLocalLabel();
    Xbyak::Label label_store_fringe, label_store_fringe_end;

    int n_iter = N_LEFT / simdWidth;
    int n_left = N_LEFT % simdWidth;

    if (c_downscale < DLP_F32) {
        // Check for is_last_k
        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, is_last_k)]);
        test(regTmp2, regTmp2);
        je(label_store_fringe, T_NEAR);

        mov(regTmpYptr,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, buf_downscale)]);

        // NULL check
        cmp(regTmpYptr, 0);
        je(label_store_fringe, T_NEAR);

        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
        lea(regTmp2, ptr[regTmp2 * sizeof(bfloat16)]);

        add(regTmpYptr, regTmp2);

        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, rs_c_downscale)]);
        lea(regTmp2, ptr[regTmp2 * sizeof(bfloat16)]); // BF16 stride

        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, post_op_c_i)]);

        imul(regKIter, regTmp2);
        add(regTmpYptr, regKIter);

        // Store complete SIMD-width chunks
        for (int i = 0; i < n_iter; i += 1) {

            RETURN_IF_ERROR(
                convertF32toBF16(tmpBaseIdx, tmpBaseIdx + 1, accumBaseIdx + i));
            movdqu(ptr[regTmpYptr], Xbyak::Xmm(tmpBaseIdx + 1));

            // vcvtneps2bf16(Xbyak::Ymm(accumBaseIdx + i),
            //             Xbyak::Zmm(accumBaseIdx + i));
            // vmovdqu16(ptr[regTmpYptr], Xbyak::Ymm(accumBaseIdx + i));
            lea(regTmpYptr, ptr[regTmpYptr + simdWidth * sizeof(bfloat16)]);
        }
        if (n_left) {

            RETURN_IF_ERROR(convertF32toBF16(tmpBaseIdx, tmpBaseIdx + 1,
                                             accumBaseIdx + n_iter));

            // Now Xmm(scratch2) contains 8×BF16 values
            // Store the BF16 result to stack for element-wise access
            movdqu(ptr[rsp + 0],
                   Xbyak::Xmm(tmpBaseIdx + 1)); // 8×16-bit to stack

            // Get n_remainder: n % 8
            mov(regTmp2, n_left);
            and_(regTmp2.cvt32(), 7); // n % 8

            // Calculate destination base address
            // lea(regKIter, ptr[regTmpCptr + bFullReg * halfRegBytes]);

            // Loop: copy n_remainder elements from stack to destination
            xor_(regKIter.cvt32(), regKIter.cvt32()); // elem_idx = 0

            Xbyak::Label loop_start, loop_end;
            L(loop_start);

            // Check if elem_idx < n_remainder
            cmp(regKIter.cvt32(), regTmp2.cvt32());
            jge(loop_end, T_NEAR);

            // Load BF16 value from stack and store to destination
            // Use regBptr as temporary (done with B matrix access)
            mov(regBptr.cvt16(), word[rsp + regKIter * sizeof(int16_t)]);
            mov(word[regTmpYptr + regKIter * sizeof(int16_t)], regBptr.cvt16());

            inc(regKIter.cvt32()); // elem_idx++
            jmp(loop_start, T_NEAR);

            L(loop_end);
            // vcvtneps2bf16(Xbyak::Ymm(accumBaseIdx + n_iter),
            //             Xbyak::Zmm(accumBaseIdx + n_iter));
            // vmovdqu16(ptr[regTmpYptr] | mask_regs[0],
            //         Xbyak::Ymm(accumBaseIdx + n_iter));
        }

        jmp(label_store_fringe_end, T_NEAR);
        L(label_store_fringe);
    }

    for (int i = 0; i < n_iter; i++) {
        vmovups(ptr[regTmpYptr + i * simdWidth * sizeof(float)],
                Xbyak::Ymm(accumBaseIdx + i));
    }
    if (n_left) {
        vmaskmovps(ptr[regTmpYptr + n_iter * simdWidth * sizeof(float)],
                   Xbyak::Ymm(maskBaseIdx), Xbyak::Ymm(accumBaseIdx + n_iter));
    }

    L(label_store_fringe_end);
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::storeYValues(bool nMask)
{

    inLocalLabel();
    Xbyak::Label label_store, label_store_end;

    mov(regTmpYptr, regYptr);

    if (!nMask) {

        if (c_downscale < DLP_F32) {
            // Check for is_last_k
            mov(regTmp2,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                    + offsetof(lpgemm_post_op_attr, is_last_k)]);
            test(regTmp2, regTmp2);
            je(label_store, T_NEAR);

            mov(regTmpYptr,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                    + offsetof(lpgemm_post_op_attr, buf_downscale)]);

            // NULL check
            cmp(regTmpYptr, 0);
            je(label_store, T_NEAR);

            mov(regTmp2,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                    + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
            lea(regTmp2, ptr[regTmp2 * sizeof(bfloat16)]);

            add(regTmpYptr, regTmp2);

            mov(regTmp2,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                    + offsetof(lpgemm_post_op_attr, rs_c_downscale)]);
            lea(regTmp2, ptr[regTmp2 * sizeof(bfloat16)]); // BF16 stride

            mov(regKIter,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                    + offsetof(lpgemm_post_op_attr, post_op_c_i)]);

            imul(regKIter, regTmp2);
            add(regTmpYptr, regKIter);

            // Store complete SIMD-width chunks
            for (int i = 0; i < NR / simdWidth; i += 1) {

                RETURN_IF_ERROR(convertF32toBF16(tmpBaseIdx, tmpBaseIdx + 1,
                                                 accumBaseIdx + i));
                movdqu(ptr[regTmpYptr], Xbyak::Xmm(tmpBaseIdx + 1));

                // vcvtneps2bf16(Xbyak::Ymm(accumBaseIdx + i),
                //             Xbyak::Zmm(accumBaseIdx + i));
                // vmovdqu16(ptr[regTmpYptr], Xbyak::Ymm(accumBaseIdx + i));
                lea(regTmpYptr, ptr[regTmpYptr + simdWidth * sizeof(bfloat16)]);
            }
            jmp(label_store_end, T_NEAR);
            L(label_store);
        }

        for (int i = 0; i < NR / simdWidth; i += 1) {
            vmovups(ptr[regTmpYptr + i * simdWidth * sizeof(float)],
                    RegType(accumBaseIdx + i));
        }
    } else {
        storeYValuesFringe();
    }

    L(label_store_end);
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitF32GEMVM1<KType>::generateKernel(utils::gemvM1GeneratorParams& params)
{
    // Using Xbyak's utility for managing the stack frame
    Xbyak::util::StackFrame frame(this, 1, 13, 48);
    initializeStackFrame(frame);

    // Initializing the parameters
    initializeParameters(params);

    // Allocating valid ranges for register usage
    RETURN_IF_ERROR(allocateRegisters());

    inLocalLabel();

    // Create kernel ops handler once for the entire kernel
    std::unique_ptr<gen::kernelOpsHandler> kernelOpsHandlerPtr;
    if (!params.kernelOps.empty()) {
        kernelOpsHandlerPtr =
            std::make_unique<gen::kernelOpsHandler>(this, params.kType);
    }

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

        if (kernelOpsHandlerPtr) {
            RETURN_IF_ERROR((kernelOpsHandlerPtr->generateKernelOps(
                params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemv_m1, 1,
                params.NR, false, 1, accumBaseIdx, yReg)));

            kernelOpsHandlerPtr->generateKernelOpsAttributes();
        }

        storeYValues(false);

        // Update the pointers for next n iteration(NOTE : B pointer is set
        // inside the kloop, owing to the implementation in static kernels)
        mov(regTmp2, NR);
        add(regIncN, regTmp2);
        lea(regYptr, ptr[regYptr + regTmp2 * sizeof(float)]);

        // Update post_op_c_j for the next n-iteration (similar to GEMM pattern)
        // This ensures each iteration uses the correct offset for post-ops
        if (c_downscale < DLP_F32 || !params.kernelOps.empty()) {
            mov(regTmp1,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                    + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
            add(regTmp1, NR);
            mov(ptr[stackPtr
                    + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                    + offsetof(lpgemm_post_op_attr, post_op_c_j)],
                regTmp1);
        }

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

        if (kernelOpsHandlerPtr) {
            RETURN_IF_ERROR((kernelOpsHandlerPtr->generateKernelOps(
                params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemv_m1, 1,
                params.N_LEFT, true, 1, accumBaseIdx, N_LEFT / simdWidth)));

            // This call will skip embedding tables (already done in main loop)
            kernelOpsHandlerPtr->generateKernelOpsAttributes();

            // For avx2 config, we use YMM registers for masks
            // these mask registers may have been used as scratch registers
            // by the post-ops handler. Thus, we need to reload them.
            if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
                loadMasks();
            }
        }

        storeYValues(true);
    }

    L(label_n_fringe_end);
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

} // namespace amdzen::codegen
template class amdzen::codegen::jitF32GEMVN1<
    amdzen::utils::kernelInstrType::avx2_ymm_16_reg>;
template class amdzen::codegen::jitF32GEMVN1<
    amdzen::utils::kernelInstrType::avx512_ymm_32_reg>;
template class amdzen::codegen::jitF32GEMVN1<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
template class amdzen::codegen::jitF32GEMVM1<
    amdzen::utils::kernelInstrType::avx2_ymm_16_reg>;
template class amdzen::codegen::jitF32GEMVM1<
    amdzen::utils::kernelInstrType::avx512_ymm_32_reg>;
template class amdzen::codegen::jitF32GEMVM1<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
