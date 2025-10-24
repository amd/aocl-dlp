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

#include "bf16_gemv.hh"
#include "jit_register/jit_register.hh"

namespace amdzen::codegen {

template<utils::kernelInstrType KType>
jitBF16GEMVN1<KType>::jitBF16GEMVN1(void* buffer, size_t size)
    : Xbyak::CodeGenerator(size, buffer) // Call base class constructor
{
}

template<utils::kernelInstrType KType>
void
jitBF16GEMVN1<KType>::initializeStackFrame(Xbyak::util::StackFrame& frame)
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
jitBF16GEMVN1<KType>::initializeParameters(
    const utils::gemvN1GeneratorParams& params)
{
    // Set dimensions from params
    MR               = params.MR; // Number of rows to process
    M_LEFT           = params.M_LEFT;
    c_downscale      = params.c_downscale;
    yFormat          = params.yFormat;          // Storage format of C matrix
    alphaScalingType = params.alphaScalingType; // Type of alpha scaling
    betaScalingType  = params.betaScalingType;  // Type of beta scaling

    RegBytes = Traits::regBytes;
    numRegs  = Traits::numRegs;

    // For BF16 GEMV, output is F32, so calculate simdWidthF32 based on output
    // data type
    simdWidthF32 = RegBytes / sizeof(float); // For F32 output (16 for AVX512)
    simdWidthBF16 =
        RegBytes / sizeof(bfloat16); // For BF16 input (32 for AVX512)

    // Load pointers and strides from the stack
    // gemvn1params variable created is loaded to stack via execute kernel at
    // runtime like any other function parameter loaded on its stack

    if (c_downscale < DLP_F32) {
        // Broadcast the left shift offset onto a ZMM register
        // Store to allocated stack space, then broadcast from memory
        // Using rsp(instead of stackPtr in order to use the local stack space)
        mov(dword[rsp + 0],
            0x10); // Store value 16 to local stack (safely allocated)
    }

    mov(regRsA, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, rsA)]);
    mov(regCsA, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, csA)]);
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, rsC)]);

    // Scale strides by data type size
    lea(regRsA, ptr[regRsA * sizeof(bfloat16)]);
    lea(regCsA, ptr[regCsA * sizeof(bfloat16)]);
    lea(regRsC, ptr[regRsC * sizeof(float)]);
}

template<amdzen::utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitBF16GEMVN1<KType>::allocateRegisters()
{
    // Check if MR is valid
    if (MR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Allocate registers according to the rules:
    // maskReg = 0; // Set this only when AVX512 codepath is disabled.

    // 1. Accumulation registers : MR registers for partial dot products
    // bf16 gemv mr = 16, 16 rows of A(containing 32 bf16 elements each) will be
    // multiplied with 1 vector x(32 bf16 elements) to produce 16 outputs in y
    // which would need 16 zmm registers for accumulation
    accumReg     = MR;
    accumBaseIdx = numRegs - accumReg; // Start from the end

    // Calculate y registers needed for F32 output (16/16 = 1 register for
    // AVX512)
    yReg     = MR / simdWidthF32; // Ceiling division for safety
    yBaseIdx = numRegs - yReg;    // Place before accumulation registers

    // NOTE : Before loading from y, we would be using MR registers from the end
    //        for accumulating alpha*A*B. This would then be reduced to MR/16
    //        registers, starting from accumBaseIdx. We would still have
    //        15*MR/16 registers left, which we would use for storing the
    //        result, indexed from yBaseIdx(which would be the last MR/16
    //        registers).

    // Ex : If MR is 16 for AVX512
    //      accumReg = 16
    //      accumBaseIdx = 32 - 16 = 16
    //      yReg = 16 / (64 / 4) = 1
    //      yBaseIdx = 31
    //      tmpReg = 4
    //      tmpBaseIdx = 0
    //      xReg = 31 - 16 - 4 = 11
    //      xBaseIdx = 11

    // Temporary registers (tmpReg): Use remaining registers for reduction
    tmpReg     = 4;
    tmpBaseIdx = 0; // To make sure we index YMM greater than 16

    // X registers (xReg): Use remaining registers for vector x
    // We need to only consider accumReg, tmpReg and xReg for total register
    // count.
    xReg     = 1;
    xBaseIdx = tmpReg;

    maskBaseIdx = xBaseIdx;

    // Check if we have enough registers
    if (maskBaseIdx >= accumBaseIdx) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitBF16GEMVN1<KType>::loadMasks()
{
    // Ensuring mapping only from mask_regs[0] to k7(to avoid k0 usage
    // internally)
    for (int i = 0; i < NUM_USABLE_MASKS; i++) {
        mask_regs[i] = Xbyak::Opmask(MASK_START_IDX + i);
    }

    // Loading the 32 bit k mask(bf16), 16 bit m mask(f32)
    kmovd(mask_regs[0],
          ptr[stackPtr
              + offsetof(dlp::kernels::gemvN1Params, kmask_bf16_avx512)]);
    kmovw(mask_regs[1],
          ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, mmask_avx512)]);

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitBF16GEMVN1<KType>::regInit(int baseIdx, int numRegs)
{
    // Zero out vector registers
    vxorps(RegType(baseIdx), RegType(baseIdx), RegType(baseIdx));
    for (int i = 1; i < numRegs; i++) {
        vmovaps(RegType(baseIdx + i), RegType(baseIdx));
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitBF16GEMVN1<KType>::loadXValues(bool isFringe)
{
    if (isFringe) {
        vmovdqu16(RegType(xBaseIdx) | mask_regs[0], ptr[regXptr]);
    } else {
        vmovdqu16(RegType(xBaseIdx), ptr[regXptr]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitBF16GEMVN1<KType>::loadAValues(int aRegIdx, bool isFringe)
{
    if (isFringe) {
        vmovdqu16(RegType(tmpBaseIdx + aRegIdx) | mask_regs[0],
                  ptr[regTmpAptr + regTmp1]);
    } else {
        vmovdqu16(RegType(tmpBaseIdx + aRegIdx), ptr[regTmpAptr + regTmp1]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitBF16GEMVN1<KType>::computeDP(int aRegIdx, int accumRegIdx)
{
    vdpbf16ps(RegType(accumBaseIdx + accumRegIdx), RegType(xBaseIdx),
              RegType(tmpBaseIdx + aRegIdx));

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitBF16GEMVN1<KType>::processMRBlock(int mSize, bool isFringe)
{
    // Perform the compute over the MR rows
    int mLeft = mSize % 4;
    xor_(regTmp1, regTmp1);
    regInit(tmpBaseIdx, tmpReg);

    // compute by unrolling loadA and DP in blocks of 4 rows at a time
    for (int i = 0; i < mSize / 4; i++) {
        for (int j = 0; j < 4; j++) {
            RETURN_IF_ERROR((loadAValues(j, isFringe)));
            RETURN_IF_ERROR((computeDP(j, i * 4 + j)));

            add(regTmp1, regRsA);
        }
    }

    // compute by unrolling loadA and DP with left rows
    // this will used in M fringe case where mSize < MR
    for (int j = 0; j < mLeft; j++) {
        RETURN_IF_ERROR((loadAValues(j, isFringe)));
        RETURN_IF_ERROR((computeDP(j, (mSize / 4) * 4 + j)));
        add(regTmp1, regRsA);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitBF16GEMVN1<KType>::reduceToXmm(int startIdx, int tmpIdx, int blockSize)
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
jitBF16GEMVN1<KType>::reduceAccumulation(int mSize)
{
    // Process mSize registers in blocks of the simdWidthF32
    for (int i = 0; i < mSize; i += simdWidthF32) {
        // Number of registers to process in this ZMM block
        int blockSize = (mSize - i) < simdWidthF32 ? (mSize - i) : simdWidthF32;

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
            vinsertf32x4(RegType(accumBaseIdx + i / simdWidthF32),
                         RegType(accumBaseIdx + i / simdWidthF32),
                         Xbyak::Xmm(tmpBaseIdx), j / 4);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitBF16GEMVN1<KType>::scaleAccumulationWithAlpha(int mSize)
{
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, alpha)]);
    vbroadcastss(RegType(tmpBaseIdx), ptr[regKIter]);
    for (int i = 0; i < (mSize + simdWidthF32 - 1) / simdWidthF32; i += 1) {
        vmulps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
               RegType(tmpBaseIdx));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitBF16GEMVN1<KType>::scaleYWithBetaRowStored(int mSize, bool betaOne)
{
    if (!betaOne) {
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, beta)]);
        vbroadcastss(RegType(xBaseIdx), ptr[regKIter]);
    }

    inLocalLabel();
    Xbyak::Label label_betaop_row, label_betaop_row_end;

    // Check for BF16 downscaling path
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

        for (int i = 0; i < (mSize + simdWidthF32 - 1) / simdWidthF32; i += 1) {
            int blockSize  = ((mSize - i * simdWidthF32) < simdWidthF32)
                                 ? (mSize - i * simdWidthF32)
                                 : simdWidthF32;
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

                vinsertf32x4(RegType(yBaseIdx + i), RegType(yBaseIdx + i),
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

                vinsertf32x4(RegType(yBaseIdx + i), RegType(yBaseIdx + i),
                             Xbyak::Xmm(tmpBaseIdx), num_blocks);
            }

            if (betaOne) {
                vaddps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(yBaseIdx + i));
            } else {
                vmulps(RegType(tmpBaseIdx), RegType(xBaseIdx),
                       RegType(yBaseIdx + i));
                vaddps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(tmpBaseIdx));
            }
        }

        jmp(label_betaop_row_end, T_NEAR);
        L(label_betaop_row);
    }

    // F32 path (original code)
    // Store offsets for Y, using it's row-stride
    lea(regTmp3, ptr[regRsC + 2 * regRsC]); // regTmp3 = rsC + 2*rsC
    for (int i = 0; i < (mSize + simdWidthF32 - 1) / simdWidthF32; i += 1) {
        int blockSize  = ((mSize - i * simdWidthF32) < simdWidthF32)
                             ? (mSize - i * simdWidthF32)
                             : simdWidthF32;
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
            // Todo: change it to fma later along with reference using fma,
            // or else gives accuracy diff
            vmulps(RegType(tmpBaseIdx), RegType(xBaseIdx),
                   RegType(yBaseIdx + i));
            vaddps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                   RegType(tmpBaseIdx));
        }
    }

    L(label_betaop_row_end);
    outLocalLabel();
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitBF16GEMVN1<KType>::scaleYWithBetaColStored(int mSize, bool betaOne)
{
    inLocalLabel();
    Xbyak::Label label_betaop_col, label_betaop_col_end;
    if (!betaOne) {
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, beta)]);
        vbroadcastss(RegType(xBaseIdx), ptr[regKIter]);
    }
    int mLeft = mSize % simdWidthF32;

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

        vpbroadcastd(Xbyak::Zmm(tmpBaseIdx),
                     ptr[rsp + 0]); // Broadcast from memory

        // Store complete SIMD-width chunks
        for (int i = 0; i < mSize / simdWidthF32; i += 1) {
            vmovdqu16(Xbyak::Ymm(tmpBaseIdx + 1), ptr[regTmpYptr]);
            vpmovsxwd(Xbyak::Zmm(tmpBaseIdx + 1), Xbyak::Ymm(tmpBaseIdx + 1));
            vpsllvd(Xbyak::Zmm(tmpBaseIdx + 1), Xbyak::Zmm(tmpBaseIdx + 1),
                    Xbyak::Zmm(tmpBaseIdx));
            vmulps(Xbyak::Zmm(tmpBaseIdx + 2), Xbyak::Zmm(xBaseIdx),
                   Xbyak::Zmm(tmpBaseIdx + 1));
            vaddps(Xbyak::Zmm(accumBaseIdx + i), Xbyak::Zmm(accumBaseIdx + i),
                   Xbyak::Zmm(tmpBaseIdx + 2));
            lea(regTmpYptr, ptr[regTmpYptr + simdWidthF32 * sizeof(bfloat16)]);
        }
        if (mLeft) {
            vmovdqu16(Xbyak::Ymm(tmpBaseIdx + 1) | mask_regs[1],
                      ptr[regTmpYptr]);
            vpmovsxwd(Xbyak::Zmm(tmpBaseIdx + 1), Xbyak::Ymm(tmpBaseIdx + 1));
            vpsllvd(Xbyak::Zmm(tmpBaseIdx + 1), Xbyak::Zmm(tmpBaseIdx + 1),
                    Xbyak::Zmm(tmpBaseIdx));
            vmulps(Xbyak::Zmm(tmpBaseIdx + 2), Xbyak::Zmm(xBaseIdx),
                   Xbyak::Zmm(tmpBaseIdx + 1));
            vaddps(Xbyak::Zmm(accumBaseIdx + (mSize / simdWidthF32)),
                   Xbyak::Zmm(accumBaseIdx + (mSize / simdWidthF32)),
                   Xbyak::Zmm(tmpBaseIdx + 2));
        }

        jmp(label_betaop_col_end, T_NEAR);
        L(label_betaop_col);
    }

    for (int i = 0; i < mSize / simdWidthF32; i += 1) {
        if (betaOne) {
            vaddps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                   ptr[regTmpYptr]);
        } else {
            // Todo: change it to fma later along with reference using fma,
            // or else gives accuracy diff
            vmulps(RegType(tmpBaseIdx), RegType(xBaseIdx), ptr[regTmpYptr]);
            vaddps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                   RegType(tmpBaseIdx));
        }
        lea(regTmpYptr, ptr[regTmpYptr + simdWidthF32 * sizeof(float)]);
    }
    if (mLeft) {
        if (betaOne) {
            vaddps(RegType(accumBaseIdx + (mSize / simdWidthF32))
                       | mask_regs[1],
                   RegType(accumBaseIdx + (mSize / simdWidthF32)),
                   ptr[regTmpYptr]);
        } else {
            // Todo: change it to fma later along with reference using fma,
            // or else gives accuracy diff
            vmulps(RegType(tmpBaseIdx) | mask_regs[1], RegType(xBaseIdx),
                   ptr[regTmpYptr]);
            vaddps(RegType(accumBaseIdx + (mSize / simdWidthF32)),
                   RegType(accumBaseIdx + (mSize / simdWidthF32)),
                   RegType(tmpBaseIdx));
        }
    }

    L(label_betaop_col_end);
    outLocalLabel();
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitBF16GEMVN1<KType>::scaleYWithBeta(int mSize)
{
    bool is_beta_one = (betaScalingType == dlp::kernel_frame::scalingType::one);
    if (betaScalingType != dlp::kernel_frame::scalingType::zero) {
        mov(regTmpYptr, regYptr);
        // yFormat is set to colMajor as part of runtime params when rsC = 1
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
jitBF16GEMVN1<KType>::storeYValuesColStored(int mSize)
{
    int mLeft = mSize % simdWidthF32;
    inLocalLabel();
    Xbyak::Label label_storeop_col, label_storeop_col_end;

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
        for (int i = 0; i < mSize / simdWidthF32; i += 1) {
            vcvtneps2bf16(Xbyak::Ymm(accumBaseIdx + i),
                          Xbyak::Zmm(accumBaseIdx + i));
            vmovdqu16(ptr[regTmpYptr], Xbyak::Ymm(accumBaseIdx + i));
            lea(regTmpYptr, ptr[regTmpYptr + simdWidthF32 * sizeof(bfloat16)]);
        }
        if (mLeft) {
            vcvtneps2bf16(Xbyak::Ymm(accumBaseIdx + (mSize / simdWidthF32)),
                          Xbyak::Zmm(accumBaseIdx + (mSize / simdWidthF32)));
            vmovdqu16(ptr[regTmpYptr] | mask_regs[1],
                      Xbyak::Ymm(accumBaseIdx + (mSize / simdWidthF32)));
        }

        jmp(label_storeop_col_end, T_NEAR);
        L(label_storeop_col);
    }

    // Store complete SIMD-width chunks
    for (int i = 0; i < mSize / simdWidthF32; i += 1) {
        vmovups(ptr[regTmpYptr], RegType(accumBaseIdx + i));
        lea(regTmpYptr, ptr[regTmpYptr + simdWidthF32 * sizeof(float)]);
    }
    if (mLeft) {
        vmovups(ptr[regTmpYptr] | mask_regs[1],
                RegType(accumBaseIdx + (mSize / simdWidthF32)));
    }

    L(label_storeop_col_end);
    outLocalLabel();
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitBF16GEMVN1<KType>::storeYValuesRowStored(int mSize)
{
    // Store all reduced results - after reduction, results are in
    // RegType(accumBaseIdx + 0) For MR=16, we need to store 16 float values
    // from 1 ZMM register vmovups(ptr[regTmpYptr], RegType(accumBaseIdx + 0));

    inLocalLabel();
    for (int i = 0; i < (mSize + simdWidthF32 - 1) / simdWidthF32; i++) {
        Xbyak::Label label_storeop_row, label_storeop_row_end;
        int          elements_in_reg =
            (i < mSize / simdWidthF32) ? simdWidthF32 : (mSize % simdWidthF32);
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
                vcvtneps2bf16(Xbyak::Ymm(tmpBaseIdx + j),
                              Xbyak::Zmm(tmpBaseIdx + j));
            }

            // Now store each extracted value to its proper row-strided location
            for (int j = 0; j < elements_in_reg; j++) {
                int tmp_reg    = j / 4; // Which temp register has our value
                int pos_in_reg = j % 4; // Position within that temp register

                vpextrw(ptr[regTmpYptr], Xbyak::Xmm(tmpBaseIdx + tmp_reg),
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
jitBF16GEMVN1<KType>::storeYValues(int mSize)
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
jitBF16GEMVN1<KType>::generateKernel(utils::gemvN1GeneratorParams& params)
{
    Xbyak::util::StackFrame frame(this, 1, 13, 16);
    initializeStackFrame(frame);
    // Initializes generator params
    initializeParameters(params);
    // initialize register allocation params based on ISA
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
                add(regTmpYptr, RegBytes);
                mov(regTmpAptr, regTmpYptr);
                add(regXptr, RegBytes); // Since X will be unit-strided (64
                                        // bytes for ZMM)

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

        if (c_downscale < DLP_F32) {
            mov(regKIter,
                ptr[stackPtr
                    + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                    + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
            mov(regTmp2, MR);
            add(regKIter, regTmp2);
            mov(ptr[stackPtr
                    + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                    + offsetof(lpgemm_post_op_attr, post_op_c_i)],
                regKIter);
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

                // Save current A pointer and update pointers for next k
                // iteration
                add(regTmpYptr, RegBytes);
                mov(regTmpAptr, regTmpYptr);
                add(regXptr, RegBytes); // Since X will be unit-strided (64
                                        // bytes for ZMM)

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
                params.M_LEFT, 1, true, 1, accumBaseIdx,
                M_LEFT / simdWidthF32)));

            // This call will skip embedding tables (already done in main loop)
            kernelOpsHandlerPtr->generateKernelOpsAttributes();
        }

        storeYValues(M_LEFT);
    }
    L(label_m_fringe_end);
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

// Explicit template instantiations
template class jitBF16GEMVN1<utils::kernelInstrType::avx512_zmm_32_reg>;

} // namespace amdzen::codegen
