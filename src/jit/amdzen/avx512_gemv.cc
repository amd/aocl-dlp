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

#include "avx512_gemv.hh"
#include "jit_register/jit_register.hh"

// GEMV JIT backend(AVX512) when n = 1
namespace amdzen::avx512gen {

jitAVX512GEMVN1::jitAVX512GEMVN1(void* buffer, size_t size)
    : Xbyak::CodeGenerator(size, buffer) // Call base class constructor
{
}

void
jitAVX512GEMVN1::initializeStackFrame(Xbyak::util::StackFrame& frame)
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

void
jitAVX512GEMVN1::regInitZmm(int baseIdx, int numRegs)
{
    // Zero out accumulation registers
    vxorps(Xbyak::Zmm(baseIdx), Xbyak::Zmm(baseIdx), Xbyak::Zmm(baseIdx));
    for (int i = 0; i < numRegs; i++) {
        vmovaps(Xbyak::Zmm(baseIdx + i), Xbyak::Zmm(baseIdx));
    }
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::allocateRegisters<float>()
{
    // Check if MR is valid
    if (MR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Allocate registers according to the rules:
    // 1. Accumulation registers : MR registers for partial dot products
    accumReg     = MR;
    accumBaseIdx = numRegs - accumReg; // Start from the end

    yReg     = (MR + 15) / (RegBytes / sizeof(float));
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

    // Temporary registers (tmpReg): Use remaining registers for reduction
    tmpReg     = 4;
    tmpBaseIdx = 0; // To make sure we index YMM greater than 16

    // X registers (xReg): Use remaining registers for vector x
    // We need to only consider accumReg, tmpReg and xReg for total register
    // count.
    xReg     = numRegs - accumReg - tmpReg;
    xBaseIdx = tmpReg;

    // Check if we have enough registers
    if (xReg < 1) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
void
jitAVX512GEMVN1::initializeParameters<float, float, float>(
    const utils::gemvN1GeneratorParams& params)
{
    // TODO : Reimplement base on the the latest runtime params struct
    // Set dimensions from params
    MR               = params.MR; // Number of rows to process
    M_LEFT           = params.M_LEFT;
    yFormat          = params.yFormat;          // Storage format of C matrix
    alphaScalingType = params.alphaScalingType; // Type of alpha scaling
    betaScalingType  = params.betaScalingType;  // Type of beta scaling

    // Load pointers and strides from the stack
    // mov(regAptr, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, a)]);
    // mov(regXptr, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, x)]);
    // mov(regYptr, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, y)]);
    mov(regRsA, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, rsA)]);
    mov(regCsA, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, csA)]);
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, rsC)]);

    // Scale strides by data type size
    lea(regRsA, ptr[regRsA * sizeof(float)]);
    lea(regCsA, ptr[regCsA * sizeof(float)]);
    lea(regRsC, ptr[regRsC * sizeof(float)]);
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::loadAValues<float>(int aRegIdx, int rowIdx, bool isFringe)
{
    // Calculate row offset first
    mov(regTmp2, rowIdx);
    imul(regTmp2, regRsA); // regTmp1 = rowIdx * regRsA

    // Load 16 or lesser elements, based on whether it is a fringe case or not.
    if (isFringe) {
        vmovups(Xbyak::Zmm(aBaseIdx + aRegIdx) | k3, ptr[regTmpAptr + regTmp2]);
    } else {
        vmovups(Xbyak::Zmm(aBaseIdx + aRegIdx), ptr[regTmpAptr + regTmp2]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::loadXValues<float>(bool isFringe)
{
    // Load 16 or lesser elements, based on whether it is a fringe case or not.
    if (isFringe) {
        vmovups(Xbyak::Zmm(xBaseIdx) | k3, ptr[regXptr]);
    } else {
        vmovups(Xbyak::Zmm(xBaseIdx), ptr[regXptr]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::loadYValues<float>(int yIdx)
{
    // Load values from Y
    vmovups(Xbyak::Zmm(yBaseIdx + yIdx), ptr[regTmp2]);

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::computeFMA<float, float, float>(int aRegIdx, int accumRegIdx)
{
    vfmadd231ps(Xbyak::Zmm(accumBaseIdx + accumRegIdx),
                Xbyak::Zmm(aBaseIdx + aRegIdx), Xbyak::Zmm(xBaseIdx));

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::computeLoadFMA<float, float, float>(int rowIdx, bool isFringe)
{
    if (isFringe) {

        switch (rowIdx % 8) {
            case 0:
            case 1:
            case 2:
            case 4:
                // Direct scaling for 0,1,2,4
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx) | k3,
                            Xbyak::Zmm(xBaseIdx),
                            ptr[regTmpAptr + regRsA * (rowIdx % 8)]);
                break;
            case 3:
                // Use pre-calculated 3*rsA from regTmp2
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx) | k3,
                            Xbyak::Zmm(xBaseIdx), ptr[regTmpAptr + regTmp1]);
                break;
            case 5:
                // Use pre-calculated 5*rsA from regTmp1
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx) | k3,
                            Xbyak::Zmm(xBaseIdx), ptr[regTmpAptr + regTmp2]);
                break;
            case 6:
                // Use 2*3*rsA (scale regTmp2 by 2)
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx) | k3,
                            Xbyak::Zmm(xBaseIdx),
                            ptr[regTmpAptr + regTmp1 * 2]);
                break;
            case 7:
                // Use pre-calculated 7*rsA
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx) | k3,
                            Xbyak::Zmm(xBaseIdx), ptr[regTmpAptr + regTmp3]);
                break;
        }
    } else {
        // Same cases without masking
        switch (rowIdx % 8) {
            case 0:
            case 1:
            case 2:
            case 4:
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx),
                            Xbyak::Zmm(xBaseIdx),
                            ptr[regTmpAptr + regRsA * (rowIdx % 8)]);
                break;
            case 3:
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx),
                            Xbyak::Zmm(xBaseIdx), ptr[regTmpAptr + regTmp1]);
                break;
            case 5:
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx),
                            Xbyak::Zmm(xBaseIdx), ptr[regTmpAptr + regTmp2]);
                break;
            case 6:
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx),
                            Xbyak::Zmm(xBaseIdx),
                            ptr[regTmpAptr + regTmp1 * 2]);
                break;
            case 7:
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + rowIdx),
                            Xbyak::Zmm(xBaseIdx), ptr[regTmpAptr + regTmp3]);
                break;
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::reduce4ZMMtoXMM<float>(int startIdx, int tmpIdx, int blockSize)
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
    for (int i = 0; i < blockSize; i++) {
        // Extract upper 256-bits to temp YMM
        vextractf32x8(Xbyak::Ymm(tmpIdx + i), Xbyak::Zmm(startIdx + i), 1);
        // Add to lower 256-bits of input ZMM, storing in original ZMM's YMM
        // part
        vaddps(Xbyak::Ymm(tmpIdx + i), Xbyak::Ymm(tmpIdx + i),
               Xbyak::Ymm(startIdx + i));
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

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::reduceAccumulation<float>(int mSize)
{
    // Process mSize registers in blocks of 16 (ZMM width for f32)

    for (int i = 0; i < mSize; i += 16) {
        // Number of registers to process in this ZMM block
        int blockSize = (mSize - i) < 16 ? (mSize - i) : 16;

        // Process this ZMM block in groups of 4 registers
        for (int j = 0; j < blockSize; j += 4) {
            int subBlockSize = (blockSize - j) < 4 ? (blockSize - j) : 4;

            // Reduce 4 (or fewer) ZMMs to one XMM
            RETURN_IF_ERROR((reduce4ZMMtoXMM<float>(accumBaseIdx + i + j,
                                                    tmpBaseIdx, subBlockSize)));

            // Insert the resulting XMM
            // into the appropriate
            // position in destination
            // ZMM
            vinsertf32x4(Xbyak::Zmm(accumBaseIdx + i / 16),
                         Xbyak::Zmm(accumBaseIdx + i / 16),
                         Xbyak::Xmm(tmpBaseIdx), j / 4);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::scaleAccumulationWithAlpha<float>(int mSize)
{
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, alpha)]);
    vbroadcastss(Xbyak::Zmm(tmpBaseIdx), ptr[regKIter]);
    for (int i = 0; i < (mSize + 15) / 16; i += 1) {
        vmulps(Xbyak::Zmm(accumBaseIdx + i), Xbyak::Zmm(accumBaseIdx + i),
               Xbyak::Zmm(tmpBaseIdx));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::scaleYWithBetaColStored<float>(int  mSize,
                                                bool betaOne,
                                                bool maskType)
{
    if (!betaOne) {
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, beta)]);
        vbroadcastss(Xbyak::Zmm(xBaseIdx), ptr[regKIter]);
    }
    int mLeft = mSize % 16;
    for (int i = 0; i < mSize / 16; i += 1) {
        if (betaOne) {
            vaddps(Xbyak::Zmm(accumBaseIdx + i), Xbyak::Zmm(accumBaseIdx + i),
                   ptr[regTmpYptr]);
        } else {
            vfmadd231ps(Xbyak::Zmm(accumBaseIdx + i), Xbyak::Zmm(xBaseIdx),
                        ptr[regTmpYptr]);
        }
        lea(regTmpYptr, ptr[regTmpYptr + 16 * sizeof(float)]);
    }
    if (mLeft) {
        if (!maskType) {
            kmovw(
                k3,
                ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, mr_mask)]);
        } else {
            kmovw(k3,
                  ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, m_mask)]);
        }
        if (betaOne) {
            vaddps(Xbyak::Zmm(accumBaseIdx + (mSize / 16)) | k3,
                   Xbyak::Zmm(accumBaseIdx + (mSize / 16)), ptr[regTmpYptr]);
        } else {
            vfmadd231ps(Xbyak::Zmm(accumBaseIdx + (mSize / 16)) | k3,
                        Xbyak::Zmm(xBaseIdx), ptr[regTmpYptr]);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::scaleYWithBetaRowStored<float>(int mSize, bool betaOne)
{
    if (!betaOne) {
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, beta)]);
        vbroadcastss(Xbyak::Zmm(xBaseIdx), ptr[regKIter]);
    }
    // Store offsets for Y, using it's row-stride
    lea(regTmp3, ptr[regRsC + 2 * regRsC]); // regTmp3 = rsC + 2*rsC
    for (int i = 0; i < (mSize + 15) / 16; i += 1) {
        int blockSize  = (mSize - i * 16) < 16 ? (mSize - i * 16) : 16;
        int num_blocks = blockSize / 4;
        int rem_block  = blockSize % 4;
        regInitZmm(tmpBaseIdx, tmpReg);
        for (int j = 0; j < num_blocks; j += 1) {
            vbroadcastss(Xbyak::Zmm(tmpBaseIdx), ptr[regTmpYptr]);
            vbroadcastss(Xbyak::Zmm(tmpBaseIdx + 1), ptr[regTmpYptr + regRsC]);
            vbroadcastss(Xbyak::Zmm(tmpBaseIdx + 2),
                         ptr[regTmpYptr + 2 * regRsC]);
            vbroadcastss(Xbyak::Zmm(tmpBaseIdx + 3), ptr[regTmpYptr + regTmp3]);
            vunpcklps(Xbyak::Zmm(tmpBaseIdx), Xbyak::Zmm(tmpBaseIdx),
                      Xbyak::Zmm(tmpBaseIdx + 1));
            vunpcklps(Xbyak::Zmm(tmpBaseIdx + 2), Xbyak::Zmm(tmpBaseIdx + 2),
                      Xbyak::Zmm(tmpBaseIdx + 3));
            vshufps(Xbyak::Zmm(tmpBaseIdx), Xbyak::Zmm(tmpBaseIdx),
                    Xbyak::Zmm(tmpBaseIdx + 2), 0x44);
            vinsertf32x4(Xbyak::Zmm(yBaseIdx + i), Xbyak::Zmm(yBaseIdx + i),
                         Xbyak::Xmm(tmpBaseIdx), j);
            lea(regTmpYptr, ptr[regTmpYptr + regRsC * 4]);
        }
        if (rem_block) {
            switch (rem_block) {
                case 3:
                    vbroadcastss(Xbyak::Zmm(tmpBaseIdx + 2),
                                 ptr[regTmpYptr + regRsC * 2]);
                case 2:
                    vbroadcastss(Xbyak::Zmm(tmpBaseIdx + 1),
                                 ptr[regTmpYptr + regRsC]);
                case 1:
                    vbroadcastss(Xbyak::Zmm(tmpBaseIdx), ptr[regTmpYptr]);
                case 0:
                    break;
            }
            vunpcklps(Xbyak::Zmm(tmpBaseIdx), Xbyak::Zmm(tmpBaseIdx),
                      Xbyak::Zmm(tmpBaseIdx + 1));
            vunpcklps(Xbyak::Zmm(tmpBaseIdx + 2), Xbyak::Zmm(tmpBaseIdx + 2),
                      Xbyak::Zmm(tmpBaseIdx + 3));
            vshufps(Xbyak::Zmm(tmpBaseIdx), Xbyak::Zmm(tmpBaseIdx),
                    Xbyak::Zmm(tmpBaseIdx + 2), 0x44);
            vinsertf32x4(Xbyak::Zmm(yBaseIdx + i), Xbyak::Zmm(yBaseIdx + i),
                         Xbyak::Xmm(tmpBaseIdx), num_blocks);
        }

        if (betaOne) {
            vaddps(Xbyak::Zmm(accumBaseIdx + i), Xbyak::Zmm(accumBaseIdx + i),
                   Xbyak::Zmm(yBaseIdx + i));
        } else {
            vfmadd231ps(Xbyak::Zmm(accumBaseIdx + i), Xbyak::Zmm(xBaseIdx),
                        Xbyak::Zmm(yBaseIdx + i));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::scaleYWithBeta<float>(int mSize, bool maskType)
{
    bool is_beta_one = (betaScalingType == dlp::kernel_frame::scalingType::one);
    if (betaScalingType != dlp::kernel_frame::scalingType::zero) {
        mov(regTmpYptr, regYptr);
        if (yFormat == dlp::kernel_frame::storageFormat::colMajor) {
            RETURN_IF_ERROR(
                (scaleYWithBetaColStored<float>(mSize, is_beta_one, maskType)));
        } else {
            RETURN_IF_ERROR(
                (scaleYWithBetaRowStored<float>(mSize, is_beta_one)));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::storeYValuesColStored<float>(int mSize, bool maskType)
{
    int mLeft = mSize % 16;
    for (int i = 0; i < mSize / 16; i += 1) {
        vmovups(ptr[regTmpYptr], Xbyak::Zmm(accumBaseIdx + i));
        lea(regTmpYptr, ptr[regTmpYptr + 16 * sizeof(float)]);
    }
    if (mLeft) {
        if (!maskType) {
            kmovw(
                k3,
                ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, mr_mask)]);
        } else {
            kmovw(k3,
                  ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, m_mask)]);
        }
        vmovups(ptr[regTmpYptr] | k3, Xbyak::Zmm(accumBaseIdx + (mSize / 16)));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::storeYValuesRowStored<float>(int mSize)
{
    // Process each ZMM register (which contains 16 elements)
    for (int i = 0; i < (mSize + 15) / 16; i++) {
        int elements_in_zmm = (i < mSize / 16) ? 16 : (mSize % 16);
        if (elements_in_zmm == 0)
            break;

        // Extract 4 chunks of 128-bits (4 floats each) from the ZMM
        for (int j = 0; j < elements_in_zmm; j += 4) {
            vextractf32x4(Xbyak::Xmm(tmpBaseIdx + j / 4),
                          Xbyak::Zmm(accumBaseIdx + i), j / 4);
        }

        // Now store each extracted value to its proper row-strided location
        for (int j = 0; j < elements_in_zmm; j++) {
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
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::storeYValues<float>(int mSize, bool maskType)
{
    // Store values from Y
    mov(regTmpYptr, regYptr);
    if (yFormat == dlp::kernel_frame::storageFormat::colMajor) {
        RETURN_IF_ERROR((storeYValuesColStored<float>(mSize, maskType)));
    } else {
        RETURN_IF_ERROR((storeYValuesRowStored<float>(mSize)));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::process8RowBlock<float, float, float>(int mSize, bool isFringe)
{
    // Calculate number of full 8-row blocks
    int full_blocks = mSize / 8;
    int remaining   = mSize % 8;

    // Handle full blocks without conditionals
    for (int block = 0; block < full_blocks; block++) {
        // Process all 8 rows in the block
        for (int i = 0; i < 8; i++) {
            RETURN_IF_ERROR(
                (computeLoadFMA<float, float, float>(block * 8 + i, isFringe)));
        }

        // Update base pointer for next block
        lea(regTmpAptr, ptr[regTmpAptr + regRsA * 8]);
    }

    // Handle remaining rows
    if (remaining > 0) {
        for (int i = 0; i < remaining; i++) {
            RETURN_IF_ERROR((computeLoadFMA<float, float, float>(
                full_blocks * 8 + i, isFringe)));
        }
        if ((remaining == 1) || (remaining == 2) || (remaining == 4)) {
            lea(regTmpAptr, ptr[regTmpAptr + regRsA * remaining]);
        } else if (remaining == 3) {
            lea(regTmpAptr, ptr[regTmpAptr + regTmp1]);
        } else if (remaining == 5) {
            lea(regTmpAptr, ptr[regTmpAptr + regTmp2]);
        } else if (remaining == 6) {
            lea(regTmpAptr, ptr[regTmpAptr + regTmp1 * 2]);
        } else if (remaining == 7) {
            lea(regTmpAptr, ptr[regTmpAptr + regTmp3]);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVN1::generateKernel<
    dlp::kernel_frame::kernelDatatype::f32f32f32of32>(
    const utils::gemvN1GeneratorParams& params)
{
    using aType     = float;
    using xType     = float;
    using yType     = float;
    using accumType = float;

    Xbyak::util::StackFrame frame(this, 1, 13, 0);
    initializeStackFrame(frame);

    initializeParameters<aType, xType, yType>(params);
    RETURN_IF_ERROR((allocateRegisters<accumType>()));

    // Acquire the addresses of A and Y
    mov(regAptr, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, a)]);
    mov(regYptr, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, y)]);

    inLocalLabel();

    // Set the for-loop sequence for m-dimension
    if (params.mloop) {
        mov(regMIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, m_iter)]);
        test(regMIter, regMIter);
        jz(label_m_loop_end, T_NEAR);
        L(label_m_loop_start);
        // }

        // Zero out accumulator registers for this m iteration
        regInitZmm(accumBaseIdx, MR);

        // // Y prefetch, before the k-loop
        // if (betaScalingType != dlp::kernel_frame::scalingType::zero) {
        //     prefetcht0(ptr[regYptr]);
        // }

        // K-loop is not needed if alpha is zero
        if (params.alphaScalingType != dlp::kernel_frame::scalingType::zero) {
            // Pre-calculate useful multiples of rsA
            lea(regTmp1,
                ptr[regRsA + regRsA * 2]); // regTmp1 = rsA + 2*rsA = 3*rsA
            lea(regTmp2,
                ptr[regRsA + regRsA * 4]); // regTmp2 = rsA + 4*rsA = 5*rsA
            lea(regTmp3,
                ptr[regTmp2 + regRsA * 2]); // regTmp3 = 5*rsA + 2*rsA = 7*rsA

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
                RETURN_IF_ERROR((loadXValues<xType>()));

                // Process all rows including fringe
                RETURN_IF_ERROR(
                    (process8RowBlock<aType, xType, accumType>(MR)));

                // Save current A pointer and update pointers for next k
                // iteration
                lea(regTmpYptr, ptr[regTmpYptr + regCsA * 8]);
                lea(regTmpYptr, ptr[regTmpYptr + regCsA * 8]);
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

                kmovw(k3, ptr[stackPtr
                              + offsetof(dlp::kernels::gemvN1Params, k_mask)]);
                RETURN_IF_ERROR((loadXValues<xType>(params.kfringe)));
                RETURN_IF_ERROR(
                    (process8RowBlock<aType, xType, accumType>(MR, true)));
            }
            L(label_m_loop_k_fringe_end);

            // Reduce the accumulation registers to XMMs, and put it in
            // ZMMs
            reduceAccumulation<accumType>(MR);

            // Alpha scaling
            if (params.alphaScalingType
                != dlp::kernel_frame::scalingType::one) {
                scaleAccumulationWithAlpha<accumType>(MR);
            }
        }

        // Working good for element-wise loads/stores for C.
        scaleYWithBeta<float>(MR, false);
        storeYValues<float>(MR, false);

        // if (params.mloop) {
        // Update pointers for next m iteration(for A and y)
        mov(regTmp2, MR);
        imul(regTmp2, regRsA);
        add(regAptr, regTmp2);
        mov(regTmp1, MR);
        imul(regTmp1, regRsC);
        add(regYptr, regTmp1);

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

        regInitZmm(accumBaseIdx, M_LEFT);

        // K-loop is not needed if alpha is zero
        if (params.alphaScalingType != dlp::kernel_frame::scalingType::zero) {
            // Pre-calculate useful multiples of rsA
            lea(regTmp1,
                ptr[regRsA + regRsA * 2]); // regTmp1 = rsA + 2*rsA = 3*rsA
            lea(regTmp2,
                ptr[regRsA + regRsA * 4]); // regTmp2 = rsA + 4*rsA = 5*rsA
            lea(regTmp3,
                ptr[regTmp2 + regRsA * 2]); // regTmp3 = 5*rsA + 2*rsA = 7*rsA

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
                RETURN_IF_ERROR((loadXValues<xType>()));

                // Process all rows including fringe
                RETURN_IF_ERROR(
                    (process8RowBlock<aType, xType, accumType>(M_LEFT)));

                // Update pointers for next k iteration
                lea(regTmpYptr, ptr[regTmpYptr + regCsA * 8]);
                lea(regTmpYptr, ptr[regTmpYptr + regCsA * 8]);
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

                kmovw(k3, ptr[stackPtr
                              + offsetof(dlp::kernels::gemvN1Params, k_mask)]);
                RETURN_IF_ERROR((loadXValues<xType>(true)));

                RETURN_IF_ERROR(
                    (process8RowBlock<aType, xType, accumType>(M_LEFT, true)));
            }
            L(label_m_fringe_k_fringe_end);

            // Reduce the accumulation registers to XMMs, and put it in
            // ZMMs
            reduceAccumulation<accumType>(M_LEFT);
            // Alpha scaling
            if (params.alphaScalingType
                != dlp::kernel_frame::scalingType::one) {
                scaleAccumulationWithAlpha<accumType>(M_LEFT);
            }
        }

        scaleYWithBeta<float>(M_LEFT, true);
        storeYValues<float>(M_LEFT, true);
    }
    // Might need more labels for the fringe cases.
    L(label_m_fringe_end);
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

// GEMV JIT backend(AVX512) when m == 1
jitAVX512GEMVM1::jitAVX512GEMVM1(void* buffer, size_t size)
    : Xbyak::CodeGenerator(size, buffer) // Call base class constructor
{
}

void
jitAVX512GEMVM1::initializeStackFrame(Xbyak::util::StackFrame& frame)
{
    stackPtr    = frame.p[0];
    regTmpBptr  = frame.t[0];
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

void
jitAVX512GEMVM1::regInitZmm(int baseIdx, int numRegs)
{
    for (int i = 0; i < numRegs; i++) {
        vxorps(Xbyak::Zmm(baseIdx + i), Xbyak::Zmm(baseIdx + i),
               Xbyak::Zmm(baseIdx + i));
    }
}

template<>
void
jitAVX512GEMVM1::initializeParameters<float, float, float>(
    const utils::gemvM1GeneratorParams& params)
{
    NR               = params.NR;
    KC               = params.KC;
    yFormat          = params.yFormat;
    alphaScalingType = params.alphaScalingType;
    betaScalingType  = params.betaScalingType;
    mtag_b           = params.mtag_b;

    if ((mtag_b == REORDERED) || (mtag_b == PACK)) {
        mov(regRsB, NR);
        lea(regRsB, ptr[regRsB * sizeof(float)]); // rsB = NR * sizeof(float)
    } else {

        mov(regRsB, ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, rsB)]);
        lea(regRsB, ptr[regRsB * sizeof(float)]);
    }
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVM1::allocateRegisters<float>()
{
    if (NR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Allocate registers according to the following rules:
    // x Registers : K_SUB_ITER(hardset to 4 for now)
    // y Registers : NR/16
    // Accumulation registers : (NR/16) * K_SUB_ITER

    // NOTE : For now, the generator only supports NR being a multiple of 16
    // TODO : To make it support any NR value, similar to jitAVX512GEMVN1
    // generator
    yReg     = NR / 16;
    xReg     = 4; // K_SUB_ITER;
    bReg     = 4;
    accumReg = (NR / 16) * 4;

    // Direct addressing mode on FMA instructions are avoided here, since
    // we could initiate loads eariler with explicit loads.
    // Thus, both x and B loads are done into registers.
    accumBaseIdx = numRegs - accumReg;
    xBaseIdx     = accumBaseIdx - xReg;
    yBaseIdx     = numRegs - yReg;
    bBaseIdx     = xBaseIdx - bReg;

    if (bBaseIdx < 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVM1::compute4x64<float, float, float>(bool nMask)
{
    mov(regTmp2, regTmpBptr);

    vbroadcastss(Xbyak::Zmm(xBaseIdx), ptr[regXptr]);
    vbroadcastss(Xbyak::Zmm(xBaseIdx + 1), ptr[regXptr + sizeof(float)]);
    vbroadcastss(Xbyak::Zmm(xBaseIdx + 2), ptr[regXptr + 2 * sizeof(float)]);
    vbroadcastss(Xbyak::Zmm(xBaseIdx + 3), ptr[regXptr + 3 * sizeof(float)]);
    if (!nMask) {
        for (int i = 0; i < 4; i += 1) {
            vmovups(Xbyak::Zmm(bBaseIdx), ptr[regTmp2]);
            vmovups(Xbyak::Zmm(bBaseIdx + 1), ptr[regTmp2 + regRsB]);
            vmovups(Xbyak::Zmm(bBaseIdx + 2), ptr[regTmp2 + regRsB * 2]);
            vmovups(Xbyak::Zmm(bBaseIdx + 3), ptr[regTmp2 + regTmp1]);

            vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 4 * i), Xbyak::Zmm(xBaseIdx),
                        Xbyak::Zmm(bBaseIdx));
            vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 4 * i + 1),
                        Xbyak::Zmm(xBaseIdx + 1), Xbyak::Zmm(bBaseIdx + 1));
            vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 4 * i + 2),
                        Xbyak::Zmm(xBaseIdx + 2), Xbyak::Zmm(bBaseIdx + 2));
            vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 4 * i + 3),
                        Xbyak::Zmm(xBaseIdx + 3), Xbyak::Zmm(bBaseIdx + 3));

            // Update the pointer for B
            add(regTmp2, 16 * sizeof(float));
        }
    } else {
        vmovups(Xbyak::Zmm(bBaseIdx) | k4, ptr[regTmp2]);
        vmovups(Xbyak::Zmm(bBaseIdx + 1) | k4, ptr[regTmp2 + regRsB]);
        vmovups(Xbyak::Zmm(bBaseIdx + 2) | k4, ptr[regTmp2 + regRsB * 2]);
        vmovups(Xbyak::Zmm(bBaseIdx + 3) | k4, ptr[regTmp2 + regTmp1]);

        // For n0_mask
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 0), Xbyak::Zmm(xBaseIdx),
                    Xbyak::Zmm(bBaseIdx));
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 1), Xbyak::Zmm(xBaseIdx + 1),
                    Xbyak::Zmm(bBaseIdx + 1));
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 2), Xbyak::Zmm(xBaseIdx + 2),
                    Xbyak::Zmm(bBaseIdx + 2));
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 3), Xbyak::Zmm(xBaseIdx + 3),
                    Xbyak::Zmm(bBaseIdx + 3));

        // Update the pointer for B
        add(regTmp2, 16 * sizeof(float));

        // For n1_mask
        vmovups(Xbyak::Zmm(bBaseIdx) | k1, ptr[regTmp2]);
        vmovups(Xbyak::Zmm(bBaseIdx + 1) | k1, ptr[regTmp2 + regRsB]);
        vmovups(Xbyak::Zmm(bBaseIdx + 2) | k1, ptr[regTmp2 + regRsB * 2]);
        vmovups(Xbyak::Zmm(bBaseIdx + 3) | k1, ptr[regTmp2 + regTmp1]);

        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 4), Xbyak::Zmm(xBaseIdx),
                    Xbyak::Zmm(bBaseIdx));
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 5), Xbyak::Zmm(xBaseIdx + 1),
                    Xbyak::Zmm(bBaseIdx + 1));
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 6), Xbyak::Zmm(xBaseIdx + 2),
                    Xbyak::Zmm(bBaseIdx + 2));
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 7), Xbyak::Zmm(xBaseIdx + 3),
                    Xbyak::Zmm(bBaseIdx + 3));

        // Update the pointer for B
        add(regTmp2, 16 * sizeof(float));

        // For n2_mask
        vmovups(Xbyak::Zmm(bBaseIdx) | k2, ptr[regTmp2]);
        vmovups(Xbyak::Zmm(bBaseIdx + 1) | k2, ptr[regTmp2 + regRsB]);
        vmovups(Xbyak::Zmm(bBaseIdx + 2) | k2, ptr[regTmp2 + regRsB * 2]);
        vmovups(Xbyak::Zmm(bBaseIdx + 3) | k2, ptr[regTmp2 + regTmp1]);

        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 8), Xbyak::Zmm(xBaseIdx),
                    Xbyak::Zmm(bBaseIdx));
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 9), Xbyak::Zmm(xBaseIdx + 1),
                    Xbyak::Zmm(bBaseIdx + 1));
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 10), Xbyak::Zmm(xBaseIdx + 2),
                    Xbyak::Zmm(bBaseIdx + 2));
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 11), Xbyak::Zmm(xBaseIdx + 3),
                    Xbyak::Zmm(bBaseIdx + 3));

        // Update the pointer for B
        add(regTmp2, 16 * sizeof(float));

        // For n3_mask
        vmovups(Xbyak::Zmm(bBaseIdx) | k3, ptr[regTmp2]);
        vmovups(Xbyak::Zmm(bBaseIdx + 1) | k3, ptr[regTmp2 + regRsB]);
        vmovups(Xbyak::Zmm(bBaseIdx + 2) | k3, ptr[regTmp2 + regRsB * 2]);
        vmovups(Xbyak::Zmm(bBaseIdx + 3) | k3, ptr[regTmp2 + regTmp1]);

        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 12), Xbyak::Zmm(xBaseIdx),
                    Xbyak::Zmm(bBaseIdx));
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 13), Xbyak::Zmm(xBaseIdx + 1),
                    Xbyak::Zmm(bBaseIdx + 1));
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 14), Xbyak::Zmm(xBaseIdx + 2),
                    Xbyak::Zmm(bBaseIdx + 2));
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 15), Xbyak::Zmm(xBaseIdx + 3),
                    Xbyak::Zmm(bBaseIdx + 3));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVM1::compute1x64<float, float, float>(bool nMask)
{
    mov(regTmp2, regTmpBptr);

    vbroadcastss(Xbyak::Zmm(xBaseIdx), ptr[regXptr]);
    if (!nMask) {
        vmovups(Xbyak::Zmm(bBaseIdx), ptr[regTmp2]);
        vmovups(Xbyak::Zmm(bBaseIdx + 1), ptr[regTmp2 + 16 * sizeof(float)]);
        vmovups(Xbyak::Zmm(bBaseIdx + 2), ptr[regTmp2 + 32 * sizeof(float)]);
        vmovups(Xbyak::Zmm(bBaseIdx + 3), ptr[regTmp2 + 48 * sizeof(float)]);
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 0), Xbyak::Zmm(xBaseIdx),
                    Xbyak::Zmm(bBaseIdx));
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 4), Xbyak::Zmm(xBaseIdx),
                    Xbyak::Zmm(bBaseIdx + 1));
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 8), Xbyak::Zmm(xBaseIdx),
                    Xbyak::Zmm(bBaseIdx + 2));
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 12), Xbyak::Zmm(xBaseIdx),
                    Xbyak::Zmm(bBaseIdx + 3));

    } else {
        vmovups(Xbyak::Zmm(bBaseIdx) | k4, ptr[regTmp2]);
        vmovups(Xbyak::Zmm(bBaseIdx + 1) | k1,
                ptr[regTmp2 + 16 * sizeof(float)]);
        vmovups(Xbyak::Zmm(bBaseIdx + 2) | k2,
                ptr[regTmp2 + 32 * sizeof(float)]);
        vmovups(Xbyak::Zmm(bBaseIdx + 3) | k3,
                ptr[regTmp2 + 48 * sizeof(float)]);
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 0), Xbyak::Zmm(xBaseIdx),
                    Xbyak::Zmm(bBaseIdx));
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 4), Xbyak::Zmm(xBaseIdx),
                    Xbyak::Zmm(bBaseIdx + 1));
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 8), Xbyak::Zmm(xBaseIdx),
                    Xbyak::Zmm(bBaseIdx + 2));
        vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 12), Xbyak::Zmm(xBaseIdx),
                    Xbyak::Zmm(bBaseIdx + 3));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVM1::loopKSubIter<float, float, float>(bool kfringe, bool nfringe)
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

        compute4x64<float, float, float>(nfringe);

        // Update the pointers for next k iteration
        lea(regXptr, ptr[regXptr + 4 * sizeof(float)]);
        lea(regTmpBptr, ptr[regTmpBptr + regRsB * 4]);

        dec(regKSubIter);
        jnz(sub_loop_kc_main_loop_start, T_NEAR);

        L(sub_loop_kc_main_loop_end);
        mov(regKSubIter,
            ptr[stackPtr
                + offsetof(dlp::kernels::gemvM1Params, k_iter_sub_left)]);
        test(regKSubIter, regKSubIter);
        jz(sub_loop_kc_fringe_loop_end, T_NEAR);
        L(sub_loop_kc_fringe_loop_start);

        compute1x64<float, float, float>(nfringe);

        // Update the pointers for next k iteration
        lea(regXptr, ptr[regXptr + sizeof(float)]);
        lea(regTmpBptr, ptr[regTmpBptr + regRsB]);

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

        compute4x64<float, float, float>(nfringe);

        // Update the pointers for next k iteration
        lea(regXptr, ptr[regXptr + 4 * sizeof(float)]);
        lea(regTmpBptr, ptr[regTmpBptr + regRsB * 4]);

        dec(regKSubIter);
        jnz(sub_loop_kf_main_loop_start, T_NEAR);

        L(sub_loop_kf_main_loop_end);
        mov(regKSubIter,
            ptr[stackPtr
                + offsetof(dlp::kernels::gemvM1Params, k_left_sub_left)]);
        test(regKSubIter, regKSubIter);
        jz(sub_loop_kf_fringe_loop_end, T_NEAR);
        L(sub_loop_kf_fringe_loop_start);

        compute1x64<float, float, float>(nfringe);

        // Update the pointers for next k iteration
        lea(regXptr, ptr[regXptr + sizeof(float)]);
        lea(regTmpBptr, ptr[regTmpBptr + regRsB]);

        dec(regKSubIter);
        jnz(sub_loop_kf_fringe_loop_start, T_NEAR);

        L(sub_loop_kf_fringe_loop_end);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVM1::finalAccumulate<float, float, float>()
{
    for (int i = 0; i < 4; i += 1) {
        vaddps(Xbyak::Zmm(accumBaseIdx + 4 * i),
               Xbyak::Zmm(accumBaseIdx + 4 * i),
               Xbyak::Zmm(accumBaseIdx + 4 * i + 1));
        vaddps(Xbyak::Zmm(accumBaseIdx + 4 * i + 2),
               Xbyak::Zmm(accumBaseIdx + 4 * i + 2),
               Xbyak::Zmm(accumBaseIdx + 4 * i + 3));
        vaddps(Xbyak::Zmm(accumBaseIdx + i), Xbyak::Zmm(accumBaseIdx + 4 * i),
               Xbyak::Zmm(accumBaseIdx + 4 * i + 2));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVM1::scaleWithAlpha<float, float, float>()
{
    if (alphaScalingType != dlp::kernel_frame::scalingType::one) {
        mov(regKSubIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, alpha)]);
        vbroadcastss(Xbyak::Zmm(xBaseIdx), ptr[regKSubIter]);
        for (int i = 0; i < 4; i += 1) {
            vmulps(Xbyak::Zmm(accumBaseIdx + i), Xbyak::Zmm(xBaseIdx),
                   Xbyak::Zmm(accumBaseIdx + i));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVM1::scaleYWithBeta<float>(bool nMask)
{
    mov(regTmpYptr, regYptr);

    bool isBetaZero = (betaScalingType == dlp::kernel_frame::scalingType::zero);
    bool isBetaOne  = (betaScalingType == dlp::kernel_frame::scalingType::one);

    if (!isBetaZero) {
        if (!isBetaOne) {
            mov(regKSubIter,
                ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, beta)]);
            vbroadcastss(Xbyak::Zmm(xBaseIdx), ptr[regKSubIter]);
        }
        if (!nMask) {
            if (!isBetaOne) {
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx), Xbyak::Zmm(xBaseIdx),
                            ptr[regTmpYptr]);
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 1), Xbyak::Zmm(xBaseIdx),
                            ptr[regTmpYptr + 16 * sizeof(float)]);
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 2), Xbyak::Zmm(xBaseIdx),
                            ptr[regTmpYptr + 32 * sizeof(float)]);
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 3), Xbyak::Zmm(xBaseIdx),
                            ptr[regTmpYptr + 48 * sizeof(float)]);
            } else {
                vaddps(Xbyak::Zmm(accumBaseIdx), Xbyak::Zmm(accumBaseIdx),
                       ptr[regTmpYptr]);
                vaddps(Xbyak::Zmm(accumBaseIdx + 1),
                       Xbyak::Zmm(accumBaseIdx + 1),
                       ptr[regTmpYptr + 16 * sizeof(float)]);
                vaddps(Xbyak::Zmm(accumBaseIdx + 2),
                       Xbyak::Zmm(accumBaseIdx + 2),
                       ptr[regTmpYptr + 32 * sizeof(float)]);
                vaddps(Xbyak::Zmm(accumBaseIdx + 3),
                       Xbyak::Zmm(accumBaseIdx + 3),
                       ptr[regTmpYptr + 48 * sizeof(float)]);
            }

        } else {
            if (!isBetaOne) {
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx) | k4, Xbyak::Zmm(xBaseIdx),
                            ptr[regTmpYptr]);
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 1) | k1,
                            Xbyak::Zmm(xBaseIdx),
                            ptr[regTmpYptr + 16 * sizeof(float)]);
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 2) | k2,
                            Xbyak::Zmm(xBaseIdx),
                            ptr[regTmpYptr + 32 * sizeof(float)]);
                vfmadd231ps(Xbyak::Zmm(accumBaseIdx + 3) | k3,
                            Xbyak::Zmm(xBaseIdx),
                            ptr[regTmpYptr + 48 * sizeof(float)]);
            } else {
                vaddps(Xbyak::Zmm(accumBaseIdx) | k4, Xbyak::Zmm(accumBaseIdx),
                       ptr[regTmpYptr]);
                vaddps(Xbyak::Zmm(accumBaseIdx + 1) | k1,
                       Xbyak::Zmm(accumBaseIdx + 1),
                       ptr[regTmpYptr + 16 * sizeof(float)]);
                vaddps(Xbyak::Zmm(accumBaseIdx + 2) | k2,
                       Xbyak::Zmm(accumBaseIdx + 2),
                       ptr[regTmpYptr + 32 * sizeof(float)]);
                vaddps(Xbyak::Zmm(accumBaseIdx + 3) | k3,
                       Xbyak::Zmm(accumBaseIdx + 3),
                       ptr[regTmpYptr + 48 * sizeof(float)]);
            }
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVM1::storeYValues<float>(bool nMask)
{
    mov(regTmpYptr, regYptr);

    if (!nMask) {
        vmovups(ptr[regTmpYptr], Xbyak::Zmm(accumBaseIdx));
        vmovups(ptr[regTmpYptr + 16 * sizeof(float)],
                Xbyak::Zmm(accumBaseIdx + 1));
        vmovups(ptr[regTmpYptr + 32 * sizeof(float)],
                Xbyak::Zmm(accumBaseIdx + 2));
        vmovups(ptr[regTmpYptr + 48 * sizeof(float)],
                Xbyak::Zmm(accumBaseIdx + 3));
    } else {
        vmovups(ptr[regTmpYptr] | k4, Xbyak::Zmm(accumBaseIdx));
        vmovups(ptr[regTmpYptr + 16 * sizeof(float)] | k1,
                Xbyak::Zmm(accumBaseIdx + 1));
        vmovups(ptr[regTmpYptr + 32 * sizeof(float)] | k2,
                Xbyak::Zmm(accumBaseIdx + 2));
        vmovups(ptr[regTmpYptr + 48 * sizeof(float)] | k3,
                Xbyak::Zmm(accumBaseIdx + 3));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitAVX512GEMVM1::generateKernel<
    dlp::kernel_frame::kernelDatatype::f32f32f32of32>(
    const utils::gemvM1GeneratorParams& params)
{
    using bType     = float;
    using xType     = float;
    using yType     = float;
    using accumType = float;

    Xbyak::util::StackFrame frame(this, 1, 13, 0);
    initializeStackFrame(frame);

    initializeParameters<bType, xType, yType>(params);
    RETURN_IF_ERROR(allocateRegisters<accumType>());

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
            prefetcht0(ptr[regYptr]);
            prefetcht0(ptr[regYptr + 16 * sizeof(float)]);
            prefetcht0(ptr[regYptr + 32 * sizeof(float)]);
            prefetcht0(ptr[regYptr + 48 * sizeof(float)]);
        }

        // Zero out accumulator registers for this n iteration
        regInitZmm(accumBaseIdx, accumReg);
        xor_(regIncK,
             regIncK); // regIncK is used to increment
                       // the pointer for K dimension(zeroed before the kloop)

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
                mov(regTmpBptr,
                    ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, b)]);

                // The base pointer to B is should be updated based on whether
                // the matrix is packed/reordered or not This logic is ported
                // from the static kernels, which requires us to update it
                // inside the k-loop.
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

                    lea(regTmpBptr, ptr[regTmpBptr + regTmpYptr]);
                    lea(regTmpBptr, ptr[regTmpBptr + regTmp2]);
                } else {
                    mov(regPsB, 1);
                    lea(regPsB, ptr[regPsB * sizeof(float)]);
                    mov(regTmp2, regRsB);
                    imul(regTmp2, regIncK);

                    add(regTmpBptr, regTmp2);
                }

                // Set the base pointer for the iteration
                mov(regTmp2, regIncN);
                imul(regTmp2, regPsB);

                add(regTmpBptr, regTmp2);

                prefetcht0(ptr[regTmpBptr + 4 * regRsB]);

                // This is a sub-loop over the k-dimension
                // This is intended to utlize more registers(it is not just
                // a code-unroll) The block size is KC(since it is the main
                // loop). The booleans indicate which runtime parameter we
                // have to use for iteration. The pointer to x is automatically
                // updated inside.
                loopKSubIter<bType, xType, yType>(false, false);

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
                mov(regTmpBptr,
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

                    lea(regTmpBptr, ptr[regTmpBptr + regTmpYptr]);
                    lea(regTmpBptr, ptr[regTmpBptr + regTmp2]);
                } else {
                    mov(regPsB, 1);
                    lea(regPsB, ptr[regPsB * sizeof(float)]);

                    mov(regTmp2, regRsB);
                    imul(regTmp2, regIncK);

                    add(regTmpBptr, regTmp2);
                }

                mov(regTmp2, regIncN);
                imul(regTmp2, regPsB);

                add(regTmpBptr, regTmp2);

                prefetcht0(ptr[regTmpBptr + 4 * regRsB]);

                // This is a sub-loop over the k-dimension
                // This is intended to utlize more registers(it is not just
                // a code-unroll) The block size is K_LEFT(since it is the
                // fringe loop) The boolean indicates which runtime
                // parameter we have to use for iteration. The pointers are
                // updated as part of the loopKSubIter function.
                loopKSubIter<bType, xType, yType>(true, false);
            }

            L(label_n_loop_k_fringe_end);

            // Final accumulattion of the result
            finalAccumulate<bType, xType, yType>();

            // Scale with alpha
            scaleWithAlpha<bType, xType, yType>();
        }

        // Scale the result by beta, and store it accordingly
        scaleYWithBeta<yType>(false);
        storeYValues<yType>(false);

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

        // If mask is used, we need to load the masks from runtime parameters
        kmovw(k4,
              ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, n0_mask)]);
        kmovw(k1,
              ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, n1_mask)]);
        kmovw(k2,
              ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, n2_mask)]);
        kmovw(k3,
              ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, n3_mask)]);

        mov(regNIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, n_left)]);
        test(regNIter, regNIter);
        jz(label_n_fringe_end, T_NEAR);
        L(label_n_fringe_start);

        // Zero out accumulator registers for this n iteration
        regInitZmm(accumBaseIdx, accumReg);
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
                mov(regTmpBptr,
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

                    lea(regTmpBptr, ptr[regTmpBptr + regTmpYptr]);
                    lea(regTmpBptr, ptr[regTmpBptr + regTmp2]);
                } else {
                    mov(regPsB, 1);
                    lea(regPsB, ptr[regPsB * sizeof(float)]);

                    mov(regTmp2, regRsB);
                    imul(regTmp2, regIncK);

                    add(regTmpBptr, regTmp2);
                }

                mov(regTmp2, regIncN);
                imul(regTmp2, regPsB);

                add(regTmpBptr, regTmp2);

                prefetcht0(ptr[regTmpBptr + 4 * regRsB]);

                loopKSubIter<bType, xType, yType>(false, true);

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
                mov(regTmpBptr,
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

                    lea(regTmpBptr, ptr[regTmpBptr + regTmpYptr]);
                    lea(regTmpBptr, ptr[regTmpBptr + regTmp2]);
                } else {
                    mov(regPsB, 1);
                    lea(regPsB, ptr[regPsB * sizeof(float)]);

                    mov(regTmp2, regRsB);
                    imul(regTmp2, regIncK);

                    add(regTmpBptr, regTmp2);
                }

                mov(regTmp2, regIncN);
                imul(regTmp2, regPsB);

                add(regTmpBptr, regTmp2);

                prefetcht0(ptr[regTmpBptr + 4 * regRsB]);

                loopKSubIter<bType, xType, yType>(true, true);
            }

            L(label_n_fringe_k_fringe_end);

            // Final accumulattion of the result
            finalAccumulate<bType, xType, yType>();

            // Scale with alpha
            scaleWithAlpha<bType, xType, yType>();
        }

        // Scale the result by beta, and store it accordingly
        scaleYWithBeta<yType>(true);
        storeYValues<yType>(true);
    }

    L(label_n_fringe_end);
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}
} // namespace amdzen::avx512gen
