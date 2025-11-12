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

#include "aocl_dlp_config.h"

#include "jit_register/jit_register.hh"
#include "s8_gemv_generator.hh"

namespace amdzen::gen {

using namespace Xbyak;

// Begin S8 GEMV N=1 JIT
template<utils::kernelInstrType KType>
jitGEMVS8N1<KType>::jitGEMVS8N1(void* buffer, size_t bufferSize)
    : Xbyak::CodeGenerator(bufferSize, buffer)
{
}

// Initialize the Stack Frame and assign registers
template<utils::kernelInstrType KType>
void
jitGEMVS8N1<KType>::initializeStackFrame(Xbyak::util::StackFrame& frame)
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
dlp::jit::jitGeneratorError
jitGEMVS8N1<KType>::allocateRegisters()
{
    if (MR <= 0) { // invalid MR
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    accumReg     = MR;
    accumBaseIdx = numRegs - accumReg;

    yReg     = MR / vnniWidth;
    yBaseIdx = numRegs - yReg;

    tmpReg     = 4;
    tmpBaseIdx = 0;

    xReg     = 1;
    xBaseIdx = tmpReg;

    vec128Reg = 1;
    vec128Idx = xBaseIdx + xReg;

    if (vec128Idx >= accumBaseIdx) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMVS8N1<KType>::initializeParameters(utils::gemvN1GeneratorParams& params)
{
    // Set dimensions from params
    MR               = params.MR; // Number of rows to process
    M_LEFT           = params.M_LEFT;
    yFormat          = params.yFormat;          // Storage format of C matrix
    alphaScalingType = params.alphaScalingType; // Type of alpha scaling
    betaScalingType  = params.betaScalingType;  // Type of beta scaling
    c_downscale      = params.c_downscale;      // Type of downscale

    RegBytes = Traits::regBytes;
    numRegs  = Traits::numRegs;

    vnniWidth = RegBytes / sizeof(int32_t); // Since accumulation is being done
                                            // in S32.

    // Load strides from the stack
    mov(regRsA, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, rsA)]);
    mov(regCsA, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, csA)]);
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, rsC)]);

    // Scale strides by data type size
    // lea(regRsA, ptr[regRsA * sizeof(int8_t)]);
    // lea(regCsA, ptr[regCsA * sizeof(int8_t)]);
    lea(regRsC, ptr[regRsC * sizeof(int32_t)]);

    // Load post_op_c_i for downscale buffer addressing (if using downscale)
    if (c_downscale != DLP_S32) {
        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
    }
}

template<utils::kernelInstrType KType>
void
jitGEMVS8N1<KType>::regInit(int baseIdx, int numRegs)
{
    // Zero out accumulation registers
    vxorps(RegType(baseIdx), RegType(baseIdx), RegType(baseIdx));
    for (int i = 1; i < numRegs; ++i) {
        vmovaps(RegType(baseIdx + i), RegType(baseIdx));
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8N1<KType>::loadXValues(bool isFringe)
{
    if (isFringe) {
        vmovdqu32(RegType(xBaseIdx) | k1 | T_z, ptr[regXptr]);
    } else {
        vmovdqu32(RegType(xBaseIdx), ptr[regXptr]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8N1<KType>::loadAValues(int aRegIdx, bool isFringe)
{
    vxorps(RegType(vec128Idx), RegType(vec128Idx), RegType(vec128Idx));
    mov(regTmp3, 128);
    vpbroadcastb(RegType(vec128Idx), regTmp3.cvt8());

    if (isFringe) {
        vmovdqu32(RegType(tmpBaseIdx + aRegIdx) | k1 | T_z,
                  ptr[regTmpAptr + regTmp1]);
        vpaddb(RegType(tmpBaseIdx + aRegIdx) | k1 | T_z,
               RegType(tmpBaseIdx + aRegIdx), RegType(vec128Idx));
    } else {
        vmovdqu32(RegType(tmpBaseIdx + aRegIdx), ptr[regTmpAptr + regTmp1]);
        vpaddb(RegType(tmpBaseIdx + aRegIdx), RegType(tmpBaseIdx + aRegIdx),
               RegType(vec128Idx));
    }

    return dlp::jit::jitGeneratorError::success;
}
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8N1<KType>::computeVNNI(int aRegIdx, int accumRegIdx)
{
    vpdpbusd(RegType(accumBaseIdx + accumRegIdx), RegType(tmpBaseIdx + aRegIdx),
             RegType(xBaseIdx));

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8N1<KType>::processMRBlock(int mSize, bool isFringe)
{
    int mIter = mSize / 4;
    int mLeft = mSize % 4;
    xor_(regTmp1, regTmp1);
    regInit(tmpBaseIdx, tmpReg);

    for (int i = 0; i < mIter; ++i) {
        for (int j = 0; j < 4; ++j) {
            RETURN_IF_ERROR(loadAValues(j, isFringe));
            RETURN_IF_ERROR(computeVNNI(j, ((i * 4) + j)));

            add(regTmp1, regRsA);
        }
    }

    for (int j = 0; j < mLeft; ++j) {
        RETURN_IF_ERROR(loadAValues(j, isFringe));
        RETURN_IF_ERROR(computeVNNI(j, ((mIter * 4) + j)));

        add(regTmp1, regRsA);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8N1<KType>::reduceToXmm(int startIdx, int tmpIdx, int blockSize)
{
    if (blockSize > 4) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Zero out temp registers
    for (int i = 0; i < 4; ++i) {
        vxorps(RegType(tmpIdx + i), RegType(tmpIdx + i), RegType(tmpIdx + i));
    }

    // Extract upper 256-bits and add to lower 256-bits for each register
    for (int i = 0; i < blockSize; ++i) {
        vextracti32x8(Xbyak::Ymm(tmpIdx + i), RegType(startIdx + i), 1);
        vpaddd(Xbyak::Ymm(tmpIdx + i), Xbyak::Ymm(tmpIdx + i),
               Xbyak::Ymm(startIdx + i));
    }

    // First round of horizontal adds
    vphaddd(Xbyak::Ymm(tmpIdx), Xbyak::Ymm(tmpIdx),
            Xbyak::Ymm(tmpIdx + 1)); // First pair (with zero if blockSize=1)

    // Second round of horizontal adds
    vphaddd(Xbyak::Ymm(tmpIdx + 2), Xbyak::Ymm(tmpIdx + 2),
            Xbyak::Ymm(tmpIdx + 3));

    // Third round of horizontal adds
    vphaddd(Xbyak::Ymm(tmpIdx), Xbyak::Ymm(tmpIdx), Xbyak::Ymm(tmpIdx + 2));

    // Final reduction from YMM to XMM
    vextracti128(Xbyak::Xmm(tmpIdx + 1), Xbyak::Ymm(tmpIdx), 1);
    vpaddd(Xbyak::Xmm(tmpIdx), Xbyak::Xmm(tmpIdx + 1), Xbyak::Xmm(tmpIdx));

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8N1<KType>::reduceAccumulation(int mSize)
{
    for (int i = 0; i < mSize; i += vnniWidth) {
        int blockSize = ((mSize - i) < vnniWidth) ? (mSize - i) : vnniWidth;

        for (int j = 0; j < blockSize; j += 4) {
            int subBlockSize = ((blockSize - j) < 4) ? (blockSize - j) : 4;
            RETURN_IF_ERROR(
                reduceToXmm((accumBaseIdx + i + j), tmpBaseIdx, subBlockSize));

            // Insert the resulting XMM into appropriate index in destination
            // ZMM.
            vinserti32x4(RegType(accumBaseIdx + i / vnniWidth),
                         RegType(accumBaseIdx + i / vnniWidth),
                         Xbyak::Xmm(tmpBaseIdx), j / 4);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8N1<KType>::conversionCompensation(int mSize)
{
    // Load bsumptr
    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
            + offsetof(lpgemm_post_op_attr, b_col_sum_vec)]);
    vpbroadcastd(RegType(xBaseIdx), ptr[regTmp1]);

    int mLeft = mSize % vnniWidth;

    // Process full SIMD registers
    for (int i = 0; i < mSize / vnniWidth; ++i) {
        vpsubd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
               RegType(xBaseIdx));
    }

    // Process fringe elements (remaining elements less than vnniWidth)
    if (mLeft) {
        vpsubd(RegType(accumBaseIdx + mSize / vnniWidth) | k2 | T_z,
               RegType(accumBaseIdx + mSize / vnniWidth), RegType(xBaseIdx));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8N1<KType>::scaleAccByAlpha(int mSize)
{
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, alpha)]);
    vpbroadcastd(RegType(tmpBaseIdx), ptr[regKIter]);

    for (int i = 0; i < (mSize + vnniWidth - 1) / vnniWidth; ++i) {
        vpmulld(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                RegType(tmpBaseIdx));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8N1<KType>::updateCBufferPointers()
{
    mov(regTmpYptr,
        ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
            + offsetof(lpgemm_post_op_attr, buf_downscale)]);

    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
            + offsetof(lpgemm_post_op_attr, rs_c_downscale)]);

    if (c_downscale == DLP_BF16) {
        lea(regTmp1, ptr[regTmp1 * 2]);
    } else if (c_downscale == DLP_F32) {
        lea(regTmp1, ptr[regTmp1 * 4]);
    }

    mov(regKIter, regTmp2);
    imul(regKIter, regTmp1); // post_ops_c_i * rs_c_downscale
    add(regTmpYptr, regKIter);
    // regTmp1 now contains rs_c_downscale for caller to use

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8N1<KType>::scaleYWithBetaColStored(int mSize, bool is_unit_beta)
{
    if (!is_unit_beta) {
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, beta)]);
        vpbroadcastd(RegType(xBaseIdx), ptr[regKIter]);
    }

    int mLeft = mSize % vnniWidth;

    if (c_downscale == DLP_U8) {
        // Load buffer pointers for the downscaled output
        updateCBufferPointers();

        for (int i = 0; i < mSize / vnniWidth; ++i) {
            vmovdqu8(Xbyak::Xmm(yBaseIdx), ptr[regTmpYptr]);
            vpmovzxbd(RegType(yBaseIdx), Xbyak::Xmm(yBaseIdx));
            if (is_unit_beta) {
                vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(yBaseIdx));
            } else {
                vpmulld(RegType(yBaseIdx), RegType(yBaseIdx),
                        RegType(xBaseIdx));
                vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(yBaseIdx));
            }
        }

        if (mLeft) {
            // Use zero-masking (T_z) to zero unmasked elements
            vmovdqu8(Xbyak::Xmm(yBaseIdx) | k2 | T_z, ptr[regTmpYptr]);
            vpmovzxbd(RegType(yBaseIdx), Xbyak::Xmm(yBaseIdx));
            if (is_unit_beta) {
                vpaddd(RegType(accumBaseIdx + (mSize / vnniWidth)) | k2,
                       RegType(accumBaseIdx + (mSize / vnniWidth)),
                       RegType(yBaseIdx));
            } else {
                vpmulld(RegType(yBaseIdx), RegType(yBaseIdx),
                        RegType(xBaseIdx));
                vpaddd(RegType(accumBaseIdx + (mSize / vnniWidth)),
                       RegType(accumBaseIdx + (mSize / vnniWidth)),
                       RegType(yBaseIdx));
            }
        }
    } else if (c_downscale == DLP_S8) {
        // Load buffer pointers for the downscaled output
        updateCBufferPointers();

        for (int i = 0; i < mSize / vnniWidth; ++i) {
            vmovdqu8(Xbyak::Xmm(yBaseIdx), ptr[regTmpYptr]);
            vpmovsxbd(RegType(yBaseIdx), Xbyak::Xmm(yBaseIdx));
            if (is_unit_beta) {
                vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(yBaseIdx));
            } else {
                vpmulld(RegType(yBaseIdx), RegType(yBaseIdx),
                        RegType(xBaseIdx));
                vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(yBaseIdx));
            }
        }
        if (mLeft) {
            // Use zero-masking (T_z) to zero unmasked elements
            vmovdqu8(Xbyak::Xmm(yBaseIdx) | k2 | T_z, ptr[regTmpYptr]);
            vpmovsxbd(RegType(yBaseIdx), Xbyak::Xmm(yBaseIdx));
            if (is_unit_beta) {
                vpaddd(RegType(accumBaseIdx + (mSize / vnniWidth)) | k2,
                       RegType(accumBaseIdx + (mSize / vnniWidth)),
                       RegType(yBaseIdx));
            } else {
                vpmulld(RegType(yBaseIdx), RegType(yBaseIdx),
                        RegType(xBaseIdx));
                vpaddd(RegType(accumBaseIdx + (mSize / vnniWidth)),
                       RegType(accumBaseIdx + (mSize / vnniWidth)),
                       RegType(yBaseIdx));
            }
        }
    } else if (c_downscale == DLP_BF16) {
        // Load buffer pointers for the downscaled output
        updateCBufferPointers();

        for (int i = 0; i < mSize / vnniWidth; ++i) {
            vmovdqu16(Xbyak::Ymm(yBaseIdx), ptr[regTmpYptr]);
            vpmovsxwd(RegType(yBaseIdx), Xbyak::Ymm(yBaseIdx));
            vpslld(RegType(yBaseIdx), RegType(yBaseIdx), 16);
            vcvtps2dq(RegType(yBaseIdx), RegType(yBaseIdx));

            if (is_unit_beta) {
                vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(yBaseIdx));
            } else {
                vpmulld(RegType(yBaseIdx), RegType(yBaseIdx),
                        RegType(xBaseIdx));
                vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(yBaseIdx));
            }
        }

        if (mLeft) {
            // Use zero-masking (T_z) to zero unmasked elements
            vmovdqu16(Xbyak::Ymm(yBaseIdx) | k2 | T_z, ptr[regTmpYptr]);
            vpmovsxwd(RegType(yBaseIdx), Xbyak::Ymm(yBaseIdx));
            vpslld(RegType(yBaseIdx), RegType(yBaseIdx), 16);
            vcvtps2dq(RegType(yBaseIdx), RegType(yBaseIdx));
            if (is_unit_beta) {
                vpaddd(RegType(accumBaseIdx + (mSize / vnniWidth)) | k2,
                       RegType(accumBaseIdx + (mSize / vnniWidth)),
                       RegType(yBaseIdx));
            } else {
                vpmulld(RegType(yBaseIdx), RegType(yBaseIdx),
                        RegType(xBaseIdx));
                vpaddd(RegType(accumBaseIdx + (mSize / vnniWidth)),
                       RegType(accumBaseIdx + (mSize / vnniWidth)),
                       RegType(yBaseIdx));
            }
        }
    } else if (c_downscale == DLP_F32) {
        // Load buffer pointers for the downscaled output
        updateCBufferPointers();

        for (int i = 0; i < mSize / vnniWidth; ++i) {
            vcvtps2dq(RegType(yBaseIdx), ptr[regTmpYptr]);
            if (is_unit_beta) {
                vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(yBaseIdx));
            } else {
                vpmulld(RegType(yBaseIdx), RegType(yBaseIdx),
                        RegType(xBaseIdx));
                vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(yBaseIdx));
            }
        }
        if (mLeft) {
            // Use zero-masking (T_z) to zero unmasked elements
            vcvtps2dq(RegType(yBaseIdx) | k2 | T_z, ptr[regTmpYptr]);
            if (is_unit_beta) {
                vpaddd(RegType(accumBaseIdx + (mSize / vnniWidth)) | k2,
                       RegType(accumBaseIdx + (mSize / vnniWidth)),
                       RegType(yBaseIdx));
            } else {
                vpmulld(RegType(yBaseIdx), RegType(yBaseIdx),
                        RegType(xBaseIdx));
                vpaddd(RegType(accumBaseIdx + (mSize / vnniWidth)),
                       RegType(accumBaseIdx + (mSize / vnniWidth)),
                       RegType(yBaseIdx));
            }
        }
    } else { // c_downscale == DLP_S32
        for (int i = 0; i < (mSize / vnniWidth); ++i) {
            if (is_unit_beta) {
                vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       ptr[regTmpYptr]);
            } else {
                vmovdqu32(RegType(yBaseIdx), ptr[regTmpYptr]);
                vpmulld(RegType(yBaseIdx), RegType(yBaseIdx),
                        RegType(xBaseIdx));
                vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(yBaseIdx));
            }
            lea(regTmpYptr, ptr[regTmpYptr + vnniWidth * sizeof(int32_t)]);
        }

        if (mLeft) {
            if (is_unit_beta) {
                vpaddd(RegType(accumBaseIdx + (mSize / vnniWidth)) | k2,
                       RegType(accumBaseIdx + (mSize / vnniWidth)),
                       ptr[regTmpYptr]);
            } else {
                // Use zero-masking (T_z) to zero unmasked elements
                vmovdqu32(RegType(yBaseIdx) | k2 | T_z, ptr[regTmpYptr]);
                vpmulld(RegType(yBaseIdx) | k2 | T_z, RegType(yBaseIdx),
                        RegType(xBaseIdx));
                vpaddd(RegType(accumBaseIdx + (mSize / vnniWidth)) | k2,
                       RegType(accumBaseIdx + (mSize / vnniWidth)),
                       RegType(yBaseIdx));
            }
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8N1<KType>::scaleYWithBetaRowStored(int mSize, bool is_unit_beta)
{
    if (!is_unit_beta) {
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, beta)]);
        vpbroadcastd(RegType(xBaseIdx), ptr[regKIter]);
    }

    if (c_downscale == DLP_U8) {
        // Load buffer pointers for the downscaled output
        updateCBufferPointers();

        // Store temporary pointer for 3rd row access
        lea(regTmp3,
            ptr[regTmp1 + 2 * regTmp1]); // regTmp3 = 3 * rs_c_downscale

        for (int i = 0; i < ((mSize + vnniWidth - 1) / vnniWidth); ++i) {
            int blockSize  = ((mSize - vnniWidth * i) < vnniWidth)
                                 ? (mSize - vnniWidth * i)
                                 : vnniWidth;
            int num_blocks = blockSize / 4;
            int rem_elems  = blockSize % 4;

            regInit(tmpBaseIdx, tmpReg);

            for (int j = 0; j < num_blocks; j++) {
                vpbroadcastb(Xbyak::Xmm(tmpBaseIdx), ptr[regTmpYptr]);
                vpbroadcastb(Xbyak::Xmm(tmpBaseIdx + 1),
                             ptr[regTmpYptr + regTmp1]);
                vpbroadcastb(Xbyak::Xmm(tmpBaseIdx + 2),
                             ptr[regTmpYptr + 2 * regTmp1]);
                vpbroadcastb(Xbyak::Xmm(tmpBaseIdx + 3),
                             ptr[regTmpYptr + regTmp3]);

                vpunpcklbw(Xbyak::Xmm(tmpBaseIdx), Xbyak::Xmm(tmpBaseIdx),
                           Xbyak::Xmm(tmpBaseIdx + 1));
                vpunpcklbw(Xbyak::Xmm(tmpBaseIdx + 2),
                           Xbyak::Xmm(tmpBaseIdx + 2),
                           Xbyak::Xmm(tmpBaseIdx + 3));

                vpunpcklwd(Xbyak::Xmm(tmpBaseIdx), Xbyak::Xmm(tmpBaseIdx),
                           Xbyak::Xmm(tmpBaseIdx + 2));

                vpextrd(regKIter.cvt32(), Xbyak::Xmm(tmpBaseIdx), 0);
                vpinsrd(Xbyak::Xmm(yBaseIdx + i), Xbyak::Xmm(yBaseIdx + i),
                        regKIter.cvt32(), j);

                lea(regTmpYptr, ptr[regTmpYptr + regTmp1 * 4]);
            }
            if (rem_elems) {
                switch (rem_elems) {
                    case 3:
                        vpbroadcastb(Xbyak::Xmm(tmpBaseIdx + 2),
                                     ptr[regTmpYptr + 2 * regTmp1]);
                    case 2:
                        vpbroadcastb(Xbyak::Xmm(tmpBaseIdx + 1),
                                     ptr[regTmpYptr + regTmp1]);
                    case 1:
                        vpbroadcastb(Xbyak::Xmm(tmpBaseIdx), ptr[regTmpYptr]);
                    case 0:
                        break;
                }

                vpunpcklbw(Xbyak::Xmm(tmpBaseIdx), Xbyak::Xmm(tmpBaseIdx),
                           Xbyak::Xmm(tmpBaseIdx + 1));
                vpunpcklbw(Xbyak::Xmm(tmpBaseIdx + 2),
                           Xbyak::Xmm(tmpBaseIdx + 2),
                           Xbyak::Xmm(tmpBaseIdx + 3));
                vpunpcklwd(Xbyak::Xmm(tmpBaseIdx), Xbyak::Xmm(tmpBaseIdx),
                           Xbyak::Xmm(tmpBaseIdx + 2));
                vpextrd(regKIter.cvt32(), Xbyak::Xmm(tmpBaseIdx), 0);
                vpinsrd(Xbyak::Xmm(yBaseIdx + i), Xbyak::Xmm(yBaseIdx + i),
                        regKIter.cvt32(), num_blocks);
            }
        }

        vpmovzxbd(RegType(yBaseIdx), Xbyak::Xmm(yBaseIdx));
        if (!is_unit_beta) {
            vpmulld(RegType(yBaseIdx), RegType(yBaseIdx), RegType(xBaseIdx));
        }
        vpaddd(RegType(accumBaseIdx), RegType(accumBaseIdx), RegType(yBaseIdx));

    } else if (c_downscale == DLP_S8) {
        // Load buffer pointers for the downscaled output
        updateCBufferPointers();

        // Store temporary pointer for 3rd row access
        lea(regTmp3,
            ptr[regTmp1 + 2 * regTmp1]); // regTmp3 = 3 * rs_c_downscale

        for (int i = 0; i < ((mSize + vnniWidth - 1) / vnniWidth); ++i) {
            int blockSize  = ((mSize - vnniWidth * i) < vnniWidth)
                                 ? (mSize - vnniWidth * i)
                                 : vnniWidth;
            int num_blocks = blockSize / 4;
            int rem_elems  = blockSize % 4;

            regInit(tmpBaseIdx, tmpReg);

            for (int j = 0; j < num_blocks; j++) {
                vpbroadcastb(Xbyak::Xmm(tmpBaseIdx), ptr[regTmpYptr]);
                vpbroadcastb(Xbyak::Xmm(tmpBaseIdx + 1),
                             ptr[regTmpYptr + regTmp1]);
                vpbroadcastb(Xbyak::Xmm(tmpBaseIdx + 2),
                             ptr[regTmpYptr + 2 * regTmp1]);
                vpbroadcastb(Xbyak::Xmm(tmpBaseIdx + 3),
                             ptr[regTmpYptr + regTmp3]);

                vpunpcklbw(Xbyak::Xmm(tmpBaseIdx), Xbyak::Xmm(tmpBaseIdx),
                           Xbyak::Xmm(tmpBaseIdx + 1));
                vpunpcklbw(Xbyak::Xmm(tmpBaseIdx + 2),
                           Xbyak::Xmm(tmpBaseIdx + 2),
                           Xbyak::Xmm(tmpBaseIdx + 3));

                vpunpcklwd(Xbyak::Xmm(tmpBaseIdx), Xbyak::Xmm(tmpBaseIdx),
                           Xbyak::Xmm(tmpBaseIdx + 2));

                vpextrd(regKIter.cvt32(), Xbyak::Xmm(tmpBaseIdx), 0);
                vpinsrd(Xbyak::Xmm(yBaseIdx + i), Xbyak::Xmm(yBaseIdx + i),
                        regKIter.cvt32(), j);

                lea(regTmpYptr, ptr[regTmpYptr + regTmp1 * 4]);
            }
            if (rem_elems) {
                switch (rem_elems) {
                    case 3:
                        vpbroadcastb(Xbyak::Xmm(tmpBaseIdx + 2),
                                     ptr[regTmpYptr + 2 * regTmp1]);
                    case 2:
                        vpbroadcastb(Xbyak::Xmm(tmpBaseIdx + 1),
                                     ptr[regTmpYptr + regTmp1]);
                    case 1:
                        vpbroadcastb(Xbyak::Xmm(tmpBaseIdx), ptr[regTmpYptr]);
                    case 0:
                        break;
                }

                vpunpcklbw(Xbyak::Xmm(tmpBaseIdx), Xbyak::Xmm(tmpBaseIdx),
                           Xbyak::Xmm(tmpBaseIdx + 1));
                vpunpcklbw(Xbyak::Xmm(tmpBaseIdx + 2),
                           Xbyak::Xmm(tmpBaseIdx + 2),
                           Xbyak::Xmm(tmpBaseIdx + 3));
                vpunpcklwd(Xbyak::Xmm(tmpBaseIdx), Xbyak::Xmm(tmpBaseIdx),
                           Xbyak::Xmm(tmpBaseIdx + 2));
                vpextrd(regKIter.cvt32(), Xbyak::Xmm(tmpBaseIdx), 0);
                vpinsrd(Xbyak::Xmm(yBaseIdx + i), Xbyak::Xmm(yBaseIdx + i),
                        regKIter.cvt32(), num_blocks);
            }
        }

        vpmovsxbd(RegType(yBaseIdx), Xbyak::Xmm(yBaseIdx));
        if (is_unit_beta) {
            vpaddd(RegType(accumBaseIdx), RegType(accumBaseIdx),
                   RegType(yBaseIdx));
        } else {
            vpmulld(RegType(yBaseIdx), RegType(yBaseIdx), RegType(xBaseIdx));
            vpaddd(RegType(accumBaseIdx), RegType(accumBaseIdx),
                   RegType(yBaseIdx));
        }
    } else if (c_downscale == DLP_BF16) {
        // Load buffer pointers for the downscaled output
        updateCBufferPointers();

        // Store temporary pointer for 3rd row access
        lea(regTmp3,
            ptr[regTmp1 + 2 * regTmp1]); // regTmp3 = 3 * rs_c_downscale

        for (int i = 0; i < ((mSize + vnniWidth - 1) / vnniWidth); ++i) {
            int blockSize = ((mSize - vnniWidth * i) < vnniWidth)
                                ? (mSize - vnniWidth * i)
                                : vnniWidth;

            regInit(tmpBaseIdx, tmpReg);

            for (int j = 0; j < blockSize; ++j) {
                movsx(regKIter.cvt32(), word[regTmpYptr]);
                int reg_idx = (j < 8) ? tmpBaseIdx : (tmpBaseIdx + 1);
                int offset  = (j < 8) ? j : (j - 8);

                vpinsrw(Xbyak::Xmm(reg_idx), Xbyak::Xmm(reg_idx),
                        regKIter.cvt32(), offset);
                add(regTmpYptr, regTmp1);
            }

            vinserti32x4(Xbyak::Ymm(yBaseIdx + i), Xbyak::Ymm(yBaseIdx + i),
                         Xbyak::Xmm(tmpBaseIdx), 0);
            vinserti32x4(Xbyak::Ymm(yBaseIdx + i), Xbyak::Ymm(yBaseIdx + i),
                         Xbyak::Xmm(tmpBaseIdx + 1), 1);
        }

        vpmovsxwd(RegType(yBaseIdx), Xbyak::Ymm(yBaseIdx));
        vpslld(RegType(yBaseIdx), RegType(yBaseIdx), 16);
        vcvtps2dq(RegType(yBaseIdx), RegType(yBaseIdx));

        if (is_unit_beta) {
            vpaddd(RegType(accumBaseIdx), RegType(accumBaseIdx),
                   RegType(yBaseIdx));
        } else {
            vpmulld(RegType(yBaseIdx), RegType(yBaseIdx), RegType(xBaseIdx));
            vpaddd(RegType(accumBaseIdx), RegType(accumBaseIdx),
                   RegType(yBaseIdx));
        }
    } else if (c_downscale == DLP_F32) {
        updateCBufferPointers();

        lea(regTmp3, ptr[regTmp1 + 2 * regTmp1]);

        for (int i = 0; i < ((mSize + vnniWidth - 1) / vnniWidth); ++i) {
            int blockSize = ((mSize - vnniWidth * i) < vnniWidth)
                                ? (mSize - vnniWidth * i)
                                : vnniWidth;
            int n_blocks  = blockSize / 4;
            int rem_elems = blockSize % 4;

            regInit(tmpBaseIdx, tmpReg);

            for (int j = 0; j < n_blocks; j++) {
                vbroadcastss(RegType(tmpBaseIdx), ptr[regTmpYptr]);
                vbroadcastss(RegType(tmpBaseIdx + 1),
                             ptr[regTmpYptr + regTmp1]);
                vbroadcastss(RegType(tmpBaseIdx + 2),
                             ptr[regTmpYptr + 2 * regTmp1]);
                vbroadcastss(RegType(tmpBaseIdx + 3),
                             ptr[regTmpYptr + regTmp3]);

                vunpcklps(RegType(tmpBaseIdx), RegType(tmpBaseIdx),
                          RegType(tmpBaseIdx + 1));
                vunpcklps(RegType(tmpBaseIdx + 2), RegType(tmpBaseIdx + 2),
                          RegType(tmpBaseIdx + 3));

                vshufps(RegType(tmpBaseIdx), RegType(tmpBaseIdx),
                        RegType(tmpBaseIdx + 2), 0x44);

                vinsertf32x4(RegType(yBaseIdx + i), RegType(yBaseIdx + i),
                             Xbyak::Xmm(tmpBaseIdx), j);
                lea(regTmpYptr, ptr[regTmpYptr + regTmp1 * 4]);
            }

            if (rem_elems) {

                regInit(tmpBaseIdx, tmpReg);

                switch (rem_elems) {
                    case 3:
                        vbroadcastss(RegType(tmpBaseIdx + 2),
                                     ptr[regTmpYptr + regTmp1 * 2]);
                    case 2:
                        vbroadcastss(RegType(tmpBaseIdx + 1),
                                     ptr[regTmpYptr + regTmp1]);
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

                vinsertf32x4(RegType(yBaseIdx + i), RegType(yBaseIdx + i),
                             Xbyak::Xmm(tmpBaseIdx), n_blocks);
            }
        }

        vcvtps2dq(RegType(yBaseIdx), RegType(yBaseIdx));
        if (is_unit_beta) {
            vpaddd(RegType(accumBaseIdx), RegType(accumBaseIdx),
                   RegType(yBaseIdx));
        } else {
            vpmulld(RegType(yBaseIdx), RegType(yBaseIdx), RegType(xBaseIdx));
            vpaddd(RegType(accumBaseIdx), RegType(accumBaseIdx),
                   RegType(yBaseIdx));
        }
    } else { // c_downscale == DLP_S32
        // Store temporary pointer for 3rd row access
        lea(regTmp3, ptr[regRsC + 2 * regRsC]); // regTmp3 = 3 * rsC;

        for (int i = 0; i < ((mSize + vnniWidth - 1) / vnniWidth); i++) {
            int blockSize  = ((mSize - vnniWidth * i) < vnniWidth)
                                 ? (mSize - vnniWidth * i)
                                 : vnniWidth;
            int num_blocks = blockSize / 4;
            int rem_elems  = blockSize % 4;

            regInit(tmpBaseIdx, tmpReg);

            for (int j = 0; j < num_blocks; j++) {
                vpbroadcastd(RegType(tmpBaseIdx), ptr[regTmpYptr]);
                vpbroadcastd(RegType(tmpBaseIdx + 1), ptr[regTmpYptr + regRsC]);
                vpbroadcastd(RegType(tmpBaseIdx + 2),
                             ptr[regTmpYptr + 2 * regRsC]);
                vpbroadcastd(RegType(tmpBaseIdx + 3),
                             ptr[regTmpYptr + regTmp3]);

                vpunpckldq(RegType(tmpBaseIdx), RegType(tmpBaseIdx),
                           RegType(tmpBaseIdx + 1));
                vpunpckldq(RegType(tmpBaseIdx + 2), RegType(tmpBaseIdx + 2),
                           RegType(tmpBaseIdx + 3));

                vshufps(RegType(tmpBaseIdx), RegType(tmpBaseIdx),
                        RegType(tmpBaseIdx + 2), 0x44);

                vinserti32x4(RegType(yBaseIdx + i), RegType(yBaseIdx + i),
                             Xbyak::Xmm(tmpBaseIdx), j);
                lea(regTmpYptr, ptr[regTmpYptr + regRsC * 4]);
            }
            if (rem_elems) {
                switch (rem_elems) {
                    case 3:
                        vpbroadcastd(RegType(tmpBaseIdx + 2),
                                     ptr[regTmpYptr + regRsC * 2]);
                    case 2:
                        vpbroadcastd(RegType(tmpBaseIdx + 1),
                                     ptr[regTmpYptr + regRsC]);
                    case 1:
                        vpbroadcastd(RegType(tmpBaseIdx), ptr[regTmpYptr]);
                    case 0:
                        break;
                }

                vpunpckldq(RegType(tmpBaseIdx), RegType(tmpBaseIdx),
                           RegType(tmpBaseIdx + 1));
                vpunpckldq(RegType(tmpBaseIdx + 2), RegType(tmpBaseIdx + 2),
                           RegType(tmpBaseIdx + 3));

                vshufps(RegType(tmpBaseIdx), RegType(tmpBaseIdx),
                        RegType(tmpBaseIdx + 2), 0x44);

                vinserti32x4(RegType(yBaseIdx + i), RegType(yBaseIdx + i),
                             Xbyak::Xmm(tmpBaseIdx), num_blocks);
            }
        }

        if (is_unit_beta) {
            vpaddd(RegType(accumBaseIdx), RegType(accumBaseIdx),
                   RegType(yBaseIdx));
        } else {
            vpmulld(RegType(tmpBaseIdx), RegType(xBaseIdx), RegType(yBaseIdx));
            vpaddd(RegType(accumBaseIdx), RegType(accumBaseIdx),
                   RegType(tmpBaseIdx));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8N1<KType>::scaleYByBeta(int mSize)
{
    bool is_unit_beta =
        (betaScalingType == dlp::kernel_frame::scalingType::one);

    if (betaScalingType != dlp::kernel_frame::scalingType::zero) {
        mov(regTmpYptr, regYptr);

        if (yFormat == dlp::kernel_frame::storageFormat::colMajor) {
            RETURN_IF_ERROR(scaleYWithBetaColStored(mSize, is_unit_beta));
        } else {
            RETURN_IF_ERROR(scaleYWithBetaRowStored(mSize, is_unit_beta));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8N1<KType>::storeYColStored(int mSize, bool hasPostOps)
{
    int mLeft = mSize % vnniWidth;

    if (c_downscale == DLP_U8) {
        // Load buffer pointers for the downscaled output
        updateCBufferPointers();

        // Initialize temp register indices for clamping.
        int lBoundRegIdx = tmpBaseIdx + 1; // index of lower bound register
        int uBoundRegIdx = tmpBaseIdx + 2; // index of upper bound register

        // Initialize temp register with 0s to be used as the lower bound for
        // clamping.
        vpxord(RegType(lBoundRegIdx), RegType(lBoundRegIdx),
               RegType(lBoundRegIdx));
        mov(regKIter, UINT8_MAX); // Upper bound for clamping (UINT8_MAX=255).
        // Broadcast upper bound to a temp register.
        vpbroadcastd(RegType(uBoundRegIdx), regKIter.cvt32());

        for (int i = 0; i < mSize / vnniWidth; ++i) {
            if (hasPostOps) {
                // Convert post-ops accumulated result from F32 to S32.
                vcvtps2dq(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i));
            }
            // Clamp the accumulated values to lower bound.
            vpmaxsd(RegType(tmpBaseIdx), RegType(lBoundRegIdx),
                    RegType(accumBaseIdx + i));
            // Clamp the accumulated values to upper bound.
            vpminsd(RegType(tmpBaseIdx), RegType(uBoundRegIdx),
                    RegType(tmpBaseIdx));
            // Store the clamped values to downscaled buffer.
            vpmovdb(ptr[regTmpYptr], RegType(tmpBaseIdx));
        }
        if (mLeft) {
            if (hasPostOps) {
                // Convert post-ops accumulated result from F32 to S32.
                vcvtps2dq(RegType(accumBaseIdx + (mSize / vnniWidth)),
                          RegType(accumBaseIdx + (mSize / vnniWidth)));
            }
            // Clamp the accumulated values to lower bound.
            vpmaxsd(RegType(tmpBaseIdx), RegType(lBoundRegIdx),
                    RegType(accumBaseIdx + (mSize / vnniWidth)));
            // Clamp the accumulated values to upper bound.
            vpminsd(RegType(tmpBaseIdx), RegType(uBoundRegIdx),
                    RegType(tmpBaseIdx));
            // Store the clamped values to downscaled buffer based on mask.
            vpmovdb(ptr[regTmpYptr] | k2 | T_z, RegType(tmpBaseIdx));
        }
    } else if (c_downscale == DLP_S8) {
        // Load buffer pointers for the downscaled output
        updateCBufferPointers();

        // Convert and store S32 integers into packed S8 integers using signed
        // saturation.
        for (int i = 0; i < mSize / vnniWidth; ++i) {
            if (hasPostOps) {
                // Convert post-ops accumulated result from F32 to S32.
                vcvtps2dq(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i));
            }
            vpmovsdb(ptr[regTmpYptr], RegType(accumBaseIdx + i));
        }
        if (mLeft) {
            if (hasPostOps) {
                // Convert post-ops accumulated result from F32 to S32.
                vcvtps2dq(RegType(accumBaseIdx + (mSize / vnniWidth)),
                          RegType(accumBaseIdx + (mSize / vnniWidth)));
            }
            vpmovsdb(ptr[regTmpYptr] | k2 | T_z,
                     RegType(accumBaseIdx + (mSize / vnniWidth)));
        }
    } else if (c_downscale == DLP_BF16) {
        // Load buffer pointers for the downscaled output
        updateCBufferPointers();

        for (int i = 0; i < mSize / vnniWidth; ++i) {
            if (!hasPostOps) {
                // Convert accumulated S32 results to F32.
                vcvtdq2ps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i));
            }
            // Convert F32 to BF16
            vpsrld(RegType(tmpBaseIdx), RegType(accumBaseIdx + i), 16);
            mov(regTmp3, 0x00000001);
            vpbroadcastd(RegType(tmpBaseIdx + 1), regTmp3.cvt32());
            vpandd(RegType(tmpBaseIdx), RegType(tmpBaseIdx),
                   RegType(tmpBaseIdx + 1));
            mov(regTmp3, 0x00007FFF);
            vpbroadcastd(RegType(tmpBaseIdx + 1), regTmp3.cvt32());
            vpaddd(RegType(tmpBaseIdx + 2), RegType(accumBaseIdx + i),
                   RegType(tmpBaseIdx + 1));
            vpaddd(RegType(tmpBaseIdx + 2), RegType(tmpBaseIdx + 2),
                   RegType(tmpBaseIdx));
            vpsrld(RegType(tmpBaseIdx + 2), RegType(tmpBaseIdx + 2), 16);
            vpmovdw(Xbyak::Ymm(tmpBaseIdx), RegType(tmpBaseIdx + 2));
            // Store BF16 result to memory.
            vmovdqu16(ptr[regTmpYptr], Xbyak::Ymm(tmpBaseIdx));
        }

        if (mLeft) {
            if (!hasPostOps) {
                // Convert accumulated S32 results to F32.
                vcvtdq2ps(RegType(accumBaseIdx + (mSize / vnniWidth)),
                          RegType(accumBaseIdx + (mSize / vnniWidth)));
            }
            // Convert F32 to BF16
            vpsrld(RegType(tmpBaseIdx),
                   RegType(accumBaseIdx + (mSize / vnniWidth)), 16);
            mov(regTmp3, 0x00000001);
            vpbroadcastd(RegType(tmpBaseIdx + 1), regTmp3.cvt32());
            vpandd(RegType(tmpBaseIdx), RegType(tmpBaseIdx),
                   RegType(tmpBaseIdx + 1));
            mov(regTmp3, 0x00007FFF);
            vpbroadcastd(RegType(tmpBaseIdx + 1), regTmp3.cvt32());
            vpaddd(RegType(tmpBaseIdx + 2),
                   RegType(accumBaseIdx + (mSize / vnniWidth)),
                   RegType(tmpBaseIdx + 1));
            vpaddd(RegType(tmpBaseIdx + 2), RegType(tmpBaseIdx + 2),
                   RegType(tmpBaseIdx));
            vpsrld(RegType(tmpBaseIdx + 2), RegType(tmpBaseIdx + 2), 16);
            vpmovdw(Xbyak::Ymm(tmpBaseIdx), RegType(tmpBaseIdx + 2));
            // Store BF16 result to memory based on mask.
            vmovdqu16(ptr[regTmpYptr] | k2 | T_z, Xbyak::Ymm(tmpBaseIdx));
        }
    } else if (c_downscale == DLP_F32) {
        // Load buffer pointers for the downscaled output
        updateCBufferPointers();

        for (int i = 0; i < mSize / vnniWidth; ++i) {
            if (!hasPostOps) {
                // Convert from int32 to float and write to memory.
                vcvtdq2ps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i));
            }
            vmovups(ptr[regTmpYptr], RegType(accumBaseIdx + i));
        }
        if (mLeft) {
            if (!hasPostOps) {
                // Convert from S32 to F32 and write to memory based on mask.
                vcvtdq2ps(RegType(accumBaseIdx + (mSize / vnniWidth)),
                          RegType(accumBaseIdx + (mSize / vnniWidth)));
            }
            vmovups(ptr[regTmpYptr] | k2 | T_z,
                    RegType(accumBaseIdx + (mSize / vnniWidth)));
        }
    } else { // c_downscale == DLP_S32
        for (int i = 0; i < mSize / vnniWidth; ++i) {
            if (hasPostOps) {
                // Convert post-ops accumulated result from F32 to S32.
                vcvtps2dq(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i));
            }
            vmovdqu32(ptr[regTmpYptr], RegType(accumBaseIdx + i));
        }
        if (mLeft) {
            if (hasPostOps) {
                // Convert post-ops accumulated result from F32 to S32.
                vcvtps2dq(RegType(accumBaseIdx + (mSize / vnniWidth)),
                          RegType(accumBaseIdx + (mSize / vnniWidth)));
            }
            vmovdqu32(ptr[regTmpYptr] | k2 | T_z,
                      RegType(accumBaseIdx + (mSize / vnniWidth)));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8N1<KType>::storeYRowStored(int mSize, bool hasPostOps)
{
    if (c_downscale == DLP_U8) {
        // Load buffer pointers for the downscaled output
        updateCBufferPointers();

        // Initialize temp register indices for clamping.
        int lBoundRegIdx = tmpBaseIdx + 1; // index of lower bound register
        int uBoundRegIdx = tmpBaseIdx + 2; // index of upper bound register

        // Initialize temp register with 0s to be used as the lower bound for
        // clamping.
        vpxord(RegType(lBoundRegIdx), RegType(lBoundRegIdx),
               RegType(lBoundRegIdx));
        mov(regKIter, UINT8_MAX); // Upper bound for clamping (UINT8_MAX=255).
        // Broadcast upper bound to a temp register.
        vpbroadcastd(RegType(uBoundRegIdx), regKIter.cvt32());

        // Process each ZMM register
        for (int i = 0; i < ((mSize + vnniWidth - 1) / vnniWidth); ++i) {
            int elems_in_reg = (i < mSize / vnniWidth) ? vnniWidth
                                                       : (mSize % vnniWidth);
            if (elems_in_reg == 0)
                break;

            if (hasPostOps) {
                // Convert post-ops accumulated result from F32 to S32.
                vcvtps2dq(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i));
            }

            // Clamp the accumulated values to lower bound.
            vpmaxsd(RegType(tmpBaseIdx), RegType(lBoundRegIdx),
                    RegType(accumBaseIdx + i));
            // Clamp the accumulated values to upper bound.
            vpminsd(RegType(tmpBaseIdx), RegType(uBoundRegIdx),
                    RegType(tmpBaseIdx));
            // Store the clamped values to downscaled buffer.
            vpmovdb(Xbyak::Xmm(tmpBaseIdx), RegType(tmpBaseIdx));

            // Store the clamped values to downscaled buffer.
            for (int j = 0; j < elems_in_reg; j++) {
                vpextrb(ptr[regTmpYptr], Xbyak::Xmm(tmpBaseIdx), j);
                add(regTmpYptr, regTmp1); // Move to next row
            }
        }
    } else if (c_downscale == DLP_S8) {
        // Load buffer pointers for the downscaled output
        updateCBufferPointers();

        // Process each ZMM register
        for (int i = 0; i < ((mSize + vnniWidth - 1) / vnniWidth); ++i) {
            int elems_in_reg = (i < mSize / vnniWidth) ? vnniWidth
                                                       : (mSize % vnniWidth);
            if (elems_in_reg == 0)
                break;

            if (hasPostOps) {
                // Convert post-ops accumulated result from F32 to S32.
                vcvtps2dq(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i));
            }

            // Convert S32 to packed S8 using signed saturation.
            vpmovsdb(Xbyak::Xmm(tmpBaseIdx), RegType(accumBaseIdx + i));

            // Extract and store each S8 value to memory
            for (int j = 0; j < elems_in_reg; j++) {
                vpextrb(ptr[regTmpYptr], Xbyak::Xmm(tmpBaseIdx), j);
                add(regTmpYptr, regTmp1); // Move to next row
            }
        }
    } else if (c_downscale == DLP_BF16) {
        // Load buffer pointers for the downscaled output
        updateCBufferPointers();

        // Process each ZMM register
        for (int i = 0; i < ((mSize + vnniWidth - 1) / vnniWidth); ++i) {
            int elems_in_reg = (i < mSize / vnniWidth) ? vnniWidth
                                                       : (mSize % vnniWidth);
            if (elems_in_reg == 0)
                break;

            if (!hasPostOps) {
                // Convert from S32 to F32.
                vcvtdq2ps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i));
            }
            // Convert from F32 to BF16.
            vpsrld(RegType(tmpBaseIdx + 2), RegType(accumBaseIdx + i), 16);
            mov(regTmp3, 0x00000001);
            vpbroadcastd(RegType(tmpBaseIdx + 1), regTmp3.cvt32());
            vpandd(RegType(tmpBaseIdx + 2), RegType(tmpBaseIdx + 2),
                   RegType(tmpBaseIdx + 1));
            mov(regTmp3, 0x00007FFF);
            vpbroadcastd(RegType(tmpBaseIdx + 1), regTmp3.cvt32());
            vpaddd(RegType(tmpBaseIdx), RegType(accumBaseIdx + i),
                   RegType(tmpBaseIdx + 1));
            vpaddd(RegType(tmpBaseIdx), RegType(tmpBaseIdx),
                   RegType(tmpBaseIdx + 2));
            vpsrld(RegType(tmpBaseIdx), RegType(tmpBaseIdx), 16);
            vpmovdw(Xbyak::Ymm(tmpBaseIdx), RegType(tmpBaseIdx));

            // Extract 2 128-bit chunks containing 8 bfloat16 elements each from
            // the 512-bit accumulator register into separate temp registers.
            for (int j = 0; j < elems_in_reg; j += 8) {
                int tmp_reg_idx = j / 8;
                vextracti32x4(Xbyak::Xmm(tmpBaseIdx + 1 + tmp_reg_idx),
                              Xbyak::Ymm(tmpBaseIdx), tmp_reg_idx);
            }

            for (int j = 0; j < elems_in_reg; j++) {
                int tmp_reg_idx = (j / 8) + 1; // temp register index containing
                                               // required value.
                int reg_idx = j % 8;           // position of required value
                                               // within above temp register.

                if (reg_idx == 0) {
                    // Extract and store the first element directly to memory.
                    vpextrw(ptr[regTmpYptr],
                            Xbyak::Xmm(tmpBaseIdx + tmp_reg_idx), 0);
                } else {
                    // Extract and store latter elements directly to memory
                    // based on index.
                    vpextrw(ptr[regTmpYptr],
                            Xbyak::Xmm(tmpBaseIdx + tmp_reg_idx), reg_idx);
                }

                add(regTmpYptr, regTmp1); // Move to next row
            }
        }
    } else if (c_downscale == DLP_F32) {
        // Load buffer pointers for the downscaled output
        updateCBufferPointers();

        // Process each ZMM register
        for (int i = 0; i < ((mSize + vnniWidth - 1) / vnniWidth); ++i) {
            int elems_in_reg = (i < mSize / vnniWidth) ? vnniWidth
                                                       : (mSize % vnniWidth);
            if (elems_in_reg == 0)
                break;

            if (!hasPostOps) {
                // Convert from S32 to F32.
                vcvtdq2ps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i));
            }

            // Extract 4 128-bit chunks containing 4 int32 elements each from
            // the 512-bit accumulator register into separate temp registers.
            for (int j = 0; j < elems_in_reg; j += 4) {
                int tmp_reg_idx = j / 4;
                vextractf32x4(Xbyak::Xmm(tmpBaseIdx + tmp_reg_idx),
                              RegType(accumBaseIdx + i), tmp_reg_idx);
            }

            // Store each extracted value into its appropriate position in
            // memory.
            for (int j = 0; j < elems_in_reg; j++) {
                int tmp_reg_idx = j / 4; // temp register index containing
                                         // required value.
                int reg_idx = j % 4;     // position of required value within
                                         // above temp register.

                if (reg_idx == 0) {
                    // Store the first element directly to memory.
                    vmovss(ptr[regTmpYptr],
                           Xbyak::Xmm(tmpBaseIdx + tmp_reg_idx));
                } else {
                    // Store latter elements directly to memory based on index.
                    vpextrd(ptr[regTmpYptr],
                            Xbyak::Xmm(tmpBaseIdx + tmp_reg_idx), reg_idx);
                }

                add(regTmpYptr, regTmp1); // Move to next row
            }
        }
    } else { // c_downscale == DLP_S32
        // Process each ZMM register
        for (int i = 0; i < ((mSize + vnniWidth - 1) / vnniWidth); ++i) {
            int elems_in_reg = (i < mSize / vnniWidth) ? vnniWidth
                                                       : (mSize % vnniWidth);
            if (elems_in_reg == 0)
                break;

            if (hasPostOps) {
                // Convert post-ops accumulated result from F32 to S32.
                vcvtps2dq(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i));
            }

            // Extract 4 128-bit chunks containing 4 int32 elements each from
            // the 512-bit accumulator register into separate temp registers.
            for (int j = 0; j < elems_in_reg; j += 4) {
                int tmp_reg_idx = j / 4;
                vextracti32x4(Xbyak::Xmm(tmpBaseIdx + tmp_reg_idx),
                              RegType(accumBaseIdx + i), tmp_reg_idx);
            }

            // Store each extracted value into its appropriate position in
            // memory.
            for (int j = 0; j < elems_in_reg; j++) {
                int tmp_reg_idx = j / 4; // temp register index containing
                                         // required value.
                int reg_idx = j % 4;     // position of required value within
                                         // above temp register.

                if (reg_idx == 0) {
                    // Store the first element directly to memory.
                    vmovd(ptr[regTmpYptr],
                          Xbyak::Xmm(tmpBaseIdx + tmp_reg_idx));
                } else {
                    // Store latter elements directly to memory based on index.
                    vpextrd(ptr[regTmpYptr],
                            Xbyak::Xmm(tmpBaseIdx + tmp_reg_idx), reg_idx);
                }

                add(regTmpYptr, regRsC); // Move to next row
            }
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8N1<KType>::storeY(int mSize, bool hasPostOps)
{
    // Store values from Y
    mov(regTmpYptr, regYptr);
    if (yFormat == dlp::kernel_frame::storageFormat::colMajor) {
        RETURN_IF_ERROR(storeYColStored(mSize, hasPostOps));
    } else {
        RETURN_IF_ERROR(storeYRowStored(mSize, hasPostOps));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8N1<KType>::generateKernel(utils::gemvN1GeneratorParams& params)
{
    Xbyak::util::StackFrame frame(this, 1, 13, 0);
    initializeStackFrame(frame);
    initializeParameters(params);

    RETURN_IF_ERROR(allocateRegisters());

    // Load A & Y pointers from the stack
    mov(regAptr, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, a)]);
    mov(regYptr, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, y)]);
    mov(regTmpAptr, regAptr);
    mov(regTmpYptr, regAptr);

    // Broadcast 128 to temp register
    // vxorps(RegType(vec128Idx), RegType(vec128Idx), RegType(vec128Idx));
    // mov(regTmp1, 128);
    // vpbroadcastb(RegType(vec128Idx), regTmp1.cvt8());

    inLocalLabel();

    // Load Masks for fringe handling
    kmovq(
        k1,
        ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kmask_i8_avx512)]);
    kmovd(k2,
          ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, mmask_avx512)]);

    if (params.mloop) { // Handle m-loop
        mov(regMIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, m_iter)]);
        test(regMIter, regMIter);
        jz(".MLOOP_END", T_NEAR);

        L(".MLOOP_BEGIN");

        // Zero out accumulator registers for this m iteration
        regInit(accumBaseIdx, MR);

        // K-loop is not required if alpha = 0. Simply perform beta*Y and store
        // result in that case.
        if (alphaScalingType != dlp::kernel_frame::scalingType::zero) {
            // Load A pointers for the current iteration.
            // One is used in the m-loop, while other in the k-loop
            mov(regTmpAptr, regAptr);
            mov(regTmpYptr, regAptr);

            // Load X pointer
            mov(regXptr,
                ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, x)]);

            if (params.kloop) {
                mov(regKIter,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvN1Params, k_iter)]);
                test(regKIter, regKIter);
                jz(".KLOOP_END", T_NEAR);

                L(".KLOOP_BEGIN");

                // Load X vector
                RETURN_IF_ERROR(loadXValues());

                // Process all rows
                RETURN_IF_ERROR(processMRBlock(MR));

                add(regTmpYptr,
                    RegBytes); // Move the A column pointer to the next K chunk
                mov(regTmpAptr, regTmpYptr); // Update the A pointer
                add(regXptr,
                    RegBytes); // Update the X pointer to the next K chunk

                dec(regKIter);
                jnz(".KLOOP_BEGIN", T_NEAR);
            }
            L(".KLOOP_END");

            if (params.kfringe) {
                mov(regKIter,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvN1Params, k_left)]);
                test(regKIter, regKIter);
                jz(".KFRINGE_END", T_NEAR);

                L(".KFRINGE_BEGIN");

                // Load X vector
                RETURN_IF_ERROR(loadXValues(true));

                // Process all rows
                RETURN_IF_ERROR(processMRBlock(MR, true));
            }

            L(".KFRINGE_END");

            RETURN_IF_ERROR(reduceAccumulation(MR));

            RETURN_IF_ERROR(conversionCompensation(MR));

            if (alphaScalingType != dlp::kernel_frame::scalingType::one) {
                RETURN_IF_ERROR(scaleAccByAlpha(MR));
            }
        }

        RETURN_IF_ERROR(scaleYByBeta(MR));

        // Create and set up kernelOphandler if there are post-ops
        if (!params.kernelOps.empty()) {
            // Convert to F32 since post-ops expect accumulators in F32.
            for (int i = 0; i < ((MR + vnniWidth - 1) / vnniWidth); ++i) {
                vcvtdq2ps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i));
            }

            gen::kernelOpsHandler kernelOpsHandler(this, params.kType);

            int  NR          = 1;
            bool useMask     = !(MR / vnniWidth);
            int  numMaskRegs = 1;
            int  numCRegs    = (MR / vnniWidth);

            RETURN_IF_ERROR(kernelOpsHandler.generateKernelOps(
                params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemv_n1, MR,
                NR, useMask, numMaskRegs, accumBaseIdx, numCRegs));

            kernelOpsHandler.generateKernelOpsAttributes();

            // store C assuming F32 accumulators after post-ops
            RETURN_IF_ERROR(storeY(MR, true));
            jmp(".POST_STOREC", T_NEAR);
        }

        RETURN_IF_ERROR(storeY(MR));
        L(".POST_STOREC");

        mov(regTmp1, MR);
        imul(regTmp1, regRsA);
        add(regAptr, regTmp1);

        mov(regTmp1, MR);
        imul(regTmp1, regRsC);
        add(regYptr, regTmp1);

        if (c_downscale != DLP_S32) {
            lea(regTmp2, ptr[regTmp2 + MR]);
        }

        dec(regMIter);
        jnz(".MLOOP_BEGIN", T_NEAR);
    }
    L(".MLOOP_END");

    if (params.mfringe) { // Handle m-fringe
        L(".MFRINGE_BEGIN");

        regInit(accumBaseIdx, M_LEFT);

        // K-loop is not required if alpha = 0. Simply perform beta*Y and store
        // result in that case.
        if (alphaScalingType != dlp::kernel_frame::scalingType::zero) {
            // Load A pointers for the current iteration.
            // One is used in the m-loop, while other in the k-loop
            mov(regTmpAptr, regAptr);
            mov(regTmpYptr, regAptr);

            // Load X pointer
            mov(regXptr,
                ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, x)]);

            if (params.kloop) {
                mov(regKIter,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvN1Params, k_iter)]);
                test(regKIter, regKIter);
                jz(".MFRINGE_KLOOP_END", T_NEAR);

                L(".MFRINGE_KLOOP_BEGIN");

                // Load X vector
                RETURN_IF_ERROR(loadXValues());

                RETURN_IF_ERROR(processMRBlock(M_LEFT));

                add(regTmpYptr,
                    RegBytes); // Move the A column pointer to the next K chunk
                mov(regTmpAptr, regTmpYptr); // Update the A pointer
                add(regXptr,
                    RegBytes); // Update the X pointer to the next K chunk

                dec(regKIter);
                jnz(".MFRINGE_KLOOP_BEGIN", T_NEAR);
            }
            L(".MFRINGE_KLOOP_END");

            if (params.kfringe) {
                mov(regKIter,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvN1Params, k_left)]);
                test(regKIter, regKIter);
                jz(".MFRINGE_KFRINGE_END", T_NEAR);

                L(".M_FRINGE_KFRINGE_BEGIN");

                // Load X vector
                RETURN_IF_ERROR(loadXValues(true));

                // Process all rows
                RETURN_IF_ERROR(processMRBlock(M_LEFT, true));
            }
            L(".MFRINGE_KFRINGE_END");

            // Reduce the accumulation registers to XMMs and store in ZMMs.
            RETURN_IF_ERROR(reduceAccumulation(M_LEFT));

            RETURN_IF_ERROR(conversionCompensation(M_LEFT));

            if (alphaScalingType != dlp::kernel_frame::scalingType::one) {
                RETURN_IF_ERROR(scaleAccByAlpha(M_LEFT));
            }
        }

        RETURN_IF_ERROR(scaleYByBeta(M_LEFT));

        // Create and set up kernelOphandler if there are post-ops
        if (!params.kernelOps.empty()) {
            // Convert to F32 since post-ops expect accumulators in F32.
            for (int i = 0; i < ((M_LEFT + vnniWidth - 1) / vnniWidth); ++i) {
                vcvtdq2ps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i));
            }

            gen::kernelOpsHandler kernelOpsHandler(this, params.kType);

            int  NR          = 1;
            bool useMask     = !(M_LEFT / vnniWidth);
            int  numMaskRegs = 1;
            int  numCRegs    = (M_LEFT / vnniWidth);

            RETURN_IF_ERROR(kernelOpsHandler.generateKernelOps(
                params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemv_n1,
                M_LEFT, NR, useMask, numMaskRegs, accumBaseIdx, numCRegs));

            kernelOpsHandler.generateKernelOpsAttributes();

            // store C assuming F32 accumulators after post-ops
            RETURN_IF_ERROR(storeY(M_LEFT, true));
            jmp(".POST_STOREC_MFRINGE", T_NEAR);
        }

        RETURN_IF_ERROR(storeY(M_LEFT));
        L(".POST_STOREC_MFRINGE");

        mov(regTmp1, M_LEFT);
        imul(regTmp1, regRsA);
        add(regAptr, regTmp1);

        mov(regTmp1, M_LEFT);
        imul(regTmp1, regRsC);
        add(regYptr, regTmp1);

        if (c_downscale != DLP_S32) {
            lea(regTmp2, ptr[regTmp2 + M_LEFT]);
        }
    }
    L(".MFRINGE_END");

    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}
// End S8 GEMV N=1 JIT

// Begin S8 GEMV M=1 JIT
template<utils::kernelInstrType KType>
jitGEMVS8M1<KType>::jitGEMVS8M1(void* buffer, size_t bufferSize)
    : Xbyak::CodeGenerator(bufferSize, buffer)
{
}

template<utils::kernelInstrType KType>
void
jitGEMVS8M1<KType>::initializeStackFrame(Xbyak::util::StackFrame& frame)
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
jitGEMVS8M1<KType>::initializeParameters(utils::gemvM1GeneratorParams& params)
{
    RegBytes = Traits::regBytes;
    numRegs  = Traits::numRegs;

    // Set dimensions from params
    NR               = params.NR;
    N_LEFT           = params.N_LEFT;
    N_LEFT_16        = params.N_LEFT_16;
    N_LEFT_LT16      = params.N_LEFT_LT16;
    KC               = params.KC;
    K_SUB_ITER       = params.K_SUB_ITER;
    yFormat          = params.yFormat;
    alphaScalingType = params.alphaScalingType;
    betaScalingType  = params.betaScalingType;
    c_downscale      = params.c_downscale; // Type of downscale

    vnniWidth = RegBytes / sizeof(int32_t); // Since accumulation is being done
                                            // in S32.

    mov(regRsB, ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, rsB)]);
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8M1<KType>::allocateRegisters()
{
    if (NR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    yReg      = (NR / vnniWidth);
    xReg      = K_SUB_ITER;
    bReg      = NR / vnniWidth;
    accumReg  = (NR / vnniWidth) * K_SUB_ITER;
    vec128Reg = 1;

    accumBaseIdx  = numRegs - accumReg;
    xBaseIdx      = accumBaseIdx - xReg;
    yBaseIdx      = numRegs - yReg;
    bBaseIdx      = xBaseIdx - bReg;
    vec128BaseIdx = bBaseIdx - vec128Reg;

    if (vec128BaseIdx < 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMVS8M1<KType>::regInit(int baseIdx, int numRegs)
{
    // Zero out accumulation registers
    vxorps(RegType(baseIdx), RegType(baseIdx), RegType(baseIdx));
    for (int i = 1; i < numRegs; ++i) {
        vmovaps(RegType(baseIdx + i), RegType(baseIdx));
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8M1<KType>::offsetBPtr(int temp)
{
    // Calculate B pointer offset using power-of-2 decomposition
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
jitGEMVS8M1<KType>::computeKxNfringe()
{
    for (int j = 0; j < K_SUB_ITER; j++) {
        vpbroadcastd(RegType(xBaseIdx + j), ptr[regXptr + j * 4]);
        vpaddb(RegType(xBaseIdx + j), RegType(xBaseIdx + j),
               RegType(vec128BaseIdx));
    }

    int n_iter = N_LEFT / vnniWidth;
    int n_left = N_LEFT % vnniWidth;

    for (int j = 0; j < K_SUB_ITER; j++) {
        for (int i = 0; i < n_iter; i++) {
            vmovdqu32(RegType(bBaseIdx + i),
                      ptr[regTmp2 + i * vnniWidth * sizeof(int32_t)]);

            vpdpbusd(RegType(accumBaseIdx + K_SUB_ITER * i + j),
                     RegType(xBaseIdx + j), RegType(bBaseIdx + i));
        }
        if (n_left) {
            vmovdqu32(RegType(bBaseIdx + n_iter) | k1 | T_z,
                      ptr[regTmpYptr + j * 64]);
            vpdpbusd(RegType(accumBaseIdx + K_SUB_ITER * n_iter + j),
                     RegType(xBaseIdx + j), RegType(bBaseIdx + n_iter));
        }
        add(regTmp2, regRsB);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8M1<KType>::computeKxNR(bool nMask)
{
    mov(regTmp2, regBptr);

    if (!nMask) {
        for (int i = 0; i < K_SUB_ITER; ++i) {
            vpbroadcastd(RegType(xBaseIdx + i), ptr[regXptr + i * 4]);
            vpaddb(RegType(xBaseIdx + i), RegType(xBaseIdx + i),
                   RegType(vec128BaseIdx));
        }

        int nIter = NR / vnniWidth;
        for (int j = 0; j < K_SUB_ITER; j += 1) {
            for (int i = 0; i < nIter; ++i) {
                vmovdqu32(RegType(bBaseIdx + i),
                          ptr[regTmp2 + i * vnniWidth * sizeof(int32_t)]);
                vpdpbusd(RegType(accumBaseIdx + K_SUB_ITER * i + j),
                         RegType(xBaseIdx + j), RegType(bBaseIdx + i));
            }
            add(regTmp2, regRsB);
        }
    } else {
        computeKxNfringe();
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8M1<KType>::compute1xNfringe(bool isLastKGroup)
{
    if (isLastKGroup) {
        vmovdqu8(Xbyak::Xmm(xBaseIdx) | k2 | T_z, ptr[regXptr]);
        vpbroadcastd(RegType(xBaseIdx), Xbyak::Xmm(xBaseIdx));
        vpaddb(RegType(xBaseIdx), RegType(xBaseIdx), RegType(vec128BaseIdx));
    } else {
        vpbroadcastd(RegType(xBaseIdx), ptr[regXptr]);
        vpaddb(RegType(xBaseIdx), RegType(xBaseIdx), RegType(vec128BaseIdx));
    }

    int n_iter = N_LEFT / vnniWidth;
    int n_left = N_LEFT % vnniWidth;

    for (int i = 0; i < n_iter; i++) {
        xor_(regTmp1, regTmp1);
        lea(regTmp1, ptr[regTmp1 + i * vnniWidth * sizeof(int32_t)]);
        vmovdqu32(RegType(bBaseIdx + i), ptr[regTmp2 + regTmp1]);
        vpdpbusd(RegType(accumBaseIdx + K_SUB_ITER * i), RegType(xBaseIdx),
                 RegType(bBaseIdx + i));
    }

    if (n_left) {
        vmovdqu32(RegType(bBaseIdx + n_iter) | k1 | T_z, ptr[regTmpYptr]);
        vpdpbusd(RegType(accumBaseIdx + K_SUB_ITER * n_iter), RegType(xBaseIdx),
                 RegType(bBaseIdx + n_iter));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8M1<KType>::compute1xNR(bool nMask, bool isLastKGroup)
{
    mov(regTmp2, regBptr);

    if (!nMask) {
        if (isLastKGroup) {
            vmovdqu8(Xbyak::Xmm(xBaseIdx) | k2 | T_z, ptr[regXptr]);
            vpbroadcastd(RegType(xBaseIdx), Xbyak::Xmm(xBaseIdx));
            vpaddb(RegType(xBaseIdx), RegType(xBaseIdx),
                   RegType(vec128BaseIdx));
        } else {
            vpbroadcastd(RegType(xBaseIdx), ptr[regXptr]);
            vpaddb(RegType(xBaseIdx), RegType(xBaseIdx),
                   RegType(vec128BaseIdx));
        }

        int nIter = NR / vnniWidth;
        for (int i = 0; i < nIter; ++i) {
            vmovdqu32(RegType(bBaseIdx + i),
                      ptr[regTmp2 + i * vnniWidth * sizeof(int32_t)]);
            vpdpbusd(RegType(accumBaseIdx + K_SUB_ITER * i), RegType(xBaseIdx),
                     RegType(bBaseIdx + i));
        }
    } else {
        compute1xNfringe(isLastKGroup);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8M1<KType>::kLoop(bool kfringe, bool nfringe)
{
    // Defining labels locally to avoid redefinition issues
    Xbyak::Label sub_loop_kc_main_loop_start;
    Xbyak::Label sub_loop_kc_main_loop_end;
    Xbyak::Label sub_loop_kc_fringe_loop_start;
    Xbyak::Label sub_loop_kc_fringe_loop_end;
    Xbyak::Label sub_loop_kf_main_loop_start;
    Xbyak::Label sub_loop_kf_main_loop_end;
    Xbyak::Label sub_loop_kf_fringe_loop_start;
    Xbyak::Label sub_loop_kf_fringe_loop_end;
    Xbyak::Label sub_loop_k_last;
    Xbyak::Label sub_loop_k_fringe;

    mov(regTmpYptr, regBptr);

    // Update pointers to process only the nLeft panel
    if (N_LEFT > 16) {
        mov(regTmp2, regRsB);
        shr(regTmp2, 2);
        imul(regTmp2, regPsB);
        lea(regTmpYptr, ptr[regTmpYptr + regTmp2]);
    }

    if (!kfringe) {
        mov(regKSubIter,
            ptr[stackPtr
                + offsetof(dlp::kernels::gemvM1Params, k_iter_sub_iter)]);
        test(regKSubIter, regKSubIter);
        jz(sub_loop_kc_main_loop_end, T_NEAR);
        L(sub_loop_kc_main_loop_start);

        computeKxNR(nfringe);

        // Update pointers
        lea(regXptr, ptr[regXptr + K_SUB_ITER * 4]);
        lea(regBptr, ptr[regBptr + regRsB * K_SUB_ITER]);
        lea(regTmpYptr, ptr[regTmpYptr + 64 * K_SUB_ITER]);

        dec(regKSubIter);
        jnz(sub_loop_kc_main_loop_start, T_NEAR);

        L(sub_loop_kc_main_loop_end);
        mov(regKSubIter,
            ptr[stackPtr
                + offsetof(dlp::kernels::gemvM1Params, k_iter_sub_left)]);
        test(regKSubIter, regKSubIter);
        jz(sub_loop_kc_fringe_loop_end, T_NEAR);
        L(sub_loop_kc_fringe_loop_start);

        compute1xNR(nfringe, false);

        // Update pointers
        lea(regXptr, ptr[regXptr + 4]);
        lea(regBptr, ptr[regBptr + regRsB]);
        lea(regTmpYptr, ptr[regTmpYptr + 64]);

        dec(regKSubIter);
        jnz(sub_loop_kc_fringe_loop_start, T_NEAR);

        L(sub_loop_kc_fringe_loop_end);
    } else {
        inLocalLabel();
        mov(regKSubIter,
            ptr[stackPtr
                + offsetof(dlp::kernels::gemvM1Params, k_left_sub_iter)]);
        test(regKSubIter, regKSubIter);
        jz(sub_loop_kf_main_loop_end, T_NEAR);

        L(sub_loop_kf_main_loop_start);

        computeKxNR(nfringe);

        // Update pointers
        lea(regXptr, ptr[regXptr + K_SUB_ITER * 4]);
        lea(regBptr, ptr[regBptr + regRsB * K_SUB_ITER]);
        lea(regTmpYptr, ptr[regTmpYptr + 64 * K_SUB_ITER]);

        dec(regKSubIter);
        jnz(sub_loop_kf_main_loop_start, T_NEAR);

        L(sub_loop_kf_main_loop_end);
        mov(regKSubIter,
            ptr[stackPtr
                + offsetof(dlp::kernels::gemvM1Params, k_left_sub_left)]);
        test(regKSubIter, regKSubIter);
        jz(sub_loop_kf_fringe_loop_end, T_NEAR);

        L(sub_loop_kf_fringe_loop_start);

        cmp(regKSubIter, 1);
        je(sub_loop_k_last, T_NEAR);

        compute1xNR(nfringe, false);
        jmp(sub_loop_k_fringe, T_NEAR);

        L(sub_loop_k_last);
        compute1xNR(nfringe, true);

        L(sub_loop_k_fringe);
        lea(regXptr, ptr[regXptr + 4]);
        lea(regBptr, ptr[regBptr + regRsB]);
        lea(regTmpYptr, ptr[regTmpYptr + 64]);

        dec(regKSubIter);
        jnz(sub_loop_kf_fringe_loop_start, T_NEAR);

        L(sub_loop_kf_fringe_loop_end);
        outLocalLabel();
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8M1<KType>::accumulateKSubIters()
{
    int nIter = NR / vnniWidth;
    for (int i = 0; i < nIter; ++i) {
        for (int j = 1; j < K_SUB_ITER; j += 1) {
            vpaddd(RegType(accumBaseIdx + K_SUB_ITER * i),
                   RegType(accumBaseIdx + K_SUB_ITER * i),
                   RegType(accumBaseIdx + K_SUB_ITER * i + j));
        }
        vmovdqu32(RegType(accumBaseIdx + i),
                  RegType(accumBaseIdx + K_SUB_ITER * i));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8M1<KType>::conversionCompensation(bool isFringe)
{
    if (!isFringe) {
        // Load b_col_sum_vec pointer
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, b_col_sum_vec)]);

        // Load b_sum_offset
        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, b_sum_offset)]);

        // Calculate address: b_col_sum_vec + b_sum_offset * sizeof(int32_t)
        // and store in regTmp1
        lea(regTmp1, ptr[regTmp1 + regTmp2 * sizeof(int32_t)]);

        int nIter = NR / vnniWidth;
        for (int i = 0; i < nIter; i++) {
            vmovdqu32(RegType(xBaseIdx), ptr[regTmp1 + i * RegBytes]);
            vpsubd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                   RegType(xBaseIdx));
        }

        return dlp::jit::jitGeneratorError::success;
    }

    // isFringe == true
    // Load b_col_sum_vec masks to k3-6 mask registers.
    kmovw(
        k3,
        ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, k1Mask_i8_avx512)]);
    kmovw(
        k4,
        ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, k2Mask_i8_avx512)]);
    kmovw(
        k5,
        ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, k3Mask_i8_avx512)]);
    kmovw(
        k6,
        ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, k4Mask_i8_avx512)]);

    // Load b_col_sum_vec pointer
    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
            + offsetof(lpgemm_post_op_attr, b_col_sum_vec)]);

    // Load b_sum_offset
    mov(regTmp2,
        ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
            + offsetof(lpgemm_post_op_attr, b_sum_offset)]);

    // Calculate address: b_col_sum_vec + b_sum_offset * sizeof(int32_t)
    // and store in regTmp1
    lea(regTmp1, ptr[regTmp1 + regTmp2 * sizeof(int32_t)]);

    int nIter = NR / vnniWidth;
    for (int i = 0; i < nIter; i++) {
        if (i == 0) {
            vmovdqu32(RegType(xBaseIdx) | k3 | T_z,
                      ptr[regTmp1 + i * RegBytes]);
        } else if (i == 1) {
            vmovdqu32(RegType(xBaseIdx) | k4 | T_z,
                      ptr[regTmp1 + i * RegBytes]);
        } else if (i == 2) {
            vmovdqu32(RegType(xBaseIdx) | k5 | T_z,
                      ptr[regTmp1 + i * RegBytes]);
        } else if (i == 3) {
            vmovdqu32(RegType(xBaseIdx) | k6 | T_z,
                      ptr[regTmp1 + i * RegBytes]);
        }

        vpsubd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
               RegType(xBaseIdx));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8M1<KType>::scaleWithAlpha()
{
    if (alphaScalingType != dlp::kernel_frame::scalingType::one) {
        mov(regKSubIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, alpha)]);
        vpbroadcastd(RegType(xBaseIdx), ptr[regKSubIter]);

        int nIter = NR / vnniWidth;
        for (int i = 0; i < nIter; ++i) {
            vpmulld(RegType(accumBaseIdx + i), RegType(xBaseIdx),
                    RegType(accumBaseIdx + i));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMVS8M1<KType>::updateYBufferPointers()
{
    mov(regTmpYptr,
        ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
            + offsetof(lpgemm_post_op_attr, buf_downscale)]);

    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
            + offsetof(lpgemm_post_op_attr, post_op_c_j)]);

    add(regTmp1, regIncN);
    if (c_downscale == DLP_BF16) {
        lea(regTmp1, ptr[regTmp1 * 2]);
    } else if (c_downscale == DLP_F32) {
        lea(regTmp1, ptr[regTmp1 * 4]);
    }

    add(regTmpYptr, regTmp1);
    // regTmpYptr now points to the start of the Y buffer for current N index
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8M1<KType>::scaleYWithBeta(bool nMask)
{
    if (betaScalingType == dlp::kernel_frame::scalingType::zero) {
        return dlp::jit::jitGeneratorError::success;
    }

    bool is_unit_beta =
        (betaScalingType == dlp::kernel_frame::scalingType::one);
    if (!is_unit_beta) {
        mov(regKSubIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, beta)]);

        vpbroadcastd(RegType(xBaseIdx), ptr[regKSubIter]);
    }

    if (c_downscale == DLP_S8) {
        // Load buffer pointers for the downscaled output
        updateYBufferPointers();

        if (!nMask) {
            int nIter = NR / vnniWidth;
            for (int i = 0; i < nIter; i++) {
                vmovdqu8(Xbyak::Xmm(yBaseIdx + i), ptr[regTmpYptr + i * 16]);
                vpmovsxbd(RegType(yBaseIdx + i), Xbyak::Xmm(yBaseIdx + i));

                if (!is_unit_beta) {
                    vpmulld(RegType(yBaseIdx + i), RegType(yBaseIdx + i),
                            RegType(xBaseIdx));
                }
                vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(yBaseIdx + i));
            }
        } else {
            int n_iter = N_LEFT / vnniWidth;
            int n_left = N_LEFT % vnniWidth;

            for (int i = 0; i < n_iter; i++) {
                vmovdqu8(Xbyak::Xmm(yBaseIdx + i), ptr[regTmpYptr + i * 16]);
                vpmovsxbd(RegType(yBaseIdx + i), Xbyak::Xmm(yBaseIdx + i));

                if (!is_unit_beta) {
                    vpmulld(RegType(yBaseIdx + i), RegType(yBaseIdx + i),
                            RegType(xBaseIdx));
                }
                vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(yBaseIdx + i));
            }

            if (n_left) {
                // Use zero-masking (T_z) to zero unmasked elements
                vmovdqu8(Xbyak::Xmm(yBaseIdx + n_iter) | k1 | T_z,
                         ptr[regTmpYptr + n_iter * 16]);
                vpmovsxbd(RegType(yBaseIdx + n_iter),
                          Xbyak::Xmm(yBaseIdx + n_iter));

                if (!is_unit_beta) {
                    vpmulld(RegType(yBaseIdx + n_iter),
                            RegType(yBaseIdx + n_iter), RegType(xBaseIdx));
                }
                vpaddd(RegType(accumBaseIdx + n_iter),
                       RegType(accumBaseIdx + n_iter),
                       RegType(yBaseIdx + n_iter));
            }
        }
    } else if (c_downscale == DLP_U8) {
        // Load buffer pointers for the downscaled output
        updateYBufferPointers();

        if (!nMask) {
            int nIter = NR / vnniWidth;
            for (int i = 0; i < nIter; i++) {
                vmovdqu8(Xbyak::Xmm(yBaseIdx + i), ptr[regTmpYptr + i * 16]);
                vpmovzxbd(RegType(yBaseIdx + i), Xbyak::Xmm(yBaseIdx + i));

                if (!is_unit_beta) {
                    vpmulld(RegType(yBaseIdx + i), RegType(yBaseIdx + i),
                            RegType(xBaseIdx));
                }
                vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(yBaseIdx + i));
            }
        } else {
            int n_iter = N_LEFT / vnniWidth;
            int n_left = N_LEFT % vnniWidth;

            for (int i = 0; i < n_iter; i++) {
                vmovdqu8(Xbyak::Xmm(yBaseIdx + i), ptr[regTmpYptr + i * 16]);
                vpmovzxbd(RegType(yBaseIdx + i), Xbyak::Xmm(yBaseIdx + i));

                if (!is_unit_beta) {
                    vpmulld(RegType(yBaseIdx + i), RegType(yBaseIdx + i),
                            RegType(xBaseIdx));
                }
                vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(yBaseIdx + i));
            }

            if (n_left) {
                // Use zero-masking (T_z) to zero unmasked elements
                vmovdqu8(Xbyak::Xmm(yBaseIdx + n_iter) | k1 | T_z,
                         ptr[regTmpYptr + n_iter * 16]);
                vpmovzxbd(RegType(yBaseIdx + n_iter),
                          Xbyak::Xmm(yBaseIdx + n_iter));

                if (!is_unit_beta) {
                    vpmulld(RegType(yBaseIdx + n_iter),
                            RegType(yBaseIdx + n_iter), RegType(xBaseIdx));
                }
                vpaddd(RegType(accumBaseIdx + n_iter),
                       RegType(accumBaseIdx + n_iter),
                       RegType(yBaseIdx + n_iter));
            }
        }
    } else if (c_downscale == DLP_BF16) {
        // Load buffer pointers for the downscaled output
        updateYBufferPointers();

        if (!nMask) {
            int nIter = NR / vnniWidth;
            for (int i = 0; i < nIter; i++) {
                vmovdqu16(Xbyak::Ymm(yBaseIdx + i), ptr[regTmpYptr + i * 32]);
                vpmovsxwd(RegType(yBaseIdx + i), Xbyak::Ymm(yBaseIdx + i));
                vpslld(RegType(yBaseIdx + i), RegType(yBaseIdx + i), 16);
                vcvtps2dq(RegType(yBaseIdx + i), RegType(yBaseIdx + i));

                if (!is_unit_beta) {
                    vpmulld(RegType(yBaseIdx + i), RegType(yBaseIdx + i),
                            RegType(xBaseIdx));
                }
                vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(yBaseIdx + i));
            }
        } else {
            int n_iter = N_LEFT / vnniWidth;
            int n_left = N_LEFT % vnniWidth;

            for (int i = 0; i < n_iter; i++) {
                vmovdqu16(Xbyak::Ymm(yBaseIdx + i), ptr[regTmpYptr + i * 32]);
                vpmovsxwd(RegType(yBaseIdx + i), Xbyak::Ymm(yBaseIdx + i));
                vpslld(RegType(yBaseIdx + i), RegType(yBaseIdx + i), 16);
                vcvtps2dq(RegType(yBaseIdx + i), RegType(yBaseIdx + i));

                if (!is_unit_beta) {
                    vpmulld(RegType(yBaseIdx + i), RegType(yBaseIdx + i),
                            RegType(xBaseIdx));
                }
                vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(yBaseIdx + i));
            }

            if (n_left) {
                // Use zero-masking (T_z) to zero unmasked elements
                vmovdqu16(Xbyak::Ymm(yBaseIdx + n_iter) | k1 | T_z,
                          ptr[regTmpYptr + n_iter * 32]);
                vpmovsxwd(RegType(yBaseIdx + n_iter),
                          Xbyak::Ymm(yBaseIdx + n_iter));
                vpslld(RegType(yBaseIdx + n_iter), RegType(yBaseIdx + n_iter),
                       16);
                vcvtps2dq(RegType(yBaseIdx + n_iter),
                          RegType(yBaseIdx + n_iter));

                if (!is_unit_beta) {
                    vpmulld(RegType(yBaseIdx + n_iter),
                            RegType(yBaseIdx + n_iter), RegType(xBaseIdx));
                }
                vpaddd(RegType(accumBaseIdx + n_iter),
                       RegType(accumBaseIdx + n_iter),
                       RegType(yBaseIdx + n_iter));
            }
        }
    } else if (c_downscale == DLP_F32) {
        // Load buffer pointers for the downscaled output
        updateYBufferPointers();

        if (!nMask) {
            int nIter = NR / vnniWidth;
            for (int i = 0; i < nIter; i++) {
                vcvtps2dq(RegType(yBaseIdx + i),
                          ptr[regTmpYptr + i * RegBytes]);

                if (!is_unit_beta) {
                    vpmulld(RegType(yBaseIdx + i), RegType(yBaseIdx + i),
                            RegType(xBaseIdx));
                }
                vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(yBaseIdx + i));
            }
        } else {
            int n_iter = N_LEFT / vnniWidth;
            int n_left = N_LEFT % vnniWidth;

            for (int i = 0; i < n_iter; i++) {
                vcvtps2dq(RegType(yBaseIdx + i),
                          ptr[regTmpYptr + i * RegBytes]);

                if (!is_unit_beta) {
                    vpmulld(RegType(yBaseIdx + i), RegType(yBaseIdx + i),
                            RegType(xBaseIdx));
                }
                vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                       RegType(yBaseIdx + i));
            }

            if (n_left) {
                // Use zero-masking (T_z) to zero unmasked elements
                vcvtps2dq(RegType(yBaseIdx + n_iter) | k1 | T_z,
                          ptr[regTmpYptr + n_iter * RegBytes]);

                if (!is_unit_beta) {
                    vpmulld(RegType(yBaseIdx + n_iter),
                            RegType(yBaseIdx + n_iter), RegType(xBaseIdx));
                }
                vpaddd(RegType(accumBaseIdx + n_iter),
                       RegType(accumBaseIdx + n_iter),
                       RegType(yBaseIdx + n_iter));
            }
        }
    } else { // c_downscale == DLP_S32
        mov(regTmpYptr, regYptr);

        if (!nMask) {
            int nIter = NR / vnniWidth;
            if (!is_unit_beta) {
                for (int i = 0; i < nIter; i++) {
                    vpmulld(RegType(yBaseIdx + i), RegType(xBaseIdx),
                            ptr[regTmpYptr + i * vnniWidth * sizeof(int32_t)]);
                    vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                           RegType(yBaseIdx + i));
                }
            } else {
                for (int i = 0; i < nIter; i++) {
                    vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                           ptr[regTmpYptr + i * vnniWidth * sizeof(int32_t)]);
                }
            }
        } else {
            int n_iter = N_LEFT / vnniWidth;
            int n_left = N_LEFT % vnniWidth;

            if (!is_unit_beta) {
                for (int i = 0; i < n_iter; i++) {
                    vpmulld(RegType(yBaseIdx + i), RegType(xBaseIdx),
                            ptr[regTmpYptr + i * vnniWidth * sizeof(int32_t)]);
                    vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                           RegType(yBaseIdx + i));
                }
                if (n_left) {
                    // Use zero-masking (T_z) to zero unmasked result lanes
                    vpmulld(
                        RegType(yBaseIdx + n_iter) | k1 | T_z,
                        RegType(xBaseIdx),
                        ptr[regTmpYptr + n_iter * vnniWidth * sizeof(int32_t)]);
                    vpaddd(RegType(accumBaseIdx + n_iter),
                           RegType(accumBaseIdx + n_iter),
                           RegType(yBaseIdx + n_iter));
                }
            } else {
                for (int i = 0; i < n_iter; i++) {
                    vpaddd(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i),
                           ptr[regTmpYptr + i * vnniWidth * sizeof(int32_t)]);
                }
                if (n_left) {
                    vpaddd(
                        RegType(accumBaseIdx + n_iter) | k1,
                        RegType(accumBaseIdx + n_iter),
                        ptr[regTmpYptr + n_iter * vnniWidth * sizeof(int32_t)]);
                }
            }
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8M1<KType>::storeY(bool nMask, bool hasPostOps)
{
    if (c_downscale == DLP_S8) {
        // Load buffer pointers for the downscaled output
        updateYBufferPointers();

        if (!nMask) {
            int nIter = NR / vnniWidth;
            for (int i = 0; i < nIter; i++) {
                if (hasPostOps) {
                    // Convert post-ops accumulated result from F32 to S32.
                    vcvtps2dq(RegType(accumBaseIdx + i),
                              RegType(accumBaseIdx + i));
                }
                vpmovsdb(ptr[regTmpYptr + i * 16], RegType(accumBaseIdx + i));
            }
        } else {
            int n_iter = N_LEFT / vnniWidth;
            int n_left = N_LEFT % vnniWidth;

            for (int i = 0; i < n_iter; i++) {
                if (hasPostOps) {
                    // Convert post-ops accumulated result from F32 to S32.
                    vcvtps2dq(RegType(accumBaseIdx + i),
                              RegType(accumBaseIdx + i));
                }
                vpmovsdb(ptr[regTmpYptr + i * 16], RegType(accumBaseIdx + i));
            }

            if (n_left) {
                if (hasPostOps) {
                    // Convert post-ops accumulated result from F32 to S32.
                    vcvtps2dq(RegType(accumBaseIdx + n_iter),
                              RegType(accumBaseIdx + n_iter));
                }
                vpmovsdb(ptr[regTmpYptr + n_iter * 16] | k1 | T_z,
                         RegType(accumBaseIdx + n_iter));
            }
        }
    } else if (c_downscale == DLP_U8) {
        // Load buffer pointers for the downscaled output
        updateYBufferPointers();

        vpxord(RegType(xBaseIdx + 1), RegType(xBaseIdx + 1),
               RegType(xBaseIdx + 1)); // 0
        mov(regKSubIter, 255);
        vpbroadcastd(RegType(xBaseIdx + 2), regKSubIter.cvt32()); // 255

        if (!nMask) {
            int nIter = NR / vnniWidth;
            for (int i = 0; i < nIter; i++) {
                if (hasPostOps) {
                    // Convert post-ops accumulated result from F32 to S32.
                    vcvtps2dq(RegType(accumBaseIdx + i),
                              RegType(accumBaseIdx + i));
                }
                // Clamp S32 to [0, 255] then pack to U8
                vpmaxsd(RegType(xBaseIdx), RegType(accumBaseIdx + i),
                        RegType(xBaseIdx + 1));
                vpminsd(RegType(xBaseIdx), RegType(xBaseIdx),
                        RegType(xBaseIdx + 2));
                vpmovdb(ptr[regTmpYptr + i * 16], RegType(xBaseIdx));
            }
        } else {
            vpxord(RegType(xBaseIdx + 1), RegType(xBaseIdx + 1),
                   RegType(xBaseIdx + 1)); // 0
            mov(regKSubIter, 255);
            vpbroadcastd(RegType(xBaseIdx + 2), regKSubIter.cvt32()); // 255

            int n_iter = N_LEFT / vnniWidth;
            int n_left = N_LEFT % vnniWidth;

            for (int i = 0; i < n_iter; i++) {
                if (hasPostOps) {
                    // Convert post-ops accumulated result from F32 to S32.
                    vcvtps2dq(RegType(accumBaseIdx + i),
                              RegType(accumBaseIdx + i));
                }
                vpmaxsd(RegType(xBaseIdx), RegType(accumBaseIdx + i),
                        RegType(xBaseIdx + 1));
                vpminsd(RegType(xBaseIdx), RegType(xBaseIdx),
                        RegType(xBaseIdx + 2));
                vpmovdb(ptr[regTmpYptr + i * 16], RegType(xBaseIdx));
            }

            if (n_left) {
                if (hasPostOps) {
                    // Convert post-ops accumulated result from F32 to S32.
                    vcvtps2dq(RegType(accumBaseIdx + n_iter),
                              RegType(accumBaseIdx + n_iter));
                }
                vpmaxsd(RegType(xBaseIdx), RegType(accumBaseIdx + n_iter),
                        RegType(xBaseIdx + 1));
                vpminsd(RegType(xBaseIdx), RegType(xBaseIdx),
                        RegType(xBaseIdx + 2));
                vpmovdb(ptr[regTmpYptr + n_iter * 16] | k1 | T_z,
                        RegType(xBaseIdx));
            }
        }
    } else if (c_downscale == DLP_BF16) {
        // Load buffer pointers for the downscaled output
        updateYBufferPointers();

        if (!nMask) {
            int nIter = NR / vnniWidth;
            for (int i = 0; i < nIter; i++) {
                if (!hasPostOps) {
                    // Convert intermediate result from S32 to F32.
                    vcvtdq2ps(RegType(accumBaseIdx + i),
                              RegType(accumBaseIdx + i));
                }
                // Convert from F32 to BF16.
                vpsrld(RegType(xBaseIdx), RegType(accumBaseIdx + i), 16);
                mov(regTmp1, 0x00000001);
                vpbroadcastd(RegType(xBaseIdx + 1), regTmp1.cvt32());
                vpandd(RegType(xBaseIdx), RegType(xBaseIdx),
                       RegType(xBaseIdx + 1));
                mov(regTmp1, 0x00007FFF);
                vpbroadcastd(RegType(xBaseIdx + 1), regTmp1.cvt32());
                vpaddd(RegType(xBaseIdx + 2), RegType(accumBaseIdx + i),
                       RegType(xBaseIdx + 1));
                vpaddd(RegType(xBaseIdx + 2), RegType(xBaseIdx + 2),
                       RegType(xBaseIdx));
                vpsrld(RegType(xBaseIdx + 2), RegType(xBaseIdx + 2), 16);
                vpmovdw(Xbyak::Ymm(xBaseIdx + 2), RegType(xBaseIdx + 2));
                vmovdqu16(ptr[regTmpYptr + i * 32], Xbyak::Ymm(xBaseIdx + 2));
            }
        } else {
            int n_iter = N_LEFT / vnniWidth;
            int n_left = N_LEFT % vnniWidth;

            for (int i = 0; i < n_iter; i++) {
                if (!hasPostOps) {
                    // Convert intermediate result from S32 to F32.
                    vcvtdq2ps(RegType(accumBaseIdx + i),
                              RegType(accumBaseIdx + i));
                }
                // Convert from F32 to BF16.
                vpsrld(RegType(xBaseIdx), RegType(accumBaseIdx + i), 16);
                mov(regTmp1, 0x00000001);
                vpbroadcastd(RegType(xBaseIdx + 1), regTmp1.cvt32());
                vpandd(RegType(xBaseIdx), RegType(xBaseIdx),
                       RegType(xBaseIdx + 1));
                mov(regTmp1, 0x00007FFF);
                vpbroadcastd(RegType(xBaseIdx + 1), regTmp1.cvt32());
                vpaddd(RegType(xBaseIdx + 2), RegType(accumBaseIdx + i),
                       RegType(xBaseIdx + 1));
                vpaddd(RegType(xBaseIdx + 2), RegType(xBaseIdx + 2),
                       RegType(xBaseIdx));
                vpsrld(RegType(xBaseIdx + 2), RegType(xBaseIdx + 2), 16);
                vpmovdw(Xbyak::Ymm(xBaseIdx + 2), RegType(xBaseIdx + 2));
                vmovdqu16(ptr[regTmpYptr + i * 32], Xbyak::Ymm(xBaseIdx + 2));
            }

            if (n_left) {
                if (!hasPostOps) {
                    // Convert intermediate result from S32 to F32.
                    vcvtdq2ps(RegType(accumBaseIdx + n_iter),
                              RegType(accumBaseIdx + n_iter));
                }
                vpsrld(RegType(xBaseIdx), RegType(accumBaseIdx + n_iter), 16);
                mov(regTmp1, 0x00000001);
                vpbroadcastd(RegType(xBaseIdx + 1), regTmp1.cvt32());
                vpandd(RegType(xBaseIdx), RegType(xBaseIdx),
                       RegType(xBaseIdx + 1));
                mov(regTmp1, 0x00007FFF);
                vpbroadcastd(RegType(xBaseIdx + 1), regTmp1.cvt32());
                vpaddd(RegType(xBaseIdx + 2), RegType(accumBaseIdx + n_iter),
                       RegType(xBaseIdx + 1));
                vpaddd(RegType(xBaseIdx + 2), RegType(xBaseIdx + 2),
                       RegType(xBaseIdx));
                vpsrld(RegType(xBaseIdx + 2), RegType(xBaseIdx + 2), 16);
                vpmovdw(Xbyak::Ymm(xBaseIdx + 2), RegType(xBaseIdx + 2));
                vmovdqu16(ptr[regTmpYptr + n_iter * 32] | k1 | T_z,
                          Xbyak::Ymm(xBaseIdx + 2));
            }
        }
    } else if (c_downscale == DLP_F32) {
        // Load buffer pointers for the downscaled output
        updateYBufferPointers();

        if (!nMask) {
            int nIter = NR / vnniWidth;
            for (int i = 0; i < nIter; i++) {
                if (!hasPostOps) {
                    // Convert intermediate result from S32 to F32.
                    vcvtdq2ps(RegType(accumBaseIdx + i),
                              RegType(accumBaseIdx + i));
                }
                vmovups(ptr[regTmpYptr + i * RegBytes],
                        RegType(accumBaseIdx + i));
            }
        } else {
            int n_iter = N_LEFT / vnniWidth;
            int n_left = N_LEFT % vnniWidth;

            for (int i = 0; i < n_iter; i++) {
                if (!hasPostOps) {
                    // Convert intermediate result from S32 to F32.
                    vcvtdq2ps(RegType(accumBaseIdx + i),
                              RegType(accumBaseIdx + i));
                }
                vmovups(ptr[regTmpYptr + i * RegBytes],
                        RegType(accumBaseIdx + i));
            }

            if (n_left) {
                if (!hasPostOps) {
                    // Convert intermediate result from S32 to F32.
                    vcvtdq2ps(RegType(accumBaseIdx + n_iter),
                              RegType(accumBaseIdx + n_iter));
                }
                vmovups(ptr[regTmpYptr + n_iter * RegBytes] | k1 | T_z,
                        RegType(accumBaseIdx + n_iter));
            }
        }
    } else { // c_downscale == DLP_S32
        mov(regTmpYptr, regYptr);

        if (!nMask) {
            int nIter = NR / vnniWidth;
            for (int i = 0; i < nIter; ++i) {
                if (hasPostOps) {
                    // Convert post-ops accumulated result from F32 to S32.
                    vcvtps2dq(RegType(accumBaseIdx + i),
                              RegType(accumBaseIdx + i));
                }
                vmovdqu32(ptr[regTmpYptr + i * vnniWidth * sizeof(int32_t)],
                          RegType(accumBaseIdx + i));
            }
        } else {
            int n_iter = N_LEFT / vnniWidth;
            int n_left = N_LEFT % vnniWidth;

            for (int i = 0; i < n_iter; i++) {
                if (hasPostOps) {
                    // Convert post-ops accumulated result from F32 to S32.
                    vcvtps2dq(RegType(accumBaseIdx + i),
                              RegType(accumBaseIdx + i));
                }
                vmovdqu32(ptr[regTmpYptr + i * vnniWidth * sizeof(int32_t)],
                          RegType(accumBaseIdx + i));
            }
            if (n_left) {
                if (hasPostOps) {
                    // Convert post-ops accumulated result from F32 to S32.
                    vcvtps2dq(RegType(accumBaseIdx + n_iter),
                              RegType(accumBaseIdx + n_iter));
                }
                vmovdqu32(ptr[regTmpYptr + n_iter * vnniWidth * sizeof(int32_t)]
                              | k1 | T_z,
                          RegType(accumBaseIdx + n_iter));
            }
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMVS8M1<KType>::generateKernel(utils::gemvM1GeneratorParams& params)
{
    Xbyak::util::StackFrame frame(this, 1, 13, 0);
    initializeStackFrame(frame);
    initializeParameters(params);

    RETURN_IF_ERROR(allocateRegisters());

    inLocalLabel();

    vxorps(RegType(vec128BaseIdx), RegType(vec128BaseIdx),
           RegType(vec128BaseIdx));
    mov(regTmp2, 128);
    vpbroadcastb(RegType(vec128BaseIdx), regTmp2.cvt8());

    mov(regYptr, ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, y)]);
    xor_(regIncN, regIncN); // regIncN is used to increment the pointer for N
                            // dimension(zeroed before the nloop)

    // Load kLeftmask into k2 before n-loop processing
    kmovw(k2, ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kLeftmask)]);

    if (params.nloop) {
        mov(regNIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, n_iter)]);
        test(regNIter, regNIter);
        jz(label_n_loop_end, T_NEAR);
        L(label_n_loop_start);

        regInit(accumBaseIdx, accumReg); // zero out accumulator registers
        xor_(regIncK, regIncK); // regIncK is used to increment the pointer for
                                // K dimensions

        // Skip k-loop if alpha is zero.
        if (params.alphaScalingType != dlp::kernel_frame::scalingType::zero) {
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
                    ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, b)]);

                mov(regPsB, KC);
                mov(regTmpYptr, ptr[stackPtr
                                    + offsetof(dlp::kernels::gemvM1Params,
                                               jc_cur_loop_rem)]);
                mov(regTmp2,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvM1Params, n_sub_updated)]);
                imul(regTmpYptr, regPsB);
                imul(regTmp2, regIncK);

                lea(regBptr, ptr[regBptr + regTmpYptr]);
                lea(regBptr, ptr[regBptr + regTmp2]);

                mov(regTmp2, regIncN);
                imul(regTmp2, regPsB);
                add(regBptr, regTmp2);

                kLoop(false, false);

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

                mov(regPsB,
                    ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, psB)]);
                mov(regTmpYptr, ptr[stackPtr
                                    + offsetof(dlp::kernels::gemvM1Params,
                                               jc_cur_loop_rem)]);
                mov(regTmp2,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvM1Params, n_sub_updated)]);
                imul(regTmpYptr, regPsB);
                imul(regTmp2, regIncK);

                lea(regBptr, ptr[regBptr + regTmpYptr]);
                lea(regBptr, ptr[regBptr + regTmp2]);

                mov(regTmp2, regIncN);
                imul(regTmp2, regPsB);

                add(regBptr, regTmp2);

                kLoop(true, false);
            }

            L(label_n_loop_k_fringe_end);

            accumulateKSubIters();

            conversionCompensation();

            scaleWithAlpha();
        }

        // Scale result by beta
        scaleYWithBeta(false);

        // Create and set up kernelOphandler if there are post-ops
        if (!params.kernelOps.empty()) {
            // Convert to F32 since post-ops expect accumulators in F32.
            for (int i = 0; i < (NR / vnniWidth); ++i) {
                vcvtdq2ps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i));
            }

            gen::kernelOpsHandler kernelOpsHandler(this, params.kType);

            int MR          = 1;
            int numMaskRegs = 1;
            int numCRegs    = (NR / vnniWidth);

            RETURN_IF_ERROR(kernelOpsHandler.generateKernelOps(
                params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemv_m1, MR,
                NR, false, numMaskRegs, accumBaseIdx, numCRegs));

            kernelOpsHandler.generateKernelOpsAttributes();

            // store C assuming F32 accumulators after post-ops
            RETURN_IF_ERROR(storeY(false, true));
        } else {
            // Store Result
            RETURN_IF_ERROR(storeY(false, false));
        }

        // Update b_sum_offset and store to memory
        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, b_sum_offset)]);
        add(regTmp2, NR);
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, b_sum_offset)],
            regTmp2);

        // Update pointers
        mov(regTmp2, NR);
        add(regIncN, regTmp2);
        lea(regYptr, ptr[regYptr + regTmp2 * sizeof(int32_t)]);

        dec(regNIter);
        jnz(label_n_loop_start, T_NEAR);
    }

    L(label_n_loop_end);
    if (params.nfringe) {
        kmovw(
            k1,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, nmask_avx512)]);

        mov(regNIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, n_left)]);
        test(regNIter, regNIter);
        jz(label_n_fringe_end, T_NEAR);

        // Adjust rs_b based on N_LEFT panels.
        if (N_LEFT < 32)
            mov(regRsB, 64);
        else if (N_LEFT < 48)
            mov(regRsB, 128);
        else if (N_LEFT < 64)
            mov(regRsB, 192);

        L(label_n_fringe_start);
        regInit(accumBaseIdx, accumReg);
        xor_(regIncK, regIncK);

        // Skip k-loop if alpha is zero.
        if (params.alphaScalingType != dlp::kernel_frame::scalingType::zero) {
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
                    ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, b)]);

                mov(regPsB, KC);
                mov(regTmpYptr, ptr[stackPtr
                                    + offsetof(dlp::kernels::gemvM1Params,
                                               jc_cur_loop_rem)]);
                mov(regTmp2,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvM1Params, n_sub_updated)]);

                imul(regTmpYptr, regPsB);
                imul(regTmp2, regIncK);

                lea(regBptr, ptr[regBptr + regTmpYptr]);
                lea(regBptr, ptr[regBptr + regTmp2]);

                mov(regTmp2, regIncN);
                imul(regTmp2, regPsB);

                add(regBptr, regTmp2);

                kLoop(false, true);

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

                mov(regPsB,
                    ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, psB)]);
                mov(regTmpYptr, ptr[stackPtr
                                    + offsetof(dlp::kernels::gemvM1Params,
                                               jc_cur_loop_rem)]);
                mov(regTmp2,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemvM1Params, n_sub_updated)]);
                imul(regTmpYptr, regPsB);
                imul(regTmp2, regIncK);

                lea(regBptr, ptr[regBptr + regTmpYptr]);
                lea(regBptr, ptr[regBptr + regTmp2]);

                mov(regTmp2, regIncN);
                imul(regTmp2, regPsB);

                add(regBptr, regTmp2);

                kLoop(true, true);
            }

            L(label_n_fringe_k_fringe_end);

            accumulateKSubIters();

            conversionCompensation(params.nfringe);

            // Scale result by alpha
            scaleWithAlpha();
        }

        // Scale result by beta
        scaleYWithBeta(true);

        // Create and set up kernelOphandler if there are post-ops
        if (!params.kernelOps.empty()) {
            // Convert to F32 since post-ops expect accumulators in F32.
            for (int i = 0; i < ((N_LEFT + vnniWidth - 1) / vnniWidth); ++i) {
                vcvtdq2ps(RegType(accumBaseIdx + i), RegType(accumBaseIdx + i));
            }

            gen::kernelOpsHandler kernelOpsHandler(this, params.kType);

            int MR          = 1;
            int numMaskRegs = 1;
            int numCRegs    = (N_LEFT / vnniWidth);

            RETURN_IF_ERROR(kernelOpsHandler.generateKernelOps(
                params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemv_m1, MR,
                N_LEFT, true, numMaskRegs, accumBaseIdx, numCRegs));

            kernelOpsHandler.generateKernelOpsAttributes();

            // store C assuming F32 accumulators after post-ops
            RETURN_IF_ERROR(storeY(true, true));
        } else {
            // Store Result
            RETURN_IF_ERROR(storeY(true, false));
        }

        // Update b_sum_offset and store to memory
        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, b_sum_offset)]);
        add(regTmp2, NR);
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, b_sum_offset)],
            regTmp2);
    }

    L(label_n_fringe_end);
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}
// End S8 GEMV M=1 JIT

} // namespace amdzen::gen

// Explicit template instantiations for classes
template class amdzen::gen::jitGEMVS8N1<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
template class amdzen::gen::jitGEMVS8M1<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
