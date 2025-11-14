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

#include "u8s8_gemv.hh"

namespace amdzen::gen {

using namespace Xbyak;

template<utils::kernelInstrType KType>
jitU8S8VNNI_GEMVN1<KType>::jitU8S8VNNI_GEMVN1(void* buffer, size_t bufferSize)
    : Xbyak::CodeGenerator(bufferSize, buffer)
{
    // Constructor body is empty - all initialization happens in member
    // initializer list Xbyak::CodeGenerator handles the JIT buffer setup for
    // code generation
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::loadMasks()
{
    // Ensuring mapping only from mask_regs[0] to k7(to avoid k0 usage
    // internally)
    for (int i = 0; i < NUM_USABLE_MASKS; i++) {
        mask_regs[i] = Xbyak::Opmask(MASK_START_IDX + i);
    }

    // Loading the 32 bit k mask(bf16), 16 bit m mask(f32)
    kmovq(
        mask_regs[0],
        ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kmask_i8_avx512)]);
    kmovd(mask_regs[1],
          ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, mmask_avx512)]);

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitU8S8VNNI_GEMVN1<KType>::regInit(int baseIdx, int numRegs)
{
    // Zero out vector registers using vpxord (XOR with itself = 0)
    vpxord(Zmm(baseIdx), Zmm(baseIdx), Zmm(baseIdx));
    for (int i = 1; i < numRegs; i++) {
        vmovdqa32(Zmm(baseIdx + i), Zmm(baseIdx));
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::allocateRegisters()
{
    // Validate MR parameter
    if (MR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Calculate SIMD width for int32 accumulators
    nElemsPerReg = RegBytes / sizeof(int32_t); // ZMM: 64/4 = 16 int32 elements

    // Register allocation strategy for GEMV N=1:
    // 1. Accumulation registers: MR registers (one per matrix row)
    accumReg     = MR;
    accumBaseIdx = numRegs - accumReg; // Start from end: [16, 17, 18, ...]

    // 2. Y vector registers: MR/nElemsPerReg registers for loading/storing Y
    yReg     = MR / nElemsPerReg;
    yBaseIdx = numRegs - yReg; // Start from the very end (zmm31)

    // 3. Temporary registers for A matrix loading and reduction operations
    tmpReg     = 4; // Standard number for A loading and reduction
    tmpBaseIdx = 0; // Start from beginning (zmm0-zmm3)

    // 4. X vector register: 1 register for broadcasting X elements
    xReg     = 1;
    xBaseIdx = tmpReg; // After temporary registers (zmm4)

    // 5. A matrix registers: NO dedicated registers - A values loaded into
    // tmpReg
    // aReg    = 0;          // A uses tmpReg space, no separate allocation
    // aRegIdx = tmpBaseIdx; // A loaded into tmpBaseIdx + offset

    // Validate register allocation (A reuses tmpReg space, so no double
    // counting)
    int totalRegsUsed = accumReg + yReg + xReg + tmpReg;
    if (totalRegsUsed > numRegs) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitU8S8VNNI_GEMVN1<KType>::initializeStackFrame(
    Xbyak::util::StackFrame& stackFrame)
{
    // Map preserved and transient registers similar to f32 GEMV implementation
    stackPtr   = stackFrame.p[0];
    regAptr    = stackFrame.t[0];
    regTmpAptr = stackFrame.t[1];
    regXptr    = stackFrame.t[2];
    regYptr    = stackFrame.t[3];
    regTmpYptr = stackFrame.t[4];
    regRsA     = stackFrame.t[5];
    regCsA     = stackFrame.t[6];
    regRsC     = stackFrame.t[7];
    regMIter   = stackFrame.t[8];
    regKIter   = stackFrame.t[9];
    regTmp1    = stackFrame.t[10];
    regTmp2    = stackFrame.t[11];
    regTmp3    = stackFrame.t[12];
}

template<utils::kernelInstrType KType>
void
jitU8S8VNNI_GEMVN1<KType>::initializeParameters()
{
    nElemsPerReg = RegBytes / sizeof(int32_t); // ZMM: 64/4 = 16 int32 elements

    // Load pointers and strides
    mov(regRsA, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, rsA)]);
    mov(regCsA, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, csA)]);
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, rsC)]);

    lea(regRsC, ptr[regRsC * sizeof(int32_t)]);

    // Load post_op_c_i for downscale buffer addressing (if using downscale)
    if (c_downscale != DLP_S32) {
        mov(regTmp2,
            ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
    }

    mov(regAptr, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, a)]);
    mov(regYptr, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, y)]);
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::scaleAlpha(int mSize)
{

    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, alpha)]);
    vpbroadcastd(Zmm(tmpBaseIdx), ptr[regKIter]);

    for (int i = 0; i < (mSize + nElemsPerReg - 1) / nElemsPerReg; i++) {
        vpmulld(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i), Zmm(tmpBaseIdx));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitU8S8VNNI_GEMVN1<KType>::updateCBufferPointers()
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
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::storeResult_S32(int mSize, bool isRowStored)
{
    int mLeft = mSize % nElemsPerReg;

    if (isRowStored) {
        RETURN_IF_ERROR(storeY_rowStored_S32(mSize));
    } else {
        // Column-major storage for S32
        for (int i = 0; i < mSize / nElemsPerReg; i += 1) {
            vmovdqu32(ptr[regTmpYptr], Zmm(accumBaseIdx + i));
        }
        if (mLeft) {
            vmovdqu32(ptr[regTmpYptr] | mask_regs[1] | T_z,
                      Zmm(accumBaseIdx + (mSize / nElemsPerReg)));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::storeResult_U8(int mSize, bool isRowStored)
{
    int mLeft = mSize % nElemsPerReg;

    if (isRowStored) {
        RETURN_IF_ERROR(storeY_rowStored_U8(mSize));
    } else {
        // Column-major storage for U8 with saturation
        updateCBufferPointers();

        vpxord(Zmm(tmpBaseIdx + 1), Zmm(tmpBaseIdx + 1),
               Zmm(tmpBaseIdx + 1)); // 0
        mov(regKIter, 255);
        vpbroadcastd(Zmm(tmpBaseIdx + 2), regKIter.cvt32());

        for (int i = 0; i < mSize / nElemsPerReg; i += 1) {
            vpmaxsd(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 1),
                    Zmm(accumBaseIdx + i));
            vpminsd(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 2), Zmm(tmpBaseIdx));
            vpmovdb(ptr[regTmpYptr], Zmm(tmpBaseIdx));
        }
        if (mLeft) {
            vpmaxsd(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 1),
                    Zmm(accumBaseIdx + (mSize / nElemsPerReg)));
            vpminsd(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 2), Zmm(tmpBaseIdx));
            vpmovdb(ptr[regTmpYptr] | mask_regs[1] | T_z, Zmm(tmpBaseIdx));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::storeResult_S8(int mSize, bool isRowStored)
{
    int mLeft = mSize % nElemsPerReg;

    if (isRowStored) {
        RETURN_IF_ERROR(storeY_rowStored_S8(mSize));
    } else {
        // Column-major storage for S8 with saturation
        updateCBufferPointers();

        for (int i = 0; i < mSize / nElemsPerReg; i += 1) {
            vpmovsdb(ptr[regTmpYptr], Zmm(accumBaseIdx + i));
        }
        if (mLeft) {
            vpmovsdb(ptr[regTmpYptr] | mask_regs[1] | T_z,
                     Zmm(accumBaseIdx + (mSize / nElemsPerReg)));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::storeResult_F32(int mSize, bool isRowStored)
{
    int mLeft = mSize % nElemsPerReg;

    if (isRowStored) {
        RETURN_IF_ERROR(storeY_rowStored_F32(mSize));
    } else {
        // Column-major storage for F32 with conversion
        updateCBufferPointers();

        for (int i = 0; i < mSize / nElemsPerReg; i += 1) {
            vcvtdq2ps(Zmm(tmpBaseIdx), Zmm(accumBaseIdx + i));
            vmovups(ptr[regTmpYptr], Zmm(tmpBaseIdx));
        }
        if (mLeft) {
            vcvtdq2ps(Zmm(tmpBaseIdx),
                      Zmm(accumBaseIdx + (mSize / nElemsPerReg)));
            vmovups(ptr[regTmpYptr] | mask_regs[1] | T_z, Zmm(tmpBaseIdx));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::storeResult_BF16(int mSize, bool isRowStored)
{
    int mLeft = mSize % nElemsPerReg;

    if (isRowStored) {
        RETURN_IF_ERROR(storeY_rowStored_BF16(mSize));
    } else {
        // Column-major storage for BF16 with conversion
        updateCBufferPointers();

        for (int i = 0; i < mSize / nElemsPerReg; i += 1) {
            vcvtdq2ps(Zmm(tmpBaseIdx), Zmm(accumBaseIdx + i));
            vcvtneps2bf16(Ymm(tmpBaseIdx), Zmm(tmpBaseIdx));
            vmovdqu16(ptr[regTmpYptr], Ymm(tmpBaseIdx));
        }

        if (mLeft) {
            vcvtdq2ps(Zmm(tmpBaseIdx),
                      Zmm(accumBaseIdx + (mSize / nElemsPerReg)));

            vcvtneps2bf16(Ymm(tmpBaseIdx), Zmm(tmpBaseIdx));
            vmovdqu16(ptr[regTmpYptr] | mask_regs[1] | T_z, Ymm(tmpBaseIdx));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::storeY_rowStored_S32(int mSize)
{
    // Process each ZMM register (which contains 16 elements) for S32 output
    for (int i = 0; i < (mSize + nElemsPerReg - 1) / nElemsPerReg; i++) {
        int elems_in_reg = (i < mSize / nElemsPerReg) ? nElemsPerReg
                                                      : (mSize % nElemsPerReg);
        if (elems_in_reg == 0)
            break;

        // Extract 4 chunks of 128-bits from the ZMM
        for (int j = 0; j < elems_in_reg; j += 4) {
            vextracti32x4(Xmm(tmpBaseIdx + j / 4), // ISA specific
                          Zmm(accumBaseIdx + i), j / 4);
        }

        for (int j = 0; j < elems_in_reg; j++) {
            int tmp_reg    = j / 4;
            int pos_in_reg = j % 4;

            if (pos_in_reg == 0) {
                // First element in XMM can be stored directly (32-bit)
                vmovd(ptr[regTmpYptr], Xbyak::Xmm(tmpBaseIdx + tmp_reg));
            } else {
                // Extract to memory directly (32-bit)
                vpextrd(ptr[regTmpYptr], Xbyak::Xmm(tmpBaseIdx + tmp_reg),
                        pos_in_reg);
            }

            // Move to next row (S32 stride)
            add(regTmpYptr, regRsC);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::storeY_rowStored_U8(int mSize)
{
    updateCBufferPointers();

    // Setup saturation constants for U8 (0-255)
    vpxord(Zmm(tmpBaseIdx + 1), Zmm(tmpBaseIdx + 1), Zmm(tmpBaseIdx + 1)); // 0
    mov(regKIter, 255);
    vpbroadcastd(Zmm(tmpBaseIdx + 2), regKIter.cvt32());

    // Process each ZMM register for U8 output
    for (int i = 0; i < (mSize + nElemsPerReg - 1) / nElemsPerReg; i++) {
        int elems_in_reg = (i < mSize / nElemsPerReg) ? nElemsPerReg
                                                      : (mSize % nElemsPerReg);
        if (elems_in_reg == 0)
            break;

        // Apply saturation to entire ZMM
        vpmaxsd(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 1), Zmm(accumBaseIdx + i));
        vpminsd(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 2), Zmm(tmpBaseIdx));
        vpmovdb(Xmm(tmpBaseIdx), Zmm(tmpBaseIdx));

        // Extract and store each element as U8
        for (int j = 0; j < elems_in_reg; j++) {
            vpextrb(ptr[regTmpYptr], Xmm(tmpBaseIdx), j);
            add(regTmpYptr, regTmp1); // Move to next row (U8 stride)
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::storeY_rowStored_S8(int mSize)
{
    updateCBufferPointers();

    // Process each ZMM register for S8 output
    for (int i = 0; i < (mSize + nElemsPerReg - 1) / nElemsPerReg; i++) {
        int elems_in_reg = (i < mSize / nElemsPerReg) ? nElemsPerReg
                                                      : (mSize % nElemsPerReg);
        if (elems_in_reg == 0)
            break;

        // Apply signed saturation and pack to bytes
        vpmovsdb(Xmm(tmpBaseIdx), Zmm(accumBaseIdx + i));

        // Extract and store each element as S8
        for (int j = 0; j < elems_in_reg; j++) {
            vpextrb(ptr[regTmpYptr], Xmm(tmpBaseIdx), j);
            add(regTmpYptr, regTmp1); // Move to next row (S8 stride)
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::storeY_rowStored_F32(int mSize)
{
    updateCBufferPointers();

    // Process each ZMM register (which contains 16 elements) for S32 output
    for (int i = 0; i < (mSize + nElemsPerReg - 1) / nElemsPerReg; i++) {
        int elems_in_reg = (i < mSize / nElemsPerReg) ? nElemsPerReg
                                                      : (mSize % nElemsPerReg);
        if (elems_in_reg == 0)
            break;

        vcvtdq2ps(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i));
        // Extract 4 chunks of 128-bits from the ZMM
        for (int j = 0; j < elems_in_reg; j += 4) {
            vextractf32x4(Xmm(tmpBaseIdx + j / 4), // ISA specific
                          Zmm(accumBaseIdx + i), j / 4);
        }

        for (int j = 0; j < elems_in_reg; j++) {
            int tmp_reg    = j / 4;
            int pos_in_reg = j % 4;

            if (pos_in_reg == 0) {
                // First element in XMM can be stored directly (32-bit)
                vmovss(ptr[regTmpYptr], Xbyak::Xmm(tmpBaseIdx + tmp_reg));
            } else {
                // Extract to memory directly (32-bit)
                vpextrd(ptr[regTmpYptr], Xbyak::Xmm(tmpBaseIdx + tmp_reg),
                        pos_in_reg);
            }

            // Move to next row (S32 stride)
            add(regTmpYptr, regTmp1);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::storeY_rowStored_BF16(int mSize)
{
    updateCBufferPointers();

    // Process each ZMM register for BF16 output
    for (int i = 0; i < (mSize + nElemsPerReg - 1) / nElemsPerReg; i++) {
        int elems_in_reg = (i < mSize / nElemsPerReg) ? nElemsPerReg
                                                      : (mSize % nElemsPerReg);
        if (elems_in_reg == 0)
            break;

        // Convert int32 to float32, then to BF16
        vcvtdq2ps(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i));
        vcvtneps2bf16(Ymm(tmpBaseIdx), Zmm(accumBaseIdx + i));

        // Extract 2 chunks of 128-bits from the YMM (each chunk has 8 BF16
        // elements)
        for (int j = 0; j < elems_in_reg; j += 8) {
            vextracti32x4(Xmm(tmpBaseIdx + 1 + j / 8), Ymm(tmpBaseIdx), j / 8);
        }

        for (int j = 0; j < elems_in_reg; j++) {
            int tmp_reg    = j / 8 + 1; // Each XMM holds 8 BF16 elements
            int pos_in_reg = j % 8;     // Position within XMM (0-7)

            if (pos_in_reg == 0) {
                // First element in XMM - store lower 16 bits
                vpextrw(ptr[regTmpYptr], Xbyak::Xmm(tmpBaseIdx + tmp_reg), 0);
            } else {
                // Extract to memory directly (16-bit)
                vpextrw(ptr[regTmpYptr], Xbyak::Xmm(tmpBaseIdx + tmp_reg),
                        pos_in_reg);
            }

            // Move to next row (BF16 stride)
            add(regTmpYptr, regTmp1);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::storeResult(int mSize)
{
    mov(regTmpYptr, regYptr);
    bool isRowStored = (yFormat == dlp::kernel_frame::storageFormat::rowMajor);

    if (c_downscale == DLP_S32) {
        RETURN_IF_ERROR(storeResult_S32(mSize, isRowStored));
    } else if (c_downscale == DLP_U8) {
        RETURN_IF_ERROR(storeResult_U8(mSize, isRowStored));
    } else if (c_downscale == DLP_S8) {
        RETURN_IF_ERROR(storeResult_S8(mSize, isRowStored));
    } else if (c_downscale == DLP_F32) {
        RETURN_IF_ERROR(storeResult_F32(mSize, isRowStored));
    } else if (c_downscale == DLP_BF16) {
        RETURN_IF_ERROR(storeResult_BF16(mSize, isRowStored));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::scaleYWithBeta_S32(int mSize, bool isRowStored)
{
    int mLeft = mSize % nElemsPerReg;

    if (isRowStored) {
        vpmulld(Zmm(tmpBaseIdx), Zmm(xBaseIdx), Zmm(yBaseIdx));
        vpaddd(Zmm(accumBaseIdx), Zmm(accumBaseIdx), Zmm(tmpBaseIdx));
    } else {
        for (int i = 0; i < mSize / nElemsPerReg; i++) {
            vpmulld(Zmm(yBaseIdx), Zmm(xBaseIdx), ptr[regTmpYptr]);
            vpaddd(Zmm(accumBaseIdx), Zmm(accumBaseIdx), Zmm(yBaseIdx));
        }

        if (mLeft) {
            vpmulld(Zmm(yBaseIdx) | mask_regs[1], Zmm(xBaseIdx),
                    ptr[regTmpYptr]);
            vpaddd(Zmm(accumBaseIdx + (mSize / nElemsPerReg)),
                   Zmm(accumBaseIdx + (mSize / nElemsPerReg)), Zmm(yBaseIdx));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::scaleYWithBeta_U8(int mSize, bool isRowStored)
{
    int mLeft = mSize % nElemsPerReg;

    if (isRowStored) {
        vpmovzxbd(Zmm(yBaseIdx), Xmm(yBaseIdx));
        vpdpbusd(Zmm(accumBaseIdx), Zmm(yBaseIdx), Zmm(xBaseIdx));
    } else {
        // For column-stored data, load from downscale buffer
        updateCBufferPointers();

        for (int i = 0; i < mSize / nElemsPerReg; i++) {
            vmovdqu8(Xmm(yBaseIdx), ptr[regTmpYptr]);
            vpmovzxbd(Zmm(yBaseIdx), Xmm(yBaseIdx));
            vpdpbusd(Zmm(accumBaseIdx + i), Zmm(yBaseIdx), Zmm(xBaseIdx));
        }

        if (mLeft) {
            vmovdqu8(Xmm(yBaseIdx) | mask_regs[1], ptr[regTmpYptr]);
            vpmovzxbd(Zmm(yBaseIdx), Xmm(yBaseIdx));
            vpdpbusd(Zmm(accumBaseIdx + (mSize / nElemsPerReg)), Zmm(yBaseIdx),
                     Zmm(xBaseIdx));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::scaleYWithBeta_S8(int mSize, bool isRowStored)
{
    int mLeft = mSize % nElemsPerReg;

    if (isRowStored) {
        vpmovsxbd(Zmm(yBaseIdx), Xmm(yBaseIdx));
        vpmulld(Zmm(yBaseIdx), Zmm(yBaseIdx), Zmm(xBaseIdx));
        vpaddd(Zmm(accumBaseIdx), Zmm(accumBaseIdx), Zmm(yBaseIdx));
    } else {
        updateCBufferPointers();
        for (int i = 0; i < mSize / nElemsPerReg; i++) {
            vmovdqu8(Xmm(yBaseIdx), ptr[regTmpYptr]);
            vpmovsxbd(Zmm(yBaseIdx), Xmm(yBaseIdx));
            vpmulld(Zmm(yBaseIdx), Zmm(yBaseIdx), Zmm(xBaseIdx));
            vpaddd(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i), Zmm(yBaseIdx));
        }
        if (mLeft) {
            vmovdqu8(Xmm(yBaseIdx) | mask_regs[1], ptr[regTmpYptr]);
            vpmovsxbd(Zmm(yBaseIdx), Xmm(yBaseIdx));
            vpmulld(Zmm(yBaseIdx), Zmm(yBaseIdx), Zmm(xBaseIdx));
            vpaddd(Zmm(accumBaseIdx + (mSize / nElemsPerReg)),
                   Zmm(accumBaseIdx + (mSize / nElemsPerReg)), Zmm(yBaseIdx));
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::scaleYWithBeta_F32(int mSize, bool isRowStored)
{
    int mLeft = mSize % nElemsPerReg;

    if (isRowStored) {
        vcvtps2dq(Zmm(yBaseIdx), Zmm(yBaseIdx));
        vpmulld(Zmm(yBaseIdx), Zmm(yBaseIdx), Zmm(xBaseIdx));
        vpaddd(Zmm(accumBaseIdx), Zmm(accumBaseIdx), Zmm(yBaseIdx));
    } else {
        updateCBufferPointers();
        for (int i = 0; i < mSize / nElemsPerReg; i++) {
            vcvtps2dq(Zmm(yBaseIdx), ptr[regTmpYptr]);
            vpmulld(Zmm(yBaseIdx), Zmm(yBaseIdx), Zmm(xBaseIdx));
            vpaddd(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i), Zmm(yBaseIdx));
        }
        if (mLeft) {
            vcvtps2dq(Zmm(yBaseIdx) | mask_regs[1], ptr[regTmpYptr]);
            vpmulld(Zmm(yBaseIdx), Zmm(yBaseIdx), Zmm(xBaseIdx));
            vpaddd(Zmm(accumBaseIdx + (mSize / nElemsPerReg)),
                   Zmm(accumBaseIdx + (mSize / nElemsPerReg)), Zmm(yBaseIdx));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::scaleYWithBeta_BF16(int mSize, bool isRowStored)
{
    int mLeft = mSize % nElemsPerReg;

    if (isRowStored) {
        vpmovsxwd(Zmm(yBaseIdx), Ymm(yBaseIdx));
        vpslld(Zmm(yBaseIdx), Zmm(yBaseIdx), 16);
        vcvtps2dq(Zmm(yBaseIdx), Zmm(yBaseIdx));
        vpmulld(Zmm(yBaseIdx), Zmm(yBaseIdx), Zmm(xBaseIdx));
        vpaddd(Zmm(accumBaseIdx), Zmm(accumBaseIdx), Zmm(yBaseIdx));

    } else {
        updateCBufferPointers();

        for (int i = 0; i < mSize / nElemsPerReg; i++) {
            vmovdqu16(Ymm(yBaseIdx), ptr[regTmpYptr]);
            vpmovsxwd(Zmm(yBaseIdx), Ymm(yBaseIdx));
            vpslld(Zmm(yBaseIdx), Zmm(yBaseIdx), 16);
            vcvtps2dq(Zmm(yBaseIdx), Zmm(yBaseIdx));
            vpmulld(Zmm(yBaseIdx), Zmm(yBaseIdx), Zmm(xBaseIdx));
            vpaddd(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i), Zmm(yBaseIdx));
        }

        if (mLeft) {
            vmovdqu16(Ymm(yBaseIdx) | mask_regs[1], ptr[regTmpYptr]);
            vpmovsxwd(Zmm(yBaseIdx), Ymm(yBaseIdx));
            vpslld(Zmm(yBaseIdx), Zmm(yBaseIdx), 16);
            vcvtps2dq(Zmm(yBaseIdx), Zmm(yBaseIdx));
            // Store the YMM result to debug buffer for inspection
            vpmulld(Zmm(yBaseIdx), Zmm(yBaseIdx), Zmm(xBaseIdx));
            vpaddd(Zmm(accumBaseIdx + (mSize / nElemsPerReg)),
                   Zmm(accumBaseIdx + (mSize / nElemsPerReg)), Zmm(yBaseIdx));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::rearrangeY_rowStored_S32(int mSize)
{
    lea(regTmp3, ptr[regRsC + 2 * regRsC]);

    for (int i = 0; i < (mSize + nElemsPerReg - 1) / nElemsPerReg; i++) {
        int blockSize = (mSize - nElemsPerReg * i) < nElemsPerReg
                            ? (mSize - nElemsPerReg * i)
                            : nElemsPerReg;
        int n_blocks  = blockSize / 4;
        int rem_elems = blockSize % 4;

        // Zero out the Y register before inserting values
        vpxord(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i), Zmm(yBaseIdx + i));

        for (int j = 0; j < n_blocks; j++) {
            vpbroadcastd(Zmm(tmpBaseIdx), ptr[regTmpYptr]);
            vpbroadcastd(Zmm(tmpBaseIdx + 1), ptr[regTmpYptr + regRsC]);
            vpbroadcastd(Zmm(tmpBaseIdx + 2), ptr[regTmpYptr + 2 * regRsC]);
            vpbroadcastd(Zmm(tmpBaseIdx + 3), ptr[regTmpYptr + regTmp3]);

            vpunpckldq(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 1));
            vpunpckldq(Zmm(tmpBaseIdx + 2), Zmm(tmpBaseIdx + 2),
                       Zmm(tmpBaseIdx + 3));

            vshufps(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 2),
                    0x44);

            vinserti32x4(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i), Xmm(tmpBaseIdx),
                         j);
            lea(regTmpYptr, ptr[regTmpYptr + regRsC * 4]);
        }
        if (rem_elems) {
            switch (rem_elems) {
                case 3:
                    vpbroadcastd(Zmm(tmpBaseIdx + 2),
                                 ptr[regTmpYptr + regRsC * 2]);
                case 2:
                    vpbroadcastd(Zmm(tmpBaseIdx + 1), ptr[regTmpYptr + regRsC]);
                case 1:
                    vpbroadcastd(Zmm(tmpBaseIdx), ptr[regTmpYptr]);
                case 0:
                    break;
            }

            vpunpckldq(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 1));
            vpunpckldq(Zmm(tmpBaseIdx + 2), Zmm(tmpBaseIdx + 2),
                       Zmm(tmpBaseIdx + 3));

            vshufps(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 2),
                    0x44);

            vinserti32x4(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i), Xmm(tmpBaseIdx),
                         n_blocks);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::rearrangeY_rowStored_U8(int mSize)
{
    updateCBufferPointers();

    // Initialize temporary registers
    // regInit(tmpBaseIdx, tmpReg);

    lea(regTmp3, ptr[regTmp1 + 2 * regTmp1]); // 3 * regTmp1 (stride)

    for (int i = 0; i < (mSize + 15) / 16; i++) {
        int blockSize = (mSize - nElemsPerReg * i) < nElemsPerReg
                            ? (mSize - nElemsPerReg * i)
                            : nElemsPerReg;
        int n_blocks  = blockSize / 4; // Process 4 U8 elements per block
        int rem_elems = blockSize % 4;

        // Zero out the result XMM register
        vpxord(Xmm(yBaseIdx + i), Xmm(yBaseIdx + i), Xmm(yBaseIdx + i));

        for (int j = 0; j < n_blocks; j++) {
            // Use vpbroadcastb pattern - broadcast each U8 value across XMM
            vpbroadcastb(Xmm(tmpBaseIdx), ptr[regTmpYptr]);
            vpbroadcastb(Xmm(tmpBaseIdx + 1), ptr[regTmpYptr + regTmp1]);
            vpbroadcastb(Xmm(tmpBaseIdx + 2), ptr[regTmpYptr + 2 * regTmp1]);
            vpbroadcastb(Xmm(tmpBaseIdx + 3), ptr[regTmpYptr + regTmp3]);

            // Interleave the broadcasted bytes
            vpunpcklbw(Xmm(tmpBaseIdx), Xmm(tmpBaseIdx), Xmm(tmpBaseIdx + 1));
            vpunpcklbw(Xmm(tmpBaseIdx + 2), Xmm(tmpBaseIdx + 2),
                       Xmm(tmpBaseIdx + 3));

            vpunpcklwd(Xmm(tmpBaseIdx), Xmm(tmpBaseIdx), Xmm(tmpBaseIdx + 2));

            vpextrd(regKIter.cvt32(), Xmm(tmpBaseIdx), 0);
            vpinsrd(Xmm(yBaseIdx + i), Xmm(yBaseIdx + i), regKIter.cvt32(), j);

            // Move to next 4 rows
            lea(regTmpYptr, ptr[regTmpYptr + regTmp1 * 4]);
        }

        // Handle remaining elements (less than 4)
        if (rem_elems) {

            regInit(tmpBaseIdx, tmpReg);

            switch (rem_elems) {
                case 3:
                    vpbroadcastb(Xmm(tmpBaseIdx + 2),
                                 ptr[regTmpYptr + 2 * regTmp1]);
                case 2:
                    vpbroadcastb(Xmm(tmpBaseIdx + 1),
                                 ptr[regTmpYptr + regTmp1]);
                case 1:
                    vpbroadcastb(Xmm(tmpBaseIdx), ptr[regTmpYptr]);
                case 0:
                    break;
            }

            // Interleave and pack the remaining elements
            vpunpcklbw(Xmm(tmpBaseIdx), Xmm(tmpBaseIdx), Xmm(tmpBaseIdx + 1));
            vpunpcklbw(Xmm(tmpBaseIdx + 2), Xmm(tmpBaseIdx + 2),
                       Xmm(tmpBaseIdx + 3));
            vpunpcklwd(Xmm(tmpBaseIdx), Xmm(tmpBaseIdx), Xmm(tmpBaseIdx + 2));

            // Insert the remaining elements
            vpextrd(regKIter.cvt32(), Xmm(tmpBaseIdx), 0);
            vpinsrd(Xmm(yBaseIdx + i), Xmm(yBaseIdx + i), regKIter.cvt32(),
                    n_blocks);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::rearrangeY_rowStored_S8(int mSize)
{
    // Same logic as U8 since both are 8-bit data types
    return rearrangeY_rowStored_U8(mSize);
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::rearrangeY_rowStored_F32(int mSize)
{
    updateCBufferPointers();

    lea(regTmp3, ptr[regTmp1 + 2 * regTmp1]);

    for (int i = 0; i < (mSize + nElemsPerReg - 1) / nElemsPerReg; i++) {
        int blockSize = (mSize - nElemsPerReg * i) < nElemsPerReg
                            ? (mSize - nElemsPerReg * i)
                            : nElemsPerReg;
        int n_blocks  = blockSize / 4;
        int rem_elems = blockSize % 4;

        // Zero out the Y register before inserting values
        vpxord(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i), Zmm(yBaseIdx + i));

        for (int j = 0; j < n_blocks; j++) {
            vbroadcastss(Zmm(tmpBaseIdx), ptr[regTmpYptr]);
            vbroadcastss(Zmm(tmpBaseIdx + 1), ptr[regTmpYptr + regTmp1]);
            vbroadcastss(Zmm(tmpBaseIdx + 2), ptr[regTmpYptr + 2 * regTmp1]);
            vbroadcastss(Zmm(tmpBaseIdx + 3), ptr[regTmpYptr + regTmp3]);

            vunpcklps(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 1));
            vunpcklps(Zmm(tmpBaseIdx + 2), Zmm(tmpBaseIdx + 2),
                      Zmm(tmpBaseIdx + 3));

            vshufps(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 2),
                    0x44);

            vinsertf32x4(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i), Xmm(tmpBaseIdx),
                         j);
            lea(regTmpYptr, ptr[regTmpYptr + regTmp1 * 4]);
        }

        if (rem_elems) {

            regInit(tmpBaseIdx, tmpReg);

            switch (rem_elems) {
                case 3:
                    vbroadcastss(Zmm(tmpBaseIdx + 2),
                                 ptr[regTmpYptr + regTmp1 * 2]);
                case 2:
                    vbroadcastss(Zmm(tmpBaseIdx + 1),
                                 ptr[regTmpYptr + regTmp1]);
                case 1:
                    vbroadcastss(Zmm(tmpBaseIdx), ptr[regTmpYptr]);
                case 0:
                    break;
            }

            vunpcklps(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 1));
            vunpcklps(Zmm(tmpBaseIdx + 2), Zmm(tmpBaseIdx + 2),
                      Zmm(tmpBaseIdx + 3));

            vshufps(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 2),
                    0x44);

            vinsertf32x4(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i), Xmm(tmpBaseIdx),
                         n_blocks);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::rearrangeY_rowStored_BF16(int mSize)
{
    updateCBufferPointers();

    for (int i = 0; i < (mSize + nElemsPerReg - 1) / nElemsPerReg; i++) {
        int blockSize = (mSize - nElemsPerReg * i) < nElemsPerReg
                            ? (mSize - nElemsPerReg * i)
                            : nElemsPerReg;

        // Zero out the two XMM registers
        vpxord(Xmm(tmpBaseIdx), Xmm(tmpBaseIdx),
               Xmm(tmpBaseIdx)); // First 8 BF16 values
        vpxord(Xmm(tmpBaseIdx + 1), Xmm(tmpBaseIdx + 1),
               Xmm(tmpBaseIdx + 1)); // Second 8 BF16 values

        // Load first 8 BF16 values into Xmm(tmpBaseIdx)
        for (int j = 0; j < 8 && j < blockSize; j++) {
            // Load BF16 value from current row
            movsx(regKIter.cvt32(), word[regTmpYptr]);

            // Insert the 16-bit BF16 value into the j-th position of first XMM
            vpinsrw(Xmm(tmpBaseIdx), Xmm(tmpBaseIdx), regKIter.cvt32(), j);

            // Move to next row
            add(regTmpYptr, regTmp1);
        }

        for (int j = 8; j < blockSize; j++) {
            // Load BF16 value from current row
            movsx(regKIter.cvt32(), word[regTmpYptr]);

            // Insert the 16-bit BF16 value into the (j-8)-th position of second
            // XMM
            vpinsrw(Xmm(tmpBaseIdx + 1), Xmm(tmpBaseIdx + 1), regKIter.cvt32(),
                    j - 8);

            // Move to next row
            add(regTmpYptr, regTmp1);
        }

        // Combine the two XMM registers into YMM
        // Xmm(tmpBaseIdx) contains BF16 values [0-7]
        // Xmm(tmpBaseIdx + 1) contains BF16 values [8-15]
        vinserti32x4(Ymm(yBaseIdx + i), Ymm(yBaseIdx + i), Xmm(tmpBaseIdx),
                     0); // Lower 128 bits
        vinserti32x4(Ymm(yBaseIdx + i), Ymm(yBaseIdx + i), Xmm(tmpBaseIdx + 1),
                     1); // Upper 128 bits
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::scaleBeta(int mSize)
{

    mov(regTmpYptr, regYptr);

    if (yFormat == dlp::kernel_frame::storageFormat::rowMajor) {
        // Use data-type specific rearrange function
        if (c_downscale == DLP_S32) {
            RETURN_IF_ERROR(rearrangeY_rowStored_S32(mSize));
        } else if (c_downscale == DLP_U8) {
            RETURN_IF_ERROR(rearrangeY_rowStored_U8(mSize));
        } else if (c_downscale == DLP_S8) {
            RETURN_IF_ERROR(rearrangeY_rowStored_S8(mSize));
        } else if (c_downscale == DLP_F32) {
            RETURN_IF_ERROR(rearrangeY_rowStored_F32(mSize));
        } else if (c_downscale == DLP_BF16) {
            RETURN_IF_ERROR(rearrangeY_rowStored_BF16(mSize));
        }
    }

    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, beta)]);
    vpbroadcastd(Zmm(xBaseIdx),
                 ptr[regKIter]); // Store beta in xBaseIdx for use

    if (c_downscale == DLP_U8) {
        RETURN_IF_ERROR(scaleYWithBeta_U8(
            mSize, yFormat == dlp::kernel_frame::storageFormat::rowMajor));
    } else if (c_downscale == DLP_S8) {
        RETURN_IF_ERROR(scaleYWithBeta_S8(
            mSize, yFormat == dlp::kernel_frame::storageFormat::rowMajor));
    } else if (c_downscale == DLP_S32) {
        RETURN_IF_ERROR(scaleYWithBeta_S32(
            mSize, yFormat == dlp::kernel_frame::storageFormat::rowMajor));
    } else if (c_downscale == DLP_F32) {
        RETURN_IF_ERROR(scaleYWithBeta_F32(
            mSize, yFormat == dlp::kernel_frame::storageFormat::rowMajor));
    } else if (c_downscale == DLP_BF16) {
        RETURN_IF_ERROR(scaleYWithBeta_BF16(
            mSize, yFormat == dlp::kernel_frame::storageFormat::rowMajor));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::generateIrLoop(int mSize)
{

    inLocalLabel();
    // Generate the inner loop for GEMV N=1
    // This loop iterates over the columns of the matrix A
    // and performs the dot product for each SIMD column chunk
    // with the vector X

    mov(regTmpAptr, regAptr);
    mov(regTmpYptr, regAptr); // To trace the columns for A matrix

    mov(regXptr, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, x)]);

    regInit(accumBaseIdx, MR);

    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, k_iter)]);

    test(regKIter, regKIter);
    jz(".KLOOP_FRINGE", T_NEAR);

    L(".KLOOP_START");

    // Load X elements(64 elements)
    vmovdqu32(Zmm(xBaseIdx), ptr[regXptr]);

    RETURN_IF_ERROR(processMRBlock(mSize, false));

    add(regTmpYptr, RegBytes); // Move the A column pointer to the next K chunk
    mov(regTmpAptr, regTmpYptr); // Update the A pointer
    add(regXptr, RegBytes);      // Update the X pointer to the next K chunk

    dec(regKIter);
    jnz(".KLOOP_START", T_NEAR);

    L(".KLOOP_FRINGE");

    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, k_left)]);
    test(regKIter, regKIter);
    jz(".KLOOP_FRINGE_END", T_NEAR);

    // Load remaining X elements (kLeft - 8 bit elements)
    vmovdqu8(Zmm(xBaseIdx) | mask_regs[0] | T_z, ptr[regXptr]);
    processMRBlock(mSize, true);

    L(".KLOOP_FRINGE_END");

    RETURN_IF_ERROR(reduceAccumulation(mSize));

    if (alphaScalingType != dlp::kernel_frame::scalingType::one) {
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
jitU8S8VNNI_GEMVN1<KType>::generateMLoop(utils::gemvN1GeneratorParams& params)
{
    inLocalLabel();
    // Generate the main M-loop for GEMV N=1
    // This loop iterates over the rows of the matrix
    // and performs the necessary computations for each row
    mov(regMIter, ptr[stackPtr + offsetof(dlp::kernels::gemvN1Params, m_iter)]);
    test(regMIter, regMIter);
    jz(".M_FRINGE", T_NEAR);

    L(".MLOOP_START");

    // if (params.betaScalingType != dlp::kernel_frame::scalingType::zero) {
    //     prefetcht0(ptr[regYptr]);
    // }

    if (params.alphaScalingType != dlp::kernel_frame::scalingType::zero) {

        RETURN_IF_ERROR(
            generateIrLoop(MR)); // Processes and Accumulates entire K dim

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
        jnz(".MLOOP_START", T_NEAR);
    }

    L(".MLOOP_END");
    L(".M_FRINGE");

    RETURN_IF_ERROR(generateIrLoop(params.M_LEFT));

    outLocalLabel();
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::generateKernel(utils::gemvN1GeneratorParams& params)
{
    Xbyak::util::StackFrame stackFrame(this, 1, 13, 0);
    c_downscale      = params.c_downscale;
    MR               = params.MR;
    M_LEFT           = params.M_LEFT;
    yFormat          = params.yFormat;
    alphaScalingType = params.alphaScalingType;
    betaScalingType  = params.betaScalingType;

    initializeStackFrame(stackFrame);

    RETURN_IF_ERROR(allocateRegisters());

    initializeParameters();

    loadMasks();

    if (params.mloop) {
        RETURN_IF_ERROR(generateMLoop(params));
    } else {
        RETURN_IF_ERROR(generateIrLoop(params.M_LEFT));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::reduceAccToXmm(int accIdx,
                                          int tmpBaseIdx,
                                          int subBlockSize)
{
    // Reduce the accumulation register to XMM
    for (int i = 0; i < 4; i++)
        vpxord(Xbyak::Ymm(tmpBaseIdx + i), Xbyak::Ymm(tmpBaseIdx + i),
               Xbyak::Ymm(tmpBaseIdx + i));

    for (int r = 0; r < subBlockSize; r++) {
        int accIdxT      = accIdx + r;     // original accumulator (row)
        int tmpScalarIdx = tmpBaseIdx + r; // temp slot for scalar reduction

        // Step 1: Fold hi 256 + lo 256 (16 -> 8 lanes)
        vextracti32x8(Xbyak::Ymm(tmpScalarIdx), Zmm(accIdxT), 1);
        vpaddd(Xbyak::Ymm(tmpScalarIdx), Xbyak::Ymm(tmpScalarIdx),
               Xbyak::Ymm(accIdxT));
    }

    vphaddd(Xbyak::Ymm(tmpBaseIdx), Xbyak::Ymm(tmpBaseIdx),
            Xbyak::Ymm(tmpBaseIdx + 1));
    vphaddd(Xbyak::Ymm(tmpBaseIdx + 2), Xbyak::Ymm(tmpBaseIdx + 2),
            Xbyak::Ymm(tmpBaseIdx + 3));
    vphaddd(Xbyak::Ymm(tmpBaseIdx), Xbyak::Ymm(tmpBaseIdx),
            Xbyak::Ymm(tmpBaseIdx + 2));

    vextracti32x4(Xbyak::Xmm(tmpBaseIdx + 1), Xbyak::Ymm(tmpBaseIdx), 1);

    vpaddd(Xbyak::Xmm(tmpBaseIdx), Xbyak::Xmm(tmpBaseIdx + 1),
           Xbyak::Xmm(tmpBaseIdx));
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::reduceAccumulation(int mSize)
{

    // Process in sub-blocks of up to 4 rows
    for (int j = 0; j < mSize; j += 4) {
        int subBlockSize = (mSize - j) < 4 ? (mSize - j) : 4;

        RETURN_IF_ERROR(
            reduceAccToXmm(accumBaseIdx + j, tmpBaseIdx, subBlockSize));

        // Xmm(tmpBaseIdx) lane0 now holds full int32 dot product for
        // row

        // Insert scalar XMM into packed destination register lane
        // group. We use vinserti32x4 to place each row scalar XMM in
        // 128-bit slots. Each slot chooses which 128-bit
        // position within the ZMM.
        vinserti32x4(Xbyak::Zmm(accumBaseIdx), Xbyak::Zmm(accumBaseIdx),
                     Xbyak::Xmm(tmpBaseIdx), j / 4);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVN1<KType>::processMRBlock(int mSize, bool isFringe)
{
    // Perform the compute over the MR rows
    int mLeft = mSize % 4;
    regInit(tmpBaseIdx, tmpReg);
    xor_(regTmp1, regTmp1);

    for (int i = 0; i < mSize / 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (!isFringe) {
                vmovdqu32(Zmm(tmpBaseIdx + j), ptr[regTmpAptr + regTmp1]);
            } else {
                vmovdqu8(Zmm(tmpBaseIdx + j) | mask_regs[0] | T_z,
                         ptr[regTmpAptr + regTmp1]);
            }
            vpdpbusd(Zmm(accumBaseIdx + i * 4 + j), Zmm(tmpBaseIdx + j),
                     Zmm(xBaseIdx));
            add(regTmp1, regRsA);
        }
    }

    for (int j = 0; j < mLeft; j++) {
        if (!isFringe) {
            vmovdqu32(Zmm(tmpBaseIdx + j), ptr[regTmpAptr + regTmp1]);
        } else {
            vmovdqu8(Zmm(tmpBaseIdx + j) | mask_regs[0] | T_z,
                     ptr[regTmpAptr + regTmp1]);
        }
        vpdpbusd(Zmm(accumBaseIdx + (mSize / 4) * 4 + j), Zmm(tmpBaseIdx + j),
                 Zmm(xBaseIdx));
        add(regTmp1, regRsA);
    }

    return dlp::jit::jitGeneratorError::success;
}
} // namespace amdzen::gen

// Explicit template instantiation
template class amdzen::gen::jitU8S8VNNI_GEMVN1<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
