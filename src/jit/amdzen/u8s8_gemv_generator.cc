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

#include "u8s8_gemv_generator.hh"

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
            if (accumulatorsAreF32) {
                // F32 accumulators (after post-ops) - convert to S32 before
                // storing
                vcvtps2dq(Zmm(tmpBaseIdx), Zmm(accumBaseIdx + i));
                vmovdqu32(ptr[regTmpYptr], Zmm(tmpBaseIdx));
            } else {
                // S32 accumulators (no post-ops) - direct store
                vmovdqu32(ptr[regTmpYptr], Zmm(accumBaseIdx + i));
            }
        }
        if (mLeft) {
            if (accumulatorsAreF32) {
                // F32 accumulators - convert to S32 before storing
                vcvtps2dq(Zmm(tmpBaseIdx),
                          Zmm(accumBaseIdx + (mSize / nElemsPerReg)));
                vmovdqu32(ptr[regTmpYptr] | mask_regs[1], Zmm(tmpBaseIdx));
            } else {
                // S32 accumulators - direct store
                vmovdqu32(ptr[regTmpYptr] | mask_regs[1],
                          Zmm(accumBaseIdx + (mSize / nElemsPerReg)));
            }
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
            if (accumulatorsAreF32) {
                // F32 accumulators (after post-ops) - convert F32→S32 then
                // clamp to U8
                vcvtps2dq(Zmm(tmpBaseIdx), Zmm(accumBaseIdx + i));
                vpmaxsd(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 1));
                vpminsd(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 2));
                vpmovdb(ptr[regTmpYptr], Zmm(tmpBaseIdx));
            } else {
                // S32 accumulators (no post-ops) - direct clamp to U8
                vpmaxsd(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 1),
                        Zmm(accumBaseIdx + i));
                vpminsd(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 2), Zmm(tmpBaseIdx));
                vpmovdb(ptr[regTmpYptr], Zmm(tmpBaseIdx));
            }
        }
        if (mLeft) {
            if (accumulatorsAreF32) {
                // F32 accumulators - convert F32→S32 then clamp to U8
                vcvtps2dq(Zmm(tmpBaseIdx),
                          Zmm(accumBaseIdx + (mSize / nElemsPerReg)));
                vpmaxsd(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 1));
                vpminsd(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 2));
                vpmovdb(ptr[regTmpYptr] | mask_regs[1], Zmm(tmpBaseIdx));
            } else {
                // S32 accumulators - direct clamp to U8
                vpmaxsd(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 1),
                        Zmm(accumBaseIdx + (mSize / nElemsPerReg)));
                vpminsd(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 2), Zmm(tmpBaseIdx));
                vpmovdb(ptr[regTmpYptr] | mask_regs[1], Zmm(tmpBaseIdx));
            }
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
            if (accumulatorsAreF32) {
                // F32 accumulators (after post-ops) - convert F32→S32 then
                // saturate to S8
                vcvtps2dq(Zmm(tmpBaseIdx), Zmm(accumBaseIdx + i));
                vpmovsdb(ptr[regTmpYptr], Zmm(tmpBaseIdx));
            } else {
                // S32 accumulators (no post-ops) - direct saturation to S8
                vpmovsdb(ptr[regTmpYptr], Zmm(accumBaseIdx + i));
            }
        }
        if (mLeft) {
            if (accumulatorsAreF32) {
                // F32 accumulators - convert F32→S32 then saturate to S8
                vcvtps2dq(Zmm(tmpBaseIdx),
                          Zmm(accumBaseIdx + (mSize / nElemsPerReg)));
                vpmovsdb(ptr[regTmpYptr] | mask_regs[1], Zmm(tmpBaseIdx));
            } else {
                // S32 accumulators - direct saturation to S8
                vpmovsdb(ptr[regTmpYptr] | mask_regs[1],
                         Zmm(accumBaseIdx + (mSize / nElemsPerReg)));
            }
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
        // Column-major storage for F32
        updateCBufferPointers();

        for (int i = 0; i < mSize / nElemsPerReg; i += 1) {
            if (accumulatorsAreF32) {
                // F32 accumulators (after post-ops) - direct store
                vmovups(ptr[regTmpYptr], Zmm(accumBaseIdx + i));
            } else {
                // S32 accumulators (no post-ops) - convert to F32 and store
                vcvtdq2ps(Zmm(tmpBaseIdx), Zmm(accumBaseIdx + i));
                vmovups(ptr[regTmpYptr], Zmm(tmpBaseIdx));
            }
        }
        if (mLeft) {
            if (accumulatorsAreF32) {
                // F32 accumulators - direct store
                vmovups(ptr[regTmpYptr] | mask_regs[1],
                        Zmm(accumBaseIdx + (mSize / nElemsPerReg)));
            } else {
                // S32 accumulators - convert to F32 and store
                vcvtdq2ps(Zmm(tmpBaseIdx),
                          Zmm(accumBaseIdx + (mSize / nElemsPerReg)));
                vmovups(ptr[regTmpYptr] | mask_regs[1], Zmm(tmpBaseIdx));
            }
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
            if (accumulatorsAreF32) {
                // F32 accumulators (after post-ops) - direct F32→BF16
                // conversion
                vmovups(Zmm(tmpBaseIdx), Zmm(accumBaseIdx + i));
            } else {
                // S32 accumulators (no post-ops) - convert S32→F32 first
                vcvtdq2ps(Zmm(tmpBaseIdx), Zmm(accumBaseIdx + i));
            }

            // Software BF16 conversion
            // Algorithm: bf16 = (f32 + 0x00007FFF + ((f32 >> 16) & 1)) >> 16

            // Extract LSB of bit 16 for rounding (ties-to-even)
            vpsrld(Zmm(tmpBaseIdx + 2), Zmm(tmpBaseIdx), 16);
            vpandd(Zmm(tmpBaseIdx + 2), Zmm(tmpBaseIdx + 2),
                   ptr[rip + label_bf16_lsb_mask]);

            // Add rounding bias (0x00007FFF) + LSB
            vpaddd(Zmm(tmpBaseIdx + 1), Zmm(tmpBaseIdx),
                   ptr[rip + label_bf16_round_bias]);
            vpaddd(Zmm(tmpBaseIdx + 1), Zmm(tmpBaseIdx + 1),
                   Zmm(tmpBaseIdx + 2));

            // Shift right 16 bits to get BF16 in lower 16 bits of each dword
            vpsrld(Zmm(tmpBaseIdx + 1), Zmm(tmpBaseIdx + 1), 16);

            // Pack 16x32-bit to 16x16-bit: use vpmovdw to convert dword to word
            vpmovdw(Ymm(tmpBaseIdx + 1), Zmm(tmpBaseIdx + 1));

            vmovdqu16(ptr[regTmpYptr], Ymm(tmpBaseIdx + 1));
        }

        if (mLeft) {
            if (accumulatorsAreF32) {
                // F32 accumulators - direct use
                vmovups(Zmm(tmpBaseIdx),
                        Zmm(accumBaseIdx + (mSize / nElemsPerReg)));
            } else {
                // S32 accumulators - convert to F32 first
                vcvtdq2ps(Zmm(tmpBaseIdx),
                          Zmm(accumBaseIdx + (mSize / nElemsPerReg)));
            }

            // Software BF16 conversion for masked case
            vpsrld(Zmm(tmpBaseIdx + 2), Zmm(tmpBaseIdx), 16);
            vpandd(Zmm(tmpBaseIdx + 2), Zmm(tmpBaseIdx + 2),
                   ptr[rip + label_bf16_lsb_mask]);
            vpaddd(Zmm(tmpBaseIdx + 1), Zmm(tmpBaseIdx),
                   ptr[rip + label_bf16_round_bias]);
            vpaddd(Zmm(tmpBaseIdx + 1), Zmm(tmpBaseIdx + 1),
                   Zmm(tmpBaseIdx + 2));
            vpsrld(Zmm(tmpBaseIdx + 1), Zmm(tmpBaseIdx + 1), 16);
            vpmovdw(Ymm(tmpBaseIdx + 1), Zmm(tmpBaseIdx + 1));

            vmovdqu16(ptr[regTmpYptr] | mask_regs[1], Ymm(tmpBaseIdx + 1));
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

        // Handle both S32 and F32 accumulators
        if (accumulatorsAreF32) {
            // F32 accumulators (after post-ops) - convert to S32 before
            // extraction
            vcvtps2dq(Zmm(tmpBaseIdx), Zmm(accumBaseIdx + i));
            // Extract 4 chunks of 128-bits from the converted S32 ZMM
            for (int j = 0; j < elems_in_reg; j += 4) {
                vextracti32x4(Xmm(tmpBaseIdx + 1 + j / 4), Zmm(tmpBaseIdx),
                              j / 4);
            }
        } else {
            // S32 accumulators (no post-ops) - direct extraction
            for (int j = 0; j < elems_in_reg; j += 4) {
                vextracti32x4(Xmm(tmpBaseIdx + j / 4), Zmm(accumBaseIdx + i),
                              j / 4);
            }
        }

        for (int j = 0; j < elems_in_reg; j++) {
            int tmp_reg    = accumulatorsAreF32 ? (1 + j / 4) : (j / 4);
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

        // Handle both S32 and F32 accumulators
        if (accumulatorsAreF32) {
            // F32 accumulators (after post-ops) - convert F32→S32 then clamp to
            // U8
            vcvtps2dq(Zmm(tmpBaseIdx), Zmm(accumBaseIdx + i));
            vpmaxsd(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 1));
            vpminsd(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 2));
            vpmovdb(Xmm(tmpBaseIdx), Zmm(tmpBaseIdx));
        } else {
            // S32 accumulators (no post-ops) - direct clamp to U8
            vpmaxsd(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 1),
                    Zmm(accumBaseIdx + i));
            vpminsd(Zmm(tmpBaseIdx), Zmm(tmpBaseIdx + 2), Zmm(tmpBaseIdx));
            vpmovdb(Xmm(tmpBaseIdx), Zmm(tmpBaseIdx));
        }

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

        // Handle both S32 and F32 accumulators
        if (accumulatorsAreF32) {
            // F32 accumulators (after post-ops) - convert F32→S32 then saturate
            // to S8
            vcvtps2dq(Zmm(tmpBaseIdx), Zmm(accumBaseIdx + i));
            vpmovsdb(Xmm(tmpBaseIdx), Zmm(tmpBaseIdx));
        } else {
            // S32 accumulators (no post-ops) - direct saturation to S8
            vpmovsdb(Xmm(tmpBaseIdx), Zmm(accumBaseIdx + i));
        }

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

    // Process each ZMM register (which contains 16 elements) for F32 output
    for (int i = 0; i < (mSize + nElemsPerReg - 1) / nElemsPerReg; i++) {
        int elems_in_reg = (i < mSize / nElemsPerReg) ? nElemsPerReg
                                                      : (mSize % nElemsPerReg);
        if (elems_in_reg == 0)
            break;

        // Handle both S32 and F32 accumulators
        if (accumulatorsAreF32) {
            // F32 accumulators (after post-ops) - direct use
            // Extract 4 chunks of 128-bits from the F32 ZMM
            for (int j = 0; j < elems_in_reg; j += 4) {
                vextractf32x4(Xmm(tmpBaseIdx + j / 4), Zmm(accumBaseIdx + i),
                              j / 4);
            }
        } else {
            // S32 accumulators (no post-ops) - convert to F32 first
            vcvtdq2ps(Zmm(tmpBaseIdx), Zmm(accumBaseIdx + i));
            // Extract 4 chunks of 128-bits from the converted F32 ZMM
            for (int j = 0; j < elems_in_reg; j += 4) {
                vextractf32x4(Xmm(tmpBaseIdx + 1 + j / 4), Zmm(tmpBaseIdx),
                              j / 4);
            }
        }

        for (int j = 0; j < elems_in_reg; j++) {
            int tmp_reg    = accumulatorsAreF32 ? (j / 4) : (1 + j / 4);
            int pos_in_reg = j % 4;

            if (pos_in_reg == 0) {
                // First element in XMM can be stored directly (32-bit)
                vmovss(ptr[regTmpYptr], Xbyak::Xmm(tmpBaseIdx + tmp_reg));
            } else {
                // Extract to memory directly (32-bit)
                vextractps(ptr[regTmpYptr], Xbyak::Xmm(tmpBaseIdx + tmp_reg),
                           pos_in_reg);
            }

            // Move to next row (F32 stride)
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

        // Handle both S32 and F32 accumulators
        if (accumulatorsAreF32) {
            // F32 accumulators (after post-ops) - direct F32→BF16 conversion
            vmovups(Zmm(tmpBaseIdx), Zmm(accumBaseIdx + i));
        } else {
            // S32 accumulators (no post-ops) - convert S32→F32 first
            vcvtdq2ps(Zmm(tmpBaseIdx), Zmm(accumBaseIdx + i));
        }

        // Software BF16 conversion
        // Algorithm: bf16 = (f32 + 0x00007FFF + ((f32 >> 16) & 1)) >> 16

        // Extract LSB of bit 16 for rounding (ties-to-even)
        vpsrld(Zmm(tmpBaseIdx + 2), Zmm(tmpBaseIdx), 16);
        vpandd(Zmm(tmpBaseIdx + 2), Zmm(tmpBaseIdx + 2),
               ptr[rip + label_bf16_lsb_mask]);

        // Add rounding bias (0x00007FFF) + LSB
        vpaddd(Zmm(tmpBaseIdx + 1), Zmm(tmpBaseIdx),
               ptr[rip + label_bf16_round_bias]);
        vpaddd(Zmm(tmpBaseIdx + 1), Zmm(tmpBaseIdx + 1), Zmm(tmpBaseIdx + 2));

        // Shift right 16 bits to get BF16 in lower 16 bits of each dword
        vpsrld(Zmm(tmpBaseIdx + 1), Zmm(tmpBaseIdx + 1), 16);

        // Pack 16x32-bit to 16x16-bit: use vpmovdw to convert dword to word
        vpmovdw(Ymm(tmpBaseIdx + 1), Zmm(tmpBaseIdx + 1));

        // Extract 2 chunks of 128-bits from the YMM (each chunk has 8 BF16
        // elements)
        for (int j = 0; j < elems_in_reg; j += 8) {
            vextracti32x4(Xmm(tmpBaseIdx + 2 + j / 8), Ymm(tmpBaseIdx + 1),
                          j / 8);
        }

        for (int j = 0; j < elems_in_reg; j++) {
            int tmp_reg    = j / 8 + 2; // Each XMM holds 8 BF16 elements
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
            vpmulld(Zmm(yBaseIdx) | mask_regs[1] | T_z, Zmm(xBaseIdx),
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

    bool is_unit_beta =
        (betaScalingType == dlp::kernel_frame::scalingType::one);

    if (isRowStored) {
        vpmovzxbd(Zmm(yBaseIdx), Xmm(yBaseIdx));
        if (!is_unit_beta) {
            vpmulld(Zmm(yBaseIdx), Zmm(yBaseIdx), Zmm(xBaseIdx));
        }
        vpaddd(Zmm(accumBaseIdx), Zmm(accumBaseIdx), Zmm(yBaseIdx));
    } else {
        // For column-stored data, load from downscale buffer
        updateCBufferPointers();

        for (int i = 0; i < mSize / nElemsPerReg; i++) {
            vmovdqu8(Xmm(yBaseIdx), ptr[regTmpYptr]);
            vpmovzxbd(Zmm(yBaseIdx), Xmm(yBaseIdx));
            if (is_unit_beta) {
                vpaddd(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i),
                       Zmm(yBaseIdx));
            } else {
                vpmulld(Zmm(yBaseIdx), Zmm(yBaseIdx), Zmm(xBaseIdx));
                vpaddd(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i),
                       Zmm(yBaseIdx));
            }
        }

        if (mLeft) {
            // Use zero-masking (T_z) to zero unmasked elements
            vmovdqu8(Xmm(yBaseIdx) | mask_regs[1] | T_z, ptr[regTmpYptr]);
            vpmovzxbd(Zmm(yBaseIdx), Xmm(yBaseIdx));
            if (is_unit_beta) {
                vpaddd(Zmm(accumBaseIdx + (mSize / nElemsPerReg)) | k2,
                       Zmm(accumBaseIdx + (mSize / nElemsPerReg)),
                       Zmm(yBaseIdx));
            } else {
                vpmulld(Zmm(yBaseIdx), Zmm(yBaseIdx), Zmm(xBaseIdx));
                vpaddd(Zmm(accumBaseIdx + (mSize / nElemsPerReg)),
                       Zmm(accumBaseIdx + (mSize / nElemsPerReg)),
                       Zmm(yBaseIdx));
            }
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
            // Use zero-masking (T_z) to zero unmasked elements
            vmovdqu8(Xmm(yBaseIdx) | mask_regs[1] | T_z, ptr[regTmpYptr]);
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
            // Use zero-masking (T_z) to zero unmasked elements
            vcvtps2dq(Zmm(yBaseIdx) | mask_regs[1] | T_z, ptr[regTmpYptr]);
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
            // Use zero-masking (T_z) to zero unmasked elements
            vmovdqu16(Ymm(yBaseIdx) | mask_regs[1] | T_z, ptr[regTmpYptr]);
            vpmovsxwd(Zmm(yBaseIdx), Ymm(yBaseIdx));
            vpslld(Zmm(yBaseIdx), Zmm(yBaseIdx), 16);
            vcvtps2dq(Zmm(yBaseIdx), Zmm(yBaseIdx));
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

    // Post-ops integration for GEMV N=1
    if (kernelOpsHandlerPtr) {
        // Convert S32 accumulators to F32 for post-ops compatibility
        for (int i = 0; i < (mSize + nElemsPerReg - 1) / nElemsPerReg; i++) {
            vcvtdq2ps(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i));
        }

        // Mark that accumulators are now F32 for store functions
        accumulatorsAreF32 = true;
        bool useMask       = !(mSize / nElemsPerReg);
        // Note: kernelOps vector is stored in the kernelOpsHandlerPtr
        // The handler will iterate through the operations automatically
        RETURN_IF_ERROR(kernelOpsHandlerPtr->generateKernelOps(
            kernelOpsVector, stackPtr, dlp::jit::jitAlgoType::gemv_n1, mSize, 1,
            useMask, 1, accumBaseIdx, (mSize / nElemsPerReg)));

        kernelOpsHandlerPtr->generateKernelOpsAttributes();
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
    c_downscale      = params.c_downscale;
    MR               = params.MR;
    M_LEFT           = params.M_LEFT;
    yFormat          = params.yFormat;
    alphaScalingType = params.alphaScalingType;
    betaScalingType  = params.betaScalingType;

    // Initialize kernel ops handler if post-ops are present
    if (!params.kernelOps.empty()) {
        kernelOpsHandlerPtr =
            std::make_unique<gen::kernelOpsHandler>(this, params.kType);
        kernelOpsVector = params.kernelOps; // Store kernelOps for later use
    }

    RETURN_IF_ERROR(allocateRegisters());

    // Putting inside a scope so that constant data can be generated post
    // the ret instr. StackFrame inserts a ret instr in its destructor.
    {
        Xbyak::util::StackFrame stackFrame(this, 1, 13, 0);
        initializeStackFrame(stackFrame);

        initializeParameters();

        accumulatorsAreF32 = false;
        loadMasks();

        if (params.mloop) {
            RETURN_IF_ERROR(generateMLoop(params));
        } else {
            RETURN_IF_ERROR(generateIrLoop(params.M_LEFT));
        }
    } // StackFrame destructor inserts 'ret' here

    // Generate constant data tables after the return instruction
    generateConstantData();

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

template<utils::kernelInstrType KType>
jitU8S8VNNI_GEMVM1<KType>::jitU8S8VNNI_GEMVM1(void* buffer, size_t bufferSize)
    : Xbyak::CodeGenerator(bufferSize, buffer)
{
}

template<utils::kernelInstrType KType>
void
jitU8S8VNNI_GEMVM1<KType>::initializeStackFrame(Xbyak::util::StackFrame& frame)
{
    stackPtr = frame.p[0];

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
jitU8S8VNNI_GEMVM1<KType>::initializeParameters(
    utils::gemvM1GeneratorParams& params)
{
    NR               = params.NR;
    N_LEFT           = params.N_LEFT;
    KC               = params.KC;
    K_SUB_ITER       = params.K_SUB_ITER;
    yFormat          = params.yFormat;
    alphaScalingType = params.alphaScalingType;
    betaScalingType  = params.betaScalingType;
    mtag_b           = params.mtag_b;
    c_downscale      = params.c_downscale;

    RegBytes = Traits::regBytes;
    numRegs  = Traits::numRegs;

    loadMasks();
    mov(regRsB, ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, rsB)]);
    // mov(regPsB, ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, psB)]);
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::allocateRegisters()
{
    if (NR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Register allocation for U8S8 GEMV M=1:
    // x Registers : K_SUB_ITER (4 registers for X broadcasts)
    // b Registers : K_SUB_ITER (4 registers for B loads)
    // y Registers : NR/nElemsPerReg (for Y load/store)
    // Accumulation registers : (NR/nElemsPerReg) * K_SUB_ITER
    nElemsPerReg =
        RegBytes / sizeof(int32_t); // For int32 output (16 for AVX512)

    yReg     = NR / nElemsPerReg;
    xReg     = K_SUB_ITER;
    bReg     = NR / nElemsPerReg;
    accumReg = (NR / nElemsPerReg) * K_SUB_ITER;
    maskReg  = 0; // AVX512 has dedicated mask registers

    // Register index assignment
    accumBaseIdx = numRegs - accumReg;
    xBaseIdx     = accumBaseIdx - xReg;
    yBaseIdx     = numRegs - yReg;
    bBaseIdx     = xBaseIdx - bReg;
    maskBaseIdx  = bBaseIdx;

    if (maskBaseIdx < 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitU8S8VNNI_GEMVM1<KType>::regInit(int baseIdx, int numRegs)
{
    for (int i = 0; i < numRegs; i++) {
        vpxord(Zmm(baseIdx + i), Zmm(baseIdx + i), Zmm(baseIdx + i));
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::loadMasks()
{
    // Ensuring mapping only from k1 to k7(to avoid k0 usage internally)
    for (int i = 0; i < NUM_USABLE_MASKS; i++) {
        mask_regs[i] = Xbyak::Opmask(MASK_START_IDX + i);
    }

    // Load the N-dimension mask for int32 elements
    kmovw(mask_regs[0],
          ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, nmask_avx512)]);
    kmovw(mask_regs[1],
          ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kLeftmask)]);

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::maskLoadB(int regIdx, int maskIdx)
{
    vmovdqu32(Zmm(bBaseIdx + regIdx) | mask_regs[maskIdx],
              ptr[regTmp2 + regIdx * nElemsPerReg * sizeof(int32_t)]);

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::offsetBPtr(int temp)
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
jitU8S8VNNI_GEMVM1<KType>::computeKxnfringe()
{
    // Broadcast K_SUB_ITER VNNI groups (4 bytes each)
    for (int j = 0; j < K_SUB_ITER; j++) {
        vpbroadcastd(Zmm(xBaseIdx + j), ptr[regXptr + j * 4]);
    }

    int n_iter = N_LEFT / nElemsPerReg;
    int n_left = N_LEFT % nElemsPerReg;

    for (int j = 0; j < K_SUB_ITER; j++) {
        for (int i = 0; i < n_iter; i++) {
            // load elements of B
            vmovdqu32(Zmm(bBaseIdx + i),
                      ptr[regTmp2 + i * nElemsPerReg * sizeof(int32_t)]);

            vpdpbusd(Zmm(accumBaseIdx + K_SUB_ITER * i + j), Zmm(xBaseIdx + j),
                     Zmm(bBaseIdx + i));
        }
        if (n_left) {
            vmovdqu32(Zmm(bBaseIdx + n_iter) | mask_regs[0] | T_z,
                      ptr[regTmpYptr + j * 64]);
            vpdpbusd(Zmm(accumBaseIdx + K_SUB_ITER * n_iter + j),
                     Zmm(xBaseIdx + j), Zmm(bBaseIdx + n_iter));
        }
        add(regTmp2, regRsB);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::computeKxNR(bool nMask)
{
    mov(regTmp2, regBptr);

    if (!nMask) {
        // Broadcast K_SUB_ITER VNNI groups
        for (int i = 0; i < K_SUB_ITER; i += 1) {
            vpbroadcastd(Zmm(xBaseIdx + i), ptr[regXptr + i * 4]);
        }

        for (int j = 0; j < K_SUB_ITER; j += 1) {
            for (int i = 0; i < NR / nElemsPerReg; i += 1) {
                // load elements of B
                vmovdqu32(Zmm(bBaseIdx + i),
                          ptr[regTmp2 + i * nElemsPerReg * sizeof(int32_t)]);
                vpdpbusd(Zmm(accumBaseIdx + K_SUB_ITER * i + j),
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
jitU8S8VNNI_GEMVM1<KType>::compute1xnfringe(bool isLastKGroup)
{
    if (isLastKGroup) {
        vmovdqu8(Xmm(xBaseIdx) | mask_regs[1] | T_z, ptr[regXptr]);
        vpbroadcastd(Zmm(xBaseIdx), Xmm(xBaseIdx));
    } else
        vpbroadcastd(Zmm(xBaseIdx), ptr[regXptr]);

    int n_iter = N_LEFT / nElemsPerReg;
    int n_left = N_LEFT % nElemsPerReg;

    for (int i = 0; i < n_iter; i++) {
        xor_(regTmp1, regTmp1);
        lea(regTmp1, ptr[regTmp1 + i * nElemsPerReg * sizeof(int32_t)]);
        vmovdqu32(Zmm(bBaseIdx + i), ptr[regTmp2 + regTmp1]);
        vpdpbusd(Zmm(accumBaseIdx + K_SUB_ITER * i), Zmm(xBaseIdx),
                 Zmm(bBaseIdx + i));
    }

    if (n_left) {
        vmovdqu32(Zmm(bBaseIdx + n_iter) | mask_regs[0] | T_z, ptr[regTmpYptr]);
        vpdpbusd(Zmm(accumBaseIdx + K_SUB_ITER * n_iter), Zmm(xBaseIdx),
                 Zmm(bBaseIdx + n_iter));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::compute1xNR(bool nMask, bool isLastKGroup)
{
    mov(regTmp2, regBptr);

    if (!nMask) {
        if (isLastKGroup) {
            vmovdqu8(Xmm(xBaseIdx) | mask_regs[1] | T_z, ptr[regXptr]);
            vpbroadcastd(Zmm(xBaseIdx), Xmm(xBaseIdx));
        } else {
            vpbroadcastd(Zmm(xBaseIdx), ptr[regXptr]);
        }

        for (int i = 0; i < NR / nElemsPerReg; i += 1) {
            vmovdqu32(Zmm(bBaseIdx + i),
                      ptr[regTmp2 + i * nElemsPerReg * sizeof(int32_t)]);
            vpdpbusd(Zmm(accumBaseIdx + K_SUB_ITER * i), Zmm(xBaseIdx),
                     Zmm(bBaseIdx + i));
        }
    } else {
        compute1xnfringe(isLastKGroup);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::loopKSubIter(bool kfringe, bool nfringe)
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

    // Moving regTmpYptr to the nLeft Panel

    mov(regTmpYptr, regBptr);
    // In case of N_LEFT < 16, the nLeft panel pointer is the only Panel to
    // process
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

        // Update the pointers for next k iteration
        lea(regXptr, ptr[regXptr + K_SUB_ITER * 4]); // 4 VNNI groups * 4 bytes
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

        // Update the pointers for next k iteration
        lea(regXptr, ptr[regXptr + 4]); // 1 VNNI group = 4 bytes
        lea(regBptr, ptr[regBptr + regRsB]);
        lea(regTmpYptr,
            ptr[regTmpYptr + 64]); // For the nleft, rowStride will always be 16

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
        // Check if this is the very last iteration

        computeKxNR(nfringe);

        // Update the pointers for next k iteration
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
        je(".LAST_K_GROUP", T_NEAR);

        compute1xNR(nfringe, false);
        jmp(".CONTINUE_K_FRINGE");

        L(".LAST_K_GROUP");
        compute1xNR(nfringe, true);

        L(".CONTINUE_K_FRINGE");
        lea(regXptr, ptr[regXptr + 4]);
        lea(regBptr, ptr[regBptr + regRsB]);
        lea(regTmpYptr,
            ptr[regTmpYptr + 64]); // For the nleft, rowStride will always be 16

        dec(regKSubIter);
        jnz(sub_loop_kf_fringe_loop_start, T_NEAR);

        L(sub_loop_kf_fringe_loop_end);
        outLocalLabel();
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::finalAccumulate()
{
    for (int i = 0; i < NR / nElemsPerReg; i += 1) {
        for (int j = 1; j < K_SUB_ITER; j += 1) {
            vpaddd(Zmm(accumBaseIdx + K_SUB_ITER * i),
                   Zmm(accumBaseIdx + K_SUB_ITER * i),
                   Zmm(accumBaseIdx + K_SUB_ITER * i + j));
        }
        vmovdqa32(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + K_SUB_ITER * i));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::scaleWithAlpha()
{
    if (alphaScalingType != dlp::kernel_frame::scalingType::one) {
        mov(regKSubIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, alpha)]);
        vpbroadcastd(Zmm(xBaseIdx), ptr[regKSubIter]);
        for (int i = 0; i < NR / nElemsPerReg; i += 1) {
            vpmulld(Zmm(accumBaseIdx + i), Zmm(xBaseIdx),
                    Zmm(accumBaseIdx + i));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::scaleYWithBeta(bool nMask)
{
    bool isBetaZero = (betaScalingType == dlp::kernel_frame::scalingType::zero);
    bool isBetaOne  = (betaScalingType == dlp::kernel_frame::scalingType::one);

    if (isBetaZero) {
        return dlp::jit::jitGeneratorError::success;
    }

    // Load beta (unless beta==1)
    if (!isBetaOne) {
        mov(regKSubIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, beta)]);
        vpbroadcastd(Zmm(xBaseIdx), ptr[regKSubIter]);
    }

    // Dispatch to type-specific beta scaling
    if (c_downscale == DLP_S32) {
        return scaleYWithBeta_S32(nMask, isBetaOne);
    } else if (c_downscale == DLP_U8) {
        return scaleYWithBeta_U8(nMask, isBetaOne);
    } else if (c_downscale == DLP_S8) {
        return scaleYWithBeta_S8(nMask, isBetaOne);
    } else if (c_downscale == DLP_F32) {
        return scaleYWithBeta_F32(nMask, isBetaOne);
    } else if (c_downscale == DLP_BF16) {
        return scaleYWithBeta_BF16(nMask, isBetaOne);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::scaleYWithBeta_S32(bool nMask, bool isBetaOne)
{
    mov(regTmpYptr, regYptr);

    if (!nMask) {
        if (!isBetaOne) {
            for (int i = 0; i < NR / nElemsPerReg; i++) {
                vpmulld(Zmm(yBaseIdx + i), Zmm(xBaseIdx),
                        ptr[regTmpYptr + i * nElemsPerReg * sizeof(int32_t)]);
                vpaddd(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i),
                       Zmm(yBaseIdx + i));
            }
        } else {
            for (int i = 0; i < NR / nElemsPerReg; i++) {
                vpaddd(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i),
                       ptr[regTmpYptr + i * nElemsPerReg * sizeof(int32_t)]);
            }
        }
    } else {
        scaleYWithBetaFringe_S32(isBetaOne);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::scaleYWithBeta_U8(bool nMask, bool isBetaOne)
{
    updateYBufferPointers(); // Point to downscale buffer

    if (!nMask) {
        for (int i = 0; i < NR / nElemsPerReg; i++) {
            vmovdqu8(Xmm(yBaseIdx + i), ptr[regTmpYptr + i * 16]);
            vpmovzxbd(Zmm(yBaseIdx + i), Xmm(yBaseIdx + i)); // U8 → S32

            if (!isBetaOne) {
                vpmulld(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i), Zmm(xBaseIdx));
            }
            vpaddd(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i),
                   Zmm(yBaseIdx + i));
        }
    } else {
        scaleYWithBetaFringe_U8(isBetaOne);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::scaleYWithBeta_S8(bool nMask, bool isBetaOne)
{
    updateYBufferPointers();

    if (!nMask) {
        for (int i = 0; i < NR / nElemsPerReg; i++) {
            vmovdqu8(Xmm(yBaseIdx + i), ptr[regTmpYptr + i * 16]);
            vpmovsxbd(Zmm(yBaseIdx + i), Xmm(yBaseIdx + i)); // S8 → S32

            if (!isBetaOne) {
                vpmulld(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i), Zmm(xBaseIdx));
            }
            vpaddd(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i),
                   Zmm(yBaseIdx + i));
        }
    } else {
        scaleYWithBetaFringe_S8(isBetaOne);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::scaleYWithBeta_F32(bool nMask, bool isBetaOne)
{
    updateYBufferPointers();

    if (!nMask) {
        for (int i = 0; i < NR / nElemsPerReg; i++) {
            vcvtps2dq(Zmm(yBaseIdx + i),
                      ptr[regTmpYptr + i * RegBytes]); // F32 → S32

            if (!isBetaOne) {
                vpmulld(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i), Zmm(xBaseIdx));
            }
            vpaddd(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i),
                   Zmm(yBaseIdx + i));
        }
    } else {
        scaleYWithBetaFringe_F32(isBetaOne);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::scaleYWithBeta_BF16(bool nMask, bool isBetaOne)
{
    updateYBufferPointers();

    if (!nMask) {
        for (int i = 0; i < NR / nElemsPerReg; i++) {
            vmovdqu16(Ymm(yBaseIdx + i), ptr[regTmpYptr + i * 32]);
            vpmovsxwd(Zmm(yBaseIdx + i),
                      Ymm(yBaseIdx + i)); // BF16 → S32 (16→32)
            vpslld(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i),
                   16); // Shift to upper bits
            vcvtps2dq(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i)); // F32 → S32

            if (!isBetaOne) {
                vpmulld(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i), Zmm(xBaseIdx));
            }
            vpaddd(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i),
                   Zmm(yBaseIdx + i));
        }
    } else {
        scaleYWithBetaFringe_BF16(isBetaOne);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::scaleYWithBetaFringe_S32(bool isBetaOne)
{
    int n_iter = N_LEFT / nElemsPerReg;
    int n_left = N_LEFT % nElemsPerReg;

    if (!isBetaOne) {
        for (int i = 0; i < n_iter; i++) {
            vpmulld(Zmm(yBaseIdx + i), Zmm(xBaseIdx),
                    ptr[regTmpYptr + i * nElemsPerReg * sizeof(int32_t)]);
            vpaddd(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i),
                   Zmm(yBaseIdx + i));
        }
        if (n_left) {
            vpmulld(Zmm(yBaseIdx + n_iter) | mask_regs[0] | T_z, Zmm(xBaseIdx),
                    ptr[regTmpYptr + n_iter * nElemsPerReg * sizeof(int32_t)]);
            vpaddd(Zmm(accumBaseIdx + n_iter), Zmm(accumBaseIdx + n_iter),
                   Zmm(yBaseIdx + n_iter));
        }
    } else {
        for (int i = 0; i < n_iter; i++) {
            vpaddd(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i),
                   ptr[regTmpYptr + i * nElemsPerReg * sizeof(int32_t)]);
        }
        if (n_left) {
            vpaddd(Zmm(accumBaseIdx + n_iter) | mask_regs[0] | T_z,
                   Zmm(accumBaseIdx + n_iter),
                   ptr[regTmpYptr + n_iter * nElemsPerReg * sizeof(int32_t)]);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::scaleYWithBetaFringe_U8(bool isBetaOne)
{
    updateYBufferPointers();

    int n_iter = N_LEFT / nElemsPerReg;
    int n_left = N_LEFT % nElemsPerReg;

    for (int i = 0; i < n_iter; i++) {
        vmovdqu8(Xmm(yBaseIdx + i), ptr[regTmpYptr + i * 16]);
        vpmovzxbd(Zmm(yBaseIdx + i), Xmm(yBaseIdx + i)); // U8 → S32

        if (!isBetaOne) {
            vpmulld(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i), Zmm(xBaseIdx));
        }
        vpaddd(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i), Zmm(yBaseIdx + i));
    }

    if (n_left) {
        vmovdqu8(Xmm(yBaseIdx + n_iter) | mask_regs[0] | T_z,
                 ptr[regTmpYptr + n_iter * 16]);
        vpmovzxbd(Zmm(yBaseIdx + n_iter), Xmm(yBaseIdx + n_iter));

        if (!isBetaOne) {
            vpmulld(Zmm(yBaseIdx + n_iter), Zmm(yBaseIdx + n_iter),
                    Zmm(xBaseIdx));
        }
        vpaddd(Zmm(accumBaseIdx + n_iter), Zmm(accumBaseIdx + n_iter),
               Zmm(yBaseIdx + n_iter));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::scaleYWithBetaFringe_S8(bool isBetaOne)
{
    updateYBufferPointers();

    int n_iter = N_LEFT / nElemsPerReg;
    int n_left = N_LEFT % nElemsPerReg;

    for (int i = 0; i < n_iter; i++) {
        vmovdqu8(Xmm(yBaseIdx + i), ptr[regTmpYptr + i * 16]);
        vpmovsxbd(Zmm(yBaseIdx + i), Xmm(yBaseIdx + i)); // S8 → S32

        if (!isBetaOne) {
            vpmulld(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i), Zmm(xBaseIdx));
        }
        vpaddd(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i), Zmm(yBaseIdx + i));
    }

    if (n_left) {
        // Use zero-masking (T_z) to zero unmasked elements
        vmovdqu8(Xmm(yBaseIdx + n_iter) | mask_regs[0] | T_z,
                 ptr[regTmpYptr + n_iter * 16]);
        vpmovsxbd(Zmm(yBaseIdx + n_iter), Xmm(yBaseIdx + n_iter));

        if (!isBetaOne) {
            vpmulld(Zmm(yBaseIdx + n_iter), Zmm(yBaseIdx + n_iter),
                    Zmm(xBaseIdx));
        }
        vpaddd(Zmm(accumBaseIdx + n_iter), Zmm(accumBaseIdx + n_iter),
               Zmm(yBaseIdx + n_iter));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::scaleYWithBetaFringe_F32(bool isBetaOne)
{
    updateYBufferPointers();

    int n_iter = N_LEFT / nElemsPerReg;
    int n_left = N_LEFT % nElemsPerReg;

    for (int i = 0; i < n_iter; i++) {
        vcvtps2dq(Zmm(yBaseIdx + i),
                  ptr[regTmpYptr + i * RegBytes]); // F32 → S32

        if (!isBetaOne) {
            vpmulld(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i), Zmm(xBaseIdx));
        }
        vpaddd(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i), Zmm(yBaseIdx + i));
    }

    if (n_left) {
        vcvtps2dq(Zmm(yBaseIdx + n_iter) | mask_regs[0] | T_z,
                  ptr[regTmpYptr + n_iter * RegBytes]);

        if (!isBetaOne) {
            vpmulld(Zmm(yBaseIdx + n_iter), Zmm(yBaseIdx + n_iter),
                    Zmm(xBaseIdx));
        }
        vpaddd(Zmm(accumBaseIdx + n_iter), Zmm(accumBaseIdx + n_iter),
               Zmm(yBaseIdx + n_iter));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::scaleYWithBetaFringe_BF16(bool isBetaOne)
{
    updateYBufferPointers();

    int n_iter = N_LEFT / nElemsPerReg;
    int n_left = N_LEFT % nElemsPerReg;

    for (int i = 0; i < n_iter; i++) {
        vmovdqu16(Ymm(yBaseIdx + i), ptr[regTmpYptr + i * 32]);
        vpmovsxwd(Zmm(yBaseIdx + i), Ymm(yBaseIdx + i));  // BF16 → S32 (16→32)
        vpslld(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i), 16); // Shift to upper bits
        vcvtps2dq(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i));  // F32 → S32

        if (!isBetaOne) {
            vpmulld(Zmm(yBaseIdx + i), Zmm(yBaseIdx + i), Zmm(xBaseIdx));
        }
        vpaddd(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i), Zmm(yBaseIdx + i));
    }

    if (n_left) {
        vmovdqu16(Ymm(yBaseIdx + n_iter) | mask_regs[0] | T_z,
                  ptr[regTmpYptr + n_iter * 32]);
        vpmovsxwd(Zmm(yBaseIdx + n_iter), Ymm(yBaseIdx + n_iter));
        vpslld(Zmm(yBaseIdx + n_iter), Zmm(yBaseIdx + n_iter), 16);
        vcvtps2dq(Zmm(yBaseIdx + n_iter), Zmm(yBaseIdx + n_iter));

        if (!isBetaOne) {
            vpmulld(Zmm(yBaseIdx + n_iter), Zmm(yBaseIdx + n_iter),
                    Zmm(xBaseIdx));
        }
        vpaddd(Zmm(accumBaseIdx + n_iter), Zmm(accumBaseIdx + n_iter),
               Zmm(yBaseIdx + n_iter));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitU8S8VNNI_GEMVM1<KType>::updateYBufferPointers()
{
    mov(regTmpYptr,
        ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
            + offsetof(lpgemm_post_op_attr, buf_downscale)]);

    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)
            + offsetof(lpgemm_post_op_attr, post_op_c_j)]);

    add(regTmp1, regIncN);

    // Scale by data type size
    if (c_downscale == DLP_BF16) {
        lea(regTmp1, ptr[regTmp1 * 2]);
    } else if (c_downscale == DLP_F32) {
        lea(regTmp1, ptr[regTmp1 * 4]);
    }

    add(regTmpYptr, regTmp1);
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::storeYValues(bool nMask)
{
    // Dispatch to type-specific store functions
    if (c_downscale == DLP_S32) {
        return storeYValues_S32(nMask);
    } else if (c_downscale == DLP_U8) {
        return storeYValues_U8(nMask);
    } else if (c_downscale == DLP_S8) {
        return storeYValues_S8(nMask);
    } else if (c_downscale == DLP_F32) {
        return storeYValues_F32(nMask);
    } else if (c_downscale == DLP_BF16) {
        return storeYValues_BF16(nMask);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::storeYValues_S32(bool nMask)
{
    mov(regTmpYptr, regYptr);

    if (!nMask) {
        for (int i = 0; i < NR / nElemsPerReg; i += 1) {
            if (accumulatorsAreF32) {
                // F32 accumulators (after post-ops) - convert to S32 before
                // storing
                vcvtps2dq(Zmm(xBaseIdx), Zmm(accumBaseIdx + i));
                vmovdqu32(ptr[regTmpYptr + i * nElemsPerReg * sizeof(int32_t)],
                          Zmm(xBaseIdx));
            } else {
                // S32 accumulators (no post-ops) - direct store
                vmovdqu32(ptr[regTmpYptr + i * nElemsPerReg * sizeof(int32_t)],
                          Zmm(accumBaseIdx + i));
            }
        }
    } else {
        storeYValuesFringe_S32();
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::storeYValues_U8(bool nMask)
{
    updateYBufferPointers(); // Point to downscale buffer

    // Setup clamping constants for U8 (0-255)
    vpxord(Zmm(xBaseIdx + 1), Zmm(xBaseIdx + 1), Zmm(xBaseIdx + 1)); // 0
    mov(regKSubIter, 255);
    vpbroadcastd(Zmm(xBaseIdx + 2), regKSubIter.cvt32()); // 255

    if (!nMask) {
        for (int i = 0; i < NR / nElemsPerReg; i++) {
            if (accumulatorsAreF32) {
                // F32 accumulators (after post-ops) - convert F32→S32 then
                // clamp to U8
                vcvtps2dq(Zmm(xBaseIdx), Zmm(accumBaseIdx + i));
                vpmaxsd(Zmm(xBaseIdx), Zmm(xBaseIdx), Zmm(xBaseIdx + 1));
                vpminsd(Zmm(xBaseIdx), Zmm(xBaseIdx), Zmm(xBaseIdx + 2));
                vpmovdb(ptr[regTmpYptr + i * 16], Zmm(xBaseIdx));
            } else {
                // S32 accumulators (no post-ops) - direct clamp to U8
                vpmaxsd(Zmm(xBaseIdx), Zmm(accumBaseIdx + i),
                        Zmm(xBaseIdx + 1));
                vpminsd(Zmm(xBaseIdx), Zmm(xBaseIdx), Zmm(xBaseIdx + 2));
                vpmovdb(ptr[regTmpYptr + i * 16], Zmm(xBaseIdx));
            }
        }
    } else {
        storeYValuesFringe_U8();
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::storeYValues_S8(bool nMask)
{
    updateYBufferPointers();

    if (!nMask) {
        for (int i = 0; i < NR / nElemsPerReg; i++) {
            if (accumulatorsAreF32) {
                // F32 accumulators (after post-ops) - convert F32→S32 then
                // saturate to S8
                vcvtps2dq(Zmm(xBaseIdx), Zmm(accumBaseIdx + i));
                vpmovsdb(ptr[regTmpYptr + i * 16], Zmm(xBaseIdx));
            } else {
                // S32 accumulators (no post-ops) - direct saturation to S8
                vpmovsdb(ptr[regTmpYptr + i * 16], Zmm(accumBaseIdx + i));
            }
        }
    } else {
        storeYValuesFringe_S8();
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::storeYValues_F32(bool nMask)
{
    updateYBufferPointers();

    if (!nMask) {
        for (int i = 0; i < NR / nElemsPerReg; i++) {
            if (accumulatorsAreF32) {
                // F32 accumulators (after post-ops) - direct store
                vmovups(ptr[regTmpYptr + i * RegBytes], Zmm(accumBaseIdx + i));
            } else {
                // S32 accumulators (no post-ops) - convert to F32 and store
                vcvtdq2ps(Zmm(xBaseIdx), Zmm(accumBaseIdx + i));
                vmovups(ptr[regTmpYptr + i * RegBytes], Zmm(xBaseIdx));
            }
        }
    } else {
        storeYValuesFringe_F32();
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::storeYValues_BF16(bool nMask)
{
    updateYBufferPointers();

    if (!nMask) {
        for (int i = 0; i < NR / nElemsPerReg; i++) {
            if (accumulatorsAreF32) {
                // F32 accumulators (after post-ops) - direct F32→BF16
                // conversion
                vmovups(Zmm(xBaseIdx), Zmm(accumBaseIdx + i));
            } else {
                // S32 accumulators (no post-ops) - convert S32→F32 first
                vcvtdq2ps(Zmm(xBaseIdx), Zmm(accumBaseIdx + i));
            }

            // Software BF16 conversion
            // Algorithm: bf16 = (f32 + 0x00007FFF + ((f32 >> 16) & 1)) >> 16

            // Extract LSB of bit 16 for rounding (ties-to-even)
            vpsrld(Zmm(xBaseIdx + 2), Zmm(xBaseIdx), 16);
            vpandd(Zmm(xBaseIdx + 2), Zmm(xBaseIdx + 2),
                   ptr[rip + label_bf16_lsb_mask]);

            // Add rounding bias (0x00007FFF) + LSB
            vpaddd(Zmm(xBaseIdx + 1), Zmm(xBaseIdx),
                   ptr[rip + label_bf16_round_bias]);
            vpaddd(Zmm(xBaseIdx + 1), Zmm(xBaseIdx + 1), Zmm(xBaseIdx + 2));

            // Shift right 16 bits to get BF16 in lower 16 bits of each dword
            vpsrld(Zmm(xBaseIdx + 1), Zmm(xBaseIdx + 1), 16);

            // Pack 16x32-bit to 16x16-bit: use vpmovdw to convert dword to word
            vpmovdw(Ymm(xBaseIdx + 1), Zmm(xBaseIdx + 1));

            vmovdqu16(ptr[regTmpYptr + i * 32], Ymm(xBaseIdx + 1));
        }
    } else {
        storeYValuesFringe_BF16();
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::storeYValuesFringe_S32()
{
    int n_iter = N_LEFT / nElemsPerReg;
    int n_left = N_LEFT % nElemsPerReg;

    for (int i = 0; i < n_iter; i++) {
        if (accumulatorsAreF32) {
            // F32 accumulators (after post-ops) - convert to S32 before storing
            vcvtps2dq(Zmm(xBaseIdx), Zmm(accumBaseIdx + i));
            vmovdqu32(ptr[regTmpYptr + i * nElemsPerReg * sizeof(int32_t)],
                      Zmm(xBaseIdx));
        } else {
            // S32 accumulators (no post-ops) - direct store
            vmovdqu32(ptr[regTmpYptr + i * nElemsPerReg * sizeof(int32_t)],
                      Zmm(accumBaseIdx + i));
        }
    }
    if (n_left) {
        if (accumulatorsAreF32) {
            // F32 accumulators - convert to S32 before storing
            vcvtps2dq(Zmm(xBaseIdx), Zmm(accumBaseIdx + n_iter));
            vmovdqu32(ptr[regTmpYptr + n_iter * nElemsPerReg * sizeof(int32_t)]
                          | mask_regs[0],
                      Zmm(xBaseIdx));
        } else {
            // S32 accumulators - direct store
            vmovdqu32(ptr[regTmpYptr + n_iter * nElemsPerReg * sizeof(int32_t)]
                          | mask_regs[0],
                      Zmm(accumBaseIdx + n_iter));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::storeYValuesFringe_U8()
{
    updateYBufferPointers();

    // Setup clamping constants for U8 (0-255)
    vpxord(Zmm(xBaseIdx + 1), Zmm(xBaseIdx + 1), Zmm(xBaseIdx + 1)); // 0
    mov(regKSubIter, 255);
    vpbroadcastd(Zmm(xBaseIdx + 2), regKSubIter.cvt32()); // 255

    int n_iter = N_LEFT / nElemsPerReg;
    int n_left = N_LEFT % nElemsPerReg;

    for (int i = 0; i < n_iter; i++) {
        if (accumulatorsAreF32) {
            // F32 accumulators (after post-ops) - convert F32→S32 then clamp to
            // U8
            vcvtps2dq(Zmm(xBaseIdx), Zmm(accumBaseIdx + i));
            vpmaxsd(Zmm(xBaseIdx), Zmm(xBaseIdx), Zmm(xBaseIdx + 1));
            vpminsd(Zmm(xBaseIdx), Zmm(xBaseIdx), Zmm(xBaseIdx + 2));
            vpmovdb(ptr[regTmpYptr + i * 16], Zmm(xBaseIdx));
        } else {
            // S32 accumulators (no post-ops) - direct clamp to U8
            vpmaxsd(Zmm(xBaseIdx), Zmm(accumBaseIdx + i), Zmm(xBaseIdx + 1));
            vpminsd(Zmm(xBaseIdx), Zmm(xBaseIdx), Zmm(xBaseIdx + 2));
            vpmovdb(ptr[regTmpYptr + i * 16], Zmm(xBaseIdx));
        }
    }

    if (n_left) {
        if (accumulatorsAreF32) {
            // F32 accumulators - convert F32→S32 then clamp to U8
            vcvtps2dq(Zmm(xBaseIdx), Zmm(accumBaseIdx + n_iter));
            vpmaxsd(Zmm(xBaseIdx), Zmm(xBaseIdx), Zmm(xBaseIdx + 1));
            vpminsd(Zmm(xBaseIdx), Zmm(xBaseIdx), Zmm(xBaseIdx + 2));
            vpmovdb(ptr[regTmpYptr + n_iter * 16] | mask_regs[0],
                    Zmm(xBaseIdx));
        } else {
            // S32 accumulators - direct clamp to U8
            vpmaxsd(Zmm(xBaseIdx), Zmm(accumBaseIdx + n_iter),
                    Zmm(xBaseIdx + 1));
            vpminsd(Zmm(xBaseIdx), Zmm(xBaseIdx), Zmm(xBaseIdx + 2));
            vpmovdb(ptr[regTmpYptr + n_iter * 16] | mask_regs[0],
                    Zmm(xBaseIdx));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::storeYValuesFringe_S8()
{
    updateYBufferPointers();

    int n_iter = N_LEFT / nElemsPerReg;
    int n_left = N_LEFT % nElemsPerReg;

    for (int i = 0; i < n_iter; i++) {
        if (accumulatorsAreF32) {
            // F32 accumulators (after post-ops) - convert F32→S32 then saturate
            // to S8
            vcvtps2dq(Zmm(xBaseIdx), Zmm(accumBaseIdx + i));
            vpmovsdb(ptr[regTmpYptr + i * 16], Zmm(xBaseIdx));
        } else {
            // S32 accumulators (no post-ops) - direct saturation to S8
            vpmovsdb(ptr[regTmpYptr + i * 16], Zmm(accumBaseIdx + i));
        }
    }

    if (n_left) {
        if (accumulatorsAreF32) {
            // F32 accumulators - convert F32→S32 then saturate to S8
            vcvtps2dq(Zmm(xBaseIdx), Zmm(accumBaseIdx + n_iter));
            vpmovsdb(ptr[regTmpYptr + n_iter * 16] | mask_regs[0],
                     Zmm(xBaseIdx));
        } else {
            // S32 accumulators - direct saturation to S8
            vpmovsdb(ptr[regTmpYptr + n_iter * 16] | mask_regs[0],
                     Zmm(accumBaseIdx + n_iter));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::storeYValuesFringe_F32()
{
    updateYBufferPointers();

    int n_iter = N_LEFT / nElemsPerReg;
    int n_left = N_LEFT % nElemsPerReg;

    for (int i = 0; i < n_iter; i++) {
        if (accumulatorsAreF32) {
            // F32 accumulators (after post-ops) - direct store
            vmovups(ptr[regTmpYptr + i * RegBytes], Zmm(accumBaseIdx + i));
        } else {
            // S32 accumulators (no post-ops) - convert to F32 and store
            vcvtdq2ps(Zmm(xBaseIdx), Zmm(accumBaseIdx + i));
            vmovups(ptr[regTmpYptr + i * RegBytes], Zmm(xBaseIdx));
        }
    }

    if (n_left) {
        if (accumulatorsAreF32) {
            // F32 accumulators - direct store
            vmovups(ptr[regTmpYptr + n_iter * RegBytes] | mask_regs[0],
                    Zmm(accumBaseIdx + n_iter));
        } else {
            // S32 accumulators - convert to F32 and store
            vcvtdq2ps(Zmm(xBaseIdx), Zmm(accumBaseIdx + n_iter));
            vmovups(ptr[regTmpYptr + n_iter * RegBytes] | mask_regs[0],
                    Zmm(xBaseIdx));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::storeYValuesFringe_BF16()
{
    updateYBufferPointers();

    int n_iter = N_LEFT / nElemsPerReg;
    int n_left = N_LEFT % nElemsPerReg;

    for (int i = 0; i < n_iter; i++) {
        if (accumulatorsAreF32) {
            // F32 accumulators (after post-ops) - direct F32→BF16 conversion
            vmovups(Zmm(xBaseIdx), Zmm(accumBaseIdx + i));
        } else {
            // S32 accumulators (no post-ops) - convert S32→F32 first
            vcvtdq2ps(Zmm(xBaseIdx), Zmm(accumBaseIdx + i));
        }

        // Software BF16 conversion
        // Algorithm: bf16 = (f32 + 0x00007FFF + ((f32 >> 16) & 1)) >> 16

        // Extract LSB of bit 16 for rounding (ties-to-even)
        vpsrld(Zmm(xBaseIdx + 2), Zmm(xBaseIdx), 16);
        vpandd(Zmm(xBaseIdx + 2), Zmm(xBaseIdx + 2),
               ptr[rip + label_bf16_lsb_mask]);

        // Add rounding bias (0x00007FFF) + LSB
        vpaddd(Zmm(xBaseIdx + 1), Zmm(xBaseIdx),
               ptr[rip + label_bf16_round_bias]);
        vpaddd(Zmm(xBaseIdx + 1), Zmm(xBaseIdx + 1), Zmm(xBaseIdx + 2));

        // Shift right 16 bits to get BF16 in lower 16 bits of each dword
        vpsrld(Zmm(xBaseIdx + 1), Zmm(xBaseIdx + 1), 16);

        // Pack 16x32-bit to 16x16-bit: use vpmovdw to convert dword to word
        vpmovdw(Ymm(xBaseIdx + 1), Zmm(xBaseIdx + 1));

        vmovdqu16(ptr[regTmpYptr + i * 32], Ymm(xBaseIdx + 1));
    }

    if (n_left) {
        if (accumulatorsAreF32) {
            // F32 accumulators - direct use
            vmovups(Zmm(xBaseIdx), Zmm(accumBaseIdx + n_iter));
        } else {
            // S32 accumulators - convert to F32 first
            vcvtdq2ps(Zmm(xBaseIdx), Zmm(accumBaseIdx + n_iter));
        }

        // Software BF16 conversion for masked case
        vpsrld(Zmm(xBaseIdx + 2), Zmm(xBaseIdx), 16);
        vpandd(Zmm(xBaseIdx + 2), Zmm(xBaseIdx + 2),
               ptr[rip + label_bf16_lsb_mask]);
        vpaddd(Zmm(xBaseIdx + 1), Zmm(xBaseIdx),
               ptr[rip + label_bf16_round_bias]);
        vpaddd(Zmm(xBaseIdx + 1), Zmm(xBaseIdx + 1), Zmm(xBaseIdx + 2));
        vpsrld(Zmm(xBaseIdx + 1), Zmm(xBaseIdx + 1), 16);
        vpmovdw(Ymm(xBaseIdx + 1), Zmm(xBaseIdx + 1));

        vmovdqu16(ptr[regTmpYptr + n_iter * 32] | mask_regs[0],
                  Ymm(xBaseIdx + 1));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMVM1<KType>::generateKernel(utils::gemvM1GeneratorParams& params)
{
    // Initialize kernel ops handler if post-ops are present
    if (!params.kernelOps.empty()) {
        kernelOpsHandlerPtr =
            std::make_unique<gen::kernelOpsHandler>(this, params.kType);
        kernelOpsVector = params.kernelOps; // Store kernelOps for later use
    }

    // Putting inside a scope so that constant data can be generated post
    // the ret instr. StackFrame inserts a ret instr in its destructor.
    {
        // Using Xbyak's utility for managing the stack frame
        Xbyak::util::StackFrame frame(this, 1, 13, 0);
        initializeStackFrame(frame);

        // Initializing the parameters
        initializeParameters(params);

        // Allocating valid ranges for register usage
        RETURN_IF_ERROR(allocateRegisters());

        // Reset accumulator type flag for each kernel generation
        accumulatorsAreF32 = false;

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

            // Zero out accumulator registers for this n iteration
            regInit(accumBaseIdx, accumReg);
            xor_(regIncK,
                 regIncK); // regIncK is used to increment
                           // the pointer for K dimension(zeroed before the
                           // kloop)

            // K-loop is not needed if alpha is zero
            if (params.alphaScalingType
                != dlp::kernel_frame::scalingType::zero) {
                // Vector x
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

                    mov(regPsB, KC);
                    // lea(regPsB, ptr[regPsB * sizeof(int32_t)]);
                    mov(regTmpYptr, ptr[stackPtr
                                        + offsetof(dlp::kernels::gemvM1Params,
                                                   jc_cur_loop_rem)]);
                    mov(regTmp2, ptr[stackPtr
                                     + offsetof(dlp::kernels::gemvM1Params,
                                                n_sub_updated)]);
                    imul(regTmpYptr, regPsB);
                    imul(regTmp2, regIncK);
                    // lea(regTmp2, ptr[regTmp2 * sizeof(int32_t)]);

                    lea(regBptr, ptr[regBptr + regTmpYptr]);
                    lea(regBptr, ptr[regBptr + regTmp2]);

                    // Set the base pointer for the iteration
                    mov(regTmp2, regIncN);
                    imul(regTmp2, regPsB);

                    add(regBptr, regTmp2);

                    // This is a sub-loop over the k-dimension
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
                        ptr[stackPtr
                            + offsetof(dlp::kernels::gemvM1Params, b)]);

                    mov(regPsB,
                        ptr[stackPtr
                            + offsetof(dlp::kernels::gemvM1Params, psB)]);
                    mov(regTmpYptr, ptr[stackPtr
                                        + offsetof(dlp::kernels::gemvM1Params,
                                                   jc_cur_loop_rem)]);
                    mov(regTmp2, ptr[stackPtr
                                     + offsetof(dlp::kernels::gemvM1Params,
                                                n_sub_updated)]);
                    imul(regTmpYptr, regPsB);
                    imul(regTmp2, regIncK);

                    lea(regBptr, ptr[regBptr + regTmpYptr]);
                    lea(regBptr, ptr[regBptr + regTmp2]);

                    mov(regTmp2, regIncN);
                    imul(regTmp2, regPsB);

                    add(regBptr, regTmp2);

                    loopKSubIter(true, false);
                }

                L(label_n_loop_k_fringe_end);

                // Final accumulation of the result
                finalAccumulate();

                // Scale with alpha
                scaleWithAlpha();
            }

            // Scale the result by beta, and store it accordingly
            scaleYWithBeta(false);

            // Post-ops integration for GEMV M=1
            if (kernelOpsHandlerPtr) {
                // Convert S32 accumulators to F32 for post-ops compatibility
                for (int i = 0; i < NR / nElemsPerReg; i++) {
                    vcvtdq2ps(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i));
                }

                // Mark that accumulators are now F32 for store functions
                accumulatorsAreF32 = true;

                RETURN_IF_ERROR(kernelOpsHandlerPtr->generateKernelOps(
                    kernelOpsVector, stackPtr, dlp::jit::jitAlgoType::gemv_m1,
                    1, NR, false, 1, accumBaseIdx, NR / nElemsPerReg));

                kernelOpsHandlerPtr->generateKernelOpsAttributes();
            }

            storeYValues(false);

            // Update the pointers for next n iteration
            mov(regTmp2, NR);
            add(regIncN, regTmp2);
            lea(regYptr, ptr[regYptr + regTmp2 * sizeof(int32_t)]);

            dec(regNIter);
            jnz(label_n_loop_start, T_NEAR);
        }

        L(label_n_loop_end);
        if (params.nfringe) {
            mov(regNIter,
                ptr[stackPtr + offsetof(dlp::kernels::gemvM1Params, n_left)]);
            test(regNIter, regNIter);
            jz(label_n_fringe_end, T_NEAR);

            // The RowStride needs to be adjusted based on the N_LEFT since the
            // packing ensures the N_LEFT panel to be padded
            if (N_LEFT < 32)
                mov(regRsB, 64);
            else if (N_LEFT < 48)
                mov(regRsB, 128);
            else if (N_LEFT < 64)
                mov(regRsB, 192);

            L(label_n_fringe_start);
            // Zero out accumulator registers for this n iteration
            regInit(accumBaseIdx, accumReg);
            xor_(regIncK, regIncK);

            // K-loop is not needed if alpha is zero
            if (params.alphaScalingType
                != dlp::kernel_frame::scalingType::zero) {
                // Vector x
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

                    mov(regPsB, KC);
                    // lea(regPsB, ptr[regPsB * sizeof(int32_t)]);
                    mov(regTmpYptr, ptr[stackPtr
                                        + offsetof(dlp::kernels::gemvM1Params,
                                                   jc_cur_loop_rem)]);
                    mov(regTmp2, ptr[stackPtr
                                     + offsetof(dlp::kernels::gemvM1Params,
                                                n_sub_updated)]);

                    imul(regTmpYptr, regPsB);
                    imul(regTmp2, regIncK);
                    // lea(regTmp2, ptr[regTmp2 * sizeof(int32_t)]);

                    lea(regBptr, ptr[regBptr + regTmpYptr]);
                    lea(regBptr, ptr[regBptr + regTmp2]);

                    mov(regTmp2, regIncN);
                    imul(regTmp2, regPsB);

                    add(regBptr, regTmp2);

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
                        ptr[stackPtr
                            + offsetof(dlp::kernels::gemvM1Params, b)]);

                    mov(regPsB,
                        ptr[stackPtr
                            + offsetof(dlp::kernels::gemvM1Params, psB)]);
                    // lea(regPsB, ptr[regPsB * sizeof(int32_t)]);
                    mov(regTmpYptr, ptr[stackPtr
                                        + offsetof(dlp::kernels::gemvM1Params,
                                                   jc_cur_loop_rem)]);
                    mov(regTmp2, ptr[stackPtr
                                     + offsetof(dlp::kernels::gemvM1Params,
                                                n_sub_updated)]);
                    imul(regTmpYptr, regPsB);
                    imul(regTmp2, regIncK);
                    // lea(regTmp2, ptr[regTmp2 * sizeof(int32_t)]);

                    lea(regBptr, ptr[regBptr + regTmpYptr]);
                    lea(regBptr, ptr[regBptr + regTmp2]);

                    mov(regTmp2, regIncN);
                    imul(regTmp2, regPsB);

                    add(regBptr, regTmp2);

                    loopKSubIter(true, true);
                }

                L(label_n_fringe_k_fringe_end);

                // Final accumulation of the result
                finalAccumulate();

                // Scale with alpha
                scaleWithAlpha();
            }

            // Scale the result by beta, and store it accordingly
            scaleYWithBeta(true);

            // Post-ops integration for GEMV M=1 n-fringe
            if (kernelOpsHandlerPtr) {
                // Convert S32 accumulators to F32 for post-ops compatibility
                for (int i = 0; i < (N_LEFT + nElemsPerReg - 1) / nElemsPerReg;
                     i++) {
                    vcvtdq2ps(Zmm(accumBaseIdx + i), Zmm(accumBaseIdx + i));
                }

                // Mark that accumulators are now F32 for store functions
                accumulatorsAreF32 = true;

                // For n-fringe, use proper masking based on N_LEFT
                RETURN_IF_ERROR(kernelOpsHandlerPtr->generateKernelOps(
                    kernelOpsVector, stackPtr, dlp::jit::jitAlgoType::gemv_m1,
                    1, N_LEFT, true, 1, accumBaseIdx, N_LEFT / nElemsPerReg));

                kernelOpsHandlerPtr->generateKernelOpsAttributes();
            }

            storeYValues(true);
        }

        L(label_n_fringe_end);
        outLocalLabel();
    } // StackFrame destructor inserts 'ret' here

    // Generate constant data tables after the return instruction
    generateConstantData();

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitU8S8VNNI_GEMVN1<KType>::generateConstantData()
{
    // Constant data for BF16 conversion (accessed via RIP-relative addressing)
    // These are placed after the function return, so they won't be executed

    if (c_downscale == DLP_BF16) {
        align(16);
        L(label_bf16_round_bias);
        for (int i = 0; i < 16; i++)
            dd(0x00007FFF);
        L(label_bf16_lsb_mask);
        for (int i = 0; i < 16; i++)
            dd(0x00000001);
    }
}

template<utils::kernelInstrType KType>
void
jitU8S8VNNI_GEMVM1<KType>::generateConstantData()
{
    // Constant data for BF16 conversion (accessed via RIP-relative addressing)
    // These are placed after the function return, so they won't be executed

    if (c_downscale == DLP_BF16) {
        align(16);
        L(label_bf16_round_bias);
        for (int i = 0; i < 16; i++)
            dd(0x00007FFF);
        L(label_bf16_lsb_mask);
        for (int i = 0; i < 16; i++)
            dd(0x00000001);
    }
}

} // namespace amdzen::gen

// Explicit template instantiation
template class amdzen::gen::jitU8S8VNNI_GEMVN1<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
template class amdzen::gen::jitU8S8VNNI_GEMVM1<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
