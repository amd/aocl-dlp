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

#include "u8s8_gemm_generator.hh"

namespace amdzen::gen {

using namespace Xbyak;

template<utils::kernelInstrType KType>
jitU8S8VNNI_GEMM<KType>::jitU8S8VNNI_GEMM(void* buffer, size_t bufferSize)
    : Xbyak::CodeGenerator(bufferSize, buffer)
{
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::allocateRegisters()
{
    // check if MR, NR are valid
    if (MR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
    // For u8s8 VNNI: each register holds RegBytes/4 VNNI groups (4 int8s each)
    // But we work with int32 accumulator, so calculate based on int32 elements
    int nElemsPerReg = RegBytes / sizeof(int32_t);
    bFullReg         = ((NR) / nElemsPerReg);
    // useMask is set from generation parameters, not calculated here
    bMaskReg = (useMask ? 1 : 0);
    bReg     = bFullReg + bMaskReg;
    cReg     = MR * bReg;

    // Calculate available A registers
    aReg = numRegs - cReg - bReg;

    // Check if we have enough registers
    if (aReg < 1) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Register index assignment
    aRegIdx = 0;           // A registers start at index 0
    bRegIdx = aReg;        // B registers follow A registers
    cRegIdx = aReg + bReg; // C registers follow B registers

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitU8S8VNNI_GEMM<KType>::initializeParameters(bool mLoop)
{
    mov(regTmpAptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);
    if (mLoop) {
        mov(regAPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);
        mov(regMiter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, mIter)]);
        mov(regTmp3, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, psA)]);
    }

    // Load post_op_c_i for downscale buffer addressing
    mov(regTmp2,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(lpgemm_post_op_attr, post_op_c_i)]);

    // Initialize parameter pointers from gemmParams structure
    mov(regCPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, c)]);
    mov(regRsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsA)]);
    mov(regCsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, csA)]);
    mov(regRsB, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsB)]);
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsC)]);

    // Load k-fringe mask to k2
    kmovw(k2, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeftmask)]);

    // Scale strides for VNNI format and data types
    lea(regRsC, ptr[regRsC * sizeof(int32_t)]);

    mov(regTmpCptr, regCPtr);

    if (useMask) {
        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            // For AVX512 ZMM: load 16-bit mask for int32 elements
            kmovw(k3,
                  ptr[stackPtr + offsetof(dlp::kernels::gemmParams, maskS32)]);
        }
    }
}

template<utils::kernelInstrType KType>
void
jitU8S8VNNI_GEMM<KType>::initializeAccumulators(utils::generatorParams& params)
{

    // Zero out accumulator registers for int32 results
    if constexpr (Traits::isAVX512) {
        vpxord(RegType(cRegIdx), RegType(cRegIdx), RegType(cRegIdx));
    }

    for (int i = 1; i < cReg; i++) {
        vmovdqa32(RegType(cRegIdx + i), RegType(cRegIdx));
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::generateIrLoop(utils::generatorParams& params)
{
    initializeAccumulators(params);

    inLocalLabel();

    if (params.alphaScalingType != dlp::kernel_frame::scalingType::zero) {
        // Load B matrix pointer
        mov(regBptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, b)]);

        // Zero out accumulator registers

        // Generate K-loop with proper error handling
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kIterBP)]);
        test(regKIter, regKIter);
        je(".BCONSIDKLEFT", T_NEAR);

        // Main unrolled K-loop
        L(".BLOOPKITER");
        RETURN_IF_ERROR(kUnroll(params.K_UNROLL, false));
        dec(regKIter);
        jne(".BLOOPKITER", T_NEAR);

        L(".BCONSIDKLEFT");
        // Handle remaining K iterations
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeft)]);
        test(regKIter, regKIter);
        je(".BPOSTACCUM", T_NEAR);

        RETURN_IF_ERROR(kUnroll(1, true));

        L(".BPOSTACCUM");
    }
    // Use consolidated post-ops
    RETURN_IF_ERROR(generatePostOps(params));

    vzeroupper();
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::loadBValues()
{

    // Load B matrix values (int8 in VNNI format)
    for (int i = 0; i < bFullReg; i++) {
        // Add memory alignment check for AVX-512 (64-byte alignment preferred)
        if constexpr (Traits::isAVX512) {
            vmovdqu32(RegType(bRegIdx + i), ptr[regBptr + i * RegBytes]);
        }
    }

    if (useMask) {
        int maskRegIndex = bRegIdx + bFullReg;
        if (maskRegIndex >= numRegs) {
            return dlp::jit::jitGeneratorError::badKernelInfo;
        }

        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            // AVX-512: Load B values for masked register
            // Note: Use correct offset when there are full registers before
            // the masked register
            vmovdqu8(RegType(maskRegIndex), ptr[regBptr + bFullReg * RegBytes]);
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::storeResult()
{
    // Dispatch to output-specific store function based on c_downscale parameter
    mov(regTmpCptr, regCPtr);
    if (c_downscale == DLP_S32) {
        return storeResultS32();
    } else {
        // Use local labels to avoid redefinition when storeResult is called
        // multiple times (e.g., once for F32 path and once for S32 path)
        inLocalLabel();

        // Check is_last_k and call the downscale
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, is_last_k)]);
        test(regTmp1, regTmp1);
        je(".STORE_S32", T_NEAR);

        if (c_downscale == DLP_S8) {
            storeResultS8();
            jmp(".END_STORE", T_NEAR);
        } else if (c_downscale == DLP_U8) {
            storeResultU8();
            jmp(".END_STORE", T_NEAR);
        } else if (c_downscale == DLP_F32) {
            storeResultF32();
            jmp(".END_STORE", T_NEAR);
        } else if (c_downscale == DLP_BF16) {
            storeResultBF16();
            jmp(".END_STORE", T_NEAR);
        } else {
            outLocalLabel();
            return dlp::jit::jitGeneratorError::badKernelInfo;
        }

        L(".STORE_S32");
        storeResultS32();
        L(".END_STORE");

        outLocalLabel();
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::storeResultS32()
{
    // Store int32 accumulator results
    // Handle both S32 and F32 accumulators (F32 when post-ops are present)
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < bFullReg; j++) {
            if (accumulatorsAreF32) {
                // F32 accumulators (after post-ops) - convert to S32 before
                // storing
                vcvtps2dq(Zmm(aRegIdx), Zmm(cRegIdx + i * bReg + j));
                vmovdqu32(ptr[regTmpCptr + j * RegBytes], Zmm(aRegIdx));
            } else {
                // S32 accumulators (no post-ops) - direct store
                vmovdqu32(ptr[regTmpCptr + j * RegBytes],
                          RegType(cRegIdx + i * bReg + j));
            }
        }
        if (bMaskReg > 0) {
            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                if (accumulatorsAreF32) {
                    // F32 accumulators - convert to S32 before storing
                    vcvtps2dq(Zmm(aRegIdx), Zmm(cRegIdx + i * bReg + bFullReg));
                    vmovdqu32(ptr[regTmpCptr + bFullReg * RegBytes] | k3 | T_z,
                              Zmm(aRegIdx));
                } else {
                    // S32 accumulators - direct store
                    vmovdqu32(ptr[regTmpCptr + bFullReg * RegBytes] | k3 | T_z,
                              RegType(cRegIdx + i * bReg + bFullReg));
                }
            }
        }
        add(regTmpCptr, regRsC);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::storeResultS8()
{

    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        // Store int8 results with signed saturation
        updateCBufferPointers();
        for (int i = 0; i < MR; i++) {
            for (int j = 0; j < bFullReg; j++) {
                if (accumulatorsAreF32) {
                    // F32 accumulators (after post-ops) - convert F32→S32 then
                    // saturate to S8
                    vcvtps2dq(Zmm(aRegIdx), Zmm(cRegIdx + i * bReg + j));
                    vpmovsdb(ptr[regTmpCptr + j * 16], Zmm(aRegIdx));
                } else {
                    // S32 accumulators (no post-ops) - direct saturation to S8
                    vpmovsdb(ptr[regTmpCptr + j * 16],
                             RegType(cRegIdx + i * bReg + j));
                }
            }
            if (bMaskReg > 0) {
                if (accumulatorsAreF32) {
                    // F32 accumulators - convert F32→S32 then saturate to S8
                    vcvtps2dq(Zmm(aRegIdx), Zmm(cRegIdx + i * bReg + bFullReg));
                    vpmovsdb(ptr[regTmpCptr + bFullReg * 16] | k3,
                             Zmm(aRegIdx));
                } else {
                    // S32 accumulators - direct saturation to S8
                    vpmovsdb(ptr[regTmpCptr + bFullReg * 16] | k3,
                             RegType(cRegIdx + i * bReg + bFullReg));
                }
            }
            add(regTmpCptr, regTmp1);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::storeResultU8()
{

    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        // Store uint8 results with unsigned clipping
        updateCBufferPointers();

        // Prepare constants for clamping values to [0, 255]
        vpxord(Zmm(aRegIdx + 1), Zmm(aRegIdx + 1), Zmm(aRegIdx + 1)); // 0
        mov(regKIter, 255);
        vpbroadcastd(Zmm(aRegIdx + 2), regKIter.cvt32()); // 255

        for (int i = 0; i < MR; i++) {
            for (int j = 0; j < bFullReg; j++) {
                if (accumulatorsAreF32) {
                    // F32 accumulators (after post-ops) - convert F32→S32 then
                    // clamp to U8
                    vcvtps2dq(Zmm(aRegIdx), Zmm(cRegIdx + i * bReg + j));
                    vpmaxsd(Zmm(aRegIdx), Zmm(aRegIdx),
                            Zmm(aRegIdx + 1)); // max(s32, 0)
                    vpminsd(Zmm(aRegIdx), Zmm(aRegIdx),
                            Zmm(aRegIdx + 2)); // min(s32, 255)
                    vpmovdb(ptr[regTmpCptr + j * 16], Zmm(aRegIdx));
                } else {
                    // S32 accumulators (no post-ops) - direct clamp to U8
                    vpmaxsd(Zmm(aRegIdx), Zmm(cRegIdx + i * bReg + j),
                            Zmm(aRegIdx + 1)); // max(s32, 0)
                    vpminsd(Zmm(aRegIdx), Zmm(aRegIdx),
                            Zmm(aRegIdx + 2)); // min(s32, 255)
                    vpmovdb(ptr[regTmpCptr + j * 16], Zmm(aRegIdx));
                }
            }
            if (bMaskReg > 0) {
                if (accumulatorsAreF32) {
                    // F32 accumulators - convert F32→S32 then clamp to U8
                    vcvtps2dq(Zmm(aRegIdx), Zmm(cRegIdx + i * bReg + bFullReg));
                    vpmaxsd(Zmm(aRegIdx), Zmm(aRegIdx), Zmm(aRegIdx + 1));
                    vpminsd(Zmm(aRegIdx), Zmm(aRegIdx), Zmm(aRegIdx + 2));
                    vpmovdb(ptr[regTmpCptr + bFullReg * 16] | k3, Zmm(aRegIdx));
                } else {
                    // S32 accumulators - direct clamp to U8
                    vpmaxsd(Zmm(aRegIdx), Zmm(cRegIdx + i * bReg + bFullReg),
                            Zmm(aRegIdx + 1));
                    vpminsd(Zmm(aRegIdx), Zmm(aRegIdx), Zmm(aRegIdx + 2));
                    vpmovdb(ptr[regTmpCptr + bFullReg * 16] | k3, Zmm(aRegIdx));
                }
            }
            add(regTmpCptr, regTmp1);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::storeResultF32()
{
    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {

        updateCBufferPointers();

        // Store float32 results
        for (int i = 0; i < MR; i++) {
            for (int j = 0; j < bFullReg; j++) {
                if (accumulatorsAreF32) {
                    // F32 accumulators (after post-ops) - direct store
                    vmovups(ptr[regTmpCptr + j * RegBytes],
                            Zmm(cRegIdx + i * bReg + j));
                } else {
                    // S32 accumulators (no post-ops) - convert to F32 and store
                    vcvtdq2ps(Zmm(aRegIdx), Zmm(cRegIdx + i * bReg + j));
                    vmovups(ptr[regTmpCptr + j * RegBytes], Zmm(aRegIdx));
                }
            }
            if (bMaskReg > 0) {
                if (accumulatorsAreF32) {
                    // F32 accumulators - direct masked store
                    vmovups(ptr[regTmpCptr + bFullReg * RegBytes] | k3,
                            Zmm(cRegIdx + i * bReg + bFullReg));
                } else {
                    // S32 accumulators - convert to F32 and store
                    vcvtdq2ps(Zmm(aRegIdx), Zmm(cRegIdx + i * bReg + bFullReg));
                    vmovups(ptr[regTmpCptr + bFullReg * RegBytes] | k3,
                            Zmm(aRegIdx));
                }
            }
            add(regTmpCptr, regTmp1);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::storeResultBF16()
{

    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        // Store bfloat16 results
        updateCBufferPointers();

        for (int i = 0; i < MR; i++) {
            for (int j = 0; j < bFullReg; j++) {
                // Handle both S32 and F32 accumulators
                if (accumulatorsAreF32) {
                    // F32 accumulators (after post-ops) - direct F32→BF16
                    // conversion
                    vmovups(Zmm(aRegIdx), Zmm(cRegIdx + i * bReg + j));
                } else {
                    // S32 accumulators (no post-ops) - convert S32→F32 first
                    vcvtdq2ps(Zmm(aRegIdx), Zmm(cRegIdx + i * bReg + j));
                }

                // Convert F32 to BF16 using software algorithm
                // Algorithm: bf16 = (f32 + 0x00007FFF + ((f32 >> 16) & 1)) >>
                // 16

                // Extract LSB of bit 16 for rounding (ties-to-even)
                vpsrld(Zmm(aRegIdx + 2), Zmm(aRegIdx), 16); // Shift right 16
                vpandd(Zmm(aRegIdx + 2), Zmm(aRegIdx + 2),
                       ptr[rip + label_bf16_lsb_mask]); // Keep only bit 0 (LSB)

                // Add rounding bias (0x00007FFF) + LSB
                vpaddd(Zmm(aRegIdx + 1), Zmm(aRegIdx),
                       ptr[rip + label_bf16_round_bias]);
                vpaddd(Zmm(aRegIdx + 1), Zmm(aRegIdx + 1), Zmm(aRegIdx + 2));

                // Shift right 16 bits to get BF16 in lower 16 bits of each
                // dword
                vpsrld(Zmm(aRegIdx + 1), Zmm(aRegIdx + 1), 16);

                // Pack 16x32-bit to 16x16-bit: use vpmovdw to convert dword to
                // word
                vpmovdw(Ymm(aRegIdx + 1), Zmm(aRegIdx + 1));

                // Store 32 bytes (16 bfloat16 values)
                vmovdqu16(ptr[regTmpCptr + j * 32], Ymm(aRegIdx + 1));
            }
            if (bMaskReg > 0) {
                // Masked store for remainder elements
                if (accumulatorsAreF32) {
                    // F32 accumulators - direct use
                    vmovups(Zmm(aRegIdx), Zmm(cRegIdx + i * bReg + bFullReg));
                } else {
                    // S32 accumulators - convert to F32 first
                    vcvtdq2ps(Zmm(aRegIdx), Zmm(cRegIdx + i * bReg + bFullReg));
                }

                // Software BF16 conversion for masked case
                vpsrld(Zmm(aRegIdx + 2), Zmm(aRegIdx), 16);
                vpandd(Zmm(aRegIdx + 2), Zmm(aRegIdx + 2),
                       ptr[rip + label_bf16_lsb_mask]);
                vpaddd(Zmm(aRegIdx + 1), Zmm(aRegIdx),
                       ptr[rip + label_bf16_round_bias]);
                vpaddd(Zmm(aRegIdx + 1), Zmm(aRegIdx + 1), Zmm(aRegIdx + 2));
                vpsrld(Zmm(aRegIdx + 1), Zmm(aRegIdx + 1), 16);
                vpmovdw(Ymm(aRegIdx + 1), Zmm(aRegIdx + 1));

                vmovdqu16(ptr[regTmpCptr + bFullReg * 32] | k3,
                          Ymm(aRegIdx + 1));
            }
            add(regTmpCptr, regTmp1);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::scaleAlpha()
{
    int alphaRegIdx = aRegIdx;

    // Load alpha scaling factor
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, alpha)]);
    vpbroadcastd(Zmm(alphaRegIdx), ptr[regTmp1]);

    // Scale all accumulator registers with alpha
    for (int i = 0; i < cReg; i++) {
        vpmulld(Zmm(cRegIdx + i), Zmm(cRegIdx + i), Zmm(alphaRegIdx));
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitU8S8VNNI_GEMM<KType>::updateCBufferPointers()
{
    mov(regTmpCptr,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(lpgemm_post_op_attr, buf_downscale)]);

    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(lpgemm_post_op_attr, post_op_c_j)]);

    if (c_downscale == DLP_BF16) {
        lea(regTmp1, ptr[regTmp1 * 2]);
    } else if (c_downscale == DLP_F32) {
        lea(regTmp1, ptr[regTmp1 * 4]);
    }

    add(regTmpCptr, regTmp1);

    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(lpgemm_post_op_attr, rs_c_downscale)]);

    if (c_downscale == DLP_BF16) {
        lea(regTmp1, ptr[regTmp1 * 2]);
    } else if (c_downscale == DLP_F32) {
        lea(regTmp1, ptr[regTmp1 * 4]);
    }

    mov(regKIter, regTmp2);
    imul(regKIter, regTmp1); // post_ops_c_i * rs_c_downscale
    add(regTmpCptr, regKIter);
    // regTmp1 now contains rs_c_downscale for caller to use
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::scaleBeta()
{
    int betaRegIdx = aRegIdx;

    // Load beta scaling factor
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, beta)]);
    vpbroadcastd(Zmm(betaRegIdx), ptr[regTmp1]);
    mov(regTmpCptr, regCPtr);

    if (c_downscale == DLP_U8) {
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, is_first_k)]);
        test(regTmp1, regTmp1);
        je("BETA_S32",
           T_NEAR); // NOT first_k → accumulate from S32 buffer (beta=1)

        updateCBufferPointers();

        for (int i = 0; i < MR; i++) {
            for (int j = 0; j < bFullReg; j++) {
                vmovdqu8(Xmm(bRegIdx + j), ptr[regTmpCptr + j * 16]);
                vpmovzxbd(Zmm(bRegIdx + j), Xmm(bRegIdx + j));
                vpdpbusd(Zmm(cRegIdx + i * bReg + j), Zmm(bRegIdx + j),
                         Zmm(betaRegIdx));
            }

            // Handle masked beta scaling
            if (bMaskReg > 0) {
                vmovdqu8(Xmm(bRegIdx + bFullReg) | k3 | T_z,
                         ptr[regTmpCptr + bFullReg * 16]);
                vpmovzxbd(Zmm(bRegIdx + bFullReg), Xmm(bRegIdx + bFullReg));
                vpdpbusd(Zmm(cRegIdx + i * bReg + bFullReg),
                         Zmm(bRegIdx + bFullReg), Zmm(betaRegIdx));
            }

            add(regTmpCptr, regTmp1);
        }

        jmp("BETA_END", T_NEAR);
        L("BETA_S32");
    } else if (c_downscale == DLP_S8) {
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, is_first_k)]);
        test(regTmp1, regTmp1);
        je("BETA_S32",
           T_NEAR); // NOT first_k → accumulate from S32 buffer (beta=1)

        updateCBufferPointers();

        for (int i = 0; i < MR; i++) {
            for (int j = 0; j < bFullReg; j++) {
                vmovdqu8(Xmm(bRegIdx + j), ptr[regTmpCptr + j * 16]);
                vpmovsxbd(Zmm(bRegIdx + j), Xmm(bRegIdx + j));
                vpmulld(Zmm(bRegIdx + j), Zmm(bRegIdx + j), Zmm(betaRegIdx));
                vpaddd(Zmm(cRegIdx + i * bReg + j), Zmm(cRegIdx + i * bReg + j),
                       Zmm(bRegIdx + j));
            }
            if (bMaskReg > 0) {
                vmovdqu8(Xmm(bRegIdx + bFullReg) | k3 | T_z,
                         ptr[regTmpCptr + bFullReg * 16]);
                vpmovsxbd(Zmm(bRegIdx + bFullReg), Xmm(bRegIdx + bFullReg));
                vpmulld(Zmm(bRegIdx + bFullReg), Zmm(bRegIdx + bFullReg),
                        Zmm(betaRegIdx));
                vpaddd(Zmm(cRegIdx + i * bReg + bFullReg),
                       Zmm(cRegIdx + i * bReg + bFullReg),
                       Zmm(bRegIdx + bFullReg));
            }
            add(regTmpCptr, regTmp1);
        }
        jmp("BETA_END", T_NEAR);
        L("BETA_S32");
    } else if (c_downscale == DLP_F32) {
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, is_first_k)]);
        test(regTmp1, regTmp1);
        je("BETA_S32",
           T_NEAR); // NOT first_k → accumulate from S32 buffer (beta=1)

        updateCBufferPointers();

        for (int i = 0; i < MR; i++) {
            for (int j = 0; j < bFullReg; j++) {
                vcvtps2dq(Zmm(bRegIdx + j), ptr[regTmpCptr + j * RegBytes]);
                vpmulld(Zmm(bRegIdx + j), Zmm(bRegIdx + j), Zmm(betaRegIdx));
                vpaddd(Zmm(cRegIdx + i * bReg + j), Zmm(cRegIdx + i * bReg + j),
                       Zmm(bRegIdx + j));
            }
            if (bMaskReg > 0) {
                vcvtps2dq(Zmm(bRegIdx + bFullReg) | k3 | T_z,
                          ptr[regTmpCptr + bFullReg * RegBytes]);
                vpmulld(Zmm(bRegIdx + bFullReg), Zmm(bRegIdx + bFullReg),
                        Zmm(betaRegIdx));
                vpaddd(Zmm(cRegIdx + i * bReg + bFullReg),
                       Zmm(cRegIdx + i * bReg + bFullReg),
                       Zmm(bRegIdx + bFullReg));
            }
            add(regTmpCptr, regTmp1);
        }

        jmp("BETA_END", T_NEAR);
        L("BETA_S32");
    } else if (c_downscale == DLP_BF16) {
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, is_first_k)]);
        test(regTmp1, regTmp1);
        je("BETA_S32",
           T_NEAR); // NOT first_k → accumulate from S32 buffer (beta=1)

        updateCBufferPointers();

        for (int i = 0; i < MR; i++) {
            for (int j = 0; j < bFullReg; j++) {
                vmovdqu16(Xbyak::Ymm(bRegIdx + j), ptr[regTmpCptr + j * 32]);
                vpmovsxwd(Xbyak::Zmm(bRegIdx + j), Xbyak::Ymm(bRegIdx + j));
                vpslld(Xbyak::Zmm(bRegIdx + j), Xbyak::Zmm(bRegIdx + j), 16);
                vcvtps2dq(Zmm(bRegIdx + j), Zmm(bRegIdx + j));
                vpmulld(Zmm(bRegIdx + j), Zmm(bRegIdx + j), Zmm(betaRegIdx));
                vpaddd(Zmm(cRegIdx + i * bReg + j), Zmm(cRegIdx + i * bReg + j),
                       Zmm(bRegIdx + j));
            }
            if (bMaskReg > 0) {
                vmovdqu16(Xbyak::Ymm(bRegIdx + bFullReg) | k3 | T_z,
                          ptr[regTmpCptr + bFullReg * 32]);
                vpmovsxwd(Xbyak::Zmm(bRegIdx + bFullReg),
                          Xbyak::Ymm(bRegIdx + bFullReg));
                vpslld(Xbyak::Zmm(bRegIdx + bFullReg),
                       Xbyak::Zmm(bRegIdx + bFullReg), 16);
                vcvtps2dq(Zmm(bRegIdx + bFullReg), Zmm(bRegIdx + bFullReg));
                vpmulld(Zmm(bRegIdx + bFullReg), Zmm(bRegIdx + bFullReg),
                        Zmm(betaRegIdx));
                vpaddd(Zmm(cRegIdx + i * bReg + bFullReg),
                       Zmm(cRegIdx + i * bReg + bFullReg),
                       Zmm(bRegIdx + bFullReg));
            }
            add(regTmpCptr, regTmp1);
        }

        jmp("BETA_END", T_NEAR);
        L("BETA_S32");
    }

    // Scale existing C values and accumulate
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < bFullReg; j++) {
            vmovdqu32(RegType(bRegIdx + j), ptr[regTmpCptr + j * RegBytes]);
            vpmulld(RegType(bRegIdx + j), RegType(bRegIdx + j),
                    RegType(betaRegIdx));
            vpaddd(RegType(cRegIdx + i * bReg + j),
                   RegType(cRegIdx + i * bReg + j), RegType(bRegIdx + j));
        }
        if (bMaskReg > 0) {
            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                vmovdqu32(RegType(bRegIdx + bFullReg) | k3 | T_z,
                          ptr[regTmpCptr + bFullReg * RegBytes]);
                vpmulld(RegType(bRegIdx + bFullReg),
                        RegType(bRegIdx + bFullReg), RegType(betaRegIdx));
                vpaddd(RegType(cRegIdx + i * bReg + bFullReg),
                       RegType(cRegIdx + i * bReg + bFullReg),
                       RegType(bRegIdx + bFullReg));
            }
        }
        add(regTmpCptr, regRsC);
    }

    L("BETA_END");
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::generatePostOps(utils::generatorParams& params)
{
    // Use local Xbyak::Label objects to avoid redefinition across kernel
    // variants
    Xbyak::Label local_store_result;
    Xbyak::Label local_end_store;

    // Handle alpha scaling
    if (params.alphaScalingType != dlp::kernel_frame::scalingType::one) {
        RETURN_IF_ERROR(scaleAlpha());
    }

    // Handle beta scaling
    if (params.betaScalingType != dlp::kernel_frame::scalingType::zero) {
        RETURN_IF_ERROR(scaleBeta());
    }

    // Post-Ops dispatcher
    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(lpgemm_post_op_attr, is_last_k)]);
    test(regTmp1, regTmp1);
    je(local_store_result, T_NEAR);

    if (!params.kernelOps.empty()) {
        // Convert S32 accumulators to F32 for post-ops compatibility
        // Post-ops module expects F32 accumulators for all operations
        for (int i = 0; i < cReg; i++) {
            vcvtdq2ps(Zmm(cRegIdx + i), Zmm(cRegIdx + i));
        }

        // Mark that accumulators are now F32 for store functions
        accumulatorsAreF32 = true;

        gen::kernelOpsHandler kernelOpsHandler(this, params.kType);

        RETURN_IF_ERROR(kernelOpsHandler.generateKernelOps(
            params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemm, params.MR,
            params.NR, params.useMask, params.numMaskRegs, cRegIdx, cReg));

        kernelOpsHandler.generateKernelOpsAttributes();

        // Store results with F32 accumulators (post-ops path)
        RETURN_IF_ERROR(storeResult());
        jmp(local_end_store, T_NEAR);
    }

    // Store results with S32 accumulators (no post-ops or is_last_k == false)
    L(local_store_result);
    accumulatorsAreF32 = false; // Accumulators are still S32 in this path
    RETURN_IF_ERROR(storeResult());

    L(local_end_store);
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::generateKernel(utils::generatorParams& params)
{
    MR          = params.MR;
    NR          = params.NR;
    useMask     = params.useMask; // Use generation-time mask setting
    c_downscale = params.c_downscale;

    // Reset accumulator type flag for each kernel generation
    accumulatorsAreF32 = false;

    RETURN_IF_ERROR(allocateRegisters());

    // There are 14 general purpose(64 bit) registers.
    // StackFrame manages these registers, since we are using
    // one register for the input parameter of the function,
    // the rest are used as scratch registers to store variables like
    // pointers, strides, counters, etc.
    // Note that all the scratch registers allocated by the stack frame
    // need not be used by the kernel.
    // Putting inside a scope so that some tables can be generated post
    // the ret instr. StackFrame inserts a ret instr in its destructor.
    {
        Xbyak::util::StackFrame stackFrame(this, 1, 13, 0);
        initializeStackFrame(stackFrame);
        initializeParameters(params.mLoop);

        // Generate M-loop if needed, otherwise just IR loop
        if (params.mLoop) {
            RETURN_IF_ERROR(generateMLoop(params));
        } else {
            RETURN_IF_ERROR(generateIrLoop(params));
        }
    } // StackFrame destructor inserts 'ret' here

    // Generate constant data tables after the return instruction
    generateConstantData();

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitU8S8VNNI_GEMM<KType>::generateConstantData()
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
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::generateMLoop(utils::generatorParams& params)
{
    // Check if M iterations are needed
    test(regMiter, regMiter);
    je(".MLOOP_END", T_NEAR);

    L(".MLOOP_START");

    // Generate the inner IR loop
    RETURN_IF_ERROR(generateIrLoop(params));

    // Move to next M block
    RETURN_IF_ERROR(moveCPtr());

    // Update A pointer for next M block
    mov(regTmpAptr, regAPtr);
    lea(regTmpAptr, ptr[regTmpAptr + regTmp3]);
    mov(regAPtr, regTmpAptr);

    lea(regTmp2, ptr[regTmp2 + MR]);
    mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(lpgemm_post_op_attr, post_op_c_i)],
        regTmp2);
    // Decrement M counter
    dec(regMiter);
    jne(".MLOOP_START", T_NEAR);

    L(".MLOOP_END");

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::broadcastAVNNIwithB(bool isVNNIrem)
{

    for (int i = 0; i < MR; i++) {
        if (isVNNIrem) {

            // Masked load with zero-extension: load kLeft bytes, zero the rest
            vmovdqu8(Ymm(aRegIdx) | k2 | T_z, ptr[regTmpAptr]);

            // Broadcast the loaded 4-byte pattern to all lanes
            vpbroadcastd(RegType(aRegIdx), Xmm(aRegIdx));
        } else {
            // Load 4 bytes from A (VNNI group) and broadcast to all lanes
            vpbroadcastd(RegType(aRegIdx), ptr[regTmpAptr]);
        }

        for (int j = 0; j < bReg; j++) {
            // VNNI dot product: u8 * s8 -> s32 accumulate
            if constexpr (Traits::isAVX512
                          || KType == utils::kernelInstrType::avx2_ymm_16_reg) {
                vpdpbusd(RegType(cRegIdx + i * bReg + j), RegType(aRegIdx),
                         RegType(bRegIdx + j));
            }
        }
        // Advance A pointer to next row using pre-scaled rs_a
        add(regTmpAptr, regRsA);
    }
    return dlp::jit::jitGeneratorError::success;
}
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::kUnroll(int unroll, bool isVNNIrem)
{
    // Unroll the VNNI kernel loop
    for (int p = 0; p < unroll; p++) {
        // Save A pointer
        mov(regTmp1, regTmpAptr);

        // Load B registers
        RETURN_IF_ERROR(loadBValues());
        add(regBptr, regRsB);

        // Perform VNNI compute
        RETURN_IF_ERROR(broadcastAVNNIwithB(isVNNIrem));

        // Advance A pointer to next VNNI group (4 bytes)
        lea(regTmpAptr, ptr[regTmp1 + regCsA]);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitU8S8VNNI_GEMM<KType>::initializeStackFrame(
    Xbyak::util::StackFrame& stackFrame)
{
    stackPtr = stackFrame.p[0];

    regTmpAptr = stackFrame.t[0];
    regBptr    = stackFrame.t[1];
    regTmpCptr = stackFrame.t[2];
    regRsA     = stackFrame.t[3];
    regCsA     = stackFrame.t[4];
    regRsB     = stackFrame.t[5];
    regRsC     = stackFrame.t[6];
    regKIter   = stackFrame.t[7];
    regCPtr    = stackFrame.t[8];
    regAPtr    = stackFrame.t[9];
    regTmp1    = stackFrame.t[10];
    regTmp2    = stackFrame.t[11];
    regTmp3    = stackFrame.t[12];
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::moveCPtr()
{
    if (MR <= 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Optimized C pointer advancement: regCPtr += MR * regRsC
    if (MR == 3) {
        lea(regCPtr, ptr[regCPtr + regRsC + regRsC * 2]);
    } else if (MR == 5) {
        lea(regCPtr, ptr[regCPtr + regRsC + regRsC * 4]);
    } else {
        // General power-of-2 decomposition - optimal for all other cases
        // Already perfect for MR = 1,2,4,6,8.
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

    return dlp::jit::jitGeneratorError::success;
}

} // namespace amdzen::gen

// Explicit template instantiations
template class amdzen::gen::jitU8S8VNNI_GEMM<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
