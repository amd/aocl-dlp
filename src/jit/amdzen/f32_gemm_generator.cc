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

#include "aocl_dlp_config.h"

#include "f32_gemm_generator.hh"
#include "jit_register/jit_register.hh"
#include "transpose_generator.hh"

namespace amdzen::GEMMcodeGenerator {

using namespace Xbyak;

template<utils::kernelInstrType KType>
jitGEMMF32<KType>::jitGEMMF32(size_t maxSize)
    : Xbyak::CodeGenerator(maxSize, Xbyak::AutoGrow)
{
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::checkValidF32GemmParams(const utils::generatorParams& params)
{
    // Check: K_UNROLL (when !is_k1), c_downscale (MR/NR/numMaskRegs validated
    // in allocateReg/allocateMaskRegisters).
    if ((!params.is_k1 && params.K_UNROLL <= 0)
        || params.c_downscale <= DLP_INVALID || params.c_downscale >= DLP_MAX) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::allocateReg()
{
    // check if MR, NR are valid
    if (currentMR <= 0 || currentNR < 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
    bFullReg = ((currentNR) / numElemsPerReg);
    bMaskReg = (useMask ? numMaskRegs : 0);
    bReg     = bFullReg + bMaskReg;
    cReg     = currentMR * bReg;
    aReg     = numRegs - cReg - bReg - maskVecReg;

    // Check if we have enough registers
    if (aReg < 1) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
    cRegIdx = numRegs - cReg;
    bRegIdx = cRegIdx - bReg;
    aRegIdx = maskRegIdx + maskVecReg;

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMMF32<KType>::initializeParameters(bool addIrLoop)
{
    mov(regAPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);

    if (c_downscale < DLP_F32) {
        // Initialize F32→BF16 conversion constants on stack
        // Stack layout:
        // [0-15]: intermediate storage for fringe BF16 (16 bytes -
        // AVX2 only) [16-19]: value 16 for BF16→F32 shift [20-23]:
        // 0x00010000 for bit 16 extraction [24-27]: 0x00007FFF for rounding
        // [28-63]: padding/reserved

        // Store value 16 at [rsp + 16]
        mov(dword[rsp + 16], 0x10);

        // Store constant 0x00010000 at [rsp + 20] for bit 16 extraction
        mov(regTmp1.cvt32(), 0x00010000);
        mov(dword[rsp + 20], regTmp1.cvt32());

        // Store constant 0x00007FFF at [rsp + 24] for rounding
        mov(regTmp1.cvt32(), 0x00007FFF);
        mov(dword[rsp + 24], regTmp1.cvt32());
    }

    mov(regCPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, c)]);
    mov(regRsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsA)]);
    mov(regCsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, csA)]);
    mov(regRsB, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsB)]);
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsC)]);

    lea(regRsA, ptr[regRsA * sizeof(float)]);
    lea(regCsA, ptr[regCsA * sizeof(float)]);
    lea(regRsB, ptr[regRsB * sizeof(float)]);
    lea(regRsC, ptr[regRsC * sizeof(float)]);

    mov(regTmpCptr, regCPtr);
}

template<utils::kernelInstrType KType>
void
jitGEMMF32<KType>::loadMasks()
{
    if (useMask) {
        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            for (iter_t ii = 0; ii < numMaskRegs; ++ii) {
                kmovw(fringeMask[ii],
                      ptr[stackPtr
                          + offsetof(dlp::kernels::gemmParams, maskF32[0])
                          + (ii * sizeof(uint16_t))]);
            }
        } else if constexpr (KType
                             == utils::kernelInstrType::avx512_ymm_32_reg) {
            kmovb(fringeMask[0],
                  ptr[stackPtr
                      + offsetof(dlp::kernels::gemmParams, maskF32_8[0])]);
        } else if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            lea(regTmp1,
                ptr[stackPtr + offsetof(dlp::kernels::gemmParams, maskArray)]);
            vmovdqu(Ymm(maskRegIdx), ptr[regTmp1]);
        } else {
            // Kept for future use
        }
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::loadBValues()
{
    for (iter_t i = 0; i < bFullReg; i++) {
        vmovups(RegType(bRegIdx + i), ptr[regBptr + i * RegBytes]);
    }
    if (bMaskReg > 0) {
        if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            vmaskmovps(Ymm(bRegIdx + bFullReg), Ymm(maskRegIdx),
                       ptr[regBptr + bFullReg * RegBytes]);
        } else {
            // For AVX512_zmm and _ymm, mask instruction is same.
            for (iter_t ii = bFullReg; ii < bReg; ++ii) {
                vmovups(RegType(bRegIdx + ii) | fringeMask[ii - bFullReg],
                        ptr[regBptr + ii * RegBytes]);
            }
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::BroadcastAFMAwithB()
{

    for (iter_t i = 0; i < currentMR; i++) {
        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            // Fusing broadcast with fma when there is no reuse of A values
            if (bReg == 1) {
                // EVEX broadcast path: fuse scalar broadcast from memory
                // with FMA in a single instruction
                int imod4 = i % 4;
                switch (imod4) {
                    case 0:
                    case 1:
                    case 2:
                        vfmadd231ps(RegType(cRegIdx + i * bReg + 0),
                                    RegType(bRegIdx + 0),
                                    ptr_b[regTmpAptr + imod4 * regRsA]);
                        break;
                    case 3:
                        vfmadd231ps(RegType(cRegIdx + i * bReg + 0),
                                    RegType(bRegIdx + 0),
                                    ptr_b[regTmpAptr + regTmp2]);
                        lea(regTmpAptr, ptr[regTmpAptr + 4 * regRsA]);
                        break;
                }
            } else {
                vbroadcastss(RegType(aRegIdx), ptr[regTmpAptr]);
                add(regTmpAptr, regRsA);
                for (iter_t j = 0; j < bReg; j++) {
                    vfmadd231ps(RegType(cRegIdx + i * bReg + j),
                                RegType(bRegIdx + j), RegType(aRegIdx));
                }
            }
        } else {
            vbroadcastss(RegType(aRegIdx), ptr[regTmpAptr]);
            add(regTmpAptr, regRsA);
            for (iter_t j = 0; j < bReg; j++) {
                vfmadd231ps(RegType(cRegIdx + i * bReg + j),
                            RegType(bRegIdx + j), RegType(aRegIdx));
            }
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::kernelUnroll(int unroll)
{
    dlp::jit::jitGeneratorError err = dlp::jit::jitGeneratorError::error;

    // 3*rsA is needed for the EVEX broadcast path in BroadcastAFMAwithB
    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        lea(regTmp2, ptr[regRsA + regRsA * 2]);
    }

    // Unroll the kernel loop
    for (iter_t p = 0; p < unroll; p++) {
        // copy A ptr to another register
        mov(regTmp1, regTmpAptr);
        // load B regs
        RETURN_IF_ERROR(loadBValues());
        add(regBptr, regRsB);
        RETURN_IF_ERROR(BroadcastAFMAwithB());
        lea(regTmpAptr, ptr[regTmp1 + regCsA]);
    }
    return dlp::jit::jitGeneratorError::success;
}
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::storeResult(bool fuseBetaWithStore, bool mLoop)
{
    mov(regTmpCptr, regCPtr);

    inLocalLabel();
    // Load cs_c value and check if cs_c is 1
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, csC)]);
    cmp(regTmp1, 1);
    je(".storeResultRowMajor", T_NEAR);
    // RETURN_IF_ERROR(storeResultColumnMajor());
    x86gen::TransposeGenerator<KType> transposeGenerator(
        this, currentMR, currentNR, useMask, mLoop, numMaskRegs, cRegIdx, cReg,
        regTmpCptr);
    RETURN_IF_ERROR(transposeGenerator.generateTranspose(
        stackPtr, dlp::jit::jitAlgoType::gemm, fuseBetaWithStore));
    jmp(".end", T_NEAR);
    L(".storeResultRowMajor");
    RETURN_IF_ERROR(storeResultRowMajor(fuseBetaWithStore));
    L(".end");

    outLocalLabel();
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::storeResultColumnMajor(bool fuseBetaWithStore)
{
    return dlp::jit::jitGeneratorError::notSupported;
}

template<>
dlp::jit::jitGeneratorError
jitGEMMF32<utils::kernelInstrType::avx512_zmm_32_reg>::storeResultColumnMajor(
    bool fuseBetaWithStore)
{

    std::queue<int> scratch_reg_queue;
    for (iter_t i = 0; i < cRegIdx; i++) {
        scratch_reg_queue.push(i);
    }

    // take two zmm from scratch_reg_queue and load the offsets
    int offsets1RegIdx = scratch_reg_queue.front();
    scratch_reg_queue.pop();
    int offsets2RegIdx = scratch_reg_queue.front();
    scratch_reg_queue.pop();
    int scratch3RegIdx = scratch_reg_queue.front();
    scratch_reg_queue.pop();

    // load offsets into register
    vmovdqu32(RegType(offsets1RegIdx), ptr[rip + offsets]);
    vmovdqu32(RegType(offsets2RegIdx), ptr[rip + offsets + RegBytes]);

    // load csC value multiply by sizeof(float)
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, csC)]);
    lea(regTmp1, ptr[regTmp1 * sizeof(float)]);

    // reuse the betaRegIdx register to broadcast the csC*sizeof(float) value
    // and multiply with offsets.
    vpbroadcastq(RegType(scratch3RegIdx), regTmp1);

    // now multiply offsets with csC
    vpmullq(RegType(offsets1RegIdx), RegType(offsets1RegIdx),
            RegType(scratch3RegIdx));
    vpmullq(RegType(offsets2RegIdx), RegType(offsets2RegIdx),
            RegType(scratch3RegIdx));

    // calculate 16*csC*sizeof(float) and store in regTmp1
    lea(regTmp1, ptr[regTmp1 * 8]);
    lea(regTmp1, ptr[regTmp1 * 2]);

    // take two zmm from scratch_reg_queue
    // for scatter/gather operations
    int scratchReg1 = scratch_reg_queue.front();
    scratch_reg_queue.pop();
    int scratchReg2 = scratch_reg_queue.front();
    scratch_reg_queue.pop();

    for (iter_t i = 0; i < currentMR; i++) {
        mov(regTmp2, regTmpCptr);
        for (iter_t j = 0; j < bFullReg; j++) {
            resetMasks(false, 0);
            vextractf32x8(halfRegType(scratchReg2),
                          RegType(cRegIdx + i * bReg + j), 1);
            vscatterqps(ptr[regTmp2 + RegType(offsets1RegIdx) * 1] | mask0,
                        halfRegType(cRegIdx + i * bReg + j));
            vscatterqps(ptr[regTmp2 + RegType(offsets2RegIdx) * 1] | mask1,
                        halfRegType(scratchReg2));
            // move to next set of cols
            lea(regTmp2, ptr[regTmp2 + regTmp1]);
        }
        if (bMaskReg > 0) {
            for (iter_t idx = 0; idx < numMaskRegs; idx++) {
                resetMasks(true, idx);
                vextractf32x8(halfRegType(scratchReg2),
                              RegType(cRegIdx + i * bReg + bFullReg), 1);
                vscatterqps(ptr[regTmp2 + RegType(offsets1RegIdx) * 1] | mask0,
                            halfRegType(cRegIdx + i * bReg + bFullReg));
                vscatterqps(ptr[regTmp2 + RegType(offsets2RegIdx) * 1] | mask1,
                            halfRegType(scratchReg2));
                // move to next set of cols
                lea(regTmp2, ptr[regTmp2 + regTmp1]);
            }
        }
        add(regTmpCptr, sizeof(float));
    }

    // release all the scratch registers
    scratch_reg_queue.push(scratchReg1);
    scratch_reg_queue.push(scratchReg2);

    scratch_reg_queue.push(offsets1RegIdx);
    scratch_reg_queue.push(offsets2RegIdx);
    scratch_reg_queue.push(scratch3RegIdx);

    return dlp::jit::jitGeneratorError::success;
}

// Generic template for AVX2 and AVX512_256 (both use Ymm as RegType)
// Converts 8×F32 in Ymm(destIdx) → 8×BF16 in Xmm(scratch2)
template<>
dlp::jit::jitGeneratorError
jitGEMMF32<utils::kernelInstrType::avx2_ymm_16_reg>::convertF32toBF16(
    int scratch1, int scratch2, int destIdx)
{
    // Manual F32→BF16 conversion using round-to-nearest-even with tie-break
    // This generic implementation works for AVX2 and AVX512_256

    vbroadcastss(RegType(scratch1), ptr[rsp + 20]); // Load 0x00010000
    vpand(RegType(scratch1), RegType(destIdx),
          RegType(scratch1)); // Extract bit 16
    vpsrld(RegType(scratch1), RegType(scratch1),
           16); // Shift to position 0 → tlsb

    vbroadcastss(RegType(scratch2), ptr[rsp + 24]); // Load 0x00007FFF
    vpaddd(RegType(scratch2), RegType(destIdx),
           RegType(scratch2)); // Add rounding
    vpaddd(RegType(scratch2), RegType(scratch2),
           RegType(scratch1)); // Add tlsb → rounded

    vpsrld(RegType(scratch2), RegType(scratch2), 16); // Shift right 16 bits

    // Extract upper 128 bits of YMM → XMM
    vextracti128(halfRegType(scratch1), RegType(scratch2), 1);

    // Pack 8×32-bit to 8×16-bit (result in halfRegType(scratch2))
    vpackusdw(halfRegType(scratch2), halfRegType(scratch2),
              halfRegType(scratch1));

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitGEMMF32<utils::kernelInstrType::avx512_zmm_32_reg>::convertF32toBF16(
    int scratch1, int scratch2, int destIdx)
{
    // AVX512 ZMM variant: Convert 16 f32 in Zmm to 16 bf16 in Ymm
    vbroadcastss(RegType(scratch1),
                 ptr[rsp + 20]); // Load 0x00010000
    vpandd(RegType(scratch1), RegType(destIdx),
           RegType(scratch1)); // Extract bit 16
    vpsrld(RegType(scratch1), RegType(scratch1),
           16); // Shift to position 0 → tlsb

    vbroadcastss(RegType(scratch2),
                 ptr[rsp + 24]); // Load 0x00007FFF
    vpaddd(RegType(scratch2), RegType(destIdx),
           RegType(scratch2)); // Add rounding to original

    vpaddd(RegType(scratch2), RegType(scratch2),
           RegType(scratch1)); // Add tlsb → rounded

    vpsrld(RegType(scratch2), RegType(scratch2),
           16); // Shift right 16 bits

    // Extract upper 256 bits of ZMM → YMM
    vextracti64x4(halfRegType(scratch1), RegType(scratch2), 1);

    // Pack 16×32-bit to 16×16-bit (lower 8 from scratch2, upper 8 from
    // scratch1)
    vpackusdw(halfRegType(scratch2), halfRegType(scratch2),
              halfRegType(scratch1));

    // Permute to get correct order: [0,1,2,3,8,9,10,11,4,5,6,7,12,13,14,15] ->
    // [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    vpermq(halfRegType(scratch2), halfRegType(scratch2),
           0xD8); // 0xD8 = 11011000b = [0,2,1,3]

    return dlp::jit::jitGeneratorError::success;
}

// F32 to BF16 conversion routines are currently only implemented for avx512_zmm
// and avx2 variants
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::convertF32toBF16(int scratch1, int scratch2, int destIdx)
{
    return dlp::jit::jitGeneratorError::notSupported;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::storeResultRowMajor(bool fuseBetaWithStore)
{
    mov(regTmpCptr, regCPtr);

    inLocalLabel();

    // if fuseBetaWithStore is true, then we need to check if beta is zero
    // during run-time. This is majorly done to avoid accessing of beta value
    // if original beta is 0 during first k iteration.
    if (fuseBetaWithStore) {
        // broadcast beta value
        int betaRegIdx    = aRegIdx;
        int scratchRegIdx = bRegIdx;
        mov(regTmp2, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, beta)]);
        vbroadcastss(RegType(betaRegIdx), ptr[regTmp2]);

        // check if beta is 0
        vxorps(RegType(scratchRegIdx), RegType(scratchRegIdx),
               RegType(scratchRegIdx));
        vucomiss(Xmm(betaRegIdx), Xmm(scratchRegIdx));
        je(".storeResultRowMajorBZ", T_NEAR);

        // beta is non-zero, fuse with store
        for (iter_t i = 0; i < currentMR; i++) {
            for (iter_t j = 0; j < bFullReg; j++) {
                // Regular store
                vmovups(RegType(bRegIdx + j), ptr[regTmpCptr + j * RegBytes]);
                vfmadd231ps(RegType(cRegIdx + i * bReg + j),
                            RegType(betaRegIdx), RegType(bRegIdx + j));
                vmovups(ptr[regTmpCptr + j * RegBytes],
                        RegType(cRegIdx + i * bReg + j));
            }
            if (bMaskReg > 0) {
                if constexpr (KType
                              == utils::kernelInstrType::avx2_ymm_16_reg) {
                    vmaskmovps(Ymm(bRegIdx + bFullReg), Ymm(maskRegIdx),
                               ptr[regTmpCptr + bFullReg * RegBytes]);
                    vfmadd231ps(Ymm(cRegIdx + i * bReg + bFullReg),
                                Ymm(bRegIdx + bFullReg), Ymm(betaRegIdx));
                    vmaskmovps(ptr[regTmpCptr + bFullReg * RegBytes],
                               Ymm(maskRegIdx),
                               Ymm(cRegIdx + i * bReg + bFullReg));
                } else {
                    for (iter_t maskI = bFullReg; maskI < bReg; ++maskI) {
                        vmovups(RegType(bRegIdx + maskI)
                                    | fringeMask[maskI - bFullReg],
                                ptr[regTmpCptr + maskI * RegBytes]);
                        vfmadd231ps(RegType(cRegIdx + i * bReg + maskI),
                                    RegType(betaRegIdx),
                                    RegType(bRegIdx + maskI));
                        vmovups(ptr[regTmpCptr + maskI * RegBytes]
                                    | fringeMask[maskI - bFullReg],
                                RegType(cRegIdx + i * bReg + maskI));
                    }
                }
            }
            add(regTmpCptr, regRsC);
        }
        jmp(".afterStoreResultRowMajor", T_NEAR);
    }

    L(".storeResultRowMajorBZ");
    if (c_downscale < DLP_F32) {
        // Check for is_last_k
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
        test(regTmp1, regTmp1);
        je(".storeResultF32ToBF16", T_NEAR);

        // Get buf_downscale
        mov(regTmpCptr,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, buf_downscale)]);
        // NULL check
        cmp(regTmpCptr, 0);
        je(".storeResultF32ToBF16", T_NEAR);

        // Get post_op_c_j
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
        lea(regTmp1, ptr[regTmp1 * sizeof(int16_t)]);
        add(regTmpCptr, regTmp1);

        // Get rs_c_downscale
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, rs_c_downscale)]);
        lea(regTmp1, ptr[regTmp1 * sizeof(int16_t)]);
        mov(regTmp3,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
        imul(regTmp3, regTmp1);
        add(regTmpCptr, regTmp3);

        // Use convertF32toBF16 for all ISAs (no AVX512_BF16 dependency)
        int scratch1 = aRegIdx;
        int scratch2 = bRegIdx;

        // Calculate number of F32 elements per register and BF16 output bytes
        // For AVX2: 8 F32 elements → 16 bytes BF16 output (Xmm)
        int nElemsPerReg = RegBytes / sizeof(float);

        for (iter_t i = 0; i < currentMR; i++) {
            for (iter_t j = 0; j < bFullReg; j++) {
                RETURN_IF_ERROR(convertF32toBF16(scratch1, scratch2,
                                                 cRegIdx + i * bReg + j));
                // Store the BF16 result (in halfRegType(scratch2))
                // AVX512: Ymm output (32 bytes), others: Xmm output (16 bytes)
                if constexpr (KType
                              == utils::kernelInstrType::avx2_ymm_16_reg) {
                    // AVX2 and AVX512_256: Use movdqu for Xmm (VEX-encoded)
                    movdqu(ptr[regTmpCptr + j * halfRegBytes],
                           halfRegType(scratch2));
                } else {
                    // AVX512: Store 32 bytes (16×BF16) from Ymm
                    vmovdqu16(ptr[regTmpCptr + j * halfRegBytes],
                              halfRegType(scratch2));
                }
            }
            if (bMaskReg > 0) {
                // Store converted BF16 to stack for element-wise access
                if constexpr (KType
                              == utils::kernelInstrType::avx2_ymm_16_reg) {
                    // Convert fringe register
                    RETURN_IF_ERROR(convertF32toBF16(
                        scratch1, scratch2, cRegIdx + i * bReg + bFullReg));
                    // AVX2: Store converted BF16 to stack [rsp+0] for
                    // element-wise access
                    movdqu(ptr[rsp + 0], halfRegType(scratch2));
                    // Get n_remainder: n % nElemsPerReg
                    mov(regTmp2,
                        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, n)]);
                    and_(regTmp2.cvt32(), nElemsPerReg - 1);

                    // Calculate destination base address for fringe elements
                    lea(regKIter, ptr[regTmpCptr + bFullReg * halfRegBytes]);

                    // Loop: copy n_remainder BF16 elements from stack to
                    // destination
                    xor_(regTmp3.cvt32(), regTmp3.cvt32()); // elem_idx = 0

                    Label loop_start, loop_end;
                    L(loop_start);

                    // Check if elem_idx < n_remainder
                    cmp(regTmp3.cvt32(), regTmp2.cvt32());
                    jge(loop_end, T_NEAR);

                    // Load BF16 value from stack and store to destination
                    mov(regBptr.cvt16(), word[rsp + regTmp3 * sizeof(int16_t)]);
                    mov(word[regKIter + regTmp3 * sizeof(int16_t)],
                        regBptr.cvt16());

                    inc(regTmp3.cvt32()); // elem_idx++
                    jmp(loop_start, T_NEAR);

                    L(loop_end);
                } else if (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                    // AVX512_ZMM/AVX512_YMM: Store BF16 output directly to
                    // memory with mask AVX512_ZMM: 32 bytes (16×BF16) from Ymm
                    // AVX512_YMM: 16 bytes (8×BF16) from Xmm
                    for (iter_t maskI = bFullReg; maskI < bReg; ++maskI) {
                        RETURN_IF_ERROR(convertF32toBF16(
                            scratch1, scratch2, cRegIdx + i * bReg + maskI));
                        vmovdqu16(ptr[regTmpCptr + maskI * halfRegBytes]
                                      | fringeMask[maskI - bFullReg],
                                  halfRegType(scratch2));
                    }
                } else {
                    return dlp::jit::jitGeneratorError::notSupported;
                }
            }
            add(regTmpCptr, regTmp1);
        }

        jmp(".afterStoreResultRowMajor", T_NEAR);
        L(".storeResultF32ToBF16");
    }

    for (iter_t i = 0; i < currentMR; i++) {
        for (iter_t j = 0; j < bFullReg; j++) {
            // Regular store
            vmovups(ptr[regTmpCptr + j * RegBytes],
                    RegType(cRegIdx + i * bReg + j));
        }
        if (bMaskReg > 0) {
            if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
                vmaskmovps(ptr[regTmpCptr + bFullReg * RegBytes],
                           Ymm(maskRegIdx), Ymm(cRegIdx + i * bReg + bFullReg));
            } else {
                for (iter_t maskI = bFullReg; maskI < bReg; ++maskI) {
                    vmovups(ptr[regTmpCptr + maskI * RegBytes]
                                | fringeMask[maskI - bFullReg],
                            RegType(cRegIdx + i * bReg + maskI));
                }
            }
        }
        add(regTmpCptr, regRsC);
    }
    L(".afterStoreResultRowMajor");
    outLocalLabel();
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMMF32<KType>::initializeStackFrame(Xbyak::util::StackFrame& stackFrame)
{
    stackPtr = stackFrame.p[0];

    regTmpCptr = stackFrame.t[0];
    regTmpAptr = stackFrame.t[1];
    regBptr    = stackFrame.t[2];
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
void
jitGEMMF32<KType>::regInit()
{
    vxorps(RegType(cRegIdx), RegType(cRegIdx), RegType(cRegIdx));
    for (iter_t i = 1; i < cReg; i++) {
        vmovaps(RegType(cRegIdx + i), RegType(cRegIdx));
    }
}

template<utils::kernelInstrType KType>
void
jitGEMMF32<KType>::moveCPtr(Xbyak::Reg64 regPtr,
                            Xbyak::Reg64 regStride,
                            int          val)
{
    // Update C pointer for next row : cbuf += m * MR * rs_c
    // lea() doesn't work with non-power-of-2 values, and avx2
    // doesn't support imul. Hence, decomposing m value to power
    // of 2 to handle all the cases commonly. Let's say if MR = 13
    // then we can represent it as 8 + 4 + 1
    int m_val       = val;
    int power2scale = 1;
    while (m_val > 0) {
        if (m_val & 1) {
            // lea() only supports scale factors of 1, 2, 4, and 8
            // For powers of 2 greater than 8, use a temporary register
            if (power2scale <= 8) {
                lea(regPtr, ptr[regPtr + power2scale * regStride]);
            } else {
                // Use regTmp1 as temporary register
                // regTmp1 = regStride << log2(power2scale)
                mov(regTmp1, regStride);
                int shift_amount = 0;
                int temp_scale   = power2scale;
                while (temp_scale > 1) {
                    shift_amount++;
                    temp_scale >>= 1;
                }
                shl(regTmp1, shift_amount);
                // regPtr += regTmp1
                add(regPtr, regTmp1);
            }
        }
        m_val >>= 1;
        power2scale <<= 1;
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::scaleAlpha()
{
    int          alphaRegIdx = aRegIdx;
    Xbyak::Reg64 alphaReg    = regTmp1;
    mov(alphaReg, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, alpha)]);
    vbroadcastss(RegType(alphaRegIdx), ptr[alphaReg]);
    for (iter_t i = 0; i < cReg; i++) {
        vmulps(RegType(cRegIdx + i), RegType(cRegIdx + i),
               RegType(alphaRegIdx));
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::scaleBeta()
{
    // copy c address from regCptr
    mov(regTmpCptr, regCPtr);

    // Handling col-major C matrix is not supported for avx2 and avx512_256
    // paths. For these cases, we ensure that the C matrix is row-major and then
    // call the JIT kernel. The logic to ensure this is in
    // aocl_gemm_f32f32f32of32.c file.

    // Additionally, when C is downscaled the current implementation only
    // supports the row-major beta-scaling path.
    if (KType == utils::kernelInstrType::avx2_ymm_16_reg
        || c_downscale < DLP_F32) {
        return scaleBetaRowMajor();
    }

    inLocalLabel();
    // broadcast beta value
    int betaRegIdx    = aRegIdx;
    int scratchRegIdx = bRegIdx;
    mov(regTmp2, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, beta)]);
    vbroadcastss(RegType(betaRegIdx), ptr[regTmp2]);

    // check if beta is 0
    vxorps(RegType(scratchRegIdx), RegType(scratchRegIdx),
           RegType(scratchRegIdx));
    vucomiss(Xmm(betaRegIdx), Xmm(scratchRegIdx));
    je(".end", T_NEAR);

    // Load cs_c value and check if cs_c is 1
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, csC)]);
    cmp(regTmp1, 1);
    je(".scaleBetaRowMajor", T_NEAR);
    RETURN_IF_ERROR(scaleBetaColumnMajor());
    jmp(".end", T_NEAR);
    L(".scaleBetaRowMajor");
    RETURN_IF_ERROR(scaleBetaRowMajor());
    L(".end");
    outLocalLabel();
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::allocateMaskRegisters()
{
    maskVecReg = 0;
    maskRegIdx = 0;

    if (numMaskRegs < 0) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        if (useMask) {
            maskVecReg = numMaskRegs;
        }
        return dlp::jit::jitGeneratorError::success;
    }

    // Right now, numMaskRegs is limited to 5 in DE.
    // so, we are sure that 2 masks are available for gather/scatter operations.
    // TODO: handle the case when numMaskRegs > 5.
    if (dlp::kernels::maxNumMasks < numMaskRegs + 2) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // This same logic will be copied to post-ops and transpose generator
    // modules to ensure handshake between GEMM and post-ops/transpose
    // generator. If this logic changes, we need to change the logic in
    // post-ops/transpose generator module as well.
    for (iter_t i = 0; i < numMaskRegs; i++) {
        fringeMask[i] = Opmask(i + 1);
    }

    mask0 = Opmask(numMaskRegs + 1);
    mask1 = Opmask(numMaskRegs + 2);

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMMF32<KType>::resetMasks(bool mask, int idx)
{
    if (mask) {
        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            // For Zmm: 16-bit mask, split into 8 bits each
            // Load lower 8 bits of maskF32 (covers elements 0-7)
            kmovb(mask0, fringeMask[idx]);
            // Load upper 8 bits of maskF32 (covers elements 8-15)
            kshiftrw(mask1, fringeMask[idx], 8);
        } else if constexpr (KType
                             == utils::kernelInstrType::avx512_ymm_32_reg) {
            // For Ymm: 8-bit mask, split into 4 bits each
            // Load lower 4 bits of maskF32 (covers elements 0-3)
            // Note: kmovb loads 8 bits, but instruction will only use lower 4
            kmovb(mask0, fringeMask[idx]);
            // Load upper 4 bits of maskF32 (covers elements 4-7)
            kshiftrw(mask1, fringeMask[idx], 4);
        } else {
            // saved for future use
        }
    } else {
        kxnorw(mask0, mask0, mask0);
        kxnorw(mask1, mask1, mask1);
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::scaleBetaColumnMajor()
{
    // we need atleast 5 ZMM spare registers to do this.
    // For now, assuming that we have 5 ZMM spare registers.
    // TODO: Implement a strategy like gelu_tanh post-ops using
    // stack incase 5 regs are not available.

    if (cRegIdx < 5) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    std::queue<int> scratch_reg_queue;
    for (iter_t i = 0; i < cRegIdx; i++) {
        scratch_reg_queue.push(i);
    }

    // take two zmm from scratch_reg_queue and load the offsets
    int offsets1RegIdx = scratch_reg_queue.front();
    scratch_reg_queue.pop();
    int offsets2RegIdx = scratch_reg_queue.front();
    scratch_reg_queue.pop();
    int betaRegIdx = scratch_reg_queue.front();
    scratch_reg_queue.pop();

    // load offsets into register
    vmovdqu32(RegType(offsets1RegIdx), ptr[rip + offsets]);
    vmovdqu32(RegType(offsets2RegIdx), ptr[rip + offsets + RegBytes]);

    // load csC value multiply by sizeof(float)
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, csC)]);
    lea(regTmp1, ptr[regTmp1 * sizeof(float)]);

    // reuse the betaRegIdx register to broadcast the csC*sizeof(float) value
    // and multiply with offsets.
    int scratch3RegIdx = betaRegIdx;
    vpbroadcastq(RegType(scratch3RegIdx), regTmp1);

    // now multiply offsets with csC
    vpmullq(RegType(offsets1RegIdx), RegType(offsets1RegIdx),
            RegType(scratch3RegIdx));
    vpmullq(RegType(offsets2RegIdx), RegType(offsets2RegIdx),
            RegType(scratch3RegIdx));

    // calculate 16*csC*sizeof(float) and store in regTmp1
    lea(regTmp1, ptr[regTmp1 * 8]);
    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        lea(regTmp1, ptr[regTmp1 * 2]);
    }

    // broadcast beta value
    mov(regTmp2, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, beta)]);
    vbroadcastss(RegType(betaRegIdx), ptr[regTmp2]);

    // take two zmm from scratch_reg_queue
    // for scatter/gather operations
    int scratchReg1 = scratch_reg_queue.front();
    scratch_reg_queue.pop();
    int scratchReg2 = scratch_reg_queue.front();
    scratch_reg_queue.pop();

    for (iter_t i = 0; i < currentMR; i++) {
        mov(regTmp2, regTmpCptr);
        for (iter_t j = 0; j < bFullReg; j++) {
            resetMasks(false, 0);

            vgatherqps(halfRegType(scratchReg1) | mask0,
                       ptr[regTmp2 + RegType(offsets1RegIdx) * 1]);
            vgatherqps(halfRegType(scratchReg2) | mask1,
                       ptr[regTmp2 + RegType(offsets2RegIdx) * 1]);
            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                vinsertf32x8(RegType(scratchReg1), RegType(scratchReg1),
                             halfRegType(scratchReg2), 1);
            } else if constexpr (KType
                                 == utils::kernelInstrType::avx512_ymm_32_reg) {
                vinsertf32x4(RegType(scratchReg1), RegType(scratchReg1),
                             halfRegType(scratchReg2), 1);
            } else {
                // saved for future use
            }
            // fma with beta
            vfmadd231ps(RegType(cRegIdx + i * bReg + j), RegType(scratchReg1),
                        RegType(betaRegIdx));

            // move to next set of cols
            lea(regTmp2, ptr[regTmp2 + regTmp1]);
        }
        if (bMaskReg > 0) {
            for (iter_t idx = 0; idx < numMaskRegs; idx++) {
                resetMasks(true, idx);
                vgatherqps(halfRegType(scratchReg1) | mask0,
                           ptr[regTmp2 + RegType(offsets1RegIdx) * 1]);
                vgatherqps(halfRegType(scratchReg2) | mask1,
                           ptr[regTmp2 + RegType(offsets2RegIdx) * 1]);
                if constexpr (KType
                              == utils::kernelInstrType::avx512_zmm_32_reg) {
                    vinsertf32x8(RegType(scratchReg1), RegType(scratchReg1),
                                 halfRegType(scratchReg2), 1);
                } else if constexpr (KType
                                     == utils::kernelInstrType::
                                         avx512_ymm_32_reg) {
                    vinsertf32x4(RegType(scratchReg1), RegType(scratchReg1),
                                 halfRegType(scratchReg2), 1);
                } else {
                    // saved for future use
                }
                // fma with beta
                vfmadd231ps(RegType(cRegIdx + i * bReg + bFullReg + idx),
                            RegType(scratchReg1), RegType(betaRegIdx));

                // move to next set of cols
                lea(regTmp2, ptr[regTmp2 + regTmp1]);
            }
        }
        add(regTmpCptr, sizeof(float));
    }

    // release all the scratch registers
    scratch_reg_queue.push(scratchReg1);
    scratch_reg_queue.push(scratchReg2);
    scratch_reg_queue.push(offsets1RegIdx);
    scratch_reg_queue.push(offsets2RegIdx);
    scratch_reg_queue.push(betaRegIdx);

    if (!dumpedOffsets) {
        jmp("offsets_end", T_NEAR);
        {
            size_t remain = getSize() % 64;
            if (remain)
                nop(64 - remain);
        }
        L(offsets);
        int64_t offsets[16] = { 0, 1, 2,  3,  4,  5,  6,  7,
                                8, 9, 10, 11, 12, 13, 14, 15 };
        db(reinterpret_cast<uint8_t*>(&offsets), sizeof(offsets));
        L("offsets_end");
        dumpedOffsets = true;
    }

    return dlp::jit::jitGeneratorError::success;
}

template<>
dlp::jit::jitGeneratorError
jitGEMMF32<utils::kernelInstrType::avx2_ymm_16_reg>::scaleBetaColumnMajor()
{
    return dlp::jit::jitGeneratorError::notSupported;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::scaleBetaRowMajor()
{
    inLocalLabel();

    // broadcast beta value
    Xbyak::Reg64 betaReg    = regTmp1;
    int          betaRegIdx = aRegIdx;
    mov(betaReg, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, beta)]);
    vbroadcastss(RegType(betaRegIdx), ptr[betaReg]);

    if (c_downscale < DLP_F32) {
        // Check for is_first_k
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_first_k)]);
        test(regTmp1, regTmp1);
        je(".betaOp", T_NEAR);

        // Get buf_downscale
        mov(regTmpCptr,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, buf_downscale)]);
        // NULL check
        cmp(regTmpCptr, 0);
        je(".betaOp", T_NEAR);

        // Get post_op_c_j
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
        lea(regTmp1, ptr[regTmp1 * sizeof(int16_t)]);
        add(regTmpCptr, regTmp1);

        // Get rs_c_downscale
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, rs_c_downscale)]);
        lea(regTmp1, ptr[regTmp1 * sizeof(int16_t)]);
        mov(regTmp3,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
        imul(regTmp3, regTmp1);
        add(regTmpCptr, regTmp3);
        mov(regKIter, regTmpCptr);

        // Broadcast shift value 16 for BF16→F32 conversion
        vpbroadcastd(RegType(aRegIdx + 1), ptr[rsp + 16]);

        // Calculate BF16 input bytes per register
        // For AVX2/AVX512_256: 8 F32 elements → 16 bytes BF16 input (Xmm)
        // For AVX512: 16 F32 elements → 32 bytes BF16 input (Ymm)
        int nElemsPerReg   = RegBytes / sizeof(float);
        int bf16InputBytes = nElemsPerReg * sizeof(int16_t);

        for (iter_t i = 0; i < currentMR; i++) {
            for (iter_t j = 0; j < bFullReg; j++) {
                // Load BF16 values from buf_downscale and convert to F32
                // Stack layout [rsp+16] contains shift value 16 for conversion
                // AVX2: Load Xmm (16 bytes = 8×BF16), expand to Ymm (8×F32)
                // AVX512_YMM: Load Xmm (16 bytes = 8×BF16), expand to Ymm
                // (8×F32) AVX512_ZMM: Load Ymm (32 bytes = 16×BF16), expand to
                // Zmm (16×F32)

                if constexpr (KType
                              == utils::kernelInstrType::avx2_ymm_16_reg) {
                    // AVX2: Load 16 bytes (8×BF16) from buf_downscale
                    movdqu(halfRegType(bRegIdx + j),
                           ptr[regTmpCptr + j * halfRegBytes]);
                    vpmovsxwd(RegType(bRegIdx + j), halfRegType(bRegIdx + j));
                    vpsllvd(RegType(bRegIdx + j), RegType(bRegIdx + j),
                            RegType(aRegIdx + 1));
                    vfmadd231ps(RegType(cRegIdx + i * bReg + j),
                                RegType(bRegIdx + j), RegType(betaRegIdx));
                } else {
                    // AVX512_YMM/ZMM: Load BF16 data and convert to F32
                    vmovdqu16(halfRegType(bRegIdx + j),
                              ptr[regTmpCptr + j * halfRegBytes]);
                    vpmovsxwd(RegType(bRegIdx + j), halfRegType(bRegIdx + j));
                    vpsllvd(RegType(bRegIdx + j), RegType(bRegIdx + j),
                            RegType(aRegIdx + 1));
                    vfmadd231ps(RegType(cRegIdx + i * bReg + j),
                                RegType(bRegIdx + j), RegType(betaRegIdx));
                }
            }
            if (bMaskReg > 0) {
                if constexpr (KType
                              == utils::kernelInstrType::avx2_ymm_16_reg) {
                    // AVX2: Handle fringe BF16 elements via stack [rsp+0]
                    // Load n_remainder BF16 values to stack, convert, then
                    // apply beta Get n_remainder: n % nElemsPerReg
                    mov(regTmp2,
                        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, n)]);
                    and_(regTmp2.cvt32(), nElemsPerReg - 1);

                    // Calculate source base address for fringe BF16 elements
                    lea(regKIter, ptr[regTmpCptr + bFullReg * halfRegBytes]);

                    // Loop: Load n_remainder BF16 elements from C matrix to
                    // stack
                    xor_(regTmp3.cvt32(), regTmp3.cvt32()); // elem_idx = 0

                    Label loop_load_start, loop_load_end;
                    L(loop_load_start);

                    // Check if elem_idx < n_remainder
                    cmp(regTmp3.cvt32(), regTmp2.cvt32());
                    jge(loop_load_end, T_NEAR);

                    // Load BF16 value from C matrix to stack
                    mov(regBptr.cvt16(),
                        word[regKIter + regTmp3 * sizeof(int16_t)]);
                    mov(word[rsp + regTmp3 * sizeof(int16_t)], regBptr.cvt16());

                    inc(regTmp3.cvt32()); // elem_idx++
                    jmp(loop_load_start, T_NEAR);

                    L(loop_load_end);
                    // Load BF16 data from stack [rsp+0] and convert to F32
                    movdqu(halfRegType(bRegIdx + bFullReg), ptr[rsp]);
                    vpmovsxwd(RegType(bRegIdx + bFullReg),
                              halfRegType(bRegIdx + bFullReg));
                    vpsllvd(RegType(bRegIdx + bFullReg),
                            RegType(bRegIdx + bFullReg), RegType(aRegIdx + 1));
                    vfmadd231ps(RegType(cRegIdx + i * bReg + bFullReg),
                                RegType(bRegIdx + bFullReg),
                                RegType(betaRegIdx));

                } else if (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                    // AVX512_YMM/ZMM: Load fringe BF16 elements directly from
                    // buf_downscale with mask Load BF16 data, sign-extend to
                    // 32-bit, shift left by 16, then apply beta
                    for (iter_t maskI = bFullReg; maskI < bReg; ++maskI) {
                        vmovdqu16(halfRegType(bRegIdx + maskI)
                                      | fringeMask[maskI - bFullReg],
                                  ptr[regTmpCptr + maskI * halfRegBytes]);
                        vpmovsxwd(RegType(bRegIdx + maskI),
                                  halfRegType(bRegIdx + maskI));
                        vpsllvd(RegType(bRegIdx + maskI),
                                RegType(bRegIdx + maskI), RegType(aRegIdx + 1));
                        vfmadd231ps(RegType(cRegIdx + i * bReg + maskI),
                                    RegType(bRegIdx + maskI),
                                    RegType(betaRegIdx));
                    }
                } else {
                    return dlp::jit::jitGeneratorError::notSupported;
                }
            }
            add(regTmpCptr, regTmp1);
        }

        jmp(".betaOpEnd", T_NEAR);
        L(".betaOp");
    }

    for (iter_t i = 0; i < currentMR; i++) {
        for (iter_t j = 0; j < bFullReg; j++) {
            vmovups(RegType(bRegIdx + j), ptr[regTmpCptr + j * RegBytes]);
            vfmadd231ps(RegType(cRegIdx + i * bReg + j), RegType(betaRegIdx),
                        RegType(bRegIdx + j));
        }
        if (bMaskReg > 0) {
            if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
                // Handle masked C values for fringe elements using dedicated
                // mask register
                vmaskmovps(Ymm(bRegIdx + bFullReg), Ymm(maskRegIdx),
                           ptr[regTmpCptr + bFullReg * RegBytes]);
                vfmadd231ps(Ymm(cRegIdx + i * bReg + bFullReg),
                            Ymm(bRegIdx + bFullReg), Ymm(betaRegIdx));
            } else {
                for (iter_t maskI = bFullReg; maskI < bReg; ++maskI) {
                    vmovups(RegType(bRegIdx + maskI)
                                | fringeMask[maskI - bFullReg],
                            ptr[regTmpCptr + maskI * RegBytes]);
                    vfmadd231ps(RegType(cRegIdx + i * bReg + maskI),
                                RegType(betaRegIdx), RegType(bRegIdx + maskI));
                }
            }
        }
        add(regTmpCptr, regRsC);
    }

    L(".betaOpEnd");
    outLocalLabel();
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::generateIrLoop(utils::generatorParams& params)
{
    inLocalLabel();
    // calculate and load pointers
    if (params.mLoop) {
        mov(regMiter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, mIter)]);
        L(".BLOOP6X64I");
        // Move A and C pointers to stack so
        // they can be accessed at the start or end of IR-loop
    }
    mov(regTmpAptr, regAPtr);
    mov(regBptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, b)]);

    // zero out accumulators
    regInit();

    // Generate K-loop
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kIterBP)]);
    test(regKIter, regKIter);
    je(".BCONSIDKLEFT", T_NEAR);

    // Kernel unroll loop
    L(".BLOOPKITER");
    RETURN_IF_ERROR(kernelUnroll(params.K_UNROLL));
    sub(regKIter, 1); // i -= 1
    jne(".BLOOPKITER", T_NEAR);

    L(".BCONSIDKLEFT");
    // load k_left
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeft)]);
    test(regKIter, regKIter);
    je(".BPOSTACCUM", T_NEAR);

    RETURN_IF_ERROR(kernelUnroll(1));

    L(".BPOSTACCUM");

    if (params.alphaScalingType == dlp::kernel_frame::scalingType::one) {
        // skip alpha scaling if alpha is 1
    } else {
        // alpha scaling
        RETURN_IF_ERROR(scaleAlpha());
    }

    // check if is_last_k is set
    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
    test(regTmp1, regTmp1);
    je(label_store_result, T_NEAR);

    // Create and set up kernelOphandler if there are post-ops
    if (!params.kernelOps.empty()) {
        gen::kernelOpsHandler kernelOpsHandler(this, params.kType);

        // post-ops are applied in the last k iteration, so we need to
        // scale beta before applying post-ops
        // To-Do: add support for beta scaling if beta is 1 using vaddps
        if (params.betaScalingType == dlp::kernel_frame::scalingType::zero) {
            // skip beta scaling if beta is 0
        } else {
            // beta scaling
            RETURN_IF_ERROR(scaleBeta());
        }

        RETURN_IF_ERROR((kernelOpsHandler.generateKernelOps(
            params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemm, currentMR,
            currentNR, useMask, numMaskRegs, cRegIdx, cReg)));

        // The gelu constants are embedded within the generated JIT kernel.
        // Otherwise a bug was observed whereby the address of gelu constants
        // inside the class turned out to be beyond what JIT can access.
        kernelOpsHandler.generateKernelOpsAttributes();

        // store result without fusing beta with store
        RETURN_IF_ERROR(storeResult(false, params.mLoop));
        jmp(".AfterStore", T_NEAR);
    }

    L(label_store_result);

    // Decoupled beta scaling: scaleBeta() is called separately from
    // storeResult(). This is required for c_downscale, and on AVX512_ZMM
    // it is also used for bReg == 1 since scaleBeta() handles both
    // row-major and column-major C on that ISA.
    bool decoupleBeta = (c_downscale < DLP_F32);
    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        decoupleBeta = decoupleBeta || (bReg == 1);
    }

    if (decoupleBeta) {
        if (params.betaScalingType != dlp::kernel_frame::scalingType::zero) {
            RETURN_IF_ERROR(scaleBeta());
        }
        RETURN_IF_ERROR(storeResult(false, params.mLoop));
    } else if (params.betaScalingType == dlp::kernel_frame::scalingType::zero) {
        RETURN_IF_ERROR(storeResult(false, params.mLoop));
    } else {
        RETURN_IF_ERROR(storeResult(true, params.mLoop));
    }

    L(".AfterStore");

    if (params.mLoop) {
        // get A pointer from stack
        // add psA to A pointer
        mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, psA)]);
        lea(regTmp1, ptr[regTmp1 * sizeof(float)]);
        lea(regAPtr, ptr[regAPtr + regTmp1]);

        // Update post_op_attr c_i. kernelOpsAttr is not a pointer, so adding
        // two offsets at the same time is safe.
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
        add(regTmp1, currentMR);
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_i)],
            regTmp1);

        moveCPtr(regCPtr, regRsC, currentMR);

        // decrement m_iter
        sub(regMiter, 1);

        jne(".BLOOP6X64I", T_NEAR);
    }
    vzeroupper();
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

// Generate kernel for F32 operations
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::generateKernel(utils::generatorParams& params)
{
    RETURN_IF_ERROR(checkValidF32GemmParams(params));

    // Check if this is a k=1 kernel (entire matrix iteration inside JIT)
    if (params.is_k1) {
        return generateKernelK1(params);
    }

    MR          = params.MR;
    currentMR   = MR;
    NR          = params.NR;
    currentNR   = NR;
    useMask     = params.useMask;
    numMaskRegs = params.numMaskRegs;
    c_downscale = params.c_downscale;

    RETURN_IF_ERROR(allocateMaskRegisters());
    RETURN_IF_ERROR(allocateReg());

    // There are 14 general purpose(64 bit) registers.
    // StackFrame manages these registers, since we are using
    // one register for the input parameter of the function,
    // the rest are used as scratch registers to store variables like
    // pointers, strides, counters, etc.
    // Note that all the scratch registers allocated by the stack frame
    // need not be used by the kernel.
    // Putting inside a scope so that some tables can be generated post
    // the ret instr. StackFrame inserts a ret instr in its destructor.
    // Allocate 64 bytes of local stack space:
    // - [0-15]: 16 bytes for intermediate storage (fringe BF16 elements - AVX2
    // only)
    // - [16-19]: value 16 for BF16→F32 shift
    // - [20-23]: 0x00010000 for bit 16 extraction
    // - [24-27]: 0x00007FFF for rounding
    // - [28-63]: 36 bytes padding/reserved
    Xbyak::util::StackFrame stackFrame(this, 1, 13, 64);
    initializeStackFrame(stackFrame);
    initializeParameters(params.mLoop);
    loadMasks();

    RETURN_IF_ERROR(generateIrLoop(params));

    return dlp::jit::jitGeneratorError::success;
}

// Generate kernel for F32 operations
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::generateKernelBodyK1(utils::generatorParams& params,
                                        gen::kernelOpsHandler* kernelOpsHandler)
{
    inLocalLabel();

    mov(regTmpAptr, regAPtr);
    mov(regBptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, b)]);

    // zero out accumulators
    regInit();

    // k is always 1 for k=1 kernel
    RETURN_IF_ERROR(kernelUnroll(1));

    L(".BPOSTACCUM");

    if (params.alphaScalingType == dlp::kernel_frame::scalingType::one) {
        // skip alpha scaling if alpha is 1
    } else {
        // alpha scaling
        RETURN_IF_ERROR(scaleAlpha());
    }

    // check if is_last_k is set
    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
    test(regTmp1, regTmp1);
#ifdef IR_JR_LOOP_ORDER_FOR_K1
    je(label_store_result_k1[currentMR - 1][currentNRIdx],
       T_NEAR); // Use 2D indexing: [MR][NR]
#else
    je(label_store_result_k1[currentNRIdx][currentMR - 1],
       T_NEAR); // Use 2D indexing: [NR][MR]
#endif

    // Create and set up kernelOphandler if there are post-ops
    if (!params.kernelOps.empty()) {
        // post-ops are applied in the last k iteration, so we need to
        // scale beta before applying post-ops
        // To-Do: add support for beta scaling if beta is 1 using vaddps
        if (params.betaScalingType == dlp::kernel_frame::scalingType::zero) {
            // skip beta scaling if beta is 0
        } else {
            // beta scaling
            RETURN_IF_ERROR(scaleBeta());
        }

        RETURN_IF_ERROR((kernelOpsHandler->generateKernelOps(
            params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemm, currentMR,
            currentNR, useMask, numMaskRegs, cRegIdx, cReg)));

        // The gelu constants are embedded within the generated JIT kernel.
        // Otherwise a bug was observed whereby the address of gelu constants
        // inside the class turned out to be beyond what JIT can access.
        kernelOpsHandler->generateKernelOpsAttributes();

        // store result without fusing beta with store
        RETURN_IF_ERROR(storeResult(false, params.mLoop));
        jmp(".AfterStore", T_NEAR);
    }

#ifdef IR_JR_LOOP_ORDER_FOR_K1
    L(label_store_result_k1[currentMR - 1]
                           [currentNRIdx]); // Use 2D indexing: [MR][NR]
#else
    L(label_store_result_k1[currentNRIdx]
                           [currentMR - 1]); // Use 2D indexing: [NR][MR]
#endif
    if (c_downscale < DLP_F32) {
        RETURN_IF_ERROR(scaleBeta());
        RETURN_IF_ERROR(storeResult(false, params.mLoop));

    } else {
        if (params.betaScalingType == dlp::kernel_frame::scalingType::zero) {
            // skip beta scaling if beta is 0
            RETURN_IF_ERROR(storeResult(false, params.mLoop));
        } else {
            // beta scaling
            RETURN_IF_ERROR(storeResult(true, params.mLoop));
        }
    }

    L(".AfterStore");

    outLocalLabel();
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::generateKernelK1(utils::generatorParams& params)
{
#ifdef IR_JR_LOOP_ORDER_FOR_K1
    return generateKernel_IR_JR(params);
#else
    return generateKernel_JR_IR(params);
#endif
}

// ----------------------------unused for now ----------------------------------
// this function generates the kernel with JR loop outside and IR loop inside
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::generateKernel_JR_IR(utils::generatorParams& params)
{
    MR          = params.MR;
    NR          = params.NR;
    c_downscale = params.c_downscale;

    // There are 14 general purpose(64 bit) registers.
    // StackFrame manages these registers, since we are using
    // one register for the input parameter of the function,
    // the rest are used as scratch registers to store variables like
    // pointers, strides, counters, etc.
    // Note that all the scratch registers allocated by the stack frame
    // need not be used by the kernel.
    // Putting inside a scope so that some tables can be generated post
    // the ret instr. StackFrame inserts a ret instr in its destructor.
    // Allocate 48 bytes of local stack space:
    // - 32 bytes for mask storage (fringe BF16 stores)
    // - 16 bytes for F32→BF16 conversion constants
    Xbyak::util::StackFrame stackFrame(this, 1, 13, 48);
    initializeStackFrame(stackFrame);
    regNiter = regTmp2;
    regNLeft = regKIter;
    initializeParameters(true);

    // Create kernel ops handler once for the entire kernel (like f32_gemv.cc)
    std::unique_ptr<gen::kernelOpsHandler> kernelOpsHandlerPtr;
    if (!params.kernelOps.empty()) {
        kernelOpsHandlerPtr =
            std::make_unique<gen::kernelOpsHandler>(this, params.kType);
    }

    // Calculate number of NR variants we'll need to handle at runtime
    // For now, we'll handle a single NR variant per kernel (as set in
    // params.NR) Future: extend to handle multiple NR values in one kernel
    numNRVariants = NR / numElemsPerReg + 1; // +1 for the mask variant
    numMRVariants = MR;

    // Initialize 2D label vectors: [numNRVariants][MR]
    std::vector<std::vector<Xbyak::Label>> m_labels_k1;
    Xbyak::Label                           done_k1;
    Xbyak::Label                           NIterLoop;
    std::vector<Xbyak::Label>              done_mVariants;
    std::vector<Xbyak::Label>              considerMLeft;
    Xbyak::Label                           considerNLeft;
    std::vector<Xbyak::Label>              skip_NR;

    m_labels_k1.resize(numNRVariants);
    label_store_result_k1.resize(numNRVariants);
    skip_NR.resize(numNRVariants); // Resize skip_NR for NR variant jumps
    considerMLeft.resize(
        numNRVariants); // One considerMLeft label per NR variant
    done_mVariants.resize(
        numNRVariants); // One done_mVariants label per NR variant

    // Generate all NR variants
    for (currentNRIdx = numNRVariants - 1; currentNRIdx >= 0; currentNRIdx--) {

        // Initialize the label vectors for the current NR variant
        m_labels_k1[currentNRIdx].resize(MR);
        label_store_result_k1[currentNRIdx].resize(MR);

        // Set currentNR based on currentNRIdx
        currentNR            = currentNRIdx * numElemsPerReg; // Full variants
        bool isMainNRVariant = currentNRIdx == numNRVariants - 1;

        // For the main NR variant, we first load nIter and check if > 0,
        // else we jump to considerNLeft label.
        if (isMainNRVariant) {
            mov(regNiter,
                ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nIter)]);
            test(regNiter, regNiter);
            je(considerNLeft, T_NEAR);
            L(NIterLoop);
        } else {
            cmp(regNLeft, currentNR); // Compare with immediate value
            // if nLeft < currentNR, we need to skip the current NR variant
            jl(skip_NR[currentNRIdx], T_NEAR);
        }

        // at the start of each NR variant, load A, B, C pointers
        mov(regAPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);

        useMask     = currentNRIdx == 0;
        numMaskRegs = useMask ? 1 : 0;
        RETURN_IF_ERROR(allocateMaskRegisters());
        loadMasks();

        // before executing the IR loop, we need to save the original
        // post_op_c_i value
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
        push(regTmp1);

        // Generate all MR variants for this NR variant
        for (iter_t mr = MR; mr > 0; mr--) {

            L(m_labels_k1[currentNRIdx][mr - 1]);

            bool isMainMRVariant = mr == MR;

            currentMR = mr;

            RETURN_IF_ERROR(allocateReg());

            inLocalLabel();
            // calculate and load pointers
            if (isMainMRVariant) {
                mov(regMiter,
                    ptr[stackPtr + offsetof(dlp::kernels::gemmParams, mIter)]);
                test(regMiter, regMiter);
                je(considerMLeft[currentNRIdx], T_NEAR);
                L(".BLOOP6X64I");
            }

            RETURN_IF_ERROR(
                generateKernelBodyK1(params, kernelOpsHandlerPtr.get()));

            if (isMainMRVariant) {
                // get A pointer from stack
                // add psA to A pointer
                mov(regTmp1,
                    ptr[stackPtr + offsetof(dlp::kernels::gemmParams, psA)]);
                lea(regTmp1, ptr[regTmp1 * sizeof(float)]);
                lea(regAPtr, ptr[regAPtr + regTmp1]);

                // Update post_op_attr c_i. kernelOpsAttr is not a pointer, so
                // adding two offsets at the same time is safe.
                mov(regTmp1,
                    ptr[stackPtr
                        + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                        + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
                add(regTmp1, currentMR);
                mov(ptr[stackPtr
                        + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                        + offsetof(dlp_gemm_post_op_attr, post_op_c_i)],
                    regTmp1);

                moveCPtr(regCPtr, regRsC, currentMR);

                // decrement m_iter
                sub(regMiter, 1);

                jne(".BLOOP6X64I", T_NEAR);

                L(considerMLeft[currentNRIdx]);
                // load m_left
                mov(regMiter,
                    ptr[stackPtr + offsetof(dlp::kernels::gemmParams, mLeft)]);
                test(regMiter, regMiter);
                je(done_mVariants[currentNRIdx], T_NEAR);

                // Jump to appropriate label based on mLeft value
                for (iter_t mr = MR - 1; mr > 0; mr--) {
                    cmp(regMiter, mr);
                    je(m_labels_k1[currentNRIdx][mr - 1],
                       T_NEAR); // Use 2D indexing: [NR][MR]
                }
            } else {
                jmp(done_mVariants[currentNRIdx], T_NEAR);
            }

            outLocalLabel();
        }

        L(done_mVariants[currentNRIdx]);

        // restore the original post_op_c_i value
        pop(regTmp1);
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_i)],
            regTmp1);

        // move B pointer to the next NR panel
        mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, b)]);
        add(regTmp1, currentNR * sizeof(float));
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, b)], regTmp1);

        // move c ptr by csC * currentNR * sizeof(float)
        mov(regCPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, c)]);
        mov(regTmp3, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, csC)]);
        lea(regTmp3, ptr[regTmp3 * sizeof(float)]);
        moveCPtr(regCPtr, regTmp3, currentNR);
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, c)], regCPtr);

        // update post_op_c_j value to point to next NR panel
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
        add(regTmp1, currentNR);
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_j)],
            regTmp1);

        if (isMainNRVariant) {

            // regNiter register may have been overwritten since regTmp2 is used
            // for nIter load regNiter freshly decrement, store-back and then
            // check if nIter is 0
            mov(regNiter,
                ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nIter)]);
            sub(regNiter, 1);
            mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nIter)],
                regNiter);
            jnz(NIterLoop, T_NEAR);

            L(considerNLeft);
            mov(regNLeft,
                ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nLeft)]);
            test(regNLeft, regNLeft);
            je(done_k1, T_NEAR);
        } else {
            // update nLeft value
            mov(regNLeft,
                ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nLeft)]);
            sub(regNLeft, currentNR);
            // store back the value of nLeft
            mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nLeft)],
                regNLeft);
            test(regNLeft, regNLeft);
            jz(done_k1, T_NEAR);
            L(skip_NR[currentNRIdx]);
        }
    }
    L(done_k1);

    vzeroupper();

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32<KType>::generateKernel_IR_JR(utils::generatorParams& params)
{
    MR          = params.MR;
    NR          = params.NR;
    c_downscale = params.c_downscale;

    // setting params.mLoop to false
    params.mLoop = false;

    // There are 14 general purpose(64 bit) registers.
    // StackFrame manages these registers, since we are using
    // one register for the input parameter of the function,
    // the rest are used as scratch registers to store variables like
    // pointers, strides, counters, etc.
    // Note that all the scratch registers allocated by the stack frame
    // need not be used by the kernel.
    // Putting inside a scope so that some tables can be generated post
    // the ret instr. StackFrame inserts a ret instr in its destructor.
    // Allocate 48 bytes of local stack space:
    // - 32 bytes for mask storage (fringe BF16 stores)
    // - 16 bytes for F32→BF16 conversion constants
    Xbyak::util::StackFrame stackFrame(this, 1, 13, 48);
    initializeStackFrame(stackFrame);
    regNiter = regTmp2;
    regNLeft = regKIter;
    initializeParameters(true);

    // Create kernel ops handler once for the entire kernel (like f32_gemv.cc)
    std::unique_ptr<gen::kernelOpsHandler> kernelOpsHandlerPtr;
    if (!params.kernelOps.empty()) {
        kernelOpsHandlerPtr =
            std::make_unique<gen::kernelOpsHandler>(this, params.kType);
    }

    // Calculate number of NR variants we'll need to handle at runtime
    // For now, we'll handle a single NR variant per kernel (as set in
    // params.NR) Future: extend to handle multiple NR values in one kernel
    numNRVariants = NR / numElemsPerReg + 1; // +1 for the mask variant
    numMRVariants = MR;

    // Initialize 2D label vectors: [numNRVariants][MR]
    std::vector<Xbyak::Label>              m_labels_k1;
    Xbyak::Label                           done_k1;
    std::vector<Xbyak::Label>              NIterLoop;
    Xbyak::Label                           MIterLoop;
    std::vector<Xbyak::Label>              done_nVariants;
    Xbyak::Label                           considerMLeft;
    std::vector<Xbyak::Label>              considerNLeft;
    std::vector<std::vector<Xbyak::Label>> skip_NR;

    m_labels_k1.resize(numMRVariants);
    label_store_result_k1.resize(numMRVariants);
    skip_NR.resize(numMRVariants); // Resize skip_NR for NR variant jumps
    considerNLeft.resize(
        numMRVariants); // One considerMLeft label per NR variant
    done_nVariants.resize(
        numMRVariants); // One done_nVariants label per MR variant
    NIterLoop.resize(numMRVariants);

    // Generate all MR variants for this NR variant
    for (iter_t mr = MR; mr > 0; mr--) {

        // Initialize the label vectors for the current NR variant
        label_store_result_k1[mr - 1].resize(numNRVariants);
        skip_NR[mr - 1].resize(numNRVariants);

        bool isMainMRVariant = mr == MR;

        currentMR = mr;

        L(m_labels_k1[mr - 1]);

        // calculate and load pointers
        if (isMainMRVariant) {
            mov(regMiter,
                ptr[stackPtr + offsetof(dlp::kernels::gemmParams, mIter)]);
            test(regMiter, regMiter);
            je(considerMLeft, T_NEAR);
            L(MIterLoop);
        }

        // before executing the JR loop, we need to save the original
        // post_op_c_j
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
        push(regTmp1);
        mov(regNiter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nIter)]);
        push(regNiter);
        mov(regNLeft,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nLeft)]);
        push(regNLeft);
        mov(regBptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, b)]);
        push(regBptr);

        // Generate all NR variants
        for (currentNRIdx = numNRVariants - 1; currentNRIdx >= 0;
             currentNRIdx--) {

            // Set currentNR based on currentNRIdx
            currentNR = currentNRIdx * numElemsPerReg; // Full variants
            bool isMainNRVariant = currentNRIdx == numNRVariants - 1;

            // For the main NR variant, we first load nIter and check if > 0,
            // else we jump to considerNLeft label.
            if (isMainNRVariant) {
                mov(regNiter,
                    ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nIter)]);
                test(regNiter, regNiter);
                je(considerNLeft[mr - 1], T_NEAR);
                L(NIterLoop[mr - 1]);
            } else {
                cmp(regNLeft, currentNR); // Compare with immediate value
                // if nLeft < currentNR, we need to skip the current NR variant
                jl(skip_NR[mr - 1][currentNRIdx], T_NEAR);
            }

            // at the start of each NR variant, load A, B, C pointers
            mov(regAPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);

            // Transpose Generator loads the value of n instead of nLeft to
            // decide how many cols to store. Hence move the value of nLeft to
            // n.
            mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, n)],
                regNLeft);

            useMask     = currentNRIdx == 0;
            numMaskRegs = useMask ? 1 : 0;
            RETURN_IF_ERROR(allocateMaskRegisters());
            loadMasks();

            RETURN_IF_ERROR(allocateReg());

            RETURN_IF_ERROR(
                generateKernelBodyK1(params, kernelOpsHandlerPtr.get()));

            // move B pointer to the next NR panel
            mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, b)]);
            add(regTmp1, currentNR * sizeof(float));
            mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, b)], regTmp1);

            // move c ptr by csC * currentNR * sizeof(float)
            mov(regTmp3,
                ptr[stackPtr + offsetof(dlp::kernels::gemmParams, csC)]);
            lea(regTmp3, ptr[regTmp3 * sizeof(float)]);
            moveCPtr(regCPtr, regTmp3, currentNR);

            // update post_op_c_j value to point to next NR panel
            mov(regTmp1,
                ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                    + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
            add(regTmp1, currentNR);
            mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                    + offsetof(dlp_gemm_post_op_attr, post_op_c_j)],
                regTmp1);

            if (isMainNRVariant) {

                // regNiter register may have been overwritten since regTmp2 is
                // used for nIter load regNiter freshly decrement, store-back
                // and then check if nIter is 0
                mov(regNiter,
                    ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nIter)]);
                sub(regNiter, 1);
                mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nIter)],
                    regNiter);
                jnz(NIterLoop[mr - 1], T_NEAR);

                L(considerNLeft[mr - 1]);
                mov(regNLeft,
                    ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nLeft)]);
                test(regNLeft, regNLeft);
                je(done_nVariants[mr - 1], T_NEAR);
            } else {
                // update nLeft value
                mov(regNLeft,
                    ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nLeft)]);
                sub(regNLeft, currentNR);
                // store back the value of nLeft
                mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nLeft)],
                    regNLeft);
                test(regNLeft, regNLeft);
                jz(done_nVariants[mr - 1], T_NEAR);
                L(skip_NR[mr - 1][currentNRIdx]);
            }
        }

        L(done_nVariants[mr - 1]);
        // restore the original post_op_c_j value
        pop(regBptr);
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, b)], regBptr);
        pop(regNLeft);
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nLeft)],
            regNLeft);
        pop(regNiter);
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nIter)],
            regNiter);
        pop(regTmp1);
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_j)],
            regTmp1);

        if (isMainMRVariant) {
            // get A pointer from stack
            // add psA to A pointer
            mov(regTmp1,
                ptr[stackPtr + offsetof(dlp::kernels::gemmParams, psA)]);
            lea(regTmp1, ptr[regTmp1 * sizeof(float)]);
            lea(regAPtr, ptr[regAPtr + regTmp1]);
            mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)], regAPtr);

            // Update post_op_attr c_i. kernelOpsAttr is not a pointer, so
            // adding two offsets at the same time is safe.
            mov(regTmp1,
                ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                    + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
            add(regTmp1, currentMR);
            mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                    + offsetof(dlp_gemm_post_op_attr, post_op_c_i)],
                regTmp1);

            mov(regCPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, c)]);
            moveCPtr(regCPtr, regRsC, currentMR);
            mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, c)], regCPtr);

            // decrement m_iter
            sub(regMiter, 1);

            jne(MIterLoop, T_NEAR);

            L(considerMLeft);
            // load m_left
            mov(regMiter,
                ptr[stackPtr + offsetof(dlp::kernels::gemmParams, mLeft)]);
            test(regMiter, regMiter);
            je(done_k1, T_NEAR);

            // Jump to appropriate label based on mLeft value
            for (iter_t mr_idx = MR - 1; mr_idx > 0; mr_idx--) {
                cmp(regMiter, mr_idx);
                je(m_labels_k1[mr_idx - 1],
                   T_NEAR); // Use 2D indexing: [NR][MR]
            }
        } else {
            jmp(done_k1, T_NEAR);
        }
    }

    L(done_k1);

    vzeroupper();

    return dlp::jit::jitGeneratorError::success;
}

} // namespace amdzen::GEMMcodeGenerator

// Explicit template instantiations for classes
template class amdzen::GEMMcodeGenerator::jitGEMMF32<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
template class amdzen::GEMMcodeGenerator::jitGEMMF32<
    amdzen::utils::kernelInstrType::avx512_ymm_32_reg>;
template class amdzen::GEMMcodeGenerator::jitGEMMF32<
    amdzen::utils::kernelInstrType::avx2_ymm_16_reg>;
