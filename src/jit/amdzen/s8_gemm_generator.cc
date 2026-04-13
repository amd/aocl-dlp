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

#include "jit_register/jit_register.hh"
#include "s8_gemm_generator.hh"

namespace amdzen::GEMMcodeGenerator {

using namespace Xbyak;

template<utils::kernelInstrType KType>
jitGEMMS8<KType>::jitGEMMS8(size_t maxSize)
    : Xbyak::CodeGenerator(maxSize, Xbyak::AutoGrow)
{
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMS8<KType>::allocateReg()
{
    // Using int32 since accumulation is done in int32
    int nElemsPerReg =
        RegBytes
        / sizeof(int32_t); // Number of int32 elements per vector register
    bFullReg =
        (NR) / nElemsPerReg; // Number of full vector registers needed for B
    bMaskReg = (useMask ? 1 : 0); // Number of mask registers needed for B (1 if
                                  // needed, else 0)
    bReg      = bFullReg + bMaskReg; // Total number of registers needed for B
    cReg      = MR * bReg;           // Total number of registers needed for C
    vec128Reg = 1; // Always reserve one register for converting int8 to uint8.
    aReg = numRegs - cReg - vec128Reg; // Remaining registers are dedicated to A

    // Check if we have enough registers
    if (aReg < 1) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
    cRegIdx      = numRegs - cReg; // Starting index for C registers
    bRegIdx      = cRegIdx - bReg; // Starting index for B registers
    vec128RegIdx = bRegIdx - 1;    // Index for reserved vec128Register
    aRegIdx      = 0;              // Starting index for A registers

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMMS8<KType>::initializeStackFrame(Xbyak::util::StackFrame& stackFrame)
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
void
jitGEMMS8<KType>::initializeParameters(bool addIrLoop)
{
    if (addIrLoop) {
        // Load A and mIter pointers from stack to registers for access within
        // the IR-loop.
        mov(regAPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);
        mov(regMiter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, mIter)]);
    } else {
        // Store a copy of A pointer in a temporary register.
        mov(regTmpAptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);
    }

    // Load other parameters from stack to registers.
    mov(regCPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, c)]);
    mov(regRsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsA)]);
    mov(regCsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, csA)]);
    mov(regRsB, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsB)]);
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsC)]);

    lea(regRsC, ptr[regRsC * sizeof(int32_t)]);

    // Store original C pointer in a temporary register.
    mov(regTmpCptr, regCPtr);

    mov(regTmp2,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);

    for (int i = 0; i < utils::NUM_USABLE_MASKS; i++) {
        mask_regs[i] = Xbyak::Opmask(utils::MASK_START_IDX + i);
    }

    kmovw(mask_regs[0],
          ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeftmask)]);

    if (useMask) {
        kmovw(mask_regs[1],
              ptr[stackPtr + offsetof(dlp::kernels::gemmParams, maskS32)]);
    }
}

template<utils::kernelInstrType KType>
void
jitGEMMS8<KType>::initializeRegisters()
{
    using RegType = typename Traits::RegType;

    vxorps(RegType(cRegIdx), RegType(cRegIdx), RegType(cRegIdx));
    for (iter_t i = 1; i < cReg; ++i) {
        vmovaps(RegType(cRegIdx + i), RegType(cRegIdx));
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMS8<KType>::loadBValues()
{
    using RegType = typename Traits::RegType;
    for (iter_t i = 0; i < bFullReg; ++i) {
        vmovdqu32(RegType(bRegIdx + i), ptr[regBptr + i * RegBytes]);
    }

    // Handle masked b loads

    if (useMask) {
        int maskRegIndex = bRegIdx + bFullReg;
        if (maskRegIndex >= numRegs) {
            return dlp::jit::jitGeneratorError::badKernelInfo;
        }

        vmovdqu8(RegType(maskRegIndex), ptr[regBptr + bFullReg * RegBytes]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMS8<KType>::BroadcastAVNNIB(bool isVNNIrem)
{
    using RegType = typename Traits::RegType;

    for (iter_t i = 0; i < MR; ++i) {
        if (isVNNIrem) {

            // Masked load with zero-extension: load kLeft bytes, zero the rest
            vmovdqu8(Xbyak::Ymm(aRegIdx) | mask_regs[0] | T_z, ptr[regTmpAptr]);

            // Broadcast the loaded 4-byte pattern to all lanes
            vpbroadcastd(RegType(aRegIdx), Xbyak::Xmm(aRegIdx));
        } else {
            vpbroadcastd(RegType(aRegIdx), ptr[regTmpAptr]);
        }
        vpaddb(RegType(aRegIdx), RegType(aRegIdx), RegType(vec128RegIdx));

        add(regTmpAptr, regRsA);

        for (iter_t j = 0; j < bReg; ++j) {
            vpdpbusd(RegType(cRegIdx + i * bReg + j), RegType(aRegIdx),
                     RegType(bRegIdx + j));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMS8<KType>::kLoop(int unroll, bool isVNNIrem)
{
    dlp::jit::jitGeneratorError err = dlp::jit::jitGeneratorError::error;

    // Unrolling the kernel loop
    for (iter_t u = 0; u < unroll; ++u) {
        // Copy a ptr to another register
        mov(regTmp1, regTmpAptr);

        // load B registers
        RETURN_IF_ERROR(loadBValues());
        add(regBptr, regRsB); // Move B pointer to next column

        RETURN_IF_ERROR(BroadcastAVNNIB(isVNNIrem));
        lea(regTmpAptr, ptr[regTmp1 + regCsA]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMS8<KType>::loadBSumValues()
{
    using RegType = typename Traits::RegType;

    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, b_col_sum_vec)]);
    mov(regTmp3,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, b_sum_offset)]);

    lea(regTmp1, ptr[regTmp1 + (regTmp3 * sizeof(int32_t))]);

    for (iter_t i = 0; i < bFullReg; ++i) {
        vmovdqu8(RegType(bRegIdx + i), ptr[regTmp1 + i * RegBytes]);
    }

    // Handle masked b_sum loads
    if (bMaskReg > 0) {
        vmovdqu8(RegType(bRegIdx + bFullReg), ptr[regTmp1]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMS8<KType>::conversionCompensation()
{
    using RegType = typename Traits::RegType;

    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
    test(regTmp1, regTmp1);
    je("BYPASS_COMP", T_NEAR);

    RETURN_IF_ERROR(loadBSumValues());

    for (iter_t j = 0; j < bReg; ++j) {
        for (iter_t i = 0; i < MR; ++i) {
            vpsubd(RegType(cRegIdx + i * bReg + j),
                   RegType(cRegIdx + i * bReg + j), RegType(bRegIdx + j));
        }
    }

    L("BYPASS_COMP");

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMS8<KType>::scaleAlpha()
{
    using RegType   = typename Traits::RegType;
    int alphaRegIdx = aRegIdx;
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, alpha)]);
    vpbroadcastd(RegType(alphaRegIdx), ptr[regTmp1]);

    for (iter_t i = 0; i < cReg; i++) {
        vpmulld(RegType(cRegIdx + i), RegType(cRegIdx + i),
                RegType(alphaRegIdx));
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMS8<KType>::updateCBufferPointers()
{
    mov(regTmpCptr,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, buf_downscale)]);

    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);

    if (c_downscale == DLP_BF16) {
        lea(regTmp1, ptr[regTmp1 * 2]);
    } else if (c_downscale == DLP_F32) {
        lea(regTmp1, ptr[regTmp1 * 4]);
    }

    add(regTmpCptr, regTmp1);

    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, rs_c_downscale)]);

    if (c_downscale == DLP_BF16) {
        lea(regTmp1, ptr[regTmp1 * 2]);
    } else if (c_downscale == DLP_F32) {
        lea(regTmp1, ptr[regTmp1 * 4]);
    }

    mov(regKIter, regTmp2);
    imul(regKIter, regTmp1); // post_ops_c_i * rs_c_downscale
    add(regTmpCptr, regKIter);
    // regTmp1 now contains rs_c_downscale for caller to use

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMS8<KType>::scaleBeta()
{
    int betaRegIdx = aRegIdx;

    // Load beta scaling factor
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, beta)]);
    vpbroadcastd(Zmm(betaRegIdx), ptr[regTmp1]);
    mov(regTmpCptr, regCPtr);

    if (c_downscale == DLP_S8) {
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_first_k)]);
        test(regTmp1, regTmp1);
        je("BETAOP", T_NEAR);

        RETURN_IF_ERROR(updateCBufferPointers());

        for (iter_t i = 0; i < MR; i++) {
            for (iter_t j = 0; j < bFullReg; j++) {
                vmovdqu8(Xbyak::Xmm(bRegIdx + j), ptr[regTmpCptr + j * 16]);
                vpmovsxbd(RegType(bRegIdx + j), Xbyak::Xmm(bRegIdx + j));
                vpmulld(RegType(bRegIdx + j), RegType(bRegIdx + j),
                        RegType(betaRegIdx));
                vpaddd(RegType(cRegIdx + i * bReg + j),
                       RegType(cRegIdx + i * bReg + j), RegType(bRegIdx + j));
            }
            if (bMaskReg > 0) {
                vmovdqu8(Xbyak::Xmm(bRegIdx + bFullReg) | mask_regs[1] | T_z,
                         ptr[regTmpCptr + bFullReg * 16]);
                vpmovsxbd(RegType(bRegIdx + bFullReg),
                          Xbyak::Xmm(bRegIdx + bFullReg));
                vpmulld(RegType(bRegIdx + bFullReg),
                        RegType(bRegIdx + bFullReg), RegType(betaRegIdx));
                vpaddd(RegType(cRegIdx + i * bReg + bFullReg),
                       RegType(cRegIdx + i * bReg + bFullReg),
                       RegType(bRegIdx + bFullReg));
            }
            add(regTmpCptr, regTmp1);
        }

        jmp("BETAOP_END", T_NEAR);
        L("BETAOP");
    } else if (c_downscale == DLP_U8) {
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_first_k)]);
        test(regTmp1, regTmp1);
        je("BETAOP", T_NEAR);

        RETURN_IF_ERROR(updateCBufferPointers());

        for (iter_t i = 0; i < MR; i++) {
            for (iter_t j = 0; j < bFullReg; j++) {
                vmovdqu8(Xbyak::Xmm(bRegIdx + j),
                         ptr[regTmpCptr + j * (RegBytes / 4)]);
                vpmovzxbd(RegType(bRegIdx + j), Xbyak::Xmm(bRegIdx + j));
                vpmulld(RegType(bRegIdx + j), RegType(bRegIdx + j),
                        RegType(betaRegIdx));
                vpaddd(RegType(cRegIdx + i * bReg + j),
                       RegType(cRegIdx + i * bReg + j), RegType(bRegIdx + j));
            }

            // Handle masked beta scaling
            if (bMaskReg > 0) {
                vmovdqu8(Xbyak::Xmm(bRegIdx + bFullReg) | mask_regs[1] | T_z,
                         ptr[regTmpCptr + (bFullReg * (RegBytes / 4))]);
                vpmovzxbd(RegType(bRegIdx + bFullReg),
                          Xbyak::Xmm(bRegIdx + bFullReg));
                vpmulld(RegType(bRegIdx + bFullReg),
                        RegType(bRegIdx + bFullReg), RegType(betaRegIdx));
                vpaddd(RegType(cRegIdx + i * bReg + bFullReg),
                       RegType(cRegIdx + i * bReg + bFullReg),
                       RegType(bRegIdx + bFullReg));
            }

            add(regTmpCptr, regTmp1);
        }

        jmp("BETAOP_END", T_NEAR);
        L("BETAOP");
    } else if (c_downscale == DLP_BF16) {
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_first_k)]);
        test(regTmp1, regTmp1);
        je("BETAOP", T_NEAR);

        RETURN_IF_ERROR(updateCBufferPointers());

        for (iter_t i = 0; i < MR; i++) {
            for (iter_t j = 0; j < bFullReg; j++) {
                vmovdqu16(Xbyak::Ymm(bRegIdx + j),
                          ptr[regTmpCptr + j * (RegBytes / 2)]);
                vpmovsxwd(RegType(bRegIdx + j), Xbyak::Ymm(bRegIdx + j));
                vpslld(RegType(bRegIdx + j), RegType(bRegIdx + j), 0x10);
                vcvtps2dq(RegType(bRegIdx + j), RegType(bRegIdx + j));
                vpmulld(RegType(bRegIdx + j), RegType(bRegIdx + j),
                        RegType(betaRegIdx));
                vpaddd(RegType(cRegIdx + i * bReg + j),
                       RegType(cRegIdx + i * bReg + j), RegType(bRegIdx + j));
            }

            // Handle masked beta scaling
            if (bMaskReg > 0) {
                // Use zero-masking (T_z) to zero unmasked elements
                vmovdqu16(Xbyak::Ymm(bRegIdx + bFullReg) | mask_regs[1] | T_z,
                          ptr[regTmpCptr + (bFullReg * (RegBytes / 2))]);
                vpmovsxwd(RegType(bRegIdx + bFullReg),
                          Xbyak::Ymm(bRegIdx + bFullReg));
                vpslld(RegType(bRegIdx + bFullReg), RegType(bRegIdx + bFullReg),
                       0x10);
                vcvtps2dq(RegType(bRegIdx + bFullReg),
                          RegType(bRegIdx + bFullReg));
                vpmulld(RegType(bRegIdx + bFullReg),
                        RegType(bRegIdx + bFullReg), RegType(betaRegIdx));
                vpaddd(RegType(cRegIdx + i * bReg + bFullReg),
                       RegType(cRegIdx + i * bReg + bFullReg),
                       RegType(bRegIdx + bFullReg));
            }

            add(regTmpCptr, regTmp1);
        }

        jmp("BETAOP_END", T_NEAR);
        L("BETAOP");
    } else if (c_downscale == DLP_F32) {
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_first_k)]);
        test(regTmp1, regTmp1);
        je("BETAOP", T_NEAR);

        RETURN_IF_ERROR(updateCBufferPointers());

        for (iter_t i = 0; i < MR; i++) {
            for (iter_t j = 0; j < bFullReg; j++) {
                vcvtps2dq(RegType(bRegIdx + j), ptr[regTmpCptr + j * RegBytes]);
                vpmulld(RegType(bRegIdx + j), RegType(bRegIdx + j),
                        RegType(betaRegIdx));
                vpaddd(RegType(cRegIdx + i * bReg + j),
                       RegType(cRegIdx + i * bReg + j), RegType(bRegIdx + j));
            }

            // Handle masked beta scaling
            if (bMaskReg > 0) {
                vcvtps2dq(RegType(bRegIdx + bFullReg) | mask_regs[1] | T_z,
                          ptr[regTmpCptr + bFullReg * RegBytes]);
                vpmulld(RegType(bRegIdx + bFullReg),
                        RegType(bRegIdx + bFullReg), RegType(betaRegIdx));
                vpaddd(RegType(cRegIdx + i * bReg + bFullReg),
                       RegType(cRegIdx + i * bReg + bFullReg),
                       RegType(bRegIdx + bFullReg));
            }

            add(regTmpCptr, regTmp1);
        }

        jmp("BETAOP_END", T_NEAR);
        L("BETAOP");
    }

    for (iter_t i = 0; i < MR; i++) {
        for (iter_t j = 0; j < bFullReg; j++) {
            vmovdqu32(RegType(bRegIdx + j), ptr[regTmpCptr + j * RegBytes]);
            vpmulld(RegType(bRegIdx + j), RegType(bRegIdx + j),
                    RegType(betaRegIdx));
            vpaddd(RegType(cRegIdx + i * bReg + j),
                   RegType(cRegIdx + i * bReg + j), RegType(bRegIdx + j));
        }

        // Handle masked beta scaling
        if (bMaskReg > 0) {
            vmovdqu32(RegType(bRegIdx + bFullReg) | mask_regs[1] | T_z,
                      ptr[regTmpCptr + bFullReg * RegBytes]);
            vpmulld(RegType(bRegIdx + bFullReg), RegType(bRegIdx + bFullReg),
                    RegType(betaRegIdx));
            vpaddd(RegType(cRegIdx + i * bReg + bFullReg),
                   RegType(cRegIdx + i * bReg + bFullReg),
                   RegType(bRegIdx + bFullReg));
        }

        add(regTmpCptr, regRsC);
    }

    L("BETAOP_END");
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMS8<KType>::storeResult(bool hasPostOps)
{
    using RegType = typename Traits::RegType;

    // Defining labels locally to avoid redefinition issues
    Xbyak::Label label_storeop, label_storeop_end;

    mov(regTmpCptr, regCPtr);

    if (c_downscale == DLP_S8) {
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
        test(regTmp1, regTmp1);
        je(label_storeop, T_NEAR);

        RETURN_IF_ERROR(updateCBufferPointers());

        for (iter_t i = 0; i < MR; ++i) {
            for (iter_t j = 0; j < bFullReg; ++j) {
                if (hasPostOps) {
                    // Convert post-ops accumulated result from F32 to S32.
                    vcvtps2dq(RegType(aRegIdx),
                              RegType(cRegIdx + i * bReg + j));
                    vpmovsdb(ptr[regTmpCptr + j * (RegBytes / 4)],
                             RegType(aRegIdx));
                } else {
                    // Convert accumulated S32 results to S8 with saturation
                    vpmovsdb(ptr[regTmpCptr + j * (RegBytes / 4)],
                             RegType(cRegIdx + i * bReg + j));
                }
            }

            // Masked Store
            if (bMaskReg > 0) {
                if (hasPostOps) {
                    // Convert post-ops accumulated result from F32 to S32.
                    vcvtps2dq(RegType(aRegIdx),
                              RegType(cRegIdx + i * bReg + bFullReg));
                    vpmovsdb(ptr[regTmpCptr + bFullReg * (RegBytes / 4)]
                                 | mask_regs[1],
                             RegType(aRegIdx));
                } else {
                    vpmovsdb(ptr[regTmpCptr + bFullReg * (RegBytes / 4)]
                                 | mask_regs[1],
                             RegType(cRegIdx + i * bReg + bFullReg));
                }
            }

            add(regTmpCptr, regTmp1);
        }

        jmp(label_storeop_end, T_NEAR);
        L(label_storeop);
    } else if (c_downscale == DLP_U8) {
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
        test(regTmp1, regTmp1);
        je(label_storeop, T_NEAR);

        RETURN_IF_ERROR(updateCBufferPointers());

        // Convert S32 to U8 with saturation by first clamping to [0, 255]
        // Zero out the temporary register for clamping
        vpxord(RegType(aRegIdx + 1), RegType(aRegIdx + 1),
               RegType(aRegIdx + 1)); // 0
        mov(regKIter, 255);
        vpbroadcastd(RegType(aRegIdx + 2), regKIter.cvt32()); // 255

        for (iter_t i = 0; i < MR; ++i) {
            for (iter_t j = 0; j < bFullReg; ++j) {
                if (hasPostOps) {
                    // Convert post-ops accumulated result from F32 to S32.
                    vcvtps2dq(RegType(cRegIdx + i * bReg + j),
                              RegType(cRegIdx + i * bReg + j));
                }
                // Convert the accumulated result from S32 to U8.
                vpmaxsd(RegType(aRegIdx), RegType(cRegIdx + i * bReg + j),
                        RegType(aRegIdx + 1));
                vpminsd(RegType(aRegIdx), RegType(aRegIdx),
                        RegType(aRegIdx + 2));
                vpmovdb(ptr[regTmpCptr + j * 16], RegType(aRegIdx));
            }

            // Masked Store
            if (bMaskReg > 0) {
                if (hasPostOps) {
                    // Convert post-ops accumulated result from F32 to S32.
                    vcvtps2dq(RegType(cRegIdx + i * bReg + bFullReg),
                              RegType(cRegIdx + i * bReg + bFullReg));
                }
                // Convert the accumulated result from S32 to U8.
                vpmaxsd(RegType(aRegIdx),
                        RegType(cRegIdx + i * bReg + bFullReg),
                        RegType(aRegIdx + 1));
                vpminsd(RegType(aRegIdx), RegType(aRegIdx),
                        RegType(aRegIdx + 2));
                vpmovdb(ptr[regTmpCptr + bFullReg * 16] | mask_regs[1],
                        RegType(aRegIdx));
            }

            add(regTmpCptr, regTmp1);
        }

        jmp(label_storeop_end, T_NEAR);
        L(label_storeop);
    }
    if (c_downscale == DLP_BF16) {
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
        test(regTmp1, regTmp1);
        je(label_storeop, T_NEAR);

        RETURN_IF_ERROR(updateCBufferPointers());

        for (iter_t i = 0; i < MR; ++i) {
            for (iter_t j = 0; j < bFullReg; ++j) {
                if (!hasPostOps) {
                    // Convert accumulated S32 results to F32.
                    vcvtdq2ps(RegType(cRegIdx + i * bReg + j),
                              RegType(cRegIdx + i * bReg + j));
                }

                // Convert F32 to BF16
                vpsrld(RegType(aRegIdx), RegType(cRegIdx + i * bReg + j), 16);
                mov(regTmp3, 0x00000001);
                vpbroadcastd(RegType(aRegIdx + 1), regTmp3.cvt32());
                vpandd(RegType(aRegIdx), RegType(aRegIdx),
                       RegType(aRegIdx + 1));
                mov(regTmp3, 0x00007FFF);
                vpbroadcastd(RegType(aRegIdx + 1), regTmp3.cvt32());
                vpaddd(RegType(aRegIdx + 2), RegType(cRegIdx + i * bReg + j),
                       RegType(aRegIdx + 1));
                vpaddd(RegType(aRegIdx + 2), RegType(aRegIdx + 2),
                       RegType(aRegIdx));
                vpsrld(RegType(aRegIdx + 2), RegType(aRegIdx + 2), 16);
                vpmovdw(Xbyak::Ymm(aRegIdx + 2), RegType(aRegIdx + 2));
                vmovdqu16(ptr[regTmpCptr + j * (RegBytes / 2)],
                          Xbyak::Ymm(aRegIdx + 2));
            }

            // Masked Store
            if (bMaskReg > 0) {
                if (!hasPostOps) {
                    // Convert accumulated S32 results to F32.
                    vcvtdq2ps(RegType(cRegIdx + i * bReg + bFullReg),
                              RegType(cRegIdx + i * bReg + bFullReg));
                }

                // Convert F32 to BF16
                vpsrld(RegType(aRegIdx), RegType(cRegIdx + i * bReg + bFullReg),
                       16);
                mov(regTmp3, 0x00000001);
                vpbroadcastd(RegType(aRegIdx + 1), regTmp3.cvt32());
                vpandd(RegType(aRegIdx), RegType(aRegIdx),
                       RegType(aRegIdx + 1));
                mov(regTmp3, 0x00007FFF);
                vpbroadcastd(RegType(aRegIdx + 1), regTmp3.cvt32());
                vpaddd(RegType(aRegIdx + 2),
                       RegType(cRegIdx + i * bReg + bFullReg),
                       RegType(aRegIdx + 1));
                vpaddd(RegType(aRegIdx + 2), RegType(aRegIdx + 2),
                       RegType(aRegIdx));
                vpsrld(RegType(aRegIdx + 2), RegType(aRegIdx + 2), 16);
                vpmovdw(Xbyak::Ymm(aRegIdx + 2), RegType(aRegIdx + 2));
                vmovdqu16(ptr[regTmpCptr + bFullReg * (RegBytes / 2)]
                              | mask_regs[1],
                          Xbyak::Ymm(aRegIdx + 2));
            }

            add(regTmpCptr, regTmp1);
        }

        jmp(label_storeop_end, T_NEAR);
        L(label_storeop);
    } else if (c_downscale == DLP_F32) {
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
        test(regTmp1, regTmp1);
        je(label_storeop, T_NEAR);

        RETURN_IF_ERROR(updateCBufferPointers());

        for (iter_t i = 0; i < MR; ++i) {
            for (iter_t j = 0; j < bFullReg; ++j) {
                if (!hasPostOps) {
                    // Converting the accumulated result from S32 to F32.
                    vcvtdq2ps(RegType(cRegIdx + i * bReg + j),
                              RegType(cRegIdx + i * bReg + j));
                }

                vmovups(ptr[regTmpCptr + j * RegBytes],
                        RegType(cRegIdx + i * bReg + j));
            }

            // Masked Store
            if (bMaskReg > 0) {
                if (!hasPostOps) {
                    // Converting the accumulated result from S32 to F32.
                    vcvtdq2ps(RegType(cRegIdx + i * bReg + bFullReg),
                              RegType(cRegIdx + i * bReg + bFullReg));
                }

                vmovups(ptr[regTmpCptr + bFullReg * RegBytes] | mask_regs[1],
                        RegType(cRegIdx + i * bReg + bFullReg));
            }

            add(regTmpCptr, regTmp1);
        }

        jmp(label_storeop_end, T_NEAR);
        L(label_storeop);
    }

    // Default S32 store
    for (iter_t i = 0; i < MR; ++i) {
        // Regular Unmasked Store
        for (iter_t j = 0; j < bFullReg; ++j) {
            if (hasPostOps) {
                // Convert post-ops accumulated result from F32 to S32.
                vcvtps2dq(RegType(cRegIdx + i * bReg + j),
                          RegType(cRegIdx + i * bReg + j));
            }
            vmovdqu32(ptr[regTmpCptr + j * RegBytes],
                      RegType(cRegIdx + i * bReg + j));
        }

        // Masked Store
        if (bMaskReg > 0) {
            if (hasPostOps) {
                // Convert post-ops accumulated result from F32 to S32.
                vcvtps2dq(RegType(cRegIdx + i * bReg + bFullReg),
                          RegType(cRegIdx + i * bReg + bFullReg));
            }

            vmovdqu32(ptr[regTmpCptr + bFullReg * RegBytes] | mask_regs[1]
                          | T_z,
                      RegType(cRegIdx + i * bReg + bFullReg));
        }

        add(regTmpCptr, regRsC);
    }

    L(label_storeop_end);
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMMS8<KType>::moveCPtr()
{
    // Update C pointer for next row : cbuf += m * MR * rs_c
    // lea() doesn't work with non-power-of-2 values, and avx2
    // doesn't support imul. Hence, decomposing m value to power
    // of 2 to handle all the cases commonly. Let's say if MR = 13
    // then we can represent it as 8 + 4 + 1
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

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMS8<KType>::generateIrLoop(utils::generatorParams& params)
{
    inLocalLabel();
    if (params.mLoop) {
        L(".BLOOP6x64I"); // Main 6x64 loop label.

        // Move A pointer to a temporary register.
        mov(regTmpAptr, regAPtr);
    }
    mov(regBptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, b)]);

    initializeRegisters(); // zero-out accumulators

    // Broadcast 128 to a vector register.
    // This value will be added to A to convert the broadcasted A values from
    // signed int8 to unsigned int8 for the VNNI instruction.
    mov(regTmp1, 128);
    vxorps(RegType(vec128RegIdx), RegType(vec128RegIdx), RegType(vec128RegIdx));
    vpbroadcastb(RegType(vec128RegIdx), regTmp1.cvt8());

    // Jump to K-left loop if kIter is zero.
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kIterBP)]);
    test(regKIter, regKIter);    // if regKIter == 0, jump to k-left loop.
    je(".BCONSIDKLEFT", T_NEAR); // using T_NEAR, since size between jmp and
                                 // label is larger than 127 byte.
                                 // Xbyak throws error otherwise.

    // K-loop
    L(".BLOOPKITER");
    RETURN_IF_ERROR(kLoop(params.K_UNROLL, false));
    sub(regKIter, 1);
    jne(".BLOOPKITER", T_NEAR);

    // K-left loop
    L(".BCONSIDKLEFT");
    // load kLeft into regKIter.
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeft)]);
    test(regKIter, regKIter); // if regKIter == 0, skip k-left loop.
    je(".BPOSTACCUM", T_NEAR);

    RETURN_IF_ERROR(kLoop(1, true));

    // Post-accumulation operations
    L(".BPOSTACCUM");

    // Compensate for conversion of A from int8 to uint8 for the VNNI
    // instruction.
    RETURN_IF_ERROR(conversionCompensation());

    L(".SCALING");
    if (params.alphaScalingType != dlp::kernel_frame::scalingType::one) {
        // Scale by alpha
        RETURN_IF_ERROR(scaleAlpha());
    }

    if (params.betaScalingType != dlp::kernel_frame::scalingType::zero) {
        // Scale by beta
        RETURN_IF_ERROR(scaleBeta());
    }

    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
    test(regTmp1, regTmp1);
    je(".STOREC", T_NEAR); // skip post-ops if not the final k iteration

    // Create kernel ops handler if there are post-ops
    std::unique_ptr<gen::kernelOpsHandler<KType>> kernelOpsHandlerPtr;
    if (!params.kernelOps.empty()) {
        kernelOpsHandlerPtr =
            std::make_unique<gen::kernelOpsHandler<KType>>(this);
    }

    if (kernelOpsHandlerPtr) {
        using VecPoolType =
            utils::registerPool<typename Traits::RegType, Traits::numRegs>;
        using MaskPoolType =
            utils::registerPool<Xbyak::Opmask, Traits::numMaskRegs>;

        // Convert to F32 since post-ops expect accumulators in F32.
        for (iter_t i = 0; i < cReg; i++) {
            vcvtdq2ps(RegType(cRegIdx + i), RegType(cRegIdx + i));
        }

        VecPoolType vecPool;
        vecPool.setAccumulators(cRegIdx, cReg);
        RETURN_IF_ERROR(vecPool.init(this, Traits::regBytes));

        // S8 GEMM preserves 2 masks when fringe: one for S32/F32 data, one for
        // comparison
        int          maskCount = useMask ? 2 : 1;
        MaskPoolType maskPool;
        maskPool.addPreserve(utils::MASK_START_IDX, maskCount);
        RETURN_IF_ERROR(maskPool.init(this, utils::maskSaveWidth<KType>(),
                                      Traits::reservedMaskBits));

        int maskOffset =
            useMask
                ? static_cast<int>(offsetof(dlp::kernels::gemmParams, maskS32))
                : -1;

        RETURN_IF_ERROR((kernelOpsHandlerPtr->generateKernelOps(
            params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemm, params.MR,
            params.NR, params.useMask, params.numMaskRegs, cRegIdx, cReg,
            vecPool, maskPool, maskOffset)));

        // store C assuming F32 accumulators after post-ops
        RETURN_IF_ERROR(storeResult(true));
        jmp(".POST_STOREC", T_NEAR);
    }

    // store C
    L(".STOREC");
    RETURN_IF_ERROR(storeResult());

    L(".POST_STOREC");

    if (params.mLoop) {
        moveCPtr();

        // get A pointer from stack
        // add psA to A pointer
        mov(regTmpAptr, regAPtr);
        mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, psA)]);
        lea(regTmpAptr, ptr[regTmpAptr + regTmp1]);
        mov(regAPtr, regTmpAptr);

        // Update post_op_attr c_i. kernelOpsAttr is not a pointer, so adding
        // two offsets at the same time is safe.
        lea(regTmp2, ptr[regTmp2 + MR]);
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_i)],
            regTmp2);

        // decrement m_iter
        sub(regMiter, 1);

        jne(".BLOOP6x64I", T_NEAR);
    }
    vzeroupper();
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

// Generate kernel for S8 operations
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMS8<KType>::generateKernel(utils::generatorParams& params)
{
    RETURN_IF_ERROR(utils::jitGeneratorUtils::checkValidGemmParams(params));

    MR          = params.MR;
    NR          = params.NR;
    useMask     = params.useMask;
    c_downscale = params.c_downscale;

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
    Xbyak::util::StackFrame stackFrame(this, 1, 13, 0);
    initializeStackFrame(stackFrame);
    initializeParameters(params.mLoop);
    RETURN_IF_ERROR(generateIrLoop(params));

    return dlp::jit::jitGeneratorError::success;
}
} // namespace amdzen::GEMMcodeGenerator

// Explicit template instantiations for classes
template class amdzen::GEMMcodeGenerator::jitGEMMS8<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
