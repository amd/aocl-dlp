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

#include "bf16_gemm_generator.hh"
#include "jit_register/jit_register.hh"

namespace amdzen::GEMMcodeGenerator {

using namespace Xbyak;

template<utils::kernelInstrType KType>
jitGEMMBF16<KType>::jitGEMMBF16(size_t maxSize)
    : Xbyak::CodeGenerator(maxSize, Xbyak::AutoGrow)
{
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMBF16<KType>::allocateReg()
{
    // For BF16: B registers load BF16 data, C registers accumulate F32
    // results

    // Calculate register allocation for BF16 VNNI
    // B registers: load BF16 data, based on it's packing. Also used for C loads
    // C registers: Accumulate over F32 precision
    bFullReg = (2 * NR) / nBF16ElemsPerReg;
    bMaskReg = (useMask ? 1 : 0);
    bReg     = bFullReg + bMaskReg;
    cReg     = MR * bReg;
    aReg     = numRegs - cReg - bReg;

    // Check if we have enough registers
    if ((aReg < 1) || ((c_downscale < DLP_F32) && (aReg < 2))) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Register index assignment
    cRegIdx = numRegs - cReg;
    bRegIdx = cRegIdx - bReg;
    aRegIdx = 0;

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMMBF16<KType>::initializeParameters(bool addIrLoop)
{
    if (addIrLoop) {
        // Move A and C pointers to stack for IR-loop access
        mov(regAPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);
        mov(regMiter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, mIter)]);
    } else {
        mov(regTmpAptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);
    }
    // post_op_c_i is reloaded from memory at each use site rather than
    // being held in a dedicated register, since UseRBPAsFramePointer
    // limits us to 12 temp registers.

    if (c_downscale < DLP_F32) {
        // Broadcast the left shift offset onto a ZMM register
        // Store to allocated stack space, then broadcast from memory
        // Using rsp(instead of stackPtr in order to use the local stack space)
        mov(dword[rsp + 0],
            0x10); // Store value 16 to local stack (safely allocated)
    }

    mov(regCPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, c)]);
    mov(regRsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsA)]);
    mov(regCsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, csA)]);
    mov(regRsB, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsB)]);
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsC)]);

    // Scale strides for BF16 (2 bytes) and F32 (4 bytes)
    lea(regRsA, ptr[regRsA * sizeof(int16_t)]); // BF16 stride
    lea(regCsA, ptr[regCsA * sizeof(int16_t)]); // BF16 stride
    lea(regRsB, ptr[regRsB * sizeof(int16_t)]); // BF16 stride
    lea(regRsC, ptr[regRsC * sizeof(float)]);   // F32 stride

    mov(regTmpCptr, regCPtr);

    if (useMask) {
        loadMask();
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMBF16<KType>::loadBValuesBF16()
{
    // Load BF16 values from B matrix
    for (iter_t i = 0; i < bReg; i++) {
        // Load 32 BF16 elements (64 bytes) into ZMM register
        vmovdqu16(RegType(bRegIdx + i), ptr[regBptr + i * RegBytes]);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMBF16<KType>::BroadcastABF16withB(bool isRemainder)
{
    for (iter_t i = 0; i < MR; i++) {
        if (isRemainder) {
            vpbroadcastw(RegType(aRegIdx), ptr[regTmpAptr]);
        } else {
            vpbroadcastd(RegType(aRegIdx), ptr[regTmpAptr]);
        }
        add(regTmpAptr, regRsA);
        for (iter_t j = 0; j < bReg; j++) {
            vdpbf16ps(RegType(cRegIdx + i * bReg + j), RegType(bRegIdx + j),
                      RegType(aRegIdx));
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMBF16<KType>::kLoopCompute(bool isRemainder, int kUnroll)
{
    dlp::jit::jitGeneratorError err = dlp::jit::jitGeneratorError::error;

    // Unroll the kernel loop for BF16 VNNI
    // Save A pointer
    for (iter_t i = 0; i < kUnroll; i++) {
        mov(regTmp1, regTmpAptr);

        // Load B registers with BF16 data
        RETURN_IF_ERROR(loadBValuesBF16());
        add(regBptr, regRsB);

        // Perform BF16 VNNI computation
        RETURN_IF_ERROR(BroadcastABF16withB(isRemainder));

        // Advance A pointer for next K iteration
        lea(regTmpAptr, ptr[regTmp1 + regCsA]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMMBF16<KType>::initializeStackFrame(Xbyak::util::StackFrame& stackFrame)
{
    // StackFrame gives us 1 param register (p[0]) and 12 temp registers
    // (t[0]–t[11]).  With UseRBPAsFramePointer, rbp is reserved so 12
    // is the maximum.
    //
    // post_op_c_i is not held in a dedicated register; it is reloaded
    // from memory (gemmParams::kernelOpsAttr.post_op_c_i) at each use
    // site and written back after updates.
    stackPtr = stackFrame.p[0];

    regTmpAptr = stackFrame.t[0];
    regBptr    = stackFrame.t[1];
    regTmpCptr = stackFrame.t[2];
    regRsA     = stackFrame.t[3];
    regCsA     = stackFrame.t[4];
    regRsB     = stackFrame.t[5];
    regRsC     = stackFrame.t[6];
    regKIter   = stackFrame.t[7];
    regMiter   = stackFrame.t[8];
    regCPtr    = stackFrame.t[9];
    regAPtr    = stackFrame.t[10];
    regTmp1    = stackFrame.t[11];
}

template<utils::kernelInstrType KType>
void
jitGEMMBF16<KType>::regInit()
{
    // Initialize F32 accumulator registers to zero
    vxorps(RegType(cRegIdx), RegType(cRegIdx), RegType(cRegIdx));
    for (iter_t i = 1; i < cReg; i++) {
        vmovaps(RegType(cRegIdx + i), RegType(cRegIdx));
    }
}

template<utils::kernelInstrType KType>
void
jitGEMMBF16<KType>::moveCPtr()
{
    // Update C pointer for next row: cbuf += m * MR * rs_c
    int m_val       = MR;
    int power2scale = 1;
    while (m_val > 0) {
        if (m_val & 1) {
            // lea() only supports scale factors of 1, 2, 4, and 8.
            // For larger powers of 2, shift a temporary register and add it.
            if (power2scale <= 8) {
                lea(regCPtr, ptr[regCPtr + power2scale * regRsC]);
            } else {
                mov(regTmp1, regRsC);
                int shift_amount = 0;
                int temp_scale   = power2scale;
                while (temp_scale > 1) {
                    shift_amount++;
                    temp_scale >>= 1;
                }
                shl(regTmp1, shift_amount);
                add(regCPtr, regTmp1);
            }
        }
        m_val >>= 1;
        power2scale <<= 1;
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMBF16<KType>::scaleAlpha()
{
    int alphaRegIdx = aRegIdx;
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, alpha)]);
    vbroadcastss(RegType(alphaRegIdx), ptr[regTmp1]);
    for (iter_t i = 0; i < cReg; i++) {
        vmulps(RegType(cRegIdx + i), RegType(cRegIdx + i),
               RegType(alphaRegIdx));
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMBF16<KType>::scaleBeta()
{
    int betaRegIdx = aRegIdx;
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, beta)]);
    vbroadcastss(RegType(betaRegIdx), ptr[regTmp1]);

    // NOTE: The Decision Engine will pass betaScalingType as generic for
    // k > KC even when beta = 0. Hence, broadcasting beta and checking if
    // it is actually zero during run-time. This conforms to the standard of
    // avoiding accesses to C when beta = 0.
    int scratchRegIdx = aRegIdx + 1;
    vxorps(RegType(scratchRegIdx), RegType(scratchRegIdx),
           RegType(scratchRegIdx));
    vucomiss(Xbyak::Xmm(betaRegIdx), Xbyak::Xmm(scratchRegIdx));
    je("BETAOP_END", T_NEAR);

    mov(regTmpCptr, regCPtr);
    if (c_downscale < DLP_F32) {
        // Check for is_first_k
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_first_k)]);
        test(regTmp1, regTmp1);
        je("BETAOP", T_NEAR);

        mov(regTmpCptr,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, buf_downscale)]);

        // NULL check
        cmp(regTmpCptr, 0);
        je("BETAOP", T_NEAR);

        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
        lea(regTmp1, ptr[regTmp1 * sizeof(int16_t)]);

        add(regTmpCptr, regTmp1);

        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, rs_c_downscale)]);
        lea(regTmp1, ptr[regTmp1 * sizeof(int16_t)]); // BF16 stride

        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
        imul(regKIter, regTmp1);
        add(regTmpCptr, regKIter);

        vpbroadcastd(Xbyak::Zmm(aRegIdx + 1),
                     ptr[rsp + 0]); // Broadcast from memory

        for (iter_t i = 0; i < MR; i++) {
            for (iter_t j = 0; j < bFullReg; j++) {
                vmovdqu16(Xbyak::Ymm(bRegIdx + j), ptr[regTmpCptr + j * 32]);
                vpmovsxwd(Xbyak::Zmm(bRegIdx + j), Xbyak::Ymm(bRegIdx + j));
                vpsllvd(Xbyak::Zmm(bRegIdx + j), Xbyak::Zmm(bRegIdx + j),
                        Xbyak::Zmm(aRegIdx + 1));
                // vcvtdq2ps(Xbyak::Zmm(bRegIdx + j), Xbyak::Zmm(bRegIdx + j));
                vfmadd231ps(Xbyak::Zmm(cRegIdx + i * bReg + j),
                            Xbyak::Zmm(betaRegIdx), Xbyak::Zmm(bRegIdx + j));
            }
            if (bMaskReg > 0) {
                // Use zero-masking (T_z) to zero unmasked elements
                vmovdqu16(Xbyak::Ymm(bRegIdx + bFullReg) | mask_regs[0] | T_z,
                          ptr[regTmpCptr + bFullReg * halfRegBytes]);
                vpmovsxwd(Xbyak::Zmm(bRegIdx + bFullReg),
                          Xbyak::Ymm(bRegIdx + bFullReg));
                vpsllvd(Xbyak::Zmm(bRegIdx + bFullReg),
                        Xbyak::Zmm(bRegIdx + bFullReg),
                        Xbyak::Zmm(aRegIdx + 1));
                vfmadd231ps(Xbyak::Zmm(cRegIdx + i * bReg + bFullReg),
                            Xbyak::Zmm(betaRegIdx),
                            Xbyak::Zmm(bRegIdx + bFullReg));
            }
            add(regTmpCptr, regTmp1);
        }

        jmp("BETAOP_END", T_NEAR);
        L("BETAOP");
    }
    for (iter_t i = 0; i < MR; i++) {
        for (iter_t j = 0; j < bFullReg; j++) {
            vmovups(RegType(bRegIdx + j), ptr[regTmpCptr + j * RegBytes]);
            vfmadd231ps(RegType(cRegIdx + i * bReg + j), RegType(betaRegIdx),
                        RegType(bRegIdx + j));
        }
        if (bMaskReg > 0) {
            // Use zero-masking (T_z) to zero unmasked elements
            vmovups(RegType(bRegIdx + bFullReg) | mask_regs[0] | T_z,
                    ptr[regTmpCptr + bFullReg * RegBytes]);
            vfmadd231ps(RegType(cRegIdx + i * bReg + bFullReg),
                        RegType(betaRegIdx), RegType(bRegIdx + bFullReg));
        }
        add(regTmpCptr, regRsC);
    }
    L("BETAOP_END");
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMBF16<KType>::storeResult()
{
    // Scope-local labels: the GLU path emits storeResult() twice per kernel
    // (pre-fold raw-C store + non-last-k full-width store), so global labels
    // would collide with "label is redefined".
    inLocalLabel();
    mov(regTmpCptr, regCPtr);
    if (c_downscale < DLP_F32) {
        // Check for is_last_k
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
        test(regTmp1, regTmp1);
        je(".STOREOP", T_NEAR);

        mov(regTmpCptr,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, buf_downscale)]);

        // NULL check
        cmp(regTmpCptr, 0);
        je(".STOREOP", T_NEAR);

        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
        lea(regTmp1, ptr[regTmp1 * sizeof(int16_t)]);

        add(regTmpCptr, regTmp1);

        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, rs_c_downscale)]);
        lea(regTmp1, ptr[regTmp1 * sizeof(int16_t)]); // BF16 stride

        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
        imul(regKIter, regTmp1);
        add(regTmpCptr, regKIter);

        for (iter_t i = 0; i < MR; i++) {
            for (iter_t j = 0; j < bFullReg; j++) {
                vcvtneps2bf16(Xbyak::Ymm(bRegIdx + j),
                              Xbyak::Zmm(cRegIdx + i * bReg + j));
                vmovdqu16(ptr[regTmpCptr + j * halfRegBytes],
                          Xbyak::Ymm(bRegIdx + j));
            }
            if (bMaskReg > 0) {
                vcvtneps2bf16(Xbyak::Ymm(bRegIdx + bFullReg),
                              Xbyak::Zmm(cRegIdx + i * bReg + bFullReg));
                vmovdqu16(ptr[regTmpCptr + bFullReg * halfRegBytes]
                              | mask_regs[0],
                          Xbyak::Ymm(bRegIdx + bFullReg));
            }
            add(regTmpCptr, regTmp1);
        }

        jmp(".STOREOP_END", T_NEAR);
        L(".STOREOP");
    }
    for (iter_t i = 0; i < MR; i++) {
        for (iter_t j = 0; j < bFullReg; j++) {
            // Regular store
            vmovups(ptr[regTmpCptr + j * RegBytes],
                    RegType(cRegIdx + i * bReg + j));
        }
        if (bMaskReg > 0) {
            vmovups(ptr[regTmpCptr + bFullReg * RegBytes] | mask_regs[0],
                    RegType(cRegIdx + i * bReg + bFullReg));
        }
        add(regTmpCptr, regRsC);
    }
    L(".STOREOP_END");
    outLocalLabel();
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMBF16<KType>::emitHalfWidthResult(bool colMajor)
{
    // Halve the 2I tile to I outputs into the caller's D buffer (buf_d) at this
    // tile's disjoint home; raw 2I already went to C, only on is_last_k. Always
    // a row-major store; the axis differs by output layout:
    //   colMajor==false (row-major): MR rows, low I lanes; fringe mask_regs[1].
    //   colMajor==true  (col-major): MR/2 rows, full NR lanes; fringe
    //   mask_regs[0].
    // base = buf_d + og_col*elt + og_row*(ld_d*elt), storeTile advances
    // ld_d*elt per row (row-major: ld_d=I, og_col=c_j/2, og_row=c_i; col-major:
    // ld_d=m, og_col=c_j, og_row=c_i/2).
    const int    f32HalfBytes  = halfRegBytes;     // 8 f32  = 32 B (low Ymm)
    const int    bf16HalfBytes = halfRegBytes / 2; // 8 bf16 = 16 B (low Xmm)
    const int    eltBytes      = (c_downscale < DLP_F32) ? (int)sizeof(int16_t)
                                                         : (int)sizeof(float);
    const iter_t rows          = colMajor ? (MR / 2) : MR;

    // Row-major halves the lanes: derive the half-width fringe mask
    // (mask_regs[1]) as (1 << (popcnt(mask_regs[0])/2)) - 1. Column-major keeps
    // full lanes and reuses mask_regs[0].
    if (!colMajor && bMaskReg > 0) {
        kmovw(regTmp1.cvt32(), mask_regs[0]);
        popcnt(regKIter.cvt32(), regTmp1.cvt32());
        shr(regKIter.cvt32(), 1);
        mov(regTmp1.cvt32(), 0xFFFF);
        bzhi(regTmp1.cvt32(), regTmp1.cvt32(), regKIter.cvt32());
        kmovw(mask_regs[1], regTmp1.cvt32());
    }
    const Xbyak::Opmask& fringeMask = colMajor ? mask_regs[0] : mask_regs[1];

    // Emit `rows` stores from regTmpCptr, advancing rowStrideBytes per row.
    // Column-major writes the full register; row-major writes the low half.
    auto storeTile = [&](const Xbyak::Reg64& rowStrideBytes) {
        for (iter_t i = 0; i < rows; i++) {
            for (iter_t j = 0; j < bFullReg; j++) {
                if (c_downscale < DLP_F32) {
                    if (colMajor) {
                        vcvtneps2bf16(Xbyak::Ymm(bRegIdx + j),
                                      Xbyak::Zmm(cRegIdx + i * bReg + j));
                        vmovdqu16(ptr[regTmpCptr + j * halfRegBytes],
                                  Xbyak::Ymm(bRegIdx + j));
                    } else {
                        vcvtneps2bf16(Xbyak::Xmm(bRegIdx + j),
                                      Xbyak::Ymm(cRegIdx + i * bReg + j));
                        vmovdqu16(ptr[regTmpCptr + j * bf16HalfBytes],
                                  Xbyak::Xmm(bRegIdx + j));
                    }
                } else {
                    if (colMajor)
                        vmovups(ptr[regTmpCptr + j * RegBytes],
                                Xbyak::Zmm(cRegIdx + i * bReg + j));
                    else
                        vmovups(ptr[regTmpCptr + j * f32HalfBytes],
                                Xbyak::Ymm(cRegIdx + i * bReg + j));
                }
            }
            if (bMaskReg > 0) {
                if (c_downscale < DLP_F32) {
                    if (colMajor) {
                        vcvtneps2bf16(
                            Xbyak::Ymm(bRegIdx + bFullReg),
                            Xbyak::Zmm(cRegIdx + i * bReg + bFullReg));
                        vmovdqu16(ptr[regTmpCptr + bFullReg * halfRegBytes]
                                      | fringeMask,
                                  Xbyak::Ymm(bRegIdx + bFullReg));
                    } else {
                        vcvtneps2bf16(
                            Xbyak::Xmm(bRegIdx + bFullReg),
                            Xbyak::Ymm(cRegIdx + i * bReg + bFullReg));
                        vmovdqu16(ptr[regTmpCptr + bFullReg * bf16HalfBytes]
                                      | fringeMask,
                                  Xbyak::Xmm(bRegIdx + bFullReg));
                    }
                } else {
                    if (colMajor)
                        vmovups(ptr[regTmpCptr + bFullReg * RegBytes]
                                    | fringeMask,
                                Xbyak::Zmm(cRegIdx + i * bReg + bFullReg));
                    else
                        vmovups(ptr[regTmpCptr + bFullReg * f32HalfBytes]
                                    | fringeMask,
                                Xbyak::Ymm(cRegIdx + i * bReg + bFullReg));
                }
            }
            add(regTmpCptr, rowStrideBytes);
        }
    };

    // base = buf_d + og_col*elt + og_row*(ld_d*elt). Row-major halves the
    // column (c_j/2), column-major halves the row (c_i/2).
    mov(regTmpCptr,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, buf_d)]);

    // og_col = post_op_c_j (halved for row-major); base += og_col * elt.
    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
    if (!colMajor)
        shr(regTmp1, 1);
    lea(regTmp1, ptr[regTmp1 * eltBytes]);
    add(regTmpCptr, regTmp1);

    mov(regKIter,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, ld_d)]);
    lea(regKIter, ptr[regKIter * eltBytes]); // row stride bytes

    // og_row = post_op_c_i (halved for column-major); base += og_row * stride.
    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
    if (colMajor)
        shr(regTmp1, 1);
    imul(regTmp1, regKIter);
    add(regTmpCptr, regTmp1);
    storeTile(regKIter);
    return dlp::jit::jitGeneratorError::success;
}

// Row-major GLU half-width store: halve along the lanes (MR rows, low I lanes).
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMBF16<KType>::storeHalfWidthResult()
{
    return emitHalfWidthResult(/*colMajor=*/false);
}

// Column-major GLU half-width store: halve along M (MR/2 full-width rows).
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMBF16<KType>::storeHalfWidthResultAlongM()
{
    return emitHalfWidthResult(/*colMajor=*/true);
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMBF16<KType>::prefetchC()
{
    mov(regTmpCptr, regCPtr);
    if ((PREFETCH_C_DIST > 0)) {
        for (iter_t i = 0; i < MR; i++) {
            for (iter_t j = 0; j < bFullReg; j++) {
                prefetcht0(ptr[regTmpCptr + j * RegBytes]);
            }
            add(regTmpCptr, regRsC);
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMBF16<KType>::generateIrLoop(utils::generatorParams& params)
{
    inLocalLabel();

    // Calculate and load pointers
    if (params.mLoop) {
        L(".BLOOP6X64I");
        mov(regTmpAptr, regAPtr);
    }
    mov(regBptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, b)]);

    // Zero out F32 accumulators
    regInit();

    // Generate K-loop
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kIterBP)]);
    test(regKIter, regKIter);
    je(".BCONSIDKITERAP", T_NEAR);

    // Kernel unroll loop
    L(".BLOOPKITERBP");
    RETURN_IF_ERROR(kLoopCompute(false, 1));
    // B prefetch
    sub(regKIter, 1);
    jne(".BLOOPKITERBP", T_NEAR);

    if (params.betaScalingType != dlp::kernel_frame::scalingType::zero) {
        RETURN_IF_ERROR(prefetchC());
    }

    L(".BCONSIDKITERAP");
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kIterAP)]);
    test(regKIter, regKIter);
    je(".BCONSIDKLEFTREM", T_NEAR);

    L(".BLOOPKITERAP");
    RETURN_IF_ERROR(kLoopCompute(false, 1));
    sub(regKIter, 1);
    jne(".BLOOPKITERAP", T_NEAR);

    L(".BCONSIDKLEFTREM");
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeft)]);
    test(regKIter, regKIter);
    je(".BPOSTACCUM", T_NEAR);

    RETURN_IF_ERROR(kLoopCompute(true, 1));
    // No need to decrement regKIter as it could only be 1 or 0
    // This is due to the BF16 packing factor being 2

    L(".BPOSTACCUM");

    if (params.alphaScalingType != dlp::kernel_frame::scalingType::one) {
        // alpha scaling
        RETURN_IF_ERROR(scaleAlpha());
    }

    // To-Do: add support for beta scaling if beta is 1 using vaddps
    if (params.betaScalingType != dlp::kernel_frame::scalingType::zero) {
        // beta scaling
        RETURN_IF_ERROR(scaleBeta());
    }

    // check if is_last_k is set
    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
    test(regTmp1, regTmp1);
    je(label_store_result, T_NEAR);

    // Pre-fold raw-C store (terminal GLU): on the last K-block store the raw
    // pre-GLU 2I accumulators to C *before* the fold overwrites them; the fold
    // then writes the compacted result to D. storeResult() only reads the regs.
    if (params.storeHalfWidthResults) {
        RETURN_IF_ERROR(storeResult());
    }

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

        VecPoolType vecPool;
        vecPool.setAccumulators(cRegIdx, cReg);
        RETURN_IF_ERROR(vecPool.init(this, Traits::regBytes));

        MaskPoolType maskPool;
        maskPool.addPreserve(utils::MASK_START_IDX, params.useMask ? 1 : 0);
        RETURN_IF_ERROR(maskPool.init(this, utils::maskSaveWidth<KType>(),
                                      Traits::reservedMaskBits));

        int maskOffset =
            params.useMask ? static_cast<int>(
                                 offsetof(dlp::kernels::gemmParams, maskF32[0]))
                           : -1;

        RETURN_IF_ERROR((kernelOpsHandlerPtr->generateKernelOps(
            params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemm, params.MR,
            params.NR, params.useMask, params.numMaskRegs, cRegIdx, cReg,
            vecPool, maskPool, maskOffset)));
    }

    // store C
    L(label_store_result);
    if (params.storeHalfWidthResults) {
        // Terminal shape-changing post-op (e.g. GLU): on the last K-block store
        // the folded result half-width; earlier blocks store raw 2I partials so
        // the next block can accumulate. Branch at runtime on is_last_k.
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
        test(regTmp1, regTmp1);
        je(".HW_FULL", T_NEAR);
        // Column-major output (realized by an upstream m/n swap) puts gate/up
        // pairs across accumulator ROWS, so it halves along M instead of lanes.
        // Pick the axis from the shape-changing op's storage format.
        bool colMajor = false;
        for (const auto& kop : params.kernelOps) {
            if (dlp::kernel_frame::isShapeChangingOp(kop.type)) {
                colMajor = (kop.cMatFormat
                            == dlp::kernel_frame::storageFormat::colMajor);
            }
        }
        if (colMajor) {
            RETURN_IF_ERROR(storeHalfWidthResultAlongM());
        } else {
            RETURN_IF_ERROR(storeHalfWidthResult());
        }
        jmp(".HW_DONE", T_NEAR);
        L(".HW_FULL");
        RETURN_IF_ERROR(storeResult());
        L(".HW_DONE");
    } else {
        RETURN_IF_ERROR(storeResult());
    }

    if (params.mLoop) {
        // Update A pointer
        mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, psA)]);
        lea(regTmp1, ptr[regTmp1 * sizeof(int16_t)]); // BF16 size
        imul(regTmp1, regTmp1, MR);
        lea(regAPtr, ptr[regAPtr + regTmp1]);

        // Update post_op_c_i for the next m-iteration.
        // Reload from memory, increment, and write back (no dedicated
        // register — freed for UseRBPAsFramePointer).
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
        add(regTmp1, MR);
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_i)],
            regTmp1);

        moveCPtr();

        // Decrement m_iter
        sub(regMiter, 1);
        jne(".BLOOP6X64I", T_NEAR);
    }

    vzeroupper();
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMBF16<KType>::loadMask()
{
    for (iter_t i = 0; i < utils::NUM_USABLE_MASKS; i++) {
        mask_regs[i] = Xbyak::Opmask(utils::MASK_START_IDX + i);
    }

    kmovw(mask_regs[0],
          ptr[stackPtr + offsetof(dlp::kernels::gemmParams, maskF32[0])]);

    return dlp::jit::jitGeneratorError::success;
}

// Generate kernel for BF16 operations
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMBF16<KType>::generateKernel(utils::generatorParams& params)
{
    RETURN_IF_ERROR(utils::jitGeneratorUtils::checkValidGemmParams(params));

    MR              = params.MR;
    NR              = params.NR;
    K_UNROLL        = params.K_UNROLL;
    PREFETCH_C_DIST = params.PREFETCH_C_DIST;
    useMask         = params.useMask;
    c_downscale     = params.c_downscale;

    RETURN_IF_ERROR(allocateReg());

    // Initialize stack frame and parameters
    // Allocate 16 bytes of local stack space for temporary constants
    Xbyak::util::StackFrame stackFrame(
        this, 1, 12 | Xbyak::util::UseRBPAsFramePointer, 16);
    initializeStackFrame(stackFrame);

    // Preserve callee-saved xmm6-15 across the kernel call on Windows x64
    // (no-op on Linux/SysV). See utils::winAbiVectorGuard.
    utils::winAbiVectorGuard winAbiGuard(this);

    initializeParameters(params.mLoop);

    RETURN_IF_ERROR(generateIrLoop(params));

    return dlp::jit::jitGeneratorError::success;
}

} // namespace amdzen::GEMMcodeGenerator

// Explicit template instantiations for BF16 VNNI-capable instruction sets
template class amdzen::GEMMcodeGenerator::jitGEMMBF16<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
