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

#include "s8_gemm_quant_generator.hh"

#include "classic/aocl_gemm_metadata.h" // DLP_F32 / DLP_BF16
#include "jit/jit_generator_base.hh"
#include "jit_generator_utils.hh"
#include "kernel_frame/kernel_frame_base.hh"

namespace amdzen::GEMMcodeGenerator {

using namespace Xbyak;

// Byte offset of a dlp_gemm_grp_post_op_attr field within gemmParams.
#define GRP_ATTR_OFF(field)                                                    \
    (offsetof(dlp::kernels::gemmParams, grpKernelOpsAttr)                      \
     + offsetof(dlp_gemm_grp_post_op_attr, field))

template<utils::kernelInstrType KType>
jitGEMMQuant<KType>::jitGEMMQuant(size_t maxSize)
    : Xbyak::CodeGenerator(maxSize, Xbyak::AutoGrow)
{
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMQuant<KType>::allocateReg(utils::quantGeneratorParams& params)
{
    // Using int32 since accumulation is done in int32
    int nElemsPerReg = RegBytes / sizeof(int32_t);

    // Number of full vector registers needed for B
    bFullReg = (NR) / nElemsPerReg;

    // Number of mask registers needed for B (1 if needed, else 0)
    bMaskReg = (useMask ? 1 : 0);

    // Total number of registers needed for B
    bReg = bFullReg + bMaskReg;

    // Total number of registers needed for C
    cReg = MR * bReg;

    // Reserve one register for converting int8 to uint8 for symmetric
    // quantization path.
    if ((params.aQuant.zeroPoint.storeDt
         == dlp::kernel_frame::DataType::invalid)
        || (params.bQuant.zeroPoint.storeDt
            == dlp::kernel_frame::DataType::invalid)) {
        vec128Reg = 1;
    } else {
        vec128Reg = 0;
    }

    // For 6x64 primary kernel we are essentially left with 3 registers that
    // can be used for aPool.
    const int aPoolMin = 3;

    // True for small NR (16/32), false for NR 48/64 (which keep the stack
    // bank).
    fBankInRegs = (2 * cReg + vec128Reg + bReg + aPoolMin) <= numRegs;

    cRegIdx = numRegs - cReg; // Starting index for C (int32 accumulators)
    if (fBankInRegs) {
        // Layout (top-down): C | fReg (f32 bank) | B | vec128 | A pool.
        fRegIdx = cRegIdx - cReg;
        bRegIdx = fRegIdx - bReg;
    } else {
        // stack-resident F32 bank ([rsp + 0 .. cReg*RegBytes]).
        fRegIdx = 0;
        bRegIdx = cRegIdx - bReg;
    }

    vec128RegIdx = vec128Reg == 1 ? bRegIdx - 1 : 0;
    aRegIdx      = 0;
    aReg = bRegIdx - vec128RegIdx; // free A pool = zmm0 .. vec128RegIdx-1

    // Validate register count (need a non-empty A scratch pool below vec128).
    if (aReg < 1) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMMQuant<KType>::initializeStackFrame(Xbyak::util::StackFrame& stackFrame)
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

    // Alias the three running group pointers onto temps that are unused
    // across .QGROUPLOOP. regTmpCptr is the freed group index (gAbs); regRsC
    // and regCPtr are unused inside the group loop and are re-materialized /
    // restored after it.
    regBsumPtr = regTmpCptr; // freed gAbs
    regBsclPtr = regRsC;     // rs_c re-assigned after the loop
    regAsclPtr = regCPtr;    // C base saved/restored around the loop
}

template<utils::kernelInstrType KType>
void
jitGEMMQuant<KType>::initializeParameters(bool addIrLoop)
{
    if (addIrLoop) {
        mov(regAPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);
        mov(regMiter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, mIter)]);
    } else {
        mov(regTmpAptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);
    }

    mov(regCPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, c)]);
    mov(regRsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsA)]);
    mov(regCsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, csA)]);
    mov(regRsB, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsB)]);
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsC)]);

    // rs_c is in elements; convert to a byte stride (F32 output).
    lea(regRsC, ptr[regRsC * sizeof(float)]);

    // Track the absolute output row base for a-scale indexing. grp_post_op_i is
    // the tile's row origin (ic); the mLoop advances this by MR per iteration.
    mov(regTmp2, ptr[stackPtr + GRP_ATTR_OFF(grp_post_op_i)]);

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
jitGEMMQuant<KType>::initializeRegisters()
{
    vxorps(RegType(cRegIdx), RegType(cRegIdx), RegType(cRegIdx));
    for (iter_t i = 1; i < cReg; ++i) {
        vmovaps(RegType(cRegIdx + i), RegType(cRegIdx));
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMQuant<KType>::loadBValues()
{
    for (iter_t i = 0; i < bFullReg; ++i) {
        vmovdqu32(RegType(bRegIdx + i), ptr[regBptr + i * RegBytes]);
    }

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
jitGEMMQuant<KType>::BroadcastAVNNIB(bool isVNNIrem)
{
    // Rotate the A broadcast over a small register pool (zmm0 .. aPool-1)
    // instead of funnelling all MR rows through aRegIdx, so consecutive rows
    // form independent broadcast -> +128 -> vpdpbusd chains.
    const int aPool = (MR < vec128RegIdx) ? MR : vec128RegIdx;
    for (iter_t i = 0; i < MR; ++i) {
        int ar = aRegIdx + (i % aPool);
        if (isVNNIrem) {
            vmovdqu8(Xbyak::Ymm(ar) | mask_regs[0] | T_z, ptr[regTmpAptr]);
            vpbroadcastd(RegType(ar), Xbyak::Xmm(ar));
        } else {
            vpbroadcastd(RegType(ar), ptr[regTmpAptr]);
        }
        vpaddb(RegType(ar), RegType(ar), RegType(vec128RegIdx));

        add(regTmpAptr, regRsA);

        for (iter_t j = 0; j < bReg; ++j) {
            vpdpbusd(RegType(cRegIdx + i * bReg + j), RegType(ar),
                     RegType(bRegIdx + j));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMQuant<KType>::kLoop(int unroll, bool isVNNIrem)
{
    for (iter_t u = 0; u < unroll; ++u) {
        mov(regTmp1, regTmpAptr);

        RETURN_IF_ERROR(loadBValues());
        add(regBptr, regRsB);

        RETURN_IF_ERROR(BroadcastAVNNIB(isVNNIrem));
        lea(regTmpAptr, ptr[regTmp1 + regCsA]);
    }

    return dlp::jit::jitGeneratorError::success;
}

// Load the group's B column sums (b_col_sum_vec + b_sum_offset) into the B
// registers. the fast path uses group 0 (b_sum offset base),
// which matches num_groups == 1.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMQuant<KType>::loadBSumValues()
{
    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, b_col_sum_vec)]);
    // Group nLeft's column sums: b_col_sum_vec + nLeft*sum_ld + b_sum_offset.
    mov(regTmp3, ptr[stackPtr + GRP_ATTR_OFF(grp_post_op_sum_ld)]);
    imul(regTmp3, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nLeft)]);
    add(regTmp3,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, b_sum_offset)]);

    lea(regTmp1, ptr[regTmp1 + (regTmp3 * sizeof(int32_t))]);

    for (iter_t i = 0; i < bFullReg; ++i) {
        vmovdqu8(RegType(bRegIdx + i), ptr[regTmp1 + i * RegBytes]);
    }

    if (bMaskReg > 0) {
        vmovdqu8(RegType(bRegIdx + bFullReg),
                 ptr[regTmp1 + bFullReg * RegBytes]);
    }

    return dlp::jit::jitGeneratorError::success;
}

// Subtract the per-group B column sums to compensate for the +128 added to A.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMQuant<KType>::conversionCompensation()
{
    RETURN_IF_ERROR(loadBSumValues());

    for (iter_t j = 0; j < bReg; ++j) {
        for (iter_t i = 0; i < MR; ++i) {
            vpsubd(RegType(cRegIdx + i * bReg + j),
                   RegType(cRegIdx + i * bReg + j), RegType(bRegIdx + j));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

// Fast path (num_groups == 1)
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMQuant<KType>::applyGroupScalesFastPath()
{
    // B scale for group nLeft: b_scale_factor + nLeft*ldb + grp_post_op_j.
    // A K-collapsing B (PER_CHANNEL) has one scale per column, so the group
    // term drops out and only grp_post_op_j selects the column base.
    mov(regTmp1, ptr[stackPtr + GRP_ATTR_OFF(b_scale_factor)]);
    if (bPerGroupK) {
        mov(regTmp3, ptr[stackPtr + GRP_ATTR_OFF(grp_post_op_ldb)]);
        imul(regTmp3,
             ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nLeft)]);
        add(regTmp3, ptr[stackPtr + GRP_ATTR_OFF(grp_post_op_j)]);
    } else {
        mov(regTmp3, ptr[stackPtr + GRP_ATTR_OFF(grp_post_op_j)]);
    }
    lea(regTmp1, ptr[regTmp1 + regTmp3 * bScaleElemBytes()]);

    RETURN_IF_ERROR(loadBScales(regTmp1));

    // A scale base for group nLeft, row = regTmp2 (absolute row base):
    // a_scale_factor + (regTmp2 * grp_post_op_lda + nLeft). A K-collapsing A
    // (PER_TOKEN) has one scale per row, and the frame sets lda to 1 for it, so
    // the row term alone addresses the scale and the group term drops out.
    mov(regKIter, ptr[stackPtr + GRP_ATTR_OFF(a_scale_factor)]);
    mov(regBptr, ptr[stackPtr + GRP_ATTR_OFF(grp_post_op_lda)]);
    mov(regTmpAptr, regTmp2);
    imul(regTmpAptr, regBptr); // row_base * lda
    if (aPerGroupK) {
        add(regTmpAptr,
            ptr[stackPtr
                + offsetof(dlp::kernels::gemmParams, nLeft)]); // + nLeft
    }
    lea(regKIter, ptr[regKIter + regTmpAptr * aScaleElemBytes()]);

    const int aPool = (MR < vec128RegIdx) ? MR : vec128RegIdx;
    for (iter_t i = 0; i < MR; ++i) {
        int ar = aRegIdx + (i % aPool);
        RETURN_IF_ERROR(broadcastAScale(ar, regKIter)); // a_scale[row i, g0]

        for (iter_t j = 0; j < bReg; ++j) {
            const int c = cRegIdx + i * bReg + j;
            vcvtdq2ps(RegType(c), RegType(c));
            vmulps(RegType(c), RegType(c), RegType(bRegIdx + j));
            vmulps(RegType(c), RegType(c), RegType(ar));
        }

        // Advance a_scale pointer by lda elements for the next row.
        lea(regKIter, ptr[regKIter + regBptr * aScaleElemBytes()]);
    }

    return dlp::jit::jitGeneratorError::success;
}

// Scale the F32 accumulators by alpha (converted from int32). No-op when
// alpha == 1 (alphaScalingType == one).
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMQuant<KType>::scaleAlphaF32()
{
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, alpha)]);
    vpbroadcastd(RegType(aRegIdx), ptr[regTmp1]);
    vcvtdq2ps(RegType(aRegIdx), RegType(aRegIdx));

    for (iter_t i = 0; i < cReg; ++i) {
        vmulps(RegType(cRegIdx + i), RegType(cRegIdx + i), RegType(aRegIdx));
    }

    return dlp::jit::jitGeneratorError::success;
}

// Retarget regTmpCptr from the F32 accumulator buffer to the downscaled output
// buffer: &buf_downscale[row_base][post_op_c_j]. The downscale row stride, in
// bytes, is left in regTmp1 for the caller's MR walk, so callers must treat
// regTmp1 as live for the whole walk.
template<utils::kernelInstrType KType>
void
jitGEMMQuant<KType>::updateCBufferPointers()
{
    mov(regTmpCptr,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, buf_downscale)]);

    // + post_op_c_j columns.
    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);

    if (c_downscale == DLP_BF16) {
        lea(regTmp1, ptr[regTmp1 * 2]);
    } else if (c_downscale == DLP_F32) {
        lea(regTmp1, ptr[regTmp1 * 4]);
    }

    add(regTmpCptr, regTmp1);

    // Row stride in bytes, handed back to the caller in regTmp1.
    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, rs_c_downscale)]);

    if (c_downscale == DLP_BF16) {
        lea(regTmp1, ptr[regTmp1 * 2]);
    } else if (c_downscale == DLP_F32) {
        lea(regTmp1, ptr[regTmp1 * 4]);
    }

    // + row_base * rs_c_downscale. regTmp2 is the absolute output row base, so
    // this stays correct for mc0 > MR. The stride is already byte-scaled here,
    // hence no second width factor.
    mov(regKIter, regTmp2);
    imul(regKIter, regTmp1);
    add(regTmpCptr, regKIter);
}

// acc += beta * C, reading the existing C tile. The tile is F32 in the
// accumulator buffer, except on the first K panel of a DLP_BF16 output, where
// it is BF16 in buf_downscale. No-op when beta == 0 (betaScalingType == zero).
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMQuant<KType>::scaleBeta()
{
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, beta)]);
    vpbroadcastd(RegType(aRegIdx), ptr[regTmp1]);
    vcvtdq2ps(RegType(aRegIdx), RegType(aRegIdx)); // aRegIdx = beta (f32)

    mov(regTmpCptr, regCPtr);

    if (c_downscale == DLP_BF16) {
        // Read the downscaled buffer only on the first K panel; later panels
        // accumulate against the F32 scratch the frame hands us instead.
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, buf_downscale)]);
        test(regTmp1, regTmp1);
        je(".QBETAOP", T_NEAR);

        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_first_k)]);
        test(regTmp1, regTmp1);
        je(".QBETAOP", T_NEAR);

        updateCBufferPointers(); // regTmpCptr = base, regTmp1 = row stride

        for (iter_t i = 0; i < MR; ++i) {
            for (iter_t j = 0; j < bFullReg; ++j) {
                // Widen bf16 to f32 in place, as in loadBScales.
                vmovdqu16(Xbyak::Ymm(bRegIdx + j),
                          ptr[regTmpCptr + j * (RegBytes / 2)]);
                vpmovsxwd(RegType(bRegIdx + j), Xbyak::Ymm(bRegIdx + j));
                vpslld(RegType(bRegIdx + j), RegType(bRegIdx + j), 16);
                vfmadd231ps(RegType(cRegIdx + i * bReg + j), RegType(aRegIdx),
                            RegType(bRegIdx + j));
            }
            if (bMaskReg > 0) {
                vmovdqu16(Xbyak::Ymm(bRegIdx + bFullReg) | mask_regs[1] | T_z,
                          ptr[regTmpCptr + bFullReg * (RegBytes / 2)]);
                vpmovsxwd(RegType(bRegIdx + bFullReg),
                          Xbyak::Ymm(bRegIdx + bFullReg));
                vpslld(RegType(bRegIdx + bFullReg), RegType(bRegIdx + bFullReg),
                       16);
                vfmadd231ps(RegType(cRegIdx + i * bReg + bFullReg),
                            RegType(aRegIdx), RegType(bRegIdx + bFullReg));
            }
            add(regTmpCptr, regTmp1);
        }

        jmp(".QBETAOP_END", T_NEAR);
        L(".QBETAOP");
    }

    for (iter_t i = 0; i < MR; ++i) {
        for (iter_t j = 0; j < bFullReg; ++j) {
            vmovups(RegType(bRegIdx + j), ptr[regTmpCptr + j * RegBytes]);
            vfmadd231ps(RegType(cRegIdx + i * bReg + j), RegType(aRegIdx),
                        RegType(bRegIdx + j));
        }
        if (bMaskReg > 0) {
            vmovups(RegType(bRegIdx + bFullReg) | mask_regs[1] | T_z,
                    ptr[regTmpCptr + bFullReg * RegBytes]);
            vfmadd231ps(RegType(cRegIdx + i * bReg + bFullReg),
                        RegType(aRegIdx), RegType(bRegIdx + bFullReg));
        }
        add(regTmpCptr, regRsC);
    }

    if (c_downscale == DLP_BF16) {
        L(".QBETAOP_END");
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMQuant<KType>::generatePostOps(utils::generatorParams& params)
{
    if (params.kernelOps.empty()) {
        return dlp::jit::jitGeneratorError::success;
    }

    Xbyak::Label skipPostOps;

    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
    test(regTmp1, regTmp1);
    je(skipPostOps, T_NEAR);

    using VecPoolType =
        utils::registerPool<typename Traits::RegType, Traits::numRegs>;
    using MaskPoolType =
        utils::registerPool<Xbyak::Opmask, Traits::numMaskRegs>;

    VecPoolType vecPool;
    vecPool.setAccumulators(cRegIdx, cReg);
    RETURN_IF_ERROR(vecPool.init(this, Traits::regBytes));

    int          maskCount = useMask ? 2 : 1;
    MaskPoolType maskPool;
    maskPool.addPreserve(utils::MASK_START_IDX, maskCount);
    RETURN_IF_ERROR(maskPool.init(this, utils::maskSaveWidth<KType>(),
                                  Traits::reservedMaskBits));

    int maskOffset =
        useMask ? static_cast<int>(offsetof(dlp::kernels::gemmParams, maskS32))
                : -1;

    gen::kernelOpsHandler<KType> handler(this);
    RETURN_IF_ERROR(handler.generateKernelOps(
        params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemm, params.MR,
        params.NR, params.useMask, params.numMaskRegs, cRegIdx, cReg, vecPool,
        maskPool, maskOffset));

    L(skipPostOps);

    return dlp::jit::jitGeneratorError::success;
}

// Store the F32 accumulators.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMQuant<KType>::storeResult()
{
    if (c_downscale != DLP_BF16 && c_downscale != DLP_F32) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    mov(regTmpCptr, regCPtr);

    if (c_downscale == DLP_BF16) {
        // Write the downscaled buffer only on the final K panel; earlier panels
        // leave their partial sums in the F32 scratch.
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, buf_downscale)]);
        test(regTmp1, regTmp1);
        je(".QSTOREOP", T_NEAR);

        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
        test(regTmp1, regTmp1);
        je(".QSTOREOP", T_NEAR);

        updateCBufferPointers(); // regTmpCptr = base, regTmp1 = row stride

        mov(regKIter, 0x00000001);
        vpbroadcastd(RegType(aRegIdx), regKIter.cvt32()); // lsb mask
        mov(regKIter, 0x00007FFF);
        vpbroadcastd(RegType(aRegIdx + 1), regKIter.cvt32()); // rounding bias

        for (iter_t i = 0; i < MR; ++i) {
            for (iter_t j = 0; j < bFullReg; ++j) {
                const int c = cRegIdx + i * bReg + j;
                // bf16 = (c + 0x7FFF + ((c >> 16) & 1)) >> 16.
                vpsrld(RegType(bRegIdx), RegType(c), 16);
                vpandd(RegType(bRegIdx), RegType(bRegIdx), RegType(aRegIdx));
                vpaddd(RegType(c), RegType(c), RegType(aRegIdx + 1));
                vpaddd(RegType(c), RegType(c), RegType(bRegIdx));
                vpsrld(RegType(c), RegType(c), 16);
                vpmovdw(Xbyak::Ymm(c), RegType(c));
                vmovdqu16(ptr[regTmpCptr + j * (RegBytes / 2)], Xbyak::Ymm(c));
            }
            if (bMaskReg > 0) {
                const int c = cRegIdx + i * bReg + bFullReg;
                vpsrld(RegType(bRegIdx), RegType(c), 16);
                vpandd(RegType(bRegIdx), RegType(bRegIdx), RegType(aRegIdx));
                vpaddd(RegType(c), RegType(c), RegType(aRegIdx + 1));
                vpaddd(RegType(c), RegType(c), RegType(bRegIdx));
                vpsrld(RegType(c), RegType(c), 16);
                vpmovdw(Xbyak::Ymm(c), RegType(c));
                vmovdqu16(ptr[regTmpCptr + bFullReg * (RegBytes / 2)]
                              | mask_regs[1],
                          Xbyak::Ymm(c));
            }
            add(regTmpCptr, regTmp1);
        }

        jmp(".QSTOREOP_END", T_NEAR);
        L(".QSTOREOP");
    }

    // F32 accumulator buffer
    for (iter_t i = 0; i < MR; ++i) {
        for (iter_t j = 0; j < bFullReg; ++j) {
            vmovups(ptr[regTmpCptr + j * RegBytes],
                    RegType(cRegIdx + i * bReg + j));
        }
        if (bMaskReg > 0) {
            vmovups(ptr[regTmpCptr + bFullReg * RegBytes] | mask_regs[1],
                    RegType(cRegIdx + i * bReg + bFullReg));
        }
        add(regTmpCptr, regRsC);
    }

    if (c_downscale == DLP_BF16) {
        L(".QSTOREOP_END");
    }

    return dlp::jit::jitGeneratorError::success;
}

// Zero the stack-resident F32 accumulator bank ([rsp + 0 .. cReg*RegBytes]).
template<utils::kernelInstrType KType>
void
jitGEMMQuant<KType>::zeroStackBank()
{
    vxorps(RegType(aRegIdx), RegType(aRegIdx), RegType(aRegIdx));
    for (iter_t i = 0; i < cReg; ++i) {
        vmovups(ptr[rsp + i * RegBytes], RegType(aRegIdx));
    }
}

// Load the stack F32 bank back into the cReg accumulator registers.
template<utils::kernelInstrType KType>
void
jitGEMMQuant<KType>::loadStackBankToRegs()
{
    for (iter_t i = 0; i < cReg; ++i) {
        vmovups(RegType(cRegIdx + i), ptr[rsp + i * RegBytes]);
    }
}

// Zero the register-resident F32 accumulator bank (fReg block).
template<utils::kernelInstrType KType>
void
jitGEMMQuant<KType>::zeroRegBank()
{
    vxorps(RegType(fRegIdx), RegType(fRegIdx), RegType(fRegIdx));
    for (iter_t i = 1; i < cReg; ++i) {
        vmovaps(RegType(fRegIdx + i), RegType(fRegIdx));
    }
}

// Fold the register-resident F32 bank into the cReg accumulators
// (register-register move; the total then feeds the alpha/beta/store rails).
template<utils::kernelInstrType KType>
void
jitGEMMQuant<KType>::foldRegBankToRegs()
{
    for (iter_t i = 0; i < cReg; ++i) {
        vmovaps(RegType(cRegIdx + i), RegType(fRegIdx + i));
    }
}

// Per-group +128 compensation: subtract b_col_sum[gAbs] (offset
// gAbs*grp_post_op_sum_ld + b_sum_offset) from the int32 accumulators.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMQuant<KType>::conversionCompensationGroup()
{
    // regBsumPtr is the running b_col_sum pointer for the current group.
    // It is initialized before .QGROUPLOOP and advanced by grp_post_op_sum_ld
    // each group, so no per-group base recomputation (mov/imul/add/lea) is
    // emitted here.
    for (iter_t j = 0; j < bReg; ++j) {
        vmovdqu32(RegType(bRegIdx + j), ptr[regBsumPtr + j * RegBytes]);
    }
    for (iter_t j = 0; j < bReg; ++j) {
        for (iter_t i = 0; i < MR; ++i) {
            vpsubd(RegType(cRegIdx + i * bReg + j),
                   RegType(cRegIdx + i * bReg + j), RegType(bRegIdx + j));
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

// Load a group's b_scale[gAbs, 0:NR] vector into the B registers as F32.
// `base` is the running b_scale pointer: regBsclPtr in the group loop, or the
// locally computed regTmp1 in the fast path.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMQuant<KType>::loadBScales(const Xbyak::Reg64& base)
{
    if (bScaleType == dlp::kernel_frame::DataType::f32) {
        for (iter_t j = 0; j < bFullReg; ++j) {
            vmovups(RegType(bRegIdx + j), ptr[base + j * RegBytes]);
        }
        if (bMaskReg > 0) {
            vmovups(RegType(bRegIdx + bFullReg) | mask_regs[1] | T_z,
                    ptr[base + bFullReg * RegBytes]);
        }
    } else if (bScaleType == dlp::kernel_frame::DataType::bf16) {
        for (iter_t j = 0; j < bFullReg; ++j) {
            vmovdqu16(Xbyak::Ymm(bRegIdx + j), ptr[base + j * (RegBytes / 2)]);
            vpmovsxwd(RegType(bRegIdx + j), Xbyak::Ymm(bRegIdx + j));
            vpslld(RegType(bRegIdx + j), RegType(bRegIdx + j), 16);
        }
        if (bMaskReg > 0) {
            vmovdqu16(Xbyak::Ymm(bRegIdx + bFullReg) | mask_regs[1] | T_z,
                      ptr[base + bFullReg * (RegBytes / 2)]);
            vpmovsxwd(RegType(bRegIdx + bFullReg),
                      Xbyak::Ymm(bRegIdx + bFullReg));
            vpslld(RegType(bRegIdx + bFullReg), RegType(bRegIdx + bFullReg),
                   16);
        }
    } else {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    return dlp::jit::jitGeneratorError::success;
}

// Broadcast one a_scale[row, gAbs] scalar into `ar` as F32. `base` is the
// per-row a_scale pointer: regTmp1 (a walking copy of regAsclPtr) in the group
// loop, or regKIter in the fast path.
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMQuant<KType>::broadcastAScale(int ar, const Xbyak::Reg64& base)
{
    if (aScaleType == dlp::kernel_frame::DataType::f32) {
        vbroadcastss(RegType(ar), ptr[base]);
    } else if (aScaleType == dlp::kernel_frame::DataType::bf16) {
        vpbroadcastw(RegType(ar), ptr[base]);
        vpmovsxwd(RegType(ar), Xbyak::Ymm(ar));
        vpslld(RegType(ar), RegType(ar), 16);
    } else {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    return dlp::jit::jitGeneratorError::success;
}

// Per-group dequant: acc_stack += cvt(int32) * b_scale[gAbs,j] *
// a_scale[row,gAbs]. Scales are F32 or BF16, fixed at generation time. Row base
// is tracked in regTmp2 (advanced by the mLoop).
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMQuant<KType>::dequantAccumToStackGroup()
{
    // regBsclPtr is the running b_scale pointer for the current group (advanced
    // by grp_post_op_ldb each group, or held fixed for a K-collapsing B).
    RETURN_IF_ERROR(loadBScales(regBsclPtr));

    // regAsclPtr is the running a_scale pointer for (row_base, group) (advanced
    // by one element each group, or held fixed for a K-collapsing A). The
    // per-row stride is lda either way -- the frame sets it to 1 for PER_TOKEN
    // -- so load lda into regTmp3 and walk a copy of regAsclPtr (regTmp1) down
    // MR rows.
    mov(regTmp3, ptr[stackPtr + GRP_ATTR_OFF(grp_post_op_lda)]);
    mov(regTmp1, regAsclPtr);

    const int aPool = (MR < vec128RegIdx) ? MR : vec128RegIdx;
    for (iter_t i = 0; i < MR; ++i) {
        int ar = aRegIdx + (i % aPool);
        RETURN_IF_ERROR(broadcastAScale(ar, regTmp1)); // a_scale[row i, gAbs]

        for (iter_t j = 0; j < bReg; ++j) {
            const int c = cRegIdx + i * bReg + j;
            const int t = i * bReg + j;
            vcvtdq2ps(RegType(c), RegType(c));

            vmulps(RegType(c), RegType(c), RegType(bRegIdx + j));
            vmulps(RegType(c), RegType(c), RegType(ar));

            if (fBankInRegs) {
                // register-resident bank -- accumulate this group into
                // fReg (no [rsp] traffic). Zeroed once before the group loop,
                // folded into cReg once after it.
                vaddps(RegType(fRegIdx + t), RegType(fRegIdx + t), RegType(c));
            } else {
                // stack bank: read-add-store [rsp].
                vaddps(RegType(c), RegType(c), ptr[rsp + t * RegBytes]);
                vmovups(ptr[rsp + t * RegBytes], RegType(c));
            }
        }

        // Advance a_scale pointer by lda elements for the next row.
        lea(regTmp1, ptr[regTmp1 + regTmp3 * aScaleElemBytes()]);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMMQuant<KType>::moveCPtr()
{
    int m_val       = MR;
    int power2scale = 1;
    while (m_val > 0) {
        if (m_val & 1) {
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
jitGEMMQuant<KType>::generateIrLoop(utils::quantGeneratorParams& qParams)
{
    utils::generatorParams& params = qParams.base;
    inLocalLabel();

    // Reserve the stack-resident F32 accumulator bank (multi-group path) plus a
    // small scratch area (one qword to preserve the mLoop counter across the
    // group loop). Addressed rsp-relative; rsp is stable inside the mLoop body
    // (no push/pop). Balanced by `add rsp` before the epilogue.
    // The F32 bank is stack-resident only when it does not fit in
    // registers (NR 48/64); for NR<=32 it lives in fReg, so only the 64B
    // scratch (saved mLoop counter) is reserved.
    const int bankBytes  = fBankInRegs ? 0 : cReg * RegBytes;
    const int scratchOff = bankBytes; // saved mLoop counter slot
    // Second 8B scratch slot for the saved C base (regCPtr is repurposed
    // as the running a_scale pointer across the group loop). Both slots live
    // inside the existing 64B scratch, so totalStack is unchanged.
    const int scratchOff2 = scratchOff + 8; // saved C base slot
    const int totalStack  = bankBytes + 64; // 64B scratch (keeps 64B align)
    sub(rsp, totalStack);

    if (params.mLoop) {
        L(".QBLOOPI");
        mov(regTmpAptr, regAPtr);
    }
    mov(regBptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, b)]);

    // Broadcast 128 for sym_quant to convert signed int8 A to uint8
    if ((qParams.aQuant.zeroPoint.storeDt
         == dlp::kernel_frame::DataType::invalid)
        || (qParams.bQuant.zeroPoint.storeDt
            == dlp::kernel_frame::DataType::invalid)) {
        mov(regTmp1, 128);
        vxorps(RegType(vec128RegIdx), RegType(vec128RegIdx),
               RegType(vec128RegIdx));
        vpbroadcastb(RegType(vec128RegIdx), regTmp1.cvt8());
    }

    // nIter (num_groups) == 1 ? fast path : group loop
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nIter)]);
    cmp(regTmp1, 1);
    jne(".QMULTIGROUP", T_NEAR);

    // Fast Path
    // Single group over the whole tile when group_size == 0 || group_size == k
    initializeRegisters();

    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kIterBP)]);
    test(regKIter, regKIter);
    je(".QCONSIDKLEFT", T_NEAR);

    L(".QLOOPKITER");
    RETURN_IF_ERROR(kLoop(params.K_UNROLL, false));
    sub(regKIter, 1);
    jne(".QLOOPKITER", T_NEAR);

    L(".QCONSIDKLEFT");
    if (params.K_UNROLL == 1) {
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeft)]);
        test(regKIter, regKIter);
        je(".QPOSTACCUM", T_NEAR);

        RETURN_IF_ERROR(kLoop(1, true));
    } else {
        L(".QCONSIDKLEFTITER");
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeftIter)]);
        test(regKIter, regKIter);
        je(".QCONSIDKLEFTREM", T_NEAR);

        L(".QLOOPKLEFTITER");
        RETURN_IF_ERROR(kLoop(1, false));
        sub(regKIter, 1);
        jne(".QLOOPKLEFTITER", T_NEAR);

        L(".QCONSIDKLEFTREM");
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeftRem)]);
        test(regKIter, regKIter);
        je(".QPOSTACCUM", T_NEAR);

        RETURN_IF_ERROR(kLoop(1, true));
    }

    L(".QPOSTACCUM");

    // Compensation for converting signed int8 A to uint8 for sym_quant
    if ((qParams.aQuant.zeroPoint.storeDt
         == dlp::kernel_frame::DataType::invalid)
        || (qParams.bQuant.zeroPoint.storeDt
            == dlp::kernel_frame::DataType::invalid)) {
        RETURN_IF_ERROR(conversionCompensation());
    }

    RETURN_IF_ERROR(applyGroupScalesFastPath());
    jmp(".QSCALE", T_NEAR);

    // Group loop; per-group accumulate + dequant when group_size > 0
    L(".QMULTIGROUP");
    if (fBankInRegs) {
        zeroRegBank(); // zero the register-resident bank
    } else {
        zeroStackBank();
    }

    // Preserve mLoop counter (regMiter) and C base pointer (regCPtr).
    // regMiter --(repurposed)-> group counter
    // regCPtr  --(repurposed)-> regAsclPtr
    mov(ptr[rsp + scratchOff], regMiter);
    mov(ptr[rsp + scratchOff2], regCPtr);

    // Initialize the three running group pointers once for the starting
    // group (gAbs == nLeft). Each is advanced by its stride at the loop bottom
    // instead of being recomputed with an imul every group. regKIter holds
    // nLeft as scratch during setup.
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nLeft)]);
    mov(regMiter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, nIter)]);

    // regBsumPtr = b_col_sum_vec + (nLeft*sum_ld + b_sum_offset) * int32
    mov(regBsumPtr,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, b_col_sum_vec)]);
    mov(regTmp3, ptr[stackPtr + GRP_ATTR_OFF(grp_post_op_sum_ld)]);
    imul(regTmp3, regKIter);
    add(regTmp3,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, b_sum_offset)]);
    lea(regBsumPtr, ptr[regBsumPtr + regTmp3 * sizeof(int32_t)]);

    // regBsclPtr = b_scale_factor + (nLeft*ldb + grp_post_op_j) * scale elem.
    // The group term is omitted for a K-collapsing B, which holds a single
    // scale per column for every group.
    mov(regBsclPtr, ptr[stackPtr + GRP_ATTR_OFF(b_scale_factor)]);
    if (bPerGroupK) {
        mov(regTmp3, ptr[stackPtr + GRP_ATTR_OFF(grp_post_op_ldb)]);
        imul(regTmp3, regKIter);
        add(regTmp3, ptr[stackPtr + GRP_ATTR_OFF(grp_post_op_j)]);
    } else {
        mov(regTmp3, ptr[stackPtr + GRP_ATTR_OFF(grp_post_op_j)]);
    }
    lea(regBsclPtr, ptr[regBsclPtr + regTmp3 * bScaleElemBytes()]);

    // regAsclPtr = a_scale_factor + (row_base*lda + nLeft) * scale elem.
    // row_base (regTmp2) and lda are invariant across the group loop, so
    // row_base*lda is hoisted here instead of being recomputed via imul every
    // group.
    mov(regAsclPtr, ptr[stackPtr + GRP_ATTR_OFF(a_scale_factor)]);
    mov(regTmp3, ptr[stackPtr + GRP_ATTR_OFF(grp_post_op_lda)]);
    mov(regTmp1, regTmp2);
    imul(regTmp1, regTmp3);
    if (aPerGroupK) {
        add(regTmp1, regKIter);
    }
    lea(regAsclPtr, ptr[regAsclPtr + regTmp1 * aScaleElemBytes()]);

    L(".QGROUPLOOP");
    initializeRegisters();

    // The final group of the tile (regMiter == 1) may be a partial
    // K-remainder group when group_size does not divide this tile's K: it runs
    // kIterAPLast full VNNI chunks + an optional masked sub-4 tail. Every
    // earlier group is a full group_size (kIterAP chunks). This stops the
    // partial group from reading past the round4(k)-padded A/B tile.
    cmp(regMiter, 1);
    je(".QGRP_LAST", T_NEAR);

    // Full group: kIterAP == group_size / 4 VNNI chunks.
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kIterAP)]);
    test(regKIter, regKIter);
    je(".QGRP_AFTERK", T_NEAR);
    L(".QGKITER");
    RETURN_IF_ERROR(kLoop(1, false));
    sub(regKIter, 1);
    jne(".QGKITER", T_NEAR);
    jmp(".QGRP_AFTERK", T_NEAR);

    // Final (possibly partial) group.
    L(".QGRP_LAST");
    mov(regKIter,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kIterAPLast)]);
    test(regKIter, regKIter);
    je(".QGRP_LASTREM", T_NEAR);
    L(".QGKITERLAST");
    RETURN_IF_ERROR(kLoop(1, false));
    sub(regKIter, 1);
    jne(".QGKITERLAST", T_NEAR);
    L(".QGRP_LASTREM");
    // Masked sub-4 K tail: mask_regs[0] holds kLeftmask == lastGroupK % 4.
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeftRem)]);
    test(regKIter, regKIter);
    je(".QGRP_AFTERK", T_NEAR);
    RETURN_IF_ERROR(kLoop(1, true));

    L(".QGRP_AFTERK");

    // Compensation for converting signed int8 A to uint8 for sym_quant
    if ((qParams.aQuant.zeroPoint.storeDt
         == dlp::kernel_frame::DataType::invalid)
        || (qParams.bQuant.zeroPoint.storeDt
            == dlp::kernel_frame::DataType::invalid)) {
        RETURN_IF_ERROR(conversionCompensationGroup());
    }

    // Dequantization phase
    RETURN_IF_ERROR(dequantAccumToStackGroup());

    // Advance the running group pointers.
    // Strides are re-loaded into regKIter (free scratch here); no imul/group.
    // The B column sums always advance: they are reduced from the packed B tile
    // per K-group regardless of how coarse the scale granularity is. A scale
    // pointer only advances if its array is actually tiled along K; a
    // K-collapsing operand reuses the same scale for every group.
    mov(regKIter, ptr[stackPtr + GRP_ATTR_OFF(grp_post_op_sum_ld)]);
    lea(regBsumPtr, ptr[regBsumPtr + regKIter * sizeof(int32_t)]);
    if (bPerGroupK) {
        mov(regKIter, ptr[stackPtr + GRP_ATTR_OFF(grp_post_op_ldb)]);
        lea(regBsclPtr, ptr[regBsclPtr + regKIter * bScaleElemBytes()]);
    }
    if (aPerGroupK) {
        add(regAsclPtr, aScaleElemBytes());
    }

    sub(regMiter, 1); // groups remaining--
    jne(".QGROUPLOOP", T_NEAR);

    // Restore the mLoop counter, then bring the running F32 total into cReg for
    // the alpha/beta/store rails.
    mov(regMiter, ptr[rsp + scratchOff]);
    if (fBankInRegs) {
        foldRegBankToRegs();
    } else {
        loadStackBankToRegs();
    }

    // Restore the two temps repurposed as running pointers, which the
    // scale/store rails below consume. regRsC (aliased by regBsclPtr) held the
    // byte row stride of C -- restoring it; regCPtr (aliased by regAsclPtr)
    // held the C base -- restore it from the scratch slot.
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsC)]);
    lea(regRsC, ptr[regRsC * sizeof(float)]);
    mov(regCPtr, ptr[rsp + scratchOff2]);

    // Scale by alpha, beta.
    L(".QSCALE");
    if (params.alphaScalingType != dlp::kernel_frame::scalingType::one) {
        RETURN_IF_ERROR(scaleAlphaF32());
    }
    if (params.betaScalingType != dlp::kernel_frame::scalingType::zero) {
        RETURN_IF_ERROR(scaleBeta());
    }

    RETURN_IF_ERROR(generatePostOps(params));

    RETURN_IF_ERROR(storeResult());

    if (params.mLoop) {
        moveCPtr();

        mov(regTmpAptr, regAPtr);
        mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, psA)]);
        lea(regTmpAptr, ptr[regTmpAptr + regTmp1]);
        mov(regAPtr, regTmpAptr);

        // Advance the a-scale row base by MR rows.
        lea(regTmp2, ptr[regTmp2 + MR]);

        // Row-indexed post-ops (bias, matrix add/mul, per-token scale) read the
        // tile's row origin back out of the params block, so the register copy
        // is not enough on its own.
        if (!params.kernelOps.empty()) {
            mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                    + offsetof(dlp_gemm_post_op_attr, post_op_c_i)],
                regTmp2);
        }

        sub(regMiter, 1);
        jne(".QBLOOPI", T_NEAR);
    }

    add(rsp, totalStack);
    vzeroupper();
    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMQuant<KType>::generateKernel(utils::quantGeneratorParams& params)
{
    // Validate parameters.
    RETURN_IF_ERROR(
        utils::jitGeneratorUtils::checkValidGemmParams(params.base));

    MR          = params.base.MR;
    NR          = params.base.NR;
    useMask     = params.base.useMask;
    c_downscale = params.base.c_downscale;

    // NOTE Asymmetric quant is not supported for now; return.
    if ((params.aQuant.zeroPoint.storeDt
         != dlp::kernel_frame::DataType::invalid)
        || (params.bQuant.zeroPoint.storeDt
            != dlp::kernel_frame::DataType::invalid)) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    // Set A and B scale datatypes
    aScaleType = params.aQuant.scale.storeDt;
    bScaleType = params.bQuant.scale.storeDt;

    // Set whether each scale array is tiled along K, which decides if the
    // group index offsets that operand's scale address at all.
    aPerGroupK = params.aQuant.scale.perGroupK;
    bPerGroupK = params.bQuant.scale.perGroupK;

    // Scale-factor storage: f32 and bf16 are supported. Anything else would be
    // silently read as f32 by loadBScales/broadcastAScale, so reject it.
    if ((aScaleType != dlp::kernel_frame::DataType::f32
         && aScaleType != dlp::kernel_frame::DataType::bf16)
        || (bScaleType != dlp::kernel_frame::DataType::f32
            && bScaleType != dlp::kernel_frame::DataType::bf16)) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    // A mixed f32/bf16 pair cannot be described at run time.
    if ((aScaleType == dlp::kernel_frame::DataType::bf16)
        != (bScaleType == dlp::kernel_frame::DataType::bf16)) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    // Allocate registers.
    RETURN_IF_ERROR(allocateReg(params));

    Xbyak::util::StackFrame stackFrame(this, 1, 13, 0);
    initializeStackFrame(stackFrame);

    // Preserve callee-saved xmm6-15 across the kernel call on Windows x64
    // (no-op on Linux/SysV). See utils::winAbiVectorGuard.
    utils::winAbiVectorGuard winAbiGuard(this);

    // Initialize parameters based on mLoop flag.
    initializeParameters(params.base.mLoop);

    // Generate IR loop.
    RETURN_IF_ERROR(generateIrLoop(params));

    return dlp::jit::jitGeneratorError::success;
}

} // namespace amdzen::GEMMcodeGenerator

// Explicit template instantiation
template class amdzen::GEMMcodeGenerator::jitGEMMQuant<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
