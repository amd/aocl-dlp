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

#include "f32_gemm_rd_generator.hh"
#include "jit_register/jit_register.hh"

namespace amdzen::GEMMcodeGenerator {

using namespace Xbyak;

// Constructor
template<utils::kernelInstrType KType>
jitGEMMF32RD<KType>::jitGEMMF32RD(size_t maxSize)
    : Xbyak::CodeGenerator(maxSize, Xbyak::AutoGrow)
{
    // initialize all non-static members to 0
    nSubBlockSize = 0;
    MR            = 0;
    NR            = 0;
    useMask       = 0;
    numMaskRegs   = 0;
    c_downscale   = 0;
    aReg          = 0;
    bReg          = 0;
    cReg          = 0;
    accumReg      = 0;
    aRegIdx       = 0;
    bRegIdx       = 0;
    cRegIdx       = 0;
    accumRegIdx   = 0;
    nMaskRegIdx   = 0;
}

// Generate kernel for F32 RD operations
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32RD<KType>::generateKernel(utils::generatorParams& params)
{
    MR = params.MR;
    NR = params.NR;

    c_downscale = params.c_downscale;
    useMask     = params.useMask;
    numMaskRegs = params.numMaskRegs;

    RETURN_IF_ERROR(allocateReg());
    RETURN_IF_ERROR(allocateMaskRegisters());

    // There are 14 general purpose(64 bit) registers.
    // StackFrame manages these registers, since we are using
    // one register for the input parameter of the function,
    // the rest are used as scratch registers to store variables like
    // pointers, strides, counters, etc.
    // Note that all the scratch registers allocated by the stack frame
    // need not be used by the kernel.
    // Putting inside a scope so that some tables can be generated post
    // the ret instr. StackFrame inserts a ret instr in its destructor.
    Xbyak::util::StackFrame stackFrame(
        this, 1, 12 | Xbyak::util::UseRBPAsFramePointer, 0);
    initializeStackFrame(stackFrame);

    initializeParameters(params.mLoop);

    RETURN_IF_ERROR(generateJrLoop(params));

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMMF32RD<KType>::createMaskFromConstant(int value)
{
    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        int scratch = aRegIdx;

        vxorps(RegType(scratch), RegType(scratch), RegType(scratch));
        vcmpeqps(RegType(scratch), RegType(scratch), RegType(scratch));
        vxorps(RegType(nMaskRegIdx), RegType(nMaskRegIdx),
               RegType(nMaskRegIdx));

        uint8_t blend_imm = (1 << value) - 1;
        vblendps(RegType(nMaskRegIdx), RegType(nMaskRegIdx), RegType(scratch),
                 blend_imm);
    } else {
        // mask is already set in allocateMaskRegisters() for avx512/avx512_256
        // config.
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32RD<KType>::allocateReg()
{

    nSubBlockSize = NR > nElemsPerXmm ? nElemsPerXmm : NR;

    // check if MR, NR are valid
    // valid values of NR are 1 to nElemsPerXmm and all multiples of
    // nElemsPerXmm.
    if (!(MR > 0) || ((NR > nElemsPerXmm) && ((NR % nElemsPerXmm) != 0))) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    aReg     = MR;
    accumReg = MR * nSubBlockSize; // currently only supports when NR %4 == 0
    cReg     = MR;
    bReg     = numRegs - cReg - aReg;

    // Check if we have enough registers
    if (bReg < 1) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
    accumRegIdx = numRegs - accumReg; // start from the end

    // accum registers are reused to store the result after reduction.
    cRegIdx = numRegs - cReg; // start from the end

    bReg    = 1;
    bRegIdx = 0;
    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        if (useMask) {
            // the mask is being set after the accumulation is done.
            // so mask register overlapping with b register is allowed.
            nMaskRegIdx = 0;
        }
    }
    aRegIdx = bRegIdx + bReg;

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32RD<KType>::allocateMaskRegisters()
{
    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        return dlp::jit::jitGeneratorError::success;
    }

    // This same logic will be copied to post-ops
    // module as well to ensure handshake between GEMM and post-ops
    // generator. If this logic changes, we need to change the logic in
    // post-ops module as well. This mask is used to generate masked
    // instructions in post-ops module and store module.
    for (iter_t i = 0; i < numMaskRegs; i++) {
        fringeMask[i] = Opmask(i + 1);
    }

    // setting fringe mask for n-dimension.
    // since this is happening before any values are loaded,
    // it is safe to use ecx as a scratch register.
    int numMaskElems = NR > nElemsPerXmm ? nElemsPerXmm : NR;
    mov(ecx, 0x0F >> (nElemsPerXmm - numMaskElems));
    kmovb(fringeMask[0], ecx);

    kLeftMask = Opmask(numMaskRegs + 1); // mask for k_left

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMMF32RD<KType>::initializeStackFrame(Xbyak::util::StackFrame& stackFrame)
{
    stackPtr     = stackFrame.p[0];
    regCptr      = stackFrame.t[0];
    regBptr      = stackFrame.t[1];
    regAptr      = stackFrame.t[2];
    regTmpAptr   = stackFrame.t[3];
    regTmpBptr   = stackFrame.t[4];
    regTmpCptr   = stackFrame.t[5];
    regRsA       = stackFrame.t[6];
    regCsB       = stackFrame.t[7];
    regCsB3      = stackFrame.t[8];
    regMiter     = stackFrame.t[9];
    regJJCounter = stackFrame.t[10];

    regTmp1 = stackFrame.t[11];

    // regKIter aliases regTmpCptr: regKIter is live during K-loop only,
    // regTmpCptr is reconstructed after K-loop via mov(regTmpCptr, regCptr)
    regKIter = regTmpCptr;
    // regRsC aliases regCsB3: regCsB3 is live during K-loop only,
    // regRsC is live after K-loop only; regCsB3 reconstructed before each
    // K-loop
    regRsC = regCsB3;
}

template<utils::kernelInstrType KType>
void
jitGEMMF32RD<KType>::initializeParameters(bool addIrLoop)
{
    mov(regRsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsA)]);
    lea(regRsA, ptr[regRsA * sizeof(float)]);

    mov(regCsB, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, csB)]);
    lea(regCsB, ptr[regCsB * sizeof(float)]);

    lea(regCsB3, ptr[regCsB + regCsB * 2]); // 3 * cs_b

    if constexpr (KType == utils::kernelInstrType::avx512_ymm_32_reg) {
        kmovb(kLeftMask,
              ptr[stackPtr + offsetof(dlp::kernels::gemmParams, maskF32_8[0])]);
    } else if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        kmovw(kLeftMask,
              ptr[stackPtr + offsetof(dlp::kernels::gemmParams, maskF32[0])]);
    } else {
        // Kept for future use
    }
}
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32RD<KType>::generateKrLoop(int unrollFactor, bool isKFringe)
{
    for (iter_t unroll = 0; unroll < unrollFactor; unroll++) {
        // load MR rows of A
        loadMRrowsOfA(isKFringe);

        for (iter_t unrollB = 0; unrollB < nSubBlockSize; unrollB++) {
            // load column from B
            loadRegF32Values(
                bRegIdx,
                ptr[regTmpBptr + ((unrollB == 3) ? regCsB3 : unrollB * regCsB)],
                isKFringe);
            for (iter_t row = 0; row < MR; row++) {
                // fma with A values
                vfmadd231ps(RegType(accumRegIdx + unrollB * MR + row),
                            RegType(aRegIdx + row), RegType(bRegIdx));
            }
        }
        // increment B pointer
        if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            if (isKFringe) {
                // for avx2 config, we need to increment B pointer by
                // sizeof(float) since we process one element at a time.
                lea(regTmpBptr, ptr[regTmpBptr + sizeof(float)]);
            } else {
                lea(regTmpBptr, ptr[regTmpBptr + RegBytes]);
            }
        } else {
            lea(regTmpBptr, ptr[regTmpBptr + RegBytes]);
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32RD<KType>::generateIrLoop(utils::generatorParams& params)
{

    if (params.mLoop) {

        // Restore the original post_op_c_i from the JR-pushed stack slot
        // into gemmParams, so each IR loop starts from the correct value.
        mov(regTmp1, ptr[rsp]);
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_i)],
            regTmp1);

        mov(regMiter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, mIter)]);
        L(".SLOOP6X4I");
    }

    mov(regTmpAptr, regAptr);
    mov(regTmpBptr, regBptr);
    mov(regTmpCptr, regCptr);

    regInit();

    // Reconstruct regCsB3 = 3 * cs_b before K-loop entry.
    // Required because regRsC (aliased to same register) overwrites it
    // during the previous iteration's store phase.
    lea(regCsB3, ptr[regCsB + regCsB * 2]);

    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kIterBP)]);
    test(regKIter, regKIter);
    je(".CONSIDER_K_LEFT", T_NEAR);

    L(".K_UNROLL_LOOP");

    RETURN_IF_ERROR(generateKrLoop(params.K_UNROLL, false));

    sub(regKIter, 1);
    jne(".K_UNROLL_LOOP", T_NEAR);

    L(".CONSIDER_K_LEFT");

    // handle remaining k_iter
    if (params.K_UNROLL > 1) {
        // if unrolled by more than 1, we can generate a single unroll loop
        // to handle multiples of nElemsPerReg.

        // load k_left
        mov(regKIter,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeft)]);
        // divide k_left by nElemsPerReg, use shift right operation
        shr(regKIter, amdzen::utils::int_log2(nElemsPerReg));
        test(regKIter, regKIter);
        je(".HANDLE_K_LEFT_WITH_MASK", T_NEAR);

        L(".K_UNROLL_1");

        RETURN_IF_ERROR(generateKrLoop(1, false));

        sub(regKIter, 1);
        jne(".K_UNROLL_1", T_NEAR);
    }

    L(".HANDLE_K_LEFT_WITH_MASK");
    // load k_left and calculate k_left % nElemsPerReg
    mov(regKIter, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeft)]);
    and_(regKIter, nElemsPerReg - 1);
    test(regKIter, regKIter);
    je(".POST_ACCUMULATE", T_NEAR);

    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        L(".LOOP_K_LEFT1");
    }

    RETURN_IF_ERROR(generateKrLoop(1, true));

    // For avx2 config, for k < nElemsPerReg, we generate code that
    // processes one element at a time using vmovss instruction.
    // For avx512/avx512_256 config, we generate code that processes
    // all kLeft elements at once using masked load instructions.
    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        sub(regKIter, 1);
        jne(".LOOP_K_LEFT1", T_NEAR);
    }

    L(".POST_ACCUMULATE");

    // Reconstruct regTmpCptr = regCptr after K-loop.
    // Required because regKIter (aliased to same register) overwrites it
    // during the K-loop.
    mov(regTmpCptr, regCptr);

    // Accumulate results convert ZMM or YMM to XMM
    RETURN_IF_ERROR(reduceAccumulation());

    // create n-dimension mask
    if (useMask) {
        createMaskFromConstant(nSubBlockSize);
    }

    // alpha scale
    if (params.alphaScalingType != dlp::kernel_frame::scalingType::one) {
        RETURN_IF_ERROR(alphaScale());
    }

    // check if is_last_k is set
    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
    test(regTmp1, regTmp1);
    je(label_store_result, T_NEAR);

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

        // post-ops are applied in the last k iteration, so we need to
        // scale beta before applying post-ops
        // To-Do: add support for beta scaling if beta is 1 using vaddps
        if (params.betaScalingType == dlp::kernel_frame::scalingType::zero) {
            // skip beta scaling if beta is 0
        } else {
            // beta scaling
            RETURN_IF_ERROR(betaScale());
        }

        // Build vec pool:
        //   - A regs [aRegIdx, bRegIdx): free (safe to clobber)
        //   - B regs [bRegIdx, cRegIdx): preserve (needed post-ops for testing)
        VecPoolType vecPool;
        vecPool.addPreserve(bRegIdx,
                            cRegIdx - bRegIdx); // B regs -> preserve
        vecPool.setAccumulators(cRegIdx, cReg);
        RETURN_IF_ERROR(vecPool.init(this, Traits::regBytes));

        MaskPoolType maskPool;
        maskPool.addPreserve(utils::MASK_START_IDX, useMask ? numMaskRegs : 0);
        if constexpr (KType != utils::kernelInstrType::avx2_ymm_16_reg) {
            maskPool.addPreserve(kLeftMask.getIdx());
        }
        RETURN_IF_ERROR(maskPool.init(this, utils::maskSaveWidth<KType>(),
                                      Traits::reservedMaskBits));

        // Encode the n-dimension mask as an immediate via maskOffset.
        // nSubBlockSize is a JIT-gen-time constant, so no memory write needed.
        int maskOffset =
            useMask ? utils::encodeMaskImmediate((1u << nSubBlockSize) - 1)
                    : -1;

        // Results are always in MR number of XMM registers.
        // since we are using RegType in post-ops module,
        // we always need to use mask to load elements.
        RETURN_IF_ERROR((kernelOpsHandlerPtr->generateKernelOps(
            params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemm, params.MR,
            0, true, 1, cRegIdx, cReg, vecPool, maskPool, maskOffset)));

        mov(regTmpCptr, regCptr);

        // store result without fusing beta with store
        RETURN_IF_ERROR(storeResults(false));
        jmp(".AfterStore", T_NEAR);
    }

    L(label_store_result);
    // store results
    if (params.betaScalingType == dlp::kernel_frame::scalingType::zero) {
        RETURN_IF_ERROR(storeResults(false));
    } else {
        RETURN_IF_ERROR(storeResults(true));
    }

    L(".AfterStore");

    if (params.mLoop) {

        mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsC)]);
        lea(regRsC, ptr[regRsC * sizeof(float)]);

        // move C pointer to next MR block
        calculateRowOffset(MR, regTmp1, regRsC);
        lea(regCptr, ptr[regCptr + regTmp1]);

        // move A pointer to next MR block
        calculateRowOffset(MR, regTmp1, regRsA);
        lea(regAptr, ptr[regAptr + regTmp1]);

        // Update post_op_attr c_i. kernelOpsAttr is not a pointer, so adding
        // two offsets at the same time is safe.
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
        add(regTmp1, MR);
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, post_op_c_i)],
            regTmp1);

        // decrement m_iter
        sub(regMiter, 1);
        jne(".SLOOP6X4I", T_NEAR);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32RD<KType>::alphaScale()
{
    int          alphaRegIdx = aRegIdx;
    Xbyak::Reg64 alphaReg    = regTmp1;
    mov(alphaReg, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, alpha)]);
    vbroadcastss(Xmm(alphaRegIdx), ptr[alphaReg]);
    for (iter_t i = 0; i < MR; i++) {
        vmulps(Xmm(cRegIdx + i), Xmm(cRegIdx + i), Xmm(alphaRegIdx));
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMMF32RD<KType>::calculateRowOffset(int           row,
                                        Xbyak::Reg64& regTmp,
                                        Xbyak::Reg64& regRs)
{
    xor_(regTmp, regTmp);
    // calculate row * rs_a
    int m_val       = row;
    int power2scale = 1;
    while (m_val > 0) {
        if (m_val & 1) {
            lea(regTmp, ptr[regTmp + power2scale * regRs]);
        }
        m_val >>= 1;
        power2scale <<= 1;
    }
}

template<utils::kernelInstrType KType>
void
jitGEMMF32RD<KType>::loadRegF32Values(int                   regIdx,
                                      const Xbyak::Address& address,
                                      bool                  isFringe)
{
    if (isFringe) {
        // zero out the register first
        if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            vmovss(Xmm(regIdx), address);
        } else {
            vxorps(RegType(regIdx), RegType(regIdx), RegType(regIdx));
            vmovups(RegType(regIdx) | kLeftMask, address);
        }
    } else {
        vmovups(RegType(regIdx), address);
    }
}

template<utils::kernelInstrType KType>
void
jitGEMMF32RD<KType>::loadMRrowsOfA(bool isFringe)
{
    mov(regTmp1, regTmpAptr);
    for (iter_t row = 0; row < MR; row++) {
        loadRegF32Values(aRegIdx + row, ptr[regTmp1], isFringe);
        add(regTmp1, regRsA);
    }

    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        if (isFringe) {
            // for avx2 config, we need to increment A pointer by sizeof(float)
            // since we process one element at a time while handling fringe
            // elements.
            add(regTmpAptr, sizeof(float));
        } else {
            add(regTmpAptr, RegBytes);
        }
    } else {
        add(regTmpAptr, RegBytes);
    }
}

template<utils::kernelInstrType KType>
void
jitGEMMF32RD<KType>::regInit()
{
    vxorps(RegType(bRegIdx), RegType(bRegIdx), RegType(bRegIdx));
    for (iter_t i = bRegIdx + 1; i < numRegs; i++) {
        vmovaps(RegType(i), RegType(bRegIdx));
    }
}
template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32RD<KType>::reduceToXmm(int offset, int tmpIdx, int blockSize)
{
    // Function only handles blocks of 4 or fewer ZMMs
    if (blockSize > nElemsPerXmm) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Zero out the temporary registers we'll need
    for (iter_t i = 0; i < nElemsPerXmm; i++) {
        vxorps(Xbyak::Ymm(tmpIdx + i), Xbyak::Ymm(tmpIdx + i),
               Xbyak::Ymm(tmpIdx + i));
    }

    // Extract upper 256-bits and add to lower 256-bits for valid inputs
    // This extact + add logic is specific to AVX512 ISA, when using ZMM
    // registers. In case of using YMM registers, just move it onto temporary
    // registers.
    for (iter_t i = 0; i < blockSize; i++) {
        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            // Extract upper 256-bits to temp YMM
            vextractf32x8(Xbyak::Ymm(tmpIdx + i),
                          Xbyak::Zmm(accumRegIdx + offset + i * MR), 1);
            // Add to lower 256-bits of input ZMM, storing in original ZMM's YMM
            // part
            vaddps(Xbyak::Ymm(tmpIdx + i), Xbyak::Ymm(tmpIdx + i),
                   Xbyak::Ymm(accumRegIdx + offset + i * MR));
        } else {
            vmovups(Xbyak::Ymm(tmpIdx + i),
                    Xbyak::Ymm(accumRegIdx + offset + i * MR));
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
    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        vextractf128(Xbyak::Xmm(tmpIdx + 1), Xbyak::Ymm(tmpIdx),
                     1); // Extract upper 128-bits
    } else {
        vextractf32x4(Xbyak::Xmm(tmpIdx + 1), Xbyak::Ymm(tmpIdx), 1);
    }
    vaddps(Xbyak::Xmm(cRegIdx + offset), Xbyak::Xmm(tmpIdx + 1),
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
jitGEMMF32RD<KType>::reduceAccumulation()
{
    for (iter_t i = 0; i < MR; i++) {
        RETURN_IF_ERROR(reduceToXmm(i, bRegIdx, nSubBlockSize));
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32RD<KType>::betaScale()
{
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsC)]);
    lea(regRsC, ptr[regRsC * sizeof(float)]);

    int betaRegIdx = bRegIdx;
    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        if (useMask) {
            // For cases where aRegIdx + aReg overlaps with accumRegIdx,
            // the accumRegIdx register would have been freed by the
            // reduceAccumulation() function.
            betaRegIdx = aRegIdx + aReg;
        }
    }
    mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, beta)]);
    vbroadcastss(Xmm(betaRegIdx), ptr[regTmp1]);

    for (iter_t i = 0; i < MR; i++) {
        loadRegF32Xmm(aRegIdx + i, ptr[regTmpCptr]);
        vfmadd231ps(Xmm(cRegIdx + i), Xmm(aRegIdx + i), Xmm(betaRegIdx));
        add(regTmpCptr, regRsC);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
jitGEMMF32RD<KType>::storeF32Xmm(int regIdx, const Xbyak::Address& address)
{
    if (useMask) {
        if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            vmaskmovps(address, Xmm(nMaskRegIdx), Xmm(regIdx));
        } else {
            vmovups(address | fringeMask[0], Xmm(regIdx));
        }
    } else {
        vmovups(address, Xmm(regIdx));
    }
}

template<utils::kernelInstrType KType>
void
jitGEMMF32RD<KType>::loadRegF32Xmm(int regIdx, const Xbyak::Address& address)
{
    if (useMask) {
        if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            vmaskmovps(Xmm(regIdx), Xmm(nMaskRegIdx), address);
        } else {
            vmovups(Xmm(regIdx) | fringeMask[0], address);
        }
    } else {
        vmovups(Xmm(regIdx), address);
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32RD<KType>::storeResults(bool fuseBetaWithStore)
{
    Xbyak::Label l_store_bz, l_post_store_bz;
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsC)]);
    lea(regRsC, ptr[regRsC * sizeof(float)]);

    if (fuseBetaWithStore) {
        int betaRegIdx = bRegIdx;
        if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            if (useMask) {
                // For cases where aRegIdx + aReg overlaps with accumRegIdx,
                // the accumRegIdx register would have been freed by the
                // reduceAccumulation() function.
                betaRegIdx = aRegIdx + aReg;
            }
        }
        mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, beta)]);
        vbroadcastss(Xmm(betaRegIdx), ptr[regTmp1]);

        // NOTE: If fuseBetaWithStore is true, then the Decision Engine will
        // pass betaScalingType as generic for k > KC even when beta = 0. Hence,
        // broadcasting beta and checking if it is actually zero during
        // run-time. This conforms to the standard of avoiding accesses to C
        // when beta = 0.
        int scratchRegIdx = aRegIdx;
        vxorps(RegType(scratchRegIdx), RegType(scratchRegIdx),
               RegType(scratchRegIdx));
        vucomiss(Xmm(betaRegIdx), Xmm(scratchRegIdx));
        je(l_store_bz, T_NEAR);

        for (iter_t i = 0; i < MR; i++) {
            loadRegF32Xmm(aRegIdx + i, ptr[regTmpCptr]);
            vfmadd231ps(Xmm(cRegIdx + i), Xmm(aRegIdx + i), Xmm(betaRegIdx));
            storeF32Xmm(cRegIdx + i, ptr[regTmpCptr]);
            add(regTmpCptr, regRsC);
        }
        jmp(l_post_store_bz, T_NEAR);

        L(l_store_bz);
        for (iter_t i = 0; i < MR; i++) {
            storeF32Xmm(cRegIdx + i, ptr[regTmpCptr]);
            add(regTmpCptr, regRsC);
        }

        L(l_post_store_bz);
    } else {
        for (iter_t i = 0; i < MR; i++) {
            storeF32Xmm(cRegIdx + i, ptr[regTmpCptr]);
            add(regTmpCptr, regRsC);
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMF32RD<KType>::generateJrLoop(utils::generatorParams& params)
{
    // Simple Jr loop - just generates the Ir loop for now
    // In a full implementation, this would handle the outer loop over N
    // dimension

    inLocalLabel();

    // before starting the Jr loop, save the value of post_op_c_i
    // so that it can be restored before every Ir loop.
    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
    push(regTmp1);

    mov(regJJCounter, 0);
    L(".SLOOP6X4J");
    mov(regAptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, a)]);
    mov(regBptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, b)]);
    mov(regCptr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, c)]);

    mov(regTmp1, regJJCounter);
    lea(regCptr,
        ptr[regCptr + regTmp1 * sizeof(float)]); // c += jj*sizeof(float)

    mov(regTmp1, regJJCounter);
    imul(regTmp1, regCsB);
    lea(regBptr, ptr[regBptr + regTmp1]); // b += jj*cs_b

    RETURN_IF_ERROR(generateIrLoop(params));

    // Update post_op_attr c_j. kernelOpsAttr is not a pointer, so adding
    // two offsets at the same time is safe.
    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
    add(regTmp1, nSubBlockSize);
    mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, post_op_c_j)],
        regTmp1);

    add(regJJCounter, nSubBlockSize);
    cmp(regJJCounter, NR);
    jl(".SLOOP6X4J", T_NEAR);

    // pop the value of post_op_c_i and discard it
    pop(regTmp1);

    outLocalLabel();

    return dlp::jit::jitGeneratorError::success;
}

// Explicit template instantiations for the supported instruction sets
template class jitGEMMF32RD<utils::kernelInstrType::avx512_zmm_32_reg>;
template class jitGEMMF32RD<utils::kernelInstrType::avx512_ymm_32_reg>;
template class jitGEMMF32RD<utils::kernelInstrType::avx2_ymm_16_reg>;

} // namespace amdzen::GEMMcodeGenerator
