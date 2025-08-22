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

#include "x86_kernel_ops_generator.hh"
#include "jit_generator_utils.hh"

namespace amdzen::x86gen {

using namespace dlp::kernel_frame;
using namespace dlp::jit;
using namespace Xbyak;

template<utils::kernelInstrType KType>
kernelOpsGeneratorX86<KType>::kernelOpsGeneratorX86(Xbyak::CodeGenerator* jit,
                                                    int                   MR,
                                                    int                   NR,
                                                    bool useMask,
                                                    int  cRegStartIdx,
                                                    int  cRegCount)
    : MR(MR)
    , NR(NR)
    , useMask(useMask)
    , cRegStartIdx(cRegStartIdx)
    , cRegCount(cRegCount)
    , regkernelOpsList(jit->rdx)
    , regkernelOpsAttr(jit->r8)
    , regTmp1(jit->r9)
    , regTmp2(jit->r10)
    , regTmp3(jit->r11)
    , regTmp4(jit->rax)
    , regTmp5(jit->rbx)
    , regTmp6(jit->rdi)
    , regTmp7(jit->rsi)
    , regTmp4Half(jit->eax)
    , regTmp5Half(jit->ebx)
    , jit_(jit)
{
}
template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::generateKernelOps(
    std::vector<kernelOpsMetaData>& kernelOps,
    const Xbyak::Reg64&             postOpsArgWrapperPtrReg)
{
    RETURN_IF_ERROR((setPostOpsContext()));

    // Save registers used by this generator.
    utils::registerGuard<Xbyak::Reg64> rG{ jit_ };
    rG.saveRegister(regkernelOpsList);
    rG.saveRegister(regkernelOpsAttr);
    rG.saveRegister(regTmp1);
    rG.saveRegister(regTmp2);
    rG.saveRegister(regTmp3);
    rG.saveRegister(regTmp4);
    rG.saveRegister(regTmp5);
    rG.saveRegister(regTmp6);
    rG.saveRegister(regTmp7);

    // Load the post-ops node and post-ops attr pointers.
    jit_->mov(regkernelOpsList,
              jit_->ptr[postOpsArgWrapperPtrReg
                        + offsetof(dlp::kernels::gemmParams, kernelOpsList)]);

    // Load pointer to kernelOpsAttr struct instead of the struct itself.
    jit_->lea(regkernelOpsAttr,
              jit_->ptr[postOpsArgWrapperPtrReg
                        + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)]);

    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        jit_->lea(regTmp1,
                  jit_->ptr[postOpsArgWrapperPtrReg
                            + offsetof(dlp::kernels::gemmParams, maskArray)]);
        jit_->vmovdqu(ymmMask,
                      jit_->ptr[regTmp1]); // Load all 8 floats (32 bytes)
    } else {
        if (useMask) {
            jit_->kmovw(
                maskReg,
                jit_->ptr[postOpsArgWrapperPtrReg
                          + offsetof(dlp::kernels::gemmParams, maskF32)]);
        }
    }

    auto retVal = this->dispatchKernelOps<kernelOpsGeneratorX86>(kernelOps);

    return retVal;
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::setPostOpsContext()
{
    int numElemsPerReg = Traits::regBytes / sizeof(float);
    numFullRegsPerRow  = NR / numElemsPerReg;
    numMaskRegsPerRow  = useMask ? 1 : 0;
    numRegsPerRow      = numFullRegsPerRow + numMaskRegsPerRow;

    // Assuming that we will always use registers from last for
    // accumulators.For example, if we need 24 accumulators, we will use
    // registers from 8 to 31. and the rest will be used for scratch registers.

    // pushing all the scratch registers to the queue
    for (int i = 0; i < cRegStartIdx; i++) {
        scratch_reg_queue.push(RegType(i));
    }

    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        int totalRegsNeeded = cRegCount + numRegsPerRow + (useMask ? 1 : 0);
        if (totalRegsNeeded > Traits::numRegs) {
            return jitGeneratorError::badKernelInfo;
        }

        if (useMask) {
            ymmMask = popAndGetScratchReg();
        }

        scratchBcstRegIdx = useMask ? 1 : 0;
    } else {
        // check if MR and NR values are correct
        if ((MR * numRegsPerRow != cRegCount)
            || (cRegCount + numRegsPerRow >= Traits::numRegs)) {
            return jitGeneratorError::badKernelInfo;
        }
        scratchBcstRegIdx = 0;
    }
    // For post-ops like bias, downscale, matadd, matmul, we will need to
    // load NR elements from bias, scale factor, matadd, matmul pointers.
    // So, we will need to use numRegsPerRow scratch registers for this.
    scratchLoadRegIdx = cRegStartIdx - numRegsPerRow;

    // registers for gelu_tanh

    x_tanh = scratchBcstRegIdx + 0;
    const1 = scratchBcstRegIdx + 1;
    const2 = scratchBcstRegIdx + 2;
    x      = scratchBcstRegIdx + 3;
    r      = scratchBcstRegIdx + 4;
    r2     = scratchBcstRegIdx + 5;
    z      = scratchBcstRegIdx + 6;
    dn     = scratchBcstRegIdx + 7;
    q      = scratchBcstRegIdx + 8;

    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::advancePostOpsPtr()
{
    jit_->mov(regkernelOpsList,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, next)]);
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::embedKernelOpsAttributes()
{
    // The extra jump is to ensure none of the embedded gelu contants are
    // executed like they are instructions.
    jit_->jmp("TABLE_STORE_END", jit_->T_NEAR);
    jit_->align(64);
    jit_->L(tables);
    jit_->db(reinterpret_cast<uint8_t*>(&gelu_consts), sizeof(gelu_consts));
    jit_->db(reinterpret_cast<uint8_t*>(&gelu_macros), sizeof(gelu_macros));
    jit_->db(reinterpret_cast<uint8_t*>(&lpgemm_exp), sizeof(lpgemm_exp));
    jit_->db(reinterpret_cast<uint8_t*>(&erf_consts), sizeof(erf_consts));
    jit_->db(reinterpret_cast<uint8_t*>(&lpgemm_erf), sizeof(lpgemm_erf));
    jit_->L("TABLE_STORE_END");
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::loadRowF32(Xbyak::Reg64 addressReg,
                                         int          regStartIdx)
{
    for (int i = 0; i < numFullRegsPerRow; i++) {
        jit_->vmovups(RegType(regStartIdx + i),
                      jit_->ptr[addressReg + i * Traits::regBytes]);
    }
    if (useMask) {
        if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            jit_->vmaskmovps(
                RegType(regStartIdx + numFullRegsPerRow), ymmMask,
                jit_->ptr[addressReg + numFullRegsPerRow * Traits::regBytes]);
        } else {
            jit_->vmovups(
                RegType(regStartIdx + numFullRegsPerRow) | maskReg,
                jit_->ptr[addressReg + numFullRegsPerRow * Traits::regBytes]);
        }
    }
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::relu(kernelOpsMetaData& op)
{
    RegType zeroReg = popAndGetScratchReg();
    jit_->vxorps(zeroReg, zeroReg, zeroReg);
    for (int i = 0; i < MR * numRegsPerRow; i++) {
        jit_->vmaxps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                     zeroReg);
    }
    scratch_reg_queue.push(zeroReg);
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::biasRowMajorImpl()
{
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(T)]);
    jit_->add(regTmp1, regTmp2);

    // Load bias values into Registers
    if constexpr (std::is_same_v<T, float>) {
        loadRowF32(regTmp1, scratchLoadRegIdx);
    } else {
        return jitGeneratorError::notSupported;
    }

    // Add bias to accumulators
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < numRegsPerRow; j++) {
            jit_->vaddps(RegType(cRegStartIdx + i * numRegsPerRow + j),
                         RegType(cRegStartIdx + i * numRegsPerRow + j),
                         RegType(scratchLoadRegIdx + j));
        }
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::biasColMajorImpl()
{
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(T)]);
    jit_->add(regTmp1, regTmp2);

    // Utilizing all the scratch registers for bias ensuring
    // maximum possible independent instructions.
    for (int i = 0; i < MR; i++) {
        RegType bcstReg = popAndGetScratchReg();
        if constexpr (std::is_same_v<T, float>) {
            jit_->vbroadcastss(bcstReg, jit_->ptr[regTmp1 + i * sizeof(float)]);
        } else {
            return jitGeneratorError::notSupported;
        }
        for (int j = 0; j < numRegsPerRow; j++) {
            jit_->vaddps(RegType(cRegStartIdx + i * numRegsPerRow + j),
                         RegType(cRegStartIdx + i * numRegsPerRow + j),
                         bcstReg);
        }
        scratch_reg_queue.push(bcstReg);
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::bias(kernelOpsMetaData& op)
{

    // bias pointer is in op_args1 of lpgemm_post_op struct
    // load bias pointer to regTmp1
    // First load the kernelOpsList pointer, then dereference it to get op_args1
    jit_->mov(regTmp1,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args1)]);

    if (op.cMatFormat == storageFormat::rowMajor) {
        // regkernelOpsAttr now contains a pointer to the struct, so we can
        // access members normally
        jit_->mov(regTmp2,
                  jit_->ptr[regkernelOpsAttr
                            + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
        DISPATCH_BY_DATATYPE(op.paramStorageDt, biasRowMajorImpl);
    }
    // matrix is col-major
    else {
        jit_->mov(regTmp2,
                  jit_->ptr[regkernelOpsAttr
                            + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
        DISPATCH_BY_DATATYPE(op.paramStorageDt, biasColMajorImpl);
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::reluScaleImpl()
{
    RegType zeroReg  = popAndGetScratchReg();
    RegType scaleReg = popAndGetScratchReg();

    // Address of the scale value.
    jit_->mov(regTmp1,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args2)]);

    // Broadcast the scale value.
    if constexpr (std::is_same_v<T, float>) {
        jit_->vbroadcastss(RegType(scaleReg), jit_->ptr[regTmp1]);
    } else {
        return jitGeneratorError::notSupported;
    }

    // Zero out the zeroreg.
    jit_->vxorps(RegType(zeroReg), RegType(zeroReg), RegType(zeroReg));

    if constexpr (Traits::hasMaskSupport) {
        for (int i = 0; i < MR * numRegsPerRow; i++) {
            jit_->vcmpps(jit_->k5, RegType(cRegStartIdx + i), RegType(zeroReg),
                         0x02);
            jit_->vmulps(RegType(cRegStartIdx + i) | jit_->k5,
                         RegType(cRegStartIdx + i), RegType(scaleReg));
        }
    } else {
        RegType scratchReg = popAndGetScratchReg();
        for (int i = 0; i < MR * numRegsPerRow; i++) {
            jit_->vminps(scratchReg, RegType(cRegStartIdx + i), zeroReg);
            jit_->vmaxps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                         zeroReg);
            jit_->vmulps(scratchReg, scratchReg, scaleReg);
            jit_->vorps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                        scratchReg);
        }
        scratch_reg_queue.push(scratchReg);
    }
    scratch_reg_queue.push(zeroReg);
    scratch_reg_queue.push(scaleReg);

    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::reluScale(kernelOpsMetaData& op)
{
    DISPATCH_BY_DATATYPE(op.paramStorageDt, reluScaleImpl);
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::clipImpl()
{
    RegType minReg = popAndGetScratchReg();
    RegType maxReg = popAndGetScratchReg();

    // Broadcast min and max values.
    if constexpr (std::is_same_v<T, float>) {
        jit_->vbroadcastss(minReg, jit_->ptr[regTmp1]);
        jit_->vbroadcastss(maxReg, jit_->ptr[regTmp2]);
    } else {
        return jitGeneratorError::notSupported;
    }

    for (int i = 0; i < MR * numRegsPerRow; i++) {
        jit_->vmaxps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                     minReg);
        jit_->vminps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                     maxReg);
    }
    scratch_reg_queue.push(minReg);
    scratch_reg_queue.push(maxReg);

    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::clip(kernelOpsMetaData& op)
{
    // Load address of min value.
    jit_->mov(regTmp1,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args2)]);

    // Load address of max value.
    jit_->mov(regTmp2,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args3)]);

    DISPATCH_BY_DATATYPE(op.paramStorageDt, clipImpl);
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::scaleFactorScalarImpl()
{
    RegType sfReg = popAndGetScratchReg();

    // Broadcast the scale factor.
    if constexpr (std::is_same_v<T, float>) {
        jit_->vbroadcastss(sfReg, jit_->ptr[regTmp1]);
    } else {
        return jitGeneratorError::notSupported;
    }

    for (int i = 0; i < MR * numRegsPerRow; i++) {
        jit_->vmulps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                     sfReg);
    }
    scratch_reg_queue.push(sfReg);

    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::scaleFactorRowMajorImpl()
{
    // Since we are keeping enough registers to load NR elements of B,
    // we can safely assume that we will have enough registers to load
    // the NR elements of scale factor.
    jit_->mov(regTmp2, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(T)]);
    jit_->lea(regTmp3, jit_->ptr[regTmp1]);
    jit_->add(regTmp3, regTmp2);

    if constexpr (std::is_same_v<T, float>) {
        loadRowF32(regTmp3, scratchLoadRegIdx);
    } else {
        return jitGeneratorError::notSupported;
    }

    for (int i = 0; i < numRegsPerRow; i++) {
        for (int j = 0; j < MR; j++) {
            jit_->vmulps(RegType(cRegStartIdx + j * numRegsPerRow + i),
                         RegType(cRegStartIdx + j * numRegsPerRow + i),
                         RegType(scratchLoadRegIdx + i));
        }
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::scaleFactorColMajorImpl()
{
    // since we are keeping atleast one register for broadcasting A,
    // it is safe to broadcast and apply one at a time.
    jit_->mov(regTmp2, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(T)]);
    jit_->lea(regTmp3, jit_->ptr[regTmp1]);
    jit_->add(regTmp3, regTmp2);
    for (int i = 0; i < MR; i++) {
        RegType bcstReg = popAndGetScratchReg();
        if constexpr (std::is_same_v<T, float>) {
            jit_->vbroadcastss(bcstReg, jit_->ptr[regTmp3 + i * sizeof(float)]);
        } else {
            return jitGeneratorError::notSupported;
        }
        for (int j = 0; j < numRegsPerRow; j++) {
            jit_->vmulps(RegType(cRegStartIdx + i * numRegsPerRow + j),
                         RegType(cRegStartIdx + i * numRegsPerRow + j),
                         bcstReg);
        }
        scratch_reg_queue.push(bcstReg);
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::zeroPointScalarImpl()
{
    RegType zpReg = popAndGetScratchReg();
    if constexpr (std::is_same_v<T, float>) {
        jit_->vbroadcastss(zpReg, jit_->ptr[regTmp1]);
    } else {
        return jitGeneratorError::notSupported;
    }
    for (int i = 0; i < MR * numRegsPerRow; i++) {
        jit_->vaddps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                     zpReg);
    }
    scratch_reg_queue.push(zpReg);
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::zeroPointRowMajorImpl()
{
    jit_->mov(regTmp2, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(T)]);
    jit_->lea(regTmp3, jit_->ptr[regTmp1]);
    jit_->add(regTmp3, regTmp2);

    if constexpr (std::is_same_v<T, float>) {
        loadRowF32(regTmp3, scratchLoadRegIdx);
    } else {
        return jitGeneratorError::notSupported;
    }

    for (int i = 0; i < numRegsPerRow; i++) {
        for (int j = 0; j < MR; j++) {
            jit_->vaddps(RegType(cRegStartIdx + j * numRegsPerRow + i),
                         RegType(cRegStartIdx + j * numRegsPerRow + i),
                         RegType(scratchLoadRegIdx + i));
        }
    }

    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::zeroPointColMajorImpl()
{
    jit_->mov(regTmp2, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(T)]);
    jit_->lea(regTmp3, jit_->ptr[regTmp1]);
    jit_->add(regTmp3, regTmp2);
    for (int i = 0; i < MR; i++) {
        RegType bcstReg = popAndGetScratchReg();
        if constexpr (std::is_same_v<T, float>) {
            jit_->vbroadcastss(bcstReg, jit_->ptr[regTmp3 + i * sizeof(float)]);
        } else {
            return jitGeneratorError::notSupported;
        }
        for (md_t j = 0; j < numRegsPerRow; j++) {
            jit_->vaddps(RegType(cRegStartIdx + i * numRegsPerRow + j),
                         RegType(cRegStartIdx + i * numRegsPerRow + j),
                         bcstReg);
        }
        scratch_reg_queue.push(bcstReg);
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::downscale(kernelOpsMetaData& op)
{
    jit_->mov(
        regTmp1,
        jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, scale_factor)]);

    if (op.scalarScaleFactorRequired) {
        DISPATCH_BY_DATATYPE(op.scaleFactorDt, scaleFactorScalarImpl);
    } else if (op.cMatFormat == storageFormat::rowMajor) {
        DISPATCH_BY_DATATYPE(op.scaleFactorDt, scaleFactorRowMajorImpl);
    } else {
        DISPATCH_BY_DATATYPE(op.scaleFactorDt, scaleFactorColMajorImpl);
    }

    jit_->mov(regTmp1,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args1)]);
    if (op.scalarZeroPointRequired) {
        DISPATCH_BY_DATATYPE(op.zeroPointDt, zeroPointScalarImpl);
    } else if (op.cMatFormat == storageFormat::rowMajor) {
        DISPATCH_BY_DATATYPE(op.zeroPointDt, zeroPointRowMajorImpl);
    } else {
        DISPATCH_BY_DATATYPE(op.zeroPointDt, zeroPointColMajorImpl);
    }

    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::matOpScaleFactorImpl(matOpType      opType,
                                                   matOpScaleType sclType)
{
    md_t sf_reg = scratchBcstRegIdx;
    if (sclType == matOpScaleType::scalar) {
        if constexpr (std::is_same_v<T, float>) {
            jit_->vbroadcastss(RegType(sf_reg), jit_->ptr[regTmp1]);
        } else {
            return jitGeneratorError::notSupported;
        }
    }

    jit_->mov(regTmp7, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
    jit_->lea(regTmp7, jit_->ptr[regTmp7 * sizeof(float)]);
    jit_->mov(regTmp6, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_i)]);

    if (sclType == matOpScaleType::rowVector) {
        jit_->lea(regTmp3, jit_->ptr[regTmp1]);
        jit_->add(regTmp3, regTmp7);
        // load NR elements of scale factor
        if constexpr (std::is_same_v<T, float>) {
            loadRowF32(regTmp3, sf_reg);
        } else {
            return jitGeneratorError::notSupported;
        }
    }

    // regTmp2 = matPtr
    jit_->mov(regTmp2,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args1)]);
    // regTmp3 = ldm
    jit_->mov(regTmp3,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args3)]);
    // ldm is pointer, need to dereference again to get actual ldm value.
    jit_->mov(regTmp3, jit_->ptr[regTmp3]);
    jit_->lea(regTmp3, jit_->ptr[regTmp3 * sizeof(float)]);

    jit_->imul(regTmp6, regTmp3);
    jit_->add(regTmp7, regTmp6);
    jit_->add(regTmp2, regTmp7);

    if (sclType == matOpScaleType::columnVector) {
        jit_->mov(regTmp6,
                  jit_->ptr[regkernelOpsAttr
                            + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
        jit_->lea(regTmp6, jit_->ptr[regTmp6 * sizeof(float)]);
        jit_->add(regTmp6, regTmp1);
    }

    auto opLambda = [&](matOpType opType, int sfRegIdx, int scratchLoadRegIdx,
                        int accumRegIdx, int sclIdx, int loadIdx) -> void {
        // load matOp elements
        jit_->vmovups(RegType(scratchLoadRegIdx), jit_->ptr[regTmp2 + loadIdx]);
        // multiply scale factor with matOp
        jit_->vmulps(RegType(scratchLoadRegIdx), RegType(scratchLoadRegIdx),
                     RegType(sfRegIdx + sclIdx));
        if (opType == matOpType::matOpAdd) {
            jit_->vaddps(RegType(accumRegIdx), RegType(accumRegIdx),
                         RegType(scratchLoadRegIdx));
        } else if (opType == matOpType::matOpMul) {
            jit_->vmulps(RegType(accumRegIdx), RegType(accumRegIdx),
                         RegType(scratchLoadRegIdx));
        }
    };

    int sclIdx = 0;
    for (int i = 0; i < MR; i++) {
        if (sclType == matOpScaleType::columnVector) {
            // broadcast scale factor along the m dimension since the A and B
            // matrices are swapped for column major inputs.
            if constexpr (std::is_same_v<T, float>) {
                jit_->vbroadcastss(RegType(sf_reg),
                                   jit_->ptr[regTmp6 + i * sizeof(float)]);
            } else {
                return jitGeneratorError::notSupported;
            }
        }
        for (int j = 0; j < numFullRegsPerRow; j++) {
            if (sclType == matOpScaleType::rowVector) {
                sclIdx = j;
            }
            opLambda(opType, sf_reg, scratchLoadRegIdx,
                     (cRegStartIdx + (i * numRegsPerRow) + j), sclIdx,
                     (j * RegBytes));
        }
        if (numMaskRegsPerRow > 0) {
            if (sclType == matOpScaleType::rowVector) {
                sclIdx = numFullRegsPerRow;
            }
            opLambda(opType, sf_reg, scratchLoadRegIdx,
                     (cRegStartIdx + (i * numRegsPerRow) + numFullRegsPerRow),
                     sclIdx, (numFullRegsPerRow * RegBytes));
        }
        // add ldm to matadd pointer
        jit_->add(regTmp2, regTmp3);
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::matadd(kernelOpsMetaData& op)
{
    jit_->mov(
        regTmp1,
        jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, scale_factor)]);

    if (op.scalarScaleFactorRequired) {
        DISPATCH_BY_DATATYPE(op.scaleFactorDt, matOpScaleFactorImpl,
                             matOpType::matOpAdd, matOpScaleType::scalar);
    } else if (op.cMatFormat == storageFormat::rowMajor) {
        DISPATCH_BY_DATATYPE(op.scaleFactorDt, matOpScaleFactorImpl,
                             matOpType::matOpAdd, matOpScaleType::rowVector);
    } else {
        DISPATCH_BY_DATATYPE(op.scaleFactorDt, matOpScaleFactorImpl,
                             matOpType::matOpAdd, matOpScaleType::columnVector);
    }

    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::matmul(kernelOpsMetaData& op)
{
    jit_->mov(
        regTmp1,
        jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, scale_factor)]);

    if (op.scalarScaleFactorRequired) {
        DISPATCH_BY_DATATYPE(op.scaleFactorDt, matOpScaleFactorImpl,
                             matOpType::matOpMul, matOpScaleType::scalar);
    } else if (op.cMatFormat == storageFormat::rowMajor) {
        DISPATCH_BY_DATATYPE(op.scaleFactorDt, matOpScaleFactorImpl,
                             matOpType::matOpMul, matOpScaleType::rowVector);
    } else {
        DISPATCH_BY_DATATYPE(op.scaleFactorDt, matOpScaleFactorImpl,
                             matOpType::matOpMul, matOpScaleType::columnVector);
    }

    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::store_reg_in_stack(md_t reg_start_idx,
                                                 md_t num_regs)
{
    jit_->sub(jit_->rsp, (num_regs * RegBytes));
    for (md_t idx = 0; idx < num_regs; idx++) {
        jit_->vmovups(jit_->ptr[jit_->rsp + idx * RegBytes],
                      RegType(reg_start_idx + idx));
    }
}

template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::get_reg_from_stack(md_t reg_start_idx,
                                                 md_t num_regs)
{
    for (md_t idx = 0; idx < num_regs; idx++) {
        jit_->vmovups(RegType(reg_start_idx + idx),
                      jit_->ptr[jit_->rsp + idx * RegBytes]);
    }
    jit_->add(jit_->rsp, (num_regs * RegBytes));
}

template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::apply_post_ops_in_high_reg_pressure(
    const md_t num_post_op_regs, std::function<void(md_t)> op_fn)
{
    md_t num_push_regs = num_post_op_regs - cRegStartIdx;

    // If number of registers required to compute pots op is more than
    // registers available, then push some accum registers to stack
    // and use them to compute gelu.
    store_reg_in_stack(cRegStartIdx, num_push_regs);

    md_t post_op_start = num_push_regs > 0 ? cRegStartIdx + num_push_regs
                                           : cRegStartIdx;

    // operate on non-pushed regs
    for (md_t reg = post_op_start; reg < numRegs; reg++) {
        op_fn(reg);
    }

    // Get the saved lower index registers from stack, save last index
    // registers to stack, copy the lower index registers to last index
    // registers, perform op on the last index registers, and then copy
    // from last to lower index registers. This is done since the op uses
    // registers from the lower indices for its computation, and we need
    // to preserve them.
    get_reg_from_stack(cRegStartIdx, num_push_regs);
    store_reg_in_stack(numRegs - num_push_regs, num_push_regs);

    for (md_t reg = 0; reg < num_push_regs; reg++) {
        jit_->vmovups(RegType(numRegs - num_push_regs + reg),
                      RegType(cRegStartIdx + reg));
    }

    for (md_t reg = 0; reg < num_push_regs; reg++) {
        op_fn(numRegs - num_push_regs + reg);
    }

    for (md_t reg = 0; reg < num_push_regs; reg++) {
        jit_->vmovups(RegType(cRegStartIdx + reg),
                      RegType(numRegs - num_push_regs + reg));
    }

    get_reg_from_stack(numRegs - num_push_regs, num_push_regs);
}

template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::POLY_EVAL_6()
{
    jit_->vmulps(RegType(r2), RegType(r), RegType(r));

    jit_->vbroadcastss(RegType(const1), get_constant(lpgemm_exp_off, 3));

    jit_->vbroadcastss(RegType(const2), get_constant(lpgemm_exp_off, 2));

    jit_->vmovups(RegType(q), RegType(const2));
    jit_->vfmadd231ps(RegType(q), RegType(const1), RegType(r));

    jit_->vbroadcastss(RegType(const1), get_constant(lpgemm_exp_off, 1));

    jit_->vbroadcastss(RegType(const2), get_constant(lpgemm_exp_off, 0));

    jit_->vmovups(RegType(z), RegType(const2));
    jit_->vfmadd231ps(RegType(z), RegType(const1), RegType(r));

    jit_->vfmadd231ps(RegType(z), RegType(r2), RegType(q));

    jit_->vmulps(RegType(r2), RegType(r2), RegType(r2));

    jit_->vbroadcastss(RegType(const1), get_constant(lpgemm_exp_off, 5));

    jit_->vbroadcastss(RegType(const2), get_constant(lpgemm_exp_off, 4));

    jit_->vfmadd231ps(RegType(const2), RegType(const1), RegType(r));

    jit_->vfmadd231ps(RegType(z), RegType(const2), RegType(r2));
    jit_->vmovups(RegType(r), RegType(z));
}

template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::EXPF()
{
    jit_->vbroadcastss(RegType(const1), get_constant(gelu_macros_off, 0));

    jit_->vmulps(RegType(z), RegType(x), RegType(const1));

    jit_->vbroadcastss(RegType(const2), get_constant(gelu_macros_off, 1));

    jit_->vaddps(RegType(dn), RegType(z), RegType(const2));

    jit_->vsubps(RegType(r), RegType(dn), RegType(const2));
    jit_->vsubps(RegType(r), RegType(z), RegType(r));

    POLY_EVAL_6();

    jit_->vpslld(RegType(dn), RegType(dn), 0x17);

    jit_->vpaddd(RegType(q), RegType(r), RegType(dn));

    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        // _mm256_set1_ps(EXPF_MAX)
        jit_->vpbroadcastd(RegType(const1), get_constant(gelu_macros_off, 3));

        jit_->vcmpps(RegType(const1), RegType(const1), RegType(x), 0x01);

        // _mm256_set1_ps(inf)
        jit_->vbroadcastss(RegType(const2), get_constant(gelu_macros_off, 4));

        jit_->vblendvps(RegType(q), RegType(q), RegType(const2),
                        RegType(const1));

        // _mm256_set1_ps(EXPF_MIN)
        jit_->vbroadcastss(RegType(const1), get_constant(gelu_macros_off, 2));

        jit_->vcmpps(RegType(const1), RegType(x), RegType(const1), 0x01);

        // _mm256_set1_ps(0.0)
        jit_->vxorps(RegType(const2), RegType(const2), RegType(const2));

        jit_->vblendvps(RegType(q), RegType(q), RegType(const2),
                        RegType(const1));
    } else {
        jit_->vpxorq(RegType(const2), RegType(const2), RegType(const2));

        jit_->vpbroadcastd(RegType(const1), get_constant(gelu_macros_off, 2));

        jit_->vcmpps(jit_->k5, RegType(const1), RegType(x), 0x06);

        jit_->vpandd(RegType(q) | jit_->k5, RegType(q), RegType(const2));

        jit_->vbroadcastss(RegType(const1), get_constant(gelu_macros_off, 3));

        jit_->vcmpps(jit_->k5, RegType(const1), RegType(x), 0x06);

        jit_->vbroadcastss(RegType(x), get_constant(gelu_macros_off, 4));

        jit_->vpxord(RegType(x) | jit_->k5, RegType(q), RegType(const2));
        jit_->vmovups(RegType(q), RegType(x));
    }
}

template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::TANHF()
{
    jit_->vbroadcastss(RegType(const1), get_constant(gelu_consts_off, 2));

    // vpandd is not supported in AVX2 YMM 16 regs.
    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        jit_->vbroadcastss(RegType(const2), get_constant(gelu_consts_off, 7));
        jit_->vandnps(RegType(x), RegType(const2), RegType(x_tanh));
    } else {
        jit_->mov(regTmp5Half, 0x7FFFFFFF);
        jit_->vpbroadcastd(RegType(const2), regTmp5Half);
        jit_->vpandd(RegType(x), RegType(x_tanh), RegType(const2));
    }

    jit_->vmulps(RegType(x), RegType(x), RegType(const1));

    EXPF();

    jit_->mov(regTmp4Half, -1);
    jit_->vbroadcastss(RegType(const1), get_constant(gelu_consts_off, 4));

    jit_->vaddps(RegType(z), RegType(q), RegType(const1));

    jit_->vbroadcastss(RegType(const2), get_constant(gelu_consts_off, 5));

    jit_->vaddps(RegType(r), RegType(z), RegType(const2));

    jit_->vdivps(RegType(z), RegType(z), RegType(r));

    jit_->vmulps(RegType(z), RegType(z), RegType(const1));

    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        jit_->mov(regTmp4Half, -2147483648);
        jit_->movd(Xmm(const1), regTmp4Half);
        jit_->vpbroadcastd(RegType(const1), Xmm(const1));
        jit_->vandps(RegType(q), RegType(x_tanh), RegType(const1));
    } else {
        jit_->mov(regTmp4Half, -2147483648);
        jit_->vpbroadcastd(RegType(const1), regTmp4Half);
        jit_->vpandd(RegType(q), RegType(x_tanh), RegType(const1));
    }

    jit_->vxorps(RegType(x_tanh), RegType(q), RegType(z));
}

template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::GELU_TANH_F32_DEF(md_t reg)
{
    jit_->vmulps(RegType(r2), RegType(reg), RegType(reg));
    jit_->vmulps(RegType(r2), RegType(r2), RegType(reg));

    jit_->vbroadcastss(RegType(const1), get_constant(gelu_consts_off, 0));
    jit_->vmovups(RegType(r), RegType(reg));
    jit_->vfmadd231ps(RegType(r), RegType(r2), RegType(const1));

    jit_->vbroadcastss(RegType(const2), get_constant(gelu_consts_off, 1));
    jit_->vmulps(RegType(x_tanh), RegType(r), RegType(const2));

    TANHF();

    jit_->vbroadcastss(RegType(const2), get_constant(gelu_consts_off, 6));
    jit_->vaddps(RegType(x_tanh), RegType(x_tanh), RegType(const2));
    jit_->vmulps(RegType(x_tanh), RegType(x_tanh), RegType(reg));

    jit_->vbroadcastss(RegType(const1), get_constant(gelu_consts_off, 3));
    jit_->vmulps(RegType(reg), RegType(x_tanh), RegType(const1));
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::geluTanh(kernelOpsMetaData& op)
{
    // Always done on float accumulators, so we need not check for
    // datatype.
    apply_post_ops_in_high_reg_pressure(
        num_gelu_regs,
        std::bind(&kernelOpsGeneratorX86<KType>::GELU_TANH_F32_DEF, this,
                  std::placeholders::_1));

    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::POLY_EVAL_HORNER_16_0(int r)
{
    jit_->vbroadcastsd(RegType(const1), get_constant_dbl(lpgemm_erf_off, 15));
    jit_->vbroadcastsd(RegType(const2), get_constant_dbl(lpgemm_erf_off, 14));

    jit_->vfmadd231pd(RegType(const2), RegType(r), RegType(const1));

    jit_->vbroadcastsd(RegType(const1), get_constant_dbl(lpgemm_erf_off, 13));
    jit_->vfmadd231pd(RegType(const1), RegType(r), RegType(const2));

    jit_->vbroadcastsd(RegType(const2), get_constant_dbl(lpgemm_erf_off, 12));
    jit_->vfmadd231pd(RegType(const2), RegType(r), RegType(const1));

    jit_->vbroadcastsd(RegType(const1), get_constant_dbl(lpgemm_erf_off, 11));
    jit_->vfmadd231pd(RegType(const1), RegType(r), RegType(const2));

    jit_->vbroadcastsd(RegType(const2), get_constant_dbl(lpgemm_erf_off, 10));
    jit_->vfmadd231pd(RegType(const2), RegType(r), RegType(const1));

    jit_->vbroadcastsd(RegType(const1), get_constant_dbl(lpgemm_erf_off, 9));
    jit_->vfmadd231pd(RegType(const1), RegType(r), RegType(const2));

    jit_->vbroadcastsd(RegType(const2), get_constant_dbl(lpgemm_erf_off, 8));
    jit_->vfmadd231pd(RegType(const2), RegType(r), RegType(const1));

    jit_->vbroadcastsd(RegType(const1), get_constant_dbl(lpgemm_erf_off, 7));
    jit_->vfmadd231pd(RegType(const1), RegType(r), RegType(const2));

    jit_->vbroadcastsd(RegType(const2), get_constant_dbl(lpgemm_erf_off, 6));
    jit_->vfmadd231pd(RegType(const2), RegType(r), RegType(const1));

    jit_->vbroadcastsd(RegType(const1), get_constant_dbl(lpgemm_erf_off, 5));
    jit_->vfmadd231pd(RegType(const1), RegType(r), RegType(const2));

    jit_->vbroadcastsd(RegType(const2), get_constant_dbl(lpgemm_erf_off, 4));
    jit_->vfmadd231pd(RegType(const2), RegType(r), RegType(const1));

    jit_->vbroadcastsd(RegType(const1), get_constant_dbl(lpgemm_erf_off, 3));
    jit_->vfmadd231pd(RegType(const1), RegType(r), RegType(const2));

    jit_->vbroadcastsd(RegType(const2), get_constant_dbl(lpgemm_erf_off, 2));
    jit_->vfmadd231pd(RegType(const2), RegType(r), RegType(const1));

    jit_->vbroadcastsd(RegType(const1), get_constant_dbl(lpgemm_erf_off, 1));
    jit_->vfmadd231pd(RegType(const1), RegType(r), RegType(const2));

    jit_->vbroadcastsd(RegType(const2), get_constant_dbl(lpgemm_erf_off, 0));
    jit_->vfmadd231pd(RegType(const2), RegType(r), RegType(const1));

    jit_->vmulpd(RegType(r), RegType(const2), RegType(r));
}

template<>
void
kernelOpsGeneratorX86<utils::kernelInstrType::avx512_zmm_32_reg>::ERF(int y,
                                                                      int r)
{
    jit_->inLocalLabel();
    // r2  = _mm512_abs_ps(r); -- to be preserved for later
    jit_->mov(regTmp5Half, 0x7FFFFFFF);
    jit_->vpbroadcastd(RegType(const2), regTmp5Half);
    jit_->vpandd(RegType(r2), RegType(r), RegType(const2));

    // Convert first half of float values to double (lower 8 floats -> 4
    // doubles)
    jit_->vcvtps2pd(RegType(y), halfRegType(r2));

    // Extract upper half of float values and convert to double (upper 8 floats
    // -> 4 doubles)
    jit_->vextractf32x8(halfRegType(x), RegType(r2),
                        0x01); // Extract upper 4 floats
    jit_->vcvtps2pd(RegType(x), halfRegType(x));

    POLY_EVAL_HORNER_16_0(y);

    POLY_EVAL_HORNER_16_0(x);

    // Convert double values back to single precision and insert into y
    // Convert x (doubles) back to singles and insert at position 0
    jit_->vcvtpd2ps(halfRegType(y), RegType(y)); // Convert doubles to singles

    // Convert r (doubles) back to singles and insert at position 1
    jit_->vcvtpd2ps(halfRegType(x), RegType(x)); // Convert doubles to singles
    jit_->vinsertf32x8(RegType(y), RegType(y), halfRegType(x),
                       0x01); // Insert at position 1

    // __m512i sign =
    // _mm512_and_epi32(_mm512_castps_si512(r),
    //                  _mm512_set1_epi32((unsigned int)0x80000000));
    jit_->mov(regTmp4Half, 0x80000000);
    jit_->vpbroadcastd(RegType(const2), regTmp4Half);
    jit_->vpandd(RegType(const1), RegType(r), RegType(const2));

    jit_->vorps(RegType(y), RegType(const1), RegType(y));

    // ERF_UBOUND
    jit_->mov(regTmp4Half, 0x407AD447);

    // Assuming absr contains 16 float values
    // Find the maximum value across all lanes
    jit_->vreduceps(RegType(x), RegType(r2), 0x0E); // 0x0E = MAX operation
    // Convert the result to integer representation
    jit_->vextractps(regTmp5Half, Xmm(x), 0); // Extract the scalar result

    // Check if regTmp5Half <= regTmp4Half and jump if true
    jit_->cmp(regTmp5Half, regTmp4Half);
    jit_->jg(".erf_end", jit_->T_NEAR); // Jump if regTmp5Half > regTmp4Half

    jit_->vbroadcastss(RegType(const2), get_constant(erf_consts_off, 4));
    jit_->vcmpps(jit_->k5, RegType(const2), RegType(r2), 0x11);

    jit_->vbroadcastss(RegType(const2), get_constant(erf_consts_off, 1));
    jit_->vblendmps(RegType(y) | jit_->k5, RegType(y), RegType(const2));

    jit_->vorps(RegType(y), RegType(const1), RegType(y));

    jit_->L(".erf_end");

    jit_->outLocalLabel();
}

template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::ERF(int y, int r)
{
    // r2  = _mm512_abs_ps(r); -- to be preserved for later
    jit_->mov(regTmp5Half, 0x7FFFFFFF);
    jit_->vmovd(halfRegType(const2), regTmp5Half);
    jit_->vpbroadcastd(RegType(const2), halfRegType(const2));
    // jit_->vpbroadcastd(RegType(const2), regTmp5Half);
    jit_->vandps(RegType(r2), RegType(r), RegType(const2));

    // Convert first half of float values to double (lower 8 floats -> 4
    // doubles)
    jit_->vcvtps2pd(RegType(y), halfRegType(r2));

    // Extract upper half of float values and convert to double (upper 8 floats
    // -> 4 doubles)
    jit_->vextractf128(halfRegType(x), RegType(r2),
                       0x01); // Extract upper 4 floats
    jit_->vcvtps2pd(RegType(x), halfRegType(x));

    POLY_EVAL_HORNER_16_0(y);

    POLY_EVAL_HORNER_16_0(x);

    // Convert double values back to single precision and insert into y
    // Convert x (doubles) back to singles and insert at position 0
    jit_->vcvtpd2ps(halfRegType(y), RegType(y)); // Convert doubles to singles

    // Convert r (doubles) back to singles and insert at position 1
    jit_->vcvtpd2ps(halfRegType(x), RegType(x)); // Convert doubles to singles
    jit_->vinsertf128(RegType(y), RegType(y), halfRegType(x),
                      0x01); // Insert at position 1

    // ERF_UBOUND
    jit_->mov(regTmp4Half, 0x407AD447);
    jit_->vmovd(halfRegType(const2), regTmp4Half);
    jit_->vpbroadcastd(RegType(const2), halfRegType(const2));
    jit_->vcmpps(RegType(const1), RegType(const2), RegType(r2), 0x01);

    // _mm256_set1_ps(1)
    jit_->vbroadcastss(RegType(const2), get_constant(erf_consts_off, 1));

    jit_->vblendvps(RegType(y), RegType(y), RegType(const2), RegType(const1));

    jit_->vcmpps(RegType(const1), RegType(const2), RegType(y), 0x01);

    jit_->vblendvps(RegType(y), RegType(y), RegType(const2), RegType(const1));

    // // _mm256_and_ps(r, (__m256)_mm256_set1_epi32(~(0x7FFFFFFF)));
    jit_->mov(regTmp4Half, 0x80000000);
    jit_->vmovd(halfRegType(const2), regTmp4Half);
    jit_->vpbroadcastd(RegType(const2), halfRegType(const2));
    jit_->vandps(RegType(const1), RegType(r), RegType(const2));

    jit_->vorps(RegType(y), RegType(const1), RegType(y));
}

template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::GELU_ERF_F32_DEF(md_t reg)
{
    jit_->vbroadcastss(RegType(const1), get_constant(erf_consts_off, 0));
    jit_->vmulps(RegType(r), RegType(reg), RegType(const1));

    jit_->vxorps(RegType(x_tanh), RegType(x_tanh), RegType(x_tanh));

    ERF(x_tanh, r);

    jit_->vbroadcastss(RegType(const2), get_constant(erf_consts_off, 1));
    jit_->vaddps(RegType(r2), RegType(x_tanh), RegType(const2));

    jit_->vmulps(RegType(r2), RegType(r2), RegType(reg));
    jit_->vbroadcastss(RegType(const2), get_constant(erf_consts_off, 2));
    jit_->vmulps(RegType(reg), RegType(r2), RegType(const2));
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::geluErf(kernelOpsMetaData& op)
{
    if constexpr (KType == utils::kernelInstrType::avx512_ymm_32_reg) {
        return jitGeneratorError::notSupported;
    }
    apply_post_ops_in_high_reg_pressure(
        num_gelu_regs,
        std::bind(&kernelOpsGeneratorX86<KType>::GELU_ERF_F32_DEF, this,
                  std::placeholders::_1));
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::SWISH_F32_DEF(md_t reg)
{
    jit_->vxorps(RegType(x), RegType(x), RegType(x));
    jit_->vfnmadd231ps(RegType(x), RegType(reg), RegType(x_tanh));

    // Input reg x and output reg q.
    EXPF();

    jit_->vbroadcastss(RegType(const1), get_constant(gelu_consts_off, 6));
    jit_->vaddps(RegType(q), RegType(q), RegType(const1));
    jit_->vdivps(RegType(reg), RegType(reg), RegType(q));
}

template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::swishImpl()
{
    jit_->mov(regTmp1,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args2)]);
    if constexpr (std::is_same_v<T, float>) {
        jit_->vbroadcastss(RegType(x_tanh), jit_->ptr[regTmp1]);
    } else {
        return jitGeneratorError::notSupported;
    }

    apply_post_ops_in_high_reg_pressure(
        num_gelu_regs, std::bind(&kernelOpsGeneratorX86<KType>::SWISH_F32_DEF,
                                 this, std::placeholders::_1));
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::swish(kernelOpsMetaData& op)
{
    DISPATCH_BY_DATATYPE(op.paramStorageDt, swishImpl);
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::TANHF_DEF(md_t reg)
{
    jit_->vxorps(RegType(x), RegType(x), RegType(x));
    jit_->vmovups(RegType(x_tanh), RegType(reg));
    TANHF();
    jit_->vmovups(RegType(reg), RegType(x_tanh));
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::tanh(kernelOpsMetaData& op)
{
    apply_post_ops_in_high_reg_pressure(
        num_gelu_regs, std::bind(&kernelOpsGeneratorX86<KType>::TANHF_DEF, this,
                                 std::placeholders::_1));
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::SIGMOID_DEF(md_t reg)
{
    jit_->vbroadcastss(RegType(const1), get_constant(gelu_consts_off, 4));
    jit_->vmulps(RegType(x), RegType(const1), RegType(reg));

    // Input is x, output is q
    EXPF();

    jit_->vbroadcastss(RegType(const1), get_constant(gelu_consts_off, 6));
    jit_->vaddps(RegType(q), RegType(q), RegType(const1));
    jit_->vdivps(RegType(reg), RegType(const1), RegType(q));
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::sigmoid(kernelOpsMetaData& op)
{
    apply_post_ops_in_high_reg_pressure(
        num_gelu_regs, std::bind(&kernelOpsGeneratorX86<KType>::SIGMOID_DEF,
                                 this, std::placeholders::_1));
    return jitGeneratorError::success;
}

} // namespace amdzen::x86gen

// Explicit template instantiations to resolve linker errors
template class amdzen::x86gen::kernelOpsGeneratorX86<
    amdzen::utils::kernelInstrType::avx2_ymm_16_reg>;
template class amdzen::x86gen::kernelOpsGeneratorX86<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
template class amdzen::x86gen::kernelOpsGeneratorX86<
    amdzen::utils::kernelInstrType::avx512_ymm_32_reg>;
