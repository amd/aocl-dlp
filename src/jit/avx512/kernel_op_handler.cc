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

#include "kernel_op_handler.hh"
#include "avx512_generator_utils.hh"

namespace avx512gen::generator {

using namespace Xbyak;

kernelOpHandler::kernelOpHandler(Xbyak::CodeGenerator* jit,
                                 int                   MR,
                                 int                   NR,
                                 bool                  useMask,
                                 int                   cRegStartIdx,
                                 int                   cRegCount)
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

jitGeneratorError
kernelOpHandler::allocateRegs()
{
    int numElemsPerReg = RegBytes / sizeof(float);
    numFullRegsPerRow  = NR / numElemsPerReg;
    numMaskRegsPerRow  = useMask ? 1 : 0;
    numRegsPerRow      = numFullRegsPerRow + numMaskRegsPerRow;

    // Assuming that we will always use registers from last for
    // accumulators.For example, if we need 24 accumulators, we will use
    // registers from 8 to 31. and the rest will be used for scratch registers.

    // For post-ops like bias, downscale, matadd, matmul, we will need to
    // load NR elements from bias, scale factor, matadd, matmul pointers.
    // So, we will need to use numRegsPerRow scratch registers for this.
    scratchLoadRegIdx = cRegStartIdx - numRegsPerRow;
    scratchBcstRegIdx = 0;

    // registers for gelu_tanh
    const1 = scratchBcstRegIdx + 0;
    const2 = scratchBcstRegIdx + 1;
    x      = scratchBcstRegIdx + 2;
    r      = scratchBcstRegIdx + 3;
    r2     = scratchBcstRegIdx + 4;
    z      = scratchBcstRegIdx + 5;
    dn     = scratchBcstRegIdx + 6;
    x_tanh = scratchBcstRegIdx + 7;
    q      = scratchBcstRegIdx + 8;

    // registers for gelu_erf
    x_erf = scratchBcstRegIdx + 4;

    // check if MR and NR values are correct
    if ((MR * numRegsPerRow != cRegCount)
        || (cRegCount + numRegsPerRow >= numRegs)) {
        return jitGeneratorError::badKernelInfo;
    }

    return jitGeneratorError::success;
}

jitGeneratorError
kernelOpHandler::generatekernelOps(std::vector<kernelOpsMetaData>& kernelOps,
                                   const Xbyak::Reg64&             stackPtr)
{
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

    jit_->mov(regkernelOpsList,
              jit_->ptr[stackPtr
                        + offsetof(dlp::kernels::gemmParams, kernelOpsList)]);

    // Load pointer to kernelOpsAttr struct instead of the struct itself
    jit_->lea(regkernelOpsAttr,
              jit_->ptr[stackPtr
                        + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)]);

    RETURN_IF_ERROR((allocateRegs()));
    for (auto& op : kernelOps) {
        switch (op.type) {
            case kernelOps::bias:
                biasZmm(op);
                break;
            case kernelOps::relu:
                reluZmm();
                break;
            case kernelOps::reluScale:
                reluScale(op);
                break;
            case kernelOps::geluTanh:
                geluTanh(op);
                break;
            case kernelOps::geluErf:
                geluErf(op);
                break;
            case kernelOps::clip:
                clip(op);
                break;
            case kernelOps::downscale:
                downscaleZmm(op);
                break;
            case kernelOps::matAdd:
                mataddZmm(op);
                break;
            case kernelOps::matMul:
                matmulZmm(op);
                break;
            case kernelOps::swish:
                swish(op);
                break;
            case kernelOps::tanh:
                tanh(op);
                break;
            case kernelOps::sigmoid:
                sigmoid(op);
                break;
            default:
                return jitGeneratorError::notSupported;
                break;
        }
        jit_->mov(regkernelOpsList,
                  jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, next)]);
    }

    return jitGeneratorError::success;
}

jitGeneratorError
kernelOpHandler::generateTableStores()
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

jitGeneratorError
kernelOpHandler::reluZmm()
{
    int zeroReg = scratchBcstRegIdx;
    jit_->vpxorq(Zmm(zeroReg), Zmm(zeroReg), Zmm(zeroReg));
    for (int i = 0; i < MR * numRegsPerRow; i++) {
        jit_->vmaxps(Zmm(cRegStartIdx + i), Zmm(cRegStartIdx + i),
                     Zmm(zeroReg));
    }
    return jitGeneratorError::success;
}

template<>
void
kernelOpHandler::biasRowMajorZmm<float>()
{
    // Add code to load bias pointer and offset after deciding the
    // structure definition for post-op run-time parameters

    // Load bias values into ZMMs
    for (int i = 0; i < numFullRegsPerRow; i++) {
        jit_->vmovups(Zmm(scratchLoadRegIdx + i),
                      jit_->ptr[regTmp1 + i * RegBytes]);
    }
    if (numMaskRegsPerRow > 0) {
        jit_->vmovups(Zmm(scratchLoadRegIdx + numFullRegsPerRow) | jit_->k3,
                      jit_->ptr[regTmp1 + numFullRegsPerRow * RegBytes]);
    }

    // Add bias to accumulators
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < numRegsPerRow; j++) {
            jit_->vaddps(Zmm(cRegStartIdx + i * numRegsPerRow + j),
                         Zmm(cRegStartIdx + i * numRegsPerRow + j),
                         Zmm(scratchLoadRegIdx + j));
        }
    }
}
template<>
void
kernelOpHandler::biasColMajorZmm<float>()
{
    // Add code to load bias pointer and offset after deciding the
    // structure definition for post-op run-time parameters

    for (int i = 0; i < MR; i++) {
        jit_->vbroadcastss(Zmm(scratchBcstRegIdx),
                           jit_->ptr[regTmp1 + i * sizeof(float)]);
        for (int j = 0; j < numRegsPerRow; j++) {
            jit_->vaddps(Zmm(cRegStartIdx + i * numRegsPerRow + j),
                         Zmm(cRegStartIdx + i * numRegsPerRow + j),
                         Zmm(scratchBcstRegIdx));
        }
    }
}

jitGeneratorError
kernelOpHandler::biasZmm(kernelOpsMetaData& op)
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
        if (op.paramStorageDt == DataType::f32) {
            jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(float)]);
            jit_->add(regTmp1, regTmp2);
            biasRowMajorZmm<float>();
        } else {
            return jitGeneratorError::notSupported;
        }
    }
    // matrix is col-major
    else {
        jit_->mov(regTmp2,
                  jit_->ptr[regkernelOpsAttr
                            + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
        if (op.paramStorageDt == DataType::f32) {
            jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(float)]);
            jit_->add(regTmp1, regTmp2);
            biasColMajorZmm<float>();
        } else {
            return jitGeneratorError::notSupported;
        }
    }
    return jitGeneratorError::success;
}

template<>
void
kernelOpHandler::reluScaleZmm<float>()
{
    int zeroReg  = scratchBcstRegIdx;
    int scaleReg = scratchLoadRegIdx;

    // Scale value.
    jit_->mov(regTmp1,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args2)]);
    jit_->vbroadcastss(Zmm(scaleReg), jit_->ptr[regTmp1]);

    // Zero out the zeroreg.
    jit_->vpxorq(Zmm(zeroReg), Zmm(zeroReg), Zmm(zeroReg));

    for (int i = 0; i < MR * numRegsPerRow; i++) {
        jit_->vcmpps(jit_->k5, Zmm(cRegStartIdx + i), Zmm(zeroReg), 0x02);
        jit_->vmulps(Zmm(cRegStartIdx + i) | jit_->k5, Zmm(cRegStartIdx + i),
                     Zmm(scaleReg));
    }
}

jitGeneratorError
kernelOpHandler::reluScale(kernelOpsMetaData& op)
{
    if (op.paramStorageDt == DataType::f32) {
        reluScaleZmm<float>();
    } else {
        return jitGeneratorError::notSupported;
    }
    return jitGeneratorError::success;
}

void
kernelOpHandler::store_zmms_in_stack(md_t reg_start_idx, md_t num_regs)
{
    jit_->sub(jit_->rsp, (num_regs * 64));
    for (md_t idx = 0; idx < num_regs; idx++) {
        jit_->vmovups(jit_->ptr[jit_->rsp + idx * 64],
                      Zmm(reg_start_idx + idx));
    }
}

void
kernelOpHandler::get_zmms_from_stack(md_t reg_start_idx, md_t num_regs)
{
    for (md_t idx = 0; idx < num_regs; idx++) {
        jit_->vmovups(Zmm(reg_start_idx + idx),
                      jit_->ptr[jit_->rsp + idx * 64]);
    }
    jit_->add(jit_->rsp, (num_regs * 64));
}

void
kernelOpHandler::apply_post_ops_in_high_reg_pressure(
    const md_t num_post_op_regs, std::function<void(md_t)> op_fn)
{
    md_t num_push_regs = num_post_op_regs - cRegStartIdx;

    // If number of registers required to compute pots op is more than
    // registers available, then push some accum registers to stack
    // and use them to compute gelu.
    store_zmms_in_stack(cRegStartIdx, num_push_regs);

    md_t post_op_start = num_push_regs > 0 ? cRegStartIdx + num_push_regs
                                           : cRegStartIdx;

    // operate on non-pushed regs
    for (md_t reg = post_op_start; reg < 32; reg++) {
        op_fn(reg);
    }

    // Get the saved lower index registers from stack, save last index
    // registers to stack, copy the lower index registers to last index
    // registers, perform op on the last index registers, and then copy
    // from last to lower index registers. This is done since the op uses
    // registers from the lower indices for its computation, and we need
    // to preserve them.
    get_zmms_from_stack(cRegStartIdx, num_push_regs);
    store_zmms_in_stack(32 - num_push_regs, num_push_regs);

    for (md_t reg = 0; reg < num_push_regs; reg++) {
        jit_->vmovups(Zmm(32 - num_push_regs + reg), Zmm(cRegStartIdx + reg));
    }

    for (md_t reg = 0; reg < num_push_regs; reg++) {
        op_fn(32 - num_push_regs + reg);
    }

    for (md_t reg = 0; reg < num_push_regs; reg++) {
        jit_->vmovups(Zmm(cRegStartIdx + reg), Zmm(32 - num_push_regs + reg));
    }

    get_zmms_from_stack(32 - num_push_regs, num_push_regs);
}

// r2 and z, q are scratch regs
// r will be passed in and out of parent function.
void
kernelOpHandler::POLY_EVAL_6_AVX512()
{
    jit_->vmulps(Zmm(r2), Zmm(r), Zmm(r));

    jit_->vbroadcastss(Zmm(const1), get_constant(lpgemm_exp_off, 3));

    jit_->vbroadcastss(Zmm(const2), get_constant(lpgemm_exp_off, 2));

    jit_->vmovups(Zmm(q), Zmm(const2));
    jit_->vfmadd231ps(Zmm(q), Zmm(const1), Zmm(r));

    jit_->vbroadcastss(Zmm(const1), get_constant(lpgemm_exp_off, 1));

    jit_->vbroadcastss(Zmm(const2), get_constant(lpgemm_exp_off, 0));

    jit_->vmovups(Zmm(z), Zmm(const2));
    jit_->vfmadd231ps(Zmm(z), Zmm(const1), Zmm(r));

    jit_->vfmadd231ps(Zmm(z), Zmm(r2), Zmm(q));

    jit_->vmulps(Zmm(r2), Zmm(r2), Zmm(r2));

    jit_->vbroadcastss(Zmm(const1), get_constant(lpgemm_exp_off, 5));

    jit_->vbroadcastss(Zmm(const2), get_constant(lpgemm_exp_off, 4));

    jit_->vfmadd231ps(Zmm(const2), Zmm(const1), Zmm(r));

    jit_->vfmadd231ps(Zmm(z), Zmm(const2), Zmm(r2));
    jit_->vmovups(Zmm(r), Zmm(z));
}

// z, r, dn is a scratch register
// takes 'x' as input and returns 'q' to the parent
void
kernelOpHandler::EXPF_AVX512()
{
    jit_->vbroadcastss(Zmm(const1), get_constant(gelu_macros_off, 0));

    jit_->vmulps(Zmm(z), Zmm(x), Zmm(const1));

    jit_->vbroadcastss(Zmm(const2), get_constant(gelu_macros_off, 1));

    jit_->vaddps(Zmm(dn), Zmm(z), Zmm(const2));

    jit_->vsubps(Zmm(r), Zmm(dn), Zmm(const2));
    jit_->vsubps(Zmm(r), Zmm(z), Zmm(r));

    POLY_EVAL_6_AVX512();

    jit_->vpslld(Zmm(dn), Zmm(dn), 0x17);

    jit_->vpaddd(Zmm(q), Zmm(r), Zmm(dn));

    jit_->vpxorq(Zmm(const2), Zmm(const2), Zmm(const2));

    jit_->vpbroadcastd(Zmm(const1), get_constant(gelu_macros_off, 2));

    jit_->vcmpps(jit_->k5, Zmm(const1), Zmm(x), 0x06);

    jit_->vpandd(Zmm(q) | jit_->k5, Zmm(q), Zmm(const2));

    jit_->vbroadcastss(Zmm(const1), get_constant(gelu_macros_off, 3));

    jit_->vcmpps(jit_->k5, Zmm(const1), Zmm(x), 0x06);

    jit_->vbroadcastss(Zmm(x), get_constant(gelu_macros_off, 4));

    jit_->vpxord(Zmm(x) | jit_->k5, Zmm(q), Zmm(const2));
    jit_->vmovups(Zmm(q), Zmm(x));
}

// uses z, dn, r as scratch regs
// passes r to child macro and gets q
// takes x_tanh as input and gives back x_tanh
void
kernelOpHandler::TANHF_AVX512()
{
    jit_->vbroadcastss(Zmm(const1), get_constant(gelu_consts_off, 2));

    jit_->mov(regTmp5Half, 0x7FFFFFFF);
    jit_->vpbroadcastd(Zmm(const2), regTmp5Half);
    jit_->vpandd(Zmm(x), Zmm(x_tanh), Zmm(const2));

    jit_->vmulps(Zmm(x), Zmm(x), Zmm(const1));

    EXPF_AVX512();

    jit_->mov(regTmp4Half, -1);
    jit_->vbroadcastss(Zmm(const1), get_constant(gelu_consts_off, 4));

    jit_->vaddps(Zmm(z), Zmm(q), Zmm(const1));

    jit_->vbroadcastss(Zmm(const2), get_constant(gelu_consts_off, 5));

    jit_->vaddps(Zmm(r), Zmm(z), Zmm(const2));

    jit_->vdivps(Zmm(z), Zmm(z), Zmm(r));

    jit_->vmulps(Zmm(z), Zmm(z), Zmm(const1));

    jit_->mov(regTmp4Half, -2147483648);
    jit_->vpbroadcastd(Zmm(const1), regTmp4Half);

    jit_->vpandd(Zmm(q), Zmm(x_tanh), Zmm(const1));

    jit_->vpxord(Zmm(x_tanh), Zmm(q), Zmm(z));
}

void
kernelOpHandler::GELU_TANH_F32_AVX512_DEF(md_t reg)
{
    jit_->vmulps(Zmm(r2), Zmm(reg), Zmm(reg));
    jit_->vmulps(Zmm(r2), Zmm(r2), Zmm(reg));

    jit_->vbroadcastss(Zmm(const1), get_constant(gelu_consts_off, 0));
    jit_->vmovups(Zmm(r), Zmm(reg));
    jit_->vfmadd231ps(Zmm(r), Zmm(r2), Zmm(const1));

    jit_->vbroadcastss(Zmm(const2), get_constant(gelu_consts_off, 1));
    jit_->vmulps(Zmm(x_tanh), Zmm(r), Zmm(const2));

    TANHF_AVX512();

    jit_->vbroadcastss(Zmm(const2), get_constant(gelu_consts_off, 6));
    jit_->vaddps(Zmm(x_tanh), Zmm(x_tanh), Zmm(const2));
    jit_->vmulps(Zmm(x_tanh), Zmm(x_tanh), Zmm(reg));

    jit_->vbroadcastss(Zmm(const1), get_constant(gelu_consts_off, 3));
    jit_->vmulps(Zmm(reg), Zmm(x_tanh), Zmm(const1));
}

template<>
void
kernelOpHandler::geluTanhZmm<float>()
{
    apply_post_ops_in_high_reg_pressure(
        num_gelu_regs, std::bind(&kernelOpHandler::GELU_TANH_F32_AVX512_DEF,
                                 this, std::placeholders::_1));
}

jitGeneratorError
kernelOpHandler::geluTanh(kernelOpsMetaData& op)
{
    if (op.paramStorageDt == DataType::f32) {
        geluTanhZmm<float>();
    } else {
        return jitGeneratorError::notSupported;
    }
    return jitGeneratorError::success;
}

void
kernelOpHandler::POLY_EVAL_HORNER_16_0_AVX512()
{
    jit_->vbroadcastss(Zmm(const1), get_constant(lpgemm_erf_off, 15));
    jit_->vbroadcastss(Zmm(const2), get_constant(lpgemm_erf_off, 14));

    jit_->vfmadd231ps(Zmm(const2), Zmm(r), Zmm(const1));

    jit_->vbroadcastss(Zmm(const1), get_constant(lpgemm_erf_off, 13));
    jit_->vfmadd231ps(Zmm(const1), Zmm(r), Zmm(const2));

    jit_->vbroadcastss(Zmm(const2), get_constant(lpgemm_erf_off, 12));
    jit_->vfmadd231ps(Zmm(const2), Zmm(r), Zmm(const1));

    jit_->vbroadcastss(Zmm(const1), get_constant(lpgemm_erf_off, 11));
    jit_->vfmadd231ps(Zmm(const1), Zmm(r), Zmm(const2));

    jit_->vbroadcastss(Zmm(const2), get_constant(lpgemm_erf_off, 10));
    jit_->vfmadd231ps(Zmm(const2), Zmm(r), Zmm(const1));

    jit_->vbroadcastss(Zmm(const1), get_constant(lpgemm_erf_off, 9));
    jit_->vfmadd231ps(Zmm(const1), Zmm(r), Zmm(const2));

    jit_->vbroadcastss(Zmm(const2), get_constant(lpgemm_erf_off, 8));
    jit_->vfmadd231ps(Zmm(const2), Zmm(r), Zmm(const1));

    jit_->vbroadcastss(Zmm(const1), get_constant(lpgemm_erf_off, 7));
    jit_->vfmadd231ps(Zmm(const1), Zmm(r), Zmm(const2));

    jit_->vbroadcastss(Zmm(const2), get_constant(lpgemm_erf_off, 6));
    jit_->vfmadd231ps(Zmm(const2), Zmm(r), Zmm(const1));

    jit_->vbroadcastss(Zmm(const1), get_constant(lpgemm_erf_off, 5));
    jit_->vfmadd231ps(Zmm(const1), Zmm(r), Zmm(const2));

    jit_->vbroadcastss(Zmm(const2), get_constant(lpgemm_erf_off, 4));
    jit_->vfmadd231ps(Zmm(const2), Zmm(r), Zmm(const1));

    jit_->vbroadcastss(Zmm(const1), get_constant(lpgemm_erf_off, 3));
    jit_->vfmadd231ps(Zmm(const1), Zmm(r), Zmm(const2));

    jit_->vbroadcastss(Zmm(const2), get_constant(lpgemm_erf_off, 2));
    jit_->vfmadd231ps(Zmm(const2), Zmm(r), Zmm(const1));

    jit_->vbroadcastss(Zmm(const1), get_constant(lpgemm_erf_off, 1));
    jit_->vfmadd231ps(Zmm(const1), Zmm(r), Zmm(const2));

    jit_->vbroadcastss(Zmm(const2), get_constant(lpgemm_erf_off, 0));
    jit_->vfmadd231ps(Zmm(const2), Zmm(r), Zmm(const1));

    jit_->vmulps(Zmm(x), Zmm(const2), Zmm(r));
}

void
kernelOpHandler::ERF_AVX512()
{
    jit_->mov(regTmp4Half, 0x7FFFFFFF);
    jit_->vpbroadcastd(Zmm(const2), regTmp4Half);
    jit_->vpandd(Zmm(r), Zmm(x_erf), Zmm(const2));

    POLY_EVAL_HORNER_16_0_AVX512();

    jit_->vbroadcastss(Zmm(const1), get_constant(erf_consts_off, 1));

    jit_->vbroadcastss(Zmm(const2), get_constant(erf_consts_off, 3));

    jit_->vcmpps(jit_->k5, Zmm(const2), Zmm(r), 0x06);

    jit_->vpxorq(Zmm(const2), Zmm(const2), Zmm(const2));

    jit_->vpxord(Zmm(const1) | jit_->k5, Zmm(x), Zmm(const2));
    jit_->vmovups(Zmm(x), Zmm(const1));

    jit_->vbroadcastss(Zmm(const1), get_constant(erf_consts_off, 1));

    jit_->vcmpps(jit_->k5, Zmm(const1), Zmm(x), 0x06);

    jit_->vpxord(Zmm(const1) | jit_->k5, Zmm(x), Zmm(const2));

    jit_->mov(regTmp4Half, ~(0x7FFFFFFF));
    jit_->vpbroadcastd(Zmm(const2), regTmp4Half);

    jit_->vpandd(Zmm(x_erf), Zmm(x_erf), Zmm(const2));

    jit_->vpord(Zmm(x_erf), Zmm(x_erf), Zmm(const1));
}

void
kernelOpHandler::GELU_ERF_F32_AVX512_DEF(md_t reg)
{
    jit_->vbroadcastss(Zmm(const1), get_constant(erf_consts_off, 0));
    jit_->vmulps(Zmm(x_erf), Zmm(reg), Zmm(const1));

    ERF_AVX512();

    jit_->vbroadcastss(Zmm(const2), get_constant(erf_consts_off, 1));
    jit_->vaddps(Zmm(x_erf), Zmm(x_erf), Zmm(const2));

    jit_->vmulps(Zmm(x_erf), Zmm(x_erf), Zmm(reg));
    jit_->vbroadcastss(Zmm(const2), get_constant(erf_consts_off, 2));
    jit_->vmulps(Zmm(reg), Zmm(x_erf), Zmm(const2));
}

template<>
void
kernelOpHandler::geluErfZmm<float>()
{
    apply_post_ops_in_high_reg_pressure(
        num_gelu_regs, std::bind(&kernelOpHandler::GELU_ERF_F32_AVX512_DEF,
                                 this, std::placeholders::_1));
}

jitGeneratorError
kernelOpHandler::geluErf(kernelOpsMetaData& op)
{
    if (op.paramStorageDt == DataType::f32) {
        geluErfZmm<float>();
    } else {
        return jitGeneratorError::notSupported;
    }
    return jitGeneratorError::success;
}

template<>
void
kernelOpHandler::clipZmm<float>()
{
    // add code to query clip min and max arg type and then load and
    // convert them to float and then broadcast them to a ZMM

    int minReg = scratchBcstRegIdx;
    int maxReg = scratchLoadRegIdx;

    // Min value.
    jit_->mov(regTmp1,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args2)]);
    jit_->vbroadcastss(Zmm(minReg), jit_->ptr[regTmp1]);

    // Max value.
    jit_->mov(regTmp1,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args3)]);
    jit_->vbroadcastss(Zmm(maxReg), jit_->ptr[regTmp1]);

    for (int i = 0; i < MR * numRegsPerRow; i++) {
        jit_->vmaxps(Zmm(cRegStartIdx + i), Zmm(cRegStartIdx + i), Zmm(minReg));
        jit_->vminps(Zmm(cRegStartIdx + i), Zmm(cRegStartIdx + i), Zmm(maxReg));
    }
}

jitGeneratorError
kernelOpHandler::clip(kernelOpsMetaData& op)
{
    if (op.paramStorageDt == DataType::f32) {
        clipZmm<float>();
    } else {
        return jitGeneratorError::notSupported;
    }
    return jitGeneratorError::success;
}

template<>
void
kernelOpHandler::ScaleFactorScalarZmm<float>()
{
    md_t sf_reg = scratchBcstRegIdx;
    // add code to move scalefactor pointer to regTmp1
    jit_->vbroadcastss(Zmm(sf_reg), jit_->ptr[regTmp1]);
    for (md_t i = 0; i < MR * numRegsPerRow; i++) {
        jit_->vmulps(Zmm(cRegStartIdx + i), Zmm(cRegStartIdx + i), Zmm(sf_reg));
    }
}

template<>
void
kernelOpHandler::ScaleFactorRowMajorZmm<float>()
{
    // Since we are keeping enough registers to load NR elements of B,
    // we can safely assume that we will have enough registers to load
    // the NR elements of scale factor.
    jit_->mov(regTmp2, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(float)]);
    jit_->lea(regTmp3, jit_->ptr[regTmp1]);
    jit_->add(regTmp3, regTmp2);
    for (int i = 0; i < numFullRegsPerRow; i++) {
        jit_->vmovups(Zmm(scratchLoadRegIdx + i),
                      jit_->ptr[regTmp3 + i * RegBytes]);
        for (md_t j = 0; j < MR; j++) {
            jit_->vmulps(Zmm(cRegStartIdx + j * numRegsPerRow + i),
                         Zmm(cRegStartIdx + j * numRegsPerRow + i),
                         Zmm(scratchLoadRegIdx + i));
        }
    }
    if (numMaskRegsPerRow > 0) {
        // load matadd elements
        jit_->vmovups(Zmm(scratchLoadRegIdx + numFullRegsPerRow) | jit_->k3
                          | jit_->T_z,
                      jit_->ptr[regTmp3 + numFullRegsPerRow * RegBytes]);
        for (md_t j = 0; j < MR; j++) {
            jit_->vmulps(
                Zmm(cRegStartIdx + j * numRegsPerRow + numFullRegsPerRow),
                Zmm(cRegStartIdx + j * numRegsPerRow + numFullRegsPerRow),
                Zmm(scratchLoadRegIdx + numFullRegsPerRow));
        }
    }
}

template<>
void
kernelOpHandler::ScaleFactorColMajorZmm<float>()
{
    // since we are keeping atleast one register for broadcasting A,
    // it is safe to broadcast and apply one at a time.
    jit_->mov(regTmp2, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(float)]);
    jit_->lea(regTmp3, jit_->ptr[regTmp1]);
    jit_->add(regTmp3, regTmp2);
    for (int i = 0; i < MR; i++) {
        jit_->vbroadcastss(Zmm(scratchBcstRegIdx),
                           jit_->ptr[regTmp3 + i * sizeof(float)]);
        for (md_t j = 0; j < numRegsPerRow; j++) {
            jit_->vmulps(Zmm(cRegStartIdx + i * numRegsPerRow + j),
                         Zmm(cRegStartIdx + i * numRegsPerRow + j),
                         Zmm(scratchBcstRegIdx));
        }
    }
}

template<>
void
kernelOpHandler::ZeroPointScalarZmm<float>()
{
    md_t zp_reg = scratchBcstRegIdx;
    jit_->vbroadcastss(Zmm(zp_reg), jit_->ptr[regTmp1]);
    for (md_t i = 0; i < MR * numRegsPerRow; i++) {
        jit_->vaddps(Zmm(cRegStartIdx + i), Zmm(cRegStartIdx + i), Zmm(zp_reg));
    }
}

template<>
void
kernelOpHandler::ZeroPointRowMajorZmm<float>()
{
    jit_->mov(regTmp2, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(float)]);
    jit_->lea(regTmp3, jit_->ptr[regTmp1]);
    jit_->add(regTmp3, regTmp2);
    for (int i = 0; i < numFullRegsPerRow; i++) {
        jit_->vmovups(Zmm(scratchLoadRegIdx + i),
                      jit_->ptr[regTmp3 + i * RegBytes]);
        for (md_t j = 0; j < MR; j++) {
            jit_->vaddps(Zmm(cRegStartIdx + j * numRegsPerRow + i),
                         Zmm(cRegStartIdx + j * numRegsPerRow + i),
                         Zmm(scratchLoadRegIdx + i));
        }
    }
    if (numMaskRegsPerRow > 0) {
        // load matadd elements
        jit_->vmovups(Zmm(scratchLoadRegIdx + numFullRegsPerRow) | jit_->k3
                          | jit_->T_z,
                      jit_->ptr[regTmp3 + numFullRegsPerRow * RegBytes]);
        for (md_t j = 0; j < MR; j++) {
            jit_->vaddps(
                Zmm(cRegStartIdx + j * numRegsPerRow + numFullRegsPerRow),
                Zmm(cRegStartIdx + j * numRegsPerRow + numFullRegsPerRow),
                Zmm(scratchLoadRegIdx + numFullRegsPerRow));
        }
    }
}

template<>
void
kernelOpHandler::ZeroPointColMajorZmm<float>()
{
    jit_->mov(regTmp2, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(float)]);
    jit_->lea(regTmp3, jit_->ptr[regTmp1]);
    jit_->add(regTmp3, regTmp2);
    for (int i = 0; i < MR; i++) {
        jit_->vbroadcastss(Zmm(scratchBcstRegIdx),
                           jit_->ptr[regTmp3 + i * sizeof(float)]);
        for (md_t j = 0; j < numRegsPerRow; j++) {
            jit_->vaddps(Zmm(cRegStartIdx + i * numRegsPerRow + j),
                         Zmm(cRegStartIdx + i * numRegsPerRow + j),
                         Zmm(scratchBcstRegIdx));
        }
    }
}

jitGeneratorError
kernelOpHandler::downscaleZmm(kernelOpsMetaData& op)
{
    if (op.scaleFactorDt == DataType::f32) {
        jit_->mov(regTmp1, jit_->ptr[regkernelOpsList
                                     + offsetof(lpgemm_post_op, scale_factor)]);
        if (op.scalarScaleFactorRequired) {
            ScaleFactorScalarZmm<float>();
        } else if (op.cMatFormat == storageFormat::rowMajor) {
            ScaleFactorRowMajorZmm<float>();
        } else {
            ScaleFactorColMajorZmm<float>();
        }
    } else {
        return jitGeneratorError::notSupported;
    }

    if (op.zeroPointDt == DataType::f32) {
        jit_->mov(
            regTmp1,
            jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args1)]);
        if (op.scalarZeroPointRequired) {
            ZeroPointScalarZmm<float>();
        } else if (op.cMatFormat == storageFormat::rowMajor) {
            ZeroPointRowMajorZmm<float>();
        } else {
            ZeroPointColMajorZmm<float>();
        }
    } else {
        return jitGeneratorError::notSupported;
    }
    return jitGeneratorError::success;
}

template<>
void
kernelOpHandler::matOpScaleFactorZmm<float>(matOpType      opType,
                                            matOpScaleType sclType)
{
    md_t sf_reg = scratchBcstRegIdx;
    if (sclType == matOpScaleType::scalar) {
        jit_->vbroadcastss(Zmm(sf_reg), jit_->ptr[regTmp1]);
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
        for (int i = 0; i < numFullRegsPerRow; i++) {
            jit_->vmovups(Zmm(sf_reg + i), jit_->ptr[regTmp3 + i * RegBytes]);
        }
        if (numMaskRegsPerRow > 0) {
            jit_->vmovups(Zmm(sf_reg + numFullRegsPerRow) | jit_->k3
                              | jit_->T_z,
                          jit_->ptr[regTmp3 + numFullRegsPerRow * RegBytes]);
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
        jit_->vmovups(Zmm(scratchLoadRegIdx), jit_->ptr[regTmp2 + loadIdx]);
        // multiply scale factor with matOp
        jit_->vmulps(Zmm(scratchLoadRegIdx), Zmm(scratchLoadRegIdx),
                     Zmm(sfRegIdx + sclIdx));
        if (opType == matOpType::matOpAdd) {
            jit_->vaddps(Zmm(accumRegIdx), Zmm(accumRegIdx),
                         Zmm(scratchLoadRegIdx));
        } else if (opType == matOpType::matOpMul) {
            jit_->vmulps(Zmm(accumRegIdx), Zmm(accumRegIdx),
                         Zmm(scratchLoadRegIdx));
        }
    };

    int sclIdx = 0;
    for (int i = 0; i < MR; i++) {
        if (sclType == matOpScaleType::columnVector) {
            // broadcast scale factor along the m dimension since the A and B
            // matrices are swapped for column major inputs.
            jit_->vbroadcastss(Zmm(sf_reg),
                               jit_->ptr[regTmp6 + i * sizeof(float)]);
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
}

jitGeneratorError
kernelOpHandler::mataddZmm(kernelOpsMetaData& op)
{
    if (op.scaleFactorDt == DataType::f32) {
        jit_->mov(regTmp1, jit_->ptr[regkernelOpsList
                                     + offsetof(lpgemm_post_op, scale_factor)]);
        if (op.scalarScaleFactorRequired) {
            matOpScaleFactorZmm<float>(matOpType::matOpAdd,
                                       matOpScaleType::scalar);
        } else if (op.cMatFormat == storageFormat::rowMajor) {
            matOpScaleFactorZmm<float>(matOpType::matOpAdd,
                                       matOpScaleType::rowVector);
        } else {
            matOpScaleFactorZmm<float>(matOpType::matOpAdd,
                                       matOpScaleType::columnVector);
        }
    } else {
        return jitGeneratorError::notSupported;
    }
    return jitGeneratorError::success;
}

jitGeneratorError
kernelOpHandler::matmulZmm(kernelOpsMetaData& op)
{
    if (op.scaleFactorDt == DataType::f32) {
        jit_->mov(regTmp1, jit_->ptr[regkernelOpsList
                                     + offsetof(lpgemm_post_op, scale_factor)]);
        if (op.scalarScaleFactorRequired) {
            matOpScaleFactorZmm<float>(matOpType::matOpMul,
                                       matOpScaleType::scalar);
        } else if (op.cMatFormat == storageFormat::rowMajor) {
            matOpScaleFactorZmm<float>(matOpType::matOpMul,
                                       matOpScaleType::rowVector);
        } else {
            matOpScaleFactorZmm<float>(matOpType::matOpMul,
                                       matOpScaleType::columnVector);
        }
    } else {
        return jitGeneratorError::notSupported;
    }
    return jitGeneratorError::success;
}

void
kernelOpHandler::SWISH_F32_AVX512_DEF(md_t reg)
{
    jit_->vpxorq(Zmm(x), Zmm(x), Zmm(x));
    jit_->vfnmadd231ps(Zmm(x), Zmm(reg), Zmm(x_tanh));

    // Input reg x and output reg q.
    EXPF_AVX512();

    jit_->vbroadcastss(Zmm(const1), get_constant(gelu_consts_off, 6));
    jit_->vaddps(Zmm(q), Zmm(q), Zmm(const1));
    jit_->vdivps(Zmm(reg), Zmm(reg), Zmm(q));
}

template<>
void
kernelOpHandler::swishZmm<float>()
{
    jit_->mov(regTmp1,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args2)]);
    jit_->vbroadcastss(Zmm(x_tanh), jit_->ptr[regTmp1]);

    apply_post_ops_in_high_reg_pressure(
        num_gelu_regs, std::bind(&kernelOpHandler::SWISH_F32_AVX512_DEF, this,
                                 std::placeholders::_1));
}

jitGeneratorError
kernelOpHandler::swish(kernelOpsMetaData& op)
{
    if (op.paramStorageDt == DataType::f32) {
        swishZmm<float>();
    } else {
        return jitGeneratorError::notSupported;
    }
    return jitGeneratorError::success;
}

void
kernelOpHandler::TANHF_AVX512_DEF(md_t reg)
{
    jit_->vpxorq(Zmm(x), Zmm(x), Zmm(x));
    jit_->vmovups(Zmm(x_tanh), Zmm(reg));
    TANHF_AVX512();
    jit_->vmovups(Zmm(reg), Zmm(x_tanh));
}

template<>
void
kernelOpHandler::tanhZmm<float>()
{
    apply_post_ops_in_high_reg_pressure(
        num_gelu_regs, std::bind(&kernelOpHandler::TANHF_AVX512_DEF, this,
                                 std::placeholders::_1));
}

jitGeneratorError
kernelOpHandler::tanh(kernelOpsMetaData& op)
{
    if (op.paramStorageDt == DataType::f32) {
        tanhZmm<float>();
    } else {
        return jitGeneratorError::notSupported;
    }
    return jitGeneratorError::success;
}

void
kernelOpHandler::SIGMOID_AVX512_DEF(md_t reg)
{
    jit_->vbroadcastss(Zmm(const1), get_constant(gelu_consts_off, 4));
    jit_->vmulps(Zmm(x), Zmm(const1), Zmm(reg));

    // Input is x, output is q
    EXPF_AVX512();

    jit_->vbroadcastss(Zmm(const1), get_constant(gelu_consts_off, 6));
    jit_->vaddps(Zmm(q), Zmm(q), Zmm(const1));
    jit_->vdivps(Zmm(reg), Zmm(const1), Zmm(q));
}

template<>
void
kernelOpHandler::sigmoidZmm<float>()
{
    apply_post_ops_in_high_reg_pressure(
        num_gelu_regs, std::bind(&kernelOpHandler::SIGMOID_AVX512_DEF, this,
                                 std::placeholders::_1));
}

jitGeneratorError
kernelOpHandler::sigmoid(kernelOpsMetaData& op)
{
    if (op.paramStorageDt == DataType::f32) {
        sigmoidZmm<float>();
    } else {
        return jitGeneratorError::notSupported;
    }
    return jitGeneratorError::success;
}

} // namespace avx512gen::generator
