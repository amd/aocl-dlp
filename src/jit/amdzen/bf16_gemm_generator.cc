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
jitGEMMBF16<KType>::jitGEMMBF16(void* buffer, size_t bufferSize)
    : Xbyak::CodeGenerator(bufferSize, buffer)
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
    // Load post_op_c_i into regTmp2 (used for M-loop post-op updates)
    // This is needed for both obf16 and of32 since M-loop updates
    // post_op_c_i
    mov(regTmp2,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(lpgemm_post_op_attr, post_op_c_i)]);

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
    for (int i = 0; i < bReg; i++) {
        // Load 32 BF16 elements (64 bytes) into ZMM register
        vmovdqu16(RegType(bRegIdx + i), ptr[regBptr + i * RegBytes]);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMBF16<KType>::BroadcastABF16withB(bool isRemainder)
{
    for (int i = 0; i < MR; i++) {
        if (isRemainder) {
            vpbroadcastw(RegType(aRegIdx), ptr[regTmpAptr]);
        } else {
            vpbroadcastd(RegType(aRegIdx), ptr[regTmpAptr]);
        }
        add(regTmpAptr, regRsA);
        for (int j = 0; j < bReg; j++) {
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
    for (int i = 0; i < kUnroll; i++) {
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
    regTmp2    = stackFrame.t[12];
}

template<utils::kernelInstrType KType>
void
jitGEMMBF16<KType>::regInit()
{
    // Initialize F32 accumulator registers to zero
    vxorps(RegType(cRegIdx), RegType(cRegIdx), RegType(cRegIdx));
    for (int i = 1; i < cReg; i++) {
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
            lea(regCPtr, ptr[regCPtr + power2scale * regRsC]);
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
    for (int i = 0; i < cReg; i++) {
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
                + offsetof(lpgemm_post_op_attr, is_first_k)]);
        test(regTmp1, regTmp1);
        je("BETAOP", T_NEAR);

        mov(regTmpCptr,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, buf_downscale)]);

        // NULL check
        cmp(regTmpCptr, 0);
        je("BETAOP", T_NEAR);

        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
        lea(regTmp1, ptr[regTmp1 * sizeof(int16_t)]);

        add(regTmpCptr, regTmp1);

        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, rs_c_downscale)]);
        lea(regTmp1, ptr[regTmp1 * sizeof(int16_t)]); // BF16 stride

        mov(regKIter, regTmp2);
        imul(regKIter, regTmp1);
        add(regTmpCptr, regKIter);

        vpbroadcastd(Xbyak::Zmm(aRegIdx + 1),
                     ptr[rsp + 0]); // Broadcast from memory

        for (int i = 0; i < MR; i++) {
            for (int j = 0; j < bFullReg; j++) {
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
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < bFullReg; j++) {
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
    mov(regTmpCptr, regCPtr);
    if (c_downscale < DLP_F32) {
        // Check for is_last_k
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, is_last_k)]);
        test(regTmp1, regTmp1);
        je("STOREOP", T_NEAR);

        mov(regTmpCptr,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, buf_downscale)]);

        // NULL check
        cmp(regTmpCptr, 0);
        je("STOREOP", T_NEAR);

        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
        lea(regTmp1, ptr[regTmp1 * sizeof(int16_t)]);

        add(regTmpCptr, regTmp1);

        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, rs_c_downscale)]);
        lea(regTmp1, ptr[regTmp1 * sizeof(int16_t)]); // BF16 stride

        mov(regKIter, regTmp2);
        imul(regKIter, regTmp1);
        add(regTmpCptr, regKIter);

        for (int i = 0; i < MR; i++) {
            for (int j = 0; j < bFullReg; j++) {
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

        jmp("STOREOP_END", T_NEAR);
        L("STOREOP");
    }
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < bFullReg; j++) {
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
    L("STOREOP_END");
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitGEMMBF16<KType>::prefetchC()
{
    mov(regTmpCptr, regCPtr);
    if ((PREFETCH_C_DIST > 0)) {
        for (int i = 0; i < MR; i++) {
            for (int j = 0; j < bFullReg; j++) {
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
    dec(regKIter);
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
    dec(regKIter);
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
            + offsetof(lpgemm_post_op_attr, is_last_k)]);
    test(regTmp1, regTmp1);
    je(label_store_result, T_NEAR);

    // Create and set up kernelOphandler if there are post-ops
    if (!params.kernelOps.empty()) {
        gen::kernelOpsHandler kernelOpsHandler(this, params.kType);

        // post-ops are applied in the last k iteration
        RETURN_IF_ERROR((kernelOpsHandler.generateKernelOps(
            params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemm, params.MR,
            params.NR, params.useMask, params.numMaskRegs, cRegIdx, cReg)));

        // The gelu constants are embedded within the generated JIT kernel.
        // Otherwise a bug was observed whereby the address of gelu constants
        // inside the class turned out to be beyond what JIT can access.
        kernelOpsHandler.generateKernelOpsAttributes();
    }

    // store C
    L(label_store_result);
    RETURN_IF_ERROR(storeResult());

    if (params.mLoop) {
        // Update A pointer
        mov(regTmp1, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, psA)]);
        lea(regTmp1, ptr[regTmp1 * sizeof(int16_t)]); // BF16 size
        imul(regTmp1, regTmp1, MR);
        lea(regAPtr, ptr[regAPtr + regTmp1]);

        // Update post_op_c_i for the next m-iteration
        add(regTmp2, MR);
        // write back the updated post_op_c_i offset to memory since,
        // kernel-ops module reads this offset from memory.
        mov(ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(lpgemm_post_op_attr, post_op_c_i)],
            regTmp2);

        moveCPtr();

        // Decrement m_iter
        dec(regMiter);
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
    for (int i = 0; i < NUM_USABLE_MASKS; i++) {
        mask_regs[i] = Xbyak::Opmask(MASK_START_IDX + i);
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
    Xbyak::util::StackFrame stackFrame(this, 1, 13, 16);
    initializeStackFrame(stackFrame);
    initializeParameters(params.mLoop);

    RETURN_IF_ERROR(generateIrLoop(params));

    return dlp::jit::jitGeneratorError::success;
}

} // namespace amdzen::GEMMcodeGenerator

// Explicit template instantiations for BF16 VNNI-capable instruction sets
template class amdzen::GEMMcodeGenerator::jitGEMMBF16<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
