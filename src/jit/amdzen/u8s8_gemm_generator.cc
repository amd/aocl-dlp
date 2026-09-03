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
#include "store/store_emit.hh"

namespace amdzen::gen {

using namespace Xbyak;

template<utils::kernelInstrType KType>
jitU8S8VNNI_GEMM<KType>::jitU8S8VNNI_GEMM(size_t maxSize)
    : Xbyak::CodeGenerator(maxSize, Xbyak::AutoGrow)
{
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::allocateRegisters()
{
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
            + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);

    // Initialize parameter pointers from gemmParams structure
    mov(regCPtr, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, c)]);
    mov(regRsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsA)]);
    mov(regCsA, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, csA)]);
    mov(regRsB, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsB)]);
    mov(regRsC, ptr[stackPtr + offsetof(dlp::kernels::gemmParams, rsC)]);

    for (int i = 0; i < utils::NUM_USABLE_MASKS; i++) {
        mask_regs[i] = Xbyak::Opmask(utils::MASK_START_IDX + i);
    }

    kmovw(mask_regs[0],
          ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeftmask)]);

    // Scale strides for VNNI format and data types
    lea(regRsC, ptr[regRsC * sizeof(int32_t)]);

    mov(regTmpCptr, regCPtr);

    if (useMask) {
        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            kmovw(mask_regs[1],
                  ptr[stackPtr + offsetof(dlp::kernels::gemmParams, maskS32)]);
        }
    }
}

template<utils::kernelInstrType KType>
void
jitU8S8VNNI_GEMM<KType>::initializeAccumulators(
    [[maybe_unused]] utils::generatorParams& params)
{

    // Zero out accumulator registers for int32 results
    if constexpr (Traits::isAVX512) {
        vpxord(RegType(cRegIdx), RegType(cRegIdx), RegType(cRegIdx));
    }

    for (iter_t i = 1; i < cReg; i++) {
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
        sub(regKIter, 1);
        jne(".BLOOPKITER", T_NEAR);

        L(".BCONSIDKLEFT");
        if (params.K_UNROLL == 1) {
            // Legacy single-stage tail: with K_UNROLL=1, kLeft is already
            // the 0..3 K-element residual consumed by the masked VNNI
            // load. Skipping the two-stage tail dispatch removes a
            // per-call test and branch from every K_UNROLL=1 microkernel
            // invocation.
            mov(regKIter,
                ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeft)]);
            test(regKIter, regKIter);
            je(".BPOSTACCUM", T_NEAR);

            RETURN_IF_ERROR(kUnroll(1, true));
        } else {
            // K_UNROLL>1 needs the two-stage tail: kLeftIter full VNNI
            // groups followed by the 0..3 masked K-element residual.
            L(".BCONSIDKLEFTITER");
            mov(regKIter,
                ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeftIter)]);
            test(regKIter, regKIter);
            je(".BCONSIDKLEFTREM", T_NEAR);

            L(".BLOOPKLEFTITER");
            RETURN_IF_ERROR(kUnroll(1, false));
            sub(regKIter, 1);
            jne(".BLOOPKLEFTITER", T_NEAR);

            L(".BCONSIDKLEFTREM");
            mov(regKIter,
                ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kLeftRem)]);
            test(regKIter, regKIter);
            je(".BPOSTACCUM", T_NEAR);

            RETURN_IF_ERROR(kUnroll(1, true));
        }

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
    for (iter_t i = 0; i < bFullReg; i++) {
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
                + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
        test(regTmp1, regTmp1);
        je(".STORE_S32", T_NEAR);

        if (c_downscale == DLP_S8) {
            RETURN_IF_ERROR(storeResultS8());
            jmp(".END_STORE", T_NEAR);
        } else if (c_downscale == DLP_U8) {
            RETURN_IF_ERROR(storeResultU8());
            jmp(".END_STORE", T_NEAR);
        } else if (c_downscale == DLP_F32) {
            RETURN_IF_ERROR(storeResultF32());
            jmp(".END_STORE", T_NEAR);
        } else if (c_downscale == DLP_F16) {
            RETURN_IF_ERROR(storeResultF16());
            jmp(".END_STORE", T_NEAR);
        } else if (c_downscale == DLP_BF16) {
            RETURN_IF_ERROR(storeResultBF16());
            jmp(".END_STORE", T_NEAR);
        } else {
            outLocalLabel();
            return dlp::jit::jitGeneratorError::badKernelInfo;
        }

        L(".STORE_S32");
        RETURN_IF_ERROR(storeResultS32());
        L(".END_STORE");

        outLocalLabel();
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::storeResultS32()
{
    auto finalMask = store::StoreMask::none();
    if (bMaskReg > 0) {
        finalMask = store::StoreMask::opmask(mask_regs[1]);
    }
    const auto convert =
        accumulatorsAreF32
            ? store::StoreSpec::convert(dlp::kernel_frame::DataType::f32,
                                        dlp::kernel_frame::DataType::s32)
            : store::StoreSpec::convert(dlp::kernel_frame::DataType::s32,
                                        dlp::kernel_frame::DataType::s32);
    const store::GemmStoreRequest request{ { cRegIdx, MR, bReg,
                                             store::SourceRegWidth::zmm },
                                           { regTmpCptr, regRsC, finalMask },
                                           convert,
                                           store::StoreTemps::none() };
    return store::emit<KType>(*this, request);
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::storeResultS8()
{
    if constexpr (KType != utils::kernelInstrType::avx512_zmm_32_reg) {
        return dlp::jit::jitGeneratorError::notSupported;
    } else {
        updateCBufferPointers();
        auto finalMask = store::StoreMask::none();
        if (bMaskReg > 0) {
            finalMask = store::StoreMask::opmask(mask_regs[1]);
        }
        const auto convert =
            accumulatorsAreF32
                ? store::StoreSpec::convert(dlp::kernel_frame::DataType::f32,
                                            dlp::kernel_frame::DataType::s8)
                : store::StoreSpec::convert(dlp::kernel_frame::DataType::s32,
                                            dlp::kernel_frame::DataType::s8);
        const auto scratch =
            accumulatorsAreF32
                ? store::StoreTemps::withZmmAndGpr(aRegIdx, regKIter)
                : store::StoreTemps::none();
        const store::GemmStoreRequest request{
            { cRegIdx, MR, bReg, store::SourceRegWidth::zmm },
            { regTmpCptr, regTmp1, finalMask },
            convert,
            scratch
        };
        return store::emit<KType>(*this, request);
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::storeResultU8()
{
    if constexpr (KType != utils::kernelInstrType::avx512_zmm_32_reg) {
        return dlp::jit::jitGeneratorError::notSupported;
    } else {
        updateCBufferPointers();
        auto finalMask = store::StoreMask::none();
        if (bMaskReg > 0) {
            finalMask = store::StoreMask::opmask(mask_regs[1]);
        }
        const auto convert =
            accumulatorsAreF32
                ? store::StoreSpec::convert(dlp::kernel_frame::DataType::f32,
                                            dlp::kernel_frame::DataType::u8)
                : store::StoreSpec::convert(dlp::kernel_frame::DataType::s32,
                                            dlp::kernel_frame::DataType::u8);
        const auto scratch =
            store::StoreTemps::withZmmAndGpr(aRegIdx, regKIter);
        const store::GemmStoreRequest request{
            { cRegIdx, MR, bReg, store::SourceRegWidth::zmm },
            { regTmpCptr, regTmp1, finalMask },
            convert,
            scratch
        };
        return store::emit<KType>(*this, request);
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::storeResultF32()
{
    if constexpr (KType != utils::kernelInstrType::avx512_zmm_32_reg) {
        return dlp::jit::jitGeneratorError::notSupported;
    } else {
        updateCBufferPointers();
        auto finalMask = store::StoreMask::none();
        if (bMaskReg > 0) {
            finalMask = store::StoreMask::opmask(mask_regs[1]);
        }
        const auto convert =
            accumulatorsAreF32
                ? store::StoreSpec::convert(dlp::kernel_frame::DataType::f32,
                                            dlp::kernel_frame::DataType::f32)
                : store::StoreSpec::convert(dlp::kernel_frame::DataType::s32,
                                            dlp::kernel_frame::DataType::f32);
        const store::GemmStoreRequest request{
            { cRegIdx, MR, bReg, store::SourceRegWidth::zmm },
            { regTmpCptr, regTmp1, finalMask },
            convert,
            store::StoreTemps::none()
        };
        return store::emit<KType>(*this, request);
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::storeResultF16()
{
    if constexpr (KType != utils::kernelInstrType::avx512_zmm_32_reg) {
        return dlp::jit::jitGeneratorError::notSupported;
    } else {
        updateCBufferPointers();
        auto finalMask = store::StoreMask::none();
        if (bMaskReg > 0) {
            finalMask = store::StoreMask::opmask(mask_regs[1]);
        }
        const auto scratch = store::StoreTemps::withZmm(aRegIdx);
        const auto convert =
            accumulatorsAreF32
                ? store::StoreSpec::convert(dlp::kernel_frame::DataType::f32,
                                            dlp::kernel_frame::DataType::f16)
                : store::StoreSpec::convert(dlp::kernel_frame::DataType::s32,
                                            dlp::kernel_frame::DataType::f16);
        const store::GemmStoreRequest request{
            { cRegIdx, MR, bReg, store::SourceRegWidth::zmm },
            { regTmpCptr, regTmp1, finalMask },
            convert,
            scratch
        };
        return store::emit<KType>(*this, request);
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::storeResultBF16()
{
    if constexpr (KType != utils::kernelInstrType::avx512_zmm_32_reg) {
        return dlp::jit::jitGeneratorError::notSupported;
    } else {
        updateCBufferPointers();
        auto finalMask = store::StoreMask::none();
        if (bMaskReg > 0) {
            finalMask = store::StoreMask::opmask(mask_regs[1]);
        }
        const auto scratch =
            store::StoreTemps::withZmmAndGpr(aRegIdx, regKIter);
        const auto srcType = accumulatorsAreF32
                                 ? dlp::kernel_frame::DataType::f32
                                 : dlp::kernel_frame::DataType::s32;
        const auto convert = store::StoreSpec::softwareBf16(srcType);
        const store::GemmStoreRequest request{
            { cRegIdx, MR, bReg, store::SourceRegWidth::zmm },
            { regTmpCptr, regTmp1, finalMask },
            convert,
            scratch
        };
        return store::emit<KType>(*this, request);
    }
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
    for (iter_t i = 0; i < cReg; i++) {
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
            + offsetof(dlp_gemm_post_op_attr, buf_downscale)]);

    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);

    if (c_downscale == DLP_BF16 || c_downscale == DLP_F16) {
        lea(regTmp1, ptr[regTmp1 * 2]);
    } else if (c_downscale == DLP_F32) {
        lea(regTmp1, ptr[regTmp1 * 4]);
    }

    add(regTmpCptr, regTmp1);

    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, rs_c_downscale)]);

    if (c_downscale == DLP_BF16 || c_downscale == DLP_F16) {
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
                + offsetof(dlp_gemm_post_op_attr, is_first_k)]);
        test(regTmp1, regTmp1);
        je("BETA_S32",
           T_NEAR); // NOT first_k → accumulate from S32 buffer (beta=1)

        updateCBufferPointers();

        for (iter_t i = 0; i < MR; i++) {
            for (iter_t j = 0; j < bFullReg; j++) {
                vmovdqu8(Xmm(bRegIdx + j), ptr[regTmpCptr + j * 16]);
                vpmovzxbd(Zmm(bRegIdx + j), Xmm(bRegIdx + j));
                // vpdpbusd(Zmm(cRegIdx + i * bReg + j), Zmm(bRegIdx + j),
                //          Zmm(betaRegIdx));
                vpmulld(RegType(bRegIdx + j), RegType(bRegIdx + j),
                        RegType(betaRegIdx));
                vpaddd(RegType(cRegIdx + i * bReg + j),
                       RegType(cRegIdx + i * bReg + j), RegType(bRegIdx + j));
            }

            // Handle masked beta scaling
            if (bMaskReg > 0) {
                vmovdqu8(Xmm(bRegIdx + bFullReg) | mask_regs[1] | T_z,
                         ptr[regTmpCptr + bFullReg * 16]);
                vpmovzxbd(Zmm(bRegIdx + bFullReg), Xmm(bRegIdx + bFullReg));
                // vpdpbusd(Zmm(cRegIdx + i * bReg + bFullReg),
                //          Zmm(bRegIdx + bFullReg), Zmm(betaRegIdx));
                vpmulld(RegType(bRegIdx + bFullReg),
                        RegType(bRegIdx + bFullReg), RegType(betaRegIdx));
                vpaddd(RegType(cRegIdx + i * bReg + bFullReg),
                       RegType(cRegIdx + i * bReg + bFullReg),
                       RegType(bRegIdx + bFullReg));
            }

            add(regTmpCptr, regTmp1);
        }

        jmp("BETA_END", T_NEAR);
        L("BETA_S32");
    } else if (c_downscale == DLP_S8) {
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_first_k)]);
        test(regTmp1, regTmp1);
        je("BETA_S32",
           T_NEAR); // NOT first_k → accumulate from S32 buffer (beta=1)

        updateCBufferPointers();

        for (iter_t i = 0; i < MR; i++) {
            for (iter_t j = 0; j < bFullReg; j++) {
                vmovdqu8(Xmm(bRegIdx + j), ptr[regTmpCptr + j * 16]);
                vpmovsxbd(Zmm(bRegIdx + j), Xmm(bRegIdx + j));
                vpmulld(Zmm(bRegIdx + j), Zmm(bRegIdx + j), Zmm(betaRegIdx));
                vpaddd(Zmm(cRegIdx + i * bReg + j), Zmm(cRegIdx + i * bReg + j),
                       Zmm(bRegIdx + j));
            }
            if (bMaskReg > 0) {
                vmovdqu8(Xmm(bRegIdx + bFullReg) | mask_regs[1] | T_z,
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
                + offsetof(dlp_gemm_post_op_attr, is_first_k)]);
        test(regTmp1, regTmp1);
        je("BETA_S32",
           T_NEAR); // NOT first_k → accumulate from S32 buffer (beta=1)

        updateCBufferPointers();

        for (iter_t i = 0; i < MR; i++) {
            for (iter_t j = 0; j < bFullReg; j++) {
                vcvtps2dq(Zmm(bRegIdx + j), ptr[regTmpCptr + j * RegBytes]);
                vpmulld(Zmm(bRegIdx + j), Zmm(bRegIdx + j), Zmm(betaRegIdx));
                vpaddd(Zmm(cRegIdx + i * bReg + j), Zmm(cRegIdx + i * bReg + j),
                       Zmm(bRegIdx + j));
            }
            if (bMaskReg > 0) {
                vcvtps2dq(Zmm(bRegIdx + bFullReg) | mask_regs[1] | T_z,
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
    } else if (c_downscale == DLP_F16) {
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_first_k)]);
        test(regTmp1, regTmp1);
        je("BETA_S32", T_NEAR);

        updateCBufferPointers();

        for (iter_t i = 0; i < MR; i++) {
            for (iter_t j = 0; j < bFullReg; j++) {
                vmovdqu16(Ymm(bRegIdx + j), ptr[regTmpCptr + j * 32]);
                vcvtph2ps(Zmm(bRegIdx + j), Ymm(bRegIdx + j));
                vcvtps2dq(Zmm(bRegIdx + j), Zmm(bRegIdx + j));
                vpmulld(Zmm(bRegIdx + j), Zmm(bRegIdx + j), Zmm(betaRegIdx));
                vpaddd(Zmm(cRegIdx + i * bReg + j), Zmm(cRegIdx + i * bReg + j),
                       Zmm(bRegIdx + j));
            }
            if (bMaskReg > 0) {
                vmovdqu16(Ymm(bRegIdx + bFullReg) | mask_regs[1] | T_z,
                          ptr[regTmpCptr + bFullReg * 32]);
                vcvtph2ps(Zmm(bRegIdx + bFullReg), Ymm(bRegIdx + bFullReg));
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
    } else if (c_downscale == DLP_BF16) {
        mov(regTmp1,
            ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
                + offsetof(dlp_gemm_post_op_attr, is_first_k)]);
        test(regTmp1, regTmp1);
        je("BETA_S32",
           T_NEAR); // NOT first_k → accumulate from S32 buffer (beta=1)

        updateCBufferPointers();

        for (iter_t i = 0; i < MR; i++) {
            for (iter_t j = 0; j < bFullReg; j++) {
                vmovdqu16(Xbyak::Ymm(bRegIdx + j), ptr[regTmpCptr + j * 32]);
                vpmovsxwd(Xbyak::Zmm(bRegIdx + j), Xbyak::Ymm(bRegIdx + j));
                vpslld(Xbyak::Zmm(bRegIdx + j), Xbyak::Zmm(bRegIdx + j), 16);
                vcvtps2dq(Zmm(bRegIdx + j), Zmm(bRegIdx + j));
                vpmulld(Zmm(bRegIdx + j), Zmm(bRegIdx + j), Zmm(betaRegIdx));
                vpaddd(Zmm(cRegIdx + i * bReg + j), Zmm(cRegIdx + i * bReg + j),
                       Zmm(bRegIdx + j));
            }
            if (bMaskReg > 0) {
                vmovdqu16(Xbyak::Ymm(bRegIdx + bFullReg) | mask_regs[1] | T_z,
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
    for (iter_t i = 0; i < MR; i++) {
        for (iter_t j = 0; j < bFullReg; j++) {
            vmovdqu32(RegType(bRegIdx + j), ptr[regTmpCptr + j * RegBytes]);
            vpmulld(RegType(bRegIdx + j), RegType(bRegIdx + j),
                    RegType(betaRegIdx));
            vpaddd(RegType(cRegIdx + i * bReg + j),
                   RegType(cRegIdx + i * bReg + j), RegType(bRegIdx + j));
        }
        if (bMaskReg > 0) {
            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                vmovdqu32(RegType(bRegIdx + bFullReg) | mask_regs[1] | T_z,
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
    if ((params.alphaScalingType != dlp::kernel_frame::scalingType::one)
        && (params.alphaScalingType != dlp::kernel_frame::scalingType::zero)) {
        RETURN_IF_ERROR(scaleAlpha());
    }

    // Handle beta scaling
    if (params.betaScalingType != dlp::kernel_frame::scalingType::zero) {
        RETURN_IF_ERROR(scaleBeta());
    }

    // Post-Ops dispatcher
    mov(regTmp1,
        ptr[stackPtr + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)
            + offsetof(dlp_gemm_post_op_attr, is_last_k)]);
    test(regTmp1, regTmp1);
    je(local_store_result, T_NEAR);

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

        // Convert S32 accumulators to F32 for post-ops compatibility
        for (iter_t i = 0; i < cReg; i++) {
            vcvtdq2ps(Zmm(cRegIdx + i), Zmm(cRegIdx + i));
        }

        accumulatorsAreF32 = true;

        VecPoolType vecPool;
        vecPool.setAccumulators(cRegIdx, cReg);
        RETURN_IF_ERROR(vecPool.init(this, Traits::regBytes));

        // U8S8 GEMM preserves 2 masks when fringe: one for S32/F32 data, one
        // for comparison
        int          maskCount = useMask ? 2 : 1;
        MaskPoolType maskPool;
        maskPool.addPreserve(utils::MASK_START_IDX, maskCount);
        RETURN_IF_ERROR(maskPool.init(this, utils::maskSaveWidth<KType>(),
                                      Traits::reservedMaskBits));

        int maskOffset =
            useMask
                ? static_cast<int>(offsetof(dlp::kernels::gemmParams, maskS32))
                : -1;

        RETURN_IF_ERROR(kernelOpsHandlerPtr->generateKernelOps(
            params.kernelOps, stackPtr, dlp::jit::jitAlgoType::gemm, params.MR,
            params.NR, params.useMask, params.numMaskRegs, cRegIdx, cReg,
            vecPool, maskPool, maskOffset));

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
    RETURN_IF_ERROR(utils::jitGeneratorUtils::checkValidGemmParams(params));

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

        // Preserve callee-saved xmm6-15 across the kernel call on Windows x64
        // (no-op on Linux/SysV). See utils::winAbiVectorGuard.
        utils::winAbiVectorGuard winAbiGuard(this);

        initializeParameters(params.mLoop);

        // Generate M-loop if needed, otherwise just IR loop
        if (params.mLoop) {
            RETURN_IF_ERROR(generateMLoop(params));
        } else {
            RETURN_IF_ERROR(generateIrLoop(params));
        }
    } // StackFrame destructor inserts 'ret' here

    return dlp::jit::jitGeneratorError::success;
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
            + offsetof(dlp_gemm_post_op_attr, post_op_c_i)],
        regTmp2);
    // Decrement M counter
    sub(regMiter, 1);
    jne(".MLOOP_START", T_NEAR);

    L(".MLOOP_END");

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
jitU8S8VNNI_GEMM<KType>::broadcastAVNNIwithB(bool isVNNIrem)
{

    for (iter_t i = 0; i < MR; i++) {
        if (isVNNIrem) {

            // Masked load with zero-extension: load kLeft bytes, zero the rest
            vmovdqu8(Ymm(aRegIdx) | mask_regs[0] | T_z, ptr[regTmpAptr]);

            // Broadcast the loaded 4-byte pattern to all lanes
            vpbroadcastd(RegType(aRegIdx), Xmm(aRegIdx));
        } else {
            // Load 4 bytes from A (VNNI group) and broadcast to all lanes
            vpbroadcastd(RegType(aRegIdx), ptr[regTmpAptr]);
        }

        for (iter_t j = 0; j < bReg; j++) {
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
    for (iter_t p = 0; p < unroll; p++) {
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

    // Advance C by MR rows. Decompose MR into encodable power-of-two scales;
    // x86 addressing permits only one base and one scaled index per LEA.
    int m_val       = MR;
    int power2scale = 1;
    while (m_val > 0) {
        if (m_val & 1) {
            // lea() only supports scale factors of 1, 2, 4, and 8.
            // For larger powers of 2, shift a temporary register and add.
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

    return dlp::jit::jitGeneratorError::success;
}

} // namespace amdzen::gen

// Explicit template instantiations
template class amdzen::gen::jitU8S8VNNI_GEMM<
    amdzen::utils::kernelInstrType::avx512_zmm_32_reg>;
