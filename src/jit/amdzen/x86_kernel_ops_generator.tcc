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
#include "kernel_ops.hh"
#include "jit_generator_utils.hh"
#include "classic/dlp_macros.h"

#include <memory>
#include <vector>
#include <cstdio>  // For debug prints

#ifndef DEBUG_LAYOUT_DISPATCH
#define DEBUG_LAYOUT_DISPATCH 0
#endif

#if DEBUG_LAYOUT_DISPATCH
#define LAYOUT_DEBUG_PRINT(fmt, ...) \
    fprintf(stderr, "[LAYOUT_DEBUG:x86_kernel_ops] " fmt "\n", ##__VA_ARGS__)
#else
#define LAYOUT_DEBUG_PRINT(fmt, ...) ((void)0)
#endif

namespace amdzen::x86gen {

using namespace dlp::kernel_frame;
using namespace dlp::jit;
using namespace Xbyak;

// ═══════════════════════════════════════════════════════════════════════════
// Base Class (Orchestrator) Implementation
// ═══════════════════════════════════════════════════════════════════════════

template<utils::kernelInstrType KType>
kernelOpsGeneratorX86<KType>::kernelOpsGeneratorX86(Xbyak::CodeGenerator* jitGen)
    : jit(jitGen)
    , regkernelOpsList(jitGen->rdx)
    , regkernelOpsAttr(jitGen->r8)
    , regTmp1(jitGen->r9)
    , regTmp2(jitGen->r10)
    , regTmp3(jitGen->r11)
    , regTmp4(jitGen->rax)
    , regTmp5(jitGen->rbx)
    , regTmp6(jitGen->rdi)
    , regTmp7(jitGen->rsi)
    , regcsC(jitGen->r12)
    , regTmp4Half(jitGen->eax)
    , regTmp5Half(jitGen->ebx)
    , vecPool(nullptr)
    , maskPool(nullptr)
    , maskOffset(-1)
    , numFringeMasks(0)
    , tableOwner(this)
{}


template<utils::kernelInstrType KType>
kernelOpsGeneratorX86<KType>::kernelOpsGeneratorX86(kernelOpsGeneratorX86& base)
    : jit(base.jit)
    , regkernelOpsList(base.regkernelOpsList)
    , regkernelOpsAttr(base.regkernelOpsAttr)
    , regTmp1(base.regTmp1)
    , regTmp2(base.regTmp2)
    , regTmp3(base.regTmp3)
    , regTmp4(base.regTmp4)
    , regTmp5(base.regTmp5)
    , regTmp6(base.regTmp6)
    , regTmp7(base.regTmp7)
    , regcsC(base.regcsC)
    , regTmp4Half(base.regTmp4Half)
    , regTmp5Half(base.regTmp5Half)
    , vecPool(base.vecPool)
    , maskPool(base.maskPool)
    , maskOffset(base.maskOffset)
    , numFringeMasks(base.numFringeMasks)
    , ymmMask(base.ymmMask)
    , algoType(base.algoType)
    , MR(base.MR)
    , NR(base.NR)
    , useMask(base.useMask)
    , numMaskRegs(base.numMaskRegs)
    , cRegStartIdx(base.cRegStartIdx)
    , cRegCount(base.cRegCount)
    , numRegsPerRow(base.numRegsPerRow)
    , numFullRegsPerRow(base.numFullRegsPerRow)
    , numMaskRegsPerRow(base.numMaskRegsPerRow)
    , tableOwner(base.tableOwner) // shared table labels
{
    for (int i = 0; i < numFringeMasks; i++) {
        fringeMaskIdx[i] = base.fringeMaskIdx[i];
    }
}


template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::generateKernelOps(
    std::vector<kernelOpsMetaData>& kernelOps,
    const Xbyak::Reg64& postOpsArgWrapperPtrReg,
    jitAlgoType algoType,
    int MRarg, int NRarg,
    bool useMaskarg, int numMaskRegsarg,
    int cRegStartIdxarg, int cRegCountarg,
    VecPoolType& vecPool,
    MaskPoolType& maskPool,
    int maskOffset)
{
    this->vecPool  = &vecPool;
    this->maskPool = &maskPool;

    this->algoType = algoType;
    this->maskOffset = maskOffset;
    RETURN_IF_ERROR(setPostOpsContext(MRarg, NRarg, useMaskarg,
                                    numMaskRegsarg, cRegStartIdxarg, cRegCountarg));

    requiredTables = TABLE_NONE;

    utils::registerGuard<Reg64> gprGuard{ jit };
    gprGuard.saveRegister(regkernelOpsList);
    gprGuard.saveRegister(regkernelOpsAttr);
    gprGuard.saveRegister(regcsC);
    gprGuard.saveRegister(regTmp1);
    gprGuard.saveRegister(regTmp2);
    gprGuard.saveRegister(regTmp3);
    gprGuard.saveRegister(regTmp4);
    gprGuard.saveRegister(regTmp5);
    gprGuard.saveRegister(regTmp6);
    gprGuard.saveRegister(regTmp7);

    utils::registerGuard<RegType> vecMaskGuard;
    std::vector<utils::registerGuard<Xbyak::Opmask>> fringeMaskGuards;
    RETURN_IF_ERROR(initializeFringeMasks(
        useMaskarg ? numMaskRegsarg : 0, maskOffset,
        postOpsArgWrapperPtrReg, fringeMaskGuards, vecMaskGuard));

    if (useMaskarg && numFringeMasks < numMaskRegsarg) {
        return jitGeneratorError::notSupported;
    }

    // 6. Load metadata pointers based on algorithm type.
    if (algoType == jitAlgoType::gemv_m1) {
        jit->mov(regkernelOpsList,
                  jit->ptr[postOpsArgWrapperPtrReg
                           + offsetof(dlp::kernels::gemvM1Params, kernelOpsList)]);
        jit->mov(regcsC,
                  jit->ptr[postOpsArgWrapperPtrReg
                           + offsetof(dlp::kernels::gemvM1Params, csY)]);
        jit->lea(regkernelOpsAttr,
                  jit->ptr[postOpsArgWrapperPtrReg
                           + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)]);
    } else if (algoType == jitAlgoType::gemv_n1) {
        jit->mov(regkernelOpsList,
                  jit->ptr[postOpsArgWrapperPtrReg
                           + offsetof(dlp::kernels::gemvN1Params, kernelOpsList)]);
        jit->mov(regcsC,
                  jit->ptr[postOpsArgWrapperPtrReg
                           + offsetof(dlp::kernels::gemvN1Params, csC)]);
        jit->lea(regkernelOpsAttr,
                  jit->ptr[postOpsArgWrapperPtrReg
                           + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)]);
    } else {
        // GEMM case
        jit->mov(regkernelOpsList,
                  jit->ptr[postOpsArgWrapperPtrReg
                           + offsetof(dlp::kernels::gemmParams, kernelOpsList)]);
        jit->mov(regcsC,
                  jit->ptr[postOpsArgWrapperPtrReg
                           + offsetof(dlp::kernels::gemmParams, csC)]);
        jit->lea(regkernelOpsAttr,
                  jit->ptr[postOpsArgWrapperPtrReg
                           + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)]);
    }

    // 7. Generate post-ops (creates impl on-demand, then destroys it)
    for (auto& op : kernelOps) {
        RETURN_IF_ERROR(generateKernelOp(*this, op));
        advancePostOpsPtr();
    }

    embedKernelOpsAttributes();

    return jitGeneratorError::success;
}


template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::setPostOpsContext(
    int MRval, int NRval, bool useMaskval,
    int numMaskRegsval, int cRegStartIdxval, int cRegCountval)
{
    MR = MRval;
    NR = NRval;
    useMask = useMaskval;
    numMaskRegs = numMaskRegsval;
    cRegStartIdx = cRegStartIdxval;
    cRegCount = cRegCountval;

    int numElemsPerReg = RegBytes / sizeof(float);
    numFullRegsPerRow = (algoType == jitAlgoType::gemv_n1)
                             ? (MR / numElemsPerReg)
                             : (NR / numElemsPerReg);
    numMaskRegsPerRow = (useMask) ? numMaskRegs : 0;
    numRegsPerRow = numFullRegsPerRow + numMaskRegsPerRow;

    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        // AVX2: Total registers = accumulators + scratch + mask
        int totalRegsNeeded = cRegCount + numRegsPerRow + (useMask ? 1 : 0);
        if (totalRegsNeeded > RegisterCount)
            return jitGeneratorError::badKernelInfo;
    } else {
        if (cRegCount + numRegsPerRow >= RegisterCount)
            return jitGeneratorError::badKernelInfo;

        bool configValid = false;
        switch (algoType) {
            case jitAlgoType::gemv_m1:
                configValid = (cRegCount == (NR + numElemsPerReg - 1) / numElemsPerReg);
                break;
            case jitAlgoType::gemv_n1:
                configValid = (cRegCount == (MR + numElemsPerReg - 1) / numElemsPerReg);
                break;
            case jitAlgoType::gemm:
                configValid = (MR * numRegsPerRow == cRegCount);
                break;
            default:
                return jitGeneratorError::badKernelInfo;
        }
        if (!configValid)
            return jitGeneratorError::badKernelInfo;
    }

    return jitGeneratorError::success;
}


template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::initializeFringeMasks(
    int numMasks, int maskOffset,
    const Xbyak::Reg64& stackPtr,
    std::vector<utils::registerGuard<Xbyak::Opmask>>& guards,
    utils::registerGuard<RegType>& vecMaskGuard)
{
    numFringeMasks = 0;

    bool hasMaskSource = (maskOffset >= 0) || utils::isMaskImmediate(maskOffset);

    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        if (numMasks > 0 && hasMaskSource) {
            RETURN_IF_ERROR(vecPool->acquireGuard(vecMaskGuard));
            ymmMask = Xbyak::Ymm(vecMaskGuard.idx());

            if (utils::isMaskImmediate(maskOffset)) {
                int imm = utils::decodeMaskImmediate(maskOffset);
                utils::registerGuard<RegType> tmpGuard;
                RETURN_IF_ERROR(vecPool->acquireGuard(tmpGuard));
                Xbyak::Ymm tmpYmm(tmpGuard.idx());
                jit->vxorps(tmpYmm, tmpYmm, tmpYmm);
                jit->vcmpeqps(tmpYmm, tmpYmm, tmpYmm);
                jit->vxorps(ymmMask, ymmMask, ymmMask);
                jit->vblendps(ymmMask, ymmMask, tmpYmm,
                              static_cast<uint8_t>(imm));
            } else {
                jit->vmovdqu(ymmMask, jit->ptr[stackPtr + maskOffset]);
            }
            numFringeMasks = numMasks;
        }
    } else {
        if (numMasks > 0 && hasMaskSource && maskPool) {
            for (int i = 0; i < numMasks; i++) {
                auto guard = maskPool->acquireGuard();
                if (!guard.isValid()) break;

                Xbyak::Opmask kReg(guard.idx());
                if (utils::isMaskImmediate(maskOffset)) {
                    int imm = utils::decodeMaskImmediate(maskOffset);
                    jit->mov(jit->r9d, imm);
                    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                        jit->kmovw(kReg, jit->r9d);
                    } else {
                        jit->kmovb(kReg, jit->r9d);
                    }
                } else if (maskOffset >= 0) {
                    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                        jit->kmovw(kReg,
                            jit->ptr[stackPtr + maskOffset + i * sizeof(uint16_t)]);
                    } else if constexpr (KType == utils::kernelInstrType::avx512_ymm_32_reg) {
                        jit->kmovb(kReg,
                            jit->ptr[stackPtr + maskOffset + i * sizeof(uint8_t)]);
                    }
                }

                fringeMaskIdx[i] = static_cast<uint8_t>(guard.idx());
                guards.push_back(std::move(guard));
                numFringeMasks++;
            }
        }
    }

    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::embedKernelOpsAttributes()
{
    if (tablesEmbedded) return;

    if (requiredTables == TABLE_NONE) return;

    jit->jmp(tableEnd, jit->T_NEAR);

    if (requiredTables & TABLE_GELU) {
        { size_t r = jit->getSize() % 64; if (r) jit->nop(64 - r); }
        jit->L(geluTable);
        jit->db(reinterpret_cast<const uint8_t*>(gen::tables::gelu_consts),
                 sizeof(gen::tables::gelu_consts));
        jit->db(reinterpret_cast<const uint8_t*>(gen::tables::gelu_macros),
                 sizeof(gen::tables::gelu_macros));
    }

    if (requiredTables & TABLE_EXP) {
        { size_t r = jit->getSize() % 64; if (r) jit->nop(64 - r); }
        jit->L(expTable);
        jit->db(reinterpret_cast<const uint8_t*>(gen::tables::dlp_gemm_exp),
                 sizeof(gen::tables::dlp_gemm_exp));
    }

    if (requiredTables & TABLE_ERF) {
        { size_t r = jit->getSize() % 64; if (r) jit->nop(64 - r); }
        jit->L(erfTable);
        jit->db(reinterpret_cast<const uint8_t*>(gen::tables::erf_consts),
                 sizeof(gen::tables::erf_consts));
        jit->db(reinterpret_cast<const uint8_t*>(gen::tables::dlp_gemm_erf),
                 sizeof(gen::tables::dlp_gemm_erf));
        jit->db(reinterpret_cast<const uint8_t*>(gen::tables::erf_f32_coeffs_hex),
                 sizeof(gen::tables::erf_f32_coeffs_hex));
        jit->db(reinterpret_cast<const uint8_t*>(gen::tables::erf_f32_constants_hex),
                 sizeof(gen::tables::erf_f32_constants_hex));
    }

    jit->L(tableEnd);
    tablesEmbedded = true;
}


template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::advancePostOpsPtr()
{
    jit->mov(regkernelOpsList,
              jit->ptr[regkernelOpsList + offsetof(dlp_gemm_post_op, next)]);
}


// ═══════════════════════════════════════════════════════════════════════════
// Common Helper Methods
// ═══════════════════════════════════════════════════════════════════════════

template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::loadVectorFloat32(
    const Xbyak::Address& addr, const RegType& dest,
    bool useMaskOp, const Xbyak::Opmask& mask)
{
    if (useMaskOp) {
        if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            jit->vmaskmovps(dest, ymmMask, addr);
        } else {
            jit->vmovups(dest | mask | jit->T_z, addr);
        }
    } else {
        jit->vmovups(dest, addr);
    }
}


template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::loadVectorBF16andConvertToF32(
    const Xbyak::Address& addr, const RegType& dest,
    bool useMaskOp, const Xbyak::Opmask& mask)
{
    utils::registerGuard<RegType> tmpGuard;
    RETURN_IF_ERROR(vecPool->acquireGuard(tmpGuard));
    int tmpRegIdx = tmpGuard.idx();

    if (useMaskOp) {
        if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            jit->lea(regTmp5, addr);
            jit->vxorps(Xbyak::Xmm(tmpRegIdx), Xbyak::Xmm(tmpRegIdx), Xbyak::Xmm(tmpRegIdx));

            Xbyak::Reg32 regMaskBits = regTmp4Half;
            jit->vmovmskps(regMaskBits, ymmMask);

            for (int lane = 0; lane < 8; lane++) {
                jit->inLocalLabel();
                Xbyak::Label skipLane;
                jit->bt(regMaskBits, lane);
                jit->jnc(skipLane);
                jit->pinsrw(halfRegType(tmpRegIdx),
                            jit->word[regTmp5 + lane * sizeof(bfloat16)],
                            lane);
                jit->L(skipLane);
                jit->outLocalLabel();
            }
        } else {
            jit->vmovdqu16(halfRegType(tmpRegIdx) | mask | jit->T_z, addr);
        }
    } else {
        if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            jit->vmovdqu(halfRegType(tmpRegIdx), addr);
        } else {
            jit->vmovdqu16(halfRegType(tmpRegIdx), addr);
        }
    }

    jit->vpmovsxwd(dest, halfRegType(tmpRegIdx));
    jit->vpslld(dest, dest, 16);
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::loadVectorFP16andConvertToF32(
    const Xbyak::Address& addr, const RegType& dest,
    bool useMaskOp, const Xbyak::Opmask& mask)
{
    if (useMaskOp) {
        if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            return jitGeneratorError::notSupported;
        } else {
            jit->vmovdqu16(halfRegType(dest.getIdx()) | mask | jit->T_z, addr);
        }
    } else {
        if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
           return jitGeneratorError::notSupported;
        } else {
            jit->vmovdqu16(halfRegType(dest.getIdx()), addr);
        }
    }
    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        jit->vcvtph2ps(dest, halfRegType(dest.getIdx()));
    } else {
        return jitGeneratorError::notSupported;
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::loadVectorInt8andConvertToF32(
    const Xbyak::Address& addr, const RegType& dest,
    bool useMaskOp, const Xbyak::Opmask& mask)
{
    utils::registerGuard<RegType> tmpGuard;
    RETURN_IF_ERROR(vecPool->acquireGuard(tmpGuard));
    int tmpRegIdx = tmpGuard.idx();

    if (useMaskOp) {
        if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            jit->lea(regTmp5, addr);
            jit->vpxor(Xbyak::Xmm(tmpRegIdx), Xbyak::Xmm(tmpRegIdx), Xbyak::Xmm(tmpRegIdx));

            Xbyak::Reg32 regMaskBits = regTmp4Half;
            jit->vmovmskps(regMaskBits, ymmMask);

            for (int lane = 0; lane < 8; lane++) {
                jit->inLocalLabel();
                Xbyak::Label skipLane;
                jit->bt(regMaskBits, lane);
                jit->jnc(skipLane);
                jit->pinsrb(Xbyak::Xmm(tmpRegIdx),
                            jit->byte[regTmp5 + lane * sizeof(int8_t)],
                            lane);
                jit->L(skipLane);
                jit->outLocalLabel();
            }
        } else {
            jit->vmovdqu8(Xbyak::Xmm(tmpRegIdx) | mask | jit->T_z, addr);
        }
    } else {
        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            jit->vmovdqu8(Xbyak::Xmm(tmpRegIdx), addr);
        } else {
            jit->vmovq(Xbyak::Xmm(tmpRegIdx), addr);
        }
    }

    jit->vpmovsxbd(dest, Xbyak::Xmm(tmpRegIdx));
    jit->vcvtdq2ps(dest, dest);
    return jitGeneratorError::success;
}


template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::loadVectorUInt8andConvertToF32(
    const Xbyak::Address& addr, const RegType& dest,
    bool useMaskOp, const Xbyak::Opmask& mask)
{
    utils::registerGuard<RegType> tmpGuard;
    RETURN_IF_ERROR(vecPool->acquireGuard(tmpGuard));
    int tmpRegIdx = tmpGuard.idx();

    if (useMaskOp) {
        if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
            jit->lea(regTmp5, addr);
            jit->vpxor(Xbyak::Xmm(tmpRegIdx), Xbyak::Xmm(tmpRegIdx), Xbyak::Xmm(tmpRegIdx));

            Xbyak::Reg32 regMaskBits = regTmp4Half;
            jit->vmovmskps(regMaskBits, ymmMask);

            for (int lane = 0; lane < 8; lane++) {
                jit->inLocalLabel();
                Xbyak::Label skipLane;
                jit->bt(regMaskBits, lane);
                jit->jnc(skipLane);
                jit->pinsrb(Xbyak::Xmm(tmpRegIdx),
                            jit->byte[regTmp5 + lane * sizeof(uint8_t)],
                            lane);
                jit->L(skipLane);
                jit->outLocalLabel();
            }
        } else {
            jit->vmovdqu8(Xbyak::Xmm(tmpRegIdx) | mask | jit->T_z, addr);
        }
    } else {
        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            jit->vmovdqu8(Xbyak::Xmm(tmpRegIdx), addr);
        } else {
            jit->vmovq(Xbyak::Xmm(tmpRegIdx), addr);
        }
    }

    jit->vpmovzxbd(dest, Xbyak::Xmm(tmpRegIdx));
    jit->vcvtdq2ps(dest, dest);
    return jitGeneratorError::success;
}


template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::broadcastScalarFloat32(
    const Xbyak::Address& addr, const RegType& dest)
{
    jit->vbroadcastss(dest, addr);
}


template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::broadcastScalarBF16andConvertToF32(
    const Xbyak::Address& addr, const RegType& dest)
{
    utils::registerGuard<RegType> tmpGuard;
    RETURN_IF_ERROR(vecPool->acquireGuard(tmpGuard));
    int tmpRegIdx = tmpGuard.idx();

    jit->vpbroadcastw(RegType(tmpRegIdx), addr);
    jit->vpmovsxwd(dest, halfRegType(tmpRegIdx));
    jit->vpslld(dest, dest, 16);
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::broadcastScalarFP16andConvertToF32(
    const Xbyak::Address& addr, const RegType& dest)
{
    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        utils::registerGuard<RegType> tmpGuard;
        RETURN_IF_ERROR(vecPool->acquireGuard(tmpGuard));
        int tmpRegIdx = tmpGuard.idx();
        jit->vpbroadcastw(RegType(tmpRegIdx), addr);
        jit->vcvtph2ps(dest, halfRegType(tmpRegIdx));
        return jitGeneratorError::success;
    } else {
        return jitGeneratorError::notSupported;
    }
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::broadcastScalarInt8andConvertToF32(
    const Xbyak::Address& addr, const RegType& dest)
{
    utils::registerGuard<RegType> tmpGuard;
    RETURN_IF_ERROR(vecPool->acquireGuard(tmpGuard));
    int tmpRegIdx = tmpGuard.idx();

    jit->vpbroadcastb(RegType(tmpRegIdx), addr);
    jit->vpmovsxbd(dest, halfRegType(tmpRegIdx));
    jit->vcvtdq2ps(dest, dest);
    return jitGeneratorError::success;
}


template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::broadcastScalarUInt8andConvertToF32(
    const Xbyak::Address& addr, const RegType& dest)
{
    utils::registerGuard<RegType> tmpGuard;
    RETURN_IF_ERROR(vecPool->acquireGuard(tmpGuard));
    int tmpRegIdx = tmpGuard.idx();

    jit->vpbroadcastb(RegType(tmpRegIdx), addr);
    jit->vpmovzxbd(dest, halfRegType(tmpRegIdx));
    jit->vcvtdq2ps(dest, dest);
    return jitGeneratorError::success;
}


// ═══════════════════════════════════════════════════════════════════════════
// Op: MatOps Implementation(MatrixAdd/MatrixMul)
// Matrix Add -- C[i][j] += sf * aux[i][j]
// ═══════════════════════════════════════════════════════════════════════════

template<utils::kernelInstrType KType>
jitGeneratorError
MatOps<KType>::generateImpl(kernelOpsMetaData& op)
{
    bool hasSF = (op.scalarScaleFactorRequired || op.vectorScaleFactorRequired);
    DataType sfDtype = hasSF ? op.scaleFactorDt : DataType::f32;

    if (op.paramStorageDt == DataType::invalid ||
        (hasSF && sfDtype == DataType::invalid))
        return jitGeneratorError::notSupported;

    matOpType opType = (op.type == kernelOps::matAdd)
                       ? matOpType::matOpAdd
                       : matOpType::matOpMul;

    matOpScaleType sclType = matOpScaleType::scalar;
    if (hasSF && !op.scalarScaleFactorRequired) {
        sclType = (op.cMatFormat == storageFormat::rowMajor)
                  ? matOpScaleType::rowVector
                  : matOpScaleType::columnVector;
    }

    auto* jit = this->jit;
    const int sfElemSize = opBase::getElementSize(sfDtype);
    const int matElemSize = opBase::getElementSize(op.paramStorageDt);

    utils::registerGuard<RegType> matRegGuard;
    RETURN_IF_ERROR(this->vecPool->acquireGuard(matRegGuard));
    int matRegIdx = matRegGuard.idx();

    utils::registerGuard<RegType> sfRegGuard;
    int sfRegIdx = -1;
    if (hasSF) {
        RETURN_IF_ERROR(this->vecPool->acquireGuard(sfRegGuard));
        sfRegIdx = sfRegGuard.idx();
        jit->mov(this->regTmp1,
                 jit->ptr[this->regkernelOpsList + offsetof(dlp_gemm_post_op, scale_factor)]);
    }

    jit->mov(this->regTmp7, jit->ptr[this->regkernelOpsAttr
        + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
    jit->mov(this->regTmp6, jit->ptr[this->regkernelOpsAttr
        + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);

    if (hasSF) {
        if (sclType == matOpScaleType::scalar) {
            RETURN_IF_ERROR(this->broadcastScalar(sfDtype, jit->ptr[this->regTmp1], RegType(sfRegIdx)));
        } else if (sclType == matOpScaleType::columnVector) {
            jit->lea(this->regTmp1, jit->ptr[this->regTmp1 + this->regTmp6 * sfElemSize]);
        } else {
            jit->lea(this->regTmp1, jit->ptr[this->regTmp1 + this->regTmp7 * sfElemSize]);
        }
    }

    jit->mov(this->regTmp2, jit->ptr[this->regkernelOpsList + offsetof(dlp_gemm_post_op, op_args1)]);

    if (this->isGEMVN1()) {
        return gemvN1Path(opType, sclType, hasSF, sfDtype, op.paramStorageDt, matRegIdx, sfRegIdx);
    }

    jit->mov(this->regTmp3, jit->ptr[this->regkernelOpsList + offsetof(dlp_gemm_post_op, op_args3)]);
    jit->mov(this->regTmp3, jit->ptr[this->regTmp3]);
    jit->lea(this->regTmp3, jit->ptr[this->regTmp3 * matElemSize]);

    jit->inLocalLabel();

    jit->cmp(this->regcsC, 1);
    jit->je(".rowMajor", jit->T_NEAR);

    RETURN_IF_ERROR(colMajorPath(opType, sclType, hasSF, sfDtype, op.paramStorageDt, matRegIdx, sfRegIdx));
    jit->jmp(".end", jit->T_NEAR);

    jit->L(".rowMajor");
    RETURN_IF_ERROR(rowMajorPath(opType, sclType, hasSF, sfDtype, op.paramStorageDt, matRegIdx, sfRegIdx));

    jit->L(".end");
    jit->outLocalLabel();

    return jitGeneratorError::success;
}

// ─────────────────────────────────────────────────────────────────────────────
// MatOps::applyMatOp - Unified operation application (reduces duplication)
// ─────────────────────────────────────────────────────────────────────────────
template<utils::kernelInstrType KType>
inline void MatOps<KType>::applyMatOp(
    matOpType opType, bool hasSF, bool isFringe,
    int accumIdx, int matRegIdx, int sfRegIdx,
    const Xbyak::Opmask& fringeMask)
{
    auto* jit = this->jit;

    // AVX2 fringe: blend-based masking
    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        if (isFringe) {
            if (hasSF) {
                jit->vmulps(RegType(matRegIdx), RegType(matRegIdx), RegType(sfRegIdx));
            }
            if (opType == matOpType::matOpAdd) {
                jit->vandps(RegType(matRegIdx), RegType(matRegIdx), this->ymmMask);
                jit->vaddps(RegType(accumIdx), RegType(accumIdx), RegType(matRegIdx));
            } else {
                jit->vmulps(RegType(matRegIdx), RegType(accumIdx), RegType(matRegIdx));
                jit->vblendvps(RegType(accumIdx), RegType(accumIdx),
                              RegType(matRegIdx), this->ymmMask);
            }
            return;
        }
    }

    // Full register or AVX-512 (with/without k-mask)
    if (hasSF && opType == matOpType::matOpAdd) {
        // FMA optimization: accum = accum + mat * sf
        if (isFringe) {
            jit->vfmadd231ps(RegType(accumIdx) | fringeMask,
                            RegType(matRegIdx), RegType(sfRegIdx));
        } else {
            jit->vfmadd231ps(RegType(accumIdx), RegType(matRegIdx), RegType(sfRegIdx));
        }
    } else if (hasSF) {
        jit->vmulps(RegType(matRegIdx), RegType(matRegIdx), RegType(sfRegIdx));
        if (isFringe) {
            jit->vmulps(RegType(accumIdx) | fringeMask, RegType(accumIdx), RegType(matRegIdx));
        } else {
            jit->vmulps(RegType(accumIdx), RegType(accumIdx), RegType(matRegIdx));
        }
    } else if (opType == matOpType::matOpAdd) {
        if (isFringe) {
            jit->vaddps(RegType(accumIdx) | fringeMask, RegType(accumIdx), RegType(matRegIdx));
        } else {
            jit->vaddps(RegType(accumIdx), RegType(accumIdx), RegType(matRegIdx));
        }
    } else {
        if (isFringe) {
            jit->vmulps(RegType(accumIdx) | fringeMask, RegType(accumIdx), RegType(matRegIdx));
        } else {
            jit->vmulps(RegType(accumIdx), RegType(accumIdx), RegType(matRegIdx));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MatOps::rowMajorPath - All KTypes (unified)
// ─────────────────────────────────────────────────────────────────────────────
template<utils::kernelInstrType KType>
jitGeneratorError MatOps<KType>::rowMajorPath(
    matOpType opType, matOpScaleType sclType, bool hasSF,
    DataType sfDtype, DataType matOpDtype,
    int matRegIdx, int sfRegIdx)
{
    auto* jit = this->jit;
    const int sfElemSize = opBase::getElementSize(sfDtype);
    const int matElemSize = opBase::getElementSize(matOpDtype);
    const int sfLoadBytes = opBase::getLoadBytes(sfDtype);
    const int matLoadBytes = opBase::getLoadBytes(matOpDtype);

    jit->lea(this->regTmp7, jit->ptr[this->regTmp7 * matElemSize]);
    jit->imul(this->regTmp6, this->regTmp3);
    jit->add(this->regTmp7, this->regTmp6);
    jit->add(this->regTmp2, this->regTmp7);

    for (int i = 0; i < this->MR; i++) {
        if (hasSF && sclType == matOpScaleType::columnVector) {
            RETURN_IF_ERROR(this->broadcastScalar(sfDtype,
                jit->ptr[this->regTmp1 + i * sfElemSize], RegType(sfRegIdx)));
        }

        for (int j = 0; j < this->numRegsPerRow; j++) {
            bool isFringe = (j >= this->numFullRegsPerRow);
            int maskIdx = isFringe ? (j - this->numFullRegsPerRow) : 0;
            int accumIdx = this->cRegStartIdx + (i * this->numRegsPerRow) + j;
            Xbyak::Opmask fringeMask = this->getFringeMask(maskIdx);

            if (hasSF && sclType == matOpScaleType::rowVector) {
                RETURN_IF_ERROR(this->loadVector(sfDtype,
                    jit->ptr[this->regTmp1 + j * sfLoadBytes],
                    RegType(sfRegIdx), isFringe, fringeMask));
            }

            RETURN_IF_ERROR(this->loadVector(matOpDtype, jit->ptr[this->regTmp2 + j * matLoadBytes],
                            RegType(matRegIdx), isFringe, fringeMask));

            applyMatOp(opType, hasSF, isFringe, accumIdx, matRegIdx, sfRegIdx, fringeMask);
        }

        jit->add(this->regTmp2, this->regTmp3);
    }
    return jitGeneratorError::success;
}

// ─────────────────────────────────────────────────────────────────────────────
// MatOps::colMajorPath - AVX-512 (gather-based)
// ─────────────────────────────────────────────────────────────────────────────
template<utils::kernelInstrType KType>
jitGeneratorError MatOps<KType>::colMajorPath(
    matOpType opType, matOpScaleType sclType, bool hasSF,
    DataType sfDtype, DataType matOpDtype,
    int matRegIdx, int sfRegIdx)
{
    auto* jit = this->jit;
    const int sfElemSize = opBase::getElementSize(sfDtype);
    const int matElemSize = opBase::getElementSize(matOpDtype);
    const int sfLoadBytes = opBase::getLoadBytes(sfDtype);

    if (this->maskPool) {
        gatherMask0 = this->maskPool->acquireGuard();
        gatherMask1 = this->maskPool->acquireGuard();
    }
    if (!gatherMask0.isValid() || !gatherMask1.isValid())
        return jitGeneratorError::notSupported;

    Xbyak::Label offsets_label;
    int64_t offsets[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    jit->jmp(".offset_end", jit->T_NEAR);
    { size_t r = jit->getSize() % 64; if (r) jit->nop(64 - r); }
    jit->L(offsets_label);
    jit->db(reinterpret_cast<uint8_t*>(&offsets), sizeof(offsets));
    jit->L(".offset_end");

    utils::registerGuard<RegType> off1Guard;
    RETURN_IF_ERROR(this->vecPool->acquireGuard(off1Guard));
    utils::registerGuard<RegType> off2Guard;
    RETURN_IF_ERROR(this->vecPool->acquireGuard(off2Guard));
    utils::registerGuard<RegType> scrReg2Guard;
    RETURN_IF_ERROR(this->vecPool->acquireGuard(scrReg2Guard));

    int off1 = off1Guard.idx();
    int off2 = off2Guard.idx();
    int scrReg2 = scrReg2Guard.idx();

    jit->vmovdqu32(RegType(off1), jit->ptr[jit->rip + offsets_label]);
    jit->vmovdqu32(RegType(off2), jit->ptr[jit->rip + offsets_label + opBase::RegBytes]);
    jit->vpbroadcastq(RegType(scrReg2), this->regTmp3);
    jit->vpmullq(RegType(off1), RegType(off1), RegType(scrReg2));
    jit->vpmullq(RegType(off2), RegType(off2), RegType(scrReg2));

    // Column-major address: mat + c_j * ldm + c_i * elemSize
    jit->lea(this->regTmp6, jit->ptr[this->regTmp6 * matElemSize]);
    jit->imul(this->regTmp7, this->regTmp3);
    jit->add(this->regTmp7, this->regTmp6);
    jit->add(this->regTmp2, this->regTmp7);

    // x86 addressing only supports scales 1,2,4,8; use lea*8 then *2 for ZMM
    jit->lea(this->regTmp3, jit->ptr[this->regTmp3 * 8]);
    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        jit->lea(this->regTmp3, jit->ptr[this->regTmp3 * 2]);  // total: *16
    }

    utils::registerGuard<RegType> dw1Guard, dw2Guard;
    int dw1 = -1, dw2 = -1;
    bool isFloat = (matOpDtype == DataType::f32);
    if (!isFloat) {
        RETURN_IF_ERROR(this->vecPool->acquireGuard(dw1Guard));
        RETURN_IF_ERROR(this->vecPool->acquireGuard(dw2Guard));
        dw1 = dw1Guard.idx();
        dw2 = dw2Guard.idx();
    }

    for (int i = 0; i < this->MR; i++) {
        if (hasSF && sclType == matOpScaleType::columnVector) {
            RETURN_IF_ERROR(this->broadcastScalar(sfDtype,
                jit->ptr[this->regTmp1 + i * sfElemSize], RegType(sfRegIdx)));
        }

        jit->mov(this->regTmp4, this->regTmp2);

        for (int j = 0; j < this->numRegsPerRow; j++) {
            bool isFringe = (j >= this->numFullRegsPerRow);
            int maskIdx = isFringe ? (j - this->numFullRegsPerRow) : 0;
            int accumIdx = this->cRegStartIdx + (i * this->numRegsPerRow) + j;

            if (hasSF && sclType == matOpScaleType::rowVector) {
                RETURN_IF_ERROR(this->loadVector(sfDtype, jit->ptr[this->regTmp1 + j * sfLoadBytes],
                                RegType(sfRegIdx), false));
            }

            resetGatherMasks(isFringe, maskIdx);

            if (isFloat) {
                jit->vgatherqps(halfRegType(matRegIdx) | Xbyak::Opmask(gatherMask0.idx()),
                               jit->ptr[this->regTmp4 + RegType(off1) * 1]);
                jit->vgatherqps(halfRegType(scrReg2) | Xbyak::Opmask(gatherMask1.idx()),
                               jit->ptr[this->regTmp4 + RegType(off2) * 1]);
            } else {
                jit->vpmovqd(halfRegType(dw1), RegType(off1));
                jit->vpmovqd(halfRegType(dw2), RegType(off2));
                jit->vpgatherdd(halfRegType(matRegIdx) | Xbyak::Opmask(gatherMask0.idx()),
                               jit->ptr[this->regTmp4 + halfRegType(dw1) * 1]);
                jit->vpgatherdd(halfRegType(scrReg2) | Xbyak::Opmask(gatherMask1.idx()),
                               jit->ptr[this->regTmp4 + halfRegType(dw2) * 1]);
            }

            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                if (isFloat) {
                    jit->vinsertf32x8(RegType(matRegIdx), RegType(matRegIdx), halfRegType(scrReg2), 1);
                } else {
                    jit->vinserti32x8(RegType(matRegIdx), RegType(matRegIdx), halfRegType(scrReg2), 1);
                }
            } else {
                if (isFloat) {
                    jit->vinsertf32x4(RegType(matRegIdx), RegType(matRegIdx), halfRegType(scrReg2), 1);
                } else {
                    jit->vinserti32x4(RegType(matRegIdx), RegType(matRegIdx), halfRegType(scrReg2), 1);
                }
            }

            switch (matOpDtype) {
                case DataType::s8:
                    jit->vpslld(RegType(matRegIdx), RegType(matRegIdx), 24);
                    jit->vpsrad(RegType(matRegIdx), RegType(matRegIdx), 24);
                    jit->vcvtdq2ps(RegType(matRegIdx), RegType(matRegIdx));
                    break;
                case DataType::u8:
                    jit->vpslld(RegType(matRegIdx), RegType(matRegIdx), 24);
                    jit->vpsrld(RegType(matRegIdx), RegType(matRegIdx), 24);
                    jit->vcvtdq2ps(RegType(matRegIdx), RegType(matRegIdx));
                    break;
                case DataType::bf16:
                    jit->vpslld(RegType(matRegIdx), RegType(matRegIdx), 16);
                    break;
                case DataType::f16:
                    jit->vpmovdw(halfRegType(matRegIdx), RegType(matRegIdx));
                    jit->vcvtph2ps(RegType(matRegIdx), halfRegType(matRegIdx));
                    break;
                case DataType::s32:
                    jit->vcvtdq2ps(RegType(matRegIdx), RegType(matRegIdx));
                    break;
                default: break;
            }

            if (hasSF && opType == matOpType::matOpAdd) {
                jit->vfmadd231ps(RegType(accumIdx), RegType(matRegIdx), RegType(sfRegIdx));
            } else if (hasSF) {
                jit->vmulps(RegType(matRegIdx), RegType(matRegIdx), RegType(sfRegIdx));
                jit->vmulps(RegType(accumIdx), RegType(accumIdx), RegType(matRegIdx));
            } else if (opType == matOpType::matOpAdd) {
                jit->vaddps(RegType(accumIdx), RegType(accumIdx), RegType(matRegIdx));
            } else {
                jit->vmulps(RegType(accumIdx), RegType(accumIdx), RegType(matRegIdx));
            }

            jit->add(this->regTmp4, this->regTmp3);
        }

        jit->add(this->regTmp2, matElemSize);
    }

    gatherMask1 = utils::registerGuard<Xbyak::Opmask>();
    gatherMask0 = utils::registerGuard<Xbyak::Opmask>();
    return jitGeneratorError::success;
}

// ─────────────────────────────────────────────────────────────────────────────
// MatOps::colMajorPath - AVX2 Specialization (scalar loads, no gather)
// ─────────────────────────────────────────────────────────────────────────────
template<>
inline jitGeneratorError MatOps<utils::kernelInstrType::avx2_ymm_16_reg>::colMajorPath(
    matOpType opType, matOpScaleType sclType, bool hasSF,
    DataType sfDtype, DataType matOpDtype,
    int matRegIdx, int sfRegIdx)
{
    using Base = kernelOpsGeneratorX86<utils::kernelInstrType::avx2_ymm_16_reg>;
    using RegType = typename opBase::RegType;

    auto* jit = this->jit;

    const int sfElemSize = opBase::getElementSize(sfDtype);
    const int matElemSize = opBase::getElementSize(matOpDtype);
    const int sfLoadBytes = opBase::getLoadBytes(sfDtype);

    // Column-major address: mat + c_j * ldm + c_i * elemSize
    jit->lea(this->regTmp6, jit->ptr[this->regTmp6 * matElemSize]);
    jit->imul(this->regTmp7, this->regTmp3);
    jit->add(this->regTmp7, this->regTmp6);
    jit->add(this->regTmp2, this->regTmp7);

    utils::registerGuard<RegType> tmpRegGuard;
    RETURN_IF_ERROR(this->vecPool->acquireGuard(tmpRegGuard));
    int tmpRegIdx = tmpRegGuard.idx();

    constexpr int elemsPerReg = 8; // YMM = 8 floats

    for (int i = 0; i < this->MR; i++) {
        if (hasSF && sclType == matOpScaleType::columnVector) {
            RETURN_IF_ERROR(this->broadcastScalar(sfDtype,
                jit->ptr[this->regTmp1 + i * sfElemSize], RegType(sfRegIdx)));
        }

        jit->mov(this->regTmp4, this->regTmp2);

        for (int j = 0; j < this->numRegsPerRow; j++) {
            bool isFringe = (j >= this->numFullRegsPerRow);
            int accumIdx = this->cRegStartIdx + (i * this->numRegsPerRow) + j;
            int numElems = isFringe ? (this->NR % elemsPerReg) : elemsPerReg;

            if (hasSF && sclType == matOpScaleType::rowVector) {
                RETURN_IF_ERROR(this->loadVector(sfDtype, jit->ptr[this->regTmp1 + j * sfLoadBytes],
                                RegType(sfRegIdx), isFringe));
            }

            jit->vxorps(RegType(matRegIdx), RegType(matRegIdx), RegType(matRegIdx));

            for (int k = 0; k < numElems; k++) {
                RETURN_IF_ERROR(this->broadcastScalar(
                    matOpDtype, jit->ptr[this->regTmp4], RegType(tmpRegIdx)));

                if (k == 0) {
                    jit->vmovaps(RegType(matRegIdx), RegType(tmpRegIdx));
                } else {
                    jit->vblendps(RegType(matRegIdx), RegType(matRegIdx),
                                 RegType(tmpRegIdx), (1 << k));
                }

                jit->add(this->regTmp4, this->regTmp3);
            }

            applyMatOp(opType, hasSF, isFringe, accumIdx, matRegIdx, sfRegIdx, Xbyak::Opmask(0));
        }

        jit->add(this->regTmp2, matElemSize);
    }
    return jitGeneratorError::success;
}

// ─────────────────────────────────────────────────────────────────────────────
// MatOps::gemvN1Path - GEMV N=1 (all KTypes)
// ─────────────────────────────────────────────────────────────────────────────
template<utils::kernelInstrType KType>
jitGeneratorError MatOps<KType>::gemvN1Path(
    matOpType opType, matOpScaleType sclType, bool hasSF,
    DataType sfDtype, DataType matOpDtype,
    int matRegIdx, int sfRegIdx)
{
    auto* jit = this->jit;
    const int sfElemSize = opBase::getElementSize(sfDtype);
    const int matElemSize = opBase::getElementSize(matOpDtype);
    const int sfLoadBytes = opBase::getLoadBytes(sfDtype);
    const int matLoadBytes = opBase::getLoadBytes(matOpDtype);

    // mat + c_i * elemSize (contiguous, ldm=1)
    jit->lea(this->regTmp3, jit->ptr[this->regTmp6 * matElemSize]);
    jit->add(this->regTmp2, this->regTmp3);

    if (hasSF && sclType == matOpScaleType::rowVector) {
        jit->lea(this->regTmp1, jit->ptr[this->regTmp1 + this->regTmp7 * sfElemSize]);
        RETURN_IF_ERROR(this->broadcastScalar(sfDtype, jit->ptr[this->regTmp1], RegType(sfRegIdx)));
    }

    for (int j = 0; j < this->numRegsPerRow; j++) {
        bool isFringe = (j >= this->numFullRegsPerRow);
        int maskIdx = isFringe ? (j - this->numFullRegsPerRow) : 0;
        int accumIdx = this->cRegStartIdx + j;
        Xbyak::Opmask fringeMask = this->getFringeMask(maskIdx);

        RETURN_IF_ERROR(this->loadVector(matOpDtype, jit->ptr[this->regTmp2 + j * matLoadBytes],
                        RegType(matRegIdx), isFringe, fringeMask));

        if (hasSF && sclType == matOpScaleType::columnVector) {
            RETURN_IF_ERROR(this->loadVector(sfDtype, jit->ptr[this->regTmp1 + j * sfLoadBytes],
                            RegType(sfRegIdx), isFringe, fringeMask));
        }

        applyMatOp(opType, hasSF, isFringe, accumIdx, matRegIdx, sfRegIdx, fringeMask);
    }
    return jitGeneratorError::success;
}

// ═══════════════════════════════════════════════════════════════════════════
// Op: GeluErf -- 0.5 * x * (1 + erf(x / sqrt(2)))
// Bottom-up order: callees before callers so specializations are visible.
// ═══════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// GeluErf::polyEvalHorner16 -- Degree-16 Horner in double precision
// poly = (((c15*x + c14)*x + c13)*x + ... + c0) * x
// ─────────────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
void
GeluErf<KType>::polyEvalHorner16(int r, int c1, int c2)
{
    auto* jit = this->jit;

    jit->vbroadcastsd(RegType(c1), this->erfAddr(dlpgemmErfOff + 15 * (int)sizeof(double)));
    jit->vbroadcastsd(RegType(c2), this->erfAddr(dlpgemmErfOff + 14 * (int)sizeof(double)));
    jit->vfmadd231pd(RegType(c2), RegType(r), RegType(c1));

    jit->vbroadcastsd(RegType(c1), this->erfAddr(dlpgemmErfOff + 13 * (int)sizeof(double)));
    jit->vfmadd231pd(RegType(c1), RegType(r), RegType(c2));

    jit->vbroadcastsd(RegType(c2), this->erfAddr(dlpgemmErfOff + 12 * (int)sizeof(double)));
    jit->vfmadd231pd(RegType(c2), RegType(r), RegType(c1));

    jit->vbroadcastsd(RegType(c1), this->erfAddr(dlpgemmErfOff + 11 * (int)sizeof(double)));
    jit->vfmadd231pd(RegType(c1), RegType(r), RegType(c2));

    jit->vbroadcastsd(RegType(c2), this->erfAddr(dlpgemmErfOff + 10 * (int)sizeof(double)));
    jit->vfmadd231pd(RegType(c2), RegType(r), RegType(c1));

    jit->vbroadcastsd(RegType(c1), this->erfAddr(dlpgemmErfOff + 9 * (int)sizeof(double)));
    jit->vfmadd231pd(RegType(c1), RegType(r), RegType(c2));

    jit->vbroadcastsd(RegType(c2), this->erfAddr(dlpgemmErfOff + 8 * (int)sizeof(double)));
    jit->vfmadd231pd(RegType(c2), RegType(r), RegType(c1));

    jit->vbroadcastsd(RegType(c1), this->erfAddr(dlpgemmErfOff + 7 * (int)sizeof(double)));
    jit->vfmadd231pd(RegType(c1), RegType(r), RegType(c2));

    jit->vbroadcastsd(RegType(c2), this->erfAddr(dlpgemmErfOff + 6 * (int)sizeof(double)));
    jit->vfmadd231pd(RegType(c2), RegType(r), RegType(c1));

    jit->vbroadcastsd(RegType(c1), this->erfAddr(dlpgemmErfOff + 5 * (int)sizeof(double)));
    jit->vfmadd231pd(RegType(c1), RegType(r), RegType(c2));

    jit->vbroadcastsd(RegType(c2), this->erfAddr(dlpgemmErfOff + 4 * (int)sizeof(double)));
    jit->vfmadd231pd(RegType(c2), RegType(r), RegType(c1));

    jit->vbroadcastsd(RegType(c1), this->erfAddr(dlpgemmErfOff + 3 * (int)sizeof(double)));
    jit->vfmadd231pd(RegType(c1), RegType(r), RegType(c2));

    jit->vbroadcastsd(RegType(c2), this->erfAddr(dlpgemmErfOff + 2 * (int)sizeof(double)));
    jit->vfmadd231pd(RegType(c2), RegType(r), RegType(c1));

    jit->vbroadcastsd(RegType(c1), this->erfAddr(dlpgemmErfOff + 1 * (int)sizeof(double)));
    jit->vfmadd231pd(RegType(c1), RegType(r), RegType(c2));

    jit->vbroadcastsd(RegType(c2), this->erfAddr(dlpgemmErfOff + 0 * (int)sizeof(double)));
    jit->vfmadd231pd(RegType(c2), RegType(r), RegType(c1));

    jit->vmulpd(RegType(r), RegType(c2), RegType(r));
}

// ─────────────────────────────────────────────────────────────────────────────
// GeluErf::erf -- AVX512-ZMM specialization (piecewise f32 polynomial)
// Uses VPERMT2PS to select coefficients from 24-region piecewise polynomial.
// ─────────────────────────────────────────────────────────────────────────────

template<>
inline void
GeluErf<utils::kernelInstrType::avx512_zmm_32_reg>::erf(
    int y, int r, int c1, int c2,
    int x, int r2, int z, int dn, int q)
{
    using ZmmReg = Xbyak::Zmm;

    // Absolute value and sign extraction
    jit->vpbroadcastd(ZmmReg(c2), this->erfAddr(erfF32ConstantsOff + 1 * (int)sizeof(uint32_t)));  // abs_mask
    jit->vpandd(ZmmReg(x), ZmmReg(r), ZmmReg(c2));

    jit->vpbroadcastd(ZmmReg(c1), this->erfAddr(erfF32ConstantsOff + 2 * (int)sizeof(uint32_t)));  // sign_mask
    jit->vpandd(ZmmReg(c1), ZmmReg(r), ZmmReg(c1));

    // Clamp to upper bound (7.0)
    jit->vbroadcastss(ZmmReg(c2), this->erfAddr(erfF32ConstantsOff + 3 * (int)sizeof(uint32_t)));  // rbound
    Xbyak::Opmask erfMask(erfCmpMaskIdx);
    jit->vcmpps(erfMask, ZmmReg(x), ZmmReg(c2), 0x1E);
    jit->vblendmps(ZmmReg(x) | erfMask, ZmmReg(x), ZmmReg(c2));

    // Compute region indices
    jit->vpbroadcastd(ZmmReg(r2), this->erfAddr(erfF32ConstantsOff + 0 * (int)sizeof(uint32_t)));  // erf_idx_bias
    jit->vpaddd(ZmmReg(r2), ZmmReg(x), ZmmReg(r2));
    jit->vpsrad(ZmmReg(r2), ZmmReg(r2), 21);

    // Clamp indices >= 0
    jit->vpxord(ZmmReg(c2), ZmmReg(c2), ZmmReg(c2));
    jit->vpmaxsd(ZmmReg(r2), ZmmReg(r2), ZmmReg(c2));

    // Horner's method with VPERMT2PS coefficient selection
    jit->vmovups(ZmmReg(z), this->erfAddr(erfF32CoeffsOff + 5 * 32 * (int)sizeof(uint32_t)));
    jit->vmovups(ZmmReg(dn), this->erfAddr(erfF32CoeffsOff + (5 * 32 + 16) * (int)sizeof(uint32_t)));
    jit->vmovdqa32(ZmmReg(c2), ZmmReg(r2));
    jit->vpermt2ps(ZmmReg(z), ZmmReg(c2), ZmmReg(dn));
    jit->vmovups(ZmmReg(q), ZmmReg(z));

    for (int deg = 4; deg >= 0; deg--) {
        jit->vmovups(ZmmReg(z), this->erfAddr(erfF32CoeffsOff + deg * 32 * (int)sizeof(uint32_t)));
        jit->vmovups(ZmmReg(dn), this->erfAddr(erfF32CoeffsOff + (deg * 32 + 16) * (int)sizeof(uint32_t)));
        jit->vmovdqa32(ZmmReg(c2), ZmmReg(r2));
        jit->vpermt2ps(ZmmReg(z), ZmmReg(c2), ZmmReg(dn));
        jit->vfmadd213ps(ZmmReg(q), ZmmReg(x), ZmmReg(z));
    }

    // Restore sign
    jit->vxorps(ZmmReg(y), ZmmReg(q), ZmmReg(c1));
}

// ─────────────────────────────────────────────────────────────────────────────
// GeluErf::erf -- Generic (AVX2 / AVX512-YMM): double-precision Horner
// ─────────────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
void
GeluErf<KType>::erf(
    int y, int r, int c1, int c2,
    int x, int r2, int z, int dn, int q)
{
    auto* jit = this->jit;

    jit->mov(this->regTmp5Half, 0x7FFFFFFF);
    jit->vmovd(halfRegType(c2), this->regTmp5Half);
    jit->vpbroadcastd(RegType(c2), halfRegType(c2));
    jit->vandps(RegType(r2), RegType(r), RegType(c2));

    jit->vcvtps2pd(RegType(y), halfRegType(r2));
    // vextractf128 is VEX-only (regs 0-15); use vextractf32x4 for AVX-512
    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        jit->vextractf128(halfRegType(x), RegType(r2), 0x01);
    } else {
        jit->vextractf32x4(halfRegType(x), RegType(r2), 0x01);
    }
    jit->vcvtps2pd(RegType(x), halfRegType(x));

    polyEvalHorner16(y, c1, c2);
    polyEvalHorner16(x, c1, c2);

    // vinsertf128 is VEX-only; use vinsertf32x4 for AVX-512
    jit->vcvtpd2ps(halfRegType(y), RegType(y));
    jit->vcvtpd2ps(halfRegType(x), RegType(x));
    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        jit->vinsertf128(RegType(y), RegType(y), halfRegType(x), 0x01);
    } else {
        jit->vinsertf32x4(RegType(y), RegType(y), halfRegType(x), 0x01);
    }

    jit->mov(this->regTmp4Half, 0x407AD447);
    jit->vmovd(halfRegType(c2), this->regTmp4Half);
    jit->vpbroadcastd(RegType(c2), halfRegType(c2));

    // vcmpps(ymm)+vblendvps are VEX-only; AVX-512 uses kmask+vblendmps
    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        jit->vcmpps(RegType(c1), RegType(c2), RegType(r2), 0x01);

        jit->vbroadcastss(RegType(c2), this->erfAddr(1 * (int)sizeof(float)));  // 1.0
        jit->vblendvps(RegType(y), RegType(y), RegType(c2), RegType(c1));

        jit->vcmpps(RegType(c1), RegType(c2), RegType(y), 0x01);
        jit->vblendvps(RegType(y), RegType(y), RegType(c2), RegType(c1));
    } else {
        Xbyak::Opmask erfMask(erfCmpMaskIdx);
        jit->vcmpps(erfMask, RegType(c2), RegType(r2), 0x01);

        jit->vbroadcastss(RegType(c2), this->erfAddr(1 * (int)sizeof(float)));  // 1.0
        jit->vblendmps(RegType(y) | erfMask, RegType(y), RegType(c2));

        jit->vcmpps(erfMask, RegType(c2), RegType(y), 0x01);
        jit->vblendmps(RegType(y) | erfMask, RegType(y), RegType(c2));
    }

    // Restore sign
    jit->mov(this->regTmp4Half, 0x80000000);
    jit->vmovd(halfRegType(c2), this->regTmp4Half);
    jit->vpbroadcastd(RegType(c2), halfRegType(c2));
    jit->vandps(RegType(c1), RegType(r), RegType(c2));

    jit->vorps(RegType(y), RegType(c1), RegType(y));
}

// ─────────────────────────────────────────────────────────────────────────────
// GeluErf::geluErfF32 -- AVX512-ZMM specialization (no 1/sqrt2 pre-scaling)
// ERF coefficients approximate erf(x/sqrt(2)) directly.
// ─────────────────────────────────────────────────────────────────────────────

template<>
inline void
GeluErf<utils::kernelInstrType::avx512_zmm_32_reg>::geluErfF32(
    int reg, int xtanh, int c1, int c2,
    int x, int r, int r2, int z, int dn, int q)
{
    using ZmmReg = Xbyak::Zmm;

    jit->vxorps(ZmmReg(xtanh), ZmmReg(xtanh), ZmmReg(xtanh));

    erf(xtanh, reg, c1, c2, x, r2, z, dn, q);

    jit->vbroadcastss(ZmmReg(c2), this->erfAddr(1 * (int)sizeof(float)));  // 1.0
    jit->vaddps(ZmmReg(r2), ZmmReg(xtanh), ZmmReg(c2));

    jit->vmulps(ZmmReg(r2), ZmmReg(r2), ZmmReg(reg));
    jit->vbroadcastss(ZmmReg(c2), this->erfAddr(2 * (int)sizeof(float)));  // 0.5
    jit->vmulps(ZmmReg(reg), ZmmReg(r2), ZmmReg(c2));
}

// ─────────────────────────────────────────────────────────────────────────────
// GeluErf::geluErfF32 -- Generic (AVX2 / AVX512-YMM): pre-scale by 1/sqrt(2)
// ─────────────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
void
GeluErf<KType>::geluErfF32(
    int reg, int xtanh, int c1, int c2,
    int x, int r, int r2, int z, int dn, int q)
{
    auto* jit = this->jit;

    jit->vbroadcastss(RegType(c1), this->erfAddr(0 * (int)sizeof(float)));  // 1/sqrt(2)
    jit->vmulps(RegType(r), RegType(reg), RegType(c1));

    jit->vxorps(RegType(xtanh), RegType(xtanh), RegType(xtanh));

    erf(xtanh, r, c1, c2, x, r2, z, dn, q);

    jit->vbroadcastss(RegType(c2), this->erfAddr(1 * (int)sizeof(float)));  // 1.0
    jit->vaddps(RegType(r2), RegType(xtanh), RegType(c2));

    jit->vmulps(RegType(r2), RegType(r2), RegType(reg));
    jit->vbroadcastss(RegType(c2), this->erfAddr(2 * (int)sizeof(float)));  // 0.5
    jit->vmulps(RegType(reg), RegType(r2), RegType(c2));
}

// ─────────────────────────────────────────────────────────────────────────────
// GeluErf::generateImpl() -- Entry point
// ─────────────────────────────────────────────────────────────────────────────

template<utils::kernelInstrType KType>
jitGeneratorError
GeluErf<KType>::generateImpl(kernelOpsMetaData& op)
{
    if constexpr (Traits::hasMaskSupport) {
        if (erfCmpMaskIdx < 0)
            return jitGeneratorError::notSupported;
    }

    this->requestTable(this->TABLE_ERF);

    RETURN_IF_ERROR(this->vecPool->applyOp(NUM_SCRATCH_NEEDED, nullptr,
        [this](int reg, const int* s, int) {
            int x_tanh = s[0], c1 = s[1], c2 = s[2], x = s[3];
            int r = s[4], r2 = s[5], z = s[6], dn = s[7], q = s[8];
            geluErfF32(reg, x_tanh, c1, c2, x, r, r2, z, dn, q);
        }));

    return jitGeneratorError::success;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shared Elementwise Utilities (expF, polyEval6, tanhF)
// ═══════════════════════════════════════════════════════════════════════════════

template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::polyEval6(
    int const1, int const2, int r, int r2, int z, int q)
{
    jit->vmulps(RegType(r2), RegType(r), RegType(r));

    jit->vbroadcastss(RegType(const1), expAddr(3 * (int)sizeof(float)));
    jit->vbroadcastss(RegType(const2), expAddr(2 * (int)sizeof(float)));

    jit->vmovups(RegType(q), RegType(const2));
    jit->vfmadd231ps(RegType(q), RegType(const1), RegType(r));

    jit->vbroadcastss(RegType(const1), expAddr(1 * (int)sizeof(float)));
    jit->vbroadcastss(RegType(const2), expAddr(0 * (int)sizeof(float)));

    jit->vmovups(RegType(z), RegType(const2));
    jit->vfmadd231ps(RegType(z), RegType(const1), RegType(r));

    jit->vfmadd231ps(RegType(z), RegType(r2), RegType(q));

    jit->vmulps(RegType(r2), RegType(r2), RegType(r2));

    jit->vbroadcastss(RegType(const1), expAddr(5 * (int)sizeof(float)));
    jit->vbroadcastss(RegType(const2), expAddr(4 * (int)sizeof(float)));

    jit->vfmadd231ps(RegType(const2), RegType(const1), RegType(r));

    jit->vfmadd231ps(RegType(z), RegType(const2), RegType(r2));
    jit->vmovups(RegType(r), RegType(z));
}


template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::expF(
    int x, int const1, int const2, int r, int r2, int z, int dn, int q,
    int expCmpMaskIdx)
{
    jit->vbroadcastss(RegType(const1),
                      geluAddr(geluMacrosOff + 0 * (int)sizeof(float)));

    jit->vmulps(RegType(z), RegType(x), RegType(const1));

    jit->vbroadcastss(RegType(const2),
                      geluAddr(geluMacrosOff + 1 * (int)sizeof(float)));

    jit->vaddps(RegType(dn), RegType(z), RegType(const2));

    jit->vsubps(RegType(r), RegType(dn), RegType(const2));
    jit->vsubps(RegType(r), RegType(z), RegType(r));

    polyEval6(const1, const2, r, r2, z, q);

    jit->vpslld(RegType(dn), RegType(dn), 0x17);

    jit->vpaddd(RegType(q), RegType(r), RegType(dn));

    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        jit->vpbroadcastd(RegType(const1),
                          geluAddr(geluMacrosOff + 3 * (int)sizeof(float)));

        jit->vcmpps(RegType(const1), RegType(const1), RegType(x), 0x01);

        jit->vbroadcastss(RegType(const2),
                          geluAddr(geluMacrosOff + 4 * (int)sizeof(float)));

        jit->vblendvps(RegType(q), RegType(q), RegType(const2), RegType(const1));

        jit->vbroadcastss(RegType(const1),
                          geluAddr(geluMacrosOff + 2 * (int)sizeof(float)));

        jit->vcmpps(RegType(const1), RegType(x), RegType(const1), 0x01);

        jit->vxorps(RegType(const2), RegType(const2), RegType(const2));

        jit->vblendvps(RegType(q), RegType(q), RegType(const2), RegType(const1));
    } else {
        Xbyak::Opmask expMask(expCmpMaskIdx);

        jit->vpxorq(RegType(const2), RegType(const2), RegType(const2));

        jit->vpbroadcastd(RegType(const1),
                          geluAddr(geluMacrosOff + 2 * (int)sizeof(float)));

        jit->vcmpps(expMask, RegType(const1), RegType(x), 0x06);

        jit->vpandd(RegType(q) | expMask, RegType(q), RegType(const2));

        jit->vbroadcastss(RegType(const1),
                          geluAddr(geluMacrosOff + 3 * (int)sizeof(float)));

        jit->vcmpps(expMask, RegType(const1), RegType(x), 0x06);

        jit->vbroadcastss(RegType(x),
                          geluAddr(geluMacrosOff + 4 * (int)sizeof(float)));

        jit->vpxord(RegType(x) | expMask, RegType(q), RegType(const2));
        jit->vmovups(RegType(q), RegType(x));
    }
}


template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::tanhF(
    int x_tanh, int x, int const1, int const2,
    int r, int r2, int z, int dn, int q,
    int expCmpMaskIdx)
{
    jit->vbroadcastss(RegType(const1), geluAddr(2 * (int)sizeof(float)));

    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        jit->vbroadcastss(RegType(const2), geluAddr(7 * (int)sizeof(float)));
        jit->vandnps(RegType(x), RegType(const2), RegType(x_tanh));
    } else {
        jit->mov(regTmp5Half, 0x7FFFFFFF);
        jit->vpbroadcastd(RegType(const2), regTmp5Half);
        jit->vpandd(RegType(x), RegType(x_tanh), RegType(const2));
    }

    jit->vmulps(RegType(x), RegType(x), RegType(const1));

    expF(x, const1, const2, r, r2, z, dn, q, expCmpMaskIdx);

    jit->mov(regTmp4Half, -1);
    jit->vbroadcastss(RegType(const1), geluAddr(4 * (int)sizeof(float)));

    jit->vaddps(RegType(z), RegType(q), RegType(const1));

    jit->vbroadcastss(RegType(const2), geluAddr(5 * (int)sizeof(float)));

    jit->vaddps(RegType(r), RegType(z), RegType(const2));

    jit->vdivps(RegType(z), RegType(z), RegType(r));

    jit->vmulps(RegType(z), RegType(z), RegType(const1));

    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        jit->mov(regTmp4Half, -2147483648);
        jit->movd(Xbyak::Xmm(const1), regTmp4Half);
        jit->vpbroadcastd(RegType(const1), Xbyak::Xmm(const1));
        jit->vandps(RegType(q), RegType(x_tanh), RegType(const1));
    } else {
        jit->mov(regTmp4Half, -2147483648);
        jit->vpbroadcastd(RegType(const1), regTmp4Half);
        jit->vpandd(RegType(q), RegType(x_tanh), RegType(const1));
    }

    jit->vxorps(RegType(x_tanh), RegType(q), RegType(z));
}


template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::broadcastScalarInt32andConvertToF32(
    const Xbyak::Address& addr, const RegType& dest)
{
    jit->vpbroadcastd(dest, addr);
    jit->vcvtdq2ps(dest, dest);
}


template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::loadVectorInt32andConvertToF32(
    const Xbyak::Address& addr, const RegType& dest,
    bool useMaskOp, const Xbyak::Opmask& mask)
{
    loadVectorFloat32(addr, dest, useMaskOp, mask);
    jit->vcvtdq2ps(dest, dest);
}


// ═══════════════════════════════════════════════════════════════════════════════
// Op: Relu -- max(0, x)
// ═══════════════════════════════════════════════════════════════════════════════

template<utils::kernelInstrType KType>
jitGeneratorError
Relu<KType>::generateImpl(kernelOpsMetaData& op)
{
    utils::registerGuard<RegType> zeroGuard;
    RETURN_IF_ERROR(this->vecPool->acquireGuard(zeroGuard));

    RegType zeroReg(zeroGuard.idx());
    this->jit->vxorps(zeroReg, zeroReg, zeroReg);

    for (int i = 0; i < this->cRegCount; i++) {
        RegType acc(this->cRegStartIdx + i);
        this->jit->vmaxps(acc, acc, zeroReg);
    }
    return jitGeneratorError::success;
}


// ═══════════════════════════════════════════════════════════════════════════════
// Op: Swish -- x * sigmoid(alpha * x)
// ═══════════════════════════════════════════════════════════════════════════════

template<utils::kernelInstrType KType>
void
Swish<KType>::swishF32(
    int reg, int x_tanh, int x, int const1, int const2,
    int r, int r2, int z, int dn, int q)
{
    auto* jit = this->jit;

    jit->vxorps(RegType(x), RegType(x), RegType(x));
    jit->vfnmadd231ps(RegType(x), RegType(reg), RegType(x_tanh));

    this->expF(x, const1, const2, r, r2, z, dn, q, expCmpMaskIdx);

    jit->vbroadcastss(RegType(const1), this->geluAddr(6 * (int)sizeof(float)));
    jit->vaddps(RegType(q), RegType(q), RegType(const1));
    jit->vdivps(RegType(reg), RegType(reg), RegType(q));
}

template<utils::kernelInstrType KType>
jitGeneratorError
Swish<KType>::generateImpl(kernelOpsMetaData& op)
{
    if constexpr (Traits::hasMaskSupport) {
        if (expCmpMaskIdx < 0)
            return jitGeneratorError::notSupported;
    }

    this->requestTable(this->TABLE_GELU | this->TABLE_EXP);

    this->jit->mov(this->regTmp1,
                   this->jit->ptr[this->regkernelOpsList
                                  + offsetof(dlp_gemm_post_op, op_args2)]);

    utils::registerGuard<RegType> alphaGuard;
    RETURN_IF_ERROR(this->vecPool->acquireGuard(alphaGuard));
    int alphaIdx = alphaGuard.idx();
    RETURN_IF_ERROR(this->broadcastScalar(op.paramStorageDt,
                          this->jit->ptr[this->regTmp1], RegType(alphaIdx)));

    RETURN_IF_ERROR(this->vecPool->applyOp(NUM_SCRATCH_NEEDED - 1,
        nullptr,
        [this, alphaIdx](int reg, const int* s, int) {
            int x = s[0], c1 = s[1], c2 = s[2];
            int r = s[3], r2 = s[4], z = s[5], dn = s[6], q = s[7];
            swishF32(reg, alphaIdx, x, c1, c2, r, r2, z, dn, q);
        }));

    return jitGeneratorError::success;
}


// ═══════════════════════════════════════════════════════════════════════════════
// Op: GeluTanh -- 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ═══════════════════════════════════════════════════════════════════════════════

template<utils::kernelInstrType KType>
void
GeluTanh<KType>::geluTanhF32(
    int reg, int x_tanh, int x, int const1, int const2,
    int r, int r2, int z, int dn, int q)
{
    auto* jit = this->jit;

    jit->vmulps(RegType(r2), RegType(reg), RegType(reg));
    jit->vmulps(RegType(r2), RegType(r2), RegType(reg));

    jit->vbroadcastss(RegType(const1), this->geluAddr(0 * (int)sizeof(float)));
    jit->vmovups(RegType(r), RegType(reg));
    jit->vfmadd231ps(RegType(r), RegType(r2), RegType(const1));

    jit->vbroadcastss(RegType(const2), this->geluAddr(1 * (int)sizeof(float)));
    jit->vmulps(RegType(x_tanh), RegType(r), RegType(const2));

    this->tanhF(x_tanh, x, const1, const2, r, r2, z, dn, q, expCmpMaskIdx);

    jit->vbroadcastss(RegType(const2), this->geluAddr(6 * (int)sizeof(float)));
    jit->vaddps(RegType(x_tanh), RegType(x_tanh), RegType(const2));
    jit->vmulps(RegType(x_tanh), RegType(x_tanh), RegType(reg));

    jit->vbroadcastss(RegType(const1), this->geluAddr(3 * (int)sizeof(float)));
    jit->vmulps(RegType(reg), RegType(x_tanh), RegType(const1));
}

template<utils::kernelInstrType KType>
jitGeneratorError
GeluTanh<KType>::generateImpl(kernelOpsMetaData& op)
{
    if constexpr (Traits::hasMaskSupport) {
        if (expCmpMaskIdx < 0)
            return jitGeneratorError::notSupported;
    }

    this->requestTable(this->TABLE_GELU | this->TABLE_EXP);

    RETURN_IF_ERROR(this->vecPool->applyOp(NUM_SCRATCH_NEEDED, nullptr,
        [this](int reg, const int* s, int) {
            int x_tanh = s[0], c1 = s[1], c2 = s[2], x = s[3];
            int r = s[4], r2 = s[5], z = s[6], dn = s[7], q = s[8];
            geluTanhF32(reg, x_tanh, c1, c2, x, r, r2, z, dn, q);
        }));

    return jitGeneratorError::success;
}


// ═══════════════════════════════════════════════════════════════════════════════
// Op: Tanh -- tanh(x) via expF
// ═══════════════════════════════════════════════════════════════════════════════

template<utils::kernelInstrType KType>
void
Tanh<KType>::tanhDef(
    int reg, int x_tanh, int x, int const1, int const2,
    int r, int r2, int z, int dn, int q)
{
    auto* jit = this->jit;

    jit->vxorps(RegType(x), RegType(x), RegType(x));
    jit->vmovups(RegType(x_tanh), RegType(reg));

    this->tanhF(x_tanh, x, const1, const2, r, r2, z, dn, q, expCmpMaskIdx);

    jit->vmovups(RegType(reg), RegType(x_tanh));
}

template<utils::kernelInstrType KType>
jitGeneratorError
Tanh<KType>::generateImpl(kernelOpsMetaData& op)
{
    if constexpr (Traits::hasMaskSupport) {
        if (expCmpMaskIdx < 0)
            return jitGeneratorError::notSupported;
    }

    this->requestTable(this->TABLE_GELU | this->TABLE_EXP);

    RETURN_IF_ERROR(this->vecPool->applyOp(NUM_SCRATCH_NEEDED, nullptr,
        [this](int reg, const int* s, int) {
            int x_tanh = s[0], c1 = s[1], c2 = s[2], x = s[3];
            int r = s[4], r2 = s[5], z = s[6], dn = s[7], q = s[8];
            tanhDef(reg, x_tanh, c1, c2, x, r, r2, z, dn, q);
        }));

    return jitGeneratorError::success;
}


// ═══════════════════════════════════════════════════════════════════════════════
// Op: Sigmoid -- 1 / (1 + exp(-x))
// ═══════════════════════════════════════════════════════════════════════════════

template<utils::kernelInstrType KType>
void
Sigmoid<KType>::sigmoidDef(
    int reg, int x, int const1, int const2,
    int r, int r2, int z, int dn, int q)
{
    auto* jit = this->jit;

    jit->vbroadcastss(RegType(const1), this->geluAddr(4 * (int)sizeof(float)));
    jit->vmulps(RegType(x), RegType(const1), RegType(reg));

    this->expF(x, const1, const2, r, r2, z, dn, q, expCmpMaskIdx);

    jit->vbroadcastss(RegType(const1), this->geluAddr(6 * (int)sizeof(float)));
    jit->vaddps(RegType(q), RegType(q), RegType(const1));
    jit->vdivps(RegType(reg), RegType(const1), RegType(q));
}

template<utils::kernelInstrType KType>
jitGeneratorError
Sigmoid<KType>::generateImpl(kernelOpsMetaData& op)
{
    if constexpr (Traits::hasMaskSupport) {
        if (expCmpMaskIdx < 0)
            return jitGeneratorError::notSupported;
    }

    this->requestTable(this->TABLE_GELU | this->TABLE_EXP);

    RETURN_IF_ERROR(this->vecPool->applyOp(NUM_SCRATCH_NEEDED, nullptr,
        [this](int reg, const int* s, int) {
            int x = s[0], const1 = s[1], const2 = s[2];
            int r = s[3], r2 = s[4], z = s[5], dn = s[6], q = s[7];
            sigmoidDef(reg, x, const1, const2, r, r2, z, dn, q);
        }));

    return jitGeneratorError::success;
}


// ═══════════════════════════════════════════════════════════════════════════════
// Op: Clip -- clamp(x, min, max)
// ═══════════════════════════════════════════════════════════════════════════════

template<utils::kernelInstrType KType>
jitGeneratorError
Clip<KType>::generateImpl(kernelOpsMetaData& op)
{
    this->jit->mov(this->regTmp1,
                   this->jit->ptr[this->regkernelOpsList
                                  + offsetof(dlp_gemm_post_op, op_args2)]);
    this->jit->mov(this->regTmp2,
                   this->jit->ptr[this->regkernelOpsList
                                  + offsetof(dlp_gemm_post_op, op_args3)]);

    utils::registerGuard<RegType> minGuard, maxGuard;
    RETURN_IF_ERROR(this->vecPool->acquireGuard(minGuard));
    RETURN_IF_ERROR(this->vecPool->acquireGuard(maxGuard));

    RegType minReg(minGuard.idx()), maxReg(maxGuard.idx());
    RETURN_IF_ERROR(this->broadcastScalar(op.paramStorageDt,
                          this->jit->ptr[this->regTmp1], minReg));
    RETURN_IF_ERROR(this->broadcastScalar(op.paramStorageDt,
                          this->jit->ptr[this->regTmp2], maxReg));

    for (int i = 0; i < this->cRegCount; i++) {
        RegType acc(this->cRegStartIdx + i);
        this->jit->vmaxps(acc, acc, minReg);
        this->jit->vminps(acc, acc, maxReg);
    }
    return jitGeneratorError::success;
}


// ═══════════════════════════════════════════════════════════════════════════════
// Op: ReluScale -- max(0,x) + alpha * min(0,x)
// ═══════════════════════════════════════════════════════════════════════════════

template<utils::kernelInstrType KType>
jitGeneratorError
ReluScale<KType>::generateImpl(kernelOpsMetaData& op)
{
    if constexpr (Traits::hasMaskSupport) {
        if (cmpMaskIdx < 0)
            return jitGeneratorError::notSupported;
    }

    this->jit->mov(this->regTmp1,
                   this->jit->ptr[this->regkernelOpsList
                                  + offsetof(dlp_gemm_post_op, op_args2)]);

    utils::registerGuard<RegType> scaleGuard;
    RETURN_IF_ERROR(this->vecPool->acquireGuard(scaleGuard));
    int scaleIdx = scaleGuard.idx();
    RETURN_IF_ERROR(this->broadcastScalar(op.paramStorageDt,
                          this->jit->ptr[this->regTmp1], RegType(scaleIdx)));

    RETURN_IF_ERROR(this->vecPool->applyOp(NUM_SCRATCH_NEEDED - 1,
        [this](const int* s, int) -> jitGeneratorError {
            RegType zeroReg(s[0]);
            this->jit->vxorps(zeroReg, zeroReg, zeroReg);
            return jitGeneratorError::success;
        },
        [this, scaleIdx](int reg, const int* s, int) {
            RegType zeroReg(s[0]), scaleReg(scaleIdx);
            if constexpr (Traits::hasMaskSupport) {
                Xbyak::Opmask cmpMask(cmpMaskIdx);
                this->jit->vcmpps(cmpMask, RegType(reg), zeroReg, 0x02);
                this->jit->vmulps(RegType(reg) | cmpMask,
                                  RegType(reg), scaleReg);
            } else {
                RegType scratchReg(s[1]);
                this->jit->vminps(scratchReg, RegType(reg), zeroReg);
                this->jit->vmaxps(RegType(reg), RegType(reg), zeroReg);
                this->jit->vmulps(scratchReg, scratchReg, scaleReg);
                this->jit->vorps(RegType(reg), RegType(reg), scratchReg);
            }
        }));

    return jitGeneratorError::success;
}


// ═══════════════════════════════════════════════════════════════════════════
// Op: Downscale Implementation
// C[i][j] = C[i][j] * scale_factor + zero_point
// ═══════════════════════════════════════════════════════════════════════════

template<utils::kernelInstrType KType>
typename Downscale<KType>::LoadMode
Downscale<KType>::getLoadMode(bool isScalar, storageFormat fmt) const
{
    if (isScalar) return LoadMode::Scalar;
    if (this->isGEMVN1()) {
        if (fmt == storageFormat::rowMajor)
            return LoadMode::Scalar;
        return LoadMode::PerCol;
    }
    if (fmt == storageFormat::rowMajor) return LoadMode::PerCol;
    return LoadMode::PerRow;
}

template<utils::kernelInstrType KType>
jitGeneratorError
Downscale<KType>::applyDownscale(
    int effectiveMR, bool hasSF, bool hasZP,
    LoadMode sfMode, LoadMode zpMode,
    DataType sfDt, DataType zpDt,
    const Xbyak::Reg64& sfBase, const Xbyak::Reg64& zpBase)
{
    using VecGuard = utils::registerGuard<RegType>;
    auto* jit = this->jit;

    int sfLoadBytes = hasSF ? opBase::getLoadBytes(sfDt) : 0;
    int zpLoadBytes = hasZP ? opBase::getLoadBytes(zpDt) : 0;
    int sfElemSize  = hasSF ? opBase::getElementSize(sfDt) : 0;
    int zpElemSize  = hasZP ? opBase::getElementSize(zpDt) : 0;

    // Phase 1: Pre-load scalar SF/ZP
    VecGuard scalarSfG, scalarZpG;
    if (hasSF && sfMode == LoadMode::Scalar) {
        RETURN_IF_ERROR(this->vecPool->acquireGuard(scalarSfG));
        RETURN_IF_ERROR(this->broadcastScalar(sfDt, jit->ptr[sfBase],
                              RegType(scalarSfG.idx())));
    }
    if (hasZP && zpMode == LoadMode::Scalar) {
        RETURN_IF_ERROR(this->vecPool->acquireGuard(scalarZpG));
        RETURN_IF_ERROR(this->broadcastScalar(zpDt, jit->ptr[zpBase],
                              RegType(scalarZpG.idx())));
    }

    // Resolve SF/ZP register for a given mode; for Scalar mode, returns the
    // pre-loaded register.  For PerCol/PerRow the caller loads into vecGuard
    // first and passes it.
    auto resolveSfReg = [&](VecGuard& vecG) -> RegType {
        return (hasSF && sfMode != LoadMode::Scalar)
                   ? RegType(vecG.idx())
                   : RegType(scalarSfG.idx());
    };
    auto resolveZpReg = [&](VecGuard& vecG) -> RegType {
        return (hasZP && zpMode != LoadMode::Scalar)
                   ? RegType(vecG.idx())
                   : RegType(scalarZpG.idx());
    };

    // Phase 2: Apply operation to accumulators
    // The instruction depends on which parameters are present:
    //   SF+ZP → vfmadd213ps(C, SF, ZP)
    //   SF    → vmulps(C, C, SF)
    //   ZP    → vaddps(C, C, ZP)
    auto emitOp = [&](RegType cReg, RegType sfReg, RegType zpReg) {
        if (hasSF && hasZP) jit->vfmadd213ps(cReg, sfReg, zpReg);
        else if (hasSF)     jit->vmulps(cReg, cReg, sfReg);
        else                jit->vaddps(cReg, cReg, zpReg);
    };

    bool hasPerCol = (hasSF && sfMode == LoadMode::PerCol)
                  || (hasZP && zpMode == LoadMode::PerCol);
    bool hasPerRow = (hasSF && sfMode == LoadMode::PerRow)
                  || (hasZP && zpMode == LoadMode::PerRow);

    // ── PerCol path: iterate columns, load vectors per column ──
    if (hasPerCol) {
        VecGuard fringeBlendG;
        if constexpr (!Traits::hasMaskSupport) {
            if (this->useMask)
                RETURN_IF_ERROR(this->vecPool->acquireGuard(fringeBlendG));
        }

        for (int j = 0; j < this->numRegsPerRow; j++) {
            bool isFringe = this->useMask && (j >= this->numFullRegsPerRow);
            Xbyak::Opmask fMask;
            if (isFringe)
                fMask = this->getFringeMask(j - this->numFullRegsPerRow);

            VecGuard vecSfG, vecZpG;
            if (hasSF && sfMode == LoadMode::PerCol) {
                RETURN_IF_ERROR(this->vecPool->acquireGuard(vecSfG));
                RETURN_IF_ERROR(this->loadVector(sfDt,
                    jit->ptr[sfBase + j * sfLoadBytes],
                    RegType(vecSfG.idx()), isFringe, fMask));
            }
            if (hasZP && zpMode == LoadMode::PerCol) {
                RETURN_IF_ERROR(this->vecPool->acquireGuard(vecZpG));
                RETURN_IF_ERROR(this->loadVector(zpDt,
                    jit->ptr[zpBase + j * zpLoadBytes],
                    RegType(vecZpG.idx()), isFringe, fMask));
            }

            RegType sfReg = hasSF ? resolveSfReg(vecSfG) : RegType(-1);
            RegType zpReg = hasZP ? resolveZpReg(vecZpG) : RegType(-1);

            for (int row = 0; row < effectiveMR; row++) {
                RegType cReg(this->cRegStartIdx
                             + row * this->numRegsPerRow + j);
                if (isFringe) {
                    if constexpr (Traits::hasMaskSupport) {
                        emitOp(cReg | fMask, sfReg, zpReg);
                    } else {
                        RegType scratch(fringeBlendG.idx());
                        if (hasSF && hasZP) {
                            jit->vmovaps(scratch, cReg);
                            jit->vfmadd213ps(scratch, sfReg, zpReg);
                            jit->vblendvps(cReg, cReg, scratch,
                                           this->ymmMask);
                        } else if (hasSF) {
                            jit->vmulps(scratch, cReg, sfReg);
                            jit->vblendvps(cReg, cReg, scratch,
                                           this->ymmMask);
                        } else {
                            jit->vandps(scratch, zpReg, this->ymmMask);
                            jit->vaddps(cReg, cReg, scratch);
                        }
                    }
                } else {
                    emitOp(cReg, sfReg, zpReg);
                }
            }
        }
    }
    // ── PerRow path: iterate rows, broadcast per row ──
    else if (hasPerRow) {
        for (int row = 0; row < effectiveMR; row++) {
            VecGuard rowSfG, rowZpG;
            if (hasSF && sfMode == LoadMode::PerRow) {
                RETURN_IF_ERROR(this->vecPool->acquireGuard(rowSfG));
                RETURN_IF_ERROR(this->broadcastScalar(sfDt,
                    jit->ptr[sfBase + row * sfElemSize],
                    RegType(rowSfG.idx())));
            }
            if (hasZP && zpMode == LoadMode::PerRow) {
                RETURN_IF_ERROR(this->vecPool->acquireGuard(rowZpG));
                RETURN_IF_ERROR(this->broadcastScalar(zpDt,
                    jit->ptr[zpBase + row * zpElemSize],
                    RegType(rowZpG.idx())));
            }

            RegType sfReg = hasSF ? resolveSfReg(rowSfG) : RegType(-1);
            RegType zpReg = hasZP ? resolveZpReg(rowZpG) : RegType(-1);

            for (int col = 0; col < this->numRegsPerRow; col++)
                emitOp(RegType(this->cRegStartIdx
                               + row * this->numRegsPerRow + col),
                       sfReg, zpReg);
        }
    }
    // ── All-scalar path: flat loop ──
    else {
        RegType sfReg = hasSF ? RegType(scalarSfG.idx()) : RegType(-1);
        RegType zpReg = hasZP ? RegType(scalarZpG.idx()) : RegType(-1);

        for (int i = 0; i < effectiveMR * this->numRegsPerRow; i++)
            emitOp(RegType(this->cRegStartIdx + i), sfReg, zpReg);
    }

    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
Downscale<KType>::generateImpl(kernelOpsMetaData& op)
{
    auto* jit = this->jit;

    bool hasSF = (op.scalarScaleFactorRequired || op.vectorScaleFactorRequired);
    bool hasZP = (op.scalarZeroPointRequired   || op.vectorZeroPointRequired);

    if (!hasSF && !hasZP)
        return jitGeneratorError::badKernelInfo;

    auto isSupported = [](DataType dt) {
        return dt == DataType::f32  || dt == DataType::bf16 ||
               dt == DataType::f16  ||
               dt == DataType::s8   || dt == DataType::u8   ||
               dt == DataType::s32;
    };
    if (hasSF && !isSupported(op.scaleFactorDt))
        return jitGeneratorError::badKernelInfo;
    if (hasZP && !isSupported(op.zeroPointDt))
        return jitGeneratorError::badKernelInfo;

    DataType sfDt = hasSF ? op.scaleFactorDt : DataType::f32;
    DataType zpDt = hasZP ? op.zeroPointDt   : DataType::f32;

    LoadMode sfMode = hasSF ? getLoadMode(op.scalarScaleFactorRequired,
                                          op.cMatFormat)
                            : LoadMode::Scalar;
    LoadMode zpMode = hasZP ? getLoadMode(op.scalarZeroPointRequired,
                                          op.cMatFormat)
                            : LoadMode::Scalar;

    int effectiveMR = this->isGEMVN1() ? 1 : this->MR;
    int sfElemSize  = hasSF ? opBase::getElementSize(sfDt) : 0;
    int zpElemSize  = hasZP ? opBase::getElementSize(zpDt) : 0;

    // Load SF pointer + apply position offset
    if (hasSF) {
        jit->mov(this->regTmp1,
                 jit->ptr[this->regkernelOpsList
                          + offsetof(dlp_gemm_post_op, scale_factor)]);

        if (sfMode == LoadMode::PerCol) {
            if (!this->isGEMVN1()) {
                jit->mov(this->regTmp7,
                         jit->ptr[this->regkernelOpsAttr
                                  + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
                jit->lea(this->regTmp1,
                         jit->ptr[this->regTmp1 + this->regTmp7 * sfElemSize]);
            } else {
                jit->mov(this->regTmp6,
                         jit->ptr[this->regkernelOpsAttr
                                  + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
                jit->lea(this->regTmp1,
                         jit->ptr[this->regTmp1 + this->regTmp6 * sfElemSize]);
            }
        } else if (sfMode == LoadMode::PerRow) {
            jit->mov(this->regTmp6,
                     jit->ptr[this->regkernelOpsAttr
                              + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
            jit->lea(this->regTmp1,
                     jit->ptr[this->regTmp1 + this->regTmp6 * sfElemSize]);
        }
    }

    // Load ZP pointer + apply position offset
    if (hasZP) {
        jit->mov(this->regTmp2,
                 jit->ptr[this->regkernelOpsList
                          + offsetof(dlp_gemm_post_op, op_args1)]);

        if (zpMode == LoadMode::PerCol) {
            if (!this->isGEMVN1()) {
                jit->mov(this->regTmp7,
                         jit->ptr[this->regkernelOpsAttr
                                  + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
                jit->lea(this->regTmp2,
                         jit->ptr[this->regTmp2 + this->regTmp7 * zpElemSize]);
            } else {
                jit->mov(this->regTmp6,
                         jit->ptr[this->regkernelOpsAttr
                                  + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
                jit->lea(this->regTmp2,
                         jit->ptr[this->regTmp2 + this->regTmp6 * zpElemSize]);
            }
        } else if (zpMode == LoadMode::PerRow) {
            jit->mov(this->regTmp6,
                     jit->ptr[this->regkernelOpsAttr
                              + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
            jit->lea(this->regTmp2,
                     jit->ptr[this->regTmp2 + this->regTmp6 * zpElemSize]);
        }
    }

    return applyDownscale(effectiveMR, hasSF, hasZP,
                          sfMode, zpMode, sfDt, zpDt,
                          this->regTmp1, this->regTmp2);
}


// ═══════════════════════════════════════════════════════════════════════════
// Bias Implementation
// ═══════════════════════════════════════════════════════════════════════════

template<utils::kernelInstrType KType>
jitGeneratorError
Bias<KType>::legacyBiasVector(DataType biasDt, int effectiveMR,
                              const Xbyak::Reg64& biasBase)
{
    using VecGuard = utils::registerGuard<RegType>;
    auto* jit = this->jit;
    int loadBytes = opBase::getLoadBytes(biasDt);

    for (int j = 0; j < this->numRegsPerRow; j++) {
        bool isFringe = this->useMask && (j >= this->numFullRegsPerRow);

        VecGuard biasGuard;
        RETURN_IF_ERROR(this->vecPool->acquireGuard(biasGuard));
        RegType biasReg(biasGuard.idx());

        if (isFringe) {
            Xbyak::Opmask mask = this->getFringeMask(j - this->numFullRegsPerRow);
            RETURN_IF_ERROR(this->loadVector(biasDt, jit->ptr[biasBase + j * loadBytes],
                             biasReg, true, mask));
        } else {
            RETURN_IF_ERROR(this->loadVector(biasDt, jit->ptr[biasBase + j * loadBytes],
                             biasReg, false));
        }

        for (int row = 0; row < effectiveMR; row++)
            jit->vaddps(
                RegType(this->cRegStartIdx + row * this->numRegsPerRow + j),
                RegType(this->cRegStartIdx + row * this->numRegsPerRow + j),
                biasReg);
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
Bias<KType>::legacyBiasBroadcast(DataType biasDt, int effectiveMR,
                                 const Xbyak::Reg64& biasBase)
{
    using VecGuard = utils::registerGuard<RegType>;
    auto* jit = this->jit;
    int elemSize = opBase::getElementSize(biasDt);

    for (int row = 0; row < effectiveMR; row++) {
        VecGuard biasGuard;
        RETURN_IF_ERROR(this->vecPool->acquireGuard(biasGuard));
        RegType biasReg(biasGuard.idx());

        RETURN_IF_ERROR(this->broadcastScalar(biasDt,
                              jit->ptr[biasBase + row * elemSize],
                              biasReg));

        for (int j = 0; j < this->numRegsPerRow; j++)
            jit->vaddps(
                RegType(this->cRegStartIdx + row * this->numRegsPerRow + j),
                RegType(this->cRegStartIdx + row * this->numRegsPerRow + j),
                biasReg);
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
Bias<KType>::scalarBiasBroadcast(DataType biasDt, int effectiveMR,
                                 const Xbyak::Reg64& biasBase)
{
    using VecGuard = utils::registerGuard<RegType>;
    auto* jit = this->jit;

    VecGuard biasGuard;
    RETURN_IF_ERROR(this->vecPool->acquireGuard(biasGuard));
    RegType biasReg(biasGuard.idx());

    RETURN_IF_ERROR(this->broadcastScalar(biasDt, jit->ptr[biasBase], biasReg));

    int totalRegs = effectiveMR * this->numRegsPerRow;
    for (int i = 0; i < totalRegs; i++)
        jit->vaddps(RegType(this->cRegStartIdx + i),
                    RegType(this->cRegStartIdx + i),
                    biasReg);
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
Bias<KType>::biasDeQuantPerCol(int effectiveMR, bool biasIsBroadcast,
                               DataType biasDt, DataType sfDt, DataType zpDt,
                               bool hasSF, bool hasZP,
                               bool sfIsScalar, bool zpIsScalar,
                               const Xbyak::Reg64& biasBase,
                               const Xbyak::Reg64& sfBase,
                               const Xbyak::Reg64& zpBase)
{
    using VecGuard = utils::registerGuard<RegType>;
    auto* jit = this->jit;

    int biasLoadBytes = opBase::getLoadBytes(biasDt);
    int sfLoadBytes   = hasSF ? opBase::getLoadBytes(sfDt) : 0;
    int zpLoadBytes   = hasZP ? opBase::getLoadBytes(zpDt) : 0;

    for (int j = 0; j < this->numRegsPerRow; j++) {
        bool isFringe = this->useMask && (j >= this->numFullRegsPerRow);
        Xbyak::Opmask fringeMask;
        if (isFringe)
            fringeMask = this->getFringeMask(j - this->numFullRegsPerRow);

        VecGuard biasGuard;
        RETURN_IF_ERROR(this->vecPool->acquireGuard(biasGuard));
        RegType biasReg(biasGuard.idx());

        if (biasIsBroadcast) {
            RETURN_IF_ERROR(this->broadcastScalar(biasDt, jit->ptr[biasBase], biasReg));
        } else if (isFringe) {
            RETURN_IF_ERROR(this->loadVector(biasDt, jit->ptr[biasBase + j * biasLoadBytes],
                             biasReg, true, fringeMask));
        } else {
            RETURN_IF_ERROR(this->loadVector(biasDt, jit->ptr[biasBase + j * biasLoadBytes],
                             biasReg, false));
        }

        if (hasZP) {
            VecGuard zpGuard;
            RETURN_IF_ERROR(this->vecPool->acquireGuard(zpGuard));
            RegType zpReg(zpGuard.idx());

            if (zpIsScalar) {
                RETURN_IF_ERROR(this->broadcastScalar(zpDt, jit->ptr[zpBase], zpReg));
            } else if (isFringe) {
                RETURN_IF_ERROR(this->loadVector(zpDt, jit->ptr[zpBase + j * zpLoadBytes],
                                 zpReg, true, fringeMask));
            } else {
                RETURN_IF_ERROR(this->loadVector(zpDt, jit->ptr[zpBase + j * zpLoadBytes],
                                 zpReg, false));
            }

            jit->vsubps(biasReg, biasReg, zpReg);
        }

        if (hasSF) {
            VecGuard sfGuard;
            RETURN_IF_ERROR(this->vecPool->acquireGuard(sfGuard));
            RegType sfReg(sfGuard.idx());

            if (sfIsScalar) {
                RETURN_IF_ERROR(this->broadcastScalar(sfDt, jit->ptr[sfBase], sfReg));
            } else if (isFringe) {
                RETURN_IF_ERROR(this->loadVector(sfDt, jit->ptr[sfBase + j * sfLoadBytes],
                                 sfReg, true, fringeMask));
            } else {
                RETURN_IF_ERROR(this->loadVector(sfDt, jit->ptr[sfBase + j * sfLoadBytes],
                                 sfReg, false));
            }

            jit->vmulps(biasReg, biasReg, sfReg);
        }

        for (int row = 0; row < effectiveMR; row++)
            jit->vaddps(
                RegType(this->cRegStartIdx + row * this->numRegsPerRow + j),
                RegType(this->cRegStartIdx + row * this->numRegsPerRow + j),
                biasReg);
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
Bias<KType>::biasDeQuantPerRow(int effectiveMR, bool biasIsBroadcast,
                               DataType biasDt, DataType sfDt, DataType zpDt,
                               bool hasSF, bool hasZP,
                               bool sfIsScalar, bool zpIsScalar,
                               const Xbyak::Reg64& biasBase,
                               const Xbyak::Reg64& sfBase,
                               const Xbyak::Reg64& zpBase)
{
    using VecGuard = utils::registerGuard<RegType>;
    auto* jit = this->jit;

    int biasElemSize = opBase::getElementSize(biasDt);
    int sfElemSize   = hasSF ? opBase::getElementSize(sfDt) : 0;
    int zpElemSize   = hasZP ? opBase::getElementSize(zpDt) : 0;

    for (int row = 0; row < effectiveMR; row++) {

        VecGuard biasGuard;
        RETURN_IF_ERROR(this->vecPool->acquireGuard(biasGuard));
        RegType biasReg(biasGuard.idx());

        int biasOffset = biasIsBroadcast ? 0 : row * biasElemSize;
        RETURN_IF_ERROR(this->broadcastScalar(biasDt,
                              jit->ptr[biasBase + biasOffset],
                              biasReg));

        if (hasZP) {
            VecGuard zpGuard;
            RETURN_IF_ERROR(this->vecPool->acquireGuard(zpGuard));
            RegType zpReg(zpGuard.idx());

            int zpOffset = zpIsScalar ? 0 : row * zpElemSize;
            RETURN_IF_ERROR(this->broadcastScalar(zpDt, jit->ptr[zpBase + zpOffset], zpReg));

            jit->vsubps(biasReg, biasReg, zpReg);
        }

        if (hasSF) {
            VecGuard sfGuard;
            RETURN_IF_ERROR(this->vecPool->acquireGuard(sfGuard));
            RegType sfReg(sfGuard.idx());

            int sfOffset = sfIsScalar ? 0 : row * sfElemSize;
            RETURN_IF_ERROR(this->broadcastScalar(sfDt, jit->ptr[sfBase + sfOffset], sfReg));

            jit->vmulps(biasReg, biasReg, sfReg);
        }

        for (int j = 0; j < this->numRegsPerRow; j++)
            jit->vaddps(
                RegType(this->cRegStartIdx + row * this->numRegsPerRow + j),
                RegType(this->cRegStartIdx + row * this->numRegsPerRow + j),
                biasReg);
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
Bias<KType>::generateImpl(kernelOpsMetaData& op)
{
    using namespace dlp::kernel_frame;
    auto* jit = this->jit;

    bool hasSF = (op.scalarScaleFactorRequired || op.vectorScaleFactorRequired);
    bool hasZP = (op.scalarZeroPointRequired   || op.vectorZeroPointRequired);
    bool hasDequant = hasSF || hasZP;

    auto isSupported = [](DataType dt) {
        return dt == DataType::f32  || dt == DataType::bf16 ||
               dt == DataType::f16  ||
               dt == DataType::s8   || dt == DataType::u8   ||
               dt == DataType::s32;
    };

    if (!isSupported(op.paramStorageDt))
        return jitGeneratorError::badKernelInfo;

    if (hasSF && !isSupported(op.scaleFactorDt))
        return jitGeneratorError::badKernelInfo;
    if (hasZP && !isSupported(op.zeroPointDt))
        return jitGeneratorError::badKernelInfo;

    DataType biasDt = op.paramStorageDt;
    bool isRowMajor = (op.cMatFormat == storageFormat::rowMajor);
    bool isGEMVN1   = this->isGEMVN1();
    int  effectiveMR = isGEMVN1 ? 1 : this->MR;
    int  biasElemSize = opBase::getElementSize(biasDt);

    bool isScalar = op.isScalarBias;

    jit->mov(this->regTmp1,
             jit->ptr[this->regkernelOpsList
                      + offsetof(dlp_gemm_post_op, op_args1)]);

    if (isScalar && !hasDequant) {
        return scalarBiasBroadcast(biasDt, effectiveMR, this->regTmp1);
    }

    if (!isScalar) {
        if (isRowMajor) {
            jit->mov(this->regTmp7,
                     jit->ptr[this->regkernelOpsAttr
                              + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
            jit->lea(this->regTmp1,
                     jit->ptr[this->regTmp1 + this->regTmp7 * biasElemSize]);
        } else {
            jit->mov(this->regTmp6,
                     jit->ptr[this->regkernelOpsAttr
                              + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
            jit->lea(this->regTmp1,
                     jit->ptr[this->regTmp1 + this->regTmp6 * biasElemSize]);
        }
    }

    if (!hasDequant) {
        if (isRowMajor != isGEMVN1)
            return legacyBiasVector(biasDt, effectiveMR, this->regTmp1);
        else
            return legacyBiasBroadcast(biasDt, effectiveMR, this->regTmp1);
    }

    DataType sfDt = hasSF ? op.scaleFactorDt : DataType::f32;
    DataType zpDt = hasZP ? op.zeroPointDt   : DataType::f32;
    bool sfIsScalar = op.scalarScaleFactorRequired;
    bool zpIsScalar = op.scalarZeroPointRequired;

    int sfElemSize = hasSF ? opBase::getElementSize(sfDt) : 0;
    int zpElemSize = hasZP ? opBase::getElementSize(zpDt) : 0;

    if (hasSF) {
        jit->mov(this->regTmp2,
                 jit->ptr[this->regkernelOpsList
                          + offsetof(dlp_gemm_post_op, scale_factor)]);

        if (!sfIsScalar) {
            if (!isGEMVN1 && isRowMajor) {
                jit->mov(this->regTmp7,
                         jit->ptr[this->regkernelOpsAttr
                                  + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
                jit->lea(this->regTmp2,
                         jit->ptr[this->regTmp2 + this->regTmp7 * sfElemSize]);
            } else {
                jit->mov(this->regTmp6,
                         jit->ptr[this->regkernelOpsAttr
                                  + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
                jit->lea(this->regTmp2,
                         jit->ptr[this->regTmp2 + this->regTmp6 * sfElemSize]);
            }
        }
    }

    if (hasZP) {
        jit->mov(this->regTmp3,
                 jit->ptr[this->regkernelOpsList
                          + offsetof(dlp_gemm_post_op, bias_zp)]);

        if (!zpIsScalar) {
            if (!isGEMVN1 && isRowMajor) {
                jit->mov(this->regTmp7,
                         jit->ptr[this->regkernelOpsAttr
                                  + offsetof(dlp_gemm_post_op_attr, post_op_c_j)]);
                jit->lea(this->regTmp3,
                         jit->ptr[this->regTmp3 + this->regTmp7 * zpElemSize]);
            } else {
                jit->mov(this->regTmp6,
                         jit->ptr[this->regkernelOpsAttr
                                  + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
                jit->lea(this->regTmp3,
                         jit->ptr[this->regTmp3 + this->regTmp6 * zpElemSize]);
            }
        }
    }

    if (isGEMVN1 || isRowMajor) {
        return biasDeQuantPerCol(effectiveMR,
                          /*biasIsBroadcast=*/isScalar || (isGEMVN1 && isRowMajor),
                          biasDt, sfDt, zpDt,
                          hasSF, hasZP,
                          sfIsScalar, zpIsScalar,
                          this->regTmp1, this->regTmp2, this->regTmp3);
    } else {
        return biasDeQuantPerRow(effectiveMR,
                          /*biasIsBroadcast=*/isScalar,
                          biasDt, sfDt, zpDt,
                          hasSF, hasZP,
                          sfIsScalar, zpIsScalar,
                          this->regTmp1, this->regTmp2, this->regTmp3);
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// Op: ADQuantize Implementation
// A-matrix dequantization: result = (acc + b_col_sum * a_zp) * a_sf
// ═══════════════════════════════════════════════════════════════════════════

template<utils::kernelInstrType KType>
jitGeneratorError
ADQuantize<KType>::aDQuantScaleFactorScalarImpl(DataType sfDt, const Xbyak::Reg64& sfBase)
{
    using VecGuard = utils::registerGuard<RegType>;
    auto* jit = this->jit;

    VecGuard sfGuard;
    RETURN_IF_ERROR(this->vecPool->acquireGuard(sfGuard));
    RETURN_IF_ERROR(this->broadcastScalar(sfDt, jit->ptr[sfBase], RegType(sfGuard.idx())));

    for (int i = 0; i < this->MR * this->numRegsPerRow; i++) {
        jit->vmulps(RegType(this->cRegStartIdx + i),
                    RegType(this->cRegStartIdx + i),
                    RegType(sfGuard.idx()));
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
ADQuantize<KType>::aDQuantScaleFactorScalarImplGEMVN1(DataType sfDt, const Xbyak::Reg64& sfBase)
{
    using VecGuard = utils::registerGuard<RegType>;
    auto* jit = this->jit;

    VecGuard sfGuard;
    RETURN_IF_ERROR(this->vecPool->acquireGuard(sfGuard));
    RETURN_IF_ERROR(this->broadcastScalar(sfDt, jit->ptr[sfBase], RegType(sfGuard.idx())));

    for (int i = 0; i < this->numRegsPerRow; i++) {
        jit->vmulps(RegType(this->cRegStartIdx + i),
                    RegType(this->cRegStartIdx + i),
                    RegType(sfGuard.idx()));
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
ADQuantize<KType>::aDQuantScaleFactorRowMajorImpl(DataType sfDt, const Xbyak::Reg64& sfBase)
{
    using VecGuard = utils::registerGuard<RegType>;
    auto* jit = this->jit;

    int sfElemSize = opBase::getElementSize(sfDt);

    jit->mov(this->regTmp2,
             jit->ptr[this->regkernelOpsAttr
                      + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
    jit->lea(this->regTmp2, jit->ptr[this->regTmp2 * sfElemSize]);
    jit->lea(this->regTmp3, jit->ptr[sfBase]);
    jit->add(this->regTmp3, this->regTmp2);

    for (int i = 0; i < this->MR; i++) {
        VecGuard sfGuard;
        RETURN_IF_ERROR(this->vecPool->acquireGuard(sfGuard));
        RETURN_IF_ERROR(this->broadcastScalar(sfDt,
                              jit->ptr[this->regTmp3 + i * sfElemSize],
                              RegType(sfGuard.idx())));

        for (int j = 0; j < this->numRegsPerRow; j++) {
            jit->vmulps(
                RegType(this->cRegStartIdx + i * this->numRegsPerRow + j),
                RegType(this->cRegStartIdx + i * this->numRegsPerRow + j),
                RegType(sfGuard.idx()));
        }
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
ADQuantize<KType>::aDQuantScaleFactorRowMajorImplGEMVN1(DataType sfDt,
                                                        const Xbyak::Reg64& sfBase)
{
    using VecGuard = utils::registerGuard<RegType>;
    auto* jit = this->jit;

    int sfElemSize = opBase::getElementSize(sfDt);

    jit->lea(this->regTmp2, jit->ptr[this->regTmp2 * sfElemSize]);
    jit->add(this->regTmp1, this->regTmp2);

    int sfLoadBytes = opBase::getLoadBytes(sfDt);

    for (int j = 0; j < this->numRegsPerRow; j++) {
        bool isFringe = this->useMask && (j >= this->numFullRegsPerRow);
        Xbyak::Opmask fMask;
        if (isFringe)
            fMask = this->getFringeMask(j - this->numFullRegsPerRow);

        VecGuard sfGuard;
        RETURN_IF_ERROR(this->vecPool->acquireGuard(sfGuard));
        RETURN_IF_ERROR(this->loadVector(sfDt,
            jit->ptr[this->regTmp1 + j * sfLoadBytes],
            RegType(sfGuard.idx()), isFringe, fMask));

        jit->vmulps(RegType(this->cRegStartIdx + j),
                    RegType(this->cRegStartIdx + j),
                    RegType(sfGuard.idx()));
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
ADQuantize<KType>::aDQuantScaleFactorImpl(kernelOpsMetaData& op)
{
    auto* jit = this->jit;

    jit->mov(this->regTmp1,
             jit->ptr[this->regkernelOpsList
                      + offsetof(dlp_gemm_post_op, scale_factor)]);

    if (this->isGEMVN1()) {
        if (op.scalarScaleFactorRequired) {
            return aDQuantScaleFactorScalarImplGEMVN1(op.scaleFactorDt, this->regTmp1);
        } else if (op.cMatFormat == storageFormat::rowMajor) {
            jit->mov(this->regTmp2,
                     jit->ptr[this->regkernelOpsAttr
                              + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
            return aDQuantScaleFactorRowMajorImplGEMVN1(op.scaleFactorDt, this->regTmp1);
        } else {
            return jitGeneratorError::notSupported;
        }
    } else {
        if (op.scalarScaleFactorRequired) {
            return aDQuantScaleFactorScalarImpl(op.scaleFactorDt, this->regTmp1);
        } else if (op.cMatFormat == storageFormat::rowMajor) {
            return aDQuantScaleFactorRowMajorImpl(op.scaleFactorDt, this->regTmp1);
        } else {
            return jitGeneratorError::notSupported;
        }
    }
}

template<utils::kernelInstrType KType>
jitGeneratorError
ADQuantize<KType>::aDQuantZeroPointScalarImpl(DataType zpDt, const Xbyak::Reg64& zpBase)
{
    using VecGuard = utils::registerGuard<RegType>;
    auto* jit = this->jit;

    jit->mov(this->regTmp2,
             jit->ptr[this->regkernelOpsAttr
                      + offsetof(dlp_gemm_post_op_attr, b_col_sum_vec)]);
    jit->mov(this->regTmp3,
             jit->ptr[this->regkernelOpsAttr
                      + offsetof(dlp_gemm_post_op_attr, b_sum_offset)]);
    jit->lea(this->regTmp2,
             jit->ptr[this->regTmp2 + this->regTmp3 * sizeof(int32_t)]);

    VecGuard zpGuard;
    RETURN_IF_ERROR(this->vecPool->acquireGuard(zpGuard));
    RETURN_IF_ERROR(this->broadcastScalar(zpDt, jit->ptr[zpBase], RegType(zpGuard.idx())));

    std::vector<VecGuard> bColSumGuards;
    for (int i = 0; i < this->numRegsPerRow; i++) {
        bColSumGuards.emplace_back();
        RETURN_IF_ERROR(this->vecPool->acquireGuard(bColSumGuards.back()));
    }

    for (int i = 0; i < this->numRegsPerRow; i++) {
        jit->vmovdqu32(RegType(bColSumGuards[i].idx()),
                       jit->ptr[this->regTmp2 + i * this->RegBytes]);
        jit->vpsrad(RegType(bColSumGuards[i].idx()),
                    RegType(bColSumGuards[i].idx()), 7);
        jit->vcvtdq2ps(RegType(bColSumGuards[i].idx()),
                       RegType(bColSumGuards[i].idx()));

        VecGuard tempGuard;
        RETURN_IF_ERROR(this->vecPool->acquireGuard(tempGuard));
        jit->vmulps(RegType(tempGuard.idx()),
                    RegType(bColSumGuards[i].idx()),
                    RegType(zpGuard.idx()));
        for (int j = 0; j < this->MR; j++) {
            jit->vaddps(
                RegType(this->cRegStartIdx + j * this->numRegsPerRow + i),
                RegType(this->cRegStartIdx + j * this->numRegsPerRow + i),
                RegType(tempGuard.idx()));
        }
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
ADQuantize<KType>::aDQuantZeroPointScalarImplGEMVN1(DataType zpDt,
                                                    const Xbyak::Reg64& zpBase)
{
    using VecGuard = utils::registerGuard<RegType>;
    auto* jit = this->jit;

    VecGuard zpGuard;
    RETURN_IF_ERROR(this->vecPool->acquireGuard(zpGuard));
    RETURN_IF_ERROR(this->broadcastScalar(zpDt, jit->ptr[zpBase], RegType(zpGuard.idx())));

    jit->mov(this->regTmp2,
             jit->ptr[this->regkernelOpsAttr
                      + offsetof(dlp_gemm_post_op_attr, b_col_sum_vec)]);

    VecGuard bColSumGuard;
    RETURN_IF_ERROR(this->vecPool->acquireGuard(bColSumGuard));
    jit->vpbroadcastd(RegType(bColSumGuard.idx()), jit->ptr[this->regTmp2]);
    jit->vpsrad(RegType(bColSumGuard.idx()),
                RegType(bColSumGuard.idx()), 7);
    jit->vcvtdq2ps(RegType(bColSumGuard.idx()),
                   RegType(bColSumGuard.idx()));
    jit->vmulps(RegType(bColSumGuard.idx()),
                RegType(bColSumGuard.idx()),
                RegType(zpGuard.idx()));

    for (int i = 0; i < this->numRegsPerRow; i++) {
        jit->vaddps(RegType(this->cRegStartIdx + i),
                    RegType(this->cRegStartIdx + i),
                    RegType(bColSumGuard.idx()));
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
ADQuantize<KType>::aDQuantZeroPointRowMajorImpl(DataType zpDt, const Xbyak::Reg64& zpBase)
{
    using VecGuard = utils::registerGuard<RegType>;
    auto* jit = this->jit;

    int zpElemSize = opBase::getElementSize(zpDt);

    jit->mov(this->regTmp2,
             jit->ptr[this->regkernelOpsAttr
                      + offsetof(dlp_gemm_post_op_attr, b_col_sum_vec)]);
    jit->mov(this->regTmp3,
             jit->ptr[this->regkernelOpsAttr
                      + offsetof(dlp_gemm_post_op_attr, b_sum_offset)]);
    jit->lea(this->regTmp2,
             jit->ptr[this->regTmp2 + this->regTmp3 * sizeof(int32_t)]);

    jit->mov(this->regTmp4,
             jit->ptr[this->regkernelOpsAttr
                      + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
    jit->lea(this->regTmp4, jit->ptr[this->regTmp4 * zpElemSize]);
    jit->lea(this->regTmp3, jit->ptr[zpBase]);
    jit->add(this->regTmp3, this->regTmp4);

    std::vector<VecGuard> bColSumGuards;
    for (int i = 0; i < this->numRegsPerRow; i++) {
        bColSumGuards.emplace_back();
        RETURN_IF_ERROR(this->vecPool->acquireGuard(bColSumGuards.back()));
    }

    for (int i = 0; i < this->numRegsPerRow; i++) {
        jit->vmovdqu32(RegType(bColSumGuards[i].idx()),
                       jit->ptr[this->regTmp2 + i * this->RegBytes]);
        jit->vpsrad(RegType(bColSumGuards[i].idx()),
                    RegType(bColSumGuards[i].idx()), 7);
        jit->vcvtdq2ps(RegType(bColSumGuards[i].idx()),
                       RegType(bColSumGuards[i].idx()));
    }

    for (int j = 0; j < this->MR; j++) {
        VecGuard zpGuard;
        RETURN_IF_ERROR(this->vecPool->acquireGuard(zpGuard));
        RETURN_IF_ERROR(this->broadcastScalar(zpDt,
                              jit->ptr[this->regTmp3 + j * zpElemSize],
                              RegType(zpGuard.idx())));

        for (int i = 0; i < this->numRegsPerRow; i++) {
            VecGuard tempGuard;
            RETURN_IF_ERROR(this->vecPool->acquireGuard(tempGuard));
            jit->vmulps(RegType(tempGuard.idx()),
                        RegType(bColSumGuards[i].idx()),
                        RegType(zpGuard.idx()));
            jit->vaddps(
                RegType(this->cRegStartIdx + j * this->numRegsPerRow + i),
                RegType(this->cRegStartIdx + j * this->numRegsPerRow + i),
                RegType(tempGuard.idx()));
        }
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
ADQuantize<KType>::aDQuantZeroPointRowMajorImplGEMVN1(DataType zpDt,
                                                      const Xbyak::Reg64& zpBase)
{
    using VecGuard = utils::registerGuard<RegType>;
    auto* jit = this->jit;

    int zpElemSize = opBase::getElementSize(zpDt);

    jit->lea(this->regTmp2, jit->ptr[this->regTmp2 * zpElemSize]);
    jit->add(this->regTmp1, this->regTmp2);

    int zpLoadBytes = opBase::getLoadBytes(zpDt);

    jit->mov(this->regTmp3,
             jit->ptr[this->regkernelOpsAttr
                      + offsetof(dlp_gemm_post_op_attr, b_col_sum_vec)]);
    VecGuard bColSumGuard;
    RETURN_IF_ERROR(this->vecPool->acquireGuard(bColSumGuard));
    jit->vpbroadcastd(RegType(bColSumGuard.idx()), jit->ptr[this->regTmp3]);
    jit->vpsrad(RegType(bColSumGuard.idx()),
                RegType(bColSumGuard.idx()), 7);
    jit->vcvtdq2ps(RegType(bColSumGuard.idx()),
                   RegType(bColSumGuard.idx()));

    for (int j = 0; j < this->numRegsPerRow; j++) {
        bool isFringe = this->useMask && (j >= this->numFullRegsPerRow);
        Xbyak::Opmask fMask;
        if (isFringe)
            fMask = this->getFringeMask(j - this->numFullRegsPerRow);

        VecGuard zpGuard;
        RETURN_IF_ERROR(this->vecPool->acquireGuard(zpGuard));
        RETURN_IF_ERROR(this->loadVector(zpDt,
            jit->ptr[this->regTmp1 + j * zpLoadBytes],
            RegType(zpGuard.idx()), isFringe, fMask));

        VecGuard tempGuard;
        RETURN_IF_ERROR(this->vecPool->acquireGuard(tempGuard));
        jit->vmulps(RegType(tempGuard.idx()),
                    RegType(bColSumGuard.idx()),
                    RegType(zpGuard.idx()));
        jit->vaddps(RegType(this->cRegStartIdx + j),
                    RegType(this->cRegStartIdx + j),
                    RegType(tempGuard.idx()));
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
ADQuantize<KType>::aDQuantZeroPointImpl(kernelOpsMetaData& op)
{
    if (!op.scalarZeroPointRequired && !op.vectorZeroPointRequired)
        return jitGeneratorError::success;

    auto* jit = this->jit;

    jit->mov(this->regTmp1,
             jit->ptr[this->regkernelOpsList
                      + offsetof(dlp_gemm_post_op, op_args1)]);

    if (this->isGEMVN1()) {
        if (op.scalarZeroPointRequired) {
            return aDQuantZeroPointScalarImplGEMVN1(op.zeroPointDt, this->regTmp1);
        } else if (op.cMatFormat == storageFormat::rowMajor) {
            jit->mov(this->regTmp2,
                     jit->ptr[this->regkernelOpsAttr
                              + offsetof(dlp_gemm_post_op_attr, post_op_c_i)]);
            return aDQuantZeroPointRowMajorImplGEMVN1(op.zeroPointDt, this->regTmp1);
        } else {
            return jitGeneratorError::notSupported;
        }
    } else {
        if (op.scalarZeroPointRequired) {
            return aDQuantZeroPointScalarImpl(op.zeroPointDt, this->regTmp1);
        } else if (op.cMatFormat == storageFormat::rowMajor) {
            return aDQuantZeroPointRowMajorImpl(op.zeroPointDt, this->regTmp1);
        } else {
            return jitGeneratorError::notSupported;
        }
    }
}

template<utils::kernelInstrType KType>
jitGeneratorError
ADQuantize<KType>::generateImpl(kernelOpsMetaData& op)
{
    RETURN_IF_ERROR(aDQuantZeroPointImpl(op));
    RETURN_IF_ERROR(aDQuantScaleFactorImpl(op));
    return jitGeneratorError::success;
}

} // namespace amdzen::x86gen
