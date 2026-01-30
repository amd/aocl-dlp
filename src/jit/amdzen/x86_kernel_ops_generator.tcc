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
#include "classic/dlp_macros.h"

namespace amdzen::x86gen {

using namespace dlp::kernel_frame;
using namespace dlp::jit;
using namespace Xbyak;

template<utils::kernelInstrType KType>
kernelOpsGeneratorX86<KType>::kernelOpsGeneratorX86(Xbyak::CodeGenerator* jit)
    : regkernelOpsList(jit->rdx)
    , regkernelOpsAttr(jit->r8)
    , regTmp1(jit->r9)
    , regTmp2(jit->r10)
    , regTmp3(jit->r11)
    , regTmp4(jit->rax)
    , regTmp5(jit->rbx)
    , regTmp6(jit->rdi)
    , regTmp7(jit->rsi)
    , regcsC(jit->r12)
    , regTmp4Half(jit->eax)
    , regTmp5Half(jit->ebx)
    , jit_(jit)
{
    // MR, NR, useMask, cRegStartIdx, cRegCount will be set by
    // setPostOpsContext()
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::generateKernelOps(
    std::vector<kernelOpsMetaData>& kernelOps,
    const Xbyak::Reg64&             postOpsArgWrapperPtrReg,
    dlp::jit::jitAlgoType           algoType,
    int                             MR,
    int                             NR,
    bool                            useMask,
    int                             numMaskRegs,
    int                             cRegStartIdx,
    int                             cRegCount)
{
    // Store algorithm type for later use in post-op dispatching
    algoType_ = algoType;
    // Set the post-ops context with the provided parameters
    RETURN_IF_ERROR((setPostOpsContext(MR, NR, useMask, numMaskRegs,
                                       cRegStartIdx, cRegCount)));

    // Save registers used by this generator.
    utils::registerGuard<Xbyak::Reg64> rG{ jit_ };
    rG.saveRegister(regkernelOpsList);
    rG.saveRegister(regkernelOpsAttr);
    rG.saveRegister(regcsC);
    rG.saveRegister(regTmp1);
    rG.saveRegister(regTmp2);
    rG.saveRegister(regTmp3);
    rG.saveRegister(regTmp4);
    rG.saveRegister(regTmp5);
    rG.saveRegister(regTmp6);
    rG.saveRegister(regTmp7);

    // Load the post-ops node and post-ops attr pointers.
    if (algoType == dlp::jit::jitAlgoType::gemv_m1) {
        jit_->mov(
            regkernelOpsList,
            jit_->ptr[postOpsArgWrapperPtrReg
                      + offsetof(dlp::kernels::gemvM1Params, kernelOpsList)]);

        jit_->mov(regcsC,
                  jit_->ptr[postOpsArgWrapperPtrReg
                            + offsetof(dlp::kernels::gemvM1Params, csY)]);

        // Load pointer to kernelOpsAttr struct instead of the struct
        // itself.
        jit_->lea(
            regkernelOpsAttr,
            jit_->ptr[postOpsArgWrapperPtrReg
                      + offsetof(dlp::kernels::gemvM1Params, kernelOpsAttr)]);

        if (useMask) {
            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                fringeMask[0] = Opmask(1);
                jit_->kmovw(fringeMask[0],
                            jit_->ptr[postOpsArgWrapperPtrReg
                                      + offsetof(dlp::kernels::gemvM1Params,
                                                 nmask_avx512)]);
            } else if constexpr (KType
                                 == utils::kernelInstrType::avx512_ymm_32_reg) {
                fringeMask[0] = Opmask(1);
                jit_->kmovb(fringeMask[0],
                            jit_->ptr[postOpsArgWrapperPtrReg
                                      + offsetof(dlp::kernels::gemvM1Params,
                                                 nmask_avx512_256)]);
            } else if constexpr (KType
                                 == utils::kernelInstrType::avx2_ymm_16_reg) {
                jit_->lea(regTmp1,
                          jit_->ptr[postOpsArgWrapperPtrReg
                                    + offsetof(dlp::kernels::gemvM1Params,
                                               nmask_avx2)]);
                jit_->vmovdqu(
                    ymmMask,
                    jit_->ptr[regTmp1]); // Load all 8 floats (32 bytes)
            } else {
                // Currently returning not supported, will have to add
                // the required mask parameter for each configuration as
                // the support gets enabled.
                return jitGeneratorError::notSupported;
            }
        }
    } else if (algoType == dlp::jit::jitAlgoType::gemv_n1) {
        jit_->mov(
            regkernelOpsList,
            jit_->ptr[postOpsArgWrapperPtrReg
                      + offsetof(dlp::kernels::gemvN1Params, kernelOpsList)]);

        jit_->mov(regcsC,
                  jit_->ptr[postOpsArgWrapperPtrReg
                            + offsetof(dlp::kernels::gemvN1Params, csC)]);

        // Load pointer to kernelOpsAttr struct instead of the struct
        // itself.
        jit_->lea(
            regkernelOpsAttr,
            jit_->ptr[postOpsArgWrapperPtrReg
                      + offsetof(dlp::kernels::gemvN1Params, kernelOpsAttr)]);

        if (useMask) {
            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                fringeMask[0] = Opmask(1);
                jit_->kmovw(fringeMask[0],
                            jit_->ptr[postOpsArgWrapperPtrReg
                                      + offsetof(dlp::kernels::gemvN1Params,
                                                 mmask_avx512)]);
            } else if constexpr (KType
                                 == utils::kernelInstrType::avx512_ymm_32_reg) {
                fringeMask[0] = Opmask(1);
                jit_->kmovb(fringeMask[0],
                            jit_->ptr[postOpsArgWrapperPtrReg
                                      + offsetof(dlp::kernels::gemvN1Params,
                                                 mmask_avx512_256)]);
            } else if constexpr (KType
                                 == utils::kernelInstrType::avx2_ymm_16_reg) {
                jit_->lea(regTmp1,
                          jit_->ptr[postOpsArgWrapperPtrReg
                                    + offsetof(dlp::kernels::gemvN1Params,
                                               mmask_avx2)]);
                jit_->vmovdqu(
                    ymmMask,
                    jit_->ptr[regTmp1]); // Load all 8 floats (32 bytes)
            } else {
                // Currently returning not supported, will have to add
                // the required mask parameter for each configuration as
                // the support gets enabled.
                return jitGeneratorError::notSupported;
            }
        }
    } else if (algoType == dlp::jit::jitAlgoType::gemm) {
        jit_->mov(
            regkernelOpsList,
            jit_->ptr[postOpsArgWrapperPtrReg
                      + offsetof(dlp::kernels::gemmParams, kernelOpsList)]);

        jit_->mov(regcsC, jit_->ptr[postOpsArgWrapperPtrReg
                                    + offsetof(dlp::kernels::gemmParams, csC)]);

        // Load pointer to kernelOpsAttr struct instead of the struct itself.
        jit_->lea(
            regkernelOpsAttr,
            jit_->ptr[postOpsArgWrapperPtrReg
                      + offsetof(dlp::kernels::gemmParams, kernelOpsAttr)]);

        if (useMask) {
            // Assuming that the mask registers are carried forward from the
            // GEMM kernel If the mask allocation changes in the caller
            // function(GEMM, GEMV, ELTWISE), we need to change the logic
            // here. we only set the indices of registers accordingly.
            // No need to load the mask values again here.

            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                for (int i = 0; i < numMaskRegs; i++) {
                    fringeMask[i] = Opmask(i + 1);
                    // TODO: Initialize the mask registers once and reuse them
                    // across post-ops. However this assumes that only 4 mask
                    // registers are required. If any more masks are required,
                    // proper mechanism to save and restore already initialized
                    // masks will need to be implemented.
                    jit_->kmovw(fringeMask[i],
                                jit_->ptr[postOpsArgWrapperPtrReg
                                          + offsetof(dlp::kernels::gemmParams,
                                                     maskF32[0])
                                          + (i * sizeof(uint16_t))]);
                }
            } else if constexpr (KType
                                 == utils::kernelInstrType::avx512_ymm_32_reg) {
                for (int i = 0; i < numMaskRegs; i++) {
                    fringeMask[i] = Opmask(i + 1);
                }
            } else if constexpr (KType
                                 == utils::kernelInstrType::avx2_ymm_16_reg) {
                // ymmMask index is already set in setPostOpsContext()
            } else {
                // Currently returning not supported, will have to add
                // the required mask parameter for each configuration as
                // the support gets enabled.
                return jitGeneratorError::notSupported;
            }
        }
    } else {
        // This is an algorithm type not supported by our JIT kernels
        // Returning error code appropriately
        return jitGeneratorError::notSupported;
    }
    // set mask0 and mask1 for avx512_zmm_32_reg to be used for matadd and
    // matmul post-ops in all algorithms
    if constexpr (KType != utils::kernelInstrType::avx2_ymm_16_reg) {
        mask0 = Opmask(numMaskRegs + 1);
        mask1 = Opmask(numMaskRegs + 2);
    }
    auto retVal = this->dispatchKernelOps<kernelOpsGeneratorX86>(kernelOps);

    return retVal;
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::setPostOpsContext(int  MR_param,
                                                int  NR_param,
                                                bool useMask_param,
                                                int  numMaskRegs_param,
                                                int  cRegStartIdx_param,
                                                int  cRegCount_param)
{
    // Update member variables with provided parameters
    this->MR           = MR_param;
    this->NR           = NR_param;
    this->useMask      = useMask_param;
    this->numMaskRegs  = numMaskRegs_param;
    this->cRegStartIdx = cRegStartIdx_param;
    this->cRegCount    = cRegCount_param;

    // Clear the scratch register queue before repopulating. This is essential
    //  when setPostOpsContext() is called multiple times.
    while (!scratch_reg_queue.empty()) {
        scratch_reg_queue.pop();
    }

    int numElemsPerReg = Traits::regBytes / sizeof(float);
    numFullRegsPerRow  = (algoType_ == dlp::jit::jitAlgoType::gemv_n1)
                             ? (MR / numElemsPerReg)
                             : (NR / numElemsPerReg);
    numMaskRegsPerRow  = useMask ? numMaskRegs : 0;
    numRegsPerRow      = numFullRegsPerRow + numMaskRegsPerRow;
    // Assuming that we will always use registers from last for
    // accumulators.For example, if we need 24 accumulators, we will use
    // registers from 8 to 31. and the rest will be used for scratch registers.

    // pushing all the scratch registers to the queue
    for (int i = 0; i < cRegStartIdx; i++) {
        scratch_reg_queue.push(RegType(i));
    }

    // if any leftover registers are present after the accumulators,
    // push them to the queue.
    for (int i = cRegStartIdx + cRegCount; i < numRegs; i++) {
        scratch_reg_queue.push(RegType(i));
    }

    if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
        int totalRegsNeeded = cRegCount + numRegsPerRow + (useMask ? 1 : 0);
        if (totalRegsNeeded > Traits::numRegs) {
            return jitGeneratorError::badKernelInfo;
        }

        if (useMask) {
            // Making sure that ymm0 is unused as it contains the mask array for
            // n dimension. since this is hardcoded in f32_gemm_generator.cc and
            // f32_gemm_rd_generator.cc, we need to pop it from the scratch
            // register queue.
            ymmMask = popAndGetScratchReg();
        }
        scratchBcstRegIdx = useMask ? 1 : 0;
    } else {
        // check if MR and NR values are correct
        if (algoType_ == dlp::jit::jitAlgoType::gemv_m1) {
            if ((cRegCount != (NR / numElemsPerReg))
                || (cRegCount + numRegsPerRow >= Traits::numRegs)) {
                return jitGeneratorError::badKernelInfo;
            }
        } else if (algoType_ == dlp::jit::jitAlgoType::gemv_n1) {
            if ((cRegCount != (MR / numElemsPerReg))
                || (cRegCount + numRegsPerRow >= Traits::numRegs)) {
                return jitGeneratorError::badKernelInfo;
            }
        } else if (algoType_ == dlp::jit::jitAlgoType::gemm) {
            if ((MR * numRegsPerRow != cRegCount)
                || (cRegCount + numRegsPerRow >= Traits::numRegs)) {
                return jitGeneratorError::badKernelInfo;
            }
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
    // Check if tables have already been embedded to avoid code bloat
    // This is especially important for GEMV n=1/m=1 where this function
    // is called multiple times (main loop + fringe sections)
    if (tablesEmbedded) {
        return jitGeneratorError::success;
    }

    // The extra jump is to ensure none of the embedded gelu constants are
    // executed like they are instructions.
    // Using instance-specific Xbyak::Label to avoid any potential conflicts
    // with string labels in multi-threaded kernel generation scenarios
    jit_->jmp(table_store_end, jit_->T_NEAR);
    jit_->align(64);
    jit_->L(tables);
    jit_->db(reinterpret_cast<uint8_t*>(&gelu_consts), sizeof(gelu_consts));
    jit_->db(reinterpret_cast<uint8_t*>(&gelu_macros), sizeof(gelu_macros));
    jit_->db(reinterpret_cast<uint8_t*>(&lpgemm_exp), sizeof(lpgemm_exp));
    jit_->db(reinterpret_cast<uint8_t*>(&erf_consts), sizeof(erf_consts));
    jit_->db(reinterpret_cast<uint8_t*>(&lpgemm_erf), sizeof(lpgemm_erf));
    jit_->db(reinterpret_cast<uint8_t*>(&erf_f32_coeffs_hex),
             sizeof(erf_f32_coeffs_hex));
    jit_->db(reinterpret_cast<uint8_t*>(&erf_f32_constants_hex),
             sizeof(erf_f32_constants_hex));
    jit_->L(table_store_end);

    // Mark tables as embedded
    tablesEmbedded = true;

    return jitGeneratorError::success;
}

// ============================================================================
// Generic datatype visitor for compile-time type dispatch
// ============================================================================
template<typename Func>
dlp::jit::jitGeneratorError
dispatchByDatatype(dlp::kernel_frame::DataType dt, Func&& func)
{
    switch (dt) {
        case DataType::f32:
            return func(float{});
        case DataType::bf16:
            return func(bfloat16{});
        case DataType::s8:
            return func(int8_t{});
        case DataType::u8:
            return func(uint8_t{});
        case DataType::s32:
            return func(int32_t{});
        default:
            return jitGeneratorError::notSupported;
    }
}

// Dual datatype dispatch helper function
template<typename Func>
dlp::jit::jitGeneratorError
dispatch2Datatypes(dlp::kernel_frame::DataType dt1,
                   dlp::kernel_frame::DataType dt2,
                   Func&&                      func)
{
    return dispatchByDatatype(dt1, [&](auto t1) {
        return dispatchByDatatype(dt2, [&](auto t2) { return func(t1, t2); });
    });
}

// Triple datatype dispatch helper function
template<typename Func>
dlp::jit::jitGeneratorError
dispatch3Datatypes(dlp::kernel_frame::DataType dt1,
                   dlp::kernel_frame::DataType dt2,
                   dlp::kernel_frame::DataType dt3,
                   Func&&                      func)
{
    return dispatchByDatatype(dt1, [&](auto t1) {
        return dispatchByDatatype(dt2, [&](auto t2) {
            return dispatchByDatatype(
                dt3, [&](auto t3) { return func(t1, t2, t3); });
        });
    });
}

template<utils::kernelInstrType KType>
template<typename T>
int
kernelOpsGeneratorX86<KType>::getLoadBytes()
{
    if constexpr (std::is_same_v<T, bfloat16>) {
        return Traits::regBytes / 2; // Loads 16 bytes, expands to 32
    } else if constexpr (std::is_same_v<T, int8_t>
                         || std::is_same_v<T, uint8_t>) {
        return Traits::regBytes / 4; // Loads 8 bytes, expands to 32
    } else {
        return Traits::regBytes;
    }
}

template<utils::kernelInstrType KType>
template<typename T>
void
kernelOpsGeneratorX86<KType>::loadAndConvertRows(Xbyak::Reg64 addressReg,
                                                 int          regStartIdx)
{
    int loadBytes = getLoadBytes<T>();
    // Load full registers using direct addressing
    for (int i = 0; i < numFullRegsPerRow; i++) {
        loadAndConvertVector<T>(RegType(regStartIdx + i),
                                jit_->ptr[addressReg + i * loadBytes], false);
    }

    // Load masked register if needed
    if (useMask) {
        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            for (int i = numFullRegsPerRow; i < numRegsPerRow; i++) {
                loadAndConvertVector<T>(RegType(regStartIdx + i),
                                        jit_->ptr[addressReg + i * loadBytes],
                                        true, i - numFullRegsPerRow);
            }
        } else {
            loadAndConvertVector<T>(
                RegType(regStartIdx + numFullRegsPerRow),
                jit_->ptr[addressReg + numFullRegsPerRow * loadBytes], true);
        }
    }
}

// Helper functions for Broadcast and convertion into float
template<utils::kernelInstrType KType>
template<typename T>
void
kernelOpsGeneratorX86<KType>::broadcastAndConvertScalar(RegType        bcstReg,
                                                        Xbyak::Address src)
{
    if constexpr (std::is_same_v<T, float>) {
        jit_->vbroadcastss(bcstReg, src);
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        RegType tmpReg = popAndGetScratchReg();
        // Broadcast 16-bit bfloat16 value into XMM register
        // Convert 16-bit bfloat16 to 32-bit float
        // Shift left 16 bits to move to upper half of float32
        jit_->vpbroadcastw(tmpReg, src);
        jit_->vpmovsxwd(bcstReg, halfRegType(tmpReg));
        jit_->vpslld(bcstReg, bcstReg, 16);
        scratch_reg_queue.push(tmpReg);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        // Broadcast 32-bit integer value into XMM register
        // Convert 32-bit integer to 32-bit float
        jit_->vpbroadcastd(bcstReg, src);
        jit_->vcvtdq2ps(bcstReg, bcstReg);
    } else if constexpr (std::is_same_v<T, int8_t>) {
        RegType tmpReg = popAndGetScratchReg();
        // Broadcast 8-bit integer value into XMM register
        // Convert 8-bit integer to 32-bit float
        jit_->vpbroadcastb(tmpReg, src);
        jit_->vpmovsxbd(bcstReg, halfRegType(tmpReg));
        jit_->vcvtdq2ps(bcstReg, bcstReg);
        scratch_reg_queue.push(tmpReg);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        RegType tmpReg = popAndGetScratchReg();
        // Broadcast 8-bit integer value into XMM register
        // Convert 8-bit integer to 32-bit float
        jit_->vpbroadcastb(tmpReg, src);
        jit_->vpmovzxbd(bcstReg, halfRegType(tmpReg));
        jit_->vcvtdq2ps(bcstReg, bcstReg);
        scratch_reg_queue.push(tmpReg);
    }
}

/*
    Helper function to load and convert a vector of data from memory into a
    register. This function handles loading the data in a given datatype, and
    convert to float for both full and masked loads. Any new datatype conversion
    needs to be added here.
*/
template<utils::kernelInstrType KType>
template<typename T>
void
kernelOpsGeneratorX86<KType>::loadAndConvertVector(RegType        destReg,
                                                   Xbyak::Address src,
                                                   bool           useMaskOp,
                                                   int            fringeMaskId)
{
    if constexpr (std::is_same_v<T, float>) {
        // Float: Direct load, no conversion needed
        if (useMaskOp) {
            if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
                jit_->vmaskmovps(destReg, ymmMask, src);
            } else {
                jit_->vmovups(destReg | fringeMask[fringeMaskId] | jit_->T_z,
                              src);
            }
        } else {
            jit_->vmovups(destReg, src);
        }

    } else if constexpr (std::is_same_v<T, bfloat16>) {
        // BF16: Load 16-bit values, extend to 32-bit, shift left 16 bits
        RegType tmpReg    = popAndGetScratchReg();
        int     tmpRegIdx = tmpReg.getIdx();

        if (useMaskOp) {
            if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
                // AVX2 Masked Load for BF16:
                // Emulate masked load using per-lane scalar loads with
                // conditional insertion. Key: Must compute full address
                // (base+offset) into regTmp5 to preserve offset when loading
                // from ptr[base + offset] expressions.

                jit_->lea(regTmp5, src); // Compute address into GPR
                jit_->vxorps(Xbyak::Xmm(tmpRegIdx), Xbyak::Xmm(tmpRegIdx),
                             Xbyak::Xmm(tmpRegIdx)); // Zero destination

                Xbyak::Reg32 regMaskBits = regTmp4.cvt32();
                jit_->vmovmskps(regMaskBits, ymmMask); // Extract mask bits

                // Loop through each of the 8 lanes:
                //    - Test if the corresponding mask bit is set
                //    - If set, load the 16-bit bfloat16 value from memory and
                //    insert it into the correct lane position using pinsrw
                //    - If not set, skip the lane (leaving it as zero)
                // This approach ensures only masked lanes are loaded from
                // memory
                for (int lane = 0; lane < 8; lane++) {
                    jit_->inLocalLabel();
                    Xbyak::Label skipLane;
                    jit_->bt(regMaskBits, lane);
                    jit_->jnc(skipLane);
                    jit_->pinsrw(halfRegType(tmpRegIdx),
                                 jit_->word[regTmp5 + lane * sizeof(bfloat16)],
                                 lane);
                    jit_->L(skipLane);
                    jit_->outLocalLabel();
                }
            } else {
                // AVX512: Use native masked load with zeroing
                jit_->vmovdqu16(
                    halfRegType(tmpRegIdx) | fringeMask[0] | jit_->T_z, src);
            }
        } else {
            // Full Load: Load all elements
            if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
                // AVX2: Load 16 bytes (8 bf16) into XMM half of register
                jit_->vmovdqu(halfRegType(tmpRegIdx), src);
            } else {
                // AVX512: Use native bf16 load instruction
                jit_->vmovdqu16(halfRegType(tmpRegIdx), src);
            }
        }

        // Convert BF16 to F32:
        // 1. Sign-extend 16-bit to 32-bit (halfReg → fullReg)
        // 2. Shift left 16 bits (move to upper half of float32)
        jit_->vpmovsxwd(destReg, halfRegType(tmpRegIdx));
        jit_->vpslld(destReg, destReg, 16);

        scratch_reg_queue.push(tmpReg);

    } else if constexpr (std::is_same_v<T, int8_t>) {
        // INT8: Load 8-bit values, convert to float32
        RegType tmpReg    = popAndGetScratchReg();
        int     tmpRegIdx = tmpReg.getIdx();

        if (useMaskOp) {
            if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
                // AVX2 Masked Load for INT8 (same approach as BF16, but 8-bit
                // loads)
                jit_->lea(regTmp5, src);
                jit_->vpxor(Xbyak::Xmm(tmpRegIdx), Xbyak::Xmm(tmpRegIdx),
                            Xbyak::Xmm(tmpRegIdx));

                Xbyak::Reg32 regMaskBits = regTmp4.cvt32();
                jit_->vmovmskps(regMaskBits, ymmMask);

                for (int lane = 0; lane < 8; lane++) {
                    jit_->inLocalLabel();
                    Xbyak::Label skipLane;
                    jit_->bt(regMaskBits, lane);
                    jit_->jnc(skipLane);
                    jit_->pinsrb(Xbyak::Xmm(tmpRegIdx),
                                 jit_->byte[regTmp5 + lane * sizeof(int8_t)],
                                 lane);
                    jit_->L(skipLane);
                    jit_->outLocalLabel();
                }
            } else {
                // AVX512: Use masked load into XMM with zeroing
                jit_->vmovdqu8(
                    Xbyak::Xmm(tmpRegIdx) | fringeMask[0] | jit_->T_z, src);
            }
        } else {
            // Full Load: Load all elements
            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                // AVX512: Load 16 bytes into XMM (vpmovsxbd reads XMM source)
                jit_->vmovdqu8(Xbyak::Xmm(tmpRegIdx), src);
            } else {
                // AVX2: Load 8 bytes (8 int8) into XMM
                jit_->vmovq(Xbyak::Xmm(tmpRegIdx), src);
            }
        }

        // Convert S8 to F32:
        // 1. Sign-extend 8-bit to 32-bit (reads XMM, writes ZMM)
        // 2. Convert int32 to float32
        jit_->vpmovsxbd(destReg, Xbyak::Xmm(tmpRegIdx));
        jit_->vcvtdq2ps(destReg, destReg);

        scratch_reg_queue.push(tmpReg);

    } else if constexpr (std::is_same_v<T, uint8_t>) {
        // UINT8: Similar to INT8 but zero-extend
        RegType tmpReg    = popAndGetScratchReg();
        int     tmpRegIdx = tmpReg.getIdx();

        if (useMaskOp) {
            if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
                // AVX2 Masked Load for UINT8 (same as INT8, but zero-extended
                // later)
                jit_->lea(regTmp5, src);
                jit_->vpxor(Xbyak::Xmm(tmpRegIdx), Xbyak::Xmm(tmpRegIdx),
                            Xbyak::Xmm(tmpRegIdx));

                Xbyak::Reg32 regMaskBits = regTmp4.cvt32(); // eax
                jit_->vmovmskps(regMaskBits, ymmMask);

                for (int lane = 0; lane < 8; lane++) {
                    jit_->inLocalLabel();
                    Xbyak::Label skipLane;
                    jit_->bt(regMaskBits, lane);
                    jit_->jnc(skipLane);
                    jit_->pinsrb(Xbyak::Xmm(tmpRegIdx),
                                 jit_->byte[regTmp5 + lane * sizeof(uint8_t)],
                                 lane);
                    jit_->L(skipLane);
                    jit_->outLocalLabel();
                }
            } else {
                // AVX512: Use masked load into XMM with zeroing
                jit_->vmovdqu8(
                    Xbyak::Xmm(tmpRegIdx) | fringeMask[0] | jit_->T_z, src);
            }
        } else {
            // Full Load: Load all elements
            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                // AVX512: Load 16 bytes into XMM (vpmovzxbd reads XMM source)
                jit_->vmovdqu8(Xbyak::Xmm(tmpRegIdx), src);
            } else {
                // AVX2: Load 8 bytes (8 uint8) into XMM
                jit_->vmovq(Xbyak::Xmm(tmpRegIdx), src);
            }
        }

        // Convert U8 to F32:
        // 1. Zero-extend 8-bit to 32-bit (reads XMM, writes ZMM)
        // 2. Convert int32 to float32
        jit_->vpmovzxbd(destReg, Xbyak::Xmm(tmpRegIdx));
        jit_->vcvtdq2ps(destReg, destReg);

        scratch_reg_queue.push(tmpReg);

    } else if constexpr (std::is_same_v<T, int32_t>) {
        // INT32: Load 32-bit values, convert to float32
        if (useMaskOp) {
            if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
                jit_->vpmaskmovd(destReg, ymmMask, src);
            } else {
                jit_->vmovdqu32(destReg | fringeMask[0] | jit_->T_z, src);
            }
        } else {
            if constexpr (KType == utils::kernelInstrType::avx2_ymm_16_reg) {
                jit_->vmovdqu(destReg, src);
            } else {
                jit_->vmovdqu32(destReg, src);
            }
        }

        // Convert int32 to float32
        jit_->vcvtdq2ps(destReg, destReg);
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

// ============================================================================
// GEMM Bias Implementations
// Formula: C = C + Bias
// ============================================================================

template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::biasRowMajorImplGEMM()
{
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(T)]);
    jit_->add(regTmp1, regTmp2);

    // Reserve destination registers to prevent them from being allocated as
    // scratch during BF16/S8/U8 conversions in loadRowGeneric
    auto saved_queue = reserveDestRegisters(scratchLoadRegIdx, numRegsPerRow);

    // Load bias values into registers using abstracted helper
    loadAndConvertRows<T>(regTmp1, scratchLoadRegIdx);

    // Add bias to accumulators
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < numRegsPerRow; j++) {
            jit_->vaddps(RegType(cRegStartIdx + i * numRegsPerRow + j),
                         RegType(cRegStartIdx + i * numRegsPerRow + j),
                         RegType(scratchLoadRegIdx + j));
        }
    }

    // Restore original scratch queue
    restoreScratchQueue(saved_queue);
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::biasColMajorImplGEMM()
{
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(T)]);
    jit_->add(regTmp1, regTmp2);

    // Utilizing all the scratch registers for bias ensuring
    // maximum possible independent instructions.
    for (int i = 0; i < MR; i++) {
        RegType bcstReg = popAndGetScratchReg();
        broadcastAndConvertScalar<T>(bcstReg,
                                     jit_->ptr[regTmp1 + i * sizeof(T)]);
        for (int j = 0; j < numRegsPerRow; j++) {
            jit_->vaddps(RegType(cRegStartIdx + i * numRegsPerRow + j),
                         RegType(cRegStartIdx + i * numRegsPerRow + j),
                         bcstReg);
        }
        scratch_reg_queue.push(bcstReg);
    }
    return jitGeneratorError::success;
}

// ============================================================================
// GEMV n=1 Bias Implementations
// ============================================================================

template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::biasRowMajorImplGEMVN1()
{
    // GEMV n=1 with row-major bias ('r' or 'R' in op_args2):
    // Bias is scalar - broadcast single value to all MR outputs
    // No offset needed - regTmp1 already points to the correct bias value

    RegType biasReg = popAndGetScratchReg();
    broadcastAndConvertScalar<T>(biasReg, jit_->ptr[regTmp1]);

    // Add to all accumulator registers (MR outputs packed across full + masked
    // registers)
    for (int i = 0; i < numRegsPerRow; i++) {
        jit_->vaddps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                     biasReg);
    }

    scratch_reg_queue.push(biasReg);
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::biasColMajorImplGEMVN1()
{
    // GEMV n=1 with column-major bias ('c' or 'C' in op_args2):
    // Bias is vector - one value per output element (MR values)
    // Need to offset by post_op_c_i to get to correct position

    // Calculate offset: bias_ptr + post_op_c_i * sizeof(T)
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(T)]);
    jit_->add(regTmp1, regTmp2);

    // Reserve destination registers to prevent corruption during BF16/S8/U8
    // conversions
    auto saved_queue = reserveDestRegisters(scratchLoadRegIdx, numRegsPerRow);

    // Load bias values into registers using abstracted helper
    loadAndConvertRows<T>(regTmp1, scratchLoadRegIdx);

    // Add bias to all accumulator registers - MR outputs packed
    // across full + masked registers
    for (int i = 0; i < numRegsPerRow; i++) {
        jit_->vaddps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                     RegType(scratchLoadRegIdx + i));
    }

    // Restore original scratch queue
    restoreScratchQueue(saved_queue);
    return jitGeneratorError::success;
}

// ============================================================================
// Unified Row-Major Bias Implementation with Dequantization Support
// Formula: C = C + SF * (Bias - ZP)
// ============================================================================

template<utils::kernelInstrType KType>
template<typename BiasDt, typename SfDt, typename ZpDt>
jitGeneratorError
kernelOpsGeneratorX86<KType>::biasRowMajorImplUnified(kernelOpsMetaData& op,
                                                      bool               hasSF,
                                                      bool               hasZP)
{
    bool isGEMVN1 = (algoType_ == dlp::jit::jitAlgoType::gemv_n1);
    // regTmp1: bias pointer (op_args1)
    jit_->mov(regTmp1,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args1)]);

    // regTmp2: offset index (post_op_c_j for row-major)
    jit_->mov(regTmp2, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_j)]);

    // Load SF and ZP pointers (only for dequantization)
    if (hasSF) {
        jit_->mov(regTmp3, jit_->ptr[regkernelOpsList
                                     + offsetof(lpgemm_post_op, scale_factor)]);
    }
    if (hasZP) {
        jit_->mov(
            regTmp6,
            jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, bias_zp)]);
    }

    // Calculate load bytes for each datatype
    int biasLoadBytes = getLoadBytes<BiasDt>();
    int sfLoadBytes   = getLoadBytes<SfDt>();
    int zpLoadBytes   = getLoadBytes<ZpDt>();

    // Calculate bias pointer offset first
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(BiasDt)]);
    jit_->add(regTmp1, regTmp2);

    // Apply post_op_c_j offset to SF pointer (for vector scale factors)
    if (hasSF && !op.scalarScaleFactorRequired) {
        // Reload post_op_c_j and apply SF-specific scaling
        jit_->mov(regTmp2,
                  jit_->ptr[regkernelOpsAttr
                            + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
        jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(SfDt)]);
        jit_->add(regTmp3, regTmp2);
    }

    // Apply post_op_c_j offset to ZP pointer (for vector zero points)
    if (hasZP && !op.scalarZeroPointRequired) {
        // Reload post_op_c_j and apply ZP-specific scaling
        jit_->mov(regTmp2,
                  jit_->ptr[regkernelOpsAttr
                            + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
        jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(ZpDt)]);
        jit_->add(regTmp6, regTmp2);
    }

    // Element-wise strategy: reserve only 1 register for safety
    auto    saved_queue = reserveDestRegisters(scratchLoadRegIdx, 1);
    RegType tempReg     = RegType(scratchLoadRegIdx);

    // Pre-load scalar values for scale factor and zero point
    RegType scalarSFReg, scalarZPReg;
    if (hasSF && op.scalarScaleFactorRequired) {
        scalarSFReg = popAndGetScratchReg();
        broadcastAndConvertScalar<SfDt>(scalarSFReg, jit_->ptr[regTmp3]);
    }

    if (hasZP && op.scalarZeroPointRequired) {
        scalarZPReg = popAndGetScratchReg();
        broadcastAndConvertScalar<ZpDt>(scalarZPReg, jit_->ptr[regTmp6]);
    }

    // ============================================================
    // GEMV N=1 PATH
    // ============================================================
    if (isGEMVN1) {

        for (int i = 0; i < numRegsPerRow; i++) {
            int  accumIdx  = cRegStartIdx + i;
            bool useMaskOp = useMask && (i >= numFullRegsPerRow);

            // Step 1: Load Bias (broadcast scalar - same for all MR rows)
            broadcastAndConvertScalar<BiasDt>(tempReg, jit_->ptr[regTmp1]);

            // Step 2: Subtract Zero Point (if present)
            if (hasZP) {
                if (op.scalarZeroPointRequired) {
                    jit_->vsubps(tempReg, tempReg, scalarZPReg);
                } else {
                    RegType zpTempReg = popAndGetScratchReg();
                    loadAndConvertVector<ZpDt>(
                        zpTempReg, jit_->ptr[regTmp6 + i * zpLoadBytes],
                        useMaskOp);
                    jit_->vsubps(tempReg, tempReg, zpTempReg);
                    scratch_reg_queue.push(zpTempReg);
                }
            }

            // Step 3: Multiply by Scale Factor (if present)
            if (hasSF) {
                if (op.scalarScaleFactorRequired) {
                    jit_->vmulps(tempReg, tempReg, scalarSFReg);
                } else {
                    RegType sfTempReg = popAndGetScratchReg();
                    loadAndConvertVector<SfDt>(
                        sfTempReg, jit_->ptr[regTmp3 + i * sfLoadBytes],
                        useMaskOp);
                    jit_->vmulps(tempReg, tempReg, sfTempReg);
                    scratch_reg_queue.push(sfTempReg);
                }
            }

            // Step 4: Add to accumulator
            jit_->vaddps(RegType(accumIdx), RegType(accumIdx), tempReg);
        }
    }
    // ============================================================
    // GEMM / GEMV M=1 PATH
    // ============================================================
    else {
        for (int i = 0; i < MR; i++) {
            for (int j = 0; j < numRegsPerRow; j++) {
                int  accumIdx  = cRegStartIdx + i * numRegsPerRow + j;
                bool useMaskOp = useMask && (j >= numFullRegsPerRow);

                // Step 1: Load Bias (vector - different per column segment)
                loadAndConvertVector<BiasDt>(
                    tempReg, jit_->ptr[regTmp1 + j * biasLoadBytes], useMaskOp);

                // Step 2: Subtract Zero Point (if present)
                if (hasZP) {
                    if (op.scalarZeroPointRequired) {
                        jit_->vsubps(tempReg, tempReg, scalarZPReg);
                    } else {
                        RegType zpTempReg = popAndGetScratchReg();
                        loadAndConvertVector<ZpDt>(
                            zpTempReg, jit_->ptr[regTmp6 + j * zpLoadBytes],
                            useMaskOp);
                        jit_->vsubps(tempReg, tempReg, zpTempReg);
                        scratch_reg_queue.push(zpTempReg);
                    }
                }

                // Step 3: Multiply by Scale Factor (if present)
                if (hasSF) {
                    if (op.scalarScaleFactorRequired) {
                        jit_->vmulps(tempReg, tempReg, scalarSFReg);
                    } else {
                        RegType sfTempReg = popAndGetScratchReg();
                        loadAndConvertVector<SfDt>(
                            sfTempReg, jit_->ptr[regTmp3 + j * sfLoadBytes],
                            useMaskOp);
                        jit_->vmulps(tempReg, tempReg, sfTempReg);
                        scratch_reg_queue.push(sfTempReg);
                    }
                }

                // Step 4: Add to accumulator
                jit_->vaddps(RegType(accumIdx), RegType(accumIdx), tempReg);
            }
        }
    }

    // ============================================================
    // CLEANUP
    // ============================================================
    if (hasZP && op.scalarZeroPointRequired) {
        scratch_reg_queue.push(scalarZPReg);
    }
    if (hasSF && op.scalarScaleFactorRequired) {
        scratch_reg_queue.push(scalarSFReg);
    }

    restoreScratchQueue(saved_queue);

    return jitGeneratorError::success;
}

// ============================================================================
// Unified Column-Major Bias Implementation with Dequantization Support
// ============================================================================

template<utils::kernelInstrType KType>
template<typename BiasDt, typename SfDt, typename ZpDt>
jitGeneratorError
kernelOpsGeneratorX86<KType>::biasColMajorImplUnified(kernelOpsMetaData& op,
                                                      bool               hasSF,
                                                      bool               hasZP)
{
    bool isGEMVN1 = (algoType_ == dlp::jit::jitAlgoType::gemv_n1);
    // regTmp1: bias pointer (op_args1)
    jit_->mov(regTmp1,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args1)]);

    // regTmp2: offset index (post_op_c_i for col-major)
    jit_->mov(regTmp2, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_i)]);

    // Load SF and ZP pointers (only for dequantization)
    if (hasSF) {
        jit_->mov(regTmp3, jit_->ptr[regkernelOpsList
                                     + offsetof(lpgemm_post_op, scale_factor)]);
    }
    if (hasZP) {
        jit_->mov(
            regTmp6,
            jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, bias_zp)]);
    }

    // Calculate load bytes for each datatype
    int biasLoadBytes = getLoadBytes<BiasDt>();
    int sfLoadBytes   = getLoadBytes<SfDt>();
    int zpLoadBytes   = getLoadBytes<ZpDt>();

    // Calculate bias pointer offset first
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(BiasDt)]);
    jit_->add(regTmp1, regTmp2);

    // Apply post_op_c_i offset to SF pointer (for vector scale factors)
    if (hasSF && !op.scalarScaleFactorRequired) {
        // Reload post_op_c_i and apply SF-specific scaling
        jit_->mov(regTmp2,
                  jit_->ptr[regkernelOpsAttr
                            + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
        jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(SfDt)]);
        jit_->add(regTmp3, regTmp2);
    }

    // Apply post_op_c_i offset to ZP pointer (for vector zero points)
    if (hasZP && !op.scalarZeroPointRequired) {
        // Reload post_op_c_i and apply ZP-specific scaling
        jit_->mov(regTmp2,
                  jit_->ptr[regkernelOpsAttr
                            + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
        jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(ZpDt)]);
        jit_->add(regTmp6, regTmp2);
    }

    // Element-wise strategy: reserve only 1 register
    auto    saved_queue = reserveDestRegisters(scratchLoadRegIdx, 1);
    RegType tempReg     = RegType(scratchLoadRegIdx);

    RegType scalarSFReg, scalarZPReg;
    if (hasSF && op.scalarScaleFactorRequired) {
        scalarSFReg = popAndGetScratchReg();
        broadcastAndConvertScalar<SfDt>(scalarSFReg, jit_->ptr[regTmp3]);
    }

    if (hasZP && op.scalarZeroPointRequired) {
        scalarZPReg = popAndGetScratchReg();
        broadcastAndConvertScalar<ZpDt>(scalarZPReg, jit_->ptr[regTmp6]);
    }

    // ============================================================
    // GEMV N=1 PATH
    // ============================================================
    if (isGEMVN1) {

        for (int i = 0; i < numRegsPerRow; i++) {
            int  accumIdx  = cRegStartIdx + i;
            bool useMaskOp = useMask && (i >= numFullRegsPerRow);

            // Step 1: Load Bias (scalar per row - broadcast)
            broadcastAndConvertScalar<BiasDt>(tempReg, jit_->ptr[regTmp1]);

            // Step 2: Subtract Zero Point (if present)
            if (hasZP) {
                if (op.scalarZeroPointRequired) {
                    jit_->vsubps(tempReg, tempReg, scalarZPReg);
                } else {
                    RegType zpTempReg = popAndGetScratchReg();
                    loadAndConvertVector<ZpDt>(
                        zpTempReg, jit_->ptr[regTmp6 + i * zpLoadBytes],
                        useMaskOp);
                    jit_->vsubps(tempReg, tempReg, zpTempReg);
                    scratch_reg_queue.push(zpTempReg);
                }
            }

            // Step 3: Multiply by Scale Factor (if present)
            if (hasSF) {
                if (op.scalarScaleFactorRequired) {
                    jit_->vmulps(tempReg, tempReg, scalarSFReg);
                } else {
                    RegType sfTempReg = popAndGetScratchReg();
                    loadAndConvertVector<SfDt>(
                        sfTempReg, jit_->ptr[regTmp3 + i * sfLoadBytes],
                        useMaskOp);
                    jit_->vmulps(tempReg, tempReg, sfTempReg);
                    scratch_reg_queue.push(sfTempReg);
                }
            }

            // Step 4: Add to accumulator
            jit_->vaddps(RegType(accumIdx), RegType(accumIdx), tempReg);
        }
    }
    // ============================================================
    // GEMM / GEMV M=1 PATH (Column-major bias)
    // ============================================================
    // In column-major, bias is per-row (one value per M row, broadcast to all
    // N columns). SF and ZP (if vector) are also per-row - one value per row.
    else {
        for (int i = 0; i < MR; i++) {

            // Step 1: Load Bias (scalar for this row, broadcast)
            broadcastAndConvertScalar<BiasDt>(
                tempReg, jit_->ptr[regTmp1 + i * sizeof(BiasDt)]);

            // Step 2: Subtract Zero Point (if present)
            // For column-major with vector ZP: each row has one ZP value
            if (hasZP) {
                if (op.scalarZeroPointRequired) {
                    jit_->vsubps(tempReg, tempReg, scalarZPReg);
                } else {
                    // Vector ZP for column-major: one ZP per row, broadcast it
                    RegType zpTempReg = popAndGetScratchReg();
                    broadcastAndConvertScalar<ZpDt>(
                        zpTempReg, jit_->ptr[regTmp6 + i * sizeof(ZpDt)]);
                    jit_->vsubps(tempReg, tempReg, zpTempReg);
                    scratch_reg_queue.push(zpTempReg);
                }
            }

            // Step 3: Multiply by Scale Factor (if present)
            // For column-major with vector SF: each row has one SF value
            if (hasSF) {
                if (op.scalarScaleFactorRequired) {
                    jit_->vmulps(tempReg, tempReg, scalarSFReg);
                } else {
                    // Vector SF for column-major: one SF per row, broadcast it
                    RegType sfTempReg = popAndGetScratchReg();
                    broadcastAndConvertScalar<SfDt>(
                        sfTempReg, jit_->ptr[regTmp3 + i * sizeof(SfDt)]);
                    jit_->vmulps(tempReg, tempReg, sfTempReg);
                    scratch_reg_queue.push(sfTempReg);
                }
            }

            // Step 4: Add to all accumulators in this row (all columns)
            for (int j = 0; j < numRegsPerRow; j++) {
                int accumIdx = cRegStartIdx + i * numRegsPerRow + j;
                jit_->vaddps(RegType(accumIdx), RegType(accumIdx), tempReg);
            }
        }
    }

    // ============================================================
    // CLEANUP
    // ============================================================
    if (hasZP && op.scalarZeroPointRequired) {
        scratch_reg_queue.push(scalarZPReg);
    }
    if (hasSF && op.scalarScaleFactorRequired) {
        scratch_reg_queue.push(scalarSFReg);
    }

    restoreScratchQueue(saved_queue);

    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::bias(kernelOpsMetaData& op)
{
    using namespace dlp::kernel_frame;

    // Check for dequantization (SF/ZP)
    bool hasSF = (op.scalarScaleFactorRequired || op.vectorScaleFactorRequired);
    bool hasZP = (op.scalarZeroPointRequired || op.vectorZeroPointRequired);

    DataType biasDt = op.paramStorageDt;

    // Regular BIAS (no dequantization) - Executing the normal BIAS path
    if (!hasSF && !hasZP) {
        // Load pointers into temporary registers needed by simple BIAS
        // functions.
        // regTmp1: bias pointer (op_args1)
        jit_->mov(
            regTmp1,
            jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args1)]);

        // regTmp2: offset index (post_op_c_j for row-major, post_op_c_i for
        // col-major)
        if (op.cMatFormat == storageFormat::rowMajor) {
            jit_->mov(regTmp2,
                      jit_->ptr[regkernelOpsAttr
                                + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
        } else {
            jit_->mov(regTmp2,
                      jit_->ptr[regkernelOpsAttr
                                + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
        }

        // Only dispatch on bias datatype - no SF/ZP involved
        bool isGEMVN1 = (algoType_ == dlp::jit::jitAlgoType::gemv_n1);

        if (op.cMatFormat == storageFormat::rowMajor) {
            return dispatchByDatatype(biasDt, [&](auto bt) {
                if (isGEMVN1) {
                    return biasRowMajorImplGEMVN1<decltype(bt)>();
                } else {
                    return biasRowMajorImplGEMM<decltype(bt)>();
                }
            });
        } else {
            return dispatchByDatatype(biasDt, [&](auto bt) {
                if (isGEMVN1) {
                    return biasColMajorImplGEMVN1<decltype(bt)>();
                } else {
                    return biasColMajorImplGEMM<decltype(bt)>();
                }
            });
        }
    }

    // BIAS with dequantization.
    DataType sfDt = op.scaleFactorDt;
    DataType zpDt = op.zeroPointDt;

    // Using default types when SF or ZP is not present to avoid dispatch
    // failure. The actual SF/ZP code paths are guarded by hasSF/hasZP guards,
    // so the type doesn't matter when not present just need a valid type for
    // dispatch.
    if (!hasSF) {
        sfDt =
            DataType::f32; // Default placeholder, never used if hasSF is false
    }
    if (!hasZP) {
        zpDt =
            DataType::f32; // Default placeholder, never used if hasZP is false
    }

    // Triple dispatch for dequantization
    if (op.cMatFormat == storageFormat::rowMajor) {
        return dispatch3Datatypes(
            biasDt, sfDt, zpDt, [&](auto bt, auto st, auto zt) {
                return biasRowMajorImplUnified<decltype(bt), decltype(st),
                                               decltype(zt)>(op, hasSF, hasZP);
            });
    } else {
        return dispatch3Datatypes(
            biasDt, sfDt, zpDt, [&](auto bt, auto st, auto zt) {
                return biasColMajorImplUnified<decltype(bt), decltype(st),
                                               decltype(zt)>(op, hasSF, hasZP);
            });
    }
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
    broadcastAndConvertScalar<T>(scaleReg, jit_->ptr[regTmp1]);

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
    broadcastAndConvertScalar<T>(minReg, jit_->ptr[regTmp1]);
    broadcastAndConvertScalar<T>(maxReg, jit_->ptr[regTmp2]);

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
    // Broadcast the scale factor using abstracted helper
    broadcastAndConvertScalar<T>(sfReg, jit_->ptr[regTmp1]);

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

    // Reserve destination registers to prevent them from being allocated as
    // scratch during BF16/S8/U8 conversions in loadRowGeneric
    auto saved_queue = reserveDestRegisters(scratchLoadRegIdx, numRegsPerRow);

    // Load scale factor values into registers using abstracted helper
    loadAndConvertRows<T>(regTmp3, scratchLoadRegIdx);

    for (int i = 0; i < numRegsPerRow; i++) {
        for (int j = 0; j < MR; j++) {
            jit_->vmulps(RegType(cRegStartIdx + j * numRegsPerRow + i),
                         RegType(cRegStartIdx + j * numRegsPerRow + i),
                         RegType(scratchLoadRegIdx + i));
        }
    }

    // Restore original scratch queue
    restoreScratchQueue(saved_queue);
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
        broadcastAndConvertScalar<T>(bcstReg,
                                     jit_->ptr[regTmp3 + i * sizeof(T)]);
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
    broadcastAndConvertScalar<T>(zpReg, jit_->ptr[regTmp1]);

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

    // Reserve destination registers to prevent them from being allocated as
    // scratch during BF16/S8/U8 conversions in loadRowGeneric
    auto saved_queue = reserveDestRegisters(scratchLoadRegIdx, numRegsPerRow);

    // Load zero point values into registers using abstracted helper
    loadAndConvertRows<T>(regTmp3, scratchLoadRegIdx);

    for (int i = 0; i < numRegsPerRow; i++) {
        for (int j = 0; j < MR; j++) {
            jit_->vaddps(RegType(cRegStartIdx + j * numRegsPerRow + i),
                         RegType(cRegStartIdx + j * numRegsPerRow + i),
                         RegType(scratchLoadRegIdx + i));
        }
    }

    // Restore original scratch queue
    restoreScratchQueue(saved_queue);
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
        broadcastAndConvertScalar<T>(bcstReg,
                                     jit_->ptr[regTmp3 + i * sizeof(T)]);
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
kernelOpsGeneratorX86<KType>::scaleFactorImpl(kernelOpsMetaData& op)
{
    jit_->mov(
        regTmp1,
        jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, scale_factor)]);

    if (algoType_ == dlp::jit::jitAlgoType::gemv_n1) {
        // ============ GEMV n=1 path ============
        // Row-major: scale_factor_len is always 1 (scalar broadcast)
        // Column-major: scale_factor_len can be 1 (scalar) or > 1 (vector load)

        if (op.cMatFormat == storageFormat::rowMajor) {
            // Row-major: always scalar (array is [1, n=1])
            DISPATCH_BY_DATATYPE(op.scaleFactorDt, scaleFactorScalarImplGEMVN1);
        } else {
            // Column-major: check if scalar or vector
            if (op.scalarScaleFactorRequired) {
                // Scalar: broadcast single value
                DISPATCH_BY_DATATYPE(op.scaleFactorDt,
                                     scaleFactorScalarImplGEMVN1);
            } else {
                // Vector: load m scale factors with mask
                jit_->mov(
                    regTmp2,
                    jit_->ptr[regkernelOpsAttr
                              + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
                DISPATCH_BY_DATATYPE(op.scaleFactorDt,
                                     scaleFactorColMajorImplGEMVN1);
            }
        }
    } else {
        // ============ GEMM path ============
        if (op.scalarScaleFactorRequired) {
            // Scalar: broadcast single scale factor
            DISPATCH_BY_DATATYPE(op.scaleFactorDt, scaleFactorScalarImpl);
        } else if (op.cMatFormat == storageFormat::rowMajor) {
            // Row-major: load vector of scale factors per column
            DISPATCH_BY_DATATYPE(op.scaleFactorDt, scaleFactorRowMajorImpl);
        } else {
            // Column-major: broadcast scale factor per row
            DISPATCH_BY_DATATYPE(op.scaleFactorDt, scaleFactorColMajorImpl);
        }
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::zeroPointImpl(kernelOpsMetaData& op)
{
    jit_->mov(regTmp1,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args1)]);

    if (algoType_ == dlp::jit::jitAlgoType::gemv_n1) {
        // ============ GEMV n=1 path ============
        // Row-major: zp_len is always 1 (scalar broadcast)
        // Column-major: zp_len can be 1 (scalar) or > 1 (vector load)

        if (op.cMatFormat == storageFormat::rowMajor) {
            // Row-major: always scalar (array is [1, n=1])
            DISPATCH_BY_DATATYPE(op.zeroPointDt, zeroPointScalarImplGEMVN1);
        } else {
            // Column-major: check if scalar or vector
            if (op.scalarZeroPointRequired) {
                // Scalar: broadcast single value
                DISPATCH_BY_DATATYPE(op.zeroPointDt, zeroPointScalarImplGEMVN1);
            } else {
                // Vector: load m zero points with mask
                jit_->mov(
                    regTmp2,
                    jit_->ptr[regkernelOpsAttr
                              + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
                DISPATCH_BY_DATATYPE(op.zeroPointDt,
                                     zeroPointColMajorImplGEMVN1);
            }
        }
    } else {
        // ============ GEMM path ============
        if (op.scalarZeroPointRequired) {
            // Scalar: broadcast single zero point
            DISPATCH_BY_DATATYPE(op.zeroPointDt, zeroPointScalarImpl);
        } else if (op.cMatFormat == storageFormat::rowMajor) {
            // Row-major: load vector of zero points per column
            DISPATCH_BY_DATATYPE(op.zeroPointDt, zeroPointRowMajorImpl);
        } else {
            // Column-major: broadcast zero point per row
            DISPATCH_BY_DATATYPE(op.zeroPointDt, zeroPointColMajorImpl);
        }
    }
    return jitGeneratorError::success;
}

// Scale factor and zero point implementations for GEMV n=1
template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::scaleFactorScalarImplGEMVN1()
{
    // GEMV n=1 with scalar scale factor
    // MR outputs are packed across cRegCount accumulator registers

    RegType sfReg = popAndGetScratchReg();

    // Broadcast the scalar scale factor
    broadcastAndConvertScalar<T>(sfReg, jit_->ptr[regTmp1]);

    // Apply to all accumulator registers - MR outputs packed
    // across full + masked registers
    for (int i = 0; i < numRegsPerRow; i++) {
        jit_->vmulps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                     sfReg);
    }

    scratch_reg_queue.push(sfReg);
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::scaleFactorColMajorImplGEMVN1()
{
    // GEMV n=1 with column-major: vector scale factors
    // Load MR scale factors starting at scale_factor[post_op_c_i]
    // regTmp2 already contains post_op_c_i

    // Calculate offset: scale_factor + post_op_c_i * sizeof(T)
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(T)]);
    jit_->add(regTmp1, regTmp2);

    // Reserve destination registers to prevent corruption during BF16/S8/U8
    // conversions
    auto saved_queue = reserveDestRegisters(scratchLoadRegIdx, numRegsPerRow);

    // Load scale factor values into registers using abstracted helper
    loadAndConvertRows<T>(regTmp1, scratchLoadRegIdx);

    // Apply scale factor to all accumulator registers (MR outputs packed across
    // full + masked registers)
    for (int i = 0; i < numRegsPerRow; i++) {
        jit_->vmulps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                     RegType(scratchLoadRegIdx + i));
    }

    // Restore original scratch queue
    restoreScratchQueue(saved_queue);
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::zeroPointScalarImplGEMVN1()
{
    // GEMV n=1 with scalar zero point
    // MR outputs are packed across cRegCount accumulator registers
    // Apply zero point to all registers

    RegType zpReg = popAndGetScratchReg();

    // Broadcast the scalar zero point
    broadcastAndConvertScalar<T>(zpReg, jit_->ptr[regTmp1]);

    // Apply to all accumulator registers - MR outputs packed
    // across full + masked registers
    for (int i = 0; i < numRegsPerRow; i++) {
        jit_->vaddps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                     zpReg);
    }

    scratch_reg_queue.push(zpReg);
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::zeroPointColMajorImplGEMVN1()
{
    // GEMV n=1 with column-major: vector zero points
    // Load MR zero points starting at zero_point[post_op_c_i]
    // regTmp2 already contains post_op_c_i

    // Calculate offset: zero_point + post_op_c_i * sizeof(T)
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(T)]);
    jit_->add(regTmp1, regTmp2);

    // Reserve destination registers to prevent corruption during BF16/S8/U8
    // conversions
    auto saved_queue = reserveDestRegisters(scratchLoadRegIdx, numRegsPerRow);

    // Load zero point values into registers using abstracted helper
    loadAndConvertRows<T>(regTmp1, scratchLoadRegIdx);

    // Apply zero point to all accumulator registers (MR outputs packed across
    // full + masked registers)
    for (int i = 0; i < numRegsPerRow; i++) {
        jit_->vaddps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                     RegType(scratchLoadRegIdx + i));
    }

    // Restore original scratch queue
    restoreScratchQueue(saved_queue);
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::downscale(kernelOpsMetaData& op)
{
    jitGeneratorError err_sf = scaleFactorImpl(op);
    jitGeneratorError err_zp = zeroPointImpl(op);
    if (err_sf != jitGeneratorError::success
        || err_zp != jitGeneratorError::success) {
        return jitGeneratorError::notSupported;
    }
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::resetMasks(bool mask, int idx)
{
    if (mask) {
        if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
            // For Zmm: 16-bit mask, split into 8 bits each
            // Load lower 8 bits of maskF32 (covers elements 0-7)
            jit_->kmovb(mask0, fringeMask[idx]);
            // Load upper 8 bits of maskF32 (covers elements 8-15)
            jit_->kshiftrw(mask1, fringeMask[idx], 8);
        } else if constexpr (KType
                             == utils::kernelInstrType::avx512_ymm_32_reg) {
            // For Ymm: 8-bit mask, split into 4 bits each
            // Load lower 4 bits of maskF32 (covers elements 0-3)
            // Note: kmovb loads 8 bits, but instruction will only use lower 4
            jit_->kmovb(mask0, fringeMask[idx]);
            // Load upper 4 bits of maskF32 (covers elements 4-7)
            jit_->kshiftrw(mask1, fringeMask[idx], 4);
        } else {
            // saved for future use
        }
    } else {
        jit_->kxnorw(mask0, mask0, mask0);
        jit_->kxnorw(mask1, mask1, mask1);
    }
}

template<utils::kernelInstrType KType>
template<typename sfDt, typename matOpDt>
jitGeneratorError
kernelOpsGeneratorX86<KType>::matOpScaleFactorImplMerged(matOpType      opType,
                                                         matOpScaleType sclType,
                                                         bool           hasSF)
{
    // Float types need 5 ZMM spare registers minimum (with SF).
    // Non-float types (int8/uint8/bf16/int32) need 7 ZMM spare registers
    // minimum due to additional dword offset conversion requirements.
    // When hasSF is false, we need 1 less register (no sf_reg needed).
    // TODO: Implement a strategy like gelu_tanh post-ops using
    // stack in case sufficient registers are not available.

    int minRegsRequired = hasSF ? 5 : 4; // Default for float
    if constexpr (!std::is_same_v<matOpDt, float>) {
        minRegsRequired = hasSF ? 7 : 6;
    }

    if (cRegStartIdx < minRegsRequired) {
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // Reserve registers from global scratch queue to prevent corruption
    // Float: Uses 5 registers: matRegIdx(0), sf_reg(1), offsets1RegIdx(2),
    //        offsets2RegIdx(3), scratch3RegIdx(4)
    // Non-float: Uses 7 registers: Above 5 + dwordOffsets1(5), dwordOffsets2(6)
    auto saved_global_queue = reserveDestRegisters(0, minRegsRequired);

    std::queue<int> scratch_reg_queue;
    for (int i = 0; i < cRegStartIdx; i++) {
        scratch_reg_queue.push(i);
    }

    int matRegIdx = scratch_reg_queue.front();
    scratch_reg_queue.pop();

    jit_->mov(regTmp7, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
    jit_->mov(regTmp6, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_i)]);

    int sf_reg      = -1;
    int sfLoadBytes = 0;
    if (hasSF) {
        sf_reg = scratch_reg_queue.front();
        scratch_reg_queue.pop();

        if (sclType == matOpScaleType::scalar) {
            broadcastAndConvertScalar<sfDt>(RegType(sf_reg), jit_->ptr[regTmp1]);
        } else if (sclType == matOpScaleType::columnVector) {
            jit_->lea(regTmp1, jit_->ptr[regTmp1 + regTmp6 * sizeof(sfDt)]);
        } else { // row-vector
            jit_->lea(regTmp1, jit_->ptr[regTmp1 + regTmp7 * sizeof(sfDt)]);
        }
        sfLoadBytes = getLoadBytes<sfDt>();
    }

    // load ldm into regTmp3 and multiply with sizeof(dt)
    jit_->mov(regTmp3,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args3)]);
    // ldm is pointer, need to dereference again to get actual ldm value.
    jit_->mov(regTmp3, jit_->ptr[regTmp3]);
    jit_->lea(regTmp3, jit_->ptr[regTmp3 * sizeof(matOpDt)]);

    // regTmp2 = matPtr
    jit_->mov(regTmp2,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args1)]);

    jit_->inLocalLabel();

    // Load csC and check if csC is 1
    jit_->cmp(regcsC, 1);
    jit_->je(".rowMajorMatOp", jit_->T_NEAR);

    Xbyak::Label offsets_label;

    int64_t offsets[16] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    };

    jit_->jmp(".offset_end", jit_->T_NEAR);
    jit_->align(64);
    jit_->L(offsets_label);
    jit_->db(reinterpret_cast<uint8_t*>(&offsets), sizeof(offsets));
    jit_->L(".offset_end");

    // take two zmm from scratch_reg_queue and load the offsets
    int offsets1RegIdx = scratch_reg_queue.front();
    scratch_reg_queue.pop();
    int offsets2RegIdx = scratch_reg_queue.front();
    scratch_reg_queue.pop();
    int scratch3RegIdx = scratch_reg_queue.front();
    scratch_reg_queue.pop();

    // load offsets into register
    jit_->vmovdqu32(RegType(offsets1RegIdx),
                    jit_->ptr[jit_->rip + offsets_label]);
    jit_->vmovdqu32(RegType(offsets2RegIdx),
                    jit_->ptr[jit_->rip + offsets_label + RegBytes]);

    // broadcast the ldm*sizeof(float) value and multiply with offsets.
    jit_->vpbroadcastq(RegType(scratch3RegIdx), regTmp3);

    // now multiply offsets with ldm
    jit_->vpmullq(RegType(offsets1RegIdx), RegType(offsets1RegIdx),
                  RegType(scratch3RegIdx));
    jit_->vpmullq(RegType(offsets2RegIdx), RegType(offsets2RegIdx),
                  RegType(scratch3RegIdx));

    scratch_reg_queue.push(scratch3RegIdx);

    jit_->lea(regTmp6, jit_->ptr[regTmp6 * sizeof(matOpDt)]);

    // matptr += post_op_c_j * ldm + post_op_c_i
    jit_->imul(regTmp7, regTmp3);
    jit_->add(regTmp7, regTmp6);
    jit_->add(regTmp2, regTmp7);

    // calculate 16*ldm*sizeof(float) and store in regTmp3
    jit_->lea(regTmp3, jit_->ptr[regTmp3 * 8]);
    if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
        jit_->lea(regTmp3, jit_->ptr[regTmp3 * 2]);
    }

    // take two zmm from scratch_reg_queue
    // for scatter/gather operations
    int scratchReg2 = scratch_reg_queue.front();
    scratch_reg_queue.pop();

    // Additional scratch registers for dword offset conversion (non-float
    // types) These must be allocated ONCE outside the lambda, not per-call
    int dwordOffsets1 = -1;
    int dwordOffsets2 = -1;
    if constexpr (!std::is_same_v<matOpDt, float>) {
        dwordOffsets1 = scratch_reg_queue.front();
        scratch_reg_queue.pop();
        dwordOffsets2 = scratch_reg_queue.front();
        scratch_reg_queue.pop();
    }

    auto opLambdaColMat = [&](int sfRegIdx, int mask_idx, int accumRegIdx,
                              bool useMask) -> jitGeneratorError {
        if constexpr (std::is_same_v<matOpDt, float>) {
            resetMasks(useMask, mask_idx);
            jit_->vgatherqps(halfRegType(matRegIdx) | mask0,
                             jit_->ptr[regTmp4 + RegType(offsets1RegIdx) * 1]);
            jit_->vgatherqps(halfRegType(scratchReg2) | mask1,
                             jit_->ptr[regTmp4 + RegType(offsets2RegIdx) * 1]);
            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                jit_->vinsertf32x8(RegType(matRegIdx), RegType(matRegIdx),
                                   halfRegType(scratchReg2), 1);
            } else if constexpr (KType
                                 == utils::kernelInstrType::avx512_ymm_32_reg) {
                jit_->vinsertf32x4(RegType(matRegIdx), RegType(matRegIdx),
                                   halfRegType(scratchReg2), 1);
            } else {
                // saved for future use
            }
        } else if constexpr (std::is_same_v<matOpDt, int8_t>
                             || std::is_same_v<matOpDt, uint8_t>
                             || std::is_same_v<matOpDt, bfloat16>
                             || std::is_same_v<matOpDt, int32_t>) {
            // Load matOp elements from column-major layout
            resetMasks(useMask, mask_idx);

            // Convert qword offsets to dword offsets for vpgatherdd
            // offsets1RegIdx (ZMM, 8 qwords) → 8 dwords in dwordOffsets1 (YMM)
            // offsets2RegIdx (ZMM, 8 qwords) → 8 dwords in dwordOffsets2 (YMM)
            jit_->vpmovqd(halfRegType(dwordOffsets1), RegType(offsets1RegIdx));
            jit_->vpmovqd(halfRegType(dwordOffsets2), RegType(offsets2RegIdx));

            // Non-float types: Use vpgatherdd to gather dwords, then convert to
            // float Gather dwords at byte offsets (vpgatherdd uses dword
            // indices)
            jit_->vpgatherdd(
                halfRegType(matRegIdx) | mask0,
                jit_->ptr[regTmp4 + halfRegType(dwordOffsets1) * 1]);
            jit_->vpgatherdd(
                halfRegType(scratchReg2) | mask1,
                jit_->ptr[regTmp4 + halfRegType(dwordOffsets2) * 1]);

            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                // Combine the two halves into one ZMM register
                jit_->vinserti32x8(RegType(matRegIdx), RegType(matRegIdx),
                                   halfRegType(scratchReg2), 1);
            } else if constexpr (KType
                                 == utils::kernelInstrType::avx512_ymm_32_reg) {
                // Combine the two halves into one YMM register
                jit_->vinserti32x4(RegType(matRegIdx), RegType(matRegIdx),
                                   halfRegType(scratchReg2), 1);
            } else {
                // saved for future use
            }

            // Datatype-specific conversion to float32
            if constexpr (std::is_same_v<matOpDt, int8_t>) {
                // INT8: Extract byte, sign-extend, and convert to float
                jit_->vpslld(RegType(matRegIdx), RegType(matRegIdx), 24);
                jit_->vpsrad(RegType(matRegIdx), RegType(matRegIdx), 24);
                jit_->vcvtdq2ps(RegType(matRegIdx), RegType(matRegIdx));

            } else if constexpr (std::is_same_v<matOpDt, uint8_t>) {
                // UINT8: Extract byte, zero-extend, and convert to float
                jit_->vpslld(RegType(matRegIdx), RegType(matRegIdx), 24);
                jit_->vpsrld(RegType(matRegIdx), RegType(matRegIdx), 24);
                jit_->vcvtdq2ps(RegType(matRegIdx), RegType(matRegIdx));

            } else if constexpr (std::is_same_v<matOpDt, bfloat16>) {
                // BF16: Shift left 16 bits to convert to float32 format
                jit_->vpslld(RegType(matRegIdx), RegType(matRegIdx), 16);

            } else if constexpr (std::is_same_v<matOpDt, int32_t>) {
                // INT32: Convert directly to float32
                jit_->vcvtdq2ps(RegType(matRegIdx), RegType(matRegIdx));
            }
        } else {
            return jitGeneratorError::notSupported;
        }

        // multiply scale factor with matOp (only if hasSF)
        if (hasSF) {
            jit_->vmulps(RegType(matRegIdx), RegType(matRegIdx), RegType(sfRegIdx));
        }
        if (opType == matOpType::matOpAdd) {
            jit_->vaddps(RegType(accumRegIdx), RegType(accumRegIdx),
                         RegType(matRegIdx));
        } else if (opType == matOpType::matOpMul) {
            jit_->vmulps(RegType(accumRegIdx), RegType(accumRegIdx),
                         RegType(matRegIdx));
        }

        return jitGeneratorError::success;
    };

    for (int i = 0; i < MR; i++) {
        if (hasSF && sclType == matOpScaleType::columnVector) {
            broadcastAndConvertScalar<sfDt>(
                RegType(sf_reg), jit_->ptr[regTmp1 + i * sizeof(sfDt)]);
        }
        jit_->mov(regTmp4, regTmp2);
        for (int j = 0; j < numFullRegsPerRow; j++) {
            if (hasSF && sclType == matOpScaleType::rowVector) {
                loadAndConvertVector<sfDt>(RegType(sf_reg),
                                           jit_->ptr[regTmp1 + j * sfLoadBytes],
                                           false);
            }

            jitGeneratorError err = opLambdaColMat(
                sf_reg, 0, (cRegStartIdx + (i * numRegsPerRow) + j), false);
            if (err != jitGeneratorError::success) {
                return err;
            }

            // move to next 16 elements.
            jit_->add(regTmp4, regTmp3);
        }
        if (numMaskRegsPerRow > 0) {
            for (int k = 0; k < numMaskRegsPerRow; k++) {

                if (hasSF && sclType == matOpScaleType::rowVector) {
                    loadAndConvertVector<sfDt>(
                        RegType(sf_reg),
                        jit_->ptr[regTmp1 + numFullRegsPerRow * sfLoadBytes],
                        false);
                }

                jitGeneratorError err =
                    opLambdaColMat(sf_reg, k,
                                   (cRegStartIdx + (i * numRegsPerRow)
                                    + numFullRegsPerRow + k),
                                   true);
                if (err != jitGeneratorError::success) {
                    return err;
                }

                // move to next 16 elements.
                jit_->add(regTmp4, regTmp3);
            }
        }
        // move matadd pointer to next row
        jit_->add(regTmp2, sizeof(matOpDt));
    }

    scratch_reg_queue.push(offsets1RegIdx);
    scratch_reg_queue.push(offsets2RegIdx);
    scratch_reg_queue.push(scratchReg2);

    jit_->jmp(".endOfMatOp", jit_->T_NEAR);
    jit_->L(".rowMajorMatOp");

    if (hasSF && sclType == matOpScaleType::rowVector) {
        loadAndConvertRows<sfDt>(regTmp1, sf_reg);
    }

    jit_->lea(regTmp7, jit_->ptr[regTmp7 * sizeof(matOpDt)]);

    // matptr += post_op_c_i * ldm + post_op_c_j
    jit_->imul(regTmp6, regTmp3);
    jit_->add(regTmp7, regTmp6);
    jit_->add(regTmp2, regTmp7);

    // Calculate load bytes based on matrix datatype (not scale factor!)
    int matLoadBytes = getLoadBytes<matOpDt>();

    auto opLambdaRowMat = [&](int sfRegIdx, int matRegIdx, int accumRegIdx,
                              int sclIdx, int loadIdx, bool useMask,
                              int fringeMaskId = 0) -> jitGeneratorError {
        // load matOp elements - supports all datatypes via loadAndConvertVector
        loadAndConvertVector<matOpDt>(RegType(matRegIdx),
                                      jit_->ptr[regTmp2 + loadIdx], useMask,
                                      fringeMaskId);

        // multiply scale factor with matOp (only if hasSF)
        if (hasSF) {
            jit_->vmulps(RegType(matRegIdx), RegType(matRegIdx),
                         RegType(sfRegIdx + sclIdx));
        }
        if (opType == matOpType::matOpAdd) {
            if (useMask) {
                jit_->vaddps(RegType(accumRegIdx) | fringeMask[fringeMaskId],
                             RegType(accumRegIdx), RegType(matRegIdx));
            } else {
                jit_->vaddps(RegType(accumRegIdx), RegType(accumRegIdx),
                             RegType(matRegIdx));
            }
        } else if (opType == matOpType::matOpMul) {
            if (useMask) {
                jit_->vmulps(RegType(accumRegIdx) | fringeMask[fringeMaskId],
                             RegType(accumRegIdx), RegType(matRegIdx));
            } else {
                jit_->vmulps(RegType(accumRegIdx), RegType(accumRegIdx),
                             RegType(matRegIdx));
            }
        }
        return jitGeneratorError::success;
    };

    int sclIdx = 0;
    for (int i = 0; i < MR; i++) {
        if (hasSF && sclType == matOpScaleType::columnVector) {
            broadcastAndConvertScalar<sfDt>(
                RegType(sf_reg), jit_->ptr[regTmp1 + i * sizeof(sfDt)]);
        }
        for (int j = 0; j < numFullRegsPerRow; j++) {
            if (hasSF && sclType == matOpScaleType::rowVector) {
                sclIdx = j;
            }

            jitGeneratorError err = opLambdaRowMat(
                sf_reg, matRegIdx, (cRegStartIdx + (i * numRegsPerRow) + j),
                sclIdx, (j * matLoadBytes), false);
            if (err != jitGeneratorError::success) {
                return err;
            }
        }
        if (numMaskRegsPerRow > 0) {
            if constexpr (KType == utils::kernelInstrType::avx512_zmm_32_reg) {
                for (int mIdx = numFullRegsPerRow; mIdx < numRegsPerRow;
                     mIdx++) {
                    if (hasSF && sclType == matOpScaleType::rowVector) {
                        sclIdx = mIdx;
                    }
                    jitGeneratorError err = opLambdaRowMat(
                        sf_reg, matRegIdx,
                        (cRegStartIdx + (i * numRegsPerRow) + mIdx), sclIdx,
                        (mIdx * matLoadBytes), true, mIdx - numFullRegsPerRow);

                    if (err != jitGeneratorError::success) {
                        return err;
                    }
                }
            } else {
                if (hasSF && sclType == matOpScaleType::rowVector) {
                    sclIdx = numFullRegsPerRow;
                }
                jitGeneratorError err = opLambdaRowMat(
                    sf_reg, matRegIdx,
                    (cRegStartIdx + (i * numRegsPerRow) + numFullRegsPerRow),
                    sclIdx, (numFullRegsPerRow * matLoadBytes), true);

                if (err != jitGeneratorError::success) {
                    return err;
                }
            }
        }
        // add ldm to matadd pointer
        jit_->add(regTmp2, regTmp3);
    }

    jit_->L(".endOfMatOp");

    if (hasSF) {
        scratch_reg_queue.push(sf_reg);
    }
    scratch_reg_queue.push(matRegIdx);

    // Restore global scratch queue
    restoreScratchQueue(saved_global_queue);

    jit_->outLocalLabel();
    return jitGeneratorError::success;
}

template<>
template<typename sfDt, typename matOpDt>
jitGeneratorError
kernelOpsGeneratorX86<utils::kernelInstrType::avx2_ymm_16_reg>::
    matOpScaleFactorImplMerged(matOpType opType, matOpScaleType sclType,
                               bool hasSF)
{
    // Reserve registers to prevent corruption during BF16/S8/U8 conversions
    auto saved_queue =
        reserveDestRegisters(scratchLoadRegIdx, numRegsPerRow + 2);

    md_t sf_reg = -1;
    if (hasSF) {
        sf_reg = scratchLoadRegIdx;
        if (sclType == matOpScaleType::scalar) {
            broadcastAndConvertScalar<sfDt>(RegType(sf_reg), jit_->ptr[regTmp1]);
        }
    }

    jit_->mov(regTmp7, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
    jit_->mov(regTmp6, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_i)]);

    // Load scale factors FIRST - use sizeof(sfDt) for offset calculation
    if (hasSF && sclType == matOpScaleType::rowVector) {
        jit_->lea(regTmp3, jit_->ptr[regTmp1 + regTmp7 * sizeof(sfDt)]);
        loadAndConvertRows<sfDt>(regTmp3, sf_reg);
    }

    // NOW scale regTmp7 for matrix data offset - use sizeof(matOpDt)
    jit_->lea(regTmp7, jit_->ptr[regTmp7 * sizeof(matOpDt)]);

    // regTmp2 = matPtr
    jit_->mov(regTmp2,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args1)]);
    // regTmp3 = ldm
    jit_->mov(regTmp3,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args3)]);
    // ldm is pointer, need to dereference again to get actual ldm value.
    jit_->mov(regTmp3, jit_->ptr[regTmp3]);
    jit_->lea(regTmp3, jit_->ptr[regTmp3 * sizeof(matOpDt)]);

    jit_->imul(regTmp6, regTmp3);
    jit_->add(regTmp7, regTmp6);
    jit_->add(regTmp2, regTmp7);

    if (hasSF && sclType == matOpScaleType::columnVector) {
        jit_->mov(regTmp6,
                  jit_->ptr[regkernelOpsAttr
                            + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
        jit_->lea(regTmp6, jit_->ptr[regTmp6 * sizeof(sfDt)]);
        jit_->add(regTmp6, regTmp1);
    }

    // Calculate load bytes based on matrix datatype
    int matLoadBytes = getLoadBytes<matOpDt>();

    auto opLambda = [&](matOpType opType, int sfRegIdx, int matRegIdx,
                        int accumRegIdx, int sclIdx, int loadIdx,
                        bool useMask) -> jitGeneratorError {
        // load matOp elements - supports all datatypes via loadAndConvertVector
        loadAndConvertVector<matOpDt>(RegType(matRegIdx),
                                      jit_->ptr[regTmp2 + loadIdx], useMask);

        // multiply scale factor with matOp (only if hasSF)
        if (hasSF) {
            jit_->vmulps(RegType(matRegIdx), RegType(matRegIdx),
                         RegType(sfRegIdx + sclIdx));
        }
        if (opType == matOpType::matOpAdd) {
            if (useMask) {
                // AVX2 doesn't support masking on arithmetic ops, zero out
                // garbage lanes
                jit_->vandps(RegType(matRegIdx), RegType(matRegIdx), ymmMask);
                jit_->vaddps(RegType(accumRegIdx), RegType(accumRegIdx),
                             RegType(matRegIdx));
            } else {
                jit_->vaddps(RegType(accumRegIdx), RegType(accumRegIdx),
                             RegType(matRegIdx));
            }
        } else if (opType == matOpType::matOpMul) {
            if (useMask) {
                // AVX2 doesn't support masking on arithmetic ops, use blend
                jit_->vmulps(RegType(matRegIdx), RegType(accumRegIdx),
                             RegType(matRegIdx));
                jit_->vblendvps(RegType(accumRegIdx), RegType(accumRegIdx),
                                RegType(matRegIdx), ymmMask);
            } else {
                jit_->vmulps(RegType(accumRegIdx), RegType(accumRegIdx),
                             RegType(matRegIdx));
            }
        }
        return jitGeneratorError::success;
    };

    int sclIdx = 0;
    for (int i = 0; i < MR; i++) {
        if (hasSF && sclType == matOpScaleType::columnVector) {
            broadcastAndConvertScalar<sfDt>(
                RegType(sf_reg), jit_->ptr[regTmp6 + i * sizeof(sfDt)]);
        }
        for (int j = 0; j < numFullRegsPerRow; j++) {
            if (hasSF && sclType == matOpScaleType::rowVector) {
                sclIdx = j;
            }
            jitGeneratorError err =
                opLambda(opType, sf_reg, scratchBcstRegIdx,
                         (cRegStartIdx + (i * numRegsPerRow) + j), sclIdx,
                         (j * matLoadBytes), false);
            if (err != jitGeneratorError::success) {
                return err;
            }
        }
        if (numMaskRegsPerRow > 0) {
            if (hasSF && sclType == matOpScaleType::rowVector) {
                sclIdx = numFullRegsPerRow;
            }
            jitGeneratorError err = opLambda(
                opType, sf_reg, scratchBcstRegIdx,
                (cRegStartIdx + (i * numRegsPerRow) + numFullRegsPerRow),
                sclIdx, (numFullRegsPerRow * matLoadBytes), true);
            if (err != jitGeneratorError::success) {
                return err;
            }
        }
        // add ldm to matadd pointer
        jit_->add(regTmp2, regTmp3);
    }

    // Restore original scratch queue
    restoreScratchQueue(saved_queue);
    return jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
template<typename sfDt, typename matOpDt>
jitGeneratorError
kernelOpsGeneratorX86<KType>::matOpScaleFactorImplGEMVN1(matOpType      opType,
                                                         matOpScaleType sclType,
                                                         bool           hasSF)
{
    // GEMV n=1: MR outputs packed across accumulator registers
    // ldm is always 1 for n=1 case (contiguous storage)

    // Check register availability before proceeding
    int minRegsNeeded = 1; // Base: matReg
    if (hasSF) {
        minRegsNeeded++; // sfReg
        if constexpr (!std::is_same_v<sfDt, float>) {
            minRegsNeeded++; // Extra temp for scale factor conversion
        }
    }
    if constexpr (!std::is_same_v<matOpDt, float>) {
        minRegsNeeded++; // Extra temp for matrix data conversion
    }

    if (static_cast<int>(scratch_reg_queue.size()) < minRegsNeeded) {
        return jitGeneratorError::badKernelInfo;
    }

    // Reserve all accumulator registers to prevent conflicts
    auto saved_queue = reserveDestRegisters(cRegStartIdx, numRegsPerRow + 2);

    RegType matReg = popAndGetScratchReg();

    jit_->mov(regTmp7, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_j)]);
    jit_->mov(regTmp6, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_i)]);

    // Calculate load bytes based on matrix datatype
    int matLoadBytes = getLoadBytes<matOpDt>();

    // ============ GEMV n=1: Always process M elements as a vector ============
    // For n=1, auxiliary matrix is [M, 1] with contiguous elements
    // No need to check csC - always load M elements

    // Get matrix operation data pointer (op_args1)
    jit_->mov(regTmp2,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args1)]);

    // Add offset for post_op_c_i (starting row)
    jit_->lea(regTmp3, jit_->ptr[regTmp6 * sizeof(matOpDt)]);
    jit_->add(regTmp2, regTmp3);

    // Handle scale factor setup and loading (only if hasSF)
    int sf_reg = -1;
    RegType sfReg       = RegType(sf_reg);
    int     sfLoadBytes = 0;
    if (hasSF) {
        sfReg       = popAndGetScratchReg();
        sfLoadBytes = getLoadBytes<sfDt>();

        if (sclType == matOpScaleType::scalar) {
            broadcastAndConvertScalar<sfDt>(sfReg, jit_->ptr[regTmp1]);
        } else if (sclType == matOpScaleType::rowVector) {
            jit_->lea(regTmp1, jit_->ptr[regTmp1 + regTmp7 * sizeof(sfDt)]);
            broadcastAndConvertScalar<sfDt>(sfReg, jit_->ptr[regTmp1]);
        } else {
            jit_->lea(regTmp1, jit_->ptr[regTmp1 + regTmp6 * sizeof(sfDt)]);
        }
    }

    // Process full accumulator registers
    for (int i = 0; i < numFullRegsPerRow; i++) {
        // Load matrix data
        loadAndConvertVector<matOpDt>(
            matReg, jit_->ptr[regTmp2 + i * matLoadBytes], false);

        if (hasSF && sclType == matOpScaleType::columnVector) {
            loadAndConvertVector<sfDt>(
                sfReg, jit_->ptr[regTmp1 + i * sfLoadBytes], false);
        }

        if (hasSF) {
            jit_->vmulps(matReg, matReg, sfReg);
        }

        if (opType == matOpType::matOpAdd) {
            jit_->vaddps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                         matReg);
        } else if (opType == matOpType::matOpMul) {
            jit_->vmulps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                         matReg);
        }
    }

    // Process masked register if needed
    if (useMask) {
        int i = numFullRegsPerRow;

        // Load matrix data with masking
        loadAndConvertVector<matOpDt>(
            matReg, jit_->ptr[regTmp2 + i * matLoadBytes], true);

        // Load scale factor (if column vector, load per register)
        if (hasSF && sclType == matOpScaleType::columnVector) {
            loadAndConvertVector<sfDt>(
                sfReg, jit_->ptr[regTmp1 + i * sfLoadBytes], true);
        }
        // Multiply scale factor with matrix data
        if (hasSF) {
            jit_->vmulps(matReg, matReg, sfReg);
        }

        // Apply operation to accumulator
        if (opType == matOpType::matOpAdd) {
            jit_->vaddps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                         matReg);
        } else if (opType == matOpType::matOpMul) {
            jit_->vmulps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                         matReg);
        }
    }

    // Push registers back to the queue
    scratch_reg_queue.push(matReg);
    if (hasSF) {
        scratch_reg_queue.push(sfReg);
    }

    restoreScratchQueue(saved_queue);
    return jitGeneratorError::success;
}

// ============================================================================
// Unified MatOp Implementation - Handles both matadd and matmul
// ============================================================================
template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::matOp(kernelOpsMetaData& op, matOpType opType)
{
    // Check if scale factor is required
    bool hasSF = (op.scalarScaleFactorRequired || op.vectorScaleFactorRequired);

    // Only load scale factor pointer if it exists
    if (hasSF) {
        jit_->mov(
            regTmp1,
            jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, scale_factor)]);
    }

    // f32 is used as a placeholder type when SF is not present; the actual value
    // won't be used since all SF code paths are guarded by hasSF
    DataType sfDt = hasSF ? op.scaleFactorDt : DataType::f32;

    // Determine scale type
    matOpScaleType sclType = matOpScaleType::scalar;
    if (hasSF && !op.scalarScaleFactorRequired) {
        sclType = (op.cMatFormat == storageFormat::rowMajor)
                      ? matOpScaleType::rowVector
                      : matOpScaleType::columnVector;
    }

    bool isGEMVN1 = (algoType_ == dlp::jit::jitAlgoType::gemv_n1);

    // Single dispatch - no macros, no intermediate functions
    return dispatch2Datatypes(
        sfDt, op.paramStorageDt, [&](auto sfType, auto matType) {
            using SfT  = decltype(sfType);
            using MatT = decltype(matType);
            return isGEMVN1
                       ? matOpScaleFactorImplGEMVN1<SfT, MatT>(opType, sclType,
                                                               hasSF)
                       : matOpScaleFactorImplMerged<SfT, MatT>(opType, sclType,
                                                               hasSF);
        });
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::matadd(kernelOpsMetaData& op)
{
    return matOp(op, matOpType::matOpAdd);
}

template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::matmul(kernelOpsMetaData& op)
{
    return matOp(op, matOpType::matOpMul);
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
    // In case of avx2, one register is being used for mask. So, we need to add
    // scratchBcstRegIdx to determine the number of registers to push to stack.
    md_t num_push_regs = scratchBcstRegIdx + num_post_op_regs - cRegStartIdx;

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
DLP_ALWAYS_INLINE void
kernelOpsGeneratorX86<utils::kernelInstrType::avx512_zmm_32_reg>::ERF(int y,
                                                                      int r)
{
    // Piecewise polynomial ERF approximation using VPERMT2PS
    // Uses coefficients that approximate erf(x/√2) for input x
    // Input: r = input values (zmm register index)
    // Output: y = erf(r/√2) (zmm register index)

    using RegType = Xbyak::Zmm;

    // Register allocation:
    // x = x_abs (absolute value, reused from input r)
    // r2 = indices
    // z = coefficient low
    // dn = coefficient high
    // q = poly accumulator
    // const1 = sign bits
    // const2 = temporary

    // Step 1: Get absolute value and save sign
    // x_abs = abs(r) using abs_mask (0x7FFFFFFF)
    jit_->vpbroadcastd(RegType(const2), get_erf_f32_constant(1)); // abs_mask
    jit_->vpandd(RegType(x), RegType(r), RegType(const2));

    // Save sign bit: x_sign = r & sign_mask (0x80000000)
    jit_->vpbroadcastd(RegType(const1), get_erf_f32_constant(2)); // sign_mask
    jit_->vpandd(RegType(const1), RegType(r), RegType(const1));
    // const1 now holds the sign bits

    // Step 2: Clamp input to upper bound (7.0) BEFORE index calculation
    // This prevents extrapolation and ensures indices are computed from clamped
    // values
    jit_->vbroadcastss(RegType(const2),
                       get_erf_f32_constant(3)); // rbound (7.0)
    jit_->vcmpps(jit_->k5, RegType(x), RegType(const2),
                 0x1E); // CMP_GT: k5 = (x > rbound)
    jit_->vblendmps(RegType(x) | jit_->k5, RegType(x),
                    RegType(const2)); // Select: rbound if k5, else x

    // Step 3: Compute indices using bit-based calculation (from CLAMPED x)
    // indices = (x_bits + bias) >> 21
    jit_->vpbroadcastd(RegType(r2), get_erf_f32_constant(0)); // erf_idx_bias
    jit_->vpaddd(RegType(r2), RegType(x), RegType(r2));       // Add bias
    jit_->vpsrad(RegType(r2), RegType(r2), 21); // Arithmetic shift right

    // Step 4: Clamp indices to [0, 23]
    // Note: Since input is clamped to 7.0, max index = 23, so only lower clamp
    // needed
    jit_->vpxord(RegType(const2), RegType(const2), RegType(const2)); // zero
    jit_->vpmaxsd(RegType(r2), RegType(r2), RegType(const2)); // max(indices, 0)

    // Step 5: Evaluate polynomial using VPERMT2PS and Horner's method
    // Note: vpermt2ps modifies the index register, so we copy it each time

    // Start with degree 5 (highest degree) - outside the loop
    jit_->vmovups(RegType(z), get_erf_f32_coeff_array_lo(5));  // Load c5[0:15]
    jit_->vmovups(RegType(dn), get_erf_f32_coeff_array_hi(5)); // Load c5[16:31]
    jit_->vmovdqa32(RegType(const2), RegType(r2));             // Copy indices
    jit_->vpermt2ps(RegType(z), RegType(const2), RegType(dn)); // Select c5
    jit_->vmovups(RegType(q), RegType(z));                     // poly = c5

    // Horner's method: poly = poly * x + c[deg] for deg = 4, 3, 2, 1, 0
    for (int deg = 4; deg >= 0; deg--) {
        jit_->vmovups(RegType(z), get_erf_f32_coeff_array_lo(deg));
        jit_->vmovups(RegType(dn), get_erf_f32_coeff_array_hi(deg));
        jit_->vmovdqa32(RegType(const2), RegType(r2)); // Copy indices again
        jit_->vpermt2ps(RegType(z), RegType(const2), RegType(dn));
        jit_->vfmadd213ps(RegType(q), RegType(x),
                          RegType(z)); // q = q * x + c[deg]
    }

    // Step 6: Restore sign (XOR with sign bits)
    // Works because poly is always positive (computed from |x|)
    jit_->vxorps(RegType(y), RegType(q), RegType(const1));
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

// AVX-512 specialization: Skip 1/√2 scaling since ERF coefficients handle it
template<>
DLP_ALWAYS_INLINE void
kernelOpsGeneratorX86<
    utils::kernelInstrType::avx512_zmm_32_reg>::GELU_ERF_F32_DEF(md_t reg)
{
    // For AVX-512, the ERF implementation uses coefficients that
    // approximate erf(x/√2) directly. So we pass the input without scaling.
    // ERF will compute erf(reg/√2) and return it.

    jit_->vxorps(RegType(x_tanh), RegType(x_tanh), RegType(x_tanh));

    ERF(x_tanh, reg); // Pass reg directly (no 1/√2 multiplication)

    // Compute GELU: 0.5 * reg * (1 + erf_result)
    jit_->vbroadcastss(RegType(const2), get_constant(erf_consts_off, 1)); // 1.0
    jit_->vaddps(RegType(r2), RegType(x_tanh), RegType(const2));

    jit_->vmulps(RegType(r2), RegType(r2), RegType(reg));
    jit_->vbroadcastss(RegType(const2), get_constant(erf_consts_off, 2)); // 0.5
    jit_->vmulps(RegType(reg), RegType(r2), RegType(const2));
}

// Generic implementation for other architectures (AVX2, etc.)
template<utils::kernelInstrType KType>
void
kernelOpsGeneratorX86<KType>::GELU_ERF_F32_DEF(md_t reg)
{
    // For non-AVX-512, ERF expects input pre-scaled by 1/√2
    jit_->vbroadcastss(RegType(const1),
                       get_constant(erf_consts_off, 0)); // 1/√2
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

    broadcastAndConvertScalar<T>(RegType(x_tanh), jit_->ptr[regTmp1]);

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

// ============================================================================
// A-Dequantization: Scale Factor Application (Per-Tensor)
// ============================================================================
// Formula: acc[i][j] = round(acc[i][j] * a_post_quant_sf)
// Used for per-tensor quantization (single scale factor for entire matrix).
template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::aDQuantScaleFactorScalarImpl()
{
    // Broadcast scalar scale factor (regTmp1 set by caller)
    RegType sfReg = popAndGetScratchReg();
    broadcastAndConvertScalar<T>(sfReg, jit_->ptr[regTmp1]);

    // Apply: acc = round(acc * a_post_quant_sf) for all accumulator registers
    for (int i = 0; i < MR * numRegsPerRow; i++) {
        jit_->vmulps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                     sfReg);
        jit_->vrndscaleps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                          0x00); // Round to nearest even
    }
    scratch_reg_queue.push(sfReg);
    return jitGeneratorError::success;
}

// ============================================================================
// A-Dequantization: Scale Factor Application (Per-Row)
// ============================================================================
// Formula: acc[row][col] = round(acc[row][col] * a_post_quant_sf[row])
// Used for per-row/per-channel quantization (different scale factor per row).
template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::aDQuantScaleFactorRowMajorImpl()
{
    // Calculate address: scale_factor_base + (post_op_c_i * sizeof(T))
    jit_->mov(regTmp2, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(T)]);
    jit_->lea(regTmp3, jit_->ptr[regTmp1]);
    jit_->add(regTmp3, regTmp2);

    // For each row: broadcast its scale factor and apply with rounding
    for (int i = 0; i < MR; i++) {
        RegType sfReg = popAndGetScratchReg();
        broadcastAndConvertScalar<T>(sfReg, jit_->ptr[regTmp3 + i * sizeof(T)]);

        for (int j = 0; j < numRegsPerRow; j++) {
            jit_->vmulps(RegType(cRegStartIdx + i * numRegsPerRow + j),
                         RegType(cRegStartIdx + i * numRegsPerRow + j), sfReg);
            jit_->vrndscaleps(RegType(cRegStartIdx + i * numRegsPerRow + j),
                              RegType(cRegStartIdx + i * numRegsPerRow + j),
                              0x00); // Round to nearest even
        }
        scratch_reg_queue.push(sfReg);
    }
    return jitGeneratorError::success;
}

// ============================================================
// A-Dequantization: Scale Factor Application (per-tensor) (GEMV N=1)
// ============================================================
// Formula: acc[i] = round(acc[i] * a_post_quant_sf)
// Used for GEMV N=1 with scalar scale factor.
template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::aDQuantScaleFactorScalarImplGEMVN1()
{
    // Broadcast scalar scale factor (regTmp1 set by caller)
    RegType sfReg = popAndGetScratchReg();
    broadcastAndConvertScalar<T>(sfReg, jit_->ptr[regTmp1]);

    // Apply: acc = round(acc * a_post_quant_sf) for all accumulator registers
    for (int i = 0; i < numRegsPerRow; i++) {
        jit_->vmulps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                     sfReg);
        jit_->vrndscaleps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                          0x00);
    }
    // Return scratch register to the pool
    scratch_reg_queue.push(sfReg);
    return jitGeneratorError::success;
}

// ============================================================================
// A-Dequantization: Scale Factor Application per-row (GEMV N=1)
// ============================================================================
// Formula: acc[i] = round(acc[i] * a_post_quant_sf)
// Used for GEMV N=1 with row-major scale factor.
template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::aDQuantScaleFactorRowMajorImplGEMVN1()
{
    // Calculate address: scale_factor_base + (post_op_c_i * sizeof(T))
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(T)]);
    jit_->add(regTmp1, regTmp2);

    // Reserve destination registers to prevent corruption during BF16/S8/U8
    // conversions
    auto saved_queue = reserveDestRegisters(scratchLoadRegIdx, numRegsPerRow);

    // Load scale factor values into registers using abstracted helper
    loadAndConvertRows<T>(regTmp1, scratchLoadRegIdx);

    // Apply: acc = round(acc * a_post_quant_sf) for all accumulator registers
    for (int i = 0; i < numRegsPerRow; i++) {
        jit_->vmulps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                     RegType(scratchLoadRegIdx + i));
        jit_->vrndscaleps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                          0x00); // Round to nearest even
    }
    restoreScratchQueue(saved_queue);
    return jitGeneratorError::success;
}

// ============================================================================
// A-Dequantization: Zero-Point Correction (Per-Tensor)
// ============================================================================
// Formula: acc[i][j] = acc[i][j] + (b_col_sum[j] * a_post_quant_zp)
// Compensates for zero-point bias. Used for per-tensor asymmetric quantization.
template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::aDQuantZeroPointScalarImpl()
{
    // Load B column sums pointer and calculate address for current column block
    jit_->mov(regTmp2,
              jit_->ptr[regkernelOpsAttr
                        + offsetof(lpgemm_post_op_attr, b_col_sum_vec)]);
    jit_->mov(regTmp3, jit_->ptr[regkernelOpsAttr
            + offsetof(lpgemm_post_op_attr, b_sum_offset)]);

    jit_->lea(regTmp2, jit_->ptr[regTmp2 + regTmp3 * sizeof(int32_t)]);

    // Broadcast scalar zero-point (regTmp1 set by caller)
    RegType zpReg = popAndGetScratchReg();
    broadcastAndConvertScalar<T>(zpReg, jit_->ptr[regTmp1]);

    auto saved_queue = reserveDestRegisters(scratchLoadRegIdx, numRegsPerRow);

    // Load b_col_sum and apply: acc += b_col_sum * a_post_quant_zp
    for (int i = 0; i < numRegsPerRow; i++) {
        jit_->vmovdqu32(RegType(scratchLoadRegIdx + i),
                        jit_->ptr[regTmp2 + i * RegBytes]);
        // Right shift by 7: compensates for left shift during B packing
        // Refer: "classic/kernels/zen4/s8s8s32/lpgemm_packb_s8_amd512vnni.c"
        jit_->vpsrad(RegType(scratchLoadRegIdx + i),
                     RegType(scratchLoadRegIdx + i), 7);
        jit_->vcvtdq2ps(RegType(scratchLoadRegIdx + i),
                        RegType(scratchLoadRegIdx + i));
        // Use separate multiply and add instead of FMA to match reference
        // rounding behavior (two rounding points instead of one)
        RegType tempReg = popAndGetScratchReg();
        jit_->vmulps(tempReg, RegType(scratchLoadRegIdx + i), zpReg);
        for (int j = 0; j < MR; j++) {
            jit_->vaddps(RegType(cRegStartIdx + j * numRegsPerRow + i),
                         RegType(cRegStartIdx + j * numRegsPerRow + i),
                         tempReg);
        }
        scratch_reg_queue.push(tempReg);
    }

    scratch_reg_queue.push(zpReg);
    restoreScratchQueue(saved_queue);
    return jitGeneratorError::success;
}

// ============================================================================
// A-Dequantization: Zero-Point Correction (Per-Row)
// ============================================================================
// Formula: acc[row][col] = acc[row][col] + (b_col_sum[col] * a_post_quant_zp[row])
// Compensates for zero-point bias with per-row granularity. Used for per-row
// asymmetric quantization.
template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::aDQuantZeroPointRowMajorImpl()
{
    // Load B column sums pointer and calculate address for current column block
    jit_->mov(regTmp2,
              jit_->ptr[regkernelOpsAttr
                        + offsetof(lpgemm_post_op_attr, b_col_sum_vec)]);
    jit_->mov(regTmp3, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, b_sum_offset)]);
    jit_->lea(regTmp2, jit_->ptr[regTmp2 + regTmp3 * sizeof(int32_t)]);

    // Calculate address for zero-points: zero_point_base + (post_op_c_i *
    // sizeof(T))
    jit_->mov(regTmp4, jit_->ptr[regkernelOpsAttr
                                 + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
    jit_->lea(regTmp4, jit_->ptr[regTmp4 * sizeof(T)]);
    jit_->lea(regTmp3, jit_->ptr[regTmp1]);
    jit_->add(regTmp3, regTmp4);

    auto saved_queue = reserveDestRegisters(scratchLoadRegIdx, numRegsPerRow);

    // Load b_col_sum once (reused for each row with different zp)
    for (int i = 0; i < numRegsPerRow; i++) {
        jit_->vmovdqu32(RegType(scratchLoadRegIdx + i),
                        jit_->ptr[regTmp2 + i * RegBytes]);
        // Right shift by 7: compensates for left shift during B packing
        // Refer: "classic/kernels/zen4/s8s8s32/lpgemm_packb_s8_amd512vnni.c"
        jit_->vpsrad(RegType(scratchLoadRegIdx + i),
                     RegType(scratchLoadRegIdx + i), 7);
        jit_->vcvtdq2ps(RegType(scratchLoadRegIdx + i),
                        RegType(scratchLoadRegIdx + i));
    }

    // For each row: broadcast its zero-point and apply correction
    for (int j = 0; j < MR; j++) {
        RegType zpReg = popAndGetScratchReg();
        broadcastAndConvertScalar<T>(zpReg, jit_->ptr[regTmp3 + j * sizeof(T)]);

        for (int i = 0; i < numRegsPerRow; i++) {
            // Use separate multiply and add instead of FMA to match reference
            // rounding behavior (two rounding points instead of one)
            RegType tempReg = popAndGetScratchReg();
            jit_->vmulps(tempReg, RegType(scratchLoadRegIdx + i), zpReg);
            jit_->vaddps(RegType(cRegStartIdx + j * numRegsPerRow + i),
                         RegType(cRegStartIdx + j * numRegsPerRow + i),
                         tempReg);
            scratch_reg_queue.push(tempReg);
        }
        scratch_reg_queue.push(zpReg);
    }

    restoreScratchQueue(saved_queue);
    return jitGeneratorError::success;
}

// ============================================================================
// A-Dequantization: Zero-Point Correction (Per-Tensor) (GEMV N=1)
// ============================================================================
// Formula: acc[i] = acc[i] + (b_col_sum * a_post_quant_zp)
// Used for GEMV N=1 with scalar zero-point.
template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::aDQuantZeroPointScalarImplGEMVN1()
{
    // Broadcast scalar zero-point (regTmp1 set by caller)
    RegType zpReg = popAndGetScratchReg();
    broadcastAndConvertScalar<T>(zpReg, jit_->ptr[regTmp1]);

    // Load b_col_sum pointer and calculate address for current column block
    jit_->mov(regTmp2,
              jit_->ptr[regkernelOpsAttr
                        + offsetof(lpgemm_post_op_attr, b_col_sum_vec)]);
    RegType bColSumReg = popAndGetScratchReg();
    // Broadcast b_col_sum
    jit_->vpbroadcastd(bColSumReg, jit_->ptr[regTmp2]);
    // Apply right shift by 7 to b_col_sum
    jit_->vpsrad(bColSumReg, bColSumReg, 7);
    // Convert integer to float after shift
    jit_->vcvtdq2ps(bColSumReg, bColSumReg);
    // Multiply b_col_sum by zero-point
    jit_->vmulps(bColSumReg, bColSumReg, zpReg);
    // Apply zero-point to all accumulator registers
    for (int i = 0; i < numRegsPerRow; i++) {
        jit_->vaddps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                     bColSumReg);
    }
    scratch_reg_queue.push(zpReg);
    scratch_reg_queue.push(bColSumReg);
    return jitGeneratorError::success;
}

// ============================================================================
// A-Dequantization: Zero-Point Correction (Per-Row) (GEMV N=1)
// ============================================================================
// Formula: acc[i] = acc[i] + (b_col_sum * a_post_quant_zp[i])
// Used for GEMV N=1 with row-major zero-point.
template<utils::kernelInstrType KType>
template<typename T>
jitGeneratorError
kernelOpsGeneratorX86<KType>::aDQuantZeroPointRowMajorImplGEMVN1()
{
    // Calculate address: b_col_sum_base + (post_op_c_i * sizeof(T))
    jit_->lea(regTmp2, jit_->ptr[regTmp2 * sizeof(T)]);
    jit_->add(regTmp1, regTmp2);

    // Reserve destination registers to prevent corruption during BF16/S8/U8
    // conversions
    auto saved_queue = reserveDestRegisters(scratchLoadRegIdx, numRegsPerRow);

    loadAndConvertRows<T>(regTmp1, scratchLoadRegIdx);

    // b_col_sum pointer load
    jit_->mov(regTmp3,
              jit_->ptr[regkernelOpsAttr
                        + offsetof(lpgemm_post_op_attr, b_col_sum_vec)]);
    RegType bColSumReg = popAndGetScratchReg();
    // Broadcast b_col_sum
    jit_->vpbroadcastd(bColSumReg, jit_->ptr[regTmp3]);
    // Apply right shift by 7 to b_col_sum
    jit_->vpsrad(bColSumReg, bColSumReg, 7);
    // Convert integer to float after shift
    jit_->vcvtdq2ps(bColSumReg, bColSumReg);

    for (int i = 0; i < numRegsPerRow; i++) {
        RegType tempReg = popAndGetScratchReg();
        // Multiply b_col_sum by accumulator register
        jit_->vmulps(tempReg, bColSumReg, RegType(scratchLoadRegIdx + i));
        // Add b_col_sum to accumulator register
        jit_->vaddps(RegType(cRegStartIdx + i), RegType(cRegStartIdx + i),
                     tempReg);
        scratch_reg_queue.push(tempReg);
    }

    scratch_reg_queue.push(bColSumReg);
    restoreScratchQueue(saved_queue);
    return jitGeneratorError::success;
}

// ============================================================================
// A-Dequantization Dispatch Functions
// ============================================================================

// Dispatcher: selects per-tensor or per-row scale factor implementation
template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::aDQuantScaleFactorImpl(kernelOpsMetaData& op)
{
    jit_->mov(
        regTmp1,
        jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, scale_factor)]);

    if (algoType_ == dlp::jit::jitAlgoType::gemv_n1) {
        if (op.scalarScaleFactorRequired) {
            DISPATCH_BY_DATATYPE(op.scaleFactorDt,
                                 aDQuantScaleFactorScalarImplGEMVN1);
        } else if (op.cMatFormat == storageFormat::rowMajor) {
            jit_->mov(regTmp2,
                      jit_->ptr[regkernelOpsAttr
                                + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
            DISPATCH_BY_DATATYPE(op.scaleFactorDt,
                                 aDQuantScaleFactorRowMajorImplGEMVN1);
        } else {
            return jitGeneratorError::notSupported;
        }
    } else {
        if (op.scalarScaleFactorRequired) {
            DISPATCH_BY_DATATYPE(op.scaleFactorDt,
                                 aDQuantScaleFactorScalarImpl);
        } else if (op.cMatFormat == storageFormat::rowMajor) {
            DISPATCH_BY_DATATYPE(op.scaleFactorDt,
                                 aDQuantScaleFactorRowMajorImpl);
        } else {
            return jitGeneratorError::notSupported;
        }
    }

    return jitGeneratorError::success;
}

// Dispatcher: selects per-tensor or per-row zero-point implementation
// Skipped for symmetric quantization (zero-point implicitly 0)
template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::aDQuantZeroPointImpl(kernelOpsMetaData& op)
{
    if (!op.scalarZeroPointRequired && !op.vectorZeroPointRequired) {
        return jitGeneratorError::success; // Symmetric quantization
    }

    jit_->mov(regTmp1,
              jit_->ptr[regkernelOpsList + offsetof(lpgemm_post_op, op_args1)]);

    if (algoType_ == dlp::jit::jitAlgoType::gemv_n1) {
        // ============ GEMV n=1 path ============
        if (op.scalarZeroPointRequired) {
            DISPATCH_BY_DATATYPE(op.zeroPointDt,
                                 aDQuantZeroPointScalarImplGEMVN1);
        } else if (op.cMatFormat == storageFormat::rowMajor) {
            jit_->mov(regTmp2,
                      jit_->ptr[regkernelOpsAttr
                                + offsetof(lpgemm_post_op_attr, post_op_c_i)]);
            DISPATCH_BY_DATATYPE(op.zeroPointDt,
                                 aDQuantZeroPointRowMajorImplGEMVN1);
        } else {
            return jitGeneratorError::notSupported;
        }
    } else {
        if (op.scalarZeroPointRequired) {
            DISPATCH_BY_DATATYPE(op.zeroPointDt, aDQuantZeroPointScalarImpl);
        } else if (op.cMatFormat == storageFormat::rowMajor) {
            DISPATCH_BY_DATATYPE(op.zeroPointDt, aDQuantZeroPointRowMajorImpl);
        } else {
            return jitGeneratorError::notSupported;
        }
    }
    return jitGeneratorError::success;
}

// A-dequantization: result = round((acc + b_col_sum * a_post_quant_zp) * a_post_quant_sf)
// Order: (1) zero-point correction, (2) scale factor application with rounding
template<utils::kernelInstrType KType>
jitGeneratorError
kernelOpsGeneratorX86<KType>::aDQuantize(kernelOpsMetaData& op)
{
    jitGeneratorError err_dequant_zp = aDQuantZeroPointImpl(op);
    jitGeneratorError err_dequant_sf = aDQuantScaleFactorImpl(op);

    if (err_dequant_zp != jitGeneratorError::success
        || err_dequant_sf != jitGeneratorError::success) {
        return jitGeneratorError::notSupported;
    }
    return jitGeneratorError::success;
}

} // namespace amdzen::x86gen
