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

#pragma once

#include "jit_generator_utils.hh"
#include "kernel_ops_generator_base.hh"
#include "traits.hh"
#include <queue>

namespace amdzen::x86gen {

#define DISPATCH_BY_DATATYPE(dt, func, ...)                                    \
    switch (dt) {                                                              \
        case DataType::f32:                                                    \
            return func<float>(__VA_ARGS__);                                   \
        case DataType::bf16:                                                   \
            return func<bfloat16>(__VA_ARGS__);                                \
        case DataType::s8:                                                     \
            return func<int8_t>(__VA_ARGS__);                                  \
        case DataType::u8:                                                     \
            return func<uint8_t>(__VA_ARGS__);                                 \
        case DataType::s32:                                                    \
            return func<int32_t>(__VA_ARGS__);                                 \
        default:                                                               \
            return jitGeneratorError::notSupported;                            \
    }

#define DISPATCH_BY_DUAL_DATATYPE(sfDt, matOpDt, func, ...)                    \
    switch (sfDt) {                                                            \
        case DataType::f32:                                                    \
            switch (matOpDt) {                                                 \
                case DataType::f32:                                            \
                    return func<float, float>(__VA_ARGS__);                    \
                default:                                                       \
                    return jitGeneratorError::notSupported;                    \
            }                                                                  \
        case DataType::bf16:                                                   \
            switch (matOpDt) {                                                 \
                case DataType::f32:                                            \
                    return func<bfloat16, float>(__VA_ARGS__);                 \
                default:                                                       \
                    return jitGeneratorError::notSupported;                    \
            }                                                                  \
        case DataType::s8:                                                     \
            switch (matOpDt) {                                                 \
                case DataType::f32:                                            \
                    return func<int8_t, float>(__VA_ARGS__);                   \
                default:                                                       \
                    return jitGeneratorError::notSupported;                    \
            }                                                                  \
        case DataType::u8:                                                     \
            switch (matOpDt) {                                                 \
                case DataType::f32:                                            \
                    return func<uint8_t, float>(__VA_ARGS__);                  \
                default:                                                       \
                    return jitGeneratorError::notSupported;                    \
            }                                                                  \
        case DataType::s32:                                                    \
            switch (matOpDt) {                                                 \
                case DataType::f32:                                            \
                    return func<int32_t, float>(__VA_ARGS__);                  \
                default:                                                       \
                    return jitGeneratorError::notSupported;                    \
            }                                                                  \
        case DataType::u32:                                                    \
            switch (matOpDt) {                                                 \
                case DataType::f32:                                            \
                    return func<float, uint32_t>(__VA_ARGS__);                 \
                default:                                                       \
                    return jitGeneratorError::notSupported;                    \
            }                                                                  \
        default:                                                               \
            return jitGeneratorError::notSupported;                            \
    }

template<utils::kernelInstrType KType>
class kernelOpsGeneratorX86 : public gen::kernelOpsGeneratorInterface
{
    using Traits      = traits::ArchitectureTraits<KType>;
    using RegType     = typename Traits::RegType;
    using halfRegType = typename Traits::halfRegType;

  public:
    kernelOpsGeneratorX86(Xbyak::CodeGenerator* jit);

    ~kernelOpsGeneratorX86()                                       = default;
    kernelOpsGeneratorX86(const kernelOpsGeneratorX86&)            = delete;
    kernelOpsGeneratorX86& operator=(const kernelOpsGeneratorX86&) = delete;
    kernelOpsGeneratorX86(kernelOpsGeneratorX86&&)                 = delete;
    kernelOpsGeneratorX86& operator=(kernelOpsGeneratorX86&&)      = delete;

    dlp::jit::jitGeneratorError generateKernelOps(
        std::vector<dlp::kernel_frame::kernelOpsMetaData>& kernelOps,
        const Xbyak::Reg64&   postOpsArgWrapperPtrReg,
        dlp::jit::jitAlgoType algoType,
        int                   MR,
        int                   NR,
        bool                  useMask,
        int                   numMaskRegs,
        int                   cRegStartIdx,
        int                   cRegCount) override;
    dlp::jit::jitGeneratorError bias(
        dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError relu(
        [[maybe_unused]] dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError reluScale(
        dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError geluTanh(
        [[maybe_unused]] dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError geluErf(
        [[maybe_unused]] dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError clip(
        dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError downscale(
        dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError matadd(
        dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError matmul(
        dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError swish(
        dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError tanh(
        [[maybe_unused]] dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError sigmoid(
        [[maybe_unused]] dlp::kernel_frame::kernelOpsMetaData& op) override;
    dlp::jit::jitGeneratorError embedKernelOpsAttributes() override;

    void advancePostOpsPtr() override;

  private:
    int numRegs  = Traits::numRegs;
    int RegBytes = Traits::regBytes;

    int MR, NR, useMask, numMaskRegs;
    int numFullRegsPerRow;
    int numMaskRegsPerRow;
    int numRegsPerRow;

    int cRegStartIdx, cRegCount;
    int scratchLoadRegIdx, scratchBcstRegIdx;

    // Algorithm type for dispatching appropriate implementations
    dlp::jit::jitAlgoType algoType_;

    // Flag to track if constant tables have been embedded (to avoid
    // duplication) - Instance variable, not static, so each handler
    // instance tracks its own state independently
    bool tablesEmbedded = false;

    // registers used for gelu_tanh
    int num_gelu_regs = 9;
    int const1;
    int const2;
    int x;
    int r;
    int r2;
    int z;
    int dn;
    int x_tanh;
    int q;

    // registers for gelu_erf
    int num_erf_regs = 5;
    int x_erf;

    // registers used for swish. Reusing the gelu_tanh registers.
    int num_swish_regs = 9;

    const Xbyak::Reg64 &regkernelOpsList, &regkernelOpsAttr;
    const Xbyak::Reg64 &regTmp1, &regTmp2, &regTmp3, &regTmp4, &regTmp5,
        &regTmp6, &regTmp7, &regcsC;
    const Xbyak::Reg32 &regTmp4Half, &regTmp5Half;

    Xbyak::CodeGenerator* jit_; // Back reference to access registers and state

    dlp::jit::jitGeneratorError setPostOpsContext(
        int MR, int NR, bool useMask, int cRegStartIdx, int cRegCount);

    // Helper implementations for different storage formats
    // GEMM implementations
    template<typename T>
    dlp::jit::jitGeneratorError biasRowMajorImplGEMM();

    template<typename T>
    dlp::jit::jitGeneratorError biasColMajorImplGEMM();

    // GEMV n=1 implementations
    template<typename T>
    dlp::jit::jitGeneratorError biasRowMajorImplGEMVN1();

    template<typename T>
    dlp::jit::jitGeneratorError biasColMajorImplGEMVN1();

    template<typename T>
    dlp::jit::jitGeneratorError reluScaleImpl();

    template<typename T>
    dlp::jit::jitGeneratorError clipImpl();

    dlp::jit::jitGeneratorError scaleFactorImpl(
        dlp::kernel_frame::kernelOpsMetaData& op);

    dlp::jit::jitGeneratorError zeroPointImpl(
        dlp::kernel_frame::kernelOpsMetaData& op);

    template<typename T>
    dlp::jit::jitGeneratorError scaleFactorScalarImpl();

    template<typename T>
    dlp::jit::jitGeneratorError scaleFactorRowMajorImpl();

    template<typename T>
    dlp::jit::jitGeneratorError scaleFactorColMajorImpl();

    template<typename T>
    dlp::jit::jitGeneratorError zeroPointScalarImpl();

    template<typename T>
    dlp::jit::jitGeneratorError zeroPointRowMajorImpl();

    template<typename T>
    dlp::jit::jitGeneratorError zeroPointColMajorImpl();

    // GEMV n=1 specific implementations
    template<typename T>
    dlp::jit::jitGeneratorError scaleFactorScalarImplGEMVN1();

    template<typename T>
    dlp::jit::jitGeneratorError scaleFactorColMajorImplGEMVN1();

    template<typename T>
    dlp::jit::jitGeneratorError zeroPointScalarImplGEMVN1();

    template<typename T>
    dlp::jit::jitGeneratorError zeroPointColMajorImplGEMVN1();

    template<typename T>
    dlp::jit::jitGeneratorError swishImpl();

    enum class matOpType
    {
        matOpAdd,
        matOpMul
    };

    enum class matOpScaleType
    {
        scalar,
        rowVector,
        columnVector
    };

    template<typename sfDt, typename matOpDt>
    dlp::jit::jitGeneratorError matOpScaleFactorImpl(matOpType      opType,
                                                     matOpScaleType sclType);

    template<typename sfDt, typename matOpDt>
    dlp::jit::jitGeneratorError matOpScaleFactorImplColMat(
        matOpType opType, matOpScaleType sclType);

    template<typename sfDt, typename matOpDt>
    dlp::jit::jitGeneratorError matOpScaleFactorImplMerged(
        matOpType opType, matOpScaleType sclType);

    template<typename sfDt, typename matOpDt>
    dlp::jit::jitGeneratorError matOpScaleFactorImplGEMVN1(
        matOpType opType, matOpScaleType sclType);

    // TODO: Math Utils, move to different class.
    void POLY_EVAL_6();
    void EXPF();
    void TANHF();
    void GELU_TANH_F32_DEF(md_t reg);
    void TANHF_DEF(md_t reg);
    void POLY_EVAL_HORNER_16_0(int r);
    void ERF(int y, int r);
    void GELU_ERF_F32_DEF(md_t reg);
    void SWISH_F32_DEF(md_t reg);
    void SIGMOID_DEF(md_t reg);
    void apply_post_ops_in_high_reg_pressure(const md_t num_post_op_regs,
                                             std::function<void(md_t)> op_fn);
    void store_reg_in_stack(md_t reg_start_idx, md_t num_regs);
    void get_reg_from_stack(md_t reg_start_idx, md_t num_regs);

    // Table of constants used in gelu_tanh and gelu_erf.
    // TODO: Clean this up and move to a more appropriate place.
    float gelu_consts[8] = { 0.044715, 0.797884, -2, 0.5, -1, 2, 1, -0.0f };
    float gelu_macros[6] = { 1.4426950408889634, 1.2582912E7, -88.0f, 88.0f,
                             (float)(1.0 / 0.0), -2147483648 };

    float lpgemm_exp[6] = { 1.0000000754895704,   0.6931472254087585,
                            0.2402210737432219,   0.05550297297702539,
                            0.009676036358193323, 0.001341000536524434 };

    float  erf_consts[5]  = { 0.70710678118654f, 1.0, 0.5, 3.553f,
                              3.91920638084411621F };
    double lpgemm_erf[16] = { 0x1.20dd7890d27e1cec99fce48c29cp0,
                              -0x1.ab4bed70f238422edeeba9c558p-16,
                              -0x1.80a1bd5878e0b0689c5ff4fcdd4p-2,
                              -0x1.07cb4cde6a7d9528c8a732990e4p-8,
                              0x1.092cba598f96f00ddc5854cf7cp-3,
                              -0x1.51f0ce4ac87c55f11f685864714p-5,
                              0x1.4101f320bf8bc4d41c228faaa6cp-5,
                              -0x1.2300882a7d1b712726997de80ep-4,
                              0x1.d45745fff0e4b6d0604a9ab6284p-5,
                              -0x1.9eb1491956e31ded96176d7c8acp-6,
                              0x1.b9183fc75d326b9044bc63c9694p-8,
                              -0x1.10e8f8c89ad8645e7d769cd596cp-10,
                              0x1.224ffc80cc19957a48ecedad6c8p-14,
                              0x1.12a30f42c71308321e7e7cb0174p-18,
                              -0x1.155445e2e006723066d72d22ddcp-20,
                              0x1.c6a4181da4ef76f22bd39bb5dcp-25 };

    const md_t gelu_consts_off = 0;
    const md_t gelu_macros_off = gelu_consts_off + sizeof(gelu_consts);
    const md_t lpgemm_exp_off  = gelu_macros_off + sizeof(gelu_macros);
    const md_t erf_consts_off  = lpgemm_exp_off + sizeof(lpgemm_exp);
    const md_t lpgemm_erf_off  = erf_consts_off + sizeof(erf_consts);

    template<typename T>
    Xbyak::Address get_constant_T(md_t table_off, md_t value_off)
    {
        return jit_->ptr[jit_->rip + tables + table_off
                         + (value_off * (md_t)sizeof(T))];
    }

    Xbyak::Address get_constant(md_t table_off, md_t value_off)
    {
        return get_constant_T<float>(table_off, value_off);
    }

    Xbyak::Address get_constant_dbl(md_t table_off, md_t value_off)
    {
        return get_constant_T<double>(table_off, value_off);
    }

    Xbyak::Label erf_end;
    Xbyak::Label tables;
    Xbyak::Label table_store_end; // Instance-specific label for jump target

    Xbyak::Opmask fringeMask[dlp::kernels::maxNumMasks];
    Xbyak::Opmask mask0, mask1;

    Xbyak::Ymm ymmMask;

    std::queue<RegType> scratch_reg_queue;

    RegType popAndGetScratchReg()
    {
        RegType reg = scratch_reg_queue.front();
        scratch_reg_queue.pop();
        return reg;
    }

    void resetMasks(bool mask, int idx);
    // Helper functions for scratch register management
    // Reserve destination registers from scratch pool and return saved state
    inline std::queue<RegType> reserveDestRegisters(int destRegStart, int count)
    {
        std::queue<RegType> original_queue = scratch_reg_queue;
        std::queue<RegType> temp_queue;

        while (!scratch_reg_queue.empty()) {
            RegType reg = scratch_reg_queue.front();
            scratch_reg_queue.pop();

            // Check if this register is in the destination range
            bool isDestReg = false;
            for (int i = 0; i < count; i++) {
                if (reg.getIdx() == RegType(destRegStart + i).getIdx()) {
                    isDestReg = true;
                    break;
                }
            }

            if (!isDestReg) {
                temp_queue.push(reg);
            }
        }

        scratch_reg_queue = std::move(temp_queue);
        return original_queue;
    }

    // Restore scratch register queue to previous state
    inline void restoreScratchQueue(const std::queue<RegType>& saved)
    {
        scratch_reg_queue = saved;
    }

    // Helper to get the load bytes for a given type
    template<typename T>
    int getLoadBytes();

    // Helper to load and convert vector data to float32
    template<typename T>
    void loadAndConvertVector(RegType        destReg,
                              Xbyak::Address src,
                              bool           useMaskOp);

    template<typename T>
    void loadAndConvertRows(Xbyak::Reg64 addressReg, int regStartIdx);

    // Broadcast and convert scalar helper functions for all data types
    template<typename T>
    void broadcastAndConvertScalar(RegType bcstReg, Xbyak::Address src);
};

// Type aliases using kernelInstrType
using kernelOpsGeneratorAvx2 =
    kernelOpsGeneratorX86<utils::kernelInstrType::avx2_ymm_16_reg>;
using kernelOpsGeneratorAvx512_ymm =
    kernelOpsGeneratorX86<utils::kernelInstrType::avx512_ymm_32_reg>;
using kernelOpsGeneratorAvx512 =
    kernelOpsGeneratorX86<utils::kernelInstrType::avx512_zmm_32_reg>;

} // namespace amdzen::x86gen
