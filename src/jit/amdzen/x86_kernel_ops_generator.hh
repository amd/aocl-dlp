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

// Dispatch by dual datatype: one as template parameter, one at runtime
// Use this when you have one type (SfType) as a template parameter and need to
// dispatch on another datatype
#define DISPATCH_OP_BY_DUAL_DATATYPE(SfType, matOpDt, func, ...)               \
    switch (matOpDt) {                                                         \
        case DataType::f32:                                                    \
            return func<SfType, float>(__VA_ARGS__);                           \
        case DataType::bf16:                                                   \
            return func<SfType, bfloat16>(__VA_ARGS__);                        \
        case DataType::s8:                                                     \
            return func<SfType, int8_t>(__VA_ARGS__);                          \
        case DataType::u8:                                                     \
            return func<SfType, uint8_t>(__VA_ARGS__);                         \
        case DataType::s32:                                                    \
            return func<SfType, int32_t>(__VA_ARGS__);                         \
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

    dlp::jit::jitGeneratorError setPostOpsContext(int  MR,
                                                  int  NR,
                                                  bool useMask,
                                                  int  numMaskRegs,
                                                  int  cRegStartIdx,
                                                  int  cRegCount);

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

    // Helper to dispatch matOpScaleFactorImplMerged on matrix datatype
    template<typename SfType>
    dlp::jit::jitGeneratorError dispatchMatOpByMatrixType(
        dlp::kernel_frame::DataType matOpDt,
        matOpType                   opType,
        matOpScaleType              sclType);

    // Helper to dispatch matOpScaleFactorImplGEMVN1 on matrix datatype
    template<typename SfType>
    dlp::jit::jitGeneratorError dispatchMatOpByMatrixTypeGEMVN1(
        dlp::kernel_frame::DataType matOpDt,
        matOpType                   opType,
        matOpScaleType              sclType);

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

    // ERF coefficients for AVX-512 piecewise polynomial
    // approximation 6 degrees × 32 regions (24 active + 8 padding for
    // VPERMT2PS). Approximates erf(x/√2) using degree-5 polynomials across 24
    // regions
    uint32_t erf_f32_coeffs_hex[6][32] = {
        { // Degree 0
          0x31919200, 0x32807195, 0x3382b14e, 0x34505ab6, 0x350c5db6,
          0x35f2ff5c, 0x36f037e3, 0x37b9c4fc, 0x3870c660, 0x3942160a,
          0x3a2ab0e1, 0x3ae3c00e, 0x3b757566, 0x3c0825ed, 0x3c510b4f,
          0xbb54959c, 0xbd8aa4c0, 0xbe8a61b3, 0xbf0bf55a, 0xbecccf5d,
          0x3e0cff67, 0x3f461d8a, 0x3f7d51a9, 0x3f7ff731, 0x00000000,
          0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
          0x00000000, 0x00000000 },
        { // Degree 1
          0x3f4c4226, 0x3f4c421f, 0x3f4c4207, 0x3f4c41cc, 0x3f4c414e,
          0x3f4c3fae, 0x3f4c3a22, 0x3f4c2d24, 0x3f4c12e0, 0x3f4bc213,
          0x3f4acfdf, 0x3f48f77d, 0x3f46093b, 0x3f405752, 0x3f3b9cdd,
          0x3f48ee6b, 0x3f77b93f, 0x3fbad86d, 0x40018bfc, 0x3fe53efe,
          0x3f82f0de, 0x3e70a95c, 0x3c15f930, 0x38d32404, 0x00000000,
          0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
          0x00000000, 0x00000000 },
        { // Degree 2
          0x3691b95d, 0x3733f15f, 0x37f43886, 0x388b4cd9, 0x390cecf5,
          0x39ab0a08, 0x3a622cc3, 0x3afad85a, 0x3b74f8ce, 0x3c0b61be,
          0x3ca5eef7, 0x3d216fc4, 0x3d86408c, 0x3ddf37e7, 0x3e0f0b63,
          0x3d936854, 0xbe0ad73d, 0xbf1d4b56, 0xbf89c432, 0xbf6d9f4a,
          0xbefa57fe, 0xbdc89d25, 0xbb51e003, 0xb7fd21d0, 0x00000000,
          0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
          0x00000000, 0x00000000 },
        { // Degree 3
          0xbe083848, 0xbe084539, 0xbe0863d5, 0xbe0897b7, 0xbe08e8dd,
          0xbe09ab6f, 0xbe0b6a61, 0xbe0e45ae, 0xbe128735, 0xbe1bfd38,
          0xbe2f2738, 0xbe494387, 0xbe67db90, 0xbe89ae93, 0xbe96c166,
          0xbe802bd6, 0xbe0787ed, 0x3dcee55b, 0x3e94a722, 0x3e79387e,
          0x3df0e54b, 0x3ca7993e, 0x3a12efe6, 0x3697c013, 0x00000000,
          0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
          0x00000000, 0x00000000 },
        { // Degree 4
          0x398accca, 0x39ef1989, 0x3a58f003, 0x3ab1623b, 0x3b06f06d,
          0x3b653277, 0x3bcb8f35, 0x3c22886d, 0x3c703256, 0x3cc18958,
          0x3d1df607, 0x3d639359, 0x3d94cbcb, 0x3dbf6bcf, 0x3dd53028,
          0x3db7bc9d, 0x3d65fb7c, 0xba571897, 0xbd228f61, 0xbd041cef,
          0xbc6932ba, 0xbb0c49ca, 0xb84dd98d, 0xb4b5f4b3, 0x00000000,
          0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
          0x00000000, 0x00000000 },
        { // Degree 5
          0x3c9dd788, 0x3c9b68a5, 0x3c9786ce, 0x3c92f153, 0x3c8daa4e,
          0x3c848448, 0x3c6ca387, 0x3c4c4966, 0x3c28d0bd, 0x3bdf6be2,
          0x3b058174, 0xbb23ec7e, 0xbbd24ce4, 0xbc2c1681, 0xbc491abb,
          0xbc2a591c, 0xbbd760ce, 0xba832610, 0x3b0ff520, 0x3ae25578,
          0x3a35958a, 0x38bc345e, 0x35e6ce12, 0x322e8b39, 0x00000000,
          0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
          0x00000000, 0x00000000 }
    };

    // ERF constants for F32 AVX-512 implementation (AOCL-DLP approach)
    // Array of constants used in ERF computation
    uint32_t erf_f32_constants_hex[4] = {
        0xc21fffff, // [0] erf_idx_bias: Bias for index calculation
        0x7fffffff, // [1] abs_mask: Mask to extract absolute value (clear sign
                    // bit)
        0x80000000, // [2] sign_mask: Mask to extract sign bit
        0x40e00000  // [3] rbound: 7.0f (upper bound for input clamping)
    };

    const md_t gelu_consts_off    = 0;
    const md_t gelu_macros_off    = gelu_consts_off + sizeof(gelu_consts);
    const md_t lpgemm_exp_off     = gelu_macros_off + sizeof(gelu_macros);
    const md_t erf_consts_off     = lpgemm_exp_off + sizeof(lpgemm_exp);
    const md_t lpgemm_erf_off     = erf_consts_off + sizeof(erf_consts);
    const md_t erf_f32_coeffs_off = lpgemm_erf_off + sizeof(lpgemm_erf);
    const md_t erf_f32_constants_off =
        erf_f32_coeffs_off + sizeof(erf_f32_coeffs_hex);

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

    // Helper functions for ERF coefficient access
    // Get base address of a coefficient array (32 floats) for a given degree
    Xbyak::Address get_erf_f32_coeff_array_lo(md_t degree)
    {
        md_t offset = erf_f32_coeffs_off + (degree * 32 * sizeof(uint32_t));
        return jit_->ptr[jit_->rip + tables + offset];
    }

    // Get address of the second half of coefficient array (elements 16-31)
    Xbyak::Address get_erf_f32_coeff_array_hi(md_t degree)
    {
        md_t offset =
            erf_f32_coeffs_off + (degree * 32 + 16) * sizeof(uint32_t);
        return jit_->ptr[jit_->rip + tables + offset];
    }

    // Get address of ERF F32 constants
    // constant_idx: 0=bias, 1=abs_mask, 2=sign_mask, 3=rbound
    Xbyak::Address get_erf_f32_constant(md_t constant_idx)
    {
        md_t offset = erf_f32_constants_off + (constant_idx * sizeof(uint32_t));
        return jit_->ptr[jit_->rip + tables + offset];
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
                              bool           useMaskOp,
                              int            fringeMaskId = 0);

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
