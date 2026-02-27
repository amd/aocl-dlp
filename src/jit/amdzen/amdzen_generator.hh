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

#include "debug_utils/gdb_helper_utils.hh"
#include "debug_utils/jit_debug_utils.hh"
#include "jit/jit_generator_base.hh"
#include "jit_generator_utils.hh"
#include "kernels/kernel_base.hh"
#include "traits.hh"

namespace amdzen::gen {

// Single generator for both AVX512 and AVX2 micro-kernels.
class jitAmdZenFP32 : public dlp::jit::jitGeneratorBase
{

    std::vector<dlp::kernel_frame::kernelDatatype> mKernelDatatypes;
    std::vector<dlp::cpu_utils::isaFeature>        mIsaFeaturesRequired;
    utils::kernelInstrType                         kType;
    int                                            numElemsPerReg;

    md_t MR, NR, KC;
    md_t numMRVariants, numNRVariants;

    // Number of registers to be used for handling NR fringe cases.
    // To be noted, termNRFringeRegCount = 0 mean it is for the lt (<)
    // numElemsPerReg case. It will still required mask registers to
    // handle the fringe.
    int  termNRFringeRegCount;
    bool isGenLtKrnlForAvailFullKrnl;

    bool usingRDKernels; // Flag to track if RD kernels were generated

    md_t               numKernelVariants;
    md_t               K_UNROLL, PREFETCH_C_DIST;
    md_t               c_downscale;
    std::vector<void*> kernelCodeBlocks;
    size_t             kernelSize; // Size of each kernel variant in bytes
    std::vector<std::unique_ptr<Xbyak::CodeGenerator>> codeGenerators;

    void setGeneratorKernelMetaInfo(
        dlp::kernel_frame::kernelInstrPreference kInstPref);

    /* Function to retrieve the process block size of the kernel */
    int getProcessBlockSize() const;

    dlp::jit::jitGeneratorError deriveGEMMNumNRVariants(
        const dlp::jit::jitGeneratorContext& jI);

    void deriveGEMMNRAndMaskUse(int                     nr,
                                utils::generatorParams& params,
                                int& correspondingMainFringe);

    void setMaskForGEMMLtFringe(dlp::kernels::gemmParams* params,
                                int                       nRemainder);

    // This needs to be inlined, so keeping in class declaration.
    void executeGEMMMLoop(dlp::kernels::gemmParams* params,
                          int                       mFullPieces,
                          int                       mPartialPieces,
                          int                       kernel_n_idx,
                          int                       elementsToProcess,
                          md_t&                     n,
                          float**                   c_jr,
                          md_t                      og_post_op_c_i,
                          float*                    aPtr)
    {
        params->a = aPtr;
        params->c = (*c_jr);
        params->n = elementsToProcess;

        md_t og_post_op_c_j = (params->kernelOpsAttr).post_op_c_j;

        if (params->m >= MR) {
            params->mIter            = mFullPieces;
            int               m_idx  = 0;
            utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                kernelCodeBlocks[m_idx * numNRVariants + kernel_n_idx]);

            DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(kernel));
            kernel(params);
        }

        // Rd kernels overwrite the post_op_c_j value. So, we need to restore
        // it.
        (params->kernelOpsAttr).post_op_c_j = og_post_op_c_j;

        if (mPartialPieces) {
            (params->a) = (float*)(params->a) + mFullPieces * params->psA;
            (params->c) = (float*)(params->c) + mFullPieces * MR * params->rsC;
            int               m_idx  = mPartialPieces;
            utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                kernelCodeBlocks[m_idx * numNRVariants + kernel_n_idx]);

            DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(kernel));
            kernel(params);
        }

        params->b = (float*)(params->b) + (elementsToProcess * params->csB);
        (*c_jr)   = (float*)(*c_jr) + (elementsToProcess * params->csC);
        n -= elementsToProcess;
        (params->kernelOpsAttr).post_op_c_j += elementsToProcess;

        // The following line is necessary to ensure a subtle bug does not
        // occur. Unlike the classic kernels where the kernelOpsAttr is
        // passed by value to the fringe kernels, here it is kind of passed
        // by reference, since the params ptr is the only kernel argument
        // (inside which is the kernelOpsAttr). Since the while(n) loop can
        // execute multiple times, any state variable, like post_op_c_i, if
        // modified inside kernel, it needs to be reverted.
        (params->kernelOpsAttr).post_op_c_i = og_post_op_c_i;
    }

    dlp::jit::jitGeneratorError generateAllKernels(
        const dlp::jit::jitGeneratorContext& jI);

    dlp::jit::jitGeneratorError generateAllKernelsRD(
        const dlp::jit::jitGeneratorContext& jI);

  public:
    jitAmdZenFP32();
    ~jitAmdZenFP32();
    jitAmdZenFP32(const jitAmdZenFP32&)            = delete;
    jitAmdZenFP32& operator=(const jitAmdZenFP32&) = delete;
    jitAmdZenFP32(jitAmdZenFP32&&)                 = delete;
    jitAmdZenFP32& operator=(jitAmdZenFP32&&)      = delete;

    dlp::jit::jitGeneratorError operator()(
        const dlp::jit::jitGeneratorContext& jI) override
    {
        return generateAllKernels(jI);
    }

    std::vector<dlp::kernel_frame::kernelDatatype>& getKernelDatatypes()
        override
    {
        return mKernelDatatypes;
    }

    std::vector<dlp::cpu_utils::isaFeature>& getIsaFeaturesRequired() override
    {
        return mIsaFeaturesRequired;
    }

    dlp::kernels::kernelError executeKernel(
        dlp::kernels::kernelParams* _params) override;

    dlp::kernels::kernelError executeKernelRD(
        dlp::kernels::kernelParams* _params);

    std::unique_ptr<jitGeneratorBase> clone() override
    {
        return std::make_unique<jitAmdZenFP32>();
    }
};

class jitAmdZenBF16 : public dlp::jit::jitGeneratorBase
{

    std::vector<dlp::kernel_frame::kernelDatatype> mKernelDatatypes;
    std::vector<dlp::cpu_utils::isaFeature>        mIsaFeaturesRequired;
    utils::kernelInstrType                         kType;
    int                                            numElemsPerReg;

    void setGeneratorKernelMetaInfo(
        dlp::kernel_frame::kernelInstrPreference kInstPref);

  public:
    // jitAVX512 base;
    md_t                           MR, NR, KC;
    md_t                           numMRVariants, numNRVariants;
    md_t                           numKernelVariants;
    md_t                           K_UNROLL, PREFETCH_C_DIST;
    md_t                           c_downscale;
    std::vector<void*>             kernelCodeBlocks;
    std::unique_ptr<jitAmdZenFP32> f32JitGenerator;
    std::vector<std::unique_ptr<Xbyak::CodeGenerator>> codeGenerators;

    jitAmdZenBF16();
    ~jitAmdZenBF16();
    jitAmdZenBF16(const jitAmdZenBF16&)            = delete;
    jitAmdZenBF16& operator=(const jitAmdZenBF16&) = delete;
    jitAmdZenBF16(jitAmdZenBF16&&)                 = delete;
    jitAmdZenBF16& operator=(jitAmdZenBF16&&)      = delete;

    /* Function to retrieve the process block size of the kernel */
    int getProcessBlockSize() const;

    dlp::jit::jitGeneratorError generateAllKernels(
        const dlp::jit::jitGeneratorContext& jI);

    dlp::jit::jitGeneratorError operator()(
        const dlp::jit::jitGeneratorContext& jI) override
    {
        return generateAllKernels(jI);
    }

    std::vector<dlp::kernel_frame::kernelDatatype>& getKernelDatatypes()
        override
    {
        return mKernelDatatypes;
    }

    std::vector<dlp::cpu_utils::isaFeature>& getIsaFeaturesRequired() override
    {
        return mIsaFeaturesRequired;
    }

    dlp::kernels::kernelError executeKernel(
        dlp::kernels::kernelParams* _params) override;

    std::unique_ptr<jitGeneratorBase> clone() override
    {
        return std::make_unique<jitAmdZenBF16>();
    }
};

class jitAmdZenU8S8 : public dlp::jit::jitGeneratorBase
{

    std::vector<dlp::kernel_frame::kernelDatatype> mKernelDatatypes;
    std::vector<dlp::cpu_utils::isaFeature>        mIsaFeaturesRequired;
    utils::kernelInstrType                         kType;
    int                                            numElemsPerReg;
    int                                            numBytesPerElem;

    void setGeneratorKernelMetaInfo(
        dlp::kernel_frame::kernelInstrPreference kInstPref);

  public:
    // Kernel blocking parameters for u8s8s32
    md_t               MR, NR, KC;
    md_t               numMRVariants, numNRVariants;
    md_t               numKernelVariants;
    md_t               K_UNROLL;
    md_t               c_downscale;
    std::vector<void*> kernelCodeBlocks;
    std::vector<std::unique_ptr<Xbyak::CodeGenerator>> codeGenerators;

    // u8s8s32-specific parameters
    bool requiresBPacking; // B matrix packing is mandatory for VNNI
    bool supportsPostOps;  // Post-operations support
    int  vnniGroupSize;    // VNNI group size (typically 4 for int8)

    jitAmdZenU8S8();
    ~jitAmdZenU8S8();
    jitAmdZenU8S8(const jitAmdZenU8S8&)            = delete;
    jitAmdZenU8S8& operator=(const jitAmdZenU8S8&) = delete;
    jitAmdZenU8S8(jitAmdZenU8S8&&)                 = delete;
    jitAmdZenU8S8& operator=(jitAmdZenU8S8&&)      = delete;

    /* Function to retrieve the process block size of the kernel */
    int getProcessBlockSize() const;

    /* Function to get VNNI group size for int8 operations */
    int getVNNIGroupSize() const { return vnniGroupSize; }

    /* Function to check if B matrix packing is required */
    bool isBPackingRequired() const { return requiresBPacking; }

    dlp::jit::jitGeneratorError generateAllKernels(
        const dlp::jit::jitGeneratorContext& jI);

    dlp::jit::jitGeneratorError operator()(
        const dlp::jit::jitGeneratorContext& jI) override
    {
        return generateAllKernels(jI);
    }

    std::vector<dlp::kernel_frame::kernelDatatype>& getKernelDatatypes()
        override
    {
        return mKernelDatatypes;
    }

    std::vector<dlp::cpu_utils::isaFeature>& getIsaFeaturesRequired() override
    {
        return mIsaFeaturesRequired;
    }

    dlp::kernels::kernelError executeKernel(
        dlp::kernels::kernelParams* _params) override;

    std::unique_ptr<jitGeneratorBase> clone() override
    {
        return std::make_unique<jitAmdZenU8S8>();
    }
};

class jitAmdZenS8 : public dlp::jit::jitGeneratorBase
{

    std::vector<dlp::kernel_frame::kernelDatatype> mKernelDatatypes;
    std::vector<dlp::cpu_utils::isaFeature>        mIsaFeaturesRequired;
    utils::kernelInstrType                         kType;
    int                                            numElemsPerReg;

    // VNNI_CONST defines the number of 8-bit signed integers that are packed
    // together in VNNI format operations.
    int VNNI_CONST = 4;

    void setGeneratorKernelMetaInfo(
        dlp::kernel_frame::kernelInstrPreference kInstPref);

  public:
    md_t               MR, NR, KC;
    md_t               numMRVariants, numNRVariants;
    md_t               numKernelVariants;
    md_t               K_UNROLL, PREFETCH_C_DIST;
    md_t               c_downscale;
    std::vector<void*> kernelCodeBlocks;
    std::vector<std::unique_ptr<Xbyak::CodeGenerator>> codeGenerators;

    jitAmdZenS8();
    ~jitAmdZenS8();
    jitAmdZenS8(const jitAmdZenS8&)            = delete;
    jitAmdZenS8& operator=(const jitAmdZenS8&) = delete;
    jitAmdZenS8(jitAmdZenS8&&)                 = delete;
    jitAmdZenS8& operator=(jitAmdZenS8&&)      = delete;

    int getProcessBlockSize() const;

    dlp::jit::jitGeneratorError generateAllKernels(
        const dlp::jit::jitGeneratorContext& jI);

    dlp::jit::jitGeneratorError operator()(
        const dlp::jit::jitGeneratorContext& jI) override
    {
        return generateAllKernels(jI);
    }

    std::vector<dlp::kernel_frame::kernelDatatype>& getKernelDatatypes()
        override
    {
        return mKernelDatatypes;
    }

    std::vector<dlp::cpu_utils::isaFeature>& getIsaFeaturesRequired() override
    {
        return mIsaFeaturesRequired;
    }

    dlp::kernels::kernelError executeKernel(
        dlp::kernels::kernelParams* _params) override;

    std::unique_ptr<jitGeneratorBase> clone() override
    {
        return std::make_unique<jitAmdZenS8>();
    }
};

/**
 * @brief JIT generator for FP16 GEMM operations using AVX-512-FP16
 *
 * This class generates optimized FP16 GEMM kernels at runtime using
 * native AVX-512-FP16 instructions (vfmadd231ph, vmulph, etc.).
 * Supports MR=6, NR=128 blocking with 32 FP16 elements per ZMM register.
 */
class jitAmdZenFP16 : public dlp::jit::jitGeneratorBase
{
    std::vector<dlp::kernel_frame::kernelDatatype> mKernelDatatypes;
    std::vector<dlp::cpu_utils::isaFeature>        mIsaFeaturesRequired;
    utils::kernelInstrType                         kType;
    int                                            numElemsPerReg;
    AOCL_MEMORY_TAG mtag_b; // Memory tag for B matrix

    using FP16Types = amdzen::traits::kernel_types<
        dlp::kernel_frame::kernelDatatype::f16f16f16of16>;
    static constexpr int FP16_ELEM_SIZE = FP16Types::elemSize;
    static constexpr int FP16_PER_ZMM   = FP16Types::elemsPerZmm;

    void setGeneratorKernelMetaInfo(
        dlp::kernel_frame::kernelInstrPreference kInstPref);

  public:
    md_t               MR, NR, KC;
    md_t               numMRVariants, numNRVariants;
    md_t               numKernelVariants;
    md_t               K_UNROLL, PREFETCH_C_DIST;
    md_t               c_downscale;
    std::vector<void*> kernelCodeBlocks;
    std::vector<std::unique_ptr<Xbyak::CodeGenerator>> codeGenerators;

    jitAmdZenFP16();
    ~jitAmdZenFP16();
    jitAmdZenFP16(const jitAmdZenFP16&)            = delete;
    jitAmdZenFP16& operator=(const jitAmdZenFP16&) = delete;
    jitAmdZenFP16(jitAmdZenFP16&&)                 = delete;
    jitAmdZenFP16& operator=(jitAmdZenFP16&&)      = delete;

    int getProcessBlockSize() const;

    dlp::jit::jitGeneratorError generateAllKernels(
        const dlp::jit::jitGeneratorContext& jI);

    dlp::jit::jitGeneratorError operator()(
        const dlp::jit::jitGeneratorContext& jI) override
    {
        return generateAllKernels(jI);
    }

    std::vector<dlp::kernel_frame::kernelDatatype>& getKernelDatatypes()
        override
    {
        return mKernelDatatypes;
    }

    std::vector<dlp::cpu_utils::isaFeature>& getIsaFeaturesRequired() override
    {
        return mIsaFeaturesRequired;
    }

    dlp::kernels::kernelError executeKernel(
        dlp::kernels::kernelParams* _params) override;

    std::unique_ptr<jitGeneratorBase> clone() override
    {
        return std::make_unique<jitAmdZenFP16>();
    }
};

} // namespace amdzen::gen
