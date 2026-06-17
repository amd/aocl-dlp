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

#include "amdzen_generator.hh"
#include "arch_utils/arch_config_manager.hh"
#include "bf16_gemm_generator.hh"
#include "bf16_gemv_generator.hh"
#include "cpu_utils/cpu_features.hh"
#include "f32_gemm_generator.hh"
#include "f32_gemm_rd_generator.hh"
#include "f32_gemv_generator.hh"
#include "f32f16_gemm_generator.hh"
#include "f32f16_gemv_generator.hh"
#include "fp16_gemm_generator.hh"
#include "fp16_gemv_generator.hh"
#include "jit_register/jit_register.hh"
#include "s8_gemm_generator.hh"
#include "s8_gemv_generator.hh"
#include "traits.hh"
#include "u8s8_gemm_generator.hh"
#include "u8s8_gemv_generator.hh"
#include "utils/float16_types.hh"

namespace amdzen::gen {

// F32 JIT Generator
jitAmdZenFP32::jitAmdZenFP32()
    : mKernelDatatypes({ dlp::kernel_frame::kernelDatatype::f32f32f32of32 })
    , mIsaFeaturesRequired{ dlp::cpu_utils::isaFeature::avx2 }
    , kType(utils::kernelInstrType::none)
    , numElemsPerReg(1)     // Initializing with 1 to avoid div by zero
    , usingRDKernels(false) // Initialize to false
    , kernelSize(0)         // Initialize to 0, set when allocating kernels
{
}

jitAmdZenFP32::~jitAmdZenFP32()
{
    codeGenerators.clear();
    for (auto& block : kernelCodeBlocks) {
        block = nullptr;
    }
}

int
jitAmdZenFP32::getProcessBlockSize() const
{
    switch (kType) {
        case utils::kernelInstrType::avx512_zmm_32_reg:
            return NR; // Process full NR (64) elements at once

        case utils::kernelInstrType::avx512_ymm_32_reg:
            return NR / 2; // Process NR/2 (32) elements at once

        case utils::kernelInstrType::avx2_ymm_16_reg:
            return NR; // Process full NR (16) elements at once

        default:
            return 0; // Invalid/unsupported kernel type
    }
}

void
jitAmdZenFP32::setGeneratorKernelMetaInfo(
    dlp::kernel_frame::kernelInstrPreference kInstPref)
{
    kType = utils::kernelInstrType::none;
    switch (kInstPref) {
        case dlp::kernel_frame::kernelInstrPreference::avx512_zmm_favour: {
            kType = utils::kernelInstrType::avx512_zmm_32_reg;
            // Acquiring the SIMD width of the kernel type
            numElemsPerReg =
                traits::ArchitectureTraits<
                    utils::kernelInstrType::avx512_zmm_32_reg>::regBytes
                / sizeof(float);
            break;
        }
        case dlp::kernel_frame::kernelInstrPreference::avx512_ymm_favour: {
            kType = utils::kernelInstrType::avx512_ymm_32_reg;
            // Acquiring the SIMD width of the kernel type
            numElemsPerReg =
                traits::ArchitectureTraits<
                    utils::kernelInstrType::avx512_ymm_32_reg>::regBytes
                / sizeof(float);
            break;
        }
        case dlp::kernel_frame::kernelInstrPreference::avx512_xmm_favour: {
            kType = utils::kernelInstrType::avx512_xmm_32_reg;
            // Acquiring the SIMD width of the kernel type
            numElemsPerReg =
                traits::ArchitectureTraits<
                    utils::kernelInstrType::avx512_xmm_32_reg>::regBytes
                / sizeof(float);
            break;
        }
        case dlp::kernel_frame::kernelInstrPreference::avx2_ymm_favour: {
            kType = utils::kernelInstrType::avx2_ymm_16_reg;
            // Acquiring the SIMD width of the kernel type
            numElemsPerReg =
                traits::ArchitectureTraits<
                    utils::kernelInstrType::avx2_ymm_16_reg>::regBytes
                / sizeof(float);
            break;
        }
        case dlp::kernel_frame::kernelInstrPreference::avx2_xmm_favour: {
            kType = utils::kernelInstrType::avx2_xmm_16_reg;
            // Acquiring the SIMD width of the kernel type
            numElemsPerReg =
                traits::ArchitectureTraits<
                    utils::kernelInstrType::avx2_xmm_16_reg>::regBytes
                / sizeof(float);
            break;
        }
        default:
            break;
    }
}

dlp::jit::jitGeneratorError
jitAmdZenFP32::deriveGEMMNumNRVariants(const dlp::jit::jitGeneratorContext& jI)
{
    // ToDo : This variable is supposed to hold the details of number of
    // blocks of N to be processed in each iteration. At a later stage, this
    // can be used to update the value of pack kernel block size, if we
    // support multiple pack kernel block sizes. Currently this is needed
    // to make sure of using the correct block sizes for each kernels,
    // given the restrictions in the available pack kernels.
    int processBlockSize = getProcessBlockSize();

    termNRFringeRegCount        = 1;
    isGenLtKrnlForAvailFullKrnl = jI.kI.genLtKrnlForAvailFullKrnl;
    if (jI.kI.term_fringe_nr <= 0) {
        if (jI.kI.genLtKrnlForAvailFullKrnl) {
            // Generate kernels with multiples of numElemsPerReg along
            // with < version for each multiple. For example, lets say
            // NR=64, then following kernels are generated:
            // 64, lt64, 48, lt48, 32, lt32, 16, lt16.
            numNRVariants = 2 * (processBlockSize / numElemsPerReg);
        } else {
            // Generate kernels with multiples of numElemsPerReg and then
            // one kernel to handle "< numElemsPerReg" cases. For example,
            // lets say NR=64, then following kernels are generated:
            // 64, 48, 32, 16, lt16.
            numNRVariants = (processBlockSize / numElemsPerReg) + 1;
        }
    } else {
        if (jI.kI.term_fringe_nr > processBlockSize) {
            return dlp::jit::jitGeneratorError::badKernelInfo;
        } else {
            // It is expected DE will supply term_fringe_nr which is a
            // perfect multiple of numElemsPerReg. Additionally now a
            // term_fringe_nr and lt_term_fringe_nr will be generated.
            termNRFringeRegCount = jI.kI.term_fringe_nr / numElemsPerReg;
            if (jI.kI.genLtKrnlForAvailFullKrnl) {
                // Generate kernels with multiples of numElemsPerReg along
                // with < version for each multiple till term_fringe_nr.
                // For example, lets say NR=64 and term_fringe_nr=48, then
                // following kernels are generated:
                // 64, lt64, 48, lt48.
                numNRVariants = 2
                                * ((processBlockSize - jI.kI.term_fringe_nr
                                    + numElemsPerReg)
                                   / numElemsPerReg);
            } else {
                // Generate kernels with multiples of numElemsPerReg till
                // term_fringe_nr kernel and then one kernel to handle "<
                // term_fringe_nr" cases. For example if term_fringe_nr=48,
                // then following kernels are generated:
                // 64, 48, lt48.
                numNRVariants =
                    ((processBlockSize - jI.kI.term_fringe_nr + numElemsPerReg)
                     / numElemsPerReg)
                    + 1;
            }
        }
    }

    return dlp::jit::jitGeneratorError::success;
}

// This function is to be called inside the mr, nr loop for generating
// kernels.
void
jitAmdZenFP32::deriveGEMMNRAndMaskUse(int                     nr,
                                      utils::generatorParams& params,
                                      int& correspondingMainFringe)
{
    correspondingMainFringe = 0;
    if (isGenLtKrnlForAvailFullKrnl) {
        // This is the case where we generate both "==" and "<" kernels for
        // each multiple of numElemsPerReg including "0". Here index 0, 2, 4
        // (even indices) corresponds to lt fringe (<) kernels and 1, 3, 5
        // (odd indices) corresponds to fringe + main (==) kernels. e.g. if
        // numElemsPerReg=16 and NR=64, then following kernels are generated
        // (id:fringe): 0-lt16, 1-16, 2-lt32, 3-32, 4-lt48, 5-48, 6-lt64, 7-64.
        // Additionally, there is the option of configuring the smallest
        // fringe kernel using term_fringe_nr (from kernelInfo). e.g. if
        // term_fringe_nr=32, then the kernels generated are: (id:fringe):
        // 0-lt32, 1-32, 2-lt48, 3-48, 4-lt64, 5-64.
        // Now here since lt32 is the last fringe kernel, it will cater to
        // inputs originally meant for lt32, 16, and lt16. So for lt32, all
        // loads needs to be masked, since they can potentially access out of
        // bounds memory. Say n=7, and 2 ZMM registers are used for loads in
        // lt32, then loads to both the ZMM registers needs to be masked.
        if ((nr % 2) == 0) {
            // This is the lt fringe case that uses mask.
            params.useMask = true;
            if (nr == 0) {
                params.NR          = 0;
                params.numMaskRegs = termNRFringeRegCount;
            } else {
                params.NR =
                    ((termNRFringeRegCount - 1) + (nr / 2)) * numElemsPerReg;
                params.numMaskRegs = 1;
            }
            correspondingMainFringe =
                (termNRFringeRegCount + (nr / 2)) * numElemsPerReg;
        } else {
            // This is the fringe case without mask. No nr=0 or
            // even values for nr here.
            params.NR =
                ((termNRFringeRegCount - 1) + ((nr + 1) / 2)) * numElemsPerReg;
            params.useMask          = false;
            params.numMaskRegs      = 0;
            correspondingMainFringe = params.NR;
        }
    } else {
        // This is the case where we generate "==" kernels for each multiple
        // of numElemsPerReg including and 1 "<" kernel for the smallest
        // configured multiple of numElemsPerReg, "0". Here index 0 corresponds
        // to lt fringe (<) kernels and all the other odd indices corresponds
        // to fringe + main (==) kernels. e.g. if numElemsPerReg=16 and NR=64,
        // then following kernels are generated (id:fringe):
        // 0-lt16, 1-16, 2-32, 3-48, 4-64.
        // Additionally, there is the option of configuring the smallest
        // fringe kernel using term_fringe_nr (from kernelInfo). e.g., if
        // term_fringe_nr=32, then the kernels generated are (id:fringe):
        // 0-lt32, 1-32, 2-48, 3-64.
        // The usage of masks for all load registers in lt32 applies in a
        // similar manner as mentioned in the comment in
        // isGenLtKrnlForAvailFullKrnl case.
        if (nr == 0) {
            params.NR               = 0;
            params.useMask          = true;
            params.numMaskRegs      = termNRFringeRegCount;
            correspondingMainFringe = termNRFringeRegCount * numElemsPerReg;
        } else {
            params.NR      = ((termNRFringeRegCount - 1) + nr) * numElemsPerReg;
            params.useMask = false;
            params.numMaskRegs      = 0;
            correspondingMainFringe = params.NR;
        }
    }
}

/* Function to generate all kernels */
dlp::jit::jitGeneratorError
jitAmdZenFP32::generateAllKernels(const dlp::jit::jitGeneratorContext& jI)
{
    dlp::jit::jitGeneratorError err = dlp::jit::jitGeneratorError::error;

    MR              = (jI.kI).mr;
    NR              = (jI.kI).nr;
    KC              = (jI.kI).kc;
    K_UNROLL        = (jI.kI).k_unroll;
    PREFETCH_C_DIST = (jI.kI).prefetch_c_dist;
    c_downscale     = (jI.kI).c_downscale;

    // Hardcoding the FP32 kernel datatype for now
    const dlp::kernel_frame::kernelDatatype kdt =
        dlp::kernel_frame::kernelDatatype::f32f32f32of32;

    // Convert kernelInstrPreference to kernelType
    setGeneratorKernelMetaInfo(jI.kI.kInstPref);

    if (MR == 1) {
        // Generate kernel for GEMV(when MR == 1)
        AOCL_DLP_MEMORY_TAG mtag_b = (jI.kI).mtag_b;

        // We will be generating NR kernels, each having the main loop
        // and having the specific fringe case implemented.
        numKernelVariants = NR;
        kernelCodeBlocks.resize(numKernelVariants);
        kernelSize = utils::JIT_KERNEL_SIZE; // Standard size for GEMV M1

        utils::gemvM1GeneratorParams params(
            c_downscale, 0, 0, 0, 0, mtag_b, true, true, true, true,
            dlp::kernel_frame::storageFormat::rowMajor,
            (jI.kI).alphaScalingType, (jI.kI).betaScalingType, kType);

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            // Copy the kernelOps from the kernelInfo to params
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        params.NR         = NR;
        params.N_LEFT     = 0;
        params.KC         = KC;
        params.K_SUB_ITER = K_UNROLL;
        params.nloop      = true;
        params.kloop      = true;
        params.nfringe    = false;
        params.kfringe    = true;

        for (iter_t i = 0; i < NR; i += 1) {
            params.N_LEFT  = i;
            params.nfringe = (i != 0);

            std::unique_ptr<Xbyak::CodeGenerator> gen;
            switch (kType) {
                case utils::kernelInstrType::avx2_ymm_16_reg: {
                    auto g = std::make_unique<codegen::jitF32GEMVM1<
                        utils::kernelInstrType::avx2_ymm_16_reg>>(kernelSize);
                    err    = g->generateKernel(params);
                    gen    = std::move(g);
                    break;
                }
                case utils::kernelInstrType::avx512_ymm_32_reg: {
                    auto g = std::make_unique<codegen::jitF32GEMVM1<
                        utils::kernelInstrType::avx512_ymm_32_reg>>(kernelSize);
                    err    = g->generateKernel(params);
                    gen    = std::move(g);
                    break;
                }
                case utils::kernelInstrType::avx512_zmm_32_reg: {
                    auto g = std::make_unique<codegen::jitF32GEMVM1<
                        utils::kernelInstrType::avx512_zmm_32_reg>>(kernelSize);
                    err    = g->generateKernel(params);
                    gen    = std::move(g);
                    break;
                }
                default: {
                    err = dlp::jit::jitGeneratorError::error;
                    break;
                }
            }
            if (err != dlp::jit::jitGeneratorError::success) {
                goto cleanup;
            }
            // Must call ready() to readjust jump/branch targets with
            // respect to any new buffer created as part of AutoGrow mode
            // in Xbyak.
            gen->ready();
            kernelCodeBlocks[i] =
                const_cast<void*>(static_cast<const void*>(gen->getCode()));
            codeGenerators.push_back(std::move(gen));

            int n_left_suf = (i != 0) ? i : params.NR;
            // The file naming is as such : jit_gemv_m1_kernel.
            DLP_ENABLE_JIT_DUMP_AND_MONITOR(kernelCodeBlocks[i], kernelSize,
                                            "jit_gemv_m1_kernel", 1, n_left_suf,
                                            false, i);
        }

    } else if (NR == 1) {
        // Generate kernel for GEMV(when NR == 1)

        // Logic behind kernel generation:
        // 1. We generate kernels for all m_left values from 0 to MR-1.
        // 2. For each m_left, we generate 4 kernels:
        //    - 0: row-stored, without mloop
        //    - 1: row-stored, with mloop
        //    - 2: col-stored, without mloop
        //    - 3: col-stored, with mloop
        // Ex : m_left = 0, MR = 16, then we generate 4 kernels:
        //    - 0: row-stored, without mloop
        //    - 1: row-stored, with mloop
        //    - 2: col-stored, without mloop
        //    - 3: col-stored, with mloop
        // Ex : m_left = 1, MR = 16, then we generate 4 kernels:
        //    - 4: row-stored, without mloop
        //    - 5: row-stored, with mloop
        //    - 6: col-stored, without mloop
        //    - 7: col-stored, with mloop

        // If the input is say with m = 7 and MR = 16(col-stored Y), then we
        // would pick pick a kernel that handles col-storage of output, without
        // having the main loop. It would only have the necessay fringe case
        // code, specific to 7.

        // If the input is say with m = 23 and MR = 16(row-stored Y), then we
        // would pick pick a kernel that handles row-storage of output, that
        // also having the main loop. It decomposes m into 16 + 7.

        // TODO: We could extend the no-loop decomposition to handle sizes until
        //       MR + MR - 1. Since, we would not require any looping for such
        //       sizes.

        numKernelVariants = MR * 4; // Each MR has a kernel for vector and
                                    // element-wise loads/stores for C

        kernelCodeBlocks.resize(numKernelVariants);
        kernelSize = utils::JIT_KERNEL_SIZE; // Standard size for GEMV N1

        // Initializing with default values.
        utils::gemvN1GeneratorParams params(
            0, 0, c_downscale, false, false, false, false,
            dlp::kernel_frame::storageFormat::rowMajor,
            (jI.kI).alphaScalingType, (jI.kI).betaScalingType, kType);

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            // Copy the kernelOps from the kernelInfo to params
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        params.MR           = MR;
        params.mloop        = true;
        params.kloop        = true;
        params.mfringe      = false;
        params.kfringe      = true;
        params.aliasMrSplit = (jI.kI).aliasMrSplit;

        for (iter_t m_left = 0; m_left < MR; m_left++) {
            params.M_LEFT  = m_left;
            params.mfringe = (m_left != 0); // The first two kernels that we
                                            // generate are main kernels
            for (iter_t j = 0; j < 4; j++) {
                // We generate 4 kernels for each M_LEFT, and index them as
                // follows:
                // 0: row-stored, without mloop
                // 1: row-stored, with mloop
                // 2: col-stored, without mloop
                // 3: col-stored, with mloop
                params.mloop = ((j == 1) || (j == 3));
                params.yFormat =
                    ((j / 2) == 0) ? dlp::kernel_frame::storageFormat::rowMajor
                                   : dlp::kernel_frame::storageFormat::colMajor;

                std::unique_ptr<Xbyak::CodeGenerator> gen;
                switch (kType) {
                    case utils::kernelInstrType::avx512_zmm_32_reg: {
                        auto g = std::make_unique<codegen::jitF32GEMVN1<
                            utils::kernelInstrType::avx512_zmm_32_reg>>(
                            kernelSize);
                        err = g->generateKernel(params);
                        gen = std::move(g);
                        break;
                    }
                    case utils::kernelInstrType::avx512_ymm_32_reg: {
                        auto g = std::make_unique<codegen::jitF32GEMVN1<
                            utils::kernelInstrType::avx512_ymm_32_reg>>(
                            kernelSize);
                        err = g->generateKernel(params);
                        gen = std::move(g);
                        break;
                    }
                    case utils::kernelInstrType::avx2_ymm_16_reg: {
                        auto g = std::make_unique<codegen::jitF32GEMVN1<
                            utils::kernelInstrType::avx2_ymm_16_reg>>(
                            kernelSize);
                        err = g->generateKernel(params);
                        gen = std::move(g);
                        break;
                    }
                    default: {
                        err = dlp::jit::jitGeneratorError::error;
                        break;
                    }
                }
                if (err != dlp::jit::jitGeneratorError::success) {
                    goto cleanup;
                }
                // Must call ready() to readjust jump/branch targets with
                // respect to any new buffer created as part of AutoGrow mode
                // in Xbyak.
                gen->ready();
                kernelCodeBlocks[m_left * 4 + j] =
                    const_cast<void*>(static_cast<const void*>(gen->getCode()));
                codeGenerators.push_back(std::move(gen));

                int m_left_suf = (m_left != 0) ? m_left : params.MR;
                // The file naming is as such : id_jit_gemv_n1_kernels_MR_idx.
                // The idx represents what configuration was used to generate
                // the kernel.
                DLP_ENABLE_JIT_DUMP_AND_MONITOR(
                    kernelCodeBlocks[m_left * 4 + j], kernelSize,
                    "jit_gemv_n1_kernel", m_left_suf, j, false, m_left * 4 + j);
            }
        }
    } else if ((KC == 1) && (!jI.kI.invokeRD)) {
        int processBlockSize = getProcessBlockSize();

        numMRVariants = 1;
        // Generate: 1 mask kernel + kernels for 1x, 2x, 3x, 4x numElemsPerReg
        // For ZMM: mask(<=16), 16, 32, 48, 64 = 5 kernels total
        numNRVariants = (processBlockSize / numElemsPerReg) + 1;

        numKernelVariants = 1;
        kernelCodeBlocks.resize(numKernelVariants);
        // Single large kernel containing all NR variants
        // Each NR variant contains MR sub-variants, so total size is:
        // numNRVariants * MR * base_size_per_MR_variant
        // Using a conservative estimate for base size
        kernelSize = numNRVariants * MR * utils::JIT_KERNEL_SIZE;

        // Initializing with default values.
        utils::generatorParams params(
            0, 0, (jI.kI).k_unroll, 0, c_downscale, 0, false, false, true,
            (jI.kI).alphaScalingType, (jI.kI).betaScalingType, kType);

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            // Copy the kernelOps from the kernelInfo to params
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        params.MR = MR;
        params.NR = processBlockSize;

        {
            std::unique_ptr<Xbyak::CodeGenerator> gen;
            switch (kType) {
                case utils::kernelInstrType::avx512_zmm_32_reg: {
                    auto g = std::make_unique<GEMMcodeGenerator::jitGEMMF32<
                        utils::kernelInstrType::avx512_zmm_32_reg>>(kernelSize);
                    err    = g->generateKernel(params);
                    gen    = std::move(g);
                    break;
                }
                case utils::kernelInstrType::avx512_ymm_32_reg: {
                    auto g = std::make_unique<GEMMcodeGenerator::jitGEMMF32<
                        utils::kernelInstrType::avx512_ymm_32_reg>>(kernelSize);
                    err    = g->generateKernel(params);
                    gen    = std::move(g);
                    break;
                }
                case utils::kernelInstrType::avx2_ymm_16_reg: {
                    auto g = std::make_unique<GEMMcodeGenerator::jitGEMMF32<
                        utils::kernelInstrType::avx2_ymm_16_reg>>(kernelSize);
                    err    = g->generateKernel(params);
                    gen    = std::move(g);
                    break;
                }
                default: {
                    err = dlp::jit::jitGeneratorError::error;
                    break;
                }
            }
            if (err != dlp::jit::jitGeneratorError::success) {
                goto cleanup;
            }
            // Must call ready() to readjust jump/branch targets with
            // respect to any new buffer created as part of AutoGrow mode
            // in Xbyak.
            gen->ready();
            kernelCodeBlocks[0] =
                const_cast<void*>(static_cast<const void*>(gen->getCode()));
            codeGenerators.push_back(std::move(gen));
        }

        DLP_ENABLE_JIT_DUMP_AND_MONITOR(kernelCodeBlocks[0], kernelSize,
                                        "jit_kernel_k1", MR, processBlockSize,
                                        false, processBlockSize);
    } else {

        // If the invokeRD flag is set, then we generate the RD kernels.
        if (jI.kI.invokeRD) {
            return generateAllKernelsRD(jI);
        }

        err = deriveGEMMNumNRVariants(jI);
        if (err != dlp::jit::jitGeneratorError::success) {
            goto cleanup;
        }

        numMRVariants     = MR;
        numKernelVariants = numMRVariants * numNRVariants;

        kernelCodeBlocks.resize(numKernelVariants);
        kernelSize = utils::JIT_KERNEL_SIZE; // Standard size for general GEMM

        // Initializing with default values.
        utils::generatorParams params(
            0, 0, K_UNROLL, PREFETCH_C_DIST, c_downscale, 0, false, false,
            false, (jI.kI).alphaScalingType, (jI.kI).betaScalingType, kType);

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            // Copy the kernelOps from the kernelInfo to params
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        // Generate all kernels for the given MR and NR. Any per-variant
        // generator failure (after passing the feasibility filter below)
        // is fatal (goto cleanup): we want the existing fail-fast contract
        // to hold, so we never call generateKernel() for an (mr, nr) pair
        // we already know cannot fit.
        //
        // Feasibility filter: when the DE bumps MR (e.g. MR=16 for the
        // skinny-N n<=16 override), the wider-NR variants (NR>=32) would
        // exceed the register budget (cReg=MR*bReg, aReg = numRegs -
        // cReg - bReg - maskVecReg, must have aReg >= 1 -- mirrors
        // jitGEMMF32::allocateReg()). Skip those slots up front instead
        // of relying on the per-variant generator to return badKernelInfo.
        // The dispatcher only reaches the lt-mask kernel and the NR=16
        // full kernel for n<=16, both of which always pass the filter.
        const int kNumRegs =
            (kType == utils::kernelInstrType::avx2_ymm_16_reg) ? 16 : 32;
        for (iter_t mr = 0; mr < numMRVariants; mr++) {
            for (iter_t nr = 0; nr < numNRVariants; nr++) {
                params.MR    = mr == 0 ? MR : mr;
                params.mLoop = mr == 0;

                int correspondingMainFringe = 0;
                deriveGEMMNRAndMaskUse(nr, params, correspondingMainFringe);

                // Skinny-N override: when the DE has bumped MR via the
                // n<=16 override (jI.kI.skinnyN), only the lt-numElems
                // (nr=0) and the full numElems (nr=1) variants are ever
                // dispatched at runtime. The wider NR slots (nr>=2:
                // lt2x/2x/lt3x/3x/lt4x/4x) are unreachable AND would
                // exceed the register budget at bumped MR -- skip them
                // entirely so we don't waste codegen on dead kernels
                // (and don't trigger badKernelInfo on infeasible ones).
                if (jI.kI.skinnyN && nr >= 2) {
                    continue;
                }

                // Pre-filter register-infeasible (MR, NR) variants. This
                // mirrors jitGEMMF32<KType>::allocateReg(): bFullReg =
                // NR / numElemsPerReg, bMaskReg = useMask ? numMaskRegs
                // : 0, bReg = bFullReg + bMaskReg, cReg = MR * bReg.
                // For AVX2 ymm, the mask consumes vector registers
                // (maskVecReg = numMaskRegs); for AVX-512 the mask is in
                // Opmask regs and does not draw from the vector budget.
                {
                    int bFullReg = params.NR / numElemsPerReg;
                    int bMaskReg = params.useMask ? params.numMaskRegs : 0;
                    int bReg     = bFullReg + bMaskReg;
                    int cReg     = params.MR * bReg;
                    int maskVecReg =
                        (kType == utils::kernelInstrType::avx2_ymm_16_reg)
                            ? bMaskReg
                            : 0;
                    if (kNumRegs - cReg - bReg - maskVecReg < 1) {
                        // Slot stays nullptr (zero-initialized by
                        // resize). The dispatcher never reaches it for
                        // any DE-blessed shape that bumped MR.
                        continue;
                    }
                }

                std::unique_ptr<Xbyak::CodeGenerator> gen;
                switch (kType) {
                    case utils::kernelInstrType::avx512_zmm_32_reg: {
                        auto g = std::make_unique<GEMMcodeGenerator::jitGEMMF32<
                            utils::kernelInstrType::avx512_zmm_32_reg>>(
                            kernelSize);
                        err = g->generateKernel(params);
                        gen = std::move(g);
                        break;
                    }
                    case utils::kernelInstrType::avx512_ymm_32_reg: {
                        auto g = std::make_unique<GEMMcodeGenerator::jitGEMMF32<
                            utils::kernelInstrType::avx512_ymm_32_reg>>(
                            kernelSize);
                        err = g->generateKernel(params);
                        gen = std::move(g);
                        break;
                    }
                    case utils::kernelInstrType::avx2_ymm_16_reg: {
                        auto g = std::make_unique<GEMMcodeGenerator::jitGEMMF32<
                            utils::kernelInstrType::avx2_ymm_16_reg>>(
                            kernelSize);
                        err = g->generateKernel(params);
                        gen = std::move(g);
                        break;
                    }
                    default:
                        err = dlp::jit::jitGeneratorError::error;
                        break;
                }
                if (err != dlp::jit::jitGeneratorError::success) {
                    goto cleanup;
                }
                // Must call ready() to readjust jump/branch targets with
                // respect to any new buffer created as part of AutoGrow mode
                // in Xbyak.
                gen->ready();
                kernelCodeBlocks[mr * numNRVariants + nr] =
                    const_cast<void*>(static_cast<const void*>(gen->getCode()));
                codeGenerators.push_back(std::move(gen));

                // params.useMask=false implies a fringe or main kernel.
                // params.useMask=true implies a lt fringe or lt main kernel.
                DLP_ENABLE_JIT_DUMP_AND_MONITOR(
                    kernelCodeBlocks[mr * numNRVariants + nr], kernelSize,
                    "jit_kernel", params.MR, correspondingMainFringe,
                    params.useMask, mr * numNRVariants + nr);
            }
        }
    }

    return dlp::jit::jitGeneratorError::success;

cleanup:
    codeGenerators.clear();
    for (auto& block : kernelCodeBlocks) {
        block = nullptr;
    }
    return err;
}

dlp::jit::jitGeneratorError
jitAmdZenFP32::generateAllKernelsRD(const dlp::jit::jitGeneratorContext& jI)
{
    dlp::jit::jitGeneratorError err = dlp::jit::jitGeneratorError::error;

    MR = (jI.kI).mr;

    // In Rd kernels, for Avx2 path, we generate kernels upto MR 3 instead of 6.
    // so we need to set MR to 3 if the kernelInstrPreference is
    // avx2_ymm_16_reg.
    if (kType == utils::kernelInstrType::avx2_ymm_16_reg) {
        MR = 3;
    }

    NR          = (jI.kI).nr;
    KC          = (jI.kI).kc;
    K_UNROLL    = (jI.kI).k_unroll;
    c_downscale = (jI.kI).c_downscale;

    usingRDKernels = true;

    // Hardcoding the FP32 kernel datatype for now
    const dlp::kernel_frame::kernelDatatype kdt =
        dlp::kernel_frame::kernelDatatype::f32f32f32of32;

    // Convert kernelInstrPreference to kernelType
    setGeneratorKernelMetaInfo(jI.kI.kInstPref);

    int processBlockSize = getProcessBlockSize();

    numMRVariants = MR;
    numNRVariants =
        (processBlockSize / numElemsPerReg); // from NR to numElemsPerReg
    md_t dummy = numElemsPerReg / 2;         // from numElemsPerReg/2 to 1
    while (dummy > 0) {
        numNRVariants++;
        dummy /= 2;
    }

    numKernelVariants = numMRVariants * numNRVariants;
    kernelCodeBlocks.resize(numKernelVariants);

    // Initializing with default values.
    utils::generatorParams params(0, 0, (jI.kI).k_unroll, 0, c_downscale, 0,
                                  false, false, false, (jI.kI).alphaScalingType,
                                  (jI.kI).betaScalingType, kType);

    for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
        // Copy the kernelOps from the kernelInfo to params
        params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
    }

    int nElemsPerRegLog2 = amdzen::utils::int_log2(numElemsPerReg);

    // Generate all kernels for the given MR and NR
    for (iter_t mr = 0; mr < numMRVariants; mr++) {
        for (iter_t nr = 0; nr < numNRVariants; nr++) {
            params.MR    = mr == 0 ? MR : mr;
            params.mLoop = mr == 0;

            params.NR = nr <= nElemsPerRegLog2
                            ? (1 << nr)
                            : (nr - nElemsPerRegLog2 + 1) * numElemsPerReg;

            // always set and use mask as the same mask will be needed in
            // post-ops as we are processing only 4 elements(XMM) at a time.
            params.useMask     = true;
            params.numMaskRegs = params.useMask ? 1 : 0;

            std::unique_ptr<Xbyak::CodeGenerator> gen;
            switch (kType) {
                case utils::kernelInstrType::avx512_zmm_32_reg: {
                    auto g = std::make_unique<GEMMcodeGenerator::jitGEMMF32RD<
                        utils::kernelInstrType::avx512_zmm_32_reg>>(
                        utils::JIT_KERNEL_SIZE);
                    err = g->generateKernel(params);
                    gen = std::move(g);
                    break;
                }
                case utils::kernelInstrType::avx512_ymm_32_reg: {
                    auto g = std::make_unique<GEMMcodeGenerator::jitGEMMF32RD<
                        utils::kernelInstrType::avx512_ymm_32_reg>>(
                        utils::JIT_KERNEL_SIZE);
                    err = g->generateKernel(params);
                    gen = std::move(g);
                    break;
                }
                case utils::kernelInstrType::avx2_ymm_16_reg: {
                    auto g = std::make_unique<GEMMcodeGenerator::jitGEMMF32RD<
                        utils::kernelInstrType::avx2_ymm_16_reg>>(
                        utils::JIT_KERNEL_SIZE);
                    err = g->generateKernel(params);
                    gen = std::move(g);
                    break;
                }
                default:
                    err = dlp::jit::jitGeneratorError::error;
                    break;
            }
            if (err != dlp::jit::jitGeneratorError::success) {
                goto cleanup;
            }
            // Must call ready() to readjust jump/branch targets with
            // respect to any new buffer created as part of AutoGrow mode
            // in Xbyak.
            gen->ready();
            kernelCodeBlocks[mr * numNRVariants + nr] =
                const_cast<void*>(static_cast<const void*>(gen->getCode()));
            codeGenerators.push_back(std::move(gen));

            // params.useMask=false implies a fringe or main kernel.
            // params.useMask=true implies a lt fringe or lt main kernel.
            DLP_ENABLE_JIT_DUMP_AND_MONITOR(
                kernelCodeBlocks[mr * numNRVariants + nr],
                utils::JIT_KERNEL_SIZE, "jit_kernel_rd", params.MR, params.NR,
                false, mr * numNRVariants + nr);
        }
    }

    return dlp::jit::jitGeneratorError::success;

cleanup:
    codeGenerators.clear();
    for (auto& block : kernelCodeBlocks) {
        block = nullptr;
    }
    return err;
}
void
jitAmdZenFP32::setMaskForGEMMLtFringe(dlp::kernels::gemmParams* params,
                                      int                       nRemainder)
{
    if (kType == utils::kernelInstrType::avx512_zmm_32_reg) {
        // The support for creating all lt fringe kernels (lt48, lt32) is
        // only supported for avx512 when using zmm registers.
        int nRemainderRegCount = nRemainder / numElemsPerReg;
        for (iter_t ii = 0; ii < nRemainderRegCount; ++ii) {
            params->maskF32[ii] = 0xFFFF;
        }
        params->maskF32[nRemainderRegCount] =
            0xFFFF >> (numElemsPerReg
                       - (nRemainder - (nRemainderRegCount * numElemsPerReg)));
        for (iter_t ii = nRemainderRegCount + 1; ii < dlp::kernels::maxNumMasks;
             ++ii) {
            params->maskF32[ii] = 0x0;
        }
    } else if (kType == utils::kernelInstrType::avx512_ymm_32_reg) {
        params->maskF32_8[0] = 0xFF >> (numElemsPerReg - nRemainder);
    } else if (kType == utils::kernelInstrType::avx2_ymm_16_reg) {
        for (iter_t i = 0; i < 8; i++) {
            params->maskArray[i] = (i < nRemainder) ? 0xFFFFFFFF : 0;
        }
    }
}

dlp::kernels::kernelError
jitAmdZenFP32::executeKernel(dlp::kernels::kernelParams* _params)
{
    // For GEMV(M1) execution
    if (MR == 1) {
        auto params = static_cast<dlp::kernels::gemvM1Params*>(_params);
        // Setting the remaining values of gemvM1Params(that are not set as
        // part of it's parameterized constructor)

        params->n_iter = params->n / NR;
        params->n_left = params->n % NR;
        params->k_iter = params->k / KC;
        params->k_left = params->k % KC;

        params->k_iter_sub_iter = KC / K_UNROLL;
        params->k_iter_sub_left = KC % K_UNROLL;
        params->k_left_sub_iter = (params->k_left) / K_UNROLL;
        params->k_left_sub_left = (params->k_left) % K_UNROLL;

        int full_bands =
            params->n_left / numElemsPerReg; // Number of complete 8-element
        int partial_elements =
            params->n_left % numElemsPerReg; // Elements in partial band

        if (params->n_left > 0) {
            switch (kType) {
                case utils::kernelInstrType::avx512_zmm_32_reg: {
                    params->nmask_avx512 =
                        0xFFFF >> (numElemsPerReg - partial_elements);
                    break;
                }
                case utils::kernelInstrType::avx512_ymm_32_reg: {
                    params->nmask_avx512_256 =
                        0xFF >> (numElemsPerReg - partial_elements);
                    break;
                }
                case utils::kernelInstrType::avx2_ymm_16_reg: {
                    for (iter_t i = 0; i < 8; i++) {
                        params->nmask_avx2[i] = (i < partial_elements) ? -1 : 0;
                    }
                    break;
                }
                default:
                    break;
            }
        }

        // Deploy the associated kernel
        md_t kernel_idx = params->n_left;

        utils::jit_gemv_m1_kernel kernel =
            reinterpret_cast<utils::jit_gemv_m1_kernel>(
                kernelCodeBlocks[kernel_idx]);
        kernel(params);

        // Update post_op_c_j by the total n processed in this kernel call
        // (similar to GEMM pattern where post_op_c_j += elementsToProcess)
        (params->kernelOpsAttr).post_op_c_j += params->n;

    } else if (NR == 1) {
        auto params = static_cast<dlp::kernels::gemvN1Params*>(_params);
        // Setting the remaining values of gemvN1Params(that are not set as
        // part of it's parameterized constructor) NOTE : The associated NR
        // value for the generated kernels is 16.
        //        The mdulo operations are added when calcuating the masks,
        //        to make sure we use the same code for generic NR values in
        //        future.
        params->m_iter = params->m / MR;
        params->m_left = params->m % MR;
        params->k_iter = params->k / numElemsPerReg;
        params->k_left = params->k % numElemsPerReg;

        if (kType == utils::kernelInstrType::avx512_zmm_32_reg) {
            params->mmask_avx512 =
                0xFFFF >> (numElemsPerReg - (params->m_left % numElemsPerReg));
            params->kmask_avx512 =
                0xFFFF >> (numElemsPerReg - (params->k_left) % numElemsPerReg);
        } else if (kType == utils::kernelInstrType::avx512_ymm_32_reg) {
            params->mmask_avx512_256 =
                0xFF >> (numElemsPerReg - (params->m_left) % numElemsPerReg);
            params->kmask_avx512_256 =
                0xFF >> (numElemsPerReg - (params->k_left) % numElemsPerReg);
        } else if (kType == utils::kernelInstrType::avx2_ymm_16_reg) {
            int m_iter_left = (params->m_left) % numElemsPerReg;
            int k_iter_left = (params->k_left) % numElemsPerReg;
            for (iter_t i = 0; i < m_iter_left; i++) {
                params->mmask_avx2[i] = -1;
            }
            for (iter_t i = 0; i < k_iter_left; i++) {
                params->kmask_avx2[i] = -1;
            }
        }

        int is_m_loop     = ((params->m) >= MR);
        int is_col_stored = ((params->rsC) == 1);

        int kernel_idx = params->m_left * 4 + is_col_stored * 2 + is_m_loop;

        // Deploy the associated kernel
        utils::jit_gemv_n1_kernel kernel =
            reinterpret_cast<utils::jit_gemv_n1_kernel>(
                kernelCodeBlocks[kernel_idx]);
        kernel(params);

    } else if ((KC == 1) && (!usingRDKernels)) {
        // Special execution path for k=1 fused kernels
        // K=1 kernels have both M and N loops internal to the JIT kernel
        // Framework just sets up parameters and calls the kernel once
        // Note: It is expected the generateAllKernels is called before
        // calling this function.
        auto params = static_cast<dlp::kernels::gemmParams*>(_params);

        int processBlockSize = getProcessBlockSize();

        // Set up M-loop parameters
        int mFullPieces    = params->m / MR;
        int mPartialPieces = params->m % MR;
        params->mIter      = mFullPieces;
        params->mLeft      = mPartialPieces;

        // Set up N-loop parameters for the JIT kernel
        // For k=1 kernels, we call ONE kernel that handles the full NR
        // dimension The kernel variant to call is determined by the
        // processBlockSize (NR value)
        md_t n = params->n;

        // Set nIter and nLeft for the kernel
        // The kernel will handle multiple NR iterations if nIter > 0
        params->nIter =
            n / processBlockSize; // Number of full processBlockSize chunks
        params->nLeft   = n % processBlockSize; // Remaining elements
        md_t nRemainder = n % numElemsPerReg;
        if (nRemainder > 0) {
            setMaskForGEMMLtFringe(params, nRemainder);
        }

        int kernel_idx = 0; // single kernel index for k=1 kernels

        // Call the k=1 fused kernel - it handles M loop internally
        // N-loop will be added in next step
        auto kernel = reinterpret_cast<void (*)(dlp::kernels::gemmParams*)>(
            kernelCodeBlocks[kernel_idx]);
        kernel(params);

    } else {

        if (usingRDKernels) {
            return executeKernelRD(_params);
        }

        // Note: It is expected the generateAllKernels is called before
        // calling this function.
        auto     params = static_cast<dlp::kernels::gemmParams*>(_params);
        int      processBlockSize = getProcessBlockSize();
        uint64_t og_post_op_c_i   = (params->kernelOpsAttr).post_op_c_i;
        uint64_t og_post_op_c_j   = (params->kernelOpsAttr).post_op_c_j;
        float*   aPtr             = static_cast<float*>(params->a);
        float*   og_BPtr          = static_cast<float*>(params->b);
        float*   og_CPtr          = static_cast<float*>(params->c);
        md_t     og_n             = params->n;
        md_t     og_m             = params->m;
        md_t     og_k             = params->k;

        for (iter_t jr = 0; jr < og_n; jr += NR) {
            // Post ops meta attributes.
            params->kernelOpsAttr.post_op_c_j = og_post_op_c_j + jr;

            md_t n    = dlp_min((og_n - jr), NR);
            params->n = n;
            params->m = og_m;
            params->k = og_k;

            int mFullPieces    = og_m / MR;
            int mPartialPieces = og_m % MR;

            params->kIterBP = og_k / K_UNROLL;
            params->kLeft   = og_k % K_UNROLL;

            // Initialize pointers.
            float* bPtr = og_BPtr + (jr * params->psB);
            params->b   = bPtr;
            float* c_jr = og_CPtr + (jr * params->csC);

            // OVERVIEW:
            // This unified approach works for all kernel types (ZMM, YMM,
            // AVX2). The key insight is that all architectures follow the same
            // pattern: process elements in blocks, decompose fringe blocks into
            // complete registers + remainder, then execute appropriate kernels.
            //
            // EXECUTION LOGIC:
            // 1. Process elements in chunks of 'processBlockSize':
            //    - ZMM: processBlockSize = 64 (full NR), numElemsPerReg = 16
            //    - YMM: processBlockSize = 32 (NR/2), numElemsPerReg = 8
            //
            // 2. For each chunk, decompose into complete SIMD registers +
            // remainder:
            //    - Complete registers: Use kernel_idx = nFullpieces (no
            //    masking)
            //    - Remainder elements: Use kernel_idx = 0 (mask kernel with
            //    masking)
            //
            // 3. Execute kernels in sequence: complete registers first, then
            // remainder
            //
            // EXAMPLE WALKTHROUGH (n=15 with YMM):
            // - processBlockSize=32, numElemsPerReg=8
            // - Iteration 1: nBlockSize = min(15, 32) = 15
            //   - nFullpieces = 15/8 = 1  -> Execute kernel_idx=1 for 8
            //   elements (no mask)
            //   - nRemainder = 15%8 = 7   -> Execute kernel_idx=0 for 7
            //   elements (with mask)
            //   - Total processed: 8 + 7 = 15, n becomes 0, loop exits
            //
            // EXAMPLE WALKTHROUGH (n=95 with YMM):
            // - Iteration 1: nBlockSize = min(95, 32) = 32
            //   - nFullpieces = 32/8 = 4  -> Execute kernel_idx=4 for 32
            //   elements (no mask)
            //   - nRemainder = 32%8 = 0   -> No remainder processing
            //   - Processed: 32, n becomes 63
            // - Iteration 2: nBlockSize = min(63, 32) = 32 -> Process 32 more
            // elements
            //   - Processed: 32, n becomes 31
            // - Iteration 3: nBlockSize = min(31, 32) = 31
            //   - nFullpieces = 31/8 = 3  -> Execute kernel_idx=3 for 24
            //   elements (no mask)
            //   - nRemainder = 31%8 = 7   -> Execute kernel_idx=0 for 7
            //   elements (with mask)
            //   - Processed: 24 + 7 = 31, n becomes 0, loop exits

            while (n > 0) {
                // Its expected that every NR value is multiple of
                // numElemsPerReg.
                int nBlockSize = (n >= processBlockSize) ? processBlockSize : n;
                int nFullpieces = nBlockSize / numElemsPerReg;
                int nRemainder =
                    ((nFullpieces - termNRFringeRegCount) >= 0)
                        ? (nBlockSize - (nFullpieces * numElemsPerReg))
                        : nBlockSize;

                if (isGenLtKrnlForAvailFullKrnl) {
                    // Case where we generate both "==" and "<" kernels for
                    // each multiple of numElemsPerReg including "0".
                    int idBase = ((nFullpieces - termNRFringeRegCount) >= 0)
                                     ? (nFullpieces - termNRFringeRegCount)
                                     : -1;
                    int kernel_n_idx = (2 * (idBase + 1)) - 1;
                    if (nRemainder > 0) {
                        setMaskForGEMMLtFringe(params, nRemainder);
                        kernel_n_idx += 1;
                    }

                    int elementsToProcess = nBlockSize;
                    executeGEMMMLoop(params, mFullPieces, mPartialPieces,
                                     kernel_n_idx, elementsToProcess, n, &c_jr,
                                     og_post_op_c_i, aPtr);
                } else {
                    // Case where we generate "==" kernels for each multiple
                    // of numElemsPerReg including and 1 "<" kernel for the
                    // smallest multiple of numElemsPerReg, "0".
                    if (nFullpieces >= termNRFringeRegCount) {
                        int kernel_n_idx =
                            nFullpieces - termNRFringeRegCount + 1;
                        int elementsToProcess = nFullpieces * numElemsPerReg;
                        executeGEMMMLoop(params, mFullPieces, mPartialPieces,
                                         kernel_n_idx, elementsToProcess, n,
                                         &c_jr, og_post_op_c_i, aPtr);
                    }

                    // Process remainder with mask (if any)
                    if (nRemainder > 0) {
                        setMaskForGEMMLtFringe(params, nRemainder);
                        int kernel_n_idx =
                            0; // Use lt mask kernel for nRemainder.
                        executeGEMMMLoop(params, mFullPieces, mPartialPieces,
                                         kernel_n_idx, nRemainder, n, &c_jr,
                                         og_post_op_c_i, aPtr);
                    }
                }
            }
        }
    }

    return dlp::kernels::kernelError::success;
}

dlp::kernels::kernelError
jitAmdZenFP32::executeKernelRD(dlp::kernels::kernelParams* _params)
{
    auto     params           = static_cast<dlp::kernels::gemmParams*>(_params);
    int      processBlockSize = getProcessBlockSize();
    uint64_t og_post_op_c_i   = (params->kernelOpsAttr).post_op_c_i;
    uint64_t og_post_op_c_j_nrloop = (params->kernelOpsAttr).post_op_c_j;
    float*   aPtr                  = static_cast<float*>(params->a);
    float*   og_BPtr               = static_cast<float*>(params->b);
    float*   og_CPtr               = static_cast<float*>(params->c);
    md_t     og_n                  = params->n;
    md_t     og_m                  = params->m;
    md_t     og_k                  = params->k;

    // Iterate over N in NR-sized panels (framework no longer runs the JR loop).
    // The inner while(n) loop further decomposes each NR panel into
    // processBlockSize / vector-sized chunks (main + fringe).
    for (iter_t jr = 0; jr < og_n; jr += NR) {
        // Post ops meta attributes.
        params->kernelOpsAttr.post_op_c_j = og_post_op_c_j_nrloop + jr;
        uint64_t og_post_op_c_j           = (params->kernelOpsAttr).post_op_c_j;

        md_t n    = dlp_min((og_n - jr), NR);
        params->n = n;
        params->m = og_m;
        params->k = og_k;

        int mFullPieces    = og_m / MR;
        int mPartialPieces = og_m % MR;

        params->kIterBP = og_k / (K_UNROLL * numElemsPerReg);
        params->kLeft   = og_k % (K_UNROLL * numElemsPerReg);

        // Initialize pointers.
        float* bPtr = og_BPtr + (jr * params->psB);
        params->b   = bPtr;
        float* c_jr = og_CPtr + (jr * params->csC);

        // In RD kernels, mask is set for k-dimension.
        setMaskForGEMMLtFringe(params, (params->kLeft % numElemsPerReg));

        int kernel_n_idx = 0, kernel_m_idx = 0;
        int elementsToProcess = 0;

        // MR and psA are passed as 6 and 6*rs_a respectively from the
        // framework. since we are using MR as 3 for Avx2 path, we recalculate
        // and update psA here. Since mtag_a will always be unpacked, it is safe
        // to overwrite psA to MR * rs_a here.
        params->psA = MR * (params->rsA);

        int nElemsPerRegLog2 = amdzen::utils::int_log2(numElemsPerReg);
        int power            = nElemsPerRegLog2;

        while (n > 0) {
            int nBlockSize  = (n >= processBlockSize) ? processBlockSize : n;
            int nFullpieces = nBlockSize / numElemsPerReg;
            int nRemainder  = nBlockSize % numElemsPerReg;

            // process multiples of numElemsPerReg
            if (nFullpieces >= 1) {
                kernel_n_idx      = nElemsPerRegLog2 + nFullpieces - 1;
                elementsToProcess = nFullpieces * numElemsPerReg;
                executeGEMMMLoop(params, mFullPieces, mPartialPieces,
                                 kernel_n_idx, elementsToProcess, n, &c_jr,
                                 og_post_op_c_i, aPtr);

                // post_op_c_j attribute is updated inside assembly code. so
                // reset it to the original value and increment properly.
                og_post_op_c_j += elementsToProcess;
                (params->kernelOpsAttr).post_op_c_j = og_post_op_c_j;
            }
            // process remainder by dividing into powers of 2
            // For example, if nRemainder is 15, then we will process 8, 4,
            // 2, 1.
            if (nRemainder > 0) {
                int n_chunk = (1 << power); // 2^power
                if (n >= n_chunk) {
                    kernel_n_idx      = power;
                    elementsToProcess = n_chunk;
                    executeGEMMMLoop(params, mFullPieces, mPartialPieces,
                                     kernel_n_idx, elementsToProcess, n, &c_jr,
                                     og_post_op_c_i, aPtr);

                    // post_op_c_j attribute is updated inside assembly code. so
                    // reset it to the original value and increment properly.
                    og_post_op_c_j += elementsToProcess;
                    (params->kernelOpsAttr).post_op_c_j = og_post_op_c_j;
                }
                power--;
            }
        }
    }
    return dlp::kernels::kernelError::success;
}

// BF16 JIT Generator
jitAmdZenBF16::jitAmdZenBF16()
    : mKernelDatatypes({ dlp::kernel_frame::kernelDatatype::bf16bf16f32of32,
                         dlp::kernel_frame::kernelDatatype::bf16bf16f32obf16 })
    , mIsaFeaturesRequired{ dlp::cpu_utils::isaFeature::avx2 }
    , kType(utils::kernelInstrType::none)
    , numElemsPerReg(1) // Initializing with 1 to avoid div by zero
    , f32JitGenerator(nullptr)
{
}

jitAmdZenBF16::~jitAmdZenBF16()
{
    codeGenerators.clear();
    for (auto& block : kernelCodeBlocks) {
        block = nullptr;
    }
}

dlp::jit::jitGeneratorError
jitAmdZenBF16::generateAllKernels(const dlp::jit::jitGeneratorContext& jI)
{

    dlp::jit::jitGeneratorError err = dlp::jit::jitGeneratorError::error;

    MR              = (jI.kI).mr;
    NR              = (jI.kI).nr;
    KC              = (jI.kI).kc;
    K_UNROLL        = (jI.kI).k_unroll;
    PREFETCH_C_DIST = (jI.kI).prefetch_c_dist;
    c_downscale     = (jI.kI).c_downscale;

    // Reroute to f32 JIT generator if kernel instruction preference is
    // avx2_ymm_favour (avx2), avx512_zmm_favour (no avx512_bf16 support)
    if (((jI.kI).kInstPref
         == dlp::kernel_frame::kernelInstrPreference::avx2_ymm_favour)
        || ((jI.kI).kInstPref
            == dlp::kernel_frame::kernelInstrPreference::avx512_zmm_favour)) {
        // Ideally, a try-catch block should handle a memory allocation failure
        // here.
        f32JitGenerator = std::make_unique<jitAmdZenFP32>();
        return (*f32JitGenerator)(jI);
    }

    // This code-section is taken if the underlying architecture supports
    // AVX512-BF16.

    // Setting the kernel type based on the instruction preference
    // kernel instruction preference is set avx512_zmm_bf16_favour by DE when
    // underlying machine has avx512_bf16 support
    kType =
        ((jI.kI).kInstPref
         == dlp::kernel_frame::kernelInstrPreference::avx512_zmm_bf16_favour)
            ? utils::kernelInstrType::avx512_zmm_32_reg
            : utils::kernelInstrType::none;

    if (kType == utils::kernelInstrType::none) {
        err = dlp::jit::jitGeneratorError::notSupported;
        goto cleanup;
    }

    if (MR == 1) {

        // mtag_b is always packed by default, can be removed
        AOCL_DLP_MEMORY_TAG mtag_b = (jI.kI).mtag_b;

        // Number of F32(accumulators) elements that can fit in an avx512 zmm
        // register
        numElemsPerReg = 16;

        // We will be generating NR kernels, each having the main loop
        // and having the specific fringe case implemented.
        numKernelVariants = NR;
        kernelCodeBlocks.resize(numKernelVariants);

        utils::gemvM1GeneratorParams params(
            c_downscale, 0, 0, 0, 0, mtag_b, true, true, true, true,
            dlp::kernel_frame::storageFormat::rowMajor,
            (jI.kI).alphaScalingType, (jI.kI).betaScalingType, kType);

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            // Copy the kernelOps from the kernelInfo to params
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        params.NR               = NR;
        params.N_LEFT           = 0;
        params.N_LEFT_16        = 0;
        params.N_LEFT_LT16      = 0;
        params.KC               = KC;
        params.K_SUB_ITER       = K_UNROLL;
        params.nloop            = true;
        params.nfringe          = false;
        params.kloop            = true;
        params.nfringe_main     = false;
        params.nfringe_left     = false;
        params.kfringe          = true;
        params.RS_B_N_LEFT_16   = 0;
        params.RS_B_N_LEFT_LT16 = 0;

        for (iter_t i = 0; i < NR; i += 1) {

            params.N_LEFT      = i;
            params.N_LEFT_16   = (i / 16) * 16;
            params.N_LEFT_LT16 = i % 16;

            params.nfringe      = (i != 0);
            params.nfringe_main = (params.N_LEFT_16 != 0);
            params.nfringe_left = (params.N_LEFT_LT16 != 0);

            params.RS_B_N_LEFT_16   = (params.N_LEFT_16) * 2;
            params.RS_B_N_LEFT_LT16 = 32;

            auto gen = std::make_unique<codegen::jitBF16GEMVM1<
                utils::kernelInstrType::avx512_zmm_32_reg>>(
                utils::JIT_KERNEL_SIZE);
            err = gen->generateKernel(params);

            if (err != dlp::jit::jitGeneratorError::success) {
                goto cleanup;
            }
            // Must call ready() to readjust jump/branch targets with
            // respect to any new buffer created as part of AutoGrow mode
            // in Xbyak.
            gen->ready();
            kernelCodeBlocks[i] =
                const_cast<void*>(static_cast<const void*>(gen->getCode()));
            codeGenerators.push_back(std::move(gen));

            int n_left_suf = (i != 0) ? i : params.NR;
            // The file naming is as such : jit_gemv_m1_kernel.
            DLP_ENABLE_JIT_DUMP_AND_MONITOR(
                kernelCodeBlocks[i], utils::JIT_KERNEL_SIZE,
                "jit_bf16_gemv_m1_kernel", 1, n_left_suf, false, i);
        }
    } else if (NR == 1) {
        // Logic behind kernel generation:
        // 1. We generate kernels for all m_left values from 0 to MR-1.
        // 2. For each m_left, we generate 4 kernels:
        //    - 0: row-stored, without mloop
        //    - 1: row-stored, with mloop
        //    - 2: col-stored, without mloop
        //    - 3: col-stored, with mloop

        numElemsPerReg = 32;

        numKernelVariants = MR * 4; // Each MR has a kernel for vector and
                                    // element-wise loads/stores for C

        kernelCodeBlocks.resize(numKernelVariants);

        utils::gemvN1GeneratorParams params(
            MR, 0, c_downscale, false, false, false, false,
            dlp::kernel_frame::storageFormat::rowMajor,
            (jI.kI).alphaScalingType, (jI.kI).betaScalingType, kType);

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            // Copy the kernelOps from the kernelInfo to params
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        params.MR           = MR;
        params.mloop        = true;
        params.kloop        = true;
        params.mfringe      = false;
        params.kfringe      = true;
        params.aliasMrSplit = (jI.kI).aliasMrSplit;

        for (iter_t m_left = 0; m_left < MR; m_left++) {
            params.M_LEFT  = m_left;
            params.mfringe = (m_left != 0);
            // Generate 4 kernels for each m_left
            for (iter_t variant = 0; variant < 4; variant++) {
                params.mloop = (variant == 1)
                               || (variant == 3); // 1,3 has mloop true
                params.yFormat =
                    ((variant / 2) == 0)
                        ? dlp::kernel_frame::storageFormat::rowMajor
                        : dlp::kernel_frame::storageFormat::
                              colMajor; // 2,3 has column major true

                auto gen = std::make_unique<codegen::jitBF16GEMVN1<
                    utils::kernelInstrType::avx512_zmm_32_reg>>(
                    utils::JIT_KERNEL_SIZE);
                err = gen->generateKernel(params);
                if (err != dlp::jit::jitGeneratorError::success) {
                    goto cleanup;
                }
                // Must call ready() to readjust jump/branch targets with
                // respect to any new buffer created as part of AutoGrow mode
                // in Xbyak.
                gen->ready();
                kernelCodeBlocks[m_left * 4 + variant] =
                    const_cast<void*>(static_cast<const void*>(gen->getCode()));
                codeGenerators.push_back(std::move(gen));

                DLP_ENABLE_JIT_DUMP_AND_MONITOR(
                    kernelCodeBlocks[m_left * 4 + variant],
                    utils::JIT_KERNEL_SIZE, "jit_bf16_gemv_n1_kernel", m_left,
                    variant, false, m_left * 4 + variant);
            }
        }

    } else {
        // Here, we only generate kernels with multiples of numElemsPerReg
        // and then one kernel to handle "< numElemsPerReg" cases.
        // Here, the problem will be divided first by NR and the fringe will be
        // divided further into two regions. One for "multiples of
        // numElemsPerReg" and the other for "< numElemsPerReg" cases.

        // This approach works well with the current reordering strategy but is
        // inefficient for the cases where n < NR cases especially with "lt16"
        // fringe being taken.

        // We set the numElemsPerReg to 16(hardcode) here since we intend to
        // generate kernels only with the AVX512BF16 ISA. The utilities to set
        // numElemsPerReg and the kernel meta-data is common, and this has to be
        // abstracted out.
        numElemsPerReg = 16;

        numNRVariants     = (NR / numElemsPerReg) + 1;
        numMRVariants     = MR;
        numKernelVariants = numMRVariants * numNRVariants;

        kernelCodeBlocks.resize(numKernelVariants);

        utils::generatorParams params(
            0, 0, K_UNROLL, PREFETCH_C_DIST, c_downscale, 0, false, false,
            false, (jI.kI).alphaScalingType, (jI.kI).betaScalingType, kType);

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            // Copy the kernelOps from the kernelInfo to params
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        // Generate all kernels for the given MR and NR. Any per-variant
        // generator failure (after passing the feasibility filter below)
        // is fatal (goto cleanup): we want the existing fail-fast contract
        // to hold, so we never call generateKernel() for an (mr, nr) pair
        // we already know cannot fit.
        //
        // Feasibility filter: when the DE bumps MR (e.g. MR=16 for the
        // skinny-N n<=16 override), the wider-NR variants (NR>=32) would
        // exceed the 32-ZMM budget (cReg=MR*bReg, must have aReg = 32 -
        // cReg - bReg >= aRegMin -- mirrors jitGEMMBF16::allocateReg()).
        // Skip those slots up front instead of relying on the per-variant
        // generator to return badKernelInfo. The dispatcher only reaches
        // the lt16-mask kernel and the NR=16 full kernel for n<=16, both
        // of which always pass the filter.
        constexpr int kZmmRegs = 32;
        const int     aRegMin  = ((jI.kI).c_downscale < DLP_F32) ? 2 : 1;
        for (iter_t mr = 0; mr < numMRVariants; mr++) {
            for (iter_t nr = 0; nr < numNRVariants; nr++) {
                params.MR          = (mr == 0) ? MR : mr;
                params.mLoop       = (mr == 0);
                params.NR          = (nr * numElemsPerReg);
                params.useMask     = (nr == 0);
                params.numMaskRegs = (params.useMask) ? 1 : 0;

                // Skinny-N override: when the DE has bumped MR via the
                // n<=16 override (jI.kI.skinnyN), only the lt16 (nr=0)
                // and full-16 (nr=1) variants are ever dispatched. The
                // wider NR slots (nr>=2: lt32/32/lt48/48/lt64/64) are
                // unreachable AND would exceed the 32-ZMM budget at
                // bumped MR -- skip them entirely so we don't waste
                // codegen on dead kernels (and don't trigger
                // badKernelInfo on infeasible ones).
                if (jI.kI.skinnyN && nr >= 2) {
                    continue;
                }

                // For BF16 ZMM: bFullReg = (2*NR)/nBF16ElemsPerReg
                // = NR/16 = nr (with numElemsPerReg=16). bMaskReg=1 for
                // useMask, else 0. So bReg = max(1, nr).
                int bReg = (nr == 0) ? 1 : static_cast<int>(nr);
                int cReg = params.MR * bReg;
                if (kZmmRegs - cReg - bReg < aRegMin) {
                    // Slot stays nullptr (zero-initialized by resize).
                    // The dispatcher never reaches it for any DE-blessed
                    // shape that bumped MR.
                    continue;
                }

                auto gen = std::make_unique<GEMMcodeGenerator::jitGEMMBF16<
                    utils::kernelInstrType::avx512_zmm_32_reg>>(
                    utils::JIT_KERNEL_SIZE);
                err = gen->generateKernel(params);
                if (err != dlp::jit::jitGeneratorError::success) {
                    goto cleanup;
                }
                // Must call ready() to readjust jump/branch targets with
                // respect to any new buffer created as part of AutoGrow mode
                // in Xbyak.
                gen->ready();
                kernelCodeBlocks[mr * numNRVariants + nr] =
                    const_cast<void*>(static_cast<const void*>(gen->getCode()));
                codeGenerators.push_back(std::move(gen));

                DLP_ENABLE_JIT_DUMP_AND_MONITOR(
                    kernelCodeBlocks[mr * numNRVariants + nr],
                    utils::JIT_KERNEL_SIZE, "bf16_jit_kernel", params.MR,
                    params.NR, params.useMask, mr * numNRVariants + nr);
            }
        }
    }

    return dlp::jit::jitGeneratorError::success;
cleanup:
    codeGenerators.clear();
    for (auto& block : kernelCodeBlocks) {
        block = nullptr;
    }
    return err;
}

dlp::kernels::kernelError
jitAmdZenBF16::executeKernel(dlp::kernels::kernelParams* _params)
{
    // Execute f32 JIT kernel if f32JitGenerator is available, which means bf16
    // is not supported and we want to fall back to f32
    if (f32JitGenerator != nullptr) {
        return f32JitGenerator->executeKernel(_params);
    }

    // This code-section is taken if the underlying architecture supports
    // AVX512-BF16.
    if (MR == 1) {
        auto params = static_cast<dlp::kernels::gemvM1Params*>(_params);
        // Setting the remaining values of gemvM1Params(that are not set as
        // part of it's parameterized constructor)

        params->n_iter = params->n / NR;
        params->n_left = params->n % NR;
        params->k_iter = params->k / KC;
        params->k_left = params->k % KC;

        params->n_left_16   = (params->n_left / 16) * 16;
        params->n_left_lt16 = params->n_left % 16;

        params->rsB      = (params->n_left / 16) * 16;
        params->rsB      = (params->rsB != 0) ? params->rsB : 16;
        params->is_k_odd = params->k_left & 1;

        params->k_iter_sub_iter = (KC / 2) / K_UNROLL;
        params->k_iter_sub_left = (KC / 2) % K_UNROLL;
        params->k_left_sub_iter = ((params->k_left) / 2) / K_UNROLL;
        params->k_left_sub_left = ((params->k_left) / 2) % K_UNROLL;

        int partial_elements =
            params->n_left % numElemsPerReg; // partial elements < simdWidth

        params->nmask_avx512 = 0xFFFF >> (numElemsPerReg - partial_elements);

        // Deploy the associated kernel
        md_t kernel_idx = params->n_left;

        utils::jit_gemv_m1_kernel kernel =
            reinterpret_cast<utils::jit_gemv_m1_kernel>(
                kernelCodeBlocks[kernel_idx]);
        kernel(params);

    } else if (NR == 1) {
        auto params = static_cast<dlp::kernels::gemvN1Params*>(_params);

        // Setting the remaining values of gemvN1Params(that are not set as
        // part of it's parameterized constructor) NOTE : The associated NR
        // value for the generated kernels is 16.
        // The mdulo operations are added when calcuating the masks,
        // to make sure we use the same code for generic NR values in
        // future.
        params->m_iter = params->m / MR;
        params->m_left = params->m % MR;
        params->k_iter = params->k / numElemsPerReg;
        params->k_left = params->k % numElemsPerReg;

        params->mmask_avx512 = 0xFFFF >> (MR - (params->m_left));
        // When k_left == 0, shifting by numElemsPerReg (32) is undefined
        // behavior since the shift count equals the bit width of uint32_t.
        params->kmask_bf16_avx512 =
            (params->k_left == 0)
                ? 0xFFFFFFFFu
                : 0xFFFFFFFFu >> (numElemsPerReg - params->k_left);

        int is_m_loop = ((params->m) >= MR);
        // when rsC = 1, we logically have a col-major layout , where beta
        // scaling would not need striding for C
        int is_col_stored = ((params->rsC) == 1);

        // Determine the kernel index based on m_left, storage format of C
        // and whether mloop is needed
        int kernel_idx = 4 * params->m_left + 2 * is_col_stored + is_m_loop;

        // Deploy the associated kernel
        utils::jit_gemv_n1_kernel kernel =
            reinterpret_cast<utils::jit_gemv_n1_kernel>(
                kernelCodeBlocks[kernel_idx]);
        kernel(params);
    } else {
        auto params = static_cast<dlp::kernels::gemmParams*>(_params);

        int processBlockSize = NR;
        // Since JR loop is in framework, the 'n' dimension passed to this
        // function is always <= NR.
        int mFullPieces    = params->m / MR;
        int mPartialPieces = params->m % MR;

        // For now, we will use kIter and kLeft to represent the
        // iteration counts in the k direction, based on packing mandate.

        // Ideally, these should be used to represent the unroll factor as well
        // in which case, we will have to update the calculation accordingly.
        int bf16PackingFactor = 2;
        int kBlocks           = params->k / bf16PackingFactor;
        int kPadSize          = params->k % bf16PackingFactor;

        // For B prefetch.
        if (kBlocks > PREFETCH_C_DIST) {
            params->kIterBP = kBlocks - PREFETCH_C_DIST;
            params->kIterAP = PREFETCH_C_DIST;
        } else {
            params->kIterBP = kBlocks;
            params->kIterAP = 0;
        }
        params->kLeft = kPadSize;

        int16_t* aPtr = static_cast<int16_t*>(params->a);
        int16_t* bPtr = static_cast<int16_t*>(params->b);
        float*   cPtr = static_cast<float*>(params->c);
        // Initialize pointers to A and B
        float* c_jr = cPtr;
        float* c_ir = cPtr;

        md_t n              = params->n;
        md_t m              = params->m;
        md_t k              = params->k;
        int  nBlockSize     = (n < processBlockSize) ? n : processBlockSize;
        int  nFullpieces    = nBlockSize / numElemsPerReg;
        int  nRemainder     = nBlockSize % numElemsPerReg;
        int  rsB            = params->rsB;
        int  og_post_op_c_i = (params->kernelOpsAttr).post_op_c_i;

        // Process complete registers first (if any)
        if (nFullpieces > 0) {
            // Updates to the strides based on the packing mandate
            params->rsB = (rsB / (NR / numElemsPerReg)) * nFullpieces;

            int elementsToProcess = nFullpieces * numElemsPerReg;

            params->a = aPtr;
            params->c = c_jr;
            params->n = elementsToProcess;

            int kernel_n_idx = nFullpieces;
            if (params->m >= MR) {
                params->mIter            = mFullPieces;
                utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                    kernelCodeBlocks[kernel_n_idx]);

                DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(kernel));
                kernel(params);
                (params->a) =
                    (int16_t*)(params->a) + MR * mFullPieces * params->psA;
                (params->c) =
                    (float*)(params->c) + MR * mFullPieces * params->rsC;
            }

            if (mPartialPieces) {
                int               m_idx  = mPartialPieces;
                utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                    kernelCodeBlocks[m_idx * numNRVariants + kernel_n_idx]);

                DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(kernel));
                kernel(params);
            }

            params->b = bPtr + (elementsToProcess * (params->k + kPadSize));
            c_jr      = c_jr + elementsToProcess;
            (params->kernelOpsAttr).post_op_c_i = og_post_op_c_i;
            (params->kernelOpsAttr).post_op_c_j += elementsToProcess;
            n -= elementsToProcess;
        }
        if (nRemainder > 0) {
            // Updates to the strides based on the packing mandate
            params->rsB = (rsB / (NR / numElemsPerReg));
            params->a   = aPtr;
            params->c   = c_jr;
            params->n   = nRemainder;

            // Unlike F32 JIT, for now, we are producing only lt16 as the
            // mask based fringe kernel for BF16 JIT.
            params->maskF32[0] = 0xFFFF >> (numElemsPerReg - nRemainder);

            if (params->m >= MR) {
                params->mIter = mFullPieces;
                utils::jit_kernel kernel =
                    reinterpret_cast<utils::jit_kernel>(kernelCodeBlocks[0]);

                DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(kernel));
                kernel(params);
                (params->a) =
                    (int16_t*)(params->a) + MR * mFullPieces * params->psA;
                (params->c) =
                    (float*)(params->c) + MR * mFullPieces * params->rsC;
            }

            if (mPartialPieces) {
                int               m_idx  = mPartialPieces;
                utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                    kernelCodeBlocks[m_idx * numNRVariants]);

                DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(kernel));
                kernel(params);
            }
        }
    }

    return dlp::kernels::kernelError::success;
}

jitAmdZenU8S8::jitAmdZenU8S8()
    : mKernelDatatypes({ dlp::kernel_frame::kernelDatatype::u8s8s32os32,
                         dlp::kernel_frame::kernelDatatype::u8s8s32of32,
                         dlp::kernel_frame::kernelDatatype::u8s8s32of16,
                         dlp::kernel_frame::kernelDatatype::u8s8s32obf16,
                         dlp::kernel_frame::kernelDatatype::u8s8s32ou8,
                         dlp::kernel_frame::kernelDatatype::u8s8s32os8 })
    , mIsaFeaturesRequired({ dlp::cpu_utils::isaFeature::avx512vnni })
    , kType(utils::kernelInstrType::none)
    , numElemsPerReg(1) // Initializing with 1 to avoid div by zero
    , numBytesPerElem(1)
    , requiresBPacking(true) // B matrix packing is mandatory for VNNI
    , supportsPostOps(true)  // Integer kernels support post-operations
    , vnniGroupSize(4)       // VNNI groups 4 int8 elements
{
    // Set required ISA features for u8s8s32 kernels
    mIsaFeaturesRequired.push_back(dlp::cpu_utils::isaFeature::avx512f);
    mIsaFeaturesRequired.push_back(dlp::cpu_utils::isaFeature::avx512bw);
}

jitAmdZenU8S8::~jitAmdZenU8S8()
{
    codeGenerators.clear();
    for (auto& block : kernelCodeBlocks) {
        block = nullptr;
    }
}

int
jitAmdZenU8S8::getProcessBlockSize() const
{
    switch (kType) {
        case utils::kernelInstrType::avx512_zmm_32_reg:
            return NR; // Process full NR

        default:
            return 0; // Invalid/unsupported kernel type
    }
}

void
jitAmdZenU8S8::setGeneratorKernelMetaInfo(
    dlp::kernel_frame::kernelInstrPreference kInstPref)
{
    kType = utils::kernelInstrType::none;
    switch (kInstPref) {
        case dlp::kernel_frame::kernelInstrPreference::avx512_zmm_favour: {
            kType = utils::kernelInstrType::avx512_zmm_32_reg;
            // For u8s8s32 VNNI: think in terms of int32 accumulator elements
            // ZMM holds 16 int32 elements, each consuming 4 int8 inputs (VNNI
            // group)
            numElemsPerReg =
                traits::ArchitectureTraits<
                    utils::kernelInstrType::avx512_zmm_32_reg>::regBytes
                / sizeof(int32_t);
            break;
        }
        default: {
            break;
        }
    }
}

/* Function to generate all kernels */
dlp::jit::jitGeneratorError
jitAmdZenU8S8::generateAllKernels(const dlp::jit::jitGeneratorContext& jI)
{
    dlp::jit::jitGeneratorError err = dlp::jit::jitGeneratorError::error;

    MR          = (jI.kI).mr;
    NR          = (jI.kI).nr;
    KC          = (jI.kI).kc;
    K_UNROLL    = (jI.kI).k_unroll;
    c_downscale = (jI.kI).c_downscale;

    // Convert kernelInstrPreference to kernelType
    setGeneratorKernelMetaInfo(jI.kI.kInstPref);

    int8_t processBlockSize = getProcessBlockSize();

    // Verify AVX512_VNNI support is available
    if (kType == utils::kernelInstrType::none) {
        err = dlp::jit::jitGeneratorError::notSupported;
        goto cleanup;
    }

    if (MR == 1) {
        // Handle GEMV M = 1 case
        numKernelVariants = NR;
        kernelCodeBlocks.resize(numKernelVariants);
        AOCL_DLP_MEMORY_TAG mtag_b = (jI.kI).mtag_b;

        utils::gemvM1GeneratorParams params(
            0, NR, 0, KC, K_UNROLL, mtag_b, true, true, true, true,
            dlp::kernel_frame::storageFormat::rowMajor,
            (jI.kI).alphaScalingType, (jI.kI).betaScalingType, kType);

        params.c_downscale = c_downscale;

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            // Copy the kernelOps from the kernelInfo to params
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        params.NR         = NR;
        params.KC         = KC;
        params.N_LEFT     = 0;
        params.K_SUB_ITER = K_UNROLL;
        params.nloop      = true;
        params.kloop      = true;
        params.nfringe    = false;
        params.kfringe    = true;

        for (iter_t i = 0; i < NR; i++) {
            params.N_LEFT  = i;
            params.nfringe = (i != 0);

            auto gen = std::make_unique<amdzen::gen::jitU8S8VNNI_GEMVM1<
                utils::kernelInstrType::avx512_zmm_32_reg>>(
                utils::JIT_KERNEL_SIZE);
            err = gen->generateKernel(params);
            if (err != dlp::jit::jitGeneratorError::success) {
                goto cleanup;
            }
            // Must call ready() to readjust jump/branch targets with
            // respect to any new buffer created as part of AutoGrow mode
            // in Xbyak.
            gen->ready();
            kernelCodeBlocks[i] =
                const_cast<void*>(static_cast<const void*>(gen->getCode()));
            codeGenerators.push_back(std::move(gen));
            int n_left_suf = (i != 0) ? i : params.NR;
            DLP_ENABLE_JIT_DUMP_AND_MONITOR(
                kernelCodeBlocks[i], utils::JIT_KERNEL_SIZE,
                "jit_gemv_m1_kernel", 1, n_left_suf, false, i);
        }

    } else if (NR == 1) {
        // TODO: We could extend the no-loop decomposition to handle sizes until
        //       MR + MR - 1. Since, we would not require any looping for such
        //       sizes.

        numKernelVariants = MR * 4; // Each MR has a kernel for vector and
                                    // element-wise loads/stores for C

        kernelCodeBlocks.resize(numKernelVariants);

        // Initializing with default values.
        utils::gemvN1GeneratorParams params(
            0, 0, c_downscale, false, false, false, false,
            dlp::kernel_frame::storageFormat::rowMajor,
            (jI.kI).alphaScalingType, (jI.kI).betaScalingType, kType);

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            // Copy the kernelOps from the kernelInfo to params
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        params.MR      = MR;
        params.mloop   = true;
        params.kloop   = true;
        params.mfringe = false;
        params.kfringe = true;

        for (iter_t m_left = 0; m_left < MR; m_left++) {
            params.M_LEFT  = m_left;
            params.mfringe = (m_left != 0); // The first two kernels that we
                                            // generate are main kernels
            for (iter_t j = 0; j < 4; j++) {
                // We generate 2 kernels for each M_LEFT, and index them as
                // follows:
                // 0: row-stored, without mloop
                // 1: row-stored, with mloop
                // 2: col-stored, without mloop
                // 3: col-stored, with mloop

                params.mloop = (j == 1) || (j == 3);
                params.yFormat =
                    ((j / 2) == 0) ? dlp::kernel_frame::storageFormat::rowMajor
                                   : dlp::kernel_frame::storageFormat::colMajor;

                std::unique_ptr<Xbyak::CodeGenerator> gen;
                switch (kType) {
                    case utils::kernelInstrType::avx512_zmm_32_reg: {
                        auto g =
                            std::make_unique<amdzen::gen::jitU8S8VNNI_GEMVN1<
                                utils::kernelInstrType::avx512_zmm_32_reg>>(
                                utils::JIT_KERNEL_SIZE);
                        err = g->generateKernel(params);
                        gen = std::move(g);
                        break;
                    }
                    default: {
                        err = dlp::jit::jitGeneratorError::error;
                        break;
                    }
                }
                if (err != dlp::jit::jitGeneratorError::success) {
                    goto cleanup;
                }
                // Must call ready() to readjust jump/branch targets with
                // respect to any new buffer created as part of AutoGrow mode
                // in Xbyak.
                gen->ready();
                kernelCodeBlocks[m_left * 4 + j] =
                    const_cast<void*>(static_cast<const void*>(gen->getCode()));
                codeGenerators.push_back(std::move(gen));

                int m_left_suf = (m_left != 0) ? m_left : params.MR;
                // The file naming is as such : jit_gemv_n1_kernels_MR_idx.
                // The idx represents what configuration was used to generate
                // the kernel.
                DLP_ENABLE_JIT_DUMP_AND_MONITOR(
                    kernelCodeBlocks[m_left * 4 + j], utils::JIT_KERNEL_SIZE,
                    "jit_gemv_n1_kernel", m_left_suf, j, false, m_left * 4 + j);
            }
        }
    } else if (MR > 1 && NR > 1) {

        // Initializing with default values.
        utils::generatorParams params(0, 0, K_UNROLL, 0, c_downscale, 0, false,
                                      false, false, (jI.kI).alphaScalingType,
                                      (jI.kI).betaScalingType, kType);

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            // Copy the kernelOps from the kernelInfo to params
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        // Generate GEMM kernel variants
        numMRVariants = MR;                                      // MR variants
        numNRVariants = (processBlockSize / numElemsPerReg) + 1; // NR variants
        numKernelVariants = numMRVariants * numNRVariants;

        kernelCodeBlocks.resize(numKernelVariants);

        for (iter_t mr_var = 0; mr_var < numMRVariants; mr_var++) {
            for (iter_t nr_var = 0; nr_var < numNRVariants; nr_var++) {
                int variant_idx = mr_var * numNRVariants + nr_var;

                // Set parameters for this specific kernel variant
                params.c_downscale = c_downscale;
                params.MR          = mr_var == 0 ? MR : mr_var;
                params.mLoop       = mr_var == 0;
                params.NR          = nr_var * numElemsPerReg;
                params.useMask     = (nr_var == 0);
                params.numMaskRegs = (params.useMask) ? 1 : 0;

                std::unique_ptr<Xbyak::CodeGenerator> gen;
                switch (kType) {
                    case utils::kernelInstrType::avx512_zmm_32_reg: {
                        auto g = std::make_unique<amdzen::gen::jitU8S8VNNI_GEMM<
                            utils::kernelInstrType::avx512_zmm_32_reg>>(
                            utils::JIT_KERNEL_SIZE);
                        err = g->generateKernel(params);
                        gen = std::move(g);
                        break;
                    }
                    default:
                        err = dlp::jit::jitGeneratorError::error;
                        break;
                }
                if (err != dlp::jit::jitGeneratorError::success) {
                    goto cleanup;
                }
                // Must call ready() to readjust jump/branch targets with
                // respect to any new buffer created as part of AutoGrow mode
                // in Xbyak.
                gen->ready();
                kernelCodeBlocks[variant_idx] =
                    const_cast<void*>(static_cast<const void*>(gen->getCode()));
                codeGenerators.push_back(std::move(gen));

                // Enhanced file naming with fringe info and index
                bool isFringe =
                    params.useMask; // nr_var==0 kernels use masks (fringe)
                DLP_ENABLE_JIT_DUMP_AND_MONITOR(
                    kernelCodeBlocks[variant_idx], utils::JIT_KERNEL_SIZE,
                    "jit_u8s8s32_kernel", params.MR, params.NR, isFringe,
                    variant_idx);
            }
        }
    }

    return dlp::jit::jitGeneratorError::success;

cleanup:
    codeGenerators.clear();
    for (auto& block : kernelCodeBlocks) {
        block = nullptr;
    }
    return err;
}

dlp::kernels::kernelError
jitAmdZenU8S8::executeKernel(dlp::kernels::kernelParams* _params)
{
    auto params = static_cast<dlp::kernels::gemvM1Params*>(_params);

    if (MR == 1) {
        numElemsPerReg = 16;

        int remainder     = params->k % 4;
        params->kLeftmask = remainder ? (0xF >> (4 - remainder)) : 0xF;

        params->n_iter = params->n / NR;
        params->n_left = params->n % NR;

        params->k_iter = params->k / KC;
        params->k_left = params->k % KC; // Always multiple of 4

        params->psB = (params->k_left + 3) & ~3;
        // Within KC block, process in chunks of 16 bytes (4 VNNI groups)
        params->k_iter_sub_iter = KC / 16;

        // Remaining VNNI groups in KC (always 0, 1, 2, or 3 groups)
        params->k_iter_sub_left = (KC % 16) / 4;

        // Within k_left, same decomposition
        params->k_left_sub_iter = params->k_left / 16;
        params->k_left_sub_left = (params->k_left % 16) / 4;

        if (params->k % 4 != 0)
            params->k_left_sub_left++;

        int partial_elements = params->n_left % numElemsPerReg;

        if (params->n_left > 0) {
            params->nmask_avx512 =
                0xFFFF >> (numElemsPerReg - partial_elements);
        }

        md_t kernel_idx = params->n_left;

        utils::jit_gemv_m1_kernel kernel =
            reinterpret_cast<utils::jit_gemv_m1_kernel>(
                kernelCodeBlocks[kernel_idx]);

        md_t og_post_op_c_j = (params->kernelOpsAttr).post_op_c_j;
        kernel(params);
        // post_op_c_j is updated inside the JIT kernel's N-loop. Reset it
        // to the saved value and increment by the total n processed.
        og_post_op_c_j += params->n;
        (params->kernelOpsAttr).post_op_c_j = og_post_op_c_j;

        return dlp::kernels::kernelError::success;

    } else if (NR == 1) {
        numElemsPerReg = 64;

        auto params    = static_cast<dlp::kernels::gemvN1Params*>(_params);
        params->m_iter = params->m / MR;
        params->m_left = params->m % MR;
        params->k_iter = params->k / numElemsPerReg;
        params->k_left = params->k % numElemsPerReg;
        // md_t og_post_ops_c_i = (params->kernelOpsAttr).post_op_c_i;

        // When k_left == 0, shifting by numElemsPerReg (64) is undefined
        // behavior since the shift count equals the bit width of uint64_t.
        params->kmask_i8_avx512 =
            (params->k_left == 0)
                ? 0xFFFFFFFFFFFFFFFFULL
                : 0xFFFFFFFFFFFFFFFFULL >> (numElemsPerReg - params->k_left);
        params->mmask_avx512 = 0xFFFF >> (MR - params->m_left);

        bool m_loop        = params->m_iter >= 1;
        bool is_col_stored = ((params->rsC) == 1);

        int kernel_idx = 4 * params->m_left + 2 * is_col_stored + m_loop;

        utils::jit_gemv_n1_kernel kernel =
            reinterpret_cast<utils::jit_gemv_n1_kernel>(
                kernelCodeBlocks[kernel_idx]);
        kernel(params);
    } else {
        auto params = static_cast<dlp::kernels::gemmParams*>(_params);

        int processBlockSize = getProcessBlockSize();
        // Since JR loop is in framework, the 'n' dimension passed to this
        // function is always <= NR.
        md_t mFullPieces    = params->m / MR;
        md_t mPartialPieces = params->m % MR;

        // The JIT body emits K_UNROLL VNNI groups per outer-loop iteration,
        // so the trip count must divide out both factors. Mirrors the
        // FP32-VNNI template earlier in this file; K_UNROLL=1 reduces to
        // the original formula.
        //
        // Generalised K-tail decomposition: kLeft (residual K count) splits
        // into kLeftIter (full VNNI groups in the tail, 0..K_UNROLL-1) and
        // kLeftRem (K-element residual, 0..vnniGroupSize-1). The two-stage
        // tail in u8s8_gemm_generator.cc consumes kLeftIter via full-group
        // iterations and kLeftRem via the masked load. The kLeftmask
        // formula moves to (1 << kLeftRem) - 1: identical value semantics
        // on any kLeftIter == 0 shape (which includes every
        // K%(K_UNROLL*vnniGroupSize)==0 shape) and strictly correct on the
        // K_UNROLL=2 K%8 != 0 shapes that the generalised tail unblocks.
        params->kIterBP   = params->k / (K_UNROLL * vnniGroupSize);
        params->kLeft     = params->k % (K_UNROLL * vnniGroupSize);
        params->kLeftIter = params->kLeft / vnniGroupSize;
        params->kLeftRem  = params->kLeft % vnniGroupSize;

        // Handle VNNI remainder case using AVX-512 masked memory load
        // (mask is on the kLeftRem residual; identical to the previous
        //  formula on any kLeftIter == 0 shape).
        params->kLeftmask = (1 << params->kLeftRem) - 1;

        uint8_t* aPtr           = static_cast<uint8_t*>(params->a);
        int8_t*  bPtr           = static_cast<int8_t*>(params->b);
        int32_t* cPtr           = static_cast<int32_t*>(params->c);
        int32_t* c_jr           = cPtr;
        int32_t* c_ir           = cPtr;
        md_t     rsB            = params->rsB;
        md_t     og_post_op_c_i = (params->kernelOpsAttr).post_op_c_i;

        md_t n = params->n;
        md_t m = params->m;
        md_t k = params->k;

        md_t nBlockSize  = (n >= processBlockSize) ? processBlockSize : n;
        md_t nFullpieces = nBlockSize / numElemsPerReg;
        md_t nRemainder  = nBlockSize % numElemsPerReg;
        params->psA      = MR * params->psA;
        // Process complete registers first (if any)
        if (nFullpieces > 0) {
            md_t elementsToProcess = nFullpieces * numElemsPerReg;

            params->a = aPtr;
            params->c = c_jr;
            params->n = elementsToProcess;

            params->rsB = (nFullpieces * rsB) / vnniGroupSize;

            // Match the kernel generation pattern: nr_var corresponds to
            // number of registers used nr_var=0: mask kernel,
            // nr_var=1,2,3,4: 1,2,3,4 registers respectively
            md_t kernel_n_idx = nFullpieces;

            if (params->m >= MR) {
                params->mIter            = mFullPieces;
                int               m_idx  = 0;
                utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                    kernelCodeBlocks[m_idx * numNRVariants + kernel_n_idx]);

                DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(kernel));
                kernel(params);
                (params->a) = (uint8_t*)(params->a) + mFullPieces * params->psA;
                (params->c) =
                    (int32_t*)(params->c) + mFullPieces * MR * params->rsC;
            }

            if (mPartialPieces) {
                int               m_idx  = mPartialPieces;
                utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                    kernelCodeBlocks[m_idx * numNRVariants + kernel_n_idx]);

                DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(kernel));
                kernel(params);
            }

            md_t k_updated = ((params->k + vnniGroupSize - 1) / vnniGroupSize)
                             * vnniGroupSize;
            params->b = (int8_t*)(params->b) + elementsToProcess * k_updated;

            c_jr = (int32_t*)(c_jr) + elementsToProcess;
            (params->kernelOpsAttr).post_op_c_i = og_post_op_c_i;
            (params->kernelOpsAttr).post_op_c_j += elementsToProcess;

            n -= elementsToProcess;
        }

        // Process remainder with mask (if any)
        if (nRemainder) {
            params->a = aPtr;
            params->c = c_jr;
            params->n = nRemainder;

            params->rsB = numElemsPerReg * vnniGroupSize;

            if (kType == utils::kernelInstrType::avx512_zmm_32_reg) {
                // For AVX-512 ZMM: 16-bit mask for int32 elements (16 int32
                // per ZMM)
                params->maskS32 = 0xFFFF >> (numElemsPerReg - nRemainder);
                // Post-ops module uses maskF32 for its mask registers, so we
                // need to set it as well for u8s8 kernels with post-ops
                params->maskF32[0] = params->maskS32;
            }

            // INLINE: Kernel execution with mask
            // Use nr_var=0 kernel which was generated with useMask=true
            int kernel_n_idx =
                0; // Always use mask kernel (nr_var=0) for nRemainder
            if (params->m >= MR) {
                params->mIter            = mFullPieces;
                int               m_idx  = 0;
                utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                    kernelCodeBlocks[m_idx * numNRVariants + kernel_n_idx]);

                DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(kernel));
                kernel(params);
                (params->a) = (uint8_t*)(params->a) + mFullPieces * params->psA;
                (params->c) =
                    (int32_t*)(params->c) + mFullPieces * MR * params->rsC;
            }
            if (mPartialPieces) {
                int               m_idx  = mPartialPieces;
                utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                    kernelCodeBlocks[m_idx * numNRVariants + kernel_n_idx]);

                DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(kernel));
                kernel(params);
            }
        }
    }

    return dlp::kernels::kernelError::success;
}

jitAmdZenS8::jitAmdZenS8()
    : mKernelDatatypes({ dlp::kernel_frame::kernelDatatype::s8s8s32ou8,
                         dlp::kernel_frame::kernelDatatype::s8s8s32os8,
                         dlp::kernel_frame::kernelDatatype::s8s8s32obf16,
                         dlp::kernel_frame::kernelDatatype::s8s8s32of32,
                         dlp::kernel_frame::kernelDatatype::s8s8s32of16,
                         dlp::kernel_frame::kernelDatatype::s8s8s32os32 })
    , mIsaFeaturesRequired({ dlp::cpu_utils::isaFeature::avx512vnni })
    , kType(utils::kernelInstrType::none)
    , numElemsPerReg(1) // Initializing with 1 to avoid div by zero
{
    mIsaFeaturesRequired.push_back(dlp::cpu_utils::isaFeature::avx512f);
    mIsaFeaturesRequired.push_back(dlp::cpu_utils::isaFeature::avx512bw);
}

jitAmdZenS8::~jitAmdZenS8()
{
    codeGenerators.clear();
    for (auto& block : kernelCodeBlocks) {
        block = nullptr;
    }
}

int
jitAmdZenS8::getProcessBlockSize() const
{
    switch (kType) {
        case utils::kernelInstrType::avx512_zmm_32_reg:
            return NR; // Process full NR (64) elements at once

        // Not Supported!
        case utils::kernelInstrType::avx512_ymm_32_reg:
        case utils::kernelInstrType::avx2_ymm_16_reg:
        default:
            return 0; // Invalid/unsupported kernel type
    }
}

void
jitAmdZenS8::setGeneratorKernelMetaInfo(
    dlp::kernel_frame::kernelInstrPreference kInstPref)
{
    kType = utils::kernelInstrType::none;
    switch (kInstPref) {
        case dlp::kernel_frame::kernelInstrPreference::avx512_zmm_favour: {
            kType = utils::kernelInstrType::avx512_zmm_32_reg;
            // Acquiring the VNNI width of the kernel type
            numElemsPerReg =
                traits::ArchitectureTraits<
                    utils::kernelInstrType::avx512_zmm_32_reg>::regBytes
                / VNNI_CONST; // Dividing by VNNI_CONST (4) since each VNNI
                              // operation processes 4 int8_t elements.
            break;
        }

        // Not Supported!
        case dlp::kernel_frame::kernelInstrPreference::avx512_ymm_favour:
        case dlp::kernel_frame::kernelInstrPreference::avx512_xmm_favour:
        case dlp::kernel_frame::kernelInstrPreference::avx2_ymm_favour:
        case dlp::kernel_frame::kernelInstrPreference::avx2_xmm_favour:
        default:
            break;
    }
}

dlp::jit::jitGeneratorError
jitAmdZenS8::generateAllKernels(const dlp::jit::jitGeneratorContext& jI)
{
    dlp::jit::jitGeneratorError err = dlp::jit::jitGeneratorError::error;

    MR              = (jI.kI).mr;
    NR              = (jI.kI).nr;
    KC              = (jI.kI).kc;
    K_UNROLL        = (jI.kI).k_unroll;
    PREFETCH_C_DIST = (jI.kI).prefetch_c_dist;
    c_downscale     = (jI.kI).c_downscale;

    const dlp::kernel_frame::kernelDatatype kdt =
        dlp::kernel_frame::kernelDatatype::s8s8s32os32;

    // Convert kernelInstrPreference to kernelType
    setGeneratorKernelMetaInfo(jI.kI.kInstPref);

    int processBlockSize = getProcessBlockSize();

    if (MR == 1) { // S8 GEMV M=1 kernel generation
        AOCL_DLP_MEMORY_TAG mtag_b = (jI.kI).mtag_b;
        numKernelVariants          = NR;
        kernelCodeBlocks.resize(numKernelVariants);

        utils::gemvM1GeneratorParams params(
            0, NR, 0, KC, K_UNROLL, mtag_b, true, true, true, true,
            dlp::kernel_frame::storageFormat::rowMajor,
            (jI.kI).alphaScalingType, (jI.kI).betaScalingType, kType);

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        params.NR          = NR;
        params.N_LEFT      = 0;
        params.N_LEFT_16   = 0;
        params.N_LEFT_LT16 = 0;
        params.KC          = KC;
        params.K_SUB_ITER  = K_UNROLL;
        params.nloop       = true;
        params.kloop       = true;
        params.nfringe     = false;
        params.kfringe     = true;
        params.c_downscale = c_downscale;

        for (iter_t i = 0; i < NR; ++i) {
            params.N_LEFT      = i;
            params.N_LEFT_16   = (i / 16) * 16;
            params.N_LEFT_LT16 = i % 16;
            params.nfringe     = (i != 0);

            std::unique_ptr<Xbyak::CodeGenerator> gen;
            switch (kType) {
                case utils::kernelInstrType::avx512_zmm_32_reg: {
                    auto g = std::make_unique<amdzen::gen::jitGEMVS8M1<
                        utils::kernelInstrType::avx512_zmm_32_reg>>(
                        utils::JIT_KERNEL_SIZE);
                    err = g->generateKernel(params);
                    gen = std::move(g);
                    break;
                }
                case utils::kernelInstrType::avx512_ymm_32_reg:
                case utils::kernelInstrType::avx2_ymm_16_reg:
                default:
                    err = dlp::jit::jitGeneratorError::error;
                    break;
            }
            if (err != dlp::jit::jitGeneratorError::success) {
                goto cleanup;
            }
            // Must call ready() to readjust jump/branch targets with
            // respect to any new buffer created as part of AutoGrow mode
            // in Xbyak.
            gen->ready();
            kernelCodeBlocks[i] =
                const_cast<void*>(static_cast<const void*>(gen->getCode()));
            codeGenerators.push_back(std::move(gen));

            int n_left_suf = (i != 0) ? i : params.NR;
            DLP_ENABLE_JIT_DUMP_AND_MONITOR(
                kernelCodeBlocks[i], utils::JIT_KERNEL_SIZE,
                "jit_s8_gemv_m1_kernel", 1, n_left_suf, false, i);
        }
    } else if (NR == 1) { // S8 GEMV N=1 kernel generation
        // 4 kernels are generated for each m (< MR):
        // 0: row-stored, without mloop
        // 1: row-stored, with mloop
        // 2: col-stored, without mloop
        // 3: col-stored, with mloop
        // Hence, total variants = MR * 4
        numKernelVariants = MR * 4;

        kernelCodeBlocks.resize(numKernelVariants);

        // Initializing with default values.
        utils::gemvN1GeneratorParams params(
            0, 0, c_downscale, false, false, false, false,
            dlp::kernel_frame::storageFormat::rowMajor,
            (jI.kI).alphaScalingType, (jI.kI).betaScalingType, kType);

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            // Copy the kernelOps from the kernelInfo to params
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        params.MR      = MR;
        params.mloop   = true;
        params.kloop   = true;
        params.mfringe = false;
        params.kfringe = true;

        for (iter_t m_left = 0; m_left < MR; ++m_left) {
            params.M_LEFT  = m_left;
            params.mfringe = (m_left != 0); // The first four kernels that we
                                            // generate are the primary kernels.
            for (iter_t j = 0; j < 4; ++j) {
                // We generate 4 kernels for each M_LEFT, and index them as
                // follows:
                // 0: row-stored, without mloop
                // 1: row-stored, with mloop
                // 2: col-stored, without mloop
                // 3: col-stored, with mloop
                params.mloop = ((j == 1) || (j == 3));
                params.yFormat =
                    ((j / 2) == 0) ? dlp::kernel_frame::storageFormat::rowMajor
                                   : dlp::kernel_frame::storageFormat::colMajor;

                std::unique_ptr<Xbyak::CodeGenerator> gen;
                switch (kType) {
                    case utils::kernelInstrType::avx512_zmm_32_reg: {
                        auto g = std::make_unique<amdzen::gen::jitGEMVS8N1<
                            utils::kernelInstrType::avx512_zmm_32_reg>>(
                            utils::JIT_KERNEL_SIZE);
                        err = g->generateKernel(params);
                        gen = std::move(g);
                        break;
                    }
                    case utils::kernelInstrType::avx512_ymm_32_reg:
                    case utils::kernelInstrType::avx2_ymm_16_reg:
                    default:
                        err = dlp::jit::jitGeneratorError::error;
                        break;
                }
                if (err != dlp::jit::jitGeneratorError::success) {
                    goto cleanup;
                }
                // Must call ready() to readjust jump/branch targets with
                // respect to any new buffer created as part of AutoGrow mode
                // in Xbyak.
                gen->ready();
                kernelCodeBlocks[m_left * 4 + j] =
                    const_cast<void*>(static_cast<const void*>(gen->getCode()));
                codeGenerators.push_back(std::move(gen));

                int m_left_suf = (m_left != 0) ? m_left : params.MR;
                // The file naming is as such : id_jit_s8_gemv_n1_kernel_MR_idx.
                // The idx represents what configuration was used to generate
                // the kernel.
                DLP_ENABLE_JIT_DUMP_AND_MONITOR(
                    kernelCodeBlocks[m_left * 4 + j], utils::JIT_KERNEL_SIZE,
                    "jit_s8_gemv_n1_kernel", m_left_suf, j, false,
                    m_left * 4 + j);
            }
        }
    } else { // S8 GEMM kernel generation
        numNRVariants = ((processBlockSize / numElemsPerReg)) + 1;

        numMRVariants     = MR;
        numKernelVariants = numMRVariants * numNRVariants;

        kernelCodeBlocks.resize(numKernelVariants);

        // Initializing with default values.
        utils::generatorParams params(0, 0, (jI.kI).k_unroll, PREFETCH_C_DIST,
                                      c_downscale, 1, false, false, false,
                                      (jI.kI).alphaScalingType,
                                      (jI.kI).betaScalingType, kType);

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            // Copy the kernelOps from the kernelInfo to params
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        // Generate all kernels for the given MR and NR
        for (iter_t mr = 0; mr < numMRVariants; mr++) {
            for (iter_t nr = 0; nr < numNRVariants; nr++) {
                params.MR    = mr == 0 ? MR : mr;
                params.mLoop = mr == 0;

                params.NR      = nr * numElemsPerReg;
                params.useMask = (nr == 0);

                params.numMaskRegs = (params.useMask) ? 1 : 0;

                std::unique_ptr<Xbyak::CodeGenerator> gen;
                switch (kType) {
                    case utils::kernelInstrType::avx512_zmm_32_reg: {
                        auto g = std::make_unique<GEMMcodeGenerator::jitGEMMS8<
                            utils::kernelInstrType::avx512_zmm_32_reg>>(
                            utils::JIT_KERNEL_SIZE);
                        err = g->generateKernel(params);
                        gen = std::move(g);
                        break;
                    }
                    case utils::kernelInstrType::avx512_ymm_32_reg:
                    case utils::kernelInstrType::avx2_ymm_16_reg:
                    default:
                        err = dlp::jit::jitGeneratorError::error;
                        break;
                }
                if (err != dlp::jit::jitGeneratorError::success) {
                    goto cleanup;
                }
                // Must call ready() to readjust jump/branch targets with
                // respect to any new buffer created as part of AutoGrow mode
                // in Xbyak.
                gen->ready();
                kernelCodeBlocks[mr * numNRVariants + nr] =
                    const_cast<void*>(static_cast<const void*>(gen->getCode()));
                codeGenerators.push_back(std::move(gen));

                DLP_ENABLE_JIT_DUMP_AND_MONITOR(
                    kernelCodeBlocks[mr * numNRVariants + nr],
                    utils::JIT_KERNEL_SIZE, "jit_s8_gemm_kernel", params.MR,
                    params.NR, false, mr * numNRVariants + nr);
            }
        }
    }

    return dlp::jit::jitGeneratorError::success;

cleanup:
    codeGenerators.clear();
    for (auto& block : kernelCodeBlocks) {
        block = nullptr;
    }
    return err;
}

dlp::kernels::kernelError
jitAmdZenS8::executeKernel(dlp::kernels::kernelParams* _params)
{
    if (MR == 1) { // S8 GEMV M=1 kernel execution
        auto params = static_cast<dlp::kernels::gemvM1Params*>(_params);

        numElemsPerReg = 16;

        params->n_iter      = params->n / NR;
        params->n_left      = params->n % NR;
        params->n_left_16   = (params->n_left / 16) * 16;
        params->n_left_lt16 = params->n_left % 16;

        params->k_iter = params->k / KC;
        params->k_left = params->k % KC;
        params->kLeftmask =
            (params->k % VNNI_CONST == 0)
                ? 0xF
                : (0xF >> (VNNI_CONST - (params->k % VNNI_CONST)));

        params->k_iter_sub_iter = KC / (4 * VNNI_CONST);
        params->k_iter_sub_left = (KC % (4 * VNNI_CONST)) / VNNI_CONST;
        params->k_left_sub_iter = (params->k_left) / (4 * VNNI_CONST);
        params->k_left_sub_left =
            ((params->k_left) % (4 * VNNI_CONST)) / VNNI_CONST;

        // Masks for loading b_col_sum_vec to compensate for the conversion of
        // A elements from s8 to u8 for VNNI operations.
        // Initializing masks to 0xFFFF.
        params->k4Mask_i8_avx512 = 0xFFFF;
        params->k3Mask_i8_avx512 = 0xFFFF;
        params->k2Mask_i8_avx512 = 0xFFFF;
        params->k1Mask_i8_avx512 = 0xFFFF;

        if (params->n_left_16 == 48) {
            // k4 = (0xFFFF >> (16 - (nr0 & 0x0F)));
            params->k4Mask_i8_avx512 = 0xFFFF >> (16 - (params->n_left_lt16));
        } else if (params->n_left_16 == 32) {
            // k4 = 0x0
            params->k4Mask_i8_avx512 = 0x0;

            // k3 = (0xFFFF >> (16 - (nr0 & 0x0F)));
            params->k3Mask_i8_avx512 = 0xFFFF >> (16 - (params->n_left_lt16));
        } else if (params->n_left_16 == 16) {
            // k3 = k4 = 0x0
            params->k4Mask_i8_avx512 = 0x0;
            params->k3Mask_i8_avx512 = 0x0;

            // k2 = (0xFFFF >> (16 - (nr0 & 0x0F)));
            params->k2Mask_i8_avx512 = 0xFFFF >> (16 - (params->n_left_lt16));
        } else if (params->n_left_lt16 > 0) {
            // k2 = k3 = k4 = 0x0
            params->k2Mask_i8_avx512 = 0x0;
            params->k3Mask_i8_avx512 = 0x0;
            params->k4Mask_i8_avx512 = 0x0;

            // k1 = (0xFFFF >> (16 - (nr0 & 0x0F)));
            params->k1Mask_i8_avx512 = 0xFFFF >> (16 - (params->n_left_lt16));
        }

        if ((params->k % VNNI_CONST) != 0)
            params->k_left_sub_left++;

        int partial_elements =
            params->n_left % numElemsPerReg; // Elements in partial band

        if (params->n_left > 0) {
            params->nmask_avx512 =
                0xFFFF >> (numElemsPerReg - partial_elements);
        }

        params->psB = (params->k_left + 3) & ~3;

        // Deploy associated M=1 kernel
        md_t ker_idx = params->n_left;

        utils::jit_gemv_m1_kernel kernel =
            reinterpret_cast<utils::jit_gemv_m1_kernel>(
                kernelCodeBlocks[ker_idx]);

        md_t og_post_op_c_j = (params->kernelOpsAttr).post_op_c_j;
        kernel(params);
        // post_op_c_j is updated inside the JIT kernel's N-loop. Reset it
        // to the saved value and increment by the total n processed.
        og_post_op_c_j += params->n;
        (params->kernelOpsAttr).post_op_c_j = og_post_op_c_j;

    } else if (NR == 1) { // S8 GEMV N=1 kernel execution
        auto params = static_cast<dlp::kernels::gemvN1Params*>(_params);

        numElemsPerReg = (16 * VNNI_CONST); // 64 elements for AVX-512
        params->m_iter = params->m / MR;
        params->m_left = params->m % MR;
        params->k_iter = params->k / numElemsPerReg;
        params->k_left = params->k % numElemsPerReg;

        params->mmask_avx512 = (0xFFFF >> (MR - params->m_left));
        // When k_left == 0, shifting by numElemsPerReg (64) is undefined
        // behavior since the shift count equals the bit width of uint64_t.
        params->kmask_i8_avx512 =
            (params->k_left == 0)
                ? 0xFFFFFFFFFFFFFFFFULL
                : 0xFFFFFFFFFFFFFFFFULL >> (numElemsPerReg - params->k_left);

        int is_m_loop     = (params->m_iter != 0);
        int is_col_stored = ((params->rsC) == 1);

        int ker_idx = ((params->m_left * 4) + (is_col_stored * 2) + is_m_loop);

        // Deploy associated N=1 kernel
        utils::jit_gemv_n1_kernel kernel =
            reinterpret_cast<utils::jit_gemv_n1_kernel>(
                kernelCodeBlocks[ker_idx]);

        kernel(params);

    } else { // S8 GEMM kernel execution
        auto params = static_cast<dlp::kernels::gemmParams*>(_params);

        int processBlockSize = getProcessBlockSize();

        // Since JR loop is in framework, the 'n' dimension passed to this
        // function is always <= NR.
        md_t mFullPieces    = params->m / MR;
        md_t mPartialPieces = params->m % MR;

        // The JIT body emits K_UNROLL VNNI groups per outer-loop iteration,
        // so the trip count must divide out both factors. Mirrors the
        // FP32-VNNI template earlier in this file; K_UNROLL=1 reduces to
        // the original formula. VNNI_CONST = 4 K-elements per VNNI group.
        //
        // Generalised K-tail decomposition: kLeft (residual K count) splits
        // into kLeftIter (full VNNI groups in the tail, 0..K_UNROLL-1) and
        // kLeftRem (K-element residual, 0..VNNI_CONST-1). The two-stage
        // tail in s8_gemm_generator.cc consumes kLeftIter via full-group
        // iterations and kLeftRem via the masked load. The kLeftmask
        // formula moves to (1 << kLeftRem) - 1: identical value semantics
        // on any kLeftIter == 0 shape (which includes every
        // K%(K_UNROLL*VNNI_CONST)==0 shape) and strictly correct on the
        // K_UNROLL=2 K%8 != 0 shapes that the generalised tail unblocks.
        params->kIterBP   = params->k / (K_UNROLL * VNNI_CONST);
        params->kLeft     = params->k % (K_UNROLL * VNNI_CONST);
        params->kLeftIter = params->kLeft / VNNI_CONST;
        params->kLeftRem  = params->kLeft % VNNI_CONST;

        // Handle VNNI remainder case using AVX-512 masked memory load
        // (mask is on the kLeftRem residual; identical to the previous
        //  formula on any kLeftIter == 0 shape).
        params->kLeftmask = (1 << params->kLeftRem) - 1;

        int8_t*  aPtr = static_cast<int8_t*>(params->a);
        int8_t*  bPtr = static_cast<int8_t*>(params->b);
        int32_t* cPtr = static_cast<int32_t*>(params->c);
        int32_t* c_jr = cPtr;
        int32_t* c_ir = cPtr;
        md_t     rsB  = params->rsB;

        md_t n = params->n;
        md_t m = params->m;
        md_t k = params->k;

        // Update panel stride of A to account for MR rows
        params->psA = MR * params->psA;

        md_t og_post_op_c_i = (params->kernelOpsAttr).post_op_c_i;

        md_t nBlockSize  = (n >= processBlockSize) ? processBlockSize : n;
        md_t nFullpieces = nBlockSize / numElemsPerReg;
        md_t nRemainder  = nBlockSize % numElemsPerReg;

        // Process complete registers first (if any)
        if (nFullpieces > 0) {
            md_t elementsToProcess = nFullpieces * numElemsPerReg;
            params->rsB            = (nFullpieces * rsB) / VNNI_CONST;

            params->a = aPtr;
            params->c = c_jr;
            params->n = elementsToProcess;

            md_t kernel_n_idx = nFullpieces;
            if (params->m >= MR) {
                params->mIter = mFullPieces;
                int m_idx     = 0;
                int ker_idx   = m_idx * numNRVariants + kernel_n_idx;

                utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                    kernelCodeBlocks[ker_idx]);
                kernel(params);
            }

            if (mPartialPieces) {
                (params->a) = (int8_t*)(params->a) + mFullPieces * params->psA;
                (params->c) =
                    (int32_t*)(params->c) + mFullPieces * MR * params->rsC;
                int m_idx   = mPartialPieces;
                int ker_idx = m_idx * numNRVariants + kernel_n_idx;

                utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                    kernelCodeBlocks[ker_idx]);
                kernel(params);
            }

            // Advance B between the nFullpieces kernel call and the
            // nRemainder mask-kernel call by the same K-rounded-up amount
            // the u8s8 path uses earlier in this file. With the generalised
            // K-tail kLeft now covers up to K_UNROLL * VNNI_CONST - 1
            // K-elements; the previous (VNNI_CONST - kLeft) form produced
            // a negative offset when kLeftIter >= 1, advancing B by the
            // wrong number of bytes. The k_updated form below is
            // byte-equivalent to the old formula on every shape that
            // pre-fix had kLeftIter == 0, and correct on the
            // K_UNROLL=2 K%8 != 0 shapes the generalised tail unblocks.
            md_t k_updated =
                ((params->k + VNNI_CONST - 1) / VNNI_CONST) * VNNI_CONST;
            params->b = (int8_t*)(params->b) + elementsToProcess * k_updated;

            c_jr = (int32_t*)(c_jr) + elementsToProcess;
            (params->kernelOpsAttr).post_op_c_j += elementsToProcess;
            (params->kernelOpsAttr).b_sum_offset += nFullpieces * 16;

            // The following line is necessary to ensure a subtle bug
            // does not occur. Unlike the classic kernels where the
            // kernelOpsAttr is passed by value to the fringe kernels,
            // here it is kind of passed by reference, since the params
            // ptr is the only kernel argument (inside which is the
            // kernelOpsAttr). Since the while(n) loop can execute
            // multiple times, any state variable, like post_op_c_i, if
            // modified inside kernel, it needs to be reverted.
            (params->kernelOpsAttr).post_op_c_i = og_post_op_c_i;

            n -= elementsToProcess;
        }

        // Process remainder with mask (if any)
        if (nRemainder > 0) {
            params->a = aPtr;
            params->c = c_jr;
            params->n = nRemainder;

            params->rsB = numElemsPerReg * VNNI_CONST;

            // Mask calculation (architecture-specific)
            if (kType == utils::kernelInstrType::avx512_zmm_32_reg) {
                params->maskS32 = 0xFFFF >> (numElemsPerReg - nRemainder);
                // Post-ops module uses maskF32 for its mask registers, so we
                // need to set it as well for s8 kernels with post-ops
                params->maskF32[0] = params->maskS32;
            }

            // INLINE: Kernel execution with mask
            int kernel_n_idx = 0; // Always use mask kernel for nRemainder
            if (params->m >= MR) {
                params->mIter = mFullPieces;
                int m_idx     = 0;
                int ker_idx   = m_idx * numNRVariants + kernel_n_idx;

                utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                    kernelCodeBlocks[ker_idx]);
                kernel(params);
            }
            if (mPartialPieces) {
                (params->a) = (int8_t*)(params->a) + mFullPieces * params->psA;
                (params->c) =
                    (int32_t*)(params->c) + mFullPieces * MR * params->rsC;
                int m_idx   = mPartialPieces;
                int ker_idx = m_idx * numNRVariants + kernel_n_idx;

                utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                    kernelCodeBlocks[ker_idx]);
                kernel(params);
            }
        }
    }
    return dlp::kernels::kernelError::success;
}

// FP16 JIT Generator
jitAmdZenFP16::jitAmdZenFP16()
    : mKernelDatatypes({ dlp::kernel_frame::kernelDatatype::f16f16f16of16,
                         dlp::kernel_frame::kernelDatatype::f16f16f16of32 })
    , mIsaFeaturesRequired({ dlp::cpu_utils::isaFeature::avx512fp16 })
    , kType(utils::kernelInstrType::none)
    , numElemsPerReg(1) // Initializing with 1 to avoid div by zero
{
    // Add additional required ISA features
    mIsaFeaturesRequired.push_back(dlp::cpu_utils::isaFeature::avx512f);
    mIsaFeaturesRequired.push_back(dlp::cpu_utils::isaFeature::avx512bw);
}

jitAmdZenFP16::~jitAmdZenFP16()
{
    codeGenerators.clear();
    for (auto& block : kernelCodeBlocks) {
        block = nullptr;
    }
}

int
jitAmdZenFP16::getProcessBlockSize() const
{
    switch (kType) {
        case utils::kernelInstrType::avx512_zmm_32_reg:
            return NR; // Process full NR (128) elements at once

        default:
            return 0; // Invalid/unsupported kernel type
    }
}

void
jitAmdZenFP16::setGeneratorKernelMetaInfo(
    dlp::kernel_frame::kernelInstrPreference kInstPref)
{
    kType = utils::kernelInstrType::none;
    switch (kInstPref) {
        case dlp::kernel_frame::kernelInstrPreference::avx512_zmm_favour: {
            kType = utils::kernelInstrType::avx512_zmm_32_reg;
            // For FP16: 32 elements per ZMM (64 bytes / 2 bytes per FP16)
            numElemsPerReg = FP16_PER_ZMM; // 32
            break;
        }
        default:
            break;
    }
}

dlp::jit::jitGeneratorError
jitAmdZenFP16::generateAllKernels(const dlp::jit::jitGeneratorContext& jI)
{
    dlp::jit::jitGeneratorError err = dlp::jit::jitGeneratorError::error;

    MR              = (jI.kI).mr;
    NR              = (jI.kI).nr;
    KC              = (jI.kI).kc;
    K_UNROLL        = (jI.kI).k_unroll;
    PREFETCH_C_DIST = (jI.kI).prefetch_c_dist;
    c_downscale     = (jI.kI).c_downscale;
    mtag_b          = (jI.kI).mtag_b; // Store memory tag for B matrix

    // Set kernel type based on instruction preference
    setGeneratorKernelMetaInfo(jI.kI.kInstPref);

    if (kType == utils::kernelInstrType::none) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    int processBlockSize = getProcessBlockSize();

    if (MR == 1) {
        // FP16 GEMV M=1: y = x * B (1 x K vector * K x N matrix = 1 x N vector)
        // Generate NR kernels, each handling specific n_left fringe case
        numKernelVariants = NR;
        kernelCodeBlocks.resize(numKernelVariants);

        utils::gemvM1GeneratorParams params(
            c_downscale, NR, 0, KC, K_UNROLL, mtag_b, true, true, true, true,
            dlp::kernel_frame::storageFormat::rowMajor,
            (jI.kI).alphaScalingType, (jI.kI).betaScalingType, kType);

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        params.NR         = NR;
        params.N_LEFT     = 0;
        params.KC         = KC;
        params.K_SUB_ITER = K_UNROLL;
        params.nloop      = true;
        params.kloop      = true;
        params.nfringe    = false;
        params.kfringe    = true;

        for (iter_t i = 0; i < NR; ++i) {
            params.N_LEFT  = i;
            params.nfringe = (i != 0);

            std::unique_ptr<Xbyak::CodeGenerator> gen;
            switch (kType) {
                case utils::kernelInstrType::avx512_zmm_32_reg: {
                    auto g = std::make_unique<jitFP16GEMVM1<
                        utils::kernelInstrType::avx512_zmm_32_reg>>(
                        Xbyak::AutoGrow, utils::JIT_KERNEL_SIZE);
                    err = g->generateKernel(params);
                    gen = std::move(g);
                    break;
                }
                default:
                    err = dlp::jit::jitGeneratorError::error;
                    break;
            }

            if (err != dlp::jit::jitGeneratorError::success) {
                goto cleanup;
            }
            gen->ready();
            kernelCodeBlocks[i] =
                const_cast<void*>(static_cast<const void*>(gen->getCode()));
            codeGenerators.push_back(std::move(gen));

            int n_left_suf = (i != 0) ? i : params.NR;
            DLP_ENABLE_JIT_DUMP_AND_MONITOR(
                kernelCodeBlocks[i], utils::JIT_KERNEL_SIZE,
                "jit_fp16_gemv_m1_kernel", 1, n_left_suf, false, i);
        }

    } else if (NR == 1) {
        // FP16 GEMV N=1: y = A * x (M x K matrix * K x 1 vector = M x 1 vector)
        // Generate MR * 4 kernels for different m_left, storage format, mloop
        // combinations
        numKernelVariants = MR * 4;
        kernelCodeBlocks.resize(numKernelVariants);

        utils::gemvN1GeneratorParams params(
            MR, 0, c_downscale, false, false, false, false,
            dlp::kernel_frame::storageFormat::rowMajor,
            (jI.kI).alphaScalingType, (jI.kI).betaScalingType, kType);

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        params.MR      = MR;
        params.mloop   = true;
        params.kloop   = true;
        params.mfringe = false;
        params.kfringe = true;

        for (iter_t m_left = 0; m_left < MR; ++m_left) {
            params.M_LEFT  = m_left;
            params.mfringe = (m_left != 0);

            for (iter_t j = 0; j < 4; ++j) {
                // Kernel indexing:
                // 0: row-stored, without mloop
                // 1: row-stored, with mloop
                // 2: col-stored, without mloop
                // 3: col-stored, with mloop
                params.mloop = ((j == 1) || (j == 3));
                params.yFormat =
                    ((j / 2) == 0) ? dlp::kernel_frame::storageFormat::rowMajor
                                   : dlp::kernel_frame::storageFormat::colMajor;

                std::unique_ptr<Xbyak::CodeGenerator> gen;
                switch (kType) {
                    case utils::kernelInstrType::avx512_zmm_32_reg: {
                        auto g = std::make_unique<jitFP16GEMVN1<
                            utils::kernelInstrType::avx512_zmm_32_reg>>(
                            Xbyak::AutoGrow, utils::JIT_KERNEL_SIZE);
                        err = g->generateKernel(params);
                        gen = std::move(g);
                        break;
                    }
                    default:
                        err = dlp::jit::jitGeneratorError::error;
                        break;
                }

                if (err != dlp::jit::jitGeneratorError::success) {
                    goto cleanup;
                }
                gen->ready();
                kernelCodeBlocks[m_left * 4 + j] =
                    const_cast<void*>(static_cast<const void*>(gen->getCode()));
                codeGenerators.push_back(std::move(gen));

                int m_left_suf = (m_left != 0) ? m_left : params.MR;
                DLP_ENABLE_JIT_DUMP_AND_MONITOR(
                    kernelCodeBlocks[m_left * 4 + j], utils::JIT_KERNEL_SIZE,
                    "jit_fp16_gemv_n1_kernel", m_left_suf, j, false,
                    m_left * 4 + j);
            }
        }

    } else {
        // Generate GEMM kernels (MR > 1, NR > 1)
        // For FP16: numElemsPerReg = 32, so NR=128 means 4 registers
        numNRVariants     = (processBlockSize / numElemsPerReg) + 1;
        numMRVariants     = MR;
        numKernelVariants = numMRVariants * numNRVariants;

        kernelCodeBlocks.resize(numKernelVariants);

        // Initialize generator params
        utils::generatorParams params(
            0, 0, K_UNROLL, PREFETCH_C_DIST, c_downscale, 0, false, false,
            false, (jI.kI).alphaScalingType, (jI.kI).betaScalingType, kType);

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        // Generate all kernels for the given MR and NR
        for (iter_t mr = 0; mr < numMRVariants; mr++) {
            for (iter_t nr = 0; nr < numNRVariants; nr++) {
                params.MR          = (mr == 0) ? MR : mr;
                params.mLoop       = (mr == 0);
                params.NR          = nr * numElemsPerReg;
                params.useMask     = (nr == 0);
                params.numMaskRegs = (params.useMask) ? 1 : 0;

                std::unique_ptr<Xbyak::CodeGenerator> gen;
                switch (kType) {
                    case utils::kernelInstrType::avx512_zmm_32_reg: {
                        auto g = std::make_unique<jitFP16_GEMM<
                            utils::kernelInstrType::avx512_zmm_32_reg>>(
                            Xbyak::AutoGrow, utils::JIT_KERNEL_SIZE);
                        err = g->generateKernel(params);
                        gen = std::move(g);
                        break;
                    }
                    default:
                        err = dlp::jit::jitGeneratorError::error;
                        break;
                }

                if (err != dlp::jit::jitGeneratorError::success) {
                    goto cleanup;
                }
                gen->ready();
                kernelCodeBlocks[mr * numNRVariants + nr] =
                    const_cast<void*>(static_cast<const void*>(gen->getCode()));
                codeGenerators.push_back(std::move(gen));

                DLP_ENABLE_JIT_DUMP_AND_MONITOR(
                    kernelCodeBlocks[mr * numNRVariants + nr],
                    utils::JIT_KERNEL_SIZE, "jit_fp16_gemm_kernel", params.MR,
                    params.NR, params.useMask, mr * numNRVariants + nr);
            }
        }
    }

    return dlp::jit::jitGeneratorError::success;

cleanup:
    codeGenerators.clear();
    for (auto& block : kernelCodeBlocks) {
        block = nullptr;
    }
    return err;
}

dlp::kernels::kernelError
jitAmdZenFP16::executeKernel(dlp::kernels::kernelParams* _params)
{
    // FP16 GEMV M=1 execution
    if (MR == 1) {
        auto params = static_cast<dlp::kernels::gemvM1Params*>(_params);

        params->n_iter = params->n / NR;
        params->n_left = params->n % NR;
        params->k_iter = params->k / KC;
        params->k_left = params->k % KC;

        params->k_iter_sub_iter = KC / K_UNROLL;
        params->k_iter_sub_left = KC % K_UNROLL;
        params->k_left_sub_iter = (params->k_left) / K_UNROLL;
        params->k_left_sub_left = (params->k_left) % K_UNROLL;

        int partial_elements = params->n_left % numElemsPerReg;
        if (params->n_left > 0) {
            // Use nmask_fp16_avx512 for FP16 (32-bit mask for 32 elements per
            // ZMM). When partial_elements == 0, n_left is a multiple of
            // numElemsPerReg and we want a full mask (all elements active),
            // while also avoiding undefined behavior from shifting by 32.
            if (partial_elements == 0) {
                params->nmask_fp16_avx512 = 0xFFFFFFFFu;
            } else {
                params->nmask_fp16_avx512 =
                    0xFFFFFFFFu >> (numElemsPerReg - partial_elements);
            }

            // F32 postops mask: 16 F32 elements per ZMM
            static constexpr int F32_PER_ZMM = 16;
            int                  f32_partial = params->n_left % F32_PER_ZMM;
            if (f32_partial == 0) {
                params->nmask_avx512 = 0xFFFFu;
            } else {
                params->nmask_avx512 = static_cast<uint16_t>(
                    0xFFFFu >> (F32_PER_ZMM - f32_partial));
            }
        }

        // Deploy the associated kernel based on n_left
        md_t kernel_idx = params->n_left;

        utils::jit_gemv_m1_kernel kernel =
            reinterpret_cast<utils::jit_gemv_m1_kernel>(
                kernelCodeBlocks[kernel_idx]);
        kernel(params);

        // Update post_op_c_j by the total n processed in this kernel call
        (params->kernelOpsAttr).post_op_c_j += params->n;

        return dlp::kernels::kernelError::success;
    }

    // FP16 GEMV N=1 execution
    if (NR == 1) {
        auto params = static_cast<dlp::kernels::gemvN1Params*>(_params);

        params->m_iter = params->m / MR;
        params->m_left = params->m % MR;
        params->k_iter = params->k / numElemsPerReg;
        params->k_left = params->k % numElemsPerReg;

        // Set masks for FP16 (32-bit masks for 32 elements per ZMM)
        // Use kmask_avx512 for k-dimension mask (32-bit for 32 FP16 elements)
        if (params->k_left == 0) {
            params->kmask_fp16_avx512 = 0xFFFFFFFFu;
        } else {
            params->kmask_fp16_avx512 =
                0xFFFFFFFFu >> (numElemsPerReg - params->k_left);
        }
        params->mmask_avx512 = 0xFFFF >> (MR - params->m_left);

        int is_m_loop     = ((params->m) >= MR);
        int is_col_stored = ((params->rsC) == 1);

        // Kernel index: m_left * 4 + is_col_stored * 2 + is_m_loop
        int kernel_idx = params->m_left * 4 + is_col_stored * 2 + is_m_loop;

        utils::jit_gemv_n1_kernel kernel =
            reinterpret_cast<utils::jit_gemv_n1_kernel>(
                kernelCodeBlocks[kernel_idx]);
        kernel(params);

        return dlp::kernels::kernelError::success;
    }

    // FP16 GEMM execution (MR > 1, NR > 1)
    auto params = static_cast<dlp::kernels::gemmParams*>(_params);

    int  processBlockSize = getProcessBlockSize();
    md_t mFullPieces      = params->m / MR;
    md_t mPartialPieces   = params->m % MR;

    params->kIterBP = params->k / K_UNROLL;
    params->kLeft   = params->k % K_UNROLL;

    dlp::float16* aPtr = static_cast<dlp::float16*>(params->a);
    dlp::float16* bPtr = static_cast<dlp::float16*>(params->b);
    dlp::float16* cPtr = static_cast<dlp::float16*>(params->c);
    dlp::float16* c_jr = cPtr;

    // On the of32 rail params->c aliases user F32 C, so any element-typed
    // advance (M-step, JR-step, remainder pass) must scale by
    // sizeof(float) instead of sizeof(float16). c_downscale is a
    // build-time constant per kernel, so we cache the rail bool once here
    // and the conditional casts below are zero-cost on the hot path.
    bool cIsF32 = ((params->kernelOpsAttr).c_stor_type == DLP_F32);

    md_t rsB            = params->rsB;
    md_t n              = params->n;
    md_t og_post_op_c_i = (params->kernelOpsAttr).post_op_c_i;

    md_t nBlockSize  = (n >= processBlockSize) ? processBlockSize : n;
    md_t nFullpieces = nBlockSize / numElemsPerReg;
    md_t nRemainder  = nBlockSize % numElemsPerReg;

    // Process complete registers first (if any)
    if (nFullpieces > 0) {
        md_t elementsToProcess = nFullpieces * numElemsPerReg;

        params->a = aPtr;
        params->c = c_jr;
        params->n = elementsToProcess;

        // Update B stride for FP16 packing (only for packed B)
        if (mtag_b == PACK || mtag_b == REORDERED) {
            params->rsB = (nFullpieces * rsB) / (NR / numElemsPerReg);
        }
        // For UNPACKED B, rsB stays as original value

        md_t kernel_n_idx = nFullpieces;
        if (params->m >= MR) {
            params->mIter = mFullPieces;
            int m_idx     = 0;

            utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                kernelCodeBlocks[m_idx * numNRVariants + kernel_n_idx]);

            DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(kernel));
            kernel(params);
            // psA already contains MR * rs_a, so don't multiply by MR again
            (params->a) =
                (dlp::float16*)(params->a) + mFullPieces * params->psA;
            // Type-aware M-step on params->c: of32 advances by float
            // elements; of16 by float16 elements. params->rsC is the
            // element-typed user ldc passed through unchanged.
            if (cIsF32) {
                params->c = (dlp::float16*)((float*)(params->c)
                                            + MR * mFullPieces * params->rsC);
            } else {
                params->c =
                    (dlp::float16*)(params->c) + MR * mFullPieces * params->rsC;
            }
        }

        if (mPartialPieces) {
            int m_idx = mPartialPieces;

            utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                kernelCodeBlocks[m_idx * numNRVariants + kernel_n_idx]);

            DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(kernel));
            kernel(params);
        }

        // Update B pointer for next block
        // For packed B: advance by elementsToProcess * k (NR*K block layout)
        // For unpacked B (row-major): advance by elementsToProcess (next
        // columns)
        if (mtag_b == PACK || mtag_b == REORDERED) {
            params->b =
                (dlp::float16*)(params->b) + elementsToProcess * params->k;
        } else {
            params->b = (dlp::float16*)(params->b) + elementsToProcess;
        }
        // Type-aware JR-step on c_jr: of32 advances by F32 elements.
        if (cIsF32) {
            c_jr = (dlp::float16*)((float*)c_jr + elementsToProcess);
        } else {
            c_jr = c_jr + elementsToProcess;
        }
        (params->kernelOpsAttr).post_op_c_i = og_post_op_c_i;
        (params->kernelOpsAttr).post_op_c_j += elementsToProcess;
        n -= elementsToProcess;
    }

    // Process remainder with mask (if any)
    if (nRemainder > 0) {
        params->a = aPtr;
        params->c = c_jr;
        params->n = nRemainder;

        // Update B stride for FP16 packing (only for packed B)
        if (mtag_b == PACK || mtag_b == REORDERED) {
            params->rsB = rsB / (NR / numElemsPerReg);
        }
        // For UNPACKED B, rsB stays as original value

        // Set mask for FP16 elements (32-bit mask for 32 elements per ZMM)
        uint32_t fp16Mask =
            0xFFFFFFFFu >> (numElemsPerReg - static_cast<int>(nRemainder));
        params->maskFP16 = fp16Mask;

        int kernel_n_idx = 0; // Use mask kernel
        if (params->m >= MR) {
            params->mIter = mFullPieces;
            int m_idx     = 0;

            utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                kernelCodeBlocks[m_idx * numNRVariants + kernel_n_idx]);

            DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(kernel));
            kernel(params);
            // psA already contains MR * rs_a, so don't multiply by MR again
            (params->a) =
                (dlp::float16*)(params->a) + mFullPieces * params->psA;
            // Type-aware M-step (remainder pass), same contract as above.
            if (cIsF32) {
                params->c = (dlp::float16*)((float*)(params->c)
                                            + MR * mFullPieces * params->rsC);
            } else {
                params->c =
                    (dlp::float16*)(params->c) + MR * mFullPieces * params->rsC;
            }
        }

        if (mPartialPieces) {
            int m_idx = mPartialPieces;

            utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                kernelCodeBlocks[m_idx * numNRVariants + kernel_n_idx]);

            DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(kernel));
            kernel(params);
        }
    }

    return dlp::kernels::kernelError::success;
}

// F32×FP16→F32 JIT Generator
jitAmdZenF32FP16::jitAmdZenF32FP16()
    : mKernelDatatypes({ dlp::kernel_frame::kernelDatatype::f32f16f32of32 })
    , mIsaFeaturesRequired({ dlp::cpu_utils::isaFeature::avx512f })
    , kType(utils::kernelInstrType::none)
    , numElemsPerReg(1)
{
    mIsaFeaturesRequired.push_back(dlp::cpu_utils::isaFeature::avx512bw);
}

jitAmdZenF32FP16::~jitAmdZenF32FP16()
{
    codeGenerators.clear();
    for (auto& block : kernelCodeBlocks) {
        block = nullptr;
    }
}

int
jitAmdZenF32FP16::getProcessBlockSize() const
{
    switch (kType) {
        case utils::kernelInstrType::avx512_zmm_32_reg:
            return NR;

        default:
            return 0;
    }
}

void
jitAmdZenF32FP16::setGeneratorKernelMetaInfo(
    dlp::kernel_frame::kernelInstrPreference kInstPref)
{
    kType = utils::kernelInstrType::none;
    switch (kInstPref) {
        case dlp::kernel_frame::kernelInstrPreference::avx512_zmm_favour: {
            kType = utils::kernelInstrType::avx512_zmm_32_reg;
            // For F32 output: 16 elements per ZMM (64 bytes / 4 bytes per F32)
            numElemsPerReg = F32_PER_ZMM; // 16
            break;
        }
        default:
            break;
    }
}

dlp::jit::jitGeneratorError
jitAmdZenF32FP16::generateAllKernels(const dlp::jit::jitGeneratorContext& jI)
{
    dlp::jit::jitGeneratorError err = dlp::jit::jitGeneratorError::error;

    MR              = (jI.kI).mr;
    NR              = (jI.kI).nr;
    KC              = (jI.kI).kc;
    K_UNROLL        = (jI.kI).k_unroll;
    PREFETCH_C_DIST = (jI.kI).prefetch_c_dist;
    c_downscale     = (jI.kI).c_downscale;
    mtag_b          = (jI.kI).mtag_b;

    setGeneratorKernelMetaInfo(jI.kI.kInstPref);

    if (kType == utils::kernelInstrType::none) {
        return dlp::jit::jitGeneratorError::notSupported;
    }

    int processBlockSize = getProcessBlockSize();

    if (MR == 1) {
        // F32×FP16 GEMV M=1: y = x * B (1×K F32 × K×N FP16 = 1×N F32)
        // NR = 64 (4 ZMMs of 16 F32 each)
        // Generate NR kernels, one per n_left fringe case (0..NR-1)
        numKernelVariants = NR;
        kernelCodeBlocks.resize(numKernelVariants);

        utils::gemvM1GeneratorParams params(
            c_downscale, NR, 0, KC, K_UNROLL, mtag_b, true, true, true, true,
            dlp::kernel_frame::storageFormat::rowMajor,
            (jI.kI).alphaScalingType, (jI.kI).betaScalingType, kType);

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        params.NR         = NR;
        params.N_LEFT     = 0;
        params.KC         = KC;
        params.K_SUB_ITER = K_UNROLL;
        params.nloop      = true;
        params.kloop      = true;
        params.nfringe    = false;
        params.kfringe    = true;

        for (int i = 0; i < NR; ++i) {
            params.N_LEFT  = i;
            params.nfringe = (i != 0);

            std::unique_ptr<Xbyak::CodeGenerator> gen;
            switch (kType) {
                case utils::kernelInstrType::avx512_zmm_32_reg: {
                    auto g = std::make_unique<jitF32FP16GEMVM1<
                        utils::kernelInstrType::avx512_zmm_32_reg>>(
                        Xbyak::AutoGrow, utils::JIT_KERNEL_SIZE);
                    err = g->generateKernel(params);
                    gen = std::move(g);
                    break;
                }
                default:
                    err = dlp::jit::jitGeneratorError::error;
                    break;
            }

            if (err != dlp::jit::jitGeneratorError::success) {
                goto cleanup;
            }
            gen->ready();
            kernelCodeBlocks[i] =
                const_cast<void*>(static_cast<const void*>(gen->getCode()));
            codeGenerators.push_back(std::move(gen));

            int n_left_suf = (i != 0) ? i : params.NR;
            DLP_ENABLE_JIT_DUMP_AND_MONITOR(
                kernelCodeBlocks[i], utils::JIT_KERNEL_SIZE,
                "jit_f32fp16_gemv_m1_kernel", 1, n_left_suf, false, i);
        }

    } else if (NR == 1) {
        // F32×FP16 GEMV N=1: y = A * x (M×K F32 × K×1 FP16 = M×1 F32)
        // MR = 16 (process 16 rows at a time)
        // Generate MR * 4 kernels for m_left × {rowMajor,colMajor} ×
        // {mloop,no-mloop}
        numKernelVariants = MR * 4;
        kernelCodeBlocks.resize(numKernelVariants);

        utils::gemvN1GeneratorParams params(
            MR, 0, c_downscale, false, false, false, false,
            dlp::kernel_frame::storageFormat::rowMajor,
            (jI.kI).alphaScalingType, (jI.kI).betaScalingType, kType);

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        params.MR      = MR;
        params.mloop   = true;
        params.kloop   = true;
        params.mfringe = false;
        params.kfringe = true;

        for (int m_left = 0; m_left < MR; ++m_left) {
            params.M_LEFT  = m_left;
            params.mfringe = (m_left != 0);

            for (int j = 0; j < 4; ++j) {
                params.mloop = ((j == 1) || (j == 3));
                params.yFormat =
                    ((j / 2) == 0) ? dlp::kernel_frame::storageFormat::rowMajor
                                   : dlp::kernel_frame::storageFormat::colMajor;

                std::unique_ptr<Xbyak::CodeGenerator> gen;
                switch (kType) {
                    case utils::kernelInstrType::avx512_zmm_32_reg: {
                        auto g = std::make_unique<jitF32FP16GEMVN1<
                            utils::kernelInstrType::avx512_zmm_32_reg>>(
                            Xbyak::AutoGrow, utils::JIT_KERNEL_SIZE);
                        err = g->generateKernel(params);
                        gen = std::move(g);
                        break;
                    }
                    default:
                        err = dlp::jit::jitGeneratorError::error;
                        break;
                }

                if (err != dlp::jit::jitGeneratorError::success) {
                    goto cleanup;
                }
                gen->ready();
                kernelCodeBlocks[m_left * 4 + j] =
                    const_cast<void*>(static_cast<const void*>(gen->getCode()));
                codeGenerators.push_back(std::move(gen));

                int m_left_suf = (m_left != 0) ? m_left : params.MR;
                DLP_ENABLE_JIT_DUMP_AND_MONITOR(
                    kernelCodeBlocks[m_left * 4 + j], utils::JIT_KERNEL_SIZE,
                    "jit_f32fp16_gemv_n1_kernel", m_left_suf, j, false,
                    m_left * 4 + j);
            }
        }

    } else {
        // Generate GEMM kernels (MR > 1, NR > 1)
        numNRVariants     = (processBlockSize / numElemsPerReg) + 1;
        numMRVariants     = MR;
        numKernelVariants = numMRVariants * numNRVariants;

        kernelCodeBlocks.resize(numKernelVariants);

        utils::generatorParams params(
            0, 0, K_UNROLL, PREFETCH_C_DIST, c_downscale, 0, false, false,
            false, (jI.kI).alphaScalingType, (jI.kI).betaScalingType, kType);

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        for (int mr = 0; mr < numMRVariants; mr++) {
            for (int nr = 0; nr < numNRVariants; nr++) {
                params.MR          = (mr == 0) ? MR : mr;
                params.mLoop       = (mr == 0);
                params.NR          = nr * numElemsPerReg;
                params.useMask     = (nr == 0);
                params.numMaskRegs = (params.useMask) ? 1 : 0;

                std::unique_ptr<Xbyak::CodeGenerator> gen;
                switch (kType) {
                    case utils::kernelInstrType::avx512_zmm_32_reg: {
                        auto g = std::make_unique<jitF32FP16_GEMM<
                            utils::kernelInstrType::avx512_zmm_32_reg>>(
                            Xbyak::AutoGrow, utils::JIT_KERNEL_SIZE);
                        err = g->generateKernel(params);
                        gen = std::move(g);
                        break;
                    }
                    default:
                        err = dlp::jit::jitGeneratorError::error;
                        break;
                }

                if (err != dlp::jit::jitGeneratorError::success) {
                    goto cleanup;
                }
                gen->ready();
                kernelCodeBlocks[mr * numNRVariants + nr] =
                    const_cast<void*>(static_cast<const void*>(gen->getCode()));
                codeGenerators.push_back(std::move(gen));

                DLP_ENABLE_JIT_DUMP_AND_MONITOR(
                    kernelCodeBlocks[mr * numNRVariants + nr],
                    utils::JIT_KERNEL_SIZE, "jit_f32fp16_gemm_kernel",
                    params.MR, params.NR, params.useMask,
                    mr * numNRVariants + nr);
            }
        }
    }

    return dlp::jit::jitGeneratorError::success;

cleanup:
    codeGenerators.clear();
    for (auto& block : kernelCodeBlocks) {
        block = nullptr;
    }
    return err;
}

dlp::kernels::kernelError
jitAmdZenF32FP16::executeKernel(dlp::kernels::kernelParams* _params)
{
    // F32×FP16→F32 GEMV M=1 execution
    if (MR == 1) {
        auto params = static_cast<dlp::kernels::gemvM1Params*>(_params);

        params->n_iter = params->n / NR;
        params->n_left = params->n % NR;
        params->k_iter = params->k / KC;
        params->k_left = params->k % KC;

        params->k_iter_sub_iter = KC / K_UNROLL;
        params->k_iter_sub_left = KC % K_UNROLL;
        params->k_left_sub_iter = (params->k_left) / K_UNROLL;
        params->k_left_sub_left = (params->k_left) % K_UNROLL;

        // N-dimension mask for F32: 16-bit (16 F32 elements per ZMM)
        int partial_elements = params->n_left % numElemsPerReg;
        if (params->n_left > 0) {
            if (partial_elements == 0) {
                params->nmask_fp16_avx512 = 0xFFFF;
            } else {
                params->nmask_fp16_avx512 =
                    0xFFFF >> (numElemsPerReg - partial_elements);
            }
            params->nmask_avx512 = params->nmask_fp16_avx512;
        }

        int kernel_idx = static_cast<int>(params->n_left);

        utils::jit_gemv_m1_kernel kernel =
            reinterpret_cast<utils::jit_gemv_m1_kernel>(
                kernelCodeBlocks[kernel_idx]);
        kernel(params);

        return dlp::kernels::kernelError::success;
    }

    // F32×FP16→F32 GEMV N=1 execution
    if (NR == 1) {
        auto params = static_cast<dlp::kernels::gemvN1Params*>(_params);

        // K-iterations: process F32_PER_ZMM (16) elements at a time
        // x is FP16 but we load 16 FP16 → convert to 16 F32
        params->m_iter = params->m / MR;
        params->m_left = params->m % MR;
        params->k_iter = params->k / numElemsPerReg;
        params->k_left = params->k % numElemsPerReg;

        // K-mask: 16-bit for 16 F32/FP16 elements per ZMM
        if (params->k_left == 0) {
            params->kmask_fp16_avx512 = 0xFFFF;
        } else {
            params->kmask_fp16_avx512 =
                0xFFFF >> (numElemsPerReg - params->k_left);
        }
        params->mmask_avx512 = 0xFFFF >> (MR - params->m_left);

        int is_m_loop     = ((params->m) >= MR);
        int is_col_stored = ((params->rsC) == 1);

        int kernel_idx = params->m_left * 4 + is_col_stored * 2 + is_m_loop;

        utils::jit_gemv_n1_kernel kernel =
            reinterpret_cast<utils::jit_gemv_n1_kernel>(
                kernelCodeBlocks[kernel_idx]);
        kernel(params);

        return dlp::kernels::kernelError::success;
    }

    // F32×FP16→F32 GEMM execution (MR > 1, NR > 1)
    auto params = static_cast<dlp::kernels::gemmParams*>(_params);

    int processBlockSize = getProcessBlockSize();
    int mFullPieces      = params->m / MR;
    int mPartialPieces   = params->m % MR;

    params->kIterBP = params->k / K_UNROLL;
    params->kLeft   = params->k % K_UNROLL;

    float*        aPtr = static_cast<float*>(params->a);
    dlp::float16* bPtr = static_cast<dlp::float16*>(params->b);
    float*        cPtr = static_cast<float*>(params->c);
    float*        c_jr = cPtr;

    int  rsB            = params->rsB;
    md_t n              = params->n;
    md_t og_post_op_c_i = (params->kernelOpsAttr).post_op_c_i;

    int nBlockSize  = (n >= processBlockSize) ? processBlockSize : n;
    int nFullpieces = nBlockSize / numElemsPerReg;
    int nRemainder  = nBlockSize % numElemsPerReg;

    if (nFullpieces > 0) {
        int elementsToProcess = nFullpieces * numElemsPerReg;

        params->a = aPtr;
        params->c = c_jr;
        params->n = elementsToProcess;

        int kernel_n_idx = nFullpieces;
        if (params->m >= MR) {
            params->mIter = mFullPieces;
            int m_idx     = 0;

            utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                kernelCodeBlocks[m_idx * numNRVariants + kernel_n_idx]);

            DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(kernel));
            kernel(params);
            (params->a) = (float*)(params->a) + mFullPieces * params->psA;
            (params->c) = (float*)(params->c) + MR * mFullPieces * params->rsC;
        }

        if (mPartialPieces) {
            int m_idx = mPartialPieces;

            utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                kernelCodeBlocks[m_idx * numNRVariants + kernel_n_idx]);

            DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(kernel));
            kernel(params);
        }

        params->b = (dlp::float16*)(params->b) + elementsToProcess;
        c_jr      = c_jr + elementsToProcess;
        (params->kernelOpsAttr).post_op_c_i = og_post_op_c_i;
        (params->kernelOpsAttr).post_op_c_j += elementsToProcess;
        n -= elementsToProcess;
    }

    if (nRemainder > 0) {
        params->a = aPtr;
        params->c = c_jr;
        params->n = nRemainder;

        uint16_t f32Mask   = 0xFFFF >> (numElemsPerReg - nRemainder);
        params->maskFP16   = f32Mask;
        params->maskF32[0] = f32Mask;

        int kernel_n_idx = 0;
        if (params->m >= MR) {
            params->mIter = mFullPieces;
            int m_idx     = 0;

            utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                kernelCodeBlocks[m_idx * numNRVariants + kernel_n_idx]);

            DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(kernel));
            kernel(params);
            (params->a) = (float*)(params->a) + mFullPieces * params->psA;
            (params->c) = (float*)(params->c) + MR * mFullPieces * params->rsC;
        }

        if (mPartialPieces) {
            int m_idx = mPartialPieces;

            utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                kernelCodeBlocks[m_idx * numNRVariants + kernel_n_idx]);

            DLP_JIT_DEBUG_HELPER_BREAK(reinterpret_cast<void*>(kernel));
            kernel(params);
        }
    }

    return dlp::kernels::kernelError::success;
}

DLP_REGISTER_STATIC_GEMM_JIT_GENERATOR(jitAmdZenFP32, "dlp_amdzen_jit");
DLP_REGISTER_STATIC_GEMM_JIT_GENERATOR(jitAmdZenBF16, "dlp_amdzen_jit");
DLP_REGISTER_STATIC_GEMM_JIT_GENERATOR(jitAmdZenU8S8, "dlp_amdzen_jit");
DLP_REGISTER_STATIC_GEMM_JIT_GENERATOR(jitAmdZenS8, "dlp_amdzen_s8_jit");
DLP_REGISTER_STATIC_GEMM_JIT_GENERATOR(jitAmdZenFP16, "dlp_amdzen_fp16_jit");
DLP_REGISTER_STATIC_GEMM_JIT_GENERATOR(jitAmdZenF32FP16,
                                       "dlp_amdzen_f32fp16_jit");

} // namespace amdzen::gen
