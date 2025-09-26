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
#include "cpu_utils/cpu_features.hh"
#include "f32_gemm_generator.hh"
#include "f32_gemv.hh"
#include "jit_register/jit_register.hh"
#include "traits.hh"

// #define DLP_DUMP_JIT_CODE 1
namespace amdzen::gen {

jitAmdZenFP32::jitAmdZenFP32()
    : mKernelDatatypes({ dlp::kernel_frame::kernelDatatype::f32f32f32of32 })
    // TODO: Hardcoded for now, need to make it dynamic
    , mIsaFeaturesRequired{}
    , kType(utils::kernelInstrType::none)
    , numElemsPerReg(1) // Initializing with 1 to avoid div by zero
{
}

jitAmdZenFP32::~jitAmdZenFP32()
{
    for (auto& codeBlock : kernelCodeBlocks) {
        utils::jitHelperUtils::deallocateJitMemory(codeBlock,
                                                   utils::JIT_KERNEL_SIZE);
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

/* Function to generate all kernels */
dlp::jit::jitGeneratorError
jitAmdZenFP32::generateAllKernels(const dlp::jit::jitGeneratorContext& jI)
{

    dlp::jit::jitGeneratorError err = dlp::jit::jitGeneratorError::error;

    MR       = (jI.kI).mr;
    NR       = (jI.kI).nr;
    KC       = (jI.kI).kc;
    K_UNROLL = (jI.kI).k_unroll;

    // Hardcoding the FP32 kernel datatype for now
    const dlp::kernel_frame::kernelDatatype kdt =
        dlp::kernel_frame::kernelDatatype::f32f32f32of32;

    // Convert kernelInstrPreference to kernelType
    setGeneratorKernelMetaInfo(jI.kI.kInstPref);

    if (MR == 1) {
        // Generate kernel for GEMV(when MR == 1)
        AOCL_MEMORY_TAG mtag_b = (jI.kI).mtag_b;

        // We will be generating NR kernels, each having the main loop
        // and having the specific fringe case implemented.
        numKernelVariants = NR;
        kernelCodeBlocks.resize(numKernelVariants);

        utils::gemvM1GeneratorParams params(
            0, 0, 0, 0, mtag_b, true, true, true, true,
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

        for (int i = 0; i < NR; i += 1) {
            params.N_LEFT  = i;
            params.nfringe = (i != 0);

            void* codeBuffer = utils::jitHelperUtils::allocateJitMemory(
                utils::JIT_KERNEL_SIZE);
            if (codeBuffer == nullptr) {
                err = dlp::jit::jitGeneratorError::errorAllocatingMemory;
                goto cleanup;
            }
            kernelCodeBlocks[i] = codeBuffer;

            // Handle different kernel types based on instruction preference.
            switch (kType) {
                case utils::kernelInstrType::avx2_ymm_16_reg: {
                    codegen::jitF32GEMVM1<
                        utils::kernelInstrType::avx2_ymm_16_reg>
                        base(codeBuffer, utils::JIT_KERNEL_SIZE);
                    err = base.generateKernel(params);
                    break;
                }
                case utils::kernelInstrType::avx512_ymm_32_reg: {
                    codegen::jitF32GEMVM1<
                        utils::kernelInstrType::avx512_ymm_32_reg>
                        base(codeBuffer, utils::JIT_KERNEL_SIZE);
                    err = base.generateKernel(params);
                    break;
                }
                case utils::kernelInstrType::avx512_zmm_32_reg: {
                    codegen::jitF32GEMVM1<
                        utils::kernelInstrType::avx512_zmm_32_reg>
                        base(codeBuffer, utils::JIT_KERNEL_SIZE);
                    err = base.generateKernel(params);
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
#ifdef DLP_DUMP_JIT_CODE
            int n_left_suf = (i != 0) ? i : params.NR;
            // The file naming is as such : jit_gemv_m1_kernel.
            utils::jitHelperUtils::dump_jit_code(
                kernelCodeBlocks[i], utils::JIT_KERNEL_SIZE,
                "jit_gemv_m1_kernel", 1, n_left_suf);
#endif
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

        // Initializing with default values.
        utils::gemvN1GeneratorParams params(
            0, 0, false, false, false, false,
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

        for (int m_left = 0; m_left < MR; m_left++) {
            params.M_LEFT  = m_left;
            params.mfringe = (m_left != 0); // The first two kernels that we
                                            // generate are main kernels
            for (int j = 0; j < 4; j++) {
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

                void* codeBuffer = kernelCodeBlocks[m_left * 4 + j];
                // Allocate executable memory

                codeBuffer = utils::jitHelperUtils::allocateJitMemory(
                    utils::JIT_KERNEL_SIZE);
                if (codeBuffer == nullptr) {
                    err = dlp::jit::jitGeneratorError::errorAllocatingMemory;
                    goto cleanup;
                }

                kernelCodeBlocks[m_left * 4 + j] = codeBuffer;

                // Handle different kernel types based on instruction
                // preference.
                switch (kType) {
                    case utils::kernelInstrType::avx512_zmm_32_reg: {
                        // Create a new instance of jitAVX512GEMVN1 with the
                        // code buffer and size
                        codegen::jitF32GEMVN1<
                            utils::kernelInstrType::avx512_zmm_32_reg>
                            base(codeBuffer, utils::JIT_KERNEL_SIZE);

                        err = base.generateKernel(params);
                        break;
                    }
                    case utils::kernelInstrType::avx512_ymm_32_reg: {
                        codegen::jitF32GEMVN1<
                            utils::kernelInstrType::avx512_ymm_32_reg>
                            base(codeBuffer, utils::JIT_KERNEL_SIZE);
                        err = base.generateKernel(params);
                        break;
                    }
                    case utils::kernelInstrType::avx2_ymm_16_reg: {
                        codegen::jitF32GEMVN1<
                            utils::kernelInstrType::avx2_ymm_16_reg>
                            base(codeBuffer, utils::JIT_KERNEL_SIZE);
                        err = base.generateKernel(params);
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

#ifdef DLP_DUMP_JIT_CODE
                int m_left_suf = (m_left != 0) ? m_left : params.MR;
                // The file naming is as such : jit_gemv_n1_kernels_MR_idx.
                // The idx represents what configuration was used to generate
                // the kernel.
                utils::jitHelperUtils::dump_jit_code(
                    kernelCodeBlocks[m_left * 4 + j], utils::JIT_KERNEL_SIZE,
                    "jit_gemv_n1_kernel", m_left_suf, j);
#endif
            }
        }
    } else {
        // ToDo : This variable is supposed to hold the details of number of
        // blocks of N to be processed in each iteration. At a later stage, this
        // can be used to update the value of pack kernel block size, if we
        // support multiple pack kernel block sizes. Currently this is needed
        // to make sure of using the correct block sizes for each kernels,
        // given the restrictions in the available pack kernels.
        int processBlockSize = getProcessBlockSize();
#if 0
        // The idea is to generate all kernels with multiple of nElemsPerReg
        // and then use the mask to handle in-between cases.

        // This approach is more efficient for the cases where n < NR cases
        //  where you can operate the entire "<NR" region in one go.

        // These kernels can be used currently for the cases where reordering is not done.

        // To-Do: Design pack kernels to support this approach.
        numNRVariants      = (NR / numElemsPerReg) * 2;
#else
        // Here, we only generate kernels with multiples of numElemsPerReg
        // and then one kernel to handle "< numElemsPerReg" cases.
        // Here, the problem will be divided first by NR and the fringe will be
        // divided further into two regions. One for "multiples of
        // numElemsPerReg" and the other for "< numElemsPerReg" cases.

        // This approach works well with the current reordering strategy but is
        // inefficient for the cases where n < NR cases especially with "lt16"
        // fringe being taken.
        numNRVariants = (processBlockSize / numElemsPerReg) + 1;
#endif
        numMRVariants     = MR;
        numKernelVariants = numMRVariants * numNRVariants;

        kernelCodeBlocks.resize(numKernelVariants);

        // Initializing with default values.
        utils::generatorParams params(0, 0, (jI.kI).k_unroll, false, false,
                                      (jI.kI).alphaScalingType,
                                      (jI.kI).betaScalingType, kType);

        for (std::size_t ii = 0; ii < (jI.kI).kOpsArrSize; ++ii) {
            // Copy the kernelOps from the kernelInfo to params
            params.kernelOps.push_back((jI.kI).kOpsArr[ii]);
        }

        // Generate all kernels for the given MR and NR
        for (int mr = 0; mr < numMRVariants; mr++) {
            for (int nr = 0; nr < numNRVariants; nr++) {
                params.MR    = mr == 0 ? MR : mr;
                params.mLoop = mr == 0;
#if 0 // case where fringe is handled all at once
                params.NR        = (nr/2) * numElemsPerReg;
                params.useMask   = (nr % 2) == 0;
#else // current approach where fringe is handled in two parts
                params.NR      = nr * numElemsPerReg;
                params.useMask = (nr == 0);
#endif
                void* codeBuffer = kernelCodeBlocks[mr * numNRVariants + nr];
                // Allocate executable memory
                codeBuffer = utils::jitHelperUtils::allocateJitMemory(
                    utils::JIT_KERNEL_SIZE);
                if (codeBuffer == nullptr) {
                    err = dlp::jit::jitGeneratorError::errorAllocatingMemory;
                    goto cleanup;
                }
                kernelCodeBlocks[mr * numNRVariants + nr] = codeBuffer;

                // Architecture specific dispatch happens here.
                switch (kType) {
                    case utils::kernelInstrType::avx512_zmm_32_reg: {
                        // Generate the kernel for AVX512 ZMM 32 register
                        GEMMcodeGenerator::jitGEMMF32<
                            utils::kernelInstrType::avx512_zmm_32_reg>
                            base(codeBuffer, utils::JIT_KERNEL_SIZE);
                        err = base.generateKernel(params);
                        break;
                    }
                    case utils::kernelInstrType::avx512_ymm_32_reg: {
                        // Generate the kernel for AVX512 YMM 32 register
                        GEMMcodeGenerator::jitGEMMF32<
                            utils::kernelInstrType::avx512_ymm_32_reg>
                            base(codeBuffer, utils::JIT_KERNEL_SIZE);
                        err = base.generateKernel(params);
                        break;
                    }
                    case utils::kernelInstrType::avx2_ymm_16_reg: {
                        // Generate the kernel for AVX2 YMM 16 register
                        GEMMcodeGenerator::jitGEMMF32<
                            utils::kernelInstrType::avx2_ymm_16_reg>
                            base(codeBuffer, utils::JIT_KERNEL_SIZE);
                        err = base.generateKernel(params);
                        break;
                    }
                    default:
                        err = dlp::jit::jitGeneratorError::error;
                        break;
                }
                if (err != dlp::jit::jitGeneratorError::success) {
                    goto cleanup;
                }
#ifdef DLP_DUMP_JIT_CODE
                utils::jitHelperUtils::dump_jit_code(
                    kernelCodeBlocks[mr * numNRVariants + nr],
                    utils::JIT_KERNEL_SIZE, "jit_kernel", params.MR, params.NR);
#endif
            }
        }
    }

    return dlp::jit::jitGeneratorError::success;

cleanup:
    // Free the memory allocated for the kernel code blocks if
    // allocation fails or if the kernel generation fails
    for (auto& codeBlock : kernelCodeBlocks) {
        utils::jitHelperUtils::deallocateJitMemory(codeBlock,
                                                   utils::JIT_KERNEL_SIZE);
    }
    return err;
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
                    for (int i = 0; i < 8; i++) {
                        params->nmask_avx2[i] = (i < partial_elements) ? -1 : 0;
                    }
                    break;
                }
                default:
                    break;
            }
        }

        // Deploy the associated kernel
        int kernel_idx = params->n_left;

        utils::jit_gemv_m1_kernel kernel =
            reinterpret_cast<utils::jit_gemv_m1_kernel>(
                kernelCodeBlocks[kernel_idx]);
        kernel(params);

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
            for (int i = 0; i < m_iter_left; i++) {
                params->mmask_avx2[i] = -1;
            }
            for (int i = 0; i < k_iter_left; i++) {
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

    } else {
        auto params = static_cast<dlp::kernels::gemmParams*>(_params);

        int processBlockSize = getProcessBlockSize();
        // Since JR loop is in framework, the 'n' dimension passed to this
        // function is always <= NR.
        int mFullPieces    = params->m / MR;
        int mPartialPieces = params->m % MR;

        params->kIter = params->k / K_UNROLL;
        params->kLeft = params->k % K_UNROLL;

        float* aPtr = static_cast<float*>(params->a);
        float* bPtr = static_cast<float*>(params->b);
        float* cPtr = static_cast<float*>(params->c);
        // Initialize pointers to A and B
        float* c_jr = cPtr;
        float* c_ir = cPtr;

        md_t n = params->n;
        md_t m = params->m;
        md_t k = params->k;

        md_t og_post_op_c_i = (params->kernelOpsAttr).post_op_c_i;

        // OVERVIEW:
        // This unified approach works for all kernel types (ZMM, YMM, AVX2).
        // The key insight is that all architectures follow the same pattern:
        // process elements in blocks, decompose fringe blocks into complete
        // registers + remainder, then execute appropriate kernels.
        //
        // EXECUTION LOGIC:
        // 1. Process elements in chunks of 'processBlockSize':
        //    - ZMM: processBlockSize = 64 (full NR), numElemsPerReg = 16
        //    - YMM: processBlockSize = 32 (NR/2), numElemsPerReg = 8
        //
        // 2. For each chunk, decompose into complete SIMD registers +
        // remainder:
        //    - Complete registers: Use kernel_idx = nFullpieces (no masking)
        //    - Remainder elements: Use kernel_idx = 0 (mask kernel with
        //    masking)
        //
        // 3. Execute kernels in sequence: complete registers first, then
        // remainder
        //
        // EXAMPLE WALKTHROUGH (n=15 with YMM):
        // - processBlockSize=32, numElemsPerReg=8
        // - Iteration 1: nBlockSize = min(15, 32) = 15
        //   - nFullpieces = 15/8 = 1  -> Execute kernel_idx=1 for 8 elements
        //   (no mask)
        //   - nRemainder = 15%8 = 7   -> Execute kernel_idx=0 for 7 elements
        //   (with mask)
        //   - Total processed: 8 + 7 = 15, n becomes 0, loop exits
        //
        // EXAMPLE WALKTHROUGH (n=95 with YMM):
        // - Iteration 1: nBlockSize = min(95, 32) = 32
        //   - nFullpieces = 32/8 = 4  -> Execute kernel_idx=4 for 32 elements
        //   (no mask)
        //   - nRemainder = 32%8 = 0   -> No remainder processing
        //   - Processed: 32, n becomes 63
        // - Iteration 2: nBlockSize = min(63, 32) = 32 -> Process 32 more
        // elements
        //   - Processed: 32, n becomes 31
        // - Iteration 3: nBlockSize = min(31, 32) = 31
        //   - nFullpieces = 31/8 = 3  -> Execute kernel_idx=3 for 24 elements
        //   (no mask)
        //   - nRemainder = 31%8 = 7   -> Execute kernel_idx=0 for 7 elements
        //   (with mask)
        //   - Processed: 24 + 7 = 31, n becomes 0, loop exits

        while (n > 0) {
            int nBlockSize  = (n >= processBlockSize) ? processBlockSize : n;
            int nFullpieces = nBlockSize / numElemsPerReg;
            int nRemainder  = nBlockSize % numElemsPerReg;

            // Process complete registers first (if any)
            if (nFullpieces > 0) {
                int elementsToProcess = nFullpieces * numElemsPerReg;

                params->a = aPtr;
                params->c = c_jr;
                params->n = elementsToProcess;

                int kernel_n_idx = nFullpieces;
                if (params->m >= MR) {
                    params->mIter           = mFullPieces;
                    int               m_idx = 0;
                    utils::jit_kernel kernel =
                        reinterpret_cast<utils::jit_kernel>(
                            kernelCodeBlocks[m_idx * numNRVariants
                                             + kernel_n_idx]);
                    kernel(params);
                }

                if (mPartialPieces) {
                    (params->a) =
                        (float*)(params->a) + mFullPieces * params->psA;
                    (params->c) =
                        (float*)(params->c) + mFullPieces * MR * params->rsC;
                    int               m_idx = mPartialPieces;
                    utils::jit_kernel kernel =
                        reinterpret_cast<utils::jit_kernel>(
                            kernelCodeBlocks[m_idx * numNRVariants
                                             + kernel_n_idx]);
                    kernel(params);
                }

                params->b = (float*)(params->b) + elementsToProcess;
                c_jr      = (float*)(c_jr) + elementsToProcess * params->csC;
                n -= elementsToProcess;
                (params->kernelOpsAttr).post_op_c_j += elementsToProcess;
                // The following line is necessary to ensure a subtle bug
                // does not occur. Unlike the classic kernels where the
                // kernelOpsAttr is passed by value to the fringe kernels,
                // here it is kind of passed by reference, since the params
                // ptr is the only kernel argument (inside which is the
                // kernelOpsAttr). Since the while(n) loop can execute
                // multiple times, any state variable, like post_op_c_i, if
                // modified inside kernel, it needs to be reverted.
                (params->kernelOpsAttr).post_op_c_i = og_post_op_c_i;
            }

            // Process remainder with mask (if any)
            if (nRemainder > 0) {
                params->a = aPtr;
                params->c = c_jr;
                params->n = nRemainder;

                // Mask calculation (architecture-specific)
                if (kType == utils::kernelInstrType::avx512_zmm_32_reg) {
                    params->maskF32 = 0xFFFF >> (numElemsPerReg - nRemainder);
                } else if (kType == utils::kernelInstrType::avx512_ymm_32_reg) {
                    params->maskF32_8 = 0xFF >> (numElemsPerReg - nRemainder);
                } else if (kType == utils::kernelInstrType::avx2_ymm_16_reg) {
                    for (int i = 0; i < 8; i++) {
                        params->maskArray[i] = (i < nRemainder) ? 0xFFFFFFFF
                                                                : 0;
                    }
                }

                // INLINE: Kernel execution with mask
                int kernel_n_idx = 0; // Always use mask kernel for nRemainder
                if (params->m >= MR) {
                    params->mIter           = mFullPieces;
                    int               m_idx = 0;
                    utils::jit_kernel kernel =
                        reinterpret_cast<utils::jit_kernel>(
                            kernelCodeBlocks[m_idx * numNRVariants
                                             + kernel_n_idx]);
                    kernel(params);
                }
                if (mPartialPieces) {
                    (params->a) =
                        (float*)(params->a) + mFullPieces * params->psA;
                    (params->c) =
                        (float*)(params->c) + mFullPieces * MR * params->rsC;
                    int               m_idx = mPartialPieces;
                    utils::jit_kernel kernel =
                        reinterpret_cast<utils::jit_kernel>(
                            kernelCodeBlocks[m_idx * numNRVariants
                                             + kernel_n_idx]);
                    kernel(params);
                }

                // update pointers for next iteration
                params->b = (float*)(params->b) + nRemainder;
                c_jr      = (float*)(c_jr) + nRemainder * params->csC;
                n -= nRemainder;
                (params->kernelOpsAttr).post_op_c_j += nRemainder;
                // The following line is necessary to ensure a subtle bug
                // does not occur. Unlike the classic kernels where the
                // kernelOpsAttr is passed by value to the fringe kernels,
                // here it is kind of passed by reference, since the params
                // ptr is the only kernel argument (inside which is the
                // kernelOpsAttr). Since the while(n) loop can execute
                // multiple times, any state variable, like post_op_c_i, if
                // modified inside kernel, it needs to be reverted.
                (params->kernelOpsAttr).post_op_c_i = og_post_op_c_i;
            }
        }
    }

    return dlp::kernels::kernelError::success;
}

DLP_REGISTER_STATIC_GEMM_JIT_GENERATOR(jitAmdZenFP32, "dlp_amdzen_jit");

} // namespace amdzen::gen
