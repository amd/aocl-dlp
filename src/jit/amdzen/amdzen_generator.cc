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
#include "avx2_generator.hh"
#include "avx512_gemv.hh"
#include "avx512_generator.hh"
#include "cpu_utils/cpu_features.hh"
#include "jit_register/jit_register.hh"

namespace amdzen::gen {

jitAmdZenFP32::jitAmdZenFP32()
    : mKernelDatatypes({ dlp::kernel_frame::kernelDatatype::f32f32f32of32 })
    // TODO: Hardcoded for now, need to make it dynamic
    , mIsaFeaturesRequired{}
    , isAvx512(false)
    , isAvx2(false)
{
    isAvx512 = dlp::arch_utils::archConfigManager::getInstance()
                   .isAvx512SupportedByArch();

    // isAvx2 is only checked for and set to true if machine is guaranteed
    // to not be zen4 or zen5.
    if (!isAvx512) {
        isAvx2 = dlp::arch_utils::archConfigManager::getInstance()
                     .isAvx2Fma3SupportedByArch();
    }
}

jitAmdZenFP32::~jitAmdZenFP32()
{
    for (auto& codeBlock : kernelCodeBlocks) {
        utils::jitHelperUtils::deallocateJitMemory(codeBlock,
                                                   utils::JIT_KERNEL_SIZE);
    }
}

utils::kernelInstrType
jitAmdZenFP32::getGeneratorKernelType(
    dlp::kernel_frame::kernelInstrPreference kInstPref)
{
    utils::kernelInstrType _kType = utils::kernelInstrType::none;
    if (isAvx512) {
        // Set to using zmm register by default for zen4 machines.
        _kType = utils::kernelInstrType::avx512_zmm_32_reg;
        if (kInstPref
            == dlp::kernel_frame::kernelInstrPreference::avx512_ymm_favour) {
            _kType = utils::kernelInstrType::avx512_ymm_32_reg;
        } else if (kInstPref
                   == dlp::kernel_frame::kernelInstrPreference::
                       avx512_xmm_favour) {
            _kType = utils::kernelInstrType::avx512_xmm_32_reg;
        } else if (kInstPref
                   == dlp::kernel_frame::kernelInstrPreference::
                       avx2_ymm_favour) {
            // TODO: This is inconsistent, and ideally should have been set to
            // avx2_ymm_16_reg. However the existing logic in classic lpgemm
            // prevents us from doing that (i.e. MR,NR = 6,64 even when zen3
            // arch is enabled). Need to revisit this later.
            _kType = utils::kernelInstrType::avx512_ymm_32_reg;
        }
    } else if (isAvx2) {
        // Set to using ymm register by default for zen machines.
        _kType = utils::kernelInstrType::avx2_ymm_16_reg;
        if (kInstPref
            == dlp::kernel_frame::kernelInstrPreference::avx2_xmm_favour) {
            _kType = utils::kernelInstrType::avx2_xmm_16_reg;
        }
    }

    return _kType;
}

dlp::jit::jitGeneratorError
jitAmdZenFP32::generateAllKernels(const dlp::jit::jitGeneratorContext& jI)
{

    dlp::jit::jitGeneratorError err = dlp::jit::jitGeneratorError::error;

    MR                 = (jI.kI).mr;
    NR                 = (jI.kI).nr;
    KC                 = (jI.kI).kc;
    K_UNROLL           = (jI.kI).k_unroll;
    int numElemsPerReg = 0; // RegBytes / this->sizeofType<accumType>();

    if (isAvx512) {
        numElemsPerReg = 16; // AVX512 SIMD width
    } else if (isAvx2) {
        numElemsPerReg = 8; // AVX2 SIMD width
    } else {
        // Unsupported architecture
        return dlp::jit::jitGeneratorError::notSupported;
    }

    // Hardcoding the FP32 kernel datatype for now
    const dlp::kernel_frame::kernelDatatype kdt =
        dlp::kernel_frame::kernelDatatype::f32f32f32of32;

    // Convert kernelInstrPreference to kernelType
    utils::kernelInstrType _kType = getGeneratorKernelType(jI.kI.kInstPref);

    if (MR == 1) {
        // Generate kernel for GEMV(when MR == 1)
        AOCL_MEMORY_TAG mtag_b = (jI.kI).mtag_b;

        // We generate only 3 kernels for now, with the following combinations:
        // kloop = true, kfringe = true
        // Index 0 : nloop = false, nfringe = true
        // Index 1 : nloop = true, nfringe = false
        // Index 2 : nloop = true, nfringe = true
        numKernelVariants = 3;
        kernelCodeBlocks.resize(numKernelVariants);

        utils::gemvM1GeneratorParams params(
            0, 0, mtag_b, true, true, true, true,
            dlp::kernel_frame::storageFormat::rowMajor,
            (jI.kI).alphaScalingType, (jI.kI).betaScalingType, _kType);

        params.NR      = NR;
        params.KC      = KC;
        params.nloop   = true;
        params.kloop   = true;
        params.nfringe = true;
        params.kfringe = true;

        for (int i = 1; i <= 3; i += 1) {
            params.nfringe = (i % 2) ? true : false;
            params.nloop   = (i > 1) ? true : false;

            void* codeBuffer = utils::jitHelperUtils::allocateJitMemory(
                utils::JIT_KERNEL_SIZE);
            if (codeBuffer == nullptr) {
                err = dlp::jit::jitGeneratorError::errorAllocatingMemory;
                goto cleanup;
            }
            kernelCodeBlocks[i - 1] = codeBuffer;

            // Architecture specific dispatch happens here.
            if (isAvx512) {
                // Create a new instance of jitAVX512GEMVN1 with the code buffer
                // and size
                avx512gen::jitAVX512GEMVM1 base(codeBuffer,
                                                utils::JIT_KERNEL_SIZE);

                err = base.generateKernel<kdt>(params);
                if (err != dlp::jit::jitGeneratorError::success) {
                    goto cleanup;
                }
            } else if (isAvx2) { // AVX2 generation not supported yet
                return dlp::jit::jitGeneratorError::error;
            } else {
                return dlp::jit::jitGeneratorError::error;
            }
#ifdef DLP_JIT_DUMP_CODE
            // The file naming is as such : jit_gemv_m1_kernel.
            utils::jitHelperUtils::dump_jit_code(
                kernelCodeBlocks[i - 1], utils::JIT_KERNEL_SIZE,
                "jit_gemv_m1_kernel", params.NR, i);
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
            0, 0, 0, false, false, false, false,
            dlp::kernel_frame::storageFormat::rowMajor,
            (jI.kI).alphaScalingType, (jI.kI).betaScalingType, _kType);

        params.MR      = MR;
        params.MR_LEFT = 0;
        params.mloop   = true;
        params.kloop   = true;
        params.mfringe = false;
        params.kfringe = true;

        for (int m_left = 0; m_left < MR; m_left++) {
            params.M_LEFT  = m_left;
            params.mfringe = (m_left != 0); // The first two kernels that we
                                            // generate are main kernels
            for (int j = 0; j < 4; j++) {
                // We generate 4 kernels for each MR_LEFT, and index them as
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

                // Architecture specific dispatch happens here.
                if (isAvx512) {
                    // Create a new instance of jitAVX512GEMVN1 with the code
                    // buffer and size
                    avx512gen::jitAVX512GEMVN1 base(codeBuffer,
                                                    utils::JIT_KERNEL_SIZE);

                    err = base.generateKernel<kdt>(params);
                    if (err != dlp::jit::jitGeneratorError::success) {
                        goto cleanup;
                    }
                } else if (isAvx2) { // AVX2 generation not supported yet
                    return dlp::jit::jitGeneratorError::error;
                } else {
                    return dlp::jit::jitGeneratorError::error;
                }

#ifdef DLP_DUMP_JIT_CODE
                // The file naming is as such : jit_gemv_n1_kernels_MR_idx.
                // The idx represents what configuration was used to generate
                // the kernel.
                utils::jitHelperUtils::dump_jit_code(
                    kernelCodeBlocks[m_left * 4 + j], utils::JIT_KERNEL_SIZE,
                    "jit_gemv_n1_kernel", params.MR, m_left * 4 + j);
#endif
            }
        }
    } else {
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
        numNRVariants = (NR / numElemsPerReg) + 1;
#endif
        numMRVariants     = MR;
        numKernelVariants = numMRVariants * numNRVariants;

        kernelCodeBlocks.resize(numKernelVariants);

        // Initializing with default values.
        utils::generatorParams params(0, 0, (jI.kI).k_unroll, false, false,
                                      (jI.kI).alphaScalingType,
                                      (jI.kI).betaScalingType, _kType);

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
                if (isAvx512) {
                    // Create a new instance of jitAVX512 with the code buffer
                    // and size
                    avx512gen::jitAVX512 base(codeBuffer,
                                              utils::JIT_KERNEL_SIZE);

                    err = base.generateKernel<kdt>(params);
                    if (err != dlp::jit::jitGeneratorError::success) {
                        goto cleanup;
                    }
                } else if (isAvx2) {
                    // Create a new instance of jitAVX2 with the code buffer and
                    // size
                    avx2gen::jitAVX2 base(codeBuffer, utils::JIT_KERNEL_SIZE);

                    err = base.generateKernel<kdt>(params);
                    if (err != dlp::jit::jitGeneratorError::success) {
                        goto cleanup;
                    }
                } else {
                    return dlp::jit::jitGeneratorError::error;
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
    if (MR == 1) {
        auto params = static_cast<dlp::kernels::gemvM1Params*>(_params);
        // Setting the remaining values of gemvM1Params(that are not set as part
        // of it's parameterized constructor)
        params->n_iter = params->n / NR;
        params->n_left = params->n % NR;
        params->k_iter = params->k / KC;
        params->k_left = params->k % KC;

        // NOTE :
        // The parameterized constructor sets the masks to 0 by default
        // Based on the fringe case, we have to set the appropriate mask
        // to it's value. Again, the JIT generator for MR == 1 is written
        // specific to the architecture's kernel dimension(64), thereby
        // needing 4 masks. The generator would be further generalized in
        // the future, similar to how the backend support exists for NR == 1

        // All masks are intially set to 0x111...1
        params->n0_mask = 0xFFFF;
        params->n1_mask = 0xFFFF;
        params->n2_mask = 0xFFFF;
        params->n3_mask = 0xFFFF;

        uint16_t nmask = 0xFFFF >> (16 - ((params->n_left) % 16));

        if (params->n_left >= 48) {
            params->n3_mask = nmask;
        } else if (params->n_left >= 32) {
            params->n3_mask = 0;
            params->n2_mask = nmask;
        } else if (params->n_left >= 16) {
            params->n3_mask = 0;
            params->n2_mask = 0;
            params->n1_mask = nmask;
        } else {
            params->n3_mask = 0;
            params->n2_mask = 0;
            params->n1_mask = 0;
            params->n0_mask = nmask;
        }

        params->k_iter_sub_iter = KC / 4;
        params->k_iter_sub_left = KC % 4;
        params->k_left_sub_iter = (params->k_left) / 4;
        params->k_left_sub_left = (params->k_left) % 4;

        // Deploy the associated kernel
        int  kernel_idx = 0;
        bool nloop      = (params->n_iter) ? true : false;
        bool nleft      = (params->n_left) ? true : false;
        kernel_idx += 2 * nloop + nleft;
        utils::jit_gemv_m1_kernel kernel =
            reinterpret_cast<utils::jit_gemv_m1_kernel>(
                kernelCodeBlocks[kernel_idx - 1]);
        kernel(params);

    } else if (NR == 1) {
        auto params = static_cast<dlp::kernels::gemvN1Params*>(_params);
        // Setting the remaining values of gemvN1Params(that are not set as part
        // of it's parameterized constructor)
        // NOTE : The associated NR value for the generated kernels is 16.
        //        The mdulo operations are added when calcuating the masks, to
        //        make sure we use the same code for generic NR values in
        //        future.
        params->m_iter = params->m / MR;
        params->m_left = params->m % MR;
        params->k_iter = params->k / 16;
        params->k_left = params->k % 16;
        params->m_mask = 0xFFFF >> (16 - (params->m_left % 16));
        params->k_mask = 0xFFFF >> (16 - (params->k_left) % 16);
        params->mr_mask =
            0xFFFF
            >> (16
                - ((MR > 16) ? (MR % 16) : 0)); // TODO: Update generator params
                                                // to have this as a parameter

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
        // Since JR loop is in framework, the 'n' dimension passed to this
        // function is always <= NR.
        int mFullPieces    = params->m / MR;
        int mPartialPieces = params->m % MR;

        params->kIter = params->k / K_UNROLL;
        params->kLeft = params->k % K_UNROLL;

        int numElemsPerReg = 0; // RegBytes / this->sizeofType<accumType>();

        if (isAvx512) {
            numElemsPerReg = 16; // AVX512 SIMD width
        } else if (isAvx2) {
            numElemsPerReg = 8; // AVX2 SIMD width
        } else {
            // Unsupported architecture
            // NOTE : As part of kernel generation, we have already checked
            //        for the architecture support. So, this should not happen.
            //        This is purely defensive programming.
            return dlp::kernels::kernelError::error;
        }

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
        while (n) {
            int n_idx = n / numElemsPerReg;
            int nElemsProcessing =
                dlp_max(n_idx * numElemsPerReg, n % numElemsPerReg);

            params->a = aPtr;
            params->c = c_jr;
            if (isAvx512) {
                // maskF32 would be used for AVX512 fringe handling
                params->maskF32 = 0xFFFF >> (numElemsPerReg - nElemsProcessing);
            } else if (isAvx2) {
                // Array would be used for AVX2 fringe handling
                for (int i = 0; i < 8; i++) {
                    params->maskArray[i] = (i < nElemsProcessing) ? 0xFFFFFFFF
                                                                  : 0;
                }
            }
            if (params->m >= MR) {
                params->mIter = mFullPieces;
                int m_idx     = 0;
                params->n     = nElemsProcessing;

                utils::jit_kernel kernel = reinterpret_cast<utils::jit_kernel>(
                    kernelCodeBlocks[m_idx * numNRVariants + n_idx]);
                kernel(params);
            }
            if (mPartialPieces) {
                (params->a) = (float*)(params->a) + mFullPieces * params->psA;
                (params->c) =
                    (float*)(params->c) + mFullPieces * MR * params->rsC;
                int m_idx = mPartialPieces;
                {
                    utils::jit_kernel kernel =
                        reinterpret_cast<utils::jit_kernel>(
                            kernelCodeBlocks[m_idx * numNRVariants + n_idx]);
                    kernel(params);
                }
            }
            // move b and c pointers
            params->b = (float*)(params->b) + n_idx * numElemsPerReg;
            c_jr      = (float*)(c_jr) + n_idx * numElemsPerReg;
            n -= nElemsProcessing;

            // This is the case where a fringe is encountered, with the fringe
            // itself composed of a part which is a multiple of numElemsPerReg
            // and a part which is < numElemsPerReg.
            (params->kernelOpsAttr).post_op_c_j += nElemsProcessing;

            // The following line is necessary to ensure a subtle bug does not
            // occur. Unlike the classic kernels where the kernelOpsAttr is
            // passed by value to the fringe kernels, here it is kind of passed
            // by reference, since the params ptr is the only kernel argument
            // (inside which is the kernelOpsAttr). Since the while(n) loop can
            // execute multiple times, any state variable, like post_op_c_i, if
            // modified inside kernel, it needs to be reverted.
            (params->kernelOpsAttr).post_op_c_i = og_post_op_c_i;
        }
    }

    return dlp::kernels::kernelError::success;
}

DLP_REGISTER_STATIC_GEMM_JIT_GENERATOR(jitAmdZenFP32, "dlp_amdzen_jit");

} // namespace amdzen::gen
