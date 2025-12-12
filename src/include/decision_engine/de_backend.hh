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

#include <memory>
#include <optional>

#include "bindings/c_wrappers/capi_kernel_frame_wrappers.h"
#include "de_backend_utils.hh"
#include "de_input.hh"
#include "kernel_frame/kernel_frame_base.hh"

namespace dlp::de {

static const kernel_frame::kernelInfo INVALID_KERNEL_INFO{
    0,
    0,
    0,
    0,
    0,
    0,
    kernel_frame::scalingType::generic,
    kernel_frame::scalingType::generic,
    AOCL_MEMORY_TAG::UNPACKED,
    AOCL_MEMORY_TAG::UNPACKED,
    false,
    false,
    nullptr,
    0,
    false,
    kernel_frame::kernelInstrPreference::none,
    0
};

class iDEBackend
{
  public:
    virtual ~iDEBackend() = default;
    virtual std::optional<dlp::kernel_frame::kernelInfo> getKernelInfoForInput(
        iDEInput* in) = 0;

    virtual dlp::kernel_frame::kernelInfo getGemmKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype k_dtype,
        md_t                              m,
        md_t                              n,
        md_t                              k,
        md_t                              rs_a,
        md_t                              cs_a,
        md_t                              rs_b,
        md_t                              cs_b,
        md_t                              rs_c,
        md_t                              cs_c,
        void*                             alpha,
        void*                             beta,
        AOCL_MEMORY_TAG                   mtag_a,
        AOCL_MEMORY_TAG                   mtag_b,
        lpgemm_post_op*                   metadata,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale,
        bool                              rerouted_from_other_backend) = 0;

    virtual dlp::kernel_frame::kernelInfo getGemvKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype k_dtype,
        md_t                              m,
        md_t                              n,
        md_t                              k,
        md_t                              rs_a,
        md_t                              cs_a,
        md_t                              rs_b,
        md_t                              cs_b,
        md_t                              rs_c,
        md_t                              cs_c,
        void*                             alpha,
        void*                             beta,
        AOCL_MEMORY_TAG                   mtag_a,
        AOCL_MEMORY_TAG                   mtag_b,
        lpgemm_post_op*                   metadata,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale,
        bool                              rerouted_from_other_backend) = 0;
};

class gemmF32DEBackend : public iDEBackend
{
    bool                                isAvx512;
    bool                                isAvx2;
    int32_t                             numRegisters;
    int32_t                             numVectorMaskRegisters;
    kernel_frame::kernelInstrPreference eKernelInstPref;
    bool                                canGenerateKernelInfo;

  public:
    gemmF32DEBackend();
    ~gemmF32DEBackend()                                  = default;
    gemmF32DEBackend(const gemmF32DEBackend&)            = delete;
    gemmF32DEBackend(gemmF32DEBackend&&)                 = delete;
    gemmF32DEBackend& operator=(const gemmF32DEBackend&) = delete;
    gemmF32DEBackend& operator=(gemmF32DEBackend&&)      = delete;

    std::optional<dlp::kernel_frame::kernelInfo> getKernelInfoForInput(
        iDEInput* in) override;

    [[gnu::always_inline]]
    dlp::kernel_frame::kernelInfo getGemvKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype k_dtype,
        md_t                              m,
        md_t                              n,
        md_t                              k,
        md_t                              rs_a,
        md_t                              cs_a,
        md_t                              rs_b,
        md_t                              cs_b,
        md_t                              rs_c,
        md_t                              cs_c,
        void*                             alpha,
        void*                             beta,
        AOCL_MEMORY_TAG                   mtag_a,
        AOCL_MEMORY_TAG                   mtag_b,
        lpgemm_post_op*                   metadata,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale,
        [[maybe_unused]] bool rerouted_from_other_backend) override final
    {
        if (!canGenerateKernelInfo) {
            return INVALID_KERNEL_INFO;
        }

        kernel_frame::scalingType alphaScalingType;
        kernel_frame::scalingType betaScalingType;
        std::tie(alphaScalingType, betaScalingType) =
            gemmDEBackendUtils::getScalingTypes<float>(alpha, beta, k, kc_hint);

        md_t mr              = mr_hint;
        md_t nr              = nr_hint;
        md_t k_unroll        = 1;
        md_t kc              = kc_hint;
        md_t prefetch_c_dist = 0; // Setting this to 0, until we use it.
        bool anyKOpsOrder    = false;

        // Set the kernel instruction preference based on the CPU features.
        // NOTE : This could be overridden by the user in future
        //        Ex : Wanting to run AVX2 kernels on Zen4, based on
        //             AOCL_ENABLE_INSTRUCTIONS
        kernel_frame::kernelInstrPreference kInstPref = eKernelInstPref;

        // In case the environment variable was not set at all, resort to
        // setting it based on the native hardware support.
        if (kInstPref == kernel_frame::kernelInstrPreference::none) {
            if (isAvx512) {
                kInstPref =
                    kernel_frame::kernelInstrPreference::avx512_zmm_favour;
            } else if (isAvx2) {
                kInstPref =
                    kernel_frame::kernelInstrPreference::avx2_ymm_favour;
            } else {
                // This is an invalid case, disable jit kernel generation.
                return INVALID_KERNEL_INFO;
            }
        }

        if (n == 1) {
            if (isAvx2) {
                mr       = (kInstPref
                      == kernel_frame::kernelInstrPreference::avx2_ymm_favour)
                               ? 8
                               : 16;
                nr       = 1;
                k_unroll = 1; // k-unroll is 1 for GEMV N1
            } else if (isAvx512) {
                mr       = 16;
                nr       = 1;
                k_unroll = 1; // k-unroll is 1 for GEMV N1
            } else {
                return INVALID_KERNEL_INFO;
            }
        } else if (m == 1) {
            // The booleans isAvx2 and isAvx512 represent the configured
            // hardware Thus, on an AVX512 machine, if the env var is set to
            // avx2, isAvx2 would be set to true by the constructor.
            if (isAvx2) {
                mr = 1;
                // This setting is a follow-up of the hack we have defined in
                // the DE constructor. With this, we force the AVX512_256 path
                // even when the env var is set to AVX2.
                nr       = (kInstPref
                      == kernel_frame::kernelInstrPreference::avx2_ymm_favour)
                               ? 16
                               : 64;
                k_unroll = 2;
                kc       = 512; // This is hardcoded from ZEN4 context.
            } else if (isAvx512) {
                mr = 1;
                nr = 64;
                k_unroll =
                    (kInstPref
                     == kernel_frame::kernelInstrPreference::avx512_ymm_favour)
                        ? 2
                        : 4;
                kc = 512;
            } else {
                return INVALID_KERNEL_INFO;
            }
        } else {
            return INVALID_KERNEL_INFO;
        }

        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, false, false, anyKOpsOrder,
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata);
    }

    [[gnu::always_inline]]
    dlp::kernel_frame::kernelInfo getGemmKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype k_dtype,
        md_t                              m,
        md_t                              n,
        md_t                              k,
        md_t                              rs_a,
        md_t                              cs_a,
        md_t                              rs_b,
        md_t                              cs_b,
        md_t                              rs_c,
        md_t                              cs_c,
        void*                             alpha,
        void*                             beta,
        AOCL_MEMORY_TAG                   mtag_a,
        AOCL_MEMORY_TAG                   mtag_b,
        lpgemm_post_op*                   metadata,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale,
        bool rerouted_from_other_backend) override final
    {
        if (!canGenerateKernelInfo) {
            return INVALID_KERNEL_INFO;
        }

        kernel_frame::scalingType alphaScalingType;
        kernel_frame::scalingType betaScalingType;
        std::tie(alphaScalingType, betaScalingType) =
            gemmDEBackendUtils::getScalingTypes<float>(alpha, beta, k, kc_hint);

        md_t mr              = mr_hint;
        md_t nr              = nr_hint;
        md_t k_unroll        = 1;
        md_t kc              = kc_hint;
        md_t prefetch_c_dist = 0; // Setting this to 0, until we use it.
        bool anyKOpsOrder    = false;

        // Set the kernel instruction preference based on the CPU features.
        // NOTE : This could be overridden by the user in future
        //        Ex : Wanting to run AVX2 kernels on Zen4, based on
        //             AOCL_ENABLE_INSTRUCTIONS
        kernel_frame::kernelInstrPreference kInstPref = eKernelInstPref;

        // In case the environment variable was not set at all, resort to
        // setting it based on the native hardware support.
        if (kInstPref == kernel_frame::kernelInstrPreference::none) {
            if (isAvx512) {
                kInstPref =
                    kernel_frame::kernelInstrPreference::avx512_zmm_favour;
            } else if (isAvx2) {
                kInstPref =
                    kernel_frame::kernelInstrPreference::avx2_ymm_favour;
            } else {
                // This is an invalid case, disable jit kernel generation.
                return INVALID_KERNEL_INFO;
            }
        }

        bool allLtFringeKernels = false;
        // For now masked lt fringes can only be enabled on avx512 machines
        // with zmm preferred instructions. Additionally to begin with,
        //  only enabling masked lt fringes when no post-ops are involved
        // for row-major C matrix.
        if (isAvx512
            && (kInstPref
                == kernel_frame::kernelInstrPreference::avx512_zmm_favour)) {
            md_t availableMasks = numVectorMaskRegisters;
            md_t requiredMasks  = nr / 16;
            // If masks required are more than available masks, we can't
            // generate all lt fringes. Also limit the mask usage to 4.
            if ((requiredMasks <= availableMasks) && (requiredMasks < 5)) {
                allLtFringeKernels = true;
            }
        }

        bool invokeRD = false;

        // TODO: For RD inputs kMask is set from the generator and nmask is
        // calculated on the fly within the RD GEMM. Currently, when post-ops
        // module reloads the maskF32 values into the opMask, the kMask gets
        // overwritten for RD kernels while post-ops expects nMask. Until a
        // proper handshake of the mask registers is established between
        // post-ops and GEMM the RD kenrels will be disabled with post-ops.
        bool hasPostOps =
            (metadata != nullptr) && (metadata->op_code != POST_OPS_DISABLE);

        // The pack-conversion function from f32 to bf16 only supports
        // packing of matrices to row-major format. Hence Rd kernels can't
        // be used when bf16 API is rerouted to FP32.
        // using rerouted_from_other_backend to check if the DE is rerouted from
        // other backend.
        if (!hasPostOps && !(rerouted_from_other_backend)
            && ((n < 48) || (m < 16)) && (rs_b == 1) && (mtag_b == PACK)
            && (mtag_a == UNPACKED)) {
            invokeRD = true;
            k_unroll = 4; // equal to intrinsics kernel. To be tuned later.
        }
        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, allLtFringeKernels, invokeRD,
            anyKOpsOrder, kInstPref, c_downscale, k_dtype, rs_c, cs_c,
            metadata);
    }
};

class gemmBF16DEBackend : public iDEBackend
{
    bool                                isAvx512;
    bool                                isAvx2;
    bool                                isAvx512Bf16;
    kernel_frame::kernelInstrPreference eKernelInstPref;
    bool                                canGenerateKernelInfo;
    std::unique_ptr<gemmF32DEBackend>
        f32Backend; // For rerouting when AVX512BF16 is not supported

    [[gnu::always_inline]] constexpr md_t getPrefetchDistance()
    {
        // Setting this to 40, which works for ZEN5. Should we set this in
        // the DE constructor, based on the underlying arch?
        constexpr md_t prefetch_c_dist = 40;
        return prefetch_c_dist;
    }

  public:
    gemmBF16DEBackend();
    ~gemmBF16DEBackend()                                   = default;
    gemmBF16DEBackend(const gemmBF16DEBackend&)            = delete;
    gemmBF16DEBackend(gemmBF16DEBackend&&)                 = delete;
    gemmBF16DEBackend& operator=(const gemmBF16DEBackend&) = delete;
    gemmBF16DEBackend& operator=(gemmBF16DEBackend&&)      = delete;
    std::optional<dlp::kernel_frame::kernelInfo> getKernelInfoForInput(
        iDEInput* in) override;

    [[gnu::always_inline]]
    dlp::kernel_frame::kernelInfo getGemvKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype k_dtype,
        md_t                              m,
        md_t                              n,
        md_t                              k,
        md_t                              rs_a,
        md_t                              cs_a,
        md_t                              rs_b,
        md_t                              cs_b,
        md_t                              rs_c,
        md_t                              cs_c,
        void*                             alpha,
        void*                             beta,
        AOCL_MEMORY_TAG                   mtag_a,
        AOCL_MEMORY_TAG                   mtag_b,
        lpgemm_post_op*                   metadata,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale,
        [[maybe_unused]] bool rerouted_from_other_backend) override final
    {
        if (!canGenerateKernelInfo) {
            return INVALID_KERNEL_INFO;
        }

        // This rerouting currently happens only on machines without AVX512BF16
        // and AVX512 support, that still have AVX2 support.
        if (f32Backend != nullptr) {
            return f32Backend->getGemvKernelInfoForInputFastPath(
                k_dtype, m, n, k, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, alpha,
                beta, mtag_a, mtag_b, metadata, mr_hint, nr_hint, kc_hint,
                c_downscale, true);
        }

        kernel_frame::scalingType alphaScalingType;
        kernel_frame::scalingType betaScalingType;
        std::tie(alphaScalingType, betaScalingType) =
            gemmDEBackendUtils::getScalingTypes<float>(alpha, beta, k, kc_hint);

        md_t mr              = mr_hint;
        md_t nr              = nr_hint;
        md_t k_unroll        = 1;
        md_t kc              = kc_hint;
        md_t prefetch_c_dist = getPrefetchDistance();
        bool anyKOpsOrder    = false;

        // Set the kernel instruction preference based on the CPU features.
        // The DE constructor sets it to a safe value, based on the hardware
        // support.
        kernel_frame::kernelInstrPreference kInstPref = eKernelInstPref;

        if (n == 1) {
            mr       = 16;
            nr       = 1;
            k_unroll = 1; // k-unroll is 1 for GEMV N1
        } else if (m == 1) {
            nr       = 64;
            mr       = 1;
            k_unroll = 4;
            kc       = 4096;
        }

        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, false, false, anyKOpsOrder,
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata);
    }

    [[gnu::always_inline]]
    dlp::kernel_frame::kernelInfo getGemmKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype k_dtype,
        md_t                              m,
        md_t                              n,
        md_t                              k,
        md_t                              rs_a,
        md_t                              cs_a,
        md_t                              rs_b,
        md_t                              cs_b,
        md_t                              rs_c,
        md_t                              cs_c,
        void*                             alpha,
        void*                             beta,
        AOCL_MEMORY_TAG                   mtag_a,
        AOCL_MEMORY_TAG                   mtag_b,
        lpgemm_post_op*                   metadata,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale,
        [[maybe_unused]] bool rerouted_from_other_backend) override final
    {
        if (!canGenerateKernelInfo) {
            return INVALID_KERNEL_INFO;
        }

        // This rerouting currently happens only on machines without AVX512BF16
        // and AVX512 support, that still have AVX2 support.
        if (f32Backend != nullptr) {
            return f32Backend->getGemmKernelInfoForInputFastPath(
                k_dtype, m, n, k, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, alpha,
                beta, mtag_a, mtag_b, metadata, mr_hint, nr_hint, kc_hint,
                c_downscale, true);
        }

        // At this point, we know that the underlying architecture supports
        // AVX512-BF16.
        kernel_frame::scalingType alphaScalingType;
        kernel_frame::scalingType betaScalingType;
        std::tie(alphaScalingType, betaScalingType) =
            gemmDEBackendUtils::getScalingTypes<float>(alpha, beta, k, kc_hint);

        md_t mr              = mr_hint;
        md_t nr              = nr_hint;
        md_t k_unroll        = 1;
        md_t kc              = kc_hint;
        md_t prefetch_c_dist = getPrefetchDistance();
        bool anyKOpsOrder    = false;

        // Set the kernel instruction preference based on the CPU features.
        // The DE constructor sets it to a safe value, based on the hardware
        // support.
        kernel_frame::kernelInstrPreference kInstPref = eKernelInstPref;

        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, false, false, anyKOpsOrder,
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata);
    }
};

class gemmU8S8DEBackend : public iDEBackend
{
    bool                                isAvx512;
    bool                                isAvx2;
    bool                                isAvx512Vnni;
    kernel_frame::kernelInstrPreference eKernelInstPref;
    bool                                canGenerateKernelInfo;

    [[gnu::always_inline]]
    constexpr md_t getPrefetchDistance()
    {
        // Setting this to 0 for now, should be tuned based on arch
        constexpr md_t prefetch_c_dist = 0;
        return prefetch_c_dist;
    }

  public:
    gemmU8S8DEBackend();
    ~gemmU8S8DEBackend()                                   = default;
    gemmU8S8DEBackend(const gemmU8S8DEBackend&)            = delete;
    gemmU8S8DEBackend(gemmU8S8DEBackend&&)                 = delete;
    gemmU8S8DEBackend& operator=(const gemmU8S8DEBackend&) = delete;
    gemmU8S8DEBackend& operator=(gemmU8S8DEBackend&&)      = delete;

    std::optional<dlp::kernel_frame::kernelInfo> getKernelInfoForInput(
        iDEInput* in) override;

    [[gnu::always_inline]]
    dlp::kernel_frame::kernelInfo getGemvKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype k_dtype,
        md_t                              m,
        md_t                              n,
        md_t                              k,
        md_t                              rs_a,
        md_t                              cs_a,
        md_t                              rs_b,
        md_t                              cs_b,
        md_t                              rs_c,
        md_t                              cs_c,
        void*                             alpha,
        void*                             beta,
        AOCL_MEMORY_TAG                   mtag_a,
        AOCL_MEMORY_TAG                   mtag_b,
        lpgemm_post_op*                   metadata,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale,
        [[maybe_unused]] bool rerouted_from_other_backend) override final
    {
        if (!canGenerateKernelInfo) {
            return INVALID_KERNEL_INFO;
        }

        kernel_frame::scalingType alphaScalingType;
        kernel_frame::scalingType betaScalingType;
        std::tie(alphaScalingType, betaScalingType) =
            gemmDEBackendUtils::getScalingTypes<int32_t>(alpha, beta, k,
                                                         kc_hint);

        md_t mr              = mr_hint;
        md_t nr              = nr_hint;
        md_t k_unroll        = 1;
        md_t kc              = kc_hint;
        md_t prefetch_c_dist = getPrefetchDistance();
        bool anyKOpsOrder    = false;

        // Set the kernel instruction preference based on the CPU features.
        // The DE constructor sets it to a safe value, based on the hardware
        // support.
        kernel_frame::kernelInstrPreference kInstPref = eKernelInstPref;

        // In case the environment variable was not set at all, resort to
        // setting it based on the native hardware support.
        if (kInstPref == kernel_frame::kernelInstrPreference::none) {
            if (isAvx512) {
                kInstPref =
                    kernel_frame::kernelInstrPreference::avx512_zmm_favour;
            } else {
                // This is an invalid case, disable jit kernel generation.
                return INVALID_KERNEL_INFO;
            }
        } else if (kInstPref
                   != kernel_frame::kernelInstrPreference::avx512_zmm_favour) {
            kInstPref = kernel_frame::kernelInstrPreference::
                avx512_zmm_favour; // At this point we know that it is an AVX512
                                   // machine
        }

        if (n == 1) {
            mr       = 16;
            nr       = 1;
            k_unroll = 1; // k-unroll is 1 for GEMV N1
        } else {
            nr       = 64;
            mr       = 1;
            k_unroll = 4;
        }

        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, false, false, anyKOpsOrder,
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata);
    }

    [[gnu::always_inline]]
    dlp::kernel_frame::kernelInfo getGemmKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype k_dtype,
        md_t                              m,
        md_t                              n,
        md_t                              k,
        md_t                              rs_a,
        md_t                              cs_a,
        md_t                              rs_b,
        md_t                              cs_b,
        md_t                              rs_c,
        md_t                              cs_c,
        void*                             alpha,
        void*                             beta,
        AOCL_MEMORY_TAG                   mtag_a,
        AOCL_MEMORY_TAG                   mtag_b,
        lpgemm_post_op*                   metadata,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale,
        [[maybe_unused]] bool rerouted_from_other_backend) override final
    {
        if (!canGenerateKernelInfo) {
            return INVALID_KERNEL_INFO;
        }

        kernel_frame::scalingType alphaScalingType;
        kernel_frame::scalingType betaScalingType;
        std::tie(alphaScalingType, betaScalingType) =
            gemmDEBackendUtils::getScalingTypes<int32_t>(alpha, beta, k,
                                                         kc_hint);

        md_t mr              = mr_hint;
        md_t nr              = nr_hint;
        md_t k_unroll        = 1;
        md_t kc              = kc_hint;
        md_t prefetch_c_dist = getPrefetchDistance();
        bool anyKOpsOrder    = false;

        // Set the kernel instruction preference based on the CPU features.
        // The DE constructor sets it to a safe value, based on the hardware
        // support.
        kernel_frame::kernelInstrPreference kInstPref = eKernelInstPref;

        // In case the environment variable was not set at all, resort to
        // setting it based on the native hardware support.
        if (kInstPref == kernel_frame::kernelInstrPreference::none) {
            if (isAvx512) {
                kInstPref =
                    kernel_frame::kernelInstrPreference::avx512_zmm_favour;
            } else {
                // This is an invalid case, disable jit kernel generation.
                return INVALID_KERNEL_INFO;
            }
        } else if (kInstPref
                   != kernel_frame::kernelInstrPreference::avx512_zmm_favour) {
            kInstPref = kernel_frame::kernelInstrPreference::avx512_zmm_favour;
        }

        // Currently only general GEMM is supported, specific GEMM optimizations
        // will be added later
        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, false, false, anyKOpsOrder,
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata);
    }
};

class gemmS8DEBackend : public iDEBackend
{
    bool                                isAvx512;
    bool                                isAvx2;
    bool                                isAvx512Bf16;
    bool                                isAvx512Vnni;
    kernel_frame::kernelInstrPreference eKernelInstPref;
    bool                                canGenerateKernelInfo;

    [[gnu::always_inline]] constexpr md_t getPrefetchDistance()
    {
        // Setting this to 40, which works for ZEN5. Should we set this in
        // the DE constructor, based on the underlying arch?
        constexpr md_t prefetch_c_dist = 40;
        return prefetch_c_dist;
    }

  public:
    gemmS8DEBackend();
    ~gemmS8DEBackend()                                 = default;
    gemmS8DEBackend(const gemmS8DEBackend&)            = delete;
    gemmS8DEBackend(gemmS8DEBackend&&)                 = delete;
    gemmS8DEBackend& operator=(const gemmS8DEBackend&) = delete;
    gemmS8DEBackend& operator=(gemmS8DEBackend&&)      = delete;

    std::optional<dlp::kernel_frame::kernelInfo> getKernelInfoForInput(
        iDEInput* in) override;

    [[gnu::always_inline]] dlp::kernel_frame::kernelInfo
    getGemvKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype k_dtype,
        md_t                              m,
        md_t                              n,
        md_t                              k,
        md_t                              rs_a,
        md_t                              cs_a,
        md_t                              rs_b,
        md_t                              cs_b,
        md_t                              rs_c,
        md_t                              cs_c,
        void*                             alpha,
        void*                             beta,
        AOCL_MEMORY_TAG                   mtag_a,
        AOCL_MEMORY_TAG                   mtag_b,
        lpgemm_post_op*                   metadata,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale,
        [[maybe_unused]] bool rerouted_from_other_backend) override final
    {
        if (!canGenerateKernelInfo) {
            return INVALID_KERNEL_INFO;
        }

        // At this point, we know that the underlying architecture supports
        // AVX512-VNNI.
        kernel_frame::scalingType alphaScalingType;
        kernel_frame::scalingType betaScalingType;
        std::tie(alphaScalingType, betaScalingType) =
            gemmDEBackendUtils::getScalingTypes<int32_t>(alpha, beta, k,
                                                         kc_hint);

        md_t mr              = mr_hint;
        md_t nr              = nr_hint;
        md_t k_unroll        = 1;
        md_t kc              = kc_hint;
        md_t prefetch_c_dist = getPrefetchDistance();
        bool anyKOpsOrder    = false;

        // Set the kernel instruction preference based on the CPU features.
        // The DE constructor sets it to a safe value, based on the hardware
        // support.
        kernel_frame::kernelInstrPreference kInstPref = eKernelInstPref;

        // In case the environment variable was not set at all, resort to
        // setting it based on the native hardware support.
        if (kInstPref == kernel_frame::kernelInstrPreference::none) {
            if (isAvx512) {
                kInstPref =
                    kernel_frame::kernelInstrPreference::avx512_zmm_favour;
            } else {
                // This is an invalid case, disable jit kernel generation.
                return INVALID_KERNEL_INFO;
            }
        }

        if (n == 1) { // S8 GEMV N=1
            mr       = 16;
            nr       = 1;
            k_unroll = 1;
        } else { // S8 GEMV M=1
            nr       = 64;
            mr       = 1;
            k_unroll = 4;
        }

        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, false, false, anyKOpsOrder,
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata);
    }

    [[gnu::always_inline]]
    dlp::kernel_frame::kernelInfo getGemmKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype k_dtype,
        md_t                              m,
        md_t                              n,
        md_t                              k,
        md_t                              rs_a,
        md_t                              cs_a,
        md_t                              rs_b,
        md_t                              cs_b,
        md_t                              rs_c,
        md_t                              cs_c,
        void*                             alpha,
        void*                             beta,
        AOCL_MEMORY_TAG                   mtag_a,
        AOCL_MEMORY_TAG                   mtag_b,
        lpgemm_post_op*                   metadata,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale,
        [[maybe_unused]] bool rerouted_from_other_backend) override final
    {
        if (!canGenerateKernelInfo) {
            return INVALID_KERNEL_INFO;
        }

        // At this point, we know that the underlying architecture supports
        // AVX512-VNNI.
        kernel_frame::scalingType alphaScalingType;
        kernel_frame::scalingType betaScalingType;
        std::tie(alphaScalingType, betaScalingType) =
            gemmDEBackendUtils::getScalingTypes<int32_t>(alpha, beta, k,
                                                         kc_hint);

        md_t mr              = mr_hint;
        md_t nr              = nr_hint;
        md_t k_unroll        = 1;
        md_t kc              = kc_hint;
        md_t prefetch_c_dist = getPrefetchDistance();
        bool anyKOpsOrder    = false;

        // Set the kernel instruction preference based on the CPU features.
        // The DE constructor sets it to a safe value, based on the hardware
        // support.
        kernel_frame::kernelInstrPreference kInstPref = eKernelInstPref;

        // In case the environment variable was not set at all, resort to
        // setting it based on the native hardware support.
        if (kInstPref == kernel_frame::kernelInstrPreference::none) {
            if (isAvx512) {
                kInstPref =
                    kernel_frame::kernelInstrPreference::avx512_zmm_favour;
            } else {
                // This is an invalid case, disable jit kernel generation.
                return INVALID_KERNEL_INFO;
            }
        }

        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, false, false, anyKOpsOrder,
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata);
    }
};

} // namespace dlp::de
