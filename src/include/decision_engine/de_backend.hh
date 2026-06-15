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

#include <algorithm>
#include <memory>
#include <optional>

#include "alias_detection_utils.hh"
#include "bindings/c_wrappers/capi_kernel_frame_wrappers.h"
#include "classic/dlp_macros.h"
#include "de_backend_utils.hh"
#include "de_input.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "utils/float16_types.hh"

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
    AOCL_DLP_MEMORY_TAG::UNPACKED,
    AOCL_DLP_MEMORY_TAG::UNPACKED,
    false,
    false,
    nullptr,
    0,
    false,
    kernel_frame::kernelInstrPreference::none,
    0,
    false
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
        AOCL_DLP_MEMORY_TAG               mtag_a,
        AOCL_DLP_MEMORY_TAG               mtag_b,
        dlp_gemm_post_op*                 metadata,
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
        AOCL_DLP_MEMORY_TAG               mtag_a,
        AOCL_DLP_MEMORY_TAG               mtag_b,
        dlp_gemm_post_op*                 metadata,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale,
        bool                              rerouted_from_other_backend) = 0;

    virtual dlp::kernel_frame::packKernelInfo getGemmPackBInfoForInputFastPath(
        [[maybe_unused]] md_t nc,
        [[maybe_unused]] md_t kc,
        [[maybe_unused]] md_t cs_src,
        [[maybe_unused]] md_t nr_hint)
    {
        return kernel_frame::INVALID_PACK_KERNEL_INFO;
    }
};

class gemmF32DEBackend : public iDEBackend
{
    bool                                isAvx512;
    bool                                isAvx2;
    int32_t                             numRegisters;
    int32_t                             numVectorMaskRegisters;
    kernel_frame::kernelInstrPreference eKernelInstPref;
    bool                                canGenerateKernelInfo;
    bool                                canGeneratePackBKernelInfo;

  public:
    gemmF32DEBackend();
    ~gemmF32DEBackend()                                  = default;
    gemmF32DEBackend(const gemmF32DEBackend&)            = delete;
    gemmF32DEBackend(gemmF32DEBackend&&)                 = delete;
    gemmF32DEBackend& operator=(const gemmF32DEBackend&) = delete;
    gemmF32DEBackend& operator=(gemmF32DEBackend&&)      = delete;

    std::optional<dlp::kernel_frame::kernelInfo> getKernelInfoForInput(
        iDEInput* in) override;

    DLP_ALWAYS_INLINE
    dlp::kernel_frame::kernelInfo getGemvKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype k_dtype,
        md_t                              m,
        md_t                              n,
        md_t                              k,
        md_t                              rs_a,
        [[maybe_unused]] md_t             cs_a,
        [[maybe_unused]] md_t             rs_b,
        [[maybe_unused]] md_t             cs_b,
        md_t                              rs_c,
        md_t                              cs_c,
        void*                             alpha,
        void*                             beta,
        AOCL_DLP_MEMORY_TAG               mtag_a,
        AOCL_DLP_MEMORY_TAG               mtag_b,
        dlp_gemm_post_op*                 metadata,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale,
        bool rerouted_from_other_backend) override final
    {
        if (!canGenerateKernelInfo) {
            return INVALID_KERNEL_INFO;
        }

        // BF16 operations don't support AVX512_YMM (256-bit) kernel variants.
        // When BF16 backend reroutes to F32 backend on AVX512 machines and
        // AOCL_DLP_ENABLE_INSTRUCTIONS is set to avx512_ymm, we must override
        // the preference to avx512_zmm_favour since AVX512_YMM kernels are
        // not available for the rerouted BF16->F32 operations.
        if (rerouted_from_other_backend
            && eKernelInstPref
                   == kernel_frame::kernelInstrPreference::avx512_ymm_favour) {
            eKernelInstPref =
                kernel_frame::kernelInstrPreference::avx512_zmm_favour;
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
        //             AOCL_DLP_ENABLE_INSTRUCTIONS
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
                kc       = 512; // This is hardcoded from ZEN3 context.
            } else if (isAvx512) {
                mr = 1;
                nr = 64;
                k_unroll =
                    (kInstPref
                     == kernel_frame::kernelInstrPreference::avx512_ymm_favour)
                        ? 2
                        : 4;
            } else {
                return INVALID_KERNEL_INFO;
            }
        } else {
            return INVALID_KERNEL_INFO;
        }

        // L1-cache aliasing mitigation for GEMV N=1 (MR=16 on Zen4/5).
        // When rsA (in bytes) is within +-ALIAS_GUARD_BYTES (16 B) of a
        // multiple of the L1D way size (4096 B), the kernel's MR rows
        // map to the same L1 set on each k-iteration and trigger
        // conflict misses. The GEMV N=1 generator consumes this flag
        // at codegen time and emits a two-pass MR/2 k-loop body (no
        // runtime check).
        //
        // Only applies to unpacked A: a packed/reordered A is laid out
        // in a packed panel format whose row stride no longer matches
        // the user-visible `rs_a`, so the predicate would spuriously
        // fire on `rs_a == 1` (1*sizeof(float) = 4 B < 16 B guard) and
        // force the slow two-pass kernel even though packed-A is not
        // alias-vulnerable.
        const bool aliasMrSplit =
            (n == 1 && mtag_a == AOCL_DLP_MEMORY_TAG::UNPACKED)
                ? alias_detection::shouldUseMrSplit(rs_a, sizeof(float), mr)
                : false;

        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, false, false, anyKOpsOrder,
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata,
            /*skinnyN=*/false, aliasMrSplit);
    }

    DLP_ALWAYS_INLINE
    dlp::kernel_frame::kernelInfo getGemmKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype k_dtype,
        md_t                              m,
        md_t                              n,
        md_t                              k,
        md_t                              rs_a,
        [[maybe_unused]] md_t             cs_a,
        [[maybe_unused]] md_t             rs_b,
        md_t                              cs_b,
        md_t                              rs_c,
        md_t                              cs_c,
        void*                             alpha,
        void*                             beta,
        AOCL_DLP_MEMORY_TAG               mtag_a,
        AOCL_DLP_MEMORY_TAG               mtag_b,
        dlp_gemm_post_op*                 metadata,
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

        md_t mr       = mr_hint;
        md_t nr       = nr_hint;
        md_t k_unroll = 1;
        // if k == 1, send hint to generate a single kernel with IR loop outside
        // and JR loop inside within the microkernel.
        md_t kc = kc_hint;
        if ((k == 1) && (!rerouted_from_other_backend)) {
            kc = 1;
        }

        // BF16 operations don't support AVX512_YMM (256-bit) kernel variants.
        // When BF16 backend reroutes to F32 backend on AVX512 machines and
        // AOCL_DLP_ENABLE_INSTRUCTIONS is set to avx512_ymm, we must override
        // the preference to avx512_zmm_favour since AVX512_YMM kernels are
        // not available for the rerouted BF16->F32 operations.
        if (rerouted_from_other_backend
            && eKernelInstPref
                   == kernel_frame::kernelInstrPreference::avx512_ymm_favour) {
            eKernelInstPref =
                kernel_frame::kernelInstrPreference::avx512_zmm_favour;
        }

        md_t prefetch_c_dist = 0; // Setting this to 0, until we use it.
        bool anyKOpsOrder    = false;

        // Set the kernel instruction preference based on the CPU features.
        // NOTE : This could be overridden by the user in future
        //        Ex : Wanting to run AVX2 kernels on Zen4, based on
        //             AOCL_DLP_ENABLE_INSTRUCTIONS
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

        // The pack-conversion function from f32 to bf16 only supports
        // packing of matrices to row-major format. Hence Rd kernels can't
        // be used when bf16 API is rerouted to FP32.
        // using rerouted_from_other_backend to check if the DE is rerouted from
        // other backend.
        //
        // RD kernel threshold: RD avoids B packing (cost ~ n*k*sizeof(float)).
        // RD is beneficial when:
        // - k-dominated: k >= m*8 ensures packing cost outweighs compute
        // benefit
        // - Narrow B (n<=8) with sufficient k: B matrix width is small
        // - Tight fringe n: n % 16 in [1-8] causes RV vectorization
        // inefficiency
        //   (just above 16-boundary), bounded by n<=72 and k>=64
        // - Very small m (<=6): RD wins broadly for n<=72
        if (!(rerouted_from_other_backend)
            && ((k >= m * 8) || (n <= 8 && k >= 32)
                || (n % 16 != 0 && n % 16 <= 8 && n <= 72 && k >= 64
                    && (n <= 48 || m <= 32))
                || (m <= 6 && n <= 72))
            && (cs_b != 1) && (mtag_b == PACK) && (mtag_a == UNPACKED)) {
            invokeRD = true;
            k_unroll = 4;
        }

        // Increasing MR helps when n<=16 (mirrors the BF16 fix below).
        //
        // The default DE returns mr=6, nr=64. For shapes with n<=16 the
        // dispatcher only ever reaches the NR=16 family of kernels (the
        // lt16-mask kernel for n<16 and the NR=16 full kernel for n==16),
        // so most of the 32 ZMMs sit idle as C accumulators. Bumping mr
        // lets each cached B line be consumed by more rows of A.
        //
        // nr stays at nr_hint (=64) so the existing row-major NR=64
        // packed-B layout and the framework's N-direction blocking are
        // reused unchanged. The JIT generator below will see the bumped
        // MR and skip the wider NR variants whose register budget would
        // overflow at MR=16 (NR>=32 needs cReg>=32); those slots are
        // never reached at runtime for n<=16 anyway.
        //
        // Override only when:
        //  - n <= 16              (skinny-N: only NR=16 family reached)
        //  - !invokeRD            (RD path has its own internal MR)
        //  - kInstPref is ZMM     (32-ZMM budget is what the math relies on;
        //                          AVX2 path has 16 regs, MR=16 won't fit)
        //  - kc != 1              (the k=1 fused path is sized differently)
        //
        // For M < 16 we cap mr at m so the kernel uses an MR-partial
        // kernel sized exactly to the input row count.
        bool skinnyN = false;
        if (!invokeRD && n <= 16 && m > 0 && kc != 1
            && kInstPref
                   == kernel_frame::kernelInstrPreference::avx512_zmm_favour) {
            mr      = (m < 16) ? m : 16;
            skinnyN = true;
        }

        // L1-cache aliasing mitigation for the skinny-N GEMM bump.
        // When skinnyN raises MR to 16 (or to m, if m>=8) and the user
        // calls with an unpacked A whose row stride in bytes lands
        // within +-ALIAS_GUARD_BYTES (16 B) of a multiple of the L1D
        // way size (4096 B), every k-iteration suffers L1 conflict
        // misses, costing ~40% throughput. Cap MR at the alias-safe
        // associativity bound (8) so each k-iter's MR rows fit within
        // L1D associativity on every Zen variant. Skipped entirely
        // when A is packed (rsA = 1, no aliasing).
        if (skinnyN && mtag_a == AOCL_DLP_MEMORY_TAG::UNPACKED
            && alias_detection::shouldUseMrSplit(rs_a, sizeof(float), mr)) {
            mr = std::min<md_t>(mr, alias_detection::getAliasSafeMrCap());
        }

        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, allLtFringeKernels, invokeRD,
            anyKOpsOrder, kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata,
            skinnyN);
    }

    DLP_ALWAYS_INLINE
    dlp::kernel_frame::packKernelInfo getGemmPackBInfoForInputFastPath(
        [[maybe_unused]] md_t nc,
        [[maybe_unused]] md_t kc,
        md_t                  cs_src,
        md_t                  nr_hint) override final
    {
        if (!canGeneratePackBKernelInfo) {
            return kernel_frame::INVALID_PACK_KERNEL_INFO;
        }

        if (!isAvx512 && !isAvx2) {
            return kernel_frame::INVALID_PACK_KERNEL_INFO;
        }

        bool colMajor = (cs_src != 1);

        kernel_frame::kernelInstrPreference kInstPref =
            kernel_frame::kernelInstrPreference::none;

        if (isAvx512) {
            kInstPref = kernel_frame::kernelInstrPreference::avx512_zmm_favour;
        } else if (isAvx2) {
            kInstPref = kernel_frame::kernelInstrPreference::avx2_ymm_favour;
        }

        return dlp::kernel_frame::packKernelInfo(
            nr_hint, 1, kInstPref, kernel_frame::DataType::f32,
            kernel_frame::DataType::f32, colMajor);
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

    DLP_ALWAYS_INLINE constexpr md_t getPrefetchDistance()
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

    DLP_ALWAYS_INLINE
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
        AOCL_DLP_MEMORY_TAG               mtag_a,
        AOCL_DLP_MEMORY_TAG               mtag_b,
        dlp_gemm_post_op*                 metadata,
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

        // L1-cache aliasing mitigation for BF16 GEMV N=1 (MR=16).
        // Same condition as F32; rsA in BF16 elements -> bytes via
        // sizeof(uint16_t). See alias_detection_utils.hh. Restricted
        // to unpacked A so packed/reordered tags with rs_a==1 don't
        // spuriously trigger the two-pass kernel.
        const bool aliasMrSplit =
            (n == 1 && mtag_a == AOCL_DLP_MEMORY_TAG::UNPACKED)
                ? alias_detection::shouldUseMrSplit(rs_a, sizeof(uint16_t), mr)
                : false;

        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, false, false, anyKOpsOrder,
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata,
            /*skinnyN=*/false, aliasMrSplit);
    }

    DLP_ALWAYS_INLINE
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
        AOCL_DLP_MEMORY_TAG               mtag_a,
        AOCL_DLP_MEMORY_TAG               mtag_b,
        dlp_gemm_post_op*                 metadata,
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

        md_t mr = mr_hint;
        md_t nr = nr_hint;

        // Increasing MR helps when n<=16.
        //
        // The default DE returns mr=6, nr=64. For shapes with n<=16 the
        // dispatcher only ever reaches the NR=16 family of kernels (the
        // lt16-mask kernel for n<16 and the NR=16 full kernel for n==16),
        // so only 6 of the 32 ZMM registers are used as C accumulators --
        // the other ~25 ZMMs sit idle.
        //
        // We bump mr to min(16, M) so each cached B line is now consumed
        // by up to 16 rows of A instead of 6, raising B reuse and cutting
        // the M-iteration count from ceil(M/6) to ceil(M/16). With
        // bReg=1 (the only NR variant the skinny-N dispatch reaches),
        // cReg=16 and aReg=15: well inside the 32-ZMM budget.
        //
        // nr stays at nr_hint (=64) so the existing row-major NR=64
        // packed-B layout and the framework's N-direction blocking are
        // reused unchanged. The JIT generator below skips the wider NR
        // variants whose register budget would overflow at MR=16
        // (NR>=32 needs cReg>=32); those slots are never reached at
        // runtime for n<=16 anyway.
        //
        // For M < 16 we cap mr at m so the kernel uses an MR-partial
        // kernel sized exactly to the input row count (single full
        // panel, no fringe). This avoids leaving C ZMMs idle for tiny-M
        // shapes.
        bool skinnyN = false;
        if (n <= 16 && m > 0) {
            mr      = (m < 16) ? m : 16;
            skinnyN = true;
        }

        // L1-cache aliasing mitigation for the BF16 skinny-N GEMM bump.
        // Same vulnerability as F32 skinnyN: MR=16 + unpacked A with
        // rsA near a multiple of 4096B -> L1 conflict misses, ~40%
        // slowdown. Cap MR to the alias-safe associativity bound.
        if (skinnyN && mtag_a == AOCL_DLP_MEMORY_TAG::UNPACKED
            && alias_detection::shouldUseMrSplit(rs_a, sizeof(uint16_t), mr)) {
            mr = std::min<md_t>(mr, alias_detection::getAliasSafeMrCap());
        }

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
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata, skinnyN);
    }
};

class gemmU8S8DEBackend : public iDEBackend
{
    bool                                isAvx512;
    bool                                isAvx2;
    bool                                isAvx512Vnni;
    kernel_frame::kernelInstrPreference eKernelInstPref;
    bool                                canGenerateKernelInfo;

    DLP_ALWAYS_INLINE
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

    DLP_ALWAYS_INLINE
    dlp::kernel_frame::kernelInfo getGemvKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype k_dtype,
        [[maybe_unused]] md_t             m,
        md_t                              n,
        md_t                              k,
        [[maybe_unused]] md_t             rs_a,
        [[maybe_unused]] md_t             cs_a,
        [[maybe_unused]] md_t             rs_b,
        [[maybe_unused]] md_t             cs_b,
        md_t                              rs_c,
        md_t                              cs_c,
        void*                             alpha,
        void*                             beta,
        AOCL_DLP_MEMORY_TAG               mtag_a,
        AOCL_DLP_MEMORY_TAG               mtag_b,
        dlp_gemm_post_op*                 metadata,
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

    DLP_ALWAYS_INLINE
    dlp::kernel_frame::kernelInfo getGemmKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype k_dtype,
        [[maybe_unused]] md_t             m,
        [[maybe_unused]] md_t             n,
        md_t                              k,
        [[maybe_unused]] md_t             rs_a,
        [[maybe_unused]] md_t             cs_a,
        [[maybe_unused]] md_t             rs_b,
        [[maybe_unused]] md_t             cs_b,
        md_t                              rs_c,
        md_t                              cs_c,
        void*                             alpha,
        void*                             beta,
        AOCL_DLP_MEMORY_TAG               mtag_a,
        AOCL_DLP_MEMORY_TAG               mtag_b,
        dlp_gemm_post_op*                 metadata,
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

        md_t mr = mr_hint;
        md_t nr = nr_hint;
        // k_unroll=2 emits two VNNI groups per outer-loop iteration. The
        // generalised K-tail in u8s8_gemm_generator.cc makes K_UNROLL=2
        // safe on any K, so divisibility is no longer required. A
        // short-K guard keeps K_UNROLL=1 below kUnroll2MinK, because
        // body-doubling does not amortise on very short K and small
        // shapes; K >= 256 is the measured clean region across both
        // u8s8 and s8s8 paths.
        constexpr md_t kUnroll2MinK    = 256;
        md_t           k_unroll        = (k >= kUnroll2MinK) ? 2 : 1;
        md_t           kc              = kc_hint;
        md_t           prefetch_c_dist = getPrefetchDistance();
        bool           anyKOpsOrder    = false;

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

    DLP_ALWAYS_INLINE constexpr md_t getPrefetchDistance()
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

    DLP_ALWAYS_INLINE dlp::kernel_frame::kernelInfo
                      getGemvKernelInfoForInputFastPath(
                          dlp::kernel_frame::kernelDatatype k_dtype,
                          [[maybe_unused]] md_t             m,
                          md_t                              n,
                          md_t                              k,
                          [[maybe_unused]] md_t             rs_a,
                          [[maybe_unused]] md_t             cs_a,
                          [[maybe_unused]] md_t             rs_b,
                          [[maybe_unused]] md_t             cs_b,
                          md_t                              rs_c,
                          md_t                              cs_c,
                          void*                             alpha,
                          void*                             beta,
                          AOCL_DLP_MEMORY_TAG               mtag_a,
                          AOCL_DLP_MEMORY_TAG               mtag_b,
                          dlp_gemm_post_op*                 metadata,
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

    DLP_ALWAYS_INLINE
    dlp::kernel_frame::kernelInfo getGemmKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype k_dtype,
        [[maybe_unused]] md_t             m,
        [[maybe_unused]] md_t             n,
        md_t                              k,
        [[maybe_unused]] md_t             rs_a,
        [[maybe_unused]] md_t             cs_a,
        [[maybe_unused]] md_t             rs_b,
        [[maybe_unused]] md_t             cs_b,
        md_t                              rs_c,
        md_t                              cs_c,
        void*                             alpha,
        void*                             beta,
        AOCL_DLP_MEMORY_TAG               mtag_a,
        AOCL_DLP_MEMORY_TAG               mtag_b,
        dlp_gemm_post_op*                 metadata,
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

        md_t mr = mr_hint;
        md_t nr = nr_hint;
        // k_unroll=2 emits two VNNI groups per outer-loop iteration. The
        // generalised K-tail in s8_gemm_generator.cc makes K_UNROLL=2
        // safe on any K, so divisibility is no longer required. The
        // +128/vpaddb pre-step on s8s8 operates per A-broadcast (not per
        // K-element) and is unaffected by the tail count. A short-K
        // guard keeps K_UNROLL=1 below kUnroll2MinK, because body-
        // doubling does not amortise on very short K and small shapes;
        // K >= 256 is the measured clean region.
        constexpr md_t kUnroll2MinK    = 256;
        md_t           k_unroll        = (k >= kUnroll2MinK) ? 2 : 1;
        md_t           kc              = kc_hint;
        md_t           prefetch_c_dist = getPrefetchDistance();
        bool           anyKOpsOrder    = false;

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

class gemmFP16DEBackend : public iDEBackend
{
    bool                                isAvx512;
    bool                                isAvx512FP16;
    kernel_frame::kernelInstrPreference eKernelInstPref;
    bool                                canGenerateKernelInfo;

    DLP_ALWAYS_INLINE constexpr md_t getPrefetchDistance()
    {
        // Setting prefetch distance for FP16, can be tuned per architecture
        constexpr md_t prefetch_c_dist = 0;
        return prefetch_c_dist;
    }

  public:
    gemmFP16DEBackend();
    ~gemmFP16DEBackend()                                   = default;
    gemmFP16DEBackend(const gemmFP16DEBackend&)            = delete;
    gemmFP16DEBackend(gemmFP16DEBackend&&)                 = delete;
    gemmFP16DEBackend& operator=(const gemmFP16DEBackend&) = delete;
    gemmFP16DEBackend& operator=(gemmFP16DEBackend&&)      = delete;

    std::optional<dlp::kernel_frame::kernelInfo> getKernelInfoForInput(
        iDEInput* in) override;

    DLP_ALWAYS_INLINE
    dlp::kernel_frame::kernelInfo getGemvKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype k_dtype,
        md_t                              m,
        md_t                              n,
        md_t                              k,
        [[maybe_unused]] md_t             rs_a,
        [[maybe_unused]] md_t             cs_a,
        [[maybe_unused]] md_t             rs_b,
        [[maybe_unused]] md_t             cs_b,
        md_t                              rs_c,
        md_t                              cs_c,
        void*                             alpha,
        void*                             beta,
        AOCL_DLP_MEMORY_TAG               mtag_a,
        AOCL_DLP_MEMORY_TAG               mtag_b,
        dlp_gemm_post_op*                 metadata,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale,
        [[maybe_unused]] bool rerouted_from_other_backend) override final
    {
        if (!canGenerateKernelInfo) {
            return INVALID_KERNEL_INFO;
        }

        // FP16 uses float16 for alpha/beta scaling
        kernel_frame::scalingType alphaScalingType;
        kernel_frame::scalingType betaScalingType;
        std::tie(alphaScalingType, betaScalingType) =
            gemmDEBackendUtils::getScalingTypes<dlp::float16>(alpha, beta, k,
                                                              kc_hint);

        md_t mr              = mr_hint;
        md_t nr              = nr_hint;
        md_t k_unroll        = 1;
        md_t kc              = kc_hint;
        md_t prefetch_c_dist = getPrefetchDistance();
        bool anyKOpsOrder    = false;

        kernel_frame::kernelInstrPreference kInstPref = eKernelInstPref;

        if (n == 1) {
            // GEMV N=1: y = A * x (M x K matrix * K x 1 vector)
            // MR = 16 for FP16 (16 rows processed at once)
            mr       = 16;
            nr       = 1;
            k_unroll = 1;
        } else if (m == 1) {
            // GEMV M=1: y = x * B (1 x K vector * K x N matrix)
            // NR = 128 for FP16 (4 ZMMs of 32 FP16 elements each)
            mr       = 1;
            nr       = 128;
            k_unroll = 4;
        } else {
            return INVALID_KERNEL_INFO;
        }

        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, false, false, anyKOpsOrder,
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata);
    }

    DLP_ALWAYS_INLINE
    dlp::kernel_frame::kernelInfo getGemmKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype k_dtype,
        [[maybe_unused]] md_t             m,
        [[maybe_unused]] md_t             n,
        md_t                              k,
        [[maybe_unused]] md_t             rs_a,
        [[maybe_unused]] md_t             cs_a,
        [[maybe_unused]] md_t             rs_b,
        [[maybe_unused]] md_t             cs_b,
        md_t                              rs_c,
        md_t                              cs_c,
        void*                             alpha,
        void*                             beta,
        AOCL_DLP_MEMORY_TAG               mtag_a,
        AOCL_DLP_MEMORY_TAG               mtag_b,
        dlp_gemm_post_op*                 metadata,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale,
        [[maybe_unused]] bool rerouted_from_other_backend) override final
    {
        if (!canGenerateKernelInfo) {
            return INVALID_KERNEL_INFO;
        }

        // FP16 uses float16 for alpha/beta scaling
        kernel_frame::scalingType alphaScalingType;
        kernel_frame::scalingType betaScalingType;
        std::tie(alphaScalingType, betaScalingType) =
            gemmDEBackendUtils::getScalingTypes<dlp::float16>(alpha, beta, k,
                                                              kc_hint);

        md_t mr              = mr_hint;
        md_t nr              = nr_hint;
        md_t k_unroll        = 1;
        md_t kc              = kc_hint;
        md_t prefetch_c_dist = getPrefetchDistance();
        bool anyKOpsOrder    = false;

        kernel_frame::kernelInstrPreference kInstPref = eKernelInstPref;

        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, false, false, anyKOpsOrder,
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata);
    }
};

/**
 * @brief Decision engine backend for F32×FP16→F32 mixed-precision GEMM
 *
 * Uses vcvtph2ps (F16C) for FP16→F32 conversion + vfmadd231ps for F32 FMA.
 * Requires AVX-512F + AVX-512BW only (NOT avx512fp16).
 * Alpha/beta are F32, accumulation is F32.
 */
class gemmF32FP16DEBackend : public iDEBackend
{
    bool                                isAvx512;
    kernel_frame::kernelInstrPreference eKernelInstPref;
    bool                                canGenerateKernelInfo;

    DLP_ALWAYS_INLINE constexpr md_t getPrefetchDistance()
    {
        constexpr md_t prefetch_c_dist = 0;
        return prefetch_c_dist;
    }

  public:
    gemmF32FP16DEBackend();
    ~gemmF32FP16DEBackend()                                      = default;
    gemmF32FP16DEBackend(const gemmF32FP16DEBackend&)            = delete;
    gemmF32FP16DEBackend(gemmF32FP16DEBackend&&)                 = delete;
    gemmF32FP16DEBackend& operator=(const gemmF32FP16DEBackend&) = delete;
    gemmF32FP16DEBackend& operator=(gemmF32FP16DEBackend&&)      = delete;

    std::optional<dlp::kernel_frame::kernelInfo> getKernelInfoForInput(
        iDEInput* in) override;

    DLP_ALWAYS_INLINE
    dlp::kernel_frame::kernelInfo getGemvKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype k_dtype,
        md_t                              m,
        md_t                              n,
        md_t                              k,
        [[maybe_unused]] md_t             rs_a,
        [[maybe_unused]] md_t             cs_a,
        [[maybe_unused]] md_t             rs_b,
        [[maybe_unused]] md_t             cs_b,
        md_t                              rs_c,
        md_t                              cs_c,
        void*                             alpha,
        void*                             beta,
        AOCL_DLP_MEMORY_TAG               mtag_a,
        AOCL_DLP_MEMORY_TAG               mtag_b,
        dlp_gemm_post_op*                 metadata,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale,
        [[maybe_unused]] bool rerouted_from_other_backend) override final
    {
        if (!canGenerateKernelInfo) {
            return INVALID_KERNEL_INFO;
        }

        // F32×FP16 uses float for alpha/beta scaling
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

        kernel_frame::kernelInstrPreference kInstPref = eKernelInstPref;

        if (n == 1) {
            // GEMV N=1: y = A * x (M×K F32 × K×1 FP16 = M×1 F32)
            // MR = 16 for F32×FP16 (16 rows processed at once)
            mr       = 16;
            nr       = 1;
            k_unroll = 1;
        } else if (m == 1) {
            // GEMV M=1: y = x * B (1×K F32 × K×N FP16 = 1×N F32)
            // NR = 64 for F32×FP16 (4 ZMMs of 16 F32 elements each)
            mr       = 1;
            nr       = 64;
            k_unroll = 4;
        } else {
            return INVALID_KERNEL_INFO;
        }

        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, false, false, anyKOpsOrder,
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata);
    }

    DLP_ALWAYS_INLINE
    dlp::kernel_frame::kernelInfo getGemmKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype k_dtype,
        [[maybe_unused]] md_t             m,
        [[maybe_unused]] md_t             n,
        md_t                              k,
        [[maybe_unused]] md_t             rs_a,
        [[maybe_unused]] md_t             cs_a,
        [[maybe_unused]] md_t             rs_b,
        [[maybe_unused]] md_t             cs_b,
        md_t                              rs_c,
        md_t                              cs_c,
        void*                             alpha,
        void*                             beta,
        AOCL_DLP_MEMORY_TAG               mtag_a,
        AOCL_DLP_MEMORY_TAG               mtag_b,
        dlp_gemm_post_op*                 metadata,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale,
        [[maybe_unused]] bool rerouted_from_other_backend) override final
    {
        if (!canGenerateKernelInfo) {
            return INVALID_KERNEL_INFO;
        }

        // F32×FP16 uses float for alpha/beta scaling
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

        kernel_frame::kernelInstrPreference kInstPref = eKernelInstPref;

        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, false, false, anyKOpsOrder,
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata);
    }
};

} // namespace dlp::de
