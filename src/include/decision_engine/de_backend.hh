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
#include <iterator>
#include <memory>
#include <optional>

#include "alias_detection_utils.hh"
#include "bindings/c_wrappers/capi_kernel_frame_wrappers.h"
#include "classic/dlp_macros.h"
#include "de_backend_utils.hh"
#include "de_input.hh"
#include "de_shape_model.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "optimizer/de_optimizer.hh"
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

static const kernel_frame::quantKernelInfo INVALID_GEMM_QUANT_KERNEL_INFO{};

class iDEBackend : public optimizer::gemmOptimizer
{
  public:
    ~iDEBackend() override = default;

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
        dlp_gemm_thread_info_t*           thread_info,
        const dlp_gemm_kernel_hints_t*    gemm_hints,
        md_t                              blksz_set_mask,
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

    // Pack-B kernel selection, and with it the packed panel width.
    //
    // This is where the width is decided on the Reorder path, so it consults
    // the same rule the GEMM path does over the same model input. It takes the
    // fields and builds that input itself, for the reason the GEMM path does:
    // the two ends agree on a width only by running one search over one object,
    // and an object assembled at the call site is an object that can drift from
    // the one the other end assembled.
    //
    // Inside a GEMM the kernel init has already marked the tile in
    // blksz_set_mask, which makes the model ineligible here and leaves nr_hint
    // standing.
    virtual dlp::kernel_frame::packKernelInfo getGemmPackBInfoForInputFastPath(
        [[maybe_unused]] md_t                           nc,
        [[maybe_unused]] md_t                           cs_src,
        [[maybe_unused]] md_t                           n,
        [[maybe_unused]] md_t                           k,
        [[maybe_unused]] md_t                           mr_hint,
        [[maybe_unused]] md_t                           nr_hint,
        [[maybe_unused]] md_t                           blksz_set_mask,
        [[maybe_unused]] const dlp_gemm_kernel_hints_t* gemm_hints)
    {
        return kernel_frame::INVALID_PACK_KERNEL_INFO;
    }
};

class iQuantDEBackend
{
  public:
    virtual ~iQuantDEBackend() = default;

    virtual dlp::kernel_frame::quantKernelInfo
    getGemmQuantKernelInfoForInputFastPath(
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
        dlp_group_op*                     group_ops,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale) = 0;

    virtual dlp::kernel_frame::quantKernelInfo
    getGemvQuantKernelInfoForInputFastPath(
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
        dlp_group_op*                     group_ops,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale) = 0;
};

class gemmF32DEBackend final : public iDEBackend
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
        dlp::kernel_frame::kernelDatatype               k_dtype,
        md_t                                            m,
        md_t                                            n,
        md_t                                            k,
        md_t                                            rs_a,
        [[maybe_unused]] md_t                           cs_a,
        [[maybe_unused]] md_t                           rs_b,
        md_t                                            cs_b,
        md_t                                            rs_c,
        md_t                                            cs_c,
        void*                                           alpha,
        void*                                           beta,
        AOCL_DLP_MEMORY_TAG                             mtag_a,
        AOCL_DLP_MEMORY_TAG                             mtag_b,
        dlp_gemm_post_op*                               metadata,
        md_t                                            mr_hint,
        md_t                                            nr_hint,
        md_t                                            kc_hint,
        md_t                                            c_downscale,
        [[maybe_unused]] dlp_gemm_thread_info_t*        thread_info,
        [[maybe_unused]] const dlp_gemm_kernel_hints_t* gemm_hints,
        md_t                                            blksz_set_mask,
        bool rerouted_from_other_backend) override final
    {
        if (!canGenerateKernelInfo) {
            return INVALID_KERNEL_INFO;
        }

        if ((mr_hint <= 1) || (nr_hint <= 1)) {
            // Invalid MR/NR hints for GEMM, these hints should only be used
            // in GEMV path. The generators are hard set to only generate
            // GEMV kernels for MR/NR <= 1.
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
        // kernel sized exactly to the input row count. The skinnyN flag
        // tells the JIT which NR slots to emit; the MR bump is skipped when
        // the caller pinned MR through metadata.
        shape_model::gemmShapeModelInput mrPinCheck{};
        mrPinCheck.frozen = blksz_set_mask;
        const bool mrFixedByCaller =
            gemmShapeModelUtils::isMRFixedByCaller(mrPinCheck);

        const bool skinnyN =
            !invokeRD && n <= 16 && m > 0 && kc != 1
            && kInstPref
                   == kernel_frame::kernelInstrPreference::avx512_zmm_favour;
        if (skinnyN && !mrFixedByCaller) {
            mr = (m < 16) ? m : 16;
        }

        // L1-cache aliasing mitigation for the skinny-N GEMM bump.
        // When skinnyN raises MR to 16 (or to m, if m>=8) and the user
        // calls with an unpacked A whose row stride in bytes lands
        // within +-ALIAS_GUARD_BYTES (16 B) of a multiple of the L1D
        // way size (4096 B), every k-iteration suffers L1 conflict
        // misses, costing ~40% throughput. Cap MR at the alias-safe
        // associativity bound (8) so each k-iter's MR rows fit within
        // L1D associativity on every Zen variant. Skipped entirely
        // when A is packed (rsA = 1, no aliasing), and when MR was
        // pinned by the caller.
        if (skinnyN && !mrFixedByCaller
            && mtag_a == AOCL_DLP_MEMORY_TAG::UNPACKED
            && alias_detection::shouldUseMrSplit(rs_a, sizeof(float), mr)) {
            mr = std::min<md_t>(mr, alias_detection::getAliasSafeMrCap());
        }

        const bool jitSkinnyN = skinnyN && !mrFixedByCaller;

        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, allLtFringeKernels, invokeRD,
            anyKOpsOrder, kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata,
            jitSkinnyN);
    }

    DLP_ALWAYS_INLINE
    dlp::kernel_frame::packKernelInfo getGemmPackBInfoForInputFastPath(
        [[maybe_unused]] md_t                           nc,
        md_t                                            cs_src,
        [[maybe_unused]] md_t                           n,
        [[maybe_unused]] md_t                           k,
        [[maybe_unused]] md_t                           mr_hint,
        md_t                                            nr_hint,
        [[maybe_unused]] md_t                           blksz_set_mask,
        [[maybe_unused]] const dlp_gemm_kernel_hints_t* gemm_hints)
        override final // Should we pass a pointer, so that it can be NULL for
                       // non-supported DEs?
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

        // No shape model on this backend, so the incoming NR stands.
        return dlp::kernel_frame::packKernelInfo(
            nr_hint, 1, kInstPref, kernel_frame::DataType::f32,
            kernel_frame::DataType::f32, colMajor);
    }
};

// final, and not as documentation. The optimizer reaches its datatype hooks,
// adjustWays and costEval, through the base that defines them. With this class
// left open, a further-derived backend could re-override either, so the
// compiler would have to keep the vtable load. That measured as two indirect
// calls inside the per-candidate loop; final removes them.
class gemmBF16DEBackend final : public iDEBackend
{
  public:
    // This backend's half of the shape model: the tiles its JIT generator can
    // emit. That is a fact about this datatype on this microarchitecture, so it
    // is not the model's to know. proposeShape below hands the set to the
    // shared search in optimizer/de_optimizer.hh, which is why another datatype
    // contributes a different set rather than a different algorithm.
    //
    // One entry, the Zen5 BF16 default, which leaves the sweep nothing to rank:
    // the only admissible tile wins, and it is the tile the context already
    // held. Widening this set is what gives costEval something to tell apart.
    static constexpr shape_model::kernelDims candidateTiles[] = {
        { 6, 64 },
    };

    // The n at or below which only the NR=16 kernel family is reachable, and
    // the MR raised to there. Named because the rule that applies them and the
    // flag that tells the JIT generator which NR variants it can skip both read
    // them. Where the 16 comes from is written down at the rule, in the
    // kernel-info fold.
    static constexpr md_t skinnyNThreshold = 16;
    static constexpr md_t skinnyNMr        = 16;

  private:
    bool                                isAvx512;
    bool                                isAvx2;
    bool                                isAvx512Bf16;
    kernel_frame::kernelInstrPreference eKernelInstPref;
    bool                                canGenerateKernelInfo;
    std::unique_ptr<gemmF32DEBackend>
        f32Backend; // For rerouting when AVX512BF16 is not supported

    // The tiles are tuned against the Zen5 cache hierarchy, so the model is
    // fenced to that architecture. Resolved once at construction, since it
    // cannot change for the life of the process.
    bool isAnalyticalShapeModelArch;

    // The whole screen: this architecture, and whether the object describes a
    // GEMM well enough to choose a tile for. See
    // gemmShapeModelUtils::isEligible for the second half.
    //
    // Over a reordered B the object holds the hints, so an unstated pair fails
    // this and both ends fall to the context tile. Under any other tag it holds
    // the call, which describes itself.
    //
    // This is the one predicate that decides the arm. The kernel-info fold asks
    // it again before publishing a split, and asking twice is the point: a
    // split may only be published by the arm that resolved one.
    DLP_ALWAYS_INLINE bool canUseAnalyticalShapeModel(
        const shape_model::gemmShapeModelInput& in) const
    {
        return isAnalyticalShapeModelArch
               && gemmShapeModelUtils::isEligible(in);
    }

  protected:
    // What BF16 does to the seed partition. It matches
    // dlp_gemm_bf16bf16f32of32_get_threading, with the runtime and context
    // reads taken as arguments so a scoring loop can call it once per
    // candidate.
    //
    // These two rebalances are the whole of this datatype's divergence from
    // the shared skeleton in gemmThreadPartitioner, which is why the hook is
    // per-datatype: F32's classic factorizer works from MC, NC and KC and does
    // neither.
    //
    // always_inline for the same reason resolveGemmShape is, and it matters
    // more here: this runs once per candidate rather than once per call, so an
    // out-of-line copy would put a PLT call inside that loop.
    DLP_ALWAYS_INLINE void adjustWays(md_t  mr,
                                      md_t  nr,
                                      md_t  m,
                                      md_t  n,
                                      md_t& nThreads,
                                      md_t& icWays,
                                      md_t& jcWays) const override final
    {
        const md_t mrBlks = (m + mr - 1) / mr;

        // The following attempts to further redistribute the threads among the
        // ic and jc ways, in case the initial partitioning is not optimal.
        // This is done purely based on the total panels of work per thread,
        // and we attempt to mitigate the imbalance by increasing the ic ways
        // and decreasing the jc ways. We check if there is oversubsciption in
        // the ic direction, before calling it.
        if (mrBlks >= icWays) {
            thread_partition::adjustIcJcWays(mr, nr, m, n, nThreads, icWays,
                                             jcWays);
        }

        if (utils::math::isPrime(nThreads)) {
            thread_partition::adjustForPrimeThreadCount(mr, nr, m, n, nThreads,
                                                        icWays, jcWays);
        }
    }

  public:
    // This backend has a candidate set, so it runs the sweep instead of the
    // inherited baseline.
    DLP_ALWAYS_INLINE shape_model::gemmShapeModelResult proposeShape(
        const shape_model::gemmShapeModelInput& in) const override final
    {
        return sweepCandidates(in, candidateTiles);
    }

    // final so the call devirtualises, and always_inline because final alone
    // is not enough. A virtual override is emitted as an interposable
    // definition, which the compiler will not inline through even once the
    // call binds statically. The sweep has to stay inlined here: measured out
    // of line, it costs a PLT call on the per-call path.
    //
    // An out-of-line copy is still emitted to fill the vtable slot, and that
    // is the one a caller holding an iDEBackend* reaches.
    DLP_ALWAYS_INLINE shape_model::gemmShapeModelResult resolveGemmShape(
        const shape_model::gemmShapeModelInput& in) const override final
    {
        return canUseAnalyticalShapeModel(in)
                   ? gemmBF16DEBackend::proposeShape(in)
                   : gemmBF16DEBackend::baselineShape(in);
    }

  private:
    DLP_ALWAYS_INLINE constexpr md_t getPrefetchDistance()
    {
        // Setting this to 40, which works for ZEN5. Should we set this in
        // the DE constructor, based on the underlying arch?
        constexpr md_t prefetch_c_dist = 40;
        return prefetch_c_dist;
    }

    const md_t k_pack_factor = 2; // BF16 packing: 2 x bf16 -> float

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

        md_t kc_rounded =
            ((kc + k_pack_factor - 1) / k_pack_factor) * k_pack_factor;
        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc_rounded, prefetch_c_dist, alphaScalingType,
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
        dlp_gemm_thread_info_t*           thread_info,
        const dlp_gemm_kernel_hints_t*    gemm_hints,
        md_t                              blksz_set_mask,
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
                c_downscale, thread_info, gemm_hints, blksz_set_mask, true);
        }

        if ((mr_hint <= 1) || (nr_hint <= 1)) {
            // Invalid MR/NR hints for GEMM, these hints should only be used
            // in GEMV path. The generators are hard set to only generate
            // GEMV kernels for MR/NR <= 1.
            return INVALID_KERNEL_INFO;
        }

        // At this point, we know that the underlying architecture supports
        // AVX512-BF16.
        kernel_frame::scalingType alphaScalingType;
        kernel_frame::scalingType betaScalingType;
        std::tie(alphaScalingType, betaScalingType) =
            gemmDEBackendUtils::getScalingTypes<float>(alpha, beta, k, kc_hint);

        md_t nr = nr_hint;

        // For n<=16 only the NR=16 kernel family is reachable (lt16-mask for
        // n<16, the full kernel at n==16), so the JIT generator can skip the
        // wider NR variants. This says which kernels exist for the shape, not
        // which tile was chosen, so it is read off the call on either arm.
        const bool skinnyN = (n <= skinnyNThreshold) && (m > 0);

        // A caller may not offer thread info at all, so read the pool through a
        // zeroed default rather than branching on the pointer at each use.
        static constexpr dlp_gemm_thread_info_t noThreadInfo{};
        const dlp_gemm_thread_info_t&           threads =
            (thread_info != nullptr) ? *thread_info : noThreadInfo;

        // The hints are read whole and handed over as they were stated. Whether
        // they describe anything is the tag's business, and makeModelInput
        // settles it in one place: only a reordered B has a packed panel whose
        // width has to be reproduced, and only there do the hints stand for the
        // GEMM being modelled. Under any other tag they are not read, stated or
        // not, so nothing upstream has to blank them.
        const md_t m_hint  = (gemm_hints != nullptr) ? gemm_hints->m_hint : 0;
        const md_t nt_hint = (gemm_hints != nullptr) ? gemm_hints->nt_hint : 0;

        const bool b_reordered = (mtag_b == AOCL_DLP_MEMORY_TAG::REORDERED);

        const shape_model::gemmShapeModelInput in = shape_model::makeModelInput(
            m, n, k, mr_hint, nr_hint, blksz_set_mask, m_hint, nt_hint,
            threads.num_threads, threads.ic_ways, threads.jc_ways, b_reordered);

        // Qualified so the call binds statically. Left unqualified it is a
        // virtual call the linker may interpose, which costs the inlining.
        const shape_model::gemmShapeModelResult shape =
            gemmBF16DEBackend::resolveGemmShape(in);

        md_t mr = shape.mr;

        nr = shape.nr;

        // Two register-block rules, applied to whichever tile came back.
        //
        // Both are facts about the call in hand, and the model answers for a
        // GEMM that may not be this call: its input holds the hinted extents
        // over a reordered B and carries no strides at all. Applying them here
        // is sound because a packed B panel records only NR, which neither rule
        // touches, so the two ends of a reorder still meet at the same width.
        //
        // The skinny-N bump. The default tile is 6x64, and at n<=16 only the
        // NR=16 kernel family is reachable, where six of the 32 ZMMs hold C
        // accumulators and roughly 25 sit idle. Raising MR to 16 feeds each
        // cached B line to 16 rows of A instead of 6 and cuts the M-loop from
        // ceil(m/6) to ceil(m/16), at cReg=16, aReg=15 and bReg=1, inside the
        // budget. NR stays put, so the row-major NR=64 packed-B layout and the
        // N-direction blocking are reused untouched.
        //
        // Below m=16 the cap is m itself, sizing the kernel to one full panel
        // with no fringe. MR cannot exceed the rows the caller brought, which
        // is the other half of why this reads m and not m_hint. Skipped when
        // the caller pinned MR through metadata.
        if (skinnyN && !gemmShapeModelUtils::isMRFixedByCaller(in)) {
            mr = std::min<md_t>(m, skinnyNMr);
        }

        // The L1 aliasing guard, which the bump above is what makes reachable.
        // The vulnerability needs an MR over the L1D associativity to exist at
        // all, so at the default 6 shouldUseMrSplit returns false and this
        // costs a compare. Where MR did reach 16 and A is unpacked with a row
        // stride landing within a hair of a multiple of the 4096 B L1D way
        // size, every k-iteration takes conflict misses instead, for about 40%
        // of throughput. Capping MR at the associativity bound puts each
        // k-iteration's rows back inside one set's worth of ways. Skipped when
        // the caller pinned MR through metadata.
        if (skinnyN && !gemmShapeModelUtils::isMRFixedByCaller(in)
            && (mtag_a == AOCL_DLP_MEMORY_TAG::UNPACKED)
            && alias_detection::shouldUseMrSplit(rs_a, sizeof(uint16_t), mr)) {
            mr = std::min<md_t>(mr, alias_detection::getAliasSafeMrCap());
        }

        // Publish the split, so the threading decorator takes it instead of
        // deriving its own. Both run the same heuristic over the same extents,
        // so the partition does not change; what changes is that the split a
        // candidate was costed against is the split that executes. Over a
        // reordered B those extents are the hinted GEMM, and the split goes out
        // all the same, because substituting another here would break the one
        // property publishing exists to hold.
        //
        // Two cases publish nothing, and both are decided here because only
        // this level can see them.
        //
        // The baseline arm resolves no split, so it has none to publish. Its
        // result echoes the input's ways, which over an unhinted reordered B
        // are zeros standing for a GEMM nobody described.
        //
        // A pinned runtime is already an answer. Over a reordered B the model
        // cannot even see the pin, since the ways go in zeroed so that both
        // ends rank alike, so a split from there would discard DLP_IC_NT /
        // DLP_JC_NT. Such a call is still served the tile both ends agree on:
        // that width came from nt_hint, which the entry point has already held
        // the pin's product to. Elsewhere the factorizer was handed the pin and
        // returns it unchanged, so withholding it costs nothing and says the
        // true thing, that the partition is the caller's.
        const bool publishSplit = canUseAnalyticalShapeModel(in)
                                  && (threads.ic_ways <= 0)
                                  && (threads.jc_ways <= 0);

        if ((thread_info != nullptr) && publishSplit) {
            thread_info->num_threads = shape.num_threads;
            thread_info->ic_ways     = shape.ic_ways;
            thread_info->jc_ways     = shape.jc_ways;
        }

        md_t k_unroll        = 1;
        md_t kc              = kc_hint;
        md_t prefetch_c_dist = getPrefetchDistance();
        bool anyKOpsOrder    = false;

        // Set the kernel instruction preference based on the CPU features.
        // The DE constructor sets it to a safe value, based on the hardware
        // support.
        kernel_frame::kernelInstrPreference kInstPref = eKernelInstPref;

        md_t kc_rounded =
            ((kc + k_pack_factor - 1) / k_pack_factor) * k_pack_factor;

        const bool jitSkinnyN =
            skinnyN && !gemmShapeModelUtils::isMRFixedByCaller(in);

        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc_rounded, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, false, false, anyKOpsOrder,
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata, jitSkinnyN);
    }

    DLP_ALWAYS_INLINE
    dlp::kernel_frame::packKernelInfo getGemmPackBInfoForInputFastPath(
        [[maybe_unused]] md_t          nc,
        md_t                           cs_src,
        md_t                           n,
        md_t                           k,
        md_t                           mr_hint,
        md_t                           nr_hint,
        md_t                           blksz_set_mask,
        const dlp_gemm_kernel_hints_t* gemm_hints) override final
    {
        // BF16 pack-B JIT is only valid on AVX-512-BF16. On a non-AVX-512-BF16
        // machine no JIT-based pack-B is taken at all: when the backend has
        // rerouted to F32 (no native BF16 path, or an AVX2/arch downgrade), the
        // dedicated BF16->F32 fallback 5-loop converts/unreorders B to F32 and
        // computes with the F32 micro-kernels. Report invalid here so the frame
        // skips JIT pack-B on that path.
        if (!isAvx512Bf16 || f32Backend != nullptr) {
            return kernel_frame::INVALID_PACK_KERNEL_INFO;
        }

        bool colMajor = (cs_src != 1);

        // BF16 fuses two consecutive K elements per lane (vdpbf16ps); the
        // packed panel is laid out in K-pairs, hence k_factor = 2. Both src and
        // dst of the packer are bf16 (the GEMM output type does not affect B
        // packing).
        constexpr md_t k_factor = 2;

        // The model input, built exactly as the kernel init builds it, which is
        // the whole of how a Reorder and a later GEMM arrive at one width.
        //
        // A Reorder is not a GEMM, so there is no call to answer for: no m, no
        // thread count and no ways, and nothing in a packed panel depends on
        // any of the three. The zeros say that rather than stand in for it.
        //
        // b_reordered is passed true because that is what this path is. There
        // is no tag to read, reordering being the operation rather than a
        // property of an operand, and the hints are the only extents on offer.
        const shape_model::gemmShapeModelInput in = shape_model::makeModelInput(
            /*m=*/0, n, k, mr_hint, nr_hint, blksz_set_mask,
            (gemm_hints != nullptr) ? gemm_hints->m_hint : 0,
            (gemm_hints != nullptr) ? gemm_hints->nt_hint : 0,
            /*num_threads=*/0, /*ic_ways=*/0, /*jc_ways=*/0,
            /*b_reordered=*/true);

        // The Reorder path is the one that decides the width, and it decides it
        // with the same call the GEMM path makes, so the two cannot drift.
        // Inside a GEMM the kernel init has already marked the tile in
        // blksz_set_mask, so this is ineligible and in.nr -- the NR that init
        // settled -- stands.
        //
        // Only the NR is taken. The MR and the split that come back describe
        // the GEMM the hints predict, and nothing in a packed B panel depends
        // on either. Resolving them anyway is the price of running the
        // identical search: a cost model ranks candidates by the split they
        // would run under, so a tile-only variant here would be free to
        // disagree with the GEMM.
        const md_t nr = canUseAnalyticalShapeModel(in)
                            ? gemmBF16DEBackend::proposeShape(in).nr
                            : in.nr;

        return dlp::kernel_frame::packKernelInfo(
            nr, k_factor, eKernelInstPref, kernel_frame::DataType::bf16,
            kernel_frame::DataType::bf16, colMajor);
    }
};

class gemmU8S8DEBackend final : public iDEBackend
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

    const md_t k_pack_factor = 4; // U8/S8 packing: 4 x u8/s8 -> int32

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
            if (isAvx512Vnni) {
                kInstPref =
                    kernel_frame::kernelInstrPreference::avx512_zmm_favour;
            } else {
                // This is an invalid case, disable jit kernel generation.
                return INVALID_KERNEL_INFO;
            }
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

        md_t kc_rounded =
            ((kc + k_pack_factor - 1) / k_pack_factor) * k_pack_factor;
        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc_rounded, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, false, false, anyKOpsOrder,
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata);
    }

    DLP_ALWAYS_INLINE
    dlp::kernel_frame::kernelInfo getGemmKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype               k_dtype,
        [[maybe_unused]] md_t                           m,
        [[maybe_unused]] md_t                           n,
        md_t                                            k,
        [[maybe_unused]] md_t                           rs_a,
        [[maybe_unused]] md_t                           cs_a,
        [[maybe_unused]] md_t                           rs_b,
        [[maybe_unused]] md_t                           cs_b,
        md_t                                            rs_c,
        md_t                                            cs_c,
        void*                                           alpha,
        void*                                           beta,
        AOCL_DLP_MEMORY_TAG                             mtag_a,
        AOCL_DLP_MEMORY_TAG                             mtag_b,
        dlp_gemm_post_op*                               metadata,
        md_t                                            mr_hint,
        md_t                                            nr_hint,
        md_t                                            kc_hint,
        md_t                                            c_downscale,
        [[maybe_unused]] dlp_gemm_thread_info_t*        thread_info,
        [[maybe_unused]] const dlp_gemm_kernel_hints_t* gemm_hints,
        [[maybe_unused]] md_t                           blksz_set_mask,
        [[maybe_unused]] bool rerouted_from_other_backend) override final
    {
        if (!canGenerateKernelInfo) {
            return INVALID_KERNEL_INFO;
        }

        if ((mr_hint <= 1) || (nr_hint <= 1)) {
            // Invalid MR/NR hints for GEMM, these hints should only be used
            // in GEMV path. The generators are hard set to only generate
            // GEMV kernels for MR/NR <= 1.
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
            if (isAvx512Vnni) {
                kInstPref =
                    kernel_frame::kernelInstrPreference::avx512_zmm_favour;
            } else {
                // This is an invalid case, disable jit kernel generation.
                return INVALID_KERNEL_INFO;
            }
        }

        md_t kc_rounded =
            ((kc + k_pack_factor - 1) / k_pack_factor) * k_pack_factor;
        // Currently only general GEMM is supported, specific GEMM optimizations
        // will be added later
        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc_rounded, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, false, false, anyKOpsOrder,
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata);
    }
};

class gemmS8DEBackend final : public iDEBackend
{
    bool                                isAvx512;
    bool                                isAvx2;
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

    const md_t k_pack_factor = 4; // S8 packing: 4 x s8 -> int32

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
            if (isAvx512Vnni) {
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

        md_t kc_rounded =
            ((kc + k_pack_factor - 1) / k_pack_factor) * k_pack_factor;
        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc_rounded, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, false, false, anyKOpsOrder,
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata);
    }

    DLP_ALWAYS_INLINE
    dlp::kernel_frame::kernelInfo getGemmKernelInfoForInputFastPath(
        dlp::kernel_frame::kernelDatatype               k_dtype,
        [[maybe_unused]] md_t                           m,
        [[maybe_unused]] md_t                           n,
        md_t                                            k,
        [[maybe_unused]] md_t                           rs_a,
        [[maybe_unused]] md_t                           cs_a,
        [[maybe_unused]] md_t                           rs_b,
        [[maybe_unused]] md_t                           cs_b,
        md_t                                            rs_c,
        md_t                                            cs_c,
        void*                                           alpha,
        void*                                           beta,
        AOCL_DLP_MEMORY_TAG                             mtag_a,
        AOCL_DLP_MEMORY_TAG                             mtag_b,
        dlp_gemm_post_op*                               metadata,
        md_t                                            mr_hint,
        md_t                                            nr_hint,
        md_t                                            kc_hint,
        md_t                                            c_downscale,
        [[maybe_unused]] dlp_gemm_thread_info_t*        thread_info,
        [[maybe_unused]] const dlp_gemm_kernel_hints_t* gemm_hints,
        [[maybe_unused]] md_t                           blksz_set_mask,
        [[maybe_unused]] bool rerouted_from_other_backend) override final
    {
        if (!canGenerateKernelInfo) {
            return INVALID_KERNEL_INFO;
        }

        if ((mr_hint <= 1) || (nr_hint <= 1)) {
            // Invalid MR/NR hints for GEMM, these hints should only be used
            // in GEMV path. The generators are hard set to only generate
            // GEMV kernels for MR/NR <= 1.
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
            if (isAvx512Vnni) {
                kInstPref =
                    kernel_frame::kernelInstrPreference::avx512_zmm_favour;
            } else {
                // This is an invalid case, disable jit kernel generation.
                return INVALID_KERNEL_INFO;
            }
        }

        md_t kc_rounded =
            ((kc + k_pack_factor - 1) / k_pack_factor) * k_pack_factor;
        return gemmDEBackendUtils::checkPostOpsAndCreateKernelInfo(
            mr, nr, 0, k_unroll, kc_rounded, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, false, false, anyKOpsOrder,
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata);
    }
};

class gemmQuantS8DEBackend : public iQuantDEBackend
{
    bool                                isAvx512;
    bool                                isAvx2;
    bool                                isAvx512Bf16;
    bool                                isAvx512Vnni;
    kernel_frame::kernelInstrPreference eKernelInstPref;
    bool                                canGenerateKernelInfo;

  public:
    gemmQuantS8DEBackend();
    ~gemmQuantS8DEBackend()                                      = default;
    gemmQuantS8DEBackend(const gemmQuantS8DEBackend&)            = delete;
    gemmQuantS8DEBackend(gemmQuantS8DEBackend&&)                 = delete;
    gemmQuantS8DEBackend& operator=(const gemmQuantS8DEBackend&) = delete;
    gemmQuantS8DEBackend& operator=(gemmQuantS8DEBackend&&)      = delete;

    DLP_ALWAYS_INLINE
    dlp::kernel_frame::quantKernelInfo getGemmQuantKernelInfoForInputFastPath(
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
        dlp_group_op*                     group_ops,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale) override final
    {
        if (!canGenerateKernelInfo || group_ops == nullptr) {
            return INVALID_GEMM_QUANT_KERNEL_INFO;
        }

        kernel_frame::scalingType alphaScalingType;
        kernel_frame::scalingType betaScalingType;
        std::tie(alphaScalingType, betaScalingType) =
            gemmDEBackendUtils::getScalingTypes<int32_t>(alpha, beta, k,
                                                         kc_hint);

        md_t           mr              = mr_hint;
        md_t           nr              = nr_hint;
        constexpr md_t kUnroll2MinK    = 256;
        md_t           k_unroll        = (k >= kUnroll2MinK) ? 2 : 1;
        md_t           kc              = kc_hint;
        md_t           prefetch_c_dist = 0;
        bool           anyKOpsOrder    = false;

        kernel_frame::kernelInstrPreference kInstPref = eKernelInstPref;

        if (kInstPref == kernel_frame::kernelInstrPreference::none) {
            if (isAvx512) {
                kInstPref =
                    kernel_frame::kernelInstrPreference::avx512_zmm_favour;
            } else {
                // Invalid ISA, disable JIT kernel generation.
                return INVALID_GEMM_QUANT_KERNEL_INFO;
            }
        }

        return gemmDEBackendUtils::checkPostOpsAndCreateQuantKernelInfo(
            mr, nr, 0, k_unroll, kc, prefetch_c_dist, alphaScalingType,
            betaScalingType, mtag_a, mtag_b, false, false, anyKOpsOrder,
            kInstPref, c_downscale, k_dtype, rs_c, cs_c, metadata, group_ops);
    }

    DLP_ALWAYS_INLINE
    dlp::kernel_frame::quantKernelInfo getGemvQuantKernelInfoForInputFastPath(
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
        dlp_group_op*                     group_ops,
        md_t                              mr_hint,
        md_t                              nr_hint,
        md_t                              kc_hint,
        md_t                              c_downscale) override final
    {
        if (!canGenerateKernelInfo || group_ops == nullptr) {
            return INVALID_GEMM_QUANT_KERNEL_INFO;
        }

        // The sym-quant GEMV kernels dequantize into f32 accumulators and only
        // carry the f32 and bf16 store rails. Anything else must stay on the
        // GEMM path, which has the wider set.
        if ((c_downscale != DLP_F32) && (c_downscale != DLP_BF16)) {
            return INVALID_GEMM_QUANT_KERNEL_INFO;
        }

        // A GEMV-shaped input does not by itself mean the frame will run a GEMV
        // kernel: The framework dispatches to the GEMV path only when B is
        // reordered and the group size tiles both K and KC evenly.
        // Otherwise falls through to the GEMM 5-loop. The kernel handle
        // carries only one blocking, and mr/nr is what selects the runtime
        // parameter struct, so a GEMV key handed to the GEMM 5-loop would feed
        // GEMM machine code a GEMV parameter block. Mirror the frame's gate
        // here and report "no GEMV kernel" when it does not hold, so the caller
        // falls back to the GEMM key that path actually needs.
        if (mtag_b != REORDERED) {
            return INVALID_GEMM_QUANT_KERNEL_INFO;
        }

        // group_ops points at the caller's metadata, where group_size 0 means
        // one group spanning the full K. The frame compares against the
        // already-normalised value, so normalise identically before testing.
        md_t group_size = 0;
        if (group_ops->a_post_quant_op != nullptr) {
            group_size = group_ops->a_post_quant_op->group_size;
        } else if (group_ops->b_post_quant_op != nullptr) {
            group_size = group_ops->b_post_quant_op->group_size;
        }
        if ((group_size == 0) || (group_size > k)) {
            group_size = k;
        }
        if ((group_size <= 0) || ((k % group_size) != 0)
            || ((kc_hint % group_size) != 0)) {
            return INVALID_GEMM_QUANT_KERNEL_INFO;
        }

        md_t mr;
        md_t nr;
        md_t k_unroll;

        if (n == 1) {
            // Matches the MR the n == 1 branch of the sym_quant 5-loop frame
            // hardcodes; the kernel reduces MR rows along K per group.
            mr       = 16;
            nr       = 1;
            k_unroll = 1;
        } else {
            // m == 1. The M=1 kernel bakes NR and KC into the generated code
            // and owns its own N-tile loop, so both must agree with what the
            // frame drives that loop with (lcntx->blksz.NR / .KC). Deriving
            // them from the hints rather than hardcoding keeps that contract
            // even if the block-size table changes.
            mr       = 1;
            nr       = nr_hint;
            k_unroll = 4;

            if ((nr <= 0) || ((nr % 16) != 0) || (nr > 64)) {
                return INVALID_GEMM_QUANT_KERNEL_INFO;
            }
        }

        kernel_frame::scalingType alphaScalingType;
        kernel_frame::scalingType betaScalingType;
        std::tie(alphaScalingType, betaScalingType) =
            gemmDEBackendUtils::getScalingTypes<int32_t>(alpha, beta, k,
                                                         kc_hint);

        kernel_frame::kernelInstrPreference kInstPref = eKernelInstPref;

        if (kInstPref == kernel_frame::kernelInstrPreference::none) {
            if (isAvx512) {
                kInstPref =
                    kernel_frame::kernelInstrPreference::avx512_zmm_favour;
            } else {
                // Invalid ISA, disable JIT kernel generation.
                return INVALID_GEMM_QUANT_KERNEL_INFO;
            }
        }

        return gemmDEBackendUtils::checkPostOpsAndCreateQuantKernelInfo(
            mr, nr, 0, k_unroll, kc_hint, 0, alphaScalingType, betaScalingType,
            mtag_a, mtag_b, false, false, false, kInstPref, c_downscale,
            k_dtype, rs_c, cs_c, metadata, group_ops);
    }
};

class gemmFP16DEBackend final : public iDEBackend
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
        dlp::kernel_frame::kernelDatatype               k_dtype,
        [[maybe_unused]] md_t                           m,
        [[maybe_unused]] md_t                           n,
        md_t                                            k,
        [[maybe_unused]] md_t                           rs_a,
        [[maybe_unused]] md_t                           cs_a,
        [[maybe_unused]] md_t                           rs_b,
        [[maybe_unused]] md_t                           cs_b,
        md_t                                            rs_c,
        md_t                                            cs_c,
        void*                                           alpha,
        void*                                           beta,
        AOCL_DLP_MEMORY_TAG                             mtag_a,
        AOCL_DLP_MEMORY_TAG                             mtag_b,
        dlp_gemm_post_op*                               metadata,
        md_t                                            mr_hint,
        md_t                                            nr_hint,
        md_t                                            kc_hint,
        md_t                                            c_downscale,
        [[maybe_unused]] dlp_gemm_thread_info_t*        thread_info,
        [[maybe_unused]] const dlp_gemm_kernel_hints_t* gemm_hints,
        [[maybe_unused]] md_t                           blksz_set_mask,
        [[maybe_unused]] bool rerouted_from_other_backend) override final
    {
        if (!canGenerateKernelInfo) {
            return INVALID_KERNEL_INFO;
        }

        if ((mr_hint <= 1) || (nr_hint <= 1)) {
            // Invalid MR/NR hints for GEMM, these hints should only be used
            // in GEMV path. The generators are hard set to only generate
            // GEMV kernels for MR/NR <= 1.
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
class gemmF32FP16DEBackend final : public iDEBackend
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
        dlp::kernel_frame::kernelDatatype               k_dtype,
        [[maybe_unused]] md_t                           m,
        [[maybe_unused]] md_t                           n,
        md_t                                            k,
        [[maybe_unused]] md_t                           rs_a,
        [[maybe_unused]] md_t                           cs_a,
        [[maybe_unused]] md_t                           rs_b,
        [[maybe_unused]] md_t                           cs_b,
        md_t                                            rs_c,
        md_t                                            cs_c,
        void*                                           alpha,
        void*                                           beta,
        AOCL_DLP_MEMORY_TAG                             mtag_a,
        AOCL_DLP_MEMORY_TAG                             mtag_b,
        dlp_gemm_post_op*                               metadata,
        md_t                                            mr_hint,
        md_t                                            nr_hint,
        md_t                                            kc_hint,
        md_t                                            c_downscale,
        [[maybe_unused]] dlp_gemm_thread_info_t*        thread_info,
        [[maybe_unused]] const dlp_gemm_kernel_hints_t* gemm_hints,
        [[maybe_unused]] md_t                           blksz_set_mask,
        [[maybe_unused]] bool rerouted_from_other_backend) override final
    {
        if (!canGenerateKernelInfo) {
            return INVALID_KERNEL_INFO;
        }

        if ((mr_hint <= 1) || (nr_hint <= 1)) {
            // Invalid MR/NR hints for GEMM, these hints should only be used
            // in GEMV path. The generators are hard set to only generate
            // GEMV kernels for MR/NR <= 1.
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
