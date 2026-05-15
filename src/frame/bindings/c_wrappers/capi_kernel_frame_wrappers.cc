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

#include <iostream>
#include <optional>

#include "bindings/c_wrappers/capi_kernel_frame_wrappers.h"
#include "classic/dlp_macros.h"
#include "decision_engine/decision_engine.hh"
#include "jit/jit_kernel_adapter.hh"
#include "jit_register/jit_register.hh"
#include "kernel_register/kernel_register.hh"
#include "utils/ctype_utils.hh"

using namespace dlp::kernel_frame;
using namespace dlp::jit;
using namespace dlp::kernels;
using namespace dlp::utils;

// function to return kernel family name string using kdtype as argument
static std::string
get_kernel_family_name(kernelDatatype kDtype)
{
    switch (kDtype) {
        case kernelDatatype::f32f32f32of32:
            return "FP32JitKernel";
        case kernelDatatype::bf16bf16f32of32:
            return "BF16JitKernel";
        case kernelDatatype::bf16bf16f32obf16:
            return "BF16JitKernel";
        case kernelDatatype::u8s8s32os32:
            return "dlp_u8s8s32os32_jit_kernel";
        case kernelDatatype::u8s8s32of32:
            return "dlp_u8s8s32of32_jit_kernel";
        case kernelDatatype::u8s8s32of16:
            return "dlp_u8s8s32of16_jit_kernel";
        case kernelDatatype::u8s8s32obf16:
            return "dlp_u8s8s32obf16_jit_kernel";
        case kernelDatatype::u8s8s32os8:
            return "dlp_u8s8s32os8_jit_kernel";
        case kernelDatatype::u8s8s32ou8:
            return "dlp_u8s8s32ou8_jit_kernel";
        case kernelDatatype::s8s8s32os32:
            return "dlp_s8s8s32os32_jit_kernel";
        case kernelDatatype::s8s8s32of32:
            return "dlp_s8s8s32of32_jit_kernel";
        case kernelDatatype::s8s8s32of16:
            return "dlp_s8s8s32of16_jit_kernel";
        case kernelDatatype::s8s8s32obf16:
            return "dlp_s8s8s32obf16_jit_kernel";
        case kernelDatatype::s8s8s32os8:
            return "dlp_s8s8s32os8_jit_kernel";
        case kernelDatatype::s8s8s32ou8:
            return "dlp_s8s8s32ou8_jit_kernel";
        case kernelDatatype::f16f16f16of16:
            return "dlp_f16f16f16of16_jit_kernel";
        case kernelDatatype::f16f16f16of32:
            return "dlp_f16f16f16of32_jit_kernel";
        case kernelDatatype::f32f16f32of32:
            return "dlp_f32f16f32of32_jit_kernel";
        default:
            return "dlp_unknown_jit_kernel";
    }
}

DLP_ALWAYS_INLINE static dlp::kernel_frame::kernelInfo
dlp_get_gemm_kernelInfo_by_dtype(kernelDatatype      kDType,
                                 md_t                m,
                                 md_t                n,
                                 md_t                k,
                                 md_t                rs_a,
                                 md_t                cs_a,
                                 md_t                rs_b,
                                 md_t                cs_b,
                                 md_t                rs_c,
                                 md_t                cs_c,
                                 void*               alpha,
                                 void*               beta,
                                 AOCL_DLP_MEMORY_TAG mtag_a,
                                 AOCL_DLP_MEMORY_TAG mtag_b,
                                 dlp_gemm_post_op*   metadata,
                                 md_t                mr_hint,
                                 md_t                nr_hint,
                                 md_t                kc_hint,
                                 md_t                c_downscale)
{
    if (kDType == dlp::kernel_frame::kernelDatatype::f32f32f32of32) {
        return dlp::de::decisionEngineInstance()
            .getGemmKernelInfoForInputFastPath<dlp::de::gemmF32DEBackend>(
                m, n, k, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, alpha, beta,
                mtag_a, mtag_b, metadata, mr_hint, nr_hint, kc_hint,
                c_downscale, kernelRoutineType::gemm, kDType);
    } else if (kDType == dlp::kernel_frame::kernelDatatype::f16f16f16of16) {
        return dlp::de::decisionEngineInstance()
            .getGemmKernelInfoForInputFastPath<dlp::de::gemmFP16DEBackend>(
                m, n, k, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, alpha, beta,
                mtag_a, mtag_b, metadata, mr_hint, nr_hint, kc_hint,
                c_downscale, kernelRoutineType::gemm, kDType);
    } else if (kDType == dlp::kernel_frame::kernelDatatype::f16f16f16of32) {
        // Of32: GEMV-shaped (m=1 or n=1) inputs route through the
        // dedicated FP16 GEMV kernels which carry a c_downscale-aware F32
        // store rail (beta-combine + F32 store directly to the user's
        // float* C). The framework half of the contract lives in
        // dlp_gemv_rowvar_f16f16f16of16 (type-aware C-pointer arithmetic
        // via c_elem_size and beta widened with fp16_to_f32).
        return dlp::de::decisionEngineInstance()
            .getGemmKernelInfoForInputFastPath<dlp::de::gemmFP16DEBackend>(
                m, n, k, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, alpha, beta,
                mtag_a, mtag_b, metadata, mr_hint, nr_hint, kc_hint,
                c_downscale, kernelRoutineType::gemm, kDType);
    } else if (kDType == dlp::kernel_frame::kernelDatatype::f32f16f32of32) {
        // F32×FP16→F32 mixed-precision: uses separate backend (no avx512fp16
        // requirement)
        return dlp::de::decisionEngineInstance()
            .getGemmKernelInfoForInputFastPath<dlp::de::gemmF32FP16DEBackend>(
                m, n, k, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, alpha, beta,
                mtag_a, mtag_b, metadata, mr_hint, nr_hint, kc_hint,
                c_downscale, kernelRoutineType::gemm, kDType);
    } else if ((kDType == dlp::kernel_frame::kernelDatatype::bf16bf16f32obf16)
               || (kDType
                   == dlp::kernel_frame::kernelDatatype::bf16bf16f32of32)) {
        return dlp::de::decisionEngineInstance()
            .getGemmKernelInfoForInputFastPath<dlp::de::gemmBF16DEBackend>(
                m, n, k, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, alpha, beta,
                mtag_a, mtag_b, metadata, mr_hint, nr_hint, kc_hint,
                c_downscale, kernelRoutineType::gemm, kDType);
    } else if ((kDType == dlp::kernel_frame::kernelDatatype::u8s8s32os32)
               || (kDType == dlp::kernel_frame::kernelDatatype::u8s8s32of32)
               || (kDType == dlp::kernel_frame::kernelDatatype::u8s8s32of16)
               || (kDType == dlp::kernel_frame::kernelDatatype::u8s8s32obf16)
               || (kDType == dlp::kernel_frame::kernelDatatype::u8s8s32os8)
               || (kDType == dlp::kernel_frame::kernelDatatype::u8s8s32ou8)) {
        return dlp::de::decisionEngineInstance()
            .getGemmKernelInfoForInputFastPath<dlp::de::gemmU8S8DEBackend>(
                m, n, k, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, alpha, beta,
                mtag_a, mtag_b, metadata, mr_hint, nr_hint, kc_hint,
                c_downscale, kernelRoutineType::gemm, kDType);
    } else if ((kDType == dlp::kernel_frame::kernelDatatype::s8s8s32os32)
               || (kDType == dlp::kernel_frame::kernelDatatype::s8s8s32of32)
               || (kDType == dlp::kernel_frame::kernelDatatype::s8s8s32of16)
               || (kDType == dlp::kernel_frame::kernelDatatype::s8s8s32obf16)
               || (kDType == dlp::kernel_frame::kernelDatatype::s8s8s32os8)
               || (kDType == dlp::kernel_frame::kernelDatatype::s8s8s32ou8)) {
        return dlp::de::decisionEngineInstance()
            .getGemmKernelInfoForInputFastPath<dlp::de::gemmS8DEBackend>(
                m, n, k, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, alpha, beta,
                mtag_a, mtag_b, metadata, mr_hint, nr_hint, kc_hint,
                c_downscale, kernelRoutineType::gemm, kDType);
    } else {
        return dlp::kernel_frame::kernelInfo();
    }
}

// This function is not in the hot path, and therefore not inlined to avoid
// unnecessary hot path code bloat.
[[gnu::noinline]]
static dlp::kernel_frame::kernelBaseRef
dlp_generate_jit_kernel(dlp::kernel_frame::kernelInfo& fastKI,
                        kernelDatatype                 kDType)
{
    // First check if the kernel is already present in the fallback table
    // in kernel register.
    auto fallKernPtr =
        dlpKernelRegisterInstance().getGemmKernelFallback(&fastKI, kDType);
    if (fallKernPtr) {
        return fallKernPtr;
    }

    auto jitGen = dlpJitGeneratorRegisterInstance().getGemmJitGenerator(kDType);
    auto kB =
        std::make_unique<jitKernelAdapter>(fastKI, std::move(jitGen), true);

    if (!kB->isJitGenerated()) {
        // Register a dummy kernel that will be used to denote a
        // jit kernel cannot be generated for this kernelInfo.
        dlpKernelRegisterInstance().registerEmptyGemmKernel(fastKI, kDType);
    } else {
        // Generate datatype-specific kernel name for proper registry
        // management.
        std::string kernelName = get_kernel_family_name(kDType);
        auto        retVal     = dlpKernelRegisterInstance().registerGemmKernel(
            std::move(kB), std::move(kernelName));
        if (retVal != kernelFrameError::success) {
            std::cerr << "Kernel table insertion failed for datatype: "
                      << static_cast<int>(kDType) << ". Fatal Error."
                      << std::endl;
        }
    }

    auto kernPtr = dlpKernelRegisterInstance().getGemmKernel(&fastKI, kDType);
    if (kernPtr.isValid()) {
        return kernPtr;
    } else {
        // The fallback is guaranteed to work at this point.
        return dlpKernelRegisterInstance().getGemmKernelFallback(&fastKI,
                                                                 kDType);
    }
}

// Do NOT add likely/unlikely hints or __builtin_expect to the if-conditions
// in this function, even though the error paths are rare. [[gnu::flatten]]
// and [[gnu::aligned(64)]] attributes create a specific code layout optimized
// for instruction cache locality. Adding branch hints causes the compiler to
// move "unlikely" code out-of-line, which fragments hot path across multiple
// cache lines and degrades performance despite the branches being perfectly
// predicted. Modern branch predictors achieve >99% accuracy on these
// conditions after warmup, so prediction hints provide zero benefit while the
// code layout destruction causes measurable harm.
[[gnu::flatten]] [[gnu::aligned(64)]]
void
dlp_init_and_get_kernel_hndl(kernel_datatype_t     k_dtype,
                             [[maybe_unused]] char storage_format,
                             AOCL_DLP_MEMORY_TAG   mtag_a,
                             AOCL_DLP_MEMORY_TAG   mtag_b,
                             md_t                  m,
                             md_t                  n,
                             md_t                  k,
                             md_t                  rs_a,
                             md_t                  cs_a,
                             md_t                  rs_b,
                             md_t                  cs_b,
                             md_t                  rs_c,
                             md_t                  cs_c,
                             void*                 alpha,
                             void*                 beta,
                             dlp_gemm_post_op*     metadata,
                             md_t                  mr_hint,
                             md_t                  nr_hint,
                             md_t                  kc_hint,
                             md_t                  c_downscale,
                             dlp_kernel_hndl_t*    kernel_hndl)
{
    if (!kernel_hndl) {
        return;
    }

    kernelDatatype kDType = getKernelDatatype(k_dtype);
    if (kDType == kernelDatatype::invalid) {
        kernel_hndl->kernel_base = nullptr;
        return;
    }

    dlp::kernel_frame::kernelInfo fastKI = dlp_get_gemm_kernelInfo_by_dtype(
        kDType, m, n, k, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, alpha, beta,
        mtag_a, mtag_b, metadata, mr_hint, nr_hint, kc_hint, c_downscale);

    if ((fastKI.mr <= 0) || (fastKI.nr <= 0)) {
        kernel_hndl->kernel_base = nullptr;
        return;
    }

    auto kernPtr = dlpKernelRegisterInstance().getGemmKernel(&fastKI, kDType);

    if (!kernPtr) {
        kernPtr = dlp_generate_jit_kernel(fastKI, kDType);
    }

    kernel_hndl->kernel_base = (kernPtr.isValid() && kernPtr.getPtr()->isValid)
                                   ? static_cast<void*>(kernPtr.getPtr())
                                   : nullptr;
    kernel_hndl->mr          = fastKI.mr;
    kernel_hndl->nr          = fastKI.nr;
    kernel_hndl->kDtype      = k_dtype;
    kernel_hndl->invokeRD    = fastKI.invokeRD;
}

[[gnu::noinline]] static dlp::kernel_frame::kernelBaseRef
dlp_generate_packb_jit_kernel(dlp::kernel_frame::packKernelInfo& packKI,
                              kernelDatatype                     kDType)
{
    auto jitGen =
        dlpJitGeneratorRegisterInstance().getPackBJitGenerator(kDType);
    if (!jitGen) {
        dlpKernelRegisterInstance().registerEmptyPackBKernel(packKI, kDType);
        return dlp::kernel_frame::kernelBaseRef(nullptr);
    }

    auto kB =
        std::make_unique<jitKernelAdapter>(packKI, std::move(jitGen), true);

    if (!kB->isJitGenerated()) {
        dlpKernelRegisterInstance().registerEmptyPackBKernel(packKI, kDType);
    } else {
        std::string kernelName = get_kernel_family_name(kDType);
        auto        retVal = dlpKernelRegisterInstance().registerPackBKernel(
            std::move(kB), std::move(kernelName));
        if (retVal != kernelFrameError::success) {
            std::cerr << "PackB JIT kernel registration failed for datatype: "
                      << static_cast<int>(kDType) << std::endl;
        }
    }

    return dlpKernelRegisterInstance().getPackBKernel(&packKI, kDType);
}

DLP_ALWAYS_INLINE static dlp::kernel_frame::packKernelInfo
dlp_get_packb_kernelInfo_by_dtype(
    kernelDatatype kDType, md_t nc, md_t kc, md_t cs_src, md_t nr_hint)
{
    if (kDType == kernelDatatype::f32f32f32of32) {
        return dlp::de::decisionEngineInstance()
            .getGemmPackBInfoForInputFastPath<dlp::de::gemmF32DEBackend>(
                nc, kc, cs_src, nr_hint, kDType);
    }

    return dlp::kernel_frame::packKernelInfo();
}

void
dlp_init_and_get_packb_kernel_hndl(kernel_datatype_t     k_dtype,
                                   md_t                  nc,
                                   md_t                  kc,
                                   md_t                  rs_src,
                                   md_t                  cs_src,
                                   md_t                  nr_hint,
                                   dlp_pack_info_hndl_t* kernel_hndl)
{
    (void)rs_src;

    if (!kernel_hndl) {
        return;
    }

    kernelDatatype kDType = getKernelDatatype(k_dtype);
    if (kDType == kernelDatatype::invalid) {
        kernel_hndl->kernel_base = nullptr;
        return;
    }

    dlp::kernel_frame::packKernelInfo packKI =
        dlp_get_packb_kernelInfo_by_dtype(kDType, nc, kc, cs_src, nr_hint);

    if (packKI.panel_dim <= 0) {
        kernel_hndl->kernel_base = nullptr;
        return;
    }

    auto kernPtr = dlpKernelRegisterInstance().getPackBKernel(&packKI, kDType);

    if (!kernPtr) {
        kernPtr = dlp_generate_packb_jit_kernel(packKI, kDType);
    }

    auto* rawPtr             = (kernPtr.isValid() && kernPtr.getPtr()->isValid)
                                   ? kernPtr.getPtr()
                                   : nullptr;
    kernel_hndl->kernel_base = static_cast<void*>(rawPtr);
    kernel_hndl->panel_dim   = packKI.panel_dim;
    kernel_hndl->k_factor    = packKI.k_factor;
    kernel_hndl->kDtype      = k_dtype;
    kernel_hndl->src_type    = static_cast<uint8_t>(packKI.src_type);
    kernel_hndl->dst_type    = static_cast<uint8_t>(packKI.dst_type);
}

[[gnu::aligned(64)]] void
dlp_execute_packb_kernel(dlp_pack_info_hndl_t kernel_hndl,
                         void*                src,
                         void*                dst,
                         md_t                 n,
                         md_t                 k,
                         md_t                 rs_src,
                         md_t                 cs_src,
                         md_t*                rs_dst,
                         md_t*                cs_dst)
{
    if (kernel_hndl.kernel_base == nullptr) {
        return;
    }

    packBParams packBParamsIn(src, dst, n, k, rs_src, cs_src);

    kernelBase* kB = static_cast<kernelBase*>(kernel_hndl.kernel_base);
    kB->operator()(std::addressof(packBParamsIn));

    if (rs_dst != nullptr) {
        *rs_dst = packBParamsIn.rs_dst;
    }
    if (cs_dst != nullptr) {
        *cs_dst = packBParamsIn.cs_dst;
    }
}

// Experimentally derived alignment, needs further analysis but gives
// consistent good performance on zen5 machines.
[[gnu::aligned(64)]]
// Force inlining of dlp_execute_kernel to ensure optimal performance,
// especially when building with Link Time Optimization (LTO). Without the
// always_inline attribute, some compilers may not inline this function even
// with LTO enabled, which can lead to suboptimal performance in tiny shape
// scenarios. Explicitly marking this function as always_inline guarantees that
// the optimizer can inline it as intended when LTO is enabled. Note: With LLVM
// 19, this attribute has no effect unless LTO is enabled; in non-LTO builds,
// the compiler may still choose not to inline this function.
#if defined(__clang__) && __clang_major__ >= 19
__attribute__((always_inline))
#endif
void
dlp_execute_kernel(dlp_kernel_hndl_t*    kernel_hndl,
                   md_t                  m,
                   md_t                  n,
                   md_t                  k,
                   void*                 A,
                   md_t                  rs_a,
                   md_t                  cs_a,
                   md_t                  ps_a,
                   void*                 B,
                   md_t                  rs_b,
                   md_t                  cs_b,
                   md_t                  n_sub_updated,
                   md_t                  jc_cur_loop_rem,
                   void*                 C,
                   md_t                  rs_c,
                   md_t                  cs_c,
                   void*                 alpha,
                   void*                 beta,
                   dlp_gemm_post_op*     post_ops_list,
                   dlp_gemm_post_op_attr post_ops_attr)
{
    if (!kernel_hndl || !kernel_hndl->kernel_base) {
        return;
    }

    // Dont use new/delete and malloc/free calls here, since they are lock
    // based and will result in performance degradation.
    if (kernel_hndl->mr == 1) {
        gemvM1Params gemvM1ParamsIn{ A,
                                     B,
                                     C,
                                     n,
                                     k,
                                     rs_a,
                                     cs_a,
                                     rs_b,
                                     cs_b,
                                     rs_c,
                                     cs_c,
                                     n_sub_updated,
                                     jc_cur_loop_rem,
                                     alpha,
                                     beta,
                                     post_ops_list,
                                     post_ops_attr };
        kernelBase*  kB = static_cast<kernelBase*>(kernel_hndl->kernel_base);
        kB->operator()(std::addressof(gemvM1ParamsIn));
    } else if (kernel_hndl->nr == 1) {
        gemvN1Params gemvN1ParamsIn{
            A,    B,    C,    m,     k,    rs_a,          cs_a,         rs_b,
            cs_b, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
        };

        kernelBase* kB = static_cast<kernelBase*>(kernel_hndl->kernel_base);
        kB->operator()(std::addressof(gemvN1ParamsIn));
    } else {
        gemmParams gemmParamsIn{ A,
                                 B,
                                 C,
                                 m,
                                 n,
                                 k,
                                 rs_a,
                                 cs_a,
                                 ps_a,
                                 rs_b,
                                 cs_b,
                                 rs_c,
                                 cs_c,
                                 alpha,
                                 beta,
                                 post_ops_list,
                                 post_ops_attr };

        kernelBase* kB = static_cast<kernelBase*>(kernel_hndl->kernel_base);
        kB->operator()(std::addressof(gemmParamsIn));
    }

    return;
}
