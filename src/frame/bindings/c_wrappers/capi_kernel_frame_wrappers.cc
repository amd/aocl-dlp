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
#include "kernel_frame/kernel_frame_base.hh"
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
dlp_get_gemm_kernelInfo_by_dtype(kernelDatatype                 kDType,
                                 md_t                           m,
                                 md_t                           n,
                                 md_t                           k,
                                 md_t                           rs_a,
                                 md_t                           cs_a,
                                 md_t                           rs_b,
                                 md_t                           cs_b,
                                 md_t                           rs_c,
                                 md_t                           cs_c,
                                 void*                          alpha,
                                 void*                          beta,
                                 AOCL_DLP_MEMORY_TAG            mtag_a,
                                 AOCL_DLP_MEMORY_TAG            mtag_b,
                                 dlp_gemm_post_op*              metadata,
                                 md_t                           mr_hint,
                                 md_t                           nr_hint,
                                 md_t                           kc_hint,
                                 md_t                           c_downscale,
                                 dlp_gemm_thread_info_t*        thread_info,
                                 const dlp_gemm_kernel_hints_t* gemm_hints,
                                 md_t                           blksz_set_mask)
{
    if (kDType == dlp::kernel_frame::kernelDatatype::f32f32f32of32) {
        return dlp::de::decisionEngineInstance()
            .getGemmKernelInfoForInputFastPath<dlp::de::gemmF32DEBackend>(
                m, n, k, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, alpha, beta,
                mtag_a, mtag_b, metadata, mr_hint, nr_hint, kc_hint,
                c_downscale, thread_info, gemm_hints, blksz_set_mask,
                kernelRoutineType::gemm, kDType);
    } else if (kDType == dlp::kernel_frame::kernelDatatype::f16f16f16of16) {
        return dlp::de::decisionEngineInstance()
            .getGemmKernelInfoForInputFastPath<dlp::de::gemmFP16DEBackend>(
                m, n, k, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, alpha, beta,
                mtag_a, mtag_b, metadata, mr_hint, nr_hint, kc_hint,
                c_downscale, thread_info, gemm_hints, blksz_set_mask,
                kernelRoutineType::gemm, kDType);
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
                c_downscale, thread_info, gemm_hints, blksz_set_mask,
                kernelRoutineType::gemm, kDType);
    } else if (kDType == dlp::kernel_frame::kernelDatatype::f32f16f32of32) {
        // F32×FP16→F32 mixed-precision: uses separate backend (no avx512fp16
        // requirement)
        return dlp::de::decisionEngineInstance()
            .getGemmKernelInfoForInputFastPath<dlp::de::gemmF32FP16DEBackend>(
                m, n, k, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, alpha, beta,
                mtag_a, mtag_b, metadata, mr_hint, nr_hint, kc_hint,
                c_downscale, thread_info, gemm_hints, blksz_set_mask,
                kernelRoutineType::gemm, kDType);
    } else if ((kDType == dlp::kernel_frame::kernelDatatype::bf16bf16f32obf16)
               || (kDType
                   == dlp::kernel_frame::kernelDatatype::bf16bf16f32of32)) {
        return dlp::de::decisionEngineInstance()
            .getGemmKernelInfoForInputFastPath<dlp::de::gemmBF16DEBackend>(
                m, n, k, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, alpha, beta,
                mtag_a, mtag_b, metadata, mr_hint, nr_hint, kc_hint,
                c_downscale, thread_info, gemm_hints, blksz_set_mask,
                kernelRoutineType::gemm, kDType);
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
                c_downscale, thread_info, gemm_hints, blksz_set_mask,
                kernelRoutineType::gemm, kDType);
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
                c_downscale, thread_info, gemm_hints, blksz_set_mask,
                kernelRoutineType::gemm, kDType);
    } else {
        return dlp::kernel_frame::kernelInfo();
    }
}

// This function is not in the hot path, and therefore not inlined to avoid
// unnecessary hot path code bloat.
[[gnu::noinline]] static dlp::kernel_frame::kernelBaseRef
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

// Strides of the packed A and B buffers, a function of the tile and the number
// of elements the kernel's dot-product instruction consumes per step: 4 int8
// for VNNI, 2 bf16 for DPBF16PS, 1 for the float kernels.
//
// Not derived when application metadata is merged into the context, because
// that happens before the DE has chosen a tile. packb_rs is the row stride of
// the panel the pack kernel is about to write; a value derived against a
// different NR than the one selected does not fail, it returns wrong numbers.
// The caller runs this once its inits are done -- MR is settled by the kernel
// init, NR by whichever of the two ran last -- which is what keeps the strides
// and the panel in agreement.
void
dlp_upd_pack_strides(kernel_datatype_t k_dtype, dlp_gemm_cntx_t* cntx)
{
    if (!cntx) {
        return;
    }

    constexpr md_t cache_line_size = 64;

    const kernelDatatype kDType = getKernelDatatype(k_dtype);

    const md_t mr = cntx->blksz.MR;
    const md_t nr = cntx->blksz.NR;

    md_t elems_per_step = 0;
    md_t packb_cs       = 0;

    switch (kDType) {
        case kernelDatatype::u8s8s32os32:
        case kernelDatatype::u8s8s32of32:
        case kernelDatatype::u8s8s32of16:
        case kernelDatatype::u8s8s32obf16:
        case kernelDatatype::u8s8s32ou8:
        case kernelDatatype::u8s8s32os8:
        case kernelDatatype::s8s8s32ou8:
        case kernelDatatype::s8s8s32os8:
        case kernelDatatype::s8s8s32obf16:
        case kernelDatatype::s8s8s32of32:
        case kernelDatatype::s8s8s32of16:
        case kernelDatatype::s8s8s32os32:
            elems_per_step = 4;
            packb_cs       = cache_line_size / (md_t)sizeof(int8_t);
            break;

        case kernelDatatype::bf16bf16f32obf16:
        case kernelDatatype::bf16bf16f32of32:
            elems_per_step = 2;
            packb_cs       = cache_line_size / (md_t)sizeof(int16_t);
            break;

        case kernelDatatype::f32f32f32of32:
        case kernelDatatype::f16f16f16of16:
        case kernelDatatype::f16f16f16of32:
        case kernelDatatype::f32f16f32of32:
            elems_per_step = 1;
            packb_cs       = 1;
            break;

        default:
            // No kernel, so nothing reads these. Leave the defaults alone.
            return;
    }

    cntx->pack_s.packa_rs = elems_per_step;
    cntx->pack_s.packa_cs = elems_per_step * mr;
    cntx->pack_s.packb_rs = elems_per_step * nr;
    cntx->pack_s.packb_cs = packb_cs;
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
[[gnu::flatten]] [[gnu::aligned(64)]] void
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
                             dlp_gemm_cntx_t*      cntx,
                             md_t                  c_downscale)
{
    if (!cntx) {
        return;
    }

    kernelDatatype kDType = getKernelDatatype(k_dtype);
    if (kDType == kernelDatatype::invalid) {
        cntx->dlp_kernel_hndl.kernel_base = nullptr;
        return;
    }

    // Nothing here tells the optimizer whether to run. It works that out from
    // blksz_set_mask, which already records which block sizes the application
    // authored and which an earlier init settled. Either way those are not the
    // library's to choose.
    //
    // A reordered B needs one more thing. The model has to reproduce a width
    // the panel already has, and it can only do that from the hint pair. If
    // either hint is zero it is ineligible, and both ends stay on the context
    // NR.
    dlp::kernel_frame::kernelInfo fastKI = dlp_get_gemm_kernelInfo_by_dtype(
        kDType, m, n, k, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, alpha, beta,
        mtag_a, mtag_b, metadata, cntx->blksz.MR, cntx->blksz.NR,
        cntx->blksz.KC, c_downscale, &cntx->thread_info,
        &cntx->gemm_kernel_hints, cntx->blksz_set_mask);

    if ((fastKI.mr <= 0) || (fastKI.nr <= 0)) {
        cntx->dlp_kernel_hndl.kernel_base = nullptr;
        return;
    }

    auto kernPtr = dlpKernelRegisterInstance().getGemmKernel(&fastKI, kDType);

    if (!kernPtr) {
        kernPtr = dlp_generate_jit_kernel(fastKI, kDType);
    }

    cntx->dlp_kernel_hndl.kernel_base =
        (kernPtr.isValid() && kernPtr.getPtr()->isValid)
            ? static_cast<void*>(kernPtr.getPtr())
            : nullptr;
    cntx->blksz.MR = cntx->dlp_kernel_hndl.mr = fastKI.mr;
    cntx->blksz.NR = cntx->dlp_kernel_hndl.nr = fastKI.nr;
    cntx->blksz.KC                            = fastKI.kc;

    // The tile is settled for the rest of this call, so record it the same way
    // an application-authored tile is recorded. That is what a pack-B init
    // later in this call reads to know the width is not its to choose, without
    // either side having to know which of them runs first. The context is a
    // per-call copy, so this does not reach the next call.
    cntx->blksz_set_mask |= DLP_BLKSZ_SET_MR | DLP_BLKSZ_SET_NR;
    // Snap MC and NC onto the chosen tile unconditionally, including where the
    // application authored them: metadata may state MC and NC while leaving MR
    // and NR to the library, and the pair it stated need not divide the tile
    // the DE picked. What reaches the five-loop has to fit the kernel.
    //
    // Whether the snap was acceptable is the validator's call --
    // dlp_gemm_validate_metadata_with_lcntx rejects a blksz that no longer
    // matches what was asked for. Snapping is what makes that check fire on
    // the stated value rather than on a weaker divisibility predicate, and it
    // is a no-op wherever the stated value already divides the tile.
    cntx->blksz.MC = ((cntx->blksz.MC + fastKI.mr - 1) / fastKI.mr) * fastKI.mr;
    cntx->blksz.NC = ((cntx->blksz.NC + fastKI.nr - 1) / fastKI.nr) * fastKI.nr;

    // The context now carries the block sizes decided above. The thread
    // factorization in cntx->thread_info is not decided here. The shape model
    // hands the caller's ways back as it found them, and the threading
    // decorator derives the split later, against this call's real extents and
    // the tile just chosen.

    cntx->dlp_kernel_hndl.kDtype   = k_dtype;
    cntx->dlp_kernel_hndl.invokeRD = fastKI.invokeRD;
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
dlp_get_packb_kernelInfo_by_dtype(kernelDatatype                 kDType,
                                  md_t                           nc,
                                  md_t                           cs_src,
                                  md_t                           n,
                                  md_t                           k,
                                  md_t                           mr_hint,
                                  md_t                           nr_hint,
                                  md_t                           blksz_set_mask,
                                  const dlp_gemm_kernel_hints_t* gemm_hints)
{
    if (kDType == kernelDatatype::f32f32f32of32) {
        return dlp::de::decisionEngineInstance()
            .getGemmPackBInfoForInputFastPath<dlp::de::gemmF32DEBackend>(
                nc, cs_src, n, k, mr_hint, nr_hint, blksz_set_mask, gemm_hints,
                kDType);
    }

    else if (kDType == kernelDatatype::bf16bf16f32of32
             || kDType == kernelDatatype::bf16bf16f32obf16) {
        return dlp::de::decisionEngineInstance()
            .getGemmPackBInfoForInputFastPath<dlp::de::gemmBF16DEBackend>(
                nc, cs_src, n, k, mr_hint, nr_hint, blksz_set_mask, gemm_hints,
                kDType);
    }

    return dlp::kernel_frame::packKernelInfo();
}

void
dlp_init_and_get_packb_kernel_hndl(kernel_datatype_t k_dtype,
                                   md_t              n,
                                   md_t              k,
                                   md_t              rs_src,
                                   md_t              cs_src,
                                   dlp_gemm_cntx_t*  cntx)
{
    (void)rs_src;

    if (!cntx) {
        return;
    }

    dlp_pack_info_hndl_t* b_hndl =
        std::addressof((cntx->dlp_pack_kernel_hndl).pack_b_hndl);

    kernelDatatype kDType = getKernelDatatype(k_dtype);
    if (kDType == kernelDatatype::invalid) {
        b_hndl->kernel_base = nullptr;
        return;
    }

    // Currently n is passed in place of nc. Need to revisit this when there
    // is clarity on what is required.
    md_t nc = n;

    // The context fields the model is a function of go down as they stand, and
    // the backend assembles them, exactly as the kernel init above hands its
    // own down for the same treatment. The hints are the only description
    // available here of the GEMM this panel is being prepared for, and a later
    // GEMM over this buffer hands the same pair in turn. A zero in either
    // leaves the model ineligible and the context NR standing, which is still a
    // width both ends agree on.
    dlp::kernel_frame::packKernelInfo packKI =
        dlp_get_packb_kernelInfo_by_dtype(
            kDType, nc, cs_src, n, k, cntx->blksz.MR, cntx->blksz.NR,
            cntx->blksz_set_mask, &cntx->gemm_kernel_hints);

    if (packKI.panel_dim <= 0) {
        b_hndl->kernel_base = nullptr;
        return;
    }

    auto kernPtr = dlpKernelRegisterInstance().getPackBKernel(&packKI, kDType);

    if (!kernPtr) {
        kernPtr = dlp_generate_packb_jit_kernel(packKI, kDType);
    }

    auto* rawPtr        = (kernPtr.isValid() && kernPtr.getPtr()->isValid)
                              ? kernPtr.getPtr()
                              : nullptr;
    b_hndl->kernel_base = static_cast<void*>(rawPtr);
    cntx->blksz.NR = b_hndl->panel_dim = packKI.panel_dim;

    // The width is settled, whether this init chose it or inherited it from a
    // kernel init earlier in the call. Recording it is what lets a kernel init
    // that runs after this one honour it instead of deriving its own.
    //
    // Only NR, though the model returns the pair and the MR is there for the
    // taking. NR is the half that is layout-bearing, so it is the half this
    // init has standing to settle; MR is a per-call compute choice, and on the
    // Reorder path there is no call yet to make it for. Writing one would also
    // put this path on the hook for snapping MC onto it -- the validator
    // requires MC % MR -- and a pack-B init has no business rewriting the
    // blocking parameters.
    cntx->blksz_set_mask |= DLP_BLKSZ_SET_NR;
    // Always round NC to multiples of panel_dim (NR) for PackB kernel.
    cntx->blksz.NC =
        ((cntx->blksz.NC + packKI.panel_dim - 1) / packKI.panel_dim)
        * packKI.panel_dim;
    b_hndl->k_factor = packKI.k_factor;
    b_hndl->kDtype   = k_dtype;
    b_hndl->src_type = static_cast<uint8_t>(packKI.src_type);
    b_hndl->dst_type = static_cast<uint8_t>(packKI.dst_type);
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

    // Don't use new/delete and malloc/free calls here, since they are lock
    // based and will result in performance degradation.
    // Extra m==1 check to ensure the mr=1 kernel used is intended for
    // GEMV-shaped inputs, not a tiny shape GEMM. Similarly for nr=1.
    if ((kernel_hndl->mr == 1) && (m == 1)) {
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
    } else if ((kernel_hndl->nr == 1) && (n == 1)) {
        gemvN1Params gemvN1ParamsIn{
            A,    B,    C,    m,     k,    rs_a,          cs_a,         rs_b,
            cs_b, rs_c, cs_c, alpha, beta, post_ops_list, post_ops_attr
        };

        kernelBase* kB = static_cast<kernelBase*>(kernel_hndl->kernel_base);
        kB->operator()(std::addressof(gemvN1ParamsIn));
    } else {
        // NOTE: For F32 GEMM kernel this carries ps_b (B panel/column stride).
        // To be revisited later with a more comprehensive fix.
        md_t       ps_b = n_sub_updated;
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
                                 ps_b,
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

DLP_ALWAYS_INLINE static quantKernelInfo
dlp_get_gemm_quant_kernelInfo_by_dtype(kernelDatatype      kDType,
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
                                       dlp_group_op*       group_ops,
                                       md_t                mr_hint,
                                       md_t                nr_hint,
                                       md_t                kc_hint,
                                       md_t                c_downscale)
{
    if ((kDType == kernelDatatype::s8s8s32of32)
        || (kDType == kernelDatatype::s8s8s32obf16)) {
        return dlp::de::decisionEngineInstance()
            .getGemmQuantKernelInfoForInputFastPath<
                dlp::de::gemmQuantS8DEBackend>(
                m, n, k, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, alpha, beta,
                mtag_a, mtag_b, metadata, group_ops, mr_hint, nr_hint, kc_hint,
                c_downscale, kDType);
    } else {
        return dlp::kernel_frame::quantKernelInfo();
    }
}

[[gnu::noinline]] static kernelBaseRef
dlp_generate_gemm_quant_jit_kernel(quantKernelInfo& qKI, kernelDatatype kDType)
{
    // First check if the kernel is already present in the fallback table
    // in kernel register.
    auto fallKernPtr =
        dlpKernelRegisterInstance().getGemmQuantKernelFallback(&qKI, kDType);
    if (fallKernPtr) {
        return fallKernPtr;
    }

    auto jitGen =
        dlpJitGeneratorRegisterInstance().getGemmQuantJitGenerator(kDType);
    if (!jitGen) {
        dlpKernelRegisterInstance().registerEmptyGemmQuantKernel(qKI, kDType);
        return kernelBaseRef(nullptr);
    }

    auto kB = std::make_unique<jitKernelAdapter>(qKI, std::move(jitGen), true);

    if (!kB->isJitGenerated()) {
        // Register a dummy kernel that will be used to denote a
        // jit kernel cannot be generated for this kernelInfo.
        dlpKernelRegisterInstance().registerEmptyGemmQuantKernel(qKI, kDType);
    } else {
        // Generate datatype-specific kernel name for proper registry
        // management.
        std::string kernelName = get_kernel_family_name(kDType);
        auto retVal = dlpKernelRegisterInstance().registerGemmQuantKernel(
            std::move(kB), std::move(kernelName));
        if (retVal != kernelFrameError::success) {
            std::cerr << "Quant kernel table insertion failed for datatype: "
                      << static_cast<int>(kDType) << ". Fatal Error."
                      << std::endl;
        }
    }

    auto kernPtr = dlpKernelRegisterInstance().getGemmQuantKernel(&qKI, kDType);
    if (kernPtr.isValid()) {
        return kernPtr;
    }

    // The fallback is guaranteed to work at this point.
    return dlpKernelRegisterInstance().getGemmQuantKernelFallback(&qKI, kDType);
}

void
dlp_init_and_get_gemm_quant_kernel_hndl(kernel_datatype_t     k_dtype,
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
                                        dlp_group_op*         group_ops,
                                        dlp_gemm_cntx_t*      cntx,
                                        md_t                  c_downscale)
{
    if (!cntx)
        return;

    kernelDatatype kDType = getKernelDatatype(k_dtype);
    if (kDType == kernelDatatype::invalid) {
        cntx->dlp_quant_kernel_hndl.kernel_base = nullptr;
        return;
    }

    dlp::kernel_frame::quantKernelInfo qKI =
        dlp_get_gemm_quant_kernelInfo_by_dtype(
            kDType, m, n, k, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, alpha, beta,
            mtag_a, mtag_b, metadata, group_ops, cntx->blksz.MR, cntx->blksz.NR,
            cntx->blksz.KC, c_downscale);

    if ((qKI.base.mr <= 0) || (qKI.base.nr <= 0)) {
        cntx->dlp_quant_kernel_hndl.kernel_base = nullptr;
        return;
    }

    auto kernPtr = dlpKernelRegisterInstance().getGemmQuantKernel(&qKI, kDType);
    if (!kernPtr) {
        kernPtr = dlp_generate_gemm_quant_jit_kernel(qKI, kDType);
    }

    cntx->dlp_quant_kernel_hndl.kernel_base =
        (kernPtr.isValid() && kernPtr.getPtr()->isValid)
            ? static_cast<void*>(kernPtr.getPtr())
            : nullptr;

    cntx->dlp_quant_kernel_hndl.mr     = qKI.base.mr;
    cntx->dlp_quant_kernel_hndl.nr     = qKI.base.nr;
    cntx->dlp_quant_kernel_hndl.kDtype = k_dtype;
    cntx->blksz.KC                     = qKI.base.kc;
    cntx->blksz.MC                     = ((cntx->blksz.MC % qKI.base.mr) == 0)
                                             ? cntx->blksz.MC
                                             : (((cntx->blksz.MC + qKI.base.mr - 1) / qKI.base.mr)
                            * qKI.base.mr);
    cntx->blksz.NC                     = ((cntx->blksz.NC % qKI.base.nr) == 0)
                                             ? cntx->blksz.NC
                                             : (((cntx->blksz.NC + qKI.base.nr - 1) / qKI.base.nr)
                            * qKI.base.nr);
    cntx->dlp_quant_kernel_hndl.kDtype = k_dtype;
}

void
dlp_execute_gemm_quant_kernel(dlp_gemm_quant_kernel_hndl_t* kernel_hndl,
                              md_t                          m,
                              md_t                          n,
                              md_t                          k,
                              void*                         A,
                              md_t                          rs_a,
                              md_t                          cs_a,
                              md_t                          ps_a,
                              void*                         B,
                              md_t                          rs_b,
                              md_t                          cs_b,
                              md_t                          n_sub_updated,
                              md_t                          jc_cur_loop_rem,
                              void*                         C,
                              md_t                          rs_c,
                              md_t                          cs_c,
                              void*                         alpha,
                              void*                         beta,
                              dlp_gemm_post_op*             post_ops_list,
                              dlp_gemm_post_op_attr         post_ops_attr,
                              dlp_gemm_grp_post_op_attr     grp_post_ops_attr)
{
    if (!kernel_hndl || !kernel_hndl->kernel_base) {
        return;
    }

    kernelBase* kB = static_cast<kernelBase*>(kernel_hndl->kernel_base);

    // Don't use new/delete and malloc/free calls here, since they are lock
    // based and will result in performance degradation.
    // Extra m==1 check to ensure the mr=1 kernel used is intended for
    // GEMV-shaped inputs, not a tiny shape GEMM. Similarly for nr=1
    if ((kernel_hndl->mr == 1) && (m == 1)) {
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
                                     post_ops_attr,
                                     grp_post_ops_attr };
        kB->operator()(std::addressof(gemvM1ParamsIn));
    } else if ((kernel_hndl->nr == 1) && (n == 1)) {
        gemvN1Params gemvN1ParamsIn{ A,
                                     B,
                                     C,
                                     m,
                                     k,
                                     rs_a,
                                     cs_a,
                                     rs_b,
                                     cs_b,
                                     rs_c,
                                     cs_c,
                                     alpha,
                                     beta,
                                     post_ops_list,
                                     post_ops_attr,
                                     grp_post_ops_attr };
        kB->operator()(std::addressof(gemvN1ParamsIn));
    } else {
        // The quant GEMM generator never reads psB, and the sole GEMM caller
        // passes 0 for n_sub_updated, so this deliberately does not replicate
        // the ps_b = n_sub_updated aliasing dlp_execute_kernel carries.
        md_t       ps_b = 0;
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
                                 ps_b,
                                 rs_c,
                                 cs_c,
                                 alpha,
                                 beta,
                                 post_ops_list,
                                 post_ops_attr,
                                 grp_post_ops_attr };
        kB->operator()(std::addressof(gemmParamsIn));
    }

    return;
}
