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

#include <optional>
#include <set>
#include <vector>

#include "classic/dlp_macros.h"
#include "de_backend.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "utils/type_utils.hh"

namespace dlp::de {

/**
 * @brief Singleton decision engine for kernel selection and optimization
 *
 * decisionEngine manages kernel selection strategies using a 2D array of
 * backend implementations indexed by [routine_type][datatype]. Implements
 * Meyer's singleton pattern with thread-safe initialization. Each backend
 * provides specialized decision logic for optimal kernel selection based on
 * input characteristics.
 *
 * DESIGN PHILOSOPHY:
 * - 2D backend organization: routine_type × datatype
 * - Pluggable backend architecture for extensible decision strategies
 * - Centralized kernel selection logic with distributed backend implementations
 *
 * @note Currently supports F32 and BF16 GEMM operations, extensible for
 * additional datatypes and routine types
 */
class decisionEngine
{
    void registerDecisionEngine()
    {
        // Registering all the GEMM decision engines for the datatypes supported
        auto kTypeIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelRoutineType::gemm);
        auto f32DtIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelDatatype::f32f32f32of32);
        backends[kTypeIdx][f32DtIdx] = new gemmF32DEBackend;

        // Support BF16OF32 decision engine
        auto bf16of32DtIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelDatatype::bf16bf16f32of32);
        backends[kTypeIdx][bf16of32DtIdx] = new gemmBF16DEBackend;

        // Support BF16OF32 decision engine
        auto bf16obf16DtIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelDatatype::bf16bf16f32obf16);
        backends[kTypeIdx][bf16obf16DtIdx] = new gemmBF16DEBackend;

        // Support U8S8OS32 decision engine
        auto u8s8DtIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelDatatype::u8s8s32os32);
        backends[kTypeIdx][u8s8DtIdx] = new gemmU8S8DEBackend;

        // Support U8S8OF32 decision engine
        auto u8s8of32DtIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelDatatype::u8s8s32of32);
        backends[kTypeIdx][u8s8of32DtIdx] = new gemmU8S8DEBackend;

        auto u8s8of16DtIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelDatatype::u8s8s32of16);
        backends[kTypeIdx][u8s8of16DtIdx] = new gemmU8S8DEBackend;

        // Support U8S8OBF16 decision engine
        auto u8s8obf16DtIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelDatatype::u8s8s32obf16);
        backends[kTypeIdx][u8s8obf16DtIdx] = new gemmU8S8DEBackend;

        // Support U8S8OU8 decision engine
        auto u8s8ou8DtIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelDatatype::u8s8s32ou8);
        backends[kTypeIdx][u8s8ou8DtIdx] = new gemmU8S8DEBackend;

        auto u8s8os8DtIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelDatatype::u8s8s32os8);
        backends[kTypeIdx][u8s8os8DtIdx] = new gemmU8S8DEBackend;

        // Register S8 decision engine.
        auto s8s8s32os32DtIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelDatatype::s8s8s32os32);
        backends[kTypeIdx][s8s8s32os32DtIdx] = new gemmS8DEBackend;

        // Register s8s8s32os8 decision engine.
        auto s8s8s32os8DtIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelDatatype::s8s8s32os8);
        backends[kTypeIdx][s8s8s32os8DtIdx] = new gemmS8DEBackend;

        // Register s8s8s32obf16 decision engine.
        auto s8s8s32obf16DtIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelDatatype::s8s8s32obf16);
        backends[kTypeIdx][s8s8s32obf16DtIdx] = new gemmS8DEBackend;

        // Register s8s8s32ou8 decision engine.
        auto s8s8s32ou8DtIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelDatatype::s8s8s32ou8);
        backends[kTypeIdx][s8s8s32ou8DtIdx] = new gemmS8DEBackend;

        // Register s8s8s32of32 decision engine.
        auto s8s8s32of32DtIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelDatatype::s8s8s32of32);
        backends[kTypeIdx][s8s8s32of32DtIdx] = new gemmS8DEBackend;

        // Register s8s8s32of16 decision engine.
        auto s8s8s32of16DtIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelDatatype::s8s8s32of16);
        backends[kTypeIdx][s8s8s32of16DtIdx] = new gemmS8DEBackend;

        // Register FP16 decision engine. Both the of16 and of32 rails use
        // the same gemmFP16DEBackend class (same JIT generator, same kernel
        // selection), but each datatype gets its OWN instance so the
        // 2D-array-by-datatype dispatch model stays uniform across all
        // datatypes (cosmetic uniformity with the bf16/u8s8/s8 backends
        // which also instantiate per-datatype). The destructor uses a
        // pointer-set to avoid double-deletion.
        auto f16f16f16of16DtIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelDatatype::f16f16f16of16);
        backends[kTypeIdx][f16f16f16of16DtIdx] = new gemmFP16DEBackend;

        auto f16f16f16of32DtIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelDatatype::f16f16f16of32);
        backends[kTypeIdx][f16f16f16of32DtIdx] = new gemmFP16DEBackend;

        // Register F32×FP16→F32 mixed-precision decision engine
        auto f32f16f32of32DtIdx = utils::getUnderlyingValueOfEnum(
            kernel_frame::kernelDatatype::f32f16f32of32);
        backends[kTypeIdx][f32f16f32of32DtIdx] = new gemmF32FP16DEBackend;
    }

    decisionEngine()
    {
        backends.resize(
            static_cast<std::size_t>(utils::getUnderlyingValueOfEnum(
                dlp::kernel_frame::kernelRoutineType::max_kernel_routines)));

        for (auto& ele : backends) {
            ele.resize(
                static_cast<std::size_t>(utils::getUnderlyingValueOfEnum(
                    dlp::kernel_frame::kernelDatatype::max_kernel_datatypes)),
                nullptr);
        }

        registerDecisionEngine();
    }

    ~decisionEngine()
    {
        // Using a set to avoid deleting the same value multiple times.
        // This is required since the same VALUE_TYPE could be inserted
        // multiple times depending on its usage by the composing class.
        // For example, a backend could be registered multiple times with
        // different kernelDatatypes.
        std::set<iDEBackend*> valueSet;
        for (auto& ele : backends) {
            for (auto& ele2 : ele) {
                if ((ele2 != nullptr) && (valueSet.count(ele2) == 0)) {
                    valueSet.insert(ele2);
                    delete ele2;
                }
            }
        }
    }

    // Copy/move operations disabled for singleton
    decisionEngine(const decisionEngine&)            = delete;
    decisionEngine& operator=(const decisionEngine&) = delete;
    decisionEngine(decisionEngine&&)                 = delete;
    decisionEngine& operator=(decisionEngine&&)      = delete;

    // 2D array of backend implementations: backends[routine_type][datatype]
    std::vector<std::vector<iDEBackend*>> backends;

  public:
    /**
     * @brief Meyer's singleton instance accessor with thread-safe
     * initialization
     *
     * Returns reference to the single global decision engine instance.
     * Thread-safe initialization guaranteed by C++11 standard.
     *
     * THREAD SAFETY: Thread-safe initialization, concurrent access safe
     *
     * @return Reference to singleton decisionEngine instance
     */
    static decisionEngine& instance()
    {
        static decisionEngine de;
        return de;
    }

    /**
     * @brief Retrieves optimal kernel configuration for given input
     * characteristics
     *
     * Queries the appropriate backend implementation based on routine type and
     * datatype to determine optimal kernel parameters for the given input.
     * Returns kernelInfo containing recommended kernel dimensions and
     * configuration.
     *
     * THREAD SAFETY: Thread-safe if underlying backends are thread-safe
     *
     * @param in Pointer to input characteristics for kernel selection
     * @param kType Kernel routine type (e.g., GEMM, GEMV)
     * @param dt Datatype specification for kernel operation
     * @return Optional kernelInfo with optimal kernel configuration, or nullopt
     * if no suitable backend is registered
     */
    std::optional<dlp::kernel_frame::kernelInfo> getKernelInfoForInput(
        iDEInput*                            in,
        dlp::kernel_frame::kernelRoutineType kType,
        dlp::kernel_frame::kernelDatatype    dt)
    {
        auto kTypeIdx = utils::getUnderlyingValueOfEnum(kType);
        auto dtIdx    = utils::getUnderlyingValueOfEnum(dt);
        if (backends[kTypeIdx][dtIdx] != nullptr) {
            return backends[kTypeIdx][dtIdx]->getKernelInfoForInput(in);
        }
        return std::nullopt;
    }

    /**
     * @brief Retrieves optimal GEMM/GEMV kernel configuration for given input
     * characteristics. However this function takes individual parameters
     * instead of an input object. Additionally the template parameter T
     * specifies the backend type to be used for the query. These changes
     * helps with performance uplift in fast path scenarios since it avoids
     * unnecessary object construction and virtual function dispatch.
     *
     * THREAD SAFETY: Thread-safe if underlying backends are thread-safe
     *
     * @param m Number of rows of matrix A and C
     * @param n Number of columns of matrix B and C
     * @param k Number of columns of matrix A and rows of matrix B
     * @param rs_a Row stride of matrix A
     * @param cs_a Column stride of matrix A
     * @param rs_b Row stride of matrix B
     * @param cs_b Column stride of matrix B
     * @param rs_c Row stride of matrix C
     * @param cs_c Column stride of matrix C
     * @param alpha Pointer to scalar multiplier for the product of A and B
     * @param beta Pointer to scalar multiplier for matrix C
     * @param mtag_a Memory tag for matrix A
     * @param mtag_b Memory tag for matrix B
     * @param metadata Pointer to linked list of post operations metadata
     * @param mr_hint Micro-panel row size hint
     * @param nr_hint Micro-panel column size hint
     * @param kc_hint K dimension blocking size hint
     * @param c_downscale Downscale store type for matrix C
     * @param kType Kernel routine type (e.g., GEMM, GEMV)
     * @param dt Datatype specification for kernel operation
     * @return kernelInfo with optimal kernel configuration, or
     * INVALID_KERNEL_INFO if no suitable backend is registered
     */
    template<typename T>
    DLP_ALWAYS_INLINE dlp::kernel_frame::kernelInfo
                      getGemmKernelInfoForInputFastPath(
                          md_t                                 m,
                          md_t                                 n,
                          md_t                                 k,
                          md_t                                 rs_a,
                          md_t                                 cs_a,
                          md_t                                 rs_b,
                          md_t                                 cs_b,
                          md_t                                 rs_c,
                          md_t                                 cs_c,
                          void*                                alpha,
                          void*                                beta,
                          AOCL_DLP_MEMORY_TAG                  mtag_a,
                          AOCL_DLP_MEMORY_TAG                  mtag_b,
                          dlp_gemm_post_op*                    metadata,
                          md_t                                 mr_hint,
                          md_t                                 nr_hint,
                          md_t                                 kc_hint,
                          md_t                                 c_downscale,
                          dlp::kernel_frame::kernelRoutineType kType,
                          dlp::kernel_frame::kernelDatatype    dt)
    {
        auto kTypeIdx = utils::getUnderlyingValueOfEnum(kType);
        auto dtIdx    = utils::getUnderlyingValueOfEnum(dt);
        if (backends[kTypeIdx][dtIdx] != nullptr) {
            T* backend = static_cast<T*>(backends[kTypeIdx][dtIdx]);
            // Explicitly scope calls to bypass the vtable.
            if ((m == 1) || (n == 1)) {
                return backend->T::getGemvKernelInfoForInputFastPath(
                    dt, m, n, k, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, alpha,
                    beta, mtag_a, mtag_b, metadata, mr_hint, nr_hint, kc_hint,
                    c_downscale, false);
            }
            return backend->T::getGemmKernelInfoForInputFastPath(
                dt, m, n, k, rs_a, cs_a, rs_b, cs_b, rs_c, cs_c, alpha, beta,
                mtag_a, mtag_b, metadata, mr_hint, nr_hint, kc_hint,
                c_downscale, false);
        }

        return INVALID_KERNEL_INFO;
    }

    template<typename T>
    DLP_ALWAYS_INLINE dlp::kernel_frame::packKernelInfo
                      getGemmPackBInfoForInputFastPath(md_t                              nc,
                                                       md_t                              kc,
                                                       md_t                              cs_src,
                                                       md_t                              nr_hint,
                                                       dlp::kernel_frame::kernelDatatype dt)
    {
        auto kTypeIdx = utils::getUnderlyingValueOfEnum(
            dlp::kernel_frame::kernelRoutineType::gemm);
        auto dtIdx = utils::getUnderlyingValueOfEnum(dt);
        if (backends[kTypeIdx][dtIdx] != nullptr) {
            T* backend = static_cast<T*>(backends[kTypeIdx][dtIdx]);
            return backend->T::getGemmPackBInfoForInputFastPath(nc, kc, cs_src,
                                                                nr_hint);
        }

        return kernel_frame::INVALID_PACK_KERNEL_INFO;
    }
};

/**
 * @brief Convenience accessor for decision engine singleton instance
 *
 * Returns reference to the singleton decisionEngine instance.
 * Equivalent to decisionEngine::instance().
 *
 * THREAD SAFETY: Thread-safe (forwards to thread-safe singleton)
 *
 * @return Reference to decision engine singleton
 */
inline decisionEngine&
decisionEngineInstance()
{
    return decisionEngine::instance();
}

} // namespace dlp::de
