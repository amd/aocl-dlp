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
        // Support F32 decision engine
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
    }

    decisionEngine()
    {
        backends.reserve(
            static_cast<std::size_t>(utils::getUnderlyingValueOfEnum(
                dlp::kernel_frame::kernelRoutineType::max_kernel_routines)));
        backends.resize(
            static_cast<std::size_t>(utils::getUnderlyingValueOfEnum(
                dlp::kernel_frame::kernelRoutineType::max_kernel_routines)));

        for (auto& ele : backends) {
            ele.reserve(
                static_cast<std::size_t>(utils::getUnderlyingValueOfEnum(
                    dlp::kernel_frame::kernelDatatype::max_kernel_datatypes)));
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
