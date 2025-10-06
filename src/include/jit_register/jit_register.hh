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

#include <atomic>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "jit/jit_generator_base.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "utils/macro_utils.hh"
#include "utils/ptr_wrappers.hh"

namespace dlp::jit {

using jitGeneratorBaseRef = utils::ptrWrapper<jitGeneratorBase,
                                              const kernel_frame::kernelInfo&,
                                              jitGeneratorError>;

enum class jitGeneratorFrameError
{
    success,
    failure
};

/**
 * @brief Singleton registry for JIT generators with CPU feature validation
 *
 * jitGeneratorRegister manages registration and lookup of JIT code generators
 * using a 2D array indexed by [routine_type][datatype]. Each JIT generator
 * is validated against current CPU features before registration to ensure
 * runtime compatibility.
 *
 * DESIGN PHILOSOPHY:
 * - 2D array organization: routine_type × datatype for O(1) lookup
 * - Single JIT generator per datatype with family-based precedence
 */
class jitGeneratorRegister
{
    jitGeneratorRegister()
    {
        vecJITGenerators.reserve(
            static_cast<std::size_t>(utils::getUnderlyingValueOfEnum(
                kernel_frame::kernelRoutineType::max_kernel_routines)));
        vecJITGenerators.resize(
            static_cast<std::size_t>(utils::getUnderlyingValueOfEnum(
                kernel_frame::kernelRoutineType::max_kernel_routines)));

        for (auto& ele : vecJITGenerators) {
            ele.reserve(
                static_cast<std::size_t>(utils::getUnderlyingValueOfEnum(
                    kernel_frame::kernelDatatype::max_kernel_datatypes)));
            ele.resize(static_cast<std::size_t>(utils::getUnderlyingValueOfEnum(
                           kernel_frame::kernelDatatype::max_kernel_datatypes)),
                       JITGeneratorEntry());
        }
    }

    ~jitGeneratorRegister()
    {
        // Using a set to avoid deleting the same value multiple times.
        // This is required since the same VALUE_TYPE could be inserted
        // multiple times depending on its usage by the composing class.
        // For example, a jitGenerator could be registered multiple time with
        // different kernelDatatypes.
        std::set<jitGeneratorBase*> valueSet;
        for (auto& ele : vecJITGenerators) {
            for (auto& ele2 : ele) {
                if (valueSet.count(
                        ele2.jitGenerator.load(std::memory_order_relaxed))
                    == 0) {
                    valueSet.insert(
                        ele2.jitGenerator.load(std::memory_order_relaxed));
                    delete ele2.jitGenerator.load(std::memory_order_relaxed);
                }
            }
        }

        valueSet.clear();
        for (auto& ele : replacedJitGeneratorSink) {
            if (valueSet.count(ele) == 0) {
                valueSet.insert(ele);
                delete ele;
            }
        }
        replacedJitGeneratorSink.clear();
    }

    // Copy/move operations disabled for singleton
    jitGeneratorRegister(const jitGeneratorRegister& jReg)            = delete;
    jitGeneratorRegister(jitGeneratorRegister&& jReg)                 = delete;
    jitGeneratorRegister& operator=(const jitGeneratorRegister& jReg) = delete;
    jitGeneratorRegister& operator=(jitGeneratorRegister&& jReg)      = delete;

    // Internal struct to store the JIT generator and its occupation status.
    struct JITGeneratorEntry
    {
        std::atomic<jitGeneratorBase*> jitGenerator;
        std::atomic<bool>              isOccupied;

        JITGeneratorEntry()
            : jitGenerator(nullptr)
            , isOccupied(false)
        {
        }

        JITGeneratorEntry(const JITGeneratorEntry& other)
            : jitGenerator(other.jitGenerator.load(std::memory_order_relaxed))
            , isOccupied(other.isOccupied.load(std::memory_order_relaxed))
        {
        }

        JITGeneratorEntry(JITGeneratorEntry&& other)
            : jitGenerator(other.jitGenerator.load(std::memory_order_relaxed))
            , isOccupied(other.isOccupied.load(std::memory_order_relaxed))
        {
        }

        JITGeneratorEntry& operator=(const JITGeneratorEntry& other)
        {
            jitGenerator.store(
                other.jitGenerator.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
            isOccupied.store(other.isOccupied.load(std::memory_order_relaxed),
                             std::memory_order_relaxed);
            return *this;
        }

        JITGeneratorEntry& operator=(JITGeneratorEntry&& other)
        {
            jitGenerator.store(
                other.jitGenerator.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
            isOccupied.store(other.isOccupied.load(std::memory_order_relaxed),
                             std::memory_order_relaxed);
            return *this;
        }
    };

    // 2D array of JIT generators: vecJITGenerators[routine_type][datatype]
    std::vector<std::vector<JITGeneratorEntry>> vecJITGenerators;

    // Sink for replaced JIT generators. Unlike kernelRegister, we don't need
    // a separate struct for acting as the sink for the replaced JIT generators.
    // This is because kernelRegister depends on an external lookup table to
    // store the kernelInfo objects.
    std::vector<jitGeneratorBase*> replacedJitGeneratorSink;
    std::mutex                     replacedJitGeneratorSinkMutex;

    [[nodiscard]] jitGeneratorFrameError registerJitGenerator(
        std::unique_ptr<jitGeneratorBase> jitGen,
        std::string&&                     kernelFamily,
        kernel_frame::kernelRoutineType   kType);

    [[nodiscard]] jitGeneratorBaseRef registerAndGetJitGenerator(
        std::unique_ptr<jitGeneratorBase> jitGen,
        std::string&&                     kernelFamily,
        kernel_frame::kernelRoutineType   kType,
        kernel_frame::kernelDatatype      kDtype);

    [[nodiscard]] jitGeneratorBaseRef getJitGenerator(
        kernel_frame::kernelRoutineType kType,
        kernel_frame::kernelDatatype    kDtype);

  public:
    /**
     * @brief Meyer's singleton instance accessor with thread-safe
     * initialization
     *
     * Returns reference to the single global JIT generator registry instance.
     * Thread-safe initialization guaranteed by C++11 standard.
     *
     * THREAD SAFETY: Thread-safe initialization, concurrent access safe for
     * reads
     *
     * @return Reference to singleton jitGeneratorRegister instance
     */
    static jitGeneratorRegister& instance()
    {
        static jitGeneratorRegister jReg;
        return jReg;
    }

    /**
     * @brief Registers GEMM JIT generator for all its supported datatypes
     *
     * Validates JIT generator's ISA feature requirements against current CPU
     * features before registration. Registers the generator for all datatypes
     * returned by jitGen->getKernelDatatypes(). Only one JIT generator per
     * datatype is allowed, with family-based precedence determining conflicts.
     *
     * THREAD SAFETY: Not thread-safe (intended for startup registration only)
     *
     * @param jitGen Pointer to JIT generator instance (ownership transferred)
     * @param kernelFamily String identifier for JIT generator family (moved)
     * @return kernelFrameError::success on successful registration,
     *         kernelFrameError::failure if CPU features insufficient or other
     * error
     */
    [[nodiscard]] jitGeneratorFrameError registerGemmJitGenerator(
        std::unique_ptr<jitGeneratorBase> jitGen, std::string&& kernelFamily)
    {
        return registerJitGenerator(std::move(jitGen), std::move(kernelFamily),
                                    kernel_frame::kernelRoutineType::gemm);
    }

    /**
     * @brief Registers JIT generator and returns reference for specific
     * datatype
     *
     * Combines registration with immediate generator reference retrieval for
     * a specific datatype. Useful when you need the JIT generator immediately
     * after registration for code generation.
     *
     * THREAD SAFETY: Not thread-safe (intended for startup registration only)
     *
     * @param jitGen Pointer to JIT generator instance (ownership transferred)
     * @param kernelFamily String identifier for JIT generator family (moved)
     * @param kDtype Specific datatype for reference retrieval
     * @return Reference wrapper to registered JIT generator, or nullptr on
     * failure
     */
    [[nodiscard]] jitGeneratorBaseRef registerAndGetGemmJitGenerator(
        std::unique_ptr<jitGeneratorBase> jitGen,
        std::string&&                     kernelFamily,
        kernel_frame::kernelDatatype      kDtype)
    {
        return registerAndGetJitGenerator(
            std::move(jitGen), std::move(kernelFamily),
            kernel_frame::kernelRoutineType::gemm, kDtype);
    }

    /**
     * @brief Retrieves reference to registered GEMM JIT generator
     *
     * Looks up previously registered JIT generator by datatype. Returns
     * reference to generator pointer ready for code generation operations.
     *
     * THREAD SAFETY: Thread-safe for read-only access after initialization
     *
     * @param kDtype Datatype for JIT generator lookup
     * @return Smart pointer reference to JIT generator, or nullptr if not found
     */
    [[nodiscard]] jitGeneratorBaseRef getGemmJitGenerator(
        kernel_frame::kernelDatatype kDtype)
    {
        return getJitGenerator(kernel_frame::kernelRoutineType::gemm, kDtype);
    }
};

/**
 * @brief Convenience accessor for default DLP JIT generator register instance
 *
 * Returns reference to the singleton jitGeneratorRegister instance.
 * Equivalent to jitGeneratorRegister::instance().
 *
 * THREAD SAFETY: Thread-safe (forwards to thread-safe singleton)
 *
 * @return Reference to default DLP JIT generator register singleton
 */
inline jitGeneratorRegister&
dlpJitGeneratorRegisterInstance()
{
    return jitGeneratorRegister::instance();
}

} // namespace dlp::jit

/**
 * @brief Macro for static registration of GEMM JIT generators at startup
 *
 * Automatically registers a JIT generator class at program startup using static
 * initialization. Creates a uniquely named static variable to hold the
 * registration result. CPU feature validation occurs during registration.
 *
 * REQUIREMENTS:
 * - className must be default constructible
 * - className must derive from jitGeneratorBase
 * - Registration happens before main() execution
 *
 * @param className Class name of JIT generator to register (must be default
 * constructible)
 * @param kernelFamily String literal for JIT generator family identifier
 */
#define DLP_REGISTER_STATIC_GEMM_JIT_GENERATOR(className, kernelFamily)        \
    static_assert(                                                             \
        std::is_default_constructible_v<className>,                            \
        "Requires trivially constructible classes for jit generators.");       \
    static_assert(std::is_base_of_v<dlp::jit::jitGeneratorBase, className>,    \
                  "Requires classes derived from jitGeneratorBase.");          \
    static auto DLP_SUBS_CONCAT_3TOK(static_mgc_dlp_jit_reg_var_, className,   \
                                     __LINE__) =                               \
        dlp::jit::dlpJitGeneratorRegisterInstance().registerGemmJitGenerator(  \
            std::make_unique<className>(), std::string{ kernelFamily });
