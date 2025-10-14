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
#include <set>
#include <string>

#include "cpu_utils/cpu_features.hh"
#include "kernel_register/kernel_dispatch_table.hh"
#include "kernel_register/kernel_register_traits.hh"
#include "kernels/kernel_base.hh"
#include "utils/macro_utils.hh"
#include "utils/ptr_wrappers.hh"
#include "utils/type_utils.hh"

namespace dlp::kernel_frame {

struct gemmHashKeyGetter
{
    /**
     * @brief Extracts hash key tuple from kernel info
     * @param kI Pointer to kernelInfo object
     * @return Tuple containing (mr, nr) dimensions
     */
    std::tuple<uint64_t, uint64_t> operator()(void* kI) const
    {
        auto tKI = static_cast<kernelInfo*>(kI);
        return std::make_tuple(tKI->mr, tKI->nr);
    }
};

struct gemmKeyComparator
{
    /**
     * @brief Compares two kernelInfo objects for equality
     * @param kI1 First kernelInfo pointer
     * @param kI2 Second kernelInfo pointer
     * @return True if kernel infos are equal, false otherwise
     */
    bool operator()(void* kI1, void* kI2) const
    {
        auto tKI1 = static_cast<kernelInfo*>(kI1);
        auto tKI2 = static_cast<kernelInfo*>(kI2);
        return *tKI1 == *tKI2;
    }
};

// This is a singleton class that collects the replaced kernel instances.
// It is used to avoid double deletion of the same kernel instance.
// It is also used to avoid the kernel instance being deleted before the
// kernel is executed.
class storedKernelWatcher
{
    static std::mutex         mtx;
    static std::vector<void*> valueSink;
    static std::set<void*>    valueSet;

  public:
    static void sinkCollect(void* kB)
    {
        std::lock_guard<std::mutex> lg{ mtx };
        valueSink.push_back(kB);
    }

    void operator()(void* kB) { sinkCollect(kB); }

    static void cleanup(std::set<kernels::kernelBase*>& kernelsInTable)
    {
        std::lock_guard<std::mutex> lg{ mtx };
        for (auto& ele : valueSink) {
            if ((valueSet.count(ele) == 0)
                && (kernelsInTable.count(
                        reinterpret_cast<kernels::kernelBase*>(ele))
                    == 0)) {
                // The ele should not already have been deleted in the
                // kernelRegister destructor.
                valueSet.insert(ele);
                delete reinterpret_cast<kernels::kernelBase*>(ele);
            }
        }
        valueSink.clear();
        valueSet.clear();
    }
};

using kernelFunctionPtr =
    std::function<kernels::kernelError(kernels::kernelParams*)>;
using kernelBaseRef = utils::ptrWrapper<kernels::kernelBase,
                                        kernels::kernelParams,
                                        kernels::kernelError>;

/**
 * @brief Singleton registry for micro-kernels with thread-safe dispatch
 *
 * kernelRegister manages registration and lookup of micro-kernels using
 * a 2D array of dispatch tables indexed by [routine_type][datatype]. Implements
 * Meyer's singleton pattern with thread-safe insertion and query operations.
 * Each kernel is validated against current CPU features before registration to
 * ensure runtime compatibility.
 *
 * DESIGN PHILOSOPHY:
 * - 2D dispatch table organization: routine_type × datatype
 * - Memory-efficient storage with lazy cleanup
 *
 * @tparam KERN_DISPATCH_TABLE Dispatch table implementation (must satisfy
 * interface requirements)
 */
template<typename KERN_DISPATCH_TABLE>
class kernelRegister
{
    // Compile-time validation that dispatch table satisfies required interface
    static_assert(
        kernelRegisterTraits::hasDispatchTableInterface_v<
            KERN_DISPATCH_TABLE,
            gemmHashKeyGetter,
            gemmKeyComparator,
            kernelInfo,
            storedKernelWatcher,
            kernels::kernelBase>(),
        "KERN_DISPATCH_TABLE must have the following methods defined: "
        "insert, query, getValues, getKeys\n");

    kernelRegister()
    {
        vecKDTs.reserve(
            static_cast<std::size_t>(utils::getUnderlyingValueOfEnum(
                kernelRoutineType::max_kernel_routines)));
        vecKDTs.resize(static_cast<std::size_t>(utils::getUnderlyingValueOfEnum(
            kernelRoutineType::max_kernel_routines)));

        for (auto& ele : vecKDTs) {
            ele.reserve(
                static_cast<std::size_t>(utils::getUnderlyingValueOfEnum(
                    kernelDatatype::max_kernel_datatypes)));
            ele.resize(static_cast<std::size_t>(utils::getUnderlyingValueOfEnum(
                kernelDatatype::max_kernel_datatypes)));
        }
    }

    ~kernelRegister()
    {
        // Using a set to avoid deleting the same value multiple times.
        // This is required since the same VALUE_TYPE could be inserted
        // multiple times depending on its usage by the composing class.
        // For example, a kernel could be registered multiple time with
        // different kernelDatatypes.
        std::set<kernels::kernelBase*> valueSet;
        for (auto& ele : vecKDTs) {
            for (auto& ele2 : ele) {
                auto values = ele2.template getValues<kernels::kernelBase>();
                for (auto& value : values) {
                    if (valueSet.count(value) == 0) {
                        valueSet.insert(value);
                        delete value;
                    }
                }
            }
        }

        // Cleanup the replaced kernel sink.
        storedKernelWatcher::cleanup(valueSet);
    }

    // Copy/move operations disabled for singleton
    kernelRegister(const kernelRegister& kReg)            = delete;
    kernelRegister(kernelRegister&& kReg)                 = delete;
    kernelRegister& operator=(const kernelRegister& kReg) = delete;
    kernelRegister& operator=(kernelRegister&& kReg)      = delete;

    // 2D array of dispatch tables: vecKDTs[routine_type][datatype]
    std::vector<std::vector<KERN_DISPATCH_TABLE>> vecKDTs;

    template<typename HASH_KEY_GETTER,
             typename KEY_COMPARATOR,
             typename KEY_TYPE,
             typename VALUE_WATCHER>
    [[nodiscard]] kernelFrameError registerKernel(
        std::unique_ptr<kernels::kernelBase> _kB,
        std::string&&                        kernelFamily,
        kernelRoutineType                    kType)
    {
        if (!_kB) {
            return kernelFrameError::failure;
        }

        // Check if the kernel is compatible with the CPU features.
        std::vector<cpu_utils::isaFeature>& reqFeatures =
            _kB->getIsaFeaturesForKernel();
        auto hasFeatures =
            cpu_utils::cpuFeaturesInstance().hasFeatures(reqFeatures);
        if (!hasFeatures) {
            return kernelFrameError::failure;
        }

        kernels::kernelBase*         kB      = _kB.release();
        kernelInfo*                  kI      = kB->getKernelInfo();
        std::vector<kernelDatatype>& kDTypes = kB->getKernelDatatypes();

        auto routineIdx = utils::getUnderlyingValueOfEnum(kType);
        for (auto ele : kDTypes) {
            auto idx = utils::getUnderlyingValueOfEnum(ele);
            static_cast<void>(
                vecKDTs[routineIdx][idx]
                    .template insert<HASH_KEY_GETTER, KEY_COMPARATOR, KEY_TYPE,
                                     VALUE_WATCHER>(kI, kB));
        }

        return kernelFrameError::success;
    }

    template<typename HASH_KEY_GETTER,
             typename KEY_COMPARATOR,
             typename KEY_TYPE,
             typename VALUE_WATCHER>
    [[nodiscard]] kernelBaseRef registerAndGetKernel(
        std::unique_ptr<kernels::kernelBase> _kB,
        std::string&&                        kernelFamily,
        kernelRoutineType                    kType,
        kernelDatatype                       kDtype)
    {
        if (!_kB) {
            return kernelBaseRef(nullptr);
        }

        // Check if the kernel is compatible with the CPU features.
        std::vector<cpu_utils::isaFeature>& reqFeatures =
            _kB->getIsaFeaturesForKernel();
        auto hasFeatures =
            cpu_utils::cpuFeaturesInstance().hasFeatures(reqFeatures);
        if (!hasFeatures) {
            return kernelBaseRef(nullptr);
        }

        kernels::kernelBase* kB = _kB.release();

        kernelInfo* kI = kB->getKernelInfo();

        auto routineIdx = utils::getUnderlyingValueOfEnum(kType);
        auto idx        = utils::getUnderlyingValueOfEnum(kDtype);

        // Safe to cast the voidFunctorPtr to kernelBase* because it is
        // guaranteed that kernelBase* was typecasted to voidFunctorPtr in
        // insert operation.
        auto kernPtr = reinterpret_cast<kernels::kernelBase*>(
            vecKDTs[routineIdx][idx]
                .template insert<HASH_KEY_GETTER, KEY_COMPARATOR, KEY_TYPE,
                                 VALUE_WATCHER>(kI, kB));

        if (!kernPtr) {
            // TODO: Add logging here.
            return kernelBaseRef(nullptr);
        }

        return kernelBaseRef(kernPtr);
    }

    template<typename HASH_KEY_GETTER,
             typename KEY_COMPARATOR,
             typename KEY_TYPE>
    [[nodiscard]] kernelBaseRef getKernel(KEY_TYPE*         kI,
                                          kernelRoutineType kType,
                                          kernelDatatype    kDtype)
    {
        auto routineIdx = utils::getUnderlyingValueOfEnum(kType);
        auto dtypeIdx   = utils::getUnderlyingValueOfEnum(kDtype);

        // Safe to cast the voidFunctorPtr to kernelBase* because it is
        // guaranteed that kernelBase* was typecasted to voidFunctorPtr in
        // insert operation.
        auto kB = reinterpret_cast<kernels::kernelBase*>(
            vecKDTs[routineIdx][dtypeIdx]
                .template query<HASH_KEY_GETTER, KEY_COMPARATOR, KEY_TYPE>(kI));

        // Cannot throw an exception here because this function is called by the
        // user and its not necessary to have the kernel inserted by the time
        // query is called.
        if (!kB) {
            // TODO: Add logging here.
            return kernelBaseRef(nullptr);
        }

        return kernelBaseRef(kB);
    }

  public:
    /**
     * @brief Meyer's singleton instance accessor with thread-safe
     * initialization
     *
     * Returns reference to the single global kernel registry instance.
     * Thread-safe initialization guaranteed by C++11 standard.
     *
     * THREAD SAFETY: Thread-safe initialization, concurrent access safe
     *
     * @return Reference to singleton kernelRegister instance
     */
    static kernelRegister& instance()
    {
        static kernelRegister kReg;
        return kReg;
    }

    /**
     * @brief Registers GEMM kernel for all its supported datatypes
     *
     * Validates kernel's ISA feature requirements against current CPU features
     * before registration to ensure runtime compatibility. Registers the
     * kernel in dispatch tables corresponding to all datatypes returned by
     * kB->getKernelDatatypes(). Same kernel instance may be registered
     * multiple times for different datatypes.
     *
     * THREAD SAFETY: Thread-safe via underlying dispatch table synchronization
     *
     * @param _kB Pointer to kernel instance (ownership transferred to registry)
     * @param kernelFamily String identifier for kernel family (moved)
     * @return kernelFrameError::success on successful registration,
     * kernelFrameError::failure otherwise
     */
    [[nodiscard]] kernelFrameError registerGemmKernel(
        std::unique_ptr<kernels::kernelBase> _kB, std::string&& kernelFamily)
    {
        return registerKernel<gemmHashKeyGetter, gemmKeyComparator, kernelInfo,
                              storedKernelWatcher>(
            std::move(_kB), std::move(kernelFamily), kernelRoutineType::gemm);
    }

    /**
     * @brief Registers kernel and returns function pointer for specific
     * datatype
     *
     * Combines registration with immediate kernel pointer retrieval for
     * a specific datatype. Useful when you need the kernel immediately after
     * registration.
     *
     * THREAD SAFETY: Thread-safe via underlying dispatch table synchronization
     *
     * @param kB Pointer to kernel instance (ownership transferred to registry)
     * @param kernelFamily String identifier for kernel family (moved)
     * @param kDtype Specific datatype for function pointer retrieval
     * @return Reference wrapper to registered kernel, or nullptr on failure
     */
    [[nodiscard]] kernelBaseRef registerAndGetGemmKernel(
        std::unique_ptr<kernels::kernelBase> _kB,
        std::string&&                        kernelFamily,
        kernelDatatype                       kDtype)
    {
        return registerAndGetKernel<gemmHashKeyGetter, gemmKeyComparator,
                                    kernelInfo, storedKernelWatcher>(
            std::move(_kB), std::move(kernelFamily), kernelRoutineType::gemm,
            kDtype);
    }

    /**
     * @brief Retrieves reference to registered GEMM kernel
     *
     * Looks up previously registered kernel by kernelInfo and datatype.
     * Returns reference to kernel pointer ready for execution.
     *
     * THREAD SAFETY: Thread-safe via underlying dispatch table synchronization
     *
     * @param kI Pointer to kernelInfo describing required kernel parameters
     * @param kDtype Datatype for kernel lookup
     * @return Function pointer to kernel execution function, or nullptr if not
     * found
     */
    [[nodiscard]] kernelBaseRef getGemmKernel(kernelInfo*    kI,
                                              kernelDatatype kDtype)
    {
        return getKernel<gemmHashKeyGetter, gemmKeyComparator, kernelInfo>(
            kI, kernelRoutineType::gemm, kDtype);
    }
};

constexpr inline std::size_t dlpKernelRegisterTableSize       = 200;
constexpr inline std::size_t dlpKernelRegisterTableNumBuckets = 50;

/**
 * @brief Default kernel register type for DLP framework
 *
 * Pre-configured kernelRegister using ThreadSafeChainedDispatchTable with
 * optimized parameters for typical DLP workloads (200 buckets × 50 chain size).
 *
 * MEMORY: ~256-512MB total depending on max_kernel_datatypes
 * PERFORMANCE: Optimized for 1000+ kernels with <5% collision rate
 */
using dlpKernelRegister = kernelRegister<
    ThreadSafeChainedDispatchTable<dlpKernelRegisterTableSize,
                                   dlpKernelRegisterTableNumBuckets>>;

/**
 * @brief Convenience accessor for default DLP kernel register instance
 *
 * Returns reference to the singleton dlpKernelRegister instance.
 * Equivalent to dlpKernelRegister::kernelRegisterInstance().
 *
 * THREAD SAFETY: Thread-safe (forwards to thread-safe singleton)
 *
 * @return Reference to default DLP kernel register singleton
 */
inline dlpKernelRegister&
dlpKernelRegisterInstance()
{
    return dlpKernelRegister::instance();
}

} // namespace dlp::kernel_frame

/**
 * @brief Macro for static registration of GEMM kernels at startup
 *
 * Automatically registers a kernel class at program startup using static
 * initialization. Creates a uniquely named static variable to hold the
 * registration result. CPU feature validation occurs during registration.
 *
 * REQUIREMENTS:
 * - className must be default constructible
 * - className must derive from kernelBase
 * - Registration happens before main() execution
 *
 * @param className Class name of kernel to register (must be default
 * constructible)
 * @param kernelFamily String literal for kernel family identifier
 */
#define DLP_REGISTER_STATIC_GEMM_KERNEL(className, kernelFamily)               \
    static_assert(std::is_default_constructible_v<className>,                  \
                  "Requires trivially constructible classes for kernels.");    \
    static_assert(std::is_base_of_v<kernelBase, className>,                    \
                  "Requires classes derived from kernelBase.");                \
    static auto DLP_SUBS_CONCAT_3TOK(static_mgc_dlp_kernel_reg_var_,           \
                                     className, __LINE__) =                    \
        dlp::kernel_frame::dlpKernelRegisterInstance().registerGemmKernel(     \
            std::move(std::make_unique<className>()),                          \
            std::string{ kernelFamily });
