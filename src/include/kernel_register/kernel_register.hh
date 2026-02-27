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

#include "aocl_dlp_config.h"
#include "cpu_utils/cpu_features.hh"
#include "kernel_register/kernel_dispatch_table.hh"
#include "kernel_register/kernel_fallback_storage.hh"
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

class emptyKernel : public kernels::kernelBase
{
    kernel_frame::kernelInfo                  kInfo;
    std::vector<cpu_utils::isaFeature>        isaVec;
    std::vector<kernel_frame::kernelDatatype> dTypeVec;

  public:
    emptyKernel(const kernel_frame::kernelInfo& kI)
        : kInfo(kI)
    {
        isValid = false;
    }

    emptyKernel(const emptyKernel&)            = delete;
    emptyKernel& operator=(const emptyKernel&) = delete;

    emptyKernel(emptyKernel&& other)
    {
        this->isValid = false;
        this->kInfo   = std::move(other.kInfo);
    }

    emptyKernel& operator=(emptyKernel&& other)
    {
        this->isValid = false;
        this->kInfo   = std::move(other.kInfo);
        return *this;
    }

    virtual std::vector<cpu_utils::isaFeature>& getIsaFeaturesForKernel()
        override final
    {
        return isaVec;
    }

    virtual kernel_frame::kernelInfo* getKernelInfo() override final
    {
        return std::addressof(kInfo);
    }

    virtual std::vector<kernel_frame::kernelDatatype>& getKernelDatatypes()
        override final
    {
        return dTypeVec;
    }

    virtual kernels::kernelError operator()(
        kernels::kernelParams* kP) override final
    {
        return kernels::kernelError::error;
    }
};

using kernelFunctionPtr =
    std::function<kernels::kernelError(kernels::kernelParams*)>;
using kernelBaseRef = utils::ptrWrapper<kernels::kernelBase,
                                        kernels::kernelParams,
                                        kernels::kernelError>;

// Forward declare thread-local tier 2 storage
// Actual definition in kernel_register.cc
extern thread_local ThreadLocalFallbackKernelStorage tlsTier2KernelStorage;

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
        vecKDTs.resize(static_cast<std::size_t>(utils::getUnderlyingValueOfEnum(
            kernelRoutineType::max_kernel_routines)));

        for (auto& ele : vecKDTs) {
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

    // This is usually called in the slow path, so both the tier 1 and tier
    // 2 insertion logic is fused together in this function. The same wont
    // be the case for getKernel since get operation is expected to be in
    // the fast path. There will be a getKernel and a getKernelFallback where
    // getKernel will only look at tier 1 and getKernelFallback will look at
    // only tier 2.
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

        std::vector<kernelDatatype>& kDTypes = _kB->getKernelDatatypes();
        if (kDTypes.empty()) {
            return kernelFrameError::failure;
        }

        kernels::kernelBase* kB = _kB.release();
        kernelInfo*          kI = kB->getKernelInfo();

        // This boolean will be used to track if the kernel was inserted
        // at least once, and if not to free the kernel pointer. Insertion
        // in atleast one kernel table will ensure its cleaned up properly.
        bool isInsertedAtleastOnceTier1 = false;
        bool isInsertedAtleastOnceTier2 = false;
        auto routineIdx = utils::getUnderlyingValueOfEnum(kType);
        for (auto ele : kDTypes) {
            auto idx = utils::getUnderlyingValueOfEnum(ele);
            auto retPtr =
                vecKDTs[routineIdx][idx]
                    .template insert<HASH_KEY_GETTER, KEY_COMPARATOR, KEY_TYPE,
                                     VALUE_WATCHER, kernels::kernelBase>(kI,
                                                                         kB);

            // If insertion in tier 1 dispatch table fails, attempt to insert in
            // tier 2 fallback storage. The kernel is guaranteed to be inserted
            // in at least one of the two tiers.
            if (!retPtr) {
                retPtr = tlsTier2KernelStorage.insertAndGet(kB, kType, ele);
                if (retPtr) {
                    isInsertedAtleastOnceTier2 = true;
                }
            } else {
                isInsertedAtleastOnceTier1 = true;
            }
        }

        if (!isInsertedAtleastOnceTier1 && !isInsertedAtleastOnceTier2) {
            // NOTE: The std::default_delete in std::unique_ptr maps to delete.
            delete kB;
            return kernelFrameError::failure;
        } else if (isInsertedAtleastOnceTier1 && isInsertedAtleastOnceTier2) {
            // The tier 1 table is already tracking the kernel, so we ask
            // tier2 storage to not track the kernel pointer.
            tlsTier2KernelStorage.donotTrackKernelPtr(kB);
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

        auto kernPtr =
            vecKDTs[routineIdx][idx]
                .template insert<HASH_KEY_GETTER, KEY_COMPARATOR, KEY_TYPE,
                                 VALUE_WATCHER, kernels::kernelBase>(kI, kB);

        // If insertion in tier 1 dispatch table fails, attempt to insert in
        // tier 2 fallback storage. The kernel is guaranteed to be inserted
        // in at least one of the two tiers.
        if (!kernPtr) {
            kernPtr = tlsTier2KernelStorage.insertAndGet(kB, kType, kDtype);
        }

        if (!kernPtr) {
            // NOTE: The std::default_delete in std::unique_ptr maps to delete.
            delete kB;
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

        auto kB = vecKDTs[routineIdx][dtypeIdx]
                      .template query<HASH_KEY_GETTER, KEY_COMPARATOR, KEY_TYPE,
                                      kernels::kernelBase>(kI);

        return kernelBaseRef(kB);
    }

    template<typename KEY_COMPARATOR, typename KEY_TYPE>
    [[nodiscard]] kernelBaseRef getKernelFallback(KEY_TYPE*         kI,
                                                  kernelRoutineType kType,
                                                  kernelDatatype    kDtype)
    {
        auto kB = tlsTier2KernelStorage.query<KEY_COMPARATOR, KEY_TYPE>(
            kType, kDtype, kI);

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

    /**
     * @brief Retrieves reference to registered GEMM kernel in the fallback
     * or tier 2 storage.
     *
     * Looks up previously registered kernel by kernelInfo and datatype.
     * Returns reference to kernel pointer ready for execution.
     *
     * THREAD SAFETY: Thread-safe since the underlying fallback storage is
     * thread-local and does not require synchronization for access.
     *
     * @param kI Pointer to kernelInfo describing required kernel parameters
     * @param kDtype Datatype for kernel lookup
     * @return Function pointer to kernel execution function, or nullptr if not
     * found
     */
    [[nodiscard]] kernelBaseRef getGemmKernelFallback(kernelInfo*    kI,
                                                      kernelDatatype kDtype)
    {
        return getKernelFallback<gemmKeyComparator, kernelInfo>(
            kI, kernelRoutineType::gemm, kDtype);
    }

    /**
     * @brief Registers an empty placeholder kernel for failed JIT generation
     *
     * Creates and registers an emptyKernel instance that serves as a sentinel
     * value to indicate a kernel could not be generated (e.g., JIT compilation
     * failure). The empty kernel returns kernelError::error when called and
     * has isValid set to false. This allows the dispatch table to distinguish
     * between "kernel not attempted" (nullptr) and "kernel generation failed"
     * (emptyKernel).
     *
     * THREAD SAFETY: Thread-safe via underlying dispatch table synchronization
     *
     * @param kI KernelInfo describing the kernel configuration that failed
     * @param kDtype Datatype for which the kernel registration failed
     */
    void registerEmptyGemmKernel(const kernelInfo& kI, kernelDatatype kDtype)
    {
        auto eK = new emptyKernel(kI);
        auto routineIdx =
            utils::getUnderlyingValueOfEnum(kernelRoutineType::gemm);
        auto idx = utils::getUnderlyingValueOfEnum(kDtype);

        auto retPtr = vecKDTs[routineIdx][idx]
                          .template insert<gemmHashKeyGetter, gemmKeyComparator,
                                           kernelInfo, storedKernelWatcher>(
                              eK->getKernelInfo(), eK);

        if (!retPtr) {
            // Need to cleanup the pointer if insertion failed since there
            // is no kernel table tracking the empty kernel.
            delete eK;
        }
    }
};

// DLP_KDT_TABLE_SIZE and DLP_KDT_CHAIN_SIZE are set to 16 and 128 by default,
// respectively.
constexpr inline std::size_t dlpKernelRegisterTableNumBuckets =
    DLP_KDT_TABLE_SIZE;
constexpr inline std::size_t dlpKernelRegisterTableChainSize =
    DLP_KDT_CHAIN_SIZE;

/**
 * @brief Default kernel register type for DLP framework
 *
 * Pre-configured kernelRegister using ThreadSafeChainedDispatchTable with
 * optimized default parameters for typical DLP workloads (16 buckets × 128
 * chain size).
 *
 * MEMORY: ~256-512MB total depending on max_kernel_datatypes based on an upper
 *         bound of 1024 buckets x 512 chains.
 * PERFORMANCE: Optimized for 1000+ kernels with <5% collision rate
 */
using dlpKernelRegister = kernelRegister<
    ThreadSafeChainedDispatchTable<dlpKernelRegisterTableNumBuckets,
                                   dlpKernelRegisterTableChainSize>>;

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
