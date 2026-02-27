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

#include <map>
#include <vector>

#include "kernel_frame/kernel_frame_base.hh"
#include "kernels/kernel_base.hh"

namespace dlp::kernel_frame {

/**
 * @brief Thread-local fallback kernel storage with RAII cleanup
 *
 * Designed to be used as thread_local variable:
 *   thread_local ThreadLocalFallbackKernelStorage storage;
 *
 * Provides overflow storage when global kernel register table is full.
 * Each thread maintains its own 2D array of kernel pointers indexed by
 * [routine_type][datatype], with a maximum of one kernel per datatype slot.
 *
 */
class ThreadLocalFallbackKernelStorage
{
    // Tracker for kernels in fallback table so as to enable safe deletion of
    // kernel pointer. However users can opt out of this tracking on a per
    // kernel pointer basis.
    std::map<kernels::kernelBase*, md_t> trackedKernels_;

    // The following operations are guaranteed to be thread-safe since the
    // fallback storage is expected to be used as thread_local object.
    void trackKernelPtrInsert(kernels::kernelBase* kB)
    {
        if (trackedKernels_.count(kB) > 0) {
            trackedKernels_[kB] += 1;
        } else {
            trackedKernels_[kB] = 1;
        }
    }

    void trackKernelPtrRemove(kernels::kernelBase* kB)
    {
        if (trackedKernels_.count(kB) > 0) {
            trackedKernels_[kB] -= 1;
            if (trackedKernels_[kB] <= 0) {
                trackedKernels_.erase(kB);
                delete kB;
            }
        }
    }

  public:
    ThreadLocalFallbackKernelStorage()
    {
        // Allocate outer vector for routine types
        storage_.resize(static_cast<size_t>(utils::getUnderlyingValueOfEnum(
            kernelRoutineType::max_kernel_routines)));

        // Allocate inner vectors for datatypes
        for (auto& inner : storage_) {
            inner.resize(static_cast<size_t>(utils::getUnderlyingValueOfEnum(
                             kernelDatatype::max_kernel_datatypes)),
                         nullptr);
        }
    }

    ~ThreadLocalFallbackKernelStorage()
    {
        for (auto kernelPtrPair : trackedKernels_) {
            delete kernelPtrPair.first;
        }
        trackedKernels_.clear();
    }

    // Disable copy/move - thread_local should not be copied
    ThreadLocalFallbackKernelStorage(const ThreadLocalFallbackKernelStorage&) =
        delete;
    ThreadLocalFallbackKernelStorage& operator=(
        const ThreadLocalFallbackKernelStorage&) = delete;
    ThreadLocalFallbackKernelStorage(ThreadLocalFallbackKernelStorage&&) =
        delete;
    ThreadLocalFallbackKernelStorage& operator=(
        ThreadLocalFallbackKernelStorage&&) = delete;

    /**
     * @brief Opt out of tracking for a kernel pointer in fallback storage.
     *
     * This is useful when the kernel pointer is being tracked by the global
     * kernel register table, and we want to avoid double deletion.
     *
     * @param kB Kernel pointer to not track (ownership with fallback now).
     */
    void donotTrackKernelPtr(kernels::kernelBase* kB)
    {
        if (trackedKernels_.count(kB) > 0) {
            trackedKernels_.erase(kB);
        }
    }

    /**
     * @brief Insert kernel into fallback storage and return pointer
     *
     * Replaces existing kernel and frees it if its reference count has reached
     * zero. The inserted kernel is tracked for safe deletion, unless the
     * caller opts out of tracking via donotTrackKernelPtr.
     *
     * @param kType Kernel routine type (e.g., gemm)
     * @param kDtype Kernel datatype
     * @param kB Kernel pointer (ownership transferred to storage)
     * @return Kernel pointer after insertion, or nullptr on bounds error
     */
    kernels::kernelBase* insertAndGet(kernels::kernelBase* kB,
                                      kernelRoutineType    kType,
                                      kernelDatatype       kDtype)
    {
        auto routineIdx = utils::getUnderlyingValueOfEnum(kType);
        auto dtypeIdx   = utils::getUnderlyingValueOfEnum(kDtype);

        // Bounds validation
        if (routineIdx >= storage_.size()
            || dtypeIdx >= storage_[routineIdx].size()) {
            return nullptr;
        }

        if (storage_[routineIdx][dtypeIdx] != nullptr
            && storage_[routineIdx][dtypeIdx] != kB) {
            // If there is an existing kernel, we need to track its deletion
            trackKernelPtrRemove(storage_[routineIdx][dtypeIdx]);
        }

        storage_[routineIdx][dtypeIdx] = kB;
        trackKernelPtrInsert(kB);
        return storage_[routineIdx][dtypeIdx];
    }

    /**
     * @brief Query kernel from fallback storage
     *
     * @param kType Kernel routine type
     * @param kDtype Kernel datatype
     * @return Kernel pointer or nullptr if not found
     */
    template<typename KEY_COMPARATOR, typename KEY_TYPE>
    kernels::kernelBase* query(kernelRoutineType kType,
                               kernelDatatype    kDtype,
                               KEY_TYPE*         kInfo = nullptr)
    {
        auto routineIdx = utils::getUnderlyingValueOfEnum(kType);
        auto dtypeIdx   = utils::getUnderlyingValueOfEnum(kDtype);

        // Bounds validation
        if (routineIdx >= storage_.size()
            || dtypeIdx >= storage_[routineIdx].size()) {
            return nullptr;
        }

        if (kInfo && (storage_[routineIdx][dtypeIdx] != nullptr)) {
            // Optional: Validate kernelInfo matches expected parameters.
            if (KEY_COMPARATOR{}(
                    storage_[routineIdx][dtypeIdx]->getKernelInfo(), kInfo)) {
                return storage_[routineIdx][dtypeIdx];
            } else {
                return nullptr;
            }
        } else {
            return storage_[routineIdx][dtypeIdx];
        }
    }

  private:
    std::vector<std::vector<kernels::kernelBase*>> storage_;
};

} // namespace dlp::kernel_frame
