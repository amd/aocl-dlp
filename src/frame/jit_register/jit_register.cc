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

#include "cpu_utils/cpu_features.hh"
#include "jit_register/jit_register.hh"

namespace dlp::jit {

jitGeneratorFrameError
jitGeneratorRegister::registerJitGenerator(
    std::unique_ptr<jitGeneratorBase> jitGen,
    [[maybe_unused]] std::string&&    kernelFamily,
    kernel_frame::kernelRoutineType   kType)
{
    if (!jitGen) {
        return jit::jitGeneratorFrameError::failure;
    }

    // Check if the jit generator is compatible with the CPU features.
    std::vector<cpu_utils::isaFeature>& reqFeatures =
        jitGen->getIsaFeaturesRequired();
    auto hasFeatures =
        cpu_utils::cpuFeaturesInstance().hasFeatures(reqFeatures);
    if (!hasFeatures) {
        return jit::jitGeneratorFrameError::failure;
    }

    // Return if there are no kernel datatypes supported by this JIT generator.
    std::vector<kernel_frame::kernelDatatype>& kDTypes =
        jitGen->getKernelDatatypes();
    if (kDTypes.empty()) {
        return jit::jitGeneratorFrameError::failure;
    }

    jitGeneratorBase* jitGenPtr = jitGen.release();

    auto routineIdx = utils::getUnderlyingValueOfEnum(kType);
    for (auto ele : kDTypes) {
        auto idx = utils::getUnderlyingValueOfEnum(ele);
        if (vecJITGenerators[routineIdx][idx].isOccupied.load(
                std::memory_order_acquire)) {
            if (vecJITGenerators[routineIdx][idx].jitGenerator.load(
                    std::memory_order_relaxed)
                != jitGenPtr) {
                {
                    // Limit locking to the minimum scope.
                    std::lock_guard<std::mutex> lock(
                        replacedJitGeneratorSinkMutex);
                    replacedJitGeneratorSink.push_back(jitGenPtr);
                }
                // If the key is already in the chain, update the value. In
                // future this will need to be based on kernelFamily and
                // priority.
                vecJITGenerators[routineIdx][idx].jitGenerator.store(
                    jitGenPtr, std::memory_order_relaxed);
            }
        } else {
            {
                // All the stored values are tracked in the replaced sink for
                // proper deletion during destruction. Refer the comments in
                // insert api of ThreadSafeChainedDispatchTable for key
                // already present case for a detailed explanation.
                std::lock_guard<std::mutex> lock(replacedJitGeneratorSinkMutex);
                replacedJitGeneratorSink.push_back(jitGenPtr);
            }
            // If the key is not in the chain, insert the key and value.
            // In future this will need to be based on kernelFamily and
            // priority.
            vecJITGenerators[routineIdx][idx].jitGenerator.store(
                jitGenPtr, std::memory_order_relaxed);
            vecJITGenerators[routineIdx][idx].isOccupied.store(
                true, std::memory_order_release);
        }
    }

    return jitGeneratorFrameError::success;
}

jitGeneratorBaseRef
jitGeneratorRegister::registerAndGetJitGenerator(
    std::unique_ptr<jitGeneratorBase> jitGen,
    [[maybe_unused]] std::string&&    kernelFamily,
    kernel_frame::kernelRoutineType   kType,
    kernel_frame::kernelDatatype      kDtype)
{
    if (!jitGen) {
        return jitGeneratorBaseRef(nullptr);
    }

    // Check if the jit generator is compatible with the CPU features.
    std::vector<cpu_utils::isaFeature>& reqFeatures =
        jitGen->getIsaFeaturesRequired();
    auto hasFeatures =
        cpu_utils::cpuFeaturesInstance().hasFeatures(reqFeatures);
    if (!hasFeatures) {
        return jitGeneratorBaseRef(nullptr);
    }

    jitGeneratorBase* jitGenPtr = jitGen.release();

    auto routineIdx = utils::getUnderlyingValueOfEnum(kType);
    auto idx        = utils::getUnderlyingValueOfEnum(kDtype);

    if (vecJITGenerators[routineIdx][idx].isOccupied.load(
            std::memory_order_acquire)) {
        if (vecJITGenerators[routineIdx][idx].jitGenerator.load(
                std::memory_order_relaxed)
            != jitGenPtr) {
            {
                // Limit locking to the minimum scope.
                std::lock_guard<std::mutex> lock(replacedJitGeneratorSinkMutex);
                replacedJitGeneratorSink.push_back(
                    vecJITGenerators[routineIdx][idx].jitGenerator.load(
                        std::memory_order_relaxed));
            }
            // If the key is already in the chain, update the value. In future
            // this will need to be based on kernelFamily and priority.
            vecJITGenerators[routineIdx][idx].jitGenerator.store(
                jitGenPtr, std::memory_order_relaxed);
        }
    } else {
        // If the key is not in the chain, insert the key and value. In future
        // this will need to be based on kernelFamily and priority.
        vecJITGenerators[routineIdx][idx].jitGenerator.store(
            jitGenPtr, std::memory_order_relaxed);
        vecJITGenerators[routineIdx][idx].isOccupied.store(
            true, std::memory_order_release);
    }

    return jitGeneratorBaseRef(
        vecJITGenerators[routineIdx][idx].jitGenerator.load(
            std::memory_order_relaxed));
}

jitGeneratorBaseRef
jitGeneratorRegister::getJitGenerator(kernel_frame::kernelRoutineType kType,
                                      kernel_frame::kernelDatatype    kDtype)
{
    auto routineIdx = utils::getUnderlyingValueOfEnum(kType);
    auto idx        = utils::getUnderlyingValueOfEnum(kDtype);

    if (vecJITGenerators[routineIdx][idx].isOccupied.load(
            std::memory_order_acquire)) {
        auto jitGenPtr = vecJITGenerators[routineIdx][idx].jitGenerator.load(
            std::memory_order_relaxed);
        if (jitGenPtr) {
            return jitGeneratorBaseRef(jitGenPtr);
        }
    }

    return jitGeneratorBaseRef(nullptr);
}

} // namespace dlp::jit
