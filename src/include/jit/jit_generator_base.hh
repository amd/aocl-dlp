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

#include <functional>
#include <memory>
#include <vector>

#include "cpu_utils/cpu_features.hh"
#include "kernels/kernel_base.hh"

namespace dlp::jit {

enum class jitGeneratorError
{
    success = 0,
    notSupported,
    errorAllocatingMemory,
    badKernelInfo,
    error
};

enum class jitAlgoType
{
    gemm = 0,
    gemv_m1,
    gemv_n1,
    unsupportedAlgo
};

struct jitGeneratorContext
{
    const kernel_frame::kernelInfo& kI;

    // Can expand to more entities in future, like profiler, compilation
    // target, etc.

    jitGeneratorContext(const kernel_frame::kernelInfo& kernelInfo)
        : kI(kernelInfo)
    {
    }

    ~jitGeneratorContext() = default;

    jitGeneratorContext(const jitGeneratorContext& other)            = delete;
    jitGeneratorContext& operator=(const jitGeneratorContext& other) = delete;
    jitGeneratorContext(jitGeneratorContext&& other)                 = delete;
    jitGeneratorContext& operator=(jitGeneratorContext&& other)      = delete;
};

class jitGeneratorBase
{
  public:
    virtual ~jitGeneratorBase() {}

    virtual std::vector<cpu_utils::isaFeature>& getIsaFeaturesRequired()    = 0;
    virtual std::vector<kernel_frame::kernelDatatype>& getKernelDatatypes() = 0;
    virtual jitGeneratorError operator()(const jitGeneratorContext& jI)     = 0;
    virtual std::unique_ptr<jitGeneratorBase> clone()                       = 0;

    // TODO: Remove this once the JIT generator and execution code are
    // separated.
    virtual kernels::kernelError executeKernel(kernels::kernelParams* _params)
    {
        return kernels::kernelError::error;
    }
};

} // namespace dlp::jit
