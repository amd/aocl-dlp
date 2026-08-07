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

#include "cpu_utils/cpu_features.hh"
#include "jit_register/jit_register.hh"
#include "kernels/kernel_base.hh"

namespace dlp::jit {

/**
 * @brief Wrapper class for integrating JIT generators with extensible kernel
 * framework
 *
 * This class provides a clean integration path for JIT generator developers.
 * It handles the conversion between framework interfaces and delegates actual
 * kernel operations to the JIT generator implementation.
 *
 * Usage for JIT generator developers:
 * 1. Implement JitGeneratorBase interface for your specific ISA
 * 2. Create JitKernelBase wrapper with your implementation
 * 3. Register with extensible kernel framework
 */
class jitKernelAdapter : public kernels::kernelBase
{
  public:
    jitKernelAdapter(const kernel_frame::kernelInfo& kI,
                     gemmJitGeneratorRef             jitGen,
                     bool shouldGenerateKernels = true)
        : mKernelInfo{ kI }
        , mIsJitGenerated(false)
    {
        auto gemmGen = jitGen ? jitGen->clone() : nullptr;

        // TODO: The shouldGenerateKernels flag is used to enable registration
        // of dummy kernels without actually running the JIT generator. This is
        // a temporary solution to help with simulating a kernelRegister that is
        // under extreme load from too many kernels being registered.
        if ((shouldGenerateKernels) && (gemmGen)) {
            gemmJitGeneratorContext jC{ mKernelInfo };
            auto                    ret = gemmGen->operator()(jC);
            if (ret == jitGeneratorError::success) {
                mIsJitGenerated = true;
            }
        }

        mJitGen = std::move(gemmGen);
    }

    jitKernelAdapter(const kernel_frame::packKernelInfo& pKI,
                     packBJitGeneratorRef                jitGen,
                     bool shouldGenerateKernels = true)
        : mPackKernelInfo(pKI)
        , mHasPackKernelInfo(true)
        , mIsJitGenerated(false)
    {
        auto packBGen = jitGen ? jitGen->clone() : nullptr;

        if ((shouldGenerateKernels) && (packBGen)) {
            packBJitGeneratorContext jC{ mPackKernelInfo };
            auto                     ret = packBGen->operator()(jC);
            if (ret == jitGeneratorError::success) {
                mIsJitGenerated = true;
            }
        }

        mJitGen = std::move(packBGen);
    }

    // Quant GEMM ctor. Stores the composed quantKernelInfo and
    // drives the quant generator with a quant-aware context. getKernelInfo()
    // returns &base so non-quant-aware readers still see a real kernelInfo;
    // getGemmQuantKernelInfo() returns the full quant key for the register.
    jitKernelAdapter(const kernel_frame::quantKernelInfo& qKI,
                     gemmQuantJitGeneratorRef             jitGen,
                     bool shouldGenerateKernels = true)
        : mKernelInfo(qKI.base)
        , mQuantKernelInfo(qKI)
        , mHasQuantKernelInfo(true)
        , mIsJitGenerated(false)
    {
        auto quantGen = jitGen ? jitGen->clone() : nullptr;

        if ((shouldGenerateKernels) && (quantGen)) {
            gemmQuantJitGeneratorContext jC{ mQuantKernelInfo };
            auto                         ret = quantGen->operator()(jC);
            if (ret == jitGeneratorError::success) {
                mIsJitGenerated = true;
            }
        }

        mJitGen = std::move(quantGen);
    }

    ~jitKernelAdapter() {}

    jitKernelAdapter(const jitKernelAdapter& other)            = delete;
    jitKernelAdapter& operator=(const jitKernelAdapter& other) = delete;

    jitKernelAdapter(jitKernelAdapter&& other)
        : mKernelInfo{ std::move(other.mKernelInfo) }
        , mPackKernelInfo{ std::move(other.mPackKernelInfo) }
        , mHasPackKernelInfo(other.mHasPackKernelInfo)
        , mQuantKernelInfo{ std::move(other.mQuantKernelInfo) }
        , mHasQuantKernelInfo(other.mHasQuantKernelInfo)
        , mJitGen(std::move(other.mJitGen))
        , mIsJitGenerated(std::move(other.mIsJitGenerated))
    {
        other.mIsJitGenerated = false;
        other.mJitGen.reset();
    }

    jitKernelAdapter& operator=(jitKernelAdapter&& other)
    {
        mKernelInfo           = std::move(other.mKernelInfo);
        mPackKernelInfo       = std::move(other.mPackKernelInfo);
        mHasPackKernelInfo    = other.mHasPackKernelInfo;
        mQuantKernelInfo      = std::move(other.mQuantKernelInfo);
        mHasQuantKernelInfo   = other.mHasQuantKernelInfo;
        mJitGen               = std::move(other.mJitGen);
        mIsJitGenerated       = std::move(other.mIsJitGenerated);
        other.mIsJitGenerated = false;
        other.mJitGen.reset();
        return *this;
    }

    bool isJitGenerated() const { return mIsJitGenerated; }

    operator bool() const { return isJitGenerated(); }

    // kernelBase interface implementation
    virtual std::vector<cpu_utils::isaFeature>& getIsaFeaturesForKernel()
        override
    {
        return mJitGen->getIsaFeaturesRequired();
    }

    virtual kernel_frame::kernelInfo* getKernelInfo() override
    {
        return std::addressof(mKernelInfo);
    }

    virtual kernel_frame::packKernelInfo* getPackKernelInfo() override
    {
        return mHasPackKernelInfo ? std::addressof(mPackKernelInfo) : nullptr;
    }

    virtual kernel_frame::quantKernelInfo* getGemmQuantKernelInfo() override
    {
        return mHasQuantKernelInfo ? std::addressof(mQuantKernelInfo) : nullptr;
    }

    virtual std::vector<kernel_frame::kernelDatatype>& getKernelDatatypes()
        override
    {
        return mJitGen->getKernelDatatypes();
    }

    /**
     * @brief Execute JIT kernel - delegates to JIT generator implementation
     * @param kP Kernel parameters from extensible framework
     * @return kernelError based on JIT generator result
     */
    virtual kernels::kernelError operator()(kernels::kernelParams* kP) override
    {
        if (!isJitGenerated()) {
            return kernels::kernelError::error;
        }

        // TODO: This is a temporary solution to get the kernelBase pointer from
        // the JIT generator. This is because the JIT generator now implements
        // both the jitGeneratorBase and kernelBase interfaces.
        if (mJitGen) {
            return mJitGen->executeKernel(kP);
        }

        return kernels::kernelError::error;
    }

  private:
    kernel_frame::kernelInfo          mKernelInfo;
    kernel_frame::packKernelInfo      mPackKernelInfo;
    bool                              mHasPackKernelInfo = false;
    kernel_frame::quantKernelInfo     mQuantKernelInfo;
    bool                              mHasQuantKernelInfo = false;
    std::unique_ptr<jitGeneratorBase> mJitGen;
    bool                              mIsJitGenerated;
};

} // namespace dlp::jit
