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
#include <vector>

#include "cpu_feature_list.hh"

namespace dlp::cpu_utils {

class cpuFeatureDetectorBase
{
  public:
    cpuFeatureDetectorBase()          = default;
    virtual ~cpuFeatureDetectorBase() = default;
    virtual bool hasFeatures(
        const std::vector<isaFeature>& featureList) const               = 0;
    virtual bool hasFeature(const isaFeature feature) const             = 0;
    virtual std::vector<isaFeature> getFeatures() const                 = 0;
    virtual bool                    isCpuVendor(cpuVendor vendor) const = 0;
    virtual cpuVendor               getCpuVendor() const                = 0;
    virtual int32_t                 getNumVectorRegisters() const       = 0;
    virtual int32_t                 getNumVectorMaskRegisters() const   = 0;
};

class cpuFeatures
{
    cpuFeatures();
    cpuFeatures(const cpuFeatures&)            = delete;
    cpuFeatures(cpuFeatures&&)                 = delete;
    cpuFeatures& operator=(const cpuFeatures&) = delete;
    cpuFeatures& operator=(cpuFeatures&&)      = delete;
    ~cpuFeatures()                             = default;

    std::unique_ptr<cpuFeatureDetectorBase> pDetector;

  public:
    static cpuFeatures& instance()
    {
        static cpuFeatures instance;
        return instance;
    }

    // Using a std::vector even if the performance will be bad compared to
    // say if bit poistions were used for ISA features. This vector based
    // approach will give much better flexibility as and when this class
    // needs to start supporting multiple ISAs.
    bool hasFeatures(const std::vector<isaFeature>& featureList)
    {
        return pDetector->hasFeatures(featureList);
    }

    bool hasFeature(const isaFeature feature)
    {
        return pDetector->hasFeature(feature);
    }

    std::vector<isaFeature> getFeatures() { return pDetector->getFeatures(); }

    cpuVendor getCpuVendor() { return pDetector->getCpuVendor(); }

    int32_t getNumVectorRegisters()
    {
        return pDetector->getNumVectorRegisters();
    }

    int32_t getNumVectorMaskRegisters()
    {
        return pDetector->getNumVectorMaskRegisters();
    }
};

inline cpuFeatures&
cpuFeaturesInstance()
{
    return cpuFeatures::instance();
}

} // namespace dlp::cpu_utils
