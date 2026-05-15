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

#include "x86_cpu_features.hh"

namespace dlp::cpu_utils {

// This reference class is to ensure pDetector is always initialized
// in the cpuFeatures class, even if the architecture is not x86_64 or arm.
class referenceCpuFeatureDetector : public cpuFeatureDetectorBase
{
  public:
    referenceCpuFeatureDetector()          = default;
    virtual ~referenceCpuFeatureDetector() = default;

    bool hasFeatures(
        [[maybe_unused]] const std::vector<isaFeature>& featureList) const final
    {
        return false;
    }

    bool hasFeature([[maybe_unused]] const isaFeature feature) const final
    {
        return false;
    }

    std::vector<isaFeature> getFeatures() const final
    {
        return std::vector<isaFeature>{};
    }

    bool isCpuVendor([[maybe_unused]] cpuVendor vendor) const final
    {
        return false;
    }

    cpuVendor getCpuVendor() const final { return cpuVendor::invalid; }

    int32_t getNumVectorRegisters() const final { return 0; }

    int32_t getNumVectorMaskRegisters() const final { return 0; }
};

cpuFeatures::cpuFeatures()
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386)                  \
    || defined(_M_IX86)
    pDetector.reset(new x86CpuFeatureDetector);
#else
    pDetector.reset(new referenceCpuFeatureDetector);
#endif
}

} // namespace dlp::cpu_utils
