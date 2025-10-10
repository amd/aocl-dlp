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

#include "de_input.hh"
#include "kernel_frame/kernel_frame_base.hh"

namespace dlp::de {

class iDEBackend
{
  public:
    virtual ~iDEBackend() = default;
    virtual std::optional<dlp::kernel_frame::kernelInfo> getKernelInfoForInput(
        iDEInput* in) = 0;
};

class gemmF32DEBackend : public iDEBackend
{
    static inline void setKernelOps(
        dlp::kernel_frame::kernelOpsMetaData* metaData,
        lpgemm_post_op*                       post_op,
        kernel_frame::kernelDatatype          k_dtype);
    bool                                isAvx512;
    bool                                isAvx2;
    int32_t                             numRegisters;
    kernel_frame::kernelInstrPreference eKernelInstPref;
    bool                                canGenerateKernelInfo;

  public:
    gemmF32DEBackend();
    ~gemmF32DEBackend()                                  = default;
    gemmF32DEBackend(const gemmF32DEBackend&)            = delete;
    gemmF32DEBackend(gemmF32DEBackend&&)                 = delete;
    gemmF32DEBackend& operator=(const gemmF32DEBackend&) = delete;
    gemmF32DEBackend& operator=(gemmF32DEBackend&&)      = delete;

    std::optional<dlp::kernel_frame::kernelInfo> getKernelInfoForInput(
        iDEInput* in) override;
};

} // namespace dlp::de
