/*******************************************************************************
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
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
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

#ifndef AOCL_DLP_JIT_AMDZEN_GEMV_M1_STORE_EMITTER_HH
#define AOCL_DLP_JIT_AMDZEN_GEMV_M1_STORE_EMITTER_HH

#include "../jit_generator_utils.hh"
#include "../traits.hh"
#include "jit/jit_generator_base.hh"
#include "store_common.hh"
#include "xbyak/xbyak.h"

namespace amdzen::store {

struct GemvM1Destination
{
    // F32 and low-half BF16 stores preserve regYptr. Full-vector BF16
    // advances after every complete register, including a complete final one.
    Xbyak::Reg64 regYptr;
    StoreMask    finalMask;
};

struct GemvM1StoreRequest
{
    RegSpan           source;
    GemvM1Destination destination;
    StoreSpec         store;
    StoreTemps        temps;
};

template<utils::kernelInstrType KType>
class GemvM1StoreEmitter
{
  public:
    explicit GemvM1StoreEmitter(Xbyak::CodeGenerator& jit)
        : jit_(jit)
    {
    }

    dlp::jit::jitGeneratorError emit(const GemvM1StoreRequest& request);

  private:
    template<typename EmitRegister>
    void emitPreserveYPtrTraversal(const GemvM1StoreRequest& request,
                                   int                       bytesPerRegister,
                                   EmitRegister              emitRegister);

    dlp::jit::jitGeneratorError emitF32ToF32(const GemvM1StoreRequest& request);
    dlp::jit::jitGeneratorError emitF32ToBf16(
        const GemvM1StoreRequest& request);
    dlp::jit::jitGeneratorError emitF32ToS32(const GemvM1StoreRequest& request);
    dlp::jit::jitGeneratorError emitS32ToS32(const GemvM1StoreRequest& request);
    dlp::jit::jitGeneratorError emitS8(const GemvM1StoreRequest& request);
    dlp::jit::jitGeneratorError emitU8(const GemvM1StoreRequest& request);
    dlp::jit::jitGeneratorError emitS32ToF32(const GemvM1StoreRequest& request);
    dlp::jit::jitGeneratorError emitF16(const GemvM1StoreRequest& request);
    dlp::jit::jitGeneratorError emitBf16Software(
        const GemvM1StoreRequest& request);

    Xbyak::CodeGenerator& jit_;
};

extern template class GemvM1StoreEmitter<
    utils::kernelInstrType::avx512_zmm_32_reg>;

} // namespace amdzen::store

#endif // AOCL_DLP_JIT_AMDZEN_GEMV_M1_STORE_EMITTER_HH
