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

#ifndef AOCL_DLP_JIT_AMDZEN_GEMV_N1_STORE_EMITTER_HH
#define AOCL_DLP_JIT_AMDZEN_GEMV_N1_STORE_EMITTER_HH

#include "../jit_generator_utils.hh"
#include "../traits.hh"
#include "jit/jit_generator_base.hh"
#include "store_common.hh"
#include "xbyak/xbyak.h"

#include <array>
#include <cassert>

namespace amdzen::store {

enum class GemvN1SourceLayout : uint8_t
{
    packedLanes,
    extractedXmmChunks,
};

enum class GemvN1DestinationLayout : uint8_t
{
    contiguous,
    rowStrided,
};

enum class GemvN1PtrUpdate : uint8_t
{
    preserve,
    advanceFullRegs,
};

struct ExtractedXmmSource
{
    std::array<int, 4> registers{};
    std::size_t        registerCount = 0;
    int                validValues   = 0;
};

struct GemvN1Source
{
    GemvN1SourceLayout layout;
    RegSpan            packed;
    ExtractedXmmSource extracted;

    static GemvN1Source packedLanes(RegSpan block)
    {
        return { GemvN1SourceLayout::packedLanes, block, {} };
    }

    static GemvN1Source extractedXmm(const int*  registers,
                                     std::size_t registerCount,
                                     int         validValues)
    {
        GemvN1Source source{ GemvN1SourceLayout::extractedXmmChunks,
                             RegSpan{ 0, SourceRegWidth::zmm, 0 },
                             {} };
        assert(registerCount <= source.extracted.registers.size());
        assert(validValues >= 0 && validValues <= 16);
        assert(validValues <= static_cast<int>(registerCount * 4));
        assert(registerCount == 0 || registers != nullptr);
        source.extracted.registerCount = registerCount;
        source.extracted.validValues   = validValues;
        for (std::size_t i = 0; i < registerCount; ++i) {
            source.extracted.registers[i] = registers[i];
        }
        return source;
    }
};

struct GemvN1Destination
{
    GemvN1DestinationLayout layout;
    GemvN1PtrUpdate         ptrUpdate;
    // Contiguous stores advance after complete registers only. Scalar-strided
    // stores advance once for every stored value.
    Xbyak::Reg64 regYptr;
    Xbyak::Reg64 regRsC;
    Xbyak::Reg64 regAdvanceBytes;
    StoreMask    storeMask;

    static GemvN1Destination contiguous(Xbyak::Reg64 regYptr,
                                        StoreMask    storeMask)
    {
        return { GemvN1DestinationLayout::contiguous,
                 GemvN1PtrUpdate::preserve,
                 regYptr,
                 Xbyak::Reg64(0),
                 Xbyak::Reg64(0),
                 storeMask };
    }

    static GemvN1Destination contiguousAdvanceComplete(
        Xbyak::Reg64 regYptr, Xbyak::Reg64 regAdvanceBytes, StoreMask storeMask)
    {
        return { GemvN1DestinationLayout::contiguous,
                 GemvN1PtrUpdate::advanceFullRegs,
                 regYptr,
                 Xbyak::Reg64(0),
                 regAdvanceBytes,
                 storeMask };
    }

    static GemvN1Destination scalarStrided(Xbyak::Reg64 regYptr,
                                           Xbyak::Reg64 regRsC)
    {
        return { GemvN1DestinationLayout::rowStrided,
                 GemvN1PtrUpdate::preserve,
                 regYptr,
                 regRsC,
                 Xbyak::Reg64(0),
                 StoreMask::none() };
    }
};

struct GemvN1StoreRequest
{
    GemvN1Source      source;
    GemvN1Destination destination;
    StoreSpec         store;
    StoreTemps        temps;
};

template<utils::kernelInstrType KType>
class GemvN1StoreEmitter
{
  public:
    explicit GemvN1StoreEmitter(Xbyak::CodeGenerator& jit)
        : jit_(jit)
    {
    }

    dlp::jit::jitGeneratorError emit(const GemvN1StoreRequest& request);

  private:
    dlp::jit::jitGeneratorError emitFromF32(const GemvN1StoreRequest& request);
    dlp::jit::jitGeneratorError emitContiguousF32(
        const GemvN1StoreRequest& request);
    dlp::jit::jitGeneratorError emitContiguousBf16(
        const GemvN1StoreRequest& request);
    dlp::jit::jitGeneratorError emitScalarF32(
        const GemvN1StoreRequest& request);
    dlp::jit::jitGeneratorError emitScalarBf16(
        const GemvN1StoreRequest& request);
    dlp::jit::jitGeneratorError emitContiguousS32(
        const GemvN1StoreRequest& request, bool advanceCompleteRegisters);
    dlp::jit::jitGeneratorError emitScalarS32(
        const GemvN1StoreRequest& request);
    dlp::jit::jitGeneratorError emitInt8(const GemvN1StoreRequest& request);
    dlp::jit::jitGeneratorError emitF32Output(
        const GemvN1StoreRequest& request);
    dlp::jit::jitGeneratorError emitF16(const GemvN1StoreRequest& request);
    dlp::jit::jitGeneratorError emitBf16Software(
        const GemvN1StoreRequest& request);

    Xbyak::CodeGenerator& jit_;
};

extern template class GemvN1StoreEmitter<
    utils::kernelInstrType::avx512_zmm_32_reg>;

} // namespace amdzen::store

#endif // AOCL_DLP_JIT_AMDZEN_GEMV_N1_STORE_EMITTER_HH
