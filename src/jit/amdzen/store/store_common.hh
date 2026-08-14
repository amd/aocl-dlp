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

#ifndef AOCL_DLP_JIT_AMDZEN_STORE_COMMON_HH
#define AOCL_DLP_JIT_AMDZEN_STORE_COMMON_HH

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "kernel_frame/kernel_frame_base.hh"
#include "utils/ctype_utils.hh"
#include "xbyak/xbyak.h"

namespace amdzen::store {

enum class SourceRegWidth : uint8_t
{
    ymm = 32,
    zmm = 64,
};

constexpr int
registerSizeBytes(SourceRegWidth width) noexcept
{
    return static_cast<int>(width);
}

constexpr int
valuesPerRegister(SourceRegWidth              width,
                  dlp::kernel_frame::DataType sourceType) noexcept
{
    const int typeBits = dlp::utils::dataTypeSizeBits(sourceType);
    return typeBits == 0 ? 0 : registerSizeBytes(width) * 8 / typeBits;
}

struct RegSpan
{
    int            baseIdx;
    SourceRegWidth width;
    int            numValues;

    constexpr int valuesPerRegister(
        dlp::kernel_frame::DataType sourceType) const noexcept
    {
        return amdzen::store::valuesPerRegister(width, sourceType);
    }

    constexpr int registerCount(
        dlp::kernel_frame::DataType sourceType) const noexcept
    {
        const int capacity = valuesPerRegister(sourceType);
        return numValues <= 0 || capacity == 0
                   ? 0
                   : (numValues + capacity - 1) / capacity;
    }

    constexpr int validValuesInLastRegister(
        dlp::kernel_frame::DataType sourceType) const noexcept
    {
        const int capacity = valuesPerRegister(sourceType);
        if (numValues <= 0 || capacity == 0) {
            return 0;
        }
        const int remainder = numValues % capacity;
        return remainder == 0 ? capacity : remainder;
    }

    constexpr int valuesInRegister(
        int index, dlp::kernel_frame::DataType sourceType) const noexcept
    {
        const int count = registerCount(sourceType);
        if (index < 0 || index >= count) {
            return 0;
        }
        return index == count - 1 ? validValuesInLastRegister(sourceType)
                                  : valuesPerRegister(sourceType);
    }
};

class StoreSpec
{
  public:
    dlp::kernel_frame::DataType srcRegType;
    dlp::kernel_frame::DataType dstMemType;

    static StoreSpec convert(dlp::kernel_frame::DataType src,
                             dlp::kernel_frame::DataType dst)
    {
        assert(dst != dlp::kernel_frame::DataType::bf16);
        return { src, dst, Bf16Mode::notBf16 };
    }

    static StoreSpec nativeBf16(dlp::kernel_frame::DataType src)
    {
        return { src, dlp::kernel_frame::DataType::bf16, Bf16Mode::native };
    }

    static StoreSpec softwareBf16(dlp::kernel_frame::DataType src)
    {
        return { src, dlp::kernel_frame::DataType::bf16,
                 Bf16Mode::softwareRne };
    }

    bool isNativeBf16() const { return bf16Mode_ == Bf16Mode::native; }
    bool isSoftwareBf16() const { return bf16Mode_ == Bf16Mode::softwareRne; }

  private:
    enum class Bf16Mode : uint8_t
    {
        notBf16,
        native,
        softwareRne,
    };

    StoreSpec(dlp::kernel_frame::DataType src,
              dlp::kernel_frame::DataType dst,
              Bf16Mode                    bf16Mode)
        : srcRegType(src)
        , dstMemType(dst)
        , bf16Mode_(bf16Mode)
    {
    }

    Bf16Mode bf16Mode_;
};

enum class StoreMaskKind : uint8_t
{
    none,
    opmask,
};

struct StoreMask
{
    StoreMaskKind kind;
    int           maskRegIdx;

    static StoreMask none() { return { StoreMaskKind::none, -1 }; }

    static StoreMask opmask(const Xbyak::Opmask& mask)
    {
        return { StoreMaskKind::opmask, mask.getIdx() };
    }

    Xbyak::Address apply(Xbyak::Address address) const
    {
        if (kind == StoreMaskKind::opmask) {
            address = address | Xbyak::Opmask(maskRegIdx);
        }
        return address;
    }
};

inline void
emitF32ClampToInt8Range(Xbyak::CodeGenerator&       jit,
                        const Xbyak::Zmm&           source,
                        const Xbyak::Zmm&           bound,
                        const Xbyak::Reg64&         immediate,
                        dlp::kernel_frame::DataType destinationType)
{
    using dlp::kernel_frame::DataType;
    assert(destinationType == DataType::s8 || destinationType == DataType::u8);

    const uint32_t upperBits = destinationType == DataType::u8 ? 0x437f0000u
                                                               : 0x42fe0000u;
    const uint32_t lowerBits = destinationType == DataType::u8 ? 0x00000000u
                                                               : 0xc3000000u;

    // Keep the bound as source 2 so unordered lanes select the upper limit.
    jit.mov(immediate, upperBits);
    jit.vpbroadcastd(bound, immediate.cvt32());
    jit.vminps(source, source, bound);
    jit.mov(immediate, lowerBits);
    jit.vpbroadcastd(bound, immediate.cvt32());
    jit.vmaxps(source, source, bound);
}

class StoreTemps
{
  public:
    static StoreTemps none() { return StoreTemps(-1, -1); }

    static StoreTemps withZmm(int baseIdx) { return StoreTemps(baseIdx, -1); }

    static StoreTemps withZmmAndGpr(int baseIdx, Xbyak::Reg64 gpr)
    {
        return StoreTemps(baseIdx, gpr.getIdx());
    }

    int          zmm(int offset) const { return zmmBaseIdx_ + offset; }
    Xbyak::Reg64 gpr() const { return Xbyak::Reg64(gprIdx_); }

  private:
    StoreTemps(int zmmBaseIdx, int gprIdx)
        : zmmBaseIdx_(zmmBaseIdx)
        , gprIdx_(gprIdx)
    {
    }

    int zmmBaseIdx_;
    int gprIdx_;
};

} // namespace amdzen::store

#endif // AOCL_DLP_JIT_AMDZEN_STORE_COMMON_HH
