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

#include "gemm_store_emitter.hh"

namespace amdzen::store {

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemmStoreEmitter<KType>::emit(const GemmStoreRequest& request)
{
    using dlp::kernel_frame::DataType;
    assert(
        valuesPerRegister(request.tile.sourceRegWidth, request.store.srcRegType)
            > 0
        && "GEMM store: unsupported source type/width");
    const auto src = request.store.srcRegType;
    if (request.tile.sourceRegWidth == SourceRegWidth::ymm
        && !(src == DataType::f32
             && (request.store.dstMemType == DataType::f32
                 || (request.store.dstMemType == DataType::bf16
                     && request.store.isNativeBf16())))) {
        return dlp::jit::jitGeneratorError::notSupported;
    }
    switch (request.store.dstMemType) {
        case DataType::f32:
            if (src == DataType::f32)
                return emitF32ToF32(request);
            if (src == DataType::s32)
                return emitS32ToF32(request);
            break;
        case DataType::bf16:
            if (request.store.isSoftwareBf16()
                && (src == DataType::f32 || src == DataType::s32))
                return emitBf16Software(request);
            if (request.store.isNativeBf16() && src == DataType::f32)
                return emitF32ToBf16(request);
            break;
        case DataType::s32:
            if (src == DataType::f32)
                return emitF32ToS32(request);
            if (src == DataType::s32)
                return emitS32ToS32(request);
            break;
        case DataType::s8:
            if (src == DataType::f32)
                return emitF32ToS8(request);
            if (src == DataType::s32)
                return emitS32ToS8(request);
            break;
        case DataType::u8:
            if (src == DataType::f32)
                return emitF32ToU8(request);
            if (src == DataType::s32)
                return emitS32ToU8(request);
            break;
        case DataType::f16:
            if (src == DataType::f32)
                return emitF32ToF16(request);
            if (src == DataType::s32)
                return emitS32ToF16(request);
            break;
        default:
            break;
    }
    return dlp::jit::jitGeneratorError::notSupported;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemmStoreEmitter<KType>::emitF32ToF32(const GemmStoreRequest& request)
{
    const auto& tile = request.tile;
    const int   sourceValues =
        valuesPerRegister(tile.sourceRegWidth, request.store.srcRegType);
    const int destinationBits =
        dlp::utils::dataTypeSizeBits(request.store.dstMemType);
    assert(destinationBits > 0 && destinationBits % 8 == 0);
    const int bytesPerRegister = sourceValues * destinationBits / 8;
    for (int row = 0; row < tile.rowCount; ++row) {
        for (int column = 0; column < tile.registersPerRow; ++column) {
            const int sourceIndex =
                tile.accumBaseIdx + row * tile.registersPerRow + column;
            auto destination = Xbyak::util::ptr[request.destination.regCptr
                                                + column * bytesPerRegister];

            const auto finalMask = column == tile.registersPerRow - 1
                                       ? request.destination.finalMask
                                       : StoreMask::none();
            destination          = finalMask.apply(destination);
            if (tile.sourceRegWidth == SourceRegWidth::zmm) {
                jit_.vmovups(destination, Xbyak::Zmm(sourceIndex));
            } else {
                jit_.vmovups(destination, Xbyak::Ymm(sourceIndex));
            }
        }
        jit_.add(request.destination.regCptr, request.destination.regRsC);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemmStoreEmitter<KType>::emitF32ToBf16(const GemmStoreRequest& request)
{
    const auto& tile = request.tile;
    const int   sourceValues =
        valuesPerRegister(tile.sourceRegWidth, request.store.srcRegType);
    const int bytesPerRegister = sourceValues * sizeof(uint16_t);
    for (int row = 0; row < tile.rowCount; ++row) {
        for (int column = 0; column < tile.registersPerRow; ++column) {
            const int sourceIndex =
                tile.accumBaseIdx + row * tile.registersPerRow + column;
            const int scratchIndex = request.temps.zmm(column);
            auto      destination = Xbyak::util::ptr[request.destination.regCptr
                                                + column * bytesPerRegister];
            const auto finalMask  = column == tile.registersPerRow - 1
                                        ? request.destination.finalMask
                                        : StoreMask::none();
            destination           = finalMask.apply(destination);

            if (tile.sourceRegWidth == SourceRegWidth::zmm) {
                jit_.vcvtneps2bf16(Xbyak::Ymm(scratchIndex),
                                   Xbyak::Zmm(sourceIndex));
                jit_.vmovdqu16(destination, Xbyak::Ymm(scratchIndex));
            } else {
                jit_.vcvtneps2bf16(Xbyak::Xmm(scratchIndex),
                                   Xbyak::Ymm(sourceIndex));
                jit_.vmovdqu16(destination, Xbyak::Xmm(scratchIndex));
            }
        }
        jit_.add(request.destination.regCptr, request.destination.regRsC);
    }

    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemmStoreEmitter<KType>::emitF32ToS32(const GemmStoreRequest& request)
{
    const auto& tile = request.tile;
    const int   sourceValues =
        valuesPerRegister(tile.sourceRegWidth, request.store.srcRegType);
    for (int row = 0; row < tile.rowCount; ++row) {
        for (int column = 0; column < tile.registersPerRow; ++column) {
            const int sourceIndex =
                tile.accumBaseIdx + row * tile.registersPerRow + column;
            auto destination =
                Xbyak::util::ptr[request.destination.regCptr
                                 + column * sizeof(int32_t) * sourceValues];
            const auto finalMask = column == tile.registersPerRow - 1
                                       ? request.destination.finalMask
                                       : StoreMask::none();
            destination          = finalMask.apply(destination);
            jit_.vcvtps2dq(Xbyak::Zmm(sourceIndex), Xbyak::Zmm(sourceIndex));
            jit_.vmovdqu32(destination, Xbyak::Zmm(sourceIndex));
        }
        jit_.add(request.destination.regCptr, request.destination.regRsC);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemmStoreEmitter<KType>::emitS32ToS32(const GemmStoreRequest& request)
{
    const auto& tile = request.tile;
    const int   sourceValues =
        valuesPerRegister(tile.sourceRegWidth, request.store.srcRegType);
    for (int row = 0; row < tile.rowCount; ++row) {
        for (int column = 0; column < tile.registersPerRow; ++column) {
            const int sourceIndex =
                tile.accumBaseIdx + row * tile.registersPerRow + column;
            auto destination =
                Xbyak::util::ptr[request.destination.regCptr
                                 + column * sizeof(int32_t) * sourceValues];
            const auto finalMask = column == tile.registersPerRow - 1
                                       ? request.destination.finalMask
                                       : StoreMask::none();
            destination          = finalMask.apply(destination);
            jit_.vmovdqu32(destination, Xbyak::Zmm(sourceIndex));
        }
        jit_.add(request.destination.regCptr, request.destination.regRsC);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemmStoreEmitter<KType>::emitF32ToS8(const GemmStoreRequest& request)
{
    const auto& tile = request.tile;
    const int   sourceValues =
        valuesPerRegister(tile.sourceRegWidth, request.store.srcRegType);
    const int          bound     = request.temps.zmm(0);
    const Xbyak::Reg64 immediate = request.temps.gpr();
    for (int row = 0; row < tile.rowCount; ++row) {
        for (int column = 0; column < tile.registersPerRow; ++column) {
            const int sourceIndex =
                tile.accumBaseIdx + row * tile.registersPerRow + column;
            auto destination     = Xbyak::util::ptr[request.destination.regCptr
                                                + column * sourceValues];
            const auto finalMask = column == tile.registersPerRow - 1
                                       ? request.destination.finalMask
                                       : StoreMask::none();
            destination          = finalMask.apply(destination);
            emitF32ClampToInt8Range(jit_, Xbyak::Zmm(sourceIndex),
                                    Xbyak::Zmm(bound), immediate,
                                    dlp::kernel_frame::DataType::s8);
            jit_.vcvtps2dq(Xbyak::Zmm(sourceIndex), Xbyak::Zmm(sourceIndex));
            jit_.vpmovsdb(destination, Xbyak::Zmm(sourceIndex));
        }
        jit_.add(request.destination.regCptr, request.destination.regRsC);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemmStoreEmitter<KType>::emitS32ToS8(const GemmStoreRequest& request)
{
    const auto& tile = request.tile;
    const int   sourceValues =
        valuesPerRegister(tile.sourceRegWidth, request.store.srcRegType);
    for (int row = 0; row < tile.rowCount; ++row) {
        for (int column = 0; column < tile.registersPerRow; ++column) {
            const int sourceIndex =
                tile.accumBaseIdx + row * tile.registersPerRow + column;
            auto destination     = Xbyak::util::ptr[request.destination.regCptr
                                                + column * sourceValues];
            const auto finalMask = column == tile.registersPerRow - 1
                                       ? request.destination.finalMask
                                       : StoreMask::none();
            destination          = finalMask.apply(destination);
            jit_.vpmovsdb(destination, Xbyak::Zmm(sourceIndex));
        }
        jit_.add(request.destination.regCptr, request.destination.regRsC);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemmStoreEmitter<KType>::emitF32ToU8(const GemmStoreRequest& request)
{
    const auto& tile = request.tile;
    const int   sourceValues =
        valuesPerRegister(tile.sourceRegWidth, request.store.srcRegType);
    const int          bound     = request.temps.zmm(0);
    const Xbyak::Reg64 immediate = request.temps.gpr();
    for (int row = 0; row < tile.rowCount; ++row) {
        for (int column = 0; column < tile.registersPerRow; ++column) {
            const int source =
                tile.accumBaseIdx + row * tile.registersPerRow + column;
            auto destination     = Xbyak::util::ptr[request.destination.regCptr
                                                + column * sourceValues];
            const auto finalMask = column == tile.registersPerRow - 1
                                       ? request.destination.finalMask
                                       : StoreMask::none();
            destination          = finalMask.apply(destination);
            emitF32ClampToInt8Range(jit_, Xbyak::Zmm(source), Xbyak::Zmm(bound),
                                    immediate, dlp::kernel_frame::DataType::u8);
            jit_.vcvtps2dq(Xbyak::Zmm(source), Xbyak::Zmm(source));
            jit_.vpmovdb(destination, Xbyak::Zmm(source));
        }
        jit_.add(request.destination.regCptr, request.destination.regRsC);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemmStoreEmitter<KType>::emitS32ToU8(const GemmStoreRequest& request)
{
    const auto& tile = request.tile;
    const int   sourceValues =
        valuesPerRegister(tile.sourceRegWidth, request.store.srcRegType);
    const int          result    = request.temps.zmm(0);
    const int          zero      = request.temps.zmm(1);
    const int          max       = request.temps.zmm(2);
    const Xbyak::Reg64 immediate = request.temps.gpr();

    jit_.vpxord(Xbyak::Zmm(zero), Xbyak::Zmm(zero), Xbyak::Zmm(zero));
    jit_.mov(immediate, 255);
    jit_.vpbroadcastd(Xbyak::Zmm(max), immediate.cvt32());
    for (int row = 0; row < tile.rowCount; ++row) {
        for (int column = 0; column < tile.registersPerRow; ++column) {
            const int source =
                tile.accumBaseIdx + row * tile.registersPerRow + column;
            auto destination     = Xbyak::util::ptr[request.destination.regCptr
                                                + column * sourceValues];
            const auto finalMask = column == tile.registersPerRow - 1
                                       ? request.destination.finalMask
                                       : StoreMask::none();
            destination          = finalMask.apply(destination);
            jit_.vpmaxsd(Xbyak::Zmm(result), Xbyak::Zmm(source),
                         Xbyak::Zmm(zero));
            jit_.vpminsd(Xbyak::Zmm(result), Xbyak::Zmm(result),
                         Xbyak::Zmm(max));
            jit_.vpmovdb(destination, Xbyak::Zmm(result));
        }
        jit_.add(request.destination.regCptr, request.destination.regRsC);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemmStoreEmitter<KType>::emitS32ToF32(const GemmStoreRequest& request)
{
    const auto& tile = request.tile;
    const int   sourceValues =
        valuesPerRegister(tile.sourceRegWidth, request.store.srcRegType);
    const int bytesPerRegister = sourceValues * sizeof(float);
    for (int row = 0; row < tile.rowCount; ++row) {
        for (int column = 0; column < tile.registersPerRow; ++column) {
            const int source =
                tile.accumBaseIdx + row * tile.registersPerRow + column;
            auto destination     = Xbyak::util::ptr[request.destination.regCptr
                                                + column * bytesPerRegister];
            const auto finalMask = column == tile.registersPerRow - 1
                                       ? request.destination.finalMask
                                       : StoreMask::none();
            destination          = finalMask.apply(destination);
            jit_.vcvtdq2ps(Xbyak::Zmm(source), Xbyak::Zmm(source));
            jit_.vmovups(destination, Xbyak::Zmm(source));
        }
        jit_.add(request.destination.regCptr, request.destination.regRsC);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemmStoreEmitter<KType>::emitF32ToF16(const GemmStoreRequest& request)
{
    const auto& tile = request.tile;
    const int   sourceValues =
        valuesPerRegister(tile.sourceRegWidth, request.store.srcRegType);
    const int bytesPerRegister = sourceValues * sizeof(uint16_t);
    const int packed           = request.temps.zmm(0);
    for (int row = 0; row < tile.rowCount; ++row) {
        for (int column = 0; column < tile.registersPerRow; ++column) {
            const int source =
                tile.accumBaseIdx + row * tile.registersPerRow + column;
            auto destination     = Xbyak::util::ptr[request.destination.regCptr
                                                + column * bytesPerRegister];
            const auto finalMask = column == tile.registersPerRow - 1
                                       ? request.destination.finalMask
                                       : StoreMask::none();
            destination          = finalMask.apply(destination);
            jit_.vcvtps2ph(Xbyak::Ymm(packed), Xbyak::Zmm(source), 0);
            jit_.vmovdqu16(destination, Xbyak::Ymm(packed));
        }
        jit_.add(request.destination.regCptr, request.destination.regRsC);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemmStoreEmitter<KType>::emitS32ToF16(const GemmStoreRequest& request)
{
    const auto& tile = request.tile;
    const int   sourceValues =
        valuesPerRegister(tile.sourceRegWidth, request.store.srcRegType);
    const int bytesPerRegister = sourceValues * sizeof(uint16_t);
    const int packed           = request.temps.zmm(0);
    for (int row = 0; row < tile.rowCount; ++row) {
        for (int column = 0; column < tile.registersPerRow; ++column) {
            const int source =
                tile.accumBaseIdx + row * tile.registersPerRow + column;
            auto destination     = Xbyak::util::ptr[request.destination.regCptr
                                                + column * bytesPerRegister];
            const auto finalMask = column == tile.registersPerRow - 1
                                       ? request.destination.finalMask
                                       : StoreMask::none();
            destination          = finalMask.apply(destination);
            jit_.vcvtdq2ps(Xbyak::Zmm(source), Xbyak::Zmm(source));
            jit_.vcvtps2ph(Xbyak::Ymm(packed), Xbyak::Zmm(source), 0);
            jit_.vmovdqu16(destination, Xbyak::Ymm(packed));
        }
        jit_.add(request.destination.regCptr, request.destination.regRsC);
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemmStoreEmitter<KType>::emitBf16Software(const GemmStoreRequest& request)
{
    const auto& tile = request.tile;
    const int   sourceValues =
        valuesPerRegister(tile.sourceRegWidth, request.store.srcRegType);
    const int          bytesPerRegister = sourceValues * sizeof(uint16_t);
    const int          lsbMask          = request.temps.zmm(0);
    const int          roundBias        = request.temps.zmm(1);
    const int          lsbTemp          = request.temps.zmm(2);
    const Xbyak::Reg64 immediate        = request.temps.gpr();
    const bool         convertFromS32 =
        request.store.srcRegType == dlp::kernel_frame::DataType::s32;
    jit_.mov(immediate, 1);
    jit_.vpbroadcastd(Xbyak::Zmm(lsbMask), immediate.cvt32());
    jit_.mov(immediate, 0x7fff);
    jit_.vpbroadcastd(Xbyak::Zmm(roundBias), immediate.cvt32());
    for (int row = 0; row < tile.rowCount; ++row) {
        for (int column = 0; column < tile.registersPerRow; ++column) {
            const int source =
                tile.accumBaseIdx + row * tile.registersPerRow + column;
            auto destination     = Xbyak::util::ptr[request.destination.regCptr
                                                + column * bytesPerRegister];
            const auto finalMask = column == tile.registersPerRow - 1
                                       ? request.destination.finalMask
                                       : StoreMask::none();
            destination          = finalMask.apply(destination);
            if (convertFromS32) {
                jit_.vcvtdq2ps(Xbyak::Zmm(source), Xbyak::Zmm(source));
            }
            // Round-to-nearest-even: rounded = (f32 + 0x7FFF + lsb) >> 16.
            jit_.vpsrld(Xbyak::Zmm(lsbTemp), Xbyak::Zmm(source), 16);
            jit_.vpandd(Xbyak::Zmm(lsbTemp), Xbyak::Zmm(lsbTemp),
                        Xbyak::Zmm(lsbMask));
            jit_.vpaddd(Xbyak::Zmm(source), Xbyak::Zmm(source),
                        Xbyak::Zmm(roundBias));
            jit_.vpaddd(Xbyak::Zmm(source), Xbyak::Zmm(source),
                        Xbyak::Zmm(lsbTemp));
            jit_.vpsrld(Xbyak::Zmm(source), Xbyak::Zmm(source), 16);
            jit_.vpmovdw(Xbyak::Ymm(source), Xbyak::Zmm(source));
            jit_.vmovdqu16(destination, Xbyak::Ymm(source));
        }
        jit_.add(request.destination.regCptr, request.destination.regRsC);
    }
    return dlp::jit::jitGeneratorError::success;
}

template class GemmStoreEmitter<utils::kernelInstrType::avx512_zmm_32_reg>;

} // namespace amdzen::store
