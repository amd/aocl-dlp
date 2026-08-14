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

#include "gemv_m1_store_emitter.hh"

namespace amdzen::store {

template<utils::kernelInstrType KType>
template<typename EmitRegister>
void
GemvM1StoreEmitter<KType>::emitPreserveYPtrTraversal(
    const GemvM1StoreRequest& request,
    int                       bytesPerRegister,
    EmitRegister              emitRegister)
{
    const auto& source        = request.source;
    const int   registerCount = source.registerCount(request.store.srcRegType);
    for (int i = 0; i < registerCount; ++i) {
        auto destination = Xbyak::util::ptr[request.destination.regYptr
                                            + i * bytesPerRegister];
        if (i == registerCount - 1) {
            destination = request.destination.finalMask.apply(destination);
        }
        emitRegister(source.baseIdx + i, destination);
    }
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvM1StoreEmitter<KType>::emit(const GemvM1StoreRequest& request)
{
    using dlp::kernel_frame::DataType;
    assert(request.source.valuesPerRegister(request.store.srcRegType) > 0
           && "GEMV-M1 store: unsupported source type/width");
    const auto src = request.store.srcRegType;
    if (request.source.width == SourceRegWidth::ymm
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
            if (src == DataType::f32 || src == DataType::s32)
                return emitS8(request);
            break;
        case DataType::u8:
            if (src == DataType::f32 || src == DataType::s32)
                return emitU8(request);
            break;
        case DataType::f16:
            if (src == DataType::f32 || src == DataType::s32)
                return emitF16(request);
            break;
        default:
            break;
    }
    return dlp::jit::jitGeneratorError::notSupported;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvM1StoreEmitter<KType>::emitF32ToF32(const GemvM1StoreRequest& request)
{
    const auto& source   = request.source;
    const int   capacity = source.valuesPerRegister(request.store.srcRegType);
    const int   bytesPerReg = capacity * sizeof(float);
    emitPreserveYPtrTraversal(
        request, bytesPerReg,
        [&](int sourceRegister, const Xbyak::Address& destination) {
            if (source.width == SourceRegWidth::zmm) {
                jit_.vmovups(destination, Xbyak::Zmm(sourceRegister));
            } else {
                jit_.vmovups(destination, Xbyak::Ymm(sourceRegister));
            }
        });
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvM1StoreEmitter<KType>::emitF32ToBf16(const GemvM1StoreRequest& request)
{
    const auto& source   = request.source;
    const int   capacity = source.valuesPerRegister(request.store.srcRegType);
    const int   bytesPerReg = capacity * sizeof(uint16_t);
    const bool  hasFringe =
        source.validValuesInLastRegister(request.store.srcRegType) != capacity;
    if (source.width != SourceRegWidth::zmm) {
        emitPreserveYPtrTraversal(
            request, bytesPerReg,
            [&](int sourceRegister, const Xbyak::Address& destination) {
                jit_.vcvtneps2bf16(Xbyak::Xmm(sourceRegister),
                                   Xbyak::Ymm(sourceRegister));
                jit_.vmovdqu16(destination, Xbyak::Xmm(sourceRegister));
            });
        return dlp::jit::jitGeneratorError::success;
    }
    const int registerCount = source.registerCount(request.store.srcRegType);
    for (int i = 0; i < registerCount; ++i) {
        auto destination = Xbyak::util::ptr[request.destination.regYptr];
        if (i == registerCount - 1) {
            destination = request.destination.finalMask.apply(destination);
        }
        jit_.vcvtneps2bf16(Xbyak::Ymm(source.baseIdx + i),
                           Xbyak::Zmm(source.baseIdx + i));
        jit_.vmovdqu16(destination, Xbyak::Ymm(source.baseIdx + i));
        if (!(hasFringe && i == registerCount - 1)) {
            jit_.lea(
                request.destination.regYptr,
                Xbyak::util::ptr[request.destination.regYptr + bytesPerReg]);
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvM1StoreEmitter<KType>::emitF32ToS32(const GemvM1StoreRequest& request)
{
    const auto& source   = request.source;
    const int   capacity = source.valuesPerRegister(request.store.srcRegType);
    const int   bytesPerReg = capacity * sizeof(int32_t);
    emitPreserveYPtrTraversal(
        request, bytesPerReg,
        [&](int sourceRegister, const Xbyak::Address& destination) {
            jit_.vcvtps2dq(Xbyak::Zmm(sourceRegister),
                           Xbyak::Zmm(sourceRegister));
            jit_.vmovdqu32(destination, Xbyak::Zmm(sourceRegister));
        });
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvM1StoreEmitter<KType>::emitS32ToS32(const GemvM1StoreRequest& request)
{
    const auto& source   = request.source;
    const int   capacity = source.valuesPerRegister(request.store.srcRegType);
    const int   bytesPerReg = capacity * sizeof(int32_t);
    emitPreserveYPtrTraversal(
        request, bytesPerReg,
        [&](int sourceRegister, const Xbyak::Address& destination) {
            jit_.vmovdqu32(destination, Xbyak::Zmm(sourceRegister));
        });
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvM1StoreEmitter<KType>::emitS8(const GemvM1StoreRequest& request)
{
    const bool fromF32 =
        request.store.srcRegType == dlp::kernel_frame::DataType::f32;
    const int destinationBits =
        dlp::utils::dataTypeSizeBits(request.store.dstMemType);
    assert(destinationBits > 0 && destinationBits % 8 == 0);
    emitPreserveYPtrTraversal(
        request,
        request.source.valuesPerRegister(request.store.srcRegType)
            * destinationBits / 8,
        [&](int src, const Xbyak::Address& dst) {
            if (fromF32) {
                emitF32ClampToInt8Range(
                    jit_, Xbyak::Zmm(src), Xbyak::Zmm(request.temps.zmm(0)),
                    request.temps.gpr(), dlp::kernel_frame::DataType::s8);
                jit_.vcvtps2dq(Xbyak::Zmm(src), Xbyak::Zmm(src));
            }
            jit_.vpmovsdb(dst, Xbyak::Zmm(src));
        });
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvM1StoreEmitter<KType>::emitU8(const GemvM1StoreRequest& request)
{
    const int          result    = request.temps.zmm(0);
    const int          zero      = request.temps.zmm(1);
    const int          upper     = request.temps.zmm(2);
    const Xbyak::Reg64 immediate = request.temps.gpr();
    const bool         fromF32 =
        request.store.srcRegType == dlp::kernel_frame::DataType::f32;
    if (!fromF32) {
        jit_.vpxord(Xbyak::Zmm(zero), Xbyak::Zmm(zero), Xbyak::Zmm(zero));
        jit_.mov(immediate, 255);
        jit_.vpbroadcastd(Xbyak::Zmm(upper), immediate.cvt32());
    }
    const int destinationBits =
        dlp::utils::dataTypeSizeBits(request.store.dstMemType);
    assert(destinationBits > 0 && destinationBits % 8 == 0);
    emitPreserveYPtrTraversal(
        request,
        request.source.valuesPerRegister(request.store.srcRegType)
            * destinationBits / 8,
        [&](int src, const Xbyak::Address& dst) {
            if (fromF32) {
                emitF32ClampToInt8Range(jit_, Xbyak::Zmm(src), Xbyak::Zmm(zero),
                                        immediate,
                                        dlp::kernel_frame::DataType::u8);
                jit_.vcvtps2dq(Xbyak::Zmm(src), Xbyak::Zmm(src));
                jit_.vpmovdb(dst, Xbyak::Zmm(src));
            } else {
                jit_.vpmaxsd(Xbyak::Zmm(result), Xbyak::Zmm(src),
                             Xbyak::Zmm(zero));
                jit_.vpminsd(Xbyak::Zmm(result), Xbyak::Zmm(result),
                             Xbyak::Zmm(upper));
                jit_.vpmovdb(dst, Xbyak::Zmm(result));
            }
        });
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvM1StoreEmitter<KType>::emitS32ToF32(const GemvM1StoreRequest& request)
{
    const int destinationBits =
        dlp::utils::dataTypeSizeBits(request.store.dstMemType);
    assert(destinationBits > 0 && destinationBits % 8 == 0);
    emitPreserveYPtrTraversal(
        request,
        request.source.valuesPerRegister(request.store.srcRegType)
            * destinationBits / 8,
        [&](int sourceRegister, const Xbyak::Address& destination) {
            jit_.vcvtdq2ps(Xbyak::Zmm(sourceRegister),
                           Xbyak::Zmm(sourceRegister));
            jit_.vmovups(destination, Xbyak::Zmm(sourceRegister));
        });
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvM1StoreEmitter<KType>::emitF16(const GemvM1StoreRequest& request)
{
    const int  half = request.temps.zmm(0);
    const bool fromF32 =
        request.store.srcRegType == dlp::kernel_frame::DataType::f32;
    const int destinationBits =
        dlp::utils::dataTypeSizeBits(request.store.dstMemType);
    assert(destinationBits > 0 && destinationBits % 8 == 0);
    emitPreserveYPtrTraversal(
        request,
        request.source.valuesPerRegister(request.store.srcRegType)
            * destinationBits / 8,
        [&](int sourceRegister, const Xbyak::Address& destination) {
            if (!fromF32) {
                jit_.vcvtdq2ps(Xbyak::Zmm(sourceRegister),
                               Xbyak::Zmm(sourceRegister));
            }
            jit_.vcvtps2ph(Xbyak::Ymm(half), Xbyak::Zmm(sourceRegister), 0);
            jit_.vmovdqu16(destination, Xbyak::Ymm(half));
        });
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvM1StoreEmitter<KType>::emitBf16Software(const GemvM1StoreRequest& request)
{
    const int          lsbMask   = request.temps.zmm(0);
    const int          roundBias = request.temps.zmm(1);
    const int          lsb       = request.temps.zmm(2);
    const Xbyak::Reg64 gpr       = request.temps.gpr();
    jit_.mov(gpr, 1);
    jit_.vpbroadcastd(Xbyak::Zmm(lsbMask), gpr.cvt32());
    jit_.mov(gpr, 0x7fff);
    jit_.vpbroadcastd(Xbyak::Zmm(roundBias), gpr.cvt32());
    const int destinationBits =
        dlp::utils::dataTypeSizeBits(request.store.dstMemType);
    assert(destinationBits > 0 && destinationBits % 8 == 0);
    emitPreserveYPtrTraversal(
        request,
        request.source.valuesPerRegister(request.store.srcRegType)
            * destinationBits / 8,
        [&](int src, const Xbyak::Address& destination) {
            if (request.store.srcRegType == dlp::kernel_frame::DataType::s32) {
                jit_.vcvtdq2ps(Xbyak::Zmm(src), Xbyak::Zmm(src));
            }
            jit_.vpsrld(Xbyak::Zmm(lsb), Xbyak::Zmm(src), 16);
            jit_.vpandd(Xbyak::Zmm(lsb), Xbyak::Zmm(lsb), Xbyak::Zmm(lsbMask));
            jit_.vpaddd(Xbyak::Zmm(src), Xbyak::Zmm(src),
                        Xbyak::Zmm(roundBias));
            jit_.vpaddd(Xbyak::Zmm(src), Xbyak::Zmm(src), Xbyak::Zmm(lsb));
            jit_.vpsrld(Xbyak::Zmm(src), Xbyak::Zmm(src), 16);
            jit_.vpmovdw(Xbyak::Ymm(src), Xbyak::Zmm(src));
            jit_.vmovdqu16(destination, Xbyak::Ymm(src));
        });
    return dlp::jit::jitGeneratorError::success;
}

template class GemvM1StoreEmitter<utils::kernelInstrType::avx512_zmm_32_reg>;

} // namespace amdzen::store
