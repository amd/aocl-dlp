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

#include "gemv_n1_store_emitter.hh"

namespace amdzen::store {

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvN1StoreEmitter<KType>::emit(const GemvN1StoreRequest& request)
{
    using dlp::kernel_frame::DataType;
    const auto src = request.store.srcRegType;
    if (request.source.layout == GemvN1SourceLayout::extractedXmmChunks) {
        if (request.destination.layout != GemvN1DestinationLayout::rowStrided)
            return dlp::jit::jitGeneratorError::notSupported;
        if (src == DataType::f32 && request.store.dstMemType == DataType::f32)
            return emitScalarF32(request);
        if (src == DataType::f32 && request.store.dstMemType == DataType::bf16
            && request.store.isNativeBf16())
            return emitScalarBf16(request);
        if (src == DataType::s32 && request.store.dstMemType == DataType::s32)
            return emitScalarS32(request);
        return dlp::jit::jitGeneratorError::notSupported;
    }

    assert(request.source.packed.valuesPerRegister(request.store.srcRegType) > 0
           && "GEMV-N1 store: unsupported source type/width");
    if (request.source.packed.width == SourceRegWidth::ymm
        && !(request.destination.layout == GemvN1DestinationLayout::rowStrided
             && src == DataType::f32
             && (request.store.dstMemType == DataType::f32
                 || (request.store.dstMemType == DataType::bf16
                     && request.store.isNativeBf16())))) {
        return dlp::jit::jitGeneratorError::notSupported;
    }
    switch (request.store.dstMemType) {
        case DataType::f32:
            if (src == DataType::f32)
                return emitFromF32(request);
            if (src == DataType::s32)
                return emitF32Output(request);
            break;
        case DataType::bf16:
            if (request.store.isSoftwareBf16()
                && (src == DataType::f32 || src == DataType::s32))
                return emitBf16Software(request);
            if (request.store.isNativeBf16() && src == DataType::f32)
                return emitFromF32(request);
            break;
        case DataType::s32:
            if (src == DataType::f32 || src == DataType::s32)
                return request.destination.layout
                               == GemvN1DestinationLayout::contiguous
                           ? emitContiguousS32(
                                 request,
                                 request.destination.ptrUpdate
                                     == GemvN1PtrUpdate::advanceFullRegs)
                           : emitScalarS32(request);
            break;
        case DataType::s8:
        case DataType::u8:
            if (src == DataType::f32 || src == DataType::s32)
                return emitInt8(request);
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
GemvN1StoreEmitter<KType>::emitFromF32(const GemvN1StoreRequest& request)
{
    using dlp::kernel_frame::DataType;
    if (request.destination.layout == GemvN1DestinationLayout::contiguous) {
        return request.store.dstMemType == DataType::f32
                   ? emitContiguousF32(request)
                   : emitContiguousBf16(request);
    }
    return request.store.dstMemType == DataType::f32 ? emitScalarF32(request)
                                                     : emitScalarBf16(request);
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvN1StoreEmitter<KType>::emitContiguousF32(const GemvN1StoreRequest& request)
{
    const auto& source   = request.source.packed;
    const int   capacity = source.valuesPerRegister(request.store.srcRegType);
    const int   registerCount = source.registerCount(request.store.srcRegType);
    for (int i = 0; i < registerCount; ++i) {
        auto destination =
            request.destination.ptrUpdate == GemvN1PtrUpdate::preserve
                ? Xbyak::util::ptr[request.destination.regYptr
                                   + i * capacity * sizeof(float)]
                : Xbyak::util::ptr[request.destination.regYptr];
        if (i == registerCount - 1) {
            destination = request.destination.storeMask.apply(destination);
        }
        jit_.vmovups(destination, Xbyak::Zmm(source.baseIdx + i));
        if (request.destination.ptrUpdate == GemvN1PtrUpdate::advanceFullRegs
            && !(i == registerCount - 1
                 && source.validValuesInLastRegister(request.store.srcRegType)
                        != capacity)) {
            jit_.lea(request.destination.regYptr,
                     Xbyak::util::ptr[request.destination.regYptr
                                      + capacity * sizeof(float)]);
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvN1StoreEmitter<KType>::emitContiguousBf16(const GemvN1StoreRequest& request)
{
    const auto& source   = request.source.packed;
    const int   capacity = source.valuesPerRegister(request.store.srcRegType);
    const int   registerCount = source.registerCount(request.store.srcRegType);
    for (int i = 0; i < registerCount; ++i) {
        auto destination =
            request.destination.ptrUpdate == GemvN1PtrUpdate::preserve
                ? Xbyak::util::ptr[request.destination.regYptr
                                   + i * capacity * sizeof(uint16_t)]
                : Xbyak::util::ptr[request.destination.regYptr];
        if (i == registerCount - 1) {
            destination = request.destination.storeMask.apply(destination);
        }
        jit_.vcvtneps2bf16(Xbyak::Ymm(source.baseIdx + i),
                           Xbyak::Zmm(source.baseIdx + i));
        jit_.vmovdqu16(destination, Xbyak::Ymm(source.baseIdx + i));
        if (request.destination.ptrUpdate == GemvN1PtrUpdate::advanceFullRegs
            && !(i == registerCount - 1
                 && source.validValuesInLastRegister(request.store.srcRegType)
                        != capacity)) {
            jit_.lea(request.destination.regYptr,
                     Xbyak::util::ptr[request.destination.regYptr
                                      + capacity * sizeof(uint16_t)]);
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvN1StoreEmitter<KType>::emitContiguousS32(const GemvN1StoreRequest& request,
                                             bool advanceCompleteRegisters)
{
    const auto& source   = request.source.packed;
    const int   capacity = source.valuesPerRegister(request.store.srcRegType);
    const int   registerCount = source.registerCount(request.store.srcRegType);
    for (int i = 0; i < registerCount; ++i) {
        if (request.store.srcRegType == dlp::kernel_frame::DataType::f32) {
            jit_.vcvtps2dq(Xbyak::Zmm(source.baseIdx + i),
                           Xbyak::Zmm(source.baseIdx + i));
        }
        auto destination =
            request.destination.ptrUpdate == GemvN1PtrUpdate::preserve
                ? Xbyak::util::ptr[request.destination.regYptr
                                   + i * capacity * sizeof(int32_t)]
                : Xbyak::util::ptr[request.destination.regYptr];
        if (i == registerCount - 1) {
            destination = request.destination.storeMask.apply(destination);
        }
        jit_.vmovdqu32(destination, Xbyak::Zmm(source.baseIdx + i));
        if (advanceCompleteRegisters
            && !(i == registerCount - 1
                 && source.validValuesInLastRegister(request.store.srcRegType)
                        != capacity)) {
            jit_.lea(request.destination.regYptr,
                     Xbyak::util::ptr[request.destination.regYptr
                                      + capacity * sizeof(int32_t)]);
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvN1StoreEmitter<KType>::emitScalarF32(const GemvN1StoreRequest& request)
{
    if (request.source.layout == GemvN1SourceLayout::extractedXmmChunks) {
        const auto& source = request.source.extracted;
        for (int lane = 0; lane < source.validValues; ++lane) {
            const int sourceRegister = source.registers[lane / 4];
            if (lane % 4 == 0) {
                jit_.vmovss(Xbyak::util::ptr[request.destination.regYptr],
                            Xbyak::Xmm(sourceRegister));
            } else {
                jit_.vpextrd(Xbyak::util::ptr[request.destination.regYptr],
                             Xbyak::Xmm(sourceRegister), lane % 4);
            }
            jit_.add(request.destination.regYptr, request.destination.regRsC);
        }
        return dlp::jit::jitGeneratorError::success;
    }

    const auto& source        = request.source.packed;
    const int   registerCount = source.registerCount(request.store.srcRegType);
    for (int reg = 0; reg < registerCount; ++reg) {
        const int elements =
            source.valuesInRegister(reg, request.store.srcRegType);
        for (int lane = 0; lane < elements; lane += 4) {
            jit_.vextractf32x4(Xbyak::Xmm(request.temps.zmm(lane / 4)),
                               Xbyak::Zmm(source.baseIdx + reg), lane / 4);
        }
        for (int lane = 0; lane < elements; ++lane) {
            const int scratch = request.temps.zmm(lane / 4);
            if (lane % 4 == 0) {
                jit_.vmovss(Xbyak::util::ptr[request.destination.regYptr],
                            Xbyak::Xmm(scratch));
            } else {
                jit_.vpextrd(Xbyak::util::ptr[request.destination.regYptr],
                             Xbyak::Xmm(scratch), lane % 4);
            }
            jit_.add(request.destination.regYptr, request.destination.regRsC);
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvN1StoreEmitter<KType>::emitScalarBf16(const GemvN1StoreRequest& request)
{
    if (request.source.layout == GemvN1SourceLayout::extractedXmmChunks) {
        const auto& source = request.source.extracted;
        for (std::size_t chunk = 0; chunk < source.registerCount; ++chunk) {
            const int sourceRegister = source.registers[chunk];
            jit_.vcvtneps2bf16(Xbyak::Ymm(sourceRegister),
                               Xbyak::Zmm(sourceRegister));
        }
        for (int lane = 0; lane < source.validValues; ++lane) {
            const int sourceRegister = source.registers[lane / 4];
            jit_.vpextrw(Xbyak::util::ptr[request.destination.regYptr],
                         Xbyak::Xmm(sourceRegister), lane % 4);
            jit_.add(request.destination.regYptr, request.destination.regRsC);
        }
        return dlp::jit::jitGeneratorError::success;
    }

    const auto& source        = request.source.packed;
    const int   registerCount = source.registerCount(request.store.srcRegType);
    for (int reg = 0; reg < registerCount; ++reg) {
        const int elements =
            source.valuesInRegister(reg, request.store.srcRegType);
        if (source.width == SourceRegWidth::ymm) {
            const int scratch = request.temps.zmm(0);
            jit_.vcvtneps2bf16(Xbyak::Xmm(scratch),
                               Xbyak::Ymm(source.baseIdx + reg));
            for (int lane = 0; lane < elements; ++lane) {
                jit_.vpextrw(Xbyak::util::ptr[request.destination.regYptr],
                             Xbyak::Xmm(scratch), lane);
                jit_.add(request.destination.regYptr,
                         request.destination.regRsC);
            }
        } else {
            for (int lane = 0; lane < elements; lane += 4) {
                const int scratch = request.temps.zmm(lane / 4);
                jit_.vextractf32x4(Xbyak::Xmm(scratch),
                                   Xbyak::Zmm(source.baseIdx + reg), lane / 4);
            }
            for (int lane = 0; lane < elements; lane += 4) {
                const int scratch = request.temps.zmm(lane / 4);
                jit_.vcvtneps2bf16(Xbyak::Ymm(scratch), Xbyak::Zmm(scratch));
            }
            for (int lane = 0; lane < elements; ++lane) {
                const int scratch = request.temps.zmm(lane / 4);
                jit_.vpextrw(Xbyak::util::ptr[request.destination.regYptr],
                             Xbyak::Xmm(scratch), lane % 4);
                jit_.add(request.destination.regYptr,
                         request.destination.regRsC);
            }
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvN1StoreEmitter<KType>::emitScalarS32(const GemvN1StoreRequest& request)
{
    if (request.source.layout == GemvN1SourceLayout::extractedXmmChunks) {
        const auto& source = request.source.extracted;
        for (int lane = 0; lane < source.validValues; ++lane) {
            const int sourceRegister = source.registers[lane / 4];
            if (lane % 4 == 0) {
                jit_.vmovd(Xbyak::util::ptr[request.destination.regYptr],
                           Xbyak::Xmm(sourceRegister));
            } else {
                jit_.vpextrd(Xbyak::util::ptr[request.destination.regYptr],
                             Xbyak::Xmm(sourceRegister), lane % 4);
            }
            jit_.add(request.destination.regYptr, request.destination.regRsC);
        }
        return dlp::jit::jitGeneratorError::success;
    }

    const auto& source        = request.source.packed;
    const int   registerCount = source.registerCount(request.store.srcRegType);
    for (int reg = 0; reg < registerCount; ++reg) {
        const int elements =
            source.valuesInRegister(reg, request.store.srcRegType);
        if (request.store.srcRegType == dlp::kernel_frame::DataType::f32) {
            jit_.vcvtps2dq(Xbyak::Zmm(source.baseIdx + reg),
                           Xbyak::Zmm(source.baseIdx + reg));
        }
        for (int lane = 0; lane < elements; lane += 4) {
            jit_.vextracti32x4(Xbyak::Xmm(request.temps.zmm(lane / 4)),
                               Xbyak::Zmm(source.baseIdx + reg), lane / 4);
        }
        for (int lane = 0; lane < elements; ++lane) {
            const int scratch = request.temps.zmm(lane / 4);
            if (lane % 4 == 0) {
                jit_.vmovd(Xbyak::util::ptr[request.destination.regYptr],
                           Xbyak::Xmm(scratch));
            } else {
                jit_.vpextrd(Xbyak::util::ptr[request.destination.regYptr],
                             Xbyak::Xmm(scratch), lane % 4);
            }
            jit_.add(request.destination.regYptr, request.destination.regRsC);
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvN1StoreEmitter<KType>::emitInt8(const GemvN1StoreRequest& request)
{
    using dlp::kernel_frame::DataType;
    const auto& source  = request.source.packed;
    const bool  fromF32 = request.store.srcRegType == DataType::f32;
    const bool  isU8    = request.store.dstMemType == DataType::u8;
    const bool  scalar =
        request.destination.layout == GemvN1DestinationLayout::rowStrided;
    const int packed = isU8 || scalar ? request.temps.zmm(0) : 0;
    if (isU8 && !fromF32) {
        const int          zero      = request.temps.zmm(1);
        const int          high      = request.temps.zmm(2);
        const Xbyak::Reg64 immediate = request.temps.gpr();
        jit_.vpxord(Xbyak::Zmm(zero), Xbyak::Zmm(zero), Xbyak::Zmm(zero));
        jit_.mov(immediate, 255);
        jit_.vpbroadcastd(Xbyak::Zmm(high), immediate.cvt32());
    }
    const int registerCount = source.registerCount(request.store.srcRegType);
    const int capacity = source.valuesPerRegister(request.store.srcRegType);
    for (int reg = 0; reg < registerCount; ++reg) {
        const int src = source.baseIdx + reg;
        const int elements =
            source.valuesInRegister(reg, request.store.srcRegType);
        if (fromF32) {
            emitF32ClampToInt8Range(
                jit_, Xbyak::Zmm(src), Xbyak::Zmm(request.temps.zmm(1)),
                request.temps.gpr(), request.store.dstMemType);
            jit_.vcvtps2dq(Xbyak::Zmm(src), Xbyak::Zmm(src));
        }
        if (isU8 && !fromF32) {
            jit_.vpmaxsd(Xbyak::Zmm(packed), Xbyak::Zmm(src),
                         Xbyak::Zmm(request.temps.zmm(1)));
            jit_.vpminsd(Xbyak::Zmm(packed), Xbyak::Zmm(packed),
                         Xbyak::Zmm(request.temps.zmm(2)));
        }
        if (request.destination.layout == GemvN1DestinationLayout::contiguous) {
            auto dst =
                Xbyak::util::ptr[request.destination.regYptr + reg * capacity];
            if (reg == registerCount - 1)
                dst = request.destination.storeMask.apply(dst);
            if (isU8)
                jit_.vpmovdb(dst, Xbyak::Zmm(fromF32 ? src : packed));
            else
                jit_.vpmovsdb(dst, Xbyak::Zmm(src));
        } else {
            if (!isU8)
                jit_.vpmovsdb(Xbyak::Xmm(packed), Xbyak::Zmm(src));
            else
                jit_.vpmovdb(Xbyak::Xmm(packed),
                             Xbyak::Zmm(fromF32 ? src : packed));
            for (int lane = 0; lane < elements; ++lane) {
                jit_.vpextrb(Xbyak::util::ptr[request.destination.regYptr],
                             Xbyak::Xmm(packed), lane);
                jit_.add(request.destination.regYptr,
                         request.destination.regRsC);
            }
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvN1StoreEmitter<KType>::emitF32Output(const GemvN1StoreRequest& request)
{
    using dlp::kernel_frame::DataType;
    const auto& source        = request.source.packed;
    const int   registerCount = source.registerCount(request.store.srcRegType);
    const int   capacity = source.valuesPerRegister(request.store.srcRegType);
    for (int reg = 0; reg < registerCount; ++reg) {
        const int src = source.baseIdx + reg;
        const int elements =
            source.valuesInRegister(reg, request.store.srcRegType);
        if (request.store.srcRegType == DataType::s32)
            jit_.vcvtdq2ps(Xbyak::Zmm(src), Xbyak::Zmm(src));
        if (request.destination.layout == GemvN1DestinationLayout::contiguous) {
            auto dst = Xbyak::util::ptr[request.destination.regYptr
                                        + reg * capacity * sizeof(float)];
            if (reg == registerCount - 1)
                dst = request.destination.storeMask.apply(dst);
            jit_.vmovups(dst, Xbyak::Zmm(src));
        } else {
            for (int lane = 0; lane < elements; lane += 4)
                jit_.vextractf32x4(Xbyak::Xmm(request.temps.zmm(lane / 4)),
                                   Xbyak::Zmm(src), lane / 4);
            for (int lane = 0; lane < elements; ++lane) {
                const int chunk = request.temps.zmm(lane / 4);
                if (lane % 4 == 0)
                    jit_.vmovss(Xbyak::util::ptr[request.destination.regYptr],
                                Xbyak::Xmm(chunk));
                else
                    jit_.vpextrd(Xbyak::util::ptr[request.destination.regYptr],
                                 Xbyak::Xmm(chunk), lane % 4);
                jit_.add(request.destination.regYptr,
                         request.destination.regRsC);
            }
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvN1StoreEmitter<KType>::emitF16(const GemvN1StoreRequest& request)
{
    using dlp::kernel_frame::DataType;
    const auto& source = request.source.packed;
    const bool  advanceCompleteRegisters =
        request.destination.ptrUpdate == GemvN1PtrUpdate::advanceFullRegs;
    const int half          = request.temps.zmm(0);
    const int registerCount = source.registerCount(request.store.srcRegType);
    const int capacity = source.valuesPerRegister(request.store.srcRegType);
    for (int reg = 0; reg < registerCount; ++reg) {
        const int src = source.baseIdx + reg;
        const int elements =
            source.valuesInRegister(reg, request.store.srcRegType);
        if (request.store.srcRegType == DataType::s32)
            jit_.vcvtdq2ps(Xbyak::Zmm(src), Xbyak::Zmm(src));
        jit_.vcvtps2ph(Xbyak::Ymm(half), Xbyak::Zmm(src), 0);
        if (request.destination.layout == GemvN1DestinationLayout::contiguous) {
            auto dst =
                advanceCompleteRegisters
                    ? Xbyak::util::ptr[request.destination.regYptr]
                    : Xbyak::util::ptr[request.destination.regYptr
                                       + reg * capacity * sizeof(uint16_t)];
            if (reg == registerCount - 1)
                dst = request.destination.storeMask.apply(dst);
            jit_.vmovdqu16(dst, Xbyak::Ymm(half));
            if (advanceCompleteRegisters
                && !(reg == registerCount - 1 && elements != capacity))
                jit_.add(request.destination.regYptr,
                         request.destination.regAdvanceBytes);
        } else {
            for (int lane = 0; lane < elements; lane += 8)
                jit_.vextracti32x4(Xbyak::Xmm(request.temps.zmm(1 + lane / 8)),
                                   Xbyak::Ymm(half), lane / 8);
            for (int lane = 0; lane < elements; ++lane) {
                jit_.vpextrw(Xbyak::util::ptr[request.destination.regYptr],
                             Xbyak::Xmm(request.temps.zmm(1 + lane / 8)),
                             lane % 8);
                jit_.add(request.destination.regYptr,
                         request.destination.regRsC);
            }
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template<utils::kernelInstrType KType>
dlp::jit::jitGeneratorError
GemvN1StoreEmitter<KType>::emitBf16Software(const GemvN1StoreRequest& request)
{
    using dlp::kernel_frame::DataType;
    const auto& source = request.source.packed;
    const bool  scalar =
        request.destination.layout == GemvN1DestinationLayout::rowStrided;
    const int          lsbMask   = request.temps.zmm(0);
    const int          roundBias = request.temps.zmm(1);
    const int          lsb       = request.temps.zmm(2);
    const Xbyak::Reg64 gpr       = request.temps.gpr();
    jit_.mov(gpr, 1);
    jit_.vpbroadcastd(Xbyak::Zmm(lsbMask), gpr.cvt32());
    jit_.mov(gpr, 0x7fff);
    jit_.vpbroadcastd(Xbyak::Zmm(roundBias), gpr.cvt32());
    const int registerCount = source.registerCount(request.store.srcRegType);
    const int capacity = source.valuesPerRegister(request.store.srcRegType);
    for (int reg = 0; reg < registerCount; ++reg) {
        const int src = source.baseIdx + reg;
        const int elements =
            source.valuesInRegister(reg, request.store.srcRegType);
        if (request.store.srcRegType == DataType::s32)
            jit_.vcvtdq2ps(Xbyak::Zmm(src), Xbyak::Zmm(src));
        jit_.vpsrld(Xbyak::Zmm(lsb), Xbyak::Zmm(src), 16);
        jit_.vpandd(Xbyak::Zmm(lsb), Xbyak::Zmm(lsb), Xbyak::Zmm(lsbMask));
        jit_.vpaddd(Xbyak::Zmm(src), Xbyak::Zmm(src), Xbyak::Zmm(roundBias));
        jit_.vpaddd(Xbyak::Zmm(src), Xbyak::Zmm(src), Xbyak::Zmm(lsb));
        jit_.vpsrld(Xbyak::Zmm(src), Xbyak::Zmm(src), 16);
        jit_.vpmovdw(Xbyak::Ymm(src), Xbyak::Zmm(src));
        if (request.destination.layout == GemvN1DestinationLayout::contiguous) {
            auto dst = Xbyak::util::ptr[request.destination.regYptr
                                        + reg * capacity * sizeof(uint16_t)];
            if (reg == registerCount - 1)
                dst = request.destination.storeMask.apply(dst);
            jit_.vmovdqu16(dst, Xbyak::Ymm(src));
        } else {
            for (int lane = 0; lane < elements; lane += 8)
                jit_.vextracti32x4(Xbyak::Xmm(request.temps.zmm(3 + lane / 8)),
                                   Xbyak::Ymm(src), lane / 8);
            for (int lane = 0; lane < elements; ++lane) {
                jit_.vpextrw(Xbyak::util::ptr[request.destination.regYptr],
                             Xbyak::Xmm(request.temps.zmm(3 + lane / 8)),
                             lane % 8);
                jit_.add(request.destination.regYptr,
                         request.destination.regRsC);
            }
        }
    }
    return dlp::jit::jitGeneratorError::success;
}

template class GemvN1StoreEmitter<utils::kernelInstrType::avx512_zmm_32_reg>;

} // namespace amdzen::store
