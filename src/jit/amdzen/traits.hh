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
#include "jit/jit_generator_base.hh"
#include "jit_generator_utils.hh"

namespace amdzen::traits {

// Define type traits for each datatype
template<dlp::kernel_frame::kernelDatatype KDT>
struct kernel_types;

template<>
struct kernel_types<dlp::kernel_frame::kernelDatatype::f32f32f32of32>
{
    using aType        = float;
    using bType        = float;
    using cType        = float;
    using accumType    = float;
    using kernelOpType = float;
};

template<>
struct kernel_types<dlp::kernel_frame::kernelDatatype::u8s8s32os32>
{
    using aType        = uint8_t;
    using bType        = int8_t;
    using cType        = int32_t;
    using accumType    = int32_t;
    using kernelOpType = int32_t;
};

template<>
struct kernel_types<dlp::kernel_frame::kernelDatatype::s8s8s32os32>
{
    using aType        = int8_t;
    using bType        = int8_t;
    using cType        = int32_t;
    using accumType    = int32_t;
    using kernelOpType = int32_t;
};

/**
 * ArchitectureTraits provides a set of type and constant definitions describing
 * the properties of a given instruction set architecture, as identified by
 * kernelInstrType. It defines the register type (e.g., Ymm, Zmm), register size
 * in bits and bytes, the number of available registers, and whether features
 * like mask support and AVX512 are present. This traits struct is used by the
 * templated kernel generator to abstract over architecture-specific details,
 * enabling the generation of optimized kernels for different instruction sets
 * in a generic and type-safe manner.
 */
template<utils::kernelInstrType KType>
struct ArchitectureTraits;

template<>
struct ArchitectureTraits<utils::kernelInstrType::avx2_ymm_16_reg>
{
    using RegType                        = Xbyak::Ymm;
    using halfRegType                    = Xbyak::Xmm;
    static constexpr int  regSize        = 256;
    static constexpr int  regBytes       = regSize / 8;
    static constexpr int  numRegs        = 16;
    static constexpr bool hasMaskSupport = false;
    static constexpr bool isAVX512       = false;
};

template<>
struct ArchitectureTraits<utils::kernelInstrType::avx2_xmm_16_reg>
{
    using RegType                        = Xbyak::Xmm;
    using halfRegType                    = Xbyak::Xmm;
    static constexpr int  regSize        = 128;
    static constexpr int  regBytes       = regSize / 8;
    static constexpr int  numRegs        = 16;
    static constexpr bool hasMaskSupport = false;
    static constexpr bool isAVX512       = false;
};

template<>
struct ArchitectureTraits<utils::kernelInstrType::avx512_zmm_32_reg>
{
    using RegType                        = Xbyak::Zmm;
    using halfRegType                    = Xbyak::Ymm;
    static constexpr int  regSize        = 512;
    static constexpr int  regBytes       = regSize / 8;
    static constexpr int  numRegs        = 32;
    static constexpr bool hasMaskSupport = true;
    static constexpr bool isAVX512       = true;
};

template<>
struct ArchitectureTraits<utils::kernelInstrType::avx512_ymm_32_reg>
{
    using RegType                        = Xbyak::Ymm;
    using halfRegType                    = Xbyak::Xmm;
    static constexpr int  regSize        = 256;
    static constexpr int  regBytes       = regSize / 8;
    static constexpr int  numRegs        = 32;
    static constexpr bool hasMaskSupport = true;
    static constexpr bool isAVX512       = true;
};

template<>
struct ArchitectureTraits<utils::kernelInstrType::avx512_xmm_32_reg>
{
    using RegType                        = Xbyak::Xmm;
    using halfRegType                    = Xbyak::Xmm;
    static constexpr int  regSize        = 128;
    static constexpr int  regBytes       = regSize / 8;
    static constexpr int  numRegs        = 32;
    static constexpr bool hasMaskSupport = true;
    static constexpr bool isAVX512       = true;
};

} // namespace amdzen::traits
