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

#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <optional>
#include <stack>
#include <type_traits>
#include <vector>

#if DLP_OS_WINDOWS
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include "jit/jit_generator_base.hh"
#include "xbyak/xbyak.h"

// Error handling macro to reduce repetitive code
#define RETURN_IF_ERROR(expr)                                                  \
    do {                                                                       \
        auto err = (expr);                                                     \
        if (err != dlp::jit::jitGeneratorError::success) {                     \
            return err;                                                        \
        }                                                                      \
    } while (0);

namespace amdzen::utils {

typedef void (*jit_kernel)(dlp::kernels::gemmParams*);
using jit_gemv_n1_kernel = void (*)(dlp::kernels::gemvN1Params*);
using jit_gemv_m1_kernel = void (*)(dlp::kernels::gemvM1Params*);
using jit_pack_b_kernel  = void (*)(dlp::kernels::packBParams*);

// This size is used to allocate the memory for the JIT code generator, where
// xbyak in autogrow mode will automatically grow the buffer size if needed.
constexpr uint64_t JIT_KERNEL_SIZE = 8 * 4096;

enum class kernelInstrType : uint16_t
{
    none = 0,
    avx2_xmm_16_reg,
    avx2_ymm_16_reg,
    avx512_xmm_32_reg,
    avx512_ymm_32_reg,
    avx512_zmm_32_reg
};

// x86 AVX-512 Opmask register allocation constants.
// k0 is reserved by hardware for implicit unmasked operations.
static constexpr int MASK_START_IDX   = 1; // First usable Opmask index (k1)
static constexpr int NUM_USABLE_MASKS = 7; // k1 through k7

struct generatorParams
{
    int  MR; // This MR can be of either main kernel or fringe kernel
    int  NR; // This NR can be of either main kernel or fringe kernel
    int  K_UNROLL;
    int  PREFETCH_C_DIST;
    md_t c_downscale;
    int  numMaskRegs;
    // This will be used to generate NR + " < nElemsPerReg" kernels,
    // where NR is a multiple of nElemsPerReg including "0".
    bool useMask;
    bool mLoop; // This will be set to true only for the main kernel
    bool is_k1; // this will be set to true only for the K=1 kernel where we
                // need not generate k-unroll loop
    dlp::kernel_frame::scalingType                    alphaScalingType;
    dlp::kernel_frame::scalingType                    betaScalingType;
    kernelInstrType                                   kType;
    std::vector<dlp::kernel_frame::kernelOpsMetaData> kernelOps;

    generatorParams(md_t                           _MR,
                    md_t                           _NR,
                    int                            _K_UNROLL,
                    int                            _PREFETCH_C_DIST,
                    md_t                           _c_downscale,
                    int                            _numMaskRegs,
                    bool                           _useMask,
                    bool                           _mLoop,
                    bool                           _is_k1,
                    dlp::kernel_frame::scalingType _alphaScalingType =
                        dlp::kernel_frame::scalingType::generic,
                    dlp::kernel_frame::scalingType _betaScalingType =
                        dlp::kernel_frame::scalingType::generic,
                    kernelInstrType _kType = kernelInstrType::none)
        : MR(_MR)
        , NR(_NR)
        , K_UNROLL(_K_UNROLL)
        , PREFETCH_C_DIST(_PREFETCH_C_DIST)
        , c_downscale(_c_downscale)
        , numMaskRegs(_numMaskRegs)
        , useMask(_useMask)
        , mLoop(_mLoop)
        , is_k1(_is_k1)
        , alphaScalingType(_alphaScalingType)
        , betaScalingType(_betaScalingType)
        , kType(_kType)
    {
    }

    generatorParams(const generatorParams& other)
        : MR(other.MR)
        , NR(other.NR)
        , K_UNROLL(other.K_UNROLL)
        , PREFETCH_C_DIST(other.PREFETCH_C_DIST)
        , c_downscale(other.c_downscale)
        , numMaskRegs(other.numMaskRegs)
        , useMask(other.useMask)
        , mLoop(other.mLoop)
        , is_k1(other.is_k1)
        , alphaScalingType(other.alphaScalingType)
        , betaScalingType(other.betaScalingType)
        , kType(other.kType)
        , kernelOps(other.kernelOps)
    {
    }

    generatorParams& operator=(const generatorParams& other)
    {
        if (this != std::addressof(other)) {
            MR               = other.MR;
            NR               = other.NR;
            K_UNROLL         = other.K_UNROLL;
            PREFETCH_C_DIST  = other.PREFETCH_C_DIST;
            c_downscale      = other.c_downscale;
            numMaskRegs      = other.numMaskRegs;
            useMask          = other.useMask;
            mLoop            = other.mLoop;
            is_k1            = other.is_k1;
            alphaScalingType = other.alphaScalingType;
            betaScalingType  = other.betaScalingType;
            kType            = other.kType;
            kernelOps        = other.kernelOps;
        }
        return *this;
    }

    generatorParams(generatorParams&& other)
        : MR(other.MR)
        , NR(other.NR)
        , K_UNROLL(other.K_UNROLL)
        , PREFETCH_C_DIST(other.PREFETCH_C_DIST)
        , c_downscale(other.c_downscale)
        , numMaskRegs(other.numMaskRegs)
        , useMask(other.useMask)
        , mLoop(other.mLoop)
        , is_k1(other.is_k1)
        , alphaScalingType(other.alphaScalingType)
        , betaScalingType(other.betaScalingType)
        , kType(other.kType)
        , kernelOps(std::move(other.kernelOps))
    {
    }

    generatorParams& operator=(generatorParams&& other)
    {
        if (this != std::addressof(other)) {
            MR               = other.MR;
            NR               = other.NR;
            K_UNROLL         = other.K_UNROLL;
            PREFETCH_C_DIST  = other.PREFETCH_C_DIST;
            c_downscale      = other.c_downscale;
            numMaskRegs      = other.numMaskRegs;
            useMask          = other.useMask;
            mLoop            = other.mLoop;
            is_k1            = other.is_k1;
            alphaScalingType = other.alphaScalingType;
            betaScalingType  = other.betaScalingType;
            kType            = other.kType;
            kernelOps        = std::move(other.kernelOps);
        }
        return *this;
    }

    ~generatorParams() = default;
};

// GEMV specific generator parameters
// This is used by the JIT generator to generate the GEMV kernel
struct gemvN1GeneratorParams
{
    // Dimensions and loop control
    int  MR;          // Vector length (number of rows to process at once)
    int  M_LEFT;      // M-dimension left over elements
    int  c_downscale; // Downscale factor for C
    bool mloop;       // Whether to loop in m direction in steps of MR
    bool kloop;   // Whether to loop in k direction in steps of numElemsPerReg
    bool mfringe; // Whether to generate code for m-dimension fringe
    bool kfringe; // Whether to generate code for k-dimension fringe

    dlp::kernel_frame::storageFormat yFormat; // Storage format of the C
                                              // matrix (row-major or
                                              // column-major)

    dlp::kernel_frame::scalingType alphaScalingType; // Scaling type for alpha
    dlp::kernel_frame::scalingType betaScalingType;  // Scaling type for beta

    kernelInstrType kType; // Instruction type for the kernel
    std::vector<dlp::kernel_frame::kernelOpsMetaData>
        kernelOps; // List of post-ops

    // True when the DE has detected an L1-cache aliasing case for this
    // shape. The generator emits a two-pass MR/2 k-loop body instead
    // of the single-pass MR body. No runtime stride check is emitted.
    bool aliasMrSplit = false;

    // Constructor
    gemvN1GeneratorParams(int                              _MR,
                          int                              _M_LEFT,
                          int                              _c_downscale,
                          bool                             _mloop,
                          bool                             _kloop,
                          bool                             _mfringe,
                          bool                             _kfringe,
                          dlp::kernel_frame::storageFormat _yFormat,
                          dlp::kernel_frame::scalingType   _alphaScalingType,
                          dlp::kernel_frame::scalingType   _betaScalingType,
                          kernelInstrType                  _kType)
        : MR(_MR)
        , M_LEFT(_M_LEFT)
        , c_downscale(_c_downscale)
        , mloop(_mloop)
        , kloop(_kloop)
        , mfringe(_mfringe)
        , kfringe(_kfringe)
        , yFormat(_yFormat)
        , alphaScalingType(_alphaScalingType)
        , betaScalingType(_betaScalingType)
        , kType(_kType)
    {
    }

    // Copy constructor
    gemvN1GeneratorParams(const gemvN1GeneratorParams& other)
        : MR(other.MR)
        , M_LEFT(other.M_LEFT)
        , c_downscale(other.c_downscale)
        , mloop(other.mloop)
        , kloop(other.kloop)
        , mfringe(other.mfringe)
        , kfringe(other.kfringe)
        , yFormat(other.yFormat)
        , alphaScalingType(other.alphaScalingType)
        , betaScalingType(other.betaScalingType)
        , kType(other.kType)
        , kernelOps(other.kernelOps)
        , aliasMrSplit(other.aliasMrSplit)
    {
    }
    // Copy assignment operator
    gemvN1GeneratorParams& operator=(const gemvN1GeneratorParams& other)
    {
        if (this != std::addressof(other)) {
            MR               = other.MR;
            M_LEFT           = other.M_LEFT;
            c_downscale      = other.c_downscale;
            mloop            = other.mloop;
            kloop            = other.kloop;
            mfringe          = other.mfringe;
            kfringe          = other.kfringe;
            yFormat          = other.yFormat;
            alphaScalingType = other.alphaScalingType;
            betaScalingType  = other.betaScalingType;
            kType            = other.kType;
            kernelOps        = other.kernelOps;
            aliasMrSplit     = other.aliasMrSplit;
        }
        return *this;
    }

    // Move constructor
    gemvN1GeneratorParams(gemvN1GeneratorParams&& other)
        : MR(other.MR)
        , M_LEFT(other.M_LEFT)
        , c_downscale(other.c_downscale)
        , mloop(other.mloop)
        , kloop(other.kloop)
        , mfringe(other.mfringe)
        , kfringe(other.kfringe)
        , yFormat(other.yFormat)
        , alphaScalingType(other.alphaScalingType)
        , betaScalingType(other.betaScalingType)
        , kType(other.kType)
        , kernelOps(std::move(other.kernelOps))
        , aliasMrSplit(other.aliasMrSplit)
    {
    }

    // Move assignment operator
    gemvN1GeneratorParams& operator=(gemvN1GeneratorParams&& other) noexcept
    {
        if (this != std::addressof(other)) {
            MR               = other.MR;
            M_LEFT           = other.M_LEFT;
            c_downscale      = other.c_downscale;
            mloop            = other.mloop;
            kloop            = other.kloop;
            mfringe          = other.mfringe;
            kfringe          = other.kfringe;
            yFormat          = other.yFormat;
            alphaScalingType = other.alphaScalingType;
            betaScalingType  = other.betaScalingType;
            kType            = other.kType;
            kernelOps        = std::move(other.kernelOps);
            aliasMrSplit     = other.aliasMrSplit;
        }
        return *this;
    }

    // Destructor
    ~gemvN1GeneratorParams() = default;
};

// GEMV M=1 specific generator parameters
// This is used by the JIT generator to generate the GEMV M=1 kernel
// When m=1, we have a single row of A multiplied by vector x to produce scalar
// output
struct gemvM1GeneratorParams
{
    md_t c_downscale; // Downscale factor for C
    // Dimensions and loop control
    int NR; // Vector length (number of elements to process at once) - typically
            // 64 for AVX512
    int N_LEFT;      // N-dimension left over elements
    int N_LEFT_16;   // N-dimension left over elements in the main (16,32,48)
    int N_LEFT_LT16; // N-dimension left over elements in the fringe (remainder)
    int RS_B_N_LEFT_16;   // Row stride for B matrix in the main (16,32,48)
    int RS_B_N_LEFT_LT16; // Row stride for B matrix in the fringe (remainder)
    int KC;               // K-dimension blocksize
    int K_SUB_ITER; // Sub-iterations size(since KC is usually large, and thus
                    // is further iterated over in blocks of K_SUB_ITER)

    AOCL_DLP_MEMORY_TAG mtag_b; // Memory tag for the B matrix

    bool nloop;   // Whether to loop in n direction in steps of NR
    bool kloop;   // Whether to loop in k direction in steps of numElemsPerReg
    bool nfringe; // Whether to generate code for n-dimension fringe
    bool nfringe_main;
    bool nfringe_left;
    bool kfringe; // Whether to generate code for k-dimension fringe

    dlp::kernel_frame::storageFormat yFormat; // Storage format of the y vector

    dlp::kernel_frame::scalingType alphaScalingType; // Scaling type for alpha
    dlp::kernel_frame::scalingType betaScalingType;  // Scaling type for beta

    kernelInstrType kType; // Instruction type for the kernel
    std::vector<dlp::kernel_frame::kernelOpsMetaData>
        kernelOps; // List of post-ops

    // Constructor
    gemvM1GeneratorParams(int                              _c_downscale,
                          int                              _NR,
                          int                              _N_LEFT,
                          int                              _KC,
                          int                              _K_SUB_ITER,
                          AOCL_DLP_MEMORY_TAG              _mtag_b,
                          bool                             _nloop,
                          bool                             _kloop,
                          bool                             _nfringe,
                          bool                             _kfringe,
                          dlp::kernel_frame::storageFormat _yFormat,
                          dlp::kernel_frame::scalingType   _alphaScalingType,
                          dlp::kernel_frame::scalingType   _betaScalingType,
                          kernelInstrType                  _kType)
        : c_downscale(_c_downscale)
        , NR(_NR)
        , N_LEFT(_N_LEFT)
        , N_LEFT_16(0)
        , N_LEFT_LT16(0)
        , RS_B_N_LEFT_16(0)
        , RS_B_N_LEFT_LT16(0)
        , KC(_KC)
        , K_SUB_ITER(_K_SUB_ITER)
        , mtag_b(_mtag_b)
        , nloop(_nloop)
        , kloop(_kloop)
        , nfringe(_nfringe)
        , kfringe(_kfringe)
        , yFormat(_yFormat)
        , alphaScalingType(_alphaScalingType)
        , betaScalingType(_betaScalingType)
        , kType(_kType)
    {
    }

    // Copy constructor
    gemvM1GeneratorParams(const gemvM1GeneratorParams& other)
        : c_downscale(other.c_downscale)
        , NR(other.NR)
        , N_LEFT(other.N_LEFT)
        , N_LEFT_16(other.N_LEFT_16)
        , N_LEFT_LT16(other.N_LEFT_LT16)
        , RS_B_N_LEFT_16(other.RS_B_N_LEFT_16)
        , RS_B_N_LEFT_LT16(other.RS_B_N_LEFT_LT16)
        , KC(other.KC)
        , K_SUB_ITER(other.K_SUB_ITER)
        , mtag_b(other.mtag_b)
        , nloop(other.nloop)
        , kloop(other.kloop)
        , nfringe(other.nfringe)
        , nfringe_main(other.nfringe_main)
        , nfringe_left(other.nfringe_left)
        , kfringe(other.kfringe)
        , yFormat(other.yFormat)
        , alphaScalingType(other.alphaScalingType)
        , betaScalingType(other.betaScalingType)
        , kType(other.kType)
        , kernelOps(other.kernelOps)
    {
    }

    // Copy assignment operator
    gemvM1GeneratorParams& operator=(const gemvM1GeneratorParams& other)
    {
        if (this != std::addressof(other)) {
            c_downscale      = other.c_downscale;
            NR               = other.NR;
            N_LEFT           = other.N_LEFT;
            N_LEFT_16        = other.N_LEFT_16;
            N_LEFT_LT16      = other.N_LEFT_LT16;
            RS_B_N_LEFT_16   = other.RS_B_N_LEFT_16;
            RS_B_N_LEFT_LT16 = other.RS_B_N_LEFT_LT16;
            KC               = other.KC;
            K_SUB_ITER       = other.K_SUB_ITER;
            mtag_b           = other.mtag_b;
            nloop            = other.nloop;
            kloop            = other.kloop;
            nfringe          = other.nfringe;
            nfringe_main     = other.nfringe_main;
            nfringe_left     = other.nfringe_left;
            kfringe          = other.kfringe;
            yFormat          = other.yFormat;
            alphaScalingType = other.alphaScalingType;
            betaScalingType  = other.betaScalingType;
            kType            = other.kType;
            kernelOps        = other.kernelOps;
        }
        return *this;
    }

    // Move constructor
    gemvM1GeneratorParams(gemvM1GeneratorParams&& other)
        : c_downscale(other.c_downscale)
        , NR(other.NR)
        , N_LEFT(other.N_LEFT)
        , N_LEFT_16(other.N_LEFT_16)
        , N_LEFT_LT16(other.N_LEFT_LT16)
        , RS_B_N_LEFT_16(other.RS_B_N_LEFT_16)
        , RS_B_N_LEFT_LT16(other.RS_B_N_LEFT_LT16)
        , KC(other.KC)
        , K_SUB_ITER(other.K_SUB_ITER)
        , mtag_b(other.mtag_b)
        , nloop(other.nloop)
        , kloop(other.kloop)
        , nfringe(other.nfringe)
        , nfringe_main(other.nfringe_main)
        , nfringe_left(other.nfringe_left)
        , kfringe(other.kfringe)
        , yFormat(other.yFormat)
        , alphaScalingType(other.alphaScalingType)
        , betaScalingType(other.betaScalingType)
        , kType(other.kType)
        , kernelOps(std::move(other.kernelOps))
    {
    }

    // Move assignment operator
    gemvM1GeneratorParams& operator=(gemvM1GeneratorParams&& other)
    {
        if (this != std::addressof(other)) {
            c_downscale      = other.c_downscale;
            NR               = other.NR;
            N_LEFT           = other.N_LEFT;
            N_LEFT_16        = other.N_LEFT_16;
            N_LEFT_LT16      = other.N_LEFT_LT16;
            RS_B_N_LEFT_16   = other.RS_B_N_LEFT_16;
            RS_B_N_LEFT_LT16 = other.RS_B_N_LEFT_LT16;
            KC               = other.KC;
            K_SUB_ITER       = other.K_SUB_ITER;
            mtag_b           = other.mtag_b;
            nloop            = other.nloop;
            kloop            = other.kloop;
            nfringe          = other.nfringe;
            nfringe_main     = other.nfringe_main;
            nfringe_left     = other.nfringe_left;
            kfringe          = other.kfringe;
            yFormat          = other.yFormat;
            alphaScalingType = other.alphaScalingType;
            betaScalingType  = other.betaScalingType;
            kType            = other.kType;
            kernelOps        = std::move(other.kernelOps);
        }
        return *this;
    }

    // Destructor
    ~gemvM1GeneratorParams() = default;
};

struct packBGeneratorParams
{
    md_t            NR;
    md_t            K_FACTOR;
    kernelInstrType kType;
    bool            useMask;
    int             numMaskRegs;

    packBGeneratorParams(md_t            _NR,
                         md_t            _K_FACTOR,
                         kernelInstrType _kType,
                         bool            _useMask     = false,
                         int             _numMaskRegs = 0)
        : NR(_NR)
        , K_FACTOR(_K_FACTOR)
        , kType(_kType)
        , useMask(_useMask)
        , numMaskRegs(_numMaskRegs)
    {
    }

    ~packBGeneratorParams() = default;
};

inline int
int_log2(int value)
{
    // For GCC/Clang
    return 31 - __builtin_clz(value);
    // This is much faster as it compiles to a single instruction (bsr/lzcnt)
}

class jitGeneratorUtils
{
  public:
    // GEMM param validation (common across bf16, s8, u8s8)
    static dlp::jit::jitGeneratorError checkValidGemmParams(
        const generatorParams& params)
    {
        if (params.MR <= 0 || params.NR < 0 || params.K_UNROLL <= 0
            || params.numMaskRegs < 0
            || params.numMaskRegs > dlp::kernels::maxNumMasks
            || params.c_downscale <= DLP_INVALID
            || params.c_downscale >= DLP_MAX) {
            return dlp::jit::jitGeneratorError::badKernelInfo;
        }
        return dlp::jit::jitGeneratorError::success;
    }

    // GEMV-N1 param validation (identical across ALL datatypes)
    static dlp::jit::jitGeneratorError checkValidGemvN1Params(
        const gemvN1GeneratorParams& params)
    {
        if (params.MR <= 0 || params.M_LEFT < 0 || params.M_LEFT > params.MR
            || params.c_downscale <= DLP_INVALID
            || params.c_downscale >= DLP_MAX) {
            return dlp::jit::jitGeneratorError::badKernelInfo;
        }
        return dlp::jit::jitGeneratorError::success;
    }

    // GEMV-M1 param validation (common across f32, s8, u8s8)
    static dlp::jit::jitGeneratorError checkValidGemvM1Params(
        const gemvM1GeneratorParams& params)
    {
        if (params.NR <= 0 || params.N_LEFT < 0 || params.N_LEFT > params.NR
            || params.KC <= 0 || params.K_SUB_ITER <= 0
            || params.c_downscale <= DLP_INVALID
            || params.c_downscale >= DLP_MAX) {
            return dlp::jit::jitGeneratorError::badKernelInfo;
        }
        return dlp::jit::jitGeneratorError::success;
    }
};

template<kernelInstrType KType>
constexpr int
maskSaveWidth()
{
    if constexpr (KType == kernelInstrType::avx512_zmm_32_reg)
        return 2; // kmovw
    else if constexpr (KType == kernelInstrType::avx512_ymm_32_reg)
        return 1; // kmovb
    else if constexpr (KType == kernelInstrType::avx512_xmm_32_reg)
        return 1; // kmovb
    else
        return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// maskOffset encoding helpers
//
// The maskOffset parameter in generateKernelOps carries either a memory
// offset (>= 0) or an encoded immediate (<= -2). This lets callers like the
// RD generator pass a JIT-gen-time constant mask without touching memory.
//
//   maskOffset >= 0  : byte offset from stackPtr into gemmParams (GEMM/GEMV)
//   maskOffset == -1 : no mask
//   maskOffset <= -2 : encoded immediate, decode with decodeMaskImmediate()
// ═══════════════════════════════════════════════════════════════════════════
static constexpr int
encodeMaskImmediate(int value)
{
    return -(value + 2);
}
static constexpr int
decodeMaskImmediate(int encoded)
{
    return -(encoded + 2);
}
static constexpr bool
isMaskImmediate(int maskOffset)
{
    return maskOffset <= -2;
}

// ═══════════════════════════════════════════════════════════════════════════
// Unified Register Management
// ═══════════════════════════════════════════════════════════════════════════

template<typename REG_TYPE, int MAX_REGS>
class registerPool;
template<typename REG_TYPE>
struct registerHighPressureHandling;

// ─────────────────────────────────────────────────────────────────────────
// registerStackOperations<REG_TYPE>: type-specific save/restore instructions
// ─────────────────────────────────────────────────────────────────────────

template<typename REG_TYPE>
struct registerStackOperations
{
    static void saveToStack(Xbyak::CodeGenerator* jit, int idx, int regBytes)
    {
        jit->sub(jit->rsp, regBytes);
        jit->vmovups(jit->ptr[jit->rsp], REG_TYPE(idx));
    }

    static void restoreFromStack(Xbyak::CodeGenerator* jit,
                                 int                   idx,
                                 int                   regBytes)
    {
        jit->vmovups(REG_TYPE(idx), jit->ptr[jit->rsp]);
        jit->add(jit->rsp, regBytes);
    }
};

// Extend with additional kmov variants (kmovd, kmovq) if wider
// mask save/restore is needed in the future.
template<>
struct registerStackOperations<Xbyak::Opmask>
{
    static void saveToStack(Xbyak::CodeGenerator* jit, int idx, int regBytes)
    {
        jit->sub(jit->rsp, regBytes);
        if (regBytes == 1)
            jit->kmovb(jit->ptr[jit->rsp], Xbyak::Opmask(idx));
        else if (regBytes == 2)
            jit->kmovw(jit->ptr[jit->rsp], Xbyak::Opmask(idx));
        else {
            jit->add(jit->rsp, regBytes);
            assert(false && "Unsupported mask regBytes for stack save");
        }
    }

    static void restoreFromStack(Xbyak::CodeGenerator* jit,
                                 int                   idx,
                                 int                   regBytes)
    {
        if (regBytes == 1)
            jit->kmovb(Xbyak::Opmask(idx), jit->ptr[jit->rsp]);
        else if (regBytes == 2)
            jit->kmovw(Xbyak::Opmask(idx), jit->ptr[jit->rsp]);
        else {
            assert(false && "Unsupported mask regBytes for stack restore");
            return;
        }
        jit->add(jit->rsp, regBytes);
    }
};

template<>
struct registerStackOperations<Xbyak::Reg64>
{
    static void saveToStack(Xbyak::CodeGenerator* jit,
                            int                   idx,
                            int /*regBytes*/)
    {
        jit->push(Xbyak::Reg64(idx));
    }

    static void restoreFromStack(Xbyak::CodeGenerator* jit,
                                 int                   idx,
                                 int /*regBytes*/)
    {
        jit->pop(Xbyak::Reg64(idx));
    }
};

static constexpr uint8_t INVALID_REG_SOURCE = 255;

// ─────────────────────────────────────────────────────────────────────────
// registerGuard<REG_TYPE>: standalone RAII register guard (dual-mode)
//
// Pool mode (vector/mask): created by registerPool::acquireGuard(), releases
//   back to pool via type-erased function pointer on destruction.
// Direct-save mode (GP): created directly, push on ctor, pop on dtor.
// ─────────────────────────────────────────────────────────────────────────

template<typename REG_TYPE>
class registerGuard
{
    using ReleaseFn = void (*)(void* pool, int idx, uint8_t source);

    void*                 pool      = nullptr;
    ReleaseFn             releaseFn = nullptr;
    Xbyak::CodeGenerator* jit       = nullptr;
    int                   regIdx    = -1;
    uint8_t               source    = INVALID_REG_SOURCE;
    int                   regBytes  = 0;
    std::vector<int>      savedRegs;

  public:
    registerGuard() = default;

    // Pool mode: created by registerPool::acquireGuard()
    registerGuard(void*                 poolArg,
                  ReleaseFn             fn,
                  int                   idxArg,
                  uint8_t               sourceArg,
                  int                   regBytesArg,
                  Xbyak::CodeGenerator* gen)
        : pool(poolArg)
        , releaseFn(fn)
        , jit(gen)
        , regIdx(idxArg)
        , source(sourceArg)
        , regBytes(regBytesArg)
    {
    }

    // Direct-save mode: push on construction, pop on destruction
    registerGuard(Xbyak::CodeGenerator* gen, int idxArg, int regBytesArg = 8)
        : jit(gen)
        , regIdx(idxArg)
        , regBytes(regBytesArg)
    {
        registerStackOperations<REG_TYPE>::saveToStack(jit, regIdx, regBytes);
    }

    // Bulk-save mode (GP only): no initial register, use saveRegister() to add
    template<typename R = REG_TYPE,
             typename   = std::enable_if_t<std::is_same_v<R, Xbyak::Reg64>>>
    explicit registerGuard(Xbyak::CodeGenerator* gen)
        : jit(gen)
    {
    }

    ~registerGuard()
    {
        if (!savedRegs.empty()) {
            for (int i = static_cast<int>(savedRegs.size()) - 1; i >= 0; --i)
                jit->pop(Xbyak::Reg64(savedRegs[i]));
            return;
        }
        if (regIdx < 0)
            return;
        if (pool && releaseFn) {
            releaseFn(pool, regIdx, source);
        } else if (jit) {
            registerStackOperations<REG_TYPE>::restoreFromStack(jit, regIdx,
                                                                regBytes);
        }
    }

    // Move semantics
    registerGuard(registerGuard&& o) noexcept
        : pool(o.pool)
        , releaseFn(o.releaseFn)
        , jit(o.jit)
        , regIdx(o.regIdx)
        , source(o.source)
        , regBytes(o.regBytes)
        , savedRegs(std::move(o.savedRegs))
    {
        o.regIdx = -1;
        o.pool   = nullptr;
        o.jit    = nullptr;
    }

    registerGuard& operator=(registerGuard&& o) noexcept
    {
        if (this != &o) {
            if (!savedRegs.empty()) {
                for (int i = static_cast<int>(savedRegs.size()) - 1; i >= 0;
                     --i)
                    jit->pop(Xbyak::Reg64(savedRegs[i]));
            } else if (regIdx >= 0) {
                if (pool && releaseFn)
                    releaseFn(pool, regIdx, source);
                else if (jit)
                    registerStackOperations<REG_TYPE>::restoreFromStack(
                        jit, regIdx, regBytes);
            }
            pool      = o.pool;
            releaseFn = o.releaseFn;
            jit       = o.jit;
            regIdx    = o.regIdx;
            source    = o.source;
            regBytes  = o.regBytes;
            savedRegs = std::move(o.savedRegs);
            o.regIdx  = -1;
            o.pool    = nullptr;
            o.jit     = nullptr;
        }
        return *this;
    }

    registerGuard(const registerGuard&)            = delete;
    registerGuard& operator=(const registerGuard&) = delete;

    int idx() const { return regIdx; }
    // WARNING: Returns REG_TYPE(-1) if guard is invalid. Callers MUST check
    // isValid() or use RETURN_IF_ERROR(pool.acquireGuard(guard)) before
    // using the guard as a register operand. REG_TYPE(-1) will produce
    // garbage x86 register encoding in JIT output.
    operator REG_TYPE() const { return REG_TYPE(regIdx); }
    explicit operator bool() const { return regIdx >= 0; }
    bool     isValid() const { return regIdx >= 0; }

    // ── GP bulk save (SFINAE: only for Reg64) ──

    template<typename R = REG_TYPE,
             typename   = std::enable_if_t<std::is_same_v<R, Xbyak::Reg64>>>
    void saveRegister(Xbyak::Reg64 reg)
    {
        jit->push(reg);
        savedRegs.push_back(reg.getIdx());
    }

    // ── Mask-specific reset methods (SFINAE: only for Opmask) ──

    template<typename R = REG_TYPE,
             typename   = std::enable_if_t<std::is_same_v<R, Xbyak::Opmask>>>
    void resetToAllOnes()
    {
        if (regIdx < 0 || !jit)
            return;
        jit->kxnorw(Xbyak::Opmask(regIdx), Xbyak::Opmask(regIdx),
                    Xbyak::Opmask(regIdx));
    }

    template<typename R = REG_TYPE,
             typename   = std::enable_if_t<std::is_same_v<R, Xbyak::Opmask>>>
    void resetFromFringe(const Xbyak::Opmask& fringeMask, int shiftBits)
    {
        if (regIdx < 0 || !jit)
            return;
        if (shiftBits == 0) {
            jit->kmovb(Xbyak::Opmask(regIdx), fringeMask);
        } else {
            jit->kshiftrw(Xbyak::Opmask(regIdx), fringeMask, shiftBits);
        }
    }

    template<typename R = REG_TYPE,
             typename   = std::enable_if_t<std::is_same_v<R, Xbyak::Opmask>>>
    void reset(bool useFringe, const Xbyak::Opmask& fringeMask, int shiftBits)
    {
        if (useFringe)
            resetFromFringe(fringeMask, shiftBits);
        else
            resetToAllOnes();
    }
};

// ─────────────────────────────────────────────────────────────────────────
// registerPool<REG_TYPE, MAX_REGS>: unified register pool with builder pattern
//
// Manages free, preserve, and accumulator register lists.
// Builder: addPreserve(), setAccumulators(), then init() (free list
// auto-computed).
// ─────────────────────────────────────────────────────────────────────────

template<typename REG_TYPE, int MAX_REGS>
class registerPool
{
    template<typename R>
    friend struct registerHighPressureHandling;

    int freeList[MAX_REGS];
    int preserveList[MAX_REGS];
    int accumList[MAX_REGS];
    int regInUseList[MAX_REGS];

    int freeCount     = 0;
    int preserveCount = 0;
    int accumCount    = 0;
    int regInUseCount = 0;

    uint64_t reservedReg = 0;

    Xbyak::CodeGenerator* jit      = nullptr;
    int                   regBytes = 0;

    dlp::jit::jitGeneratorError regPoolError =
        dlp::jit::jitGeneratorError::success;

    // ── Internal helpers ──

    void saveToStack(int idx)
    {
        registerStackOperations<REG_TYPE>::saveToStack(jit, idx, regBytes);
    }

    void restoreFromStack(int idx)
    {
        registerStackOperations<REG_TYPE>::restoreFromStack(jit, idx, regBytes);
    }

    void removeFromInUse(int idx)
    {
        for (int i = 0; i < regInUseCount; ++i) {
            if (regInUseList[i] == idx) {
                regInUseList[i] = regInUseList[--regInUseCount];
                return;
            }
        }
    }

    static void releaseToPool(void* pool, int idx, uint8_t source)
    {
        static_cast<registerPool*>(pool)->release(idx, source);
    }

    void addToList(int* list, int& count, int idx)
    {
        if (idx >= 0 && count < MAX_REGS)
            list[count++] = idx;
    }

    dlp::jit::jitGeneratorError filterAndValidate(int*  list,
                                                  int&  count,
                                                  bool* claimed)
    {
        int n = 0;
        for (int i = 0; i < count; ++i) {
            int idx = list[i];
            if (idx < 0 || idx >= MAX_REGS)
                return dlp::jit::jitGeneratorError::badKernelInfo;
            if ((reservedReg >> idx) & 1u)
                continue;
            if (claimed[idx])
                return dlp::jit::jitGeneratorError::badKernelInfo;
            claimed[idx] = true;
            list[n++]    = idx;
        }
        count = n;
        return dlp::jit::jitGeneratorError::success;
    }

    dlp::jit::jitGeneratorError applyOpWithRegister(
        int        scratchRegNeeded,
        const int* accumIndices,
        int        numAccums,
        std::function<dlp::jit::jitGeneratorError(const int*, int)>& initFn,
        std::function<void(int, const int*, int)>&                   opFn)
    {
        if (scratchRegNeeded <= 0 || scratchRegNeeded > MAX_REGS)
            return dlp::jit::jitGeneratorError::notSupported;

        struct Reg
        {
            int     idx;
            uint8_t source;
        };
        Reg acquiredRegs[MAX_REGS];
        int scratchIndices[MAX_REGS];
        int numAcquired = 0;

        for (int i = 0; i < scratchRegNeeded; ++i) {
            auto r = acquire();
            if (r.idx < 0) {
                for (int j = numAcquired - 1; j >= 0; --j)
                    release(acquiredRegs[j].idx, acquiredRegs[j].source);
                return dlp::jit::jitGeneratorError::notSupported;
            }
            acquiredRegs[numAcquired]   = { r.idx, r.source };
            scratchIndices[numAcquired] = r.idx;
            ++numAcquired;
        }

        if (initFn) {
            auto initErr = initFn(scratchIndices, numAcquired);
            if (initErr != dlp::jit::jitGeneratorError::success) {
                for (int i = numAcquired - 1; i >= 0; --i)
                    release(acquiredRegs[i].idx, acquiredRegs[i].source);
                return initErr;
            }
        }

        for (int i = 0; i < numAccums; ++i)
            opFn(accumIndices[i], scratchIndices, numAcquired);

        for (int i = numAcquired - 1; i >= 0; --i)
            release(acquiredRegs[i].idx, acquiredRegs[i].source);

        return dlp::jit::jitGeneratorError::success;
    }

  public:
    struct AcquiredRegister
    {
        int idx;
        uint8_t
            source; // 0=free, 1=preserve, 2=accum, INVALID_REG_SOURCE=invalid

        operator REG_TYPE() const { return REG_TYPE(idx); }
        explicit operator bool() const { return idx >= 0; }
    };

    registerPool() = default;

    dlp::jit::jitGeneratorError checkError() const { return regPoolError; }

    // ── Builder methods ──

    registerPool& addPreserve(int idx)
    {
        addToList(preserveList, preserveCount, idx);
        return *this;
    }

    registerPool& addPreserve(int start, int count)
    {
        for (int i = 0; i < count; ++i)
            addToList(preserveList, preserveCount, start + i);
        return *this;
    }

    registerPool& setAccumulators(int startIdx, int count)
    {
        for (int i = 0; i < count; ++i)
            addToList(accumList, accumCount, startIdx + i);
        return *this;
    }

    // ── Validation + activation ──

    dlp::jit::jitGeneratorError init(
        Xbyak::CodeGenerator*  gen,
        int                    regBytesArg,
        std::optional<uint8_t> reservedBits = std::nullopt)
    {
        // Guard against double-init: jit is nullptr at construction (default
        // member initializer) and set to gen at the end of this method.
        // A second init() call would append duplicate entries to freeList
        // without resetting counts, corrupting pool state.
        if (jit != nullptr)
            return dlp::jit::jitGeneratorError::badKernelInfo;

        if (reservedBits.has_value())
            reservedReg = reservedBits.value();

        bool claimed[MAX_REGS] = {};

        auto err = filterAndValidate(preserveList, preserveCount, claimed);
        if (err != dlp::jit::jitGeneratorError::success)
            return err;

        err = filterAndValidate(accumList, accumCount, claimed);
        if (err != dlp::jit::jitGeneratorError::success)
            return err;

        for (int i = 0; i < MAX_REGS; ++i)
            if (!((reservedReg >> i) & 1u) && !claimed[i])
                freeList[freeCount++] = i;

        jit      = gen;
        regBytes = regBytesArg;
        return dlp::jit::jitGeneratorError::success;
    }

    // ── Acquire / Release ──

    AcquiredRegister acquire()
    {
        if (!jit)
            return { -1, INVALID_REG_SOURCE };

        AcquiredRegister r = { -1, INVALID_REG_SOURCE };
        if (freeCount > 0) {
            r = { freeList[--freeCount], 0 };
        } else if (preserveCount > 0) {
            int idx = preserveList[--preserveCount];
            saveToStack(idx);
            r = { idx, 1 };
        } else if (accumCount > 1) {
            int idx = accumList[--accumCount];
            saveToStack(idx);
            r = { idx, 2 };
        }

        if (r.idx >= 0) {
            if (regInUseCount >= MAX_REGS) {
                regPoolError = dlp::jit::jitGeneratorError::badKernelInfo;
                return { -1, INVALID_REG_SOURCE };
            }
            regInUseList[regInUseCount++] = r.idx;
        }

        return r;
    }

    void release(int idx, uint8_t source)
    {
        if (idx < 0)
            return;
        removeFromInUse(idx);
        switch (source) {
            case 0:
                freeList[freeCount++] = idx;
                break;
            case 1:
                restoreFromStack(idx);
                preserveList[preserveCount++] = idx;
                break;
            case 2:
                restoreFromStack(idx);
                accumList[accumCount++] = idx;
                break;
            default:
                regPoolError = dlp::jit::jitGeneratorError::badKernelInfo;
                break;
        }
    }

    // ── RAII guard acquisition ──

    registerGuard<REG_TYPE> acquireGuard()
    {
        auto r = acquire();
        if (r.idx < 0)
            return registerGuard<REG_TYPE>();
        return registerGuard<REG_TYPE>(this, &releaseToPool, r.idx, r.source,
                                       regBytes, jit);
    }

    dlp::jit::jitGeneratorError acquireGuard(registerGuard<REG_TYPE>& out)
    {
        auto r = acquire();
        if (r.idx < 0)
            return dlp::jit::jitGeneratorError::notSupported;
        out = registerGuard<REG_TYPE>(this, &releaseToPool, r.idx, r.source,
                                      regBytes, jit);
        return dlp::jit::jitGeneratorError::success;
    }

    // ── Bulk operation ──

    dlp::jit::jitGeneratorError applyOp(
        int scratchRegNeeded,
        std::function<dlp::jit::jitGeneratorError(const int*, int)> initFn,
        std::function<void(int, const int*, int)>                   opFn)
    {
        if ((freeCount + preserveCount) >= scratchRegNeeded) {
            return applyOpWithRegister(scratchRegNeeded, accumList, accumCount,
                                       initFn, opFn);
        } else {
            return registerHighPressureHandling<REG_TYPE>::apply(
                *this, scratchRegNeeded, initFn, opFn);
        }
    }

    registerPool(const registerPool&)            = delete;
    registerPool& operator=(const registerPool&) = delete;
    registerPool& operator=(registerPool&&)      = delete;

    registerPool(registerPool&& o) noexcept
        : freeCount(o.freeCount)
        , preserveCount(o.preserveCount)
        , accumCount(o.accumCount)
        , regInUseCount(o.regInUseCount)
        , reservedReg(o.reservedReg)
        , jit(o.jit)
        , regBytes(o.regBytes)
    {
        for (int i = 0; i < this->freeCount; ++i)
            freeList[i] = o.freeList[i];
        for (int i = 0; i < this->preserveCount; ++i)
            preserveList[i] = o.preserveList[i];
        for (int i = 0; i < this->accumCount; ++i)
            accumList[i] = o.accumList[i];
        for (int i = 0; i < this->regInUseCount; ++i)
            regInUseList[i] = o.regInUseList[i];
        o.jit         = nullptr;
        o.reservedReg = 0;
    }
};

// ─────────────────────────────────────────────────────────────────────────
// registerHighPressureHandling<REG_TYPE>: high-pressure allocation strategies
//
// Primary template: vector accumulator rotation (Phase 1 + Phase 2)
// Opmask specialization: nested borrow from regInUseList
// ─────────────────────────────────────────────────────────────────────────

template<typename REG_TYPE>
struct registerHighPressureHandling
{
    template<int MAX_REGS>
    static dlp::jit::jitGeneratorError apply(
        registerPool<REG_TYPE, MAX_REGS>& pool,
        int                               scratchRegNeeded,
        std::function<dlp::jit::jitGeneratorError(const int*, int)> initFn,
        std::function<void(int, const int*, int)>                   opFn)
    {
        int numToBorrow =
            scratchRegNeeded - (pool.freeCount + pool.preserveCount);
        if (numToBorrow <= 0)
            numToBorrow = 1;
        if (numToBorrow > pool.accumCount)
            numToBorrow = pool.accumCount;
        if (pool.freeCount + numToBorrow > MAX_REGS)
            return dlp::jit::jitGeneratorError::notSupported;

        int accumsRemaining = pool.accumCount - numToBorrow;
        if (accumsRemaining <= 0)
            return dlp::jit::jitGeneratorError::notSupported;

        int borrowedRegs[MAX_REGS];
        for (int i = 0; i < numToBorrow; ++i)
            borrowedRegs[i] = pool.accumList[i];

        // Phase 1: Save lower accums to stack, promote to free list
        for (int i = 0; i < numToBorrow; ++i) {
            pool.saveToStack(pool.accumList[i]);
            pool.freeList[pool.freeCount++] = pool.accumList[i];
        }
        for (int i = 0; i < accumsRemaining; ++i)
            pool.accumList[i] = pool.accumList[i + numToBorrow];
        pool.accumCount = accumsRemaining;

        auto err1 = pool.applyOpWithRegister(scratchRegNeeded, pool.accumList,
                                             pool.accumCount, initFn, opFn);

        // Undo phase 1 promotion
        pool.freeCount -= numToBorrow;
        for (int i = accumsRemaining - 1; i >= 0; --i)
            pool.accumList[i + numToBorrow] = pool.accumList[i];
        for (int i = 0; i < numToBorrow; ++i)
            pool.accumList[i] = borrowedRegs[i];
        pool.accumCount = accumsRemaining + numToBorrow;

        if (err1 != dlp::jit::jitGeneratorError::success)
            return err1;

        // Phase 2: Rotation for borrowed accumulators
        for (int i = numToBorrow - 1; i >= 0; --i)
            pool.restoreFromStack(borrowedRegs[i]);
        for (int i = 0; i < numToBorrow; ++i) {
            int upperIdx = pool.accumCount - numToBorrow + i;
            pool.saveToStack(pool.accumList[upperIdx]);
            pool.freeList[pool.freeCount++] = pool.accumList[upperIdx];
        }

        auto err2 = pool.applyOpWithRegister(scratchRegNeeded, borrowedRegs,
                                             numToBorrow, initFn, opFn);

        // Undo phase 2 promotion
        pool.freeCount -= numToBorrow;
        for (int i = numToBorrow - 1; i >= 0; --i)
            pool.restoreFromStack(
                pool.accumList[pool.accumCount - numToBorrow + i]);

        return err2;
    }
};

template<>
struct registerHighPressureHandling<Xbyak::Opmask>
{
    template<int MAX_REGS>
    static dlp::jit::jitGeneratorError apply(
        registerPool<Xbyak::Opmask, MAX_REGS>& pool,
        int                                    scratchRegNeeded,
        std::function<dlp::jit::jitGeneratorError(const int*, int)> initFn,
        std::function<void(int, const int*, int)>                   opFn)
    {
        int numToBorrow =
            scratchRegNeeded - (pool.freeCount + pool.preserveCount);
        if (numToBorrow <= 0)
            numToBorrow = 1;
        if (numToBorrow > pool.regInUseCount)
            return dlp::jit::jitGeneratorError::notSupported;

        int borrowedRegs[MAX_REGS];
        for (int i = 0; i < numToBorrow; ++i)
            borrowedRegs[i] = pool.regInUseList[--pool.regInUseCount];

        // Save in-use values, promote to free
        for (int i = 0; i < numToBorrow; ++i) {
            pool.saveToStack(borrowedRegs[i]);
            pool.freeList[pool.freeCount++] = borrowedRegs[i];
        }

        int  emptyAccums[1] = {};
        auto err = pool.applyOpWithRegister(scratchRegNeeded, emptyAccums, 0,
                                            initFn, opFn);

        // Undo: remove from free, restore in-use values
        pool.freeCount -= numToBorrow;
        for (int i = numToBorrow - 1; i >= 0; --i) {
            pool.restoreFromStack(borrowedRegs[i]);
            pool.regInUseList[pool.regInUseCount++] = borrowedRegs[i];
        }

        return err;
    }
};

} // namespace amdzen::utils
