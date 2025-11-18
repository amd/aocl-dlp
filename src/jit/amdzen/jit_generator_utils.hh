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

#include <cmath>
#include <stack>

#if DLP_OS_WINDOWS
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include "jit/jit_generator_base.hh"
#include "jit/xbyak/xbyak.h"

// Error handling macro to reduce repetitive code
#define RETURN_IF_ERROR(expr)                                                  \
    do {                                                                       \
        auto err = (expr);                                                     \
        if (err != dlp::jit::jitGeneratorError::success) {                     \
            return err;                                                        \
        }                                                                      \
    } while (0);

namespace amdzen::utils {

class jitHelperUtils
{
  public:
    static void* allocateJitMemory(std::size_t jitKernSize)
    {
        void* codeBuffer = nullptr;
#if DLP_OS_WINDOWS
        codeBuffer =
            VirtualAlloc(nullptr, jitKernSize, MEM_COMMIT | MEM_RESERVE,
                         PAGE_EXECUTE_READWRITE);
#else
        codeBuffer =
            mmap(nullptr, jitKernSize, PROT_READ | PROT_WRITE | PROT_EXEC,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (codeBuffer == MAP_FAILED) {
            codeBuffer = nullptr;
        }
#endif
        return codeBuffer;
    }

    static void deallocateJitMemory(void*& codeBuffer, std::size_t jitKernSize)
    {
        if (codeBuffer != nullptr) {
#if DLP_OS_WINDOWS
            VirtualFree(codeBuffer, 0, MEM_RELEASE);
#else
            munmap(codeBuffer, jitKernSize);
#endif
            codeBuffer = nullptr;
        }
    }
};

typedef void (*jit_kernel)(dlp::kernels::gemmParams*);
using jit_gemv_n1_kernel = void (*)(dlp::kernels::gemvN1Params*);
using jit_gemv_m1_kernel = void (*)(dlp::kernels::gemvM1Params*);

// JIT_KERNEL_SIZE has been increased from 8 * 4096 to 16 * 4096 to accommodate
// the larger code size required for M=1 kernels. This change ensures that
// generated kernels do not overflow the buffer, but may cause excessive memory
// usage for smaller kernels. Consider making this size dynamic in the future
// to optimize memory usage.
constexpr uint64_t JIT_KERNEL_SIZE = 16 * 4096;

enum class kernelInstrType : uint16_t
{
    none = 0,
    avx2_xmm_16_reg,
    avx2_ymm_16_reg,
    avx512_xmm_32_reg,
    avx512_ymm_32_reg,
    avx512_zmm_32_reg
};

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
            c_downscale      = other.c_downscale;
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
            c_downscale      = other.c_downscale;
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

    AOCL_MEMORY_TAG mtag_b; // Memory tag for the B matrix

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
                          AOCL_MEMORY_TAG                  _mtag_b,
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
        , kfringe(other.kfringe)
        , yFormat(other.yFormat)
        , alphaScalingType(other.alphaScalingType)
        , betaScalingType(other.betaScalingType)
        , kType(other.kType)
        , kernelOps(other.kernelOps)
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
            kfringe          = other.kfringe;
            yFormat          = other.yFormat;
            alphaScalingType = other.alphaScalingType;
            betaScalingType  = other.betaScalingType;
            kType            = other.kType;
            kernelOps        = other.kernelOps;
        }
        return *this;
    }

    // Destructor
    ~gemvM1GeneratorParams() = default;
};

inline int
int_log2(int value)
{
    // For GCC/Clang
    return 31 - __builtin_clz(value);
    // This is much faster as it compiles to a single instruction (bsr/lzcnt)
}

template<typename REG_TYPE>
class registerGuard
{
    Xbyak::CodeGenerator* jit;
    std::stack<REG_TYPE>  regStack;

  public:
    registerGuard(Xbyak::CodeGenerator* _jit)
        : jit{ _jit }
    {
    }

    void saveRegister(REG_TYPE reg)
    {
        regStack.push(reg);
        jit->push(reg);
    }

    ~registerGuard()
    {
        while (!regStack.empty()) {
            auto tReg = regStack.top();
            jit->pop(tReg);
            regStack.pop();
        }
    }

    registerGuard(registerGuard&)             = delete;
    registerGuard(registerGuard&&)            = delete;
    registerGuard& operator=(registerGuard&)  = delete;
    registerGuard& operator=(registerGuard&&) = delete;
};

} // namespace amdzen::utils
