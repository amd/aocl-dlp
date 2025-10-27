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
    // Function to dump JIT code to a file for debugging purposes. This
    // function will create a file with the name <code_name>_<m>x<n>.bin".
    // The code will be dumped in binary format.
    static void dump_jit_code(
        const void* code, int code_size, const char* code_name, int m, int n)
    {
        if (code) {
            static int counter = 0;
#define MAX_FNAME_LEN 256
            char fname[MAX_FNAME_LEN + 1];
            // TODO (Roma): support prefix for code / linux perf dumps
            snprintf(fname, MAX_FNAME_LEN, "%s_%dx%d.bin", code_name, m, n);
            counter++;
            FILE* fp = fopen(fname, "wb+");
            // Failure to dump code is not fatal
            if (fp) {
                int unused = fwrite(code, code_size, 1, fp);
                fclose(fp);
            }
        }
#undef MAX_FNAME_LEN
    }

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
    int MR; // This MR can be of either main kernel or fringe kernel
    int NR; // This NR can be of either main kernel or fringe kernel
    int K_UNROLL;
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
    int MR;      // Vector length (number of rows to process at once)
    int M_LEFT;  // M-dimension left over elements
    int MR_LEFT; // MR-dimension left over elements(used when loading/storing
                 // from C)

    bool mloop;   // Whether to loop in m direction in steps of MR
    bool kloop;   // Whether to loop in k direction in steps of numElemsPerReg
    bool mfringe; // Whether to generate code for m-dimension fringe
    bool kfringe; // Whether to generate code for k-dimension fringe

    dlp::kernel_frame::storageFormat yFormat; // Storage format of the C
                                              // matrix (row-major or
                                              // column-major)

    dlp::kernel_frame::scalingType alphaScalingType; // Scaling type for alpha
    dlp::kernel_frame::scalingType betaScalingType;  // Scaling type for beta

    kernelInstrType kType; // Instruction type for the kernel

    // Constructor
    gemvN1GeneratorParams(int                              _MR,
                          int                              _M_LEFT,
                          int                              _MR_LEFT,
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
        , MR_LEFT(_MR_LEFT)
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
        , MR_LEFT(other.MR_LEFT)
        , mloop(other.mloop)
        , kloop(other.kloop)
        , mfringe(other.mfringe)
        , kfringe(other.kfringe)
        , yFormat(other.yFormat)
        , alphaScalingType(other.alphaScalingType)
        , betaScalingType(other.betaScalingType)
        , kType(other.kType)
    {
    }
    // Copy assignment operator
    gemvN1GeneratorParams& operator=(const gemvN1GeneratorParams& other)
    {
        if (this != std::addressof(other)) {
            MR               = other.MR;
            M_LEFT           = other.M_LEFT;
            MR_LEFT          = other.MR_LEFT;
            mloop            = other.mloop;
            kloop            = other.kloop;
            mfringe          = other.mfringe;
            kfringe          = other.kfringe;
            yFormat          = other.yFormat;
            alphaScalingType = other.alphaScalingType;
            betaScalingType  = other.betaScalingType;
            kType            = other.kType;
        }
        return *this;
    }

    // Move constructor
    gemvN1GeneratorParams(gemvN1GeneratorParams&& other)
        : MR(other.MR)
        , M_LEFT(other.M_LEFT)
        , MR_LEFT(other.MR_LEFT)
        , mloop(other.mloop)
        , kloop(other.kloop)
        , mfringe(other.mfringe)
        , kfringe(other.kfringe)
        , yFormat(other.yFormat)
        , alphaScalingType(other.alphaScalingType)
        , betaScalingType(other.betaScalingType)
        , kType(other.kType)
    {
    }

    // Move assignment operator
    gemvN1GeneratorParams& operator=(gemvN1GeneratorParams&& other) noexcept
    {
        if (this != std::addressof(other)) {
            MR               = other.MR;
            M_LEFT           = other.M_LEFT;
            MR_LEFT          = other.MR_LEFT;
            mloop            = other.mloop;
            kloop            = other.kloop;
            mfringe          = other.mfringe;
            kfringe          = other.kfringe;
            yFormat          = other.yFormat;
            alphaScalingType = other.alphaScalingType;
            betaScalingType  = other.betaScalingType;
            kType            = other.kType;
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
    // Dimensions and loop control
    int NR; // Vector length (number of elements to process at once) - typically
            // 64 for AVX512
    int KC; // K-dimension blocksize
    int K_SUB_ITER; // Sub-iterations size(since KC is usually large, and thus
                    // is further iterated over in blocks of K_SUB_ITER)

    AOCL_MEMORY_TAG mtag_b; // Memory tag for the B matrix

    bool nloop;   // Whether to loop in n direction in steps of NR
    bool kloop;   // Whether to loop in k direction in steps of numElemsPerReg
    bool nfringe; // Whether to generate code for n-dimension fringe
    bool kfringe; // Whether to generate code for k-dimension fringe

    dlp::kernel_frame::storageFormat yFormat; // Storage format of the y vector

    dlp::kernel_frame::scalingType alphaScalingType; // Scaling type for alpha
    dlp::kernel_frame::scalingType betaScalingType;  // Scaling type for beta

    kernelInstrType kType; // Instruction type for the kernel

    // Constructor
    gemvM1GeneratorParams(int                              _NR,
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
        : NR(_NR)
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
        : NR(other.NR)
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
    {
    }

    // Copy assignment operator
    gemvM1GeneratorParams& operator=(const gemvM1GeneratorParams& other)
    {
        if (this != std::addressof(other)) {
            NR               = other.NR;
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
        }
        return *this;
    }

    // Move constructor
    gemvM1GeneratorParams(gemvM1GeneratorParams&& other)
        : NR(other.NR)
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
    {
    }

    // Move assignment operator
    gemvM1GeneratorParams& operator=(gemvM1GeneratorParams&& other)
    {
        if (this != std::addressof(other)) {
            NR               = other.NR;
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
        }
        return *this;
    }

    // Destructor
    ~gemvM1GeneratorParams() = default;
};

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
