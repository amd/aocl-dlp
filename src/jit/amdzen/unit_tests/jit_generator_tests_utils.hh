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

#include "arch_utils/arch_config_manager.hh"
#include "bindings/c_wrappers/capi_kernel_frame_wrappers.h"
#include "cpu_utils/cpu_features.hh"
#include "jit/amdzen/bf16_gemm_generator.hh"
#include "jit/amdzen/bf16_gemv_generator.hh"
#include "jit/amdzen/f32_gemm_generator.hh"
#include "jit/amdzen/f32_gemv_generator.hh"
#include "jit/amdzen/jit_generator_utils.hh"
#include "jit/amdzen/s8_gemm_generator.hh"
#include "jit/amdzen/s8_gemv_generator.hh"
#include "jit/amdzen/u8s8_gemm_generator.hh"
#include "jit/amdzen/u8s8_gemv_generator.hh"
#include "kernel_frame/kernel_frame_base.hh"
#include "xbyak/xbyak.h"
#include <chrono>
#include <csignal>
#include <cstring>
#include <functional>
#include <gtest/gtest.h>
#include <iomanip>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace dlp::kernel_frame;
using namespace amdzen::utils;
using namespace amdzen::gen;
using namespace amdzen::codegen;

namespace test_jit_utils {

// ============================================================================
// Process-Based Crash Isolation (SIGABRT, SIGSEGV, etc.)
// ============================================================================

/**
 * @brief Uses fork() to isolate potentially-crashing code in a child process
 *
 * Usage:
 *   auto result = CrashIsolation::runIsolated([&]() {
 *       return static_cast<int>(generateKernel(params));
 *   });
 *   if (result.crashed) { FAIL() << ...; }
 */
class CrashIsolation
{
  private:
    // POD struct serialized over a pipe from child to parent.
    // Carries either the function's return code (on normal completion)
    // or an exception message (if the function threw).
    // If the child crashes (signal), no message is written and the
    // parent detects the crash via waitpid().
    struct ChildMessage
    {
        enum Status : uint8_t
        {
            COMPLETED = 0,
            EXCEPTION = 1
        };
        Status status     = COMPLETED;
        int    returnCode = 0;
        char   message[256]{};
    };

    // Runs in the forked child process. Executes the test function,
    // writes the outcome to the pipe, and terminates with _exit()
    // (not exit()) to avoid running GTest atexit handlers or flushing
    // the parent's stdio buffers. If the function crashes (e.g., SIGSEGV),
    // the write never happens and the parent detects it via waitpid().
    [[noreturn]] static void runInChild(int writeFd, std::function<int()>& func)
    {
        ChildMessage msg{};
        try {
            msg.returnCode = func();
            msg.status     = ChildMessage::COMPLETED;
        } catch (const std::exception& e) {
            msg.status = ChildMessage::EXCEPTION;
            std::strncpy(msg.message, e.what(), sizeof(msg.message) - 1);
            msg.message[sizeof(msg.message) - 1] = '\0';
        } catch (...) {
            msg.status = ChildMessage::EXCEPTION;
            std::strncpy(msg.message, "Unknown exception",
                         sizeof(msg.message) - 1);
            msg.message[sizeof(msg.message) - 1] = '\0';
        }

        ssize_t n = write(writeFd, &msg, sizeof(msg));
        (void)n;
        if (n != static_cast<ssize_t>(sizeof(msg))) {
            // Treat partial or failed writes as a fatal error in the child.
            close(writeFd);
            _exit(1);
        }
        _exit(0);
    }

  public:
    // Outcome of running an isolated function. Exactly one of
    // {completed, threw, crashed} will be true.
    struct Result
    {
        bool        completed  = false; // Function returned normally
        int         returnCode = 0; // Return value (e.g., jitGeneratorError)
        bool        threw      = false; // Function threw a C++ exception
        std::string exceptionMessage;   // The exception's what() string
        bool crashed = false; // Child killed by signal (or pipe/fork failed)
        int  crashSignal = 0; // Signal number (SIGSEGV, SIGABRT, etc.)
    };

    /**
     * @brief Run a function in an isolated child process
     * @param func Function returning an int (e.g., cast of jitGeneratorError)
     * @return Result describing whether func completed, threw, or crashed
     *
     * How it works:
     *   1. Create a pipe for child-to-parent communication.
     *   2. fork() a child process (gets a COW copy of the parent's memory,
     *      including the mmap'd JIT code buffer and all captured state).
     *   3. Child: execute the function, write a ChildMessage to the pipe,
     *      then _exit(0). If the function crashes, the write never happens.
     *   4. Parent: waitpid() for the child.
     *      - WIFSIGNALED: child was killed by a signal -> report crash.
     *      - Normal exit:  read the ChildMessage from the pipe to get the
     *        return code or exception message.
     */
    static Result runIsolated(std::function<int()> func)
    {
        // Step 1: Create a unidirectional pipe (pipefd[0]=read,
        // pipefd[1]=write)
        int pipefd[2];
        if (pipe(pipefd) != 0) {
            Result r;
            r.crashed = true;
            return r;
        }

        // Step 2: Fork a child process to run the test function in isolation
        pid_t pid = fork();
        if (pid < 0) {
            close(pipefd[0]);
            close(pipefd[1]);
            Result r;
            r.crashed = true;
            return r;
        }

        // Step 3: Child process -- close read end and execute
        if (pid == 0) {
            close(pipefd[0]);
            runInChild(pipefd[1], func);
        }

        // Step 4: Parent process -- close write end, wait for child, read
        // result
        close(pipefd[1]);

        constexpr int kChildTimeoutSec = 30;
        auto          deadline         = std::chrono::steady_clock::now()
                        + std::chrono::seconds(kChildTimeoutSec);

        int   status;
        pid_t ret;
        while ((ret = waitpid(pid, &status, WNOHANG)) == 0) {
            if (std::chrono::steady_clock::now() >= deadline) {
                kill(pid, SIGKILL);
                waitpid(pid, &status, 0);
                Result r;
                r.crashed     = true;
                r.crashSignal = SIGKILL;
                close(pipefd[0]);
                return r;
            }
            usleep(1000);
        }

        Result result;

        // Child was killed by a signal (SIGSEGV, SIGABRT, etc.)
        if (WIFSIGNALED(status)) {
            result.crashed     = true;
            result.crashSignal = WTERMSIG(status);
            close(pipefd[0]);
            return result;
        }

        // Child exited normally -- read the serialized ChildMessage from the
        // pipe
        ChildMessage msg{};
        ssize_t      bytesRead = read(pipefd[0], &msg, sizeof(msg));
        close(pipefd[0]);

        if (bytesRead == static_cast<ssize_t>(sizeof(msg))) {
            if (msg.status == ChildMessage::COMPLETED) {
                result.completed  = true;
                result.returnCode = msg.returnCode;
            } else {
                result.threw            = true;
                result.exceptionMessage = msg.message;
            }
        } else {
            // Incomplete/missing message -- treat as crash
            result.crashed = true;
        }

        return result;
    }

    // Convert a signal number to a human-readable description for test output
    static const char* signalName(int sig)
    {
        switch (sig) {
            case SIGABRT:
                return "SIGABRT (Abort/Assert failure)";
            case SIGSEGV:
                return "SIGSEGV (Segmentation fault)";
            case SIGILL:
                return "SIGILL (Illegal instruction)";
            case SIGFPE:
                return "SIGFPE (Floating point exception)";
            case SIGBUS:
                return "SIGBUS (Bus error)";
            default:
                return "Unknown signal";
        }
    }
};

// ============================================================================
// Architecture Detection Helper
// ============================================================================

class ArchBasedKernelTypes
{
  public:
    /**
     * @brief Get kernel instruction types for F32 supported by the underlying
     * hardware
     * @return Vector of kernel types based on detected CPU architecture
     *         (e.g., AVX512 returns ZMM and YMM types, AVX2 returns YMM only)
     */
    static std::vector<kernelInstrType> getAllKernelTypesForF32()
    {
        std::vector<kernelInstrType> types;
        bool isAvx512 = dlp::arch_utils::archConfigManager::getInstance()
                            .isAvx512SupportedByArch();
        bool isAvx2 = dlp::arch_utils::archConfigManager::getInstance()
                          .isAvx2Fma3SupportedByArch();

        if (isAvx512) {
            types.push_back(kernelInstrType::avx512_zmm_32_reg);
            types.push_back(
                kernelInstrType::avx512_ymm_32_reg); // Test YMM on AVX512
        } else if (isAvx2) {
            types.push_back(kernelInstrType::avx2_ymm_16_reg);
        }
        return types;
    }

    /**
     * @brief Get kernel instruction types for BF16 supported by the underlying
     * hardware
     * @return Vector of kernel types based on detected CPU architecture
     *         (requires AVX512-BF16 instruction set support)
     */
    static std::vector<kernelInstrType> getAllKernelTypesForBF16()
    {
        std::vector<kernelInstrType> types;
        bool isAvx512Bf16 = dlp::arch_utils::archConfigManager::getInstance()
                                .isAvx512Bf16SupportedByArch();

        if (isAvx512Bf16) {
            types.push_back(kernelInstrType::avx512_zmm_32_reg);
        }
        return types;
    }

    /**
     * @brief Get kernel instruction types for U8S8 supported by the underlying
     * hardware
     * @return Vector of kernel types based on detected CPU architecture
     *         (requires AVX512-VNNI instruction set support)
     */
    static std::vector<kernelInstrType> getAllKernelTypesForU8S8()
    {
        std::vector<kernelInstrType> types;
        bool isAvx512Vnni = dlp::cpu_utils::cpuFeaturesInstance().hasFeature(
            dlp::cpu_utils::isaFeature::avx512vnni);

        if (isAvx512Vnni) {
            types.push_back(kernelInstrType::avx512_zmm_32_reg);
        }
        return types;
    }

    /**
     * @brief Get kernel instruction types for S8 supported by the underlying
     * hardware
     * @return Vector of kernel types based on detected CPU architecture
     *         (requires AVX512-VNNI instruction set support)
     */
    static std::vector<kernelInstrType> getAllKernelTypesForS8()
    {
        std::vector<kernelInstrType> types;
        bool isAvx512Vnni = dlp::cpu_utils::cpuFeaturesInstance().hasFeature(
            dlp::cpu_utils::isaFeature::avx512vnni);

        if (isAvx512Vnni) {
            types.push_back(kernelInstrType::avx512_zmm_32_reg);
        }
        return types;
    }
};

/**
 * @brief Convert kernelInstrType enum to human-readable architecture string
 * @param kType The kernel instruction type
 * @return String representation of the architecture variant
 */
inline std::string
kTypeToArchString(kernelInstrType kType)
{
    switch (kType) {
        case kernelInstrType::avx512_zmm_32_reg:
            return "avx512_zmm_32";
        case kernelInstrType::avx512_ymm_32_reg:
            return "avx512_ymm_32";
        case kernelInstrType::avx2_ymm_16_reg:
            return "avx2_ymm_16";
        default:
            return "unknown";
    }
}

// ============================================================================
// Mock KernelInfo Generator (bypasses Decision Engine)
// ============================================================================

class MockKernelInfoGenerator
{
  public:
    /**
     * @brief Base kernel info creator - all other creators call this
     * @param kType Kernel instruction type
     * @param mr Number of rows in micro-kernel
     * @param nr Number of columns in micro-kernel
     * @param kc K-dimension blocksize
     * @param k_unroll K unroll factor
     * @param prefetch_c_dist Prefetch distance for C matrix
     * @param genLtKrnlForAvailFullKrnl Generate less-than kernel flag
     * @param term_fringe_nr Terminal fringe NR
     * @param alphaType Alpha scaling type
     * @param betaType Beta scaling type
     * @param kInstPref Kernel instruction preference
     * @return Initialized kernelInfo structure
     */
    static kernelInfo createKernelInfo(
        kernelInstrType       kType,
        int                   mr,
        int                   nr,
        int                   kc                        = 4096,
        int                   k_unroll                  = 4,
        int                   prefetch_c_dist           = 8,
        bool                  genLtKrnlForAvailFullKrnl = false,
        int                   term_fringe_nr            = 0,
        scalingType           alphaType                 = scalingType::generic,
        scalingType           betaType                  = scalingType::generic,
        kernelInstrPreference kInstPref =
            kernelInstrPreference::avx512_zmm_favour)
    {
        kernelInfo ki;
        ki.mr                        = mr;
        ki.nr                        = nr;
        ki.kc                        = kc;
        ki.k_unroll                  = k_unroll;
        ki.prefetch_c_dist           = prefetch_c_dist;
        ki.genLtKrnlForAvailFullKrnl = genLtKrnlForAvailFullKrnl;
        ki.term_fringe_nr            = term_fringe_nr;
        ki.alphaScalingType          = alphaType;
        ki.betaScalingType           = betaType;
        ki.kInstPref                 = kInstPref;
        ki.c_downscale               = DLP_F32;
        ki.mtag_b                    = PACK;
        ki.mtag_a                    = PACK;
        ki.invokeRD                  = false;
        ki.kOpsArr                   = nullptr;
        ki.kOpsArrSize               = 0;
        ki.anyKOpsOrder              = false;
        return ki;
    }

    /**
     * @brief Create GEMM kernel info with architecture-aware defaults
     */
    static kernelInfo createGEMMKernelInfo(
        kernelInstrType       kType,
        int                   mr                        = 6,
        int                   nr                        = -1,
        int                   kc                        = 4096,
        int                   k_unroll                  = 4,
        int                   prefetch_c_dist           = 8,
        bool                  genLtKrnlForAvailFullKrnl = false,
        int                   term_fringe_nr            = 0,
        scalingType           alphaType                 = scalingType::generic,
        scalingType           betaType                  = scalingType::generic,
        kernelInstrPreference kInstPref =
            kernelInstrPreference::avx512_zmm_favour)
    {
        // If nr is not specified, determine based on provided kType
        if (nr == -1) {
            if (kType == kernelInstrType::avx2_ymm_16_reg) {
                nr = 16; // AVX2 uses nr=16 for GEMM
            } else if (kType == kernelInstrType::avx512_ymm_32_reg) {
                nr = 32; // AVX512 YMM uses nr=32 for GEMM (half of ZMM)
            } else {
                nr = 64; // AVX512 ZMM uses nr=64 for GEMM
            }
        }

        return createKernelInfo(kType, mr, nr, kc, k_unroll, prefetch_c_dist,
                                genLtKrnlForAvailFullKrnl, term_fringe_nr,
                                alphaType, betaType, kInstPref);
    }

    /**
     * @brief Create GEMV N=1 kernel info with architecture-aware defaults
     */
    static kernelInfo createGEMVN1KernelInfo(
        kernelInstrType       kType,
        int                   mr                        = -1,
        int                   kc                        = 4096,
        int                   k_unroll                  = 4,
        int                   prefetch_c_dist           = 0,
        bool                  genLtKrnlForAvailFullKrnl = false,
        int                   term_fringe_nr            = 0,
        scalingType           alphaType                 = scalingType::generic,
        scalingType           betaType                  = scalingType::generic,
        kernelInstrPreference kInstPref =
            kernelInstrPreference::avx512_zmm_favour)
    {
        // If mr is not specified, determine based on provided kType
        if (mr == -1) {
            if (kType == kernelInstrType::avx2_ymm_16_reg) {
                mr = 8; // AVX2 uses mr=8 for GEMV N=1
            } else if (kType == kernelInstrType::avx512_ymm_32_reg) {
                mr = 8; // AVX512 YMM uses mr=8 for GEMV N=1 (half of ZMM)
            } else {
                mr = 16; // AVX512 ZMM uses mr=16 for GEMV N=1
            }
        }

        // GEMV N=1 always has nr=1
        return createKernelInfo(kType, mr, 1, kc, k_unroll, prefetch_c_dist,
                                genLtKrnlForAvailFullKrnl, term_fringe_nr,
                                alphaType, betaType, kInstPref);
    }

    /**
     * @brief Create GEMV M=1 kernel info with architecture-aware defaults
     */
    static kernelInfo createGEMVM1KernelInfo(
        kernelInstrType       kType,
        int                   nr                        = -1,
        int                   kc                        = 4096,
        int                   k_unroll                  = -1,
        int                   prefetch_c_dist           = 0,
        bool                  genLtKrnlForAvailFullKrnl = false,
        int                   term_fringe_nr            = 0,
        scalingType           alphaType                 = scalingType::generic,
        scalingType           betaType                  = scalingType::generic,
        kernelInstrPreference kInstPref =
            kernelInstrPreference::avx512_zmm_favour)
    {
        // Set nr and k_unroll (if not specified) based on provided kType
        if (kType == kernelInstrType::avx2_ymm_16_reg) {
            // AVX2 configuration
            if (nr == -1) {
                nr = 16; // AVX2 uses nr=16 for GEMV M=1
            }
            if (k_unroll == -1) {
                k_unroll = 2; // AVX2 uses k_unroll=2
            }
        } else if (kType == kernelInstrType::avx512_ymm_32_reg) {
            // AVX512 YMM configuration (half of ZMM)
            if (nr == -1) {
                nr = 32; // AVX512 YMM uses nr=32 for GEMV M=1 (half of ZMM)
            }
            if (k_unroll == -1) {
                k_unroll = 2; // Keep same k_unroll as ZMM
            }
        } else {
            // AVX512 ZMM configuration
            if (nr == -1) {
                nr = 64; // AVX512 ZMM uses nr=64 for GEMV M=1
            }
            if (k_unroll == -1) {
                k_unroll = 4; // AVX512 ZMM uses k_unroll=4
            }
        }

        // GEMV M=1 always has mr=1
        return createKernelInfo(kType, 1, nr, kc, k_unroll, prefetch_c_dist,
                                genLtKrnlForAvailFullKrnl, term_fringe_nr,
                                alphaType, betaType, kInstPref);
    }
};

// ============================================================================
// KernelOps Array Builder
// ============================================================================

class KernelOpsBuilder
{
  public:
    std::vector<kernelOpsMetaData> ops;

    // Add BIAS with optional dequantization
    KernelOpsBuilder& addBias(DataType      biasDt = DataType::f32,
                              storageFormat format = storageFormat::rowMajor,
                              bool          withScaleFactor = false,
                              DataType      sfDt            = DataType::f32,
                              bool          scalarSF        = true,
                              bool          withZeroPoint   = false,
                              DataType      zpDt            = DataType::s8,
                              bool          scalarZP        = true)
    {
        kernelOpsMetaData bias;
        bias.type           = kernelOps::bias;
        bias.paramStorageDt = biasDt;
        bias.cMatFormat     = format;

        if (withScaleFactor) {
            bias.scaleFactorDt             = sfDt;
            bias.scalarScaleFactorRequired = scalarSF;
            bias.vectorScaleFactorRequired = !scalarSF;
        }

        if (withZeroPoint) {
            bias.zeroPointDt             = zpDt;
            bias.scalarZeroPointRequired = scalarZP;
            bias.vectorZeroPointRequired = !scalarZP;
        }

        ops.push_back(bias);
        return *this;
    }

    // Add activation functions
    KernelOpsBuilder& addReLU()
    {
        kernelOpsMetaData op;
        op.type = kernelOps::relu;
        ops.push_back(op);
        return *this;
    }

    KernelOpsBuilder& addReLUScale(DataType dt = DataType::f32)
    {
        kernelOpsMetaData op;
        op.type           = kernelOps::reluScale;
        op.paramStorageDt = dt;
        ops.push_back(op);
        return *this;
    }

    KernelOpsBuilder& addGeLUTanh()
    {
        kernelOpsMetaData op;
        op.type = kernelOps::geluTanh;
        ops.push_back(op);
        return *this;
    }

    KernelOpsBuilder& addGeLUErf()
    {
        kernelOpsMetaData op;
        op.type = kernelOps::geluErf;
        ops.push_back(op);
        return *this;
    }

    KernelOpsBuilder& addClip(DataType dt = DataType::f32)
    {
        kernelOpsMetaData op;
        op.type           = kernelOps::clip;
        op.paramStorageDt = dt;
        ops.push_back(op);
        return *this;
    }

    KernelOpsBuilder& addSwish(DataType dt = DataType::f32)
    {
        kernelOpsMetaData op;
        op.type           = kernelOps::swish;
        op.paramStorageDt = dt;
        ops.push_back(op);
        return *this;
    }

    KernelOpsBuilder& addTanh()
    {
        kernelOpsMetaData op;
        op.type = kernelOps::tanh;
        ops.push_back(op);
        return *this;
    }

    KernelOpsBuilder& addSigmoid()
    {
        kernelOpsMetaData op;
        op.type = kernelOps::sigmoid;
        ops.push_back(op);
        return *this;
    }

    // Add DOWNSCALE
    KernelOpsBuilder& addDownscale(
        DataType      sfDt     = DataType::f32,
        DataType      zpDt     = DataType::s8,
        storageFormat format   = storageFormat::rowMajor,
        bool          scalarSF = true,
        bool          scalarZP = true)
    {
        kernelOpsMetaData op;
        op.type                      = kernelOps::downscale;
        op.scaleFactorDt             = sfDt;
        op.zeroPointDt               = zpDt;
        op.cMatFormat                = format;
        op.scalarScaleFactorRequired = scalarSF;
        op.vectorScaleFactorRequired = !scalarSF;
        op.scalarZeroPointRequired   = scalarZP;
        op.vectorZeroPointRequired   = !scalarZP;
        ops.push_back(op);
        return *this;
    }

    // Add MATADD
    KernelOpsBuilder& addMatAdd(DataType      matDt  = DataType::f32,
                                DataType      sfDt   = DataType::f32,
                                storageFormat format = storageFormat::rowMajor,
                                bool          scalarSF = true)
    {
        kernelOpsMetaData op;
        op.type                      = kernelOps::matAdd;
        op.paramStorageDt            = matDt;
        op.scaleFactorDt             = sfDt;
        op.cMatFormat                = format;
        op.scalarScaleFactorRequired = scalarSF;
        op.vectorScaleFactorRequired = !scalarSF;
        ops.push_back(op);
        return *this;
    }

    // Add MATMUL
    KernelOpsBuilder& addMatMul(DataType      matDt  = DataType::f32,
                                DataType      sfDt   = DataType::f32,
                                storageFormat format = storageFormat::rowMajor,
                                bool          scalarSF = true)
    {
        kernelOpsMetaData op;
        op.type                      = kernelOps::matMul;
        op.paramStorageDt            = matDt;
        op.scaleFactorDt             = sfDt;
        op.cMatFormat                = format;
        op.scalarScaleFactorRequired = scalarSF;
        op.vectorScaleFactorRequired = !scalarSF;
        ops.push_back(op);
        return *this;
    }

    // Add A_DEQUANTIZE (for integer kernels)
    KernelOpsBuilder& addADequantize(
        DataType      sfDt     = DataType::f32,
        DataType      zpDt     = DataType::s8,
        storageFormat format   = storageFormat::rowMajor,
        bool          scalarSF = true,
        bool          scalarZP = true)
    {
        kernelOpsMetaData op;
        op.type                      = kernelOps::aDQuantize;
        op.scaleFactorDt             = sfDt;
        op.zeroPointDt               = zpDt;
        op.cMatFormat                = format;
        op.scalarScaleFactorRequired = scalarSF;
        op.vectorScaleFactorRequired = !scalarSF;
        op.scalarZeroPointRequired   = scalarZP;
        op.vectorZeroPointRequired   = !scalarZP;
        ops.push_back(op);
        return *this;
    }

    // Preset: Maximal configuration (all post-ops, heaviest paths)
    static KernelOpsBuilder createMaximalConfig()
    {
        return KernelOpsBuilder()
            .addBias(DataType::bf16, storageFormat::colMajor, true,
                     DataType::bf16, false, true, DataType::s8, false)
            .addReLU()
            .addReLUScale(DataType::bf16)
            .addGeLUTanh()
            .addGeLUErf()
            .addClip(DataType::s32)
            .addSwish(DataType::bf16)
            .addTanh()
            .addSigmoid()
            .addDownscale(DataType::bf16, DataType::s8, storageFormat::colMajor,
                          false, false)
            .addMatAdd(DataType::s8, DataType::bf16, storageFormat::colMajor,
                       false)
            .addMatMul(DataType::u8, DataType::s32, storageFormat::colMajor,
                       false)
            .addADequantize(DataType::bf16, DataType::s8,
                            storageFormat::rowMajor, false, false);
    }

    std::vector<kernelOpsMetaData> build() { return ops; }
};

// ============================================================================
// Base Test Fixture for JIT Generator Tests
// ============================================================================

/**
 * @brief Base class for JIT generator tests providing common functionality
 *
 * This class provides:
 * - Architecture detection and caching
 * - Kernel generation dispatch methods for all datatypes and kernel shapes
 *
 * Generators use Xbyak AutoGrow mode and manage their own memory internally.
 */
class JitGeneratorTestBase : public ::testing::Test
{
  protected:
    // Cached vectors of all supported kernel types
    // F32 includes both ZMM and YMM for AVX512, others currently only ZMM
    std::vector<kernelInstrType> allKTypes_f32;
    std::vector<kernelInstrType> allKTypes_bf16;
    std::vector<kernelInstrType> allKTypes_u8s8;
    std::vector<kernelInstrType> allKTypes_s8;

    void SetUp() override
    {
        allKTypes_f32  = ArchBasedKernelTypes::getAllKernelTypesForF32();
        allKTypes_bf16 = ArchBasedKernelTypes::getAllKernelTypesForBF16();
        allKTypes_u8s8 = ArchBasedKernelTypes::getAllKernelTypesForU8S8();
        allKTypes_s8   = ArchBasedKernelTypes::getAllKernelTypesForS8();
    }

    void expectGenerationSuccess(
        const std::string&                               testName,
        std::function<dlp::jit::jitGeneratorError(void)> generateFunc)
    {
        auto result = CrashIsolation::runIsolated(
            [&]() { return static_cast<int>(generateFunc()); });

        if (result.crashed) {
            FAIL() << testName << " crashed with "
                   << CrashIsolation::signalName(result.crashSignal)
                   << " (signal " << result.crashSignal << ")";
        } else if (result.threw) {
            FAIL() << testName
                   << ": Exception thrown: " << result.exceptionMessage;
        } else {
            EXPECT_EQ(
                static_cast<dlp::jit::jitGeneratorError>(result.returnCode),
                dlp::jit::jitGeneratorError::success)
                << testName << " failed to generate";
        }
    }

    // ========================================================================
    // GEMM kernel generation dispatch (by kType, returns error code)
    // ========================================================================

    dlp::jit::jitGeneratorError generateF32GemmKernel(
        kernelInstrType kType, generatorParams& gen_params)
    {
        if (kType == kernelInstrType::avx512_zmm_32_reg) {
            amdzen::GEMMcodeGenerator::jitGEMMF32<
                kernelInstrType::avx512_zmm_32_reg>
                base(JIT_KERNEL_SIZE);
            return base.generateKernel(gen_params);
        } else if (kType == kernelInstrType::avx512_ymm_32_reg) {
            amdzen::GEMMcodeGenerator::jitGEMMF32<
                kernelInstrType::avx512_ymm_32_reg>
                base(JIT_KERNEL_SIZE);
            return base.generateKernel(gen_params);
        } else if (kType == kernelInstrType::avx2_ymm_16_reg) {
            amdzen::GEMMcodeGenerator::jitGEMMF32<
                kernelInstrType::avx2_ymm_16_reg>
                base(JIT_KERNEL_SIZE);
            return base.generateKernel(gen_params);
        }
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    dlp::jit::jitGeneratorError generateBF16GemmKernel(
        kernelInstrType kType, generatorParams& gen_params)
    {
        if (kType == kernelInstrType::avx512_zmm_32_reg) {
            amdzen::GEMMcodeGenerator::jitGEMMBF16<
                kernelInstrType::avx512_zmm_32_reg>
                base(JIT_KERNEL_SIZE);
            return base.generateKernel(gen_params);
        }
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    dlp::jit::jitGeneratorError generateU8S8GemmKernel(
        kernelInstrType kType, generatorParams& gen_params)
    {
        if (kType == kernelInstrType::avx512_zmm_32_reg) {
            amdzen::gen::jitU8S8VNNI_GEMM<kernelInstrType::avx512_zmm_32_reg>
                base(JIT_KERNEL_SIZE);
            return base.generateKernel(gen_params);
        }
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    dlp::jit::jitGeneratorError generateS8GemmKernel(
        kernelInstrType kType, generatorParams& gen_params)
    {
        if (kType == kernelInstrType::avx512_zmm_32_reg) {
            amdzen::GEMMcodeGenerator::jitGEMMS8<
                kernelInstrType::avx512_zmm_32_reg>
                base(JIT_KERNEL_SIZE);
            return base.generateKernel(gen_params);
        }
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // ========================================================================
    // GEMV N=1 kernel generation dispatch (by kType, returns error code)
    // ========================================================================

    dlp::jit::jitGeneratorError generateF32GemvN1Kernel(
        kernelInstrType kType, gemvN1GeneratorParams& gen_params)
    {
        if (kType == kernelInstrType::avx512_zmm_32_reg) {
            jitF32GEMVN1<kernelInstrType::avx512_zmm_32_reg> base(
                JIT_KERNEL_SIZE);
            return base.generateKernel(gen_params);
        } else if (kType == kernelInstrType::avx512_ymm_32_reg) {
            jitF32GEMVN1<kernelInstrType::avx512_ymm_32_reg> base(
                JIT_KERNEL_SIZE);
            return base.generateKernel(gen_params);
        } else if (kType == kernelInstrType::avx2_ymm_16_reg) {
            jitF32GEMVN1<kernelInstrType::avx2_ymm_16_reg> base(
                JIT_KERNEL_SIZE);
            return base.generateKernel(gen_params);
        }
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    dlp::jit::jitGeneratorError generateBF16GemvN1Kernel(
        kernelInstrType kType, gemvN1GeneratorParams& gen_params)
    {
        if (kType == kernelInstrType::avx512_zmm_32_reg) {
            jitBF16GEMVN1<kernelInstrType::avx512_zmm_32_reg> base(
                JIT_KERNEL_SIZE);
            return base.generateKernel(gen_params);
        }
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    dlp::jit::jitGeneratorError generateU8S8GemvN1Kernel(
        kernelInstrType kType, gemvN1GeneratorParams& gen_params)
    {
        if (kType == kernelInstrType::avx512_zmm_32_reg) {
            jitU8S8VNNI_GEMVN1<kernelInstrType::avx512_zmm_32_reg> base(
                JIT_KERNEL_SIZE);
            return base.generateKernel(gen_params);
        }
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    dlp::jit::jitGeneratorError generateS8GemvN1Kernel(
        kernelInstrType kType, gemvN1GeneratorParams& gen_params)
    {
        if (kType == kernelInstrType::avx512_zmm_32_reg) {
            jitGEMVS8N1<kernelInstrType::avx512_zmm_32_reg> base(
                JIT_KERNEL_SIZE);
            return base.generateKernel(gen_params);
        }
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    // ========================================================================
    // GEMV M=1 kernel generation dispatch (by kType, returns error code)
    // ========================================================================

    dlp::jit::jitGeneratorError generateF32GemvM1Kernel(
        kernelInstrType kType, gemvM1GeneratorParams& gen_params)
    {
        if (kType == kernelInstrType::avx512_zmm_32_reg) {
            jitF32GEMVM1<kernelInstrType::avx512_zmm_32_reg> base(
                JIT_KERNEL_SIZE);
            return base.generateKernel(gen_params);
        } else if (kType == kernelInstrType::avx512_ymm_32_reg) {
            jitF32GEMVM1<kernelInstrType::avx512_ymm_32_reg> base(
                JIT_KERNEL_SIZE);
            return base.generateKernel(gen_params);
        } else if (kType == kernelInstrType::avx2_ymm_16_reg) {
            gen_params.N_LEFT = 15;
            jitF32GEMVM1<kernelInstrType::avx2_ymm_16_reg> base(
                JIT_KERNEL_SIZE);
            return base.generateKernel(gen_params);
        }
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    dlp::jit::jitGeneratorError generateBF16GemvM1Kernel(
        kernelInstrType kType, gemvM1GeneratorParams& gen_params)
    {
        if (kType == kernelInstrType::avx512_zmm_32_reg) {
            jitBF16GEMVM1<kernelInstrType::avx512_zmm_32_reg> base(
                JIT_KERNEL_SIZE);
            return base.generateKernel(gen_params);
        }
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    dlp::jit::jitGeneratorError generateU8S8GemvM1Kernel(
        kernelInstrType kType, gemvM1GeneratorParams& gen_params)
    {
        if (kType == kernelInstrType::avx512_zmm_32_reg) {
            jitU8S8VNNI_GEMVM1<kernelInstrType::avx512_zmm_32_reg> base(
                JIT_KERNEL_SIZE);
            return base.generateKernel(gen_params);
        }
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }

    dlp::jit::jitGeneratorError generateS8GemvM1Kernel(
        kernelInstrType kType, gemvM1GeneratorParams& gen_params)
    {
        if (kType == kernelInstrType::avx512_zmm_32_reg) {
            jitGEMVS8M1<kernelInstrType::avx512_zmm_32_reg> base(
                JIT_KERNEL_SIZE);
            return base.generateKernel(gen_params);
        }
        return dlp::jit::jitGeneratorError::badKernelInfo;
    }
};

} // namespace test_jit_utils
