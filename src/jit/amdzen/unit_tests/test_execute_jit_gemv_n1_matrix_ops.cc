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
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "jit_generator_tests_utils.hh"
#if !DLP_OS_WINDOWS
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace amdzen::codegen;
using namespace amdzen::utils;
using namespace dlp::kernel_frame;
using namespace test_jit_utils;

namespace {

enum class MatrixOp
{
    Add,
    Mul
};

enum class ScaleMode
{
    Scalar,
    Vector
};

class JitGemvN1MatrixOpsTest : public JitGeneratorTestBase
{
  protected:
    static constexpr size_t kGatherOverreadBytes = 16;

    static int auxElemSize(DataType dt)
    {
        switch (dt) {
            case DataType::s8:
                return 1;
            case DataType::bf16:
                return 2;
            default:
                return 4;
        }
    }

    static DLP_TYPE toDlpType(DataType dt)
    {
        switch (dt) {
            case DataType::s8:
                return DLP_S8;
            case DataType::bf16:
                return DLP_BF16;
            default:
                return DLP_F32;
        }
    }

    static uint16_t f32ToBf16Bits(float value)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return static_cast<uint16_t>(bits >> 16);
    }

    static void fillAuxPoison(std::vector<uint8_t>& buf, DataType dt)
    {
        if (dt == DataType::s8) {
            std::fill(buf.begin(), buf.end(), static_cast<uint8_t>(0x80));
            return;
        }
        if (dt == DataType::bf16) {
            const uint16_t bits = f32ToBf16Bits(-1000.0f);
            for (size_t i = 0; i + 1 < buf.size(); i += 2) {
                std::memcpy(buf.data() + i, &bits, sizeof(bits));
            }
            return;
        }
        const float poison = -1000.0f;
        for (size_t i = 0; i + 3 < buf.size(); i += 4) {
            std::memcpy(buf.data() + i, &poison, sizeof(poison));
        }
    }

    static void storeAuxElem(uint8_t* base,
                             size_t   index,
                             DataType dt,
                             float    value)
    {
        if (dt == DataType::s8) {
            base[index] = static_cast<uint8_t>(static_cast<int8_t>(value));
        } else if (dt == DataType::bf16) {
            const uint16_t bits = f32ToBf16Bits(value);
            std::memcpy(base + index * 2, &bits, sizeof(bits));
        } else {
            std::memcpy(base + index * 4, &value, sizeof(value));
        }
    }

    template<kernelInstrType KType>
    void runCase(MatrixOp      opType,
                 storageFormat format,
                 ScaleMode     scaleMode,
                 DataType      auxDt               = DataType::f32,
                 int           startCol            = 1,
                 int           logicalCols         = -1,
                 bool          inducedColMajorCsC1 = false)
    {
        constexpr int mr = (KType == kernelInstrType::avx512_zmm_32_reg) ? 16
                                                                         : 8;
        constexpr int fringe   = 3;
        constexpr int m        = mr + fringe;
        constexpr int startRow = 2;
        ASSERT_TRUE(!inducedColMajorCsC1 || format == storageFormat::colMajor);
        if (logicalCols < 0) {
            logicalCols = startCol + 2;
        }

        gemvN1GeneratorParams genParams(mr, fringe, DLP_F32, true, false, true,
                                        false, format, scalingType::zero,
                                        scalingType::one, KType);

        kernelOpsMetaData op;
        op.type           = (opType == MatrixOp::Add) ? kernelOps::matAdd
                                                      : kernelOps::matMul;
        op.paramStorageDt = auxDt;
        op.scaleFactorDt  = DataType::f32;
        op.sfDim          = (scaleMode == ScaleMode::Scalar) ? ParamDim::Scalar
                                                             : ParamDim::PerN;
        op.cMatFormat     = format;
        genParams.kernelOps.push_back(op);

        jitF32GEMVN1<KType> generator(JIT_KERNEL_SIZE);
        ASSERT_EQ(generator.generateKernel(genParams),
                  dlp::jit::jitGeneratorError::success);
        generator.ready();
        auto kernel = generator.template getCode<jit_gemv_n1_kernel>();
        ASSERT_NE(kernel, nullptr);

        const int logicalRows = startRow + m + 1;
        // Induced M=1 col-major stores the 1xN aux with stride ldm along
        // kernel M (same indexing as row-major). Compact ldm=1 hits the
        // contiguous arm; padded ldm hits the strided arm.
        const bool strideAlongM = (format == storageFormat::rowMajor)
                                  || inducedColMajorCsC1;
        const md_t compactLdm =
            inducedColMajorCsC1 ? md_t{ 1 }
                                : (strideAlongM ? logicalCols : logicalRows);
        const md_t paddedLdm = compactLdm + 5;
        const int  elemBytes = auxElemSize(auxDt);

        std::vector<float> scaleFactors;
        float              scalarScale = 2.0f;
        if (scaleMode == ScaleMode::Vector) {
            const int scaleCount =
                (format == storageFormat::rowMajor) ? logicalCols : logicalRows;
            scaleFactors.resize(scaleCount);
            for (int i = 0; i < scaleCount; ++i) {
                scaleFactors[i] = static_cast<float>(i + 1);
            }
        }

        std::vector<float> compactOutput;
        for (const md_t ldm : { compactLdm, paddedLdm }) {
            md_t         runtimeLdm = ldm;
            const size_t auxElems =
                strideAlongM ? static_cast<size_t>(logicalRows * ldm)
                             : static_cast<size_t>(logicalCols * ldm);
            std::vector<uint8_t> auxiliary(auxElems
                                               * static_cast<size_t>(elemBytes)
                                           + kGatherOverreadBytes);
            fillAuxPoison(auxiliary, auxDt);

            for (int i = 0; i < m; ++i) {
                const size_t index =
                    strideAlongM
                        ? static_cast<size_t>((startRow + i) * ldm + startCol)
                        : static_cast<size_t>(startCol * ldm + startRow + i);
                storeAuxElem(auxiliary.data(), index, auxDt,
                             static_cast<float>(i + 2));
            }

            char order = (format == storageFormat::rowMajor) ? 'r' : 'c';
            dlp_gemm_post_op runtimeOp{};
            runtimeOp.op_code  = (opType == MatrixOp::Add) ? POST_OPS_MATRIX_ADD
                                                           : POST_OPS_MATRIX_MUL;
            runtimeOp.op_args1 = auxiliary.data();
            runtimeOp.op_args2 = &order;
            runtimeOp.op_args3 = &runtimeLdm;
            runtimeOp.scale_factor =
                (scaleMode == ScaleMode::Scalar)
                    ? static_cast<void*>(&scalarScale)
                    : static_cast<void*>(scaleFactors.data());
            runtimeOp.scale_factor_len =
                (scaleMode == ScaleMode::Scalar)
                    ? 1
                    : static_cast<md_t>(scaleFactors.size());
            runtimeOp.stor_type        = toDlpType(auxDt);
            runtimeOp.sf_stor_type     = DLP_F32;
            runtimeOp.scale_factor_dim = (scaleMode == ScaleMode::Scalar)
                                             ? DLP_PARAM_DIM_PER_TENSOR
                                             : DLP_PARAM_DIM_PER_CHANNEL;

            dlp_gemm_post_op_attr attr{};
            attr.post_op_c_i = startRow;
            attr.post_op_c_j = startCol;
            attr.is_first_k  = 1;
            attr.is_last_k   = 1;
            attr.c_stor_type = DLP_F32;

            std::vector<float> a(m, 0.0f);
            std::vector<float> x(1, 0.0f);
            std::vector<float> y(m);
            for (int i = 0; i < m; ++i) {
                y[i] = static_cast<float>(10 + i);
            }
            const std::vector<float> inputY = y;
            float                    alpha  = 0.0f;
            float                    beta   = 1.0f;

            const md_t csC =
                (inducedColMajorCsC1 || format == storageFormat::rowMajor)
                    ? 1
                    : logicalRows;
            dlp::kernels::gemvN1Params params(a.data(), x.data(), y.data(), m,
                                              1, 1, 1, 1, 1, 1, csC, &alpha,
                                              &beta, &runtimeOp, attr);
            params.m_iter = 1;
            params.m_left = fringe;

            if constexpr (KType == kernelInstrType::avx512_zmm_32_reg) {
                params.mmask_avx512 =
                    static_cast<uint16_t>((1u << fringe) - 1u);
            } else if constexpr (KType == kernelInstrType::avx512_ymm_32_reg) {
                params.mmask_avx512_256 =
                    static_cast<uint8_t>((1u << fringe) - 1u);
            } else {
                std::fill_n(params.mmask_avx2.begin(), fringe, -1);
            }

            kernel(&params);

            for (int i = 0; i < m; ++i) {
                const float scale    = (scaleMode == ScaleMode::Scalar)
                                           ? scalarScale
                                           : ((format == storageFormat::rowMajor)
                                                  ? scaleFactors[startCol]
                                                  : scaleFactors[startRow + i]);
                const float operand  = static_cast<float>(i + 2) * scale;
                const float expected = (opType == MatrixOp::Add)
                                           ? inputY[i] + operand
                                           : inputY[i] * operand;
                EXPECT_FLOAT_EQ(y[i], expected)
                    << "row=" << i << " ldm=" << ldm;
            }

            if (compactOutput.empty()) {
                compactOutput = y;
            } else {
                EXPECT_EQ(y, compactOutput)
                    << "padded ldm changed the logical result";
            }
        }
    }

    template<kernelInstrType KType>
    void runAllCases()
    {
        for (const MatrixOp opType : { MatrixOp::Add, MatrixOp::Mul }) {
            for (const storageFormat format :
                 { storageFormat::rowMajor, storageFormat::colMajor }) {
                for (const ScaleMode scaleMode :
                     { ScaleMode::Scalar, ScaleMode::Vector }) {
                    SCOPED_TRACE(opType == MatrixOp::Add ? "MATRIX_ADD"
                                                         : "MATRIX_MUL");
                    SCOPED_TRACE(format == storageFormat::rowMajor
                                     ? "row-major auxiliary"
                                     : "column-major auxiliary");
                    SCOPED_TRACE(scaleMode == ScaleMode::Scalar
                                     ? "scalar scale"
                                     : "vector scale");
                    runCase<KType>(opType, format, scaleMode);
                }
            }
        }
    }

#if !DLP_OS_WINDOWS
    // Place the last strided aux element in the last elemSize bytes of a
    // mapped page, with the next page PROT_NONE. AVX-512 vpgatherdd would
    // over-read and SIGSEGV for s8/bf16; element-sized loads must not.
    template<kernelInstrType KType>
    void runGuardPageCase(DataType auxDt)
    {
        constexpr int mr = (KType == kernelInstrType::avx512_zmm_32_reg) ? 16
                                                                         : 8;
        gemvN1GeneratorParams genParams(
            mr, 0, DLP_F32, true, false, false, false, storageFormat::rowMajor,
            scalingType::zero, scalingType::one, KType);

        kernelOpsMetaData op;
        op.type           = kernelOps::matAdd;
        op.paramStorageDt = auxDt;
        op.scaleFactorDt  = DataType::f32;
        op.sfDim          = ParamDim::Scalar;
        op.cMatFormat     = storageFormat::rowMajor;
        genParams.kernelOps.push_back(op);

        jitF32GEMVN1<KType> generator(JIT_KERNEL_SIZE);
        ASSERT_EQ(generator.generateKernel(genParams),
                  dlp::jit::jitGeneratorError::success);
        generator.ready();
        auto kernel = generator.template getCode<jit_gemv_n1_kernel>();
        ASSERT_NE(kernel, nullptr);

        const long pageSize = sysconf(_SC_PAGESIZE);
        ASSERT_GT(pageSize, 0);
        // Coverity does not treat gtest ASSERT_* as a noreturn; keep an
        // explicit guard so a sysconf failure cannot wrap into size_t.
        if (pageSize <= 0) {
            return;
        }
        const size_t mapLen = static_cast<size_t>(pageSize) * 2;

        struct PageMap
        {
            void*  p = MAP_FAILED;
            size_t n = 0;
            ~PageMap()
            {
                if (p != MAP_FAILED && n > 0) {
                    munmap(p, n);
                }
            }
        } pages;
        pages.p = mmap(nullptr, mapLen, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ASSERT_NE(pages.p, MAP_FAILED);
        pages.n = mapLen;
        ASSERT_EQ(mprotect(static_cast<char*>(pages.p) + pageSize, pageSize,
                           PROT_NONE),
                  0);

        const int    elemBytes = auxElemSize(auxDt);
        const md_t   ldm       = 3;
        const int    startRow  = 0;
        const int    startCol  = 0;
        const size_t lastIndex =
            static_cast<size_t>((startRow + mr - 1) * ldm + startCol);
        const size_t lastByteOff = lastIndex * static_cast<size_t>(elemBytes);
        ASSERT_LT(lastByteOff + static_cast<size_t>(elemBytes),
                  static_cast<size_t>(pageSize));

        auto* const    pageEnd  = static_cast<uint8_t*>(pages.p) + pageSize;
        uint8_t* const lastElem = pageEnd - elemBytes;
        uint8_t* const auxBase  = lastElem - lastByteOff;
        ASSERT_GE(auxBase, static_cast<uint8_t*>(pages.p));

        std::memset(pages.p, 0, static_cast<size_t>(pageSize));
        for (int i = 0; i < mr; ++i) {
            storeAuxElem(auxBase,
                         static_cast<size_t>((startRow + i) * ldm + startCol),
                         auxDt, static_cast<float>(i + 2));
        }

        char             order       = 'r';
        md_t             runtimeLdm  = ldm;
        float            scalarScale = 2.0f;
        dlp_gemm_post_op runtimeOp{};
        runtimeOp.op_code          = POST_OPS_MATRIX_ADD;
        runtimeOp.op_args1         = auxBase;
        runtimeOp.op_args2         = &order;
        runtimeOp.op_args3         = &runtimeLdm;
        runtimeOp.scale_factor     = &scalarScale;
        runtimeOp.scale_factor_len = 1;
        runtimeOp.stor_type        = toDlpType(auxDt);
        runtimeOp.sf_stor_type     = DLP_F32;
        runtimeOp.scale_factor_dim = DLP_PARAM_DIM_PER_TENSOR;

        dlp_gemm_post_op_attr attr{};
        attr.post_op_c_i = startRow;
        attr.post_op_c_j = startCol;
        attr.is_first_k  = 1;
        attr.is_last_k   = 1;
        attr.c_stor_type = DLP_F32;

        std::vector<float> a(mr, 0.0f);
        std::vector<float> x(1, 0.0f);
        std::vector<float> y(mr);
        for (int i = 0; i < mr; ++i) {
            y[i] = static_cast<float>(10 + i);
        }
        const std::vector<float> inputY = y;
        float                    alpha  = 0.0f;
        float                    beta   = 1.0f;

        dlp::kernels::gemvN1Params params(a.data(), x.data(), y.data(), mr, 1,
                                          1, 1, 1, 1, 1, 1, &alpha, &beta,
                                          &runtimeOp, attr);
        params.m_iter = 1;
        params.m_left = 0;

        auto result = CrashIsolation::runIsolated([&]() {
            kernel(&params);
            for (int i = 0; i < mr; ++i) {
                const float operand  = static_cast<float>(i + 2) * scalarScale;
                const float expected = inputY[i] + operand;
                if (y[i] != expected) {
                    return 2;
                }
            }
            return 0;
        });

        ASSERT_FALSE(result.crashed)
            << "strided "
            << (auxDt == DataType::s8     ? "s8"
                : auxDt == DataType::bf16 ? "bf16"
                                          : "f32")
            << " aux at page end: "
            << CrashIsolation::signalName(result.crashSignal);
        ASSERT_TRUE(result.completed);
        ASSERT_EQ(result.returnCode, 0) << "matrix-add result mismatch";
    }
#endif
};

TEST_F(JitGemvN1MatrixOpsTest, DirectKernelHonorsAuxiliaryLeadingDimension)
{
    std::vector<kernelInstrType> kernelTypes = allKTypes_f32;
    const bool hasAvx2 = dlp::arch_utils::archConfigManager::getInstance()
                             .isAvx2Fma3SupportedByArch();
    if (hasAvx2
        && std::find(kernelTypes.begin(), kernelTypes.end(),
                     kernelInstrType::avx2_ymm_16_reg)
               == kernelTypes.end()) {
        kernelTypes.push_back(kernelInstrType::avx2_ymm_16_reg);
    }

    if (kernelTypes.empty()) {
        GTEST_SKIP() << "F32 JIT GEMV-N1 requires AVX2 or AVX-512";
    }

    for (const kernelInstrType kType : kernelTypes) {
        SCOPED_TRACE(kTypeToArchString(kType));
        switch (kType) {
            case kernelInstrType::avx512_zmm_32_reg:
                runAllCases<kernelInstrType::avx512_zmm_32_reg>();
                break;
            case kernelInstrType::avx512_ymm_32_reg:
                runAllCases<kernelInstrType::avx512_ymm_32_reg>();
                break;
            case kernelInstrType::avx2_ymm_16_reg:
                runAllCases<kernelInstrType::avx2_ymm_16_reg>();
                break;
            default:
                FAIL() << "Unexpected F32 kernel type";
        }
    }
}

TEST_F(JitGemvN1MatrixOpsTest, DirectKernelHonorsContiguousRowMajorLdm)
{
    std::vector<kernelInstrType> kernelTypes = allKTypes_f32;
    const bool hasAvx2 = dlp::arch_utils::archConfigManager::getInstance()
                             .isAvx2Fma3SupportedByArch();
    if (hasAvx2
        && std::find(kernelTypes.begin(), kernelTypes.end(),
                     kernelInstrType::avx2_ymm_16_reg)
               == kernelTypes.end()) {
        kernelTypes.push_back(kernelInstrType::avx2_ymm_16_reg);
    }
    if (kernelTypes.empty()) {
        GTEST_SKIP() << "F32 JIT GEMV-N1 requires AVX2 or AVX-512";
    }

    for (const kernelInstrType kType : kernelTypes) {
        SCOPED_TRACE(kTypeToArchString(kType));
        switch (kType) {
            case kernelInstrType::avx512_zmm_32_reg:
                runCase<kernelInstrType::avx512_zmm_32_reg>(
                    MatrixOp::Add, storageFormat::rowMajor, ScaleMode::Scalar,
                    DataType::f32, /*startCol=*/0, /*logicalCols=*/1);
                break;
            case kernelInstrType::avx512_ymm_32_reg:
                runCase<kernelInstrType::avx512_ymm_32_reg>(
                    MatrixOp::Add, storageFormat::rowMajor, ScaleMode::Scalar,
                    DataType::f32, /*startCol=*/0, /*logicalCols=*/1);
                break;
            case kernelInstrType::avx2_ymm_16_reg:
                runCase<kernelInstrType::avx2_ymm_16_reg>(
                    MatrixOp::Add, storageFormat::rowMajor, ScaleMode::Scalar,
                    DataType::f32, /*startCol=*/0, /*logicalCols=*/1);
                break;
            default:
                FAIL() << "Unexpected F32 kernel type";
        }
    }
}

// Public aocl_gemm M=1 column-major is induced to GEMV N=1 with csC==1.
TEST_F(JitGemvN1MatrixOpsTest, DirectKernelHonorsInducedColMajorCsC)
{
    std::vector<kernelInstrType> kernelTypes = allKTypes_f32;
    const bool hasAvx2 = dlp::arch_utils::archConfigManager::getInstance()
                             .isAvx2Fma3SupportedByArch();
    if (hasAvx2
        && std::find(kernelTypes.begin(), kernelTypes.end(),
                     kernelInstrType::avx2_ymm_16_reg)
               == kernelTypes.end()) {
        kernelTypes.push_back(kernelInstrType::avx2_ymm_16_reg);
    }
    if (kernelTypes.empty()) {
        GTEST_SKIP() << "F32 JIT GEMV-N1 requires AVX2 or AVX-512";
    }

    for (const kernelInstrType kType : kernelTypes) {
        SCOPED_TRACE(kTypeToArchString(kType));
        for (const ScaleMode scaleMode :
             { ScaleMode::Scalar, ScaleMode::Vector }) {
            SCOPED_TRACE(scaleMode == ScaleMode::Scalar ? "scalar scale"
                                                        : "vector scale");
            switch (kType) {
                case kernelInstrType::avx512_zmm_32_reg:
                    runCase<kernelInstrType::avx512_zmm_32_reg>(
                        MatrixOp::Add, storageFormat::colMajor, scaleMode,
                        DataType::f32, /*startCol=*/1, /*logicalCols=*/-1,
                        /*inducedColMajorCsC1=*/true);
                    runCase<kernelInstrType::avx512_zmm_32_reg>(
                        MatrixOp::Mul, storageFormat::colMajor, scaleMode,
                        DataType::f32, /*startCol=*/1, /*logicalCols=*/-1,
                        /*inducedColMajorCsC1=*/true);
                    break;
                case kernelInstrType::avx512_ymm_32_reg:
                    runCase<kernelInstrType::avx512_ymm_32_reg>(
                        MatrixOp::Add, storageFormat::colMajor, scaleMode,
                        DataType::f32, /*startCol=*/1, /*logicalCols=*/-1,
                        /*inducedColMajorCsC1=*/true);
                    runCase<kernelInstrType::avx512_ymm_32_reg>(
                        MatrixOp::Mul, storageFormat::colMajor, scaleMode,
                        DataType::f32, /*startCol=*/1, /*logicalCols=*/-1,
                        /*inducedColMajorCsC1=*/true);
                    break;
                case kernelInstrType::avx2_ymm_16_reg:
                    runCase<kernelInstrType::avx2_ymm_16_reg>(
                        MatrixOp::Add, storageFormat::colMajor, scaleMode,
                        DataType::f32, /*startCol=*/1, /*logicalCols=*/-1,
                        /*inducedColMajorCsC1=*/true);
                    runCase<kernelInstrType::avx2_ymm_16_reg>(
                        MatrixOp::Mul, storageFormat::colMajor, scaleMode,
                        DataType::f32, /*startCol=*/1, /*logicalCols=*/-1,
                        /*inducedColMajorCsC1=*/true);
                    break;
                default:
                    FAIL() << "Unexpected F32 kernel type";
            }
        }
    }
}

TEST_F(JitGemvN1MatrixOpsTest, DirectKernelHonorsNonF32StridedAux)
{
    std::vector<kernelInstrType> kernelTypes = allKTypes_f32;
    const bool hasAvx2 = dlp::arch_utils::archConfigManager::getInstance()
                             .isAvx2Fma3SupportedByArch();
    if (hasAvx2
        && std::find(kernelTypes.begin(), kernelTypes.end(),
                     kernelInstrType::avx2_ymm_16_reg)
               == kernelTypes.end()) {
        kernelTypes.push_back(kernelInstrType::avx2_ymm_16_reg);
    }
    if (kernelTypes.empty()) {
        GTEST_SKIP() << "F32 JIT GEMV-N1 requires AVX2 or AVX-512";
    }

    for (const kernelInstrType kType : kernelTypes) {
        SCOPED_TRACE(kTypeToArchString(kType));
        for (const DataType auxDt : { DataType::bf16, DataType::s8 }) {
            SCOPED_TRACE(auxDt == DataType::bf16 ? "bf16 aux" : "s8 aux");
            switch (kType) {
                case kernelInstrType::avx512_zmm_32_reg:
                    runCase<kernelInstrType::avx512_zmm_32_reg>(
                        MatrixOp::Add, storageFormat::rowMajor,
                        ScaleMode::Scalar, auxDt);
                    runCase<kernelInstrType::avx512_zmm_32_reg>(
                        MatrixOp::Mul, storageFormat::rowMajor,
                        ScaleMode::Scalar, auxDt);
                    break;
                case kernelInstrType::avx512_ymm_32_reg:
                    runCase<kernelInstrType::avx512_ymm_32_reg>(
                        MatrixOp::Add, storageFormat::rowMajor,
                        ScaleMode::Scalar, auxDt);
                    runCase<kernelInstrType::avx512_ymm_32_reg>(
                        MatrixOp::Mul, storageFormat::rowMajor,
                        ScaleMode::Scalar, auxDt);
                    break;
                case kernelInstrType::avx2_ymm_16_reg:
                    runCase<kernelInstrType::avx2_ymm_16_reg>(
                        MatrixOp::Add, storageFormat::rowMajor,
                        ScaleMode::Scalar, auxDt);
                    runCase<kernelInstrType::avx2_ymm_16_reg>(
                        MatrixOp::Mul, storageFormat::rowMajor,
                        ScaleMode::Scalar, auxDt);
                    break;
                default:
                    FAIL() << "Unexpected F32 kernel type";
            }
        }
    }
}

#if !DLP_OS_WINDOWS
TEST_F(JitGemvN1MatrixOpsTest, DirectKernelStridedAuxDoesNotOverreadGuardPage)
{
    std::vector<kernelInstrType> kernelTypes = allKTypes_f32;
    const bool hasAvx2 = dlp::arch_utils::archConfigManager::getInstance()
                             .isAvx2Fma3SupportedByArch();
    if (hasAvx2
        && std::find(kernelTypes.begin(), kernelTypes.end(),
                     kernelInstrType::avx2_ymm_16_reg)
               == kernelTypes.end()) {
        kernelTypes.push_back(kernelInstrType::avx2_ymm_16_reg);
    }
    if (kernelTypes.empty()) {
        GTEST_SKIP() << "F32 JIT GEMV-N1 requires AVX2 or AVX-512";
    }

    for (const kernelInstrType kType : kernelTypes) {
        SCOPED_TRACE(kTypeToArchString(kType));
        for (const DataType auxDt :
             { DataType::s8, DataType::bf16, DataType::f32 }) {
            SCOPED_TRACE(auxDt == DataType::s8     ? "s8 aux"
                         : auxDt == DataType::bf16 ? "bf16 aux"
                                                   : "f32 aux");
            switch (kType) {
                case kernelInstrType::avx512_zmm_32_reg:
                    runGuardPageCase<kernelInstrType::avx512_zmm_32_reg>(auxDt);
                    break;
                case kernelInstrType::avx512_ymm_32_reg:
                    runGuardPageCase<kernelInstrType::avx512_ymm_32_reg>(auxDt);
                    break;
                case kernelInstrType::avx2_ymm_16_reg:
                    runGuardPageCase<kernelInstrType::avx2_ymm_16_reg>(auxDt);
                    break;
                default:
                    FAIL() << "Unexpected F32 kernel type";
            }
        }
    }
}
#endif

} // namespace

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
