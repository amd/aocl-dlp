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

#include "framework/ual.hh"
#include "framework/ual_plan.hh"
#include <functional>

namespace dlp::testing::classic {

using dlp::testing::framework::BatchGroup;
using dlp::testing::framework::IUal;
using dlp::testing::framework::Matrix;
using dlp::testing::framework::MatrixLayout;
using dlp::testing::framework::MatrixType;
using dlp::testing::framework::ParamDim;
using dlp::testing::framework::PreparedBatchGemmArgs;
using dlp::testing::framework::UALError;
using dlp::testing::framework::UALType;

/**
 * @class UalRef
 * @brief Reference implementation of UAL
 *
 * This class implements the IUAL interface using reference implementations,
 * providing basic matrix operations for various data types.
 */
class UalRef : public IUal
{
  public:
    /**
     * @brief Constructor
     *
     * Initializes a reference-based UAL implementation.
     */
    UalRef();

    /**
     * @brief Create a REF execution plan
     * @return Unique pointer to a RefUalPlan
     */
    std::unique_ptr<dlp::testing::framework::IUalPlan> createPlan() override;

    /**
     * @brief Get the UAL type
     *
     * @return UALType The UAL implementation type (REF)
     */
    UALType getUALType() const override;

    // Helper function to encode type combinations for switch statement
    template<MatrixType A_type,
             MatrixType B_type,
             MatrixType C_type,
             MatrixType acc_type>
    static constexpr uint64_t encode_types()
    {
        return (static_cast<uint64_t>(A_type) << 48)
               | (static_cast<uint64_t>(B_type) << 32)
               | (static_cast<uint64_t>(C_type) << 16)
               | static_cast<uint64_t>(acc_type);
    }

    static uint64_t encode_types(MatrixType a,
                                 MatrixType b,
                                 MatrixType c,
                                 MatrixType acc)
    {
        return (static_cast<uint64_t>(a) << 48)
               | (static_cast<uint64_t>(b) << 32)
               | (static_cast<uint64_t>(c) << 16) | static_cast<uint64_t>(acc);
    }

    /**
     * @brief Get string representation of UAL type
     *
     * @param type UAL type
     * @return std::string Human-readable name
     */
    std::string toString(UALType type) override;

    /**
     * @brief Reorder matrix data to specified accumulation type
     *
     * @param in Input matrix to reorder
     * @param out Output matrix to store reordered data
     * @param A_type Type of matrix A in GEMM context
     * @param B_type Type of matrix B in GEMM context
     * @param C_type Type of matrix C in GEMM context
     * @param accType Accumulation type
     * @return UALError Error code indicating success or failure
     */
    UALError reorder(const Matrix& in,
                     Matrix&       out,
                     MatrixType    A_type,
                     MatrixType    B_type,
                     MatrixType    C_type,
                     MatrixType    accType) override;

    UALError batch_gemm(std::vector<BatchGroup>& groups,
                        MatrixType               accType) override;

    /**
     * @brief Prepare backend-specific metadata for batch GEMM
     *
     * Reference implementation doesn't need backend-specific metadata,
     * so this is a no-op.
     *
     * @param args Prepared batch GEMM arguments (unused)
     */
    void batch_prepare_metadata(PreparedBatchGemmArgs& args) override;

    /**
     * @brief Perform optimized batch GEMM using pre-prepared metadata
     *
     * Reference implementation doesn't use metadata optimization,
     * so this delegates to the standard batch_gemm by reconstructing groups.
     *
     * @param prepared Pre-prepared batch GEMM arguments
     * @return UALError Error code indicating success or failure
     */
    UALError batch_gemm(const PreparedBatchGemmArgs& prepared) override;

    friend class RefUalPlan;

  private:
    /**
     * @brief Perform general matrix multiplication: C = alpha*A*B + beta*C
     *
     * @param A First input matrix
     * @param B Second input matrix
     * @param C Output matrix
     * @param accType Accumulation type
     * @param alpha Scaling factor for A*B (default: 1.0)
     * @param beta Scaling factor for C (default: 0.0)
     * @return true on success
     */
    bool gemm(const Matrix& A,
              const Matrix& B,
              Matrix&       C,
              MatrixType    accType,
              double        alpha = 1.0,
              double        beta  = 0.0);

    /**
     * @brief Validate GEMM parameters for correctness
     *
     * @param A First input matrix
     * @param B Second input matrix
     * @param C Output matrix
     * @param hasMetadata Whether the plan has post-ops metadata
     * @param group_size group_size for sym_quant API
     * @return bool True if parameters are valid, false otherwise
     */
    bool checkValidGemmParams(const Matrix& A,
                              const Matrix& B,
                              const Matrix& C,
                              bool          hasMetadata,
                              md_t          group_size = 0);

    /**
     * @brief Internal implementation of reorder with layout and
     * transposition support
     *
     * @param A Pointer to matrix data
     * @param AType Type of matrix data
     * @param rows Number of rows
     * @param cols Number of columns
     * @param leadingDim Leading dimension
     * @param layout Memory layout
     * @param transposed Whether the matrix is transposed
     * @param accType Accumulation type
     * @return true on success
     */
    bool reorder(void*        A,
                 MatrixType   AType,
                 uint32_t     rows,
                 uint32_t     cols,
                 uint32_t     leadingDim,
                 MatrixLayout layout,
                 bool         transposed,
                 MatrixType   accType);

    /**
     * @brief Apply a post-operation to a matrix
     *
     * @param matrix The matrix to apply the operation to
     * @param op The post-operation parameter
     */
    template<typename T>
    void applyPostOperation(Matrix& matrix, const T& op);

    // Helper function to copy and convert source matrix to s32 matrix
    void copyAndConvertToS32(const Matrix& src, Matrix& dst_s32);

    // Helper function to convert s32 matrix to target integer type with
    // saturation
    void convertS32MatrixToTarget(const Matrix& src_s32,
                                  Matrix&       dst,
                                  MatrixType    targetType);

    // Helper function to copy and convert source matrix to f32 matrix
    // (for floating-point post-ops that need precision)
    void copyAndConvertToF32(const Matrix& src, Matrix& dst_f32);

    // Helper function to convert f32 matrix to target type with proper
    // rounding/saturation
    void convertF32MatrixToTarget(const Matrix& src_f32,
                                  Matrix&       dst,
                                  MatrixType    targetType);

    // Unified postop helper that handles type conversion
    void applyUnifiedPostOp(Matrix&                             matrix,
                            std::function<void(float*, size_t)> operation);

    // Helper methods for applying specific post-operations
    void applyRelu(Matrix& matrix);
    void applyPrelu(Matrix& matrix, const Matrix* alpha);
    void applyGeluTanh(Matrix& matrix);
    void applyGeluErf(Matrix& matrix);
    void applyClip(Matrix& matrix, const Matrix* lower, const Matrix* upper);
    void applySwish(Matrix& matrix, const Matrix* alpha);
    void applyTanh(Matrix& matrix);
    void applySigmoid(Matrix& matrix);
    void applyMish(Matrix& matrix);
    void applyScale(Matrix&       matrix,
                    const Matrix* scaleFactor,
                    ParamDim      sfDim,
                    const Matrix* zeroPoint);
    void applyBias(Matrix&       matrix,
                   const Matrix& bias,
                   const Matrix* scaleFactor = nullptr,
                   const Matrix* zeroPoint   = nullptr);
    void applyMatrixAdd(Matrix&       matrix,
                        const Matrix& other,
                        const Matrix* scaleFactor);
    void applyMatrixMul(Matrix&       matrix,
                        const Matrix& other,
                        const Matrix* scaleFactor);
};

} // namespace dlp::testing::classic
