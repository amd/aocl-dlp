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

namespace dlp::testing::classic {

using dlp::testing::framework::IOperation;
using dlp::testing::framework::IUal;
using dlp::testing::framework::Matrix;
using dlp::testing::framework::MatrixLayout;
using dlp::testing::framework::MatrixType;
using dlp::testing::framework::UALError;
using dlp::testing::framework::UALType;

/**
 * @class UalDlp
 * @brief Deep Learning Primitives implementation of UAL
 *
 * This class implements the IUAL interface using the DLP backend library,
 * providing optimized matrix operations for various data types.
 */
class UalDlp : public IUal
{
  public:
    /**
     * @brief Constructor
     *
     * Initializes a DLP-based UAL implementation.
     */
    UalDlp();

    /**
     * @brief Get the UAL type
     *
     * @return UALType The UAL implementation type (DLP)
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

    /**
     * @brief Perform general matrix multiplication with post-operations: C =
     * alpha*A*B + beta*C + PostOps
     *
     * @param A First input matrix
     * @param B Second input matrix
     * @param C Output matrix
     * @param accType Accumulation type
     * @param postOps Post-operations to apply (nullptr for no post-ops)
     * @param alpha Scaling factor for A*B (default: 1.0)
     * @param beta Scaling factor for C (default: 0.0)
     * @return UALError Error code indicating success or failure
     */
    UALError gemm(const Matrix&                      A,
                  const Matrix&                      B,
                  Matrix&                            C,
                  MatrixType                         accType,
                  const std::shared_ptr<IOperation>& postOps,
                  double                             alpha = 1.0,
                  double                             beta  = 0.0) override;

    /**
     * @brief Perform general matrix multiplication with raw pointers for
     * benchmarking
     *
     * @param m Number of rows in A and C
     * @param n Number of columns in B and C
     * @param k Number of columns in A and rows in B
     * @param matA Pointer to matrix A data
     * @param matA_type Data type of matrix A
     * @param matA_layout Memory layout of matrix A
     * @param matA_transposed Whether matrix A is transposed
     * @param matA_reordered Whether matrix A is reordered
     * @param matA_leadingDim Leading dimension of matrix A
     * @param matB Pointer to matrix B data
     * @param matB_type Data type of matrix B
     * @param matB_layout Memory layout of matrix B
     * @param matB_transposed Whether matrix B is transposed
     * @param matB_reordered Whether matrix B is reordered
     * @param matB_leadingDim Leading dimension of matrix B
     * @param matC Pointer to matrix C data
     * @param matC_type Data type of matrix C
     * @param matC_layout Memory layout of matrix C
     * @param matC_transposed Whether matrix C is transposed
     * @param matC_leadingDim Leading dimension of matrix C
     * @param accType Accumulation type
     * @param alpha Scaling factor for A*B
     * @param beta Scaling factor for C
     * @return UAL_SUCCESS on success, or an appropriate UALError error code on
     * failure
     */
    UALError gemm(md_t         m,
                  md_t         n,
                  md_t         k,
                  void*        matA,
                  MatrixType   matA_type,
                  MatrixLayout matA_layout,
                  bool         matA_transposed,
                  char         memFormatA,
                  md_t         matA_leadingDim,
                  void*        matB,
                  MatrixType   matB_type,
                  MatrixLayout matB_layout,
                  bool         matB_transposed,
                  char         memFormatB,
                  md_t         matB_leadingDim,
                  void*        matC,
                  MatrixType   matC_type,
                  MatrixLayout matC_layout,
                  bool         matC_transposed,
                  md_t         matC_leadingDim,
                  MatrixType   accType,
                  double       alpha = 1.0,
                  double       beta  = 0.0) const override;
};

} // namespace dlp::testing::classic
