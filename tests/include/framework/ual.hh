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

#include "matrix.hh"
#include <memory>
#include <string>

namespace dlp::testing::framework {

// Forward declaration for IOperation
class IOperation;

/**
 * @enum UALType
 * @brief Unified Abstraction Layer implementation type enumeration
 *
 * Defines the available backend implementations for the UAL.
 */
enum class UALType : uint32_t
{
    DLP,    ///< Deep Learning Primitives implementation
    MKL,    ///< Intel Math Kernel Library implementation
    ONEDNN, ///< OneDNN implementation
    REF,    ///< Reference implementation
};

/**
 * @class IUal
 * @brief Interface for Unified Abstraction Layer
 *
 * This abstract class defines the interface for Unified Abstraction Layer
 * (UAL) implementations, providing a common API for different backend
 * libraries.
 */
class IUal
{
  public:
    /**
     * @brief Constructor
     *
     * @param type UAL implementation type
     */
    IUal(UALType type)
        : m_type(type)
    {
    }

    /**
     * @brief Virtual destructor
     */
    virtual ~IUal() = default;

    /**
     * @brief Reorder matrix data to specified accumulation type
     *
     * @param in Input matrix to reorder
     * @param out Output matrix to store reordered data
     * @param accType Accumulation type
     * @return true on success
     */
    virtual bool reorder(const Matrix& in, Matrix& out, MatrixType accType) = 0;

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
    virtual bool gemm(const Matrix& A,
                      const Matrix& B,
                      Matrix&       C,
                      MatrixType    accType,
                      double        alpha = 1.0,
                      double        beta  = 0.0) = 0;

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
     * @return true on success
     */
    virtual bool gemm(const Matrix&                      A,
                      const Matrix&                      B,
                      Matrix&                            C,
                      MatrixType                         accType,
                      const std::shared_ptr<IOperation>& postOps,
                      double                             alpha = 1.0,
                      double                             beta  = 0.0) = 0;

    /**
     * @brief Get string representation of UAL type
     *
     * @param type UAL type
     * @return std::string Human-readable name
     */
    virtual std::string toString(UALType type) = 0;

    /**
     * @brief Get the UAL type
     *
     * @return UALType
     */
    virtual UALType getUALType() const = 0;

  private:
    UALType m_type;
};

} // namespace dlp::testing::framework
