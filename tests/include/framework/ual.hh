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

#include "framework/types.hh"
#include "matrix.hh"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dlp::testing::framework {

// Forward declarations
class IOperation;
struct PreparedBatchGemmArgs;

/**
 * @brief Enumerates matrix memory formats used inside the UAL test harness.
 *
 * The underlying AOCL-DLP library continues to rely on legacy character flags
 * ('n', 'r', 'p'). The testing layer converts between this strongly typed enum
 * and the legacy representation when calling into the library.
 */
enum class MatrixMemFormat : uint8_t
{
    Normal,    ///< Unmodified layout ('n')
    Reordered, ///< AOCL reorder buffer ('r')
    Packed     ///< Packed/blocked representation ('p')
};

/**
 * @brief Convert MatrixMemFormat to the legacy AOCL-DLP character flag.
 */
inline char
to_aocl_mem_format(MatrixMemFormat fmt)
{
    switch (fmt) {
        case MatrixMemFormat::Normal:
            return 'n';
        case MatrixMemFormat::Reordered:
            return 'r';
        case MatrixMemFormat::Packed:
            return 'p';
        default:
            return 'n';
    }
}

/**
 * @brief Deduce the MatrixMemFormat from the matrix metadata flags.
 */
inline MatrixMemFormat
deduce_mem_format(const Matrix& matrix)
{
    if (matrix.isPacked()) {
        return MatrixMemFormat::Packed;
    }
    if (matrix.isReordered()) {
        return MatrixMemFormat::Reordered;
    }
    return MatrixMemFormat::Normal;
}

/**
 * @brief Represents a group of matrices with uniform dimensions and metadata.
 */
struct BatchGroup
{
    std::vector<Matrix> A_matrices;
    std::vector<Matrix> B_matrices;
    std::vector<Matrix> C_matrices;

    md_t   m     = 0;
    md_t   n     = 0;
    md_t   k     = 0;
    double alpha = 1.0;
    double beta  = 0.0;

    std::shared_ptr<IOperation> postOps;

    MatrixMemFormat memFormatA = MatrixMemFormat::Normal;
    MatrixMemFormat memFormatB = MatrixMemFormat::Normal;

    size_t size() const { return A_matrices.size(); }

    bool validate() const
    {
        const size_t count = A_matrices.size();

        // Empty groups are valid (group_size=0 case)
        if (count == 0) {
            return true;
        }

        if (B_matrices.size() != count || C_matrices.size() != count) {
            return false;
        }

        const MatrixType      a_type    = A_matrices.front().getMatrixType();
        const MatrixType      b_type    = B_matrices.front().getMatrixType();
        const MatrixType      c_type    = C_matrices.front().getMatrixType();
        const MatrixLayout    a_layout  = A_matrices.front().getLayout();
        const MatrixLayout    b_layout  = B_matrices.front().getLayout();
        const MatrixLayout    c_layout  = C_matrices.front().getLayout();
        const MatrixMemFormat expectedA = deduce_mem_format(A_matrices.front());
        const MatrixMemFormat expectedB = deduce_mem_format(B_matrices.front());

        if (memFormatA != expectedA || memFormatB != expectedB) {
            return false;
        }

        for (std::size_t i = 0; i < count; ++i) {
            const Matrix& A = A_matrices[i];
            const Matrix& B = B_matrices[i];
            const Matrix& C = C_matrices[i];

            if (A.getMatrixType() != a_type || B.getMatrixType() != b_type
                || C.getMatrixType() != c_type) {
                return false;
            }

            if (A.getLayout() != a_layout || B.getLayout() != b_layout
                || C.getLayout() != c_layout) {
                return false;
            }

            if (A.getEffectiveRows() != m || A.getEffectiveCols() != k) {
                return false;
            }
            if (B.getEffectiveRows() != k || B.getEffectiveCols() != n) {
                return false;
            }
            if (C.getEffectiveRows() != m || C.getEffectiveCols() != n) {
                return false;
            }

            if (deduce_mem_format(A) != expectedA
                || deduce_mem_format(B) != expectedB) {
                return false;
            }
        }

        return true;
    }
};

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

enum class UALError
{
    UAL_SUCCESS = 0,      /**< Operation completed successfully */
    UAL_FAILURE,          /**< General failure occurred */
    UAL_POSTOPS_MISMATCH, /**< Post-operations mismatch occurred */
    UAL_CAST_ERROR,       /**< Casting error occurred */
    UAL_NOT_SUPPORTED,    /**< Operation or feature not supported */
    UAL_ERROR_MAX         /**< Maximum error code value (for bounds checking) */
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
     * @param A_type Type of matrix A in GEMM context
     * @param B_type Type of matrix B in GEMM context
     * @param C_type Type of matrix C in GEMM context
     * @param accType Accumulation type
     * @return UALError Error code indicating success or failure
     */
    virtual UALError reorder(const Matrix& in,
                             Matrix&       out,
                             MatrixType    A_type,
                             MatrixType    B_type,
                             MatrixType    C_type,
                             MatrixType    accType) = 0;

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
    virtual UALError gemm(const Matrix&                      A,
                          const Matrix&                      B,
                          Matrix&                            C,
                          MatrixType                         accType,
                          const std::shared_ptr<IOperation>& postOps,
                          double                             alpha = 1.0,
                          double                             beta  = 0.0) = 0;

    /**
     * @brief Perform batch matrix multiplication organized into groups.
     *
     * Each group contains matrices with identical dimensions and metadata. The
     * operation applies the same alpha, beta, and post-operations to all
     * matrices within a group while allowing different configurations across
     * groups.
     */
    virtual UALError batch_gemm(std::vector<BatchGroup>& groups,
                                MatrixType               accType) = 0;

    /**
     * @brief Prepare backend-specific metadata for batch GEMM
     *
     * This method is called once during benchmark setup to extract or create
     * all metadata objects. For DLP, this populates dlp_metadata_t pointers.
     * This separates the metadata preparation overhead from the hot execution
     * path.
     *
     * @param args Prepared batch GEMM arguments to populate with backend
     * metadata
     */
    virtual void batch_prepare_metadata(PreparedBatchGemmArgs& args) = 0;

    /**
     * @brief Perform optimized batch matrix multiplication using pre-prepared
     * metadata
     *
     * This optimized variant uses pre-prepared backend-specific metadata to
     * eliminate allocation and casting overhead from the hot execution path.
     * Call batch_prepare_metadata() once before using this method.
     *
     * NOTE: This method assumes batch_prepare_metadata() was called
     * successfully. Validation is minimal for performance - ensure args are
     * valid before calling.
     *
     * @param prepared Pre-prepared batch GEMM arguments with backend metadata
     * @return UALError Error code indicating success or failure
     */
    virtual UALError batch_gemm(const PreparedBatchGemmArgs& prepared) = 0;

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
     * @param matA_leadingDim Leading dimension of matrix A
     * @param matB Pointer to matrix B data
     * @param matB_type Data type of matrix B
     * @param matB_layout Memory layout of matrix B
     * @param matB_transposed Whether matrix B is transposed
     * @param matB_leadingDim Leading dimension of matrix B
     * @param matC Pointer to matrix C data
     * @param matC_type Data type of matrix C
     * @param matC_layout Memory layout of matrix C
     * @param matC_transposed Whether matrix C is transposed
     * @param matC_leadingDim Leading dimension of matrix C
     * @param accType Accumulation type
     * @param alpha Scaling factor for A*B
     * @param beta Scaling factor for C
     * @return true on success
     */
    virtual UALError gemm(md_t         m,
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
                          double       beta  = 0.0) const = 0;

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
