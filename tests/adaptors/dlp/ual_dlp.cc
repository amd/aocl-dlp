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

/**
 * @file ual_dlp.cc
 * @brief Implementation of the DLP Unified Abstraction Layer
 *
 * This file contains the implementation of the DLP-based UAL, providing
 * optimized matrix operations with support for various data types, memory
 * layouts, and virtual transposition.
 */

#include "adaptors/dlp/ual_dlp.hh"
#include "adaptors/dlp/operation_dlp.hh"
#include "batch_post_ops.hh"
#include "framework/batch_gemm_args.hh"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include "aocl_dlp.h" // IWYU pragma: keep
#include "classic/aocl_gemm_post_ops.h"
#include "classic/dlp_errors.h"

using namespace dlp::testing::framework;

namespace dlp::testing::classic {

/**
 * @brief Constructor for UalDlp
 *
 * Initializes a DLP-based Unified Abstraction Layer implementation.
 */
UalDlp::UalDlp()
    : IUal(UALType::DLP)
{
}

/**
 * @brief Get the UAL implementation type
 *
 * @return UALType::DLP for this implementation
 */
UALType
UalDlp::getUALType() const
{
    return UALType::DLP;
}

/**
 * @brief Convert UAL type to human-readable string
 *
 * @param type The UAL type to convert
 * @return std::string Human-readable description
 */
std::string
UalDlp::toString(UALType type)
{
    switch (type) {
        case UALType::DLP:
            return "Deep Learning Primitives";
        case UALType::MKL:
            return "Intel MKL";
        case UALType::ONEDNN:
            return "OneDNN";
        case UALType::REF:
            return "Reference";
        default:
            return "Unknown UAL";
    }
}

/**
 * @brief Public reorder interface that unpacks Matrix object
 *
 * @param in Input matrix to reorder
 * @param out Output matrix to store reordered data
 * @param A_type Type of matrix A in GEMM context
 * @param B_type Type of matrix B in GEMM context
 * @param C_type Type of matrix C in GEMM context
 * @param accType Target accumulation type
 * @return UALError Error code indicating success or failure
 */
UALError
UalDlp::reorder(const Matrix& in,
                Matrix&       out,
                MatrixType    A_type,
                MatrixType    B_type,
                MatrixType    C_type,
                MatrixType    accType)
{
    dlp_metadata_t meta;
    meta.error_hndl.error_code = DLP_CLSC_SUCCESS;

    // Use effective (logical) dimensions for reordering
    md_t effective_rows = in.getEffectiveRows();
    md_t effective_cols = in.getEffectiveCols();

    // Determine appropriate reorder function based on input type and GEMM
    // context The A, B, C types provide context for optimal reordering strategy
    msz_t alloc_bytes = 0;

    // Select reorder function based on input matrix type and GEMM context
    if (in.getMatrixType() == MatrixType::f32) {
        // For mixed precision scenarios, we might need different handling
        if (A_type == MatrixType::f32 && B_type == MatrixType::f32
            && C_type == MatrixType::f32) {
            alloc_bytes = aocl_get_reorder_buf_size_f32f32f32of32(
                in.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c',
                in.isTransposed() ? 't' : 'n', 'B', effective_rows,
                effective_cols, &meta);
        } else {
            // Handle mixed precision cases - for now, fall back to standard f32
            // reorder
            alloc_bytes = aocl_get_reorder_buf_size_f32f32f32of32(
                in.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c',
                in.isTransposed() ? 't' : 'n', 'B', effective_rows,
                effective_cols, &meta);
        }
    } else if (in.getMatrixType() == MatrixType::bf16) {
        // For bf16, consider the accumulation type and output type
        if (accType == MatrixType::f32) {
            alloc_bytes = aocl_get_reorder_buf_size_bf16bf16f32of32(
                in.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c',
                in.isTransposed() ? 't' : 'n', 'B', effective_rows,
                effective_cols, &meta);
        } else {
            // Handle other accumulation types - for now, fall back to standard
            alloc_bytes = aocl_get_reorder_buf_size_bf16bf16f32of32(
                in.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c',
                in.isTransposed() ? 't' : 'n', 'B', effective_rows,
                effective_cols, &meta);
        }
    } else if (in.getMatrixType() == MatrixType::s8) {
        // For s8, consider the accumulation type and output type
        if (accType == MatrixType::s32) {
            alloc_bytes = aocl_get_reorder_buf_size_s8s8s32os32(
                in.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c',
                in.isTransposed() ? 't' : 'n', 'B', effective_rows,
                effective_cols, &meta);
        } else {
            // Handle other accumulation types - for now, fall back to standard
            alloc_bytes = aocl_get_reorder_buf_size_s8s8s32os32(
                in.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c',
                in.isTransposed() ? 't' : 'n', 'B', effective_rows,
                effective_cols, &meta);
        }
    } else {
        return UALError::UAL_NOT_SUPPORTED;
    }

    if (meta.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED) {
        return UALError::UAL_NOT_SUPPORTED;
    }
    if (meta.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        return UALError::UAL_FAILURE;
    }

    // Create output matrix with dimensions that preserve the logical operation
    // If input was transposed, the reordered output should have physical
    // dimensions that match the effective dimensions (since reordering handles
    // transposition)
    md_t out_rows       = effective_rows;
    md_t out_cols       = effective_cols;
    md_t out_leadingDim = in.getLayout() == MatrixLayout::ROW_MAJOR ? out_cols
                                                                    : out_rows;

    // Create output matrix using the new interface with external memory
    // allocation
    // Leading dimension needs to be recomputed as the matrix is copied.
    auto memory = MatrixMemory::allocateBytes(alloc_bytes, 4096);
    out =
        Matrix(out_rows, out_cols, in.getMatrixType(), std::move(memory),
               alloc_bytes, in.getLayout(), out_leadingDim, false, true, 4096);

    char layout = in.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c';

    // Use the type information to select appropriate reorder function
    switch (in.getMatrixType()) {
        case MatrixType::f32:
            // Consider GEMM context for optimal reordering
            if (A_type == MatrixType::f32 && B_type == MatrixType::f32
                && C_type == MatrixType::f32) {
                aocl_reorder_f32f32f32of32(
                    layout, in.isTransposed() ? 't' : 'n', 'B',
                    reinterpret_cast<const float*>(
                        in.getMatrixData().getMatrixPtr()),
                    reinterpret_cast<float*>(
                        out.getMatrixData().getMatrixPtr()),
                    effective_rows, effective_cols, in.getLeadingDimension(),
                    &meta);
            } else {
                // Handle mixed precision - for now, use standard f32 reorder
                aocl_reorder_f32f32f32of32(
                    layout, in.isTransposed() ? 't' : 'n', 'B',
                    reinterpret_cast<const float*>(
                        in.getMatrixData().getMatrixPtr()),
                    reinterpret_cast<float*>(
                        out.getMatrixData().getMatrixPtr()),
                    effective_rows, effective_cols, in.getLeadingDimension(),
                    &meta);
            }
            break;
        case MatrixType::bf16:
            // Consider accumulation type for bf16 reordering
            aocl_reorder_bf16bf16f32of32(
                layout, in.isTransposed() ? 't' : 'n', 'B',
                reinterpret_cast<const bfloat16*>(
                    in.getMatrixData().getMatrixPtr()),
                reinterpret_cast<bfloat16*>(out.getMatrixData().getMatrixPtr()),
                effective_rows, effective_cols, in.getLeadingDimension(),
                &meta);
            break;
        case MatrixType::s8:
            aocl_reorder_s8s8s32os32(
                layout, in.isTransposed() ? 't' : 'n', 'B',
                reinterpret_cast<const int8_t*>(
                    in.getMatrixData().getMatrixPtr()),
                reinterpret_cast<int8_t*>(out.getMatrixData().getMatrixPtr()),
                effective_rows, effective_cols, in.getLeadingDimension(),
                &meta);
            break;
        default:
            return UALError::UAL_NOT_SUPPORTED;
    }

    if (meta.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED) {
        return UALError::UAL_NOT_SUPPORTED;
    }
    if (meta.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        return UALError::UAL_FAILURE;
    }

    return UALError::UAL_SUCCESS;
}

/**
 * @brief Perform general matrix multiplication with post-operations: C =
 * alpha*A*B + beta*C + PostOps
 *
 * @param A First input matrix
 * @param B Second input matrix
 * @param C Output matrix
 * @param accType Accumulation type
 * @param postOps Post-operations to apply (err_code.get() for no post-ops)
 * @param alpha Scaling factor for A*B
 * @param beta Scaling factor for C
 * @return bool Success status
 */
UALError
UalDlp::gemm(const Matrix&                      A,
             const Matrix&                      B,
             Matrix&                            C,
             MatrixType                         accType,
             const std::shared_ptr<IOperation>& postOps,
             double                             alpha,
             double                             beta)
{

    // For now, if no postOps are provided, delegate to the original gemm method
    std::shared_ptr<dlp_metadata_t> aocl_postops;

    if (!postOps) {
        // PostOps is still needed for error code.
        // FIXME: Rename postops to metadata
        aocl_postops = std::make_shared<dlp_metadata_t>();
        std::memset(aocl_postops.get(), 0, sizeof(dlp_metadata_t));
    } else {
        // Cast to DlpOperation and access the serialized dlp_metadata_t
        // structure
        // using friend access
        auto* dlpOp = dynamic_cast<DlpOperation*>(postOps.get());
        if (!dlpOp) {
            return UALError::UAL_CAST_ERROR;
        }

        // Validate that the postOps are for DLP backend
        if (postOps->getUALType() != UALType::DLP) {
            return UALError::UAL_POSTOPS_MISMATCH;
        }

        // Get the shared dlp_metadata_t structure using friend access
        aocl_postops = dlpOp->m_postops;
    }

    uint64_t type = encode_types(A.getMatrixType(), B.getMatrixType(),
                                 C.getMatrixType(), accType);

    char transA = A.isTransposed() ? 't' : 'n';
    char transB = B.isReordered() ? 'n' : (B.isTransposed() ? 't' : 'n');

    char layoutA = A.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c';

    // Determine memory format for matrices A and B
    // Pack takes precedence over reorder since it includes optimization
    char memFormatA = A.isPacked() ? 'p' : (A.isReordered() ? 'r' : 'n');
    char memFormatB = B.isPacked() ? 'p' : (B.isReordered() ? 'r' : 'n');

    switch (type) {
        case encode_types<MatrixType::f32, MatrixType::f32, MatrixType::f32,
                          MatrixType::f32>(): {
            // For f32 operations, alpha/beta are float type
            float alpha_f32 = static_cast<float>(alpha);
            float beta_f32  = static_cast<float>(beta);

            aocl_gemm_f32f32f32of32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_f32,
                reinterpret_cast<float*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<float*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_f32,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops.get());

            break;
        }

        case encode_types<MatrixType::bf16, MatrixType::bf16, MatrixType::f32,
                          MatrixType::f32>(): {
            float alpha_f32 = static_cast<float>(alpha);
            float beta_f32  = static_cast<float>(beta);

            aocl_gemm_bf16bf16f32of32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_f32,
                reinterpret_cast<bfloat16*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<bfloat16*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_f32,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops.get());
            break;
        }

        case encode_types<MatrixType::bf16, MatrixType::bf16, MatrixType::bf16,
                          MatrixType::f32>(): {
            float alpha_f32 = static_cast<float>(alpha);
            float beta_f32  = static_cast<float>(beta);

            aocl_gemm_bf16bf16f32obf16(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_f32,
                reinterpret_cast<bfloat16*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<bfloat16*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_f32,
                reinterpret_cast<bfloat16*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops.get());
            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::s32,
                          MatrixType::s32>(): {
            // For u8/s8 operations, alpha/beta are int32_t type
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);
            aocl_gemm_u8s8s32os32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<uint8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<int32_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops.get());
            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::s32,
                          MatrixType::s32>(): {
            // For s8/s8 operations, alpha/beta are int32_t type
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32os32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<int8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<int32_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops.get());
            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::s8,
                          MatrixType::s32>(): {
            // For u8/s8 operations, alpha/beta are int32_t type
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_u8s8s32os8(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<uint8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<int8_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops.get());
            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::f32,
                          MatrixType::s32>(): {
            // For u8/s8 operations, alpha/beta are int32_t type
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_u8s8s32of32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<uint8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops.get());
            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::u8,
                          MatrixType::s32>(): {
            // uint8 × int8 → uint8 directly supported
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_u8s8s32ou8(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<uint8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<uint8_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops.get());
            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::bf16,
                          MatrixType::s32>(): {
            // uint8 × int8 → bf16 directly supported
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_u8s8s32obf16(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<uint8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<bfloat16*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops.get());
            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::f32,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32of32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<int8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops.get());
            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::s8,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32os8(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<int8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<int8_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops.get());
            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::u8,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32ou8(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<int8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<uint8_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops.get());
            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::bf16,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32obf16(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<int8_t*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<bfloat16*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops.get());
            break;
        }

        case encode_types<MatrixType::bf16, MatrixType::s8, MatrixType::bf16,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_bf16s8s32obf16(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<bfloat16*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<bfloat16*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops.get());
            break;
        }

        case encode_types<MatrixType::bf16, MatrixType::s8, MatrixType::f32,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_bf16s8s32of32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<bfloat16*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<float*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops.get());
            break;
        }

        case encode_types<MatrixType::bf16, MatrixType::s8, MatrixType::s32,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_bf16s8s32os32(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<bfloat16*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<int32_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops.get());
            break;
        }

        case encode_types<MatrixType::bf16, MatrixType::s8, MatrixType::s8,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_bf16s8s32os8(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<bfloat16*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<int8_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops.get());
            break;
        }

        case encode_types<MatrixType::bf16, MatrixType::s8, MatrixType::u8,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_bf16s8s32ou8(
                layoutA, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<bfloat16*>(A.getMatrixData().getMatrixPtr()),
                A.getLeadingDimension(), memFormatA,
                reinterpret_cast<int8_t*>(B.getMatrixData().getMatrixPtr()),
                B.getLeadingDimension(), memFormatB, beta_s32,
                reinterpret_cast<uint8_t*>(C.getMatrixData().getMatrixPtr()),
                C.getLeadingDimension(), aocl_postops.get());
            break;
        }

        default:
            return UALError::UAL_FAILURE;
    }

    if (aocl_postops->error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED)
        return UALError::UAL_NOT_SUPPORTED;

    if (aocl_postops->error_hndl.error_code == DLP_CLSC_SUCCESS)
        return UALError::UAL_SUCCESS;

    return UALError::UAL_FAILURE;
}

UALError
UalDlp::batch_gemm(std::vector<BatchGroup>& groups, MatrixType accType)
{
    PreparedBatchGemmArgs prepared;
    UALError prep_status = prepare_batch_gemm_args(groups, accType, prepared);
    if (prep_status != UALError::UAL_SUCCESS) {
        return prep_status;
    }

    // Create metadata objects for all groups (needed for error reporting)
    std::vector<std::shared_ptr<dlp_metadata_t>> metadata_storage(
        groups.size());
    std::vector<dlp_metadata_t*> metadata(groups.size());

    for (size_t i = 0; i < groups.size(); ++i) {
        if (groups[i].postOps
            && groups[i].postOps->getUALType() == UALType::DLP) {
            // Extract existing metadata from postops
            auto* dlpOp = dynamic_cast<DlpOperation*>(groups[i].postOps.get());
            if (!dlpOp) {
                return UALError::UAL_CAST_ERROR;
            }
            metadata[i] = dlpOp->getMetadata();
        } else {
            // Create new metadata for error reporting
            metadata_storage[i] = std::make_shared<dlp_metadata_t>();
            std::memset(metadata_storage[i].get(), 0, sizeof(dlp_metadata_t));
            metadata[i] = metadata_storage[i].get();
        }
    }

    uint64_t type = encode_types(prepared.a_type, prepared.b_type,
                                 prepared.c_type, accType);

    switch (type) {
        case encode_types<MatrixType::f32, MatrixType::f32, MatrixType::f32,
                          MatrixType::f32>(): {
            std::vector<float> alpha_f32(prepared.group_count);
            std::vector<float> beta_f32(prepared.group_count);
            for (size_t i = 0; i < static_cast<size_t>(prepared.group_count);
                 ++i) {
                alpha_f32[i] = static_cast<float>(prepared.alpha[i]);
                beta_f32[i]  = static_cast<float>(prepared.beta[i]);
            }

            auto a_ptrs =
                reinterpret_cast<const float* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const float* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<float* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_f32f32f32of32(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(), alpha_f32.data(),
                const_cast<const float**>(a_ptrs), prepared.lda.data(),
                const_cast<const float**>(b_ptrs), prepared.ldb.data(),
                beta_f32.data(), const_cast<float**>(c_ptrs),
                prepared.ldc.data(), prepared.group_count,
                prepared.group_size.data(), prepared.mem_format_a.data(),
                prepared.mem_format_b.data(), metadata.data());
            break;
        }

        case encode_types<MatrixType::bf16, MatrixType::bf16, MatrixType::f32,
                          MatrixType::f32>(): {
            std::vector<float> alpha_f32(prepared.group_count);
            std::vector<float> beta_f32(prepared.group_count);
            for (size_t i = 0; i < static_cast<size_t>(prepared.group_count);
                 ++i) {
                alpha_f32[i] = static_cast<float>(prepared.alpha[i]);
                beta_f32[i]  = static_cast<float>(prepared.beta[i]);
            }

            auto a_ptrs = reinterpret_cast<const bfloat16* const*>(
                prepared.a_ptrs.data());
            auto b_ptrs = reinterpret_cast<const bfloat16* const*>(
                prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<float* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_bf16bf16f32of32(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(), alpha_f32.data(),
                const_cast<const bfloat16**>(a_ptrs), prepared.lda.data(),
                const_cast<const bfloat16**>(b_ptrs), prepared.ldb.data(),
                beta_f32.data(), const_cast<float**>(c_ptrs),
                prepared.ldc.data(), prepared.group_count,
                prepared.group_size.data(), prepared.mem_format_a.data(),
                prepared.mem_format_b.data(), metadata.data());
            break;
        }

        case encode_types<MatrixType::bf16, MatrixType::bf16, MatrixType::bf16,
                          MatrixType::f32>(): {
            std::vector<float> alpha_f32(prepared.group_count);
            std::vector<float> beta_f32(prepared.group_count);
            for (size_t i = 0; i < static_cast<size_t>(prepared.group_count);
                 ++i) {
                alpha_f32[i] = static_cast<float>(prepared.alpha[i]);
                beta_f32[i]  = static_cast<float>(prepared.beta[i]);
            }

            auto a_ptrs = reinterpret_cast<const bfloat16* const*>(
                prepared.a_ptrs.data());
            auto b_ptrs = reinterpret_cast<const bfloat16* const*>(
                prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<bfloat16* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_bf16bf16f32obf16(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(), alpha_f32.data(),
                const_cast<const bfloat16**>(a_ptrs), prepared.lda.data(),
                const_cast<const bfloat16**>(b_ptrs), prepared.ldb.data(),
                beta_f32.data(), const_cast<bfloat16**>(c_ptrs),
                prepared.ldc.data(), prepared.group_count,
                prepared.group_size.data(), prepared.mem_format_a.data(),
                prepared.mem_format_b.data(), metadata.data());
            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::s32,
                          MatrixType::s32>(): {
            std::vector<int32_t> alpha_s32(prepared.group_count);
            std::vector<int32_t> beta_s32(prepared.group_count);
            for (size_t i = 0; i < static_cast<size_t>(prepared.group_count);
                 ++i) {
                alpha_s32[i] = static_cast<int32_t>(prepared.alpha[i]);
                beta_s32[i]  = static_cast<int32_t>(prepared.beta[i]);
            }

            auto a_ptrs =
                reinterpret_cast<const uint8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<int32_t* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_u8s8s32os32(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(), alpha_s32.data(),
                const_cast<const uint8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                beta_s32.data(), const_cast<int32_t**>(c_ptrs),
                prepared.ldc.data(), prepared.group_count,
                prepared.group_size.data(), prepared.mem_format_a.data(),
                prepared.mem_format_b.data(), metadata.data());
            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::s8,
                          MatrixType::s32>(): {
            std::vector<int32_t> alpha_s32(prepared.group_count);
            std::vector<int32_t> beta_s32(prepared.group_count);
            for (size_t i = 0; i < static_cast<size_t>(prepared.group_count);
                 ++i) {
                alpha_s32[i] = static_cast<int32_t>(prepared.alpha[i]);
                beta_s32[i]  = static_cast<int32_t>(prepared.beta[i]);
            }

            auto a_ptrs =
                reinterpret_cast<const uint8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<int8_t* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_u8s8s32os8(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(), alpha_s32.data(),
                const_cast<const uint8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                beta_s32.data(), const_cast<int8_t**>(c_ptrs),
                prepared.ldc.data(), prepared.group_count,
                prepared.group_size.data(), prepared.mem_format_a.data(),
                prepared.mem_format_b.data(), metadata.data());
            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::f32,
                          MatrixType::s32>(): {
            std::vector<int32_t> alpha_s32(prepared.group_count);
            std::vector<int32_t> beta_s32(prepared.group_count);
            for (size_t i = 0; i < static_cast<size_t>(prepared.group_count);
                 ++i) {
                alpha_s32[i] = static_cast<int32_t>(prepared.alpha[i]);
                beta_s32[i]  = static_cast<int32_t>(prepared.beta[i]);
            }

            auto a_ptrs =
                reinterpret_cast<const uint8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<float* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_u8s8s32of32(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(), alpha_s32.data(),
                const_cast<const uint8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                beta_s32.data(), const_cast<float**>(c_ptrs),
                prepared.ldc.data(), prepared.group_count,
                prepared.group_size.data(), prepared.mem_format_a.data(),
                prepared.mem_format_b.data(), metadata.data());
            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::bf16,
                          MatrixType::s32>(): {
            std::vector<int32_t> alpha_s32(prepared.group_count);
            std::vector<int32_t> beta_s32(prepared.group_count);
            for (size_t i = 0; i < static_cast<size_t>(prepared.group_count);
                 ++i) {
                alpha_s32[i] = static_cast<int32_t>(prepared.alpha[i]);
                beta_s32[i]  = static_cast<int32_t>(prepared.beta[i]);
            }

            auto a_ptrs =
                reinterpret_cast<const uint8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<bfloat16* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_u8s8s32obf16(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(), alpha_s32.data(),
                const_cast<const uint8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                beta_s32.data(), const_cast<bfloat16**>(c_ptrs),
                prepared.ldc.data(), prepared.group_count,
                prepared.group_size.data(), prepared.mem_format_a.data(),
                prepared.mem_format_b.data(), metadata.data());
            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::u8,
                          MatrixType::s32>(): {
            std::vector<int32_t> alpha_s32(prepared.group_count);
            std::vector<int32_t> beta_s32(prepared.group_count);
            for (size_t i = 0; i < static_cast<size_t>(prepared.group_count);
                 ++i) {
                alpha_s32[i] = static_cast<int32_t>(prepared.alpha[i]);
                beta_s32[i]  = static_cast<int32_t>(prepared.beta[i]);
            }

            auto a_ptrs =
                reinterpret_cast<const uint8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<uint8_t* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_u8s8s32ou8(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(), alpha_s32.data(),
                const_cast<const uint8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                beta_s32.data(), const_cast<uint8_t**>(c_ptrs),
                prepared.ldc.data(), prepared.group_count,
                prepared.group_size.data(), prepared.mem_format_a.data(),
                prepared.mem_format_b.data(), metadata.data());
            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::s32,
                          MatrixType::s32>(): {
            std::vector<int32_t> alpha_s32(prepared.group_count);
            std::vector<int32_t> beta_s32(prepared.group_count);
            for (size_t i = 0; i < static_cast<size_t>(prepared.group_count);
                 ++i) {
                alpha_s32[i] = static_cast<int32_t>(prepared.alpha[i]);
                beta_s32[i]  = static_cast<int32_t>(prepared.beta[i]);
            }

            auto a_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<int32_t* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_s8s8s32os32(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(), alpha_s32.data(),
                const_cast<const int8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                beta_s32.data(), const_cast<int32_t**>(c_ptrs),
                prepared.ldc.data(), prepared.group_count,
                prepared.group_size.data(), prepared.mem_format_a.data(),
                prepared.mem_format_b.data(), metadata.data());
            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::s8,
                          MatrixType::s32>(): {
            std::vector<int32_t> alpha_s32(prepared.group_count);
            std::vector<int32_t> beta_s32(prepared.group_count);
            for (size_t i = 0; i < static_cast<size_t>(prepared.group_count);
                 ++i) {
                alpha_s32[i] = static_cast<int32_t>(prepared.alpha[i]);
                beta_s32[i]  = static_cast<int32_t>(prepared.beta[i]);
            }

            auto a_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<int8_t* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_s8s8s32os8(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(), alpha_s32.data(),
                const_cast<const int8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                beta_s32.data(), const_cast<int8_t**>(c_ptrs),
                prepared.ldc.data(), prepared.group_count,
                prepared.group_size.data(), prepared.mem_format_a.data(),
                prepared.mem_format_b.data(), metadata.data());
            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::f32,
                          MatrixType::s32>(): {
            std::vector<int32_t> alpha_s32(prepared.group_count);
            std::vector<int32_t> beta_s32(prepared.group_count);
            for (size_t i = 0; i < static_cast<size_t>(prepared.group_count);
                 ++i) {
                alpha_s32[i] = static_cast<int32_t>(prepared.alpha[i]);
                beta_s32[i]  = static_cast<int32_t>(prepared.beta[i]);
            }

            auto a_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<float* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_s8s8s32of32(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(), alpha_s32.data(),
                const_cast<const int8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                beta_s32.data(), const_cast<float**>(c_ptrs),
                prepared.ldc.data(), prepared.group_count,
                prepared.group_size.data(), prepared.mem_format_a.data(),
                prepared.mem_format_b.data(), metadata.data());
            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::bf16,
                          MatrixType::s32>(): {
            std::vector<int32_t> alpha_s32(prepared.group_count);
            std::vector<int32_t> beta_s32(prepared.group_count);
            for (size_t i = 0; i < static_cast<size_t>(prepared.group_count);
                 ++i) {
                alpha_s32[i] = static_cast<int32_t>(prepared.alpha[i]);
                beta_s32[i]  = static_cast<int32_t>(prepared.beta[i]);
            }

            auto a_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<bfloat16* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_s8s8s32obf16(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(), alpha_s32.data(),
                const_cast<const int8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                beta_s32.data(), const_cast<bfloat16**>(c_ptrs),
                prepared.ldc.data(), prepared.group_count,
                prepared.group_size.data(), prepared.mem_format_a.data(),
                prepared.mem_format_b.data(), metadata.data());
            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::u8,
                          MatrixType::s32>(): {
            std::vector<int32_t> alpha_s32(prepared.group_count);
            std::vector<int32_t> beta_s32(prepared.group_count);
            for (size_t i = 0; i < static_cast<size_t>(prepared.group_count);
                 ++i) {
                alpha_s32[i] = static_cast<int32_t>(prepared.alpha[i]);
                beta_s32[i]  = static_cast<int32_t>(prepared.beta[i]);
            }

            auto a_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<uint8_t* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_s8s8s32ou8(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(), alpha_s32.data(),
                const_cast<const int8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                beta_s32.data(), const_cast<uint8_t**>(c_ptrs),
                prepared.ldc.data(), prepared.group_count,
                prepared.group_size.data(), prepared.mem_format_a.data(),
                prepared.mem_format_b.data(), metadata.data());
            break;
        }

        default:
            return UALError::UAL_NOT_SUPPORTED;
    }

    // Check metadata for errors (e.g., ISA not supported)
    for (size_t i = 0; i < metadata.size(); ++i) {
        if (metadata[i] != nullptr) {
            if (metadata[i]->error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED) {
                return UALError::UAL_NOT_SUPPORTED;
            }
            if (metadata[i]->error_hndl.error_code != DLP_CLSC_SUCCESS) {
                return UALError::UAL_FAILURE;
            }
        }
    }

    return UALError::UAL_SUCCESS;
}

void
UalDlp::batch_prepare_metadata(PreparedBatchGemmArgs& args)
{
    args.backend_metadata.resize(args.group_count);
    args.backend_metadata_storage.reserve(args.group_count);

    for (size_t i = 0; i < static_cast<size_t>(args.group_count); ++i) {
        std::shared_ptr<IOperation> postOp = nullptr;
        if (i < args.post_ops.size()) {
            postOp = args.post_ops[i];
        }

        if (postOp && postOp->getUALType() == UALType::DLP) {
            auto* dlpOp = dynamic_cast<DlpOperation*>(postOp.get());
            if (dlpOp) {
                args.backend_metadata[i] = dlpOp->getMetadata();
            } else {
                auto meta = std::make_shared<dlp_metadata_t>();
                std::memset(meta.get(), 0, sizeof(dlp_metadata_t));
                args.backend_metadata[i] = meta.get();
                args.backend_metadata_storage.push_back(
                    std::static_pointer_cast<void>(meta));
            }
        } else {
            auto meta = std::make_shared<dlp_metadata_t>();
            std::memset(meta.get(), 0, sizeof(dlp_metadata_t));
            args.backend_metadata[i] = meta.get();
            args.backend_metadata_storage.push_back(
                std::static_pointer_cast<void>(meta));
        }
    }

    // Precompute alpha/beta in correct types based on matrix types
    // This eliminates per-iteration allocation in the hot loop
    uint64_t type =
        encode_types(args.a_type, args.b_type, args.c_type, args.acc_type);

    // Check if we need float or int32 alpha/beta
    bool needs_f32 =
        (args.a_type == MatrixType::f32 || args.a_type == MatrixType::bf16);
    bool needs_s32 =
        (args.a_type == MatrixType::u8 || args.a_type == MatrixType::s8);

    if (needs_f32) {
        args.alpha_f32.resize(args.group_count);
        args.beta_f32.resize(args.group_count);
        for (size_t i = 0; i < static_cast<size_t>(args.group_count); ++i) {
            args.alpha_f32[i] = static_cast<float>(args.alpha[i]);
            args.beta_f32[i]  = static_cast<float>(args.beta[i]);
        }
    }

    if (needs_s32) {
        args.alpha_s32.resize(args.group_count);
        args.beta_s32.resize(args.group_count);
        for (size_t i = 0; i < static_cast<size_t>(args.group_count); ++i) {
            args.alpha_s32[i] = static_cast<int32_t>(args.alpha[i]);
            args.beta_s32[i]  = static_cast<int32_t>(args.beta[i]);
        }
    }
}

UALError
UalDlp::batch_gemm(const PreparedBatchGemmArgs& prepared)
{
    // NOTE: No validation - assumes batch_prepare_metadata() was called
    // successfully. This is intentional for performance: validation should
    // happen once at setup, not on every hot-path call.

    // Pre-computed metadata and alpha/beta - ZERO allocations in this function!
    dlp_metadata_t** metadata = reinterpret_cast<dlp_metadata_t**>(
        const_cast<void**>(prepared.backend_metadata.data()));

    uint64_t type = encode_types(prepared.a_type, prepared.b_type,
                                 prepared.c_type, prepared.acc_type);

    switch (type) {
        case encode_types<MatrixType::f32, MatrixType::f32, MatrixType::f32,
                          MatrixType::f32>(): {
            // Use precomputed alpha/beta - NO allocation!
            auto a_ptrs =
                reinterpret_cast<const float* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const float* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<float* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_f32f32f32of32(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(),
                const_cast<float*>(prepared.alpha_f32.data()),
                const_cast<const float**>(a_ptrs), prepared.lda.data(),
                const_cast<const float**>(b_ptrs), prepared.ldb.data(),
                const_cast<float*>(prepared.beta_f32.data()),
                const_cast<float**>(c_ptrs), prepared.ldc.data(),
                prepared.group_count, prepared.group_size.data(),
                prepared.mem_format_a.data(), prepared.mem_format_b.data(),
                metadata);
            break;
        }

        case encode_types<MatrixType::bf16, MatrixType::bf16, MatrixType::f32,
                          MatrixType::f32>(): {
            // Use precomputed alpha/beta - NO allocation!
            auto a_ptrs = reinterpret_cast<const bfloat16* const*>(
                prepared.a_ptrs.data());
            auto b_ptrs = reinterpret_cast<const bfloat16* const*>(
                prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<float* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_bf16bf16f32of32(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(),
                const_cast<float*>(prepared.alpha_f32.data()),
                const_cast<const bfloat16**>(a_ptrs), prepared.lda.data(),
                const_cast<const bfloat16**>(b_ptrs), prepared.ldb.data(),
                const_cast<float*>(prepared.beta_f32.data()),
                const_cast<float**>(c_ptrs), prepared.ldc.data(),
                prepared.group_count, prepared.group_size.data(),
                prepared.mem_format_a.data(), prepared.mem_format_b.data(),
                metadata);
            break;
        }

        case encode_types<MatrixType::bf16, MatrixType::bf16, MatrixType::bf16,
                          MatrixType::f32>(): {
            // Use precomputed alpha/beta - NO allocation!
            auto a_ptrs = reinterpret_cast<const bfloat16* const*>(
                prepared.a_ptrs.data());
            auto b_ptrs = reinterpret_cast<const bfloat16* const*>(
                prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<bfloat16* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_bf16bf16f32obf16(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(),
                const_cast<float*>(prepared.alpha_f32.data()),
                const_cast<const bfloat16**>(a_ptrs), prepared.lda.data(),
                const_cast<const bfloat16**>(b_ptrs), prepared.ldb.data(),
                const_cast<float*>(prepared.beta_f32.data()),
                const_cast<bfloat16**>(c_ptrs), prepared.ldc.data(),
                prepared.group_count, prepared.group_size.data(),
                prepared.mem_format_a.data(), prepared.mem_format_b.data(),
                metadata);
            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::s32,
                          MatrixType::s32>(): {
            // Use precomputed alpha/beta - NO allocation!
            auto a_ptrs =
                reinterpret_cast<const uint8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<int32_t* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_u8s8s32os32(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(),
                const_cast<int32_t*>(prepared.alpha_s32.data()),
                const_cast<const uint8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                const_cast<int32_t*>(prepared.beta_s32.data()),
                const_cast<int32_t**>(c_ptrs), prepared.ldc.data(),
                prepared.group_count, prepared.group_size.data(),
                prepared.mem_format_a.data(), prepared.mem_format_b.data(),
                metadata);
            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::s8,
                          MatrixType::s32>(): {
            // Use precomputed alpha/beta - NO allocation!
            auto a_ptrs =
                reinterpret_cast<const uint8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<int8_t* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_u8s8s32os8(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(),
                const_cast<int32_t*>(prepared.alpha_s32.data()),
                const_cast<const uint8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                const_cast<int32_t*>(prepared.beta_s32.data()),
                const_cast<int8_t**>(c_ptrs), prepared.ldc.data(),
                prepared.group_count, prepared.group_size.data(),
                prepared.mem_format_a.data(), prepared.mem_format_b.data(),
                metadata);
            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::f32,
                          MatrixType::s32>(): {
            // Use precomputed alpha/beta - NO allocation!
            auto a_ptrs =
                reinterpret_cast<const uint8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<float* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_u8s8s32of32(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(),
                const_cast<int32_t*>(prepared.alpha_s32.data()),
                const_cast<const uint8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                const_cast<int32_t*>(prepared.beta_s32.data()),
                const_cast<float**>(c_ptrs), prepared.ldc.data(),
                prepared.group_count, prepared.group_size.data(),
                prepared.mem_format_a.data(), prepared.mem_format_b.data(),
                metadata);
            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::bf16,
                          MatrixType::s32>(): {
            // Use precomputed alpha/beta - NO allocation!
            auto a_ptrs =
                reinterpret_cast<const uint8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<bfloat16* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_u8s8s32obf16(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(),
                const_cast<int32_t*>(prepared.alpha_s32.data()),
                const_cast<const uint8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                const_cast<int32_t*>(prepared.beta_s32.data()),
                const_cast<bfloat16**>(c_ptrs), prepared.ldc.data(),
                prepared.group_count, prepared.group_size.data(),
                prepared.mem_format_a.data(), prepared.mem_format_b.data(),
                metadata);
            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::u8,
                          MatrixType::s32>(): {
            // Use precomputed alpha/beta - NO allocation!
            auto a_ptrs =
                reinterpret_cast<const uint8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<uint8_t* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_u8s8s32ou8(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(),
                const_cast<int32_t*>(prepared.alpha_s32.data()),
                const_cast<const uint8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                const_cast<int32_t*>(prepared.beta_s32.data()),
                const_cast<uint8_t**>(c_ptrs), prepared.ldc.data(),
                prepared.group_count, prepared.group_size.data(),
                prepared.mem_format_a.data(), prepared.mem_format_b.data(),
                metadata);
            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::s32,
                          MatrixType::s32>(): {
            // Use precomputed alpha/beta - NO allocation!
            auto a_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<int32_t* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_s8s8s32os32(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(),
                const_cast<int32_t*>(prepared.alpha_s32.data()),
                const_cast<const int8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                const_cast<int32_t*>(prepared.beta_s32.data()),
                const_cast<int32_t**>(c_ptrs), prepared.ldc.data(),
                prepared.group_count, prepared.group_size.data(),
                prepared.mem_format_a.data(), prepared.mem_format_b.data(),
                metadata);
            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::s8,
                          MatrixType::s32>(): {
            // Use precomputed alpha/beta - NO allocation!
            auto a_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<int8_t* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_s8s8s32os8(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(),
                const_cast<int32_t*>(prepared.alpha_s32.data()),
                const_cast<const int8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                const_cast<int32_t*>(prepared.beta_s32.data()),
                const_cast<int8_t**>(c_ptrs), prepared.ldc.data(),
                prepared.group_count, prepared.group_size.data(),
                prepared.mem_format_a.data(), prepared.mem_format_b.data(),
                metadata);
            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::f32,
                          MatrixType::s32>(): {
            // Use precomputed alpha/beta - NO allocation!
            auto a_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<float* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_s8s8s32of32(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(),
                const_cast<int32_t*>(prepared.alpha_s32.data()),
                const_cast<const int8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                const_cast<int32_t*>(prepared.beta_s32.data()),
                const_cast<float**>(c_ptrs), prepared.ldc.data(),
                prepared.group_count, prepared.group_size.data(),
                prepared.mem_format_a.data(), prepared.mem_format_b.data(),
                metadata);
            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::bf16,
                          MatrixType::s32>(): {
            // Use precomputed alpha/beta - NO allocation!
            auto a_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<bfloat16* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_s8s8s32obf16(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(),
                const_cast<int32_t*>(prepared.alpha_s32.data()),
                const_cast<const int8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                const_cast<int32_t*>(prepared.beta_s32.data()),
                const_cast<bfloat16**>(c_ptrs), prepared.ldc.data(),
                prepared.group_count, prepared.group_size.data(),
                prepared.mem_format_a.data(), prepared.mem_format_b.data(),
                metadata);
            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::u8,
                          MatrixType::s32>(): {
            // Use precomputed alpha/beta - NO allocation!
            auto a_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.a_ptrs.data());
            auto b_ptrs =
                reinterpret_cast<const int8_t* const*>(prepared.b_ptrs.data());
            auto c_ptrs =
                reinterpret_cast<uint8_t* const*>(prepared.c_ptrs.data());

            aocl_batch_gemm_s8s8s32ou8(
                prepared.order.data(), prepared.transa.data(),
                prepared.transb.data(), prepared.m.data(), prepared.n.data(),
                prepared.k.data(),
                const_cast<int32_t*>(prepared.alpha_s32.data()),
                const_cast<const int8_t**>(a_ptrs), prepared.lda.data(),
                const_cast<const int8_t**>(b_ptrs), prepared.ldb.data(),
                const_cast<int32_t*>(prepared.beta_s32.data()),
                const_cast<uint8_t**>(c_ptrs), prepared.ldc.data(),
                prepared.group_count, prepared.group_size.data(),
                prepared.mem_format_a.data(), prepared.mem_format_b.data(),
                metadata);
            break;
        }

        default:
            return UALError::UAL_NOT_SUPPORTED;
    }

    // Error checking
    for (size_t i = 0; i < static_cast<size_t>(prepared.group_count); ++i) {
        if (metadata[i]->error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED) {
            return UALError::UAL_NOT_SUPPORTED;
        }
        if (metadata[i]->error_hndl.error_code != DLP_CLSC_SUCCESS) {
            return UALError::UAL_FAILURE;
        }
    }
    return UALError::UAL_SUCCESS;
}

UALError
UalDlp::gemm(md_t         m,
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
             double       alpha,
             double       beta) const
{
    // For now, if no postOps are provided, delegate to the original gemm method
    std::unique_ptr<dlp_metadata_t> err_code =
        std::make_unique<dlp_metadata_t>();

    uint64_t type = encode_types(matA_type, matB_type, matC_type, accType);

    char transA  = matA_transposed ? 't' : 'n';
    char transB  = matB_transposed ? 't' : 'n';
    char layoutA = matA_layout == MatrixLayout::ROW_MAJOR ? 'r' : 'c';

    switch (type) {
        case encode_types<MatrixType::f32, MatrixType::f32, MatrixType::f32,
                          MatrixType::f32>(): {
            // For f32 operations, alpha/beta are float type
            float alpha_f32 = static_cast<float>(alpha);
            float beta_f32  = static_cast<float>(beta);

            aocl_gemm_f32f32f32of32(
                layoutA, transA, transB, m, n, k, alpha_f32,
                reinterpret_cast<float*>(matA), matA_leadingDim, memFormatA,
                reinterpret_cast<float*>(matB), matB_leadingDim, memFormatB,
                beta_f32, reinterpret_cast<float*>(matC), matC_leadingDim,
                err_code.get());

            break;
        }

        case encode_types<MatrixType::bf16, MatrixType::bf16, MatrixType::f32,
                          MatrixType::f32>(): {
            float alpha_f32 = static_cast<float>(alpha);
            float beta_f32  = static_cast<float>(beta);

            aocl_gemm_bf16bf16f32of32(
                layoutA, transA, transB, m, n, k, alpha_f32,
                reinterpret_cast<bfloat16*>(matA), matA_leadingDim, memFormatA,
                reinterpret_cast<bfloat16*>(matB), matB_leadingDim, memFormatB,
                beta_f32, reinterpret_cast<float*>(matC), matC_leadingDim,
                err_code.get());

            break;
        }

        case encode_types<MatrixType::bf16, MatrixType::bf16, MatrixType::bf16,
                          MatrixType::f32>(): {
            float alpha_f32 = static_cast<float>(alpha);
            float beta_f32  = static_cast<float>(beta);

            aocl_gemm_bf16bf16f32obf16(
                layoutA, transA, transB, m, n, k, alpha_f32,
                reinterpret_cast<bfloat16*>(matA), matA_leadingDim, memFormatA,
                reinterpret_cast<bfloat16*>(matB), matB_leadingDim, memFormatB,
                beta_f32, reinterpret_cast<bfloat16*>(matC), matC_leadingDim,
                err_code.get());

            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::s32,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_u8s8s32os32(
                layoutA, transA, transB, m, n, k, alpha_s32,
                reinterpret_cast<uint8_t*>(matA), matA_leadingDim, memFormatA,
                reinterpret_cast<int8_t*>(matB), matB_leadingDim, memFormatB,
                beta_s32, reinterpret_cast<int32_t*>(matC), matC_leadingDim,
                err_code.get());

            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::s8,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_u8s8s32os8(
                layoutA, transA, transB, m, n, k, alpha_s32,
                reinterpret_cast<uint8_t*>(matA), matA_leadingDim, memFormatA,
                reinterpret_cast<int8_t*>(matB), matB_leadingDim, memFormatB,
                beta_s32, reinterpret_cast<int8_t*>(matC), matC_leadingDim,
                err_code.get());

            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::u8,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_u8s8s32ou8(
                layoutA, transA, transB, m, n, k, alpha_s32,
                reinterpret_cast<uint8_t*>(matA), matA_leadingDim, memFormatA,
                reinterpret_cast<int8_t*>(matB), matB_leadingDim, memFormatB,
                beta_s32, reinterpret_cast<uint8_t*>(matC), matC_leadingDim,
                err_code.get());

            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::f32,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_u8s8s32of32(
                layoutA, transA, transB, m, n, k, alpha_s32,
                reinterpret_cast<uint8_t*>(matA), matA_leadingDim, memFormatA,
                reinterpret_cast<int8_t*>(matB), matB_leadingDim, memFormatB,
                beta_s32, reinterpret_cast<float*>(matC), matC_leadingDim,
                err_code.get());

            break;
        }

        case encode_types<MatrixType::u8, MatrixType::s8, MatrixType::bf16,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_u8s8s32obf16(
                layoutA, transA, transB, m, n, k, alpha_s32,
                reinterpret_cast<uint8_t*>(matA), matA_leadingDim, memFormatA,
                reinterpret_cast<int8_t*>(matB), matB_leadingDim, memFormatB,
                beta_s32, reinterpret_cast<bfloat16*>(matC), matC_leadingDim,
                err_code.get());

            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::s32,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32os32(
                layoutA, transA, transB, m, n, k, alpha_s32,
                reinterpret_cast<int8_t*>(matA), matA_leadingDim, memFormatA,
                reinterpret_cast<int8_t*>(matB), matB_leadingDim, memFormatB,
                beta_s32, reinterpret_cast<int32_t*>(matC), matC_leadingDim,
                err_code.get());

            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::s8,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32os8(
                layoutA, transA, transB, m, n, k, alpha_s32,
                reinterpret_cast<int8_t*>(matA), matA_leadingDim, memFormatA,
                reinterpret_cast<int8_t*>(matB), matB_leadingDim, memFormatB,
                beta_s32, reinterpret_cast<int8_t*>(matC), matC_leadingDim,
                err_code.get());

            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::u8,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32ou8(
                layoutA, transA, transB, m, n, k, alpha_s32,
                reinterpret_cast<int8_t*>(matA), matA_leadingDim, memFormatA,
                reinterpret_cast<int8_t*>(matB), matB_leadingDim, memFormatB,
                beta_s32, reinterpret_cast<uint8_t*>(matC), matC_leadingDim,
                err_code.get());

            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::f32,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32of32(
                layoutA, transA, transB, m, n, k, alpha_s32,
                reinterpret_cast<int8_t*>(matA), matA_leadingDim, memFormatA,
                reinterpret_cast<int8_t*>(matB), matB_leadingDim, memFormatB,
                beta_s32, reinterpret_cast<float*>(matC), matC_leadingDim,
                err_code.get());

            break;
        }

        case encode_types<MatrixType::s8, MatrixType::s8, MatrixType::bf16,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_s8s8s32obf16(
                layoutA, transA, transB, m, n, k, alpha_s32,
                reinterpret_cast<int8_t*>(matA), matA_leadingDim, memFormatA,
                reinterpret_cast<int8_t*>(matB), matB_leadingDim, memFormatB,
                beta_s32, reinterpret_cast<bfloat16*>(matC), matC_leadingDim,
                err_code.get());

            break;
        }
        case encode_types<MatrixType::bf16, MatrixType::s8, MatrixType::bf16,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_bf16s8s32obf16(
                layoutA, transA, transB, m, n, k, alpha_s32,
                reinterpret_cast<bfloat16*>(matA), matA_leadingDim, memFormatA,
                reinterpret_cast<int8_t*>(matB), matB_leadingDim, memFormatB,
                beta_s32, reinterpret_cast<bfloat16*>(matC), matC_leadingDim,
                err_code.get());

            break;
        }
        case encode_types<MatrixType::bf16, MatrixType::s8, MatrixType::f32,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_bf16s8s32of32(
                layoutA, transA, transB, m, n, k, alpha_s32,
                reinterpret_cast<bfloat16*>(matA), matA_leadingDim, memFormatA,
                reinterpret_cast<int8_t*>(matB), matB_leadingDim, memFormatB,
                beta_s32, reinterpret_cast<float*>(matC), matC_leadingDim,
                err_code.get());
            break;
        }
        case encode_types<MatrixType::bf16, MatrixType::s8, MatrixType::s32,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_bf16s8s32os32(
                layoutA, transA, transB, m, n, k, alpha_s32,
                reinterpret_cast<bfloat16*>(matA), matA_leadingDim, memFormatA,
                reinterpret_cast<int8_t*>(matB), matB_leadingDim, memFormatB,
                beta_s32, reinterpret_cast<int32_t*>(matC), matC_leadingDim,
                err_code.get());
            break;
        }
        case encode_types<MatrixType::bf16, MatrixType::s8, MatrixType::s8,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_bf16s8s32os8(
                layoutA, transA, transB, m, n, k, alpha_s32,
                reinterpret_cast<bfloat16*>(matA), matA_leadingDim, memFormatA,
                reinterpret_cast<int8_t*>(matB), matB_leadingDim, memFormatB,
                beta_s32, reinterpret_cast<int8_t*>(matC), matC_leadingDim,
                err_code.get());
            break;
        }
        case encode_types<MatrixType::bf16, MatrixType::s8, MatrixType::u8,
                          MatrixType::s32>(): {
            int32_t alpha_s32 = static_cast<int32_t>(alpha);
            int32_t beta_s32  = static_cast<int32_t>(beta);

            aocl_gemm_bf16s8s32ou8(
                layoutA, transA, transB, m, n, k, alpha_s32,
                reinterpret_cast<bfloat16*>(matA), matA_leadingDim, memFormatA,
                reinterpret_cast<int8_t*>(matB), matB_leadingDim, memFormatB,
                beta_s32, reinterpret_cast<uint8_t*>(matC), matC_leadingDim,
                err_code.get());
            break;
        }
        default:
            return UALError::UAL_FAILURE;
    }

    if (err_code->error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED)
        return UALError::UAL_NOT_SUPPORTED;

    return err_code->error_hndl.error_code == DLP_CLSC_SUCCESS
               ? UALError::UAL_SUCCESS
               : UALError::UAL_FAILURE;
}

} // namespace dlp::testing::classic
