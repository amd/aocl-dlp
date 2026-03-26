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
#include "adaptors/dlp/ual_plan_dlp.hh"
#include "framework/batch_gemm_args.hh"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include "aocl_dlp.h" // IWYU pragma: keep
#include "classic/aocl_fp16_type.h"
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

std::unique_ptr<dlp::testing::framework::IUalPlan>
UalDlp::createPlan()
{
    return std::make_unique<DlpUalPlan>();
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
    } else if (in.getMatrixType() == MatrixType::fp16) {
        // For fp16, use the f16f16f16of16 reorder function
        alloc_bytes = aocl_get_reorder_buf_size_f16f16f16of16(
            in.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c',
            in.isTransposed() ? 't' : 'n', 'B', effective_rows, effective_cols,
            &meta);
    } else if (in.getMatrixType() == MatrixType::s4) {
        // For s4, bf16s4 with f32 accumulation
        // Note: s4 is used with bf16 A matrix and f32 accumulation
        alloc_bytes = aocl_get_reorder_buf_size_bf16s4f32of32(
            in.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c',
            in.isTransposed() ? 't' : 'n', 'B', effective_rows, effective_cols,
            &meta);
    } else if (in.getMatrixType() == MatrixType::u4) {
        // For u4, bf16u4 with f32 accumulation (same buffer layout as bf16s4)
        alloc_bytes = aocl_get_reorder_buf_size_bf16s4f32of32(
            in.getLayout() == MatrixLayout::ROW_MAJOR ? 'r' : 'c',
            in.isTransposed() ? 't' : 'n', 'B', effective_rows, effective_cols,
            &meta);
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
        case MatrixType::fp16:
            aocl_reorder_f16f16f16of16(
                layout, in.isTransposed() ? 't' : 'n', 'B',
                reinterpret_cast<const float16*>(
                    in.getMatrixData().getMatrixPtr()),
                reinterpret_cast<float16*>(out.getMatrixData().getMatrixPtr()),
                effective_rows, effective_cols, in.getLeadingDimension(),
                &meta);
            break;
        case MatrixType::s4:
            // For s4, use bf16s4 reorder function
            aocl_reorder_bf16s4f32of32(
                layout, in.isTransposed() ? 't' : 'n', 'B',
                reinterpret_cast<const int8_t*>(
                    in.getMatrixData().getMatrixPtr()),
                reinterpret_cast<int8_t*>(out.getMatrixData().getMatrixPtr()),
                effective_rows, effective_cols, in.getLeadingDimension(),
                &meta);
            break;
        case MatrixType::u4:
            // For u4, same packed layout as s4; reuse bf16s4 reorder (uint8_t
            // and int8_t have same bit layout for 4-bit nibbles)
            aocl_reorder_bf16s4f32of32(
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

UALError
UalDlp::batch_gemm(std::vector<BatchGroup>& groups, MatrixType accType)
{
    PreparedBatchGemmArgs prepared;
    UALError prep_status = prepare_batch_gemm_args(groups, accType, prepared);
    if (prep_status != UALError::UAL_SUCCESS) {
        return prep_status;
    }

    // Build metadata for each group using DlpUalPlan for post-ops
    std::vector<std::unique_ptr<DlpUalPlan>>     plan_storage(groups.size());
    std::vector<std::shared_ptr<dlp_metadata_t>> metadata_storage(
        groups.size());
    std::vector<dlp_metadata_t*> metadata(groups.size());

    for (std::size_t i = 0; i < groups.size(); ++i) {
        bool hasOps = !groups[i].post_op_params.empty() || groups[i].a_quant;
        if (hasOps) {
            // Build metadata via DlpUalPlan
            auto plan = std::make_unique<DlpUalPlan>();
            for (const auto& p : groups[i].post_op_params) {
                if (p) {
                    plan->addPostOp(p->clone());
                }
            }
            if (groups[i].a_quant) {
                plan->setAQuant(
                    std::make_unique<AQuantParam>(*groups[i].a_quant));
            }
            // Configure minimal dimensions for prepare()
            plan->setDimensions(groups[i].m, groups[i].n, groups[i].k);
            plan->setTypes(groups[i].A_matrices[0].getMatrixType(),
                           groups[i].B_matrices[0].getMatrixType(),
                           groups[i].C_matrices[0].getMatrixType(), accType);
            plan->setAlpha(groups[i].alpha);
            plan->setBeta(groups[i].beta);
            plan->prepare();
            metadata[i]     = plan->getMetadata();
            plan_storage[i] = std::move(plan);
        } else {
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
            for (std::size_t i = 0;
                 i < static_cast<size_t>(prepared.group_count); ++i) {
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
            for (std::size_t i = 0;
                 i < static_cast<size_t>(prepared.group_count); ++i) {
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
            for (std::size_t i = 0;
                 i < static_cast<size_t>(prepared.group_count); ++i) {
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
            for (std::size_t i = 0;
                 i < static_cast<size_t>(prepared.group_count); ++i) {
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
            for (std::size_t i = 0;
                 i < static_cast<size_t>(prepared.group_count); ++i) {
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
            for (std::size_t i = 0;
                 i < static_cast<size_t>(prepared.group_count); ++i) {
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
            for (std::size_t i = 0;
                 i < static_cast<size_t>(prepared.group_count); ++i) {
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
            for (std::size_t i = 0;
                 i < static_cast<size_t>(prepared.group_count); ++i) {
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
            for (std::size_t i = 0;
                 i < static_cast<size_t>(prepared.group_count); ++i) {
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
            for (std::size_t i = 0;
                 i < static_cast<size_t>(prepared.group_count); ++i) {
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
            for (std::size_t i = 0;
                 i < static_cast<size_t>(prepared.group_count); ++i) {
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
            for (std::size_t i = 0;
                 i < static_cast<size_t>(prepared.group_count); ++i) {
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
            for (std::size_t i = 0;
                 i < static_cast<size_t>(prepared.group_count); ++i) {
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
    for (std::size_t i = 0; i < metadata.size(); ++i) {
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

    for (std::size_t i = 0; i < static_cast<size_t>(args.group_count); ++i) {
        bool hasPostOps = i < args.post_ops.size() && !args.post_ops[i].empty();

        if (hasPostOps) {
            // Build metadata via DlpUalPlan for post-ops
            auto plan = std::make_shared<DlpUalPlan>();
            for (const auto& p : args.post_ops[i]) {
                if (p) {
                    plan->addPostOp(p->clone());
                }
            }
            plan->setTypes(args.a_type, args.b_type, args.c_type,
                           args.acc_type);
            plan->setDimensions(args.m[i], args.n[i], args.k[i]);
            plan->prepare();
            args.backend_metadata[i] = plan->getMetadata();
            args.backend_metadata_storage.push_back(
                std::static_pointer_cast<void>(plan));
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
        for (std::size_t i = 0; i < static_cast<size_t>(args.group_count);
             ++i) {
            args.alpha_f32[i] = static_cast<float>(args.alpha[i]);
            args.beta_f32[i]  = static_cast<float>(args.beta[i]);
        }
    }

    if (needs_s32) {
        args.alpha_s32.resize(args.group_count);
        args.beta_s32.resize(args.group_count);
        for (std::size_t i = 0; i < static_cast<size_t>(args.group_count);
             ++i) {
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
    for (std::size_t i = 0; i < static_cast<size_t>(prepared.group_count);
         ++i) {
        if (metadata[i]->error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED) {
            return UALError::UAL_NOT_SUPPORTED;
        }
        if (metadata[i]->error_hndl.error_code != DLP_CLSC_SUCCESS) {
            return UALError::UAL_FAILURE;
        }
    }
    return UALError::UAL_SUCCESS;
}

} // namespace dlp::testing::classic
