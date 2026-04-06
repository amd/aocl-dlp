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

#include "adaptors/ref/ual_plan_ref.hh"
#include "adaptors/ref/gemm_ref.hh"
#include "adaptors/ref/ual_ref.hh"
#include "utils/conversion_utils.hh"
#include "utils/matrix_conversion_utils.hh"
#include <cassert>
#include <cstring>
#include <memory>

using namespace dlp::testing::framework;
using dlp::testing::utils::isIntegerType;
using dlp::testing::utils::truncateF32ToMicro;

namespace dlp::testing::classic {

void
RefUalPlan::prepare()
{
    m_prepared = true;
}

UALError
RefUalPlan::execute()
{
    if (!m_prepared || !m_buffers_set) {
        return UALError::UAL_FAILURE;
    }
    if (!m_a_matrix || !m_b_matrix || !m_c_matrix) {
        return UALError::UAL_FAILURE;
    }

    const Matrix& A = *m_a_matrix;
    const Matrix& B = *m_b_matrix;
    Matrix&       C = *m_c_matrix;

    UalRef ualRef;

    MatrixType aType      = A.getMatrixType();
    MatrixType bType      = B.getMatrixType();
    MatrixType outputType = C.getMatrixType();

    bool hasPostOps = !m_post_ops.empty();

    // Handle WOQ path (bf16×s4, bf16×u4)
    bool isBf16S4Gemm = (aType == MatrixType::bf16 && bType == MatrixType::s4);
    bool isBf16U4Gemm = (aType == MatrixType::bf16 && bType == MatrixType::u4);

    if ((isBf16S4Gemm || isBf16U4Gemm) && m_woq) {
        if (!ualRef.checkValidGemmParams(A, B, C, true)) {
            return UALError::UAL_FAILURE;
        }

        Matrix b_scale_factor;
        Matrix b_zero_point;
        if (m_woq->hasB_ScaleFactor()) {
            b_scale_factor = *m_woq->getB_ScaleFactor();
        }
        if (m_woq->hasB_ZeroPoint()) {
            b_zero_point = *m_woq->getB_ZeroPoint();
        }

        void* b_scale_data = b_scale_factor.getData();
        void* b_zp_data    = b_zero_point.getData();
        if (!b_scale_data) {
            return UALError::UAL_FAILURE;
        }

        md_t       sf_len = b_scale_factor.getRows() * b_scale_factor.getCols();
        md_t       zp_len = b_zero_point.getRows() * b_zero_point.getCols();
        MatrixType sf_type = b_scale_factor.getMatrixType();
        MatrixType zp_type = b_zero_point.getMatrixType();

        char  layout = (A.getLayout() == MatrixLayout::ROW_MAJOR) ? 'R' : 'C';
        char  transA = A.isTransposed() ? 'T' : 'N';
        char  transB = B.isTransposed() ? 'T' : 'N';
        float alpha_f32 = static_cast<float>(m_alpha);
        float beta_f32  = static_cast<float>(m_beta);

        // Compute into f32 intermediate, apply post-ops, convert to target
        Matrix tempC_f32(C.getEffectiveRows(), C.getEffectiveCols(),
                         MatrixType::f32, C.getLayout());
        if (beta_f32 != 0.0f) {
            dlp::testing::utils::copyMatrixTo<float>(
                C, reinterpret_cast<float*>(tempC_f32.getData()),
                tempC_f32.getLeadingDimension(), tempC_f32.getLayout());
        } else {
            std::memset(tempC_f32.getData(), 0, tempC_f32.getDataSizeBytes());
        }

        if (isBf16S4Gemm) {
            if (b_zp_data != nullptr) {
                return UALError::UAL_FAILURE;
            }
            dlp::testing::classic::ref::aocl_gemm_bf16s4f32of32_ref(
                layout, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_f32,
                reinterpret_cast<const bfloat16*>(
                    A.getMatrixData().getMatrixPtr()),
                static_cast<int>(A.getLeadingDimension()),
                reinterpret_cast<const int8_t*>(
                    B.getMatrixData().getMatrixPtr()),
                static_cast<int>(B.getLeadingDimension()), beta_f32,
                reinterpret_cast<float*>(tempC_f32.getData()),
                static_cast<int>(tempC_f32.getLeadingDimension()), b_scale_data,
                sf_len, sf_type, B.isReordered());
        } else {
            if (b_zp_data == nullptr) {
                return UALError::UAL_FAILURE;
            }
            dlp::testing::classic::ref::aocl_gemm_bf16u4f32of32_ref(
                layout, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_f32,
                reinterpret_cast<const bfloat16*>(
                    A.getMatrixData().getMatrixPtr()),
                static_cast<int>(A.getLeadingDimension()),
                reinterpret_cast<const uint8_t*>(
                    B.getMatrixData().getMatrixPtr()),
                static_cast<int>(B.getLeadingDimension()), beta_f32,
                reinterpret_cast<float*>(tempC_f32.getData()),
                static_cast<int>(tempC_f32.getLeadingDimension()), b_scale_data,
                b_zp_data, sf_len, zp_len, sf_type, zp_type, B.isReordered());
        }

        applyPostOps(tempC_f32);
        if (hasPostOps && isIntegerType(outputType)) {
            truncateF32ToMicro(reinterpret_cast<float*>(tempC_f32.getData()),
                               static_cast<size_t>(tempC_f32.getRows())
                                   * tempC_f32.getCols());
        }
        dlp::testing::utils::copyToMatrix<float>(
            reinterpret_cast<const float*>(tempC_f32.getData()),
            tempC_f32.getLeadingDimension(), C, tempC_f32.getLayout());
        return UALError::UAL_SUCCESS;
    }

    // Handle AQuant path (bf16×s8, f32×s8 with quantisation)
    bool isBf16S8QuantGemm =
        (aType == MatrixType::bf16 && bType == MatrixType::s8);
    bool isF32S8QuantGemm =
        (aType == MatrixType::f32 && bType == MatrixType::s8);

    if ((isBf16S8QuantGemm || isF32S8QuantGemm) && m_a_quant) {
        if (!ualRef.checkValidGemmParams(A, B, C, true)) {
            return UALError::UAL_FAILURE;
        }

        Matrix a_pre_sf, a_pre_zp, a_post_sf, a_post_zp;
        if (m_a_quant->hasA_PreOpScaleFactor())
            a_pre_sf = *m_a_quant->getA_PreOpScaleFactor();
        if (m_a_quant->hasA_PreOpZeroPoint())
            a_pre_zp = *m_a_quant->getA_PreOpZeroPoint();
        if (m_a_quant->hasA_PostOpScaleFactor())
            a_post_sf = *m_a_quant->getA_PostOpScaleFactor();
        if (m_a_quant->hasA_PostOpZeroPoint())
            a_post_zp = *m_a_quant->getA_PostOpZeroPoint();

        void* a_pre_sf_data  = a_pre_sf.getData();
        void* a_pre_zp_data  = a_pre_zp.getData();
        void* a_post_sf_data = a_post_sf.getData();
        void* a_post_zp_data = a_post_zp.getData();
        if (!a_pre_sf_data || !a_post_sf_data) {
            return UALError::UAL_FAILURE;
        }

        md_t       sf_len  = a_pre_sf.getCols();
        md_t       zp_len  = a_pre_zp.getCols();
        MatrixType sf_type = a_pre_sf.getMatrixType();
        MatrixType zp_type = a_pre_zp.getMatrixType();

        char    layout = (A.getLayout() == MatrixLayout::ROW_MAJOR) ? 'R' : 'C';
        char    transA = A.isTransposed() ? 'T' : 'N';
        char    transB = B.isTransposed() ? 'T' : 'N';
        int32_t alpha_s32 = static_cast<int32_t>(m_alpha);
        int32_t beta_s32  = static_cast<int32_t>(m_beta);

        // Compute to f32 intermediate
        Matrix tempC_f32(C.getEffectiveRows(), C.getEffectiveCols(),
                         MatrixType::f32, C.getLayout());
        if (beta_s32 != 0) {
            dlp::testing::utils::copyMatrixTo<float>(
                C, reinterpret_cast<float*>(tempC_f32.getData()),
                tempC_f32.getLeadingDimension(), tempC_f32.getLayout());
        } else {
            std::memset(tempC_f32.getData(), 0, tempC_f32.getDataSizeBytes());
        }

        if (isF32S8QuantGemm) {
            dlp::testing::classic::ref::aocl_gemm_f32s8s32of32_ref(
                layout, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<const float*>(
                    A.getMatrixData().getMatrixPtr()),
                static_cast<int>(A.getLeadingDimension()),
                reinterpret_cast<const int8_t*>(
                    B.getMatrixData().getMatrixPtr()),
                static_cast<int>(B.getLeadingDimension()), beta_s32,
                reinterpret_cast<float*>(tempC_f32.getData()),
                static_cast<int>(tempC_f32.getLeadingDimension()),
                a_pre_sf_data, a_pre_zp_data, a_post_sf_data, a_post_zp_data,
                sf_len, zp_len, sf_type, zp_type);
        } else {
            dlp::testing::classic::ref::aocl_gemm_bf16s8s32of32_ref(
                layout, transA, transB, A.getEffectiveRows(),
                B.getEffectiveCols(), A.getEffectiveCols(), alpha_s32,
                reinterpret_cast<const bfloat16*>(
                    A.getMatrixData().getMatrixPtr()),
                static_cast<int>(A.getLeadingDimension()),
                reinterpret_cast<const int8_t*>(
                    B.getMatrixData().getMatrixPtr()),
                static_cast<int>(B.getLeadingDimension()), beta_s32,
                reinterpret_cast<float*>(tempC_f32.getData()),
                static_cast<int>(tempC_f32.getLeadingDimension()),
                a_pre_sf_data, a_pre_zp_data, a_post_sf_data, a_post_zp_data,
                sf_len, zp_len, sf_type, zp_type);
        }

        applyPostOps(tempC_f32);
        if (hasPostOps && isIntegerType(outputType)) {
            truncateF32ToMicro(reinterpret_cast<float*>(tempC_f32.getData()),
                               static_cast<size_t>(tempC_f32.getRows())
                                   * tempC_f32.getCols());
        }
        dlp::testing::utils::copyToMatrix<float>(
            reinterpret_cast<const float*>(tempC_f32.getData()),
            tempC_f32.getLeadingDimension(), C, tempC_f32.getLayout());
        return UALError::UAL_SUCCESS;
    }

    // Standard path: GEMM with optional f32 intermediate for post-ops
    bool isIntegerGemm =
        ((aType == MatrixType::u8 && bType == MatrixType::s8)
         || (aType == MatrixType::s8 && bType == MatrixType::s8));
    bool isBf16Gemm = (aType == MatrixType::bf16 && bType == MatrixType::bf16);
    bool isF32Gemm  = (aType == MatrixType::f32 && bType == MatrixType::f32);
    bool needsF32Intermediate =
        (isIntegerGemm || isBf16Gemm || isF32Gemm) && hasPostOps;

    if (needsF32Intermediate && !ualRef.checkValidGemmParams(A, B, C, true)) {
        needsF32Intermediate = false;
    }

    if (needsF32Intermediate) {
        md_t   M = C.getEffectiveRows();
        md_t   N = C.getEffectiveCols();
        Matrix tempC_f32(M, N, MatrixType::f32, C.getLayout());
        if (m_beta != 0.0) {
            dlp::testing::utils::copyMatrixTo<float>(
                C, reinterpret_cast<float*>(tempC_f32.getData()),
                tempC_f32.getLeadingDimension(), tempC_f32.getLayout());
        } else {
            std::memset(tempC_f32.getData(), 0, tempC_f32.getDataSizeBytes());
        }

        bool result = ualRef.gemm(A, B, tempC_f32, m_acc_type, m_alpha, m_beta);
        if (!result) {
            return UALError::UAL_FAILURE;
        }

        applyPostOps(tempC_f32);
        if (isIntegerType(outputType)) {
            truncateF32ToMicro(reinterpret_cast<float*>(tempC_f32.getData()),
                               static_cast<size_t>(tempC_f32.getRows())
                                   * tempC_f32.getCols());
        }
        dlp::testing::utils::copyToMatrix<float>(
            reinterpret_cast<const float*>(tempC_f32.getData()),
            tempC_f32.getLeadingDimension(), C, tempC_f32.getLayout());
        return UALError::UAL_SUCCESS;
    }

    // Simple path: no post-ops or no f32 intermediate needed
    bool result = ualRef.checkValidGemmParams(A, B, C, hasPostOps)
                  && ualRef.gemm(A, B, C, m_acc_type, m_alpha, m_beta);
    if (!result) {
        return UALError::UAL_FAILURE;
    }

    if (hasPostOps) {
        applyPostOps(C);
    }

    return UALError::UAL_SUCCESS;
}

void
RefUalPlan::applyPostOps(Matrix& C)
{
    UalRef ualRef;
    for (const auto& param : m_post_ops) {
        if (!param)
            continue;
        switch (param->getType()) {
            case OperationType::ElementWise:
                ualRef.applyPostOperation(
                    C, static_cast<const ElementWiseParam&>(*param));
                break;
            case OperationType::Scale:
                ualRef.applyPostOperation(
                    C, static_cast<const ScaleParam&>(*param));
                break;
            case OperationType::Bias:
                ualRef.applyPostOperation(
                    C, static_cast<const BiasParam&>(*param));
                break;
            case OperationType::MatAdd:
                ualRef.applyPostOperation(
                    C, static_cast<const MatrixAddParam&>(*param));
                break;
            case OperationType::MatMul:
                ualRef.applyPostOperation(
                    C, static_cast<const MatrixMulParam&>(*param));
                break;
            default:
                break;
        }
    }
}

} // namespace dlp::testing::classic
