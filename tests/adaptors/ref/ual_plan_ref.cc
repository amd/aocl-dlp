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
#include <vector>

using namespace dlp::testing::framework;
using dlp::testing::utils::isIntegerType;
using dlp::testing::utils::truncateF32ToMicro;

namespace dlp::testing::classic {

namespace {

    // Software twin of the kernel's storeHalfWidthResult(): copies the low half
    // (I = N/2) columns of the f32 source (M x N) into dst, honoring each
    // matrix's leading dimension and layout. dst's columns [I, N) are left
    // untouched so they keep their init (matching the DLP half-width store,
    // which never writes them).
    //
    // The f32 source values are converted to dst's element type via
    // convertFrom<>, so this works for any output dtype (f32, bf16, integer).
    // Writing through a hardcoded float* would overflow a narrower buffer (e.g.
    // bf16 = 2 bytes/elem) and corrupt the heap.
    void copyHalfWidthResult(const Matrix& src, Matrix& dst)
    {
        const float*     s     = reinterpret_cast<const float*>(src.getData());
        void*            d     = dst.getData();
        const MatrixType dtype = dst.getMatrixType();
        const md_t       rows  = src.getRows();
        const md_t       I     = src.getCols() / 2;
        const md_t       sld   = src.getLeadingDimension();
        const md_t       dld   = dst.getLeadingDimension();
        const bool       srow  = (src.getLayout() == MatrixLayout::ROW_MAJOR);
        const bool       drow  = (dst.getLayout() == MatrixLayout::ROW_MAJOR);

        for (md_t i = 0; i < rows; ++i) {
            for (md_t j = 0; j < I; ++j) {
                const size_t sidx = srow ? (static_cast<size_t>(i) * sld + j)
                                         : (static_cast<size_t>(j) * sld + i);
                const size_t didx = drow ? (static_cast<size_t>(i) * dld + j)
                                         : (static_cast<size_t>(j) * dld + i);
                dlp::testing::utils::convertFrom<float>(d, dtype, didx,
                                                        s[sidx]);
            }
        }
    }

} // namespace

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

        // Dequantization support is through the ADQUANTIZE postop right now,
        // and beta != 0 is not supported with it. Mirrors the rejection in the
        // bf16s8s32 and f32s8s32 APIs. To be removed once beta != 0 is
        // supported.
        if (beta_s32 != 0)
            return UALError::UAL_NOT_SUPPORTED;

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

    bool isS8S8GroupScale =
        (aType == MatrixType::s8 && bType == MatrixType::s8 && m_group_scale);

    // s8 x s4 symmetric static quantization: unpack the nibble-packed s4 B to
    // s8 and reuse the s8s8 sym-quant reference below.
    bool isS8S4GroupScale =
        (aType == MatrixType::s8 && bType == MatrixType::s4 && m_group_scale);

    bool needsF32Intermediate = (isIntegerGemm || isBf16Gemm || isF32Gemm)
                                && hasPostOps && !isS8S8GroupScale
                                && !isS8S4GroupScale;

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

        // A terminal GLU op is shape-changing: it produces only the low
        // I = N/2 output columns (in tempC_f32's columns [0, I)); columns
        // [I, N) hold the raw pre-GLU accumulator and are not part of the
        // result. The compacted low I columns are written to the D buffer
        // below (never back into C), mirroring the kernel's half-width store.
        const bool gluTerminal =
            hasPostOps && !m_post_ops.empty()
            && isTerminalPostOp(m_post_ops.back()->getType());

        if (gluTerminal) {
            // Match the non-GLU path: integer outputs truncate the f32 result
            // before the dtype conversion. Only the low I columns are valid,
            // but truncating the full buffer is harmless (the tail is unused)
            // and keeps this in lockstep with the standard conversion.
            if (isIntegerType(outputType)) {
                truncateF32ToMicro(
                    reinterpret_cast<float*>(tempC_f32.getData()),
                    static_cast<size_t>(tempC_f32.getRows())
                        * tempC_f32.getCols());
            }
            // The compacted (m x I) GLU result goes solely to the caller-owned
            // D buffer (m_glu_out): copyHalfWidthResult writes the low I
            // columns of the 2I source into D's [0, I) columns (all of D),
            // honoring D's layout/leading-dim/dtype. C is left untouched as the
            // raw 2I workspace -- the harness compares D vs D_ref, never C, so
            // there is no half-width store into C. A terminal GLU without a
            // bound D buffer is a harness misconfiguration.
            if (m_glu_out == nullptr) {
                return UALError::UAL_FAILURE;
            }
            copyHalfWidthResult(tempC_f32, *m_glu_out);
            return UALError::UAL_SUCCESS;
        }

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

    // GroupScale (symmetric quantization) path:
    // Uses specialized ref that handles per-group scale application during
    // K-panel accumulation, which is required for correct results when
    // group_size > 0.
    if (isS8S8GroupScale || isS8S4GroupScale) {
        md_t gs = m_group_scale->getGroupSize();

        if (!ualRef.checkValidGemmParams(A, B, C, false, gs)) {
            return UALError::UAL_FAILURE;
        }

        md_t M = C.getEffectiveRows();
        md_t N = C.getEffectiveCols();
        md_t K = A.getEffectiveCols();

        md_t gs_eff = (gs == 0) ? K : gs;
        md_t ng     = (K + gs_eff - 1) / gs_eff;

        if (!m_group_scale->hasAScaleFactor()
            || !m_group_scale->hasBScaleFactor()) {
            return UALError::UAL_FAILURE;
        }

        const void* a_sf_src = m_group_scale->getAScaleFactor()->getData();
        const void* b_sf_src = m_group_scale->getBScaleFactor()->getData();
        md_t        a_sf_len = m_group_scale->getAScaleFactor()->getCols();
        md_t        b_sf_len = m_group_scale->getBScaleFactor()->getCols();

        // A and B scale factors must share one storage type: read_sf() reads
        // both buffers with the same type, and the library likewise rejects a
        // mixed bf16/f32 pair ("A and B scale factor type mismatch"). Reject a
        // mismatch here before reinterpreting either buffer.
        MatrixType a_sf_type =
            m_group_scale->getAScaleFactor()->getMatrixType();
        MatrixType b_sf_type =
            m_group_scale->getBScaleFactor()->getMatrixType();
        if (a_sf_type != b_sf_type) {
            return UALError::UAL_NOT_SUPPORTED;
        }
        MatrixType sf_type = a_sf_type;

        // The Zen4 sym-quant kernel accepts f32 or bf16 scale factors: bf16 is
        // widened to f32 losslessly before the dequant multiply (see the
        // sf_stor_type == DLP_BF16 branch in
        // dlp_gemm_6x64rowmajor_s8_grp_amd512vnni.c). Mirror that here and
        // treat any other storage type as genuinely unsupported.
        if (sf_type != MatrixType::f32 && sf_type != MatrixType::bf16) {
            return UALError::UAL_NOT_SUPPORTED;
        }

        // Validate supported scale factor lengths:
        //   A: 1 (scalar), M (per-row), or M*ng (fully tiled)
        //   B: 1 (scalar), N (per-col), or ng*N (fully tiled)
        bool a_len_ok = (a_sf_len == 1 || a_sf_len == M || a_sf_len == M * ng);
        bool b_len_ok = (b_sf_len == 1 || b_sf_len == N || b_sf_len == ng * N);
        if (!a_len_ok || !b_len_ok) {
            return UALError::UAL_NOT_SUPPORTED;
        }

        // Read one source scale element as f32. bf16 -> f32 is a lossless
        // widen, identical to the kernel's pre-multiply conversion, so the
        // reference stays bit-consistent with the DLP path for bf16-stored
        // scales.
        auto read_sf = [sf_type](const void* base, md_t idx) -> float {
            if (sf_type == MatrixType::bf16) {
                return dlp::testing::utils::bf16_to_f32(
                    static_cast<const bfloat16*>(base)[idx]);
            }
            return static_cast<const float*>(base)[idx];
        };

        // The specialized ref indexes scale factors as 2D arrays:
        //   a_scale[i * ng + g]  — needs M * ng elements
        //   b_scale[g * n + j]   — needs ng * N elements
        // Always materialize widened f32 buffers, broadcasting scalar/per-row
        // (A) or scalar/per-col (B) as needed, so the ref kernel reads plain
        // f32 regardless of the source scale storage type.
        std::vector<float> a_sf_buf(static_cast<size_t>(M) * ng);
        std::vector<float> b_sf_buf(static_cast<size_t>(ng) * N);

        for (md_t i = 0; i < M; ++i) {
            for (md_t g = 0; g < ng; ++g) {
                md_t src             = (a_sf_len == M * ng) ? (i * ng + g)
                                       : (a_sf_len == M)    ? i
                                                            : 0;
                a_sf_buf[i * ng + g] = read_sf(a_sf_src, src);
            }
        }
        for (md_t g = 0; g < ng; ++g) {
            for (md_t j = 0; j < N; ++j) {
                md_t src            = (b_sf_len == ng * N) ? (g * N + j)
                                      : (b_sf_len == N)    ? j
                                                           : 0;
                b_sf_buf[g * N + j] = read_sf(b_sf_src, src);
            }
        }

        const void* a_sf_ptr = a_sf_buf.data();
        const void* b_sf_ptr = b_sf_buf.data();

        char    layout = (A.getLayout() == MatrixLayout::ROW_MAJOR) ? 'r' : 'c';
        char    transA_   = A.isTransposed() ? 't' : 'n';
        char    transB_   = B.isTransposed() ? 't' : 'n';
        int32_t alpha_s32 = static_cast<int32_t>(m_alpha);
        int32_t beta_s32  = static_cast<int32_t>(m_beta);

        Matrix tempC_f32(M, N, MatrixType::f32, C.getLayout());
        if (beta_s32 != 0) {
            dlp::testing::utils::copyMatrixTo<float>(
                C, reinterpret_cast<float*>(tempC_f32.getData()),
                tempC_f32.getLeadingDimension(), tempC_f32.getLayout());
        } else {
            std::memset(tempC_f32.getData(), 0, tempC_f32.getDataSizeBytes());
        }

        // B pointer for the s8s8 sym-quant ref. For s8s8 this is the raw s8 B.
        // For s8s4, unpack the nibble-packed s4 B into a contiguous s8 buffer
        // with the same linear layout (low nibble first, sign-extended); the
        // ref then reads it identically via B.getLeadingDimension().
        const int8_t* b_ptr =
            reinterpret_cast<const int8_t*>(B.getMatrixData().getMatrixPtr());
        std::vector<int8_t> b_s8_unpacked;
        if (isS8S4GroupScale) {
            const int8_t* packed       = b_ptr;
            size_t        nibble_count = B.getDataSizeBytes() * 2;
            b_s8_unpacked.resize(nibble_count);
            for (size_t idx = 0; idx < nibble_count; ++idx) {
                int     shift = static_cast<int>((idx & 1) * 4);
                uint8_t bits4 =
                    (static_cast<uint8_t>(packed[idx / 2]) >> shift) & 0x0F;
                b_s8_unpacked[idx] = (bits4 & 0x08)
                                         ? static_cast<int8_t>(bits4 | 0xF0)
                                         : static_cast<int8_t>(bits4);
            }
            b_ptr = b_s8_unpacked.data();
        }

        dlp::testing::classic::ref::aocl_gemm_s8s8s32of32_sym_quant_ref(
            layout, transA_, transB_, M, N, K, alpha_s32,
            reinterpret_cast<const int8_t*>(A.getMatrixData().getMatrixPtr()),
            static_cast<int>(A.getLeadingDimension()), b_ptr,
            static_cast<int>(B.getLeadingDimension()), beta_s32,
            reinterpret_cast<float*>(tempC_f32.getData()),
            static_cast<int>(tempC_f32.getLeadingDimension()), gs, a_sf_ptr,
            b_sf_ptr, ng, MatrixType::f32);

        applyPostOps(tempC_f32);
        dlp::testing::utils::copyToMatrix<float>(
            reinterpret_cast<const float*>(tempC_f32.getData()),
            tempC_f32.getLeadingDimension(), C, tempC_f32.getLayout());
        return UALError::UAL_SUCCESS;
    }

    // FP16 path: GEMM natively in FP16, post-ops in F32 intermediate
    if ((aType == MatrixType::fp16 && bType == MatrixType::fp16)
        && hasPostOps) {
        bool result = ualRef.checkValidGemmParams(A, B, C, hasPostOps)
                      && ualRef.gemm(A, B, C, m_acc_type, m_alpha, m_beta);
        if (!result) {
            return UALError::UAL_FAILURE;
        }

        md_t   M = C.getEffectiveRows();
        md_t   N = C.getEffectiveCols();
        Matrix tempC_f32(M, N, MatrixType::f32, C.getLayout());
        dlp::testing::utils::copyMatrixTo<float>(
            C, reinterpret_cast<float*>(tempC_f32.getData()),
            tempC_f32.getLeadingDimension(), tempC_f32.getLayout());

        applyPostOps(tempC_f32);

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
            case OperationType::GLU:
                ualRef.applyPostOperation(C,
                                          static_cast<const GluParam&>(*param));
                break;
            default:
                break;
        }
    }
}

} // namespace dlp::testing::classic
