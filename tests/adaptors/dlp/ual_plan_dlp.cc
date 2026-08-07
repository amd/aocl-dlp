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

#include "adaptors/dlp/ual_plan_dlp.hh"
#include "aocl_dlp.h"
#include "classic/aocl_fp16_type.h"
#include "classic/dlp_errors.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

using namespace dlp::testing::framework;

namespace dlp::testing::classic {

namespace {

    DLP_PARAM_DIM_TYPE
    getScalarOrVectorDim(md_t len, DLP_PARAM_DIM_TYPE vector_dim)
    {
        return (len == 1) ? DLP_PARAM_DIM_PER_TENSOR : vector_dim;
    }

    DLP_TYPE
    getQuantComputeDstType(MatrixType src_type)
    {
        switch (src_type) {
            case MatrixType::s4:
            case MatrixType::u4:
                // Weight-only 4-bit B operands are dequantized to BF16 so the
                // BF16 GEMM/VNNI kernels can consume them.
                return DLP_BF16;
            case MatrixType::f32:
            case MatrixType::bf16:
            case MatrixType::s8:
                // F32/BF16 quant APIs target S8 operands for INT8 S8S8 VNNI.
                // Existing S8 operands remain in the S8 compute domain.
                return DLP_S8;
            case MatrixType::u8:
                return DLP_U8;
            default:
                return DLP_INVALID;
        }
    }

} // namespace

DlpUalPlan::DlpUalPlan()
{
    m_metadata = new dlp_metadata_t;
    std::memset(m_metadata, 0, sizeof(dlp_metadata_t));
}

DlpUalPlan::~DlpUalPlan()
{
    cleanupMetadata();
}

void
DlpUalPlan::cleanupMetadata()
{
    if (m_metadata) {
        // Clean up manually allocated memory for zero points in Scale
        // operations
        if (m_metadata->scale) {
            for (std::size_t i = 0; i < m_scale_ops.size(); ++i) {
                // Check if this is a dynamically allocated zero point
                // (for SCALE operations where we allocated it ourselves)
                if (m_metadata->scale[i].zp
                    && m_metadata->scale[i].zp->zero_point_len == 1
                    && m_metadata->scale[i].zp->zero_point_type == DLP_S8) {
                    // Check if this was allocated by us (no zero point provided
                    // originally)
                    if (i < m_scale_ops.size()
                        && !m_scale_ops[i]->hasZeroPoint()) {
                        // This was our dynamically allocated zero point
                        delete static_cast<int8_t*>(
                            m_metadata->scale[i].zp->zero_point);
                    }
                }

                // Clean up the dlp_sf_t and dlp_zp_t structures themselves
                delete m_metadata->scale[i].sf;
                delete m_metadata->scale[i].zp;
            }
        }

        // Clean up manually allocated memory for scale factors in MatrixAdd
        // operations
        if (m_metadata->matrix_add) {
            for (std::size_t i = 0; i < m_matrix_add_ops.size(); ++i) {
                if (m_metadata->matrix_add[i].sf) {
                    delete m_metadata->matrix_add[i].sf;
                }
            }
        }

        // Clean up manually allocated memory for scale factors in MatrixMul
        // operations
        if (m_metadata->matrix_mul) {
            for (std::size_t i = 0; i < m_matrix_mul_ops.size(); ++i) {
                if (m_metadata->matrix_mul[i].sf) {
                    delete m_metadata->matrix_mul[i].sf;
                }
            }
        }

        // Clean up manually allocated memory for scale factors and zero points
        // in Bias operations (both are optional user features)
        if (m_metadata->bias) {
            for (std::size_t i = 0; i < m_bias_ops.size(); ++i) {
                if (m_metadata->bias[i].sf) {
                    delete m_metadata->bias[i].sf;
                }
                if (m_metadata->bias[i].zp) {
                    delete m_metadata->bias[i].zp;
                }
            }
        }

        // Clean up manually allocated quantization parameter descriptors.
        if (m_metadata->a_quant_op) {
            delete m_metadata->a_quant_op->quant_scale_factors;
            delete m_metadata->a_quant_op->dequant_scale_factors;
            delete m_metadata->a_quant_op->zero_point;
            delete m_metadata->a_quant_op;
        }
        if (m_metadata->b_quant_op) {
            delete m_metadata->b_quant_op->quant_scale_factors;
            delete m_metadata->b_quant_op->dequant_scale_factors;
            delete m_metadata->b_quant_op->zero_point;
            delete m_metadata->b_quant_op;
        }

        // Clean up the GLU op (single struct, no nested allocations)
        if (m_metadata->glu) {
            delete m_metadata->glu;
            m_metadata->glu = nullptr;
        }

        // Clean up all allocated arrays
        delete[] m_metadata->eltwise;
        delete[] m_metadata->scale;
        delete[] m_metadata->bias;
        delete[] m_metadata->matrix_add;
        delete[] m_metadata->matrix_mul;
        free(m_metadata->seq_vector);

        delete m_metadata;
        m_metadata = nullptr;
    }
}

void
DlpUalPlan::prepare()
{
    // Reset state: clean up old metadata and typed op vectors
    cleanupMetadata();
    m_elementwise_ops.clear();
    m_scale_ops.clear();
    m_bias_ops.clear();
    m_matrix_add_ops.clear();
    m_matrix_mul_ops.clear();
    m_glu_op.reset();

    // Allocate fresh metadata
    m_metadata = new dlp_metadata_t;
    std::memset(m_metadata, 0, sizeof(dlp_metadata_t));

    // Sort m_post_ops into typed vectors by casting
    for (auto& param : m_post_ops) {
        if (!param) {
            continue;
        }

        switch (param->getType()) {
            case OperationType::ElementWise: {
                auto ew_param = std::unique_ptr<ElementWiseParam>(
                    static_cast<ElementWiseParam*>(param->clone().release()));
                m_elementwise_ops.push_back(std::move(ew_param));
                break;
            }
            case OperationType::Scale: {
                auto scale_param = std::unique_ptr<ScaleParam>(
                    static_cast<ScaleParam*>(param->clone().release()));
                m_scale_ops.push_back(std::move(scale_param));
                break;
            }
            case OperationType::Bias: {
                auto bias_param = std::unique_ptr<BiasParam>(
                    static_cast<BiasParam*>(param->clone().release()));
                m_bias_ops.push_back(std::move(bias_param));
                break;
            }
            case OperationType::MatAdd: {
                auto mat_add_param = std::unique_ptr<MatrixAddParam>(
                    static_cast<MatrixAddParam*>(param->clone().release()));
                m_matrix_add_ops.push_back(std::move(mat_add_param));
                break;
            }
            case OperationType::MatMul: {
                auto mat_mul_param = std::unique_ptr<MatrixMulParam>(
                    static_cast<MatrixMulParam*>(param->clone().release()));
                m_matrix_mul_ops.push_back(std::move(mat_mul_param));
                break;
            }
            case OperationType::GLU: {
                m_glu_op = std::unique_ptr<GluParam>(
                    static_cast<GluParam*>(param->clone().release()));
                break;
            }
            default:
                throw std::runtime_error("Unsupported post-op type in plan");
        }
    }

    // Convert each post-op type to metadata arrays
    if (!m_elementwise_ops.empty()) {
        convertElementWiseOperations();
    }

    if (!m_scale_ops.empty()) {
        convertScaleOperations();
    }

    if (!m_bias_ops.empty()) {
        convertBiasOperations();
    }

    if (!m_matrix_add_ops.empty()) {
        convertMatrixAddOperations();
    }

    if (!m_matrix_mul_ops.empty()) {
        convertMatrixMulOperations();
    }

    // Build the sequence vector based on post-op order
    // Caveat: GLU is not included in the sequence vector, as it is a
    // terminal shape-changing op, and is handled separately in the
    // metadata. However GLU is still included in the m_post_ops vector,
    // so that it can be easily validated and converted to metadata.
    buildSequenceVector();

    if (m_glu_op) {
        convertGluOperations();
    }

    // Convert quantization parameters
    if (m_a_quant) {
        convertA_QuantOperations();
    }

    if (m_b_quant) {
        convertB_QuantOperations();
    }

    if (m_woq) {
        convertWOQOperations();
    }

    if (m_group_scale) {
        convertGroupScaleOperations();
    }

    // Compute type code for dispatch
    m_type_code = encodeTypes(m_a_type, m_b_type, m_c_type, m_acc_type);

    // Compute layout and transpose chars
    m_layout_char = m_a_layout == MatrixLayout::ROW_MAJOR ? 'r' : 'c';
    m_transA_char = m_transA ? 't' : 'n';
    m_transB_char = m_transB ? 't' : 'n';

    // Resolve transB: if B is reordered, force to 'n'
    m_transB_resolved = (m_memFormatB == 'r') ? 'n' : m_transB_char;

    // Pre-cast alpha/beta for non-FP16 rails. The FP16 forms
    // (m_alpha_fp16, m_beta_fp16) are populated once in the base class
    // setAlpha/setBeta via f32_to_fp16 — no extra conversion here.
    m_alpha_f32 = static_cast<float>(m_alpha);
    m_beta_f32  = static_cast<float>(m_beta);
    m_alpha_s32 = static_cast<int32_t>(m_alpha);
    m_beta_s32  = static_cast<int32_t>(m_beta);

    // Resolve dispatch function - captures all pre-computed args.
    // Only a_ptr/b_ptr/c_ptr and lda/ldb/ldc come from setBuffers().
    // Local copies for lambda capture (avoids capturing 'this')
    char            layout     = m_layout_char;
    char            transA     = m_transA_char;
    char            transB     = m_transB_resolved;
    md_t            m_dim      = m_m;
    md_t            n_dim      = m_n;
    md_t            k_dim      = m_k;
    float           alpha_f    = m_alpha_f32;
    float           beta_f     = m_beta_f32;
    int32_t         alpha_i    = m_alpha_s32;
    int32_t         beta_i     = m_beta_s32;
    float16         alpha_fp16 = m_alpha_fp16;
    float16         beta_fp16  = m_beta_fp16;
    char            memA       = m_memFormatA;
    char            memB       = m_memFormatB;
    dlp_metadata_t* meta       = m_metadata;

    m_dispatch = nullptr;

    switch (m_type_code) {
        case encodeTypes<MatrixType::f32, MatrixType::f32, MatrixType::f32,
                         MatrixType::f32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_f32f32f32of32(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_f,
                    reinterpret_cast<float*>(a), lda, memA,
                    reinterpret_cast<float*>(b), ldb, memB, beta_f,
                    reinterpret_cast<float*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::bf16, MatrixType::bf16, MatrixType::f32,
                         MatrixType::f32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_bf16bf16f32of32(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_f,
                    reinterpret_cast<bfloat16*>(a), lda, memA,
                    reinterpret_cast<bfloat16*>(b), ldb, memB, beta_f,
                    reinterpret_cast<float*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::bf16, MatrixType::bf16, MatrixType::bf16,
                         MatrixType::f32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_bf16bf16f32obf16(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_f,
                    reinterpret_cast<bfloat16*>(a), lda, memA,
                    reinterpret_cast<bfloat16*>(b), ldb, memB, beta_f,
                    reinterpret_cast<bfloat16*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::u8, MatrixType::s8, MatrixType::s32,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_u8s8s32os32(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<uint8_t*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<int32_t*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::s8, MatrixType::s8, MatrixType::s32,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_s8s8s32os32(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<int8_t*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<int32_t*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::u8, MatrixType::s8, MatrixType::s8,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_u8s8s32os8(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<uint8_t*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<int8_t*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::u8, MatrixType::s8, MatrixType::f32,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_u8s8s32of32(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<uint8_t*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<float*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::u8, MatrixType::s8, MatrixType::fp16,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_u8s8s32of16(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<uint8_t*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<float16*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::u8, MatrixType::s8, MatrixType::u8,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_u8s8s32ou8(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<uint8_t*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<uint8_t*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::u8, MatrixType::s8, MatrixType::bf16,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_u8s8s32obf16(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<uint8_t*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<bfloat16*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::s8, MatrixType::s8, MatrixType::f32,
                         MatrixType::s32>():
            if (m_group_scale) {
                m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                                 md_t ldc) {
                    aocl_gemm_s8s8s32of32_sym_quant(
                        layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                        reinterpret_cast<int8_t*>(a), lda, memA,
                        reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                        reinterpret_cast<float*>(c), ldc, meta);
                };
            } else {
                m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                                 md_t ldc) {
                    aocl_gemm_s8s8s32of32(
                        layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                        reinterpret_cast<int8_t*>(a), lda, memA,
                        reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                        reinterpret_cast<float*>(c), ldc, meta);
                };
            }
            break;

        case encodeTypes<MatrixType::s8, MatrixType::s8, MatrixType::fp16,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_s8s8s32of16(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<int8_t*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<float16*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::s8, MatrixType::s8, MatrixType::s8,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_s8s8s32os8(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<int8_t*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<int8_t*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::s8, MatrixType::s8, MatrixType::u8,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_s8s8s32ou8(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<int8_t*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<uint8_t*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::s8, MatrixType::s8, MatrixType::bf16,
                         MatrixType::s32>():
            if (m_group_scale) {
                m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                                 md_t ldc) {
                    aocl_gemm_s8s8s32obf16_sym_quant(
                        layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                        reinterpret_cast<int8_t*>(a), lda, memA,
                        reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                        reinterpret_cast<bfloat16*>(c), ldc, meta);
                };
            } else {
                m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                                 md_t ldc) {
                    aocl_gemm_s8s8s32obf16(
                        layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                        reinterpret_cast<int8_t*>(a), lda, memA,
                        reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                        reinterpret_cast<bfloat16*>(c), ldc, meta);
                };
            }
            break;

        // s8 x s4 symmetric static quantization. Only the sym-quant API exists
        // for s8s4 (there is no plain variant), so these require m_group_scale;
        // B is nibble-packed s4 passed as int8_t*.
        case encodeTypes<MatrixType::s8, MatrixType::s4, MatrixType::f32,
                         MatrixType::s32>():
            if (m_group_scale) {
                m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                                 md_t ldc) {
                    aocl_gemm_s8s4s32of32(
                        layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                        reinterpret_cast<int8_t*>(a), lda, memA,
                        reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                        reinterpret_cast<float*>(c), ldc, meta);
                };
            }
            break;

        case encodeTypes<MatrixType::s8, MatrixType::s4, MatrixType::bf16,
                         MatrixType::s32>():
            if (m_group_scale) {
                m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                                 md_t ldc) {
                    aocl_gemm_s8s4s32obf16(
                        layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                        reinterpret_cast<int8_t*>(a), lda, memA,
                        reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                        reinterpret_cast<bfloat16*>(c), ldc, meta);
                };
            }
            break;

        case encodeTypes<MatrixType::bf16, MatrixType::s8, MatrixType::bf16,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_bf16s8s32obf16(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<bfloat16*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<bfloat16*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::fp16, MatrixType::fp16, MatrixType::fp16,
                         MatrixType::fp16>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_f16f16f16of16(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_fp16,
                    reinterpret_cast<float16*>(a), lda, memA,
                    reinterpret_cast<float16*>(b), ldb, memB, beta_fp16,
                    reinterpret_cast<float16*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::fp16, MatrixType::fp16, MatrixType::f32,
                         MatrixType::fp16>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_f16f16f16of32(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_fp16,
                    reinterpret_cast<float16*>(a), lda, memA,
                    reinterpret_cast<float16*>(b), ldb, memB, beta_fp16,
                    reinterpret_cast<float*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::f32, MatrixType::fp16, MatrixType::f32,
                         MatrixType::f32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_f32f16f32of32(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_f,
                    reinterpret_cast<float*>(a), lda, memA,
                    reinterpret_cast<float16*>(b), ldb, memB, beta_f,
                    reinterpret_cast<float*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::bf16, MatrixType::s8, MatrixType::f32,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_bf16s8s32of32(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<bfloat16*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<float*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::bf16, MatrixType::s8, MatrixType::s32,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_bf16s8s32os32(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<bfloat16*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<int32_t*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::bf16, MatrixType::s8, MatrixType::s8,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_bf16s8s32os8(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<bfloat16*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<int8_t*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::bf16, MatrixType::s8, MatrixType::u8,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_bf16s8s32ou8(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<bfloat16*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<uint8_t*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::f32, MatrixType::s8, MatrixType::f32,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_f32s8s32of32(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<float*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<float*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::f32, MatrixType::s8, MatrixType::s32,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_f32s8s32os32(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<float*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<int32_t*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::f32, MatrixType::s8, MatrixType::bf16,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_f32s8s32obf16(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<float*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<bfloat16*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::f32, MatrixType::s8, MatrixType::s8,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_f32s8s32os8(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<float*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<int8_t*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::f32, MatrixType::s8, MatrixType::u8,
                         MatrixType::s32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_f32s8s32ou8(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<float*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<uint8_t*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::bf16, MatrixType::s4, MatrixType::f32,
                         MatrixType::f32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_bf16s4f32of32(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_f,
                    reinterpret_cast<bfloat16*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_f,
                    reinterpret_cast<float*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::bf16, MatrixType::s4, MatrixType::bf16,
                         MatrixType::f32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_bf16s4f32obf16(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_f,
                    reinterpret_cast<bfloat16*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_f,
                    reinterpret_cast<bfloat16*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::bf16, MatrixType::u4, MatrixType::f32,
                         MatrixType::f32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_bf16u4f32of32(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_f,
                    reinterpret_cast<bfloat16*>(a), lda, memA,
                    reinterpret_cast<uint8_t*>(b), ldb, memB, beta_f,
                    reinterpret_cast<float*>(c), ldc, meta);
            };
            break;

        case encodeTypes<MatrixType::bf16, MatrixType::u4, MatrixType::bf16,
                         MatrixType::f32>():
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_bf16u4f32obf16(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_f,
                    reinterpret_cast<bfloat16*>(a), lda, memA,
                    reinterpret_cast<uint8_t*>(b), ldb, memB, beta_f,
                    reinterpret_cast<bfloat16*>(c), ldc, meta);
            };
            break;

        default:
            // Unknown type combo - dispatch will remain null, execute()
            // will return UAL_FAILURE
            break;
    }

    m_prepared = true;
}

UALError
DlpUalPlan::execute()
{
    if (!m_prepared) {
        return UALError::UAL_FAILURE;
    }
    if (!m_buffers_set) {
        return UALError::UAL_FAILURE;
    }
    if (!m_dispatch) {
        return UALError::UAL_FAILURE;
    }

    // Reset error state before dispatch
    m_metadata->error_hndl.error_code = DLP_CLSC_SUCCESS;

    m_dispatch(m_a_ptr, m_buf_lda, m_b_ptr, m_buf_ldb, m_c_ptr, m_buf_ldc);

    if (m_metadata->error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED)
        return UALError::UAL_NOT_SUPPORTED;

    if (m_metadata->error_hndl.error_code == DLP_CLSC_SUCCESS)
        return UALError::UAL_SUCCESS;

    return UALError::UAL_FAILURE;
}

// ─── Post-Op Conversion Methods ─────────────────────────────────────────────

void
DlpUalPlan::convertElementWiseOperations()
{
    size_t count = m_elementwise_ops.size();
    if (count == 0)
        return;

    // Allocate array once with exact size
    m_metadata->eltwise = new dlp_post_op_eltwise[count];
    std::memset(m_metadata->eltwise, 0, count * sizeof(dlp_post_op_eltwise));
    m_metadata->num_eltwise = count;

    // Fill the array
    for (std::size_t i = 0; i < count; ++i) {
        const auto& param = *m_elementwise_ops[i];

        m_metadata->eltwise[i].algo.algo_type =
            getElementWiseAlgoType(param.getOperation());

        // Set storage type for alpha and beta parameters.
        // For CLIP: alpha (lower bound) and beta (upper bound) must have the
        // same type. The parser layer ensures type consistency, so we can use
        // either parameter's type. Default to f32 if no parameters are
        // provided.
        m_metadata->eltwise[i].algo.stor_type = DLP_F32;

        // Set alpha and beta if provided
        if (param.hasAlpha()) {
            m_metadata->eltwise[i].algo.alpha =
                convertMatrixToPtr(*param.getAlpha());
            m_metadata->eltwise[i].algo.stor_type =
                getStorageType(param.getAlpha()->getMatrixType());
        }
        if (param.hasBeta()) {
            m_metadata->eltwise[i].algo.beta =
                convertMatrixToPtr(*param.getBeta());
            // If alpha wasn't present, use beta's type for stor_type
            // For CLIP, parser guarantees both types match if both are
            // specified
            if (!param.hasAlpha()) {
                m_metadata->eltwise[i].algo.stor_type =
                    getStorageType(param.getBeta()->getMatrixType());
            }
        }
    }
}

void
DlpUalPlan::convertScaleOperations()
{
    size_t count = m_scale_ops.size();
    if (count == 0)
        return;

    // Allocate array once with exact size
    m_metadata->scale = new dlp_scale_t[count];
    std::memset(m_metadata->scale, 0, count * sizeof(dlp_scale_t));

    // Fill the array
    for (std::size_t i = 0; i < count; ++i) {
        const auto& param = *m_scale_ops[i];

        // Initialize all fields to safe defaults
        m_metadata->scale[i].sf = nullptr;
        m_metadata->scale[i].zp = nullptr;

        // Set scale factor if provided
        if (param.hasScaleFactor()) {
            // Single source of truth for (ParamDim, SF matrix shape)
            // coherence on the test path. ScaleBuilder deliberately
            // defers this to the adaptor — this is where the matrix is
            // turned into a (dlp_sf_t::scale_factor, scale_factor_len)
            // pair, so a mismatch here would either silently truncate
            // (PerTensor + vector matrix -> len = 1, rest of the SF
            // dropped) or alias past the SF buffer (PerChannel /
            // PerToken + 1x1 matrix -> len > 1). Failing fast keeps
            // inconsistent metadata from ever reaching the C validator
            // or JIT codegen.
            const Matrix* sfMat = param.getScaleFactor();
            const md_t    rows  = static_cast<md_t>(sfMat->getRows());
            const md_t    cols  = static_cast<md_t>(sfMat->getCols());
            ParamDim      sfDim = param.getScaleFactorDim();

            switch (sfDim) {
                case ParamDim::PerTensor:
                    if (rows * cols != 1) {
                        throw std::runtime_error(
                            "DlpUalPlan: SCALE PerTensor requires a 1x1 "
                            "scale-factor matrix");
                    }
                    break;
                case ParamDim::PerChannel:
                    if (rows != 1 || cols < 1) {
                        throw std::runtime_error(
                            "DlpUalPlan: SCALE PerChannel requires a 1xN "
                            "scale-factor matrix");
                    }
                    break;
                case ParamDim::PerToken:
                    if (cols != 1 || rows < 1) {
                        throw std::runtime_error(
                            "DlpUalPlan: SCALE PerToken requires an Mx1 "
                            "scale-factor matrix");
                    }
                    break;
            }

            m_metadata->scale[i].sf               = new dlp_sf_t{};
            m_metadata->scale[i].sf->scale_factor = convertMatrixToPtr(*sfMat);
            m_metadata->scale[i].sf->scale_factor_type =
                getStorageType(sfMat->getMatrixType());
            m_metadata->scale[i].sf->scale_factor_dim = getParamDim(sfDim);
            m_metadata->scale[i].sf->scale_factor_len =
                (sfDim == ParamDim::PerToken)     ? rows
                : (sfDim == ParamDim::PerChannel) ? cols
                                                  : md_t{ 1 };
        }

        // Set zero point if provided
        if (param.hasZeroPoint()) {
            m_metadata->scale[i].zp = new dlp_zp_t;
            m_metadata->scale[i].zp->zero_point =
                convertMatrixToPtr(*param.getZeroPoint());
            m_metadata->scale[i].zp->zero_point_len =
                param.getZeroPoint()->getCols();
            m_metadata->scale[i].zp->zero_point_type =
                getStorageType(param.getZeroPoint()->getMatrixType());
        } else {
            // Based on DLP library validation, if we have a scale factor, we
            // need zero point
            if (param.hasScaleFactor()) {
                // Allocate and set default zero point for SCALE operations
                int8_t* zero_point_data                  = new int8_t(0);
                m_metadata->scale[i].zp                  = new dlp_zp_t;
                m_metadata->scale[i].zp->zero_point      = zero_point_data;
                m_metadata->scale[i].zp->zero_point_len  = 1;
                m_metadata->scale[i].zp->zero_point_type = DLP_S8;
            } else {
                // No scale factor provided; invalid for SCALE op in testing.
                throw std::runtime_error(
                    "Scale operation requires scale factor");
            }
        }
    }
}

void
DlpUalPlan::convertBiasOperations()
{
    size_t count = m_bias_ops.size();
    if (count == 0)
        return;

    // Allocate array once with exact size
    m_metadata->bias = new dlp_post_op_bias[count];
    std::memset(m_metadata->bias, 0, count * sizeof(dlp_post_op_bias));

    // Fill the array
    for (std::size_t i = 0; i < count; ++i) {
        const auto& param = *m_bias_ops[i];

        m_metadata->bias[i].bias = convertMatrixToPtr(param.getBias());
        m_metadata->bias[i].bias_len =
            param.getBias().getRows() * param.getBias().getCols();
        m_metadata->bias[i].stor_type =
            getStorageType(param.getBias().getMatrixType());

        // Set scale factor if provided
        if (param.hasScaleFactor()) {
            md_t bias_sf_len       = param.getScaleFactor()->getCols();
            m_metadata->bias[i].sf = new dlp_sf_t{};
            m_metadata->bias[i].sf->scale_factor =
                convertMatrixToPtr(*param.getScaleFactor());
            m_metadata->bias[i].sf->scale_factor_len = bias_sf_len;
            m_metadata->bias[i].sf->scale_factor_type =
                getStorageType(param.getScaleFactor()->getMatrixType());
            m_metadata->bias[i].sf->scale_factor_dim =
                (bias_sf_len == 1) ? DLP_PARAM_DIM_PER_TENSOR
                                   : DLP_PARAM_DIM_PER_CHANNEL;
        }

        // Set zero point if provided
        if (param.hasZeroPoint()) {
            m_metadata->bias[i].zp = new dlp_zp_t;
            m_metadata->bias[i].zp->zero_point =
                convertMatrixToPtr(*param.getZeroPoint());
            m_metadata->bias[i].zp->zero_point_len =
                param.getZeroPoint()->getCols();
            m_metadata->bias[i].zp->zero_point_type =
                getStorageType(param.getZeroPoint()->getMatrixType());
        }
    }
}

void
DlpUalPlan::convertMatrixAddOperations()
{
    size_t count = m_matrix_add_ops.size();
    if (count == 0)
        return;

    // Allocate array once with exact size
    m_metadata->matrix_add = new dlp_post_op_matrix_add[count];
    std::memset(m_metadata->matrix_add, 0,
                count * sizeof(dlp_post_op_matrix_add));

    // Fill the array
    for (std::size_t i = 0; i < count; ++i) {
        const auto& param = *m_matrix_add_ops[i];

        // Initialize sf pointer to null
        m_metadata->matrix_add[i].sf = nullptr;

        m_metadata->matrix_add[i].matrix =
            convertMatrixToPtr(param.getMatrix());
        m_metadata->matrix_add[i].ldm = param.getMatrix().getLeadingDimension();
        m_metadata->matrix_add[i].stor_type =
            getStorageType(param.getMatrix().getMatrixType());

        if (param.hasScaleFactor()) {
            md_t madd_sf_len             = param.getScaleFactor()->getCols();
            m_metadata->matrix_add[i].sf = new dlp_sf_t{};
            m_metadata->matrix_add[i].sf->scale_factor =
                convertMatrixToPtr(*param.getScaleFactor());
            m_metadata->matrix_add[i].sf->scale_factor_len = madd_sf_len;
            m_metadata->matrix_add[i].sf->scale_factor_type =
                getStorageType(param.getScaleFactor()->getMatrixType());
            m_metadata->matrix_add[i].sf->scale_factor_dim =
                (madd_sf_len == 1) ? DLP_PARAM_DIM_PER_TENSOR
                                   : DLP_PARAM_DIM_PER_CHANNEL;
        }
    }
}

void
DlpUalPlan::convertMatrixMulOperations()
{
    size_t count = m_matrix_mul_ops.size();
    if (count == 0)
        return;

    // Allocate array once with exact size
    m_metadata->matrix_mul = new dlp_post_op_matrix_mul[count];
    std::memset(m_metadata->matrix_mul, 0,
                count * sizeof(dlp_post_op_matrix_mul));

    // Fill the array
    for (std::size_t i = 0; i < count; ++i) {
        const auto& param = *m_matrix_mul_ops[i];

        // Initialize sf pointer to null
        m_metadata->matrix_mul[i].sf = nullptr;

        m_metadata->matrix_mul[i].matrix =
            convertMatrixToPtr(param.getMatrix());
        m_metadata->matrix_mul[i].ldm = param.getMatrix().getLeadingDimension();
        m_metadata->matrix_mul[i].stor_type =
            getStorageType(param.getMatrix().getMatrixType());

        if (param.hasScaleFactor()) {
            md_t mmul_sf_len             = param.getScaleFactor()->getCols();
            m_metadata->matrix_mul[i].sf = new dlp_sf_t{};
            m_metadata->matrix_mul[i].sf->scale_factor =
                convertMatrixToPtr(*param.getScaleFactor());
            m_metadata->matrix_mul[i].sf->scale_factor_len = mmul_sf_len;
            m_metadata->matrix_mul[i].sf->scale_factor_type =
                getStorageType(param.getScaleFactor()->getMatrixType());
            m_metadata->matrix_mul[i].sf->scale_factor_dim =
                (mmul_sf_len == 1) ? DLP_PARAM_DIM_PER_TENSOR
                                   : DLP_PARAM_DIM_PER_CHANNEL;
        }
    }
}

void
DlpUalPlan::convertGluOperations()
{
    if (!m_glu_op)
        return;

    // GLU lives in a single metadata slot (at most one per chain).
    m_metadata->glu = new dlp_term_op_glu;
    std::memset(m_metadata->glu, 0, sizeof(dlp_term_op_glu));
    switch (m_glu_op->getOperation()) {
        case GluOperation::GatedSwiglu:
            m_metadata->glu->algo_type = GATED_SWIGLU;
            break;
        case GluOperation::GatedSwigluAndMul:
            m_metadata->glu->algo_type = GATED_SWIGLU_AND_MUL;
            break;
        default:
            throw std::runtime_error("Unsupported GLU operation");
    }

    // Scalar slots point at the GluParam's own storage (m_glu_op outlives
    // execute()). Both are NULL for current variants (constants baked into the
    // kernel); the slots remain for future runtime-scalar variants.
    m_metadata->glu->alpha = m_glu_op->getAlphaPtr();
    m_metadata->glu->beta  = m_glu_op->getBetaPtr();

    // stor_type is meaningful only when a scalar is supplied; leave it
    // DLP_INVALID otherwise (see aocl_gemm_metadata.h).
    m_metadata->glu->stor_type =
        (m_metadata->glu->alpha != nullptr || m_metadata->glu->beta != nullptr)
            ? DLP_F32
            : DLP_INVALID;

    // Mandatory compacted output buffer D (m x I), owned by the harness. The
    // library folds the 2I (gate, up) accumulator to width I here; ld_d carries
    // the layout (I for row-major, m for column-major).
    if (m_glu_out != nullptr) {
        m_metadata->glu->d = m_glu_out->getMatrixData().getMatrixPtr();
        m_metadata->glu->ld_d =
            static_cast<md_t>(m_glu_out->getLeadingDimension());
    }
}

void
DlpUalPlan::buildSequenceVector()
{
    // Build sequence vector based on original post-op order
    std::vector<DLP_POST_OP_TYPE> sequence;
    sequence.reserve(m_post_ops.size());

    for (const auto& param : m_post_ops) {
        switch (param->getType()) {
            case OperationType::ElementWise:
                sequence.push_back(ELTWISE);
                break;
            case OperationType::Scale:
                sequence.push_back(SCALE);
                break;
            case OperationType::Bias:
                sequence.push_back(BIAS);
                break;
            case OperationType::MatAdd:
                sequence.push_back(MATRIX_ADD);
                break;
            case OperationType::MatMul:
                sequence.push_back(MATRIX_MUL);
                break;
            case OperationType::GLU:
                break; // Ignore GLU in sequence; it has its own metadata slot.
            default:
                throw std::runtime_error(
                    "Unsupported operation type in sequence");
        }
    }

    // Set sequence information
    m_metadata->seq_length = sequence.size();
    if (!sequence.empty()) {
        m_metadata->seq_vector = static_cast<DLP_POST_OP_TYPE*>(
            malloc(sequence.size() * sizeof(DLP_POST_OP_TYPE)));
        std::copy(sequence.begin(), sequence.end(), m_metadata->seq_vector);
    }
}

// ─── Quantisation Conversion Methods ────────────────────────────────────────

void
DlpUalPlan::convertA_QuantOperations()
{
    if (!m_a_quant)
        return;

    const auto& param = *m_a_quant;

    if (!m_metadata->a_quant_op) {
        m_metadata->a_quant_op = new dlp_quant_op_t{};
    }
    auto& a_quant         = *m_metadata->a_quant_op;
    a_quant.quant_op_kind = DLP_QUANT_OP_QUANTIZE;
    a_quant.src_type      = getStorageType(m_a_type);
    a_quant.dst_type      = DLP_S8;

    // Scale factor assignment
    if (param.hasA_PreOpScaleFactor()) {
        if (!a_quant.quant_scale_factors) {
            a_quant.quant_scale_factors = new dlp_qparam_t{};
        }
        auto* scl = a_quant.quant_scale_factors;
        scl->data = convertMatrixToPtr(*param.getA_PreOpScaleFactor());
        scl->len  = param.getA_PreOpScaleFactor()->getCols();
        scl->stor_type =
            getStorageType(param.getA_PreOpScaleFactor()->getMatrixType());
        scl->outer_dim =
            getScalarOrVectorDim(scl->len, DLP_PARAM_DIM_PER_TOKEN);
    }

    // Zero point assignment
    if (param.hasA_PreOpZeroPoint()) {
        if (!a_quant.zero_point) {
            a_quant.zero_point = new dlp_qparam_t{};
        }
        auto* zp = a_quant.zero_point;
        zp->data = convertMatrixToPtr(*param.getA_PreOpZeroPoint());
        zp->len  = param.getA_PreOpZeroPoint()->getCols();
        zp->stor_type =
            getStorageType(param.getA_PreOpZeroPoint()->getMatrixType());
        zp->outer_dim = getScalarOrVectorDim(zp->len, DLP_PARAM_DIM_PER_TOKEN);
    }

    // Scale factor assignment
    if (param.hasA_PostOpScaleFactor()) {
        if (!a_quant.dequant_scale_factors) {
            a_quant.dequant_scale_factors = new dlp_qparam_t{};
        }
        auto* scl = a_quant.dequant_scale_factors;
        scl->data = convertMatrixToPtr(*param.getA_PostOpScaleFactor());
        scl->len  = param.getA_PostOpScaleFactor()->getCols();
        scl->stor_type =
            getStorageType(param.getA_PostOpScaleFactor()->getMatrixType());
        scl->outer_dim =
            getScalarOrVectorDim(scl->len, DLP_PARAM_DIM_PER_TOKEN);
    }

    // Prefer the explicit post-op zero-point when present.
    if (param.hasA_PostOpZeroPoint()) {
        if (!a_quant.zero_point) {
            a_quant.zero_point = new dlp_qparam_t{};
        }
        auto* zp = a_quant.zero_point;
        zp->data = convertMatrixToPtr(*param.getA_PostOpZeroPoint());
        zp->len  = param.getA_PostOpZeroPoint()->getCols();
        zp->stor_type =
            getStorageType(param.getA_PostOpZeroPoint()->getMatrixType());
        zp->outer_dim = getScalarOrVectorDim(zp->len, DLP_PARAM_DIM_PER_TOKEN);
    }
}

void
DlpUalPlan::convertB_QuantOperations()
{
    if (!m_b_quant)
        return;

    const auto& param = *m_b_quant;

    if (!m_metadata->b_quant_op) {
        m_metadata->b_quant_op = new dlp_quant_op_t{};
    }
    auto& b_quant         = *m_metadata->b_quant_op;
    b_quant.quant_op_kind = DLP_QUANT_OP_QUANTIZE;
    b_quant.src_type      = getStorageType(m_b_type);
    // Destination type describes the compute operand selected by the API/ISA,
    // not the final C storage type.
    b_quant.dst_type = getQuantComputeDstType(m_b_type);

    // Scale factor assignment for B pre-quant
    if (param.hasB_PreOpScaleFactor()) {
        if (!b_quant.quant_scale_factors) {
            b_quant.quant_scale_factors = new dlp_qparam_t{};
        }
        auto* scl = b_quant.quant_scale_factors;
        scl->data = convertMatrixToPtr(*param.getB_PreOpScaleFactor());
        scl->len  = param.getB_PreOpScaleFactor()->getCols();
        scl->stor_type =
            getStorageType(param.getB_PreOpScaleFactor()->getMatrixType());
        scl->outer_dim =
            getScalarOrVectorDim(scl->len, DLP_PARAM_DIM_PER_CHANNEL);
    }

    // Zero point assignment for B pre-quant
    if (param.hasB_PreOpZeroPoint()) {
        if (!b_quant.zero_point) {
            b_quant.zero_point = new dlp_qparam_t{};
        }
        auto* zp = b_quant.zero_point;
        zp->data = convertMatrixToPtr(*param.getB_PreOpZeroPoint());
        zp->len  = param.getB_PreOpZeroPoint()->getCols();
        zp->stor_type =
            getStorageType(param.getB_PreOpZeroPoint()->getMatrixType());
        zp->outer_dim =
            getScalarOrVectorDim(zp->len, DLP_PARAM_DIM_PER_CHANNEL);
    }

    // Scale factor assignment for B post-quant
    if (param.hasB_PostOpScaleFactor()) {
        if (!b_quant.dequant_scale_factors) {
            b_quant.dequant_scale_factors = new dlp_qparam_t{};
        }
        auto* scl = b_quant.dequant_scale_factors;
        scl->data = convertMatrixToPtr(*param.getB_PostOpScaleFactor());
        scl->len  = param.getB_PostOpScaleFactor()->getCols();
        scl->stor_type =
            getStorageType(param.getB_PostOpScaleFactor()->getMatrixType());
        scl->outer_dim =
            getScalarOrVectorDim(scl->len, DLP_PARAM_DIM_PER_CHANNEL);
    }

    // Prefer the explicit post-op zero-point when present.
    if (param.hasB_PostOpZeroPoint()) {
        if (!b_quant.zero_point) {
            b_quant.zero_point = new dlp_qparam_t{};
        }
        auto* zp = b_quant.zero_point;
        zp->data = convertMatrixToPtr(*param.getB_PostOpZeroPoint());
        zp->len  = param.getB_PostOpZeroPoint()->getCols();
        zp->stor_type =
            getStorageType(param.getB_PostOpZeroPoint()->getMatrixType());
        zp->outer_dim =
            getScalarOrVectorDim(zp->len, DLP_PARAM_DIM_PER_CHANNEL);
    }
}

void
DlpUalPlan::convertWOQOperations()
{
    if (!m_woq)
        return;

    const auto& param = *m_woq;

    if (!m_metadata->b_quant_op) {
        m_metadata->b_quant_op = new dlp_quant_op_t{};
    }
    auto& b_quant         = *m_metadata->b_quant_op;
    b_quant.quant_op_kind = DLP_QUANT_OP_DEQUANTIZE;
    b_quant.src_type      = getStorageType(m_b_type);
    b_quant.dst_type      = DLP_BF16;

    // Scale factor assignment for B matrix
    if (param.hasB_ScaleFactor()) {
        if (!b_quant.dequant_scale_factors) {
            b_quant.dequant_scale_factors = new dlp_qparam_t{};
        }
        auto* scl = b_quant.dequant_scale_factors;
        scl->data = convertMatrixToPtr(*param.getB_ScaleFactor());
        scl->len  = param.getB_ScaleFactor()->getCols();
        scl->stor_type =
            getStorageType(param.getB_ScaleFactor()->getMatrixType());
        scl->outer_dim =
            getScalarOrVectorDim(scl->len, DLP_PARAM_DIM_PER_CHANNEL);
    }

    // Zero point assignment for B matrix
    if (param.hasB_ZeroPoint()) {
        if (!b_quant.zero_point) {
            b_quant.zero_point = new dlp_qparam_t{};
        }
        auto* zp      = b_quant.zero_point;
        zp->data      = convertMatrixToPtr(*param.getB_ZeroPoint());
        zp->len       = param.getB_ZeroPoint()->getCols();
        zp->stor_type = getStorageType(param.getB_ZeroPoint()->getMatrixType());
        zp->outer_dim =
            getScalarOrVectorDim(zp->len, DLP_PARAM_DIM_PER_CHANNEL);
    }

    b_quant.group_size = 0; // one group over full K
}

void
DlpUalPlan::convertGroupScaleOperations()
{
    if (!m_group_scale)
        return;

    const auto& param = *m_group_scale;

    if (!param.hasAScaleFactor() || !param.hasBScaleFactor()) {
        throw std::runtime_error(
            "convertGroupScaleOperations: both A and B scale factors "
            "are required");
    }

    if (!m_metadata->a_quant_op) {
        m_metadata->a_quant_op = new dlp_quant_op_t{};
    }
    if (!m_metadata->b_quant_op) {
        m_metadata->b_quant_op = new dlp_quant_op_t{};
    }
    auto& a_quant         = *m_metadata->a_quant_op;
    auto& b_quant         = *m_metadata->b_quant_op;
    a_quant.quant_op_kind = DLP_QUANT_OP_QUANTIZE;
    b_quant.quant_op_kind = DLP_QUANT_OP_QUANTIZE;
    a_quant.src_type      = DLP_S8;
    b_quant.src_type      = DLP_S8;
    a_quant.dst_type      = DLP_S8;
    b_quant.dst_type      = DLP_S8;

    md_t gs            = m_group_scale->getGroupSize();
    a_quant.group_size = gs;
    b_quant.group_size = gs;

    // In the unified metadata API the granularity mode is conveyed to DLP via
    // each scale factor's own outer_dim (PER_TOKEN for A, PER_CHANNEL for B);
    // the frame carries those per-matrix dims through to the kernels. A and B
    // granularity are independent.
    AScaleGranularity a_gran = m_group_scale->getAGranularity();
    BScaleGranularity b_gran = m_group_scale->getBGranularity();

    // The sym_quant kernel indexes scale factors as 2D arrays:
    //   A scale: a_scale[row * num_groups + group], needing m * num_groups
    //   elems B scale: b_scale[group * n + col],          needing num_groups *
    //   n elems
    // With group_size=0 (defaults to k), num_groups=1, so A needs m and B needs
    // n elements. When scale_factor_len=1 (scalar), we must broadcast to the
    // full expected size so the kernel does not read out of bounds.
    // NOTE: Broadcast only handles scalar (len=1) → per-dim expansion.
    // Per-row/per-col → per-group tiling is not implemented here; the ref
    // path handles that case. The DLP path relies on the kernel's own
    // scale factor indexing for correctly-sized arrays.
    md_t m = getM();
    md_t n = getN();
    md_t k = getK();

    // DLP converts group_size=0 to a group_size of K internally, hence using
    // eff_gs for calculating expected scale factor lengths.
    // NOTE: The metadata passed to DLP still receives the original group_size
    // and not the effective group size.
    md_t eff_gs = (gs == 0) ? k : gs;
    // If K == 0, set num_groups = 0
    md_t ng = (k == 0) ? 0 : (k + eff_gs - 1) / eff_gs; // number of groups

    // Per-matrix group counts. A PER_TOKEN -> one scale per row (a_ng=1);
    // B PER_CHANNEL -> one scale per column (b_ng=1); else per-group.
    md_t a_ng = (a_gran == AScaleGranularity::PerToken) ? 1 : ng;
    md_t b_ng = (b_gran == BScaleGranularity::PerChannel) ? 1 : ng;

    // Set A scale factor
    if (param.hasAScaleFactor()) {
        if (!a_quant.dequant_scale_factors) {
            a_quant.dequant_scale_factors = new dlp_qparam_t{};
        }
        auto* scl      = a_quant.dequant_scale_factors;
        md_t  a_sf_len = param.getAScaleFactor()->getCols();
        scl->stor_type =
            getStorageType(param.getAScaleFactor()->getMatrixType());
        // PER_TOKEN collapses A to one scale per row regardless of the number
        // of K-groups, so it takes precedence over the per-group layout.
        scl->outer_dim =
            (a_gran == AScaleGranularity::PerToken) ? DLP_PARAM_DIM_PER_TOKEN
            : (ng > 1)
                ? DLP_PARAM_DIM_PER_GROUP
                : getScalarOrVectorDim(a_sf_len, DLP_PARAM_DIM_PER_TOKEN);

        md_t eff_a_sf_len = m * a_ng;
        if (a_sf_len == 1 && eff_a_sf_len > 1) {
            // Broadcast scalar to m elements (one per row, single group)
            size_t elem_size = (scl->stor_type == DLP_BF16) ? sizeof(int16_t)
                                                            : sizeof(float);
            m_broadcast_a_scale.resize(eff_a_sf_len * elem_size);
            const uint8_t* src = static_cast<const uint8_t*>(
                convertMatrixToPtr(*param.getAScaleFactor()));
            for (md_t i = 0; i < eff_a_sf_len; ++i) {
                std::copy(src, src + elem_size,
                          m_broadcast_a_scale.data() + i * elem_size);
            }
            scl->data = m_broadcast_a_scale.data();
            scl->len  = eff_a_sf_len;
        } else {
            scl->data = convertMatrixToPtr(*param.getAScaleFactor());
            scl->len  = a_sf_len;
        }
    }

    // Set B scale factor
    if (param.hasBScaleFactor()) {
        if (!b_quant.dequant_scale_factors) {
            b_quant.dequant_scale_factors = new dlp_qparam_t{};
        }
        auto* scl      = b_quant.dequant_scale_factors;
        md_t  b_sf_len = param.getBScaleFactor()->getCols();
        scl->stor_type =
            getStorageType(param.getBScaleFactor()->getMatrixType());
        // PER_CHANNEL collapses B to one scale per column regardless of the
        // number of K-groups, so it takes precedence over the per-group layout.
        scl->outer_dim =
            (b_gran == BScaleGranularity::PerChannel)
                ? DLP_PARAM_DIM_PER_CHANNEL
            : (ng > 1)
                ? DLP_PARAM_DIM_PER_GROUP
                : getScalarOrVectorDim(b_sf_len, DLP_PARAM_DIM_PER_CHANNEL);

        md_t eff_b_sf_len = n * b_ng;
        if (b_sf_len == 1 && eff_b_sf_len > 1) {
            // Broadcast scalar to n elements (one per column, single group)
            size_t elem_size = (scl->stor_type == DLP_BF16) ? sizeof(int16_t)
                                                            : sizeof(float);
            m_broadcast_b_scale.resize(eff_b_sf_len * elem_size);
            const uint8_t* src = static_cast<const uint8_t*>(
                convertMatrixToPtr(*param.getBScaleFactor()));
            for (md_t i = 0; i < eff_b_sf_len; ++i) {
                std::copy(src, src + elem_size,
                          m_broadcast_b_scale.data() + i * elem_size);
            }
            scl->data = m_broadcast_b_scale.data();
            scl->len  = eff_b_sf_len;
        } else {
            scl->data = convertMatrixToPtr(*param.getBScaleFactor());
            scl->len  = b_sf_len;
        }
    }
}

// ─── Static Helper Methods ──────────────────────────────────────────────────

void*
DlpUalPlan::convertMatrixToPtr(const Matrix& matrix)
{
    return matrix.getData();
}

DLP_TYPE
DlpUalPlan::getStorageType(MatrixType type)
{
    switch (type) {
        case MatrixType::f32:
            return DLP_F32;
        case MatrixType::fp16:
            return DLP_F16;
        case MatrixType::bf16:
            return DLP_BF16;
        case MatrixType::s8:
            return DLP_S8;
        case MatrixType::s4:
            return DLP_S4;
        case MatrixType::u4:
            return DLP_U4;
        case MatrixType::u8:
            return DLP_U8;
        case MatrixType::s32:
            return DLP_S32;
        default:
            return DLP_INVALID;
    }
}

DLP_ELT_ALGO_TYPE
DlpUalPlan::getElementWiseAlgoType(ElementWiseOperation op)
{
    switch (op) {
        case ElementWiseOperation::Relu:
            return RELU;
        case ElementWiseOperation::Prelu:
            return PRELU;
        case ElementWiseOperation::Gelu_Tanh:
            return GELU_TANH;
        case ElementWiseOperation::Gelu_Erf:
            return GELU_ERF;
        case ElementWiseOperation::Clip:
            return CLIP;
        case ElementWiseOperation::Swish:
            return SWISH;
        case ElementWiseOperation::Tanh:
            return TANH;
        case ElementWiseOperation::Sigmoid:
            return SIGMOID;
        case ElementWiseOperation::Mish:
            return MISH;
        default:
            throw std::runtime_error("Unsupported element-wise operation");
    }
}

DLP_POST_OP_TYPE
DlpUalPlan::getPostOpType(OperationType type)
{
    switch (type) {
        case OperationType::ElementWise:
            return ELTWISE;
        case OperationType::Bias:
            return BIAS;
        case OperationType::Scale:
            return SCALE;
        case OperationType::MatAdd:
            return MATRIX_ADD;
        case OperationType::MatMul:
            return MATRIX_MUL;
        default:
            throw std::runtime_error("Unsupported operation type");
    }
}

DLP_PARAM_DIM_TYPE
DlpUalPlan::getParamDim(ParamDim dim)
{
    switch (dim) {
        case ParamDim::PerTensor:
            return DLP_PARAM_DIM_PER_TENSOR;
        case ParamDim::PerChannel:
            return DLP_PARAM_DIM_PER_CHANNEL;
        case ParamDim::PerToken:
            return DLP_PARAM_DIM_PER_TOKEN;
        default:
            throw std::runtime_error("Unsupported ParamDim value");
    }
}

} // namespace dlp::testing::classic
