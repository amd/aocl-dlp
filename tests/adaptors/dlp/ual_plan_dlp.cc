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

        // Clean up manually allocated memory for A matrix quantization scale
        // factors and zero points
        if (m_metadata->a_pre_quant) {
            if (m_metadata->a_pre_quant->scl) {
                delete m_metadata->a_pre_quant->scl;
            }
            if (m_metadata->a_pre_quant->zp) {
                delete m_metadata->a_pre_quant->zp;
            }
            delete m_metadata->a_pre_quant;
        }
        if (m_metadata->a_post_quant) {
            if (m_metadata->a_post_quant->scl) {
                delete m_metadata->a_post_quant->scl;
            }
            if (m_metadata->a_post_quant->zp) {
                delete m_metadata->a_post_quant->zp;
            }
            delete m_metadata->a_post_quant;
        }

        // Clean up manually allocated memory for B matrix quantization scale
        // factors and zero points
        if (m_metadata->b_pre_quant) {
            if (m_metadata->b_pre_quant->scl) {
                delete m_metadata->b_pre_quant->scl;
            }
            if (m_metadata->b_pre_quant->zp) {
                delete m_metadata->b_pre_quant->zp;
            }
            delete m_metadata->b_pre_quant;
        }
        if (m_metadata->b_post_quant) {
            if (m_metadata->b_post_quant->scl) {
                delete m_metadata->b_post_quant->scl;
            }
            if (m_metadata->b_post_quant->zp) {
                delete m_metadata->b_post_quant->zp;
            }
            delete m_metadata->b_post_quant;
        }

        // Clean up manually allocated memory for B matrix WOQ quantization
        if (m_metadata->pre_ops) {
            if (m_metadata->pre_ops->b_scl) {
                delete m_metadata->pre_ops->b_scl;
            }
            if (m_metadata->pre_ops->b_zp) {
                delete m_metadata->pre_ops->b_zp;
            }
            delete m_metadata->pre_ops;
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
    buildSequenceVector();

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

    // Compute type code for dispatch
    m_type_code = encodeTypes(m_a_type, m_b_type, m_c_type, m_acc_type);

    // Compute layout and transpose chars
    m_layout_char = m_a_layout == MatrixLayout::ROW_MAJOR ? 'r' : 'c';
    m_transA_char = m_transA ? 't' : 'n';
    m_transB_char = m_transB ? 't' : 'n';

    // Resolve transB: if B is reordered, force to 'n'
    m_transB_resolved = (m_memFormatB == 'r') ? 'n' : m_transB_char;

    // Pre-cast alpha/beta
    m_alpha_f32 = static_cast<float>(m_alpha);
    m_beta_f32  = static_cast<float>(m_beta);
    m_alpha_s32 = static_cast<int32_t>(m_alpha);
    m_beta_s32  = static_cast<int32_t>(m_beta);

    // Resolve dispatch function - captures all pre-computed args.
    // Only a_ptr/b_ptr/c_ptr and lda/ldb/ldc come from setBuffers().
    // Local copies for lambda capture (avoids capturing 'this')
    char            layout  = m_layout_char;
    char            transA  = m_transA_char;
    char            transB  = m_transB_resolved;
    md_t            m_dim   = m_m;
    md_t            n_dim   = m_n;
    md_t            k_dim   = m_k;
    float           alpha_f = m_alpha_f32;
    float           beta_f  = m_beta_f32;
    int32_t         alpha_i = m_alpha_s32;
    int32_t         beta_i  = m_beta_s32;
    char            memA    = m_memFormatA;
    char            memB    = m_memFormatB;
    dlp_metadata_t* meta    = m_metadata;

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
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_s8s8s32of32(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<int8_t*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<float*>(c), ldc, meta);
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
            m_dispatch = [=](void* a, md_t lda, void* b, md_t ldb, void* c,
                             md_t ldc) {
                aocl_gemm_s8s8s32obf16(
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_i,
                    reinterpret_cast<int8_t*>(a), lda, memA,
                    reinterpret_cast<int8_t*>(b), ldb, memB, beta_i,
                    reinterpret_cast<bfloat16*>(c), ldc, meta);
            };
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
                    layout, transA, transB, m_dim, n_dim, k_dim, alpha_f,
                    reinterpret_cast<float16*>(a), lda, memA,
                    reinterpret_cast<float16*>(b), ldb, memB, beta_f,
                    reinterpret_cast<float16*>(c), ldc, meta);
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
            m_metadata->scale[i].sf = new dlp_sf_t;
            m_metadata->scale[i].sf->scale_factor =
                convertMatrixToPtr(*param.getScaleFactor());
            m_metadata->scale[i].sf->scale_factor_len =
                param.getScaleFactor()->getCols();
            m_metadata->scale[i].sf->scale_factor_type =
                getStorageType(param.getScaleFactor()->getMatrixType());
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
        m_metadata->bias[i].stor_type =
            getStorageType(param.getBias().getMatrixType());

        // Set scale factor if provided
        if (param.hasScaleFactor()) {
            m_metadata->bias[i].sf = new dlp_sf_t;
            m_metadata->bias[i].sf->scale_factor =
                convertMatrixToPtr(*param.getScaleFactor());
            m_metadata->bias[i].sf->scale_factor_len =
                param.getScaleFactor()->getCols();
            m_metadata->bias[i].sf->scale_factor_type =
                getStorageType(param.getScaleFactor()->getMatrixType());
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
            m_metadata->matrix_add[i].sf = new dlp_sf_t;
            m_metadata->matrix_add[i].sf->scale_factor =
                convertMatrixToPtr(*param.getScaleFactor());
            m_metadata->matrix_add[i].sf->scale_factor_len =
                param.getScaleFactor()->getCols();
            m_metadata->matrix_add[i].sf->scale_factor_type =
                getStorageType(param.getScaleFactor()->getMatrixType());
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
            m_metadata->matrix_mul[i].sf = new dlp_sf_t;
            m_metadata->matrix_mul[i].sf->scale_factor =
                convertMatrixToPtr(*param.getScaleFactor());
            m_metadata->matrix_mul[i].sf->scale_factor_len =
                param.getScaleFactor()->getCols();
            m_metadata->matrix_mul[i].sf->scale_factor_type =
                getStorageType(param.getScaleFactor()->getMatrixType());
        }
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

    // Ensure the quant structure is allocated and zeroed
    if (!m_metadata->a_pre_quant) {
        m_metadata->a_pre_quant = new dlp_quant_op;
        std::memset(m_metadata->a_pre_quant, 0, sizeof(dlp_quant_op));
        m_metadata->a_pre_quant->symmetric = true;
    }
    m_metadata->a_pre_op_seq_length = 1;

    if (!m_metadata->a_post_quant) {
        m_metadata->a_post_quant = new dlp_quant_op;
        std::memset(m_metadata->a_post_quant, 0, sizeof(dlp_quant_op));
        m_metadata->a_post_quant->symmetric = true;
    }
    m_metadata->a_post_op_seq_length = 1;

    // Scale factor assignment
    if (param.hasA_PreOpScaleFactor()) {
        if (!m_metadata->a_pre_quant->scl) {
            m_metadata->a_pre_quant->scl = new dlp_sf_t;
        }
        auto* scl         = m_metadata->a_pre_quant->scl;
        scl->scale_factor = convertMatrixToPtr(*param.getA_PreOpScaleFactor());
        scl->scale_factor_len = param.getA_PreOpScaleFactor()->getCols();
        scl->scale_factor_type =
            getStorageType(param.getA_PreOpScaleFactor()->getMatrixType());
    }

    // Zero point assignment
    if (param.hasA_PreOpZeroPoint()) {
        if (!m_metadata->a_pre_quant->zp) {
            m_metadata->a_pre_quant->zp = new dlp_zp_t;
        }
        auto* zp           = m_metadata->a_pre_quant->zp;
        zp->zero_point     = convertMatrixToPtr(*param.getA_PreOpZeroPoint());
        zp->zero_point_len = param.getA_PreOpZeroPoint()->getCols();
        zp->zero_point_type =
            getStorageType(param.getA_PreOpZeroPoint()->getMatrixType());
        m_metadata->a_pre_quant->symmetric = false;
    }

    // Scale factor assignment
    if (param.hasA_PostOpScaleFactor()) {
        if (!m_metadata->a_post_quant->scl) {
            m_metadata->a_post_quant->scl = new dlp_sf_t;
        }
        auto* scl         = m_metadata->a_post_quant->scl;
        scl->scale_factor = convertMatrixToPtr(*param.getA_PostOpScaleFactor());
        scl->scale_factor_len = param.getA_PostOpScaleFactor()->getCols();
        scl->scale_factor_type =
            getStorageType(param.getA_PostOpScaleFactor()->getMatrixType());
    }

    // Zero point assignment
    if (param.hasA_PostOpZeroPoint()) {
        if (!m_metadata->a_post_quant->zp) {
            m_metadata->a_post_quant->zp = new dlp_zp_t;
        }
        auto* zp           = m_metadata->a_post_quant->zp;
        zp->zero_point     = convertMatrixToPtr(*param.getA_PostOpZeroPoint());
        zp->zero_point_len = param.getA_PostOpZeroPoint()->getCols();
        zp->zero_point_type =
            getStorageType(param.getA_PostOpZeroPoint()->getMatrixType());
        m_metadata->a_post_quant->symmetric = false;
    }
}

void
DlpUalPlan::convertB_QuantOperations()
{
    if (!m_b_quant)
        return;

    const auto& param = *m_b_quant;

    // Ensure the quant structure is allocated and zeroed
    if (!m_metadata->b_pre_quant) {
        m_metadata->b_pre_quant = new dlp_quant_op;
        std::memset(m_metadata->b_pre_quant, 0, sizeof(dlp_quant_op));
        m_metadata->b_pre_quant->symmetric = true;
    }
    m_metadata->b_pre_op_seq_length = 1;

    if (!m_metadata->b_post_quant) {
        m_metadata->b_post_quant = new dlp_quant_op;
        std::memset(m_metadata->b_post_quant, 0, sizeof(dlp_quant_op));
        m_metadata->b_post_quant->symmetric = true;
    }
    m_metadata->b_post_op_seq_length = 1;

    // Scale factor assignment for B pre-quant
    if (param.hasB_PreOpScaleFactor()) {
        if (!m_metadata->b_pre_quant->scl) {
            m_metadata->b_pre_quant->scl = new dlp_sf_t;
        }
        auto* scl         = m_metadata->b_pre_quant->scl;
        scl->scale_factor = convertMatrixToPtr(*param.getB_PreOpScaleFactor());
        scl->scale_factor_len = param.getB_PreOpScaleFactor()->getCols();
        scl->scale_factor_type =
            getStorageType(param.getB_PreOpScaleFactor()->getMatrixType());
    }

    // Zero point assignment for B pre-quant
    if (param.hasB_PreOpZeroPoint()) {
        if (!m_metadata->b_pre_quant->zp) {
            m_metadata->b_pre_quant->zp = new dlp_zp_t;
        }
        auto* zp           = m_metadata->b_pre_quant->zp;
        zp->zero_point     = convertMatrixToPtr(*param.getB_PreOpZeroPoint());
        zp->zero_point_len = param.getB_PreOpZeroPoint()->getCols();
        zp->zero_point_type =
            getStorageType(param.getB_PreOpZeroPoint()->getMatrixType());
        m_metadata->b_pre_quant->symmetric = false;
    }

    // Scale factor assignment for B post-quant
    if (param.hasB_PostOpScaleFactor()) {
        if (!m_metadata->b_post_quant->scl) {
            m_metadata->b_post_quant->scl = new dlp_sf_t;
        }
        auto* scl         = m_metadata->b_post_quant->scl;
        scl->scale_factor = convertMatrixToPtr(*param.getB_PostOpScaleFactor());
        scl->scale_factor_len = param.getB_PostOpScaleFactor()->getCols();
        scl->scale_factor_type =
            getStorageType(param.getB_PostOpScaleFactor()->getMatrixType());
    }

    // Zero point assignment for B post-quant
    if (param.hasB_PostOpZeroPoint()) {
        if (!m_metadata->b_post_quant->zp) {
            m_metadata->b_post_quant->zp = new dlp_zp_t;
        }
        auto* zp           = m_metadata->b_post_quant->zp;
        zp->zero_point     = convertMatrixToPtr(*param.getB_PostOpZeroPoint());
        zp->zero_point_len = param.getB_PostOpZeroPoint()->getCols();
        zp->zero_point_type =
            getStorageType(param.getB_PostOpZeroPoint()->getMatrixType());
        m_metadata->b_post_quant->symmetric = false;
    }
}

void
DlpUalPlan::convertWOQOperations()
{
    if (!m_woq)
        return;

    const auto& param = *m_woq;

    // Allocate and zero the pre_ops structure
    if (!m_metadata->pre_ops) {
        m_metadata->pre_ops = new dlp_pre_op;
        std::memset(m_metadata->pre_ops, 0, sizeof(dlp_pre_op));
    }

    // Scale factor assignment for B matrix
    if (param.hasB_ScaleFactor()) {
        if (!m_metadata->pre_ops->b_scl) {
            m_metadata->pre_ops->b_scl = new dlp_sf_t;
        }
        auto* scl             = m_metadata->pre_ops->b_scl;
        scl->scale_factor     = convertMatrixToPtr(*param.getB_ScaleFactor());
        scl->scale_factor_len = param.getB_ScaleFactor()->getCols();
        scl->scale_factor_type =
            getStorageType(param.getB_ScaleFactor()->getMatrixType());
    }

    // Zero point assignment for B matrix
    if (param.hasB_ZeroPoint()) {
        if (!m_metadata->pre_ops->b_zp) {
            m_metadata->pre_ops->b_zp = new dlp_zp_t;
        }
        auto* zp           = m_metadata->pre_ops->b_zp;
        zp->zero_point     = convertMatrixToPtr(*param.getB_ZeroPoint());
        zp->zero_point_len = param.getB_ZeroPoint()->getCols();
        zp->zero_point_type =
            getStorageType(param.getB_ZeroPoint()->getMatrixType());
    }

    // Set sequence information
    m_metadata->pre_ops->seq_length = 1; // One WOQ operation
    m_metadata->pre_ops->group_size =
        0; // 0 here indicates no explicit group size for WOQ
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

} // namespace dlp::testing::classic
