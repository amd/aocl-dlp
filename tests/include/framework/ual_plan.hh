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

#include "framework/matrix.hh"
#include "framework/operation.hh"
#include "framework/ual.hh"
#include <cassert>
#include <memory>
#include <stdexcept>
#include <vector>

namespace dlp::testing::framework {

/**
 * @class IUalPlan
 * @brief Unified plan for GEMM execution: configure → prepare() → execute()
 *
 * Replaces the old IOperation + gemm() split-responsibility model.
 * All configuration (types, layouts, post-ops, quant) is owned by the plan.
 * prepare() pre-builds all backend state. execute() is the hot-loop call.
 */
class IUalPlan
{
  public:
    virtual ~IUalPlan() = default;

    // ─── Matrix Configuration ───────────────────────────────────
    void setDimensions(md_t m, md_t n, md_t k)
    {
        m_m = m;
        m_n = n;
        m_k = k;
    }

    void setTypes(MatrixType a_type,
                  MatrixType b_type,
                  MatrixType c_type,
                  MatrixType acc_type)
    {
        m_a_type   = a_type;
        m_b_type   = b_type;
        m_c_type   = c_type;
        m_acc_type = acc_type;
    }

    void setLayouts(MatrixLayout a_layout,
                    MatrixLayout b_layout,
                    MatrixLayout c_layout)
    {
        m_a_layout = a_layout;
        m_b_layout = b_layout;
        m_c_layout = c_layout;
    }

    void setTranspose(bool transA, bool transB)
    {
        m_transA = transA;
        m_transB = transB;
    }

    void setLeadingDims(md_t lda, md_t ldb, md_t ldc)
    {
        m_lda = lda;
        m_ldb = ldb;
        m_ldc = ldc;
    }

    void setMemFormats(char memFormatA, char memFormatB)
    {
        m_memFormatA = memFormatA;
        m_memFormatB = memFormatB;
    }

    // ─── Scaling ────────────────────────────────────────────────
    void setAlpha(double alpha) { m_alpha = alpha; }
    void setBeta(double beta) { m_beta = beta; }

    // ─── Quantisation (kernel config, NOT post-ops) ─────────────
    void setAQuant(std::unique_ptr<AQuantParam> param)
    {
        m_a_quant = std::move(param);
    }

    void setBQuant(std::unique_ptr<BQuantParam> param)
    {
        m_b_quant = std::move(param);
    }

    void setWOQ(std::unique_ptr<WOQParam> param) { m_woq = std::move(param); }

    void setGroupScale(std::unique_ptr<GroupScaleParam> param)
    {
        m_group_scale = std::move(param);
    }

    // ─── Post-Operations (real fusion ops only) ─────────────────
    void addPostOp(std::unique_ptr<IOperationParam> param)
    {
        if (!param) {
            return;
        }
        if (m_prepared) {
            throw std::runtime_error(
                "Cannot add post-ops after prepare(). "
                "Create a new plan to modify configuration.");
        }
        if (!isPostOp(param->getType())) {
            throw std::runtime_error(
                "addPostOp() only accepts post-op types "
                "(ElementWise, Bias, Scale, MatAdd, MatMul). "
                "Use setAQuant/setBQuant/setWOQ for quant params.");
        }
        m_post_ops.push_back(std::move(param));
    }

    // ─── Buffer Binding ──────────────────────────────────────────
    /**
     * @brief Bind matrix buffers for subsequent execute() calls
     *
     * Captures raw data pointers and leading dimensions from Matrix objects.
     * Can be called after prepare() without re-preparing. setBuffers() also
     * stores const Matrix pointers for backends (e.g. RefUalPlan) that need
     * full Matrix access.
     */
    void setBuffers(const Matrix& A, const Matrix& B, Matrix& C)
    {
        m_a_ptr       = A.getMatrixData().getMatrixPtr();
        m_b_ptr       = B.getMatrixData().getMatrixPtr();
        m_c_ptr       = C.getMatrixData().getMatrixPtr();
        m_buf_lda     = A.getLeadingDimension();
        m_buf_ldb     = B.getLeadingDimension();
        m_buf_ldc     = C.getLeadingDimension();
        m_a_matrix    = &A;
        m_b_matrix    = &B;
        m_c_matrix    = &C;
        m_buffers_set = true;
    }

    /**
     * @brief Bind raw buffer pointers for subsequent execute() calls
     *
     * Low-level overload for internal use or when raw pointers are available.
     */
    void setBuffers(void* a, void* b, void* c, md_t lda, md_t ldb, md_t ldc)
    {
        m_a_ptr       = a;
        m_b_ptr       = b;
        m_c_ptr       = c;
        m_buf_lda     = lda;
        m_buf_ldb     = ldb;
        m_buf_ldc     = ldc;
        m_a_matrix    = nullptr;
        m_b_matrix    = nullptr;
        m_c_matrix    = nullptr;
        m_buffers_set = true;
    }

    // ─── Lifecycle ──────────────────────────────────────────────
    virtual void     prepare() = 0;
    virtual UALError execute() = 0;

    /**
     * @brief Convenience: setBuffers + execute in one call
     *
     * Equivalent to calling setBuffers(A, B, C) followed by execute().
     */
    UALError executeWith(const Matrix& A, const Matrix& B, Matrix& C)
    {
        setBuffers(A, B, C);
        return execute();
    }

    // ─── Convenience: configure from Matrix objects ─────────────
    void configureFrom(const Matrix& A,
                       const Matrix& B,
                       const Matrix& C,
                       MatrixType    accType,
                       double        alpha = 1.0,
                       double        beta  = 0.0)
    {
        setDimensions(A.getEffectiveRows(), B.getEffectiveCols(),
                      A.getEffectiveCols());
        setTypes(A.getMatrixType(), B.getMatrixType(), C.getMatrixType(),
                 accType);
        setLayouts(A.getLayout(), B.getLayout(), C.getLayout());
        setTranspose(A.isTransposed(), B.isTransposed());
        setLeadingDims(A.getLeadingDimension(), B.getLeadingDimension(),
                       C.getLeadingDimension());

        char memA = A.isPacked() ? 'p' : (A.isReordered() ? 'r' : 'n');
        char memB = B.isPacked() ? 'p' : (B.isReordered() ? 'r' : 'n');
        setMemFormats(memA, memB);

        setAlpha(alpha);
        setBeta(beta);
    }

    // ─── Accessors ──────────────────────────────────────────────
    md_t getM() const { return m_m; }
    md_t getN() const { return m_n; }
    md_t getK() const { return m_k; }

    bool hasPostOps() const { return !m_post_ops.empty(); }
    bool hasAQuant() const { return m_a_quant != nullptr; }
    bool hasBQuant() const { return m_b_quant != nullptr; }
    bool hasWOQ() const { return m_woq != nullptr; }
    bool hasGroupScale() const { return m_group_scale != nullptr; }

    const std::vector<std::unique_ptr<IOperationParam>>& getPostOps() const
    {
        return m_post_ops;
    }

  protected:
    // Lifecycle state
    bool m_prepared    = false;
    bool m_buffers_set = false;

    // Buffer pointers (set by setBuffers)
    void* m_a_ptr   = nullptr;
    void* m_b_ptr   = nullptr;
    void* m_c_ptr   = nullptr;
    md_t  m_buf_lda = 0, m_buf_ldb = 0, m_buf_ldc = 0;

    // Full Matrix pointers for backends that need Matrix access
    const Matrix* m_a_matrix = nullptr;
    const Matrix* m_b_matrix = nullptr;
    Matrix*       m_c_matrix = nullptr;

    // Matrix shape
    md_t m_m = 0, m_n = 0, m_k = 0;
    // Types
    MatrixType m_a_type   = MatrixType::f32;
    MatrixType m_b_type   = MatrixType::f32;
    MatrixType m_c_type   = MatrixType::f32;
    MatrixType m_acc_type = MatrixType::f32;
    // Layouts
    MatrixLayout m_a_layout = MatrixLayout::ROW_MAJOR;
    MatrixLayout m_b_layout = MatrixLayout::ROW_MAJOR;
    MatrixLayout m_c_layout = MatrixLayout::ROW_MAJOR;
    // Transpose
    bool m_transA = false, m_transB = false;
    // Leading dims
    md_t m_lda = 0, m_ldb = 0, m_ldc = 0;
    // Memory formats
    char m_memFormatA = 'n', m_memFormatB = 'n';
    // Scaling
    double m_alpha = 1.0, m_beta = 0.0;
    // Quantisation (kernel config)
    std::unique_ptr<AQuantParam>     m_a_quant;
    std::unique_ptr<BQuantParam>     m_b_quant;
    std::unique_ptr<WOQParam>        m_woq;
    std::unique_ptr<GroupScaleParam> m_group_scale;
    // Post-ops (fusion only)
    std::vector<std::unique_ptr<IOperationParam>> m_post_ops;
};

} // namespace dlp::testing::framework
