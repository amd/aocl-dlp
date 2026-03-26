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
#include <limits>
#include <vector>

namespace dlp::testing::framework {

struct PreparedBatchGemmArgs
{
    MatrixType a_type   = MatrixType::f32;
    MatrixType b_type   = MatrixType::f32;
    MatrixType c_type   = MatrixType::f32;
    MatrixType acc_type = MatrixType::f32;

    md_t   group_count    = 0;
    size_t total_matrices = 0;

    std::vector<char> order;
    std::vector<char> transa;
    std::vector<char> transb;

    std::vector<md_t> m;
    std::vector<md_t> n;
    std::vector<md_t> k;

    std::vector<double> alpha;
    std::vector<double> beta;

    std::vector<char> mem_format_a;
    std::vector<char> mem_format_b;

    std::vector<md_t> lda;
    std::vector<md_t> ldb;
    std::vector<md_t> ldc;

    std::vector<md_t> group_size;

    std::vector<const void*> a_ptrs;
    std::vector<const void*> b_ptrs;
    std::vector<void*>       c_ptrs;

    // Backend-specific metadata (raw pointers for zero-overhead access)
    std::vector<void*> backend_metadata;

    // Storage to manage metadata lifetime (keeps objects alive)
    std::vector<std::shared_ptr<void>> backend_metadata_storage;

    // Post-operation params per group (stored for lifetime management)
    std::vector<std::vector<std::unique_ptr<IOperationParam>>> post_ops;

    // Pre-converted alpha/beta for zero-overhead type casting
    // These are populated during metadata preparation based on data types
    std::vector<float>   alpha_f32;
    std::vector<float>   beta_f32;
    std::vector<int32_t> alpha_s32;
    std::vector<int32_t> beta_s32;

    void clear() { *this = PreparedBatchGemmArgs{}; }
};

inline bool
validate_type_consistency(const Matrix& A,
                          const Matrix& B,
                          const Matrix& C,
                          bool&         first_group,
                          MatrixType&   a_type,
                          MatrixType&   b_type,
                          MatrixType&   c_type)
{
    if (first_group) {
        a_type      = A.getMatrixType();
        b_type      = B.getMatrixType();
        c_type      = C.getMatrixType();
        first_group = false;
        return true;
    }

    return A.getMatrixType() == a_type && B.getMatrixType() == b_type
           && C.getMatrixType() == c_type;
}

inline UALError
prepare_batch_gemm_args(const std::vector<BatchGroup>& groups,
                        MatrixType                     accType,
                        PreparedBatchGemmArgs&         out)
{
    out.clear();

    if (groups.empty()) {
        return UALError::UAL_FAILURE;
    }

    if (groups.size() > static_cast<size_t>(std::numeric_limits<md_t>::max())) {
        return UALError::UAL_NOT_SUPPORTED;
    }

    out.group_count = static_cast<md_t>(groups.size());
    out.acc_type    = accType;

    out.order.reserve(groups.size());
    out.transa.reserve(groups.size());
    out.transb.reserve(groups.size());
    out.m.reserve(groups.size());
    out.n.reserve(groups.size());
    out.k.reserve(groups.size());
    out.alpha.reserve(groups.size());
    out.beta.reserve(groups.size());
    out.mem_format_a.reserve(groups.size());
    out.mem_format_b.reserve(groups.size());
    out.lda.reserve(groups.size());
    out.ldb.reserve(groups.size());
    out.ldc.reserve(groups.size());
    out.group_size.reserve(groups.size());

    bool       first_group  = true;
    MatrixType first_a_type = MatrixType::f32;
    MatrixType first_b_type = MatrixType::f32;
    MatrixType first_c_type = MatrixType::f32;

    size_t total_matrices = 0;

    for (const auto& group : groups) {
        if (!group.validate()) {
            return UALError::UAL_FAILURE;
        }

        if (group.A_matrices.size()
            > static_cast<size_t>(std::numeric_limits<md_t>::max())) {
            return UALError::UAL_NOT_SUPPORTED;
        }

        // For empty groups (group_size=0), we still need to populate arrays
        // with default values For non-empty groups, validate type consistency
        // and extract matrix properties
        if (!group.A_matrices.empty()) {
            const Matrix& A = group.A_matrices.front();
            const Matrix& B = group.B_matrices.front();
            const Matrix& C = group.C_matrices.front();

            if (!validate_type_consistency(A, B, C, first_group, first_a_type,
                                           first_b_type, first_c_type)) {
                return UALError::UAL_NOT_SUPPORTED;
            }

            out.order.push_back(A.getLayout() == MatrixLayout::ROW_MAJOR ? 'r'
                                                                         : 'c');
            out.transa.push_back(A.isTransposed() ? 't' : 'n');

            char transb_char = B.isReordered() ? 'n'
                                               : (B.isTransposed() ? 't' : 'n');
            out.transb.push_back(transb_char);

            out.mem_format_a.push_back(to_aocl_mem_format(group.memFormatA));
            out.mem_format_b.push_back(to_aocl_mem_format(group.memFormatB));

            out.lda.push_back(A.getLeadingDimension());
            out.ldb.push_back(B.getLeadingDimension());
            out.ldc.push_back(C.getLeadingDimension());
        } else {
            // Empty group: use default values
            out.order.push_back('r');        // Default row-major
            out.transa.push_back('n');       // Default no transpose
            out.transb.push_back('n');       // Default no transpose
            out.mem_format_a.push_back('n'); // Default unpacked
            out.mem_format_b.push_back('n'); // Default unpacked
            out.lda.push_back(group.k);      // Use group dims for defaults
            out.ldb.push_back(group.n);
            out.ldc.push_back(group.n);
        }

        out.m.push_back(group.m);
        out.n.push_back(group.n);
        out.k.push_back(group.k);

        out.alpha.push_back(group.alpha);
        out.beta.push_back(group.beta);

        md_t group_sz = static_cast<md_t>(group.A_matrices.size());
        out.group_size.push_back(group_sz);

        total_matrices += group.A_matrices.size();
    }

    out.a_ptrs.reserve(total_matrices);
    out.b_ptrs.reserve(total_matrices);
    out.c_ptrs.reserve(total_matrices);
    out.post_ops.reserve(groups.size());

    for (const auto& group : groups) {
        for (iter_t i = 0; i < static_cast<md_t>(group.A_matrices.size());
             ++i) {
            out.a_ptrs.push_back(static_cast<const void*>(
                group.A_matrices[i].getMatrixData().getMatrixPtr()));
            out.b_ptrs.push_back(static_cast<const void*>(
                group.B_matrices[i].getMatrixData().getMatrixPtr()));
            out.c_ptrs.push_back(
                group.C_matrices[i].getMatrixData().getMatrixPtr());
        }

        // Store post-operations per group (clone params)
        std::vector<std::unique_ptr<IOperationParam>> cloned_params;
        for (const auto& p : group.post_op_params) {
            if (p) {
                cloned_params.push_back(p->clone());
            }
        }
        out.post_ops.push_back(std::move(cloned_params));
    }

    out.total_matrices = total_matrices;
    out.a_type         = first_a_type;
    out.b_type         = first_b_type;
    out.c_type         = first_c_type;

    return UALError::UAL_SUCCESS;
}

} // namespace dlp::testing::framework
