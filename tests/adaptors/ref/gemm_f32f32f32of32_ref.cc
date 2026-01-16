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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
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

#include "adaptors/ref/gemm_ref.hh"

namespace dlp::testing::classic::ref {

/**
 * @brief Performs general matrix-matrix multiplication (GEMM) and support
 * of fused element and buffer based operations on C matrix after GEMM
 *
 * This function computes the product of two matrices A and B, and
 * optionally adds the result to a scaled matrix C. All matrices are of type
 * float. The function supports both row-major and column-major storage
 * orders.
 *
 * The operation performed is: C = alpha * A * B + beta * C, where alpha and
 * beta are scalars.
 *
 * @param order CBLAS_ORDER specifies the order of the matrices (row-major
 * or column-major).
 * @param transa CBLAS_TRANSPOSE specifies whether to transpose matrix A.
 * @param transb CBLAS_TRANSPOSE specifies whether to transpose matrix B.
 * @param M Number of rows of matrices A and C.
 * @param N Number of columns of matrices B and C.
 * @param K Number of columns of A and rows of B.
 * @param alpha Scalar multiplier for the product of A and B.
 * @param A Pointer to the first input matrix (A), of size M x K.
 * @param lda Leading dimension of matrix A.
 * @param B Pointer to the second input matrix (B), of size K x N.
 * @param ldb Leading dimension of matrix B.
 * @param beta Scalar multiplier for matrix C.
 * @param C Pointer to the output matrix (C), of size M x N.
 * @param ldc Leading dimension of matrix C.
 * @return void
 */

void
aocl_gemm_f32f32f32of32_ref(const char      order,
                            const char      transa,
                            const char      transb,
                            const md_t      m,
                            const md_t      n,
                            const md_t      k,
                            float           alpha,
                            const float*    A,
                            int             lda,
                            const float*    B,
                            int             ldb,
                            float           beta,
                            float*          C,
                            int             ldc,
                            dlp_metadata_t* metadata)
{

    md_t i, j, l;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            float        sum = 0.0f;
            const float *a_ptr, *b_ptr;
            int          a_stride, b_stride;

            // Setup pointers and strides for A
            if (order == 'R' || order == 'r') {
                if (transa == 'N' || transa == 'n') {
                    a_ptr    = A + i * lda;
                    a_stride = 1;
                } else {
                    a_ptr    = A + i;
                    a_stride = lda;
                }
            } else { // ColMajor
                if (transa == 'N' || transa == 'n') {
                    a_ptr    = A + i;
                    a_stride = lda;
                } else {
                    a_ptr    = A + i * lda;
                    a_stride = 1;
                }
            }

            // Setup pointers and strides for B
            if (order == 'R' || order == 'r') {
                if (transb == 'N' || transb == 'n') {
                    b_ptr    = B + j;
                    b_stride = ldb;
                } else {
                    b_ptr    = B + j * ldb;
                    b_stride = 1;
                }
            } else { // ColMajor
                if (transb == 'N' || transb == 'n') {
                    b_ptr    = B + j * ldb;
                    b_stride = 1;
                } else {
                    b_ptr    = B + j;
                    b_stride = ldb;
                }
            }

            // Main k loop using pointer increments
            const float* a_k = a_ptr;
            const float* b_k = b_ptr;
            for (l = 0; l < k; ++l) {
                sum += (*a_k) * (*b_k);
                a_k += a_stride;
                b_k += b_stride;
            }

            if (beta != 0.0f) {
                // Scale C with beta and write back the result.
                if (order == 'R' || order == 'r')
                    C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                else
                    C[j * ldc + i] = alpha * sum + beta * C[j * ldc + i];
            } else {
                // If beta is zero, do NOT access C to prevent potential NaN
                // propagation.
                if (order == 'R' || order == 'r')
                    C[i * ldc + j] = alpha * sum;
                else
                    C[j * ldc + i] = alpha * sum;
            }
        }
    }
}

} // namespace dlp::testing::classic::ref
