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

#ifndef AOCL_GEMM_CHECK_H
#define AOCL_GEMM_CHECK_H

#include <stdbool.h>
#include <stdint.h>

#include "classic/dlp_errors.h"

/*
 * Maximum allowed value for any single GEMM dimension (m, n, k).
 * Set to INT32_MAX so that any pairwise product (m*k, k*n, m*n)
 * is guaranteed to fit in int64_t without overflow, and internal
 * buffer-size computations remain safe.
 */
#define DLP_MAX_GEMM_DIM ((int64_t)INT32_MAX)

/* Maximum allowed group size. Set to INT32_MAX so that group_size can be
 * safely used in loop bounds and indexed operations that assume 32-bit
 * indices, without risking overflow when promoted to int64_t.
 */
#define DLP_MAX_GROUP_SIZE ((int64_t)INT32_MAX)

#define AOCL_ERROR_CHECK(op_str, arg_pos, err_no)                              \
    if (arg_pos != 0) {                                                        \
        char print_msg[256];                                                   \
        snprintf(print_msg, sizeof(print_msg),                                 \
                 "** On entry to %6s, parameter number %2i had an illegal "    \
                 "value, error code: %i",                                      \
                 op_str, arg_pos, (int)err_no);                                \
        dlp_print_msg(print_msg, __FILE__, __LINE__);                          \
    }

#define AOCL_REORDER_BUF_SIZE_CHECK(op_str, order, trans, mat_type, k, n,      \
                                    err_no)                                    \
    {                                                                          \
        int32_t arg_pos = 0;                                                   \
        err_no          = DLP_CLSC_SUCCESS;                                    \
        if ((order != 'r') && (order != 'R') && (order != 'c')                 \
            && (order != 'C')) {                                               \
            arg_pos = 1;                                                       \
            err_no  = DLP_CLSC_INVALID_ORDER;                                  \
        } else if ((trans != 'n') && (trans != 'N') && (trans != 't')          \
                   && (trans != 'T')) {                                        \
            arg_pos = 2;                                                       \
            err_no  = DLP_CLSC_INVALID_TRANSPOSE;                              \
        } else if ((mat_type != 'A') && (mat_type != 'B') && (mat_type != 'W') \
                   && (mat_type != 'a') && (mat_type != 'b')                   \
                   && (mat_type != 'w')) {                                     \
            arg_pos = 3;                                                       \
            err_no  = DLP_CLSC_INVALID_MATRIX_TYPE;                            \
        } else if (k <= 0) {                                                   \
            arg_pos = 4;                                                       \
            err_no  = DLP_CLSC_INVALID_MATRIX_DIMENSION;                       \
        } else if (n <= 0) {                                                   \
            arg_pos = 5;                                                       \
            err_no  = DLP_CLSC_INVALID_MATRIX_DIMENSION;                       \
        }                                                                      \
                                                                               \
        AOCL_ERROR_CHECK(op_str, arg_pos, err_no);                             \
    }

#define AOCL_REORDER_CHECK(op_str, order, trans, mat_type, input_buf_addr,     \
                           reorder_buf_addr, k, n, ldb, err_no)                \
    {                                                                          \
        int32_t arg_pos = 0;                                                   \
        err_no          = DLP_CLSC_SUCCESS;                                    \
        bool col_stored = FALSE, row_stored = FALSE;                           \
        bool notrans_b = FALSE, trans_b = FALSE;                               \
                                                                               \
        col_stored = (order == 'c') || (order == 'C');                         \
        row_stored = (order == 'r') || (order == 'R');                         \
                                                                               \
        notrans_b = (trans == 'n') || (trans == 'N');                          \
        trans_b   = (trans == 't') || (trans == 'T');                          \
                                                                               \
        if ((order != 'r') && (order != 'R') && (order != 'c')                 \
            && (order != 'C')) {                                               \
            arg_pos = 1;                                                       \
            err_no  = DLP_CLSC_INVALID_ORDER;                                  \
        } else if ((trans != 'n') && (trans != 'N') && (trans != 't')          \
                   && (trans != 'T')) {                                        \
            arg_pos = 2;                                                       \
            err_no  = DLP_CLSC_INVALID_TRANSPOSE;                              \
        } else if ((mat_type != 'A') && (mat_type != 'B') && (mat_type != 'W') \
                   && (mat_type != 'a') && (mat_type != 'b')                   \
                   && (mat_type != 'w')) {                                     \
            arg_pos = 3;                                                       \
            err_no  = DLP_CLSC_INVALID_MATRIX_TYPE;                            \
        } else if (input_buf_addr == NULL) {                                   \
            arg_pos = 4;                                                       \
            err_no  = DLP_CLSC_NULL_POINTER;                                   \
        } else if (reorder_buf_addr == NULL) {                                 \
            arg_pos = 5;                                                       \
            err_no  = DLP_CLSC_NULL_POINTER;                                   \
        } else if (k <= 0) {                                                   \
            arg_pos = 6;                                                       \
            err_no  = DLP_CLSC_INVALID_MATRIX_DIMENSION;                       \
        } else if (n <= 0) {                                                   \
            arg_pos = 7;                                                       \
            err_no  = DLP_CLSC_INVALID_MATRIX_DIMENSION;                       \
        } else if (row_stored                                                  \
                   && ((notrans_b && (ldb < n)) || (trans_b && (ldb < k)))) {  \
            arg_pos = 8;                                                       \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        } else if (col_stored                                                  \
                   && ((notrans_b && (ldb < k)) || (trans_b && (ldb < n)))) {  \
            arg_pos = 8;                                                       \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        }                                                                      \
                                                                               \
        AOCL_ERROR_CHECK(op_str, arg_pos, err_no);                             \
    }

#define AOCL_UNREORDER_CHECK(op_str, order, mat_type, reorder_buf_addr,        \
                             output_buf_addr, k, n, ldb, err_no)               \
    {                                                                          \
        int32_t arg_pos = 0;                                                   \
        err_no          = DLP_CLSC_SUCCESS;                                    \
        bool col_stored = FALSE, row_stored = FALSE;                           \
                                                                               \
        col_stored = (order == 'c') || (order == 'C');                         \
        row_stored = (order == 'r') || (order == 'R');                         \
                                                                               \
        if ((order != 'r') && (order != 'R') && (order != 'c')                 \
            && (order != 'C')) {                                               \
            arg_pos = 1;                                                       \
            err_no  = DLP_CLSC_INVALID_ORDER;                                  \
        } else if ((mat_type != 'A') && (mat_type != 'B') && (mat_type != 'W') \
                   && (mat_type != 'a') && (mat_type != 'b')                   \
                   && (mat_type != 'w')) {                                     \
            arg_pos = 2;                                                       \
            err_no  = DLP_CLSC_INVALID_MATRIX_TYPE;                            \
        } else if (reorder_buf_addr == NULL) {                                 \
            arg_pos = 3;                                                       \
            err_no  = DLP_CLSC_NULL_POINTER;                                   \
        } else if (output_buf_addr == NULL) {                                  \
            arg_pos = 4;                                                       \
            err_no  = DLP_CLSC_NULL_POINTER;                                   \
        } else if (k <= 0) {                                                   \
            arg_pos = 5;                                                       \
            err_no  = DLP_CLSC_INVALID_MATRIX_DIMENSION;                       \
        } else if (n <= 0) {                                                   \
            arg_pos = 6;                                                       \
            err_no  = DLP_CLSC_INVALID_MATRIX_DIMENSION;                       \
        } else if (row_stored && (ldb < n)) {                                  \
            arg_pos = 7;                                                       \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        } else if (col_stored && (ldb < k)) {                                  \
            arg_pos = 7;                                                       \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        }                                                                      \
                                                                               \
        AOCL_ERROR_CHECK(op_str, arg_pos, err_no);                             \
    }

#define AOCL_GEMM_CHECK(op_str, order, transa, transb, m, n, k, a, lda,        \
                        mtag_a, b, ldb, mtag_b, c, ldc, err_no)                \
    {                                                                          \
        int32_t arg_pos = 0;                                                   \
        err_no          = DLP_CLSC_SUCCESS;                                    \
        bool col_stored, row_stored;                                           \
        bool nota, notb, ta, tb;                                               \
                                                                               \
        col_stored = (order == 'c') || (order == 'C');                         \
        row_stored = (order == 'r') || (order == 'R');                         \
                                                                               \
        nota = (transa == 'n') || (transa == 'N');                             \
        notb = (transb == 'n') || (transb == 'N');                             \
                                                                               \
        ta = (transa == 't') || (transa == 'T');                               \
        tb = (transb == 't') || (transb == 'T');                               \
                                                                               \
        if ((order != 'r') && (order != 'R') && (order != 'c')                 \
            && (order != 'C')) {                                               \
            arg_pos = 1;                                                       \
            err_no  = DLP_CLSC_INVALID_ORDER;                                  \
        } else if ((transa != 'n') && (transa != 'N') && (transa != 't')       \
                   && (transa != 'T')) {                                       \
            arg_pos = 2;                                                       \
            err_no  = DLP_CLSC_INVALID_TRANSPOSE;                              \
        } else if ((transb != 'n') && (transb != 'N') && (transb != 't')       \
                   && (transb != 'T')) {                                       \
            arg_pos = 3;                                                       \
            err_no  = DLP_CLSC_INVALID_TRANSPOSE;                              \
        } else if ((m <= 0) || (m > DLP_MAX_GEMM_DIM)) {                       \
            arg_pos = 4;                                                       \
            err_no  = DLP_CLSC_INVALID_MATRIX_DIMENSION;                       \
        } else if ((n <= 0) || (n > DLP_MAX_GEMM_DIM)) {                       \
            arg_pos = 5;                                                       \
            err_no  = DLP_CLSC_INVALID_MATRIX_DIMENSION;                       \
        } else if ((k <= 0) || (k > DLP_MAX_GEMM_DIM)) {                       \
            arg_pos = 6;                                                       \
            err_no  = DLP_CLSC_INVALID_MATRIX_DIMENSION;                       \
        } else if (a == NULL) {                                                \
            arg_pos = 7;                                                       \
            err_no  = DLP_CLSC_NULL_POINTER;                                   \
        } else if (row_stored && ((nota && (lda < k)) || (ta && (lda < m)))) { \
            arg_pos = 9;                                                       \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        } else if (col_stored && ((nota && (lda < m)) || (ta && (lda < k)))) { \
            arg_pos = 9;                                                       \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        } else if ((mtag_a != 'n') && (mtag_a != 'N') && (mtag_a != 'p')       \
                   && (mtag_a != 'P') && (mtag_a != 'r') && (mtag_a != 'R')) { \
            arg_pos = 10;                                                      \
            err_no  = DLP_CLSC_INVALID_MEMORY_TAG;                             \
        } else if (b == NULL) {                                                \
            arg_pos = 11;                                                      \
            err_no  = DLP_CLSC_NULL_POINTER;                                   \
        } else if (row_stored && ((notb && (ldb < n)) || (tb && (ldb < k)))) { \
            arg_pos = 12;                                                      \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        } else if (col_stored && ((notb && (ldb < k)) || (tb && (ldb < n)))) { \
            arg_pos = 12;                                                      \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        } else if ((mtag_b != 'n') && (mtag_b != 'N') && (mtag_b != 'p')       \
                   && (mtag_b != 'P') && (mtag_b != 'r') && (mtag_b != 'R')) { \
            arg_pos = 13;                                                      \
            err_no  = DLP_CLSC_INVALID_MEMORY_TAG;                             \
        } else if (c == NULL) {                                                \
            arg_pos = 15;                                                      \
            err_no  = DLP_CLSC_NULL_POINTER;                                   \
        } else if (row_stored && (ldc < n)) {                                  \
            arg_pos = 16;                                                      \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        } else if (col_stored && (ldc < m)) {                                  \
            arg_pos = 16;                                                      \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        }                                                                      \
                                                                               \
        AOCL_ERROR_CHECK(op_str, arg_pos, err_no);                             \
    }

#define AOCL_BATCH_GEMM_CHECK(op_str, order, transa, transb, group_count_idx,  \
                              group_size, m, n, k, a, lda, mtag_a, b, ldb,     \
                              mtag_b, c, ldc, err_no)                          \
    {                                                                          \
        int32_t arg_pos = 0;                                                   \
        err_no          = DLP_CLSC_SUCCESS;                                    \
        bool col_stored, row_stored;                                           \
        bool nota, notb, ta, tb;                                               \
                                                                               \
        col_stored = (order == 'c') || (order == 'C');                         \
        row_stored = (order == 'r') || (order == 'R');                         \
                                                                               \
        nota = (transa == 'n') || (transa == 'N');                             \
        notb = (transb == 'n') || (transb == 'N');                             \
                                                                               \
        ta = (transa == 't') || (transa == 'T');                               \
        tb = (transb == 't') || (transb == 'T');                               \
                                                                               \
        if ((order != 'r') && (order != 'R') && (order != 'c')                 \
            && (order != 'C')) {                                               \
            arg_pos = 1;                                                       \
            err_no  = DLP_CLSC_INVALID_ORDER;                                  \
        } else if ((transa != 'n') && (transa != 'N') && (transa != 't')       \
                   && (transa != 'T')) {                                       \
            arg_pos = 2;                                                       \
            err_no  = DLP_CLSC_INVALID_TRANSPOSE;                              \
        } else if ((transb != 'n') && (transb != 'N') && (transb != 't')       \
                   && (transb != 'T')) {                                       \
            arg_pos = 3;                                                       \
            err_no  = DLP_CLSC_INVALID_TRANSPOSE;                              \
        } else if ((m <= 0) || (m > DLP_MAX_GEMM_DIM)) {                       \
            arg_pos = 4;                                                       \
            err_no  = DLP_CLSC_INVALID_MATRIX_DIMENSION;                       \
        } else if ((n <= 0) || (n > DLP_MAX_GEMM_DIM)) {                       \
            arg_pos = 5;                                                       \
            err_no  = DLP_CLSC_INVALID_MATRIX_DIMENSION;                       \
        } else if ((k <= 0) || (k > DLP_MAX_GEMM_DIM)) {                       \
            arg_pos = 6;                                                       \
            err_no  = DLP_CLSC_INVALID_MATRIX_DIMENSION;                       \
        } else if (a == NULL) {                                                \
            arg_pos = 8;                                                       \
            err_no  = DLP_CLSC_NULL_POINTER;                                   \
        } else if (row_stored && ((nota && (lda < k)) || (ta && (lda < m)))) { \
            arg_pos = 10;                                                      \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        } else if (col_stored && ((nota && (lda < m)) || (ta && (lda < k)))) { \
            arg_pos = 10;                                                      \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        } else if ((mtag_a != 'n') && (mtag_a != 'N') && (mtag_a != 'p')       \
                   && (mtag_a != 'P') && (mtag_a != 'r') && (mtag_a != 'R')) { \
            arg_pos = 11;                                                      \
            err_no  = DLP_CLSC_INVALID_MEMORY_TAG;                             \
        } else if (b == NULL) {                                                \
            arg_pos = 12;                                                      \
            err_no  = DLP_CLSC_NULL_POINTER;                                   \
        } else if (row_stored && ((notb && (ldb < n)) || (tb && (ldb < k)))) { \
            arg_pos = 13;                                                      \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        } else if (col_stored && ((notb && (ldb < k)) || (tb && (ldb < n)))) { \
            arg_pos = 13;                                                      \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        } else if ((mtag_b != 'n') && (mtag_b != 'N') && (mtag_b != 'p')       \
                   && (mtag_b != 'P') && (mtag_b != 'r') && (mtag_b != 'R')) { \
            arg_pos = 14;                                                      \
            err_no  = DLP_CLSC_INVALID_MEMORY_TAG;                             \
        } else if (c == NULL) {                                                \
            arg_pos = 16;                                                      \
            err_no  = DLP_CLSC_NULL_POINTER;                                   \
        } else if (row_stored && (ldc < n)) {                                  \
            arg_pos = 17;                                                      \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        } else if (col_stored && (ldc < m)) {                                  \
            arg_pos = 17;                                                      \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        } else if ((group_size <= 0) || (group_size > DLP_MAX_GROUP_SIZE)) {   \
            arg_pos = 16;                                                      \
            err_no  = DLP_CLSC_INVALID_GROUP_DIMENSION;                        \
        } else if ((group_count_idx < 0)                                       \
                   || (group_count_idx > DLP_MAX_GROUP_SIZE)) {                \
            arg_pos = 15;                                                      \
            err_no  = DLP_CLSC_INVALID_GROUP_DIMENSION;                        \
        }                                                                      \
        AOCL_ERROR_CHECK(op_str, arg_pos, err_no);                             \
    }

#define AOCL_UTIL_ELTWISE_OPS_CHECK(op_str, order, transa, transb, m, n, a,    \
                                    lda, b, ldb, err_no)                       \
    {                                                                          \
        int32_t arg_pos = 0;                                                   \
        err_no          = DLP_CLSC_SUCCESS;                                    \
        bool col_stored, row_stored;                                           \
        bool nota, notb, ta, tb;                                               \
                                                                               \
        col_stored = (order == 'c') || (order == 'C');                         \
        row_stored = (order == 'r') || (order == 'R');                         \
                                                                               \
        nota = (transa == 'n') || (transa == 'N');                             \
        notb = (transb == 'n') || (transb == 'N');                             \
                                                                               \
        ta = (transa == 't') || (transa == 'T');                               \
        tb = (transb == 't') || (transb == 'T');                               \
                                                                               \
        if ((order != 'r') && (order != 'R') && (order != 'c')                 \
            && (order != 'C')) {                                               \
            arg_pos = 1;                                                       \
            err_no  = DLP_CLSC_INVALID_ORDER;                                  \
        } else if ((transa != 'n') && (transa != 'N') && (transa != 't')       \
                   && (transa != 'T')) {                                       \
            arg_pos = 2;                                                       \
            err_no  = DLP_CLSC_INVALID_TRANSPOSE;                              \
        } else if ((transb != 'n') && (transb != 'N') && (transb != 't')       \
                   && (transb != 'T')) {                                       \
            arg_pos = 3;                                                       \
            err_no  = DLP_CLSC_INVALID_TRANSPOSE;                              \
        } else if (m <= 0) {                                                   \
            arg_pos = 4;                                                       \
            err_no  = DLP_CLSC_INVALID_MATRIX_DIMENSION;                       \
        } else if (n <= 0) {                                                   \
            arg_pos = 5;                                                       \
            err_no  = DLP_CLSC_INVALID_MATRIX_DIMENSION;                       \
        } else if (a == NULL) {                                                \
            arg_pos = 6;                                                       \
            err_no  = DLP_CLSC_NULL_POINTER;                                   \
        } else if (row_stored && ((nota && (lda < n)) || (ta && (lda < m)))) { \
            arg_pos = 7;                                                       \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        } else if (col_stored && ((nota && (lda < m)) || (ta && (lda < n)))) { \
            arg_pos = 7;                                                       \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        } else if (b == NULL) {                                                \
            arg_pos = 8;                                                       \
            err_no  = DLP_CLSC_NULL_POINTER;                                   \
        } else if (row_stored && ((notb && (ldb < n)) || (tb && (ldb < m)))) { \
            arg_pos = 9;                                                       \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        } else if (col_stored && ((notb && (ldb < m)) || (tb && (ldb < n)))) { \
            arg_pos = 9;                                                       \
            err_no  = DLP_CLSC_INVALID_LEADING_DIMENSION;                      \
        }                                                                      \
                                                                               \
        AOCL_ERROR_CHECK(op_str, arg_pos, err_no);                             \
    }

#endif // AOCL_GEMM_CHECK_H
