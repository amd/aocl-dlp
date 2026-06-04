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

#ifndef DLP_ERRORS_H
#define DLP_ERRORS_H

/**
 * @brief Error codes for DLP classic library operations
 *
 * This enumeration defines the various error conditions that can occur
 * during DLP classic library function calls. Each error code provides
 * specific information about the type of failure encountered.
 */
typedef enum
{
    DLP_CLSC_SUCCESS = 0,           /**< Operation completed successfully */
    DLP_CLSC_FAILURE,               /**< General failure occurred */
    DLP_CLSC_NULL_POINTER,          /**< Null pointer passed as argument */
    DLP_CLSC_UNEXPECTED_VECTOR_DIM, /**< Vector dimension is unexpected or
                                        invalid */
    DLP_CLSC_NOT_SUPPORTED,         /**< Operation or feature not supported */
    DLP_CLSC_INVALID_ORDER,      /**< Invalid memory layout order specified */
    DLP_CLSC_INVALID_TRANSPOSE,  /**< Invalid transpose operation specified */
    DLP_CLSC_INVALID_MEMORY_TAG, /**< Invalid memory tag or format specified */
    DLP_CLSC_INVALID_MATRIX_DIMENSION, /**< Invalid matrix dimension provided */
    DLP_CLSC_INVALID_LEADING_DIMENSION, /**< Invalid leading dimension specified
                                         */
    DLP_CLSC_INVALID_MATRIX_TYPE,       /**< Invalid matrix type specified */
    DLP_CLSC_INVALID_GROUP_DIMENSION, /**< Invalid group dimension specified */
    DLP_CLSC_INVALID_SF_LEN,     /**< Invalid scale factor length specified */
    DLP_CLSC_INVALID_ZP_LEN,     /**< Invalid zero point length specified */
    DLP_CLSC_TYPE_MISMATCH,      /**< Data type mismatch encountered */
    DLP_CLSC_INVALID_JIT_KERNEL, /**< JIT kernel generation failed or no
                                    fallback kernel available */
    DLP_CLSC_INVALID_KERNEL, /**< Static kernel not found for given parameters
                              */
    DLP_CLSC_ERROR_MAX /**< Maximum error code value (for bounds checking) */
} dlp_clsc_err_t;

typedef struct
{
    dlp_clsc_err_t error_code;
    // More error information can be added here.
} dlp_error_hndl_t;

#endif // DLP_ERRORS_H
