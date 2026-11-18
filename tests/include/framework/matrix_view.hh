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

namespace dlp { namespace testing { namespace framework {

    /**
     * @class MatrixView
     * @brief Non-owning view into externally-managed matrix memory.
     *
     * Derived from Matrix so it is usable anywhere a Matrix is accepted.
     * The key difference: MatrixView does NOT free the backing memory on
     * destruction.
     *
     * @par Slicing safety
     * When a MatrixView is std::move'd into a std::vector<Matrix>, the base
     * Matrix move constructor runs and transfers m_ownsMemory=false. The
     * resulting Matrix in the vector will NOT free the backing memory.
     *
     * @par Lifetime
     * The caller must ensure the backing storage (e.g. ContiguousBatchStorage)
     * outlives all MatrixView instances and any Matrix objects created by
     * moving from them.
     */
    class MatrixView : public Matrix
    {
      public:
        /**
         * @brief Construct a non-owning view into an existing buffer.
         *
         * @param rows       Number of rows
         * @param cols       Number of columns
         * @param type       Element data type
         * @param data       Pointer into backing storage (NOT owned)
         * @param dataSize   Size in bytes of this matrix's region
         * @param layout     Memory layout (default: ROW_MAJOR)
         * @param leadingDim Leading dimension (0 = auto)
         * @param transposed Whether logically transposed
         * @param reordered  Whether the matrix is reordered
         * @param alignment  Alignment hint (for metadata only, NOT for freeing)
         */
        MatrixView(md_t         rows,
                   md_t         cols,
                   MatrixType   type,
                   uint8_t*     data,
                   size_t       dataSize,
                   MatrixLayout layout     = MatrixLayout::ROW_MAJOR,
                   md_t         leadingDim = 0,
                   bool         transposed = false,
                   bool         reordered  = false,
                   size_t       alignment  = 0)
            : Matrix(rows,
                     cols,
                     type,
                     data,
                     dataSize,
                     layout,
                     // The base Matrix constructor's "auto-compute leading
                     // dimension from rows/cols" sentinel is -1, not 0 (see
                     // tests/framework/matrix.cc). MatrixView documents 0 as
                     // its own "auto" default, so translate it here;
                     // otherwise a caller relying on that default would get
                     // a literal leadingDim of 0 — a zero stride that
                     // overlaps every row and produces invalid GEMM
                     // descriptors downstream.
                     leadingDim == 0 ? -1 : leadingDim,
                     transposed,
                     reordered,
                     alignment)
        {
            m_ownsMemory = false;
        }

        /// Destructor is a no-op — memory is not owned.
        ~MatrixView() override = default;

        // MatrixView should not be copied (would create an owning Matrix).
        // Use the base Matrix copy constructor explicitly if a deep copy is
        // intended.
        MatrixView(const MatrixView&)            = delete;
        MatrixView& operator=(const MatrixView&) = delete;

        // Move is fine — transfers the non-owning pointer.
        MatrixView(MatrixView&&) noexcept            = default;
        MatrixView& operator=(MatrixView&&) noexcept = default;
    };

}}} // namespace dlp::testing::framework
