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

#include "classic/dlp_compat.h"
#include "framework/matrix_view.hh"

#include <cstdlib>
#include <memory>
#include <new>

namespace dlp { namespace testing { namespace framework {

    /**
     * @class ContiguousBatchStorage
     * @brief Allocates a single contiguous aligned buffer for N matrices of
     *        identical physical size.
     *
     * Each matrix occupies exactly bytesPerMatrix() bytes, densely packed.
     *
     * Provides:
     *   - data()           : base pointer for oneDNN 3D memory descriptor
     *   - at(i)            : pointer to i-th matrix (for DLP pointer arrays)
     *   - view(i, ...)     : MatrixView into i-th matrix slot
     *
     * Ownership is via std::unique_ptr with a custom aligned deleter.
     * Intended to be held by std::shared_ptr in BatchGroup so that the
     * backing memory outlives all MatrixView / sliced-Matrix objects.
     */
    class ContiguousBatchStorage
    {
        struct AlignedDeleter
        {
            void operator()(uint8_t* p) const noexcept { dlp_aligned_free(p); }
        };

        std::unique_ptr<uint8_t[], AlignedDeleter> m_data;
        size_t m_count;          ///< number of matrices
        size_t m_bytesPerMatrix; ///< physical bytes per matrix (dense)
        size_t m_alignment;      ///< alignment in bytes
        size_t m_totalBytes; ///< m_count * m_bytesPerMatrix (before rounding)

      public:
        /**
         * @brief Allocate contiguous storage for @p count matrices.
         *
         * @param count          Number of matrices
         * @param bytesPerMatrix Physical bytes consumed by one matrix
         * @param alignment      Alignment for the allocation (must be power of
         * 2)
         *
         * @throws std::bad_alloc if allocation fails
         */
        ContiguousBatchStorage(size_t count,
                               size_t bytesPerMatrix,
                               size_t alignment)
            : m_count(count)
            , m_bytesPerMatrix(bytesPerMatrix)
            , m_alignment(alignment)
            , m_totalBytes(count * bytesPerMatrix)
        {
            // count * bytesPerMatrix above is computed in the initializer
            // list before this body runs, so an overflow there would
            // already have wrapped m_totalBytes to some smaller value by
            // the time we get here -- checking m_totalBytes itself can't
            // detect that. Re-derive the overflow condition independently
            // from the original (un-multiplied) count/bytesPerMatrix via
            // division instead, so a huge count with a huge-looking total
            // that actually wrapped to something small (and would
            // therefore under-allocate while count()/at() still assume the
            // full requested size) is rejected rather than silently
            // producing an undersized buffer.
            const size_t max_size = static_cast<size_t>(-1);
            if (bytesPerMatrix != 0 && count > max_size / bytesPerMatrix) {
                throw std::bad_alloc();
            }

            if (m_totalBytes == 0) {
                return;
            }

            // Guard the alignment-rounding addition below from overflowing
            // too: if m_totalBytes is within (alignment - 1) of SIZE_MAX,
            // "m_totalBytes + alignment - 1" itself wraps around to a tiny
            // value, again yielding an undersized allocation.
            if (alignment == 0 || (alignment & (alignment - 1)) != 0
                || m_totalBytes > max_size - (alignment - 1)) {
                throw std::bad_alloc();
            }

            // Round up to alignment boundary
            size_t allocSize =
                (m_totalBytes + alignment - 1) & ~(alignment - 1);
            auto* raw =
                static_cast<uint8_t*>(dlp_aligned_alloc(alignment, allocSize));

            if (!raw) {
                throw std::bad_alloc();
            }

            m_data.reset(raw);
        }

        /// Base pointer to the contiguous buffer (for oneDNN 3D descriptor).
        void* data() const { return m_data.get(); }

        /// Pointer to the i-th matrix slot.
        void* at(size_t i) const { return m_data.get() + i * m_bytesPerMatrix; }

        /// Physical bytes per matrix.
        size_t bytesPerMatrix() const { return m_bytesPerMatrix; }

        /// Number of matrix slots.
        size_t count() const { return m_count; }

        /// Total allocation size in bytes.
        size_t totalBytes() const { return m_totalBytes; }

        /**
         * @brief Create a non-owning MatrixView into the i-th slot.
         *
         * @param i          Matrix index (0-based)
         * @param rows       Number of rows
         * @param cols       Number of columns
         * @param type       Element data type
         * @param layout     Memory layout (default: ROW_MAJOR)
         * @param leadingDim Leading dimension (0 = auto)
         * @param transposed Whether logically transposed
         * @param reordered  Whether the matrix is reordered
         *
         * @return A MatrixView that does NOT own the memory.
         */
        MatrixView view(size_t       i,
                        md_t         rows,
                        md_t         cols,
                        MatrixType   type,
                        MatrixLayout layout     = MatrixLayout::ROW_MAJOR,
                        md_t         leadingDim = 0,
                        bool         transposed = false,
                        bool         reordered  = false) const
        {
            return MatrixView(rows, cols, type, static_cast<uint8_t*>(at(i)),
                              m_bytesPerMatrix, layout, leadingDim, transposed,
                              reordered, m_alignment);
        }
    };

}}} // namespace dlp::testing::framework
