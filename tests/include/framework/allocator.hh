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

/**
 * @file allocator.hh
 * @brief Optional guard-page allocator for out-of-bounds (OOB) detection.
 *
 * Opt-in via the environment variable DLP_TEST_GUARD_PAGE=1. When enabled,
 * matrix buffers are allocated via mmap with a trailing PROT_NONE guard page
 * and the logical buffer is placed flush against that guard page, so any
 * read/write past the end of the buffer faults immediately (SIGSEGV) instead
 * of silently landing in heap padding.
 *
 * The default (env unset / != 1) leaves the allocator disabled so normal test
 * runs are completely unaffected.
 *
 * These hooks live in their own translation unit (allocator.cc) so that both
 * MatrixMemory::allocateBytes (in types.hh) and Matrix::allocateAlignedMemory
 * (in matrix.cc) can share a single implementation.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace dlp { namespace testing { namespace framework {

    /**
     * @brief Whether guard-page allocation mode is enabled.
     * @return true if DLP_TEST_GUARD_PAGE=1 in the environment.
     *
     * The result is cached on first call.
     */
    bool guard_page_enabled();

    /**
     * @brief Allocate a buffer flush against a trailing PROT_NONE guard
     *        page so any over-read/write past the end faults immediately.
     *
     * Allocation metadata (a small header with a magic sentinel plus the
     * mmap base/length) is stored immediately BEFORE the returned pointer,
     * glibc-style, so guard_free() needs no external registry.
     *
     * @param sizeBytes Requested usable size in bytes.
     * @param alignment Required alignment (0 = none). In guard mode the
     *        buffer end is flushed against a page-aligned guard page, so
     *        the returned pointer is only guaranteed to honor `alignment`
     *        when it divides the page size (and is therefore <= page size);
     *        alignments that cannot be satisfied are rejected with
     *        std::invalid_argument.
     * @return Pointer to the usable buffer.
     * @throws std::bad_alloc on mmap/mprotect failure (callers must handle
     *         this), std::invalid_argument for unsatisfiable alignment.
     */
    uint8_t* guard_alloc(size_t sizeBytes, size_t alignment = 0);

    /**
     * @brief Free a pointer previously returned by guard_alloc().
     *
     * Reads the inline metadata header immediately preceding @p ptr. If the
     * magic sentinel matches, the underlying mmap region is unmapped and
     * true is returned. If it does not match (i.e. @p ptr was not produced
     * by guard_alloc), the pointer is left untouched and false is returned
     * — this keeps the guard run free of spurious "bad-free" reports so the
     * only fault surfaced is a genuine library OOB SIGSEGV.
     *
     * @param ptr Pointer to free (may be null).
     * @return true if it was a guard allocation (and was unmapped).
     */
    bool guard_free(uint8_t* ptr);

}}} // namespace dlp::testing::framework
