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
 * @file allocator.cc
 * @brief Implementation of the opt-in guard-page allocator (see allocator.hh).
 *
 * DESIGN (inline metadata, glibc-style):
 *   Instead of an external registry keyed by the returned pointer, each
 *   allocation stores a small header — a magic sentinel plus the mmap base and
 *   length — immediately BEFORE the pointer handed back to the caller. free
 *   reads that header back:
 *     - magic matches  -> it is our allocation; unmap the recorded region.
 *     - magic mismatch -> not a guard allocation; leave it alone (return
 *                         false). Reading the bytes just before an ordinary
 *                         heap pointer is safe (that memory is mapped) and the
 *                         sentinel simply won't match, so no spurious
 *                         "bad-free" is ever produced.
 *   This removes the mutex + unordered_map + range-fallback scan entirely while
 *   preserving the "only genuine dlp_* SIGSEGV surfaces" property.
 *
 *   Buffer layout inside the single mmap region (low -> high address):
 *     [ base ...padding... ][ header ][ usable bytes ][ PROT_NONE guard page ]
 *   The usable buffer END is flush against the guard page so an over-read/write
 *   past the end faults immediately.
 */

#include "framework/allocator.hh"

#include <cstring>   // std::memcpy
#include <new>       // std::bad_alloc
#include <stdexcept> // std::invalid_argument

#if defined(__linux__)
#include <cstdlib>    // std::getenv
#include <sys/mman.h> // mmap, mprotect, munmap
#include <unistd.h>   // sysconf
#endif

// Detect AddressSanitizer (GCC defines __SANITIZE_ADDRESS__; Clang exposes it
// through __has_feature, and newer Clang also defines __SANITIZE_ADDRESS__).
#if defined(__SANITIZE_ADDRESS__)
#define DLP_ASAN_ENABLED 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define DLP_ASAN_ENABLED 1
#endif
#endif

#if defined(__linux__) && defined(DLP_ASAN_ENABLED)
// ASan runtime query: returns non-null if any byte in [beg, beg+size) is
// poisoned. Used to avoid probing the inline header of a non-guard pointer,
// whose preceding bytes may lie in a poisoned redzone.
extern "C" const void*
__asan_region_is_poisoned(const void* beg, size_t size);
#endif

namespace dlp { namespace testing { namespace framework {

#if defined(__linux__)

    namespace {
        // Sentinel identifying a guard allocation ("DLPGUARM").
        constexpr uint64_t kGuardMagic = 0x444C50475541524Dull;

        // Metadata stored immediately before the returned pointer.
        struct GuardHeader
        {
            uint64_t magic;
            void*    base;  // mmap base to unmap
            size_t   total; // mmap length
        };

        // Bytes reserved for the header. A fixed 64-byte slot comfortably
        // holds GuardHeader and keeps the layout simple; it sits below the
        // returned pointer and therefore does not affect its alignment.
        constexpr size_t kHeaderBytes = 64;
    } // namespace

    bool guard_page_enabled()
    {
        static const bool enabled = []() {
            const char* v = std::getenv("DLP_TEST_GUARD_PAGE");
            return v != nullptr && v[0] == '1';
        }();
        return enabled;
    }

    uint8_t* guard_alloc(size_t sizeBytes, size_t alignment)
    {
        const long pageLong = sysconf(_SC_PAGESIZE);
        if (pageLong <= 0) {
            throw std::bad_alloc();
        }
        const size_t page = static_cast<size_t>(pageLong);

        // In guard mode the buffer end is flushed against the page-aligned
        // guard page (dataPtr = guardStart - usable), so the returned
        // pointer only honors `alignment` when the alignment divides the
        // page size (and is therefore also <= page size). Reject alignments
        // that cannot be satisfied rather than silently returning a
        // misaligned buffer.
        if (alignment > 0 && (alignment > page || (page % alignment) != 0)) {
            throw std::invalid_argument(
                "Alignment must be <= page size and divide the page size "
                "in guard mode");
        }

        size_t usable = (sizeBytes == 0) ? 1 : sizeBytes;
        if (alignment > 0) {
            usable = (usable + alignment - 1) & ~(alignment - 1);
        }

        // Reserve room for the inline metadata header that precedes the
        // returned pointer, then round the data region up to whole pages
        // and add one trailing guard page.
        const size_t needed    = usable + kHeaderBytes;
        const size_t dataPages = ((needed + page - 1) / page) * page;
        const size_t total     = dataPages + page; // + guard page

        void* base = mmap(nullptr, total, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) {
            throw std::bad_alloc();
        }
        uint8_t* guardStart = static_cast<uint8_t*>(base) + dataPages;
        if (mprotect(guardStart, page, PROT_NONE) != 0) {
            munmap(base, total);
            throw std::bad_alloc();
        }

        // Flush the buffer end against the guard page.
        uint8_t* dataPtr = guardStart - usable;

        // Write the header immediately before the returned pointer. Since
        // dataPages >= usable + kHeaderBytes, (dataPtr - kHeaderBytes) is
        // guaranteed to lie within the mapped, writable region. Use memcpy
        // rather than a placed struct write: dataPtr - kHeaderBytes is not
        // guaranteed to satisfy alignof(GuardHeader), so a typed store would
        // be alignment/aliasing UB.
        GuardHeader hdr{ kGuardMagic, base, total };
        std::memcpy(dataPtr - kHeaderBytes, &hdr, sizeof(hdr));

        return dataPtr;
    }

    bool guard_free(uint8_t* ptr)
    {
        if (!ptr) {
            return false;
        }
#if defined(DLP_ASAN_ENABLED)
        // Under ASan, probing the header of a non-guard pointer (e.g. an
        // ordinary heap buffer) may read into a poisoned redzone and raise a
        // false positive. If the header region is poisoned it cannot be one of
        // ours, so bail out before the memcpy probe.
        if (__asan_region_is_poisoned(ptr - kHeaderBytes, sizeof(GuardHeader))
            != nullptr) {
            return false;
        }
#endif
        // Read the header via memcpy (the source address is not guaranteed to
        // be GuardHeader-aligned, so a typed load would be UB).
        GuardHeader hdr;
        std::memcpy(&hdr, ptr - kHeaderBytes, sizeof(hdr));
        if (hdr.magic != kGuardMagic) {
            // Not one of ours (e.g. an ordinary heap / externally-wrapped
            // buffer). Leave it untouched — no false "bad-free".
            return false;
        }
        // Poison the magic in-place to catch accidental double-free.
        const uint64_t poison = 0;
        std::memcpy(ptr - kHeaderBytes, &poison, sizeof(poison));
        munmap(hdr.base, hdr.total);
        return true;
    }

#else // !__linux__

    bool guard_page_enabled()
    {
        return false;
    }
    uint8_t* guard_alloc(size_t, size_t)
    {
        throw std::bad_alloc();
    }
    bool guard_free(uint8_t*)
    {
        return false;
    }

#endif // __linux__

}}} // namespace dlp::testing::framework
