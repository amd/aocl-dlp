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

/*
 * Round-trip and bounds tests for the aocl_unreorder_* APIs.
 *
 * Every un-reorder API is the exact inverse of the matching reorder, so
 * reorder followed by un-reorder must reproduce the input bit for bit. These
 * call the raw C API directly rather than the UAL harness, both so the
 * assertions land on the public contract and so the cases double as worked
 * examples of how to drive the APIs.
 *
 * Buffers are allocated flush against a PROT_NONE guard page, so a read or
 * write even one element past the end of the reordered buffer or the output
 * matrix faults immediately. As in test_dim_bounds.cc, a crash here should be
 * read as "an out-of-bounds access regressed", not as harness flakiness. The
 * faulting shape is printed before each case runs so the log identifies it.
 *
 * Shapes deliberately straddle the internal blocking: multiples and
 * non-multiples of the 16/32/64-wide packing granularities, odd k values that
 * force k-padding, single rows and columns, and panels wider than one NC
 * block. Nothing here assumes a particular NC, KC or NR, since those depend on
 * the target processor and may be overridden by the caller.
 *
 * Small shapes only ever exercise a single macro panel, which leaves the panel
 * stepping itself untested: the reorder walks jc over NC and pc over KC and
 * places each nc0 x kc0 tile at a computed offset, so a shape that never takes
 * a second pass through either loop cannot catch a wrong panel stride. The
 * larger shapes below clear the biggest KC and NC of any supported target so
 * that both loops iterate several times, with odd k and n so the multi-panel
 * layout is combined with k-padding and a narrow trailing NR fringe.
 *
 * A large shape on its own is not enough, because the jc loop is split across
 * threads: on a machine with many cores each thread is handed roughly one
 * NR-wide slice, nc0 collapses to NR and the multi-panel path is skipped no
 * matter how wide the matrix is. Every case therefore runs at a pinned single
 * thread, where one call has to walk a full NC panel, as well as at the
 * ambient thread count, which is the only way to reach the partial-panel
 * offsets that a split jc range produces.
 */

#include "aocl_dlp.h"
#include <climits>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

struct Shape
{
    md_t k;
    md_t n;
};

// Shapes chosen to straddle the packing granularities and cache blocks.
const std::vector<Shape>&
shapes()
{
    static const std::vector<Shape> s = {
        // Exact multiples of the common NR values.
        { 64, 64 },
        { 128, 256 },
        { 32, 16 },
        // Not a multiple of anything in particular.
        { 100, 100 },
        { 17, 33 },
        { 5, 7 },
        { 3, 3 },
        // Odd k forces the k-padding path in the 16-bit and int8 layouts.
        { 63, 129 },
        { 33, 257 },
        { 1, 64 },
        { 1025, 2 },
        // Single column and single row.
        { 64, 1 },
        { 2, 1025 },
        // Large enough to span more than one NC panel on any target.
        { 2048, 64 },
        { 128, 1024 },
        { 1024, 128 },
        // Ends of the 20..20000 n range, paired with a modest k so the
        // existing suites also see those widths without a 20000 x 20000
        // allocation.
        { 20, 20 },
        { 24, 48 },
        { 64, 20000 },
        { 20000, 20 },
        { 20000, 64 },
    };
    return s;
}

// Shapes that force the reorder to walk several KC x NC macro panels, and
// several KC x NR micro panels within each of them.
//
// The block sizes are per target and per dtype, so rather than tune to one
// processor these clear the largest of any supported target: KC tops out at
// 4096 (bf16, and int8 on the newest cores) and NC is 1024 for the 16-bit and
// int8 layouts. f32 is the outlier with NC in the 8064-8160 range; it is left
// to F32SpansMultipleNcPanels below so the other dtypes do not have to carry a
// matrix that wide.
const std::vector<Shape>&
large_shapes()
{
    static const std::vector<Shape> s = {
        // k past the largest KC and n past NC, both odd, so the pc and jc
        // loops both take several passes while k-padding is also in play and
        // the trailing panel is a partial one.
        { 4099, 1091 },
        // Exact multiples of the largest KC and of NC: panel boundaries land
        // precisely on the tile edges and no fringe kernel runs at all, which
        // is where an off-by-one in the panel stride tends to hide.
        { 4096, 1024 },
        // Several NC panels where the last is only 7 columns wide, so the
        // narrow lt16 fringe kernels run at the end of a multi-panel layout
        // rather than in isolation.
        { 1027, 3079 },
        // Tall and narrow: many pc steps against a single partial NC panel.
        { 8195, 129 },
    };
    return s;
}

#ifndef _WIN32
// A buffer whose last usable byte sits immediately before an unmapped page,
// so any overrun faults at the offending access rather than corrupting the
// heap and surfacing somewhere unrelated.
template<typename T>
class GuardedBuffer
{
  public:
    explicit GuardedBuffer(size_t elems)
        : elems_(elems)
    {
        const size_t page   = (size_t)sysconf(_SC_PAGESIZE);
        const size_t bytes  = elems * sizeof(T);
        const size_t usable = ((bytes + page - 1) / page) * page;
        total_              = usable + page;
        base_               = mmap(nullptr, total_, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base_ == MAP_FAILED) {
            base_ = nullptr;
            return;
        }
        mprotect((char*)base_ + usable, page, PROT_NONE);
        std::memset(base_, 0, usable);
        data_ = (T*)((char*)base_ + (usable - bytes));
    }

    ~GuardedBuffer()
    {
        if (base_) {
            munmap(base_, total_);
        }
    }

    GuardedBuffer(const GuardedBuffer&)            = delete;
    GuardedBuffer& operator=(const GuardedBuffer&) = delete;

    T*     data() const { return data_; }
    size_t size() const { return elems_; }
    bool   valid() const { return data_ != nullptr; }

  private:
    size_t elems_;
    size_t total_ = 0;
    void*  base_  = nullptr;
    T*     data_  = nullptr;
};
#else
// Guard pages are POSIX-only here; fall back to a plain allocation so the
// correctness assertions still run on Windows.
template<typename T>
class GuardedBuffer
{
  public:
    explicit GuardedBuffer(size_t elems)
        : storage_(elems)
    {
    }
    T*     data() const { return const_cast<T*>(storage_.data()); }
    size_t size() const { return storage_.size(); }
    bool   valid() const { return true; }

  private:
    std::vector<T> storage_;
};
#endif

// Pins the library thread count for as long as it is in scope. The destructor
// always returns to the ambient setting (-1); do not nest these, because the
// inner destructor would drop an outer pin.
class PinnedThreadCount
{
  public:
    explicit PinnedThreadCount(md_t n_threads)
    {
        dlp_thread_set_num_threads(n_threads);
    }
    ~PinnedThreadCount() { dlp_thread_set_num_threads(-1); }

    PinnedThreadCount(const PinnedThreadCount&)            = delete;
    PinnedThreadCount& operator=(const PinnedThreadCount&) = delete;
};

// A guard-page hit kills the process, and gtest reports nothing about the case
// that was in flight, which leaves a bare SIGSEGV and no shape to go on. Keep
// the current case in a buffer that a handler can write out unchanged.
char g_current_case[256];

#ifndef _WIN32
void
report_current_case(int sig)
{
    static const char prefix[] = "\nfaulted while running: ";
    ssize_t           ignored;
    ignored = write(STDERR_FILENO, prefix, sizeof(prefix) - 1);
    ignored = write(STDERR_FILENO, g_current_case, std::strlen(g_current_case));
    ignored = write(STDERR_FILENO, "\n", 1);
    (void)ignored;
    _exit(128 + sig);
}
#endif

void
set_current_case(const std::string& desc)
{
#ifndef _WIN32
    static const bool installed = [] {
        std::signal(SIGSEGV, report_current_case);
        std::signal(SIGBUS, report_current_case);
        return true;
    }();
    (void)installed;
#endif

    std::snprintf(g_current_case, sizeof(g_current_case), "%s", desc.c_str());
}

dlp_metadata_t
fresh_metadata()
{
    dlp_metadata_t md;
    std::memset(&md, 0, sizeof(md));
    return md;
}

// Deterministic, dtype-agnostic fill. Values are produced from the raw bit
// pattern so that no dtype has to survive a float conversion; the round trip
// is a pure data movement and must be bit-exact regardless of what the bits
// mean.
template<typename T>
void
fill_pattern(T* buf, size_t elems)
{
    for (size_t i = 0; i < elems; i++) {
        uint32_t v = (uint32_t)(i * 2654435761u);
        uint32_t b = (v >> 13) ^ (v >> 3);
        std::memcpy(&buf[i], &b, sizeof(T) < sizeof(b) ? sizeof(T) : sizeof(b));
    }
}

// The layout core is plain C and the internal headers it lives in are not
// C++-clean, so declare just the one entry point this needs rather than
// widening the test's include surface.
extern "C" void
dlp_reorder_ref_packb(void*       pack_b,
                      const void* b,
                      md_t        elem_sz,
                      md_t        k_factor,
                      md_t        nc0,
                      md_t        kc0,
                      md_t        NR,
                      md_t        min_NR,
                      md_t        rs_b,
                      md_t        cs_b);

md_t
round_up_to(md_t v, md_t granularity)
{
    return ((v + granularity - 1) / granularity) * granularity;
}

bool
is_row_major(char order)
{
    return (order == 'r') || (order == 'R');
}

bool
is_notrans(char trans)
{
    return (trans == 'n') || (trans == 'N');
}

md_t
plain_ldb(char order, char trans, md_t k, md_t n)
{
    if (is_row_major(order)) {
        return is_notrans(trans) ? n : k;
    }
    return is_notrans(trans) ? k : n;
}

size_t
plain_elems(char order, char trans, md_t k, md_t n, md_t ldb)
{
    if (is_row_major(order)) {
        return (size_t)(is_notrans(trans) ? k : n) * (size_t)ldb;
    }
    return (size_t)(is_notrans(trans) ? n : k) * (size_t)ldb;
}

size_t
plain_idx(char order, char trans, md_t kr, md_t jc, md_t ldb)
{
    md_t rs_b;
    md_t cs_b;
    if (is_row_major(order)) {
        rs_b = is_notrans(trans) ? ldb : 1;
        cs_b = is_notrans(trans) ? 1 : ldb;
    } else {
        rs_b = is_notrans(trans) ? 1 : ldb;
        cs_b = is_notrans(trans) ? ldb : 1;
    }
    return (size_t)kr * (size_t)rs_b + (size_t)jc * (size_t)cs_b;
}

using SizeFn = msz_t (*)(const char,
                         const char,
                         const char,
                         const md_t,
                         const md_t,
                         dlp_metadata_t*);

// Reorder then un-reorder and compare logical B. trans_in is how the input
// is stored for the pack; trans_out is how the unpack writes. They need not
// match. Returns false if the dtype is unsupported on this processor.
template<typename T, typename ReorderFn, typename UnreorderFn>
bool
round_trip(SizeFn                     size_fn,
           ReorderFn                  reorder_fn,
           UnreorderFn                unreorder_fn,
           char                       order,
           const Shape&               s,
           const dlp_gemm_blocking_t* bp        = nullptr,
           char                       trans_in  = 'n',
           char                       trans_out = 'n',
           md_t                       extra_ldb = 0)
{
    const md_t   ldb_in  = plain_ldb(order, trans_in, s.k, s.n) + extra_ldb;
    const md_t   ldb_out = plain_ldb(order, trans_out, s.k, s.n) + extra_ldb;
    const size_t in_n    = plain_elems(order, trans_in, s.k, s.n, ldb_in);
    const size_t out_n   = plain_elems(order, trans_out, s.k, s.n, ldb_out);

    auto with_bp = [bp]() {
        dlp_metadata_t m = fresh_metadata();
        m.block_params   = const_cast<dlp_gemm_blocking_t*>(bp);
        return m;
    };

    dlp_metadata_t md     = with_bp();
    msz_t          buf_sz = size_fn(order, trans_in, 'B', s.k, s.n, &md);
    if (buf_sz == 0) {
        return false; // Not supported on this processor.
    }

    GuardedBuffer<T>       input(in_n);
    GuardedBuffer<T>       output(out_n);
    GuardedBuffer<uint8_t> packed((size_t)buf_sz);

    if (!input.valid() || !output.valid() || !packed.valid()) {
        ADD_FAILURE() << "guarded allocation failed";
        return true;
    }

    fill_pattern(input.data(), in_n);

    md = with_bp();
    reorder_fn(order, trans_in, 'B', input.data(), (T*)packed.data(), s.k, s.n,
               ldb_in, &md);
    if (md.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED) {
        return false;
    }
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_SUCCESS)
        << "reorder failed for k=" << s.k << " n=" << s.n << " order=" << order
        << " trans=" << trans_in;
    if (md.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        return true;
    }

    md = with_bp();
    unreorder_fn(order, trans_out, 'B', (const T*)packed.data(), output.data(),
                 s.k, s.n, ldb_out, &md);
    if (md.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED) {
        return false;
    }
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_SUCCESS)
        << "un-reorder failed for k=" << s.k << " n=" << s.n
        << " order=" << order << " trans=" << trans_out;
    if (md.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        return true;
    }

    for (md_t kr = 0; kr < s.k; kr++) {
        for (md_t jc = 0; jc < s.n; jc++) {
            const T in_v =
                input.data()[plain_idx(order, trans_in, kr, jc, ldb_in)];
            const T out_v =
                output.data()[plain_idx(order, trans_out, kr, jc, ldb_out)];
            EXPECT_EQ(in_v, out_v)
                << "round trip differs k=" << s.k << " n=" << s.n
                << " order=" << order << " trans_in=" << trans_in
                << " trans_out=" << trans_out << " at (" << kr << "," << jc
                << ")";
            if (in_v != out_v) {
                return true;
            }
        }
    }

    return true;
}

// Drives every shape in both layouts for one dtype.
template<typename T, typename ReorderFn, typename UnreorderFn>
void
round_trip_all_shapes(const char* label,
                      SizeFn      size_fn,
                      ReorderFn   reorder_fn,
                      UnreorderFn unreorder_fn)
{
    int supported = 0;
    // 1 forces a single call to cover a whole NC panel; -1 hands the jc loop
    // back to every core, which is the only way to reach the partial-panel
    // offset arithmetic.
    for (md_t threads : { (md_t)1, (md_t)-1 }) {
        PinnedThreadCount pin(threads);
        for (char order : { 'r', 'c' }) {
            for (const std::vector<Shape>* list :
                 { &shapes(), &large_shapes() }) {
                for (const Shape& s : *list) {
                    const std::string desc =
                        std::string(label) + " threads="
                        + (threads < 0 ? std::string("default")
                                       : std::to_string(threads))
                        + " order=" + order + " k=" + std::to_string(s.k)
                        + " n=" + std::to_string(s.n);
                    // Recorded up front so that if a guard page fires, the log
                    // still names the shape that overran.
                    SCOPED_TRACE(desc);
                    set_current_case(desc);
                    if (round_trip<T>(size_fn, reorder_fn, unreorder_fn, order,
                                      s)) {
                        supported++;
                    }
                }
            }
        }
    }
    if (supported == 0) {
        GTEST_SKIP() << label << " is not supported on this processor";
    }
}

// Reorder/un-reorder must follow whatever blocking the decision engine hands
// them, not the compiled-in default. This drives the round trip again with NR,
// KC and NC supplied through metadata so that a kernel which silently assumes
// the default width fails here.
//
// NC must stay a multiple of NR, which the metadata validator enforces.
// Does the reorder actually lay B out at the width it was handed?
//
// Not every dtype can. Dynamic JIT packers honour the context NR where the
// corresponding kernel exists, while fixed-width intrinsic packers may ignore
// it. Such a reorder quietly emits its native width, so measure support rather
// than assuming it.
//
// The measurement itself is always single-threaded. The jc range is split
// NR-wide, so a 2*NR probe at the ambient count is handed to the packer as
// two NR-wide slices. Int8 fringe kernels emit exactly that, so a packer
// hardcoded to NR=64 still matches an NR=32/16/48 oracle and the probe
// lets those widths through. A later whole-NC panel then fails the round
// trip. Pinning one thread is what makes the assumed width show up.
template<typename T, typename SizeF, typename ReorderF>
bool
reorder_honours_width(SizeF                      size_fn,
                      ReorderF                   reorder_fn,
                      md_t                       kf,
                      md_t                       min_NR,
                      const dlp_gemm_blocking_t& bp)
{
    PinnedThreadCount pin((md_t)1);

    // One panel (n <= NC, k <= KC) so the reordered buffer is a single packed
    // panel at offset zero and the core is a valid oracle. Two NR chunks wide,
    // because at one chunk or less every width produces the same bytes and the
    // probe would not be able to tell them apart.
    const md_t n = dlp_min(bp.NC, round_up_to(2 * bp.NR, min_NR));
    const md_t k = dlp_min(bp.KC, (md_t)64);
    if ((n < 2 * bp.NR) || (k < 1)) {
        return false; // Too narrow to distinguish; treat as unsupported.
    }

    const md_t   ldb   = n; // Row major is enough to settle the width.
    const size_t elems = (size_t)k * (size_t)n;

    dlp_gemm_blocking_t probe_bp = bp;
    dlp_metadata_t      md       = fresh_metadata();
    md.block_params              = &probe_bp;

    const msz_t sz = size_fn('r', 'n', 'B', k, n, &md);
    if (sz == 0) {
        return false; // Unsupported on this processor.
    }

    std::vector<T> in(elems);
    fill_pattern(in.data(), elems);

    std::vector<uint8_t> simd((size_t)sz, 0);
    std::vector<uint8_t> generic((size_t)sz, 0);

    md              = fresh_metadata();
    md.block_params = &probe_bp;
    reorder_fn('r', 'n', 'B', in.data(), (T*)simd.data(), k, n, ldb, &md);
    if (md.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        return false;
    }

    dlp_reorder_ref_packb(generic.data(), in.data(), (md_t)sizeof(T), kf, n, k,
                          bp.NR, min_NR, ldb, 1);

    // Panel only: s8 parks a block of per-column sums after it that a pure
    // packer has no business writing.
    const size_t panel_bytes =
        (size_t)round_up_to(k, kf) * (size_t)round_up_to(n, min_NR) * sizeof(T);
    if (panel_bytes > (size_t)sz) {
        return false;
    }

    return std::memcmp(simd.data(), generic.data(), panel_bytes) == 0;
}

template<typename T, typename ReorderFn, typename UnreorderFn>
void
round_trip_block_params(const char*                             label,
                        SizeFn                                  size_fn,
                        ReorderFn                               reorder_fn,
                        UnreorderFn                             unreorder_fn,
                        const std::vector<dlp_gemm_blocking_t>& params,
                        md_t                                    kf,
                        md_t                                    min_NR,
                        bool skip_width_probe = false)
{
    int supported = 0;
    // The width probe pins a single thread of its own (see
    // reorder_honours_width). It has to run outside this loop: a nested pin
    // would restore ambient on the way out and drop the 1/-1 pin below.
    for (const dlp_gemm_blocking_t& bp : params) {
        if (!skip_width_probe
            && !reorder_honours_width<T>(size_fn, reorder_fn, kf, min_NR, bp)) {
            continue;
        }
        // 1 forces a whole NC panel onto one call; -1 is the ambient split.
        for (md_t threads : { (md_t)1, (md_t)-1 }) {
            PinnedThreadCount pin(threads);
            for (char order : { 'r', 'c' }) {
                for (const Shape& s : shapes()) {
                    const std::string desc =
                        std::string(label) + " threads="
                        + (threads < 0 ? std::string("default")
                                       : std::to_string(threads))
                        + " NR=" + std::to_string(bp.NR) + " KC="
                        + std::to_string(bp.KC) + " NC=" + std::to_string(bp.NC)
                        + " order=" + order + " k=" + std::to_string(s.k)
                        + " n=" + std::to_string(s.n);
                    SCOPED_TRACE(desc);
                    set_current_case(desc);
                    if (round_trip<T>(size_fn, reorder_fn, unreorder_fn, order,
                                      s, &bp)) {
                        supported++;
                    }
                }
            }
        }
    }
    if (supported == 0) {
        GTEST_SKIP() << label << " is not supported on this processor";
    }
}

// Widths below, at and above the default NR, plus non-default KC/NC, so the
// full-panel, fringe and tail paths are all exercised at each width.
//
// The last two are deliberately awkward: a width that is not a power of two,
// an NC that is an odd multiple of it, and a KC that divides nothing in
// particular. Round numbers can hide an assumption by happening to agree with
// it, so these are the cases that actually hold the un-reorder to being
// parameter-driven.
//
// KC must be a multiple of the dtype's k_factor. Each pc panel pads its own kc0
// up to k_factor, so an odd KC makes the panels sum to more than the
// round_up(k, k_factor) rows the buffer was sized for.
//
// Every dtype is offered the same widths. Which of them a given dtype can
// actually reach is not fixed here but measured per machine by
// reorder_honours_width, so a path whose packer ignores NR skips instead of
// failing.
std::vector<dlp_gemm_blocking_t>
variable_block_params()
{
    return {
        { 0, 64, 0, 1024, 256 },  { 0, 32, 0, 1024, 1024 },
        { 0, 16, 0, 512, 384 },   { 0, 48, 0, 336, 100 },
        { 0, 16, 0, 240, 44 },    { 0, 80, 0, 560, 256 },
        { 0, 96, 0, 1056, 256 },  { 0, 112, 0, 784, 256 },
        { 0, 128, 0, 1024, 256 }, { 0, 144, 0, 1008, 256 },
        { 0, 160, 0, 1120, 256 },
    };
}

// This larger stress sweep is pinned at NR=64 to bound its runtime. NC and KC
// vary independently of it, always as multiples of 64 so they remain legal for
// every dtype (NC must be a multiple of NR; KC must be a multiple of the k
// interleave factor, which is 1, 2 or 4).
std::vector<dlp_gemm_blocking_t>
nr64_sweep_block_params()
{
    return {
        { 0, 64, 0, 1024, 256 }, // typical cache blocks
        { 0, 64, 0, 64, 1024 },  // one NR-wide panel, many KC steps
        { 0, 64, 0, 320, 192 },  // awkward but legal multiples
        { 0, 64, 0, 2048, 64 },  // wide NC, short KC
    };
}

bool
is_sweep_partner(md_t n)
{
    // The sparse axis used once either k or n is past the dense cutoff: a
    // mix of packing-granularity edges (powers of two and one-off) and the
    // 16k bound, so a large dimension is still crossed with a fringe, a
    // full NR, a full NC-ish panel and the other extreme.
    static const md_t partners[] = { 1,    7,    16,   32,   64,  65,
                                     128,  192,  256,  320,  512, 1024,
                                     2048, 4096, 8192, 16384 };
    for (md_t p : partners) {
        if (p == n) {
            return true;
        }
    }
    return false;
}

// 1 < k,n <= 16k, non-linear. Below 512 every listed pair runs so packing
// fringes are covered densely; past that, each large value is crossed with
// the partner axis (and with itself) so 16k x 16k is in the set without
// paying for a 16k x 16k cartesian.
std::vector<Shape>
nr64_sweep_shapes()
{
    static const md_t dims[] = { 1,    2,    3,     5,     7,    8,    15,
                                 16,   17,   31,    32,    33,   63,   64,
                                 65,   96,   127,   128,   129,  192,  255,
                                 256,  257,  320,   384,   511,  512,  513,
                                 768,  1023, 1024,  1025,  1536, 2047, 2048,
                                 2049, 3072, 4095,  4096,  4097, 6144, 8191,
                                 8192, 8193, 12288, 16383, 16384 };

    std::vector<Shape> out;
    for (md_t k : dims) {
        for (md_t n : dims) {
            if ((k <= 512 && n <= 512) || is_sweep_partner(n) || k == n) {
                out.push_back({ k, n });
            }
        }
    }
    return out;
}

template<typename T, typename ReorderFn, typename UnreorderFn>
void
round_trip_nr64_sweep(const char* label,
                      SizeFn      size_fn,
                      ReorderFn   reorder_fn,
                      UnreorderFn unreorder_fn,
                      md_t        kf,
                      md_t        min_NR)
{
    const std::vector<Shape>               cases  = nr64_sweep_shapes();
    const std::vector<dlp_gemm_blocking_t> params = nr64_sweep_block_params();
    int                                    supported = 0;

    // Multithreaded only: the existing round-trip suite already pins a
    // single thread to force a whole-NC panel. -1 uses every core the
    // process is bound to (soc2 when launched under taskset 128-255).
    PinnedThreadCount pin((md_t)-1);
    for (const dlp_gemm_blocking_t& bp : params) {
        if (!reorder_honours_width<T>(size_fn, reorder_fn, kf, min_NR, bp)) {
            continue;
        }
        for (char order : { 'r', 'c' }) {
            for (const Shape& s : cases) {
                const std::string desc =
                    std::string(label) + " threads=default NR="
                    + std::to_string(bp.NR) + " KC=" + std::to_string(bp.KC)
                    + " NC=" + std::to_string(bp.NC) + " order=" + order
                    + " k=" + std::to_string(s.k) + " n=" + std::to_string(s.n);
                SCOPED_TRACE(desc);
                set_current_case(desc);
                if (round_trip<T>(size_fn, reorder_fn, unreorder_fn, order, s,
                                  &bp)) {
                    supported++;
                }
            }
        }
    }
    if (supported == 0) {
        GTEST_SKIP() << label << " is not supported on this processor";
    }
}

// ---------------------------------------------------------------------------
// Round trip, one test per data type.
// ---------------------------------------------------------------------------

TEST(UnreorderRoundTrip, F32F32F32OF32Reference)
{
    round_trip_all_shapes<float>(
        "f32f32f32of32", aocl_get_reorder_buf_size_f32f32f32of32,
        aocl_reorder_f32f32f32of32, aocl_unreorder_f32f32f32of32_reference);
}

TEST(UnreorderRoundTrip, Bf16Bf16F32OF32Reference)
{
    round_trip_all_shapes<bfloat16>(
        "bf16bf16f32of32", aocl_get_reorder_buf_size_bf16bf16f32of32,
        aocl_reorder_bf16bf16f32of32, aocl_unreorder_bf16bf16f32of32_reference);
}

TEST(UnreorderRoundTrip, S8S8S32OS32Reference)
{
    round_trip_all_shapes<int8_t>(
        "s8s8s32os32", aocl_get_reorder_buf_size_s8s8s32os32,
        aocl_reorder_s8s8s32os32, aocl_unreorder_s8s8s32os32_reference);
}

TEST(UnreorderRoundTrip, U8S8S32OS32Reference)
{
    round_trip_all_shapes<int8_t>(
        "u8s8s32os32", aocl_get_reorder_buf_size_u8s8s32os32,
        aocl_reorder_u8s8s32os32, aocl_unreorder_u8s8s32os32_reference);
}

TEST(UnreorderRoundTrip, F16F16F16OF16Reference)
{
    round_trip_all_shapes<float16>(
        "f16f16f16of16", aocl_get_reorder_buf_size_f16f16f16of16,
        aocl_reorder_f16f16f16of16, aocl_unreorder_f16f16f16of16_reference);
}

// ---------------------------------------------------------------------------
// Round trip under caller-supplied blocking. These are the tests that hold the
// references to "works at any NR/KC/NC" rather than only at the default.
// ---------------------------------------------------------------------------

TEST(UnreorderBlockParams, Bf16Bf16F32OF32Reference)
{
    round_trip_block_params<bfloat16>(
        "bf16bf16f32of32", aocl_get_reorder_buf_size_bf16bf16f32of32,
        aocl_reorder_bf16bf16f32of32, aocl_unreorder_bf16bf16f32of32_reference,
        variable_block_params(), 2, 16);
}

TEST(UnreorderBlockParams, S8S8S32OS32Reference)
{
    round_trip_block_params<int8_t>(
        "s8s8s32os32", aocl_get_reorder_buf_size_s8s8s32os32,
        aocl_reorder_s8s8s32os32, aocl_unreorder_s8s8s32os32_reference,
        variable_block_params(), 4, 16);
}

TEST(UnreorderBlockParams, U8S8S32OS32Reference)
{
    round_trip_block_params<int8_t>(
        "u8s8s32os32", aocl_get_reorder_buf_size_u8s8s32os32,
        aocl_reorder_u8s8s32os32, aocl_unreorder_u8s8s32os32_reference,
        variable_block_params(), 4, 16);
}

// f32 has k_factor 1, so no panel pads its k extent and KC is free to divide
// nothing at all. The odd values here are legal only for this dtype, and they
// are the strongest available check that the pc loop carries no assumption
// about KC dividing k or anything else.
TEST(UnreorderBlockParams, F32F32F32OF32Reference)
{
    round_trip_block_params<float>(
        "f32f32f32of32", aocl_get_reorder_buf_size_f32f32f32of32,
        aocl_reorder_f32f32f32of32, aocl_unreorder_f32f32f32of32_reference,
        { { 0, 64, 0, 1024, 256 },
          { 0, 16, 0, 512, 384 },
          { 0, 64, 0, 1024, 37 },
          { 0, 16, 0, 240, 101 } },
        1, 64);
}

// fp16 packs 128 wide against a 32-element granularity, so it gets its own
// ladder. As everywhere else, the widths its reorder cannot actually emit are
// skipped by the probe rather than listed here.
//
// Untested locally: this host has no AVX512-FP16, so both fp16 cases skip.
TEST(UnreorderBlockParams, F16F16F16OF16Reference)
{
    round_trip_block_params<float16>(
        "f16f16f16of16", aocl_get_reorder_buf_size_f16f16f16of16,
        aocl_reorder_f16f16f16of16, aocl_unreorder_f16f16f16of16_reference,
        { { 0, 128, 0, 1024, 512 },
          { 0, 64, 0, 1024, 256 },
          { 0, 32, 0, 512, 2048 } },
        1, 32);
}

// ---------------------------------------------------------------------------
// Reference reorder <-> reference un-reorder. Both sides honour NR/NC/KC, so
// these do not probe the production packer and do not skip non-native widths.
// ---------------------------------------------------------------------------

TEST(ReorderRefRoundTrip, F32F32F32OF32)
{
    round_trip_all_shapes<float>("f32f32f32of32_ref",
                                 aocl_get_reorder_buf_size_f32f32f32of32,
                                 aocl_reorder_f32f32f32of32_reference,
                                 aocl_unreorder_f32f32f32of32_reference);
}

TEST(ReorderRefRoundTrip, Bf16Bf16F32OF32)
{
    round_trip_all_shapes<bfloat16>("bf16bf16f32of32_ref",
                                    aocl_get_reorder_buf_size_bf16bf16f32of32,
                                    aocl_reorder_bf16bf16f32of32_reference,
                                    aocl_unreorder_bf16bf16f32of32_reference);
}

TEST(ReorderRefRoundTrip, S8S8S32OS32)
{
    round_trip_all_shapes<int8_t>("s8s8s32os32_ref",
                                  aocl_get_reorder_buf_size_s8s8s32os32,
                                  aocl_reorder_s8s8s32os32_reference,
                                  aocl_unreorder_s8s8s32os32_reference);
}

TEST(ReorderRefRoundTrip, U8S8S32OS32)
{
    round_trip_all_shapes<int8_t>("u8s8s32os32_ref",
                                  aocl_get_reorder_buf_size_u8s8s32os32,
                                  aocl_reorder_u8s8s32os32_reference,
                                  aocl_unreorder_u8s8s32os32_reference);
}

TEST(ReorderRefRoundTrip, F16F16F16OF16)
{
    round_trip_all_shapes<float16>("f16f16f16of16_ref",
                                   aocl_get_reorder_buf_size_f16f16f16of16,
                                   aocl_reorder_f16f16f16of16_reference,
                                   aocl_unreorder_f16f16f16of16_reference);
}

TEST(ReorderRefBlockParams, Bf16Bf16F32OF32)
{
    round_trip_block_params<bfloat16>("bf16bf16f32of32_ref",
                                      aocl_get_reorder_buf_size_bf16bf16f32of32,
                                      aocl_reorder_bf16bf16f32of32_reference,
                                      aocl_unreorder_bf16bf16f32of32_reference,
                                      variable_block_params(), 2, 16, true);
}

TEST(ReorderRefBlockParams, S8S8S32OS32)
{
    round_trip_block_params<int8_t>("s8s8s32os32_ref",
                                    aocl_get_reorder_buf_size_s8s8s32os32,
                                    aocl_reorder_s8s8s32os32_reference,
                                    aocl_unreorder_s8s8s32os32_reference,
                                    variable_block_params(), 4, 16, true);
}

TEST(ReorderRefBlockParams, U8S8S32OS32)
{
    round_trip_block_params<int8_t>("u8s8s32os32_ref",
                                    aocl_get_reorder_buf_size_u8s8s32os32,
                                    aocl_reorder_u8s8s32os32_reference,
                                    aocl_unreorder_u8s8s32os32_reference,
                                    variable_block_params(), 4, 16, true);
}

TEST(ReorderRefBlockParams, F32F32F32OF32)
{
    round_trip_block_params<float>("f32f32f32of32_ref",
                                   aocl_get_reorder_buf_size_f32f32f32of32,
                                   aocl_reorder_f32f32f32of32_reference,
                                   aocl_unreorder_f32f32f32of32_reference,
                                   { { 0, 64, 0, 1024, 256 },
                                     { 0, 16, 0, 512, 384 },
                                     { 0, 64, 0, 1024, 37 },
                                     { 0, 16, 0, 240, 101 } },
                                   1, 16, true);
}

TEST(ReorderRefBlockParams, F16F16F16OF16)
{
    round_trip_block_params<float16>("f16f16f16of16_ref",
                                     aocl_get_reorder_buf_size_f16f16f16of16,
                                     aocl_reorder_f16f16f16of16_reference,
                                     aocl_unreorder_f16f16f16of16_reference,
                                     { { 0, 128, 0, 1024, 512 },
                                       { 0, 64, 0, 1024, 256 },
                                       { 0, 32, 0, 512, 2048 } },
                                     1, 32, true);
}

// Size-grid coverage lives in test_reorder_ref_yaml.cc, driven by
// configs/reorder_ref_coverage.yaml, so each (dtype, k, n, order) is a
// named gtest.

TEST(ReorderRefValidation, RejectsKcNotMultipleOfKFactor)
{
    dlp_gemm_blocking_t bp = { 0, 64, 0, 1024, 5 };
    dlp_metadata_t      md = fresh_metadata();
    md.block_params        = &bp;

    const md_t            k = 32;
    const md_t            n = 64;
    std::vector<bfloat16> in(k * n, 0);
    std::vector<bfloat16> packed(k * n * 2, 0);

    aocl_reorder_bf16bf16f32of32_reference('r', 'n', 'B', in.data(),
                                           packed.data(), k, n, n, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_BLOCK_PARAMS);
}

TEST(ReorderRefValidation, RejectsNrNotMultipleOfMinNr)
{
    // bf16 min_NR is 16. NR=8 is a requested coverage width, but illegal here.
    dlp_gemm_blocking_t bp = { 0, 8, 0, 16, 512 };
    dlp_metadata_t      md = fresh_metadata();
    md.block_params        = &bp;

    const md_t            k = 32;
    const md_t            n = 16;
    std::vector<bfloat16> in(k * n, 0);
    std::vector<bfloat16> packed(k * n * 2, 0);

    aocl_reorder_bf16bf16f32of32_reference('r', 'n', 'B', in.data(),
                                           packed.data(), k, n, n, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_BLOCK_PARAMS);
}

TEST(ReorderRefValidation, RejectsWMatrix)
{
    const md_t         k = 32, n = 32;
    std::vector<float> in(k * n, 0);
    std::vector<float> packed(k * n, 0);

    dlp_metadata_t md = fresh_metadata();
    aocl_reorder_f32f32f32of32_reference('r', 'n', 'W', in.data(),
                                         packed.data(), k, n, n, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);

    md = fresh_metadata();
    std::vector<bfloat16> in_bf(k * n, 0);
    std::vector<bfloat16> packed_bf(k * n, 0);
    aocl_reorder_bf16bf16f32of32_reference('r', 'n', 'W', in_bf.data(),
                                           packed_bf.data(), k, n, n, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST(ReorderBufSize, ReferenceDtypesDoNotRequireTunedIsa)
{
    dlp_metadata_t md = fresh_metadata();
    EXPECT_GT(
        aocl_get_reorder_buf_size_f16f16f16of16('r', 'n', 'B', 32, 64, &md), 0);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_SUCCESS);

    md = fresh_metadata();
    EXPECT_GT(aocl_get_reorder_buf_size_u8s8s32os32('r', 'n', 'B', 32, 64, &md),
              0);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_SUCCESS);

    md = fresh_metadata();
    EXPECT_GT(aocl_get_reorder_buf_size_s8s8s32os32('r', 'n', 'B', 32, 64, &md),
              0);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_SUCCESS);

    md = fresh_metadata();
    EXPECT_GT(
        aocl_get_reorder_buf_size_f32f32f32of32('r', 'n', 'B', 32, 64, &md), 0);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_SUCCESS);
}

TEST(ReorderBufSize, Fp16RejectsNrNotMultipleOfMinNr)
{
    // NC % NR == 0 so metadata validation passes; NR % 32 != 0 so the
    // reference layout must still refuse.
    dlp_gemm_blocking_t bp = { 0, 16, 0, 64, 256 };
    dlp_metadata_t      md = fresh_metadata();
    md.block_params        = &bp;

    EXPECT_EQ(
        aocl_get_reorder_buf_size_f16f16f16of16('r', 'n', 'B', 32, 64, &md), 0);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_BLOCK_PARAMS);
}

TEST(ReorderTunedValidation, SupportsDynamicInt8Nr)
{
    const Shape shape = { 67, 199 };

    // Establish ISA support once using the default configuration. A rejection
    // of any explicitly tested NR below is then a failure, not a silent skip.
    if (!round_trip<int8_t>(aocl_get_reorder_buf_size_u8s8s32os32,
                            aocl_reorder_u8s8s32os32,
                            aocl_unreorder_u8s8s32os32_reference, 'r', shape)) {
        GTEST_SKIP() << "INT8 reorder is not supported on this processor";
    }

    for (md_t nr : { (md_t)16, (md_t)32, (md_t)48, (md_t)64, (md_t)80, (md_t)96,
                     (md_t)112, (md_t)128 }) {
        dlp_gemm_blocking_t bp = { 0, nr, 0, 12 * nr, 100 };
        SCOPED_TRACE("NR=" + std::to_string(nr));

        EXPECT_TRUE(round_trip<int8_t>(
            aocl_get_reorder_buf_size_u8s8s32os32, aocl_reorder_u8s8s32os32,
            aocl_unreorder_u8s8s32os32_reference, 'r', shape, &bp));
        EXPECT_TRUE(round_trip<int8_t>(
            aocl_get_reorder_buf_size_s8s8s32os32, aocl_reorder_s8s8s32os32,
            aocl_unreorder_s8s8s32os32_reference, 'r', shape, &bp));
    }

    // U8 packing has no fused column-sum accumulators, so its register-feasible
    // GEMM widths extend beyond the S8 limit of 128.
    for (md_t nr : { (md_t)144, (md_t)160 }) {
        dlp_gemm_blocking_t bp = { 0, nr, 0, 7 * nr, 100 };
        SCOPED_TRACE("U8-only NR=" + std::to_string(nr));
        EXPECT_TRUE(round_trip<int8_t>(
            aocl_get_reorder_buf_size_u8s8s32os32, aocl_reorder_u8s8s32os32,
            aocl_unreorder_u8s8s32os32_reference, 'r', shape, &bp));
    }
}

TEST(ReorderTunedValidation, Fp16RejectsNonNativeNr)
{
    dlp_gemm_blocking_t bp = { 0, 32, 0, 1024, 256 };
    dlp_metadata_t      md = fresh_metadata();
    md.block_params        = &bp;
    const md_t           k = 32;
    const md_t           n = 64;
    std::vector<float16> in(k * n, 0);
    std::vector<float16> packed(k * n * 2, 0);

    aocl_reorder_f16f16f16of16('r', 'n', 'B', in.data(), packed.data(), k, n, n,
                               &md);
    if (md.error_hndl.error_code != DLP_CLSC_NOT_SUPPORTED) {
        EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_BLOCK_PARAMS);
    }
}

// ---------------------------------------------------------------------------
// Native-width stress sweep: 1..16k non-linear k/n, several NC/KC,
// multithreaded.
//
// Variable NR is covered by UnreorderBlockParams and the focused INT8 test
// above. Keep this larger matrix sweep at NR=64 to bound its runtime while
// stressing the macro-panel and fringe arithmetic.
// ---------------------------------------------------------------------------

TEST(UnreorderNr64Sweep, Bf16Bf16F32OF32Reference)
{
    round_trip_nr64_sweep<bfloat16>(
        "bf16bf16f32of32", aocl_get_reorder_buf_size_bf16bf16f32of32,
        aocl_reorder_bf16bf16f32of32, aocl_unreorder_bf16bf16f32of32_reference,
        2, 16);
}

TEST(UnreorderNr64Sweep, S8S8S32OS32Reference)
{
    round_trip_nr64_sweep<int8_t>(
        "s8s8s32os32", aocl_get_reorder_buf_size_s8s8s32os32,
        aocl_reorder_s8s8s32os32, aocl_unreorder_s8s8s32os32_reference, 4, 16);
}

TEST(UnreorderNr64Sweep, U8S8S32OS32Reference)
{
    round_trip_nr64_sweep<int8_t>(
        "u8s8s32os32", aocl_get_reorder_buf_size_u8s8s32os32,
        aocl_reorder_u8s8s32os32, aocl_unreorder_u8s8s32os32_reference, 4, 16);
}

TEST(UnreorderNr64Sweep, F32F32F32OF32Reference)
{
    round_trip_nr64_sweep<float>("f32f32f32of32",
                                 aocl_get_reorder_buf_size_f32f32f32of32,
                                 aocl_reorder_f32f32f32of32,
                                 aocl_unreorder_f32f32f32of32_reference, 1, 64);
}

TEST(UnreorderNr64Sweep, F16F16F16OF16Reference)
{
    round_trip_nr64_sweep<float16>(
        "f16f16f16of16", aocl_get_reorder_buf_size_f16f16f16of16,
        aocl_reorder_f16f16f16of16, aocl_unreorder_f16f16f16of16_reference, 1,
        32);
}

// ---------------------------------------------------------------------------
// Layout equivalence.
//
// A round trip only proves the un-reorder inverts whatever the reorder did; it
// would still pass if both sides agreed on a layout nobody else writes. These
// tests pin the shared layout core to the real thing: pack a matrix with
// dlp_reorder_ref_packb along the same panel walk the frame uses, and require
// the result to be byte-identical to what the production reorder produced.
//
// That is what lets the un-reorder reference be trusted as an inverse of the
// vectorised packers and not just of itself.
// ---------------------------------------------------------------------------

// Restricted to n > 1. n == 1 is a GEMV vector and is memcpy'd, not packed.
// The oracle is aocl_reorder_*_reference rather than dlp_reorder_ref_packb
// with a compiled-in NR: f32 packs at NR=64 on AVX-512 (Zen4+) and NR=16 on
// AVX2 (Zen/Zen3), and the production packer ignores metadata NR. Driving
// both sides through the public APIs uses the host's native width on both
// sides. s8 parks per-column sums after the panel; min_NR sizes that panel
// so the trailer is not part of the compare.
template<typename T, typename SizeF, typename ReorderF>
void
expect_layout_matches(const char* label,
                      SizeF       size_fn,
                      ReorderF    prod_fn,
                      ReorderF    ref_fn,
                      md_t        kf,
                      md_t        min_NR)
{
    int checked = 0;

    for (char order : { 'r', 'c' }) {
        for (const Shape& s : shapes()) {
            if (s.n == 1) {
                continue;
            }

            const md_t   ldb   = (order == 'r') ? s.n : s.k;
            const size_t elems = (size_t)s.k * (size_t)s.n;

            const std::string desc = std::string(label) + " order=" + order
                                     + " k=" + std::to_string(s.k)
                                     + " n=" + std::to_string(s.n);
            SCOPED_TRACE(desc);
            set_current_case(desc);

            dlp_metadata_t md = fresh_metadata();
            const msz_t    sz = size_fn(order, 'n', 'B', s.k, s.n, &md);
            if (sz == 0) {
                continue; // Unsupported on this processor.
            }

            std::vector<T> in(elems);
            fill_pattern(in.data(), elems);

            std::vector<uint8_t> simd((size_t)sz, 0);
            std::vector<uint8_t> generic((size_t)sz, 0);

            md = fresh_metadata();
            prod_fn(order, 'n', 'B', in.data(), (T*)simd.data(), s.k, s.n, ldb,
                    &md);
            if (md.error_hndl.error_code != DLP_CLSC_SUCCESS) {
                continue;
            }

            md = fresh_metadata();
            ref_fn(order, 'n', 'B', in.data(), (T*)generic.data(), s.k, s.n,
                   ldb, &md);
            if (md.error_hndl.error_code != DLP_CLSC_SUCCESS) {
                continue;
            }

            const size_t panel_bytes = (size_t)round_up_to(s.k, kf)
                                       * (size_t)round_up_to(s.n, min_NR)
                                       * sizeof(T);
            ASSERT_LE(panel_bytes, (size_t)sz);

            // s8 parks 128*col_sum after the panel; the size API includes
            // that trailer, so compare the whole buffer. u8 has no trailer.
            EXPECT_EQ(std::memcmp(simd.data(), generic.data(), (size_t)sz), 0)
                << "the reference reorder disagrees with the production pack "
                   "for "
                << desc;
            checked++;
        }
    }

    if (checked == 0) {
        GTEST_SKIP() << label << " is not supported on this processor";
    }
}

TEST(UnreorderLayoutModel, F32MatchesPackedLayout)
{
    // min_NR only sizes the s8-style panel check; f32 compares the whole
    // size-API buffer. 16 is the AVX2 packing granularity and is <= the
    // AVX-512 NR, so ASSERT_LE(panel, sz) holds on both.
    expect_layout_matches<float>("f32f32f32of32",
                                 aocl_get_reorder_buf_size_f32f32f32of32,
                                 aocl_reorder_f32f32f32of32,
                                 aocl_reorder_f32f32f32of32_reference, 1, 16);
}

TEST(UnreorderLayoutModel, Bf16MatchesPackedLayout)
{
    expect_layout_matches<bfloat16>(
        "bf16bf16f32of32", aocl_get_reorder_buf_size_bf16bf16f32of32,
        aocl_reorder_bf16bf16f32of32, aocl_reorder_bf16bf16f32of32_reference, 2,
        16);
}

TEST(UnreorderLayoutModel, S8MatchesPackedLayout)
{
    expect_layout_matches<int8_t>(
        "s8s8s32os32", aocl_get_reorder_buf_size_s8s8s32os32,
        aocl_reorder_s8s8s32os32, aocl_reorder_s8s8s32os32_reference, 4, 16);
}

TEST(ReorderRefValidation, S8WritesColumnSums)
{
    const md_t k = 17, n = 19;
    const md_t k_pad = ((k + 3) / 4) * 4;
    const md_t n_pad = ((n + 15) / 16) * 16;

    dlp_metadata_t md = fresh_metadata();
    msz_t sz = aocl_get_reorder_buf_size_s8s8s32os32('r', 'n', 'B', k, n, &md);
    ASSERT_GT(sz, 0u);

    std::vector<int8_t>  in((size_t)k * n);
    std::vector<uint8_t> packed((size_t)sz, 0xff);
    fill_pattern(in.data(), in.size());

    md = fresh_metadata();
    aocl_reorder_s8s8s32os32_reference('r', 'n', 'B', in.data(),
                                       (int8_t*)packed.data(), k, n, n, &md);
    ASSERT_EQ(md.error_hndl.error_code, DLP_CLSC_SUCCESS);

    const int32_t* sums =
        (const int32_t*)(packed.data() + ((size_t)k_pad * (size_t)n_pad));
    for (md_t j = 0; j < n; j++) {
        int32_t acc = 0;
        for (md_t i = 0; i < k; i++) {
            acc += (int32_t)in[(size_t)i * n + (size_t)j];
        }
        EXPECT_EQ(sums[j], acc * 128) << "col " << j;
    }
    for (md_t j = n; j < n_pad; j++) {
        EXPECT_EQ(sums[j], 0) << "pad col " << j;
    }
}

// The tuned bf16 entry point must round-trip at every width the reorder can
// emit, which it does by handing anything but its native width to the
// reference. The vectorised unpacker itself decodes a fixed 64, so removing
// that guard makes the narrow widths here return silently wrong data rather
// than fail loudly -- single-threaded, where a whole NC panel reaches the
// kernel instead of an NR-wide slice.
TEST(UnreorderBlockParams, Bf16TunedHonoursCallerWidth)
{
    round_trip_block_params<bfloat16>(
        "bf16bf16f32of32 tuned", aocl_get_reorder_buf_size_bf16bf16f32of32,
        aocl_reorder_bf16bf16f32of32, aocl_unreorder_bf16bf16f32of32,
        variable_block_params(), 2, 16);
}

// The tuned bf16 entry point has to agree with the reference wherever it is
// available, and must report NOT_SUPPORTED rather than misbehave where it is
// not.
TEST(UnreorderRoundTrip, Bf16TunedMatchesReference)
{
    round_trip_all_shapes<bfloat16>(
        "bf16bf16f32of32 tuned", aocl_get_reorder_buf_size_bf16bf16f32of32,
        aocl_reorder_bf16bf16f32of32, aocl_unreorder_bf16bf16f32of32);
}

// f32 packs with a much wider NC than the other layouts (8064 on AVX-512
// targets, 8160 on AVX2), so none of the shared shapes are wide enough to make
// its jc loop take a second pass. These are, which is the only way to cover
// the f32 macro-panel stride. k stays past the f32 KC (512-1024) so the pc
// loop iterates too, and both dimensions are odd so a partial trailing panel
// is packed as well.
TEST(UnreorderRoundTrip, F32SpansMultipleNcPanels)
{
    static const std::vector<Shape> wide = {
        { 1025, 8161 },  // two NC panels, the second a narrow remainder.
        { 129, 16193 },  // three NC panels.
        { 1025, 16128 }, // an exact multiple of NC=8064, so no fringe panel.
    };

    int supported = 0;
    for (md_t threads : { (md_t)1, (md_t)-1 }) {
        PinnedThreadCount pin(threads);
        for (char order : { 'r', 'c' }) {
            for (const Shape& s : wide) {
                const std::string desc =
                    std::string("f32f32f32of32 wide threads=")
                    + (threads < 0 ? std::string("default")
                                   : std::to_string(threads))
                    + " order=" + order + " k=" + std::to_string(s.k)
                    + " n=" + std::to_string(s.n);
                SCOPED_TRACE(desc);
                set_current_case(desc);
                if (round_trip<float>(aocl_get_reorder_buf_size_f32f32f32of32,
                                      aocl_reorder_f32f32f32of32,
                                      aocl_unreorder_f32f32f32of32_reference,
                                      order, s)) {
                    supported++;
                }
            }
        }
    }
    if (supported == 0) {
        GTEST_SKIP() << "f32f32f32of32 is not supported on this processor";
    }
}

// ---------------------------------------------------------------------------
// Negative / security checks for every reference dtype.
//
// Size, aocl_reorder_*_reference and aocl_unreorder_*_reference share the
// same validator. Dummy scratch is a few elements: oversized k/n must be
// rejected before any k*n allocation or pointer walk (CWE-190 / CWE-787).
// ---------------------------------------------------------------------------

template<typename T>
void
ref_api_negative(const char* label,
                 msz_t (*size_fn)(const char,
                                  const char,
                                  const char,
                                  const md_t,
                                  const md_t,
                                  dlp_metadata_t*),
                 void (*reorder_ref)(const char,
                                     const char,
                                     const char,
                                     const T*,
                                     T*,
                                     const md_t,
                                     const md_t,
                                     const md_t,
                                     dlp_metadata_t*),
                 void (*unreorder_ref)(const char,
                                       const char,
                                       const char,
                                       const T*,
                                       T*,
                                       const md_t,
                                       const md_t,
                                       const md_t,
                                       dlp_metadata_t*),
                 md_t min_NR,
                 md_t k_factor)
{
    SCOPED_TRACE(label);

    T          scratch[16] = {};
    const md_t k           = 32;
    const md_t n           = 16;
    const md_t too_big     = (md_t)INT32_MAX + 1;

    auto expect_size = [&](const char* what, char order, char trans, char mt,
                           md_t kk, md_t nn, dlp_gemm_blocking_t* bp,
                           dlp_clsc_err_t want) {
        dlp_metadata_t md = fresh_metadata();
        md.block_params   = bp;
        msz_t sz          = size_fn(order, trans, mt, kk, nn, &md);
        EXPECT_EQ(md.error_hndl.error_code, want) << label << " size " << what;
        if (want != DLP_CLSC_SUCCESS) {
            EXPECT_EQ(sz, (msz_t)0) << label << " size " << what;
        }
    };

    auto expect_reorder = [&](const char* what, char order, char trans, char mt,
                              const T* in, T* packed, md_t kk, md_t nn,
                              md_t ldb, dlp_gemm_blocking_t* bp,
                              dlp_clsc_err_t want) {
        dlp_metadata_t md = fresh_metadata();
        md.block_params   = bp;
        reorder_ref(order, trans, mt, in, packed, kk, nn, ldb, &md);
        EXPECT_EQ(md.error_hndl.error_code, want)
            << label << " reorder " << what;
    };

    auto expect_unreorder = [&](const char* what, char order, char trans,
                                char mt, const T* packed, T* out, md_t kk,
                                md_t nn, md_t ldb, dlp_gemm_blocking_t* bp,
                                dlp_clsc_err_t want) {
        dlp_metadata_t md = fresh_metadata();
        md.block_params   = bp;
        unreorder_ref(order, trans, mt, packed, out, kk, nn, ldb, &md);
        EXPECT_EQ(md.error_hndl.error_code, want)
            << label << " unreorder " << what;
    };

    expect_size("invalid order", 'x', 'n', 'B', k, n, nullptr,
                DLP_CLSC_INVALID_ORDER);
    expect_reorder("invalid order", 'x', 'n', 'B', scratch, scratch, k, n, n,
                   nullptr, DLP_CLSC_INVALID_ORDER);
    expect_unreorder("invalid order", 'x', 'n', 'B', scratch, scratch, k, n, n,
                     nullptr, DLP_CLSC_INVALID_ORDER);

    expect_size("invalid trans", 'r', 'x', 'B', k, n, nullptr,
                DLP_CLSC_INVALID_TRANSPOSE);
    expect_reorder("invalid trans", 'r', 'x', 'B', scratch, scratch, k, n, n,
                   nullptr, DLP_CLSC_INVALID_TRANSPOSE);
    expect_unreorder("invalid trans", 'r', 'x', 'B', scratch, scratch, k, n, n,
                     nullptr, DLP_CLSC_INVALID_TRANSPOSE);

    expect_size("A matrix", 'r', 'n', 'A', k, n, nullptr,
                DLP_CLSC_NOT_SUPPORTED);
    expect_reorder("A matrix", 'r', 'n', 'A', scratch, scratch, k, n, n,
                   nullptr, DLP_CLSC_NOT_SUPPORTED);
    expect_unreorder("A matrix", 'r', 'n', 'A', scratch, scratch, k, n, n,
                     nullptr, DLP_CLSC_NOT_SUPPORTED);

    expect_size("W matrix", 'r', 'n', 'W', k, n, nullptr,
                DLP_CLSC_NOT_SUPPORTED);
    expect_reorder("W matrix", 'r', 'n', 'W', scratch, scratch, k, n, n,
                   nullptr, DLP_CLSC_NOT_SUPPORTED);
    expect_unreorder("W matrix", 'r', 'n', 'W', scratch, scratch, k, n, n,
                     nullptr, DLP_CLSC_NOT_SUPPORTED);

    expect_size("invalid mat_type", 'r', 'n', 'Z', k, n, nullptr,
                DLP_CLSC_INVALID_MATRIX_TYPE);
    expect_reorder("invalid mat_type", 'r', 'n', 'Z', scratch, scratch, k, n, n,
                   nullptr, DLP_CLSC_INVALID_MATRIX_TYPE);
    expect_unreorder("invalid mat_type", 'r', 'n', 'Z', scratch, scratch, k, n,
                     n, nullptr, DLP_CLSC_INVALID_MATRIX_TYPE);

    expect_reorder("null input", 'r', 'n', 'B', nullptr, scratch, k, n, n,
                   nullptr, DLP_CLSC_NULL_POINTER);
    expect_reorder("null packed", 'r', 'n', 'B', scratch, nullptr, k, n, n,
                   nullptr, DLP_CLSC_NULL_POINTER);
    expect_unreorder("null packed", 'r', 'n', 'B', nullptr, scratch, k, n, n,
                     nullptr, DLP_CLSC_NULL_POINTER);
    expect_unreorder("null output", 'r', 'n', 'B', scratch, nullptr, k, n, n,
                     nullptr, DLP_CLSC_NULL_POINTER);

    for (auto dims : std::vector<std::pair<md_t, md_t>>{ { 0, n },
                                                         { k, 0 },
                                                         { -1, n },
                                                         { k, -1 } }) {
        const std::string what = std::string("dims k=")
                                 + std::to_string(dims.first)
                                 + " n=" + std::to_string(dims.second);
        expect_size(what.c_str(), 'r', 'n', 'B', dims.first, dims.second,
                    nullptr, DLP_CLSC_INVALID_MATRIX_DIMENSION);
        expect_reorder(what.c_str(), 'r', 'n', 'B', scratch, scratch,
                       dims.first, dims.second, n, nullptr,
                       DLP_CLSC_INVALID_MATRIX_DIMENSION);
        expect_unreorder(what.c_str(), 'r', 'n', 'B', scratch, scratch,
                         dims.first, dims.second, n, nullptr,
                         DLP_CLSC_INVALID_MATRIX_DIMENSION);
    }

    expect_size("k > DLP_MAX_GEMM_DIM", 'r', 'n', 'B', too_big, n, nullptr,
                DLP_CLSC_INVALID_MATRIX_DIMENSION);
    expect_size("n > DLP_MAX_GEMM_DIM", 'r', 'n', 'B', k, too_big, nullptr,
                DLP_CLSC_INVALID_MATRIX_DIMENSION);
    expect_reorder("k > DLP_MAX_GEMM_DIM", 'r', 'n', 'B', scratch, scratch,
                   too_big, n, n, nullptr, DLP_CLSC_INVALID_MATRIX_DIMENSION);
    expect_reorder("n > DLP_MAX_GEMM_DIM", 'r', 'n', 'B', scratch, scratch, k,
                   too_big, too_big, nullptr,
                   DLP_CLSC_INVALID_MATRIX_DIMENSION);
    expect_unreorder("k > DLP_MAX_GEMM_DIM", 'r', 'n', 'B', scratch, scratch,
                     too_big, n, n, nullptr, DLP_CLSC_INVALID_MATRIX_DIMENSION);
    expect_unreorder("n > DLP_MAX_GEMM_DIM", 'r', 'n', 'B', scratch, scratch, k,
                     too_big, too_big, nullptr,
                     DLP_CLSC_INVALID_MATRIX_DIMENSION);

    expect_reorder("row n ldb < n", 'r', 'n', 'B', scratch, scratch, k, n,
                   n - 1, nullptr, DLP_CLSC_INVALID_LEADING_DIMENSION);
    expect_unreorder("row n ldb < n", 'r', 'n', 'B', scratch, scratch, k, n,
                     n - 1, nullptr, DLP_CLSC_INVALID_LEADING_DIMENSION);
    expect_reorder("row t ldb < k", 'r', 't', 'B', scratch, scratch, k, n,
                   k - 1, nullptr, DLP_CLSC_INVALID_LEADING_DIMENSION);
    expect_unreorder("row t ldb < k", 'r', 't', 'B', scratch, scratch, k, n,
                     k - 1, nullptr, DLP_CLSC_INVALID_LEADING_DIMENSION);
    expect_reorder("col n ldb < k", 'c', 'n', 'B', scratch, scratch, k, n,
                   k - 1, nullptr, DLP_CLSC_INVALID_LEADING_DIMENSION);
    expect_unreorder("col n ldb < k", 'c', 'n', 'B', scratch, scratch, k, n,
                     k - 1, nullptr, DLP_CLSC_INVALID_LEADING_DIMENSION);
    expect_reorder("col t ldb < n", 'c', 't', 'B', scratch, scratch, k, n,
                   n - 1, nullptr, DLP_CLSC_INVALID_LEADING_DIMENSION);
    expect_unreorder("col t ldb < n", 'c', 't', 'B', scratch, scratch, k, n,
                     n - 1, nullptr, DLP_CLSC_INVALID_LEADING_DIMENSION);

    expect_reorder("ldb > DLP_MAX_GEMM_DIM", 'r', 'n', 'B', scratch, scratch, k,
                   n, too_big, nullptr, DLP_CLSC_INVALID_LEADING_DIMENSION);
    expect_unreorder("ldb > DLP_MAX_GEMM_DIM", 'r', 'n', 'B', scratch, scratch,
                     k, n, too_big, nullptr,
                     DLP_CLSC_INVALID_LEADING_DIMENSION);

    if (min_NR > 1) {
        // NR=min_NR/2 is a coverage width the layout cannot pack. NC is a
        // multiple of that NR so metadata validation is not the one that
        // fires.
        dlp_gemm_blocking_t bp = { 0, min_NR / 2, 0, min_NR, 256 };
        expect_size("NR % min_NR != 0", 'r', 'n', 'B', k, n, &bp,
                    DLP_CLSC_INVALID_BLOCK_PARAMS);
        expect_reorder("NR % min_NR != 0", 'r', 'n', 'B', scratch, scratch, k,
                       n, n, &bp, DLP_CLSC_INVALID_BLOCK_PARAMS);
        expect_unreorder("NR % min_NR != 0", 'r', 'n', 'B', scratch, scratch, k,
                         n, n, &bp, DLP_CLSC_INVALID_BLOCK_PARAMS);
    } else {
        // f32 min_NR == NR, so the layout check cannot fail on granularity.
        // NC not a multiple of NR is still illegal.
        dlp_gemm_blocking_t bp = { 0, 16, 0, 24, 256 };
        expect_size("NC % NR != 0", 'r', 'n', 'B', k, n, &bp,
                    DLP_CLSC_INVALID_BLOCK_PARAMS);
        expect_reorder("NC % NR != 0", 'r', 'n', 'B', scratch, scratch, k, n, n,
                       &bp, DLP_CLSC_INVALID_BLOCK_PARAMS);
        expect_unreorder("NC % NR != 0", 'r', 'n', 'B', scratch, scratch, k, n,
                         n, &bp, DLP_CLSC_INVALID_BLOCK_PARAMS);
    }

    if (k_factor > 1) {
        dlp_gemm_blocking_t bp = { 0, min_NR, 0, min_NR * 4, k_factor + 1 };
        expect_size("KC % k_factor != 0", 'r', 'n', 'B', k, n, &bp,
                    DLP_CLSC_INVALID_BLOCK_PARAMS);
        expect_reorder("KC % k_factor != 0", 'r', 'n', 'B', scratch, scratch, k,
                       n, n, &bp, DLP_CLSC_INVALID_BLOCK_PARAMS);
        expect_unreorder("KC % k_factor != 0", 'r', 'n', 'B', scratch, scratch,
                         k, n, n, &bp, DLP_CLSC_INVALID_BLOCK_PARAMS);
    }

    {
        dlp_metadata_t md = fresh_metadata();
        msz_t          sz =
            size_fn('r', 'n', 'B', (md_t)INT32_MAX, (md_t)INT32_MAX, &md);
        if (sz == 0) {
            EXPECT_EQ(md.error_hndl.error_code,
                      DLP_CLSC_INVALID_MATRIX_DIMENSION)
                << label << " INT32_MAX size must not wrap to a silent 0";
        } else {
            EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_SUCCESS)
                << label << " INT32_MAX size";
            EXPECT_GT(sz, (msz_t)(1ULL << 40))
                << label << " INT32_MAX size wrapped to a small msz_t";
        }
    }

    EXPECT_GT(size_fn('r', 'n', 'B', k, n, nullptr), (msz_t)0)
        << label << " size with NULL metadata";
}

TEST(RefApiNegative, F32F32F32OF32)
{
    ref_api_negative<float>("f32f32f32of32",
                            aocl_get_reorder_buf_size_f32f32f32of32,
                            aocl_reorder_f32f32f32of32_reference,
                            aocl_unreorder_f32f32f32of32_reference, 1, 1);
}

TEST(RefApiNegative, Bf16Bf16F32OF32)
{
    ref_api_negative<bfloat16>("bf16bf16f32of32",
                               aocl_get_reorder_buf_size_bf16bf16f32of32,
                               aocl_reorder_bf16bf16f32of32_reference,
                               aocl_unreorder_bf16bf16f32of32_reference, 16, 2);
}

TEST(RefApiNegative, S8S8S32OS32)
{
    ref_api_negative<int8_t>("s8s8s32os32",
                             aocl_get_reorder_buf_size_s8s8s32os32,
                             aocl_reorder_s8s8s32os32_reference,
                             aocl_unreorder_s8s8s32os32_reference, 16, 4);
}

TEST(RefApiNegative, U8S8S32OS32)
{
    ref_api_negative<int8_t>("u8s8s32os32",
                             aocl_get_reorder_buf_size_u8s8s32os32,
                             aocl_reorder_u8s8s32os32_reference,
                             aocl_unreorder_u8s8s32os32_reference, 16, 4);
}

TEST(RefApiNegative, F16F16F16OF16)
{
    ref_api_negative<float16>("f16f16f16of16",
                              aocl_get_reorder_buf_size_f16f16f16of16,
                              aocl_reorder_f16f16f16of16_reference,
                              aocl_unreorder_f16f16f16of16_reference, 32, 1);
}

// ---------------------------------------------------------------------------
// Parameter validation. These document what the APIs reject.
// ---------------------------------------------------------------------------

TEST(UnreorderValidation, RejectsAMatrix)
{
    std::vector<float> in(64 * 64, 0.0f);
    std::vector<float> out(64 * 64, 0.0f);

    dlp_metadata_t md = fresh_metadata();
    aocl_unreorder_f32f32f32of32_reference('r', 'n', 'A', in.data(), out.data(),
                                           64, 64, 64, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST(UnreorderValidation, RejectsWMatrix)
{
    std::vector<float> in(64 * 64, 0.0f);
    std::vector<float> out(64 * 64, 0.0f);

    dlp_metadata_t md = fresh_metadata();
    aocl_unreorder_f32f32f32of32_reference('r', 'n', 'W', in.data(), out.data(),
                                           64, 64, 64, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
}

TEST(UnreorderValidation, RejectsNullBuffers)
{
    std::vector<float> buf(64 * 64, 0.0f);

    dlp_metadata_t md = fresh_metadata();
    aocl_unreorder_f32f32f32of32_reference('r', 'n', 'B', nullptr, buf.data(),
                                           64, 64, 64, &md);
    EXPECT_NE(md.error_hndl.error_code, DLP_CLSC_SUCCESS);

    md = fresh_metadata();
    aocl_unreorder_f32f32f32of32_reference('r', 'n', 'B', buf.data(), nullptr,
                                           64, 64, 64, &md);
    EXPECT_NE(md.error_hndl.error_code, DLP_CLSC_SUCCESS);
}

TEST(UnreorderValidation, RejectsNonPositiveDimensions)
{
    std::vector<float> buf(64 * 64, 0.0f);

    for (auto dims : std::vector<std::pair<md_t, md_t>>{ { 0, 64 },
                                                         { 64, 0 },
                                                         { -1, 64 },
                                                         { 64, -1 } }) {
        dlp_metadata_t md = fresh_metadata();
        aocl_unreorder_f32f32f32of32_reference('r', 'n', 'B', buf.data(),
                                               buf.data(), dims.first,
                                               dims.second, 64, &md);
        EXPECT_NE(md.error_hndl.error_code, DLP_CLSC_SUCCESS)
            << "k=" << dims.first << " n=" << dims.second << " was accepted";
    }
}

// A dimension above DLP_MAX_GEMM_DIM must be rejected by the validator rather
// than reaching stride math and wrapping (CWE-190 -> CWE-787).
TEST(UnreorderValidation, RejectsOversizedDimensions)
{
    std::vector<float> buf(64 * 64, 0.0f);
    const md_t         too_big = (md_t)INT32_MAX + 1;

    dlp_metadata_t md = fresh_metadata();
    aocl_unreorder_f32f32f32of32_reference('r', 'n', 'B', buf.data(),
                                           buf.data(), too_big, 16, 16, &md);
    EXPECT_NE(md.error_hndl.error_code, DLP_CLSC_SUCCESS);

    md = fresh_metadata();
    aocl_unreorder_f32f32f32of32_reference(
        'r', 'n', 'B', buf.data(), buf.data(), 16, too_big, too_big, &md);
    EXPECT_NE(md.error_hndl.error_code, DLP_CLSC_SUCCESS);
}

// metadata is optional; passing NULL must not crash.
TEST(UnreorderValidation, AcceptsNullMetadata)
{
    const md_t k = 64, n = 64;

    dlp_metadata_t md = fresh_metadata();
    msz_t          buf_sz =
        aocl_get_reorder_buf_size_f32f32f32of32('r', 'n', 'B', k, n, &md);
    ASSERT_GT(buf_sz, 0u);

    std::vector<float>   in((size_t)k * n);
    std::vector<float>   out((size_t)k * n, 0.0f);
    std::vector<uint8_t> packed((size_t)buf_sz);
    fill_pattern(in.data(), in.size());

    aocl_reorder_f32f32f32of32('r', 'n', 'B', in.data(), (float*)packed.data(),
                               k, n, n, nullptr);
    aocl_unreorder_f32f32f32of32_reference('r', 'n', 'B', (float*)packed.data(),
                                           out.data(), k, n, n, nullptr);

    EXPECT_EQ(std::memcmp(in.data(), out.data(), in.size() * sizeof(float)), 0);
}

// An ldb larger than the matrix width leaves padding columns between rows;
// the un-reorder must honour the stride and leave that padding untouched.
TEST(UnreorderRoundTrip, HonoursPaddedLeadingDimension)
{
    const md_t k = 65, n = 33;
    const md_t ldb = 48; // > n, so each row is followed by padding.

    dlp_metadata_t md = fresh_metadata();
    msz_t          buf_sz =
        aocl_get_reorder_buf_size_f32f32f32of32('r', 'n', 'B', k, n, &md);
    ASSERT_GT(buf_sz, 0u);

    std::vector<float>   in((size_t)k * ldb);
    std::vector<float>   out((size_t)k * ldb);
    std::vector<uint8_t> packed((size_t)buf_sz);
    fill_pattern(in.data(), in.size());

    const float sentinel = -12345.0f;
    for (size_t i = 0; i < out.size(); i++) {
        out[i] = sentinel;
    }

    md = fresh_metadata();
    aocl_reorder_f32f32f32of32('r', 'n', 'B', in.data(), (float*)packed.data(),
                               k, n, ldb, &md);
    ASSERT_EQ(md.error_hndl.error_code, DLP_CLSC_SUCCESS);

    md = fresh_metadata();
    aocl_unreorder_f32f32f32of32_reference('r', 'n', 'B', (float*)packed.data(),
                                           out.data(), k, n, ldb, &md);
    ASSERT_EQ(md.error_hndl.error_code, DLP_CLSC_SUCCESS);

    for (md_t row = 0; row < k; row++) {
        for (md_t col = 0; col < n; col++) {
            EXPECT_EQ(in[row * ldb + col], out[row * ldb + col])
                << "data differs at row " << row << " col " << col;
        }
        for (md_t col = n; col < ldb; col++) {
            EXPECT_EQ(out[row * ldb + col], sentinel)
                << "padding overwritten at row " << row << " col " << col;
        }
    }
}

TEST(UnreorderValidation, RejectsInvalidTranspose)
{
    std::vector<float> buf(64 * 64, 0.0f);
    dlp_metadata_t     md = fresh_metadata();
    aocl_unreorder_f32f32f32of32_reference('r', 'x', 'B', buf.data(),
                                           buf.data(), 64, 64, 64, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_TRANSPOSE);
}

// Row-major trans='t' needs ldb >= k, not ldb >= n.
TEST(UnreorderValidation, RejectsTooSmallLdbForTranspose)
{
    const md_t         k = 32, n = 16;
    std::vector<float> buf(64 * 64, 0.0f);
    dlp_metadata_t     md = fresh_metadata();
    aocl_unreorder_f32f32f32of32_reference('r', 't', 'B', buf.data(),
                                           buf.data(), k, n, n, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_LEADING_DIMENSION);
}

// input -> reorder(trans_in) -> unreorder(trans_out) -> output, compared on
// the logical k x n. trans_in and trans_out may differ; extra_ldb covers a
// padded leading dimension on both sides.
TEST(UnreorderRoundTrip, TransFlagMatchesLogicalInput)
{
    static const Shape cases[] = { { 17, 33 },
                                   { 64, 1 },
                                   { 5, 7 },
                                   { 63, 129 } };

    int               supported = 0;
    PinnedThreadCount pin((md_t)1);
    for (char order : { 'r', 'c' }) {
        for (char trans_in : { 'n', 't' }) {
            for (char trans_out : { 'n', 't' }) {
                for (const Shape& s : cases) {
                    const std::string desc =
                        std::string("f32 trans order=") + order
                        + " in=" + trans_in + " out=" + trans_out + " k="
                        + std::to_string(s.k) + " n=" + std::to_string(s.n);
                    SCOPED_TRACE(desc);
                    set_current_case(desc);
                    if (round_trip<float>(
                            aocl_get_reorder_buf_size_f32f32f32of32,
                            aocl_reorder_f32f32f32of32,
                            aocl_unreorder_f32f32f32of32_reference, order, s,
                            nullptr, trans_in, trans_out, /*extra_ldb=*/4)) {
                        supported++;
                    }
                }
            }
        }
    }
    if (supported == 0) {
        GTEST_SKIP() << "f32f32f32of32 is not supported on this processor";
    }
}

} // namespace
