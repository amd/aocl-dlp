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
#include "kernels/f32f32f32/dlp_gemm_pack_f32.h"
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
           const dlp_gemm_blocking_t* bp              = nullptr,
           char                       trans_in        = 'n',
           char                       trans_out       = 'n',
           md_t                       extra_ldb       = 0,
           const dlp_gemm_hints_t*    gemm_hints      = nullptr,
           const md_t*                sym_quant_group = nullptr)
{
    const md_t   ldb_in  = plain_ldb(order, trans_in, s.k, s.n) + extra_ldb;
    const md_t   ldb_out = plain_ldb(order, trans_out, s.k, s.n) + extra_ldb;
    const size_t in_n    = plain_elems(order, trans_in, s.k, s.n, ldb_in);
    const size_t out_n   = plain_elems(order, trans_out, s.k, s.n, ldb_out);

    // Symmetric-quant reorder reads only the B quantization group size. A
    // value of zero means one group spanning K. The object has to outlive
    // every metadata copy handed to the size, reorder and unreorder calls.
    dlp_quant_op_t quant_op;
    std::memset(&quant_op, 0, sizeof(quant_op));
    if (sym_quant_group != nullptr) {
        quant_op.quant_op_kind = DLP_QUANT_OP_QUANTIZE;
        quant_op.src_type      = DLP_S8;
        quant_op.dst_type      = DLP_S8;
        quant_op.group_size    = *sym_quant_group;
    }

    // The reorder buffer does not record its blocking or hint-derived NR.
    // Every operation which interprets it therefore needs the same metadata
    // configuration the reorder used.
    auto with_metadata = [&]() {
        dlp_metadata_t m = fresh_metadata();
        m.block_params   = const_cast<dlp_gemm_blocking_t*>(bp);
        m.gemm_hints     = const_cast<dlp_gemm_hints_t*>(gemm_hints);
        if (sym_quant_group != nullptr) {
            m.b_quant_op = &quant_op;
        }
        return m;
    };

    dlp_metadata_t md     = with_metadata();
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

    md = with_metadata();
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

    md = with_metadata();
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

// group_size 0 is one group over K. Otherwise it must be positive, no larger
// than K, and a multiple of align unless it already spans K. align is 4 for
// symmetric quant s8 and s8s4 and 2 for bf16s4.
bool
quant_group_legal(md_t k, md_t group_size, md_t align)
{
    if (group_size == 0) {
        return true;
    }
    if ((group_size <= 0) || (group_size > k) || (align <= 0)) {
        return false;
    }
    return (group_size == k) || ((group_size % align) == 0);
}

// Same shape, layout and thread sweep as the plain round trip, once per
// quantization group. A group that is illegal for a particular K is skipped;
// the API rejects it rather than packing a layout the unreorder could invert.
template<typename T, typename ReorderFn, typename UnreorderFn>
void
round_trip_sym_quant_groups(const char*               label,
                            SizeFn                    size_fn,
                            ReorderFn                 reorder_fn,
                            UnreorderFn               unreorder_fn,
                            const std::vector<Shape>& cases,
                            const std::vector<md_t>&  group_sizes,
                            bool                      vary_transpose,
                            bool                      include_shape_k)
{
    int supported = 0;
    for (md_t group_size : group_sizes) {
        for (md_t threads : { (md_t)1, (md_t)-1 }) {
            PinnedThreadCount pin(threads);
            for (char order : { 'r', 'c' }) {
                const char trans_values[] = { 'n', 't' };
                const int  trans_n        = vary_transpose ? 2 : 1;
                for (int trans_i = 0; trans_i < trans_n; trans_i++) {
                    for (int trans_o = 0; trans_o < trans_n; trans_o++) {
                        const char trans_in  = trans_values[trans_i];
                        const char trans_out = trans_values[trans_o];
                        for (const Shape& s : cases) {
                            if (!quant_group_legal(s.k, group_size, 4)) {
                                continue;
                            }
                            const std::string desc =
                                std::string(label) + " threads="
                                + (threads < 0 ? std::string("default")
                                               : std::to_string(threads))
                                + " group=" + std::to_string(group_size)
                                + " order=" + order + " trans_in=" + trans_in
                                + " trans_out=" + trans_out
                                + " k=" + std::to_string(s.k)
                                + " n=" + std::to_string(s.n);
                            SCOPED_TRACE(desc);
                            set_current_case(desc);
                            if (round_trip<T>(size_fn, reorder_fn, unreorder_fn,
                                              order, s, nullptr, trans_in,
                                              trans_out, 0, nullptr,
                                              &group_size)) {
                                supported++;
                            }
                        }
                    }
                }
            }
        }
    }
    if (include_shape_k) {
        for (const Shape& s : cases) {
            if (!quant_group_legal(s.k, s.k, 4)) {
                continue;
            }
            bool listed = false;
            for (md_t group_size : group_sizes) {
                if (group_size == s.k) {
                    listed = true;
                }
            }
            if (listed) {
                continue;
            }
            PinnedThreadCount pin((md_t)1);
            const std::string desc = std::string(label)
                                     + " group=k=" + std::to_string(s.k)
                                     + " order=r n=" + std::to_string(s.n);
            SCOPED_TRACE(desc);
            set_current_case(desc);
            if (round_trip<T>(size_fn, reorder_fn, unreorder_fn, 'r', s,
                              nullptr, 'n', 'n', 0, nullptr, &s.k)) {
                supported++;
            }
        }
    }
    if (supported == 0) {
        GTEST_SKIP() << label << " is not supported on this processor";
    }
}

// Drives every shape with each set of GEMM hints. As with blocking parameters,
// the reorder buffer carries no record of the hint-derived NR, so each
// un-reorder call must receive the same hints the reorder received.
template<typename T, typename ReorderFn, typename UnreorderFn>
void
round_trip_gemm_hints(const char*                          label,
                      SizeFn                               size_fn,
                      ReorderFn                            reorder_fn,
                      UnreorderFn                          unreorder_fn,
                      const std::vector<dlp_gemm_hints_t>& hint_params)
{
    int supported = 0;
    for (size_t hint_i = 0; hint_i < hint_params.size(); hint_i++) {
        const dlp_gemm_hints_t& hints = hint_params[hint_i];
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
                            + " m_hint=" + std::to_string(hints.m_hint)
                            + " nt_hint=" + std::to_string(hints.nt_hint)
                            + " order=" + order + " k=" + std::to_string(s.k)
                            + " n=" + std::to_string(s.n);
                        SCOPED_TRACE(desc);
                        set_current_case(desc);
                        if (round_trip<T>(size_fn, reorder_fn, unreorder_fn,
                                          order, s, nullptr, 'n', 'n', 0,
                                          &hints)) {
                            supported++;
                        }
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

// A serial consumer, a modestly parallel consumer and a wider, highly
// parallel consumer. All values are positive so the decision engine has a
// complete model of the GEMM that will consume the reordered buffer.
const std::vector<dlp_gemm_hints_t>&
gemm_hint_params()
{
    static const std::vector<dlp_gemm_hints_t> hints = {
        { 4, 32 },
        { 16, 8 },
        { 64, 32 },
    };
    return hints;
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
    // 2k bound, so a large dimension is still crossed with a fringe, a full
    // NR, a full NC-ish panel and the other extreme without constructing the
    // very large matrices that made this sweep dominate CI runtime.
    static const md_t partners[] = { 1,   7,   16,  32,   64,  65,
                                     128, 256, 512, 1024, 2048 };
    for (md_t p : partners) {
        if (p == n) {
            return true;
        }
    }
    return false;
}

// 1 <= k,n <= 2k, non-linear. Below 256 every listed pair runs so packing
// fringes are covered densely; past that, each large value is crossed with
// the partner axis (and with itself). The separate round-trip and blocking
// tests still cover the other widths and awkward KC/NC combinations.
std::vector<Shape>
nr64_sweep_shapes()
{
    static const md_t dims[] = { 1,   2,    3,    5,    7,    8,    15,
                                 16,  17,   31,   32,   33,   63,   64,
                                 65,  96,   127,  128,  129,  192,  255,
                                 256, 257,  320,  384,  511,  512,  513,
                                 768, 1023, 1024, 1025, 1536, 2047, 2048 };

    std::vector<Shape> out;
    for (md_t k : dims) {
        for (md_t n : dims) {
            if ((k <= 256 && n <= 256) || is_sweep_partner(n) || k == n) {
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

// Symmetric-quant reorder uses the same weight-panel nesting as plain s8,
// with KC aligned to the quantization group. group_size 0 is one group over
// K. Groups that do not divide K stay on the wide packer, including n == 1.
TEST(UnreorderRoundTrip, S8S8S32OS32SymQuantReference)
{
    std::vector<Shape>        cases = shapes();
    const std::vector<Shape>& wide  = large_shapes();
    cases.insert(cases.end(), wide.begin(), wide.end());

    round_trip_sym_quant_groups<int8_t>(
        "s8s8s32os32_sym_quant",
        aocl_get_reorder_buf_size_s8s8s32os32_sym_quant,
        aocl_reorder_s8s8s32os32_sym_quant,
        aocl_unreorder_s8s8s32os32_sym_quant, cases,
        { (md_t)0, (md_t)4, (md_t)16, (md_t)64 },
        /*vary_transpose=*/false, /*include_shape_k=*/false);
}

// Table KC for s8 is 2048 (Zen through Zen5) or 4096 (Zen6), and the groups
// above all divide both, so they never change it. A group of 12 divides
// neither, so KC shrinks; both shapes pass that boundary. A group of 6144 is
// larger than either table KC, so KC grows. 6144 is legal only for K = 8195;
// the K = 4101 pair is skipped because the group is larger than K.
TEST(UnreorderRoundTrip, S8S8S32OS32SymQuantKcAlign)
{
    const std::vector<Shape> cases = {
        { 4101, 32 },
        { 8195, 32 },
    };

    round_trip_sym_quant_groups<int8_t>(
        "s8s8s32os32_sym_quant_kc_align",
        aocl_get_reorder_buf_size_s8s8s32os32_sym_quant,
        aocl_reorder_s8s8s32os32_sym_quant,
        aocl_unreorder_s8s8s32os32_sym_quant, cases, { (md_t)12, (md_t)6144 },
        /*vary_transpose=*/false, /*include_shape_k=*/false);
}

// The main sweep keeps transpose fixed. These n > 1 shapes cover every
// input/output transpose pair on the wide packer.
TEST(UnreorderRoundTrip, S8S8S32OS32SymQuantTranspose)
{
    const std::vector<Shape> cases = {
        { 5, 7 },
        { 17, 33 },
        { 64, 80 },
    };

    round_trip_sym_quant_groups<int8_t>(
        "s8s8s32os32_sym_quant_transpose",
        aocl_get_reorder_buf_size_s8s8s32os32_sym_quant,
        aocl_reorder_s8s8s32os32_sym_quant,
        aocl_unreorder_s8s8s32os32_sym_quant, cases,
        { (md_t)0, (md_t)4, (md_t)16 },
        /*vary_transpose=*/true, /*include_shape_k=*/false);
}

// n == 1 is a tight column when the group divides K, and the normal padded
// packer otherwise. group == K covers odd lengths that cannot use a smaller
// group, and the transpose pairs exercise the strided copy.
TEST(UnreorderRoundTrip, S8S8S32OS32SymQuantN1)
{
    const std::vector<Shape> cases = {
        { 1, 1 },  { 3, 1 },  { 4, 1 },   { 5, 1 },   { 7, 1 },   { 8, 1 },
        { 12, 1 }, { 15, 1 }, { 16, 1 },  { 17, 1 },  { 31, 1 },  { 32, 1 },
        { 63, 1 }, { 64, 1 }, { 100, 1 }, { 128, 1 }, { 1024, 1 }
    };

    round_trip_sym_quant_groups<int8_t>(
        "s8s8s32os32_sym_quant_n1",
        aocl_get_reorder_buf_size_s8s8s32os32_sym_quant,
        aocl_reorder_s8s8s32os32_sym_quant,
        aocl_unreorder_s8s8s32os32_sym_quant, cases,
        { (md_t)0, (md_t)4, (md_t)8, (md_t)16, (md_t)32, (md_t)64 },
        /*vary_transpose=*/true, /*include_shape_k=*/true);
}

TEST(UnreorderSymQuantValidation, RejectsIllegalGroupSize)
{
    const md_t          k = 16;
    const md_t          n = 16;
    std::vector<int8_t> packed(k * n * 8, 0);
    std::vector<int8_t> out(k * n, 0);

    auto reject = [&](md_t group_size) {
        dlp_quant_op_t quant;
        std::memset(&quant, 0, sizeof(quant));
        quant.quant_op_kind = DLP_QUANT_OP_QUANTIZE;
        quant.src_type      = DLP_S8;
        quant.dst_type      = DLP_S8;
        quant.group_size    = group_size;

        dlp_metadata_t md = fresh_metadata();
        md.b_quant_op     = &quant;
        aocl_unreorder_s8s8s32os32_sym_quant('r', 'n', 'B', packed.data(),
                                             out.data(), k, n, n, &md);
        EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_GROUP_DIMENSION)
            << "group_size=" << group_size;
    };

    reject(6);        // Not a multiple of 4 and not full K.
    reject(32);       // Larger than K.
    reject((md_t)-4); // Negative, so not the full-K spelling of zero.
}

// The reference unreorder has no AVX-512 VNNI gate, so it always reports the
// group or matrix-type error. Size and reorder do have that gate and return
// DLP_CLSC_NOT_SUPPORTED first when VNNI is absent, so their specific error is
// checked only when a legal call is supported. A null metadata pointer cannot
// store an error code.
TEST(UnreorderSymQuantValidation, RejectsBadMetadataAndMatrixType)
{
    const md_t          k = 16;
    const md_t          n = 16;
    std::vector<int8_t> in(k * n, 7);
    std::vector<int8_t> packed(k * n * 8, 3);
    std::vector<int8_t> out(k * n, 9);

    auto make_quant = [](dlp_quant_op_t* op, md_t group_size) {
        std::memset(op, 0, sizeof(*op));
        op->quant_op_kind = DLP_QUANT_OP_QUANTIZE;
        op->src_type      = DLP_S8;
        op->dst_type      = DLP_S8;
        op->group_size    = group_size;
    };

    dlp_quant_op_t probe_op;
    make_quant(&probe_op, 4);
    dlp_metadata_t probe_md = fresh_metadata();
    probe_md.b_quant_op     = &probe_op;
    const bool supported    = aocl_get_reorder_buf_size_s8s8s32os32_sym_quant(
                               'r', 'n', 'B', k, n, &probe_md)
                           != 0;

    auto expect_all = [&](const char* what, dlp_metadata_t* md, char mat_type,
                          dlp_clsc_err_t want) {
        const bool check_early_api = supported
                                     || (want == DLP_CLSC_NOT_SUPPORTED);
        dlp_metadata_t size_md = fresh_metadata();
        if (md != nullptr) {
            size_md = *md;
        }
        dlp_metadata_t* size_arg = (md == nullptr) ? nullptr : &size_md;
        msz_t           sz = aocl_get_reorder_buf_size_s8s8s32os32_sym_quant(
            'r', 'n', mat_type, k, n, size_arg);
        if ((size_arg != nullptr) && check_early_api) {
            EXPECT_EQ(size_arg->error_hndl.error_code, want) << what << " size";
        }
        EXPECT_EQ(sz, (msz_t)0) << what << " size";

        dlp_metadata_t reorder_md = fresh_metadata();
        if (md != nullptr) {
            reorder_md = *md;
        }
        const std::vector<int8_t> packed_before = packed;
        aocl_reorder_s8s8s32os32_sym_quant(
            'r', 'n', mat_type, in.data(), packed.data(), k, n, n,
            (md == nullptr) ? nullptr : &reorder_md);
        if ((md != nullptr) && check_early_api) {
            EXPECT_EQ(reorder_md.error_hndl.error_code, want)
                << what << " reorder";
        }
        EXPECT_EQ(packed, packed_before) << what << " reorder wrote output";

        dlp_metadata_t unreorder_md = fresh_metadata();
        if (md != nullptr) {
            unreorder_md = *md;
        }
        const std::vector<int8_t> out_before = out;
        aocl_unreorder_s8s8s32os32_sym_quant(
            'r', 'n', mat_type, packed.data(), out.data(), k, n, n,
            (md == nullptr) ? nullptr : &unreorder_md);
        if (md != nullptr) {
            EXPECT_EQ(unreorder_md.error_hndl.error_code, want)
                << what << " unreorder";
        }
        EXPECT_EQ(out, out_before) << what << " unreorder wrote output";
    };

    {
        dlp_quant_op_t b_op;
        make_quant(&b_op, 4);
        b_op.quant_op_kind = DLP_QUANT_OP_DEQUANTIZE;
        dlp_metadata_t md  = fresh_metadata();
        md.b_quant_op      = &b_op;
        expect_all("dequantize op", &md, 'B', DLP_CLSC_NOT_SUPPORTED);
    }

    {
        dlp_quant_op_t a_op;
        dlp_quant_op_t b_op;
        make_quant(&a_op, 4);
        make_quant(&b_op, 8);
        dlp_metadata_t md = fresh_metadata();
        md.a_quant_op     = &a_op;
        md.b_quant_op     = &b_op;
        expect_all("mismatched groups", &md, 'B',
                   DLP_CLSC_INVALID_GROUP_DIMENSION);
    }

    expect_all("null metadata", nullptr, 'B', DLP_CLSC_NULL_POINTER);

    {
        dlp_quant_op_t b_op;
        make_quant(&b_op, 4);
        dlp_metadata_t md = fresh_metadata();
        md.b_quant_op     = &b_op;
        expect_all("A matrix", &md, 'A', DLP_CLSC_NOT_SUPPORTED);
        expect_all("W matrix", &md, 'W', DLP_CLSC_NOT_SUPPORTED);
    }
}

// No quantization op means one group over K. An A op alone supplies the group
// when B's op is absent. Both must still round-trip.
TEST(UnreorderSymQuantValidation, MissingQuantOpUsesFullKOrAGroup)
{
    const Shape s{ 16, 16 };
    if (!round_trip<int8_t>(aocl_get_reorder_buf_size_s8s8s32os32_sym_quant,
                            aocl_reorder_s8s8s32os32_sym_quant,
                            aocl_unreorder_s8s8s32os32_sym_quant, 'r', s)) {
        GTEST_SKIP() << "s8s8s32os32_sym_quant is not supported";
    }

    dlp_quant_op_t a_op;
    std::memset(&a_op, 0, sizeof(a_op));
    a_op.quant_op_kind = DLP_QUANT_OP_QUANTIZE;
    a_op.src_type      = DLP_S8;
    a_op.dst_type      = DLP_S8;
    a_op.group_size    = 4;

    const md_t          k     = s.k;
    const md_t          n     = s.n;
    const size_t        elems = (size_t)k * (size_t)n;
    std::vector<int8_t> input(elems);
    fill_pattern(input.data(), elems);

    dlp_metadata_t md = fresh_metadata();
    md.a_quant_op     = &a_op;
    msz_t buf_sz      = aocl_get_reorder_buf_size_s8s8s32os32_sym_quant(
        'r', 'n', 'B', k, n, &md);
    ASSERT_NE(buf_sz, (msz_t)0);
    ASSERT_EQ(md.error_hndl.error_code, DLP_CLSC_SUCCESS);

    std::vector<int8_t> packed((size_t)buf_sz, 0);
    std::vector<int8_t> output(elems, 0);
    md            = fresh_metadata();
    md.a_quant_op = &a_op;
    aocl_reorder_s8s8s32os32_sym_quant('r', 'n', 'B', input.data(),
                                       packed.data(), k, n, n, &md);
    ASSERT_EQ(md.error_hndl.error_code, DLP_CLSC_SUCCESS);

    md            = fresh_metadata();
    md.a_quant_op = &a_op;
    aocl_unreorder_s8s8s32os32_sym_quant('r', 'n', 'B', packed.data(),
                                         output.data(), k, n, n, &md);
    ASSERT_EQ(md.error_hndl.error_code, DLP_CLSC_SUCCESS);
    EXPECT_EQ(input, output);
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

// Exercise the hint-derived layout for several predicted GEMM shapes and
// thread pools. The same hint set is supplied to the size query, reorder and
// un-reorder because the buffer has no header recording the derived NR.
TEST(UnreorderRoundTrip, Bf16Bf16F32OF32WithGemmHints)
{
    round_trip_gemm_hints<bfloat16>(
        "bf16bf16f32of32 hints", aocl_get_reorder_buf_size_bf16bf16f32of32,
        aocl_reorder_bf16bf16f32of32, aocl_unreorder_bf16bf16f32of32,
        gemm_hint_params());
}

TEST(UnreorderRoundTrip, Bf16Bf16F32OF32ReferenceWithGemmHints)
{
    round_trip_gemm_hints<bfloat16>(
        "bf16bf16f32of32 hints", aocl_get_reorder_buf_size_bf16bf16f32of32,
        aocl_reorder_bf16bf16f32of32, aocl_unreorder_bf16bf16f32of32_reference,
        gemm_hint_params());
}

TEST(UnreorderRoundTrip, S8S8S32OS32ReferenceWithGemmHints)
{
    round_trip_gemm_hints<int8_t>(
        "s8s8s32os32 hints", aocl_get_reorder_buf_size_s8s8s32os32,
        aocl_reorder_s8s8s32os32, aocl_unreorder_s8s8s32os32_reference,
        gemm_hint_params());
}

TEST(UnreorderRoundTrip, U8S8S32OS32ReferenceWithGemmHints)
{
    round_trip_gemm_hints<int8_t>(
        "u8s8s32os32 hints", aocl_get_reorder_buf_size_u8s8s32os32,
        aocl_reorder_u8s8s32os32, aocl_unreorder_u8s8s32os32_reference,
        gemm_hint_params());
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

// 'W' is a distinct matrix tag from 'B'. The size calculator and the
// reference reorder reject it. The tuned reorder must do the same, and
// must not write a packed buffer the size API said was 0 bytes.
TEST(ReorderTunedValidation, RejectsWMatrix)
{
    const md_t         k = 8, n = 8;
    std::vector<float> src(k * n, 1.0f);

    // Size the packed buffer from the API. The packer pads n to NR, which is
    // 16 on Zen3 but 64 on Zen4 and Zen5, so a fixed allowance overflows.
    dlp_metadata_t szmd = fresh_metadata();
    const msz_t    bytes =
        aocl_get_reorder_buf_size_f32f32f32of32('r', 'n', 'B', k, n, &szmd);
    if (szmd.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED) {
        GTEST_SKIP() << "f32 reorder not supported on this processor";
    }
    ASSERT_EQ(szmd.error_hndl.error_code, DLP_CLSC_SUCCESS);
    ASSERT_GT(bytes, (msz_t)0);
    std::vector<float> dst(bytes / sizeof(float), 0.0f);

    dlp_metadata_t md = fresh_metadata();
    aocl_reorder_f32f32f32of32('r', 'n', 'B', src.data(), dst.data(), k, n, n,
                               &md);
    ASSERT_EQ(md.error_hndl.error_code, DLP_CLSC_SUCCESS);

    dst.assign(dst.size(), 0.0f);
    md = fresh_metadata();
    aocl_reorder_f32f32f32of32('r', 'n', 'W', src.data(), dst.data(), k, n, n,
                               &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
    EXPECT_EQ(dst[0], 0.0f);

    md = fresh_metadata();
    msz_t sz =
        aocl_get_reorder_buf_size_f32f32f32of32('r', 'n', 'W', k, n, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);
    EXPECT_EQ(sz, (msz_t)0);
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
}

TEST(ReorderTunedValidation, RejectsOversizedBf16PackBJitNr)
{
    const Shape shape = { 4, 16 };

    // Establish support using round_trip, consistent with
    // SupportsDynamicInt8Nr. If not supported on this processor, the test
    // skips.
    if (!round_trip<bfloat16>(aocl_get_reorder_buf_size_bf16bf16f32of32,
                              aocl_reorder_bf16bf16f32of32,
                              aocl_unreorder_bf16bf16f32of32_reference, 'r',
                              shape)) {
        GTEST_SKIP() << "BF16 reorder not supported on this processor";
    }

    std::vector<bfloat16> bf16_in(shape.k * shape.n, 0);
    std::vector<bfloat16> bf16_out(shape.k * shape.n, 0);

    for (md_t nr : { (md_t)144, (md_t)1048576 }) {
        SCOPED_TRACE("NR=" + std::to_string(nr));
        dlp_gemm_blocking_t bp = { 0, nr, 0, 0, 0 };

        dlp_metadata_t md = fresh_metadata();
        md.block_params   = &bp;
        aocl_reorder_bf16bf16f32of32('r', 'n', 'B', bf16_in.data(),
                                     bf16_out.data(), shape.k, shape.n, shape.n,
                                     &md);
        // If JIT is available, expect rejection. If reference fallback is
        // being used (e.g., AOCL_DLP_ENABLE_INSTRUCTIONS=avx2), the reference
        // will succeed (no JIT to reject), which is fine.
        if (md.error_hndl.error_code != DLP_CLSC_SUCCESS) {
            EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_JIT_KERNEL);
        }
    }
}

TEST(ReorderTunedValidation, RejectsOversizedInt8PackBJitNr)
{
    const Shape shape = { 4, 16 };

    // Establish support using round_trip. If not supported on this processor,
    // the test skips.
    if (!round_trip<int8_t>(aocl_get_reorder_buf_size_u8s8s32os32,
                            aocl_reorder_u8s8s32os32,
                            aocl_unreorder_u8s8s32os32_reference, 'r', shape)) {
        GTEST_SKIP() << "U8S8 reorder not supported on this processor";
    }
    if (!round_trip<int8_t>(aocl_get_reorder_buf_size_s8s8s32os32,
                            aocl_reorder_s8s8s32os32,
                            aocl_unreorder_s8s8s32os32_reference, 'r', shape)) {
        GTEST_SKIP() << "S8S8 reorder not supported on this processor";
    }

    std::vector<int8_t> int8_in(shape.k * shape.n, 0);
    // The S8 baseline probe also writes column sums after the packed bytes.
    std::vector<int8_t> int8_out(4 * shape.k * shape.n, 0);

    for (md_t nr : { (md_t)144, (md_t)1048576 }) {
        SCOPED_TRACE("NR=" + std::to_string(nr));
        dlp_gemm_blocking_t bp = { 0, nr, 0, 0, 0 };

        dlp_metadata_t md = fresh_metadata();
        md.block_params   = &bp;
        aocl_reorder_u8s8s32os32('r', 'n', 'B', int8_in.data(), int8_out.data(),
                                 shape.k, shape.n, shape.n, &md);
        EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_JIT_KERNEL);

        md              = fresh_metadata();
        md.block_params = &bp;
        aocl_reorder_s8s8s32os32('r', 'n', 'B', int8_in.data(), int8_out.data(),
                                 shape.k, shape.n, shape.n, &md);
        EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_JIT_KERNEL);
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
    }

    {
        // NC that is not a multiple of NR is illegal even when NR itself is
        // a legal tile width.
        const md_t          nr = (min_NR > 1) ? min_NR : (md_t)16;
        dlp_gemm_blocking_t bp = { 0, nr, 0, nr + (nr / 2), 256 };
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
                            aocl_unreorder_f32f32f32of32_reference,
                            dlp_get_packb_f32f32f32of32_min_NR(), 1);
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

int
s4_get_nibble(const uint8_t* bytes, size_t index)
{
    uint8_t byte = bytes[index >> 1];
    if ((index & 1u) != 0u) {
        return (int)((byte >> 4) & 0x0Fu);
    }
    return (int)(byte & 0x0Fu);
}

void
s4_set_nibble(uint8_t* bytes, size_t index, int value)
{
    uint8_t* byte   = bytes + (index >> 1);
    uint8_t  nibble = (uint8_t)(value & 0x0F);
    if ((index & 1u) != 0u) {
        *byte = (uint8_t)((*byte & 0x0Fu) | (nibble << 4));
    } else {
        *byte = (uint8_t)((*byte & 0xF0u) | nibble);
    }
}

// Nibble round trip shared by the s4 reorder APIs. group NULL leaves
// b_quant_op unset. Returns false when the ISA cannot reorder. The input
// allocation has 128 extra nibbles because the packers vector-read past a
// short row or an odd-length column.
template<typename SizeFn, typename ReorderFn, typename UnreorderFn>
bool
s4_round_trip(SizeFn                     size_fn,
              ReorderFn                  reorder_fn,
              UnreorderFn                unreorder_fn,
              char                       order,
              const Shape&               s,
              char                       mat_type,
              const md_t*                group,
              char                       trans_in  = 'n',
              char                       trans_out = 'n',
              md_t                       extra_ldb = 0,
              const dlp_gemm_blocking_t* bp        = nullptr,
              DLP_QUANT_OP_KIND          kind      = DLP_QUANT_OP_DEQUANTIZE)
{
    const md_t   ldb_in  = plain_ldb(order, trans_in, s.k, s.n) + extra_ldb;
    const md_t   ldb_out = plain_ldb(order, trans_out, s.k, s.n) + extra_ldb;
    const size_t in_n    = plain_elems(order, trans_in, s.k, s.n, ldb_in);
    const size_t out_n   = plain_elems(order, trans_out, s.k, s.n, ldb_out);

    dlp_quant_op_t quant_op;
    std::memset(&quant_op, 0, sizeof(quant_op));
    if (group != nullptr) {
        quant_op.quant_op_kind = kind;
        quant_op.group_size    = *group;
    }

    auto with_metadata = [&]() {
        dlp_metadata_t m = fresh_metadata();
        m.block_params   = const_cast<dlp_gemm_blocking_t*>(bp);
        if (group != nullptr) {
            m.b_quant_op = &quant_op;
        }
        return m;
    };

    dlp_metadata_t md     = with_metadata();
    msz_t          buf_sz = size_fn(order, trans_in, mat_type, s.k, s.n, &md);
    if (buf_sz == 0) {
        return false;
    }

    GuardedBuffer<uint8_t> input((in_n + 128u + 1u) / 2u);
    GuardedBuffer<uint8_t> output((out_n + 1u) / 2u);
    GuardedBuffer<uint8_t> packed((size_t)buf_sz);
    if (!input.valid() || !output.valid() || !packed.valid()) {
        ADD_FAILURE() << "guarded allocation failed";
        return true;
    }
    std::memset(input.data(), 0, input.size());
    std::memset(output.data(), 0, output.size());
    std::memset(packed.data(), 0, packed.size());

    for (md_t kr = 0; kr < s.k; kr++) {
        for (md_t jc = 0; jc < s.n; jc++) {
            s4_set_nibble(input.data(),
                          plain_idx(order, trans_in, kr, jc, ldb_in),
                          (int)(((kr * 3) + (jc * 5) + 1) & 0xF));
        }
    }

    md = with_metadata();
    reorder_fn(order, trans_in, mat_type, (const int8_t*)input.data(),
               (int8_t*)packed.data(), s.k, s.n, ldb_in, &md);
    if (md.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED) {
        return false;
    }
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_SUCCESS)
        << "s4 reorder failed k=" << s.k << " n=" << s.n;
    if (md.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        return true;
    }

    md = with_metadata();
    unreorder_fn(order, trans_out, mat_type, (const int8_t*)packed.data(),
                 (int8_t*)output.data(), s.k, s.n, ldb_out, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_SUCCESS)
        << "s4 unreorder failed k=" << s.k << " n=" << s.n;
    if (md.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        return true;
    }

    for (md_t kr = 0; kr < s.k; kr++) {
        for (md_t jc = 0; jc < s.n; jc++) {
            const int expect = (int)(((kr * 3) + (jc * 5) + 1) & 0xF);
            const int got    = s4_get_nibble(
                output.data(), plain_idx(order, trans_out, kr, jc, ldb_out));
            EXPECT_EQ(got, expect)
                << "s4 round trip k=" << s.k << " n=" << s.n
                << " mat=" << mat_type << " at (" << kr << "," << jc << ")";
            if (got != expect) {
                return true;
            }
        }
    }
    return true;
}

TEST(UnreorderRoundTrip, S8s4Reference)
{
    const std::vector<Shape> cases = {
        { 5, 7 }, { 17, 33 }, { 64, 1 }, { 64, 80 }, { 4099, 1091 },
    };
    int supported = 0;
    for (md_t threads : { (md_t)1, (md_t)-1 }) {
        PinnedThreadCount pin(threads);
        for (char order : { 'r', 'c' }) {
            for (const Shape& s : cases) {
                for (md_t group : { (md_t)0, (md_t)4, (md_t)12, s.k }) {
                    if (!quant_group_legal(s.k, group, 4)) {
                        continue;
                    }
                    const std::string desc =
                        std::string("s8s4 threads=")
                        + (threads < 0 ? std::string("default")
                                       : std::to_string(threads))
                        + " order=" + order + " group=" + std::to_string(group)
                        + " k=" + std::to_string(s.k)
                        + " n=" + std::to_string(s.n);
                    SCOPED_TRACE(desc);
                    set_current_case(desc);
                    if (s4_round_trip(aocl_get_reorder_buf_size_s8s4s32os32,
                                      aocl_reorder_s8s4s32os32,
                                      aocl_unreorder_s8s4s32os32, order, s, 'B',
                                      &group, 'n', 'n', 0, nullptr,
                                      DLP_QUANT_OP_QUANTIZE)) {
                        supported++;
                    }
                }
            }
        }
    }
    if (supported == 0) {
        GTEST_SKIP() << "s8s4s32os32 is not supported on this processor";
    }
}

TEST(UnreorderRoundTrip, S8s4ReferenceTranspose)
{
    int               supported = 0;
    PinnedThreadCount pin((md_t)1);
    for (char order : { 'r', 'c' }) {
        for (char trans_in : { 'n', 't' }) {
            for (char trans_out : { 'n', 't' }) {
                const Shape       s{ 17, 33 };
                const md_t        group = 4;
                const std::string desc  = std::string("s8s4 trans order=")
                                         + order + " in=" + trans_in
                                         + " out=" + trans_out;
                SCOPED_TRACE(desc);
                set_current_case(desc);
                if (s4_round_trip(
                        aocl_get_reorder_buf_size_s8s4s32os32,
                        aocl_reorder_s8s4s32os32, aocl_unreorder_s8s4s32os32,
                        order, s, 'B', &group, trans_in, trans_out,
                        /*extra_ldb=*/1, nullptr, DLP_QUANT_OP_QUANTIZE)) {
                    supported++;
                }
            }
        }
    }
    if (supported == 0) {
        GTEST_SKIP() << "s8s4s32os32 is not supported on this processor";
    }
}

// n == 1 is a tight expanded-s8 column when the group divides K, and the
// compact VNNI image otherwise. Odd K with group == K stays on the tight
// path; a group of 4 that does not divide K uses the padded packer.
TEST(UnreorderRoundTrip, S8s4N1)
{
    const std::vector<Shape> cases          = { { 1, 1 },  { 3, 1 },  { 4, 1 },
                                                { 5, 1 },  { 7, 1 },  { 15, 1 },
                                                { 16, 1 }, { 17, 1 }, { 63, 1 },
                                                { 64, 1 }, { 100, 1 } };
    const Shape              trans_shapes[] = { { 16, 1 }, { 17, 1 } };
    int                      supported      = 0;

    for (md_t threads : { (md_t)1, (md_t)-1 }) {
        PinnedThreadCount pin(threads);
        for (char order : { 'r', 'c' }) {
            for (const Shape& s : cases) {
                for (md_t group : { (md_t)0, (md_t)4, s.k }) {
                    if (!quant_group_legal(s.k, group, 4)) {
                        continue;
                    }
                    const std::string desc =
                        std::string("s8s4 n1 threads=")
                        + (threads < 0 ? std::string("default")
                                       : std::to_string(threads))
                        + " order=" + order + " group=" + std::to_string(group)
                        + " k=" + std::to_string(s.k);
                    SCOPED_TRACE(desc);
                    set_current_case(desc);
                    if (s4_round_trip(aocl_get_reorder_buf_size_s8s4s32os32,
                                      aocl_reorder_s8s4s32os32,
                                      aocl_unreorder_s8s4s32os32, order, s, 'B',
                                      &group, 'n', 'n', 0, nullptr,
                                      DLP_QUANT_OP_QUANTIZE)) {
                        supported++;
                    }
                }
            }
        }
    }

    PinnedThreadCount pin((md_t)1);
    for (char order : { 'r', 'c' }) {
        for (char trans_in : { 'n', 't' }) {
            for (char trans_out : { 'n', 't' }) {
                for (const Shape& s : trans_shapes) {
                    const md_t        group = s.k;
                    const std::string desc = std::string("s8s4 n1 trans order=")
                                             + order + " in=" + trans_in
                                             + " out=" + trans_out
                                             + " k=" + std::to_string(s.k);
                    SCOPED_TRACE(desc);
                    set_current_case(desc);
                    if (s4_round_trip(aocl_get_reorder_buf_size_s8s4s32os32,
                                      aocl_reorder_s8s4s32os32,
                                      aocl_unreorder_s8s4s32os32, order, s, 'B',
                                      &group, trans_in, trans_out,
                                      /*extra_ldb=*/1, nullptr,
                                      DLP_QUANT_OP_QUANTIZE)) {
                        supported++;
                    }
                }
            }
        }
    }

    if (supported == 0) {
        GTEST_SKIP() << "s8s4s32os32 is not supported on this processor";
    }
}

TEST(UnreorderS8s4Validation, RejectsBadMetadataAndMatrixType)
{
    const Shape         s{ 16, 16 };
    std::vector<int8_t> packed(256, 0);
    std::vector<int8_t> output(256, 0x5A);

    auto expect_reject = [&](const char* what, dlp_metadata_t* md,
                             char mat_type, dlp_clsc_err_t want) {
        const std::vector<int8_t> before = output;
        aocl_unreorder_s8s4s32os32('r', 'n', mat_type, packed.data(),
                                   output.data(), s.k, s.n, s.n, md);
        if (md != nullptr) {
            EXPECT_EQ(md->error_hndl.error_code, want) << what;
        }
        EXPECT_EQ(output, before) << what << " wrote output";
    };

    auto make_quant = [](dlp_quant_op_t* op, md_t group_size) {
        std::memset(op, 0, sizeof(*op));
        op->quant_op_kind = DLP_QUANT_OP_QUANTIZE;
        op->group_size    = group_size;
    };

    for (md_t group : { (md_t)5, (md_t)7, (md_t)32, (md_t)-4 }) {
        dlp_quant_op_t op;
        make_quant(&op, group);
        dlp_metadata_t md = fresh_metadata();
        md.b_quant_op     = &op;
        expect_reject("illegal group", &md, 'B',
                      DLP_CLSC_INVALID_GROUP_DIMENSION);
    }

    {
        dlp_quant_op_t op;
        make_quant(&op, 4);
        op.quant_op_kind  = DLP_QUANT_OP_DEQUANTIZE;
        dlp_metadata_t md = fresh_metadata();
        md.b_quant_op     = &op;
        expect_reject("dequantize op", &md, 'B', DLP_CLSC_NOT_SUPPORTED);
    }

    {
        dlp_quant_op_t op;
        make_quant(&op, 4);
        dlp_metadata_t md = fresh_metadata();
        md.b_quant_op     = &op;
        expect_reject("A matrix", &md, 'A', DLP_CLSC_NOT_SUPPORTED);
        expect_reject("W matrix", &md, 'W', DLP_CLSC_NOT_SUPPORTED);
    }

    expect_reject("null metadata", nullptr, 'B', DLP_CLSC_NULL_POINTER);

    {
        dlp_quant_op_t a_op;
        dlp_quant_op_t b_op;
        make_quant(&a_op, 4);
        make_quant(&b_op, 8);
        dlp_metadata_t md = fresh_metadata();
        md.a_quant_op     = &a_op;
        md.b_quant_op     = &b_op;
        expect_reject("mismatched groups", &md, 'B',
                      DLP_CLSC_INVALID_GROUP_DIMENSION);

        dlp_metadata_t size_md = md;
        msz_t sz = aocl_get_reorder_buf_size_s8s4s32os32('r', 'n', 'B', s.k,
                                                         s.n, &size_md);
        // VNNI is checked before the group, so a machine without it reports
        // NOT_SUPPORTED and a zero size instead.
        if (size_md.error_hndl.error_code != DLP_CLSC_NOT_SUPPORTED) {
            EXPECT_EQ(size_md.error_hndl.error_code,
                      DLP_CLSC_INVALID_GROUP_DIMENSION);
        }
        EXPECT_EQ(sz, (msz_t)0);

        std::vector<int8_t> packed_before = packed;
        dlp_metadata_t      reorder_md    = md;
        aocl_reorder_s8s4s32os32('r', 'n', 'B', (const int8_t*)output.data(),
                                 packed.data(), s.k, s.n, s.n, &reorder_md);
        if (reorder_md.error_hndl.error_code != DLP_CLSC_NOT_SUPPORTED) {
            EXPECT_EQ(reorder_md.error_hndl.error_code,
                      DLP_CLSC_INVALID_GROUP_DIMENSION);
        }
        EXPECT_EQ(packed, packed_before);
    }
}

// 12 does not divide the table KC (2048 or 4096), so KC shrinks. 6144 is
// larger than either table KC, so KC grows. 6144 is legal only when K is at
// least that large.
TEST(UnreorderRoundTrip, S8s4KcAlign)
{
    const Shape cases[]   = { { 4101, 32 }, { 8195, 32 } };
    const md_t  groups[]  = { 12, 6144 };
    int         supported = 0;
    for (md_t threads : { (md_t)1, (md_t)-1 }) {
        PinnedThreadCount pin(threads);
        for (char order : { 'r', 'c' }) {
            for (const Shape& s : cases) {
                for (md_t group : groups) {
                    if (!quant_group_legal(s.k, group, 4)) {
                        continue;
                    }
                    const std::string desc =
                        std::string("s8s4 kc-align order=") + order + " group="
                        + std::to_string(group) + " k=" + std::to_string(s.k);
                    SCOPED_TRACE(desc);
                    set_current_case(desc);
                    if (s4_round_trip(aocl_get_reorder_buf_size_s8s4s32os32,
                                      aocl_reorder_s8s4s32os32,
                                      aocl_unreorder_s8s4s32os32, order, s, 'B',
                                      &group, 'n', 'n', 0, nullptr,
                                      DLP_QUANT_OP_QUANTIZE)) {
                        supported++;
                    }
                }
            }
        }
    }
    if (supported == 0) {
        GTEST_SKIP() << "s8s4s32os32 is not supported on this processor";
    }
}

// No quant op is one group over K. An A op alone supplies the group, and that
// packed image matches a B op with the same group.
TEST(UnreorderRoundTrip, S8s4MissingQuantOp)
{
    const Shape       s{ 32, 16 };
    PinnedThreadCount pin((md_t)1);
    int               supported = 0;
    for (char order : { 'r', 'c' }) {
        const std::string desc = std::string("s8s4 no-quant-op order=") + order;
        SCOPED_TRACE(desc);
        set_current_case(desc);
        if (s4_round_trip(aocl_get_reorder_buf_size_s8s4s32os32,
                          aocl_reorder_s8s4s32os32, aocl_unreorder_s8s4s32os32,
                          order, s, 'B', nullptr, 'n', 'n', 0, nullptr,
                          DLP_QUANT_OP_QUANTIZE)) {
            supported++;
        }
    }
    if (supported == 0) {
        GTEST_SKIP() << "s8s4s32os32 is not supported on this processor";
    }

    const md_t     group   = 4;
    const md_t     ldb     = s.n;
    const size_t   nibbles = (size_t)s.k * (size_t)ldb;
    dlp_quant_op_t b_op;
    std::memset(&b_op, 0, sizeof(b_op));
    b_op.quant_op_kind     = DLP_QUANT_OP_QUANTIZE;
    b_op.group_size        = group;
    dlp_metadata_t size_md = fresh_metadata();
    size_md.b_quant_op     = &b_op;
    msz_t nbytes = aocl_get_reorder_buf_size_s8s4s32os32('r', 'n', 'B', s.k,
                                                         s.n, &size_md);
    ASSERT_NE(nbytes, (msz_t)0);

    GuardedBuffer<uint8_t> input((nibbles + 128u + 1u) / 2u);
    GuardedBuffer<uint8_t> packed_a((size_t)nbytes);
    GuardedBuffer<uint8_t> packed_b((size_t)nbytes);
    GuardedBuffer<uint8_t> output((nibbles + 1u) / 2u);
    ASSERT_TRUE(input.valid() && packed_a.valid() && packed_b.valid()
                && output.valid());
    std::memset(input.data(), 0, input.size());
    for (size_t i = 0; i < nibbles; i++) {
        s4_set_nibble(input.data(), i, (int)((i * 3 + 1) & 0xF));
    }

    dlp_quant_op_t a_op = b_op;
    dlp_metadata_t md_a = fresh_metadata();
    md_a.a_quant_op     = &a_op;
    aocl_reorder_s8s4s32os32('r', 'n', 'B', (const int8_t*)input.data(),
                             (int8_t*)packed_a.data(), s.k, s.n, ldb, &md_a);
    ASSERT_EQ(md_a.error_hndl.error_code, DLP_CLSC_SUCCESS);

    dlp_metadata_t md_b = fresh_metadata();
    md_b.b_quant_op     = &b_op;
    aocl_reorder_s8s4s32os32('r', 'n', 'B', (const int8_t*)input.data(),
                             (int8_t*)packed_b.data(), s.k, s.n, ldb, &md_b);
    ASSERT_EQ(md_b.error_hndl.error_code, DLP_CLSC_SUCCESS);
    EXPECT_EQ(std::memcmp(packed_a.data(), packed_b.data(), (size_t)nbytes), 0);

    md_a            = fresh_metadata();
    md_a.a_quant_op = &a_op;
    aocl_unreorder_s8s4s32os32('r', 'n', 'B', (const int8_t*)packed_a.data(),
                               (int8_t*)output.data(), s.k, s.n, ldb, &md_a);
    ASSERT_EQ(md_a.error_hndl.error_code, DLP_CLSC_SUCCESS);
    for (size_t i = 0; i < nibbles; i++) {
        EXPECT_EQ(s4_get_nibble(output.data(), i), (int)((i * 3 + 1) & 0xF))
            << "nibble " << i;
    }
}

TEST(UnreorderRoundTrip, Bf16s4Reference)
{
    const md_t groups[]  = { 0, 4, 12 };
    int        supported = 0;
    for (md_t threads : { (md_t)1, (md_t)-1 }) {
        PinnedThreadCount pin(threads);
        for (char order : { 'r', 'c' }) {
            for (char mat : { 'B' }) {
                for (const Shape& s : shapes()) {
                    for (md_t group : groups) {
                        if (!quant_group_legal(s.k, group, 2)) {
                            continue;
                        }
                        const std::string desc =
                            std::string("bf16s4 threads=")
                            + (threads < 0 ? std::string("default")
                                           : std::to_string(threads))
                            + " order=" + order + " mat=" + mat
                            + " group=" + std::to_string(group) + " k="
                            + std::to_string(s.k) + " n=" + std::to_string(s.n);
                        SCOPED_TRACE(desc);
                        set_current_case(desc);
                        if (s4_round_trip(
                                aocl_get_reorder_buf_size_bf16s4f32of32,
                                aocl_reorder_bf16s4f32of32,
                                aocl_unreorder_bf16s4f32of32, order, s, mat,
                                &group)) {
                            supported++;
                        }
                    }
                    // group == K is full K, same KC as a missing quant op.
                    const std::string desc = std::string("bf16s4 full-K order=")
                                             + order + " mat=" + mat
                                             + " k=" + std::to_string(s.k);
                    SCOPED_TRACE(desc);
                    set_current_case(desc);
                    md_t shape_k = s.k;
                    if (s4_round_trip(aocl_get_reorder_buf_size_bf16s4f32of32,
                                      aocl_reorder_bf16s4f32of32,
                                      aocl_unreorder_bf16s4f32of32, order, s,
                                      mat, &shape_k)) {
                        supported++;
                    }
                    // No b_quant_op at all, rather than a present op whose
                    // group_size is 0 or K.
                    const std::string missing =
                        std::string("bf16s4 no-quant-op order=") + order
                        + " k=" + std::to_string(s.k);
                    SCOPED_TRACE(missing);
                    set_current_case(missing);
                    if (s4_round_trip(aocl_get_reorder_buf_size_bf16s4f32of32,
                                      aocl_reorder_bf16s4f32of32,
                                      aocl_unreorder_bf16s4f32of32, order, s,
                                      mat, nullptr)) {
                        supported++;
                    }
                }
            }
        }
    }
    if (supported == 0) {
        GTEST_SKIP() << "bf16s4f32of32 is not supported on this processor";
    }
}

TEST(UnreorderRoundTrip, Bf16s4MissingQuantOpMatchesFullK)
{
    // 4104 does not divide the table KC, so a real group changes the packed
    // panel boundaries. A missing quant op must keep the same image as a
    // full-K group.
    const Shape       s = { 4104, 128 };
    PinnedThreadCount pin((md_t)1);
    dlp_metadata_t    probe = fresh_metadata();
    msz_t nbytes = aocl_get_reorder_buf_size_bf16s4f32of32('r', 'n', 'B', s.k,
                                                           s.n, &probe);
    if (nbytes == 0) {
        GTEST_SKIP() << "bf16s4f32of32 is not supported on this processor";
    }

    for (char order : { 'r', 'c' }) {
        const md_t             ldb     = plain_ldb(order, 'n', s.k, s.n);
        const size_t           nibbles = plain_elems(order, 'n', s.k, s.n, ldb);
        GuardedBuffer<uint8_t> input((nibbles + 128u + 1u) / 2u);
        GuardedBuffer<uint8_t> packed_missing((size_t)nbytes);
        GuardedBuffer<uint8_t> packed_zero((size_t)nbytes);
        GuardedBuffer<uint8_t> packed_k((size_t)nbytes);
        GuardedBuffer<uint8_t> packed_null_md((size_t)nbytes);
        ASSERT_TRUE(input.valid() && packed_missing.valid()
                    && packed_zero.valid() && packed_k.valid()
                    && packed_null_md.valid());
        std::memset(input.data(), 0, input.size());
        for (md_t kr = 0; kr < s.k; kr++) {
            for (md_t jc = 0; jc < s.n; jc++) {
                s4_set_nibble(input.data(), plain_idx(order, 'n', kr, jc, ldb),
                              (int)(((kr * 3) + (jc * 5) + 1) & 0xF));
            }
        }

        dlp_metadata_t missing = fresh_metadata();
        aocl_reorder_bf16s4f32of32(order, 'n', 'B', (const int8_t*)input.data(),
                                   (int8_t*)packed_missing.data(), s.k, s.n,
                                   ldb, &missing);
        ASSERT_EQ(missing.error_hndl.error_code, DLP_CLSC_SUCCESS);

        dlp_quant_op_t op;
        std::memset(&op, 0, sizeof(op));
        op.quant_op_kind    = DLP_QUANT_OP_DEQUANTIZE;
        op.group_size       = 0;
        dlp_metadata_t zero = fresh_metadata();
        zero.b_quant_op     = &op;
        aocl_reorder_bf16s4f32of32(order, 'n', 'B', (const int8_t*)input.data(),
                                   (int8_t*)packed_zero.data(), s.k, s.n, ldb,
                                   &zero);
        ASSERT_EQ(zero.error_hndl.error_code, DLP_CLSC_SUCCESS);

        op.group_size         = s.k;
        dlp_metadata_t full_k = fresh_metadata();
        full_k.b_quant_op     = &op;
        aocl_reorder_bf16s4f32of32(order, 'n', 'B', (const int8_t*)input.data(),
                                   (int8_t*)packed_k.data(), s.k, s.n, ldb,
                                   &full_k);
        ASSERT_EQ(full_k.error_hndl.error_code, DLP_CLSC_SUCCESS);

        aocl_reorder_bf16s4f32of32(order, 'n', 'B', (const int8_t*)input.data(),
                                   (int8_t*)packed_null_md.data(), s.k, s.n,
                                   ldb, nullptr);

        EXPECT_EQ(std::memcmp(packed_missing.data(), packed_zero.data(),
                              (size_t)nbytes),
                  0)
            << "order=" << order;
        EXPECT_EQ(
            std::memcmp(packed_missing.data(), packed_k.data(), (size_t)nbytes),
            0)
            << "order=" << order;
        EXPECT_EQ(std::memcmp(packed_missing.data(), packed_null_md.data(),
                              (size_t)nbytes),
                  0)
            << "order=" << order;
    }
}

TEST(UnreorderRoundTrip, Bf16s4KcAlignAndTranspose)
{
    // 12 does not divide the table KC (4096 or 2048), so KC shrinks.
    // 6144 is larger than either table KC, so KC grows to the group.
    const Shape grow       = { 8195, 32 };
    const md_t  grow_group = 6144;
    // 4104 x 128 keeps group 12 across a KC boundary.
    const Shape panels[] = { { 4099, 1091 }, { 8195, 129 }, { 4104, 128 } };
    const Shape trans_shapes[] = { { 5, 7 },
                                   { 17, 33 },
                                   { 63, 129 },
                                   { 16, 24 } };
    int         supported      = 0;

    for (md_t threads : { (md_t)1, (md_t)-1 }) {
        PinnedThreadCount pin(threads);
        for (char order : { 'r', 'c' }) {
            for (char mat : { 'B' }) {
                for (const Shape& s : panels) {
                    md_t              group = 12;
                    const std::string desc = std::string("bf16s4 panels order=")
                                             + order + " mat=" + mat
                                             + " k=" + std::to_string(s.k);
                    SCOPED_TRACE(desc);
                    set_current_case(desc);
                    if (s4_round_trip(aocl_get_reorder_buf_size_bf16s4f32of32,
                                      aocl_reorder_bf16s4f32of32,
                                      aocl_unreorder_bf16s4f32of32, order, s,
                                      mat, &group)) {
                        supported++;
                    }
                }
                SCOPED_TRACE("bf16s4 grow KC");
                set_current_case("bf16s4 grow KC");
                if (s4_round_trip(aocl_get_reorder_buf_size_bf16s4f32of32,
                                  aocl_reorder_bf16s4f32of32,
                                  aocl_unreorder_bf16s4f32of32, order, grow,
                                  mat, &grow_group)) {
                    supported++;
                }
            }
        }
    }

    PinnedThreadCount pin((md_t)1);
    for (char order : { 'r', 'c' }) {
        for (char mat : { 'B' }) {
            for (char trans_in : { 'n', 't' }) {
                for (char trans_out : { 'n', 't' }) {
                    for (const Shape& s : trans_shapes) {
                        md_t group = 4;
                        if (!quant_group_legal(s.k, group, 2)) {
                            continue;
                        }
                        const std::string desc =
                            std::string("bf16s4 trans order=") + order + " mat="
                            + mat + " in=" + trans_in + " out=" + trans_out;
                        SCOPED_TRACE(desc);
                        set_current_case(desc);
                        if (s4_round_trip(
                                aocl_get_reorder_buf_size_bf16s4f32of32,
                                aocl_reorder_bf16s4f32of32,
                                aocl_unreorder_bf16s4f32of32, order, s, mat,
                                &group, trans_in, trans_out,
                                /*extra_ldb=*/1)) {
                            supported++;
                        }
                    }
                }
            }
        }
    }

    if (supported == 0) {
        GTEST_SKIP() << "bf16s4f32of32 is not supported on this processor";
    }
}

// n == 1 packs into a min_NR=16 slot (same as a trailing fringe under 16).
// Odd and even K, plus a group that does not divide table KC, keep the
// unreorder on the same panel boundaries as reorder.
TEST(UnreorderRoundTrip, Bf16s4N1)
{
    const std::vector<Shape> cases = { { 1, 1 },   { 2, 1 },   { 3, 1 },
                                       { 4, 1 },   { 5, 1 },   { 7, 1 },
                                       { 15, 1 },  { 16, 1 },  { 17, 1 },
                                       { 63, 1 },  { 64, 1 },  { 100, 1 },
                                       { 128, 1 }, { 4104, 1 } };
    const Shape trans_shapes[]     = { { 16, 1 }, { 17, 1 }, { 64, 1 } };
    int         supported          = 0;

    for (md_t threads : { (md_t)1, (md_t)-1 }) {
        PinnedThreadCount pin(threads);
        for (char order : { 'r', 'c' }) {
            for (const Shape& s : cases) {
                for (md_t group :
                     { (md_t)0, (md_t)2, (md_t)4, (md_t)12, s.k }) {
                    if (!quant_group_legal(s.k, group, 2)) {
                        continue;
                    }
                    const std::string desc =
                        std::string("bf16s4 n1 threads=")
                        + (threads < 0 ? std::string("default")
                                       : std::to_string(threads))
                        + " order=" + order + " group=" + std::to_string(group)
                        + " k=" + std::to_string(s.k);
                    SCOPED_TRACE(desc);
                    set_current_case(desc);
                    if (s4_round_trip(aocl_get_reorder_buf_size_bf16s4f32of32,
                                      aocl_reorder_bf16s4f32of32,
                                      aocl_unreorder_bf16s4f32of32, order, s,
                                      'B', &group)) {
                        supported++;
                    }
                }
            }
        }
    }

    PinnedThreadCount pin((md_t)1);
    for (char order : { 'r', 'c' }) {
        for (char trans_in : { 'n', 't' }) {
            for (char trans_out : { 'n', 't' }) {
                for (const Shape& s : trans_shapes) {
                    const md_t        group = s.k;
                    const std::string desc =
                        std::string("bf16s4 n1 trans order=") + order
                        + " in=" + trans_in + " out=" + trans_out
                        + " k=" + std::to_string(s.k);
                    SCOPED_TRACE(desc);
                    set_current_case(desc);
                    if (s4_round_trip(aocl_get_reorder_buf_size_bf16s4f32of32,
                                      aocl_reorder_bf16s4f32of32,
                                      aocl_unreorder_bf16s4f32of32, order, s,
                                      'B', &group, trans_in, trans_out,
                                      /*extra_ldb=*/1)) {
                        supported++;
                    }
                }
            }
        }
    }

    if (supported == 0) {
        GTEST_SKIP() << "bf16s4f32of32 is not supported on this processor";
    }
}

TEST(UnreorderRoundTrip, Bf16s4IgnoresBlockParams)
{
    const Shape         s  = { 100, 80 };
    dlp_gemm_blocking_t bp = {};
    bp.KC                  = 128;
    PinnedThreadCount pin((md_t)-1);

    dlp_metadata_t plain = fresh_metadata();
    msz_t nbytes = aocl_get_reorder_buf_size_bf16s4f32of32('r', 'n', 'B', s.k,
                                                           s.n, &plain);
    if (nbytes == 0) {
        GTEST_SKIP() << "bf16s4f32of32 is not supported on this processor";
    }

    const md_t             ldb     = s.n;
    const size_t           nibbles = (size_t)s.k * (size_t)ldb;
    GuardedBuffer<uint8_t> input((nibbles + 1u) / 2u);
    GuardedBuffer<uint8_t> packed_plain((size_t)nbytes);
    GuardedBuffer<uint8_t> packed_kc((size_t)nbytes);
    std::memset(input.data(), 0, input.size());
    for (size_t i = 0; i < nibbles; i++) {
        s4_set_nibble(input.data(), i, (int)((i * 3 + 1) & 0xF));
    }

    plain = fresh_metadata();
    aocl_reorder_bf16s4f32of32('r', 'n', 'B', (const int8_t*)input.data(),
                               (int8_t*)packed_plain.data(), s.k, s.n, ldb,
                               &plain);
    ASSERT_EQ(plain.error_hndl.error_code, DLP_CLSC_SUCCESS);

    dlp_metadata_t with_kc = fresh_metadata();
    with_kc.block_params   = &bp;
    aocl_reorder_bf16s4f32of32('r', 'n', 'B', (const int8_t*)input.data(),
                               (int8_t*)packed_kc.data(), s.k, s.n, ldb,
                               &with_kc);
    ASSERT_EQ(with_kc.error_hndl.error_code, DLP_CLSC_SUCCESS);
    EXPECT_EQ(
        std::memcmp(packed_plain.data(), packed_kc.data(), (size_t)nbytes), 0);

    md_t group = 12;
    EXPECT_TRUE(s4_round_trip(
        aocl_get_reorder_buf_size_bf16s4f32of32, aocl_reorder_bf16s4f32of32,
        aocl_unreorder_bf16s4f32of32, 'r', s, 'B', &group, 'n', 'n', 0, &bp));
    EXPECT_TRUE(s4_round_trip(
        aocl_get_reorder_buf_size_bf16s4f32of32, aocl_reorder_bf16s4f32of32,
        aocl_unreorder_bf16s4f32of32, 'c', s, 'B', &group, 'n', 'n', 0, &bp));
}

TEST(UnreorderBf16s4Validation, RejectsIllegalGroupAndMatrixType)
{
    const Shape    s     = { 32, 16 };
    const md_t     ldb   = s.n;
    dlp_metadata_t probe = fresh_metadata();
    msz_t nbytes = aocl_get_reorder_buf_size_bf16s4f32of32('r', 'n', 'B', s.k,
                                                           s.n, &probe);
    if (nbytes == 0) {
        GTEST_SKIP() << "bf16s4f32of32 is not supported on this processor";
    }

    std::vector<uint8_t> packed((size_t)nbytes, 0);
    std::vector<uint8_t> output(((size_t)s.k * (size_t)ldb + 1u) / 2u, 0);
    const md_t           illegal[] = { 5, 7, 64, -2 };
    for (md_t group : illegal) {
        dlp_quant_op_t op;
        std::memset(&op, 0, sizeof(op));
        op.quant_op_kind  = DLP_QUANT_OP_DEQUANTIZE;
        op.group_size     = group;
        dlp_metadata_t md = fresh_metadata();
        md.b_quant_op     = &op;
        aocl_unreorder_bf16s4f32of32(
            'r', 'n', 'B', (const int8_t*)packed.data(), (int8_t*)output.data(),
            s.k, s.n, ldb, &md);
        EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_GROUP_DIMENSION)
            << "group=" << group;

        md            = fresh_metadata();
        md.b_quant_op = &op;
        aocl_reorder_bf16s4f32of32('r', 'n', 'B', (const int8_t*)output.data(),
                                   (int8_t*)packed.data(), s.k, s.n, ldb, &md);
        EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_GROUP_DIMENSION)
            << "reorder group=" << group;
    }

    dlp_quant_op_t quantize;
    std::memset(&quantize, 0, sizeof(quantize));
    quantize.quant_op_kind = DLP_QUANT_OP_QUANTIZE;
    quantize.group_size    = 4;
    dlp_metadata_t md      = fresh_metadata();
    md.b_quant_op          = &quantize;
    aocl_unreorder_bf16s4f32of32('r', 'n', 'B', (const int8_t*)packed.data(),
                                 (int8_t*)output.data(), s.k, s.n, ldb, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);

    md = fresh_metadata();
    aocl_unreorder_bf16s4f32of32('r', 'n', 'A', (const int8_t*)packed.data(),
                                 (int8_t*)output.data(), s.k, s.n, ldb, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);

    md = fresh_metadata();
    aocl_unreorder_bf16s4f32of32('r', 'n', 'W', (const int8_t*)packed.data(),
                                 (int8_t*)output.data(), s.k, s.n, ldb, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);

    // size / reorder also accept only B (not A or W).
    md = fresh_metadata();
    EXPECT_EQ(
        aocl_get_reorder_buf_size_bf16s4f32of32('r', 'n', 'W', s.k, s.n, &md),
        (msz_t)0);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);

    md = fresh_metadata();
    aocl_reorder_bf16s4f32of32('r', 'n', 'W', (const int8_t*)output.data(),
                               (int8_t*)packed.data(), s.k, s.n, ldb, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NOT_SUPPORTED);

    md_t expand_group = 4;
    EXPECT_TRUE(s4_round_trip(
        aocl_get_reorder_buf_size_bf16s4f32of32, aocl_reorder_bf16s4f32of32,
        aocl_unreorder_bf16s4f32of32, 'r', s, 'B', &expand_group, 'n', 'n', 0,
        nullptr, DLP_QUANT_OP_EXPAND));
}

// Col-major pack with n < 16 and an odd ldb exercises the nr<16 odd-stride
// mask path (shared-byte nibble load).
TEST(UnreorderRoundTrip, Bf16s4ColMajorNrLt16OddLdb)
{
    const Shape s     = { 17, 7 }; // n in (0, 16)
    md_t        group = 4;
    if (!s4_round_trip(aocl_get_reorder_buf_size_bf16s4f32of32,
                       aocl_reorder_bf16s4f32of32, aocl_unreorder_bf16s4f32of32,
                       'c', s, 'B', &group, 'n', 'n',
                       /*extra_ldb=*/1)) {
        GTEST_SKIP() << "bf16s4f32of32 is not supported on this processor";
    }
}

// NR must be a multiple of the f32 pack-B SIMD width (8 on AVX2, 16 on
// AVX-512). NR=17 and NR=65 used to pass the size API because the check was
// NR % NR, then the tuned reorder rejected the same metadata.
TEST(ReorderBufSize, F32RejectsNrNotMultipleOfMinTile)
{
    const md_t k      = 37;
    const md_t n      = 45;
    const md_t min_NR = dlp_get_packb_f32f32f32of32_min_NR();

    auto size_for = [&](md_t nr, dlp_clsc_err_t* err) {
        dlp_gemm_blocking_t bp = { 0, nr, 0, 0, 0 };
        dlp_metadata_t      md = fresh_metadata();
        md.block_params        = &bp;
        msz_t sz =
            aocl_get_reorder_buf_size_f32f32f32of32('r', 'n', 'B', k, n, &md);
        *err = md.error_hndl.error_code;
        return sz;
    };

    dlp_clsc_err_t err = DLP_CLSC_SUCCESS;
    EXPECT_EQ(size_for(17, &err), (msz_t)0);
    EXPECT_EQ(err, DLP_CLSC_INVALID_BLOCK_PARAMS);
    EXPECT_EQ(size_for(65, &err), (msz_t)0);
    EXPECT_EQ(err, DLP_CLSC_INVALID_BLOCK_PARAMS);

    EXPECT_GT(size_for(16, &err), (msz_t)0);
    EXPECT_EQ(err, DLP_CLSC_SUCCESS);
    EXPECT_GT(size_for(64, &err), (msz_t)0);
    EXPECT_EQ(err, DLP_CLSC_SUCCESS);

    // AVX2 pack-B accepts NR=8 and NR=24; AVX-512 ZMM does not.
    if (min_NR == 8) {
        EXPECT_GT(size_for(8, &err), (msz_t)0);
        EXPECT_EQ(err, DLP_CLSC_SUCCESS);
        EXPECT_GT(size_for(24, &err), (msz_t)0);
        EXPECT_EQ(err, DLP_CLSC_SUCCESS);
    } else {
        EXPECT_EQ(size_for(8, &err), (msz_t)0);
        EXPECT_EQ(err, DLP_CLSC_INVALID_BLOCK_PARAMS);
        EXPECT_EQ(size_for(24, &err), (msz_t)0);
        EXPECT_EQ(err, DLP_CLSC_INVALID_BLOCK_PARAMS);
    }

    std::vector<float>  scratch(k * n, 0.0f);
    dlp_gemm_blocking_t bp = { 0, 17, 0, 0, 0 };
    dlp_metadata_t      md = fresh_metadata();
    md.block_params        = &bp;
    aocl_reorder_f32f32f32of32('r', 'n', 'B', scratch.data(), scratch.data(), k,
                               n, n, &md);
    if (md.error_hndl.error_code != DLP_CLSC_NOT_SUPPORTED) {
        EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_JIT_KERNEL);
    }

    md              = fresh_metadata();
    md.block_params = &bp;
    aocl_reorder_f32f32f32of32_reference('r', 'n', 'B', scratch.data(),
                                         scratch.data(), k, n, n, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_BLOCK_PARAMS);
}

// Argument checks run before the symmetric-quant group-size lookup, so k <= 0
// and a NULL buffer are not reported as INVALID_GROUP_DIMENSION.
TEST(ReorderValidation, S8S4ChecksArgsBeforeGroupSize)
{
    dlp_metadata_t md      = fresh_metadata();
    int8_t         out[32] = {};
    int8_t         in[32]  = {};

    aocl_reorder_s8s4s32os32('r', 'n', 'B', in, out, /*k=*/0, 21, 21, &md);
    if (md.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED) {
        GTEST_SKIP() << "s8s4 reorder is not supported on this processor";
    }
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_INVALID_MATRIX_DIMENSION);

    md = fresh_metadata();
    aocl_reorder_s8s4s32os32('r', 'n', 'B', nullptr, out, /*k=*/0, 21, 21, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NULL_POINTER);

    md = fresh_metadata();
    aocl_reorder_s8s4s32os32('r', 'n', 'B', nullptr, out, /*k=*/8, 16, 16, &md);
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_NULL_POINTER);
}

} // namespace
