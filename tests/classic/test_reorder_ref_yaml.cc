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
 * YAML-driven round trips for aocl_reorder_*_reference /
 * aocl_unreorder_*_reference. Each (dtype, k, n, order) is a named gtest
 * so coverage and --gtest_list_tests list every size in
 * configs/reorder_ref_coverage.yaml.
 */

#include "aocl_dlp.h"
#include "test_config.hh"

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace dlp_test {

struct YamlCase
{
    std::string                      set;
    std::string                      dtype;
    char                             order;
    char                             trans;
    md_t                             k;
    md_t                             n;
    md_t                             min_nr;
    md_t                             k_factor;
    std::vector<dlp_gemm_blocking_t> blocks;
};

void
PrintTo(const YamlCase& c, std::ostream* os)
{
    *os << c.set << " " << c.dtype << " k=" << c.k << " n=" << c.n
        << " order=" << c.order << " trans=" << c.trans;
}

} // namespace dlp_test

namespace {

using dlp_test::YamlCase;

dlp_metadata_t
fresh_metadata()
{
    dlp_metadata_t md;
    std::memset(&md, 0, sizeof(md));
    return md;
}

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

std::vector<md_t>
as_md_list(const YAML::Node& node)
{
    std::vector<md_t> out;
    for (const auto& v : node) {
        out.push_back((md_t)v.as<int64_t>());
    }
    return out;
}

bool
blocks_legal(const dlp_gemm_blocking_t& bp, md_t min_nr, md_t kf)
{
    return (min_nr > 0) && (kf > 0) && (bp.NR > 0) && (bp.NC > 0) && (bp.KC > 0)
           && ((bp.NR % min_nr) == 0) && ((bp.NC % bp.NR) == 0)
           && ((bp.KC % kf) == 0);
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

// Tight leading dimension for a plain B stored at (order, trans). Matches
// AOCL_DLP_UNREORDER_CHECK: row+n => ldb>=n, row+t => ldb>=k, and the
// column-major swap of those two.
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

// Returns false when the ISA reports NOT_SUPPORTED so the caller can skip;
// true once a reorder was attempted (success or EXPECT failure).
template<typename T, typename SizeFn, typename ReorderFn, typename UnreorderFn>
bool
round_trip_one(SizeFn                     size_fn,
               ReorderFn                  reorder_fn,
               UnreorderFn                unreorder_fn,
               char                       order,
               char                       trans,
               md_t                       k,
               md_t                       n,
               const dlp_gemm_blocking_t& bp)
{
    const md_t   ldb   = plain_ldb(order, trans, k, n);
    const size_t elems = plain_elems(order, trans, k, n, ldb);

    dlp_metadata_t md = fresh_metadata();
    md.block_params   = const_cast<dlp_gemm_blocking_t*>(&bp);
    msz_t buf_sz      = size_fn(order, trans, 'B', k, n, &md);
    EXPECT_GT(buf_sz, (msz_t)0)
        << "size API returned 0 for legal blocking k=" << k << " n=" << n
        << " trans=" << trans;
    if (buf_sz == 0) {
        return true;
    }

    std::vector<T>       input(elems);
    std::vector<T>       output(elems, T{});
    std::vector<uint8_t> packed((size_t)buf_sz, 0);
    fill_pattern(input.data(), elems);

    md              = fresh_metadata();
    md.block_params = const_cast<dlp_gemm_blocking_t*>(&bp);
    reorder_fn(order, trans, 'B', input.data(), (T*)packed.data(), k, n, ldb,
               &md);
    if (md.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED) {
        return false;
    }
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_SUCCESS)
        << "reorder k=" << k << " n=" << n << " NR=" << bp.NR;
    if (md.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        return true;
    }

    md              = fresh_metadata();
    md.block_params = const_cast<dlp_gemm_blocking_t*>(&bp);
    unreorder_fn(order, trans, 'B', (const T*)packed.data(), output.data(), k,
                 n, ldb, &md);
    if (md.error_hndl.error_code == DLP_CLSC_NOT_SUPPORTED) {
        return false;
    }
    EXPECT_EQ(md.error_hndl.error_code, DLP_CLSC_SUCCESS)
        << "un-reorder k=" << k << " n=" << n << " NR=" << bp.NR;
    if (md.error_hndl.error_code != DLP_CLSC_SUCCESS) {
        return true;
    }
    EXPECT_EQ(std::memcmp(input.data(), output.data(), elems * sizeof(T)), 0)
        << "k=" << k << " n=" << n << " NR=" << bp.NR << " NC=" << bp.NC
        << " KC=" << bp.KC;
    return true;
}

std::vector<YamlCase>
load_yaml_cases()
{
    const std::string path =
        std::string(TEST_CONFIG_DIR) + "/reorder_ref_coverage.yaml";
    YAML::Node root = YAML::LoadFile(path);
    YAML::Node sets = root["reorder_ref_tests"];
    if (!sets || !sets.IsSequence()) {
        throw std::runtime_error("reorder_ref_tests missing in " + path);
    }

    std::vector<YamlCase> out;
    for (const auto& set : sets) {
        const std::string name = set["name"].as<std::string>();
        const char     trans = set["trans"] ? set["trans"].as<std::string>()[0]
                                            : 'n';
        const uint64_t max_elems =
            set["max_elems"] ? set["max_elems"].as<uint64_t>() : 4000000ull;
        const uint64_t col_cap = set["col_major_max_elems"]
                                     ? set["col_major_max_elems"].as<uint64_t>()
                                     : 65536ull;

        std::vector<char> orders;
        for (const auto& o : set["orders"]) {
            orders.push_back(o.as<std::string>()[0]);
        }

        const std::vector<md_t> ks     = as_md_list(set["k"]);
        const std::vector<md_t> ns     = as_md_list(set["n"]);
        const std::vector<md_t> nrs    = as_md_list(set["nr"]);
        const std::vector<md_t> nc_mul = as_md_list(set["nc_mul"]);
        const std::vector<md_t> kcs    = as_md_list(set["kc"]);

        std::vector<dlp_gemm_blocking_t> blocks;
        for (md_t nr : nrs) {
            for (md_t mul : nc_mul) {
                for (md_t kc : kcs) {
                    blocks.push_back({ 0, nr, 0, nr * mul, kc });
                }
            }
        }

        for (const auto& dt : set["dtypes"]) {
            YamlCase proto;
            proto.set      = name;
            proto.dtype    = dt["name"].as<std::string>();
            proto.trans    = trans;
            proto.min_nr   = (md_t)dt["min_nr"].as<int64_t>();
            proto.k_factor = (md_t)dt["k_factor"].as<int64_t>();
            proto.blocks   = blocks;

            for (md_t k : ks) {
                for (md_t n : ns) {
                    if ((uint64_t)k * (uint64_t)n > max_elems) {
                        continue;
                    }
                    for (char order : orders) {
                        if (order == 'c'
                            && ((uint64_t)k * (uint64_t)n > col_cap)) {
                            continue;
                        }
                        YamlCase c = proto;
                        c.order    = order;
                        c.k        = k;
                        c.n        = n;
                        out.push_back(c);
                    }
                }
            }
        }
    }
    return out;
}

class ReorderRefYaml : public ::testing::TestWithParam<YamlCase>
{};

class PinnedThreadCount
{
  public:
    explicit PinnedThreadCount(md_t n_threads)
    {
        dlp_thread_set_num_threads(n_threads);
    }
    ~PinnedThreadCount() { dlp_thread_set_num_threads(-1); }
};

TEST_P(ReorderRefYaml, RoundTrip)
{
    const YamlCase&   c = GetParam();
    PinnedThreadCount pin((md_t)1);

    int ran = 0;
    for (const dlp_gemm_blocking_t& bp : c.blocks) {
        if (!blocks_legal(bp, c.min_nr, c.k_factor)) {
            continue;
        }
        SCOPED_TRACE(c.dtype + " k=" + std::to_string(c.k)
                     + " n=" + std::to_string(c.n) + " order=" + c.order
                     + " trans=" + c.trans + " NR=" + std::to_string(bp.NR));
        bool ok = false;
        if (c.dtype == "f32") {
            ok = round_trip_one<float>(aocl_get_reorder_buf_size_f32f32f32of32,
                                       aocl_reorder_f32f32f32of32_reference,
                                       aocl_unreorder_f32f32f32of32_reference,
                                       c.order, c.trans, c.k, c.n, bp);
        } else if (c.dtype == "bf16") {
            ok = round_trip_one<bfloat16>(
                aocl_get_reorder_buf_size_bf16bf16f32of32,
                aocl_reorder_bf16bf16f32of32_reference,
                aocl_unreorder_bf16bf16f32of32_reference, c.order, c.trans, c.k,
                c.n, bp);
        } else if (c.dtype == "s8") {
            ok = round_trip_one<int8_t>(aocl_get_reorder_buf_size_s8s8s32os32,
                                        aocl_reorder_s8s8s32os32_reference,
                                        aocl_unreorder_s8s8s32os32_reference,
                                        c.order, c.trans, c.k, c.n, bp);
        } else if (c.dtype == "u8s8") {
            ok = round_trip_one<int8_t>(aocl_get_reorder_buf_size_u8s8s32os32,
                                        aocl_reorder_u8s8s32os32_reference,
                                        aocl_unreorder_u8s8s32os32_reference,
                                        c.order, c.trans, c.k, c.n, bp);
        } else if (c.dtype == "fp16") {
            ok =
                round_trip_one<float16>(aocl_get_reorder_buf_size_f16f16f16of16,
                                        aocl_reorder_f16f16f16of16_reference,
                                        aocl_unreorder_f16f16f16of16_reference,
                                        c.order, c.trans, c.k, c.n, bp);
        } else {
            FAIL() << "unknown dtype " << c.dtype;
        }
        if (ok) {
            ran++;
        }
    }
    if (ran == 0) {
        GTEST_SKIP() << c.dtype << " had no legal blocking for this size";
    }
}

std::string
yaml_case_name(const ::testing::TestParamInfo<YamlCase>& info)
{
    const YamlCase& c = info.param;
    return c.set + "_" + c.dtype + "_k" + std::to_string(c.k) + "_n"
           + std::to_string(c.n) + "_" + c.order + "_" + c.trans;
}

INSTANTIATE_TEST_SUITE_P(YamlSize,
                         ReorderRefYaml,
                         ::testing::ValuesIn(load_yaml_cases()),
                         yaml_case_name);

} // namespace
