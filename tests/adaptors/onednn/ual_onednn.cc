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

#include "adaptors/onednn/ual_onednn.hh"
#include "adaptors/onednn/ual_plan_onednn.hh"
#include "framework/batch_gemm_args.hh"
#include "framework/operation.hh"
#include "utils/matrix_conversion_utils.hh"
#include <cstring>
#include <stdexcept>
#include <vector>

namespace dlp::testing::classic {

// ============================================================================
// Constructor
// ============================================================================

UalOneDNN::UalOneDNN()
    : IUal(UALType::ONEDNN)
    , m_engine(dnnl::engine::kind::cpu, 0)
    , m_stream(m_engine)
{
    // Engine and stream created once, reused for all operations
}

// ============================================================================
// Type Conversion Utilities
// ============================================================================

dnnl::memory::data_type
UalOneDNN::toOneDNNType(MatrixType type)
{
    switch (type) {
        case MatrixType::f32:
            return dnnl::memory::data_type::f32;
        case MatrixType::bf16:
            return dnnl::memory::data_type::bf16;
        case MatrixType::fp16:
            return dnnl::memory::data_type::f16;
        case MatrixType::s8:
            return dnnl::memory::data_type::s8;
        case MatrixType::u8:
            return dnnl::memory::data_type::u8;
        case MatrixType::s32:
            return dnnl::memory::data_type::s32;
        default:
            return dnnl::memory::data_type::undef;
    }
}

bool
UalOneDNN::isTypeSupported(MatrixType type)
{
    switch (type) {
        case MatrixType::f32:
        case MatrixType::bf16:
        case MatrixType::fp16:
        case MatrixType::s8:
        case MatrixType::u8:
        case MatrixType::s32:
            return true;
        default:
            return false;
    }
}

bool
UalOneDNN::isSaturatingAvx2IntGemm(MatrixType a_type,
                                   MatrixType b_type,
                                   MatrixType acc_type)
{
    if (acc_type != MatrixType::s32) {
        return false;
    }

    const bool is_u8s8 = (a_type == MatrixType::u8 && b_type == MatrixType::s8);
    const bool is_s8s8 = (a_type == MatrixType::s8 && b_type == MatrixType::s8);
    if (!is_u8s8 && !is_s8s8) {
        return false;
    }

    // avx2_vnni (and everything above it in this monotonically-ordered
    // enum -- see dnnl_cpu_isa_t) has a genuine non-saturating int8
    // dot-product instruction, EXCEPT avx512_core (0x27): it sorts above
    // avx2_vnni (0x0f) numerically but is plain AVX-512F/CD/BW/DQ/VL with
    // no VNNI/DL-Boost, so it falls back to the same saturating
    // VPMADDUBSW-based emulation as plain avx2/avx/sse41/default. Every
    // other level at or above avx512_core_vnni genuinely has VNNI, so
    // avx512_core is the sole exception that needs an explicit check.
    const auto isa = dnnl::get_effective_cpu_isa();
    return isa < dnnl::cpu_isa::avx2_vnni || isa == dnnl::cpu_isa::avx512_core;
}

bool
UalOneDNN::isExactlyRepresentableAsFloat(int32_t value)
{
    return static_cast<int32_t>(static_cast<float>(value)) == value;
}

// ============================================================================
// Memory Descriptor Creation
// ============================================================================

dnnl::memory::desc
UalOneDNN::createMemoryDesc(md_t         rows,
                            md_t         cols,
                            MatrixType   type,
                            MatrixLayout layout,
                            bool         transposed,
                            md_t         leadingDim)
{
    auto dnnl_type = toOneDNNType(type);

    // Logical dimensions for matmul: A[M,K], B[K,N], C[M,N]
    // The dims represent the logical shape of the matrix operation
    dnnl::memory::dims dims = { static_cast<dnnl::memory::dim>(rows),
                                static_cast<dnnl::memory::dim>(cols) };

    // Calculate strides based on layout and transposition
    // Stride defines how to traverse memory to access element (i, j)
    dnnl::memory::dims strides;

    if (layout == MatrixLayout::ROW_MAJOR) {
        if (transposed) {
            // Transposed row-major: element(i,j) = data[j * ld + i]
            // Row stride = 1, Column stride = ld
            strides = { 1, static_cast<dnnl::memory::dim>(leadingDim) };
        } else {
            // Normal row-major: element(i,j) = data[i * ld + j]
            // Row stride = ld, Column stride = 1
            strides = { static_cast<dnnl::memory::dim>(leadingDim), 1 };
        }
    } else { // COLUMN_MAJOR
        if (transposed) {
            // Transposed column-major: element(i,j) = data[i * ld + j]
            // Row stride = ld, Column stride = 1
            strides = { static_cast<dnnl::memory::dim>(leadingDim), 1 };
        } else {
            // Normal column-major: element(i,j) = data[j * ld + i]
            // Row stride = 1, Column stride = ld
            strides = { 1, static_cast<dnnl::memory::dim>(leadingDim) };
        }
    }

    return dnnl::memory::desc(dims, dnnl_type, strides);
}

// ============================================================================
// Alpha/Beta Scaling Helper
// ============================================================================

void
UalOneDNN::applyAlphaScaling(void*        data,
                             MatrixType   type,
                             md_t         m,
                             md_t         n,
                             md_t         ld,
                             MatrixLayout layout,
                             double       alpha)
{
    // Skip if alpha is 1.0
    if (alpha == 1.0) {
        return;
    }

    size_t rows = static_cast<size_t>(m);
    size_t cols = static_cast<size_t>(n);

    switch (type) {
        case MatrixType::f32: {
            float* ptr     = static_cast<float*>(data);
            float  alpha_f = static_cast<float>(alpha);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    size_t idx = (layout == MatrixLayout::ROW_MAJOR)
                                     ? i * static_cast<size_t>(ld) + j
                                     : j * static_cast<size_t>(ld) + i;
                    ptr[idx] *= alpha_f;
                }
            }
            break;
        }
        case MatrixType::bf16: {
            // bf16 scaling - convert to float, scale, convert back
            // Note: This is inefficient but correct for Phase 1
            uint16_t* ptr     = static_cast<uint16_t*>(data);
            float     alpha_f = static_cast<float>(alpha);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    size_t idx = (layout == MatrixLayout::ROW_MAJOR)
                                     ? i * static_cast<size_t>(ld) + j
                                     : j * static_cast<size_t>(ld) + i;
                    // Convert bf16 to f32, scale, convert back
                    uint32_t bf16_bits = static_cast<uint32_t>(ptr[idx]) << 16;
                    float    val;
                    std::memcpy(&val, &bf16_bits, sizeof(float));
                    val *= alpha_f;
                    uint32_t result_bits;
                    std::memcpy(&result_bits, &val, sizeof(uint32_t));
                    ptr[idx] = static_cast<uint16_t>(result_bits >> 16);
                }
            }
            break;
        }
        case MatrixType::s32: {
            int32_t* ptr = static_cast<int32_t*>(data);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    size_t idx = (layout == MatrixLayout::ROW_MAJOR)
                                     ? i * static_cast<size_t>(ld) + j
                                     : j * static_cast<size_t>(ld) + i;
                    ptr[idx]   = static_cast<int32_t>(
                        static_cast<double>(ptr[idx]) * alpha);
                }
            }
            break;
        }
        default:
            // Unsupported type for alpha scaling
            break;
    }
}

/**
 * @brief Apply combined alpha and beta scaling
 *
 * Computes: C = alpha * C + beta * C_old
 * where C contains the matmul result and C_old is the original C data
 */
static void
applyAlphaBetaScaling(void*        c_data,
                      void*        c_old_data,
                      MatrixType   type,
                      md_t         m,
                      md_t         n,
                      md_t         ld,
                      MatrixLayout layout,
                      double       alpha,
                      double       beta)
{
    size_t rows = static_cast<size_t>(m);
    size_t cols = static_cast<size_t>(n);

    switch (type) {
        case MatrixType::f32: {
            float* c_ptr     = static_cast<float*>(c_data);
            float* c_old_ptr = static_cast<float*>(c_old_data);
            float  alpha_f   = static_cast<float>(alpha);
            float  beta_f    = static_cast<float>(beta);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    size_t idx   = (layout == MatrixLayout::ROW_MAJOR)
                                       ? i * static_cast<size_t>(ld) + j
                                       : j * static_cast<size_t>(ld) + i;
                    float  c_new = c_ptr[idx];
                    float  c_old = (c_old_ptr != nullptr) ? c_old_ptr[idx]
                                                          : 0.0f;
                    c_ptr[idx]   = alpha_f * c_new + beta_f * c_old;
                }
            }
            break;
        }
        case MatrixType::bf16: {
            uint16_t* c_ptr     = static_cast<uint16_t*>(c_data);
            uint16_t* c_old_ptr = static_cast<uint16_t*>(c_old_data);
            float     alpha_f   = static_cast<float>(alpha);
            float     beta_f    = static_cast<float>(beta);

            auto bf16_to_f32 = [](uint16_t bf16) -> float {
                uint32_t bits = static_cast<uint32_t>(bf16) << 16;
                float    val;
                std::memcpy(&val, &bits, sizeof(float));
                return val;
            };

            auto f32_to_bf16 = [](float val) -> uint16_t {
                uint32_t bits;
                std::memcpy(&bits, &val, sizeof(uint32_t));
                return static_cast<uint16_t>(bits >> 16);
            };

            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    size_t idx   = (layout == MatrixLayout::ROW_MAJOR)
                                       ? i * static_cast<size_t>(ld) + j
                                       : j * static_cast<size_t>(ld) + i;
                    float  c_new = bf16_to_f32(c_ptr[idx]);
                    float  c_old = (c_old_ptr != nullptr)
                                       ? bf16_to_f32(c_old_ptr[idx])
                                       : 0.0f;
                    c_ptr[idx] = f32_to_bf16(alpha_f * c_new + beta_f * c_old);
                }
            }
            break;
        }
        case MatrixType::s32: {
            int32_t* c_ptr     = static_cast<int32_t*>(c_data);
            int32_t* c_old_ptr = static_cast<int32_t*>(c_old_data);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    size_t idx   = (layout == MatrixLayout::ROW_MAJOR)
                                       ? i * static_cast<size_t>(ld) + j
                                       : j * static_cast<size_t>(ld) + i;
                    double c_new = static_cast<double>(c_ptr[idx]);
                    double c_old = (c_old_ptr != nullptr)
                                       ? static_cast<double>(c_old_ptr[idx])
                                       : 0.0;
                    c_ptr[idx] =
                        static_cast<int32_t>(alpha * c_new + beta * c_old);
                }
            }
            break;
        }
        default:
            break;
    }
}

// ============================================================================
// Primitive Caching
// ============================================================================

UalOneDNN::CachedPrimitive&
UalOneDNN::getOrCreatePrimitive(const PrimitiveKey& key)
{
    auto it = m_primitiveCache.find(key);
    if (it != m_primitiveCache.end()) {
        return it->second;
    }

    // Create memory descriptors based on key
    auto a_md = createMemoryDesc(key.m, key.k, key.a_type, key.a_layout,
                                 key.trans_a, key.lda);
    auto b_md = createMemoryDesc(key.k, key.n, key.b_type, key.b_layout,
                                 key.trans_b, key.ldb);
    auto c_md = createMemoryDesc(key.m, key.n, key.c_type, key.c_layout, false,
                                 key.ldc);

    // Create primitive attributes with native alpha/beta support
    //
    // We want: C = alpha * (A * B) + beta * C_old
    //
    // oneDNN strategy (single fused kernel pass):
    // - SRC scales: multiplies A by alpha before matmul → alpha * (A @ B)
    // - Sum post-op: adds beta * C_old to matmul result
    // - Combined: dst = alpha * (A @ B) + beta * C_old  ✓
    //
    // This avoids any post-execution scaling loops.
    //
    // Special case: alpha = 0 is handled at execution time (skip matmul)
    dnnl::primitive_attr attr;

    // Native alpha: SRC scales → the matmul kernel applies alpha * A internally
    // The actual alpha value is provided at execution time via
    // DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC
    if (key.alpha != 1.0f) {
        attr.set_scales_mask(DNNL_ARG_SRC, 0);
    }

    // Native beta: sum post-op → kernel reads C_old and adds beta * C_old
    if (key.beta != 0.0f) {
        dnnl::post_ops po;
        po.append_sum(key.beta);
        attr.set_post_ops(po);
    }

    // Create primitive descriptor and primitive
    auto pd   = dnnl::matmul::primitive_desc(m_engine, a_md, b_md, c_md, attr);
    auto prim = dnnl::matmul(pd);

    // Cache and return
    CachedPrimitive cached;
    cached.prim = std::move(prim);
    cached.a_md = std::move(a_md);
    cached.b_md = std::move(b_md);
    cached.c_md = std::move(c_md);
    cached.attr = std::move(attr);

    auto [iter, inserted] = m_primitiveCache.emplace(key, std::move(cached));
    return iter->second;
}

// ============================================================================
// Weight Reorder Implementation (Optimal Blocking for Fair Benchmarking)
// ============================================================================

/**
 * @brief Create matmul primitive descriptor for weight blocking query
 *
 * Creates a matmul primitive descriptor with format_tag::any for weights
 * to query the optimal blocking format. This follows the ZenDNN pattern
 * for fair benchmarking between AOCL-DLP and oneDNN.
 */
dnnl::matmul::primitive_desc
UalOneDNN::createBlockedMatmulPD(md_t         m,
                                 md_t         n,
                                 md_t         k,
                                 MatrixType   a_type,
                                 MatrixType   b_type,
                                 MatrixType   c_type,
                                 bool         trans_a,
                                 bool         trans_b,
                                 MatrixLayout a_layout,
                                 MatrixLayout b_layout,
                                 MatrixLayout c_layout,
                                 md_t         lda,
                                 md_t         ldb,
                                 md_t         ldc)
{
    // Create memory descriptors for A and C (with explicit strides)
    auto a_md = createMemoryDesc(m, k, a_type, a_layout, trans_a, lda);
    auto c_md = createMemoryDesc(m, n, c_type, c_layout, false, ldc);

    // Create B memory descriptor with format_tag::any for optimal blocking
    // This allows oneDNN to choose the best internal format
    auto               dnnl_b_type = toOneDNNType(b_type);
    dnnl::memory::dims b_dims      = { static_cast<dnnl::memory::dim>(k),
                                       static_cast<dnnl::memory::dim>(n) };

    // Use format_tag::any to let oneDNN choose optimal blocking
    dnnl::memory::desc b_md_any(b_dims, dnnl_b_type,
                                dnnl::memory::format_tag::any);

    // Create primitive descriptor - this will determine optimal B format
    return dnnl::matmul::primitive_desc(m_engine, a_md, b_md_any, c_md);
}

UALError
UalOneDNN::reorder(const Matrix& in,
                   Matrix&       out,
                   MatrixType    A_type,
                   MatrixType    B_type,
                   MatrixType    C_type,
                   MatrixType    accType,
                   md_t          M_param,
                   bool          transA,
                   const GroupScaleParam* /*group_scale*/,
                   const WOQParam* /*woq*/)
{
    // This reorder implementation uses oneDNN's dnnl::reorder primitive with
    // format_tag::any to transform weights into the optimal blocked format.
    // This enables "apples-to-apples" benchmarking with AOCL-DLP:
    //
    // - DLP: Uses explicit aocl_reorder_* to pack weights into blocked format
    // - oneDNN: Uses dnnl::reorder to transform to optimal blocked format
    //
    // Both approaches have:
    // - One-time upfront cost: Weight transformation/reordering
    // - Zero per-call overhead: Use pre-transformed weights in GEMM

    try {
        // Validate input type is supported
        if (!isTypeSupported(in.getMatrixType())) {
            return UALError::UAL_NOT_SUPPORTED;
        }

        // Get weight dimensions (K x N for B matrix)
        md_t K = in.getEffectiveRows();
        md_t N = in.getEffectiveCols();

        // Validate leading dimension for input matrix B
        // For row-major: ldb >= N (not transposed) or ldb >= K (transposed)
        // For col-major: ldb >= K (not transposed) or ldb >= N (transposed)
        md_t ldb        = in.getLeadingDimension();
        bool row_major  = (in.getLayout() == MatrixLayout::ROW_MAJOR);
        bool transposed = in.isTransposed();

        if (row_major) {
            md_t min_ldb = transposed ? K : N;
            if (ldb < min_ldb) {
                return UALError::UAL_FAILURE;
            }
        } else {
            md_t min_ldb = transposed ? N : K;
            if (ldb < min_ldb) {
                return UALError::UAL_FAILURE;
            }
        }

        // Use M parameter or default heuristic (256 is typical for inference)
        md_t M = (M_param > 0) ? M_param : 256;

        // Determine C type (use same as input type for simplicity if C_type not
        // specified) Note: C_type is always valid in our framework, so just use
        // it directly
        MatrixType c_type = C_type;

        // Create cache key for weight lookup
        WeightKey key;
        key.data_ptr = in.getData();
        key.k        = K;
        key.n        = N;
        key.m        = M;
        key.a_type   = A_type;
        key.b_type   = B_type;
        key.c_type   = c_type;
        key.trans_a  = transA;
        key.trans_b  = in.isTransposed();
        key.ldb      = in.getLeadingDimension();
        key.b_layout = in.getLayout();

        // An explicit reorder() call must observe the input's *current*
        // contents. The cache key is address + shape/type metadata only, not
        // a content hash, so a repeated call on the same buffer (refilled
        // with new data) or a new Matrix that happens to reuse a freed
        // allocation's address would otherwise silently return stale blocked
        // weights from a previous call. Drop any existing entry so the
        // lookup below always misses and rebuilds from the live input.
        m_weightCache.erase(key);
        auto cache_it = m_weightCache.find(key);
        if (cache_it == m_weightCache.end()) {
            // Cache miss - need to create blocked weights using dnnl::reorder

            // Create source memory descriptor (plain format)
            auto src_md =
                createMemoryDesc(K, N, in.getMatrixType(), in.getLayout(),
                                 in.isTransposed(), in.getLeadingDimension());

            // Create matmul primitive descriptor to query optimal B format
            md_t lda = transA ? M : K; // A is M x K
            md_t ldc = N;              // C is M x N

            auto matmul_pd = createBlockedMatmulPD(
                M, N, K, A_type, B_type, c_type, transA, in.isTransposed(),
                MatrixLayout::ROW_MAJOR, // Assume row-major for A
                in.getLayout(),
                MatrixLayout::ROW_MAJOR, // Assume row-major for C
                lda, in.getLeadingDimension(), ldc);

            // Get optimal blocked memory descriptor from primitive descriptor
            auto blocked_desc = matmul_pd.weights_desc();

            // Create source memory object
            dnnl::memory src_mem(src_md, m_engine,
                                 const_cast<void*>(in.getData()));

            // Create destination memory object with optimal blocked format
            dnnl::memory blocked_mem(blocked_desc, m_engine);

            // Execute reorder: plain -> blocked
            dnnl::reorder reorder_prim(src_mem, blocked_mem);
            reorder_prim.execute(m_stream, src_mem, blocked_mem);
            m_stream.wait();

            // Cache the blocked weights
            CachedWeight cached;
            cached.blocked_memory = std::move(blocked_mem);
            cached.blocked_desc   = blocked_desc;
            cached.k              = K;
            cached.n              = N;
            m_weightCache[key]    = std::move(cached);

            // Update iterator to point to newly inserted entry
            cache_it = m_weightCache.find(key);
        }

        // Now cache_it points to the cached weight (either existing or just
        // created)
        const auto& cached = cache_it->second;

        // Get size of blocked memory
        size_t blocked_size = cached.blocked_desc.get_size();

        // Create output matrix with a copy of the blocked data.
        // The blocked data is owned by m_weightCache (dnnl::memory objects).
        // When UalOneDNN is destroyed, m_weightCache is destroyed, which
        // destroys all CachedWeight entries, which destroys their dnnl::memory
        // objects, which frees the underlying blocked data buffers.
        //
        // FIXME: This copy can be avoided if Matrix class supports external
        // allocation (memory that shouldn't be deallocated on Matrix
        // destruction). The cached.blocked_memory already contains the
        // transformed data, and its lifetime is tied to UalOneDNN. We could
        // have Matrix point directly to cached.blocked_memory.get_data_handle()
        // if Matrix had a flag like m_ownsMemory = false to skip deallocation
        // in its destructor.
        //
        // Use input's physical dimensions to preserve effective dimensions
        // correctly when the transpose flag is preserved.
        // For row-major: ld = physical cols = in.getCols()
        // For col-major: ld = physical rows = in.getRows()
        md_t out_ld = (in.getLayout() == MatrixLayout::ROW_MAJOR)
                          ? in.getCols()
                          : in.getRows();

        uint8_t* memory = dlp::testing::framework::MatrixMemory::allocateBytes(
            blocked_size, 64);

        // Copy blocked data to output matrix
        void* blocked_ptr = cached.blocked_memory.get_data_handle();
        std::memcpy(memory, blocked_ptr, blocked_size);

        // Preserve the transpose flag from input - the blocked data is
        // conceptually still for a transposed or non-transposed matrix, just in
        // a different format.
        // CRITICAL: Use input's physical dimensions (getRows/getCols) not
        // effective dimensions (K/N) to ensure getEffectiveRows/Cols return
        // the correct logical dimensions after considering transposition.
        out = Matrix(in.getRows(), in.getCols(), in.getMatrixType(), memory,
                     blocked_size, in.getLayout(), out_ld, in.isTransposed(),
                     true, 64);

        // Record the EXTERNAL parameters (M, A's transpose, C's type) this
        // reorder() call used to choose B's blocked format. oneDNN's
        // format_tag::any blocking choice is a function of these, not just
        // B's own k/n/type/layout -- so a later GEMM call that reuses this
        // Matrix (directly, or via a Matrix copy/move, both of which carry
        // this context -- see Matrix::operator=/Matrix(Matrix&&)) with a
        // DIFFERENT M (e.g. the M omitted here and defaulted below, then
        // the matrix reused at a smaller/larger M) can verify the mismatch
        // instead of silently binding a re-derived descriptor to bytes it
        // doesn't actually describe. See OneDNNUalPlan::execute()'s
        // matching check and UalOneDNN::gemm()'s unconditional reject for
        // this raw-pointer path (which has no Matrix/context to check).
        out.setReorderContext(M, transA, c_type);

        // Cache the format descriptor by matrix properties (without data_ptr).
        // This is needed because gemm() will receive a COPY of the output
        // matrix (via B = B_reordered assignment), which has a different data
        // pointer. The format descriptor is the same regardless of data
        // location.
        FormatKey fmt_key;
        fmt_key.k        = K;
        fmt_key.n        = N;
        fmt_key.m        = M;
        fmt_key.a_type   = A_type;
        fmt_key.b_type   = B_type;
        fmt_key.c_type   = c_type;
        fmt_key.trans_a  = transA;
        fmt_key.trans_b  = in.isTransposed();
        fmt_key.b_layout = out.getLayout();

        m_weightFormatCache[fmt_key] = cached.blocked_desc;

        // The blocked data (cached.blocked_memory) has already been copied
        // into `out`'s own buffer above, and the only thing worth keeping
        // long-term -- the format descriptor -- was just captured into
        // m_weightFormatCache. No GEMM path reads m_weightCache (it's only
        // ever written/read within this function), and the unconditional
        // erase() at the top of reorder() means a repeated call never
        // reuses this entry either. Without this, every distinct
        // (data_ptr, shape, type) reorder() call leaves its CachedWeight
        // (and its underlying dnnl::memory data buffer) sitting in
        // m_weightCache for the rest of this UalOneDNN instance's
        // lifetime -- an unbounded leak across a test run that reuses one
        // suite-wide instance (e.g. test_gemm) across many reorder() calls.
        m_weightCache.erase(key);

        return UALError::UAL_SUCCESS;

    } catch (const dnnl::error& e) {
        // oneDNN error - check if it's an unsupported configuration
        // dnnl_unimplemented means the type combination isn't supported
        if (e.status == dnnl_unimplemented) {
            return UALError::UAL_NOT_SUPPORTED;
        }
        return UALError::UAL_FAILURE;
    } catch (const std::exception& e) {
        return UALError::UAL_FAILURE;
    }
}

// ============================================================================
// UalOneDNN: createPlan / getUALType
// ============================================================================

std::unique_ptr<dlp::testing::framework::IUalPlan>
UalOneDNN::createPlan()
{
    return std::make_unique<OneDNNUalPlan>(*this);
}

// ============================================================================
// OneDNNUalPlan Implementation
// ============================================================================

OneDNNUalPlan::OneDNNUalPlan(UalOneDNN& ual)
    : m_ual(ual)
{
}

// ----------------------------------------------------------------------------
// Post-op dispatch skeleton
// ----------------------------------------------------------------------------
//
// Single centralised extension point for oneDNN native post-op fusion. Ordering
// is fixed here so families never have to reason about it: the beta sum-post-op
// is appended FIRST (oneDNN requires sum to precede eltwise), then getPostOps()
// is iterated in chain order.
//
// Each `case` below is a FAMILY-SEAM: a family agent replaces exactly ONE throw
// with real native oneDNN calls (append_eltwise, DST scales, a binary/prelu
// operand registered into m_postop_mems + m_exec_args, etc.). Do NOT wrap REF
// post-op loops around oneDNN output — that would destroy the oracle
// independence. Everything must go through oneDNN's native post_ops / attr API.
//
// Bias is deliberately NOT handled here: in oneDNN bias is a matmul
// primitive-descriptor constructor argument, not a post_op. The bias seam sets
// m_has_bias / m_bias_md so prepare() selects the bias-carrying PD overload;
// see the OperationType::Bias case for the marker.
// ----------------------------------------------------------------------------
// Eltwise family helper (FAMILY-SEAM: eltwise)
//
// Maps a framework ElementWiseParam onto a single native oneDNN eltwise
// post-op. Every mapping below is a bit-for-formula match against the REF
// oracle (tests/adaptors/ref/ual_ref.cc):
//   Relu      max(0,x)                          -> eltwise_relu (alpha=0)
//   Prelu     x<0 ? alpha*x : x  (scalar alpha) -> eltwise_relu (alpha=slope)
//   Gelu_Tanh 0.5x(1+tanh(...))                 -> eltwise_gelu_tanh
//   Gelu_Erf  0.5x(1+erf(x/sqrt2))              -> eltwise_gelu_erf
//   Clip      max(lo,min(hi,x))                 -> eltwise_clip(lo,hi)
//   Swish     x*sigmoid(alpha*x)                -> eltwise_swish(alpha)
//   Tanh      tanh(x)                           -> eltwise_tanh
//   Sigmoid   1/(1+e^-x)                        -> eltwise_logistic
//   Mish      x*tanh(softplus(x))               -> eltwise_mish
//
// REF defaults (used when the corresponding operand Matrix is absent) are
// reproduced exactly here: Prelu slope 0.01, Clip [-6,6], Swish alpha 2.0.
// oneDNN has no scalar-broadcast native prelu (its eltwise_prelu needs a
// per-channel weights operand); the REF prelu is a scalar leaky-relu, which is
// exactly oneDNN relu with a non-zero negative slope, so we map it there.
void
OneDNNUalPlan::appendEltwisePostOp(dnnl::post_ops&         po,
                                   const ElementWiseParam& ew,
                                   int&                    post_op_index)
{
    auto readScalar = [](const Matrix* m, float fallback) -> float {
        if (m == nullptr) {
            return fallback;
        }
        return dlp::testing::utils::convertTo<float>(m->getData(),
                                                     m->getMatrixType(), 0);
    };

    switch (ew.getOperation()) {
        case ElementWiseOperation::Relu:
            po.append_eltwise(dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
            break;
        case ElementWiseOperation::Prelu: {
            // REF: scalar alpha (default 0.01), x<0 ? alpha*x : x.
            float slope = readScalar(ew.getAlpha(), 0.01f);
            po.append_eltwise(dnnl::algorithm::eltwise_relu, slope, 0.0f);
            break;
        }
        case ElementWiseOperation::Gelu_Tanh:
            po.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0.0f, 0.0f);
            break;
        case ElementWiseOperation::Gelu_Erf:
            po.append_eltwise(dnnl::algorithm::eltwise_gelu_erf, 0.0f, 0.0f);
            break;
        case ElementWiseOperation::Clip: {
            // REF defaults: lower -6, upper 6.
            float lo = readScalar(ew.getAlpha(), -6.0f);
            float hi = readScalar(ew.getBeta(), 6.0f);
            po.append_eltwise(dnnl::algorithm::eltwise_clip, lo, hi);
            break;
        }
        case ElementWiseOperation::Swish: {
            // REF default alpha 2.0; oneDNN swish = x/(1+e^{-alpha x}).
            float alpha = readScalar(ew.getAlpha(), 2.0f);
            po.append_eltwise(dnnl::algorithm::eltwise_swish, alpha, 0.0f);
            break;
        }
        case ElementWiseOperation::Tanh:
            po.append_eltwise(dnnl::algorithm::eltwise_tanh, 0.0f, 0.0f);
            break;
        case ElementWiseOperation::Sigmoid:
            po.append_eltwise(dnnl::algorithm::eltwise_logistic, 0.0f, 0.0f);
            break;
        case ElementWiseOperation::Mish:
            po.append_eltwise(dnnl::algorithm::eltwise_mish, 0.0f, 0.0f);
            break;
        default:
            // Unknown eltwise op: no native mapping. Defer to a skip rather
            // than emulate (would destroy oracle independence). Do NOT advance
            // the index — nothing was appended.
            markNotSupported("oneDNN plan: unsupported element-wise operation");
            return;
    }
    // Every mapped case above appends exactly one eltwise post-op.
    ++post_op_index;
}

// ----------------------------------------------------------------------------
// Scale family helper (FAMILY-SEAM: scale)
//
// REF semantic (tests/adaptors/ref/ual_ref.cc, applyScale):
//     out = x * scale + zp
// with scale-then-add zero-point (NOT sf*(x+zp)). Broadcast:
//   PerTensor  (1x1): scalar scale, scalar zp.
//   PerChannel (1xN): scale indexed by output column (N axis).
//   PerToken   (Mx1): scale indexed by output row    (M axis).
//   zero-point: REF always indexes zp by output column (N axis), whether the
//               zp is scalar or a vector.
//
// Order-preserving mapping (Scale can appear mid-chain, so DST scales are NOT
// usable — those apply after the whole pipeline). Everything below is appended
// into the chain at the Scale op's position:
//   * PerTensor: fold into one eltwise_linear(alpha=scale, beta=zp).
//   * Vector scale: binary_mul with a broadcast SF operand ([1,N] or [M,1]),
//     followed (only when a zero-point exists) by binary_add with a broadcast
//     ZP operand ([1,N] scalar-or-vector over the N axis).
// SF/ZP values are copied into owned dnnl::memory kept in m_postop_mems so the
// operands outlive prepare(); the matching exec-args are registered under
// DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1.
void
OneDNNUalPlan::appendScalePostOp(dnnl::post_ops&   po,
                                 const ScaleParam& sp,
                                 int&              post_op_index)
{
    const Matrix* sf  = sp.getScaleFactor();
    const Matrix* zp  = sp.getZeroPoint();
    ParamDim      dim = sp.getScaleFactorDim();

    // A Scale with neither operand is a no-op.
    if (sf == nullptr && zp == nullptr) {
        return;
    }

    auto convertVector = [](const Matrix* m) -> std::vector<float> {
        std::vector<float> out;
        if (m == nullptr) {
            return out;
        }
        size_t count = static_cast<size_t>(m->getRows()) * m->getCols();
        out.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            out.push_back(dlp::testing::utils::convertTo<float>(
                m->getData(), m->getMatrixType(), i));
        }
        return out;
    };

    size_t sf_count = sf ? static_cast<size_t>(sf->getRows()) * sf->getCols()
                         : 0;
    size_t zp_count = zp ? static_cast<size_t>(zp->getRows()) * zp->getCols()
                         : 0;

    // ── Per-tensor fast path: single native eltwise_linear (alpha*x + beta).
    // Valid whenever both operands are effectively scalar (count <= 1),
    // regardless of the ParamDim tag: a scalar SF/ZP broadcasts identically
    // across every element, so a scalar labelled PerChannel/PerToken is still a
    // per-tensor scale. This preserves chain order and folds scale + zero-point
    // into one op (and avoids needlessly skipping a scalar tagged non-tensor).
    bool sf_scalar = (sf == nullptr) || (sf_count == 1);
    bool zp_scalar = (zp == nullptr) || (zp_count == 1);
    if (sf_scalar && zp_scalar) {
        float scale = 1.0f;
        if (sf) {
            scale = dlp::testing::utils::convertTo<float>(
                sf->getData(), sf->getMatrixType(), 0);
        }
        float zpv = 0.0f;
        if (zp) {
            zpv = dlp::testing::utils::convertTo<float>(zp->getData(),
                                                        zp->getMatrixType(), 0);
        }
        po.append_eltwise(dnnl::algorithm::eltwise_linear, scale, zpv);
        ++post_op_index;
        return;
    }

    // ── Vector scale path: binary_mul broadcast operand.
    // Output C is 2D [M, N]. PerChannel broadcasts over N (dims [1, N]);
    // PerToken broadcasts over M (dims [M, 1]).
    if (sf != nullptr) {
        dnnl::memory::dims sf_dims;
        if (dim == ParamDim::PerToken) {
            sf_dims = { static_cast<dnnl::memory::dim>(m_m), 1 };
        } else {
            // PerChannel (and any residual PerTensor with a vector SF, which
            // should not occur but is handled defensively as per-N).
            sf_dims = { 1, static_cast<dnnl::memory::dim>(m_n) };
        }

        size_t             expected = (dim == ParamDim::PerToken)
                                          ? static_cast<size_t>(m_m)
                                          : static_cast<size_t>(m_n);
        std::vector<float> sf_vals  = convertVector(sf);
        if (sf_vals.size() != expected) {
            // SF length does not match the broadcast axis: no faithful native
            // mapping. Skip rather than silently truncate/pad.
            markNotSupported(
                "oneDNN plan: scale-factor length does not match output axis");
            return;
        }

        dnnl::memory::desc sf_md(sf_dims, dnnl::memory::data_type::f32,
                                 dnnl::memory::format_tag::ab);
        dnnl::memory       sf_mem(sf_md, m_ual.m_engine);
        std::memcpy(sf_mem.get_data_handle(), sf_vals.data(),
                    sf_vals.size() * sizeof(float));
        m_postop_mems.push_back(sf_mem);

        po.append_binary(dnnl::algorithm::binary_mul, sf_md);
        m_exec_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index)
                    | DNNL_ARG_SRC_1] = sf_mem;
        ++post_op_index;
    }

    // ── Zero-point (only reached on the vector path). REF always indexes zp by
    // the output column (N axis), so broadcast the zp over N: dims [1, N] for a
    // vector zp, or [1, 1] for a scalar zp. Applied as a binary_add AFTER the
    // scale multiply — matching REF's out = x*scale + zp evaluated in place.
    if (zp != nullptr) {
        std::vector<float> zp_vals = convertVector(zp);

        dnnl::memory::dims zp_dims;
        if (zp_count == 1) {
            zp_dims = { 1, 1 };
        } else if (zp_vals.size() == static_cast<size_t>(m_n)) {
            zp_dims = { 1, static_cast<dnnl::memory::dim>(m_n) };
        } else {
            markNotSupported(
                "oneDNN plan: zero-point length does not match N axis");
            return;
        }

        dnnl::memory::desc zp_md(zp_dims, dnnl::memory::data_type::f32,
                                 dnnl::memory::format_tag::ab);
        dnnl::memory       zp_mem(zp_md, m_ual.m_engine);
        std::memcpy(zp_mem.get_data_handle(), zp_vals.data(),
                    zp_vals.size() * sizeof(float));
        m_postop_mems.push_back(zp_mem);

        po.append_binary(dnnl::algorithm::binary_add, zp_md);
        m_exec_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index)
                    | DNNL_ARG_SRC_1] = zp_mem;
        ++post_op_index;
    }
}

void
OneDNNUalPlan::buildPostOpChain(dnnl::post_ops& po, dnnl::primitive_attr& attr)
{
    // post_op_index tracks the append position so binary post-ops can register
    // their exec-arg operand under DNNL_ARG_ATTR_MULTIPLE_POST_OP(index). It
    // must count EVERY appended post-op (including the beta sum) so the index
    // matches oneDNN's internal ordering.
    int post_op_index = 0;

    // Beta (C = ... + beta * C_old) is a sum post-op and must come first.
    if (m_beta != 0.0) {
        po.append_sum(static_cast<float>(m_beta));
        ++post_op_index;
    }

    for (const auto& param : getPostOps()) {
        switch (param->getType()) {
            case OperationType::ElementWise: { // FAMILY-SEAM: eltwise
                // appendEltwisePostOp advances post_op_index internally (kept
                // symmetric with appendScalePostOp) so binary operands from any
                // family register on the correct slot.
                appendEltwisePostOp(
                    po, static_cast<const ElementWiseParam&>(*param),
                    post_op_index);
                break;
            }
            case OperationType::Bias: // FAMILY-SEAM: bias
                // Bias is NOT a oneDNN post_op: it is a matmul
                // primitive-descriptor constructor argument (an extra bias md
                // bound at execute time under DNNL_ARG_BIAS). The staging is
                // done once in stageBias() (called from prepare() BEFORE the PD
                // is built) so this seam records state only and never appends
                // to `po`. Reaching here means stageBias() decided the config
                // is unsupported (e.g. SF/ZP dequant, non-leading bias, second
                // bias) — but prepare() defers via markNotSupported() rather
                // than building the chain, so this case must be a no-op.
                break;
            case OperationType::Scale: { // FAMILY-SEAM: scale
                appendScalePostOp(po, static_cast<const ScaleParam&>(*param),
                                  post_op_index);
                break;
            }
            case OperationType::MatAdd: // FAMILY-SEAM: matadd
                throw std::runtime_error(
                    "oneDNN MatAdd post-op not yet implemented");
            case OperationType::MatMul: // FAMILY-SEAM: matmul
                throw std::runtime_error(
                    "oneDNN MatMul post-op not yet implemented");
            case OperationType::GLU: // FAMILY-SEAM: glu
                throw std::runtime_error(
                    "oneDNN does not support GLU post-op (shape-changing, no "
                    "native equivalent)");
            default:
                throw std::runtime_error(
                    "oneDNN plan: unsupported post-op type");
        }
    }

    attr.set_post_ops(po);
}

// ----------------------------------------------------------------------------
// Bias staging (matmul primitive-descriptor constructor argument)
// ----------------------------------------------------------------------------
//
// In oneDNN, bias is NOT a post_op: it is passed to the bias-carrying
// matmul::primitive_desc overload and bound at execute() under DNNL_ARG_BIAS.
// oneDNN adds the bias buffer (broadcast across M rows) to the matmul result
// BEFORE any post-op runs. This matches the REF chain only when Bias is the
// first entry (REF applies ops in chain order, and plain bias is C[i,j] +=
// bias[j]).
//
// Returns true and sets m_has_bias/m_bias_md/m_bias_mem when the bias maps
// natively. Returns false (after markNotSupported) for sub-cases oneDNN cannot
// express without emulation:
//   - bias with a scale-factor / zero-point: REF dequantizes the bias as
//     (bias - zp) * scale before adding; DNNL_ARG_BIAS has no dequant, and
//     pre-computing the dequant on the host would be emulation of the operand
//     under test, so it is deferred instead.
//   - bias not first in the chain: oneDNN always applies bias before every
//     post-op, so a non-leading bias would reorder the pipeline vs REF.
//   - more than one bias: oneDNN exposes a single bias argument.
bool
OneDNNUalPlan::stageBias()
{
    const auto& post_ops = getPostOps();

    // Locate the (single) bias entry and enforce chain-position invariants.
    const BiasParam* bias_param = nullptr;
    size_t           bias_index = 0;
    size_t           idx        = 0;
    for (const auto& param : post_ops) {
        if (param->getType() == OperationType::Bias) {
            if (bias_param != nullptr) {
                markNotSupported("oneDNN plan: multiple bias post-ops have no "
                                 "native mapping (single DNNL_ARG_BIAS)");
                return false;
            }
            bias_param = static_cast<const BiasParam*>(param.get());
            bias_index = idx;
        }
        ++idx;
    }

    if (bias_param == nullptr) {
        return false; // no bias in this plan
    }

    // oneDNN applies bias before every post-op; REF applies ops in chain order.
    // They only agree when bias is first.
    if (bias_index != 0) {
        markNotSupported("oneDNN plan: bias is not the first post-op; oneDNN "
                         "applies bias before all post-ops (order mismatch)");
        return false;
    }

    // Bias-with-SF/ZP is a dequantization of the bias operand ((bias - zp) *
    // scale in REF). oneDNN's bias argument adds the buffer verbatim, so there
    // is no native expression — defer rather than emulate on the host.
    if (bias_param->hasScaleFactor() || bias_param->hasZeroPoint()) {
        markNotSupported(
            "oneDNN plan: bias scale-factor/zero-point dequant has "
            "no native oneDNN bias mapping");
        return false;
    }

    const Matrix& bias     = bias_param->getBias();
    md_t          bias_len = bias.getRows() * bias.getCols();
    if (bias_len <= 0) {
        markNotSupported("oneDNN plan: empty bias vector");
        return false;
    }

    // REF broadcasts bias by column as bias[j % bias_size]. oneDNN broadcasts a
    // length-N bias across M rows. These agree only when the bias length equals
    // N (the common test case: one bias element per output column). A shorter
    // bias with modulo wrap-around is not expressible natively.
    if (bias_len != m_n) {
        markNotSupported("oneDNN plan: bias length != N has no native oneDNN "
                         "broadcast (REF uses bias[j % len])");
        return false;
    }

    // Build a 2D bias md {1, N} (broadcast over M) in f32, matching oneDNN's
    // matmul bias example (bias ndims == dst ndims, broadcast dims == 1).
    dnnl::memory::dims bias_dims = { 1, static_cast<dnnl::memory::dim>(m_n) };
    m_bias_md = dnnl::memory::desc(bias_dims, dnnl::memory::data_type::f32,
                                   dnnl::memory::format_tag::ab);

    // Own the bias storage: a dnnl::memory constructed with the engine (no
    // external handle) allocates and owns its buffer, so the staged data
    // outlives prepare() and stays valid until the plan is destroyed.
    m_bias_mem = dnnl::memory(m_bias_md, m_ual.m_engine);
    float* dst = static_cast<float*>(m_bias_mem.get_data_handle());
    for (md_t i = 0; i < bias_len; ++i) {
        dst[i] = dlp::testing::utils::convertTo<float>(
            bias.getData(), bias.getMatrixType(), static_cast<size_t>(i));
    }

    m_has_bias = true;
    return true;
}

void
OneDNNUalPlan::prepare()
{
    if (m_prepared) {
        return;
    }

    // Quant families have no valid native oneDNN mapping (confirmed by
    // feasibility probing: oneDNN has no float-src x int8-weights matmul path,
    // and the float->s8 round/clamp of A is the exact behaviour under test, so
    // emulating it would defeat the independent-oracle property). Defer the
    // rejection to execute() as UAL_NOT_SUPPORTED so the harness SKIPS the case
    // instead of scoring an uncaught throw as a FAILURE.
    if (hasAQuant() || hasBQuant()) {
        markNotSupported(
            "oneDNN plan: A/B quantization has no native oneDNN mapping");
        m_prepared = true;
        return;
    }
    if (hasGroupScale()) {
        markNotSupported(
            "oneDNN plan: GroupScale has no native oneDNN mapping");
        m_prepared = true;
        return;
    }
    if (hasWOQ()) {
        // Weight-only quant (bf16 x s4/u4): the s4/u4 unpack + dequant is the
        // behaviour under test and has no native oneDNN matmul path. Defer.
        // (In practice these are reorderB configs skipped earlier at the
        // reorder stage; this guard makes the intent explicit and covers any
        // non-reorder WOQ config that might be added later.)
        markNotSupported("oneDNN plan: WOQ has no native oneDNN mapping");
        m_prepared = true;
        return;
    }

    // Some post-op types have no native oneDNN mapping yet. Defer them as
    // NOT_SUPPORTED (harness SKIPS) rather than throwing from buildPostOpChain
    // (an uncaught throw from prepare() is scored as a FAILURE):
    //   - GLU: terminal, shape-changing (2I->I fold); no native equivalent.
    //   - MatAdd / MatMul: binary-tensor post-ops not yet implemented.
    for (const auto& param : getPostOps()) {
        switch (param->getType()) {
            case OperationType::GLU:
                markNotSupported(
                    "oneDNN plan: GLU has no native oneDNN mapping");
                m_prepared = true;
                return;
            case OperationType::MatAdd:
                markNotSupported(
                    "oneDNN plan: MatAdd post-op not yet implemented");
                m_prepared = true;
                return;
            case OperationType::MatMul:
                markNotSupported(
                    "oneDNN plan: MatMul post-op not yet implemented");
                m_prepared = true;
                return;
            default:
                break;
        }
    }

    // Validate types. An unsupported dtype (e.g. s4/u4) is a "no native
    // mapping" finding, not a test failure: defer to a NOT_SUPPORTED skip
    // rather than throwing out of prepare() (which the harness scores FAILURE).
    if (!UalOneDNN::isTypeSupported(m_a_type)
        || !UalOneDNN::isTypeSupported(m_b_type)
        || !UalOneDNN::isTypeSupported(m_c_type)) {
        markNotSupported("oneDNN plan: unsupported matrix dtype");
        m_prepared = true;
        return;
    }

    // acc_type=fp16 means DLP/REF perform NATIVE fp16 accumulation: every FMA
    // in the micro-kernel rounds to fp16 (see tests/adaptors/ref/ual_ref.cc,
    // the f16f16f16of16 / f16f16f16of32 cases). oneDNN's matmul primitive has
    // no public control to replicate that bit-for-bit; it accumulates
    // internally (typically f32) and only rounds once at the dst boundary.
    // That is numerically different arithmetic, not the same op at lower
    // precision, so treat it as "no native mapping" rather than silently
    // comparing/benchmarking against a different accumulation path.
    if (m_acc_type == MatrixType::fp16) {
        markNotSupported(
            "oneDNN plan: native FP16 accumulation has no oneDNN mapping "
            "(oneDNN matmul does not expose fp16-rounded internal FMA)");
        m_prepared = true;
        return;
    }

    // u8/s8 x s8 GEMM with s32 accumulation on a pre-AVX2-VNNI CPU runs
    // oneDNN's "gemm:jit" AVX2 fallback, whose VPMADDUBSW-based inner loop
    // SATURATES its 16-bit intermediate instead of widening to int32 (see
    // isSaturatingAvx2IntGemm()). This is data-dependent wrongness we
    // cannot bound (prepare() never sees the actual matrix contents), so
    // reject unconditionally rather than risk silently returning a
    // saturated result while reporting UAL_SUCCESS.
    if (UalOneDNN::isSaturatingAvx2IntGemm(m_a_type, m_b_type, m_acc_type)) {
        markNotSupported(
            "oneDNN plan: u8/s8 x s8 GEMM with s32 accumulation on this CPU "
            "(no AVX2-VNNI or better) uses a saturating AVX2 integer "
            "intermediate and is not an exact oracle");
        m_prepared = true;
        return;
    }

    // Stage bias BEFORE constructing the matmul PD: in oneDNN bias is a PD
    // constructor argument, not a post_op. stageBias() sets
    // m_has_bias/m_bias_md for the native case and defers unsupported sub-cases
    // via markNotSupported() (harness SKIPS at execute()).
    if (hasPostOps()) {
        for (const auto& param : getPostOps()) {
            if (param->getType() == OperationType::Bias) {
                stageBias();
                break;
            }
        }
        if (m_not_supported) {
            m_prepared = true;
            return;
        }
    }

    // DLP/REF truncate alpha/beta to int32_t for s32-accumulation kernels
    // (see e.g. tests/adaptors/ref/ual_ref.cc's u8s8s32os32 case) — the
    // native AOCL integer-GEMM API takes integer coefficients. oneDNN has
    // no such notion and would apply the fractional value as a real
    // floating-point scale, diverging numerically (e.g. alpha=beta=0.5
    // truncates to 0 for DLP/REF but stays 0.5 for an unnormalized oneDNN
    // call). Normalize before anything below consumes m_alpha/m_beta.
    if (m_acc_type == MatrixType::s32) {
        m_alpha = static_cast<double>(static_cast<int32_t>(m_alpha));
        m_beta  = static_cast<double>(static_cast<int32_t>(m_beta));

        // DLP/REF apply these as exact int32_t arithmetic (see
        // isExactlyRepresentableAsFloat()'s doc comment). Everything
        // below hands alpha/beta to oneDNN as an f32 scale / sum post-op
        // coefficient, which cannot exactly represent every int32_t --
        // reject before that silent precision loss happens.
        if (!UalOneDNN::isExactlyRepresentableAsFloat(
                static_cast<int32_t>(m_alpha))
            || !UalOneDNN::isExactlyRepresentableAsFloat(
                static_cast<int32_t>(m_beta))) {
            markNotSupported(
                "oneDNN plan: alpha/beta coefficient for s32 accumulation "
                "cannot be represented exactly as f32");
            m_prepared = true;
            return;
        }

        // For s32-accumulation with a floating-point output (e.g.
        // u8s8s32of32), REF rounds C into the s32 accumulator (std::rint,
        // matching the native u8s8/s8s8 quantization contract -- see
        // createAndInitializeTempMatrix() in tests/adaptors/ref/ual_ref.cc)
        // BEFORE applying beta. oneDNN's beta sum post-op has no such
        // rounding step -- it adds beta * the raw fractional dst value
        // as-is. With a fractional C and beta != 0 that diverges from REF
        // (e.g. C=0.75, alpha=beta=1 -> REF rounds C to 1 first giving 2,
        // oneDNN's sum post-op gives 1.75). Integer C types (s8/u8/s32/...)
        // are unaffected since they hold no fractional value to round.
        // Defer rather than silently disagreeing.
        if (m_beta != 0.0
            && (m_c_type == MatrixType::f32 || m_c_type == MatrixType::bf16
                || m_c_type == MatrixType::fp16)) {
            markNotSupported(
                "oneDNN plan: fractional-C rounding for s32 accumulation "
                "with floating-point output is not implemented");
            m_prepared = true;
            return;
        }
    }

    // BLAS contract: alpha == 0 means the result must be independent of A
    // and B (REF explicitly skips the A@B accumulation to honor this — see
    // gemm_f32f32f32of32_ref.cc). A runtime SRC scale of 0 does NOT give the
    // same guarantee: oneDNN's matmul still computes the real A@B product
    // and only scales it afterward, so e.g. a NaN in A propagates through
    // (0 * NaN == NaN, not 0) instead of being masked out. Defer rather than
    // executing a scaled matmul that can disagree with REF on NaN/Inf inputs.
    if (m_alpha == 0.0) {
        markNotSupported("oneDNN plan: alpha-zero execution without reading "
                         "A/B is not implemented");
        m_prepared = true;
        return;
    }

    // m_memFormatA == 'r' means A was already reordered/blocked into some
    // backend-specific internal layout (e.g. by a prior reorder() call) --
    // it is NOT a plain row/col-major strided buffer, so m_lda is
    // meaningless for it. Every a_md built below (in every branch: blocked
    // B, cached primitive, etc.) unconditionally wraps A in a plain
    // strided descriptor via createMemoryDesc(..., m_a_layout, m_transA,
    // m_lda), which would misinterpret a reordered buffer's actual bytes
    // as plain strided data -- computing a wrong result while still
    // returning UAL_SUCCESS. The raw-pointer gemm() overload already
    // rejects this same case (see suggestion 32); mirror it here until a
    // source-format (blocked-A) descriptor path is implemented.
    if (m_memFormatA == 'r') {
        markNotSupported("oneDNN plan: reordered A is not supported");
        m_prepared = true;
        return;
    }

    m_b_blocked        = (m_memFormatB == 'r');
    m_has_alpha_scale  = (m_alpha != 1.0);
    m_alpha_val        = static_cast<float>(m_alpha);
    m_uses_postop_attr = hasPostOps();

    bool has_beta_sum = (m_beta != 0.0);

    dnnl::memory::desc a_md;
    dnnl::memory::desc b_md;
    dnnl::memory::desc c_md;

    if (m_b_blocked) {
        // Blocked B path: look up / create format descriptor
        UalOneDNN::FormatKey fmt_key;
        fmt_key.k        = m_k;
        fmt_key.n        = m_n;
        fmt_key.m        = m_m;
        fmt_key.a_type   = m_a_type;
        fmt_key.b_type   = m_b_type;
        fmt_key.c_type   = m_c_type;
        fmt_key.trans_a  = m_transA;
        fmt_key.trans_b  = m_transB;
        fmt_key.b_layout = m_b_layout;

        auto fmt_it = m_ual.m_weightFormatCache.find(fmt_key);
        if (fmt_it == m_ual.m_weightFormatCache.end()) {
            auto user_a_md = UalOneDNN::createMemoryDesc(
                m_m, m_k, m_a_type, m_a_layout, m_transA, m_lda);

            dnnl::memory::dims weights_dims   = { static_cast<int64_t>(m_k),
                                                  static_cast<int64_t>(m_n) };
            auto               any_weights_md = dnnl::memory::desc(
                weights_dims, UalOneDNN::toOneDNNType(m_b_type),
                dnnl::memory::format_tag::any);

            c_md = UalOneDNN::createMemoryDesc(m_m, m_n, m_c_type, m_c_layout,
                                               false, m_ldc);

            // The format-query PD can also be rejected by oneDNN for an
            // unsupported dtype/ISA combo; defer to a NOT_SUPPORTED skip like
            // every other PD-build site rather than throwing out of prepare().
            try {
                auto query_pd = dnnl::matmul::primitive_desc(
                    m_ual.m_engine, user_a_md, any_weights_md, c_md);
                m_ual.m_weightFormatCache[fmt_key] = query_pd.weights_desc();
            } catch (const dnnl::error& e) {
                markNotSupported(
                    std::string("oneDNN plan: blocked-B format query not "
                                "supported for this dtype/ISA combo: ")
                    + e.message);
                m_prepared = true;
                return;
            }
            fmt_it = m_ual.m_weightFormatCache.find(fmt_key);
        }

        // Wrap the full build (descs + post-op chain + PD) in one try/catch.
        // createMemoryDesc() can throw for strided col-major shapes oneDNN
        // rejects, and buildPostOpChain() operand descs can too; both are
        // "no native mapping" findings → deferred SKIP, not a FAILURE.
        try {
            a_md = UalOneDNN::createMemoryDesc(m_m, m_k, m_a_type, m_a_layout,
                                               m_transA, m_lda);
            b_md = fmt_it->second;
            c_md = UalOneDNN::createMemoryDesc(m_m, m_n, m_c_type, m_c_layout,
                                               false, m_ldc);

            // Create primitive with alpha/beta attrs
            dnnl::primitive_attr attr;
            if (m_has_alpha_scale) {
                attr.set_scales_mask(DNNL_ARG_SRC, 0);
            }
            if (m_uses_postop_attr) {
                // Post-ops present: build the full chain (beta sum + fusion
                // ops) through the centralised dispatch point.
                dnnl::post_ops po;
                buildPostOpChain(po, attr);
                if (m_not_supported) {
                    m_prepared = true;
                    return;
                }
            } else if (has_beta_sum) {
                dnnl::post_ops po;
                po.append_sum(static_cast<float>(m_beta));
                attr.set_post_ops(po);
            }

            // Build the primitive. Bias (blocked-B path): select the
            // bias-carrying PD overload — in oneDNN bias is a PD constructor
            // argument, applied before post-ops.
            auto pd =
                m_has_bias
                    ? dnnl::matmul::primitive_desc(m_ual.m_engine, a_md, b_md,
                                                   m_bias_md, c_md, attr)
                    : dnnl::matmul::primitive_desc(m_ual.m_engine, a_md, b_md,
                                                   c_md, attr);
            m_prim = dnnl::matmul(pd);
        } catch (const dnnl::error& e) {
            markNotSupported(
                std::string("oneDNN plan: blocked-B path rejected during "
                            "build: ")
                + e.message);
            m_prepared = true;
            return;
        }

    } else if (m_uses_postop_attr) {
        // Standard path WITH post-ops: build the primitive fresh (the cache key
        // does not capture post-op identity, so it must be bypassed).
        //
        // The ENTIRE build is wrapped in try/catch: not just the PD build but
        // also createMemoryDesc() and buildPostOpChain(). oneDNN's
        // memory::desc(dims,type,strides) legitimately throws for some strided
        // col-major shapes (the no-post-op cached path proves these are real
        // NOT_SUPPORTED cases), and the post-op operand descs built inside
        // buildPostOpChain can throw the same way. Any such rejection is a
        // "no native mapping" finding → deferred SKIP, not a FAILURE.
        try {
            a_md = UalOneDNN::createMemoryDesc(m_m, m_k, m_a_type, m_a_layout,
                                               m_transA, m_lda);
            b_md = UalOneDNN::createMemoryDesc(m_k, m_n, m_b_type, m_b_layout,
                                               m_transB, m_ldb);
            c_md = UalOneDNN::createMemoryDesc(m_m, m_n, m_c_type, m_c_layout,
                                               false, m_ldc);

            dnnl::primitive_attr attr;
            if (m_has_alpha_scale) {
                attr.set_scales_mask(DNNL_ARG_SRC, 0);
            }
            dnnl::post_ops po;
            buildPostOpChain(po, attr);

            // A family seam may have deferred an unmappable config via
            // markNotSupported() (e.g. an SF whose length doesn't match the
            // output axis). Bail before touching oneDNN so execute() SKIPS.
            if (m_not_supported) {
                m_prepared = true;
                return;
            }

            // Build the primitive. Bias (standard path): select the
            // bias-carrying PD overload (in oneDNN bias is a PD constructor
            // argument, applied before post-ops).
            auto pd =
                m_has_bias
                    ? dnnl::matmul::primitive_desc(m_ual.m_engine, a_md, b_md,
                                                   m_bias_md, c_md, attr)
                    : dnnl::matmul::primitive_desc(m_ual.m_engine, a_md, b_md,
                                                   c_md, attr);
            m_prim = dnnl::matmul(pd);
        } catch (const dnnl::error& e) {
            markNotSupported(
                std::string("oneDNN plan: post-op path rejected during build: ")
                + e.message);
            m_prepared = true;
            return;
        }

    } else {
        // Standard path: use primitive caching (no post-ops)
        UalOneDNN::PrimitiveKey key;
        key.m        = m_m;
        key.n        = m_n;
        key.k        = m_k;
        key.a_type   = m_a_type;
        key.b_type   = m_b_type;
        key.c_type   = m_c_type;
        key.trans_a  = m_transA;
        key.trans_b  = m_transB;
        key.lda      = m_lda;
        key.ldb      = m_ldb;
        key.ldc      = m_ldc;
        key.alpha    = m_alpha_val;
        key.beta     = static_cast<float>(m_beta);
        key.a_layout = m_a_layout;
        key.b_layout = m_b_layout;
        key.c_layout = m_c_layout;

        // oneDNN may reject a dtype/ISA combo at PD build time (e.g. a bf16
        // matmul on a host without AVX512_BF16). Defer that to execute() as
        // NOT_SUPPORTED so the harness SKIPS — matching how the DLP backend
        // skips unsupported-ISA configs — rather than letting the throw escape
        // prepare() and score as a FAILURE. Unexpected errors are rethrown.
        try {
            auto& cached = m_ual.getOrCreatePrimitive(key);
            m_prim       = cached.prim;
            a_md         = cached.a_md;
            b_md         = cached.b_md;
            c_md         = cached.c_md;
        } catch (const dnnl::error& e) {
            markNotSupported(
                std::string("oneDNN plan: matmul not supported for this "
                            "dtype/ISA combo: ")
                + e.message);
            m_prepared = true;
            return;
        }
    }

    // Pre-build memory objects with nullptr data handles.
    // execute() will swap in real pointers via set_data_handle().
    m_a_mem = dnnl::memory(a_md, m_ual.m_engine, nullptr);
    m_b_mem = dnnl::memory(b_md, m_ual.m_engine, nullptr);
    m_c_mem = dnnl::memory(c_md, m_ual.m_engine, nullptr);

    // Pre-build exec_args once -- no allocation in execute().
    // NOTE: buildPostOpChain() may have already registered binary post-op
    // operands (scale SF / ZP) into m_exec_args under
    // DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1. Insert the base
    // src/weights/dst args WITHOUT clobbering those entries.
    m_exec_args[DNNL_ARG_SRC]     = m_a_mem;
    m_exec_args[DNNL_ARG_WEIGHTS] = m_b_mem;
    m_exec_args[DNNL_ARG_DST]     = m_c_mem;

    if (m_has_alpha_scale) {
        m_alpha_mem                                      = dnnl::memory({ { 1 },
                                                                          dnnl::memory::data_type::f32,
                                                                          dnnl::memory::format_tag::x },
                                                                        m_ual.m_engine, &m_alpha_val);
        m_exec_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = m_alpha_mem;
    }

    // Bias buffer is staged in owned storage during stageBias(); register it
    // once so execute() stays allocation-free.
    if (m_has_bias) {
        m_exec_args[DNNL_ARG_BIAS] = m_bias_mem;
    }

    m_prepared = true;
}

UALError
OneDNNUalPlan::execute()
{
    if (!m_prepared || !m_buffers_set) {
        return UALError::UAL_FAILURE;
    }

    // Configuration has no valid native oneDNN mapping (recorded in prepare()).
    // Report NOT_SUPPORTED so the harness SKIPS rather than fails this case.
    if (m_not_supported) {
        return UALError::UAL_NOT_SUPPORTED;
    }

    // setBuffers() is allowed to rebind to a different buffer after
    // prepare() without re-preparing (see IUalPlan::setBuffers() docs), but
    // it only captures new leading dimensions into m_buf_lda/ldb/ldc -- it
    // does not rebuild anything. Unlike DLP/REF (which read lda/ldb/ldc
    // fresh at execute time), oneDNN's dnnl::memory::desc objects have
    // their strides baked in at prepare()-time construction, and
    // set_data_handle() below only swaps the raw pointer, not the
    // descriptor. Rebinding to a same-shaped buffer with a different
    // stride would therefore read/write under the stale stride (padding,
    // wrong rows, or OOB on a smaller buffer) while still reporting
    // success. Reject rather than silently computing something wrong.
    // Blocked B has no nominal ldb (it's packed into oneDNN's internal
    // blocked layout via format_tag::any), so its leading dimension is
    // not meaningful here and is excluded from the check.
    //
    // configureFrom() only forwards A/B's transpose flags (setTranspose()
    // takes no C argument), so a transposed C is silently dropped at
    // configuration time, and every c_md built in prepare() hardcodes
    // transposed=false -- exactly like the raw-pointer gemm() overload,
    // which explicitly rejects matC_transposed for the same reason. Without
    // this check here too, binding a transposed C via setBuffers(const
    // Matrix&, ...) (which does carry m_c_matrix/isTransposed()) builds a
    // descriptor that treats ldc as a normal (not transposed) stride: e.g.
    // M=3,N=2 with a transposed row-major C physically shaped 2x3/ld=3 (6
    // elements) would compute row offsets assuming 3 rows of stride ldc,
    // writing up to offset 2*ldc+1=7 -- out-of-bounds past the 6-element
    // buffer. m_c_matrix is null when the raw-pointer setBuffers() overload
    // was used instead, which carries no transpose info to check here.
    if ((m_c_matrix && m_c_matrix->isTransposed()) || m_buf_lda != m_lda
        || (!m_b_blocked && m_buf_ldb != m_ldb) || m_buf_ldc != m_ldc) {
        return UALError::UAL_NOT_SUPPORTED;
    }

    // The stride checks above only catch a changed leading dimension --
    // they say nothing about shape, element type, or layout. setBuffers()
    // lets a caller rebind to an entirely different Matrix, and two
    // differently-SHAPED matrices can coincidentally share the same
    // leading dimension (e.g. a smaller C[1,2] vs the C[2,2] this plan was
    // prepare()'d for, both row-major with ldc=2): the stride check alone
    // would pass, then set_data_handle() below points the ORIGINAL (larger)
    // a_md/b_md/c_md descriptors at the new (possibly smaller, differently
    // typed, or differently laid out) buffer -- the primitive would then
    // write past a too-small allocation or misinterpret the bytes. Validate
    // each bound Matrix's effective shape/type/layout against what this
    // plan was actually prepared for. m_a_matrix/m_b_matrix/m_c_matrix are
    // null when the raw-pointer setBuffers() overload was used instead,
    // which carries no Matrix to validate here (same reasoning as the
    // transpose-C check above).
    auto matches = [](const Matrix* matrix, md_t rows, md_t cols,
                      MatrixType type, MatrixLayout layout) {
        return !matrix
               || (matrix->getEffectiveRows() == rows
                   && matrix->getEffectiveCols() == cols
                   && matrix->getMatrixType() == type
                   && matrix->getLayout() == layout);
    };
    //
    // matches() alone is not enough: it compares effective rows/cols,
    // which already fold in each matrix's OWN transposed flag
    // (getEffectiveRows()/getEffectiveCols() -- see matrix.cc), so a square
    // A (or a plain/non-blocked B) rebound with a FLIPPED transpose flag
    // still reports the same effective shape/type/layout and would slip
    // through. The prepared a_md/b_md descriptors have m_transA/m_transB
    // baked in at prepare()-time and are not rebuilt here, so such a
    // mismatch would execute() against the wrong operand orientation
    // while still reporting success. Blocked B is excluded for the same
    // reason its ldb is excluded from the stride check above: it has no
    // meaningful user-facing transpose once packed into oneDNN's internal
    // blocked layout (trans_b is already baked into the cached blocked
    // descriptor's format key).
    //
    // matches() also says nothing about isReordered(): reorder()'s output
    // Matrix keeps the ORIGINAL logical rows/cols (only the underlying
    // bytes become oneDNN's packed blocked format -- see reorder()'s
    // construction of `out`), so a reordered Matrix passes matches() just
    // as readily as a plain one of the same shape/type/layout. m_b_blocked
    // is frozen from whatever B looked like at prepare()-time and is never
    // re-derived here, so rebinding to a B whose isReordered() disagrees
    // with m_b_blocked (either direction) would bind the wrong descriptor
    // (plain strided vs oneDNN's internal blocked layout) to bytes it
    // doesn't actually describe. A is never allowed to be reordered at
    // execute() at all -- prepare() already rejects m_memFormatA == 'r'
    // for the prepare()-time A, but that check never re-runs if
    // setBuffers() later rebinds to a different (reordered) A -- so check
    // isReordered() here too rather than letting a rebind bypass it.
    if (!matches(m_a_matrix, m_m, m_k, m_a_type, m_a_layout)
        || !matches(m_b_matrix, m_k, m_n, m_b_type, m_b_layout)
        || !matches(m_c_matrix, m_m, m_n, m_c_type, m_c_layout)
        || (m_a_matrix
            && (m_a_matrix->isTransposed() != m_transA
                || m_a_matrix->isReordered()))
        || (m_b_matrix && m_b_matrix->isReordered() != m_b_blocked)
        || (!m_b_blocked && m_b_matrix
            && m_b_matrix->isTransposed() != m_transB)) {
        return UALError::UAL_FAILURE;
    }

    // A blocked (already-reordered) B's bytes were packed by a PRIOR
    // reorder() call using oneDNN's format_tag::any, which chooses its
    // "optimal" blocking as a function of parameters EXTERNAL to B itself
    // (the GEMM's M, A's transpose flag, C's type) -- not just B's own
    // K x N shape/type/layout, which is all `matches()` above can check.
    // m_b_md (built in prepare() via the shape-keyed FormatKey cache) is
    // only guaranteed to describe B's ACTUAL bytes when every one of those
    // external parameters at reorder()-time matches this plan's own
    // configured values -- which m_weightFormatCache's key cannot verify,
    // since it has no idea what reorder() actually produced versus what
    // prepare() is merely assuming/re-deriving. Without this check, a B
    // reordered for a different M (e.g. omitted-M defaulting to 256, then
    // reused at a smaller M) silently binds the wrong-shaped descriptor to
    // real bytes: wrong results at best, and reading past the smaller
    // original allocation (OOB) when the mismatched descriptor is larger.
    // Reject rather than trusting bytes we cannot verify the format of.
    if (m_b_blocked
        && (!m_b_matrix || m_b_matrix->getReorderContextM() != m_m
            || m_b_matrix->getReorderContextTransA() != m_transA
            || m_b_matrix->getReorderContextCType() != m_c_type)) {
        return UALError::UAL_NOT_SUPPORTED;
    }

    // Swap data handles -- cheap pointer assignment, no allocation
    m_a_mem.set_data_handle(m_a_ptr);
    m_b_mem.set_data_handle(m_b_ptr);
    m_c_mem.set_data_handle(m_c_ptr);

    m_prim.execute(m_ual.m_stream, m_exec_args);
    m_ual.m_stream.wait();

    return UALError::UAL_SUCCESS;
}

// ============================================================================
// GEMM with Raw Pointers (Benchmarking Path)
// ============================================================================

UALError
UalOneDNN::gemm(md_t         m,
                md_t         n,
                md_t         k,
                void*        matA,
                MatrixType   matA_type,
                MatrixLayout matA_layout,
                bool         matA_transposed,
                char         memFormatA,
                md_t         matA_leadingDim,
                void*        matB,
                MatrixType   matB_type,
                MatrixLayout matB_layout,
                bool         matB_transposed,
                char         memFormatB,
                md_t         matB_leadingDim,
                void*        matC,
                MatrixType   matC_type,
                MatrixLayout matC_layout,
                bool         matC_transposed,
                md_t         matC_leadingDim,
                MatrixType   accType,
                double       alpha,
                double       beta)
{
    try {
        // acc_type=fp16 requires DLP/REF's native fp16-rounded FMA
        // accumulation, which oneDNN's matmul has no way to replicate (it
        // accumulates internally, typically in f32, and rounds once at the
        // dst boundary). That is different arithmetic, not an emulation of
        // it, so reject rather than silently benchmark/compare against it.
        if (accType == MatrixType::fp16) {
            return UALError::UAL_NOT_SUPPORTED;
        }

        // Validate types
        if (!isTypeSupported(matA_type) || !isTypeSupported(matB_type)
            || !isTypeSupported(matC_type)) {
            return UALError::UAL_NOT_SUPPORTED;
        }

        // See isSaturatingAvx2IntGemm(): without AVX2-VNNI or better,
        // oneDNN's u8/s8 x s8 int8 GEMM kernel saturates its 16-bit
        // VPMADDUBSW intermediate instead of widening to int32. Reject
        // unconditionally since this call site has no way to bound the
        // actual data values.
        if (isSaturatingAvx2IntGemm(matA_type, matB_type, accType)) {
            return UALError::UAL_NOT_SUPPORTED;
        }

        // DLP/REF truncate alpha/beta to int32_t for s32-accumulation
        // kernels (see e.g. tests/adaptors/ref/ual_ref.cc's u8s8s32os32
        // case). Normalize here, before alpha/beta are used below, so this
        // raw-pointer path matches that integer-coefficient semantic
        // instead of applying the fractional value as a real float scale.
        if (accType == MatrixType::s32) {
            alpha = static_cast<double>(static_cast<int32_t>(alpha));
            beta  = static_cast<double>(static_cast<int32_t>(beta));

            // See isExactlyRepresentableAsFloat()'s doc comment: REF
            // applies alpha/beta as exact int32_t arithmetic, but
            // everything below hands them to oneDNN as an f32 coefficient,
            // which cannot exactly represent every int32_t value. Reject
            // before that silent precision loss happens.
            if (!isExactlyRepresentableAsFloat(static_cast<int32_t>(alpha))
                || !isExactlyRepresentableAsFloat(static_cast<int32_t>(beta))) {
                return UALError::UAL_NOT_SUPPORTED;
            }

            // REF rounds C into the s32 accumulator before applying beta
            // for s32-accumulation with a floating-point output (see
            // OneDNNUalPlan::prepare()'s matching guard for the full
            // rationale); oneDNN's beta sum post-op below adds beta * the
            // raw fractional C as-is, which diverges for a fractional C.
            if (beta != 0.0
                && (matC_type == MatrixType::f32
                    || matC_type == MatrixType::bf16
                    || matC_type == MatrixType::fp16)) {
                return UALError::UAL_NOT_SUPPORTED;
            }
        }

        // BLAS contract: alpha == 0 means the result must be independent of
        // A and B (REF explicitly skips the A@B accumulation to honor this
        // -- see gemm_f32f32f32of32_ref.cc). A runtime SRC scale of 0 does
        // NOT give the same guarantee here: oneDNN's matmul still computes
        // the real A@B product and only scales it afterward, so e.g. a NaN
        // in A propagates through (0 * NaN == NaN, not 0) instead of being
        // masked out, even though beta*C alone would be well-defined.
        // Same guard as OneDNNUalPlan::prepare() (suggestion 16); this
        // raw-pointer path was missing it. Defer rather than executing a
        // scaled matmul that can disagree with REF on NaN/Inf inputs.
        if (alpha == 0.0) {
            return UALError::UAL_NOT_SUPPORTED;
        }

        // C transposed not well supported - return not supported
        if (matC_transposed) {
            return UALError::UAL_NOT_SUPPORTED;
        }

        // Validate leading dimensions
        bool row_major_a = (matA_layout == MatrixLayout::ROW_MAJOR);
        bool row_major_b = (matB_layout == MatrixLayout::ROW_MAJOR);
        bool row_major_c = (matC_layout == MatrixLayout::ROW_MAJOR);

        if (memFormatA != 'r') {
            if (row_major_a) {
                md_t min_lda = matA_transposed ? m : k;
                if (matA_leadingDim < min_lda)
                    return UALError::UAL_FAILURE;
            } else {
                md_t min_lda = matA_transposed ? k : m;
                if (matA_leadingDim < min_lda)
                    return UALError::UAL_FAILURE;
            }
        }

        if (memFormatB != 'r') {
            if (row_major_b) {
                md_t min_ldb = matB_transposed ? k : n;
                if (matB_leadingDim < min_ldb)
                    return UALError::UAL_FAILURE;
            } else {
                md_t min_ldb = matB_transposed ? n : k;
                if (matB_leadingDim < min_ldb)
                    return UALError::UAL_FAILURE;
            }
        }

        if (row_major_c) {
            if (matC_leadingDim < n)
                return UALError::UAL_FAILURE;
        } else {
            if (matC_leadingDim < m)
                return UALError::UAL_FAILURE;
        }

        dnnl::memory a_mem;
        dnnl::memory b_mem;
        dnnl::memory c_mem;
        dnnl::matmul matmul_prim;

        // memFormatA == 'r' means A was already reordered/blocked into some
        // backend-specific internal layout (e.g. by a prior reorder() call)
        // -- it is NOT a plain row/col-major strided buffer, so
        // matA_leadingDim is meaningless for it. That is exactly why the
        // leading-dimension validation above skips matA_leadingDim checks
        // when memFormatA == 'r' (same for B). But the primitive build
        // below unconditionally wraps matA in a *plain* strided descriptor
        // via createMemoryDesc(..., matA_layout, matA_transposed,
        // matA_leadingDim), which would misinterpret a reordered buffer's
        // actual bytes as plain strided data -- computing a wrong result
        // while still returning UAL_SUCCESS. Reject rather than silently
        // discarding the format signal (as the old `(void)memFormatA;` did).
        if (memFormatA == 'r') {
            return UALError::UAL_NOT_SUPPORTED;
        }

        // memFormatB == 'r' means matB's bytes were already packed by a
        // PRIOR reorder() call into oneDNN's format_tag::any "optimal"
        // blocked layout -- a format that is a function of parameters
        // EXTERNAL to B itself (that reorder() call's M, A's transpose,
        // C's type), not just B's own k/n/type/layout. This raw-pointer
        // API only ever receives a bare void*, never the Matrix that
        // reorder() annotated with the M/transA/cType it actually used
        // (see Matrix::setReorderContext()/getReorderContextM() and the
        // equivalent check in OneDNNUalPlan::execute()), so there is no
        // way here to verify whether THIS call's (m, matA_transposed,
        // matC_type) match what matB's bytes were actually packed for.
        // Re-deriving a "guessed" descriptor from this call's own
        // parameters (the previous behaviour) and binding it to matB can
        // silently compute the wrong product, or -- if the guessed
        // descriptor is larger than what matB was actually packed as --
        // read past matB's real allocation. Reject rather than trusting
        // bytes whose format cannot be verified from this call site.
        if (memFormatB == 'r') {
            return UALError::UAL_NOT_SUPPORTED;
        }

        {
            PrimitiveKey key;
            key.m        = m;
            key.n        = n;
            key.k        = k;
            key.a_type   = matA_type;
            key.b_type   = matB_type;
            key.c_type   = matC_type;
            key.trans_a  = matA_transposed;
            key.trans_b  = matB_transposed;
            key.lda      = matA_leadingDim;
            key.ldb      = matB_leadingDim;
            key.ldc      = matC_leadingDim;
            key.alpha    = static_cast<float>(alpha);
            key.beta     = static_cast<float>(beta);
            key.a_layout = matA_layout;
            key.b_layout = matB_layout;
            key.c_layout = matC_layout;

            auto& cached = getOrCreatePrimitive(key);
            matmul_prim  = cached.prim;

            a_mem = dnnl::memory(cached.a_md, m_engine, matA);
            b_mem = dnnl::memory(cached.b_md, m_engine, matB);
            c_mem = dnnl::memory(cached.c_md, m_engine, matC);
        }

        std::unordered_map<int, dnnl::memory> exec_args = {
            { DNNL_ARG_SRC, a_mem },
            { DNNL_ARG_WEIGHTS, b_mem },
            { DNNL_ARG_DST, c_mem }
        };

        // alpha_val must outlive execute()/stream.wait(): dnnl::memory below
        // wraps the raw &alpha_val pointer (no copy), and that pointer is
        // only dereferenced when the primitive actually runs. Declaring it
        // in the enclosing scope (not inside the `if`) keeps it alive long
        // enough instead of dangling by the time execute() reads it.
        float        alpha_val = static_cast<float>(alpha);
        dnnl::memory alpha_mem;
        if (alpha != 1.0) {
            alpha_mem = dnnl::memory({ { 1 },
                                       dnnl::memory::data_type::f32,
                                       dnnl::memory::format_tag::x },
                                     m_engine, &alpha_val);
            exec_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = alpha_mem;
        }

        matmul_prim.execute(m_stream, exec_args);
        m_stream.wait();

        return UALError::UAL_SUCCESS;

    } catch (const dnnl::error& e) {
        if (e.status == dnnl_unimplemented) {
            return UALError::UAL_NOT_SUPPORTED;
        }
        return UALError::UAL_FAILURE;
    } catch (const std::exception& e) {
        return UALError::UAL_FAILURE;
    }
}

// ============================================================================
// Batch GEMM Helpers
// ============================================================================

namespace {

    size_t getElementSize(MatrixType type)
    {
        switch (type) {
            case MatrixType::f32:
                return sizeof(float);
            case MatrixType::bf16:
            case MatrixType::fp16:
                return 2;
            case MatrixType::s8:
            case MatrixType::u8:
                return 1;
            case MatrixType::s32:
                return sizeof(int32_t);
            default:
                return 0;
        }
    }

    size_t computePhysicalElements(
        md_t rows, md_t cols, md_t ld, MatrixLayout layout, bool transposed)
    {
        if (layout == MatrixLayout::ROW_MAJOR) {
            return transposed ? static_cast<size_t>(cols) * ld
                              : static_cast<size_t>(rows) * ld;
        } else {
            return transposed ? static_cast<size_t>(rows) * ld
                              : static_cast<size_t>(cols) * ld;
        }
    }

    dnnl::memory::desc create3DMemoryDesc(md_t                    batch,
                                          md_t                    rows,
                                          md_t                    cols,
                                          dnnl::memory::data_type dnnl_type,
                                          MatrixLayout            layout,
                                          bool                    transposed,
                                          md_t                    leadingDim)
    {
        dnnl::memory::dims dims = { static_cast<dnnl::memory::dim>(batch),
                                    static_cast<dnnl::memory::dim>(rows),
                                    static_cast<dnnl::memory::dim>(cols) };

        auto batch_stride =
            static_cast<dnnl::memory::dim>(computePhysicalElements(
                rows, cols, leadingDim, layout, transposed));

        dnnl::memory::dims strides;
        if (layout == MatrixLayout::ROW_MAJOR) {
            if (transposed) {
                strides = { batch_stride, 1,
                            static_cast<dnnl::memory::dim>(leadingDim) };
            } else {
                strides = { batch_stride,
                            static_cast<dnnl::memory::dim>(leadingDim), 1 };
            }
        } else {
            if (transposed) {
                strides = { batch_stride,
                            static_cast<dnnl::memory::dim>(leadingDim), 1 };
            } else {
                strides = { batch_stride, 1,
                            static_cast<dnnl::memory::dim>(leadingDim) };
            }
        }

        return dnnl::memory::desc(dims, dnnl_type, strides);
    }

    struct CachedBatchGroup
    {
        dnnl::matmul       prim;
        dnnl::memory::desc a_md, b_md, c_md;
        dnnl::memory       a_mem, b_mem, c_mem;
        md_t               batch_size      = 0;
        bool               has_alpha_scale = false;
        float              alpha_val       = 1.0f;
        dnnl::memory       alpha_mem;

        /// Pre-built execution arguments — populated once during
        /// batch_prepare_metadata() so the hot loop does zero allocation.
        std::unordered_map<int, dnnl::memory> exec_args;

        // NOTE: this hot path only supports the zero-copy case (matrices
        // backed by ContiguousBatchStorage or otherwise already densely
        // packed). A "copy fallback" for scattered matrices previously lived
        // here but was removed: batch_gemm(prepared) has no gather-before /
        // scatter-after step, so it could never correctly read fresh inputs
        // or publish results back to the caller's C matrices for that case.
        // Scattered matrices must use
        // batch_gemm(std::vector<BatchGroup>&, MatrixType) instead, which
        // already gathers/scatters correctly. batch_prepare_metadata()
        // rejects (null metadata) any group that isn't fully contiguous.
    };

    /// Check whether @p count pointers are evenly spaced by @p stride bytes.
    /// Returns true when the pointers form a dense contiguous layout suitable
    /// for a oneDNN 3D memory descriptor.
    static bool isContiguousLayout(const void* const* ptrs,
                                   md_t               count,
                                   size_t             stride)
    {
        if (count <= 1) {
            return true;
        }
        auto base = reinterpret_cast<uintptr_t>(ptrs[0]);
        for (md_t i = 1; i < count; ++i) {
            if (reinterpret_cast<uintptr_t>(ptrs[i]) != base + i * stride) {
                return false;
            }
        }
        return true;
    }

} // anonymous namespace

// ============================================================================
// Batch GEMM – Correctness Path (not performance-critical)
//
// This method handles the general case where matrices within a BatchGroup may
// be individually allocated (scattered memory).  oneDNN's native 3D batched
// matmul requires contiguous [batch, rows, cols] tensors, so when matrices
// are NOT contiguous we allocate temporary contiguous buffers and memcpy the
// data in, execute the batch, then copy results back to the individual C
// matrices.
//
// For performance-critical benchmarking, use the batch_prepare_metadata() +
// batch_gemm(PreparedBatchGemmArgs&) path instead, which detects contiguous
// storage and operates with zero copies.
// ============================================================================

UALError
UalOneDNN::batch_gemm(std::vector<BatchGroup>& groups, MatrixType accType)
{
    using dlp::testing::framework::MatrixMemFormat;

    // Native fp16 accumulation has no oneDNN equivalent (see the matching
    // guard in OneDNNUalPlan::prepare()/gemm() above) — reject consistently
    // here too rather than silently running with oneDNN's default (f32)
    // internal accumulation.
    if (accType == MatrixType::fp16) {
        return UALError::UAL_NOT_SUPPORTED;
    }

    for (auto& group : groups) {
        if (!group.validate()) {
            return UALError::UAL_FAILURE;
        }

        // Reject group attributes this batch path does not implement.
        // a_quant / group_scale are stored separately from post_op_params
        // (see BatchGroup in tests/include/framework/ual.hh) and are NOT
        // consulted anywhere below, so without this guard a quantized group
        // (e.g. native s8xs8 with group_scale) would silently run as a plain
        // unscaled matmul and report UAL_SUCCESS. The single-GEMM plan path
        // (OneDNNUalPlan::prepare(), via hasAQuant()/hasGroupScale()) already
        // defers these to NOT_SUPPORTED; mirror that here until the batch
        // path gains real support.
        if (!group.post_op_params.empty() || group.a_quant
            || group.group_scale) {
            return UALError::UAL_NOT_SUPPORTED;
        }

        md_t batch_size = static_cast<md_t>(group.size());
        if (batch_size == 0) {
            continue;
        }

        // Fall back to individual GEMM loop for non-normal memory formats
        if (group.memFormatA != MatrixMemFormat::Normal
            || group.memFormatB != MatrixMemFormat::Normal) {
            for (size_t i = 0; i < group.size(); ++i) {
                auto plan = createPlan();
                plan->configureFrom(group.A_matrices[i], group.B_matrices[i],
                                    group.C_matrices[i], accType, group.alpha,
                                    group.beta);
                plan->prepare();
                UALError err =
                    plan->executeWith(group.A_matrices[i], group.B_matrices[i],
                                      group.C_matrices[i]);
                if (err != UALError::UAL_SUCCESS) {
                    return err;
                }
            }
            continue;
        }

        const Matrix& A0 = group.A_matrices[0];
        const Matrix& B0 = group.B_matrices[0];
        const Matrix& C0 = group.C_matrices[0];

        MatrixType a_type = A0.getMatrixType();
        MatrixType b_type = B0.getMatrixType();
        MatrixType c_type = C0.getMatrixType();

        if (!isTypeSupported(a_type) || !isTypeSupported(b_type)
            || !isTypeSupported(c_type)) {
            return UALError::UAL_NOT_SUPPORTED;
        }

        // See isSaturatingAvx2IntGemm(): reject u8/s8 x s8 s32-accumulation
        // groups on CPUs without AVX2-VNNI or better, where oneDNN's int8
        // GEMM kernel saturates rather than widening its intermediate.
        if (isSaturatingAvx2IntGemm(a_type, b_type, accType)) {
            return UALError::UAL_NOT_SUPPORTED;
        }

        MatrixLayout a_layout = A0.getLayout();
        MatrixLayout b_layout = B0.getLayout();
        MatrixLayout c_layout = C0.getLayout();

        bool trans_a = A0.isTransposed();
        bool trans_b = B0.isTransposed();

        md_t lda = A0.getLeadingDimension();
        md_t ldb = B0.getLeadingDimension();
        md_t ldc = C0.getLeadingDimension();

        md_t grp_m = group.m;
        md_t grp_n = group.n;
        md_t grp_k = group.k;

        size_t elem_a = getElementSize(a_type);
        size_t elem_b = getElementSize(b_type);
        size_t elem_c = getElementSize(c_type);

        if (elem_a == 0 || elem_b == 0 || elem_c == 0) {
            return UALError::UAL_NOT_SUPPORTED;
        }

        size_t phys_a =
            computePhysicalElements(grp_m, grp_k, lda, a_layout, trans_a)
            * elem_a;
        size_t phys_b =
            computePhysicalElements(grp_k, grp_n, ldb, b_layout, trans_b)
            * elem_b;
        size_t phys_c =
            computePhysicalElements(grp_m, grp_n, ldc, c_layout, false)
            * elem_c;

        try {
            // Allocate contiguous 3D buffers and copy individual matrices in
            std::vector<uint8_t> a_buf(batch_size * phys_a);
            std::vector<uint8_t> b_buf(batch_size * phys_b);
            std::vector<uint8_t> c_buf(batch_size * phys_c);

            for (md_t i = 0; i < batch_size; ++i) {
                // phys_a/b/c above were sized from A0/B0/C0's leading
                // dimension and transpose alone. BatchGroup::validate()
                // checks logical shape/type/layout but NOT leading
                // dimension or transpose, so a later matrix in the group
                // could legally have a different stride (e.g. extra
                // padding) or transpose flag than matrix 0 while still
                // passing validate(). Gathering/scattering it at phys_a/
                // phys_c would then read past its real allocation or
                // overwrite memory beyond what it owns. Reject nonuniform
                // strides/transposes instead of assuming matrix 0's
                // metadata applies to the whole group.
                if (group.A_matrices[i].getLeadingDimension() != lda
                    || group.B_matrices[i].getLeadingDimension() != ldb
                    || group.C_matrices[i].getLeadingDimension() != ldc
                    || group.A_matrices[i].isTransposed() != trans_a
                    || group.B_matrices[i].isTransposed() != trans_b
                    || group.C_matrices[i].isTransposed()) {
                    return UALError::UAL_NOT_SUPPORTED;
                }
                std::memcpy(a_buf.data() + i * phys_a,
                            group.A_matrices[i].getData(), phys_a);
                std::memcpy(b_buf.data() + i * phys_b,
                            group.B_matrices[i].getData(), phys_b);
                std::memcpy(c_buf.data() + i * phys_c,
                            group.C_matrices[i].getData(), phys_c);
            }

            auto a_dnnl_type = toOneDNNType(a_type);
            auto b_dnnl_type = toOneDNNType(b_type);
            auto c_dnnl_type = toOneDNNType(c_type);

            auto a_md_3d = create3DMemoryDesc(
                batch_size, grp_m, grp_k, a_dnnl_type, a_layout, trans_a, lda);
            auto b_md_3d = create3DMemoryDesc(
                batch_size, grp_k, grp_n, b_dnnl_type, b_layout, trans_b, ldb);
            auto c_md_3d = create3DMemoryDesc(
                batch_size, grp_m, grp_n, c_dnnl_type, c_layout, false, ldc);

            // DLP/REF truncate alpha/beta to int32_t for s32-accumulation
            // kernels (see e.g. tests/adaptors/ref/ual_ref.cc's
            // u8s8s32os32 case) — the native AOCL integer-GEMM API takes
            // integer coefficients. oneDNN has no such notion and would
            // apply the fractional value as a real floating-point scale,
            // diverging numerically (e.g. alpha=beta=0.5 truncates to 0 for
            // DLP/REF but stays 0.5 for an unnormalized oneDNN call).
            // Normalize locally so group.alpha/beta (caller-owned) are left
            // untouched.
            double norm_alpha = group.alpha;
            double norm_beta  = group.beta;
            if (accType == MatrixType::s32) {
                norm_alpha =
                    static_cast<double>(static_cast<int32_t>(norm_alpha));
                norm_beta =
                    static_cast<double>(static_cast<int32_t>(norm_beta));

                // See isExactlyRepresentableAsFloat()'s doc comment: REF
                // applies alpha/beta as exact int32_t arithmetic, but this
                // function hands them to oneDNN as an f32 coefficient
                // below, which cannot exactly represent every int32_t
                // value. Reject before that silent precision loss happens.
                if (!isExactlyRepresentableAsFloat(
                        static_cast<int32_t>(norm_alpha))
                    || !isExactlyRepresentableAsFloat(
                        static_cast<int32_t>(norm_beta))) {
                    return UALError::UAL_NOT_SUPPORTED;
                }

                // REF rounds C into the s32 accumulator before applying
                // beta for s32-accumulation with a floating-point output
                // (see OneDNNUalPlan::prepare()'s matching guard for the
                // full rationale); the beta sum post-op below adds
                // beta * the raw fractional C as-is, which diverges for a
                // fractional C.
                if (norm_beta != 0.0
                    && (c_type == MatrixType::f32 || c_type == MatrixType::bf16
                        || c_type == MatrixType::fp16)) {
                    return UALError::UAL_NOT_SUPPORTED;
                }
            }

            // BLAS contract: alpha == 0 means the result must be
            // independent of A and B. A runtime SRC scale of 0 does not
            // give that guarantee here -- oneDNN still computes the real
            // A@B product and only scales it afterward, so NaN/Inf in A/B
            // would contaminate C instead of yielding beta*C, and this
            // function would scatter that bad result back while still
            // reporting success. Same guard as OneDNNUalPlan::prepare()
            // (suggestion 16) and the raw-pointer gemm() (suggestion 25);
            // this vector batch overload was missing it too.
            if (norm_alpha == 0.0) {
                return UALError::UAL_NOT_SUPPORTED;
            }

            dnnl::primitive_attr attr;
            if (norm_alpha != 1.0) {
                attr.set_scales_mask(DNNL_ARG_SRC, 0);
            }
            if (norm_beta != 0.0) {
                dnnl::post_ops po;
                po.append_sum(static_cast<float>(norm_beta));
                attr.set_post_ops(po);
            }

            auto pd   = dnnl::matmul::primitive_desc(m_engine, a_md_3d, b_md_3d,
                                                     c_md_3d, attr);
            auto prim = dnnl::matmul(pd);

            auto a_mem = dnnl::memory(a_md_3d, m_engine, a_buf.data());
            auto b_mem = dnnl::memory(b_md_3d, m_engine, b_buf.data());
            auto c_mem = dnnl::memory(c_md_3d, m_engine, c_buf.data());

            std::unordered_map<int, dnnl::memory> exec_args = {
                { DNNL_ARG_SRC, a_mem },
                { DNNL_ARG_WEIGHTS, b_mem },
                { DNNL_ARG_DST, c_mem }
            };

            // Same lifetime requirement as UalOneDNN::gemm(): alpha_val must
            // outlive prim.execute()/stream.wait() below, since dnnl::memory
            // wraps the raw &alpha_val pointer rather than copying it.
            float        alpha_val = static_cast<float>(norm_alpha);
            dnnl::memory alpha_mem;
            if (norm_alpha != 1.0) {
                alpha_mem = dnnl::memory({ { 1 },
                                           dnnl::memory::data_type::f32,
                                           dnnl::memory::format_tag::x },
                                         m_engine, &alpha_val);
                exec_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = alpha_mem;
            }

            prim.execute(m_stream, exec_args);
            m_stream.wait();

            // Copy results back to individual C matrices
            for (md_t i = 0; i < batch_size; ++i) {
                std::memcpy(group.C_matrices[i].getData(),
                            c_buf.data() + i * phys_c, phys_c);
            }

        } catch (const dnnl::error& e) {
            if (e.status == dnnl_unimplemented) {
                return UALError::UAL_NOT_SUPPORTED;
            }
            return UALError::UAL_FAILURE;
        } catch (const std::exception&) {
            return UALError::UAL_FAILURE;
        }
    }

    return UALError::UAL_SUCCESS;
}

// ============================================================================
// Batch GEMM Metadata Preparation (One-Time Setup for Benchmarking)
// ============================================================================

void
UalOneDNN::batch_prepare_metadata(PreparedBatchGemmArgs& args)
{
    args.backend_metadata.clear();
    args.backend_metadata_storage.clear();
    m_batchGroupStatus.assign(static_cast<size_t>(args.group_count),
                              UALError::UAL_SUCCESS);

    if (args.alpha_f32.empty()) {
        args.alpha_f32.reserve(args.alpha.size());
        args.beta_f32.reserve(args.beta.size());
        for (size_t i = 0; i < args.alpha.size(); ++i) {
            args.alpha_f32.push_back(static_cast<float>(args.alpha[i]));
            args.beta_f32.push_back(static_cast<float>(args.beta[i]));
        }
    }

    if (!isTypeSupported(args.a_type) || !isTypeSupported(args.b_type)
        || !isTypeSupported(args.c_type)) {
        return;
    }

    // Native fp16 accumulation has no oneDNN equivalent — see the matching
    // guard in OneDNNUalPlan::prepare(). Leave backend_metadata empty so the
    // hot-path batch_gemm(PreparedBatchGemmArgs&) fails fast per group
    // instead of silently running oneDNN's default (f32) accumulation.
    if (args.acc_type == MatrixType::fp16) {
        return;
    }

    // See isSaturatingAvx2IntGemm(): without AVX2-VNNI or better, oneDNN's
    // u8/s8 x s8 int8 GEMM kernel saturates its 16-bit intermediate
    // instead of widening to int32. a_type/b_type/acc_type are fixed for
    // the whole PreparedBatchGemmArgs, so this applies to every group.
    // Mark each group NOT_SUPPORTED explicitly via m_batchGroupStatus
    // (rather than returning with backend_metadata left empty) so
    // batch_gemm(prepared)'s per-group status consultation reports
    // UAL_NOT_SUPPORTED instead of its empty-metadata bounds check
    // surfacing UAL_FAILURE.
    if (isSaturatingAvx2IntGemm(args.a_type, args.b_type, args.acc_type)) {
        for (md_t grp = 0; grp < args.group_count; ++grp) {
            args.backend_metadata_storage.push_back(nullptr);
            args.backend_metadata.push_back(nullptr);
            m_batchGroupStatus[static_cast<size_t>(grp)] =
                UALError::UAL_NOT_SUPPORTED;
        }
        return;
    }

    size_t elem_a = getElementSize(args.a_type);
    size_t elem_b = getElementSize(args.b_type);
    size_t elem_c = getElementSize(args.c_type);

    if (elem_a == 0 || elem_b == 0 || elem_c == 0) {
        return;
    }

    auto a_dnnl_type = toOneDNNType(args.a_type);
    auto b_dnnl_type = toOneDNNType(args.b_type);
    auto c_dnnl_type = toOneDNNType(args.c_type);

    size_t matrix_idx = 0;
    for (md_t grp = 0; grp < args.group_count; ++grp) {
        md_t group_size = args.group_size[grp];

        if (group_size == 0) {
            args.backend_metadata_storage.push_back(nullptr);
            args.backend_metadata.push_back(nullptr);
            matrix_idx += group_size;
            continue;
        }

        bool has_non_normal =
            (args.mem_format_a[grp] != 'n' || args.mem_format_b[grp] != 'n');
        if (has_non_normal) {
            args.backend_metadata_storage.push_back(nullptr);
            args.backend_metadata.push_back(nullptr);
            m_batchGroupStatus[static_cast<size_t>(grp)] =
                UALError::UAL_NOT_SUPPORTED;
            matrix_idx += group_size;
            continue;
        }

        md_t   grp_m     = args.m[grp];
        md_t   grp_n     = args.n[grp];
        md_t   grp_k     = args.k[grp];
        md_t   grp_lda   = args.lda[grp];
        md_t   grp_ldb   = args.ldb[grp];
        md_t   grp_ldc   = args.ldc[grp];
        bool   trans_a   = (args.transa[grp] == 't');
        bool   trans_b   = (args.transb[grp] == 't');
        double grp_alpha = args.alpha[grp];
        double grp_beta  = args.beta[grp];

        // DLP/REF truncate alpha/beta to int32_t for s32-accumulation
        // kernels (see e.g. tests/adaptors/ref/ual_ref.cc's u8s8s32os32
        // case). Normalize here, before they're baked into the cached
        // primitive/scale below, so this benchmarking hot path matches
        // that integer-coefficient semantic instead of applying the
        // fractional value as a real float scale.
        if (args.acc_type == MatrixType::s32) {
            grp_alpha = static_cast<double>(static_cast<int32_t>(grp_alpha));
            grp_beta  = static_cast<double>(static_cast<int32_t>(grp_beta));

            // See isExactlyRepresentableAsFloat()'s doc comment: REF
            // applies alpha/beta as exact int32_t arithmetic, but this
            // group's cached primitive below bakes them in as an f32
            // coefficient, which cannot exactly represent every int32_t
            // value. Skip the group rather than silently losing precision.
            if (!isExactlyRepresentableAsFloat(static_cast<int32_t>(grp_alpha))
                || !isExactlyRepresentableAsFloat(
                    static_cast<int32_t>(grp_beta))) {
                args.backend_metadata_storage.push_back(nullptr);
                args.backend_metadata.push_back(nullptr);
                m_batchGroupStatus[static_cast<size_t>(grp)] =
                    UALError::UAL_NOT_SUPPORTED;
                matrix_idx += group_size;
                continue;
            }
        }

        MatrixLayout layout = (args.order[grp] == 'r')
                                  ? MatrixLayout::ROW_MAJOR
                                  : MatrixLayout::COLUMN_MAJOR;

        size_t phys_a =
            computePhysicalElements(grp_m, grp_k, grp_lda, layout, trans_a)
            * elem_a;
        size_t phys_b =
            computePhysicalElements(grp_k, grp_n, grp_ldb, layout, trans_b)
            * elem_b;
        size_t phys_c =
            computePhysicalElements(grp_m, grp_n, grp_ldc, layout, false)
            * elem_c;

        // ---------------------------------------------------------------
        // Detect whether the matrix pointers are already contiguous (e.g.
        // from ContiguousBatchStorage). This hot path only supports that
        // zero-copy case: batch_gemm(prepared) just calls prim.execute()
        // with no gather-before / scatter-after step, so a scattered
        // ("COPY FALLBACK") group would have its result land in a private
        // buffer that never gets copied back into args.c_ptrs — silently
        // leaving the caller's C matrices unchanged while reporting
        // UAL_SUCCESS. Reject rather than reporting a wrong/no-op result;
        // callers with scattered matrices should use
        // batch_gemm(std::vector<BatchGroup>&, MatrixType) instead, which
        // already gathers inputs and scatters results correctly.
        bool a_contig =
            isContiguousLayout(&args.a_ptrs[matrix_idx], group_size, phys_a);
        bool b_contig =
            isContiguousLayout(&args.b_ptrs[matrix_idx], group_size, phys_b);
        bool c_contig = isContiguousLayout(
            reinterpret_cast<const void* const*>(&args.c_ptrs[matrix_idx]),
            group_size, phys_c);

        if (!(a_contig && b_contig && c_contig)) {
            args.backend_metadata_storage.push_back(nullptr);
            args.backend_metadata.push_back(nullptr);
            m_batchGroupStatus[static_cast<size_t>(grp)] =
                UALError::UAL_NOT_SUPPORTED;
            matrix_idx += group_size;
            continue;
        }

        try {
            auto cached        = std::make_shared<CachedBatchGroup>();
            cached->batch_size = group_size;

            // Create 3D memory descriptors.
            cached->a_md =
                create3DMemoryDesc(group_size, grp_m, grp_k, a_dnnl_type,
                                   layout, trans_a, grp_lda);
            cached->b_md =
                create3DMemoryDesc(group_size, grp_k, grp_n, b_dnnl_type,
                                   layout, trans_b, grp_ldb);
            cached->c_md = create3DMemoryDesc(
                group_size, grp_m, grp_n, c_dnnl_type, layout, false, grp_ldc);

            // === ZERO-COPY PATH ===
            // Point dnnl::memory directly at existing matrix data.
            cached->a_mem =
                dnnl::memory(cached->a_md, m_engine,
                             const_cast<void*>(args.a_ptrs[matrix_idx]));
            cached->b_mem =
                dnnl::memory(cached->b_md, m_engine,
                             const_cast<void*>(args.b_ptrs[matrix_idx]));
            cached->c_mem =
                dnnl::memory(cached->c_md, m_engine, args.c_ptrs[matrix_idx]);

            // Build primitive attributes (alpha / beta)
            dnnl::primitive_attr attr;
            cached->has_alpha_scale = (grp_alpha != 1.0);
            cached->alpha_val       = static_cast<float>(grp_alpha);

            if (cached->has_alpha_scale) {
                attr.set_scales_mask(DNNL_ARG_SRC, 0);
            }
            if (grp_beta != 0.0) {
                dnnl::post_ops po;
                po.append_sum(static_cast<float>(grp_beta));
                attr.set_post_ops(po);
            }

            auto pd = dnnl::matmul::primitive_desc(
                m_engine, cached->a_md, cached->b_md, cached->c_md, attr);
            cached->prim = dnnl::matmul(pd);

            if (cached->has_alpha_scale) {
                cached->alpha_mem =
                    dnnl::memory({ { 1 },
                                   dnnl::memory::data_type::f32,
                                   dnnl::memory::format_tag::x },
                                 m_engine, &cached->alpha_val);
            }

            // ---------------------------------------------------------
            // Pre-build exec_args ONCE so the hot loop does zero
            // allocation and zero map construction.
            // ---------------------------------------------------------
            cached->exec_args = { { DNNL_ARG_SRC, cached->a_mem },
                                  { DNNL_ARG_WEIGHTS, cached->b_mem },
                                  { DNNL_ARG_DST, cached->c_mem } };
            if (cached->has_alpha_scale) {
                cached->exec_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] =
                    cached->alpha_mem;
            }

            args.backend_metadata_storage.push_back(cached);
            args.backend_metadata.push_back(cached.get());

        } catch (const dnnl::error& e) {
            // dnnl_unimplemented means oneDNN itself rejected this
            // dtype/shape/ISA combo -- the same condition the vector
            // overload (batch_gemm(std::vector<BatchGroup>&, accType))
            // maps to UAL_NOT_SUPPORTED. Record that distinction here too
            // so batch_gemm(prepared) doesn't collapse it into a generic
            // UAL_FAILURE below.
            args.backend_metadata_storage.push_back(nullptr);
            args.backend_metadata.push_back(nullptr);
            m_batchGroupStatus[static_cast<size_t>(grp)] =
                (e.status == dnnl_unimplemented) ? UALError::UAL_NOT_SUPPORTED
                                                 : UALError::UAL_FAILURE;
        } catch (const std::exception&) {
            args.backend_metadata_storage.push_back(nullptr);
            args.backend_metadata.push_back(nullptr);
            m_batchGroupStatus[static_cast<size_t>(grp)] =
                UALError::UAL_FAILURE;
        }

        matrix_idx += group_size;
    }
}

// ============================================================================
// Batch GEMM Hot Path — Benchmark-Optimized
//
// This method is designed for minimal overhead in the measurement loop:
//   - ZERO memcpy (data already contiguous or copied once during prepare)
//   - ZERO allocation (exec_args pre-built during prepare)
//   - Minimal branches (one null check per group, fail-fast on error)
//
// Requires: batch_prepare_metadata() must have been called successfully.
//           If metadata is missing, the call returns UAL_FAILURE immediately.
// ============================================================================

UALError
UalOneDNN::batch_gemm(const PreparedBatchGemmArgs& prepared)
{
    // batch_prepare_metadata() returns early -- leaving backend_metadata
    // completely empty (size 0), even though group_count can be > 0 --
    // for exactly these two conditions (unsupported dtype, or native fp16
    // accumulation with no oneDNN mapping; see its matching checks).
    // Without this check, those cases fall through to the bounds check
    // below (grp >= backend_metadata.size()), which reports a generic
    // UAL_FAILURE instead of UAL_NOT_SUPPORTED -- indistinguishable from
    // a genuine execution failure to callers, unlike the single-GEMM plan
    // path, raw gemm(), and the vector batch overload, which all
    // correctly classify these same conditions as UAL_NOT_SUPPORTED.
    if (!isTypeSupported(prepared.a_type) || !isTypeSupported(prepared.b_type)
        || !isTypeSupported(prepared.c_type)
        || prepared.acc_type == MatrixType::fp16) {
        return UALError::UAL_NOT_SUPPORTED;
    }

    // Reject attributes this hot path never applies. batch_prepare_metadata()
    // builds each group's cached primitive from only m/n/k/types/alpha/beta —
    // it does not consult post_ops, a_quant_per_group, or group_scale_per_group
    // at all. Without this guard, a batch request with e.g. ReLU post-ops or
    // group-scale quantization would silently execute as a plain unscaled
    // matmul and report UAL_SUCCESS. Check every group up front so the whole
    // call fails fast instead of computing a wrong result for some groups.
    for (md_t grp = 0; grp < prepared.group_count; ++grp) {
        const auto index = static_cast<size_t>(grp);

        // BLAS contract: alpha == 0 means the result must be independent
        // of A and B. The cached primitives built by batch_prepare_
        // metadata() apply alpha as a real SRC scale, so oneDNN still
        // computes the real A@B product and only scales it afterward --
        // NaN/Inf in A/B would produce NaN instead of beta*C, and this hot
        // path would execute the cached primitive and scatter that bad
        // result back while still reporting success. Same guard as
        // OneDNNUalPlan::prepare() (suggestion 16), the raw-pointer gemm()
        // (suggestion 25), and the vector batch overload (suggestion 26).
        // Only non-empty groups matter here -- group_size[index] == 0 is a
        // legitimate no-op group with nothing to check.
        if (prepared.group_size[index] != 0) {
            const double alpha = prepared.acc_type == MatrixType::s32
                                     ? static_cast<double>(static_cast<int32_t>(
                                           prepared.alpha[index]))
                                     : prepared.alpha[index];
            if (alpha == 0.0) {
                return UALError::UAL_NOT_SUPPORTED;
            }

            // REF rounds C into the s32 accumulator before applying beta
            // for s32-accumulation with a floating-point output (see
            // OneDNNUalPlan::prepare()'s matching guard for the full
            // rationale). batch_prepare_metadata() bakes beta into the
            // cached primitive's sum post-op, which adds beta * the raw
            // fractional C as-is instead of rounding it first -- diverges
            // for a fractional C. prepared.c_type is a single value shared
            // across every group (see PreparedBatchGemmArgs).
            if (prepared.acc_type == MatrixType::s32
                && (prepared.c_type == MatrixType::f32
                    || prepared.c_type == MatrixType::bf16
                    || prepared.c_type == MatrixType::fp16)) {
                const double beta = static_cast<double>(
                    static_cast<int32_t>(prepared.beta[index]));
                if (beta != 0.0) {
                    return UALError::UAL_NOT_SUPPORTED;
                }
            }
        }

        if ((index < prepared.post_ops.size()
             && !prepared.post_ops[index].empty())
            || (index < prepared.a_quant_per_group.size()
                && prepared.a_quant_per_group[index])
            || (index < prepared.group_scale_per_group.size()
                && prepared.group_scale_per_group[index])) {
            return UALError::UAL_NOT_SUPPORTED;
        }
    }

    try {
        for (md_t grp = 0; grp < prepared.group_count; ++grp) {
            // batch_prepare_metadata() clears and can leave this vector
            // entirely empty (e.g. unsupported dtype, or fp16 accumulation —
            // see batch_prepare_metadata()'s early returns), while
            // group_count can still be > 0. Unchecked operator[] on a
            // possibly-empty vector is undefined behaviour that a
            // dnnl::error catch below cannot save us from, so bounds-check
            // before indexing.
            if (static_cast<size_t>(grp) >= prepared.backend_metadata.size()) {
                return UALError::UAL_FAILURE;
            }

            auto* cached =
                static_cast<CachedBatchGroup*>(prepared.backend_metadata[grp]);

            // A null entry is ambiguous by construction: batch_prepare_
            // metadata() pushes nullptr for several structurally different
            // reasons -- a legitimate empty group (group_size == 0, which
            // BatchGroup::validate() explicitly documents as valid and
            // which the vector overload (batch_gemm(groups, accType))
            // already treats as a no-op via `continue`), an unsupported
            // memory format / non-contiguous storage, or a genuine
            // dnnl::error / std::exception. m_batchGroupStatus (populated
            // by the batch_prepare_metadata() call that must precede this)
            // records which one applied, so the "not supported" cases can
            // report UAL_NOT_SUPPORTED here too -- matching the vector
            // overload's handling of the same underlying conditions --
            // instead of collapsing every non-empty null entry into a
            // generic UAL_FAILURE.
            if (!cached) {
                if (prepared.group_size[grp] == 0) {
                    continue;
                }
                if (static_cast<size_t>(grp) < m_batchGroupStatus.size()
                    && m_batchGroupStatus[static_cast<size_t>(grp)]
                           == UALError::UAL_NOT_SUPPORTED) {
                    return UALError::UAL_NOT_SUPPORTED;
                }
                return UALError::UAL_FAILURE;
            }

            // Single call — pre-built exec_args, no allocation, no copy.
            cached->prim.execute(m_stream, cached->exec_args);
        }
        m_stream.wait();
        return UALError::UAL_SUCCESS;

    } catch (const dnnl::error& e) {
        if (e.status == dnnl_unimplemented) {
            return UALError::UAL_NOT_SUPPORTED;
        }
        return UALError::UAL_FAILURE;
    } catch (const std::exception&) {
        return UALError::UAL_FAILURE;
    }
}

} // namespace dlp::testing::classic
