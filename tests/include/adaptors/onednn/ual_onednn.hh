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

#include "framework/ual.hh"
#include <oneapi/dnnl/dnnl.hpp>
#include <unordered_map>

namespace dlp::testing::classic {

using dlp::testing::framework::BatchGroup;
using dlp::testing::framework::GroupScaleParam;
using dlp::testing::framework::IUal;
using dlp::testing::framework::Matrix;
using dlp::testing::framework::MatrixLayout;
using dlp::testing::framework::MatrixMemFormat;
using dlp::testing::framework::MatrixType;
using dlp::testing::framework::PreparedBatchGemmArgs;
using dlp::testing::framework::UALError;
using dlp::testing::framework::UALType;
using dlp::testing::framework::WOQParam;

/**
 * @class UalOneDNN
 * @brief oneDNN implementation of UAL
 *
 * This class implements the IUAL interface using Intel oneDNN library,
 * providing GEMM operations for various data types.
 *
 * Phase 1: Core GEMM only, no post-operations support.
 */
class UalOneDNN : public IUal
{
  public:
    /**
     * @brief Constructor
     *
     * Initializes a oneDNN-based UAL implementation with CPU engine.
     */
    UalOneDNN();

    /**
     * @brief Create an oneDNN execution plan
     * @return Unique pointer to an OneDNNUalPlan
     */
    std::unique_ptr<dlp::testing::framework::IUalPlan> createPlan() override;

    /**
     * @brief Destructor
     */
    ~UalOneDNN() override = default;

    /**
     * @brief Reorder matrix data to optimized layout using dnnl::reorder
     *
     * This method uses oneDNN's reorder primitive with format_tag::any to
     * transform weights into the optimal blocked format for the matmul kernel.
     * The optimal blocking depends on the full GEMM shape (M, N, K), so M
     * enables better blocking selection.
     *
     * @param in Input matrix to reorder (typically B/weights)
     * @param out Output matrix to store reordered data
     * @param A_type Type of matrix A in GEMM context
     * @param B_type Type of matrix B in GEMM context
     * @param C_type Type of matrix C in GEMM context
     * @param accType Accumulation type
     * @param M M dimension for optimal blocking (0 = use default 256)
     * @param transA Whether matrix A is transposed
     * @param group_scale Optional symmetric-quantization group-scale parameters
     *                    (currently unused by oneDNN adaptor)
     * @param woq Optional weight-only-quantization parameters
     *            (currently unused by oneDNN adaptor)
     * @return UALError Error code indicating success or failure
     */
    UALError reorder(const Matrix&          in,
                     Matrix&                out,
                     MatrixType             A_type,
                     MatrixType             B_type,
                     MatrixType             C_type,
                     MatrixType             accType,
                     md_t                   M           = 0,
                     bool                   transA      = false,
                     const GroupScaleParam* group_scale = nullptr,
                     const WOQParam*        woq         = nullptr) override;

    /**
     * @brief Perform batch GEMM operations
     *
     * @param groups Vector of batch groups
     * @param accType Accumulation type
     * @return UALError Error code indicating success or failure
     */
    UALError batch_gemm(std::vector<BatchGroup>& groups,
                        MatrixType               accType) override;

    /**
     * @brief Prepare backend-specific metadata for batch GEMM
     *
     * @param args Prepared batch GEMM arguments
     */
    void batch_prepare_metadata(PreparedBatchGemmArgs& args) override;

    /**
     * @brief Perform optimized batch GEMM using pre-prepared metadata
     *
     * @param prepared Pre-prepared batch GEMM arguments
     * @return UALError Error code indicating success or failure
     */
    UALError batch_gemm(const PreparedBatchGemmArgs& prepared) override;

    /**
     * @brief Perform GEMM with raw pointers for benchmarking
     */
    UALError gemm(md_t         m,
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
                  double       alpha = 1.0,
                  double       beta  = 0.0) override;

    friend class OneDNNUalPlan;

  private:
    // === Type Conversion ===

    /**
     * @brief Convert MatrixType to oneDNN data type
     */
    static dnnl::memory::data_type toOneDNNType(MatrixType type);

    /**
     * @brief Check if type is supported by oneDNN
     */
    static bool isTypeSupported(MatrixType type);

    /**
     * @brief Check whether a u8/s8 x s8 GEMM with s32 accumulation on the
     * CURRENT effective CPU ISA risks oneDNN's AVX2 "gemm:jit" saturation
     * hazard.
     *
     * Without true VNNI-class int8 dot-product instructions (VPDPBUSD,
     * available from AVX2-VNNI / AVX-512 VNNI onward), oneDNN's x8s8s32
     * kernel computes each multiply-accumulate pair via VPMADDUBSW, whose
     * intermediate is a SIGNED 16-bit sum of TWO products that SATURATES
     * at [-32768, 32767] rather than widening to 32-bit. For full-range
     * inputs (e.g. u8 A=255, s8 B=127 at K=2: 2*255*127 = 64770) this
     * silently returns a saturated 32767 instead of the exact int32
     * result, while still reporting UAL_SUCCESS -- making it NOT a valid
     * independent oracle for this dtype combination on such hardware.
     * AVX2-VNNI (and every ISA level above it: AVX-512 VNNI, AMX, ...)
     * uses a genuine non-saturating int32 dot product and is unaffected.
     *
     * This cannot be data-dependent (prepare()/reorder() never see the
     * actual matrix contents), so it rejects the dtype+ISA combination
     * unconditionally, the same way acc_type==fp16 is rejected
     * unconditionally elsewhere in this backend regardless of whether a
     * given call's data would actually have triggered the divergence.
     */
    static bool isSaturatingAvx2IntGemm(MatrixType a_type,
                                        MatrixType b_type,
                                        MatrixType acc_type);

    /**
     * @brief Check whether an int32 value can be represented EXACTLY as
     * an f32.
     *
     * For s32 accumulation, DLP/REF truncate alpha/beta to int32_t and
     * apply them with true integer arithmetic (see e.g.
     * tests/adaptors/ref/gemm_u8s8s32os32_ref.cc, where alpha is an
     * int32_t consumed via exact uint32_t multiply/add). This backend
     * instead hands alpha/beta to oneDNN as a plain f32 scale / sum
     * post-op coefficient, which has only 24 bits of mantissa: any
     * int32_t whose magnitude needs more precision than that (e.g.
     * 16777217, which f32 rounds to 16777216) silently becomes a
     * DIFFERENT coefficient than the exact integer DLP/REF actually
     * applies -- computing a wrong result while still reporting
     * UAL_SUCCESS, even for as trivial a case as C = alpha * 1 * 1 with
     * beta = 0. Reject rather than silently losing precision.
     */
    static bool isExactlyRepresentableAsFloat(int32_t value);

    // === Memory Descriptor Creation ===

    /**
     * @brief Create oneDNN memory descriptor with proper strides
     *
     * @param rows Number of rows (M or K)
     * @param cols Number of columns (K or N)
     * @param type Data type
     * @param layout Memory layout (row-major or column-major)
     * @param transposed Whether the matrix is transposed
     * @param leadingDim Leading dimension
     * @return dnnl::memory::desc Memory descriptor
     */
    static dnnl::memory::desc createMemoryDesc(md_t         rows,
                                               md_t         cols,
                                               MatrixType   type,
                                               MatrixLayout layout,
                                               bool         transposed,
                                               md_t         leadingDim);

    // === Alpha Scaling Helper ===

    /**
     * @brief Apply alpha scaling to output matrix (post-multiply)
     *
     * oneDNN matmul doesn't directly support alpha scaling,
     * so we apply it as a post-processing step.
     */
    static void applyAlphaScaling(void*        data,
                                  MatrixType   type,
                                  md_t         m,
                                  md_t         n,
                                  md_t         ld,
                                  MatrixLayout layout,
                                  double       alpha);

    // === Primitive Caching ===

    /**
     * @brief Key for primitive cache lookup
     *
     * Note: alpha and beta are included in the key because oneDNN's native
     * scaling is configured at primitive creation time via output scales
     * and sum post-op.
     */
    struct PrimitiveKey
    {
        md_t       m, n, k;
        MatrixType a_type, b_type, c_type;
        bool       trans_a, trans_b;
        md_t       lda, ldb, ldc;
        float      alpha; // Output scale factor
        float      beta;  // Sum post-op scale (0 = no sum)
        // Include layout info for correctness
        MatrixLayout a_layout, b_layout, c_layout;

        bool operator==(const PrimitiveKey& other) const
        {
            return m == other.m && n == other.n && k == other.k
                   && a_type == other.a_type && b_type == other.b_type
                   && c_type == other.c_type && trans_a == other.trans_a
                   && trans_b == other.trans_b && lda == other.lda
                   && ldb == other.ldb && ldc == other.ldc
                   && alpha == other.alpha && beta == other.beta
                   && a_layout == other.a_layout && b_layout == other.b_layout
                   && c_layout == other.c_layout;
        }
    };

    /**
     * @brief Hash function for PrimitiveKey
     */
    struct PrimitiveKeyHash
    {
        size_t operator()(const PrimitiveKey& key) const
        {
            size_t h            = 0;
            auto   hash_combine = [&h](size_t val) {
                h ^= val + 0x9e3779b9 + (h << 6) + (h >> 2);
            };
            hash_combine(std::hash<md_t>{}(key.m));
            hash_combine(std::hash<md_t>{}(key.n));
            hash_combine(std::hash<md_t>{}(key.k));
            hash_combine(static_cast<size_t>(key.a_type));
            hash_combine(static_cast<size_t>(key.b_type));
            hash_combine(static_cast<size_t>(key.c_type));
            hash_combine(key.trans_a ? 1 : 0);
            hash_combine(key.trans_b ? 1 : 0);
            hash_combine(std::hash<md_t>{}(key.lda));
            hash_combine(std::hash<md_t>{}(key.ldb));
            hash_combine(std::hash<md_t>{}(key.ldc));
            hash_combine(std::hash<float>{}(key.alpha));
            hash_combine(std::hash<float>{}(key.beta));
            hash_combine(static_cast<size_t>(key.a_layout));
            hash_combine(static_cast<size_t>(key.b_layout));
            hash_combine(static_cast<size_t>(key.c_layout));
            return h;
        }
    };

    /**
     * @brief Cached primitive and related descriptors
     */
    struct CachedPrimitive
    {
        dnnl::matmul         prim;
        dnnl::memory::desc   a_md;
        dnnl::memory::desc   b_md;
        dnnl::memory::desc   c_md;
        dnnl::primitive_attr attr;
    };

    /**
     * @brief Get or create cached primitive for given key
     *
     * Creates a primitive with native oneDNN alpha/beta support:
     * - alpha: Applied via output scales (DNNL_ARG_DST)
     * - beta: Applied via sum post-op (fused into kernel)
     *
     * This avoids the overhead of manual alpha/beta scaling loops.
     */
    CachedPrimitive& getOrCreatePrimitive(const PrimitiveKey& key);

    // === Weight Caching for Optimal Blocking ===

    /**
     * @brief Key for weight cache lookup
     *
     * Weight blocking depends on:
     * - Weight dimensions (K, N)
     * - Data types (affects kernel selection)
     * - M dimension (affects optimal blocking)
     * - Transpose flags
     */
    struct WeightKey
    {
        const void*
             data_ptr; // Input data pointer - ensures unique cache per input
        md_t k, n;
        md_t m; // M dimension affects optimal blocking
        MatrixType   a_type, b_type, c_type;
        bool         trans_a, trans_b;
        md_t         ldb;
        MatrixLayout b_layout;

        bool operator==(const WeightKey& other) const
        {
            return data_ptr == other.data_ptr && k == other.k && n == other.n
                   && m == other.m && a_type == other.a_type
                   && b_type == other.b_type && c_type == other.c_type
                   && trans_a == other.trans_a && trans_b == other.trans_b
                   && ldb == other.ldb && b_layout == other.b_layout;
        }
    };

    /**
     * @brief Hash function for WeightKey
     */
    struct WeightKeyHash
    {
        size_t operator()(const WeightKey& key) const
        {
            size_t h            = 0;
            auto   hash_combine = [&h](size_t val) {
                h ^= val + 0x9e3779b9 + (h << 6) + (h >> 2);
            };
            hash_combine(std::hash<const void*>{}(key.data_ptr));
            hash_combine(std::hash<md_t>{}(key.k));
            hash_combine(std::hash<md_t>{}(key.n));
            hash_combine(std::hash<md_t>{}(key.m));
            hash_combine(static_cast<size_t>(key.a_type));
            hash_combine(static_cast<size_t>(key.b_type));
            hash_combine(static_cast<size_t>(key.c_type));
            hash_combine(key.trans_a ? 1 : 0);
            hash_combine(key.trans_b ? 1 : 0);
            hash_combine(std::hash<md_t>{}(key.ldb));
            hash_combine(static_cast<size_t>(key.b_layout));
            return h;
        }
    };

    /**
     * @brief Cached reordered weight and its descriptor
     *
     * Stores the oneDNN memory object with optimally blocked weights
     * and the corresponding memory descriptor for use in matmul.
     */
    struct CachedWeight
    {
        dnnl::memory       blocked_memory; // Optimally blocked weight data
        dnnl::memory::desc blocked_desc; // Memory descriptor for blocked format
        md_t               k, n;         // Original dimensions
    };

    // === Source (A) Caching for Optimal Blocking ===

    /**
     * @brief Key for source (A) cache lookup
     *
     * Source blocking depends on:
     * - Source dimensions (M, K)
     * - Data types (affects kernel selection)
     * - N dimension (affects optimal blocking)
     * - Transpose flags
     */
    struct SrcKey
    {
        const void*
             data_ptr; // Input data pointer - ensures unique cache per input
        md_t m, k;
        md_t n; // N dimension affects optimal blocking
        MatrixType   a_type, b_type, c_type;
        bool         trans_a, trans_b;
        md_t         lda;
        MatrixLayout a_layout;

        bool operator==(const SrcKey& other) const
        {
            return data_ptr == other.data_ptr && m == other.m && k == other.k
                   && n == other.n && a_type == other.a_type
                   && b_type == other.b_type && c_type == other.c_type
                   && trans_a == other.trans_a && trans_b == other.trans_b
                   && lda == other.lda && a_layout == other.a_layout;
        }
    };

    /**
     * @brief Hash function for SrcKey
     */
    struct SrcKeyHash
    {
        size_t operator()(const SrcKey& key) const
        {
            size_t h            = 0;
            auto   hash_combine = [&h](size_t val) {
                h ^= val + 0x9e3779b9 + (h << 6) + (h >> 2);
            };
            hash_combine(std::hash<const void*>{}(key.data_ptr));
            hash_combine(std::hash<md_t>{}(key.m));
            hash_combine(std::hash<md_t>{}(key.k));
            hash_combine(std::hash<md_t>{}(key.n));
            hash_combine(static_cast<size_t>(key.a_type));
            hash_combine(static_cast<size_t>(key.b_type));
            hash_combine(static_cast<size_t>(key.c_type));
            hash_combine(key.trans_a ? 1 : 0);
            hash_combine(key.trans_b ? 1 : 0);
            hash_combine(std::hash<md_t>{}(key.lda));
            hash_combine(static_cast<size_t>(key.a_layout));
            return h;
        }
    };

    /**
     * @brief Cached reordered source and its descriptor
     *
     * Stores the oneDNN memory object with optimally blocked source (A)
     * and the corresponding memory descriptor for use in matmul.
     */
    struct CachedSrc
    {
        dnnl::memory       blocked_memory; // Optimally blocked source data
        dnnl::memory::desc blocked_desc; // Memory descriptor for blocked format
        md_t               m, k;         // Original dimensions
    };

    /**
     * @brief Create matmul primitive descriptor for weight blocking query
     *
     * Creates a matmul primitive descriptor with format_tag::any for weights
     * to query the optimal blocking format.
     */
    dnnl::matmul::primitive_desc createBlockedMatmulPD(md_t         m,
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
                                                       md_t         ldc);

    // === Members ===
    dnnl::engine m_engine;
    dnnl::stream m_stream;

    // Primitive cache for benchmarking performance
    std::unordered_map<PrimitiveKey, CachedPrimitive, PrimitiveKeyHash>
        m_primitiveCache;

    // Weight cache for optimally blocked weights (fair benchmarking)
    // gemm() looks up by WeightKey constructed from GEMM parameters
    std::unordered_map<WeightKey, CachedWeight, WeightKeyHash> m_weightCache;

    // Source cache for optimally blocked source/activations (fair benchmarking)
    // Used when packA=true to cache A matrix transformations
    std::unordered_map<SrcKey, CachedSrc, SrcKeyHash> m_srcCache;

    // === Format cache for pre-reordered matrices ===
    // When a matrix has been pre-reordered via reorder(), gemm() needs to know
    // the blocked format descriptor but the data pointer may have changed due
    // to Matrix copy operations. This cache stores just the format descriptor
    // keyed by matrix properties (without data_ptr).
    struct FormatKey
    {
        md_t         k, n;
        md_t         m; // M dimension affects optimal blocking
        MatrixType   a_type, b_type, c_type;
        bool         trans_a, trans_b;
        MatrixLayout b_layout;

        bool operator==(const FormatKey& other) const
        {
            return k == other.k && n == other.n && m == other.m
                   && a_type == other.a_type && b_type == other.b_type
                   && c_type == other.c_type && trans_a == other.trans_a
                   && trans_b == other.trans_b && b_layout == other.b_layout;
        }
    };

    struct FormatKeyHash
    {
        size_t operator()(const FormatKey& key) const
        {
            size_t h            = 0;
            auto   hash_combine = [&h](size_t val) {
                h ^= val + 0x9e3779b9 + (h << 6) + (h >> 2);
            };
            hash_combine(std::hash<md_t>{}(key.k));
            hash_combine(std::hash<md_t>{}(key.n));
            hash_combine(std::hash<md_t>{}(key.m));
            hash_combine(static_cast<size_t>(key.a_type));
            hash_combine(static_cast<size_t>(key.b_type));
            hash_combine(static_cast<size_t>(key.c_type));
            hash_combine(key.trans_a ? 1 : 0);
            hash_combine(key.trans_b ? 1 : 0);
            hash_combine(static_cast<size_t>(key.b_layout));
            return h;
        }
    };

    // Cache for blocked weight format descriptors (used with pre-reordered
    // matrices)
    std::unordered_map<FormatKey, dnnl::memory::desc, FormatKeyHash>
        m_weightFormatCache;

    // Per-group status from the last batch_prepare_metadata() call, indexed
    // by group index. batch_prepare_metadata() pushes a null
    // backend_metadata entry for several structurally different reasons
    // (unsupported memory format, non-contiguous storage, a genuine
    // dnnl::error/unimplemented) but a bare nullptr can't carry WHICH
    // reason applied. batch_gemm(prepared) consults this parallel vector
    // when it sees a null entry so it can report UAL_NOT_SUPPORTED
    // (matching batch_gemm(std::vector<BatchGroup>&, accType)'s handling
    // of the same underlying conditions) instead of always collapsing
    // every null entry to UAL_FAILURE.
    std::vector<UALError> m_batchGroupStatus;
};

} // namespace dlp::testing::classic
