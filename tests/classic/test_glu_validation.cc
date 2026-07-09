/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * Unit tests for the GLU post-op input validation enforced by the classic
 * translation layer (dlp_gemm_translate_to_post_ops_list): GLU must be the
 * terminal post-op, may appear at most once per chain, and requires an even
 * interleaved output width n = 2I. Each case drives the public BF16 GEMM API
 * and asserts the error code surfaced on the metadata.
 */

#include "aocl_dlp.h"
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

namespace {

bfloat16
f2bf16(float v)
{
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return (bfloat16)(bits >> 16);
}

// Drive the public BF16 GEMM API with a GLU post-op described by `seq` and
// return the error code the API records on the metadata. Buffers are sized
// M x 2I (row-major, ldc = n); for the rejection cases the translation layer
// returns before any kernel runs.
dlp_clsc_err_t
run_glu(md_t m, md_t n, md_t k, const std::vector<DLP_POST_OP_TYPE>& seq)
{
    const md_t            lda = k, ldb = n, ldc = n;
    std::vector<bfloat16> A((size_t)m * lda, f2bf16(0.1f));
    std::vector<bfloat16> B((size_t)k * ldb, f2bf16(0.1f));
    std::vector<float>    C((size_t)m * ldc, 0.0f);

    // Compacted GLU output buffer D (m x I, row-major ld_d = I = n/2). D is
    // mandatory for a shape-changing GLU; size it for the largest even n <=
    // this n so the buffer is valid even for the odd-n rejection case.
    const md_t         I = n / 2;
    std::vector<float> D((size_t)m * (I > 0 ? I : 1), 0.0f);

    dlp_metadata_t md;
    std::memset(&md, 0, sizeof(md));

    std::vector<DLP_POST_OP_TYPE> seqv = seq;
    md.seq_length                      = (md_t)seqv.size();
    md.seq_vector                      = seqv.data();

    dlp_post_op_glu glu;
    std::memset(&glu, 0, sizeof(glu));
    glu.algo_type = GATED_SWIGLU;
    glu.alpha     = NULL;
    glu.beta      = NULL;
    glu.stor_type = DLP_INVALID;
    glu.d         = D.data();
    glu.ld_d      = I;
    md.glu        = &glu;

    aocl_gemm_bf16bf16f32of32('R', 'N', 'N', m, n, k, 1.0f, A.data(), lda, 'N',
                              B.data(), ldb, 'N', 0.0f, C.data(), ldc, &md);

    return md.error_hndl.error_code;
}

} // namespace

// n must be even (n = 2I); an odd interleaved width has no gate/up pairing.
TEST(GluValidation, RejectsOddInterleavedWidth)
{
    EXPECT_EQ(run_glu(4, 15, 32, { GLU }), DLP_CLSC_INVALID_MATRIX_DIMENSION);
}

// GEMV n == 1: the odd-width case that shows up in practice (a single output
// column). A shape-changing GLU folds n = 2I to I, so n == 1 has no gate/up
// pairing and must be rejected like any other odd n -- a known non-existent
// shape, not a valid GEMV.
TEST(GluValidation, RejectsGemvSingleColumnWidth)
{
    EXPECT_EQ(run_glu(6, 1, 64, { GLU }), DLP_CLSC_INVALID_MATRIX_DIMENSION);
}

// GLU is unique: at most one per chain (n stays even so the rejection is due
// to the duplicate GLU, not the width).
TEST(GluValidation, RejectsGluChain)
{
    EXPECT_EQ(run_glu(8, 32, 16, { GLU, GLU }), DLP_CLSC_UNEXPECTED_VECTOR_DIM);
}

// GLU is terminal: nothing may follow it (it changes the shape 2I -> I). n
// stays even so the rejection is due to the trailing op, not the width.
TEST(GluValidation, RejectsGluNotLast)
{
    EXPECT_EQ(run_glu(2, 8, 48, { GLU, BIAS }), DLP_CLSC_UNEXPECTED_VECTOR_DIM);
}

// Positive control: a valid even-width single GLU must not be rejected by the
// validation guards. On the AVX512-BF16 BF16 datapath it runs and returns
// SUCCESS. On machines without AVX512-BF16 (e.g. Zen3) the BF16 API reroutes to
// the F32 kernels, which do not implement the GLU half-width store, so the
// framework returns NOT_SUPPORTED -- a clean platform skip, not a validation
// rejection. We cannot query AVX512-BF16 from this test (it links the shared
// library and only sees the public API), so treat NOT_SUPPORTED as a skip and
// otherwise require SUCCESS. A validation-rejection code (INVALID_MATRIX_
// DIMENSION / UNEXPECTED_VECTOR_DIM) would still fail the EXPECT_EQ below.
TEST(GluValidation, AcceptsEvenWidthSingleGlu)
{
    const dlp_clsc_err_t err = run_glu(4, 16, 32, { GLU });
    if (err == DLP_CLSC_NOT_SUPPORTED) {
        GTEST_SKIP() << "Shape-changing GLU is not supported on this platform "
                        "(BF16 reroutes to F32; no half-width store). The "
                        "validation guards passed -- nothing to assert here.";
    }
    EXPECT_EQ(err, DLP_CLSC_SUCCESS);
}

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
