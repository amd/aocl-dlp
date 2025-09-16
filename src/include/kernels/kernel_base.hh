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

#include <cstdint>
#include <vector>

#include "bindings/c_wrappers/capi_kernel_frame_wrappers.h"
#include "classic/dlp_base_types.h"

#include "cpu_utils/cpu_feature_list.hh"
#include "kernel_frame/kernel_frame_base.hh"

namespace dlp::kernels {

enum class kernelError
{
    success = 0,
    error,
    badInputParams,
    badInputkernelOps
};

struct kernelParams
{};

struct gemmParams : public kernelParams
{
    void* a;
    void* b;
    void* c;

    md_t m;
    md_t n;
    md_t k;

    md_t rsA;
    md_t csA;
    md_t psA;
    md_t rsB;
    md_t csB;
    md_t rsC;
    md_t csC;

    void* alpha;
    void* beta;

    md_t     mIter;
    md_t     kIter;
    md_t     kLeft;
    uint16_t maskF32;
    alignas(64) std::array<int32_t, 8> maskArray;

    lpgemm_post_op*     kernelOpsList;
    lpgemm_post_op_attr kernelOpsAttr;

    gemmParams(void*               A,
               void*               B,
               void*               C_acc,
               md_t                _m,
               md_t                _n,
               md_t                _k,
               md_t                rs_a,
               md_t                cs_a,
               md_t                ps_a,
               md_t                rs_b,
               md_t                cs_b,
               md_t                rs_c,
               md_t                cs_c,
               void*               alpha_acc,
               void*               beta_acc,
               lpgemm_post_op*     kernelOpsList,
               lpgemm_post_op_attr kernelOpsAttr)
        : a(A)
        , b(B)
        , c(C_acc)
        , m(_m)
        , n(_n)
        , k(_k)
        , rsA(rs_a)
        , csA(cs_a)
        , psA(ps_a)
        , rsB(rs_b)
        , csB(cs_b)
        , rsC(rs_c)
        , csC(cs_c)
        , alpha(alpha_acc)
        , beta(beta_acc)
        , mIter(0)
        , kIter(0)
        , kLeft(0)
        , maskF32(0)
        , maskArray{ 0, 0, 0, 0, 0, 0, 0, 0 }
        , kernelOpsList(kernelOpsList)
        , kernelOpsAttr(kernelOpsAttr)
    {
    }

    gemmParams(const gemmParams& other)
        : a(other.a)
        , b(other.b)
        , c(other.c)
        , m(other.m)
        , n(other.n)
        , k(other.k)
        , rsA(other.rsA)
        , csA(other.csA)
        , psA(other.psA)
        , rsB(other.rsB)
        , csB(other.csB)
        , rsC(other.rsC)
        , csC(other.csC)
        , alpha(other.alpha)
        , beta(other.beta)
        , mIter(other.mIter)
        , kIter(other.kIter)
        , kLeft(other.kLeft)
        , maskF32(other.maskF32)
        , kernelOpsList(other.kernelOpsList)
        , kernelOpsAttr(other.kernelOpsAttr)
    {
        std::copy(std::begin(other.maskArray), std::end(other.maskArray),
                  std::begin(maskArray));
    }

    gemmParams(gemmParams&& other)
        : a(other.a)
        , b(other.b)
        , c(other.c)
        , m(other.m)
        , n(other.n)
        , k(other.k)
        , rsA(other.rsA)
        , csA(other.csA)
        , psA(other.psA)
        , rsB(other.rsB)
        , csB(other.csB)
        , rsC(other.rsC)
        , csC(other.csC)
        , alpha(other.alpha)
        , beta(other.beta)
        , mIter(other.mIter)
        , kIter(other.kIter)
        , kLeft(other.kLeft)
        , maskF32(other.maskF32)
        , kernelOpsList(other.kernelOpsList)
        , kernelOpsAttr(other.kernelOpsAttr)
    {
        maskArray.swap(other.maskArray);
    }

    gemmParams& operator=(const gemmParams& other)
    {
        a       = other.a;
        b       = other.b;
        c       = other.c;
        m       = other.m;
        n       = other.n;
        k       = other.k;
        rsA     = other.rsA;
        csA     = other.csA;
        psA     = other.psA;
        rsB     = other.rsB;
        csB     = other.csB;
        rsC     = other.rsC;
        csC     = other.csC;
        alpha   = other.alpha;
        beta    = other.beta;
        mIter   = other.mIter;
        kIter   = other.kIter;
        kLeft   = other.kLeft;
        maskF32 = other.maskF32;
        std::copy(std::begin(other.maskArray), std::end(other.maskArray),
                  std::begin(maskArray));
        kernelOpsList = other.kernelOpsList;
        kernelOpsAttr = other.kernelOpsAttr;
        return *this;
    }

    gemmParams& operator=(gemmParams&& other)
    {
        *this = other;
        return *this;
    }

    ~gemmParams()
    {
        a             = nullptr;
        b             = nullptr;
        c             = nullptr;
        alpha         = nullptr;
        beta          = nullptr;
        mIter         = 0;
        kIter         = 0;
        kLeft         = 0;
        maskF32       = 0;
        kernelOpsList = nullptr;
    }
};

// Runtime parameters for GEMV N1 kernel
struct gemvN1Params : public kernelParams
{
    void*               a;             // Input matrix A
    void*               x;             // Input vector x
    void*               y;             // Output vector y
    md_t                m;             // Number of rows in A
    md_t                k;             // Number of columns in A
    md_t                rsA;           // Row stride for A
    md_t                csA;           // Column stride for A
    md_t                rsB;           // Row stride for B
    md_t                csB;           // Column stride for B
    md_t                rsC;           // Row stride for C
    md_t                csC;           // Column stride for C
    md_t                k_iter;        // Number of full k iterations
    md_t                k_left;        // Remaining k elements
    md_t                m_iter;        // Number of full m iterations (m/MR)
    md_t                m_left;        // Remaining m elements
    void*               alpha;         // Scaling factor for A*x
    void*               beta;          // Scaling factor for y
    uint16_t            k_mask;        // Mask for k-dimension fringe
    uint16_t            m_mask;        // Mask for m-dimension fringe
    uint16_t            mr_mask;       // Mask for MR-dimension fringe
    lpgemm_post_op*     kernelOpsList; // List of post-ops
    lpgemm_post_op_attr kernelOpsAttr; // Attributes for post-ops

    // Constructor
    gemvN1Params(void*               A,
                 void*               X,
                 void*               Y,
                 md_t                M,
                 md_t                K,
                 md_t                rs_a,
                 md_t                cs_a,
                 md_t                rs_b,
                 md_t                cs_b,
                 md_t                rs_c,
                 md_t                cs_c,
                 void*               alpha_acc,
                 void*               beta_acc,
                 lpgemm_post_op*     kernelOps     = nullptr,
                 lpgemm_post_op_attr kernelOpsAttr = {})
        : a(A)
        , x(X)
        , y(Y)
        , m(M)
        , k(K)
        , rsA(rs_a)
        , csA(cs_a)
        , rsB(rs_b)
        , csB(cs_b)
        , rsC(rs_c)
        , csC(cs_c)
        , k_iter(0)
        , k_left(0)
        , m_iter(0)
        , m_left(0)
        , alpha(alpha_acc)
        , beta(beta_acc)
        , k_mask(0)
        , m_mask(0)
        , mr_mask(0)
        , kernelOpsList(kernelOps)
        , kernelOpsAttr(kernelOpsAttr)
    {
    }

    // Copy constructor
    gemvN1Params(const gemvN1Params& other)
        : a(other.a)
        , x(other.x)
        , y(other.y)
        , m(other.m)
        , k(other.k)
        , rsA(other.rsA)
        , csA(other.csA)
        , rsB(other.rsB)
        , csB(other.csB)
        , rsC(other.rsC)
        , csC(other.csC)
        , k_iter(other.k_iter)
        , k_left(other.k_left)
        , m_iter(other.m_iter)
        , m_left(other.m_left)
        , alpha(other.alpha)
        , beta(other.beta)
        , k_mask(other.k_mask)
        , m_mask(other.m_mask)
        , mr_mask(other.mr_mask)
        , kernelOpsList(other.kernelOpsList)
        , kernelOpsAttr(other.kernelOpsAttr)
    {
    }

    // Move constructor
    gemvN1Params(gemvN1Params&& other)
        : a(other.a)
        , x(other.x)
        , y(other.y)
        , m(other.m)
        , k(other.k)
        , rsA(other.rsA)
        , csA(other.csA)
        , rsB(other.rsB)
        , csB(other.csB)
        , rsC(other.rsC)
        , csC(other.csC)
        , k_iter(other.k_iter)
        , k_left(other.k_left)
        , m_iter(other.m_iter)
        , m_left(other.m_left)
        , alpha(other.alpha)
        , beta(other.beta)
        , k_mask(other.k_mask)
        , m_mask(other.m_mask)
        , mr_mask(other.mr_mask)
        , kernelOpsList(other.kernelOpsList)
        , kernelOpsAttr(other.kernelOpsAttr)
    {
    }

    // Copy assignment operator
    gemvN1Params& operator=(const gemvN1Params& other)
    {
        a             = other.a;
        x             = other.x;
        y             = other.y;
        m             = other.m;
        k             = other.k;
        rsA           = other.rsA;
        csA           = other.csA;
        rsB           = other.rsB;
        csB           = other.csB;
        rsC           = other.rsC;
        csC           = other.csC;
        k_iter        = other.k_iter;
        k_left        = other.k_left;
        m_iter        = other.m_iter;
        m_left        = other.m_left;
        alpha         = other.alpha;
        beta          = other.beta;
        k_mask        = other.k_mask;
        m_mask        = other.m_mask;
        mr_mask       = other.mr_mask;
        kernelOpsList = other.kernelOpsList;
        kernelOpsAttr = other.kernelOpsAttr;
        return *this;
    }

    // Move assignment operator
    gemvN1Params& operator=(gemvN1Params&& other)
    {
        *this = other;
        return *this;
    }

    // Destructor
    ~gemvN1Params()
    {
        a             = nullptr;
        x             = nullptr;
        y             = nullptr;
        alpha         = nullptr;
        beta          = nullptr;
        k_iter        = 0;
        k_left        = 0;
        m_iter        = 0;
        m_left        = 0;
        k_mask        = 0;
        m_mask        = 0;
        mr_mask       = 0;
        kernelOpsList = nullptr;
    }
};

// Runtime parameters for GEMV M=1 kernel
struct gemvM1Params : public kernelParams
{
    void* x;   // Input vector x
    void* b;   // Input matrix B
    void* y;   // Output scalar y (stored as single element)
    md_t  n;   // Number of columns in A (length of vector x)
    md_t  k;   // Number of elements to process (same as n for GEMV)
    md_t  rsX; // Row stride for x (should be 1 for unit stride)
    md_t  csX; // Column stride for x (element spacing, typically 1)
    md_t  rsB; // Row stride for B (should be 1 for row-major single row)
    md_t  csB; // Column stride for B (element spacing within the row)
    md_t  psB; // Post stride for B (element spacing within the row) // REMOVE
               // THIS
    md_t  rsY; // Row stride for y (should be 1 for scalar output)
    md_t  csY; // Column stride for y (should be 1 for scalar output)
    md_t  n_sub_updated;   // Number of updated n elements
    md_t  jc_cur_loop_rem; // Remaining jc elements in the current loop
    md_t  n_iter;          // Number of full n iterations (n/NR)
    md_t  n_left;          // Remaining n elements
    md_t  k_iter;          // Number of full k iterations for vectorization
    md_t  k_left;          // Remaining k elements
    md_t  k_iter_sub_iter; // Number of full k iterations for sub-iteration
    md_t  k_iter_sub_left; // Remaining k elements for sub-iteration
    md_t  k_left_sub_iter; // Remaining k elements for sub-iteration
    md_t  k_left_sub_left; // Remaining k elements for sub-iteration
    void* alpha;           // Scaling factor for A*x
    void* beta;            // Scaling factor for y

    // NOTE : The masks here are defined specific to NR being 64
    // TODO : Generalize the support for other NR values, similar to how
    //        it is done for GEMV(n=1).
    // Masks used by the AVX512 kernel
    std::array<uint16_t, 4> nmask_avx512;
    // Masks used by AVX512_256 kernel
    std::array<uint8_t, 8> nmask_avx512_256;
    // Lines 440-443 are needed as part of compilation. They are only used
    // by the existing AVX512 GEMV_M1 generator, and not the unified generator.
    uint16_t n0_mask; // Mask for n-dimension fringe(first 16 elements)
    uint16_t n1_mask; // Mask for n-dimension fringe(second 16 elements)
    uint16_t n2_mask; // Mask for n-dimension fringe(third 16 elements)
    uint16_t n3_mask; // Mask for n-dimension fringe(fourth 16 elements)
    // Mask used by the AVX2 kernel
    std::array<std::array<int32_t, 8>, 2> nmask_avx2;
    lpgemm_post_op*                       kernelOpsList; // List of post-ops
    lpgemm_post_op_attr kernelOpsAttr; // Attributes for post-ops

    // Constructor
    gemvM1Params(void*               X,
                 void*               B,
                 void*               Y,
                 md_t                N,
                 md_t                K,
                 md_t                rs_x,
                 md_t                cs_x,
                 md_t                rs_b,
                 md_t                cs_b,
                 md_t                rs_y,
                 md_t                cs_y,
                 md_t                n_sub_updated,
                 md_t                jc_cur_loop_rem,
                 void*               alpha_acc,
                 void*               beta_acc,
                 lpgemm_post_op*     kernelOps     = nullptr,
                 lpgemm_post_op_attr kernelOpsAttr = {})
        : x(X)
        , b(B)
        , y(Y)
        , n(N)
        , k(K)
        , rsX(rs_x)
        , csX(cs_x)
        , rsB(rs_b)
        , csB(cs_b)
        , rsY(rs_y)
        , csY(cs_y)
        , n_sub_updated(n_sub_updated)
        , jc_cur_loop_rem(jc_cur_loop_rem)
        , n_iter(0)
        , n_left(0)
        , k_iter(0)
        , k_left(0)
        , k_iter_sub_iter(0)
        , k_iter_sub_left(0)
        , k_left_sub_iter(0)
        , k_left_sub_left(0)
        , alpha(alpha_acc)
        , beta(beta_acc)
        , nmask_avx512{ 0, 0, 0, 0 }
        , nmask_avx512_256{ 0, 0, 0, 0, 0, 0, 0, 0 }
        , n0_mask(0)
        , n1_mask(0)
        , n2_mask(0)
        , n3_mask(0)
        , nmask_avx2{ { { 0, 0, 0, 0, 0, 0, 0, 0 },
                        { 0, 0, 0, 0, 0, 0, 0, 0 } } }
        , kernelOpsList(kernelOps)
        , kernelOpsAttr(kernelOpsAttr)
    {
    }

    // Copy constructor
    gemvM1Params(const gemvM1Params& other)
        : x(other.x)
        , b(other.b)
        , y(other.y)
        , n(other.n)
        , k(other.k)
        , rsX(other.rsX)
        , csX(other.csX)
        , rsB(other.rsB)
        , csB(other.csB)
        , rsY(other.rsY)
        , csY(other.csY)
        , n_sub_updated(other.n_sub_updated)
        , jc_cur_loop_rem(other.jc_cur_loop_rem)
        , n_iter(other.n_iter)
        , n_left(other.n_left)
        , k_iter(other.k_iter)
        , k_left(other.k_left)
        , k_iter_sub_iter(other.k_iter_sub_iter)
        , k_iter_sub_left(other.k_iter_sub_left)
        , k_left_sub_iter(other.k_left_sub_iter)
        , k_left_sub_left(other.k_left_sub_left)
        , alpha(other.alpha)
        , beta(other.beta)
        , nmask_avx512{ 0, 0, 0, 0 }
        , nmask_avx512_256{ 0, 0, 0, 0, 0, 0, 0, 0 }
        , n0_mask(0)
        , n1_mask(0)
        , n2_mask(0)
        , n3_mask(0)
        , nmask_avx2{ { { 0, 0, 0, 0, 0, 0, 0, 0 },
                        { 0, 0, 0, 0, 0, 0, 0, 0 } } }
        , kernelOpsList(other.kernelOpsList)
        , kernelOpsAttr(other.kernelOpsAttr)
    {
    }

    // Move constructor
    gemvM1Params(gemvM1Params&& other)
        : x(other.x)
        , b(other.b)
        , y(other.y)
        , n(other.n)
        , k(other.k)
        , rsX(other.rsX)
        , csX(other.csX)
        , rsB(other.rsB)
        , csB(other.csB)
        , rsY(other.rsY)
        , csY(other.csY)
        , n_sub_updated(other.n_sub_updated)
        , jc_cur_loop_rem(other.jc_cur_loop_rem)
        , n_iter(other.n_iter)
        , n_left(other.n_left)
        , k_iter(other.k_iter)
        , k_left(other.k_left)
        , k_iter_sub_iter(other.k_iter_sub_iter)
        , k_iter_sub_left(other.k_iter_sub_left)
        , k_left_sub_iter(other.k_left_sub_iter)
        , k_left_sub_left(other.k_left_sub_left)
        , alpha(other.alpha)
        , beta(other.beta)
        , nmask_avx512{ 0, 0, 0, 0 }
        , nmask_avx512_256{ 0, 0, 0, 0, 0, 0, 0, 0 }
        , n0_mask(0)
        , n1_mask(0)
        , n2_mask(0)
        , n3_mask(0)
        , nmask_avx2{ { { 0, 0, 0, 0, 0, 0, 0, 0 },
                        { 0, 0, 0, 0, 0, 0, 0, 0 } } }
        , kernelOpsList(other.kernelOpsList)
        , kernelOpsAttr(other.kernelOpsAttr)
    {
    }

    // Copy assignment operator
    gemvM1Params& operator=(const gemvM1Params& other)
    {
        x                = other.x;
        b                = other.b;
        y                = other.y;
        n                = other.n;
        k                = other.k;
        rsX              = other.rsX;
        csX              = other.csX;
        rsB              = other.rsB;
        csB              = other.csB;
        rsY              = other.rsY;
        csY              = other.csY;
        n_sub_updated    = other.n_sub_updated;
        jc_cur_loop_rem  = other.jc_cur_loop_rem;
        n_iter           = other.n_iter;
        n_left           = other.n_left;
        k_iter           = other.k_iter;
        k_left           = other.k_left;
        k_iter_sub_iter  = other.k_iter_sub_iter;
        k_iter_sub_left  = other.k_iter_sub_left;
        k_left_sub_iter  = other.k_left_sub_iter;
        k_left_sub_left  = other.k_left_sub_left;
        alpha            = other.alpha;
        beta             = other.beta;
        nmask_avx512     = other.nmask_avx512;
        nmask_avx512_256 = other.nmask_avx512_256;
        n0_mask          = other.n0_mask;
        n1_mask          = other.n1_mask;
        n2_mask          = other.n2_mask;
        n3_mask          = other.n3_mask;
        nmask_avx2       = other.nmask_avx2;
        kernelOpsList    = other.kernelOpsList;
        kernelOpsAttr    = other.kernelOpsAttr;
        return *this;
    }

    // Move assignment operator
    gemvM1Params& operator=(gemvM1Params&& other)
    {
        *this = other;
        return *this;
    }

    // Destructor
    ~gemvM1Params()
    {
        x                = nullptr;
        b                = nullptr;
        y                = nullptr;
        alpha            = nullptr;
        beta             = nullptr;
        n_iter           = 0;
        n_left           = 0;
        k_iter           = 0;
        k_left           = 0;
        k_iter_sub_iter  = 0;
        k_iter_sub_left  = 0;
        k_left_sub_iter  = 0;
        k_left_sub_left  = 0;
        nmask_avx512     = { 0, 0, 0, 0 };
        nmask_avx512_256 = { 0, 0, 0, 0, 0, 0, 0, 0 };
        n0_mask          = 0;
        n1_mask          = 0;
        n2_mask          = 0;
        n3_mask          = 0;
        nmask_avx2       = { { { 0, 0, 0, 0, 0, 0, 0, 0 },
                               { 0, 0, 0, 0, 0, 0, 0, 0 } } };
        kernelOpsList    = nullptr;
        kernelOpsAttr    = {};
    }
};

class kernelBase
{
  public:
    virtual ~kernelBase() {}

    virtual std::vector<cpu_utils::isaFeature>& getIsaFeaturesForKernel()   = 0;
    virtual kernel_frame::kernelInfo*           getKernelInfo()             = 0;
    virtual std::vector<kernel_frame::kernelDatatype>& getKernelDatatypes() = 0;
    virtual kernels::kernelError operator()(kernels::kernelParams* kP)      = 0;
};

} // namespace dlp::kernels
