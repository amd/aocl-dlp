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
    }

    gemmParams& operator=(const gemmParams& other)
    {
        a             = other.a;
        b             = other.b;
        c             = other.c;
        m             = other.m;
        n             = other.n;
        k             = other.k;
        rsA           = other.rsA;
        csA           = other.csA;
        psA           = other.psA;
        rsB           = other.rsB;
        csB           = other.csB;
        rsC           = other.rsC;
        csC           = other.csC;
        alpha         = other.alpha;
        beta          = other.beta;
        mIter         = other.mIter;
        kIter         = other.kIter;
        kLeft         = other.kLeft;
        maskF32       = other.maskF32;
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
