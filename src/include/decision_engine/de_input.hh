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

#include "bindings/c_wrappers/capi_kernel_frame_wrappers.h"
#include "kernel_frame/kernel_frame_base.hh"

namespace dlp::de {

// This class will be used only for enforcing the interface, and will not be
// used as part of strategy pattern. Pointers of this type needs to be
// static_cast to the concrete derived class and then processed. Basically
// no virtual dispatch is required.
class iDEInput
{};

class gemmDEInput : public iDEInput
{
  public:
    kernel_frame::kernelDatatype k_dtype;
    md_t                         m;
    md_t                         n;
    md_t                         k;
    md_t                         rs_a;
    md_t                         cs_a;
    md_t                         rs_b;
    md_t                         cs_b;
    md_t                         rs_c;
    md_t                         cs_c;
    void*                        alpha;
    void*                        beta;
    AOCL_MEMORY_TAG              mtag_a;
    AOCL_MEMORY_TAG              mtag_b;
    lpgemm_post_op*              metadata;
    md_t                         mr_hint;
    md_t                         nr_hint;
    md_t                         kc_hint;
    md_t                         c_downscale;

    gemmDEInput(kernel_frame::kernelDatatype k_dtype,
                md_t                         m,
                md_t                         n,
                md_t                         k,
                md_t                         rs_a,
                md_t                         cs_a,
                md_t                         rs_b,
                md_t                         cs_b,
                md_t                         rs_c,
                md_t                         cs_c,
                void*                        alpha,
                void*                        beta,
                AOCL_MEMORY_TAG              mtag_a,
                AOCL_MEMORY_TAG              mtag_b,
                lpgemm_post_op*              metadata,
                md_t                         mr_hint,
                md_t                         nr_hint,
                md_t                         kc_hint,
                md_t                         c_downscale)
        : k_dtype(k_dtype)
        , m(m)
        , n(n)
        , k(k)
        , rs_a(rs_a)
        , cs_a(cs_a)
        , rs_b(rs_b)
        , cs_b(cs_b)
        , rs_c(rs_c)
        , cs_c(cs_c)
        , alpha(alpha)
        , beta(beta)
        , mtag_a(mtag_a)
        , mtag_b(mtag_b)
        , metadata(metadata)
        , mr_hint(mr_hint)
        , nr_hint(nr_hint)
        , kc_hint(kc_hint)
        , c_downscale(c_downscale)
    {
    }

    gemmDEInput(const gemmDEInput& other)
        : k_dtype(other.k_dtype)
        , m(other.m)
        , n(other.n)
        , k(other.k)
        , rs_a(other.rs_a)
        , cs_a(other.cs_a)
        , rs_b(other.rs_b)
        , cs_b(other.cs_b)
        , rs_c(other.rs_c)
        , cs_c(other.cs_c)
        , alpha(other.alpha)
        , beta(other.beta)
        , mtag_a(other.mtag_a)
        , mtag_b(other.mtag_b)
        , metadata(other.metadata)
        , mr_hint(other.mr_hint)
        , nr_hint(other.nr_hint)
        , kc_hint(other.kc_hint)
        , c_downscale(other.c_downscale)
    {
    }

    gemmDEInput(gemmDEInput&& other)
        : k_dtype(other.k_dtype)
        , m(other.m)
        , n(other.n)
        , k(other.k)
        , rs_a(other.rs_a)
        , cs_a(other.cs_a)
        , rs_b(other.rs_b)
        , cs_b(other.cs_b)
        , rs_c(other.rs_c)
        , cs_c(other.cs_c)
        , alpha(other.alpha)
        , beta(other.beta)
        , mtag_a(other.mtag_a)
        , mtag_b(other.mtag_b)
        , metadata(other.metadata)
        , mr_hint(other.mr_hint)
        , nr_hint(other.nr_hint)
        , kc_hint(other.kc_hint)
        , c_downscale(other.c_downscale)
    {
    }

    gemmDEInput& operator=(const gemmDEInput& other)
    {
        k_dtype     = other.k_dtype;
        m           = other.m;
        n           = other.n;
        k           = other.k;
        rs_a        = other.rs_a;
        cs_a        = other.cs_a;
        rs_b        = other.rs_b;
        cs_b        = other.cs_b;
        rs_c        = other.rs_c;
        cs_c        = other.cs_c;
        alpha       = other.alpha;
        beta        = other.beta;
        mtag_a      = other.mtag_a;
        mtag_b      = other.mtag_b;
        metadata    = other.metadata;
        mr_hint     = other.mr_hint;
        nr_hint     = other.nr_hint;
        kc_hint     = other.kc_hint;
        c_downscale = other.c_downscale;
        return *this;
    }

    gemmDEInput& operator=(gemmDEInput&& other)
    {
        *this = other;
        return *this;
    }
};

} // namespace dlp::de
