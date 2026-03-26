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

#include "framework/ual_plan.hh"

namespace dlp::testing::classic {

using dlp::testing::framework::IUalPlan;
using dlp::testing::framework::Matrix;
using dlp::testing::framework::UALError;

/**
 * @class RefUalPlan
 * @brief Reference implementation of IUalPlan
 *
 * Implements the plan-based GEMM interface for the reference backend.
 * prepare() is a no-op (ref does all work in execute). execute() applies
 * post-ops directly from the plan's quant and post-op params, then
 * delegates to UalRef to perform the reference GEMM computation.
 */
class RefUalPlan : public IUalPlan
{
  public:
    RefUalPlan()           = default;
    ~RefUalPlan() override = default;

    /**
     * @brief Prepare the plan for execution
     *
     * The reference backend does all work in execute(), so prepare()
     * simply marks the plan as ready.
     */
    void prepare() override;

    /**
     * @brief Execute the GEMM computation
     *
     * Builds a temporary RefOperation from the plan's quant and post-op
     * parameters, creates a temporary UalRef instance, and delegates the
     * GEMM computation to UalRef::gemm(). Uses matrix pointers stored by
     * setBuffers().
     *
     * @return UALError Error code indicating success or failure
     */
    UALError execute() override;

  private:
    /**
     * @brief Apply post-operations from m_post_ops to matrix C
     *
     * Iterates over the plan's post-op list and applies each operation
     * to the output matrix.
     *
     * @param C The output matrix to apply post-ops to
     */
    void applyPostOps(Matrix& C);
};

} // namespace dlp::testing::classic
