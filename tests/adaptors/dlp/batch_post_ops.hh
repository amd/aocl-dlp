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

#include "adaptors/dlp/operation_dlp.hh"
#include "framework/ual.hh"

namespace dlp::testing::classic {

inline UALError
extract_dlp_metadata(const std::vector<BatchGroup>& groups,
                     std::vector<dlp_metadata_t*>&  metadata)
{
    metadata.assign(groups.size(), nullptr);

    for (std::size_t idx = 0; idx < groups.size(); ++idx) {
        const auto& group = groups[idx];
        if (!group.postOps) {
            continue;
        }

        if (group.postOps->getUALType() != UALType::DLP) {
            return UALError::UAL_POSTOPS_MISMATCH;
        }

        auto* dlpOp = dynamic_cast<DlpOperation*>(group.postOps.get());
        if (!dlpOp) {
            return UALError::UAL_CAST_ERROR;
        }

        metadata[idx] = dlpOp->getMetadata();
    }

    return UALError::UAL_SUCCESS;
}

} // namespace dlp::testing::classic
