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

#include "classic/dlp_base_types.h"
#include "framework/operation.hh"

#include <any>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace dlp::testing {

/**
 * @class PostOpsIterator
 * @brief Iterator for generating PostOps combinations and creating IOperation
 * objects
 *
 * This class manages the generation of all possible PostOps combinations based
 * on the YAML configuration. It supports both cartesian (all combinations and
 * permutations) and simple (individual operations only) modes.
 */
class PostOpsIterator
{
  public:
    /**
     * @struct PostOpConfig
     * @brief Configuration for a single PostOps operation from YAML
     */
    struct PostOpConfig
    {
        std::string
            type; ///< Operation type (e.g., "Elementwise-PRELU", "Bias", "Sum")
        std::map<std::string, std::vector<std::any>>
            params; ///< Parameters with their possible values
    };

  private:
    std::vector<PostOpConfig>
           m_operations;        ///< All PostOps operations from YAML
    bool   m_cartesian_enabled; ///< Whether to generate all combinations
    size_t m_current_index;     ///< Current combination index
    std::vector<std::vector<size_t>>
        m_combinations; ///< All possible combinations (indices into
                        ///< m_operations)

  public:
    /**
     * @brief Construct PostOpsIterator with operations and mode
     * @param operations Vector of PostOp configurations from YAML
     * @param cartesian_enabled Whether to generate all
     * combinations/permutations
     */
    PostOpsIterator(const std::vector<PostOpConfig>& operations,
                    bool                             cartesian_enabled);

    /**
     * @brief Check if there are more combinations available
     * @return true if more combinations exist, false otherwise
     */
    bool hasNext() const;

    /**
     * @brief Advance to the next combination
     */
    void next();

    /**
     * @brief Reset iterator to the first combination
     */
    void reset();

    /**
     * @brief Get total number of combinations
     * @return Number of combinations that will be generated
     */
    size_t getSize() const;

    /**
     * @brief Get current combination as ordered vector of operation indices
     * @return Vector of indices into m_operations representing current
     * combination
     */
    std::vector<size_t> getCurrentCombination() const;

    /**
     * @brief Get all operations configurations
     * @return Reference to the operations vector
     */
    const std::vector<PostOpConfig>& getOperations() const
    {
        return m_operations;
    }

  private:
    /**
     * @brief Generate all possible combinations based on cartesian mode
     */
    void generateCombinations();
};

} // namespace dlp::testing
