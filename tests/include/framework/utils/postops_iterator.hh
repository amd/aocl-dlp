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

namespace dlp::testing::utils {

/**
 * @class PostOpsIterator
 * @brief Iterator for generating combinations of PostOps configurations
 *
 * This class generates test combinations based on two levels of expansion:
 * 1. **Operation Combination** (controlled by cartesian_enabled)
 * 2. **Parameter Expansion** (always applied via cartesian product)
 *
 * @section operation_modes Operation Combination Modes
 *
 * @subsection simple_mode Simple Mode (cartesian_enabled = false)
 * Generates a single combination with all operations applied in the specified
 * order.
 * - Input: [Scale, Bias]
 * - Output: 1 operation combination: [Scale, Bias]
 * - Use Case: Test a specific PostOps pipeline with parameter variations
 *
 * @subsection cartesian_mode Cartesian Mode (cartesian_enabled = true)
 * Generates all possible subsets and permutations of operations, including:
 * - Empty set (no PostOps)
 * - Each operation individually
 * - All combinations of operations in all possible orders
 *
 * - Input: [Scale, Bias]
 * - Output: 5 operation combinations:
 *   1. [] (empty)
 *   2. [Scale] alone
 *   3. [Bias] alone
 *   4. [Scale, Bias] (Scale first)
 *   5. [Bias, Scale] (Bias first)
 * - Use Case: Comprehensive testing of each operation independently and in all
 * sequences
 *
 * @section param_expansion Parameter Expansion (Both Modes)
 *
 * After generating operation combinations, each combination is expanded by the
 * cartesian product of all parameter values within each operation.
 *
 * Example:
 * @code
 *   Scale:
 *     scale_factor_len: [n, 1]      // 2 values
 *     scale_factor_type: [f32, bf16] // 2 values
 *   // Results in 4 parameter variants: (n,f32), (n,bf16), (1,f32), (1,bf16)
 * @endcode
 *
 * Parameter expansion uses an "odometer" approach where the rightmost parameter
 * varies fastest (similar to nested loops).
 *
 * @section total_combinations Total Combinations Formula
 *
 * @subsection simple_formula Simple Mode
 * Total = (Param combinations for Op1) × (Param combinations for Op2) × ...
 *
 * Example: Scale(2×2) + Bias(2) → 4 × 2 = 8 total combinations
 *
 * @subsection cartesian_formula Cartesian Mode
 * Total = Σ (each operation subset × its parameter combinations)
 *
 * Example: Scale(2 params) + Bias(2 params)
 * - Empty: 1
 * - Scale alone: 2
 * - Bias alone: 2
 * - Scale+Bias: 2×2 = 4
 * - Bias+Scale: 2×2 = 4
 * Total: 1 + 2 + 2 + 4 + 4 = 13 combinations
 *
 * @note The iterator maintains synchronization with GEMM parameter iterators,
 *       resetting PostOps for each GEMM configuration.
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
        std::string type; ///< Operation type (e.g., "Elementwise-PRELU",
                          ///< "Bias", "Scale")
        std::map<std::string, std::vector<std::any>>
            params; ///< Parameters with their possible values

        /**
         * @brief Get list of all parameter names
         * @return Vector of parameter names that have values
         */
        std::vector<std::string> getParameterNames() const
        {
            std::vector<std::string> names;
            for (const auto& [name, values] : params) {
                if (!values.empty()) {
                    names.push_back(name);
                }
            }
            return names;
        }

        /**
         * @brief Calculate number of parameter combinations
         * @return Total combinations (cartesian product of all param values)
         */
        size_t getParameterCombinationCount() const
        {
            if (params.empty())
                return 1;

            size_t count = 1;
            for (const auto& [name, values] : params) {
                if (!values.empty()) {
                    count *= values.size();
                }
            }
            return count;
        }
    };

  private:
    std::vector<PostOpConfig>
           m_operations;        ///< All PostOps operations from YAML
    bool   m_cartesian_enabled; ///< Whether to generate all combinations
    size_t m_current_index;     ///< Current combination index
    std::vector<std::vector<size_t>>
        m_combinations; ///< All possible combinations (indices into
                        ///< m_operations)

    /**
     * @brief Store parameter indices for each parameter name
     *
     * Structure: m_param_variants[combo_idx][op_idx] = {param_name →
     * array_index} This allows correct extraction of parameter values for
     * cartesian products.
     *
     * Example: For Scale with scale_factor_len: ["n", "1"] and
     *          scale_factor_type: ["f32", "bf16"], combo 1 might have:
     *          m_param_variants[1][0] = {"scale_factor_len": 0,
     * "scale_factor_type": 1} which means use len[0]="n" and type[1]="bf16"
     */
    std::vector<std::map<size_t, std::map<std::string, size_t>>>
        m_param_variants;

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

    /**
     * @brief Get parameter indices for an operation in current combination
     * @param op_index The operation index to get parameters for
     * @return Map from parameter name to array index for that parameter
     *
     * This returns the specific parameter indices to use for extracting
     * parameter values from the PostOpConfig. For example, if the map
     * contains {"scale_factor_len": 1}, use params["scale_factor_len"][1].
     */
    const std::map<std::string, size_t>& getParameterIndices(
        size_t op_index) const;

  private:
    /**
     * @brief Generate all possible combinations based on cartesian mode
     */
    void generateCombinations();
};

} // namespace dlp::testing::utils
