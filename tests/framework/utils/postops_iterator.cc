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

#include "framework/utils/postops_iterator.hh"

#include <algorithm>
#include <iostream>

namespace dlp::testing::utils {

PostOpsIterator::PostOpsIterator(const std::vector<PostOpConfig>& operations,
                                 bool cartesian_enabled)
    : m_operations(operations)
    , m_cartesian_enabled(cartesian_enabled)
    , m_current_index(0)
{
    generateCombinations();
}

bool
PostOpsIterator::hasNext() const
{
    return (m_current_index + 1) < m_combinations.size();
}

void
PostOpsIterator::next()
{
    if (hasNext()) {
        m_current_index++;
    }
}

void
PostOpsIterator::reset()
{
    m_current_index = 0;
}

size_t
PostOpsIterator::getSize() const
{
    return m_combinations.size();
}

std::vector<size_t>
PostOpsIterator::getCurrentCombination() const
{
    if (m_current_index < m_combinations.size()) {
        return m_combinations[m_current_index];
    }
    return {}; // Empty combination if out of bounds
}

const std::map<std::string, size_t>&
PostOpsIterator::getParameterIndices(size_t op_index) const
{
    static const std::map<std::string, size_t> empty_map;

    // Bounds check
    if (m_current_index >= m_param_variants.size()) {
        return empty_map;
    }

    const auto& variant_map = m_param_variants[m_current_index];
    auto        it          = variant_map.find(op_index);

    if (it != variant_map.end()) {
        return it->second;
    }

    return empty_map;
}

void
PostOpsIterator::generateCombinations()
{
    m_combinations.clear();
    m_param_variants.clear();

    if (m_operations.empty()) {
        return;
    }

    size_t n = m_operations.size();

    // Generate operation combinations into temporary storage
    std::vector<std::vector<size_t>> temp_op_combinations;

    if (!m_cartesian_enabled) {
        // Simple mode: apply all operations in the specified order
        std::vector<size_t> sequence;
        for (size_t i = 0; i < n; i++) {
            sequence.push_back(i);
        }
        temp_op_combinations.push_back(sequence);
    } else {
        // Cartesian mode: generate all possible subsets and their permutations
        for (size_t subset_size = 0; subset_size <= n; subset_size++) {
            std::vector<bool> selector(n, false);
            std::fill(selector.end() - subset_size, selector.end(), true);

            do {
                std::vector<size_t> subset;
                for (size_t i = 0; i < n; i++) {
                    if (selector[i]) {
                        subset.push_back(i);
                    }
                }

                if (subset.empty()) {
                    temp_op_combinations.emplace_back();
                } else {
                    std::sort(subset.begin(), subset.end());
                    do {
                        temp_op_combinations.push_back(subset);
                    } while (
                        std::next_permutation(subset.begin(), subset.end()));
                }
            } while (std::next_permutation(selector.begin(), selector.end()));
        }
    }

    // Expand each operation combination with parameter variants
    for (const auto& op_combo : temp_op_combinations) {
        if (op_combo.empty()) {
            // Empty combination (no operations)
            m_combinations.emplace_back();
            m_param_variants.emplace_back();
            continue;
        }

        // Build parameter metadata for each operation in this combination
        std::vector<size_t>                   op_indices = op_combo;
        std::vector<std::vector<std::string>> param_names_per_op;
        std::vector<std::vector<size_t>>      param_sizes_per_op;

        for (size_t op_idx : op_indices) {
            const auto& op_config   = m_operations[op_idx];
            auto        param_names = op_config.getParameterNames();

            std::vector<size_t> sizes;
            for (const auto& name : param_names) {
                sizes.push_back(op_config.params.at(name).size());
            }

            param_names_per_op.push_back(param_names);
            param_sizes_per_op.push_back(sizes);
        }

        // Calculate total parameter combinations for each operation
        std::vector<size_t> param_combos_per_op;
        for (size_t op_idx : op_indices) {
            param_combos_per_op.push_back(
                m_operations[op_idx].getParameterCombinationCount());
        }

        // Check if any operation has parameter variations
        bool has_variations = false;
        for (size_t count : param_combos_per_op) {
            if (count > 1) {
                has_variations = true;
                break;
            }
        }

        if (!has_variations) {
            // Fast path: all operations have single parameter values
            m_combinations.push_back(op_indices);

            std::map<size_t, std::map<std::string, size_t>> variant_map;
            for (size_t i = 0; i < op_indices.size(); ++i) {
                std::map<std::string, size_t> param_map;
                for (const auto& param_name : param_names_per_op[i]) {
                    param_map[param_name] = 0;
                }
                variant_map[op_indices[i]] = param_map;
            }
            m_param_variants.push_back(variant_map);
            continue;
        }

        // Generate all combinations across all operations' parameters
        // Use safe odometer for outer loop (operations)
        std::vector<size_t> op_param_indices(param_combos_per_op.size(), 0);

        while (true) {
            m_combinations.push_back(op_indices);

            // For this operation combination, decode parameter indices
            std::map<size_t, std::map<std::string, size_t>> combo_variant_map;

            for (size_t i = 0; i < op_indices.size(); ++i) {
                size_t op_idx             = op_indices[i];
                size_t op_param_combo_idx = op_param_indices[i];

                // Decode the linear parameter combination index into individual
                // param indices
                std::map<std::string, size_t> param_map;
                size_t                        remainder = op_param_combo_idx;

                const auto& param_names = param_names_per_op[i];
                const auto& param_sizes = param_sizes_per_op[i];

                // Convert linear index to multi-dimensional indices (RIGHT TO
                // LEFT)
                for (int param_idx = static_cast<int>(param_names.size()) - 1;
                     param_idx >= 0; param_idx--) {
                    size_t param_value_idx = remainder % param_sizes[param_idx];
                    param_map[param_names[param_idx]] = param_value_idx;
                    remainder /= param_sizes[param_idx];
                }

                combo_variant_map[op_idx] = param_map;
            }

            m_param_variants.push_back(combo_variant_map);

            // Advance operation-level odometer (SIGNED INT for safety)
            bool done = true;
            for (int op_param_idx =
                     static_cast<int>(op_param_indices.size()) - 1;
                 op_param_idx >= 0; op_param_idx--) {
                op_param_indices[op_param_idx]++;

                if (op_param_indices[op_param_idx]
                    < param_combos_per_op[op_param_idx]) {
                    done = false;
                    break;
                }

                op_param_indices[op_param_idx] = 0;
            }

            if (done) {
                break;
            }
        }
    }

    // Safety assertion
    if (m_combinations.size() != m_param_variants.size()) {
        throw std::runtime_error(
            "PostOpsIterator: combinations and param_variants out of sync");
    }

#ifdef DEBUG_POSTOPS_COMBINATIONS
    std::cout << "Generated " << m_combinations.size()
              << " PostOps combinations (with parameter variants):"
              << std::endl;
    for (size_t i = 0; i < m_combinations.size(); i++) {
        std::cout << "  " << i << ": ops=[";
        for (size_t j = 0; j < m_combinations[i].size(); j++) {
            if (j > 0)
                std::cout << ", ";
            std::cout << m_combinations[i][j];
        }
        std::cout << "] params={";
        for (const auto& [op_idx, param_map] : m_param_variants[i]) {
            std::cout << " op" << op_idx << ":{";
            bool first = true;
            for (const auto& [param_name, idx] : param_map) {
                if (!first)
                    std::cout << ", ";
                std::cout << param_name << ":" << idx;
                first = false;
            }
            std::cout << "}";
        }
        std::cout << " }" << std::endl;
    }
#endif
}

} // namespace dlp::testing::utils
