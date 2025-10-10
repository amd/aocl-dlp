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
    return m_current_index < m_combinations.size();
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

void
PostOpsIterator::generateCombinations()
{
    m_combinations.clear();

    if (m_operations.empty()) {
        return;
    }

    size_t n = m_operations.size();

    if (!m_cartesian_enabled) {
        // Simple mode: apply all operations in the specified order
        // Generate one combination with all operations in sequence [0, 1, 2,
        // ..., n-1]
        std::vector<size_t> sequence;
        for (size_t i = 0; i < n; i++) {
            sequence.push_back(i);
        }
        m_combinations.push_back(sequence);
        return;
    }

    // Cartesian mode: generate all possible subsets and their permutations
    // This includes:
    // - Empty combination: 1
    // - Single operations: n
    // - Pairs: C(n,2) * 2! permutations
    // - Triples: C(n,3) * 3! permutations
    // - etc.

    // Generate all possible subsets (power set)
    for (size_t subset_size = 0; subset_size <= n; subset_size++) {
        // Generate all combinations of subset_size elements
        std::vector<bool> selector(n, false);
        std::fill(selector.end() - subset_size, selector.end(), true);

        do {
            // Create subset based on selector
            std::vector<size_t> subset;
            for (size_t i = 0; i < n; i++) {
                if (selector[i]) {
                    subset.push_back(i);
                }
            }

            if (subset.empty()) {
                // Empty combination - add it once
                // Use emplace_back to avoid GCC false positive warning
                m_combinations.emplace_back();
            } else {
                // Generate all permutations of this subset
                std::sort(subset.begin(), subset.end());
                do {
                    m_combinations.push_back(subset);
                } while (std::next_permutation(subset.begin(), subset.end()));
            }
        } while (std::next_permutation(selector.begin(), selector.end()));
    }

// Debug output for development
#ifdef DEBUG_POSTOPS_COMBINATIONS
    std::cout << "Generated " << m_combinations.size()
              << " PostOps combinations:" << std::endl;
    for (size_t i = 0; i < m_combinations.size(); i++) {
        std::cout << "  " << i << ": [";
        for (size_t j = 0; j < m_combinations[i].size(); j++) {
            if (j > 0)
                std::cout << ", ";
            std::cout << m_combinations[i][j];
        }
        std::cout << "]" << std::endl;
    }
#endif
}

} // namespace dlp::testing::utils
