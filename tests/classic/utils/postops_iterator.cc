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

#include "utils/postops_iterator.hh"

#include <algorithm>
#include <iostream>

namespace dlp::testing {

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

    // Cartesian mode: generate all permutations of the full sequence
    // Start with the natural order [0, 1, 2, ..., n-1]
    std::vector<size_t> sequence;
    for (size_t i = 0; i < n; i++) {
        sequence.push_back(i);
    }

    // Generate all permutations of the full sequence
    std::sort(sequence.begin(),
              sequence.end()); // Ensure we start from the beginning
    do {
        m_combinations.push_back(sequence);
    } while (std::next_permutation(sequence.begin(), sequence.end()));

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

} // namespace dlp::testing
