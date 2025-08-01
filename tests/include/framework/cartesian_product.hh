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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
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

#include "framework/iterator.hh"
#include "framework/product.hh"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace dlp::testing::framework {

/**
 * @class CartesianProduct
 * @brief Generates cartesian product combinations from multiple type-erased
 * iterators
 *
 * The CartesianProduct class takes a collection of TypeErasedIterators and
 * generates all possible combinations of their values. This is useful for
 * comprehensive testing where you want to test all combinations of different
 * parameter values.
 *
 * @note This class implements a lazy evaluation strategy - combinations are
 * generated on-demand rather than pre-computed, making it memory efficient even
 * for large parameter spaces.
 *
 * Example usage:
 * @code
 * std::vector<TypeErasedIterator> iterators;
 * // ... populate iterators with different parameter values
 * CartesianProduct product(std::move(iterators));
 *
 * while (product.has_next()) {
 *     auto combination = product.next();
 *     // Process the current combination of parameter values
 * }
 * @endcode
 */
class CartesianProduct
{
  public:
    /**
     * @brief Construct a CartesianProduct from a collection of iterators
     *
     * @param iterators Vector of TypeErasedIterators to generate combinations
     * from
     *
     * Each iterator represents one parameter dimension. The cartesian product
     * will generate all possible combinations across all dimensions.
     */
    CartesianProduct(std::vector<TypeErasedIterator> iterators)
        : m_iterators(std::move(iterators))
        , m_finished(false)
    {
    }

    /**
     * @brief Check if there are more combinations available
     *
     * @return true if more combinations can be generated, false otherwise
     */
    bool has_next() const { return !m_finished; }

    /**
     * @brief Get the next combination of parameter values
     *
     * @return std::vector<std::any> Vector containing one value from each
     * iterator
     * @throws std::runtime_error if called when has_next() returns false
     *
     * The returned vector contains values in the same order as the iterators
     * were provided to the constructor. Each std::any contains a value from
     * the corresponding iterator.
     */
    std::vector<std::any> next()
    {
        if (m_finished) {
            throw std::runtime_error("End of cartesian product reached");
        }

        // Get current values
        std::vector<std::any> result;
        for (const auto& iterator : m_iterators)
            result.push_back(iterator.dereference());

        // Try to advance to next combination
        advance_to_next();

        return result;
    }

    /**
     * @brief Calculate the total number of combinations
     *
     * @return size_t Total number of combinations that will be generated
     *
     * The size is calculated as the product of the sizes of all individual
     * iterators. For example, if you have 3 iterators with sizes 2, 3, and 4,
     * the total combinations will be 2 × 3 × 4 = 24.
     */
    size_t size() const
    {
        // Multiply the size of every iterator
        size_t size = 1;
        for (const auto& iterator : m_iterators) {
            size *= iterator.size();
        }
        return size;
    }

    /**
     * @brief Check if all combinations have been exhausted
     *
     * @return true if no more combinations are available, false otherwise
     */
    bool empty() const { return m_finished; }

  protected:
    std::vector<TypeErasedIterator>
        m_iterators; ///< Collection of iterators for each parameter dimension
    bool
        m_finished; ///< Flag indicating if all combinations have been generated

  private:
    /**
     * @brief Advance to the next combination using odometer-style counting
     *
     * This method implements an odometer-style algorithm, incrementing
     * iterators from right to left (least significant to most significant).
     * When an iterator reaches its end, it's reset and the next iterator is
     * incremented.
     *
     * When all iterators have been exhausted, m_finished is set to true.
     */
    void advance_to_next()
    {
        // Try to increment iterators from right to left
        for (int i = static_cast<int>(m_iterators.size()) - 1; i >= 0; i--) {
            if (m_iterators[i].has_next()) {
                m_iterators[i].increment();
                return; // Successfully advanced
            } else {
                // Reset this iterator and try the next one
                m_iterators[i].reset();
                // Continue to next iterator (left)
            }
        }
        // If we get here, all iterators have been reset and none could advance
        m_finished = true;
    }
};

} // namespace dlp::testing::framework
