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
#include <stdexcept>
#include <vector>

namespace dlp::testing::framework {

/**
 * @class VectorIterable
 * @brief Type-erased iterable wrapper for std::vector containers
 *
 * This class provides a type-erased interface for iterating over std::vector
 * containers. It implements the IIterable interface to work seamlessly with
 * the cartesian product framework and other type-erased iterator consumers.
 *
 * @tparam T The type of elements stored in the vector
 *
 * Key features:
 * - Memory safe: stores a copy of the vector to avoid dangling references
 * - Full cartesian product support: reset, has_next, size operations
 * - Type-erased access through TypeErasedIterator
 * - Exception safety: proper bounds checking
 *
 * Example usage:
 * @code
 * std::vector<int> values = {1, 2, 3, 4, 5};
 * VectorIterable<int> iterable(values);
 *
 * // Iterate using type-erased iterators
 * auto begin_iter = iterable.begin(); // TypeErasedIterator
 * auto end_iter = iterable.end();     // TypeErasedIterator
 *
 * while (begin_iter != end_iter) {
 *     auto value = std::any_cast<int>(*begin_iter);
 *     std::cout << value << " ";
 *     ++begin_iter;
 * }
 *
 * // Use with cartesian product
 * std::vector<TypeErasedIterator> iterators;
 * iterators.push_back(iterable.begin());
 * CartesianProduct product(std::move(iterators));
 * @endcode
 */
template<typename T>
class VectorIterable : public IIterable
{
  public:
    /**
     * @class vector_iterator
     * @brief Concrete iterator implementation for vector elements
     *
     * This class implements the IIterator interface to provide type-erased
     * iteration over vector elements. It stores a copy of the vector for
     * memory safety and supports all extended operations needed for cartesian
     * product generation.
     *
     * The iterator uses index-based access rather than storing STL iterators
     * to avoid memory corruption issues when the original vector goes out of
     * scope.
     */
    class vector_iterator : public IIterator
    {
      public:
        /**
         * @brief Construct a vector iterator at the specified position
         *
         * @param vec Vector to iterate over (will be copied for safety)
         * @param index Starting index position (default: 0 for begin iterator)
         *
         * The vector is copied rather than referenced to ensure memory safety
         * and avoid dangling references when the original vector is destroyed.
         */
        vector_iterator(const std::vector<T>& vec, size_t index = 0)
            : m_vec(vec) // Store a copy of the vector
            , m_index(index)
        {
        }

        /**
         * @brief Create a deep copy of this iterator
         *
         * @return std::unique_ptr<IIterator> A new iterator with the same state
         *
         * Required for TypeErasedIterator copy semantics. The returned iterator
         * will have its own copy of the vector and be positioned at the same
         * index.
         */
        std::unique_ptr<IIterator> clone() const override
        {
            return std::make_unique<vector_iterator>(m_vec, m_index);
        }

        /**
         * @brief Get the current element as type-erased std::any
         *
         * @return std::any The current element wrapped in std::any
         * @throws std::out_of_range if iterator is positioned beyond vector
         * bounds
         *
         * Returns the element at the current index position, type-erased as
         * std::any for compatibility with the type-erased iterator framework.
         */
        std::any dereference() const override
        {
            if (m_index >= m_vec.size()) {
                throw std::out_of_range("vector_iterator index out of bounds");
            }
            return m_vec[m_index];
        }

        /**
         * @brief Advance the iterator to the next element
         *
         * Increments the internal index to point to the next element in the
         * vector. No bounds checking is performed - use has_next() to check
         * before calling.
         */
        void increment() override { ++m_index; }

        /**
         * @brief Compare this iterator with another for equality
         *
         * @param other The iterator to compare with (must be of same concrete
         * type)
         * @return true If both iterators are at the same index position
         *
         * Uses dynamic_cast to ensure type safety. Only compares index
         * positions, assuming both iterators reference the same logical vector.
         */
        bool equals(const IIterator& other) const override
        {
            const auto* other_vector =
                dynamic_cast<const vector_iterator*>(&other);
            return other_vector && m_index == other_vector->m_index;
        }

        /**
         * @brief Check if the iterator can advance to the next element
         *
         * @return true If there is a next element available, false if at the
         * end
         *
         * Used by cartesian product generation to determine when this dimension
         * has been exhausted and needs to be reset.
         */
        bool has_next() const override { return (m_index + 1) < m_vec.size(); }

        /**
         * @brief Reset the iterator to the first element
         *
         * Returns the iterator to index 0 (beginning of the vector).
         * Used in cartesian product generation when resetting a dimension.
         */
        void reset() override { m_index = 0; }

        /**
         * @brief Get the total number of elements in the vector
         *
         * @return size_t Number of elements the iterator can traverse
         *
         * Used for calculating cartesian product sizes and memory allocation
         * decisions.
         */
        size_t size() const override { return m_vec.size(); }

      private:
        std::vector<T>
               m_vec; ///< Copy of the vector being iterated (for memory safety)
        size_t m_index; ///< Current index position in the vector
    };

    /**
     * @brief Construct a VectorIterable from a std::vector
     *
     * @param vec Vector to wrap (will be copied for memory safety)
     *
     * The provided vector is copied rather than referenced to ensure memory
     * safety. This allows the VectorIterable to outlive the original vector
     * without issues.
     */
    VectorIterable(const std::vector<T>& vec)
        : m_vec(vec) // This now copies the vector
    {
    }

    /**
     * @brief Get a type-erased iterator to the beginning of the vector
     *
     * @return TypeErasedIterator Iterator pointing to the first element
     *
     * Returns a TypeErasedIterator that wraps a vector_iterator positioned
     * at index 0. The returned iterator can be used with cartesian product
     * generation and other type-erased iterator consumers.
     */
    TypeErasedIterator begin() const override
    {
        return TypeErasedIterator(std::make_unique<vector_iterator>(m_vec, 0));
    }

    /**
     * @brief Get a type-erased iterator to the end of the vector
     *
     * @return TypeErasedIterator End iterator (points past the last element)
     *
     * Returns a TypeErasedIterator positioned at the end of the vector
     * (index == vector.size()). This iterator should not be dereferenced.
     */
    TypeErasedIterator end() const override
    {
        return TypeErasedIterator(
            std::make_unique<vector_iterator>(m_vec, m_vec.size()));
    }

    /**
     * @brief Get the number of elements in the vector
     *
     * @return size_t Number of elements
     */
    size_t size() const override { return m_vec.size(); }

    /**
     * @brief Check if the vector is empty
     *
     * @return true If the vector contains no elements, false otherwise
     */
    bool empty() const override { return m_vec.empty(); }

  private:
    std::vector<T>
        m_vec; ///< Copy of the vector (stored by value for memory safety)
};

} // namespace dlp::testing::framework
