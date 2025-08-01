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

// Standard Headers
#include <any>
#include <cstddef>
#include <memory>

// Custom Headers
#include "framework/iterator.hh"

namespace dlp::testing::framework {

/**
 * @class IRange
 * @brief Interface for range classes with type erasure support
 *
 * This interface extends IIterable to provide a common base for all range
 * implementations that need to work with the type-erased iterator system.
 * Range classes implementing this interface can be used polymorphically
 * in containers and algorithms that work with type-erased iterators.
 */
class IRange : public IIterable
{
  public:
    // Inherit all methods from IIterable
    // No additional methods needed for basic range functionality
};

/**
 * @class Range
 * @brief Templated range class for generating sequences of values
 *
 * This class generates a sequence of values from a start value to an end value
 * (exclusive) with a specified step size. It provides STL-compatible iterators
 * and supports both positive and negative step sizes for ascending and
 * descending sequences.
 *
 * @tparam T The type of values in the range (must support arithmetic
 * operations)
 *
 * Features:
 * - STL-compatible iterator interface
 * - Support for positive and negative step sizes
 * - Proper bounds checking and size calculation
 * - Empty range detection
 * - Special handling for edge cases (step=0, start==end)
 *
 * Example usage:
 * @code
 * // Create a range from 0 to 10 with step 2: [0, 2, 4, 6, 8]
 * Range<int> range(0, 10, 2);
 *
 * // Iterate using range-based for loop
 * for (auto value : range) {
 *     std::cout << value << " ";
 * }
 *
 * // Iterate using iterators
 * for (auto it = range.begin(); it != range.end(); ++it) {
 *     std::cout << *it << " ";
 * }
 * @endcode
 */
template<typename T>
class Range
{
  public:
    /**
     * @class iterator
     * @brief STL-compatible iterator for Range<T>
     *
     * This iterator provides standard forward iterator functionality for
     * traversing range values. It properly handles bounds checking and
     * supports both positive and negative step sizes.
     */
    class iterator
    {
      public:
        /**
         * @brief Construct an iterator for the range
         *
         * @param value Current value pointed to by the iterator
         * @param step Step size for advancing the iterator
         * @param end End boundary (not included in the range)
         * @param is_end Whether this is an end iterator marker
         */
        iterator(T value, T step, T end, bool is_end = false)
            : m_value(value)
            , m_step(step)
            , m_end(end)
            , m_is_end(is_end)
        {
        }

        /**
         * @brief Dereference operator - returns the current value
         *
         * @return T The current value in the range
         */
        T operator*() const { return m_value; }

        /**
         * @brief Pre-increment operator - advance to next value
         *
         * @return iterator& Reference to the incremented iterator
         */
        iterator& operator++()
        {
            m_value += m_step;
            return *this;
        }

        /**
         * @brief Post-increment operator - advance to next value
         *
         * @param int Dummy parameter to distinguish from pre-increment
         * @return iterator Copy of the iterator before incrementing
         */
        iterator operator++(int)
        {
            iterator temp = *this;
            m_value += m_step;
            return temp;
        }

        /**
         * @brief Equality comparison operator
         *
         * @param other Iterator to compare with
         * @return true If iterators point to the same position or both are at
         * end
         *
         * Properly handles end iterator comparisons and bounds checking
         * for both positive and negative step ranges.
         */
        bool operator==(const iterator& other) const
        {
            // If either is marked as end iterator, check bounds
            if (m_is_end || other.m_is_end) {
                return has_reached_end() && other.has_reached_end();
            }
            return m_value == other.m_value;
        }

        /**
         * @brief Inequality comparison operator
         *
         * @param other Iterator to compare with
         * @return true If iterators point to different positions
         */
        bool operator!=(const iterator& other) const
        {
            return !(*this == other);
        }

      private:
        T    m_value;  ///< Current value pointed to by the iterator
        T    m_step;   ///< Step size for advancing
        T    m_end;    ///< End boundary (exclusive)
        bool m_is_end; ///< Flag indicating this is an end iterator

        /**
         * @brief Check if iterator has reached or passed the end boundary
         *
         * @return true If the iterator has reached or exceeded the end boundary
         *
         * Handles both positive and negative step directions correctly.
         */
        bool has_reached_end() const
        {
            if (m_is_end)
                return true;

            if (m_step > 0) {
                return m_value >= m_end;
            } else if (m_step < 0) {
                return m_value <= m_end;
            }
            return true; // step == 0 case
        }
    };

    /**
     * @brief Construct a range with specified start, end, and step values
     *
     * @param start Starting value of the range (inclusive)
     * @param end Ending value of the range (exclusive)
     * @param step Step size for advancing through the range (default: 1)
     *
     * The range will generate values from start to end (exclusive) with
     * the specified step size. For positive steps, start should be less
     * than end. For negative steps, start should be greater than end.
     */
    Range(T start, T end, T step = T(1))
        : m_start(start)
        , m_end(end)
        , m_step(step)
    {
    }

    /**
     * @brief Get iterator to the beginning of the range
     *
     * @return iterator Iterator pointing to the first value in the range
     */
    iterator begin() const { return iterator(m_start, m_step, m_end, false); }

    /**
     * @brief Get iterator to the end of the range
     *
     * @return iterator End iterator (points past the last valid value)
     */
    iterator end() const { return iterator(m_end, m_step, m_end, true); }

    /**
     * @brief Calculate the number of values in the range
     *
     * @return size_t Number of values that will be generated by the range
     *
     * The calculation handles positive and negative steps correctly:
     * - For positive steps: counts values from start up to (but not including)
     * end
     * - For negative steps: counts values from start down to (but not
     * including) end
     * - Special case: step=0 with start==end returns 1 (single value)
     * - Invalid ranges return 0
     */
    size_t size() const
    {
        if constexpr (!std::is_same<T, bool>::value) {
            if (m_step == T(-1) && m_start == m_end) {
                // Special case: step=-1 with start==end means single value
                return 1;
            } else if ((m_step > 0 && m_start < m_end)
                       || (m_step < 0 && m_start > m_end)) {
                return static_cast<size_t>(
                    (std::abs(m_end - m_start) + std::abs(m_step) - 1)
                    / std::abs(m_step));
            }
        }
        return 0;
    }

    /**
     * @brief Check if the range is empty (contains no values)
     *
     * @return true If the range would generate no values, false otherwise
     *
     * A range is empty if:
     * - Positive step with start >= end
     * - Negative step with start <= end
     * - Step size is zero (except when start == end)
     */
    bool empty() const
    {
        if constexpr (!std::is_same<T, bool>::value) {
            if (m_step == T(-1) && m_start == m_end) {
                // Special case: step=-1 with start==end means single value (not
                // empty)
                return false;
            }
            return (m_step > 0 && m_start >= m_end)
                   || (m_step < 0 && m_start <= m_end) || (m_step == 0);
        }
        return false;
    }

  private:
    T m_start, m_end,
        m_step; ///< Range parameters: start value, end boundary, and step size
};

/**
 * @class TypeErasedRange
 * @brief Type-erased wrapper for Range<T> with polymorphic behavior
 *
 * This class wraps a Range<T> to provide type-erased access through the IRange
 * interface. It's used when polymorphic behavior is needed, such as in
 * containers that hold different range types or in cartesian product
 * generation.
 *
 * @tparam T The type of values in the range
 *
 * The class provides TypeErasedIterator objects that can be used with the
 * cartesian product framework and other type-erased iterator consumers.
 *
 * Example usage:
 * @code
 * // Create a type-erased range
 * TypeErasedRange<int> range(0, 5, 1);
 *
 * // Get type-erased iterators
 * auto begin_iter = range.begin(); // TypeErasedIterator
 * auto end_iter = range.end();     // TypeErasedIterator
 *
 * // Use with cartesian product
 * std::vector<TypeErasedIterator> iterators;
 * iterators.push_back(range.begin());
 * CartesianProduct product(std::move(iterators));
 * @endcode
 */
template<typename T>
class TypeErasedRange : public IRange
{
  public:
    /**
     * @class concrete_iterator
     * @brief Concrete implementation of IIterator for TypeErasedRange<T>
     *
     * This class implements the IIterator interface to provide type-erased
     * iteration over range values. It supports all the extended functionality
     * needed for cartesian product generation, including cloning, resetting,
     * and bounds checking.
     */
    class concrete_iterator : public IIterator
    {
      public:
        /**
         * @brief Construct a concrete iterator for the range
         *
         * @param value Current value pointed to by the iterator
         * @param step Step size for advancing the iterator
         * @param end End boundary (not included in the range)
         * @param is_end Whether this is an end iterator marker
         */
        concrete_iterator(T value, T step, T end, bool is_end = false)
            : m_value(value)
            , m_step(step)
            , m_end(end)
            , m_is_end(is_end)
            , m_start(value) // Store original start value for reset
        {
        }

        /**
         * @brief Create a deep copy of this iterator
         *
         * @return std::unique_ptr<IIterator> A new iterator with the same state
         *
         * Required for TypeErasedIterator copy semantics.
         */
        std::unique_ptr<IIterator> clone() const override
        {
            return std::make_unique<concrete_iterator>(m_value, m_step, m_end,
                                                       m_is_end);
        }

        /**
         * @brief Get the current value as type-erased std::any
         *
         * @return std::any The current value wrapped in std::any
         */
        std::any dereference() const override { return m_value; }

        /**
         * @brief Advance the iterator to the next value
         *
         * Increments the current value by the step size.
         */
        void increment() override { m_value += m_step; }

        /**
         * @brief Compare this iterator with another for equality
         *
         * @param other The iterator to compare with (must be of same concrete
         * type)
         * @return true If both iterators point to equivalent positions
         *
         * Uses dynamic_cast to ensure type safety and handles end iterator
         * comparisons correctly.
         */
        bool equals(const IIterator& other) const override
        {
            const auto* other_concrete =
                dynamic_cast<const concrete_iterator*>(&other);
            if (!other_concrete)
                return false;

            // If either is marked as end iterator, check bounds
            if (m_is_end || other_concrete->m_is_end) {
                return has_reached_end() && other_concrete->has_reached_end();
            }
            return m_value == other_concrete->m_value;
        }

        /**
         * @brief Check if the iterator can advance to the next value
         *
         * @return true If there is a next value within the range bounds
         *
         * Used by cartesian product generation to determine when a dimension
         * has been exhausted.
         */
        bool has_next() const override
        {
            if (m_is_end)
                return false;

            // Special case: step==-1 with start==end means single value, no
            // next
            if constexpr (!std::is_same<T, bool>::value) {
                if (m_step == T(-1) && m_start == m_end) {
                    return false;
                }
            }

            T next_value = m_value + m_step;
            if constexpr (!std::is_same<T, bool>::value) {
                if (m_step > 0) {
                    return next_value < m_end;
                } else if (m_step < 0) {
                    return next_value > m_end;
                }
            }
            return false; // step == 0 case
        }

        /**
         * @brief Reset the iterator to its initial position
         *
         * Returns the iterator to the starting value and clears the end flag.
         * Used in cartesian product generation when resetting a dimension.
         */
        void reset() override
        {
            m_value  = m_start;
            m_is_end = false;
        }

        /**
         * @brief Get the total number of values in the range
         *
         * @return size_t Number of values the iterator can traverse
         *
         * Used for calculating cartesian product sizes and memory allocation.
         */
        size_t size() const override
        {
            if constexpr (!std::is_same<T, bool>::value) {
                if (m_step == T(-1) && m_start == m_end) {
                    // Special case: step=-1 with start==end means single value
                    return 1;
                } else if ((m_step > 0 && m_start < m_end)
                           || (m_step < 0 && m_start > m_end)) {
                    return static_cast<size_t>(
                        (std::abs(m_end - m_start) + std::abs(m_step) - 1)
                        / std::abs(m_step));
                }
            }
            return 0;
        }

      private:
        T    m_value;
        T    m_step;
        T    m_end;
        bool m_is_end;
        T    m_start; // Store original start value for reset

        // Check if iterator has reached or passed the end
        bool has_reached_end() const
        {
            if (m_is_end)
                return true;

            if constexpr (!std::is_same<T, bool>::value) {
                if (m_step > 0) {
                    return m_value >= m_end;
                } else if (m_step < 0) {
                    return m_value <= m_end;
                }
                return true; // step == 0 case
            }
            return true;
        }
    };

    // Constructor
    TypeErasedRange(T start, T end, T step = T(1))
        : m_start(start)
        , m_end(end)
        , m_step(step)
    {
    }

    // Begin iterator - starts at m_start
    TypeErasedIterator begin() const override
    {
        return TypeErasedIterator(
            std::make_unique<concrete_iterator>(m_start, m_step, m_end, false));
    }

    // End iterator - ends at m_end (not included)
    TypeErasedIterator end() const override
    {
        return TypeErasedIterator(
            std::make_unique<concrete_iterator>(m_end, m_step, m_end, true));
    }

    // Size of the range
    size_t size() const override
    {
        if constexpr (!std::is_same<T, bool>::value) {
            if (m_step == T(-1) && m_start == m_end) {
                // Special case: step=-1 with start==end means single value
                return 1;
            } else if ((m_step > 0 && m_start < m_end)
                       || (m_step < 0 && m_start > m_end)) {
                return static_cast<size_t>(
                    (std::abs(m_end - m_start) + std::abs(m_step) - 1)
                    / std::abs(m_step));
            }
        }
        return 0;
    }

    // Check if range is empty
    bool empty() const override
    {
        if constexpr (!std::is_same<T, bool>::value) {
            if (m_step == T(-1) && m_start == m_end) {
                // Special case: step=-1 with start==end means single value (not
                // empty)
                return false;
            }
            return (m_step > 0 && m_start >= m_end)
                   || (m_step < 0 && m_start <= m_end) || (m_step == 0);
        }
        return false;
    }

  private:
    T m_start, m_end, m_step;
};

} // namespace dlp::testing::framework
