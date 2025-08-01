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

#include <any>
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace dlp::testing::framework {

/**
 * @class IIterator
 * @brief Common iterator interface for type erasure.
 *
 * This interface provides a common base for all iterator implementations
 * that need to work with type erasure. It supports both basic iterator
 * operations and extended functionality needed for cartesian product
 * generation.
 *
 * Any class that wants to provide iterators with type erasure should implement
 * this interface. The TypeErasedIterator class wraps implementations of this
 * interface to provide a uniform iterator experience.
 */
class IIterator
{
  public:
    /**
     * @brief Virtual destructor for proper cleanup of derived classes
     */
    virtual ~IIterator() = default;

    /**
     * @brief Create a deep copy of this iterator
     *
     * @return std::unique_ptr<IIterator> A new iterator that is a copy of this
     * one
     *
     * This method is essential for value semantics in TypeErasedIterator.
     * The returned iterator should be positioned at the same element and
     * have the same state as the original.
     */
    virtual std::unique_ptr<IIterator> clone() const = 0;

    /**
     * @brief Get the current value pointed to by the iterator
     *
     * @return std::any The current value, type-erased as std::any
     *
     * This method should return the value that the iterator is currently
     * pointing to, wrapped in std::any for type erasure.
     */
    virtual std::any dereference() const = 0;

    /**
     * @brief Move the iterator to the next position
     *
     * Advances the iterator to point to the next element in the sequence.
     * The behavior when called on an iterator that has reached the end
     * depends on the specific implementation.
     */
    virtual void increment() = 0;

    /**
     * @brief Compare this iterator with another for equality
     *
     * @param other The iterator to compare with
     * @return true if the iterators are equivalent, false otherwise
     *
     * Two iterators are considered equal if they point to the same position
     * in the same sequence. This is used for implementing TypeErasedIterator
     * comparison operators.
     */
    virtual bool equals(const IIterator& other) const = 0;

    /**
     * @brief Check if the iterator can advance to the next element
     *
     * @return true if increment() can be called, false if at the end
     *
     * This method is used for cartesian product support to determine
     * when an iterator dimension has been exhausted.
     */
    virtual bool has_next() const = 0;

    /**
     * @brief Reset the iterator to its initial position
     *
     * Returns the iterator to the first element in its sequence.
     * This is used in cartesian product generation when a dimension
     * needs to be reset while advancing other dimensions.
     */
    virtual void reset() = 0;

    /**
     * @brief Get the total number of elements in the iterator's sequence
     *
     * @return size_t The number of elements this iterator can traverse
     *
     * This is used for calculating the total size of cartesian products
     * and for memory allocation decisions.
     */
    virtual size_t size() const = 0;
};

/**
 * @class TypeErasedIterator
 * @brief Type-erased iterator wrapper that works with any IIterator
 * implementation
 *
 * This class provides a common iterator interface for all iterator types by
 * wrapping implementations of the IIterator interface. It handles memory
 * management, copy semantics, and provides both STL-style iterator operations
 * and extended functionality for cartesian product support.
 *
 * The class implements value semantics, meaning TypeErasedIterator objects can
 * be copied, assigned, and stored in containers safely. The underlying
 * IIterator implementation is properly cloned when needed.
 *
 * Example usage:
 * @code
 * // Create from an IIterator implementation
 * auto iter = TypeErasedIterator(std::make_unique<SomeIteratorImpl>());
 *
 * // Use STL-style operations
 * auto value = *iter;
 * ++iter;
 *
 * // Use cartesian product operations
 * if (iter.has_next()) {
 *     iter.increment();
 * }
 * @endcode
 */
class TypeErasedIterator
{
  public:
    /**
     * @brief Default constructor - creates an empty iterator
     *
     * Creates an iterator with no underlying implementation. This iterator
     * will throw exceptions if used for operations other than assignment
     * or destruction. Useful for default initialization of containers.
     */
    TypeErasedIterator()
        : m_impl(nullptr)
    {
    }

    /**
     * @brief Construct from an IIterator implementation
     *
     * @param impl Unique pointer to an IIterator implementation
     *
     * Takes ownership of the provided IIterator implementation and wraps
     * it for type-erased access.
     */
    TypeErasedIterator(std::unique_ptr<IIterator> impl)
        : m_impl(std::move(impl))
    {
    }

    /**
     * @brief Copy constructor - creates a deep copy
     *
     * @param other The TypeErasedIterator to copy from
     *
     * Creates a new TypeErasedIterator that is a deep copy of the source.
     * If the source has a valid implementation, it will be cloned.
     */
    TypeErasedIterator(const TypeErasedIterator& other)
        : m_impl(other.m_impl ? other.m_impl->clone() : nullptr)
    {
    }

    /**
     * @brief Copy assignment operator
     *
     * @param other The TypeErasedIterator to copy from
     * @return TypeErasedIterator& Reference to this iterator
     *
     * Assigns this iterator to be a deep copy of the source iterator.
     * Handles self-assignment correctly.
     */
    TypeErasedIterator& operator=(const TypeErasedIterator& other)
    {
        if (this != &other) {
            m_impl = other.m_impl ? other.m_impl->clone() : nullptr;
        }
        return *this;
    }

    /**
     * @brief Move constructor
     *
     * @param other The TypeErasedIterator to move from
     *
     * Transfers ownership of the underlying implementation from the source
     * iterator to this iterator. The source iterator becomes empty.
     */
    TypeErasedIterator(TypeErasedIterator&& other) noexcept = default;

    /**
     * @brief Move assignment operator
     *
     * @param other The TypeErasedIterator to move from
     * @return TypeErasedIterator& Reference to this iterator
     *
     * Transfers ownership of the underlying implementation from the source
     * iterator to this iterator. The source iterator becomes empty.
     */
    TypeErasedIterator& operator=(TypeErasedIterator&& other) noexcept =
        default;

    /**
     * @brief Dereference operator - returns the current value
     *
     * @return std::any The current value pointed to by the iterator
     * @throws std::runtime_error if the iterator is empty (default-constructed)
     *
     * STL-style dereference operation that returns the current value
     * as a type-erased std::any object.
     */
    std::any operator*() const
    {
        if (!m_impl)
            throw std::runtime_error("Cannot dereference empty iterator");
        return m_impl->dereference();
    }

    /**
     * @brief Pre-increment operator - advance to next position
     *
     * @return TypeErasedIterator& Reference to this iterator after incrementing
     * @throws std::runtime_error if the iterator is empty
     *
     * STL-style pre-increment operation that advances the iterator to the
     * next position and returns a reference to the modified iterator.
     */
    TypeErasedIterator& operator++()
    {
        if (!m_impl)
            throw std::runtime_error("Cannot increment empty iterator");
        m_impl->increment();
        return *this;
    }

    /**
     * @brief Post-increment operator - advance to next position
     *
     * @param int Dummy parameter to distinguish from pre-increment
     * @return TypeErasedIterator Copy of the iterator before incrementing
     * @throws std::runtime_error if the iterator is empty
     *
     * STL-style post-increment operation that advances the iterator to the
     * next position and returns a copy of the iterator before modification.
     */
    TypeErasedIterator operator++(int)
    {
        if (!m_impl)
            throw std::runtime_error("Cannot increment empty iterator");
        TypeErasedIterator temp = *this;
        m_impl->increment();
        return temp;
    }

    /**
     * @brief Equality comparison operator
     *
     * @param other The iterator to compare with
     * @return true if the iterators are equal, false otherwise
     *
     * Two TypeErasedIterators are equal if:
     * - Both are empty (no implementation), or
     * - Both have implementations and those implementations are equal
     */
    bool operator==(const TypeErasedIterator& other) const
    {
        if (!m_impl && !other.m_impl)
            return true;
        if (!m_impl || !other.m_impl)
            return false;
        return m_impl->equals(*other.m_impl);
    }

    /**
     * @brief Inequality comparison operator
     *
     * @param other The iterator to compare with
     * @return true if the iterators are not equal, false otherwise
     */
    bool operator!=(const TypeErasedIterator& other) const
    {
        return !(*this == other);
    }

    /**
     * @brief Check if the iterator can advance further
     *
     * @return true if more elements are available, false if at the end
     *
     * Extended operation for cartesian product support. Returns false
     * for empty iterators.
     */
    bool has_next() const
    {
        if (!m_impl)
            return false;
        return m_impl->has_next();
    }

    /**
     * @brief Reset the iterator to its initial position
     *
     * Extended operation for cartesian product support. Resets the iterator
     * to point to the first element in its sequence. No-op for empty iterators.
     */
    void reset()
    {
        if (m_impl)
            m_impl->reset();
    }

    /**
     * @brief Get the total number of elements in the sequence
     *
     * @return size_t Number of elements, or 0 for empty iterators
     *
     * Extended operation for cartesian product support. Returns the total
     * number of elements this iterator can traverse.
     */
    size_t size() const
    {
        if (!m_impl)
            return 0;
        return m_impl->size();
    }

    /**
     * @brief Get the current value (alternative to operator*)
     *
     * @return std::any The current value pointed to by the iterator
     * @throws std::runtime_error if the iterator is empty
     *
     * Extended operation that provides the same functionality as operator*
     * but with a more explicit name for cartesian product usage.
     */
    std::any dereference() const
    {
        if (!m_impl)
            throw std::runtime_error("Cannot dereference empty iterator");
        return m_impl->dereference();
    }

    /**
     * @brief Advance to the next position (alternative to operator++)
     *
     * @throws std::runtime_error if the iterator is empty
     *
     * Extended operation that provides the same functionality as operator++
     * but with a more explicit name for cartesian product usage.
     */
    void increment()
    {
        if (!m_impl)
            throw std::runtime_error("Cannot increment empty iterator");
        m_impl->increment();
    }

  private:
    std::unique_ptr<IIterator>
        m_impl; ///< Pointer to the underlying iterator implementation
};

/**
 * @class IIterable
 * @brief Interface for any iterable container using type erasure
 *
 * This interface can be implemented by ranges, vectors, or any other iterable
 * class that needs to work with the type-erased iterator system. It provides
 * basic container operations and iterator access.
 *
 * Implementations should ensure that begin() and end() return compatible
 * TypeErasedIterator objects that can be used together for iteration.
 */
class IIterable
{
  public:
    /**
     * @brief Virtual destructor for proper cleanup of derived classes
     */
    virtual ~IIterable() = default;

    /**
     * @brief Get the number of elements in the container
     *
     * @return size_t Number of elements
     */
    virtual size_t size() const = 0;

    /**
     * @brief Check if the container is empty
     *
     * @return true if the container has no elements, false otherwise
     */
    virtual bool empty() const = 0;

    /**
     * @brief Get an iterator to the beginning of the container
     *
     * @return TypeErasedIterator Iterator pointing to the first element
     */
    virtual TypeErasedIterator begin() const = 0;

    /**
     * @brief Get an iterator to the end of the container
     *
     * @return TypeErasedIterator Iterator pointing past the last element
     */
    virtual TypeErasedIterator end() const = 0;
};

} // namespace dlp::testing::framework
